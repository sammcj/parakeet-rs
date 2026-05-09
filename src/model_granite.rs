//! ONNX session wrapper for the autoregressive Granite Speech 4.1 family
//! (`granite-speech-4.1-2b` base and `granite-speech-4.1-2b-plus`).
//!
//! Both variants share the same graph topology, so this module backs both
//! of them. The differences are entirely on the prompt and post-decode
//! side and live in [`granite`](crate::granite) and
//! [`granite_plus`](crate::granite_plus).
//!
//! ## Bundle layout
//!
//! Bundles are produced by the export pipeline at
//! [`sammcj/granite-speech-4.1-onnx`](https://github.com/sammcj/granite-speech-4.1-onnx)
//! and published per-variant on the Hugging Face Hub. The directory
//! layout this loader expects:
//!
//! ```text
//! <bundle_dir>/
//!   {fp32, fp16w, int8}/
//!     encoder.onnx        + encoder.onnx_data
//!     prompt_encode.onnx  + prompt_encode.onnx_data
//!     decode_step.onnx    + decode_step.onnx_data
//!     embed_tokens.onnx   + embed_tokens.onnx_data
//!   tokenizer.json
//!   chat_template.jinja                (base / plus)
//!   processor_config.json              (base / plus)
//!   granite_export_metadata.json
//! ```
//!
//! `embed_tokens.onnx` is the LLM's input-embedding table exported as a
//! single Gather op (`input_ids [B,N] -> inputs_embeds [B,N,2048]`). It
//! ships in every bundle so consumers don't need a separate weight format.
//! The base/plus LLM graphs deliberately consume `inputs_embeds` rather
//! than `input_ids` because that's the cleanest way to splice the
//! projector-output audio embeddings into the rendered chat-template
//! sequence.
//!
//! ## Graph IO contract
//!
//! All graphs target opset 20 / IR 10 / `ai.onnx`-only and load under
//! `ort` 2.0-rc.x. KV cache uses the standard `causal-lm-with-past`
//! split-graph pattern: `prompt_encode` produces the initial cache,
//! `decode_step` consumes one token plus the running cache and returns a
//! grown cache. The Granite 4.0 1B LLM is decoder-only with grouped-query
//! attention (4 KV heads), so the cache shape is
//! `[B, NUM_KV_HEADS, T, HEAD_DIM] = [B, 4, T, 128]` per layer per
//! key/value, across 40 layers (160 inputs / 80 outputs total per
//! `decode_step` call).
//!
//! See [`granite_export_metadata.json`](https://github.com/sammcj/granite-speech-4.1-onnx)
//! in any bundle for the authoritative IO names and shapes.
//!
//! ## Acceleration
//!
//! Pass an [`ExecutionConfig`] with the GPU/Metal execution provider of
//! choice (CUDA, CoreML, DirectML, etc.). The bundles' FP16w tier
//! (`fp16w/`) is the recommended precision for accelerated inference: it
//! stores weights as FP16 but compute and IO stay FP32, matching what the
//! crate's existing GPU-targeted models expect. Some opset-20 ops may
//! lack kernels on a given EP and fall back to CPU silently at session
//! load; that's an `ort` runtime detail, not a bundle issue.

use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use ndarray::{Array1, Array2, Array3, Array4};
use ort::session::{Session, SessionInputValue};
use std::borrow::Cow;
use std::path::{Path, PathBuf};

/// Granite 4.0 1B LLM topology - 40 transformer blocks.
pub(crate) const NUM_LLM_LAYERS: usize = 40;
/// Grouped-query attention KV head count (the LLM has 32 query heads
/// over 4 KV heads, hence the head ratio 8 GQA used by Granite 4.0).
pub(crate) const NUM_KV_HEADS: usize = 4;
/// Per-head dimensionality of the KV cache tensors.
pub(crate) const HEAD_DIM: usize = 128;
/// Hidden state size across encoder projector output, prompt
/// embeddings, decode-step embeddings, and KV head dim * 16. This is
/// what the LLM and projector both expect. Documented as a named
/// constant; tests reference it via [`HIDDEN_SIZE`].
#[allow(dead_code)]
pub(crate) const HIDDEN_SIZE: usize = 2048;

/// Precision tier as published in each bundle. Defaults to `Fp16w`
/// because that's the recommended quality/size trade-off; FP32 matches
/// the original PyTorch checkpoint within numeric tolerance and INT8 is
/// the smallest tier with a mild capitalisation-and-punctuation drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GranitePrecision {
    #[default]
    Fp16w,
    Fp32,
    Int8,
}

impl GranitePrecision {
    pub(crate) fn dirname(self) -> &'static str {
        match self {
            GranitePrecision::Fp16w => "fp16w",
            GranitePrecision::Fp32 => "fp32",
            GranitePrecision::Int8 => "int8",
        }
    }
}

/// One layer's pair of `[1, NUM_KV_HEADS, T, HEAD_DIM]` cache tensors.
pub(crate) type LayerKv = (Array4<f32>, Array4<f32>);

/// 40-layer KV cache. Each entry is the (key, value) pair for one
/// transformer block. On the first prompt-encode call all entries start
/// empty (T = 0); on subsequent decode-step calls the model writes the
/// new (T_total) cache back into each entry.
pub(crate) struct KvCache {
    pub(crate) layers: Vec<LayerKv>,
}

impl KvCache {
    /// Build a freshly-zeroed cache with `T = 0` per layer. Not used
    /// in the production path (the LLM always sees a populated cache
    /// from `prompt_encode`), but exposed for tests and as a hint at
    /// the per-layer cache shape.
    #[allow(dead_code)]
    pub(crate) fn empty() -> Self {
        let zero = Array4::<f32>::zeros((1, NUM_KV_HEADS, 0, HEAD_DIM));
        let layers: Vec<LayerKv> = (0..NUM_LLM_LAYERS)
            .map(|_| (zero.clone(), zero.clone()))
            .collect();
        Self { layers }
    }

    pub(crate) fn past_len(&self) -> usize {
        self.layers[0].0.shape()[2]
    }
}

/// Result of running the encoder graph. `audio_embeds` is
/// `[1, T_audio, HIDDEN_SIZE]`; `audio_embed_sizes` is `[1]` containing
/// the number of valid frames so the caller can size the `<|audio|>`
/// token run in the chat-template prompt.
pub(crate) struct EncoderOutput {
    pub(crate) audio_embeds: Array3<f32>,
    pub(crate) audio_embed_sizes: Array1<i64>,
}

/// Holds the four ort sessions: encoder, prompt_encode, decode_step,
/// embed_tokens. Construction is one-shot via [`from_pretrained`].
pub(crate) struct GraniteArModel {
    encoder: Session,
    prompt_encode: Session,
    decode_step: Session,
    embed_tokens: Session,
}

impl GraniteArModel {
    pub(crate) fn from_pretrained<P: AsRef<Path>>(
        bundle_dir: P,
        precision: GranitePrecision,
        exec_config: ExecutionConfig,
    ) -> Result<Self> {
        let bundle_dir = bundle_dir.as_ref();
        let prec_dir = bundle_dir.join(precision.dirname());
        if !prec_dir.is_dir() {
            return Err(Error::Config(format!(
                "Granite Speech bundle is missing the '{}' precision directory at {}. Available bundles ship fp32/, fp16w/, and int8/ subdirs - download the variant you want or pass a different GranitePrecision.",
                precision.dirname(),
                bundle_dir.display()
            )));
        }

        let encoder_path = require_file(&prec_dir, "encoder.onnx")?;
        let prompt_encode_path = require_file(&prec_dir, "prompt_encode.onnx")?;
        let decode_step_path = require_file(&prec_dir, "decode_step.onnx")?;
        let embed_tokens_path = require_file(&prec_dir, "embed_tokens.onnx").map_err(|e| match e {
            Error::Config(msg) => Error::Config(format!(
                "{msg}\nGranite AR variants need embed_tokens.onnx in the precision directory. Older bundles may pre-date this graph - re-export with the latest pipeline at https://github.com/sammcj/granite-speech-4.1-onnx."
            )),
            other => other,
        })?;

        let encoder = build_session(&encoder_path, &exec_config)?;
        let prompt_encode = build_session(&prompt_encode_path, &exec_config)?;
        let decode_step = build_session(&decode_step_path, &exec_config)?;
        let embed_tokens = build_session(&embed_tokens_path, &exec_config)?;

        Ok(Self {
            encoder,
            prompt_encode,
            decode_step,
            embed_tokens,
        })
    }

    /// Run the encoder + projector graph on stacked log-mel features.
    /// `input_features` shape is `[1, T_stacked, 160]`.
    pub(crate) fn run_encoder(&mut self, input_features: &Array3<f32>) -> Result<EncoderOutput> {
        let feats = ort::value::TensorRef::<f32>::from_array_view(input_features.view())?;
        let outputs = self.encoder.run(ort::inputs!("input_features" => feats))?;

        let (e_shape, e_data) = outputs["audio_embeds"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("encoder.audio_embeds extract: {e}")))?;
        let audio_embeds = Array3::from_shape_vec(
            (
                e_shape[0] as usize,
                e_shape[1] as usize,
                e_shape[2] as usize,
            ),
            e_data.to_vec(),
        )
        .map_err(|e| Error::Model(format!("encoder.audio_embeds reshape: {e}")))?;

        let (s_shape, s_data) = outputs["audio_embed_sizes"]
            .try_extract_tensor::<i64>()
            .map_err(|e| Error::Model(format!("encoder.audio_embed_sizes extract: {e}")))?;
        let audio_embed_sizes = Array1::from_shape_vec(s_shape[0] as usize, s_data.to_vec())
            .map_err(|e| Error::Model(format!("encoder.audio_embed_sizes reshape: {e}")))?;

        Ok(EncoderOutput {
            audio_embeds,
            audio_embed_sizes,
        })
    }

    /// Embed token IDs through the LLM's input-embedding table.
    /// Input: `[B, N]` int64. Output: `[B, N, HIDDEN_SIZE]` float32.
    pub(crate) fn run_embed_tokens(&mut self, input_ids: &Array2<i64>) -> Result<Array3<f32>> {
        let ids = ort::value::TensorRef::<i64>::from_array_view(input_ids.view())?;
        let outputs = self.embed_tokens.run(ort::inputs!("input_ids" => ids))?;
        let (shape, data) = outputs["inputs_embeds"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("embed_tokens.inputs_embeds extract: {e}")))?;
        Array3::from_shape_vec(
            (shape[0] as usize, shape[1] as usize, shape[2] as usize),
            data.to_vec(),
        )
        .map_err(|e| Error::Model(format!("embed_tokens.inputs_embeds reshape: {e}")))
    }

    /// Run prompt encoding (LLM forward over the full prompt). Returns
    /// the last-position logits and the populated KV cache. The cache
    /// will then be threaded into [`run_decode_step`] one token at a time.
    pub(crate) fn run_prompt_encode(
        &mut self,
        inputs_embeds: &Array3<f32>,
        position_ids: &Array2<i64>,
        attention_mask_4d: &Array4<f32>,
    ) -> Result<(Array1<f32>, KvCache)> {
        let n = inputs_embeds.shape()[1];

        let embeds_ref = ort::value::TensorRef::<f32>::from_array_view(inputs_embeds.view())?;
        let pos_ref = ort::value::TensorRef::<i64>::from_array_view(position_ids.view())?;
        let mask_ref = ort::value::TensorRef::<f32>::from_array_view(attention_mask_4d.view())?;

        let inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = vec![
            (Cow::Borrowed("inputs_embeds"), embeds_ref.into()),
            (Cow::Borrowed("position_ids"), pos_ref.into()),
            (Cow::Borrowed("attention_mask"), mask_ref.into()),
        ];

        let outputs = self.prompt_encode.run(inputs)?;
        let last_logits = extract_last_position_logits(&outputs, n)?;
        let kv = read_present_kv(&outputs)?;
        Ok((last_logits, kv))
    }

    /// Run one decode step. `inputs_embeds` is `[1, 1, HIDDEN_SIZE]`
    /// (one token's embedding); `position_ids` is `[1, 1]`;
    /// `attention_mask` is `[1, 1, 1, T_total]` all-zeros (no padding,
    /// allow attend-all over past+current). `past_kv` carries the
    /// running cache; the returned cache replaces it for the next step.
    pub(crate) fn run_decode_step(
        &mut self,
        inputs_embeds: &Array3<f32>,
        position_ids: &Array2<i64>,
        attention_mask_4d: &Array4<f32>,
        past_kv: &KvCache,
    ) -> Result<(Array1<f32>, KvCache)> {
        let embeds_ref = ort::value::TensorRef::<f32>::from_array_view(inputs_embeds.view())?;
        let pos_ref = ort::value::TensorRef::<i64>::from_array_view(position_ids.view())?;
        let mask_ref = ort::value::TensorRef::<f32>::from_array_view(attention_mask_4d.view())?;

        let mut inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> =
            Vec::with_capacity(3 + 2 * NUM_LLM_LAYERS);
        inputs.push((Cow::Borrowed("inputs_embeds"), embeds_ref.into()));
        inputs.push((Cow::Borrowed("position_ids"), pos_ref.into()));
        inputs.push((Cow::Borrowed("attention_mask"), mask_ref.into()));

        // Bind 80 KV-cache tensors as zero-copy views.
        let past_refs: Vec<(String, ort::value::TensorRef<'_, f32>)> = past_kv
            .layers
            .iter()
            .enumerate()
            .flat_map(|(i, (k, v))| {
                let kref = ort::value::TensorRef::<f32>::from_array_view(k.view())
                    .expect("past key view shape valid");
                let vref = ort::value::TensorRef::<f32>::from_array_view(v.view())
                    .expect("past value view shape valid");
                vec![
                    (format!("past_key_values.{i}.key"), kref),
                    (format!("past_key_values.{i}.value"), vref),
                ]
            })
            .collect();
        for (name, r) in past_refs {
            inputs.push((Cow::Owned(name), r.into()));
        }

        let outputs = self.decode_step.run(inputs)?;
        let last_logits = extract_last_position_logits(&outputs, 1)?;
        let kv = read_present_kv(&outputs)?;
        Ok((last_logits, kv))
    }
}

fn require_file(dir: &Path, name: &str) -> Result<PathBuf> {
    let p = dir.join(name);
    if !p.exists() {
        return Err(Error::Config(format!(
            "Missing {name} in {}",
            dir.display()
        )));
    }
    Ok(p)
}

fn build_session(path: &Path, exec_config: &ExecutionConfig) -> Result<Session> {
    let builder = Session::builder()?;
    let mut builder = exec_config.apply_to_session_builder(builder)?;
    let session = builder.commit_from_file(path)?;
    Ok(session)
}

/// Pull the last-position logits out of an `[B, N, V]` logits output,
/// regardless of whether the graph emits all positions or just the last
/// one. We're a single-batch greedy decoder so we only ever need the
/// final next-token distribution.
fn extract_last_position_logits(
    outputs: &ort::session::SessionOutputs,
    expected_n: usize,
) -> Result<Array1<f32>> {
    let (shape, data) = outputs["logits"]
        .try_extract_tensor::<f32>()
        .map_err(|e| Error::Model(format!("logits extract: {e}")))?;
    let n_positions = shape[1] as usize;
    let vocab = shape[2] as usize;
    if n_positions == 0 || vocab == 0 {
        return Err(Error::Model(format!(
            "logits has zero-sized dim: shape={shape:?} (expected_n={expected_n})"
        )));
    }
    let last_start = (n_positions - 1) * vocab;
    Ok(Array1::from_vec(
        data[last_start..last_start + vocab].to_vec(),
    ))
}

fn read_present_kv(outputs: &ort::session::SessionOutputs) -> Result<KvCache> {
    let mut layers = Vec::with_capacity(NUM_LLM_LAYERS);
    for i in 0..NUM_LLM_LAYERS {
        let k = extract_cache_4d(outputs, &format!("present.{i}.key"))?;
        let v = extract_cache_4d(outputs, &format!("present.{i}.value"))?;
        layers.push((k, v));
    }
    Ok(KvCache { layers })
}

fn extract_cache_4d(outputs: &ort::session::SessionOutputs, name: &str) -> Result<Array4<f32>> {
    let (shape, data) = outputs[name]
        .try_extract_tensor::<f32>()
        .map_err(|e| Error::Model(format!("{name} extract: {e}")))?;
    Array4::from_shape_vec(
        (
            shape[0] as usize,
            shape[1] as usize,
            shape[2] as usize,
            shape[3] as usize,
        ),
        data.to_vec(),
    )
    .map_err(|e| Error::Model(format!("{name} reshape: {e}")))
}

/// Build a 4-D causal additive attention mask of shape `[1, 1, N, N]`,
/// suitable for the prompt-encode graph. `0.0` for allowed positions,
/// `-INFINITY` for masked positions in the upper triangle (i.e. the
/// future). With a single utterance and no padding the whole thing
/// reduces to a standard upper-triangular causal mask.
pub(crate) fn build_causal_mask(n: usize) -> Array4<f32> {
    let mut mask = Array4::<f32>::zeros((1, 1, n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            mask[[0, 0, i, j]] = f32::NEG_INFINITY;
        }
    }
    mask
}

/// Build the decode-step attention mask of shape `[1, 1, 1, T_total]`,
/// all zeros (the new token may attend to every cached + current
/// position; no padding to mask).
pub(crate) fn build_decode_mask(t_total: usize) -> Array4<f32> {
    Array4::<f32>::zeros((1, 1, 1, t_total))
}

/// Splice projector audio embeddings into a text-embedding sequence at
/// every position where `input_ids[i] == audio_token_id`. The number of
/// such positions must equal the number of audio embedding rows; the
/// caller sizes the prompt's `<|audio|>` run from `audio_embed_sizes[0]`
/// returned by [`GraniteArModel::run_encoder`].
pub(crate) fn splice_audio_embeddings(
    text_embeds: &mut Array3<f32>,
    input_ids: &Array2<i64>,
    audio_embeds: &Array3<f32>,
    audio_token_id: i64,
) -> Result<()> {
    if text_embeds.shape()[0] != 1 || input_ids.shape()[0] != 1 || audio_embeds.shape()[0] != 1 {
        return Err(Error::Model(
            "splice_audio_embeddings only supports batch=1 (parakeet-rs single-utterance API)"
                .into(),
        ));
    }
    let n = text_embeds.shape()[1];
    let h = text_embeds.shape()[2];
    if audio_embeds.shape()[2] != h {
        return Err(Error::Model(format!(
            "audio_embeds hidden dim {} != text_embeds hidden dim {}",
            audio_embeds.shape()[2],
            h
        )));
    }
    let positions: Vec<usize> = (0..n)
        .filter(|&i| input_ids[[0, i]] == audio_token_id)
        .collect();
    let n_audio = audio_embeds.shape()[1];
    if positions.len() != n_audio {
        return Err(Error::Model(format!(
            "splice mismatch: {} <|audio|> placeholder slots vs {n_audio} projector embeddings. The prompt must repeat the audio token exactly audio_embed_sizes times.",
            positions.len()
        )));
    }
    for (slot, &pos) in positions.iter().enumerate() {
        for k in 0..h {
            text_embeds[[0, pos, k]] = audio_embeds[[0, slot, k]];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_kv_cache_has_zero_past_len() {
        let kv = KvCache::empty();
        assert_eq!(kv.layers.len(), NUM_LLM_LAYERS);
        assert_eq!(kv.past_len(), 0);
        for (k, v) in &kv.layers {
            assert_eq!(k.shape(), &[1, NUM_KV_HEADS, 0, HEAD_DIM]);
            assert_eq!(v.shape(), &[1, NUM_KV_HEADS, 0, HEAD_DIM]);
        }
    }

    #[test]
    fn causal_mask_zeroes_lower_triangle_and_negs_upper() {
        let m = build_causal_mask(3);
        // Diagonal and below = 0
        for i in 0..3 {
            for j in 0..=i {
                assert_eq!(m[[0, 0, i, j]], 0.0);
            }
            for j in (i + 1)..3 {
                assert!(m[[0, 0, i, j]].is_infinite() && m[[0, 0, i, j]] < 0.0);
            }
        }
    }

    #[test]
    fn decode_mask_is_all_zero() {
        let m = build_decode_mask(5);
        assert_eq!(m.shape(), &[1, 1, 1, 5]);
        assert!(m.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn splice_overwrites_audio_token_positions() {
        const H: usize = HIDDEN_SIZE;
        let mut text = Array3::<f32>::zeros((1, 5, H));
        for k in 0..H {
            text[[0, 1, k]] = 99.0;
            text[[0, 3, k]] = 99.0;
        }
        let ids = ndarray::arr2(&[[10, 100352, 11, 100352, 12]]);
        let mut audio = Array3::<f32>::zeros((1, 2, H));
        for k in 0..H {
            audio[[0, 0, k]] = 1.0;
            audio[[0, 1, k]] = 2.0;
        }
        splice_audio_embeddings(&mut text, &ids, &audio, 100352).unwrap();
        assert_eq!(text[[0, 0, 0]], 0.0);
        assert_eq!(text[[0, 1, 0]], 1.0);
        assert_eq!(text[[0, 2, 0]], 0.0);
        assert_eq!(text[[0, 3, 0]], 2.0);
        assert_eq!(text[[0, 4, 0]], 0.0);
    }

    #[test]
    fn splice_fails_when_slot_count_mismatches() {
        let mut text = Array3::<f32>::zeros((1, 3, HIDDEN_SIZE));
        let ids = ndarray::arr2(&[[100352, 1, 2]]);
        let audio = Array3::<f32>::zeros((1, 2, HIDDEN_SIZE));
        let err = splice_audio_embeddings(&mut text, &ids, &audio, 100352).unwrap_err();
        assert!(matches!(err, Error::Model(msg) if msg.contains("slots vs")));
    }
}
