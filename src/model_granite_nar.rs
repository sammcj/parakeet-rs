//! ONNX session wrapper for the IBM Granite Speech 4.1 2b NAR
//! (non-autoregressive) variant.
//!
//! Three graphs cooperate, no KV cache, no per-token loop:
//! - `encoder.onnx` runs the Conformer + CTC heads + BPE-collapsing
//!   projector. Outputs `bpe_logits_dense`, `bpe_mask`, `audio_embeds`,
//!   `audio_lengths`, and `char_logits` (diagnostic only).
//! - `embed_tokens.onnx` looks up text-token embeddings for the CTC
//!   draft after slot insertion (same graph as base/plus).
//! - `editor.onnx` runs the bidirectional NLE editor over the
//!   concatenation of `audio_embeds[:audio_len]` and the slot-inserted
//!   text embeddings. Output is `logits [1, N, V_LLM]`.
//!
//! See [`granite-speech-4.1-onnx`](https://github.com/sammcj/granite-speech-4.1-onnx)
//! for the bundle layout and the canonical call sequence.
//!
//! Bundle layout (same precision tiers as base / plus, without
//! `prompt_encode.onnx` / `decode_step.onnx`):
//!
//! ```text
//! <bundle_dir>/
//!   {fp32, fp16w, int8}/
//!     encoder.onnx       + encoder.onnx_data
//!     editor.onnx        + editor.onnx_data
//!     embed_tokens.onnx  + embed_tokens.onnx_data
//!   tokenizer.json
//!   preprocessor_config.json
//!   granite_export_metadata.json
//! ```

use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::model_granite::GranitePrecision;
use ndarray::{Array1, Array2, Array3, Array4};
use ort::session::{Session, SessionInputValue};
use std::borrow::Cow;
use std::path::{Path, PathBuf};

/// Hidden state size of the LLM/editor and projector output. Same value
/// as the base/plus variants because the LLM is the same Granite 4.0 1B.
pub(crate) const HIDDEN_SIZE: usize = 2048;

/// Result of running the NAR encoder graph. The CTC draft is recovered
/// from `bpe_logits_dense` masked by `bpe_mask`; the audio embeddings
/// and their valid length feed the editor's input-embeds splice.
pub(crate) struct NarEncoderOutput {
    /// Per-position raw BPE logits, `[B, T_bpe, V_bpe]`. The CTC
    /// vocabulary is the LLM vocabulary plus a leading blank class at
    /// index 0; index `i` corresponds to LLM token `i - 1`.
    pub(crate) bpe_logits_dense: Array3<f32>,
    /// Boolean mask in float, `[B, T_bpe]`. `1.0` for real positions,
    /// `0.0` for padding. The encoder pads the BPE output to a
    /// rectangular tensor; only mask=1 positions feed the CTC decode.
    pub(crate) bpe_mask: Array2<f32>,
    /// Q-Former projector output, `[B, T_audio, 2048]`. Only the first
    /// `audio_lengths[b]` rows per sample are valid.
    pub(crate) audio_embeds: Array3<f32>,
    /// Per-sample number of valid audio embedding rows.
    pub(crate) audio_lengths: Array1<i64>,
}

pub(crate) struct GraniteNarModel {
    encoder: Session,
    embed_tokens: Session,
    editor: Session,
}

impl GraniteNarModel {
    pub(crate) fn from_pretrained<P: AsRef<Path>>(
        bundle_dir: P,
        precision: GranitePrecision,
        exec_config: ExecutionConfig,
    ) -> Result<Self> {
        let bundle_dir = bundle_dir.as_ref();
        let prec_dir = bundle_dir.join(precision.dirname());
        if !prec_dir.is_dir() {
            return Err(Error::Config(format!(
                "Granite Speech NAR bundle is missing the '{}' precision directory at {}. Available bundles ship fp32/, fp16w/, and int8/ subdirs.",
                precision.dirname(),
                bundle_dir.display()
            )));
        }

        let encoder_path = require_file(&prec_dir, "encoder.onnx")?;
        let editor_path = require_file(&prec_dir, "editor.onnx")?;
        let embed_tokens_path = require_file(&prec_dir, "embed_tokens.onnx")?;

        let encoder = build_session(&encoder_path, &exec_config)?;
        let editor = build_session(&editor_path, &exec_config)?;
        let embed_tokens = build_session(&embed_tokens_path, &exec_config)?;

        Ok(Self {
            encoder,
            embed_tokens,
            editor,
        })
    }

    /// Run the NAR encoder. `input_features` is `[1, T_stacked, 160]`
    /// from [`crate::GraniteNar::extract_input_features_nar`];
    /// `attention_mask` is `[1, T_stacked]` int64 (1 for real frames,
    /// 0 for padding).
    pub(crate) fn run_encoder(
        &mut self,
        input_features: &Array3<f32>,
        attention_mask: &Array2<i64>,
    ) -> Result<NarEncoderOutput> {
        let feats = ort::value::TensorRef::<f32>::from_array_view(input_features.view())?;
        let mask = ort::value::TensorRef::<i64>::from_array_view(attention_mask.view())?;
        let outputs = self.encoder.run(ort::inputs!(
            "input_features" => feats,
            "attention_mask" => mask,
        ))?;

        let bpe_logits_dense = extract_3d(&outputs, "bpe_logits_dense")?;
        let bpe_mask = extract_bpe_mask(&outputs)?;
        let audio_embeds = extract_3d(&outputs, "audio_embeds")?;
        let audio_lengths = extract_1d_i64(&outputs, "audio_lengths")?;

        Ok(NarEncoderOutput {
            bpe_logits_dense,
            bpe_mask,
            audio_embeds,
            audio_lengths,
        })
    }

    /// Embed token IDs through the LLM input-embedding table. Same graph
    /// signature as the base/plus `embed_tokens.onnx`; reused here so the
    /// CTC-draft + slot tokens map back to the editor's input-embeds.
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

    /// Run the bidirectional editor over the concatenated
    /// `(audio_embeds, text_embeds_with_slots)` sequence. Returns the
    /// raw `logits [1, N, V_LLM]`. The caller slices off the audio
    /// prefix and argmaxes the text segment to get final token IDs.
    pub(crate) fn run_editor(
        &mut self,
        inputs_embeds: &Array3<f32>,
        position_ids: &Array2<i64>,
        attention_mask_4d: &Array4<f32>,
    ) -> Result<Array3<f32>> {
        let embeds_ref = ort::value::TensorRef::<f32>::from_array_view(inputs_embeds.view())?;
        let pos_ref = ort::value::TensorRef::<i64>::from_array_view(position_ids.view())?;
        let mask_ref = ort::value::TensorRef::<f32>::from_array_view(attention_mask_4d.view())?;

        let inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = vec![
            (Cow::Borrowed("inputs_embeds"), embeds_ref.into()),
            (Cow::Borrowed("position_ids"), pos_ref.into()),
            (Cow::Borrowed("attention_mask"), mask_ref.into()),
        ];

        let outputs = self.editor.run(inputs)?;
        let (shape, data) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("editor.logits extract: {e}")))?;
        Array3::from_shape_vec(
            (shape[0] as usize, shape[1] as usize, shape[2] as usize),
            data.to_vec(),
        )
        .map_err(|e| Error::Model(format!("editor.logits reshape: {e}")))
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

fn extract_3d(outputs: &ort::session::SessionOutputs, name: &str) -> Result<Array3<f32>> {
    let (shape, data) = outputs[name]
        .try_extract_tensor::<f32>()
        .map_err(|e| Error::Model(format!("{name} extract: {e}")))?;
    Array3::from_shape_vec(
        (shape[0] as usize, shape[1] as usize, shape[2] as usize),
        data.to_vec(),
    )
    .map_err(|e| Error::Model(format!("{name} reshape: {e}")))
}

/// `bpe_mask` is exported as bool by the NAR encoder despite the
/// `granite_export_metadata.json` declaring float32. The host-side CTC
/// decoder treats it as `>= 0.5` so we cast to `f32` here.
fn extract_bpe_mask(outputs: &ort::session::SessionOutputs) -> Result<Array2<f32>> {
    if let Ok((shape, data)) = outputs["bpe_mask"].try_extract_tensor::<bool>() {
        let floats: Vec<f32> = data.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        return Array2::from_shape_vec((shape[0] as usize, shape[1] as usize), floats)
            .map_err(|e| Error::Model(format!("bpe_mask reshape: {e}")));
    }
    let (shape, data) = outputs["bpe_mask"]
        .try_extract_tensor::<f32>()
        .map_err(|e| Error::Model(format!("bpe_mask extract: {e}")))?;
    Array2::from_shape_vec((shape[0] as usize, shape[1] as usize), data.to_vec())
        .map_err(|e| Error::Model(format!("bpe_mask reshape: {e}")))
}

fn extract_1d_i64(outputs: &ort::session::SessionOutputs, name: &str) -> Result<Array1<i64>> {
    let (shape, data) = outputs[name]
        .try_extract_tensor::<i64>()
        .map_err(|e| Error::Model(format!("{name} extract: {e}")))?;
    Array1::from_shape_vec(shape[0] as usize, data.to_vec())
        .map_err(|e| Error::Model(format!("{name} reshape: {e}")))
}
