//! IBM Granite Speech 4.1 2b NAR (non-autoregressive) ASR engine.
//!
//! 440M Conformer encoder + Q-Former projector + bidirectional NLE
//! editor over a CTC-draft + insertion-slot sequence. Single-pass
//! parallel decode: no autoregressive loop, no KV cache. English only,
//! transcription only (no translation, no punctuation, no speaker
//! tags). Trades feature breadth for latency; the editor sees the whole
//! utterance at once and rewrites the CTC draft in one forward pass.
//!
//! Loads from a bundle produced by
//! [`sammcj/granite-speech-4.1-onnx`](https://github.com/sammcj/granite-speech-4.1-onnx).
//! See [`model_granite_nar`](crate::model_granite_nar) for the bundle
//! layout and ONNX IO contract.
//!
//! ## Quick start
//!
//! ```no_run
//! use parakeet_rs::{GraniteNar, GraniteNarOptions};
//!
//! let mut nar = GraniteNar::from_pretrained("./granite-speech-4.1-2b-nar-onnx", None)?;
//! let audio: Vec<f32> = vec![/* 16 kHz mono float samples */];
//! let text = nar.transcribe_audio(&audio, &GraniteNarOptions::default())?;
//! println!("{text}");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::audio_granite;
use crate::decoder_nar::{add_insertion_slots, argmax_text_segment, ctc_greedy_decode};
use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::granite_common::require_token;
use crate::model_granite::GranitePrecision;
use crate::model_granite_nar::{GraniteNarModel, HIDDEN_SIZE};
use ndarray::{Array2, Array3, Array4};
use std::path::Path;
use tokenizers::Tokenizer;

/// EOS literal for the Granite 4 LLM tokeniser. Used both as the
/// stop-token (irrelevant in NAR since the editor is single-pass) and
/// as the slot-fill token in `add_insertion_slots`.
const EOS_TOKEN: &str = "<|end_of_text|>";

/// `embedding_multiplier` from the upstream Granite 4 LLM config.
/// The NAR variant has `scale_projected_embeddings = true`, so the
/// host divides `audio_embeds` by this value before splicing them
/// alongside the LLM-side text embeddings; the LLM expects its
/// embedding table outputs at this scale and the projector emits at
/// the unscaled scale. Verified against
/// [`ibm-granite/granite-speech-4.1-2b-nar`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar)
/// `config.json`.
const EMBEDDING_MULTIPLIER: f32 = 12.0;

/// User-facing options for one NAR transcription call.
///
/// The NAR pipeline has no task switches (it's English ASR only, no
/// punctuation, no translation, no speaker tags), so this struct is
/// intentionally minimal. It exists so the three Granite engines share
/// the `transcribe_audio(&audio, &opts)` shape.
#[derive(Debug, Clone)]
pub struct GraniteNarOptions {
    /// Strip leading and trailing whitespace from the decoded
    /// transcript. Defaults to `true`. Set to `false` to keep raw
    /// detokeniser output, including any leading space the BPE merge
    /// rules emit.
    pub trim_whitespace: bool,
}

impl GraniteNarOptions {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for GraniteNarOptions {
    fn default() -> Self {
        Self {
            trim_whitespace: true,
        }
    }
}

/// IBM Granite Speech 4.1 2b NAR ASR engine.
pub struct GraniteNar {
    model: GraniteNarModel,
    tokenizer: Tokenizer,
    eos_token_id: i64,
}

impl GraniteNar {
    /// Load the NAR model from a bundle directory at the default
    /// `Fp16w` precision. See [`model_granite_nar`](crate::model_granite_nar)
    /// for the expected layout.
    pub fn from_pretrained<P: AsRef<Path>>(
        bundle_dir: P,
        exec_config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        Self::from_pretrained_with_precision(bundle_dir, GranitePrecision::default(), exec_config)
    }

    /// Same as [`from_pretrained`](Self::from_pretrained) but with an
    /// explicit precision tier (`fp32`, `fp16w`, or `int8`).
    pub fn from_pretrained_with_precision<P: AsRef<Path>>(
        bundle_dir: P,
        precision: GranitePrecision,
        exec_config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        let bundle_dir = bundle_dir.as_ref();
        let exec = exec_config.unwrap_or_default();

        let model = GraniteNarModel::from_pretrained(bundle_dir, precision, exec)?;

        let tok_path = bundle_dir.join("tokenizer.json");
        if !tok_path.exists() {
            return Err(Error::Config(format!(
                "Missing tokenizer.json in {}",
                bundle_dir.display()
            )));
        }
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| Error::Tokenizer(format!("Failed to load tokenizer.json: {e}")))?;

        let eos_token_id = require_token(&tokenizer, EOS_TOKEN)?;

        Ok(Self {
            model,
            tokenizer,
            eos_token_id,
        })
    }

    /// Run only the encoder + projector and return the raw outputs.
    /// `(input_features, attention_mask)` come from
    /// [`GraniteNar::extract_input_features_nar`]. Useful for fixture
    /// parity tests against `expected_audio_embeds.npy`.
    pub fn run_encoder(
        &mut self,
        input_features: &Array3<f32>,
        attention_mask: &Array2<i64>,
    ) -> Result<(Array3<f32>, usize)> {
        let enc = self.model.run_encoder(input_features, attention_mask)?;
        let n_audio = enc.audio_lengths[0] as usize;
        Ok((enc.audio_embeds, n_audio))
    }

    /// Transcribe raw 16 kHz mono float audio. Returns a plain
    /// transcript - NAR is transcription-only with no punctuation or
    /// structural tags.
    pub fn transcribe_audio(
        &mut self,
        audio: &[f32],
        options: &GraniteNarOptions,
    ) -> Result<String> {
        if audio.is_empty() {
            return Ok(String::new());
        }

        // 1. NAR audio frontend: log-mel + 2-frame stack with
        //    truncation to `2 * (T // (2 * hop))` mel frames before
        //    the log/normalise step. Returns `(input_features, attention_mask)`.
        let (input_features, attention_mask) = audio_granite::extract_input_features_nar(audio)?;

        // 2. Encoder + projector + CTC heads.
        let enc = self.model.run_encoder(&input_features, &attention_mask)?;
        let audio_len = enc.audio_lengths[0] as usize;
        if audio_len == 0 {
            return Ok(String::new());
        }
        if enc.audio_embeds.shape()[1] < audio_len {
            return Err(Error::Model(format!(
                "encoder returned {} audio_embeds rows but audio_lengths[0] reports {audio_len}",
                enc.audio_embeds.shape()[1]
            )));
        }

        // 3. CTC greedy decode of `bpe_logits_dense`. Returns LLM
        //    token IDs (already mapped from BPE indices, blanks dropped).
        let ctc_token_ids = ctc_greedy_decode(&enc.bpe_logits_dense, &enc.bpe_mask)?;

        // 4. Detokenise the CTC draft, normalise to lowercase + a
        //    single space if empty, then re-tokenise. The detokenise +
        //    re-tokenise round-trip mirrors the upstream reference and
        //    canonicalises the BPE merge boundaries.
        let ctc_text = self.detokenise_ctc(&ctc_token_ids)?;
        let llm_token_ids = self.retokenise(&ctc_text)?;

        // 5. Insertion slots: `[eos, t0, eos, t1, eos, ...]` padded to
        //    >= 8 entries. The editor learns to rewrite or expand each
        //    `eos` slot.
        let slots = add_insertion_slots(&llm_token_ids, self.eos_token_id);
        let slots_len = slots.len();

        // 6. Embed the slot tokens through the LLM input table.
        let slot_ids = Array2::from_shape_vec((1, slots_len), slots)
            .map_err(|e| Error::Model(format!("slot ids reshape: {e}")))?;
        let text_embeds = self.model.run_embed_tokens(&slot_ids)?;

        // 7. Scale audio embeddings (`scale_projected_embeddings = true`
        //    on this checkpoint), then concatenate the audio prefix
        //    with the text-with-slots embeddings into one flat sequence.
        let n_total = audio_len + slots_len;
        let mut inputs_embeds = Array3::<f32>::zeros((1, n_total, HIDDEN_SIZE));
        for t in 0..audio_len {
            for k in 0..HIDDEN_SIZE {
                inputs_embeds[[0, t, k]] = enc.audio_embeds[[0, t, k]] / EMBEDDING_MULTIPLIER;
            }
        }
        for t in 0..slots_len {
            for k in 0..HIDDEN_SIZE {
                inputs_embeds[[0, audio_len + t, k]] = text_embeds[[0, t, k]];
            }
        }

        // 8. Build the editor's auxiliary inputs. position_ids is a
        //    plain arange; the 4-D attention mask is identically zero
        //    (additive convention, bidirectional, no masking).
        let position_ids = Array2::from_shape_vec((1, n_total), (0..n_total as i64).collect())
            .map_err(|e| Error::Model(format!("position_ids reshape: {e}")))?;
        let attention_mask_4d = Array4::<f32>::zeros((1, 1, n_total, n_total));

        // 9. Editor forward pass. Argmax the text segment to recover
        //    the final LLM token IDs.
        let logits = self
            .model
            .run_editor(&inputs_embeds, &position_ids, &attention_mask_4d)?;
        let final_ids = argmax_text_segment(&logits, audio_len)?;

        // 10. Detokenise (skip EOS slot fillers and any stray special
        //     tokens). Optionally trim leading/trailing whitespace.
        let text = self
            .tokenizer
            .decode(
                &final_ids.iter().map(|&i| i as u32).collect::<Vec<_>>(),
                true,
            )
            .map_err(|e| Error::Tokenizer(format!("decode failed: {e}")))?;
        if options.trim_whitespace {
            Ok(text.trim().to_string())
        } else {
            Ok(text)
        }
    }

    /// Multi-channel-friendly variant: mixes interleaved audio to mono
    /// and rejects non-16 kHz inputs.
    pub fn transcribe_samples(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        channels: u16,
        options: &GraniteNarOptions,
    ) -> Result<String> {
        let mono = audio_granite::prepare_mono_16k(audio, sample_rate, channels)?;
        self.transcribe_audio(&mono, options)
    }

    /// Compute the post-stack `input_features` tensor and matching
    /// `attention_mask` that `encoder.onnx` expects for the NAR variant.
    ///
    /// Static method exposed so tooling and parity tests can compare
    /// against the bundle's `expected_input_features.npy` and
    /// `expected_attention_mask.npy` fixtures without having to load the
    /// full ONNX session.
    pub fn extract_input_features_nar(audio: &[f32]) -> Result<(Array3<f32>, Array2<i64>)> {
        audio_granite::extract_input_features_nar(audio)
    }

    fn detokenise_ctc(&self, ids: &[i64]) -> Result<String> {
        if ids.is_empty() {
            return Ok(" ".to_string());
        }
        let text = self
            .tokenizer
            .decode(&ids.iter().map(|&i| i as u32).collect::<Vec<_>>(), true)
            .map_err(|e| Error::Tokenizer(format!("CTC draft decode failed: {e}")))?;
        let normalised = text.trim().to_lowercase();
        Ok(if normalised.is_empty() {
            " ".to_string()
        } else {
            normalised
        })
    }

    fn retokenise(&self, text: &str) -> Result<Vec<i64>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| Error::Tokenizer(format!("CTC draft re-encode failed: {e}")))?;
        Ok(encoding.get_ids().iter().map(|&i| i as i64).collect())
    }
}

