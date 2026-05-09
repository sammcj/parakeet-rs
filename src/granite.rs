//! IBM Granite Speech 4.1 base 2b ASR engine (autoregressive).
//!
//! 440M Conformer encoder + Q-Former projector + 1B Granite 4.0 LLM
//! decoder (LoRA-adapted). Multilingual transcription across English,
//! French, German, Spanish, Portuguese, and Japanese; bidirectional
//! translation between English and the other supported languages.
//! Also supports keyword biasing via prompt.
//!
//! Loads from a bundle produced by
//! [`sammcj/granite-speech-4.1-onnx`](https://github.com/sammcj/granite-speech-4.1-onnx).
//! See [`model_granite`](crate::model_granite) for the bundle layout.
//!
//! ## Quick start
//!
//! ```no_run
//! use parakeet_rs::{Granite, GraniteOptions, GraniteTask};
//!
//! let mut granite = Granite::from_pretrained("./granite-speech-4.1-2b-onnx", None)?;
//! let audio: Vec<f32> = vec![/* 16 kHz mono float samples */];
//! let opts = GraniteOptions::transcribe_with_punctuation();
//! let text = granite.transcribe_audio(&audio, &opts)?;
//! println!("{text}");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Acceleration
//!
//! Pass an [`ExecutionConfig`] with the desired execution provider (CUDA /
//! DirectML / etc., feature-gated). The bundles' `fp16w/` precision tier is
//! the recommended default: weights stored as FP16, compute and IO stay FP32,
//! transcripts byte-exact vs the PyTorch reference.
//!
//! On Apple Silicon the CPU EP at fp16w is the recommended path. The CoreML
//! EP is reachable via `with_execution_provider(ExecutionProvider::CoreML)`
//! and configured via the `with_coreml_*` builder methods on
//! [`ExecutionConfig`]. The dynamic-shape bundles published in
//! [`sammcj/granite-speech-4.1-onnx`](https://github.com/sammcj/granite-speech-4.1-onnx)
//! do not compile under CoreML's MIL framework, which requires
//! statically-known tensor shapes; the provider option builders are present
//! for use against statically-shaped re-exports.

use crate::audio_granite;
use crate::decoder_granite::{decode_greedy, run_prompt, DecodeOptions};
use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::granite_common::{expand_audio_token, require_token, AUDIO_TOKEN};
use crate::model_granite::{splice_audio_embeddings, GraniteArModel, GranitePrecision};
use ndarray::Array2;
use std::path::Path;
use tokenizers::Tokenizer;

/// End-of-text token literal used to stop autoregressive generation in
/// the Granite 4 LLM tokenizer.
const EOS_TOKEN: &str = "<|end_of_text|>";

/// Default `max_new_tokens` for transcription. Long-form transcription
/// can exceed this; users can raise it via [`GraniteOptions`].
const DEFAULT_MAX_NEW_TOKENS: usize = 512;

/// Languages the base 2b model supports for ASR (transcription) and as
/// translation targets.
const SUPPORTED_LANGUAGES: &[(&str, &str)] = &[
    ("en", "English"),
    ("fr", "French"),
    ("de", "German"),
    ("es", "Spanish"),
    ("pt", "Portuguese"),
    ("ja", "Japanese"),
];

/// Task selection for [`Granite::transcribe_audio`]. The Granite base
/// model is feature-switched entirely via the user-side prompt; this
/// enum is just typed sugar that renders to the documented prompt
/// strings so callers don't hand-write them.
#[derive(Debug, Clone)]
pub enum GraniteTask {
    /// Plain ASR with no punctuation or capitalisation.
    TranscribeRaw,
    /// ASR with proper punctuation and capitalisation. This is the
    /// default and matches the prompt used in the upstream model card.
    TranscribeWithPunctuation,
    /// ASR with a keyword bias list. Names, acronyms, or technical
    /// terms passed here are more likely to be recognised correctly.
    TranscribeWithKeywords(Vec<String>),
    /// Speech translation to one of the supported target languages.
    /// `target` is the human-readable English name (e.g. `"French"`).
    TranslateTo(String),
}

impl GraniteTask {
    /// Render this task to the canonical user-message text. The audio
    /// placeholder `<|audio|>` is included as a single literal token;
    /// it will be expanded to the projector-output length before
    /// tokenisation.
    fn to_user_prompt(&self) -> String {
        match self {
            GraniteTask::TranscribeRaw => {
                format!("{AUDIO_TOKEN}can you transcribe the speech into a written format?")
            }
            GraniteTask::TranscribeWithPunctuation => format!(
                "{AUDIO_TOKEN}transcribe the speech with proper punctuation and capitalization."
            ),
            GraniteTask::TranscribeWithKeywords(kws) => {
                let kws = kws.join(", ");
                format!("{AUDIO_TOKEN}transcribe the speech to text. Keywords: {kws}")
            }
            GraniteTask::TranslateTo(lang) => {
                format!("{AUDIO_TOKEN}translate the speech to {lang}.")
            }
        }
    }
}

/// User-facing options for one transcription call.
#[derive(Debug, Clone)]
pub struct GraniteOptions {
    pub task: GraniteTask,
    /// Hard cap on tokens generated. Defaults to 512.
    pub max_new_tokens: usize,
}

impl GraniteOptions {
    /// Default: ASR with punctuation and capitalisation.
    pub fn transcribe_with_punctuation() -> Self {
        Self {
            task: GraniteTask::TranscribeWithPunctuation,
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
        }
    }

    pub fn transcribe_raw() -> Self {
        Self {
            task: GraniteTask::TranscribeRaw,
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
        }
    }

    pub fn transcribe_with_keywords(keywords: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            task: GraniteTask::TranscribeWithKeywords(
                keywords.into_iter().map(Into::into).collect(),
            ),
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
        }
    }

    pub fn translate_to(language: impl Into<String>) -> Self {
        Self {
            task: GraniteTask::TranslateTo(language.into()),
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
        }
    }

    pub fn with_max_new_tokens(mut self, n: usize) -> Self {
        self.max_new_tokens = n;
        self
    }
}

impl Default for GraniteOptions {
    fn default() -> Self {
        Self::transcribe_with_punctuation()
    }
}

/// IBM Granite Speech 4.1 base 2b ASR engine.
pub struct Granite {
    model: GraniteArModel,
    tokenizer: Tokenizer,
    audio_token_id: i64,
    eos_token_id: i64,
}

impl Granite {
    /// Load the model from a bundle directory. See
    /// [`model_granite`](crate::model_granite) for the expected layout.
    /// `precision` selects which subdirectory (`fp16w/`, `int8/`,
    /// `fp32/`) to load; defaults to FP16w.
    pub fn from_pretrained<P: AsRef<Path>>(
        bundle_dir: P,
        exec_config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        Self::from_pretrained_with_precision(bundle_dir, GranitePrecision::default(), exec_config)
    }

    /// Same as [`Self::from_pretrained`] but with an explicit precision tier.
    pub fn from_pretrained_with_precision<P: AsRef<Path>>(
        bundle_dir: P,
        precision: GranitePrecision,
        exec_config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        let bundle_dir = bundle_dir.as_ref();
        let exec = exec_config.unwrap_or_default();

        let model = GraniteArModel::from_pretrained(bundle_dir, precision, exec)?;

        let tok_path = bundle_dir.join("tokenizer.json");
        if !tok_path.exists() {
            return Err(Error::Config(format!(
                "Missing tokenizer.json in {}",
                bundle_dir.display()
            )));
        }
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| Error::Tokenizer(format!("Failed to load tokenizer.json: {e}")))?;

        let audio_token_id = require_token(&tokenizer, AUDIO_TOKEN)?;
        let eos_token_id = require_token(&tokenizer, EOS_TOKEN)?;

        Ok(Self {
            model,
            tokenizer,
            audio_token_id,
            eos_token_id,
        })
    }

    /// Sorted (code, name) list of languages this model supports for
    /// transcription and as translation targets.
    pub fn supported_languages(&self) -> &'static [(&'static str, &'static str)] {
        SUPPORTED_LANGUAGES
    }

    /// Run only the encoder + projector and return the raw audio embeddings.
    /// `input_features` is the post-mel + frame-stack tensor produced by
    /// [`Granite::extract_input_features`]; the return tuple is
    /// `(audio_embeds [1, T_audio, 2048], n_valid_audio_tokens)` where
    /// `n_valid_audio_tokens` is `audio_embed_sizes[0]` (number of rows
    /// to splice into the prompt). Useful for caching encoder output
    /// across multiple prompts on the same audio, or for fixture parity
    /// tests against `expected_audio_embeds.npy`.
    pub fn run_encoder(
        &mut self,
        input_features: &ndarray::Array3<f32>,
    ) -> Result<(ndarray::Array3<f32>, usize)> {
        let enc = self.model.run_encoder(input_features)?;
        let n = enc.audio_embed_sizes[0] as usize;
        Ok((enc.audio_embeds, n))
    }

    /// Transcribe (or translate) raw 16 kHz mono float audio.
    ///
    /// The `audio` slice must already be 16 kHz mono. For multi-channel
    /// input, use [`Self::transcribe_samples`] which mixes down for you.
    pub fn transcribe_audio(&mut self, audio: &[f32], options: &GraniteOptions) -> Result<String> {
        if audio.is_empty() {
            return Ok(String::new());
        }

        // 1. Audio frontend: log-mel + 2-frame stack -> [1, T_stacked, 160].
        let input_features = audio_granite::extract_input_features(audio)?;

        // 2. Encoder + projector -> audio_embeds [1, T_audio, 2048],
        //    audio_embed_sizes [1].
        let enc = self.model.run_encoder(&input_features)?;
        let n_audio = enc.audio_embed_sizes[0] as usize;
        if n_audio == 0 {
            return Ok(String::new());
        }
        if enc.audio_embeds.shape()[1] < n_audio {
            return Err(Error::Model(format!(
                "encoder returned {} audio_embeds rows but audio_embed_sizes[0] reports {n_audio}",
                enc.audio_embeds.shape()[1]
            )));
        }

        // 3. Render the chat template manually. Base uses the simple
        //    "USER: {content}\n ASSISTANT:" form documented in
        //    chat_template.jinja.
        let user_prompt = options.task.to_user_prompt();
        let user_prompt_with_audio = expand_audio_token(&user_prompt, n_audio);
        let prompt = format!("USER: {user_prompt_with_audio}\n ASSISTANT:");

        // 4. Tokenise. add_special_tokens=false because the chat
        //    template above already wraps the content; the tokeniser
        //    only adds BOS/EOS otherwise (which would corrupt the
        //    documented prompt format).
        let encoding = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|e| Error::Tokenizer(format!("encode failed: {e}")))?;
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&i| i as i64).collect();
        let n_prompt = ids.len();
        let input_ids = Array2::from_shape_vec((1, n_prompt), ids)
            .map_err(|e| Error::Model(format!("input_ids reshape: {e}")))?;

        // 5. Embed text tokens then splice in audio embeddings at
        //    every <|audio|> position.
        let mut inputs_embeds = self.model.run_embed_tokens(&input_ids)?;
        splice_audio_embeddings(
            &mut inputs_embeds,
            &input_ids,
            &enc.audio_embeds
                .slice(ndarray::s![.., ..n_audio, ..])
                .to_owned(),
            self.audio_token_id,
        )?;

        // 6. Prompt-encode the LLM, then greedy-decode token by token.
        let (initial_logits, kv, prompt_len) = run_prompt(&mut self.model, &inputs_embeds)?;
        let opts = DecodeOptions {
            max_new_tokens: options.max_new_tokens,
            eos_token_id: self.eos_token_id,
        };
        let token_ids = decode_greedy(&mut self.model, initial_logits, kv, prompt_len, opts)?;

        // 7. Detokenise. skip_special_tokens=true strips role markers
        //    if any leak through.
        let text = self
            .tokenizer
            .decode(
                &token_ids.iter().map(|&i| i as u32).collect::<Vec<_>>(),
                true,
            )
            .map_err(|e| Error::Tokenizer(format!("decode failed: {e}")))?;
        Ok(text.trim().to_string())
    }

    /// Transcribe interleaved multi-channel float audio. Mixes to mono
    /// internally and rejects sample rates other than 16 kHz so callers
    /// don't accidentally feed misframed audio.
    pub fn transcribe_samples(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        channels: u16,
        options: &GraniteOptions,
    ) -> Result<String> {
        let mono = audio_granite::prepare_mono_16k(audio, sample_rate, channels)?;
        self.transcribe_audio(&mono, options)
    }

    /// Compute the post-stack `input_features` tensor that
    /// `encoder.onnx` expects for the base / plus variants.
    ///
    /// Static method exposed so tooling and parity tests can compare
    /// against the bundle's `expected_input_features.npy` fixture
    /// without having to load the full ONNX session. Same audio
    /// contract as [`Granite::transcribe_audio`]: 16 kHz mono float.
    pub fn extract_input_features(audio: &[f32]) -> Result<ndarray::Array3<f32>> {
        audio_granite::extract_input_features(audio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_renders_documented_prompts() {
        assert!(GraniteTask::TranscribeRaw
            .to_user_prompt()
            .ends_with("can you transcribe the speech into a written format?"));
        assert!(GraniteTask::TranscribeWithPunctuation
            .to_user_prompt()
            .ends_with("transcribe the speech with proper punctuation and capitalization."));
        assert!(
            GraniteTask::TranscribeWithKeywords(vec!["Sammy".into(), "MFA".into(),])
                .to_user_prompt()
                .ends_with("Keywords: Sammy, MFA")
        );
        assert!(GraniteTask::TranslateTo("French".into())
            .to_user_prompt()
            .ends_with("translate the speech to French."));
    }

    #[test]
    fn audio_token_expands_to_n_copies() {
        let prompt = "<|audio|>transcribe the speech.";
        let expanded = expand_audio_token(prompt, 3);
        assert_eq!(
            expanded,
            "<|audio|><|audio|><|audio|>transcribe the speech."
        );
    }

    #[test]
    fn audio_token_only_replaces_first_placeholder() {
        // Defensive: if a user prompt accidentally includes a literal
        // <|audio|> in keyword text, we only expand the first one
        // (the prompt-template position).
        let prompt = "<|audio|>transcribe with keywords: <|audio|>filler";
        let expanded = expand_audio_token(prompt, 2);
        assert_eq!(
            expanded,
            "<|audio|><|audio|>transcribe with keywords: <|audio|>filler"
        );
    }

    #[test]
    fn options_carry_max_new_tokens() {
        let opts = GraniteOptions::transcribe_raw().with_max_new_tokens(1024);
        assert_eq!(opts.max_new_tokens, 1024);
    }
}
