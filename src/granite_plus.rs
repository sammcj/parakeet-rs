//! IBM Granite Speech 4.1 plus 2b ASR engine.
//!
//! Same architecture as the base 2b model, retrained for richer outputs.
//! Drops Japanese and translation; adds three features that all activate
//! via prompt change:
//!
//! - **Speaker-attributed ASR (SAA)**: model emits inline `[Speaker N]:`
//!   tags before each speaker turn, numbered by order of appearance.
//! - **Word-level timestamps**: model emits `[T:N]` after each word
//!   where `N = round(t*100) mod 1000` (centiseconds, last three
//!   digits, rolling over every 10 seconds). Silences are `_`.
//! - **Incremental decoding**: callers can pass `prefix_text` (a
//!   previously decoded segment) so the model continues from there.
//!   Used to keep speaker numbering stable across chunked long audio.
//!
//! Tag parsing is purely Rust-side: the LLM emits the tags as ordinary
//! BPE tokens; we lift them out of the decoded string and into typed
//! [`SpeakerSegment`] / [`TimedWord`] fields. The rollover counter is
//! held on the wrapper across calls so true wall-clock time can be
//! reconstructed for audio longer than 10 seconds.
//!
//! Per the plus model card, the model was trained on audio up to 9
//! minutes long for ASR and SAA, and up to 5 minutes for timestamps.
//! For longer inputs split into overlapping chunks and use
//! `prefix_text` to preserve speaker numbering.
//!
//! The plus model **does not produce punctuation or capitalisation**;
//! that's a deliberate trade-off for the structural features. If you
//! need punctuation, use the base 2b model via [`Granite`](crate::Granite).

use crate::audio_granite;
use crate::decoder_granite::{decode_greedy, run_prompt, DecodeOptions};
use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::granite_common::{expand_audio_token, require_token, AUDIO_TOKEN};
use crate::granite_plus_tags::TagParserState;
use crate::model_granite::{splice_audio_embeddings, GraniteArModel, GranitePrecision};
use ndarray::Array2;
use std::path::Path;
use tokenizers::Tokenizer;

const EOS_TOKEN: &str = "<|end_of_text|>";

/// IBM-supplied system prompt for plus 2b. The plus model card lists
/// this verbatim and warns that omitting it degrades quality.
const SYSTEM_PROMPT: &str = "Knowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant";

const DEFAULT_MAX_NEW_TOKENS: usize = 2000;
/// Higher cap for timestamp mode: every word emits an extra `[T:N]`
/// tag, so transcripts run a few thousand tokens for long audio.
const TIMESTAMP_MAX_NEW_TOKENS: usize = 10_000;

/// Task selection for plus. Mirrors the prompts documented in the plus
/// model card (<https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus>).
#[derive(Debug, Clone)]
pub enum GranitePlusTask {
    /// Plain ASR (no punctuation, no speaker tags, no timestamps).
    TranscribeRaw,
    /// Add inline `[Speaker N]:` markers before each speaker turn.
    SpeakerAttributed,
    /// Emit a `[T:N]` tag after every word giving the end-of-word time
    /// in centiseconds (mod 1000).
    WordTimestamps,
    /// ASR with a keyword bias list. Plus accepts the same
    /// `Keywords: ...` suffix as base.
    TranscribeWithKeywords(Vec<String>),
}

impl GranitePlusTask {
    fn to_user_prompt(&self) -> String {
        match self {
            GranitePlusTask::TranscribeRaw => format!(
                "{AUDIO_TOKEN} can you transcribe the speech into a written format?"
            ),
            GranitePlusTask::SpeakerAttributed => format!(
                "{AUDIO_TOKEN} Speaker attribution: Transcribe and denote who is speaking by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns."
            ),
            GranitePlusTask::WordTimestamps => format!(
                "{AUDIO_TOKEN} Timestamps: Transcribe the speech. After each word, add a timestamp tag showing the end time in centiseconds, e.g. hello [T:45] world [T:82]"
            ),
            GranitePlusTask::TranscribeWithKeywords(kws) => {
                let kws = kws.join(", ");
                format!(
                    "{AUDIO_TOKEN} Can you transcribe the speech into a written format? Keywords: {kws}"
                )
            }
        }
    }
}

/// User-facing options for one plus transcription call.
#[derive(Debug, Clone)]
pub struct GranitePlusOptions {
    pub task: GranitePlusTask,
    pub max_new_tokens: usize,
    /// Previously transcribed text for incremental decoding. The model
    /// continues generating after this prefix and uses it as context
    /// for speaker numbering / language consistency.
    pub prefix_text: Option<String>,
}

impl GranitePlusOptions {
    pub fn transcribe_raw() -> Self {
        Self {
            task: GranitePlusTask::TranscribeRaw,
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            prefix_text: None,
        }
    }
    pub fn speaker_attributed() -> Self {
        Self {
            task: GranitePlusTask::SpeakerAttributed,
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            prefix_text: None,
        }
    }
    pub fn word_timestamps() -> Self {
        Self {
            task: GranitePlusTask::WordTimestamps,
            max_new_tokens: TIMESTAMP_MAX_NEW_TOKENS,
            prefix_text: None,
        }
    }
    pub fn transcribe_with_keywords(keywords: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            task: GranitePlusTask::TranscribeWithKeywords(
                keywords.into_iter().map(Into::into).collect(),
            ),
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            prefix_text: None,
        }
    }

    /// Provide a previously decoded transcript to continue from. Used
    /// for chunked long-audio inference: pass the prior chunk's
    /// transcript here so the model keeps speaker numbering stable.
    /// Trailing whitespace is trimmed automatically before the prompt
    /// is built; otherwise BPE merges the trailing space into the
    /// model's first generated token and the continuation decodes
    /// truncated by one character.
    pub fn with_prefix_text(mut self, prefix: impl Into<String>) -> Self {
        self.prefix_text = Some(prefix.into());
        self
    }

    pub fn with_max_new_tokens(mut self, n: usize) -> Self {
        self.max_new_tokens = n;
        self
    }
}

impl Default for GranitePlusOptions {
    fn default() -> Self {
        Self::transcribe_raw()
    }
}

/// Speaker-tagged segment lifted from the model's inline `[Speaker N]:`
/// markers. `speaker` is `1`, `2`, ... numbered in order of first
/// appearance. `text` excludes the marker itself.
#[derive(Debug, Clone)]
pub struct SpeakerSegment {
    pub speaker: u32,
    pub text: String,
}

/// One word with an end-of-word timestamp in seconds, reconstructed
/// from a `[T:N]` tag plus the rollover counter.
#[derive(Debug, Clone)]
pub struct TimedWord {
    pub word: String,
    /// End time in seconds from the start of the audio.
    pub end_time: f32,
    /// True if this is a `_` silence marker rather than a transcribed
    /// word. Useful when surfacing speech / non-speech regions.
    pub is_silence: bool,
}

/// Output of one [`GranitePlus::transcribe_audio`] call. `text` is the
/// full transcript with structural tags stripped (so it reads
/// naturally). `segments` is populated when SAA mode is requested;
/// `words` is populated when timestamp mode is requested. They are
/// independent: timestamp mode does not produce speaker segments and
/// vice versa.
#[derive(Debug, Clone)]
pub struct GranitePlusResult {
    pub text: String,
    pub raw_text: String,
    pub segments: Vec<SpeakerSegment>,
    pub words: Vec<TimedWord>,
}

/// IBM Granite Speech 4.1 plus 2b ASR engine.
pub struct GranitePlus {
    model: GraniteArModel,
    tokenizer: Tokenizer,
    audio_token_id: i64,
    eos_token_id: i64,
    /// Persistent timestamp parser state held across calls so timestamp
    /// rollover works for chunked long audio. Reset via
    /// [`reset_rollover`].
    parser_state: TagParserState,
}

impl GranitePlus {
    pub fn from_pretrained<P: AsRef<Path>>(
        bundle_dir: P,
        exec_config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        Self::from_pretrained_with_precision(bundle_dir, GranitePrecision::default(), exec_config)
    }

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
            parser_state: TagParserState::default(),
        })
    }

    /// Reset the persistent rollover counter. Call this between
    /// unrelated audio sources so timestamp rollover doesn't bleed
    /// across files.
    pub fn reset_rollover(&mut self) {
        self.parser_state = TagParserState::default();
    }

    /// Run only the encoder + projector and return the raw audio embeddings.
    /// `input_features` is the post-mel + frame-stack tensor produced by
    /// [`crate::Granite::extract_input_features`]; the return tuple is
    /// `(audio_embeds [1, T_audio, 2048], n_valid_audio_tokens)`. Same
    /// shape as the base variant; useful for caching encoder output and
    /// for fixture parity tests.
    pub fn run_encoder(
        &mut self,
        input_features: &ndarray::Array3<f32>,
    ) -> Result<(ndarray::Array3<f32>, usize)> {
        let enc = self.model.run_encoder(input_features)?;
        let n = enc.audio_embed_sizes[0] as usize;
        Ok((enc.audio_embeds, n))
    }

    pub fn transcribe_audio(
        &mut self,
        audio: &[f32],
        options: &GranitePlusOptions,
    ) -> Result<GranitePlusResult> {
        if audio.is_empty() {
            return Ok(empty_result());
        }

        let input_features = audio_granite::extract_input_features(audio)?;
        let enc = self.model.run_encoder(&input_features)?;
        let n_audio = enc.audio_embed_sizes[0] as usize;
        if n_audio == 0 {
            return Ok(empty_result());
        }

        // Plus uses the Granite 4 chat template (with role markers)
        // and the IBM-supplied system prompt. Render manually to
        // avoid pulling in a Jinja engine.
        let user_prompt = options.task.to_user_prompt();
        let user_prompt_with_audio = expand_audio_token(&user_prompt, n_audio);
        let prefix_owned = sanitise_prefix(options.prefix_text.as_deref());
        let prefix = prefix_owned.as_str();
        let prompt = format!(
            "<|start_of_role|>system<|end_of_role|>{SYSTEM_PROMPT}<|end_of_text|>\n\
             <|start_of_role|>user<|end_of_role|>{user_prompt_with_audio}<|end_of_text|>\n\
             <|start_of_role|>assistant<|end_of_role|>{prefix}"
        );

        let encoding = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|e| Error::Tokenizer(format!("encode failed: {e}")))?;
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&i| i as i64).collect();
        let n_prompt = ids.len();
        let input_ids = Array2::from_shape_vec((1, n_prompt), ids)
            .map_err(|e| Error::Model(format!("input_ids reshape: {e}")))?;

        let mut inputs_embeds = self.model.run_embed_tokens(&input_ids)?;
        splice_audio_embeddings(
            &mut inputs_embeds,
            &input_ids,
            &enc.audio_embeds
                .slice(ndarray::s![.., ..n_audio, ..])
                .to_owned(),
            self.audio_token_id,
        )?;

        let (initial_logits, kv, prompt_len) = run_prompt(&mut self.model, &inputs_embeds)?;
        let opts = DecodeOptions {
            max_new_tokens: options.max_new_tokens,
            eos_token_id: self.eos_token_id,
        };
        let token_ids = decode_greedy(&mut self.model, initial_logits, kv, prompt_len, opts)?;

        let raw_text = self
            .tokenizer
            .decode(
                &token_ids.iter().map(|&i| i as u32).collect::<Vec<_>>(),
                true,
            )
            .map_err(|e| Error::Tokenizer(format!("decode failed: {e}")))?;
        let raw_text = raw_text.trim().to_string();
        let (clean_text, segments, words) = self.parser_state.parse(&raw_text);

        Ok(GranitePlusResult {
            text: clean_text,
            raw_text,
            segments,
            words,
        })
    }

    /// Multichannel-friendly variant: mixes to mono and rejects
    /// sample rates other than 16 kHz before transcribing.
    pub fn transcribe_samples(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        channels: u16,
        options: &GranitePlusOptions,
    ) -> Result<GranitePlusResult> {
        let mono = audio_granite::prepare_mono_16k(audio, sample_rate, channels)?;
        self.transcribe_audio(&mono, options)
    }
}

fn empty_result() -> GranitePlusResult {
    GranitePlusResult {
        text: String::new(),
        raw_text: String::new(),
        segments: Vec::new(),
        words: Vec::new(),
    }
}

/// Trim trailing whitespace from a `prefix_text` value before it is
/// injected into the assistant role. A trailing space (the common shape
/// when callers concatenate prefixes by hand) collides with BPE merge
/// boundaries: the model's first generated token ends up sharing a
/// merge with the prompt's trailing space token, and the continuation
/// decodes with its leading character absorbed - e.g. "...stretched, "
/// continued by `first` surfaces as `irst`. Trimming pins the boundary
/// at the previous real character and the continuation decodes intact.
/// Matches how the IBM reference is intended to be called: prefix_text
/// is the model's own prior output, which never has a trailing space.
fn sanitise_prefix(prefix: Option<&str>) -> String {
    match prefix {
        Some(p) => p.trim_end().to_string(),
        None => String::new(),
    }
}

/// Tag-parser state held by [`GranitePlus`] (and exposed to tests).
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitise_prefix_strips_trailing_whitespace() {
        // None and empty are passthrough.
        assert_eq!(sanitise_prefix(None), "");
        assert_eq!(sanitise_prefix(Some("")), "");
        // Trailing whitespace is removed (regression test: a trailing
        // space caused the model's first generated token to decode with
        // its leading char absorbed into the prompt's space token).
        assert_eq!(sanitise_prefix(Some("hello ")), "hello");
        assert_eq!(sanitise_prefix(Some("After his nap, ")), "After his nap,");
        assert_eq!(sanitise_prefix(Some("foo\t\n  ")), "foo");
        // Leading whitespace and inner whitespace are preserved.
        assert_eq!(sanitise_prefix(Some(" hello world")), " hello world");
        assert_eq!(sanitise_prefix(Some("clean")), "clean");
    }

    #[test]
    fn task_renders_documented_prompts() {
        assert!(GranitePlusTask::TranscribeRaw
            .to_user_prompt()
            .contains("can you transcribe the speech into a written format"));
        assert!(GranitePlusTask::SpeakerAttributed
            .to_user_prompt()
            .contains("Speaker attribution"));
        assert!(GranitePlusTask::WordTimestamps
            .to_user_prompt()
            .contains("Timestamps:"));
        assert!(GranitePlusTask::TranscribeWithKeywords(vec!["IBM".into()])
            .to_user_prompt()
            .contains("Keywords: IBM"));
    }

    #[test]
    fn audio_token_expands_to_n_copies() {
        let prompt = "<|audio|> Speaker attribution: ...";
        let expanded = expand_audio_token(prompt, 3);
        assert_eq!(
            expanded,
            "<|audio|><|audio|><|audio|> Speaker attribution: ..."
        );
    }

    #[test]
    fn options_carry_max_new_tokens() {
        let opts = GranitePlusOptions::word_timestamps().with_max_new_tokens(50_000);
        assert_eq!(opts.max_new_tokens, 50_000);
    }

    #[test]
    fn options_carry_prefix_text() {
        let opts =
            GranitePlusOptions::speaker_attributed().with_prefix_text("[Speaker 1]: previous text");
        assert_eq!(
            opts.prefix_text.as_deref(),
            Some("[Speaker 1]: previous text")
        );
    }
}
