//! Shared helpers for the Granite Speech 4.1 family.
//!
//! Both autoregressive variants (base + plus) splice a run of
//! `<|audio|>` placeholders into the prompt before tokenisation; all
//! three variants resolve a fixed set of literal token IDs at load
//! time. These helpers live here so the engine modules don't have to
//! re-implement them.

use crate::error::{Error, Result};
use tokenizers::Tokenizer;

/// Audio placeholder token literal in Granite Speech 4.1 base / plus.
/// Expanded at prompt-construction time to a run of repeats whose
/// length matches the projector output.
pub(crate) const AUDIO_TOKEN: &str = "<|audio|>";

/// Replace the literal `<|audio|>` placeholder in `text` with `n` copies
/// of the audio token. This must be done as a string replace BEFORE
/// tokenisation, not after, because `<|audio|>` is a single special
/// token in the BPE vocab and will otherwise be tokenised as exactly
/// one ID regardless of how many copies you "want".
#[allow(dead_code)] // unused on `granite-nar` only builds
pub(crate) fn expand_audio_token(text: &str, n: usize) -> String {
    let repeats = AUDIO_TOKEN.repeat(n);
    text.replacen(AUDIO_TOKEN, &repeats, 1)
}

/// Resolve a literal token to its ID, returning a clear error if the
/// tokenizer is missing it. Used at engine load time.
pub(crate) fn require_token(tokenizer: &Tokenizer, literal: &str) -> Result<i64> {
    tokenizer
        .token_to_id(literal)
        .map(|id| id as i64)
        .ok_or_else(|| Error::Tokenizer(format!("Tokenizer is missing required token {literal}")))
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn expand_no_op_when_placeholder_absent() {
        let expanded = expand_audio_token("no placeholder here", 4);
        assert_eq!(expanded, "no placeholder here");
    }
}
