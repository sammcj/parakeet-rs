//! Shared decoder helpers used by the autoregressive engines (Cohere
//! Transcribe, Granite Speech 4.1 base / plus). Lives at crate-internal
//! scope; engines call these free functions directly. No traits, no
//! abstractions.

/// Greedy argmax over a slice of `f32` logits. Returns the index of the
/// largest element, with the first occurrence winning on ties.
pub(crate) fn argmax(logits: &[f32]) -> i64 {
    let mut best_idx: i64 = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as i64;
        }
    }
    best_idx
}

/// Detect whether the tail of `tokens` repeats an n-gram of length
/// `>= min_len`. Returns `Some(repeat_len)` if the last `repeat_len`
/// tokens are an exact copy of the preceding segment. Used by the
/// autoregressive decoders to short-circuit runaway repetition loops.
pub(crate) fn find_ngram_repetition(tokens: &[i64], min_len: usize) -> Option<usize> {
    let n = tokens.len();
    if n < min_len * 2 {
        return None;
    }
    for repeat_len in min_len..=(n / 2) {
        let tail = &tokens[n - repeat_len..];
        let prev = &tokens[n - 2 * repeat_len..n - repeat_len];
        if tail == prev {
            return Some(repeat_len);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_picks_first_largest() {
        assert_eq!(argmax(&[0.1, 0.5, 0.3, 0.9, 0.2]), 3);
        assert_eq!(argmax(&[1.0, 0.0, 0.0]), 0);
        assert_eq!(argmax(&[0.5, 0.5]), 0);
    }

    #[test]
    fn ngram_detection() {
        assert_eq!(find_ngram_repetition(&[1, 2, 3, 4, 5, 6, 7, 8], 4), None);
        assert_eq!(find_ngram_repetition(&[1, 2, 3, 4, 1, 2, 3, 4], 4), Some(4));
        assert_eq!(find_ngram_repetition(&[1, 2, 1, 2], 4), None);
        let tokens = [9_i64, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(find_ngram_repetition(&tokens, 8), Some(8));
    }
}
