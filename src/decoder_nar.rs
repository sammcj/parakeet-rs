//! Host-side glue for the Granite Speech 4.1 NAR pipeline.
//!
//! Three steps live here:
//!
//! 1. [`ctc_greedy_decode`] takes the encoder's `bpe_logits_dense`
//!    (`[1, T_bpe, V_bpe]`) plus `bpe_mask` (`[1, T_bpe]`), runs greedy
//!    argmax, collapses consecutive duplicates, drops blanks, and shifts
//!    the surviving indices down by one to recover LLM token IDs.
//! 2. [`add_insertion_slots`] interleaves `eos_id` between every CTC
//!    draft token (and at the boundaries) so the editor has a fixed
//!    insertion slot to rewrite or expand each span. Output length is
//!    `max(2*n + 1, 8)`.
//! 3. [`argmax_text_segment`] argmaxes the editor's `logits[:, audio_len:, :]`
//!    text segment to recover final LLM token IDs.
//!
//! These mirror the IBM reference in `modeling_nle.py`
//! (`_decode_bpe_ctc_greedy`, `add_insertion_slots`,
//! `_build_flat_llm_inputs`) byte-for-byte modulo running through the
//! exported ONNX graphs instead of PyTorch modules.

use crate::error::{Error, Result};
use ndarray::{Array2, Array3};

/// CTC blank index in the BPE vocabulary. The BPE label mapping is:
/// index 0 = blank, index `i` = LLM token id `i - 1`. Blanks are
/// dropped after consecutive-duplicate collapse and the remaining
/// indices are shifted down by 1 to recover LLM token IDs.
pub(crate) const CTC_BLANK_INDEX: i64 = 0;

/// Greedy CTC decode of the encoder's BPE head. `bpe_logits_dense` is
/// `[1, T_bpe, V_bpe]`; `bpe_mask` is `[1, T_bpe]` with `1.0` on real
/// positions and `0.0` on padding. Returns the collapsed, blank-free
/// LLM token IDs (already mapped: BPE index `i` -> LLM id `i - 1`).
/// Single-batch only - parakeet-rs does not currently batch utterances.
pub(crate) fn ctc_greedy_decode(
    bpe_logits_dense: &Array3<f32>,
    bpe_mask: &Array2<f32>,
) -> Result<Vec<i64>> {
    if bpe_logits_dense.shape()[0] != 1 || bpe_mask.shape()[0] != 1 {
        return Err(Error::Model(
            "ctc_greedy_decode only supports batch=1".into(),
        ));
    }
    let t_bpe = bpe_logits_dense.shape()[1];
    if bpe_mask.shape()[1] != t_bpe {
        return Err(Error::Model(format!(
            "bpe_mask T_bpe={} != bpe_logits_dense T_bpe={}",
            bpe_mask.shape()[1],
            t_bpe
        )));
    }
    let v_bpe = bpe_logits_dense.shape()[2];

    let mut out: Vec<i64> = Vec::new();
    let mut prev: i64 = -1;
    for t in 0..t_bpe {
        if bpe_mask[[0, t]] < 0.5 {
            continue;
        }
        let mut best_idx: i64 = 0;
        let mut best_val = f32::NEG_INFINITY;
        for v in 0..v_bpe {
            let x = bpe_logits_dense[[0, t, v]];
            if x > best_val {
                best_val = x;
                best_idx = v as i64;
            }
        }
        if best_idx == prev {
            continue;
        }
        prev = best_idx;
        if best_idx != CTC_BLANK_INDEX {
            out.push(best_idx - 1);
        }
    }
    Ok(out)
}

/// Interleave `eos_id` between every CTC draft token and at both
/// boundaries, padding to `max(2 * n + 1, 8)` total positions. With
/// `n = 0` the output is a run of 8 `eos_id`s; with `n = 1` it is
/// `[eos, t0, eos]` padded to 8. The editor learns to rewrite or
/// expand each `eos` slot, so the slot count must match the per-span
/// granularity it was trained on.
///
/// Mirrors `add_insertion_slots(t, eos_id)` from the IBM Python
/// reference exactly.
pub(crate) fn add_insertion_slots(tokens: &[i64], eos_id: i64) -> Vec<i64> {
    let n = tokens.len();
    let total_len = (2 * n + 1).max(8);
    let mut out = vec![eos_id; total_len];
    for (i, &tok) in tokens.iter().enumerate() {
        out[2 * i + 1] = tok;
    }
    out
}

/// Argmax the editor's text segment - the rows after the audio prefix -
/// to recover final LLM token IDs. `logits` is `[1, N, V_LLM]` where
/// `N = audio_len + slots_len`. Returns the `slots_len` argmax IDs.
pub(crate) fn argmax_text_segment(logits: &Array3<f32>, audio_len: usize) -> Result<Vec<i64>> {
    if logits.shape()[0] != 1 {
        return Err(Error::Model(
            "argmax_text_segment only supports batch=1".into(),
        ));
    }
    let n = logits.shape()[1];
    let v = logits.shape()[2];
    if audio_len > n {
        return Err(Error::Model(format!(
            "argmax_text_segment: audio_len {audio_len} > total positions {n}"
        )));
    }
    let mut out = Vec::with_capacity(n - audio_len);
    for t in audio_len..n {
        let mut best_idx: i64 = 0;
        let mut best_val = f32::NEG_INFINITY;
        for k in 0..v {
            let x = logits[[0, t, k]];
            if x > best_val {
                best_val = x;
                best_idx = k as i64;
            }
        }
        out.push(best_idx);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn add_insertion_slots_short_input_pads_to_eight() {
        // n = 0 -> all eos
        let r = add_insertion_slots(&[], 99);
        assert_eq!(r, vec![99; 8]);
        // n = 1 -> [eos, t0, eos] padded to 8
        let r = add_insertion_slots(&[7], 99);
        assert_eq!(r.len(), 8);
        assert_eq!(r[0], 99);
        assert_eq!(r[1], 7);
        assert_eq!(r[2], 99);
        for &x in &r[3..] {
            assert_eq!(x, 99);
        }
    }

    #[test]
    fn add_insertion_slots_uses_2n_plus_1_for_long_input() {
        let toks = vec![1, 2, 3, 4, 5];
        let r = add_insertion_slots(&toks, 99);
        assert_eq!(r.len(), 11); // 2*5 + 1
        let expected = [99, 1, 99, 2, 99, 3, 99, 4, 99, 5, 99];
        assert_eq!(r, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn ctc_greedy_collapses_duplicates_and_drops_blanks() {
        // T_bpe=6, V_bpe=4. Sequence: [1, 1, 0, 2, 0, 3] all mask=1.
        // Collapse consecutive: [1, 0, 2, 0, 3]. Drop blanks (0): [1, 2, 3].
        // Shift down by 1: [0, 1, 2].
        let logits = Array::from_shape_vec((1, 6, 4), vec![
            0.0, 9.0, 0.0, 0.0,
            0.0, 9.0, 0.0, 0.0,
            9.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 9.0, 0.0,
            9.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 9.0,
        ]).unwrap();
        let mask = Array::from_shape_vec((1, 6), vec![1.0; 6]).unwrap();
        let out = ctc_greedy_decode(&logits, &mask).unwrap();
        assert_eq!(out, vec![0, 1, 2]);
    }

    #[rustfmt::skip]
    #[test]
    fn ctc_greedy_skips_padded_positions() {
        // T_bpe=4, V_bpe=3. argmax sequence would be [1, 2, 1, 2].
        // mask = [1, 1, 0, 0] - only first two count.
        // Collapse: [1, 2]. Shift: [0, 1].
        let logits = Array::from_shape_vec((1, 4, 3), vec![
            0.0, 9.0, 0.0,
            0.0, 0.0, 9.0,
            0.0, 9.0, 0.0,
            0.0, 0.0, 9.0,
        ]).unwrap();
        let mask = Array::from_shape_vec((1, 4), vec![1.0, 1.0, 0.0, 0.0]).unwrap();
        let out = ctc_greedy_decode(&logits, &mask).unwrap();
        assert_eq!(out, vec![0, 1]);
    }

    #[rustfmt::skip]
    #[test]
    fn argmax_text_segment_skips_audio_prefix() {
        // N=4, V=3. audio_len=2. argmax of last two positions only.
        let logits = Array::from_shape_vec((1, 4, 3), vec![
            9.0, 0.0, 0.0, // audio
            0.0, 9.0, 0.0, // audio
            0.0, 0.0, 9.0, // text -> 2
            0.0, 9.0, 0.0, // text -> 1
        ]).unwrap();
        let out = argmax_text_segment(&logits, 2).unwrap();
        assert_eq!(out, vec![2, 1]);
    }
}
