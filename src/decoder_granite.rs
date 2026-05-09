//! Autoregressive greedy decode loop for Granite Speech 4.1 base / plus.
//!
//! Owns the inner generation loop: embed-the-next-token, run one
//! `decode_step`, argmax, repeat. Exposed at crate-internal scope; the
//! variant-specific wrappers ([`granite`](crate::granite),
//! [`granite_plus`](crate::granite_plus)) drive it after building the
//! initial prompt, encoder output, and `inputs_embeds` splice.
//!
//! The decode loop is variant-agnostic: it doesn't care whether the
//! caller is base or plus. Plus's structural tags
//! (`[Speaker N]:`, `[T:N]`) are emitted as ordinary BPE tokens and
//! show up in the decoded text; the caller parses them post-hoc.

use crate::decode_util::{argmax, find_ngram_repetition};
use crate::error::{Error, Result};
use crate::model_granite::{build_decode_mask, GraniteArModel, KvCache};
use ndarray::{Array1, Array2, Array3};

/// Hard upper bound on tokens generated per call. The Granite 4 LLM
/// supports 128k context, but for ASR-style outputs anything beyond a
/// few hundred tokens almost always indicates a runaway repetition
/// loop. The wrapper crate lets users tune this within `[1, 4096]`.
pub(crate) const MAX_DECODE_TOKENS_LIMIT: usize = 4096;

/// Decoder configuration. The wrapper structs own a copy of this; the
/// decode function is passed `&DecodeOptions` rather than reading from
/// model state so tests can construct it without a live session.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DecodeOptions {
    pub max_new_tokens: usize,
    pub eos_token_id: i64,
}

/// Greedy decode after the prompt has already been encoded. Continues
/// stepping until EOS, max-new-tokens, or n-gram repetition is hit.
///
/// Inputs:
/// - `model`: the live `GraniteArModel` for embedding and stepping.
/// - `initial_logits`: logits for the *first* generated position
///   (i.e. the output of `prompt_encode` at its last position).
/// - `kv`: KV cache returned from `prompt_encode`.
/// - `prompt_len`: number of prompt tokens that have already been
///   consumed; new `position_ids` start from here.
/// - `opts`: decode configuration.
///
/// Returns the generated token IDs (excluding EOS).
pub(crate) fn decode_greedy(
    model: &mut GraniteArModel,
    initial_logits: Array1<f32>,
    kv: KvCache,
    prompt_len: usize,
    opts: DecodeOptions,
) -> Result<Vec<i64>> {
    let mut output: Vec<i64> = Vec::new();
    let mut past_kv = kv;

    let mut next_token = argmax(initial_logits.as_slice().unwrap());
    if next_token == opts.eos_token_id {
        return Ok(output);
    }
    output.push(next_token);

    let max_iters = opts.max_new_tokens.min(MAX_DECODE_TOKENS_LIMIT);
    for _ in 1..max_iters {
        // 1. Embed the just-emitted token through the LLM input table.
        let token_ids = Array2::from_shape_vec((1, 1), vec![next_token])
            .map_err(|e| Error::Model(format!("token id reshape: {e}")))?;
        let embed = model.run_embed_tokens(&token_ids)?; // [1, 1, HIDDEN_SIZE]

        // 2. Build position_ids and attention mask.
        let pos_idx = (prompt_len + output.len() - 1) as i64;
        let position_ids = Array2::from_shape_vec((1, 1), vec![pos_idx])
            .map_err(|e| Error::Model(format!("position_ids reshape: {e}")))?;
        let t_total = past_kv.past_len() + 1;
        let mask = build_decode_mask(t_total);

        // 3. Step the LLM.
        let (logits, new_kv) = model.run_decode_step(&embed, &position_ids, &mask, &past_kv)?;
        past_kv = new_kv;

        next_token = argmax(logits.as_slice().unwrap());
        if next_token == opts.eos_token_id {
            break;
        }
        output.push(next_token);

        // 4. Stop on detected repetition (and discard the repeated
        // tail so we don't include it in the final transcript).
        if let Some(repeat_len) = find_ngram_repetition(&output, 8) {
            output.truncate(output.len() - repeat_len);
            break;
        }
    }
    Ok(output)
}

/// Wrapper around the prompt-encode step that builds the inputs the
/// model graph expects: `position_ids` `[1, N]` and the 4-D causal mask
/// `[1, 1, N, N]`. Returns the same `(initial_logits, kv)` tuple
/// suitable for passing straight into [`decode_greedy`].
pub(crate) fn run_prompt(
    model: &mut GraniteArModel,
    inputs_embeds: &Array3<f32>,
) -> Result<(Array1<f32>, KvCache, usize)> {
    let n = inputs_embeds.shape()[1];
    let position_ids = Array2::from_shape_vec((1, n), (0..n as i64).collect())
        .map_err(|e| Error::Model(format!("position_ids reshape: {e}")))?;
    let mask = crate::model_granite::build_causal_mask(n);
    let (logits, kv) = model.run_prompt_encode(inputs_embeds, &position_ids, &mask)?;
    Ok((logits, kv, n))
}

