//! Audio frontend for IBM Granite Speech 4.1 (base, plus, and NAR).
//!
//! This is a direct port of the canonical Python feature extractors:
//! - `GraniteSpeechFeatureExtractor` (transformers, used by base + plus)
//! - `feature_extraction_nle.NLEFeatureExtractor` (NAR)
//!
//! Both pipelines share the same numerical recipe, which differs from the
//! NeMo/Cohere frontends in `audio.rs`:
//!
//! - `torchaudio.MelSpectrogram` defaults: HTK mel scale, no filterbank
//!   normalisation, power spectrogram (`power = 2.0`), `center = True`,
//!   `pad_mode = "reflect"`, Hann window of `win_length` zero-padded into
//!   `n_fft` if `win_length < n_fft`.
//! - log10, not natural log, with a `1e-10` clamp floor.
//! - Per-utterance global-max normalisation:
//!   `logmel = max(logmel, mx - 8.0); logmel = logmel / 4 + 1` where `mx`
//!   is the maximum over both time and feature axes for each item in the
//!   batch.
//! - 2-frame stack along the time axis: the trailing 80 mels of frame `t+1`
//!   are concatenated to the 80 mels of frame `t`, halving the time axis
//!   and producing 160 features per frame.
//!
//! Variant differences:
//!
//! - **Base / plus** (`GraniteSpeechFeatureExtractor`): mel is computed on
//!   the full audio. If the resulting mel-frame count is odd, the last
//!   frame is dropped before frame stacking. Emits `input_features` only;
//!   `audio_embed_sizes` and `input_features_mask` are derived elsewhere.
//! - **NAR** (`NLEFeatureExtractor`): the mel sequence is truncated to
//!   `l = 2 * (T // (2 * hop_length))` frames *before* the log /
//!   normalisation step, so encoder frame count is `T // (2*hop) =
//!   T // 320`. Emits both `input_features` and an `attention_mask` at
//!   the post-stack rate.
//!
//! The mel filter bank is built once per call from the closed-form HTK
//! frequency mapping, mirroring `torchaudio.functional.melscale_fbanks`
//! with `norm=None` and `mel_scale="htk"`.
//!
//! Parity expectation: byte-exact (within FP32 tolerance) to the Python
//! reference for any 16 kHz mono float waveform. The intent is that the
//! same `input_features` tensor produces identical encoder output between
//! the upstream PyTorch implementation and parakeet-rs.

// Some helpers in this module are only consumed by the NAR variant
// (gated behind `granite-nar`) or only by base/plus (`granite` /
// `granite-plus`). Allow dead-code at the module level so the same
// frontend file can serve all three feature flag combinations without
// per-cfg gating noise.
#![allow(dead_code)]

use crate::audio::{create_mel_filterbank, stft, MelNorm, MelScale, PadMode, WindowMode};
use crate::error::{Error, Result};
#[cfg(test)]
use ndarray::Array1;
use ndarray::{Array2, Array3};

/// Defaults from `processor_config.json` / `preprocessor_config.json` of
/// every Granite Speech 4.1 bundle. The encoder graph hard-bakes these,
/// so they are not user-configurable.
pub const SAMPLE_RATE: usize = 16_000;
pub const N_FFT: usize = 512;
pub const WIN_LENGTH: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const N_MELS: usize = 80;
/// 2 raw mel frames concatenated -> 160 features per encoder frame.
pub const STACKED_FEATURES: usize = N_MELS * 2;

/// Power-spectrogram STFT matching `torchaudio.transforms.Spectrogram`
/// with `power=2.0, center=True, pad_mode="reflect"`. Thin wrapper over
/// [`crate::audio::stft`] with the Granite Speech variant flags.
fn stft_power(audio: &[f32]) -> Result<Array2<f32>> {
    stft(
        audio,
        N_FFT,
        HOP_LENGTH,
        WIN_LENGTH,
        PadMode::Reflect,
        WindowMode::Periodic,
    )
}

/// HTK mel filter bank in the shape `[N_MELS, freq_bins]`, no
/// normalisation. Mirrors `torchaudio.functional.melscale_fbanks(
/// n_freqs=freq_bins, f_min=0.0, f_max=sample_rate/2, n_mels=N_MELS,
/// sample_rate=SAMPLE_RATE, norm=None, mel_scale="htk")`.
fn htk_mel_filterbank() -> Array2<f32> {
    create_mel_filterbank(N_FFT, N_MELS, SAMPLE_RATE, MelScale::Htk, MelNorm::None)
}

/// Final stage of the Granite frontend: log10, max-8 floor,
/// `/4 + 1` rescaling, optional last-frame drop, 2-frame stack.
///
/// Input: `mel` of shape `[T_mel, N_MELS]` (already mel-projected, raw
/// power-spectrogram domain values). Output: `[T_mel/2, 160]` stacked
/// log-mel features ready to feed the encoder.
fn finalise_log_mel_and_stack(mut mel: Array2<f32>) -> Array2<f32> {
    // log10 with 1e-10 floor, in place via mapv. The Python reference
    // uses `clip_(min=1e-10).log10_()` which mutates in place, but the
    // numerical effect is the same.
    mel.mapv_inplace(|x| {
        let v = x.max(1e-10);
        v.log10()
    });

    // Per-utterance global max -> max - 8 floor -> /4 + 1.
    let mx = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let floor = mx - 8.0;
    mel.mapv_inplace(|x| (x.max(floor)) / 4.0 + 1.0);

    // Drop trailing odd frame so the time axis is even.
    let t_mel = mel.shape()[0];
    if t_mel % 2 == 1 {
        // remove last row
        mel = mel.slice(ndarray::s![..t_mel - 1, ..]).to_owned();
    }
    let t_mel = mel.shape()[0];
    let t_stacked = t_mel / 2;

    // Reshape to stack: [T_mel, 80] -> [T_stacked, 160] in row-major.
    // We reuse the underlying buffer; ndarray's reshape needs C-contig.
    let mel = mel.as_standard_layout().to_owned();
    let buf = mel.into_raw_vec_and_offset().0;
    Array2::from_shape_vec((t_stacked, STACKED_FEATURES), buf)
        .expect("frame-stack reshape always valid: T_mel is even and N_MELS*2 == STACKED_FEATURES")
}

/// Compute the post-stack `input_features` tensor for **base / plus**
/// (`GraniteSpeechFeatureExtractor`).
///
/// Input: raw 16 kHz mono float waveform. Output: `[1, T_stacked, 160]`
/// ready to feed `encoder.onnx`. The leading batch axis is always 1; the
/// crate does not currently batch utterances.
pub fn extract_input_features(audio: &[f32]) -> Result<Array3<f32>> {
    let spec = stft_power(audio)?;
    let fb = htk_mel_filterbank();

    // mel = filterbank @ spec  ->  [N_MELS, T_mel]
    let mel = fb.dot(&spec);
    // Transpose to [T_mel, N_MELS] for downstream stacking.
    let mel = mel.t().to_owned();

    let stacked = finalise_log_mel_and_stack(mel);

    // Add batch axis -> [1, T_stacked, 160] in standard (C-contig) layout
    // since ort::TensorRef requires it.
    let (t_stacked, feat) = (stacked.shape()[0], stacked.shape()[1]);
    Ok(
        Array3::from_shape_vec((1, t_stacked, feat), stacked.into_raw_vec_and_offset().0)
            .expect("batch-1 reshape always valid"),
    )
}

/// Compute the post-stack `input_features` and matching `attention_mask`
/// for **NAR** (`NLEFeatureExtractor`). The mel sequence is truncated
/// before log/normalisation to `l = 2 * (T // (2 * hop_length))` frames
/// (i.e. the encoder frame rate of 50 Hz, 1 frame per 320 samples).
///
/// Returns `(input_features [1, T_stacked, 160], attention_mask [1, T_stacked])`
/// where attention_mask is `1` for real frames and `0` for padding. With
/// a single utterance and no padding the mask is all ones.
pub fn extract_input_features_nar(audio: &[f32]) -> Result<(Array3<f32>, Array2<i64>)> {
    let raw_len = audio.len();
    let l = 2 * (raw_len / (2 * HOP_LENGTH));

    let spec = stft_power(audio)?;
    let fb = htk_mel_filterbank();
    let mel = fb.dot(&spec); // [N_MELS, T_mel_full]

    // Truncate mel to `l` frames along the time axis, matching
    // `mel[..., :l]` in the Python source.
    let t_mel_full = mel.shape()[1];
    let l_clamped = l.min(t_mel_full);
    let mel = mel.slice(ndarray::s![.., ..l_clamped]).to_owned();
    let mel = mel.t().to_owned();

    let stacked = finalise_log_mel_and_stack(mel);
    let (t_stacked, feat) = (stacked.shape()[0], stacked.shape()[1]);
    let input_features =
        Array3::from_shape_vec((1, t_stacked, feat), stacked.into_raw_vec_and_offset().0)
            .expect("batch-1 reshape always valid");
    // Single utterance, no padding -> mask is all ones.
    let attention_mask = Array2::<i64>::from_elem((1, t_stacked), 1);
    Ok((input_features, attention_mask))
}

/// Mix down to mono if the caller supplied multi-channel interleaved
/// samples and reject any sample rate other than 16 kHz. Caller is
/// expected to resample before this point.
pub fn prepare_mono_16k(audio: &[f32], sample_rate: u32, channels: u16) -> Result<Vec<f32>> {
    if sample_rate != SAMPLE_RATE as u32 {
        return Err(Error::Audio(format!(
            "Granite Speech expects 16 kHz audio; got {sample_rate} Hz. Resample before calling."
        )));
    }
    if channels <= 1 {
        return Ok(audio.to_vec());
    }
    let c = channels as usize;
    let mono = audio
        .chunks_exact(c)
        .map(|chunk| chunk.iter().sum::<f32>() / c as f32)
        .collect();
    Ok(mono)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dc_signal(len: usize, value: f32) -> Vec<f32> {
        vec![value; len]
    }

    fn sine(freq_hz: f32, samples: usize) -> Vec<f32> {
        use std::f32::consts::PI;
        (0..samples)
            .map(|i| (2.0 * PI * freq_hz * i as f32 / SAMPLE_RATE as f32).sin())
            .collect()
    }

    #[test]
    fn filterbank_shape_and_partition() {
        let fb = htk_mel_filterbank();
        assert_eq!(fb.shape(), &[N_MELS, N_FFT / 2 + 1]);
        // Triangular filters sum should be > 0 across the spectrum interior.
        let col_sum: Array1<f32> = fb.sum_axis(ndarray::Axis(0));
        // The very first FFT bin (DC) lies below mel(0)=0, so its
        // filterbank column may legitimately be zero. We only require
        // *some* interior bin to be covered.
        assert!(col_sum.iter().copied().any(|v: f32| v > 0.0));
    }

    #[test]
    fn stack_halves_time_dim_and_doubles_features() {
        // 1 second of silence -> 100 mel frames-ish; stacking halves the
        // time axis and gives 160-dim features.
        let audio = dc_signal(SAMPLE_RATE, 0.0);
        let feats = extract_input_features(&audio).unwrap();
        assert_eq!(feats.shape()[0], 1, "batch axis = 1");
        assert_eq!(feats.shape()[2], STACKED_FEATURES, "post-stack feature dim");
        assert!(feats.shape()[1] > 0);
    }

    #[test]
    fn nar_truncates_to_320_sample_grid() {
        // 8.43 seconds of audio, like the bundle's reference clip.
        let audio = sine(440.0, 134_880);
        let (feats, mask) = extract_input_features_nar(&audio).unwrap();
        // raw_len // (2*hop) = 134880 // 320 = 421 stacked frames
        assert_eq!(feats.shape()[1], 421);
        assert_eq!(mask.shape(), &[1, 421]);
        // No padding for a single utterance -> mask all ones.
        assert!(mask.iter().all(|&v| v == 1));
    }

    #[test]
    fn reject_non_16k_audio() {
        let err = prepare_mono_16k(&[0.0; 100], 22_050, 1).unwrap_err();
        match err {
            Error::Audio(msg) => assert!(msg.contains("16 kHz")),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn mono_mix_collapses_stereo_channels() {
        let stereo: Vec<f32> = vec![1.0, -1.0, 0.5, 0.5, 2.0, 0.0];
        let mono = prepare_mono_16k(&stereo, 16_000, 2).unwrap();
        assert_eq!(mono, vec![0.0, 0.5, 1.0]);
    }
}
