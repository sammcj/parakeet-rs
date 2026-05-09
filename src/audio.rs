use crate::config::PreprocessorConfig;
use crate::error::{Error, Result};
use hound::{WavReader, WavSpec};
use ndarray::Array2;
use std::f32::consts::PI;
use std::path::Path;

pub fn load_audio<P: AsRef<Path>>(path: P) -> Result<(Vec<f32>, WavSpec)> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::Audio(format!("Failed to read float samples: {e}")))?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::Audio(format!("Failed to read int samples: {e}")))?,
    };

    Ok((samples, spec))
}

pub fn apply_preemphasis(audio: &[f32], coef: f32) -> Vec<f32> {
    if audio.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(audio.len());
    result.push(audio[0]);

    for i in 1..audio.len() {
        result.push(audio[i] - coef * audio[i - 1]);
    }

    result
}

/// Padding mode for [`stft`]. `Zero` matches NeMo / Cohere defaults;
/// `Reflect` matches `torchaudio.transforms.Spectrogram(center=True,
/// pad_mode="reflect")` used by Granite Speech.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum PadMode {
    Zero,
    Reflect,
}

/// Hann window definition for [`stft`]. `Symmetric` divides by
/// `win_length - 1` and places the window at the start of the FFT
/// buffer (NeMo / Cohere). `Periodic` divides by `win_length` and
/// centre-pads the window into the FFT buffer (torchaudio default,
/// Granite Speech).
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum WindowMode {
    Symmetric,
    Periodic,
}

fn hann_symmetric(win_length: usize) -> Vec<f32> {
    (0..win_length)
        .map(|i| 0.5 - 0.5 * ((2.0 * PI * i as f32) / (win_length as f32 - 1.0)).cos())
        .collect()
}

fn hann_periodic_padded(win_length: usize, n_fft: usize) -> Vec<f32> {
    debug_assert!(win_length <= n_fft);
    let pad_left = (n_fft - win_length) / 2;
    let mut w = vec![0.0f32; n_fft];
    for i in 0..win_length {
        w[pad_left + i] = 0.5 - 0.5 * ((2.0 * PI * i as f32) / win_length as f32).cos();
    }
    w
}

fn reflect_pad(audio: &[f32], pad: usize) -> Vec<f32> {
    if audio.is_empty() {
        return vec![0.0; pad * 2];
    }
    let n = audio.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    for i in (1..=pad).rev() {
        let idx = i.min(n - 1);
        out.push(audio[idx]);
    }
    out.extend_from_slice(audio);
    for i in 1..=pad {
        let idx = if n > i { n - 1 - i } else { 0 };
        out.push(audio[idx]);
    }
    out
}

// We use proper FFT here instead of naive DFT because the model was trained
// on correctly computed spectrograms. Naive DFT produces wrong frequency bins
// and the model outputs all blank tokens. realfft (real-valued FFT wrapper around
// RustFFT) gives us O(n log n) performance and numerically correct results.
//
// Returns the per-frame power spectrogram (`|FFT|^2`) of shape
// `[n_fft/2 + 1, num_frames]`. The `pad_mode` and `window_mode` knobs let
// the same function back both the NeMo / Cohere frontends (`Zero` +
// `Symmetric`) and the torchaudio-style Granite frontend (`Reflect` +
// `Periodic`).
pub fn stft(
    audio: &[f32],
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    pad_mode: PadMode,
    window_mode: WindowMode,
) -> Result<Array2<f32>> {
    use realfft::RealFftPlanner;

    let pad_amount = n_fft / 2;
    let padded: Vec<f32> = match pad_mode {
        PadMode::Zero => {
            let mut p = vec![0.0f32; pad_amount];
            p.extend_from_slice(audio);
            p.resize(p.len() + pad_amount, 0.0);
            p
        }
        PadMode::Reflect => reflect_pad(audio, pad_amount),
    };

    let freq_bins = n_fft / 2 + 1;
    if padded.len() < n_fft {
        // Audio shorter than one analysis window after padding: emit a
        // single zero frame to match torchaudio's behaviour on tiny
        // inputs. Both pad modes converge here.
        return Ok(Array2::<f32>::zeros((freq_bins, 1)));
    }

    let num_frames = (padded.len() - n_fft) / hop_length + 1;
    let mut spectrogram = Array2::<f32>::zeros((freq_bins, num_frames));

    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(n_fft);
    let mut input = vec![0.0f32; n_fft];
    let mut output = r2c.make_output_vec();
    let mut scratch = r2c.make_scratch_vec();

    match window_mode {
        WindowMode::Symmetric => {
            let window = hann_symmetric(win_length);
            for frame_idx in 0..num_frames {
                let start = frame_idx * hop_length;
                input.fill(0.0);
                for i in 0..win_length.min(padded.len() - start) {
                    input[i] = padded[start + i] * window[i];
                }
                r2c.process_with_scratch(&mut input, &mut output, &mut scratch)
                    .map_err(|e| Error::Audio(format!("FFT failed: {e}")))?;
                for k in 0..freq_bins {
                    spectrogram[[k, frame_idx]] = output[k].norm_sqr();
                }
            }
        }
        WindowMode::Periodic => {
            let window = hann_periodic_padded(win_length, n_fft);
            for frame_idx in 0..num_frames {
                let start = frame_idx * hop_length;
                for i in 0..n_fft {
                    input[i] = padded[start + i] * window[i];
                }
                r2c.process_with_scratch(&mut input, &mut output, &mut scratch)
                    .map_err(|e| Error::Audio(format!("FFT failed: {e}")))?;
                for k in 0..freq_bins {
                    spectrogram[[k, frame_idx]] = output[k].norm_sqr();
                }
            }
        }
    }

    Ok(spectrogram)
}

/// Mel scale variant for [`create_mel_filterbank`]. Slaney follows
/// librosa's piecewise log/linear curve (NeMo / Cohere). Htk follows
/// the original 2595*log10 closed form (Granite Speech / torchaudio).
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum MelScale {
    Slaney,
    Htk,
}

/// Triangle-norm variant for [`create_mel_filterbank`]. Slaney divides
/// each filter by the area between its endpoints (`2 / (right - left)`).
/// `None` leaves the raw triangles unscaled (torchaudio `norm=None`).
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum MelNorm {
    Slaney,
    None,
}

const SLANEY_F_SP: f64 = 200.0 / 3.0;
const SLANEY_MIN_LOG_HZ: f64 = 1000.0;
const SLANEY_MIN_LOG_MEL: f64 = SLANEY_MIN_LOG_HZ / SLANEY_F_SP;
const SLANEY_LOG_STEP: f64 = 0.06875177742094912;

fn hz_to_mel_slaney(hz: f64) -> f64 {
    if hz < SLANEY_MIN_LOG_HZ {
        hz / SLANEY_F_SP
    } else {
        SLANEY_MIN_LOG_MEL + (hz / SLANEY_MIN_LOG_HZ).ln() / SLANEY_LOG_STEP
    }
}

fn mel_to_hz_slaney(mel: f64) -> f64 {
    if mel < SLANEY_MIN_LOG_MEL {
        mel * SLANEY_F_SP
    } else {
        SLANEY_MIN_LOG_HZ * ((mel - SLANEY_MIN_LOG_MEL) * SLANEY_LOG_STEP).exp()
    }
}

fn hz_to_mel_htk(f: f64) -> f64 {
    2595.0 * (1.0 + f / 700.0).log10()
}

fn mel_to_hz_htk(m: f64) -> f64 {
    700.0 * (10f64.powf(m / 2595.0) - 1.0)
}

pub fn create_mel_filterbank(
    n_fft: usize,
    n_mels: usize,
    sample_rate: usize,
    scale: MelScale,
    norm: MelNorm,
) -> Array2<f32> {
    let freq_bins = n_fft / 2 + 1;
    let mut filterbank = Array2::<f32>::zeros((n_mels, freq_bins));

    let fmax = sample_rate as f64 / 2.0;
    let mel_points: Vec<f64> = match scale {
        MelScale::Slaney => {
            let mel_min = hz_to_mel_slaney(0.0);
            let mel_max = hz_to_mel_slaney(fmax);
            (0..=n_mels + 1)
                .map(|i| {
                    mel_to_hz_slaney(
                        mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64,
                    )
                })
                .collect()
        }
        MelScale::Htk => {
            let mel_min = hz_to_mel_htk(0.0);
            let mel_max = hz_to_mel_htk(fmax);
            (0..=n_mels + 1)
                .map(|i| {
                    mel_to_hz_htk(mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64)
                })
                .collect()
        }
    };

    let fft_freqs: Vec<f64> = (0..freq_bins)
        .map(|i| i as f64 * sample_rate as f64 / n_fft as f64)
        .collect();

    let fdiff: Vec<f64> = mel_points.windows(2).map(|w| w[1] - w[0]).collect();

    for i in 0..n_mels {
        for (k, &freq) in fft_freqs.iter().enumerate() {
            let lower = (freq - mel_points[i]) / fdiff[i];
            let upper = (mel_points[i + 2] - freq) / fdiff[i + 1];
            filterbank[[i, k]] = 0.0f64.max(lower.min(upper)) as f32;
        }
    }

    if matches!(norm, MelNorm::Slaney) {
        for i in 0..n_mels {
            let enorm = 2.0 / (mel_points[i + 2] - mel_points[i]);
            for k in 0..freq_bins {
                filterbank[[i, k]] *= enorm as f32;
            }
        }
    }

    filterbank
}

/// Extract mel spectrogram features from raw audio samples.
///
/// # Arguments
///
/// * `audio` - Audio samples as f32 values
/// * `sample_rate` - Sample rate in Hz
/// * `channels` - Number of audio channels
/// * `config` - Preprocessor configuration
///
/// # Returns
///
/// 2D array of mel spectrogram features (time_steps x feature_size)
pub fn extract_features_raw(
    mut audio: Vec<f32>,
    sample_rate: u32,
    channels: u16,
    config: &PreprocessorConfig,
) -> Result<Array2<f32>> {
    if sample_rate != config.sampling_rate as u32 {
        return Err(Error::Audio(format!(
            "Audio sample rate {} doesn't match expected {}. Please resample your audio first.",
            sample_rate, config.sampling_rate
        )));
    }

    if channels > 1 {
        let mono: Vec<f32> = audio
            .chunks(channels as usize)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect();
        audio = mono;
    }

    audio = apply_preemphasis(&audio, config.preemphasis);

    let spectrogram = stft(
        &audio,
        config.n_fft,
        config.hop_length,
        config.win_length,
        PadMode::Zero,
        WindowMode::Symmetric,
    )?;

    let mel_filterbank = create_mel_filterbank(
        config.n_fft,
        config.feature_size,
        config.sampling_rate,
        MelScale::Slaney,
        MelNorm::Slaney,
    );
    let mel_spectrogram = mel_filterbank.dot(&spectrogram);
    // Log with additive guard (NeMo: log_zero_guard_type="add", value=2^-24)
    let log_zero_guard: f32 = 2.0f32.powi(-24);
    let mel_spectrogram = mel_spectrogram.mapv(|x| (x + log_zero_guard).ln());

    let mut mel_spectrogram = mel_spectrogram.t().to_owned();

    // Normalize per_feature: mean=0, std=1 with Bessel's correction (N-1)
    let num_frames = mel_spectrogram.shape()[0];
    let num_features = mel_spectrogram.shape()[1];

    if num_frames <= 1 {
        return Ok(mel_spectrogram);
    }

    for feat_idx in 0..num_features {
        let mut column = mel_spectrogram.column_mut(feat_idx);
        let mean: f32 = column.iter().sum::<f32>() / num_frames as f32;
        let variance: f32 =
            column.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / (num_frames as f32 - 1.0);
        let std = variance.sqrt() + 1e-5;

        for val in column.iter_mut() {
            *val = (*val - mean) / std;
        }
    }

    Ok(mel_spectrogram)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a pure sine wave at the given frequency and sample rate.
    fn sine_wave(freq_hz: f32, sample_rate: usize, num_samples: usize) -> Vec<f32> {
        (0..num_samples)
            .map(|i| (2.0 * PI * freq_hz * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn stft_concentrates_power_at_expected_bin() {
        // 1kHz sine at 16kHz sample rate, 1 second
        let n_fft = 512;
        let hop_length = 160;
        let win_length = 400;
        let sample_rate = 16000;
        let audio = sine_wave(1000.0, sample_rate, sample_rate);

        let spec = stft(
            &audio,
            n_fft,
            hop_length,
            win_length,
            PadMode::Zero,
            WindowMode::Symmetric,
        )
        .unwrap();

        // Expected bin for 1kHz: freq_hz * n_fft / sample_rate = 1000 * 512 / 16000 = 32
        let expected_bin = 32;
        let freq_bins = n_fft / 2 + 1;
        let num_frames = spec.shape()[1];

        // Check that bin 32 has the highest power in most frames (skip edge frames)
        let mut correct_frames = 0;
        for frame in 2..num_frames.saturating_sub(2) {
            let mut max_bin = 0;
            let mut max_power = 0.0f32;
            for bin in 0..freq_bins {
                if spec[[bin, frame]] > max_power {
                    max_power = spec[[bin, frame]];
                    max_bin = bin;
                }
            }
            if max_bin == expected_bin {
                correct_frames += 1;
            }
        }

        let interior_frames = num_frames.saturating_sub(4);
        assert!(
            correct_frames > interior_frames / 2,
            "Expected bin {expected_bin} to dominate in most frames, but only {correct_frames}/{interior_frames}"
        );
    }

    #[test]
    fn htk_mel_endpoints_match_reference() {
        // HTK closed-form: mel(0) = 0; mel(8000) ≈ 2840.023.
        let m0 = hz_to_mel_htk(0.0);
        let m1 = hz_to_mel_htk(8000.0);
        assert!(m0.abs() < 1e-9, "expected ~0, got {m0}");
        assert!(
            (m1 - 2840.0230).abs() < 1e-3,
            "expected ~2840.023, got {m1}"
        );
        let h = mel_to_hz_htk(m1);
        assert!((h - 8000.0).abs() < 1e-6);
    }

    #[test]
    fn htk_filterbank_has_n_mels_rows() {
        let fb = create_mel_filterbank(512, 80, 16000, MelScale::Htk, MelNorm::None);
        assert_eq!(fb.shape(), &[80, 257]);
    }

    #[test]
    fn stft_output_shape_is_correct() {
        let n_fft = 512;
        let hop_length = 160;
        let win_length = 400;
        let audio = vec![0.0f32; 16000]; // 1 second of silence

        let spec = stft(
            &audio,
            n_fft,
            hop_length,
            win_length,
            PadMode::Zero,
            WindowMode::Symmetric,
        )
        .unwrap();

        let freq_bins = n_fft / 2 + 1;
        assert_eq!(spec.shape()[0], freq_bins);
        // num_frames = (audio_len + n_fft - n_fft) / hop_length + 1 = 16000 / 160 + 1 = 101
        assert!(spec.shape()[1] > 0);
    }
}
