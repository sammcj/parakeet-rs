/*
IBM Granite Speech 4.1 2b NAR - non-autoregressive English ASR.

440M Conformer encoder + Q-Former projector + bidirectional NLE editor.
Single-pass parallel decode: no autoregressive loop, no KV cache. The
editor sees the whole utterance at once and rewrites the CTC draft in
one forward pass, so this is the lower-latency variant. The trade-off
is feature breadth: English transcription only, no punctuation, no
translation, no speaker tags.

Download an ONNX bundle from:
https://github.com/sammcj/granite-speech-4.1-onnx
Each bundle ships fp32/, fp16w/, and int8/ subdirectories; fp16w is the
recommended default for accelerated inference.

Usage:
  cargo run --release --example granite_nar --features granite-nar -- \
    <bundle_dir> <audio.wav> [precision]

  cargo run --release --example granite_nar --features granite-nar -- \
    <bundle_dir> verify [precision]

Precisions: fp16w (default), fp32, int8

Examples:
  cargo run --release --example granite_nar --features granite-nar -- ./bundle audio.wav
  cargo run --release --example granite_nar --features granite-nar -- ./bundle audio.wav fp32
  cargo run --release --example granite_nar --features granite-nar -- ./bundle verify

Acceleration:
  Compile with --features granite-nar,coreml on macOS for Metal acceleration:
    cargo run --release --example granite_nar --features granite-nar,coreml -- ./bundle audio.wav
  Or --features granite-nar,cuda on Linux/Windows for NVIDIA GPU.

Verify mode:
  Validates a NAR bundle against the golden test fixtures shipped at
  <bundle_dir>/test_fixtures/. Three checks run in order:
    1. Audio frontend parity vs expected_input_features.npy + expected_attention_mask.npy
    2. Encoder + projector parity vs expected_audio_embeds.npy
    3. End-to-end transcript vs the upstream reference
*/

#[cfg(feature = "granite-nar")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::env;

    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <bundle_dir> (<audio.wav> | verify) [precision]",
            args[0]
        );
        eprintln!("  precision: fp16w (default) | fp32 | int8");
        std::process::exit(1);
    }

    // Optional `ep=cpu|coreml|cuda` flag picks the execution provider.
    let mut ep_arg: Option<String> = None;
    let positional: Vec<String> = args
        .into_iter()
        .filter(|a| {
            if let Some(v) = a.strip_prefix("ep=") {
                ep_arg = Some(v.to_string());
                false
            } else {
                true
            }
        })
        .collect();

    let bundle_dir = &positional[1];
    if positional.get(2).map(String::as_str) == Some("verify") {
        let precision_arg = positional.get(3).map(String::as_str).unwrap_or("fp16w");
        return run_verify(bundle_dir, precision_arg, ep_arg.as_deref());
    }

    let audio_path = &positional[2];
    let precision = positional.get(3).map(String::as_str).unwrap_or("fp16w");
    run_transcribe(bundle_dir, audio_path, precision, ep_arg.as_deref())
}

#[cfg(feature = "granite-nar")]
fn run_transcribe(
    bundle_dir: &str,
    audio_path: &str,
    precision: &str,
    ep: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    use parakeet_rs::{GraniteNar, GraniteNarOptions};
    use std::time::Instant;

    let prec = parse_precision(precision)?;
    let exec = parse_ep(ep)?;

    let audio = load_wav_mono_16k(std::path::Path::new(audio_path))?;
    let duration_secs = audio.len() as f32 / 16000.0;
    println!(
        "Audio: {:.2}s, precision={}, ep={}",
        duration_secs,
        precision,
        ep.unwrap_or("cpu")
    );

    println!("Loading Granite Speech 4.1 2b NAR ({precision})...");
    let load_start = Instant::now();
    let mut model = GraniteNar::from_pretrained_with_precision(bundle_dir, prec, exec)?;
    println!("Loaded in {:.2}s", load_start.elapsed().as_secs_f32());

    let opts = GraniteNarOptions::default();
    let start = Instant::now();
    let text = model.transcribe_audio(&audio, &opts)?;
    let elapsed = start.elapsed().as_secs_f32();

    println!("\n{text}");
    println!(
        "\nTranscribed in {:.2}s (RTF {:.2}x)",
        elapsed,
        duration_secs / elapsed
    );
    Ok(())
}

#[cfg(feature = "granite-nar")]
fn run_verify(
    bundle_dir: &str,
    precision_arg: &str,
    ep: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    use parakeet_rs::{GraniteNar, GraniteNarOptions, GranitePrecision};
    use std::path::PathBuf;
    use std::time::Instant;

    let bundle_dir = PathBuf::from(bundle_dir);
    let precision = parse_precision(precision_arg)?;
    let exec = parse_ep(ep)?;
    println!("== Bundle ==");
    println!("  dir       : {}", bundle_dir.display());
    println!("  precision : {precision_arg}");
    println!("  ep        : {}", ep.unwrap_or("cpu"));

    let fixtures_dir = bundle_dir.join("test_fixtures");
    if !fixtures_dir.is_dir() {
        return Err(format!(
            "Bundle is missing test_fixtures/ at {}. Re-export with the latest pipeline at https://github.com/sammcj/granite-speech-4.1-onnx",
            fixtures_dir.display()
        )
        .into());
    }
    let wav_path = fixtures_dir.join("sample_audio.wav");
    let expected_features_path = fixtures_dir.join("expected_input_features.npy");
    let expected_mask_path = fixtures_dir.join("expected_attention_mask.npy");
    let expected_embeds_path = fixtures_dir.join("expected_audio_embeds.npy");

    println!("\n== Audio frontend parity ==");
    let audio = load_wav_mono_16k(&wav_path)?;
    println!(
        "  loaded {} samples ({:.2}s)",
        audio.len(),
        audio.len() as f32 / 16000.0
    );
    let (actual_features, actual_mask) = GraniteNar::extract_input_features_nar(&audio)?;
    let expected_features = read_npy_f32(&expected_features_path)?;
    let actual_features_shape = [
        actual_features.shape()[0],
        actual_features.shape()[1],
        actual_features.shape()[2],
    ];
    if expected_features.shape != actual_features_shape {
        return Err(format!(
            "Frontend shape mismatch: actual {actual_features_shape:?}, expected {:?}",
            expected_features.shape
        )
        .into());
    }
    let actual_features_flat: Vec<f32> = actual_features.iter().copied().collect();
    let stats = compare_f32(&actual_features_flat, &expected_features.data);
    print_stats("  input_features  ", &stats);
    if !within_tolerance(&stats, 1e-3, 1e-4) {
        return Err("Frontend parity failed (input_features)".into());
    }

    let expected_mask = read_npy_i64_2d(&expected_mask_path)?;
    let actual_mask_shape = [actual_mask.shape()[0], actual_mask.shape()[1]];
    if expected_mask.shape != actual_mask_shape {
        return Err(format!(
            "attention_mask shape mismatch: actual {actual_mask_shape:?}, expected {:?}",
            expected_mask.shape
        )
        .into());
    }
    let mismatched = actual_mask
        .iter()
        .zip(expected_mask.data.iter())
        .filter(|(a, b)| a != b)
        .count();
    if mismatched != 0 {
        return Err(format!("attention_mask mismatch: {mismatched} positions differ").into());
    }
    println!("  attention_mask  shape={actual_mask_shape:?} (exact match)");
    println!("  PASS (rtol=1e-3, atol=1e-4)");

    println!("\n== Encoder + projector parity ==");
    let load_start = Instant::now();
    let mut nar = GraniteNar::from_pretrained_with_precision(&bundle_dir, precision, exec)?;
    println!(
        "  loaded model in {:.2}s",
        load_start.elapsed().as_secs_f32()
    );
    let enc_start = Instant::now();
    let (audio_embeds, n_valid) = nar.run_encoder(&actual_features, &actual_mask)?;
    println!(
        "  encoder ran in {:.2}s, audio_embeds={:?}, n_valid={n_valid}",
        enc_start.elapsed().as_secs_f32(),
        audio_embeds.shape()
    );
    let expected_embeds = read_npy_f32(&expected_embeds_path)?;
    let compare_t = expected_embeds.shape[1];
    if audio_embeds.shape()[1] < compare_t {
        return Err(format!(
            "audio_embeds T={} smaller than expected fixture T={compare_t}",
            audio_embeds.shape()[1]
        )
        .into());
    }
    let enc_flat: Vec<f32> = audio_embeds
        .slice(ndarray::s![.., ..compare_t, ..])
        .iter()
        .copied()
        .collect();
    let enc_stats = compare_f32(&enc_flat, &expected_embeds.data);
    print_stats("  ", &enc_stats);
    let (enc_rtol, enc_atol, enc_check) = match precision {
        GranitePrecision::Fp32 => (1e-4_f32, 1e-5_f32, true),
        GranitePrecision::Fp16w => (5e-3_f32, 5e-3_f32, true),
        GranitePrecision::Int8 => (0.0, 0.0, false),
    };
    if enc_check {
        if !within_tolerance(&enc_stats, enc_rtol, enc_atol) {
            return Err(format!(
                "Encoder parity failed at {precision_arg} (rtol={enc_rtol}, atol={enc_atol})"
            )
            .into());
        }
        println!("  PASS (rtol={enc_rtol}, atol={enc_atol})");
    } else {
        println!("  INT8: stats only (logit values shift, argmax preserved)");
    }

    println!("\n== Transcript ==");
    let opts = GraniteNarOptions::default();
    let dec_start = Instant::now();
    let transcript = nar.transcribe_audio(&audio, &opts)?;
    let dec_elapsed = dec_start.elapsed().as_secs_f32();
    let dur = audio.len() as f32 / 16000.0;
    println!(
        "  RTF       : {:.2}x ({:.2}s decode, {:.2}s audio)",
        dur / dec_elapsed,
        dec_elapsed,
        dur
    );
    println!("  reference : after his nap timothy lazily stretched first one gray velvet foot then another strolled indolently to his plate turning over the food carefully selecting choice bits nosing out that which he scorned upon the clean hearth");
    println!("  actual    : {transcript}");
    println!("\nAll checks complete.");
    Ok(())
}

#[cfg(feature = "granite-nar")]
fn parse_precision(arg: &str) -> Result<parakeet_rs::GranitePrecision, Box<dyn std::error::Error>> {
    use parakeet_rs::GranitePrecision;
    match arg {
        "fp16w" | "" => Ok(GranitePrecision::Fp16w),
        "fp32" => Ok(GranitePrecision::Fp32),
        "int8" => Ok(GranitePrecision::Int8),
        other => Err(format!("Unknown precision '{other}' (expected fp16w/fp32/int8)").into()),
    }
}

#[cfg(feature = "granite-nar")]
fn parse_ep(
    ep: Option<&str>,
) -> Result<Option<parakeet_rs::ExecutionConfig>, Box<dyn std::error::Error>> {
    use parakeet_rs::{ExecutionConfig, ExecutionProvider};
    let Some(name) = ep else {
        return Ok(None);
    };
    let provider = match name {
        "cpu" => ExecutionProvider::Cpu,
        #[cfg(feature = "coreml")]
        "coreml" => ExecutionProvider::CoreML,
        #[cfg(feature = "cuda")]
        "cuda" => ExecutionProvider::Cuda,
        other => {
            return Err(format!(
                "Unknown ep '{other}'. Available at this build: cpu{}{}",
                if cfg!(feature = "coreml") {
                    ", coreml"
                } else {
                    ""
                },
                if cfg!(feature = "cuda") { ", cuda" } else { "" },
            )
            .into());
        }
    };
    #[allow(unused_mut)]
    let mut cfg = ExecutionConfig::new().with_execution_provider(provider);
    #[cfg(feature = "coreml")]
    if matches!(name, "coreml") {
        if let Ok(dir) = std::env::var("PARAKEET_COREML_CACHE_DIR") {
            cfg = cfg.with_coreml_cache_dir(dir);
        }
    }
    Ok(Some(cfg))
}

#[cfg(feature = "granite-nar")]
fn load_wav_mono_16k(path: &std::path::Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.sample_rate != 16000 {
        return Err(format!("Expected 16 kHz, got {} Hz", spec.sample_rate).into());
    }
    let mut audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / 32768.0))
            .collect::<Result<Vec<_>, _>>()?,
    };
    if spec.channels > 1 {
        let c = spec.channels as usize;
        audio = audio
            .chunks_exact(c)
            .map(|chunk| chunk.iter().sum::<f32>() / c as f32)
            .collect();
    }
    Ok(audio)
}

#[cfg(feature = "granite-nar")]
struct Npy3 {
    shape: [usize; 3],
    data: Vec<f32>,
}

#[cfg(feature = "granite-nar")]
struct NpyRaw {
    dims: Vec<usize>,
    descr: String,
    bytes: Vec<u8>,
}

#[cfg(feature = "granite-nar")]
fn read_npy_f32(path: &std::path::Path) -> Result<Npy3, Box<dyn std::error::Error>> {
    let raw = read_npy_raw(path)?;
    if !(raw.descr == "<f4" || raw.descr == "|f4" || raw.descr == "=f4") {
        return Err(format!("expected float32 ('<f4'), got '{}'", raw.descr).into());
    }
    if raw.dims.len() != 3 {
        return Err(format!("expected 3-D array, got shape {:?}", raw.dims).into());
    }
    let data: Vec<f32> = raw
        .bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(Npy3 {
        shape: [raw.dims[0], raw.dims[1], raw.dims[2]],
        data,
    })
}

#[cfg(feature = "granite-nar")]
struct Npy2I64 {
    shape: [usize; 2],
    data: Vec<i64>,
}

#[cfg(feature = "granite-nar")]
fn read_npy_i64_2d(path: &std::path::Path) -> Result<Npy2I64, Box<dyn std::error::Error>> {
    let raw = read_npy_raw(path)?;
    if !(raw.descr == "<i8" || raw.descr == "|i8" || raw.descr == "=i8") {
        return Err(format!("expected int64 ('<i8'), got '{}'", raw.descr).into());
    }
    if raw.dims.len() != 2 {
        return Err(format!("expected 2-D array, got shape {:?}", raw.dims).into());
    }
    let data: Vec<i64> = raw
        .bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect();
    Ok(Npy2I64 {
        shape: [raw.dims[0], raw.dims[1]],
        data,
    })
}

#[cfg(feature = "granite-nar")]
fn read_npy_raw(path: &std::path::Path) -> Result<NpyRaw, Box<dyn std::error::Error>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut head6 = [0u8; 6];
    f.read_exact(&mut head6)?;
    if &head6 != b"\x93NUMPY" {
        return Err(format!("{} is not a .npy file", path.display()).into());
    }
    let mut version = [0u8; 2];
    f.read_exact(&mut version)?;
    let header_len: usize = match version[0] {
        1 => {
            let mut l = [0u8; 2];
            f.read_exact(&mut l)?;
            u16::from_le_bytes(l) as usize
        }
        2 | 3 => {
            let mut l = [0u8; 4];
            f.read_exact(&mut l)?;
            u32::from_le_bytes(l) as usize
        }
        v => return Err(format!("unsupported NPY version {v}").into()),
    };
    let mut header_buf = vec![0u8; header_len];
    f.read_exact(&mut header_buf)?;
    let header = std::str::from_utf8(&header_buf)?;
    let descr = extract_field(header, "descr")
        .ok_or("missing descr")?
        .to_string();
    let fortran = extract_field(header, "fortran_order").ok_or("missing fortran_order")?;
    if fortran == "True" {
        return Err("fortran-order arrays not supported".into());
    }
    let shape_str = extract_shape(header).ok_or("missing shape")?;
    let dims: Vec<usize> = shape_str
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;
    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes)?;
    Ok(NpyRaw { dims, descr, bytes })
}

#[cfg(feature = "granite-nar")]
fn extract_field<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let pat = format!("'{key}':");
    let i = header.find(&pat)? + pat.len();
    let rest = header[i..].trim_start();
    if let Some(stripped) = rest.strip_prefix('\'') {
        let end = stripped.find('\'')?;
        Some(&stripped[..end])
    } else {
        let end = rest.find(',').or_else(|| rest.find('}'))?;
        Some(rest[..end].trim())
    }
}

#[cfg(feature = "granite-nar")]
fn extract_shape(header: &str) -> Option<&str> {
    let i = header.find("'shape':")? + "'shape':".len();
    let rest = header[i..].trim_start();
    let inner = rest.strip_prefix('(')?;
    let end = inner.find(')')?;
    Some(&inner[..end])
}

#[cfg(feature = "granite-nar")]
struct Stats {
    max_abs: f32,
    mean_abs: f32,
    max_rel: f32,
    n: usize,
}

#[cfg(feature = "granite-nar")]
fn compare_f32(a: &[f32], b: &[f32]) -> Stats {
    assert_eq!(a.len(), b.len());
    let mut max_abs = 0.0_f32;
    let mut sum_abs = 0.0_f64;
    let mut max_rel = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let abs = (x - y).abs();
        max_abs = max_abs.max(abs);
        sum_abs += abs as f64;
        let denom = y.abs().max(1e-12);
        max_rel = max_rel.max(abs / denom);
    }
    Stats {
        max_abs,
        mean_abs: (sum_abs / a.len() as f64) as f32,
        max_rel,
        n: a.len(),
    }
}

#[cfg(feature = "granite-nar")]
fn print_stats(prefix: &str, s: &Stats) {
    println!(
        "{prefix}elements  : {}\n{prefix}max abs   : {:.3e}\n{prefix}mean abs  : {:.3e}\n{prefix}max rel   : {:.3e}",
        s.n, s.max_abs, s.mean_abs, s.max_rel
    );
}

#[cfg(feature = "granite-nar")]
fn within_tolerance(s: &Stats, rtol: f32, atol: f32) -> bool {
    s.max_abs <= atol + rtol
}

#[cfg(not(feature = "granite-nar"))]
fn main() {
    eprintln!("Rebuild with --features granite-nar to run this example.");
    std::process::exit(1);
}
