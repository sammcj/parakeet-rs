/*
IBM Granite Speech 4.1 plus 2b - speaker-attributed ASR + word timestamps.

Same architecture as the base 2b, retrained for richer outputs:
  - Speaker-attributed ASR (SAA): inline [Speaker N]: tags
  - Word-level timestamps: [T:N] centisecond markers
  - Incremental decoding: continue from a previous transcript via prefix_text

Plus does NOT produce punctuation/capitalisation; that's a deliberate
trade-off for the structural features. For punctuated transcripts, use
the base 2b model via `examples/granite.rs`.

Download the bundle from:
https://github.com/sammcj/granite-speech-4.1-onnx
fp16w is the recommended precision for accelerated inference.

Usage:
  cargo run --release --example granite_plus --features granite-plus -- \
    <bundle_dir> <audio.wav> [task] [precision]

  cargo run --release --example granite_plus --features granite-plus -- \
    <bundle_dir> verify [precision]

Tasks:
  raw                 - Plain transcription (default)
  saa                 - Speaker-attributed ASR
  ts                  - Word-level timestamps
  keywords:<kw1,..>   - With keyword biasing
  prefix:<text>       - Incremental decoding from a prior transcript
  verify              - run bundled fixture parity checks + features round-trip

Precisions: fp16w (default), fp32, int8

Examples:
  cargo run --release --example granite_plus --features granite-plus -- ./bundle audio.wav
  cargo run --release --example granite_plus --features granite-plus -- ./bundle two_speakers.wav saa
  cargo run --release --example granite_plus --features granite-plus -- ./bundle audio.wav ts
  cargo run --release --example granite_plus --features granite-plus -- ./bundle verify

Acceleration:
  Compile with --features granite-plus,coreml on macOS for Metal acceleration.

Verify mode:
  Validates a plus bundle against the golden test fixtures shipped at
  <bundle_dir>/test_fixtures/. Runs frontend parity, encoder parity, and
  exercises every task variant end-to-end so a reviewer can see them work.

Long-form audio:
  Plus was trained on audio up to 9 minutes for ASR/SAA and 5 minutes for
  timestamps. For longer files, split at silence and feed each chunk's
  prior transcript via the `prefix:` task to keep speaker numbering stable.
*/

#[cfg(feature = "granite-plus")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::env;

    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <bundle_dir> (<audio.wav> [task] | verify) [precision]",
            args[0]
        );
        eprintln!("  task: raw (default) | saa | ts | keywords:<kw1,..> | prefix:<text>");
        eprintln!("  precision: fp16w (default) | fp32 | int8");
        std::process::exit(1);
    }

    // Optional `ep=cpu|coreml|cuda` flag picks the execution provider.
    // Filtered out before positional parsing so the rest of the CLI stays
    // unchanged. CoreML / CUDA require the matching cargo feature.
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
    let task = positional.get(3).map(String::as_str).unwrap_or("raw");
    let precision = positional.get(4).map(String::as_str).unwrap_or("fp16w");
    run_transcribe(bundle_dir, audio_path, task, precision, ep_arg.as_deref())
}

#[cfg(feature = "granite-plus")]
fn run_transcribe(
    bundle_dir: &str,
    audio_path: &str,
    task: &str,
    precision: &str,
    ep: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    use parakeet_rs::GranitePlus;
    use std::time::Instant;

    let opts = parse_task(task)?;
    let prec = parse_precision(precision)?;
    let exec = parse_ep(ep)?;

    let audio = load_wav_mono_16k(std::path::Path::new(audio_path))?;
    let duration_secs = audio.len() as f32 / 16000.0;
    println!(
        "Audio: {:.2}s, task={}, precision={}, ep={}",
        duration_secs,
        task,
        precision,
        ep.unwrap_or("cpu")
    );

    println!("Loading Granite Speech 4.1 plus 2b ({precision})...");
    let load_start = Instant::now();
    let mut model = GranitePlus::from_pretrained_with_precision(bundle_dir, prec, exec)?;
    println!("Loaded in {:.2}s", load_start.elapsed().as_secs_f32());

    let start = Instant::now();
    let result = model.transcribe_audio(&audio, &opts)?;
    let elapsed = start.elapsed().as_secs_f32();

    println!("\nTranscript:\n{}", result.text);
    if !result.segments.is_empty() {
        println!("\nSpeaker segments:");
        for seg in &result.segments {
            println!("  [Speaker {}] {}", seg.speaker, seg.text);
        }
    }
    if !result.words.is_empty() {
        println!("\nWord timestamps (end-of-word, seconds):");
        for w in &result.words {
            let label = if w.is_silence { "(silence)" } else { &w.word };
            println!("  {:>7.2}s  {}", w.end_time, label);
        }
    }
    println!(
        "\nTranscribed in {:.2}s (RTF {:.2}x)",
        elapsed,
        duration_secs / elapsed
    );
    println!("\n[debug] raw model output:\n{}", result.raw_text);
    Ok(())
}

#[cfg(feature = "granite-plus")]
fn run_verify(
    bundle_dir: &str,
    precision_arg: &str,
    ep: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    use parakeet_rs::{Granite, GranitePlus, GranitePlusOptions, GranitePrecision};
    use std::path::PathBuf;
    use std::time::Instant;

    let bundle_dir = PathBuf::from(bundle_dir);
    let precision = parse_precision(precision_arg)?;
    let exec = parse_ep(ep)?;

    let fixtures_dir = bundle_dir.join("test_fixtures");
    if !fixtures_dir.is_dir() {
        return Err(format!(
            "Bundle is missing test_fixtures/ at {}",
            fixtures_dir.display()
        )
        .into());
    }
    let wav_path = fixtures_dir.join("sample_audio.wav");
    let expected_features_path = fixtures_dir.join("expected_input_features.npy");
    let expected_embeds_path = fixtures_dir.join("expected_audio_embeds.npy");

    println!("== Bundle ==");
    println!("  dir       : {}", bundle_dir.display());
    println!("  precision : {precision_arg}");
    println!("  ep        : {}", ep.unwrap_or("cpu"));

    println!("\n== Audio frontend parity ==");
    let audio = load_wav_mono_16k(&wav_path)?;
    println!(
        "  loaded {} samples ({:.2}s)",
        audio.len(),
        audio.len() as f32 / 16000.0
    );
    let actual = Granite::extract_input_features(&audio)?;
    let expected = read_npy_f32(&expected_features_path)?;
    let actual_shape = [actual.shape()[0], actual.shape()[1], actual.shape()[2]];
    if expected.shape != actual_shape {
        return Err(format!(
            "Frontend shape mismatch: actual {actual_shape:?}, expected {:?}",
            expected.shape
        )
        .into());
    }
    let actual_flat: Vec<f32> = actual.iter().copied().collect();
    let stats = compare_f32(&actual_flat, &expected.data);
    print_stats("  ", &stats);
    if !within_tolerance(&stats, 1e-3, 1e-4) {
        return Err("Frontend parity failed".into());
    }
    println!("  PASS");

    println!("\n== Encoder graph parity ==");
    let load_start = Instant::now();
    let mut plus = GranitePlus::from_pretrained_with_precision(&bundle_dir, precision, exec)?;
    println!(
        "  loaded model in {:.2}s",
        load_start.elapsed().as_secs_f32()
    );
    let enc_start = Instant::now();
    let (encoder_out, n_valid) = plus.run_encoder(&actual)?;
    println!(
        "  encoder ran in {:.2}s, audio_embeds={:?}, n_valid={n_valid}",
        enc_start.elapsed().as_secs_f32(),
        encoder_out.shape()
    );
    let expected_embeds = read_npy_f32(&expected_embeds_path)?;
    let compare_t = expected_embeds.shape[1];
    let enc_flat: Vec<f32> = encoder_out
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
        println!("  PASS");
    } else {
        println!("  INT8: stats only");
    }

    println!("\n== Plus features round-trip ==");

    println!("\n[task: raw]");
    let r = plus.transcribe_audio(&audio, &GranitePlusOptions::transcribe_raw())?;
    println!("  text: {}", r.text);

    println!("\n[task: speaker-attributed]");
    let r = plus.transcribe_audio(&audio, &GranitePlusOptions::speaker_attributed())?;
    println!("  text: {}", r.text);
    println!("  segments: {} (raw: {})", r.segments.len(), r.raw_text);
    for s in &r.segments {
        println!("    Speaker {}: {}", s.speaker, s.text);
    }

    println!("\n[task: word timestamps]");
    let r = plus.transcribe_audio(&audio, &GranitePlusOptions::word_timestamps())?;
    println!("  words: {} (first 5)", r.words.len());
    for w in r.words.iter().take(5) {
        println!(
            "    {:.2}s  {}{}",
            w.end_time,
            w.word,
            if w.is_silence { " [silence]" } else { "" }
        );
    }

    println!("\n[task: keywords]");
    let r = plus.transcribe_audio(
        &audio,
        &GranitePlusOptions::transcribe_with_keywords(["Timothy", "indolently"]),
    )?;
    println!("  text: {}", r.text);

    println!("\n[task: prefix_text incremental]");
    let prefix = "After his nap, Timothy lazily stretched,";
    let r = plus.transcribe_audio(
        &audio,
        &GranitePlusOptions::transcribe_raw().with_prefix_text(prefix),
    )?;
    println!("  prefix: {prefix:?}");
    println!("  text  : {}", r.text);

    println!("\nAll plus checks complete.");
    Ok(())
}

#[cfg(feature = "granite-plus")]
fn parse_task(arg: &str) -> Result<parakeet_rs::GranitePlusOptions, Box<dyn std::error::Error>> {
    use parakeet_rs::GranitePlusOptions;
    if let Some(rest) = arg.strip_prefix("keywords:") {
        let kws: Vec<&str> = rest.split(',').map(str::trim).collect();
        return Ok(GranitePlusOptions::transcribe_with_keywords(kws));
    }
    if let Some(rest) = arg.strip_prefix("prefix:") {
        return Ok(GranitePlusOptions::transcribe_raw().with_prefix_text(rest));
    }
    match arg {
        "raw" | "" => Ok(GranitePlusOptions::transcribe_raw()),
        "saa" => Ok(GranitePlusOptions::speaker_attributed()),
        "ts" => Ok(GranitePlusOptions::word_timestamps()),
        other => Err(format!("Unknown task '{other}'").into()),
    }
}

#[cfg(feature = "granite-plus")]
fn parse_precision(arg: &str) -> Result<parakeet_rs::GranitePrecision, Box<dyn std::error::Error>> {
    use parakeet_rs::GranitePrecision;
    match arg {
        "fp16w" | "" => Ok(GranitePrecision::Fp16w),
        "fp32" => Ok(GranitePrecision::Fp32),
        "int8" => Ok(GranitePrecision::Int8),
        other => Err(format!("Unknown precision '{other}' (expected fp16w/fp32/int8)").into()),
    }
}

#[cfg(feature = "granite-plus")]
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

#[cfg(feature = "granite-plus")]
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

#[cfg(feature = "granite-plus")]
struct Npy {
    shape: [usize; 3],
    data: Vec<f32>,
}

#[cfg(feature = "granite-plus")]
fn read_npy_f32(path: &std::path::Path) -> Result<Npy, Box<dyn std::error::Error>> {
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
    let descr = extract_field(header, "descr").ok_or("missing descr")?;
    if !(descr == "<f4" || descr == "|f4" || descr == "=f4") {
        return Err(format!("expected float32 ('<f4'), got '{descr}'").into());
    }
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
    if dims.len() != 3 {
        return Err(format!("expected 3-D array, got shape {dims:?}").into());
    }
    let n = dims.iter().product::<usize>();
    let mut bytes = vec![0u8; n * 4];
    f.read_exact(&mut bytes)?;
    let data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(Npy {
        shape: [dims[0], dims[1], dims[2]],
        data,
    })
}

#[cfg(feature = "granite-plus")]
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

#[cfg(feature = "granite-plus")]
fn extract_shape(header: &str) -> Option<&str> {
    let i = header.find("'shape':")? + "'shape':".len();
    let rest = header[i..].trim_start();
    let inner = rest.strip_prefix('(')?;
    let end = inner.find(')')?;
    Some(&inner[..end])
}

#[cfg(feature = "granite-plus")]
struct Stats {
    max_abs: f32,
    mean_abs: f32,
    max_rel: f32,
    n: usize,
}

#[cfg(feature = "granite-plus")]
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

#[cfg(feature = "granite-plus")]
fn print_stats(prefix: &str, s: &Stats) {
    println!(
        "{prefix}elements  : {}\n{prefix}max abs   : {:.3e}\n{prefix}mean abs  : {:.3e}\n{prefix}max rel   : {:.3e}",
        s.n, s.max_abs, s.mean_abs, s.max_rel
    );
}

#[cfg(feature = "granite-plus")]
fn within_tolerance(s: &Stats, rtol: f32, atol: f32) -> bool {
    s.max_abs <= atol + rtol
}

#[cfg(not(feature = "granite-plus"))]
fn main() {
    eprintln!("Rebuild with --features granite-plus to run this example.");
    std::process::exit(1);
}
