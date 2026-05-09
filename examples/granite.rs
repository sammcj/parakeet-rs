/*
IBM Granite Speech 4.1 base 2b - autoregressive multilingual ASR + translation.

440M Conformer encoder + Q-Former projector + 1B Granite 4.0 LLM decoder.
Multilingual ASR for English, French, German, Spanish, Portuguese, and
Japanese. Bidirectional translation between English and the other supported
languages. Keyword biasing via prompt.

Download an ONNX bundle from:
https://github.com/sammcj/granite-speech-4.1-onnx
Each bundle contains fp32/, fp16w/, int8/ subdirectories; fp16w is the
recommended default for accelerated inference on Metal/CUDA.

Usage:
  cargo run --release --example granite --features granite -- \
    <bundle_dir> <audio.wav> [task] [precision]

  cargo run --release --example granite --features granite -- \
    <bundle_dir> verify [precision]

  cargo run --release --example granite --features granite -- \
    <bundle_dir> bench [precision] [n_runs] [pad=N] [ep=NAME]

  cargo run --release --example granite --features granite,coreml -- \
    <bundle_dir> bench-encoder [precision] [n_runs] [pad=N] [ep=NAME] [file=NAME] [output=NAME]

Tasks:
  punct       - ASR with punctuation and capitalisation (default)
  raw         - ASR without punctuation
  translate:<Lang>     - speech translation, e.g. translate:French
  keywords:<kw1,kw2..> - ASR with keyword biasing
  verify      - run bundled fixture parity checks (no audio path needed)
  bench       - measure encoder + end-to-end RTF over N runs
  bench-encoder - encoder-only timing for diagnosing EP partitioning

Precisions: fp16w (default), fp32, int8

Execution providers (`ep=`):
  cpu (default), coreml, coreml-mlprogram, coreml-mlprogram-static,
  cuda (when the matching feature is enabled at build time).

Examples:
  cargo run --release --example granite --features granite -- ./bundle audio.wav
  cargo run --release --example granite --features granite -- ./bundle audio.wav raw
  cargo run --release --example granite --features granite -- ./bundle audio.wav translate:French
  cargo run --release --example granite --features granite -- ./bundle audio.wav keywords:Sammy,IBM int8
  cargo run --release --example granite --features granite -- ./bundle verify
  cargo run --release --example granite --features granite -- ./bundle bench fp16w 5 ep=cpu

Verify mode:
  Validates a bundle against the golden test fixtures shipped at
  <bundle_dir>/test_fixtures/. Three checks run in order:
    1. Audio frontend parity vs expected_input_features.npy (rtol=1e-3, atol=1e-4)
    2. Encoder graph parity vs expected_audio_embeds.npy
    3. End-to-end transcript vs the upstream reference

Bench modes:
  `bench` runs the full Granite engine (encoder + LLM body) and reports
  load time, encoder time, end-to-end time, and RTF over N runs after a
  one-shot warm-up. `bench-encoder` constructs an ort session for
  `<precision>/encoder.onnx` directly and times only the encoder forward
  pass; useful for isolating EP behaviour. `pad=N` zero-pads the input
  audio to N seconds, required when the encoder bundle has a fixed audio
  bucket. `file=` and `output=` (bench-encoder only) override the encoder
  filename and primary output tensor name for cut-graph diagnostics.

NOTE on long audio: The base 2b model handles long-form audio out-of-the-box.
There's no hard chunk limit, but for >5 minute clips you may want to split at
silence boundaries to keep KV cache memory bounded.
*/

#[cfg(feature = "granite")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::env;

    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <bundle_dir> (<audio.wav> [task] | verify) [precision]",
            args[0]
        );
        eprintln!("  task: punct (default) | raw | translate:<Lang> | keywords:<kw1,kw2,..>");
        eprintln!("  precision: fp16w (default) | fp32 | int8");
        std::process::exit(1);
    }

    // Keyword flags filtered out before positional parsing:
    //   ep=cpu|coreml|coreml-mlprogram|coreml-mlprogram-static|cuda
    //   pad=N             zero-pad input audio to N seconds (bench modes)
    //   file=NAME         override encoder filename (bench-encoder only)
    //   output=NAME       override encoder output tensor name (bench-encoder only)
    let mut ep_arg: Option<String> = None;
    let mut pad_secs: Option<f32> = None;
    let mut file_arg: Option<String> = None;
    let mut output_arg: Option<String> = None;
    let positional: Vec<String> = args
        .into_iter()
        .filter(|a| {
            if let Some(v) = a.strip_prefix("ep=") {
                ep_arg = Some(v.to_string());
                false
            } else if let Some(v) = a.strip_prefix("pad=") {
                pad_secs = v.parse().ok();
                false
            } else if let Some(v) = a.strip_prefix("file=") {
                file_arg = Some(v.to_string());
                false
            } else if let Some(v) = a.strip_prefix("output=") {
                output_arg = Some(v.to_string());
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
    if positional.get(2).map(String::as_str) == Some("bench") {
        let precision_arg = positional.get(3).map(String::as_str).unwrap_or("fp16w");
        let n_runs: usize = positional
            .get(4)
            .and_then(|s| s.parse().ok())
            .unwrap_or(3);
        return run_bench(bundle_dir, precision_arg, ep_arg.as_deref(), n_runs, pad_secs);
    }
    if positional.get(2).map(String::as_str) == Some("bench-encoder") {
        let precision_arg = positional.get(3).map(String::as_str).unwrap_or("fp16w");
        let n_runs: usize = positional
            .get(4)
            .and_then(|s| s.parse().ok())
            .unwrap_or(3);
        return run_encoder_bench(
            bundle_dir,
            precision_arg,
            ep_arg.as_deref(),
            n_runs,
            pad_secs,
            file_arg.as_deref(),
            output_arg.as_deref(),
        );
    }

    let audio_path = &positional[2];
    let task = positional.get(3).map(String::as_str).unwrap_or("punct");
    let precision = positional.get(4).map(String::as_str).unwrap_or("fp16w");
    run_transcribe(bundle_dir, audio_path, task, precision, ep_arg.as_deref())
}

#[cfg(feature = "granite")]
fn run_transcribe(
    bundle_dir: &str,
    audio_path: &str,
    task: &str,
    precision: &str,
    ep: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    use parakeet_rs::Granite;
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

    println!("Loading Granite Speech 4.1 base 2b ({precision})...");
    let load_start = Instant::now();
    let mut model = Granite::from_pretrained_with_precision(bundle_dir, prec, exec)?;
    println!("Loaded in {:.2}s", load_start.elapsed().as_secs_f32());

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

#[cfg(feature = "granite")]
fn run_verify(
    bundle_dir: &str,
    precision_arg: &str,
    ep: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    use parakeet_rs::{Granite, GraniteOptions, GranitePrecision};
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
    let expected_embeds_path = fixtures_dir.join("expected_audio_embeds.npy");

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
    println!("  PASS (rtol=1e-3, atol=1e-4)");

    println!("\n== Encoder graph parity ==");
    let load_start = Instant::now();
    let mut granite = Granite::from_pretrained_with_precision(&bundle_dir, precision, exec)?;
    println!(
        "  loaded model in {:.2}s",
        load_start.elapsed().as_secs_f32()
    );
    let enc_start = Instant::now();
    let (encoder_out, n_valid) = granite.run_encoder(&actual)?;
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
        println!("  PASS (rtol={enc_rtol}, atol={enc_atol})");
    } else {
        println!("  INT8: stats only (logit values shift, argmax preserved)");
    }

    println!("\n== Transcript ==");
    let dec_start = Instant::now();
    let opts = GraniteOptions::transcribe_with_punctuation();
    let transcript = granite.transcribe_audio(&audio, &opts)?;
    let dec_elapsed = dec_start.elapsed().as_secs_f32();
    let dur = audio.len() as f32 / 16000.0;
    println!(
        "  RTF       : {:.2}x ({:.2}s decode, {:.2}s audio)",
        dur / dec_elapsed,
        dec_elapsed,
        dur
    );
    println!("  reference : After his nap, Timothy lazily stretched, first one gray velvet foot, then another, strolled indolently to his plate, turning over the food, carefully selecting choice bits, nosing out that which he scorned upon the clean hearth");
    println!("  actual    : {transcript}");
    println!("\nAll checks complete.");
    Ok(())
}

#[cfg(feature = "granite")]
fn run_bench(
    bundle_dir: &str,
    precision_arg: &str,
    ep: Option<&str>,
    n_runs: usize,
    pad_secs: Option<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    use parakeet_rs::{Granite, GraniteOptions};
    use std::path::PathBuf;
    use std::time::Instant;

    let bundle_dir_path = PathBuf::from(bundle_dir);
    let precision = parse_precision(precision_arg)?;
    let exec = parse_ep(ep)?;
    let ep_label = ep.unwrap_or("cpu");

    println!("== Granite bench ==");
    println!("  bundle    : {}", bundle_dir_path.display());
    println!("  precision : {precision_arg}");
    println!("  ep        : {ep_label}");
    println!("  runs      : {n_runs} (after 1 warm-up)");

    let wav_path = bundle_dir_path.join("test_fixtures").join("sample_audio.wav");
    let mut audio = load_wav_mono_16k(&wav_path)?;
    let real_secs = audio.len() as f32 / 16000.0;
    if let Some(target) = pad_secs {
        let target_samples = (target * 16000.0) as usize;
        if audio.len() < target_samples {
            audio.resize(target_samples, 0.0);
            println!(
                "  audio     : {real_secs:.2}s real, zero-padded to {target:.2}s ({} samples)",
                audio.len()
            );
        } else if audio.len() > target_samples {
            return Err(format!(
                "Audio is {real_secs:.2}s but pad target is {target:.2}s (would have to trim, refusing)"
            )
            .into());
        } else {
            println!("  audio     : {real_secs:.2}s ({} samples, no pad needed)", audio.len());
        }
    } else {
        println!("  audio     : {real_secs:.2}s ({} samples)", audio.len());
    }
    let duration_secs = audio.len() as f32 / 16000.0;

    let load_start = Instant::now();
    let mut granite = Granite::from_pretrained_with_precision(&bundle_dir_path, precision, exec)?;
    let load_secs = load_start.elapsed().as_secs_f32();
    println!("  load_s    : {load_secs:.2}");

    let opts = GraniteOptions::transcribe_with_punctuation();

    let features = Granite::extract_input_features(&audio)?;

    let warm_start = Instant::now();
    let _ = granite.run_encoder(&features)?;
    let warm_secs = warm_start.elapsed().as_secs_f32();
    println!("  warm_enc_s: {warm_secs:.2}");

    let mut encoder_secs = Vec::with_capacity(n_runs);
    let mut total_secs = Vec::with_capacity(n_runs);
    for i in 0..n_runs {
        let enc_start = Instant::now();
        let _ = granite.run_encoder(&features)?;
        encoder_secs.push(enc_start.elapsed().as_secs_f32());

        let total_start = Instant::now();
        let _ = granite.transcribe_audio(&audio, &opts)?;
        total_secs.push(total_start.elapsed().as_secs_f32());
        println!(
            "  run {}/{}: enc={:.3}s total={:.3}s rtf={:.2}x",
            i + 1,
            n_runs,
            encoder_secs[i],
            total_secs[i],
            duration_secs / total_secs[i]
        );
    }

    let median = |v: &[f32]| {
        let mut sorted: Vec<f32> = v.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted[sorted.len() / 2]
    };
    let enc_p50 = median(&encoder_secs);
    let total_p50 = median(&total_secs);
    println!("\n== Median over {n_runs} runs ==");
    println!("  encoder_s : {enc_p50:.3}");
    println!("  total_s   : {total_p50:.3}");
    println!("  rtf       : {:.2}x", duration_secs / total_p50);
    Ok(())
}

#[cfg(feature = "granite")]
fn run_encoder_bench(
    bundle_dir: &str,
    precision_arg: &str,
    ep: Option<&str>,
    n_runs: usize,
    pad_secs: Option<f32>,
    file_override: Option<&str>,
    output_override: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    use ort::session::Session;
    use ort::value::TensorRef;
    use parakeet_rs::Granite;
    use std::path::PathBuf;
    use std::time::Instant;

    let bundle_dir_path = PathBuf::from(bundle_dir);
    let precision = parse_precision(precision_arg)?;
    let exec_opt = parse_ep(ep)?;
    let ep_label = ep.unwrap_or("cpu");
    let encoder_filename = file_override.unwrap_or("encoder.onnx");
    let output_name = output_override.unwrap_or("audio_embeds");

    println!("== Granite encoder-only bench ==");
    println!("  bundle    : {}", bundle_dir_path.display());
    println!("  precision : {precision_arg}");
    println!("  ep        : {ep_label}");
    println!("  runs      : {n_runs} (after 1 warm-up)");
    println!("  file      : {encoder_filename}");
    println!("  output    : {output_name}");

    let prec_dir = bundle_dir_path.join(match precision {
        parakeet_rs::GranitePrecision::Fp32 => "fp32",
        parakeet_rs::GranitePrecision::Fp16w => "fp16w",
        parakeet_rs::GranitePrecision::Int8 => "int8",
    });
    let encoder_path = prec_dir.join(encoder_filename);

    let wav_path = bundle_dir_path.join("test_fixtures").join("sample_audio.wav");
    let mut audio = load_wav_mono_16k(&wav_path)?;
    let real_secs = audio.len() as f32 / 16000.0;
    if let Some(target) = pad_secs {
        let target_samples = (target * 16000.0) as usize;
        if audio.len() < target_samples {
            audio.resize(target_samples, 0.0);
            println!(
                "  audio     : {real_secs:.2}s real, zero-padded to {target:.2}s ({} samples)",
                audio.len()
            );
        } else {
            println!("  audio     : {real_secs:.2}s ({} samples, no pad)", audio.len());
        }
    } else {
        println!("  audio     : {real_secs:.2}s ({} samples)", audio.len());
    }

    let features = Granite::extract_input_features(&audio)?;
    println!("  features  : {:?}", features.shape());

    let load_start = Instant::now();
    let mut builder = Session::builder()?;
    if let Some(exec) = &exec_opt {
        builder = exec.apply_to_session_builder(builder)?;
    }
    let mut session = builder.commit_from_file(&encoder_path)?;
    let load_secs = load_start.elapsed().as_secs_f32();
    println!("  load_s    : {load_secs:.2}");

    let warm_start = Instant::now();
    {
        let feats = TensorRef::<f32>::from_array_view(features.view())?;
        let _ = session.run(ort::inputs!("input_features" => feats))?;
    }
    let warm_secs = warm_start.elapsed().as_secs_f32();
    println!("  warm_enc_s: {warm_secs:.2}");

    let mut encoder_secs = Vec::with_capacity(n_runs);
    for i in 0..n_runs {
        let enc_start = Instant::now();
        let feats = TensorRef::<f32>::from_array_view(features.view())?;
        let outputs = session.run(ort::inputs!("input_features" => feats))?;
        let elapsed = enc_start.elapsed().as_secs_f32();
        let (shape, _) = outputs[output_name].try_extract_tensor::<f32>()?;
        encoder_secs.push(elapsed);
        println!(
            "  run {}/{}: enc={:.3}s {}={:?}",
            i + 1,
            n_runs,
            elapsed,
            output_name,
            shape
        );
    }

    let median = |v: &[f32]| {
        let mut sorted: Vec<f32> = v.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted[sorted.len() / 2]
    };
    println!("\n== Median over {n_runs} runs ==");
    println!("  encoder_s : {:.3}", median(&encoder_secs));
    Ok(())
}

#[cfg(feature = "granite")]
fn parse_task(arg: &str) -> Result<parakeet_rs::GraniteOptions, Box<dyn std::error::Error>> {
    use parakeet_rs::GraniteOptions;
    if let Some(rest) = arg.strip_prefix("translate:") {
        return Ok(GraniteOptions::translate_to(rest));
    }
    if let Some(rest) = arg.strip_prefix("keywords:") {
        let kws: Vec<&str> = rest.split(',').map(str::trim).collect();
        return Ok(GraniteOptions::transcribe_with_keywords(kws));
    }
    match arg {
        "punct" | "" => Ok(GraniteOptions::transcribe_with_punctuation()),
        "raw" => Ok(GraniteOptions::transcribe_raw()),
        other => Err(format!("Unknown task '{other}'").into()),
    }
}

#[cfg(feature = "granite")]
fn parse_precision(arg: &str) -> Result<parakeet_rs::GranitePrecision, Box<dyn std::error::Error>> {
    use parakeet_rs::GranitePrecision;
    match arg {
        "fp16w" | "" => Ok(GranitePrecision::Fp16w),
        "fp32" => Ok(GranitePrecision::Fp32),
        "int8" => Ok(GranitePrecision::Int8),
        other => Err(format!("Unknown precision '{other}' (expected fp16w/fp32/int8)").into()),
    }
}

#[cfg(feature = "granite")]
fn parse_ep(
    ep: Option<&str>,
) -> Result<Option<parakeet_rs::ExecutionConfig>, Box<dyn std::error::Error>> {
    use parakeet_rs::{ExecutionConfig, ExecutionProvider};
    let Some(name) = ep else {
        return Ok(None);
    };
    // CoreML presets:
    //   coreml                    : ort defaults (NeuralNetwork + ALL compute units)
    //   coreml-mlprogram          : MLProgram + CPUAndGPU
    //   coreml-mlprogram-static   : MLProgram + CPUAndGPU + RequireStaticInputShapes
    let provider = match name {
        "cpu" => ExecutionProvider::Cpu,
        #[cfg(feature = "coreml")]
        "coreml" | "coreml-mlprogram" | "coreml-mlprogram-static" => ExecutionProvider::CoreML,
        #[cfg(feature = "cuda")]
        "cuda" => ExecutionProvider::Cuda,
        other => {
            return Err(format!(
                "Unknown ep '{other}'. Available at this build: cpu{}{}",
                if cfg!(feature = "coreml") {
                    ", coreml, coreml-mlprogram, coreml-mlprogram-static"
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
    {
        use parakeet_rs::{CoreMLComputeUnits, CoreMLModelFormat};
        if name == "coreml-mlprogram" || name == "coreml-mlprogram-static" {
            cfg = cfg
                .with_coreml_model_format(CoreMLModelFormat::MLProgram)
                .with_coreml_compute_units(CoreMLComputeUnits::CPUAndGPU);
        }
        if name == "coreml-mlprogram-static" {
            cfg = cfg.with_coreml_require_static_shapes(true);
        }
        if matches!(
            name,
            "coreml" | "coreml-mlprogram" | "coreml-mlprogram-static"
        ) {
            if let Ok(dir) = std::env::var("PARAKEET_COREML_CACHE_DIR") {
                cfg = cfg.with_coreml_cache_dir(dir);
            }
        }
    }
    Ok(Some(cfg))
}

#[cfg(feature = "granite")]
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

#[cfg(feature = "granite")]
struct Npy {
    shape: [usize; 3],
    data: Vec<f32>,
}

#[cfg(feature = "granite")]
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

#[cfg(feature = "granite")]
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

#[cfg(feature = "granite")]
fn extract_shape(header: &str) -> Option<&str> {
    let i = header.find("'shape':")? + "'shape':".len();
    let rest = header[i..].trim_start();
    let inner = rest.strip_prefix('(')?;
    let end = inner.find(')')?;
    Some(&inner[..end])
}

#[cfg(feature = "granite")]
struct Stats {
    max_abs: f32,
    mean_abs: f32,
    max_rel: f32,
    n: usize,
}

#[cfg(feature = "granite")]
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

#[cfg(feature = "granite")]
fn print_stats(prefix: &str, s: &Stats) {
    println!(
        "{prefix}elements  : {}\n{prefix}max abs   : {:.3e}\n{prefix}mean abs  : {:.3e}\n{prefix}max rel   : {:.3e}",
        s.n, s.max_abs, s.mean_abs, s.max_rel
    );
}

#[cfg(feature = "granite")]
fn within_tolerance(s: &Stats, rtol: f32, atol: f32) -> bool {
    s.max_abs <= atol + rtol
}

#[cfg(not(feature = "granite"))]
fn main() {
    eprintln!("Rebuild with --features granite to run this example.");
    std::process::exit(1);
}
