//! # parakeet-rs
//!
//! Rust bindings for NVIDIA's Parakeet speech recognition model using ONNX Runtime.
//!
//! Parakeet is a state-of-the-art automatic speech recognition (ASR) model developed by NVIDIA,
//! based on the FastConformer-TDT architecture with 600 million parameters.
//!
//! ## Features
//!
//! - Easy-to-use API for speech-to-text transcription
//! - Support for ONNX format models
//! - 16kHz mono audio input
//! - Punctuation and capitalization included in output
//! - Fast inference using ONNX Runtime
//!
//! ## Quick Start
//!
//! ```ignore
//! use parakeet_rs::{Parakeet, Transcriber, TimestampMode};
//!
//! // Load the model
//! let mut parakeet = Parakeet::from_pretrained(".")?;
//!
//! // Transcribe audio samples (see examples/raw.rs for audio loading)
//! let result = parakeet.transcribe_samples(audio, sample_rate, channels, Some(TimestampMode::Words))?;
//! println!("Transcription: {}", result.text);
//! ```
//!
//! ## Model Requirements
//!
//! Your model directory should contain:
//! - `model.onnx` - The ONNX model file
//! - `model.onnx_data` - External model weights
//! - `config.json` - Model configuration
//! - `preprocessor_config.json` - Audio preprocessing configuration
//! - `tokenizer.json` - Tokenizer vocabulary
//! - `tokenizer_config.json` - Tokenizer configuration
//!
//! ## Audio Requirements
//!
//! - Format: WAV
//! - Sample Rate: 16kHz
//! - Channels: Mono (stereo will be converted automatically)
//! - Bit Depth: 16-bit PCM or 32-bit float

mod audio;
#[cfg(feature = "cohere")]
pub mod cohere;
mod config;
#[cfg(any(feature = "cohere", feature = "granite"))]
mod decode_util;
mod decoder;
mod decoder_tdt;
mod error;
mod execution;
mod model;
#[cfg(feature = "cohere")]
mod model_cohere;
mod model_eou;
#[cfg(feature = "multitalker")]
mod model_multitalker;
mod model_nemotron;
mod model_tdt;
mod model_unified;
#[cfg(feature = "multitalker")]
pub mod multitalker;
mod nemotron;
mod parakeet;
mod parakeet_eou;
mod parakeet_tdt;
mod parakeet_unified;
#[cfg(feature = "sortformer")]
pub mod sortformer;
// IBM Granite Speech 4.1 family. The base and plus variants share the same
// graph topology (encoder + prompt_encode + decode_step + embed_tokens) and
// audio frontend, so they live behind a single shared session module. NAR
// runs a different pipeline (encoder + editor + embed_tokens, no KV cache)
// and has its own session wrapper.
#[cfg(any(feature = "granite", feature = "granite-nar"))]
mod audio_granite;
#[cfg(feature = "granite")]
mod decoder_granite;
#[cfg(any(feature = "granite", feature = "granite-nar"))]
mod granite_common;
#[cfg(feature = "granite-nar")]
mod decoder_nar;
#[cfg(feature = "granite")]
pub mod granite;
#[cfg(feature = "granite-nar")]
pub mod granite_nar;
#[cfg(feature = "granite-plus")]
pub mod granite_plus;
#[cfg(feature = "granite-plus")]
mod granite_plus_tags;
#[cfg(any(feature = "granite", feature = "granite-nar"))]
#[cfg_attr(not(feature = "granite"), allow(dead_code))]
pub mod model_granite;
#[cfg(feature = "granite-nar")]
pub mod model_granite_nar;
mod timestamps;
mod transcriber;
mod vocab;

pub use error::{Error, Result};
pub use execution::{ExecutionProvider, ModelConfig as ExecutionConfig};
#[cfg(feature = "coreml")]
pub use ort::ep::coreml::{ComputeUnits as CoreMLComputeUnits, ModelFormat as CoreMLModelFormat};
pub use parakeet::Parakeet;
pub use parakeet_tdt::ParakeetTDT;
pub use timestamps::TimestampMode;
pub use transcriber::*;

pub use config::{ModelConfig as ModelConfigJson, PreprocessorConfig};

pub use decoder::{ParakeetDecoder, TimedToken, TranscriptionResult};
pub use model::ParakeetModel;
pub use model_eou::ParakeetEOUModel;
pub use model_nemotron::{NemotronEncoderCache, NemotronModel, NemotronModelConfig};
pub use model_unified::{ParakeetUnifiedModel, UnifiedModelConfig};
pub use nemotron::{Nemotron, NemotronHandle, SentencePieceVocab};
pub use parakeet_eou::{ParakeetEOU, ParakeetEOUHandle};
pub use parakeet_unified::{ParakeetUnified, ParakeetUnifiedHandle, UnifiedStreamingConfig};

#[cfg(feature = "multitalker")]
pub use multitalker::{
    LatencyMode, MultitalkerASR, MultitalkerConfig, SpeakerTranscript, WordTimestamp,
};

#[cfg(feature = "cohere")]
pub use cohere::CohereASR;

#[cfg(feature = "granite")]
pub use granite::{Granite, GraniteOptions, GraniteTask};
#[cfg(feature = "granite-nar")]
pub use granite_nar::{GraniteNar, GraniteNarOptions};
#[cfg(feature = "granite-plus")]
pub use granite_plus::{
    GranitePlus, GranitePlusOptions, GranitePlusResult, SpeakerSegment, TimedWord,
};
#[cfg(any(feature = "granite", feature = "granite-nar"))]
pub use model_granite::GranitePrecision;
