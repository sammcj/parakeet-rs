use std::path::PathBuf;
use std::{fmt, rc::Rc};

use crate::error::Result;
use ort::session::builder::SessionBuilder;

// Hardware acceleration options. CPU is default and most reliable.
// GPU providers (CUDA, TensorRT, MIGraphX) offer 5-10x speedup but require specific hardware.
// All GPU providers automatically fall back to CPU if they fail.
//
// Note: CoreML EP currently runs slower than CPU for Sortformer/Parakeet models because
// the ONNX graphs have dynamic input shapes, preventing CoreML from building optimised
// execution plans for ANE/GPU. CoreML claims nodes but runs them on CPU with overhead.
//
// WebGPU is experimental and may produce incorrect results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionProvider {
    #[default]
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "tensorrt")]
    TensorRT,
    #[cfg(feature = "coreml")]
    CoreML,
    #[cfg(feature = "directml")]
    DirectML,
    #[cfg(feature = "migraphx")]
    MIGraphX,
    #[cfg(feature = "openvino")]
    OpenVINO,
    #[cfg(feature = "webgpu")]
    WebGPU,
    #[cfg(feature = "nnapi")]
    NNAPI,
}

#[derive(Clone)]
pub struct ModelConfig {
    pub execution_provider: ExecutionProvider,
    pub intra_threads: usize,
    pub inter_threads: usize,
    pub configure: Option<Rc<dyn Fn(SessionBuilder) -> ort::Result<SessionBuilder>>>,
    /// Optional cache directory for compiled CoreML models. When set, avoids
    /// recompiling the ONNX-to-CoreML conversion on each session load (~5s).
    /// Only used when execution_provider is CoreML.
    pub coreml_cache_dir: Option<PathBuf>,
    /// CoreML model format. `MLProgram` (Core ML 5+, macOS 12+) handles
    /// fp16 weights and large graphs much better than the rc.12 default
    /// (`NeuralNetwork`). For the Granite Speech 4.1 2b graphs in
    /// particular, switching to `MLProgram` is the difference between
    /// the CoreML compiler OOM'ing and producing a working session.
    /// `None` leaves the rc.12 default in place. Only used when
    /// execution_provider is CoreML.
    #[cfg(feature = "coreml")]
    pub coreml_model_format: Option<ort::ep::coreml::ModelFormat>,
    /// CoreML compute units. `CPUAndGPU` skips the ANE compile pass,
    /// which is the largest memory consumer when CoreML is loading the
    /// 1B Granite LLM body. Defaults to `CPUAndGPU` if `None`. Only
    /// used when execution_provider is CoreML.
    #[cfg(feature = "coreml")]
    pub coreml_compute_units: Option<ort::ep::coreml::ComputeUnits>,
    /// If `true`, restrict the CoreML EP to claim only nodes with
    /// statically known shapes. Opset-20 dynamic shapes stay on the
    /// CPU fallback. Useful for diagnosing partition behaviour or in
    /// combination with a static-shape re-export of the model. Only
    /// used when execution_provider is CoreML.
    #[cfg(feature = "coreml")]
    pub coreml_require_static_shapes: bool,
}

impl fmt::Debug for ModelConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelConfig")
            .field("execution_provider", &self.execution_provider)
            .field("intra_threads", &self.intra_threads)
            .field("inter_threads", &self.inter_threads)
            .field(
                "configure",
                &if self.configure.is_some() {
                    "<fn>"
                } else {
                    "None"
                },
            )
            .field("coreml_cache_dir", &self.coreml_cache_dir)
            .finish()
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            execution_provider: ExecutionProvider::default(),
            intra_threads: 4,
            inter_threads: 1,
            configure: None,
            coreml_cache_dir: None,
            #[cfg(feature = "coreml")]
            coreml_model_format: None,
            #[cfg(feature = "coreml")]
            coreml_compute_units: None,
            #[cfg(feature = "coreml")]
            coreml_require_static_shapes: false,
        }
    }
}

impl ModelConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_execution_provider(mut self, provider: ExecutionProvider) -> Self {
        self.execution_provider = provider;
        self
    }

    pub fn with_intra_threads(mut self, threads: usize) -> Self {
        self.intra_threads = threads;
        self
    }

    pub fn with_inter_threads(mut self, threads: usize) -> Self {
        self.inter_threads = threads;
        self
    }

    pub fn with_custom_configure(
        mut self,
        configure: impl Fn(SessionBuilder) -> ort::Result<SessionBuilder> + 'static,
    ) -> Self {
        self.configure = Some(Rc::new(configure));
        self
    }

    /// Set cache directory for compiled CoreML models.
    /// Avoids ~5s recompilation on each session load.
    pub fn with_coreml_cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.coreml_cache_dir = Some(path.into());
        self
    }

    /// Override the CoreML model format. `MLProgram` (Core ML 5+,
    /// macOS 12+) is the recommended setting for the Granite Speech
    /// 4.1 2b graphs - the rc.12 default (`NeuralNetwork`) hits
    /// CoreML compiler OOM on consumer Macs at this scale.
    #[cfg(feature = "coreml")]
    pub fn with_coreml_model_format(mut self, fmt: ort::ep::coreml::ModelFormat) -> Self {
        self.coreml_model_format = Some(fmt);
        self
    }

    /// Override CoreML compute units. Defaults to `CPUAndGPU` which
    /// skips the ANE compile pass; use `All` only on small graphs that
    /// fit comfortably in ANE memory.
    #[cfg(feature = "coreml")]
    pub fn with_coreml_compute_units(mut self, units: ort::ep::coreml::ComputeUnits) -> Self {
        self.coreml_compute_units = Some(units);
        self
    }

    /// Restrict the CoreML EP to nodes with statically known shapes.
    /// Opset-20 dynamic shapes stay on the CPU fallback. Combine with
    /// a static-shape re-export of the model for the cleanest CoreML
    /// partition.
    #[cfg(feature = "coreml")]
    pub fn with_coreml_require_static_shapes(mut self, enable: bool) -> Self {
        self.coreml_require_static_shapes = enable;
        self
    }

    /// Apply this config to an `ort` session builder. Used by the engine
    /// types when constructing sessions; exposed publicly so callers
    /// building bare `ort::session::Session`s can reuse the same EP option
    /// resolution.
    pub fn apply_to_session_builder(
        &self,
        builder: SessionBuilder,
    ) -> Result<SessionBuilder> {
        #[cfg(any(
            feature = "cuda",
            feature = "tensorrt",
            feature = "coreml",
            feature = "directml",
            feature = "migraphx",
            feature = "openvino",
            feature = "webgpu",
            feature = "nnapi"
        ))]
        use ort::ep::CPU as CPUExecutionProvider;
        use ort::session::builder::GraphOptimizationLevel;

        let mut builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(self.intra_threads)?
            .with_inter_threads(self.inter_threads)?;

        builder = match self.execution_provider {
            ExecutionProvider::Cpu => builder,

            #[cfg(feature = "cuda")]
            ExecutionProvider::Cuda => builder.with_execution_providers([
                ort::ep::CUDA::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "tensorrt")]
            ExecutionProvider::TensorRT => builder.with_execution_providers([
                ort::ep::TensorRT::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "coreml")]
            ExecutionProvider::CoreML => {
                use ort::ep::coreml::{ComputeUnits, CoreML};

                let units = self.coreml_compute_units.unwrap_or(ComputeUnits::CPUAndGPU);
                let mut coreml = CoreML::default().with_compute_units(units);

                if let Some(fmt) = self.coreml_model_format {
                    coreml = coreml.with_model_format(fmt);
                }
                if self.coreml_require_static_shapes {
                    coreml = coreml.with_static_input_shapes(true);
                }
                if let Some(cache_dir) = &self.coreml_cache_dir {
                    coreml = coreml.with_model_cache_dir(cache_dir.to_string_lossy());
                }

                builder.with_execution_providers([
                    coreml.build(),
                    CPUExecutionProvider::default().build().error_on_failure(),
                ])?
            }

            #[cfg(feature = "directml")]
            ExecutionProvider::DirectML => builder.with_execution_providers([
                ort::ep::DirectML::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "migraphx")]
            ExecutionProvider::MIGraphX => builder.with_execution_providers([
                ort::ep::MIGraphX::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "openvino")]
            ExecutionProvider::OpenVINO => builder.with_execution_providers([
                ort::ep::OpenVINO::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "webgpu")]
            ExecutionProvider::WebGPU => builder.with_execution_providers([
                ort::ep::WebGPU::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "nnapi")]
            ExecutionProvider::NNAPI => builder.with_execution_providers([
                ort::ep::NNAPI::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,
        };

        if let Some(configure) = self.configure.as_ref() {
            builder = configure(builder)?;
        }

        Ok(builder)
    }
}
