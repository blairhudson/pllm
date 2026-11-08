use super::*;
use pllm_compiler::fss_decoder::{AdmissionContext, AdmissionFailure, AdmissionGate, AdmissionStage, Error};
use pyo3::exceptions::PyValueError;

#[test]
fn contextual_errors_keep_classification_and_bounded_public_labels() {
    Python::initialize();
    Python::attach(|py| {
        let mut context = AdmissionContext::new(AdmissionStage::RuntimePreflight, AdmissionGate::KernelConstruction);
        context.node = Some(usize::MAX);
        context.kernel = Some(usize::MAX);
        context.required = Some(usize::MAX);
        context.used = Some(usize::MAX);
        context.limit = Some(usize::MAX);
        for error in [Error::Resource, Error::Domain, Error::Consumed] {
            let plain = native_error(error);
            let contextual = native_error_context(AdmissionFailure::from(error).with_context_if_absent(context));
            assert!(plain.get_type(py).is(&contextual.get_type(py)));
            if error == Error::Resource { assert!(contextual.is_instance_of::<PyValueError>(py)); }
            let text = contextual.value(py).str().unwrap().to_str().unwrap().to_owned();
            assert!(text.starts_with(&format!("FuseFSS native admission failed: {error:?}; ")));
            assert!(text.contains("stage=runtime_preflight gate=kernel_construction"));
            assert!(text.len() < 400);
        }
        let unknown = native_error_context(AdmissionFailure::from(Error::Resource)
            .with_context_if_absent(AdmissionContext::new(AdmissionStage::RuntimePreflight, AdmissionGate::KernelConstruction)));
        let text = unknown.value(py).str().unwrap().to_str().unwrap().to_owned();
        assert!(!text.contains("required="));
        assert!(!text.contains("used="));
        assert!(!text.contains("limit="));
    });
}

#[test]
fn failed_contextual_tensor_seal_retires_builder_without_retry() {
    Python::initialize();
    Python::attach(|py| {
        let config = pllm_types::canonical_bytes(&json!({
            "model_type":"gpt2", "n_embd":2, "n_head":1, "n_layer":1,
            "n_positions":4, "vocab_size":3, "n_inner":2
        }));
        let mut builder = NativeSourceBuilder {
            owner: Some(SourceBuilder::new(&config, 1, 2, sdk::GPT2_NUMERIC).unwrap()),
            pid: std::process::id(),
        };
        let failure = builder.seal_tensor(py).err().unwrap();
        assert!(failure.is_instance_of::<PyValueError>(py));
        assert!(failure.value(py).str().unwrap().to_str().unwrap().contains("stage=source_seal gate=source_assembly"));
        assert!(builder.owner.is_none());
        let retry = builder.seal_tensor(py).err().unwrap();
        assert_eq!(retry.value(py).str().unwrap().to_str().unwrap(), "FuseFSS source builder retired");
        assert!(builder.owner.is_none());
    });
}
