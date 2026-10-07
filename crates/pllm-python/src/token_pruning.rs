//! Bounded pruning benchmarks; no live material or decoder state crosses PyO3.
use crate::invalid;
use pllm_core::token_pruning::{self as numeric, Parameters};
use pllm_garble::token_pruning as native;
use pyo3::{prelude::*, types::PyBytes};
use serde_json::{json, Value};
use std::{sync::OnceLock, time::Instant};
use zeroize::Zeroizing;

fn implementation_digest() -> &'static str {
    static DIGEST: OnceLock<String> = OnceLock::new();
    DIGEST.get_or_init(|| {
        let mut bytes = Vec::new();
        for source in [
            include_bytes!("token_pruning.rs").as_slice(),
            include_bytes!("../../pllm-core/src/token_pruning.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/token_pruning.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/boolean.rs").as_slice(),
            include_bytes!("../../../Cargo.lock").as_slice(),
        ] {
            bytes.extend((source.len() as u64).to_le_bytes());
            bytes.extend(source);
        }
        pllm_types::digest_bytes("pllm.token_pruning.implementation.v1", &bytes).to_string()
    })
}

fn phase(c: native::Counts) -> Value {
    json!({
        "input_bits": c.input_bits, "output_bits": c.output_bits,
        "and_gates": c.ands, "wires": c.wires(),
        "issuer_key_bytes": c.issuer_bytes(), "evaluator_material_bytes": c.evaluator_bytes(),
        "evaluator_ciphertext_bytes": c.ands * 32, "constant_label_bytes": c.constants * 16,
        "boundary_payload_estimate_bytes": c.boundary_bytes(),
        "public_metadata_bytes_estimate": c.metadata_bytes(),
    })
}

#[pyfunction]
fn token_pruning_projection(
    tokens: usize,
    heads: usize,
    features: usize,
    threshold_q16: u32,
    reduction_threshold_q16: u32,
    retained: usize,
) -> PyResult<String> {
    let p = Parameters {
        tokens,
        heads,
        features,
        threshold_q16,
        reduction_threshold_q16,
    };
    let phases = native::project(p, retained).map_err(invalid)?;
    Ok(json!({
        "phases": phases.map(phase), "retained": retained,
        "adjacent_swaps": if retained == 0 { 0 } else { numeric::swaps(tokens, retained).map_err(invalid)? },
        "protocol_allocation_estimate_bytes": native::allocation_estimate(p).map_err(invalid)?,
        "admitted": native::admit(p).is_ok(),
        "native_implementation_digest": implementation_digest(),
    })
    .to_string())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn token_pruning_benchmark<'py>(
    py: Python<'py>,
    tokens: usize,
    heads: usize,
    features: usize,
    threshold_q16: u32,
    reduction_threshold_q16: u32,
    attention: &[u8],
    words: &[u8],
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, String)> {
    let p = Parameters {
        tokens,
        heads,
        features,
        threshold_q16,
        reduction_threshold_q16,
    };
    native::admit(p).map_err(invalid)?;
    if attention.len() != 4 * p.attention_elements() || words.len() != 4 * tokens * features {
        return Err(invalid(
            "pruning byte inputs differ from their declared shapes".into(),
        ));
    }
    let unpack = |raw: &[u8]| {
        Zeroizing::new(
            raw.chunks_exact(4)
                .map(|v| u32::from_le_bytes(v.try_into().expect("word")))
                .collect::<Vec<_>>(),
        )
    };
    let attention = unpack(attention);
    let words = unpack(words);
    let (output, degrees, metadata) = py.detach(|| -> PyResult<_> {
        let start = Instant::now();
        let expected = numeric::reference(p, &attention, &words).map_err(invalid)?;
        let clear_seconds = start.elapsed().as_secs_f64();
        let measured = native::run_reference(p, &attention, &words).map_err(invalid)?;
        if measured.output != expected { return Err(invalid("pruning integer parity failed".into())); }
        let metadata = json!({
            "tokens": tokens, "heads": heads, "features": features,
            "retained": measured.output.retained,
            "reduction_active": measured.output.high_degree.is_some(),
            "phases": measured.phases.map(phase),
            "protocol_allocation_estimate_bytes": measured.allocation_estimate_bytes,
            "sharing_seconds": measured.sharing_seconds,
            "issuance_seconds": measured.issuance_seconds, "online_seconds": measured.online_seconds,
            "clear_seconds": clear_seconds,
            "native_implementation_digest": implementation_digest(), "native_version": env!("CARGO_PKG_VERSION"),
        });
        Ok((Zeroizing::new(measured.output.words.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()),
            Zeroizing::new(measured.output.high_degree.unwrap_or_default().into_iter().map(u8::from).collect::<Vec<_>>()), metadata))
    })?;
    py.check_signals()?;
    Ok((
        PyBytes::new(py, &output),
        PyBytes::new(py, &degrees),
        metadata.to_string(),
    ))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(token_pruning_projection, module)?)?;
    module.add_function(wrap_pyfunction!(token_pruning_benchmark, module)?)
}
