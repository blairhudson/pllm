//! Bounded local scalar/vector PCF consumer; no reusable secret handle escapes.
use crate::invalid;
use pllm_garble::{secret_replication as native, shared_rescale_reference::Context};
use pyo3::{prelude::*, types::PyBytes};
use serde_json::json;
use std::{sync::OnceLock, time::Instant};
use zeroize::Zeroizing;

fn implementation_digest() -> &'static str {
    static DIGEST: OnceLock<String> = OnceLock::new();
    DIGEST.get_or_init(|| {
        let mut bytes = Vec::new();
        for source in [
            include_bytes!("secret_replication.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/secret_replication.rs").as_slice(),
            include_bytes!("../../../Cargo.lock").as_slice(),
        ] {
            bytes.extend((source.len() as u64).to_le_bytes());
            bytes.extend(source);
        }
        pllm_types::digest_bytes("pllm.secret_replication.implementation.v1", &bytes).to_string()
    })
}

#[pyfunction]
fn secret_replication_benchmark<'py>(
    py: Python<'py>,
    field: u32,
    encoding: &str,
    scalar: u32,
    vector: &[u8],
) -> PyResult<(Bound<'py, PyBytes>, String)> {
    if vector.is_empty() || vector.len() > 4 * native::MAX_LANES || vector.len() % 4 != 0 {
        return Err(invalid(
            "PCF vector must contain 1–4096 little-endian u32 values".into(),
        ));
    }
    let encoding = match encoding {
        "expanded" => native::Encoding::Expanded,
        "punctured" => native::Encoding::Punctured,
        _ => return Err(invalid("unknown PCF encoding".into())),
    };
    let p = native::Parameters {
        field,
        scalar_domain: field,
        lanes: vector.len() / 4,
        encoding,
    };
    p.validate()
        .map_err(|e| invalid(format!("PCF parameters: {e:?}")))?;
    let vector = Zeroizing::new(
        vector
            .chunks_exact(4)
            .map(|raw| u32::from_le_bytes(raw.try_into().expect("checked width")))
            .collect::<Vec<_>>(),
    );
    let scalar = Zeroizing::new(scalar);
    if *scalar >= field || vector.iter().any(|v| *v >= field) {
        return Err(invalid(
            "PCF inputs must be canonical field elements".into(),
        ));
    }
    let (output, metadata) = py.detach(|| -> PyResult<_> {
        let start = Instant::now();
        let expected = Zeroizing::new(vector.iter().map(|v| (u64::from(*v) * u64::from(*scalar) % u64::from(field)) as u32).collect::<Vec<_>>());
        let clear_ns = start.elapsed().as_nanos() as u64;
        let r = native::run_reference(p, Some((*scalar, &vector)), Context {
            plan: [0; 32], session: [0; 32], operation: [1; 32], first_tensor_index: 0,
        }).map_err(|e| invalid(format!("PCF execution: {e:?}")))?;
        let output = r.outputs.ok_or_else(|| invalid("missing PCF outputs".into()))?;
        if output != expected { return Err(invalid("PCF modular product parity failed".into())); }
        let metadata = json!({
            "lanes": p.lanes, "field": field,
            "native_implementation_digest": implementation_digest(), "native_version": env!("CARGO_PKG_VERSION"),
            "party_key_payload_bytes": [r.key_bytes_per_party, r.key_bytes_per_party],
            "expanded_correlation_bytes_per_party": [r.expanded_correlation_bytes_per_party, r.expanded_correlation_bytes_per_party],
            "prf_field_evaluations_per_party": [r.prf_field_evaluations_per_party, r.prf_field_evaluations_per_party],
            "peer_frame_bytes_per_party": [r.peer_bytes_per_party, r.peer_bytes_per_party],
            "protocol_allocation_estimate_bytes": p.allocation_estimate().map_err(|e| invalid(format!("PCF allocation: {e:?}")))?,
            "issuance_seconds": r.issuance_seconds, "expansion_seconds": r.expansion_seconds,
            "online_seconds": r.online_seconds, "clear_ns": clear_ns,
        });
        Ok((Zeroizing::new(output.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()), metadata))
    })?;
    py.check_signals()?;
    Ok((PyBytes::new(py, &output), metadata.to_string()))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(secret_replication_benchmark, module)?)
}
