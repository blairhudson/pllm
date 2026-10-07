//! Complete one-use garbled lookup/product operator; only final bytes escape.
use crate::invalid;
use pllm_core::wavelet_gated_reference::Profile;
use pllm_garble::logrow_product as native;
use pyo3::{prelude::*, types::PyBytes};
use serde_json::json;
use std::{sync::OnceLock, time::Instant};
use zeroize::Zeroizing;

fn implementation_digest() -> &'static str {
    static DIGEST: OnceLock<String> = OnceLock::new();
    DIGEST.get_or_init(|| {
        let sources = [
            include_bytes!("logrow_product.rs").as_slice(),
            include_bytes!("../../pllm-core/src/wavelet_gated_reference.rs").as_slice(),
            include_bytes!("../../pllm-core/src/fixed_point.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/logrow_product.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/logrow.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/boolean.rs").as_slice(),
            include_bytes!("../../../Cargo.lock").as_slice(),
        ];
        let mut bytes = Vec::new();
        for source in sources {
            bytes.extend((source.len() as u64).to_le_bytes());
            bytes.extend(source);
        }
        pllm_types::digest_bytes("pllm.logrow_product_benchmark.implementation.v1", &bytes)
            .to_string()
    })
}

#[pyfunction]
fn logrow_product_benchmark<'py>(
    py: Python<'py>,
    gate_q18: &[u8],
    up_q18: &[u8],
) -> PyResult<(Bound<'py, PyBytes>, String)> {
    if gate_q18.is_empty()
        || gate_q18.len() > 256 * 4
        || gate_q18.len() % 4 != 0
        || gate_q18.len() != up_q18.len()
    {
        return Err(invalid(
            "LogRow product requires matching bounded signed-i32 frames".into(),
        ));
    }
    let arrays = [gate_q18, up_q18].map(|raw| {
        Zeroizing::new(
            raw.chunks_exact(4)
                .map(|b| i32::from_le_bytes(b.try_into().expect("validated word width")))
                .collect::<Vec<_>>(),
        )
    });
    for (&g, &u) in arrays[0].iter().zip(arrays[1].iter()) {
        Profile::validate_inputs(g, u)
            .map_err(|e| invalid(format!("LogRow product domain: {e:?}")))?;
    }
    let (output, metadata) = py.detach(|| -> PyResult<_> {
        let start = Instant::now();
        let profile = Profile::new(0).map_err(|e| invalid(format!("LogRow profile: {e:?}")))?;
        let setup_ns = start.elapsed().as_nanos() as u64;
        let start = Instant::now();
        let expected = Zeroizing::new(arrays[0].iter().zip(arrays[1].iter()).map(|(&g, &u)|
            profile.evaluate(g, u).map_err(|e| invalid(format!("LogRow oracle: {e:?}"))))
            .collect::<PyResult<Vec<_>>>()?);
        let clear_ns = start.elapsed().as_nanos() as u64;
        let mut reference = None;
        let mut output = Zeroizing::new(Vec::with_capacity(gate_q18.len()));
        let mut issuance_ns = 0_u64;
        let mut online_ns = 0_u64;
        // Serial retirement: a 256-lane benchmark never owns 256 live circuits.
        for ((&g, &u), &expected) in arrays[0].iter().zip(arrays[1].iter()).zip(expected.iter()) {
            let start = Instant::now();
            let (client, program) = native::prepare_reference(&profile).map_err(invalid)?;
            issuance_ns += start.elapsed().as_nanos() as u64;
            let resource = program.resources().map_err(invalid)?;
            if reference.is_some_and(|r| r != resource) {
                return Err(invalid("LogRow topology depends on private input".into()));
            }
            reference = Some(resource);
            let start = Instant::now();
            let (input, decoder) = client.encode(g, u).map_err(invalid)?;
            let result = decoder.decode(program.evaluate(input).map_err(invalid)?).map_err(invalid)?;
            online_ns += start.elapsed().as_nanos() as u64;
            if result != expected { return Err(invalid("LogRow complete integer parity failed".into())); }
            output.extend(result.to_le_bytes());
        }
        let r = reference.expect("validated nonzero lanes");
        let lanes = arrays[0].len();
        let parts = [
            ("client_encodings", lanes * r.client_encoding_bytes, 0),
            ("logrow_lookup", 0, lanes * r.evaluator_lookup_bytes),
            ("up_rescale", 0, lanes * r.arithmetic_bytes[0]),
            ("multiply", 0, lanes * r.arithmetic_bytes[1]),
            ("output_rescale_sign", 0, lanes * r.arithmetic_bytes[2]),
            ("constant_labels", 0, lanes * r.evaluator_constant_bytes),
        ];
        let metadata = json!({
            "lanes": lanes, "profile_digest": pllm_types::Digest::from_sha256(profile.digest()),
            "party_key_payload_bytes": [lanes * r.client_encoding_bytes,
                lanes * (r.evaluator_half_gate_bytes + r.evaluator_lookup_bytes + r.evaluator_constant_bytes)],
            "key_payload_bytes_by_operation": parts,
            "evaluator_ciphertext_bytes": lanes * r.evaluator_half_gate_bytes,
            "evaluator_constant_label_bytes": lanes * r.evaluator_constant_bytes,
            "public_metadata_bytes_estimate": lanes * r.public_metadata_bytes_estimate,
            "client_entry_exit_payload_estimate_bytes": lanes * (r.input_output_label_bytes + 64),
            "protocol_allocation_estimate_bytes": native::MAX_ALLOCATION_BYTES,
            "and_gates_per_lane": r.and_gates, "wires_per_lane": r.wires,
            "native_implementation_digest": implementation_digest(), "native_version": env!("CARGO_PKG_VERSION"),
            "public_setup_ns": setup_ns, "issuance_ns": issuance_ns,
            "online_ns": online_ns, "clear_ns": clear_ns,
        });
        Ok((output, metadata))
    })?;
    py.check_signals()?;
    Ok((PyBytes::new(py, &output), metadata.to_string()))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(logrow_product_benchmark, module)?)
}
