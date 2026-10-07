//! Bounded immutable public profile and serial one-use complete operator.
use crate::invalid;
use pllm_core::compact_product::{Profile, CALIBRATION_BINS, PROFILE};
use pllm_garble::compact_product as native;
use pyo3::{prelude::*, types::PyBytes};
use serde_json::json;
use std::{sync::OnceLock, time::Instant};
use zeroize::Zeroizing;

fn implementation_digest() -> &'static str {
    static DIGEST: OnceLock<String> = OnceLock::new();
    DIGEST.get_or_init(|| {
        let sources = [
            include_bytes!("compact_product.rs").as_slice(),
            include_bytes!("../../pllm-core/src/compact_product.rs").as_slice(),
            include_bytes!("../../pllm-core/src/fixed_point.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/compact_product.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/boolean.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/logrow.rs").as_slice(),
            include_bytes!("../../../Cargo.lock").as_slice(),
        ];
        let mut bytes = Vec::new();
        for source in sources {
            bytes.extend((source.len() as u64).to_le_bytes());
            bytes.extend(source);
        }
        pllm_types::digest_bytes("pllm.compact_product.implementation.v1", &bytes).to_string()
    })
}

fn inputs(gate: &[u8], up: &[u8], maximum: usize) -> PyResult<[Zeroizing<Vec<i32>>; 2]> {
    if gate.is_empty() || gate.len() > maximum * 4 || gate.len() % 4 != 0 || gate.len() != up.len()
    {
        return Err(invalid(
            "Compact requires matching bounded signed-i32 frames".into(),
        ));
    }
    let arrays = [gate, up].map(|raw| {
        Zeroizing::new(
            raw.chunks_exact(4)
                .map(|b| i32::from_le_bytes(b.try_into().expect("validated width")))
                .collect::<Vec<_>>(),
        )
    });
    for (&g, &u) in arrays[0].iter().zip(arrays[1].iter()) {
        Profile::validate_inputs(g, u).map_err(invalid)?;
    }
    Ok(arrays)
}

#[pyclass(frozen, module = "pllm._native")]
pub struct CompactProductReference {
    profile: Profile,
}

#[pymethods]
impl CompactProductReference {
    #[new]
    fn new(
        py: Python<'_>,
        public_calibration_counts: &[u8],
        max_pieces: u8,
        degree: u8,
    ) -> PyResult<Self> {
        if public_calibration_counts.len() != CALIBRATION_BINS * 4 {
            return Err(invalid(
                "Compact calibration requires 1,024 little-endian u32 counts".into(),
            ));
        }
        let counts = public_calibration_counts
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().expect("validated width")))
            .collect::<Vec<_>>();
        let profile = py
            .detach(|| Profile::fit(&counts, max_pieces, degree))
            .map_err(invalid)?;
        Ok(Self { profile })
    }

    fn metadata(&self) -> String {
        json!({"profile": PROFILE, "degree": self.profile.degree(),
            "profile_digest": pllm_types::Digest::from_sha256(self.profile.digest()),
            "pieces": self.profile.pieces().iter().map(|p| json!({"lower": p.lower,
                "upper": p.upper, "coefficients": p.coefficients})).collect::<Vec<_>>()})
        .to_string()
    }

    fn clear<'py>(
        &self,
        py: Python<'py>,
        gate_q18: &[u8],
        up_q18: &[u8],
    ) -> PyResult<Bound<'py, PyBytes>> {
        let arrays = inputs(gate_q18, up_q18, 1 << 20)?;
        let output = py.detach(|| -> PyResult<_> {
            let mut result = Zeroizing::new(Vec::with_capacity(gate_q18.len()));
            for (&g, &u) in arrays[0].iter().zip(arrays[1].iter()) {
                result.extend(self.profile.evaluate(g, u).map_err(invalid)?.to_le_bytes());
            }
            Ok(result)
        })?;
        py.check_signals()?;
        Ok(PyBytes::new(py, &output))
    }

    fn benchmark<'py>(
        &self,
        py: Python<'py>,
        gate_q18: &[u8],
        up_q18: &[u8],
    ) -> PyResult<(Bound<'py, PyBytes>, String)> {
        let arrays = inputs(gate_q18, up_q18, 256)?;
        let (output, metadata) = py.detach(|| -> PyResult<_> {
            let start = Instant::now();
            let expected = Zeroizing::new(arrays[0].iter().zip(arrays[1].iter())
                .map(|(&g, &u)| self.profile.evaluate(g, u).map_err(invalid)).collect::<PyResult<Vec<_>>>()?);
            let clear_ns = start.elapsed().as_nanos() as u64;
            let mut output = Zeroizing::new(Vec::with_capacity(gate_q18.len()));
            let mut reference = None;
            let mut issuance_ns = 0;
            let mut online_ns = 0;
            for ((&g, &u), &want) in arrays[0].iter().zip(arrays[1].iter()).zip(expected.iter()) {
                let start = Instant::now();
                let (client, program) = native::prepare_reference(&self.profile).map_err(invalid)?;
                issuance_ns += start.elapsed().as_nanos() as u64;
                let resource = program.resources().map_err(invalid)?;
                if reference.is_some_and(|r| r != resource) {
                    return Err(invalid("Compact topology depends on private input".into()));
                }
                reference = Some(resource);
                let start = Instant::now();
                let (input, decoder) = client.encode(g, u).map_err(invalid)?;
                let actual = decoder.decode(program.evaluate(input).map_err(invalid)?).map_err(invalid)?;
                online_ns += start.elapsed().as_nanos() as u64;
                if want != actual { return Err(invalid("Compact protected integer parity failed".into())); }
                output.extend(actual.to_le_bytes());
            }
            let r = reference.expect("validated nonzero lanes");
            let lanes = arrays[0].len();
            let parts = [
                ("client_encodings", lanes * r.client_encoding_bytes, 0),
                ("input_rescale", 0, lanes * r.arithmetic_bytes[0]),
                ("private_piece_selection", 0, lanes * r.arithmetic_bytes[1]),
                ("polynomial_rounding", 0, lanes * r.arithmetic_bytes[2]),
                ("tails_product_output_rounding", 0, lanes * r.arithmetic_bytes[3]),
                ("constant_labels", 0, lanes * r.evaluator_constant_bytes)];
            let metadata = json!({"lanes": lanes,
                "profile_digest": pllm_types::Digest::from_sha256(self.profile.digest()),
                "party_key_payload_bytes": [lanes * r.client_encoding_bytes,
                    lanes * (r.evaluator_half_gate_bytes + r.evaluator_constant_bytes)],
                "key_payload_bytes_by_operation": parts,
                "evaluator_ciphertext_bytes": lanes * r.evaluator_half_gate_bytes,
                "evaluator_constant_label_bytes": lanes * r.evaluator_constant_bytes,
                "public_metadata_bytes_estimate": lanes * r.public_metadata_bytes_estimate,
                "client_entry_exit_payload_estimate_bytes": lanes * (r.input_output_label_bytes + 64),
                "protocol_allocation_estimate_bytes": native::MAX_ALLOCATION_BYTES,
                "and_gates_per_lane": r.and_gates, "wires_per_lane": r.wires,
                "native_implementation_digest": implementation_digest(), "native_version": env!("CARGO_PKG_VERSION"),
                "issuance_ns": issuance_ns, "online_ns": online_ns, "clear_ns": clear_ns});
            Ok((output, metadata))
        })?;
        py.check_signals()?;
        Ok((PyBytes::new(py, &output), metadata.to_string()))
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<CompactProductReference>()
}
