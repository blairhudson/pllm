//! Installed bounded Haar gated-block benchmarks; no live handles escape.
use crate::invalid;
use pllm_core::wavelet_gated_reference::Profile;
use pllm_garble::{
    shared_rescale_reference::{Context, Error},
    wavelet_gated_reference as native,
};
use pyo3::{prelude::*, types::PyBytes};
use serde_json::{json, Value};
use std::{sync::OnceLock, time::Instant};
use zeroize::Zeroizing;

fn error(error: impl std::fmt::Debug) -> PyErr {
    invalid(format!("wavelet gated reference: {error:?}"))
}

fn implementation_digest() -> &'static str {
    static DIGEST: OnceLock<String> = OnceLock::new();
    DIGEST.get_or_init(|| {
        let sources = [
            include_bytes!("wavelet_gated.rs").as_slice(),
            include_bytes!("../../pllm-core/src/wavelet_gated_reference.rs").as_slice(),
            include_bytes!("../../pllm-core/src/fixed_point.rs").as_slice(),
            include_bytes!("../../pllm-core/src/codec.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/wavelet_gated_reference.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/shared_lut_reference.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/shared_arithmetic_reference.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/shared_rescale_reference.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/compact_dcf.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/interval_fss.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/point_fss.rs").as_slice(),
            include_bytes!("../../../Cargo.lock").as_slice(),
        ];
        let mut bytes = Vec::new();
        for source in sources {
            bytes.extend((source.len() as u64).to_le_bytes());
            bytes.extend(source);
        }
        pllm_types::digest_bytes("pllm.wavelet_gated_benchmark.implementation.v1", &bytes)
            .to_string()
    })
}

fn resource_record(profile: &Profile, lanes: usize) -> PyResult<Value> {
    let r = native::resources(profile.levels(), lanes).map_err(error)?;
    Ok(json!({ "lanes": lanes, "levels": profile.levels(),
        "profile_digest": pllm_types::Digest::from_sha256(profile.digest()),
        "party_key_payload_bytes": r.party_key_payload_bytes,
        "party_key_bytes_by_operation": r.party_key_bytes_by_operation,
        "protocol_allocation_estimate_bytes": r.allocation_estimate_bytes,
        "phase_frame_bytes_per_party": r.phase_frame_bytes_per_party,
        "peer_frame_bytes": 2 * r.phase_frame_bytes_per_party.iter().sum::<usize>(),
        "peer_payload_bits": 2 * lanes * (64 + 16 + 64 + 32),
        "rounds": native::ROUNDS, "public_table_bytes_per_party": r.public_table_bytes_per_party,
        // Public context/nonce/party identities for three rescalings, one
        // multiplication, every lane's LUT and the enclosing block. No codec.
        "public_metadata_bytes_estimate": 2 * (5 * native::HEADER + lanes * 65),
        "native_implementation_digest": implementation_digest(),
        "native_version": env!("CARGO_PKG_VERSION") }))
}

#[pyfunction]
fn wavelet_gated_resources(py: Python<'_>, levels: u8, lanes: usize) -> PyResult<String> {
    native::resources(levels, lanes).map_err(error)?;
    py.detach(|| {
        resource_record(&Profile::new(levels).map_err(error)?, lanes).map(|v| v.to_string())
    })
}

fn inputs(gate: &[u8], up: &[u8], maximum: usize) -> PyResult<[Zeroizing<Vec<i32>>; 2]> {
    if gate.is_empty() || gate.len() != up.len() || gate.len() % 4 != 0 || gate.len() > 4 * maximum
    {
        return Err(invalid(
            "wavelet gated inputs require matching bounded i32 frames".into(),
        ));
    }
    let arrays = [gate, up].map(|raw| {
        Zeroizing::new(
            raw.chunks_exact(4)
                .map(|b| i32::from_le_bytes(b.try_into().expect("validated word size")))
                .collect::<Vec<_>>(),
        )
    });
    for (&gate, &up) in arrays[0].iter().zip(arrays[1].iter()) {
        Profile::validate_inputs(gate, up).map_err(error)?;
    }
    Ok(arrays)
}

#[pyfunction]
fn _wavelet_gated_clear<'py>(
    py: Python<'py>,
    levels: u8,
    gate_q18: &[u8],
    up_q18: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    let arrays = inputs(gate_q18, up_q18, 1 << 20)?;
    let bytes = py.detach(|| -> PyResult<_> {
        let profile = Profile::new(levels).map_err(error)?;
        let mut bytes = Zeroizing::new(Vec::with_capacity(gate_q18.len()));
        for (&gate, &up) in arrays[0].iter().zip(arrays[1].iter()) {
            bytes.extend(profile.evaluate(gate, up).map_err(error)?.to_le_bytes());
        }
        Ok(bytes)
    })?;
    py.check_signals()?;
    Ok(PyBytes::new(py, &bytes))
}

#[pyfunction]
fn wavelet_gated_benchmark<'py>(
    py: Python<'py>,
    levels: u8,
    gate_q18: &[u8],
    up_q18: &[u8],
    configuration: &[u8],
    session: &[u8],
) -> PyResult<(Bound<'py, PyBytes>, String)> {
    let arrays = inputs(gate_q18, up_q18, native::MAX_LANES)?;
    let lanes = arrays[0].len();
    native::resources(levels, lanes).map_err(error)?;
    let plan = configuration
        .try_into()
        .map_err(|_| invalid("configuration digest requires 32 bytes".into()))?;
    let context = Context {
        plan,
        operation: plan,
        first_tensor_index: 0,
        session: session
            .try_into()
            .map_err(|_| invalid("session requires 32 bytes".into()))?,
    };
    let (output, metadata) = py.detach(|| -> PyResult<_> {
        let start = Instant::now();
        let profile = Profile::new(levels).map_err(error)?;
        let setup_ns = start.elapsed().as_nanos() as u64;
        let mut metadata = resource_record(&profile, lanes)?;
        let start = Instant::now();
        let expected = Zeroizing::new(
            arrays[0]
                .iter()
                .zip(arrays[1].iter())
                .map(|(&g, &u)| profile.evaluate(g, u).map_err(error))
                .collect::<PyResult<Vec<_>>>()?,
        );
        let clear_ns = start.elapsed().as_nanos() as u64;
        let start = Instant::now();
        let [mut a, mut b] = native::issue_reference(&profile, context, lanes).map_err(error)?;
        let issuance_ns = start.elapsed().as_nanos() as u64;
        let start = Instant::now();
        let result = (|| -> Result<_, Error> {
            let resource = native::resources(levels, lanes)?;
            if a.key_payload_bytes() != resource.party_key_payload_bytes
                || b.key_payload_bytes() != resource.party_key_payload_bytes
            {
                return Err(Error::Context);
            }
            let shares = Zeroizing::new(
                pllm_core::codec::random_residues(1 << 32, 2 * lanes)
                    .map_err(|_| Error::Randomness)?,
            );
            let bg = Zeroizing::new(
                arrays[0]
                    .iter()
                    .zip(&shares[..lanes])
                    .map(|(&x, &s)| (x as u32).wrapping_sub(s))
                    .collect::<Vec<_>>(),
            );
            let bu = Zeroizing::new(
                arrays[1]
                    .iter()
                    .zip(&shares[lanes..])
                    .map(|(&x, &s)| (x as u32).wrapping_sub(s))
                    .collect::<Vec<_>>(),
            );
            let mut frames = [
                a.start(&shares[..lanes], &shares[lanes..])?,
                b.start(&bg, &bu)?,
            ];
            let mut phases = Vec::new();
            for round in 0..native::ROUNDS {
                phases.push([frames[0].len(), frames[1].len()]);
                if phases[round] != [resource.phase_frame_bytes_per_party[round]; 2] {
                    return Err(Error::Context);
                }
                match (a.advance(&frames[1])?, b.advance(&frames[0])?) {
                    (native::Progress::Open(af), native::Progress::Open(bf))
                        if round + 1 < native::ROUNDS =>
                    {
                        frames = [af, bf];
                    }
                    (native::Progress::Complete(aa), native::Progress::Complete(bb))
                        if round + 1 == native::ROUNDS =>
                    {
                        let output = Zeroizing::new(
                            aa.iter()
                                .zip(bb.iter())
                                .map(|(&a, &b)| a.wrapping_add(b) as i32)
                                .collect::<Vec<_>>(),
                        );
                        if *output != *expected {
                            return Err(Error::Context);
                        }
                        return Ok((output, phases));
                    }
                    _ => return Err(Error::Context),
                }
            }
            Err(Error::Context)
        })();
        a.cancel();
        b.cancel();
        let online_ns = start.elapsed().as_nanos() as u64;
        let (output, phases) = result.map_err(error)?;
        metadata["public_setup_ns"] = json!(setup_ns);
        metadata["issuance_ns"] = json!(issuance_ns);
        metadata["online_ns"] = json!(online_ns);
        metadata["clear_ns"] = json!(clear_ns);
        metadata["phase_frame_bytes"] = json!(phases);
        Ok((output, metadata))
    })?;
    py.check_signals()?;
    let bytes = Zeroizing::new(
        output
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    Ok((PyBytes::new(py, &bytes), metadata.to_string()))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(wavelet_gated_resources, module)?)?;
    module.add_function(wrap_pyfunction!(wavelet_gated_benchmark, module)?)?;
    module.add_function(wrap_pyfunction!(_wavelet_gated_clear, module)?)
}
