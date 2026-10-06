//! Bounded installed helper benchmark. Every call owns and retires both local parties.
use crate::invalid;
use pllm_garble::shared_rescale_reference::{
    self as native, Backend, Context, Descriptor, Layout, Progress, Rounding,
};
use pyo3::{prelude::*, types::PyBytes};
use serde_json::{json, Value};
use std::{sync::OnceLock, time::Instant};
use zeroize::Zeroizing;

type Settings = (u8, u8, String, String, String);
const MAX_LANES: usize = 256;

fn error(value: native::Error) -> PyErr {
    invalid(format!("shared-rescale reference: {value:?}"))
}

fn descriptor(settings: &Settings) -> PyResult<(Descriptor, Layout)> {
    let rounding = match settings.2.as_str() {
        "floor" => Rounding::Floor,
        "ties_even" => Rounding::TiesEven,
        _ => return Err(invalid("unknown rescale rounding".into())),
    };
    let layout = match settings.3.as_str() {
        "fused" => Layout::Fused,
        "unfused" => Layout::Unfused,
        _ => return Err(invalid("unknown rescale layout".into())),
    };
    let backend = match settings.4.as_str() {
        "prefix_dpf" => Backend::PrefixDpf,
        "compact_dcf" => Backend::CompactDcf,
        "interval_dcf" => Backend::IntervalDcf,
        _ => return Err(invalid("unknown rescale backend".into())),
    };
    Ok((
        Descriptor::new(settings.0, settings.1, rounding)
            .map_err(error)?
            .with_backend(backend),
        layout,
    ))
}

fn implementation_digest() -> &'static str {
    static DIGEST: OnceLock<String> = OnceLock::new();
    DIGEST.get_or_init(|| {
        let sources = [
            include_bytes!("shared_rescale.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/shared_rescale_reference.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/compact_dcf.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/interval_fss.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/point_fss.rs").as_slice(),
            include_bytes!("../../pllm-core/src/codec.rs").as_slice(),
            include_bytes!("../../../Cargo.lock").as_slice(),
        ];
        let mut bytes = Vec::new();
        for source in sources {
            bytes.extend((source.len() as u64).to_le_bytes());
            bytes.extend(source);
        }
        pllm_types::digest_bytes("pllm.shared_rescale_benchmark.implementation.v1", &bytes)
            .to_string()
    })
}

fn resource_record(settings: &Settings, lanes: usize) -> PyResult<Value> {
    if !(1..=MAX_LANES).contains(&lanes) {
        return Err(invalid("rescale benchmark requires 1..256 lanes".into()));
    }
    let (d, layout) = descriptor(settings)?;
    let r = d.resources(layout, lanes).map_err(error)?;
    Ok(json!({
        "lanes": lanes,
        "bits": d.bits(),
        "shift": d.shift(),
        "rounding": settings.2,
        "layout": settings.3,
        "backend": settings.4,
        "party_key_payload_bytes": r.party_key_payload_bytes,
        "protocol_allocation_estimate_bytes": r.total_allocation_estimate_bytes,
        "peer_frame_bytes": r.peer_frame_bytes,
        "peer_payload_bits": 2 * r.rounds * lanes * 32,
        "rounds": r.rounds,
        "comparison_widths": r.comparison_widths,
        "comparison_evaluations_per_lane": r.comparison_evaluations_per_lane,
        "native_implementation_digest": implementation_digest(),
        "native_version": env!("CARGO_PKG_VERSION"),
    }))
}

#[pyfunction]
fn shared_rescale_resources(settings: Settings, lanes: usize) -> PyResult<String> {
    Ok(resource_record(&settings, lanes)?.to_string())
}

#[pyfunction]
fn shared_rescale_benchmark<'py>(
    py: Python<'py>,
    values: &[u8],
    settings: Settings,
    configuration: &[u8],
    session: &[u8],
) -> PyResult<(Bound<'py, PyBytes>, String)> {
    // Preflight bytes, complete material and commitments before input decoding or issuance.
    if values.is_empty() || values.len() > MAX_LANES * 8 || values.len() % 8 != 0 {
        return Err(invalid(
            "rescale benchmark requires 1..256 signed i64 words".into(),
        ));
    }
    let lanes = values.len() / 8;
    let mut metadata = resource_record(&settings, lanes)?;
    let (d, layout) = descriptor(&settings)?;
    let plan = configuration
        .try_into()
        .map_err(|_| invalid("rescale configuration digest requires 32 bytes".into()))?;
    let context = Context {
        plan,
        session: session
            .try_into()
            .map_err(|_| invalid("rescale session requires 32 bytes".into()))?,
        // Exactly one configured operator; native framing also binds its descriptor.
        operation: plan,
        first_tensor_index: 0,
    };
    let inputs = Zeroizing::new(
        values
            .chunks_exact(8)
            .map(|word| i64::from_le_bytes(word.try_into().expect("checked width")))
            .collect::<Vec<_>>(),
    );
    let bound = 1i64 << (d.bits() - 1);
    if inputs.iter().any(|&x| x < -bound || x >= bound) {
        return Err(invalid("rescale input outside signed ring domain".into()));
    }
    let (output, issue_ns, online_ns, clear_ns, phases) = py
        .detach(move || {
            let mask = ((1u64 << d.bits()) - 1) as u32;
            let start = Instant::now();
            let mut expected = Zeroizing::new(Vec::with_capacity(2 * lanes));
            for &x in inputs.iter() {
                let divisor = 1i64 << d.shift();
                let floor = x.div_euclid(divisor);
                let remainder = x.rem_euclid(divisor);
                let up = d.rounding() == Rounding::TiesEven
                    && (2 * remainder > divisor || (2 * remainder == divisor && floor % 2 != 0));
                expected.extend([((floor + i64::from(up)) as u32) & mask, u32::from(x >= 0)]);
            }
            let clear_ns = start.elapsed().as_nanos() as u64;
            let start = Instant::now();
            let [mut a, mut b] = d.issue_reference(layout, context, lanes)?;
            let issue_ns = start.elapsed().as_nanos() as u64;
            let start = Instant::now();
            let result = (|| {
                let resource = d.resources(layout, lanes)?;
                if a.key_payload_bytes() != resource.party_key_payload_bytes
                    || b.key_payload_bytes() != resource.party_key_payload_bytes
                {
                    return Err(native::Error::Context);
                }
                let left = Zeroizing::new(
                    pllm_core::codec::random_residues(1u64 << d.bits(), lanes)
                        .map_err(|_| native::Error::Randomness)?,
                );
                let mut right = Zeroizing::new(Vec::with_capacity(lanes));
                for (&x, &share) in inputs.iter().zip(left.iter()) {
                    right.push((x as u32).wrapping_sub(share) & mask);
                }
                let mut frames = [a.start(&left)?, b.start(&right)?];
                let mut phases = Vec::with_capacity(resource.rounds);
                for round in 0..resource.rounds {
                    phases.push([frames[0].len(), frames[1].len()]);
                    match (a.advance(&frames[1])?, b.advance(&frames[0])?) {
                        (Progress::Open(l), Progress::Open(r)) if round + 1 < resource.rounds => {
                            frames = [l, r];
                        }
                        (Progress::Complete(l), Progress::Complete(r))
                            if round + 1 == resource.rounds =>
                        {
                            if l.len() != lanes || r.len() != lanes {
                                return Err(native::Error::Context);
                            }
                            let mut out = Zeroizing::new(Vec::with_capacity(2 * lanes));
                            for (mut l, mut r) in l.into_iter().zip(r) {
                                out.extend([
                                    l.rescaled.wrapping_add(r.rescaled) & mask,
                                    l.nonnegative.wrapping_add(r.nonnegative) & mask,
                                ]);
                                use zeroize::Zeroize;
                                l.rescaled.zeroize();
                                l.nonnegative.zeroize();
                                r.rescaled.zeroize();
                                r.nonnegative.zeroize();
                            }
                            return Ok((out, phases));
                        }
                        _ => return Err(native::Error::Context),
                    }
                }
                Err(native::Error::Context)
            })();
            a.cancel();
            b.cancel();
            let online_ns = start.elapsed().as_nanos() as u64;
            let (output, phases) = result?;
            if *output != *expected
                || phases.iter().flatten().sum::<usize>()
                    != d.resources(layout, lanes)?.peer_frame_bytes
            {
                return Err(native::Error::Context);
            }
            Ok((output, issue_ns, online_ns, clear_ns, phases))
        })
        .map_err(error)?;
    py.check_signals()?;
    metadata["issuance_ns"] = json!(issue_ns);
    metadata["online_ns"] = json!(online_ns);
    metadata["clear_ns"] = json!(clear_ns);
    metadata["phase_frame_bytes"] = json!(phases);
    let bytes = Zeroizing::new(
        output
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    Ok((PyBytes::new(py, &bytes), metadata.to_string()))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(shared_rescale_resources, module)?)?;
    module.add_function(wrap_pyfunction!(shared_rescale_benchmark, module)?)
}
