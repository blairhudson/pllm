//! Installed, bounded local benchmark boundary. No reusable secret handles escape.
use crate::invalid;
use pllm_core::residual_silu_reference::Profile;
use pllm_garble::masked_wire_reference::{self as native, Context, Layout, MaterialEncoding, Step};
use pyo3::{prelude::*, types::PyBytes};
use serde_json::{json, Value};
use std::{sync::OnceLock, time::Instant};
use zeroize::Zeroizing;

mod transport;

fn error(value: native::Error) -> PyErr {
    invalid(format!("masked-wire reference: {value:?}"))
}

fn implementation_digest() -> &'static str {
    static DIGEST: OnceLock<String> = OnceLock::new();
    DIGEST.get_or_init(|| {
        let sources = [
            include_bytes!("masked_wire.rs").as_slice(),
            include_bytes!("masked_wire/transport.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/masked_wire_reference.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/masked_wire_reference/coins.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/masked_wire_reference/seeded_codec.rs")
                .as_slice(),
            include_bytes!("../../pllm-garble/src/masked_wire_reference/transport.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/compact_dpf.rs").as_slice(),
            include_bytes!("../../pllm-core/src/residual_silu_reference.rs").as_slice(),
            include_bytes!("../../../Cargo.lock").as_slice(),
        ];
        let mut bytes = Vec::new();
        for source in sources {
            bytes.extend((source.len() as u64).to_le_bytes());
            bytes.extend(source);
        }
        pllm_types::digest_bytes("pllm.masked_wire_benchmark.implementation.v1", &bytes).to_string()
    })
}

fn parse_layout(layout: &str) -> PyResult<Layout> {
    match layout {
        "unfused" => Ok(Layout::Unfused),
        "fused" => Ok(Layout::Fused),
        _ => Err(invalid(
            "masked-wire layout must be unfused or fused".into(),
        )),
    }
}

fn parse_material(material: &str) -> PyResult<MaterialEncoding> {
    match material {
        "full" => Ok(MaterialEncoding::Full),
        "seeded" => Ok(MaterialEncoding::Seeded),
        _ => Err(invalid(
            "masked-wire material_encoding must be full or seeded".into(),
        )),
    }
}

fn resource_record(lanes: usize, layout: Layout, material: MaterialEncoding) -> PyResult<Value> {
    let resource = native::resources_with_layout(lanes, layout).map_err(error)?;
    let payload =
        native::transport::material_payload_bits(lanes, layout, material).map_err(error)?;
    Ok(json!({
        "lanes": lanes,
        "party_key_bits": resource.party_key_bits,
        "party_key_payload_bytes": resource.party_key_bytes,
        "material_payload_bits_per_party": payload,
        "material_payload_bytes_per_party": payload.map(|bits| bits.div_ceil(8)),
        "material_encoding": if material == MaterialEncoding::Full { "full" } else { "seeded" },
        "peer_frame_bytes": resource.peer_frame_bytes,
        "peer_payload_bits": resource.peer_payload_bits,
        "public_table_bytes_per_party": resource.public_table_bytes,
        "protocol_allocation_estimate_bytes": resource.combined_allocation_estimate_bytes,
        "rounds": layout.rounds(),
        "frame_header_bytes": native::HEADER,
        "profile_digest": pllm_types::Digest::from_sha256(Profile::new().digest()),
        "native_implementation_digest": implementation_digest(),
        "native_version": env!("CARGO_PKG_VERSION"),
    }))
}

#[pyfunction]
#[pyo3(signature = (lanes, layout="unfused", material_encoding="full"))]
fn masked_wire_resources(lanes: usize, layout: &str, material_encoding: &str) -> PyResult<String> {
    Ok(resource_record(
        lanes,
        parse_layout(layout)?,
        parse_material(material_encoding)?,
    )?
    .to_string())
}

#[pyfunction]
#[pyo3(signature = (gate_q24, up_q24, configuration, session, layout="unfused", material_encoding="full"))]
fn masked_wire_benchmark<'py>(
    py: Python<'py>,
    gate_q24: &[u8],
    up_q24: &[u8],
    configuration: &[u8],
    session: &[u8],
    layout: &str,
    material_encoding: &str,
) -> PyResult<(Bound<'py, PyBytes>, String)> {
    // Lengths and resource admission precede decoding, allocation or issuance.
    if gate_q24.is_empty()
        || gate_q24.len() > native::MAX_LANES * 8
        || gate_q24.len() % 8 != 0
        || up_q24.len() != gate_q24.len()
    {
        return Err(invalid(
            "masked-wire inputs require 1..256 matching i64 lanes".into(),
        ));
    }
    let lanes = gate_q24.len() / 8;
    let layout = parse_layout(layout)?;
    let material = parse_material(material_encoding)?;
    let mut metadata = resource_record(lanes, layout, material)?;
    let plan = configuration
        .try_into()
        .map_err(|_| invalid("masked-wire configuration digest requires 32 bytes".into()))?;
    let session = session
        .try_into()
        .map_err(|_| invalid("masked-wire session requires 32 bytes".into()))?;
    let profile = Profile::new();
    let context = Context {
        plan,
        session,
        operation: profile.digest(),
        first_tensor_index: 0,
    };
    let decode = |bytes: &[u8]| {
        Zeroizing::new(
            bytes
                .chunks_exact(8)
                .map(|x| i64::from_le_bytes(x.try_into().expect("checked i64 width")))
                .collect::<Vec<_>>(),
        )
    };
    let gate = decode(gate_q24);
    let up = decode(up_q24);
    let (output, issuance_ns, online_ns, clear_ns, phase_frame_bytes) = py
        .detach(move || {
            let now = Instant::now();
            let expected = Zeroizing::new(
                gate.iter()
                    .zip(up.iter())
                    .map(|(&g, &u)| profile.evaluate(g, u).map_err(|_| native::Error::Domain))
                    .collect::<Result<Vec<_>, _>>()?,
            );
            let clear_ns = now.elapsed().as_nanos() as u64;
            let now = Instant::now();
            let (mut client, [a, b]) =
                native::issue_reference_with_material(context, lanes, layout, material)?;
            // Both encodings pay codec and reconstruction costs before evaluation.
            let left = native::transport::export_party(a)?;
            let right = native::transport::export_party(b)?;
            let expected_bytes = native::transport::material_packet_bytes(lanes, layout, material)?;
            if [left.len(), right.len()] != expected_bytes {
                return Err(native::Error::Context);
            }
            let mut a = native::transport::import_party_with_material(
                &left, context, lanes, layout, 0, material,
            )?;
            let mut b = native::transport::import_party_with_material(
                &right, context, lanes, layout, 1, material,
            )?;
            drop((left, right));
            let issuance_ns = now.elapsed().as_nanos() as u64;
            let now = Instant::now();
            let result = (|| {
                let input = client.encode(&gate, &up)?;
                let mut left = a.start(&input)?;
                let mut right = b.start(&input)?;
                let mut phases = vec![[left.len(), right.len()]];
                for round in 0..layout.rounds() {
                    match (a.advance(&right)?, b.advance(&left)?) {
                        (Step::Send(l), Step::Send(r)) if round + 1 < layout.rounds() => {
                            phases.push([l.len(), r.len()]);
                            left = l;
                            right = r;
                        }
                        (Step::Done(output), Step::Done(_)) if round + 1 == layout.rounds() => {
                            return Ok((Zeroizing::new(client.decode(&output)?), phases));
                        }
                        _ => return Err(native::Error::Context),
                    }
                }
                Err(native::Error::Context)
            })();
            // Any failed step also retires the other local participant's material.
            a.cancel();
            b.cancel();
            let online_ns = now.elapsed().as_nanos() as u64;
            let (output, phases) = result?;
            if *output != *expected
                || phases.len() != layout.rounds()
                || phases.iter().flatten().sum::<usize>()
                    != native::resources_with_layout(lanes, layout)?.peer_frame_bytes
            {
                return Err(native::Error::Context);
            }
            Ok((output, issuance_ns, online_ns, clear_ns, phases))
        })
        .map_err(error)?;
    py.check_signals()?;
    metadata["issuance_ns"] = json!(issuance_ns);
    metadata["online_ns"] = json!(online_ns);
    metadata["clear_ns"] = json!(clear_ns);
    metadata["phase_frame_bytes"] = json!(phase_frame_bytes);
    let bytes = Zeroizing::new(
        output
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    Ok((PyBytes::new(py, &bytes), metadata.to_string()))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    transport::register(module)?;
    module.add_function(wrap_pyfunction!(masked_wire_resources, module)?)?;
    module.add_function(wrap_pyfunction!(masked_wire_benchmark, module)?)
}
