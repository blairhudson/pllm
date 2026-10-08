//! Measurement-only shared-record selection; no live owner crosses PyO3.
use crate::invalid;
use pllm_garble::shared_rescale_reference::Context;
use pllm_garble::shared_selection::{self as native, Plan, Selection};
use pyo3::{prelude::*, types::PyBytes};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::OnceLock;
use zeroize::Zeroizing;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Specification {
    rows: usize,
    words: usize,
    #[serde(default = "serial_batch")]
    batch_size: usize,
    #[serde(default)]
    bounded_metadata: bool,
    selection: Method,
}

fn serial_batch() -> usize {
    1
}

#[derive(Deserialize)]
#[serde(tag = "method", rename_all = "snake_case", deny_unknown_fields)]
enum Method {
    TopK { keep: usize, recent: usize },
    Threshold { threshold: u32, reduction: u32 },
}

fn parse(raw: &str) -> PyResult<Plan> {
    if raw.len() > 4096 {
        return Err(invalid("selection specification exceeds 4096 bytes".into()));
    }
    let p: Specification = serde_json::from_str(raw)
        .map_err(|_| invalid("invalid shared-selection specification".into()))?;
    Ok(Plan {
        rows: p.rows,
        words: p.words,
        batch_size: p.batch_size,
        bounded_metadata: p.bounded_metadata,
        selection: match p.selection {
            Method::TopK { keep, recent } => Selection::TopK { keep, recent },
            Method::Threshold {
                threshold,
                reduction,
            } => Selection::Threshold {
                threshold,
                reduction,
            },
        },
    })
}

fn failure(e: pllm_garble::shared_rescale_reference::Error) -> PyErr {
    invalid(format!("shared-selection reference: {e:?}"))
}

fn implementation_digest() -> &'static str {
    static DIGEST: OnceLock<String> = OnceLock::new();
    DIGEST.get_or_init(|| {
        let mut bytes = Vec::new();
        for source in [
            include_bytes!("shared_selection.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/shared_selection.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/shared_selection/batch.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/shared_circuit.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/shared_arithmetic_reference.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/boolean.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/token_pruning.rs").as_slice(),
            include_bytes!("../../pllm-core/src/token_pruning.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/retained_selection.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/retained_selection/cost.rs").as_slice(),
            include_bytes!("../../../Cargo.lock").as_slice(),
        ] {
            bytes.extend((source.len() as u64).to_le_bytes());
            bytes.extend(source);
        }
        pllm_types::digest_bytes("pllm.shared_selection.implementation.v1", &bytes).to_string()
    })
}

fn resources(r: &native::Resources) -> Value {
    json!({
        "key_bytes": r.key_bytes, "frame_bytes": r.frame_bytes, "rounds": r.rounds,
        "framing_bytes": r.framing_bytes,
        "predicates": r.predicates, "products": r.products,
        "ciphertext_bytes": r.ciphertext_bytes, "constant_bytes": r.constant_bytes,
        "public_metadata_bytes_estimate": r.public_metadata_bytes,
        "phases": r.phases,
        "key_parts": [["garbled-decisions", r.key_bytes[0] - 12 * r.products as u64, r.key_bytes[1] - 12 * r.products as u64],
                      ["record-movement", 12 * r.products as u64, 12 * r.products as u64]],
    })
}

#[pyfunction]
fn shared_selection_projection(specification: &str, retained: usize) -> PyResult<String> {
    use pllm_garble::retained_selection::{project, Layout, Policy};
    let plan = parse(specification)?;
    let r = plan.project(retained).map_err(failure)?;
    let allocation = plan.admit();
    let legacy = match plan.selection {
        Selection::TopK { keep, recent } if plan.words >= 3 && plan.words % 2 == 1 => {
            let state_bytes = (plan.rows + 2) * plan.words * 4;
            let controls = [Layout::FullRecords, Layout::CompactIndices].map(|layout| {
                let p = project(plan.rows, (plan.words - 1) / 2, Policy { keep, recent }, state_bytes, layout).map_err(invalid)?;
                Ok::<_, PyErr>(json!({
                    "layout": if layout == Layout::FullRecords { "full_records" } else { "compact_indices" },
                    "admitted": p.admitted(), "ciphertext_bytes": 32 * p.and_gates,
                    "constant_bytes": p.constant_label_bytes,
                    "client_encoding_bytes": 32 * (p.estimate.input_bits + p.estimate.output_bits),
                    "input_output_label_bytes": 16 * (p.estimate.input_bits + p.estimate.output_bits),
                    "public_metadata_bytes_estimate": p.public_metadata_bytes_estimate,
                    "allocation_estimate_bytes": p.estimate.combined_allocation_bound_bytes,
                }))
            });
            Some(controls.into_iter().collect::<PyResult<Vec<_>>>()?)
        }
        _ => None,
    };
    Ok(json!({"resources": resources(&r), "admitted": allocation.is_ok(),
        "allocation_estimate_bytes": allocation.ok(), "material_issued": false,
        "legacy_complete_kv_layouts": legacy, "native_implementation_digest": implementation_digest()}).to_string())
}

#[pyfunction]
fn shared_selection_benchmark<'py>(
    py: Python<'py>,
    specification: &str,
    scores: &[u8],
    words: &[u8],
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, String)> {
    let plan = parse(specification)?;
    plan.admit().map_err(failure)?;
    if scores.len() != 4 * plan.rows || words.len() != 4 * plan.rows * plan.words {
        return Err(invalid(
            "shared-selection byte inputs differ from their declared shapes".into(),
        ));
    }
    let unpack = |raw: &[u8]| {
        Zeroizing::new(
            raw.chunks_exact(4)
                .map(|v| u32::from_le_bytes(v.try_into().expect("word")))
                .collect::<Vec<_>>(),
        )
    };
    let scores = unpack(scores);
    let words = unpack(words);
    let (output, degrees, metadata) = py.detach(|| -> PyResult<_> {
        let context = Context { plan: [0;32], session: [0;32], operation: [0;32], first_tensor_index: 0 };
        let m = native::run_reference(plan, &scores, &words, context).map_err(failure)?;
        let metadata = json!({
            "rows": plan.rows, "words": plan.words, "retained": m.retained,
            "bounded_metadata": plan.bounded_metadata,
            "degree_disclosed": m.high_degree.is_some(), "resources": resources(&m.resources),
            "allocation_estimate_bytes": m.allocation_bytes, "sharing_seconds": m.sharing_seconds,
            "issuance_seconds": m.issuance_seconds, "online_seconds": m.online_seconds,
            "native_implementation_digest": implementation_digest(), "native_version": env!("CARGO_PKG_VERSION"),
        });
        Ok((Zeroizing::new(m.output.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()),
            Zeroizing::new(m.high_degree.unwrap_or_default().into_iter().map(u8::from).collect::<Vec<_>>()), metadata))
    })?;
    py.check_signals()?;
    Ok((
        PyBytes::new(py, &output),
        PyBytes::new(py, &degrees),
        metadata.to_string(),
    ))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(shared_selection_projection, module)?)?;
    module.add_function(wrap_pyfunction!(shared_selection_benchmark, module)?)
}
