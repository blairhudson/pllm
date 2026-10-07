//! Bounded packing and one-use conversion; HE orchestration is separate.
use crate::invalid;
use pllm_core::{
    codec,
    packed_embedding::{self as numeric, Geometry, Layout},
};
use pllm_garble::packed_embedding as protected;
use pyo3::{prelude::*, types::PyBytes};
use serde_json::json;
use std::{fmt::Write, sync::OnceLock, time::Instant};
use zeroize::Zeroizing;

fn digest() -> &'static str {
    static DIGEST: OnceLock<String> = OnceLock::new();
    DIGEST.get_or_init(|| {
        let mut bytes = Vec::new();
        for source in [
            include_bytes!("packed_embedding.rs").as_slice(),
            include_bytes!("../../pllm-core/src/packed_embedding.rs").as_slice(),
            include_bytes!("../../pllm-core/src/codec.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/packed_embedding.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/boolean.rs").as_slice(),
            include_bytes!("../../../Cargo.lock").as_slice(),
        ] {
            bytes.extend((source.len() as u64).to_le_bytes());
            bytes.extend(source);
        }
        pllm_types::digest_bytes("pllm.packed_embedding.implementation.v1", &bytes).to_string()
    })
}

fn polynomial(coefficients: &[u16]) -> String {
    let mut text = String::new();
    for (index, &value) in coefficients.iter().enumerate().rev() {
        if value == 0 {
            continue;
        }
        if !text.is_empty() {
            text.push_str(" + ");
        }
        if index == 0 {
            write!(&mut text, "{value:X}").expect("String write");
        } else {
            write!(&mut text, "{value:X}x^{index}").expect("String write");
        }
    }
    if text.is_empty() {
        text.push('0');
    }
    text
}

#[pyfunction]
fn packed_embedding_table(
    py: Python<'_>,
    rows: usize,
    columns: usize,
    layout: &str,
    codes: &[u8],
) -> PyResult<(Vec<String>, String)> {
    let layout = Layout::from_name(layout).map_err(invalid)?;
    let geometry = Geometry::new(rows, columns, layout).map_err(invalid)?;
    if codes.len() != rows * columns {
        return Err(invalid("embedding table byte shape differs".into()));
    }
    let codes = Zeroizing::new(codes.iter().map(|&c| c as i8).collect::<Vec<_>>());
    let result = py
        .detach(|| -> Result<_, String> {
            let coefficients = numeric::pack_table(&codes, geometry, layout)?;
            let positions = (0..geometry.packed_columns())
                .map(|i| geometry.output_coefficient(i))
                .collect::<Result<Vec<_>, _>>()?;
            Ok((coefficients.iter().map(|p| polynomial(p)).collect(), json!({
            "rows": rows, "columns": columns, "padded_rows": geometry.padded_rows(),
            "polynomials": geometry.polynomials(), "positions": positions,
            "coefficient_bytes": coefficients.len() * numeric::POLY_DEGREE * 2,
            "native_implementation_digest": digest(), "native_version": env!("CARGO_PKG_VERSION"),
        }).to_string()))
        })
        .map_err(invalid)?;
    py.check_signals()?;
    Ok(result)
}

#[pyfunction]
fn packed_embedding_mask(py: Python<'_>) -> PyResult<(String, Bound<'_, PyBytes>)> {
    let (text, bytes) = py
        .detach(|| -> Result<_, String> {
            let mask = Zeroizing::new(codec::random_residues(
                u64::from(numeric::MODULUS),
                numeric::POLY_DEGREE,
            )?);
            let coefficients = Zeroizing::new(mask.iter().map(|&v| v as u16).collect::<Vec<_>>());
            Ok((
                Zeroizing::new(polynomial(&coefficients)),
                Zeroizing::new(
                    coefficients
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                ),
            ))
        })
        .map_err(invalid)?;
    py.check_signals()?;
    Ok((text.to_string(), PyBytes::new(py, &bytes)))
}

#[pyfunction]
fn packed_embedding_convert<'py>(
    py: Python<'py>,
    layout: &str,
    client_share: u16,
    provider_share: u16,
) -> PyResult<(Bound<'py, PyBytes>, String)> {
    let layout = Layout::from_name(layout).map_err(invalid)?;
    if client_share >= numeric::MODULUS || provider_share >= numeric::MODULUS {
        return Err(invalid("packed input shares exceed thirteen bits".into()));
    }
    let (output, metadata) = py
        .detach(|| -> Result<_, String> {
            // Sample independent output masks in Rust. They are retained by the
            // simulated provider, never by the output-share decoder.
            let masks = Zeroizing::new(codec::random_residues(1_u64 << 32, layout.widths().len())?);
            let start = Instant::now();
            let (issuer, program) = protected::prepare_reference(layout)?;
            let resources = program.resources()?;
            let (inputs, decoder) = issuer.encode(client_share, provider_share, &masks)?;
            let issuance = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let client_outputs = Zeroizing::new(decoder.decode(program.evaluate(inputs)?)?);
            let online = start.elapsed().as_secs_f64();
            // Trusted benchmark boundary combines output shares only for the
            // independent Python oracle. No reusable protocol handle escapes.
            let output = Zeroizing::new(
                client_outputs
                    .iter()
                    .zip(masks.iter())
                    .flat_map(|(&a, &b)| a.wrapping_add(b).to_le_bytes())
                    .collect::<Vec<_>>(),
            );
            Ok((
                output,
                json!({
                    "issuer_encoding_bytes": resources.issuer_encoding_bytes,
                    "evaluator_material_bytes": resources.evaluator_material_bytes,
                    "boundary_label_bytes": resources.boundary_label_bytes,
                    "public_metadata_bytes_estimate": resources.public_metadata_bytes_estimate,
                    "and_gates": resources.and_gates, "wires": resources.wires,
                    "protocol_allocation_estimate_bytes": protected::MAX_ALLOCATION_BYTES,
                    "issuance_seconds": issuance, "online_seconds": online,
                })
                .to_string(),
            ))
        })
        .map_err(invalid)?;
    py.check_signals()?;
    Ok((PyBytes::new(py, &output), metadata))
}

fn floats(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if bytes.is_empty()
        || bytes.len() > 4 * numeric::MAX_TABLE_ELEMENTS
        || bytes.len().checked_rem(4) != Some(0)
    {
        return Err("embedding float chunk exceeds the bounded representation".into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().expect("four bytes")))
        .collect())
}

#[pyfunction]
fn packed_embedding_statistics(py: Python<'_>, columns: usize, weights: &[u8]) -> PyResult<String> {
    let output = py
        .detach(|| -> Result<_, String> {
            let (maxima, sums) = numeric::channel_statistics(&floats(weights)?, columns)?;
            Ok(json!({"maxima": maxima, "sums": sums}).to_string())
        })
        .map_err(invalid)?;
    py.check_signals()?;
    Ok(output)
}

#[pyfunction]
fn packed_embedding_quantize<'py>(
    py: Python<'py>,
    weights: &[u8],
    widths: &[u8],
    scales: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    if widths.len() > 8192 || scales.len() != 4 * widths.len() {
        return Err(invalid("embedding quantizer channel bounds differ".into()));
    }
    let output = py
        .detach(|| -> Result<_, String> {
            Ok(
                numeric::quantize_channels(&floats(weights)?, widths, &floats(scales)?)?
                    .into_iter()
                    .map(|v| v as u8)
                    .collect::<Vec<_>>(),
            )
        })
        .map_err(invalid)?;
    py.check_signals()?;
    Ok(PyBytes::new(py, &output))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(packed_embedding_table, module)?)?;
    module.add_function(wrap_pyfunction!(packed_embedding_mask, module)?)?;
    module.add_function(wrap_pyfunction!(packed_embedding_convert, module)?)?;
    module.add_function(wrap_pyfunction!(packed_embedding_statistics, module)?)?;
    module.add_function(wrap_pyfunction!(packed_embedding_quantize, module)?)
}
