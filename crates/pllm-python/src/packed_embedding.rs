//! Bounded packing and one-use conversion; HE orchestration is separate.
use crate::invalid;
use pllm_core::{
    codec,
    packed_embedding::{self as numeric, Geometry, Layout},
};
use pllm_garble::packed_embedding as protected;
use pllm_garble::{shared_circuit, shared_rescale_reference::Context};
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
            include_bytes!("../../pllm-garble/src/shared_circuit.rs").as_slice(),
            include_bytes!("../../pllm-core/src/coefficient_repacking.rs").as_slice(),
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
    module.add_function(wrap_pyfunction!(coefficient_repacking_extract, module)?)?;
    module.add_function(wrap_pyfunction!(coefficient_repacking_embed, module)?)?;
    module.add_class::<PackedEmbeddingParty>()?;
    module.add_function(wrap_pyfunction!(packed_embedding_issue_parties, module)?)?;
    module.add_function(wrap_pyfunction!(coefficient_repacking_divide, module)?)?;
    module.add_function(wrap_pyfunction!(packed_embedding_table, module)?)?;
    module.add_function(wrap_pyfunction!(packed_embedding_mask, module)?)?;
    module.add_function(wrap_pyfunction!(packed_embedding_convert, module)?)?;
    module.add_function(wrap_pyfunction!(packed_embedding_statistics, module)?)?;
    module.add_function(wrap_pyfunction!(packed_embedding_quantize, module)?)
}

#[pyclass]
struct PackedEmbeddingParty {
    inner: Option<shared_circuit::Party>,
}

#[pymethods]
impl PackedEmbeddingParty {
    fn cancel(&mut self) {
        self.inner = None;
    }
    fn start<'py>(
        &mut self,
        py: Python<'py>,
        share: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let mut party = self
            .inner
            .take()
            .ok_or_else(|| invalid("consumed party".into()))?;
        let raw = share
            .cast::<PyBytes>()
            .map_err(|_| invalid("share must be four bytes".into()))?
            .as_bytes();
        let word = u32::from_le_bytes(
            raw.try_into()
                .map_err(|_| invalid("share must be four bytes".into()))?,
        );
        if word >= 8192 {
            return Err(invalid("packed share exceeds ring13".into()));
        }
        let frame = py
            .detach(|| party.start(&[u64::from(word)]))
            .map_err(party_error)?;
        self.inner = Some(party);
        Ok(PyBytes::new(py, &frame))
    }
    fn advance<'py>(
        &mut self,
        py: Python<'py>,
        frame: &Bound<'_, PyAny>,
    ) -> PyResult<(bool, Bound<'py, PyBytes>)> {
        let mut party = self
            .inner
            .take()
            .ok_or_else(|| invalid("consumed party".into()))?;
        let frame = frame
            .cast::<PyBytes>()
            .map_err(|_| invalid("peer frame must be bytes".into()))?
            .as_bytes();
        match py.detach(|| party.advance(frame)).map_err(party_error)? {
            shared_circuit::Progress::Open(frame) => {
                self.inner = Some(party);
                Ok((false, PyBytes::new(py, &frame)))
            }
            shared_circuit::Progress::Complete(values) => {
                let raw = Zeroizing::new(
                    values
                        .iter()
                        .flat_map(|&v| (v as u32).to_le_bytes())
                        .collect::<Vec<_>>(),
                );
                Ok((true, PyBytes::new(py, &raw)))
            }
        }
    }
}
fn party_error(e: impl std::fmt::Debug) -> PyErr {
    invalid(format!("packed party conversion: {e:?}"))
}

#[pyfunction]
fn packed_embedding_issue_parties(
    py: Python<'_>,
    layout: &str,
    plan: &[u8],
    session: &[u8],
    index: u64,
) -> PyResult<(PackedEmbeddingParty, PackedEmbeddingParty, String)> {
    let layout = Layout::from_name(layout).map_err(invalid)?;
    let plan = plan
        .try_into()
        .map_err(|_| invalid("plan must be 32 bytes".into()))?;
    let session = session
        .try_into()
        .map_err(|_| invalid("session must be 32 bytes".into()))?;
    let context = Context {
        plan,
        session,
        operation: plan,
        first_tensor_index: index,
    };
    let ([a, b], r) = py
        .detach(|| protected::issue_party_local(layout, context))
        .map_err(invalid)?;
    if r.allocation_bytes > protected::MAX_ALLOCATION_BYTES {
        return Err(invalid("party conversion allocation exceeds bound".into()));
    }
    let meta = json!({"key_bytes":r.key_bytes,"phase_bytes":r.phase_bytes,"ciphertext_bytes":r.ciphertext_bytes,
        "constant_bytes":r.constant_bytes,"public_metadata_bytes":r.public_metadata_bytes,"allocation_bytes":r.allocation_bytes,
        "native_implementation_digest":digest(),"native_version":env!("CARGO_PKG_VERSION")}).to_string();
    Ok((
        PackedEmbeddingParty { inner: Some(a) },
        PackedEmbeddingParty { inner: Some(b) },
        meta,
    ))
}

#[pyfunction]
fn coefficient_repacking_divide<'py>(
    py: Python<'py>,
    words: &[u8],
    moduli: Vec<u64>,
    divisor: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    if words.is_empty() || words.len() % 8 != 0 || words.len() > 2 * 4096 * 2 * 8 {
        return Err(invalid("invalid RNS word frame".into()));
    }
    let mut values = words
        .chunks_exact(8)
        .map(|v| u64::from_le_bytes(v.try_into().unwrap()))
        .collect::<Vec<_>>();
    py.detach(|| {
        pllm_core::coefficient_repacking::divide_power_two(&mut values, 4096, &moduli, divisor)
    })
    .map_err(invalid)?;
    let raw = values
        .into_iter()
        .flat_map(u64::to_le_bytes)
        .collect::<Vec<_>>();
    Ok(PyBytes::new(py, &raw))
}

fn public_words(words: &[u8], max: usize) -> PyResult<Vec<u64>> {
    if words.is_empty() || words.len() % 8 != 0 || words.len() > max * 8 {
        return Err(invalid("invalid bounded public RNS frame".into()));
    }
    Ok(words
        .chunks_exact(8)
        .map(|v| u64::from_le_bytes(v.try_into().unwrap()))
        .collect())
}

#[pyfunction]
fn coefficient_repacking_extract<'py>(
    py: Python<'py>,
    words: &[u8],
    moduli: Vec<u64>,
    coefficient: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let words = public_words(words, 2 * 4096 * 2)?;
    let result = py
        .detach(|| pllm_core::coefficient_repacking::extract_lwe(&words, &moduli, coefficient))
        .map_err(invalid)?;
    Ok(PyBytes::new(
        py,
        &result
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .collect::<Vec<_>>(),
    ))
}

#[pyfunction]
fn coefficient_repacking_embed<'py>(
    py: Python<'py>,
    words: &[u8],
    moduli: Vec<u64>,
) -> PyResult<Bound<'py, PyBytes>> {
    let words = public_words(words, 4097 * 2)?;
    let result = py
        .detach(|| pllm_core::coefficient_repacking::embed_lwe(&words, &moduli))
        .map_err(invalid)?;
    Ok(PyBytes::new(
        py,
        &result
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .collect::<Vec<_>>(),
    ))
}
