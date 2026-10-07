//! Bounded installed operator measurements over existing one-use references.
//! Lanes execute serially, retiring all secret material before the next lane.
use crate::invalid;
use pllm_core::fit_compact_silu_q7;
use pllm_garble::{compact_polynomial, logrow, shared_lut_reference::Table};
use pyo3::{prelude::*, types::PyBytes};
use serde_json::json;
use std::{sync::OnceLock, time::Instant};
use zeroize::Zeroizing;

fn implementation_digest() -> &'static str {
    static DIGEST: OnceLock<String> = OnceLock::new();
    DIGEST.get_or_init(|| {
        let sources = [
            include_bytes!("activation_benchmarks.rs").as_slice(),
            include_bytes!("../../pllm-core/src/compact.rs").as_slice(),
            include_bytes!("../../pllm-core/src/fixed_point.rs").as_slice(),
            include_bytes!("../../pllm-core/src/codec.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/compact_polynomial.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/compact_coordinate.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/boolean.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/logrow.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/shared_lut_reference.rs").as_slice(),
            include_bytes!("../../../Cargo.lock").as_slice(),
        ];
        let mut bytes = Vec::new();
        for source in sources {
            bytes.extend((source.len() as u64).to_le_bytes());
            bytes.extend(source);
        }
        pllm_types::digest_bytes("pllm.activation_benchmark.implementation.v1", &bytes).to_string()
    })
}

fn reference_error(error: impl std::fmt::Debug) -> PyErr {
    invalid(format!("activation operator reference: {error:?}"))
}

#[pyfunction]
fn activation_benchmark<'py>(
    py: Python<'py>,
    method: &str,
    max_pieces: u8,
    public_counts: &[u8],
    levels: u8,
    values_q7: &[u8],
) -> PyResult<(Bound<'py, PyBytes>, String)> {
    // Validate the entire call before fitting, allocating protocol material or
    // sampling secrets. The sequential schedule never retains a lane batch.
    if !matches!(method, "compact" | "logrow" | "curl")
        || !(1..=8).contains(&max_pieces)
        || public_counts.len() != 257 * 4
        || levels > 8
        || (method != "curl" && levels != 0)
        || values_q7.is_empty()
        || values_q7.len() > 256 * 2
        || values_q7.len() % 2 != 0
    {
        return Err(invalid(
            "activation benchmark requires bounded Q7 inputs and public calibration".into(),
        ));
    }
    let counts = public_counts
        .chunks_exact(4)
        .map(|x| u32::from_le_bytes(x.try_into().expect("checked u32 width")))
        .collect::<Vec<_>>();
    if counts.iter().any(|&value| value > 1_000_000)
        || counts.iter().map(|&x| u64::from(x)).sum::<u64>() > 100_000_000
    {
        return Err(invalid(
            "activation public calibration exceeds its bounds".into(),
        ));
    }
    let values = Zeroizing::new(
        values_q7
            .chunks_exact(2)
            .map(|x| i16::from_le_bytes(x.try_into().expect("checked i16 width")))
            .collect::<Vec<_>>(),
    );
    if values.iter().any(|&x| !(-128..=128).contains(&x)) {
        return Err(invalid("activation Q7 input is outside [-128, 128]".into()));
    }
    let lanes = values.len();
    let (output, metadata) = py.detach(|| -> PyResult<_> {
        let start = Instant::now();
        let profile = fit_compact_silu_q7(&counts, max_pieces).map_err(reference_error)?;
        let original = (0..512).map(|i| profile.evaluate(i.min(256) - 128)
            .map(i64::from).map_err(reference_error)).collect::<PyResult<Vec<_>>>()?;
        // At most 256 signed-i16 values are summed. Both that integer sum and
        // its division by a power of two are exactly representable in f64.
        let coefficients = original.chunks_exact(1 << levels)
            .map(|group| (group.iter().sum::<i64>() as f64 / group.len() as f64)
                .round_ties_even() as i32 as u32)
            .collect::<Vec<_>>();
        let table = Table::new(&coefficients).map_err(reference_error)?;
        let setup_ns = start.elapsed().as_nanos() as u64;
        let start = Instant::now();
        let expected = Zeroizing::new(values.iter().map(|&value| {
            coefficients[((value + 128) as usize) >> levels] as i32 as i16
        }).collect::<Vec<_>>());
        let clear_ns = start.elapsed().as_nanos() as u64;
        let mut output = Zeroizing::new(Vec::with_capacity(lanes));
        let mut keys = [0_u64; 2];
        let mut phase = [0_usize; 2];
        let mut public_metadata = 0_u64;
        let mut entry_exit = 0_u64;
        let mut ciphertext = 0_u64;
        let mut constant_labels = 0_u64;
        let mut issuance_ns = 0_u64;
        let mut online_ns = 0_u64;
        for &value in values.iter() {
            let now = Instant::now();
            let actual = match method {
                "compact" => {
                    let (client, program) = compact_polynomial::prepare_compact_q7_polynomial(&profile)
                        .map_err(reference_error)?;
                    issuance_ns += now.elapsed().as_nanos() as u64;
                    let body = program.evaluator_ciphertext_bytes().map_err(reference_error)?;
                    let constants = program.constant_label_bytes() as u64;
                    keys[0] += program.client_encoding_bytes() as u64;
                    keys[1] += body + constants;
                    ciphertext += body;
                    constant_labels += constants;
                    public_metadata += program.public_metadata_bytes_estimate() as u64;
                    entry_exit += program.input_output_label_bytes() as u64 + 64;
                    let now = Instant::now();
                    let (inputs, decoder) = client.encode(value).map_err(reference_error)?;
                    let (_, result) = decoder.decode(program.evaluate(inputs).map_err(reference_error)?)
                        .map_err(reference_error)?;
                    online_ns += now.elapsed().as_nanos() as u64;
                    result
                }
                "logrow" => {
                    let (client, program) = logrow::prepare_compact_q7_logrow(&profile).map_err(reference_error)?;
                    issuance_ns += now.elapsed().as_nanos() as u64;
                    keys[0] += (9 + 9 + 1) * 16; // Input/output zero labels and one delta.
                    keys[1] += program.evaluator_material_bytes() as u64;
                    entry_exit += 2 * (32 + 9 * 16);
                    // Public dimensions, context and tree-level lengths. There
                    // is no external codec; this fixed-width value is an estimate.
                    public_metadata += 2 * 32 + 4 * 8 + 9 * 2 * 8;
                    let now = Instant::now();
                    let (inputs, decoder) = client.encode(value).map_err(reference_error)?;
                    let result = decoder.decode(program.evaluate(inputs).map_err(reference_error)?)
                        .map_err(reference_error)?;
                    online_ns += now.elapsed().as_nanos() as u64;
                    result
                }
                "curl" => {
                    let [mut left, mut right] = table.issue_reference().map_err(reference_error)?;
                    let entropy = Zeroizing::new(pllm_core::codec::random_residues(
                        table.rows() as u64, 1).map_err(reference_error)?);
                    let a = entropy[0] as usize;
                    let index = ((value + 128) as usize) >> levels;
                    let b = index.wrapping_sub(a) & (table.rows() - 1);
                    issuance_ns += now.elapsed().as_nanos() as u64;
                    for key in &mut keys { *key += (table.dealer_share_payload_bytes() / 2) as u64; }
                    public_metadata += 2 * (1 + 32 + 32); // Party, nonce, table digest.
                    entry_exit += 2 * (2 + 4); // Input indices and output shares.
                    let now = Instant::now();
                    let (l, lf) = left.start(a as u16).map_err(reference_error)?;
                    let (r, rf) = right.start(b as u16).map_err(reference_error)?;
                    phase[0] += lf.len();
                    phase[1] += rf.len();
                    let result = l.finish(&rf).map_err(reference_error)?
                        .wrapping_add(r.finish(&lf).map_err(reference_error)?) as i32 as i16;
                    online_ns += now.elapsed().as_nanos() as u64;
                    result
                }
                _ => unreachable!("validated method"),
            };
            output.push(actual);
        }
        if *output != *expected {
            return Err(invalid("activation protected outputs differ from their integer contract".into()));
        }
        let phases = if method == "curl" { vec![phase] } else { vec![] };
        let pieces = profile.pieces().iter().map(|piece|
            (piece.lower(), piece.upper(), piece.coefficients_q20())).collect::<Vec<_>>();
        let metadata = json!({
            "lanes": lanes, "profile_digest": pllm_types::Digest::from_sha256(profile.digest()),
            "pieces": pieces, "native_implementation_digest": implementation_digest(),
            "native_version": env!("CARGO_PKG_VERSION"),
            "party_key_payload_bytes": keys, "phase_frame_bytes": phases,
            "peer_payload_bits": if method == "curl" { lanes * 2 * 16 } else { 0 },
            "public_metadata_bytes_estimate": public_metadata,
            "client_entry_exit_payload_estimate_bytes": entry_exit,
            "evaluator_ciphertext_bytes": ciphertext, "evaluator_constant_label_bytes": constant_labels,
            "public_table_bytes_per_party": if method == "curl" { coefficients.len() * 4 } else { 0 },
            // Compact's existing 2M-wire / 1M-AND caps, geometric Vec growth,
            // simultaneous encoding/instruction allocations and evaluation
            // scratch fit this conservative serial-lane allocation ceiling.
            "protocol_allocation_estimate_bytes": if method == "compact" { 512 << 20 } else { 1 << 20 },
            "public_setup_ns": setup_ns, "issuance_ns": issuance_ns,
            "online_ns": online_ns, "clear_ns": clear_ns,
        });
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
    module.add_function(wrap_pyfunction!(activation_benchmark, module)?)
}
