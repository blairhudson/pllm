//! Installed serial-lane mixed-mode benchmark. No secret handles escape.
use crate::invalid;
use pllm_core::{compact_product as compact, wavelet_gated_reference as wavelet};
use pllm_garble::{
    mixed_gated as native, shared_circuit::random_word, shared_rescale_reference::Context,
};
use pyo3::{prelude::*, types::PyBytes};
use serde::Deserialize;
use serde_json::json;
use std::{sync::OnceLock, time::Instant};
use zeroize::Zeroizing;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Configuration {
    method: String,
    levels: u8,
    max_pieces: u8,
    degree: u8,
    counts: Vec<u32>,
}
enum Profile {
    Compact(compact::Profile),
    Curl(wavelet::Profile),
    LogRow(wavelet::Profile),
}
impl Profile {
    fn native(&self) -> native::Profile<'_> {
        match self {
            Self::Compact(p) => native::Profile::Compact(p),
            Self::Curl(p) => native::Profile::Curl(p),
            Self::LogRow(p) => native::Profile::LogRow(p),
        }
    }
    fn digest(&self) -> [u8; 32] {
        match self {
            Self::Compact(p) => p.digest(),
            Self::Curl(p) | Self::LogRow(p) => p.digest(),
        }
    }
    fn evaluate(&self, g: i32, u: i32) -> PyResult<i32> {
        match self {
            Self::Compact(p) => p.evaluate(g, u).map_err(error),
            Self::Curl(p) | Self::LogRow(p) => p.evaluate(g, u).map_err(error),
        }
    }
}
fn digest() -> &'static str {
    static VALUE: OnceLock<String> = OnceLock::new();
    VALUE.get_or_init(|| {
        let sources = [
            include_bytes!("mixed_gated.rs").as_slice(),
            include_bytes!("../../pllm-core/src/compact_product.rs").as_slice(),
            include_bytes!("../../pllm-core/src/wavelet_gated_reference.rs").as_slice(),
            include_bytes!("../../pllm-core/src/fixed_point.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/mixed_gated.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/shared_circuit.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/shared_wide_product.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/shared_lut_reference.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/boolean.rs").as_slice(),
            include_bytes!("../../pllm-garble/src/logrow.rs").as_slice(),
            include_bytes!("../../../Cargo.lock").as_slice(),
        ];
        let mut bytes = Vec::new();
        for source in sources {
            bytes.extend((source.len() as u64).to_le_bytes());
            bytes.extend(source);
        }
        pllm_types::digest_bytes("pllm.mixed_gated.implementation.v1", &bytes).to_string()
    })
}
fn error(e: impl std::fmt::Debug) -> PyErr {
    invalid(format!("mixed gated reference: {e:?}"))
}

#[pyfunction]
fn mixed_gated_benchmark<'py>(
    py: Python<'py>,
    configuration: &str,
    gate_q18: &[u8],
    up_q18: &[u8],
    plan: &[u8],
    session: &[u8],
) -> PyResult<(Bound<'py, PyBytes>, String)> {
    if configuration.len() > 16_384
        || gate_q18.is_empty()
        || gate_q18.len() != up_q18.len()
        || gate_q18.len() % 4 != 0
        || gate_q18.len() > 256 * 4
    {
        return Err(invalid(
            "mixed gated requires bounded configuration and matching i32 frames".into(),
        ));
    }
    let config: Configuration = serde_json::from_str(configuration).map_err(error)?;
    let plan = plan
        .try_into()
        .map_err(|_| invalid("plan digest requires 32 bytes".into()))?;
    let session = session
        .try_into()
        .map_err(|_| invalid("session requires 32 bytes".into()))?;
    let inputs = [gate_q18, up_q18].map(|raw| {
        Zeroizing::new(
            raw.chunks_exact(4)
                .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
                .collect::<Vec<_>>(),
        )
    });
    for (&g, &u) in inputs[0].iter().zip(inputs[1].iter()) {
        wavelet::Profile::validate_inputs(g, u).map_err(error)?;
    }
    let (output,metadata)=py.detach(||->PyResult<_>{
        let start=Instant::now();
        let profile=match config.method.as_str() {
            "compact" if config.levels==0=>Profile::Compact(compact::Profile::fit(&config.counts,config.max_pieces,config.degree).map_err(error)?),
            "curl" if config.counts.is_empty() && config.max_pieces==0 && config.degree==0=>Profile::Curl(wavelet::Profile::new(config.levels).map_err(error)?),
            "logrow" if config.levels==0 && config.counts.is_empty() && config.max_pieces==0 && config.degree==0=>Profile::LogRow(wavelet::Profile::new(0).map_err(error)?),
            _=>return Err(invalid("unsupported mixed gated profile configuration".into())),
        };
        let setup_ns=start.elapsed().as_nanos() as u64;
        let start=Instant::now();
        let expected=Zeroizing::new(inputs[0].iter().zip(inputs[1].iter()).map(|(&g,&u)|profile.evaluate(g,u)).collect::<PyResult<Vec<_>>>()?);
        let clear_ns=start.elapsed().as_nanos() as u64;
        let mut reference=None; let mut output=Zeroizing::new(Vec::with_capacity(gate_q18.len()));
        let mut phases=Vec::new(); let mut issuance_ns=0; let mut online_ns=0;
        for (lane,((&g,&u),&want)) in inputs[0].iter().zip(inputs[1].iter()).zip(expected.iter()).enumerate() {
            let start=Instant::now();
            let ([mut a,mut b],r)=native::issue(profile.native(),Context{plan,session,operation:plan,first_tensor_index:lane as u64}).map_err(error)?;
            issuance_ns+=start.elapsed().as_nanos() as u64;
            if reference.as_ref().is_some_and(|old|old!=&r) {return Err(invalid("mixed gated topology depends on private input".into()));}
            if [a.key_bytes(),b.key_bytes()]!=r.key_bytes {return Err(invalid("mixed gated key accounting mismatch".into()));}
            let start=Instant::now();
            let sg=Zeroizing::new(random_word().map_err(error)?); let su=Zeroizing::new(random_word().map_err(error)?);
            let mut frames=[a.start(*sg,*su).map_err(error)?,b.start((i64::from(g) as u64).wrapping_sub(*sg),(i64::from(u) as u64).wrapping_sub(*su)).map_err(error)?];
            let mut result=None;
            for (i,expected_frames) in r.phases.iter().enumerate() {
                let lengths=[frames[0].len(),frames[1].len()];
                if &lengths!=expected_frames {return Err(invalid("mixed gated frame conservation failure".into()));}
                phases.push(lengths);
                match (a.advance(&frames[1]).map_err(error)?,b.advance(&frames[0]).map_err(error)?) {
                    (native::Progress::Open(x),native::Progress::Open(y))=>frames=[x,y],
                    (native::Progress::Complete(x),native::Progress::Complete(y)) if i+1==r.phases.len()=>result=Some(x.wrapping_add(y) as i32),
                    _=>return Err(invalid("mixed gated party phase mismatch".into())),
                }
            }
            online_ns+=start.elapsed().as_nanos() as u64;
            a.cancel(); b.cancel();
            if result!=Some(want) {return Err(invalid("mixed gated exact integer parity failure".into()));}
            output.extend(want.to_le_bytes()); reference=Some(r);
        }
        let r=reference.expect("nonempty lanes"); let lanes=inputs[0].len();
        let parts=r.parts.iter().map(|(name,k)|(*name,lanes*k[0],lanes*k[1])).collect::<Vec<_>>();
        let metadata=json!({"lanes":lanes,"profile_digest":pllm_types::Digest::from_sha256(profile.digest()),
            "party_key_payload_bytes":[lanes*r.key_bytes[0],lanes*r.key_bytes[1]],"key_payload_bytes_by_operation":parts,
            "phase_frame_bytes":phases,"peer_rounds":lanes*r.phases.len(),
            "peer_payload_bits":lanes*r.peer_payload_bits,
            "evaluator_ciphertext_bytes":lanes*r.ciphertext_bytes,"evaluator_constant_label_bytes":lanes*r.constant_bytes,
            "public_metadata_bytes_estimate":lanes*r.public_metadata_bytes,"public_table_bytes_per_party":r.public_table_bytes,
            "protocol_allocation_estimate_bytes":r.allocation_bytes,"client_entry_exit_payload_estimate_bytes":lanes*48,
            "client_mask_payload_estimate_bytes":lanes*16,
            "native_implementation_digest":digest(),"native_version":env!("CARGO_PKG_VERSION"),
            "public_setup_ns":setup_ns,"clear_ns":clear_ns,"issuance_ns":issuance_ns,"online_ns":online_ns});
        Ok((output,metadata))
    })?;
    py.check_signals()?;
    Ok((PyBytes::new(py, &output), metadata.to_string()))
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(mixed_gated_benchmark, module)?)
}
