//! Clear public-value numeric/algebra probe; no provider or material protocol.
use pllm_core::joint_gated_reference::{
    algebra_audit, Profile, COEFFICIENT_FRACTION, GATE_BOUND, TAIL, UP_BOUND,
};
use serde_json::json;
use std::fmt::Write as _;
use std::io::{Read, Write};

fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() == 2 && args[1] == "--audit" {
        let a = algebra_audit();
        println!(
            "{}",
            json!({"checked_masked_pairs":a.checked_masked_pairs,
            "missing_wrap_wrong_outputs":a.missing_wrap_wrong_outputs,
            "sharewise_rounding":a.sharewise_rounding,"recovered_gate_up":a.recovered_gate_up,
            "protected_execution":false})
        );
        return Ok(());
    }
    if args.len() != 3 {
        return Err("fraction pieces | --audit".into());
    }
    let profile = Profile::new(
        args[1].parse().map_err(|_| "fraction")?,
        args[2].parse().map_err(|_| "pieces")?,
    )
    .map_err(|e| format!("{e:?}"))?;
    let mut digest = String::with_capacity(64);
    for b in profile.digest() {
        write!(digest, "{b:02x}").unwrap();
    }
    let manifest = json!({"fraction":profile.fraction(),"pieces":profile.pieces(),
        "coefficient_fraction":COEFFICIENT_FRACTION,"gate_bound":GATE_BOUND,"up_bound":UP_BOUND,
        "tail_boundary":TAIL,"intervals":profile.intervals(),"digest":digest,
        "maximum_numerator_absolute":profile.maximum_numerator_absolute().to_string()});
    let mut input = std::io::stdin().lock();
    let mut output = std::io::stdout().lock();
    writeln!(output, "{manifest}").map_err(|e| e.to_string())?;
    output.flush().map_err(|e| e.to_string())?;
    loop {
        let mut header = [0; 4];
        if input.read(&mut header[..1]).map_err(|e| e.to_string())? == 0 {
            break;
        }
        input
            .read_exact(&mut header[1..])
            .map_err(|e| e.to_string())?;
        let n = u32::from_le_bytes(header) as usize;
        if n == 0 || n > 1 << 20 {
            return Err("numeric frame bound".into());
        }
        let mut bytes = vec![0; 8 * n];
        input.read_exact(&mut bytes).map_err(|e| e.to_string())?;
        let values: Result<Vec<_>, _> = bytes[..4 * n]
            .chunks_exact(4)
            .zip(bytes[4 * n..].chunks_exact(4))
            .map(|(g, u)| {
                profile.evaluate_f32(
                    f32::from_le_bytes(g.try_into().unwrap()),
                    f32::from_le_bytes(u.try_into().unwrap()),
                )
            })
            .collect();
        match values {
            Ok(values) => {
                output.write_all(&[0]).map_err(|e| e.to_string())?;
                let raw: Vec<_> = values.into_iter().flat_map(f32::to_le_bytes).collect();
                output.write_all(&raw).map_err(|e| e.to_string())?;
            }
            Err(_) => output.write_all(&[1]).map_err(|e| e.to_string())?,
        }
        output.flush().map_err(|e| e.to_string())?;
    }
    Ok(())
}
