//! Public numeric stream and fresh-material, in-process two-party probe.
use pllm_core::residual_silu_reference::Profile;
use pllm_garble::masked_wire_reference::{self as mw, Context, Step};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::io::{Read, Write};
use std::time::Instant;

fn hex(data: &[u8]) -> String {
    let mut result = String::with_capacity(2 * data.len());
    for byte in data {
        write!(result, "{byte:02x}").unwrap();
    }
    result
}
fn public_value(i: usize, bound: i64) -> i64 {
    let fixed = [
        -bound * (1 << 24),
        -(16 << 24) - 1,
        -(16 << 24),
        -(1 << 24),
        -4097,
        -4096,
        -1,
        0,
        1,
        4095,
        4096,
        1 << 24,
        (16 << 24) - 1,
        16 << 24,
        bound * (1 << 24),
    ];
    if i < fixed.len() {
        fixed[i]
    } else {
        (i as i64 * 982451653) % (2 * (bound << 24) + 1) - (bound << 24)
    }
}

fn measure(lanes: usize, samples: usize) -> Result<(), String> {
    if !(1..=16).contains(&samples) {
        return Err("samples".into());
    }
    let resource = mw::resources(lanes).map_err(|e| format!("{e:?}"))?;
    let profile = Profile::new();
    let gate: Vec<_> = (0..lanes).map(|i| public_value(i, 48)).collect();
    let up: Vec<_> = (0..lanes).rev().map(|i| public_value(i, 128)).collect();
    let expected: Vec<_> = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| profile.evaluate(g, u).unwrap())
        .collect();
    let digest = hex(&Sha256::digest(
        expected
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    ));
    let (mut issuance_ns, mut online_ns, mut clear_ns) = (Vec::new(), Vec::new(), Vec::new());
    let mut seen_first_frames = std::collections::HashSet::new();
    for sample in 0..samples {
        let context = Context {
            plan: [1; 32],
            session: [2; 32],
            operation: [3; 32],
            first_tensor_index: (sample * lanes) as u64,
        };
        let now = Instant::now();
        let (mut client, [mut a, mut b]) =
            mw::issue_reference(context, lanes).map_err(|e| format!("{e:?}"))?;
        issuance_ns.push(now.elapsed().as_nanos() as u64);
        let now = Instant::now();
        let input = client.encode(&gate, &up).map_err(|e| format!("{e:?}"))?;
        let mut left = a.start(&input).map_err(|e| format!("{e:?}"))?;
        let mut right = b.start(&input).map_err(|e| format!("{e:?}"))?;
        if !seen_first_frames.insert(hex(&Sha256::digest(&left))) {
            return Err("repeated issuance".into());
        }
        let mut bytes = left.len() + right.len();
        for round in 0..mw::ROUNDS {
            match (
                a.advance(&right).map_err(|e| format!("{e:?}"))?,
                b.advance(&left).map_err(|e| format!("{e:?}"))?,
            ) {
                (Step::Send(l), Step::Send(r)) => {
                    bytes += l.len() + r.len();
                    left = l;
                    right = r;
                }
                (Step::Done(out), Step::Done(_)) => {
                    if round != mw::ROUNDS - 1
                        || client.decode(&out).map_err(|e| format!("{e:?}"))? != expected
                    {
                        return Err("oracle mismatch".into());
                    }
                }
                _ => return Err("phase mismatch".into()),
            }
        }
        online_ns.push(now.elapsed().as_nanos() as u64);
        if bytes != resource.peer_frame_bytes || a.advance(&right).is_ok() {
            return Err("resource or replay mismatch".into());
        }
        let now = Instant::now();
        for (&g, &u) in gate.iter().zip(&up) {
            std::hint::black_box(profile.evaluate(g, u).unwrap());
        }
        clear_ns.push(now.elapsed().as_nanos() as u64);
    }
    println!(
        "{}",
        json!({"lanes":lanes,"samples":samples,"profile_digest":hex(&profile.digest()),
        "output_sha256":digest,"checked_outputs":lanes*samples,"rounds":mw::ROUNDS,"header_bytes":mw::HEADER,
        "party_key_bits":resource.party_key_bits,"party_key_payload_bytes":resource.party_key_bytes,
        "combined_allocation_estimate_bytes":resource.combined_allocation_estimate_bytes,
        "peer_payload_bits":resource.peer_payload_bits,"peer_frame_bytes":resource.peer_frame_bytes,
        "public_table_bytes":resource.public_table_bytes,"issuance_ns":issuance_ns,"online_ns":online_ns,"clear_ns":clear_ns,
        "scope":"bounded party-local in-process full block; client encoding and decoding included; no role network or decoder"})
    );
    Ok(())
}

fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() == 4 && args[1] == "--measure" {
        return measure(
            args[2].parse().map_err(|_| "lanes")?,
            args[3].parse().map_err(|_| "samples")?,
        );
    }
    if args.len() != 3 || args[1] != "12" || args[2] != "1024" {
        return Err("12 1024 | --measure lanes samples".into());
    }
    let profile = Profile::new();
    let manifest = json!({"fraction":12,"pieces":1024,"ring_bits":40,"effective_gate_bits":19,"gate_bound":48,"up_bound":128,
        "input_fraction":24,"index_fraction":6,"rounding":"floor","table":profile.table().as_slice(),"digest":hex(&profile.digest())});
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
                output
                    .write_all(
                        &values
                            .into_iter()
                            .flat_map(f32::to_le_bytes)
                            .collect::<Vec<_>>(),
                    )
                    .map_err(|e| e.to_string())?;
            }
            Err(_) => output.write_all(&[1]).map_err(|e| e.to_string())?,
        }
        output.flush().map_err(|e| e.to_string())?;
    }
    Ok(())
}
