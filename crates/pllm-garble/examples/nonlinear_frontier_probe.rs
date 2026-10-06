//! Matched encoded-profile component costs, not matched complete protocols.
use pllm_core::fit_compact_silu_q7;
use pllm_garble::{compact_polynomial, logrow, shared_lut_reference::Table};
use serde_json::json;
use std::time::Instant;

fn main() {
    let method = std::env::args().nth(1).expect("method required");
    assert!(["compact", "logrow", "curl0", "curl2", "curl4"].contains(&method.as_str()));
    let profile = fit_compact_silu_q7(&[0; 257], 4).unwrap();
    let original: Vec<i32> = (0..512)
        .map(|index| profile.evaluate((index.min(256) as i16) - 128).unwrap() as i32)
        .collect();
    let levels: usize = match method.as_str() {
        "curl2" => 2,
        "curl4" => 4,
        _ => 0,
    };
    let coefficients: Vec<u32> = original
        .chunks_exact(1 << levels)
        .map(|group| {
            let sum: i64 = group.iter().map(|&x| i64::from(x)).sum();
            (sum as f64 / group.len() as f64).round_ties_even() as i32 as u32
        })
        .collect();
    let table = Table::new(&coefficients).unwrap();
    let mut rows = Vec::new();
    for value in (-128_i16..=128).step_by(16) {
        let expected = profile.evaluate(value).unwrap();
        let start = Instant::now();
        let (actual, material, peer_bytes, issue, online) = match method.as_str() {
            "compact" => {
                let (client, program) =
                    compact_polynomial::prepare_compact_q7_polynomial(&profile).unwrap();
                let material = program.evaluator_ciphertext_bytes().unwrap();
                let issue = start.elapsed().as_secs_f64();
                let start = Instant::now();
                let (inputs, decoder) = client.encode(value).unwrap();
                let (_, actual) = decoder.decode(program.evaluate(inputs).unwrap()).unwrap();
                (actual, material, None, issue, start.elapsed().as_secs_f64())
            }
            "logrow" => {
                let (client, program) = logrow::prepare_compact_q7_logrow(&profile).unwrap();
                let material = program.evaluator_material_bytes() as u64;
                let issue = start.elapsed().as_secs_f64();
                let start = Instant::now();
                let (inputs, decoder) = client.encode(value).unwrap();
                let actual = decoder.decode(program.evaluate(inputs).unwrap()).unwrap();
                (actual, material, None, issue, start.elapsed().as_secs_f64())
            }
            _ => {
                let [mut left, mut right] = table.issue_reference().unwrap();
                let material = table.dealer_share_payload_bytes() as u64;
                let issue = start.elapsed().as_secs_f64();
                let index = (value + 128) as usize >> levels;
                let mut entropy = [0; 2];
                getrandom::fill(&mut entropy).unwrap();
                let a = usize::from(u16::from_le_bytes(entropy)) & (table.rows() - 1);
                let b = index.wrapping_sub(a) & (table.rows() - 1);
                let start = Instant::now();
                let (l, lf) = left.start(a as u16).unwrap();
                let (r, rf) = right.start(b as u16).unwrap();
                let actual = l.finish(&rf).unwrap().wrapping_add(r.finish(&lf).unwrap());
                let elapsed = start.elapsed().as_secs_f64();
                assert_eq!(actual, coefficients[index]);
                (
                    actual as i32 as i16,
                    material,
                    Some(lf.len() + rf.len()),
                    issue,
                    elapsed,
                )
            }
        };
        if levels == 0 {
            assert_eq!(actual, expected);
        }
        rows.push(
            json!({"public_input_q7": value, "expected_q7": expected, "actual_q7": actual,
            "material_body_bytes": material, "peer_frame_bytes": peer_bytes,
            "issuance_seconds": issue, "local_online_seconds": online}),
        );
    }
    println!(
        "{}",
        json!({"schema": "pllm.nonlinear_frontier_probe.v1", "method": method,
        "profile_digest": profile.digest(), "requested_pieces": 4, "samples": rows,
        "input_conversion_included": false, "gated_product_included": false,
        "material_scope": if method.starts_with("curl") { "both parties' dealer-share payload" }
            else if method == "compact" { "half-gate evaluator ciphertext body only" }
            else { "LogRow evaluator material body only" }})
    );
}
