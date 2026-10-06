//! Fresh-process complete-block measurements, separate from decoder quality.
use pllm_core::piecewise_gated_reference::Profile;
use pllm_garble::{
    shared_affine_lookup_reference::Layout,
    shared_piecewise_gated_reference::{self as block, Progress},
    shared_rescale_reference::Context,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fmt::Write as _, hint::black_box, sync::Arc, time::Instant};

fn value(index: usize, bound: i32) -> i32 {
    let fixed = [
        -bound * 65536,
        -8 * 65536 - 1,
        -8 * 65536,
        -65536,
        -192,
        -128,
        -64,
        0,
        64,
        128,
        192,
        65536,
        8 * 65536 - 1,
        8 * 65536,
        bound * 65536,
    ];
    fixed
        .get(index)
        .copied()
        .unwrap_or(((index as i64 * 7919) % (i64::from(bound) * 131072 + 1)) as i32 - bound * 65536)
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(
        args.len(),
        6,
        "fraction pieces separate|vector lanes samples"
    );
    let p = Arc::new(Profile::new(args[1].parse().unwrap(), args[2].parse().unwrap()).unwrap());
    let layout = match args[3].as_str() {
        "separate" => Layout::Separate,
        "vector" => Layout::Vector,
        _ => panic!("layout"),
    };
    let lanes: usize = args[4].parse().unwrap();
    let samples: usize = args[5].parse().unwrap();
    assert!((1..=5).contains(&samples));
    let cost = block::resources(&p, layout, lanes).unwrap();
    let gates: Vec<_> = (0..lanes).map(|i| value(i, 48)).collect();
    let ups: Vec<_> = (0..lanes).map(|i| value(lanes - 1 - i, 128)).collect();
    let mut issuance_ns = Vec::new();
    let mut online_ns = Vec::new();
    let mut clear_ns = Vec::new();
    let mut digest = String::new();
    for sample in 0..samples {
        let context = Context {
            plan: [1; 32],
            session: [2; 32],
            operation: [3; 32],
            first_tensor_index: (sample * lanes) as u64,
        };
        let start = Instant::now();
        let [mut a, mut b] = block::issue_reference(p.clone(), layout, context, lanes).unwrap();
        issuance_ns.push(start.elapsed().as_nanos());
        assert_eq!(a.key_payload_bytes(), cost.party_key_payload_bytes);
        let mut ag = Vec::new();
        let mut au = Vec::new();
        let mut bg = Vec::new();
        let mut bu = Vec::new();
        for (&g, &u) in gates.iter().zip(&ups) {
            let mut bytes = [0; 8];
            getrandom::fill(&mut bytes).unwrap();
            let x = u32::from_le_bytes(bytes[..4].try_into().unwrap());
            let y = u32::from_le_bytes(bytes[4..].try_into().unwrap());
            ag.push(x);
            au.push(y);
            bg.push((g as u32).wrapping_sub(x));
            bu.push((u as u32).wrapping_sub(y));
        }
        let start = Instant::now();
        let mut frames = [a.start(&ag, &au).unwrap(), b.start(&bg, &bu).unwrap()];
        let mut bytes = 0;
        let mut output = Vec::new();
        for round in 0..cost.rounds {
            bytes += frames.iter().map(Vec::len).sum::<usize>();
            match (
                a.advance(&frames[1]).unwrap(),
                b.advance(&frames[0]).unwrap(),
            ) {
                (Progress::Open(aa), Progress::Open(bb)) => frames = [aa, bb],
                (Progress::Complete(aa), Progress::Complete(bb)) => {
                    assert_eq!(round + 1, cost.rounds);
                    output.extend(aa.iter().zip(bb.iter()).map(|(&a, &b)| a.wrapping_add(b)));
                }
                _ => panic!("phase mismatch"),
            }
        }
        online_ns.push(start.elapsed().as_nanos());
        assert_eq!(bytes, cost.peer_frame_bytes);
        assert_eq!(a.key_payload_bytes(), 0);
        assert_eq!(b.key_payload_bytes(), 0);
        let start = Instant::now();
        let expected: Vec<_> = gates
            .iter()
            .zip(&ups)
            .map(|(&g, &u)| p.evaluate(black_box(g), black_box(u)).unwrap() as u32)
            .collect();
        clear_ns.push(start.elapsed().as_nanos());
        assert_eq!(output, expected);
        let mut hash = Sha256::new();
        for value in output {
            hash.update(value.to_le_bytes());
        }
        let next = format!("{:x}", hash.finalize());
        if sample > 0 {
            assert_eq!(next, digest);
        }
        digest = next;
    }
    let mut profile_digest = String::with_capacity(64);
    for byte in p.digest() {
        write!(profile_digest, "{byte:02x}").unwrap();
    }
    println!(
        "{}",
        json!({"schema":"pllm.shared_piecewise_gated_probe.v1","fraction":p.fraction(),"pieces":p.pieces(),
        "profile_digest":profile_digest,"layout":args[3],"lanes":lanes,"samples":samples,
        "checked_outputs":lanes*samples,"output_sha256":digest,"rounds":cost.rounds,"ring_bits":32,
        "party_key_payload_bytes":cost.party_key_payload_bytes,"lookup_key_payload_bytes":cost.lookup_key_payload_bytes,
        "rescale_key_payload_bytes":cost.rescale_key_payload_bytes,"triple_key_payload_bytes":24*lanes,
        "total_allocation_estimate_bytes":cost.total_allocation_estimate_bytes,"peer_frame_bytes":cost.peer_frame_bytes,
        "input_sharing_payload_bytes":16*lanes,"output_sharing_payload_bytes":8*lanes,
        "issuance_ns":issuance_ns,"online_ns":online_ns,"clear_ns":clear_ns,"executable_sdk":false})
    );
}
