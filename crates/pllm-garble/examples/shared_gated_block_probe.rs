//! Complete bounded nonlinear block; all intermediate values remain shared.
use pllm_garble::shared_gated_block_reference::{self as block, Progress, MASK};
use pllm_garble::shared_rescale_reference::{Backend, Context, Layout};
use sha2::{Digest, Sha256};
use std::{hint::black_box, time::Instant};

fn value(i: usize) -> i32 {
    let boundaries = [
        -16384, -16320, -8256, -192, -64, 0, 64, 192, 8256, 16320, 16384,
    ];
    boundaries
        .get(i)
        .copied()
        .unwrap_or(((i * 7919) % 32769) as i32 - 16384)
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(
        args.len(),
        5,
        "compact_dcf|interval_dcf fused|unfused lanes samples"
    );
    let backend = match args[1].as_str() {
        "compact_dcf" => Backend::CompactDcf,
        "interval_dcf" => Backend::IntervalDcf,
        _ => panic!("backend"),
    };
    let layout = match args[2].as_str() {
        "fused" => Layout::Fused,
        "unfused" => Layout::Unfused,
        _ => panic!("layout"),
    };
    let lanes: usize = args[3].parse().unwrap();
    let samples: usize = args[4].parse().unwrap();
    assert!((1..=16).contains(&samples));
    let cost = block::resources(backend, layout, lanes).unwrap();
    let gates: Vec<_> = (0..lanes).map(value).collect();
    let ups: Vec<_> = (0..lanes).map(|i| value(lanes - 1 - i)).collect();
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
        let [mut a, mut b] = block::issue_reference(backend, layout, context, lanes).unwrap();
        issuance_ns.push(start.elapsed().as_nanos());
        assert_eq!(a.key_payload_bytes(), cost.party_key_payload_bytes);
        let mut ag = Vec::new();
        let mut au = Vec::new();
        let mut bg = Vec::new();
        let mut bu = Vec::new();
        for (&g, &u) in gates.iter().zip(&ups) {
            let mut entropy = [0; 8];
            getrandom::fill(&mut entropy).unwrap();
            let x = u32::from_le_bytes(entropy[..4].try_into().unwrap()) & MASK;
            let y = u32::from_le_bytes(entropy[4..].try_into().unwrap()) & MASK;
            ag.push(x);
            au.push(y);
            bg.push((g as u32).wrapping_sub(x) & MASK);
            bu.push((u as u32).wrapping_sub(y) & MASK);
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
                (Progress::Open(af), Progress::Open(bf)) => frames = [af, bf],
                (Progress::Complete(aa), Progress::Complete(bb)) => {
                    assert_eq!(round + 1, cost.rounds);
                    output.extend(
                        aa.iter()
                            .zip(bb.iter())
                            .map(|(&x, &y)| x.wrapping_add(y) & MASK),
                    );
                }
                _ => panic!("phase mismatch"),
            }
        }
        online_ns.push(start.elapsed().as_nanos());
        assert_eq!(bytes, cost.peer_frame_bytes);
        assert_eq!(output.len(), lanes);
        assert_eq!(a.key_payload_bytes(), 0);
        assert_eq!(b.key_payload_bytes(), 0);
        let start = Instant::now();
        let expected: Vec<_> = gates
            .iter()
            .zip(&ups)
            .map(|(&g, &u)| {
                pllm_core::gated_multiply_q7(
                    pllm_core::rescale_q14_to_q7(black_box(g)).unwrap(),
                    pllm_core::rescale_q14_to_q7(black_box(u)).unwrap(),
                )
                .unwrap() as u32
                    & MASK
            })
            .collect();
        clear_ns.push(start.elapsed().as_nanos());
        assert_eq!(output, expected);
        let mut hash = Sha256::new();
        for word in output {
            hash.update(word.to_le_bytes());
        }
        let next = format!("{:x}", hash.finalize());
        if sample > 0 {
            assert_eq!(next, digest);
        }
        digest = next;
    }
    println!("{{\"schema\":\"pllm.shared_gated_block_probe.v1\",\"backend\":\"{}\",\"layout\":\"{}\",\"lanes\":{lanes},\"samples\":{samples},\"checked_outputs\":{},\"ring_bits\":24,\"rescale_lanes_per_output\":4,\"multiplications_per_output\":2,\"party_key_payload_bytes\":{},\"total_allocation_estimate_bytes\":{},\"peer_frame_bytes\":{},\"rounds\":{},\"input_sharing_payload_bytes\":{},\"output_sharing_payload_bytes\":{},\"issuance_ns\":{issuance_ns:?},\"online_ns\":{online_ns:?},\"clear_ns\":{clear_ns:?},\"output_sha256\":\"{digest}\",\"executable_sdk\":false}}",args[1],args[2],samples*lanes,cost.party_key_payload_bytes,cost.total_allocation_estimate_bytes,cost.peer_frame_bytes,cost.rounds,16*lanes,8*lanes);
}
