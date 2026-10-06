//! Replayable CPU microprobe for the scoped SIGMA/FuseFSS helper comparison.
//! Deterministic public stand-ins; secret masks/keys are always OS-random.
use pllm_garble::shared_rescale_reference::{
    Backend, Context, Descriptor, Layout, Progress, Rounding,
};
use sha2::{Digest, Sha256};
use std::{hint::black_box, time::Instant};

fn value(index: usize, bits: u8, shift: u8) -> u32 {
    let mask = ((1u64 << bits) - 1) as u32;
    let half = 1u32 << (shift - 1);
    let sign = 1u32 << (bits - 1);
    let fixed = [
        0,
        mask,
        sign,
        sign - 1,
        half,
        half - 1,
        half + 1,
        half.wrapping_neg(),
        half.wrapping_neg().wrapping_sub(1),
        half.wrapping_neg().wrapping_add(1),
        half.wrapping_mul(3),
        half.wrapping_mul(3).wrapping_neg(),
    ];
    fixed.get(index).copied().unwrap_or_else(|| {
        (index as u32)
            .wrapping_mul(2_654_435_761)
            .rotate_left((index % 31) as u32)
    }) & mask
}

fn oracle(x: u32, bits: u8, shift: u8, rounding: Rounding) -> [u32; 2] {
    let signed = (i64::from(x) << (64 - bits)) >> (64 - bits);
    let divisor = 1i64 << shift;
    let floor = signed.div_euclid(divisor);
    let remainder = signed.rem_euclid(divisor);
    let up = rounding == Rounding::TiesEven
        && (2 * remainder > divisor || (2 * remainder == divisor && floor % 2 != 0));
    [
        (floor + i64::from(up)) as u32 & ((1u64 << bits) - 1) as u32,
        u32::from(signed >= 0),
    ]
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert!(
        (7..=8).contains(&args.len()),
        "bits shift floor|ties_even fused|unfused lanes samples [prefix_dpf|compact_dcf]"
    );
    let bits: u8 = args[1].parse().unwrap();
    let shift: u8 = args[2].parse().unwrap();
    let rounding = match args[3].as_str() {
        "floor" => Rounding::Floor,
        "ties_even" => Rounding::TiesEven,
        _ => panic!("rounding"),
    };
    let layout = match args[4].as_str() {
        "fused" => Layout::Fused,
        "unfused" => Layout::Unfused,
        _ => panic!("layout"),
    };
    let lanes: usize = args[5].parse().unwrap();
    let samples: usize = args[6].parse().unwrap();
    let backend_name = args.get(7).map(String::as_str).unwrap_or("prefix_dpf");
    let backend = match backend_name {
        "prefix_dpf" => Backend::PrefixDpf,
        "compact_dcf" => Backend::CompactDcf,
        _ => panic!("backend"),
    };
    assert!((1..=32).contains(&samples));
    let d = Descriptor::new(bits, shift, rounding)
        .unwrap()
        .with_backend(backend);
    let cost = d.resources(layout, lanes).unwrap();
    let mask = ((1u64 << bits) - 1) as u32;
    let values: Vec<_> = (0..lanes).map(|i| value(i, bits, shift)).collect();
    let mut clear_ns = Vec::new();
    let mut issue_ns = Vec::new();
    let mut online_ns = Vec::new();
    let mut output_digest = String::new();
    for sample in 0..samples {
        let context = Context {
            plan: [1; 32],
            session: [2; 32],
            operation: [3; 32],
            first_tensor_index: (sample * lanes) as u64,
        };
        let start = Instant::now();
        let [mut left, mut right] = d.issue_reference(layout, context, lanes).unwrap();
        issue_ns.push(start.elapsed().as_nanos());
        assert_eq!(left.key_payload_bytes(), cost.party_key_payload_bytes);
        assert_eq!(right.key_payload_bytes(), cost.party_key_payload_bytes);
        let mut a = Vec::with_capacity(lanes);
        let mut b = Vec::with_capacity(lanes);
        for &x in &values {
            let mut random = [0; 4];
            getrandom::fill(&mut random).unwrap();
            let share = u32::from_le_bytes(random) & mask;
            a.push(share);
            b.push(x.wrapping_sub(share) & mask);
        }
        let start = Instant::now();
        let mut frames = [left.start(&a).unwrap(), right.start(&b).unwrap()];
        let mut online_bytes = 0;
        let mut outputs = Vec::with_capacity(lanes);
        for _ in 0..cost.rounds {
            online_bytes += frames[0].len() + frames[1].len();
            match (
                left.advance(&frames[1]).unwrap(),
                right.advance(&frames[0]).unwrap(),
            ) {
                (Progress::Open(a), Progress::Open(b)) => frames = [a, b],
                (Progress::Complete(a), Progress::Complete(b)) => {
                    outputs.extend(a.iter().zip(&b).map(|(a, b)| {
                        [
                            a.rescaled.wrapping_add(b.rescaled) & mask,
                            a.nonnegative.wrapping_add(b.nonnegative) & mask,
                        ]
                    }))
                }
                _ => panic!("phase mismatch"),
            }
        }
        online_ns.push(start.elapsed().as_nanos());
        assert_eq!(online_bytes, cost.peer_frame_bytes);
        assert_eq!(outputs.len(), lanes);
        let start = Instant::now();
        let expected: Vec<_> = values
            .iter()
            .map(|&x| black_box(oracle(black_box(x), bits, shift, rounding)))
            .collect();
        clear_ns.push(start.elapsed().as_nanos());
        assert_eq!(outputs, expected);
        assert_eq!(left.key_payload_bytes(), 0);
        let mut hash = Sha256::new();
        for pair in outputs {
            for word in pair {
                hash.update(word.to_le_bytes());
            }
        }
        let digest = format!("{:x}", hash.finalize());
        if sample > 0 {
            assert_eq!(output_digest, digest);
        }
        output_digest = digest;
    }
    println!("{{\"schema\":\"pllm.shared_rescale_probe.v1\",\"backend\":\"{backend_name}\",\"bits\":{bits},\"shift\":{shift},\"rounding\":\"{}\",\"layout\":\"{}\",\"lanes\":{lanes},\"samples\":{samples},\"checked_outputs\":{},\"output_sha256\":\"{output_digest}\",\"rounds\":{},\"party_key_payload_bytes\":{},\"party_allocation_estimate_bytes\":{},\"total_allocation_estimate_bytes\":{},\"peer_frame_bytes\":{},\"comparison_widths\":{:?},\"issuance_ns\":{issue_ns:?},\"online_ns\":{online_ns:?},\"clear_ns\":{clear_ns:?},\"executable_sdk\":false}}",
        args[3], args[4], samples * lanes, cost.rounds, cost.party_key_payload_bytes,
        cost.party_allocation_estimate_bytes, cost.total_allocation_estimate_bytes, cost.peer_frame_bytes, cost.comparison_widths);
}
