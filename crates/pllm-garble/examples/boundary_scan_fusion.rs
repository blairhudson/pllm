//! Public-API baseline only: no fused scan is exposed by private_pages.
use pllm_core::kernels::{Executor, Matrix};
use pllm_garble::private_pages::{issue_packed, Server};
use serde_json::json;
use std::{sync::Arc, time::Instant};

fn random_ring(n: usize) -> Vec<u32> {
    let mut bytes = zeroize::Zeroizing::new(vec![0_u8; 4 * n]);
    getrandom::fill(&mut bytes).unwrap();
    bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn reference(weights: &[i8], columns: usize, input: &[u32]) -> Vec<u32> {
    weights
        .chunks_exact(columns)
        .map(|row| {
            row.iter().zip(input).fold(0_u32, |sum, (&w, &x)| {
                sum.wrapping_add((w as i32 as u32).wrapping_mul(x))
            })
        })
        .collect()
}

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|s| s.parse().expect("unsigned integer argument"))
        .collect();
    assert_eq!(
        args.len(),
        4,
        "usage: boundary_scan_fusion ROWS COLUMNS INDEX REPETITIONS"
    );
    let (rows, columns, index, repetitions) = (args[0], args[1], args[2], args[3]);
    assert!((1..=4096).contains(&rows) && (1..=4096).contains(&columns));
    assert!(rows * columns <= 4_194_304 && index < rows && (1..=32).contains(&repetitions));
    let setup = Instant::now();
    let weights: Vec<u8> = (0..rows * columns)
        .map(|i| (i.wrapping_mul(131) + i / columns) as u8)
        .collect();
    let scales: Vec<u8> = (0..rows)
        .flat_map(|i| (0.001_f32 + i as f32 / 1024.0).to_le_bytes())
        .collect();
    let matrix = Arc::new(Matrix::new(&weights, rows, columns).unwrap());
    let mut roots: Vec<_> = (0..2)
        .map(|p| Server::from_matrix(Arc::clone(&matrix), &scales, p).unwrap())
        .collect();
    assert_eq!(Arc::strong_count(&matrix), 3);
    // Independent interleaved-table oracle; deliberately extra reference storage.
    let mut records = Vec::with_capacity(rows * (columns + 4));
    for (w, s) in weights.chunks_exact(columns).zip(scales.chunks_exact(4)) {
        records.extend_from_slice(w);
        records.extend_from_slice(s);
    }
    let mut raw_roots: Vec<_> = (0..2)
        .map(|p| Server::new(&records, columns + 4, p).unwrap())
        .collect();
    assert!(roots[0].descriptor() == raw_roots[0].descriptor());
    let context = [71; 32];
    let mut servers: Vec<_> = roots
        .iter_mut()
        .map(|s| s.bind_context(context).unwrap())
        .collect();
    let mut raw: Vec<_> = raw_roots
        .iter_mut()
        .map(|s| s.bind_context(context).unwrap())
        .collect();
    assert!(roots[0].bind_context(context).is_err());
    let mut wrong_context = roots[0].bind_context([72; 32]).unwrap();
    let executor = Executor::new(1, true).unwrap();
    let setup_ns = setup.elapsed().as_nanos();
    let mut lookup_ns = 0_u128;
    let mut head_ns = 0_u128;
    for _ in 0..repetitions {
        // OS entropy for DPF and each role's random head input are separate calls.
        let (qa, qb, decoder) = issue_packed(servers[0].descriptor(), index).unwrap();
        let queries = [&qa, &qb];
        assert!(servers[0].evaluate(&qb).is_err());
        assert!(wrong_context.evaluate(&qa).is_err());
        assert!(servers[0].evaluate(&qa[..10]).is_err());
        let mut replies = Vec::new();
        for party in 0..2 {
            let input = zeroize::Zeroizing::new(random_ring(columns));
            let start = Instant::now();
            let reply = servers[party]
                .evaluate_batch(&[queries[party]])
                .unwrap()
                .remove(0);
            lookup_ns += start.elapsed().as_nanos();
            assert_eq!(reply, raw[party].evaluate(queries[party]).unwrap());
            assert!(servers[party].evaluate(queries[party]).is_err());
            let start = Instant::now();
            let head = matrix.wrap32(&executor, &input, 1).unwrap();
            head_ns += start.elapsed().as_nanos();
            assert_eq!(head, reference(matrix.as_signed_slice(), columns, &input));
            // Exact additive recombination, independent of XOR lookup shares.
            let q: Vec<u32> = (0..columns)
                .map(|i| (i as i32 % 255 - 127) as u32)
                .collect();
            let complement = zeroize::Zeroizing::new(
                q.iter()
                    .zip(input.iter())
                    .map(|(&q, &r)| q.wrapping_sub(r))
                    .collect::<Vec<_>>(),
            );
            let other = matrix.wrap32(&executor, &complement, 1).unwrap();
            let combined: Vec<_> = head
                .iter()
                .zip(other)
                .map(|(&a, b)| a.wrapping_add(b))
                .collect();
            assert_eq!(combined, reference(matrix.as_signed_slice(), columns, &q));
            replies.push(reply);
        }
        assert_eq!(
            decoder.decode(&replies[0], &replies[1]).unwrap(),
            records[index * (columns + 4)..(index + 1) * (columns + 4)]
        );
        let (ca, cb, _) = issue_packed(servers[0].descriptor(), index).unwrap();
        for (s, q) in servers.iter_mut().zip([ca, cb]) {
            s.cancel(&q).unwrap();
            assert!(s.evaluate(&q).is_err());
        }
        let (fa, fb, decoder) = issue_packed(servers[0].descriptor(), index).unwrap();
        let mut a = servers[0].evaluate(&fa).unwrap();
        let b = servers[1].evaluate(&fb).unwrap();
        a[17] ^= 1;
        assert!(decoder.decode(&a, &b).is_err()); // decoder ownership consumed on error
        assert!(servers[0].evaluate(&fa).is_err());
        assert!(servers[1].evaluate(&fb).is_err());
    }
    println!(
        "{}",
        json!({
        "schema": "boundary-scan-fusion-probe-v1", "status": "separate-call-baseline-only",
            "fused_scan_executed": false, "baseline_correct": true,
            "rows": rows, "columns": columns, "index": index, "repetitions": repetitions,
            "parties_checked": 2, "native_simd": executor.simd(),
            "setup_ns": setup_ns, "baseline_lookup_ns_both_parties": lookup_ns,
            "baseline_head_ns_both_parties": head_ns,
            "timing_scope": "separate public calls only; oracle and extra retirement checks excluded; not SDK timing",
            "physical_read_instrumentation": false
        })
    );
}
