//! Native Curl Fig. 4 lookup, with public Haar compression from §4.2.1.
//! Inputs to each lookup are freshly shared, already-truncated indices.
use pllm_garble::shared_lut_reference::Table;
use std::{hint::black_box, time::Instant};

fn main() {
    let original: Vec<i32> = (0..512)
        .map(|i| {
            let x = (i as f64 - 256.0) / 64.0;
            (65536.0 * x / (1.0 + (-x).exp())).round_ties_even() as i32
        })
        .collect();
    println!("{{\"schema\":\"pllm.curl_lookup_probe.v1\",\"input_rows\":512,\"output_fraction_bits\":16,\"cases\":[");
    for j in 0..=4 {
        let block = 1 << j;
        let coefficients: Vec<u32> = original
            .chunks_exact(block)
            .map(|group| {
                // Public offline Haar means; one final ties-to-even Q16 encoding.
                let sum: i64 = group.iter().map(|v| i64::from(*v)).sum();
                (sum as f64 / block as f64).round_ties_even() as i32 as u32
            })
            .collect();
        let table = Table::new(&coefficients).unwrap();
        let mut issue_ns = 0u128;
        let mut online_ns = 0u128;
        let mut baseline_ns = 0u128;
        let mut worst = 0f64;
        let mut absolute_sum = 0f64;
        let mut exact = 0;
        let mut online_bytes = 0;
        for (x, original_value) in original.iter().enumerate() {
            let index = x >> j;
            let start = Instant::now();
            let [mut left, mut right] = table.issue_reference().unwrap();
            issue_ns += start.elapsed().as_nanos();
            // Fresh client input sharing, excluded from the two-party protocol bodies.
            let mut entropy = [0; 2];
            getrandom::fill(&mut entropy).unwrap();
            let a = usize::from(u16::from_le_bytes(entropy)) & (table.rows() - 1);
            let b = index.wrapping_sub(a) & (table.rows() - 1);
            let start = Instant::now();
            let (l, lf) = left.start(a as u16).unwrap();
            let (r, rf) = right.start(b as u16).unwrap();
            let y = l.finish(&rf).unwrap().wrapping_add(r.finish(&lf).unwrap());
            online_ns += start.elapsed().as_nanos();
            online_bytes += lf.len() + rf.len();
            let start = Instant::now();
            black_box(coefficients[black_box(index)]);
            baseline_ns += start.elapsed().as_nanos();
            assert_eq!(y, coefficients[index]);
            let error = (f64::from(y as i32) - f64::from(*original_value)).abs() / 65536.0;
            worst = worst.max(error);
            absolute_sum += error;
            exact += usize::from(y as i32 == *original_value);
        }
        if j > 0 {
            println!(",");
        }
        print!("{{\"haar_levels\":{j},\"table_rows\":{},\"public_table_bytes\":{},\"dealer_share_payload_bytes_per_lookup\":{},\"peer_frame_bytes_per_lookup\":{},\"samples\":512,\"encoded_exact_rows\":{exact},\"worst_absolute_error\":{worst},\"mean_absolute_error\":{},\"mean_issuance_ns\":{},\"mean_local_two_party_ns\":{},\"mean_clear_lookup_ns\":{}}}",
            table.rows(), coefficients.len()*4, table.dealer_share_payload_bytes(), online_bytes/512,
            absolute_sum/512.0, issue_ns as f64/512.0, online_ns as f64/512.0, baseline_ns as f64/512.0);
    }
    println!("]}}");
}
