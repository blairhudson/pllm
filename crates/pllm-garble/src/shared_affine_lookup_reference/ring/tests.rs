use super::{interval_fss, issue_inner, size_of, Error};
use crate::shared_affine_lookup_reference::{issue_ring, RingParty, RingResources, RingTable};

fn oracle(table: &RingTable, x: u64) -> [u64; 2] {
    for (j, interval) in table.boundaries.windows(2).enumerate() {
        if interval[0] <= x && x < interval[1] {
            return table.coefficients[j];
        }
    }
    panic!("input outside independently scanned complete partition");
}

fn check(table: &RingTable, r: u64, x: u64) -> RingResources {
    let expected = oracle(table, x);
    let budget = table.resources().unwrap();
    let ([a, b], actual): ([RingParty; 2], RingResources) =
        issue_ring(table, r, budget.peak_logical_allocation_bytes).unwrap();
    assert_eq!(actual, budget);
    assert_ne!(a.table.boundaries.as_ptr(), table.boundaries.as_ptr());
    assert_ne!(b.table.coefficients.as_ptr(), table.coefficients.as_ptr());
    assert_ne!(a.table.boundaries.as_ptr(), b.table.boundaries.as_ptr());
    let mask = (1u64 << table.bits) - 1;
    let opened = (x + r) & mask;
    let aa = a.evaluate(opened).unwrap();
    let bb = b.evaluate(opened).unwrap();
    assert_eq!(
        [
            aa[0].wrapping_add(bb[0]) & mask,
            aa[1].wrapping_add(bb[1]) & mask
        ],
        expected
    );
    actual
}

#[test]
fn exhaustive_small_ring_partitions_masks_inputs_arbitrary_coefficients() {
    for bits in 1..=3 {
        let n = 1u64 << bits;
        for cuts in 0..1u64 << (n - 1) {
            let mut boundaries = vec![0];
            for x in 1..n {
                if cuts & (1 << (x - 1)) != 0 {
                    boundaries.push(x);
                }
            }
            boundaries.push(n);
            let coefficients: Vec<_> = (0..boundaries.len() - 1)
                .map(|i| {
                    [
                        (i as u64 * 7 + 3) & (n - 1),
                        ((i as u64 * i as u64 * 3) ^ 5) & (n - 1),
                    ]
                })
                .collect();
            let table = RingTable::new(bits, &boundaries, &coefficients, usize::MAX).unwrap();
            assert_eq!(table.bits(), bits);
            assert_eq!(table.boundaries(), boundaries);
            assert_eq!(table.coefficients(), coefficients);
            for r in 0..n {
                for x in 0..n {
                    check(&table, r, x);
                }
            }
        }
    }
}

fn gelu_geometry(bits: u8) -> Vec<u64> {
    let n = 1u64 << bits;
    let mut b = vec![0, 1]; // source-like 511 intervals plus a public duplicate split
    b.extend((1..=255).map(|j| j * 64));
    b.push(n / 2);
    b.extend((1..=254).map(|j| n - 16320 + j * 64));
    b.push(n);
    assert_eq!(b.len(), 513);
    b
}

#[test]
fn ring50_source_gelu_like_512_intervals_endpoints_wrap_and_costs() {
    let n = 1u64 << 50;
    let boundaries = gelu_geometry(50);
    // Geometry only: deliberately synthetic nonmonotonic payloads, not source
    // GELU coefficients or a replacement for frozen descriptor conformance.
    let coefficients: Vec<_> = (0..512)
        .map(|i| [(n - ((i * 97 + 3) % 4096)) & (n - 1), (i / 73) % 2])
        .collect();
    let table = RingTable::new(50, &boundaries, &coefficients, usize::MAX).unwrap();
    let cost = table.resources().unwrap();
    assert_eq!(cost.public_table_payload_bytes, 12296);
    assert_eq!(cost.party_table_capacity_bytes, [12296; 2]);
    assert_eq!(cost.party_private_key_payload_bytes, [1340; 2]);
    assert_eq!(cost.dcf_evaluations_per_party, 512);
    assert_eq!(cost.dcf_path_levels_per_party, 25600);
    assert_eq!(cost.coefficient_multiplications_per_party, 1022);
    assert!(cost.peak_logical_allocation_bytes < 64 * 1024);
    println!("ring50/512 scalar logical envelope: {cost:?}");
    let inputs = [
        0,
        1,
        63,
        64,
        16319,
        16320,
        n / 2 - 1,
        n / 2,
        n - 16257,
        n - 16256,
        n - 65,
        n - 64,
        n - 1,
    ];
    for r in [0, 1, n / 2, n - 1] {
        for x in inputs {
            check(&table, r, x);
        }
    }
}

#[test]
fn fixed_shape_including_zero_deltas_and_capacity_edge_513() {
    let b = [0, 1, 3, 7, 16];
    let varied = RingTable::new(4, &b, &[[15, 0], [1, 14], [1, 14], [0, 7]], usize::MAX).unwrap();
    let zeros = RingTable::new(4, &b, &[[0; 2]; 4], usize::MAX).unwrap();
    assert_eq!(varied.resources().unwrap(), zeros.resources().unwrap());
    for r in 0..16 {
        for x in 0..16 {
            check(&varied, r, x);
            check(&zeros, r, x);
        }
    }

    let b: Vec<_> = (0..=513).map(|i| if i == 513 { 1024 } else { i }).collect();
    let c = [[0, 0]; 513];
    let table = RingTable::new(10, &b, &c, usize::MAX).unwrap();
    assert_eq!(table.intervals(), 513);
    assert_eq!(table.resources().unwrap().dcf_evaluations_per_party, 513);
    check(&table, 0, 0);
    check(&table, 1023, 1023);
}

#[test]
fn malformed_words_geometry_and_table_budget() {
    for (bits, b, c) in [
        (0, vec![0, 1], vec![[0, 0]]),
        (51, vec![0, 1], vec![[0, 0]]),
        (3, vec![1, 8], vec![[0, 0]]),
        (3, vec![0, 7], vec![[0, 0]]),
        (3, vec![0, 4, 4, 8], vec![[0, 0]; 3]),
        (3, vec![0, 7, 6, 8], vec![[0, 0]; 3]),
        (3, vec![0, 8], vec![]),
        (3, vec![0, 8], vec![[0, 0]; 2]),
        (3, vec![0, 8], vec![[8, 0]]),
        (3, vec![0, 8], vec![[0, 8]]),
    ] {
        assert!(matches!(
            RingTable::new(bits, &b, &c, usize::MAX),
            Err(Error::Domain)
        ));
    }
    assert!(matches!(
        RingTable::new(10, &[0, 1024], &[[0, 0]; 514], usize::MAX),
        Err(Error::Resource)
    ));
    assert!(matches!(
        RingTable::new(3, &[0, 8], &[[0, 0]], 0),
        Err(Error::Resource)
    ));
    let bytes = size_of::<RingTable>() + 2 * 8 + 16;
    assert!(RingTable::new(3, &[0, 8], &[[0, 0]], bytes).is_ok());
    assert!(matches!(
        RingTable::new(3, &[0, 8], &[[0, 0]], bytes - 1),
        Err(Error::Resource)
    ));
}

#[test]
fn admission_precedes_any_entropy_and_failures_drop_consuming_owners() {
    let table = RingTable::new(50, &[0, 3, 1 << 50], &[[1, 2], [7, 9]], usize::MAX).unwrap();
    let limit = table.resources().unwrap().peak_logical_allocation_bytes;
    for (r, budget) in [(0, 0), (0, limit - 1), (1 << 50, limit)] {
        assert!(issue_inner(
            &table,
            r,
            budget,
            |_, _, _, _| panic!("DCF entropy entered before admission"),
            |_| panic!("base entropy entered before admission")
        )
        .is_err());
    }
    assert!(matches!(
        issue_inner(
            &table,
            0,
            limit,
            |_, _, _, _| Err(Error::Randomness),
            |_| panic!("base draw after failed DCF")
        ),
        Err(Error::Randomness)
    ));
    let mut calls = 0;
    assert!(matches!(
        issue_inner(
            &table,
            0,
            limit,
            interval_fss::RingComparison::issue,
            |bytes| {
                calls += 1;
                bytes[0] = 255;
                Err(Error::Randomness)
            }
        ),
        Err(Error::Randomness)
    ));
    assert_eq!(calls, 1); // real DCF owners already exist, then drop on this failure

    let ([a, b], _) = issue_ring(&table, 0, limit).unwrap();
    assert_eq!(a.evaluate(1 << 50), Err(Error::Domain)); // consumes a even on error
    b.cancel();
    let ([a, b], _) = issue_ring(&table, 0, limit).unwrap();
    let ([fresh_a, fresh_b], _) = issue_ring(&table, 0, limit).unwrap();
    assert_ne!(*a.base, *fresh_a.base);
    a.cancel();
    b.cancel();
    fresh_a.cancel();
    fresh_b.cancel();
}
