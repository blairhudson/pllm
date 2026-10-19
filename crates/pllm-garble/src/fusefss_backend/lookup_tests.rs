use super::*;
use std::sync::atomic::Ordering;

fn context() -> Context {
    Context {
        plan: [1; 32],
        session: [2; 32],
        operation: [3; 32],
        first_tensor_index: 9,
    }
}

fn run<'a>(parties: &mut [Party<'a>; 2], inputs: &[u64]) -> [Output<'a>; 2] {
    let mask = (1u64 << parties[0].bits) - 1;
    let left: Vec<_> = inputs
        .iter()
        .enumerate()
        .map(|(i, _)| (mask - i as u64) & mask)
        .collect();
    let right: Vec<_> = inputs
        .iter()
        .zip(&left)
        .map(|(x, s)| x.wrapping_sub(*s) & mask)
        .collect();
    let a = parties[0].start(&left).unwrap();
    let b = parties[1].start(&right).unwrap();
    [
        parties[0].finish(b.as_bytes()).unwrap(),
        parties[1].finish(a.as_bytes()).unwrap(),
    ]
}

#[test]
fn full_small_rings_and_zero_extension_match_independent_array_oracle() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    for bits in 1..=5 {
        let n = 1u64 << bits;
        for len in [1, n as usize / 2, n as usize] {
            let values: Vec<_> = (0..len)
                .map(|i| ((i * i * 7 + 11 * i + 3) as u64) % n)
                .collect();
            let table = Table::new_zero_extended(bits, &values, [8; 32], &budget).unwrap();
            for input in 0..n {
                let mut parties = table.issue(1, context()).unwrap();
                let out = run(&mut parties, &[input]);
                let actual = (out[0].shares()[0] + out[1].shares()[0]) % n;
                assert_eq!(actual, values.get(input as usize).copied().unwrap_or(0));
                assert!(parties.iter().all(Party::is_consumed));
            }
        }
    }
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn ring50_chunk_edges_full8192_table_and_original_primitive_ceiling() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mask = (1u64 << 50) - 1;
    for len in [511usize, 512, 1024, 8192] {
        let values: Vec<_> = (0..len)
            .map(|i| ((i as u64).pow(3).wrapping_add(0x1234_5678_9abc)) & mask)
            .collect();
        let table = Table::new_zero_extended(50, &values, [4; 32], &budget).unwrap();
        let resources = table.resources(4).unwrap();
        assert_eq!(resources.chunks_per_lane, len.div_ceil(256));
        assert_eq!(resources.dcf_keys_per_party, 4 * len.div_ceil(256));
        assert!(
            resources.issuance_reservation_bytes + resources.table_reservation_bytes
                < MAX_ALLOCATION_BYTES
        );
        for i in 0..resources.chunks_per_lane {
            assert!(table.intervals(i) <= 513);
            assert_eq!(table.chunk(i).unwrap().intervals(), table.intervals(i));
        }
        let mut inputs = vec![
            0,
            1,
            510,
            511,
            512,
            1021,
            1022,
            len as u64 - 1,
            len as u64,
            mask,
        ];
        // Every public chunk contributes a distinct first/last value. This catches
        // missing chunks, summing only one lane, and secret-index-dependent scans.
        for i in 0..resources.chunks_per_lane {
            inputs.extend([i as u64 * 256, ((i + 1) * 256).min(len) as u64 - 1]);
        }
        for group in inputs.chunks(4) {
            let mut parties = table.issue(group.len(), context()).unwrap();
            let out = run(&mut parties, group);
            for (lane, &input) in group.iter().enumerate() {
                assert_eq!(
                    (out[0].shares()[lane] + out[1].shares()[lane]) & mask,
                    values.get(input as usize).copied().unwrap_or(0)
                );
            }
        }
    }
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn public_source_retirement_and_exported_owners_keep_exact_reservations() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let table = Table::new_zero_extended(50, &[19; 1024], [7; 32], &budget).unwrap();
    let r = table.resources(2).unwrap();
    let mut parties = table.issue(2, context()).unwrap();
    assert_eq!(
        budget.used_bytes(),
        r.table_reservation_bytes + 2 * r.party_reservation_bytes
    );
    drop(table);
    let a = parties[0].start(&[0, 1]).unwrap();
    let b = parties[1].start(&[511, 1022]).unwrap();
    assert_eq!(a.as_bytes().len(), r.peer_frame_bytes_per_party);
    assert_eq!(b.as_bytes().len(), r.peer_frame_bytes_per_party);
    let aa = parties[0].finish(b.as_bytes()).unwrap();
    let bb = parties[1].finish(a.as_bytes()).unwrap();
    assert_eq!((aa.shares()[1] + bb.shares()[1]) & ((1 << 50) - 1), 19);
    drop(parties);
    assert_eq!(
        budget.used_bytes(),
        2 * (r.peer_frame_bytes_per_party + r.output_bytes_per_party)
    );
    drop(a);
    drop(b);
    assert_eq!(budget.used_bytes(), 2 * r.output_bytes_per_party);
    drop(aa);
    drop(bb);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn malformed_order_replay_and_substituted_source_burn_the_entire_lookup() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let table = Table::new_zero_extended(50, &[3; 512], [7; 32], &budget).unwrap();
    let source = Table::new_zero_extended(50, &[3; 512], [8; 32], &budget).unwrap();
    let changed = Table::new_zero_extended(50, &[4; 512], [7; 32], &budget).unwrap();
    assert_ne!(table.digest(), source.digest());
    assert_ne!(table.digest(), changed.digest());
    for position in [0, 8, 40, 72, 73, 74, 75, 76, HEADER_BYTES + 6] {
        let mut p = table.issue(1, context()).unwrap();
        let _a = p[0].start(&[0]).unwrap();
        let b = p[1].start(&[1]).unwrap();
        let mut bad = b.as_bytes().to_vec();
        bad[position] ^= 0x80;
        assert!(matches!(p[0].finish(&bad), Err(Error::Context)));
        assert!(matches!(p[0].finish(b.as_bytes()), Err(Error::Consumed)));
    }
    for other in [&table, &source, &changed] {
        let mut p = table.issue(1, context()).unwrap();
        let mut q = other.issue(1, context()).unwrap();
        let _a = p[0].start(&[0]).unwrap();
        let b = q[1].start(&[1]).unwrap();
        assert!(matches!(p[0].finish(b.as_bytes()), Err(Error::Context)));
    }
    let mut p = table.issue(1, context()).unwrap();
    assert!(matches!(p[0].finish(&[]), Err(Error::Context)));
    assert!(p[0].is_consumed());
    p[1].cancel();
    assert!(matches!(p[1].start(&[0]), Err(Error::Consumed)));
    drop(p);
    drop(table);
    drop(source);
    drop(changed);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn seventeen_bit_public_composition_preserves_primitive_and_aggregate_caps() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let len = MAX_TABLE_ENTRIES;
    let values: Vec<_> = (0..len as u64).map(|i| (1u64 << 38) / (i + 1)).collect();
    let table = Table::new_zero_extended(50, &values, [19; 32], &budget).unwrap();
    let cost = table.resources(4).unwrap();
    assert_eq!(cost.chunks_per_lane, 512);
    assert_eq!(cost.dcf_keys_per_party, 2048);
    assert!(cost.issuance_reservation_bytes + cost.table_reservation_bytes < MAX_ALLOCATION_BYTES);
    let indices = [0, 255, (len - 1) as u64, len as u64];
    let expected = [values[0], values[255], values[len - 1], 0];
    let mut parties = table.issue(indices.len(), context()).unwrap();
    let outputs = run(&mut parties, &indices);
    for (i, &expected) in expected.iter().enumerate() {
        assert_eq!(
            (outputs[0].shares()[i] + outputs[1].shares()[i]) & ((1 << 50) - 1),
            expected
        );
    }
    drop((parties, outputs));
    assert_eq!(budget.used_bytes(), cost.table_reservation_bytes);
    drop(table);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn admission_geometry_capacity_and_allocation_failures_retire_all_chunks() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    for (bits, values) in [
        (0, vec![0]),
        (51, vec![0]),
        (4, vec![16]),
        (4, vec![0; 17]),
        (50, vec![]),
        (50, vec![0; MAX_TABLE_ENTRIES + 1]),
    ] {
        assert!(matches!(
            Table::new_zero_extended(bits, &values, [0; 32], &budget),
            Err(Error::Domain)
        ));
        assert_eq!(budget.used_bytes(), 0);
    }
    let table = Table::new_zero_extended(50, &[5; 513], [1; 32], &budget).unwrap();
    assert!(matches!(table.resources(0), Err(Error::Domain)));
    assert!(matches!(table.resources(MAX_LANES + 1), Err(Error::Domain)));
    let mut overflow = context();
    overflow.first_tensor_index = u64::MAX;
    assert!(matches!(table.issue(1, overflow), Err(Error::Domain)));
    let cost = table.resources(2).unwrap();
    let small =
        Budget::new(cost.issuance_reservation_bytes + cost.table_reservation_bytes - 1).unwrap();
    let insufficient = Table::new_zero_extended(50, &[5; 513], [1; 32], &small).unwrap();
    assert!(matches!(
        insufficient.issue(2, context()),
        Err(Error::Resource)
    ));
    drop(insufficient);
    assert_eq!(small.used_bytes(), 0);
    drop(table);

    let mut completed = false;
    for fail in 0..64 {
        budget.fail_after.store(fail, Ordering::SeqCst);
        let result = (|| {
            let table = Table::new_zero_extended(50, &[5; 513], [1; 32], &budget)?;
            let mut p = table.issue(2, context())?;
            let a = p[0].start(&[0, 0])?;
            let b = p[1].start(&[511, 512])?;
            let aa = p[0].finish(b.as_bytes())?;
            let bb = p[1].finish(a.as_bytes())?;
            assert_eq!((aa.shares()[1] + bb.shares()[1]) & ((1 << 50) - 1), 5);
            Ok::<_, Error>(())
        })();
        assert_eq!(budget.used_bytes(), 0, "failure index {fail}");
        if result.is_ok() {
            completed = true;
            break;
        }
        assert!(matches!(result, Err(Error::Allocation)));
    }
    assert!(completed);
    budget.fail_after.store(-1, Ordering::SeqCst);
}
