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

// Independent encoding oracle: enumerate actual boundary and coefficient rows,
// not Table::intervals or the constant-time census. No private keys are needed.
fn explicit_chunk_census(table: &Table<'_>) -> (usize, usize, usize, usize) {
    let mut chunks = 0;
    let mut intervals = 0;
    let mut payload = 0;
    let mut largest = 0;
    for (chunk, values) in table.values.chunks(CHUNK_ENTRIES).enumerate() {
        let start = chunk * CHUNK_ENTRIES;
        let mut boundaries = vec![0];
        let mut coefficients = Vec::new();
        if start != 0 {
            coefficients.push([0, 0]);
            boundaries.push(start as u64);
        }
        for (offset, &value) in values.iter().enumerate() {
            coefficients.push([value, 0]);
            boundaries.push((start + offset + 1) as u64);
        }
        if *boundaries.last().unwrap() != 1u64 << table.bits {
            coefficients.push([0, 0]);
            boundaries.push(1u64 << table.bits);
        }
        let actual = table.chunk_buffers(chunk).unwrap();
        assert_eq!(actual.0, boundaries);
        assert_eq!(actual.1, coefficients);
        if let Some(lowered) = &table.lowered {
            assert_eq!(lowered.chunks[chunk].boundaries(), boundaries.as_slice());
            assert_eq!(lowered.chunks[chunk].coefficients(), coefficients.as_slice());
        }
        assert_eq!(boundaries.len(), coefficients.len() + 1);
        let bytes =
            boundaries.len() * size_of::<u64>() + coefficients.len() * size_of::<[u64; 2]>();
        chunks += 1;
        intervals += coefficients.len();
        payload += bytes;
        largest = largest.max(bytes);
    }
    (chunks, intervals, payload, largest)
}

fn explicit_resources(
    table: &Table<'_>,
    lanes: usize,
    census: (usize, usize, usize, usize),
) -> Resources {
    let (chunks, intervals, payload, largest) = census;
    let dcf = RingComparison::resources(table.bits, table.bits).unwrap();
    let keys = chunks * lanes;
    let public = if table.lowered.is_some() {
        keys * size_of::<Arc<RingTable>>() + size_of::<Arc<Lowered<'_>>>()
    } else {
        payload * lanes
    };
    let frame = HEADER_BYTES + keys * 8;
    let output = lanes * 8;
    let party = keys * (size_of::<Unit>() + dcf.key_capacity_bytes)
        + public
        + keys * 8
        + frame
        + output
        + FIXED_BYTES
        + dcf.issuance_scratch_bytes;
    let backing = if table.lowered.is_some() {
        size_of::<Lowered<'_>>()
            + 2 * size_of::<usize>()
            + chunks
                * (size_of::<Arc<RingTable>>() + size_of::<RingTable>() + 2 * size_of::<usize>())
            + payload
    } else {
        0
    };
    Resources {
        entries: table.values.len(),
        chunks_per_lane: chunks,
        dcf_keys_per_party: keys,
        dcf_evaluations_per_party: intervals * lanes,
        private_key_bytes_per_party: keys * (dcf.encoded_key_bytes + 24),
        public_table_bytes_per_party: public,
        peer_frame_bytes_per_party: frame,
        output_bytes_per_party: output,
        table_reservation_bytes: size_of::<Table<'_>>() + table.values.len() * 8 + backing,
        issuance_reservation_bytes: 2 * party
            + 2 * largest
            + 2 * FIXED_BYTES
            + 2 * dcf.issuance_scratch_bytes,
        party_reservation_bytes: party,
    }
}

#[test]
fn constant_time_census_matches_explicit_rows_and_complete_resource_records() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    for entries in [1usize, 255, 256, 257, 511, 512, 513, 131071, 131072] {
        let values: Vec<_> = (0..entries).map(|i| (i & 1) as u64).collect();
        for bits in 1..=50 {
            if entries as u64 > 1u64 << bits {
                continue;
            }
            let table = Table::new_zero_extended(bits, &values, [7; 32], &budget).unwrap();
            let census = explicit_chunk_census(&table);
            let geometry = table.geometry().unwrap();
            assert_eq!(
                (
                    geometry.chunks,
                    geometry.intervals,
                    geometry.payload_bytes,
                    geometry.largest_payload_bytes,
                ),
                census,
                "entries={entries}, bits={bits}"
            );
            assert_eq!(table.lowered_payload_bytes(), census.2);
            let backing = size_of::<Lowered<'_>>()
                + 2 * size_of::<usize>()
                + census.0
                    * (size_of::<Arc<RingTable>>() + size_of::<RingTable>() + 2 * size_of::<usize>())
                + census.2;
            assert_eq!(table.lowered_reservation_bytes(), backing);
            for lanes in 1..=4 {
                assert_eq!(
                    table.resources(lanes).unwrap(),
                    explicit_resources(&table, lanes, census)
                );
            }
            // Real shared owners at the smallest valid ring and at bit 50;
            // reuse the raw fixture, never issue a large private-key fixture.
            if bits == 50 || (bits == 1 || entries as u64 > 1u64 << (bits - 1)) {
                let table = table.with_shared_chunks().unwrap();
                assert_eq!(explicit_chunk_census(&table), census);
                for lanes in 1..=4 {
                    assert_eq!(
                        table.resources(lanes).unwrap(),
                        explicit_resources(&table, lanes, census)
                    );
                }
            } else {
                drop(table);
            }
            assert_eq!(budget.used_bytes(), 0);
        }
    }
}

#[test]
fn public_census_validation_and_checked_overflow_never_allocate_or_mutate() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let table = Table::new_zero_extended(50, &[5; 513], [1; 32], &budget).unwrap();
    let before = (
        budget.test_snapshot(),
        budget.admitted_peak_bytes(),
        budget.test_accounting_updates(),
        table.digest(),
        table.binding(2, context()),
    );
    budget.fail_after.store(0, Ordering::SeqCst);
    for lanes in [0, MAX_LANES + 1, usize::MAX] {
        assert_eq!(table.resources(lanes), Err(Error::Domain));
        let mut overflow = context();
        overflow.first_tensor_index = u64::MAX;
        assert_eq!(table.issue(lanes, overflow).err(), Some(Error::Domain));
    }
    for lanes in 1..=4 {
        assert!(table.resources(lanes).is_ok());
    }
    assert!(table.lowered_reservation_bytes() > table.lowered_payload_bytes());
    assert_eq!(checked_add(usize::MAX, 1), Err(Error::Resource));
    assert_eq!(checked_mul(usize::MAX, 2), Err(Error::Resource));
    assert_eq!(checked_add(usize::MAX, 0), Ok(usize::MAX));
    assert_eq!(checked_mul(usize::MAX, 1), Ok(usize::MAX));
    assert_eq!(budget.fail_after.load(Ordering::SeqCst), 0);
    assert_eq!(
        (
            budget.test_snapshot(),
            budget.admitted_peak_bytes(),
            budget.test_accounting_updates(),
            table.digest(),
            table.binding(2, context()),
        ),
        before
    );
    assert!(table.lowered.is_none());
    budget.fail_after.store(-1, Ordering::SeqCst);
}

#[test]
fn census_issuance_exact_fit_and_one_short_preserve_real_owners() {
    for shared in [false, true] {
        let probe = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
        let table = Table::new_zero_extended(10, &[5; 257], [1; 32], &probe).unwrap();
        let table = if shared {
            table.with_shared_chunks().unwrap()
        } else {
            table
        };
        let r = explicit_resources(&table, 2, explicit_chunk_census(&table));
        assert_eq!(table.resources(2).unwrap(), r);
        drop(table);
        assert_eq!(probe.used_bytes(), 0);
        for short in [false, true] {
            let budget = Budget::new(
                r.table_reservation_bytes + r.issuance_reservation_bytes - usize::from(short),
            )
            .unwrap();
            let table = Table::new_zero_extended(10, &[5; 257], [1; 32], &budget).unwrap();
            let table = if shared {
                table.with_shared_chunks().unwrap()
            } else {
                table
            };
            if short {
                let before = (
                    budget.test_snapshot(),
                    budget.admitted_peak_bytes(),
                    budget.test_accounting_updates(),
                    table.digest(),
                );
                budget.fail_after.store(0, Ordering::SeqCst);
                assert_eq!(table.issue(2, context()).err(), Some(Error::Resource));
                assert_eq!(
                    (
                        budget.test_snapshot(),
                        budget.admitted_peak_bytes(),
                        budget.test_accounting_updates(),
                        table.digest(),
                    ),
                    before
                );
                assert_eq!(budget.fail_after.load(Ordering::SeqCst), 0);
                if let Some(lowered) = &table.lowered {
                    assert_eq!(Arc::strong_count(lowered), 1);
                    assert!(lowered.chunks.iter().all(|c| Arc::strong_count(c) == 1));
                }
                drop(table);
            } else {
                let [a, b] = table.issue(2, context()).unwrap();
                assert_eq!(
                    budget.admitted_peak_bytes(),
                    r.table_reservation_bytes + r.issuance_reservation_bytes
                );
                let backing = if shared {
                    table.lowered_reservation_bytes()
                } else {
                    0
                };
                drop(table);
                assert_eq!(budget.used_bytes(), backing + 2 * r.party_reservation_bytes);
                drop(a);
                assert_eq!(budget.used_bytes(), backing + r.party_reservation_bytes);
                drop(b);
            }
            assert_eq!(budget.used_bytes(), 0);
        }
    }
}

#[test]
fn lowered_chunks_are_built_once_and_last_party_retains_exact_backing_lease() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let table = Table::new_zero_extended(10, &[19; 513], [7; 32], &budget).unwrap();
    let raw_bytes = budget.used_bytes();
    let legacy = table.resources(2).unwrap();
    let backing = table.lowered_reservation_bytes();
    let payload = table.lowered_payload_bytes();
    let table = table.with_shared_chunks().unwrap();
    assert_eq!(budget.used_bytes(), raw_bytes + backing);
    let lowered = table.lowered.as_ref().unwrap();
    assert_eq!(lowered.chunks.len(), 3);
    assert_eq!(payload, (257 + 258 + 3) * 24 + 3 * 8);
    assert_eq!(
        backing,
        size_of::<Lowered<'_>>()
            + 2 * size_of::<usize>()
            + 3 * (size_of::<Arc<RingTable>>() + size_of::<RingTable>() + 2 * size_of::<usize>())
            + payload
    );
    let r = table.resources(2).unwrap();
    assert_eq!(r.table_reservation_bytes, raw_bytes + backing);
    assert_eq!(
        r.private_key_bytes_per_party,
        legacy.private_key_bytes_per_party
    );
    assert_eq!(
        r.dcf_evaluations_per_party,
        legacy.dcf_evaluations_per_party
    );
    assert_eq!(
        r.peer_frame_bytes_per_party,
        legacy.peer_frame_bytes_per_party
    );
    assert_eq!(
        r.public_table_bytes_per_party,
        6 * size_of::<Arc<RingTable>>() + size_of::<Arc<Lowered<'_>>>()
    );
    let [a, b] = table.issue(2, context()).unwrap();
    let Some(State::Ready(units)) = &a.state else {
        panic!("ready")
    };
    for (i, unit) in units.iter().enumerate() {
        let Lookup::Shared(p) = &unit.lookup else {
            panic!("shared")
        };
        // Party tables are private to ring.rs; sharing is also witnessed by the
        // canonical chunk's two additional references per lane (one per role).
        let _ = p;
        assert_eq!(Arc::strong_count(&lowered.chunks[i / 2]), 1 + 2 * 2);
    }
    assert_eq!(
        budget.used_bytes(),
        r.table_reservation_bytes + 2 * r.party_reservation_bytes
    );
    drop(table);
    assert_eq!(budget.used_bytes(), backing + 2 * r.party_reservation_bytes);
    drop(a);
    assert_eq!(budget.used_bytes(), backing + r.party_reservation_bytes);
    drop(b);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn lowered_construction_failure_sweep_and_exact_admission_retire_every_arc() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut completed = false;
    for fail in 0..=11 {
        budget.fail_after.store(-1, Ordering::SeqCst);
        let table = Table::new_zero_extended(10, &[5; 513], [1; 32], &budget).unwrap();
        budget.fail_after.store(fail, Ordering::SeqCst);
        let result = table.with_shared_chunks();
        if result.is_ok() {
            assert_eq!(fail, 11);
            completed = true;
        } else {
            assert!(matches!(result, Err(Error::Allocation)));
        }
        drop(result);
        assert_eq!(budget.used_bytes(), 0, "lowering failure {fail}");
    }
    assert!(completed);
    budget.fail_after.store(-1, Ordering::SeqCst);
    let table = Table::new_zero_extended(10, &[5; 513], [1; 32], &budget).unwrap();
    let exact = budget.used_bytes() + table.lowered_reservation_bytes();
    drop(table);
    let small = Budget::new(exact - 1).unwrap();
    let table = Table::new_zero_extended(10, &[5; 513], [1; 32], &small).unwrap();
    assert!(matches!(table.with_shared_chunks(), Err(Error::Resource)));
    assert_eq!(small.used_bytes(), 0);
}

#[test]
fn lowered_import_uses_receiving_process_owner_and_unchanged_private_codec() {
    for (dealer_shared, receiver_shared) in [(false, true), (true, false), (true, true)] {
        let dealer = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
        let worker = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
        let values: Vec<_> = (0..513).map(|x| (x * 17 + 5) & 1023).collect();
        let table = Table::new_zero_extended(10, &values, [7; 32], &dealer).unwrap();
        let table = if dealer_shared {
            table.with_shared_chunks().unwrap()
        } else {
            table
        };
        let public = Table::new_zero_extended(10, &values, [7; 32], &worker).unwrap();
        let public = if receiver_shared {
            public.with_shared_chunks().unwrap()
        } else {
            public
        };
        assert_eq!(table.digest(), public.digest());
        if let (Some(a), Some(b)) = (&table.lowered, &public.lowered) {
            assert!(!Arc::ptr_eq(&a.chunks[0], &b.chunks[0]));
        }
        let r = public.resources(4).unwrap();
        let [a, b] = table.issue(4, context()).unwrap();
        let pa = a.export_material().unwrap();
        let pb = b.export_material().unwrap();
        assert_eq!(
            pa.as_bytes().len(),
            super::super::wire::ENVELOPE_BYTES + r.private_key_bytes_per_party
        );
        let nonce = pa.nonce();
        drop(table);
        // Packets contain no public pointers; only private packet bytes remain.
        assert_eq!(
            dealer.used_bytes(),
            pa.as_bytes().len() + pb.as_bytes().len()
        );
        let mut guard = super::super::wire::ImportGuard::new(4, &worker).unwrap();
        let a = Party::import_material(&public, 4, context(), 0, nonce, pa, &mut guard).unwrap();
        let b = Party::import_material(&public, 4, context(), 1, nonce, pb, &mut guard).unwrap();
        assert_eq!(dealer.used_bytes(), 0);
        drop(public);
        let mut parties = [a, b];
        let inputs = [0, 255, 512, 1023];
        let outputs = run(&mut parties, &inputs);
        for (i, x) in inputs.into_iter().enumerate() {
            assert_eq!(
                (outputs[0].shares()[i] + outputs[1].shares()[i]) & 1023,
                values.get(x as usize).copied().unwrap_or(0)
            );
        }
        drop((parties, outputs, guard));
        assert_eq!(worker.used_bytes(), 0);
    }
}

#[test]
fn lowered_import_faults_burn_claims_and_retire_partial_private_owners() {
    use super::super::wire::{ImportGuard, MaterialPacket};
    let dealer = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let worker = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let table = Table::new_zero_extended(10, &[5; 513], [7; 32], &dealer).unwrap();
    let public = Table::new_zero_extended(10, &[5; 513], [7; 32], &worker)
        .unwrap()
        .with_shared_chunks()
        .unwrap();
    let wrong = Table::new_zero_extended(10, &[6; 513], [7; 32], &worker)
        .unwrap()
        .with_shared_chunks()
        .unwrap();
    let mut completed = false;
    for fail in 0..64 {
        worker.fail_after.store(-1, Ordering::SeqCst);
        let mut guard = ImportGuard::new(2, &worker).unwrap();
        let [a, b] = table.issue(2, context()).unwrap();
        drop(b);
        let packet = a.export_material().unwrap();
        let replay = MaterialPacket::from_bytes(packet.as_bytes(), &dealer).unwrap();
        let nonce = packet.nonce();
        let before = worker.used_bytes();
        worker.fail_after.store(fail, Ordering::SeqCst);
        let result = Party::import_material(&public, 2, context(), 0, nonce, packet, &mut guard);
        let ok = result.is_ok();
        if !ok {
            assert!(matches!(result, Err(Error::Allocation)));
        }
        drop(result);
        assert_eq!(worker.used_bytes(), before, "import failure {fail}");
        // Claim burns before any import allocation, even when that allocation
        // fails. Duplicate claims return Consumed before parsing or allocating.
        worker.fail_after.store(0, Ordering::SeqCst);
        assert_eq!(
            Party::import_material(&public, 2, context(), 0, nonce, replay, &mut guard).err(),
            Some(Error::Consumed),
            "replay after import failure index {fail}"
        );
        assert_eq!(worker.used_bytes(), before, "replay failure {fail}");
        worker.fail_after.store(-1, Ordering::SeqCst);
        if ok {
            completed = true;
            break;
        }
    }
    assert!(completed);
    let mut guard = ImportGuard::new(2, &worker).unwrap();
    let [a, b] = table.issue(1, context()).unwrap();
    drop(b);
    let packet = a.export_material().unwrap();
    let replay = MaterialPacket::from_bytes(packet.as_bytes(), &dealer).unwrap();
    let nonce = packet.nonce();
    let before = worker.used_bytes();
    assert!(matches!(
        Party::import_material(&wrong, 1, context(), 0, nonce, packet, &mut guard),
        Err(Error::Context)
    ));
    assert_eq!(
        Party::import_material(&public, 1, context(), 0, nonce, replay, &mut guard).err(),
        Some(Error::Consumed)
    );
    assert_eq!(worker.used_bytes(), before);
    drop((table, public, wrong, guard));
    assert_eq!(dealer.used_bytes(), 0);
    assert_eq!(worker.used_bytes(), 0);
}

#[test]
fn lowered_shared_issuance_allocation_faults_retire_all_private_owners() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let table = Table::new_zero_extended(50, &[19; 513], [7; 32], &budget)
        .unwrap()
        .with_shared_chunks()
        .unwrap();
    let before = budget.used_bytes();
    let mut completed = false;
    for fail in 0..64 {
        budget.fail_after.store(fail, Ordering::SeqCst);
        let result = table.issue(2, context());
        let ok = result.is_ok();
        if !ok {
            assert!(
                matches!(result, Err(Error::Allocation)),
                "issuance failure {fail}"
            );
        }
        drop(result);
        assert_eq!(budget.used_bytes(), before, "issuance failure {fail}");
        let lowered = table.lowered.as_ref().unwrap();
        assert_eq!(Arc::strong_count(lowered), 1);
        assert!(lowered.chunks.iter().all(|c| Arc::strong_count(c) == 1));
        if ok {
            completed = true;
            break;
        }
    }
    assert!(completed);
    budget.fail_after.store(-1, Ordering::SeqCst);
    drop(table);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn lowered_packet_import_backing_peak_has_exact_budget_boundary() {
    use super::super::wire::{ImportGuard, ENVELOPE_BYTES};
    let probe = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let table = Table::new_zero_extended(50, &[19; 513], [7; 32], &probe)
        .unwrap()
        .with_shared_chunks()
        .unwrap();
    let r = table.resources(2).unwrap();
    let backing = table.lowered_reservation_bytes();
    let packet_bytes = ENVELOPE_BYTES + r.private_key_bytes_per_party;
    let guard_bytes = 2 * 33;
    assert!(r.party_reservation_bytes >= packet_bytes);
    // Second import overlaps first imported party, its peer packet, canonical
    // source/backing and the full remaining import reservation. First import's
    // two-packet overlap is smaller because a packet is smaller than a party.
    let peak =
        r.table_reservation_bytes + guard_bytes + packet_bytes + r.issuance_reservation_bytes;
    drop(table);
    assert_eq!(probe.used_bytes(), 0);
    for short in [false, true] {
        let budget = Budget::new(peak - usize::from(short)).unwrap();
        let table = Table::new_zero_extended(50, &[19; 513], [7; 32], &budget)
            .unwrap()
            .with_shared_chunks()
            .unwrap();
        let mut guard = ImportGuard::new(2, &budget).unwrap();
        let [a, b] = table.issue(2, context()).unwrap();
        let pa = a.export_material().unwrap();
        let pb = b.export_material().unwrap();
        let nonce = pa.nonce();
        assert_eq!(
            budget.used_bytes(),
            r.table_reservation_bytes + guard_bytes + 2 * packet_bytes
        );
        let a = Party::import_material(&table, 2, context(), 0, nonce, pa, &mut guard).unwrap();
        assert_eq!(
            budget.used_bytes(),
            r.table_reservation_bytes + guard_bytes + r.party_reservation_bytes + packet_bytes
        );
        let b = Party::import_material(&table, 2, context(), 1, nonce, pb, &mut guard);
        assert_eq!(guard.claim(1, nonce), Err(Error::Consumed));
        if short {
            assert!(matches!(b, Err(Error::Resource)));
            assert_eq!(
                budget.used_bytes(),
                r.table_reservation_bytes + guard_bytes + r.party_reservation_bytes
            );
            drop(table);
            assert_eq!(
                budget.used_bytes(),
                backing + guard_bytes + r.party_reservation_bytes
            );
            drop(a);
            assert_eq!(budget.used_bytes(), guard_bytes);
        } else {
            let b = b.unwrap();
            assert_eq!(
                budget.used_bytes(),
                r.table_reservation_bytes + guard_bytes + 2 * r.party_reservation_bytes
            );
            drop(table);
            assert_eq!(
                budget.used_bytes(),
                backing + guard_bytes + 2 * r.party_reservation_bytes
            );
            let mut parties = [a, b];
            let out = run(&mut parties, &[0, 512]);
            for lane in 0..2 {
                assert_eq!(
                    (out[0].shares()[lane] + out[1].shares()[lane]) & ((1u64 << 50) - 1),
                    19
                );
            }
            drop((parties, out));
            assert_eq!(budget.used_bytes(), guard_bytes);
        }
        drop(guard);
        assert_eq!(budget.used_bytes(), 0);
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
