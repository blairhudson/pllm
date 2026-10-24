//! Synthetic immutable metadata only. No codes, masks, protocol/model execution.
use super::*;

#[test]
fn cancellation_keeps_backing_until_erase_then_release_and_cannot_retry() {
    let originals = [matrix(1, 256, 128, Use::EightPositions)];
    let inventory = Inventory::new([1; 32], &originals, 1).unwrap();
    let reserved = MetadataReservation::admit(inventory, attachments(), limits()).unwrap();
    let bytes = inventory.group(0).unwrap().public_bytes;
    let cursor = reserved
        .cursor()
        .advance(Event::Open(0))
        .unwrap()
        .advance(Event::Read(bytes))
        .unwrap()
        .advance(Event::Validate(bytes))
        .unwrap()
        .advance(Event::Original(0))
        .unwrap()
        .advance(Event::Cancel)
        .unwrap();
    assert_eq!(cursor.phase(), Phase::Erase);
    assert_eq!(cursor.retained_bytes(), (2 * (128 + 256) * 4, 0));
    assert_eq!(cursor.horizon_attempts(), 1);
    let cursor = cursor
        .advance(Event::Erase)
        .unwrap()
        .advance(Event::Release)
        .unwrap();
    assert_eq!(cursor.retained_bytes(), (0, 0));
    assert!(matches!(
        cursor.advance(Event::OriginalBarrier),
        Err(Refused::Order)
    ));
}

#[test]
fn both_barriers_cover_the_entire_original_inventory_not_first_complete_group() {
    let originals = [
        matrix(1, 256, 128, Use::EightPositions),
        matrix(2, 128, 256, Use::FinalPosition),
    ];
    let inventory = Inventory::new([1; 32], &originals, 2).unwrap();
    let reserved = MetadataReservation::admit(inventory, attachments(), limits()).unwrap();
    for stop_before_global_original in [true, false] {
        let mut cursor = reserved.cursor();
        for ordinal in 0..if stop_before_global_original { 1 } else { 2 } {
            let g = inventory.group(ordinal).unwrap();
            cursor = cursor
                .advance(Event::Open(ordinal))
                .unwrap()
                .advance(Event::Read(g.public_bytes))
                .unwrap()
                .advance(Event::Validate(g.public_bytes))
                .unwrap();
            for b in g.first_batch..g.first_batch + g.batch_count {
                cursor = cursor.advance(Event::Original(b)).unwrap();
            }
            cursor = cursor.advance(Event::Close).unwrap();
        }
        if stop_before_global_original {
            assert!(matches!(
                cursor.advance(Event::OriginalBarrier),
                Err(Refused::Order)
            ));
        } else {
            cursor = cursor.advance(Event::OriginalBarrier).unwrap();
            let g = inventory.group(0).unwrap();
            cursor = cursor
                .advance(Event::Open(0))
                .unwrap()
                .advance(Event::Read(g.q_bytes))
                .unwrap()
                .advance(Event::Validate(g.q_bytes))
                .unwrap();
            for b in g.first_batch..g.first_batch + g.batch_count {
                cursor = cursor
                    .advance(Event::Primary(b))
                    .unwrap()
                    .advance(Event::Auxiliary(b))
                    .unwrap();
            }
            cursor = cursor.advance(Event::Close).unwrap();
            assert!(matches!(
                cursor.advance(Event::AuxiliaryBarrier),
                Err(Refused::Order)
            ));
        }
    }
}

fn matrix(id: u64, rows: u64, columns: u64, usage: Use) -> OriginalMatrix {
    let mut operation = [1; 32];
    operation[..8].copy_from_slice(&id.to_le_bytes());
    OriginalMatrix {
        operation,
        source: [2; 32],
        rows,
        columns,
        usage,
    }
}
fn attachments() -> Attachments {
    // Explicit synthetic metadata references/quotes; never source certificates.
    Attachments {
        security: Some([1; 32]),
        backend: Some([2; 32]),
        source_binding: Some([3; 32]),
        lifetime_binding: Some([4; 32]),
        client_other_bytes: Some(1024),
        active_other_bytes: Some(4096),
        source_backing_bytes: Some(1024),
        store_other_bytes: Some(512),
        additional_work: [Some(1); 7],
        additional_io: Some(1024),
    }
}
fn limits() -> Limits {
    Limits {
        client_bytes: CLIENT_CAP,
        store_bytes: STORE_CAP,
        active_bytes: ALLOCATION_CEILING as u64,
        work: 20_000_000_000_000,
        io: 20_000_000_000_000,
    }
}
fn full_inventory() -> Vec<OriginalMatrix> {
    let mut out = Vec::new();
    // Qualification data only, NEVER a model-family production scheduler.
    for _ in 0..36 {
        for (rows, columns) in [
            (4096, 2560),
            (1024, 2560),
            (1024, 2560),
            (2560, 4096),
            (9728, 2560),
            (9728, 2560),
            (2560, 9728),
        ] {
            out.push(matrix(out.len() as u64, rows, columns, Use::EightPositions));
        }
    }
    out.push(matrix(252, 151936, 2560, Use::FinalPosition));
    out
}

#[test]
fn symbolic_full_inventory_no_tile_allocation() {
    let originals = full_inventory();
    let inventory = Inventory::new([1; 32], &originals, 253).unwrap();
    let s = inventory.summary();
    // Independent literal totals from the approved design arithmetic.
    assert_eq!(
        (s.tiles, s.groups, s.batches, s.claims),
        (122760, 116820, 449460, 898920)
    );
    assert_eq!(s.frames, 2696760);
    assert_eq!(s.unpartitioned_claims, 2017);
    assert_eq!(
        (
            s.noise_samples,
            s.primary_challenges,
            s.secondary_challenges
        ),
        (898920, 898920, 898920)
    );
    assert_eq!(
        (s.original_scalar_checks, s.secondary_scalar_checks),
        (1797840, 1797840)
    );
    assert_eq!((s.code_instances, s.code_description_bytes), (5, 114688));
    assert_eq!(s.original_bytes, 1380730880);
    assert_eq!(s.spool_bytes, 1143426240);
    assert_eq!(s.wire_bytes, 2524157120);
    assert_eq!(s.public_pass_bytes, 2449788108800);
    assert_eq!(s.derived_work, 9799152435200);
    assert_eq!(s.evaluator_q_bytes, 244978810880);
    assert_eq!(s.matrix_terms, 29455482880);
    assert_eq!(s.accepted_sparse_entries, 203155920);
    assert!(inventory.group(s.groups).is_err());
    let tail = inventory.group(s.groups - 1).unwrap();
    assert_eq!(
        (tail.original, tail.rows, tail.row_start, tail.column_start),
        (252, 128, 151808, 2304)
    );
    assert_eq!(tail.first_batch + tail.batch_count, s.batches);
    let final_slot = inventory
        .claim_slice(s.groups - 1, s.batches - 1, 1)
        .unwrap();
    assert_eq!(
        (final_slot.row, final_slot.column, final_slot.position),
        (151808, 2432, 7)
    );
    assert_eq!(final_slot.source, originals[252].source);
    assert_eq!(final_slot.operation, originals[252].operation);
}

#[test]
fn independent_small_nested_loop_inventory_and_fifo() {
    let originals = [
        matrix(1, 384, 256, Use::EightPositions),
        matrix(2, 384, 256, Use::FinalPosition),
    ];
    let inventory = Inventory::new([1; 32], &originals, 2).unwrap();
    let mut group = 0;
    let mut batches = 0;
    let mut retained = 0;
    let mut products = 0;
    for (i, m) in originals.iter().enumerate() {
        for start in (0..m.rows).step_by(256) {
            let rows = 256.min(m.rows - start);
            let stride = if m.usage == Use::EightPositions {
                128
            } else {
                256
            };
            for col in (0..m.columns).step_by(stride) {
                let g = inventory.group(group).unwrap();
                assert_eq!(
                    (
                        g.original,
                        g.row_start,
                        g.column_start,
                        g.rows,
                        g.first_batch
                    ),
                    (i, start, col, rows, batches)
                );
                for _ in 0..if m.usage == Use::EightPositions { 4 } else { 1 } {
                    for slot in 0..2 {
                        let slice = inventory.claim_slice(group, batches, slot).unwrap();
                        assert_eq!(slice.columns, 128);
                        assert_eq!(slice.row, start);
                        assert_eq!(
                            slice.position,
                            if m.usage == Use::EightPositions {
                                2 * (batches - g.first_batch) + slot
                            } else {
                                7
                            }
                        );
                        assert_eq!(
                            slice.column,
                            col + if m.usage == Use::FinalPosition {
                                slot * 128
                            } else {
                                0
                            }
                        );
                    }
                    batches += 1;
                    // Two separate explicit slots, not a call to summary formulas.
                    for _ in 0..2 {
                        retained += 128 * 4 + rows * 4;
                        products += rows * 128;
                    }
                }
                group += 1;
            }
        }
    }
    let s = inventory.summary();
    assert_eq!(
        (s.groups, s.batches, s.original_bytes, s.matrix_terms),
        (group, batches, retained, products)
    );
}

#[test]
fn unknown_attachments_and_unbound_execution_refuse() {
    let originals = [matrix(1, 256, 128, Use::EightPositions)];
    let inventory = Inventory::new([1; 32], &originals, 1).unwrap();
    let mut a = attachments();
    a.security = None;
    assert!(matches!(
        MetadataReservation::admit(inventory, a, limits()),
        Err(Refused::UnknownAttachment)
    ));
    a = attachments();
    a.backend = Some([0; 32]);
    assert!(matches!(
        MetadataReservation::admit(inventory, a, limits()),
        Err(Refused::UnknownAttachment)
    ));
    a = attachments();
    a.additional_work[3] = None;
    assert!(matches!(
        MetadataReservation::admit(inventory, a, limits()),
        Err(Refused::UnknownAttachment)
    ));
    a = attachments();
    a.source_backing_bytes = None;
    assert!(matches!(
        MetadataReservation::admit(inventory, a, limits()),
        Err(Refused::UnknownAttachment)
    ));
    let reserved = MetadataReservation::admit(inventory, attachments(), limits()).unwrap();
    assert_eq!(
        reserved.execution_admission(),
        Err(Refused::OwnerBridgeMissing)
    );
    assert_eq!(reserved.supplied_bindings().0, [1; 32]);
}

#[test]
fn every_named_unknown_refuses_and_large_quotes_overflow() {
    let originals = [matrix(1, 256, 128, Use::EightPositions)];
    let inventory = Inventory::new([1; 32], &originals, 1).unwrap();
    for i in 0..16 {
        let mut a = attachments();
        match i {
            0 => a.security = None,
            1 => a.backend = None,
            2 => a.source_binding = None,
            3 => a.lifetime_binding = None,
            4 => a.client_other_bytes = None,
            5 => a.active_other_bytes = None,
            6 => a.source_backing_bytes = None,
            7 => a.store_other_bytes = None,
            8..=14 => a.additional_work[i - 8] = None,
            _ => a.additional_io = None,
        }
        assert!(matches!(
            MetadataReservation::admit(inventory, a, limits()),
            Err(Refused::UnknownAttachment)
        ));
    }
    for i in 0..4 {
        let mut a = attachments();
        match i {
            0 => a.client_other_bytes = Some(u64::MAX),
            1 => a.source_backing_bytes = Some(u64::MAX),
            2 => a.additional_work[0] = Some(u64::MAX),
            _ => a.additional_io = Some(u64::MAX),
        }
        assert!(matches!(
            MetadataReservation::admit(inventory, a, limits()),
            Err(Refused::Overflow)
        ));
    }
}

#[test]
fn proposal_envelope_is_metadata_not_execution_authority() {
    let originals = full_inventory();
    let inventory = Inventory::new([1; 32], &originals, 253).unwrap();
    let mut a = attachments();
    a.active_other_bytes = Some((16 << 20) - inventory.summary().active_payload_floor);
    a.client_other_bytes = Some((256 << 20) + 229376);
    a.source_backing_bytes = Some(5190052872);
    a.store_other_bytes = Some(33 << 20);
    let metadata = MetadataReservation::admit(inventory, a, limits()).unwrap();
    assert_eq!(metadata.reserved().client_bytes, 1666172928);
    assert_eq!(metadata.reserved().store_bytes, 6368082120);
    assert_eq!(
        metadata.execution_admission(),
        Err(Refused::OwnerBridgeMissing)
    );
}

#[test]
fn exact_fit_one_short_and_no_cap_increase() {
    let originals = [matrix(1, 256, 128, Use::EightPositions)];
    let inventory = Inventory::new([1; 32], &originals, 1).unwrap();
    let p = MetadataReservation::admit(inventory, attachments(), limits()).unwrap();
    let exact = Limits {
        client_bytes: p.client_bytes,
        store_bytes: p.store_bytes,
        active_bytes: p.active_bytes,
        work: p.work,
        io: p.io,
    };
    assert!(MetadataReservation::admit(inventory, attachments(), exact).is_ok());
    for i in 0..5 {
        let mut short = exact;
        match i {
            0 => short.client_bytes -= 1,
            1 => short.store_bytes -= 1,
            2 => short.active_bytes -= 1,
            3 => short.work -= 1,
            _ => short.io -= 1,
        }
        assert!(MetadataReservation::admit(inventory, attachments(), short).is_err());
    }
    for i in 0..3 {
        let mut excessive = limits();
        match i {
            0 => excessive.client_bytes += 1,
            1 => excessive.store_bytes += 1,
            _ => excessive.active_bytes += 1,
        }
        assert!(matches!(
            MetadataReservation::admit(inventory, attachments(), excessive),
            Err(Refused::Capacity)
        ));
    }
}

#[test]
fn cumulative_full_work_is_not_one_window_work() {
    let originals = full_inventory();
    let inventory = Inventory::new([1; 32], &originals, 253).unwrap();
    let mut cap = limits();
    cap.work = inventory.group(0).unwrap().public_bytes * 4;
    assert!(matches!(
        MetadataReservation::admit(inventory, attachments(), cap),
        Err(Refused::Work)
    ));
}

#[test]
fn overflow_before_allocation_and_missing_originals() {
    let huge = [matrix(
        1,
        u64::MAX - 127,
        u64::MAX - 127,
        Use::EightPositions,
    )];
    assert!(matches!(
        Inventory::new([1; 32], &huge, 1),
        Err(Refused::Overflow)
    ));
    let originals = [matrix(1, 256, 128, Use::EightPositions)];
    assert!(matches!(
        Inventory::new([1; 32], &originals, 2),
        Err(Refused::Binding)
    ));
    assert!(matches!(
        Inventory::new([0; 32], &originals, 1),
        Err(Refused::Binding)
    ));
    assert!(matches!(
        Inventory::new([1; 32], &[], 0),
        Err(Refused::Records)
    ));
}

#[test]
fn geometry_and_duplicate_bindings_do_not_get_repaired() {
    for (rows, columns, usage) in [
        (255, 128, Use::EightPositions),
        (128, 127, Use::EightPositions),
        (128, 128, Use::FinalPosition),
        (0, 256, Use::FinalPosition),
    ] {
        let bad = [matrix(1, rows, columns, usage)];
        assert!(matches!(
            Inventory::new([1; 32], &bad, 1),
            Err(Refused::Geometry)
        ));
    }
    let duplicate = [matrix(1, 256, 128, Use::EightPositions); 2];
    assert!(matches!(
        Inventory::new([1; 32], &duplicate, 2),
        Err(Refused::Binding)
    ));
    // This planner deliberately does NOT widen the existing capsule constructor.
    assert!(super::super::AllocationPlan::for_shape(256, 128, 2, ALLOCATION_CEILING).is_err());
}

#[test]
fn exact_record_boundary_uses_no_claim_records() {
    let originals: Vec<_> = (0..MAX_RECORDS)
        .map(|i| matrix(i as u64, 128, 256, Use::FinalPosition))
        .collect();
    let inventory = Inventory::new([1; 32], &originals, MAX_RECORDS as u64).unwrap();
    assert_eq!(inventory.summary().groups, MAX_RECORDS as u64);
}

#[test]
fn complete_consuming_lifetime_and_independent_public_visit_counts() {
    let originals = [
        matrix(1, 256, 128, Use::EightPositions),
        matrix(2, 128, 256, Use::FinalPosition),
    ];
    let inventory = Inventory::new([1; 32], &originals, 2).unwrap();
    let reserved = MetadataReservation::admit(inventory, attachments(), limits()).unwrap();
    let mut cursor = reserved.cursor();
    let s = inventory.summary();
    for phase in [Phase::Original, Phase::Primary, Phase::Check] {
        for ordinal in 0..s.groups {
            let g = inventory.group(ordinal).unwrap();
            let bytes = if phase == Phase::Primary {
                g.q_bytes
            } else {
                g.public_bytes
            };
            cursor = cursor.advance(Event::Open(ordinal)).unwrap();
            cursor = cursor.advance(Event::Read(bytes)).unwrap();
            cursor = cursor.advance(Event::Validate(bytes)).unwrap();
            for batch in g.first_batch..g.first_batch + g.batch_count {
                match phase {
                    Phase::Original => cursor = cursor.advance(Event::Original(batch)).unwrap(),
                    Phase::Primary => {
                        cursor = cursor.advance(Event::Primary(batch)).unwrap();
                        cursor = cursor.advance(Event::Auxiliary(batch)).unwrap();
                    }
                    Phase::Check => cursor = cursor.advance(Event::Check(batch)).unwrap(),
                    _ => unreachable!(),
                }
            }
            cursor = cursor.advance(Event::Close).unwrap();
            assert_eq!(cursor.horizon_attempts(), 1);
        }
        cursor = cursor
            .advance(match phase {
                Phase::Original => Event::OriginalBarrier,
                Phase::Primary => Event::AuxiliaryBarrier,
                _ => Event::Finish,
            })
            .unwrap();
    }
    assert_eq!(cursor.retained_bytes(), (s.original_bytes, s.spool_bytes));
    let expected = 4 * s.public_pass_bytes + 2 * s.evaluator_q_bytes;
    assert_eq!(cursor.cumulative_visits(), (expected, expected));
    cursor = cursor.advance(Event::Erase).unwrap();
    assert_eq!(cursor.retained_bytes(), (s.original_bytes, s.spool_bytes));
    cursor = cursor.advance(Event::Release).unwrap();
    assert_eq!(cursor.retained_bytes(), (0, 0));
    assert_eq!(cursor.phase(), Phase::Spent);
    assert!(cursor.advance(Event::Open(0)).is_err());
}

#[test]
fn premature_barriers_partial_reads_and_refund_are_terminal_errors() {
    let originals = [matrix(1, 256, 128, Use::EightPositions)];
    let inventory = Inventory::new([1; 32], &originals, 1).unwrap();
    let reserved = MetadataReservation::admit(inventory, attachments(), limits()).unwrap();
    for event in [
        Event::OriginalBarrier,
        Event::AuxiliaryBarrier,
        Event::Primary(0),
        Event::Check(0),
        Event::Finish,
        Event::Erase,
        Event::Release,
        Event::Open(1),
    ] {
        assert!(matches!(
            reserved.cursor().advance(event),
            Err(Refused::Order)
        ));
    }
    let cursor = reserved.cursor().advance(Event::Open(0)).unwrap();
    assert!(matches!(
        cursor.advance(Event::Read(1)),
        Err(Refused::Order)
    ));
    let cursor = reserved.cursor().advance(Event::Open(0)).unwrap();
    assert!(matches!(
        cursor.advance(Event::Original(0)),
        Err(Refused::Order)
    ));
}

#[test]
fn global_original_barrier_not_two_claims() {
    let originals = [matrix(1, 256, 128, Use::EightPositions)];
    let inventory = Inventory::new([1; 32], &originals, 1).unwrap();
    let reserved = MetadataReservation::admit(inventory, attachments(), limits()).unwrap();
    let bytes = inventory.group(0).unwrap().public_bytes;
    let mut cursor = reserved
        .cursor()
        .advance(Event::Open(0))
        .unwrap()
        .advance(Event::Read(bytes))
        .unwrap()
        .advance(Event::Validate(bytes))
        .unwrap();
    for batch in 0..2 {
        cursor = cursor.advance(Event::Original(batch)).unwrap();
    }
    assert!(matches!(
        cursor.advance(Event::OriginalBarrier),
        Err(Refused::Order)
    ));
}
