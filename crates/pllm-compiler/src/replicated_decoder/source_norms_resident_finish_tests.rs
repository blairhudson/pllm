use super::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

#[derive(Debug, PartialEq, Eq)]
struct PendingSnapshot {
    pending: *const Pending,
    geometry: (usize, usize, usize),
    offset: usize,
    blocks: Vec<Block>,
    pointer: *const Block,
    capacity: usize,
    // MatrixScan has no public backing borrow. This checks only its resident
    // shell, not the identity/capacity of its private column allocation.
    matrix_shell: Option<*const MatrixScan>,
}

fn snapshot(owner: &ResidentBlockFinish) -> PendingSnapshot {
    let pending = owner.scan.0.as_ref().unwrap();
    PendingSnapshot {
        pending,
        geometry: (pending.rows, pending.columns, pending.block_rows),
        offset: pending.offset,
        blocks: pending.blocks.clone(),
        pointer: pending.blocks.as_ptr(),
        capacity: pending.blocks.capacity(),
        matrix_shell: pending.scan.as_ref().map(|scan| scan as *const MatrixScan),
    }
}

fn completed_scan() -> BlockScan {
    let mut scan = BlockScan::new(3, 2, 2).unwrap();
    scan.append(0, &[1, -2, 3, -4, 5, -6]).unwrap();
    scan
}

#[test]
fn resident_conversion_moves_original_blocks_without_changing_scan_fields() {
    let mut scan = BlockScan::new(3, 2, 2).unwrap();
    scan.append(0, &[1, -2, 3, -4]).unwrap();
    let pending = scan.0.as_ref().unwrap();
    let pointer = pending.blocks.as_ptr();
    let capacity = pending.blocks.capacity();
    let blocks = pending.blocks.clone();
    let owner = scan.into_resident_finish();
    assert!(matches!(owner.state, FinishState::Ready));
    let after = snapshot(&owner);
    assert_eq!(after.geometry, (3, 2, 2));
    assert_eq!(after.offset, 4);
    assert_eq!(after.pointer, pointer);
    assert_eq!(after.capacity, capacity);
    assert_eq!(after.blocks, blocks);
    assert!(after.matrix_shell.is_some());
}

#[test]
fn incomplete_offset_retains_original_pending_blocks_and_live_matrix_shell() {
    let mut legacy = BlockScan::new(3, 2, 2).unwrap();
    legacy.append(0, &[1, -2, 3, -4, 5]).unwrap();
    assert_eq!(legacy.finish(), Err(Error::Shape));
    let mut scan = BlockScan::new(3, 2, 2).unwrap();
    scan.append(0, &[1, -2, 3, -4, 5]).unwrap();
    let mut owner = scan.into_resident_finish();
    let before = snapshot(&owner);
    let mut out = None;
    assert_eq!(owner.finish_into(&mut out), Err(Error::Shape));
    assert!(out.is_none());
    assert!(matches!(owner.state, FinishState::CleanupOnly));
    assert_eq!(snapshot(&owner), before);
    assert_eq!(owner.finish_into(&mut out), Err(Error::Binding));
    assert_eq!(snapshot(&owner), before);
}

#[test]
fn complete_offset_with_live_matrix_is_shape_not_implicitly_finished() {
    let mut legacy = BlockScan::new(3, 2, 2).unwrap();
    legacy.append(0, &[1, -2, 3, -4, 5]).unwrap();
    legacy.0.as_mut().unwrap().offset = 6;
    assert_eq!(legacy.finish(), Err(Error::Shape));
    let mut scan = BlockScan::new(3, 2, 2).unwrap();
    scan.append(0, &[1, -2, 3, -4, 5]).unwrap();
    scan.0.as_mut().unwrap().offset = 6;
    let mut owner = scan.into_resident_finish();
    let before = snapshot(&owner);
    let mut out = None;
    assert_eq!(owner.finish_into(&mut out), Err(Error::Shape));
    assert!(out.is_none());
    assert_eq!(snapshot(&owner), before);
    assert!(before.matrix_shell.is_some());
    assert!(matches!(owner.state, FinishState::CleanupOnly));
}

#[test]
fn incomplete_offset_without_live_matrix_also_retains_original_records() {
    let mut legacy = completed_scan();
    legacy.0.as_mut().unwrap().offset -= 1;
    assert_eq!(legacy.finish(), Err(Error::Shape));
    let mut scan = completed_scan();
    scan.0.as_mut().unwrap().offset -= 1;
    let mut owner = scan.into_resident_finish();
    let before = snapshot(&owner);
    let mut out = None;
    assert_eq!(owner.finish_into(&mut out), Err(Error::Shape));
    assert!(out.is_none());
    assert_eq!(snapshot(&owner), before);
    assert!(before.matrix_shell.is_none());
}

#[test]
fn invalid_record_preflight_keeps_original_pending_and_matches_legacy_errors() {
    // Safe malformed records: zero width, missing block, excess capacity, and
    // excess block count. No fabricated Vec or inaccessible backing is used.
    for case in 0..4 {
        let mutate = |scan: &mut BlockScan| {
            let pending = scan.0.as_mut().unwrap();
            match case {
                0 => pending.block_rows = 0,
                1 => {
                    pending.blocks.pop();
                }
                2 => pending
                    .blocks
                    .reserve_exact(MAX_METADATA_BYTES / std::mem::size_of::<Block>() + 1),
                _ => {
                    pending.rows = MAX_BLOCKS + 1;
                    pending.block_rows = 1;
                    pending.offset = pending.rows * pending.columns;
                    pending.blocks = vec![pending.blocks[0]; MAX_BLOCKS + 1];
                }
            }
        };
        let mut legacy = completed_scan();
        mutate(&mut legacy);
        assert_eq!(legacy.finish(), Err(Error::Shape));
        let mut scan = completed_scan();
        mutate(&mut scan);
        let mut owner = scan.into_resident_finish();
        let before = snapshot(&owner);
        let mut out = None;
        assert_eq!(owner.finish_into(&mut out), Err(Error::Shape));
        assert!(out.is_none());
        assert_eq!(snapshot(&owner), before);
        assert!(matches!(owner.state, FinishState::CleanupOnly));
        assert_eq!(owner.finish_into(&mut out), Err(Error::Binding));
        assert_eq!(snapshot(&owner), before);
    }
}

#[test]
fn occupied_output_burns_ready_attempt_without_touching_output_or_scan() {
    let mut owner = BlockScan::new(3, 2, 2).unwrap().into_resident_finish();
    let before = snapshot(&owner);
    let mut out = Some(completed_scan().finish().unwrap());
    let expected = out.clone();
    let pointer = out.as_ref().unwrap().blocks.as_ptr();
    let capacity = out.as_ref().unwrap().blocks.capacity();
    assert_eq!(owner.finish_into(&mut out), Err(Error::Binding));
    assert_eq!(out, expected);
    assert_eq!(out.as_ref().unwrap().blocks.as_ptr(), pointer);
    assert_eq!(out.as_ref().unwrap().blocks.capacity(), capacity);
    assert_eq!(snapshot(&owner), before);
    assert!(matches!(owner.state, FinishState::CleanupOnly));
    // Freeing the output slot cannot revive an attempt refused before Shape.
    drop(out.take());
    assert_eq!(owner.finish_into(&mut out), Err(Error::Binding));
    assert!(out.is_none());
    assert_eq!(snapshot(&owner), before);
}

#[test]
fn wrong_state_precedes_missing_pending_and_malformed_record_checks() {
    for state in [FinishState::CleanupOnly, FinishState::Complete] {
        let mut owner = completed_scan().into_resident_finish();
        owner.state = state;
        owner.scan.0.as_mut().unwrap().block_rows = 0;
        let before = snapshot(&owner);
        let mut out = None;
        assert_eq!(owner.finish_into(&mut out), Err(Error::Binding));
        assert!(out.is_none());
        assert!(matches!(owner.state, FinishState::CleanupOnly));
        assert_eq!(snapshot(&owner), before);
        let mut absent = BlockScan(None).into_resident_finish();
        absent.state = state;
        assert_eq!(absent.finish_into(&mut out), Err(Error::Binding));
        assert!(absent.scan.0.is_none());
        assert!(matches!(absent.state, FinishState::CleanupOnly));
    }
}

#[test]
fn missing_pending_binds_and_cannot_be_finished_again() {
    assert_eq!(BlockScan(None).finish(), Err(Error::Binding));
    let mut owner = BlockScan(None).into_resident_finish();
    let mut out = None;
    for _ in 0..2 {
        assert_eq!(owner.finish_into(&mut out), Err(Error::Binding));
        assert!(out.is_none());
        assert!(owner.scan.0.is_none());
        assert!(matches!(owner.state, FinishState::CleanupOnly));
    }
}

#[test]
fn precommit_panic_preserves_caller_resident_pending_and_original_vec_no_retry() {
    let mut owner = completed_scan().into_resident_finish();
    let before = snapshot(&owner);
    let mut out = None;
    PANIC_FINISH_PRECOMMIT.with(|flag| flag.set(true));
    let result = catch_unwind(AssertUnwindSafe(|| owner.finish_into(&mut out)));
    assert!(result.is_err());
    assert!(out.is_none());
    assert_eq!(snapshot(&owner), before);
    assert!(matches!(owner.state, FinishState::CleanupOnly));
    assert!(!PANIC_FINISH_PRECOMMIT.with(|flag| flag.get()));
    assert_eq!(owner.finish_into(&mut out), Err(Error::Binding));
    assert!(out.is_none());
    assert_eq!(snapshot(&owner), before);
}

#[test]
fn successful_finish_transfers_original_vec_and_preserves_legacy_statistics() {
    let expected = completed_scan().finish().unwrap();
    let mut owner = completed_scan().into_resident_finish();
    let before = snapshot(&owner);
    let mut out = None;
    assert_eq!(owner.finish_into(&mut out), Ok(()));
    assert!(matches!(owner.state, FinishState::Complete));
    let record = out.as_ref().unwrap();
    assert_eq!(record, &expected);
    assert_eq!(record.geometry(), before.geometry);
    assert_eq!(record.blocks, before.blocks);
    assert_eq!(record.blocks.as_ptr(), before.pointer);
    assert_eq!(record.blocks.capacity(), before.capacity);
    assert_eq!(record.block(0, 2).unwrap().operator_squared(), Ok(30));
    assert_eq!(record.block(2, 1).unwrap().operator_squared(), Ok(61));
    let shell = owner.scan.0.as_ref().unwrap();
    assert_eq!(shell as *const Pending, before.pending);
    assert_eq!((shell.rows, shell.columns, shell.block_rows), before.geometry);
    assert_eq!(shell.offset, before.offset);
    assert!(shell.scan.is_none());
    assert!(shell.blocks.is_empty());
    assert_eq!(shell.blocks.capacity(), 0);
}

#[test]
fn repeat_after_success_is_binding_and_complete_becomes_cleanup_only() {
    for occupied in [false, true] {
        let mut owner = completed_scan().into_resident_finish();
        let mut out = None;
        owner.finish_into(&mut out).unwrap();
        let retained = if occupied { None } else { out.take() };
        let expected = out.clone();
        let pointer = out.as_ref().map(|record| record.blocks.as_ptr());
        let before = snapshot(&owner);
        assert_eq!(owner.finish_into(&mut out), Err(Error::Binding));
        assert_eq!(out, expected);
        assert_eq!(out.as_ref().map(|record| record.blocks.as_ptr()), pointer);
        assert_eq!(snapshot(&owner), before);
        assert!(matches!(owner.state, FinishState::CleanupOnly));
        if let Some(retained) = retained {
            assert_eq!(retained, completed_scan().finish().unwrap());
        }
    }
}

#[test]
fn shared_borrowed_validation_preserves_geometry_short_circuit_and_block_selection() {
    let record = completed_scan().finish().unwrap();
    for (rows, columns) in [(3, 2), (0, 2), (3, 0), (2, 3), (usize::MAX, 2)] {
        assert_eq!(
            validate_parts(
                record.rows,
                record.columns,
                record.block_rows,
                &record.blocks,
                rows,
                columns,
            ),
            record.validate(rows, columns)
        );
        assert_eq!(
            record.validate(rows, columns),
            if (rows, columns) == (3, 2) {
                Ok(())
            } else {
                Err(Error::Shape)
            }
        );
    }
    for (start, rows) in [(0, 0), (1, 2), (2, 2), (3, 1), (usize::MAX, 1)] {
        assert!(record.block(start, rows).is_none());
    }
    assert!(record.block(0, 2).is_some());
    assert!(record.block(2, 1).is_some());
}

#[test]
fn overflowing_norm_product_still_uses_frobenius_in_both_finish_paths() {
    let change = |scan: &mut BlockScan| {
        for block in &mut scan.0.as_mut().unwrap().blocks {
            block.maximum_row_l1 = u128::MAX;
            block.maximum_column_l1 = 2;
            block.frobenius_squared = 19;
        }
    };
    let mut legacy = completed_scan();
    change(&mut legacy);
    let expected = legacy.finish().unwrap();
    let mut scan = completed_scan();
    change(&mut scan);
    let mut owner = scan.into_resident_finish();
    let before = snapshot(&owner);
    let mut out = None;
    owner.finish_into(&mut out).unwrap();
    let actual = out.as_ref().unwrap();
    assert_eq!(actual, &expected);
    assert_eq!(actual.blocks.as_ptr(), before.pointer);
    assert_eq!(actual.blocks.capacity(), before.capacity);
    assert_eq!(actual.block(0, 2).unwrap().operator_squared(), Ok(19));
    assert_eq!(actual.block(2, 1).unwrap().operator_squared(), Ok(19));
}
