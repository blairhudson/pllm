use super::*;

fn attempts() -> usize {
    ALLOCATION_ATTEMPTS.with(|n| n.get())
}

fn builder() -> Builder {
    Builder::new(
        WeightMode::Public,
        &[Register {
            lanes: 3,
            bounds: Interval::new(-12, 12).unwrap(),
        }],
    )
    .unwrap()
}

#[test]
fn input_view_exact_one_short_counts_all_live_vec_capacities_and_retires() {
    for spare in [false, true] {
        for short in [false, true] {
            for fault in [false, true] {
                let mut b = builder();
                if spare {
                    b.registers.try_reserve_exact(7).unwrap();
                    b.regions.try_reserve_exact(5).unwrap();
                    b.ops.try_reserve_exact(3).unwrap();
                    b.proofs.try_reserve_exact(9).unwrap();
                }
                let quote = Layout::quote_contiguous(0, 3).unwrap();
                // Empty op/template/proof ledgers: reconstruct existing census
                // from input-owner quote plus every actual spare Vec capacity.
                let required = Builder::input_metadata_bytes(1).unwrap()
                    + (b.registers.capacity() - 1) * size_of::<Register>()
                    + (b.regions.capacity() - 1) * size_of::<Box<[Region]>>()
                    + b.ops.capacity() * size_of::<Op>()
                    + b.proofs.capacity() * size_of::<Proof>()
                    + quote.construction_peak_bytes;
                b.proof_metadata_limit = required - usize::from(short);
                b.fail_builder_allocation = fault;
                let old = (
                    b.registers.capacity(),
                    b.regions.capacity(),
                    b.ops.capacity(),
                    b.proofs.capacity(),
                );
                let before = attempts();
                let result = b.input(0);
                if short || fault {
                    assert_eq!(
                        result.err(),
                        Some(if short {
                            Error::Bounds
                        } else {
                            Error::Allocation
                        })
                    );
                    assert_eq!(attempts(), before);
                    assert!(b.proof_admission_retired);
                    assert!(matches!(b.input(0), Err(Error::Binding)));
                } else {
                    assert_eq!(
                        result.unwrap().row_key(3),
                        Some(RowKey {
                            owner: 0,
                            offset: 0,
                            width: 3
                        })
                    );
                    assert_eq!(attempts(), before + 1);
                    assert!(!b.proof_admission_retired);
                }
                assert_eq!(
                    (
                        b.registers.capacity(),
                        b.regions.capacity(),
                        b.ops.capacity(),
                        b.proofs.capacity()
                    ),
                    old
                );
                assert_eq!(b.registers.len(), 1);
                assert_eq!(b.regions.len(), 1);
                assert!(b.ops.is_empty());
                assert!(b.proofs.is_empty());
                assert_eq!(b.advances, 1);
                assert_eq!(b.work, WorkBudget::default());
                if short || fault {
                    assert!(matches!(b.finish(&[]), Err(Error::Binding)));
                }
            }
        }
    }
}

#[test]
fn input_failure_after_published_work_cannot_continue_or_finish_saved_output() {
    let mut b = builder();
    let row = b.input(0).unwrap();
    let output = b.binary(&row, &row, false).unwrap();
    let old = (b.ops.len(), b.registers.len(), b.work, b.advances);
    b.proof_metadata_limit = 1;
    let before = attempts();
    assert!(matches!(b.input(0), Err(Error::Bounds)));
    assert_eq!(attempts(), before);
    assert_eq!((b.ops.len(), b.registers.len(), b.work, b.advances), old);
    assert!(matches!(b.binary(&row, &row, false), Err(Error::Binding)));
    assert_eq!(attempts(), before);
    assert!(matches!(b.finish(&[output]), Err(Error::Binding)));
}

#[test]
fn input_binding_failure_stays_nonretiring_but_quote_overflow_retires() {
    let mut b = builder();
    let before = attempts();
    assert!(matches!(b.input(1), Err(Error::Binding)));
    assert_eq!(attempts(), before);
    assert!(!b.proof_admission_retired);
    b.input(0).unwrap();
    b.registers[0].lanes = usize::MAX;
    let before = attempts();
    assert!(matches!(b.input(0), Err(Error::Bounds)));
    assert_eq!(attempts(), before);
    assert!(b.proof_admission_retired);
    assert!(matches!(b.finish(&[]), Err(Error::Binding)));
}
