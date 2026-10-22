use super::*;

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
fn proof() -> Proof {
    Proof {
        key: RowKey {
            owner: 0,
            offset: 0,
            width: 3,
        },
        energy: Some(432),
        mass: None,
    }
}

#[test]
fn input_containers_exact_one_short_and_shape_rejection() {
    let inputs = [Register {
        lanes: 3,
        bounds: Interval::new(-12, 12).unwrap(),
    }; 3];
    let required = Builder::input_metadata_bytes(inputs.len()).unwrap();
    let b = Builder::new_admitted(WeightMode::Public, &inputs, required).unwrap();
    assert_eq!(b.registers.capacity(), inputs.len());
    assert_eq!(b.regions.capacity(), inputs.len());
    assert_eq!(b.proof_live_metadata().unwrap(), required);
    // Static input quote is checked before either Vec reserve or any Region Box.
    assert!(matches!(
        Builder::new_admitted(WeightMode::Public, &inputs, required - 1),
        Err(Error::Bounds)
    ));
    assert!(matches!(
        Builder::new_admitted(WeightMode::Public, &[], required),
        Err(Error::Bounds)
    ));
    assert!(matches!(
        Builder::new_admitted(WeightMode::Private, &inputs, required),
        Err(Error::Binding)
    ));
    let invalid = [Register {
        lanes: 0,
        bounds: inputs[0].bounds,
    }];
    assert!(matches!(
        Builder::new_admitted(WeightMode::Public, &invalid, required),
        Err(Error::Shape)
    ));
    assert!(Builder::input_metadata_bytes(MAX_NODES - 1).unwrap() <= MAX_PROGRAM_BYTES);
}

fn incoming(row: &Layout) -> (Op, Vec<Region>) {
    let op = Op::Binary {
        left: row.clone(),
        right: row.clone(),
        multiply: false,
    };
    // Retained spare constructor capacity must be priced, not just len == 1.
    let mut regions = Vec::with_capacity(7);
    regions.push(Region {
        offset: 0,
        len: 3,
        bounds: Interval::new(-24, 24).unwrap(),
    });
    (op, regions)
}

#[test]
fn push_exact_one_short_prices_capacities_output_and_boxing_overlap() {
    for existing in [0, 3, 4] {
        for short in [false, true] {
            let mut b = builder();
            let row = b.input(0).unwrap();
            for _ in 0..existing {
                let (op, regions) = incoming(&row);
                b.push(op, 3, regions, WorkBudget::default(), 0).unwrap();
            }
            let old = (
                b.registers.capacity(),
                b.regions.capacity(),
                b.ops.capacity(),
            );
            let state = (
                b.registers.len(),
                b.regions.len(),
                b.ops.len(),
                b.advances,
                b.work,
            );
            let (op, regions) = incoming(&row);
            let required =
                b.proof_live_metadata().unwrap() + b.push_storage_bytes(&op, &regions, 23).unwrap();
            b.proof_metadata_limit = required - usize::from(short);
            let result = b.push_with_temporary(op, 3, regions, WorkBudget::default(), 0, 23);
            if short {
                assert!(matches!(result, Err(Error::Bounds)));
                assert_eq!(
                    (
                        b.registers.capacity(),
                        b.regions.capacity(),
                        b.ops.capacity()
                    ),
                    old
                );
                assert_eq!(
                    (
                        b.registers.len(),
                        b.regions.len(),
                        b.ops.len(),
                        b.advances,
                        b.work
                    ),
                    state
                );
                assert!(b.proof_admission_retired);
                assert!(matches!(b.finish(&[row]), Err(Error::Binding)));
            } else {
                let output = result.unwrap();
                assert_eq!(output.row_key(3).unwrap().owner, state.0);
                assert_eq!(b.ops.len(), existing + 1);
                assert!(!b.proof_admission_retired);
            }
        }
    }
    assert_eq!(owner_growth_capacity(0, 0).unwrap(), 4);
    assert_eq!(owner_growth_capacity(MAX_NODES - 1, MAX_NODES).unwrap(), 0);
    assert_eq!(
        owner_growth_capacity(MAX_NODES, MAX_NODES),
        Err(Error::Bounds)
    );
}

#[test]
fn push_bounds_reject_before_injected_allocation_and_failures_retire() {
    for short in [false, true] {
        let mut b = builder();
        let row = b.input(0).unwrap();
        let (op, regions) = incoming(&row);
        let required =
            b.proof_live_metadata().unwrap() + b.push_storage_bytes(&op, &regions, 0).unwrap();
        b.proof_metadata_limit = required - usize::from(short);
        b.fail_builder_allocation = true;
        let old = (
            b.registers.capacity(),
            b.regions.capacity(),
            b.ops.capacity(),
        );
        let error = b.push(op, 3, regions, WorkBudget::default(), 0).err();
        assert_eq!(
            error,
            Some(if short {
                Error::Bounds
            } else {
                Error::Allocation
            })
        );
        assert_eq!(
            (
                b.registers.capacity(),
                b.regions.capacity(),
                b.ops.capacity()
            ),
            old
        );
        assert!(b.ops.is_empty());
        assert_eq!(b.registers.len(), 1);
        assert_eq!(b.advances, 1);
        assert!(b.proof_admission_retired);
        assert!(matches!(b.input(0), Err(Error::Binding)));
    }
}

#[test]
fn incoming_region_spare_capacity_is_not_refunded_at_push_boundary() {
    let mut b = builder();
    let row = b.input(0).unwrap();
    let (op, large) = incoming(&row);
    let small = vec![Region {
        offset: 0,
        len: 3,
        bounds: Interval::new(-24, 24).unwrap(),
    }];
    assert_eq!(large.len(), small.len());
    assert_eq!(
        b.push_storage_bytes(&op, &large, 0).unwrap()
            - b.push_storage_bytes(&op, &small, 0).unwrap(),
        (large.capacity() - small.capacity()) * size_of::<Region>()
    );
}

#[test]
fn post_constructor_intern_exact_one_short_and_fault_retirement() {
    for fault in [false, true] {
        for short in [false, true] {
            let mut b = builder();
            // Native Program already exists: this test makes no preconstruction
            // claim for the core constructor, only subsequent Arc/map allocation.
            let p = tensor::Program::all_valid(3).unwrap();
            let digest = p.digest();
            let required = b.proof_live_metadata().unwrap()
                + b.template_insert_bytes(&digest, p.resources().unwrap(), 29)
                    .unwrap();
            b.proof_metadata_limit = required - usize::from(short);
            b.fail_builder_allocation = fault;
            let result = b.intern_with_temporary(p, 29);
            if short || fault {
                assert_eq!(
                    result.err(),
                    Some(if short {
                        Error::Bounds
                    } else {
                        Error::Allocation
                    })
                );
                assert!(b.templates.is_empty());
                assert!(b.template_metadata.is_empty());
                assert!(b.proof_admission_retired);
                assert!(matches!(b.finish(&[]), Err(Error::Binding)));
            } else {
                let p = result.unwrap();
                assert_eq!(p.digest(), digest);
                assert_eq!(b.templates.len(), 1);
                assert!(Arc::ptr_eq(&p, &b.templates[&digest]));
            }
        }
    }
}

#[test]
fn duplicate_intern_still_prices_incoming_native_template_and_buffers() {
    let mut b = builder();
    let p = b.intern(tensor::Program::all_valid(3).unwrap()).unwrap();
    let duplicate = tensor::Program::all_valid(3).unwrap();
    let required = b.proof_live_metadata().unwrap()
        + b.template_insert_bytes(&duplicate.digest(), duplicate.resources().unwrap(), 13)
            .unwrap();
    b.proof_metadata_limit = required;
    let repeated = b.intern_with_temporary(duplicate, 13).unwrap();
    assert!(Arc::ptr_eq(&p, &repeated));
    assert_eq!(b.templates.len(), 1);
    let duplicate = tensor::Program::all_valid(3).unwrap();
    b.proof_metadata_limit = required - 1;
    assert!(matches!(
        b.intern_with_temporary(duplicate, 13),
        Err(Error::Bounds)
    ));
    assert_eq!(b.templates.len(), 1);
    assert!(b.proof_admission_retired);
}

#[test]
fn exact_and_one_short_growth_charge_old_and_new_capacity() {
    for existing in [0, 4] {
        for short in [false, true] {
            let mut b = builder();
            for _ in 0..existing {
                b.record_proof(proof(), 0).unwrap();
            }
            let old = b.proofs.capacity();
            let capacity = growth_capacity(b.proofs.len(), old).unwrap();
            let required =
                b.proof_live_metadata().unwrap() + capacity * size_of::<Proof>() + 64 + 17;
            b.proof_metadata_limit = required - usize::from(short);
            let result = b.record_proof(proof(), 17);
            if short {
                assert_eq!(result, Err(Error::Bounds));
                assert_eq!(b.proofs.len(), existing);
                assert_eq!(b.proofs.capacity(), old);
                assert!(b.proof_admission_retired);
                assert!(matches!(b.input(0), Err(Error::Binding)));
                assert_eq!(b.record_proof(proof(), 0), Err(Error::Binding));
            } else {
                result.unwrap();
                assert_eq!(b.proofs.capacity(), capacity);
                assert_eq!(b.proofs.len(), existing + 1);
            }
        }
    }
}

#[test]
fn spare_capacity_is_live_and_maximum_capacity_is_bounded() {
    assert_eq!(growth_capacity(0, 0).unwrap(), 4);
    assert_eq!(growth_capacity(1, 4).unwrap(), 0);
    assert_eq!(
        growth_capacity(MAX_VALUES / 2, MAX_VALUES / 2).unwrap(),
        MAX_VALUES
    );
    assert_eq!(growth_capacity(MAX_VALUES - 1, MAX_VALUES).unwrap(), 0);
    assert_eq!(growth_capacity(MAX_VALUES, MAX_VALUES), Err(Error::Bounds));
    assert_eq!(growth_capacity(1, 0), Err(Error::Bounds));
    let mut b = builder();
    let initial = b.proof_live_metadata().unwrap();
    b.record_proof(proof(), 0).unwrap();
    assert_eq!(
        b.proof_live_metadata().unwrap() - initial,
        4 * size_of::<Proof>()
    );
    b.proof_metadata_limit = b.proof_live_metadata().unwrap() + 64;
    b.record_proof(proof(), 0).unwrap();
    assert_eq!(b.proofs.capacity(), 4);
}

#[test]
fn producer_buffers_and_bounded_row_scratch_are_admitted_before_reserve() {
    for short in [false, true] {
        let mut b = builder();
        let row = b.input(0).unwrap();
        let mut programs = Vec::new();
        let mut certificates = Vec::new();
        let required = b.proof_live_metadata().unwrap()
            + array::<Arc<tensor::Program>>(1).unwrap()
            + array::<(usize, Option<u128>, Option<u64>)>(1).unwrap()
            + Builder::proof_row_scratch(&row).unwrap()
            + 17;
        b.proof_metadata_limit = required - usize::from(short);
        let result = b.proof_producer_buffers(&row, 3, &mut programs, &mut certificates, 17);
        if short {
            assert_eq!(result, Err(Error::Bounds));
            assert_eq!(programs.capacity(), 0);
            assert_eq!(certificates.capacity(), 0);
            assert!(b.proof_admission_retired);
        } else {
            result.unwrap();
            assert_eq!(programs.capacity(), 1);
            assert_eq!(certificates.capacity(), 1);
            b.proof_producer_row_preflight(&row, &programs, &certificates, 17)
                .unwrap();
            // Adding a live template must not reuse the old row-scratch check.
            b.proof_metadata_limit = MAX_PROGRAM_BYTES;
            let p = tensor::Program::all_valid(3).unwrap();
            b.intern(p).unwrap();
            b.proof_metadata_limit = required;
            assert_eq!(
                b.proof_producer_row_preflight(&row, &programs, &certificates, 17),
                Err(Error::Bounds)
            );
            assert!(b.proof_admission_retired);
        }
    }
}

#[test]
fn injected_allocation_fault_retires_without_publishing_proof() {
    let mut b = builder();
    b.fail_proof_allocation = true;
    assert_eq!(b.record_proof(proof(), 0), Err(Error::Allocation));
    assert!(b.proofs.is_empty());
    assert_eq!(b.proofs.capacity(), 0);
    assert!(matches!(b.input(0), Err(Error::Binding)));
    assert!(matches!(b.finish(&[]), Err(Error::Binding)));
}

#[test]
fn checked_rows_admission_and_growth_fault_cannot_publish_or_finish() {
    for allocation_fault in [false, true] {
        let mut b = builder();
        let row = b.input(0).unwrap();
        let count = if allocation_fault { 4 } else { 1 };
        for _ in 0..count {
            b.record_proof(
                Proof {
                    energy: Some(150),
                    ..proof()
                },
                0,
            )
            .unwrap();
        }
        if allocation_fault {
            b.fail_proof_allocation = true;
        } else {
            b.proof_metadata_limit = 1;
        }
        let capacity = b.proofs.capacity();
        let error = b.checked_rows(&row, 3, 9, None).err();
        assert_eq!(
            error,
            Some(if allocation_fault {
                Error::Allocation
            } else {
                Error::Bounds
            })
        );
        assert_eq!(b.proofs.len(), count);
        assert_eq!(b.proofs.capacity(), capacity);
        assert!(b.proof_admission_retired);
        assert!(matches!(b.finish(&[row]), Err(Error::Binding)));
    }
}

#[test]
fn finalization_exact_one_short_includes_spare_capacity_and_boxes() {
    for short in [false, true] {
        let mut b = builder();
        let input = b.input(0).unwrap();
        let output = b.residual(&input, &input, 3).unwrap();
        assert!(b.proofs.capacity() > b.proofs.len());
        let required = b.proof_finish_bytes(&[output.clone()]).unwrap();
        assert!(required > b.proof_live_metadata().unwrap());
        b.proof_metadata_limit = required - usize::from(short);
        let result = b.finish(&[output]);
        if short {
            assert!(matches!(result, Err(Error::Bounds)));
        } else {
            assert!(result.is_ok());
        }
    }
}

#[test]
fn exact_ordered_alias_only_and_nonbeneficial_controls() {
    for energy in [150, 243, 432] {
        let mut b = builder();
        let row = b.input(0).unwrap();
        b.record_proof(
            Proof {
                energy: Some(energy),
                mass: Some(7),
                ..proof()
            },
            0,
        )
        .unwrap();
        let alias = Layout::concat(&[row.slice(0, 1).unwrap(), row.slice(1, 2).unwrap()]).unwrap();
        let (safe, _) = b.checked_rows(&alias, 3, 9, None).unwrap();
        let retained = b.proof(&safe);
        if energy < 243 {
            assert_eq!(retained.unwrap().energy, Some(energy));
            assert_eq!(retained.unwrap().mass, None);
        } else {
            assert!(retained.is_none());
        }
    }
    for indices in [[0, 0, 2], [2, 1, 0]] {
        let mut b = builder();
        let row = b.input(0).unwrap();
        b.record_proof(
            Proof {
                energy: Some(150),
                ..proof()
            },
            0,
        )
        .unwrap();
        let parts = indices.map(|i| row.slice(i, 1).unwrap());
        let alias = Layout::concat(&parts).unwrap();
        let (safe, _) = b.checked_rows(&alias, 3, 9, None).unwrap();
        assert!(b.proof(&safe).is_none());
    }
    let mut b = builder();
    let row = b.input(0).unwrap();
    b.record_proof(
        Proof {
            energy: Some(150),
            ..proof()
        },
        0,
    )
    .unwrap();
    let (partial, _) = b
        .checked_rows(&row.slice(0, 2).unwrap(), 2, 9, None)
        .unwrap();
    assert!(b.proof(&partial).is_none());
    assert!(matches!(
        b.checked_rows(&row, 2, 9, None),
        Err(Error::Shape)
    ));
    let other = b.binary(&row, &row, false).unwrap();
    let mixed = Layout::concat(&[row.slice(0, 2).unwrap(), other.slice(2, 1).unwrap()]).unwrap();
    let (safe, _) = b.checked_rows(&mixed, 3, 9, None).unwrap();
    assert!(b.proof(&safe).is_none());
    let (safe, _) = b.checked_rows(&other, 3, 9, None).unwrap();
    assert!(b.proof(&safe).is_none());
}

#[test]
fn checked_rows_zero_cap_alias_and_overflow_preserve_guard_schedule() {
    for bound in [0, 12] {
        let mut b = builder();
        let row = b.input(0).unwrap();
        b.record_proof(proof(), 0).unwrap();
        let before = b.proofs.len();
        let (safe, valid) = b.checked_rows(&row, 3, bound, None).unwrap();
        assert_eq!(b.proofs.len(), before);
        if bound == 12 {
            assert_eq!(safe.row_key(3), row.row_key(3));
            assert!(valid.is_none());
            assert!(b.ops.is_empty());
        } else {
            assert!(b.proof(&safe).is_none());
            assert!(valid.is_some());
        }
    }
    let make = || {
        Builder::new(
            WeightMode::Public,
            &[Register {
                lanes: 5,
                bounds: Interval::new(i64::MIN, i64::MAX).unwrap(),
            }],
        )
        .unwrap()
    };
    let mut baseline = make();
    let row = baseline.input(0).unwrap();
    baseline.checked(&row, i64::MAX, None).unwrap();
    let mut b = make();
    let row = b.input(0).unwrap();
    assert!(matches!(
        b.checked_rows(&row, 5, i64::MAX, None),
        Err(Error::Bounds)
    ));
    assert!(b.proofs.is_empty());
    assert_eq!(b.ops.len(), baseline.ops.len());
    assert_eq!(b.work, baseline.work);
    assert_eq!(b.advances, baseline.advances);
    assert_eq!(b.checked_return_slice_attempts, 2);
    assert!(b.proof_admission_retired);
    assert!(matches!(b.input(0), Err(Error::Binding)));
    assert!(matches!(b.finish(&[row]), Err(Error::Binding)));
}

#[test]
fn each_checked_return_slice_fault_burns_without_changing_sticky_guard_work() {
    let make = || {
        let mut b = builder();
        let row = b.input(0).unwrap();
        let (_, previous) = b.checked(&row, 9, None).unwrap();
        (b, row, previous)
    };
    let (mut baseline, row, previous) = make();
    let (safe, valid) = baseline.checked(&row, 9, previous.as_ref()).unwrap();
    let observations = baseline.checked_return_slice_attempts;
    assert_eq!(observations, 4);
    assert!(!baseline.proof_admission_retired);
    for failed_slice in 0..2 {
        let (mut b, row, previous) = make();
        let before = b.checked_return_slice_attempts;
        b.checked_return_slice_fail_at = Some(before + failed_slice);
        assert!(matches!(
            b.checked(&row, 9, previous.as_ref()),
            Err(Error::Allocation)
        ));
        assert_eq!(b.checked_return_slice_attempts, before + failed_slice + 1);
        assert_eq!(b.ops.len(), baseline.ops.len());
        assert_eq!(b.work, baseline.work);
        assert_eq!(b.advances, baseline.advances);
        assert!(b.proofs.is_empty());
        for (actual, expected) in b.ops.iter().zip(&baseline.ops) {
            match (actual, expected) {
                (
                    Op::Check {
                        checked: a,
                        valid: av,
                        ..
                    },
                    Op::Check {
                        checked: e,
                        valid: ev,
                        ..
                    },
                ) => {
                    assert!(a
                        .iter()
                        .map(|p| p.digest())
                        .eq(e.iter().map(|p| p.digest())));
                    assert!(av
                        .iter()
                        .map(|p| p.digest())
                        .eq(ev.iter().map(|p| p.digest())));
                }
                _ => panic!("expected unchanged complete guard schedule"),
            }
        }
        assert!(b.proof_admission_retired);
        assert!(matches!(b.input(0), Err(Error::Binding)));
        assert!(matches!(
            b.checked(&row, 9, previous.as_ref()),
            Err(Error::Binding)
        ));
        assert_eq!(b.checked_return_slice_attempts, before + failed_slice + 1);
        assert!(matches!(b.finish(&[row]), Err(Error::Binding)));
    }
    // Statically contained views remain aliases: no hook, work or allocation.
    baseline.checked_return_slice_fail_at = Some(observations);
    let (alias, sticky) = baseline.checked(&safe, 9, valid.as_ref()).unwrap();
    assert_eq!(alias.row_key(3), safe.row_key(3));
    assert_eq!(baseline.checked_return_slice_attempts, observations);
    assert!(!baseline.proof_admission_retired);
    assert!(baseline.finish(&[alias, sticky.unwrap()]).is_ok());
}

#[test]
fn checked_rms_native_failure_after_guard_preserves_error_and_retires() {
    let gamma = [i64::MAX; 3];
    let expected = tensor::Program::rms_norm_certified(1, 3, Interval::new(-9, 9).unwrap(), &gamma)
        .err()
        .expect("native gamma-overflow control must reject");
    let mut baseline = builder();
    let row = baseline.input(0).unwrap();
    baseline.checked(&row, 9, None).unwrap();
    let mut b = builder();
    let row = b.input(0).unwrap();
    assert_eq!(b.rms(&row, 3, &gamma, Some(9), None).err(), Some(expected));
    assert_eq!(b.ops.len(), baseline.ops.len());
    assert_eq!(b.work, baseline.work);
    assert_eq!(b.advances, baseline.advances);
    assert!(b.proofs.is_empty());
    assert!(b.proof_admission_retired);
    assert!(matches!(b.input(0), Err(Error::Binding)));
    assert!(matches!(b.finish(&[row]), Err(Error::Binding)));
}

#[test]
fn checked_rms_consumes_transferred_energy_and_original_program_digest() {
    let cap = 512 * ONE;
    let energy = (cap as u128).pow(2);
    let mut b = Builder::new(
        WeightMode::Public,
        &[Register {
            lanes: 3,
            bounds: Interval::new(-2 * cap, 2 * cap).unwrap(),
        }],
    )
    .unwrap();
    let row = b.input(0).unwrap();
    // Test-only native-ledger fixture; no public certificate constructor exists.
    b.record_proof(
        Proof {
            energy: Some(energy),
            ..proof()
        },
        0,
    )
    .unwrap();
    let (output, valid) = b.rms(&row, 3, &[ONE; 3], Some(cap), None).unwrap();
    assert!(valid.is_some());
    let (expected, squared) = tensor::Program::rms_norm_with_input_norm(
        1,
        3,
        Interval::new(-cap, cap).unwrap(),
        &[ONE; 3],
        energy,
    )
    .unwrap();
    match b.ops.last().unwrap() {
        Op::Tensor {
            input, programs, ..
        } => {
            assert_eq!(b.proof(input).unwrap().energy, Some(energy));
            assert_eq!(b.proof(input).unwrap().mass, None);
            assert_eq!(programs[0].digest(), expected.digest());
            assert_eq!(programs[0].work_budget(), expected.work_budget());
        }
        _ => panic!("expected original RMS tensor owner"),
    }
    assert_eq!(b.proof(&output).unwrap().energy, Some(squared));
}

#[test]
fn intern_uses_owner_template_sizes_without_changing_identity_or_work() {
    let mut b = builder();
    let make = || {
        let mut t = tensor::Builder::new(3, Interval::new(-12, 12).unwrap()).unwrap();
        let output = t
            .affine_lanes(
                t.input(),
                &[multiply::arithmetic::Affine {
                    left: 2,
                    right: 0,
                    constant: 1,
                }; 3],
            )
            .unwrap();
        t.finish(output).unwrap()
    };
    let program = make();
    let resources = program.resources().unwrap();
    assert!(resources.public_table_bytes > 0);
    let digest = program.digest();
    let work = program.work_budget();
    let interned = b.intern(program).unwrap();
    assert_eq!(interned.digest(), digest);
    assert_eq!(interned.work_budget(), work);
    assert_eq!(
        b.template_metadata[&digest],
        resources.immutable_template_bytes
    );
    assert!(resources.shared_admitted_bytes <= puma::MAX_ADMITTED_BYTES);
    let duplicate = make();
    assert!(Arc::ptr_eq(&interned, &b.intern(duplicate).unwrap()));
    assert_eq!(b.templates.len(), 1);
}
