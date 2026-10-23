use super::work_preflight::row_entries;
use super::*;

fn equal(lanes: usize, input: Interval) -> ClosedFactory<'static> {
    Construction::EqualZero { lanes }
        .bind(FactoryParameters::EqualZero { input })
        .unwrap()
}
fn rms(rows: usize, input: Interval, gamma: &[i64]) -> ClosedFactory<'_> {
    Construction::RmsNormCertified {
        rows,
        width: gamma.len(),
        gamma_len: gamma.len(),
    }
    .bind(FactoryParameters::RmsNormCertified { input, gamma })
    .unwrap()
}
fn same(a: &Program, e: &Program) {
    assert_eq!(a.nodes, e.nodes);
    assert_eq!(a.digest(), e.digest());
    assert_eq!(a.work_budget(), e.work_budget());
    assert_eq!(a.numeric_profile(), e.numeric_profile());
    assert_eq!(a.bit_widths, e.bit_widths);
    assert_eq!(a.last_use, e.last_use);
    assert_eq!(a.slots, e.slots);
    assert_eq!(a.register_count, e.register_count);
    assert_eq!(a.public_affines, e.public_affines);
    assert_eq!(a.checked_validity, e.checked_validity);
}

#[test]
fn closed_supported_factories_match_original_native_work_records_and_certificate_labels() {
    let input = Interval::new(-32, 32).unwrap();
    let gamma = [1 << 18; 3];
    let zero = Interval::new(0, 0).unwrap();
    let cases = [
        (
            Construction::EqualZero { lanes: 3 },
            FactoryParameters::EqualZero { input },
        ),
        (
            Construction::Builder { lanes: 3 },
            FactoryParameters::Rescale { input, shift: 5 },
        ),
        (
            Construction::CheckedRange { lanes: 3 },
            FactoryParameters::CheckedRange {
                incoming: Interval::new(i64::MIN, i64::MAX).unwrap(),
                accepted: input,
                fallback: 0,
            },
        ),
        (
            Construction::AllValid { lanes: 3 },
            FactoryParameters::AllValid,
        ),
        (
            Construction::RmsNormCertified {
                rows: 1,
                width: 3,
                gamma_len: 3,
            },
            FactoryParameters::RmsNormCertified {
                input,
                gamma: &gamma,
            },
        ),
        (
            Construction::RmsNormWithInputNorm {
                rows: 1,
                width: 3,
                gamma_len: 3,
            },
            FactoryParameters::RmsNormWithInputNorm {
                input: zero,
                gamma: &gamma,
                squared_norm: 0,
            },
        ),
        (
            Construction::SoftmaxCertified { rows: 1, width: 3 },
            FactoryParameters::SoftmaxCertified { input },
        ),
    ];
    let mut workspace = WorkPreflightWorkspace::empty();
    for (index, (construction, parameters)) in cases.into_iter().enumerate() {
        let before = row_entries();
        let factory = construction.bind(parameters).unwrap();
        let ticket = factory.preflight(3, &mut workspace).unwrap();
        let summary = ticket.summary();
        assert_eq!(row_entries(), before);
        assert_eq!(summary.full.construction, construction.quote().unwrap());
        assert_eq!(summary.full.invocations, 3);
        assert!(summary.tail.is_none());
        assert_eq!(summary.work, summary.full.unit_work.checked_mul(3).unwrap());
        let mut actual_total = WorkBudget::default();
        let admitted = {
            let mut ledger = CallerWorkLedger::new(
                &mut actual_total,
                summary.work.operations(),
                summary.work.cursors().aes_work,
            )
            .unwrap();
            ledger.admit(ticket).unwrap()
        };
        assert_eq!(actual_total, summary.work);
        assert_eq!(row_entries(), before);
        let (part, tail) = admitted.into_parts();
        assert!(tail.is_none());
        assert_eq!(part.summary(), summary.full);
        let actual = part.build().unwrap();
        assert_eq!(row_entries(), before + 1);
        assert_eq!(actual.program().work_budget(), summary.full.unit_work);
        let (expected, certificate) = match index {
            0 => (
                Program::equal_zero(3, input).unwrap(),
                ProducedCertificate::None,
            ),
            1 => {
                let mut b = Builder::new(3, input).unwrap();
                let x = b.input();
                let output = b.rescale(x, 5).unwrap();
                (b.finish(output).unwrap(), ProducedCertificate::None)
            }
            2 => (
                Program::checked_range(3, Interval::new(i64::MIN, i64::MAX).unwrap(), input, 0)
                    .unwrap(),
                ProducedCertificate::None,
            ),
            3 => (Program::all_valid(3).unwrap(), ProducedCertificate::None),
            4 => {
                let (p, c) = Program::rms_norm_certified(1, 3, input, &gamma).unwrap();
                (p, ProducedCertificate::GeneralOutputRowSquared(c))
            }
            5 => {
                let (p, c) = Program::rms_norm_with_input_norm(1, 3, zero, &gamma, 0).unwrap();
                (p, ProducedCertificate::GeneralOutputRowSquared(c))
            }
            6 => {
                let (p, c) = Program::softmax_certified(1, 3, input).unwrap();
                (p, ProducedCertificate::AttentionMass(c))
            }
            _ => unreachable!(),
        };
        same(actual.program(), &expected);
        assert_eq!(actual.certificate(), certificate);
    }
}

#[test]
fn closed_full_tail_work_binds_original_rows_gamma_and_each_fill_rounding() {
    let input = Interval::new(-32, 32).unwrap();
    let gamma = [1 << 18, -(1 << 17), 1 << 18];
    let mut workspace = WorkPreflightWorkspace::empty();
    let before = row_entries();
    let ticket = rms(2, input, &gamma)
        .preflight_full_tail(2, rms(1, input, &gamma), &mut workspace)
        .unwrap();
    let summary = ticket.summary();
    let tail = summary.tail.unwrap();
    assert_eq!(tail.invocations, 1);
    assert_eq!(
        summary.work,
        summary
            .full
            .unit_work
            .checked_mul(2)
            .unwrap()
            .checked_add(tail.unit_work)
            .unwrap()
    );
    assert_eq!(row_entries(), before);
    let mut total = WorkBudget::default();
    let admitted = {
        let mut ledger = CallerWorkLedger::new(
            &mut total,
            summary.work.operations(),
            summary.work.cursors().aes_work,
        )
        .unwrap();
        ledger.admit(ticket).unwrap()
    };
    let (full, tail) = admitted.into_parts();
    let full = full.build().unwrap();
    let tail = tail.unwrap().build().unwrap();
    assert_eq!(row_entries(), before + 2);
    assert_eq!(
        summary.work,
        full.program()
            .work_budget()
            .checked_mul(2)
            .unwrap()
            .checked_add(tail.program().work_budget())
            .unwrap()
    );
    let merged = Program::rms_norm_certified(5, 3, input, &gamma).unwrap().0;
    assert_ne!(summary.work.cursors(), merged.work_budget().cursors());
    assert_eq!(full.program().public_affines[0].len(), 6);
    assert_eq!(tail.program().public_affines[0].len(), 3);
    assert_eq!(total, summary.work);
}

#[test]
fn actual_work_short_caps_refuse_before_native_builder_entry_and_do_not_debit() {
    let input = Interval::new(-32, 32).unwrap();
    let gamma = [1 << 18; 3];
    let mut workspace = WorkPreflightWorkspace::empty();
    for use_rms in [false, true] {
        let make = || {
            if use_rms {
                rms(1, input, &gamma)
            } else {
                equal(3, input)
            }
        };
        let q = {
            let first = make().preflight(1, &mut workspace).unwrap();
            first.summary().work
        };
        let before = row_entries();
        for (ops, aes) in [
            (0, q.cursors().aes_work),
            (q.operations() - 1, q.cursors().aes_work),
            (q.operations(), 0),
            (q.operations(), q.cursors().aes_work - 1),
        ] {
            // Byte quote is valid, but cannot make work-short construction legal.
            let ticket = make().preflight(1, &mut workspace).unwrap();
            assert!(ticket.summary().full.construction.new_peak_bytes > 0);
            let mut actual_total = WorkBudget::default();
            {
                let mut ledger = CallerWorkLedger::new(&mut actual_total, ops, aes).unwrap();
                assert!(matches!(
                    ledger.admit(ticket),
                    Err(WorkPreflightError::Native(Error::Bounds))
                ));
                assert_eq!(ledger.total(), WorkBudget::default());
            }
            assert_eq!(actual_total, WorkBudget::default());
            assert_eq!(row_entries(), before);
        }
    }
}

#[test]
fn native_error_winner_poison_unsupported_and_binding_mismatch_fail_before_factories() {
    let input = Interval::new(-32, 32).unwrap();
    let gamma = [1_i64];
    let mut workspace = WorkPreflightWorkspace::empty();
    let before = row_entries();
    // Shape wins before ConstructionQuote's zero-row Bounds classification.
    let bad = Construction::RmsNormCertified {
        rows: 0,
        width: 0,
        gamma_len: 1,
    }
    .bind(FactoryParameters::RmsNormCertified {
        input,
        gamma: &gamma,
    })
    .unwrap();
    assert!(matches!(
        bad.preflight(1, &mut workspace),
        Err(WorkPreflightError::Native(Error::Shape))
    ));
    assert!(workspace.is_poisoned());
    assert_eq!(row_entries(), before);
    for _ in 0..3 {
        assert!(matches!(
            equal(1, input).preflight(1, &mut workspace),
            Err(WorkPreflightError::WorkspacePoisoned)
        ));
        assert_eq!(row_entries(), before);
    }
    assert!(matches!(
        Program::rms_norm_certified(0, 0, input, &gamma),
        Err(Error::Shape)
    ));
    workspace.reset();
    let before = row_entries();
    let bad = Construction::Builder { lanes: 3 }
        .bind(FactoryParameters::Rescale {
            input: Interval::new(i64::MIN, i64::MAX).unwrap(),
            shift: 18,
        })
        .unwrap();
    assert!(matches!(
        bad.preflight(1, &mut workspace),
        Err(WorkPreflightError::Native(Error::Bounds))
    ));
    assert_eq!(row_entries(), before);
    assert!(matches!(
        Construction::GuardedSilu { lanes: 3 }.bind(FactoryParameters::EqualZero { input }),
        Err(WorkPreflightError::Unsupported)
    ));
    assert!(matches!(
        Construction::Builder { lanes: 3 }.bind(FactoryParameters::EqualZero { input }),
        Err(WorkPreflightError::ParameterBinding)
    ));
    assert!(matches!(
        Construction::RmsNormCertified {
            rows: 1,
            width: 1,
            gamma_len: 2
        }
        .bind(FactoryParameters::RmsNormCertified {
            input,
            gamma: &gamma
        }),
        Err(WorkPreflightError::ParameterBinding)
    ));
    workspace.reset();
    assert!(equal(3, input).preflight(1, &mut workspace).is_ok());
    assert_eq!(row_entries(), before);
}

#[test]
fn checked_overflow_failed_tail_and_zero_counts_never_build_or_refund_actual_prior_work() {
    let input = Interval::new(-32, 32).unwrap();
    let mut workspace = WorkPreflightWorkspace::empty();
    let before = row_entries();
    assert!(matches!(
        equal(3, input).preflight(u64::MAX, &mut workspace),
        Err(WorkPreflightError::Native(Error::Bounds))
    ));
    assert!(workspace.is_poisoned());
    workspace.reset();
    assert!(matches!(
        equal(3, input).preflight(0, &mut workspace),
        Err(WorkPreflightError::EmptySchedule)
    ));
    workspace.reset();
    assert!(matches!(
        equal(3, input).preflight_full_tail(2, equal(0, input), &mut workspace),
        Err(WorkPreflightError::Native(Error::Bounds))
    ));
    workspace.reset();
    let mut actual_total = WorkBudget::local().checked_mul(u64::MAX).unwrap();
    let prior = actual_total;
    {
        let mut ledger = CallerWorkLedger::new(&mut actual_total, u64::MAX, u64::MAX).unwrap();
        let ticket = equal(3, input).preflight(1, &mut workspace).unwrap();
        assert!(matches!(
            ledger.admit(ticket),
            Err(WorkPreflightError::Native(Error::Bounds))
        ));
        assert_eq!(ledger.total(), prior);
    }
    assert_eq!(actual_total, prior);
    assert_eq!(row_entries(), before);
    // Reset scratch cannot reset an external ledger, and successful admission
    // remains charged even if caller drops unbuilt parts (no automatic refund).
    let mut total = WorkBudget::local();
    let prior = total;
    let ticket = equal(3, input).preflight(1, &mut workspace).unwrap();
    let cost = ticket.summary().work;
    {
        let mut ledger = CallerWorkLedger::new(&mut total, u64::MAX, u64::MAX).unwrap();
        {
            let _unbuilt = ledger.admit(ticket).unwrap();
        }
        workspace.reset();
        assert_eq!(ledger.total(), prior.checked_add(cost).unwrap());
    }
    assert_eq!(total, prior.checked_add(cost).unwrap());
    assert_eq!(row_entries(), before);
}

#[test]
fn scalar_geometry_is_owned_gamma_original_owner_is_borrowed_until_native_copy() {
    let original = Interval::new(-32, 32).unwrap();
    let mut lanes = 3;
    let bound = equal(lanes, original);
    lanes = 128;
    assert_eq!(lanes, 128);
    let mut workspace = WorkPreflightWorkspace::empty();
    let ticket = bound.preflight(1, &mut workspace).unwrap();
    let mut total = WorkBudget::default();
    let admitted = {
        let mut ledger = CallerWorkLedger::new(&mut total, u64::MAX, u64::MAX).unwrap();
        ledger.admit(ticket).unwrap()
    };
    let (part, _) = admitted.into_parts();
    let actual = part.build().unwrap();
    same(actual.program(), &Program::equal_zero(3, original).unwrap());
    let mut gamma = vec![1 << 18; 3];
    let foreign = [1_i64; 3];
    let ticket = rms(1, original, &gamma)
        .preflight(1, &mut workspace)
        .unwrap();
    let admitted = {
        let mut ledger = CallerWorkLedger::new(&mut total, u64::MAX, u64::MAX).unwrap();
        ledger.admit(ticket).unwrap()
    };
    let (part, _) = admitted.into_parts();
    let actual = part.build().unwrap();
    assert!(actual.program().public_affines[0]
        .iter()
        .all(|c| c.left == 1 << 18));
    assert!(foreign.iter().all(|c| *c == 1));
    gamma.fill(0);
    drop(gamma);
    assert!(actual.program().public_affines[0]
        .iter()
        .all(|c| c.left == 1 << 18));
}

#[test]
fn explicit_workspace_quote_before_construction_prices_concrete_slots_not_authority() {
    fn construct(
        limit: usize,
        calls: &mut usize,
    ) -> Result<WorkPreflightWorkspace, WorkPreflightError> {
        let quote = WorkPreflightWorkspace::resources();
        if quote.workspace_bytes > limit {
            return Err(WorkPreflightError::Native(Error::Bounds));
        }
        *calls += 1;
        Ok(WorkPreflightWorkspace::empty())
    }
    let q = WorkPreflightWorkspace::resources();
    assert_eq!(
        q.workspace_bytes,
        std::mem::size_of::<WorkPreflightWorkspace>()
    );
    assert!(
        q.workspace_bytes
            >= RmsWorkWorkspace::workspace_bytes()
                + SoftmaxWorkWorkspace::workspace_bytes()
                + std::mem::size_of::<bool>()
    );
    assert_eq!(
        q.ticket_bytes,
        std::mem::size_of::<WorkPreflight<'static>>()
    );
    assert_eq!(
        q.admitted_part_bytes,
        std::mem::size_of::<AdmittedFactory<'static>>()
    );
    assert_eq!(
        q.ticket_result_bytes,
        std::mem::size_of::<Result<WorkPreflight<'static>, WorkPreflightError>>()
    );
    assert_eq!(
        q.build_result_bytes,
        std::mem::size_of::<Result<FactoryOutput, Error>>()
    );
    let mut calls = 0;
    let before = row_entries();
    assert!(construct(q.workspace_bytes - 1, &mut calls).is_err());
    assert_eq!(calls, 0);
    let mut workspace = construct(q.workspace_bytes, &mut calls).unwrap();
    assert_eq!(calls, 1);
    assert!(equal(3, Interval::new(0, 0).unwrap())
        .preflight(1, &mut workspace)
        .is_ok());
    assert_eq!(row_entries(), before);
}
