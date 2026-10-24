use super::super::numeric::ONE;
use super::*;

fn factory(lanes: usize, input: Interval) -> ClosedFactory<'static> {
    Construction::GuardedSilu { lanes }
        .bind(FactoryParameters::GuardedSilu { input })
        .unwrap()
}
fn schedule(space: &mut WorkPreflightWorkspace) -> WorkPreflight<'static> {
    let input = Interval::new(-ONE, ONE).unwrap();
    factory(128, input)
        .preflight_full_tail(3, factory(3, input), space)
        .unwrap()
}

#[test]
fn preflight_never_enters_factory_and_matches_original_guarded_silu() {
    let mut space = WorkPreflightWorkspace::empty();
    for lanes in [1, 3, 64, 127, 128] {
        for (lo, hi) in [
            (-ONE, ONE),
            (0, 0),
            (-10 * ONE, -9 * ONE),
            (9 * ONE, 10 * ONE),
        ] {
            let input = Interval::new(lo, hi).unwrap();
            let before = activation_work::entries();
            let ticket = factory(lanes, input).preflight(1, &mut space).unwrap();
            assert_eq!(activation_work::entries(), before);
            let summary = ticket.summary();
            let mut total = WorkBudget::default();
            let admitted = CallerWorkLedger::new(&mut total, u64::MAX, u64::MAX)
                .unwrap()
                .admit(ticket)
                .unwrap();
            let (part, tail) = admitted.into_parts();
            assert!(tail.is_none());
            let result = part.build().unwrap();
            let mut entered = before;
            entered[3] += 1;
            assert_eq!(activation_work::entries(), entered);
            assert_eq!(result.certificate(), ProducedCertificate::None);
            // Independent old public constructor, not the new ticket path.
            let original = Program::activation_guarded(Activation::Silu, lanes, input).unwrap();
            assert_eq!(summary.work, original.work_budget());
            assert_eq!(result.program().digest(), original.digest());
            assert_eq!(result.program().output_bounds(), original.output_bounds());
            assert_eq!(result.program().validity_lanes(), original.validity_lanes());
            assert_eq!(total, summary.work);
        }
    }
}

#[test]
fn full_tail_repeated_schedule_prices_each_invocation_not_merged_lanes() {
    let mut space = WorkPreflightWorkspace::empty();
    let before = activation_work::entries();
    let ticket = schedule(&mut space);
    assert_eq!(activation_work::entries(), before);
    let summary = ticket.summary();
    let input = Interval::new(-ONE, ONE).unwrap();
    let original_full = Program::activation_guarded(Activation::Silu, 128, input).unwrap();
    let original_tail = Program::activation_guarded(Activation::Silu, 3, input).unwrap();
    let expected = original_full
        .work_budget()
        .checked_mul(3)
        .unwrap()
        .checked_add(original_tail.work_budget())
        .unwrap();
    assert_eq!(summary.work, expected);
    assert_eq!(summary.full.invocations, 3);
    assert_eq!(summary.tail.unwrap().invocations, 1);
    let mut total = WorkBudget::default();
    let (full, tail) = CallerWorkLedger::new(
        &mut total,
        expected.operations(),
        expected.cursors().aes_work,
    )
    .unwrap()
    .admit(ticket)
    .unwrap()
    .into_parts();
    assert_eq!(
        full.build().unwrap().program().digest(),
        original_full.digest()
    );
    assert_eq!(
        tail.unwrap().build().unwrap().program().digest(),
        original_tail.digest()
    );
    assert_eq!(total, expected);
}

#[test]
fn prior_exact_one_short_and_dropped_admission_never_refund_or_enter_factory() {
    let mut space = WorkPreflightWorkspace::empty();
    let prior = WorkBudget::product(3, false).unwrap();
    let pending = schedule(&mut space).summary().work;
    let expected = prior.checked_add(pending).unwrap();
    let before = activation_work::entries();
    for (ops, aes) in [
        (expected.operations() - 1, expected.cursors().aes_work),
        (expected.operations(), expected.cursors().aes_work - 1),
    ] {
        let mut total = prior;
        let mut ledger = CallerWorkLedger::new(&mut total, ops, aes).unwrap();
        assert!(matches!(
            ledger.admit(schedule(&mut space)),
            Err(WorkPreflightError::Native(Error::Bounds))
        ));
        assert_eq!(total, prior);
        assert_eq!(activation_work::entries(), before);
    }
    let mut total = prior;
    {
        let _abandoned = CallerWorkLedger::new(
            &mut total,
            expected.operations(),
            expected.cursors().aes_work,
        )
        .unwrap()
        .admit(schedule(&mut space))
        .unwrap();
    }
    space.reset();
    assert_eq!(total, expected);
    assert_eq!(activation_work::entries(), before);
}

#[test]
fn original_validation_winners_poison_and_reset_and_wrong_pair_refuses() {
    let good = Interval::new(-ONE, ONE).unwrap();
    assert!(matches!(
        Construction::Negative { lanes: 1 }.bind(FactoryParameters::GuardedSilu { input: good }),
        Err(WorkPreflightError::ParameterBinding)
    ));
    assert!(matches!(
        Construction::GuardedSilu { lanes: 1 }.bind(FactoryParameters::Negative { input: good }),
        Err(WorkPreflightError::Unsupported)
    ));
    let mut space = WorkPreflightWorkspace::empty();
    for (lanes, input) in [
        (0, good),
        (129, good),
        (usize::MAX, Interval::new(i64::MIN, i64::MAX).unwrap()),
        (1, Interval::new(i64::MIN, i64::MAX).unwrap()),
    ] {
        space.reset();
        let before = activation_work::entries();
        let result = factory(lanes, input).preflight(0, &mut space);
        assert_eq!(activation_work::entries(), before);
        let original = Program::activation_guarded(Activation::Silu, lanes, input)
            .err()
            .unwrap();
        assert!(matches!(result,Err(WorkPreflightError::Native(e)) if e==original));
        assert!(space.is_poisoned());
        assert!(matches!(
            factory(1, good).preflight(1, &mut space),
            Err(WorkPreflightError::WorkspacePoisoned)
        ));
        space.reset();
        assert!(factory(1, good).preflight(1, &mut space).is_ok());
    }
    assert!(matches!(
        factory(1, good).preflight(0, &mut space),
        Err(WorkPreflightError::EmptySchedule)
    ));
    space.reset();
    assert!(matches!(
        factory(128, good).preflight(u64::MAX, &mut space),
        Err(WorkPreflightError::Native(Error::Bounds))
    ));
    assert!(space.is_poisoned());
}

#[test]
fn concrete_workspace_contains_actual_activation_store_and_quote_precedes_creation() {
    #[repr(C)]
    struct Literal {
        rms: RmsWorkWorkspace,
        softmax: SoftmaxWorkWorkspace,
        activation: ActivationWorkWorkspace,
        poisoned: bool,
    }
    let quote = WorkPreflightWorkspace::resources();
    assert_eq!(quote.workspace_bytes, std::mem::size_of::<Literal>());
    let mut entries = 0;
    let mut create = |limit| {
        if quote.workspace_bytes > limit {
            return Err(Error::Bounds);
        }
        entries += 1;
        Ok(WorkPreflightWorkspace::empty())
    };
    assert!(create(quote.workspace_bytes - 1).is_err());
    let _workspace = create(quote.workspace_bytes).unwrap();
    assert_eq!(entries, 1);
}
