use super::super::numeric::ONE;
use super::*;

// Deliberately literal ORIGINAL composition, never the new shared emit/scale18.
fn original(lanes: usize, input: Interval, coefficient: i64) -> Result<Program, Error> {
    let mut builder = Builder::new(lanes, input)?;
    let operand = builder.input();
    let scaled = builder.affine(
        operand,
        None,
        Affine {
            left: coefficient,
            right: 0,
            constant: 0,
        },
    )?;
    let output = builder.rescale(scaled, 18)?;
    builder.finish(output)
}
fn factory(lanes: usize, input: Interval, coefficient: i64) -> ClosedFactory<'static> {
    Construction::Builder { lanes }
        .bind(FactoryParameters::Scale18 { input, coefficient })
        .unwrap()
}

// Independent literal map recipe; no shared rescale emitter/query.
fn original_rescale(lanes: usize, input: Interval, shift: u8) -> Result<Program, Error> {
    let mut builder = Builder::new(lanes, input)?;
    let output = builder.rescale(builder.input(), shift)?;
    builder.finish(output)
}

#[test]
fn rescale_query_matches_literal_native_metadata_work_and_full_tail_widths() {
    let mut workspace = SoftmaxWorkWorkspace::empty();
    for input in [
        Interval::new(-ONE, ONE).unwrap(),
        Interval::new(-3, 5).unwrap(),
        Interval::new(0, 0).unwrap(),
    ] {
        for lanes in [1, 3, 4, 64, 127, 128] {
            for shift in [1, 18, 31, 62] {
                let before = (entries(), work_preflight::row_entries());
                let work = Program::rescale_work(lanes, input, shift, &mut workspace).unwrap();
                assert_eq!((entries(), work_preflight::row_entries()), before);
                let expected = original_rescale(lanes, input, shift).unwrap();
                workspace.assert_native_metadata(&expected);
                let actual = Program::rescale(lanes, input, shift).unwrap();
                assert_eq!(actual.digest(), expected.digest());
                assert_eq!(actual.instructions(), 2, "input plus rescale only");
                assert_eq!(actual.output_lanes(), lanes);
                assert_eq!(actual.validity_lanes(), None);
                assert_eq!(actual.output_bounds(), expected.output_bounds());
                assert_eq!(work, expected.work_budget());
                assert_eq!(actual.work_budget(), work);
                assert_eq!(work, WorkBudget::rescale(lanes).unwrap());
            }
        }
    }
}

#[test]
fn rescale_invalid_shifts_domains_geometry_reset_without_factory_entries() {
    let mut workspace = SoftmaxWorkWorkspace::empty();
    let good = Interval::new(-ONE, ONE).unwrap();
    for (lanes, input, shift) in [
        (0, good, 18),
        (129, good, 18),
        (usize::MAX, good, 0),
        (1, good, 0),
        (1, good, 63),
        (1, good, u8::MAX),
        (1, Interval::new(i64::MIN, i64::MIN).unwrap(), 18),
        (1, Interval::new(i64::MAX, i64::MAX).unwrap(), 18),
        (1, Interval::new(i64::MIN, i64::MAX).unwrap(), 18),
    ] {
        let before = (entries(), work_preflight::row_entries());
        let query = Program::rescale_work(lanes, input, shift, &mut workspace);
        assert_eq!((entries(), work_preflight::row_entries()), before);
        let expected = original_rescale(lanes, input, shift).err().unwrap();
        assert_eq!(query, Err(expected));
        assert_eq!(Program::rescale(lanes, input, shift).err(), Some(expected));
        let before = (entries(), work_preflight::row_entries());
        let work = Program::rescale_work(3, good, 18, &mut workspace).unwrap();
        assert_eq!((entries(), work_preflight::row_entries()), before);
        let native = original_rescale(3, good, 18).unwrap();
        workspace.assert_native_metadata(&native);
        assert_eq!(work, native.work_budget());
    }
}

#[test]
fn exact_original_composition_for_binary32_attention_coefficients_and_signed_constants() {
    let mut workspace = SoftmaxWorkWorkspace::empty();
    for head_width in [1, 2, 3, 32, 64, 96, 128, 768] {
        // EXACT existing caller expression; not an ideal-real replacement or a
        // profile assertion for any particular imported model.
        let coefficient = ((1f32 / (head_width as f32).sqrt()) * ONE as f32) as i64;
        for lanes in [1, 3, 64, 127, 128] {
            let input = Interval::new(-4 * ONE, 4 * ONE).unwrap();
            let before = (entries(), work_preflight::row_entries());
            let work = Program::scale18_work(lanes, input, coefficient, &mut workspace).unwrap();
            assert_eq!((entries(), work_preflight::row_entries()), before);
            let expected = original(lanes, input, coefficient).unwrap();
            workspace.assert_native_metadata(&expected);
            let actual = Program::scale18(lanes, input, coefficient).unwrap();
            assert_eq!(actual.digest(), expected.digest());
            assert_eq!(actual.instructions(), expected.instructions());
            assert_eq!(actual.output_bounds(), expected.output_bounds());
            assert_eq!(actual.output_lanes(), lanes);
            assert_eq!(actual.validity_lanes(), None);
            assert_eq!(actual.work_budget(), work);
        }
    }
    for coefficient in [0, 1, -1, ONE, -ONE] {
        let input = Interval::new(-ONE, ONE).unwrap();
        let expected = original(3, input, coefficient).unwrap();
        let work = Program::scale18_work(3, input, coefficient, &mut workspace).unwrap();
        let actual = Program::scale18(3, input, coefficient).unwrap();
        assert_eq!(actual.digest(), expected.digest());
        assert_eq!(
            actual.instructions(),
            expected.instructions(),
            "no constant shortcut"
        );
        assert_eq!(actual.output_bounds(), expected.output_bounds());
        assert_eq!(work, expected.work_budget());
    }
}

#[test]
fn invalid_geometry_domain_and_coefficient_overflow_keep_original_error_winner() {
    let mut workspace = SoftmaxWorkWorkspace::empty();
    let good = Interval::new(-ONE, ONE).unwrap();
    for (lanes, input, coefficient) in [
        (0, good, ONE),
        (129, good, ONE),
        (usize::MAX, good, i64::MAX),
        (1, Interval::new(i64::MIN, i64::MAX).unwrap(), 2),
        (1, good, i64::MAX),
        (1, Interval::new(i64::MIN, i64::MIN).unwrap(), -1),
    ] {
        let before = (entries(), work_preflight::row_entries());
        let query = Program::scale18_work(lanes, input, coefficient, &mut workspace);
        assert_eq!((entries(), work_preflight::row_entries()), before);
        let expected = original(lanes, input, coefficient).err().unwrap();
        assert_eq!(query, Err(expected));
        assert_eq!(
            Program::scale18(lanes, input, coefficient).err(),
            Some(expected)
        );
        assert!(Program::scale18_work(1, good, ONE, &mut workspace).is_ok());
    }
}

#[test]
fn full_tail_consuming_ticket_preserves_native_parameters_and_complete_debit() {
    let input = Interval::new(-ONE, ONE).unwrap();
    let coefficient = ((1f32 / 64f32.sqrt()) * ONE as f32) as i64;
    let mut workspace = WorkPreflightWorkspace::empty();
    let before = (entries(), work_preflight::row_entries());
    let ticket = factory(128, input, coefficient)
        .preflight_full_tail(2, factory(3, input, coefficient), &mut workspace)
        .unwrap();
    assert_eq!((entries(), work_preflight::row_entries()), before);
    let summary = ticket.summary();
    let full = original(128, input, coefficient).unwrap();
    let tail = original(3, input, coefficient).unwrap();
    let pending = full
        .work_budget()
        .checked_mul(2)
        .unwrap()
        .checked_add(tail.work_budget())
        .unwrap();
    assert_eq!(summary.work, pending);
    let prior = WorkBudget::product(3, false).unwrap();
    let complete = prior.checked_add(pending).unwrap();
    let mut total = prior;
    let (first, last) = CallerWorkLedger::new(
        &mut total,
        complete.operations(),
        complete.cursors().aes_work,
    )
    .unwrap()
    .admit(ticket)
    .unwrap()
    .into_parts();
    assert_eq!(
        total, complete,
        "both invocations admitted before any build"
    );
    let first = first.build().unwrap();
    let last = last.unwrap().build().unwrap();
    assert_eq!(first.program().digest(), full.digest());
    assert_eq!(last.program().digest(), tail.digest());
    assert_eq!(first.certificate(), ProducedCertificate::None);
    assert_eq!(last.certificate(), ProducedCertificate::None);
}

#[test]
fn prior_one_short_and_abandoned_ticket_do_not_enter_factory_or_refund() {
    let input = Interval::new(-ONE, ONE).unwrap();
    let mut workspace = WorkPreflightWorkspace::empty();
    let ticket = || factory(128, input, ONE);
    let pending = ticket()
        .preflight(3, &mut workspace)
        .unwrap()
        .summary()
        .work;
    let prior = WorkBudget::product(3, false).unwrap();
    let complete = prior.checked_add(pending).unwrap();
    let before = (entries(), work_preflight::row_entries());
    for (ops, aes) in [
        (complete.operations() - 1, complete.cursors().aes_work),
        (complete.operations(), complete.cursors().aes_work - 1),
    ] {
        let mut total = prior;
        let mut ledger = CallerWorkLedger::new(&mut total, ops, aes).unwrap();
        assert!(matches!(
            ledger.admit(ticket().preflight(3, &mut workspace).unwrap()),
            Err(WorkPreflightError::Native(Error::Bounds))
        ));
        assert_eq!(total, prior);
    }
    let mut total = prior;
    {
        let _dropped = CallerWorkLedger::new(
            &mut total,
            complete.operations(),
            complete.cursors().aes_work,
        )
        .unwrap()
        .admit(ticket().preflight(3, &mut workspace).unwrap())
        .unwrap();
    }
    workspace.reset();
    assert_eq!(total, complete);
    assert_eq!((entries(), work_preflight::row_entries()), before);
}

#[test]
fn exact_pair_poison_reset_and_legacy_rescale_remain_distinct() {
    let good = Interval::new(-ONE, ONE).unwrap();
    assert!(matches!(
        Construction::Negative { lanes: 1 }.bind(FactoryParameters::Scale18 {
            input: good,
            coefficient: ONE
        }),
        Err(WorkPreflightError::ParameterBinding)
    ));
    let mut workspace = WorkPreflightWorkspace::empty();
    let before = (entries(), work_preflight::row_entries());
    assert!(matches!(
        factory(1, good, i64::MAX).preflight(1, &mut workspace),
        Err(WorkPreflightError::Native(Error::Bounds))
    ));
    assert!(workspace.is_poisoned());
    assert!(matches!(
        factory(1, good, ONE).preflight(1, &mut workspace),
        Err(WorkPreflightError::WorkspacePoisoned)
    ));
    workspace.reset();
    assert!(factory(1, good, ONE).preflight(1, &mut workspace).is_ok());
    let rescale = Construction::Builder { lanes: 1 }
        .bind(FactoryParameters::Rescale {
            input: good,
            shift: 18,
        })
        .unwrap()
        .preflight(1, &mut workspace)
        .unwrap();
    assert_eq!(rescale.summary().work, WorkBudget::rescale(1).unwrap());
    workspace.reset();
    assert!(matches!(
        factory(128, good, ONE).preflight(u64::MAX, &mut workspace),
        Err(WorkPreflightError::Native(Error::Bounds))
    ));
    assert!(workspace.is_poisoned());
    assert_eq!((entries(), work_preflight::row_entries()), before);
}
