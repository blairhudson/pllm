use super::*;

// Literal original factory. Expected instructions/slots/digest never call the
// shared negative recipe or work visitor.
fn original(lanes: usize, input: Interval) -> Result<Program, Error> {
    let mut b = Builder::new(lanes, input)?;
    let bit = b.sign(b.input())?;
    let output = b.from_bit(bit)?;
    b.finish(output)
}

fn same(actual: &Program, expected: &Program) {
    assert_eq!(actual.nodes, expected.nodes);
    assert_eq!(actual.digest(), expected.digest());
    assert_eq!(actual.numeric_profile(), expected.numeric_profile());
    assert_eq!(actual.work_budget(), expected.work_budget());
    assert_eq!(actual.bit_widths, expected.bit_widths);
    assert_eq!(actual.last_use, expected.last_use);
    assert_eq!(actual.slots, expected.slots);
    assert_eq!(actual.register_count, expected.register_count);
    assert_eq!(actual.checked_validity, expected.checked_validity);
    assert_eq!(actual.public_affines, expected.public_affines);
}

fn loop_work(lanes: usize, input: Interval) -> Result<WorkBudget, Error> {
    if lanes == 0 {
        return Err(Error::Bounds);
    }
    let width = lanes.min(MAX_TILE_LANES);
    let mut work = Program::negative_work(width, input)?
        .checked_mul(u64::try_from(lanes / width).map_err(|_| Error::Bounds)?)?;
    if lanes % width != 0 {
        work = work.checked_add(Program::negative_work(lanes % width, input)?)?;
    }
    Ok(work)
}

// Ordinary numeric test consumer, not a new production ledger/Budget permit.
fn admitted_loop(
    lanes: usize,
    input: Interval,
    prior: WorkBudget,
    caps: (u64, u64),
) -> Result<(Program, Option<Program>), Error> {
    loop_work(lanes, input)?.admit(prior.operations(), prior.cursors(), caps.0, caps.1)?;
    let width = lanes.min(MAX_TILE_LANES);
    let full = Program::negative(width, input)?;
    let tail = if lanes % width == 0 {
        None
    } else {
        Some(Program::negative(lanes % width, input)?)
    };
    Ok((full, tail))
}

#[test]
fn negative_work_matches_original_all_tile_widths_domains_and_native_records() {
    for lanes in 1..=MAX_TILE_LANES {
        for (lower, upper) in [
            (i64::MIN, i64::MAX),
            (i64::MIN, i64::MIN),
            (i64::MIN, -1),
            (-1024, 1024),
            (-1, -1),
            (-1, 0),
            (0, 0),
            (0, i64::MAX),
            (i64::MAX, i64::MAX),
        ] {
            let input = Interval::new(lower, upper).unwrap();
            let attempts = negative_work::factory_attempts();
            let rows = work_preflight::row_entries();
            let work = Program::negative_work(lanes, input).unwrap();
            assert_eq!(negative_work::factory_attempts(), attempts);
            assert_eq!(work_preflight::row_entries(), rows);
            let expected = original(lanes, input).unwrap();
            let actual = Program::negative(lanes, input).unwrap();
            same(&actual, &expected);
            assert_eq!(work, expected.work_budget());
            assert_eq!(work.cursors(), expected.work_budget().cursors());
            // Constant-sign/zero domains are NOT a public-constant shortcut.
            assert_eq!(actual.len, 3);
            assert!(matches!(
                actual.nodes[1].unwrap().instruction,
                Instruction::Sign(Value(0))
            ));
            assert!(matches!(
                actual.nodes[2].unwrap().instruction,
                Instruction::FromBit(Value(1))
            ));
        }
    }
}

#[test]
fn negative_work_full_tail_preserves_independent_prss_fill_rounding() {
    let input = Interval::new(-1024, 1024).unwrap();
    for lanes in [1, 2, 7, 127, 128, 129, 255, 256, 257, 385, 1025] {
        let attempts = negative_work::factory_attempts();
        let work = loop_work(lanes, input).unwrap();
        assert_eq!(negative_work::factory_attempts(), attempts);
        let width = lanes.min(MAX_TILE_LANES);
        let full = original(width, input).unwrap();
        let mut expected = WorkBudget::default();
        for _ in 0..lanes / width {
            expected = expected.checked_add(full.work_budget()).unwrap();
        }
        if lanes % width != 0 {
            expected = expected
                .checked_add(original(lanes % width, input).unwrap().work_budget())
                .unwrap();
        }
        assert_eq!(work, expected);
        assert_eq!(work.cursors(), expected.cursors());
    }
    assert_ne!(
        Program::negative_work(1, input)
            .unwrap()
            .checked_mul(2)
            .unwrap()
            .cursors(),
        Program::negative_work(2, input).unwrap().cursors()
    );
}

#[test]
fn negative_work_geometry_errors_match_original_and_no_partial_state_survives() {
    assert!(matches!(Interval::new(1, 0), Err(Error::Bounds)));
    for lanes in [0, MAX_TILE_LANES + 1, usize::MAX] {
        for input in [
            Interval::new(i64::MIN, i64::MAX).unwrap(),
            Interval::new(0, 0).unwrap(),
        ] {
            let attempts = negative_work::factory_attempts();
            let rows = work_preflight::row_entries();
            let error = Program::negative_work(lanes, input).unwrap_err();
            assert_eq!(negative_work::factory_attempts(), attempts);
            assert_eq!(work_preflight::row_entries(), rows);
            assert_eq!(original(lanes, input).err(), Some(error));
            assert_eq!(Program::negative(lanes, input).err(), Some(error));
            let next = Interval::new(-1, 1).unwrap();
            assert_eq!(
                Program::negative_work(1, next).unwrap(),
                original(1, next).unwrap().work_budget()
            );
        }
    }
}

#[test]
fn negative_work_zero_short_prior_and_overflow_refuse_before_real_factory_entry() {
    let input = Interval::new(-1024, 1024).unwrap();
    for lanes in [1, 127, 128, 129, 257] {
        let prior = WorkBudget::local().checked_mul(3).unwrap();
        let pending = loop_work(lanes, input).unwrap();
        let total = prior.checked_add(pending).unwrap();
        for caps in [
            (0, 0),
            (total.operations() - 1, total.cursors().aes_work),
            (total.operations(), total.cursors().aes_work - 1),
        ] {
            let attempts = negative_work::factory_attempts();
            let rows = work_preflight::row_entries();
            assert!(matches!(
                admitted_loop(lanes, input, prior, caps),
                Err(Error::Bounds)
            ));
            assert_eq!(negative_work::factory_attempts(), attempts);
            assert_eq!(work_preflight::row_entries(), rows);
        }
        let (full, tail) = admitted_loop(
            lanes,
            input,
            prior,
            (total.operations(), total.cursors().aes_work),
        )
        .unwrap();
        same(&full, &original(lanes.min(MAX_TILE_LANES), input).unwrap());
        if let Some(tail) = tail {
            same(&tail, &original(lanes % MAX_TILE_LANES, input).unwrap());
        }
    }
    let attempts = negative_work::factory_attempts();
    let rows = work_preflight::row_entries();
    let pending = Program::negative_work(1, input).unwrap();
    assert!(matches!(pending.checked_mul(u64::MAX), Err(Error::Bounds)));
    let prior = WorkBudget::local().checked_mul(u64::MAX).unwrap();
    assert!(matches!(
        admitted_loop(1, input, prior, (u64::MAX, u64::MAX)),
        Err(Error::Bounds)
    ));
    assert_eq!(negative_work::factory_attempts(), attempts);
    assert_eq!(work_preflight::row_entries(), rows);
}

#[test]
fn negative_closed_ticket_binds_geometry_domain_and_debits_before_native_entry() {
    let resources = WorkPreflightWorkspace::resources();
    assert!(resources.workspace_bytes > 0); // quote BEFORE construction
    let mut workspace = WorkPreflightWorkspace::empty();
    let input = Interval::new(-1024, 1024).unwrap();
    let full = || {
        Construction::Negative { lanes: 128 }
            .bind(FactoryParameters::Negative { input })
            .unwrap()
    };
    let tail = || {
        Construction::Negative { lanes: 1 }
            .bind(FactoryParameters::Negative { input })
            .unwrap()
    };
    let before = negative_work::factory_attempts();
    let ticket = full()
        .preflight_full_tail(2, tail(), &mut workspace)
        .unwrap();
    let summary = ticket.summary();
    let expected = original(128, input)
        .unwrap()
        .work_budget()
        .checked_mul(2)
        .unwrap()
        .checked_add(original(1, input).unwrap().work_budget())
        .unwrap();
    assert_eq!(summary.work, expected);
    assert_eq!(negative_work::factory_attempts(), before);
    let prior = WorkBudget::local();
    let total = prior.checked_add(expected).unwrap();
    let mut actual_total = prior;
    assert!(CallerWorkLedger::new(
        &mut actual_total,
        total.operations() - 1,
        total.cursors().aes_work
    )
    .unwrap()
    .admit(ticket)
    .is_err());
    assert_eq!(actual_total, prior);
    assert_eq!(negative_work::factory_attempts(), before);
    let ticket = full()
        .preflight_full_tail(2, tail(), &mut workspace)
        .unwrap();
    let admitted = CallerWorkLedger::new(
        &mut actual_total,
        total.operations(),
        total.cursors().aes_work,
    )
    .unwrap()
    .admit(ticket)
    .unwrap();
    assert_eq!(actual_total, total);
    assert_eq!(negative_work::factory_attempts(), before);
    let (full, tail) = admitted.into_parts();
    let full = full.build().unwrap();
    assert_eq!(negative_work::factory_attempts(), before + 1);
    same(full.program(), &original(128, input).unwrap());
    assert_eq!(full.certificate(), ProducedCertificate::None);
    let tail = tail.unwrap().build().unwrap();
    same(tail.program(), &original(1, input).unwrap());
    assert_eq!(negative_work::factory_attempts(), before + 2);
    assert!(matches!(
        Construction::Negative { lanes: 1 }.bind(FactoryParameters::EqualZero { input }),
        Err(WorkPreflightError::ParameterBinding)
    ));
    assert!(matches!(
        Construction::Negative { lanes: 0 }
            .bind(FactoryParameters::Negative { input })
            .unwrap()
            .preflight(1, &mut workspace),
        Err(WorkPreflightError::Native(Error::Bounds))
    ));
    assert!(workspace.is_poisoned());
    assert!(matches!(
        Construction::Negative { lanes: 1 }
            .bind(FactoryParameters::Negative { input })
            .unwrap()
            .preflight(1, &mut workspace),
        Err(WorkPreflightError::WorkspacePoisoned)
    ));
    workspace.reset();
    assert!(Construction::Negative { lanes: 1 }
        .bind(FactoryParameters::Negative { input })
        .unwrap()
        .preflight(1, &mut workspace)
        .is_ok());
}

#[test]
fn negative_scalar_scratch_quote_matches_independent_typed_slot_envelope() {
    use std::mem::{align_of, size_of};
    // Independent literal slots. No Counter type or production quote fields.
    let payload = size_of::<Node>()
        + 2 * size_of::<Option<Node>>()
        + 2 * size_of::<u8>()
        + size_of::<usize>()
        + size_of::<WorkBudget>();
    let alignment = align_of::<Node>()
        .max(align_of::<Option<Node>>())
        .max(align_of::<usize>())
        .max(align_of::<WorkBudget>());
    let conservative_padding = (align_of::<Node>() - 1)
        + 2 * (align_of::<Option<Node>>() - 1)
        + (align_of::<usize>() - 1)
        + (align_of::<WorkBudget>() - 1)
        + alignment
        - 1;
    let bytes = Program::negative_work_workspace_bytes();
    assert!((payload..=payload + conservative_padding).contains(&bytes));
    let resources = Program::negative_work_resources();
    // Literal logical phase inventory, independent of production phase fields.
    // Sized reference pointees do not affect reference layout.
    let entry = size_of::<usize>()
        + size_of::<Interval>()
        + size_of::<Result<(), Error>>()
        + size_of::<Result<Node, Error>>()
        + 2 * size_of::<Result<WorkBudget, Error>>();
    let recipe =
        size_of::<&mut ()>() + 2 * size_of::<Value>() + 2 * size_of::<Result<Value, Error>>();
    let primitive = 2 * size_of::<&mut ()>()
        + 2 * size_of::<Instruction>()
        + 2 * size_of::<Interval>()
        + 2 * size_of::<bool>()
        + 2 * size_of::<u8>()
        + size_of::<Value>()
        + size_of::<Node>()
        + size_of::<(Value, Node, u8)>()
        + 2 * size_of::<Result<(Value, Node, u8), Error>>()
        + 2 * size_of::<Result<Value, Error>>();
    let validation = 2 * size_of::<&()>()
        + 3 * size_of::<usize>()
        + size_of::<Instruction>()
        + size_of::<Interval>()
        + size_of::<bool>()
        + 2 * size_of::<Value>()
        + size_of::<[Option<Value>; 2]>()
        + size_of::<Node>()
        + size_of::<u8>()
        + 2 * size_of::<Result<Node, Error>>()
        + size_of::<Result<Value, Error>>()
        + size_of::<Result<u8, Error>>()
        + size_of::<(Value, Node, u8)>()
        + size_of::<Result<(Value, Node, u8), Error>>();
    let cost = size_of::<&()>()
        + size_of::<Node>()
        + size_of::<Value>()
        + size_of::<u8>()
        + 3 * size_of::<WorkBudget>()
        + 2 * size_of::<Result<WorkBudget, Error>>();
    assert_eq!(resources.state_bytes, bytes);
    assert_eq!(resources.entry_bytes, entry);
    assert_eq!(resources.recipe_bytes, recipe);
    assert_eq!(resources.primitive_bytes, primitive);
    assert_eq!(resources.validation_bytes, validation);
    assert_eq!(resources.cost_bytes, cost);
    assert_eq!(
        resources.known_peak_bytes,
        bytes + entry + recipe + primitive + validation.max(cost)
    );
    // Explicit numeric cap before calling work traversal; not Budget authority.
    let query = |limit| -> Result<WorkBudget, Error> {
        if bytes > limit {
            return Err(Error::Bounds);
        }
        Program::negative_work(1, Interval::new(-1, 1)?)
    };
    assert!(matches!(query(bytes - 1), Err(Error::Bounds)));
    assert_eq!(
        query(bytes).unwrap(),
        original(1, Interval::new(-1, 1).unwrap())
            .unwrap()
            .work_budget()
    );
}
