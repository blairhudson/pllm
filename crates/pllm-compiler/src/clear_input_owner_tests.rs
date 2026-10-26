use super::*;

fn caps() -> Caps {
    Caps {
        f64_values: 16,
        u64_values: 8,
        tensors: 3,
        operations: 1,
    }
}
fn builder() -> ClearInputBuilder {
    let q = ClearInputBuilder::resources(caps()).unwrap();
    ClearInputBuilder::new(
        OwnedBudget::new(q.minimum_fresh_budget_bytes + 16384).unwrap(),
        caps(),
        MAX_WORK,
    )
    .unwrap()
}

#[test]
fn constructor_one_short_exact_and_no_higher_work_claim() {
    let c = caps();
    let q = ClearInputBuilder::resources(c).unwrap();
    assert!(matches!(
        ClearInputBuilder::new(
            OwnedBudget::new(q.minimum_fresh_budget_bytes - 1).unwrap(),
            c,
            MAX_WORK
        ),
        Err(Error::Resource)
    ));
    assert!(matches!(
        ClearInputBuilder::new(
            OwnedBudget::new(q.minimum_fresh_budget_bytes).unwrap(),
            c,
            q.construction_work - 1
        ),
        Err(Error::Work)
    ));
    let exhausted = ClearInputBuilder::new(
        OwnedBudget::new(q.minimum_fresh_budget_bytes).unwrap(),
        c,
        q.construction_work,
    )
    .unwrap()
    .finish()
    .unwrap();
    assert!(matches!(exhausted.begin_attempt(), Err(Error::Work)));
    assert_eq!(exhausted.remaining_work(), 0);
    assert_eq!(exhausted.owned.payload().attempts.get(), 1);
    let b = ClearInputBuilder::new(
        OwnedBudget::new(q.minimum_fresh_budget_bytes).unwrap(),
        c,
        MAX_WORK,
    )
    .unwrap();
    assert_eq!(b.owned.charged_bytes(), q.charged_owner_bytes);
    let before = b.owned.payload().ledger.borrow().remaining();
    let input = b.finish().unwrap();
    assert_eq!(input.remaining_work(), before);
    assert_eq!(input.consumed_work(), q.construction_work);
    assert!(matches!(
        ClearInputBuilder::new(
            OwnedBudget::new(q.minimum_fresh_budget_bytes).unwrap(),
            c,
            MAX_WORK + 1
        ),
        Err(Error::Work)
    ));
}

#[test]
fn actual_attention_scores_dto_and_typed_u64_backing() {
    let mut b = builder();
    let q = b
        .add_f64_tensor("query", &[2, 2], &[1., 2., 3., 4.])
        .unwrap();
    let k = b.add_f64_tensor("key", &[2, 2], &[4., 3., 2., 1.]).unwrap();
    let u = b
        .add_u64_tensor("coefficients", &[4], &[0, 16, 3, 4])
        .unwrap();
    let op = b
        .add_operation(
            "scores",
            ModelOperator::AttentionScores,
            None,
            &["query", "key"],
            &[1, 1, 2, 2],
        )
        .unwrap();
    b.set_attribute(op, "group_size", Attribute::Unsigned(1))
        .unwrap();
    let input = b.finish().unwrap();
    assert_eq!(&*input.f64_tensor(q).unwrap(), &[1., 2., 3., 4.]);
    assert_eq!(&*input.f64_tensor(k).unwrap(), &[4., 3., 2., 1.]);
    assert_eq!(&*input.u64_tensor(u).unwrap(), &[0, 16, 3, 4]);
    assert!(matches!(input.f64_tensor(u), Err(Error::Shape)));
    let ops = input.operations();
    assert_eq!(ops[0].operator, ModelOperator::AttentionScores);
    assert_eq!(ops[0].inputs, ["query", "key"]);
    assert_eq!(ops[0].output_shape, [1, 1, 2, 2]);
    assert_eq!(ops[0].state_kind, None);
    assert_eq!(ops[0].attributes.as_object().unwrap().len(), 1);
    assert_eq!(ops[0].attributes["group_size"], 1);
}

#[test]
fn successive_attempts_keep_construction_prior_work_and_original_charge() {
    let input = builder().finish().unwrap();
    let owner_bytes = input.charged_bytes();
    let initial = input.remaining_work();
    let used = input.owned.budget().used_bytes();
    for step in 1..=2 {
        let output = input
            .begin_attempt()
            .unwrap()
            .admit::<[f64; 4]>(100, 20, 64)
            .unwrap()
            .finish([1., 2., 3., 4.]);
        assert!(std::ptr::eq(output.original(), &input));
        assert_eq!(output.values(), &[1., 2., 3., 4.]);
        assert!(output.phase_bytes() >= 64);
        assert!(input.owned.budget().used_bytes() > used);
        drop(output);
        assert_eq!(input.charged_bytes(), owner_bytes);
        assert_eq!(input.owned.budget().used_bytes(), used);
        assert_eq!(input.remaining_work(), initial - step * (32 + 100 + 20));
    }
}

#[test]
fn transient_control_wrappers_cannot_refill_original_control() {
    let input = builder().finish().unwrap();
    {
        let (_, _prepared) = input
            .begin_attempt()
            .unwrap()
            .with_ledger(|ledger| ledger.debit(0, 100).map_err(|_| Error::Work))
            .unwrap();
    }
    let before = input.remaining_work();
    let control_before = input.owned.payload().ledger.borrow().control_remaining();
    {
        let (_, _prepared) = input
            .begin_attempt()
            .unwrap()
            .with_ledger(|ledger| {
                assert_eq!(ledger.control_remaining(), control_before);
                ledger.debit(0, 100).map_err(|_| Error::Work)
            })
            .unwrap();
    }
    assert_eq!(input.remaining_work(), before - 132);
    assert_eq!(
        input.owned.payload().ledger.borrow().control_remaining(),
        control_before - 100
    );
}

#[test]
fn failed_quote_or_reservation_burns_ticket_without_refund() {
    let input = builder().finish().unwrap();
    let before = input.remaining_work();
    let rejected: Result<((), Prepared<'_>), Error> =
        input.begin_attempt().unwrap().with_ledger(|ledger| {
            ledger.debit(19, 0).map_err(|_| Error::Work)?;
            Err(Error::Shape)
        });
    assert!(matches!(rejected, Err(Error::Shape)));
    assert_eq!(input.remaining_work(), before - 51);
    assert_eq!(input.owned.payload().attempts.get(), 1);
    assert!(matches!(
        input
            .begin_attempt()
            .unwrap()
            .admit::<[f64; 1]>(7, 0, usize::MAX),
        Err(Error::Resource)
    ));
    assert_eq!(input.remaining_work(), before - 51 - 39);
    assert_eq!(input.owned.payload().attempts.get(), 2);
    assert!(matches!(
        input
            .begin_attempt()
            .unwrap()
            .admit::<[f64; 1]>(MAX_WORK, 0, 0),
        Err(Error::Work)
    ));
    assert_eq!(input.remaining_work(), before - 51 - 39 - 32);
}

#[test]
fn retained_capacities_and_no_arbitrary_spare_adoption() {
    let mut b = builder();
    let original_ptr = b.owned.payload().data.borrow().floats.as_ptr();
    b.add_f64_tensor("x", &[2], &[1., 2.]).unwrap();
    let input = b.finish().unwrap();
    let d = input.owned.payload().data.borrow();
    assert_eq!(original_ptr, d.floats.as_ptr()); // move, not whole-backing clone
    assert_eq!(d.floats.len(), caps().f64_values);
    assert_eq!(d.floats.capacity(), caps().f64_values);
    assert_eq!(d.words.capacity(), caps().u64_values);
    assert_eq!(d.tensors.capacity(), caps().tensors);
    assert_eq!(d.ops.capacity(), caps().operations);
    assert_eq!(d.tensors[0].name.capacity(), NAME);
    assert_eq!(d.tensors[0].shape.capacity(), RANK);
    assert!(d.floats[2..].iter().all(|&v| v == 0.));
}

#[test]
fn malformed_or_nonfinite_construction_poisoned_and_numeric_cleanup() {
    for nonfinite in [false, true] {
        let mut b = builder();
        b.add_f64_tensor("good", &[2], &[3., 4.]).unwrap();
        b.add_u64_tensor("words", &[2], &[9, 10]).unwrap();
        let failure = if nonfinite {
            b.add_f64_tensor("bad", &[1], &[f64::NAN])
        } else {
            b.add_f64_tensor("bad", &[2], &[1.])
        };
        assert_eq!(
            failure,
            Err(if nonfinite {
                Error::NonFinite
            } else {
                Error::Shape
            })
        );
        assert!(b.owned.payload().poisoned.get());
        {
            let mut d = b.owned.payload().data.borrow_mut();
            assert_eq!(d.words.len(), caps().u64_values);
            assert_eq!(d.floats.len(), caps().f64_values);
            // Seed BOTH unused tails to exercise the full initialized pools.
            let float_used = d.float_used;
            let word_used = d.word_used;
            d.floats[float_used..].fill(7.25);
            d.words[word_used..].fill(19);
            d.wipe(); // same routine used by actual payload Drop before charge
            assert!(d.floats.iter().all(|&v| v == 0.));
            assert!(d.words.iter().all(|&v| v == 0));
            assert_eq!(d.floats.len(), caps().f64_values);
            assert_eq!(d.words.len(), caps().u64_values);
            assert_eq!(d.tensors[0].shape, [2]); // PUBLIC IR, not claimed wiped
        }
        assert!(matches!(b.finish(), Err(Error::Burned)));
    }
}

#[test]
fn malformed_attribute_burns_builder_and_attempt_count_is_bounded() {
    let mut b = builder();
    let op = b
        .add_operation(
            "scores",
            ModelOperator::AttentionScores,
            None,
            &["q", "k"],
            &[1, 1, 2, 2],
        )
        .unwrap();
    assert_eq!(
        b.set_attribute(op, "scale", Attribute::Real(f64::INFINITY)),
        Err(Error::NonFinite)
    );
    assert!(matches!(b.finish(), Err(Error::Burned)));
    let input = builder().finish().unwrap();
    for _ in 0..MAX_ATTEMPTS {
        let _ticket = input.begin_attempt().unwrap();
    }
    assert!(matches!(input.begin_attempt(), Err(Error::Burned)));
}

#[test]
fn cleanup_precedes_phase_release_and_original_survives_between_attempts() {
    struct Probe<'a> {
        budget: &'a pllm_garble::shared_allocation::Budget,
        baseline: usize,
        wiped: &'a Cell<bool>,
    }
    impl Zeroize for Probe<'_> {
        fn zeroize(&mut self) {
            assert!(self.budget.used_bytes() > self.baseline);
            self.wiped.set(true);
        }
    }
    let q = ClearInputBuilder::resources(caps()).unwrap();
    let root = OwnedBudget::new(q.minimum_fresh_budget_bytes + 16384).unwrap();
    let bootstrap = root.budget().used_bytes();
    let input = ClearInputBuilder::new(root.try_clone().unwrap(), caps(), MAX_WORK)
        .unwrap()
        .finish()
        .unwrap();
    let original = root.budget().used_bytes();
    assert!(original > bootstrap);
    let wiped = Cell::new(false);
    let phase = input
        .begin_attempt()
        .unwrap()
        .admit::<Probe<'_>>(64, 0, 0)
        .unwrap();
    let output = phase.finish(Probe {
        budget: root.budget(),
        baseline: original,
        wiped: &wiped,
    });
    drop(output);
    assert!(wiped.get());
    assert_eq!(root.budget().used_bytes(), original);
    drop(input);
    assert_eq!(root.budget().used_bytes(), bootstrap);
}

#[test]
fn shared_generic_owned_charged_drop_order_not_data_destructor_evidence() {
    struct Observer<'a> {
        budget: &'a pllm_garble::shared_allocation::Budget,
        baseline: usize,
        dropped: &'a Cell<bool>,
    }
    impl Drop for Observer<'_> {
        fn drop(&mut self) {
            assert!(self.budget.used_bytes() > self.baseline);
            self.dropped.set(true);
        }
    }
    let root = OwnedBudget::new(16384).unwrap();
    let baseline = root.budget().used_bytes();
    let dropped = Cell::new(false);
    let guarded = root
        .charge_external(64)
        .unwrap()
        .try_build(|_| {
            Ok(Observer {
                budget: root.budget(),
                baseline,
                dropped: &dropped,
            })
        })
        .unwrap();
    drop(guarded);
    assert!(dropped.get());
    assert_eq!(root.budget().used_bytes(), baseline);
}

#[test]
fn dropped_ticket_prepared_and_phase_burn_attempts_and_keep_original() {
    let input = builder().finish().unwrap();
    let before = input.remaining_work();
    let owner_bytes = input.charged_bytes();
    let reserved = input.owned.budget().used_bytes();
    {
        let _ticket = input.begin_attempt().unwrap();
    }
    assert_eq!(input.remaining_work(), before - 32);
    {
        let (_, _prepared) = input
            .begin_attempt()
            .unwrap()
            .with_ledger(|ledger| ledger.debit(11, 3).map_err(|_| Error::Work))
            .unwrap();
    }
    assert_eq!(input.remaining_work(), before - 64 - 14);
    let phase = input
        .begin_attempt()
        .unwrap()
        .admit::<[f64; 4]>(17, 5, 64)
        .unwrap();
    assert!(input.owned.budget().used_bytes() > reserved);
    drop(phase); // no workspace constructed; charge still releases, work stays spent
    assert_eq!(input.remaining_work(), before - 96 - 14 - 22);
    assert_eq!(input.owned.payload().attempts.get(), 3);
    assert_eq!(input.owned.budget().used_bytes(), reserved);
    assert_eq!(input.charged_bytes(), owner_bytes);
}
