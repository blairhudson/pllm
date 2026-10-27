use super::*;
use phase::{CryptoError, MAX_WORK, Wide};

#[test]
fn original_counter_retained_through_clear_output_drop_and_short_refusal() {
    let budget = OwnedBudget::new(phase::MAX_BYTES).unwrap();
    let baseline = budget.budget().used_bytes();
    let lhs = [vec![1; 195], vec![u64::MAX; 195]];
    let rhs = [vec![3; 8385], vec![1 << 63; 8385]];
    let masks = [
        vec![Wide::default(); phase::DEGREE],
        vec![Wide::from(1); phase::DEGREE],
    ];
    let input = || Supplied {
        lhs: &lhs,
        rhs: &rhs,
        masks: &masks,
        crypto_error: CryptoError::ExplicitZero,
    };
    let mut work = Work::new(MAX_WORK).unwrap();
    let result = evaluate(&budget, &mut work, input()).unwrap();
    assert!(std::ptr::eq(result.budget(), budget.budget()));
    assert_eq!(
        budget.budget().used_bytes(),
        baseline + result.charged_bytes()
    );
    assert_eq!(result.charged_bytes(), reservation_bytes().unwrap());
    assert_eq!(result.payload().matrices().len(), phase::MATRIX_WORDS);
    assert_eq!(result.payload().scratch_words(), phase::SCRATCH_WORDS);
    // Late failure cannot refund previous output or renew cumulative work.
    let retained = budget.budget().used_bytes();
    assert!(evaluate(&budget, &mut work, input()).is_err());
    assert_eq!(budget.budget().used_bytes(), retained);
    drop(result);
    assert_eq!(budget.budget().used_bytes(), baseline);
    let required = reservation_bytes().unwrap();
    let wanted_remaining = required.checked_sub(1).unwrap();
    let blocker_charge = phase::MAX_BYTES
        .checked_sub(baseline)
        .unwrap()
        .checked_sub(wanted_remaining)
        .unwrap();
    let blocker_payload = blocker_charge
        .checked_sub(size_of::<shared_allocation::OwnedExternalCharge>())
        .unwrap();
    let mut too_short = budget.charge_external(blocker_payload).unwrap();
    let mut fresh_work = Work::new(MAX_WORK).unwrap();
    assert_eq!(
        phase::MAX_BYTES - budget.budget().used_bytes(),
        required - 1
    );
    assert_eq!(fresh_work.used(), 0);
    assert!(evaluate(&budget, &mut fresh_work, input()).is_err());
    // Reservation failure before arithmetic or payload factories.
    assert_eq!(fresh_work.used(), 0);
    assert_eq!(
        phase::MAX_BYTES - budget.budget().used_bytes(),
        required - 1
    );
    too_short.resize_payload(0).unwrap();
}

#[test]
fn unknown_error_and_noncanonical_mask_refuse_before_payload_factory() {
    let budget = OwnedBudget::new(phase::MAX_BYTES).unwrap();
    let baseline = budget.budget().used_bytes();
    let lhs = [vec![0; 195], vec![0; 195]];
    let rhs = [vec![0; 8385], vec![0; 8385]];
    let mut masks = [
        vec![Wide::default(); phase::DEGREE],
        vec![Wide::default(); phase::DEGREE],
    ];
    masks[0][phase::DEGREE - 1] = phase::Rns3::new(phase::LIMBS).unwrap().modulus();
    // Existing allocation failure hook proves no payload factory was reached.
    budget
        .budget()
        .fail_after
        .store(0, std::sync::atomic::Ordering::SeqCst);
    let mut work = Work::new(MAX_WORK).unwrap();
    let input = |crypto_error| Supplied {
        lhs: &lhs,
        rhs: &rhs,
        masks: &masks,
        crypto_error,
    };
    assert!(evaluate(&budget, &mut work, input(CryptoError::Unknown)).is_err());
    assert_eq!(work.used(), 0);
    assert!(evaluate(&budget, &mut work, input(CryptoError::ExplicitZero)).is_err());
    assert_eq!(budget.budget().used_bytes(), baseline);
}

#[test]
fn exact_work_and_remaining_bytes_retain_original_charge() {
    let q = phase::Quote::qualification().unwrap();
    let required = reservation_bytes().unwrap();
    let budget = OwnedBudget::new(phase::MAX_BYTES).unwrap();
    let bootstrap = budget.budget().used_bytes();
    let lhs = [vec![1; 195], vec![u64::MAX; 195]];
    let rhs = [vec![3; 8385], vec![1 << 63; 8385]];
    let masks = [
        vec![Wide::default(); phase::DEGREE],
        vec![Wide::from(1); phase::DEGREE],
    ];
    // Supplied backing is included in the SAME reservation, not a second ledger.
    let source_bytes = lhs
        .iter()
        .chain(rhs.iter())
        .try_fold(0usize, |sum, v| {
            assert_eq!(v.capacity(), v.len());
            sum.checked_add(v.capacity().checked_mul(size_of::<u64>())?)
        })
        .unwrap();
    let mask_bytes = masks
        .iter()
        .try_fold(0usize, |sum, v| {
            assert_eq!(v.capacity(), v.len());
            sum.checked_add(v.capacity().checked_mul(size_of::<Wide>())?)
        })
        .unwrap();
    assert_eq!(q.source_bytes(), source_bytes);
    assert_eq!(q.mask_bytes(), mask_bytes);
    assert!(q.scratch_bytes() > 0);
    assert!(q.control_bytes() > 0);
    // R covers source, scratch, retained outputs, control AND inline wrapper/guard.
    // This quote-derived boundary check is not independent quote-completeness proof.
    assert_eq!(
        required,
        q.total_bytes()
            .checked_add(size_of::<OwnedCharged<ClearRegion<'_>>>())
            .unwrap()
    );
    let blocker_charge = phase::MAX_BYTES
        .checked_sub(bootstrap)
        .unwrap()
        .checked_sub(required)
        .unwrap();
    let blocker_payload = blocker_charge
        .checked_sub(size_of::<shared_allocation::OwnedExternalCharge>())
        .unwrap();
    let blocker = budget.charge_external(blocker_payload).unwrap();
    assert_eq!(blocker.charged_bytes(), blocker_charge);
    let baseline = budget.budget().used_bytes();
    assert_eq!(baseline, bootstrap.checked_add(blocker_charge).unwrap());
    assert_eq!(phase::MAX_BYTES.checked_sub(baseline).unwrap(), required);
    let mut work = Work::new(q.arithmetic()).unwrap();
    assert_eq!(work.used(), 0);

    let result = evaluate(
        &budget,
        &mut work,
        Supplied {
            lhs: &lhs,
            rhs: &rhs,
            masks: &masks,
            crypto_error: CryptoError::ExplicitZero,
        },
    )
    .unwrap();
    assert!(std::ptr::eq(result.budget(), budget.budget()));
    assert_eq!(work.used(), q.arithmetic());
    assert_eq!(result.charged_bytes(), required);
    assert_eq!(
        budget.budget().used_bytes(),
        baseline.checked_add(required).unwrap()
    );
    assert_eq!(budget.budget().used_bytes(), phase::MAX_BYTES);
    assert_eq!(result.payload().scratch_words(), phase::SCRATCH_WORDS);
    assert_eq!(result.payload().matrices().len(), phase::MATRIX_WORDS);
    drop(result);
    assert_eq!(budget.budget().used_bytes(), baseline);
    assert_eq!(work.used(), q.arithmetic()); // Drop never refunds original work.
    drop(blocker);
    assert_eq!(budget.budget().used_bytes(), bootstrap);
}

#[test]
fn factory_failures_preserve_spent_work_and_original_byte_baseline() {
    use std::sync::atomic::Ordering::SeqCst;

    for ordinal in 0..4 {
        let q = phase::Quote::qualification().unwrap();
        let required = reservation_bytes().unwrap();
        let budget = OwnedBudget::new(phase::MAX_BYTES).unwrap();
        let bootstrap = budget.budget().used_bytes();
        let lhs = [vec![1; 195], vec![u64::MAX; 195]];
        let rhs = [vec![3; 8385], vec![1 << 63; 8385]];
        let masks = [
            vec![Wide::default(); phase::DEGREE],
            vec![Wide::from(1); phase::DEGREE],
        ];
        let input = || Supplied {
            lhs: &lhs,
            rhs: &rhs,
            masks: &masks,
            crypto_error: CryptoError::ExplicitZero,
        };
        input().validate().unwrap();
        let source_bytes = lhs
            .iter()
            .chain(rhs.iter())
            .try_fold(0usize, |sum, v| {
                sum.checked_add(v.capacity().checked_mul(size_of::<u64>())?)
            })
            .unwrap();
        let mask_bytes = masks
            .iter()
            .try_fold(0usize, |sum, v| {
                sum.checked_add(v.capacity().checked_mul(size_of::<Wide>())?)
            })
            .unwrap();
        assert_eq!(q.source_bytes(), source_bytes);
        assert_eq!(q.mask_bytes(), mask_bytes);
        let blocker_charge = phase::MAX_BYTES
            .checked_sub(bootstrap)
            .unwrap()
            .checked_sub(required)
            .unwrap();
        let blocker_payload = blocker_charge
            .checked_sub(size_of::<shared_allocation::OwnedExternalCharge>())
            .unwrap();
        let blocker = budget.charge_external(blocker_payload).unwrap();
        assert_eq!(blocker.charged_bytes(), blocker_charge);
        let baseline = budget.budget().used_bytes();
        assert_eq!(baseline, bootstrap.checked_add(blocker_charge).unwrap());
        assert_eq!(phase::MAX_BYTES.checked_sub(baseline).unwrap(), required);
        let mut work = Work::new(q.arithmetic()).unwrap();
        assert_eq!(work.used(), 0);

        // Source order in evaluate: scratch -> phases -> converted -> matrices.
        // Each zeroed calls shared_allocation::allocate -> Budget::allocation once;
        // resize uses the admitted capacity. Hook armed AFTER all setup above.
        budget.budget().fail_after.store(ordinal, SeqCst);
        assert!(matches!(
            evaluate(&budget, &mut work, input()),
            Err(shared_allocation::Error::Allocation)
        ));
        assert_eq!(budget.budget().fail_after.load(SeqCst), 0);
        assert_eq!(budget.budget().used_bytes(), baseline);
        assert_eq!(work.used(), q.arithmetic());

        // SAME original budget/work: exhausted admission cannot reach a factory.
        budget.budget().fail_after.store(1, SeqCst);
        assert!(matches!(
            evaluate(&budget, &mut work, input()),
            Err(shared_allocation::Error::Resource)
        ));
        assert_eq!(budget.budget().fail_after.load(SeqCst), 1);
        assert_eq!(budget.budget().used_bytes(), baseline);
        assert_eq!(work.used(), q.arithmetic());
        // Baseline proves reservation release, NOT independent payload-drop order.
        drop(blocker);
        assert_eq!(budget.budget().used_bytes(), bootstrap);
    }
}
