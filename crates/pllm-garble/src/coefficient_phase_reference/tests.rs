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
