use super::*;
use crate::replicated_decoder::prepared_tensor_map::native_attempts;

fn input() -> (Builder, Layout) {
    let mut builder = Builder::new(
        WeightMode::Public,
        &[Register {
            lanes: 129,
            bounds: Interval::new(-(1 << 18), 1 << 18).unwrap(),
        }],
    )
    .unwrap();
    let input = builder.input(0).unwrap();
    (builder, input)
}

#[test]
fn body_activation_preparation_keeps_actual_published_work_and_publication_debits_once() {
    let (mut builder, input) = input();
    // Addition is deliberately zero online work. Publish a genuine full/tail
    // product instead; do not manufacture a prior ledger or change local costs.
    let _earlier = builder.binary(&input, &input, true).unwrap();
    let prior = builder.work;
    assert_eq!(
        prior,
        WorkBudget::product(128, true)
            .unwrap()
            .checked_add(WorkBudget::product(1, true).unwrap())
            .unwrap()
    );
    assert!(prior.operations() > 0);
    let before = native_attempts();
    let prepared = builder.prepare_guarded_silu_map(&input, 0).unwrap();
    assert_eq!(native_attempts(), before + 2);
    let scheduled = prepared.work_budget().unwrap();
    // Preparation checks, but does not reserve/refund/debit a future Op. The
    // actual publication performs the original single debit and rechecks caps.
    assert_eq!(builder.work, prior);
    let _output = builder.prepared_tensor_map(&input, prepared, 0).unwrap();
    assert_eq!(builder.work, prior.checked_add(scheduled).unwrap());
}

#[test]
fn body_activation_real_prepare_route_refuses_exhausted_numeric_ledger_before_native_entry() {
    let (mut builder, input) = input();
    // Test-private numeric exhaustion control, NOT a fabricated certificate or
    // source/input authority. No runtime/PRSS or protected proof is installed.
    builder.work = WorkBudget::local()
        .checked_mul(puma::MAX_PROGRAM_OPERATIONS)
        .unwrap();
    let prior = builder.work;
    let before = native_attempts();
    assert!(matches!(
        builder.prepare_guarded_silu_map(&input, 0),
        Err(Error::Bounds)
    ));
    assert_eq!(native_attempts(), before);
    assert_eq!(builder.work, prior);
    assert!(builder.proof_admission_retired);
    assert!(matches!(builder.input(0), Err(Error::Binding)));
}
