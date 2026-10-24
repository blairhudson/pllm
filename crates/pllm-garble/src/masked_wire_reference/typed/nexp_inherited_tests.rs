//! Source-only ingress/resource regressions. Genuine M-X positive execution is
//! supplied by the closed outer owner's integrated tests, not forged here.
use super::super::inherited as ingress;
use super::*;

type C = OwnedBudgetContext;

#[test]
fn inherited_quote_borrows_original_scores_without_clear_root_creation() {
    let root = OwnedBudget::new(64 << 20).unwrap();
    let source = source();
    let dealer = OwnedLocalDealer::new(source, [4; 32], [92; 32], &root).unwrap();
    // Negative scores are genuine outer inputs, NOT valid clear nExp roots.
    let original = owner(&root, &[-17, 9, -2], 7);
    let guard = root.charge_external(bootstrap_bytes::<C>()).unwrap();
    let used = root.budget().used_bytes();
    let entries = ENTRY_ENCODING.with(|x| x.get().0);
    let q = ingress::resources(&dealer, &spec(source, 3), &original).unwrap();
    assert_eq!(root.budget().used_bytes(), used);
    assert_eq!(ENTRY_ENCODING.with(|x| x.get().0), entries);
    assert_eq!(dealer.counter, 0);
    assert_eq!(q.incoming_owner_bytes, original.charged_bytes());
    assert_eq!(q.child_bytes[0], 0);
    let w = q.work;
    assert_eq!(
        (
            w.input_validation_lanes,
            w.root_validation_lanes,
            w.root_initialization_words,
            w.root_encoding_word_writes,
            w.root_mask_generation_words,
            w.root_retirement_words
        ),
        (0, 0, 0, 0, 0, 0)
    );
    assert_eq!(w.input_retirement_words, 0); // still owned/retired by outer source
    assert_eq!(w.ordered_use_checks, 3 * 18);
    assert_eq!(q.peer_bytes, 1688 + 208 * 3);
    drop(guard);
}

#[test]
fn inherited_query_refuses_foreign_original_counter_without_entropy_or_binding() {
    let root = OwnedBudget::new(64 << 20).unwrap();
    let foreign = OwnedBudget::new(64 << 20).unwrap();
    let source = source();
    let dealer = OwnedLocalDealer::new(source, [4; 32], [92; 32], &root).unwrap();
    let original = owner(&foreign, &[-1, 0], 0);
    let _guard = root.charge_external(bootstrap_bytes::<C>()).unwrap();
    assert_eq!(
        ingress::resources(&dealer, &spec(source, 2), &original),
        Err(Error::Context)
    );
    assert_eq!(dealer.counter, 0);
}

#[test]
fn inherited_query_prices_actual_original_spare_guard_not_just_active_lanes() {
    let root = OwnedBudget::new(64 << 20).unwrap();
    let source = source();
    let dealer = OwnedLocalDealer::new(source, [4; 32], [92; 32], &root).unwrap();
    let original = owner(&root, &[-1, 0, 1], 31);
    let _guard = root.charge_external(bootstrap_bytes::<C>()).unwrap();
    let q = ingress::resources(&dealer, &spec(source, 2), &original).unwrap();
    assert_eq!(q.incoming_owner_bytes, original.charged_bytes());
    assert!(q.incoming_owner_bytes > 2 * size_of::<u64>());
    assert_eq!(q.work.input_retirement_words, 0);
}

#[test]
fn inherited_query_refuses_empty_original_and_exhausted_binding_namespace() {
    let root = OwnedBudget::new(64 << 20).unwrap();
    let source = source();
    let mut dealer = OwnedLocalDealer::new(source, [4; 32], [92; 32], &root).unwrap();
    let empty = owner(&root, &[], 0);
    let original = owner(&root, &[0], 0);
    let _guard = root.charge_external(bootstrap_bytes::<C>()).unwrap();
    assert!(matches!(
        ingress::resources(&dealer, &spec(source, 1), &empty),
        Err(Error::Resource)
    ));
    dealer.counter = u64::MAX - 12;
    assert!(matches!(
        ingress::resources(&dealer, &spec(source, 1), &original),
        Err(Error::Resource)
    ));
}

#[test]
fn inherited_mask_loan_cannot_upgrade_actual_clear_root_issuer() {
    let (mask, _coordinator, _roles, _q) = issue(&[0, 1]);
    assert!(matches!(ingress::loan_mask(&mask), Err(Error::Context)));
    // Wrong issuer burns its siblings; no provisional output escapes.
    assert_eq!(mask.descriptor(), Err(Error::Consumed));
}
