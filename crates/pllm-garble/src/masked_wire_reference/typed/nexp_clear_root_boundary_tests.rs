//! Real public entry boundary under existing private allocation/counter hooks.
use super::*;
use crate::masked_wire_reference::typed::{
    clear_nexp_bootstrap_bytes, clear_nexp_from_reserved, clear_nexp_resources, ClearNexpSpec,
};

#[test]
fn public_exact_slice_check_is_before_any_factory_or_binding_issuance() {
    for adjustment in [-1isize, 1] {
        let root = OwnedBudget::new(64 << 20).unwrap();
        let source = source();
        let mut dealer = OwnedLocalDealer::new(source, [4; 32], [92; 32], &root).unwrap();
        let baseline = root.budget().used_bytes();
        let input = owner(&root, &[0], 13);
        let mut working = root.charge_external(clear_nexp_bootstrap_bytes()).unwrap();
        let declaration = spec(source, 1);
        let declaration = ClearNexpSpec::checked(
            &dealer,
            declaration.edges,
            declaration.operation,
            declaration.next,
            &working,
        )
        .unwrap();
        let quote = clear_nexp_resources(&dealer, &declaration, &input, &working).unwrap();
        working
            .resize_payload(
                quote
                    .total_reserved_bytes()
                    .checked_add_signed(adjustment)
                    .unwrap(),
            )
            .unwrap();
        let counter = dealer.counter;
        root.budget().fail_after.store(0, Ordering::SeqCst);
        assert!(
            matches!(clear_nexp_from_reserved(&mut dealer, declaration, input, working),
            Err(error) if error.native()==Some(Error::Resource))
        );
        assert_eq!(root.budget().fail_after.load(Ordering::SeqCst), 0);
        assert_eq!(dealer.counter, counter);
        assert_eq!(root.budget().used_bytes(), baseline);
        root.budget().fail_after.store(-1, Ordering::SeqCst);
    }
}

#[test]
fn public_complete_original_root_rejects_invalid_tail_before_factory_or_entropy_binding() {
    for invalid in [-1, 1i64 << 38] {
        let root = OwnedBudget::new(64 << 20).unwrap();
        let source = source();
        let mut dealer = OwnedLocalDealer::new(source, [4; 32], [92; 32], &root).unwrap();
        let baseline = root.budget().used_bytes();
        let input = owner(&root, &[0, invalid], 13);
        let mut working = root.charge_external(clear_nexp_bootstrap_bytes()).unwrap();
        let declaration = spec(source, 2);
        let declaration = ClearNexpSpec::checked(
            &dealer,
            declaration.edges,
            declaration.operation,
            declaration.next,
            &working,
        )
        .unwrap();
        let quote = clear_nexp_resources(&dealer, &declaration, &input, &working).unwrap();
        working
            .resize_payload(quote.total_reserved_bytes())
            .unwrap();
        let counter = dealer.counter;
        root.budget().fail_after.store(0, Ordering::SeqCst);
        assert!(
            matches!(clear_nexp_from_reserved(&mut dealer, declaration, input, working),
            Err(error) if error.native()==Some(Error::Domain))
        );
        assert_eq!(root.budget().fail_after.load(Ordering::SeqCst), 0);
        assert_eq!(dealer.counter, counter);
        assert_eq!(root.budget().used_bytes(), baseline);
        root.budget().fail_after.store(-1, Ordering::SeqCst);
    }
}
