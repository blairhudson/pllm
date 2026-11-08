//! Canonical LN4 inventory witnesses, authored but not measured here.
//!
//! These inspect production types, not replicas of the deferred shared helper.
//! Aggregate field bounds do not establish actual source destinations, overlap,
//! unwind conformity or equivalence to the removed anonymous closures. Those
//! closure types are no longer available for measurement in this source.
use super::*;
use std::mem::align_of;

macro_rules! record_layout {
    ($record:ty; $($field:ident: $field_type:ty),+ $(,)?) => {{
        let mut field_bytes = 0usize;
        let mut field_alignment = 1usize;
        $(
            // Compile-time field-type witness against the real production record.
            let _field_type = |record: &$record| {
                let _: &$field_type = &record.$field;
            };
            field_bytes = field_bytes.checked_add(size_of::<$field_type>()).unwrap();
            field_alignment = field_alignment.max(align_of::<$field_type>());
        )+
        let bytes = size_of::<$record>();
        let alignment = align_of::<$record>();
        assert!(bytes >= field_bytes);
        assert_eq!(alignment, field_alignment);
        assert_eq!(bytes % alignment, 0);
        eprintln!("{}: size={bytes}, align={alignment}", stringify!($record));
    }};
}

fn canonical_inventory<C: AllocationContext>() -> usize {
    eprintln!(
        "{}: size={}, align={}",
        std::any::type_name::<C>(),
        size_of::<C>(),
        align_of::<C>(),
    );
    record_layout!(LeaseFor<C>; budget: C, bytes: usize);
    record_layout!(SquareWitness<'_>; d: &u128);
    record_layout!(SquareFrame;
        argument: u128, first: Option<u128>, extracted: u128,
        product: Option<u128>, mapped: Result<u128, Error>);
    record_layout!(AddFrame;
        a: usize, b: usize, option: Option<usize>, returned: Result<usize, Error>);
    record_layout!(CoreQuoteFrame;
        payload_option: Option<usize>, payload_result: Result<usize, Error>,
        terms: [usize; 5], iterator: std::array::IntoIter<usize, 5>,
        callback: AddChecked, accumulator: usize, item: usize,
        callback_frame: AddFrame, returned: Result<usize, Error>);
    record_layout!(CoreNamedControls<'_>;
        square: SquareWitness<'_>, lower: SquareFrame, upper: SquareFrame,
        quote: CoreQuoteFrame);
    record_layout!(AllocationControls;
        capacity: usize, permission: Result<(), Error>, reserved: ReserveResult,
        mapper: ReserveErrorMap, mapped: Result<(), Error>,
        exact_capacity_argument: usize, observed_capacity: usize,
        returned: AllocationResult);
    record_layout!(DenominatorControls;
        m: usize, e: usize, shifted: Option<u128>, returned: Result<u128, Error>);
    record_layout!(EntryFormals; m: usize, e: usize);
    record_layout!(CertifyFormals; value: u64, d: u128);
    record_layout!(ValueHashControls<'_>; iterator: core::slice::Iter<'_, u64>);
    record_layout!(FillControls;
        outer: core::ops::Range<usize>, inner: core::ops::Range<usize>,
        m: usize, e: usize, entry_result: Result<u64, Error>);
    record_layout!(SearchControls;
        iterator: core::ops::Range<i32>, denominator: u128, quotient: u128,
        low: u64, high: u64, middle: u64, certificate_result: Result<(), Error>);
    record_layout!(CertifiedTableFor<C>;
        values: Zeroizing<Vec<u64>>, digest: [u8; 32], lease: LeaseFor<C>);
    record_layout!(Controls<'_, '_, C>;
        value: Option<CertifiedTableFor<C>>,
        result: Result<CertifiedTableFor<C>, Error>, words: [u128; 8],
        hash: Sha256, word: [u8; 8], vector: Vec<u64>,
        digest: sha2::digest::Output<Sha256>, lease: Option<LeaseFor<C>>,
        named: CoreNamedControls<'_>, allocation: AllocationControls,
        denominator_call: DenominatorControls, entry_formals: EntryFormals,
        certify_formals: CertifyFormals, value_hash: ValueHashControls<'_>,
        fill: FillControls, search: SearchControls, constructed_capacity: usize);

    // Actual fourth operand, not a sixth summand or an old-byte-quote invariant.
    // Independent explicit additions check the production callback/fold result.
    let payload = 8usize.checked_mul(ENTRIES).unwrap();
    let allocator = payload.checked_add(64).unwrap();
    let owner = allocator.checked_add(size_of::<CertifiedTableFor<C>>()).unwrap();
    let controls = owner.checked_add(size_of::<Controls<'_, '_, C>>()).unwrap();
    let expected = controls.checked_add(C::TRANSIENT_LINK_BYTES).unwrap();
    assert_eq!(reservation_bytes::<C>().unwrap(), expected);
    eprintln!(
        "{}: five-term quote={expected}, transient link={}",
        std::any::type_name::<C>(),
        C::TRANSIENT_LINK_BYTES,
    );
    expected
}

#[test]
fn borrowed_canonical_layout_and_five_term_quote() {
    assert_eq!(<&Budget as AllocationContext>::TRANSIENT_LINK_BYTES, 0);
    canonical_inventory::<&Budget>();
}

#[test]
fn owned_canonical_layout_and_five_term_quote() {
    assert_eq!(
        OwnedBudgetContext::TRANSIENT_LINK_BYTES,
        size_of::<OwnedBudgetContext>(),
    );
    canonical_inventory::<OwnedBudgetContext>();
}
