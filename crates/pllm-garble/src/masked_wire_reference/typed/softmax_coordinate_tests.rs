//! Independent bridge inventory/native controls. Actual denominator-owner
//! issuance and outer sibling burn are tested by B's genuine causal-root prefix;
//! these tests never fabricate a DenominatorMask/Value from metadata.
use super::*;
use pllm_core::edge_contract::{EdgeIdentity, Shape};
use std::alloc::Layout;

fn layout(fields: &[Layout]) -> usize {
    fields
        .iter()
        .fold(Layout::new::<()>(), |a, b| a.extend(*b).unwrap().0)
        .pad_to_align()
        .size()
}

#[test]
fn independent_concrete_slots_include_originals_results_both_roles_and_output_loans() {
    type C = &'static Budget;
    let dealer = layout(&[
        Layout::new::<*mut LocalDealerFor<C>>(),
        Layout::new::<Binding>(),
        Layout::new::<Spec>(),
        Layout::new::<Resources>(),
        Layout::new::<Result<Resources, Error>>(),
        Layout::new::<LeaseFor<C>>(),
        Layout::new::<DeclarationFor<C>>(),
        Layout::new::<Result<DeclarationFor<C>, Error>>(),
        Layout::new::<DenominatorMaskFor<C>>(),
        Layout::new::<Result<NativeIssued<C>, Error>>(),
        Layout::new::<NativeIssued<C>>(),
        Layout::new::<Result<Issued<C>, Error>>(),
        Layout::new::<arithmetic::Context>(),
        Layout::new::<[Binding; 3]>(),
        Layout::new::<Result<Binding, Error>>(),
        Layout::new::<Interval>(),
        Layout::new::<Result<(), Error>>(),
        Layout::new::<Sha256>(),
        Layout::new::<[[u8; 32]; 6]>(),
        Layout::new::<[&[u8]; 5]>(),
        Layout::new::<arithmetic::Spec>(),
        Layout::new::<arithmetic::Resources>(),
        Layout::new::<Result<arithmetic::Resources, Error>>(),
        Layout::new::<&[u64]>(),
        Layout::new::<std::array::IntoIter<EdgeDescriptor, 3>>(),
        Layout::new::<EdgeDescriptor>(),
        Layout::new::<Result<Numeric, Error>>(),
        Layout::new::<Work>(),
        Layout::new::<Result<LeaseFor<C>, Error>>(),
        Layout::new::<[LeaseFor<C>; 2]>(),
        Layout::new::<[LeaseFor<C>; 2]>(),
        Layout::new::<usize>(),
        Layout::new::<usize>(),
        Layout::new::<usize>(),
        Layout::new::<usize>(),
        Layout::new::<usize>(),
        Layout::new::<usize>(),
        Layout::new::<usize>(),
        Layout::new::<usize>(),
        Layout::new::<usize>(),
        Layout::new::<usize>(),
        Layout::new::<Result<usize, Error>>(),
        Layout::new::<Result<usize, Error>>(),
        Layout::new::<Result<usize, Error>>(),
        Layout::new::<Result<usize, Error>>(),
    ]);
    let party = layout(&[
        Layout::new::<*mut PartyFor<C>>(),
        Layout::new::<DenominatorValueFor<C>>(),
        Layout::new::<Result<arithmetic::StepFor<C>, Error>>(),
        Layout::new::<arithmetic::StepFor<C>>(),
        Layout::new::<Result<arithmetic::FrameFor<C>, Error>>(),
        Layout::new::<Result<StepFor<C>, Error>>(),
        Layout::new::<OutputFor<C>>(),
        Layout::new::<StepFor<C>>(),
        Layout::new::<&[u8]>(),
        Layout::new::<&[u64]>(),
        Layout::new::<Result<(), Error>>(),
    ]);
    let mask = size_of::<MaskFor<C>>()
        + size_of::<MaskCoordinate<'_, C>>()
        + size_of::<Result<MaskCoordinate<'_, C>, Error>>();
    let output = size_of::<OutputFor<C>>()
        + size_of::<*mut OutputFor<C>>()
        + size_of::<ValueCoordinate<'_, C>>()
        + size_of::<Result<ValueCoordinate<'_, C>, Error>>();
    let native = arithmetic::resources_for_context::<C>(arithmetic::Spec {
        bits: 14,
        lanes: 1,
        gate: arithmetic::Gate::Tr { shift: 6 },
    })
    .unwrap();
    let q = resources_for::<C>().unwrap();
    assert_eq!(q.transition_bytes, dealer + 2 * party);
    assert_eq!(q.mask_wrapper_bytes, mask);
    assert_eq!(q.output_wrapper_bytes, output);
    assert_eq!(q.declaration_bytes, size_of::<DeclarationFor<C>>());
    assert_eq!(q.party_wrapper_bytes, size_of::<PartyFor<C>>());
    assert_eq!(
        q.total_bytes,
        native.combined_reservation_bytes
            + size_of::<DeclarationFor<C>>()
            + mask
            + 2 * size_of::<PartyFor<C>>()
            + 2 * output
            + dealer
            + 2 * party
    );
    assert_eq!(q.native.frames_per_party, 2);
    assert!(q.total_bytes <= crate::shared_allocation::MAX_ALLOCATION_BYTES);
    // Owned mode is priced as its actual concrete owner type, not a borrowed alias.
    let owned = resources_for::<OwnedBudgetContext>().unwrap();
    assert_eq!(
        owned.native,
        arithmetic::resources_for_context::<OwnedBudgetContext>(native_spec()).unwrap()
    );
}

fn descriptor(
    dealer: &LocalDealer<'_>,
    wire: u8,
    bits: u8,
    f: u8,
    kind: Interpretation,
) -> EdgeDescriptor {
    EdgeDescriptor::new(
        EdgeIdentity::new(dealer.source, dealer.plan, [3; 32], [wire; 32], 7),
        Shape::new(&[1]).unwrap(),
        Numeric::new(bits, f, kind).unwrap(),
        Representation::CommonAdditiveMasked,
        None,
    )
    .unwrap()
}

fn metadata(dealer: &mut LocalDealer<'_>) -> (Binding, Spec) {
    let input_edge = descriptor(dealer, 10, 14, 12, Interpretation::Unsigned);
    // A DECLARATION ONLY. This cannot construct B's sealed denominator owner.
    let input = dealer
        .binding(
            input_edge,
            [4; 32],
            Some(Interval {
                lower: 4096,
                upper: 12289,
            }),
        )
        .unwrap();
    let spec = Spec {
        operation: [4; 32],
        quotient: descriptor(dealer, 11, 8, 6, Interpretation::Unsigned),
        coordinate: descriptor(dealer, 11, 8, 0, Interpretation::Unsigned),
        next: [5; 32],
    };
    (input, spec)
}

fn dealer(budget: &Budget) -> LocalDealer<'_> {
    LocalDealer::new(
        SourceIdentity::new([1; 32], [2; 32], approved_implementation()),
        [6; 32],
        [7; 32],
        budget,
    )
    .unwrap()
}

#[test]
fn acyclic_precommit_keeps_one_issuance_mask_identity_and_both_scales() {
    let budget = Budget::new(4 << 20).unwrap();
    let mut dealer = dealer(&budget);
    let (input, spec) = metadata(&mut dealer);
    let q = resources_for::<&Budget>().unwrap();
    let before = dealer.counter;
    let reservation = LeaseFor::reserve(&&budget, q.total_bytes).unwrap();
    let declaration = declare_reserved(&mut dealer, input, spec, reservation).unwrap();
    let q6 = declaration.quotient_binding();
    let q0 = declaration.output_binding();
    assert_eq!(dealer.counter, before + 1);
    assert_eq!(q6.issuance, q0.issuance);
    assert_eq!(q6.mask_id, q0.mask_id);
    assert_eq!(q6.namespace, q0.namespace);
    assert_eq!(q6.session, q0.session);
    assert_eq!(q6.consumer, q0.consumer);
    assert_eq!(q6.descriptor.identity(), q0.descriptor.identity());
    assert_eq!(q6.descriptor.numeric().fractional_bits(), 6);
    assert_eq!(q0.descriptor.numeric().fractional_bits(), 0);
    assert_eq!(
        q0.interval,
        Some(Interval {
            lower: 64,
            upper: 193
        })
    );
    assert_ne!(q6.digest(), q0.digest());
}

#[test]
fn invalid_declared_ranges_signed_aliases_or_replacement_edges_do_not_issue() {
    let budget = Budget::new(4 << 20).unwrap();
    let mut dealer = dealer(&budget);
    let (input, spec) = metadata(&mut dealer);
    for range in [
        None,
        Some(Interval {
            lower: 0,
            upper: 12289,
        }),
        Some(Interval {
            lower: 4096,
            upper: 12290,
        }),
    ] {
        assert!(matches!(
            checked_declaration(
                &dealer,
                Binding {
                    interval: range,
                    ..input
                },
                spec
            ),
            Err(Error::Domain)
        ));
    }
    let signed = descriptor(&dealer, 10, 14, 12, Interpretation::Signed);
    assert!(matches!(
        checked_declaration(
            &dealer,
            Binding {
                descriptor: signed,
                ..input
            },
            spec
        ),
        Err(Error::Domain)
    ));
    let signed_index = descriptor(&dealer, 11, 8, 0, Interpretation::Signed);
    assert!(matches!(
        checked_declaration(
            &dealer,
            input,
            Spec {
                coordinate: signed_index,
                ..spec
            }
        ),
        Err(Error::Domain)
    ));
    let replacement = descriptor(&dealer, 12, 8, 0, Interpretation::Unsigned);
    assert!(matches!(
        checked_declaration(
            &dealer,
            input,
            Spec {
                coordinate: replacement,
                ..spec
            }
        ),
        Err(Error::Context)
    ));
    let foreign = Binding {
        namespace: [255; 32],
        ..input
    };
    assert!(matches!(
        checked_declaration(&dealer, foreign, spec),
        Err(Error::Context)
    ));
}

#[test]
fn exact_short_foreign_reservation_and_counter_overflow_precede_binding_issue() {
    let budget = Budget::new(4 << 20).unwrap();
    let foreign = Budget::new(4 << 20).unwrap();
    let mut dealer = dealer(&budget);
    let (input, spec) = metadata(&mut dealer);
    let q = resources_for::<&Budget>().unwrap();
    let before = dealer.counter;
    for lease in [
        LeaseFor::reserve(&&budget, q.total_bytes - 1).unwrap(),
        LeaseFor::reserve(&&foreign, q.total_bytes).unwrap(),
    ] {
        assert!(matches!(
            declare_reserved(&mut dealer, input, spec, lease),
            Err(Error::Resource)
        ));
        assert_eq!(dealer.counter, before);
    }
    dealer.counter = u64::MAX;
    let lease = LeaseFor::reserve(&&budget, q.total_bytes).unwrap();
    assert!(matches!(
        declare_reserved(&mut dealer, input, spec, lease),
        Err(Error::Resource)
    ));
    assert_eq!(dealer.counter, u64::MAX);
}

#[test]
fn actual_native_work_plus_prior_checked_before_factory_not_byte_authority() {
    let q = resources_for::<&Budget>().unwrap();
    let total = q.work.checked_total().unwrap();
    assert_eq!(q.work.check(17, total + 17).unwrap(), total + 17);
    assert_eq!(q.work.check(17, total + 16), Err(Error::Resource));
    assert_eq!(q.work.check(usize::MAX, usize::MAX), Err(Error::Resource));
    let overflow = Work {
        native_evaluate_aes_per_role: usize::MAX,
        ..q.work
    };
    assert_eq!(overflow.checked_total(), Err(Error::Resource));
    let native = arithmetic::resources_for_context::<&Budget>(native_spec()).unwrap();
    assert_eq!(
        q.work.paired_native_issue_aes,
        native.paired_issue_aes_blocks
    );
    assert_eq!(
        q.work.native_evaluate_aes_per_role,
        native.evaluation_aes_blocks_per_party
    );
    assert_eq!(q.work.random_byte_requests, native.random_bytes);
}

#[test]
fn native_unsigned_tr_controls_keep_128_and_192_not_signed_aliases() {
    // Exercises ORIGINAL native factory, not a fabricated denominator proof.
    for d in [4096u64, 4159, 4160, 8191, 8192, 12287, 12288] {
        for input_mask in [0u64, 1, 8191, 8192, 16383] {
            let budget = Budget::new(4 << 20).unwrap();
            let native = native_spec();
            let cost = arithmetic::resources_for_context::<&Budget>(native).unwrap();
            let context = arithmetic::Context {
                plan: [1; 32],
                session: [2; 32],
                operation: [3; 32],
                input_wire: [4; 32],
                output_wire: [5; 32],
                first_tensor_index: 0,
            };
            let lease = LeaseFor::reserve(&&budget, cost.combined_reservation_bytes).unwrap();
            let (mask, [mut a, mut b]) =
                arithmetic::issue_reserved_with(native, context, &[input_mask], lease, |bytes| {
                    getrandom::fill(bytes).map_err(|_| Error::Randomness)
                })
                .unwrap();
            let x = [(d + input_mask) & 16383];
            let fa = a.begin(&x).unwrap();
            let fb = b.begin(&x).unwrap();
            let arithmetic::StepFor::Frame(ga) = a.advance(fb.as_bytes()).unwrap() else {
                panic!("TR carry phase")
            };
            let arithmetic::StepFor::Frame(gb) = b.advance(fa.as_bytes()).unwrap() else {
                panic!("TR carry phase")
            };
            let arithmetic::StepFor::Output(ya) = a.advance(gb.as_bytes()).unwrap() else {
                panic!("TR output phase")
            };
            let arithmetic::StepFor::Output(yb) = b.advance(ga.as_bytes()).unwrap() else {
                panic!("TR output phase")
            };
            assert_eq!(ya.values(), yb.values());
            assert_eq!(ya.values()[0].wrapping_sub(mask.values()[0]) & 255, d / 64);
        }
    }
}

#[test]
fn reciprocal_bounds_require_reachable_coordinates_not_whole_table_extrema() {
    assert_eq!(262144u64, 1 << 18);
    for z in 64u64..=192 {
        let r = (1 << 18) / z;
        assert!((1365..=4096).contains(&r));
        assert!(4096 * r <= 1 << 24);
    }
    assert_eq!((1 << 18) / 192u64, 1365);
    assert_eq!((1 << 18) / 64u64, 4096);
}
