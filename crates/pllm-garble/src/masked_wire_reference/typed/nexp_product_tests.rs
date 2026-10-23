use super::*;
use crate::masked_wire_reference::source_lookup as lookup;
use pllm_core::edge_contract::{EdgeIdentity, Shape};

#[test]
fn reserved_declaration_checks_exact_original_counter_before_binding_and_factory() {
    let budget = Budget::new(64 << 20).unwrap();
    let foreign = Budget::new(64 << 20).unwrap();
    let mut dealer = LocalDealer::new(source(), [3; 32], [4; 32], &budget).unwrap();
    let metadata = LeaseFor::reserve(
        &dealer.budget,
        size_of::<[Binding; 3]>() + size_of::<[Sha256; 2]>(),
    )
    .unwrap();
    let bindings = std::array::from_fn(|i| {
        dealer
            .binding(
                edge(6 + i as u8, 12),
                [9; 32],
                Some(Interval {
                    lower: 0,
                    upper: 4097,
                }),
            )
            .unwrap()
    });
    let cost = native::resources_for::<&Budget>(1).unwrap().combined_bytes
        + wrapper_bytes::<&Budget>().unwrap();
    let counter = dealer.counter;
    budget
        .fail_after
        .store(0, std::sync::atomic::Ordering::SeqCst);
    for (context, bytes) in [(&budget, cost - 1), (&budget, cost + 1), (&foreign, cost)] {
        let lease = LeaseFor::reserve(&context, bytes).unwrap();
        assert!(matches!(
            declare_reserved(&mut dealer, [9; 32], bindings, edge(11, 24), [9; 32], lease),
            Err(Error::Resource)
        ));
        assert_eq!(dealer.counter, counter);
        assert_eq!(
            budget.fail_after.load(std::sync::atomic::Ordering::SeqCst),
            0
        );
    }
    let lease = LeaseFor::reserve(&&budget, cost).unwrap();
    let declaration =
        declare_reserved(&mut dealer, [9; 32], bindings, edge(11, 24), [9; 32], lease).unwrap();
    assert_eq!(dealer.counter, counter + 1);
    assert_eq!(
        declaration.output_binding().interval,
        Some(Interval {
            lower: 0,
            upper: 16_777_217
        })
    );
    budget
        .fail_after
        .store(-1, std::sync::atomic::Ordering::SeqCst);
    drop((declaration, metadata));
}

#[test]
fn injected_product_issuance_consumes_genuine_d19_masks_with_single_entropy_path() {
    let budget = Budget::new(64 << 20).unwrap();
    let mut dealer = LocalDealer::new(source(), [3; 32], [4; 32], &budget).unwrap();
    let (m0, [a0, a1]) = lookup_owners(&mut dealer, 6, 4096);
    let (m1, [b0, b1]) = lookup_owners(&mut dealer, 7, 1);
    let cost = native::resources_for::<&Budget>(1).unwrap().combined_bytes
        + wrapper_bytes::<&Budget>().unwrap();
    let original = LeaseFor::reserve(&&budget, cost).unwrap();
    let declaration = declare_reserved(
        &mut dealer,
        [9; 32],
        [m0.binding, m1.binding],
        edge(11, 24),
        [9; 32],
        original,
    )
    .unwrap();
    let mut bytes = 0;
    let mut word = 0u8;
    let (m, [mut left, mut right]) = declaration
        .issue_with([8; 32], [10; 32], [m0, m1], |out| {
            bytes += out.len();
            for b in out {
                word = word.wrapping_add(37);
                *b = word;
            }
            Ok(())
        })
        .unwrap();
    assert_eq!(
        bytes,
        native::resources_for::<&Budget>(1).unwrap().random_bytes
    );
    let a = left.begin([a0, b0]).unwrap();
    let b = right.begin([a1, b1]).unwrap();
    for output in [
        left.finish(b.as_bytes()).unwrap(),
        right.finish(a.as_bytes()).unwrap(),
    ] {
        assert_eq!(
            output.native.values()[0].wrapping_sub(m.native.values()[0])
                & super::super::super::mask(50),
            4096
        );
    }
}

fn source() -> SourceIdentity {
    SourceIdentity::new([1; 32], [2; 32], approved_implementation())
}
fn edge(wire: u8, scale: u8) -> EdgeDescriptor {
    EdgeDescriptor::new(
        EdgeIdentity::new(source(), [3; 32], [4; 32], [wire; 32], 0),
        Shape::new(&[1]).unwrap(),
        Numeric::new(50, scale, Interpretation::Signed).unwrap(),
        Representation::CommonAdditiveMasked,
        None,
    )
    .unwrap()
}
fn lookup_owners<'a>(
    dealer: &mut LocalDealer<'a>,
    wire: u8,
    payload: u64,
) -> (LookupMaskFor<&'a Budget>, [LookupValueFor<&'a Budget>; 2]) {
    let _metadata = LeaseFor::reserve(
        &dealer.budget,
        size_of::<[Binding; 3]>() + size_of::<[Sha256; 2]>(),
    )
    .unwrap();
    let binding = dealer
        .binding(
            edge(wire, 12),
            [9; 32],
            Some(Interval {
                lower: 0,
                upper: 4097,
            }),
        )
        .unwrap();
    lookup_owners_for_binding(dealer, binding, payload)
}

fn lookup_owners_for_binding<'a>(
    dealer: &mut LocalDealer<'a>,
    binding: Binding,
    payload: u64,
) -> (LookupMaskFor<&'a Budget>, [LookupValueFor<&'a Budget>; 2]) {
    // Genuine D19 owners. The constant table is a generic leaf fixture, NOT an
    // nExp table substitution or a floor-proof/activation token.
    let mut wrapper = LeaseFor::reserve(
        &dealer.budget,
        size_of::<LookupMaskFor<&Budget>>()
            + 2 * size_of::<LookupValueFor<&Budget>>()
            + size_of::<[Binding; 3]>()
            + size_of::<[Sha256; 2]>(),
    )
    .unwrap();
    let context = lookup::Context {
        plan: [3; 32],
        session: [4; 32],
        operation: binding.descriptor.identity().wire(),
        input_wire: [21; 32],
        output_wire: [22; 32],
        arithmetic_wire: [23; 32],
        correction_wire: [24; 32],
        select_wire: [25; 32],
        numeric_contract: [26; 32],
        first_tensor_index: 0,
    };
    let spec = lookup::Spec {
        index_bits: 8,
        output_bits: 50,
        lanes: 1,
        boundary: selection::Boundary::Reconstructed,
    };
    let (masks, [mut left, mut right]) =
        lookup::issue(spec, context, &[payload; 256], &[5], dealer.budget).unwrap();
    let a = left.begin(&[12]).unwrap();
    let b = right.begin(&[12]).unwrap();
    let lookup::StepFor::Frame(c) = left.advance(b.as_bytes()).unwrap() else {
        panic!("lookup phase")
    };
    let lookup::StepFor::Frame(d) = right.advance(a.as_bytes()).unwrap() else {
        panic!("lookup phase")
    };
    let lookup::StepFor::Reconstructed(l) = left.advance(d.as_bytes()).unwrap() else {
        panic!("lookup output")
    };
    let lookup::StepFor::Reconstructed(r) = right.advance(c.as_bytes()).unwrap() else {
        panic!("lookup output")
    };
    let ll = wrapper.split(size_of::<LookupValueFor<&Budget>>()).unwrap();
    let rr = wrapper.split(size_of::<LookupValueFor<&Budget>>()).unwrap();
    (
        LookupMaskFor {
            native: masks,
            binding,
            lease: wrapper,
        },
        [
            LookupValueFor {
                native: l,
                binding,
                role: 0,
                lease: ll,
            },
            LookupValueFor {
                native: r,
                binding,
                role: 1,
                lease: rr,
            },
        ],
    )
}

fn issue<'a>(
    dealer: &mut LocalDealer<'a>,
    region: [u8; 32],
    operation: [u8; 32],
    stage: [u8; 32],
    inputs: [LookupMaskFor<&'a Budget>; 2],
    output: EdgeDescriptor,
    next: [u8; 32],
) -> Result<(ProductMaskFor<&'a Budget>, [PartyFor<&'a Budget>; 2]), Error> {
    let declaration = declare(
        dealer,
        operation,
        [inputs[0].binding, inputs[1].binding],
        output,
        next,
    )?;
    declaration.issue(region, stage, inputs)
}

#[test]
fn original_d19_outputs_are_consumed_without_copy_and_preserve_q24_product_bound() {
    let budget = Budget::new(64 << 20).unwrap();
    let mut dealer = LocalDealer::new(source(), [3; 32], [4; 32], &budget).unwrap();
    let (m0, [a0, a1]) = lookup_owners(&mut dealer, 6, 4096);
    let (m1, [b0, b1]) = lookup_owners(&mut dealer, 7, 4096);
    let original = a0.native.values().as_ptr();
    let (mask, [mut left, mut right]) = issue(
        &mut dealer,
        [8; 32],
        [9; 32],
        [10; 32],
        [m0, m1],
        edge(11, 24),
        [12; 32],
    )
    .unwrap();
    let a = left.begin([a0, b0]).unwrap();
    let b = right.begin([a1, b1]).unwrap();
    assert_eq!(
        left.retained.as_ref().unwrap()[0].native.values().as_ptr(),
        original
    );
    let l = left.finish(b.as_bytes()).unwrap();
    let r = right.finish(a.as_bytes()).unwrap();
    assert_eq!(
        l.binding.interval,
        Some(Interval {
            lower: 0,
            upper: 16_777_217
        })
    );
    assert_eq!(l.binding.descriptor.numeric().fractional_bits(), 24);
    for output in [l, r] {
        assert_eq!(
            output.native.values()[0].wrapping_sub(mask.native.values()[0])
                & super::super::super::mask(50),
            1 << 24
        );
    }
    assert!(matches!(left.finish(b.as_bytes()), Err(Error::Consumed)));
}

#[test]
fn root_metadata_cannot_upgrade_to_lookup_product_capability() {
    // Real type-level negative checks inside the defining privacy scope. These
    // cannot pass merely because an external import names a private type.
    macro_rules! no_into {
        ($source:ty, $target:ty) => {{
            trait AmbiguousIfInto<A> {
                fn marker() {}
            }
            impl<T> AmbiguousIfInto<()> for T {}
            struct Imported;
            impl<T: Into<$target>> AmbiguousIfInto<Imported> for T {}
            let _ = <$source as AmbiguousIfInto<_>>::marker;
        }};
    }
    no_into!(DealerMask<'static, Ring>, LookupMaskFor<&'static Budget>);
    no_into!(CommonMasked<'static, Ring>, LookupValueFor<&'static Budget>);
    no_into!(
        CommonXorMasked<'static, Predicate>,
        LookupValueFor<&'static Budget>
    );
    no_into!(EdgeDescriptor, LookupValueFor<&'static Budget>);
}

#[test]
fn role_or_binding_failure_burns_private_product_owner_not_a_retryable_import() {
    let budget = Budget::new(64 << 20).unwrap();
    let mut dealer = LocalDealer::new(source(), [3; 32], [4; 32], &budget).unwrap();
    let (m0, [a0, a1]) = lookup_owners(&mut dealer, 6, 1);
    let (m1, [b0, b1]) = lookup_owners(&mut dealer, 7, 4096);
    let (_mask, [mut left, mut right]) = issue(
        &mut dealer,
        [8; 32],
        [9; 32],
        [10; 32],
        [m0, m1],
        edge(11, 24),
        [12; 32],
    )
    .unwrap();
    assert!(matches!(left.begin([a1, b1]), Err(Error::Context)));
    assert!(left.native.is_none());
    assert!(matches!(left.begin([a0, b0]), Err(Error::Consumed)));
    // This prerequisite is NOT the enclosing two-role coordinator. Until that
    // owner exists/review passes, no public region can issue these leaves.
    right.cancel();
}

#[test]
fn wider_interval_and_duplicate_owner_identity_refuse_before_product_binding_or_entropy() {
    let budget = Budget::new(64 << 20).unwrap();
    let mut dealer = LocalDealer::new(source(), [3; 32], [4; 32], &budget).unwrap();
    for duplicate in [false, true] {
        let (m0, a) = lookup_owners(&mut dealer, 6, 4096);
        let (mut m1, b) = lookup_owners(&mut dealer, 7, 4096);
        if duplicate {
            m1.binding = m0.binding;
        } else {
            m1.binding.interval = Some(Interval {
                lower: 0,
                upper: 8192,
            });
        }
        let before = dealer.counter;
        let result = issue(
            &mut dealer,
            [8; 32],
            [9; 32],
            [10; 32],
            [m0, m1],
            edge(11, 24),
            [12; 32],
        );
        if duplicate {
            assert!(matches!(result, Err(Error::Context)));
        } else {
            assert!(matches!(result, Err(Error::Domain)));
        }
        assert_eq!(dealer.counter, before);
        drop((a, b));
    }
}

#[test]
fn product_binding_precommits_before_actual_lookup_material_without_digest_cycle() {
    let budget = Budget::new(64 << 20).unwrap();
    let mut dealer = LocalDealer::new(source(), [3; 32], [4; 32], &budget).unwrap();
    let _metadata = LeaseFor::reserve(
        &&budget,
        size_of::<[Binding; 3]>() + size_of::<[Sha256; 2]>(),
    )
    .unwrap();
    let a = dealer
        .binding(
            edge(6, 12),
            [9; 32],
            Some(Interval {
                lower: 0,
                upper: 4097,
            }),
        )
        .unwrap();
    let b = dealer
        .binding(
            edge(7, 12),
            [9; 32],
            Some(Interval {
                lower: 0,
                upper: 4097,
            }),
        )
        .unwrap();
    let declaration = declare(&mut dealer, [9; 32], [a, b], edge(11, 24), [12; 32]).unwrap();
    let p = declaration.output_binding();
    let counter_after_declarations = dealer.counter;
    // All actual bindings now exist. This partial test is NOT the nExp region
    // digest: it demonstrates precommit order without fabricating a review token.
    let region = digest(&[
        b"product-precommit-test.v1",
        &a.digest(),
        &b.digest(),
        &p.digest(),
    ]);
    let (ma, [a0, a1]) = lookup_owners_for_binding(&mut dealer, a, 4096);
    let (mb, [b0, b1]) = lookup_owners_for_binding(&mut dealer, b, 4096);
    assert_eq!(dealer.counter, counter_after_declarations);
    let (mask, [mut left, mut right]) = declaration.issue(region, [10; 32], [ma, mb]).unwrap();
    assert_eq!(mask.binding, p);
    let f0 = left.begin([a0, b0]).unwrap();
    let f1 = right.begin([a1, b1]).unwrap();
    let p0 = left.finish(f1.as_bytes()).unwrap();
    let p1 = right.finish(f0.as_bytes()).unwrap();
    assert_eq!(p0.binding, p);
    assert_eq!(p1.binding, p);
}
