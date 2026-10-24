//! Native interval component ONLY. The scalar is an ACTUAL encoded local
//! root; this fixture asserts NO X->square->energy producer/whole-row proof.
use super::super::super::super::source_lookup::interval50 as native;
use super::*;
use pllm_core::edge_contract::{EdgeIdentity, Shape};
use std::panic::{catch_unwind, AssertUnwindSafe};

type C = OwnedBudgetContext;
struct Fixture {
    masks: [Option<ProducedMaskFor<C>>; SLOTS],
    values: [[Option<ProducedValueFor<C>>; SLOTS]; 2],
    source: Arc<SourceFor<C>>,
    ledgers: [internal_use::Ledger<USES>; 3],
    _dealer: OwnedLocalDealer,
    _controls: LeaseFor<C>,
}
const USE_SLOT: [usize; USES] = [
    0, 3, 0, 4, 5, 5, 6, 7, 7, 7, 7, 9, 10, 11, 12, 8, 13, 5, 14, 15, 16, 1, 17, 2,
];
fn identity() -> SourceIdentity {
    SourceIdentity::new([71; 32], [72; 32], approved_implementation())
}
fn edge(slot: u8) -> EdgeDescriptor {
    EdgeDescriptor::new(
        EdgeIdentity::new(identity(), [73; 32], [74; 32], [slot; 32], 0),
        Shape::new(&[1]).unwrap(),
        Numeric::new(50, if slot == 7 { 24 } else { 0 }, Interpretation::Signed).unwrap(),
        Representation::CommonAdditiveMasked,
        None,
    )
    .unwrap()
}
fn fixture(budget: &OwnedBudget, input: i64) -> Fixture {
    fixture_at_mask(budget, input, None)
}
fn fixture_at_mask(budget: &OwnedBudget, input: i64, forced: Option<u64>) -> Fixture {
    assert!((0..1 << 29).contains(&input));
    let context = budget.reserve_native(0).unwrap().budget.share();
    // Concrete fixture move/result/scope slots; not numeric outside-owner credit.
    let controls = LeaseFor::reserve(
        &context,
        size_of::<Fixture>()
            + size_of::<Result<Fixture, Error>>()
            + size_of::<[Binding; USES]>()
            + size_of::<Arc<SourceFor<C>>>()
            + size_of::<Zeroizing<[u64; 2]>>()
            + role_coordination::arc_bytes::<table::CertifiedTableFor<C>>().unwrap()
            + 64,
    )
    .unwrap();
    let mut dealer = OwnedLocalDealer::new(identity(), [73; 32], [75; 32], budget).unwrap();
    let bound = BoundRange::new(0, 1 << 29).unwrap();
    let root = if let Some(word) = forced {
        let bytes = dealer.root_reservation_bytes(edge(7), &bound).unwrap();
        dealer
            .prepare_root_reserved_with(
                edge(7),
                bound,
                [76; 32],
                LeaseFor::reserve(&context, bytes).unwrap(),
                |out| {
                    assert_eq!(out.len(), 8);
                    out.copy_from_slice(&word.to_le_bytes());
                    Ok(())
                },
            )
            .unwrap()
    } else {
        dealer.prepare_root(edge(7), bound, [76; 32]).unwrap()
    };
    let (mask, [v0, v1]) = root.encode(&[input]).unwrap();
    let table = table::construct(
        LeaseFor::reserve(&context, table::reservation_bytes::<C>().unwrap()).unwrap(),
    )
    .unwrap();
    let source_lease = LeaseFor::reserve(
        &context,
        role_coordination::arc_bytes::<SourceFor<C>>().unwrap() + 64,
    )
    .unwrap();
    let descriptors = std::array::from_fn(|i| edge(i as u8));
    let uses = std::array::from_fn(|i| {
        ConsumerUse::new(
            descriptors[USE_SLOT[i]],
            [76; 32],
            u16::from(i == 5 || i == 11 || i == 18 || i == 21),
            i as u32,
        )
    });
    let source = Arc::new(SourceFor {
        profile: [77; 32],
        session: [75; 32],
        operation: [78; 32],
        descriptors,
        kinds: std::array::from_fn(|i| match i {
            8 => ValueKind::UnsignedResidue { bits: 13 },
            9 => ValueKind::UnsignedResidue { bits: 50 },
            7 => ValueKind::SignedFixed {
                bits: 50,
                fraction: 24,
            },
            _ => ValueKind::SignedFixed {
                bits: 50,
                fraction: 0,
            },
        }),
        uses,
        table: Arc::new(table),
        lease: source_lease,
    });
    let binding = mask.binding;
    let mut masks = std::array::from_fn(|_| None);
    masks[7] = Some(ProducedMaskFor {
        backing: MaskBacking::Original(mask),
        binding,
        source: Arc::clone(&source),
        lease: LeaseFor::reserve(&context, 0).unwrap(),
    });
    let mut values: [[_; SLOTS]; 2] = std::array::from_fn(|_| std::array::from_fn(|_| None));
    for (role, value) in [v0, v1].into_iter().enumerate() {
        assert_eq!(value.binding, binding);
        values[role][7] = Some(ProducedValueFor {
            backing: ValueBacking::Original(value),
            binding,
            role: role as u8,
            source: Arc::clone(&source),
            lease: LeaseFor::reserve(&context, 0).unwrap(),
        });
    }
    let ledgers = std::array::from_fn(|role| internal_use::Ledger {
        uses,
        bindings: [binding; USES],
        role: u8::from(role == 2),
        next: 7,
    });
    Fixture {
        masks,
        values,
        source,
        ledgers,
        _dealer: dealer,
        _controls: controls,
    }
}
fn mask(f: &mut Fixture) -> EnergyMaskLoan<'_, C> {
    EnergyMaskLoan::take(&f.source, &mut f.ledgers[0], &f.masks).unwrap()
}
fn value(f: &mut Fixture, role: usize) -> EnergyValueLoan<'_, C> {
    EnergyValueLoan::take(
        &f.source,
        &mut f.ledgers[role + 1],
        &f.values[role],
        role as u8,
    )
    .unwrap()
}
fn fill(v: &mut [u8]) -> Result<(), Error> {
    v.fill(0x69);
    Ok(())
}
fn plain(input: u64) -> [u64; 2] {
    if input == 0 {
        [0, 0]
    } else {
        let e = 63 - input.leading_zeros();
        [u64::from(e), 1u64 << (49 - e)]
    }
}

#[test]
fn genuine_scalar_component_both_payloads_endpoints_and_distinct_bindings() {
    let budget = OwnedBudget::new(8 << 20).unwrap();
    for (input, forced) in [
        (0, None),
        (1, None),
        (127, None),
        (128, Some((1 << 50) - 128)),
        ((1 << 27) - 1, None),
    ] {
        let mut f = fixture_at_mask(&budget, input, forced);
        if forced.is_some() {
            assert_eq!(f.values[0][7].as_ref().unwrap().backing.values()[0], 0);
        }
        let q = native::resources::<C>().unwrap();
        let (m, [mut a, mut b]) = native::issue_reserved_with(
            mask(&mut f),
            budget.reserve_native(q.combined).unwrap(),
            fill,
        )
        .unwrap();
        let fa = a.begin(value(&mut f, 0), q.scan.work).unwrap();
        let fb = b.begin(value(&mut f, 1), q.scan.work).unwrap();
        let native::Step::Frames(ga) = a
            .advance([fb.frame(0).unwrap(), fb.frame(1).unwrap()])
            .unwrap()
        else {
            panic!("select")
        };
        let native::Step::Frames(gb) = b
            .advance([fa.frame(0).unwrap(), fa.frame(1).unwrap()])
            .unwrap()
        else {
            panic!("select")
        };
        let native::Step::Values(va) = a
            .advance([gb.frame(0).unwrap(), gb.frame(1).unwrap()])
            .unwrap()
        else {
            panic!("outputs")
        };
        let native::Step::Values(vb) = b
            .advance([ga.frame(0).unwrap(), ga.frame(1).unwrap()])
            .unwrap()
        else {
            panic!("outputs")
        };
        assert_ne!(
            va.output(&f.source, 0).unwrap().binding(),
            va.output(&f.source, 1).unwrap().binding()
        );
        for (i, bits) in [13, 50].into_iter().enumerate() {
            let left = va.output(&f.source, i).unwrap().values();
            let right = vb.output(&f.source, i).unwrap().values();
            assert_eq!(left, right);
            assert_eq!(
                left[0].wrapping_sub(m.output(i).unwrap()[0])
                    & super::super::super::super::mask(bits),
                plain(input as u64)[i]
            );
        }
    }
}

#[test]
fn one_short_and_foreign_counter_refuse_before_entropy_and_burn() {
    let budget = OwnedBudget::new(8 << 20).unwrap();
    let other = OwnedBudget::new(8 << 20).unwrap();
    let q = native::resources::<C>().unwrap();
    for foreign in [false, true] {
        let mut f = fixture(&budget, 128);
        let lease = if foreign {
            other.reserve_native(q.combined)
        } else {
            budget.reserve_native(q.combined - 1)
        }
        .unwrap();
        assert!(native::issue_reserved_with(mask(&mut f), lease, |_| panic!(
            "must precede entropy"
        ))
        .is_err());
        assert!(EnergyMaskLoan::take(&f.source, &mut f.ledgers[0], &f.masks).is_err());
    }
}

#[test]
fn wrong_role_and_equal_profile_foreign_invocation_burn_native_key() {
    let budget = OwnedBudget::new(8 << 20).unwrap();
    let q = native::resources::<C>().unwrap();
    for wrong_role in [false, true] {
        let mut f = fixture(&budget, 1);
        let mut foreign = fixture(&budget, 1);
        let (_m, [mut a, _b]) = native::issue_reserved_with(
            mask(&mut f),
            budget.reserve_native(q.combined).unwrap(),
            fill,
        )
        .unwrap();
        let loan = if wrong_role {
            value(&mut f, 1)
        } else {
            value(&mut foreign, 0)
        };
        assert!(a.begin(loan, q.scan.work).is_err());
        assert!(a.begin(value(&mut f, 0), q.scan.work).is_err());
    }
}

#[test]
fn dealer_drop_retains_guard_begin_result_and_last_frame_output() {
    let budget = OwnedBudget::new(8 << 20).unwrap();
    let baseline = budget.budget().used_bytes();
    let mut f = fixture(&budget, 128);
    let q = native::resources::<C>().unwrap();
    let (m, [mut a, mut b]) = native::issue_reserved_with(
        mask(&mut f),
        budget.reserve_native(q.combined).unwrap(),
        fill,
    )
    .unwrap();
    let controls = a.retained_evaluator_bytes();
    assert_eq!(controls, b.retained_evaluator_bytes());
    let [layout, guard, begin_return] = native::transition_layout::<C>();
    assert_eq!(controls, layout);
    assert!(controls >= guard + begin_return + size_of::<Result<native::Step<C>, Error>>());
    println!("interval logical quote: dealer={} party={} combined={} evaluator={} abort_guard={} begin_return={}; root/source/outer-parent additional",q.dealer,q.party,q.combined,controls,guard,begin_return);
    assert!(
        controls
            > size_of::<EnergyValueLoan<'_, C>>()
                + size_of::<Result<native::FramesFor<C>, Error>>()
    );
    drop(m);
    assert_eq!(a.retained_evaluator_bytes(), controls);
    assert_eq!(b.retained_evaluator_bytes(), controls);
    let fa = a.begin(value(&mut f, 0), q.scan.work).unwrap();
    let fb = b.begin(value(&mut f, 1), q.scan.work).unwrap();
    let native::Step::Frames(ga) = a
        .advance([fb.frame(0).unwrap(), fb.frame(1).unwrap()])
        .unwrap()
    else {
        panic!("select")
    };
    let native::Step::Frames(gb) = b
        .advance([fa.frame(0).unwrap(), fa.frame(1).unwrap()])
        .unwrap()
    else {
        panic!("select")
    };
    let native::Step::Values(va) = a
        .advance([gb.frame(0).unwrap(), gb.frame(1).unwrap()])
        .unwrap()
    else {
        panic!("outputs")
    };
    let native::Step::Values(vb) = b
        .advance([ga.frame(0).unwrap(), ga.frame(1).unwrap()])
        .unwrap()
    else {
        panic!("outputs")
    };
    assert_eq!(a.retained_evaluator_bytes(), controls);
    assert_eq!(b.retained_evaluator_bytes(), controls);
    assert_eq!(
        va.output(&f.source, 0).unwrap().values(),
        vb.output(&f.source, 0).unwrap().values()
    );
    drop(f);
    drop(a);
    drop(b);
    drop(fb);
    drop(ga);
    drop(gb);
    drop(vb);
    assert!(budget.budget().used_bytes() > baseline + 65536);
    drop(va);
    assert!(budget.budget().used_bytes() > baseline + 65536);
    drop(fa);
    assert_eq!(budget.budget().used_bytes(), baseline);
}

#[test]
fn original_key_material_failure_and_unwind_release_no_retry() {
    let budget = OwnedBudget::new(8 << 20).unwrap();
    let q = native::resources::<C>().unwrap();
    for panic_at_draw in [false, true] {
        let mut f = fixture(&budget, 128);
        let before = budget.budget().used_bytes();
        let mut draws = 0;
        let result = catch_unwind(AssertUnwindSafe(|| {
            native::issue_reserved_with(
                mask(&mut f),
                budget.reserve_native(q.combined).unwrap(),
                |v| {
                    draws += 1;
                    if draws == 5 {
                        if panic_at_draw {
                            panic!("partial exponent material")
                        }
                        return Err(Error::Randomness);
                    }
                    fill(v)
                },
            )
        }));
        if panic_at_draw {
            assert!(result.is_err())
        } else {
            assert!(result.unwrap().is_err())
        }
        assert_eq!(draws, 5);
        assert_eq!(budget.budget().used_bytes(), before);
        assert!(EnergyMaskLoan::take(&f.source, &mut f.ledgers[0], &f.masks).is_err());
    }
}

#[test]
fn short_scan_work_burns_before_any_paired_output() {
    let budget = OwnedBudget::new(8 << 20).unwrap();
    let mut f = fixture(&budget, 1);
    let q = native::resources::<C>().unwrap();
    let (_m, [mut a, _b]) = native::issue_reserved_with(
        mask(&mut f),
        budget.reserve_native(q.combined).unwrap(),
        fill,
    )
    .unwrap();
    let mut short = q.scan.work;
    short.less_queries -= 1;
    assert!(a.begin(value(&mut f, 0), short).is_err());
    assert!(a.advance([&[], &[]]).is_err());
}

#[test]
fn real_begin_and_advance_unwind_burns_key_and_keeps_evaluator_lease() {
    let budget = OwnedBudget::new(8 << 20).unwrap();
    let q = native::resources::<C>().unwrap();
    for begin in [true, false] {
        let mut f = fixture(&budget, 128);
        let (_m, [mut a, mut b]) = native::issue_reserved_with(
            mask(&mut f),
            budget.reserve_native(q.combined).unwrap(),
            fill,
        )
        .unwrap();
        let controls = a.retained_evaluator_bytes();
        if begin {
            native::panic_next_transition(true);
            assert!(
                catch_unwind(AssertUnwindSafe(|| a.begin(value(&mut f, 0), q.scan.work))).is_err()
            );
        } else {
            let fa = a.begin(value(&mut f, 0), q.scan.work).unwrap();
            let fb = b.begin(value(&mut f, 1), q.scan.work).unwrap();
            native::panic_next_transition(false);
            assert!(catch_unwind(AssertUnwindSafe(
                || a.advance([fb.frame(0).unwrap(), fb.frame(1).unwrap()])
            ))
            .is_err());
            assert!(!fa.frame(0).unwrap().is_empty());
        }
        assert_eq!(a.retained_evaluator_bytes(), controls);
        assert!(a.advance([&[], &[]]).is_err());
    }
}
