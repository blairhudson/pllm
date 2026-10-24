//! Native inverse SUBRECIPE fixture only. Actual original ring50 root is
//! encoded, then reduced homomorphically to its private index13 coordinate.
//! No whole-row X/energy/interval success is asserted or synthesized here.
use super::super::super::source_lookup::{self, norm13};
use super::*;
use pllm_core::edge_contract::{EdgeIdentity, Shape};

const USE_SLOT: [usize; USES] = [
    0, 3, 0, 4, 5, 5, 6, 7, 7, 7, 7, 9, 10, 11, 12, 8, 13, 5, 14, 15, 16, 1, 17, 2,
];

struct Fixture<C: AllocationContext> {
    masks: [Option<ProducedMaskFor<C>>; SLOTS],
    values: [[Option<ProducedValueFor<C>>; SLOTS]; 2],
    source: Arc<SourceFor<C>>,
    ledgers: [internal_use::Ledger<USES>; 3],
    _original_mask: DealerMaskFor<C, Ring>,
    _original_values: [CommonMaskedFor<C, Ring>; 2],
    _dealer: LocalDealerFor<C>,
    _control: LeaseFor<C>,
}

fn identity() -> SourceIdentity {
    SourceIdentity::new([41; 32], [42; 32], approved_implementation())
}
fn edge(slot: u8) -> EdgeDescriptor {
    EdgeDescriptor::new(
        EdgeIdentity::new(identity(), [43; 32], [44; 32], [slot; 32], 0),
        Shape::new(&[1]).unwrap(),
        Numeric::new(
            if slot == 13 { 13 } else { 50 },
            0,
            if slot == 13 {
                Interpretation::Unsigned
            } else {
                Interpretation::Signed
            },
        )
        .unwrap(),
        Representation::CommonAdditiveMasked,
        None,
    )
    .unwrap()
}
fn fixture<C: AllocationContext>(context: C, index: i64) -> Fixture<C> {
    assert!((0..8192).contains(&index));
    let mut dealer = LocalDealerFor::from_reserved_namespace(
        identity(),
        [43; 32],
        [45; 32],
        LeaseFor::reserve(&context, size_of::<LocalDealerFor<C>>() + 4096).unwrap(),
    )
    .unwrap();
    // Public roots/typed factories issue the actual owners; neither source
    // binding nor private mask words are adopted from a test tuple.
    let (original_mask, original_values) = dealer
        .prepare_root(edge(0), BoundRange::new(0, 8192).unwrap(), [46; 32])
        .unwrap()
        .encode(&[index])
        .unwrap();
    let table = table::construct(
        LeaseFor::reserve(&context, table::reservation_bytes::<C>().unwrap()).unwrap(),
    )
    .unwrap();
    let source_bytes = role_coordination::arc_bytes::<SourceFor<C>>().unwrap() + 64;
    let table_arc_bytes =
        role_coordination::arc_bytes::<table::CertifiedTableFor<C>>().unwrap() + 64;
    // Keep all actual inline fixture controls and three local allocations
    // resident. Original native root/table charges remain additional and live.
    let control = LeaseFor::reserve(
        &context,
        size_of::<Fixture<C>>()
            + table_arc_bytes
            + 3 * (8 + 64 + size_of::<Vec<u64>>())
            + size_of::<[Binding; USES]>()
            + size_of::<Result<Fixture<C>, Error>>(),
    )
    .unwrap();
    let descriptors = std::array::from_fn(|i| edge(i as u8));
    let uses = std::array::from_fn(|i| {
        ConsumerUse::new(
            descriptors[USE_SLOT[i]],
            [46; 32],
            u16::from(i == 5 || i == 11 || i == 18 || i == 21),
            i as u32,
        )
    });
    let source = Arc::new(SourceFor {
        profile: [47; 32],
        session: [45; 32],
        descriptors,
        kinds: std::array::from_fn(|i| {
            if i == 13 {
                ValueKind::UnsignedResidue { bits: 13 }
            } else {
                ValueKind::SignedFixed {
                    bits: 50,
                    fraction: 0,
                }
            }
        }),
        uses,
        operation: [55; 32],
        table: Arc::new(table),
        lease: LeaseFor::reserve(&context, source_bytes).unwrap(),
    });
    let binding = dealer.binding(edge(13), [46; 32], None).unwrap();
    let mut masks = std::array::from_fn(|_| None);
    let mut values: [[_; SLOTS]; 2] = std::array::from_fn(|_| std::array::from_fn(|_| None));
    let mut reduced = Zeroizing::new(allocate(1, context.budget()).unwrap());
    reduced.push(original_mask.backing.values()[0] & 8191);
    masks[13] = Some(ProducedMaskFor {
        backing: MaskBacking::Local(reduced),
        binding,
        source: Arc::clone(&source),
        lease: LeaseFor::reserve(&context, 0).unwrap(),
    });
    for role in 0..2 {
        let mut reduced = Zeroizing::new(allocate(1, context.budget()).unwrap());
        reduced.push(original_values[role].backing.values()[0] & 8191);
        values[role][13] = Some(ProducedValueFor {
            backing: ValueBacking::Local(reduced),
            binding,
            source: Arc::clone(&source),
            role: role as u8,
            lease: LeaseFor::reserve(&context, 0).unwrap(),
        });
    }
    let ledgers = std::array::from_fn(|role| internal_use::Ledger {
        uses,
        bindings: [binding; USES],
        role: u8::from(role == 2),
        next: 16,
    });
    Fixture {
        masks,
        values,
        source,
        ledgers,
        _original_mask: original_mask,
        _original_values: original_values,
        _dealer: dealer,
        _control: control,
    }
}
fn mask<C: AllocationContext>(f: &mut Fixture<C>) -> Norm13MaskLoan<'_, C> {
    Norm13MaskLoan::take(&f.source, &mut f.ledgers[0], &f.masks).unwrap()
}
fn value<C: AllocationContext>(f: &mut Fixture<C>, role: usize) -> Norm13ValueLoan<'_, C> {
    Norm13ValueLoan::take(
        &f.source,
        &mut f.ledgers[role + 1],
        &f.values[role],
        role as u8,
    )
    .unwrap()
}

#[test]
fn original_certified_table_selected_indices_and_legacy_gate() {
    let budget = Budget::new(8 << 20).unwrap();
    for index in [0, 127, 128, 8191] {
        let mut f = fixture(&budget, index);
        let expected = f.source.table.values()[index as usize];
        let q = norm13::resources::<&Budget>().unwrap();
        assert_eq!(
            (
                q.table_entries,
                q.full_expansion_scratch_bytes,
                q.evaluation_aes_blocks_per_party
            ),
            (8192, 8192, 126)
        );
        assert_eq!(q.borrowed_table_bytes, 65536);
        let (m, [mut a, mut b]) = norm13::issue_reserved_with(
            mask(&mut f),
            budget.reserve(q.combined_reservation_bytes).unwrap(),
            |v| {
                v.fill(0x65);
                Ok(())
            },
        )
        .unwrap();
        let fa = a.begin(value(&mut f, 0)).unwrap();
        let fb = b.begin(value(&mut f, 1)).unwrap();
        let norm13::Step::Frame(ga) = a.advance(fb.as_bytes()).unwrap() else {
            panic!("select frame")
        };
        let norm13::Step::Frame(gb) = b.advance(fa.as_bytes()).unwrap() else {
            panic!("select frame")
        };
        let norm13::Step::Output(va) = a.advance(gb.as_bytes()).unwrap() else {
            panic!("output")
        };
        let norm13::Step::Output(vb) = b.advance(ga.as_bytes()).unwrap() else {
            panic!("output")
        };
        assert_eq!(va.values(), vb.values());
        assert_eq!(
            va.values()[0].wrapping_sub(m.values()[0]) & super::super::super::mask(50),
            expected
        );
    }
    assert!(source_lookup::resources(source_lookup::Spec {
        index_bits: 13,
        output_bits: 50,
        lanes: 1,
        boundary: source_lookup::Boundary::Reconstructed
    })
    .is_err());
}

#[test]
fn exact_reservation_short_and_foreign_counter_before_entropy_burn_use() {
    let budget = Budget::new(8 << 20).unwrap();
    let other = Budget::new(8 << 20).unwrap();
    let q = norm13::resources::<&Budget>().unwrap();
    for foreign in [false, true] {
        let mut f = fixture(&budget, 128);
        let lease = if foreign {
            other.reserve(q.combined_reservation_bytes)
        } else {
            budget.reserve(q.combined_reservation_bytes - 1)
        }
        .unwrap();
        assert!(
            norm13::issue_reserved_with(mask(&mut f), lease, |_| panic!("preflight entropy"))
                .is_err()
        );
        assert!(Norm13MaskLoan::take(&f.source, &mut f.ledgers[0], &f.masks).is_err());
    }
}

#[test]
fn foreign_role_and_retained_original_counter_on_failure() {
    let budget = OwnedBudget::new(8 << 20).unwrap();
    let baseline = budget.budget().used_bytes();
    let context = budget.reserve_native(0).unwrap().budget.share();
    let mut f = fixture(context, 0);
    let q = norm13::resources::<OwnedBudgetContext>().unwrap();
    let (m, [mut a, b]) = norm13::issue_reserved_with(
        mask(&mut f),
        budget.reserve_native(q.combined_reservation_bytes).unwrap(),
        |v| {
            v.fill(0x54);
            Ok(())
        },
    )
    .unwrap();
    assert!(a.begin(value(&mut f, 1)).is_err());
    let live = budget.budget().used_bytes();
    drop(f);
    drop(b);
    assert!(budget.budget().used_bytes() > baseline);
    drop(a);
    assert!(budget.budget().used_bytes() > baseline);
    drop(m);
    assert_eq!(budget.budget().used_bytes(), baseline);
    assert!(live > baseline);
}

#[test]
fn failed_original_keypair_suffix_erases_and_cannot_retry() {
    let budget = Budget::new(8 << 20).unwrap();
    let mut f = fixture(&budget, 127);
    let before = budget.used_bytes();
    let q = norm13::resources::<&Budget>().unwrap();
    let mut calls = 0;
    assert!(norm13::issue_reserved_with(
        mask(&mut f),
        budget.reserve(q.combined_reservation_bytes).unwrap(),
        |v| {
            calls += 1;
            // Original nonce32, GPU pair32, then first D42 additive draw8.
            if calls == 3 {
                assert_eq!(v.len(), 8);
                return Err(Error::Randomness);
            }
            assert_eq!(v.len(), 32);
            v.fill(0x61);
            Ok(())
        }
    )
    .is_err());
    assert_eq!(calls, 3);
    assert_eq!(budget.used_bytes(), before);
    assert!(Norm13MaskLoan::take(&f.source, &mut f.ledgers[0], &f.masks).is_err());
}

#[test]
fn last_frame_and_output_keep_original_table_and_counter() {
    let budget = OwnedBudget::new(8 << 20).unwrap();
    let base = budget.budget().used_bytes();
    let context = budget.reserve_native(0).unwrap().budget.share();
    let mut f = fixture(context, 8191);
    let q = norm13::resources::<OwnedBudgetContext>().unwrap();
    let (m, [mut a, mut b]) = norm13::issue_reserved_with(
        mask(&mut f),
        budget.reserve_native(q.combined_reservation_bytes).unwrap(),
        |v| {
            v.fill(0x53);
            Ok(())
        },
    )
    .unwrap();
    let fa = a.begin(value(&mut f, 0)).unwrap();
    let fb = b.begin(value(&mut f, 1)).unwrap();
    let norm13::Step::Frame(ga) = a.advance(fb.as_bytes()).unwrap() else {
        panic!("select")
    };
    let norm13::Step::Frame(gb) = b.advance(fa.as_bytes()).unwrap() else {
        panic!("select")
    };
    let norm13::Step::Output(va) = a.advance(gb.as_bytes()).unwrap() else {
        panic!("value")
    };
    let norm13::Step::Output(vb) = b.advance(ga.as_bytes()).unwrap() else {
        panic!("value")
    };
    assert_eq!(
        va.values()[0].wrapping_sub(m.values()[0]) & super::super::super::mask(50),
        0
    );
    drop(f);
    drop(a);
    drop(b);
    drop(m);
    drop(ga);
    drop(gb);
    drop(fb);
    drop(vb);
    assert!(budget.budget().used_bytes() > base + 65536);
    drop(va);
    assert!(budget.budget().used_bytes() > base + 65536);
    drop(fa);
    assert_eq!(budget.budget().used_bytes(), base);
}

#[test]
fn same_counter_equal_table_foreign_creator_is_not_input_authority() {
    let budget = Budget::new(8 << 20).unwrap();
    let mut original = fixture(&budget, 128);
    let mut foreign = fixture(&budget, 128);
    assert_eq!(
        original.source.table.digest(),
        foreign.source.table.digest()
    );
    assert!(!Arc::ptr_eq(&original.source, &foreign.source));
    let q = norm13::resources::<&Budget>().unwrap();
    let (_m, [mut a, _b]) = norm13::issue_reserved_with(
        mask(&mut original),
        budget.reserve(q.combined_reservation_bytes).unwrap(),
        |v| {
            v.fill(0x62);
            Ok(())
        },
    )
    .unwrap();
    assert!(a.begin(value(&mut foreign, 0)).is_err());
    assert!(a.begin(value(&mut original, 0)).is_err());
}

#[test]
fn evaluator_controls_remain_admitted_after_dealer_masks_drop() {
    let budget = OwnedBudget::new(8 << 20).unwrap();
    let baseline = budget.budget().used_bytes();
    let context = budget.reserve_native(0).unwrap().budget.share();
    let mut f = fixture(context, 128);
    let q = norm13::resources::<OwnedBudgetContext>().unwrap();
    let (m, [mut a, mut b]) = norm13::issue_reserved_with(
        mask(&mut f),
        budget.reserve_native(q.combined_reservation_bytes).unwrap(),
        |v| {
            v.fill(0x44);
            Ok(())
        },
    )
    .unwrap();
    let controls = a.evaluator_reserved_bytes();
    assert_eq!(controls, b.evaluator_reserved_bytes());
    assert!(controls > size_of::<Norm13ValueLoan<'_, OwnedBudgetContext>>());
    drop(m); // No dealer guard exists during ANY following evaluator transition.
    assert_eq!(a.evaluator_reserved_bytes(), controls);
    assert_eq!(b.evaluator_reserved_bytes(), controls);
    let fa = a.begin(value(&mut f, 0)).unwrap();
    let fb = b.begin(value(&mut f, 1)).unwrap();
    let norm13::Step::Frame(ga) = a.advance(fb.as_bytes()).unwrap() else {
        panic!("frame")
    };
    let norm13::Step::Frame(gb) = b.advance(fa.as_bytes()).unwrap() else {
        panic!("frame")
    };
    let norm13::Step::Output(va) = a.advance(gb.as_bytes()).unwrap() else {
        panic!("output")
    };
    let norm13::Step::Output(vb) = b.advance(ga.as_bytes()).unwrap() else {
        panic!("output")
    };
    assert_eq!(va.values(), vb.values());
    assert_eq!(a.evaluator_reserved_bytes(), controls);
    assert_eq!(b.evaluator_reserved_bytes(), controls);
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
