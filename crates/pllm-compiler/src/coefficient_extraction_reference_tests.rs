use super::*;
use owner::{Caps, ClearInputBuilder};
use pllm_core::coefficient_extraction::{
    Destination, KeyDomain, Layout, Limb, Scale, Sign, Source,
};
use pllm_garble::shared_allocation::{ExternalCharge, OwnedBudget};

fn case(n: usize, q: u64) -> Case {
    let limb = Limb {
        id: 101,
        modulus: q,
    };
    let key = KeyDomain {
        label: 33,
        dimension: n,
    };
    let scale = Scale {
        numerator: 7,
        denominator: 3,
    };
    Case {
        source: Source {
            degree: n,
            rank: 1,
            limbs: [Some(limb), None],
            limb_count: 1,
            layout: Layout::OrderedLimbBA,
            sign: Sign::BPlusAS,
            scale,
            key,
        },
        destination: Destination {
            degree: n,
            rank: 1,
            limb,
            layout: Layout::ConstantBASecretOrder,
            sign: Sign::BPlusAS,
            scale,
            key,
        },
        selected: 0,
    }
}
fn caps(words: usize) -> Caps {
    Caps {
        f64_values: 0,
        u64_values: words,
        tensors: 2,
        operations: 0,
    }
}
fn fixture(
    source: &[u64],
    secret: &[u64],
    work: usize,
    extra: usize,
) -> (ClearInputs, OwnedBudget) {
    let caps = caps(source.len() + secret.len());
    let q = ClearInputBuilder::resources(caps).unwrap();
    let budget =
        OwnedBudget::new(q.minimum_fresh_budget_bytes + size_of::<OwnedBudget>() + extra).unwrap();
    let observer = budget.try_clone().unwrap();
    let mut builder = ClearInputBuilder::new(budget, caps, work).unwrap();
    builder
        .add_u64_tensor("source", &[source.len()], source)
        .unwrap();
    builder
        .add_u64_tensor("public_oracle", &[secret.len()], secret)
        .unwrap();
    (builder.finish().unwrap(), observer)
}
fn run(c: Case, source: &[u64], secret: &[u64]) -> (Vec<u64>, (u64, u64)) {
    let (input, observer) = fixture(source, secret, owner::MAX_WORK, 131072);
    let baseline = observer.budget().used_bytes();
    let output = prepay(&input, c, 0, 1).unwrap().extract().unwrap();
    assert!(std::ptr::eq(output.original(), &input));
    let result = (output.words().to_vec(), output.public_relation());
    assert_eq!(result.1 .0, result.1 .1);
    drop(output);
    assert_eq!(observer.budget().used_bytes(), baseline);
    result
}

#[test]
fn exhaustive_small_negacyclic_constant_independent_convolution() {
    for n in [1_usize, 2] {
        let count = 3_usize.pow(n as u32);
        for aa in 0..count {
            for ss in 0..count {
                for b in 0..3_u64 {
                    let mut source = vec![0; 2 * n];
                    let mut secret = vec![0; n];
                    source[0] = b;
                    let (mut a, mut s) = (aa, ss);
                    for i in 0..n {
                        source[n + i] = (a % 3) as u64;
                        secret[i] = (s % 3) as u64;
                        a /= 3;
                        s /= 3;
                    }
                    let mut dense = b as i64;
                    for i in 0..n {
                        for (j, &s) in secret.iter().enumerate() {
                            if (i + j) % n == 0 {
                                let product = source[n + i] as i64 * s as i64;
                                dense += if i + j < n { product } else { -product };
                            }
                        }
                    }
                    let (_, pair) = run(case(n, 3), &source, &secret);
                    assert_eq!(
                        pair,
                        (dense.rem_euclid(3) as u64, dense.rem_euclid(3) as u64)
                    );
                }
            }
        }
    }
}

#[test]
fn bounded_ordered_limbs_and_wide_modulus_execute() {
    for q in [12289, (1_u64 << 61) - 1] {
        let mut c = case(4, q);
        c.source.limbs = [
            Some(Limb {
                id: 7,
                modulus: 65537,
            }),
            c.source.limbs[0],
        ];
        c.source.limb_count = 2;
        c.selected = 1;
        let source = [5, 6, 7, 8, 9, 10, 11, 12, q - 1, 2, 3, 4, q - 2, 2, 4, 8];
        let (words, _) = run(c, &source, &[q - 1, 5, 7, 11]);
        assert_eq!(words, [q - 1, q - 2, q - 8, q - 4, q - 2]);
        assert_eq!(c.destination.key.dimension, 4);
    }
}

#[test]
fn degree_2048_is_explicitly_refused_not_claimed_executed() {
    // Two-limb source plus oracle exceeds unchanged shared u64 ceiling.
    assert!(matches!(
        ClearInputBuilder::resources(caps(5 * 2048)),
        Err(owner::Error::Resource)
    ));
    // One-limb input fits storage, but not construction + original work ceiling.
    let (input, observer) = fixture(&vec![0; 4096], &vec![0; 2048], owner::MAX_WORK, 131072);
    let baseline = observer.budget().used_bytes();
    assert!(quote(&[case(2048, 12289)]).is_ok()); // pure algebra still supports it
    assert!(matches!(
        prepay(&input, case(2048, 12289), 0, 1),
        Err(Error::Work)
    ));
    assert_eq!(observer.budget().used_bytes(), baseline);
}

#[test]
fn asymmetric_sign_order_restriction_is_not_rescale() {
    let (words, pair) = run(case(4, 17), &[8, 2, 3, 4, 1, 2, 4, 8], &[3, 5, 7, 11]);
    assert_eq!(words, [8, 1, 9, 13, 15]);
    assert_ne!(pair.0, (8 + 3 + 2 * 5 + 4 * 7 + 8 * 11) % 17);
    assert_eq!(20 % 17, 3);
    assert_ne!(20 % 17, (20 + 2) / 5);
    assert_ne!(20 % 17, (3 * 7) % 17);
}

#[test]
fn same_original_owner_exhausts_across_output_gaps() {
    let (input, observer) = fixture(&[1, 2], &[2], 9000, 131072);
    let original_work = input.consumed_work();
    let baseline = observer.budget().used_bytes();
    let mut successes = 0;
    for _ in 0..32 {
        let before = input.remaining_work();
        if let Ok(p) = prepay(&input, case(1, 3), 0, 1) {
            let output = p.extract().unwrap();
            successes += 1;
            drop(output);
        }
        assert!(input.remaining_work() <= before);
        assert_eq!(observer.budget().used_bytes(), baseline);
    }
    assert!(successes > 0 && successes < 32);
    assert!(input.consumed_work() > original_work);
    assert!(matches!(
        prepay(&input, case(1, 3), 0, 1),
        Err(Error::Burned)
    ));
}

#[test]
fn preflight_descriptor_failures_burn_all_32_attempts() {
    let (input, _) = fixture(&[1, 2], &[2], owner::MAX_WORK, 131072);
    let mut bad = case(1, 3);
    bad.destination.key.label += 1;
    let before = input.remaining_work();
    for _ in 0..32 {
        assert!(matches!(prepay(&input, bad, 0, 1), Err(Error::Descriptor)));
    }
    assert_eq!(input.remaining_work(), before - 32 * (32 + PREFLIGHT));
    assert!(matches!(
        prepay(&input, case(1, 3), 0, 1),
        Err(Error::Burned)
    ));
}

#[test]
fn cancellation_and_unwind_never_refund_original_work_or_charge() {
    let (input, observer) = fixture(&[1, 2], &[2], owner::MAX_WORK, 131072);
    let baseline = observer.budget().used_bytes();
    let before = input.remaining_work();
    drop(prepay(&input, case(1, 3), 0, 1).unwrap());
    let after_cancel = input.remaining_work();
    assert!(after_cancel < before);
    assert_eq!(observer.budget().used_bytes(), baseline);
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _payment = prepay(&input, case(1, 3), 0, 1).unwrap();
        panic!("cancel retained reservation");
    }));
    assert!(unwound.is_err());
    assert_eq!(
        input.remaining_work(),
        after_cancel - (before - after_cancel)
    );
    assert_eq!(observer.budget().used_bytes(), baseline);
}

#[test]
fn malformed_source_lengths_ids_and_oracle_fail_without_refunds() {
    for (source, secret, ids, expected) in [
        (vec![1], vec![2], (0, 1), Error::Length),
        (vec![1, 2], vec![0, 1], (0, 1), Error::Length),
        (vec![1, 3], vec![2], (0, 1), Error::NonCanonical),
        (vec![1, 2], vec![3], (0, 1), Error::NonCanonical),
        (vec![1, 2], vec![2], (0, 0), Error::Source),
        (vec![1, 2], vec![2], (99, 1), Error::Source),
    ] {
        let (input, observer) = fixture(&source, &secret, owner::MAX_WORK, 131072);
        let baseline = observer.budget().used_bytes();
        let before = input.remaining_work();
        let p = prepay(&input, case(1, 3), ids.0, ids.1).unwrap();
        let paid = input.remaining_work();
        assert!(matches!(p.extract(), Err(e) if e == expected));
        assert!(paid < before);
        assert_eq!(input.remaining_work(), paid);
        assert_eq!(observer.budget().used_bytes(), baseline);
    }
}

#[test]
fn dropped_limb_canonical_validation_remains_atomic() {
    let mut c = case(1, 3);
    c.source.limbs[1] = Some(Limb {
        id: 202,
        modulus: 5,
    });
    c.source.limb_count = 2;
    let (input, observer) = fixture(&[1, 2, 0, 5], &[2], owner::MAX_WORK, 131072);
    let baseline = observer.budget().used_bytes();
    assert!(matches!(
        prepay(&input, c, 0, 1).unwrap().extract(),
        Err(Error::NonCanonical)
    ));
    assert_eq!(observer.budget().used_bytes(), baseline);
}

#[test]
fn zero_negation_composite_ring_and_bad_moduli() {
    let (words, pair) = run(case(4, 15), &[14, 0, 0, 0, 0, 0, 0, 0], &[14; 4]);
    assert_eq!(words, [14, 0, 0, 0, 0]);
    assert_eq!(pair, (14, 14));
    for q in [0, 1, 2, 4, 1_u64 << 61, u64::MAX] {
        assert!(matches!(quote(&[case(4, q)]), Err(Error::Descriptor)));
    }
    assert!(matches!(quote(&[]), Err(Error::Resource)));
    assert!(matches!(quote(&[case(1, 3); 33]), Err(Error::Resource)));
}

#[test]
fn resource_refusal_keeps_work_debit_then_new_attempt_burns_again() {
    let (input, observer) = fixture(&[1, 2], &[2], owner::MAX_WORK, 0);
    let baseline = observer.budget().used_bytes();
    let before = input.remaining_work();
    for attempt in 1..=2 {
        assert!(matches!(
            prepay(&input, case(1, 3), 0, 1),
            Err(Error::Resource)
        ));
        let q = quote(&[case(1, 3)]).unwrap();
        let cost = 32 + PREFLIGHT + q.word_ops + 16 * q.output_words + 32;
        assert_eq!(input.remaining_work(), before - attempt * cost);
        assert_eq!(observer.budget().used_bytes(), baseline);
    }
}

#[test]
fn simultaneous_outputs_share_original_budget_and_admit_only_one_when_full() {
    let q = quote(&[case(1, 3)]).unwrap();
    let payload = q.owner_bytes - 8 * (q.input_words + q.oracle_words)
        + 8 * q.output_words
        + size_of::<Buffers>();
    let phase_bytes = payload
        + size_of::<BorrowedOutput<'static, Buffers>>()
        + size_of::<owner::Prepared<'static>>()
        + size_of::<ExternalCharge<'static>>();
    let (input, observer) = fixture(&[1, 2], &[2], owner::MAX_WORK, phase_bytes);
    let baseline = observer.budget().used_bytes();
    let output = prepay(&input, case(1, 3), 0, 1).unwrap().extract().unwrap();
    assert_eq!(observer.budget().used_bytes(), baseline + phase_bytes);
    assert!(matches!(
        prepay(&input, case(1, 3), 0, 1),
        Err(Error::Resource)
    ));
    assert_eq!(output.words(), &[1, 2]);
    drop(output);
    assert_eq!(observer.budget().used_bytes(), baseline);
    drop(prepay(&input, case(1, 3), 0, 1).unwrap().extract().unwrap());
}

#[test]
fn two_simultaneous_outputs_retain_same_backing_and_no_oracle_reexecution() {
    let (input, observer) = fixture(&[1, 2], &[2], owner::MAX_WORK, 131072);
    let baseline = observer.budget().used_bytes();
    let first = prepay(&input, case(1, 3), 0, 1).unwrap().extract().unwrap();
    let ptr = first.original().u64_tensor(0).unwrap().as_ptr();
    let second = prepay(&input, case(1, 3), 0, 1).unwrap().extract().unwrap();
    assert_eq!(ptr, second.original().u64_tensor(0).unwrap().as_ptr());
    let remaining = input.remaining_work();
    assert_eq!(first.public_relation(), second.public_relation());
    assert_eq!(input.remaining_work(), remaining);
    drop(first);
    assert!(observer.budget().used_bytes() > baseline);
    drop(second);
    assert_eq!(observer.budget().used_bytes(), baseline);
}

#[test]
fn retained_cleanup_wipes_before_free_and_phase_release() {
    let (input, observer) = fixture(&[1, 2], &[2], owner::MAX_WORK, 131072);
    let baseline = observer.budget().used_bytes();
    let output = prepay(&input, case(1, 3), 0, 1).unwrap().extract().unwrap();
    let live = observer.budget().used_bytes();
    let probe_observer = observer.try_clone().unwrap();
    let extra_handle = size_of::<OwnedBudget>();
    let calls = std::rc::Rc::new(std::cell::Cell::new(0));
    let seen = calls.clone();
    CLEANUP_PROBE.with(|probe| {
        *probe.borrow_mut() = Some(Box::new(move |b| {
            // Drop has wiped full initialized capacities, but not freed them yet;
            // original and output phase charges are still live at this exact point.
            assert_eq!(b.scratch.len(), b.scratch.capacity());
            assert_eq!(b.words.len(), b.words.capacity());
            assert!(b.scratch.iter().chain(&b.words).all(|&v| v == 0));
            assert_eq!(b.relation, [0; 2]);
            assert_eq!(probe_observer.budget().used_bytes(), live + extra_handle);
            seen.set(seen.get() + 1);
        }))
    });
    drop(output);
    CLEANUP_PROBE.with(|probe| *probe.borrow_mut() = None);
    assert_eq!(calls.get(), 1);
    assert_eq!(observer.budget().used_bytes(), baseline);
    assert_eq!(&*input.u64_tensor(0).unwrap(), &[1, 2]);
    drop(input);
    assert!(observer.budget().used_bytes() < baseline);
}

#[test]
fn failed_oracle_and_live_output_unwind_clean_before_phase_release() {
    for invalid_oracle in [false, true] {
        let secret = if invalid_oracle { 3 } else { 2 };
        let (input, observer) = fixture(&[1, 2], &[secret], owner::MAX_WORK, 131072);
        let probe_observer = observer.try_clone().unwrap();
        let baseline = observer.budget().used_bytes();
        let calls = std::rc::Rc::new(std::cell::Cell::new(0));
        let seen = calls.clone();
        CLEANUP_PROBE.with(|probe| {
            *probe.borrow_mut() = Some(Box::new(move |b| {
                assert!(b.scratch.iter().chain(&b.words).all(|&v| v == 0));
                assert_eq!(b.relation, [0; 2]);
                assert!(probe_observer.budget().used_bytes() > baseline);
                seen.set(seen.get() + 1);
            }));
        });
        let before = input.remaining_work();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let result = prepay(&input, case(1, 3), 0, 1).unwrap().extract();
            if invalid_oracle {
                assert!(matches!(result, Err(Error::NonCanonical)));
            } else {
                let _output = result.unwrap();
                panic!("cancel live output");
            }
        }));
        CLEANUP_PROBE.with(|probe| *probe.borrow_mut() = None);
        assert_eq!(result.is_err(), !invalid_oracle);
        assert_eq!(calls.get(), 1);
        assert!(input.remaining_work() < before);
        assert_eq!(
            observer.budget().used_bytes(),
            baseline - size_of::<OwnedBudget>()
        );
    }
}

#[test]
fn different_new_paid_owners_legitimately_start_fresh() {
    let (first, _) = fixture(&[1, 2], &[2], owner::MAX_WORK, 131072);
    let (second, _) = fixture(&[1, 2], &[2], owner::MAX_WORK, 131072);
    let before_second = second.remaining_work();
    drop(prepay(&first, case(1, 3), 0, 1).unwrap().extract().unwrap());
    assert_eq!(second.remaining_work(), before_second);
    assert!(first.remaining_work() < second.remaining_work());
    drop(
        prepay(&second, case(1, 3), 0, 1)
            .unwrap()
            .extract()
            .unwrap(),
    );
    assert_eq!(first.remaining_work(), second.remaining_work());
}
