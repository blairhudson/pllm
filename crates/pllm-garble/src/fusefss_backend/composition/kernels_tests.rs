use super::super::runtime::{Engine, Event};
use super::*;
use crate::fusefss_backend::{Output, MAX_ALLOCATION_BYTES};
use crate::shared_rescale_reference::Context;
use std::ops::Deref;

#[path = "kernels_energy_tests.rs"]
mod energy;

#[path = "kernels_range_tests.rs"]
mod checked_ranges;

fn binding() -> Binding {
    Binding {
        source_digest: [13; 32],
        contract_digest: [17; 32],
    }
}
fn range(min: i64, max: i64) -> InputRange {
    InputRange { min, max }
}
fn context(plan: &Plan<'_>) -> Context {
    Context {
        plan: plan.digest(),
        session: [21; 32],
        operation: [31; 32],
        first_tensor_index: 0,
    }
}
fn drive<'a, P: Deref<Target = Plan<'a>>>(
    plan: &Plan<'a>,
    ctx: Context,
    a: &mut Engine<'a, P>,
    b: &mut Engine<'a, P>,
) -> [Output<'a>; 2] {
    let (mut ae, mut be) = (a.poll().unwrap(), b.poll().unwrap());
    loop {
        match (ae, be) {
            (Event::NeedMaterial { pc: x }, Event::NeedMaterial { pc: y }) => {
                assert_eq!(x, y);
                let [am, bm] = plan.issue_step(x, ctx).unwrap();
                ae = a.feed(am).unwrap();
                be = b.feed(bm).unwrap();
            }
            (Event::Frame(af), Event::Frame(bf)) => {
                ae = a.advance(bf.as_bytes()).unwrap();
                be = b.advance(af.as_bytes()).unwrap();
            }
            (Event::Output(a), Event::Output(b)) => return [a, b],
            _ => panic!("asymmetric kernel execution"),
        }
    }
}
fn run(plan: &Plan<'_>, inputs: &[i64]) -> Vec<i64> {
    let left: Vec<_> = (0..inputs.len())
        .map(|i| (0x312391027u64 + i as u64 * 7103) & MASK)
        .collect();
    let right: Vec<_> = inputs
        .iter()
        .zip(&left)
        .map(|(&x, &a)| word(x).wrapping_sub(a) & MASK)
        .collect();
    let ctx = context(plan);
    let mut a = plan.runtime(0, ctx, &left).unwrap();
    let mut b = plan.runtime(1, ctx, &right).unwrap();
    let [ao, bo] = drive(plan, ctx, &mut a, &mut b);
    assert!(a.is_consumed() && b.is_consumed());
    assert_eq!(ao.shares().len(), bo.shares().len());
    ao.shares()
        .iter()
        .zip(bo.shares())
        .enumerate()
        .map(|(i, (&x, &y))| {
            assert!(!plan.output_is_xor(i).unwrap());
            let value = x.wrapping_add(y) & MASK;
            if value & (1 << 49) != 0 {
                value as i64 - (1 << 50)
            } else {
                value as i64
            }
        })
        .collect()
}

#[test]
fn contractions_exact_full_accumulation_and_negative_floor() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let ranges = [range(-100, 100); 3];
    let coefficients = [33, -71, 19];
    let (public, meta) =
        public_dot(binding(), &ranges, &coefficients, range(-71, 33), &budget).unwrap();
    assert_eq!(meta.raw_range, Some(range(-12300, 12300)));
    assert_eq!(meta.range, range(-4, 3));
    assert_eq!(public.resources().triples_total, 0);
    for x in [[1, 1, 1], [-1, -1, -1], [100, -100, 100], [-100, 100, -100]] {
        let raw: i128 = x
            .iter()
            .zip(coefficients)
            .map(|(&a, b)| i128::from(a) * i128::from(b))
            .sum();
        assert_eq!(run(&public, &x), [raw.div_euclid(4096) as i64]);
    }
    let fixed: Vec<_> = coefficients.iter().map(|&x| range(x, x)).collect();
    let (private, _) = private_dot(binding(), &ranges, &fixed, &budget).unwrap();
    assert_eq!(private.resources().triples_total, 6);
    assert_eq!(run(&private, &[1, 1, 1, 33, -71, 19]), [-1]);
    let (sum, _) = public_dot(
        binding(),
        &[range(1, 1); 2],
        &[3000, 3000],
        range(3000, 3000),
        &budget,
    )
    .unwrap();
    assert_eq!(run(&sum, &[1, 1]), [1]); // Per-term floor would incorrectly yield zero.
}

#[test]
fn endpoint_proofs_and_rejection_before_allocation() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let half = 1i64 << 49;
    let (p, meta) =
        public_scale_q12(binding(), range(-half, half - 1), 1, range(1, 1), &budget).unwrap();
    assert_eq!(meta.range, range(-(1 << 37), (1 << 37) - 1));
    assert_eq!(run(&p, &[-half]), [-(1 << 37)]);
    assert_eq!(run(&p, &[half - 1]), [(1 << 37) - 1]);
    drop(p);
    assert!(public_scale_q12(binding(), range(-half, -half), -1, range(-1, -1), &budget).is_err());
    assert!(public_dot(
        binding(),
        &[range(half - 1, half - 1); 2],
        &[1, 1],
        range(1, 1),
        &budget
    )
    .is_err());
    assert!(private_dot(
        binding(),
        &[range(half - 1, half - 1)],
        &[range(2, 2)],
        &budget
    )
    .is_err());
    assert!(public_scale_q12(binding(), range(0, 1), 2, range(0, 1), &budget).is_err());
    assert!(public_scale_q12(binding(), range(2, 1), 1, range(1, 1), &budget).is_err());
    assert!(argmax(binding(), &[range(-half, half - 1); 2], &budget).is_err());
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn equality_nonpower_vocabulary_and_stable_argmax() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let (hot, meta) = token_one_hot(binding(), range(0, 4), 5, &budget).unwrap();
    assert_eq!(meta.outputs, 5);
    for id in 0..5 {
        let expected: Vec<_> = (0..5).map(|i| i64::from(i == id)).collect();
        assert_eq!(run(&hot, &[id]), expected);
    }
    assert!(token_one_hot(binding(), range(0, 5), 5, &budget).is_err());
    let (max, meta) = argmax(binding(), &[range(-20, 20); 5], &budget).unwrap();
    assert_eq!(meta.outputs, 1);
    for (x, expected) in [
        ([-5, -5, -7, -5, -6], 0),
        ([-20, 0, 20, 20, 1], 2),
        ([0, 0, 0, 0, 0], 0),
        ([-10, -9, -8, -7, -6], 4),
    ] {
        assert_eq!(run(&max, &x), [expected]);
    }
}

#[test]
fn binding_changes_digest_not_arithmetic_and_caps_stay() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let (a, _) =
        public_scale_q12(binding(), range(-10, 10), 4096, range(4096, 4096), &budget).unwrap();
    let mut bound = binding();
    bound.contract_digest = [18; 32];
    let (b, _) = public_scale_q12(bound, range(-10, 10), 4096, range(4096, 4096), &budget).unwrap();
    assert_ne!(a.digest(), b.digest());
    assert_eq!(run(&a, &[-9]), run(&b, &[-9]));
    bound.source_digest = [0; 32];
    assert!(argmax(bound, &[range(0, 1)], &budget).is_err());
    assert!(token_one_hot(binding(), range(0, 0), MAX_INPUTS + 1, &budget).is_err());
    let wide = vec![range(0, 1); MAX_INPUTS];
    assert!(public_dot(binding(), &wide, &vec![1; MAX_INPUTS], range(1, 1), &budget).is_err());
}

#[test]
fn owned_runtime_exports_outlive_all_plan_owners_and_cancel() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let (plan, _) =
        public_scale_q12(binding(), range(-10, 10), 4096, range(4096, 4096), &budget).unwrap();
    let digest = plan.digest();
    let resource = plan.resources();
    let plan = plan.into_shared().unwrap();
    assert_eq!(plan.digest(), digest);
    assert_eq!(
        plan.resources().combined_reservation_bytes,
        resource.combined_reservation_bytes
    );
    let ctx = context(&plan);
    let mut a = plan.runtime_owned(0, ctx, &[99]).unwrap();
    let mut b = plan
        .runtime_owned(1, ctx, &[word(-7).wrapping_sub(99) & MASK])
        .unwrap();
    let [ao, bo] = drive(&plan, ctx, &mut a, &mut b);
    drop(plan);
    drop((a, b));
    assert_eq!(ao.shares()[0].wrapping_add(bo.shares()[0]) & MASK, word(-7));
    assert_eq!(budget.used_bytes(), 16);
    drop((ao, bo));
    assert_eq!(budget.used_bytes(), 0);
    let (plan, _) = argmax(binding(), &[range(-1, 1); 2], &budget).unwrap();
    let plan = plan.into_shared().unwrap();
    let mut worker = plan.runtime_owned(0, context(&plan), &[0, 0]).unwrap();
    drop(plan);
    assert!(matches!(worker.poll().unwrap(), Event::NeedMaterial { .. }));
    worker.cancel();
    assert!(worker.is_consumed());
    assert!(worker.poll().is_err());
    drop(worker);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn floor_affine_and_d13_rope_exact_independent_integers() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    for shift in [0, 1, 12, 49] {
        let (p, meta) = floor(binding(), range(-9000, 9000), shift, &budget).unwrap();
        for x in [-9000, -4097, -1, 0, 1, 4097, 9000] {
            assert_eq!(
                run(&p, &[x]),
                [i128::from(x).div_euclid(1i128 << shift) as i64]
            );
        }
        assert_eq!(
            meta.range.min,
            (-9000i128).div_euclid(1i128 << shift) as i64
        );
    }
    assert!(floor(binding(), range(0, 0), 50, &budget).is_err());
    let (p, _) = public_affine(
        binding(),
        range(-9000, 9000),
        -3101,
        range(-4096, 4096),
        -71,
        &budget,
    )
    .unwrap();
    for x in [-9000, -1, 0, 1, 9000] {
        assert_eq!(
            run(&p, &[x]),
            [(i128::from(x) * -3101).div_euclid(4096) as i64 - 71]
        );
    }
    let (p, _) = public_add(binding(), range(-8, 8), -12, &budget).unwrap();
    assert_eq!(run(&p, &[-8]), [-20]);
    let (p, _) = add(binding(), [range(-8, 8), range(-3, 3)], &budget).unwrap();
    assert_eq!(run(&p, &[-8, 3]), [-5]);
    let (p, _) = rope(binding(), [range(-100, 100); 2], 3000, -2001, &budget).unwrap();
    for (l, r) in [(1, 1), (-1, 1), (100, -100), (-99, -67)] {
        let f = |x: i64, c: i64| (i128::from(x) * i128::from(c)).div_euclid(4096) as i64;
        assert_eq!(
            run(&p, &[l, r]),
            [f(l, 3000) + f(-r, -2001), f(r, 3000) + f(l, -2001)]
        );
    }
    // Full contraction would give floor((3000+2001)/4096)=1, not zero.
    // Subtraction after flooring would also give one, not zero.
    // Right output is -1 rather than floor((3000-2001)/4096)=0.
    assert_eq!(run(&p, &[1, 1]), [0, -1]);
    let half = 1i64 << 49;
    assert!(public_add(binding(), range(half - 1, half - 1), 1, &budget).is_err());
    assert!(add(binding(), [range(-half, -half), range(-1, -1)], &budget).is_err());
    assert!(rope(binding(), [range(half - 1, half - 1); 2], 2, 1, &budget).is_err());
}

#[test]
fn rope_signed_term_witness_and_coefficient_negation_rejects_before_allocation() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let (p, meta) = rope(
        binding(),
        [range(339, 339), range(1165, 1165)],
        2213,
        3447,
        &budget,
    )
    .unwrap();
    // Frozen independent oracle: 183 + floor(-4015755/4096) = -798.
    // The rejected post-floor subtraction would give 183 - 980 = -797.
    assert_eq!(run(&p, &[339, 1165]), [-798, 914]);
    assert_eq!(meta.range, range(-798, 914));
    drop(p);
    budget
        .fail_after
        .store(0, std::sync::atomic::Ordering::SeqCst);
    for sine in [i64::MIN, -(1i64 << 49), 1i64 << 49] {
        assert!(matches!(
            rope(binding(), [range(0, 0); 2], 0, sine, &budget),
            Err(Error::Domain)
        ));
        assert_eq!(budget.used_bytes(), 0);
    }
    budget
        .fail_after
        .store(-1, std::sync::atomic::Ordering::SeqCst);
}

#[test]
fn equality_and_complete_vocabulary_chunk_tails() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let (eq, _) = equality(binding(), range(0, 50256), 50255, &budget).unwrap();
    for x in [0, 50254, 50255, 50256] {
        assert_eq!(run(&eq, &[x]), [i64::from(x == 50255)]);
    }
    let (outside, _) = equality(binding(), range(0, 4), 5, &budget).unwrap();
    assert_eq!(run(&outside, &[4]), [0]);
    let (tail, meta) =
        token_one_hot_chunk(binding(), range(0, 50256), 50257, 50253..50257, &budget).unwrap();
    assert_eq!(meta.outputs, 4);
    assert_eq!(run(&tail, &[50256]), [0, 0, 0, 1]);
    assert_eq!(run(&tail, &[50254]), [0, 1, 0, 0]);
    assert_eq!(run(&tail, &[0]), [0, 0, 0, 0]);
    // Every row of a non-power-of-two vocabulary is covered across chunk tails.
    for id in 0..7 {
        let mut output = Vec::new();
        for rows in [0..3, 3..6, 6..7] {
            let (p, _) = token_one_hot_chunk(binding(), range(0, 6), 7, rows, &budget).unwrap();
            output.extend(run(&p, &[id]));
        }
        assert_eq!(
            output,
            (0..7).map(|i| i64::from(i == id)).collect::<Vec<_>>()
        );
    }
    assert!(token_one_hot_chunk(binding(), range(0, 7), 7, 0..1, &budget).is_err());
    assert!(token_one_hot_chunk(binding(), range(0, 6), 7, 6..8, &budget).is_err());
    assert!(token_one_hot_chunk(binding(), range(0, 6), 7, 2..2, &budget).is_err());
    assert!(equality(binding(), range(-1, 1), 0, &budget).is_err());
}

#[test]
fn stable_max_uses_lowest_arbitrary_index_on_equal_scores() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let (p, meta) = stable_max(binding(), [range(-20, 20); 2], range(0, 50256), &budget).unwrap();
    assert_eq!(meta.outputs, 2);
    for (inputs, expected) in [
        ([7, 90, 7, 2], [7, 2]),
        ([7, 2, 7, 90], [7, 2]),
        ([-7, 50256, -7, 13], [-7, 13]),
        ([-9, 1, -8, 99], [-8, 99]),
        ([20, 91, -20, 0], [20, 91]),
        ([0, 7, 0, 7], [0, 7]),
    ] {
        assert_eq!(run(&p, &inputs), expected);
    }
    assert!(stable_max(binding(), [range(0, 1); 2], range(-1, 2), &budget).is_err());
    assert!(stable_max(
        binding(),
        [range(-(1 << 49), (1 << 49) - 1); 2],
        range(0, 2),
        &budget
    )
    .is_err());
}

#[test]
fn frozen_gelu_actual_fresh_protected_outputs_and_partition_bounds() {
    let vectors: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../../paper/reproduction-contracts/fusefss/vectors.json"
    ))
    .unwrap();
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let (p, meta) = gelu(binding(), range(-(1 << 37), (1 << 37) - 1), &budget).unwrap();
    for (x, y) in vectors["gelu_inputs_signed_q12"]
        .as_array()
        .unwrap()
        .iter()
        .zip(vectors["gelu_expected_signed_q12"].as_array().unwrap())
    {
        let x = x.as_i64().unwrap();
        let y = y.as_i64().unwrap();
        if !(-(1 << 37)..1 << 37).contains(&x) {
            continue;
        }
        assert!(meta.range.min <= y && y <= meta.range.max);
        assert_eq!(run(&p, &[x]), [y]);
    }
    for (x, expected) in [(-(1 << 37), 0), ((1 << 37) - 1, (1 << 37) - 1)] {
        assert_eq!(run(&p, &[x]), [expected]);
    }
    drop(p);
    for x in [-20000, -4096, -1, 0, 1, 4096, 20000] {
        let (point, meta) = gelu(binding(), range(x, x), &budget).unwrap();
        let result = run(&point, &[x])[0];
        assert_eq!(meta.range, range(result, result));
    }
    assert!(gelu(binding(), range(-(1 << 37) - 1, 0), &budget).is_err());
    assert!(gelu(binding(), range(0, 1 << 37), &budget).is_err());
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn new_kernel_allocation_failures_refund_and_owned_exports_survive() {
    for kernel in 0..7 {
        let mut success = false;
        for fail in 0..16 {
            let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
            budget
                .fail_after
                .store(fail, std::sync::atomic::Ordering::SeqCst);
            let result = match kernel {
                0 => floor(binding(), range(-100, 100), 12, &budget),
                1 => equality(binding(), range(0, 10), 7, &budget),
                2 => token_one_hot_chunk(binding(), range(0, 50256), 50257, 50253..50257, &budget),
                3 => stable_max(binding(), [range(-10, 10); 2], range(0, 50256), &budget),
                4 => public_affine(
                    binding(),
                    range(-10, 10),
                    3000,
                    range(-4096, 4096),
                    71,
                    &budget,
                ),
                5 => add(binding(), [range(-10, 10); 2], &budget),
                _ => rope(binding(), [range(-10, 10); 2], 3000, 2000, &budget),
            };
            match result {
                Ok((p, _)) => {
                    drop(p);
                    success = true;
                }
                Err(e) => assert!(matches!(e, Error::Allocation)),
            }
            assert_eq!(budget.used_bytes(), 0);
            if success {
                break;
            }
        }
        assert!(success);
    }
    let mut reached_success = false;
    for fail in 0..48 {
        let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
        budget
            .fail_after
            .store(fail, std::sync::atomic::Ordering::SeqCst);
        let result = gelu(binding(), range(-4096, 4096), &budget);
        match result {
            Ok((p, _)) => {
                drop(p);
                reached_success = true;
            }
            Err(e) => assert!(matches!(e, Error::Allocation)),
        }
        assert_eq!(budget.used_bytes(), 0);
        if reached_success {
            break;
        }
    }
    assert!(reached_success);
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let (p, _) = gelu(binding(), range(20000, 20000), &budget).unwrap();
    let p = p.into_shared().unwrap();
    let ctx = context(&p);
    let mut a = p.runtime_owned(0, ctx, &[71]).unwrap();
    let mut b = p
        .runtime_owned(1, ctx, &[word(20000).wrapping_sub(71) & MASK])
        .unwrap();
    let [ao, bo] = drive(&p, ctx, &mut a, &mut b);
    drop((p, a, b));
    assert_eq!(
        ao.shares()[0].wrapping_add(bo.shares()[0]) & MASK,
        word(20000)
    );
    assert_eq!(budget.used_bytes(), 16);
    drop((ao, bo));
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn stable_selection_wrong_role_and_failed_material_burn_registers() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let (p, _) = stable_max(binding(), [range(-10, 10); 2], range(0, 99), &budget).unwrap();
    let ctx = context(&p);
    let mut a = p.runtime(0, ctx, &[0, 0, 0, 0]).unwrap();
    let Event::NeedMaterial { pc } = a.poll().unwrap() else {
        panic!("missing predicate");
    };
    let [am, bm] = p.issue_step(pc, ctx).unwrap();
    assert!(matches!(a.feed(bm), Err(Error::Context)));
    assert!(a.is_consumed());
    drop(am);
    let mut b = p.runtime(1, ctx, &[0, 0, 0, 0]).unwrap();
    assert!(matches!(b.poll().unwrap(), Event::NeedMaterial { .. }));
    assert!(matches!(
        b.feed_result(Err(Error::Allocation)),
        Err(Error::Allocation)
    ));
    assert!(b.is_consumed());
    drop((a, b));
    drop(p);
    assert_eq!(budget.used_bytes(), 0);
}
