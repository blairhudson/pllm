use super::*;

fn full() -> InputRange {
    range(-(1 << 49), (1 << 49) - 1)
}

fn oracle(inputs: &[i64], accepted: InputRange, fallback: RangeFallback) -> Vec<i64> {
    let valid = inputs
        .iter()
        .all(|&x| (accepted.min..=accepted.max).contains(&x));
    let mut out: Vec<_> = inputs
        .iter()
        .map(|&x| {
            if (accepted.min..=accepted.max).contains(&x)
                && (fallback == RangeFallback::PerInput || valid)
            {
                x
            } else {
                0
            }
        })
        .collect();
    out.push(i64::from(valid));
    out
}

fn split(inputs: &[i64], mask: u64) -> [Vec<u64>; 2] {
    let left: Vec<_> = (0..inputs.len())
        .map(|i| mask.wrapping_add(173 * i as u64) & MASK)
        .collect();
    let right = inputs
        .iter()
        .zip(&left)
        .map(|(&x, &a)| word(x).wrapping_sub(a) & MASK)
        .collect();
    [left, right]
}

#[derive(Debug, Default, PartialEq, Eq)]
struct Trace {
    material: Vec<usize>,
    frames: Vec<(usize, usize)>,
}

// Only the test boundary reconstructs. This helper returns native shared owners
// so validity can flow through multiple guarded kernels without intermediate opens.
fn execute<'a>(plan: &Plan<'a>, inputs: [&[u64]; 2]) -> ([Output<'a>; 2], Trace) {
    let ctx = context(plan);
    let mut a = plan.runtime(0, ctx, inputs[0]).unwrap();
    let mut b = plan.runtime(1, ctx, inputs[1]).unwrap();
    let mut trace = Trace::default();
    let (mut ae, mut be) = (a.poll().unwrap(), b.poll().unwrap());
    loop {
        match (ae, be) {
            (Event::NeedMaterial { pc }, Event::NeedMaterial { pc: other }) => {
                assert_eq!(pc, other);
                trace.material.push(pc);
                let [am, bm] = plan.issue_step(pc, ctx).unwrap();
                ae = a.feed(am).unwrap();
                be = b.feed(bm).unwrap();
            }
            (Event::Frame(af), Event::Frame(bf)) => {
                trace
                    .frames
                    .push((af.as_bytes().len(), bf.as_bytes().len()));
                ae = a.advance(bf.as_bytes()).unwrap();
                be = b.advance(af.as_bytes()).unwrap();
            }
            (Event::Output(a), Event::Output(b)) => return ([a, b], trace),
            _ => panic!("secret-dependent schedule"),
        }
    }
}

fn reconstruct(outputs: &[Output<'_>; 2]) -> Vec<i64> {
    outputs[0]
        .shares()
        .iter()
        .zip(outputs[1].shares())
        .map(|(&a, &b)| signed_word(a.wrapping_add(b) & MASK) as i64)
        .collect()
}

#[test]
fn checked_range_full_signed50_boundaries_order_rotation_and_public_fallback() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let h = HALF as i64;
    for accepted in [
        range(-7, 11),
        range(-h, 0),
        range(0, h - 1),
        full(),
        range(0, 0),
    ] {
        for fallback in [RangeFallback::PerInput, RangeFallback::WholeRow] {
            let (plan, meta) =
                checked_range_with_fallback(binding(), &[full(); 3], accepted, fallback, &budget)
                    .unwrap();
            assert_eq!(meta.outputs, 4);
            assert_eq!(meta.range, range(accepted.min, accepted.max.max(1)));
            for value in [-h, -h + 1, -8, -7, -1, 0, 1, 11, 12, h - 2, h - 1] {
                let inputs = [value, 0, accepted.max];
                let expected = oracle(&inputs, accepted, fallback);
                for mask in [0, 1, HALF as u64, MASK] {
                    let [a, b] = split(&inputs, mask);
                    let (outputs, _) = execute(&plan, [&a, &b]);
                    assert_eq!(
                        reconstruct(&outputs),
                        expected,
                        "{accepted:?} {fallback:?} {inputs:?} mask={mask}"
                    );
                    for i in 0..4 {
                        assert!(!plan.output_is_xor(i).unwrap());
                    }
                }
            }
        }
    }
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn checked_range_checks_actual_ring_input_without_trusting_declared_membership() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let (plan, _) = checked_range(binding(), &[range(0, 0)], range(-3, 3), &budget).unwrap();
    for x in [-(1 << 49), -4, -3, 0, 3, 4, (1 << 49) - 1] {
        assert_eq!(
            run(&plan, &[x]),
            oracle(&[x], range(-3, 3), RangeFallback::PerInput)
        );
    }
}

#[test]
fn checked_range_rms_bound_sanitizes_before_squaring_and_preserves_accepted_codes() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    for width in [1, 4, 896, 4096] {
        let b = (((1u128 << 49) - 2) / width).isqrt();
        assert!(1 + width * b * b < 1 << 49);
        assert!(width * (b + 1) * (b + 1) > (1 << 49) - 2);
    }
    let width = 4;
    let b = (((1u128 << 49) - 2) / width as u128).isqrt() as i64;
    let (plan, _) = checked_range(binding(), &[full(); 4], range(-b, b), &budget).unwrap();
    for inputs in [
        [b, -b, b, -b],
        [b + 1, -b - 1, b, -b],
        [-(1 << 49), (1 << 49) - 1, 0, 1],
    ] {
        let out = run(&plan, &inputs);
        assert_eq!(out, oracle(&inputs, range(-b, b), RangeFallback::PerInput));
        let mut sum = 0i128;
        for &x in &out[..width] {
            let square = i128::from(x) * i128::from(x);
            assert!(square < HALF);
            sum += square;
            assert!(sum < HALF);
        }
        assert!(sum + 1 < HALF);
    }
}

#[test]
fn checked_range_uses_complete_identical_schedule_and_standard_resource_owners() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let (plan, _) = checked_range(binding(), &[full(); 3], range(-2, 2), &budget).unwrap();
    let resources = plan.resources();
    assert_eq!(resources.instructions, 8 * 3 + 3);
    assert_eq!(resources.secret_steps, 7 * 3);
    assert_eq!(resources.public_table_reservation_bytes, 0);
    assert_eq!(resources.public_table_copy_bytes_total, 0);
    assert!(resources.private_material_bytes_total > 0);
    assert!(resources.dcf_keys_total > 0);
    let mut previous = None;
    for inputs in [[1, 0, -2], [3, 0, -2], [1, 3, -2], [1, 0, -3], [3, 3, 3]] {
        let [a, b] = split(&inputs, MASK - 31);
        let (outputs, trace) = execute(&plan, [&a, &b]);
        assert_eq!(
            reconstruct(&outputs),
            oracle(&inputs, range(-2, 2), RangeFallback::PerInput)
        );
        assert_eq!(trace.material.len(), resources.secret_steps);
        assert_eq!(trace.frames.len() * 2, resources.frames_total);
        assert_eq!(
            trace.frames.iter().map(|(a, b)| a + b).sum::<usize>(),
            resources.frame_bytes_total
        );
        if let Some(previous) = &previous {
            assert_eq!(&trace, previous);
        }
        previous = Some(trace);
    }
    drop(plan);
    assert_eq!(budget.used_bytes(), 0);
    // Real initial RMS width, public compilation only. No checkpoint or giant VM.
    let (wide, meta) = checked_range(binding(), &[full(); 896], range(-100, 100), &budget).unwrap();
    assert_eq!(meta.outputs, 897);
    assert_eq!(wide.resources().secret_steps, 7 * 896);
    assert!(wide.resources().combined_reservation_bytes <= MAX_ALLOCATION_BYTES);
    drop(wide);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn checked_range_sticky_secret_and_and_final_unscaled_token_mask() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let (guard, _) = checked_range(binding(), &[full(); 2], range(-1, 1), &budget).unwrap();
    let (and, and_meta) = validity_and(binding(), &budget).unwrap();
    let (mask, mask_meta) = mask_by_validity(binding(), &[range(0, 151935); 2], &budget).unwrap();
    assert_eq!(and_meta.range, range(0, 1));
    assert_eq!(and.resources().triples_total, 2);
    assert_eq!(mask.resources().triples_total, 4);
    assert_eq!(mask_meta.range, range(0, 151935));
    for a in 0..=1 {
        for b in 0..=1 {
            assert_eq!(run(&and, &[a, b]), [a & b]);
        }
    }
    for sequence in [
        [true, true, true],
        [false, true, true],
        [true, false, true],
        [true, true, false],
    ] {
        let mut valid = [MASK - 17, 19]; // Sum mod 2^50 is 1; local shares aren't bits.
        for accepted in sequence {
            let [left, right] = split(&[if accepted { 1 } else { 2 }, -1], 0x817231);
            let (guarded, _) = execute(&guard, [&left, &right]);
            let (aggregate, _) = execute(
                &and,
                [
                    &[valid[0], guarded[0].shares()[2]],
                    &[valid[1], guarded[1].shares()[2]],
                ],
            );
            valid = [aggregate[0].shares()[0], aggregate[1].shares()[0]];
        }
        let [mut left, mut right] = split(&[17, 151935], MASK - 20);
        left.push(valid[0]);
        right.push(valid[1]);
        let (tokens, _) = execute(&mask, [&left, &right]);
        let expected_valid = sequence.into_iter().all(|x| x);
        assert_eq!(
            (valid[0].wrapping_add(valid[1])) & MASK,
            u64::from(expected_valid)
        );
        assert_eq!(
            reconstruct(&tokens),
            if expected_valid {
                vec![17, 151935]
            } else {
                vec![0, 0]
            }
        );
    }
    drop((guard, and, mask));
    assert_eq!(budget.used_bytes(), 0);
    let (mask, meta) = mask_by_validity(binding(), &[full(); 2], &budget).unwrap();
    assert_eq!(meta.range, full());
    for valid in [0, 1] {
        assert_eq!(
            run(&mask, &[-(1 << 49), (1 << 49) - 1, valid]),
            [-(1 << 49) * valid, ((1 << 49) - 1) * valid]
        );
    }
}

#[test]
fn checked_range_public_spec_errors_and_fallible_allocation_refund() {
    use std::sync::atomic::Ordering;
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    budget.fail_after.store(0, Ordering::SeqCst);
    for accepted in [
        range(1, 2),
        range(-2, -1),
        range(1, 0),
        range(-(1 << 49) - 1, 0),
        range(0, 1 << 49),
    ] {
        assert!(matches!(
            checked_range(binding(), &[full()], accepted, &budget),
            Err(Error::Domain)
        ));
    }
    assert!(matches!(
        checked_range(binding(), &[], range(0, 0), &budget),
        Err(Error::Domain)
    ));
    assert!(matches!(
        checked_range(binding(), &[range(1, 0)], range(0, 0), &budget),
        Err(Error::Domain)
    ));
    let wide = vec![full(); MAX_INPUTS + 1];
    assert!(matches!(
        checked_range(binding(), &wide, range(0, 0), &budget),
        Err(Error::Domain)
    ));
    assert!(matches!(
        mask_by_validity(binding(), &wide, &budget),
        Err(Error::Domain)
    ));
    assert!(matches!(
        mask_by_validity(binding(), &[], &budget),
        Err(Error::Domain)
    ));
    assert!(matches!(
        validity_and(
            Binding {
                source_digest: [0; 32],
                ..binding()
            },
            &budget
        ),
        Err(Error::Domain)
    ));
    assert_eq!(budget.used_bytes(), 0);
    assert_eq!(budget.fail_after.load(Ordering::SeqCst), 0);
    for kernel in 0..4 {
        let mut success = false;
        for fail in 0..40 {
            let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
            budget.fail_after.store(fail, Ordering::SeqCst);
            let result = match kernel {
                0 => checked_range(binding(), &[full(); 2], range(-1, 1), &budget),
                1 => checked_range_with_fallback(
                    binding(),
                    &[full(); 2],
                    range(-1, 1),
                    RangeFallback::WholeRow,
                    &budget,
                ),
                2 => validity_and(binding(), &budget),
                _ => mask_by_validity(binding(), &[full(); 2], &budget),
            };
            match result {
                Ok((plan, _)) => {
                    drop(plan);
                    success = true;
                }
                Err(error) => assert!(matches!(error, Error::Allocation)),
            }
            assert_eq!(budget.used_bytes(), 0);
            if success {
                break;
            }
        }
        assert!(success);
    }
    let tiny = Budget::new(1).unwrap();
    assert!(matches!(
        checked_range(binding(), &[full()], range(-1, 1), &tiny),
        Err(Error::Resource)
    ));
    assert_eq!(tiny.used_bytes(), 0);
}

#[test]
fn checked_range_domain_pins_and_failed_material_import_burn() {
    use crate::fusefss_backend::wire::ImportGuard;
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let (a, _) = checked_range(binding(), &[full(); 2], range(-1, 1), &budget).unwrap();
    let (same, _) = checked_range_with_fallback(
        binding(),
        &[full(); 2],
        range(-1, 1),
        RangeFallback::PerInput,
        &budget,
    )
    .unwrap();
    assert_eq!(a.digest(), same.digest());
    drop(same);
    for variant in 0..4 {
        let (b, _) = match variant {
            0 => checked_range_with_fallback(
                binding(),
                &[full(); 2],
                range(-1, 1),
                RangeFallback::WholeRow,
                &budget,
            ),
            1 => checked_range(binding(), &[full(); 2], range(-2, 1), &budget),
            2 => checked_range(binding(), &[range(-3, 3); 2], range(-1, 1), &budget),
            _ => checked_range(
                Binding {
                    contract_digest: [18; 32],
                    ..binding()
                },
                &[full(); 2],
                range(-1, 1),
                &budget,
            ),
        }
        .unwrap();
        assert_ne!(a.digest(), b.digest());
        let ctx = context(&a);
        let mut receiver = b.runtime(0, ctx, &[0; 2]).unwrap();
        let Event::NeedMaterial { pc } = receiver.poll().unwrap() else {
            panic!("comparison")
        };
        let [material, peer] = a.issue_step(pc, ctx).unwrap();
        assert!(matches!(receiver.feed(material), Err(Error::Context)));
        assert!(receiver.is_consumed());
        drop((receiver, peer));
        let [material, peer] = a.issue_step(pc, ctx).unwrap();
        drop(peer);
        let packet = material.export_packet().unwrap();
        let mut guard = ImportGuard::new_indexed(2, &budget).unwrap();
        assert!(matches!(
            b.import_step_bytes(pc, ctx, 0, packet.nonce(), packet.as_bytes(), &mut guard),
            Err(Error::Context)
        ));
        assert!(matches!(
            a.import_step_bytes(pc, ctx, 0, packet.nonce(), packet.as_bytes(), &mut guard),
            Err(Error::Consumed)
        ));
    }
    drop(a);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn checked_range_cancel_allocation_failure_and_bad_frame_permanently_burn() {
    use std::sync::atomic::Ordering;
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let (plan, _) = checked_range(binding(), &[full()], range(-1, 1), &budget).unwrap();
    let reserved = budget.used_bytes();
    for failure in 0..4 {
        let ctx = context(&plan);
        let mut runtime = plan.runtime(0, ctx, &[MASK]).unwrap();
        let Event::NeedMaterial { pc } = runtime.poll().unwrap() else {
            panic!("comparison")
        };
        match failure {
            0 => runtime.cancel(),
            1 => {
                assert!(matches!(
                    runtime.feed_result(Err(Error::Allocation)),
                    Err(Error::Allocation)
                ));
            }
            2 => {
                budget.fail_after.store(0, Ordering::SeqCst);
                let material = plan.issue_step(pc, ctx);
                budget.fail_after.store(-1, Ordering::SeqCst);
                let error = match material {
                    Err(error) => error,
                    Ok(_) => panic!("expected allocation failure"),
                };
                assert!(matches!(error, Error::Allocation));
                assert!(matches!(
                    runtime.feed_result(Err(error)),
                    Err(Error::Allocation)
                ));
            }
            _ => {
                let [material, peer] = plan.issue_step(pc, ctx).unwrap();
                drop(peer);
                let Event::Frame(frame) = runtime.feed(material).unwrap() else {
                    panic!("frame")
                };
                assert!(runtime.advance(&[]).is_err());
                drop(frame);
            }
        }
        assert!(runtime.is_consumed());
        assert!(matches!(runtime.poll(), Err(Error::Consumed)));
        drop(runtime);
        assert_eq!(budget.used_bytes(), reserved);
    }
    drop(plan);
    assert_eq!(budget.used_bytes(), 0);
}
