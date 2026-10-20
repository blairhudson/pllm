use super::*;

fn inverse_max(k: usize) -> u64 {
    ((1u128 << 48) * k as u128).isqrt() as u64
}

fn assume<'a>(kind: Kind, lo: i64, hi: i64, gamma: &'a [i64], beta: &'a [i64]) -> Spec<'a> {
    Spec {
        kind,
        input_min: lo,
        input_max: hi,
        gamma,
        beta,
    }
}

#[test]
fn public_certificate_preserves_source_reciprocal_mean_counterexample() {
    let gamma = [4096; 768];
    let beta = [0; 768];
    let spec = assume(Kind::Layer, 4096, 4096, &gamma, &beta);
    let proof = preflight(&spec, inverse_max(768)).unwrap();
    assert_eq!(proof.mean, Interval { lo: 3840, hi: 3840 });
    assert_eq!(proof.centered, Interval { lo: 256, hi: 256 });
    assert_eq!(
        proof.s,
        Interval {
            lo: 50_331_649,
            hi: 50_331_649
        }
    );
    let gamma = [4096; 4];
    let beta = [0; 4];
    let proof = preflight(
        &assume(Kind::Layer, 4096, 4096, &gamma, &beta),
        inverse_max(4),
    )
    .unwrap();
    assert_eq!(proof.mean, Interval { lo: 4096, hi: 4096 });
    assert_eq!(proof.centered, Interval { lo: 0, hi: 0 });
    assert_eq!(proof.s, Interval { lo: 1, hi: 1 });
    let proof = preflight(&assume(Kind::Rms, 0, 0, &gamma, &beta), inverse_max(4)).unwrap();
    assert_eq!(proof.mean, Interval { lo: 0, hi: 0 });
    assert_eq!(proof.s, Interval { lo: 1, hi: 1 });
}

#[test]
fn public_certificate_rejects_each_unwrapped_obligation() {
    let h = HALF as i64;
    let gamma = [4096; 4];
    let beta = [0; 4];
    let reject = |spec: Spec<'_>| {
        assert!(matches!(
            preflight(&spec, inverse_max(spec.gamma.len().max(1))),
            Err(Error::Domain)
        ))
    };
    reject(assume(Kind::Layer, 1, -1, &gamma, &beta));
    reject(assume(Kind::Layer, -h - 1, 0, &gamma, &beta));
    reject(assume(Kind::Layer, 0, h, &gamma, &beta));
    reject(assume(Kind::Layer, 0, h / 2, &gamma, &beta)); // sum
    reject(assume(Kind::Layer, 0, h / 4096, &[4096], &[0])); // mean product
    reject(assume(Kind::Layer, -h, h - 1, &[4096], &[0])); // centering / product
    reject(assume(Kind::Rms, 1 << 25, 1 << 25, &[4096], &[0])); // square
    reject(assume(Kind::Rms, 1 << 24, 1 << 24, &[4096; 2], &[0; 2])); // square sum
    reject(assume(Kind::Rms, 1, 1, &[h - 1], &[0])); // second product
    reject(assume(Kind::Rms, 1, 1, &[4096], &[h - 1])); // beta add
    reject(assume(Kind::Rms, 0, 0, &[h], &[0])); // public gamma
    reject(assume(Kind::Rms, 0, 0, &[4096], &[-h - 1])); // public beta
    reject(assume(Kind::Rms, 0, 0, &[4096], &[]));
    reject(assume(Kind::Layer, 0, 0, &[], &[]));
    // Tight add-one boundary: the sum fits signed50, but +1 does not.
    assert_eq!(Interval::checked(HALF - 1, HALF - 1).unwrap().hi, HALF - 1);
    assert!(Interval::checked(HALF, HALF).is_err());
    // Checked i128 endpoint products, including both signs.
    assert!(Interval { lo: -1, hi: 1 }.scale(i128::MAX).is_err());
    assert!(Interval {
        lo: i128::MIN,
        hi: 0
    }
    .scale(-1)
    .is_err());
}

#[test]
fn rejected_norm_has_no_table_or_builder_reservation() {
    let budget = Budget::new(64 * 1024 * 1024).unwrap();
    let spec = assume(Kind::Layer, 0, HALF as i64 - 1, &[4096], &[0]);
    assert!(matches!(
        compile([5; 32], &spec, &budget),
        Err(Error::Domain)
    ));
    assert_eq!(budget.used_bytes(), 0);
    let spec = assume(Kind::Rms, 0, 0, &[4096; 4], &[0; 4]);
    assert!(matches!(
        compile([0; 32], &spec, &budget),
        Err(Error::Domain)
    ));
    assert_eq!(budget.used_bytes(), 0);
    let tiny = Budget::new(1).unwrap();
    assert!(matches!(
        compile([5; 32], &spec, &tiny),
        Err(Error::Resource)
    ));
    assert_eq!(tiny.used_bytes(), 0);
}

#[test]
fn complete_norm_width_is_subject_to_the_unchanged_instruction_ceiling() {
    let budget = Budget::new(64 * 1024 * 1024).unwrap();
    let gamma = [4096; 4096];
    let beta = [0; 4096];
    for kind in [Kind::Layer, Kind::Rms] {
        // Numeric proof and source table admit zero rows; complete schedules
        // exceed 32768 instructions before any protected material is issued.
        let spec = assume(kind, 0, 0, &gamma, &beta);
        preflight(&spec, inverse_max(4096)).unwrap();
        assert!(matches!(
            compile([5; 32], &spec, &budget),
            Err(Error::Resource)
        ));
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn all_source_width_table_slots_preserved_including_unreachable_exponents() {
    let budget = Budget::new(64 * 1024 * 1024).unwrap();
    for k in [1, 4, 768, 4096] {
        let table = super::super::source_tables::rsqrt(k, &budget).unwrap();
        assert_eq!(table.len(), 8192);
        assert_eq!(table[0], inverse_max(k));
        for (idx, &actual) in table.iter().enumerate() {
            let m = (idx / 64) as u128;
            let e = idx % 64;
            // Independent floor-sqrt certificate, no floating point or table copy.
            let n = ((1u128 << 55) * k as u128) / ((128 + m) * (1u128 << e));
            let t = actual as u128;
            assert!(t * t <= n && (t + 1) * (t + 1) > n, "k={k}, idx={idx}");
            assert!(actual <= 1u64 << 30);
        }
    }
    assert_eq!(budget.used_bytes(), 0);
}

/// Independent oracle: locate containing power-of-two interval, then integer
/// long division. No production extraction or table lookup is called here.
fn oracle_index(s: u64) -> (u64, u64, u64) {
    assert!((1..=MASK).contains(&s));
    let e = (0u64..50).rev().find(|&e| s >= 1u64 << e).unwrap();
    let m = (((s as u128) * 128) / (1u128 << e)) as u64 - 128;
    (e, m, 64 * m + e)
}

fn edge_samples() -> Vec<u64> {
    let mut samples = vec![1, MASK];
    for e in 0u32..50 {
        let p = 1u64 << e;
        for s in [p.saturating_sub(1), p, p + 1] {
            if (1..=MASK).contains(&s) {
                samples.push(s);
            }
        }
        // Every realizable mantissa bin and both sides of its carry boundary.
        for m in 0u128..128 {
            let start = (((128 + m) << e) + 127) / 128;
            for s in [start.saturating_sub(1), start, start + 1] {
                if s > 0 && s <= MASK as u128 {
                    samples.push(s as u64);
                }
            }
        }
    }
    samples.sort_unstable();
    samples.dedup();
    samples
}

#[test]
fn fixed_schedule_algebra_covers_all_exponent_edges_and_mantissa_carries() {
    // This is ring-algebra evidence, not a protected runtime execution claim.
    // Runtime tests below exercise the actual Builder separately.
    for s in edge_samples() {
        let mut result = (0u64, (s.wrapping_mul(128).wrapping_sub(128)) & MASK);
        for e in 1u8..50 {
            let scaled = if e <= 7 {
                s.wrapping_mul(1 << (7 - e)) & MASK
            } else {
                s >> (e - 7)
            };
            let candidate = scaled.wrapping_sub(128) & MASK;
            let bit = u64::from(s < 1u64 << e);
            // Canonical ring select; unused candidates may wrap.
            result.0 =
                (e as u64).wrapping_add(bit.wrapping_mul(result.0.wrapping_sub(e as u64))) & MASK;
            result.1 =
                candidate.wrapping_add(bit.wrapping_mul(result.1.wrapping_sub(candidate))) & MASK;
        }
        assert_eq!(
            (result.0, result.1, 64 * result.1 + result.0),
            oracle_index(s),
            "S={s}"
        );
    }
    assert_eq!(oracle_index(1u64 << 49), (49, 0, 49));
    assert_eq!(oracle_index(MASK), (49, 127, 8177));
}

/// Test-only two-role driver: issue fresh native owners for every requested
/// step, exchange actual masked frames, reconstruct only final test shares.
fn execute(plan: &Plan<'_>, inputs: &[u64]) -> Vec<u64> {
    use super::super::runtime::Event;
    use crate::shared_rescale_reference::Context;
    let context = Context {
        plan: [91; 32],
        session: [92; 32],
        operation: [93; 32],
        first_tensor_index: 0,
    };
    let left: Vec<_> = inputs
        .iter()
        .enumerate()
        .map(|(i, _)| ((i as u64 + 19).wrapping_mul(0x9e3779b97f4a7c15)) & MASK)
        .collect();
    let right: Vec<_> = inputs
        .iter()
        .zip(&left)
        .map(|(&v, &l)| v.wrapping_sub(l) & MASK)
        .collect();
    let mut a = plan.runtime(0, context, &left).unwrap();
    let mut b = plan.runtime(1, context, &right).unwrap();
    let (mut ea, mut eb) = (a.poll().unwrap(), b.poll().unwrap());
    loop {
        (ea, eb) = match (ea, eb) {
            (Event::NeedMaterial { pc: pa }, Event::NeedMaterial { pc: pb }) => {
                assert_eq!(pa, pb);
                let [ma, mb] = plan.issue_step(pa, context).unwrap();
                (a.feed(ma).unwrap(), b.feed(mb).unwrap())
            }
            (Event::Frame(fa), Event::Frame(fb)) => {
                let next = (
                    a.advance(fb.as_bytes()).unwrap(),
                    b.advance(fa.as_bytes()).unwrap(),
                );
                drop((fa, fb));
                next
            }
            (Event::Output(oa), Event::Output(ob)) => {
                assert_eq!(oa.shares().len(), ob.shares().len());
                return oa
                    .shares()
                    .iter()
                    .zip(ob.shares())
                    .map(|(&l, &r)| l.wrapping_add(r) & MASK)
                    .collect();
            }
            _ => panic!("public role schedules diverged"),
        };
    }
}

#[test]
fn protected_unsigned_index_including_top_bit_and_carry() {
    let budget = Budget::new(64 * 1024 * 1024).unwrap();
    {
        let plan = compile_index([5; 32], &budget).unwrap();
        for s in [1, 127, 128, 129, (1 << 49) - 1, 1 << 49, MASK] {
            let actual = execute(&plan, &[s]);
            let (e, m, i) = oracle_index(s);
            assert_eq!(actual, [e, m, i], "S={s}");
        }
    }
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn protected_frozen_source_norm_cases() {
    struct Case {
        kind: Kind,
        row: Vec<i64>,
        gamma: Vec<i64>,
        beta: Vec<i64>,
        fields: [i64; 6],
        output: Vec<i64>,
    }
    let cases = [
        Case {
            kind: Kind::Layer,
            row: vec![4096; 4],
            gamma: vec![4096; 4],
            beta: vec![0; 4],
            fields: [4096, 1, 0, 0, 0, 33_554_432],
            output: vec![0; 4],
        },
        Case {
            kind: Kind::Layer,
            row: vec![4096; 768],
            gamma: vec![4096; 768],
            beta: vec![0; 768],
            fields: [3840, 50_331_649, 25, 64, 4121, 65_536],
            output: vec![4096; 768],
        },
        Case {
            kind: Kind::Layer,
            row: vec![-4096, -4096, 4096, 4096],
            gamma: vec![4096; 4],
            beta: vec![0; 4],
            fields: [0, 67_108_865, 26, 0, 26, 4096],
            output: vec![-4096, -4096, 4096, 4096],
        },
        Case {
            kind: Kind::Rms,
            row: vec![0; 4],
            gamma: vec![4096; 4],
            beta: vec![0; 4],
            fields: [0, 1, 0, 0, 0, 33_554_432],
            output: vec![0; 4],
        },
        Case {
            kind: Kind::Rms,
            row: vec![4096; 4],
            gamma: vec![4096; 4],
            beta: vec![0; 4],
            fields: [0, 67_108_865, 26, 0, 26, 4096],
            output: vec![4096; 4],
        },
        Case {
            kind: Kind::Layer,
            row: vec![-65, 64],
            gamma: vec![-4096, 8192],
            beta: vec![1, -1],
            fields: [-1, 8322, 13, 2, 141, 260_119],
            output: vec![4066, 8253],
        },
    ];
    let budget = Budget::new(64 * 1024 * 1024).unwrap();
    for case in cases {
        let spec = assume(
            case.kind,
            *case.row.iter().min().unwrap(),
            *case.row.iter().max().unwrap(),
            &case.gamma,
            &case.beta,
        );
        let plan = compile_fixture([5; 32], &spec, &budget).unwrap();
        let inputs: Vec<_> = case.row.iter().map(|&x| word(x)).collect();
        let actual = execute(&plan, &inputs);
        let expected: Vec<_> = case
            .fields
            .iter()
            .chain(&case.output)
            .map(|&x| word(x))
            .collect();
        assert_eq!(
            actual,
            expected,
            "kind={:?}, k={}",
            case.kind,
            case.row.len()
        );
        drop(plan);
        assert_eq!(budget.used_bytes(), 0);
    }
}

fn row_energy_oracle(row: &[i64], gamma: &[i64], beta: &[i64]) -> Vec<i64> {
    let energy: u128 = row
        .iter()
        .map(|&x| i128::from(x).unsigned_abs().pow(2))
        .sum();
    let s = energy + 1;
    assert!(s < HALF as u128);
    let e = (0..49).rev().find(|&e| (1u128 << e) <= s).unwrap();
    let j = 128 * s / (1u128 << e);
    let n = ((1u128 << 55) * row.len() as u128) / (j * (1u128 << e));
    let (mut lo, mut hi) = (0u128, 1u128 << 30);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if mid * mid <= n {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let mut output = vec![
        0,
        s as i64,
        e as i64,
        (j - 128) as i64,
        (64 * (j - 128) + e) as i64,
        lo as i64,
    ];
    for ((&x, &g), &b) in row.iter().zip(gamma).zip(beta) {
        let raw = i128::from(x) * lo as i128;
        let q = raw.div_euclid(4096);
        let affine = q * i128::from(g);
        let y = affine.div_euclid(4096) + i128::from(b);
        for v in [raw, affine, y] {
            assert!((-HALF..HALF).contains(&v));
        }
        output.push(y as i64);
    }
    output
}

#[test]
fn protected_row_energy_rms_admits_loose_intervals_with_both_original_floors() {
    let budget = Budget::new(64 * 1024 * 1024).unwrap();
    for (row, gamma, beta) in [
        (vec![2], vec![8192], vec![0]),
        (
            vec![1 << 24, -2, 17, 0],
            vec![4097, 8193, -8192, 5001],
            vec![1, -7, 2, 0],
        ),
    ] {
        let spec = assume(Kind::Rms, -(1 << 25), 1 << 25, &gamma, &beta);
        let energy = row
            .iter()
            .map(|&x| i128::from(x).unsigned_abs().pow(2))
            .sum();
        assert!(matches!(
            compile([5; 32], &spec, &budget),
            Err(Error::Domain)
        ));
        let expected = row_energy_oracle(&row, &gamma, &beta);
        if row.len() == 1 {
            // Combining the two floors would give 7327 instead.
            assert_eq!(expected[6], 7326);
        }
        let plan = compile_inner([5; 32], &spec, &budget, true, None, Some(energy)).unwrap();
        let words: Vec<_> = row.iter().map(|&x| word(x)).collect();
        assert_eq!(
            execute(&plan, &words),
            expected.iter().map(|&x| word(x)).collect::<Vec<_>>()
        );
        drop(plan);
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn row_energy_rejects_unproved_inputs_wrap_affine_and_layer_before_allocation() {
    use super::super::cache::{Limits, SourceTableCache};
    use std::sync::atomic::Ordering;
    let budget = Budget::new(64 * 1024 * 1024).unwrap();
    let cache = SourceTableCache::new(
        [7; 32],
        Limits {
            max_entries: 2,
            max_retained_bytes: 1024 * 1024,
        },
        &budget,
    )
    .unwrap();
    let baseline = budget.used_bytes();
    let h = HALF as i64;
    let max_code = [h - 1];
    let invalid_high = [h];
    let invalid_low = [-h - 1];
    let cases = [
        (assume(Kind::Layer, -1, 1, &[4096], &[0]), 1),
        (assume(Kind::Rms, 0, 0, &[], &[]), 0),
        (assume(Kind::Rms, 0, 1, &[4096], &[]), 1),
        (assume(Kind::Rms, 2, 1, &[4096], &[0]), 4),
        (assume(Kind::Rms, -h - 1, 1, &[4096], &[0]), 1),
        (assume(Kind::Rms, 0, h, &[4096], &[0]), 0),
        (assume(Kind::Rms, 1, 2, &[4096; 4], &[0; 4]), 3),
        (
            assume(Kind::Rms, -(1 << 25), 1 << 25, &[4096], &[0]),
            HALF as u128 - 1,
        ),
        (assume(Kind::Rms, 1 << 25, 1 << 25, &[4096], &[0]), 1 << 50),
        (assume(Kind::Rms, 0, 1, &[4096], &[0]), u128::MAX),
        (assume(Kind::Rms, 0, h / 2, &[4096; 4], &[0; 4]), 0),
        (assume(Kind::Rms, 1, 1, &max_code, &[0]), 1),
        (assume(Kind::Rms, 1, 1, &[4096], &max_code), 1),
        (assume(Kind::Rms, 0, 0, &invalid_high, &[0]), 0),
        (assume(Kind::Rms, 0, 0, &[4096], &invalid_low), 0),
    ];
    budget.fail_after.store(0, Ordering::SeqCst);
    for (spec, energy) in cases {
        assert!(matches!(
            compile_cached_with_row_energy([5; 32], &spec, energy, &cache),
            Err(Error::Domain)
        ));
        assert_eq!(budget.used_bytes(), baseline);
        assert_eq!(cache.stats().references, 0);
    }
    let spec = assume(Kind::Rms, -(1 << 25), 1 << 25, &[4096], &[0]);
    let proof = preflight_inner(&spec, inverse_max(1), Some(HALF as u128 - 2)).unwrap();
    assert_eq!(proof.s.hi, HALF - 1);
    assert!(matches!(
        compile_cached_with_row_energy([0; 32], &spec, 1, &cache),
        Err(Error::Domain)
    ));
    let too_wide = assume(Kind::Rms, 0, 0, &[4096; 4097], &[0; 4097]);
    assert!(matches!(
        compile_cached_with_row_energy([5; 32], &too_wide, 0, &cache),
        Err(Error::Domain)
    ));
    assert_eq!(budget.used_bytes(), baseline);
    drop(cache);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn row_energy_proof_pins_reject_substitution_and_cached_plan_owns_its_lifetime() {
    use super::super::{
        cache::{Limits, SourceTableCache},
        runtime::Event,
    };
    use crate::shared_rescale_reference::Context;
    let budget = Budget::new(64 * 1024 * 1024).unwrap();
    let cache = SourceTableCache::new(
        [7; 32],
        Limits {
            max_entries: 2,
            max_retained_bytes: 1024 * 1024,
        },
        &budget,
    )
    .unwrap();
    let spec = assume(Kind::Rms, -2, 2, &[8192], &[0]);
    let plain = compile_cached([5; 32], &spec, &cache).unwrap();
    let tight = compile_cached_with_row_energy([5; 32], &spec, 4, &cache).unwrap();
    let loose = compile_cached_with_row_energy([5; 32], &spec, 8, &cache).unwrap();
    let same = compile_cached_with_row_energy([5; 32], &spec, 4, &cache).unwrap();
    assert_ne!(plain.digest(), tight.digest());
    assert_ne!(tight.digest(), loose.digest());
    assert_eq!(tight.digest(), same.digest());
    for plan in [&tight, &loose, &same] {
        assert_eq!(plan.source, plain.source);
        assert_eq!(format!("{:?}", plan.code), format!("{:?}", plain.code));
        assert_eq!(plan.tables[0].digest(), plain.tables[0].digest());
        assert_eq!(
            plan.resources().private_material_bytes_total,
            plain.resources().private_material_bytes_total
        );
        assert_eq!(
            plan.resources().frames_total,
            plain.resources().frames_total
        );
    }
    assert_eq!(cache.stats().table_constructions, 1);
    let context = Context {
        plan: [91; 32],
        session: [92; 32],
        operation: [93; 32],
        first_tensor_index: 0,
    };
    let mut receiver = loose.runtime(0, context, &[word(2)]).unwrap();
    let pc = match receiver.poll().unwrap() {
        Event::NeedMaterial { pc } => pc,
        _ => panic!("first source product"),
    };
    let [left, right] = tight.issue_step(pc, context).unwrap();
    assert!(matches!(receiver.feed(left), Err(Error::Context)));
    drop(receiver);
    drop((right, plain, loose, same, cache));
    let expected = row_energy_oracle(&[2], &[8192], &[0]);
    assert_eq!(
        execute(&tight, &[word(2)]),
        expected[6..].iter().map(|&x| word(x)).collect::<Vec<_>>()
    );
    drop(tight);
    assert_eq!(budget.used_bytes(), 0);
}
