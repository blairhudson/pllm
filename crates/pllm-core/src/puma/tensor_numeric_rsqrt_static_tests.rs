use super::*;

fn choices(value: i128) -> [i128; 2] {
    let q = i128::from(ONE);
    let floor = value.div_euclid(q);
    [floor, floor + i128::from(value.rem_euclid(q) != 0)]
}

fn polynomial_choices(u: i128, mut visit: impl FnMut(i128)) {
    for u2 in choices(u * u) {
        for u3 in choices(u2 * u) {
            for u4 in choices(u3 * u) {
                let a = RSQRT_COEFFICIENTS.map(i128::from);
                let raw = a[1] * u + a[2] * u2 + a[3] * u3 + a[4] * u4;
                for p in choices(raw) {
                    visit(p + a[0]);
                }
            }
        }
    }
}

#[test]
fn bernstein_controls_and_rounding_radius_are_exact_encoded_rationals() {
    // Independently specified rational controls, not used by the derivation.
    for (i, (n, d)) in [
        (134207745_i128, 256_i128),
        (117567898, 256),
        (325502540, 768),
        (100757960, 256),
        (94914064, 256),
    ].into_iter().enumerate() {
        assert_eq!(CONTROLS[i] * d, n * DENOMINATOR);
    }
    assert_eq!(ERROR, 160);
    assert_eq!(LIMITS, (370598, 524410));
    assert!(std::panic::catch_unwind(|| mul(i128::MAX, 2)).is_err());
    assert!(std::panic::catch_unwind(|| add(i128::MAX, 1)).is_err());
    assert!(std::panic::catch_unwind(|| exact_div(1, 2)).is_err());
}

#[test]
fn static_enclosure_contains_every_normalized_integer_and_rounding_choice() {
    for u in ONE / 4..=ONE / 2 {
        polynomial_choices(i128::from(u), |p| {
            assert!((i128::from(LIMITS.0)..=i128::from(LIMITS.1)).contains(&p));
        });
    }
}

#[test]
fn transfer_intersects_and_refuses_invalid_or_disjoint_domains() {
    let normal = Interval::new(ONE / 4, ONE / 2).unwrap();
    assert_eq!(transfer(Interval::new(-4 * ONE, 4 * ONE).unwrap(), normal).unwrap(),
        Interval::new(LIMITS.0, LIMITS.1).unwrap());
    let shifted = Interval::new(
        LIMITS.0 + 2 - RSQRT_COEFFICIENTS[0],
        LIMITS.1 - 2 - RSQRT_COEFFICIENTS[0],
    ).unwrap();
    assert_eq!(transfer(shifted, normal).unwrap(),
        Interval::new(LIMITS.0 + 2, LIMITS.1 - 2).unwrap());
    for invalid in [
        Interval::new(0, ONE / 4 - 1).unwrap(),
        Interval::new(ONE / 2 + 1, ONE).unwrap(),
    ] {
        assert_eq!(transfer(shifted, invalid), Err(Error::Bounds));
    }
    assert_eq!(transfer(Interval::new(0, 0).unwrap(), normal), Err(Error::Bounds));
    assert!(Interval::new(1, 0).is_err());
}

fn build<const STATIC: bool>(range: Interval, partitioned: bool) -> Program {
    let mut b = Builder::row(1, range).unwrap();
    let input = b.input();
    let output = rsqrt_ranged::<_, STATIC>(&mut b, input, range, partitioned)
        .unwrap().value;
    b.finish(output).unwrap()
}

#[test]
fn original_exponent_normalization_compensation_and_domain_boundaries() {
    for j in 0..35 {
        let lo = 1_i64 << j;
        let hi = (1_i64 << (j + 1)) - 1;
        for d in [lo, lo + (hi - lo) / 2, hi] {
            let program = build::<true>(Interval::new(d, d).unwrap(), false);
            let output = program.output_bounds();
            let factor = 1_i128 << (34 - j);
            let compensation = (512_i128 + if (j + 1) % 2 == 0 { 212 } else { 0 })
                * (1_i128 << (17 - (j + 1) / 2));
            for u in choices(i128::from(d) * factor) {
                assert!((i128::from(ONE / 4)..=i128::from(ONE / 2)).contains(&u));
                polynomial_choices(u, |p| {
                    for inverse in choices(p * compensation) {
                        assert!((i128::from(output.lower())..=i128::from(output.upper()))
                            .contains(&inverse));
                    }
                });
            }
        }
    }
    for range in [Interval::new(0, 1).unwrap(), Interval::new(1, 1 << 35).unwrap()] {
        let mut b = Builder::row(1, range).unwrap();
        let input = b.input();
        assert!(matches!(rsqrt_ranged::<_, true>(&mut b, input, range, false), Err(Error::Bounds)));
    }
}

#[test]
fn ordinary_instruction_constants_and_native_work_unchanged_certified_path_identical() {
    let range = Interval::new(1, (1_i64 << 35) - 1).unwrap();
    let old = build::<false>(range, false);
    let new = build::<true>(range, false);
    assert_eq!(old.len, new.len);
    assert_eq!(old.numeric_profile, new.numeric_profile);
    assert_eq!(old.work, new.work);
    assert_eq!(old.last_use, new.last_use);
    assert_eq!(old.slots, new.slots);
    for (a, b) in old.nodes[..old.len].iter().zip(&new.nodes[..new.len]) {
        let (a, b) = (a.unwrap(), b.unwrap());
        assert_eq!((a.lanes, a.bit), (b.lanes, b.bit));
        match (a.instruction, b.instruction) {
            (Instruction::Product { left: al, right: ar, rescale: at, certified: ac },
             Instruction::Product { left: bl, right: br, rescale: bt, certified: bc }) => {
                // Certified product intervals are public bound metadata, not ops.
                assert_eq!((al, ar, at), (bl, br, bt));
                assert_eq!(ac.is_some(), bc.is_some());
            }
            (a, b) => assert_eq!(a, b),
        }
    }
    let old = build::<false>(range, true);
    let new = build::<true>(range, true);
    assert_eq!(old.nodes, new.nodes);
    assert_eq!(old.digest, new.digest);
    assert_eq!(old.work, new.work);
}

#[test]
fn ordinary_and_certified_work_queries_agree_with_factories_without_private_energy() {
    let input = Interval::new(-40960, 40960).unwrap();
    let gamma = [ONE; 4];
    let private = Program::rms_norm(1, 4, input, &gamma).unwrap();
    let mut workspace = RmsWorkWorkspace::empty();
    assert_eq!(Program::rms_norm_work(1, 4, input, &gamma, &mut workspace).unwrap(),
        private.work);
    let (_, private_energy) = Program::rms_norm_proof(1, 4, input, &gamma,
        false, None, false).unwrap();
    assert_eq!(private_energy, 0);
    let (public, energy) = Program::rms_norm_certified(1, 4, input, &gamma).unwrap();
    assert!(energy > 0);
    assert_eq!(Program::rms_norm_certified_work(1, 4, input, &gamma, &mut workspace).unwrap(),
        public.work);
    assert!(Program::rms_norm(1, 0, input, &[]).is_err());
}
