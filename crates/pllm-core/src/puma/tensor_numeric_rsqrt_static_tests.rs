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

// Provenance: own-repository b2ce7ce6^:crates/pllm-core/src/puma/tensor_numeric.rs,
// rsqrt_ranged and the ordinary (partitioned=false, input_norm=None) RMS recipe.
// Retain the old literal coefficients and operation order independently of the
// current coefficient table, rsqrt_ranged and rsqrt_static::finish. Ranged is
// deliberately shared primitive interval-box arithmetic, not the recipe oracle.
fn pre_b2_rsqrt<S: attention_work::SoftmaxSink>(
    b: &mut S, input: Value, bounds: Interval,
) -> Result<(Ranged, Ranged), Error> {
    let x = Ranged::input(input, bounds, 35)?;
    let highest = b.highest_bit(input, 35)?;
    let z = b.bit_map(highest, BitMap::ShiftLeft(1))?;
    let reversed = b.bit_map(z, BitMap::ReverseLowBits(36))?;
    let factor = b.from_boolean(reversed)?;
    let factor = x.exact_map(b, factor, |j| 1 << (34 - j))?;
    let u = x.product(b, &factor, true)?;
    let u2 = u.product(b, &u, true)?;
    let u3 = u2.product(b, &u, true)?;
    let u4 = u3.product(b, &u, true)?;
    let mut raw = u.affine(b, None, affine(-4057974, 0, 0))?;
    for (power, coefficient) in [(&u2, 10085068), (&u3, -13072088), (&u4, 6823457)] {
        raw = raw.affine(b, Some(power), affine(1, coefficient, 0))?;
    }
    // One rescale AFTER accumulating all four terms; then the old interval-box
    // affine transfer, without a static or partitioned polynomial refinement.
    let p = raw.rescale(b, 18)?.affine(b, None, affine(1, 0, 1086023))?;
    let parity = b.bit_map(z, BitMap::Parity(0x5555_5555_5555_5555))?;
    let parity = b.from_bit(parity)?;
    let parity = x.exact_map(b, parity, |j| i64::from((j + 1) % 2 == 0))?;
    let coefficient = parity.affine(b, None, affine(212, 0, 512))?;
    let folded = b.bit_map(z, BitMap::FoldPairs)?;
    let reversed = b.bit_map(folded, BitMap::ReverseLowBits(18))?;
    let scale = b.from_boolean(reversed)?;
    let scale = x.exact_map(b, scale, |j| 1 << (17 - (j + 1) / 2))?;
    let compensation = coefficient.product(b, &scale, false)?;
    Ok((p.product(b, &compensation, true)?, p))
}

// Independently retained ORDINARY RMS wrapper. The callback substitutes only
// the rsqrt under comparison; neither path calls today's rms_normalization or
// rms_work_tests::original_outer (which itself calls today's normalization).
fn pre_b2_ordinary_rms(
    rows: usize, width: usize, input: Interval, gamma: &[i64],
    inverse_recipe: impl FnOnce(&mut Builder, Value, Interval) -> Result<Ranged, Error>,
) -> Result<Program, Error> {
    let lanes = rows.checked_mul(width).ok_or(Error::Bounds)?;
    if rows == 0 || width == 0 || gamma.len() != width {
        return Err(Error::Shape);
    }
    let mut b = Builder::row(lanes, input)?;
    b.0.numeric_profile = RMS_NORM_PROFILE;
    let x = b.input();
    let square = b.square(x)?;
    let sum = b.row_map(square, width, RowMap::Sum)?;
    let inverse_width = ((1.0_f32 / width as f32) * 262144.0_f32) as i64;
    if inverse_width <= 0 {
        return Err(Error::Bounds);
    }
    let raw_mean = b.affine(sum, None, affine(inverse_width, 0, 0))?;
    let mean = b.rescale(raw_mean, 18)?;
    let denominator = b.affine(mean, None, affine(1, 0, 1))?;
    let bounds = b.node(denominator, false)?.bounds;
    let mut inverse = inverse_recipe(&mut b, denominator, bounds)?;
    inverse.value = b.row_map(inverse.value, width, RowMap::Broadcast)?;
    let denominator_boxes = Ranged::input(denominator, bounds, 35)?.boxes;
    let mut boxes = [None; DOMAINS];
    for (j, bucket) in denominator_boxes.into_iter().enumerate() {
        if let Some(bucket) = bucket {
            // d=TR(S*c)+1 => S*c < d*Q; x_i^2 < (S+1)*Q.
            // Preserve correlation with the SAME original denominator bucket.
            let sum_max = (i128::from(bucket.upper()) * 262144 - 1)
                / i128::from(inverse_width);
            let square_max = u128::try_from((sum_max + 1) * 262144 - 1)
                .map_err(|_| Error::Bounds)?;
            let magnitude = i64::try_from(square_max.isqrt()).map_err(|_| Error::Bounds)?;
            let lower = input.lower().max(-magnitude);
            let upper = input.upper().min(magnitude);
            if lower > upper {
                inverse.boxes[j] = None;
            } else {
                boxes[j] = Some(Interval::new(lower, upper)?);
            }
        }
    }
    let normalized = Ranged { value: x, boxes }.product(&mut b, &inverse, true)?.value;
    let scales: [Affine; MAX_ROW_LANES] =
        std::array::from_fn(|i| affine(gamma[i % width], 0, 0));
    let scaled = b.affine_lanes(normalized, &scales[..lanes])?;
    let output = b.rescale(scaled, 18)?;
    b.finish(output)
}

fn assert_pre_b2_identity(a: &Program, b: &Program) {
    // Node equality includes complete instructions, bounds and certified raw
    // product envelopes, not just operation kinds or an arithmetic digest.
    assert_eq!(a.nodes, b.nodes);
    assert_eq!((a.len, a.lanes), (b.len, b.lanes));
    assert_eq!(a.public_affines, b.public_affines);
    assert_eq!(a.public_affine_count, b.public_affine_count);
    assert_eq!(a.numeric_profile, b.numeric_profile);
    assert_eq!(a.checked_validity, b.checked_validity);
    assert_eq!(a.bit_widths, b.bit_widths);
    assert_eq!(a.digest, b.digest);
    assert_eq!(a.work, b.work);
    assert_eq!(a.last_use, b.last_use);
    assert_eq!(a.slots, b.slots);
    assert_eq!(a.register_count, b.register_count);
}

fn assert_same_arithmetic_schedule(a: &Program, b: &Program) {
    assert_eq!((a.len, a.lanes), (b.len, b.lanes));
    assert_eq!(a.numeric_profile, b.numeric_profile);
    assert_eq!(a.work, b.work);
    assert_eq!(a.last_use, b.last_use);
    assert_eq!(a.slots, b.slots);
    assert_eq!(a.register_count, b.register_count);
    assert_eq!(a.bit_widths, b.bit_widths);
    assert_eq!(a.public_affines, b.public_affines);
    assert_eq!(a.public_affine_count, b.public_affine_count);
    for (a, b) in a.nodes[..a.len].iter().zip(&b.nodes[..b.len]) {
        let (a, b) = (a.unwrap(), b.unwrap());
        assert_eq!((a.lanes, a.bit), (b.lanes, b.bit));
        match (a.instruction, b.instruction) {
            (Instruction::Product { left: al, right: ar, rescale: at, certified: ac },
             Instruction::Product { left: bl, right: br, rescale: bt, certified: bc }) => {
                assert_eq!((al, ar, at, ac.is_some()), (bl, br, bt, bc.is_some()));
            }
            (a, b) => assert_eq!(a, b),
        }
    }
}

#[test]
fn legacy_rsqrt_identity_matches_independent_pre_b2_recipe() {
    let actual_denominator = Interval::new(1, 262145).unwrap();
    let mut domains = vec![actual_denominator, Interval::new(1, (1_i64 << 35) - 1).unwrap()];
    for j in 0..35 {
        let lo = 1_i64 << j;
        let hi = (1_i64 << (j + 1)) - 1;
        domains.push(Interval::new(lo, hi).unwrap());
        for d in [lo, lo + (hi - lo) / 2, hi] {
            domains.push(Interval::new(d, d).unwrap());
        }
        if j > 0 {
            domains.push(Interval::new(lo - 1, lo).unwrap());
        }
    }
    for domain in domains {
        let mut b = Builder::row(1, domain).unwrap();
        let x = b.input();
        let (inverse, polynomial) = pre_b2_rsqrt(&mut b, x, domain).unwrap();
        let polynomial_node = polynomial.value.0 as usize;
        let old = b.finish(inverse.value).unwrap();
        let legacy_seam = build::<false>(domain, false);
        assert_pre_b2_identity(&old, &legacy_seam);
        let refined = build::<true>(domain, false);
        assert_same_arithmetic_schedule(&old, &refined);
        if domain == actual_denominator {
            assert_eq!(polynomial.boxes[17], Some(Interval::new(-1920005, 2815012).unwrap()));
            let first = (0..old.len).find(|&i| old.nodes[i] != refined.nodes[i]).unwrap();
            // Zero-based rsqrt-only node 29, NOT RMS node 34 or a claim
            // about the first divergence of an entire compiler schedule.
            assert_eq!((first, polynomial_node), (29, 29));
            assert_eq!(old.nodes[first].unwrap().bounds, Interval::new(-1920005, 2815012).unwrap());
            assert_eq!(refined.nodes[first].unwrap().bounds, Interval::new(370598, 524410).unwrap());
            assert_eq!(old.nodes[first].unwrap().instruction, refined.nodes[first].unwrap().instruction);
            assert_ne!(old.digest, refined.digest);
        }
        // The ordinary public default preserves the independent legacy identity;
        // the tighter enclosure remains an explicit compile-time experiment.
        let public = Program::rsqrt(1, domain).unwrap();
        assert_pre_b2_identity(&public, &old);
        if domain == actual_denominator {
            assert_ne!(public.digest, refined.digest);
        }
    }
    for domain in [Interval::new(0, 1).unwrap(), Interval::new(1, 1_i64 << 35).unwrap()] {
        let mut b = Builder::row(1, domain).unwrap();
        let x = b.input();
        assert!(matches!(pre_b2_rsqrt(&mut b, x, domain), Err(Error::Bounds)));
        assert!(Program::rsqrt(1, domain).is_err());
    }
    for (rows, width, input, gamma) in [
        (1, 4, Interval::new(-262144, 262144).unwrap(), vec![262144; 4]),
        (2, 3, Interval::new(-40960, 40960).unwrap(), vec![262144, -131072, 0]),
        (1, 2, Interval::new(-40960, 40960).unwrap(), vec![262144; 2]),
    ] {
        let oracle = pre_b2_ordinary_rms(rows, width, input, &gamma,
            |b, x, d| pre_b2_rsqrt(b, x, d).map(|(inverse, _)| inverse)).unwrap();
        let seam = pre_b2_ordinary_rms(rows, width, input, &gamma,
            |b, x, d| rsqrt_ranged::<_, false>(b, x, d, false)).unwrap();
        assert_pre_b2_identity(&oracle, &seam);
        let refined = pre_b2_ordinary_rms(rows, width, input, &gamma,
            |b, x, d| rsqrt_ranged::<_, true>(b, x, d, false)).unwrap();
        assert_same_arithmetic_schedule(&oracle, &refined);
        // Ordinary RMS preserves the complete independent legacy identity.
        let public = Program::rms_norm(rows, width, input, &gamma).unwrap();
        assert_pre_b2_identity(&public, &oracle);
        assert_ne!(public.digest, refined.digest);
        if rows == 1 && width == 2 && input == Interval::new(-40960, 40960).unwrap()
            && gamma == vec![262144; 2]
        {
            let historical = oracle.output_bounds();
            let explicit_true = refined.output_bounds();
            assert_eq!(oracle.output_bounds(), Interval::new(-2811009, 2811009).unwrap());
            assert_eq!(public.output_bounds(), Interval::new(-2811009, 2811009).unwrap());
            assert!(explicit_true.lower() > historical.lower());
            assert!(explicit_true.upper() < historical.upper());
            println!("legacy_rsqrt_compiler_private_fixture_output_interval: historical=[{}, {}] explicit_true=[{}, {}]",
                historical.lower(), historical.upper(), explicit_true.lower(), explicit_true.upper());
        }
        if width == 4 {
            assert_eq!(oracle.nodes[5].unwrap().bounds, actual_denominator);
            let first = (0..oracle.len).find(|&i| oracle.nodes[i] != refined.nodes[i]).unwrap();
            assert_eq!(first, 34); // This ordinary RMS wrapper only.
            assert_eq!(oracle.nodes[first].unwrap().bounds, Interval::new(-1920005, 2815012).unwrap());
            assert_eq!(refined.nodes[first].unwrap().bounds, Interval::new(370598, 524410).unwrap());
        }
    }
}
