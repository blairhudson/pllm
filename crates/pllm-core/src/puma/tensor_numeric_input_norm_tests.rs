//! Public integer obligations only; no trained/native/secret execution.
use super::*;
use num_bigint::BigInt;

const C: i64 = 1 << 26;

// Independent arbitrary-precision binary search, not production isqrt or a
// rounded real norm. Also checks both sides of the floor-square-root boundary.
fn oracle_radius(bound: u128) -> i128 {
    let bound = BigInt::from(bound);
    let (mut lo, mut hi) = (0_i128, 1_i128 << 64);
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if BigInt::from(mid).pow(2) <= bound { lo = mid; } else { hi = mid; }
    }
    assert!(BigInt::from(lo).pow(2) <= bound);
    assert!(BigInt::from(lo + 1).pow(2) > bound);
    lo
}

fn floor_shift(value: BigInt, shift: usize) -> BigInt {
    let divisor = BigInt::from(1) << shift;
    let remainder = &value % &divisor;
    let truncated = &value / &divisor;
    if value < BigInt::from(0) && remainder != BigInt::from(0) {
        truncated - 1
    } else {
        truncated
    }
}

// Independent encoded-integer equations, all truncations choosing floor. No
// production coefficients, range helpers, floats, output measurements or PRSS.
fn floor_rms_oracle(row: &[i64; 2], gamma: &[i64; 2]) -> Vec<BigInt> {
    let sum = row.iter().map(|&x| floor_shift(BigInt::from(x).pow(2), 18))
        .fold(BigInt::from(0), |a, b| a + b);
    let denominator = floor_shift(sum * 131_072_i64, 18) + 1_i64;
    let mut j = 0_usize;
    while (BigInt::from(1) << (j + 1)) <= denominator { j += 1; }
    assert!(j < 35);
    let u = floor_shift(denominator * (BigInt::from(1) << (34 - j)), 18);
    let u2 = floor_shift(&u * &u, 18);
    let u3 = floor_shift(&u2 * &u, 18);
    let u4 = floor_shift(&u3 * &u, 18);
    let polynomial = BigInt::from(1_086_023) + floor_shift(
        &u * -4_057_974_i64 + u2 * 10_085_068_i64 - u3 * 13_072_088_i64 + u4 * 6_823_457_i64, 18);
    let compensation = BigInt::from(if j % 2 == 1 { 724 } else { 512 })
        * (BigInt::from(1) << (17 - (j + 1) / 2));
    let inverse = floor_shift(polynomial * compensation, 18);
    row.iter().zip(gamma).map(|(&x, &g)| {
        floor_shift(floor_shift(BigInt::from(x) * &inverse, 18) * BigInt::from(g), 18)
    }).collect()
}

// Count actual lane-local stochastic sites, including exact divisions. Boolean
// conversion schedules are deliberately irrelevant to this arithmetic count.
fn graph_choice_count(program: &Program) -> usize {
    program.nodes[..program.len].iter().map(|node| {
        let node = node.unwrap();
        match node.instruction {
            Instruction::Product { rescale: true, .. } | Instruction::Rescale { .. } => node.lanes,
            _ => 0,
        }
    }).sum()
}

fn graph_round(value: BigInt, shift: usize, choices: &[bool], used: &mut usize) -> BigInt {
    let ceil = *choices.get(*used).expect("graph exhausted rounding tape");
    *used += 1;
    let inexact = &value % (BigInt::from(1) << shift) != BigInt::from(0);
    floor_shift(value, shift) + i64::from(ceil && inexact)
}

fn floor_graph(program: &Program, row: &[i64; 2]) -> Vec<BigInt> {
    choice_graph(program, row, &vec![false; graph_choice_count(program)])
}

// Mathematical interpretation of the emitted PUBLIC instruction graph, not a
// native/secret runtime. Verify every eager arithmetic interval independently.
fn choice_graph(program: &Program, row: &[i64; 2], choices: &[bool]) -> Vec<BigInt> {
    assert_eq!(choices.len(), graph_choice_count(program));
    let mut used = 0;
    let mut values: Vec<Vec<BigInt>> = Vec::new();
    for node in program.nodes[..program.len].iter().map(|node| node.unwrap()) {
        let get = |value: Value| &values[value.0 as usize];
        let output: Vec<BigInt> = match node.instruction {
            Instruction::Input => row.iter().map(|&x| BigInt::from(x)).collect(),
            Instruction::Affine { left, right, coefficients: c } => get(left).iter().enumerate()
                .map(|(i, x)| x * c.left + right.map_or_else(|| BigInt::from(0), |r| &get(r)[i] * c.right) + c.constant).collect(),
            Instruction::Product { left, right, rescale, certified } => get(left).iter().zip(get(right))
                .map(|(x, y)| {
                    let raw = x * y;
                    if let Some(bounds) = certified {
                        assert!(BigInt::from(bounds.lower()) <= raw && raw <= BigInt::from(bounds.upper()));
                    }
                    if rescale { graph_round(raw, 18, choices, &mut used) } else { raw }
                }).collect(),
            Instruction::Rescale { input, shift } => get(input).iter()
                .map(|x| graph_round(x.clone(), shift as usize, choices, &mut used)).collect(),
            Instruction::Bits(input) | Instruction::FromBit(input) => get(input).clone(),
            Instruction::LowBits(input, width) => get(input).iter()
                .map(|x| x & ((BigInt::from(1) << width as usize) - 1)).collect(),
            Instruction::BitMap(input, map) => get(input).iter().map(|x| {
                let digits = x.to_u64_digits();
                assert!(x >= &BigInt::from(0) && digits.1.len() <= 1);
                let x = digits.1.first().copied().unwrap_or(0);
                BigInt::from(match map {
                    BitMap::LowBits(n) => x & (u64::MAX >> (64 - n)),
                    BitMap::ShiftRight(n) => x >> n,
                    BitMap::ShiftLeft(n) => x << n,
                    BitMap::ReverseLowBits(n) => (0..n).fold(0, |out, i| out | (((x >> i) & 1) << (n - 1 - i))),
                    BitMap::FoldPairs => (0..32).fold(0, |out, i| out | ((((x >> (2 * i)) ^ (x >> (2 * i + 1))) & 1) << i)),
                    BitMap::Parity(mask) => u64::from((x & mask).count_ones() % 2),
                })
            }).collect(),
            Instruction::BitOr(a, b) | Instruction::BitOrLow(a, b, _) => {
                let mask = match node.instruction {
                    Instruction::BitOrLow(_, _, n) => (BigInt::from(1) << n as usize) - 1,
                    _ => BigInt::from(-1),
                };
                get(a).iter().zip(get(b)).map(|(a, b)| (a | b) & &mask).collect()
            }
            Instruction::BitXor(a, b) => get(a).iter().zip(get(b)).map(|(a, b)| a ^ b).collect(),
            Instruction::RowMap(input, width, RowMap::Sum) => get(input).chunks_exact(width)
                .map(|row| row.iter().fold(BigInt::from(0), |a, b| a + b)).collect(),
            Instruction::RowMap(input, width, RowMap::Broadcast) => get(input).iter()
                .flat_map(|x| std::iter::repeat(x.clone()).take(width)).collect(),
            Instruction::LaneAffine(input, table) => get(input).iter()
                .zip(&program.public_affines[table as usize]).map(|(x, c)| {
                    assert_eq!(c.right, 0);
                    x * c.left + c.constant
                }).collect(),
            other => panic!("unexpected RMS instruction: {other:?}"),
        };
        assert_eq!(output.len(), node.lanes);
        if !node.bit {
            for x in &output {
                assert!(BigInt::from(node.bounds.lower()) <= *x && *x <= BigInt::from(node.bounds.upper()), "{node:?}: {x}");
            }
        }
        values.push(output);
    }
    assert_eq!(used, choices.len(), "graph left unused rounding choices");
    values.pop().unwrap()
}

// Separate literal equation oracle: no graph, production constants, range/proof
// helpers or measured-output certificates. This rounding implementation uses
// BigInt arithmetic shift (floor), independently of graph_round/floor_shift.
fn choice_rms_oracle(row: &[i64; 2], gamma: &[i64; 2], choices: &[bool]) -> Vec<BigInt> {
    let mut used = 0;
    let mut tr = |value: BigInt| {
        let ceil = *choices.get(used).expect("equations exhausted rounding tape");
        used += 1;
        let lower = &value >> 18_usize;
        let exact = (&lower << 18_usize) == value;
        lower + i64::from(ceil && !exact)
    };
    let square0 = tr(BigInt::from(row[0]).pow(2));
    let square1 = tr(BigInt::from(row[1]).pow(2));
    let denominator = tr((square0 + square1) * 131_072_i64) + 1_i64;
    let mut j = 0_usize;
    assert!(denominator >= BigInt::from(1));
    while (BigInt::from(1) << (j + 1)) <= denominator { j += 1; }
    assert!(j < 35);
    let u = tr(denominator * (BigInt::from(1) << (34 - j)));
    let u2 = tr(&u * &u);
    let u3 = tr(&u2 * &u);
    let u4 = tr(&u3 * &u);
    let polynomial = BigInt::from(1_086_023) + tr(
        &u * -4_057_974_i64 + u2 * 10_085_068_i64 - u3 * 13_072_088_i64 + u4 * 6_823_457_i64);
    let compensation = BigInt::from(if j % 2 == 1 { 724 } else { 512 })
        * (BigInt::from(1) << (17 - (j + 1) / 2));
    let inverse = tr(polynomial * compensation);
    // Graph order: normalize BOTH lanes before gamma-rescaling either lane.
    let normalized0 = tr(BigInt::from(row[0]) * &inverse);
    let normalized1 = tr(BigInt::from(row[1]) * &inverse);
    let output = vec![tr(normalized0 * gamma[0]), tr(normalized1 * gamma[1])];
    assert_eq!(used, choices.len(), "equations left unused rounding choices");
    output
}

// Obligations are selected BEFORE fixtures. Observed input energy only checks
// membership; it never supplies a production certificate. 23 rows, two signed
// gamma tables: 46 row/table cases, each with every graph-derived choice tape.
fn stochastic_rescue_cases() -> [(u128, Vec<[i64; 2]>); 3] {
    [
        (0, vec![[0, 0]]),
        (2, vec![[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1],
            [1, -1], [1, 0], [1, 1]]),
        (262_144_u128.pow(2), vec![[0, 0], [262_144, 0], [-262_144, 0],
            [0, 262_144], [0, -262_144], [262_143, 1], [-262_143, 1],
            [1, 262_143], [1, -262_143], [1, -1], [-1, 1], [511, -512], [-511, 512]]),
    ]
}

#[test]
fn refined_rms_overflow_rescue_all_stochastic_tapes_bigint_oracle() {
    let incoming = Interval::new(-(1 << 40), 1 << 40).unwrap();
    for (obligation, rows) in stochastic_rescue_cases() {
        for gamma in [[262_144, -262_144], [-524_288, 131_072]] {
            let (native, energy) = Program::rms_norm_with_input_norm(
                1, 2, incoming, &gamma, obligation).unwrap();
            let (compact, compact_energy) = Program::rms_norm_with_input_norm_compact(
                1, 2, incoming, &gamma, obligation).unwrap();
            let count = graph_choice_count(&native);
            assert_eq!(count, graph_choice_count(&compact));
            // Do not silently truncate a changed graph to 8192 tapes.
            assert_eq!(count, 13, "review exhaustive fixture if graph changes");
            let tapes = 1_usize.checked_shl(u32::try_from(count).unwrap()).unwrap();
            let radius = oracle_radius(obligation);
            // E<=B implies each |xi|<=floor(sqrt(B)). Every ceil contributes at
            // most Q-1 numerator units: sum(TR(xi²))<=floor((B+n(Q-1))/Q).
            let sum_ceiling = (BigInt::from(obligation) + 2_i64 * 262_143_i64) / 262_144_i64;
            for program in [&native, &compact] {
                let input = program.nodes[0].unwrap();
                assert_eq!(i128::from(input.bounds.lower()), -radius);
                assert_eq!(i128::from(input.bounds.upper()), radius);
                let sum = program.nodes[..program.len].iter().flatten().find(|node|
                    matches!(node.instruction, Instruction::RowMap(_, 2, RowMap::Sum))).unwrap();
                assert!(sum.bounds.lower() >= 0);
                assert!(BigInt::from(sum.bounds.upper()) <= sum_ceiling);
            }
            for row in &rows {
                let measured = row.iter().map(|&x| BigInt::from(x).pow(2))
                    .fold(BigInt::from(0), |a, b| a + b);
                assert!(measured <= BigInt::from(obligation));
                for &x in row {
                    assert!((-radius..=radius).contains(&i128::from(x)));
                }
                for tape in 0..tapes {
                    let choices: Vec<bool> = (0..count).map(|i| tape & (1_usize << i) != 0).collect();
                    let expected = choice_rms_oracle(row, &gamma, &choices);
                    assert_eq!(choice_graph(&native, row, &choices), expected);
                    assert_eq!(choice_graph(&compact, row, &choices), expected);
                    let output_energy = expected.iter().map(|x| x.pow(2))
                        .fold(BigInt::from(0), |a, b| a + b);
                    assert!(output_energy <= BigInt::from(energy));
                    assert!(output_energy <= BigInt::from(compact_energy));
                }
            }
        }
    }
}

#[test]
fn refined_rms_stochastic_rescue_preflight_metadata_and_absent_refusals() {
    let incoming = Interval::new(-(1 << 40), 1 << 40).unwrap();
    for (obligation, _) in stochastic_rescue_cases() {
        for gamma in [[262_144, -262_144], [-524_288, 131_072]] {
            assert!(matches!(Program::rms_norm_certified(1, 2, incoming, &gamma), Err(Error::Bounds)));
            assert!(matches!(Program::rms_norm_certified_compact(1, 2, incoming, &gamma), Err(Error::Bounds)));
            let (native, _) = Program::rms_norm_with_input_norm(
                1, 2, incoming, &gamma, obligation).unwrap();
            let mut workspace = RmsWorkWorkspace::empty();
            let factories = rms_work::attempts();
            let work = Program::rms_norm_with_input_norm_work(
                1, 2, incoming, &gamma, obligation, &mut workspace).unwrap();
            assert_eq!(rms_work::attempts(), factories);
            assert_eq!(work, native.work_budget());
            workspace.assert_native_metadata(&native);
            // Preflight describes native scheduling only. Compact conversion
            // schedules need not equal native work or native metadata.
        }
    }
}

#[test]
fn refined_rms_fixed_integer_outputs_profiles_and_caps() {
    assert_eq!(ONE, 262_144);
    assert_eq!(RMS_NORM_EPSILON_Q18, 1);
    assert_eq!(MAX_ROW_LANES, 1024);
    assert_eq!(MAX_TILE_LANES, 128);
    assert_eq!(MAX_NODES, 192);
    let incoming = Interval::new(-(1 << 40), 1 << 40).unwrap();
    let gamma = [262_144; 2];
    for (row, bound, fixed) in [([0, 0], 0, [0, 0]), ([1, -1], 2, [511, -512])] {
        let expected = fixed.map(BigInt::from).to_vec();
        assert_eq!(floor_rms_oracle(&row, &gamma), expected);
        let (native, _) = Program::rms_norm_with_input_norm(1, 2, incoming, &gamma, bound).unwrap();
        let (compact, _) = Program::rms_norm_with_input_norm_compact(1, 2, incoming, &gamma, bound).unwrap();
        assert_eq!(native.numeric_profile, "puma.rmsnorm-epsilon1-q18.v1");
        assert_eq!(compact.numeric_profile, "puma.signed-nonlinear.compact-q18.v1");
        assert_eq!(floor_graph(&native, &row), expected);
        assert_eq!(floor_graph(&compact, &row), expected);
        let mut workspace = RmsWorkWorkspace::empty();
        let factories = rms_work::attempts();
        assert_eq!(Program::rms_norm_with_input_norm_work(1, 2, incoming, &gamma, bound,
            &mut workspace).unwrap(), native.work_budget());
        assert_eq!(rms_work::attempts(), factories);
    }
}

#[test]
fn refined_rms_preserves_downstream_domain_and_gamma_refusals() {
    let incoming = Interval::new(-(1 << 40), 1 << 40).unwrap();
    let radius = (1_i64 << 31) - 1;
    let bound = (radius as u128).pow(2);
    assert!(BigInt::from(radius).pow(2) < BigInt::from(i64::MAX));
    let mut partial = Builder::row(2, incoming).unwrap();
    assert!(matches!(rms_normalization(&mut partial, 2, incoming, true, Some(bound)), Err(Error::Bounds)));
    assert_eq!(partial.0.nodes[0].unwrap().bounds, Interval::new(-radius, radius).unwrap());
    assert!(partial.0.nodes.iter().flatten().any(|node| matches!(node.instruction,
        Instruction::RowMap(_, 2, RowMap::Sum))));
    // First square and sum succeeded; the encoded denominator domain did not.
    assert!(partial.0.nodes.iter().flatten().any(|node| matches!(node.instruction,
        Instruction::Affine { right: None, coefficients: Affine { left: 1, right: 0, constant: 1 }, .. })
        && node.bounds.upper() >= (1_i64 << 35)));
    let small = 2 * 262_144_u128.pow(2);
    let mut normalized = Builder::row(2, incoming).unwrap();
    assert!(rms_normalization(&mut normalized, 2, incoming, true, Some(small)).is_ok());
    // Normalization itself fits. Gamma remains an eager checked multiplication.
    let mut gamma_failure = Builder::row(2, incoming).unwrap();
    assert!(matches!(rms_recipe(&mut gamma_failure, 2, incoming, &[i64::MAX; 2], true, Some(small)), Err(Error::Bounds)));
    for (energy, gamma) in [(bound, [262_144; 2]), (small, [i64::MAX; 2])] {
        assert!(matches!(Program::rms_norm_with_input_norm(1, 2, incoming, &gamma, energy), Err(Error::Bounds)));
        assert!(matches!(Program::rms_norm_with_input_norm_compact(1, 2, incoming, &gamma, energy), Err(Error::Bounds)));
        let mut workspace = RmsWorkWorkspace::empty();
        let factories = rms_work::attempts();
        assert!(matches!(Program::rms_norm_with_input_norm_work(1, 2, incoming, &gamma, energy,
            &mut workspace), Err(Error::Bounds)));
        assert_eq!(rms_work::attempts(), factories);
    }
}

#[test]
fn checked_square_and_fixed_geometry_exclude_an_rms_sum_overflow_fixture() {
    // RMS square is a checked i64 product followed by Q18 truncation. Even ceil
    // at that maximum, summed over all 1024 admitted lanes, is strictly i64-fit.
    // A downstream RMS sum-overflow negative fixture is therefore unreachable
    // without changing caps or injecting an invalid intermediate; do neither.
    let largest_square = BigInt::from(i64::MAX);
    let ceil_rescaled = (&largest_square + 262_143_i64) / 262_144_i64;
    assert!(ceil_rescaled * 1024_i64 < BigInt::from(i64::MAX));
    // Independently exercise the shared sum's refusal on a genuine row input
    // that is NOT claimed reachable from the RMS square.
    let mut b = Builder::row(2, Interval::new(0, i64::MAX).unwrap()).unwrap();
    let input = b.input();
    assert!(matches!(b.row_map(input, 2, RowMap::Sum), Err(Error::Bounds)));
}

#[test]
fn squared_norm_coordinates_zero_signed_nonsquare_and_strict_thresholds() {
    let cap = (C as u128).pow(2);
    let incoming = Interval::new(-C, C).unwrap();
    for bound in [0, 1, 2, 3, 8, 15, 16, 17, cap - 1, cap, 2 * cap - 1, 2 * cap, u128::MAX] {
        let radius = oracle_radius(bound);
        let got = squared_norm_input_bounds(incoming, 2, bound).unwrap();
        assert_eq!(i128::from(got.lower()), (-radius).max(-i128::from(C)));
        assert_eq!(i128::from(got.upper()), radius.min(i128::from(C)));
        assert_eq!(got.upper() < C, bound < cap);
    }
    // Weak row-energy improvement is not coordinate improvement.
    for bound in [cap, 2 * cap - 1] {
        assert!(bound < 2 * cap);
        assert_eq!(squared_norm_input_bounds(incoming, 2, bound).unwrap(), incoming);
    }
    assert_eq!(squared_norm_input_bounds(incoming, 2, 0).unwrap(), Interval::new(0, 0).unwrap());
    assert_eq!(squared_norm_input_bounds(Interval::new(-8, -1).unwrap(), 2, 17).unwrap(),
        Interval::new(-4, -1).unwrap());
    assert_eq!(squared_norm_input_bounds(Interval::new(1, 8).unwrap(), 2, 17).unwrap(),
        Interval::new(1, 4).unwrap());
    assert!(squared_norm_input_bounds(Interval::new(3, 8).unwrap(), 2, 17).is_err());
    assert!(squared_norm_input_bounds(Interval::new(1, 8).unwrap(), 2, 0).is_err());
    // Individual coordinates fit, but the row's mandatory minimum does not.
    assert!(squared_norm_input_bounds(Interval::new(4, 4).unwrap(), 2, 16).is_err());
}

#[test]
fn rms_energy_refines_before_square_trap() {
    let input = Interval::new(-(1 << 40), 1 << 40).unwrap();
    let gamma = [ONE, -ONE];
    assert!(matches!(Program::rms_norm_certified(1, 2, input, &gamma), Err(Error::Bounds)));
    let bound = (C as u128).pow(2) - 1;
    let (program, _) = Program::rms_norm_with_input_norm(1, 2, input, &gamma, bound).unwrap();
    assert_eq!(program.nodes[0].unwrap().bounds, Interval::new(-C + 1, C - 1).unwrap());
    assert!(program.nodes.iter().flatten().any(|node| matches!(node.instruction,
        Instruction::Product { left: Value(0), right: Value(0), .. })));
}

#[test]
fn preflight_emission_agree_on_pre_square_derivation() {
    for (input, bound) in [
        (Interval::new(-(1 << 40), 1 << 40).unwrap(), (C as u128).pow(2) - 1),
        (Interval::new(-C, C).unwrap(), (C as u128).pow(2)),
        (Interval::new(-C, C).unwrap(), 0),
    ] {
        let mut workspace = RmsWorkWorkspace::empty();
        let work = Program::rms_norm_with_input_norm_work(1, 2, input, &[ONE, -ONE], bound,
            &mut workspace).unwrap();
        let (program, _) = Program::rms_norm_with_input_norm(1, 2, input, &[ONE, -ONE], bound).unwrap();
        assert_eq!(work, program.work_budget());
        workspace.assert_native_metadata(&program);
        let (compact, _) = Program::rms_norm_with_input_norm_compact(1, 2, input,
            &[ONE, -ONE], bound).unwrap();
        // Shared arithmetic refinement, not equality of conversion schedules:
        // this preflight/metadata workspace describes the native program only.
        assert_eq!(program.nodes[0].unwrap().bounds, compact.nodes[0].unwrap().bounds);
    }
}

#[test]
fn overflow_error_precedence_and_absent_or_insufficient_bound() {
    let huge = Interval::new(i64::MIN, i64::MAX).unwrap();
    // Original geometry wins before either derivation or overflowing square.
    assert!(matches!(Program::rms_norm_with_input_norm(1, 2, huge, &[], 0), Err(Error::Shape)));
    let mut workspace = RmsWorkWorkspace::empty();
    assert!(matches!(Program::rms_norm_with_input_norm_work(1, 2, huge, &[], 0,
        &mut workspace), Err(Error::Shape)));
    // Merely providing a bound does not bypass the product's ring-fit refusal.
    assert!(matches!(Program::rms_norm_with_input_norm(1, 2, huge, &[ONE; 2], u128::MAX), Err(Error::Bounds)));
    let input = Interval::new(-C, C).unwrap();
    let (absent, _) = Program::rms_norm_certified(1, 2, input, &[ONE; 2]).unwrap();
    let (redundant, _) = Program::rms_norm_with_input_norm(1, 2, input, &[ONE; 2],
        2 * (C as u128).pow(2)).unwrap();
    assert_eq!(absent.nodes, redundant.nodes);
    assert_eq!(absent.digest(), redundant.digest());
}
