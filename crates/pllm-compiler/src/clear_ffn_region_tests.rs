//! Source-only rectangular join definitions; no native run claimed.
use super::*;

const RW1: [f64; 32] = [
    0.5, -0.25, 0.125, 0.75, -0.375, 0.625, 0.25, -0.5,
    -0.125, 0.625, 0.25, -0.5, 0.75, -0.25, 0.5, 0.125,
    0.375, 0.125, -0.75, 0.25, -0.5, 0.375, 0.625, -0.125,
    -0.625, 0.5, 0.375, -0.125, 0.25, -0.75, 0.125, 0.5,
];
const RB1: [f64; 8] = [0.125, -0.0625, 0.25, -0.1875, 0.375, -0.25, 0.0625, -0.125];
const RW2: [f64; 32] = [
    -0.25, 0.625, 0.375, -0.125, 0.5, -0.375, 0.125, 0.25,
    0.75, 0.125, -0.5, 0.375, -0.125, 0.25, 0.625, -0.75,
    0.375, -0.125, 0.25, 0.5, -0.625, 0.75, -0.25, 0.125,
    0.125, 0.5, -0.375, 0.625, -0.5, 0.25, 0.75, -0.125,
];

fn rectangular_fixture(bytes: usize, work: usize, overflow: bool) -> (ClearInputs, OwnedBudget) {
    rectangular_fault_fixture(bytes, work, overflow, RectangularFault::None)
}

#[derive(Clone, Copy, Debug)]
enum RectangularFault {
    None,
    UpShape,
    ActivationShape,
    DownShape,
    BiasExtent,
    BiasOrientation,
    WeightOrientation,
    WeightAttribute,
    BiasAttribute,
    LayoutOrientation,
    UnsupportedWidth,
    ForeignEdge,
}

fn rectangular_fault_fixture(
    bytes: usize, work: usize, overflow: bool, fault: RectangularFault,
) -> (ClearInputs, OwnedBudget) {
    let root = OwnedBudget::new(bytes).unwrap();
    let c = Caps { f64_values: 84, ..caps() };
    let mut b = ClearInputBuilder::new(root.try_clone().unwrap(), c, work).unwrap();
    let mut x = X;
    let mut w = RW1;
    if overflow { x[0] = f64::MAX; w[0] = f64::MAX; }
    b.add_f64_tensor("x", &[2, 4], &x).unwrap();
    let weight_shape: &[usize] = match fault {
        RectangularFault::WeightOrientation => &[8, 4],
        RectangularFault::UnsupportedWidth => &[4, 6],
        _ => &[4, 8],
    };
    let weight_values = if matches!(fault, RectangularFault::UnsupportedWidth) { &w[..24] } else { &w[..] };
    b.add_f64_tensor("u", weight_shape, weight_values).unwrap();
    let (bias_shape, bias_values): (&[usize], &[f64]) = match fault {
        RectangularFault::BiasExtent => (&[4], &RB1[..4]),
        RectangularFault::BiasOrientation => (&[1, 8], &RB1),
        _ => (&[8], &RB1),
    };
    b.add_f64_tensor("v", bias_shape, bias_values).unwrap();
    b.add_f64_tensor("w", &[8, 4], &RW2).unwrap();
    b.add_f64_tensor("b", &[4], &B2).unwrap();
    b.add_operation("z", ModelOperator::Linear, None, &[], &[1]).unwrap();
    let up_shape: &[u64] = if matches!(fault, RectangularFault::UpShape) { &[2, 4] } else { &[2, 8] };
    let act_shape: &[u64] = if matches!(fault, RectangularFault::ActivationShape) { &[2, 4] } else { &[2, 8] };
    let down_shape: &[u64] = if matches!(fault, RectangularFault::DownShape) { &[2, 8] } else { &[2, 4] };
    let act_source = if matches!(fault, RectangularFault::ForeignEdge) { "z" } else { "up" };
    b.add_operation("up", ModelOperator::Linear, Some(7), &["x"], up_shape).unwrap();
    b.add_operation("act", ModelOperator::GeluTanh, Some(7), &[act_source], act_shape).unwrap();
    b.add_operation("down", ModelOperator::Linear, Some(7), &["act"], down_shape).unwrap();
    for (op, weight, bias, shape, bias_shape) in [
        (1, "u", "v", [4, 8], [8]), (3, "w", "b", [8, 4], [4]),
    ] {
        for (key, value) in [
            ("weight", weight), ("bias", bias), ("compute_dtype", "float64"),
            ("weight_layout", "input_output"), ("formula", "x_matmul_weight_plus_bias"),
        ] {
            let value = if op == 1 && key == "weight_layout" && matches!(fault, RectangularFault::LayoutOrientation) {
                "output_input"
            } else { value };
            b.set_attribute(op, key, Attribute::Text(value)).unwrap();
        }
        let shape = if op == 1 && matches!(fault, RectangularFault::WeightAttribute) { [8, 4] } else { shape };
        let bias_shape = if op == 1 && matches!(fault, RectangularFault::BiasAttribute) { [4] } else { bias_shape };
        b.set_attribute(op, "weight_shape", Attribute::Shape(&shape)).unwrap();
        b.set_attribute(op, "bias_shape", Attribute::Shape(&bias_shape)).unwrap();
    }
    b.set_attribute(2, "compute_dtype", Attribute::Text("float64")).unwrap();
    (b.finish().unwrap(), root)
}
fn rectangular_cap(input: &ClearInputs) -> Limits {
    Limits {
        max_work: input.consumed_work().checked_add(96).unwrap()
            .checked_add(PREFLIGHT_CONTROL).unwrap().checked_add(name_census(input)).unwrap()
            .checked_add(RECTANGULAR_CLEAR_WORK).unwrap(),
        max_owner_bytes: 1 << 20, max_phase_bytes: 1 << 20,
    }
}
fn rectangular_admit(input: &ClearInputs, cap: Limits) -> Result<Admitted<'_>, Error> {
    quote(input, IDS, ClearReferencePolicy::artifact_call_gelu(), cap)
}
// Independent row-major dense composition. No candidate packing/address/kernel.
fn rectangular_dense(x: &[f64], weights: &[f64], bias: &[f64]) -> Vec<f64> {
    let from = x.len() / 2;
    let to = bias.len();
    (0..2 * to).map(|i| {
        let mut sum = 0.;
        for k in 0..from { sum += x[(i / to) * from + k] * weights[k * to + i % to]; }
        sum + bias[i % to]
    }).collect()
}
fn rectangular_activation(x: &[f64]) -> Vec<f64> {
    let inner: Vec<_> = A_FIRST.iter().rev().copied().collect();
    let outer: Vec<_> = A_SECOND.iter().rev().map(|c| c * 0.5).collect();
    x.iter().map(|&z| {
        let u = z / 64.;
        (64. * u) * (0.5 + ordered_poly(&outer, ordered_poly(&inner, u)))
    }).collect()
}

#[test]
fn rectangular_dense_composition_retains_original_producers_and_stops_at_ff2() {
    let (input, root) = rectangular_fixture(1 << 20, owner::MAX_WORK, false);
    let base = root.budget().used_bytes();
    let fixed = rectangular_cap(&input);
    assert!(fixed.max_work <= owner::MAX_WORK);
    let admission = rectangular_admit(&input, fixed).unwrap();
    let q = admission.resources();
    assert_eq!(q.clear_work, RECTANGULAR_CLEAR_WORK);
    assert_eq!(q.additional_work, 96 + PREFLIGHT_CONTROL + name_census(&input) + RECTANGULAR_CLEAR_WORK);
    let output = execute(admission).unwrap();
    let first = rectangular_dense(&X, &RW1, &RB1);
    assert_eq!(output.first_affine_values(), first);
    assert_ne!(&first[..8], &first[8..]);
    let active = rectangular_activation(&first);
    assert_eq!(output.activated_values(), active);
    let expected = rectangular_dense(&active, &RW2, &B2);
    for i in 0..8 {
        let magnitude = (0..8).map(|k| (active[(i / 4) * 8 + k] * RW2[k * 4 + i % 4]).abs())
            .sum::<f64>() + B2[i % 4].abs();
        assert!((output.values()[i] - expected[i]).abs() <= 16. * f64::EPSILON * magnitude);
    }
    assert!(std::ptr::eq(output.original(), &input));
    assert!(std::ptr::eq(&*output.operation(), &input.operations()[3]));
    assert_eq!(root.budget().used_bytes(), base + q.phase_bytes);
    let paid = input.consumed_work();
    drop(output);
    assert_eq!(root.budget().used_bytes(), base);
    assert_eq!(input.consumed_work(), paid);
    assert_eq!(rectangular_admit(&input, fixed).err(), Some(Error::Work));
    remaining_attempts(&input, 3);
}

#[test]
fn rectangular_exact_one_short_work_storage_and_cumulative_no_reset() {
    let (probe, root) = rectangular_fixture(1 << 20, owner::MAX_WORK, false);
    let initial = probe.consumed_work();
    let fixed = rectangular_cap(&probe);
    let q = rectangular_admit(&probe, fixed).unwrap().resources();
    let base = root.budget().used_bytes();
    for short in [false, true] {
        let (input, root) = rectangular_fixture(base + q.phase_bytes - usize::from(short), owner::MAX_WORK, false);
        let used = root.budget().used_bytes();
        let count = observe::snapshot().factories;
        let result = rectangular_admit(&input, fixed);
        if short { assert!(result.is_err()); assert_eq!(observe::snapshot().factories, count); }
        else { drop(execute(result.unwrap()).unwrap()); }
        assert_eq!(root.budget().used_bytes(), used);
        assert_eq!(input.consumed_work(), fixed.max_work);
        assert_eq!(rectangular_admit(&input, fixed).err(), Some(Error::Work));
    }
    for short in [false, true] {
        let (input, root) = rectangular_fixture(1 << 20, initial + q.additional_work - usize::from(short), false);
        let used = root.budget().used_bytes();
        let count = observe::snapshot().factories;
        let result = rectangular_admit(&input, fixed);
        if short { assert!(result.is_err()); assert_eq!(observe::snapshot().factories, count); }
        else { drop(execute(result.unwrap()).unwrap()); assert_eq!(input.remaining_work(), 0); }
        assert_eq!(root.budget().used_bytes(), used);
    }
    for storage_short in [false, true] {
        let (input, _) = rectangular_fixture(1 << 20, owner::MAX_WORK, false);
        let limit = Limits { max_phase_bytes: q.phase_bytes - usize::from(storage_short), ..fixed };
        let result = rectangular_admit(&input, limit);
        if storage_short { assert_eq!(result.err(), Some(Error::Resource)); }
        else { drop(execute(result.unwrap()).unwrap()); }
    }
}

#[test]
fn rectangular_unread_capacity_tails_and_live_overflow_cleanup() {
    let mut scratch = Zeroizing::new(Affine::empty());
    // Final contraction has 16 live input slots, 32 live weights, 8 outputs.
    // Remaining fields/tails are NOT operands; poison must remain unread.
    scratch.mm.fill(f64::NAN);
    scratch.packed[8..].fill(f64::NAN);
    scratch.output[8..].fill(f64::NAN);
    let x = [0.125; 16];
    rectangular_affine(&x, &RW2, &B2, &mut scratch).unwrap();
    assert!(scratch.output[..8].iter().all(|v| v.is_finite()));
    assert!(scratch.output[8..].iter().all(|v| v.is_nan()));
    scratch.zeroize();
    assert!(scratch.output.iter().all(|&v| v == 0.));
    for overflow in [false, true] {
        let (input, root) = rectangular_fixture(1 << 20, owner::MAX_WORK, overflow);
        let used = root.budget().used_bytes();
        let count = observe::snapshot();
        let fixed = rectangular_cap(&input);
        let admission = rectangular_admit(&input, fixed).unwrap();
        if overflow { assert!(matches!(execute(admission), Err(Error::NonFinite))); }
        else {
            observe::panic_at(count.factories + 3);
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| execute(admission))).is_err());
        }
        let after = observe::snapshot();
        assert!(after.wipes > count.wipes);
        assert_eq!(after.dirty, count.dirty);
        assert_eq!(root.budget().used_bytes(), used);
        assert_eq!(input.consumed_work(), fixed.max_work);
        assert_eq!(rectangular_admit(&input, fixed).err(), Some(Error::Work));
    }
}

#[test]
fn rectangular_equal_values_foreign_owner_and_wrong_producer_refuse() {
    let (a, _) = rectangular_fixture(1 << 20, owner::MAX_WORK, false);
    let (b, _) = rectangular_fixture(1 << 20, owner::MAX_WORK, false);
    let output = execute(rectangular_admit(&a, rectangular_cap(&a)).unwrap()).unwrap();
    assert!(ValueEdge::new(&b, 3, 0).unwrap().bind(output).is_err());
    // Same original, wrong operation identity is equally insufficient.
    let fixed = Limits { max_work: owner::MAX_WORK, ..rectangular_cap(&b) };
    let output = execute(rectangular_admit(&b, fixed).unwrap()).unwrap();
    assert!(ValueEdge::new(&b, 1, 0).unwrap().bind(output).is_err());
}

#[test]
fn rectangular_checked_quote_inventory_and_square_unread_tails() {
    assert_eq!(CLEAR_WORK, 6096);
    assert_eq!(RECTANGULAR_CLEAR_WORK, 9848);
    assert_eq!(numeric_work(4), Ok(6096));
    assert_eq!(numeric_work(8), Ok(9848));
    assert_eq!(numeric_work(4), Ok(CLEAR_WORK));
    assert_eq!(numeric_work(8), Ok(RECTANGULAR_CLEAR_WORK));
    assert_eq!(numeric_work(usize::MAX), Err(Error::Shape));
    assert_eq!(mul(usize::MAX, 2), Err(Error::Resource));
    assert_eq!(add(usize::MAX, 1), Err(Error::Resource));
    let mut scratch = Zeroizing::new(Affine::empty());
    scratch.a[8..].fill(f64::NAN);
    scratch.w[16..].fill(f64::NAN);
    scratch.packed[8..].fill(f64::NAN);
    scratch.output[8..].fill(f64::NAN);
    affine(&X, &W1, &B1, &mut scratch).unwrap();
    assert_eq!(&scratch.output[..8], &dense(&X, &W1, &B1));
    assert!(scratch.a[8..].iter().all(|v| v.is_nan()));
    assert!(scratch.w[16..].iter().all(|v| v.is_nan()));
    scratch.zeroize();
    assert!(scratch.a.iter().chain(&scratch.w).chain(&scratch.output).all(|&v| v == 0.));
}

#[test]
fn rectangular_polynomial_order_and_outer_half_are_not_substitutable() {
    let first = rectangular_dense(&X, &RW1, &RB1);
    let expected = rectangular_activation(&first);
    let inner: Vec<_> = A_FIRST.iter().rev().copied().collect();
    let outer: Vec<_> = A_SECOND.iter().rev().map(|c| c * 0.5).collect();
    let unhalved: Vec<_> = A_SECOND.iter().rev().copied().collect();
    let mut payload = Zeroizing::new(Activation::empty());
    activate(&first, &mut payload).unwrap();
    assert_eq!(payload.output.as_slice(), expected);
    assert!(first.iter().enumerate().any(|(i, &z)| {
        let wrong = z * (0.5 + ordered_poly(&outer, ordered_poly(&A_FIRST, z / 64.)));
        wrong.to_bits() != expected[i].to_bits()
    }));
    assert!(first.iter().enumerate().any(|(i, &z)| {
        let wrong = z * (0.5 + ordered_poly(&unhalved, ordered_poly(&inner, z / 64.)));
        wrong.to_bits() != expected[i].to_bits()
    }));
}

#[test]
fn rectangular_core_poison_token_padding_is_unread_and_output_padding_zeroed() {
    // The production join admits two live tokens. This separately tests its
    // reused D57 primitive's padding contract; it does not broaden owner shapes.
    let input = matrix::Layout::new(2, 2, 2, 1, 2, 1, matrix::Diagonal::Lower, limits()).unwrap();
    let output = matrix::Layout::new(2, 2, 2, 1, 4, 1, matrix::Diagonal::Lower, limits()).unwrap();
    let mut packed = [f64::NAN; 8];
    for feature in 0..4 {
        packed[input.address(feature / 2, feature % 2, 0).unwrap()] = X[feature];
    }
    let weights: [f64; 32] = std::array::from_fn(|i| RW1[(i % 4) * 8 + i / 4]);
    let mut result = [f64::NAN; 16];
    matrix::rectangular_block_pc_mm(input, &weights, &packed, output, &mut result, limits()).unwrap();
    for feature in 0..8 {
        let expected: f64 = (0..4).map(|k| X[k] * RW1[k * 8 + feature]).sum();
        assert_eq!(result[output.address(feature / 2, feature % 2, 0).unwrap()], expected);
        assert_eq!(result[output.address(feature / 2, feature % 2, 1).unwrap()], 0.);
    }
}

// Independent D57 contraction-order oracle, directly in dense coordinates.
// For feature f, output block i=f/2; ascending Algorithm-1 a visits input
// blocks k=(i-a) mod H, H=max(from,to)/2. A block partial completes its two
// products BEFORE entering the outer sum. Capacity one gives lambda=j;
// token t therefore visits within-block input coordinates (t+j) mod 2.
// This derives from the two nested source sums, NOT Layout/address/packing,
// rectangular_block_pc_mm, or a candidate output. Missing blocks contribute
// explicit +0 products at their original positions; bias follows all sums.
fn d57_grouped_dense(x: &[f64], weights: &[f64], bias: &[f64]) -> Vec<f64> {
    let from = x.len() / 2;
    let to = bias.len();
    let blocks = from.max(to) / 2;
    (0..2 * to).map(|index| {
        let token = index / to;
        let feature = index % to;
        let output_block = feature / 2;
        let mut total = 0.;
        for diagonal in 0..blocks {
            let input_block = (output_block + blocks - diagonal) % blocks;
            let mut partial = 0.;
            for within in 0..2 {
                let input_feature = 2 * input_block + (token + within) % 2;
                let product = if input_feature < from {
                    weights[input_feature * to + feature] * x[token * from + input_feature]
                } else { 0. * 0. };
                partial += product;
            }
            total += partial;
        }
        total + bias[feature]
    }).collect()
}

#[test]
fn rectangular_d57_grouped_cancellation_bits_differ_from_ascending_dense() {
    // M=2^54: RN(M+1)=M and RN(-M+1)=-M. Grouped block partials
    // cancel to +0; ascending (((M+1)-M)+1) instead gives exactly 1.
    // Token one reverses each pair and retains the same exact distinction.
    // Both expansion and contraction exercise the same production join kernel.
    for (from, to) in [(4, 8), (8, 4)] {
        let mut x = vec![0.; 2 * from];
        for token in 0..2 {
            x[token * from..token * from + 4]
                .copy_from_slice(&[18014398509481984., 1., -18014398509481984., 1.]);
        }
        let weights = [1.; 32];
        let bias = vec![0.; to];
        let expected = d57_grouped_dense(&x, &weights, &bias);
        let ascending = rectangular_dense(&x, &weights, &bias);
        let expected_bits: Vec<_> = expected.iter().map(|v| v.to_bits()).collect();
        let ascending_bits: Vec<_> = ascending.iter().map(|v| v.to_bits()).collect();
        assert_eq!(expected_bits, vec![0u64; 2 * to]);
        assert_eq!(ascending_bits, vec![0x3ff0_0000_0000_0000u64; 2 * to]);
        assert_ne!(expected_bits, ascending_bits);
        let mut payload = Zeroizing::new(Affine::empty());
        rectangular_affine(&x, &weights, &bias, &mut payload).unwrap();
        let actual_bits: Vec<_> = payload.output[..2 * to].iter().map(|v| v.to_bits()).collect();
        assert_eq!(actual_bits, expected_bits);
    }
}

#[test]
fn rectangular_malformed_metadata_refuses_before_numeric_factories() {
    for fault in [
        RectangularFault::UpShape,
        RectangularFault::ActivationShape,
        RectangularFault::DownShape,
        RectangularFault::BiasExtent,
        RectangularFault::BiasOrientation,
        RectangularFault::WeightOrientation,
        RectangularFault::WeightAttribute,
        RectangularFault::BiasAttribute,
        RectangularFault::LayoutOrientation,
        RectangularFault::UnsupportedWidth,
        RectangularFault::ForeignEdge,
    ] {
        let (input, root) = rectangular_fault_fixture(1 << 20, owner::MAX_WORK, false, fault);
        let before = observe::snapshot();
        let bytes = root.budget().used_bytes();
        let work = input.consumed_work();
        let fixed = rectangular_cap(&input);
        let expected = match fault {
            RectangularFault::ForeignEdge => Error::Source,
            RectangularFault::LayoutOrientation => Error::Policy,
            _ => Error::Shape,
        };
        assert_eq!(rectangular_admit(&input, fixed).err(), Some(expected), "{fault:?}");
        assert_eq!(observe::snapshot().factories, before.factories, "{fault:?}");
        assert_eq!(root.budget().used_bytes(), bytes, "{fault:?}");
        // Every call passed the three ticket prechecks and prepaid census.
        // Even early unsupported-width refusal cannot refund that original work.
        assert!(input.consumed_work() >= work + 96 + PREFLIGHT_CONTROL, "{fault:?}");
        let paid = input.consumed_work();
        remaining_attempts(&input, 3);
        assert!(input.consumed_work() >= paid);
        assert_eq!(observe::snapshot().factories, before.factories);
        assert_eq!(root.budget().used_bytes(), bytes);
    }
}

#[test]
fn rectangular_intermediate_getters_and_square_legacy_compatibility() {
    let (input, root) = rectangular_fixture(1 << 20, owner::MAX_WORK, false);
    let base = root.budget().used_bytes();
    let output = execute(rectangular_admit(&input, rectangular_cap(&input)).unwrap()).unwrap();
    let first = rectangular_dense(&X, &RW1, &RB1);
    let activated = rectangular_activation(&first);
    assert_eq!(output.first_affine_values().len(), 16);
    assert_eq!(output.activated_values().len(), 16);
    assert_eq!(output.first_affine_values(), first);
    assert_eq!(output.activated_values(), activated);
    assert_eq!(output.values().len(), 8);
    let before = observe::snapshot();
    let paid = input.consumed_work();
    let bytes = root.budget().used_bytes();
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| output.first_affine())).is_err());
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| output.activated())).is_err());
    assert_eq!(observe::snapshot().factories, before.factories);
    assert_eq!(input.consumed_work(), paid);
    assert_eq!(root.budget().used_bytes(), bytes);
    // Panicking legacy inspection does not truncate, invalidate or replace the
    // live 16-element producers; the new getters still expose the same values.
    assert_eq!(output.first_affine_values(), first);
    assert_eq!(output.activated_values(), activated);
    drop(output);
    assert_eq!(root.budget().used_bytes(), base);
    assert_eq!(input.consumed_work(), paid);

    let (square, _) = fixture(Fault::None, 1 << 20, owner::MAX_WORK);
    let output = execute(admitted(&square)).unwrap();
    // Typed bindings lock the original square public result signatures.
    let first: &[f64; 8] = output.first_affine();
    let active: &[f64; 8] = output.activated();
    let final_values: &[f64; 8] = output.values();
    assert_eq!(output.first_affine_values().len(), 8);
    assert_eq!(output.activated_values().len(), 8);
    assert_eq!(output.first_affine_values(), first);
    assert_eq!(output.activated_values(), active);
    assert_eq!(bits(first), bits(&dense(&X, &W1, &B1)));
    assert_eq!(bits(active), bits(&ordered_activation(first)));
    assert_eq!(bits(final_values), bits(&ordered_affine(active, &W2, &B2)));
}
