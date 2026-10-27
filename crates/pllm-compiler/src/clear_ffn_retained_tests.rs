//! SOURCE-ONLY regression definitions. No native execution/qualification claim.
//! Fixtures derive from dense XW+b independently of packed address helpers.
use super::*;
use crate::clear_input_owner::{Attribute, Caps, ClearInputBuilder};
use pllm_garble::shared_allocation::OwnedBudget;

const IDS: Selection = Selection {
    operations: [1, 2, 3],
    tensors: [0, 1, 2, 3, 4],
};
const X: [f64; 8] = [0.25, -0.5, 0.75, 0.125, -0.375, 0.625, -0.25, 0.5];
const W1: [f64; 16] = [
    0.5, -0.25, 0.125, 0.75, -0.125, 0.625, 0.25, -0.5, 0.375, 0.125, -0.75, 0.25, -0.625, 0.5,
    0.375, -0.125,
];
const B1: [f64; 4] = [0.125, -0.0625, 0.25, -0.1875];
const W2: [f64; 16] = [
    -0.25, 0.625, 0.375, -0.125, 0.5, -0.375, 0.125, 0.25, 0.75, 0.125, -0.5, 0.375, -0.125, 0.25,
    0.625, -0.75,
];
const B2: [f64; 4] = [-0.25, 0.125, -0.0625, 0.1875];

fn caps() -> Caps {
    Caps {
        f64_values: 48,
        u64_values: 0,
        tensors: 5,
        operations: 4,
    }
}
fn name_census(input: &ClearInputs) -> usize {
    let names = input
        .tensors()
        .iter()
        .try_fold(0usize, |sum, t| {
            sum.checked_add(1)?.checked_add(t.name().len())
        })
        .unwrap();
    input.operations().len().checked_mul(names).unwrap()
}
// Capture once BEFORE attempts. Original construction is part of the caller's
// cumulative cap, not a new per-call allowance or a solver-observed pass cost.
fn cap(input: &ClearInputs) -> Limits {
    let max_work = input
        .consumed_work()
        .checked_add(96)
        .and_then(|n| n.checked_add(PREFLIGHT_CONTROL))
        .and_then(|n| n.checked_add(name_census(input)))
        .and_then(|n| n.checked_add(CLEAR_WORK))
        .unwrap();
    assert!(max_work <= owner::MAX_WORK);
    Limits {
        max_owner_bytes: 1 << 20,
        max_phase_bytes: 1 << 20,
        max_work,
    }
}
#[derive(Clone, Copy)]
enum Fault {
    None,
    F32,
    ActivationF32,
    OrdinaryActivation,
    WrongLayout,
    WrongOperator,
    WrongLayer,
    WrongShape,
    WrongEdge,
    Namespace,
    Overflow,
    ActivationOverflow,
}

fn fixture(fault: Fault, bytes: usize, work: usize) -> (ClearInputs, OwnedBudget) {
    let root = OwnedBudget::new(bytes).unwrap();
    let mut b = ClearInputBuilder::new(root.try_clone().unwrap(), caps(), work).unwrap();
    let mut x = X;
    if matches!(fault, Fault::Overflow | Fault::ActivationOverflow) {
        x[7] = f64::MAX;
    }
    b.add_f64_tensor("x", &[2, 4], &x).unwrap();
    let mut w = W1;
    if matches!(fault, Fault::Overflow) {
        w[12] = f64::MAX;
    }
    b.add_f64_tensor("u", &[4, 4], &w).unwrap();
    b.add_f64_tensor("v", &[4], &B1).unwrap();
    b.add_f64_tensor("w", &[4, 4], &W2).unwrap();
    b.add_f64_tensor("b", &[4], &B2).unwrap();
    // Unrelated original operation proves final identity is selected, not an
    // assumed ops[2] or reconstructed/cloned DTO. No tensor/op namespace alias.
    b.add_operation("z", ModelOperator::Linear, None, &[], &[1])
        .unwrap();
    let up = if matches!(fault, Fault::Namespace) {
        "x"
    } else {
        "up"
    };
    b.add_operation(up, ModelOperator::Linear, Some(7), &["x"], &[2, 4])
        .unwrap();
    b.add_operation(
        "act",
        if matches!(fault, Fault::WrongOperator) {
            ModelOperator::Linear
        } else {
            ModelOperator::GeluTanh
        },
        Some(7),
        &[up],
        &[2, 4],
    )
    .unwrap();
    b.add_operation(
        "down",
        ModelOperator::Linear,
        Some(if matches!(fault, Fault::WrongLayer) {
            8
        } else {
            7
        }),
        &[if matches!(fault, Fault::WrongEdge) {
            up
        } else {
            "act"
        }],
        if matches!(fault, Fault::WrongShape) {
            &[4, 2]
        } else {
            &[2, 4]
        },
    )
    .unwrap();
    for (op, weight, bias) in [(1, "u", "v"), (3, "w", "b")] {
        for (key, value) in [
            ("weight", weight),
            ("bias", bias),
            (
                "compute_dtype",
                if matches!(fault, Fault::F32) {
                    "float32"
                } else {
                    "float64"
                },
            ),
            (
                "weight_layout",
                if matches!(fault, Fault::WrongLayout) {
                    "output_input"
                } else {
                    "input_output"
                },
            ),
            ("formula", "x_matmul_weight_plus_bias"),
        ] {
            b.set_attribute(op, key, Attribute::Text(value)).unwrap();
        }
        b.set_attribute(op, "weight_shape", Attribute::Shape(&[4, 4]))
            .unwrap();
        b.set_attribute(op, "bias_shape", Attribute::Shape(&[4]))
            .unwrap();
    }
    if !matches!(fault, Fault::OrdinaryActivation) {
        b.set_attribute(
            2,
            "compute_dtype",
            Attribute::Text(if matches!(fault, Fault::ActivationF32) {
                "float32"
            } else {
                "float64"
            }),
        )
        .unwrap();
    }
    (b.finish().unwrap(), root)
}
fn admitted(input: &ClearInputs) -> Admitted<'_> {
    quote(
        input,
        IDS,
        ClearReferencePolicy::artifact_call_gelu(),
        cap(input),
    )
    .unwrap()
}
fn bits(x: &[f64; 8]) -> [u64; 8] {
    x.map(f64::to_bits)
}

// Independent dense geometry oracle. Ascending contraction establishes XW+b
// meaning, NOT binary64 order equivalence. Dyadic first-stage fixtures are exact;
// non-dyadic activation/final comparisons below use the separate order oracle.
fn dense(x: &[f64; 8], w: &[f64; 16], b: &[f64; 4]) -> [f64; 8] {
    std::array::from_fn(|i| {
        let mut y = 0.;
        for k in 0..4 {
            y += x[(i / 4) * 4 + k] * w[k * 4 + i % 4];
        }
        y + b[i % 4]
    })
}

// Primitive-order oracle derives from diagonal contractions, without Layout,
// pack/unpack/address, routing helpers or optimized_lower_mm. For token r and
// output c, diagonal contraction l visits input k=(c+l)%4. Algorithm 2 sums
// nonwrapped and wrapped terms independently, then adds those two partials.
fn ordered_affine(x: &[f64; 8], w: &[f64; 16], b: &[f64; 4]) -> [f64; 8] {
    std::array::from_fn(|i| {
        let (r, c) = (i / 4, i % 4);
        let mut direct = 0.;
        let mut wrapped = 0.;
        for diagonal in 0..4 {
            let k = (c + diagonal) % 4;
            let product = x[4 * r + k] * w[4 * k + c];
            if c + diagonal < 4 {
                direct += product;
            } else {
                wrapped += product;
            }
        }
        let sum = direct + wrapped;
        sum + b[c]
    })
}

// Separate expression tree, not a call to the shared schedule. Parentheses are
// part of the binary64 contract; no Horner/ascending-sum/mul_add substitution.
fn ordered_poly(c: &[f64], x: f64) -> f64 {
    let q = x * x;
    let fourth = q * q;
    let eighth = fourth * fourth;
    let groups: Vec<f64> = c
        .chunks_exact(4)
        .map(|c| {
            let a = c[1] * x;
            let b = c[2] * q;
            let ab = a + b;
            let cx = c[3] * x;
            let cubic = q * cx;
            let abc = ab + cubic;
            abc + c[0]
        })
        .collect();
    let pairs: Vec<f64> = groups
        .chunks(2)
        .map(|g| {
            if g.len() == 1 {
                g[0]
            } else {
                let high = g[1] * fourth;
                g[0] + high
            }
        })
        .collect();
    if c.len() == 16 {
        let high = pairs[1] * eighth;
        return pairs[0] + high;
    }
    let high0 = pairs[1] * eighth;
    let low = pairs[0] + high0;
    let high1 = pairs[3] * eighth;
    let high = pairs[2] + high1;
    let sixteenth = eighth * eighth;
    let tail = high * sixteenth;
    low + tail
}
fn ordered_activation(x: &[f64; 8]) -> [f64; 8] {
    let inner: Vec<_> = A_FIRST.iter().rev().copied().collect();
    let outer: Vec<_> = A_SECOND.iter().rev().map(|c| c * 0.5).collect();
    x.map(|x| {
        let u = x / 64.;
        let first = ordered_poly(&inner, u);
        let second = ordered_poly(&outer, first);
        let recovered = 64. * u;
        let factor = 0.5 + second;
        recovered * factor
    })
}

#[test]
fn asymmetric_dense_meaning_and_primitive_order_bits_retained_identity() {
    let (input, root) = fixture(Fault::None, 1 << 20, owner::MAX_WORK);
    let before = root.budget().used_bytes();
    let quote = admitted(&input);
    let q = quote.resources();
    assert_eq!(root.budget().used_bytes(), before + q.phase_bytes);
    assert_eq!(q.cumulative_work, input.consumed_work());
    assert_eq!(
        q.additional_work,
        3 * 32 + PREFLIGHT_CONTROL + CLEAR_WORK + 4 * (2 + 2 + 2 + 2 + 2)
    );
    assert_eq!(q.schedules[0].tensor_products, 8); // two frames, four products
    let out = execute(quote).unwrap();
    assert!(std::ptr::eq(out.original(), &input));
    assert!(std::ptr::eq(&*out.operation(), &input.operations()[3]));
    let first = dense(&X, &W1, &B1);
    assert_eq!(bits(out.first_affine()), bits(&first));
    assert_eq!(bits(&ordered_affine(&X, &W1, &B1)), bits(&first));
    let active = ordered_activation(&first);
    assert_eq!(bits(out.activated()), bits(&active));
    let expected = ordered_affine(&active, &W2, &B2);
    assert_eq!(bits(out.values()), bits(&expected));
    for (a, b) in out.values().iter().zip(dense(&active, &W2, &B2)) {
        assert!((a - b).abs() <= 16. * f64::EPSILON * b.abs().max(1.));
    }
    assert_eq!(root.budget().used_bytes(), before + q.phase_bytes);
    let paid = input.consumed_work();
    drop(out);
    assert_eq!(root.budget().used_bytes(), before);
    assert_eq!(input.consumed_work(), paid);
}

#[test]
fn dyadic_rational_polynomial_checks_are_exact_not_float_dense_sums() {
    // Rational Horner in i128 on integer coefficient numerators / 8; evaluated
    // at +/-1/2. Degree31 denominator 8*2^31 fits and all exact results fit 53
    // bits. This checks the generic primitive schedule, not a decimal/HE claim.
    for len in [16, 28, 32] {
        let integers: Vec<i128> = (0..len).map(|i| ((i * 7 + 3) % 13) as i128 - 6).collect();
        let c: Vec<f64> = integers.iter().map(|&n| n as f64 / 8.).collect();
        for sign in [-1i128, 1] {
            let mut numerator = *integers.last().unwrap();
            let mut denominator = 1i128;
            for &coefficient in integers[..len - 1].iter().rev() {
                denominator *= 2;
                numerator = numerator * sign + coefficient * denominator;
            }
            denominator *= 8;
            let exact = numerator as f64 / denominator as f64;
            let mut workspace = Zeroizing::new(poly::Workspace::empty());
            let actual = poly::stockmeyer(&c, sign as f64 / 2., &mut workspace).unwrap();
            assert_eq!(actual.to_bits(), exact.to_bits());
            assert_eq!(
                actual.to_bits(),
                ordered_poly(&c, sign as f64 / 2.).to_bits()
            );
        }
    }
    // Pinned binary64 tables at zero have the exact dyadic constant term.
    for c in [
        A_FIRST.iter().rev().copied().collect::<Vec<_>>(),
        A_SECOND.iter().rev().map(|c| c * 0.5).collect(),
    ] {
        assert_eq!(ordered_poly(&c, 0.).to_bits(), c[0].to_bits());
    }
}

#[test]
fn policy_edges_shapes_layers_and_namespaces_refuse_before_factories() {
    for fault in [
        Fault::F32,
        Fault::ActivationF32,
        Fault::OrdinaryActivation,
        Fault::WrongLayout,
        Fault::WrongOperator,
        Fault::WrongLayer,
        Fault::WrongShape,
        Fault::WrongEdge,
        Fault::Namespace,
    ] {
        let (input, root) = fixture(fault, 1 << 20, owner::MAX_WORK);
        let used = root.budget().used_bytes();
        let factories = observe::snapshot().factories;
        assert!(quote(
            &input,
            IDS,
            ClearReferencePolicy::artifact_call_gelu(),
            cap(&input)
        )
        .is_err());
        assert_eq!(observe::snapshot().factories, factories);
        assert_eq!(root.budget().used_bytes(), used);
    }
    for ids in [
        Selection {
            operations: [1, 1, 3],
            ..IDS
        },
        Selection {
            operations: [3, 2, 1],
            ..IDS
        },
        Selection {
            tensors: [0, 1, 2, 1, 4],
            ..IDS
        },
        Selection {
            tensors: [0, 1, 2, 3, 99],
            ..IDS
        },
    ] {
        let (input, _) = fixture(Fault::None, 1 << 20, owner::MAX_WORK);
        let factories = observe::snapshot().factories;
        assert!(quote(
            &input,
            ids,
            ClearReferencePolicy::artifact_call_gelu(),
            cap(&input)
        )
        .is_err());
        assert_eq!(observe::snapshot().factories, factories);
    }
}

#[test]
fn exact_cumulative_original_work_and_one_short_admission() {
    let (probe, _) = fixture(Fault::None, 1 << 20, owner::MAX_WORK);
    let initial = probe.consumed_work();
    let q = admitted(&probe).resources();
    for short in [false, true] {
        let (input, root) = fixture(
            Fault::None,
            1 << 20,
            initial + q.additional_work - usize::from(short),
        );
        let base = root.budget().used_bytes();
        let factories = observe::snapshot().factories;
        let result = quote(
            &input,
            IDS,
            ClearReferencePolicy::artifact_call_gelu(),
            cap(&input),
        );
        if short {
            assert!(result.is_err());
            assert_eq!(observe::snapshot().factories, factories);
        } else {
            drop(execute(result.unwrap()).unwrap());
            assert_eq!(input.remaining_work(), 0);
        }
        assert_eq!(root.budget().used_bytes(), base);
    }
}

// Public seam proves how many attempts earlier refusals actually burned. These
// probes deliberately spend their own work; no inspection/reset of owner state.
fn remaining_attempts(input: &ClearInputs, burned: usize) {
    for _ in burned..32 {
        let before = input.consumed_work();
        if input.remaining_work() < 32 {
            assert!(matches!(input.begin_attempt(), Err(owner::Error::Work)));
            assert_eq!(input.consumed_work(), before);
        } else {
            let _ticket = input.begin_attempt().unwrap();
            assert_eq!(input.consumed_work(), before + 32);
        }
    }
    let before = input.consumed_work();
    assert!(matches!(input.begin_attempt(), Err(owner::Error::Burned)));
    assert_eq!(input.consumed_work(), before);
}

#[test]
fn caller_cumulative_cap_checks_each_ticket_and_full_debit() {
    // Caller ceilings, not original-ledger shortages: all owners have MAX_WORK.
    for (allowance, burned) in [(31, 0), (63, 1), (95, 2), (32, 1), (64, 2), (96, 3)] {
        let (input, root) = fixture(Fault::None, 1 << 20, owner::MAX_WORK);
        let before = input.consumed_work();
        let bytes = root.budget().used_bytes();
        let factories = observe::snapshot().factories;
        let fixed = Limits {
            max_work: before + allowance,
            ..cap(&input)
        };
        assert_eq!(
            quote(&input, IDS, ClearReferencePolicy::artifact_call_gelu(), fixed).err(),
            Some(Error::Work)
        );
        assert_eq!(input.consumed_work(), before + 32 * burned);
        assert_eq!(root.budget().used_bytes(), bytes);
        assert_eq!(observe::snapshot().factories, factories);
        remaining_attempts(&input, burned);
        assert_eq!(root.budget().used_bytes(), bytes);
        assert_eq!(observe::snapshot().factories, factories);
    }
    for checkpoint in 0..3 {
        let (input, root) = fixture(Fault::None, 1 << 20, owner::MAX_WORK);
        let before = input.consumed_work();
        let bytes = root.budget().used_bytes();
        let factories = observe::snapshot().factories;
        let full = cap(&input);
        let fixed = Limits {
            max_work: match checkpoint {
                0 => before + 96 + PREFLIGHT_CONTROL - 1,
                1 => full.max_work - 1,
                _ => full.max_work,
            },
            ..full
        };
        let result = quote(
            &input,
            IDS,
            ClearReferencePolicy::artifact_call_gelu(),
            fixed,
        );
        if checkpoint < 2 {
            assert_eq!(result.err(), Some(Error::Work));
            assert_eq!(
                input.consumed_work(),
                before + 96 + if checkpoint == 1 { PREFLIGHT_CONTROL } else { 0 }
            );
            assert_eq!(observe::snapshot().factories, factories);
        } else {
            let admission = result.unwrap();
            assert_eq!(
                admission.resources().additional_work,
                fixed.max_work - before
            );
            assert_eq!(input.consumed_work(), fixed.max_work);
            drop(execute(admission).unwrap());
            assert_eq!(observe::snapshot().factories, factories + 3);
        }
        assert_eq!(root.budget().used_bytes(), bytes);
        remaining_attempts(&input, 3);
        assert_eq!(root.budget().used_bytes(), bytes);
    }
}

#[test]
fn fixed_cap_retries_preserve_original_work_and_attempts() {
    let (input, root) = fixture(Fault::None, 1 << 20, owner::MAX_WORK);
    let before = input.consumed_work();
    let bytes = root.budget().used_bytes();
    let factories = observe::snapshot().factories;
    let fixed = Limits {
        max_work: before + 127,
        ..cap(&input)
    };
    for _ in 0..40 {
        assert_eq!(
            quote(&input, IDS, ClearReferencePolicy::artifact_call_gelu(), fixed).err(),
            Some(Error::Work)
        );
        assert_eq!(input.consumed_work(), before + 96);
        assert_eq!(root.budget().used_bytes(), bytes);
        assert_eq!(observe::snapshot().factories, factories);
    }
    remaining_attempts(&input, 3);
    assert_eq!(root.budget().used_bytes(), bytes);
    assert_eq!(observe::snapshot().factories, factories);

    // Fully paid source validation failure burns the original work/tickets;
    // retrying with the SAME exact caller cap cannot buy its next first ticket.
    let (input, root) = fixture(Fault::WrongEdge, 1 << 20, owner::MAX_WORK);
    let before = input.consumed_work();
    let bytes = root.budget().used_bytes();
    let fixed = cap(&input);
    assert_eq!(
        quote(&input, IDS, ClearReferencePolicy::artifact_call_gelu(), fixed).err(),
        Some(Error::Source)
    );
    assert_eq!(input.consumed_work(), fixed.max_work);
    assert_eq!(
        input.consumed_work() - before,
        96 + PREFLIGHT_CONTROL + name_census(&input) + CLEAR_WORK
    );
    for _ in 0..40 {
        assert_eq!(
            quote(&input, IDS, ClearReferencePolicy::artifact_call_gelu(), fixed).err(),
            Some(Error::Work)
        );
        assert_eq!(input.consumed_work(), fixed.max_work);
        assert_eq!(root.budget().used_bytes(), bytes);
        assert_eq!(observe::snapshot().factories, factories);
    }
    remaining_attempts(&input, 3);
    assert_eq!(root.budget().used_bytes(), bytes);
    assert_eq!(observe::snapshot().factories, factories);
}

#[test]
fn legacy_clear_only_cap_refuses_and_work_overflow_is_nonburning() {
    let (input, root) = fixture(Fault::None, 1 << 20, owner::MAX_WORK);
    let before = input.consumed_work();
    let bytes = root.budget().used_bytes();
    let factories = observe::snapshot().factories;
    let full = cap(&input);
    let legacy = Limits {
        max_work: CLEAR_WORK,
        ..full
    };
    assert!(before + 32 > legacy.max_work);
    assert_eq!(
        quote(&input, IDS, ClearReferencePolicy::artifact_call_gelu(), legacy).err(),
        Some(Error::Work)
    );
    assert_eq!(input.consumed_work(), before);
    assert_eq!(check_work(&input, full, usize::MAX), Err(Error::Work));
    let above_global = Limits {
        max_work: usize::MAX,
        ..full
    };
    assert_eq!(
        check_work(&input, above_global, owner::MAX_WORK - before),
        Ok(())
    );
    assert_eq!(
        check_work(&input, above_global, owner::MAX_WORK - before + 1),
        Err(Error::Work)
    );
    assert_eq!(input.consumed_work(), before);
    assert_eq!(root.budget().used_bytes(), bytes);
    assert_eq!(observe::snapshot().factories, factories);
    remaining_attempts(&input, 0);
    assert_eq!(root.budget().used_bytes(), bytes);
    assert_eq!(observe::snapshot().factories, factories);
}

#[test]
fn dense_ascending_order_is_explicitly_not_binary64_bit_proof() {
    let x = [
        18014398509481984.,
        1.,
        -18014398509481984.,
        1.,
        -2.,
        3.,
        5.,
        -7.,
    ];
    let w = [1.; 16];
    let b = [0.; 4];
    let expected = ordered_affine(&x, &w, &b);
    assert_ne!(bits(&dense(&x, &w, &b)), bits(&expected));
    let mut scratch = Zeroizing::new(Affine::empty());
    affine(&x, &w, &b, &mut scratch).unwrap();
    assert_eq!(bits(&scratch.output), bits(&expected));
}

// Existing cached reconciliation remains an independent literal-table control.
// Authored subprocess test only: NOT executed during this source-only change.
fn python_artifact(xs: &[f64]) -> Vec<f64> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let script = "import sys,struct;sys.path.insert(0,sys.argv[1]);from benchmarks.research.thor_literal_reconciliation import evaluate;print(' '.join(str(struct.unpack('>Q',struct.pack('>d',evaluate('gelu','A',float(x),'binary64')))[0]) for x in sys.argv[2:]))";
    let output = std::process::Command::new("python3")
        .args(["-c", script])
        .arg(root)
        .args(xs.iter().map(ToString::to_string))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .unwrap()
        .split_whitespace()
        .map(|v| f64::from_bits(v.parse().unwrap()))
        .collect()
}

#[test]
fn complete_pinned_a_call_python_bits_and_no_order_scale_halving_tanh_substitutions() {
    let xs = [-1., -0.125, 0.5, 2., -0.5, 0.25, 1.5, -2.];
    let expected = python_artifact(&xs);
    let ours = ordered_activation(&xs);
    let mut payload = Zeroizing::new(Activation::empty());
    activate(&xs, &mut payload).unwrap();
    let reversed: Vec<_> = A_FIRST.iter().rev().copied().collect();
    let unhalved: Vec<_> = A_SECOND.iter().rev().copied().collect();
    for i in 0..8 {
        assert_eq!(ours[i].to_bits(), expected[i].to_bits());
        assert_eq!(payload.output[i].to_bits(), expected[i].to_bits());
        let inner = ordered_poly(&reversed, xs[i] / 64.);
        assert_ne!(
            inner.to_bits(),
            ordered_poly(&A_FIRST, xs[i] / 64.).to_bits()
        );
        assert_ne!(inner.to_bits(), ordered_poly(&reversed, xs[i]).to_bits());
        let wrong = xs[i] * (0.5 + ordered_poly(&unhalved, inner));
        assert_ne!(wrong.to_bits(), ours[i].to_bits());
        let ordinary = 0.5
            * xs[i]
            * (1.
                + ((2. / std::f64::consts::PI).sqrt()
                    * (xs[i] + 0.044715 * xs[i] * xs[i] * xs[i]))
                    .tanh());
        assert_ne!(ordinary.to_bits(), ours[i].to_bits());
    }
}

#[test]
fn legacy_complete_region_python_boundary_magnitude_bound_and_payload_inventory() {
    // Port the original clear_ffn_region_tests.rs numeric fixture unchanged,
    // through real original custody rather than its removed guard-adoption API.
    let x = [0.125, 0.25, 0.375, 0.5, -0.25, 0.125, 0.0625, 0.375];
    let w1 = [
        0.25, -0.125, 0.5, 0.0625, 0.125, 0.25, -0.25, 0.125,
        -0.25, 0.125, 0.0625, 0.5, 0.0625, -0.25, 0.125, 0.25,
    ];
    let w2 = [
        0.125, 0.25, -0.125, 0.0625, 0.25, -0.125, 0.0625, 0.125,
        -0.125, 0.0625, 0.25, -0.25, 0.0625, 0.125, 0.25, 0.125,
    ];
    let b1 = [0.125, -0.25, 0.0625, 0.375];
    let b2 = [-0.125, 0.0625, 0.25, -0.0625];
    let root = OwnedBudget::new(1 << 20).unwrap();
    let mut b =
        ClearInputBuilder::new(root.try_clone().unwrap(), caps(), owner::MAX_WORK).unwrap();
    b.add_f64_tensor("x", &[2, 4], &x).unwrap();
    b.add_f64_tensor("w1", &[4, 4], &w1).unwrap();
    b.add_f64_tensor("b1", &[4], &b1).unwrap();
    b.add_f64_tensor("w2", &[4, 4], &w2).unwrap();
    b.add_f64_tensor("b2", &[4], &b2).unwrap();
    for (name, operator, source) in [
        ("up", ModelOperator::Linear, "x"),
        ("activation", ModelOperator::GeluTanh, "up"),
        ("down", ModelOperator::Linear, "activation"),
    ] {
        let id = b.add_operation(name, operator, Some(0), &[source], &[2, 4]).unwrap();
        b.set_attribute(id, "compute_dtype", Attribute::Text("float64")).unwrap();
        if id != 1 {
            let (weight, bias) = if id == 0 { ("w1", "b1") } else { ("w2", "b2") };
            for (key, value) in [
                ("weight", weight), ("bias", bias),
                ("weight_layout", "input_output"), ("formula", "x_matmul_weight_plus_bias"),
            ] {
                b.set_attribute(id, key, Attribute::Text(value)).unwrap();
            }
            b.set_attribute(id, "weight_shape", Attribute::Shape(&[4, 4])).unwrap();
            b.set_attribute(id, "bias_shape", Attribute::Shape(&[4])).unwrap();
        }
    }
    let input = b.finish().unwrap();
    let base = root.budget().used_bytes();
    let admission = quote(
        &input,
        Selection { operations: [0, 1, 2], tensors: IDS.tensors },
        ClearReferencePolicy::artifact_call_gelu(),
        cap(&input),
    ).unwrap();
    let q = admission.resources();
    // New retained owner splits the former 181-double Scratch into two live
    // 56-double affine payloads and one 85-double activation payload (197 total).
    assert_eq!(size_of::<Affine>(), 56 * size_of::<f64>());
    assert_eq!(size_of::<Activation>(), 85 * size_of::<f64>());
    assert!(q.phase_bytes >= 2 * size_of::<Output<'_>>()
        + size_of::<Result<Output<'_>, Error>>()
        + 2 * (2 * size_of::<Affine>() + size_of::<Activation>()));
    assert_eq!(q.schedules[0], q.schedules[1]);
    assert_eq!(q.schedules[0].tensor_products, 8);
    assert_eq!(q.schedules[0].input_rotations, 6);
    let paid = input.consumed_work();
    let out = execute(admission).unwrap();
    let first = dense(&x, &w1, &b1);
    assert_ne!(&first[..4], &first[4..]);
    assert_eq!(bits(out.first_affine()), bits(&first));
    let activation: [f64; 8] = python_artifact(&first).try_into().unwrap();
    assert_eq!(bits(out.activated()), bits(&activation));
    let expected = dense(&activation, &w2, &b2);
    for i in 0..8 {
        let magnitude = (0..4)
            .map(|k| (activation[(i / 4) * 4 + k] * w2[k * 4 + i % 4]).abs())
            .sum::<f64>() + b2[i % 4].abs();
        assert!((out.values()[i] - expected[i]).abs() <= 8. * f64::EPSILON * magnitude);
    }
    assert_eq!(root.budget().used_bytes(), base + q.phase_bytes);
    drop(out);
    assert_eq!(root.budget().used_bytes(), base);
    assert_eq!(input.consumed_work(), paid);
}

#[test]
fn legacy_nonfinite_input_is_rejected_by_original_builder_not_adopted() {
    // The old executor accepted raw fixtures and encountered input[6] = +inf.
    // Actual original ownership now refuses that value during construction.
    let root = OwnedBudget::new(1 << 20).unwrap();
    let base = root.budget().used_bytes();
    let factories = observe::snapshot().factories;
    let mut b =
        ClearInputBuilder::new(root.try_clone().unwrap(), caps(), owner::MAX_WORK).unwrap();
    let mut x = [0.125, 0.25, 0.375, 0.5, -0.25, 0.125, 0.0625, 0.375];
    x[6] = f64::INFINITY;
    assert_eq!(b.add_f64_tensor("x", &[2, 4], &x), Err(owner::Error::NonFinite));
    assert!(matches!(b.finish(), Err(owner::Error::Burned)));
    assert_eq!(root.budget().used_bytes(), base);
    assert_eq!(observe::snapshot().factories, factories);
}

#[test]
fn original_builder_rejects_duplicate_operation_and_tensor_names() {
    for tensor in [false, true] {
        let mut b =
            ClearInputBuilder::new(OwnedBudget::new(1 << 20).unwrap(), caps(), owner::MAX_WORK)
                .unwrap();
        if tensor {
            b.add_f64_tensor("x", &[1], &[1.]).unwrap();
            assert!(b.add_f64_tensor("x", &[1], &[2.]).is_err());
        } else {
            b.add_operation("a", ModelOperator::Linear, None, &[], &[1])
                .unwrap();
            assert!(b
                .add_operation("a", ModelOperator::Linear, None, &[], &[1])
                .is_err());
        }
        assert!(b.finish().is_err());
    }
}

#[test]
fn foreign_owner_and_wrong_operation_cannot_bind_actual_ffn_producer() {
    let (a, _) = fixture(Fault::None, 1 << 20, owner::MAX_WORK);
    let (b, _) = fixture(Fault::None, 1 << 20, owner::MAX_WORK);
    let out = execute(admitted(&a)).unwrap();
    let factories = observe::snapshot().factories;
    assert!(ValueEdge::new(&b, 3, 0).unwrap().bind(out).is_err());
    assert_eq!(observe::snapshot().factories, factories);
    let (a, _) = fixture(Fault::None, 1 << 20, owner::MAX_WORK);
    let out = execute(admitted(&a)).unwrap();
    let factories = observe::snapshot().factories;
    assert!(ValueEdge::new(&a, 1, 0).unwrap().bind(out).is_err());
    assert_eq!(observe::snapshot().factories, factories);
}

#[test]
fn short_caps_and_original_reservations_create_zero_factories() {
    let (probe, _) = fixture(Fault::None, 1 << 20, owner::MAX_WORK);
    let full_cap = cap(&probe);
    let q = quote(
        &probe,
        IDS,
        ClearReferencePolicy::artifact_call_gelu(),
        full_cap,
    )
    .unwrap()
    .resources();
    for limit in [
        Limits {
            max_owner_bytes: probe.charged_bytes() - 1,
            ..full_cap
        },
        Limits {
            max_phase_bytes: q.phase_bytes - 1,
            ..full_cap
        },
        Limits {
            max_work: CLEAR_WORK - 1,
            ..full_cap
        },
    ] {
        let (input, _) = fixture(Fault::None, 1 << 20, owner::MAX_WORK);
        let count = observe::snapshot().factories;
        assert!(quote(
            &input,
            IDS,
            ClearReferencePolicy::artifact_call_gelu(),
            limit
        )
        .is_err());
        assert_eq!(observe::snapshot().factories, count);
    }
    // Root includes its additional observing handle; exact limit then one byte
    // short forces the LAST live reservation to fail, retiring earlier charges.
    let (input, root) = fixture(Fault::None, 1 << 20, owner::MAX_WORK);
    let base = root.budget().used_bytes();
    drop(input);
    drop(root);
    for short in [false, true] {
        let exact = base + q.phase_bytes;
        let (input, root) = fixture(Fault::None, exact - usize::from(short), owner::MAX_WORK);
        let before = root.budget().used_bytes();
        let count = observe::snapshot().factories;
        let result = quote(
            &input,
            IDS,
            ClearReferencePolicy::artifact_call_gelu(),
            full_cap,
        );
        if short {
            assert!(result.is_err());
            assert_eq!(observe::snapshot().factories, count);
        } else {
            let output = execute(result.unwrap()).unwrap();
            assert_eq!(root.budget().used_bytes(), exact);
            drop(output);
        }
        assert_eq!(root.budget().used_bytes(), before);
    }
}

#[test]
fn cancellation_failed_retries_and_32_unit_precheck_never_refill_original_work() {
    let (input, root) = fixture(Fault::None, 1 << 20, owner::MAX_WORK);
    let initial = input.consumed_work();
    let base = root.budget().used_bytes();
    let count = observe::snapshot().factories;
    let full_cap = cap(&input);
    let admission = quote(
        &input,
        IDS,
        ClearReferencePolicy::artifact_call_gelu(),
        full_cap,
    )
    .unwrap();
    let paid = input.consumed_work();
    drop(admission);
    assert_eq!(input.consumed_work(), paid);
    assert!(paid > initial);
    assert_eq!(observe::snapshot().factories, count);
    assert_eq!(root.budget().used_bytes(), base);
    let mut previous = paid;
    for _ in 0..40 {
        assert!(quote(
            &input,
            IDS,
            ClearReferencePolicy::artifact_call_gelu(),
            full_cap
        )
        .is_err());
        assert!(input.consumed_work() >= previous);
        previous = input.consumed_work();
    }
    assert_eq!(observe::snapshot().factories, count);
    // Measure builder cost through public observations, not a fixture-cost guess.
    let (sizing, _) = fixture(Fault::None, 1 << 20, owner::MAX_WORK);
    let build_work = sizing.consumed_work();
    for admitted_tickets in 0..3 {
        // Refuse at phase 1/2/3, retaining only earlier admitted ticket debits.
        let (short, root) = fixture(
            Fault::None,
            1 << 20,
            build_work + 32 * admitted_tickets + 31,
        );
        assert_eq!(short.consumed_work(), build_work);
        let base = root.budget().used_bytes();
        let count = observe::snapshot().factories;
        let paid = build_work + 32 * admitted_tickets;
        for _ in 0..40 {
            assert!(matches!(
                quote(
                    &short,
                    IDS,
                    ClearReferencePolicy::artifact_call_gelu(),
                    full_cap
                ),
                Err(Error::Owner(owner::Error::Work))
            ));
            assert_eq!(short.remaining_work(), 31);
            assert_eq!(short.consumed_work(), paid);
            assert_eq!(root.budget().used_bytes(), base);
            assert_eq!(observe::snapshot().factories, count);
        }
        // These PUBLIC probes themselves burn tickets before failing debit.
        // Same owner: exactly 32-k unused attempts, then Burned, proves FFN
        // refusals consumed no extra attempts (including repeated 31-work calls).
        for _ in 0..32 - admitted_tickets {
            assert!(matches!(short.begin_attempt(), Err(owner::Error::Work)));
            assert_eq!(short.remaining_work(), 31);
            assert_eq!(short.consumed_work(), paid);
            assert_eq!(root.budget().used_bytes(), base);
            assert_eq!(observe::snapshot().factories, count);
        }
        assert!(matches!(short.begin_attempt(), Err(owner::Error::Burned)));
        assert_eq!(short.consumed_work(), paid);
        assert_eq!(root.budget().used_bytes(), base);
        assert_eq!(observe::snapshot().factories, count);
    }
    // Exact 32/64/96 controls admit 1/2/3 early tickets, NOT whole quotes:
    // the last still fails later PREFLIGHT_CONTROL/CLEAR_WORK admission.
    for admitted_tickets in 1..=3 {
        let (exact, root) = fixture(
            Fault::None,
            1 << 20,
            build_work + 32 * admitted_tickets,
        );
        assert_eq!(exact.consumed_work(), build_work);
        let base = root.budget().used_bytes();
        let count = observe::snapshot().factories;
        assert!(matches!(
            quote(
                &exact,
                IDS,
                ClearReferencePolicy::artifact_call_gelu(),
                full_cap
            ),
            Err(Error::Owner(owner::Error::Work))
        ));
        let paid = build_work + 32 * admitted_tickets;
        assert_eq!(exact.remaining_work(), 0);
        assert_eq!(exact.consumed_work(), paid);
        // Deliberately burn remaining attempts through the same public seam.
        for _ in 0..32 - admitted_tickets {
            assert!(matches!(exact.begin_attempt(), Err(owner::Error::Work)));
            assert_eq!(exact.remaining_work(), 0);
            assert_eq!(exact.consumed_work(), paid);
        }
        assert!(matches!(exact.begin_attempt(), Err(owner::Error::Burned)));
        assert_eq!(exact.consumed_work(), paid);
        assert_eq!(root.budget().used_bytes(), base);
        assert_eq!(observe::snapshot().factories, count);
    }
}

#[test]
fn nonfinite_and_unwind_retire_initialized_partial_payloads_without_publication() {
    for fault in [Fault::Overflow, Fault::ActivationOverflow] {
        let (input, root) = fixture(fault, 1 << 20, owner::MAX_WORK);
        let base = root.budget().used_bytes();
        let before = observe::snapshot();
        let fixed = cap(&input);
        let admission = quote(&input, IDS, ClearReferencePolicy::artifact_call_gelu(), fixed).unwrap();
        let paid = input.consumed_work();
        assert_eq!(paid, fixed.max_work);
        assert!(matches!(execute(admission), Err(Error::NonFinite)));
        let after = observe::snapshot();
        assert!(after.wipes > before.wipes && after.partial > before.partial);
        assert_eq!(after.dirty, before.dirty);
        assert_eq!(root.budget().used_bytes(), base);
        assert_eq!(input.consumed_work(), paid);
        assert_eq!(quote(&input, IDS, ClearReferencePolicy::artifact_call_gelu(), fixed).err(), Some(Error::Work));
        assert_eq!(input.consumed_work(), paid);
        assert_eq!(observe::snapshot().factories, after.factories);
        remaining_attempts(&input, 3);
        assert_eq!(root.budget().used_bytes(), base);
    }
    for stage in [2, 3] {
        let (input, root) = fixture(Fault::None, 1 << 20, owner::MAX_WORK);
        let base = root.budget().used_bytes();
        let before = observe::snapshot();
        let admission = admitted(&input);
        let paid = input.consumed_work();
        observe::panic_at(before.factories + stage);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| execute(admission))).is_err()
        );
        let after = observe::snapshot();
        assert!(after.wipes > before.wipes && after.partial > before.partial);
        assert_eq!(after.dirty, before.dirty);
        assert_eq!(input.consumed_work(), paid);
        assert_eq!(root.budget().used_bytes(), base);
    }
}

#[test]
fn final_affine_bias_overflow_after_nonzero_output_write_burns_work_and_wipes() {
    // Explicitly synthetic public-builder fixture; original positive fixtures,
    // coefficient tables, policy, arithmetic order and admission stay unchanged.
    // Zero X/W1 and unit B1 make every first-affine output exactly 1, so u=1/64.
    //
    // Independent RATIONAL enclosure from the source decimal coefficients:
    // let c[k] = A_FIRST[31-k] interpreted as exact decimal rationals. Its cubic
    // prefix c0+c1/64+c2/64^2+c3/64^3 lies in (1519/10000,1520/10000).
    // Every remaining |c[k]| < 2312; hence the absolute tail is bounded by
    // 2312/(64^4*(1-1/64)) < 14/100000. Thus the complete inner polynomial
    // lies in (15176/100000,15214/100000), strictly inside (151/1000,153/1000).
    //
    // For the complete HALVED reversed outer tuple, let exact rational
    // d[k] = A_SECOND[27-k]/2, l=151/1000, h=153/1000. Sign-split monomial
    // bounds (including k=0) give the independently checkable rational sums
    // L = sum(d[k]*l^k for d[k]>=0) + sum(d[k]*h^k for d[k]<0),
    // H = sum(d[k]*h^k for d[k]>=0) + sum(d[k]*l^k for d[k]<0).
    // Substitution of the pinned decimal table gives 335/1000 < L <= H <
    // 348/1000. In particular, the halved outer value is in (1/4,1/2).
    // This is coefficient-based interval arithmetic, not observed native bits,
    // the ordinary tanh formula, or an ascending-order binary64 oracle.
    //
    // Rounding/finite margin for the EXISTING Stockmeyer expression tree:
    // both inputs have |x|<1; all coefficients have magnitude <30000. Powers
    // are <=1, each baby absolute sum <120000, each pair <240000, each second
    // <480000 and the final absolute sum <960000. All intermediates are finite.
    // At u=1/64 and outer x in [l,h], respectively, the rational absolute
    // monomial sums sum(|c[k]|*u^k), sum(|d[k]|*h^k) are both <1. Expanding
    // the actual tree, each monomial has fewer than 128 rounding factors,
    // including literal conversion and repeated powers (degree at most 31).
    // With binary64 unit roundoff e=2^-53, gamma_128=128e/(1-128e)<2^-40;
    // absolute evaluation error is therefore <2^-40. Nonzero terms/powers
    // here are normal; reversal is exact and halving is exact. These margins
    // preserve the inner (l,h) enclosure and outer (1/4,1/2) enclosure.
    // Recovery 64*u is exactly 1; the rounded factor 1/2+helper therefore
    // remains strictly in (3/4,1), as does each activation output a.
    //
    // With M=f64::MAX, diagonal-only W2 gives exactly one nonzero product
    // per contraction. Rounded a*M is finite, positive and nonzero, and all
    // other products/sums add zeros: no contraction overflow/cancellation.
    // Unpack initializes final output with nonzero values. Bias loop i=0
    // assigns finite a*M+0; i=1 computes a*M+M, beyond the binary64 overflow
    // threshold by a wide margin, and finite(...) returns NonFinite BEFORE
    // assigning that cell. No Output is constructed/published on this path.
    //
    // Source ancestry: final binding owns Activated, whose binding owns First,
    // while local zeroizing down is cleaned up on the affine error path.
    // Existing observations below cover nonzero payload wipes and final
    // reservation refund, NOT direct runtime cell-boundary/drop-order or
    // physical-deallocation proof. Runtime ownership-order remains unproven.
    let root = OwnedBudget::new(1 << 20).unwrap();
    let mut b =
        ClearInputBuilder::new(root.try_clone().unwrap(), caps(), owner::MAX_WORK).unwrap();
    let mut diagonal = [0.; 16];
    for i in 0..4 {
        diagonal[4 * i + i] = f64::MAX;
    }
    b.add_f64_tensor("x", &[2, 4], &[0.; 8]).unwrap();
    b.add_f64_tensor("u", &[4, 4], &[0.; 16]).unwrap();
    b.add_f64_tensor("v", &[4], &[1.; 4]).unwrap();
    b.add_f64_tensor("w", &[4, 4], &diagonal).unwrap();
    b.add_f64_tensor("b", &[4], &[0., f64::MAX, 0., 0.])
        .unwrap();
    b.add_operation("z", ModelOperator::Linear, None, &[], &[1])
        .unwrap();
    b.add_operation("up", ModelOperator::Linear, Some(7), &["x"], &[2, 4])
        .unwrap();
    b.add_operation("act", ModelOperator::GeluTanh, Some(7), &["up"], &[2, 4])
        .unwrap();
    b.add_operation("down", ModelOperator::Linear, Some(7), &["act"], &[2, 4])
        .unwrap();
    for (op, weight, bias) in [(1, "u", "v"), (3, "w", "b")] {
        for (key, value) in [
            ("weight", weight),
            ("bias", bias),
            ("compute_dtype", "float64"),
            ("weight_layout", "input_output"),
            ("formula", "x_matmul_weight_plus_bias"),
        ] {
            b.set_attribute(op, key, Attribute::Text(value)).unwrap();
        }
        b.set_attribute(op, "weight_shape", Attribute::Shape(&[4, 4]))
            .unwrap();
        b.set_attribute(op, "bias_shape", Attribute::Shape(&[4]))
            .unwrap();
    }
    b.set_attribute(2, "compute_dtype", Attribute::Text("float64"))
        .unwrap();
    let input = b.finish().unwrap();
    let base = root.budget().used_bytes();
    let build_work = input.consumed_work();
    let fixed = cap(&input); // Capture once from this genuine original ledger B.
    let before = observe::snapshot();
    let admission = quote(
        &input,
        IDS,
        ClearReferencePolicy::artifact_call_gelu(),
        fixed,
    )
    .unwrap();
    let paid = input.consumed_work();
    assert_eq!(paid, fixed.max_work);
    assert_eq!(paid - build_work, admission.resources().additional_work);
    assert!(matches!(execute(admission), Err(Error::NonFinite)));
    let after = observe::snapshot();
    assert_eq!(after.factories, before.factories + 3);
    assert!(after.wipes >= before.wipes + 3);
    assert!(after.partial >= before.partial + 3);
    assert_eq!(after.dirty, before.dirty);
    assert_eq!(input.consumed_work(), paid);
    assert_eq!(root.budget().used_bytes(), base);
}
