use super::*;
use serde_json::json;
use std::process::Command;

struct Fixture {
    ops: Vec<ModelOperation>,
    input_name: String,
    input: Vec<f64>,
    w: [Vec<f64>; 2],
    b: [Vec<f64>; 2],
    names: [String; 4],
}
impl Fixture {
    fn new() -> Self {
        let names = ["w1".into(), "b1".into(), "w2".into(), "b2".into()];
        let op = |id: &str, operator, input: &str, attributes| ModelOperation {
            id: id.into(),
            operator,
            layer: Some(0),
            state_kind: None,
            inputs: vec![input.into()],
            output_shape: vec![2, 4],
            attributes,
        };
        let affine = |w: &str, b: &str| json!({"weight":w,"bias":b,"weight_shape":[4,4],"bias_shape":[4],"weight_layout":"input_output","formula":"x_matmul_weight_plus_bias","compute_dtype":"float64"});
        let ops = vec![
            op("up", ModelOperator::Linear, "x", affine("w1", "b1")),
            op(
                "activation",
                ModelOperator::GeluTanh,
                "up",
                json!({"compute_dtype":"float64"}),
            ),
            op(
                "down",
                ModelOperator::Linear,
                "activation",
                affine("w2", "b2"),
            ),
        ];
        Self {
            ops,
            input_name: "x".into(),
            input: vec![0.125, 0.25, 0.375, 0.5, -0.25, 0.125, 0.0625, 0.375],
            w: [
                vec![
                    0.25, -0.125, 0.5, 0.0625, 0.125, 0.25, -0.25, 0.125, -0.25, 0.125, 0.0625,
                    0.5, 0.0625, -0.25, 0.125, 0.25,
                ],
                vec![
                    0.125, 0.25, -0.125, 0.0625, 0.25, -0.125, 0.0625, 0.125, -0.125, 0.0625, 0.25,
                    -0.25, 0.0625, 0.125, 0.25, 0.125,
                ],
            ],
            b: [
                vec![0.125, -0.25, 0.0625, 0.375],
                vec![-0.125, 0.0625, 0.25, -0.0625],
            ],
            names,
        }
    }
    fn policy<'a>(&'a self, g: &'a ExternalCharge<'a>) -> ClearReferencePolicy<'a> {
        ClearReferencePolicy::artifact_call_gelu(OriginalInputs {
            operations: &self.ops,
            input_name: &self.input_name,
            input: &self.input,
            sources: [
                Source {
                    operation: &self.ops[0],
                    weight_name: &self.names[0],
                    bias_name: &self.names[1],
                    weights: &self.w[0],
                    bias: &self.b[0],
                },
                Source {
                    operation: &self.ops[2],
                    weight_name: &self.names[2],
                    bias_name: &self.names[3],
                    weights: &self.w[1],
                    bias: &self.b[1],
                },
            ],
            original_guard: g,
        })
    }
}
fn cap() -> Limits {
    Limits {
        max_owner_bytes: 65536,
        max_phase_bytes: 65536,
        max_work: CLEAR_WORK,
    }
}
fn ledger() -> WorkLedger {
    WorkLedger::new(1 << 20, 1 << 20)
}
fn dense(x: &[f64], w: &[f64], bias: &[f64]) -> [f64; 8] {
    std::array::from_fn(|i| {
        let r = i / 4;
        let c = i % 4;
        let mut sum = 0.;
        for k in 0..4 {
            sum += x[r * 4 + k] * w[k * 4 + c];
        }
        sum + bias[c]
    })
}
fn python_artifact(xs: &[f64]) -> Vec<f64> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let script="import sys,struct;sys.path.insert(0,sys.argv[1]);from benchmarks.research.thor_literal_reconciliation import evaluate;print(' '.join(str(struct.unpack('>Q',struct.pack('>d',evaluate('gelu','A',float(x),'binary64')))[0]) for x in sys.argv[2:]))";
    let output = Command::new("python3")
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
fn complete_region_both_corrected_multiblock_boundaries_and_python_call_oracle() {
    let f = Fixture::new();
    let budget = Budget::new(1 << 20).unwrap();
    let guard = budget.charge_external(65536).unwrap();
    let mut state = State::Fresh;
    let mut work = ledger();
    let out = execute(f.policy(&guard), cap(), &budget, &mut state, &mut work).unwrap();
    // Dense, row-major, asymmetric two-row oracle; no packed contraction in expected.
    let first = dense(&f.input, &f.w[0], &f.b[0]);
    assert_ne!(&first[..4], &first[4..]);
    assert_eq!(out.first_affine(), &first); // Dyadics make this boundary exact.
    let activation = python_artifact(&first);
    assert_eq!(
        out.activated().map(f64::to_bits),
        std::array::from_fn::<_, 8, _>(|i| activation[i].to_bits())
    );
    let expected = dense(&activation, &f.w[1], &f.b[1]);
    for i in 0..8 {
        // IEEE dense-vs-two-part accumulation roundoff bound, not model quality.
        let magnitude = (0..4)
            .map(|k| (activation[(i / 4) * 4 + k] * f.w[1][k * 4 + i % 4]).abs())
            .sum::<f64>()
            + f.b[1][i % 4].abs();
        assert!((out.values()[i] - expected[i]).abs() <= 8. * f64::EPSILON * magnitude);
    }
    assert_eq!(state, State::Spent);
}

#[test]
fn no_paper_coefficients_wrong_order_unscaled_input_or_unhalved_outer_substitution() {
    // Four fixed cached-study controls, no sweep/quality threshold.
    for x in [-1., -0.125, 0.5, 2.] {
        let expected = python_artifact(&[x])[0];
        let mut w = poly::Workspace::empty();
        let first = std::array::from_fn::<_, 32, _>(|i| A_FIRST[31 - i]);
        let second = std::array::from_fn::<_, 28, _>(|i| A_SECOND[27 - i] * 0.5);
        let inner = poly::stockmeyer(&first, x / 64., &mut w).unwrap();
        let correct = (64. * (x / 64.)) * (0.5 + poly::stockmeyer(&second, inner, &mut w).unwrap());
        assert_eq!(correct.to_bits(), expected.to_bits());
        let wrong_order = poly::stockmeyer(&A_FIRST, x / 64., &mut w).unwrap();
        assert_ne!(wrong_order.to_bits(), inner.to_bits());
        let wrong_scale = poly::stockmeyer(&first, x, &mut w).unwrap();
        assert_ne!(wrong_scale.to_bits(), inner.to_bits());
        let unhalved = std::array::from_fn::<_, 28, _>(|i| A_SECOND[27 - i]);
        assert_ne!(
            (x * (0.5 + poly::stockmeyer(&unhalved, inner, &mut w).unwrap())).to_bits(),
            expected.to_bits()
        );
    }
}

#[test]
fn malformed_extra_ops_wrong_shape_dtype_attributes_and_foreign_binding_refuse() {
    let budget = Budget::new(1 << 20).unwrap();
    let guard = budget.charge_external(65536).unwrap();
    for kind in 0..5 {
        let mut f = Fixture::new();
        match kind {
            0 => f.ops.push(f.ops[2].clone()),
            1 => f.ops[0].output_shape = vec![1, 8],
            2 => f.ops[1].attributes["compute_dtype"] = json!("float32"),
            3 => f.ops[0].attributes["weight_layout"] = json!("output_input"),
            _ => f.ops[2].inputs[0] = "foreign".into(),
        }
        assert!(execute(
            f.policy(&guard),
            cap(),
            &budget,
            &mut State::Fresh,
            &mut ledger()
        )
        .is_err());
    }
    let f = Fixture::new();
    let foreign = f.ops[0].clone();
    let mut policy = f.policy(&guard);
    policy.original.sources[0].operation = &foreign;
    assert!(matches!(
        execute(policy, cap(), &budget, &mut State::Fresh, &mut ledger()),
        Err(Error::Source)
    ));
}

#[test]
fn exact_and_one_short_phase_with_output_and_original_owner_retention() {
    let f = Fixture::new();
    let b = Budget::new(1 << 20).unwrap();
    let g = b.charge_external(65536).unwrap();
    let q = quote(&f.policy(&g), cap(), &mut ledger()).unwrap();
    let exact = q.original_bytes + q.phase_bytes + 2 * size_of::<ExternalCharge<'_>>();
    for short in [true, false] {
        let budget = Budget::new(exact - usize::from(short)).unwrap();
        let guard = budget.charge_external(q.original_bytes).unwrap();
        let before = budget.used_bytes();
        let mut state = State::Fresh;
        let mut work = ledger();
        let result = execute(f.policy(&guard), cap(), &budget, &mut state, &mut work);
        if short {
            assert!(matches!(result, Err(Error::Resource)));
            assert_eq!(budget.used_bytes(), before);
        } else {
            let out = result.unwrap();
            assert_eq!(budget.used_bytes(), exact);
            assert_eq!(out.values().len(), 8);
            drop(out);
            assert_eq!(budget.used_bytes(), before);
        }
        assert_eq!(state, State::Spent);
    }
}

#[test]
fn exact_prior_work_one_short_and_failed_nonfinite_cleanup_never_refund_work() {
    let f = Fixture::new();
    let budget = Budget::new(1 << 20).unwrap();
    let guard = budget.charge_external(65536).unwrap();
    for short in [true, false] {
        let mut work = WorkLedger::new(
            PREFLIGHT_CONTROL + CLEAR_WORK + 17 - usize::from(short),
            1 << 20,
        );
        work.debit(17, 0).unwrap();
        let mut state = State::Fresh;
        let out = execute(f.policy(&guard), cap(), &budget, &mut state, &mut work);
        if short {
            assert!(matches!(out, Err(Error::Work)));
        } else {
            drop(out.unwrap());
            assert_eq!(work.remaining(), 0);
        }
        assert_eq!(state, State::Spent);
    }
    let mut bad = Fixture::new();
    bad.input[6] = f64::INFINITY;
    let baseline = budget.used_bytes();
    let mut work = ledger();
    let initial = work.remaining();
    let mut state = State::Fresh;
    assert!(matches!(
        execute(bad.policy(&guard), cap(), &budget, &mut state, &mut work),
        Err(Error::NonFinite)
    ));
    assert_eq!(budget.used_bytes(), baseline);
    assert_eq!(work.remaining(), initial - PREFLIGHT_CONTROL - CLEAR_WORK);
    assert!(matches!(
        execute(f.policy(&guard), cap(), &budget, &mut state, &mut work),
        Err(Error::Burned)
    ));
}

#[test]
fn foreign_counter_small_source_guard_and_actual_spare_capacities() {
    let mut f = Fixture::new();
    let b = Budget::new(1 << 20).unwrap();
    let guard = b.charge_external(65536).unwrap();
    let before = quote(&f.policy(&guard), cap(), &mut ledger())
        .unwrap()
        .original_bytes;
    f.w[0].reserve_exact(129);
    f.input.reserve_exact(33);
    let after = quote(&f.policy(&guard), cap(), &mut ledger())
        .unwrap()
        .original_bytes;
    assert!(after >= before + 129 * 8);
    let foreign = Budget::new(1 << 20).unwrap();
    assert!(matches!(
        execute(
            f.policy(&guard),
            cap(),
            &foreign,
            &mut State::Fresh,
            &mut ledger()
        ),
        Err(Error::Source)
    ));
    let tiny = b.charge_external(1).unwrap();
    assert!(matches!(
        execute(f.policy(&tiny), cap(), &b, &mut State::Fresh, &mut ledger()),
        Err(Error::Source)
    ));
}

#[test]
fn literal_complete_payload_and_transition_inventory_not_quote_padding() {
    // Ten data arrays104 + coefficient arrays60 + polynomial workspace17 doubles.
    assert_eq!(size_of::<Scratch>(), 181 * size_of::<f64>());
    let f = Fixture::new();
    let b = Budget::new(1 << 20).unwrap();
    let g = b.charge_external(65536).unwrap();
    let q = quote(&f.policy(&g), cap(), &mut ledger()).unwrap();
    assert!(
        q.phase_bytes
            >= 2 * size_of::<Output<'_>>()
                + size_of::<Result<Output<'_>, Error>>()
                + 2 * size_of::<Scratch>()
    );
    assert_eq!(q.schedules[0], q.schedules[1]);
    assert_eq!(q.schedules[0].tensor_products, 8);
    assert_eq!(q.schedules[0].input_rotations, 6);
}
