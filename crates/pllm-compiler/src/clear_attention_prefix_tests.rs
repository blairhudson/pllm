use super::*;
use crate::clear_input_owner::{Attribute, Caps, ClearInputBuilder};
use pllm_garble::shared_allocation::{OwnedBudget, OwnedBudgetResources};

// Literal source slices have static storage; no preconstructed Vec/String/JSON
// is adopted. All runtime original backing is built inside the neutral owner.
#[derive(Clone, Copy)]
struct Numbers {
    input: &'static [f64],
    weights: [&'static [f64]; 2],
    bias: [&'static [f64]; 2],
}
const BASE: Numbers = Numbers {
    input: &[0.125, -0.25, 0.5, 0.75, -0.5, 0.25, 0.125, -0.375],
    weights: [
        &[
            0.25, 0.5, -0.125, 0.75, -0.5, 0.125, 0.25, -0.25, 0.75, -0.25, 0.5, 0.125, 0.125,
            0.25, -0.75, 0.5,
        ],
        &[
            -0.125, 0.25, 0.5, -0.25, 0.5, -0.75, 0.125, 0.25, 0.25, 0.125, -0.5, 0.75, -0.5, 0.5,
            0.25, 0.125,
        ],
    ],
    bias: [&[0.125, -0.25, 0.5, -0.125], &[-0.25, 0.125, -0.125, 0.375]],
};
#[derive(Clone, Copy, PartialEq, Eq)]
enum Schema {
    Good,
    Dtype,
    WeightLayout,
    HeadLayout,
    Scaled,
    Group,
    WrongDependency,
    ScoreShape,
    ExtraOperation,
    ScaleOperator,
    WeightShape,
    TensorShape,
    WrongTensorKind,
    EighthAttribute,
}
fn caps() -> Caps {
    Caps {
        f64_values: 48,
        u64_values: 0,
        tensors: 5,
        operations: 5,
    }
}
fn limits() -> Limits {
    Limits {
        max_owner_bytes: 1 << 20,
        max_phase_bytes: 1 << 20,
        max_work: MAX_WORK,
        primitive: primitive(),
    }
}
fn policy(input: &ClearInputs) -> ClearReferencePolicy<'_> {
    ClearReferencePolicy::encoder_prefix(
        input,
        Bindings::canonical(),
        HeadReshape::BatchHeadsSequenceFeature,
        ScoreOutput::RawBidirectionalUnscaled,
    )
}
const TICKET_WORK: usize = 32; // shared begin_attempt's original-ledger debit
const INVOCATION_WORK: usize = TICKET_WORK + PREFLIGHT_CONTROL + CLEAR_WORK;
const ORACLE_WORK: usize = 2048;
const ORACLE_BYTES: usize = 8192;

struct Fixture {
    original: ClearInputs,
    // Observation only: NEVER accepted as source authority/ledger by the adapter.
    observation: OwnedBudget,
}
fn fixture(
    data: Numbers,
    schema: Schema,
    mut c: Caps,
    bytes: usize,
    work: usize,
) -> Result<Fixture, owner::Error> {
    if schema == Schema::ExtraOperation {
        c.operations = 6;
    }
    if schema == Schema::WrongTensorKind {
        c.u64_values = 8;
    }
    let budget = OwnedBudget::new(bytes).map_err(|_| owner::Error::Resource)?;
    let observation = budget.try_clone().map_err(|_| owner::Error::Resource)?;
    let mut b = ClearInputBuilder::new(budget, c, work)?;
    if schema == Schema::WrongTensorKind {
        b.add_u64_tensor("x", &[1, 2, 4], &[0; 8])?;
    } else {
        b.add_f64_tensor("x", &[1, 2, 4], data.input)?;
    }
    for (p, (weight, bias)) in [("qw", "qb"), ("kw", "kb")].into_iter().enumerate() {
        b.add_f64_tensor(
            weight,
            if schema == Schema::TensorShape {
                &[2, 8]
            } else {
                &[4, 4]
            },
            data.weights[p],
        )?;
        b.add_f64_tensor(bias, &[4], data.bias[p])?;
    }
    for (p, (id, weight, bias)) in [("q", "qw", "qb"), ("k", "kw", "kb")]
        .into_iter()
        .enumerate()
    {
        let op = b.add_operation(id, ModelOperator::Linear, Some(0), &["x"], &[1, 2, 4])?;
        b.set_attribute(op, "weight", Attribute::Text(weight))?;
        b.set_attribute(op, "bias", Attribute::Text(bias))?;
        b.set_attribute(
            op,
            "weight_shape",
            Attribute::Shape(if schema == Schema::WeightShape {
                &[2, 8]
            } else {
                &[4, 4]
            }),
        )?;
        b.set_attribute(op, "bias_shape", Attribute::Shape(&[4]))?;
        b.set_attribute(
            op,
            "weight_layout",
            Attribute::Text(if schema == Schema::WeightLayout {
                "output_input"
            } else {
                "input_output"
            }),
        )?;
        b.set_attribute(op, "formula", Attribute::Text("x_matmul_weight_plus_bias"))?;
        b.set_attribute(
            op,
            "compute_dtype",
            Attribute::Text(if schema == Schema::Dtype {
                "float32"
            } else {
                "float64"
            }),
        )?;
        if schema == Schema::EighthAttribute && p == 0 {
            b.set_attribute(op, "a", Attribute::Bool(true))?;
        }
    }
    for (id, input) in [
        ("qh", "q"),
        (
            "kh",
            if schema == Schema::WrongDependency {
                "q"
            } else {
                "k"
            },
        ),
    ] {
        let op = b.add_operation(id, ModelOperator::Reshape, Some(0), &[input], &[1, 2, 2, 2])?;
        b.set_attribute(
            op,
            "layout",
            Attribute::Text(if schema == Schema::HeadLayout {
                "batch_sequence_heads_feature"
            } else {
                "batch_heads_sequence_feature"
            }),
        )?;
        b.set_attribute(op, "input_layout", Attribute::Text("batch_sequence_hidden"))?;
    }
    let op = b.add_operation(
        "scores",
        if schema == Schema::ScaleOperator {
            ModelOperator::AttentionScale
        } else {
            ModelOperator::AttentionScores
        },
        Some(0),
        &["qh", "kh"],
        if schema == Schema::ScoreShape {
            &[2, 2, 2]
        } else {
            &[1, 2, 2, 2]
        },
    )?;
    b.set_attribute(
        op,
        "group_size",
        Attribute::Unsigned(if schema == Schema::Group { 2 } else { 1 }),
    )?;
    if schema == Schema::Scaled {
        b.set_attribute(op, "scale", Attribute::Real(0.5))?;
    }
    if schema == Schema::ExtraOperation {
        b.add_operation(
            "extra",
            ModelOperator::Reshape,
            Some(0),
            &["scores"],
            &[1, 2, 2, 2],
        )?;
    }
    Ok(Fixture {
        original: b.finish()?,
        observation,
    })
}
fn ordinary(data: Numbers) -> Fixture {
    fixture(data, Schema::Good, caps(), 1 << 20, MAX_WORK).unwrap()
}
fn run(input: &ClearInputs) -> Result<Output<'_>, Error> {
    execute(policy(input), limits())
}
fn root_resources() -> OwnedBudgetResources {
    OwnedBudget::resources().unwrap()
}
fn exact_bytes(c: Caps) -> usize {
    // The observation handle replaces the builder's consumed root handle: one
    // metadata charge remains. Builder peak is also below this phase bound.
    ClearInputBuilder::resources(c)
        .unwrap()
        .minimum_fresh_budget_bytes
        + PHASE_CHARGED_BYTES
}

// Independent ASCENDING affine: geometry-only for dyadic fixtures, never general
// bitwise parity. Read actual immutable owned payload; no candidate/kernel calls.
fn ascending_dense_geometry(f: &ClearInputs) -> ([[f64; 8]; 2], [f64; 8]) {
    let input = f.f64_tensor(0).unwrap();
    let weights = [f.f64_tensor(1).unwrap(), f.f64_tensor(3).unwrap()];
    let bias = [f.f64_tensor(2).unwrap(), f.f64_tensor(4).unwrap()];
    let mut heads = [[0.; 8]; 2];
    for p in 0..2 {
        for t in 0..2 {
            for c in 0..4 {
                let mut sum = 0.;
                for k in 0..4 {
                    let product = input[t * 4 + k] * weights[p][k * 4 + c];
                    sum += product;
                }
                heads[p][(c / 2 * 2 + t) * 2 + c % 2] = sum + bias[p][c];
            }
        }
    }
    let mut scores = [0.; 8];
    for h in 0..2 {
        for q in 0..2 {
            for k in 0..2 {
                let mut sum = 0.;
                for d in 0..2 {
                    let product = heads[0][(h * 2 + q) * 2 + d] * heads[1][(h * 2 + k) * 2 + d];
                    sum += product;
                }
                scores[(h * 2 + q) * 2 + k] = sum;
            }
        }
    }
    (heads, scores)
}
// Independent semantic coordinates for THIS artifact schedule: projection
// hidden-block order, within-block (token+l)%2; scores use feature (key+l)%2,
// wrapped/unwrapped partials and their separate final sum. No Layout/kernel calls.
fn artifact_route_oracle(f: &ClearInputs) -> ([[f64; 8]; 2], [f64; 8]) {
    let input = f.f64_tensor(0).unwrap();
    let weights = [f.f64_tensor(1).unwrap(), f.f64_tensor(3).unwrap()];
    let bias = [f.f64_tensor(2).unwrap(), f.f64_tensor(4).unwrap()];
    let mut heads = [[0.; 8]; 2];
    for p in 0..2 {
        for token in 0..2 {
            for output_feature in 0..4 {
                let mut sum = 0.;
                for block in 0..2 {
                    for route in 0..2 {
                        let hidden = block * 2 + (token + route) % 2;
                        let product =
                            input[token * 4 + hidden] * weights[p][hidden * 4 + output_feature];
                        sum += product;
                    }
                }
                heads[p][(output_feature / 2 * 2 + token) * 2 + output_feature % 2] =
                    sum + bias[p][output_feature];
            }
        }
    }
    let mut scores = [0.; 8];
    for h in 0..2 {
        for query in 0..2 {
            for key in 0..2 {
                let mut unwrapped = 0.;
                let mut wrapped = 0.;
                for route in 0..2 {
                    let feature = (key + route) % 2;
                    let product = heads[0][(h * 2 + query) * 2 + feature]
                        * heads[1][(h * 2 + key) * 2 + feature];
                    if key >= 2 - route {
                        wrapped += product;
                    } else {
                        unwrapped += product;
                    }
                }
                scores[(h * 2 + query) * 2 + key] = unwrapped + wrapped;
            }
        }
    }
    (heads, scores)
}
// Ordinary test calculations, reserved/debited through the SAME neutral owner,
// not another allocator/issuer or a caller-supplied ledger. No workspace inside
// quote callback; actual oracle buffers are created only after shared reserve.
fn independent_oracles(input: &ClearInputs) -> BorrowedOutput<'_, [[f64; 8]; 6]> {
    let (_, prepared) = input
        .begin_attempt()
        .unwrap()
        .with_ledger(|w| w.debit(ORACLE_WORK, 0).map_err(|_| owner::Error::Work))
        .unwrap();
    let phase = prepared.reserve::<[[f64; 8]; 6]>(ORACLE_BYTES).unwrap();
    let (a, sa) = ascending_dense_geometry(input);
    let (r, sr) = artifact_route_oracle(input);
    phase.finish([r[0], r[1], sr, a[0], a[1], sa])
}

#[test]
fn complete_prefix_independent_dense_bias_head_views_and_unscaled_bidirectional_scores() {
    let f = ordinary(BASE);
    let initial = f.original.consumed_work();
    let bytes = f.observation.budget().used_bytes();
    let expected = independent_oracles(&f.original);
    let e = expected.values();
    let out = run(&f.original).unwrap();
    assert_eq!(out.query(), &e[0]);
    assert_eq!(out.key(), &e[1]);
    assert_eq!(out.values(), &e[2]);
    assert_eq!(&e[..3], &e[3..]); // Dyadic geometry only.
    assert_ne!(&e[2][..4], &e[2][4..]);
    assert_ne!(e[2][1], 0.);
    assert_ne!(e[2][1], e[2][2]);
    assert_eq!(
        out.replicated_key_slots(),
        &out.inner.values().key_transposed
    );
    assert!(std::ptr::eq(out.original(), &f.original));
    assert_eq!(
        f.original.consumed_work(),
        initial + TICKET_WORK + ORACLE_WORK + INVOCATION_WORK
    );
    assert!(f.observation.budget().used_bytes() > bytes + PHASE_CHARGED_BYTES);
    drop(out);
    drop(expected);
    assert_eq!(f.observation.budget().used_bytes(), bytes);
    drop(f.original);
    assert_eq!(
        f.observation.budget().used_bytes(),
        root_resources().initial_reserved_bytes
    );
}

#[test]
fn token_one_projection_route_zero_differs_from_ascending_dense_one() {
    const DATA: Numbers = Numbers {
        input: &[0., 0., 0., 0., 1., 1., 1., 1.],
        weights: [
            &[
                1e16, 0., 0., 0., 1., 0., 0., 0., -1e16, 0., 0., 0., 1., 0., 0., 0.,
            ],
            &[0.; 16],
        ],
        bias: [&[0.; 4], &[0.; 4]],
    };
    let f = ordinary(DATA);
    let expected = independent_oracles(&f.original);
    // Token 1 [1,0,3,2]: (((0+1)+1e16)+1)-1e16 = +0.
    // Ascending [0,1,2,3]: (((0+1e16)+1)-1e16)+1 = 1.
    // +1 at 1e16 rounds away (binary64 ties-to-even).
    assert_eq!(expected.values()[0][2].to_bits(), 0.0f64.to_bits());
    assert_eq!(expected.values()[3][2].to_bits(), 1.0f64.to_bits());
    let out = run(&f.original).unwrap();
    assert_eq!(out.query(), &expected.values()[0]);
    assert_eq!(out.key(), &expected.values()[1]);
    assert_eq!(out.values(), &expected.values()[2]);
    assert_eq!(out.query()[2].to_bits(), 0.0f64.to_bits());
}

#[test]
fn biased_identity_projection_has_literal_head_orientation_and_raw_scores() {
    const ID: &[f64] = &[
        1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
    ];
    const DATA: Numbers = Numbers {
        input: &[1., 2., 3., 4., 5., 6., 7., 8.],
        weights: [ID, ID],
        bias: [&[0.5, -1., 2., -0.5], &[-2., 0.25, -3., 1.]],
    };
    const Q: [f64; 8] = [1.5, 1., 5.5, 5., 5., 3.5, 9., 7.5];
    const K: [f64; 8] = [-1., 2.25, 3., 6.25, 0., 5., 4., 9.];
    const S: [f64; 8] = [0.75, 10.75, 5.75, 47.75, 17.5, 51.5, 37.5, 103.5];
    let f = ordinary(DATA);
    let e = independent_oracles(&f.original);
    assert_eq!(e.values()[0], Q);
    assert_eq!(e.values()[1], K);
    assert_eq!(e.values()[2], S);
    let out = run(&f.original).unwrap();
    assert_eq!(out.query(), &Q);
    assert_eq!(out.key(), &K);
    assert_eq!(out.values(), &S);
}

#[test]
fn score_cancellation_preserves_wrapped_unwrapped_partials_and_final_add() {
    const DATA: Numbers = Numbers {
        input: &[1., 0., 0., 0., 0., 1., 0., 0.],
        weights: [
            &[
                1e16, 1., 1e16, -1e16, 1., 1., 1e16, 1e16, 0., 0., 0., 0., 0., 0., 0., 0.,
            ],
            &[
                1., -1e16, 1., 1., 1., -1., 1., -1., 0., 0., 0., 0., 0., 0., 0., 0.,
            ],
        ],
        bias: [&[0.; 4], &[0.; 4]],
    };
    const Q: [f64; 8] = [1e16, 1., 1., 1., 1e16, -1e16, 1e16, 1e16];
    const K: [f64; 8] = [1., -1e16, 1., -1., 1., 1., 1., -1.];
    // Key0: both unwrapped then +0; key1: feature1 unwrapped + feature0 wrapped.
    // Head0 [1e16-1e16, -1+1e16, 1-1e16, -1+1].
    // Head1 [1e16-1e16, 1e16+1e16, 1e16+1e16, -1e16+1e16].
    const S: [f64; 8] = [0., 1e16, -1e16, 0., 0., 2e16, 2e16, 0.];
    let f = ordinary(DATA);
    let e = independent_oracles(&f.original);
    assert_eq!(e.values()[0], Q);
    assert_eq!(e.values()[1], K);
    assert_eq!(e.values()[2], S);
    let out = run(&f.original).unwrap();
    assert_eq!(out.query(), &Q);
    assert_eq!(out.key(), &K);
    for (actual, expected) in out.values().iter().zip(S) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

#[test]
fn distinct_head_isolation_after_cross_hidden_block_projection() {
    const SECOND: Numbers = Numbers {
        bias: [&[0.125, -0.25, 2.5, -0.125], BASE.bias[1]],
        ..BASE
    };
    let a = ordinary(BASE);
    let b = ordinary(SECOND); // genuinely new owned backing, never a refill
    let before = run(&a.original).unwrap();
    let expected = independent_oracles(&b.original);
    let after = run(&b.original).unwrap();
    assert_eq!(&before.values()[..4], &after.values()[..4]);
    assert_ne!(&before.values()[4..], &after.values()[4..]);
    assert_eq!(after.values(), &expected.values()[2]);
}

#[test]
fn corrected_multiframe_routes_ignore_poison_padding_without_cross_head_mix() {
    // Existing primitive regression, admitted through the neutral original owner
    // too. No standalone THOR budget/ledger or unchecked test buffer factory.
    let f = ordinary(BASE);
    let (_, prepared) = f
        .original
        .begin_attempt()
        .unwrap()
        .with_ledger(|w| w.debit(8192, 0).map_err(|_| owner::Error::Work))
        .unwrap();
    let phase = prepared.reserve::<[f64; 32]>(32768).unwrap();
    let got = {
        let lim = matrix::Limits {
            max_buffer_elements: 64,
            max_work: 4096,
        };
        let l = matrix::Layout::new(4, 4, 3, 3, 2, 2, matrix::Diagonal::Lower, lim).unwrap();
        let mut a = [0.; 32];
        let mut b = [0.; 32];
        for h in 0..2 {
            for r in 0..3 {
                for c in 0..3 {
                    a[(h * 4 + r) * 4 + c] = (1 + h * 17 + r * 3 + c) as f64;
                    b[(h * 4 + r) * 4 + c] = (2 + h * 11 + r + c * 4) as f64;
                }
            }
        }
        let mut ap = [0.; 32];
        let mut bp = [0.; 32];
        l.pack(&a, &mut ap, lim).unwrap();
        l.pack(&b, &mut bp, lim).unwrap();
        for i in 0..32 {
            if l.lane(i).unwrap().is_none() {
                ap[i] = f64::NAN;
                bp[i] = f64::INFINITY;
            }
        }
        let mut scratch = [0.; 64];
        let mut slots = [0.; 32];
        let mut got = [0.; 32];
        matrix::optimized_lower_mm(l, &ap, l, &bp, l, &mut scratch, &mut slots, lim).unwrap();
        l.unpack(&slots, &mut got, lim).unwrap();
        for h in 0..2 {
            for r in 0..4 {
                for c in 0..4 {
                    let expected = if r < 3 && c < 3 {
                        (0..3)
                            .map(|k| a[(h * 4 + r) * 4 + k] * b[(h * 4 + k) * 4 + c])
                            .sum()
                    } else {
                        0.
                    };
                    assert_eq!(got[(h * 4 + r) * 4 + c], expected);
                }
            }
        }
        for (i, &slot) in slots.iter().enumerate() {
            if l.lane(i).unwrap().is_none() {
                assert_eq!(slot, 0.);
            }
        }
        got
    }; // All transient layouts/arrays retire while the original phase is live.
    let retained = phase.finish(got);
    assert!(std::ptr::eq(retained.original(), &f.original));
}

#[test]
fn unknown_dtype_shape_layout_scaled_scores_extra_ops_and_foreign_source_refused() {
    for schema in [
        Schema::Dtype,
        Schema::WeightLayout,
        Schema::HeadLayout,
        Schema::Scaled,
        Schema::Group,
        Schema::WrongDependency,
        Schema::ScoreShape,
        Schema::ExtraOperation,
        Schema::ScaleOperator,
        Schema::WeightShape,
        Schema::TensorShape,
        Schema::WrongTensorKind,
    ] {
        let f = fixture(BASE, schema, caps(), 1 << 20, MAX_WORK).unwrap();
        let before = f.original.consumed_work();
        let bytes = f.observation.budget().used_bytes();
        assert!(run(&f.original).is_err());
        assert_eq!(f.original.consumed_work(), before + INVOCATION_WORK);
        assert_eq!(f.observation.budget().used_bytes(), bytes);
    }
    let f = ordinary(BASE);
    let mut p = policy(&f.original);
    p.bindings.query_weights = p.bindings.key_weights; // cannot import foreign payload
    assert!(matches!(execute(p, limits()), Err(Error::Source)));
}

#[test]
fn exact_memory_one_short_retained_output_and_original_owner_charge() {
    for short in [true, false] {
        let f = fixture(
            BASE,
            Schema::Good,
            caps(),
            exact_bytes(caps()) - usize::from(short),
            MAX_WORK,
        )
        .unwrap();
        let before = f.original.consumed_work();
        let bytes = f.observation.budget().used_bytes();
        let result = run(&f.original);
        if short {
            assert!(matches!(result, Err(Error::Resource)));
        } else {
            let out = result.unwrap();
            assert_eq!(f.observation.budget().used_bytes(), exact_bytes(caps()));
            assert_eq!(out.inner.phase_bytes(), PHASE_CHARGED_BYTES);
            assert_eq!(
                out.original().charged_bytes(),
                ClearInputBuilder::resources(caps())
                    .unwrap()
                    .charged_owner_bytes
            );
            drop(out);
        }
        // Reservation failure burns original prepaid work too; no refund.
        assert_eq!(f.original.consumed_work(), before + INVOCATION_WORK);
        assert_eq!(f.observation.budget().used_bytes(), bytes);
    }
    let q = ClearInputBuilder::resources(caps()).unwrap();
    assert!(matches!(
        ClearInputBuilder::new(
            OwnedBudget::new(q.minimum_fresh_budget_bytes - 1).unwrap(),
            caps(),
            MAX_WORK
        ),
        Err(owner::Error::Resource)
    ));
}

#[test]
fn exact_cumulative_work_one_short_prior_debits_and_control_work() {
    let construction = {
        let f = ordinary(BASE);
        f.original.consumed_work()
    };
    for short in [true, false] {
        let total = construction + INVOCATION_WORK - usize::from(short);
        let f = fixture(BASE, Schema::Good, caps(), exact_bytes(caps()), total).unwrap();
        let mut c = limits();
        c.max_work = total;
        let result = execute(policy(&f.original), c);
        if short {
            assert!(matches!(result, Err(Error::Work)));
            assert_eq!(f.original.consumed_work(), construction + TICKET_WORK);
        } else {
            drop(result.unwrap());
            assert_eq!(f.original.remaining_work(), 0);
        }
    }
    let q = ClearInputBuilder::resources(caps()).unwrap();
    assert!(matches!(
        ClearInputBuilder::new(
            OwnedBudget::new(1 << 20).unwrap(),
            caps(),
            q.construction_work - 1
        ),
        Err(owner::Error::Work)
    ));
    assert!(matches!(
        ClearInputBuilder::new(OwnedBudget::new(1 << 20).unwrap(), caps(), MAX_WORK + 1),
        Err(owner::Error::Work)
    ));
}

#[test]
fn nonfinite_input_weight_bias_and_overflow_burn_cleanup_without_refund() {
    // Nonfinite raw sources are now rejected at shared construction, before any
    // THOR policy can exist. Failed builder is burned, not a partially usable input.
    for case in 0..3 {
        let mut b =
            ClearInputBuilder::new(OwnedBudget::new(1 << 20).unwrap(), caps(), MAX_WORK).unwrap();
        let failed = match case {
            0 => b.add_f64_tensor("x", &[1], &[f64::NAN]),
            1 => b.add_f64_tensor("qw", &[1], &[f64::INFINITY]),
            _ => b.add_f64_tensor("qb", &[1], &[f64::NEG_INFINITY]),
        };
        assert_eq!(failed, Err(owner::Error::NonFinite));
        assert!(matches!(b.finish(), Err(owner::Error::Burned)));
    }
    const OVERFLOW: Numbers = Numbers {
        input: &[f64::MAX, -0.25, 0.5, 0.75, -0.5, 0.25, 0.125, -0.375],
        weights: [
            &[
                2., 0.5, -0.125, 0.75, -0.5, 0.125, 0.25, -0.25, 0.75, -0.25, 0.5, 0.125, 0.125,
                0.25, -0.75, 0.5,
            ],
            BASE.weights[1],
        ],
        ..BASE
    };
    const SCORE_OVERFLOW: Numbers = Numbers {
        bias: [&[f64::MAX; 4], &[f64::MAX; 4]],
        ..BASE
    };
    for data in [OVERFLOW, SCORE_OVERFLOW] {
        let f = ordinary(data);
        let before = f.original.consumed_work();
        let bytes = f.observation.budget().used_bytes();
        assert!(matches!(run(&f.original), Err(Error::NonFinite)));
        assert_eq!(f.original.consumed_work(), before + INVOCATION_WORK);
        assert_eq!(f.observation.budget().used_bytes(), bytes);
        let consumed = f.original.consumed_work();
        assert!(matches!(run(&f.original), Err(Error::Work)));
        assert_eq!(f.original.consumed_work(), consumed + TICKET_WORK);
    }
}

#[test]
fn guard_identity_spare_capacity_phase_owner_caps_and_tighter_primitive_caps() {
    let mut spare = caps();
    spare.f64_values += 16;
    let f = fixture(BASE, Schema::Good, spare, 1 << 20, MAX_WORK).unwrap();
    let original_bytes = f.original.charged_bytes();
    assert_eq!(
        original_bytes,
        ClearInputBuilder::resources(spare)
            .unwrap()
            .charged_owner_bytes
    );
    assert_eq!(
        original_bytes,
        ClearInputBuilder::resources(caps())
            .unwrap()
            .charged_owner_bytes
            + 16 * 8
    );
    let ptr = f.original.f64_tensor(0).unwrap().as_ptr();
    let quoted = quote(policy(&f.original), limits()).unwrap();
    let q = quoted.resources();
    assert_eq!(q.original_bytes, original_bytes);
    assert_eq!(q.phase_bytes, PHASE_CHARGED_BYTES);
    assert_eq!(q.score_schedule.tensor_products, 4);
    assert_eq!(q.score_schedule.input_rotations, 2);
    assert_eq!(q.score_schedule.output_rotations, 2);
    let out = quoted.execute().unwrap();
    assert_eq!(out.original().f64_tensor(0).unwrap().as_ptr(), ptr);
    drop(out);
    assert_eq!(f.original.f64_tensor(0).unwrap().as_ptr(), ptr);
    assert_eq!(f.original.charged_bytes(), original_bytes);
    for case in 0..7 {
        let f = ordinary(BASE);
        let mut c = limits();
        match case {
            0 => c.max_owner_bytes = f.original.charged_bytes() - 1,
            1 => c.max_phase_bytes = PHASE_CHARGED_BYTES - 1,
            2 => c.primitive.max_buffer_elements = 15,
            3 => c.primitive.max_work = 79,
            4 => c.max_work = f.original.consumed_work() + INVOCATION_WORK - 1,
            5 => c.max_work = MAX_WORK + 1,
            _ => c.max_owner_bytes = MAX_ALLOCATION_BYTES + 1,
        }
        assert!(execute(policy(&f.original), c).is_err());
    }
    assert_eq!(size_of::<Scratch>(), 144 * size_of::<f64>());
}

#[test]
fn oversized_short_key_attributes_refused_count_first_discriminator_bounded_by_owner() {
    let f = fixture(BASE, Schema::EighthAttribute, caps(), 1 << 20, MAX_WORK).unwrap();
    let before = f.original.consumed_work();
    assert_eq!(
        f.original.operations()[0]
            .attributes
            .as_object()
            .unwrap()
            .len(),
        8
    );
    assert!(matches!(run(&f.original), Err(Error::Policy)));
    assert_eq!(f.original.consumed_work(), before + INVOCATION_WORK);
    // Public constructor makes unbounded maps impossible: ninth key is refused
    // before materialization, and the source cannot finish after failed admission.
    // This proves constructor bound, not timing/order of two same-Policy checks.
    let mut b =
        ClearInputBuilder::new(OwnedBudget::new(1 << 20).unwrap(), caps(), MAX_WORK).unwrap();
    let op = b
        .add_operation("q", ModelOperator::Linear, Some(0), &["x"], &[1, 2, 4])
        .unwrap();
    for key in ["a", "b", "c", "d", "e", "f", "g", "h"] {
        b.set_attribute(op, key, Attribute::Bool(true)).unwrap();
    }
    assert_eq!(
        b.set_attribute(op, "i", Attribute::Bool(true)),
        Err(owner::Error::Shape)
    );
    assert!(matches!(b.finish(), Err(owner::Error::Burned)));
}

#[test]
fn prior_ledger_work_counts_against_cumulative_cap_and_cannot_be_refilled() {
    let old = ordinary(BASE);
    let initial = old.original.consumed_work();
    let out = run(&old.original).unwrap();
    drop(out);
    let consumed = old.original.consumed_work();
    assert_eq!(consumed, initial + INVOCATION_WORK);
    let fresh = ordinary(BASE); // new owned copies, not an issuer for OLD buffers
    assert_ne!(
        old.original.f64_tensor(0).unwrap().as_ptr(),
        fresh.original.f64_tensor(0).unwrap().as_ptr()
    );
    // Even a real SAME-budget extra guard cannot confer fresh source/work credit.
    // It is not an accepted argument (compile-fail public API control on Output).
    let spectator = old.observation.budget().charge_external(1).unwrap();
    assert_eq!(old.original.consumed_work(), consumed);
    assert!(matches!(run(&old.original), Err(Error::Work)));
    assert_eq!(old.original.consumed_work(), consumed + TICKET_WORK);
    drop(spectator);
    assert!(run(&fresh.original).is_ok());
}

#[test]
fn one_short_preflight_storage_refuses_before_validation_and_burns_region() {
    let f = fixture(
        BASE,
        Schema::EighthAttribute,
        caps(),
        exact_bytes(caps()) - 1,
        MAX_WORK,
    )
    .unwrap();
    let before = f.original.consumed_work();
    let bytes = f.observation.budget().used_bytes();
    // Schema would return Policy if views were inspected before reservation.
    assert!(matches!(run(&f.original), Err(Error::Resource)));
    assert_eq!(f.original.consumed_work(), before + INVOCATION_WORK);
    assert_eq!(f.observation.budget().used_bytes(), bytes);
    assert!(matches!(run(&f.original), Err(Error::Work))); // no refunded failed attempt
}

#[test]
fn nonfinite_burn_retains_original_charge_and_prepaid_retirement_on_same_ledger() {
    const OVERFLOW: Numbers = Numbers {
        bias: [&[f64::MAX; 4], &[f64::MAX; 4]],
        ..BASE
    };
    let mut spare = caps();
    spare.f64_values += 16;
    let f = fixture(OVERFLOW, Schema::Good, spare, 1 << 20, MAX_WORK).unwrap();
    let before = f.original.consumed_work();
    let bytes = f.observation.budget().used_bytes();
    let ptr = f.original.f64_tensor(0).unwrap().as_ptr();
    assert!(matches!(run(&f.original), Err(Error::NonFinite)));
    assert_eq!(f.original.consumed_work(), before + INVOCATION_WORK);
    assert_eq!(f.observation.budget().used_bytes(), bytes);
    assert_eq!(f.original.f64_tensor(0).unwrap().as_ptr(), ptr);
    assert_eq!(
        f.original.charged_bytes(),
        ClearInputBuilder::resources(spare)
            .unwrap()
            .charged_owner_bytes
    );
    let consumed = f.original.consumed_work();
    drop(f.original);
    assert_eq!(
        f.observation.budget().used_bytes(),
        root_resources().initial_reserved_bytes
    );
    // No standalone ledger exists to reset/refund after actual owner retirement.
    assert!(consumed <= MAX_WORK);
}

#[test]
fn failed_quotes_and_dropped_quotes_burn_original_attempts_and_work() {
    let f = ordinary(BASE);
    let initial = f.original.consumed_work();
    let bytes = f.observation.budget().used_bytes();
    let q = quote(policy(&f.original), limits()).unwrap();
    assert_eq!(f.original.consumed_work(), initial + INVOCATION_WORK);
    drop(q);
    assert_eq!(f.observation.budget().used_bytes(), bytes);
    assert!(matches!(
        quote(policy(&f.original), limits()),
        Err(Error::Work)
    ));
    assert_eq!(
        f.original.consumed_work(),
        initial + INVOCATION_WORK + TICKET_WORK
    );
    let bad = fixture(BASE, Schema::Scaled, caps(), 1 << 20, MAX_WORK).unwrap();
    let before = bad.original.consumed_work();
    assert!(matches!(
        quote(policy(&bad.original), limits()),
        Err(Error::Policy)
    ));
    assert_eq!(bad.original.consumed_work(), before + INVOCATION_WORK);
    assert!(matches!(
        quote(policy(&bad.original), limits()),
        Err(Error::Work)
    ));
}

#[test]
fn repeated_exhausted_attempts_reach_shared_burn_limit_without_refill() {
    let f = ordinary(BASE);
    let initial = f.original.consumed_work();
    let ptr = f.original.f64_tensor(0).unwrap().as_ptr();
    drop(run(&f.original).unwrap());
    let bytes = f.observation.budget().used_bytes();
    for failed in 1..32 {
        assert!(matches!(run(&f.original), Err(Error::Work)));
        assert_eq!(
            f.original.consumed_work(),
            initial + INVOCATION_WORK + failed * TICKET_WORK
        );
        assert_eq!(f.observation.budget().used_bytes(), bytes);
        assert_eq!(f.original.f64_tensor(0).unwrap().as_ptr(), ptr);
    }
    let consumed = f.original.consumed_work();
    assert!(matches!(run(&f.original), Err(Error::Burned)));
    assert_eq!(f.original.consumed_work(), consumed);
    assert!(consumed <= MAX_WORK);
}
