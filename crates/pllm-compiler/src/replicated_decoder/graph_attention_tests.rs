//! Literal pre-extraction compositions: expectations never call AttentionSink.
use super::super::super::source_norms;
use super::*;
use pllm_models::{ModelOperation, StateKind};

#[derive(Clone, Copy)]
struct Geometry {
    sequence: usize,
    position: usize,
    keys: usize,
    zero: (usize, usize),
}

fn operation(operator: Op) -> ModelOperation {
    ModelOperation {
        id: "original.attention.site".into(),
        operator,
        layer: Some(7),
        state_kind: Some(StateKind::Key),
        inputs: vec!["left".into(), "right".into()],
        output_shape: vec![],
        attributes: serde_json::json!({"group_size":2,"head_dim":2,"pairing":"split_half","theta":10000,"mode":"append"}),
    }
}

fn input(b: &mut Builder, name: &str, shape: &[usize]) -> View {
    let id = b
        .push(
            Code::Input(name.into()),
            shape.iter().product(),
            Interval::new(0, 0).unwrap(),
        )
        .unwrap();
    b.view(id, shape.to_vec())
}

fn setup(public: bool) -> (Builder, View, View, Geometry) {
    let mut b = Builder {
        public_weights: public,
        ..Builder::default()
    };
    let q = input(&mut b, "q", &[1, 2, 2, 2]);
    let k = input(&mut b, "k", &[1, 1, 3, 2]);
    let zero = b.constant(0, 0).unwrap();
    (
        b,
        q,
        k,
        Geometry {
            sequence: 2,
            position: 1,
            keys: 3,
            zero: (zero, 0),
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn original(
    b: &mut Builder,
    op: &ModelOperation,
    shape: &[usize],
    a: Option<&View>,
    c: Option<&View>,
    state: &mut BTreeMap<(u64, StateKind), View>,
    g: Geometry,
) -> Result<View, Error> {
    let Geometry {
        sequence,
        position,
        keys,
        zero,
    } = g;
    match op.operator {
        Op::RotaryEmbedding => {
            if shape.len() != 4
                || shape[3] > 128
                || op.attributes.get("frequency_scaling").is_some()
                || op.attributes.get("pairing").and_then(|v| v.as_str()) != Some("split_half")
            {
                return Err(Error::Binding);
            }
            let theta = op
                .attributes
                .get("theta")
                .and_then(|v| v.as_u64())
                .ok_or(Error::Binding)? as f32;
            let width = shape[3];
            let mut words = Words::new();
            for (index, row) in a
                .ok_or(Error::Binding)?
                .words
                .chunks_exact(width)
                .enumerate()
            {
                let (cos, sin) = tensor::Program::rotary_coefficients(
                    theta,
                    (position + index % sequence) as u32,
                    width,
                )?;
                let energy = if b.public_norms.is_empty() {
                    None
                } else {
                    b.row_norm(&row, width)
                        .or_else(|| b.attention_energy(&row, width))
                        .map(|energy| norm_bounds::rotary_energy(energy, &cos, &sin, 18))
                        .transpose()?
                };
                let input = b.gather(&row)?;
                let program =
                    tensor::Program::rotary_half(1, width, b.nodes[input].bounds, &cos, &sin)?;
                let output = b.tensor(input, program)?;
                if let Some(energy) = energy {
                    let row = (0..width).map(|i| (output, i)).collect();
                    b.record_attention_energy(&row, energy)?;
                }
                words.extend((0..width).map(|i| (output, i)));
            }
            Ok(View {
                words,
                shape: shape.to_vec(),
            })
        }
        Op::KvCacheAppend => {
            let key = (
                op.layer.ok_or(Error::Binding)?,
                op.state_kind.ok_or(Error::Binding)?,
            );
            let mode = op
                .attributes
                .get("mode")
                .and_then(|v| v.as_str())
                .ok_or(Error::Binding)?;
            let input = if position == 0 {
                if mode != "initialize" {
                    return Err(Error::Binding);
                }
                a
            } else {
                if mode != "append" {
                    return Err(Error::Binding);
                }
                c
            }
            .ok_or(Error::Binding)?;
            let (heads, width) = (input.shape[1], input.shape[3]);
            let mut words = Words::new();
            for head in 0..heads {
                if position > 0 {
                    let previous = state.get(&key).ok_or(Error::Binding)?;
                    if previous.shape != [1, heads, position, width] {
                        return Err(Error::Shape);
                    }
                    words.extend(
                        previous
                            .words
                            .slice(head * position * width..(head + 1) * position * width),
                    );
                }
                words.extend(
                    input
                        .words
                        .slice(head * sequence * width..(head + 1) * sequence * width),
                );
            }
            let out = View {
                words,
                shape: vec![1, heads, keys, width],
            };
            state.insert(key, out.clone());
            Ok(out)
        }
        Op::AttentionScores => {
            let q = a.ok_or(Error::Binding)?;
            let k = c.ok_or(Error::Binding)?;
            let (heads, width) = (q.shape[1], q.shape[3]);
            let group = op
                .attributes
                .get("group_size")
                .and_then(|v| v.as_u64())
                .ok_or(Error::Binding)? as usize;
            if group == 0 || heads / group != k.shape[1] {
                return Err(Error::Shape);
            }
            let mut words = Words::new();
            for h in 0..heads {
                for r in 0..sequence {
                    let query = View {
                        words: q
                            .words
                            .slice((h * sequence + r) * width..(h * sequence + r + 1) * width),
                        shape: vec![1, width],
                    };
                    let key_rows = View {
                        words: k.words.slice(
                            (h / group) * keys * width
                                ..((h / group) * keys + position + r + 1) * width,
                        ),
                        shape: vec![position + r + 1, width],
                    };
                    words.extend(
                        b.matrix_with_attention(&query, MatrixView::Values(&key_rows), true, true)?
                            .words,
                    );
                    words.extend(std::iter::repeat_n(zero, sequence - r - 1));
                }
            }
            Ok(View {
                words,
                shape: shape.to_vec(),
            })
        }
        Op::AttentionScale => {
            let width = op
                .attributes
                .get("head_dim")
                .and_then(|v| v.as_u64())
                .ok_or(Error::Binding)?;
            let coefficient = ((1.0_f32 / (width as f32).sqrt()) * ONE as f32) as i64;
            let mut words = Words::new();
            for chunk in a.ok_or(Error::Binding)?.words.chunks(128) {
                let input = b.gather(&chunk)?;
                let mut native = tensor::Builder::new(chunk.len(), b.nodes[input].bounds)?;
                let raw = native.affine(native.input(), None, affine(coefficient, 0, 0))?;
                let output = native.rescale(raw, 18)?;
                let id = b.tensor(input, native.finish(output)?)?;
                words.extend((0..chunk.len()).map(|i| (id, i)));
            }
            Ok(View {
                words,
                shape: shape.to_vec(),
            })
        }
        Op::Softmax => {
            let mut words = Words::new();
            for (i, row) in a
                .ok_or(Error::Binding)?
                .words
                .chunks_exact(keys)
                .enumerate()
            {
                let width = position + i % sequence + 1;
                let input = b.gather(&row.slice(..width))?;
                let (program, mass) = if b.public_weights {
                    let (program, mass) =
                        tensor::Program::softmax_certified(1, width, b.nodes[input].bounds)?;
                    (program, Some(mass))
                } else {
                    (
                        tensor::Program::softmax(1, width, b.nodes[input].bounds)?,
                        None,
                    )
                };
                let id = b.tensor(input, program)?;
                if let Some(mass) = mass {
                    b.admission_stage = "softmax_output_certificate";
                    b.charge_program_bytes(std::mem::size_of::<(usize, u64)>())?;
                    b.row_masses
                        .try_reserve_exact(1)
                        .map_err(|_| Error::Allocation)?;
                    b.row_masses.push((id, mass));
                }
                words.extend((0..width).map(|i| (id, i)));
                words.extend(std::iter::repeat_n(zero, keys - width));
            }
            Ok(View {
                words,
                shape: shape.to_vec(),
            })
        }
        Op::AttentionValues => {
            let p = a.ok_or(Error::Binding)?;
            let values = c.ok_or(Error::Binding)?;
            let (heads, width) = (shape[1], shape[3]);
            let group = op
                .attributes
                .get("group_size")
                .and_then(|v| v.as_u64())
                .ok_or(Error::Binding)? as usize;
            if group == 0 || heads / group != values.shape[1] {
                return Err(Error::Shape);
            }
            let mut words = Words::new();
            for h in 0..heads {
                for r in 0..sequence {
                    let valid = position + r + 1;
                    let p = View {
                        words: p
                            .words
                            .slice((h * sequence + r) * keys..(h * sequence + r) * keys + valid),
                        shape: vec![1, valid],
                    };
                    let mut selected = Words::new();
                    for d in 0..width {
                        for k in 0..valid {
                            selected.push(values.words.at(((h / group) * keys + k) * width + d));
                        }
                    }
                    words.extend(
                        b.linear(
                            &p,
                            &View {
                                words: selected,
                                shape: vec![width, valid],
                            },
                            true,
                        )?
                        .words,
                    );
                }
            }
            Ok(View {
                words,
                shape: shape.to_vec(),
            })
        }
        _ => Err(Error::Binding),
    }
}

#[allow(clippy::too_many_arguments)]
fn driven(
    b: &mut Builder,
    op: &ModelOperation,
    shape: &[usize],
    a: Option<&View>,
    c: Option<&View>,
    state: &mut BTreeMap<(u64, StateKind), View>,
    g: Geometry,
) -> Result<View, Error> {
    match op.operator {
        Op::RotaryEmbedding => attention_driver::rotary(b, op, shape, a, g.position, g.sequence),
        Op::KvCacheAppend => {
            attention_driver::kv_append::<Builder>(op, a, c, g.position, g.sequence, g.keys, state)
        }
        Op::AttentionScores => {
            attention_driver::scores(b, op, shape, a, c, g.sequence, g.position, g.keys, g.zero)
        }
        Op::AttentionScale => attention_driver::scale(b, op, shape, a),
        Op::Softmax => {
            attention_driver::softmax(b, shape, a, g.sequence, g.position, g.keys, g.zero)
        }
        Op::AttentionValues => {
            attention_driver::values(b, op, shape, a, c, g.sequence, g.position, g.keys)
        }
        _ => Err(Error::Binding),
    }
}

fn same(a: &Builder, e: &Builder) {
    assert_eq!(a.nodes.len(), e.nodes.len());
    for (a, e) in a.nodes.iter().zip(&e.nodes) {
        assert_eq!((a.lanes, a.bounds), (e.lanes, e.bounds));
        match (&a.code, &e.code) {
            (Code::Input(a), Code::Input(e)) => assert_eq!(a, e),
            (Code::Gather(a), Code::Gather(e)) => assert_eq!(a, e),
            (
                Code::Affine {
                    left: al,
                    right: ar,
                    coefficients: ac,
                },
                Code::Affine {
                    left: el,
                    right: er,
                    coefficients: ec,
                },
            ) => {
                let (mut a, mut e) = (Sha256::new(), Sha256::new());
                hash_affine(&mut a, *al, *ar, *ac);
                hash_affine(&mut e, *el, *er, *ec);
                assert_eq!(a.finalize(), e.finalize());
            }
            (
                Code::Tensor {
                    input: ai,
                    program: ap,
                },
                Code::Tensor {
                    input: ei,
                    program: ep,
                },
            ) => {
                assert_eq!(ai, ei);
                assert_eq!(ap.digest(), ep.digest());
                assert_eq!(ap.work_budget(), ep.work_budget());
            }
            (
                Code::Product {
                    left: al,
                    right: ar,
                    shape: ash,
                    rescale: ac,
                    norm: an,
                },
                Code::Product {
                    left: el,
                    right: er,
                    shape: esh,
                    rescale: ec,
                    norm: en,
                },
            ) => {
                let (mut a, mut e) = (Sha256::new(), Sha256::new());
                hash_product(&mut a, *al, *ar, *ash, *ac, an.as_deref());
                hash_product(&mut e, *el, *er, *esh, *ec, en.as_deref());
                assert_eq!(a.finalize(), e.finalize());
            }
            _ => panic!("original attention node kind/order changed"),
        }
    }
    for builder in [a, e] {
        for node in &builder.nodes {
            if let Code::Tensor { program, .. } = &node.code {
                assert!(Arc::ptr_eq(
                    program,
                    builder.tensors.get(&program.digest()).unwrap()
                ));
            }
        }
    }
    assert_eq!(
        schedule_work(&a.nodes).unwrap(),
        schedule_work(&e.nodes).unwrap()
    );
    assert_eq!(a.allocated, e.allocated);
    assert_eq!(a.workspace_bytes, e.workspace_bytes);
    assert_eq!(a.value_payload_bytes, e.value_payload_bytes);
    assert_eq!(a.validity, e.validity);
    assert_eq!(a.row_norms, e.row_norms);
    assert_eq!(a.row_energies, e.row_energies);
    assert_eq!(a.attention_energies, e.attention_energies);
    assert_eq!(a.row_masses, e.row_masses);
    assert_eq!(a.admission_ledger(), e.admission_ledger());
}

fn result(a: Result<View, Error>, e: Result<View, Error>) {
    match (a, e) {
        (Ok(a), Ok(e)) => {
            assert_eq!(a.shape, e.shape);
            assert_eq!(a.words, e.words);
        }
        (Err(a), Err(e)) => assert_eq!(a, e),
        _ => panic!("original attention result changed"),
    }
}

#[test]
fn rotary_rows_positions_widths_and_native_owner_custody_match_original() {
    for width in [2, 4, 128] {
        for sequence in [1, 2] {
            for position in [0, 3] {
                let (mut a, _, _, mut g) = setup(false);
                let (mut e, _, _, _) = setup(false);
                let shape = [1, 2, sequence, width];
                let av = input(&mut a, "rotary", &shape);
                let ev = input(&mut e, "rotary", &shape);
                g.sequence = sequence;
                g.position = position;
                let op = operation(Op::RotaryEmbedding);
                result(
                    driven(
                        &mut a,
                        &op,
                        &shape,
                        Some(&av),
                        None,
                        &mut BTreeMap::new(),
                        g,
                    ),
                    original(
                        &mut e,
                        &op,
                        &shape,
                        Some(&ev),
                        None,
                        &mut BTreeMap::new(),
                        g,
                    ),
                );
                same(&a, &e);
            }
        }
    }
    let (mut a, _, _, mut g) = setup(false);
    let view = input(&mut a, "owner", &[1, 2, 1, 2]);
    g.sequence = 1;
    g.position = 0;
    let start = a.nodes.len();
    driven(
        &mut a,
        &operation(Op::RotaryEmbedding),
        &view.shape,
        Some(&view),
        None,
        &mut BTreeMap::new(),
        g,
    )
    .unwrap();
    let natives: Vec<_> = a.nodes[start..]
        .iter()
        .filter_map(|n| {
            if let Code::Tensor { program, .. } = &n.code {
                Some(program.clone())
            } else {
                None
            }
        })
        .collect();
    assert_eq!(natives.len(), 2);
    assert!(Arc::ptr_eq(&natives[0], &natives[1]));
    let weak = Arc::downgrade(&natives[0]);
    drop(a);
    assert!(weak.upgrade().is_some());
    drop(natives);
    assert!(weak.upgrade().is_none());
}

#[test]
fn grouped_scores_values_and_causal_padding_preserve_original_coordinate_order() {
    for public in [false, true] {
        for group in [1, 2] {
            let (mut a, aq, _, g) = setup(public);
            let (mut e, eq, _, _) = setup(public);
            let av = input(&mut a, "values", &[1, 2 / group, 3, 2]);
            let ev = input(&mut e, "values", &[1, 2 / group, 3, 2]);
            let mut op = operation(Op::AttentionScores);
            op.attributes["group_size"] = serde_json::json!(group);
            let ar = driven(
                &mut a,
                &op,
                &[1, 2, 2, 3],
                Some(&aq),
                Some(&av),
                &mut BTreeMap::new(),
                g,
            )
            .unwrap();
            let er = original(
                &mut e,
                &op,
                &[1, 2, 2, 3],
                Some(&eq),
                Some(&ev),
                &mut BTreeMap::new(),
                g,
            )
            .unwrap();
            assert_eq!(ar.words.at(2), g.zero);
            result(Ok(ar.clone()), Ok(er.clone()));
            same(&a, &e);
            op.operator = Op::AttentionValues;
            result(
                driven(
                    &mut a,
                    &op,
                    &[1, 2, 2, 2],
                    Some(&ar),
                    Some(&av),
                    &mut BTreeMap::new(),
                    g,
                ),
                original(
                    &mut e,
                    &op,
                    &[1, 2, 2, 2],
                    Some(&er),
                    Some(&ev),
                    &mut BTreeMap::new(),
                    g,
                ),
            );
            same(&a, &e);
        }
    }
}

#[test]
fn scale_chunk_tails_and_softmax_prefix_mass_publication_match_original() {
    for public in [false, true] {
        let (mut a, _, _, g) = setup(public);
        let (mut e, _, _, _) = setup(public);
        let av = input(&mut a, "scale", &[1, 129]);
        let ev = input(&mut e, "scale", &[1, 129]);
        result(
            driven(
                &mut a,
                &operation(Op::AttentionScale),
                &av.shape,
                Some(&av),
                None,
                &mut BTreeMap::new(),
                g,
            ),
            original(
                &mut e,
                &operation(Op::AttentionScale),
                &ev.shape,
                Some(&ev),
                None,
                &mut BTreeMap::new(),
                g,
            ),
        );
        same(&a, &e);
        let av = input(&mut a, "probabilities", &[1, 2, 2, 3]);
        let ev = input(&mut e, "probabilities", &[1, 2, 2, 3]);
        result(
            driven(
                &mut a,
                &operation(Op::Softmax),
                &av.shape,
                Some(&av),
                None,
                &mut BTreeMap::new(),
                g,
            ),
            original(
                &mut e,
                &operation(Op::Softmax),
                &ev.shape,
                Some(&ev),
                None,
                &mut BTreeMap::new(),
                g,
            ),
        );
        same(&a, &e);
        if public {
            assert_eq!(a.row_masses.len(), 4);
        } else {
            assert!(a.row_masses.is_empty());
        }
    }
}

#[test]
fn kv_initialization_append_state_kind_and_head_mapping_match_original() {
    for position in [0, 1, 3] {
        for kind in [StateKind::Key, StateKind::Value] {
            let (mut a, _, _, mut g) = setup(false);
            let (mut e, _, _, _) = setup(false);
            g.position = position;
            g.keys = position + g.sequence;
            let av = input(&mut a, "next", &[1, 2, 2, 2]);
            let ev = input(&mut e, "next", &[1, 2, 2, 2]);
            let ap = input(&mut a, "past", &[1, 2, position.max(1), 2]);
            let ep = input(&mut e, "past", &[1, 2, position.max(1), 2]);
            let mut ast = BTreeMap::from([((7, kind), ap)]);
            let mut est = BTreeMap::from([((7, kind), ep)]);
            let mut op = operation(Op::KvCacheAppend);
            op.state_kind = Some(kind);
            op.attributes["mode"] = serde_json::json!(if position == 0 {
                "initialize"
            } else {
                "append"
            });
            let ar = driven(&mut a, &op, &[], Some(&av), Some(&av), &mut ast, g).unwrap();
            let er = original(&mut e, &op, &[], Some(&ev), Some(&ev), &mut est, g).unwrap();
            assert_eq!(ar.shape, [1, 2, g.keys, 2]);
            result(Ok(ar), Ok(er));
            assert_eq!(
                ast.get(&(7, kind)).unwrap().words,
                est.get(&(7, kind)).unwrap().words
            );
            same(&a, &e);
        }
    }
}

fn panic_payload(payload: Box<dyn std::any::Any + Send>) -> (bool, String) {
    if let Some(s) = payload.downcast_ref::<String>() {
        (true, s.clone())
    } else if let Some(s) = payload.downcast_ref::<&str>() {
        (false, (*s).into())
    } else {
        panic!("unexpected non-string original panic")
    }
}

#[test]
fn malformed_binding_shape_and_panic_payload_winners_match_before_publication() {
    for operator in [
        Op::RotaryEmbedding,
        Op::KvCacheAppend,
        Op::AttentionScores,
        Op::AttentionScale,
        Op::Softmax,
        Op::AttentionValues,
    ] {
        for case in 0..6 {
            let (mut a, mut av, mut ac, mut g) = setup(false);
            let (mut e, mut ev, mut ec, _) = setup(false);
            let mut op = operation(operator);
            let mut shape = vec![1, 2, 2, 2];
            match case {
                0 => op.attributes = serde_json::json!({}),
                1 => {
                    av.shape.clear();
                    ev.shape.clear();
                }
                2 => {
                    op.attributes["group_size"] = serde_json::json!(0);
                    ac.shape.clear();
                    ec.shape.clear();
                }
                3 => {
                    shape[3] = 0;
                    g.sequence = 0;
                }
                4 => {
                    op.layer = None;
                    op.state_kind = None;
                    shape.clear();
                }
                _ => {
                    op.attributes["frequency_scaling"] = serde_json::Value::Null;
                }
            }
            let ar = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                driven(
                    &mut a,
                    &op,
                    &shape,
                    if case == 0 { None } else { Some(&av) },
                    Some(&ac),
                    &mut BTreeMap::new(),
                    g,
                )
            }));
            let er = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                original(
                    &mut e,
                    &op,
                    &shape,
                    if case == 0 { None } else { Some(&ev) },
                    Some(&ec),
                    &mut BTreeMap::new(),
                    g,
                )
            }));
            match (ar, er) {
                (Ok(a), Ok(e)) => result(a, e),
                (Err(a), Err(e)) => assert_eq!(panic_payload(a), panic_payload(e)),
                _ => panic!("original panic/error winner changed"),
            }
            same(&a, &e);
        }
    }
}

#[test]
fn native_publication_capacity_refusal_keeps_original_partial_schedule_and_diagnostics() {
    for operator in [Op::RotaryEmbedding, Op::AttentionScale, Op::Softmax] {
        let (mut a, aq, _, g) = setup(true);
        let (mut e, eq, _, _) = setup(true);
        let av = if operator == Op::Softmax {
            input(&mut a, "softmax", &[1, 2, 2, 3])
        } else {
            aq
        };
        let ev = if operator == Op::Softmax {
            input(&mut e, "softmax", &[1, 2, 2, 3])
        } else {
            eq
        };
        a.allocated = MAX_PROGRAM_BYTES;
        e.allocated = MAX_PROGRAM_BYTES;
        let op = operation(operator);
        let shape = av.shape.clone();
        let ar = driven(
            &mut a,
            &op,
            &shape,
            Some(&av),
            None,
            &mut BTreeMap::new(),
            g,
        );
        let er = original(
            &mut e,
            &op,
            &shape,
            Some(&ev),
            None,
            &mut BTreeMap::new(),
            g,
        );
        assert!(ar.is_err());
        result(ar, er);
        same(&a, &e);
    }
}

#[test]
fn rotary_attention_only_certificates_never_become_general_row_proofs() {
    // Test-private synthetic ledger fixtures exercise the original priority and
    // proof-family separation; not range authority for an external input.
    for general in [false, true] {
        let (mut a, aq, _, g) = setup(true);
        let (mut e, eq, _, _) = setup(true);
        for (b, v) in [(&mut a, &aq), (&mut e, &eq)] {
            let mut scan = source_norms::BlockScan::new(1, 2, 1).unwrap();
            scan.append(0, &[0, 0]).unwrap();
            b.public_norms.insert(usize::MAX, scan.finish().unwrap());
            for row in v.words.chunks_exact(2) {
                if general {
                    b.record_row_energy(&row, 0).unwrap();
                }
                b.record_attention_energy(&row, 1).unwrap();
            }
        }
        let before = a.row_energies.clone();
        let op = operation(Op::RotaryEmbedding);
        let ar = driven(
            &mut a,
            &op,
            &aq.shape,
            Some(&aq),
            None,
            &mut BTreeMap::new(),
            g,
        )
        .unwrap();
        let er = original(
            &mut e,
            &op,
            &eq.shape,
            Some(&eq),
            None,
            &mut BTreeMap::new(),
            g,
        )
        .unwrap();
        for row in ar.words.chunks_exact(2) {
            assert!(a.attention_energy(&row, 2).is_some());
            assert_eq!(a.row_norm(&row, 2), None);
        }
        assert_eq!(a.row_energies, before);
        result(Ok(ar), Ok(er));
        same(&a, &e);
    }
}

#[test]
fn scores_use_attention_proofs_without_granting_general_projection_authority() {
    let (mut a, aq, ak, g) = setup(true);
    let (mut e, eq, ek, _) = setup(true);
    for (b, q, k) in [(&mut a, &aq, &ak), (&mut e, &eq, &ek)] {
        for row in q.words.chunks_exact(2).chain(k.words.chunks_exact(2)) {
            b.record_attention_energy(&row, 0).unwrap();
        }
    }
    let start = a.nodes.len();
    let op = operation(Op::AttentionScores);
    let ar = driven(
        &mut a,
        &op,
        &[1, 2, 2, 3],
        Some(&aq),
        Some(&ak),
        &mut BTreeMap::new(),
        g,
    )
    .unwrap();
    let er = original(
        &mut e,
        &op,
        &[1, 2, 2, 3],
        Some(&eq),
        Some(&ek),
        &mut BTreeMap::new(),
        g,
    )
    .unwrap();
    assert!(
        a.nodes[start..]
            .iter()
            .any(|n| matches!(&n.code, Code::Product { norm: Some(_), .. }))
    );
    assert!(a.row_energies.is_empty());
    result(Ok(ar), Ok(er));
    same(&a, &e);
}

#[test]
fn original_graph_entry_preserves_semantic_site_frontier_output_and_work() {
    use pllm_models::{DecoderMode, StateTensor};
    let (mut a, _, _, mut g) = setup(true);
    let (mut e, _, _, _) = setup(true);
    g.sequence = 1;
    g.keys = 2;
    let ap = input(&mut a, "probabilities", &[1, 2, 1, 2]);
    let ep = input(&mut e, "probabilities", &[1, 2, 1, 2]);
    let at = input(&mut a, "input.tokens", &[1, 1]);
    let et = input(&mut e, "input.tokens", &[1, 1]);
    let mut op = operation(Op::Softmax);
    op.inputs = vec!["probabilities".into()];
    op.output_shape = vec![1, 2, 1, 2];
    let graph = DecoderGraph {
        mode: DecoderMode::Decode,
        batch: 1,
        query_sequence: 1,
        maximum_key_sequence: 2,
        operations: vec![op.clone()],
        state_inputs: vec![StateTensor {
            id: "probabilities".into(),
            layer: Some(7),
            kind: StateKind::Key,
            shape: vec![1, 2, 1, 2],
            maximum_sequence: 2,
        }],
        state_outputs: vec![],
        output: op.id.clone(),
    };
    let mut state = BTreeMap::from([((7, StateKind::Key), ap)]);
    let mut site = AdmissionSite::default();
    let ar = a
        .graph(
            &graph,
            &at,
            &BTreeMap::new(),
            &BTreeMap::new(),
            3,
            1,
            &mut state,
            &mut site,
        )
        .unwrap();
    let anchor = e.gather(&et.words.slice(..1)).unwrap();
    let zero = e.constant(anchor, 0).unwrap();
    e.admission_stage = "operator_lowering";
    let bounds = e.bounds(&ep.words).unwrap();
    let er = original(
        &mut e,
        &op,
        &[1, 2, 1, 2],
        Some(&ep),
        None,
        &mut BTreeMap::new(),
        Geometry {
            zero: (zero, 0),
            ..g
        },
    )
    .unwrap();
    assert_eq!(site.step, Some(0));
    assert_eq!(site.operator, Some(Op::Softmax));
    assert_eq!(site.layer, Some(7));
    assert_eq!(site.inputs, [Some((bounds.lower(), bounds.upper())), None]);
    result(Ok(ar), Ok(er));
    same(&a, &e);
}

#[test]
fn kv_missing_wrong_state_and_inactive_input_errors_do_not_publish_replacements() {
    for case in 0..4 {
        let (mut a, aq, ak, mut g) = setup(false);
        let (mut e, eq, ek, _) = setup(false);
        let mut op = operation(Op::KvCacheAppend);
        let mut ast = BTreeMap::from([((7, StateKind::Key), aq.clone())]);
        let mut est = BTreeMap::from([((7, StateKind::Key), eq.clone())]);
        match case {
            0 => {
                ast.clear();
                est.clear();
            }
            1 => {} // present, but original cached head/sequence shape is wrong
            2 => {
                op.attributes["mode"] = serde_json::json!("initialize");
            }
            _ => {
                g.position = 0;
                op.attributes["mode"] = serde_json::json!("append");
            }
        }
        let ast_before = ast.clone();
        let est_before = est.clone();
        let ar = driven(&mut a, &op, &[], None, Some(&ak), &mut ast, g);
        let er = original(&mut e, &op, &[], None, Some(&ek), &mut est, g);
        assert!(ar.is_err());
        result(ar, er);
        same(&a, &e);
        for (after, before) in [(&ast, &ast_before), (&est, &est_before)] {
            assert_eq!(after.len(), before.len());
            for (key, value) in before {
                assert_eq!(after.get(key).unwrap().words, value.words);
                assert_eq!(after.get(key).unwrap().shape, value.shape);
            }
        }
    }
}

#[test]
fn softmax_mass_exact_one_short_preserves_failure_after_native_publication() {
    let mut probe = Builder {
        public_weights: true,
        ..Builder::default()
    };
    let p = input(&mut probe, "p", &[1, 1, 1, 2]);
    let g = Geometry {
        sequence: 1,
        position: 1,
        keys: 2,
        zero: (0, 0),
    };
    let op = operation(Op::Softmax);
    let before = probe.allocated;
    original(
        &mut probe,
        &op,
        &p.shape,
        Some(&p),
        None,
        &mut BTreeMap::new(),
        g,
    )
    .unwrap();
    let charge = probe.allocated - before;
    for short in [0, 1] {
        let mut a = Builder {
            public_weights: true,
            ..Builder::default()
        };
        let ap = input(&mut a, "p", &[1, 1, 1, 2]);
        let mut e = Builder {
            public_weights: true,
            ..Builder::default()
        };
        let ep = input(&mut e, "p", &[1, 1, 1, 2]);
        a.allocated = MAX_PROGRAM_BYTES - charge + short;
        e.allocated = a.allocated;
        let ar = driven(
            &mut a,
            &op,
            &ap.shape,
            Some(&ap),
            None,
            &mut BTreeMap::new(),
            g,
        );
        let er = original(
            &mut e,
            &op,
            &ep.shape,
            Some(&ep),
            None,
            &mut BTreeMap::new(),
            g,
        );
        if short == 0 {
            assert!(ar.is_ok());
            assert_eq!(a.allocated, MAX_PROGRAM_BYTES);
            assert_eq!(a.row_masses.len(), 1);
        } else {
            assert!(matches!(&ar, Err(Error::Bounds)));
            assert!(a.row_masses.is_empty());
            assert_eq!(a.admission_stage, "softmax_output_certificate");
            assert!(
                a.nodes
                    .iter()
                    .any(|n| matches!(&n.code, Code::Tensor { .. }))
            );
        }
        result(ar, er);
        same(&a, &e);
    }
}
