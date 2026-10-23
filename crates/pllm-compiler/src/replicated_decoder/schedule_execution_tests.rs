//! Real execution, TEST-ONLY original expanded custody; not a production issuer.
//! The existing insecure-test-support factory supplies PUBLIC test keys to two
//! isolated fresh graphs. No live runtime is copied/reset; never use real secrets.
use super::super::*;
use super::*;
use crate::replicated_decoder::runtime::frozen_execution_tests::{
    with_oracle, Qualified, QUEUE_FRAMES,
};
use crate::replicated_decoder::runtime::{PeerBytes, Worker};
use puma::{Context, Party};
use std::collections::BTreeMap;

fn input_builder(lanes: usize, offset: usize) -> (Builder, View, View) {
    // Whole Input declarations; nonzero offsets are complete leading rows.
    // Compact import must not retroactively certify an arbitrary sliced Input.
    let padding = offset * lanes;
    let mut builder = Builder {
        public_weights: true,
        ..Builder::default()
    };
    let bounds = Interval::new(-ONE / 4, ONE / 4).unwrap();
    let left = builder
        .push(Code::Input("left".into()), lanes + padding, bounds)
        .unwrap();
    let right = builder
        .push(Code::Input("right".into()), lanes + padding, bounds)
        .unwrap();
    let view = |id| View {
        words: (padding..padding + lanes).map(|i| (id, i)).collect(),
        shape: vec![1, 1, lanes],
    };
    (builder, view(left), view(right))
}

fn graph(lanes: usize, padded: bool) -> DecoderGraph {
    let config = serde_json::to_vec(&serde_json::json!({"model_type":"llama","hidden_size":2,
        "intermediate_size":lanes,"num_hidden_layers":1,"num_attention_heads":1,
        "num_key_value_heads":1,"head_dim":2,"vocab_size":3,"tie_word_embeddings":false,
        "hidden_act":"silu","attention_bias":false,"max_position_embeddings":64,
        "rope_theta":10000,"rms_norm_eps":0.000001,"attention_dropout":0.0,"use_cache":true}))
    .unwrap();
    let mut graph = lower_with_weights(&config, 1, 1, WeightMode::Public)
        .unwrap()
        .prefill;
    let mut silu = graph
        .operations
        .iter()
        .find(|op| op.operator == Op::Silu)
        .unwrap()
        .clone();
    let mut product = graph
        .operations
        .iter()
        .find(|op| op.operator == Op::Multiply)
        .unwrap()
        .clone();
    let mut sum = graph
        .operations
        .iter()
        .find(|op| op.operator == Op::ResidualAdd)
        .unwrap()
        .clone();
    silu.id = "silu".into();
    silu.inputs = vec!["left".into()];
    product.id = "product".into();
    product.inputs = vec!["silu".into(), "right".into()];
    product.output_shape = silu.output_shape.clone();
    sum.id = "sum".into();
    sum.inputs = vec!["product".into(), "right".into()];
    sum.output_shape = silu.output_shape.clone();
    let mut alias = silu.clone();
    alias.id = "alias".into();
    alias.inputs = vec!["sum".into()];
    alias.operator = Op::Reshape;
    alias.attributes = serde_json::json!({"layout":"batch_heads_sequence_feature"});
    alias.output_shape = vec![1, 1, 1, lanes as u64];
    if padded {
        // Real suffix aliases precede arithmetic; their rows are original Input
        // coordinates. No late partial import or arbitrary identity Reshape.
        let suffix = |id: &str, input: &str| {
            let mut op = silu.clone();
            op.id = id.into();
            op.inputs = vec![input.into()];
            op.operator = Op::LastToken;
            op.output_shape = vec![1, lanes as u64];
            op.attributes = serde_json::json!({"axis":1,"selection":"last_valid"});
            op
        };
        let left = suffix("left.tail", "left");
        let right = suffix("right.tail", "right");
        silu.inputs = vec!["left.tail".into()];
        product.inputs[1] = "right.tail".into();
        sum.inputs[1] = "right.tail".into();
        for op in [&mut silu, &mut product, &mut sum] {
            op.output_shape = vec![1, lanes as u64];
        }
        graph.operations = vec![left, right, silu, product, sum];
        graph.output = "sum".into();
    } else {
        graph.operations = vec![silu, product, sum, alias];
        graph.output = "alias".into();
    }
    graph
}

fn freeze<'a>(
    imports: &'a Builder,
    left: &View,
    right: &View,
    sources: &'a BTreeMap<String, SourceTensor>,
    norms: &'a SourceNorms,
    lanes: usize,
) -> FrozenRegion<'a> {
    // Drop semantic/config owners BEFORE execution custody/quote. No copied
    // source tree or hidden still-live graph baseline is credited as free.
    let owner = left.words.at(0).0;
    let padded = imports.nodes[owner].lanes != lanes;
    let graph = graph(lanes, padded);
    let validated = validate_source_records(
        &graph,
        &BTreeMap::new(),
        sources,
        WeightMode::Public,
        ExecutionPolicy::default(),
        norms,
    )
    .unwrap();
    let mut compact = compact_lowering::CompactGraph::new(imports, validated, &graph, 0).unwrap();
    if padded {
        assert!(matches!(
            compact.bind_import("left", left),
            Err(Error::Binding)
        ));
        assert!(matches!(
            compact.bind_import("right", right),
            Err(Error::Binding)
        ));
    }
    for (name, selected) in [("left", left), ("right", right)] {
        let owner = selected.words.at(0).0;
        let n = imports.nodes[owner].lanes;
        assert_eq!(n % lanes, 0);
        let full = View {
            words: (0..n).map(|lane| (owner, lane)).collect(),
            shape: vec![1, n / lanes, lanes],
        };
        compact.bind_import(name, &full).unwrap();
    }
    let mut output = None;
    for _ in 0..graph.operations.len() {
        output = Some(compact.lower_next().unwrap());
    }
    compact
        .freeze(vec![output.unwrap()], MAX_PROGRAM_BYTES)
        .unwrap()
}

fn original(region: &FrozenRegion<'_>, lanes: usize, offset: usize) -> Arc<Program> {
    let (mut builder, left, right) = input_builder(lanes, offset);
    let imports = builder.nodes.len();
    let gathered = builder.gather(&left.words).unwrap();
    // Literal original lowering, with the SAME compiler-owned native Arcs.
    // Normal Builder::push/finish retain numeric/work/retirement admission.
    // No Program is fabricated from a digest, interval or recipe reconstruction.
    let descriptor = region
        .descriptor_census()
        .position(|entry| matches!(entry.kind, compact_lowering::DescriptorKind::GuardedSilu))
        .unwrap();
    let prepared = region.prepared(descriptor).unwrap();
    let full = prepared.program_at(0).unwrap().clone();
    let remainder = lanes % full.lanes();
    let tail = if remainder == 0 {
        None
    } else {
        Some(prepared.program_at(lanes - remainder).unwrap().clone())
    };
    let silu = builder
        .push(
            Code::TensorLoop {
                input: LoopInput::Slice(gathered),
                program: full,
                tail,
            },
            lanes,
            prepared.output_bounds().unwrap(),
        )
        .unwrap();
    let silu = builder.view(silu, vec![1, 1, lanes]);
    let product = View {
        words: builder.binary(&silu.words, &right.words, true).unwrap(),
        shape: vec![1, 1, lanes],
    };
    let sum = builder.residual(&product, &right).unwrap();
    Arc::new(
        builder
            .finish(
                imports,
                sum.iter().collect(),
                &mut AdmissionSite::default(),
                |h| {
                    h.update(b"frozen-execution-original-custody");
                    Ok(())
                },
                || source_binding(&BTreeMap::new()),
            )
            .unwrap(),
    )
}

fn fresh(program: Arc<Program>, lanes: usize, offset: usize) -> [Worker; 3] {
    let session = [73; 32];
    let context = Context::new(session, program.digest(), 0).unwrap();
    let runtimes = puma::prss::insecure_test_support::fresh_compound_parties(
        context,
        program.work_budget().limits().unwrap(),
    )
    .unwrap();
    let mut workers =
        runtimes.map(|runtime| Worker::new(program.clone(), runtime, session).unwrap());
    for key in ["left", "right"] {
        let padding = offset * lanes;
        let original_index = |i: usize| i as i64 - (padding - offset) as i64;
        let values: Vec<_> = (0..lanes + padding)
            .map(|i| {
                if key == "left" {
                    (original_index(i).rem_euclid(3) - 1) * ONE / 8
                } else {
                    ONE / 16
                }
            })
            .collect();
        let a: Vec<_> = (0..values.len())
            .map(|i| 0x45cafe_u64.wrapping_add((original_index(i) as u64).wrapping_mul(11)))
            .collect();
        let b: Vec<_> = (0..values.len())
            .map(|i| 0x89babe_u64.wrapping_add((original_index(i) as u64).wrapping_mul(19)))
            .collect();
        // Preserve every selected clear lane and both old fixture components;
        // only unconsumed leading rows make the declaration shape genuine.
        for lane in 0..lanes {
            let old = (offset + lane) as u64;
            assert_eq!(a[padding + lane], 0x45cafe + old * 11);
            assert_eq!(b[padding + lane], 0x89babe + old * 19);
            assert_eq!(
                values[padding + lane],
                if key == "left" {
                    ((offset + lane) as i64 % 3 - 1) * ONE / 8
                } else {
                    ONE / 16
                }
            );
        }
        let c: Vec<_> = values
            .iter()
            .enumerate()
            .map(|(i, x)| (*x as u64).wrapping_sub(a[i]).wrapping_sub(b[i]))
            .collect();
        let components = [a, b, c];
        for (i, worker) in workers.iter_mut().enumerate() {
            worker
                .import(key, &components[i], &components[(i + 1) % 3])
                .unwrap();
        }
    }
    workers
}

type Routed = (Party, Party, PeerBytes);
fn compare_frames(
    from: Party,
    a: Vec<(Party, PeerBytes)>,
    b: Vec<(Party, PeerBytes)>,
    qa: &mut Vec<Routed>,
    qb: &mut Vec<Routed>,
    bytes: &mut usize,
) {
    assert_eq!(a.len(), b.len());
    for ((to, a), (other, b)) in a.into_iter().zip(b) {
        assert_eq!(to, other);
        assert_eq!(a.as_bytes(), b.as_bytes());
        *bytes += a.as_bytes().len();
        for queue in [&mut *qa, &mut *qb] {
            if queue.capacity() == 0 {
                queue.try_reserve_exact(QUEUE_FRAMES).unwrap();
                assert_eq!(queue.capacity(), QUEUE_FRAMES);
            }
            assert!(queue.len() < QUEUE_FRAMES, "prepriced test routing bound");
        }
        qa.push((from, to, a));
        qb.push((from, other, b));
    }
}
pub(super) fn drive(
    control: &mut [Worker; 3],
    candidate: &mut [Worker; 3],
    custody: &Qualified<'_>,
) -> usize {
    drive_with_stop(control, candidate, custody, false)
}

pub(super) fn drive_with_stop(
    control: &mut [Worker; 3],
    candidate: &mut [Worker; 3],
    custody: &Qualified<'_>,
    partial: bool,
) -> usize {
    let (mut qa, mut qb) = (Vec::new(), Vec::new());
    let mut bytes = 0;
    let mut rounds = 0;
    while !control.iter().all(Worker::complete) {
        rounds += 1;
        assert!(
            rounds < 100_000,
            "test driver stalled; not a performance claim"
        );
        if qa.is_empty() {
            for i in 0..3 {
                if !control[i].complete() {
                    let a = control[i].advance().unwrap();
                    let b = custody.advance(&mut candidate[i], None).unwrap();
                    compare_frames(control[i].party(), a, b, &mut qa, &mut qb, &mut bytes);
                    custody.assert_same_state(&control[i], &candidate[i]);
                    if partial && candidate[i].loop_test_state().0 > 0 {
                        return bytes;
                    }
                }
            }
        } else {
            assert_eq!(qa.len(), qb.len());
            for ((from, to, a), (other_from, other_to, b)) in std::mem::take(&mut qa)
                .into_iter()
                .zip(std::mem::take(&mut qb))
            {
                assert_eq!((from, to), (other_from, other_to));
                let a = control[to as usize].receive(a.as_bytes(), from).unwrap();
                let b = candidate[to as usize].receive(b.as_bytes(), from).unwrap();
                compare_frames(to, a, b, &mut qa, &mut qb, &mut bytes);
                custody.assert_same_state(&control[to as usize], &candidate[to as usize]);
            }
        }
    }
    assert!(qa.is_empty() && qb.is_empty());
    assert!(candidate.iter().all(Worker::complete));
    for i in 0..3 {
        custody.assert_same_state(&control[i], &candidate[i]);
    }
    bytes
}

#[test]
fn cancellation_after_a_saved_full_tile_erases_partial_owner_and_queued_tail_state() {
    let lanes = 129;
    let (imports, left, right) = input_builder(lanes, 1);
    let sources = BTreeMap::new();
    let norms = SourceNorms::new();
    let frozen = freeze(&imports, &left, &right, &sources, &norms, lanes);
    drop((left, right));
    let program = original(&frozen, lanes, 1);
    with_oracle(
        fresh(program.clone(), lanes, 1),
        fresh(program, lanes, 1),
        &frozen,
        MAX_PROGRAM_BYTES,
        |mut control, mut candidate, custody| {
            assert!(drive_with_stop(&mut control, &mut candidate, custody, true) > 0);
            assert!(candidate.iter().any(|w| w.loop_test_state().0 == 128));
            for i in 0..3 {
                custody.assert_same_state(&control[i], &candidate[i]);
                control[i].cancel();
                candidate[i].cancel();
                custody.assert_same_state(&control[i], &candidate[i]);
                assert!(matches!(
                    custody.advance(&mut candidate[i], None),
                    Err(Error::Retired)
                ));
            }
            Ok(())
        },
    )
    .unwrap();
}

#[test]
fn qualified_frozen_runs_same_worker_full_wire_components_cursors_offsets_aliases_and_tail() {
    for (lanes, offset) in [(1, 0), (128, 0), (129, 3), (257, 1)] {
        let (imports, left, right) = input_builder(lanes, offset);
        let sources = BTreeMap::new();
        let norms = SourceNorms::new();
        let frozen = freeze(&imports, &left, &right, &sources, &norms, lanes);
        drop((left, right));
        let program = original(&frozen, lanes, offset);
        let control = fresh(program.clone(), lanes, offset);
        let candidate = fresh(program, lanes, offset);
        with_oracle(
            control,
            candidate,
            &frozen,
            MAX_PROGRAM_BYTES,
            |mut control, mut candidate, custody| {
                if lanes > 128 {
                    assert!(custody.has_packed_offsets());
                }
                assert!(matches!(
                    custody.address(usize::MAX, 0),
                    Err(Error::Binding)
                ));
                assert!(matches!(custody.address(0, usize::MAX), Err(Error::Shape)));
                assert!(drive(&mut control, &mut candidate, custody) > 0);
                // Public finish_frame uses original completion/source/identity owner,
                // but resolves every frozen export through its physical offset.
                for i in 0..3 {
                    custody.assert_same_state(&control[i], &candidate[i]);
                }
                for (a, b) in control.into_iter().zip(candidate) {
                    assert_eq!(a.finish_frame()?.as_slice(), b.finish_frame()?.as_slice());
                }
                Ok(())
            },
        )
        .unwrap();
    }
}

#[test]
fn equal_numeric_templates_from_foreign_frozen_owners_do_not_qualify_and_unwind_drops_custody() {
    let lanes = 1;
    let (imports, left, right) = input_builder(lanes, 0);
    let sources = BTreeMap::new();
    let norms = SourceNorms::new();
    let a = freeze(&imports, &left, &right, &sources, &norms, lanes);
    let b = freeze(&imports, &left, &right, &sources, &norms, lanes);
    drop((left, right));
    assert!(!Arc::ptr_eq(
        a.prepared(0).unwrap().program_at(0).unwrap(),
        b.prepared(0).unwrap().program_at(0).unwrap()
    ));
    let program = original(&a, lanes, 0);
    assert!(matches!(
        with_oracle(
            fresh(program.clone(), lanes, 0),
            fresh(program.clone(), lanes, 0),
            &b,
            MAX_PROGRAM_BYTES,
            |_, _, _| Ok(())
        ),
        Err(Error::Binding)
    ));
    let mut weak = None;
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _: Result<(), Error> = with_oracle(
            fresh(program.clone(), lanes, 0),
            fresh(program, lanes, 0),
            &a,
            MAX_PROGRAM_BYTES,
            |_, _, custody| {
                weak = Some(custody.weak_layout());
                panic!("test-only custody unwind");
            },
        );
    }));
    assert!(panic.is_err());
    assert!(weak.unwrap().upgrade().is_none());
}

#[test]
fn physical_layout_rejects_changed_retirement_before_any_execution() {
    let lanes = 1;
    let (imports, left, right) = input_builder(lanes, 0);
    let sources = BTreeMap::new();
    let norms = SourceNorms::new();
    let region = freeze(&imports, &left, &right, &sources, &norms, lanes);
    drop((left, right));
    let program = original(&region, lanes, 0);
    let loan = region.borrow_schedule(MAX_PROGRAM_BYTES).unwrap();
    let reader = PrivateScheduleRef::frozen(&loan);
    let mut retirement = program.retire.clone();
    retirement[0].push(0);
    assert!(matches!(
        physical::Layout::build(
            reader,
            &retirement,
            program.output_coordinates(),
            region.resources().retained_bytes,
            MAX_PROGRAM_BYTES
        ),
        Err(Error::Binding)
    ));
}

#[test]
fn whole_oracle_transition_exact_cap_and_one_byte_short_precede_new_storage() {
    let lanes = 1;
    let (imports, left, right) = input_builder(lanes, 0);
    let sources = BTreeMap::new();
    let norms = SourceNorms::new();
    let frozen = freeze(&imports, &left, &right, &sources, &norms, lanes);
    drop((left, right));
    let program = original(&frozen, lanes, 0);
    let a = fresh(program.clone(), lanes, 0);
    let b = fresh(program.clone(), lanes, 0);
    let exact =
        crate::replicated_decoder::runtime::frozen_execution_tests::quote(&a, &b, &frozen).unwrap();
    assert!(matches!(
        with_oracle(a, b, &frozen, exact - 1, |_, _, _| Ok(())),
        Err(Error::Bounds)
    ));
    with_oracle(
        fresh(program.clone(), lanes, 0),
        fresh(program, lanes, 0),
        &frozen,
        exact,
        |_, _, _| Ok(()),
    )
    .unwrap();
}

#[test]
fn generated_iota_never_enters_execution_as_a_fabricated_private_or_public_source() {
    let lanes = 3;
    let (imports, left, right) = input_builder(lanes, 0);
    let sources = BTreeMap::new();
    let norms = SourceNorms::new();
    let body = freeze(&imports, &left, &right, &sources, &norms, lanes);
    let program = original(&body, lanes, 0);
    let frozen = {
        let mut graph = graph(lanes, false);
        let mut op = graph.operations[0].clone();
        op.operator = Op::GreedyTokenSelection;
        op.id = "selected".into();
        op.inputs = vec!["left".into()];
        op.output_shape = vec![1];
        graph.operations = vec![op];
        graph.output = "selected".into();
        let validated = validate_source_records(
            &graph,
            &BTreeMap::new(),
            &sources,
            WeightMode::Public,
            ExecutionPolicy::default(),
            &norms,
        )
        .unwrap();
        let mut compact =
            compact_lowering::CompactGraph::new(&imports, validated, &graph, 0).unwrap();
        compact.bind_import("left", &left).unwrap();
        let output = compact.lower_next().unwrap();
        compact.freeze(vec![output], MAX_PROGRAM_BYTES).unwrap()
    };
    drop((left, right));
    let result: Result<(), Error> = with_oracle(
        fresh(program.clone(), lanes, 0),
        fresh(program, lanes, 0),
        &frozen,
        MAX_PROGRAM_BYTES,
        |_, _, _| panic!("unqualified iota execution"),
    );
    assert!(matches!(result, Err(Error::Binding)));
}

#[test]
fn frozen_custody_cap_errors_cancellation_and_malformed_frames_burn_original_owners() {
    let lanes = 129;
    let (imports, left, right) = input_builder(lanes, 2);
    let sources = BTreeMap::new();
    let norms = SourceNorms::new();
    let frozen = freeze(&imports, &left, &right, &sources, &norms, lanes);
    drop((left, right));
    let program = original(&frozen, lanes, 2);
    assert!(matches!(
        with_oracle(
            fresh(program.clone(), lanes, 2),
            fresh(program.clone(), lanes, 2),
            &frozen,
            0,
            |_, _, _| Ok(())
        ),
        Err(Error::Bounds)
    ));
    with_oracle(
        fresh(program.clone(), lanes, 2),
        fresh(program.clone(), lanes, 2),
        &frozen,
        MAX_PROGRAM_BYTES,
        |mut control, mut candidate, custody| {
            let a = control[0].advance().unwrap();
            let b = custody.advance(&mut candidate[0], None).unwrap();
            assert_eq!(a.len(), b.len());
            for (a, b) in a.iter().zip(&b) {
                assert_eq!(a.0, b.0);
                assert_eq!(a.1.as_bytes(), b.1.as_bytes());
            }
            custody.assert_same_state(&control[0], &candidate[0]);
            let a = control[0].receive(&[0; 3], Party::P1).err().unwrap();
            let b = candidate[0].receive(&[0; 3], Party::P1).err().unwrap();
            assert_eq!(format!("{a:?}"), format!("{b:?}"));
            custody.assert_same_state(&control[0], &candidate[0]);
            for i in 1..3 {
                control[i].cancel();
                candidate[i].cancel();
                custody.assert_same_state(&control[i], &candidate[i]);
            }
            assert!(matches!(
                custody.advance(&mut candidate[0], None),
                Err(Error::Retired)
            ));
            Ok(())
        },
    )
    .unwrap();
    // No implicit fallback from frozen custody into the default expanded route.
    with_oracle(
        fresh(program.clone(), lanes, 2),
        fresh(program, lanes, 2),
        &frozen,
        MAX_PROGRAM_BYTES,
        |_, mut candidate, _| {
            assert!(matches!(candidate[0].advance(), Err(Error::Binding)));
            assert!(candidate[0].retired());
            Ok(())
        },
    )
    .unwrap();
}
