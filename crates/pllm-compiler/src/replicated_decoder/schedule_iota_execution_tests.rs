//! TEST-ONLY genuine expanded custody and independent original compositions.
//! Public test keys only. Original Node/Words/iota Vec owners are setup, retained
//! and charged by the existing oracle; the candidate creates none of them.
use super::super::*;
use super::*;
use crate::replicated_decoder::program::compact_lowering::{
    frozen_region::FrozenRegion, ArgmaxCode,
};
use crate::replicated_decoder::runtime::{frozen_execution_tests as oracle, Worker};
use std::collections::BTreeMap;

fn imports(lanes: usize) -> (Builder, View) {
    let mut b = Builder {
        public_weights: true,
        ..Builder::default()
    };
    let owner = b
        .push(
            Code::Input("logits".into()),
            4 * lanes,
            Interval::new(-32, 32).unwrap(),
        )
        .unwrap();
    let view = View {
        words: (3 * lanes..4 * lanes).map(|i| (owner, i)).collect(),
        shape: vec![1, 1, lanes],
    };
    (b, view)
}

fn freeze<'a>(
    b: &'a Builder,
    input: &View,
    sources: &'a BTreeMap<String, SourceTensor>,
    norms: &'a SourceNorms,
    lanes: usize,
) -> FrozenRegion<'a> {
    let config = serde_json::to_vec(&serde_json::json!({"model_type":"llama","hidden_size":2,
        "intermediate_size":4,"num_hidden_layers":1,"num_attention_heads":1,
        "num_key_value_heads":1,"head_dim":2,"vocab_size":3,"tie_word_embeddings":false,
        "hidden_act":"silu","attention_bias":false,"max_position_embeddings":64,
        "rope_theta":10000,"rms_norm_eps":0.000001,"attention_dropout":0.0,"use_cache":true}))
    .unwrap();
    let mut graph = lower_with_weights(&config, 1, 1, WeightMode::Public)
        .unwrap()
        .prefill;
    let mut op = graph
        .operations
        .iter()
        .find(|op| op.operator == Op::GreedyTokenSelection)
        .unwrap()
        .clone();
    op.id = "selected".into();
    op.inputs = vec!["logits.tail".into()];
    op.output_shape = vec![1];
    let mut suffix = op.clone();
    suffix.id = "logits.tail".into();
    suffix.operator = Op::LastToken;
    suffix.inputs = vec!["logits".into()];
    suffix.output_shape = vec![1, lanes as u64];
    suffix.attributes = serde_json::json!({"axis":1,"selection":"last_valid"});
    graph.operations = vec![suffix, op];
    graph.output = "selected".into();
    graph.state_inputs.clear();
    graph.state_outputs.clear();
    let validated = validate_source_records(
        &graph,
        &BTreeMap::new(),
        sources,
        WeightMode::Public,
        ExecutionPolicy::default(),
        norms,
    )
    .unwrap();
    let mut compact = compact_lowering::CompactGraph::new(b, validated, &graph, 0).unwrap();
    assert!(matches!(
        compact.bind_import("logits", input),
        Err(Error::Binding)
    ));
    let owner = input.words.at(0).0;
    let full = View {
        words: (0..b.nodes[owner].lanes)
            .map(|lane| (owner, lane))
            .collect(),
        shape: vec![1, 4, lanes],
    };
    compact.bind_import("logits", &full).unwrap();
    compact.lower_next().unwrap(); // genuine zero-storage suffix publication
    let out = compact.lower_next().unwrap();
    assert_eq!(input.words.len(), lanes);
    compact.freeze(vec![out], MAX_PROGRAM_BYTES).unwrap()
}

fn old_select(b: &mut Builder, bit: usize, yes: usize, no: usize) -> usize {
    // Literal original select; not shared select()/ArgmaxSink traversal.
    let a = b.nodes[yes].bounds;
    let z = b.nodes[no].bounds;
    let range = Interval::new(a.lower().min(z.lower()), a.upper().max(z.upper())).unwrap();
    let difference = b.affine(yes, Some(no), affine(1, -1, 0)).unwrap();
    let product = b.product(bit, difference, None, false).unwrap();
    b.push(
        Code::Affine {
            left: no,
            right: Some(product),
            coefficients: affine(1, 1, 0),
        },
        b.nodes[no].lanes,
        range,
    )
    .unwrap()
}

fn original(region: &FrozenRegion<'_>, lanes: usize) -> Result<Arc<Program>, Error> {
    let (mut b, input) = imports(lanes);
    let count = b.nodes.len();
    // The expanded control's real legacy owner, NOT new candidate metadata.
    let iota = b
        .push(
            Code::PublicValues((0..lanes as i64).collect()),
            lanes,
            Interval::new(0, lanes as i64 - 1).unwrap(),
        )
        .unwrap();
    let mut values = input.words.iter().collect::<Vec<_>>();
    let mut ids = (0..lanes).map(|i| (iota, i)).collect::<Vec<_>>();
    while values.len() > 1 {
        let mut next_values = Vec::new();
        let mut next_ids = Vec::new();
        for start in (0..values.len()).step_by(256) {
            let len = (values.len() - start).min(256);
            let pairs = len / 2;
            if pairs != 0 {
                let gather = |b: &mut Builder, words: &[(usize, usize)], parity: usize| {
                    b.gather(&(0..pairs).map(|i| words[start + 2 * i + parity]).collect())
                        .unwrap()
                };
                let left = gather(&mut b, &values, 0);
                let right = gather(&mut b, &values, 1);
                let difference = b.affine(left, Some(right), affine(1, -1, 0)).unwrap();
                let ordinal = b.nodes.len();
                let mut kernel = None;
                // Only original kernel custody comes from the sealed owner. All
                // pairing, carries and selects above/below are independent.
                region
                    .visit_argmax(1, &mut |event| {
                        if event.owner == ordinal {
                            if let ArgmaxCode::Tensor { program, .. } = event.code {
                                kernel = Some(program.clone());
                            }
                        }
                        Ok(())
                    })
                    .unwrap();
                let kernel = kernel.unwrap();
                assert_eq!(kernel.lanes(), pairs);
                let range = kernel.output_bounds();
                let less = b
                    .push(
                        Code::Tensor {
                            input: difference,
                            program: kernel,
                        },
                        pairs,
                        range,
                    )
                    .unwrap();
                let best = old_select(&mut b, less, right, left);
                let left_id = gather(&mut b, &ids, 0);
                let right_id = gather(&mut b, &ids, 1);
                let index = old_select(&mut b, less, right_id, left_id);
                next_values.extend((0..pairs).map(|i| (best, i)));
                next_ids.extend((0..pairs).map(|i| (index, i)));
            }
            if len % 2 != 0 {
                next_values.push(values[start + len - 1]);
                next_ids.push(ids[start + len - 1]);
            }
        }
        values = next_values;
        ids = next_ids;
    }
    b.finish(
        count,
        ids,
        &mut AdmissionSite::default(),
        |h| {
            h.update(b"original-iota-custody");
            Ok(())
        },
        || source_binding(&BTreeMap::new()),
    )
    .map(Arc::new)
}

fn fresh(program: Arc<Program>, logits: &[i64]) -> [Worker; 3] {
    fresh_with_limits(
        program.clone(),
        logits,
        program.work_budget().limits().unwrap(),
    )
}
fn fresh_with_limits(
    program: Arc<Program>,
    logits: &[i64],
    limits: puma::prss::PrssLimits,
) -> [Worker; 3] {
    let session = [93; 32];
    let context = puma::Context::new(session, program.digest(), 0).unwrap();
    let runtimes =
        puma::prss::insecure_test_support::fresh_compound_parties(context, limits).unwrap();
    let mut workers = runtimes.map(|r| Worker::new(program.clone(), r, session).unwrap());
    let padding = 3 * logits.len();
    let values = [vec![0; padding], logits.to_vec()].concat();
    let original_index = |i: usize| (i as u64).wrapping_sub((padding - 3) as u64);
    let first: Vec<_> = (0..values.len())
        .map(|i| 0x5123_u64.wrapping_add(original_index(i).wrapping_mul(17)))
        .collect();
    let second: Vec<_> = (0..values.len())
        .map(|i| 0x9876_u64.wrapping_add(original_index(i).wrapping_mul(29)))
        .collect();
    for (lane, &logit) in logits.iter().enumerate() {
        assert_eq!(values[padding + lane], logit);
        assert_eq!(first[padding + lane], 0x5123 + (3 + lane) as u64 * 17);
        assert_eq!(second[padding + lane], 0x9876 + (3 + lane) as u64 * 29);
    }
    let third: Vec<_> = values
        .iter()
        .enumerate()
        .map(|(i, &v)| (v as u64).wrapping_sub(first[i]).wrapping_sub(second[i]))
        .collect();
    let shares = [first, second, third];
    for (i, worker) in workers.iter_mut().enumerate() {
        worker
            .import("logits", &shares[i], &shares[(i + 1) % 3])
            .unwrap();
    }
    workers
}

#[test]
fn real_frozen_argmax_lazy_iota_matches_original_wire_both_shares_cursors_tails_and_ties() {
    for lanes in [1, 2, 3, 127, 129, 258] {
        let (b, input) = imports(lanes);
        let sources = BTreeMap::new();
        let norms = SourceNorms::new();
        let region = freeze(&b, &input, &sources, &norms, lanes);
        if lanes == 1 {
            // The region has a real zero-work singleton alias/iota. This is not
            // an admissible standalone original Program: do not add fake work or
            // relax the old complete-program nonzero-work gate to execute it.
            assert_eq!(region.work_budget(), WorkBudget::default());
            assert!(matches!(original(&region, lanes), Err(Error::Bounds)));
            continue;
        }
        let program = original(&region, lanes).unwrap();
        for tie in [false, true] {
            let mut logits: Vec<_> = (0..lanes).map(|i| -31 + (i % 5) as i64).collect();
            logits[lanes - 1] = -1;
            if tie {
                logits[0] = -1;
            }
            let expected = if tie { 0 } else { lanes - 1 };
            oracle::with_oracle(
                fresh(program.clone(), &logits),
                fresh(program.clone(), &logits),
                &region,
                MAX_PROGRAM_BYTES,
                |mut old, mut new, custody| {
                    assert!(custody.address(1, 0).is_err(), "iota has no secret address");
                    for (a, b) in old.iter().zip(&new) {
                        custody.assert_same_state(a, b);
                    }
                    let bytes = super::execution_tests::drive(&mut old, &mut new, custody);
                    assert!(bytes > 0);
                    let mut sum = 0_u64;
                    for (a, b) in old.into_iter().zip(new) {
                        let old = a.finish_frame()?;
                        let new = custody.finish_frame(b)?;
                        assert_eq!(old.as_slice(), new.as_slice());
                        sum = sum.wrapping_add(u64::from_le_bytes(new[73..81].try_into().unwrap()));
                    }
                    assert_eq!(sum, expected as u64);
                    Ok(())
                },
            )
            .unwrap();
        }
    }
}

#[test]
fn sealed_iota_checks_geometry_index_and_exact_owner_without_private_slot_or_array() {
    let (b, input) = imports(3);
    let sources = BTreeMap::new();
    let norms = SourceNorms::new();
    let a = freeze(&b, &input, &sources, &norms, 3);
    let foreign = freeze(&b, &input, &sources, &norms, 3);
    let generated =
        super::super::schedule_finalization::GeneratedPublicValues::argmax_indices(&a, 1).unwrap();
    let bounds = Interval::new(0, 2).unwrap();
    assert_eq!(generated.checked_len(&a, 3, bounds).unwrap(), 3);
    assert_eq!(generated.checked_element(&a, 3, bounds, 2).unwrap(), 2);
    assert!(matches!(
        generated.checked_element(&foreign, 3, bounds, 0),
        Err(Error::Binding)
    ));
    for index in [3, usize::MAX] {
        assert!(matches!(
            generated.checked_element(&a, 3, bounds, index),
            Err(Error::Shape)
        ));
    }
    for (n, range) in [
        (0, bounds),
        (2, bounds),
        (usize::MAX, bounds),
        (3, Interval::new(-1, 2).unwrap()),
    ] {
        assert!(matches!(
            generated.checked_len(&a, n, range),
            Err(Error::Shape)
        ));
    }
    let loan = a.borrow_schedule(MAX_PROGRAM_BYTES).unwrap();
    let unbound = PrivateScheduleRef::frozen(&loan);
    assert!(matches!(
        unbound.generated_element(1, 0),
        Err(Error::Binding)
    ));
    assert!(matches!(physical::quote(unbound), Err(Error::Binding)));
    let reader = PrivateScheduleRef::frozen_owned(&loan, &a);
    let legacy = original(&a, 3).unwrap();
    let layout = physical::Layout::build(
        reader,
        &legacy.retire,
        legacy.output_coordinates(),
        0,
        MAX_PROGRAM_BYTES,
    )
    .unwrap();
    assert_eq!(
        layout.span(1).unwrap().location,
        physical::Location::GeneratedPublic
    );
    assert!(matches!(layout.address(1, 0), Err(Error::Binding)));
    assert!(matches!(layout.address(1, usize::MAX), Err(Error::Shape)));
    let frozen = physical::quote(reader).unwrap();
    let expanded = physical::quote(legacy.schedule_access()).unwrap();
    assert_eq!(frozen.owners + 1, expanded.owners);
    assert_eq!(frozen.words + 3, expanded.words);
    // A generated owner has no secret allocation, but its logical last-use
    // still must agree with the original compiler's real retirement record.
    let mut wrong_retirement = legacy.retire.clone();
    let mut removed = 0;
    for bucket in &mut wrong_retirement {
        let before = bucket.len();
        bucket.retain(|id| *id != 1);
        removed += before - bucket.len();
    }
    assert_eq!(removed, 1);
    assert!(matches!(
        physical::Layout::build(
            reader,
            &wrong_retirement,
            legacy.output_coordinates(),
            0,
            MAX_PROGRAM_BYTES
        ),
        Err(Error::Binding)
    ));
    assert!(matches!(
        control_preflight(usize::MAX, usize::MAX),
        Err(Error::Bounds)
    ));
}

#[test]
fn argmax_foreign_native_owner_cap_and_corrupt_work_refuse_before_transition() {
    let (b, input) = imports(3);
    let sources = BTreeMap::new();
    let norms = SourceNorms::new();
    let a = freeze(&b, &input, &sources, &norms, 3);
    let foreign = freeze(&b, &input, &sources, &norms, 3);
    let mut program = original(&a, 3).unwrap();
    let logits = [-3, -1, -1];
    let old = fresh(program.clone(), &logits);
    let new = fresh(program.clone(), &logits);
    let exact = oracle::quote(&old, &new, &a).unwrap();
    assert!(matches!(
        oracle::with_oracle(old, new, &a, exact - 1, |_, _, _| -> Result<(), Error> {
            panic!("short cap entered transition")
        }),
        Err(Error::Bounds)
    ));
    oracle::with_oracle(
        fresh(program.clone(), &logits),
        fresh(program.clone(), &logits),
        &a,
        exact,
        |_, _, _| Ok(()),
    )
    .unwrap();
    assert!(matches!(
        oracle::with_oracle(
            fresh(program.clone(), &logits),
            fresh(program.clone(), &logits),
            &foreign,
            MAX_PROGRAM_BYTES,
            |_, _, _| -> Result<(), Error> { panic!("foreign kernels entered transition") }
        ),
        Err(Error::Binding)
    ));
    let limits = program.work_budget().limits().unwrap();
    // Negative metadata corruption, never a positive source/work authority.
    Arc::get_mut(&mut program).unwrap().work = WorkBudget::local()
        .checked_mul(puma::MAX_PROGRAM_OPERATIONS)
        .unwrap()
        .checked_add(WorkBudget::local())
        .unwrap();
    assert!(matches!(
        oracle::with_oracle(
            fresh_with_limits(program.clone(), &logits, limits),
            fresh_with_limits(program, &logits, limits),
            &a,
            MAX_PROGRAM_BYTES,
            |_, _, _| -> Result<(), Error> { panic!("invalid work entered transition") }
        ),
        Err(Error::Bounds)
    ));
}

#[test]
fn foreign_generated_recipe_and_malformed_metadata_fail_before_physical_construction() {
    struct One<'a>(super::super::schedule_finalization::BorrowedNode<'a>);
    impl BorrowedOrderedNodes for One<'_> {
        fn node_count(&self) -> usize {
            1
        }
        fn node(
            &self,
            ordinal: usize,
        ) -> Result<super::super::schedule_finalization::BorrowedNode<'_>, Error> {
            if ordinal != 0 {
                return Err(Error::Binding);
            }
            Ok(self.0)
        }
    }
    let (b, input) = imports(3);
    let sources = BTreeMap::new();
    let norms = SourceNorms::new();
    let a = freeze(&b, &input, &sources, &norms, 3);
    let foreign = freeze(&b, &input, &sources, &norms, 3);
    let generated =
        super::super::schedule_finalization::GeneratedPublicValues::argmax_indices(&foreign, 1)
            .unwrap();
    let fake = One(super::super::schedule_finalization::BorrowedNode {
        lanes: 3,
        bounds: Interval::new(0, 2).unwrap(),
        recipe: BorrowedRecipe::GeneratedPublicValues(generated),
    });
    let reader = PrivateScheduleRef(Storage::OwnedFrozen(&fake, &a));
    assert!(matches!(reader.generated_len(0), Err(Error::Binding)));
    assert!(matches!(physical::quote(reader), Err(Error::Binding)));
    let same =
        super::super::schedule_finalization::GeneratedPublicValues::argmax_indices(&a, 1).unwrap();
    let malformed = One(super::super::schedule_finalization::BorrowedNode {
        lanes: 2,
        bounds: Interval::new(0, 1).unwrap(),
        recipe: BorrowedRecipe::GeneratedPublicValues(same),
    });
    let reader = PrivateScheduleRef(Storage::OwnedFrozen(&malformed, &a));
    assert!(matches!(reader.generated_len(0), Err(Error::Shape)));
    assert!(matches!(physical::quote(reader), Err(Error::Shape)));
}

#[test]
fn argmax_malformed_wire_burns_same_original_worker_and_lazy_storage() {
    let (b, input) = imports(3);
    let sources = BTreeMap::new();
    let norms = SourceNorms::new();
    let a = freeze(&b, &input, &sources, &norms, 3);
    let program = original(&a, 3).unwrap();
    let logits = [-3, -1, -1];
    oracle::with_oracle(
        fresh(program.clone(), &logits),
        fresh(program, &logits),
        &a,
        MAX_PROGRAM_BYTES,
        |mut old, mut new, custody| {
            let left = old[0].advance()?;
            let right = custody.advance(&mut new[0], None)?;
            assert_eq!(left.len(), right.len());
            assert!(!left.is_empty());
            for (a, b) in left.iter().zip(&right) {
                assert_eq!(a.0, b.0);
                assert_eq!(a.1.as_bytes(), b.1.as_bytes());
            }
            let a = old[0].receive(&[0; 3], puma::Party::P1).err().unwrap();
            let b = new[0].receive(&[0; 3], puma::Party::P1).err().unwrap();
            assert_eq!(format!("{a:?}"), format!("{b:?}"));
            custody.assert_same_state(&old[0], &new[0]);
            assert!(old[0].retired() && new[0].retired());
            Ok(())
        },
    )
    .unwrap();
}
