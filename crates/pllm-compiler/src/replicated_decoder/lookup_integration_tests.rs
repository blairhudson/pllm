use super::*;
use crate::replicated_decoder::runtime::lookup::{Plan, ROW_TILE};
use crate::replicated_decoder::tests::{execute, workers};

fn interval(values: &[i64]) -> Interval {
    Interval::new(*values.iter().min().unwrap(), *values.iter().max().unwrap()).unwrap()
}

fn fixture(
    rows: usize,
    vocabulary: usize,
    width: usize,
) -> (Arc<Program>, BTreeMap<String, Vec<i64>>, Vec<u64>) {
    let weights: Vec<i64> = (0..vocabulary * width)
        .map(|i| ((i * 7919 + 17) % 100003) as i64 - 50001)
        .collect();
    let digest = Sha256::digest(
        weights
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .into();
    let mut builder = Builder {
        public_weights: true,
        lookup: policy::LookupPolicy::Tiled,
        ..Builder::default()
    };
    let weight = builder
        .push(
            Code::PublicInput {
                key: "weights".into(),
                digest,
                row_l1: None,
                row_squared_norm: None,
                row_l1_squared_sum: None,
            },
            vocabulary * width,
            interval(&weights),
        )
        .unwrap();
    let input = builder
        .push(
            Code::Input("tokens".into()),
            rows,
            Interval::new(0, vocabulary as i64 - 1).unwrap(),
        )
        .unwrap();
    // Nontrivial source-view order catches accidentally gathering a public prefix.
    let tokens = View {
        words: (0..rows).rev().map(|i| (input, i)).collect(),
        shape: vec![rows],
    };
    let table = SourceView {
        owner: weight,
        shape: vec![vocabulary, width],
    };
    let output = builder
        .tiled_lookup(&tokens, &table, &[rows, width])
        .unwrap();
    let expected_tiles = rows.div_ceil(ROW_TILE);
    let tiled: Vec<_> = builder
        .nodes
        .iter()
        .filter_map(|n| match &n.code {
            Code::PublicLookup {
                input,
                weight: source,
                plan,
            } => {
                assert_eq!(*source, weight);
                assert_eq!(n.bounds, interval(&weights));
                assert!(matches!(builder.nodes[*input].code, Code::Gather(_)));
                assert_eq!(n.lanes, plan.geometry().0 * width);
                Some(plan.geometry())
            }
            _ => None,
        })
        .collect();
    assert_eq!(tiled.len(), expected_tiles);
    assert_eq!(tiled.iter().map(|g| g.0).sum::<usize>(), rows);
    assert!(tiled
        .iter()
        .all(|&(r, v, w)| r <= ROW_TILE && v == vocabulary && w == width));
    assert!(!builder
        .nodes
        .iter()
        .any(|n| matches!(n.code, Code::TensorLoop { .. } | Code::PublicProduct { .. })));
    let work = schedule_work(&builder.nodes).unwrap();
    let single = Plan::new(
        1,
        vocabulary,
        width,
        Interval::new(0, vocabulary as i64 - 1).unwrap(),
    )
    .unwrap();
    assert_eq!(
        work,
        single
            .work_budget()
            .unwrap()
            .checked_mul(rows as u64)
            .unwrap()
    );
    let mut retire = vec![vec![]; builder.nodes.len()];
    for id in 0..builder.nodes.len() {
        if let Some(last) = (id + 1..builder.nodes.len())
            .rev()
            .find(|&step| builder.nodes[step].inputs().contains(&id))
        {
            retire[last].push(id);
        }
    }
    let ids: Vec<i64> = (0..rows)
        .map(|i| match i % 4 {
            0 => vocabulary as i64 - 1,
            1 => 0,
            _ => (i % vocabulary) as i64,
        })
        .collect();
    // Oracle does not invoke lookup, multiplication, or tensor production helpers.
    let expected = ids
        .iter()
        .rev()
        .flat_map(|&id| {
            weights[id as usize * width..(id as usize + 1) * width]
                .iter()
                .map(|&v| v as u64)
        })
        .collect();
    let program = Arc::new(Program {
        nodes: builder.nodes,
        inputs: 2,
        outputs: output.words.iter().collect(),
        retire,
        digest: [93; 32],
        payload_bytes: builder.allocated,
        work,
        source_binding: [0; 32],
        request_custody: None,
    });
    (
        program,
        BTreeMap::from([("weights".into(), weights), ("tokens".into(), ids)]),
        expected,
    )
}

#[test]
fn tiled_lookup_compiler_emission_retains_geometry_work_and_source_owner() {
    for (rows, vocabulary, width) in [(1, 1, 2), (8, 128, 3), (9, 131, 71), (17, 3, 2)] {
        fixture(rows, vocabulary, width);
    }
}

#[test]
fn tiled_lookup_builder_reuses_registered_full_and_tail_programs_before_push() {
    let mut builder = Builder {
        public_weights: true,
        ..Builder::default()
    };
    let weight = builder
        .push(
            Code::PublicInput {
                key: "weights".into(),
                digest: [7; 32],
                row_l1: None,
                row_squared_norm: None,
                row_l1_squared_sum: None,
            },
            131 * 2,
            Interval::new(-1, 1).unwrap(),
        )
        .unwrap();
    let token = builder
        .push(
            Code::Input("tokens".into()),
            9,
            Interval::new(0, 130).unwrap(),
        )
        .unwrap();
    let seed = Plan::new(8, 131, 2, Interval::new(0, 130).unwrap()).unwrap();
    // Populate the ordinary compiler registry through actual priced Tensor
    // nodes, not an artificial cache. Each prior invocation is still fresh.
    for program in seed.programs() {
        let repeated: Words = std::iter::repeat_n((token, 0), program.lanes()).collect();
        let input = builder.gather(&repeated).unwrap();
        builder.tensor(input, (**program).clone()).unwrap();
    }
    assert_eq!(builder.tensors.len(), 2);
    let tokens = builder.view(token, vec![9]);
    let table = SourceView {
        owner: weight,
        shape: vec![131, 2],
    };
    let output = builder.tiled_lookup(&tokens, &table, &[9, 2]).unwrap();
    assert_eq!(output.words.len(), 18);
    let mut count = 0;
    for node in &builder.nodes {
        if let Code::PublicLookup { plan, .. } = &node.code {
            count += 1;
            for program in plan.programs() {
                assert!(Arc::ptr_eq(
                    program,
                    builder.tensors.get(&program.digest()).unwrap()
                ));
            }
        }
    }
    assert_eq!(count, 2);
}

#[test]
fn tiled_lookup_generated_workers_match_independent_ordered_row_oracle() {
    for (rows, vocabulary, width) in [(1, 1, 2), (8, 128, 3), (9, 131, 71), (17, 3, 2)] {
        let (program, clear, expected) = fixture(rows, vocabulary, width);
        assert_eq!(execute(workers(program, &clear)), expected);
    }
}

#[test]
fn tiled_lookup_rejects_unbound_or_misshaped_public_table() {
    let mut builder = Builder {
        public_weights: true,
        ..Builder::default()
    };
    let input = builder
        .push(
            Code::Input("tokens".into()),
            1,
            Interval::new(0, 2).unwrap(),
        )
        .unwrap();
    let tokens = builder.view(input, vec![1]);
    let private = builder
        .push(
            Code::Input("private".into()),
            6,
            Interval::new(-1, 1).unwrap(),
        )
        .unwrap();
    let table = SourceView {
        owner: private,
        shape: vec![3, 2],
    };
    assert!(builder.tiled_lookup(&tokens, &table, &[1, 2]).is_err());
    let public = builder
        .push(
            Code::PublicInput {
                key: "public".into(),
                digest: [1; 32],
                row_l1: None,
                row_squared_norm: None,
                row_l1_squared_sum: None,
            },
            5,
            Interval::new(-1, 1).unwrap(),
        )
        .unwrap();
    let table = SourceView {
        owner: public,
        shape: vec![3, 2],
    };
    assert!(builder.tiled_lookup(&tokens, &table, &[1, 2]).is_err());
}

#[test]
fn tiled_lookup_common_four_plus_two_cost_inspection_not_admission() {
    let plan = Plan::new(4, 151936, 896, Interval::new(0, 151935).unwrap()).unwrap();
    let next = Plan::new(5, 151936, 896, Interval::new(0, 151935).unwrap()).unwrap();
    let work = plan
        .work_budget()
        .unwrap()
        .checked_add(next.work_budget().unwrap())
        .unwrap();
    let primitive =
        tensor::Program::equal_zero_bounded(128, Interval::new(-151935, 151935).unwrap()).unwrap();
    assert_eq!(work, primitive.work_budget().checked_mul(1187 * 9).unwrap());
    assert_eq!(work.operations(), 245709);
    assert_eq!(work.cursors().public_blocks, 10683);
    assert_eq!(work.cursors().pair_blocks, 6922584);
    assert_eq!(work.cursors().private_blocks, 0);
    assert_eq!(work.cursors().aes_work, 12488427);
    assert_eq!(primitive.work_budget().operations(), 23);
    assert_eq!(primitive.work_budget().cursors().pair_blocks, 648);
    assert_eq!(primitive.work_budget().cursors().aes_work, 1169);
    let boundary = primitive.work_budget().checked_mul(1187 * 540).unwrap();
    assert_eq!(boundary.operations(), 14742540);
    assert_eq!(boundary.cursors().aes_work, 749305620);
    assert!(boundary.limits().is_err()); // Codec changes do not raise program caps.
    assert_ne!(
        primitive.digest(),
        tensor::Program::equal_zero(128, Interval::new(-151935, 151935).unwrap())
            .unwrap()
            .digest()
    );
    assert!(work.limits().is_ok());
    eprintln!(
        "4+2 lookup-only work, not complete-program admission: {:?}",
        work
    );
    eprintln!(
        "Plan inline bytes={}, Arc allocation payload+counts={}, primitive workspace={}",
        std::mem::size_of::<Plan>(),
        std::mem::size_of::<Plan>() + 2 * std::mem::size_of::<usize>(),
        plan.predicate_workspace_bytes()
    );
}

#[test]
fn tiled_lookup_common_four_plus_two_generated_metadata_capacity_probe() {
    let config = include_bytes!("../../../../tests/fixtures/qwen25_0_5b_config.json");
    // Shape-only synthetic zero matrices. Never checkpoint bytes, never a
    // pinned-source admission claim, and no full-size matrix allocation.
    let sources = Program::source_shapes_with_weights(config, 4, 2, WeightMode::Public)
        .unwrap()
        .into_iter()
        .map(|(name, shape)| {
            let public_q18 = name
                .contains("norm")
                .then(|| vec![ONE; shape.iter().product()]);
            let bound = if public_q18.is_some() { ONE } else { 0 };
            let digest = public_q18.as_ref().map_or([1; 32], |values| {
                Sha256::digest(
                    values
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .into()
            });
            (
                name,
                SourceTensor {
                    shape,
                    lower: bound,
                    upper: bound,
                    public_q18,
                    digest,
                    row_l1: None,
                    row_squared_norm: None,
                    row_l1_squared_sum: None,
                },
            )
        })
        .collect();
    let policy = policy::ExecutionPolicy {
        lookup: policy::LookupPolicy::Tiled,
        ..policy::ExecutionPolicy::default()
    };
    match Program::compile_with_policy(config, 4, 2, &sources, WeightMode::Public, policy) {
        Ok(program) => eprintln!(
            "synthetic common 4+2 compiled: nodes={} work={:?}",
            program.nodes.len(),
            program.work_budget()
        ),
        Err(reason) => eprintln!("synthetic common 4+2 rejected without cap override: {reason}"),
    }
}
