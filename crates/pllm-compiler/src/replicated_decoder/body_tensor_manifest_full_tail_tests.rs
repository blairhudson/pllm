//! Genuine original Manifest/Worker controls, not production permission.
//! Oracles/reports are separate test owners, not enclosing-request admission.
use super::*;
use std::sync::Weak;

const TOKENS: [u64; 4] = [1, 2, 1, 2];

struct Fixture {
    manifest: crate::replicated_decoder::Manifest,
    program: Arc<Program>,
    compiled: Arc<Compiled>,
    public: Arc<PublicSource>,
    workers: [Worker; 3],
    usage: [puma::PayloadUsage; 3],
    full: Weak<puma::multiply::tensor::Program>,
    tail: Option<Weak<puma::multiply::tensor::Program>>,
    input: usize,
    output: usize,
    rows: usize,
}

fn workers(
    program: &Arc<Program>,
    manifest: &crate::replicated_decoder::Manifest,
    tokens: &[u64],
) -> ([Worker; 3], [puma::PayloadUsage; 3]) {
    let roots = puma::prss::insecure_test_support::fresh_compound_parties(
        puma::Context::new([42; 32], program.digest(), 0).unwrap(),
        program.work_budget().limits().unwrap(),
    )
    .unwrap();
    let usage = roots.each_ref().map(|root| root.payload_usage());
    let mut roles = roots.map(|root| {
        Worker::new_with_source(
            program.clone(),
            root,
            [42; 32],
            Some(manifest_prepared_fixture::import_projection_geometry_source(manifest).unwrap()),
        )
        .unwrap()
    });
    for (key, lanes) in program.inputs() {
        assert_eq!(key, "input.tokens");
        assert_eq!(lanes, tokens.len());
        let first: Vec<_> = (0..lanes)
            .map(|i| 0xc751f00d12345678u64.wrapping_add(i as u64 * 17))
            .collect();
        let second: Vec<_> = (0..lanes)
            .map(|i| 0x254277811928f93du64.wrapping_add(i as u64 * 311))
            .collect();
        let third: Vec<_> = tokens
            .iter()
            .zip(&first)
            .zip(&second)
            .map(|((&clear, &a), &b)| clear.wrapping_sub(a).wrapping_sub(b))
            .collect();
        let shares = [first, second, third];
        for (party, role) in roles.iter_mut().enumerate() {
            role.import(key, &shares[party], &shares[(party + 1) % 3])
                .unwrap();
        }
    }
    (roles, usage)
}

fn fixture(hidden: usize, rows: usize, reject_prompt: bool) -> Fixture {
    let manifest =
        manifest_prepared_fixture::public_projection_geometry_manifest(hidden, rows, reject_prompt)
            .expect("ordinary Publisher/Manifest admission must accept fixture geometry");
    assert_source_geometry(&manifest, hidden, rows);
    assert!(TOKENS[..rows].iter().all(|&token| matches!(token, 1 | 2)));
    let embedding = &manifest.tensors["model.embed_tokens.weight"];
    assert_eq!(embedding.lower, ONE / 4);
    assert_eq!(
        embedding.upper,
        if reject_prompt { 2 * ONE } else { ONE * 3 / 2 }
    );
    let (program, prepared, input, output) =
        manifest_prepared_fixture::compile_manifest_rescale18(&manifest, Q_PROJECTION_SOURCE_KEY)
            .expect("exact Manifest compile must publish original full/tail capability");
    assert_eq!(program.digest(), manifest.program);
    assert_eq!(prepared.lanes(), rows * hidden);
    assert_eq!(prepared.invocations(), (rows * hidden).div_ceil(128) as u64);
    let Code::TensorLoop {
        input: LoopInput::Slice(original_input),
        program: original_full,
        tail: original_tail,
    } = &program.nodes[output].code
    else {
        panic!("genuine original Slice TensorLoop required")
    };
    assert_eq!(*original_input, input);
    assert_eq!(original_full.lanes(), 128);
    assert_eq!(original_full.instructions(), 1);
    assert!(Arc::ptr_eq(original_full, prepared.program_at(0).unwrap()));
    let remainder = (rows * hidden) % 128;
    assert_eq!(original_tail.is_some(), remainder != 0);
    if let Some(tail) = original_tail {
        assert_eq!(tail.lanes(), remainder);
        assert_eq!(tail.instructions(), 1);
        assert!(Arc::ptr_eq(
            tail,
            prepared.program_at(rows * hidden - remainder).unwrap()
        ));
    }
    let full = Arc::downgrade(original_full);
    let tail = original_tail.as_ref().map(Arc::downgrade);
    let Code::PublicProduct { weight, .. } = program.nodes[input].code else {
        panic!("capture input must be genuine source-backed public product")
    };
    assert!(matches!(&program.nodes[weight].code,
        Code::PublicInput { key, .. } if key == Q_PROJECTION_SOURCE_KEY));
    let (workers, usage) = workers(&program, &manifest, &TOKENS[..rows]);
    let public =
        Arc::new(manifest_prepared_fixture::import_projection_geometry_source(&manifest).unwrap());
    let registry = SourceRegistry::from_manifest(&manifest).unwrap();
    let outside = add(manifest_inline_baseline(&manifest), size_of::<Fixture>()).unwrap();
    let source =
        SourceDispatch::new(&program, &workers[0], public.clone(), &registry, outside).unwrap();
    let before = native_attempts();
    let compiled = Compiled::prepare_checked(
        program.clone(),
        WeightMode::Public,
        input,
        output,
        prepared,
        Some(source),
        || Ok(()),
    )
    .unwrap();
    assert_eq!(native_attempts(), before);
    Fixture {
        manifest,
        program,
        compiled,
        public,
        workers,
        usage,
        full,
        tail,
        input,
        output,
        rows,
    }
}

fn assert_source_geometry(
    manifest: &crate::replicated_decoder::Manifest,
    hidden: usize,
    rows: usize,
) {
    let config = pllm_models::QwenConfig::from_json(&manifest.config).unwrap();
    config.validate().unwrap();
    let (heads, head_dim) = match hidden {
        64 => (16, 4),
        66 => (11, 6),
        _ => panic!("unqualified projection geometry"),
    };
    assert_eq!(config.hidden_size, hidden as u64);
    assert_eq!(u64::from(config.num_attention_heads), heads);
    assert_eq!(config.head_dim(), head_dim); // Actual Qwen2 rule, not the JSON hint.
    assert_eq!(heads * head_dim, hidden as u64);
    assert_eq!(config.num_key_value_heads, 1);
    assert_eq!(config.num_attention_heads % config.num_key_value_heads, 0);
    assert!(heads <= 16 && head_dim <= 128 && head_dim % 2 == 0);
    assert_eq!(config.max_position_embeddings, 16);
    assert_eq!((manifest.rows, manifest.outputs), (rows, 1));
    assert!((rows + manifest.outputs) as u64 <= config.max_position_embeddings);
    assert_eq!(manifest.weights, WeightMode::Public);
    assert_eq!(
        manifest.execution.range_policy,
        crate::replicated_decoder::RangePolicy::Checked,
    );
    assert_eq!(manifest.execution.max_abs_activation, 1);

    // These are the genuine Publisher's parameter-file descriptors. K/V widths
    // change with the valid factorization; Q/output/embedding widths do not.
    for (key, shape) in [
        (Q_PROJECTION_SOURCE_KEY, vec![hidden, hidden]),
        (
            "model.layers.0.self_attn.k_proj.weight",
            vec![head_dim as usize, hidden],
        ),
        (
            "model.layers.0.self_attn.v_proj.weight",
            vec![head_dim as usize, hidden],
        ),
        (
            "model.layers.0.self_attn.o_proj.weight",
            vec![hidden, hidden],
        ),
        ("model.embed_tokens.weight", vec![3, hidden]),
        ("model.layers.0.self_attn.q_proj.bias", vec![hidden]),
        (
            "model.layers.0.self_attn.k_proj.bias",
            vec![head_dim as usize],
        ),
        (
            "model.layers.0.self_attn.v_proj.bias",
            vec![head_dim as usize],
        ),
        ("model.layers.0.input_layernorm.weight", vec![hidden]),
        (
            "model.layers.0.post_attention_layernorm.weight",
            vec![hidden],
        ),
        ("model.norm.weight", vec![hidden]),
    ] {
        assert_eq!(manifest.tensors[key].shape, shape);
    }
    // This original source is Qwen2: it has the three RMS vectors above,
    // not Qwen3's separate per-head Q/K norm weights.
    for key in [
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
    ] {
        assert!(!manifest.tensors.contains_key(key));
    }
    let plan = crate::replicated_decoder::program::lower_with_weights(
        &manifest.config,
        rows,
        1,
        WeightMode::Public,
    )
    .expect("unchanged ordinary source expansion budget must admit fixture");
    let projection = plan
        .prefill
        .operations
        .iter()
        .find(|op| {
            op.operator == pllm_models::ModelOperator::Linear
                && op.attributes.get("weight").and_then(|value| value.as_str())
                    == Some(Q_PROJECTION_SOURCE_KEY)
        })
        .unwrap();
    assert_eq!(projection.output_shape, [1, rows as u64, hidden as u64]);
    assert_eq!(
        projection.output_shape.iter().product::<u64>(),
        (rows * hidden) as u64
    );
    assert!(matches!(rows * hidden, 128 | 132 | 256 | 264));

    // The original two-lane-head fixture was architectural Qwen2 geometry, but
    // exceeded the caller's actual source budget. Keep that boundary negative;
    // do not replace ordinary admission with a copied or widened budget.
    let mut over_budget: serde_json::Value = serde_json::from_slice(&manifest.config).unwrap();
    over_budget["num_attention_heads"] = serde_json::json!(hidden / 2);
    over_budget["head_dim"] = serde_json::json!(2);
    let over_budget = serde_json::to_vec(&over_budget).unwrap();
    let old_config = pllm_models::QwenConfig::from_json(&over_budget).unwrap();
    old_config.validate().unwrap();
    assert_eq!(old_config.head_dim(), 2);
    assert!(old_config.num_attention_heads > 16);
    let error = crate::replicated_decoder::program::lower_with_weights(
        &over_budget,
        rows,
        1,
        WeightMode::Public,
    )
    .expect_err("32/33-head geometry must retain original source-budget rejection");
    assert!(error.contains("decoder source expansion budget exceeded or invalid"));
}

// Preserve actual wave routing/order, retaining at most ONE delivered lease.
// A larger decoder transcript is not permission to accumulate every native
// PeerBytes reservation beyond immutable runtime caps. Wire reports copy bytes.
fn route<T: Role>(
    roles: &mut [T; 3],
    mut queue: Vec<Frame>,
    wire: &mut Wire,
    held: &mut Vec<PeerBytes>,
) {
    for _ in 0..100_000 {
        if queue.is_empty() {
            return;
        }
        for (from, to, raw) in std::mem::take(&mut queue) {
            wire.push((from, to, raw.as_bytes().to_vec()));
            queue.extend(
                roles[to as usize]
                    .receive(raw.as_bytes(), from)
                    .unwrap()
                    .into_iter()
                    .map(|(next, frame)| (to, next, frame)),
            );
            if held.is_empty() {
                held.push(raw);
            }
        }
    }
    panic!("real router failed to drain");
}

fn streamed<T: Role>(roles: &mut [T; 3]) -> (Wire, Vec<PeerBytes>) {
    let mut wire = Vec::new();
    let mut held = Vec::new();
    for _ in 0..100_000 {
        if roles.iter().all(Role::complete) {
            return (wire, held);
        }
        let queue = start(roles);
        route(roles, queue, &mut wire, &mut held);
    }
    panic!("real roles failed to complete");
}

fn boundary(roles: &mut [Worker; 3], target: usize, compiled: Option<&Compiled>) -> Wire {
    let mut wire = Vec::new();
    for _ in 0..100_000 {
        let mut queue = Vec::new();
        let mut reached = 0;
        for role in roles.iter_mut() {
            let from = role.party();
            let progress = match compiled {
                Some(c) => c.advance_to_start(role),
                None => role.advance_before(target),
            }
            .unwrap();
            match progress {
                BoundaryProgress::Reached => reached += 1,
                BoundaryProgress::Frames(frames) => {
                    queue.extend(frames.into_iter().map(|(to, raw)| (from, to, raw)))
                }
            }
        }
        if reached == 3 {
            assert!(queue.is_empty());
            return wire;
        }
        route(roles, queue, &mut wire, &mut Vec::new());
    }
    panic!("original all-role boundary not reached");
}

fn sticky_before(program: &Program, output: usize) -> usize {
    let mut valid = program.outputs.last().unwrap().0;
    assert!(
        program.nodes[..output]
            .iter()
            .any(|node| matches!(node.code, Code::CheckedRange { .. })),
        "source must produce a genuine pre-Q range guard, not constant acceptance",
    );
    eprintln!(
        "original sticky producer: output={output}, valid={valid}, checked_before={}, checked_total={}, final_kind={}",
        program.nodes[..output]
            .iter()
            .filter(|node| matches!(node.code, Code::CheckedRange { .. }))
            .count(),
        program
            .nodes
            .iter()
            .filter(|node| matches!(node.code, Code::CheckedRange { .. }))
            .count(),
        match &program.nodes[valid].code {
            Code::Product { .. } => "Product",
            Code::Affine { .. } => "Affine",
            Code::Tensor { .. } => "Tensor",
            Code::Gather(_) => "Gather",
            _ => "other",
        },
    );
    while valid >= output {
        let Code::Product {
            left,
            rescale: false,
            ..
        } = program.nodes[valid].code
        else {
            panic!("original sticky aggregate must retain earlier guards")
        };
        valid = left;
    }
    assert_eq!(program.nodes[valid].lanes, 1);
    valid
}

fn clear_owner(states: &[Snapshot; 3], owner: usize) -> u64 {
    let (a, b) = states[0].shares[owner].as_ref().unwrap();
    let (_, c) = states[1].shares[owner].as_ref().unwrap();
    assert_eq!(a.len(), 1);
    a[0].wrapping_add(b[0]).wrapping_add(c[0])
}

#[test]
fn genuine_manifest_original_worker_full_tail_tile_boundaries_wire_shares_cursors_and_sticky_suffix(
) {
    run_large_stack(|| {
        for (hidden, rows, reject) in [
            (64, 2, false),
            (66, 2, false),
            (64, 4, false),
            (66, 4, false),
            (66, 2, true),
        ] {
            parity(hidden, rows, reject);
        }
    });
}

fn parity(hidden: usize, rows: usize, reject: bool) {
    let Fixture {
        manifest,
        program,
        compiled,
        public,
        workers: mut candidate,
        usage,
        full,
        tail,
        input,
        output,
        rows,
    } = fixture(hidden, rows, reject);
    // Independent unarmed default compiler/Worker, never a BODY oracle.
    let ordinary = manifest.compile().unwrap();
    assert_eq!(ordinary.digest(), program.digest());
    assert_eq!(ordinary.source_binding, program.source_binding);
    assert_eq!(ordinary.outputs, program.outputs);
    assert_eq!(ordinary.retire, program.retire);
    assert_eq!(ordinary.payload_bytes(), program.payload_bytes());
    assert_eq!(ordinary.work_budget(), program.work_budget());
    let (mut whole, _) = workers(&ordinary, &manifest, &TOKENS[..rows]);
    let before = native_attempts();
    let (whole_wire, whole_held) = streamed(&mut whole);
    let whole_state = whole.each_ref().map(snapshot);
    let whole_result = whole.map(|role| role.finish().unwrap().to_vec());
    let (mut reference, _) = workers(&ordinary, &manifest, &TOKENS[..rows]);
    let mut wire = boundary(&mut candidate, output, Some(&compiled));
    assert_eq!(wire, boundary(&mut reference, output, None));
    let prefix = candidate.each_ref().map(snapshot);
    assert_eq!(prefix, reference.each_ref().map(snapshot));
    let sticky = sticky_before(&program, output);
    assert_eq!(clear_owner(&prefix, sticky), u64::from(!reject));
    let expected_body = boundary(&mut reference, output + 1, None);
    let registry = SourceRegistry::from_manifest(&manifest).unwrap();
    let [a, b, c] = TestCoordinator {
        roles: candidate,
        queue: Vec::new(),
    }
    .certify_start(&compiled)
    .unwrap();
    let mut bodies = [
        compiled.clone().begin_with_registry(a, &registry).unwrap(),
        compiled.clone().begin_with_registry(b, &registry).unwrap(),
        compiled.clone().begin_with_registry(c, &registry).unwrap(),
    ];
    #[allow(clippy::drop_non_drop)] // End the registry borrow before original owners.
    drop(registry);
    let full_base = full.strong_count();
    let tail_base = tail.as_ref().map(Weak::strong_count);
    let lanes = rows * hidden;
    let mut body_wire = Vec::new();
    let mut body_held = Vec::new();
    for offset in (0..lanes).step_by(128) {
        let frames = start(&mut bodies);
        assert!(!frames.is_empty());
        let is_tail = lanes - offset < 128;
        assert_eq!(full.strong_count(), full_base + if is_tail { 0 } else { 3 });
        if let Some(tail) = &tail {
            assert_eq!(
                tail.strong_count(),
                tail_base.unwrap() + if is_tail { 3 } else { 0 }
            );
        }
        route(&mut bodies, frames, &mut body_wire, &mut body_held);
    }
    let (last, last_held) = streamed(&mut bodies);
    assert!(last.is_empty(), "no extra rebuilt/native invocation");
    assert!(last_held.is_empty());
    assert_eq!(full.strong_count(), full_base);
    assert_eq!(tail.as_ref().map(Weak::strong_count), tail_base);
    assert_eq!(body_wire, expected_body);
    wire.extend(body_wire);
    let mut resumed = restore_after_drive(bodies).unwrap();
    let seam = resumed.each_ref().map(snapshot);
    assert_eq!(seam, reference.each_ref().map(snapshot));
    for state in &seam {
        assert_eq!(state.step, output + 1);
        assert_eq!(state.shares[output].as_ref().unwrap().0.len(), lanes);
        assert_eq!(state.shares[output].as_ref().unwrap().1.len(), lanes);
        assert!(state.shares[input].is_none());
    }
    assert_eq!(clear_owner(&seam, sticky), u64::from(!reject));
    let (suffix, suffix_held) = streamed(&mut resumed);
    assert!(!suffix.is_empty());
    wire.extend(suffix);
    assert_eq!(wire, whole_wire);
    let final_state = resumed.each_ref().map(snapshot);
    assert_eq!(final_state, whole_state);
    assert_eq!(
        clear_owner(&final_state, program.outputs.last().unwrap().0),
        u64::from(!reject)
    );
    assert_eq!(
        resumed.map(|role| role.finish().unwrap().to_vec()),
        whole_result
    );
    assert_eq!(native_attempts(), before);
    assert!(usage.iter().any(|ledger| ledger.live_bytes() > 0));
    drop((reference, compiled, program, public, manifest));
    assert_eq!(full.strong_count(), 0);
    assert_eq!(tail.as_ref().map(Weak::strong_count), tail_base.map(|_| 0));
    drop((whole_held, body_held, suffix_held));
    assert!(usage.iter().all(|ledger| ledger.live_bytes() == 0));
    assert!(matches!(
        worker_parity_gate(),
        Err(PreparationError::UnqualifiedOriginalWorkerParity)
    ));
}

#[test]
fn genuine_manifest_original_tail_cancel_and_bound_frame_error_retire_release_native_and_source_owners(
) {
    run_large_stack(|| {
        for wrong_role in [false, true] {
            let Fixture {
                manifest,
                program,
                compiled,
                public,
                workers: mut roles,
                usage,
                full,
                tail,
                output,
                ..
            } = fixture(66, 4, false);
            let tail = tail.expect("genuine 264-lane projection must have tail");
            let source = Arc::downgrade(&public);
            let before = native_attempts();
            boundary(&mut roles, output, Some(&compiled));
            let registry = SourceRegistry::from_manifest(&manifest).unwrap();
            let [a, b, c] = TestCoordinator {
                roles,
                queue: Vec::new(),
            }
            .certify_start(&compiled)
            .unwrap();
            let mut bodies = [
                compiled.clone().begin_with_registry(a, &registry).unwrap(),
                compiled.clone().begin_with_registry(b, &registry).unwrap(),
                compiled.clone().begin_with_registry(c, &registry).unwrap(),
            ];
            #[allow(clippy::drop_non_drop)] // End the registry borrow before original owners.
            drop(registry);
            let tail_base = tail.strong_count();
            let mut wire = Vec::new();
            let mut held = Vec::new();
            for _ in 0..2 {
                let frames = start(&mut bodies);
                assert!(!frames.is_empty());
                route(&mut bodies, frames, &mut wire, &mut held);
            }
            let tail_frames = start(&mut bodies);
            assert!(!tail_frames.is_empty());
            assert_eq!(tail.strong_count(), tail_base + 3);
            let bytes: Vec<_> = tail_frames
                .iter()
                .map(|(_, _, raw)| raw.as_bytes().to_vec())
                .collect();
            if wrong_role {
                let (from, to, raw) = &tail_frames[0];
                // Genuine bound tail frame, wrong sender: no invented header
                // and no claim that this establishes cross-attempt replay.
                assert!(bodies[*to as usize]
                    .receive(raw.as_bytes(), from.next())
                    .is_err());
                assert!(matches!(
                    bodies[*to as usize].advance(),
                    Err(Error::Retired)
                ));
            }
            for body in &mut bodies {
                body.cancel();
                assert!(!body.complete());
                assert!(matches!(body.advance(), Err(Error::Retired)));
                assert!(matches!(
                    body.receive(b"bad", body.party().next()),
                    Err(Error::Retired)
                ));
            }
            assert!(matches!(restore_after_drive(bodies), Err(Error::Retired)));
            assert_eq!(tail.strong_count(), tail_base);
            drop((compiled, program, public, manifest));
            assert_eq!(full.strong_count(), 0);
            assert_eq!(tail.strong_count(), 0);
            assert_eq!(source.strong_count(), 0);
            assert!(usage.iter().any(|ledger| ledger.live_bytes() > 0));
            for ((_, _, raw), expected) in tail_frames.iter().zip(bytes) {
                assert_eq!(raw.as_bytes(), expected);
            }
            drop((held, tail_frames));
            assert!(usage.iter().all(|ledger| ledger.live_bytes() == 0));
            assert_eq!(native_attempts(), before);
            assert!(matches!(
                worker_parity_gate(),
                Err(PreparationError::UnqualifiedOriginalWorkerParity)
            ));
        }
    });
}
