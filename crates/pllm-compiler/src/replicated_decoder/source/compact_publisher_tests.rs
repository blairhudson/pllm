//! Generated source only. Definitions are NOT evidence until native gates run.
use super::*;

fn experiment() -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({"schema":"pllm.experiment.v1","name":"compact-client","pipeline":{"model":{"source":"original-tiny-source"},"components":{"linear":{"component":crate::replicated_decoder::sdk::PROTOCOL,"params":{"weights":"public","range_policy":"checked","max_abs_activation":1,"storage":"compact"}}}},"deployment":{"kind":"local","root":"."},"budget":{"requests":1,"max_input_tokens":4,"max_new_tokens":2}})).unwrap()
}
fn fixture() -> (Publisher, BTreeMap<String, Vec<i64>>) {
    let (config, _, values) = crate::replicated_decoder::tests::fixture_sources("qwen2", false);
    (
        Publisher::from_compact_source_request(&experiment(), &config, 4, 2, [2; 32], [3; 32])
            .unwrap(),
        values,
    )
}
fn collect(p: &mut Publisher, values: &BTreeMap<String, Vec<i64>>) {
    for index in 0..p.shapes().len() {
        let (key, shape) = &p.shapes()[index];
        let v = &values[key];
        assert_eq!(v.len(), shape.iter().product::<usize>());
        let key = key.clone();
        for (offset, chunk) in v.chunks(CHUNK).enumerate() {
            let bytes = chunk
                .iter()
                .flat_map(|v| (*v as f32 / ONE as f32).to_le_bytes())
                .collect::<Vec<_>>();
            p.append(&key, offset * CHUNK, &bytes).unwrap();
        }
    }
}
fn counter(p: &Publisher) -> OwnedBudget {
    let c = p.compact.as_ref().unwrap();
    let Stage::Collecting(o) = &c.stage else {
        panic!("not collecting")
    };
    o.payload().unwrap().root.try_clone().unwrap()
}

#[test]
fn compact_route_requires_two_epochs_and_real_experiment_before_collection() {
    assert!(Publisher::from_compact_source_request(&[], &[], 4, 1, [2; 32], [3; 32]).is_err());
    let (config, _, _) = crate::replicated_decoder::tests::fixture_sources("qwen2", false);
    let mut e: serde_json::Value = serde_json::from_slice(&experiment()).unwrap();
    e["pipeline"]["components"]["linear"]["params"]["storage"] = "expanded".into();
    assert!(Publisher::from_compact_source_request(
        &serde_json::to_vec(&e).unwrap(),
        &config,
        4,
        2,
        [2; 32],
        [3; 32]
    )
    .is_err());
}
#[test]
fn all_snapshots_include_gamma_and_actual_counter_survives_collection() {
    let (mut p, values) = fixture();
    let counter = counter(&p);
    assert_eq!(p.storage_kind(), None);
    collect(&mut p, &values);
    let c = p.compact.as_ref().unwrap();
    let Stage::Collecting(owner) = &c.stage else {
        panic!()
    };
    let state = owner.payload().unwrap();
    assert!(std::ptr::eq(counter.budget(), state.root.budget()));
    for ((key, shape), snapshot) in state.shapes.iter().zip(&state.snapshots) {
        assert_eq!(
            snapshot.encoded.size(),
            shape.iter().product::<usize>() as u64 * 8
        );
        assert_eq!(
            snapshot.raw.size(),
            shape.iter().product::<usize>() as u64 * 4
        );
        assert_eq!(
            snapshot.encoded.digest(),
            owner.manifest().tensors[key].digest
        );
    }
    assert!(owner
        .manifest()
        .tensors
        .values()
        .any(|v| v.public_q18.is_some()));
    let used = counter.budget().used_bytes();
    p.cancel();
    assert!(counter.budget().used_bytes() < used);
}
#[test]
fn conversion_error_burns_entire_collector_not_just_borrowed_scanner() {
    let (mut p, _) = fixture();
    let counter = counter(&p);
    let before = counter.budget().used_bytes();
    let key = p.shapes()[0].0.clone();
    assert!(p.append(&key, 0, &f32::NAN.to_le_bytes()).is_err());
    assert!(p.retired);
    assert!(p.compact.is_none());
    assert!(p.append(&key, 0, &0f32.to_le_bytes()).is_err());
    assert!(counter.budget().used_bytes() < before);
}
#[test]
fn wrong_order_and_incomplete_seal_never_export_executable_manifest() {
    let (mut p, _) = fixture();
    assert!(p
        .append("not-original-source", 0, &0f32.to_le_bytes())
        .is_err());
    assert!(p.manifest_checked().is_err());
    let (mut p, _) = fixture();
    assert!(p.seal(&[0, 1, 1, 2]).is_err());
    assert!(p.manifest_checked().is_err());
    assert!(p.retired);
}
#[test]
fn fallback_acceptance_repeats_original_constant_one_with_exact_bounds() {
    use crate::replicated_decoder::program::schedule_access::Kind;
    let (mut publisher, values) = fixture();
    collect(&mut publisher, &values);
    publisher.seal(&[0, 1, 1, 2]).unwrap();
    let Stage::Sealed(owner) = &publisher.compact.as_ref().unwrap().stage else {
        panic!("not sealed")
    };
    let access = owner.program().unwrap().program().schedule_access();
    let outputs = access.outputs().unwrap();
    assert_eq!(outputs.len(), 3);
    let (valid, lane) = outputs[2];
    assert_eq!(lane, 0);
    let one = access.node(valid).unwrap();
    let (anchor, right, coefficients) = one.affine().unwrap();
    assert_eq!(*right, None);
    assert_eq!(
        (coefficients.left, coefficients.right, coefficients.constant),
        (0, 0, 1)
    );
    assert_eq!(one.bounds(), Interval::new(1, 1).unwrap());
    let anchor = access.node(*anchor).unwrap();
    let root = anchor
        .lane_refs()
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(root.len(), 1);
    assert_eq!(root[0].1, 0);
    let input = access.node(root[0].0).unwrap();
    assert_eq!(input.kind(), Kind::Input);
    assert_eq!(input.lanes(), 4);
    assert_eq!(anchor.bounds(), input.bounds());
    let mask = access.node(outputs[0].0).unwrap();
    assert_eq!(outputs[1], (outputs[0].0, 1));
    let (selected, repeated, shape, rescale, norm) = mask.product().unwrap();
    assert!(shape.is_none() && !rescale && norm.is_none());
    let repeated = access.node(*repeated).unwrap();
    assert_eq!(repeated.bounds(), one.bounds());
    assert_eq!(
        repeated
            .lane_refs()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap(),
        [(valid, 0); 2]
    );
    assert_eq!(mask.bounds(), access.node(*selected).unwrap().bounds());
}
#[test]
fn genuine_seal_matches_expanded_oracle_without_production_expanded_construction() {
    let (mut p, values) = fixture();
    collect(&mut p, &values);
    p.seal(&[0, 1, 1, 2]).unwrap();
    assert_eq!(p.storage_kind(), Some("compact"));
    let manifest: Manifest = serde_json::from_slice(&p.manifest_checked().unwrap()).unwrap();
    assert_ne!(manifest.program, [0; 32]);
    let mut reference = Program::compile_with_norms(
        &manifest.config,
        4,
        2,
        &manifest.tensors,
        WeightMode::Public,
        manifest.execution,
        &manifest.norms,
    )
    .unwrap();
    let compact = {
        let Stage::Sealed(owner) = &p.compact.as_ref().unwrap().stage else {
            panic!()
        };
        let header = owner.program().unwrap().source_manifest().unwrap();
        assert_eq!(
            header.config, manifest.config,
            "first difference: original config bytes"
        );
        assert_eq!(
            header.source, manifest.source,
            "first difference: source identity"
        );
        assert_eq!(
            header.composition, manifest.composition,
            "first difference: composition identity"
        );
        assert_eq!(
            source_norms::digest(&header.norms),
            source_norms::digest(&manifest.norms),
            "first difference: canonical norms"
        );
        assert_eq!(
            program::source_binding_with_norms(&header.tensors, &header.norms),
            program::source_binding_with_norms(&manifest.tensors, &manifest.norms),
            "first difference: canonical source records"
        );
        Arc::clone(owner.program().unwrap().program())
    };
    let actual = compact.schedule_access();
    let expected = reference.schedule_access();
    assert_eq!(
        actual.source_binding().unwrap(),
        expected.source_binding().unwrap(),
        "first difference: compiled source prefix"
    );
    assert_eq!(
        actual.input_count().unwrap(),
        expected.input_count().unwrap(),
        "first difference: input prefix"
    );
    assert_eq!(
        actual.len(),
        expected.len(),
        "first difference: recipe count"
    );
    let mut first_exact_recipe_mismatch = None;
    for ordinal in 0..actual.len() {
        let a = actual.node(ordinal).unwrap();
        let b = expected.node(ordinal).unwrap();
        if matches!(
            (a.kind(), b.kind()),
            (
                program::schedule_access::Kind::GeneratedPublicValues,
                program::schedule_access::Kind::PublicValues
            )
        ) {
            // The original encoder hashes these representations identically.
            // Compare ACTUAL retained generated values to independent literals.
            let literals = b.public_values().unwrap();
            assert_eq!(actual.generated_len(ordinal).unwrap(), literals.len());
            for (index, literal) in literals.iter().enumerate() {
                assert_eq!(
                    actual.generated_element(ordinal, index).unwrap(),
                    *literal,
                    "first difference: generated/literal value at {ordinal}:{index}"
                );
            }
        } else {
            assert_eq!(a.kind(), b.kind(), "first difference: kind at {ordinal}");
        }
        assert_eq!(a.lanes(), b.lanes(), "first difference: lanes at {ordinal}");
        assert_eq!(
            a.bounds(),
            b.bounds(),
            "first difference: bounds at {ordinal}"
        );
        if !a.exact_oracle_recipe(b) && first_exact_recipe_mismatch.is_none() {
            first_exact_recipe_mismatch = Some(ordinal);
            eprintln!("first exact_oracle_recipe mismatch at {ordinal}, kind {:?}; this custody comparator uses Arc identity and returns false for unsupported families, NOT necessarily a numerical difference", a.kind());
        }
        // Real native digests, not pointer equality or a second canonical encoder.
        if let (Ok((ai, ap)), Ok((bi, bp))) = (a.tensor(), b.tensor()) {
            assert_eq!(
                (ai, ap.digest()),
                (bi, bp.digest()),
                "first difference: native tensor at {ordinal}"
            );
        }
        if let (Ok((ap, at)), Ok((bp, bt))) = (a.loop_programs(), b.loop_programs()) {
            assert_eq!(
                (ap.digest(), at.map(|p| p.digest())),
                (bp.digest(), bt.map(|p| p.digest())),
                "first difference: native full/tail at {ordinal}"
            );
        }
        if let (Ok((ai, aw, ap)), Ok((bi, bw, bp))) = (a.public_lookup(), b.public_lookup()) {
            assert_eq!(
                (ai, aw, ap.digest()),
                (bi, bw, bp.digest()),
                "first difference: native lookup at {ordinal}"
            );
        }
        if let (Ok((ai, aw, ap)), Ok((bi, bw, bp))) = (a.trie(), b.trie()) {
            assert_eq!(
                (ai, aw, ap.digest()),
                (bi, bw, bp.digest()),
                "first difference: native trie at {ordinal}"
            );
        }
        if let (Ok((ai, ap)), Ok((bi, bp))) = (a.selection(), b.selection()) {
            assert_eq!(
                (ai, ap.digest()),
                (bi, bp.digest()),
                "first difference: native selection at {ordinal}"
            );
        }
    }
    assert_eq!(
        compact.work_budget().operations(),
        reference.work_budget().operations(),
        "first difference: operation work"
    );
    assert_eq!(
        compact.work_budget().cursors(),
        reference.work_budget().cursors(),
        "first difference: work cursors"
    );
    assert_eq!(
        actual.outputs().unwrap(),
        expected.outputs().unwrap(),
        "first difference: output coordinates"
    );
    reference.bind(manifest.source, manifest.composition);
    assert_eq!(reference.digest(), manifest.program, "bound digest mismatch remains unresolved; first custody-recipe mismatch: {first_exact_recipe_mismatch:?}. If all structural checks above passed, inspect ORIGINAL canonical_prefix/finalization_prefix next; custody mismatch alone is not semantic proof.");
    let mut count = 0;
    while let Some(chunk) = p.next_chunk().unwrap() {
        for i in 0..chunk.shares[0].len() / 16 {
            let word = (0..3)
                .map(|j| {
                    u64::from_le_bytes(chunk.shares[j][i * 16..i * 16 + 8].try_into().unwrap())
                })
                .fold(0u64, u64::wrapping_add);
            assert_eq!(word, [0, 1, 1, 2][count]);
            count += 1;
        }
    }
    assert_eq!(count, 4);
}

fn actual_frames(mut p: Publisher) -> (Publisher, [Zeroizing<Vec<u8>>; 3]) {
    use crate::replicated_decoder::puma;
    let manifest: Manifest = serde_json::from_slice(&p.manifest_checked().unwrap()).unwrap();
    let program = {
        let Stage::Sealed(owner) = &p.compact.as_ref().unwrap().stage else {
            panic!()
        };
        Arc::clone(owner.program().unwrap().program())
    };
    let context = puma::Context::new(manifest.session, program.digest(), 0).unwrap();
    let runtimes = puma::prss::insecure_test_support::fresh_compound_parties(
        context,
        program.work_budget().limits().unwrap(),
    )
    .unwrap();
    let mut workers = runtimes.map(|runtime| {
        program
            .source_request_worker(runtime, manifest.session)
            .unwrap()
    });
    while let Some(chunk) = p.next_chunk().unwrap() {
        for (j, worker) in workers.iter_mut().enumerate() {
            let a = chunk.shares[j]
                .chunks_exact(16)
                .map(|r| u64::from_le_bytes(r[..8].try_into().unwrap()))
                .collect::<Vec<_>>();
            let b = chunk.shares[j]
                .chunks_exact(16)
                .map(|r| u64::from_le_bytes(r[8..].try_into().unwrap()))
                .collect::<Vec<_>>();
            worker.import(&chunk.key, &a, &b).unwrap();
        }
    }
    (p, routed_frames(workers))
}

fn routed_frames(mut workers: [crate::replicated_decoder::Worker; 3]) -> [Zeroizing<Vec<u8>>; 3] {
    use crate::replicated_decoder::Worker;
    let mut queue = Vec::new();
    for _ in 0..100_000 {
        if workers.iter().all(Worker::complete) {
            assert!(queue.is_empty());
            return workers.map(|w| w.finish_frame().unwrap());
        }
        if queue.is_empty() {
            for w in &mut workers {
                if !w.complete() {
                    let from = w.party();
                    queue.extend(
                        w.advance()
                            .unwrap()
                            .into_iter()
                            .map(|(to, f)| (from, to, f)),
                    );
                }
            }
        } else {
            for (from, to, frame) in std::mem::take(&mut queue) {
                queue.extend(
                    workers[to as usize]
                        .receive(frame.as_bytes(), from)
                        .unwrap()
                        .into_iter()
                        .map(|(next, f)| (to, next, f)),
                );
            }
        }
    }
    panic!("original tiny worker failed to complete");
}
#[test]
fn actual_original_workers_four_to_five_accept_two_final_ids() {
    let (mut p, values) = fixture();
    collect(&mut p, &values);
    p.seal(&[0, 1, 1, 2]).unwrap();
    let (p, f) = actual_frames(p);
    assert_eq!(
        p.finish_numeric_outcome([&f[0], &f[1], &f[2]]).unwrap(),
        Some(vec![1, 0])
    );
}
#[test]
fn actual_early_and_late_activation_failures_mask_all_final_ids() {
    let (late, _) = range_fixture(true, &[0, 0, 0, 0]);
    // Independent original one-epoch control proves SAFE prefix accepts and
    // selects feedback ID 1 before the two-epoch candidate is executed.
    let (mut control, clear) = range_fixture(false, &[0, 0, 0, 0]);
    // Ordinary stream completion advances the real publisher issuance cursor.
    // The independent clear-role control below uses the SAME fixture inputs.
    while control.next_chunk().unwrap().is_some() {}
    let workers = crate::replicated_decoder::tests::workers(
        Arc::clone(control.program.as_ref().unwrap()),
        &clear,
    );
    let f = routed_frames(workers);
    assert_eq!(opened_words(&f), [1, 1]);
    assert_eq!(
        control
            .finish_numeric_outcome([&f[0], &f[1], &f[2]])
            .unwrap(),
        Some(vec![1])
    );
    // Only feedback token 1's ±1.5 embedding violates checked cap 1 here.
    let (late, f) = actual_frames(late);
    assert_eq!(opened_words(&f), [0, 0, 0]);
    assert_eq!(
        late.finish_numeric_outcome([&f[0], &f[1], &f[2]]).unwrap(),
        None
    );
    // Genuine unsafe INITIAL activation: original shares, no post-seal mutation.
    let (early, _) = range_fixture(true, &[1, 0, 0, 0]);
    let (early, f) = actual_frames(early);
    assert_eq!(opened_words(&f), [0, 0, 0]);
    assert_eq!(
        early.finish_numeric_outcome([&f[0], &f[1], &f[2]]).unwrap(),
        None
    );
}

fn range_fixture(compact: bool, ids: &[u32; 4]) -> (Publisher, BTreeMap<String, Vec<i64>>) {
    let (config, _, _) = crate::replicated_decoder::tests::fixture_sources("qwen2", false);
    let mut p = if compact {
        Publisher::from_compact_source_request(&experiment(), &config, 4, 2, [2; 32], [3; 32])
            .unwrap()
    } else {
        Publisher::new_with_policy(
            &config,
            4,
            1,
            [2; 32],
            [3; 32],
            WeightMode::Public,
            ExecutionPolicy {
                range_policy: RangePolicy::Checked,
                max_abs_activation: 1,
                ..ExecutionPolicy::default()
            },
        )
        .unwrap()
    };
    if !compact {
        p.manifest.session = [42; 32];
    } // Original expanded test-runtime session only.
    let mut values = p
        .shapes()
        .iter()
        .map(|(key, shape)| {
            let floats = if key == "model.embed_tokens.weight" {
                vec![0.25f32, -0.25, 1.5, -1.5, 0.75, -0.75]
            } else if key == "lm_head.weight" {
                vec![-0.25, 0.25, 0.25, -0.25, 0.0, 0.0]
            } else if key.contains("norm") {
                vec![1.0; shape.iter().product()]
            } else {
                vec![0.0; shape.iter().product()]
            };
            (
                key.clone(),
                floats
                    .iter()
                    .map(|v| encode_q18(&v.to_le_bytes()).unwrap())
                    .collect(),
            )
        })
        .collect::<BTreeMap<String, Vec<i64>>>();
    collect(&mut p, &values);
    p.seal(ids).unwrap();
    values.insert(
        "input.tokens".into(),
        ids.iter().map(|v| i64::from(*v)).collect(),
    );
    (p, values)
}
fn opened_words(frames: &[Zeroizing<Vec<u8>>; 3]) -> Vec<u64> {
    (73..frames[0].len())
        .step_by(8)
        .map(|offset| {
            frames.iter().fold(0u64, |s, f| {
                s.wrapping_add(u64::from_le_bytes(
                    f[offset..offset + 8].try_into().unwrap(),
                ))
            })
        })
        .collect()
}
#[test]
fn handle_is_send_sync_and_explicit_cancel_retires_sealed_payloads() {
    fn required<T: Send + Sync>() {}
    required::<Publisher>();
    let (mut p, values) = fixture();
    let counter = counter(&p);
    collect(&mut p, &values);
    p.seal(&[0, 1, 1, 2]).unwrap();
    let live = counter.budget().used_bytes();
    p.cancel();
    assert!(counter.budget().used_bytes() < live);
    assert!(p.next_chunk().is_err());
}

#[test]
fn unwind_inside_original_collection_phase_burns_and_drops_payload_without_retry() {
    let (mut p, _) = fixture();
    let counter = counter(&p);
    let c = p.compact.as_mut().unwrap();
    let Stage::Collecting(owner) = &mut c.stage else {
        panic!()
    };
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _: Result<(), Error> =
            owner.phase(c.manifest_bytes, c.collection_bytes, c.visits, |_, _, _| {
                panic!("injected collector unwind")
            });
    }));
    assert!(failed.is_err());
    assert!(owner.payload().is_err());
    let retained = counter.budget().used_bytes();
    assert!(p.seal(&[0, 1, 1, 2]).is_err());
    assert!(p.retired);
    assert!(counter.budget().used_bytes() < retained);
    // Counter release, not empirical payload-deallocation/erasure timing.
}
#[test]
fn failed_token_binding_consumes_collection_and_cannot_be_resealed() {
    let (mut p, values) = fixture();
    collect(&mut p, &values);
    assert!(p.seal(&[0, 1, 1, 3]).is_err());
    assert!(p.seal(&[0, 1, 1, 2]).is_err());
    assert_eq!(p.storage_kind(), None);
    assert!(p.manifest_checked().is_err());
}
