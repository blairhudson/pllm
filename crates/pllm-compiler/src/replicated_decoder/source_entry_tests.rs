use super::*;
use crate::replicated_decoder::{sdk::SourceRequestImport, Publisher};

type Fixture = (Publisher, Vec<u8>, Vec<(String, Vec<u8>)>);
fn fixture() -> Fixture {
    let config=br#"{"model_type":"qwen2","hidden_size":2,"intermediate_size":2,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"vocab_size":3,"max_position_embeddings":16,"rms_norm_eps":0.000001,"hidden_act":"silu","tie_word_embeddings":false,"attention_bias":false,"rope_theta":10000.0}"#;
    let mut publisher = Publisher::new_with_policy(
        config,
        4,
        2,
        [31; 32],
        [32; 32],
        WeightMode::Public,
        ExecutionPolicy {
            range_policy: RangePolicy::Checked,
            max_abs_activation: 1,
            ..ExecutionPolicy::default()
        },
    )
    .unwrap();
    let mut data = Vec::new();
    for (key, shape) in publisher.shapes() {
        let n: usize = shape.iter().product();
        let value = if key.contains("norm") { 1f32 } else { 0f32 };
        let bytes = value.to_le_bytes().repeat(n);
        data.push((key.clone(), bytes));
    }
    for (key, bytes) in &data {
        publisher.append(key, 0, bytes).unwrap();
    }
    publisher.seal(&[0, 1, 0, 1]).unwrap();
    let raw = publisher.manifest().unwrap();
    (publisher, raw, data)
}

#[test]
fn native_entry_constructs_from_actual_config_and_files_before_request_two_ids() {
    let (mut publisher, raw, data) = fixture();
    let mut entry = SourceRequestImport::new(&raw).unwrap();
    let counter = entry.0.as_ref().unwrap().root.try_clone().unwrap();
    for (key, bytes) in &data {
        entry.append(key, 0, bytes).unwrap()
    }
    let request = entry.finish().unwrap();
    let program = Arc::clone(request.program());
    let session = serde_json::from_slice::<Manifest>(&raw).unwrap().session;
    let context = puma::Context::new(session, program.digest(), 0).unwrap();
    let parties = puma::prss::insecure_test_support::fresh_compound_parties(
        context,
        program.work_budget().limits().unwrap(),
    )
    .unwrap();
    let mut workers =
        parties.map(|runtime| program.source_request_worker(runtime, session).unwrap());
    while let Some(chunk) = publisher.next_chunk().unwrap() {
        for (worker, raw) in workers.iter_mut().zip(&chunk.shares) {
            let a: Vec<_> = raw
                .chunks_exact(16)
                .map(|x| u64::from_le_bytes(x[..8].try_into().unwrap()))
                .collect();
            let b: Vec<_> = raw
                .chunks_exact(16)
                .map(|x| u64::from_le_bytes(x[8..].try_into().unwrap()))
                .collect();
            worker
                .import_chunk(&chunk.key, chunk.offset, &a, &b)
                .unwrap();
        }
    }
    // No facade/registry/raw frontend owner remains; each original Worker must
    // alone retain source/config guards through the final selected feedback.
    drop(request);
    drop(program);
    let empty = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    assert!(counter.budget().used_bytes() > empty.budget().used_bytes());
    let mut queue = Vec::new();
    for round in 0..100_000 {
        if workers.iter().all(Worker::complete) {
            break;
        }
        assert!(round + 1 < 100_000);
        if queue.is_empty() {
            for worker in &mut workers {
                let from = worker.party();
                if !worker.complete() {
                    for (to, frame) in worker.advance().unwrap() {
                        queue.push((from, to, frame));
                    }
                }
            }
        } else {
            for (from, to, frame) in std::mem::take(&mut queue) {
                for (dest, reply) in workers[to as usize]
                    .receive(frame.as_bytes(), from)
                    .unwrap()
                {
                    queue.push((to, dest, reply));
                }
            }
        }
    }
    let frames = workers.map(|worker| worker.finish_frame().unwrap());
    assert_eq!(counter.budget().used_bytes(), empty.budget().used_bytes());
    assert_eq!(
        publisher
            .finish([&frames[0], &frames[1], &frames[2]])
            .unwrap(),
        vec![0, 0]
    );
}

#[test]
fn failed_source_append_burns_entry_and_no_precompiled_plan_can_replace_it() {
    let (_, raw, _) = fixture();
    let mut entry = SourceRequestImport::new(&raw).unwrap();
    assert!(entry.append("foreign.source", 0, &[0; 4]).is_err());
    assert!(matches!(entry.session(), Err(Error::Retired)));
    assert!(entry.finish().is_err());
}

#[test]
fn missing_source_is_not_late_admitted_as_an_empty_snapshot() {
    let (_, raw, _) = fixture();
    assert!(SourceRequestImport::new(&raw).unwrap().finish().is_err());
}

#[test]
fn genuine_source_rejects_foreign_counter_and_changed_workload_before_input_factories() {
    let (_, raw, _) = fixture();
    let owner = SourceRequestImport::new(&raw).unwrap().0.take().unwrap();
    let foreign = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let guard = foreign.charge_external(0).unwrap();
    let owners = SourceConstructionOwners {
        manifest: owner.manifest.payload(),
        incoming: None,
        public: None,
        retained_baseline_bytes: 0,
    };
    assert!(matches!(
        SourceRegistry::from_original_records_checked(
            &owner.original,
            &owners,
            &guard,
            MAX_PROGRAM_BYTES
        ),
        Err(Error::Binding)
    ));
    let mut changed: Manifest = serde_json::from_slice(&raw).unwrap();
    changed.rows += 1;
    let owners = SourceConstructionOwners {
        manifest: &changed,
        incoming: None,
        public: None,
        retained_baseline_bytes: 0,
    };
    assert!(matches!(
        SourceRegistry::from_original_records_checked(
            &owner.original,
            &owners,
            &owner.registry,
            MAX_PROGRAM_BYTES
        ),
        Err(Error::Binding)
    ));
}

#[test]
fn genuine_import_prequote_exact_and_one_short_preserve_record_and_original_counter() {
    for short in [true, false] {
        let (_, raw, data) = fixture();
        let mut entry = SourceRequestImport::new(&raw).unwrap();
        for (key, bytes) in &data {
            entry.append(key, 0, bytes).unwrap();
        }
        let Pending {
            importer,
            original,
            manifest,
            mut imports,
            registry,
            root,
            ..
        } = entry.0.take().unwrap();
        let public = Arc::new(importer.finish().unwrap());
        let owners = SourceConstructionOwners {
            manifest: manifest.payload(),
            incoming: None,
            public: Some(&public),
            retained_baseline_bytes: root.budget().used_bytes() - registry.payload_bytes(),
        };
        let checked = SourceRegistry::from_original_checked(
            &original,
            &owners,
            &public,
            &registry,
            MAX_PROGRAM_BYTES,
        )
        .unwrap();
        let quote =
            quote_imports(&original, manifest.payload(), &checked, &public, &imports).unwrap();
        imports
            .resize_payload(quote.payload_bytes - usize::from(short))
            .unwrap();
        let result = create_inputs(
            original,
            manifest.payload(),
            &checked,
            Arc::clone(&public),
            imports,
        );
        if short {
            assert!(matches!(result, Err(Error::Bounds)));
        } else {
            let created = result.unwrap();
            assert_eq!(created.builder.original_inputs.len(), 1);
            assert_eq!(
                created.builder.nodes[0].lanes,
                manifest
                    .payload()
                    .tensors
                    .iter()
                    .find(|(_, v)| v.public_q18.is_none())
                    .unwrap()
                    .1
                    .shape
                    .iter()
                    .product::<usize>()
            );
            assert!(created.charge.is_for(root.budget()));
        }
    }
}

#[test]
fn cancellation_keeps_genuine_source_guard_until_the_last_original_worker() {
    let (_, raw, data) = fixture();
    let mut entry = SourceRequestImport::new(&raw).unwrap();
    let counter = entry.0.as_ref().unwrap().root.try_clone().unwrap();
    let empty = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    for (key, bytes) in &data {
        entry.append(key, 0, bytes).unwrap();
    }
    let request = entry.finish().unwrap();
    let program = Arc::clone(request.program());
    let session = [96; 32];
    let context = puma::Context::new(session, program.digest(), 0).unwrap();
    let runtimes = puma::prss::insecure_test_support::fresh_compound_parties(
        context,
        program.work_budget().limits().unwrap(),
    )
    .unwrap();
    let mut workers =
        runtimes.map(|runtime| program.source_request_worker(runtime, session).unwrap());
    drop(request);
    drop(program);
    for worker in &mut workers {
        worker.cancel();
        assert!(worker.retired());
    }
    assert!(counter.budget().used_bytes() > empty.budget().used_bytes());
    let [a, b, c] = workers;
    drop(a);
    drop(b);
    assert!(counter.budget().used_bytes() > empty.budget().used_bytes());
    drop(c);
    assert_eq!(counter.budget().used_bytes(), empty.budget().used_bytes());
}
