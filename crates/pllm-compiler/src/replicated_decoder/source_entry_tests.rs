use super::*;
use crate::replicated_decoder::{sdk::SourceRequestImport, Publisher};

#[test]
fn source_union_quote_matches_typed_manifest_and_adds_to_bootstrap() {
    use std::alloc::{Layout, LayoutError};
    use std::array::IntoIter;
    use std::sync::atomic::{AtomicUsize, Ordering};

    type Z = usize;
    type D = pllm_garble::shared_allocation::Error;
    type X = pllm_core::puma::Error;
    type B = pllm_garble::shared_allocation::Budget;
    type O = pllm_garble::shared_allocation::OwnedBudget;
    type R = pllm_garble::shared_allocation::OwnedBudgetResources;
    type G = pllm_garble::shared_allocation::OwnedExternalCharge;
    type I = SourceRequestImport;
    type V = &'static [u8];
    type H = Option<G>;
    type SV = Option<(V, V)>;

    // Literal recipe (24), independent of production K*/FIXED constants.
    // Arithmetic/API composition only: not census exhaustiveness, actual
    // compiler call binding, physical storage, work or full-resource funding.
    let fixed = [
        // Kctor.
        (13, size_of::<Z>()),
        (4, size_of::<bool>()),
        (2, size_of::<D>()),
        (2, size_of::<AtomicUsize>()),
        (4, size_of::<B>()),
        (3, size_of::<Result<B, D>>()),
        (6, size_of::<Result<O, D>>()),
        (2, size_of::<R>()),
        (1, size_of::<Result<R, D>>()),
        (4, size_of::<&B>()),
        (4, size_of::<Result<(), D>>()),
        (3, size_of::<()>()),
        (1, size_of::<Arc<B>>()),
        (1, size_of::<O>()),
        // Kused.
        (1, size_of::<&B>()),
        (1, size_of::<&AtomicUsize>()),
        (1, size_of::<Ordering>()),
        (2, size_of::<Z>()),
        // Kresources: 61 occurrences, including the actual (Layout, Z) order.
        (26, size_of::<Z>()),
        (3, size_of::<Layout>()),
        (3, size_of::<&Layout>()),
        (1, size_of::<(Layout, Z)>()),
        (1, size_of::<Result<(Layout, Z), LayoutError>>()),
        (1, size_of::<Result<(Layout, Z), D>>()),
        (1, size_of::<LayoutError>()),
        (1, size_of::<[Z; 7]>()),
        (1, size_of::<IntoIter<Z, 7>>()),
        (1, size_of::<&mut IntoIter<Z, 7>>()),
        (3, size_of::<Option<Z>>()),
        (3, size_of::<Result<Z, D>>()),
        (9, size_of::<D>()),
        (1, size_of::<R>()),
        (6, size_of::<Result<R, D>>()),
        // Four Kreserve_fixed instances; no guessed closure capture width.
        (4 * 19, size_of::<Z>()),
        (4, size_of::<&B>()),
        (4 * 2, size_of::<&AtomicUsize>()),
        (4 * 3, size_of::<Ordering>()),
        (4 * 5, size_of::<Option<Z>>()),
        (4, size_of::<bool>()),
        (4, size_of::<Result<Z, Z>>()),
        (4, size_of::<Result<Z, D>>()),
        (4 * 3, size_of::<D>()),
        (4 * 2, size_of::<Result<(), D>>()),
        (4, size_of::<()>()),
        // Two Kcharge instances.
        (2 * 7, size_of::<Z>()),
        (2, size_of::<&O>()),
        (2, size_of::<&B>()),
        (2, size_of::<&Arc<B>>()),
        (2, size_of::<Option<Z>>()),
        (2, size_of::<Result<Z, D>>()),
        (2 * 3, size_of::<D>()),
        (2, size_of::<Result<(), D>>()),
        (2, size_of::<()>()),
        (2, size_of::<Arc<B>>()),
        (2, size_of::<G>()),
        (2 * 4, size_of::<Result<G, D>>()),
        // Kcaller.
        (56, size_of::<Z>()),
        (1, size_of::<i32>()),
        (3, size_of::<bool>()),
        (3, size_of::<V>()),
        (1, size_of::<SV>()),
        (2, size_of::<&O>()),
        (2, size_of::<O>()),
        (3, size_of::<G>()),
        (4, size_of::<H>()),
        (1, size_of::<Result<O, D>>()),
        (1, size_of::<Result<O, X>>()),
        (2, size_of::<Result<G, D>>()),
        (2, size_of::<Result<G, X>>()),
        (5, size_of::<Result<I, X>>()),
        (10, size_of::<X>()),
        (3, size_of::<D>()),
        // DELTAcaller.
        (3, size_of::<Z>()),
        (1, size_of::<&O>()),
        (1, size_of::<()>()),
        (1, size_of::<Result<Z, D>>()),
        (1, size_of::<Result<Z, X>>()),
        (1, size_of::<Result<(), D>>()),
        (1, size_of::<Result<(), X>>()),
        (2, size_of::<Result<I, X>>()),
        (6, size_of::<X>()),
        (2, size_of::<D>()),
        // KAPI.
        (2, size_of::<Z>()),
        (1, size_of::<&O>()),
        (1, size_of::<&B>()),
        (2, size_of::<Result<(), D>>()),
        // Hfixed: 31Z/8Option includes the closure_bytes/reference_bytes
        // checked_add operand reads; these are SOURCE values, not padding.
        (31, size_of::<Z>()),
        (6, size_of::<(Z, Z)>()),
        (8, size_of::<Option<Z>>()),
        (10, size_of::<Result<Z, D>>()),
        (4, size_of::<D>()),
    ]
    .into_iter()
    .try_fold(0usize, |sum, (count, width)| {
        width.checked_mul(count).and_then(|term| sum.checked_add(term))
    })
    .unwrap();
    // Measure the actual opaque closure/reference through the public layout
    // API; do not assume two usize captures or invoke the update callback.
    let widths = R::reserve_update_layout();
    let expected = fixed
        .checked_add(16usize.checked_mul(widths.0).unwrap())
        .unwrap()
        .checked_add(2usize.checked_mul(widths.1).unwrap())
        .unwrap();
    let extra = source_union_quote().unwrap();
    assert_eq!(extra, expected);

    let resources = O::resources().unwrap();
    let baseline = resources.initial_reserved_bytes;
    let root = O::new(MAX_ALLOCATION_BYTES).unwrap();
    let identity = root.budget() as *const B;
    assert_eq!(root.budget().used_bytes(), baseline);
    root.reserve_bootstrap(extra).unwrap();
    let retained = baseline.checked_add(expected).unwrap();
    assert_eq!(root.budget().used_bytes(), retained);
    assert_eq!(root.budget().admitted_peak_bytes(), retained);
    let clone = root.try_clone().unwrap();
    assert_eq!(clone.budget() as *const B, identity);
    assert_eq!(clone.budget().used_bytes(), retained + size_of::<O>());
    drop(clone);
    assert_eq!(root.budget() as *const B, identity);
    assert_eq!(root.budget().used_bytes(), retained);
    assert_eq!(O::resources().unwrap(), resources);
    // No triggerable fixed-width production quote overflow is supplied here;
    // its refusal branches and pre-JSON call order remain SOURCE-only proof.
}

#[test]
fn source_prefix_length_refusals() {
    // Actual entry guards: no fixture, model construction or native execution.
    // These results alone do not prove that JSON was never reached; ordering
    // is a separate SOURCE inspection, not a factory-trap assertion.
    let oversized = vec![b'x'; (1 << 20) + 1];
    for raw in [&b""[..], oversized.as_slice()] {
        assert!(matches!(SourceRequestImport::new(raw), Err(Error::Bounds)));
        assert!(matches!(
            SourceRequestImport::new_checked_public(raw, b"x", b"x"),
            Err(Error::Bounds)
        ));
    }
    let experiment = &oversized[..(32 << 10) + 1];
    let config = &oversized[..super::source_policy::CONFIG_BYTES + 1];
    for (experiment, config) in [
        (&b""[..], &b"x"[..]),
        (experiment, &b"x"[..]),
        (&b"x"[..], &b""[..]),
        (&b"x"[..], config),
    ] {
        assert!(matches!(
            SourceRequestImport::new_checked_public(b"x", experiment, config),
            Err(Error::Bounds)
        ));
        assert!(matches!(
            SourceRequestImport::quote_checked_public(experiment, config, 1, 1),
            Err(Error::Bounds)
        ));
    }
}

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
fn pending_binding_refusal_retires_actual_slots_before_controller() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let (publisher, manifest_bytes, data) = fixture();
    let Pending {
        directory, importer, original, manifest, raw, limits, storage,
        registry, imports, controller, root,
    } = SourceRequestImport::new(&manifest_bytes).unwrap().0.take().unwrap();
    // Unrelated preparation owners retire before any marker is armed.
    drop(importer);
    drop(registry);
    drop(imports);
    drop(manifest);
    drop(publisher);
    drop(data);

    let mut changed: Manifest = serde_json::from_slice(&manifest_bytes).unwrap();
    check_source(&original, &changed).unwrap();
    changed.rows = changed.rows.checked_add(1).unwrap();
    assert!(matches!(check_source(&original, &changed), Err(Error::Binding)));
    // Inventory this actual DTO's backing, independently of the controller.
    // try_build adds its inline payload/control storage; no arbitrary guard
    // stands in for ownership of the changed Manifest.
    let backing = changed.schema.capacity()
        .checked_add(changed.config.capacity()).unwrap()
        .checked_add(source_registry::source_metadata_bytes(
            &changed.tensors, &changed.norms,
        ).unwrap()).unwrap();
    let manifest = root.charge_external(backing).unwrap()
        .try_build(|_| Ok(changed)).unwrap();
    drop(manifest_bytes);

    let budget_id = root.budget() as *const Budget as usize;
    assert_eq!(original.budget() as *const Budget as usize, budget_id);
    assert_eq!(raw.budget() as *const Budget as usize, budget_id);
    assert_eq!(manifest.budget() as *const Budget as usize, budget_id);
    assert!(controller.is_for(root.budget()));
    assert_eq!(controller.budget() as *const Budget as usize, budget_id);
    let controller_bytes = controller.charged_bytes();
    assert!(controller_bytes > 0);
    assert!(manifest.charged_bytes() >= backing + size_of::<Manifest>());
    assert!(directory.is_none());
    let counter = root.try_clone().unwrap();
    // Move the exact non-Clone controller, source and raw owners. Only this
    // counter clone and the trace remain outside the consuming entrance.
    let ingress = ingress::OwnedSourceIngress {
        directory, original, raw, limits, storage, controller, root,
    };
    assert_eq!(ingress.controller.charged_bytes(), controller_bytes);
    assert!(ingress.controller.is_for(counter.budget()));
    let trace = Arc::new(Trace {
        next: AtomicUsize::new(1),
        completed: std::array::from_fn(|_| AtomicUsize::new(0)),
        budget_id,
    });
    assert_eq!(trace.budget_id, counter.budget() as *const Budget as usize);
    assert!(matches!(
        ingress.into_pending_observed(manifest, &trace),
        Err(Error::Binding)
    ));
    let completed = trace.completed.each_ref().map(|event| event.load(Ordering::Relaxed));
    assert!(completed.iter().all(|&ordinal| ordinal != 0));
    let controller = completed[Event::ControllerCharge as usize];
    for event in [Event::ManifestSlot, Event::DirectorySlot, Event::OriginalSlot, Event::RawSlot] {
        assert!(completed[event as usize] < controller);
    }
    // Directory was None: its event certifies only Option-slot retirement,
    // never a directory pointee, compact custody or unwind completion.
    // Sensitivity (SOURCE only, no mutation run): early destructuring of this
    // SAME pending_transition body into directory/original/raw/limits/storage/
    // controller/root before checks reverses local retirement, placing controller
    // before raw/original on Binding. These actual-owner assertions would fail.
    drop(trace);
    drop(counter);
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
    // The fresh ingress retains E after ordinary owners retire; the clone
    // replaces the dropped root's inline metadata, not a second baseline.
    let retained = empty
        .budget()
        .used_bytes()
        .checked_add(source_union_quote().unwrap())
        .unwrap();
    assert!(counter.budget().used_bytes() > retained);
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
    assert_eq!(counter.budget().used_bytes(), retained);
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
    let retained = empty
        .budget()
        .used_bytes()
        .checked_add(source_union_quote().unwrap())
        .unwrap();
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
    assert!(counter.budget().used_bytes() > retained);
    let [a, b, c] = workers;
    drop(a);
    drop(b);
    assert!(counter.budget().used_bytes() > retained);
    drop(c);
    assert_eq!(counter.budget().used_bytes(), retained);
}
