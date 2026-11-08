//! Source-only regression authoring. UNRUN while native execution is on HOLD.
use super::*;
use crate::replicated_decoder::{sdk::SourceRequestImport, Publisher};
use request_origin::compact_storage::{ControllerAccounting, ControllerLoan, EmissionScope};

// Real source-factory owner, using the same tiny Publisher/config fixture as
// source_entry_tests. No substitute ledger, fake origin, or guard-only oracle.
fn fixture() -> Pending {
    let config = br#"{"model_type":"qwen2","hidden_size":2,"intermediate_size":2,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"vocab_size":3,"max_position_embeddings":16,"rms_norm_eps":0.000001,"hidden_act":"silu","tie_word_embeddings":false,"attention_bias":false,"rope_theta":10000.0}"#;
    let mut publisher = Publisher::new_with_policy(
        config, 4, 2, [31; 32], [32; 32], WeightMode::Public,
        ExecutionPolicy { range_policy: RangePolicy::Checked, max_abs_activation: 1,
            ..ExecutionPolicy::default() },
    ).unwrap();
    let data: Vec<_> = publisher.shapes().iter().map(|(key, shape)| {
        let value = if key.contains("norm") { 1f32 } else { 0f32 };
        (key.clone(), value.to_le_bytes().repeat(shape.iter().product()))
    }).collect();
    for (key, bytes) in &data { publisher.append(key, 0, bytes).unwrap(); }
    publisher.seal(&[0, 1, 0, 1]).unwrap();
    let mut entry = SourceRequestImport::new(&publisher.manifest().unwrap()).unwrap();
    for (key, bytes) in &data { entry.append(key, 0, bytes).unwrap(); }
    entry.0.take().unwrap()
}

// Observe the ORIGINAL shared cumulative ledger through its existing read-only
// bootstrap API. Only Error::Work denotes max_work <= actual ledger.used;
// positive remaining may instead return Quote. No shadow ledger or work debit.
fn work_used(original: &OriginalSourcePlan) -> u64 {
    let mut limits = source_limits();
    assert!(!matches!(original.lower_bootstrap_bytes(limits), Err(original_source::Error::Work | original_source::Error::Burned)));
    let mut low = 0;
    let mut high = limits.max_work;
    while low + 1 < high {
        let middle = low + (high - low) / 2;
        limits.max_work = middle;
        match original.lower_bootstrap_bytes(limits) {
            Err(original_source::Error::Work) => low = middle,
            Err(original_source::Error::Burned) => panic!("original ledger unexpectedly burned"),
            _ => high = middle,
        }
    }
    low
}

// Retain independent owner/counter handles before entering the receiver. The
// original controller stays outside its loan; no registry lens enters the receiver.
macro_rules! with_receiver {
    ($receiver:ident, $manifest:ident, $checked:ident, $root:ident, $body:block) => {{
        with_receiver!($receiver, $manifest, $checked, $root, controller_owner, $body);
    }};
    ($receiver:ident, $manifest:ident, $checked:ident, $root:ident, $owner:ident, $body:block) => {{
        with_receiver!($receiver, $manifest, $checked, $root, $owner,
            _controller_identity, _controller_bytes, $body);
    }};
    ($receiver:ident, $manifest:ident, $checked:ident, $root:ident, $owner:ident,
        $identity:ident, $charged:ident, $body:block) => {{
        let Pending { importer, original, manifest, mut imports, registry, root,
            controller: mut $owner, .. } = fixture();
        let public = Arc::new(importer.finish().unwrap());
        let owners = SourceConstructionOwners {
            manifest: manifest.payload(), incoming: None, public: Some(&public),
            retained_baseline_bytes: root.budget().used_bytes() - registry.payload_bytes(),
        };
        let $checked = SourceRegistry::from_original_checked(
            &original, &owners, &public, &registry, MAX_PROGRAM_BYTES,
        ).unwrap();
        let quote = quote_imports(&original, manifest.payload(), &$checked, &public, &imports).unwrap();
        imports.resize_payload(quote.payload_bytes).unwrap();
        let $identity = $owner.controller_identity();
        let $charged = $owner.charged_bytes();
        let mut $receiver = InputReceiver::new(original, Arc::clone(&public), imports,
            ControllerLoan::new(&mut $owner), MAX_PROGRAM_BYTES);
        let $manifest = manifest.payload();
        let $root = &root;
        $body
    }};
}

// Both genuine controller owners precede either invariant receiver loan.
macro_rules! with_receivers {
    ($receiver_a:ident, $manifest_a:ident, $checked_a:ident, $root_a:ident,
        $receiver_b:ident, $manifest_b:ident, $checked_b:ident, $root_b:ident,
        $body:block) => {{
        let Pending { importer: importer_a, original: original_a, manifest: manifest_a,
            imports: mut imports_a, registry: registry_a, root: root_a,
            controller: mut controller_a, .. } = fixture();
        let public_a = Arc::new(importer_a.finish().unwrap());
        let owners_a = SourceConstructionOwners {
            manifest: manifest_a.payload(), incoming: None, public: Some(&public_a),
            retained_baseline_bytes: root_a.budget().used_bytes() - registry_a.payload_bytes(),
        };
        let $checked_a = SourceRegistry::from_original_checked(
            &original_a, &owners_a, &public_a, &registry_a, MAX_PROGRAM_BYTES,
        ).unwrap();
        let quote_a = quote_imports(&original_a, manifest_a.payload(), &$checked_a, &public_a, &imports_a).unwrap();
        imports_a.resize_payload(quote_a.payload_bytes).unwrap();
        let _controller_identity_a = controller_a.controller_identity();
        let _controller_bytes_a = controller_a.charged_bytes();
        let Pending { importer: importer_b, original: original_b, manifest: manifest_b,
            imports: mut imports_b, registry: registry_b, root: root_b,
            controller: mut controller_b, .. } = fixture();
        let public_b = Arc::new(importer_b.finish().unwrap());
        let owners_b = SourceConstructionOwners {
            manifest: manifest_b.payload(), incoming: None, public: Some(&public_b),
            retained_baseline_bytes: root_b.budget().used_bytes() - registry_b.payload_bytes(),
        };
        let $checked_b = SourceRegistry::from_original_checked(
            &original_b, &owners_b, &public_b, &registry_b, MAX_PROGRAM_BYTES,
        ).unwrap();
        let quote_b = quote_imports(&original_b, manifest_b.payload(), &$checked_b, &public_b, &imports_b).unwrap();
        imports_b.resize_payload(quote_b.payload_bytes).unwrap();
        let _controller_identity_b = controller_b.controller_identity();
        let _controller_bytes_b = controller_b.charged_bytes();
        {
            let mut $receiver_a = InputReceiver::new(original_a, Arc::clone(&public_a), imports_a,
                ControllerLoan::new(&mut controller_a), MAX_PROGRAM_BYTES);
            let mut $receiver_b = InputReceiver::new(original_b, Arc::clone(&public_b), imports_b,
                ControllerLoan::new(&mut controller_b), MAX_PROGRAM_BYTES);
            let $manifest_a = manifest_a.payload();
            let $root_a = &root_a;
            let $manifest_b = manifest_b.payload();
            let $root_b = &root_b;
            $body
        }
    }};
}

fn assert_sticky(
    receiver: &mut InputReceiver<'_>,
    manifest: &Manifest,
    checked: &SourceRegistry<'_>,
) {
    assert_eq!(receiver.state, ReceiverState::CleanupOnly);
    let backing = (receiver.builder.as_ref().unwrap().nodes.as_ptr(), receiver.builder.as_ref().unwrap().nodes.capacity());
    let declaration = receiver.declaration.test_backings();
    let used = receiver.original.budget().used_bytes();
    let work = work_used(&receiver.original);
    assert!(matches!(prepare_resident_inputs(receiver, manifest, checked), Err(Error::Retired)));
    assert_eq!((receiver.builder.as_ref().unwrap().nodes.as_ptr(), receiver.builder.as_ref().unwrap().nodes.capacity()), backing);
    assert_eq!(receiver.declaration.test_backings(), declaration);
    assert_eq!(receiver.original.budget().used_bytes(), used);
    assert_eq!(work_used(&receiver.original), work);
}

#[test]
fn resident_returned_refusal_after_public_node_retains_pending_copies_and_original_debit() {
    with_receiver!(receiver, manifest, checked, root, {
        let original = Arc::clone(&receiver.original);
        let before = work_used(&original);
        let expected = quote_imports(&original, manifest, &checked, &receiver.public, receiver.imports.as_ref().unwrap()).unwrap().metadata_work;
        let controller = receiver.controller.as_ref().unwrap().borrow().charged_bytes();
        RECEIVER_FAULT.with(|fault| fault.set(Some((1, false))));
        assert!(prepare_resident_inputs(&mut receiver, manifest, &checked).is_err());
        assert_eq!(receiver.builder.as_ref().unwrap().nodes.len(), 1);
        assert!(receiver.inputs.as_ref().unwrap().is_empty());
        assert_eq!(receiver.pending_map.owner, Some(0));
        let key = receiver.pending_map.key.as_ref().unwrap();
        assert_eq!(receiver.pending_map.shape, manifest.tensors[key].shape);
        assert_ne!(receiver.pending_map.shape.as_ptr(), manifest.tensors[key].shape.as_ptr());
        assert!(matches!(&receiver.builder.as_ref().unwrap().nodes[0].code, Code::PublicInput { key: node_key, .. } if node_key == key && node_key.as_ptr() != key.as_ptr()));
        assert!(receiver.original.same_origin(&original));
        assert!(std::ptr::eq(original.budget(), root.budget()));
        assert!(receiver.imports.as_ref().unwrap().is_for(root.budget()) && receiver.controller.as_ref().unwrap().borrow().is_for(root.budget()));
        assert_eq!(receiver.controller.as_ref().unwrap().borrow().charged_bytes(), controller);
        assert_eq!(work_used(&original) - before, expected);
        assert_sticky(&mut receiver, manifest, &checked);
        let used = root.budget().used_bytes();
        let work = work_used(&original);
        let mut src = Some(receiver);
        let mut dst = None;
        assert!(transfer_complete(&mut src, &mut dst).is_err());
        assert!(src.is_some() && dst.is_none());
        assert_eq!(root.budget().used_bytes(), used);
        assert_eq!(work_used(&original), work);
    });
}

#[test]
fn resident_post_reserve_exact_cap_refusal_retains_enlarged_backing() {
    with_receiver!(receiver, manifest, checked, root, {
        let before = work_used(&receiver.original);
        Words::test_excess_next_reserve();
        assert!(prepare_resident_inputs(&mut receiver, manifest, &checked).is_err());
        let retained = receiver.declaration.test_backings();
        assert_eq!(Some(retained[3]), Words::test_last_reserve());
        assert!(retained[3].1 > manifest.outputs + 1);
        assert!(retained[0].1 > 0 && retained[1].1 > 0);
        assert!(receiver.imports.as_ref().unwrap().is_for(root.budget()));
        assert!(work_used(&receiver.original) > before);
        assert_sticky(&mut receiver, manifest, &checked);
    });
}

#[test]
fn resident_allocator_refusal_without_new_backing_is_distinct() {
    with_receiver!(receiver, manifest, checked, root, {
        let original = Arc::clone(&receiver.original);
        let counter = root.budget() as *const _ as usize;
        let before = root.budget().used_bytes();
        let work_before = work_used(&original);
        let import_work = quote_imports(&original, manifest, &checked, &receiver.public, receiver.imports.as_ref().unwrap())
            .unwrap().metadata_work;
        let imports_before = receiver.imports.as_ref().unwrap().charged_bytes();
        let controller_before = receiver.controller.as_ref().unwrap().borrow().charged_bytes();
        receiver.declaration.test_seed_fragment_backing().unwrap();
        let destination_before = receiver.declaration.test_backings()[1];
        assert_eq!(destination_before.1, 1);
        request_origin::DeclarationReceiver::test_refuse_next_fragment_reserve();
        assert!(matches!(prepare_resident_inputs(&mut receiver, manifest, &checked), Err(Error::Allocation)));
        let retained = receiver.declaration.test_backings();
        let at_reserve = request_origin::DeclarationReceiver::test_last_backings().unwrap();
        assert_eq!(retained, at_reserve);
        assert_eq!(retained[1], destination_before); // no new fragment backing
        assert!(retained[0].1 > 0); // successful preceding epoch allocation retained
        assert_eq!(receiver.builder.as_ref().unwrap().nodes.len(), receiver.inputs.as_ref().unwrap().len());
        assert!(!receiver.inputs.as_ref().unwrap().is_empty());
        assert!(receiver.builder.as_ref().unwrap().original_inputs.is_empty());
        assert!(Arc::ptr_eq(&receiver.original, &original));
        assert_eq!(receiver.original.budget() as *const _ as usize, counter);
        assert!(receiver.imports.as_ref().unwrap().is_for(root.budget()) && receiver.controller.as_ref().unwrap().borrow().is_for(root.budget()));
        assert_eq!(receiver.controller.as_ref().unwrap().borrow().charged_bytes(), controller_before);
        let (declaration, origin) = receiver.declaration.test_covers();
        let (declaration_bytes, declaration_counter) = declaration.unwrap();
        assert_eq!(declaration_counter, counter);
        assert!(origin.is_none()); // genuine root remains in admission, not minted
        let (origin_size, _) = request_origin::DeclarationReceiver::test_origin_layout();
        let root_cover = origin_size + 2 * std::mem::size_of::<usize>()
            + std::mem::size_of::<OwnedExternalCharge>();
        assert_eq!(receiver.imports.as_ref().unwrap().charged_bytes() + declaration_bytes + root_cover, imports_before);
        assert_eq!(root.budget().used_bytes(), before); // split only; no inline debit yet
        let sources = request_origin::RequestSources::new(manifest, &checked, Arc::clone(&receiver.public)).unwrap();
        let declaration_work = request_origin::quote(receiver.builder.as_ref().unwrap(), &original, &sources).unwrap().metadata_work;
        assert_eq!(work_used(&original), work_before + import_work + declaration_work);
        assert_sticky(&mut receiver, manifest, &checked);
    });
}

#[test]
fn resident_panic_after_second_declaration_allocation_retains_both_backings() {
    with_receiver!(receiver, manifest, checked, root, {
        let original = Arc::clone(&receiver.original);
        let before = work_used(&original);
        RECEIVER_FAULT.with(|fault| fault.set(Some((2, true))));
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            prepare_resident_inputs(&mut receiver, manifest, &checked)
        })).is_err());
        let retained = receiver.declaration.test_backings();
        assert_eq!(Some(retained), request_origin::DeclarationReceiver::test_last_backings());
        assert!(retained[0].1 > 0 && retained[1].1 > 0);
        assert_eq!(receiver.builder.as_ref().unwrap().original_inputs.len(), 0);
        assert!(std::ptr::eq(original.budget(), root.budget()));
        assert!(work_used(&original) > before);
        assert_sticky(&mut receiver, manifest, &checked);
    });
}

#[test]
fn resident_panic_after_genuine_token_mint_retains_original_record_and_backings() {
    with_receiver!(receiver, manifest, checked, root, {
        let original = Arc::clone(&receiver.original);
        let before = work_used(&original);
        RECEIVER_FAULT.with(|fault| fault.set(Some((3, true))));
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            prepare_resident_inputs(&mut receiver, manifest, &checked)
        })).is_err());
        assert_eq!(receiver.builder.as_ref().unwrap().original_inputs.len(), 1);
        assert!(receiver.declaration.test_genuine_record(receiver.builder.as_ref().unwrap()));
        assert_eq!(Some(receiver.declaration.test_backings()), request_origin::DeclarationReceiver::test_last_backings());
        assert!(std::ptr::eq(original.budget(), root.budget()));
        assert!(work_used(&original) > before);
        assert_sticky(&mut receiver, manifest, &checked);
    });
}

#[test]
fn resident_initial_validation_refusal_retains_sealed_origin_and_original_charges() {
    with_receiver!(receiver, manifest, checked, root, {
        // A real post-declaration admission refusal, not a fake validator.
        receiver.setup = Some(request_origin::compact_storage::ResidentSetup::empty(0));
        let original = Arc::clone(&receiver.original);
        let before = work_used(&original);
        assert!(matches!(prepare_resident_inputs(&mut receiver, manifest, &checked), Err(Error::Bounds)));
        assert!(receiver.declaration.test_genuine_record(receiver.builder.as_ref().unwrap()));
        assert!(receiver.controller.as_ref().unwrap().borrow().is_for(root.budget()));
        assert!(work_used(&original) > before);
        assert_sticky(&mut receiver, manifest, &checked);
    });
}

#[test]
fn resident_occupied_destination_preserves_both_owners_backings_and_debit() {
    with_receivers!(receiver, manifest, checked, root,
        occupied, other_manifest, other_checked, other_root, {
        prepare_resident_inputs(&mut receiver, manifest, &checked).unwrap();
        let original = Arc::clone(&receiver.original);
        let backing = (receiver.builder.as_ref().unwrap().nodes.as_ptr(), receiver.builder.as_ref().unwrap().nodes.capacity());
        prepare_resident_inputs(&mut occupied, other_manifest, &other_checked).unwrap();
        let other_original = Arc::clone(&occupied.original);
        assert!(!original.same_origin(&other_original));
        let other_backing = (occupied.builder.as_ref().unwrap().nodes.as_ptr(), occupied.builder.as_ref().unwrap().nodes.capacity());
        let mut src = Some(receiver);
        let mut dst = Some(occupied);
        let debit = (root.budget().used_bytes(), other_root.budget().used_bytes());
        let work = (work_used(&original), work_used(&other_original));
        assert!(transfer_complete(&mut src, &mut dst).is_err());
        assert!(Arc::ptr_eq(&src.as_ref().unwrap().original, &original));
        assert!(Arc::ptr_eq(&dst.as_ref().unwrap().original, &other_original));
        assert_eq!((src.as_ref().unwrap().builder.as_ref().unwrap().nodes.as_ptr(), src.as_ref().unwrap().builder.as_ref().unwrap().nodes.capacity()), backing);
        assert_eq!((dst.as_ref().unwrap().builder.as_ref().unwrap().nodes.as_ptr(), dst.as_ref().unwrap().builder.as_ref().unwrap().nodes.capacity()), other_backing);
        assert_eq!((root.budget().used_bytes(), other_root.budget().used_bytes()), debit);
        assert_eq!((work_used(&original), work_used(&other_original)), work);
        drop(src);
    });
}

#[test]
fn resident_successful_one_shot_setup_transfers_after_lexical_lens_ends() {
    with_receiver!(receiver, manifest, checked, root, {
        let original = Arc::clone(&receiver.original);
        receiver.setup.as_ref().unwrap().test_seed_work();
        let accumulator = receiver.setup.as_ref().unwrap().test_work();
        {
            prepare_resident_inputs(&mut receiver, manifest, &checked).unwrap();
        }
        assert_eq!(receiver.state, ReceiverState::Complete);
        assert_eq!(receiver.setup.as_ref().unwrap().test_work(), accumulator);
        assert!(receiver.declaration.test_genuine_record(receiver.builder.as_ref().unwrap()));
        let backing = (receiver.builder.as_ref().unwrap().nodes.as_ptr(), receiver.builder.as_ref().unwrap().nodes.capacity());
        let declaration = receiver.declaration.test_backings();
        let debit = root.budget().used_bytes();
        let work = work_used(&original);
        assert!(matches!(prepare_resident_inputs(&mut receiver, manifest, &checked), Err(Error::Retired)));
        drop(checked); // receiver contains no registry/manifest borrow.
        let mut src = Some(receiver);
        let mut dst = None;
        transfer_complete(&mut src, &mut dst).unwrap();
        assert!(src.is_none());
        let moved = dst.as_ref().unwrap();
        assert!(Arc::ptr_eq(&moved.original, &original));
        assert_eq!((moved.builder.as_ref().unwrap().nodes.as_ptr(), moved.builder.as_ref().unwrap().nodes.capacity()), backing);
        assert_eq!(moved.declaration.test_backings(), declaration);
        assert_eq!(moved.setup.as_ref().unwrap().test_work(), accumulator);
        assert_eq!(root.budget().used_bytes(), debit);
        assert_eq!(work_used(&original), work);
        assert!(transfer_complete(&mut src, &mut None).is_err());
    });
}

#[test]
fn resident_prepare_refusal_drains_neither_receiver() {
    with_receiver!(receiver, manifest, checked, root, {
        RECEIVER_FAULT.with(|fault| fault.set(Some((1, false))));
        assert!(matches!(receiver.prepare(manifest, &checked), Err(Error::Binding)));
        let before = (receiver.builder.as_ref().unwrap().nodes.as_ptr(), receiver.declaration.test_backings(),
            root.budget().used_bytes(), work_used(&receiver.original));
        let mut sources = Some(request_origin::RequestSources::new(manifest, &checked, Arc::clone(&receiver.public)).unwrap());
        let mut destination = None;
        assert!(matches!(receiver.drain_complete(&mut sources, &mut destination), Err(Error::Binding)));
        assert!(sources.is_some() && destination.is_none());
        assert_eq!(receiver.state, ReceiverState::CleanupOnly);
        assert_eq!((receiver.builder.as_ref().unwrap().nodes.as_ptr(), receiver.declaration.test_backings(),
            root.budget().used_bytes(), work_used(&receiver.original)), before);
        assert!(receiver.setup.is_some() && receiver.imports.is_some() && receiver.controller.is_some());
        assert!(matches!(receiver.prepare(manifest, &checked), Err(Error::Retired)));
    });
}

#[test]
fn resident_unresolved_genuine_declaration_refuses_before_extraction() {
    with_receiver!(receiver, manifest, checked, root, {
        RECEIVER_FAULT.with(|fault| fault.set(Some((3, true))));
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| receiver.prepare(manifest, &checked))).is_err());
        assert!(receiver.declaration.test_genuine_record(receiver.builder.as_ref().unwrap()));
        let before = (receiver.builder.as_ref().unwrap().nodes.as_ptr(), receiver.declaration.test_backings(),
            root.budget().used_bytes(), work_used(&receiver.original));
        let mut sources = Some(request_origin::RequestSources::new(manifest, &checked, Arc::clone(&receiver.public)).unwrap());
        let mut destination = None;
        assert!(matches!(receiver.drain_complete(&mut sources, &mut destination), Err(Error::Binding)));
        assert!(sources.is_some() && destination.is_none());
        assert_eq!((receiver.builder.as_ref().unwrap().nodes.as_ptr(), receiver.declaration.test_backings(),
            root.budget().used_bytes(), work_used(&receiver.original)), before);
        assert_eq!(receiver.state, ReceiverState::CleanupOnly);
    });
}

#[test]
fn resident_occupied_destination_preserves_both_genuine_payloads() {
    with_receivers!(a, manifest_a, checked_a, root_a,
        b, manifest_b, checked_b, root_b, {
        a.prepare(manifest_a, &checked_a).unwrap();
        b.prepare(manifest_b, &checked_b).unwrap();
        let backing_a = (a.builder.as_ref().unwrap().nodes.as_ptr(), a.declaration.test_backings());
        let backing_b = (b.builder.as_ref().unwrap().nodes.as_ptr() as usize, b.builder.as_ref().unwrap().nodes.capacity());
        let mut sources_a = Some(request_origin::RequestSources::new(manifest_a, &checked_a, Arc::clone(&a.public)).unwrap());
        let mut sources_b = Some(request_origin::RequestSources::new(manifest_b, &checked_b, Arc::clone(&b.public)).unwrap());
        let mut destination = None;
        b.drain_complete(&mut sources_b, &mut destination).unwrap();
        let before = (root_a.budget().used_bytes(), root_b.budget().used_bytes(), work_used(&a.original), work_used(&b.original));
        let mints = request_origin::DeclarationReceiver::test_mints();
        assert!(matches!(a.drain_complete(&mut sources_a, &mut destination), Err(Error::Binding)));
        assert_eq!(destination.as_ref().unwrap().test_builder_backing(), backing_b);
        assert_eq!((a.builder.as_ref().unwrap().nodes.as_ptr(), a.declaration.test_backings()), backing_a);
        assert!(sources_a.is_some() && sources_b.is_none());
        assert_eq!(a.state, ReceiverState::CleanupOnly);
        assert_eq!((root_a.budget().used_bytes(), root_b.budget().used_bytes(), work_used(&a.original), work_used(&b.original)), before);
        assert_eq!(request_origin::DeclarationReceiver::test_mints(), mints);
        drop(destination);
        drop(a);
        drop(b);
        // These are distinct genuine roots, not a same-root exclusivity test.
    });
}

#[test]
fn resident_complete_move_preserves_armed_origin_backings_and_seeded_setup() {
    with_receiver!(receiver, manifest, checked, root, owner, identity, controller_bytes, {
        receiver.setup.as_ref().unwrap().test_seed_work();
        receiver.prepare(manifest, &checked).unwrap();
        let setup = receiver.setup.as_ref().unwrap().test_transport_snapshot();
        let backing = (receiver.builder.as_ref().unwrap().nodes.as_ptr() as usize, receiver.builder.as_ref().unwrap().nodes.capacity());
        let declarations = receiver.declaration.test_backings();
        let (origin, state) = receiver.declaration.test_origin_probe();
        let plan = Arc::as_ptr(&receiver.original) as usize;
        let public = Arc::as_ptr(&receiver.public) as usize;
        let work = work_used(&receiver.original);
        let used = root.budget().used_bytes();
        let mints = request_origin::DeclarationReceiver::test_mints();
        let mut sources = Some(request_origin::RequestSources::new(manifest, &checked, Arc::clone(&receiver.public)).unwrap());
        let mut destination = None;
        receiver.drain_complete(&mut sources, &mut destination).unwrap();
        assert_eq!(receiver.state, ReceiverState::Drained);
        assert!(receiver.builder.is_none() && receiver.inputs.is_none() && receiver.setup.is_none()
            && receiver.imports.is_none() && receiver.controller.is_none() && sources.is_none());
        assert_eq!(destination.as_ref().unwrap().test_builder_backing(), backing);
        assert_eq!(destination.as_ref().unwrap().test_declaration_backings(), declarations);
        assert_eq!(destination.as_ref().unwrap().test_transport_snapshot(), setup);
        assert_eq!(state(), (true, false));
        assert_eq!(root.budget().used_bytes(), used);
        assert_eq!(work_used(&receiver.original), work);
        assert_eq!(request_origin::DeclarationReceiver::test_mints(), mints);
        let mut empty = None;
        assert!(matches!(receiver.drain_complete(&mut sources, &mut empty), Err(Error::Binding)));
        assert!(sources.is_none() && empty.is_none());
        drop(empty);
        let mut scope = EmissionScope::from_resident(destination.take().unwrap());
        {
            let (_, context, _) = scope.parts();
            context.check_open().unwrap();
            assert_eq!(context.test_transport_snapshot(), setup);
            assert_eq!(context.test_source_identities(), (plan, public, origin));
            assert_eq!(context.controller_identity(), identity);
            assert_eq!(context.controller_charged_bytes(), controller_bytes);
        }
        assert_eq!(root.budget().used_bytes(), used);
        assert_eq!(work_used(&receiver.original), work);
        drop(receiver);
        assert_eq!(state(), (true, false));
        drop(scope);
        assert_eq!(state(), (false, true));
        drop(destination);
        assert_eq!(owner.controller_identity(), identity);
        assert_eq!(owner.charged_bytes(), controller_bytes);
        // The observer retains an origin Arc; this is not allocator cleanup proof.
    });
}

#[test]
fn resident_drop_destination_burns_only_the_moved_genuine_origin() {
    with_receiver!(receiver, manifest, checked, root, owner, {
        receiver.prepare(manifest, &checked).unwrap();
        let (origin, state) = receiver.declaration.test_origin_probe();
        assert_ne!(origin, 0);
        let mut sources = Some(request_origin::RequestSources::new(manifest, &checked, Arc::clone(&receiver.public)).unwrap());
        let mut destination = None;
        receiver.drain_complete(&mut sources, &mut destination).unwrap();
        drop(receiver);
        assert_eq!(state(), (true, false));
        drop(destination);
        assert_eq!(state(), (false, true));
        assert!(owner.is_for(root.budget()));
    });
}

#[test]
fn resident_scope_local_unwind_burns_origin_before_outer_controller_retires() {
    with_receiver!(receiver, manifest, checked, root, owner, identity, charged, {
        receiver.prepare(manifest, &checked).unwrap();
        let (_, state) = receiver.declaration.test_origin_probe();
        let mut sources = Some(request_origin::RequestSources::new(manifest, &checked, Arc::clone(&receiver.public)).unwrap());
        let mut destination = None;
        receiver.drain_complete(&mut sources, &mut destination).unwrap();
        drop(receiver);
        let resident = destination.take().unwrap();
        drop(destination);
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _scope = EmissionScope::from_resident(resident);
            panic!("ordinary local unwind after resident scope entry");
        })).is_err());
        assert_eq!(state(), (false, true));
        assert_eq!(owner.controller_identity(), identity);
        assert_eq!(owner.charged_bytes(), charged);
        assert!(owner.is_for(root.budget()));
        assert!(root.budget().used_bytes() >= charged);
        // No destructor-panic, Root/seal or physical cleanup conclusion follows.
    });
}

#[test]
fn retained_push_refusal_keeps_code_backing_and_legacy_accounting_order() {
    let mut retained = Builder {
        allocated: MAX_PROGRAM_BYTES, requested_bytes: Some(7),
        value_payload_bytes: 48, ..Builder::default()
    };
    let mut pending = Some(Code::Input("resident.test".into()));
    let backing = match pending.as_ref().unwrap() {
        Code::Input(key) => (key.as_ptr(), key.capacity()),
        _ => unreachable!(),
    };
    assert_eq!(backing.1, 13);
    let bounds = Interval::new(0, 1).unwrap();
    // Two simultaneous invalid conditions: zero lanes AND exhausted byte cap.
    // Historical contract rejects lanes before attempting any byte debit.
    assert!(matches!(retained.push_retained(&mut pending, 0, bounds), Err(Error::Bounds)));
    assert_eq!(retained.requested_lanes, Some(0));
    assert_eq!(retained.admission_limit, Some("values"));
    assert_eq!(retained.requested_bytes, Some(7));
    assert_eq!(retained.allocated, MAX_PROGRAM_BYTES);
    assert_eq!(retained.value_payload_bytes, 48);
    assert_eq!(retained.nodes.len(), 0);
    assert!(matches!(pending.as_ref(), Some(Code::Input(key)) if (key.as_ptr(), key.capacity()) == backing));

    // One-lane Input: 13 key bytes, one Node, 16 value bytes. Literal contract
    // expectations, independent of push/charge helpers and legacy delegation.
    assert!(matches!(retained.push_retained(&mut pending, 1, bounds), Err(Error::Bounds)));
    assert!(matches!(pending.as_ref(), Some(Code::Input(key)) if (key.as_ptr(), key.capacity()) == backing));
    assert_eq!(retained.allocated, MAX_PROGRAM_BYTES);
    assert_eq!(retained.requested_bytes, Some(MAX_PROGRAM_BYTES + std::mem::size_of::<Node>() + 29));
    assert_eq!(retained.requested_lanes, Some(1));
    assert_eq!(retained.admission_limit, Some("program_bytes"));
    assert_eq!(retained.value_payload_bytes, 48); // refused before subtotal debit
    assert_eq!(retained.nodes.len(), 0);
}

#[test]
fn resident_drop_destroys_actual_backings_before_covering_charge_release() {
    with_receiver!(receiver, manifest, checked, root, controller_owner, {
        // Keep genuine source/public owners alive independently of receiver drop.
        let original = Arc::clone(&receiver.original);
        let counter = root.budget() as *const _ as usize;
        let initial = root.budget().used_bytes();
        let initial_imports = receiver.imports.as_ref().unwrap().charged_bytes();
        let controller = receiver.controller.as_ref().unwrap().borrow().charged_bytes();
        prepare_resident_inputs(&mut receiver, manifest, &checked).unwrap();
        assert!(receiver.builder.as_ref().unwrap().nodes.capacity() > 0);
        assert!(receiver.declaration.test_backings().iter().all(|(_, capacity)| *capacity > 0));
        let (origin_size, wrapper_size) = request_origin::DeclarationReceiver::test_origin_layout();
        let guard_size = std::mem::size_of::<OwnedExternalCharge>();
        let root_backing = origin_size + 2 * std::mem::size_of::<usize>();
        let origin_cover = root_backing + wrapper_size;
        let inline_debit = wrapper_size - guard_size;
        let imports = receiver.imports.as_ref().unwrap().charged_bytes();
        let declaration_cover = initial_imports - imports - root_backing - guard_size;
        let expected_covers = (Some((declaration_cover, counter)), Some((origin_cover, counter)));
        assert_eq!(receiver.declaration.test_covers(), expected_covers);
        assert!(receiver.imports.as_ref().unwrap().is_for(root.budget()) && receiver.controller.as_ref().unwrap().borrow().is_for(root.budget()));
        assert_eq!(receiver.controller.as_ref().unwrap().borrow().charged_bytes(), controller);
        let before = root.budget().used_bytes();
        assert_eq!(before, initial + inline_debit);
        // Drain stale observations; the first retirement must survive the second
        // idempotent call from DeclarationReceiver's automatic field Drop.
        let _ = request_origin::DeclarationReceiver::test_drop_record();
        RECEIVER_DROP.with(|record| record.set(None));
        drop(receiver);
        let declaration = request_origin::DeclarationReceiver::test_drop_record().unwrap();
        assert!(declaration.backings_gone);
        assert_eq!(declaration.covers, expected_covers); // both specific owners still resident
        assert_eq!(declaration.before_origin, Some(before));
        assert_eq!(declaration.after_origin, Some(before - origin_cover));
        let (gone, during, import_cover, controller_cover) = RECEIVER_DROP.with(|record| record.get().unwrap());
        assert!(gone);
        assert_eq!(import_cover, (imports, counter));
        assert_eq!(controller_cover, (controller, counter));
        assert_eq!(during, before - origin_cover);
        // Receiver cleanup ends only the loan. The original controller retires
        // separately, after its dependent payloads and import coverage are gone.
        assert_eq!(root.budget().used_bytes(), during - declaration_cover - imports);
        assert_eq!(controller_owner.charged_bytes(), controller);
        drop(controller_owner);
        assert_eq!(root.budget().used_bytes(), during - declaration_cover - imports - controller);
        assert_eq!(root.budget().used_bytes(), initial - initial_imports - controller);
        assert_eq!(original.budget() as *const _ as usize, counter);
    });
}

#[test]
fn resident_root_inspection_uses_same_source_owner() {
    use compact_lowering::frozen_region::owning::{HistoryInspection, InspectionLoan};
    use handles::SourceAccess;
    use request_origin::tests::schedule_tests::compact_request_storage_tests::importer_geometry_with_original_row;

    let (entry, _) = importer_geometry_with_original_row(3, true, 2, true);
    let limits = entry.0.as_ref().unwrap().limits;
    let mut public_owner = None;
    let custody: sealed::ResidentRoot = entry.consume_resident(|resident, registry, _, source| {
        public_owner = registry.records().next().map(|(owner, _, _, _)| owner);
        emit_resident_compact(resident, source, limits)
    }).unwrap();
    let public_owner = public_owner.expect("genuine file-backed public import");
    assert_eq!(custody.epochs().len(), 2);
    assert!(std::ptr::eq(custody.source_plan().budget(), custody.root.budget()));
    let before = work_used(&custody.original);
    for index in 0..2 {
        let loan: InspectionLoan<'_, sealed::ResidentRoot> =
            HistoryInspection::inspect_history_epoch(&custody, index).unwrap();
        let (key, source) = loan.checked_import_source(public_owner).unwrap();
        let retained = custody.resolve_file_import(key).unwrap();
        assert!(std::ptr::eq(source, retained));
    }
    assert!(work_used(&custody.original) > before);
    custody.check_open().unwrap();

    // Use an independent genuine history for the out-of-range boundary.
    let (entry, _) = importer_geometry_with_original_row(3, true, 2, true);
    let limits = entry.0.as_ref().unwrap().limits;
    let custody: sealed::ResidentRoot = entry.consume_resident(|resident, _, _, source| {
        emit_resident_compact(resident, source, limits)
    }).unwrap();
    let before = work_used(&custody.original);
    let controller = custody.controller.charged_bytes();
    let used = custody.root.budget().used_bytes();
    assert!(matches!(
        HistoryInspection::inspect_history_epoch(&custody, custody.epochs().len()),
        Err(Error::Binding)
    ));
    // Ingress checks charge the original catalog traversal before epoch lookup.
    // No InspectionAttempt is armed on this path, so the genuine origin stays open.
    assert!(work_used(&custody.original) > before);
    assert_eq!(custody.controller.charged_bytes(), controller);
    assert_eq!(custody.root.budget().used_bytes(), used);
    custody.check_open().unwrap();
}

fn genuine_resident_root_fixture() -> (sealed::ResidentRoot, compact::TestOriginProbe) {
    use request_origin::tests::schedule_tests::compact_request_storage_tests::importer_geometry_with_original_row;

    let (entry, _) = importer_geometry_with_original_row(3, true, 2, true);
    let limits = entry.0.as_ref().unwrap().limits;
    let mut probe = None;
    let stage = std::cell::Cell::new("before callback");
    let custody: sealed::ResidentRoot = entry.consume_resident(|resident, _, _, source| {
        stage.set("callback entry");
        let mut scope = compact::EmissionScope::from_resident(resident);
        {
            let (_, context, _) = scope.parts();
            probe = Some(context.test_origin_probe());
        }
        stage.set("emitting");
        let (history, second) = emit_compact_epochs(scope, source, limits)?;
        stage.set("finish_detached");
        let produced = history.finish_detached(second)?;
        stage.set("callback success / after callback");
        Ok(produced)
    }).unwrap_or_else(|error| panic!("genuine resident fixture stage {}: {:?}", stage.get(), error));
    let probe = probe.expect("genuine resident emission entered");
    assert_ne!(probe.identity(), 0);
    assert_eq!(probe.state(), (true, false));
    (custody, probe)
}

#[test]
fn resident_and_legacy_custody_accessors_are_disjoint() {
    use request_origin::tests::schedule_tests::compact_request_storage_tests::genuine_legacy_root_fixture;
    use sealed::{CustodyOwner, LegacyRoot, ResidentRoot};

    let (resident, _probe) = genuine_resident_root_fixture();
    let legacy = genuine_legacy_root_fixture();
    let mut resident = <ResidentRoot as CustodyOwner>::into_custody(resident.seal_compact().unwrap());
    let mut legacy = <LegacyRoot as CustodyOwner>::into_custody(legacy.seal_compact().unwrap());

    assert!(resident.resident_compact().is_some());
    assert!(resident.compact().is_none());
    assert!(legacy.compact().is_some());
    assert!(legacy.resident_compact().is_none());
    assert!(<ResidentRoot as CustodyOwner>::installed_mut(&mut resident).is_ok());
    assert!(<LegacyRoot as CustodyOwner>::installed_mut(&mut legacy).is_ok());
    assert!(matches!(
        <LegacyRoot as CustodyOwner>::installed_mut(&mut resident),
        Err(Error::Binding)
    ));
    assert!(matches!(
        <ResidentRoot as CustodyOwner>::installed_mut(&mut legacy),
        Err(Error::Binding)
    ));
    assert!(resident.resident_compact().is_some() && resident.compact().is_none());
    assert!(legacy.compact().is_some() && legacy.resident_compact().is_none());
}

#[test]
fn resident_publication_refusal_burns_original_attempt() {
    use schedule_finalization::BorrowedRecipe;
    use sealed::observe_publication_once;

    for short in [false, true] {
        // Each pressure boundary gets an independent genuine history and origin.
        let (mut custody, probe) = genuine_resident_root_fixture();
        let original = Arc::clone(&custody.original);
        let outside = std::rc::Rc::new(std::cell::RefCell::new(custody.finish_lease().unwrap()));
        let sealed = custody.seal_compact().unwrap();
        let work = sealed.work();
        let before = work_used(&original);
        assert_eq!(probe.state(), (false, false));
        let native = (0..sealed.logical_len())
            .find_map(|ordinal| match sealed.read_recipe(ordinal).unwrap().recipe {
                BorrowedRecipe::Tensor { program, .. }
                | BorrowedRecipe::CheckedRange { program, .. } => Some(Arc::downgrade(program)),
                _ => None,
            })
            .expect("actual original native invocation");
        let pressure = std::rc::Rc::clone(&outside);
        observe_publication_once(move |required| {
            let bytes = MAX_PROGRAM_BYTES.checked_sub(required).unwrap() + usize::from(short);
            pressure.borrow_mut().resize_payload(bytes).unwrap();
        });
        let result = sealed.into_program();
        if short {
            assert!(matches!(result, Err(Error::Bounds)));
            assert_eq!(probe.state(), (false, true));
        } else {
            let program = result.unwrap();
            assert_eq!(program.work_budget(), work);
            assert!(native.upgrade().is_some());
            assert_eq!(probe.state(), (false, false));
            drop(program);
            assert_eq!(probe.state(), (false, false));
        }
        assert!(work_used(&original) >= before);
        assert!(native.upgrade().is_none());
        assert!(outside.borrow().budget().used_bytes() >= outside.borrow().payload_bytes());
        // The consumed storage is not returned or retried. The probe retains the
        // original origin allocation; this is not final backing-retirement proof.
    }
}

#[test]
fn resident_publication_installs_resident_branch_and_commits_original_attempt() {
    let (custody, probe) = genuine_resident_root_fixture();
    let sealed = custody.seal_compact().unwrap();
    let work = sealed.work();
    assert_eq!(probe.state(), (false, false));
    let program = sealed.into_program().unwrap();
    assert_eq!(program.storage_kind(), "compact");
    assert!(program.nodes.is_empty());
    assert_eq!(program.work_budget(), work);
    assert_eq!(program.outputs.len(), 3);
    assert!(program.compact_storage().is_none());
    let installed = program.request_custody.as_deref().unwrap();
    assert!(installed.compact().is_none());
    installed.resident_compact().unwrap().check_completed().unwrap();
    assert_eq!(probe.state(), (false, false));
    drop(program);
    assert_eq!(probe.state(), (false, false));
    // Non-burn after Program drop is consistent with publication commitment.
    // It observes neither the private commit flag nor intermediate install order.
    // The probe keeps the original origin allocation alive throughout this check.
}
