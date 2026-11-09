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
fn observer_only_bytes() -> usize {
    // One remaining OwnedBudget observer: original root metadata was refunded,
    // its clone's metadata remains, as does the shared bootstrap/control block.
    OwnedBudget::resources().unwrap().initial_reserved_bytes
}

// Authoring only: safe filesystem fixtures; no runtime qualification claimed.
fn pending_fixture() -> Publisher {
    let (config, _, _) = crate::replicated_decoder::tests::fixture_sources("qwen2", false);
    Publisher::prepare_compact_source_request(&experiment(), &config, 4, 2, [2; 32], [3; 32]).unwrap()
}
struct CompactFaultReset;
impl Drop for CompactFaultReset {
    fn drop(&mut self) {
        program::source_entry::snapshot_directory::set_compact_fault(None);
    }
}
#[test]
fn pending_compact_cancel_is_terminal_until_explicit_retirement() {
    let mut publisher = pending_fixture();
    let key = publisher.shapes().first().unwrap().0.clone();
    let budget = counter(&publisher);
    let native = publisher.compact.as_deref().unwrap() as *const CompactPublisher;
    let before = budget.budget().used_bytes();
    publisher.cancel();
    assert!(publisher.has_retained_compact());
    assert_eq!(publisher.compact.as_deref().unwrap() as *const CompactPublisher, native);
    assert!(matches!(publisher.compact.as_ref().unwrap().custody.as_ref().unwrap().state,
        CompactDirectoryState::Failed));
    let original = counter(&publisher);
    assert!(std::ptr::eq(budget.budget(), original.budget()));
    drop(original);
    assert_eq!(budget.budget().used_bytes(), before);
    assert_eq!(publisher.compact_start().unwrap_err(), "compact acquisition already attempted");
    assert_eq!(publisher.append(&key, 0, &0f32.to_le_bytes()).unwrap_err(),
        "compact acquisition not ready");
    assert!(publisher.has_retained_compact());
    assert_eq!(publisher.compact.as_deref().unwrap() as *const CompactPublisher, native);
    assert!(matches!(publisher.compact.as_ref().unwrap().custody.as_ref().unwrap().state,
        CompactDirectoryState::Failed));
    let original = counter(&publisher);
    assert!(std::ptr::eq(budget.budget(), original.budget()));
    drop(original);
    assert_eq!(budget.budget().used_bytes(), before);
    assert_eq!(publisher.compact.as_ref().unwrap().custody.as_ref().unwrap()
        .directory.pending_state_for_tests(), (false, false, false, false));
    publisher.compact_retire().unwrap();
    assert!(matches!(publisher.compact.as_ref().unwrap().stage, Stage::Retired));
    assert!(budget.budget().used_bytes() < before);
    let retired = budget.budget().used_bytes();
    publisher.compact_retire().unwrap();
    assert_eq!(budget.budget().used_bytes(), retired);
    drop(publisher);
    assert_eq!(budget.budget().used_bytes(), observer_only_bytes());
}
#[test]
fn started_compact_cancel_preserves_exact_owner_payload_directory_and_charges_until_retirement() {
    let mut publisher = pending_fixture();
    let budget = counter(&publisher);
    let (key, shape) = publisher.shapes().first().unwrap().clone();
    let (_, _, values) = crate::replicated_decoder::tests::fixture_sources("qwen2", false);
    let bytes = values[&key].iter()
        .flat_map(|value| (*value as f32 / ONE as f32).to_le_bytes())
        .collect::<Vec<_>>();
    assert_eq!(bytes.len(), shape.iter().product::<usize>() * 4);
    assert!(!bytes.is_empty() && bytes.len() <= CHUNK * 4);
    publisher.compact_start().unwrap();
    let path = publisher.collection_disk_directory().unwrap().to_path_buf();
    assert!(path.is_dir());
    let identity = |p: &Publisher| {
        let compact = p.compact.as_deref().unwrap();
        let custody = compact.custody.as_ref().unwrap();
        let Stage::Collecting(owner) = &compact.stage else { panic!("lost collecting stage") };
        assert!(std::ptr::eq(budget.budget(), owner.payload().unwrap().root.budget()));
        (
            compact as *const _ as usize,
            custody as *const _ as usize,
            owner as *const _ as usize,
            owner.payload().unwrap() as *const _ as usize,
            Arc::as_ptr(&custody.directory) as usize,
            &owner.payload().unwrap().root as *const _ as usize,
        )
    };
    let original = identity(&publisher);
    let charged = budget.budget().used_bytes();
    let disk = publisher.collection_disk_bytes();
    publisher.cancel();
    assert_eq!(identity(&publisher), original);
    assert_eq!(budget.budget().used_bytes(), charged);
    assert!(publisher.has_retained_compact());
    assert!(!publisher.retired);
    assert!(matches!(publisher.compact.as_ref().unwrap().custody.as_ref().unwrap().state,
        CompactDirectoryState::Failed));
    assert_eq!(publisher.compact_start().unwrap_err(), "compact acquisition already attempted");
    assert_eq!(publisher.append(&key, 0, &bytes).unwrap_err(), "compact acquisition not ready");
    assert_eq!(publisher.seal(&[0, 1, 1, 2]).unwrap_err(), "compact acquisition not ready");
    assert_eq!(publisher.next_chunk().err().unwrap(), "compact acquisition not ready");
    // finish_outcome checks readiness before decoding even well-typed frames.
    assert_eq!(publisher.compact_finish_numeric_outcome([&[0; 145]; 3]).unwrap_err(),
        "compact acquisition not ready");
    assert_eq!(identity(&publisher), original);
    assert_eq!(budget.budget().used_bytes(), charged);
    assert_eq!(publisher.collection_disk_bytes(), disk);
    assert!(publisher.collection_disk_directory().is_err());
    assert!(path.is_dir());
    assert_eq!(publisher.compact.as_ref().unwrap().custody.as_ref().unwrap()
        .directory.pending_state_for_tests(), (true, true, false, false));
    publisher.compact_retire().unwrap();
    assert!(!path.exists());
    assert!(matches!(publisher.compact.as_ref().unwrap().stage, Stage::Retired));
    assert!(budget.budget().used_bytes() < charged);
    let retired = budget.budget().used_bytes();
    publisher.compact_retire().unwrap();
    assert_eq!(budget.budget().used_bytes(), retired);
    drop(publisher);
    assert_eq!(budget.budget().used_bytes(), observer_only_bytes());
}
#[test]
fn pending_compact_premature_append_is_terminal_until_explicit_retirement() {
    let mut publisher = pending_fixture();
    let key = publisher.shapes().first().unwrap().0.clone();
    let budget = counter(&publisher);
    let native = publisher.compact.as_deref().unwrap() as *const CompactPublisher;
    let before = budget.budget().used_bytes();
    assert_eq!(publisher.append(&key, 0, &0f32.to_le_bytes()).unwrap_err(),
        "compact acquisition not ready");
    assert!(publisher.has_retained_compact());
    assert_eq!(publisher.compact.as_deref().unwrap() as *const CompactPublisher, native);
    assert!(matches!(publisher.compact.as_ref().unwrap().custody.as_ref().unwrap().state,
        CompactDirectoryState::Failed));
    let original = counter(&publisher);
    assert!(std::ptr::eq(budget.budget(), original.budget()));
    drop(original);
    assert_eq!(budget.budget().used_bytes(), before);
    assert_eq!(publisher.compact_start().unwrap_err(), "compact acquisition already attempted");
    assert_eq!(publisher.append(&key, 0, &0f32.to_le_bytes()).unwrap_err(),
        "compact acquisition not ready");
    assert!(publisher.has_retained_compact());
    assert_eq!(publisher.compact.as_deref().unwrap() as *const CompactPublisher, native);
    assert!(matches!(publisher.compact.as_ref().unwrap().custody.as_ref().unwrap().state,
        CompactDirectoryState::Failed));
    let original = counter(&publisher);
    assert!(std::ptr::eq(budget.budget(), original.budget()));
    drop(original);
    assert_eq!(budget.budget().used_bytes(), before);
    assert_eq!(publisher.compact.as_ref().unwrap().custody.as_ref().unwrap()
        .directory.pending_state_for_tests(), (false, false, false, false));
    publisher.compact_retire().unwrap();
    assert!(matches!(publisher.compact.as_ref().unwrap().stage, Stage::Retired));
    assert!(budget.budget().used_bytes() < before);
    let retired = budget.budget().used_bytes();
    publisher.compact_retire().unwrap();
    assert_eq!(budget.budget().used_bytes(), retired);
    drop(publisher);
    assert_eq!(budget.budget().used_bytes(), observer_only_bytes());
}
#[test]
fn pending_compact_is_close_ready_without_directory_acquisition() {
    let mut publisher = pending_fixture();
    let budget = counter(&publisher);
    let before = budget.budget().used_bytes();
    let native = publisher.compact.as_deref().unwrap() as *const CompactPublisher;
    assert_eq!(publisher.collection_disk_directory().unwrap(),
        program::source_entry::snapshot_directory::parent().unwrap());
    publisher.compact_retire().unwrap();
    assert_eq!(publisher.compact.as_deref().unwrap() as *const CompactPublisher, native);
    assert!(budget.budget().used_bytes() < before);
    let after = budget.budget().used_bytes();
    publisher.compact_retire().unwrap();
    assert_eq!(budget.budget().used_bytes(), after);
    assert!(publisher.compact_start().is_err());
    drop(publisher);
    assert_eq!(budget.budget().used_bytes(), observer_only_bytes());
}
#[test]
fn compact_start_failure_before_factory_has_no_directory_to_retire() {
    use program::source_entry::snapshot_directory::{self as directory, CompactFault};
    let _reset = CompactFaultReset;
    let mut publisher = pending_fixture();
    let budget = counter(&publisher);
    directory::set_compact_fault(Some(CompactFault::Before));
    assert!(publisher.compact_start().is_err());
    let custody = publisher.compact.as_ref().unwrap().custody.as_ref().unwrap();
    assert_eq!(custody.directory.pending_state_for_tests(), (false, false, false, false));
    publisher.compact_retire().unwrap();
    drop(publisher);
    assert_eq!(budget.budget().used_bytes(), observer_only_bytes());
}
#[test]
fn compact_synthetic_pre_factory_uncertainty_retains_unknown_not_a_live_handle() {
    use program::source_entry::snapshot_directory::{self as directory, CompactFault};
    let _reset = CompactFaultReset;
    let mut publisher = pending_fixture();
    let budget = counter(&publisher);
    let charge = budget.budget().used_bytes();
    // Synthetic failure BEFORE tempfile's factory call, after uncertainty is
    // marked. No failure inside its mkdir implementation is injected/observed.
    directory::set_compact_fault(Some(CompactFault::Opaque));
    assert!(publisher.compact_start().is_err());
    assert!(publisher.collection_disk_directory().is_err());
    let native = publisher.compact.as_deref().unwrap() as *const CompactPublisher;
    assert_eq!(publisher.compact.as_ref().unwrap().custody.as_ref().unwrap()
        .directory.pending_state_for_tests(), (false, false, false, true));
    directory::set_compact_fault(None);
    assert!(publisher.compact_retire().is_err());
    assert!(publisher.compact_retire().is_err());
    assert_eq!(publisher.compact.as_deref().unwrap() as *const CompactPublisher, native);
    assert_eq!(budget.budget().used_bytes(), charge);
    assert_eq!(publisher.compact.as_ref().unwrap().custody.as_ref().unwrap()
        .directory.pending_state_for_tests(), (false, false, true, true));
    let retained = assert_failed_directory_charge_survives_drop(publisher, &budget);
    // No returned path/handle exists here. Only unresolved charge is observable;
    // do not turn this pre-factory synthetic uncertainty into live-file evidence.
    assert_eq!(budget.budget().used_bytes(), retained);
}
#[test]
fn compact_partial_acquisition_keeps_original_owner_until_explicit_retirement() {
    use program::source_entry::snapshot_directory::{self as directory, CompactFault};
    // Existing cuts: returned directory plus copied path; returned anonymous
    // anchor; completed metadata checks. Not faults inside tempfile internals.
    for point in [CompactFault::Directory, CompactFault::Anchor, CompactFault::Metadata] {
        let _reset = CompactFaultReset;
        let mut publisher = pending_fixture();
        let budget = counter(&publisher);
        let native = publisher.compact.as_deref().unwrap() as *const CompactPublisher;
        let charge = budget.budget().used_bytes();
        directory::set_compact_fault(Some(point));
        assert!(publisher.compact_start().is_err());
        let custody = publisher.compact.as_ref().unwrap().custody.as_ref().unwrap();
        let path = custody.directory.path().to_path_buf();
        assert!(path.is_dir());
        assert_eq!(budget.budget().used_bytes(), charge);
        assert_eq!(publisher.compact.as_deref().unwrap() as *const CompactPublisher, native);
        assert!(publisher.compact_start().is_err());
        directory::set_compact_fault(None);
        publisher.compact_retire().unwrap();
        assert!(!path.exists());
        let retired = budget.budget().used_bytes();
        publisher.compact_retire().unwrap();
        assert_eq!(budget.budget().used_bytes(), retired);
        drop(publisher);
        assert_eq!(budget.budget().used_bytes(), observer_only_bytes());
    }
}
#[test]
fn compact_remove_error_preserves_exact_stage_directory_and_charge_without_retry() {
    use program::source_entry::snapshot_directory::{self as directory, CompactFault};
    let _reset = CompactFaultReset;
    let mut publisher = pending_fixture();
    let budget = counter(&publisher);
    publisher.compact_start().unwrap();
    let path = publisher.collection_disk_directory().unwrap().to_path_buf();
    let owner = publisher.compact.as_deref().unwrap() as *const CompactPublisher;
    let original = counter(&publisher);
    assert!(std::ptr::eq(budget.budget(), original.budget()));
    drop(original);
    let before = budget.budget().used_bytes();
    directory::set_compact_fault(Some(CompactFault::Remove));
    assert!(publisher.compact_retire().is_err());
    directory::set_compact_fault(None); // No reset/recovery privilege follows.
    assert!(publisher.compact_retire().is_err());
    assert!(publisher.compact_start().is_err());
    assert!(path.is_dir());
    assert_eq!(publisher.compact.as_deref().unwrap() as *const CompactPublisher, owner);
    assert!(matches!(publisher.compact.as_ref().unwrap().stage, Stage::Collecting(_)));
    assert_eq!(budget.budget().used_bytes(), before);
    let retained = assert_failed_directory_charge_survives_drop(publisher, &budget);
    assert!(path.is_dir()); // Drop did NOT automatically retry successful removal.
    // Test-only fixture removal is NOT product retirement. No publisher or
    // directory Arc remains on the caller stack to hold the native location.
    std::fs::remove_dir(&path).unwrap();
    assert!(!path.exists());
    assert_eq!(budget.budget().used_bytes(), retained);
}
// Keep only the original accounting observer and a Weak wrapper observation.
// Publisher/collector allocations legitimately disappear on Drop; the shared
// directory's original charge must remain, without a caller-owned directory Arc.
fn assert_failed_directory_charge_survives_drop(publisher: Publisher, budget: &OwnedBudget) -> usize {
    let wrapper = Arc::downgrade(&publisher.compact.as_ref().unwrap().custody.as_ref().unwrap().directory);
    let before = budget.budget().used_bytes();
    drop(publisher);
    assert!(wrapper.upgrade().is_none(), "caller-facing directory wrapper was not dropped");
    let retained = budget.budget().used_bytes();
    assert!(retained > observer_only_bytes() && retained < before,
        "payload must drop, original location charge must remain above observer-only baseline");
    drop(wrapper);
    assert_eq!(budget.budget().used_bytes(), retained);
    retained
}
#[test]
fn compact_nonempty_path_replacement_retains_both_directories_and_original_charge() {
    use program::source_entry::snapshot_directory as directory;
    let mut publisher = pending_fixture();
    let budget = counter(&publisher);
    publisher.compact_start().unwrap();
    let path = publisher.collection_disk_directory().unwrap().to_path_buf();
    let fixture = tempfile::tempdir_in(directory::parent().unwrap()).unwrap();
    let moved = fixture.path().join("original");
    std::fs::rename(&path, &moved).unwrap();
    std::fs::create_dir(&path).unwrap();
    let marker = path.join("replacement-marker");
    std::fs::write(&marker, b"replacement belongs to fixture").unwrap();
    let before = budget.budget().used_bytes();
    // Existing nonrecursive removal must fail on nonempty replacement. This
    // does NOT assert an inode-identity policy for an empty replacement.
    assert!(publisher.compact_retire().is_err());
    assert!(publisher.compact_retire().is_err());
    assert_eq!(budget.budget().used_bytes(), before);
    let retained = assert_failed_directory_charge_survives_drop(publisher, &budget);
    assert!(moved.is_dir() && path.is_dir());
    assert_eq!(std::fs::read(&marker).unwrap(), b"replacement belongs to fixture");
    // External fixture cleanup, NOT native retirement/recovery or refund proof.
    std::fs::remove_file(&marker).unwrap();
    std::fs::remove_dir(&path).unwrap();
    drop(fixture);
    assert_eq!(budget.budget().used_bytes(), retained);
}
#[cfg(unix)]
#[test]
fn compact_symlink_path_replacement_never_removes_target_or_refunds_on_drop() {
    use program::source_entry::snapshot_directory as directory;
    let mut publisher = pending_fixture();
    let budget = counter(&publisher);
    publisher.compact_start().unwrap();
    let path = publisher.collection_disk_directory().unwrap().to_path_buf();
    let fixture = tempfile::tempdir_in(directory::parent().unwrap()).unwrap();
    let moved = fixture.path().join("original");
    let target = fixture.path().join("target");
    std::fs::rename(&path, &moved).unwrap();
    std::fs::create_dir(&target).unwrap();
    std::os::unix::fs::symlink(&target, &path).unwrap();
    let before = budget.budget().used_bytes();
    assert!(publisher.compact_retire().is_err()); // rmdir does not follow symlink.
    assert!(publisher.compact_retire().is_err());
    assert_eq!(budget.budget().used_bytes(), before);
    let retained = assert_failed_directory_charge_survives_drop(publisher, &budget);
    assert!(std::fs::symlink_metadata(&path).unwrap().file_type().is_symlink());
    assert!(target.is_dir() && moved.is_dir());
    std::fs::remove_file(&path).unwrap(); // External fixture cleanup only.
    drop(fixture);
    assert_eq!(budget.budget().used_bytes(), retained);
}
#[test]
fn compact_externally_unlinked_directory_is_not_native_retirement_or_refund_proof() {
    let mut publisher = pending_fixture();
    let budget = counter(&publisher);
    publisher.compact_start().unwrap();
    let path = publisher.collection_disk_directory().unwrap().to_path_buf();
    std::fs::remove_dir(&path).unwrap(); // Fixture interference, not native close.
    let before = budget.budget().used_bytes();
    assert!(publisher.compact_retire().is_err());
    assert!(publisher.compact_retire().is_err());
    assert_eq!(publisher.compact.as_ref().unwrap().custody.as_ref().unwrap()
        .directory.pending_state_for_tests(), (true, true, true, false));
    assert_eq!(budget.budget().used_bytes(), before);
    let retained = assert_failed_directory_charge_survives_drop(publisher, &budget);
    assert!(!path.exists());
    assert_eq!(budget.budget().used_bytes(), retained);
}
#[test]
fn compact_collection_failure_does_not_implicitly_drop_acquired_directory() {
    let mut publisher = pending_fixture();
    let budget = counter(&publisher);
    publisher.compact_start().unwrap();
    let path = publisher.collection_disk_directory().unwrap().to_path_buf();
    let native = publisher.compact.as_deref().unwrap() as *const CompactPublisher;
    let (key, shape) = publisher.shapes().first().unwrap().clone();
    let elements = shape.iter().product::<usize>();
    let (_, _, values) = crate::replicated_decoder::tests::fixture_sources("qwen2", false);
    let values = &values[&key];
    assert_eq!(values.len(), elements);
    assert!(elements > 0 && elements <= CHUNK);
    let bytes = values.iter().enumerate().flat_map(|(index, value)| {
        let value = if index == 0 { f32::NAN } else { *value as f32 / ONE as f32 };
        value.to_le_bytes()
    }).collect::<Vec<_>>();
    assert_eq!(bytes.len(), elements * 4); // Full first tensor, exact F32 framing.
    assert_eq!(&bytes[..4], &f32::NAN.to_le_bytes());
    // Actual first key, offset zero and full valid frame pass ordering/length
    // guards. Scan's first-lane conversion maps NaN to Bounds, not Binding.
    assert_eq!(publisher.append(&key, 0, &bytes).unwrap_err(), Error::Bounds.to_string());
    assert!(path.is_dir());
    assert_eq!(publisher.compact.as_deref().unwrap() as *const CompactPublisher, native);
    assert!(budget.budget().used_bytes() > observer_only_bytes()); // Same original counter.
    assert!(publisher.compact_start().is_err());
    publisher.compact_retire().unwrap();
    assert!(!path.exists());
    let retired = budget.budget().used_bytes();
    publisher.compact_retire().unwrap();
    assert_eq!(budget.budget().used_bytes(), retired);
    drop(publisher);
    assert_eq!(budget.budget().used_bytes(), observer_only_bytes());
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
fn repeated_seal_preserves_actual_resident_sealed_owner_payload_program_and_charges() {
    let (mut publisher, values) = fixture();
    let budget = counter(&publisher);
    collect(&mut publisher, &values);
    publisher.seal(&[0, 1, 1, 2]).unwrap();
    let compact = publisher.compact.as_mut().unwrap();
    let Stage::Sealed(owner) = &compact.stage else {
        panic!("genuine fixture did not seal")
    };
    let identities = (
        owner as *const _,
        owner.program().unwrap() as *const _,
        owner.payload().unwrap() as *const _,
        Arc::as_ptr(owner.directory().unwrap()),
    );
    assert!(std::ptr::eq(budget.budget(), owner.payload().unwrap().root.budget()));
    assert_eq!(owner.program().unwrap().program().storage_kind(), "compact");
    let charged = budget.budget().used_bytes();
    assert!(charged > observer_only_bytes());
    // Exercise the narrow CompactPublisher guard directly. This does not claim
    // the outer legacy Publisher error path or consuming seal is repaired.
    for ids in [&[0, 1, 1, 2][..], &[][..]] {
        assert_eq!(compact.seal(ids).unwrap_err(), "compact publisher is not collecting");
        let Stage::Sealed(owner) = &compact.stage else {
            panic!("wrong-stage seal extracted the resident sealed owner")
        };
        assert_eq!((
            owner as *const _,
            owner.program().unwrap() as *const _,
            owner.payload().unwrap() as *const _,
            Arc::as_ptr(owner.directory().unwrap()),
        ), identities);
        assert!(std::ptr::eq(budget.budget(), owner.payload().unwrap().root.budget()));
        assert_eq!(owner.program().unwrap().program().storage_kind(), "compact");
        assert_eq!(budget.budget().used_bytes(), charged);
    }
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
