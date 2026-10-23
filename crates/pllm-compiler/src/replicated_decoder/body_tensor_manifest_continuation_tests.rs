//! Parent wiring: add this CHILD module to body_tensor_integration_tests.rs:
//! #[path = "body_tensor_manifest_continuation_tests.rs"]
//! mod manifest_continuation;
//!
//! Uses that module's existing real router/custody/snapshot helpers, not another
//! Worker/runtime or production capability issuer. Replaces ONE original map in
//! an ordinary Manifest::compile Program, then resumes its actual protected
//! suffix. This does not qualify a complete compact plan or enable production.
use super::*;
use crate::replicated_decoder::program::manifest_prepared_fixture::{
    self, Q_PROJECTION_SOURCE_KEY,
};

#[path = "body_tensor_restore_tests.rs"]
mod restoration;

#[path = "body_tensor_manifest_accounting_tests.rs"]
mod accounting_tests;

fn imported_workers(
    program: &Arc<Program>,
    manifest: &crate::replicated_decoder::Manifest,
) -> [Worker; 3] {
    let roots = puma::prss::insecure_test_support::fresh_compound_parties(
        puma::Context::new([42; 32], program.digest(), 0).unwrap(),
        program.work_budget().limits().unwrap(),
    )
    .unwrap();
    let mut roles = roots.map(|root| {
        Worker::new_with_source(
            program.clone(),
            root,
            [42; 32],
            Some(import_public_dispatch_source(manifest)),
        )
        .unwrap()
    });
    for (key, lanes) in program.inputs() {
        // Real token owner, NOT the negative activation values used by the
        // source-free map fixture. Keep both nontrivial replicated components.
        assert_eq!(key, "input.tokens");
        let first: Vec<_> = (0..lanes)
            .map(|i| 0xc751f00d12345678u64.wrapping_add(i as u64 * 17))
            .collect();
        let second: Vec<_> = (0..lanes)
            .map(|i| 0x254277811928f93du64.wrapping_add(i as u64 * 311))
            .collect();
        let third: Vec<_> = first
            .iter()
            .zip(&second)
            .map(|(&a, &b)| 0u64.wrapping_sub(a).wrapping_sub(b))
            .collect();
        let shares = [first, second, third];
        for (party, role) in roles.iter_mut().enumerate() {
            role.import(key, &shares[party], &shares[(party + 1) % 3])
                .unwrap();
        }
    }
    roles
}

fn source_backed_candidate() -> (
    crate::replicated_decoder::Manifest,
    Arc<Compiled>,
    Arc<PublicSource>,
    [Worker; 3],
) {
    let manifest = manifest_prepared_fixture::public_manifest().unwrap();
    let (program, prepared, input, output) =
        manifest_prepared_fixture::compile_manifest_rescale18(&manifest, Q_PROJECTION_SOURCE_KEY)
            .unwrap();
    let public = Arc::new(import_public_dispatch_source(&manifest));
    let registry = SourceRegistry::from_manifest(&manifest).unwrap();
    let roles = imported_workers(&program, &manifest);
    let source = SourceDispatch::new(
        &program,
        &roles[0],
        public.clone(),
        &registry,
        manifest_inline_baseline(&manifest),
    )
    .unwrap();
    let compiled = Compiled::prepare_checked(
        program,
        WeightMode::Public,
        input,
        output,
        prepared,
        Some(source),
        || Ok(()),
    )
    .unwrap();
    #[allow(clippy::drop_non_drop)] // End the registry borrow before original owners.
    drop(registry);
    (manifest, compiled, public, roles)
}

#[test]
fn source_backed_invocation_cancel_burns_custody_and_releases_owners() {
    run_large_stack(|| source_invocation_failure(false));
}

#[test]
fn source_backed_invocation_malformed_receive_burns_custody_and_releases_owners() {
    run_large_stack(|| source_invocation_failure(true));
}

fn source_invocation_failure(malformed: bool) {
    let (manifest, compiled, public, mut roles) = source_backed_candidate();
    let native_before = native_attempts();
    let plan = Arc::downgrade(&compiled.plan);
    let program = Arc::downgrade(&compiled.program);
    let source = Arc::downgrade(&public);
    let registry = SourceRegistry::from_manifest(&manifest).unwrap();
    reach(&mut roles, compiled.binding.output, Some(&compiled));
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
    assert_eq!(Arc::strong_count(&public), 5);
    assert_eq!(plan.strong_count(), 4);
    #[allow(clippy::drop_non_drop)] // End the registry borrow before original owners.
    drop(registry);
    drop(manifest);
    // Keep real emitted frames alive through cancellation and owner teardown.
    let held = start(&mut bodies);
    assert!(!held.is_empty());
    let bytes: Vec<_> = held
        .iter()
        .map(|(_, _, raw)| raw.as_bytes().to_vec())
        .collect();
    for body in &mut bodies {
        assert!(!body.complete());
        if malformed {
            assert!(body.receive(b"bad", body.party().next()).is_err());
        } else {
            body.cancel();
        }
        assert!(!body.complete());
        assert!(matches!(body.advance(), Err(Error::Retired)));
        assert!(matches!(
            body.receive(b"bad", body.party().next()),
            Err(Error::Retired)
        ));
    }
    assert_eq!(Arc::strong_count(&public), 2);
    assert_eq!(plan.strong_count(), 1);
    // Faulted roles cannot obtain end custody or restore the protected suffix.
    assert!(matches!(restore_after_drive(bodies), Err(Error::Retired)));
    assert_eq!(Arc::strong_count(&compiled), 1);
    assert_eq!(program.strong_count(), 1);
    drop(compiled);
    assert_eq!(plan.strong_count(), 0);
    assert_eq!(program.strong_count(), 0);
    assert_eq!(Arc::strong_count(&public), 1);
    drop(public);
    assert_eq!(source.strong_count(), 0);
    for ((_, _, raw), expected) in held.iter().zip(&bytes) {
        assert_eq!(raw.as_bytes(), expected);
    }
    assert_eq!(native_attempts(), native_before);
}

#[test]
fn source_backed_handoff_rejects_missing_substituted_and_grown_registry_without_owner_leaks() {
    run_large_stack(rejected_source_handoffs);
}

fn rejected_source_handoffs() {
    for case in 0..3 {
        let (mut manifest, compiled, public, mut roles) = source_backed_candidate();
        let native_before = native_attempts();
        let plan = Arc::downgrade(&compiled.plan);
        let program = Arc::downgrade(&compiled.program);
        let source = Arc::downgrade(&public);
        reach(&mut roles, compiled.binding.output, Some(&compiled));
        let drained = TestCoordinator {
            roles,
            queue: Vec::new(),
        }
        .certify_start(&compiled)
        .unwrap();
        match case {
            0 => (), // Source-backed compilation requires registry-bearing begin.
            1 => {
                manifest
                    .tensors
                    .get_mut(Q_PROJECTION_SOURCE_KEY)
                    .unwrap()
                    .digest[0] ^= 1
            }
            _ => manifest
                .tensors
                .get_mut(Q_PROJECTION_SOURCE_KEY)
                .unwrap()
                .shape
                .reserve(1024),
        }
        let registry = SourceRegistry::from_manifest(&manifest).unwrap();
        if case == 2 {
            // Authentic records are insufficient when live capacities exceed the
            // original source admission. Do not confuse this with substitution.
            public.check_registry(&registry).unwrap();
        }
        for owner in drained {
            let result = if case == 0 {
                compiled.clone().begin(owner)
            } else {
                compiled.clone().begin_with_registry(owner, &registry)
            };
            if case == 2 {
                assert!(matches!(result, Err(Error::Bounds)));
            } else {
                assert!(matches!(result, Err(Error::Binding)));
            }
            // Consuming rejection returns no Invocation/Worker to restore, and
            // never retains an executor, source Arc, or additional Plan owner.
            assert_eq!(Arc::strong_count(&compiled), 1);
            assert_eq!(Arc::strong_count(&public), 2);
            assert_eq!(plan.strong_count(), 1);
        }
        assert_eq!(program.strong_count(), 1);
        #[allow(clippy::drop_non_drop)] // End the registry borrow before original owners.
        drop(registry);
        drop(manifest);
        drop(compiled);
        assert_eq!(program.strong_count(), 0);
        assert_eq!(plan.strong_count(), 0);
        assert_eq!(Arc::strong_count(&public), 1);
        drop(public);
        assert_eq!(source.strong_count(), 0);
        assert_eq!(native_attempts(), native_before);
        assert!(matches!(
            worker_parity_gate(),
            Err(PreparationError::UnqualifiedOriginalWorkerParity)
        ));
    }
}

#[test]
fn exact_manifest_public_source_worker_map_handoff_matches_frames_both_shares_cursors_and_suffix() {
    run_large_stack(manifest_continuation);
}

fn manifest_continuation() {
    let manifest = manifest_prepared_fixture::public_manifest().unwrap();
    let (program, prepared, input, output) =
        manifest_prepared_fixture::compile_manifest_rescale18(&manifest, Q_PROJECTION_SOURCE_KEY)
            .unwrap();
    assert_eq!(program.digest(), manifest.program);
    // Independent default compile: capture guard is already unarmed.
    let ordinary_program = manifest.compile().unwrap();
    assert_eq!(ordinary_program.digest(), program.digest());
    assert_eq!(ordinary_program.retire, program.retire);
    assert_eq!(ordinary_program.outputs, program.outputs);
    let native_before = native_attempts();
    let full = Arc::downgrade(prepared.program_at(0).unwrap());
    let Code::TensorLoop {
        program: original, ..
    } = &program.nodes[output].code
    else {
        panic!("exact Manifest compile must publish selected loop")
    };
    assert!(Arc::ptr_eq(original, prepared.program_at(0).unwrap()));
    let public = Arc::new(import_public_dispatch_source(&manifest));
    let registry = SourceRegistry::from_manifest(&manifest).unwrap();
    let mut candidate = imported_workers(&program, &manifest);
    let source = SourceDispatch::new(
        &program,
        &candidate[0],
        public.clone(),
        &registry,
        manifest_inline_baseline(&manifest),
    )
    .unwrap();
    // Test-only candidate qualification, IDENTICAL binding/source/accounting
    // checks; no call to or change of the closed production parity gate.
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
    assert_eq!(native_attempts(), native_before);
    assert!(matches!(
        worker_parity_gate(),
        Err(PreparationError::UnqualifiedOriginalWorkerParity)
    ));
    let plan = Arc::downgrade(&compiled.plan);
    let mut ordinary = imported_workers(&ordinary_program, &manifest);
    let (whole_wire, whole_held) = drive(&mut ordinary);
    let whole_state = ordinary.each_ref().map(snapshot);
    let whole_shares = ordinary.map(|role| role.finish().unwrap().to_vec());

    let mut reference = imported_workers(&ordinary_program, &manifest);
    let mut wire = reach(&mut candidate, output, Some(&compiled));
    assert_eq!(wire, reach(&mut reference, output, None));
    assert_eq!(
        candidate.each_ref().map(snapshot),
        reference.each_ref().map(snapshot)
    );
    let expected_body_wire = reach(&mut reference, output + 1, None);
    // reach owns/routes every emitted frame and returns only after ALL roles
    // reached the same boundary and its router drained. Local idleness alone is
    // never the issuance condition. This is the existing TEST coordinator only.
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
    assert_eq!(Arc::strong_count(&public), 5); // Caller + compiled + three executors.
    assert_eq!(plan.strong_count(), 4);
    #[allow(clippy::drop_non_drop)] // End the registry borrow before original owners.
    drop(registry);
    drop(manifest); // Constructor-only source authority; no borrowed backing retained.
    let (body_wire, body_held) = drive(&mut bodies);
    assert_eq!(body_wire, expected_body_wire);
    assert!(!body_wire.is_empty());
    wire.extend(body_wire);
    let mut resumed = restore_after_drive(bodies).unwrap();
    assert_eq!(Arc::strong_count(&public), 2);
    assert_eq!(plan.strong_count(), 1);
    let seam = resumed.each_ref().map(snapshot);
    assert_eq!(seam, reference.each_ref().map(snapshot));
    for state in &seam {
        assert_eq!(state.step, output + 1);
        assert!(state.shares[input].is_none());
        assert!(state.shares[output].is_some());
    }
    let (suffix, suffix_held) = drive(&mut resumed);
    assert!(!suffix.is_empty()); // Actual protected decoder continuation.
    wire.extend(suffix);
    assert_eq!(wire, whole_wire);
    assert_eq!(resumed.each_ref().map(snapshot), whole_state);
    assert_eq!(
        resumed.map(|role| role.finish().unwrap().to_vec()),
        whole_shares
    );
    assert_eq!(native_attempts(), native_before);
    drop(reference);
    assert_eq!(Arc::strong_count(&compiled), 1);
    assert_eq!(Arc::strong_count(&program), 2);
    drop(compiled);
    assert_eq!(Arc::strong_count(&public), 1);
    assert_eq!(plan.strong_count(), 0);
    assert!(full.strong_count() > 0); // Original Program still owns original kernel.
    drop(program);
    assert_eq!(full.strong_count(), 0);
    drop((whole_held, body_held, suffix_held));
}

#[test]
fn genuine_manifest_preparation_stays_globally_unqualified() {
    run_large_stack(|| {
        let manifest = manifest_prepared_fixture::public_manifest().unwrap();
        let (program, prepared, input, output) =
            manifest_prepared_fixture::compile_manifest_rescale18(
                &manifest,
                Q_PROJECTION_SOURCE_KEY,
            )
            .unwrap();
        let roles = imported_workers(&program, &manifest);
        let public = Arc::new(import_public_dispatch_source(&manifest));
        let registry = SourceRegistry::from_manifest(&manifest).unwrap();
        let before = native_attempts();
        let result = Compiled::prepare_with_registry(
            program,
            WeightMode::Public,
            input,
            output,
            prepared,
            &roles[0],
            public,
            &registry,
            manifest_inline_baseline(&manifest),
        );
        assert!(matches!(
            result,
            Err(PreparationError::UnqualifiedOriginalWorkerParity)
        ));
        assert_eq!(native_attempts(), before);
    });
}

#[path = "body_tensor_manifest_full_tail_tests.rs"]
mod full_tail;
