//! Independent observations of ORIGINAL Manifest-compiled execution owners.
//!
//! PayloadUsage is captured before each original runtime enters its Worker. It
//! observes reservations, NOT cancellation/key erasure: zero live payload cannot
//! prove a runtime was canceled. PublicSource's Weak below observes the actual
//! dispatch Arc, NOT Worker's independently imported, by-value PublicSource.
//! Neither owner is replaced by a drop proxy. Direct failed-restore observation
//! of native cancellation and Worker source destruction still needs an existing
//! owner-specific observer that these APIs do not currently expose.
use super::*;

struct CompletedSource {
    compiled: Arc<Compiled>,
    public: Arc<PublicSource>,
    drained: [CoordinatorDrained<Invocation>; 3],
    usage: [puma::PayloadUsage; 3],
    wire: Wire,
    held: Vec<PeerBytes>,
    native_before: usize,
}

fn completed_source(replay_unwind: bool) -> CompletedSource {
    let manifest = manifest_prepared_fixture::public_manifest().unwrap();
    let (mut program, prepared, input, output) =
        manifest_prepared_fixture::compile_manifest_rescale18(&manifest, Q_PROJECTION_SOURCE_KEY)
            .unwrap();
    assert_eq!(program.digest(), manifest.program);
    assert!(program.nodes[output].lanes >= 2);
    if replay_unwind {
        // Corrupt only the TEST fixture's original retirement ledger after the
        // real Manifest::compile result. Prefix execution/body Plan do not read
        // this entry; restore_owned replays it AFTER copying every output lane.
        Arc::get_mut(&mut program).unwrap().retire[output].push(usize::MAX);
    }
    let native_before = native_attempts();
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
            Some(import_public_dispatch_source(&manifest)),
        )
        .unwrap()
    });
    // Same nontrivial replicated token-zero components as imported_workers;
    // observation handles above neither clone nor reset any consuming runtime.
    for (key, lanes) in program.inputs() {
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
    let public = Arc::new(import_public_dispatch_source(&manifest));
    let registry = SourceRegistry::from_manifest(&manifest).unwrap();
    let source = SourceDispatch::new(
        &program,
        &roles[0],
        public.clone(),
        &registry,
        manifest_inline_baseline(&manifest),
    )
    .unwrap();
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
    reach(&mut roles, output, Some(&compiled));
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
    drop((manifest, program));
    let (wire, held) = drive(&mut bodies);
    let drained = TestCoordinator {
        roles: bodies,
        queue: Vec::new(),
    }
    .certify_end()
    .unwrap();
    assert!(!held.is_empty());
    assert_eq!(held.len(), wire.len());
    assert!(usage.iter().any(|u| u.live_bytes() > 0));
    assert_eq!(native_attempts(), native_before);
    CompletedSource {
        compiled,
        public,
        drained,
        usage,
        wire,
        held,
        native_before,
    }
}

fn live(usage: &[puma::PayloadUsage; 3]) -> [usize; 3] {
    usage.each_ref().map(|u| u.live_bytes())
}

#[test]
fn manifest_restore_failure_unwind_keeps_dispatch_through_values_and_releases_original_reservations(
) {
    run_large_stack(|| {
        // Preflight rejection, failure/unwind after one copied lane, original
        // ledger replay unwind, then panic AFTER the actual Values destructor.
        for case in 0..5 {
            let CompletedSource {
                compiled,
                public,
                drained: [a, b, c],
                usage,
                wire,
                held,
                native_before,
            } = completed_source(case == 3);
            let exact = compiled.phases.required_bytes().unwrap();
            let lanes = compiled.binding.lanes;
            let owners = hooks::owners(&a.owner);
            let source = Arc::downgrade(&public);
            let observed = owners.clone();
            let observed_source = source.clone();
            let observed_usage = usage.clone();
            let observation = Rc::new(Cell::new(None));
            let record = observation.clone();
            drop((b, c, compiled, public));
            let retained = live(&usage);
            assert!(retained.iter().any(|&bytes| bytes > 0));
            let fault = match case {
                1 => hooks::Fault::ReadError,
                2 => hooks::Fault::ReadPanic,
                _ => hooks::Fault::None,
            };
            let _session = hooks::install(exact - usize::from(case == 0), fault, move || {
                record.set(Some((
                    observed.counts_after_erasure(),
                    observed_source.strong_count(),
                    live(&observed_usage),
                    hooks::counts(),
                )));
                if case == 4 {
                    panic!("injected source-backed output-erasure unwind");
                }
            });
            let result = std::panic::catch_unwind(AssertUnwindSafe(|| a.restore()));
            match case {
                0 => assert!(matches!(result, Ok(Err(Error::Bounds)))),
                1 => assert!(matches!(result, Ok(Err(Error::Binding)))),
                _ => assert!(result.is_err()),
            }
            if case == 0 {
                assert_eq!(hooks::counts(), (0, 0));
                assert!(observation.get().is_none());
            } else {
                let reads = if matches!(case, 1 | 2) { 2 } else { lanes };
                assert_eq!(hooks::counts(), (1, reads));
                assert_eq!(
                    observation.get(),
                    Some(([1, 1, 2], 1, retained, (1, reads)))
                );
            }
            owners.assert_released();
            assert_eq!(source.strong_count(), 0);
            assert!(source.upgrade().is_none());
            // These are the ORIGINAL runtimes' ledgers, not a new execution or
            // manufactured owner. Delivered frames remain charged on ALL exits.
            assert_eq!(live(&usage), retained);
            for (raw, (_, _, expected)) in held.iter().zip(&wire) {
                assert_eq!(raw.as_bytes(), expected);
            }
            drop(held);
            assert_eq!(live(&usage), [0; 3]);
            assert!(usage.iter().all(|u| u.peak_bytes() > 0));
            assert_eq!(native_attempts(), native_before);
            assert!(matches!(
                worker_parity_gate(),
                Err(PreparationError::UnqualifiedOriginalWorkerParity)
            ));
        }
    });
}

#[test]
fn manifest_restore_success_retains_original_workers_without_dispatch_then_replay_burns() {
    run_large_stack(|| {
        let CompletedSource {
            compiled,
            public,
            drained: [a, b, c],
            usage,
            wire,
            held,
            native_before,
        } = completed_source(false);
        let exact = compiled.phases.required_bytes().unwrap();
        let end = compiled.binding.end;
        let lanes = compiled.binding.lanes;
        let owners = hooks::owners(&a.owner);
        let observed = owners.clone();
        let source = Arc::downgrade(&public);
        let observed_source = source.clone();
        // Restore peers first: only a still retains the original Compiled Arc.
        let b = b.restore().unwrap();
        let c = c.restore().unwrap();
        drop((compiled, public));
        let retained = live(&usage);
        let observation = Rc::new(Cell::new(None));
        let record = observation.clone();
        let session = hooks::install(exact, hooks::Fault::None, move || {
            record.set(Some((
                observed.counts_after_erasure(),
                observed_source.strong_count(),
                hooks::counts(),
            )));
        });
        let a = a.restore().unwrap();
        assert_eq!(observation.get(), Some(([1, 1, 4], 1, (1, lanes))));
        assert_eq!(hooks::counts(), (1, lanes));
        drop(session);
        assert_eq!(source.strong_count(), 0);
        assert!(source.upgrade().is_none());
        assert_eq!(owners.counts_after_erasure(), [0, 0, 3]);
        assert_eq!(live(&usage), retained);
        let mut resumed = [a, b, c];
        let seam = resumed.each_ref().map(snapshot);
        for state in &seam {
            assert_eq!(state.step, end);
            assert!(state.state.1 && state.state.3 && state.state.4);
        }
        // No dispatch source Arc survives. The real protected decoder suffix
        // must instead use each returned Worker's ORIGINAL by-value source and
        // runtime. This proves success retention, not failure-path destruction.
        let (suffix, held_suffix) = drive(&mut resumed);
        assert!(!suffix.is_empty());
        for (role, before) in resumed.iter().zip(&seam) {
            let after = snapshot(role);
            assert!(role.complete() && !role.retired());
            assert!(after.cursors.aes_work > before.cursors.aes_work);
            assert!(after.cursors.pair_blocks > before.cursors.pair_blocks);
        }
        for mut role in resumed {
            // The body need not send a frame to every party. Replay an ACTUAL
            // original body or protected-suffix frame addressed to this party,
            // preserving its sender and recipient. Never manufacture traffic.
            let (from, _, raw) = wire
                .iter()
                .chain(&suffix)
                .find(|(_, to, _)| *to == role.party())
                .expect("each restored party received an original protected frame");
            assert!(role.receive(raw, *from).is_err());
            assert!(role.retired() && !role.complete());
            assert!(matches!(role.advance(), Err(Error::Retired)));
            assert!(matches!(role.receive(raw, *from), Err(Error::Retired)));
            assert!(matches!(role.finish(), Err(Error::Retired)));
        }
        owners.assert_released();
        for (raw, (_, _, expected)) in held.iter().zip(&wire) {
            assert_eq!(raw.as_bytes(), expected);
        }
        assert!(usage.iter().any(|u| u.live_bytes() > 0));
        drop((held, held_suffix));
        assert_eq!(live(&usage), [0; 3]);
        assert_eq!(native_attempts(), native_before);
        assert!(matches!(
            worker_parity_gate(),
            Err(PreparationError::UnqualifiedOriginalWorkerParity)
        ));
    });
}
