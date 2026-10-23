//! Independent real two-stage Worker/router oracle. Only this private test
//! coordinator issues capabilities, AFTER actual all-role quiescence and drain.
use super::*;
use crate::replicated_decoder::program::compact_lowering::frozen_region::prefix_custody::{
    execution_fixture::with_fixture, PendingPrefix,
};
use crate::replicated_decoder::runtime::frozen_execution_tests::{
    prefix_handoff as handoff, with_oracle, Qualified,
};
use crate::replicated_decoder::{MAX_PROGRAM_BYTES, ONE};
use puma::{Context, Party};

type Frame = (Party, Party, PeerBytes);
fn fresh(program: Arc<Program>, lanes: usize) -> [Worker; 3] {
    let session = [119; 32];
    let context = Context::new(session, program.digest(), 0).unwrap();
    let runtimes = puma::prss::insecure_test_support::fresh_compound_parties(
        context,
        program.work_budget().limits().unwrap(),
    )
    .unwrap();
    let mut workers =
        runtimes.map(|runtime| Worker::new(program.clone(), runtime, session).unwrap());
    let a: Vec<_> = (0..lanes)
        .map(|i| 0x123456789abcdef_u64.wrapping_add(i as u64 * 17))
        .collect();
    let b: Vec<_> = (0..lanes)
        .map(|i| 0xfedcba987654321_u64.wrapping_add(i as u64 * 29))
        .collect();
    let c: Vec<_> = (0..lanes)
        .map(|i| (((i % 3) as i64 - 1) * ONE / 8) as u64)
        .enumerate()
        .map(|(i, x)| x.wrapping_sub(a[i]).wrapping_sub(b[i]))
        .collect();
    let components = [a, b, c];
    for (i, worker) in workers.iter_mut().enumerate() {
        worker
            .import("state.source", &components[i], &components[(i + 1) % 3])
            .unwrap();
    }
    workers
}
fn append(
    from: Party,
    a: Vec<(Party, PeerBytes)>,
    b: Vec<(Party, PeerBytes)>,
    qa: &mut Vec<Frame>,
    qb: &mut Vec<Frame>,
) {
    assert_eq!(a.len(), b.len());
    for ((to, a), (other, b)) in a.into_iter().zip(b) {
        assert_eq!(to, other);
        assert_eq!(a.as_bytes(), b.as_bytes());
        qa.push((from, to, a));
        qb.push((from, to, b));
        assert!(qa.len() <= 64 && qb.len() <= 64);
    }
}
struct Router {
    control: [Worker; 3],
    candidate: [Worker; 3],
    qa: Vec<Frame>,
    qb: Vec<Frame>,
}
impl Router {
    fn reach(
        mut self,
        q: &Qualified<'_>,
        before: usize,
    ) -> ([Worker; 3], [CoordinatorDrained<Worker>; 3]) {
        for _ in 0..100_000 {
            if self.qa.is_empty()
                && self.qb.is_empty()
                && self
                    .candidate
                    .iter()
                    .all(|w| matches!(w.test_boundary_state(),Ok((step,_,_)) if step==before))
            {
                for i in 0..3 {
                    assert_eq!(self.candidate[i].party() as usize, i);
                    q.assert_same_state(&self.control[i], &self.candidate[i]);
                }
                // REAL owned router and all roles checked here. No static recipe,
                // caller drain bool, public test constructor, or production issuer.
                return (
                    self.control,
                    self.candidate.map(|owner| CoordinatorDrained { owner }),
                );
            }
            if self.qa.is_empty() {
                for i in 0..3 {
                    let a = match self.control[i].advance_before(before).unwrap() {
                        BoundaryProgress::Reached => Vec::new(),
                        BoundaryProgress::Frames(frames) => frames,
                    };
                    let b = q.advance(&mut self.candidate[i], Some(before)).unwrap();
                    append(self.control[i].party(), a, b, &mut self.qa, &mut self.qb);
                }
            } else {
                let (from, to, a) = self.qa.remove(0);
                let (from2, to2, b) = self.qb.remove(0);
                assert_eq!((from, to), (from2, to2));
                let a = self.control[to as usize]
                    .receive(a.as_bytes(), from)
                    .unwrap();
                let b = self.candidate[to as usize]
                    .receive(b.as_bytes(), from)
                    .unwrap();
                append(to, a, b, &mut self.qa, &mut self.qb);
            }
        }
        panic!("two-prefix router did not reach original boundary")
    }
}
fn reach(
    control: [Worker; 3],
    candidate: [Worker; 3],
    q: &Qualified<'_>,
    before: usize,
) -> ([Worker; 3], [CoordinatorDrained<Worker>; 3]) {
    Router {
        control,
        candidate,
        qa: Vec::with_capacity(64),
        qb: Vec::with_capacity(64),
    }
    .reach(q, before)
}
fn continue_original(
    control: &mut [Worker; 3],
    installed: &mut handoff::Installed<'_, '_>,
    q: &Qualified<'_>,
) {
    let (mut qa, mut qb) = (Vec::with_capacity(64), Vec::with_capacity(64));
    for _ in 0..100_000 {
        if control.iter().all(Worker::complete) {
            assert!(qa.is_empty() && qb.is_empty());
            assert!(installed.complete());
            installed.assert_same(q, control);
            return;
        }
        if qa.is_empty() {
            for (i, worker) in control.iter_mut().enumerate() {
                if !worker.complete() {
                    let a = worker.advance().unwrap();
                    let b = installed.advance(q, i, None).unwrap();
                    append(worker.party(), a, b, &mut qa, &mut qb);
                }
            }
        } else {
            let (from, to, a) = qa.remove(0);
            let (from2, to2, b) = qb.remove(0);
            assert_eq!((from, to), (from2, to2));
            let a = control[to as usize].receive(a.as_bytes(), from).unwrap();
            let b = installed.receive(to as usize, b.as_bytes(), from).unwrap();
            append(to, a, b, &mut qa, &mut qb);
        }
        installed.assert_same(q, control);
    }
    panic!("installed original continuation stalled")
}
fn peak(
    q: &Qualified<'_>,
    pending: &PendingPrefix<'_>,
    caps: &[CoordinatorDrained<Worker>; 3],
    mode: handoff::Install,
) -> usize {
    // Peek only in this private coordinator test child, never clone runtimes.
    // Quote consumes no permit. Source-field references do not extract owners.
    handoff::quote_drained(q, pending, caps, mode).unwrap()
}

#[test]
fn actual_alias_and_copy_installation_run_second_stage_same_prss_wire_and_components() {
    for lanes in [1, 128, 129, 257] {
        for mode in [handoff::Install::RetainAlias, handoff::Install::Copy] {
            with_fixture(lanes, |pending, _foreign, program, _, before| {
                with_oracle(
                    fresh(program.clone(), lanes),
                    fresh(program, lanes),
                    pending.source(),
                    MAX_PROGRAM_BYTES,
                    |control, candidate, q| {
                        let (mut control, caps) = reach(control, candidate, q, before);
                        let source = pending.mapping(0, 0)?.0;
                        let (slot, _) = q.address(source.0, source.1)?;
                        let old = caps[0]
                            .owner
                            .test_boundary_state()?
                            .2
                            .nth(slot)
                            .flatten()
                            .unwrap()
                            .0
                            .as_ptr();
                        let admitted = peak(q, &pending, &caps, mode);
                        let key = pending.destination_key(0)?;
                        let factories = handoff::COPY_FACTORIES.with(|n| n.get());
                        let mut installed = handoff::install(q, &pending, caps, mode, admitted)?;
                        assert_eq!(
                            handoff::COPY_FACTORIES.with(|n| n.get()) - factories,
                            if mode == handoff::Install::Copy { 3 } else { 0 }
                        );
                        assert_eq!(
                            installed.backing(0)? == old,
                            mode == handoff::Install::RetainAlias
                        );
                        for lane in 0..lanes {
                            let p0 = installed.word(0, &key, lane)?;
                            let p1 = installed.word(1, &key, lane)?;
                            let p2 = installed.word(2, &key, lane)?;
                            assert_eq!(p0.1, p1.0);
                            assert_eq!(p1.1, p2.0);
                            assert_eq!(p2.1, p0.0);
                        }
                        assert_eq!(installed.sticky(0)?, None); // no invented true/reset
                        continue_original(&mut control, &mut installed, q);
                        installed.word(0, &key, lanes - 1)?; // retained original export survives
                        Ok(())
                    },
                )
                .unwrap();
            });
        }
    }
}

#[test]
fn exact_short_cap_refuses_before_first_copy_factory_and_burns_attempt() {
    with_fixture(129, |pending, _, program, _, before| {
        with_oracle(
            fresh(program.clone(), 129),
            fresh(program, 129),
            pending.source(),
            MAX_PROGRAM_BYTES,
            |control, candidate, q| {
                let (_, caps) = reach(control, candidate, q, before);
                let admitted = peak(q, &pending, &caps, handoff::Install::Copy);
                let count = handoff::COPY_FACTORIES.with(|n| n.get());
                assert!(matches!(
                    handoff::install(q, &pending, caps, handoff::Install::Copy, admitted - 1),
                    Err(Error::Bounds)
                ));
                assert_eq!(handoff::COPY_FACTORIES.with(|n| n.get()), count);
                assert!(matches!(
                    pending.begin_runtime_attempt(),
                    Err(Error::Retired)
                ));
                Ok(())
            },
        )
        .unwrap();
    });
}

#[test]
fn genuine_wrong_destination_same_coordinates_burns_all_installed_siblings() {
    with_fixture(129, |pending, foreign, program, _, before| {
        with_oracle(
            fresh(program.clone(), 129),
            fresh(program, 129),
            pending.source(),
            MAX_PROGRAM_BYTES,
            |control, candidate, q| {
                let (_, caps) = reach(control, candidate, q, before);
                let mut installed = handoff::install(
                    q,
                    &pending,
                    caps,
                    handoff::Install::RetainAlias,
                    MAX_PROGRAM_BYTES,
                )?;
                assert!(matches!(
                    installed.word(0, &foreign.destination_key(0)?, 0),
                    Err(Error::Binding)
                ));
                assert!(matches!(
                    installed.word(1, &pending.destination_key(0)?, 0),
                    Err(Error::Retired)
                ));
                Ok(())
            },
        )
        .unwrap();
    });
}

#[test]
fn partial_copy_failure_never_publishes_receipt_or_replay() {
    with_fixture(129, |pending, _, program, _, before| {
        with_oracle(
            fresh(program.clone(), 129),
            fresh(program, 129),
            pending.source(),
            MAX_PROGRAM_BYTES,
            |control, candidate, q| {
                let (_, caps) = reach(control, candidate, q, before);
                let count = handoff::COPY_FACTORIES.with(|n| n.get());
                let erased = handoff::ERASED_WHILE_SOURCE_LIVE.with(|n| n.get());
                assert!(matches!(
                    handoff::install_inner(
                        q,
                        &pending,
                        caps,
                        handoff::Install::Copy,
                        MAX_PROGRAM_BYTES,
                        Some(1)
                    ),
                    Err(Error::Phase)
                ));
                assert_eq!(handoff::COPY_FACTORIES.with(|n| n.get()), count + 1);
                assert_eq!(
                    handoff::ERASED_WHILE_SOURCE_LIVE.with(|n| n.get()),
                    erased + 1
                );
                assert!(matches!(
                    pending.begin_runtime_attempt(),
                    Err(Error::Retired)
                ));
                Ok(())
            },
        )
        .unwrap();
    });
}

#[test]
fn foreign_same_digest_original_program_cannot_join_frozen_execution() {
    with_fixture(129, |pending, _, program, foreign, before| {
        assert_eq!(program.digest(), foreign.digest());
        with_oracle(
            fresh(program.clone(), 129),
            fresh(program, 129),
            pending.source(),
            MAX_PROGRAM_BYTES,
            |control, candidate, q| {
                let (control, good) = reach(control, candidate, q, before);
                drop((control, good));
                // Another genuine original Program, SAME native Arcs/hash and
                // coordinates, actually qualified and driven to the boundary.
                // No absent oracle_values/local-ready shortcut wins this test.
                with_oracle(
                    fresh(foreign.clone(), 129),
                    fresh(foreign, 129),
                    pending.source(),
                    MAX_PROGRAM_BYTES,
                    |control, candidate, foreign_q| {
                        let (_, wrong) = reach(control, candidate, foreign_q, before);
                        assert!(matches!(
                            handoff::install(
                                q,
                                &pending,
                                wrong,
                                handoff::Install::Copy,
                                MAX_PROGRAM_BYTES
                            ),
                            Err(Error::Binding)
                        ));
                        Ok(())
                    },
                )
            },
        )
        .unwrap();
    });
}

#[test]
fn installed_owner_lifetime_cancel_and_successful_transfer_consume_original_permit() {
    let (weak, kernel) = with_fixture(129, |pending, _, program, _, before| {
        let weak = Arc::downgrade(pending.source().execution_origin());
        let kernel = Arc::downgrade(pending.source().prepared(0).unwrap().program_at(0).unwrap());
        with_oracle(
            fresh(program.clone(), 129),
            fresh(program, 129),
            pending.source(),
            MAX_PROGRAM_BYTES,
            |control, candidate, q| {
                let (_, caps) = reach(control, candidate, q, before);
                let installed =
                    handoff::install(q, &pending, caps, handoff::Install::Copy, MAX_PROGRAM_BYTES)?;
                assert!(weak.upgrade().is_some());
                assert!(kernel.upgrade().is_some());
                assert!(matches!(
                    pending.begin_runtime_attempt(),
                    Err(Error::Retired)
                ));
                drop(installed);
                assert!(weak.upgrade().is_some());
                assert!(kernel.upgrade().is_some());
                Ok(())
            },
        )
        .unwrap();
        (weak, kernel)
    });
    assert!(weak.upgrade().is_none());
    assert!(kernel.upgrade().is_none());
}

#[test]
fn actual_installed_unwind_erases_shares_before_source_owner_and_burns_epoch() {
    with_fixture(129, |pending, _, program, _, before| {
        with_oracle(
            fresh(program.clone(), 129),
            fresh(program, 129),
            pending.source(),
            MAX_PROGRAM_BYTES,
            |control, candidate, q| {
                let (_, caps) = reach(control, candidate, q, before);
                let erased = handoff::ERASED_WHILE_SOURCE_LIVE.with(|n| n.get());
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _installed = handoff::install(
                        q,
                        &pending,
                        caps,
                        handoff::Install::Copy,
                        MAX_PROGRAM_BYTES,
                    )
                    .unwrap();
                    panic!("injected caller unwind AFTER actual installation");
                }));
                assert!(result.is_err());
                assert_eq!(
                    handoff::ERASED_WHILE_SOURCE_LIVE.with(|n| n.get()),
                    erased + 1
                );
                assert!(matches!(
                    pending.begin_runtime_attempt(),
                    Err(Error::Retired)
                ));
                Ok(())
            },
        )
        .unwrap();
    });
}

#[test]
fn spent_transition_rejects_another_genuine_routed_execution_not_only_a_marker() {
    with_fixture(129, |pending, _, program, _, before| {
        for attempt in 0..2 {
            with_oracle(
                fresh(program.clone(), 129),
                fresh(program.clone(), 129),
                pending.source(),
                MAX_PROGRAM_BYTES,
                |control, candidate, q| {
                    let (_, caps) = reach(control, candidate, q, before);
                    let copies = handoff::COPY_FACTORIES.with(|n| n.get());
                    let result = handoff::install(
                        q,
                        &pending,
                        caps,
                        handoff::Install::Copy,
                        MAX_PROGRAM_BYTES,
                    );
                    if attempt == 0 {
                        drop(result?);
                    } else {
                        assert!(matches!(result, Err(Error::Retired)));
                        assert_eq!(handoff::COPY_FACTORIES.with(|n| n.get()), copies);
                    }
                    Ok(())
                },
            )
            .unwrap();
        }
    });
}

#[test]
fn swapping_real_drained_role_capabilities_refuses_before_any_copy() {
    with_fixture(129, |pending, _, program, _, before| {
        with_oracle(
            fresh(program.clone(), 129),
            fresh(program, 129),
            pending.source(),
            MAX_PROGRAM_BYTES,
            |control, candidate, q| {
                let (_, caps) = reach(control, candidate, q, before);
                let [a, b, c] = caps;
                let copies = handoff::COPY_FACTORIES.with(|n| n.get());
                assert!(matches!(
                    handoff::install(
                        q,
                        &pending,
                        [b, a, c],
                        handoff::Install::Copy,
                        MAX_PROGRAM_BYTES
                    ),
                    Err(Error::Binding)
                ));
                assert_eq!(handoff::COPY_FACTORIES.with(|n| n.get()), copies);
                Ok(())
            },
        )
        .unwrap();
    });
}
