//! Independent scalar/window provider conformance using real, fresh issuance.
//!
//! Mount as a child of `composition::provider`, not of its existing tests module.
//! Each run gets new material and its own request-long guards. No packet/key is
//! replayed into a fresh guard to force equal random output shares or wire bytes.
//! We compare final reconstructed integers and PUBLIC peer transcript structure;
//! exact randomized peer payload/material bytes are intentionally NOT proved equal.
//!
//! Public-path positive control for the lifetime probes below. To collect these
//! doctests, mount this file with `#[cfg(any(test, doctest))]`, not just cfg(test).
//! No compiler is invoked by a unit test, and no private field/path is the reason
//! the negative examples must fail.
//! Unit fixtures/imports live in a cfg(test)-only child; rustdoc collection does
//! not compile them or require any test-only native helper.
//!
//! ```
//! use pllm_garble::fusefss_backend::{Budget, composition::provider::ReplyWindow};
//! use pllm_garble::shared_allocation::OwnedBudget;
//! let budget = Budget::new(1 << 20).unwrap();
//! let mut window = ReplyWindow::new(&budget, 0usize).unwrap();
//! window.push(0, 1).unwrap();
//! window.provide(|_, replies| replies[0].write(&[7])).unwrap();
//! assert_eq!(window.bytes(0).unwrap(), &[7]);
//! drop(window);
//! assert_eq!(budget.used_bytes(), 0);
//! let owned = OwnedBudget::new(1 << 20).unwrap();
//! let payload = owned.charge_external(0).unwrap().try_build(|budget| {
//!     let window = ReplyWindow::new(budget, 0usize)?;
//!     Ok(window.requests().len()) // An owned integer may escape, not the window.
//! }).unwrap();
//! drop(owned);
//! assert_eq!(*payload.payload(), 0);
//! ```
//!
//! The actual ReplyWindow cannot outlive its original borrowed counter:
//!
//! ```compile_fail
//! use pllm_garble::fusefss_backend::{Budget, composition::provider::ReplyWindow};
//! fn escape<'a>() -> ReplyWindow<'a, usize> {
//!     let budget = Budget::new(1 << 20).unwrap();
//!     let mut window = ReplyWindow::new(&budget, 0usize).unwrap();
//!     window.push(0, 1).unwrap();
//!     window // Returning this would leave the lease referring to a dead Budget.
//! }
//! ```
//!
//! An owning shared accounting wrapper does NOT upgrade that native borrow:
//!
//! ```compile_fail
//! use pllm_garble::fusefss_backend::composition::provider::ReplyWindow;
//! use pllm_garble::shared_allocation::OwnedBudget;
//! let owned = OwnedBudget::new(1 << 20).unwrap();
//! let payload = owned.charge_external(0).unwrap()
//!     .try_build(|budget| ReplyWindow::new(budget, 0usize)).unwrap();
//! drop(owned);
//! payload.payload().requests();
//! ```
#[cfg(test)]
mod unit {
    use super::super::{
        Reply, ReplyView, ReplyWindow, MATERIAL_WINDOW_BYTES, MATERIAL_WINDOW_RECORDS,
    };
    use crate::fusefss_backend::composition::{
        cache, gates, kernels, wavefront, wire, Event, Plan,
    };
    use crate::fusefss_backend::{Budget, Error, Output, MAX_ALLOCATION_BYTES};
    use crate::shared_rescale_reference::Context;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::atomic::Ordering;

    const MASK: u64 = (1u64 << 50) - 1;

    fn context(plan: &Plan<'_>, session: u8) -> Context {
        Context {
            plan: plan.digest(),
            session: [session; 32],
            operation: [5; 32],
            first_tensor_index: 0,
        }
    }

    fn binding() -> kernels::Binding {
        kernels::Binding {
            source_digest: [21; 32],
            contract_digest: [22; 32],
        }
    }

    fn floor_plan(budget: &Budget, lanes: usize) -> Plan<'_> {
        kernels::floor(
            binding(),
            gates::InputRange {
                min: -100_000,
                max: 100_000,
            },
            12,
            budget,
        )
        .unwrap()
        .0
        .repeat_tensor(lanes)
        .unwrap()
    }

    // Same deterministic INPUT SHARE fixture as wavefront_tests; native dealer
    // issuance still uses its ordinary entropy path, independently for each run.
    fn input_shares(values: &[u64]) -> [Vec<u64>; 2] {
        let left: Vec<_> = values
            .iter()
            .enumerate()
            .map(|(i, _)| (0x3857_fafb_9185 + i as u64 * 37) & MASK)
            .collect();
        let right = values
            .iter()
            .zip(&left)
            .map(|(&value, &share)| value.wrapping_sub(share) & MASK)
            .collect();
        [left, right]
    }

    #[derive(Debug, PartialEq, Eq)]
    struct PublicFrame {
        role: u8,
        round: u64,
        records: Vec<(usize, usize)>,
        bytes: usize,
    }

    fn public_frame(bytes: &[u8], role: u8) -> PublicFrame {
        assert!(bytes.len() >= 20);
        assert_eq!(&bytes[..4], b"FFW1");
        assert_eq!(&bytes[4..8], &[role, 0, 0, 0]);
        let round = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let count = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
        let mut offset = 20;
        let mut records = Vec::new();
        for _ in 0..count {
            let header = bytes.get(offset..offset + 8).unwrap();
            let pc = u32::from_le_bytes(header[..4].try_into().unwrap()) as usize;
            let length = u32::from_le_bytes(header[4..].try_into().unwrap()) as usize;
            assert!(length > 0);
            offset += 8;
            assert!(bytes.get(offset..offset + length).is_some());
            offset += length;
            records.push((pc, length));
        }
        assert_eq!(offset, bytes.len());
        PublicFrame {
            role,
            round,
            records,
            bytes: bytes.len(),
        }
    }

    #[derive(Debug)]
    struct Observed {
        output: Vec<u64>,
        output_words: [usize; 2],
        // PC, original public bound, actual exported packet length. No secret bytes.
        material: [Vec<(usize, usize, usize)>; 2],
        frames: [Vec<PublicFrame>; 2],
        callbacks: [usize; 2],
        windows: [Vec<usize>; 2],
        // Public numeric work quote, not instrumented cryptographic execution work.
        declared_work: [usize; 8],
    }

    impl Observed {
        fn new(plan: &Plan<'_>) -> Self {
            let r = plan.resources();
            Self {
                output: Vec::new(),
                output_words: [0; 2],
                material: std::array::from_fn(|_| Vec::new()),
                frames: std::array::from_fn(|_| Vec::new()),
                callbacks: [0; 2],
                windows: std::array::from_fn(|_| Vec::new()),
                declared_work: [
                    r.cumulative_work,
                    r.dcf_keys_per_party,
                    r.dcf_evaluations_per_party,
                    r.triples_per_party,
                    r.private_material_bytes_total,
                    r.public_table_copy_bytes_total,
                    r.frames_per_party,
                    r.frame_bytes_per_party,
                ],
            }
        }

        fn finish(&mut self, plan: &Plan<'_>, a: Output<'_>, b: Output<'_>) {
            self.output_words = [a.shares().len(), b.shares().len()];
            assert_eq!(self.output_words, [plan.output_count(); 2]);
            self.output = a
                .shares()
                .iter()
                .zip(b.shares())
                .enumerate()
                .map(|(i, (&x, &y))| {
                    if plan.output_is_xor(i).unwrap() {
                        x ^ y
                    } else {
                        x.wrapping_add(y) & ((1u64 << plan.output_bits(i).unwrap()) - 1)
                    }
                })
                .collect();
        }
    }

    // Independent oracle path: one Reply factory, one callback and one import per
    // original NeedMaterial. It never calls ReplyWindow or pending_materials.
    fn scalar<'a>(
        plan: &Plan<'a>,
        schedule: &wavefront::Schedule<'_, 'a>,
        values: &[u64],
        budget: &'a Budget,
    ) -> Observed {
        let baseline = budget.used_bytes();
        let ctx = context(plan, 41);
        let [left, right] = input_shares(values);
        let mut a = schedule.runtime(0, ctx, &left).unwrap();
        let mut b = schedule.runtime(1, ctx, &right).unwrap();
        let mut guards = [
            wire::ImportGuard::new_indexed(plan.resources().secret_steps.max(1), budget).unwrap(),
            wire::ImportGuard::new_indexed(plan.resources().secret_steps.max(1), budget).unwrap(),
        ];
        let mut observed = Observed::new(plan);
        let mut events = (a.poll().unwrap(), b.poll().unwrap());
        loop {
            events = match events {
                (Event::NeedMaterial { pc: x }, Event::NeedMaterial { pc: y }) => {
                    assert_eq!(x, y);
                    let bound = plan.step_transport_resources(x).unwrap().packet_bytes_bound;
                    let [ma, mb] = plan.issue_step(x, ctx).unwrap();
                    let packets = [ma.export_packet().unwrap(), mb.export_packet().unwrap()];
                    let mut next = Vec::new();
                    for (role, runtime) in [&mut a, &mut b].into_iter().enumerate() {
                        observed.callbacks[role] += 1;
                        observed.material[role].push((x, bound, packets[role].as_bytes().len()));
                        // Reply::with_bound retains the scalar caller's quoting duty.
                        let charge = budget.reserve(bound).unwrap();
                        let event = Reply::with_bound(bound, |reply| {
                            reply.write(packets[role].as_bytes())?;
                            runtime.feed_result(plan.import_step_bytes(
                                x,
                                ctx,
                                role as u8,
                                packets[role].nonce(),
                                reply.bytes()?,
                                &mut guards[role],
                            ))
                        })
                        .unwrap();
                        drop(charge);
                        next.push(event);
                    }
                    let mut next = next.into_iter();
                    (next.next().unwrap(), next.next().unwrap())
                }
                (Event::Frame(fa), Event::Frame(fb)) => {
                    observed.frames[0].push(public_frame(fa.as_bytes(), 0));
                    observed.frames[1].push(public_frame(fb.as_bytes(), 1));
                    let next = (
                        a.advance(fb.as_bytes()).unwrap(),
                        b.advance(fa.as_bytes()).unwrap(),
                    );
                    drop((fa, fb));
                    next
                }
                (Event::Output(oa), Event::Output(ob)) => {
                    assert!(a.is_consumed() && b.is_consumed());
                    observed.finish(plan, oa, ob);
                    break;
                }
                _ => panic!("scalar roles diverged"),
            };
        }
        drop((a, b, guards));
        assert_eq!(budget.used_bytes(), baseline);
        observed
    }

    // Only this driver uses frozen-group observation and charged ReplyWindow owners.
    fn windowed<'a>(
        plan: &Plan<'a>,
        schedule: &wavefront::Schedule<'_, 'a>,
        values: &[u64],
        budget: &'a Budget,
    ) -> Observed {
        let baseline = budget.used_bytes();
        let ctx = context(plan, 42); // Fresh run domain as well as fresh native entropy.
        let [left, right] = input_shares(values);
        let mut a = schedule.runtime(0, ctx, &left).unwrap();
        let mut b = schedule.runtime(1, ctx, &right).unwrap();
        let mut guards = [
            wire::ImportGuard::new_indexed(plan.resources().secret_steps.max(1), budget).unwrap(),
            wire::ImportGuard::new_indexed(plan.resources().secret_steps.max(1), budget).unwrap(),
        ];
        let mut observed = Observed::new(plan);
        let mut events = (a.poll().unwrap(), b.poll().unwrap());
        loop {
            events = match events {
                (Event::NeedMaterial { pc: x }, Event::NeedMaterial { pc: y }) => {
                    assert_eq!(x, y);
                    let mut windows = [
                        ReplyWindow::new(budget, usize::MAX).unwrap(),
                        ReplyWindow::new(budget, usize::MAX).unwrap(),
                    ];
                    let ca = a.pending_materials(windows[0].pcs_mut()).unwrap();
                    let cb = b.pending_materials(windows[1].pcs_mut()).unwrap();
                    assert_eq!(ca, cb);
                    assert_eq!(windows[0].pc(0).unwrap(), x);
                    let mut payload = 0;
                    let mut packets = Vec::new();
                    let mut nonces = Vec::new();
                    for i in 0..ca {
                        let pc = windows[0].pc(i).unwrap();
                        assert_eq!(pc, windows[1].pc(i).unwrap());
                        let bound = plan
                            .step_transport_resources(pc)
                            .unwrap()
                            .packet_bytes_bound;
                        if payload + bound > MATERIAL_WINDOW_BYTES {
                            break;
                        }
                        payload += bound;
                        for window in &mut windows {
                            window.push(pc, bound).unwrap();
                        }
                        let [ma, mb] = plan.issue_step(pc, ctx).unwrap();
                        let pair = [ma.export_packet().unwrap(), mb.export_packet().unwrap()];
                        nonces.push([pair[0].nonce(), pair[1].nonce()]);
                        for (role, packet) in pair.iter().enumerate() {
                            observed.material[role].push((pc, bound, packet.as_bytes().len()));
                        }
                        packets.push(pair);
                    }
                    let count = packets.len();
                    assert!((1..=MATERIAL_WINDOW_RECORDS).contains(&count));
                    assert!(payload <= MATERIAL_WINDOW_BYTES);
                    assert_eq!(windows[0].requests(), windows[1].requests());
                    for (role, window) in windows.iter_mut().enumerate() {
                        observed.callbacks[role] += 1;
                        observed.windows[role].push(count);
                        window
                            .provide(|requests, replies| {
                                assert_eq!(requests.len(), count);
                                assert_eq!(replies.len(), count);
                                for (i, reply) in replies.iter_mut().enumerate() {
                                    assert!(packets[i][role].as_bytes().len() <= reply.bound());
                                    reply.write(packets[i][role].as_bytes())?;
                                }
                                Ok(())
                            })
                            .unwrap();
                    }
                    // Native packet owners retire; every reply stays under its window
                    // lease through all imports, never detached/reassigned or cloned.
                    drop(packets);
                    let mut next = (Event::NeedMaterial { pc: x }, Event::NeedMaterial { pc: y });
                    for (i, nonce) in nonces.iter().enumerate() {
                        let pc = windows[0].requests()[i];
                        assert!(
                            matches!(next.0, Event::NeedMaterial { pc: actual } if actual == pc)
                        );
                        assert!(
                            matches!(next.1, Event::NeedMaterial { pc: actual } if actual == pc)
                        );
                        next = (
                            a.feed_result(plan.import_step_bytes(
                                pc,
                                ctx,
                                0,
                                nonce[0],
                                windows[0].bytes(i).unwrap(),
                                &mut guards[0],
                            ))
                            .unwrap(),
                            b.feed_result(plan.import_step_bytes(
                                pc,
                                ctx,
                                1,
                                nonce[1],
                                windows[1].bytes(i).unwrap(),
                                &mut guards[1],
                            ))
                            .unwrap(),
                        );
                    }
                    drop(windows);
                    next
                }
                (Event::Frame(fa), Event::Frame(fb)) => {
                    assert!(a
                        .pending_materials(&mut [0; MATERIAL_WINDOW_RECORDS])
                        .is_err());
                    assert!(b
                        .pending_materials(&mut [0; MATERIAL_WINDOW_RECORDS])
                        .is_err());
                    observed.frames[0].push(public_frame(fa.as_bytes(), 0));
                    observed.frames[1].push(public_frame(fb.as_bytes(), 1));
                    let next = (
                        a.advance(fb.as_bytes()).unwrap(),
                        b.advance(fa.as_bytes()).unwrap(),
                    );
                    drop((fa, fb));
                    next
                }
                (Event::Output(oa), Event::Output(ob)) => {
                    assert!(a.is_consumed() && b.is_consumed());
                    observed.finish(plan, oa, ob);
                    break;
                }
                _ => panic!("windowed roles diverged"),
            };
        }
        drop((a, b, guards));
        assert_eq!(budget.used_bytes(), baseline);
        observed
    }

    fn assert_equivalent(
        plan: &Plan<'_>,
        schedule: &wavefront::Schedule<'_, '_>,
        scalar: &Observed,
        batch: &Observed,
        expected: &[u64],
    ) {
        assert_eq!(scalar.output, expected);
        assert_eq!(batch.output, expected); // Independent integer oracle, not batch output.
        assert_eq!(scalar.output_words, batch.output_words);
        assert_eq!(scalar.material, batch.material);
        assert_eq!(scalar.frames, batch.frames); // PUBLIC structure only, never payload.
        assert_eq!(scalar.declared_work, batch.declared_work);
        let order: Vec<_> = schedule.material_steps().collect();
        for role in 0..2 {
            assert_eq!(scalar.callbacks[role], plan.resources().secret_steps);
            assert_eq!(batch.windows[role].iter().sum::<usize>(), order.len());
            assert_eq!(
                batch.material[role]
                    .iter()
                    .map(|item| item.0)
                    .collect::<Vec<_>>(),
                order
            );
            assert_eq!(batch.frames[role].len(), schedule.resources().rounds);
            assert_eq!(
                batch.frames[role]
                    .iter()
                    .map(|frame| frame.records.len())
                    .sum::<usize>(),
                plan.resources().frames_per_party,
            );
            assert_eq!(
                batch.frames[role]
                    .iter()
                    .map(|frame| frame.bytes)
                    .sum::<usize>(),
                schedule.resources().frame_bytes_per_party,
            );
            assert!(batch.callbacks[role] <= scalar.callbacks[role]);
        }
    }

    #[test]
    fn independent_scalar_and_window_providers_preserve_floor_outputs_and_public_transcript() {
        let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
        for lanes in [1, 8, 17] {
            let plan = floor_plan(&budget, lanes);
            let digest = plan.digest();
            let schedule = wavefront::Schedule::new(&plan).unwrap();
            let signed: Vec<i64> = (0..lanes)
                .map(|i| [-100_000, -4097, -4096, -1, 0, 1, 4095, 4096, 100_000][i % 9])
                .collect();
            let values: Vec<_> = signed.iter().map(|&value| value as u64 & MASK).collect();
            let expected: Vec<_> = signed
                .iter()
                .map(|value| value.div_euclid(4096) as u64 & MASK)
                .collect();
            let scalar = scalar(&plan, &schedule, &values, &budget);
            let batch = windowed(&plan, &schedule, &values, &budget);
            assert_equivalent(&plan, &schedule, &scalar, &batch, &expected);
            let counts = match lanes {
                1 => vec![1],
                8 => vec![8],
                17 => vec![8, 8, 1],
                _ => unreachable!(),
            };
            for role in 0..2 {
                assert_eq!(batch.windows[role], counts);
                if lanes > 1 {
                    assert!(batch.callbacks[role] < scalar.callbacks[role]);
                }
            }
            assert_eq!(plan.digest(), digest);
            drop(schedule);
            drop(plan);
            assert_eq!(budget.used_bytes(), 0);
        }
    }

    #[test]
    fn complete_private_dots_keep_products_accumulation_and_floor_with_fewer_crossings() {
        let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
        let range = gates::InputRange {
            min: -100,
            max: 100,
        };
        let plan = kernels::private_dot(binding(), &[range; 2], &[range; 2], &budget)
            .unwrap()
            .0
            .repeat_tensor(17)
            .unwrap();
        let schedule = wavefront::Schedule::new(&plan).unwrap();
        let mut values = Vec::new();
        let mut expected = Vec::new();
        for i in 0..17i64 {
            let [a, b, c, d] = [i - 8, 100 - i, -100 + i, if i % 2 == 0 { -1 } else { 1 }];
            values.extend([a, b, c, d].map(|value| value as u64 & MASK));
            expected.push((a * c + b * d).div_euclid(4096) as u64 & MASK);
        }
        let scalar = scalar(&plan, &schedule, &values, &budget);
        let batch = windowed(&plan, &schedule, &values, &budget);
        assert_equivalent(&plan, &schedule, &scalar, &batch, &expected);
        assert!(plan.resources().triples_per_party > 0);
        assert!(plan.resources().dcf_evaluations_per_party > 0);
        for role in 0..2 {
            assert!(batch.callbacks[role] < scalar.callbacks[role]);
            assert!(batch.windows[role].contains(&8));
            assert_eq!(batch.windows[role].last(), Some(&1));
        }
        drop(schedule);
        drop(plan);
        assert_eq!(budget.used_bytes(), 0);
    }

    #[test]
    fn local_only_approved_composition_has_no_material_callbacks_or_peer_frames() {
        let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
        let plan = kernels::floor(binding(), gates::InputRange { min: -1, max: 1 }, 0, &budget)
            .unwrap()
            .0;
        let schedule = wavefront::Schedule::new(&plan).unwrap();
        let scalar = scalar(&plan, &schedule, &[MASK], &budget);
        let batch = windowed(&plan, &schedule, &[MASK], &budget);
        assert_equivalent(&plan, &schedule, &scalar, &batch, &[MASK]);
        assert_eq!(batch.callbacks, [0, 0]);
        assert!(batch.windows.iter().all(Vec::is_empty));
        // Empty provider invocation is itself invalid and permanently burns its owner.
        let mut empty = ReplyWindow::new(&budget, 0usize).unwrap();
        assert_eq!(
            empty.provide(|_, _| panic!("empty window called provider")),
            Err(Error::Context)
        );
        assert_eq!(
            empty.provide(|_, _| panic!("empty window retried")),
            Err(Error::Consumed)
        );
        drop(empty);
        drop(schedule);
        drop(plan);
        assert_eq!(budget.used_bytes(), 0);
    }

    #[derive(Clone, Copy, Debug)]
    enum CallbackFault {
        MissingSuffix,
        ErrorAfterPrefix,
        DoubleWrite,
        Cancel,
        Unwind,
    }

    #[test]
    fn partial_error_cancel_and_unwind_hide_every_reply_until_complete_publication() {
        for role in 0..2u8 {
            for fault in [
                CallbackFault::MissingSuffix,
                CallbackFault::ErrorAfterPrefix,
                CallbackFault::DoubleWrite,
                CallbackFault::Cancel,
                CallbackFault::Unwind,
            ] {
                let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
                let plan = floor_plan(&budget, 3);
                let schedule = wavefront::Schedule::new(&plan).unwrap();
                let ctx = context(&plan, 51);
                let mut runtime = schedule.runtime(role, ctx, &[0; 3]).unwrap();
                assert!(matches!(
                    runtime.poll().unwrap(),
                    Event::NeedMaterial { .. }
                ));
                let guard = wire::ImportGuard::new_indexed(3, &budget).unwrap();
                let mut pcs = [0; MATERIAL_WINDOW_RECORDS];
                let count = runtime.pending_materials(&mut pcs).unwrap();
                assert_eq!(count, 3);
                let packets: Vec<_> = pcs[..count]
                    .iter()
                    .map(|&pc| {
                        let [a, b] = plan.issue_step(pc, ctx).unwrap();
                        if role == 0 {
                            drop(b);
                            a.export_packet().unwrap()
                        } else {
                            drop(a);
                            b.export_packet().unwrap()
                        }
                    })
                    .collect();
                let baseline = budget.used_bytes();
                let charge = ReplyWindow::<usize>::reservation_bytes().unwrap();
                let mut window = ReplyWindow::new(&budget, 0usize).unwrap();
                for &pc in &pcs[..count] {
                    window
                        .push(
                            pc,
                            plan.step_transport_resources(pc)
                                .unwrap()
                                .packet_bytes_bound,
                        )
                        .unwrap();
                }
                let result = catch_unwind(AssertUnwindSafe(|| {
                    window.provide(|_, replies| {
                        replies[0].write(packets[0].as_bytes())?;
                        assert_eq!(budget.used_bytes(), baseline + charge);
                        match fault {
                            CallbackFault::MissingSuffix => Ok(()),
                            CallbackFault::ErrorAfterPrefix => Err(Error::Context),
                            CallbackFault::DoubleWrite => replies[0].write(packets[0].as_bytes()),
                            CallbackFault::Cancel => {
                                runtime.cancel();
                                Err(Error::Consumed)
                            }
                            CallbackFault::Unwind => std::panic::panic_any(73u64),
                        }
                    })
                }));
                match fault {
                    CallbackFault::Unwind => {
                        assert_eq!(*result.unwrap_err().downcast::<u64>().unwrap(), 73)
                    }
                    CallbackFault::ErrorAfterPrefix => {
                        assert_eq!(result.unwrap(), Err(Error::Context))
                    }
                    _ => assert_eq!(result.unwrap(), Err(Error::Consumed)),
                }
                for index in 0..count {
                    assert_eq!(
                        window.bytes(index),
                        Err(Error::Consumed),
                        "{role} {fault:?}"
                    );
                }
                assert_eq!(window.push(99, 1), Err(Error::Consumed));
                assert_eq!(
                    window.provide(|_, _| panic!("failed callback retried")),
                    Err(Error::Consumed)
                );
                // No importer ran: the real runtime still awaits its original group.
                // Unwind/callback errors require the owning adapter to cancel/burn it;
                // ReplyWindow itself has no invented runtime-cancellation facility.
                if !matches!(fault, CallbackFault::Cancel) {
                    assert_eq!(runtime.pending_materials(&mut pcs).unwrap(), count);
                    assert!(matches!(
                        runtime.feed_result(Err(Error::Context)),
                        Err(Error::Context)
                    ));
                }
                assert!(runtime.is_consumed());
                assert!(matches!(runtime.poll(), Err(Error::Consumed)));
                assert_eq!(budget.used_bytes(), baseline + charge);
                drop(window);
                assert_eq!(budget.used_bytes(), baseline);
                drop((packets, runtime, guard));
                drop(schedule);
                drop(plan);
                assert_eq!(budget.used_bytes(), 0);
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum ImportFault {
        ShortSecond,
        Reordered,
        WrongRoleSecond,
        WrongContextSecond,
        Replay,
    }

    #[test]
    fn malformed_imports_burn_runtime_and_request_guard_without_half_output_or_suffix_retry() {
        for role in 0..2u8 {
            for fault in [
                ImportFault::ShortSecond,
                ImportFault::Reordered,
                ImportFault::WrongRoleSecond,
                ImportFault::WrongContextSecond,
                ImportFault::Replay,
            ] {
                let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
                let plan = floor_plan(&budget, 3);
                let schedule = wavefront::Schedule::new(&plan).unwrap();
                let ctx = context(&plan, 61);
                let mut runtime = schedule.runtime(role, ctx, &[0; 3]).unwrap();
                assert!(matches!(
                    runtime.poll().unwrap(),
                    Event::NeedMaterial { .. }
                ));
                let mut guard = wire::ImportGuard::new_indexed(4, &budget).unwrap();
                let mut window = ReplyWindow::new(&budget, 0usize).unwrap();
                let count = runtime.pending_materials(window.pcs_mut()).unwrap();
                assert_eq!(count, 3);
                let mut packets = Vec::new();
                for i in 0..count {
                    let pc = window.pc(i).unwrap();
                    window
                        .push(
                            pc,
                            plan.step_transport_resources(pc)
                                .unwrap()
                                .packet_bytes_bound,
                        )
                        .unwrap();
                    let mut issued_context = ctx;
                    if i == 1 && matches!(fault, ImportFault::WrongContextSecond) {
                        issued_context.operation = [8; 32];
                    }
                    let [a, b] = plan.issue_step(pc, issued_context).unwrap();
                    let recipient = if i == 1 && matches!(fault, ImportFault::WrongRoleSecond) {
                        1 - role
                    } else {
                        role
                    };
                    packets.push(if recipient == 0 {
                        drop(b);
                        a.export_packet().unwrap()
                    } else {
                        drop(a);
                        b.export_packet().unwrap()
                    });
                }
                window
                    .provide(|_, replies| {
                        for (i, reply) in replies.iter_mut().enumerate() {
                            let source = if matches!(fault, ImportFault::Reordered) {
                                count - 1 - i
                            } else {
                                i
                            };
                            let raw = packets[source].as_bytes();
                            let length = raw.len()
                                - usize::from(i == 1 && matches!(fault, ImportFault::ShortSecond));
                            reply.write(&raw[..length])?;
                        }
                        Ok(())
                    })
                    .unwrap();
                let mut accepted = 0;
                let mut failed = None;
                for i in 0..count {
                    let index = if i == 1 && matches!(fault, ImportFault::Replay) {
                        0
                    } else {
                        i
                    };
                    let pc = window.requests()[index];
                    let material = plan.import_step_bytes(
                        pc,
                        ctx,
                        role,
                        packets[index].nonce(),
                        window.bytes(index).unwrap(),
                        &mut guard,
                    );
                    match runtime.feed_result(material) {
                        Ok(event) => {
                            assert!(matches!(event, Event::NeedMaterial { .. }));
                            accepted += 1;
                        }
                        Err(error) => {
                            failed = Some((index, error));
                            break;
                        }
                    }
                }
                assert_eq!(
                    accepted,
                    if matches!(fault, ImportFault::Reordered) {
                        0
                    } else {
                        1
                    }
                );
                let (index, error) = failed.expect("malformed window produced no import rejection");
                assert_eq!(
                    error,
                    if matches!(fault, ImportFault::Replay) {
                        Error::Consumed
                    } else {
                        Error::Context
                    }
                );
                assert!(runtime.is_consumed());
                assert!(matches!(runtime.poll(), Err(Error::Consumed)));
                assert!(runtime.pending_materials(&mut [0; 8]).is_err());
                // Keep SAME guard after failure. Even a corrected retry must lose its
                // original claim; no fresh guard, key cloning, or bypass import path.
                assert!(matches!(
                    plan.import_step_bytes(
                        window.requests()[index],
                        ctx,
                        role,
                        packets[index].nonce(),
                        packets[index].as_bytes(),
                        &mut guard,
                    ),
                    Err(Error::Consumed)
                ));
                assert_eq!(
                    window.provide(|_, _| panic!("completed window reprovided")),
                    Err(Error::Consumed)
                );
                drop((window, packets, runtime, guard));
                drop(schedule);
                drop(plan);
                assert_eq!(budget.used_bytes(), 0);
            }
        }
    }

    #[test]
    fn every_window_factory_failure_preserves_real_source_and_packet_counter() {
        let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
        let plan = floor_plan(&budget, 3);
        let ctx = context(&plan, 71);
        let pcs: Vec<_> = plan.material_steps().collect();
        let packets: Vec<_> = pcs
            .iter()
            .map(|&pc| {
                let [a, b] = plan.issue_step(pc, ctx).unwrap();
                drop(b);
                a.export_packet().unwrap()
            })
            .collect();
        let baseline = budget.used_bytes();
        // Window Vec, three Reply factories, then guarded provider-view Vec.
        for fail in 0..=5 {
            budget.fail_after.store(fail, Ordering::SeqCst);
            let mut crossings = 0;
            let result = (|| {
                let mut window = ReplyWindow::new(&budget, 0usize)?;
                for &pc in &pcs {
                    window.push(pc, plan.step_transport_resources(pc)?.packet_bytes_bound)?;
                }
                window.provide(|_, replies| {
                    crossings += 1;
                    for (reply, packet) in replies.iter_mut().zip(&packets) {
                        reply.write(packet.as_bytes())?;
                    }
                    Ok(())
                })
            })();
            budget.fail_after.store(-1, Ordering::SeqCst);
            assert_eq!(
                result,
                if fail < 5 {
                    Err(Error::Allocation)
                } else {
                    Ok(())
                }
            );
            assert_eq!(crossings, usize::from(fail == 5));
            assert_eq!(budget.used_bytes(), baseline, "factory {fail}");
        }
        drop((packets, plan));
        assert_eq!(budget.used_bytes(), 0);
    }

    #[test]
    fn window_refund_uses_original_counter_while_real_cached_source_and_packets_stay_live() {
        for separate in [false, true] {
            let source_budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
            let receive_budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
            let receiver = if separate {
                &receive_budget
            } else {
                &source_budget
            };
            let source = cache::SourceTableCache::new(
                [7; 32],
                cache::Limits {
                    max_entries: cache::MAX_ENTRIES,
                    max_retained_bytes: cache::MAX_RETAINED_BYTES,
                },
                &source_budget,
            )
            .unwrap();
            let plan = gates::silu_cached(
                gates::InputRange {
                    min: -8192,
                    max: 8192,
                },
                &source,
            )
            .unwrap()
            .0;
            let public_baseline = source_budget.used_bytes();
            let pc = plan.material_steps().next().unwrap();
            let [a, b] = plan.issue_step(pc, context(&plan, 81)).unwrap();
            drop(b);
            let packet = a.export_packet().unwrap();
            let source_baseline = source_budget.used_bytes();
            assert!(source_baseline > public_baseline);
            let receive_baseline = receiver.used_bytes();
            let required = ReplyWindow::<usize>::reservation_bytes().unwrap();
            let mut window = ReplyWindow::new(receiver, pc).unwrap();
            window
                .push(
                    pc,
                    plan.step_transport_resources(pc)
                        .unwrap()
                        .packet_bytes_bound,
                )
                .unwrap();
            window
                .provide(|requests, replies| {
                    assert_eq!(requests, &[pc]);
                    assert_eq!(receiver.used_bytes(), receive_baseline + required);
                    replies[0].write(packet.as_bytes())
                })
                .unwrap();
            assert_eq!(
                source_budget.used_bytes(),
                source_baseline + if separate { 0 } else { required }
            );
            drop(window);
            assert_eq!(receiver.used_bytes(), receive_baseline);
            assert_eq!(source_budget.used_bytes(), source_baseline);
            drop(packet);
            assert_eq!(source_budget.used_bytes(), public_baseline);
            drop(plan);
            assert!(source_budget.used_bytes() > 0); // Real cache/table source owner survives.
            drop(source);
            assert_eq!(source_budget.used_bytes(), 0);
            assert_eq!(receive_budget.used_bytes(), 0);
        }
    }

    #[test]
    fn python_transport_resource_constructor_declares_both_online_receive_owners() {
        // Static cross-language packaging regression, not a Python execution claim.
        // The selected patch against 17e55e9e passed TEN positional arguments from
        // price_transport into an EIGHT-field dataclass: the moving root had the two
        // missing fields, masking the omission. Read the actual packaged source, not
        // a duplicated class, protocol double, target patch, or a generated receipt.
        let source = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../python/pllm/runtime/fss_transport.py"
        ));
        let declaration = source
            .split_once("class TransportResources:")
            .unwrap()
            .1
            .split_once("    @property")
            .unwrap()
            .0;
        let fields: Vec<_> = declaration
            .lines()
            .filter_map(|line| line.strip_prefix("    "))
            .filter_map(|line| line.split_once(": "))
            .filter(|(name, _)| {
                !name.is_empty() && name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
            })
            .collect();
        assert_eq!(
        fields.iter().map(|(name, _)| *name).collect::<Vec<_>>(),
        [
            "contract_digest",
            "tape",
            "guard_disk_bytes",
            "guard_reservation_bytes",
            "combined_reservation_bytes",
            "material_delivery",
            "python_batch_reservation_bytes",
            "native_batch_reservation_bytes",
            "python_receive_reservation_bytes",
            "native_receive_reservation_bytes",
        ],
        "price_transport's online owners must be real resource fields, including scalar/default construction"
    );
        assert_eq!(fields[8].1, "int = 0");
        assert_eq!(fields[9].1, "int = 0");
    }

    #[test]
    fn native_live_buffer_inventory_uses_actual_capacities_before_provider_entry() {
        use std::mem::{size_of, size_of_val};

        // Independent lower-bound inventory of concrete live provider storage, not
        // reservation_bytes() compared with itself. Serializer scratch is named
        // separately; this does not purport to count allocator headers, physical
        // call frames, Python objects, native crypto scratch or the compiler's R.
        for role in 0..2 {
            for count in [1, MATERIAL_WINDOW_RECORDS] {
                let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
                let plan = floor_plan(&budget, count);
                let ctx = context(&plan, 91);
                let pcs: Vec<_> = plan.material_steps().collect();
                let packets: Vec<_> = pcs
                    .iter()
                    .map(|&pc| {
                        let [a, b] = plan.issue_step(pc, ctx).unwrap();
                        if role == 0 {
                            drop(b);
                            a.export_packet().unwrap()
                        } else {
                            drop(a);
                            b.export_packet().unwrap()
                        }
                    })
                    .collect();
                let source_and_packets = budget.used_bytes();
                let mut window = ReplyWindow::new(&budget, 0usize).unwrap();
                for &pc in &pcs {
                    window
                        .push(
                            pc,
                            plan.step_transport_resources(pc)
                                .unwrap()
                                .packet_bytes_bound,
                        )
                        .unwrap();
                }
                assert_eq!(window.replies.capacity(), MATERIAL_WINDOW_RECORDS);
                let mut payload_capacity = 0;
                for ((reply, &pc), packet) in window.replies.iter().zip(&pcs).zip(&packets) {
                    let bound = plan
                        .step_transport_resources(pc)
                        .unwrap()
                        .packet_bytes_bound;
                    assert_eq!(reply.bytes.capacity(), bound);
                    assert!(packet.as_bytes().len() <= reply.bytes.capacity());
                    payload_capacity += reply.bytes.capacity();
                }
                assert!(payload_capacity <= MATERIAL_WINDOW_BYTES);
                let retained = size_of_val(&window)
                    + size_of_val(&budget)
                    + window.replies.capacity() * size_of::<Reply>()
                    + payload_capacity;
                let charged = budget.used_bytes() - source_and_packets;
                let bridge_scratch = 32 << 10;
                assert!(charged >= retained + bridge_scratch);
                window
                    .provide(|requests, replies| {
                        assert_eq!(requests, pcs.as_slice());
                        // All views and every reserved payload backing overlap the
                        // original source/packet owners before the first FFI write.
                        let view_storage = size_of::<Vec<ReplyView<'_>>>() + size_of_val(replies);
                        assert!(charged >= retained + view_storage + bridge_scratch);
                        assert_eq!(budget.used_bytes(), source_and_packets + charged);
                        for (reply, packet) in replies.iter_mut().zip(&packets) {
                            reply.write(packet.as_bytes())?;
                            assert_eq!(budget.used_bytes(), source_and_packets + charged);
                        }
                        Ok(())
                    })
                    .unwrap();
                drop(window);
                assert_eq!(budget.used_bytes(), source_and_packets);
                drop((packets, plan));
                assert_eq!(budget.used_bytes(), 0);
            }
        }
    }
}
