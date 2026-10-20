use super::super::Event;
use super::*;
use zeroize::Zeroizing;

fn context() -> Context {
    Context {
        plan: [1; 32],
        session: [2; 32],
        operation: [3; 32],
        first_tensor_index: 7,
    }
}
fn mixed_plan(budget: &Budget) -> Plan<'_> {
    let mut b = Builder::new([8; 32], b"wire-mixed-plan", &[5, 5, 50], budget).unwrap();
    let product = b.multiply(0, 1).unwrap();
    let bit = b.lt_unsigned(0, 9).unwrap();
    let selected = b.select(bit, product, 1).unwrap();
    let table = b.table(5, &[10, 11, 12, 13, 14, 15, 16, 17]).unwrap();
    let lookup = b.lookup(1, table).unwrap();
    let tr = b
        .convert(
            0,
            conversions::Operation::Tr {
                shift: 2,
                range: conversions::SignedRange::full_ring(5).unwrap(),
            },
            3,
        )
        .unwrap();
    let ars = b
        .convert(
            0,
            conversions::Operation::Ars {
                shift: 1,
                range: conversions::SignedRange::full_ring(5).unwrap(),
            },
            5,
        )
        .unwrap();
    let zero = b.convert(1, conversions::Operation::ZeroExtend, 9).unwrap();
    let signed = b
        .convert(
            0,
            conversions::Operation::SignExtend {
                range: conversions::SignedRange::full_ring(5).unwrap(),
            },
            9,
        )
        .unwrap();
    let gelu = b.gelu(2).unwrap();
    b.finish(&[product, bit, selected, lookup, tr, ars, zero, signed, gelu])
        .unwrap()
}
fn small_plan<'a>(budget: &'a Budget, contract: &[u8]) -> Plan<'a> {
    let mut b = Builder::new([8; 32], contract, &[5, 5], budget).unwrap();
    let product = b.multiply(0, 1).unwrap();
    let other = b
        .convert(
            0,
            conversions::Operation::Ars {
                shift: 1,
                range: conversions::SignedRange::full_ring(5).unwrap(),
            },
            5,
        )
        .unwrap();
    let local = b.constant(5, 1).unwrap();
    let table = b.table(5, &[10, 11, 12, 13]).unwrap();
    let lookup = b.lookup(1, table).unwrap();
    b.finish(&[product, other, local, lookup]).unwrap()
}

#[test]
fn separate_dealer_and_workers_compute_all_material_kinds_from_real_bytes() {
    let dealer = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let left = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let right = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let dp = mixed_plan(&dealer);
    let lp = mixed_plan(&left);
    let rp = mixed_plan(&right);
    assert_eq!(dp.digest(), lp.digest());
    assert_eq!(dp.digest(), rp.digest());
    let lr = lp.transport_resources(1, 32).unwrap();
    let rr = rp.transport_resources(1, 32).unwrap();
    let mut lg = ImportGuard::new(32, &left).unwrap();
    let mut rg = ImportGuard::new(32, &right).unwrap();
    let mut a = lp.runtime(0, context(), &[0, 0, 0]).unwrap();
    let mut b = rp.runtime(1, context(), &[7, 3, 4096]).unwrap();
    let mut ea = a.poll().unwrap();
    let mut eb = b.poll().unwrap();
    let mut steps = 0;
    let mut kinds = [false; 4];
    loop {
        match (ea, eb) {
            (Event::NeedMaterial { pc: pa }, Event::NeedMaterial { pc: pb }) => {
                assert_eq!(pa, pb);
                let pc = pa;
                let [ma, mb] = dp.issue_step(pc, context()).unwrap();
                let pa = ma.export_packet().unwrap();
                let pb = mb.export_packet().unwrap();
                let nonce = pa.nonce();
                let bound = dp.step_transport_resources(pc).unwrap();
                assert!(pa.as_bytes().len() <= bound.packet_bytes_bound);
                assert!(pb.as_bytes().len() <= bound.packet_bytes_bound);
                kinds[match pa.as_bytes()[8] {
                    1 => 0,
                    2 => 1,
                    3 => 2,
                    4 => 3,
                    _ => panic!("namespace"),
                }] = true;
                let ba = Zeroizing::new(pa.as_bytes().to_vec());
                let bb = Zeroizing::new(pb.as_bytes().to_vec());
                drop((pa, pb)); // Dealer parties and packet owners gone before worker import.
                assert_eq!(dealer.used_bytes(), dp.resources.plan_reservation_bytes);
                let ma = lp
                    .import_step_bytes(pc, context(), 0, nonce, &ba, &mut lg)
                    .unwrap();
                let mb = rp
                    .import_step_bytes(pc, context(), 1, nonce, &bb, &mut rg)
                    .unwrap();
                ea = a.feed(ma).unwrap();
                eb = b.feed(mb).unwrap();
                steps += 1;
            }
            (Event::Frame(fa), Event::Frame(fb)) => {
                ea = a.advance(fb.as_bytes()).unwrap();
                eb = b.advance(fa.as_bytes()).unwrap();
            }
            (Event::Output(oa), Event::Output(ob)) => {
                let expected = [21, 1, 21, 13, 1, 3, 3, 7, 3447];
                for (i, expected) in expected.into_iter().enumerate() {
                    assert_eq!(
                        oa.shares()[i].wrapping_add(ob.shares()[i])
                            & mask(lp.output_bits(i).unwrap()),
                        expected
                    );
                }
                break;
            }
            _ => panic!("public schedules diverged"),
        }
        assert!(left.used_bytes() <= lr.combined_worker_reservation_bytes);
        assert!(right.used_bytes() <= rr.combined_worker_reservation_bytes);
    }
    assert_eq!(steps, dp.resources.secret_steps);
    assert!(kinds.into_iter().all(|present| present));
    drop((a, b, lg, rg));
    drop((dp, lp, rp));
    assert_eq!(dealer.used_bytes(), 0);
    assert_eq!(left.used_bytes(), 0);
    assert_eq!(right.used_bytes(), 0);
}

#[test]
fn substitutions_and_bad_public_steps_burn_session_claims() {
    let dealer = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let worker = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let dp = small_plan(&dealer, b"same");
    let wp = small_plan(&worker, b"same");
    let foreign = small_plan(&worker, b"foreign");
    for case in 0..9 {
        let [ma, mb] = dp.issue_step(0, context()).unwrap();
        drop(mb);
        let packet = ma.export_packet().unwrap();
        let nonce = packet.nonce();
        let mut bytes = Zeroizing::new(packet.as_bytes().to_vec());
        let mut ctx = context();
        let mut role = 0;
        let mut pc = 0;
        let mut expected_nonce = nonce;
        let mut plan = &wp;
        match case {
            0 => ctx.operation[0] ^= 1,
            1 => role = 1,
            2 => pc = 1, // Different conversion operation, same input/output ring.
            3 => pc = 3, // Different material kind: public lookup.
            4 => plan = &foreign,
            5 => bytes[88] ^= 1,
            6 => expected_nonce[0] ^= 1,
            7 => pc = 2, // Local-only public instruction.
            _ => pc = usize::MAX,
        }
        let mut guard = ImportGuard::new(2, &worker).unwrap();
        assert!(plan
            .import_step_bytes(pc, ctx, role, expected_nonce, &bytes, &mut guard)
            .is_err());
        assert!(matches!(
            plan.import_step_bytes(pc, ctx, role, expected_nonce, &bytes, &mut guard),
            Err(Error::Consumed)
        ));
    }
}

#[test]
fn packet_outlives_dealer_plan_and_replay_survives_cancel() {
    let dealer = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let worker = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let material = {
        let plan = small_plan(&dealer, b"same");
        let [a, b] = plan.issue_step(0, context()).unwrap();
        drop(b);
        a
    };
    let packet = material.export_packet().unwrap();
    assert_eq!(dealer.used_bytes(), packet.as_bytes().len());
    let plan = small_plan(&worker, b"same");
    let mut guard = ImportGuard::new(2, &worker).unwrap();
    let nonce = packet.nonce();
    let mut runtime = plan.runtime(0, context(), &[0, 0]).unwrap();
    assert!(matches!(
        runtime.poll().unwrap(),
        Event::NeedMaterial { pc: 0 }
    ));
    let material = plan
        .import_step_bytes(0, context(), 0, nonce, packet.as_bytes(), &mut guard)
        .unwrap();
    let frame = match runtime.feed(material).unwrap() {
        Event::Frame(f) => f,
        _ => panic!("opening"),
    };
    runtime.cancel();
    assert!(runtime.is_consumed());
    assert!(matches!(
        plan.import_step_bytes(0, context(), 0, nonce, packet.as_bytes(), &mut guard),
        Err(Error::Consumed)
    ));
    drop(runtime);
    drop(guard);
    assert_eq!(
        worker.used_bytes(),
        plan.resources.plan_reservation_bytes + frame.as_bytes().len()
    );
    drop(frame);
    drop(plan);
    drop(packet);
    assert_eq!(worker.used_bytes(), 0);
    assert_eq!(dealer.used_bytes(), 0);
}

#[test]
fn packet_capacity_and_all_import_allocation_failpoints_are_bounded() {
    let dealer = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let worker = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let dp = small_plan(&dealer, b"same");
    let wp = small_plan(&worker, b"same");
    let [ma, mb] = dp.issue_step(0, context()).unwrap();
    drop(mb);
    let packet = ma.export_packet().unwrap();
    let nonce = packet.nonce();
    let r = wp.step_transport_resources(0).unwrap();
    assert_eq!(packet.as_bytes().len(), r.packet_bytes_bound);
    let oversized = vec![0; r.packet_bytes_bound + 1];
    let mut guard = ImportGuard::new(1, &worker).unwrap();
    worker
        .fail_after
        .store(0, std::sync::atomic::Ordering::SeqCst);
    assert!(matches!(
        wp.import_step_bytes(0, context(), 0, nonce, &oversized, &mut guard),
        Err(Error::Resource)
    ));
    worker
        .fail_after
        .store(-1, std::sync::atomic::Ordering::SeqCst);
    drop(guard);
    let baseline = worker.used_bytes();
    let mut succeeded = false;
    for fail in 0..8 {
        let mut guard = ImportGuard::new(1, &worker).unwrap();
        worker
            .fail_after
            .store(fail, std::sync::atomic::Ordering::SeqCst);
        let result = wp.import_step_bytes(0, context(), 0, nonce, packet.as_bytes(), &mut guard);
        worker
            .fail_after
            .store(-1, std::sync::atomic::Ordering::SeqCst);
        match result {
            Ok(material) => {
                drop(material);
                succeeded = true;
            }
            Err(Error::Allocation) => {}
            Err(e) => panic!("unexpected {e:?}"),
        }
        assert!(matches!(
            wp.import_step_bytes(0, context(), 0, nonce, packet.as_bytes(), &mut guard),
            Err(Error::Consumed)
        ));
        drop(guard);
        assert_eq!(worker.used_bytes(), baseline);
        if succeeded {
            break;
        }
    }
    assert!(succeeded);
}

#[test]
fn optional_pricing_preserves_historical_resources_and_plan_digest() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let plan = mixed_plan(&budget);
    let before = plan.resources();
    let digest = plan.digest();
    let resident = budget.used_bytes();
    let base = plan.transport_resources(0, 32).unwrap();
    let retained = plan.transport_resources(3, 32).unwrap();
    assert_eq!(
        retained.retained_packet_bytes,
        3 * base.largest_packet_bytes_bound
    );
    assert_eq!(
        retained.combined_worker_reservation_bytes,
        base.combined_worker_reservation_bytes + retained.retained_packet_bytes
    );
    assert_eq!(retained.guard_reservation_bytes, 32 * 33);
    assert_eq!(plan.digest(), digest);
    assert_eq!(
        plan.resources().combined_reservation_bytes,
        before.combined_reservation_bytes
    );
    assert_eq!(
        plan.resources().private_material_bytes_total,
        before.private_material_bytes_total
    );
    assert_eq!(budget.used_bytes(), resident);
    assert!(matches!(
        plan.transport_resources(usize::MAX, 32),
        Err(Error::Resource)
    ));
    assert!(matches!(
        plan.transport_resources(0, 4097),
        Err(Error::Resource)
    ));
}

#[test]
fn all_native_kinds_import_failpoints_retire_descriptors_packets_and_material() {
    let dealer = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let worker = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let dp = mixed_plan(&dealer);
    let wp = mixed_plan(&worker);
    for (pc, instruction) in dp.code.iter().enumerate() {
        if matches!(
            instruction,
            Instruction::Constant(_)
                | Instruction::Add(..)
                | Instruction::Sub(..)
                | Instruction::PublicMultiply(..)
                | Instruction::Narrow(_)
        ) {
            continue;
        }
        let [a, b] = dp.issue_step(pc, context()).unwrap();
        drop(b);
        let packet = a.export_packet().unwrap();
        let nonce = packet.nonce();
        let baseline = worker.used_bytes();
        let mut completed = false;
        for fail in 0..40 {
            // Each iteration models an independent receiving session for this
            // allocation test only, never a retry with a reset live guard.
            let mut guard = ImportGuard::new(1, &worker).unwrap();
            worker
                .fail_after
                .store(fail, std::sync::atomic::Ordering::SeqCst);
            let result =
                wp.import_step_bytes(pc, context(), 0, nonce, packet.as_bytes(), &mut guard);
            worker
                .fail_after
                .store(-1, std::sync::atomic::Ordering::SeqCst);
            match result {
                Ok(material) => {
                    drop(material);
                    completed = true;
                }
                Err(Error::Allocation) => {}
                Err(error) => panic!("pc {pc} fail {fail}: {error:?}"),
            }
            assert!(matches!(
                wp.import_step_bytes(pc, context(), 0, nonce, packet.as_bytes(), &mut guard),
                Err(Error::Consumed)
            ));
            drop(guard);
            assert_eq!(worker.used_bytes(), baseline, "pc {pc}, allocation {fail}");
            if completed {
                break;
            }
        }
        assert!(completed, "all allocations reached at pc {pc}");
        assert_eq!(
            dealer.used_bytes(),
            dp.resources.plan_reservation_bytes + packet.as_bytes().len()
        );
    }
}

#[test]
fn failed_import_feed_result_burns_waiting_runtime() {
    let dealer = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let worker = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let dp = small_plan(&dealer, b"same");
    let wp = small_plan(&worker, b"same");
    let [a, b] = dp.issue_step(0, context()).unwrap();
    drop(b);
    let packet = a.export_packet().unwrap();
    let mut guard = ImportGuard::new(1, &worker).unwrap();
    let mut runtime = wp.runtime(0, context(), &[1, 2]).unwrap();
    assert!(matches!(
        runtime.poll().unwrap(),
        Event::NeedMaterial { pc: 0 }
    ));
    let mut foreign = context();
    foreign.session[0] ^= 1;
    let result = wp.import_step_bytes(0, foreign, 0, packet.nonce(), packet.as_bytes(), &mut guard);
    assert!(matches!(runtime.feed_result(result), Err(Error::Context)));
    assert!(runtime.is_consumed());
}

#[test]
fn retained_receiver_packet_and_frame_owners_outlive_cancelled_plan() {
    let dealer = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let worker = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let dp = small_plan(&dealer, b"same");
    let wp = small_plan(&worker, b"same");
    let pricing = wp.transport_resources(1, 2).unwrap();
    let [a, b] = dp.issue_step(0, context()).unwrap();
    drop(b);
    let source = a.export_packet().unwrap();
    let packet = MaterialPacket::from_bytes(source.as_bytes(), &worker).unwrap();
    let delivered = MaterialPacket::from_bytes(source.as_bytes(), &worker).unwrap();
    drop(source);
    let mut guard = ImportGuard::new(2, &worker).unwrap();
    let material = wp
        .import_step(0, context(), 0, packet.nonce(), delivered, &mut guard)
        .unwrap();
    let mut runtime = wp.runtime(0, context(), &[0, 0]).unwrap();
    assert!(matches!(
        runtime.poll().unwrap(),
        Event::NeedMaterial { pc: 0 }
    ));
    let frame = match runtime.feed(material).unwrap() {
        Event::Frame(frame) => frame,
        _ => panic!("opening"),
    };
    assert!(worker.used_bytes() <= pricing.combined_worker_reservation_bytes);
    runtime.cancel();
    drop(runtime);
    drop(wp);
    drop(guard);
    assert_eq!(
        worker.used_bytes(),
        packet.as_bytes().len() + frame.as_bytes().len()
    );
    drop(packet);
    drop(frame);
    assert_eq!(worker.used_bytes(), 0);
}

#[test]
fn request_layout_pricing_charges_full_guard_without_changing_other_owners() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let plan = mixed_plan(&budget);
    let digest = plan.digest();
    let historical = plan.resources();
    let baseline = budget.used_bytes();
    let legacy = plan.transport_resources(1, 32).unwrap();
    assert_eq!(
        legacy,
        plan.transport_resources_with_guard(1, ImportGuardLayout::legacy(32).unwrap())
            .unwrap()
    );
    let layout = ImportGuardLayout::for_requests(49_933).unwrap();
    let priced = plan
        .transport_resources_indexed(1, layout.capacity())
        .unwrap();
    assert_eq!(
        priced,
        plan.transport_resources_with_guard(1, layout).unwrap()
    );
    assert_eq!(
        priced.guard_reservation_bytes,
        ImportGuard::indexed_reservation_bytes(layout.capacity()).unwrap()
    );
    assert_eq!(priced.guard_reservation_bytes, layout.reservation_bytes());
    assert!(priced.guard_reservation_bytes > 4096 * 33);
    assert_eq!(
        priced.combined_worker_reservation_bytes,
        legacy.combined_worker_reservation_bytes - legacy.guard_reservation_bytes
            + layout.reservation_bytes()
    );
    assert_eq!(
        priced.combined_dealer_export_reservation_bytes,
        legacy.combined_dealer_export_reservation_bytes
    );
    assert_eq!(
        priced.retained_frame_bytes_per_party,
        legacy.retained_frame_bytes_per_party
    );
    assert_eq!(
        priced.largest_import_overlap_bytes,
        legacy.largest_import_overlap_bytes
    );
    assert_eq!(priced.retained_packet_bytes, legacy.retained_packet_bytes);
    assert_eq!(
        priced.cumulative_packet_bytes_per_party_bound,
        legacy.cumulative_packet_bytes_per_party_bound
    );
    assert_eq!(plan.digest(), digest);
    assert_eq!(
        plan.resources().combined_reservation_bytes,
        historical.combined_reservation_bytes
    );
    assert_eq!(
        plan.resources().private_material_bytes_total,
        historical.private_material_bytes_total
    );
    assert_eq!(budget.used_bytes(), baseline);
    let guard = ImportGuard::with_layout(layout, &budget).unwrap();
    assert_eq!(
        budget.used_bytes(),
        baseline + priced.guard_reservation_bytes
    );
    drop(guard);
    assert_eq!(budget.used_bytes(), baseline);
    let excess_slots = (MAX_ALLOCATION_BYTES - priced.combined_worker_reservation_bytes)
        / priced.largest_packet_bytes_bound
        + 2;
    assert!(matches!(
        plan.transport_resources_with_guard(excess_slots, layout),
        Err(Error::Resource)
    ));
    assert!(matches!(
        plan.transport_resources_with_guard(usize::MAX, layout),
        Err(Error::Resource)
    ));
}

#[test]
fn request_guard_imports_4097_fresh_contexts_then_rejects_same_nonce_new_context() {
    let dealer = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let worker = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let dp = small_plan(&dealer, b"request-long-replay");
    let wp = small_plan(&worker, b"request-long-replay");
    let layout = ImportGuardLayout::for_requests(4097).unwrap();
    let mut guard = ImportGuard::new_indexed(layout.capacity(), &worker).unwrap();
    let baseline = worker.used_bytes();
    let mut first_nonce = [0; 32];
    for ordinal in 0..4097u64 {
        let mut ctx = context();
        ctx.operation[..8].copy_from_slice(&ordinal.to_le_bytes());
        // Every iteration issues entirely fresh one-use masks/triples/keys.
        let [a, b] = dp.issue_step(0, ctx).unwrap();
        drop(b);
        let packet = a.export_packet().unwrap();
        if ordinal == 0 {
            first_nonce = packet.nonce();
        }
        drop(
            wp.import_step_bytes(0, ctx, 0, packet.nonce(), packet.as_bytes(), &mut guard)
                .unwrap(),
        );
        assert_eq!(worker.used_bytes(), baseline);
    }
    let mut ctx = context();
    ctx.operation[..8].copy_from_slice(&u64::MAX.to_le_bytes());
    let [a, b] = dp.issue_step(0, ctx).unwrap();
    drop(b);
    let packet = a.export_packet().unwrap();
    let mut fresh = Zeroizing::new(packet.as_bytes().to_vec());
    // Model trusted issuance repeating ONLY the envelope nonce. Its fresh
    // material genuinely belongs to a different valid public context.
    fresh[48..80].copy_from_slice(&first_nonce);
    let end = fresh.len() - 32;
    let checksum = Sha256::digest(&fresh[..end]);
    fresh[end..].copy_from_slice(&checksum);
    {
        // Separate receiving session proves those bytes/context are valid; the
        // actual request-long guard is never retired or reset.
        let mut independent = ImportGuard::new(1, &worker).unwrap();
        drop(
            wp.import_step_bytes(0, ctx, 0, first_nonce, &fresh, &mut independent)
                .unwrap(),
        );
    }
    assert!(matches!(
        wp.import_step_bytes(0, ctx, 0, first_nonce, &fresh, &mut guard),
        Err(Error::Consumed)
    ));
    assert!(matches!(
        wp.import_step_bytes(0, ctx, 0, packet.nonce(), packet.as_bytes(), &mut guard),
        Err(Error::Resource)
    ));
    assert_eq!(worker.used_bytes(), baseline);
    drop(wp);
    assert_eq!(worker.used_bytes(), layout.reservation_bytes());
    drop(guard);
    drop(packet);
    drop(dp);
    assert_eq!(worker.used_bytes(), 0);
    assert_eq!(dealer.used_bytes(), 0);
}

#[test]
fn request_guard_survives_cancel_out_of_order_and_preparation_failure() {
    let dealer = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let worker = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let dp = small_plan(&dealer, b"request-long-failures");
    let wp = small_plan(&worker, b"request-long-failures");
    let layout = ImportGuardLayout::for_requests(5000).unwrap();
    let pricing = wp.transport_resources_with_guard(1, layout).unwrap();
    let mut guard = ImportGuard::with_layout(layout, &worker).unwrap();
    let [a, b] = dp.issue_step(0, context()).unwrap();
    drop(b);
    let source = a.export_packet().unwrap();
    let retained = MaterialPacket::from_bytes(source.as_bytes(), &worker).unwrap();
    let material = wp
        .import_step_bytes(
            0,
            context(),
            0,
            source.nonce(),
            source.as_bytes(),
            &mut guard,
        )
        .unwrap();
    let mut runtime = wp.runtime(0, context(), &[0, 0]).unwrap();
    assert!(matches!(
        runtime.poll().unwrap(),
        Event::NeedMaterial { pc: 0 }
    ));
    let frame = match runtime.feed(material).unwrap() {
        Event::Frame(frame) => frame,
        _ => panic!("opening"),
    };
    assert!(worker.used_bytes() <= pricing.combined_worker_reservation_bytes);
    runtime.cancel();
    drop(runtime);
    assert!(matches!(
        wp.import_step_bytes(
            0,
            context(),
            0,
            source.nonce(),
            source.as_bytes(),
            &mut guard
        ),
        Err(Error::Consumed)
    ));
    drop(source);

    let mut waiting = wp.runtime(0, context(), &[0, 0]).unwrap();
    assert!(matches!(
        waiting.poll().unwrap(),
        Event::NeedMaterial { pc: 0 }
    ));
    let [a, b] = dp.issue_step(1, context()).unwrap();
    drop(b);
    let later = a.export_packet().unwrap();
    let delivered = MaterialPacket::from_bytes(later.as_bytes(), &worker).unwrap();
    let out_of_order = wp.import_step(0, context(), 0, later.nonce(), delivered, &mut guard);
    assert!(matches!(
        waiting.feed_result(out_of_order),
        Err(Error::Context)
    ));
    assert!(waiting.is_consumed());
    assert!(matches!(
        wp.import_step_bytes(1, context(), 0, later.nonce(), later.as_bytes(), &mut guard),
        Err(Error::Consumed)
    ));
    drop(waiting);
    drop(later);

    let [a, b] = dp.issue_step(0, context()).unwrap();
    drop(b);
    let fresh = a.export_packet().unwrap();
    assert!(matches!(
        wp.import_step_bytes(0, context(), 0, fresh.nonce(), &[], &mut guard),
        Err(Error::Resource)
    ));
    assert!(matches!(
        wp.import_step_bytes(0, context(), 0, fresh.nonce(), fresh.as_bytes(), &mut guard),
        Err(Error::Consumed)
    ));
    drop(fresh);
    drop(wp);
    assert_eq!(
        worker.used_bytes(),
        layout.reservation_bytes() + retained.as_bytes().len() + frame.as_bytes().len()
    );
    drop(guard);
    assert_eq!(
        worker.used_bytes(),
        retained.as_bytes().len() + frame.as_bytes().len()
    );
    drop((retained, frame, dp));
    assert_eq!(worker.used_bytes(), 0);
    assert_eq!(dealer.used_bytes(), 0);
}

#[test]
fn file_guard_prices_qwen_resident_overlap_with_disk_separate() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let plan = small_plan(&budget, b"file-replay-price");
    let baseline = budget.used_bytes();
    let small = plan.transport_resources(1, 2).unwrap();
    let r = ImportGuard::indexed_file_resources(3_530_000).unwrap();
    let file = plan.transport_resources_indexed_file(1, 3_530_000).unwrap();
    assert!(matches!(
        plan.transport_resources_indexed(1, 3_530_000),
        Err(Error::Resource)
    ));
    assert_eq!(file.guard_reservation_bytes, r.resident_bytes);
    assert_eq!(
        file.combined_worker_reservation_bytes,
        small.combined_worker_reservation_bytes - small.guard_reservation_bytes + r.resident_bytes
    );
    assert_eq!(
        file.combined_dealer_export_reservation_bytes,
        small.combined_dealer_export_reservation_bytes
    );
    assert_eq!(
        file.retained_frame_bytes_per_party,
        small.retained_frame_bytes_per_party
    );
    assert_eq!(file.retained_packet_bytes, small.retained_packet_bytes);
    assert_eq!(r.disk_bytes, 8_388_608 * 33);
    assert_eq!(budget.used_bytes(), baseline);
    for capacity in [0, usize::MAX] {
        assert!(matches!(
            plan.transport_resources_indexed_file(1, capacity),
            Err(Error::Resource)
        ));
    }
    assert!(matches!(
        plan.transport_resources_indexed_file(usize::MAX, 4),
        Err(Error::Resource)
    ));
}

#[test]
fn file_guard_rejects_same_nonce_across_sessions_after_cancel_and_burns_malformed() {
    let dealer = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let worker = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let dp = small_plan(&dealer, b"file-request-replay");
    let wp = small_plan(&worker, b"file-request-replay");
    let r = ImportGuard::indexed_file_resources(4).unwrap();
    let mut guard = ImportGuard::new_indexed_file(4, r.disk_bytes, &worker).unwrap();
    let [a, b] = dp.issue_step(0, context()).unwrap();
    drop(b);
    let packet = a.export_packet().unwrap();
    let nonce = packet.nonce();
    let retained = MaterialPacket::from_bytes(packet.as_bytes(), &worker).unwrap();
    let material = wp
        .import_step_bytes(0, context(), 0, nonce, packet.as_bytes(), &mut guard)
        .unwrap();
    drop(packet);
    let mut runtime = wp.runtime(0, context(), &[0, 0]).unwrap();
    assert!(matches!(
        runtime.poll().unwrap(),
        Event::NeedMaterial { pc: 0 }
    ));
    let frame = match runtime.feed(material).unwrap() {
        Event::Frame(frame) => frame,
        _ => panic!("opening"),
    };
    runtime.cancel();
    drop(runtime);
    let ctx = Context {
        session: [19; 32],
        operation: [20; 32],
        ..context()
    };
    let [a, b] = dp.issue_step(0, ctx).unwrap();
    drop(b);
    let packet = a.export_packet().unwrap();
    let mut fresh = Zeroizing::new(packet.as_bytes().to_vec());
    fresh[48..80].copy_from_slice(&nonce);
    let end = fresh.len() - 32;
    let checksum = Sha256::digest(&fresh[..end]);
    fresh[end..].copy_from_slice(&checksum);
    {
        let mut independent = ImportGuard::new_indexed(1, &worker).unwrap();
        drop(
            wp.import_step_bytes(0, ctx, 0, nonce, &fresh, &mut independent)
                .unwrap(),
        );
    }
    assert!(matches!(
        wp.import_step_bytes(0, ctx, 0, nonce, &fresh, &mut guard),
        Err(Error::Consumed)
    ));

    // Failure before constructing a MaterialPacket still burns the new nonce.
    assert!(matches!(
        wp.import_step_bytes(0, ctx, 0, packet.nonce(), &[], &mut guard),
        Err(Error::Resource)
    ));
    assert!(matches!(
        wp.import_step_bytes(0, ctx, 0, packet.nonce(), packet.as_bytes(), &mut guard),
        Err(Error::Consumed)
    ));
    drop(packet);
    let [a, b] = dp.issue_step(0, ctx).unwrap();
    drop(b);
    let packet = a.export_packet().unwrap();
    let mut malformed = Zeroizing::new(packet.as_bytes().to_vec());
    malformed[88] ^= 1;
    assert!(matches!(
        wp.import_step_bytes(0, ctx, 0, packet.nonce(), &malformed, &mut guard),
        Err(Error::Context)
    ));
    assert!(matches!(
        wp.import_step_bytes(0, ctx, 0, packet.nonce(), packet.as_bytes(), &mut guard),
        Err(Error::Consumed)
    ));
    drop(packet);
    drop(wp);
    assert_eq!(
        worker.used_bytes(),
        r.resident_bytes + retained.as_bytes().len() + frame.as_bytes().len()
    );
    drop(guard);
    assert_eq!(
        worker.used_bytes(),
        retained.as_bytes().len() + frame.as_bytes().len()
    );
    drop((frame, retained, dp));
    assert_eq!(worker.used_bytes(), 0);
    assert_eq!(dealer.used_bytes(), 0);
}
