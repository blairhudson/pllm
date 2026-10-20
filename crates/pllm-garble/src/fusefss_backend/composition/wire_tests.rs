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
