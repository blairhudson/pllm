use super::*;
use crate::fusefss_backend::Output;
use std::sync::atomic::Ordering;
use zeroize::Zeroizing;

const MASK: u64 = (1u64 << 50) - 1;

fn binding() -> kernels::Binding {
    kernels::Binding {
        source_digest: [13; 32],
        contract_digest: [17; 32],
    }
}

fn context() -> Context {
    Context {
        plan: [21; 32],
        session: [22; 32],
        operation: [23; 32],
        first_tensor_index: 27,
    }
}

fn build(vocabulary: usize, budget: &Budget) -> Result<Plan<'_>, Error> {
    kernels::compact_token_one_hot(
        binding(),
        kernels::InputRange {
            min: 0,
            max: vocabulary as i64 - 1,
        },
        vocabulary,
        budget,
    )
    .map(|(plan, meta)| {
        assert_eq!(meta.outputs, vocabulary);
        assert_eq!(meta.range, kernels::InputRange { min: 0, max: 1 });
        assert_eq!(meta.raw_range, None);
        plan
    })
}

fn frame(event: Event<'_>) -> crate::fusefss_backend::Frame<'_> {
    let Event::Frame(frame) = event else {
        panic!("expected one masked opening")
    };
    frame
}

fn output(event: Event<'_>) -> Output<'_> {
    let Event::Output(output) = event else {
        panic!("expected complete additive output")
    };
    output
}

fn check(pair: &[Output<'_>; 2], token: usize, vocabulary: usize) {
    assert_eq!(pair[0].shares().len(), vocabulary);
    assert_eq!(pair[1].shares().len(), vocabulary);
    for (i, (&a, &b)) in pair[0].shares().iter().zip(pair[1].shares()).enumerate() {
        assert!(a <= MASK && b <= MASK);
        assert_eq!(a.wrapping_add(b) & MASK, u64::from(i == token));
    }
}

fn execute<'a>(plan: &Plan<'a>, token: usize) -> Result<[Output<'a>; 2], Error> {
    // These local shares are deliberately not token IDs or bits.
    let mut a = plan.runtime(0, context(), &[MASK - 19])?;
    let mut b = plan.runtime(
        1,
        context(),
        &[((token as u64).wrapping_sub(MASK - 19)) & MASK],
    )?;
    assert!(matches!(a.poll()?, Event::NeedMaterial { pc: 0 }));
    assert!(matches!(b.poll()?, Event::NeedMaterial { pc: 0 }));
    let [am, bm] = plan.issue_step(0, context())?;
    let af = frame(a.feed(am)?);
    let bf = frame(b.feed(bm)?);
    assert_eq!(
        af.as_bytes().len() + bf.as_bytes().len(),
        plan.resources().frame_bytes_total
    );
    let pair = [
        output(a.advance(bf.as_bytes())?),
        output(b.advance(af.as_bytes())?),
    ];
    assert!(a.is_consumed() && b.is_consumed());
    assert!(matches!(a.advance(bf.as_bytes()), Err(Error::Consumed)));
    assert!(matches!(b.poll(), Err(Error::Consumed)));
    Ok(pair)
}

#[test]
fn compact_one_hot_exact_integer_parity_nonpower_tails_and_output_lifetimes() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    for vocabulary in [1, 2, 3, 5, 9, 17, 129] {
        let plan = build(vocabulary, &budget).unwrap();
        assert_eq!(plan.material_steps().collect::<Vec<_>>(), [0]);
        for token in [0, vocabulary / 2, vocabulary - 1] {
            let pair = execute(&plan, token).unwrap();
            check(&pair, token, vocabulary);
        }
        let pair = execute(&plan, vocabulary - 1).unwrap();
        drop(plan);
        assert_eq!(budget.used_bytes(), 2 * vocabulary * 8);
        drop(pair);
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn compact_one_hot_public_contract_validation_precedes_allocations_and_binds_identity() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    budget.fail_after.store(0, Ordering::SeqCst);
    for (lo, hi, vocabulary) in [
        (0, 0, 0),
        (0, 0, one_hot::MAX_VOCABULARY + 1),
        (-1, 2, 3),
        (0, 3, 3),
        (2, 1, 3),
        (0, 1i64 << 49, 3),
    ] {
        assert!(matches!(
            kernels::compact_token_one_hot(
                binding(),
                kernels::InputRange { min: lo, max: hi },
                vocabulary,
                &budget
            ),
            Err(Error::Domain)
        ));
        assert_eq!(budget.used_bytes(), 0);
    }
    for bad in [
        kernels::Binding {
            source_digest: [0; 32],
            ..binding()
        },
        kernels::Binding {
            contract_digest: [0; 32],
            ..binding()
        },
    ] {
        assert!(matches!(
            kernels::compact_token_one_hot(bad, kernels::InputRange { min: 0, max: 2 }, 3, &budget),
            Err(Error::Domain)
        ));
    }
    budget.fail_after.store(-1, Ordering::SeqCst);
    let original = build(5, &budget).unwrap();
    let digest = original.digest();
    for (bound, range, vocabulary) in [
        (
            kernels::Binding {
                source_digest: [14; 32],
                ..binding()
            },
            kernels::InputRange { min: 0, max: 4 },
            5,
        ),
        (
            kernels::Binding {
                contract_digest: [18; 32],
                ..binding()
            },
            kernels::InputRange { min: 0, max: 4 },
            5,
        ),
        (binding(), kernels::InputRange { min: 1, max: 4 }, 5),
        (binding(), kernels::InputRange { min: 0, max: 4 }, 6),
    ] {
        let (different, _) =
            kernels::compact_token_one_hot(bound, range, vocabulary, &budget).unwrap();
        assert_ne!(different.digest(), digest);
        check(&execute(&different, 3).unwrap(), 3, vocabulary);
    }
    assert!(original.runtime(0, context(), &[]).is_err());
    assert!(original.runtime(0, context(), &[0, 1]).is_err());
    assert!(original.runtime(0, context(), &[1 << 50]).is_err());
    assert!(original.runtime(2, context(), &[0]).is_err());
    let overflow = Context {
        first_tensor_index: u64::MAX,
        ..context()
    };
    assert!(original.runtime(0, overflow, &[0]).is_err());
    assert!(original.issue_step(0, overflow).is_err());
    assert!(original.issue_step(1, context()).is_err());
    assert!(original.step_transport_resources(1).is_err());
}

#[test]
fn compact_one_hot_qwen_geometry_prices_native_owners_without_vector_plan_metadata() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut metadata_bytes = None;
    for vocabulary in [1, 5, MAX_REGISTERS + 1, 151_936, one_hot::MAX_VOCABULARY] {
        let plan = build(vocabulary, &budget).unwrap();
        let r = plan.resources();
        let native = one_hot::resources(plan.one_hot_spec(0).unwrap()).unwrap();
        assert_eq!(plan.input_bits(), [50]);
        assert_eq!(plan.output_count(), vocabulary);
        for index in [0, vocabulary - 1] {
            assert_eq!(plan.output_bits(index), Ok(50));
            assert_eq!(plan.output_is_xor(index), Ok(false));
        }
        assert_eq!(plan.output_bits(vocabulary), Err(Error::Domain));
        assert_eq!(plan.output_is_xor(usize::MAX), Err(Error::Domain));
        assert_eq!((r.instructions, r.registers, r.secret_steps), (1, 1, 1));
        assert_eq!(
            (plan.code.len(), plan.registers.len(), plan.outputs.len()),
            (1, 1, 0)
        );
        assert_eq!(
            *metadata_bytes.get_or_insert(r.plan_reservation_bytes),
            r.plan_reservation_bytes
        );
        assert_eq!(budget.used_bytes(), r.plan_reservation_bytes);
        assert!(r.plan_reservation_bytes < 8192);
        assert_eq!(r.register_owner_bytes_per_party, runtime::owner_bytes(1, 0));
        assert_eq!(
            r.largest_primitive_reservation_bytes,
            native.combined_reservation_bytes
        );
        assert_eq!(
            r.private_material_bytes_total,
            native.private_material_bytes_total
        );
        assert_eq!(r.dcf_keys_per_party, usize::from(vocabulary != 1));
        assert_eq!(
            r.dcf_evaluations_per_party,
            native.dcf_evaluations_per_party
        );
        assert_eq!(r.triples_total, 0);
        assert_eq!(r.frames_total, 2);
        assert_eq!(r.frame_bytes_total, native.frame_bytes_total);
        assert_eq!(r.public_table_copy_bytes_total, 0);
        assert_eq!(r.public_table_reservation_bytes, 0);
        assert!(r.combined_reservation_bytes < MAX_ALLOCATION_BYTES);
        assert!(r.cumulative_work <= MAX_WORK);
        let step = plan.step_transport_resources(0).unwrap();
        assert_eq!(
            step.packet_bytes_bound,
            native.material_packet_bytes_per_party
        );
        assert_eq!(
            step.decoded_party_reservation_bytes,
            native.party_reservation_bytes
        );
        assert!(
            step.import_overlap_bytes
                >= native.output_bytes_per_party + native.scalar_scratch_bytes_per_party
        );
        let heap = plan.transport_resources_indexed(1, 5000).unwrap();
        let file = plan.transport_resources_indexed_file(1, 3_530_000).unwrap();
        let storage = wire::ImportGuard::indexed_file_resources(3_530_000).unwrap();
        assert_eq!(file.secret_steps, 1);
        assert_eq!(file.guard_reservation_bytes, storage.resident_bytes);
        assert_eq!(
            file.cumulative_packet_bytes_per_party_bound,
            step.packet_bytes_bound
        );
        assert_eq!(
            file.combined_worker_reservation_bytes,
            heap.combined_worker_reservation_bytes - heap.guard_reservation_bytes
                + storage.resident_bytes
        );
        assert_eq!(
            file.combined_dealer_export_reservation_bytes,
            heap.combined_dealer_export_reservation_bytes
        );
        assert_eq!(budget.used_bytes(), r.plan_reservation_bytes);
        // Geometry only: no Qwen-sized keys, outputs, replay files or execution.
    }
    assert_eq!(budget.used_bytes(), 0);
    assert!(kernels::token_one_hot(
        binding(),
        kernels::InputRange { min: 0, max: 0 },
        MAX_INPUTS + 1,
        &budget
    )
    .is_err());
}

#[test]
fn compact_one_hot_exact_budget_headroom_and_allocation_failures_refund() {
    let large = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let plan = build(9, &large).unwrap();
    let exact = plan.resources().combined_reservation_bytes;
    drop(plan);
    let budget = Budget::new(exact).unwrap();
    let plan = build(9, &budget).unwrap();
    check(&execute(&plan, 8).unwrap(), 8, 9);
    drop(plan);
    let short = Budget::new(exact - 1).unwrap();
    assert!(matches!(build(9, &short), Err(Error::Resource)));
    assert_eq!(short.used_bytes(), 0);
    let occupied = budget.reserve(1).unwrap();
    assert!(matches!(build(9, &budget), Err(Error::Resource)));
    assert_eq!(budget.used_bytes(), 1);
    drop(occupied);

    let mut succeeded = false;
    for fail in 0..24 {
        budget.fail_after.store(fail, Ordering::SeqCst);
        let result = build(9, &budget).and_then(|plan| execute(&plan, 8));
        match result {
            Ok(pair) => {
                check(&pair, 8, 9);
                succeeded = true;
            }
            Err(error) => assert_eq!(error, Error::Allocation),
        }
        assert_eq!(budget.used_bytes(), 0);
        if succeeded {
            break;
        }
    }
    assert!(succeeded);
}

#[test]
fn compact_one_hot_wrong_feed_peer_order_replay_and_cancel_burn_owned_runtime() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let plan = build(5, &budget).unwrap().into_shared().unwrap();
    for case in 0..8 {
        let mut runtime = plan.runtime_owned(0, context(), &[0]).unwrap();
        assert!(matches!(
            runtime.poll().unwrap(),
            Event::NeedMaterial { pc: 0 }
        ));
        let mut ctx = context();
        if case == 1 {
            ctx.session[0] ^= 1;
        }
        let [a, b] = plan.issue_step(0, ctx).unwrap();
        match case {
            0 => {
                assert!(matches!(runtime.feed(b), Err(Error::Context)));
            }
            1 => {
                assert!(matches!(runtime.feed(a), Err(Error::Context)));
            }
            2 => {
                assert!(matches!(runtime.advance(&[]), Err(Error::Context)));
            }
            3 => {
                let f = frame(runtime.feed(a).unwrap());
                assert!(matches!(runtime.advance(f.as_bytes()), Err(Error::Context)));
            }
            4 => {
                let _f = frame(runtime.feed(a).unwrap());
                assert!(matches!(runtime.advance(&[0; 9]), Err(Error::Context)));
            }
            5 => {
                let _f = frame(runtime.feed(a).unwrap());
                assert!(matches!(runtime.poll(), Err(Error::Context)));
            }
            6 => {
                assert!(matches!(
                    runtime.feed_result(Err(Error::Resource)),
                    Err(Error::Resource)
                ));
            }
            _ => {
                runtime.cancel();
            }
        }
        assert!(runtime.is_consumed());
        assert!(matches!(runtime.poll(), Err(Error::Consumed)));
    }
    let mut runtime = plan.runtime_owned(0, context(), &[0]).unwrap();
    let [a, b] = plan.issue_step(0, context()).unwrap();
    drop(b);
    let retained = frame(runtime.feed(a).unwrap());
    drop(plan);
    runtime.cancel();
    drop(runtime);
    assert_eq!(budget.used_bytes(), retained.as_bytes().len());
    drop(retained);
    assert_eq!(budget.used_bytes(), 0);
}

fn guard(capacity: usize, file: bool, budget: &Budget) -> wire::ImportGuard<'_> {
    if file {
        let r = wire::ImportGuard::indexed_file_resources(capacity).unwrap();
        wire::ImportGuard::new_indexed_file(capacity, r.disk_bytes, budget).unwrap()
    } else {
        wire::ImportGuard::new_indexed(capacity, budget).unwrap()
    }
}

#[test]
fn compact_one_hot_real_packets_use_existing_heap_and_file_guards_and_keep_export_leases() {
    for file in [false, true] {
        for vocabulary in [1, 5] {
            let dealer = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
            let left = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
            let right = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
            let dp = build(vocabulary, &dealer).unwrap();
            let lp = build(vocabulary, &left).unwrap();
            let rp = build(vocabulary, &right).unwrap();
            assert_eq!(dp.digest(), lp.digest());
            assert_eq!(dp.digest(), rp.digest());
            let pricing = if file {
                lp.transport_resources_indexed_file(1, 2)
            } else {
                lp.transport_resources_indexed(1, 2)
            }
            .unwrap();
            let mut lg = guard(2, file, &left);
            let mut rg = guard(2, file, &right);
            let [am, bm] = dp.issue_step(0, context()).unwrap();
            drop(dp);
            let ap = am.export_packet().unwrap();
            let bp = bm.export_packet().unwrap();
            assert_eq!(ap.as_bytes()[8], 5); // Existing native one-hot kind.
            assert_eq!(
                ap.as_bytes().len(),
                lp.step_transport_resources(0).unwrap().packet_bytes_bound
            );
            assert_eq!(
                dealer.used_bytes(),
                ap.as_bytes().len() + bp.as_bytes().len()
            );
            assert_eq!(ap.nonce(), bp.nonce());
            let nonce = ap.nonce();
            let am = lp
                .import_step_bytes(0, context(), 0, nonce, ap.as_bytes(), &mut lg)
                .unwrap();
            let delivered = wire::MaterialPacket::from_bytes(bp.as_bytes(), &right).unwrap();
            let bm = rp
                .import_step(0, context(), 1, nonce, delivered, &mut rg)
                .unwrap();
            let token = vocabulary - 1;
            let mut a = lp.runtime(0, context(), &[MASK]).unwrap();
            let mut b = rp
                .runtime(1, context(), &[((token as u64).wrapping_sub(MASK)) & MASK])
                .unwrap();
            let af = frame(a.feed(am).unwrap());
            let bf = frame(b.feed(bm).unwrap());
            assert!(left.used_bytes() <= pricing.combined_worker_reservation_bytes);
            assert!(right.used_bytes() <= pricing.combined_worker_reservation_bytes);
            let pair = [
                output(a.advance(bf.as_bytes()).unwrap()),
                output(b.advance(af.as_bytes()).unwrap()),
            ];
            check(&pair, token, vocabulary);
            a.cancel();
            let ctx = Context {
                session: [91; 32],
                ..context()
            };
            assert!(matches!(
                lp.import_step_bytes(0, ctx, 0, nonce, ap.as_bytes(), &mut lg),
                Err(Error::Consumed)
            ));
            assert!(matches!(
                rp.import_step_bytes(0, ctx, 1, nonce, bp.as_bytes(), &mut rg),
                Err(Error::Consumed)
            ));
            drop((a, b));
            drop((lp, rp, lg, rg));
            assert_eq!(left.used_bytes(), vocabulary * 8 + af.as_bytes().len());
            assert_eq!(right.used_bytes(), vocabulary * 8 + bf.as_bytes().len());
            drop((pair, af, bf, ap, bp));
            assert_eq!(
                (dealer.used_bytes(), left.used_bytes(), right.used_bytes()),
                (0, 0, 0)
            );
        }
    }
}

#[test]
fn compact_one_hot_bad_imports_and_adapter_allocation_failures_burn_whole_request_claims() {
    let dealer = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let worker = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let dp = build(5, &dealer).unwrap();
    let wp = build(5, &worker).unwrap();
    for file in [false, true] {
        let mut replay = guard(16, file, &worker);
        for case in 0..7 {
            let [a, b] = dp.issue_step(0, context()).unwrap();
            drop(b);
            let packet = a.export_packet().unwrap();
            let mut bytes = Zeroizing::new(packet.as_bytes().to_vec());
            let nonce = packet.nonce();
            let mut ctx = context();
            let mut role = 0;
            let mut pc = 0;
            match case {
                0 => ctx.operation[0] ^= 1,
                1 => role = 1,
                2 => pc = 1,
                3 => bytes[88] ^= 1,
                4 => bytes.truncate(1),
                5 => bytes.push(0),
                _ => ctx.first_tensor_index = u64::MAX,
            }
            let mut runtime = wp.runtime(0, context(), &[0]).unwrap();
            assert!(matches!(
                runtime.poll().unwrap(),
                Event::NeedMaterial { pc: 0 }
            ));
            let result = wp.import_step_bytes(pc, ctx, role, nonce, &bytes, &mut replay);
            assert!(runtime.feed_result(result).is_err());
            assert!(runtime.is_consumed());
            // Same recipient/nonce burns even when retry repairs pc/context/bytes.
            assert!(matches!(
                wp.import_step_bytes(0, context(), role, nonce, packet.as_bytes(), &mut replay),
                Err(Error::Consumed)
            ));
        }
    }
    let [a, b] = dp.issue_step(0, context()).unwrap();
    drop(b);
    let packet = a.export_packet().unwrap();
    let base = worker.used_bytes();
    let mut succeeded = false;
    for fail in 0..16 {
        // Independent receiving sessions only for allocation fault injection.
        let mut replay = guard(1, false, &worker);
        worker.fail_after.store(fail, Ordering::SeqCst);
        let result = wp.import_step_bytes(
            0,
            context(),
            0,
            packet.nonce(),
            packet.as_bytes(),
            &mut replay,
        );
        worker.fail_after.store(-1, Ordering::SeqCst);
        match result {
            Ok(material) => {
                drop(material);
                succeeded = true;
            }
            Err(error) => assert_eq!(error, Error::Allocation),
        }
        assert!(matches!(
            wp.import_step_bytes(
                0,
                context(),
                0,
                packet.nonce(),
                packet.as_bytes(),
                &mut replay
            ),
            Err(Error::Consumed)
        ));
        drop(replay);
        assert_eq!(worker.used_bytes(), base);
        if succeeded {
            break;
        }
    }
    assert!(succeeded);
}

#[test]
fn compact_extension_preserves_scalar_digest_and_rejects_builder_composition() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut builder = Builder::new([7; 32], b"typed-test-contract", &[5], &budget).unwrap();
    assert_eq!(
        builder.push(
            Instruction::CompactTokenOneHot(5),
            Register {
                bits: 50,
                domain: Domain::Arithmetic
            }
        ),
        Err(Error::Unsupported)
    );
    let square = builder.multiply(0, 0).unwrap();
    let plan = builder.finish(&[square]).unwrap();
    assert_eq!(
        plan.digest(),
        [
            245, 72, 87, 195, 174, 8, 179, 198, 220, 88, 181, 17, 7, 77, 168, 27, 33, 3, 216, 132,
            55, 49, 172, 92, 25, 197, 53, 99, 138, 114, 184, 71,
        ]
    );
    assert_eq!(plan.output_count(), 1);
    assert_eq!(plan.output_bits(0), Ok(5));
    assert_eq!(plan.output_bits(1), Err(Error::Domain));
    assert_eq!(plan.output_is_xor(0), Ok(false));
    assert_eq!(plan.material_steps().collect::<Vec<_>>(), [0]);
    assert!(plan.one_hot_spec(0).is_err());
}
