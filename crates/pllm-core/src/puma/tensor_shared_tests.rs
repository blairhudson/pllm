use super::*;

#[test]
fn resource_breakdown_counts_backing_and_joint_working_peak_independently() {
    let mut b = Builder::new(3, Interval::new(-3, 3).unwrap()).unwrap();
    let out = b
        .affine_lanes(
            b.input(),
            &[Affine {
                left: 2,
                right: 0,
                constant: 1,
            }; 3],
        )
        .unwrap();
    let p = b.finish(out).unwrap();
    let r = p.resources().unwrap();
    assert_eq!(r.runtime_fixed_reserve_bytes, FIXED_RESERVE_BYTES);
    assert_eq!(r.public_table_bytes, 3 * std::mem::size_of::<Affine>());
    assert_eq!(
        r.immutable_template_bytes,
        std::mem::size_of::<Program>() + 2 * std::mem::size_of::<usize>() + r.public_table_bytes
    );
    assert_eq!(
        r.register_container_bytes,
        2 * std::mem::size_of::<Option<Register>>()
    );
    assert_eq!(
        r.invocation_container_bytes,
        std::mem::size_of::<Executor>() + r.register_container_bytes
    );
    // Input and output coexist: 12 words; lane-affine copies add 18 words.
    assert_eq!(r.working_bytes, (12 + 18) * 8);
    assert_eq!(r.frame_bytes, 3 * std::mem::size_of::<Frame>());
    let common =
        FIXED_RESERVE_BYTES + r.invocation_container_bytes + r.frame_bytes + r.working_bytes;
    assert_eq!(r.owned_admitted_bytes, common + r.public_table_bytes);
    assert_eq!(r.shared_admitted_bytes, common + r.immutable_template_bytes);
    assert_eq!(p.admitted_bytes(), r.owned_admitted_bytes);
    // Independent frozen instruction encoding: no storage/Arc/resource fields.
    let mut h = Sha256::new();
    h.update(b"pllm.puma.public-tensor-program.q18.v1");
    h.update(b"puma.source-derived-q18.v1");
    h.update(3u64.to_le_bytes());
    h.update(2u64.to_le_bytes());
    h.update([0]);
    h.update((-3i64).to_le_bytes());
    h.update(3i64.to_le_bytes());
    h.update([11, 0, 0]);
    for _ in 0..3 {
        h.update(2i64.to_le_bytes());
        h.update(1i64.to_le_bytes());
    }
    h.update((-5i64).to_le_bytes());
    h.update(7i64.to_le_bytes());
    assert_eq!(p.digest(), <[u8; 32]>::from(h.finalize()));
    let mut invalid = p.clone();
    invalid.register_count = usize::MAX;
    assert_eq!(invalid.resources(), Err(Error::Bounds));
}

#[test]
fn shared_checked_outputs_and_last_template_reference_retire_consumingly() {
    let p = Arc::new(
        Program::checked_range(
            3,
            Interval::new(i64::MIN, i64::MAX).unwrap(),
            Interval::new(-10, 10).unwrap(),
            0,
        )
        .unwrap(),
    );
    let weak = Arc::downgrade(&p);
    let runtimes = parties(p.work_budget().limits().unwrap());
    let mut input = inputs(&runtimes, &[-11, -10, 11]).map(Some).into_iter();
    let mut owners = runtimes
        .map(|r| Executor::new_shared(r, p.clone(), input.next().unwrap().unwrap()).unwrap());
    let usages = owners.each_ref().map(Executor::payload_usage);
    let instructions = p.instructions();
    drop(p);
    assert_eq!(weak.strong_count(), 3);
    for _ in 0..instructions {
        step(&mut owners);
    }
    let results = owners.map(|o| o.finish_checked().unwrap());
    assert_eq!(weak.strong_count(), 0);
    assert_eq!(clear_refs(results.each_ref().map(|r| &r.1)), [0, -10, 0]);
    assert_eq!(clear_refs(results.each_ref().map(|r| &r.2)), [0, 1, 0]);
    drop(results);
    assert!(usages.iter().all(|u| u.live_bytes() == 0));
}

#[test]
fn shared_templates_preserve_full_wire_both_components_and_cursors() {
    for lanes in [1, 3, 127, 128] {
        // Use ordinary tile constructors for exact sole/full/tail widths.
        let mut b = Builder::new(lanes, Interval::new(-ONE, ONE).unwrap()).unwrap();
        let out = b
            .affine_lanes(
                b.input(),
                &vec![
                    Affine {
                        left: ONE / 2,
                        right: 0,
                        constant: 7
                    };
                    lanes
                ],
            )
            .unwrap();
        let out = b.rescale(out, 18).unwrap();
        let p = Arc::new(b.finish(out).unwrap());
        let values: Vec<_> = (0..lanes).map(|i| i as i64 * 13 - 70).collect();
        let mut owned = parity_owners(&p, &values, false, 1);
        let mut shared = parity_owners(&p, &values, true, 1);
        assert_eq!(Arc::strong_count(&p), 4);
        for owner in &shared {
            assert!(std::ptr::eq::<Program>(&*owner.program, &*p));
            assert_eq!(
                owner.program.public_affines[0].as_ptr(),
                p.public_affines[0].as_ptr()
            );
        }
        compare_execution(&mut owned, &mut shared);
        let owned = owned.map(|o| o.finish().unwrap());
        let shared = shared.map(|o| o.finish().unwrap());
        for (a, b) in owned.iter().zip(&shared) {
            assert_eq!(a.1.first(), b.1.first());
            assert_eq!(a.1.second(), b.1.second());
            assert_eq!(a.0.cursors(), b.0.cursors());
        }
        assert_eq!(Arc::strong_count(&p), 1);
    }
}

fn parity_owners(p: &Arc<Program>, values: &[i64], shared: bool, repeats: u64) -> [Executor; 3] {
    let runtimes = parties(
        p.work_budget()
            .checked_mul(repeats)
            .unwrap()
            .limits()
            .unwrap(),
    );
    let mut input = inputs(&runtimes, values).map(Some).into_iter();
    runtimes.map(|r| {
        let input = input.next().unwrap().unwrap();
        if shared {
            Executor::new_shared(r, p.clone(), input)
        } else {
            Executor::new(r, (**p).clone(), input)
        }
        .unwrap()
    })
}

type Wire = Vec<(Party, usize, Vec<u8>)>;
type Retained = Vec<crate::puma::wire::Encoded<Frame>>;

fn encoded_step(owners: &mut [Executor; 3], retained: &mut Retained, retain: bool) -> Wire {
    let mut frames: Vec<_> = owners
        .iter_mut()
        .flat_map(|o| o.start_step().unwrap().into_iter().flatten())
        .collect();
    let mut wire = Vec::new();
    while !frames.is_empty() {
        let mut next = Vec::new();
        for frame in frames {
            let (sender, recipient) = (frame.sender(), frame.recipient() as usize);
            let bytes = frame.encode().unwrap();
            wire.push((sender, recipient, bytes.as_bytes().to_vec()));
            next.extend(
                owners[recipient]
                    .receive_encoded(bytes.as_bytes(), sender)
                    .unwrap()
                    .into_iter()
                    .flatten(),
            );
            if retain && retained.is_empty() {
                retained.push(bytes);
            }
        }
        frames = next;
    }
    for o in owners {
        o.finish_step().unwrap();
    }
    wire
}

fn compare_state(owned: &[Executor; 3], shared: &[Executor; 3]) {
    for (a, b) in owned.iter().zip(shared) {
        assert_eq!(a.cursors(), b.cursors()); // public/pair/private/AES
        assert_eq!(
            a.runtime.as_ref().unwrap().source.runtime.next_sequence(),
            b.runtime.as_ref().unwrap().source.runtime.next_sequence()
        );
        assert_eq!(a.step, b.step);
        assert_eq!(a.active, b.active);
        for (a, b) in a.registers.iter().zip(&b.registers) {
            match (a, b) {
                (None, None) => (),
                (Some(Register::Arithmetic(a)), Some(Register::Arithmetic(b))) => {
                    assert_eq!(a.first(), b.first());
                    assert_eq!(a.second(), b.second());
                }
                (Some(Register::Bit(a)), Some(Register::Bit(b))) => {
                    assert_eq!(a.first(), b.first());
                    assert_eq!(a.second(), b.second());
                    assert_eq!(a.bits(), b.bits());
                }
                _ => panic!("register presence/type mismatch"),
            }
        }
    }
}

fn compare_execution(owned: &mut [Executor; 3], shared: &mut [Executor; 3]) {
    compare_state(owned, shared);
    while !owned[0].complete() {
        assert_eq!(
            encoded_step(owned, &mut vec![], false),
            encoded_step(shared, &mut vec![], false)
        );
        compare_state(owned, shared);
    }
}

#[test]
fn shared_compact_and_complete_rows_preserve_unnormalized_wire_and_all_live_components() {
    let bound = Interval::new(-ONE, ONE).unwrap();
    let mut checked = Builder::new(3, Interval::new(i64::MIN, i64::MAX).unwrap()).unwrap();
    // Same public constructors as compact checked/nonlinear tests: membership
    // remains full-word, followed by an explicit compact signed predicate.
    let safe = checked.checked_range(checked.input(), bound, 0).unwrap();
    let bit = checked.sign_compact(safe.safe).unwrap();
    let out = checked.from_bit(bit).unwrap();
    let checked = checked.finish_checked(out, safe.valid).unwrap();
    for (program, values) in [
        (
            Program::activation_guarded_compact(
                Activation::Silu,
                3,
                Interval::new(-9 * ONE, 5 * ONE).unwrap(),
            )
            .unwrap(),
            vec![-9 * ONE, ONE / 3, 5 * ONE],
        ),
        (checked, vec![i64::MIN, -ONE, i64::MAX]),
        (
            Program::rms_norm(1, 3, bound, &[ONE, ONE / 2, -ONE]).unwrap(),
            vec![-ONE, 0, ONE / 3],
        ),
        (
            Program::rms_norm_compact(1, 3, bound, &[ONE, ONE / 2, -ONE]).unwrap(),
            vec![-ONE, 0, ONE / 3],
        ),
        (Program::softmax(1, 3, bound).unwrap(), vec![-ONE, 0, ONE]),
        (
            Program::softmax_guarded_compact(1, 3, Interval::new(-15 * ONE, 0).unwrap()).unwrap(),
            vec![-15 * ONE, -ONE, 0],
        ),
        (
            Program::layer_norm_compact(1, 3, bound, &[ONE, ONE / 2, -ONE], &[7, -3, 11]).unwrap(),
            vec![-ONE, 0, ONE / 3],
        ),
    ] {
        let p = Arc::new(program);
        let digest = p.digest();
        let work = p.work_budget();
        let mut owned = parity_owners(&p, &values, false, 1);
        let mut shared = parity_owners(&p, &values, true, 1);
        compare_execution(&mut owned, &mut shared);
        for (a, b) in owned.into_iter().zip(shared) {
            let finish = |o: Executor| {
                if p.validity_lanes().is_some() {
                    let (r, out, valid) = o.finish_checked().unwrap();
                    (r, vec![out, valid])
                } else {
                    let (r, out) = o.finish().unwrap();
                    (r, vec![out])
                }
            };
            let (a, av) = finish(a);
            let (b, bv) = finish(b);
            assert_eq!(a.cursors(), b.cursors());
            for (a, b) in av.iter().zip(&bv) {
                assert_eq!(a.first(), b.first());
                assert_eq!(a.second(), b.second());
            }
        }
        assert_eq!(p.digest(), digest);
        assert_eq!(p.work_budget(), work);
        assert_eq!(Arc::strong_count(&p), 1);
    }
}

#[test]
fn shared_retained_original_frame_survives_same_runtime_continuation_without_credit() {
    let p = Arc::new(Program::negative_compact(3, Interval::new(-ONE, ONE).unwrap()).unwrap());
    let mut owned = parity_owners(&p, &[-ONE, 0, ONE], false, 2);
    let mut shared = parity_owners(&p, &[-ONE, 0, ONE], true, 2);
    let usages = shared.each_ref().map(Executor::payload_usage);
    let (mut a_frames, mut b_frames) = (vec![], vec![]);
    while !owned[0].complete() {
        assert_eq!(
            encoded_step(&mut owned, &mut a_frames, true),
            encoded_step(&mut shared, &mut b_frames, true)
        );
        compare_state(&owned, &shared);
    }
    assert_eq!(a_frames.len(), 1);
    assert_eq!(b_frames.len(), 1);
    let resume = |owners: [Executor; 3], shared| {
        owners.map(|o| {
            let (r, value) = o.finish().unwrap();
            assert!(r.source.runtime.next_sequence() > 0);
            if shared {
                Executor::new_shared(r, p.clone(), value)
            } else {
                Executor::new(r, (*p).clone(), value)
            }
            .unwrap()
        })
    };
    let mut owned = resume(owned, false);
    let mut shared = resume(shared, true);
    compare_execution(&mut owned, &mut shared);
    for (a, b) in owned.into_iter().zip(shared) {
        let (a, av) = a.finish().unwrap();
        let (b, bv) = b.finish().unwrap();
        assert_eq!(a.cursors(), b.cursors());
        assert_eq!(av.first(), bv.first());
        assert_eq!(av.second(), bv.second());
    }
    assert!(usages.iter().any(|u| u.live_bytes() > 0));
    drop((a_frames, b_frames));
    assert!(usages.iter().all(|u| u.live_bytes() == 0));
    assert_eq!(Arc::strong_count(&p), 1);
}

#[test]
fn shared_failed_protocol_transition_keeps_owned_cursor_and_retirement_parity() {
    let p = Arc::new(Program::negative_compact(3, Interval::new(-ONE, ONE).unwrap()).unwrap());
    let mut owned = parity_owners(&p, &[-ONE, 0, ONE], false, 1);
    let mut shared = parity_owners(&p, &[-ONE, 0, ONE], true, 1);
    // Fail while the first real conversion's protocol frames are outstanding.
    for (a, b) in owned.iter_mut().zip(&mut shared) {
        let af = a.start_step().unwrap();
        let bf = b.start_step().unwrap();
        assert_eq!(af.iter().flatten().count(), bf.iter().flatten().count());
        assert_eq!(
            a.receive_encoded(b"bad", Party::P0).err(),
            b.receive_encoded(b"bad", Party::P0).err()
        );
        assert!(a.retired() && b.retired());
        assert_eq!(a.cursors(), b.cursors());
        assert_eq!(
            a.runtime.as_ref().unwrap().source.runtime.next_sequence(),
            b.runtime.as_ref().unwrap().source.runtime.next_sequence()
        );
        assert_eq!(a.start_step().err(), Some(Error::Retired));
        assert_eq!(b.start_step().err(), Some(Error::Retired));
    }
}

#[test]
fn shared_exact_quote_does_not_discount_retained_original_payload() {
    let p = Arc::new(Program::negative_compact(3, Interval::new(-ONE, ONE).unwrap()).unwrap());
    let quote = p.resources().unwrap().shared_admitted_bytes;
    let limits = crate::puma::Limits::new(6, crate::puma::MAX_OPERATIONS, quote).unwrap();
    let runtimes = parties(PrssLimits::new(limits, crate::puma::prss::MAX_AES_BLOCKS).unwrap());
    let mut input = inputs(&runtimes, &[-ONE, 0, ONE]).map(Some).into_iter();
    let mut runtimes = runtimes.into_iter();
    let r = runtimes.next().unwrap();
    let usage = r.payload_usage();
    // Actual ledger-backed retained buffer, not a fictitious external credit.
    let retained = usage.copy(&[7], 1).unwrap();
    assert_eq!(usage.live_bytes(), 8);
    assert!(matches!(
        Executor::new_shared(r, p.clone(), input.next().unwrap().unwrap()),
        Err(Error::Bounds)
    ));
    assert_eq!(usage.live_bytes(), 8);
    drop(retained);
    assert_eq!(usage.live_bytes(), 0);
    assert_eq!(Arc::strong_count(&p), 1);
}
