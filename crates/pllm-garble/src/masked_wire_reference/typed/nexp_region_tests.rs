//! UNRUN: canonical record vectors must be reviewed before these hash/issue tests.
use super::*;
use crate::shared_allocation::{OwnedBudget, OwnedExternalCharge};
use pllm_core::edge_contract::{EdgeIdentity, Shape};
use std::sync::atomic::Ordering;

fn spec(source: SourceIdentity, n: usize) -> Spec {
    let types = [
        (50, 12, Interpretation::Signed),
        (40, 12, Interpretation::Signed),
        (40, 12, Interpretation::Signed),
        (1, 0, Interpretation::Predicate),
        (1, 0, Interpretation::Predicate),
        (16, 12, Interpretation::Unsigned),
        (16, 12, Interpretation::Unsigned),
        (16, 12, Interpretation::Unsigned),
        (8, 0, Interpretation::Unsigned),
        (8, 0, Interpretation::Unsigned),
        (50, 12, Interpretation::Signed),
        (50, 12, Interpretation::Signed),
        (50, 24, Interpretation::Signed),
        (50, 12, Interpretation::Signed),
    ];
    Spec {
        edges: std::array::from_fn(|i| {
            EdgeDescriptor::new(
                EdgeIdentity::new(source, [4; 32], [5; 32], [10 + i as u8; 32], 7),
                Shape::new(&[n]).unwrap(),
                Numeric::new(types[i].0, types[i].1, types[i].2).unwrap(),
                if i == 3 || i == 4 {
                    Representation::CommonXorMasked
                } else {
                    Representation::CommonAdditiveMasked
                },
                None,
            )
            .unwrap()
        }),
        operation: [90; 32],
        next: [91; 32],
    }
}
fn owner(root: &OwnedBudget, values: &[i64], spare: usize) -> Input {
    let capacity = values.len() + spare;
    root.charge_external(capacity * 8)
        .unwrap()
        .try_build(|budget| {
            let mut data = crate::shared_allocation::allocate(capacity, budget)?;
            data.extend_from_slice(values);
            Ok(Zeroizing::new(data))
        })
        .unwrap()
}

// A continuous SAME-counter grow, never dropping bootstrap then reacquiring.
fn owned_working(
    dealer: &OwnedLocalDealer,
    root: &OwnedBudget,
    s: &Spec,
    input: &Input,
) -> (Quote, LeaseFor<OwnedBudgetContext>, OwnedExternalCharge) {
    let mut original = root
        .charge_external(bootstrap_bytes::<OwnedBudgetContext>())
        .unwrap();
    let q = resources(dealer, s, input).unwrap();
    original.resize_payload(q.total).unwrap();
    let lease = original
        .split_native(dealer.budget.budget(), q.total)
        .unwrap();
    (q, lease, original)
}
fn source() -> SourceIdentity {
    SourceIdentity::new([1; 32], [2; 32], approved_implementation())
}
fn seeded() -> impl FnMut(&mut [u8]) -> Result<(), Error> {
    let mut state = 0x1234_5678_9abc_def1u64;
    move |bytes: &mut [u8]| {
        for byte in bytes {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *byte = state as u8;
        }
        Ok(())
    }
}
fn issue(
    values: &[i64],
) -> (
    Mask<OwnedBudgetContext>,
    Coordinator<OwnedBudgetContext>,
    [Role<OwnedBudgetContext>; 2],
    Quote,
) {
    let root = OwnedBudget::new(64 << 20).unwrap();
    let source = source();
    let s = spec(source, values.len());
    let mut dealer = OwnedLocalDealer::new(source, [4; 32], [92; 32], &root).unwrap();
    let input = owner(&root, values, 3);
    let (q, working, original) = owned_working(&dealer, &root, &s, &input);
    let (mask, coordinator, roles) =
        construct_with(&mut dealer, s, input, working, seeded()).unwrap();
    drop((original, dealer, root));
    (mask, coordinator, roles, q)
}

std::thread_local! {
    // Fixed test-only observation at the REAL protected encode boundary. No
    // protocol callback, fake ledger replay, recovery API or dynamic storage.
    static ENTRY_ENCODING: std::cell::Cell<(usize,u8,u8,Option<Binding>,bool)> =
        const { std::cell::Cell::new((0,0,0,None,false)) };
}
pub(super) fn observe_entry_encode(ledger: &internal_use::Ledger<N>, root: Binding) {
    ENTRY_ENCODING.with(|event| {
        event.set((
            event.get().0.saturating_add(1),
            ledger.next,
            ledger.role,
            Some(root),
            ledger.bindings[0] == root && ledger.uses[0].edge() == root.descriptor,
        ));
    });
}

#[test]
fn genuine_dealer_entry_is_burned_before_protected_root_encoding() {
    ENTRY_ENCODING.with(|event| event.set((0, 0, 0, None, false)));
    let (mask, coordinator, roles, _) = issue(&[0, 65535, (1i64 << 38) - 1]);
    let (calls, next, role, observed, binding_matches) = ENTRY_ENCODING.with(|event| event.get());
    assert_eq!((calls, next, role, binding_matches), (1, 1, 2, true));
    // Compare the observed prepared binding against the actual returned root
    // capabilities, not a synthesized parallel binding or post-recovery state.
    for party in &roles {
        let state = party.slot.state.lock().unwrap();
        assert_eq!(observed, Some(state.root.as_ref().unwrap().binding));
        assert_eq!(state.ledger.next, 0); // roles burn their own entry on begin
    }
    drop((roles, coordinator, mask));
}

#[test]
fn actual_stage_validation_call_slots_and_passes_are_preadmitted() {
    let root = OwnedBudget::new(64 << 20).unwrap();
    let source = source();
    let s = spec(source, 3);
    let dealer = OwnedLocalDealer::new(source, [4; 32], [92; 32], &root).unwrap();
    let input = owner(&root, &[0, 65536, (1i64 << 38) - 1], 2);
    let original = root
        .charge_external(bootstrap_bytes::<OwnedBudgetContext>())
        .unwrap();
    let q = resources(&dealer, &s, &input).unwrap();
    assert_eq!(
        bootstrap_bytes::<OwnedBudgetContext>(),
        size_of::<Spec>()
            + size_of::<Quote>()
            + size_of::<Result<Quote, BuildError>>()
            + size_of::<[usize; 16]>()
            + size_of::<LeaseFor<OwnedBudgetContext>>()
            + size_of::<ConstructionSlots<'_, OwnedBudgetContext>>()
            + bounds::validation_slots_bytes()
    );
    assert_eq!(q.work.input_validation_lanes, 6); // two whole-input validations
    assert_eq!(q.work.root_validation_lanes, 3); // later actual encode validation
    assert_eq!(q.work.public_table_range_checks, 512);
    assert_eq!(q.work.stage_bound_descriptor_checks, 18);
    assert_eq!(q.work.stage_bound_interval_writes, 19);
    assert_eq!(q.work.stage_bound_metadata_queries, 46);
    assert_eq!(q.work.stage_bound_table_metadata_passes, 2);
    assert!(q.control_bytes >= bootstrap_bytes::<OwnedBudgetContext>());
    drop((original, input, dealer, root));
}
fn complete<C: AllocationContext>(roles: &mut [Role<C>; 2]) -> Vec<[Frame<C>; 2]> {
    let mut frames = [
        roles[0].begin(&mut || false).unwrap(),
        roles[1].begin(&mut || false).unwrap(),
    ];
    let lanes = roles[0].slot.state.lock().unwrap().work[0].len();
    check_frames(&frames, 1, lanes);
    let mut retained = Vec::new();
    for round in 0..ROUNDS {
        let left = roles[0]
            .advance(frames[1].bytes().unwrap(), &mut || false)
            .unwrap();
        let right = roles[1]
            .advance(frames[0].bytes().unwrap(), &mut || false)
            .unwrap();
        retained.push(frames);
        match (left, right) {
            (Step::Frame(a), Step::Frame(b)) => {
                assert!(round < ROUNDS - 1);
                frames = [a, b];
                check_frames(&frames, round + 2, lanes);
            }
            (Step::Ready, Step::Ready) => {
                assert_eq!(round, ROUNDS - 1);
                return retained;
            }
            _ => panic!("nExp schedules diverged"),
        }
    }
    panic!("missing final barrier")
}
fn check_frames<C: AllocationContext>(pair: &[Frame<C>; 2], stage: usize, lanes: usize) {
    let reference = integer_oracle::STAGES[stage - 1];
    let widths = integer_oracle::payload_bits(stage).unwrap();
    for (role, frame) in pair.iter().enumerate() {
        let bytes = frame.bytes().unwrap();
        assert!(matches!(
            (stage, &frame.native),
            (1 | 3 | 4 | 10 | 11, Body::Arithmetic(_))
                | (2, Body::Select(_))
                | (5..=8, Body::Lookup(_))
                | (9, Body::Product(_))
        ));
        assert_eq!(
            bytes.len(),
            integer_oracle::frame_bytes(stage, lanes).unwrap()
        );
        assert_eq!(&bytes[..8], &reference.magic);
        assert_eq!(bytes[72], role as u8);
        assert_eq!(bytes[73], reference.phase);
        assert_eq!(&bytes[74..76], &(lanes as u16).to_le_bytes());
        for (word, chunk) in bytes[reference.header..].chunks_exact(8).enumerate() {
            let value = u64::from_le_bytes(chunk.try_into().unwrap());
            assert!(value < (1u64 << widths[word % widths.len()]));
        }
    }
}
fn original_integer(tables: &Tables<OwnedBudgetContext>, value: i64) -> u64 {
    let c = (value as u64).min(65535);
    let a = tables.checked.values(tables::Table::LowByte)[(c & 255) as usize] as u128;
    let b = tables.checked.values(tables::Table::HighByte)[(c >> 8) as usize] as u128;
    ((a * b) / 4096) as u64
}

#[test]
fn full_eleven_stage_integer_oracle_masks_uses_bytes_and_source_retention() {
    for values in [
        vec![0],
        vec![
            1,
            255,
            256,
            1279,
            1280,
            2690,
            31233,
            31487,
            31488,
            32767,
            32768,
            34047,
            34049,
            34304,
            65534,
            65535,
            65536,
            (1i64 << 38) - 1,
        ],
        (0..256).map(|i| (i as i64) * 257).collect::<Vec<_>>(),
    ] {
        let (mask, coordinator, mut roles, q) = issue(&values);
        let independent = integer_oracle::fixture_tables().unwrap();
        let frames = complete(&mut roles);
        assert_eq!(frames.len(), 11);
        assert_eq!(
            frames
                .iter()
                .flatten()
                .map(|frame| frame.bytes().unwrap().len())
                .sum::<usize>(),
            1688 + 208 * values.len()
        );
        assert_eq!(q.peer_bytes, 1688 + 208 * values.len());
        let expected_magic = [
            b"PLLMSA01",
            b"PLLMSL01",
            b"PLLMSA01",
            b"PLLMSA01",
            b"PLLMLU01",
            b"PLLMSL01",
            b"PLLMLU01",
            b"PLLMSL01",
            b"PLLMSP01",
            b"PLLMSA01",
            b"PLLMSA01",
        ];
        for (round, pair) in frames.iter().enumerate() {
            for frame in pair {
                assert_eq!(&frame.bytes().unwrap()[..8], expected_magic[round]);
            }
        }
        let [left, right] = roles;
        let outputs = [left.finish().unwrap(), right.finish().unwrap()];
        for output in &outputs {
            assert_eq!(output.descriptor().unwrap(), mask.native.binding.descriptor);
            let state = output.slot.state.lock().unwrap();
            assert_eq!(state.ledger.next, 18);
            assert_eq!(state.phase, 255);
            for (i, use_) in state.tables.declaration.uses.iter().enumerate() {
                assert_eq!(use_.operand_position(), integer_oracle::USES[i].1);
                assert_eq!(use_.ordinal(), i as u32);
            }
            assert!(std::ptr::eq(
                state.tables.input.budget(),
                mask.native._lease.budget.budget()
            ));
            assert!(state.tables.input.payload().capacity() > values.len());
            let words = state.output.as_ref().unwrap().backing.values();
            for (i, value) in values.iter().enumerate() {
                let actual = words[i].wrapping_sub(mask.native.backing.values()[i]) & ring_mask(50);
                assert_eq!(actual, original_integer(&state.tables, *value), "X={value}");
                // Selected double-floor quantization has +1 carry transitions
                // and zero-to-positive resets. Never repair/smooth this map.
                if let Some(expected) = match *value {
                    1279 => Some(2995),
                    1280 => Some(2996),
                    2690 => Some(2123), // half-code: floor, not nearest-even 2124
                    31233 | 31487 | 34049 => Some(0),
                    31488 => Some(1),
                    _ => None,
                } {
                    assert_eq!(actual, expected, "selected quantization at X={value}");
                }
                assert_eq!(
                    actual,
                    integer_oracle::lane(*value, &independent).unwrap().e
                );
            }
        }
        drop(coordinator);
        assert!(outputs.iter().all(|output| output.descriptor().is_err()));
        assert!(frames.iter().flatten().all(|frame| frame.bytes().is_err()));
    }
}

#[test]
fn borrowed_roles_use_same_owned_input_counter_and_original_reserved_handoff() {
    let root = OwnedBudget::new(64 << 20).unwrap();
    let source = source();
    let s = spec(source, 2);
    let mut dealer = LocalDealer::new(source, [4; 32], [92; 32], root.budget()).unwrap();
    let input = owner(&root, &[0, 65536], 5);
    let original_ptr = input.payload().as_ptr();
    let mut original = root
        .budget()
        .charge_external(bootstrap_bytes::<&Budget>())
        .unwrap();
    let q = resources(&dealer, &s, &input).unwrap();
    original.resize_payload(q.total).unwrap();
    let working = original.split_payload(root.budget(), q.total).unwrap();
    let (mask, coordinator, mut roles) =
        construct_with(&mut dealer, s, input, working, seeded()).unwrap();
    assert_eq!(
        coordinator.shared.slots[0]
            .state
            .lock()
            .unwrap()
            .tables
            .input
            .payload()
            .as_ptr(),
        original_ptr
    );
    let frames = complete(&mut roles);
    let [a, b] = roles;
    let outputs = [a.finish().unwrap(), b.finish().unwrap()];
    assert!(outputs.iter().all(|output| output.descriptor().is_ok()));
    drop(coordinator);
    assert!(outputs.iter().all(|output| output.descriptor().is_err()));
    drop((outputs, frames, mask, original, dealer));
}

#[test]
fn last_invalid_original_integer_and_bad_schema_reject_before_entropy_or_factory() {
    for case in 0..5 {
        let root = OwnedBudget::new(64 << 20).unwrap();
        let source = source();
        let mut s = spec(source, 2);
        let mut dealer = OwnedLocalDealer::new(source, [4; 32], [92; 32], &root).unwrap();
        let values = match case {
            0 => [0, -1],
            1 => [0, 1i64 << 38],
            _ => [0, 1],
        };
        let input = owner(&root, &values, 0);
        let (_, working, original) = owned_working(&dealer, &root, &s, &input);
        if case == 2 {
            s.edges[9] = s.edges[8];
        }
        if case == 3 {
            s.operation = [91; 32];
            s.edges[0] = s.edges[1];
        }
        let before = dealer.counter;
        let mut draws = 0;
        root.budget().fail_after.store(0, Ordering::SeqCst);
        let result = construct_with(&mut dealer, s, input, working, |_| {
            draws += 1;
            Ok(())
        });
        if case < 4 {
            assert!(result.is_err());
            assert_eq!(draws, 0);
            assert_eq!(dealer.counter, before);
            assert_eq!(root.budget().fail_after.load(Ordering::SeqCst), 0);
        } else {
            assert!(matches!(result, Err(BuildError::Native(Error::Allocation))));
            assert_eq!(draws, 0);
        }
        root.budget().fail_after.store(-1, Ordering::SeqCst);
        drop(original);
    }
}

#[test]
fn exact_one_short_wrong_counter_and_actual_incoming_capacity_precede_factories() {
    for case in 0..4 {
        let root = OwnedBudget::new(64 << 20).unwrap();
        let foreign = OwnedBudget::new(64 << 20).unwrap();
        let source = source();
        let s = spec(source, 1);
        let mut dealer = OwnedLocalDealer::new(source, [4; 32], [92; 32], &root).unwrap();
        let input = owner(if case == 2 { &foreign } else { &root }, &[0], 17);
        let (q, working, original) = if case == 2 {
            let genuine = owner(&root, &[0], 17);
            assert!(matches!(
                resources(&dealer, &s, &input),
                Err(BuildError::Native(Error::Resource))
            ));
            owned_working(&dealer, &root, &s, &genuine)
        } else {
            owned_working(&dealer, &root, &s, &input)
        };
        assert_eq!(q.incoming_owner_bytes, 18 * 8 + size_of::<Input>());
        let mut working = working;
        if case == 0 {
            drop(working.split(1).unwrap());
        }
        let mut draws = 0;
        let result = construct_with(&mut dealer, s, input, working, |bytes| {
            draws += bytes.len();
            bytes.fill(71);
            Ok(())
        });
        if case == 0 || case == 2 {
            assert!(matches!(result, Err(BuildError::Native(Error::Resource))));
            assert_eq!(draws, 0);
        } else {
            let (_, coordinator, roles) = result.unwrap();
            drop((coordinator, roles));
            assert!(draws > 0);
        }
        drop(original);
    }
}

#[test]
fn pressure_after_quote_refuses_original_growth_without_counter_replacement() {
    let root = OwnedBudget::new(64 << 20).unwrap();
    let source = source();
    let s = spec(source, 1);
    let dealer = OwnedLocalDealer::new(source, [4; 32], [92; 32], &root).unwrap();
    let input = owner(&root, &[0], 0);
    let mut original = root
        .charge_external(bootstrap_bytes::<OwnedBudgetContext>())
        .unwrap();
    let q = resources(&dealer, &s, &input).unwrap();
    let capacity = (64 << 20) - root.budget().used_bytes();
    let pressure = root
        .charge_external(capacity - size_of::<OwnedExternalCharge>())
        .unwrap();
    let before = original.charged_bytes();
    assert!(original.resize_payload(q.total).is_err());
    assert_eq!(original.charged_bytes(), before);
    assert!(original.is_for(input.budget()));
    drop(pressure);
}

#[test]
fn failed_peer_early_finish_reentry_and_callback_unwind_burn_both() {
    for case in 0..4 {
        let (_mask, mut coordinator, [mut left, right], _) = issue(&[0]);
        let frame = left.begin(&mut || false).unwrap();
        match case {
            0 => assert!(left.advance(&[], &mut || false).is_err()),
            1 => assert!(left.finish().is_err()),
            2 => assert!(left
                .advance(&[], &mut || {
                    coordinator.cancel();
                    false
                })
                .is_err()),
            3 => assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = left.advance(&[], &mut || panic!("D29 callback unwind"));
            }))
            .is_err()),
            _ => unreachable!(),
        }
        assert!(coordinator.shared.control().aborted);
        assert!(frame.bytes().is_err());
        assert!(right.finish().is_err());
        for slot in &coordinator.shared.slots {
            let state = slot.state.lock().unwrap_or_else(|p| p.into_inner());
            assert_eq!(state.ledger.next, 18);
            assert_eq!(state.phase, 255);
            assert!(state.root.is_none());
            assert!(state.output.is_none());
        }
    }
}

#[test]
fn every_actual_constructor_allocation_failure_and_entropy_unwind_refunds() {
    let root = OwnedBudget::new(64 << 20).unwrap();
    let source = source();
    let s = spec(source, 1);
    let mut dealer = OwnedLocalDealer::new(source, [4; 32], [92; 32], &root).unwrap();
    let baseline = root.budget().used_bytes();
    let input = owner(&root, &[0], 0);
    let (_, working, original) = owned_working(&dealer, &root, &s, &input);
    root.budget().fail_after.store(100000, Ordering::SeqCst);
    let created = construct_with(&mut dealer, s, input, working, seeded()).unwrap();
    let count = 100000 - root.budget().fail_after.load(Ordering::SeqCst);
    root.budget().fail_after.store(-1, Ordering::SeqCst);
    drop((created, original));
    assert_eq!(root.budget().used_bytes(), baseline);
    for site in 0..count {
        let input = owner(&root, &[0], 0);
        let (_, working, original) = owned_working(&dealer, &root, &s, &input);
        root.budget().fail_after.store(site, Ordering::SeqCst);
        assert!(matches!(
            construct_with(&mut dealer, s, input, working, seeded()),
            Err(BuildError::Native(Error::Allocation))
        ));
        root.budget().fail_after.store(-1, Ordering::SeqCst);
        drop(original);
        assert_eq!(root.budget().used_bytes(), baseline, "factory {site}");
    }
    let input = owner(&root, &[0], 0);
    let (_, working, original) = owned_working(&dealer, &root, &s, &input);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = construct_with(&mut dealer, s, input, working, |_| {
            panic!("D29 entropy unwind")
        });
    }))
    .is_err());
    drop(original);
    assert_eq!(root.budget().used_bytes(), baseline);
}

#[test]
fn native_role_schedule_replay_swapped_lookup_and_wrong_role_frames_fail_closed() {
    for attack in 0..3 {
        let (_mask, coordinator, mut roles, _) = issue(&[0]);
        let mut frames = [
            roles[0].begin(&mut || false).unwrap(),
            roles[1].begin(&mut || false).unwrap(),
        ];
        let mut old: Option<Vec<u8>> = None;
        for round in 0..6 {
            if attack == 0 && round == 0 {
                assert!(roles[0]
                    .advance(frames[0].bytes().unwrap(), &mut || false)
                    .is_err());
                break;
            }
            if attack == 1 && round == 2 {
                assert!(roles[0]
                    .advance(old.as_ref().unwrap(), &mut || false)
                    .is_err());
                break;
            }
            if attack == 2 && round == 4 {
                old = Some(frames[1].bytes().unwrap().to_vec());
            }
            let a = roles[0]
                .advance(frames[1].bytes().unwrap(), &mut || false)
                .unwrap();
            let b = roles[1]
                .advance(frames[0].bytes().unwrap(), &mut || false)
                .unwrap();
            if attack == 1 && round == 0 {
                old = Some(frames[1].bytes().unwrap().to_vec());
            }
            let (Step::Frame(a), Step::Frame(b)) = (a, b) else {
                panic!("early Ready")
            };
            frames = [a, b];
        }
        if attack == 2 {
            assert!(roles[0]
                .advance(old.as_ref().unwrap(), &mut || false)
                .is_err());
        }
        assert!(coordinator.shared.control().aborted);
        assert!(frames.iter().all(|frame| frame.bytes().is_err()));
    }
}

// Frozen independent v1 edge/use encoders: do not call production digest helpers.
fn framed(parts: &[&[u8]]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for p in parts {
        bytes.extend_from_slice(&(p.len() as u64).to_le_bytes());
        bytes.extend_from_slice(p);
    }
    bytes
}
fn hash(parts: &[&[u8]]) -> [u8; 32] {
    Sha256::digest(framed(parts)).into()
}
fn golden(hex: &str) -> [u8; 32] {
    assert_eq!(hex.len(), 64);
    std::array::from_fn(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap())
}
fn edge_bytes(edge: EdgeDescriptor) -> Vec<u8> {
    let i = edge.identity();
    let s = i.source();
    let mut b = b"pllm.edge-contract\0v1\0".to_vec();
    b.extend(framed(&[
        &s.checkpoint(),
        &s.numeric_profile(),
        &s.implementation(),
        &i.plan(),
        &i.producer(),
        &i.wire(),
    ]));
    b.extend(i.first_tensor_index().to_le_bytes());
    b.push(edge.shape().dimensions().len() as u8);
    for d in edge.shape().dimensions() {
        b.extend((*d as u64).to_le_bytes());
    }
    b.extend((edge.shape().elements() as u64).to_le_bytes());
    b.extend([
        edge.numeric().ring_bits(),
        edge.numeric().fractional_bits(),
        match edge.numeric().interpretation() {
            Interpretation::Signed => 0,
            Interpretation::Unsigned => 1,
            Interpretation::Predicate => 2,
        },
        if edge.representation() == Representation::CommonXorMasked {
            4
        } else {
            3
        },
        0,
    ]);
    b
}

#[test]
fn independent_frozen_spec_coordinate_and_stage_preimage_vectors_for_review() {
    let s = spec(SourceIdentity::new([1; 32], [2; 32], [3; 32]), 2);
    let hashes = [[0xa0; 32], [0xb0; 32]];
    let mut expected = framed(&[
        b"pllm.sigma.nexp-region.v1",
        b"declaration.v1",
        b"floor(E0[min(X,65535)%256]*E1[min(X,65535)/256]/4096)",
        b"f911b715b0e0523fa4d77f21c28c6a0440c5be781e67b5b30419f20e505a6ebf",
        b"aa0f2f860266fb79901405ef5e9965b8c9fe2e88",
        &[1; 32],
        &[2; 32],
        &[3; 32],
        &[4; 32],
        &[92; 32],
        &[90; 32],
        &[91; 32],
        &[0; 16],
        &[0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        &[0; 16],
        &[1, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        &[0, 16, 0, 0, 0, 0, 0, 0],
        &[0xa0; 32],
        &[0xb0; 32],
        &[12, 4, 50, 12, 16, 8, 0, 50, 24],
        &[2, 0, 0, 0, 0, 0, 0, 0],
        &[7, 0, 0, 0, 0, 0, 0, 0],
    ]);
    let edges: [[u8; 32]; 14] =
        std::array::from_fn(|i| Sha256::digest(edge_bytes(s.edges[i])).into());
    for edge in edges {
        expected.extend(framed(&[&edge]));
    }
    let frozen_map = [0, 0, 1, 2, 3, 1, 4, 5, 6, 7, 7, 9, 8, 9, 10, 11, 12, 13];
    for (ordinal, index) in frozen_map.into_iter().enumerate() {
        let mut b = b"pllm.consumer-use\0v1\0".to_vec();
        b.extend(edges[index]);
        b.extend(if ordinal == 17 { [91; 32] } else { [90; 32] });
        b.extend((u16::from(ordinal == 7 || ordinal == 15)).to_le_bytes());
        b.extend((ordinal as u32).to_le_bytes());
        let digest: [u8; 32] = Sha256::digest(b).into();
        expected.extend(framed(&[&digest]));
    }
    assert_eq!(
        spec_digest(&s, hashes, [92; 32], &usage(&s)),
        <[u8; 32]>::from(Sha256::digest(expected))
    );
    assert_eq!(
        spec_digest(&s, hashes, [92; 32], &usage(&s)),
        golden("61459e90b393f3755e0d8afb590dfd0621a412deadb6a9333d35bf2c19a8bd7e")
    );
    for (index, tag) in [
        (0, b"low8.v1".as_slice()),
        (1, b"high8-tr16-8.v1".as_slice()),
    ] {
        let expected = hash(&[
            b"pllm.sigma.nexp-coordinate.v1",
            b"declaration.v1",
            tag,
            &edges[7],
            &edges[8 + index],
            &hashes[index],
            &[7, 0, 0, 0, 0, 0, 0, 0],
            &[2, 0, 0, 0, 0, 0, 0, 0],
            &[8, 0],
            &[0, 0],
            &[0, 1],
        ]);
        assert_eq!(coordinate(&s, hashes, index), expected);
        assert_eq!(
            expected,
            golden(
                [
                    "bf2026ab18e825e6c9328a367108180badca53827aaf3563338d45a4e0a32351",
                    "264f698ee6201c556e0cc78ea9f33c2367fd6c6df1c98d517048d7ce1d0d1099",
                ][index]
            )
        );
    }
    assert_eq!(
        stage([0x55; 32], 9),
        hash(&[&[0x55; 32], b"stage.v1", &[9, 0]])
    );
    assert_eq!(
        stage([0x55; 32], 9),
        golden("f0093f9dc41d3b1e9c46ed437e1255908eda4fcc29ef96ba8af374c524f828bf")
    );
}

#[test]
fn frozen_complete_runtime_records_match_independent_stdlib_serialization_vectors() {
    // Synthetic metadata ONLY, never issued masks/material or source/range proof.
    // Expected literals were computed independently from the documented v1 bytes
    // with Python stdlib, after all selected floors passed the rational proof.
    let s = spec(SourceIdentity::new([1; 32], [2; 32], [3; 32]), 2);
    let hashes = [[0xa0; 32], [0xb0; 32]];
    let intervals = [
        (0, 1i128 << 38),
        (0, 1i128 << 38),
        (-65536, (1i128 << 38) - 65536),
        (0, 2),
        (0, 2),
        (0, 65536),
        (0, 65536),
        (0, 65536),
        (0, 256),
        (0, 256),
        (0, 4097),
        (0, 4097),
        (0, (1i128 << 24) + 1),
        (0, 4097),
    ];
    let bindings: [Binding; 14] = std::array::from_fn(|i| Binding {
        descriptor: s.edges[i],
        session: [92; 32],
        namespace: [6; 32],
        issuance: i as u64,
        mask_id: [0x70 + i as u8; 32],
        consumer: if i == 13 { s.next } else { s.operation },
        interval: Some(Interval {
            lower: intervals[i].0,
            upper: intervals[i].1,
        }),
    });
    for (index, expected) in [
        (
            0,
            "9736e978578ea3c53714acc457d3e83b2aed138f184ee3d8325b2791f6525537",
        ),
        (
            12,
            "c6fb9ec229f645b934ff89e16b310142245f6f09ee78ebb2f9724979219f6540",
        ),
        (
            13,
            "68286f03e46021cf493abfa15e882a23bcfd80613c0a18b65b2f10a2426cb957",
        ),
    ] {
        assert_eq!(bindings[index].digest(), golden(expected));
    }
    let coordinates = [coordinate(&s, hashes, 0), coordinate(&s, hashes, 1)];
    let provenance_vectors = [
        [
            "4a407febd4def87d82da557e2e08aa89e9f5bc955640dd86a7edb5d45302eee7",
            "6dcae6879f14588883cb8df36c5e06d38b77ea00fd58c66d97ac23468c8bda3f",
        ],
        [
            "cd319acf42ffbf21bae74a2b92a0690742cbf0884ccfade4a823a1eb02b67d8f",
            "43fbd5e391d1cfeee5cd7892f0a6ca9adbeb2b9186c2ac8b7206b788ae655c2d",
        ],
        [
            "aa42da7a6aef89e65ec8d7e688b69c61b0ad216f3aac6bf57a31e9f295ada557",
            "0b937d6ae391c0ee266904c58a3e96f80ff3d2d2333261f0eebb0200bdb39a28",
        ],
    ];
    let ledger_vectors = [
        "1826b75f44eddb21c2b742fecb024c9d812c7140b2722a175f7fedb49c6ad5af",
        "a84564d8c58dc9e0b61038bf55163f18b95abcb7b3aa6e088421547d10db1e5d",
        "0d1b6b1d4f6bb8277ba7f3431c1fc26e81ea99ddfec1d680a9612891758463b5",
    ];
    let ledgers: [_; 3] = std::array::from_fn(|role| {
        for (i, coordinate) in coordinates.iter().enumerate() {
            assert_eq!(
                provenance(*coordinate, &bindings, i, role as u8),
                golden(provenance_vectors[role][i])
            );
        }
        let ledger = internal_use::Ledger {
            uses: usage(&s),
            bindings: MAP.map(|i| bindings[i]),
            role: role as u8,
            next: 0,
        };
        let digest = ledger_digest(coordinates, &bindings, &ledger);
        assert_eq!(digest, golden(ledger_vectors[role]));
        digest
    });
    let region = region_digest(
        spec_digest(&s, hashes, [92; 32], &usage(&s)),
        &bindings,
        ledgers,
        &[0xc0; 32],
    );
    assert_eq!(
        region,
        golden("1314063b70d4ac3ea2f54f13502642218add6556533b6058b30bc9ae62762b1a")
    );
    assert_eq!(
        stage(region, 9),
        golden("e4d35339159868955c2ed494b9af9e98c1d7c6a1b5ff601eda490b1fe3bc960b")
    );
}

#[test]
fn original_masked_metadata_and_d27_outputs_cannot_be_clear_input_owners() {
    macro_rules! no_into {
        ($from:ty,$to:ty) => {{
            trait Ambiguous<A> {
                fn check() {}
            }
            impl<T> Ambiguous<()> for T {}
            struct Imported;
            impl<T: Into<$to>> Ambiguous<Imported> for T {}
            let _ = <$from as Ambiguous<_>>::check;
        }};
    }
    no_into!(EdgeDescriptor, Input);
    no_into!(BoundRange<LocalAssumption>, Input);
    no_into!(CommonMasked<'static, Ring>, Input);
    no_into!(
        crate::masked_wire_reference::typed::linear_activation::LinearActivationRoleOutputFor<
            OwnedBudgetContext,
        >,
        Input
    );
}

// No shared.control/abort/bytes/descriptor call before these raw snapshots:
// poison recovery must not manufacture synchronous retirement for the oracle.
fn immediate_abort(shared: &Shared<OwnedBudgetContext>) {
    let control = match shared.control.try_lock() {
        Ok(guard) => guard,
        Err(std::sync::TryLockError::Poisoned(poison)) => poison.into_inner(),
        Err(std::sync::TryLockError::WouldBlock) => panic!("metadata lock retained after failure"),
    };
    assert!(control.aborted);
    for slot in &shared.slots {
        let state = match slot.state.try_lock() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::Poisoned(poison)) => poison.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => panic!("state lock retained after failure"),
        };
        assert_eq!(state.phase, 255);
        assert_eq!(state.ledger.next, 18);
        assert!(state.output.is_none() && state.root.is_none() && state.product.is_none());
        assert!(state.low.is_none() && state.high.is_none() && state.gap.is_none());
        assert!(state.u0.is_none() && state.u1.is_none() && state.p.is_none());
        assert!(state
            .work
            .iter()
            .all(|words| words.iter().all(|&word| word == 0)));
    }
}

#[test]
fn actual_region_one_ready_cannot_extract_or_leave_finished_sibling_material() {
    let (_mask, coordinator, [mut left, mut right], _) = issue(&[0]);
    let mut frames = [
        left.begin(&mut || false).unwrap(),
        right.begin(&mut || false).unwrap(),
    ];
    for _ in 0..10 {
        let a = left
            .advance(frames[1].bytes().unwrap(), &mut || false)
            .unwrap();
        let b = right
            .advance(frames[0].bytes().unwrap(), &mut || false)
            .unwrap();
        let (Step::Frame(a), Step::Frame(b)) = (a, b) else {
            panic!("early Ready")
        };
        frames = [a, b];
    }
    assert!(matches!(
        left.advance(frames[1].bytes().unwrap(), &mut || false)
            .unwrap(),
        Step::Ready
    ));
    assert_eq!(
        coordinator.shared.control.lock().unwrap().ready,
        [true, false]
    );
    assert!(coordinator.shared.slots[0]
        .state
        .lock()
        .unwrap()
        .output
        .is_some());
    assert!(left.finish().is_err());
    immediate_abort(&coordinator.shared);
    assert!(right.finish().is_err());
    assert!(frames.iter().all(|frame| frame.bytes().is_err()));
}

#[test]
fn actual_region_native_lock_and_unlocked_callback_panics_retire_before_recovery() {
    for locked in [false, true] {
        let (_mask, coordinator, [mut left, _right], _) = issue(&[0]);
        let frame = left.begin(&mut || false).unwrap();
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if locked {
                let _: Result<(), Error> = coordinator.shared.operation(|_| {
                    let mut state = coordinator.shared.slots[0].state.lock().unwrap();
                    state.work[0][0] = 42;
                    panic!("actual nExp locked-work unwind")
                });
            } else {
                let _ = left.advance(&[], &mut || panic!("actual nExp callback unwind"));
            }
        }))
        .is_err());
        immediate_abort(&coordinator.shared);
        assert!(frame.bytes().is_err());
    }
}

#[test]
fn final_mask_and_either_output_drop_jointly_revoke_retained_frames_without_cycles() {
    for drop_mask in [false, true] {
        let weak = {
            let (mask, coordinator, mut roles, _) = issue(&[0, 65535]);
            let weak = Arc::downgrade(&coordinator.shared);
            let frames = complete(&mut roles);
            let [left, right] = roles;
            let a = left.finish().unwrap();
            let b = right.finish().unwrap();
            if drop_mask {
                drop(mask);
            } else {
                drop(a);
            }
            immediate_abort(&coordinator.shared);
            assert!(b.descriptor().is_err());
            assert!(frames.iter().flatten().all(|frame| frame.bytes().is_err()));
            drop((coordinator, b, frames));
            // In drop-mask branch a is still a legitimate owner until scope end;
            // in output branch mask is still a legitimate owner. No Weak detach.
            assert!(weak.upgrade().is_some());
            weak
        };
        assert!(weak.upgrade().is_none());
    }
}

#[test]
fn exhausted_binding_counter_is_rejected_before_operation_entropy_or_factory() {
    let root = OwnedBudget::new(64 << 20).unwrap();
    let source = source();
    let s = spec(source, 1);
    let mut dealer = OwnedLocalDealer::new(source, [4; 32], [92; 32], &root).unwrap();
    let input = owner(&root, &[0], 0);
    let (_, working, original) = owned_working(&dealer, &root, &s, &input);
    dealer.counter = u64::MAX - 13;
    root.budget().fail_after.store(0, Ordering::SeqCst);
    let mut draws = 0;
    assert!(matches!(
        construct_with(&mut dealer, s, input, working, |_| {
            draws += 1;
            Ok(())
        }),
        Err(BuildError::Native(Error::Resource))
    ));
    assert_eq!(draws, 0);
    assert_eq!(dealer.counter, u64::MAX - 13);
    assert_eq!(root.budget().fail_after.load(Ordering::SeqCst), 0);
    root.budget().fail_after.store(-1, Ordering::SeqCst);
    drop(original);
}

#[test]
fn single_original_rng_path_matches_total_native_quote_and_every_error_burns_partial_owners() {
    let root = OwnedBudget::new(64 << 20).unwrap();
    let source = source();
    let s = spec(source, 1);
    let mut dealer = OwnedLocalDealer::new(source, [4; 32], [92; 32], &root).unwrap();
    let baseline = root.budget().used_bytes();
    let input = owner(&root, &[0], 0);
    let (q, working, original) = owned_working(&dealer, &root, &s, &input);
    let mut fill = seeded();
    let mut calls = 0;
    let mut bytes = 0;
    let created = construct_with(&mut dealer, s, input, working, |output| {
        calls += 1;
        bytes += output.len();
        fill(output)
    })
    .unwrap();
    assert_eq!(bytes, q.work.random_byte_requests);
    drop((created, original));
    assert_eq!(root.budget().used_bytes(), baseline);
    for failure in 1..=calls {
        let input = owner(&root, &[0], 0);
        let (_, working, original) = owned_working(&dealer, &root, &s, &input);
        let mut fill = seeded();
        let mut current = 0;
        assert!(matches!(
            construct_with(&mut dealer, s, input, working, |output| {
                current += 1;
                if current == failure {
                    return Err(Error::Randomness);
                }
                fill(output)
            }),
            Err(BuildError::Native(Error::Randomness))
        ));
        assert_eq!(current, failure);
        drop(original);
        assert_eq!(root.budget().used_bytes(), baseline, "draw {failure}");
    }
}

#[test]
fn concrete_region_handles_are_neither_send_nor_sync() {
    macro_rules! local {
        ($ty:ty) => {{
            trait Ambiguous<A> {
                fn check() {}
            }
            impl<T> Ambiguous<()> for T {}
            struct Sendable;
            impl<T: Send> Ambiguous<Sendable> for T {}
            let _ = <$ty as Ambiguous<_>>::check;
        };
        {
            trait Ambiguous<A> {
                fn check() {}
            }
            impl<T> Ambiguous<()> for T {}
            struct Syncable;
            impl<T: Sync> Ambiguous<Syncable> for T {}
            let _ = <$ty as Ambiguous<_>>::check;
        }};
    }
    local!(Coordinator<OwnedBudgetContext>);
    local!(Role<OwnedBudgetContext>);
    local!(Frame<OwnedBudgetContext>);
    local!(Output<OwnedBudgetContext>);
    local!(Mask<OwnedBudgetContext>);
}

#[test]
fn original_os_entropy_constructor_runs_actual_full_region_not_metadata_shell() {
    let root = OwnedBudget::new(64 << 20).unwrap();
    let source = source();
    let s = spec(source, 1);
    let mut dealer = OwnedLocalDealer::new(source, [4; 32], [92; 32], &root).unwrap();
    let input = owner(&root, &[0], 3);
    let (_, working, original) = owned_working(&dealer, &root, &s, &input);
    let (mask, coordinator, mut roles) = construct(&mut dealer, s, input, working).unwrap();
    let frames = complete(&mut roles);
    let [left, right] = roles;
    let outputs = [left.finish().unwrap(), right.finish().unwrap()];
    for output in &outputs {
        let state = output.slot.state.lock().unwrap();
        let words = state.output.as_ref().unwrap().backing.values();
        assert_eq!(
            words[0].wrapping_sub(mask.native.backing.values()[0]) & ring_mask(50),
            4096
        );
        assert_eq!(state.ledger.next, 18);
    }
    drop((coordinator, outputs, frames, mask, original));
}

#[test]
fn post_quote_pressure_refuses_growth_atomically_but_never_requires_recharge_after_full_admission()
{
    let root = OwnedBudget::new(64 << 20).unwrap();
    let source = source();
    let s = spec(source, 1);
    let mut dealer = OwnedLocalDealer::new(source, [4; 32], [92; 32], &root).unwrap();
    let input = owner(&root, &[0], 3);
    let mut original = root
        .charge_external(bootstrap_bytes::<OwnedBudgetContext>())
        .unwrap();
    let q = resources(&dealer, &s, &input).unwrap();
    let before = original.payload_bytes();
    let pressure =
        LeaseFor::reserve(&dealer.budget, (64 << 20) - root.budget().used_bytes()).unwrap();
    assert!(original.resize_payload(q.total).is_err());
    assert_eq!(original.payload_bytes(), before);
    assert_eq!(root.budget().used_bytes(), 64 << 20);
    drop(pressure);
    original.resize_payload(q.total).unwrap();
    let working = original.split_native(root.budget(), q.total).unwrap();
    let pressure =
        LeaseFor::reserve(&dealer.budget, (64 << 20) - root.budget().used_bytes()).unwrap();
    let (mask, coordinator, mut roles) =
        construct_with(&mut dealer, s, input, working, seeded()).unwrap();
    let frames = complete(&mut roles);
    let [left, right] = roles;
    let outputs = [left.finish().unwrap(), right.finish().unwrap()];
    assert!(root.budget().used_bytes() <= 64 << 20);
    assert_eq!(frames.len(), 11);
    drop((coordinator, outputs, frames, mask, pressure, original));
}

#[path = "nexp_region_lifecycle_tests.rs"]
mod lifecycle_tests;

#[path = "nexp_region_inventory_tests.rs"]
mod independent_inventory;
