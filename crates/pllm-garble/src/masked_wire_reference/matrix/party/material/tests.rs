use super::*;
use crate::masked_wire_reference::matrix::{MatrixLayout, LIVE, TEST_SERIAL};

fn spec(bits: u8, layout: MatrixLayout) -> Spec {
    Spec {
        bits,
        rows: 2,
        inner: 2,
        columns: 2,
        layout,
    }
}
fn context() -> Context {
    Context {
        plan: [1; 32],
        session: [2; 32],
        operation: [3; 32],
        input_wire: [4; 32],
        weight_wire: [5; 32],
        output_wire: [6; 32],
    }
}
fn issue(s: Spec, budget: &Budget) -> [Party<'_>; 2] {
    // Deterministic issuance only; transfer cannot call this entropy callback.
    super::super::issue_with(
        s,
        context(),
        [&[1; 4], &[1; 4], &[1; 4]],
        budget,
        &mut |bytes| {
            bytes.fill(0x73);
            Ok(())
        },
        &mut || false,
    )
    .unwrap()
}
fn budget() -> Budget {
    Budget::new(MAX_ALLOCATION_BYTES).unwrap()
}

#[test]
fn roundtrip_layouts_widths_and_retained_outputs_frames() {
    let _guard = TEST_SERIAL.lock().unwrap();
    for layout in [MatrixLayout::InputOutput, MatrixLayout::OutputInput] {
        for bits in [1, 2, 7, 32, 63, 64] {
            let s = spec(bits, layout);
            let b = budget();
            let [left, right] = issue(s, &b);
            let identity = left.material_identity().unwrap();
            let (lp, lb) = left.export_material([9; 32]).unwrap();
            let (rp, rb) = right.export_material([9; 32]).unwrap();
            assert_eq!(
                lb.as_bytes().len(),
                handoff_resources(s).unwrap().packet_bytes
            );
            let mut left = lp.import(lb).unwrap();
            let mut right = rp.import(rb).unwrap();
            assert_eq!(left.material_identity().unwrap(), identity);
            let operand = 3 & core::mask(bits);
            let lf = left
                .begin(&[operand; 4], &[operand; 4], &mut || false)
                .unwrap();
            assert_eq!(left.material_identity(), Err(Error::Consumed));
            let rf = right
                .begin(&[operand; 4], &[operand; 4], &mut || false)
                .unwrap();
            let lo = left.finish(rf.as_bytes()).unwrap();
            let ro = right.finish(lf.as_bytes()).unwrap();
            let expected = (2 * operand
                .wrapping_sub(1)
                .wrapping_mul(operand.wrapping_sub(1))
                + 1)
                & core::mask(bits);
            assert_eq!(lo.values(), &[expected; 4]);
            assert_eq!(lo.values(), ro.values());
            drop((left, right, ro, rf));
            assert!(LIVE.load(Ordering::SeqCst));
            drop(lo);
            assert!(LIVE.load(Ordering::SeqCst));
            drop(lf);
            assert!(!LIVE.load(Ordering::SeqCst));
            assert_eq!(b.used_bytes(), 0);
        }
    }
}

#[test]
fn packet_and_permit_independently_hold_singleton_and_charges() {
    let _guard = TEST_SERIAL.lock().unwrap();
    for retain_packet in [false, true] {
        let b = budget();
        let s = spec(8, MatrixLayout::InputOutput);
        let [left, right] = issue(s, &b);
        drop(right);
        let (permit, packet) = left.export_material([7; 32]).unwrap();
        let cost = handoff_resources(s).unwrap();
        let base = resources(s).unwrap().party_reservation_bytes;
        assert_eq!(b.used_bytes(), base + cost.incremental_reservation_bytes);
        if retain_packet {
            drop(permit);
            assert_eq!(
                b.used_bytes(),
                cost.packet_bytes + size_of::<Packet<'static>>() + ALLOCATOR
            );
            assert!(matches!(Live::acquire(), Err(Error::Resource)));
            drop(packet);
        } else {
            drop(packet);
            assert_eq!(
                b.used_bytes(),
                base + size_of::<ImportPermit<'static>>() + HASH_STACK
            );
            assert!(matches!(Live::acquire(), Err(Error::Resource)));
            drop(permit);
        }
        assert!(!LIVE.load(Ordering::SeqCst));
        assert_eq!(b.used_bytes(), 0);
    }
}

#[test]
fn exact_incremental_admission_and_all_transfer_allocation_failures() {
    let _guard = TEST_SERIAL.lock().unwrap();
    let s = spec(8, MatrixLayout::OutputInput);
    let base = resources(s).unwrap().party_reservation_bytes;
    let extra = handoff_resources(s).unwrap().incremental_reservation_bytes;
    for short in [false, true] {
        let limit = (base + extra).max(resources(s).unwrap().reservation_bytes);
        let b = Budget::new(limit).unwrap();
        // Keep exact export admission despite issuance dealer/sibling peak.
        let [left, right] = issue(s, &b);
        let used = b.used_bytes();
        drop(right);
        assert_eq!(used, 2 * base);
        let blocker = b
            .reserve(limit - (base + extra - usize::from(short)))
            .unwrap();
        let result = left.export_material([7; 32]);
        if short {
            assert!(matches!(result, Err(Error::Resource)));
        } else {
            let (permit, packet) = result.unwrap();
            assert_eq!(b.used_bytes(), base + extra + blocker.bytes);
            let party = permit.import(packet).unwrap();
            assert_eq!(b.used_bytes(), base + blocker.bytes);
            drop(party);
        }
        drop(blocker);
        assert_eq!(b.used_bytes(), 0);
    }
    // Codec packet allocation, then each of three recreated arrays.
    for fail in 0..4 {
        let b = budget();
        let [left, right] = issue(s, &b);
        b.fail_after
            .store(if fail == 0 { 0 } else { -1 }, Ordering::SeqCst);
        let exported = left.export_material([7; 32]);
        if fail == 0 {
            assert!(matches!(exported, Err(Error::Allocation)));
        } else {
            let (permit, packet) = exported.unwrap();
            b.fail_after.store(fail - 1, Ordering::SeqCst);
            assert!(matches!(permit.import(packet), Err(Error::Allocation)));
        }
        assert!(LIVE.load(Ordering::SeqCst)); // sibling untouched
        drop(right);
        assert_eq!(b.used_bytes(), 0);
        assert!(!LIVE.load(Ordering::SeqCst));
    }
}

#[test]
fn rejects_swaps_changed_canonical_shares_and_noncanonical_residues() {
    let _guard = TEST_SERIAL.lock().unwrap();
    let s = spec(7, MatrixLayout::InputOutput);
    for case in 0..13 {
        let b = budget();
        let [left, right] = issue(s, &b);
        drop(right);
        let (mut permit, mut packet) = left.export_material([7; 32]).unwrap();
        let bytes = packet.as_bytes();
        let mut body = bytes[material_wire::HEADER..bytes.len() - 32].to_vec();
        let mut magic = *MATERIAL_MAGIC;
        let mut kind = KIND;
        let mut role = permit.role;
        let mut binding = permit.descriptor_digest;
        let mut nonce = permit.binding;
        match case {
            0 => body[0] ^= 1,  // canonical changed A + recomputed checksum
            1 => body[32] ^= 1, // B
            2 => body[64] ^= 1, // C
            3 => magic[0] ^= 1,
            4 => kind ^= 1,
            5 => role ^= 1,
            6 => binding[0] ^= 1,
            7 => nonce[0] ^= 1,
            8 => {
                body.pop();
            }
            9 => {
                let last_word = body.len() - 8;
                body[last_word] = 0x80;
            }
            10 => permit.spec.rows = 1,
            11 => permit.spec.bits = 0,
            _ => permit.role = 2,
        }
        packet.packet = material_wire::packet(
            &magic,
            kind,
            role,
            binding,
            nonce,
            body.len(),
            &b,
            |bytes| {
                bytes.extend_from_slice(&body);
                Ok(())
            },
        )
        .unwrap();
        if case >= 9 {
            // Test-only bypass of trusted digest to exercise residue preflight.
            permit.packet_digest = Sha256::digest(packet.as_bytes()).into();
        }
        b.fail_after.store(0, Ordering::SeqCst);
        assert!(matches!(
            permit.import(packet),
            Err(Error::Context | Error::Domain)
        ));
        assert_eq!(b.used_bytes(), 0);
        assert!(!LIVE.load(Ordering::SeqCst));
    }
}

#[test]
fn cross_role_packets_consume_both_local_attempts() {
    let _guard = TEST_SERIAL.lock().unwrap();
    let b = budget();
    let [left, right] = issue(spec(8, MatrixLayout::InputOutput), &b);
    let (lp, lb) = left.export_material([7; 32]).unwrap();
    let (rp, rb) = right.export_material([7; 32]).unwrap();
    b.fail_after.store(0, Ordering::SeqCst);
    assert!(matches!(lp.import(rb), Err(Error::Context)));
    assert!(LIVE.load(Ordering::SeqCst));
    assert!(matches!(rp.import(lb), Err(Error::Context)));
    assert_eq!(b.used_bytes(), 0);
    assert!(!LIVE.load(Ordering::SeqCst));
}

#[test]
fn wrong_phase_export_consumes_party_without_releasing_retained_frame() {
    let _guard = TEST_SERIAL.lock().unwrap();
    let b = budget();
    let [mut left, right] = issue(spec(8, MatrixLayout::InputOutput), &b);
    let frame = left.begin(&[3; 4], &[3; 4], &mut || false).unwrap();
    assert!(matches!(
        left.export_material([7; 32]),
        Err(Error::Consumed)
    ));
    drop(right);
    assert!(matches!(Live::acquire(), Err(Error::Resource)));
    assert!(b.used_bytes() > 0);
    drop(frame);
    assert_eq!(b.used_bytes(), 0);
    assert!(!LIVE.load(Ordering::SeqCst));
}

#[test]
fn handoff_preserves_issuance_binding_without_entropy_or_new_issuance() {
    let _guard = TEST_SERIAL.lock().unwrap();
    let b = budget();
    let draws = std::cell::Cell::new(0usize);
    let [left, right] = super::super::issue_with(
        spec(64, MatrixLayout::InputOutput),
        context(),
        [&[1; 4], &[1; 4], &[1; 4]],
        &b,
        &mut |bytes| {
            draws.set(draws.get() + 1);
            bytes.fill(0);
            Ok(())
        },
        &mut || false,
    )
    .unwrap();
    let identity = left.material_identity().unwrap();
    let count = draws.get();
    let issuance = super::super::super::ISSUANCE.load(Ordering::SeqCst);
    let (permit, packet) = left.export_material([7; 32]).unwrap();
    let imported = permit.import(packet).unwrap();
    assert_eq!(imported.material_identity().unwrap(), identity);
    assert_eq!(draws.get(), count);
    assert_eq!(
        super::super::super::ISSUANCE.load(Ordering::SeqCst),
        issuance
    );
    drop((imported, right));
    assert_eq!(b.used_bytes(), 0);
}
