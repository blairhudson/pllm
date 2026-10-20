use super::*;
use crate::{
    fusefss_backend::{self as backend, conversions as c},
    shared_rescale_reference::Context,
};

fn context() -> Context {
    Context {
        plan: [1; 32],
        session: [2; 32],
        operation: [3; 32],
        first_tensor_index: 0,
    }
}
fn spec(operation: c::Operation, input_bits: u8, output_bits: u8) -> c::Spec {
    c::Spec {
        operation,
        input_bits,
        output_bits,
        lanes: 1,
        source_digest: [4; 32],
        profile_digest: [5; 32],
    }
}

#[test]
fn conversions_roundtrip_and_replay() {
    let dealer = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let worker = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut guard = ImportGuard::new(64, &worker).unwrap();
    for (s, input, expected) in [
        (spec(c::Operation::ZeroExtend, 5, 9), 31, 31),
        (
            spec(
                c::Operation::SignExtend {
                    range: c::SignedRange::full_ring(5).unwrap(),
                },
                5,
                9,
            ),
            31,
            511,
        ),
        (
            spec(
                c::Operation::Ars {
                    shift: 2,
                    range: c::SignedRange::full_ring(5).unwrap(),
                },
                5,
                5,
            ),
            27,
            30,
        ),
        (
            spec(
                c::Operation::Tr {
                    shift: 2,
                    range: c::SignedRange::full_ring(5).unwrap(),
                },
                5,
                3,
            ),
            27,
            6,
        ),
        (spec(c::Operation::BooleanToArithmetic, 1, 9), 1, 1),
    ] {
        let [a, b] = c::issue(s, context(), &dealer).unwrap();
        let pa = a.export_material().unwrap();
        let pb = b.export_material().unwrap();
        let nonce = pa.nonce();
        assert_eq!(nonce, pb.nonce());
        let replay = MaterialPacket::from_bytes(pa.as_bytes(), &worker).unwrap();
        let mut a =
            c::Party::import_material(s, context(), 0, nonce, pa, &mut guard, &worker).unwrap();
        let mut b =
            c::Party::import_material(s, context(), 1, nonce, pb, &mut guard, &worker).unwrap();
        assert!(matches!(
            c::Party::import_material(s, context(), 0, nonce, replay, &mut guard, &worker),
            Err(Error::Consumed)
        ));
        let fa = a.start(&[0]).unwrap();
        let fb = b.start(&[input]).unwrap();
        let oa = a.finish(fb.as_bytes()).unwrap();
        let ob = b.finish(fa.as_bytes()).unwrap();
        assert_eq!(
            (oa.shares()[0] + ob.shares()[0]) & ((1 << s.output_bits) - 1),
            expected
        );
    }
    assert_eq!(dealer.used_bytes(), 0);
    drop(guard);
    assert_eq!(worker.used_bytes(), 0);
}

#[test]
fn scalar_both_backends_roundtrip() {
    for mode in [
        backend::LookupBackend::PrivateVector,
        backend::LookupBackend::PublicBoundary,
    ] {
        let dealer = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
        let worker = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
        let compiled = backend::CompiledGelu::with_backend(&dealer, mode).unwrap();
        let public = backend::CompiledGelu::with_backend(&worker, mode).unwrap();
        let mut guard = ImportGuard::new(4, &worker).unwrap();
        let [a, b] = compiled.issue(4, context()).unwrap();
        let pa = a.export_material().unwrap();
        let pb = b.export_material().unwrap();
        let nonce = pa.nonce();
        let mut a = public
            .import_material(4, context(), 0, nonce, pa, &mut guard)
            .unwrap();
        let mut b = public
            .import_material(4, context(), 1, nonce, pb, &mut guard)
            .unwrap();
        drop(public); // Imported owners borrow only Budget, never descriptor.
        let fa = a.start(&[0; 4]).unwrap();
        let fb = b.start(&[0, 4096, 8192, 12288]).unwrap();
        let ga = a.advance(fb.as_bytes()).unwrap();
        let gb = b.advance(fa.as_bytes()).unwrap();
        let oa = a.finish(gb.as_bytes()).unwrap();
        let ob = b.finish(ga.as_bytes()).unwrap();
        // Frozen descriptor's independent clear interpreter is already covered
        // elsewhere; compare against separately issued untransported owners.
        let [mut ca, mut cb] = compiled.issue(4, context()).unwrap();
        let fca = ca.start(&[0; 4]).unwrap();
        let fcb = cb.start(&[0, 4096, 8192, 12288]).unwrap();
        let gca = ca.advance(fcb.as_bytes()).unwrap();
        let gcb = cb.advance(fca.as_bytes()).unwrap();
        let oca = ca.finish(gcb.as_bytes()).unwrap();
        let ocb = cb.finish(gca.as_bytes()).unwrap();
        for i in 0..4 {
            assert_eq!(
                (oa.shares()[i] + ob.shares()[i]) & ((1 << 50) - 1),
                (oca.shares()[i] + ocb.shares()[i]) & ((1 << 50) - 1)
            );
        }
    }
}

#[test]
fn chunked_lookup_roundtrip_and_table_identity() {
    use backend::lookup::{Party, Table};
    let dealer = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let worker = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let values: Vec<_> = (0..300).map(|x| (x * 7) % 512).collect();
    let table = Table::new_zero_extended(9, &values, [7; 32], &dealer).unwrap();
    let public = Table::new_zero_extended(9, &values, [7; 32], &worker).unwrap();
    let mut guard = ImportGuard::new(16, &worker).unwrap();
    let [a, b] = table.issue(4, context()).unwrap();
    let pa = a.export_material().unwrap();
    let pb = b.export_material().unwrap();
    let nonce = pa.nonce();
    let mut a = Party::import_material(&public, 4, context(), 0, nonce, pa, &mut guard).unwrap();
    let mut b = Party::import_material(&public, 4, context(), 1, nonce, pb, &mut guard).unwrap();
    let fa = a.start(&[0; 4]).unwrap();
    let fb = b.start(&[0, 255, 299, 400]).unwrap();
    let oa = a.finish(fb.as_bytes()).unwrap();
    let ob = b.finish(fa.as_bytes()).unwrap();
    for (i, x) in [0, 255, 299, 400].into_iter().enumerate() {
        assert_eq!(
            (oa.shares()[i] + ob.shares()[i]) & 511,
            values.get(x).copied().unwrap_or(0)
        );
    }
    let [a, b] = table.issue(1, context()).unwrap();
    drop(b);
    let packet = a.export_material().unwrap();
    let nonce = packet.nonce();
    let wrong = Table::new_zero_extended(9, &[1; 300], [7; 32], &worker).unwrap();
    assert!(matches!(
        Party::import_material(&wrong, 1, context(), 0, nonce, packet, &mut guard),
        Err(Error::Context)
    ));
}

fn boolean_descriptor() -> crate::fusefss::Descriptor {
    let json = serde_json::json!({
        "schema":"pllm.fusefss.scalar-descriptor.v1", "profile":crate::fusefss::SYNTHETIC_PROFILE,
        "gate":"wire-test", "source_lock_sha256":"0".repeat(64),
        "input":{"domain":"A", "bits":5, "fractional":0, "signed":true},
        "output":{"arithmetic":"same-as-input", "boolean_arity":2},
        "external_encoding":"nearest-ties-even-input-and-weight", "internal_rounding":"signed-floor-only",
        "partition_order":"unsigned-canonical-half-open", "degree":1,
        "coefficient_order":["constant","linear"], "coefficient_fractional":[0,0],
        "selector_semantics":"descriptor-defined", "source_schedule":{}, "phi":"identity",
        "boundaries":[0,32], "pieces":[{"coefficients":[0,1], "booleans":[
            ["and",["lt",13],["not",["low_lt",2,1]]],
            ["or",["slt",-3],["msb",1]]
        ]}]
    });
    crate::fusefss::Descriptor::from_canonical_bytes(&serde_json::to_vec(&json).unwrap()).unwrap()
}

#[test]
fn predicates_roundtrip_complete_outputs() {
    use backend::predicates::{self as p, Step};
    let descriptor = boolean_descriptor();
    let dealer = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let worker = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut guard = ImportGuard::new(64, &worker).unwrap();
    for x in 0..32 {
        let [a, b] = p::issue(&descriptor, [9; 32], context(), &dealer).unwrap();
        let pa = a.export_material().unwrap();
        let pb = b.export_material().unwrap();
        let nonce = pa.nonce();
        let mut a = p::Party::import_material(
            &descriptor,
            [9; 32],
            context(),
            0,
            nonce,
            pa,
            &mut guard,
            &worker,
        )
        .unwrap();
        let mut b = p::Party::import_material(
            &descriptor,
            [9; 32],
            context(),
            1,
            nonce,
            pb,
            &mut guard,
            &worker,
        )
        .unwrap();
        let mut fa = a.start(0).unwrap();
        let mut fb = b.start(x).unwrap();
        let (oa, ob) = loop {
            match (
                a.advance(fb.as_bytes()).unwrap(),
                b.advance(fa.as_bytes()).unwrap(),
            ) {
                (Step::Frame(aa), Step::Frame(bb)) => {
                    fa = aa;
                    fb = bb;
                }
                (Step::Output(aa), Step::Output(bb)) => break (aa, bb),
                _ => panic!("different peer schedule"),
            }
        };
        assert_eq!(
            oa.shares()[0] ^ ob.shares()[0],
            u64::from(x < 13 && x % 4 >= 1)
        );
        let signed = if x < 16 { x as i64 } else { x as i64 - 32 };
        assert_eq!(
            oa.shares()[1] ^ ob.shares()[1],
            u64::from(signed < -3 || (x + 1) % 32 >= 16)
        );
    }
    assert_eq!(dealer.used_bytes(), 0);
    drop(guard);
    assert_eq!(worker.used_bytes(), 0);
}

#[test]
fn corruption_context_role_nonce_type_length_burn_before_allocation() {
    let dealer = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let worker = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let s = spec(c::Operation::ZeroExtend, 5, 9);
    for mutation in 0..12 {
        let mut guard = ImportGuard::new(2, &worker).unwrap();
        let [a, b] = c::issue(s, context(), &dealer).unwrap();
        drop(b);
        let packet = a.export_material().unwrap();
        let nonce = packet.nonce();
        let mut bytes = Zeroizing::new(packet.as_bytes().to_vec());
        let mut ctx = context();
        let mut role = 0;
        let mut expected_nonce = nonce;
        match mutation {
            0 => bytes[0] ^= 1,
            1 => bytes[8] ^= 1,
            2 => bytes[9] ^= 1,
            3 => bytes[10] ^= 1,
            4 => bytes[16] ^= 1,
            5 => bytes[48] ^= 1,
            6 => bytes[80] ^= 1,
            7 => bytes[88] ^= 1,
            8 => {
                bytes.pop();
            }
            9 => ctx.operation[0] ^= 1,
            10 => role = 1,
            _ => expected_nonce[0] ^= 1,
        }
        let received = MaterialPacket::from_bytes(&bytes, &worker).unwrap();
        let replay = MaterialPacket::from_bytes(&bytes, &worker).unwrap();
        worker
            .fail_after
            .store(0, std::sync::atomic::Ordering::SeqCst);
        assert!(matches!(
            c::Party::import_material(s, ctx, role, expected_nonce, received, &mut guard, &worker),
            Err(Error::Context)
        ));
        assert!(matches!(
            c::Party::import_material(s, ctx, role, expected_nonce, replay, &mut guard, &worker),
            Err(Error::Consumed)
        ));
        worker
            .fail_after
            .store(-1, std::sync::atomic::Ordering::SeqCst);
    }
    assert_eq!(worker.used_bytes(), 0);
    assert_eq!(dealer.used_bytes(), 0);
}

#[test]
fn partial_allocation_cancel_and_retained_packet_leases() {
    let dealer = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let worker = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let s = spec(
        c::Operation::Ars {
            shift: 2,
            range: c::SignedRange::full_ring(5).unwrap(),
        },
        5,
        5,
    );
    let [a, b] = c::issue(s, context(), &dealer).unwrap();
    drop(b);
    let packet = a.export_material().unwrap();
    assert_eq!(dealer.used_bytes(), packet.as_bytes().len());
    let nonce = packet.nonce();
    let mut successes = 0;
    for fail in 0..5 {
        let mut guard = ImportGuard::new(1, &worker).unwrap();
        let received = MaterialPacket::from_bytes(packet.as_bytes(), &worker).unwrap();
        let baseline = worker.used_bytes() - received.as_bytes().len();
        worker
            .fail_after
            .store(fail, std::sync::atomic::Ordering::SeqCst);
        let result =
            c::Party::import_material(s, context(), 0, nonce, received, &mut guard, &worker);
        worker
            .fail_after
            .store(-1, std::sync::atomic::Ordering::SeqCst);
        match result {
            Err(Error::Allocation) => {}
            Ok(mut party) => {
                successes += 1;
                party.cancel();
                assert!(party.is_consumed());
                assert!(matches!(party.export_material(), Err(Error::Consumed)));
            }
            _ => panic!("unexpected admission"),
        }
        assert_eq!(worker.used_bytes(), baseline);
        assert_eq!(dealer.used_bytes(), packet.as_bytes().len());
    }
    assert!(successes > 0);
    drop(packet);
    assert_eq!(dealer.used_bytes(), 0);
    assert_eq!(worker.used_bytes(), 0);
}

fn reseal(bytes: &mut [u8]) {
    let end = bytes.len() - 32;
    let digest = Sha256::digest(&bytes[..end]);
    bytes[end..].copy_from_slice(&digest);
}

#[test]
fn canonical_body_preflight_precedes_any_import_allocation() {
    let dealer = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let worker = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let s = spec(c::Operation::ZeroExtend, 5, 9);
    for offset in [88, 96, 120, 152] {
        let [a, b] = c::issue(s, context(), &dealer).unwrap();
        drop(b);
        let packet = a.export_material().unwrap();
        let nonce = packet.nonce();
        let mut bytes = Zeroizing::new(packet.as_bytes().to_vec());
        // Noncanonical input-mask/base/final-word/correction-word. Offsets are
        // fixed native schema, not struct memory or compiler-dependent padding.
        bytes[offset..offset + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        reseal(&mut bytes);
        let received = MaterialPacket::from_bytes(&bytes, &worker).unwrap();
        let mut guard = ImportGuard::new(1, &worker).unwrap();
        worker
            .fail_after
            .store(0, std::sync::atomic::Ordering::SeqCst);
        assert!(matches!(
            c::Party::import_material(s, context(), 0, nonce, received, &mut guard, &worker),
            Err(Error::Context)
        ));
        worker
            .fail_after
            .store(-1, std::sync::atomic::Ordering::SeqCst);
    }
    let [a, b] = c::issue(s, context(), &dealer).unwrap();
    drop(b);
    let packet = a.export_material().unwrap();
    let nonce = packet.nonce();
    let mut bytes = Zeroizing::new(packet.as_bytes().to_vec());
    // First vector correction controls: 88 header + 16 shares + 32 root/final
    // + 16 seed + 16 vector payload.
    bytes[168] = 2;
    reseal(&mut bytes);
    let received = MaterialPacket::from_bytes(&bytes, &worker).unwrap();
    let mut guard = ImportGuard::new(1, &worker).unwrap();
    worker
        .fail_after
        .store(0, std::sync::atomic::Ordering::SeqCst);
    assert!(matches!(
        c::Party::import_material(s, context(), 0, nonce, received, &mut guard, &worker),
        Err(Error::Context)
    ));
    worker
        .fail_after
        .store(-1, std::sync::atomic::Ordering::SeqCst);
}

#[test]
fn ring50_product_and_phi_wire_preserve_wrapping_outputs() {
    let dealer = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let worker = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut guard = ImportGuard::new(4, &worker).unwrap();
    let mask = (1u64 << 50) - 1;
    for operation in [
        c::Operation::Multiply,
        c::Operation::PhiSelect {
            add: 7,
            bit_contract: [7; 32],
        },
    ] {
        let s = spec(operation, 50, 50);
        let [a, b] = c::issue(s, context(), &dealer).unwrap();
        let pa = a.export_material().unwrap();
        let pb = b.export_material().unwrap();
        let nonce = pa.nonce();
        let mut a =
            c::Party::import_material(s, context(), 0, nonce, pa, &mut guard, &worker).unwrap();
        let mut b =
            c::Party::import_material(s, context(), 1, nonce, pb, &mut guard, &worker).unwrap();
        let (fa, fb, expected) = if operation == c::Operation::Multiply {
            (
                a.start_product(&[0], &[0]).unwrap(),
                b.start_product(&[mask], &[mask - 1]).unwrap(),
                2,
            )
        } else {
            (
                a.start_phi(&[0], &[0], &[0]).unwrap(),
                b.start_phi(&[1], &[mask], &[5]).unwrap(),
                6,
            )
        };
        let oa = a.finish(fb.as_bytes()).unwrap();
        let ob = b.finish(fa.as_bytes()).unwrap();
        assert_eq!((oa.shares()[0] + ob.shares()[0]) & mask, expected);
    }
}

fn import_failpoints<'a>(
    worker: &'a Budget,
    source: &MaterialPacket<'_>,
    mut import: impl FnMut(MaterialPacket<'a>, &mut ImportGuard<'a>) -> Result<(), Error>,
) {
    let baseline = worker.used_bytes();
    let mut completed = false;
    for fail in 0..64 {
        let mut guard = ImportGuard::new(1, worker).unwrap();
        let received = MaterialPacket::from_bytes(source.as_bytes(), worker).unwrap();
        worker
            .fail_after
            .store(fail, std::sync::atomic::Ordering::SeqCst);
        let result = import(received, &mut guard);
        worker
            .fail_after
            .store(-1, std::sync::atomic::Ordering::SeqCst);
        drop(guard);
        assert_eq!(worker.used_bytes(), baseline, "allocation index {fail}");
        match result {
            Err(Error::Allocation) => {}
            Ok(()) => {
                completed = true;
                break;
            }
            Err(other) => panic!("unexpected import error: {other:?}"),
        }
    }
    assert!(completed);
}

#[test]
fn every_owner_partial_import_failure_refunds_without_dropping_source_packet() {
    let dealer = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let worker = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let scalar = backend::Spec {
        bits: 5,
        source_intervals: 1,
        lanes: 1,
        descriptor_digest: [6; 32],
        source_digest: [7; 32],
        boolean_outputs: 0,
        identity_phi: true,
    };
    let [a, b] = backend::issue(scalar, context(), &dealer, |_, mask| {
        backend::DealerPartition::from_slices(mask, &[0, 16, 32], &[[7, 3]; 2])
    })
    .unwrap();
    drop(b);
    let packet = a.export_material().unwrap();
    let nonce = packet.nonce();
    import_failpoints(&worker, &packet, |received, guard| {
        backend::Party::import_material(scalar, context(), 0, nonce, received, guard, &worker)
            .map(drop)
    });
    drop(packet);
    {
        let mode = backend::LookupBackend::PublicBoundary;
        let source = backend::CompiledGelu::with_backend(&dealer, mode).unwrap();
        let public = backend::CompiledGelu::with_backend(&worker, mode).unwrap();
        let [a, b] = source.issue(1, context()).unwrap();
        drop(b);
        let packet = a.export_material().unwrap();
        let nonce = packet.nonce();
        import_failpoints(&worker, &packet, |received, guard| {
            public
                .import_material(1, context(), 0, nonce, received, guard)
                .map(drop)
        });
    }
    let descriptor = boolean_descriptor();
    let [a, b] = backend::predicates::issue(&descriptor, [9; 32], context(), &dealer).unwrap();
    drop(b);
    let packet = a.export_material().unwrap();
    let nonce = packet.nonce();
    import_failpoints(&worker, &packet, |received, guard| {
        backend::predicates::Party::import_material(
            &descriptor,
            [9; 32],
            context(),
            0,
            nonce,
            received,
            guard,
            &worker,
        )
        .map(drop)
    });
    drop(packet);
    let table = backend::lookup::Table::new_zero_extended(9, &[1; 257], [7; 32], &dealer).unwrap();
    let public = backend::lookup::Table::new_zero_extended(9, &[1; 257], [7; 32], &worker).unwrap();
    let [a, b] = table.issue(1, context()).unwrap();
    drop(b);
    let packet = a.export_material().unwrap();
    let nonce = packet.nonce();
    import_failpoints(&worker, &packet, |received, guard| {
        backend::lookup::Party::import_material(&public, 1, context(), 0, nonce, received, guard)
            .map(drop)
    });
    drop(packet);
    drop(public);
    drop(table);
    assert_eq!(dealer.used_bytes(), 0);
    assert_eq!(worker.used_bytes(), 0);
}

#[test]
fn export_only_fresh_and_failures_keep_independent_frame_leases() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let s = spec(c::Operation::ZeroExtend, 5, 9);
    let [mut a, b] = c::issue(s, context(), &budget).unwrap();
    drop(b);
    let frame = a.start(&[3]).unwrap();
    assert!(matches!(a.export_material(), Err(Error::Context)));
    assert_eq!(budget.used_bytes(), frame.as_bytes().len());
    drop(frame);
    assert_eq!(budget.used_bytes(), 0);
    let [a, b] = c::issue(s, context(), &budget).unwrap();
    drop(b);
    budget
        .fail_after
        .store(0, std::sync::atomic::Ordering::SeqCst);
    assert!(matches!(a.export_material(), Err(Error::Allocation)));
    budget
        .fail_after
        .store(-1, std::sync::atomic::Ordering::SeqCst);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn legacy_guard_layout_preserves_capacity_pricing_and_error_precedence() {
    use std::sync::atomic::Ordering;
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    budget.fail_after.store(0, Ordering::SeqCst);
    for invalid in [0, 4097, usize::MAX] {
        assert_eq!(ImportGuardLayout::legacy(invalid), Err(Error::Resource));
        assert!(matches!(
            ImportGuard::new(invalid, &budget),
            Err(Error::Resource)
        ));
    }
    budget.fail_after.store(-1, Ordering::SeqCst);
    let guard = ImportGuard::new(4096, &budget).unwrap();
    assert_eq!(budget.used_bytes(), 4096 * 33);
    drop(guard);
    let mut guard = ImportGuard::new(2, &budget).unwrap();
    assert_eq!(budget.used_bytes(), 66);
    budget.fail_after.store(0, Ordering::SeqCst);
    assert_eq!(guard.claim(2, [0; 32]), Err(Error::Domain));
    assert_eq!(guard.claim(0, [0; 32]), Ok(()));
    assert_eq!(guard.claim(0, [0; 32]), Err(Error::Consumed));
    assert_eq!(guard.claim(1, [0; 32]), Ok(()));
    assert_eq!(guard.claim(0, [1; 32]), Err(Error::Resource));
    assert_eq!(guard.claim(1, [0; 32]), Err(Error::Consumed));
    drop(guard);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn request_guard_layout_is_checked_exact_and_allocation_failure_refunds() {
    use std::{mem::size_of, sync::atomic::Ordering};
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    budget.fail_after.store(0, Ordering::SeqCst);
    for invalid in [0, MAX_ALLOCATION_BYTES, usize::MAX / 2, usize::MAX] {
        assert_eq!(
            ImportGuardLayout::for_requests(invalid),
            Err(Error::Resource)
        );
        assert_eq!(
            ImportGuard::indexed_reservation_bytes(invalid),
            Err(Error::Resource)
        );
        assert!(matches!(
            ImportGuard::new_indexed(invalid, &budget),
            Err(Error::Resource)
        ));
    }
    let available_slots = (MAX_ALLOCATION_BYTES - size_of::<ImportGuard<'_>>()) / 33;
    let max_slots = 1usize << (usize::BITS - 1 - available_slots.leading_zeros());
    let max_layout = ImportGuardLayout::for_requests(max_slots / 2).unwrap();
    assert!(max_layout.reservation_bytes() <= MAX_ALLOCATION_BYTES);
    assert_eq!(
        ImportGuardLayout::for_requests(max_slots / 2 + 1),
        Err(Error::Resource)
    );
    let layout = ImportGuardLayout::for_requests(49_933).unwrap();
    assert_eq!(layout.capacity(), 49_933);
    assert_eq!(
        ImportGuard::indexed_reservation_bytes(49_933),
        Ok(layout.reservation_bytes())
    );
    assert!(matches!(
        ImportGuard::with_layout(layout, &budget),
        Err(Error::Allocation)
    ));
    assert_eq!(budget.used_bytes(), 0);
    let short = Budget::new(layout.reservation_bytes() - 1).unwrap();
    short.fail_after.store(0, Ordering::SeqCst);
    assert!(matches!(
        ImportGuard::with_layout(layout, &short),
        Err(Error::Resource)
    ));
    assert_eq!(short.used_bytes(), 0);

    // A failed guard allocation must not refund or erase a separately retained
    // packet owner. Successful construction uses exactly one fallible allocation.
    budget.fail_after.store(-1, Ordering::SeqCst);
    let packet = MaterialPacket::from_bytes(&[0; ENVELOPE_BYTES], &budget).unwrap();
    let baseline = budget.used_bytes();
    budget.fail_after.store(0, Ordering::SeqCst);
    assert!(matches!(
        ImportGuard::with_layout(layout, &budget),
        Err(Error::Allocation)
    ));
    assert_eq!(budget.used_bytes(), baseline);
    budget.fail_after.store(1, Ordering::SeqCst);
    let mut guard = ImportGuard::with_layout(layout, &budget).unwrap();
    assert_eq!(budget.fail_after.load(Ordering::SeqCst), 0);
    assert_eq!(guard.claims.capacity(), layout.slots);
    assert_eq!(guard.claims.len(), layout.slots);
    assert_eq!(
        layout.reservation_bytes(),
        guard.claims.capacity() * 33 + size_of::<ImportGuard<'_>>()
    );
    assert_eq!(budget.used_bytes(), baseline + layout.reservation_bytes());
    guard.claim(0, [255; 32]).unwrap();
    drop(guard); // Zeroizing backing drops before the guard's final Lease field.
    assert_eq!(budget.used_bytes(), baseline);
    assert_eq!(packet.as_bytes(), &[0; ENVELOPE_BYTES]);
    drop(packet);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn request_guard_keeps_49933_exact_claims_without_per_claim_allocation() {
    use std::sync::atomic::Ordering;
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let layout = ImportGuardLayout::for_requests(49_933).unwrap();
    let mut guard = ImportGuard::new_indexed(layout.capacity(), &budget).unwrap();
    let backing = guard.claims.as_ptr();
    budget.fail_after.store(0, Ordering::SeqCst);
    let key = |i: usize| {
        let mut nonce = [0; 32];
        // Adjacent claims share a nonce but have different roles. Include the
        // all-zero nonce and exercise every nonce byte, not only a prefix.
        let value = (i / 2) as u64;
        for part in nonce.chunks_exact_mut(8) {
            part.copy_from_slice(&value.to_le_bytes());
        }
        ((i % 2) as u8, nonce)
    };
    for i in 0..layout.capacity() {
        let (role, nonce) = key(i);
        guard.claim(role, nonce).unwrap();
    }
    assert_eq!(guard.used, layout.capacity());
    for i in (0..layout.capacity()).rev() {
        let (role, nonce) = key(i);
        assert_eq!(guard.claim(role, nonce), Err(Error::Consumed));
    }
    let (role, nonce) = key(layout.capacity());
    assert_eq!(guard.claim(role, nonce), Err(Error::Resource));
    assert_eq!(guard.claim(2, nonce), Err(Error::Domain));
    assert_eq!(guard.used, layout.capacity());
    assert_eq!(guard.claims.as_ptr(), backing);
    assert_eq!(budget.used_bytes(), layout.reservation_bytes());
    assert_eq!(budget.fail_after.load(Ordering::SeqCst), 0);
    drop(guard);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn request_guard_hash_collisions_compare_complete_keys_and_never_evict() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let layout = ImportGuardLayout::for_requests(8).unwrap();
    let mut guard = ImportGuard::with_layout(layout, &budget).unwrap();
    let mut nonces = Vec::new();
    // Force the same initial bucket; collisions must never substitute hashes
    // for exact membership, even at exhaustion. The last key stays unclaimed.
    for i in 0u64..10_000 {
        let mut claim = [0; 33];
        claim[24..32].copy_from_slice(&i.to_le_bytes());
        claim[32] = 1;
        let digest = Sha256::digest(claim);
        if u64::from_le_bytes(digest[..8].try_into().unwrap()) as usize & (layout.slots - 1) == 0 {
            nonces.push(claim[..32].try_into().unwrap());
        }
        if nonces.len() == layout.capacity() + 1 {
            break;
        }
    }
    assert_eq!(nonces.len(), 9);
    for &nonce in &nonces[..8] {
        guard.claim(0, nonce).unwrap();
    }
    assert_eq!(guard.claim(0, nonces[8]), Err(Error::Resource));
    for &nonce in &nonces[..8] {
        assert_eq!(guard.claim(0, nonce), Err(Error::Consumed));
    }
}

#[test]
fn request_guard_malformed_and_partial_imports_burn_before_allocation() {
    use std::sync::atomic::Ordering;
    let dealer = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let worker = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let layout = ImportGuardLayout::for_requests(5000).unwrap();
    let s = spec(c::Operation::ZeroExtend, 5, 9);
    let [a, b] = c::issue(s, context(), &dealer).unwrap();
    drop(b);
    let packet = a.export_material().unwrap();
    let nonce = packet.nonce();
    let mut bytes = Zeroizing::new(packet.as_bytes().to_vec());
    bytes[88..96].copy_from_slice(&u64::MAX.to_le_bytes());
    reseal(&mut bytes);
    let malformed = MaterialPacket::from_bytes(&bytes, &worker).unwrap();
    let mut guard = ImportGuard::with_layout(layout, &worker).unwrap();
    worker.fail_after.store(0, Ordering::SeqCst);
    assert!(matches!(
        c::Party::import_material(s, context(), 0, nonce, malformed, &mut guard, &worker),
        Err(Error::Context)
    ));
    assert_eq!(guard.claim(0, nonce), Err(Error::Consumed));
    assert_eq!(worker.used_bytes(), layout.reservation_bytes());
    drop(guard);
    assert_eq!(worker.used_bytes(), 0);
    worker.fail_after.store(-1, Ordering::SeqCst);

    let mut completed = false;
    for fail in 0..16 {
        // Independent receiving sessions exercise each allocator failpoint;
        // within each session the failed import remains permanently burned.
        let mut guard = ImportGuard::with_layout(layout, &worker).unwrap();
        let received = MaterialPacket::from_bytes(packet.as_bytes(), &worker).unwrap();
        worker.fail_after.store(fail, Ordering::SeqCst);
        let result =
            c::Party::import_material(s, context(), 0, nonce, received, &mut guard, &worker);
        worker.fail_after.store(-1, Ordering::SeqCst);
        match result {
            Ok(mut party) => {
                party.cancel();
                assert!(party.is_consumed());
                assert!(matches!(party.export_material(), Err(Error::Consumed)));
                completed = true;
            }
            Err(Error::Allocation) => {}
            Err(other) => panic!("unexpected failpoint {fail}: {other:?}"),
        }
        let replay = MaterialPacket::from_bytes(packet.as_bytes(), &worker).unwrap();
        assert!(matches!(
            c::Party::import_material(s, context(), 0, nonce, replay, &mut guard, &worker),
            Err(Error::Consumed)
        ));
        assert_eq!(worker.used_bytes(), layout.reservation_bytes());
        drop(guard);
        assert_eq!(worker.used_bytes(), 0);
        assert_eq!(dealer.used_bytes(), packet.as_bytes().len());
        if completed {
            break;
        }
    }
    assert!(completed);
    drop(packet);
    assert_eq!(dealer.used_bytes(), 0);
}

#[test]
fn file_guard_qwen_geometry_limits_and_resident_admission() {
    use std::sync::atomic::Ordering;
    let qwen = ImportGuard::indexed_file_resources(3_530_000).unwrap();
    assert_eq!(qwen.disk_bytes, 8_388_608 * 33);
    assert_eq!(qwen.resident_bytes, std::mem::size_of::<ImportGuard<'_>>());
    assert!(qwen.resident_bytes < 1024);
    assert_eq!(
        ImportGuard::indexed_reservation_bytes(3_530_000),
        Err(Error::Resource)
    );
    for capacity in [0, usize::MAX, usize::MAX / 2] {
        assert_eq!(
            ImportGuard::indexed_file_resources(capacity),
            Err(Error::Resource)
        );
    }
    let max_slots = 1u64 << (63 - (MAX_INDEXED_FILE_DISK_BYTES / 33).leading_zeros());
    assert!(ImportGuard::indexed_file_resources((max_slots / 2) as usize).is_ok());
    assert_eq!(
        ImportGuard::indexed_file_resources((max_slots / 2 + 1) as usize),
        Err(Error::Resource)
    );

    // Only small sparse fixtures are created; Qwen geometry above is pure pricing.
    let r = ImportGuard::indexed_file_resources(17).unwrap();
    assert_eq!(r.resident_bytes, qwen.resident_bytes);
    let short = Budget::new(r.resident_bytes - 1).unwrap();
    assert!(matches!(
        ImportGuard::new_indexed_file(17, r.disk_bytes, &short),
        Err(Error::Resource)
    ));
    assert_eq!(short.used_bytes(), 0);
    let budget = Budget::new(r.resident_bytes).unwrap();
    budget.fail_after.store(0, Ordering::SeqCst);
    for disk_limit in [
        0,
        r.disk_bytes - 1,
        MAX_INDEXED_FILE_DISK_BYTES + 1,
        u64::MAX,
    ] {
        assert!(matches!(
            ImportGuard::new_indexed_file(17, disk_limit, &budget),
            Err(Error::Resource)
        ));
        assert_eq!(budget.used_bytes(), 0);
    }
    assert!(matches!(
        ImportGuard::new_indexed_file(17, r.disk_bytes, &budget),
        Err(Error::Allocation)
    ));
    assert_eq!(budget.used_bytes(), 0);
    budget.fail_after.store(1, Ordering::SeqCst);
    let mut guard = ImportGuard::new_indexed_file(17, r.disk_bytes, &budget).unwrap();
    assert_eq!(budget.fail_after.load(Ordering::SeqCst), 0);
    guard.claim(0, [0; 32]).unwrap();
    assert_eq!(guard.claims.capacity(), 0);
    assert!(guard.file.as_ref().unwrap().scratch_is_zero());
    assert_eq!(budget.used_bytes(), r.resident_bytes);
    drop(guard);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn file_and_heap_guard_decisions_match_for_randomized_colliding_claims() {
    use std::sync::atomic::Ordering;
    let capacity = 257;
    let resources = ImportGuard::indexed_file_resources(capacity).unwrap();
    let file_budget = Budget::new(resources.resident_bytes).unwrap();
    let heap_budget =
        Budget::new(ImportGuard::indexed_reservation_bytes(capacity).unwrap()).unwrap();
    let mut file =
        ImportGuard::new_indexed_file(capacity, resources.disk_bytes, &file_budget).unwrap();
    let mut heap = ImportGuard::new_indexed(capacity, &heap_budget).unwrap();
    file_budget.fail_after.store(0, Ordering::SeqCst);
    heap_budget.fail_after.store(0, Ordering::SeqCst);
    let mut keys = Vec::new();
    keys.push((0, [0; 32]));
    keys.push((1, [0; 32]));
    // Deliberately collide initial buckets with SHA-derived, full-width nonces.
    let slots = indexed_slots(capacity).unwrap();
    for i in 0u64..100_000 {
        let nonce: [u8; 32] = Sha256::digest(i.to_le_bytes()).into();
        let mut record = [0; 33];
        record[..32].copy_from_slice(&nonce);
        record[32] = 1;
        if indexed_probes(&record, slots).0 == 0 {
            keys.push((0, nonce));
        }
        if keys.len() == 18 {
            break;
        }
    }
    assert_eq!(keys.len(), 18);
    while keys.len() < capacity {
        let mut hash = Sha256::new();
        hash.update(b"file-ledger-randomized-fixture");
        hash.update((keys.len() as u64).to_le_bytes());
        keys.push(((keys.len() % 2) as u8, hash.finalize().into()));
    }
    for &(role, nonce) in &keys {
        let expected = heap.claim(role, nonce);
        assert_eq!(expected, Ok(()));
        assert_eq!(file.claim(role, nonce), expected);
    }
    for &(role, nonce) in keys.iter().rev() {
        assert_eq!(heap.claim(role, nonce), Err(Error::Consumed));
        assert_eq!(file.claim(role, nonce), Err(Error::Consumed));
    }
    for (role, nonce) in [(0, [255; 32]), (2, [0; 32]), (0, [0; 32])] {
        assert_eq!(file.claim(role, nonce), heap.claim(role, nonce));
    }
    assert_eq!(file.used, heap.used);
    assert_eq!(file.used, capacity);
    assert!(file.file.as_ref().unwrap().scratch_is_zero());
    assert_eq!(file_budget.used_bytes(), resources.resident_bytes);
    drop(file);
    drop(heap);
    assert_eq!(file_budget.used_bytes(), 0);
    assert_eq!(heap_budget.used_bytes(), 0);
}

#[test]
fn file_guard_io_errors_and_truncation_permanently_poison_and_close() {
    let resources = ImportGuard::indexed_file_resources(8).unwrap();
    let budget = Budget::new(resources.resident_bytes).unwrap();
    for case in 0..5 {
        let mut guard = ImportGuard::new_indexed_file(8, resources.disk_bytes, &budget).unwrap();
        guard.claim(0, [7; 32]).unwrap();
        let ledger = guard.file.as_mut().unwrap();
        match case {
            0 => ledger.fault = Some(file::Fault::Read),
            1 => ledger.fault = Some(file::Fault::Write),
            2 => ledger.fault = Some(file::Fault::PartialWrite),
            3 => ledger.file.as_ref().unwrap().set_len(0).unwrap(),
            _ => ledger
                .file
                .as_ref()
                .unwrap()
                .set_len(resources.disk_bytes - 1)
                .unwrap(),
        }
        assert_eq!(guard.claim(1, [19; 32]), Err(Error::Resource));
        let ledger = guard.file.as_ref().unwrap();
        assert!(
            ledger.file.is_none(),
            "case {case}: poisoned file must close immediately"
        );
        assert!(ledger.scratch_is_zero());
        assert_eq!(guard.used, 1);
        assert_eq!(guard.claim(0, [7; 32]), Err(Error::Consumed));
        assert_eq!(guard.claim(1, [20; 32]), Err(Error::Consumed));
        assert_eq!(budget.used_bytes(), resources.resident_bytes);
        drop(guard);
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn file_guard_two_live_owners_are_private_and_independent() {
    use std::io::{Seek, SeekFrom};
    let r = ImportGuard::indexed_file_resources(4).unwrap();
    let budget = Budget::new(2 * r.resident_bytes + ENVELOPE_BYTES).unwrap();
    let mut left = ImportGuard::new_indexed_file(4, r.disk_bytes, &budget).unwrap();
    let mut right = ImportGuard::new_indexed_file(4, r.disk_bytes, &budget).unwrap();
    let retained = MaterialPacket::from_bytes(&[0; ENVELOPE_BYTES], &budget).unwrap();
    assert_eq!(budget.used_bytes(), 2 * r.resident_bytes + ENVELOPE_BYTES);
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        for guard in [&left, &right] {
            let metadata = guard
                .file
                .as_ref()
                .unwrap()
                .file
                .as_ref()
                .unwrap()
                .metadata()
                .unwrap();
            assert_eq!(metadata.nlink(), 0);
            assert_eq!(metadata.mode() & 0o077, 0);
        }
    }
    left.file
        .as_mut()
        .unwrap()
        .file
        .as_mut()
        .unwrap()
        .seek(SeekFrom::Start(5))
        .unwrap();
    left.claim(0, [4; 32]).unwrap();
    right.claim(0, [4; 32]).unwrap();
    assert_eq!(
        left.file
            .as_mut()
            .unwrap()
            .file
            .as_mut()
            .unwrap()
            .stream_position()
            .unwrap(),
        5
    );
    drop(left);
    assert_eq!(budget.used_bytes(), r.resident_bytes + ENVELOPE_BYTES);
    assert_eq!(right.claim(0, [4; 32]), Err(Error::Consumed));
    drop(right);
    assert_eq!(budget.used_bytes(), ENVELOPE_BYTES);
    drop(retained);
    assert_eq!(budget.used_bytes(), 0);
}
