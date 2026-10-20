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
