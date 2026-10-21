use super::*;
use pllm_core::edge_contract::{EdgeIdentity, Shape};
use std::sync::atomic::Ordering;

fn source() -> SourceIdentity {
    SourceIdentity::new([11; 32], [12; 32], approved_implementation())
}
fn dealer(budget: &Budget) -> LocalDealer<'_> {
    LocalDealer::new(source(), [1; 32], [2; 32], budget).unwrap()
}
fn edge(id: u8, bits: u8, f: u8, private: bool) -> EdgeDescriptor {
    EdgeDescriptor::new(
        EdgeIdentity::new(source(), [1; 32], [id + 64; 32], [id; 32], 0),
        Shape::new(&[1, 1]).unwrap(),
        Numeric::new(bits, f, Interpretation::Signed).unwrap(),
        Representation::CommonAdditiveMasked,
        private.then_some(WeightVisibility::Private),
    )
    .unwrap()
}
fn spec() -> matrix::Spec {
    matrix::Spec {
        bits: 16,
        rows: 1,
        inner: 1,
        columns: 1,
        layout: matrix::MatrixLayout::InputOutput,
    }
}
type Fixture<'b> = (
    DealerMask<'b, Ring>,
    [MatrixParty<'b>; 2],
    [CommonMasked<'b, Ring>; 2],
    [CommonMasked<'b, Ring>; 2],
);
fn issue<'b>(d: &mut LocalDealer<'b>, range: MatrixRange) -> Result<Fixture<'b>, Error> {
    let (xm, x) = d
        .prepare_root(edge(3, 16, 2, false), BoundRange::new(-4, 5)?, [5; 32])?
        .encode(&[-3])?;
    let (wm, w) = d
        .prepare_root(edge(4, 16, 2, true), BoundRange::new(-4, 5)?, [5; 32])?
        .encode(&[2])?;
    let (mask, parties) = d.matrix(
        [5; 32],
        spec(),
        xm,
        wm,
        edge(6, 16, 4, false),
        [7; 32],
        range,
    )?;
    Ok((mask, parties, x, w))
}
fn run<'b, K>(
    mut parties: [ArithmeticParty<'b, K>; 2],
    [a, b]: [CommonMasked<'b, Ring>; 2],
) -> [CommonMasked<'b, K>; 2] {
    let [left, right] = &mut parties;
    let mut l = left.begin(a).unwrap();
    let mut r = right.begin(b).unwrap();
    loop {
        match (
            left.advance(r.as_bytes()).unwrap(),
            right.advance(l.as_bytes()).unwrap(),
        ) {
            (Step::Frame(a), Step::Frame(b)) => {
                l = a;
                r = b;
            }
            (Step::Output(a), Step::Output(b)) => return [a, b],
            _ => panic!("phase mismatch"),
        }
    }
}

#[test]
fn handoff_original_inputs_then_tr_or_qualified_gap_then_drelu() {
    let _guard = matrix::TEST_SERIAL.lock().unwrap();
    for gap in [false, true] {
        let budget = Budget::new(64 << 20).unwrap();
        let mut d = dealer(&budget);
        let (ym, [a, b], [xa, xb], [wa, wb]) =
            issue(&mut d, MatrixRange::RequireSignedNoWrap).unwrap();
        let mut restored = [a, b].map(|p| {
            let original_wrapper_charge = p.lease.as_ref().unwrap().bytes;
            let handoff = p.export_material().unwrap();
            let descriptor = handoff.descriptor();
            assert_eq!(descriptor.implementation(), approved_implementation());
            assert_eq!(descriptor.operation(), [5; 32]);
            assert_eq!(descriptor.spec(), spec());
            assert_eq!(
                descriptor.provenance(),
                MatrixMaterialProvenance::LocalAssumption
            );
            let (permit, packet) = handoff.into_parts();
            assert_eq!(permit.lease.bytes, original_wrapper_charge);
            assert_eq!(
                packet.as_bytes().len(),
                matrix_material_resources(spec()).unwrap().packet_bytes
            );
            let restored = permit.import(packet).unwrap();
            assert_eq!(
                restored.lease.as_ref().unwrap().bytes,
                original_wrapper_charge
            );
            restored
        });
        let [a, b] = &mut restored;
        let l = a.begin(xa, wa, &mut || false).unwrap();
        let r = b.begin(xb, wb, &mut || false).unwrap();
        let ya = a.finish(r.as_bytes()).unwrap();
        let yb = b.finish(l.as_bytes()).unwrap();
        assert!(matches!(ya.backing, CommonBacking::Matrix(_)));
        assert_eq!(ya.binding, ym.binding);
        assert_eq!(
            ya.backing.values()[0].wrapping_sub(ym.backing.values()[0]) & 65535,
            65530
        );
        drop((restored, l, r));
        let (mask, parties) = if gap {
            d.gap_ars([7; 32], ym, 2, edge(8, 16, 2, false), [9; 32])
                .unwrap()
        } else {
            d.tr([7; 32], ym, 2, edge(8, 14, 2, false), [9; 32])
                .unwrap()
        };
        let caps = run(parties, [ya, yb]);
        let bits = mask.binding.descriptor.numeric().ring_bits();
        assert_eq!(
            crate::masked_wire_reference::signed(
                caps[0].backing.values()[0].wrapping_sub(mask.backing.values()[0])
                    & crate::masked_wire_reference::mask(bits),
                bits
            ),
            -2
        );
        let predicate = EdgeDescriptor::new(
            EdgeIdentity::new(source(), [1; 32], [74; 32], [10; 32], 0),
            Shape::new(&[1, 1]).unwrap(),
            Numeric::new(1, 0, Interpretation::Predicate).unwrap(),
            Representation::CommonXorMasked,
            None,
        )
        .unwrap();
        let (mask, parties) = d.drelu([9; 32], mask, predicate, [11; 32]).unwrap();
        let caps = run(parties, caps);
        assert_eq!(caps[0].backing.values()[0] ^ mask.backing.values()[0], 0);
        drop((mask, caps, d));
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn descriptor_commits_full_bindings_source_spec_and_provenance() {
    let _guard = matrix::TEST_SERIAL.lock().unwrap();
    let budget = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let (mask, [a, b], x, w) = issue(&mut d, MatrixRange::RequireSignedNoWrap).unwrap();
    let original_inputs = a.inputs;
    let h = a.export_material().unwrap();
    let base = h.descriptor();
    assert_eq!(
        base.input_binding_digests(),
        original_inputs.map(Binding::digest)
    );
    assert_eq!(base.output_binding_digest(), mask.binding.digest());
    let hash = base.digest();
    for index in 0..3 {
        let original = if index == 2 {
            mask.binding
        } else {
            original_inputs[index]
        };
        for field in 0..9 {
            let mut binding = original;
            match field {
                0 => binding.session[0] ^= 1,
                1 => binding.namespace[0] ^= 1,
                2 => binding.issuance += 1,
                3 => binding.mask_id[0] ^= 1,
                4 => binding.consumer[0] ^= 1,
                5 => binding.interval.as_mut().unwrap().lower -= 1,
                6..=8 => {
                    let e = binding.descriptor;
                    let i = e.identity();
                    let s = i.source();
                    let mut checkpoint = s.checkpoint();
                    let mut numeric = s.numeric_profile();
                    let mut implementation = s.implementation();
                    match field {
                        6 => checkpoint[0] ^= 1,
                        7 => numeric[0] ^= 1,
                        _ => implementation[0] ^= 1,
                    }
                    binding.descriptor = EdgeDescriptor::new(
                        EdgeIdentity::new(
                            SourceIdentity::new(checkpoint, numeric, implementation),
                            i.plan(),
                            i.producer(),
                            i.wire(),
                            i.first_tensor_index(),
                        ),
                        e.shape(),
                        e.numeric(),
                        e.representation(),
                        e.weight_visibility(),
                    )
                    .unwrap();
                }
                _ => unreachable!(),
            }
            assert_ne!(original.digest(), binding.digest());
            let mut changed = base;
            if index == 2 {
                changed.output = binding.digest();
            } else {
                changed.inputs[index] = binding.digest();
            }
            assert_ne!(hash, changed.digest());
        }
    }
    for field in 0..10 {
        let mut changed = base;
        match field {
            0 => changed.implementation[0] ^= 1,
            1 => changed.operation[0] ^= 1,
            2 => changed.role ^= 1,
            3 => changed.spec.bits -= 1,
            4 => changed.spec.layout = matrix::MatrixLayout::OutputInput,
            5 => changed.spec.rows += 1,
            6 => changed.spec.inner += 1,
            7 => changed.spec.columns += 1,
            8 => changed.native_binding[0] ^= 1,
            9 => changed.provenance = MatrixMaterialProvenance::RingOnly,
            _ => unreachable!(),
        }
        assert_ne!(hash, changed.digest());
    }
    drop((h, b, mask, x, w, d));
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn ring_only_roundtrip_preserves_missing_interval_and_rejects_gap_ars() {
    let _guard = matrix::TEST_SERIAL.lock().unwrap();
    let budget = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let (mask, parties, [xa, xb], [wa, wb]) = issue(&mut d, MatrixRange::RingOnly).unwrap();
    assert_eq!(mask.binding.interval, None);
    let mut parties = parties.map(|party| {
        let (permit, packet) = party.export_material().unwrap().into_parts();
        permit.import(packet).unwrap()
    });
    let [left, right] = &mut parties;
    let l = left.begin(xa, wa, &mut || false).unwrap();
    let r = right.begin(xb, wb, &mut || false).unwrap();
    let outputs = [
        left.finish(r.as_bytes()).unwrap(),
        right.finish(l.as_bytes()).unwrap(),
    ];
    for output in &outputs {
        assert_eq!(output.binding, mask.binding);
        assert_eq!(output.binding.interval, None);
        assert_eq!(
            output.backing.values()[0].wrapping_sub(mask.backing.values()[0]) & 65535,
            65530
        );
    }
    assert_eq!(
        d.gap_ars([7; 32], mask, 2, edge(8, 16, 2, false), [9; 32])
            .err(),
        Some(Error::Domain)
    );
    drop((parties, l, r, outputs, d));
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn sibling_packet_wrong_operation_and_role_rejected_by_native_authority() {
    let _guard = matrix::TEST_SERIAL.lock().unwrap();
    let budget = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let (mask, [a, mut b], x, w) = issue(&mut d, MatrixRange::RingOnly).unwrap();
    let a = a.export_material().unwrap();
    b.operation[0] ^= 1;
    let b = b.export_material().unwrap();
    assert_eq!(
        a.descriptor().provenance(),
        MatrixMaterialProvenance::RingOnly
    );
    assert_ne!(a.descriptor().digest(), b.descriptor().digest());
    let (pa, ba) = a.into_parts();
    let (pb, bb) = b.into_parts();
    assert_eq!(pa.import(bb).err(), Some(Error::Context));
    assert_eq!(pb.import(ba).err(), Some(Error::Context));
    drop((mask, x, w, d));
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn same_shape_reissuance_has_distinct_complete_binding_and_native_identity() {
    let _guard = matrix::TEST_SERIAL.lock().unwrap();
    let budget = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let mut previous: Option<MatrixMaterialDescriptor> = None;
    for _ in 0..2 {
        let (mask, [a, b], x, w) = issue(&mut d, MatrixRange::RingOnly).unwrap();
        let handoff = a.export_material().unwrap();
        let current = handoff.descriptor();
        if let Some(previous) = previous {
            assert_eq!(previous.spec(), current.spec());
            assert_eq!(previous.role(), current.role());
            assert_eq!(previous.operation(), current.operation());
            assert_ne!(
                previous.input_binding_digests(),
                current.input_binding_digests()
            );
            assert_ne!(
                previous.output_binding_digest(),
                current.output_binding_digest()
            );
            assert_ne!(
                previous.native_execution_binding(),
                current.native_execution_binding()
            );
            assert_ne!(previous.digest(), current.digest());
        }
        previous = Some(current);
        // A second issuance cannot coexist with either retained packet or permit
        // (tested separately). Thus no public cross-issuance packet pair exists.
        drop((handoff, b, mask, x, w));
    }
    drop(d);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn imported_or_started_finished_cancelled_owner_cannot_export() {
    let _guard = matrix::TEST_SERIAL.lock().unwrap();
    for phase in 0..4 {
        let budget = Budget::new(64 << 20).unwrap();
        let mut d = dealer(&budget);
        let (mask, [mut a, mut b], [xa, xb], [wa, wb]) =
            issue(&mut d, MatrixRange::RequireSignedNoWrap).unwrap();
        match phase {
            0 => {
                let (permit, packet) = a.export_material().unwrap().into_parts();
                let a = permit.import(packet).unwrap();
                assert_eq!(a.export_material().err(), Some(Error::Consumed));
                drop((xa, xb, wa, wb));
            }
            1 => {
                let frame = a.begin(xa, wa, &mut || false).unwrap();
                assert_eq!(a.export_material().err(), Some(Error::Consumed));
                drop((frame, xb, wb));
            }
            2 => {
                let l = a.begin(xa, wa, &mut || false).unwrap();
                let r = b.begin(xb, wb, &mut || false).unwrap();
                let out = a.finish(r.as_bytes()).unwrap();
                assert_eq!(a.export_material().err(), Some(Error::Consumed));
                drop((l, r, out));
            }
            _ => {
                a.cancel();
                assert_eq!(a.export_material().err(), Some(Error::Consumed));
                drop((xa, xb, wa, wb));
            }
        }
        b.cancel();
        drop((mask, b, d));
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn exact_incremental_admission_and_one_below_before_allocation() {
    let _guard = matrix::TEST_SERIAL.lock().unwrap();
    for below in [false, true] {
        let budget = Budget::new(64 << 20).unwrap();
        let mut d = dealer(&budget);
        let (mask, [a, b], x, w) = issue(&mut d, MatrixRange::RequireSignedNoWrap).unwrap();
        let cost = matrix_material_resources(spec()).unwrap();
        assert_eq!(
            cost.incremental_reservation_bytes,
            cost.native_incremental_reservation_bytes + cost.typed_incremental_reservation_bytes
        );
        let blocker = budget
            .reserve(
                (64 << 20) - budget.used_bytes() - cost.incremental_reservation_bytes
                    + usize::from(below),
            )
            .unwrap();
        if below {
            budget.fail_after.store(0, Ordering::SeqCst);
        }
        let result = a.export_material();
        if below {
            assert_eq!(result.err(), Some(Error::Resource));
        } else {
            let (permit, packet) = result.unwrap().into_parts();
            // Import performs no new reservation: existing native and typed
            // leases cover recreation while the packet is still charged.
            assert_eq!(budget.used_bytes(), 64 << 20);
            let restored = permit.import(packet).unwrap();
            drop(restored);
        }
        budget.fail_after.store(-1, Ordering::SeqCst);
        drop((blocker, mask, b, x, w, d));
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn export_and_each_import_allocation_failure_burns_owners_without_leaks() {
    let _guard = matrix::TEST_SERIAL.lock().unwrap();
    for failure in 0..4 {
        let budget = Budget::new(64 << 20).unwrap();
        let mut d = dealer(&budget);
        let (mask, [a, b], x, w) = issue(&mut d, MatrixRange::RequireSignedNoWrap).unwrap();
        if failure == 0 {
            budget.fail_after.store(0, Ordering::SeqCst);
            assert_eq!(a.export_material().err(), Some(Error::Allocation));
        } else {
            let (permit, packet) = a.export_material().unwrap().into_parts();
            budget.fail_after.store(failure - 1, Ordering::SeqCst);
            assert_eq!(permit.import(packet).err(), Some(Error::Allocation));
        }
        budget.fail_after.store(-1, Ordering::SeqCst);
        drop((mask, b, x, w, d));
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn permit_and_packet_each_independently_retain_singleton_and_charge() {
    let _guard = matrix::TEST_SERIAL.lock().unwrap();
    for packet_last in [false, true] {
        let budget = Budget::new(64 << 20).unwrap();
        let mut d = dealer(&budget);
        let baseline = budget.used_bytes();
        let (mask, [a, b], x, w) = issue(&mut d, MatrixRange::RequireSignedNoWrap).unwrap();
        let original_wrapper_charge = a.lease.as_ref().unwrap().bytes;
        let native_base = matrix::party::resources(spec())
            .unwrap()
            .party_reservation_bytes;
        let cost = matrix_material_resources(spec()).unwrap();
        let native_packet_metadata = size_of::<matrix::party::material::Packet<'static>>();
        let packet_charge = cost.packet_bytes
            + native_packet_metadata
            + cost.native_allocator_allowance_bytes
            + cost.typed_packet_metadata_bytes;
        let permit_charge =
            native_base + original_wrapper_charge + cost.incremental_reservation_bytes
                - packet_charge;
        let (permit, packet) = a.export_material().unwrap().into_parts();
        drop((mask, b, x, w));
        assert_eq!(
            budget.used_bytes(),
            baseline + packet_charge + permit_charge
        );
        assert_eq!(
            issue(&mut d, MatrixRange::RingOnly).err(),
            Some(Error::Resource)
        );
        if packet_last {
            drop(permit);
            assert_eq!(budget.used_bytes(), baseline + packet_charge);
            assert_eq!(
                issue(&mut d, MatrixRange::RingOnly).err(),
                Some(Error::Resource)
            );
            drop(packet);
        } else {
            drop(packet);
            assert_eq!(budget.used_bytes(), baseline + permit_charge);
            assert_eq!(
                issue(&mut d, MatrixRange::RingOnly).err(),
                Some(Error::Resource)
            );
            drop(permit);
        }
        assert_eq!(budget.used_bytes(), baseline);
        drop(issue(&mut d, MatrixRange::RingOnly).unwrap());
        assert_eq!(budget.used_bytes(), baseline);
        drop(d);
        assert_eq!(budget.used_bytes(), 0);
    }
}
