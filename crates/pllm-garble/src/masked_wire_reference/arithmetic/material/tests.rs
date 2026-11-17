use super::*;
use std::sync::atomic::Ordering;

#[derive(Clone, Copy)]
struct Observation {
    original_pointer: usize,
    original_len: usize,
    cleanup_count: usize,
    nonzero_before: bool,
    original_allocation: bool,
    all_zero: bool,
}
std::thread_local! {
    static RETIREMENT: std::cell::Cell<Option<Observation>> = const { std::cell::Cell::new(None) };
}

// Observer only reads still-live fields and compares pointer identity. No
// allocation, freed-memory access, secret substitute wipe, or asserting callback.
pub(super) fn observe_retired_lanes(lanes: &[Lane]) {
    RETIREMENT.with(|slot| {
        if let Some(mut observation) = slot.get() {
            observation.cleanup_count = observation.cleanup_count.saturating_add(1);
            observation.original_allocation &= lanes.as_ptr() as usize == observation.original_pointer
                && lanes.len() == observation.original_len;
            observation.all_zero &= lanes.iter().all(|lane|
                lane.key.secret_fields_are_zero() && lane.constants.iter().all(|c| *c == 0));
            slot.set(Some(observation));
        }
    });
}

struct ObserveRetirement;
impl ObserveRetirement {
    fn start(lanes: &[Lane]) -> Self {
        RETIREMENT.with(|slot| slot.set(Some(Observation {
            original_pointer: lanes.as_ptr() as usize,
            original_len: lanes.len(), cleanup_count: 0,
            nonzero_before: lanes.iter().any(|lane| !lane.key.secret_fields_are_zero())
                && lanes.iter().any(|lane| lane.constants.iter().any(|c| *c != 0)),
            original_allocation: true, all_zero: true,
        })));
        Self
    }
    fn finish(self) -> Observation {
        // Assertions occur after export has returned, never inside cleanup.
        RETIREMENT.with(|slot| slot.take()).expect("retirement observation armed")
    }
}
impl Drop for ObserveRetirement {
    fn drop(&mut self) { RETIREMENT.with(|slot| slot.set(None)); }
}

fn assert_retirement(observation: Observation) {
    assert_eq!(observation.cleanup_count, 1);
    assert!(observation.nonzero_before);
    assert!(observation.original_allocation);
    assert!(observation.all_zero);
}

fn context(tag: u8) -> Context {
    Context { plan: [1; 32], session: [2; 32], operation: [tag; 32],
        input_wire: [3; 32], output_wire: [4; 32], first_tensor_index: 9 }
}
fn spec() -> Spec { Spec { bits: 8, lanes: 2, gate: Gate::Tr { shift: 3 } } }
fn issue_pair<'b>(spec: Spec, tag: u8, budget: &'b Budget) -> (OutputMasks<'b>, [Party<'b>; 2]) {
    let masks = vec![5; spec.lanes];
    let mut byte = tag;
    super::super::issue_with(spec, context(tag), &masks, budget, |out| {
        for value in out { byte = byte.wrapping_mul(17).wrapping_add(31); *value = byte; }
        Ok(())
    }).unwrap()
}
fn transfer(party: Party<'_>) -> Party<'_> {
    let (permit, packet) = party.export_material().unwrap().into_parts();
    permit.import(packet).unwrap()
}
fn run<'b>(parties: &mut [Party<'b>; 2], input: &[u64]) -> [MaskedOutput<'b>; 2] {
    let [a, b] = parties;
    let left = a.begin(input).unwrap();
    let right = b.begin(input).unwrap();
    match (a.advance(right.as_bytes()).unwrap(), b.advance(left.as_bytes()).unwrap()) {
        (Step::Output(a), Step::Output(b)) => [a, b],
        (Step::Frame(a_frame), Step::Frame(b_frame)) => {
            let Step::Output(a) = a.advance(b_frame.as_bytes()).unwrap() else { panic!("output") };
            let Step::Output(b) = b.advance(a_frame.as_bytes()).unwrap() else { panic!("output") };
            [a, b]
        }
        _ => panic!("schedule"),
    }
}

#[test]
fn material_roundtrip_all_gates_and_ring_widths() {
    for bits in [3, 8, 38, 50, 64] {
        for shift in 1..bits {
            for gate in [Gate::Tr { shift }, Gate::GapArs { shift }, Gate::Drelu] {
                if matches!(gate, Gate::GapArs { .. }) && shift == bits - 1 { continue; }
                let budget = Budget::new(64 << 20).unwrap();
                let spec = Spec { bits, lanes: 3, gate };
                let (masks, [left, right]) = issue_pair(spec, 7, &budget);
                let r = resources(spec).unwrap();
                let mut restored = [left, right].map(|party| {
                    let role = party.id;
                    let binding = party.binding;
                    // Borrowed encoder must preserve consuming encoder bytes and
                    // exact callback size. Consume only a disposable decoded key.
                    for lane in &party.state.as_ref().unwrap().lanes {
                        let mut requested = None;
                        let borrowed = lane.key.material_with_reservation(r.key_codec_bytes, |n| {
                            requested = Some(n); Ok(())
                        }).unwrap();
                        assert_eq!(requested, Some(r.key_codec_bytes));
                        let disposable = GpuKey::decode(lane.key.profile(), role, &borrowed).unwrap();
                        let consumed = disposable.into_material(r.key_codec_bytes).unwrap();
                        assert_eq!(&*borrowed, &*consumed);
                    }
                    let (permit, packet) = party.export_material().unwrap().into_parts();
                    assert_eq!(packet.as_bytes().len(), r.packet_bytes);
                    assert_eq!(&packet.as_bytes()[..8], MAGIC);
                    assert_eq!(&packet.as_bytes()[8..16], &[1, role, 0, 0, 0, 0, 0, 0]);
                    assert_eq!(&packet.as_bytes()[48..80], &binding);
                    // Independent serialization of descriptor contract; do not
                    // call implementation's descriptor helper as the oracle.
                    let profile = spec.geometry().unwrap().0;
                    let out = spec.geometry().unwrap().1;
                    let (tag, amount) = match gate {
                        Gate::Tr { shift } => (0, shift),
                        Gate::GapArs { shift } => (1, shift),
                        Gate::Drelu => (2, 0),
                    };
                    let mut description = b"pllm.masked-wire.arithmetic-material.v1\0".to_vec();
                    for domain in [super::super::DOMAIN.as_bytes(), Profile::DOMAIN.as_bytes()] {
                        description.extend_from_slice(&(domain.len() as u64).to_le_bytes());
                        description.extend_from_slice(domain);
                    }
                    description.extend_from_slice(&[bits, profile.bits(), out, tag, amount, role]);
                    description.extend_from_slice(&(spec.lanes as u64).to_le_bytes());
                    description.extend_from_slice(&binding);
                    assert_eq!(&packet.as_bytes()[16..48], &Sha256::digest(&description)[..]);
                    let bytes = packet.as_bytes().to_vec(); // test-only comparison copy
                    let party = permit.import(packet).unwrap();
                    let (permit, packet) = party.export_material().unwrap().into_parts();
                    assert_eq!(packet.as_bytes(), bytes);
                    permit.import(packet).unwrap()
                });
                // Independent signed fixture lies within quarter-ring even at n=3.
                let clear = [-1i128, 0, 1];
                let input = clear.map(|v| (v + 5) as u64 & mask(bits));
                let outputs = run(&mut restored, &input);
                assert_eq!(outputs[0].values(), outputs[1].values());
                for (index, value) in clear.into_iter().enumerate() {
                    let actual = if gate == Gate::Drelu {
                        outputs[0].values()[index] ^ masks.values()[index]
                    } else {
                        outputs[0].values()[index].wrapping_sub(masks.values()[index])
                            & mask(spec.geometry().unwrap().1)
                    };
                    let expected = match gate {
                        Gate::Drelu => u64::from(value >= 0),
                        Gate::Tr { shift } | Gate::GapArs { shift } =>
                            value.div_euclid(1i128 << shift) as u64 & mask(spec.geometry().unwrap().1),
                    };
                    assert_eq!(actual, expected, "{spec:?} lane {index}");
                }
                drop((outputs, restored, masks));
                assert_eq!(budget.used_bytes(), 0);
            }
        }
        // Maximum TR shift, separate from GapARS's stricter geometry.
        resources(Spec { bits, lanes: 1, gate: Gate::Tr { shift: bits - 1 } }).unwrap();
    }
}

#[test]
fn material_rejects_cross_role_spec_context_and_issuance() {
    for variant in 0..4 {
        let budget = Budget::new(64 << 20).unwrap();
        let (ma, [a, ar]) = issue_pair(spec(), 7, &budget);
        let other_spec = if variant == 1 { Spec { gate: Gate::GapArs { shift: 3 }, ..spec() } } else { spec() };
        let (mb, [b, br]) = issue_pair(other_spec, if variant == 2 { 8 } else { 7 }, &budget);
        let (permit, packet_a) = a.export_material().unwrap().into_parts();
        let other = if variant == 0 { ar } else { drop(ar); b };
        // For variant 3 force another issuance under identical public context.
        let other = if variant == 3 {
            drop(other);
            let (mc, [c, cr]) = super::super::issue_with(spec(), context(7), &[5, 5], &budget,
                |out| { out.fill(19); Ok(()) }).unwrap();
            drop((mc, cr));
            c
        } else { other };
        let (other_permit, packet_b) = other.export_material().unwrap().into_parts();
        assert!(matches!(permit.import(packet_b), Err(Error::Context)));
        drop((other_permit, packet_a, br, ma, mb));
        // b remains local only in role-swap branch; ordinary Rust drop retires it.
    }
}

fn replace_packet<'b>(packet: Packet<'b>, bytes: &[u8], budget: &'b Budget) -> Packet<'b> {
    let Packet { packet: old, _metadata } = packet;
    drop(old);
    Packet { packet: MaterialPacket::from_bytes(bytes, budget).unwrap(), _metadata }
}
fn checksum(bytes: &mut [u8]) {
    let end = bytes.len() - 32;
    let digest = Sha256::digest(&bytes[..end]);
    bytes[end..].copy_from_slice(&digest);
}

#[test]
fn material_rejects_rechecksummed_tampering() {
    for offset in [0, 8, 16, 48, 80, 88, 100, 160] {
        let budget = Budget::new(64 << 20).unwrap();
        let (masks, [a, b]) = issue_pair(spec(), 7, &budget);
        let (permit, packet) = a.export_material().unwrap().into_parts();
        let mut bytes = packet.as_bytes().to_vec();
        bytes[offset] ^= 1;
        checksum(&mut bytes);
        let packet = replace_packet(packet, &bytes, &budget);
        assert!(matches!(permit.import(packet), Err(Error::Context)));
        drop((masks, b));
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn material_rejects_noncanonical_body() {
    let budget = Budget::new(64 << 20).unwrap();
    let s = Spec { bits: 16, lanes: 1, gate: Gate::Tr { shift: 9 } };
    let (masks, [a, b]) = issue_pair(s, 7, &budget);
    let (permit, packet) = a.export_material().unwrap().into_parts();
    let body = &packet.as_bytes()[material_wire::HEADER..packet.as_bytes().len() - 32];
    let k = resources(s).unwrap().key_codec_bytes;
    validate_body(s, 0, body).unwrap();
    assert!(validate_body(s, 0, &body[..body.len() - 1]).is_err());
    let mut trailing = body.to_vec(); trailing.push(0);
    assert!(validate_body(s, 0, &trailing).is_err());
    assert!(validate_body(s, 1, body).is_err());
    assert!(validate_body(Spec { gate: Gate::Tr { shift: 8 }, ..s }, 0, body).is_err());
    // Key header profile/role, unused final padding bit, oversized active constant,
    // and forbidden TR slot 1. Private parser bypasses trusted digest on purpose.
    for (offset, bit) in [(16, 1), (17, 1), (k - 1, 128), (k + 7, 128), (k + 8, 1)] {
        let mut bad = body.to_vec(); bad[offset] ^= bit;
        assert!(validate_body(s, 0, &bad).is_err(), "offset {offset}");
    }
    let d = Spec { bits: 8, lanes: 1, gate: Gate::Drelu };
    let (dm, [da, db]) = issue_pair(d, 4, &budget);
    let (dp, packet_d) = da.export_material().unwrap().into_parts();
    let mut body_d = packet_d.as_bytes()[material_wire::HEADER..packet_d.as_bytes().len() - 32].to_vec();
    let dk = resources(d).unwrap().key_codec_bytes;
    body_d[dk] = 2;
    assert!(validate_body(d, 0, &body_d).is_err());
    body_d[dk] = 0; body_d[dk + 16] = 1;
    assert!(validate_body(d, 0, &body_d).is_err());
    drop((permit, packet, masks, b, dm, db, dp, packet_d));
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn material_export_requires_fresh_owner() {
    for phase in [0, 2] {
        let budget = Budget::new(64 << 20).unwrap();
        let (masks, [mut a, b]) = issue_pair(spec(), 7, &budget);
        match phase {
            0 => { drop(a.begin(&[5, 6]).unwrap()); }
            _ => a.cancel(),
        }
        assert!(matches!(a.export_material(), Err(Error::Consumed)));
        drop((b, masks));
        assert_eq!(budget.used_bytes(), 0);
    }
    let budget = Budget::new(64 << 20).unwrap();
    let (masks, mut parties) = issue_pair(spec(), 7, &budget);
    drop(run(&mut parties, &[5, 6]));
    for party in parties { assert!(matches!(party.export_material(), Err(Error::Consumed))); }
    drop(masks); assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn material_admission_and_drop_preserve_cover() {
    let s = spec();
    let r = resources(s).unwrap();
    assert_eq!(r.body_bytes, s.lanes * (r.key_codec_bytes + 24));
    assert_eq!(r.packet_bytes, 120 + r.body_bytes);
    assert_eq!(r.combined_reservation_bytes, r.original_reservation_bytes + r.incremental_reservation_bytes);
    assert!(resources(Spec { lanes: usize::MAX, ..s }).is_err());
    assert!(resources(Spec { bits: 2, ..s }).is_err());
    // Invalid private geometry/scratch/role must reject before codec allocation.
    for malformed in 0..5 {
        let budget = Budget::new(64 << 20).unwrap();
        let (masks, [mut a, b]) = issue_pair(s, 7, &budget);
        match malformed {
            0 => a.id = 2,
            1 => a.spec.lanes += 1,
            2 => a.state.as_mut().unwrap().input[0] = 1,
            3 => a.state.as_mut().unwrap().local[0] = 1,
            _ => a.state.as_mut().unwrap().lanes[0].constants[1] = 1,
        }
        budget.fail_after.store(0, Ordering::SeqCst);
        assert!(a.export_material().is_err());
        assert_eq!(budget.fail_after.load(Ordering::SeqCst), 0);
        drop((masks, b)); assert_eq!(budget.used_bytes(), 0);
    }
    for short in [false, true] {
        let initial = super::super::resources(s).unwrap().combined_reservation_bytes;
        let budget = Budget::new(initial + r.incremental_reservation_bytes - usize::from(short)).unwrap();
        let (masks, [a, b]) = issue_pair(s, 7, &budget);
        if short { budget.fail_after.store(0, Ordering::SeqCst); }
        let result = a.export_material();
        if short {
            assert!(matches!(result, Err(Error::Resource)));
            assert_eq!(budget.fail_after.load(Ordering::SeqCst), 0);
        } else {
            let (permit, packet) = result.unwrap().into_parts();
            let party = permit.import(packet).unwrap();
            drop(party);
        }
        drop((masks, b));
        assert_eq!(budget.used_bytes(), 0);
    }
    for fail in 0..=s.lanes {
        let budget = Budget::new(64 << 20).unwrap();
        let (masks, [a, b]) = issue_pair(s, 7, &budget);
        budget.fail_after.store(fail as isize, Ordering::SeqCst);
        assert!(matches!(a.export_material(), Err(Error::Allocation)));
        drop((masks, b)); assert_eq!(budget.used_bytes(), 0);
    }
    for fail in 0..3 {
        let budget = Budget::new(64 << 20).unwrap();
        let (masks, [a, b]) = issue_pair(s, 7, &budget);
        let (permit, packet) = a.export_material().unwrap().into_parts();
        budget.fail_after.store(fail, Ordering::SeqCst);
        assert!(matches!(permit.import(packet), Err(Error::Allocation)));
        drop((masks, b)); assert_eq!(budget.used_bytes(), 0);
    }
    for permit_first in [false, true] {
        let budget = Budget::new(64 << 20).unwrap();
        let (masks, [a, b]) = issue_pair(s, 7, &budget);
        drop((masks, b));
        let original_budget = a.lease.budget;
        let original_bytes = a.lease.bytes;
        let (permit, packet) = a.export_material().unwrap().into_parts();
        assert!(std::ptr::eq(permit.original.budget, original_budget));
        assert_eq!(permit.original.bytes, original_bytes);
        assert_eq!(budget.used_bytes(), original_bytes + r.incremental_reservation_bytes);
        if permit_first {
            drop(permit);
            assert_eq!(budget.used_bytes(), r.packet_bytes + size_of::<Packet<'static>>() + ALLOCATOR);
            drop(packet);
        } else {
            drop(packet);
            assert_eq!(budget.used_bytes(), original_bytes + r.incremental_reservation_bytes
                - r.packet_bytes - size_of::<Packet<'static>>() - ALLOCATOR);
            drop(permit);
        }
        assert_eq!(budget.used_bytes(), 0);
    }
    let budget = Budget::new(64 << 20).unwrap();
    let (masks, [a, b]) = issue_pair(s, 7, &budget);
    let bytes = a.lease.bytes;
    let party = transfer(a);
    assert_eq!(party.lease.bytes, bytes);
    assert!(std::ptr::eq(party.lease.budget, &budget));
    drop((party, masks, b)); assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn material_export_clears_original_lane_slots_before_deallocation() {
    let budget = Budget::new(64 << 20).unwrap();
    let s = spec();
    let (masks, [a, b]) = issue_pair(s, 7, &budget);
    let observer = ObserveRetirement::start(&a.state.as_ref().unwrap().lanes);
    // Packet plus exactly two borrowed-key allocations; retirement has no
    // allocation hook and no heap allocation in its source or observer.
    budget.fail_after.store(3, Ordering::SeqCst);
    let result = a.export_material();
    let observation = observer.finish();
    assert_retirement(observation);
    assert_eq!(budget.fail_after.load(Ordering::SeqCst), 0);
    let (permit, packet) = result.unwrap().into_parts();
    budget.fail_after.store(-1, Ordering::SeqCst);
    let mut parties = [permit.import(packet).unwrap(), b];
    let outputs = run(&mut parties, &[4, 6]); // clear [-1, 1], masks [5, 5]
    assert_eq!(outputs[0].values(), outputs[1].values());
    for (i, clear) in [-1i64, 1].into_iter().enumerate() {
        assert_eq!(outputs[0].values()[i].wrapping_sub(masks.values()[i]) & 31,
            clear.div_euclid(8) as u64 & 31);
    }
    drop((outputs, parties, masks));
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn material_later_key_encode_failure_clears_original_lane_slots() {
    let budget = Budget::new(64 << 20).unwrap();
    let (masks, [a, b]) = issue_pair(spec(), 7, &budget);
    let observer = ObserveRetirement::start(&a.state.as_ref().unwrap().lanes);
    // allocation(): positive n decrements; zero refuses without decrement.
    // Shared packet: 2->1; first key: 1->0; second key: Allocation at zero.
    budget.fail_after.store(2, Ordering::SeqCst);
    let result = a.export_material();
    let observation = observer.finish();
    assert!(matches!(result, Err(Error::Allocation)));
    assert_eq!(budget.fail_after.load(Ordering::SeqCst), 0);
    assert_retirement(observation); // both original slots, including first key
    drop((masks, b));
    assert_eq!(budget.used_bytes(), 0);
}
