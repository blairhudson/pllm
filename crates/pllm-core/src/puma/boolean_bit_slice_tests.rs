use super::*;
use crate::puma::multiply::{conversion, tests::parties, Status};
use crate::puma::prss::PrssLimits;

// Independent XOR oracle, not arithmetic shares, maps, or production helpers.
// Full-u64 components deliberately include live padding above nominal width.
fn fixture(owners: &[PartyRuntime; 3], width: u8, seed: u64) -> ([boolean::Share; 3], [u64; 8]) {
    let mask = u64::MAX >> (64 - width);
    let clear = [
        0,
        mask,
        mask / 2,
        1,
        1 << (width - 1),
        151935 & mask,
        seed & mask,
        0xaaaa_aaaa_aaaa_aaaa & mask,
    ];
    let a: [u64; 8] =
        std::array::from_fn(|r| 0xfedc_ba98_7654_3210_u64.wrapping_add(seed ^ r as u64));
    let b: [u64; 8] = std::array::from_fn(|r| {
        0x0123_4567_89ab_cdef_u64
            .rotate_left(r as u32)
            .wrapping_sub(seed)
    });
    let c: [u64; 8] = std::array::from_fn(|r| clear[r] ^ a[r] ^ b[r]);
    let rows = [&a[..], &b[..], &c[..]];
    let op = Operation::new(owners[0].context(), 0, Party::P0, 8, 1, Limits::default()).unwrap();
    let shares = [Party::P0, Party::P1, Party::P2].map(|p| {
        let mut value = ReplicatedShare::from_local(
            op,
            p,
            rows[p as usize],
            rows[p.next() as usize],
            Limits::default(),
        )
        .unwrap();
        value
            .charge_to(&owners[p as usize].payload_usage())
            .unwrap();
        boolean::Share { value, bits: width }
    });
    (shares, clear)
}

#[test]
fn bit_slice_all_widths_full_ring_xor_components_and_tails() {
    for width in 1..=64 {
        let mut owners = parties(PrssLimits::default());
        let (source, clear) = fixture(&owners, width, 17);
        let origin = source[0].binding();
        let snapshots: [[u64; 8]; 3] =
            std::array::from_fn(|p| source[p].first().try_into().unwrap());
        let total = 8 * usize::from(width);
        for start in (0..total).step_by(128) {
            let count = 128.min(total - start);
            let output: [boolean::Share; 3] = std::array::from_fn(|p| {
                let before = owners[p].cursors();
                let ordinal = owners[p].source.runtime.next_sequence();
                let out = owners[p]
                    .boolean_bit_slice(&source[p], width, start, count)
                    .unwrap();
                assert_eq!(owners[p].cursors(), before);
                assert_eq!(out.value.operation.sequence(), ordinal);
                assert_eq!(owners[p].source.runtime.next_sequence(), ordinal + 1);
                assert_eq!(owners[p].payload_usage().live_bytes(), 16 * (8 + count));
                out
            });
            for p in 0..3 {
                assert_eq!(output[p].bits(), 1);
                assert_eq!(output[p].binding(), output[0].binding());
                assert_eq!(output[p].second(), output[(p + 1) % 3].first());
                assert_eq!(source[p].binding(), origin);
                assert_eq!(source[p].first(), snapshots[p]);
                for i in 0..count {
                    let flat = start + i;
                    let row = flat / usize::from(width);
                    let bit = flat % usize::from(width);
                    assert_eq!(output[p].first()[i], (source[p].first()[row] >> bit) & 1);
                    assert_eq!(output[p].second()[i], (source[p].second()[row] >> bit) & 1);
                    assert!(output[p].first()[i] <= 1 && output[p].second()[i] <= 1);
                    let got = output.iter().fold(0_u64, |x, part| x ^ part.first()[i]);
                    assert_eq!(got, (clear[row] >> bit) & 1);
                }
            }
            drop(output);
        }
        drop(source);
        assert!(owners.iter().all(|p| p.payload_usage().live_bytes() == 0));
    }
}

#[test]
fn bit_slice_128_plus_16_retains_source_and_independent_leases() {
    let mut owners = parties(PrssLimits::default());
    let (source, _) = fixture(&owners, 18, 9);
    let usages = owners.each_ref().map(|p| p.payload_usage());
    let head = std::array::from_fn::<_, 3, _>(|p| {
        owners[p].boolean_bit_slice(&source[p], 18, 0, 128).unwrap()
    });
    let tail = std::array::from_fn::<_, 3, _>(|p| {
        owners[p]
            .boolean_bit_slice(&source[p], 18, 128, 16)
            .unwrap()
    });
    for p in 0..3 {
        assert_ne!(head[p].binding(), tail[p].binding());
        assert_eq!(usages[p].live_bytes(), 16 * (8 + 128 + 16));
        owners[p].cancel();
    }
    drop(owners);
    drop(source);
    assert!(usages.iter().all(|u| u.live_bytes() == 16 * 144));
    drop(head);
    assert!(usages.iter().all(|u| u.live_bytes() == 16 * 16));
    drop(tail);
    assert!(usages.iter().all(|u| u.live_bytes() == 0));
}

#[test]
fn bit_slice_binding_uses_public_origin_geometry_not_secret_components() {
    let make = |width, start, count, seed, origin| {
        let mut owners = parties(PrssLimits::default());
        let (mut source, _) = fixture(&owners, width, seed);
        source[0].value.operation.consumer = Some([origin; 32]);
        owners[0]
            .boolean_bit_slice(&source[0], width, start, count)
            .unwrap()
            .binding()
    };
    let a = make(18, 0, 128, 1, 7);
    assert_eq!(a, make(18, 0, 128, u64::MAX, 7));
    assert_ne!(a, make(17, 0, 128, 1, 7));
    assert_ne!(a, make(18, 1, 128, 1, 7));
    assert_ne!(a, make(18, 0, 127, 1, 7));
    assert_ne!(a, make(18, 0, 128, 1, 8));
}

#[test]
fn bit_slice_invalid_geometry_role_context_and_import_retire() {
    for (width, start, count) in [
        (0, 0, 1),
        (65, 0, 1),
        (17, 0, 1),
        (19, 0, 1),
        (18, 0, 0),
        (18, 0, 129),
        (18, 144, 1),
        (18, 129, 16),
        (18, usize::MAX, 1),
    ] {
        let mut owners = parties(PrssLimits::default());
        let (source, _) = fixture(&owners, 18, 3);
        let before = owners[0].cursors();
        assert!(owners[0]
            .boolean_bit_slice(&source[0], width, start, count)
            .is_err());
        assert_eq!(owners[0].status(), Status::Retired);
        assert_eq!(owners[0].cursors(), before);
        assert!(owners[0].boolean_bit_slice(&source[0], 18, 0, 1).is_err());
    }
    for bad in 0..5 {
        let mut owners = parties(PrssLimits::default());
        let (mut source, _) = fixture(&owners, 18, 3);
        match bad {
            0 => source[0].value.party = Party::P1,
            1 => source[0].value.operation.context = Context::new([99; 32], [98; 32], 9).unwrap(),
            2 => source[0].value.operation.lanes = 7,
            _ => {
                let mut raw = ReplicatedShare::from_local(
                    source[0].value.operation,
                    Party::P0,
                    source[0].first(),
                    source[0].second(),
                    Limits::default(),
                )
                .unwrap();
                if bad == 4 {
                    raw.first.charge_to(&owners[0].payload_usage()).unwrap();
                }
                source[0] = boolean::Share {
                    value: raw,
                    bits: 18,
                };
            }
        }
        assert!(owners[0].boolean_bit_slice(&source[0], 18, 0, 1).is_err());
        assert_eq!(owners[0].status(), Status::Retired);
    }
}

#[test]
fn bit_slice_old_runtime_wrong_phase_cancellation_and_budget_fail_closed() {
    let mut old = parties(PrssLimits::default());
    let (source, _) = fixture(&old, 18, 2);
    old[0].cancel();
    let mut fresh = parties(PrssLimits::default());
    assert_eq!(fresh[0].context(), old[0].context());
    assert!(fresh[0].boolean_bit_slice(&source[0], 18, 0, 1).is_err());
    assert_eq!(fresh[0].status(), Status::Retired);
    assert!(old[0].boolean_bit_slice(&source[0], 18, 0, 1).is_err());
    let mut owners = parties(PrssLimits::program(1, 0).unwrap());
    let (source, _) = fixture(&owners, 18, 4);
    let out = owners[0].boolean_bit_slice(&source[0], 18, 0, 128).unwrap();
    assert_eq!(owners[0].cursors(), Default::default());
    assert!(owners[0]
        .boolean_bit_slice(&source[0], 18, 128, 16)
        .is_err());
    assert_eq!(owners[0].status(), Status::Retired);
    drop(out);
    let mut owners = parties(PrssLimits::default());
    let (source, _) = fixture(&owners, 18, 4);
    owners[0]
        .plan_conversion(8, conversion::Kind::LowBits(18))
        .unwrap();
    assert!(owners[0].boolean_bit_slice(&source[0], 18, 0, 1).is_err());
    assert_eq!(owners[0].status(), Status::Retired);
}

#[test]
fn bit_slice_repeated_views_have_distinct_bindings_and_mpc_substitution_retires() {
    let mut owners = parties(PrssLimits::default());
    let (source, _) = fixture(&owners, 18, 4);
    let a = owners[0].boolean_bit_slice(&source[0], 18, 0, 128).unwrap();
    let b = owners[0].boolean_bit_slice(&source[0], 18, 0, 128).unwrap();
    assert_eq!(a.first(), b.first());
    assert_ne!(a.binding(), b.binding());
    owners[0].plan_boolean_to_arithmetic(&a).unwrap();
    assert!(owners[0].begin_boolean_to_arithmetic(b).is_err());
    assert_eq!(owners[0].status(), Status::Retired);
}

#[test]
fn bit_slice_retained_outputs_cannot_exceed_runtime_payload_ceiling() {
    let mut owners = parties(PrssLimits::default());
    let (source, _) = fixture(&owners, 18, 4);
    let usage = owners[0].payload_usage();
    let mut retained = Vec::new();
    while usage.live_bytes() + 16 * 128 <= usage.limit_bytes() {
        retained.push(owners[0].boolean_bit_slice(&source[0], 18, 0, 128).unwrap());
    }
    let before = usage.live_bytes();
    let ordinal = owners[0].source.runtime.next_sequence();
    assert!(owners[0].boolean_bit_slice(&source[0], 18, 0, 128).is_err());
    assert_eq!(owners[0].status(), Status::Retired);
    assert_eq!(owners[0].source.runtime.next_sequence(), ordinal);
    assert_eq!(usage.live_bytes(), before);
    drop(source);
    assert_eq!(usage.live_bytes(), retained.len() * 16 * 128);
    drop(retained);
    assert_eq!(usage.live_bytes(), 0);
}
