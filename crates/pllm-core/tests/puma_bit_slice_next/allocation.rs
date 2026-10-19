use super::*;
use multiply::{boolean, conversion, PartyRuntime, Status};

// Real A2B owner construction, outside the allocation-failure/watch interval.
fn source() -> ([PartyRuntime; 3], [boolean::Share; 3]) {
    let mut owners = compound_owners(PrssLimits::default());
    let plans = owners
        .each_mut()
        .map(|p| p.plan_conversion(8, conversion::Kind::LowBits(18)).unwrap());
    assert!(plans.iter().all(|p| p.digest() == plans[0].digest()));
    let a = [u64::MAX - 13; 8];
    let b = [0x1357_9bdf_2468_ace0_u64; 8];
    let values = [0_u64, 1, 2, 151935, 131072, 17, 262143, 999];
    let c = values.map(|x| x.wrapping_sub(a[0]).wrapping_sub(b[0]));
    let rows = [&a[..], &b[..], &c[..]];
    let mut messages: Vec<_> = owners
        .iter_mut()
        .enumerate()
        .map(|(p, owner)| {
            let party = owner.party();
            let input = plans[p]
                .share_input(
                    party,
                    rows[p],
                    rows[party.next() as usize],
                    Limits::default(),
                )
                .unwrap();
            owner.begin_conversion(input).unwrap()
        })
        .collect();
    while !messages.is_empty() {
        let mut next = Vec::new();
        for owner in &mut owners {
            let at = messages
                .iter()
                .position(|m| m.header().to == owner.party())
                .unwrap();
            let received = messages.remove(at);
            if let Some(reply) = owner.advance_conversion(received).unwrap() {
                next.push(reply);
            }
        }
        assert!(messages.is_empty());
        messages = next;
    }
    let shares = owners.each_mut().map(|p| p.take_boolean_output().unwrap());
    (owners, shares)
}

#[test]
fn bit_slice_allocation_failures_retire_without_refunding_retained_source() {
    for fail_at in 0..=2 {
        let (mut owners, shares) = source();
        let usage = owners[0].payload_usage();
        let baseline = usage.live_bytes();
        let first: [u64; 8] = shares[0].first().try_into().unwrap();
        let watch = Watch::start(&usage, true, false);
        FAIL_AT.set(Some(fail_at));
        let result = owners[0].boolean_bit_slice(&shares[0], 18, 0, 128);
        FAIL_AT.set(None);
        if fail_at < 2 {
            assert!(matches!(result, Err(Error::Allocation)));
            assert_eq!(owners[0].status(), Status::Retired);
        } else {
            assert!(result.is_ok());
            assert_eq!(usage.live_bytes(), baseline + 16 * 128);
        }
        assert_eq!(shares[0].first(), first);
        drop(result);
        assert_eq!(usage.live_bytes(), baseline);
        let audit = watch.finish();
        assert_eq!(audit.allocations, fail_at.min(2));
        assert_audit(audit);
        drop(shares);
        drop(owners);
        assert_eq!(usage.live_bytes(), 0);
    }
}

#[test]
fn bit_slice_retained_128_plus_16_wipes_before_refund_after_cancel() {
    for p in 0..3 {
        let (mut owners, shares) = source();
        let usage = owners[p].payload_usage();
        let baseline = usage.live_bytes();
        let watch = Watch::start(&usage, true, false);
        let head = owners[p].boolean_bit_slice(&shares[p], 18, 0, 128).unwrap();
        let tail = owners[p]
            .boolean_bit_slice(&shares[p], 18, 128, 16)
            .unwrap();
        assert_eq!(usage.live_bytes(), baseline + 16 * 144);
        owners[p].cancel();
        assert_eq!(usage.live_bytes(), baseline + 16 * 144);
        drop(shares);
        drop(owners);
        assert_eq!(usage.live_bytes(), 16 * 144);
        drop(head);
        assert_eq!(usage.live_bytes(), 16 * 16);
        drop(tail);
        assert_eq!(usage.live_bytes(), 0);
        let audit = watch.finish();
        assert_eq!(audit.allocations, 4);
        assert_audit(audit);
    }
}
