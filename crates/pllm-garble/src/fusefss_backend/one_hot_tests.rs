use super::*;
use std::sync::atomic::Ordering;

fn spec(vocabulary: usize) -> Spec {
    Spec {
        vocabulary,
        source_digest: [61; 32],
        profile_digest: [62; 32],
    }
}
fn context() -> Context {
    Context {
        plan: [63; 32],
        session: [64; 32],
        operation: [65; 32],
        first_tensor_index: 7,
    }
}
fn forced(s: Spec, r: u64, r0: u64, budget: &Budget) -> [Party<'_>; 2] {
    let mut first = true;
    issue_with(s, context(), budget, |bytes| {
        if first && s.vocabulary > 1 {
            assert_eq!(bytes.len(), 16);
            bytes[..8].copy_from_slice(&r.to_le_bytes());
            bytes[8..].copy_from_slice(&r0.to_le_bytes());
            first = false;
            Ok(())
        } else {
            getrandom::fill(bytes).map_err(|_| PrimitiveError::Randomness)
        }
    })
    .unwrap()
}
fn parity(p: &Party<'_>, exhaustive: bool) {
    let State::Ready(secret, _) = p.state.as_ref().unwrap() else {
        panic!()
    };
    if let Some(key) = &secret.key {
        let n = p.resources.domain_entries;
        let mut all = vec![0u64; n];
        key.eval_all_into(p.id, &mut all).unwrap();
        if exhaustive {
            for (i, &v) in all.iter().enumerate() {
                assert_eq!(v, key.eval(p.id, i as u64).unwrap());
            }
        } else {
            for i in [0, 1, n / 2 - 1, n / 2, n - 2, n - 1]
                .into_iter()
                .chain((0..128).map(|j| j * 7919 % n))
            {
                assert_eq!(all[i], key.eval(p.id, i as u64).unwrap());
            }
        }
        assert!(key.eval_all_into(2, &mut all).is_err());
        assert!(key.eval_all_into(p.id, &mut all[..n - 1]).is_err());
    }
}
fn execute(mut parties: [Party<'_>; 2], x: u64, share: u64) -> ([Output<'_>; 2], u64) {
    let [a, b] = &mut parties;
    let af = a.start(share).unwrap();
    let bf = b.start(x.wrapping_sub(share) & A50_MASK).unwrap();
    let t = (u64::from_le_bytes(af.as_bytes()[80..].try_into().unwrap())
        + u64::from_le_bytes(bf.as_bytes()[80..].try_into().unwrap()))
        & ((1u64 << a.resources.input_bits) - 1);
    let ao = a.finish(bf.as_bytes()).unwrap();
    let bo = b.finish(af.as_bytes()).unwrap();
    assert!(a.is_consumed() && b.is_consumed());
    assert!(matches!(a.start(0), Err(Error::Consumed)));
    ([ao, bo], t)
}
fn check(outputs: &[Output<'_>; 2], x: u64, v: usize) {
    assert_eq!(outputs[0].shares().len(), v);
    assert_eq!(outputs[1].shares().len(), v);
    let mut sum = 0;
    for (i, (&a, &b)) in outputs[0]
        .shares()
        .iter()
        .zip(outputs[1].shares())
        .enumerate()
    {
        assert!(a <= A50_MASK && b <= A50_MASK);
        let value = a.wrapping_add(b) & A50_MASK;
        assert_eq!(value, u64::from(i as u64 == x));
        sum += value;
    }
    assert_eq!(sum, u64::from(x < v as u64));
}

#[test]
fn every_small_vocab_threshold_token_including_zero_wrap_and_padding() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    for v in 2usize..=16 {
        let s = spec(v);
        let n = v.next_power_of_two();
        for r in 0..n {
            for x in 0..n {
                // Only threshold/split test seam is controlled. Every invocation
                // still uses fresh OS-random DCF roots and fresh issuance nonce.
                let p = forced(s, r as u64, (r * 7 + x) as u64, &budget);
                if x == 0 {
                    for role in &p {
                        parity(role, true);
                    }
                }
                let (out, t) = execute(p, x as u64, A50_MASK - (37 * x + r) as u64);
                assert_eq!(t, ((x + r) & (n - 1)) as u64);
                check(&out, x as u64, v);
                drop(out);
                assert_eq!(budget.used_bytes(), 0);
            }
        }
    }
}

#[test]
fn independent_mask_splits_and_a50_input_narrowing_only() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    for r in 0..8 {
        for r0 in 0..8 {
            for share in [0, 1, (1 << 49) - 1, 1 << 49, A50_MASK - 1, A50_MASK] {
                let (out, t) = execute(forced(spec(5), r, r0, &budget), 4, share);
                check(&out, 4, 5);
                assert_eq!(t, (4 + r) & 7);
                assert!(out[0].shares().iter().any(|&x| x > 7));
            }
        }
    }
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn explicit_vocabulary_one_public_constant_and_large_endpoint_controls() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let cost = resources(spec(1)).unwrap();
    assert_eq!(cost.input_bits, 0);
    assert_eq!(cost.dcf_keys_per_party, 0);
    let (out, t) = execute(issue(spec(1), context(), &budget).unwrap(), 0, A50_MASK);
    check(&out, 0, 1);
    assert_eq!(t, 0);
    drop(out);
    for (v, x) in [
        (50257, 0),
        (151936, 151935),
        (MAX_VOCABULARY, MAX_VOCABULARY - 1),
    ] {
        let p = issue(spec(v), context(), &budget).unwrap();
        for role in &p {
            parity(role, false);
        }
        let (out, _) = execute(p, x as u64, A50_MASK);
        check(&out, x as u64, v);
        // Complete immutable public embedding contraction, independent oracle.
        for column in 0..3 {
            let weight =
                |row: usize| ((row as i64 * 37 + column * 71) % 10001 - 5000) as u64 & A50_MASK;
            let contracted = out
                .iter()
                .map(|role| {
                    role.shares()
                        .iter()
                        .enumerate()
                        .fold(0u64, |sum, (i, &share)| {
                            sum.wrapping_add(share.wrapping_mul(weight(i))) & A50_MASK
                        })
                })
                .fold(0u64, |a, b| a.wrapping_add(b) & A50_MASK);
            assert_eq!(contracted, weight(x));
        }
        drop(out);
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn existing_wire_roundtrip_imported_tree_parity_and_freshness() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut guard = wire::ImportGuard::new(16, &budget).unwrap();
    for v in [1, 2, 5, 17, 50257, 151936] {
        let [a, b] = issue(spec(v), context(), &budget).unwrap();
        let pa = a.export_material().unwrap();
        let pb = b.export_material().unwrap();
        assert_eq!(
            pa.as_bytes().len(),
            resources(spec(v)).unwrap().material_packet_bytes_per_party
        );
        assert_eq!(pa.nonce(), pb.nonce());
        let nonce = pa.nonce();
        let copy = wire::MaterialPacket::from_bytes(pa.as_bytes(), &budget).unwrap();
        let a =
            Party::import_material(spec(v), context(), 0, nonce, pa, &mut guard, &budget).unwrap();
        let b =
            Party::import_material(spec(v), context(), 1, nonce, pb, &mut guard, &budget).unwrap();
        assert!(matches!(
            Party::import_material(spec(v), context(), 0, nonce, copy, &mut guard, &budget),
            Err(Error::Consumed)
        ));
        parity(&a, v < 1000);
        parity(&b, v < 1000);
        let (out, _) = execute([a, b], (v - 1) as u64, A50_MASK);
        check(&out, (v - 1) as u64, v);
    }
    let [a, b] = issue(spec(5), context(), &budget).unwrap();
    let packet = a.export_material().unwrap();
    let old = packet.nonce();
    drop((packet, b));
    let [a, b] = issue(spec(5), context(), &budget).unwrap();
    let packet = a.export_material().unwrap();
    assert_ne!(packet.nonce(), old);
    drop((packet, b, guard));
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn wrong_context_role_order_replay_and_cancel_burn() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let [mut a, mut b] = issue(spec(5), context(), &budget).unwrap();
    let af = a.start(A50_MASK).unwrap();
    let bf = b.start(0).unwrap();
    assert!(matches!(a.finish(af.as_bytes()), Err(Error::Context)));
    assert!(a.is_consumed());
    b.cancel();
    assert!(b.is_consumed());
    assert!(matches!(b.finish(af.as_bytes()), Err(Error::Consumed)));
    drop((a, b, af, bf));
    let [mut a, b] = issue(spec(5), context(), &budget).unwrap();
    assert!(matches!(a.finish(&[]), Err(Error::Context)));
    assert!(a.is_consumed());
    drop((a, b));
    let [mut a, b] = issue(spec(5), context(), &budget).unwrap();
    assert!(matches!(a.start(1 << 50), Err(Error::Domain)));
    assert!(a.is_consumed());
    drop((a, b));
    let [mut a, mut b] = issue(spec(5), context(), &budget).unwrap();
    let af = a.start(0).unwrap();
    let bf = b.start(0).unwrap();
    let mut bytes = bf.as_bytes().to_vec();
    bytes[80..].copy_from_slice(&8u64.to_le_bytes());
    assert!(matches!(a.finish(&bytes), Err(Error::Context)));
    drop((a, b, af, bf));
    let [mut a, b] = issue(spec(5), context(), &budget).unwrap();
    let af = a.start(0).unwrap();
    assert!(matches!(a.start(0), Err(Error::Context)));
    assert!(a.is_consumed());
    drop((a, b, af));
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn import_canonical_preflight_and_burn_on_failure() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut guard = wire::ImportGuard::new(32, &budget).unwrap();
    for mutation in 0..9 {
        let [a, b] = issue(spec(5), context(), &budget).unwrap();
        drop(b);
        let packet = a.export_material().unwrap();
        let nonce = packet.nonce();
        let mut bytes = packet.as_bytes().to_vec();
        let mut s = spec(5);
        let mut ctx = context();
        let mut role = 0;
        let mut expected_nonce = nonce;
        match mutation {
            0 => s.source_digest[0] ^= 1,
            1 => s.profile_digest[0] ^= 1,
            2 => s.vocabulary = 6,
            3 => ctx.operation[0] ^= 1,
            4 => role = 1,
            5 => expected_nonce[0] ^= 1,
            6 => bytes[88..96].copy_from_slice(&8u64.to_le_bytes()), // noncanonical b-bit mask
            7 => bytes[96 + 16..96 + 24].copy_from_slice(&(1u64 << 50).to_le_bytes()),
            _ => bytes[96 + 24 + 24] = 2, // first correction control
        }
        let digest = Sha256::digest(&bytes[..bytes.len() - 32]);
        let end = bytes.len();
        bytes[end - 32..].copy_from_slice(&digest);
        let bad = wire::MaterialPacket::from_bytes(&bytes, &budget).unwrap();
        assert!(
            Party::import_material(s, ctx, role, expected_nonce, bad, &mut guard, &budget).is_err()
        );
        assert!(matches!(
            Party::import_material(s, ctx, role, expected_nonce, packet, &mut guard, &budget),
            Err(Error::Consumed)
        ));
    }
    drop(guard);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn all_allocations_admitted_before_entropy_and_export_lifetimes() {
    for fail in 0..4 {
        let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
        let mut calls = 0;
        let result = issue_with(spec(5), context(), &budget, |bytes| {
            let current = calls;
            calls += 1;
            if current == fail {
                Err(PrimitiveError::Randomness)
            } else {
                getrandom::fill(bytes).map_err(|_| PrimitiveError::Randomness)
            }
        });
        assert!(matches!(result, Err(Error::Randomness)));
        assert_eq!(budget.used_bytes(), 0);
    }
    for v in [1, 5] {
        let mut success = false;
        for fail in 0..16 {
            let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
            budget.fail_after.store(fail, Ordering::SeqCst);
            let mut entropy = 0;
            let result = issue_with(spec(v), context(), &budget, |bytes| {
                entropy += 1;
                getrandom::fill(bytes).map_err(|_| PrimitiveError::Randomness)
            });
            match result {
                Ok(p) => {
                    success = true;
                    drop(p);
                }
                Err(e) => {
                    assert!(matches!(e, Error::Allocation));
                    assert_eq!(entropy, 0);
                }
            }
            assert_eq!(budget.used_bytes(), 0);
            if success {
                break;
            }
        }
        assert!(success);
    }
    let cost = resources(spec(5)).unwrap();
    let short = Budget::new(cost.combined_reservation_bytes - 1).unwrap();
    let mut entropy = false;
    assert!(matches!(
        issue_with(spec(5), context(), &short, |_| {
            entropy = true;
            Ok(())
        }),
        Err(Error::Resource)
    ));
    assert!(!entropy);
    assert_eq!(short.used_bytes(), 0);
    let budget = Budget::new(cost.combined_reservation_bytes).unwrap();
    let [mut a, mut b] = issue(spec(5), context(), &budget).unwrap();
    let af = a.start(0).unwrap();
    let bf = b.start(0).unwrap();
    let ao = a.finish(bf.as_bytes()).unwrap();
    let bo = b.finish(af.as_bytes()).unwrap();
    drop((a, b));
    assert_eq!(
        budget.used_bytes(),
        2 * cost.output_bytes_per_party + cost.frame_bytes_total
    );
    drop((ao, bo, af, bf));
    assert_eq!(budget.used_bytes(), 0);
    let [a, b] = issue(spec(5), context(), &budget).unwrap();
    let packet = a.export_material().unwrap();
    drop(b);
    assert_eq!(budget.used_bytes(), cost.material_packet_bytes_per_party);
    drop(packet);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn fixed_costs_bound_large_vocab_without_legacy_cap_changes() {
    for (v, b, material, packet) in [(50257, 16, 448, 568), (151936, 18, 500, 620)] {
        let r = resources(spec(v)).unwrap();
        assert_eq!(r.input_bits, b);
        assert_eq!(r.private_material_bytes_per_party, material);
        assert_eq!(r.material_packet_bytes_per_party, packet);
        assert_eq!(r.frame_bytes_total, 176);
        assert_eq!(r.dcf_keys_per_party, 1);
        assert_eq!(r.public_table_bytes_per_party, 0);
        assert_eq!(r.triples_per_party, 0);
        assert_eq!(r.tree_expansions_per_party, (1 << b) - 1);
        assert!(r.tree_expansions_per_party < v * b as usize);
        assert!(r.combined_reservation_bytes < MAX_ALLOCATION_BYTES);
    }
    assert!(resources(spec(0)).is_err());
    assert!(resources(spec(MAX_VOCABULARY + 1)).is_err());
    let mut bad = spec(5);
    bad.source_digest = [0; 32];
    assert!(resources(bad).is_err());
}
