use super::*;

#[test]
fn exported_allocation_capacity_cannot_exceed_its_lease() {
    let too_large = Vec::<u64>::with_capacity(16);
    assert!(matches!(exact_capacity(too_large, 8), Err(Error::Resource)));
    let too_small = Vec::<u64>::with_capacity(4);
    assert!(matches!(exact_capacity(too_small, 8), Err(Error::Resource)));
    let budget = Budget::new(1024).unwrap();
    for count in [0, 1, 4, 80, 96] {
        let values = allocate::<u8>(count, &budget).unwrap();
        assert_eq!(values.capacity(), count);
    }
    assert_eq!(budget.used_bytes(), 0);
}

fn context() -> Context {
    Context {
        plan: [1; 32],
        session: [2; 32],
        operation: [3; 32],
        first_tensor_index: 0,
    }
}
fn spec(bits: u8, m: usize, lanes: usize) -> Spec {
    Spec {
        bits,
        source_intervals: m,
        lanes,
        descriptor_digest: [4; 32],
        source_digest: [5; 32],
        boolean_outputs: 0,
        identity_phi: true,
    }
}
// Test-only rotation compiler, independent of production native compiler.
fn translated(bits: u8, r: u64, source: &[(u64, [u64; 2])]) -> DealerPartition {
    let n = 1u64 << bits;
    let mut starts: Vec<_> = source
        .iter()
        .map(|(b, p)| ((b + r) & (n - 1), *p))
        .collect();
    starts.sort_by_key(|v| v.0);
    if starts[0].0 != 0 {
        starts.insert(0, (0, starts.last().unwrap().1));
    } else if source.len() < n as usize {
        let i = (0..starts.len())
            .find(|&i| {
                let end = starts.get(i + 1).map_or(n, |s| s.0);
                end - starts[i].0 >= 2
            })
            .unwrap();
        starts.insert(i + 1, (starts[i].0 + 1, starts[i].1));
    }
    let mut b: Vec<_> = starts.iter().map(|s| s.0).collect();
    b.push(n);
    let p: Vec<_> = starts.iter().map(|s| s.1).collect();
    DealerPartition::from_slices(r, &b, &p).unwrap()
}
fn clear(bits: u8, source: &[(u64, [u64; 2])], x: u64) -> u64 {
    let p = source.iter().rev().find(|s| s.0 <= x).unwrap().1;
    p[1].wrapping_mul(x).wrapping_add(p[0]) & ((1u64 << bits) - 1)
}

fn public_table(bits: u8, source: &[(u64, [u64; 2])]) -> RingTable {
    let public = translated(bits, 0, source);
    RingTable::new(
        bits,
        &public.boundaries,
        &public.payloads,
        MAX_ALLOCATION_BYTES,
    )
    .unwrap()
}
fn execute<'a>(parties: &mut [Party<'a>; 2], x: &[u64]) -> [Output<'a>; 2] {
    let mask = parties[0].mask();
    let left: Vec<_> = x
        .iter()
        .enumerate()
        .map(|(i, _)| (mask - i as u64) & mask)
        .collect();
    let right: Vec<_> = x
        .iter()
        .zip(&left)
        .map(|(x, s)| x.wrapping_sub(*s) & mask)
        .collect();
    let a = parties[0].start(&left).unwrap();
    let b = parties[1].start(&right).unwrap();
    let aa = parties[0].advance(b.as_bytes()).unwrap();
    let bb = parties[1].advance(a.as_bytes()).unwrap();
    [
        parties[0].finish(bb.as_bytes()).unwrap(),
        parties[1].finish(aa.as_bytes()).unwrap(),
    ]
}

#[test]
fn independent_small_rings_every_mask_input_and_full_partition() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    for bits in 1..=4 {
        let n = 1u64 << bits;
        for source in [
            vec![(0, [n - 1, 1])],
            vec![(0, [n - 1, n - 1]), (n / 2, [1, 0])],
            (0..n)
                .map(|i| (i, [(3 * i + 1) & (n - 1), (i + 1) & (n - 1)]))
                .collect(),
        ] {
            for r in 0..n {
                for x in 0..n {
                    let s = spec(bits, source.len(), 1);
                    let mut p = issue_inner(
                        s,
                        context(),
                        &budget,
                        |_, r| Ok(translated(bits, r, &source)),
                        || Ok(r),
                    )
                    .unwrap();
                    let output = execute(&mut p, &[x]);
                    assert_eq!(
                        (output[0].shares()[0] + output[1].shares()[0]) & (n - 1),
                        clear(bits, &source, x),
                        "bits={bits}, mask={r}, x={x}"
                    );
                    assert!(p.iter().all(Party::is_consumed));
                }
            }
        }
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn ring50_endpoints_wrap_original_x_and_no_rescale() {
    let n = 1u64 << 50;
    let source = [
        (0, [n - 31, 1]),
        (64, [991, n - 1]),
        (n / 2, [n - 1234, 7]),
        (n - 64, [17, 0]),
    ];
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    for r in [0, 1, 64, n / 2, n - 1] {
        for inputs in [
            [0, 1, 63, 64],
            [65, n / 2 - 1, n / 2, n / 2 + 1],
            [n - 65, n - 64, n - 2, n - 1],
        ] {
            let mut p = issue_inner(
                spec(50, source.len(), 4),
                context(),
                &budget,
                |_, r| Ok(translated(50, r, &source)),
                || Ok(r),
            )
            .unwrap();
            let output = execute(&mut p, &inputs);
            for (i, x) in inputs.into_iter().enumerate() {
                assert_eq!(
                    (output[0].shares()[i] + output[1].shares()[i]) & (n - 1),
                    clear(50, &source, x)
                );
            }
        }
    }
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn padded_512_513_fixed_cost_and_private_party_structure() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    for m in [511, 512] {
        let source: Vec<_> = (0..m)
            .map(|i| (i as u64, [(17 * i + 3) as u64, (11 * i + 7) as u64]))
            .collect();
        let s = spec(50, m, 1);
        let r = resources(s).unwrap();
        assert_eq!(r.dcf_keys_per_party, m + 1);
        assert_eq!(r.party_key_logical_bytes, (m + 1) * 1732 + 48);
        assert_eq!(r.masked_frame_bytes, 88);
        assert_eq!(r.product_frame_bytes, 96);
        assert_eq!(r.total_peer_frame_bytes, 368);
        for mask in [0, (1u64 << 50) - 1] {
            let mut p = issue_inner(
                s,
                context(),
                &budget,
                |_, r| Ok(translated(50, r, &source)),
                || Ok(mask),
            )
            .unwrap();
            for party in &p {
                let Some(State::Ready(lanes)) = &party.state else {
                    panic!()
                };
                let Lookup::Private { keys, .. } = &lanes[0].lookup else {
                    panic!("private lookup must not carry a public-table owner")
                };
                assert_eq!(keys.len(), m + 1);
                // View has only vector keys, one mask SHARE, base SHARES and
                // triple SHARES. No descriptor/table/translated partition owner.
                assert_eq!(r.lookup_backend, LookupBackend::PrivateVector);
            }
            let output = execute(&mut p, &[(1u64 << 50) - 1]);
            assert_eq!(
                (output[0].shares()[0] + output[1].shares()[0]) & ((1u64 << 50) - 1),
                clear(50, &source, (1u64 << 50) - 1)
            );
        }
        println!("ring50 source={m}: {r:?}");
    }
    assert_eq!(budget.used_bytes(), 0);
    assert!(
        resources(spec(50, 511, MAX_LANES))
            .unwrap()
            .combined_reservation_bytes
            < MAX_ALLOCATION_BYTES
    );
}

#[test]
fn lifecycle_role_context_nonce_order_malformed_replay_and_cancel_burn() {
    for backend in [LookupBackend::PrivateVector, LookupBackend::PublicBoundary] {
        let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
        let s = spec(8, 2, 1);
        let source = [(0, [17, 3]), (128, [13, 7])];
        let table = public_table(8, &source);
        let issue_pair = || {
            match backend {
                LookupBackend::PrivateVector => {
                    issue(s, context(), &budget, |_, r| Ok(translated(8, r, &source)))
                }
                LookupBackend::PublicBoundary => issue_public(s, context(), &budget, &table),
            }
            .unwrap()
        };
        // Wrong phase/order before start burns ready keys.
        let mut p = issue_pair();
        assert!(matches!(p[0].finish(&[]), Err(Error::Context)));
        assert!(p[0].is_consumed());
        // Reflected role, wrong binding, nonce, phase, width, reserved byte, count,
        // noncanonical ring element, short and long frames all burn.
        for mutate in 0..11 {
            let mut p = issue_pair();
            let a = p[0].start(&[1]).unwrap();
            let b = p[1].start(&[2]).unwrap();
            let mut raw = b.as_bytes().to_vec();
            match mutate {
                0 => raw = a.as_bytes().to_vec(),
                1 => raw[8] ^= 1,
                2 => raw[40] ^= 1,
                3 => raw[73] ^= 1,
                4 => raw[72] ^= 1,
                5 => raw[75] = 1,
                6 => raw[76] ^= 1,
                7 => raw[81] = 1,
                8 => {
                    raw.pop();
                }
                9 => raw.push(0),
                _ => raw[0] ^= 1,
            }
            assert!(matches!(p[0].advance(&raw), Err(Error::Context)));
            assert!(p[0].is_consumed());
            assert!(matches!(p[0].advance(b.as_bytes()), Err(Error::Consumed)));
        }
        let mut p = issue_pair();
        let mut q = issue_pair();
        let _ = p[0].start(&[1]).unwrap();
        let foreign = q[1].start(&[2]).unwrap();
        assert!(matches!(
            p[0].advance(foreign.as_bytes()),
            Err(Error::Context)
        ));
        let mut p = issue_pair();
        p[0].cancel();
        assert!(matches!(p[0].start(&[1]), Err(Error::Consumed)));
        let mut p = issue_pair();
        assert!(matches!(p[0].start(&[256]), Err(Error::Domain)));
        assert!(p[0].is_consumed());
        let mut p = issue_pair();
        let _ = execute(&mut p, &[139]);
        assert!(matches!(p[0].finish(&[]), Err(Error::Consumed)));
    }
}

#[test]
fn preflight_rejection_callback_errors_and_allocation_failures_retire() {
    let s = spec(4, 2, 1);
    let source = [(0, [15, 1]), (8, [1, 15])];
    let r = resources(s).unwrap();
    let budget = Budget::new(r.combined_reservation_bytes - 1).unwrap();
    let mut called = false;
    assert!(matches!(
        issue(s, context(), &budget, |_, _| {
            called = true;
            Err(Error::Domain)
        }),
        Err(Error::Resource)
    ));
    assert!(!called);
    assert_eq!(budget.used_bytes(), 0);
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    for fail in 0..10 {
        budget.fail_after.store(fail, Ordering::SeqCst);
        let result = issue(s, context(), &budget, |_, r| Ok(translated(4, r, &source)));
        assert!(matches!(result, Err(Error::Allocation)), "failpoint {fail}");
        assert_eq!(budget.used_bytes(), 0);
    }
    budget.fail_after.store(-1, Ordering::SeqCst);
    let table = public_table(4, &source);
    for backend in [LookupBackend::PrivateVector, LookupBackend::PublicBoundary] {
        for (phase, fail) in [
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 0),
        ] {
            let mut p = match backend {
                LookupBackend::PrivateVector => {
                    issue(s, context(), &budget, |_, r| Ok(translated(4, r, &source)))
                }
                LookupBackend::PublicBoundary => issue_public(s, context(), &budget, &table),
            }
            .unwrap();
            if phase == 0 {
                budget.fail_after.store(fail, Ordering::SeqCst);
                assert!(matches!(p[0].start(&[1]), Err(Error::Allocation)));
            } else {
                let a = p[0].start(&[1]).unwrap();
                let b = p[1].start(&[2]).unwrap();
                if phase == 1 {
                    budget.fail_after.store(fail, Ordering::SeqCst);
                    assert!(matches!(p[0].advance(b.as_bytes()), Err(Error::Allocation)));
                } else {
                    let _aa = p[0].advance(b.as_bytes()).unwrap();
                    let bb = p[1].advance(a.as_bytes()).unwrap();
                    budget.fail_after.store(fail, Ordering::SeqCst);
                    assert!(matches!(p[0].finish(bb.as_bytes()), Err(Error::Allocation)));
                }
            }
            assert!(p[0].is_consumed());
            drop(p);
            assert_eq!(budget.used_bytes(), 0);
            budget.fail_after.store(-1, Ordering::SeqCst);
        }
    }
    assert!(matches!(
        issue(s, context(), &budget, |_, _| Err(Error::Unsupported)),
        Err(Error::Unsupported)
    ));
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn public_boundary_partial_issuance_failures_and_complete_admission_retire() {
    let source = [(0, [15, 1]), (8, [1, 15])];
    let table = public_table(4, &source);
    let s = spec(4, source.len(), MAX_LANES);
    let r = resources_for(s, Some(&table)).unwrap();
    let too_small = Budget::new(r.combined_reservation_bytes - 1).unwrap();
    assert!(matches!(
        issue_public(s, context(), &too_small, &table),
        Err(Error::Resource)
    ));
    assert_eq!(too_small.used_bytes(), 0);
    let budget = Budget::new(r.combined_reservation_bytes).unwrap();
    // Two lane-vector allocations plus each common scalar lookup call. The
    // underlying primitive separately checks every table/key allocation and RNG
    // failure; this checks retirement of earlier complete lanes on integration failure.
    for fail in 0..(2 + MAX_LANES) {
        budget.fail_after.store(fail as isize, Ordering::SeqCst);
        assert!(matches!(
            issue_public(s, context(), &budget, &table),
            Err(Error::Allocation)
        ));
        assert_eq!(budget.used_bytes(), 0);
    }
    budget.fail_after.store(-1, Ordering::SeqCst);
    drop(issue_public(s, context(), &budget, &table).unwrap());
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn product_malformed_and_reordered_methods_burn() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let source = [(0, [3, 1]), (8, [1, 3])];
    let issue_pair = || {
        issue(spec(4, 2, 1), context(), &budget, |_, r| {
            Ok(translated(4, r, &source))
        })
        .unwrap()
    };
    let mut p = issue_pair();
    let _ = p[0].start(&[1]).unwrap();
    assert!(matches!(p[0].start(&[1]), Err(Error::Context)));
    assert!(p[0].is_consumed());
    let mut p = issue_pair();
    let a = p[0].start(&[1]).unwrap();
    let b = p[1].start(&[2]).unwrap();
    let _aa = p[0].advance(b.as_bytes()).unwrap();
    let bb = p[1].advance(a.as_bytes()).unwrap();
    let mut corrupt = bb.as_bytes().to_vec();
    corrupt[40] ^= 1;
    assert!(matches!(p[0].finish(&corrupt), Err(Error::Context)));
    assert!(p[0].is_consumed());
    let mut p = issue_pair();
    let a = p[0].start(&[1]).unwrap();
    let b = p[1].start(&[2]).unwrap();
    let _aa = p[0].advance(b.as_bytes()).unwrap();
    assert!(matches!(p[0].advance(a.as_bytes()), Err(Error::Context)));
    assert!(p[0].is_consumed());
}

#[test]
fn exported_frames_outputs_hold_reservations_until_drop() {
    let spec = spec(4, 2, 1);
    let budget = Budget::new(resources(spec).unwrap().combined_reservation_bytes).unwrap();
    let source = [(0, [3, 1]), (8, [1, 3])];
    let mut p = issue(spec, context(), &budget, |_, r| {
        Ok(translated(4, r, &source))
    })
    .unwrap();
    let a = p[0].start(&[1]).unwrap();
    let b = p[1].start(&[2]).unwrap();
    let aa = p[0].advance(b.as_bytes()).unwrap();
    let bb = p[1].advance(a.as_bytes()).unwrap();
    let x = p[0].finish(bb.as_bytes()).unwrap();
    let y = p[1].finish(aa.as_bytes()).unwrap();
    drop(p);
    assert_eq!(budget.used_bytes(), 2 * (88 + 96 + 8));
    assert!(matches!(
        issue(spec, context(), &budget, |_, _| panic!(
            "must reject before compiler"
        )),
        Err(Error::Resource)
    ));
    drop(a);
    drop(b);
    drop(aa);
    drop(bb);
    assert_eq!(budget.used_bytes(), 16);
    assert!(matches!(
        issue(spec, context(), &budget, |_, _| panic!(
            "output lease must remain charged"
        )),
        Err(Error::Resource)
    ));
    drop(x);
    drop(y);
    assert_eq!(budget.used_bytes(), 0);
    drop(
        issue(spec, context(), &budget, |_, r| {
            Ok(translated(4, r, &source))
        })
        .unwrap(),
    );
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn reject_bad_geometry_wrong_original_mask_boolean_phi_and_capacity() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let s = spec(4, 2, 1);
    assert!(matches!(
        issue(s, context(), &budget, |_, r| DealerPartition::from_slices(
            r,
            &[0, 1, 1, 16],
            &[[0, 1]; 3]
        )),
        Err(Error::Domain)
    ));
    assert!(matches!(
        issue(s, context(), &budget, |_, r| DealerPartition::from_slices(
            r ^ 1,
            &[0, 1, 2, 16],
            &[[0, 1]; 3]
        )),
        Err(Error::Domain)
    ));
    for invalid in [
        Spec {
            boolean_outputs: 1,
            ..s
        },
        Spec {
            identity_phi: false,
            ..s
        },
    ] {
        assert!(matches!(resources(invalid), Err(Error::Unsupported)));
    }
    for invalid in [
        spec(0, 2, 1),
        spec(51, 2, 1),
        spec(4, 17, 1),
        spec(50, 513, 1),
        spec(50, 2, 5),
    ] {
        assert!(matches!(resources(invalid), Err(Error::Domain)));
    }
    assert_eq!(budget.used_bytes(), 0);
}
