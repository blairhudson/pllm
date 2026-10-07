use super::*;

fn context() -> Context {
    Context {
        plan: [1; 32],
        source: [2; 32],
        state_contract: [3; 32],
        layer: 4,
    }
}

fn fixture(rows: usize, width: usize) -> SelectionOwner {
    let boundary = [
        0.0,
        -0.0,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        f32::MAX,
        -f32::MAX,
        -1.0,
        0.125,
    ];
    let keys: Vec<_> = (0..2 * rows * width)
        .map(|i| boundary[i % boundary.len()])
        .collect();
    let values: Vec<_> = (0..keys.len())
        .map(|i| boundary[(i + 3) % boundary.len()])
        .collect();
    let positions: Vec<_> = (0..rows).map(|i| 2 * i).collect();
    SelectionOwner::new(
        context(),
        RetainedKvCache::from_prefill(&keys, &values, width, &positions, rows + 2, 2 * rows + 2)
            .unwrap(),
    )
    .unwrap()
}

fn oracle(scores: &[f32], policy: Policy) -> Vec<usize> {
    let split = scores.len() - policy.recent;
    let mut eligible: Vec<_> = (0..split).collect();
    eligible.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap().then(a.cmp(&b)));
    eligible.truncate(policy.keep - policy.recent);
    eligible.extend(split..scores.len());
    eligible.sort_unstable();
    eligible
}

#[test]
fn stable_top_k_and_complete_kv_bits_match_independent_sort() {
    for layout in [Layout::FullRecords, Layout::CompactIndices] {
        check_top_k(layout);
    }
}

fn check_top_k(layout: Layout) {
    let mut checked = 0;
    for rows in [1_usize, 2, 3, 5, 8, 16] {
        for keep in [1, rows.div_ceil(2), rows] {
            for recent in [0, keep.min(2)] {
                for seed in 0..3 {
                    let mut owner = fixture(rows, 1);
                    let scores: Vec<_> = (0..rows)
                        .map(|i| match seed {
                            0 => [0.0, -0.0, 1.0, 1.0][i % 4],
                            1 => [f32::from_bits(1), 1.0, f32::MAX, f32::MIN_POSITIVE][i % 4],
                            _ => ((i * 71 + rows * 43) % 101) as f32 / 101.0,
                        })
                        .collect();
                    let policy = Policy { keep, recent };
                    let selected = oracle(&scores, policy);
                    let (k, v, p) = owner.active().unwrap();
                    let expected: Vec<_> = selected
                        .iter()
                        .map(|&i| (k[i].to_bits(), v[i].to_bits(), p[i]))
                        .collect();
                    let (client, evaluator) =
                        owner.issue_with_layout(&scores, policy, layout).unwrap();
                    let binding = evaluator.binding();
                    let resources = evaluator.resources();
                    let cost = project(rows, 1, policy, owner.allocated_bytes(), layout).unwrap();
                    assert_eq!(resources.wires, cost.actual_wires);
                    assert_eq!(resources.and_gates, cost.and_gates);
                    assert_eq!(resources.constant_label_bytes, cost.constant_label_bytes);
                    assert_eq!(
                        resources.public_metadata_bytes_estimate,
                        cost.public_metadata_bytes_estimate
                    );
                    assert!(resources.wires <= resources.estimate.wire_bound);
                    assert!(
                        resources.estimate.combined_allocation_bound_bytes <= MAX_ALLOCATION_BYTES
                    );
                    client
                        .finish(&mut owner, evaluator.evaluate(binding).unwrap())
                        .unwrap();
                    let (k, v, p) = owner.active().unwrap();
                    assert_eq!(
                        k.iter()
                            .zip(v)
                            .zip(p)
                            .map(|((&a, &b), &p)| (a.to_bits(), b.to_bits(), p))
                            .collect::<Vec<_>>(),
                        expected
                    );
                    assert_eq!(owner.allocated_bytes(), (keep + 2) * 12);
                    assert_eq!(owner.cache.next_position(), 2 * rows);
                    assert_eq!(owner.revision, 1);
                    checked += keep;
                }
            }
        }
    }
    assert_eq!(checked, 360);
}

#[test]
fn indexed_full_width_parity_fixed_schedule_and_complete_costs() {
    for (rows, width, keep, recent) in [
        (2, 128, 1, 0),
        (4, 128, 2, 1),
        (16, 2, 15, 1),
        (32, 1, 1, 0),
    ] {
        let policy = Policy { keep, recent };
        let mut schedule = None;
        for variant in 0..3 {
            let mut owner = fixture(rows, width);
            let scores: Vec<_> = (0..rows)
                .map(|i| {
                    if variant == 0 {
                        1.0
                    } else {
                        ((i * 7 + variant * 3) % rows) as f32
                    }
                })
                .collect();
            let selected = oracle(&scores, policy);
            let (k, v, p) = owner.active().unwrap();
            let expected_p: Vec<_> = selected.iter().map(|&i| p[i]).collect();
            let expected_k: Vec<_> = selected
                .iter()
                .flat_map(|&i| k[i * width..(i + 1) * width].iter().map(|x| x.to_bits()))
                .collect();
            let expected_v: Vec<_> = selected
                .iter()
                .flat_map(|&i| v[i * width..(i + 1) * width].iter().map(|x| x.to_bits()))
                .collect();
            let cost = project(
                rows,
                width,
                policy,
                owner.allocated_bytes(),
                Layout::CompactIndices,
            )
            .unwrap();
            assert!(cost.admitted());
            let (client, evaluator) = owner
                .issue_with_layout(&scores, policy, Layout::CompactIndices)
                .unwrap();
            let r = evaluator.resources();
            assert_eq!(r.wires, cost.actual_wires);
            assert_eq!(r.and_gates, cost.and_gates);
            assert_eq!(r.constant_label_bytes, cost.constant_label_bytes);
            assert_eq!(
                r.public_metadata_bytes_estimate,
                cost.public_metadata_bytes_estimate
            );
            if let Some(old) = schedule {
                assert_eq!(r.public_schedule_digest, old);
            }
            schedule = Some(r.public_schedule_digest);
            let binding = evaluator.binding();
            client
                .finish(&mut owner, evaluator.evaluate(binding).unwrap())
                .unwrap();
            let (k, v, p) = owner.active().unwrap();
            assert_eq!(p, expected_p);
            assert_eq!(
                k.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                expected_k
            );
            assert_eq!(
                v.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                expected_v
            );
        }
    }
}

#[test]
fn compact_cost_cannot_authorize_large_state_or_underprice_boundaries() {
    for (rows, width) in [(16, 128), (256, 128), (512, 128), (512, 8192)] {
        let policy = Policy {
            keep: rows / 2,
            recent: 1,
        };
        for layout in [Layout::FullRecords, Layout::CompactIndices] {
            let cost = project(rows, width, policy, 0, layout).unwrap();
            assert!(!cost.admitted());
            assert_eq!(cost.estimate.input_bits, rows * (64 + 64 * width));
            assert_eq!(cost.estimate.output_bits, policy.keep * (32 + 64 * width));
            assert!(estimate_layout(rows, width, policy, 0, layout).is_err());
        }
    }
    let mut owner = fixture(16, 128);
    assert!(owner
        .issue_with_layout(
            &[1.0; 16],
            Policy { keep: 8, recent: 1 },
            Layout::CompactIndices
        )
        .is_err());
    assert!(owner.pending.is_none());
    assert_eq!(owner.revision, 0);
    for (rows, width, state) in [(usize::MAX, 1, 0), (1, usize::MAX, 0), (1, 1, usize::MAX)] {
        assert!(project(
            rows,
            width,
            Policy { keep: 1, recent: 0 },
            state,
            Layout::CompactIndices
        )
        .is_err());
    }
}

#[test]
fn compact_layout_substitution_and_stale_receipts_burn() {
    let mut owner = fixture(4, 2);
    let policy = Policy { keep: 2, recent: 1 };
    let scores = [1.0, 3.0, 2.0, 0.0];
    let (client, evaluator) = owner
        .issue_with_layout(&scores, policy, Layout::CompactIndices)
        .unwrap();
    let mut wrong = evaluator.binding();
    wrong.layout = Layout::FullRecords;
    assert!(evaluator.evaluate(wrong).is_err());
    drop(client);
    owner.cancel();
    for change in 0..3 {
        let (client, evaluator) = owner
            .issue_with_layout(&scores, policy, Layout::CompactIndices)
            .unwrap();
        let binding = evaluator.binding();
        let mut output = evaluator.evaluate(binding).unwrap();
        match change {
            0 => output.binding.layout = Layout::FullRecords,
            1 => owner.cancel(),
            _ => owner.append(8, &[0.0, -0.0], &[1.0, -1.0]).unwrap(),
        }
        assert!(client.finish(&mut owner, output).is_err());
        assert!(owner.pending.is_none());
    }
}

#[test]
fn changed_private_queries_keep_fixed_structure_and_oblivious_addresses() {
    let policy = Policy { keep: 3, recent: 1 };
    let mut a = fixture(8, 4);
    let mut b = fixture(8, 4);
    let scores_a = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    let scores_b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    // Explicit negative control: publishing selected addresses distinguishes queries.
    assert_ne!(oracle(&scores_a, policy), oracle(&scores_b, policy));
    let (ca, ea) = a.issue(&scores_a, policy).unwrap();
    let (cb, eb) = b.issue(&scores_b, policy).unwrap();
    assert_eq!(ea.resources(), eb.resources());
    assert_ne!(ea.binding().issuance, eb.binding().issuance);
    // Public instruction schedule, including input/output wire positions, is
    // independent of score/KV values. Ciphertexts/labels are freshly randomized.
    assert_eq!(
        ea.program.public_schedule_digest(),
        eb.program.public_schedule_digest()
    );
    let ba = ea.binding();
    let bb = eb.binding();
    ca.finish(&mut a, ea.evaluate(ba).unwrap()).unwrap();
    cb.finish(&mut b, eb.evaluate(bb).unwrap()).unwrap();
    assert_ne!(a.active().unwrap().2, b.active().unwrap().2);
}

#[test]
fn context_substitution_consumes_evaluator_before_execution() {
    for field in 0..13 {
        let mut owner = fixture(3, 1);
        let (client, evaluator) = owner
            .issue(&[1.0, 3.0, 2.0], Policy { keep: 2, recent: 1 })
            .unwrap();
        let mut wrong = evaluator.binding();
        match field {
            0 => wrong.context.plan[0] ^= 1,
            1 => wrong.context.source[0] ^= 1,
            2 => wrong.context.state_contract[0] ^= 1,
            3 => wrong.context.layer += 1,
            4 => wrong.owner[0] ^= 1,
            5 => wrong.revision += 1,
            6 => wrong.issuance[0] ^= 1,
            7 => wrong.rows += 1,
            8 => wrong.width += 1,
            9 => wrong.next_position += 1,
            10 => wrong.policy.keep += 1,
            11 => wrong.policy.recent += 1,
            _ => wrong.context.layer = u32::MAX,
        }
        assert!(evaluator.evaluate(wrong).is_err());
        drop(client);
        owner.cancel();
        assert_eq!(owner.active().unwrap().2, &[0, 2, 4]);
    }
}

#[test]
fn state_append_cancel_close_and_old_outputs_cannot_commit() {
    for change in 0..3 {
        let mut owner = fixture(3, 1);
        let (client, evaluator) = owner
            .issue(&[1.0, 3.0, 2.0], Policy { keep: 2, recent: 0 })
            .unwrap();
        assert!(owner
            .issue(&[1.0; 3], Policy { keep: 1, recent: 0 })
            .is_err());
        let binding = evaluator.binding();
        let output = evaluator.evaluate(binding).unwrap();
        match change {
            0 => owner.append(6, &[0.0], &[-0.0]).unwrap(),
            1 => owner.cancel(),
            _ => owner.close(),
        }
        assert!(client.finish(&mut owner, output).is_err());
        if change == 2 {
            assert_eq!(owner.allocated_bytes(), 0);
            assert!(owner.active().is_err());
        } else {
            assert_eq!(owner.cache.len(), if change == 0 { 4 } else { 3 });
        }
    }
}

#[test]
fn wrong_owner_and_output_substitution_burn_only_the_owned_query() {
    let policy = Policy { keep: 2, recent: 0 };
    let mut a = fixture(3, 1);
    let mut b = fixture(3, 1);
    let (ca, ea) = a.issue(&[1.0; 3], policy).unwrap();
    let (cb, eb) = b.issue(&[1.0; 3], policy).unwrap();
    let ba = ea.binding();
    let bb = eb.binding();
    assert!(ca.finish(&mut b, ea.evaluate(ba).unwrap()).is_err());
    assert_eq!(b.pending, Some(bb.issuance));
    cb.finish(&mut b, eb.evaluate(bb).unwrap()).unwrap();
    a.cancel();

    for forged_binding in [false, true] {
        let (old_client, old_evaluator) = a.issue(&[1.0; 3], policy).unwrap();
        let old_binding = old_evaluator.binding();
        let mut old_output = old_evaluator.evaluate(old_binding).unwrap();
        a.cancel();
        drop(old_client);
        let (new_client, new_evaluator) = a.issue(&[2.0; 3], policy).unwrap();
        assert_ne!(old_binding.issuance, new_evaluator.binding().issuance);
        if forged_binding {
            old_output.binding = new_evaluator.binding();
        }
        assert!(new_client.finish(&mut a, old_output).is_err());
        assert!(a.pending.is_none());
        assert_eq!(a.cache.len(), 3);
        drop(new_evaluator);
    }
}

#[test]
fn preflight_and_invalid_scores_leave_no_material_or_mutation() {
    let mut owner = fixture(3, 1);
    for scores in [
        vec![1.0; 2],
        vec![-1.0; 3],
        vec![f32::NAN; 3],
        vec![f32::INFINITY; 3],
    ] {
        assert!(owner.issue(&scores, Policy { keep: 1, recent: 0 }).is_err());
        assert!(owner.pending.is_none());
    }
    for policy in [
        Policy { keep: 0, recent: 0 },
        Policy { keep: 4, recent: 0 },
        Policy { keep: 1, recent: 2 },
    ] {
        assert!(owner.issue(&[1.0; 3], policy).is_err());
    }
    let mut wide = fixture(16, 16);
    assert!(wide
        .issue(&[1.0; 16], Policy { keep: 8, recent: 2 })
        .err()
        .unwrap()
        .contains("allocation"));
    assert!(wide.pending.is_none());
    assert!(estimate(usize::MAX, 1, Policy { keep: 1, recent: 0 }, 0).is_err());
    assert!(estimate(1, 1, Policy { keep: 1, recent: 0 }, usize::MAX).is_err());
    assert_eq!(owner.revision, 0);
}

#[test]
fn failed_append_is_atomic_and_valid_receipt_remains_usable() {
    let mut owner = fixture(3, 1);
    let (client, evaluator) = owner
        .issue(&[1.0, 2.0, 3.0], Policy { keep: 2, recent: 0 })
        .unwrap();
    let binding = evaluator.binding();
    assert!(owner.append(5, &[0.0], &[0.0]).is_err());
    assert!(owner.append(6, &[f32::NAN], &[0.0]).is_err());
    client
        .finish(&mut owner, evaluator.evaluate(binding).unwrap())
        .unwrap();
    assert_eq!(owner.active().unwrap().2, &[2, 4]);
    owner.append(6, &[-0.0], &[0.0]).unwrap();
    assert_eq!(owner.active().unwrap().2, &[2, 4, 6]);
}
