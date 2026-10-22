use super::*;

fn vector(g: Geometry, token: usize, salt: usize) -> Vec<i64> {
    (0..g.width())
        .map(|i| ((token * 7 + i * 3 + salt) % 19) as i64 - 9)
        .collect()
}

// Independent forward physical-slot oracle: never calls production addresses,
// masks, score indexing, or its inverse token -> V ciphertext map.
fn oracle_slots(g: Geometry, dense: &[Vec<i64>], value: bool) -> Vec<i64> {
    let s = g.slots();
    let h = g.heads();
    let d = g.head_dim();
    let t = s / (h * d);
    let l = s / h;
    let cts = if value {
        g.capacity().div_ceil(l) * d
    } else {
        g.capacity().div_ceil(t)
    };
    let mut result = vec![0; cts * s];
    for ct in 0..cts {
        for slot in 0..s {
            let head = (slot / t) % h;
            let feature = slot / (t * h);
            let token = if value {
                ct / d * l + (feature * t + slot % t + (ct % d) * t) % l
            } else {
                ct * t + slot % t
            };
            if let Some(row) = dense.get(token) {
                result[ct * s + slot] = row[head * d + feature];
            }
        }
    }
    result
}

fn exercise(g: Geometry) {
    let mut keys = vec![0; g.key_layout().storage_words().unwrap()];
    let mut values = vec![0; g.value_layout().storage_words().unwrap()];
    let mut cache = RetainedCache::new(g, &mut keys, &mut values).unwrap();
    let mut dense_k = Vec::new();
    let mut dense_v = Vec::new();
    let mut previous_k = cache.derived_view().key_slots().to_vec();
    let mut previous_v = cache.derived_view().value_slots().to_vec();
    let mut old_k_mask = vec![0; previous_k.len()];
    let mut old_v_mask = vec![0; previous_v.len()];
    for n in 0..=g.capacity() {
        if n > 0 {
            let plan = cache.append_descriptor().unwrap();
            assert_eq!(plan.token, n - 1);
            assert_eq!(plan.key_tile, (n - 1) / g.tokens_per_key_tile());
            assert_eq!(plan.key_lane, (n - 1) % g.tokens_per_key_tile());
            assert_eq!(
                plan.new_key_ciphertexts,
                usize::from((n - 1) % g.tokens_per_key_tile() == 0)
            );
            assert_eq!(plan.value_group, (n - 1) / g.tokens_per_value_group());
            assert_eq!(
                plan.new_value_ciphertexts,
                if (n - 1) % g.tokens_per_value_group() == 0 {
                    g.head_dim()
                } else {
                    0
                }
            );
            let k = vector(g, n - 1, 1);
            let v = vector(g, n - 1, 4);
            cache.append(&k, &v).unwrap();
            dense_k.push(k);
            dense_v.push(v);
        }
        let view = cache.derived_view();
        let before = (view.key_slots().to_vec(), view.value_slots().to_vec());
        assert_eq!(view.key_slots(), oracle_slots(g, &dense_k, false));
        assert_eq!(view.value_slots(), oracle_slots(g, &dense_v, true));
        for (old, now, mask) in [
            (&previous_k, view.key_slots(), &old_k_mask),
            (&previous_v, view.value_slots(), &old_v_mask),
        ] {
            for i in 0..old.len() {
                if mask[i] != 0 {
                    assert_eq!(old[i], now[i], "old token moved at {i}");
                }
            }
        }
        let desc = view.descriptor();
        assert_eq!(desc.retained.active_tokens, n);
        assert_eq!(
            desc.retained.live_key_ciphertexts,
            n.div_ceil(g.tokens_per_key_tile())
        );
        assert_eq!(
            desc.retained.live_value_ciphertexts,
            n.div_ceil(g.tokens_per_value_group()) * g.head_dim()
        );
        let mut km = vec![9; before.0.len()];
        let mut vm = vec![9; before.1.len()];
        g.key_layout().validity_mask(n, &mut km).unwrap();
        g.value_layout().validity_mask(n, &mut vm).unwrap();
        // All-one logical input makes independent forward oracle a validity mask.
        let ones = vec![vec![1; g.width()]; n];
        assert_eq!(
            km,
            oracle_slots(g, &ones, false)
                .iter()
                .map(|&x| x as u8)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            vm,
            oracle_slots(g, &ones, true)
                .iter()
                .map(|&x| x as u8)
                .collect::<Vec<_>>()
        );
        for (slots, mask) in [(view.key_slots(), &km), (view.value_slots(), &vm)] {
            for (&x, &m) in slots.iter().zip(mask) {
                if m == 0 {
                    assert_eq!(x, 0);
                }
            }
        }
        let q = vector(g, 3, 8);
        let mut scores = vec![99; g.score_words()];
        view.query_key(&q, &mut scores).unwrap();
        let mut expected_scores = vec![0; scores.len()];
        let t = g.slots() / g.width();
        let l = g.slots() / g.heads();
        let mut weights = vec![MAX_ABS_INPUT; scores.len()]; // poison all inactive lanes
        let mut dense_weights = vec![vec![0; g.heads()]; n];
        for token in 0..n {
            for h in 0..g.heads() {
                let at = token / l * g.slots() + ((token % l) / t * g.heads() + h) * t + token % t;
                expected_scores[at] = (0..g.head_dim())
                    .map(|f| q[h * g.head_dim() + f] * dense_k[token][h * g.head_dim() + f])
                    .sum();
                dense_weights[token][h] = (token as i64 + h as i64) % 7 - 3;
                weights[at] = dense_weights[token][h];
            }
        }
        assert_eq!(scores, expected_scores);
        let mut out = vec![99; g.slots()];
        view.attention_value(&weights, &mut out).unwrap();
        let mut expected = vec![0; g.slots()];
        for f in 0..g.head_dim() {
            for h in 0..g.heads() {
                expected[(f * g.heads() + h) * t] = (0..n)
                    .map(|token| dense_weights[token][h] * dense_v[token][h * g.head_dim() + f])
                    .sum();
            }
        }
        assert_eq!(out, expected);
        assert_eq!(view.key_slots(), before.0);
        assert_eq!(view.value_slots(), before.1);
        previous_k = before.0;
        previous_v = before.1;
        old_k_mask = km;
        old_v_mask = vm;
    }
    let before = (
        cache.descriptor(),
        cache.derived_view().key_slots().to_vec(),
        cache.derived_view().value_slots().to_vec(),
    );
    assert_eq!(
        cache.append(&vec![1; g.width()], &vec![2; g.width()]),
        Err(LayoutError::Capacity)
    );
    assert_eq!(cache.descriptor(), before.0);
    assert_eq!(cache.derived_view().key_slots(), before.1);
    assert_eq!(cache.derived_view().value_slots(), before.2);
    assert_eq!(cache.append_descriptor(), Err(LayoutError::Capacity));
}

#[test]
fn dense_integer_oracles_cover_both_boundaries_multihead_and_old_tokens() {
    // Every prefix includes t-1/t/t+1 and S/H-1/boundary/+1; two V groups
    // also test token wraparound, partial final groups, and cross-group merging.
    for (s, h, d, capacity) in [
        (8, 1, 4, 17),
        (16, 2, 4, 17),
        (32, 4, 2, 17),
        (8, 2, 1, 9),
        (8, 2, 4, 9), // t=1 with multiple features: full-prefix physical/read oracles.
        (1, 1, 1, 3),
    ] {
        exercise(Geometry::new(s, h, d, capacity).unwrap());
    }
}

#[test]
fn figure_five_and_six_have_different_storage() {
    let g = Geometry::new(8, 1, 4, 8).unwrap();
    let k = g.key_layout();
    let v = g.value_layout();
    assert_eq!(
        k.address(1, 0, 0).unwrap(),
        Address {
            ciphertext: 0,
            slot: 1
        }
    );
    assert_eq!(
        v.address(2, 0, 0).unwrap(),
        Address {
            ciphertext: 1,
            slot: 0
        }
    );
    assert_eq!(
        v.address(0, 0, 1).unwrap(),
        Address {
            ciphertext: 3,
            slot: 2
        }
    );
    let rows = (0..5)
        .map(|token| (0..4).map(|f| (10 * token + f) as i64).collect())
        .collect::<Vec<Vec<i64>>>();
    assert_eq!(
        &oracle_slots(g, &rows, false)[..8],
        &[0, 10, 1, 11, 2, 12, 3, 13]
    );
    assert_eq!(
        &oracle_slots(g, &rows, true)[..8],
        &[0, 10, 21, 31, 42, 0, 0, 0]
    );
    assert_ne!(oracle_slots(g, &rows, false), oracle_slots(g, &rows, true));
}

#[test]
fn source_llama_geometry_addresses_counts_without_materializing_slots() {
    let g = Geometry::new(32768, 32, 128, 1025).unwrap();
    assert_eq!(g.width(), 4096);
    assert_eq!(g.tokens_per_key_tile(), 8);
    assert_eq!(g.tokens_per_value_group(), 1024);
    for (n, kc, vc) in [
        (7, 1, 128),
        (8, 1, 128),
        (9, 2, 128),
        (1023, 128, 128),
        (1024, 128, 128),
        (1025, 129, 256),
    ] {
        assert_eq!(g.key_layout().ciphertexts(n).unwrap(), kc);
        assert_eq!(g.value_layout().ciphertexts(n).unwrap(), vc);
    }
    for token in [0, 7, 8, 1023, 1024] {
        for head in [0, 1, 31] {
            for feature in [0, 1, 127] {
                let a = g.key_layout().address(token, head, feature).unwrap();
                assert_eq!(a.ciphertext, token / 8);
                assert_eq!(a.slot, (feature * 32 + head) * 8 + token % 8);
                let b = g.value_layout().address(token, head, feature).unwrap();
                assert_eq!(b.ciphertext / 128, token / 1024);
                let forward_token = (feature * 8 + b.slot % 8 + b.ciphertext % 128 * 8) % 1024;
                assert_eq!(forward_token, token % 1024);
                assert!(b.slot < 32768);
            }
        }
    }
    assert_eq!(g.key_layout().storage_words().unwrap(), 129 * 32768);
    assert_eq!(g.value_layout().storage_words().unwrap(), 256 * 32768);
}

#[test]
fn rejection_is_atomic_for_buffers_ranges_indices_capacity_and_masks() {
    let g = Geometry::new(8, 2, 2, 5).unwrap();
    let mut k = vec![0; g.key_layout().storage_words().unwrap()];
    let mut v = vec![0; g.value_layout().storage_words().unwrap()];
    assert!(matches!(
        RetainedCache::new(g, &mut k[..1], &mut v),
        Err(LayoutError::BufferLength)
    ));
    k[0] = 1;
    assert!(matches!(
        RetainedCache::new(g, &mut k, &mut v),
        Err(LayoutError::NonZeroStorage)
    ));
    assert_eq!(k[0], 1);
    k[0] = 0;
    let mut cache = RetainedCache::new(g, &mut k, &mut v).unwrap();
    cache.append(&[1; 4], &[2; 4]).unwrap();
    let before = (
        cache.descriptor(),
        cache.derived_view().key_slots().to_vec(),
        cache.derived_view().value_slots().to_vec(),
    );
    for (key, value, error) in [
        (vec![1; 3], vec![2; 4], LayoutError::BufferLength),
        (vec![1; 4], vec![2; 5], LayoutError::BufferLength),
        (vec![i64::MIN; 4], vec![2; 4], LayoutError::InputRange),
        (
            vec![1; 4],
            vec![MAX_ABS_INPUT + 1; 4],
            LayoutError::InputRange,
        ),
    ] {
        assert_eq!(cache.append(&key, &value), Err(error));
        assert_eq!(cache.descriptor(), before.0);
        assert_eq!(cache.derived_view().key_slots(), before.1);
        assert_eq!(cache.derived_view().value_slots(), before.2);
    }
    let view = cache.derived_view();
    let mut scores = vec![71; g.score_words()];
    assert_eq!(
        view.query_key(&[1; 3], &mut scores),
        Err(LayoutError::BufferLength)
    );
    assert_eq!(
        view.query_key(&[i64::MAX; 4], &mut scores),
        Err(LayoutError::InputRange)
    );
    assert!(scores.iter().all(|&x| x == 71));
    assert_eq!(
        view.query_key(&[1; 4], &mut scores[..1]),
        Err(LayoutError::BufferLength)
    );
    let mut out = vec![73; g.slots()];
    assert_eq!(
        view.attention_value(&scores[..1], &mut out),
        Err(LayoutError::BufferLength)
    );
    scores[0] = i64::MIN;
    assert_eq!(
        view.attention_value(&scores, &mut out),
        Err(LayoutError::InputRange)
    );
    assert!(out.iter().all(|&x| x == 73));
    assert_eq!(
        view.attention_value(&vec![0; g.score_words()], &mut out[..1]),
        Err(LayoutError::BufferLength)
    );
    let mut km = vec![7; g.key_layout().storage_words().unwrap()];
    let mut vm = vec![7; g.value_layout().storage_words().unwrap()];
    assert_eq!(
        g.key_layout().validity_mask(6, &mut km),
        Err(LayoutError::Capacity)
    );
    assert_eq!(
        g.value_layout().validity_mask(6, &mut vm),
        Err(LayoutError::Capacity)
    );
    assert_eq!(
        g.key_layout().validity_mask(1, &mut km[..1]),
        Err(LayoutError::BufferLength)
    );
    assert_eq!(
        g.value_layout().validity_mask(1, &mut vm[..1]),
        Err(LayoutError::BufferLength)
    );
    assert!(km.iter().chain(&vm).all(|&x| x == 7));
    assert_eq!(g.key_layout().address(5, 0, 0), Err(LayoutError::Index));
    assert_eq!(g.value_layout().address(0, 2, 0), Err(LayoutError::Index));
    assert_eq!(g.value_layout().address(0, 0, 2), Err(LayoutError::Index));
    assert_eq!(g.score_address(usize::MAX, 0), Err(LayoutError::Index));
    assert_eq!(g.output_slot(0, usize::MAX), Err(LayoutError::Index));
}

#[test]
fn geometry_rejects_overflow_nondivisibility_and_unbounded_work() {
    assert_eq!(
        Geometry::new(8, usize::MAX, 2, 1),
        Err(LayoutError::Overflow)
    );
    for args in [
        (0, 1, 1, 1),
        (8, 0, 1, 1),
        (8, 1, 0, 1),
        (8, 1, 3, 1),
        (8, 3, 2, 1),
        (8, 4, 4, 1),
        (8, 1, 1, 0),
        (12, 1, 4, 1),
    ] {
        assert_eq!(
            Geometry::new(args.0, args.1, args.2, args.3),
            Err(LayoutError::Geometry)
        );
    }
    for args in [
        (65536, 1, 1, 1),
        (8, 1, 1, usize::MAX),
        (32768, 1, 32768, 1),
        (32768, 32, 128, 32768),
    ] {
        assert_eq!(
            Geometry::new(args.0, args.1, args.2, args.3),
            Err(LayoutError::ResourceLimit)
        );
    }
}

#[test]
fn exact_domain_extremes_and_empty_reads() {
    let g = Geometry::new(8, 1, 4, 2).unwrap();
    let mut k = vec![0; g.key_layout().storage_words().unwrap()];
    let mut v = vec![0; g.value_layout().storage_words().unwrap()];
    let mut cache = RetainedCache::new(g, &mut k, &mut v).unwrap();
    let mut scores = vec![9; g.score_words()];
    let mut output = vec![9; g.slots()];
    cache
        .derived_view()
        .query_key(&[MAX_ABS_INPUT; 4], &mut scores)
        .unwrap();
    cache
        .derived_view()
        .attention_value(&vec![MAX_ABS_INPUT; g.score_words()], &mut output)
        .unwrap();
    assert!(scores.iter().chain(&output).all(|&x| x == 0));
    for _ in 0..2 {
        cache
            .append(&[-MAX_ABS_INPUT; 4], &[MAX_ABS_INPUT; 4])
            .unwrap();
    }
    cache
        .derived_view()
        .query_key(&[MAX_ABS_INPUT; 4], &mut scores)
        .unwrap();
    assert_eq!(scores[0], -4 * MAX_ABS_INPUT * MAX_ABS_INPUT);
    cache
        .derived_view()
        .attention_value(&vec![-MAX_ABS_INPUT; g.score_words()], &mut output)
        .unwrap();
    for f in 0..4 {
        assert_eq!(output[f * 2], -2 * MAX_ABS_INPUT * MAX_ABS_INPUT);
        assert_eq!(output[f * 2 + 1], 0);
    }
}
