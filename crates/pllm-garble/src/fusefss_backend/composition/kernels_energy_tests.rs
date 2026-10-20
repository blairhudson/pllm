use super::*;

fn energy(values: &[i64]) -> u128 {
    values
        .iter()
        .map(|&x| (i128::from(x) * i128::from(x)) as u128)
        .sum()
}

fn oracle(left: &[i64], right: &[i64]) -> i64 {
    let mut raw = 0i128;
    for (&left, &right) in left.iter().zip(right) {
        let product = i128::from(left) * i128::from(right);
        assert!((-HALF..HALF).contains(&product));
        raw += product;
        assert!((-HALF..HALF).contains(&raw));
    }
    raw.div_euclid(4096) as i64
}

#[test]
fn row_energy_private_dot_preserves_products_single_floor_and_material_counts() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let ranges = [range(-4000, 4000); 2];
    let (plain, _) = private_dot(binding(), &ranges, &ranges, &budget).unwrap();
    let (bounded, metadata) =
        private_dot_with_row_energy(binding(), &ranges, &ranges, [2, 18_000_000], &budget).unwrap();
    assert_eq!(metadata.raw_range, Some(range(-6000, 6000)));
    assert_eq!(metadata.range, range(-2, 1));
    assert_eq!(format!("{:?}", plain.code), format!("{:?}", bounded.code));
    let old = plain.resources();
    let new = bounded.resources();
    assert_eq!(new.instructions, old.instructions);
    assert_eq!(new.registers, old.registers);
    assert_eq!(new.triples_total, 4);
    assert_eq!(new.triples_total, old.triples_total);
    assert_eq!(new.dcf_keys_total, old.dcf_keys_total);
    assert_eq!(new.frames_total, old.frames_total);
    assert_eq!(
        new.private_material_bytes_total,
        old.private_material_bytes_total
    );
    for (left, right) in [
        ([1, 1], [3000, 3000]),
        ([-1, -1], [3000, 3000]),
        ([1, -1], [3000, 3000]),
        ([1, 0], [-1, 3000]),
        ([0, 0], [3000, -3000]),
    ] {
        assert!(energy(&left) <= 2 && energy(&right) <= 18_000_000);
        let inputs = [left[0], left[1], right[0], right[1]];
        let expected = oracle(&left, &right);
        assert!((metadata.range.min..=metadata.range.max).contains(&expected));
        assert_eq!(run(&bounded, &inputs), [expected]);
        assert_eq!(run(&plain, &inputs), [expected]);
    }
    assert_eq!(oracle(&[1, 1], &[3000, 3000]), 1); // Per-product flooring gives 0.
    drop((plain, bounded));
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn row_energy_exact_integer_vectors_bound_every_prefix_of_loose_scalar_rows() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let broad = [range(-(1 << 26), 1 << 26); 4];
    assert!(matches!(
        private_dot(binding(), &broad, &broad, &budget),
        Err(Error::Domain)
    ));
    let mut prefixes = Vec::new();
    for width in 1..=4 {
        let (plan, metadata) = private_dot_with_row_energy(
            binding(),
            &broad[..width],
            &broad[..width],
            [8, 5],
            &budget,
        )
        .unwrap();
        assert_eq!(metadata.raw_range, Some(range(-7, 7)));
        assert_eq!(metadata.range, range(-1, 0));
        prefixes.push((plan, metadata));
    }
    // Exhaustive independent integers. No production energy/root arithmetic is
    // used to decide which vectors satisfy the declared complete-row premises.
    let mut admitted = 0;
    for mut code in 0..5usize.pow(8) {
        let mut values = [0i64; 8];
        for value in &mut values {
            *value = (code % 5) as i64 - 2;
            code /= 5;
        }
        let (left, right) = values.split_at(4);
        if energy(left) > 8 || energy(right) > 5 {
            continue;
        }
        admitted += 1;
        let mut partial = 0i128;
        for i in 0..4 {
            let product = i128::from(left[i]) * i128::from(right[i]);
            partial += product;
            let raw = prefixes[i].1.raw_range.unwrap();
            assert!((i128::from(raw.min)..=i128::from(raw.max)).contains(&partial));
            assert!((-7..=7).contains(&product));
            assert!((-1..=0).contains(&partial.div_euclid(4096)));
        }
    }
    assert!(admitted > 1000);
    for inputs in [[2, 2, 0, 0, 2, 1, 0, 0], [-2, -2, 0, 0, 2, 1, 0, 0]] {
        assert_eq!(
            run(&prefixes[3].0, &inputs),
            [oracle(&inputs[..4], &inputs[4..])]
        );
    }
    drop(prefixes);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn row_energy_complete_cancellation_cannot_admit_wrapped_terms_or_prefixes() {
    use std::sync::atomic::Ordering;
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    budget.fail_after.store(0, Ordering::SeqCst);
    let h = HALF as i64;
    for (left, right) in [
        (vec![1; 4], vec![h - 1, 1, -(h - 1), -1]),
        (vec![1; 4], vec![-h, -1, h - 1, 2]),
        (vec![h - 1; 2], vec![2, -2]),
    ] {
        let raw: i128 = left
            .iter()
            .zip(&right)
            .map(|(&a, &b)| i128::from(a) * i128::from(b))
            .sum();
        assert_eq!(raw, 0); // A final-only check would incorrectly admit this.
        let left_ranges: Vec<_> = left.iter().map(|&x| range(x, x)).collect();
        let right_ranges: Vec<_> = right.iter().map(|&x| range(x, x)).collect();
        assert!(matches!(
            private_dot_with_row_energy(
                binding(),
                &left_ranges,
                &right_ranges,
                [energy(&left), energy(&right)],
                &budget
            ),
            Err(Error::Domain)
        ));
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn row_energy_signed50_edges_and_scalar_intersection_do_not_widen() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let h = HALF as i64;
    for x in [-h, h - 1] {
        let (plan, meta) = private_dot_with_row_energy(
            binding(),
            &[range(x, x)],
            &[range(1, 1)],
            [energy(&[x]), 1],
            &budget,
        )
        .unwrap();
        assert_eq!(meta.raw_range, Some(range(x, x)));
        assert_eq!(run(&plan, &[x, 1]), [i128::from(x).div_euclid(4096) as i64]);
    }
    assert!(matches!(
        private_dot_with_row_energy(
            binding(),
            &[range(h / 2, h / 2)],
            &[range(2, 2)],
            [energy(&[h / 2]), 4],
            &budget
        ),
        Err(Error::Domain)
    ));
    // An enormous but representable radius need not defeat tighter scalar bounds.
    let (plan, meta) = private_dot_with_row_energy(
        binding(),
        &[range(0, 1)],
        &[range(0, 1)],
        [u128::MAX, 1],
        &budget,
    )
    .unwrap();
    assert_eq!(meta.raw_range, Some(range(0, 1)));
    assert_eq!(meta.range, range(0, 0));
    drop(plan);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn row_energy_inconsistent_premises_shapes_binding_and_overflow_reject_before_allocation() {
    use std::sync::atomic::Ordering;
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    budget.fail_after.store(0, Ordering::SeqCst);
    let ranges = [range(3, 7), range(-9, -4), range(-2, 2)];
    let zeros = [range(0, 0); 3];
    assert_eq!(minimum_row_energy(&ranges), Ok(25));
    for (left, right, energies) in [
        (&ranges[..], &zeros[..], [24, 0]),
        (&zeros[..], &ranges[..], [0, 24]),
        (&zeros[..], &zeros[..], [u128::MAX, 2]),
        (&zeros[..], &zeros[..], [1 << 127, 2]),
        (&zeros[..0], &zeros[..0], [0, 0]),
        (&zeros[..1], &zeros[..2], [0, 0]),
    ] {
        assert!(matches!(
            private_dot_with_row_energy(binding(), left, right, energies, &budget),
            Err(Error::Domain)
        ));
    }
    for invalid in [range(2, 1), range(-(1 << 49) - 1, 0), range(0, 1 << 49)] {
        assert!(matches!(
            private_dot_with_row_energy(binding(), &[invalid], &[zeros[0]], [1, 0], &budget),
            Err(Error::Domain)
        ));
        assert!(matches!(
            private_dot_with_row_energy(binding(), &[zeros[0]], &[invalid], [0, 1], &budget),
            Err(Error::Domain)
        ));
    }
    for invalid in [
        Binding {
            source_digest: [0; 32],
            ..binding()
        },
        Binding {
            contract_digest: [0; 32],
            ..binding()
        },
    ] {
        assert!(matches!(
            private_dot_with_row_energy(invalid, &zeros, &zeros, [0, 0], &budget),
            Err(Error::Domain)
        ));
    }
    let wide = vec![range(0, 0); MAX_INPUTS / 2 + 1];
    assert!(matches!(
        private_dot_with_row_energy(binding(), &wide, &wide, [0, 0], &budget),
        Err(Error::Domain)
    ));
    assert_eq!(budget.used_bytes(), 0);
    assert_eq!(budget.fail_after.load(Ordering::SeqCst), 0);
}

#[test]
fn row_energy_pins_each_premise_and_rejects_material_and_wire_substitution() {
    use crate::fusefss_backend::wire::ImportGuard;
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let ranges = [range(-1, 1); 2];
    let (plain, plain_meta) = private_dot(binding(), &ranges, &ranges, &budget).unwrap();
    let (a, ma) =
        private_dot_with_row_energy(binding(), &ranges, &ranges, [2, 3], &budget).unwrap();
    let (b, mb) =
        private_dot_with_row_energy(binding(), &ranges, &ranges, [3, 2], &budget).unwrap();
    let (same, _) =
        private_dot_with_row_energy(binding(), &ranges, &ranges, [2, 3], &budget).unwrap();
    assert_eq!(ma, plain_meta);
    assert_eq!(ma, mb);
    assert_eq!(a.digest(), same.digest());
    assert_ne!(a.digest(), plain.digest());
    assert_ne!(a.digest(), b.digest()); // Same radius; ordered premise pins differ.
    let ctx = context(&a); // Identical outer context isolates the Plan-domain pin.
    let mut receiver = b.runtime(0, ctx, &[0; 4]).unwrap();
    let Event::NeedMaterial { pc } = receiver.poll().unwrap() else {
        panic!("product")
    };
    let [left, right] = a.issue_step(pc, ctx).unwrap();
    assert!(matches!(receiver.feed(left), Err(Error::Context)));
    assert!(receiver.is_consumed());
    drop((receiver, right));
    let [left, right] = a.issue_step(pc, ctx).unwrap();
    drop(right);
    let packet = left.export_packet().unwrap();
    let mut guard = ImportGuard::new_indexed(2, &budget).unwrap();
    assert!(matches!(
        b.import_step_bytes(pc, ctx, 0, packet.nonce(), packet.as_bytes(), &mut guard),
        Err(Error::Context)
    ));
    // Failed substituted delivery still burns the request-long nonce.
    assert!(matches!(
        a.import_step_bytes(pc, ctx, 0, packet.nonce(), packet.as_bytes(), &mut guard),
        Err(Error::Consumed)
    ));
    drop((packet, guard, same, plain, a, b));
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn row_energy_uses_normal_budget_failures_and_owned_output_lifetimes() {
    use std::sync::atomic::Ordering;
    let ranges = [range(-1, 1); 2];
    let tiny = Budget::new(1).unwrap();
    assert!(matches!(
        private_dot_with_row_energy(binding(), &ranges, &ranges, [2, 2], &tiny),
        Err(Error::Resource)
    ));
    assert_eq!(tiny.used_bytes(), 0);
    let mut success = false;
    for fail in 0..32 {
        let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
        budget.fail_after.store(fail, Ordering::SeqCst);
        match private_dot_with_row_energy(binding(), &ranges, &ranges, [2, 2], &budget) {
            Ok((plan, _)) => {
                drop(plan);
                success = true;
            }
            Err(error) => assert!(matches!(error, Error::Allocation)),
        }
        assert_eq!(budget.used_bytes(), 0);
        if success {
            break;
        }
    }
    assert!(success);
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let (plan, _) =
        private_dot_with_row_energy(binding(), &ranges, &ranges, [2, 2], &budget).unwrap();
    let plan = plan.into_shared().unwrap();
    let ctx = context(&plan);
    let mut left = plan.runtime_owned(0, ctx, &[0; 4]).unwrap();
    let mut right = plan.runtime_owned(1, ctx, &[word(-1), 0, 1, 0]).unwrap();
    let [a, b] = drive(&plan, ctx, &mut left, &mut right);
    drop((plan, left, right));
    assert_eq!(a.shares()[0].wrapping_add(b.shares()[0]) & MASK, word(-1));
    assert_eq!(budget.used_bytes(), 16);
    drop((a, b));
    assert_eq!(budget.used_bytes(), 0);
}
