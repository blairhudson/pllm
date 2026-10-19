use super::super::super::MAX_ALLOCATION_BYTES;
use super::*;

fn range(min: i64, max: i64) -> InputRange {
    InputRange { min, max }
}
fn full_budget() -> Budget {
    Budget::new(MAX_ALLOCATION_BYTES).unwrap()
}

#[test]
fn complete_public_copy_cost_counts_every_lookup_and_active_row() {
    let budget = full_budget();
    // Independently counted 256-entry chunks, including complete zero-support
    // intervals: boundary u64s plus two u64 coefficients per interval, two roles.
    for (k, bytes) in [(768, 3_174_352), (1024, 6_348_752)] {
        let (plan, _) = reciprocal(k, range(64, 64 * k as i64), &budget).unwrap();
        assert_eq!(plan.resources().public_table_copy_bytes_total, bytes);
        assert!(bytes > plan.resources().public_table_reservation_bytes);
        drop(plan);
        assert_eq!(budget.used_bytes(), 0);
    }
    let (plan, _) = softmax(
        &[range(-131072, 131072); 4],
        &[true, false, true, true],
        &budget,
    )
    .unwrap();
    // Three active rows each copy E0 and E1, followed by one k3 reciprocal.
    assert_eq!(plan.resources().public_table_copy_bytes_total, 86_464);
    drop(plan);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn public_numeric_proofs_reject_before_tables() {
    let budget = full_budget();
    for bad in [range(1, 0), range(-(1 << 37) - 1, 0), range(0, 1 << 37)] {
        assert!(matches!(silu(bad, &budget), Err(Error::Domain)));
        assert_eq!(budget.used_bytes(), 0);
    }
    for bad in [range(-1, 1), range(0, 65536), range(3, 2)] {
        assert!(matches!(nexp(bad, &budget), Err(Error::Domain)));
        assert_eq!(budget.used_bytes(), 0);
    }
    for (k, bad) in [
        (0, range(64, 64)),
        (1025, range(64, 64)),
        (1, range(63, 64)),
        (2, range(64, 129)),
    ] {
        assert!(matches!(reciprocal(k, bad, &budget), Err(Error::Domain)));
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn public_shape_mask_and_subtraction_obligations() {
    assert_eq!(check_softmax(&[], &[]), Err(Error::Domain));
    assert_eq!(check_softmax(&[range(0, 0)], &[]), Err(Error::Domain));
    assert_eq!(check_softmax(&[range(0, 0)], &[false]), Err(Error::Domain));
    assert_eq!(
        check_softmax(&[range(0, 0); 1025], &[true; 1025]),
        Err(Error::Domain)
    );
    assert_eq!(
        check_softmax(&[range(-(1 << 49), 0)], &[true]),
        Err(Error::Domain)
    );
    assert_eq!(check_softmax(&[range(-(1 << 49), -1)], &[true]), Ok(1));
    assert_eq!(
        check_softmax(
            &[range(-(1 << 49), -1), range((1 << 49) - 1, (1 << 49) - 1)],
            &[true, false]
        ),
        Ok(1)
    );
    assert_eq!(check_softmax(&[range(0, (1 << 49) - 1)], &[true]), Ok(1));
    assert_eq!(
        check_softmax(&[range(0, 0), range(1 << 49, 1 << 49)], &[true, false]),
        Err(Error::Domain)
    );
}

#[test]
fn checked_wide_arithmetic_never_wraps() {
    assert_eq!(add(HALF - 1, 1), Err(Error::Domain));
    assert_eq!(sub(-HALF, 1), Err(Error::Domain));
    assert_eq!(mul(i128::MAX, 2), Err(Error::Domain));
    assert_eq!(sub(i128::MIN, 1), Err(Error::Domain));
    assert_eq!(add(i128::MAX, 1), Err(Error::Domain));
    assert_eq!(mul(4096, 1024), Ok(1 << 22));
    assert_eq!(mul(4096, 4096), Ok(1 << 24));
    assert_eq!(reciprocal_len(1), Ok(128));
    assert_eq!(reciprocal_len(2), Ok(256));
    assert_eq!(reciprocal_len(3), Ok(256));
    assert_eq!(reciprocal_len(768), Ok(65536));
    assert_eq!(reciprocal_len(1024), Ok(131072));
}

#[test]
fn rejected_table_capacity_leaves_no_reservation() {
    let budget = Budget::new(1).unwrap();
    assert!(matches!(
        silu(range(-65, 65), &budget),
        Err(Error::Resource)
    ));
    assert_eq!(budget.used_bytes(), 0);
    assert!(matches!(
        reciprocal(1024, range(64, 65536), &budget),
        Err(Error::Resource)
    ));
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn sealed_gate_identity_binds_domain_public_k_and_mask() {
    let budget = full_budget();
    assert_eq!(source_digest(), source_tables::source_digest());
    {
        let (a, _) = silu(range(-65, 65), &budget).unwrap();
        let (b, _) = silu(range(-64, 65), &budget).unwrap();
        assert_ne!(a.digest(), b.digest());
    }
    {
        // Equal full table bytes and q, different source public k.
        let (a, _) = reciprocal(2, range(64, 128), &budget).unwrap();
        let (b, _) = reciprocal(3, range(64, 128), &budget).unwrap();
        assert_ne!(a.digest(), b.digest());
    }
    {
        let ranges = [range(0, 0); 3];
        let (a, _) = softmax(&ranges, &[true, false, true], &budget).unwrap();
        let (b, _) = softmax(&ranges, &[true, true, false], &budget).unwrap();
        assert_ne!(a.digest(), b.digest());
    }
    assert_eq!(budget.used_bytes(), 0);
}

// Small independent integer controls. These do not substitute for protected
// runtime replay; they pin source order and reject tempting mathematical changes.
fn clear_silu(x: i64) -> (i64, usize, u64) {
    let y = x.div_euclid(64);
    let index = y.unsigned_abs().min(1023) as usize;
    let output =
        (if y >= 0 { i128::from(x) } else { 0 }) - i128::from(source_tables::silu()[index]);
    (y, index, output.rem_euclid(1 << 50) as u64)
}
fn clear_exp(c: usize) -> u64 {
    source_tables::exp0()[c % 256] * source_tables::exp1()[c / 256] / 4096
}
fn clear_softmax(row: &[i64], active: &[bool]) -> (Vec<u64>, u64, u64, Vec<u64>) {
    let maximum = row
        .iter()
        .zip(active)
        .filter(|(_, &bit)| bit)
        .map(|(&x, _)| x)
        .max()
        .unwrap();
    let exps: Vec<_> = row
        .iter()
        .zip(active)
        .map(|(&x, &bit)| {
            if bit {
                clear_exp((maximum - x).min(65535) as usize)
            } else {
                0
            }
        })
        .collect();
    let denominator = exps.iter().sum::<u64>() / 64;
    let inverse = (1 << 18) / denominator;
    let output = exps.iter().map(|e| e * inverse / 4096).collect();
    (exps, denominator, inverse, output)
}

#[test]
fn negative_floor_and_original_input_controls() {
    assert_eq!(clear_silu(-1), (-1, 1, (1 << 50) - 31));
    assert_eq!(clear_silu(-63).0, -1);
    assert_eq!(clear_silu(-64).0, -1);
    assert_eq!(clear_silu(-65).0, -2);
    assert_eq!(clear_silu(-(1 << 37)), (-(1 << 31), 1023, 0));
    assert_eq!(clear_silu(63).2, 63);
    assert_eq!(clear_silu(65).2, 34);
}

#[test]
fn masked_max_and_source_rounded_sum_controls() {
    assert_eq!(clear_exp(0), 4096);
    assert_eq!(
        clear_softmax(&[0], &[true]),
        (vec![4096], 64, 4096, vec![4096])
    );
    assert_eq!(clear_softmax(&[0, 0, 0], &[true; 3]).3, vec![1365; 3]);
    assert_eq!(clear_softmax(&[0, -4096], &[true; 2]).3, vec![3013, 1107]);
    assert_eq!(
        clear_softmax(&[0, 99999, -4096], &[true, false, true]).3,
        vec![3013, 0, 1107]
    );
    assert_eq!(clear_softmax(&[0, -65536], &[true; 2]).3, vec![4096, 0]);
}

#[test]
fn independent_controls_match_frozen_fixtures() {
    let fixture: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../../../../paper/reproduction-contracts/fusefss/gate-set/fixtures.json"
    ))
    .unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        match case["gate"].as_str().unwrap() {
            "silu" => {
                let (floor, index, output) = clear_silu(case["args"][0].as_i64().unwrap());
                assert_eq!(
                    serde_json::json!({"floor":floor,"index":index,"output":output}),
                    case["expected"]
                );
            }
            "nexp" => assert_eq!(
                serde_json::json!(clear_exp(case["args"][0].as_u64().unwrap() as usize)),
                case["expected"]
            ),
            "softmax" => {
                let row: Vec<_> = case["args"][0]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_i64().unwrap())
                    .collect();
                let active: Vec<_> = if case["args"][1].is_null() {
                    vec![true; row.len()]
                } else {
                    case["args"][1]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_u64().unwrap() == 1)
                        .collect()
                };
                let (exp, denominator, reciprocal, output) = clear_softmax(&row, &active);
                assert_eq!(
                    serde_json::json!({"exp":exp,"denominator":denominator,"reciprocal":reciprocal,"output":output}),
                    case["expected"]
                );
            }
            _ => {}
        }
    }
}

fn context() -> crate::shared_rescale_reference::Context {
    crate::shared_rescale_reference::Context {
        plan: [71; 32],
        session: [72; 32],
        operation: [73; 32],
        first_tensor_index: 0,
    }
}

/// Test-only coordinator/reconstruction. Production Runtime still owns one role
/// and never receives sibling registers or an unmasked intermediate.
fn protected(plan: &Plan<'_>, inputs: &[i64]) -> Vec<u64> {
    use super::super::runtime::Event;
    const MASK: u64 = (1 << 50) - 1;
    let left: Vec<_> = inputs
        .iter()
        .enumerate()
        .map(|(i, _)| (0x123456789abcd + 91 * i as u64) & MASK)
        .collect();
    let right: Vec<_> = inputs
        .iter()
        .zip(&left)
        .map(|(&x, &share)| (x as u64).wrapping_sub(share) & MASK)
        .collect();
    let mut a = plan.runtime(0, context(), &left).unwrap();
    let mut b = plan.runtime(1, context(), &right).unwrap();
    let mut ea = a.poll().unwrap();
    let mut eb = b.poll().unwrap();
    loop {
        match (ea, eb) {
            (Event::NeedMaterial { pc }, Event::NeedMaterial { pc: other }) => {
                assert_eq!(pc, other);
                let [ma, mb] = plan.issue_step(pc, context()).unwrap();
                ea = a.feed(ma).unwrap();
                eb = b.feed(mb).unwrap();
            }
            (Event::Frame(fa), Event::Frame(fb)) => {
                ea = a.advance(fb.as_bytes()).unwrap();
                eb = b.advance(fa.as_bytes()).unwrap();
            }
            (Event::Output(oa), Event::Output(ob)) => {
                assert!(a.is_consumed() && b.is_consumed());
                assert!(matches!(a.poll(), Err(Error::Consumed)));
                assert!(matches!(b.poll(), Err(Error::Consumed)));
                return oa
                    .shares()
                    .iter()
                    .zip(ob.shares())
                    .map(|(&x, &y)| x.wrapping_add(y) & MASK)
                    .collect();
            }
            _ => panic!("public schedules diverged"),
        }
    }
}

#[test]
fn protected_silu_negative_floor_and_original_phi() {
    let budget = full_budget();
    for x in [-1, -65, 65] {
        let (plan, layout) = silu(range(-65, 65), &budget).unwrap();
        let words = protected(&plan, &[x]);
        let (floor, index, output) = clear_silu(x);
        assert_eq!(&words[layout.final_outputs], &[output]);
        for (name, slots) in layout.intermediates {
            let expected = match name {
                "floor" => (floor as u64) & ((1 << 50) - 1),
                "index" => index as u64,
                _ => unreachable!(),
            };
            assert_eq!(&words[slots], &[expected]);
        }
    }
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn protected_nexp_small_integer_control() {
    let budget = full_budget();
    let (plan, layout) = nexp(range(0, 65535), &budget).unwrap();
    let words = protected(&plan, &[257]);
    assert_eq!(&words[layout.final_outputs], &[clear_exp(257)]);
    drop(plan);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn protected_reciprocal_small_integer_control() {
    let budget = full_budget();
    let (plan, layout) = reciprocal(3, range(64, 192), &budget).unwrap();
    let words = protected(&plan, &[192]);
    assert_eq!(&words[layout.final_outputs], &[1365]);
    drop(plan);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn gate_runtime_wrong_role_and_cancel_burn_state() {
    use super::super::runtime::Event;
    let budget = full_budget();
    {
        let (plan, _) = nexp(range(0, 65535), &budget).unwrap();
        let mut runtime = plan.runtime(0, context(), &[17]).unwrap();
        let pc = match runtime.poll().unwrap() {
            Event::NeedMaterial { pc } => pc,
            _ => panic!("gate must request fresh conversion material"),
        };
        let [correct, wrong] = plan.issue_step(pc, context()).unwrap();
        assert!(matches!(runtime.feed(wrong), Err(Error::Context)));
        assert!(runtime.is_consumed());
        assert!(matches!(runtime.poll(), Err(Error::Consumed)));
        drop(correct);
        let mut cancelled = plan.runtime(1, context(), &[29]).unwrap();
        cancelled.cancel();
        assert!(matches!(cancelled.poll(), Err(Error::Consumed)));
    }
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn protected_softmax_masked_max_and_unrenormalized_equal_row() {
    let budget = full_budget();
    for (row, mask) in [
        (vec![0, 99999, -4096], vec![true, false, true]),
        (vec![0, 0, 0], vec![true; 3]),
    ] {
        let ranges: Vec<_> = row.iter().map(|&x| range(x, x)).collect();
        let (plan, layout) = softmax(&ranges, &mask, &budget).unwrap();
        let words = protected(&plan, &row);
        let (exps, denominator, inverse, outputs) = clear_softmax(&row, &mask);
        assert_eq!(&words[layout.final_outputs], outputs);
        for (name, slots) in layout.intermediates {
            let expected = match name {
                "exp" => exps.clone(),
                "denominator" => vec![denominator],
                "reciprocal" => vec![inverse],
                _ => unreachable!(),
            };
            assert_eq!(&words[slots], expected);
        }
    }
    assert_eq!(budget.used_bytes(), 0);
}
