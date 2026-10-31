//! Independent signed floor/ceil tape oracle. SOURCE ONLY; no test run claimed.
//! Tiny public matrices are local fixtures, not checkpoint/source-body scans.
use super::*;

const SCALE: i128 = 262_144;

fn rounded(n: i128, up: bool) -> i128 {
    // Independent signed Euclidean floor/ceil, including negative exact divides.
    let floor = n.div_euclid(SCALE);
    floor + i128::from(up && n.rem_euclid(SCALE) != 0)
}
fn energy(row: &[i128]) -> u128 {
    row.iter().map(|x| (x * x) as u128).sum()
}
fn magnitude(row: &[i128]) -> u128 {
    row.iter().map(|x| x.unsigned_abs()).max().unwrap_or(0)
}
fn dot_tape(x: &[i128], weights: &[&[i128]], bias: &[i128], tape: usize) -> Vec<i128> {
    weights.iter().enumerate().map(|(j, w)| {
        assert_eq!(x.len(), w.len());
        let raw: i128 = x.iter().zip(w.iter()).map(|(x, w)| x * w).sum();
        rounded(raw, tape & (1 << j) != 0) + bias[j]
    }).collect()
}
fn projection(x: &[i128], w: &[&[i128]], bias: &[i128], y: u128) -> Projection {
    // Deliberately independent fixture statistics, not a certificate issuer.
    let r = w.iter().map(|row| energy(row)).max().unwrap();
    let s = w.iter().map(|row| {
        let l1: u128 = row.iter().map(|x| x.unsigned_abs()).sum();
        l1 * l1
    }).sum();
    Projection { width: w.len() as u128, input_energy: Some(energy(x)),
        input_magnitude: magnitude(x), row_squared_norm: Some(r),
        row_l1_squared_sum: Some(s), bias_magnitude: magnitude(bias), final_magnitude: y }
}

#[test]
fn independent_signed_divisions_and_bias_order() {
    for (n, lo, hi) in [
        (-2 * SCALE, -2, -2), (-3 * SCALE / 2, -2, -1),
        (-1, -1, 0), (0, 0, 0), (1, 0, 1),
        (SCALE, 1, 1), (3 * SCALE / 2, 1, 2),
    ] {
        assert_eq!(rounded(n, false), lo);
        assert_eq!(rounded(n, true), hi);
    }
    let w: &[&[i128]] = &[&[SCALE / 2]];
    assert_eq!(dot_tape(&[3], w, &[-2], 1), vec![0]);
    // Moving a Q18 bias inside raw dot without multiplying by Q changes output.
    assert_ne!(rounded(3 * SCALE / 2 - 2, true), 0);
    // Rounding chunks is not rounding the complete dot.
    assert_ne!(rounded(SCALE / 2, true) * 2, rounded(SCALE, true));
}

#[test]
fn tiny_signed_matrices_all_rounding_tapes_and_post_tr_bias() {
    let weights: &[&[i128]] = &[
        &[SCALE, -SCALE / 2], &[-SCALE / 2, SCALE], &[1, -1],
    ];
    let mut ws = RowEnergyWorkspace::default();
    for x in [[-3, -2], [-1, 0], [0, 0], [1, -1], [3, 2]] {
        for bias in [[0, 0, 0], [-2, 1, -1]] {
            let bound = ws.projection(&projection(&x, weights, &bias, 20)).unwrap().unwrap();
            for tape in 0..8 {
                let output = dot_tape(&x, weights, &bias, tape);
                assert!(energy(&output) <= bound, "{x:?} {bias:?} {tape}");
            }
        }
    }
}

#[test]
fn product_tapes_no_independence_or_lane_spreading() {
    let mut ws = RowEnergyWorkspace::default();
    for a in [[1, 0], [-3, 2], [SCALE, -SCALE / 2], [0, 0]] {
        for u in [[1, 0], [2, -3], [SCALE / 2, SCALE], [0, 0]] {
            let p = Product { width: 2, activation_energy: Some(energy(&a)),
                up_energy: Some(energy(&u)), activation_magnitude: magnitude(&a),
                up_magnitude: magnitude(&u), final_magnitude: 2 * SCALE as u128 };
            let bound = ws.product(&p).unwrap().unwrap();
            for tape in 0..4 {
                let h = [rounded(a[0] * u[0], tape & 1 != 0),
                    rounded(a[1] * u[1], tape & 2 != 0)];
                assert!(energy(&h) <= bound);
            }
        }
    }
}

#[test]
fn negative_witnesses_rounding_correlation_and_concentration() {
    let up = rounded(3 * SCALE / 2, true);
    assert_eq!(up * up, 4);
    assert!(4 * 4 > 9 + 4); // 4 > (3/2)^2 + 1: missing cross term.
    let gate = dot_tape(&[1], &[&[SCALE]], &[0], 0);
    let up = dot_tape(&[1], &[&[SCALE]], &[0], 0);
    assert_eq!(gate[0] * up[0], 1); // identical projections are NOT orthogonal.
    let a = [1, 0];
    let u = [1, 0];
    let raw_product = [a[0] * u[0], a[1] * u[1]];
    assert_eq!(energy(&raw_product), 1);
    assert!(2 * energy(&raw_product) > energy(&a) * energy(&u));
}

#[test]
fn strict_projection_potential_is_conditional_not_a_layer_fit_claim() {
    let w: &[&[i128]] = &[&[SCALE], &[0], &[0], &[0]];
    let mut ws = RowEnergyWorkspace::default();
    for m in 3..=32_i128 {
        let p = projection(&[m], w, &[0; 4], m.unsigned_abs());
        let bound = ws.projection(&p).unwrap().unwrap();
        assert_eq!(bound, ((m + 2) * (m + 2)) as u128);
        assert!(bound < (4 * m * m) as u128);
    }
}

#[test]
fn caps_zero_missing_proofs_and_selected_activation_only() {
    assert_eq!(SELECTED_SILU_C, 2_229 + 4 * 131_072 + 16 * 59_806 + 256 * 2_913 + 4096 * 71);
    let mut ws = RowEnergyWorkspace::default();
    for a in [0, 1, 345_317, 4 * Q, SELECTED_SILU_C] {
        let bound = ws.activation(&Activation {
            width: 3, gate_energy: Some(0), final_magnitude: a,
        }).unwrap();
        assert_eq!(bound, Some(3 * a * a)); // A <= C: NEVER an improvement.
    }
    let bound = ws.activation(&Activation {
        width: 1, gate_energy: Some(0), final_magnitude: 2 * SELECTED_SILU_C,
    }).unwrap().unwrap();
    assert_eq!(bound, SELECTED_SILU_C * SELECTED_SILU_C);
    let mut p = projection(&[0], &[&[SCALE]], &[0], 0);
    assert_eq!(ws.projection(&p).unwrap(), Some(0));
    p.input_energy = None;
    assert_eq!(ws.projection(&p).unwrap(), None);
    p.input_energy = Some(0); p.row_squared_norm = None;
    assert_eq!(ws.projection(&p).unwrap(), None);
    p.row_squared_norm = Some(Q * Q); p.row_l1_squared_sum = None;
    assert_eq!(ws.projection(&p).unwrap(), None);
    assert_eq!(ws.activation(&Activation { width: 1, gate_energy: None,
        final_magnitude: 0 }).unwrap(), None);
    assert_eq!(ws.product(&Product { width: 1, activation_energy: None,
        up_energy: Some(0), activation_magnitude: 0, up_magnitude: 0,
        final_magnitude: 0 }).unwrap(), None);
    assert_eq!(ws.residual(10, None, Some(0)).unwrap(), None);
    assert_eq!(ws.residual(10, Some(4), Some(9)).unwrap(), Some(10));
}

#[test]
fn safe_fallback_energy_cannot_use_accepted_only_premise() {
    // Accepted row is zero; a separately selected safe fallback may not be.
    // The pure helper cannot authenticate B: the caller MUST certify both paths.
    let accepted = [0];
    let fallback = [7];
    let w: &[&[i128]] = &[&[SCALE]];
    let mut ws = RowEnergyWorkspace::default();
    let accepted_only = ws.projection(&projection(&accepted, w, &[0], 7)).unwrap().unwrap();
    assert!(energy(&dot_tape(&fallback, w, &[0], 0)) > accepted_only);
    let all_paths = ws.projection(&projection(&fallback, w, &[0], 7)).unwrap().unwrap();
    assert!(energy(&dot_tape(&fallback, w, &[0], 0)) <= all_paths);
}

#[test]
fn activation_checked_add_boundary_precedes_final_cap() {
    assert_eq!(SELECTED_SILU_C, 2_519_957);
    assert_eq!(SELECTED_SILU_C * SELECTED_SILU_C, 6_350_183_281_849);
    let mut ws = RowEnergyWorkspace::default();
    let mut a = Activation {
        width: 1,
        gate_energy: Some(340_282_366_920_938_463_463_374_601_081_584_929_606),
        final_magnitude: 1,
    };
    // Independent literal boundary, not authenticated production energy or source fit.
    assert_eq!(a.gate_energy.unwrap().checked_add(6_350_183_281_849), Some(u128::MAX));
    // Maximum raw sum is valid before the final cap reduces it to one.
    assert_eq!(ws.activation(&a).unwrap(), Some(1));
    a.gate_energy = a.gate_energy.map(|e| e + 1);
    // Overflow must refuse before that same tiny cap could hide it.
    assert!(matches!(ws.activation(&a), Err(Error::Bounds)));
}

#[test]
fn checked_overflow_never_becomes_none_or_a_weaker_cap() {
    let mut ws = RowEnergyWorkspace::default();
    let mut p = projection(&[1], &[&[SCALE]], &[0], 1);
    p.input_magnitude = u128::MAX;
    assert!(ws.projection(&p).is_err());
    assert!(ws.activation(&Activation { width: u128::MAX,
        gate_energy: Some(0), final_magnitude: 0 }).is_err());
    assert!(ws.product(&Product { width: 1, activation_energy: Some(u128::MAX),
        up_energy: Some(2), activation_magnitude: 1, up_magnitude: 1,
        final_magnitude: 0 }).is_err());
    assert!(ws.residual(0, Some(u128::MAX), Some(1)).is_err());
    p.width = 0; p.input_energy = None;
    assert!(ws.projection(&p).is_err());
    assert_eq!(ceil_q(Q).unwrap(), 1);
    assert_eq!(ceil_q(Q + 1).unwrap(), 2);
    assert_eq!(ceil_q(u128::MAX).unwrap(), u128::MAX / Q + 1);
}

#[test]
fn original_matrix_contract_does_not_promote_row_statistics_on_transpose() {
    use super::super::{public_matrix_contract, PublicMatrixStatistics, InputBounds, Interval};
    let (_, proof) = public_matrix_contract([1, 2, 3], true, true,
        || Ok(InputBounds { left: Interval::new(-2, 2)?, right: Interval::new(-3, 3)? }),
        PublicMatrixStatistics { row_l1: Some(1), row_squared_norm: Some(1),
            row_l1_squared_sum: Some(1) },
        || panic!("transposed rows must not consume an original-row energy premise"),
    ).unwrap();
    assert_eq!(proof, None);
}

#[test]
fn pure_per_lane_zero_substitution_exhausts_small_rows_without_energy_growth() {
    // Independent finite arithmetic model only: no actual guard owner, native
    // execution, source authentication or all-path certificate is constructed.
    let mut enumerated = 0;
    let mut mixed_nonzero = false;
    for width in [2_usize, 3] {
        for code in 0..7_usize.pow(width as u32) {
            let mut digits = code;
            let input: Vec<i128> = (0..width).map(|_| {
                let x = (digits % 7) as i128 - 3;
                digits /= 7;
                x
            }).collect();
            let selected: Vec<i128> = input.iter().map(|&x| {
                if (-2..=2).contains(&x) { x } else { 0 }
            }).collect();
            let all_valid = input.iter().all(|x| (-2..=2).contains(x));
            let any_valid = input.iter().any(|x| (-2..=2).contains(x));
            for (&x, &y) in input.iter().zip(&selected) {
                assert!((-2..=2).contains(&y));
                if (-2..=2).contains(&x) { assert_eq!(y, x); }
                else { assert_eq!(y, 0); }
                assert!(y * y <= x * x);
            }
            assert!(energy(&selected) <= energy(&input));
            if any_valid && !all_valid && energy(&selected) > 0 {
                mixed_nonzero = true;
                assert_ne!(selected, vec![0; width]);
            }
            enumerated += 1;
        }
    }
    assert_eq!(enumerated, 49 + 343);
    assert!(mixed_nonzero);
}

#[test]
fn pure_success_conditional_energy_and_sticky_rejection_do_not_zero_valid_lanes() {
    // Success-conditional bound ZERO holds only for this two-row family,
    // NOT for all successful rows in the accepted interval [-2,2].
    // Pure arithmetic oracle, not actual guard/fallback integration or custody.
    let family = [[0_i128, 0], [1, 3]];
    let success_only_bound = family.iter()
        .filter(|row| row.iter().all(|x| (-2..=2).contains(x)))
        .map(|row| energy(row)).max().unwrap();
    assert_eq!(success_only_bound, 0);
    let selected = family.map(|row| row.map(|x| {
        if (-2..=2).contains(&x) { x } else { 0 }
    }));
    let all_valid = family[1].iter().all(|x| (-2..=2).contains(x));
    assert!(!all_valid);
    assert_eq!(selected[1], [1, 0]);
    assert_eq!(energy(&selected[1]), 1);
    assert!(energy(&selected[1]) > success_only_bound);
    let incorrect_all_sticky_zero = [0_i128; 2];
    assert_ne!(selected[1], incorrect_all_sticky_zero);

    // Prior sticky rejection changes final acceptance, not lane-local selection.
    let prior_sticky = false;
    let locally_valid = [1_i128, 2];
    let local_all_valid = locally_valid.iter().all(|x| (-2..=2).contains(x));
    let selected = locally_valid.map(|x| {
        if (-2..=2).contains(&x) { x } else { 0 }
    });
    let final_acceptance = prior_sticky && local_all_valid;
    assert!(local_all_valid);
    assert!(!final_acceptance);
    assert_eq!(selected, [1, 2]);
    assert_eq!(energy(&selected), 5);
    assert_ne!(selected, incorrect_all_sticky_zero);
    assert!(energy(&locally_valid) > success_only_bound);
    // No success-only B is promoted here; existing None/overflow tests remain.
}
