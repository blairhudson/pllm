use super::*;

fn fixture(length: usize, salt: u64) -> Vec<u64> {
    (0..length)
        .map(|i| match i % 7 {
            0 => 0,
            1 => 1,
            2 => u64::MAX,
            3 => 1 << 63,
            _ => (i as u64)
                .wrapping_mul(0x9e3779b97f4a7c15)
                .wrapping_add(salt),
        })
        .collect()
}
fn negative(x: Wide, q: Wide) -> Wide {
    if x == Wide::default() {
        x
    } else {
        q.checked_sub(x).unwrap()
    }
}
fn difference(a: u64, b: u64) -> i128 {
    i128::from(a.wrapping_sub(b) as i64)
}

#[test]
fn normalized_down_matches_independent_bit_division_at_full_width_boundaries() {
    let r = Rns3::new(LIMBS).unwrap();
    let q = r.modulus();
    for i in 0..256_u64 {
        let m = i.wrapping_mul(0x9e3779b97f4a7c15);
        let x = match i % 4 {
            0 => r.encode(m).unwrap(),
            1 => q.checked_sub(Wide::from(i + 1)).unwrap(),
            2 => Wide::from(i),
            _ => q.half().checked_add(Wide::from(i)).unwrap(),
        };
        // Independent, deliberately slow 65-bit exact quotient oracle. This
        // observer's limb operations are explicitly included in work inventory.
        let mut n = x.checked_shl(64).unwrap().checked_add(q.half()).unwrap();
        let mut divisor = q.checked_shl(64).unwrap();
        let mut quotient = 0_u128;
        for bit in (0..=64).rev() {
            if n >= divisor {
                n = n.checked_sub(divisor).unwrap();
                quotient |= 1_u128 << bit;
            }
            divisor = divisor.half();
        }
        assert_eq!(r.down(x).unwrap(), quotient as u64, "case {i}");
    }
}

#[test]
fn exact_tuple_u_residual_boundaries_and_global_crt() {
    let r = Rns3::new(LIMBS).unwrap();
    let q = Wide([0xfc73a11bdfe2c001, 0xff8b000c0237bf8a, 0x7ffff, 0]);
    assert_eq!(r.modulus(), q); // decimal q=178405960980747204108026560376635421240770561
    assert_eq!(r.primes(), LIMBS);
    let mut values = vec![
        Wide::default(),
        Wide::from(1),
        q.half(),
        q.checked_sub(Wide::from(1)).unwrap(),
    ];
    for m in [0, 1, 2, 1 << 63, u64::MAX - 1, u64::MAX] {
        let u = r.encode(m).unwrap();
        let qm = q.checked_mul_word(m).unwrap();
        let tu = u.checked_shl(64).unwrap();
        assert_eq!(qm.checked_sub(tu).unwrap(), Wide::from(qm.0[0]));
        assert_eq!(r.down(u).unwrap(), m);
        values.push(u);
    }
    for x in values {
        let residues = LIMBS.map(|p| x.rem_word(p));
        assert_eq!(r.reconstruct(residues).unwrap(), x);
    }
    assert_eq!(r.down(q.checked_sub(Wide::from(1)).unwrap()).unwrap(), 0);
    assert_eq!(r.down(q.half()).unwrap(), 1 << 63);
    let below = q.half().shr_word(); // floor(q/(2*t))
    assert_eq!(r.down(below).unwrap(), 0);
    assert_eq!(
        r.down(below.checked_add(Wide::from(1)).unwrap()).unwrap(),
        1
    );
    // Odd q: (2*k+1)*q cannot equal even 2*t*x, so exact half ties do not exist.
    assert_eq!(q.0[0] & 1, 1);
    assert_eq!(r.down(q), Err(Error::NonCanonical));
    assert_eq!(r.reconstruct([LIMBS[0], 0, 0]), Err(Error::NonCanonical));
    assert_eq!(
        Wide([0, 0, 0, 1 << 63]).checked_shl(1),
        Err(Error::Capacity)
    );
    assert_eq!(
        Wide([u64::MAX; 4]).checked_add(Wide::from(1)),
        Err(Error::Capacity)
    );
    assert_eq!(
        Rns3::new([LIMBS[0], LIMBS[0], LIMBS[2]]),
        Err(Error::Unsupported)
    );
    assert_eq!(Rns3::new([9, 15, 17]), Err(Error::Capacity));
    // Preserve old ring64 refusal and old Delta oracle; no API replacement.
    assert_eq!(
        crate::coefficient_conversion::CoefficientConversion::new(64, u128::MAX),
        Err(crate::coefficient_conversion::ConversionError::ArithmeticCapacity)
    );
    assert!(crate::coefficient_layout::CoefficientRing::new(DEGREE, LIMBS[0]).is_err());
}

#[test]
// Retain the independent coefficient-index oracle and reviewed loop inventory.
#[allow(clippy::needless_range_loop)]
fn signed_coordinates_partial_trace_and_parse_addresses() {
    let c = Coordinates::bounded(DEGREE, SHAPE, WINDOW, DEGREE).unwrap();
    let lhs: Vec<_> = c.lhs(0, 0).unwrap().collect();
    assert_eq!((lhs[0].coefficient, lhs[0].negative), (0, false));
    assert_eq!((lhs[1].coefficient, lhs[1].negative), (DEGREE - 1, true));
    assert_eq!(c.lhs(2, 1).unwrap().count(), 1);
    assert_eq!(c.rhs(1, 1).unwrap().count(), 1);
    assert_eq!(c.address(2, 128).unwrap(), (0, 5));
    let mut visible = vec![false; DEGREE];
    for row in 0..3 {
        for col in 0..129 {
            let (group, at) = c.address(row, col).unwrap();
            assert_eq!(group, 0);
            visible[at] = true;
        }
    }
    for at in 0..DEGREE {
        assert_eq!(c.visible(0, at).unwrap(), visible[at]);
    }
    // Exhaustive small rings and every partial group; independent selector oracle.
    for n in [8, 16, 32] {
        for stride in [1, 2, 4, 8] {
            for count in 1..=stride {
                let p = 65537;
                let values: Vec<_> = (0..n * count).map(|i| (i * 29 + 17) as u64 % p).collect();
                let mut output = vec![0; n];
                let mut scratch = vec![0; (count.next_power_of_two() + 1) * n];
                trace_interleave(n, p, &values, stride, count, &mut output, &mut scratch).unwrap();
                for (at, &actual) in output.iter().enumerate() {
                    let lane = at % stride;
                    assert_eq!(
                        actual,
                        if lane < count {
                            values[lane * n + at - lane]
                        } else {
                            0
                        }
                    );
                }
            }
        }
    }
}

#[test]
fn shared_extraction_matches_legacy_odd_layout_without_changing_its_caps() {
    use crate::coefficient_layout::{CoefficientLayout, CoefficientRing};
    let ring = CoefficientRing::new(32, 65537).unwrap();
    let layout = CoefficientLayout::new(ring, [3, 5, 7], [2, 4, 4], 1 << 20).unwrap();
    let c = Coordinates::bounded(32, [3, 5, 7], [2, 4, 4], 4096).unwrap();
    let a: Vec<_> = (0..15).map(|i| i as u64 * 19).collect();
    let b: Vec<_> = (0..35).map(|i| i as u64 * 37).collect();
    for row in 0..2 {
        for inner in 0..2 {
            let mut legacy = [0; 32];
            let mut extracted = [0; 32];
            layout.pack_lhs(&a, row, inner, &mut legacy).unwrap();
            for at in c.lhs(row, inner).unwrap() {
                let v = a[at.matrix];
                extracted[at.coefficient] = if at.negative && v != 0 { 65537 - v } else { v };
            }
            assert_eq!(legacy, extracted);
        }
    }
    for inner in 0..2 {
        for col in 0..2 {
            let mut legacy = [0; 32];
            let mut extracted = [0; 32];
            layout.pack_rhs(&b, inner, col, &mut legacy).unwrap();
            for at in c.rhs(inner, col).unwrap() {
                extracted[at.coefficient] = b[at.matrix];
            }
            assert_eq!(legacy, extracted);
        }
    }
    for row in 0..3 {
        for col in 0..7 {
            assert_eq!(
                layout.packed_address(row, col).unwrap(),
                c.address(row, col).unwrap()
            );
        }
    }
    let values: Vec<_> = (0..3 * 32).map(|i| i as u64 * 7).collect();
    let mut legacy = [0; 32];
    let mut extracted = [0; 32];
    let mut scratch = [0; 5 * 32];
    ring.interleave(&values, 4, 3, &mut legacy, &mut scratch)
        .unwrap();
    trace_interleave(32, 65537, &values, 4, 3, &mut extracted, &mut scratch).unwrap();
    assert_eq!(legacy, extracted);
}

#[test]
fn complete_8192_both_directions_all_cells_actual_phase_and_residual() {
    // Whole campaign predebit precedes fixture construction and observer loops.
    let quote = Quote::qualification().unwrap();
    let mut work = Work::new(MAX_WORK).unwrap();
    let admitted =
        Admitted::qualification(quote.total_bytes(), &mut work, CryptoError::ExplicitZero).unwrap();
    let r = Rns3::new(LIMBS).unwrap();
    let lhs = [fixture(195, 17), fixture(195, 31)];
    let rhs = [fixture(8385, 43), fixture(8385, 59)];
    // Supplied deterministic CLEAR integers, not samples or privacy masks.
    let masks: [Vec<Wide>; 2] = std::array::from_fn(|d| {
        (0..DEGREE)
            .map(|i| match i % 5 {
                0 => Wide::default(),
                1 => r.modulus().checked_sub(Wide::from(1)).unwrap(),
                2 => r.modulus().half(),
                _ => r
                    .encode((i as u64).wrapping_mul(123456789).wrapping_add(d as u64))
                    .unwrap(),
            })
            .collect()
    });
    let input = Inputs {
        lhs: [&lhs[0], &lhs[1]],
        rhs: [&rhs[0], &rhs[1]],
        masks: [&masks[0], &masks[1]],
        crypto_error: CryptoError::ExplicitZero,
    };
    assert_eq!(quote.convolution_pairs(), 9_513_234);
    assert_eq!(quote.arithmetic(), 258_503_359);
    assert_eq!(quote.inventory().terms().len(), 40);
    assert!(quote.arithmetic() <= MAX_WORK);
    let independent_total = quote
        .inventory()
        .terms()
        .iter()
        .try_fold(0_u64, |a, t| {
            a.checked_add(t.visits().checked_mul(t.units_per_visit())?)
        })
        .unwrap();
    assert_eq!(independent_total, quote.arithmetic());
    // Native buffers are allocated only after complete mask validation too.
    admitted.validate_inputs(&input).unwrap();
    let mut scratch = vec![0; SCRATCH_WORDS];
    let mut phases = vec![0; PHASE_WORDS];
    let mut converted = vec![0; CONVERTED_WORDS];
    let mut matrices = vec![0; MATRIX_WORDS];
    admitted
        .evaluate(
            &input,
            Buffers {
                scratch: &mut scratch,
                phases: &mut phases,
                converted: &mut converted,
                matrices: &mut matrices,
            },
        )
        .unwrap();
    let tq = r.modulus().checked_shl(64).unwrap();
    for (d, (a, b)) in TwoPartyProductPlan::CROSS_TERMS.into_iter().enumerate() {
        for row in 0..3 {
            for col in 0..129 {
                let cell = row * 129 + col;
                let mut dense = 0_u64;
                let mut residual = Wide::default();
                let mut replay = [0_u64; 3];
                // Independent dense-cell oracle: no packing iterator/address use.
                for h in 0..65 {
                    let x = lhs[a as usize][row * 65 + h];
                    let y = rhs[b as usize][h * 129 + col];
                    dense = dense.wrapping_add(x.wrapping_mul(y));
                    // Independent visible-cell phase replay: every local h>0
                    // term wraps once in X^N+1; negative is taken BEFORE U.
                    let negative_term = h % 64 != 0;
                    let canonical = if negative_term { x.wrapping_neg() } else { x };
                    let qm = r.modulus().checked_mul_word(canonical).unwrap();
                    let u = qm.shr_word();
                    let rem_y = Wide::from(qm.0[0]).checked_mul_word(y).unwrap();
                    residual = residual
                        .add_mod(
                            if negative_term {
                                rem_y
                            } else {
                                negative(rem_y, tq)
                            },
                            tq,
                        )
                        .unwrap();
                    for (limb, p) in LIMBS.into_iter().enumerate() {
                        let term =
                            (u128::from(u.rem_word(p)) * u128::from(y) % u128::from(p)) as u64;
                        replay[limb] = if negative_term {
                            if replay[limb] >= term {
                                replay[limb] - term
                            } else {
                                p - (term - replay[limb])
                            }
                        } else {
                            (replay[limb] + term) % p
                        };
                    }
                }
                let at = (col % 128) * 64 + row * 2 + col / 128;
                for limb in 0..3 {
                    assert_eq!(phases[(d * 3 + limb) * DEGREE + at], replay[limb]);
                }
                let phase = r.reconstruct(replay).unwrap();
                // Whole-phase identity modulo t*q, including packed wrap K:
                // t*phase = q*(H mod t) + residual(A)*Y (mod t*q).
                assert_eq!(
                    phase.checked_shl(64).unwrap(),
                    r.modulus()
                        .checked_mul_word(dense)
                        .unwrap()
                        .add_mod(residual, tq)
                        .unwrap()
                );
                let sum = matrices[2 * d * CELLS + cell]
                    .wrapping_add(matrices[(2 * d + 1) * CELLS + cell]);
                assert!(
                    difference(sum, dense).abs() <= 1,
                    "direction={d} row={row} col={col}"
                );
                assert_eq!(
                    matrices[2 * d * CELLS + cell],
                    converted[2 * d * DEGREE + at]
                );
            }
        }
        // ALL structural tail/padding coefficients, not only lanes 6..63.
        for at in 0..DEGREE {
            let lane = at % 64;
            let column = at / 64;
            if lane >= 6 || (lane % 2 == 1 && column != 0) {
                for limb in 0..3 {
                    assert_eq!(phases[(d * 3 + limb) * DEGREE + at], 0);
                }
                assert_eq!(
                    converted[d * 2 * DEGREE + at]
                        .wrapping_add(converted[(d * 2 + 1) * DEGREE + at]),
                    0
                );
            }
        }
    }
    for row in 0..3 {
        for col in 0..129 {
            let dense = (0..65).fold(0_u64, |s, h| {
                s.wrapping_add(
                    lhs[0][row * 65 + h]
                        .wrapping_add(lhs[1][row * 65 + h])
                        .wrapping_mul(rhs[0][h * 129 + col].wrapping_add(rhs[1][h * 129 + col])),
                )
            });
            let cell = row * 129 + col;
            let sum = matrices[4 * CELLS + cell].wrapping_add(matrices[5 * CELLS + cell]);
            assert!(difference(sum, dense).abs() <= 2);
        }
    }
    // Same work counter cannot renew for a second direction pair.
    assert!(
        Admitted::qualification(quote.total_bytes, &mut work, CryptoError::ExplicitZero).is_err()
    );
}

#[test]
fn refusals_before_kernel_writes_or_work_renewal() {
    let q = Quote::qualification().unwrap();
    let mut w = Work::new(q.arithmetic() - 1).unwrap();
    assert!(Admitted::qualification(q.total_bytes, &mut w, CryptoError::ExplicitZero).is_err());
    assert_eq!(w.used(), 0);
    let mut w = Work::new(MAX_WORK).unwrap();
    assert!(Admitted::qualification(q.total_bytes - 1, &mut w, CryptoError::ExplicitZero).is_err());
    assert_eq!(w.used(), 0);
    assert!(Admitted::qualification(q.total_bytes, &mut w, CryptoError::Unknown).is_err());
    assert_eq!(w.used(), 0);
    let mut overflow = Work {
        used: u64::MAX,
        limit: MAX_WORK,
    };
    assert_eq!(overflow.consume(1), Err(Error::Overflow));
    assert!(Coordinates::bounded(8192, [usize::MAX, 65, 129], WINDOW, 8192).is_err());
    let mut output = [91; 8];
    assert_eq!(
        trace_interleave(8, 65537, &[0; 8], 4, 1, &mut output, &mut [0; 15]),
        Err(Error::BufferLength)
    );
    assert_eq!(output, [91; 8]);
}
