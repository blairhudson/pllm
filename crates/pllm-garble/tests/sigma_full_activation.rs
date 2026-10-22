//! Independent integer contract for bounded SIGMA ring50/Q12 activation.
//! Frozen public equation tables supply data only, not another backend's
//! activation evaluator. No native activation defines an expected value.
//! These checks alone make no decoder, SDK, security, or upstream-wire claim.

use sha2::{Digest, Sha256};

const INPUT_HALF: i128 = 1i128 << 37;
const GELU_HASH: &str = "b6a45a82dc8f660724943c1eb76654704c2e49563ffc0b1a0bc5388b3425a01b";
const SILU_HASH: &str = "ff696cf63579aa28571af2c56895c06b1ed2e64ce6fab942fcbbe9454744f63f";

#[derive(Clone, Copy, Debug)]
enum Activation {
    Gelu,
    Silu,
}

impl Activation {
    fn index_bits(self) -> u8 {
        match self {
            Self::Gelu => 8,
            Self::Silu => 10,
        }
    }

    fn table(self) -> Vec<u64> {
        let (record, values, expected_hash) = match self {
            Self::Gelu => {
                let record: serde_json::Value = serde_json::from_slice(include_bytes!(
                    "../../../paper/reproduction-contracts/fusefss/gelu-table.json"
                ))
                .unwrap();
                let values = record["values"].clone();
                assert_eq!(record["sha256_u64le"].as_str(), Some(GELU_HASH));
                (record, values, GELU_HASH)
            }
            Self::Silu => {
                let record: serde_json::Value = serde_json::from_slice(include_bytes!(
                    "../../../paper/reproduction-contracts/fusefss/gate-set/fixtures.json"
                ))
                .unwrap();
                let values = record["tables"]["silu"].clone();
                assert_eq!(record["sha256_u64le"]["silu"].as_str(), Some(SILU_HASH));
                (record, values, SILU_HASH)
            }
        };
        // Fixture provenance is not authority for execution semantics. Pin both
        // length and the actual u64LE bytes independently of fixture metadata.
        drop(record);
        let table: Vec<_> = values
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap())
            .collect();
        assert_eq!(table.len(), 1usize << self.index_bits());
        let mut hash = Sha256::new();
        for value in &table {
            hash.update(value.to_le_bytes());
        }
        assert_eq!(format!("{:x}", hash.finalize()), expected_hash);
        table
    }
}

fn modulus(bits: u8) -> i128 {
    1i128 << bits
}

fn residue(value: i128, bits: u8) -> u64 {
    value.rem_euclid(modulus(bits)) as u64
}

fn signed(word: u64, bits: u8) -> i128 {
    assert!(i128::from(word) < modulus(bits));
    let value = i128::from(word);
    if value >= modulus(bits) / 2 {
        value - modulus(bits)
    } else {
        value
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Oracle {
    y: i128,
    d: bool,
    absolute: i128,
    index: usize,
    output: i128,
    output_residue: u64,
}

// The direct mathematical formula, independent of the staged clipping and
// mask equations below. i128 avoids signed32 abs overflow at Y = -2^31.
fn oracle(x: i128, table: &[u64]) -> Oracle {
    assert!((-INPUT_HALF..INPUT_HALF).contains(&x));
    assert!(matches!(table.len(), 256 | 1024));
    let y = x.div_euclid(64);
    let d = y >= 0;
    let absolute = y.abs();
    let index = absolute.min(table.len() as i128 - 1) as usize;
    let output = if d { x } else { 0 } - i128::from(table[index]);
    Oracle {
        y,
        d,
        absolute,
        index,
        output,
        output_residue: residue(output, 50),
    }
}

// Literal source-plan clipping equations. A is an unsigned ring32 residue;
// only A-K receives a signed32 interpretation. No production helpers used.
fn staged_index(y: i128, bits: u8) -> (u64, i128, bool, usize) {
    assert!((-(1i128 << 31)..(1i128 << 31)).contains(&y));
    let p = if y >= 0 { y } else { 0 };
    let a = residue(2 * p - y, 32);
    let k = modulus(bits);
    let difference = signed(residue(i128::from(a) - k, 32), 32);
    let h = difference >= 0;
    let e = !h;
    let v = if e {
        i128::from(residue(i128::from(a) - k + 1, bits))
    } else {
        0
    };
    (a, difference, h, residue(v + k - 1, bits) as usize)
}

fn boundary_inputs(bits: u8) -> Vec<i128> {
    let k = modulus(bits);
    let mut values = vec![
        -INPUT_HALF,
        -INPUT_HALF + 1,
        -INPUT_HALF + 63,
        -INPUT_HALF + 64,
        INPUT_HALF - 65,
        INPUT_HALF - 64,
        INPUT_HALF - 1,
    ];
    for y in [-k - 1, -k, -k + 1, -2, -1, 0, 1, 2, k - 1, k, k + 1] {
        for offset in [-1, 0, 1, 63, 64] {
            values.push(64 * y + offset);
        }
    }
    values.sort_unstable();
    values.dedup();
    values
}

#[test]
fn frozen_table_bytes_and_literal_gelu_silu_examples() {
    for activation in [Activation::Gelu, Activation::Silu] {
        let table = activation.table();
        assert_eq!(table[0], 0);
        assert_eq!(table[1], 31);
        assert_eq!(table[table.len() - 1], 0);
        for (x, expected) in [(-1, -31), (0, 0), (1, 1), (63, 63), (64, 33)] {
            assert_eq!(oracle(x, &table).output, expected, "{activation:?}, X={x}");
            assert_eq!(oracle(x, &table).output_residue, residue(expected, 50));
        }
    }
}

#[test]
fn negative_bins_use_floor_not_truncation_toward_zero() {
    for activation in [Activation::Gelu, Activation::Silu] {
        let table = activation.table();
        for (x, y, j) in [
            (-129, -3, 3),
            (-128, -2, 2),
            (-127, -2, 2),
            (-65, -2, 2),
            (-64, -1, 1),
            (-63, -1, 1),
            (-1, -1, 1),
            (0, 0, 0),
            (63, 0, 0),
            (64, 1, 1),
        ] {
            let actual = oracle(x, &table);
            assert_eq!((actual.y, actual.index), (y, j), "X={x}");
            assert_eq!(actual.d, x >= 0);
        }
        assert_ne!((-1i128).div_euclid(64), -1i128 / 64);
    }
}

#[test]
fn unsigned_absolute_minimum_and_signed_clip_difference_are_admitted() {
    for activation in [Activation::Gelu, Activation::Silu] {
        let table = activation.table();
        let bits = activation.index_bits();
        let k = modulus(bits);
        let minimum = oracle(-INPUT_HALF, &table);
        assert_eq!(minimum.y, -(1i128 << 31));
        assert_eq!(minimum.absolute, 1i128 << 31);
        let (a, difference, h, j) = staged_index(minimum.y, bits);
        assert_eq!(a, 1u64 << 31);
        assert_eq!(signed(a, 32), -(1i128 << 31));
        assert_eq!(difference, (1i128 << 31) - k);
        assert!(h);
        assert_eq!(j, table.len() - 1);
        assert_eq!(minimum.output, 0);

        for x in boundary_inputs(bits) {
            let expected = oracle(x, &table);
            let (a, difference, h, j) = staged_index(expected.y, bits);
            assert_eq!(i128::from(a), expected.absolute, "X={x}");
            assert_eq!(difference, expected.absolute - k, "X={x}");
            assert!((-(1i128 << 31)..(1i128 << 31)).contains(&difference));
            assert_eq!(h, expected.absolute >= k, "X={x}");
            assert_eq!(j, expected.index, "X={x}");
            assert_eq!(signed(expected.output_residue, 50), expected.output);
        }
    }
}

#[test]
fn every_table_bin_and_clip_k_minus_one_k_k_plus_one_match_clear_schedule() {
    for activation in [Activation::Gelu, Activation::Silu] {
        let table = activation.table();
        let bits = activation.index_bits();
        let k = modulus(bits);
        for magnitude in 0..=k + 1 {
            for sign in [-1, 1] {
                let y = sign * magnitude;
                for offset in [0, 1, 63] {
                    let x = 64 * y + offset;
                    let expected = oracle(x, &table);
                    assert_eq!(expected.y, y);
                    assert_eq!(expected.index, magnitude.min(k - 1) as usize);
                    assert_eq!(staged_index(y, bits).3, expected.index);
                    let selected_original = if expected.d { x } else { 0 };
                    assert_eq!(
                        expected.output_residue,
                        residue(selected_original - i128::from(table[expected.index]), 50)
                    );
                }
            }
        }
    }
}

#[test]
fn projection_affine_and_boolean_masks_preserve_endpoint_equations() {
    // Forced clear masks test algebra, not a claim that public issuance exposes
    // deterministic private-mask injection. All widths use 0/1/half/max.
    let masks50 = [0, 1, modulus(50) / 2, modulus(50) - 1];
    let masks32 = [0, 1, modulus(32) / 2, modulus(32) - 1];
    for x in [-INPUT_HALF, -65, -1, 0, 63, 64, INPUT_HALF - 1] {
        for r in masks50 {
            let common50 = residue(x + r, 50);
            let projected = residue(i128::from(common50), 38);
            let projected_mask = residue(r, 38);
            assert_eq!(
                signed(
                    residue(i128::from(projected) - i128::from(projected_mask), 38),
                    38
                ),
                x
            );
        }
        let y = x.div_euclid(64);
        let d = i128::from(y >= 0);
        let p = d * y;
        for ry in masks32 {
            for rp in masks32 {
                let y_hat = i128::from(residue(y + ry, 32));
                let p_hat = i128::from(residue(p + rp, 32));
                let a_hat = i128::from(residue(2 * p_hat - y_hat, 32));
                let ra = i128::from(residue(2 * rp - ry, 32));
                assert_eq!(
                    residue(a_hat - ra, 32),
                    u64::try_from(y.unsigned_abs()).unwrap()
                );
            }
        }
        for bit_mask in [0i128, 1] {
            let d_hat = d ^ bit_mask;
            assert_eq!(d_hat ^ bit_mask, d);
            assert_eq!((d_hat ^ 1) ^ bit_mask, 1 - d);
        }
    }
}

#[test]
fn deferred_own_share_subtraction_and_final_mask_need_no_remask() {
    let masks = [0, 1, modulus(50) / 2, modulus(50) - 1];
    for activation in [Activation::Gelu, Activation::Silu] {
        let table = activation.table();
        for x in [-INPUT_HALF, -65, -1, 0, 63, 64, INPUT_HALF - 1] {
            let expected = oracle(x, &table);
            let t = i128::from(table[expected.index]);
            let f = if expected.d { x } else { 0 };
            for r in masks {
                for u in masks {
                    let rho = i128::from(residue(2 * u - r, 50));
                    assert_eq!(rho % 2, r % 2);
                    // Choose a valid common R=t+r and reconstructed Select
                    // value z=t+u. Arbitrary z0 defines z1, not native output.
                    let common_r = i128::from(residue(t + r, 50));
                    let z = i128::from(residue(t + u, 50));
                    for share0 in masks {
                        let share1 = i128::from(residue(z - share0, 50));
                        let l0 = i128::from(residue(2 * share0 - common_r, 50));
                        let l1 = i128::from(residue(2 * share1, 50));
                        assert_eq!(residue(l0 + l1, 50), residue(t + rho, 50));
                        for rf in masks {
                            let f0 = share0;
                            let f1 = i128::from(residue(f + rf - f0, 50));
                            let o0 = i128::from(residue(f0 - l0, 50));
                            let o1 = i128::from(residue(f1 - l1, 50));
                            let output_mask = rf - rho;
                            assert_eq!(
                                residue(o0 + o1, 50),
                                residue(expected.output + output_mask, 50)
                            );
                            assert_eq!(residue(o0 + o1 - output_mask, 50), expected.output_residue);
                        }
                    }
                }
            }
        }
    }
}

#[test]
#[should_panic]
fn clear_oracle_rejects_exclusive_upper_input_endpoint() {
    oracle(INPUT_HALF, &Activation::Gelu.table());
}

#[test]
#[should_panic]
fn clear_oracle_rejects_input_below_declared_lower_endpoint() {
    oracle(-INPUT_HALF - 1, &Activation::Silu.table());
}

// Public consuming API checks deliberately do not decode native outputs or
// inspect dealer masks. Numerical parity is covered by region-private tests
// against wide-integer equations; the independent algebra above stays separate.
mod public_contract {
    use pllm_core::edge_contract::{
        EdgeDescriptor, EdgeIdentity, Interpretation, Numeric, Representation, Shape,
        SourceIdentity,
    };
    use pllm_garble::masked_wire_reference::typed::{
        activation_resources, approved_implementation, ActivationKind, ActivationSpec,
        ActivationStep, Budget, Error, LocalDealer, Step,
    };

    fn source() -> SourceIdentity {
        SourceIdentity::new([1; 32], [2; 32], approved_implementation())
    }
    fn edge(wire: u8, lanes: usize, predicate: bool) -> EdgeDescriptor {
        EdgeDescriptor::new(
            EdgeIdentity::new(source(), [3; 32], [wire + 64; 32], [wire; 32], 0),
            Shape::new(&[lanes]).unwrap(),
            Numeric::new(
                if predicate { 1 } else { 50 },
                if predicate { 0 } else { 12 },
                if predicate {
                    Interpretation::Predicate
                } else {
                    Interpretation::Signed
                },
            )
            .unwrap(),
            if predicate {
                Representation::CommonXorMasked
            } else {
                Representation::CommonAdditiveMasked
            },
            None,
        )
        .unwrap()
    }
    fn spec(kind: ActivationKind, lanes: usize) -> ActivationSpec {
        ActivationSpec::new(
            kind,
            [7; 32],
            edge(5, lanes, false),
            edge(6, lanes, false),
            [9; 32],
        )
        .unwrap()
    }

    #[test]
    fn complete_public_region_preserves_opaque_consuming_output_and_frame_lifetimes() {
        for kind in [ActivationKind::Gelu, ActivationKind::Silu] {
            for lanes in [1, 256] {
                let budget = Budget::new(64 << 20).unwrap();
                let retained;
                {
                    let mut dealer = LocalDealer::new(source(), [3; 32], [4; 32], &budget).unwrap();
                    let s = spec(kind, lanes);
                    let values: Vec<_> = (0..lanes)
                        .map(|i| [-1, 64, -(1i64 << 37), (1i64 << 37) - 1][i % 4])
                        .collect();
                    let (mask, [mut left, mut right], [x0, x1]) =
                        dealer.activation_from_values(s, &values).unwrap();
                    let mut frames = vec![[left.begin(x0).unwrap(), right.begin(x1).unwrap()]];
                    for _ in 1..8 {
                        let [a, b] = frames.last().unwrap();
                        match (
                            left.advance(b.as_bytes()).unwrap(),
                            right.advance(a.as_bytes()).unwrap(),
                        ) {
                            (ActivationStep::Frame(a), ActivationStep::Frame(b)) => {
                                frames.push([a, b])
                            }
                            _ => panic!("output before eighth exchange"),
                        }
                    }
                    assert_eq!(frames.iter().flatten().count(), 16);
                    assert_eq!(
                        frames
                            .iter()
                            .flatten()
                            .map(|f| f.as_bytes().len())
                            .sum::<usize>(),
                        activation_resources(s).unwrap().total_peer_frame_bytes
                    );
                    let [a, b] = frames.last().unwrap();
                    let [o0, o1] = match (
                        left.advance(b.as_bytes()).unwrap(),
                        right.advance(a.as_bytes()).unwrap(),
                    ) {
                        (ActivationStep::Output(a), ActivationStep::Output(b)) => [a, b],
                        _ => panic!("missing sole final output"),
                    };
                    assert_eq!(o0.descriptor(), s.output());
                    assert_eq!(o1.descriptor(), s.output());
                    assert_eq!(left.advance(&[]).err(), Some(Error::Consumed));
                    assert_eq!(right.advance(&[]).err(), Some(Error::Consumed));
                    drop((left, right));
                    let predicate = edge(10, lanes, true);
                    let (mask, [mut left, mut right]) =
                        dealer.drelu([9; 32], mask, predicate, [11; 32]).unwrap();
                    let a = left.begin(o0).unwrap();
                    let b = right.begin(o1).unwrap();
                    let outputs = match (
                        left.advance(b.as_bytes()).unwrap(),
                        right.advance(a.as_bytes()).unwrap(),
                    ) {
                        (Step::Output(a), Step::Output(b)) => [a, b],
                        _ => panic!("downstream predicate did not complete"),
                    };
                    for output in &outputs {
                        assert_eq!(output.descriptor(), predicate);
                    }
                    drop((mask, outputs, a, b, left, right, dealer));
                    retained = frames;
                }
                assert!(budget.used_bytes() > 0);
                drop(retained);
                assert_eq!(budget.used_bytes(), 0);
            }
        }
    }

    #[test]
    fn public_region_rejects_swapped_inputs_and_retires_both_roles() {
        let budget = Budget::new(64 << 20).unwrap();
        {
            let mut dealer = LocalDealer::new(source(), [3; 32], [4; 32], &budget).unwrap();
            let (mask, [mut left, mut right], [x0, x1]) = dealer
                .activation_from_values(spec(ActivationKind::Silu, 1), &[-65])
                .unwrap();
            assert_eq!(
                left.begin(x1).err().map(|e| e == Error::Context),
                Some(true)
            );
            assert_eq!(left.advance(&[]).err(), Some(Error::Consumed));
            right.cancel();
            assert_eq!(
                right.begin(x0).err().map(|e| e == Error::Consumed),
                Some(true)
            );
            drop((mask, left, right));
        }
        assert_eq!(budget.used_bytes(), 0);
    }
}
