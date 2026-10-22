use std::{collections::VecDeque, sync::Arc};

use k256::{elliptic_curve::sec1::ToEncodedPoint, ProjectivePoint, Scalar};
use num_bigint::{BigInt, BigUint, Sign};
use sha2::{Digest, Sha256};

use super::*;

/// Isolate tests that reserve process-wide publication, window, or wire credit.
/// Acquire before creating any owners and keep the guard until all owners drop;
/// guarding only counter assertions cannot exclude allocations in other tests.
/// Child threads within a test remain concurrent and must not reacquire this lock.
pub(super) fn process_accounting_guard() -> std::sync::MutexGuard<'static, ()> {
    static PROCESS_ACCOUNTING: std::sync::Mutex<()> = std::sync::Mutex::new(());
    PROCESS_ACCOUNTING
        .lock()
        .unwrap_or_else(|error| error.into_inner())
}

pub(super) struct ScriptEntropy {
    pub scalars: VecDeque<[u8; 32]>,
    pub calls: Vec<Purpose>,
    pub fail: Option<Purpose>,
    next: u8,
}

impl ScriptEntropy {
    pub fn new(values: &[u64]) -> Self {
        Self {
            scalars: values
                .iter()
                .map(|&v| Scalar::from(v).to_bytes().into())
                .collect(),
            calls: Vec::new(),
            fail: None,
            next: 17,
        }
    }
}
impl Entropy for ScriptEntropy {
    fn fill(&mut self, purpose: Purpose, out: &mut [u8]) -> Result<(), Error> {
        self.calls.push(purpose);
        if self.fail == Some(purpose) {
            return Err(Error::Entropy);
        }
        if matches!(
            purpose,
            Purpose::Owner | Purpose::ClaimChallenge | Purpose::MatrixChallenge
        ) {
            out.copy_from_slice(
                &self
                    .scalars
                    .pop_front()
                    .expect("explicit test scalar exhausted"),
            );
        } else {
            for (i, byte) in out.iter_mut().enumerate() {
                *byte = self.next.wrapping_add(i as u8);
            }
            self.next = self.next.wrapping_add(53);
        }
        Ok(())
    }
}

pub(super) fn hex(value: &str) -> Vec<u8> {
    (0..value.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&value[i..i + 2], 16).unwrap())
        .collect()
}

pub(super) fn scalar_order() -> BigUint {
    BigUint::parse_bytes(
        b"fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141",
        16,
    )
    .unwrap()
}

pub(super) fn field_bytes(value: BigInt) -> [u8; 32] {
    let p = BigInt::from(scalar_order());
    let value = ((value % &p) + &p) % &p;
    let bytes = value.to_bytes_be().1;
    let mut result = [0; 32];
    result[32 - bytes.len()..].copy_from_slice(&bytes);
    result
}

pub(super) fn integer_oracle(weights: &[i16], input: &[i64]) -> Vec<i64> {
    weights
        .chunks_exact(input.len())
        .map(|row| {
            let sum: BigInt = row
                .iter()
                .zip(input)
                .map(|(&w, &x)| BigInt::from(w) * BigInt::from(x))
                .sum();
            i64::try_from(sum).unwrap()
        })
        .collect()
}

fn share_oracle(weights: &[i16], input: &[u64]) -> Vec<u64> {
    let modulus = BigInt::from(1u8) << 64;
    weights
        .chunks_exact(input.len())
        .map(|row| {
            let sum: BigInt = row
                .iter()
                .zip(input)
                .map(|(&w, &x)| BigInt::from(w) * BigInt::from(x))
                .sum();
            u64::try_from(((sum % &modulus) + &modulus) % &modulus).unwrap()
        })
        .collect()
}

/// Independent integer bit-rounding oracle: no i64->f32 cast, no candidate
/// quantizer/kernel/rescaler. All nonzero results are normal binary32 numbers.
fn rescale_bits_oracle(value: i64) -> u32 {
    if value == 0 {
        return 0;
    }
    let magnitude = value.unsigned_abs();
    let mut exponent = 63 - magnitude.leading_zeros();
    let mut mantissa = if exponent <= 23 {
        magnitude << (23 - exponent)
    } else {
        let shift = exponent - 23;
        let top = magnitude >> shift;
        let lost = magnitude & ((1u64 << shift) - 1);
        let half = 1u64 << (shift - 1);
        top + u64::from(lost > half || (lost == half && top & 1 != 0))
    };
    if mantissa == 1 << 24 {
        mantissa >>= 1;
        exponent += 1;
    }
    (u32::from(value < 0) << 31) | ((exponent + 127 - 22) << 23) | (mantissa as u32 & 0x7fffff)
}

#[test]
fn numeric_completion_ties_ranges_signed_endpoints_and_conversion_order() {
    for (scaled, expected) in [
        (-3.5, -4),
        (-2.5, -2),
        (-1.5, -2),
        (-0.5, 0),
        (0.5, 0),
        (1.5, 2),
        (2.5, 2),
        (3.5, 4),
        (257.0, 257),
        (-32768.5, -32768),
        (32767.0, 32767),
    ] {
        assert_eq!(encode_weight(scaled / 2048.0), Ok(expected));
        assert_eq!(encode_input(scaled / 2048.0), Ok(i64::from(expected)));
    }
    for value in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MAX,
        32767.5 / 2048.0,
        -32769.0 / 2048.0,
    ] {
        assert_eq!(encode_weight(value), Err(Error::Numeric));
    }
    assert_eq!(encode_input(-(2.0_f64.powi(52))), Ok(i64::MIN));
    assert_eq!(encode_input(2.0_f64.powi(52)), Err(Error::Numeric));
    assert_eq!(encode_input(f64::from_bits(1)), Ok(0));
    assert_eq!(reconstruct_signed(u64::MAX, 0), -1);
    assert_eq!(reconstruct_signed(u64::MAX, 1), 0);
    assert_eq!(reconstruct_signed(1 << 63, 0), i64::MIN);
    assert_eq!(reconstruct_signed((1 << 63) - 1, 0), i64::MAX);
    let double_round = (1i64 << 62) + (1i64 << 38) + 1;
    for value in [
        0,
        1,
        -1,
        i64::MIN,
        i64::MAX,
        (1 << 24) + 1,
        (1 << 24) + 3,
        -((1 << 24) + 3),
        double_round,
        -double_round,
    ] {
        assert_eq!(rescale_signed(value).to_bits(), rescale_bits_oracle(value));
    }
    assert_ne!(
        rescale_signed(double_round).to_bits(),
        ((double_round as f64 / 4_194_304.0) as f32).to_bits()
    );
}

#[test]
fn immutable_i16_ring64_matches_big_integer_oracle_including_wrapped_shares() {
    let mut weights = vec![i16::MIN, 32767, -257, 0, 0, 0, 300, -1, 2048];
    let original = weights.clone();
    let matrix = I16Matrix::from_encoded(3, 3, &weights, &[4096; 3]).unwrap();
    weights.fill(0);
    for input in [[4096, -4096, -1], [-1, 0, 1], [0; 3]] {
        let a = [u64::MAX, 1 << 63, 0x1234_5678_90ab_cdef];
        let b = std::array::from_fn::<_, 3, _>(|i| (input[i] as u64).wrapping_sub(a[i]));
        let left = matrix.multiply_share(&a).unwrap();
        let right = matrix.multiply_share(&b).unwrap();
        assert_eq!(*left, share_oracle(&original, &a));
        assert_eq!(*right, share_oracle(&original, &b));
        let actual = left
            .iter()
            .zip(right.iter())
            .map(|(&a, &b)| reconstruct_signed(a, b))
            .collect::<Vec<_>>();
        assert_eq!(actual, integer_oracle(&original, &input));
    }
    assert!(matrix.multiply_share(&[]).is_err());
    let imported = I16Matrix::from_f64(1, 2, &[257.5 / 2048.0, -1.5 / 2048.0], &[10; 2]).unwrap();
    assert_eq!(&*imported.weights, &[258, -2]);
}

#[test]
fn public_bounds_and_preallocation_shape_caps_are_conservative() {
    let _accounting = process_accounting_guard();
    assert!(matches!(
        I16Matrix::from_encoded(usize::MAX, 2, &[], &[]),
        Err(Error::Limit)
    ));
    assert!(matches!(
        I16Matrix::from_encoded(1, 1, &[1], &[1 << 63]),
        Err(Error::Bound)
    ));
    assert!(I16Matrix::from_encoded(1, 1, &[1], &[i64::MAX as u64]).is_ok());
    assert!(I16Matrix::from_encoded(1, 1, &[0], &[1 << 63]).is_ok());
    assert!(matches!(
        I16Matrix::from_encoded(1, 2, &[1, -1], &[1 << 62; 2]),
        Err(Error::Bound)
    ));
    assert!(matches!(
        I16Matrix::from_encoded(1, 1, &[0], &[u64::MAX]),
        Err(Error::Bound)
    ));
    assert!(matches!(
        FloatTable::from_le_bytes(usize::MAX, usize::MAX, &[]),
        Err(Error::Limit)
    ));
    assert!(PublicationResources::for_shapes(&[(usize::MAX, 2)], 1, 1).is_err());
    assert!(PublicationResources::for_shapes(&[(1, 1); MAX_STAGES + 1], 1, 1).is_err());
    let resources = PublicationResources::for_shapes(
        &[(MAX_DIMENSION, MAX_DIMENSION); MAX_STAGES],
        MAX_VOCABULARY,
        MAX_DIMENSION,
    )
    .unwrap();
    assert_eq!(
        resources.source_weight_bytes,
        MAX_STAGES * MAX_DIMENSION * MAX_DIMENSION * 2
    );
    assert!(resources.max_window_payload_bytes <= MAX_WINDOW_BYTES);
    assert!(resources.native_publication_payload_bytes > (MAX_DIMENSION * (MAX_STAGES + 1)) * 33);
    let matrix = I16Matrix::from_encoded(1, 1, &[1], &[1]).unwrap();
    let table = FloatTable::from_le_bytes(1, 1, &[0; 4]).unwrap();
    let mut rng = ScriptEntropy::new(&[]);
    let duplicate = [
        Stage {
            id: [1; 32],
            matrix: &matrix,
        },
        Stage {
            id: [1; 32],
            matrix: &matrix,
        },
    ];
    assert!(matches!(
        Publication::publish_with([0; 32], &duplicate, &table, &mut rng),
        Err(Error::Binding)
    ));
    assert!(rng.calls.is_empty());
}

#[test]
fn dense_lookup_uses_original_float_bytes_and_canonical_partial_byte() {
    let bits = [0u32, 0x80000000, 1, 0x3f800001, 0x7f7fffff, 0xbf800000];
    let bytes = (0..11 * 3)
        .flat_map(|i| bits[i % bits.len()].to_le_bytes())
        .collect::<Vec<_>>();
    let table = FloatTable::from_le_bytes(11, 3, &bytes).unwrap();
    for token in (0..11).chain([0, 10, 0]) {
        let left = [0b10100110, 0b00000101];
        let mut right = left;
        right[token / 8] ^= 1 << (token % 8);
        // Separately written byte oracle with conditional row selection.
        let byte_oracle = |selector: &[u8]| {
            let mut output = [0u8; 12];
            for row in 0..11 {
                if selector[row / 8] & (1 << (row % 8)) != 0 {
                    for column in 0..12 {
                        output[column] ^= bytes[row * 12 + column];
                    }
                }
            }
            output
        };
        let a = table.xor_rows(&left).unwrap();
        let b = table.xor_rows(&right).unwrap();
        assert_eq!(&**a, &byte_oracle(&left));
        assert_eq!(&**b, &byte_oracle(&right));
        assert_eq!(
            a.iter()
                .zip(b.iter())
                .map(|(&a, &b)| a ^ b)
                .collect::<Vec<_>>(),
            bytes[token * 12..(token + 1) * 12]
        );
    }
    assert!(table.xor_rows(&[0, 0x80]).is_err());
    assert!(table.xor_rows(&[0]).is_err());
    for bits in [0x7fc00001u32, 0x7f800000, 0xff800000] {
        assert!(matches!(
            FloatTable::from_le_bytes(1, 1, &bits.to_le_bytes()),
            Err(Error::Numeric)
        ));
    }
    assert_eq!(
        Sha256::digest([0u8; 4]).as_slice(),
        hex("df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119")
    );
}

#[test]
fn scalar_sampler_rejects_at_order_without_modulo_bias_and_keeps_zero() {
    let order: [u8; 32] = scalar_order().to_bytes_be().try_into().unwrap();
    let max: [u8; 32] = (scalar_order() - BigUint::from(1u8))
        .to_bytes_be()
        .try_into()
        .unwrap();
    let mut rng = ScriptEntropy::new(&[]);
    rng.scalars.extend([order, [255; 32], max, [0; 32]]);
    assert_eq!(
        sample_scalar(&mut rng, Purpose::ClaimChallenge)
            .unwrap()
            .to_bytes()
            .as_slice(),
        max
    );
    assert_eq!(rng.calls.len(), 3);
    assert_eq!(
        sample_scalar(&mut rng, Purpose::ClaimChallenge).unwrap(),
        Scalar::ZERO
    );
    let mut rng = ScriptEntropy::new(&[]);
    rng.scalars.extend([[255; 32]; 128]);
    assert_eq!(sample_scalar(&mut rng, Purpose::Owner), Err(Error::Entropy));
    assert_eq!(rng.calls.len(), 128);
    let mut rng = ScriptEntropy::new(&[]);
    rng.fail = Some(Purpose::Owner);
    assert_eq!(sample_scalar(&mut rng, Purpose::Owner), Err(Error::Entropy));
    assert_eq!(signed_scalar(-1).to_bytes().as_slice(), max);
    assert_ne!(signed_scalar(-1), Scalar::from(u64::MAX));
    assert_eq!(
        signed_scalar(i64::MIN).to_bytes().as_slice(),
        field_bytes(BigInt::from(i64::MIN))
    );
}

// Independent affine secp256k1 oracle. Uses arbitrary integers, coordinate
// equations and Fermat inversion; never calls k256 to calculate expected points.
type AffineOracle = Option<(BigUint, BigUint)>;
fn curve_prime() -> BigUint {
    BigUint::parse_bytes(
        b"fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f",
        16,
    )
    .unwrap()
}
fn affine_add(a: &AffineOracle, b: &AffineOracle) -> AffineOracle {
    let (Some((x, y)), Some((u, v))) = (a, b) else {
        return a.clone().or_else(|| b.clone());
    };
    let p = curve_prime();
    if x == u && (y != v || y == &BigUint::from(0u8)) {
        return None;
    }
    let (num, den) = if x == u {
        (
            (BigUint::from(3u8) * x * x) % &p,
            (BigUint::from(2u8) * y) % &p,
        )
    } else {
        ((v + &p - y) % &p, (u + &p - x) % &p)
    };
    let slope = (num * den.modpow(&(&p - BigUint::from(2u8)), &p)) % &p;
    let next_x = (&slope * &slope + &p * BigUint::from(2u8) - x - u) % &p;
    let next_y = (&slope * ((x + &p - &next_x) % &p) + &p - y) % &p;
    Some((next_x, next_y))
}
fn group_oracle(value: i128) -> Vec<u8> {
    let mut point = Some((
        BigUint::parse_bytes(
            b"79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798",
            16,
        )
        .unwrap(),
        BigUint::parse_bytes(
            b"483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8",
            16,
        )
        .unwrap(),
    ));
    let mut result = None;
    let mut scalar = value.unsigned_abs();
    while scalar != 0 {
        if scalar & 1 != 0 {
            result = affine_add(&result, &point);
        }
        point = affine_add(&point, &point);
        scalar >>= 1;
    }
    let Some((x, mut y)) = result else {
        return vec![0];
    };
    if value < 0 {
        y = curve_prime() - y;
    }
    let mut encoded = vec![0; 33];
    encoded[0] = if (&y % BigUint::from(2u8)) == BigUint::from(0u8) {
        2
    } else {
        3
    };
    let x = x.to_bytes_be();
    encoded[33 - x.len()..].copy_from_slice(&x);
    encoded
}

#[test]
fn owner_commitments_match_independent_affine_group_and_exact_signed_relation() {
    let _accounting = process_accounting_guard();
    let matrix = I16Matrix::from_encoded(2, 2, &[257, -1, -129, 32767], &[100; 2]).unwrap();
    let other = I16Matrix::from_encoded(1, 2, &[-1, 0], &[100; 2]).unwrap();
    let table = FloatTable::from_le_bytes(1, 1, &[0; 4]).unwrap();
    let publication = Publication::publish_with(
        [9; 32],
        &[
            Stage {
                id: [1; 32],
                matrix: &matrix,
            },
            Stage {
                id: [2; 32],
                matrix: &other,
            },
        ],
        &table,
        &mut ScriptEntropy::new(&[2, 3]),
    )
    .unwrap();
    for (point, exponent) in publication.r.iter().zip([2, 3]) {
        assert_eq!(
            point.to_affine().to_encoded_point(true).as_bytes(),
            group_oracle(exponent)
        );
    }
    for (point, exponent) in publication.matrices[0].s.iter().zip([127, 98299]) {
        assert_eq!(
            point.to_affine().to_encoded_point(true).as_bytes(),
            group_oracle(exponent)
        );
    }
    for (point, exponent) in publication.matrices[1].s.iter().zip([-2, 0]) {
        assert_eq!(
            point.to_affine().to_encoded_point(true).as_bytes(),
            group_oracle(exponent)
        );
    }
    let x = [-7, 11];
    let y = integer_oracle(&matrix.weights, &x);
    let left = publication.matrices[0]
        .s
        .iter()
        .zip(x)
        .fold(ProjectivePoint::IDENTITY, |sum, (s, x)| {
            sum + *s * signed_scalar(x)
        });
    let right = publication
        .r
        .iter()
        .zip(&y)
        .fold(ProjectivePoint::IDENTITY, |sum, (r, &y)| {
            sum + *r * signed_scalar(y)
        });
    assert_eq!(left, right);
    assert_eq!(
        left.to_affine().to_encoded_point(true).as_bytes(),
        group_oracle(2 * i128::from(y[0]) + 3 * i128::from(y[1]))
    );
    assert_eq!(
        publication.row_hashes[0].as_slice(),
        hex("df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119")
    );
    assert!(Client::new(Arc::new(publication), [0; 32]).is_err());
    // The coordinate prime and scalar order are intentionally different.
    assert_ne!(curve_prime(), scalar_order());
    assert_eq!(
        BigInt::from_bytes_be(Sign::Plus, &field_bytes(BigInt::from(-1))),
        BigInt::from(scalar_order()) - 1
    );
}
