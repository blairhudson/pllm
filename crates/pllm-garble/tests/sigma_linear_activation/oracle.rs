//! Pure test oracle: deliberately no production arithmetic imports.
//! Coordinator may include beside `fixture`; no public facade assumed here.

use super::fixture;
use sha2::{Digest, Sha256};

const QUARTER: i128 = 1i128 << 48;
const SIGNED_HALF: i128 = 1i128 << 49;

/// Finite binary float -> floor(value*4096), no nearest-even or toward-zero.
/// Multiplication by a power of two is exact for admitted finite operands.
pub(super) fn encode(value: f64) -> Result<i128, &'static str> {
    let scaled = value * 4096.0;
    if !value.is_finite() || !scaled.is_finite() {
        return Err("nonfinite");
    }
    let floor = scaled.floor();
    if floor < -(SIGNED_HALF as f64) || floor >= SIGNED_HALF as f64 {
        return Err("signed50 encoding");
    }
    Ok(floor as i128)
}

/// Complete unwrapped Q24 dot. No intermediate modular reduction or floor.
pub(super) fn dot(x: &[i128], w: &[i128], inner: usize) -> Result<i128, &'static str> {
    if inner == 0 || x.len() != inner || w.len() != inner {
        return Err("incomplete inner dimension");
    }
    x.iter().zip(w).try_fold(0i128, |sum, (&a, &b)| {
        if !(-SIGNED_HALF..SIGNED_HALF).contains(&a) || !(-SIGNED_HALF..SIGNED_HALF).contains(&b) {
            return Err("signed50 operand");
        }
        sum.checked_add(a.checked_mul(b).ok_or("wide overflow")?)
            .ok_or("wide overflow")
    })
}

/// Inclusive operand intervals; inclusive complete sum endpoints. Fixed
/// source weights may use their exhaustive encoded min/max. Public input
/// domain must be declared independently, never fitted to sample extrema.
fn certify(x: [i128; 2], w: [i128; 2], inner: usize) -> Result<[i128; 2], &'static str> {
    if inner == 0 || x[0] > x[1] || w[0] > w[1] {
        return Err("interval");
    }
    if [x, w]
        .iter()
        .flatten()
        .any(|v| !(-SIGNED_HALF..SIGNED_HALF).contains(v))
    {
        return Err("signed50 interval");
    }
    let products = [x[0] * w[0], x[0] * w[1], x[1] * w[0], x[1] * w[1]];
    let lo = products
        .iter()
        .min()
        .unwrap()
        .checked_mul(inner as i128)
        .ok_or("wide overflow")?;
    let hi = products
        .iter()
        .max()
        .unwrap()
        .checked_mul(inner as i128)
        .ok_or("wide overflow")?;
    if lo < -QUARTER || hi >= QUARTER {
        return Err("quarter ring");
    }
    Ok([lo, hi])
}

pub(super) fn rescale(sum: i128) -> Result<i128, &'static str> {
    if !(-QUARTER..QUARTER).contains(&sum) {
        return Err("quarter ring");
    }
    Ok(sum.div_euclid(4096))
}

#[derive(Clone, Copy)]
pub(super) enum Activation {
    Gelu,
    Silu,
}

impl Activation {
    pub(super) fn table(self) -> Vec<u64> {
        // Frozen independent paper-equation data only; no production LUT owner
        // or activation evaluator. Equations:
        // GELU floor(4096*(i/64)*(1-erf((i/64)/sqrt(2)))/2)
        // SiLU floor(4096*(i/64)/(1+exp(i/64))).
        let (record, key, hash, length) = match self {
            Self::Gelu => (
                include_str!("../../../../paper/reproduction-contracts/fusefss/gelu-table.json"),
                "",
                "b6a45a82dc8f660724943c1eb76654704c2e49563ffc0b1a0bc5388b3425a01b",
                256,
            ),
            Self::Silu => (
                include_str!(
                    "../../../../paper/reproduction-contracts/fusefss/gate-set/fixtures.json"
                ),
                "silu",
                "ff696cf63579aa28571af2c56895c06b1ed2e64ce6fab942fcbbe9454744f63f",
                1024,
            ),
        };
        let record: serde_json::Value = serde_json::from_str(record).unwrap();
        let values = if key.is_empty() {
            &record["values"]
        } else {
            &record["tables"][key]
        };
        let table: Vec<_> = values
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        assert_eq!(table.len(), length);
        let mut digest = Sha256::new();
        for value in &table {
            digest.update(value.to_le_bytes());
        }
        assert_eq!(format!("{:x}", digest.finalize()), hash);
        table
    }
}

pub(super) fn activate(z: i128, table: &[u64]) -> i128 {
    assert!((-(1i128 << 37)..(1i128 << 37)).contains(&z));
    assert!(matches!(table.len(), 256 | 1024));
    let y = z.div_euclid(64);
    let index = y.abs().min(table.len() as i128 - 1) as usize;
    (if y >= 0 { z } else { 0 }) - i128::from(table[index])
}

#[test]
fn immutable_checkpoint_submatrix_bytes_layout_and_bounds() {
    assert_eq!(fixture::REPOSITORY, "Qwen/Qwen2.5-0.5B-Instruct");
    assert_eq!(
        fixture::REVISION,
        "7ae557604adf67be50417f59c2c2f167def9a775"
    );
    assert_eq!(fixture::FILE_SHA256.len(), 64);
    assert_eq!(fixture::TENSOR, "model.layers.0.mlp.gate_proj.weight");
    assert_eq!(fixture::SOURCE_SHAPE_OI, [4864, 896]);
    let bytes = fixture::source_bytes();
    assert_eq!(bytes.len(), 512);
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        fixture::SELECTED_SHA256
    );
    let weights: Vec<_> = fixture::weights_io()
        .into_iter()
        .map(|v| encode(v).unwrap())
        .collect();
    let inputs: Vec<_> = fixture::input()
        .into_iter()
        .map(|v| encode(v).unwrap())
        .collect();
    assert_eq!(weights.len(), fixture::INNER);
    assert_eq!(
        (
            *weights.iter().min().unwrap(),
            *weights.iter().max().unwrap()
        ),
        (-280, 324)
    );
    let bounds = certify([-4096, 4096], [-280, 324], fixture::INNER).unwrap();
    assert_eq!(bounds, [-339_738_624, 339_738_624]);
    assert_eq!(bounds.map(|v| rescale(v).unwrap()), [-82_944, 82_944]);
    let sum = dot(&inputs, &weights, fixture::INNER).unwrap();
    assert!((bounds[0]..=bounds[1]).contains(&sum));
    let z = rescale(sum).unwrap();
    // SiLU follows source config; GELU is an additional D22 operator test,
    // explicitly NOT the original Qwen activation.
    for kind in [Activation::Silu, Activation::Gelu] {
        let output = activate(z, &kind.table());
        assert!((-SIGNED_HALF..SIGNED_HALF).contains(&output));
    }
}

#[test]
fn encoding_rejects_nonfinite_and_preserves_negative_floor() {
    for v in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::MAX] {
        assert!(encode(v).is_err());
    }
    assert_eq!(encode(-0.0), Ok(0));
    assert_eq!(encode(-1.0 / 8192.0), Ok(-1));
    assert_eq!(encode(1.0 / 8192.0), Ok(0));
    assert_eq!(encode(-((1u64 << 37) as f64)), Ok(-SIGNED_HALF));
    assert!(encode((1u64 << 37) as f64).is_err());
}

#[test]
fn complete_dot_not_partial_or_per_term_floor() {
    let x = vec![1; fixture::INNER];
    let mut w = vec![16; fixture::INNER];
    assert_eq!(rescale(dot(&x, &w, fixture::INNER).unwrap()), Ok(1));
    assert_eq!(
        x.iter()
            .zip(&w)
            .map(|(a, b)| (a * b).div_euclid(4096))
            .sum::<i128>(),
        0
    );
    assert!(dot(&x[..255], &w[..255], fixture::INNER).is_err());
    assert_eq!(rescale(dot(&x[..255], &w[..255], 255).unwrap()), Ok(0));
    // Last actual inner coordinate must contribute.
    w[255] = 4096;
    assert_eq!(dot(&x, &w, fixture::INNER), Ok(8176));
    assert_eq!(rescale(-1), Ok(-1));
    assert_eq!(rescale(-4095), Ok(-1));
    assert_eq!(rescale(-4096), Ok(-1));
    assert_eq!(rescale(-4097), Ok(-2));
    assert_eq!(rescale(dot(&[1, 1], &[-1, -1], 2).unwrap()), Ok(-1));
    assert_eq!((-1i128).div_euclid(4096) * 2, -2);
}

#[test]
fn quarter_ring_endpoints_and_complete_interval_admission() {
    assert_eq!(rescale(-QUARTER), Ok(-(1i128 << 36)));
    assert_eq!(rescale(QUARTER - 1), Ok((1i128 << 36) - 1));
    assert!(rescale(-QUARTER - 1).is_err());
    assert!(rescale(QUARTER).is_err());
    assert_eq!(certify([-QUARTER, -QUARTER], [1, 1], 1), Ok([-QUARTER; 2]));
    assert!(certify([QUARTER, QUARTER], [1, 1], 1).is_err());
    assert!(certify([-QUARTER - 1; 2], [1, 1], 1).is_err());
    assert!(certify([0, QUARTER - 1], [1, 1], 2).is_err());
    assert!(certify([0, SIGNED_HALF], [0, 0], 1).is_err());
    assert!(certify([1, -1], [0, 1], 1).is_err());
}

#[test]
fn unchanged_d20_tables_bins_negative_floor_and_no_final_shift() {
    for kind in [Activation::Gelu, Activation::Silu] {
        let table = kind.table();
        for (z, expected) in [(-1, -31), (0, 0), (1, 1), (63, 63), (64, 33)] {
            assert_eq!(activate(z, &table), expected);
        }
        for i in 0..table.len() + 2 {
            for sign in [-1, 1] {
                let y = sign * i as i128;
                for offset in [0, 1, 63] {
                    let z = y * 64 + offset;
                    let expected =
                        (if y >= 0 { z } else { 0 }) - i128::from(table[i.min(table.len() - 1)]);
                    assert_eq!(activate(z, &table), expected);
                }
            }
        }
        assert_eq!(activate(rescale(-QUARTER).unwrap(), &table), 0);
        assert_eq!(
            activate(rescale(QUARTER - 1).unwrap(), &table),
            (1i128 << 36) - 1
        );
    }
}
