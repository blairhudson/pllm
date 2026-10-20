use super::*;
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

#[derive(Clone, Copy, Debug)]
pub enum WeightDtype {
    F32,
    F16,
    Bf16,
}

/// Immutable, checked external public tensor. No claimed digest/range accepted.
pub struct Tensor {
    pub(super) id: String,
    pub(super) shape: Vec<usize>,
    pub(super) values: Zeroizing<Vec<i64>>,
    pub(super) range: Range,
}
impl Tensor {
    /// Canonical IEEE payloads, not host byte order. Half formats reject high
    /// bits; every format rejects NaN/infinity before quantization.
    pub fn from_ieee(
        id: &str,
        shape: &[usize],
        dtype: WeightDtype,
        words: &[u32],
    ) -> Result<Self, Error> {
        if words.len() != elements(shape)? {
            return Err(Error::Domain);
        }
        let mut codes = Zeroizing::new(vector(words.len())?);
        for &bits in words {
            let value = match dtype {
                WeightDtype::F32 => f32::from_bits(bits) as f64,
                WeightDtype::Bf16 => {
                    if bits > 65535 {
                        return Err(Error::Domain);
                    }
                    f32::from_bits(bits << 16) as f64
                }
                WeightDtype::F16 => {
                    if bits > 65535 {
                        return Err(Error::Domain);
                    }
                    let sign = if bits & 32768 == 0 { 1.0 } else { -1.0 };
                    let exp = (bits >> 10) & 31;
                    let mant = bits & 1023;
                    if exp == 31 {
                        return Err(Error::Domain);
                    }
                    sign * if exp == 0 {
                        (mant as f64) * 2f64.powi(-24)
                    } else {
                        (1024 + mant) as f64 * 2f64.powi(exp as i32 - 25)
                    }
                }
            };
            let rounded = (value * 4096.0).round_ties_even();
            if !rounded.is_finite() || rounded < -(HALF as f64) || rounded >= HALF as f64 {
                return Err(Error::Domain);
            }
            codes.push(rounded as i64);
        }
        Self::take_codes(id, shape, codes)
    }
    /// Exact IEEE-754 multiplication by 2^12 followed by nearest-even rounding.
    /// Nonfinite values, signed50 overflow, and oversized arrays fail closed.
    pub fn from_f64(id: &str, shape: &[usize], values: &[f64]) -> Result<Self, Error> {
        if values.len() != elements(shape)? {
            return Err(Error::Domain);
        }
        let mut codes = Zeroizing::new(vector(values.len())?);
        for &value in values {
            let scaled = value * 4096.0;
            if !scaled.is_finite() {
                return Err(Error::Domain);
            }
            let rounded = scaled.round_ties_even();
            if rounded < -(HALF as f64) || rounded >= HALF as f64 {
                return Err(Error::Domain);
            }
            codes.push(rounded as i64);
        }
        Self::take_codes(id, shape, codes)
    }
    /// Import already quantized public Q12 codes; still verifies every value.
    pub fn from_q12(id: &str, shape: &[usize], values: &[i64]) -> Result<Self, Error> {
        if values.len() != elements(shape)? {
            return Err(Error::Domain);
        }
        let mut codes = Zeroizing::new(vector(values.len())?);
        codes.extend_from_slice(values);
        Self::take_codes(id, shape, codes)
    }
    fn take_codes(id: &str, shape: &[usize], values: Zeroizing<Vec<i64>>) -> Result<Self, Error> {
        if id.is_empty() || id.len() > 1024 || values.len() != elements(shape)? {
            return Err(Error::Domain);
        }
        let lo = *values.iter().min().ok_or(Error::Domain)?;
        let hi = *values.iter().max().ok_or(Error::Domain)?;
        let range = Range::new(lo as i128, hi as i128)?;
        let mut dims = vector(shape.len())?;
        dims.extend_from_slice(shape);
        let mut name = String::new();
        name.try_reserve_exact(id.len())
            .map_err(|_| Error::Allocation)?;
        name.push_str(id);
        Ok(Self {
            id: name,
            shape: dims,
            values,
            range,
        })
    }
    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn values(&self) -> &[i64] {
        &self.values
    }
}

/// Explicit pinned finite-F32 split-half coefficients for one semantic operator.
/// The descriptor binds its exact operator ID, theta, dimensions and every Q12
/// code. It is not an arbitrary range proof or an implicit host libm generator.
pub struct RotaryCoefficients {
    pub(super) operation: String,
    pub(super) theta: u64,
    pub(super) positions: usize,
    pub(super) dimensions: usize,
    pub(super) cos: Zeroizing<Vec<i64>>,
    pub(super) sin: Zeroizing<Vec<i64>>,
    ieee_digest: [u8; 32],
}
impl RotaryCoefficients {
    pub fn operation_id(&self) -> &str {
        &self.operation
    }
    pub fn shape(&self) -> [usize; 2] {
        [self.positions, self.dimensions]
    }
    pub fn ieee_digest(&self) -> [u8; 32] {
        self.ieee_digest
    }
    pub fn from_f32(
        operation: &str,
        theta: u64,
        shape: [usize; 2],
        cos: &[f32],
        sin: &[f32],
    ) -> Result<Self, Error> {
        if theta == 0
            || shape[1] % 2 != 0
            || shape[0] > 16
            || shape[1] > 4096
            || cos.len() != elements(&shape)?
            || sin.len() != cos.len()
            || cos
                .iter()
                .chain(sin)
                .any(|v| !v.is_finite() || v.abs() > 1.0)
        {
            return Err(Error::Domain);
        }
        let mut c = Zeroizing::new(vector(cos.len())?);
        let mut s = Zeroizing::new(vector(sin.len())?);
        let mut ieee = Sha256::new();
        ieee.update(b"pllm.fss-decoder.rotary-f32-payloads.v1\0");
        for (&a, &b) in cos.iter().zip(sin) {
            ieee.update(a.to_bits().to_le_bytes());
            ieee.update(b.to_bits().to_le_bytes());
            c.push((a as f64 * 4096.0).round_ties_even() as i64);
            s.push((b as f64 * 4096.0).round_ties_even() as i64);
        }
        for p in 0..shape[0] {
            for d in 0..shape[1] / 2 {
                let i = p * shape[1] + d;
                let j = i + shape[1] / 2;
                if cos[i].to_bits() != cos[j].to_bits() || sin[i].to_bits() != sin[j].to_bits() {
                    return Err(Error::Domain);
                }
            }
        }
        if cos[..shape[1]]
            .iter()
            .any(|v| v.to_bits() != 1f32.to_bits())
            || sin[..shape[1]]
                .iter()
                .any(|v| v.to_bits() != 0f32.to_bits())
            || operation.is_empty()
            || operation.len() > 1024
        {
            return Err(Error::Domain);
        }
        let mut name = String::new();
        name.try_reserve_exact(operation.len())
            .map_err(|_| Error::Allocation)?;
        name.push_str(operation);
        Ok(Self {
            operation: name,
            theta,
            positions: shape[0],
            dimensions: shape[1],
            cos: c,
            sin: s,
            ieee_digest: ieee.finalize().into(),
        })
    }
}

/// Bounded public source owner. Content hashes include every shape and code.
pub struct Source {
    pub(super) tensors: Vec<Tensor>,
    pub(super) rotary: Vec<RotaryCoefficients>,
    digest: [u8; 32],
    bytes: usize,
}
impl Source {
    pub fn new(mut tensors: Vec<Tensor>) -> Result<Self, Error> {
        if tensors.is_empty() || tensors.len() > MAX_NODES {
            return Err(Error::Resource);
        }
        tensors.sort_unstable_by(|a, b| a.id.cmp(&b.id));
        let mut bytes =
            std::mem::size_of::<Self>() + tensors.capacity() * std::mem::size_of::<Tensor>();
        let mut hash = Sha256::new();
        hash.update(b"pllm.fss-decoder.public-source.ring50-q12.nearest-even.v1\0");
        for (i, t) in tensors.iter().enumerate() {
            if i != 0 && tensors[i - 1].id == t.id {
                return Err(Error::Domain);
            }
            bytes = bytes
                .checked_add(
                    t.id.capacity()
                        + t.shape.capacity() * std::mem::size_of::<usize>()
                        + t.values.capacity() * 8,
                )
                .ok_or(Error::Resource)?;
            if bytes > MAX_BYTES {
                return Err(Error::Resource);
            }
            hash.update((t.id.len() as u64).to_le_bytes());
            hash.update(t.id.as_bytes());
            hash.update((t.shape.len() as u64).to_le_bytes());
            for &d in &t.shape {
                hash.update((d as u64).to_le_bytes());
            }
            for &v in t.values.iter() {
                hash.update(v.to_le_bytes());
            }
        }
        Ok(Self {
            tensors,
            rotary: Vec::new(),
            digest: hash.finalize().into(),
            bytes,
        })
    }
    pub fn with_rotary(mut self, mut descriptors: Vec<RotaryCoefficients>) -> Result<Self, Error> {
        if !self.rotary.is_empty() || descriptors.is_empty() || descriptors.len() > MAX_NODES {
            return Err(Error::Domain);
        }
        descriptors.sort_unstable_by(|a, b| a.operation.cmp(&b.operation));
        let mut hash = Sha256::new();
        hash.update(b"pllm.fss-decoder.pinned-f32-rotary-source.v1\0");
        hash.update(self.digest);
        self.bytes = checked_bytes(
            self.bytes,
            descriptors.capacity() * std::mem::size_of::<RotaryCoefficients>(),
        )?;
        for (i, r) in descriptors.iter().enumerate() {
            if i > 0 && descriptors[i - 1].operation == r.operation {
                return Err(Error::Domain);
            }
            self.bytes = checked_bytes(
                self.bytes,
                r.operation.capacity() + 8 * (r.cos.capacity() + r.sin.capacity()),
            )?;
            hash.update((r.operation.len() as u64).to_le_bytes());
            hash.update(r.operation.as_bytes());
            for v in [r.theta, r.positions as u64, r.dimensions as u64] {
                hash.update(v.to_le_bytes());
            }
            hash.update(r.ieee_digest);
            for &v in r.cos.iter().chain(r.sin.iter()) {
                hash.update(v.to_le_bytes());
            }
        }
        self.digest = hash.finalize().into();
        self.rotary = descriptors;
        Ok(self)
    }
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
    pub fn reservation_bytes(&self) -> usize {
        self.bytes
    }
    pub(super) fn find(&self, id: &str, shape: &[usize]) -> Result<usize, Error> {
        let i = self
            .tensors
            .binary_search_by(|t| t.id.as_str().cmp(id))
            .map_err(|_| Error::Domain)?;
        if self.tensors[i].shape != shape {
            return Err(Error::Domain);
        }
        Ok(i)
    }
}
fn checked_bytes(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b)
        .filter(|&n| n <= MAX_BYTES)
        .ok_or(Error::Resource)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn external_nearest_even_and_checked_source() {
        let t = Tensor::from_f64(
            "w",
            &[6],
            &[
                0.5 / 4096.,
                1.5 / 4096.,
                2.5 / 4096.,
                -0.5 / 4096.,
                -1.5 / 4096.,
                -2.5 / 4096.,
            ],
        )
        .unwrap();
        assert_eq!(t.values(), &[0, 2, 2, 0, -2, -2]);
        assert!(Tensor::from_f64("w", &[1], &[f64::NAN]).is_err());
        assert!(Tensor::from_q12("w", &[1], &[1i64 << 49]).is_err());
        let a = Source::new(vec![t]).unwrap();
        let b = Source::new(vec![
            Tensor::from_q12("w", &[6], &[0, 2, 2, 0, -2, -2]).unwrap()
        ])
        .unwrap();
        assert_eq!(a.digest(), b.digest());
    }

    #[test]
    fn external_ieee_formats_and_rotary_pins() {
        for (dtype, words) in [
            (WeightDtype::F32, [0x3f800000, 0xbf000000]),
            (WeightDtype::F16, [0x3c00, 0xb800]),
            (WeightDtype::Bf16, [0x3f80, 0xbf00]),
        ] {
            assert_eq!(
                Tensor::from_ieee("w", &[2], dtype, &words)
                    .unwrap()
                    .values(),
                &[4096, -2048]
            );
        }
        assert!(Tensor::from_ieee("w", &[1], WeightDtype::F32, &[0x7fc00000]).is_err());
        assert!(Tensor::from_ieee("w", &[1], WeightDtype::F16, &[0x10000]).is_err());
        assert!(RotaryCoefficients::from_f32("op", 10000, [1, 2], &[1., 1.], &[0., -0.]).is_err());
        assert!(RotaryCoefficients::from_f32(
            "op",
            10000,
            [2, 2],
            &[1., 1., 0.5, 0.5],
            &[0., 0., 0.25, 0.25000003]
        )
        .is_err());
        let a = RotaryCoefficients::from_f32(
            "op",
            10000,
            [2, 2],
            &[1., 1., 0.5, 0.5],
            &[0., 0., 0.25, 0.25],
        )
        .unwrap();
        let b = RotaryCoefficients::from_f32(
            "op",
            10000,
            [2, 2],
            &[1., 1., 0.50000006, 0.50000006],
            &[0., 0., 0.25, 0.25],
        )
        .unwrap();
        // Equal rounded Q12 payloads still retain distinct original F32 pins.
        assert_eq!(a.cos, b.cos);
        assert_ne!(a.ieee_digest(), b.ieee_digest());
    }
}
