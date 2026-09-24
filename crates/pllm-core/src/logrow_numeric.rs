//! Public, bounded numeric profile for a 9-bit LogRow SiLU lookup.
//! The 257 signed input codes span [-M, M]. The returned signed 9-bit code
//! represents SiLU(x)/M. This is a numeric reference, not protected execution.

use std::{error::Error, fmt};

use sha2::{Digest, Sha256};

pub const LOGROW_SCALED_SILU_Q7_PROFILE: &str = "pllm.numeric.logrow_scaled_silu_q7.reference.v1";
pub const LOGROW_SCALED_SILU_MIN_RANGE: u8 = 1;
pub const LOGROW_SCALED_SILU_MAX_RANGE: u8 = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScaledSiluQ7Error {
    RangeOutOfBounds { max_abs: u8 },
    FloatInputOutOfRange,
    EncodedInputOutOfRange { value: i16 },
    EncodedOutputOutOfRange { value: i16 },
}

impl fmt::Display for ScaledSiluQ7Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl Error for ScaledSiluQ7Error {}

/// Integer range is public and set offline. It may not be derived from the
/// private request's maximum activation or chosen per element at runtime.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScaledSiluQ7Profile {
    max_abs: u8,
    digest: [u8; 32],
}

impl ScaledSiluQ7Profile {
    pub fn new(max_abs: u8) -> Result<Self, ScaledSiluQ7Error> {
        if !(LOGROW_SCALED_SILU_MIN_RANGE..=LOGROW_SCALED_SILU_MAX_RANGE).contains(&max_abs) {
            return Err(ScaledSiluQ7Error::RangeOutOfBounds { max_abs });
        }
        let mut hasher = Sha256::new();
        hasher.update(LOGROW_SCALED_SILU_Q7_PROFILE.as_bytes());
        hasher.update([0]);
        hasher.update(Sha256::digest(include_bytes!("logrow_numeric.rs")));
        hasher.update([max_abs]);
        Ok(Self {
            max_abs,
            digest: hasher.finalize().into(),
        })
    }

    pub const fn max_abs(&self) -> u8 {
        self.max_abs
    }

    pub const fn digest(&self) -> [u8; 32] {
        self.digest
    }

    pub fn encode_float32(&self, value: f32) -> Result<i16, ScaledSiluQ7Error> {
        if !value.is_finite() || f64::from(value).abs() > f64::from(self.max_abs) {
            return Err(ScaledSiluQ7Error::FloatInputOutOfRange);
        }
        Ok((f64::from(value) * 128.0 / f64::from(self.max_abs)).round_ties_even() as i16)
    }

    pub fn encoded_output(&self, input: i16) -> Result<i16, ScaledSiluQ7Error> {
        if !(-128..=128).contains(&input) {
            return Err(ScaledSiluQ7Error::EncodedInputOutOfRange { value: input });
        }
        let x = f64::from(input) * f64::from(self.max_abs) / 128.0;
        let silu = x / (1.0 + (-x).exp());
        let code = (silu * 128.0 / f64::from(self.max_abs)).round_ties_even() as i16;
        if !(-128..=128).contains(&code) {
            return Err(ScaledSiluQ7Error::EncodedOutputOutOfRange { value: code });
        }
        Ok(code)
    }

    pub fn decode_float32(&self, value: i16) -> Result<f32, ScaledSiluQ7Error> {
        if !(-128..=128).contains(&value) {
            return Err(ScaledSiluQ7Error::EncodedOutputOutOfRange { value });
        }
        Ok(f32::from(value) * f32::from(self.max_abs) / 128.0)
    }

    /// Conservative input-and-output quantization bound relative to exact SiLU
    /// for inputs within [-M, M], excluding float32 result rounding (<2e-6).
    /// The derivative of SiLU has absolute value below 1.1 on this domain.
    pub fn maximum_absolute_error_bound(&self) -> f64 {
        2.1 * f64::from(self.max_abs) / 256.0 + 2e-6
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_is_source_and_public_range_bound() {
        assert!(ScaledSiluQ7Profile::new(0).is_err());
        assert!(ScaledSiluQ7Profile::new(17).is_err());
        assert_ne!(
            ScaledSiluQ7Profile::new(1).unwrap().digest(),
            ScaledSiluQ7Profile::new(8).unwrap().digest()
        );
        assert_eq!(
            ScaledSiluQ7Profile::new(8).unwrap().digest(),
            ScaledSiluQ7Profile::new(8).unwrap().digest()
        );
    }

    #[test]
    fn full_table_and_float_inputs_obey_the_conservative_error_bound() {
        for max_abs in [1, 2, 4, 8, 16] {
            let profile = ScaledSiluQ7Profile::new(max_abs).unwrap();
            for code in -128..=128 {
                let output = profile.encoded_output(code).unwrap();
                assert!((-128..=128).contains(&output));
            }
            for step in -4096..=4096 {
                let value = step as f32 * f32::from(max_abs) / 4096.0;
                let encoded = profile.encode_float32(value).unwrap();
                let output = profile
                    .decode_float32(profile.encoded_output(encoded).unwrap())
                    .unwrap();
                let exact = f64::from(value) / (1.0 + (-f64::from(value)).exp());
                assert!(
                    (f64::from(output) - exact).abs() <= profile.maximum_absolute_error_bound(),
                    "max_abs={max_abs} x={value} output={output} exact={exact}"
                );
            }
        }
    }

    #[test]
    fn input_and_output_domains_fail_closed_without_saturation() {
        let profile = ScaledSiluQ7Profile::new(4).unwrap();
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -4.01, 4.01] {
            assert!(profile.encode_float32(value).is_err());
        }
        assert_eq!(profile.encode_float32(4.0), Ok(128));
        assert_eq!(profile.encode_float32(-4.0), Ok(-128));
        assert_eq!(profile.encode_float32(0.015625), Ok(0)); // half-Q7 ties to even
        assert!(profile.encoded_output(-129).is_err());
        assert!(profile.encoded_output(129).is_err());
        assert!(profile.decode_float32(256).is_err());
    }
}
