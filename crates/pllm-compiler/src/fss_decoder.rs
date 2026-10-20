//! Bounded public-weight, two-worker FuseFSS decoder.
//!
//! Checkpoint values are immutable Q12 integers. Only this compiler derives
//! intermediate range obligations; callers cannot supply range certificates.
//! Workers execute one role through an external material/transport driver. No
//! worker-side preprocessing, sibling evaluation, or reconstruction is provided.
#[path = "fss_decoder/capacity.rs"]
pub mod capacity;
#[path = "fss_decoder/channel.rs"]
pub mod channel;
#[path = "fss_decoder/energy.rs"]
mod energy;
#[path = "fss_decoder/import.rs"]
pub mod import;
#[path = "fss_decoder/program.rs"]
mod program;
#[path = "fss_decoder/runtime.rs"]
mod runtime;
#[path = "fss_decoder/sdk.rs"]
pub mod sdk;
#[path = "fss_decoder/source.rs"]
mod source;
#[path = "fss_decoder/transport.rs"]
pub mod transport;
pub use capacity::Capacity;
pub use pllm_garble::fusefss_backend::Error;
pub use program::{Limits, Program, Resources};
pub use runtime::{Driver, MaterialVisitor, NativeOutput, TokenShares, Worker};
pub use source::{RotaryCoefficients, Source, Tensor, WeightDtype};

#[cfg(test)]
#[path = "fss_decoder/conformance.rs"]
mod conformance;

const MASK: u64 = (1u64 << 50) - 1;
const HALF: i128 = 1i128 << 49;
const Q: i128 = 4096;
const MAX_VALUES: usize = 32768;
const MAX_NODES: usize = 1024;
const MAX_BYTES: usize = 8 << 20;

fn vector<T>(n: usize) -> Result<Vec<T>, Error> {
    if n.checked_mul(std::mem::size_of::<T>())
        .ok_or(Error::Resource)?
        > MAX_BYTES
    {
        return Err(Error::Resource);
    }
    let mut values = Vec::new();
    values.try_reserve_exact(n).map_err(|_| Error::Allocation)?;
    if values.capacity() != n {
        return Err(Error::Resource);
    }
    Ok(values)
}
fn copied<T: Copy>(values: &[T]) -> Result<Vec<T>, Error> {
    let mut out = vector(values.len())?;
    out.extend_from_slice(values);
    Ok(out)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Range {
    lo: i64,
    hi: i64,
}
impl Range {
    fn new(lo: i128, hi: i128) -> Result<Self, Error> {
        if lo > hi || lo < -HALF || hi >= HALF {
            return Err(Error::Domain);
        }
        Ok(Self {
            lo: lo as i64,
            hi: hi as i64,
        })
    }
    fn add(self, b: Self) -> Result<Self, Error> {
        Self::new(
            self.lo as i128 + b.lo as i128,
            self.hi as i128 + b.hi as i128,
        )
    }
    fn product(self, b: Self) -> Result<Self, Error> {
        let p = [
            self.lo as i128 * b.lo as i128,
            self.lo as i128 * b.hi as i128,
            self.hi as i128 * b.lo as i128,
            self.hi as i128 * b.hi as i128,
        ];
        Self::new(*p.iter().min().unwrap(), *p.iter().max().unwrap())
    }
    fn scale(self, b: i64) -> Result<Self, Error> {
        self.product(Self::new(b as i128, b as i128)?)
    }
    fn floor(self) -> Result<Self, Error> {
        Self::new(
            (self.lo as i128).div_euclid(Q),
            (self.hi as i128).div_euclid(Q),
        )
    }
    fn gate(self) -> pllm_garble::fusefss_backend::composition::gates::InputRange {
        pllm_garble::fusefss_backend::composition::gates::InputRange {
            min: self.lo,
            max: self.hi,
        }
    }
}

fn elements(shape: &[usize]) -> Result<usize, Error> {
    if shape.is_empty() || shape.len() > 4 || shape.iter().any(|&n| n == 0) {
        return Err(Error::Domain);
    }
    shape.iter().try_fold(1usize, |p, &n| {
        p.checked_mul(n)
            .filter(|&n| n <= MAX_VALUES)
            .ok_or(Error::Resource)
    })
}
