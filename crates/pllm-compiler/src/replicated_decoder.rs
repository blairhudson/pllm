//! Source-bound replicated-three-party decoder compilation and execution.
//! The client/model owner supplies independent input/model shares. Only final
//! token IDs may be reconstructed after the fixed full-prefix feedback loop.

mod program;
mod runtime;
pub mod sdk;
mod source;
#[cfg(test)]
mod tests;
pub use program::{Program, SourceTensor};
pub use runtime::{PeerBytes, Worker};
pub use source::{Manifest, Publisher, SourceChunk};

use multiply::{arithmetic::Affine, tensor, InputBounds, Interval, MatrixShape};
use pllm_core::puma::{self, multiply, Error};
use sha2::{Digest as _, Sha256};
use std::sync::Arc;
use zeroize::Zeroizing;

const ONE: i64 = 1 << 18;
const MAX_VALUES: usize = 1 << 20;
const MAX_NODES: usize = 1 << 16;
const MAX_PROGRAM_BYTES: usize = 64 << 20;

#[derive(Clone)]
enum Code {
    Input(String),
    Gather(Vec<(usize, usize)>),
    Affine {
        left: usize,
        right: Option<usize>,
        coefficients: Affine,
    },
    Product {
        left: usize,
        right: usize,
        shape: Option<MatrixShape>,
        rescale: bool,
    },
    Tensor {
        input: usize,
        program: Arc<tensor::Program>,
    },
}

#[derive(Clone)]
struct Node {
    code: Code,
    lanes: usize,
    bounds: Interval,
}

impl Node {
    fn inputs(&self) -> Vec<usize> {
        match &self.code {
            Code::Input(_) => vec![],
            Code::Gather(items) => {
                let mut result: Vec<_> = items.iter().map(|v| v.0).collect();
                result.sort_unstable();
                result.dedup();
                result
            }
            Code::Affine { left, right, .. } => std::iter::once(*left).chain(*right).collect(),
            Code::Product { left, right, .. } => vec![*left, *right],
            Code::Tensor { input, .. } => vec![*input],
        }
    }
}

fn vec_zero<T: Default + Clone>(n: usize) -> Result<Vec<T>, Error> {
    let bytes = n
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(Error::Bounds)?;
    if bytes > MAX_PROGRAM_BYTES {
        return Err(Error::Bounds);
    }
    let mut value = Vec::new();
    value.try_reserve_exact(n).map_err(|_| Error::Allocation)?;
    if value.capacity() != n {
        return Err(Error::Allocation);
    }
    value.resize(n, T::default());
    Ok(value)
}

struct Secret {
    first: Zeroizing<Vec<u64>>,
    second: Zeroizing<Vec<u64>>,
}
impl Secret {
    fn new(n: usize) -> Result<Self, Error> {
        if n == 0 || n > MAX_VALUES {
            return Err(Error::Bounds);
        }
        Ok(Self {
            first: Zeroizing::new(vec_zero(n)?),
            second: Zeroizing::new(vec_zero(n)?),
        })
    }
}
