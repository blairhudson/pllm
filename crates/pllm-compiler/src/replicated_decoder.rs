//! Source-bound replicated-three-party decoder compilation and execution.
//! The client/model owner supplies independent input/model shares. Only final
//! token IDs may be reconstructed after the fixed full-prefix feedback loop.

mod program;
mod public_source;
mod runtime;
pub mod sdk;
mod source;
#[cfg(test)]
mod tests;
mod views;
pub use program::{Program, SourceTensor};
pub use public_source::{PublicSource, PublicSourceImport};
pub use runtime::{PeerBytes, Worker};
pub use source::{Manifest, Publisher, SourceChunk};

use multiply::{arithmetic::Affine, tensor, InputBounds, Interval, MatrixShape};
use pllm_core::puma::{self, multiply, Error};
use sha2::{Digest as _, Sha256};
use std::sync::Arc;
use views::Words;
use zeroize::Zeroizing;

const ONE: i64 = 1 << 18;
const MAX_VALUES: usize = 1 << 20;
const MAX_NODES: usize = 1 << 16;
const MAX_PROGRAM_BYTES: usize = 64 << 20;
const MAX_PUBLIC_SOURCE_VALUES: usize = 1 << 29;

/// Public weights are an explicit open-checkpoint adaptation. The omitted
/// selector retains the original private-weight manifest and execution identity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightMode {
    #[default]
    Private,
    Public,
}
impl WeightMode {
    fn is_private(&self) -> bool {
        *self == Self::Private
    }
}

#[derive(Clone)]
enum Code {
    Input(String),
    PublicInput {
        key: String,
        digest: [u8; 32],
    },
    PublicValues(Vec<i64>),
    Gather(Words),
    SourceGather(SourceGather),
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
    PublicProduct {
        input: usize,
        weight: usize,
        rows: usize,
        inner: usize,
        columns: usize,
        transposed: bool,
    },
    Tensor {
        input: usize,
        program: Arc<tensor::Program>,
    },
    TensorLoop {
        input: LoopInput,
        program: Arc<tensor::Program>,
        tail: Option<Arc<tensor::Program>>,
    },
}

/// Public-geometry-only loops. TokenOffsets forms the complete private one-hot
/// input; Slice maps an elementwise primitive in original element order.
#[derive(Clone, Copy)]
enum LoopInput {
    Slice(usize),
    TokenOffsets { owner: usize, index: usize },
}
impl LoopInput {
    fn owner(self) -> usize {
        match self {
            Self::Slice(id) | Self::TokenOffsets { owner: id, .. } => id,
        }
    }
}

/// Public row-major view of one immutable source owner. Only the bounded
/// primitive tile is copied at execution; the compiler retains no per-weight
/// reference list. The logical gather order remains part of the program digest.
#[derive(Clone, Copy)]
struct SourceGather {
    owner: usize,
    offset: usize,
    rows: usize,
    columns: usize,
    row_stride: usize,
    column_stride: usize,
}

impl SourceGather {
    fn elements(self) -> Result<usize, Error> {
        if self.rows == 0 || self.columns == 0 {
            return Err(Error::Shape);
        }
        self.rows.checked_mul(self.columns).ok_or(Error::Bounds)
    }

    fn validate(self, lanes: usize) -> Result<usize, Error> {
        let n = self.elements()?;
        if n > puma::MAX_LANES {
            return Err(Error::Bounds);
        }
        let last = self
            .offset
            .checked_add(
                (self.rows - 1)
                    .checked_mul(self.row_stride)
                    .ok_or(Error::Bounds)?,
            )
            .and_then(|v| v.checked_add((self.columns - 1).checked_mul(self.column_stride)?))
            .ok_or(Error::Bounds)?;
        if last >= lanes {
            return Err(Error::Shape);
        }
        Ok(n)
    }

    /// Called only after validating all endpoints with checked arithmetic.
    fn indices(self) -> impl Iterator<Item = usize> {
        (0..self.rows).flat_map(move |r| {
            (0..self.columns)
                .map(move |c| self.offset + r * self.row_stride + c * self.column_stride)
        })
    }
}

#[derive(Clone)]
struct Node {
    code: Code,
    lanes: usize,
    bounds: Interval,
}

impl Node {
    fn work_units(&self) -> u64 {
        match &self.code {
            Code::Product { .. } => 1,
            Code::Tensor { program, .. } => program.instructions() as u64 * 128,
            Code::TensorLoop { program, tail, .. } => {
                (self.lanes / program.lanes()) as u64 * program.instructions() as u64 * 128
                    + tail.as_ref().map_or(0, |p| p.instructions() as u64 * 128)
            }
            _ => 0,
        }
    }

    fn inputs(&self) -> Vec<usize> {
        match &self.code {
            Code::Input(_) | Code::PublicInput { .. } | Code::PublicValues(_) => vec![],
            Code::Gather(items) => {
                let mut result: Vec<_> = items.owners().collect();
                result.sort_unstable();
                result.dedup();
                result
            }
            Code::SourceGather(view) => vec![view.owner],
            Code::Affine { left, right, .. } => std::iter::once(*left).chain(*right).collect(),
            Code::Product { left, right, .. } => vec![*left, *right],
            Code::PublicProduct { input, weight, .. } => vec![*input, *weight],
            Code::Tensor { input, .. } => vec![*input],
            Code::TensorLoop { input, .. } => vec![input.owner()],
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
