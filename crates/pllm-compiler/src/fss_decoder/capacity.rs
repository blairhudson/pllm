//! Explicit source/storage policy. This does not change the D13 arithmetic or
//! admit a complete protected program, its cumulative work, or its transport.
use super::{Error, MAX_BYTES, MAX_VALUES};
use serde::{Deserialize, Serialize};

pub const SOURCE_READ_BYTES: usize = 64 << 10;
pub const SOURCE_READ_VALUES: usize = SOURCE_READ_BYTES / 8;
pub const FILE_OWNER_BYTES: usize = 4096;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capacity {
    #[default]
    Capsule,
    Decoder,
}

impl Capacity {
    pub fn name(self) -> &'static str {
        match self {
            Self::Capsule => "capsule",
            Self::Decoder => "decoder",
        }
    }
    pub fn max_vocabulary(self) -> usize {
        match self {
            Self::Capsule => MAX_VALUES,
            Self::Decoder => 1 << 18,
        }
    }
    pub fn max_feature_width(self) -> usize {
        match self {
            Self::Capsule => MAX_VALUES,
            Self::Decoder => 8192,
        }
    }
    /// Logical graph geometry, not permission to allocate this many live values.
    pub fn max_graph_values(self) -> usize {
        match self {
            Self::Capsule => MAX_VALUES,
            Self::Decoder => 1 << 20,
        }
    }
    pub fn max_tensor_values(self) -> usize {
        match self {
            Self::Capsule => MAX_VALUES,
            Self::Decoder => (pllm_core::public_file::MAX_BYTES / 8) as usize,
        }
    }
    pub fn max_source_bytes(self) -> u64 {
        match self {
            Self::Capsule => MAX_BYTES as u64,
            Self::Decoder => 8 << 30,
        }
    }
    pub fn max_resident_bytes(self) -> usize {
        MAX_BYTES
    }
    pub fn source_elements(self, shape: &[usize]) -> Result<usize, Error> {
        if self == Self::Capsule {
            return super::elements(shape);
        }
        if shape.is_empty() || shape.len() > 2 || shape.contains(&0) {
            return Err(Error::Domain);
        }
        shape.iter().try_fold(1usize, |n, &d| {
            n.checked_mul(d)
                .filter(|&n| n <= self.max_tensor_values())
                .ok_or(Error::Resource)
        })
    }
    pub(crate) fn file_backed(self, shape: &[usize]) -> bool {
        self == Self::Decoder && shape.len() == 2
    }
}
