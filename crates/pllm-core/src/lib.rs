//! Numeric primitives for private language model inference.
//!
//! This crate owns integer arithmetic, wire codecs, quantization, and mask
//! sampling. It has no Python dependency and does not implement an HE scheme.
//! Encryption and decryption remain the responsibility of the HE backend.
pub mod codec;
pub mod kernels;
pub use kernels::{Executor, Matrix};

#[cfg(test)]
mod tests;
