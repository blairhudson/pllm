//! Numeric primitives for private language model inference.
//!
//! This crate owns integer arithmetic, wire codecs, quantization, and mask
//! sampling. It has no Python dependency and does not implement an HE scheme.
//! Encryption and decryption remain the responsibility of the HE backend.
pub mod activation;
pub mod codec;
pub mod kernels;
pub use activation::{silu_quadratic_q7, silu_quadratic_q7_tensor};
pub use kernels::{Executor, Matrix};

#[cfg(test)]
mod tests;
