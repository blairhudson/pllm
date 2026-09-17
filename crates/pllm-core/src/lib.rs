//! Numeric primitives for private language model inference.
//!
//! This crate owns integer arithmetic, wire codecs, quantization, and mask
//! sampling. It has no Python dependency and does not implement an HE scheme.
//! Encryption and decryption remain the responsibility of the HE backend.
pub mod activation;
pub mod codec;
pub mod fixed_point;
pub mod kernels;
pub mod tensor;
pub use activation::{silu_quadratic_q7, silu_quadratic_q7_tensor};
pub use fixed_point::{
    gated_multiply_q7, multiply_q7, multiply_q7_tensor, rescale_q14_to_q7, rescale_q14_to_q7_tensor,
};
pub use kernels::{Executor, Matrix};
pub use tensor::add_wrap32;

#[cfg(test)]
mod tests;
