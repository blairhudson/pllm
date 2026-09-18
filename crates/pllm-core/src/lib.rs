//! Numeric primitives for private language model inference.
//!
//! This crate owns integer arithmetic, wire codecs, quantization, and mask
//! sampling. It has no Python dependency and does not implement an HE scheme.
//! Encryption and decryption remain the responsibility of the HE backend.
pub mod activation;
mod attention;
mod attention_values;
pub mod codec;
pub mod fixed_point;
pub mod kernels;
pub mod kv_cache;
pub mod rms_norm;
pub mod rope;
mod softmax;
pub mod tensor;
mod token_lookup;
pub use activation::{silu_quadratic_q7, silu_quadratic_q7_tensor};
pub use attention::{
    attention_scale_q30, attention_scores_q20, attention_scores_window_q20, AttentionScoreQ20Error,
    AttentionScoreQ20Layout, AttentionScoreQ20Policy, AttentionScoresQ20,
    ATTENTION_Q20_MAX_DECODED_ERROR, ATTENTION_Q20_MAX_ENCODED_ERROR_RAW,
    ATTENTION_Q20_MAX_HEAD_DIM, ATTENTION_Q20_MAX_MULTIPLY_ACCUMULATES,
    ATTENTION_Q20_MAX_SCORE_ELEMENTS, ATTENTION_Q30_COEFFICIENT_MAX_ERROR_RAW, ATTENTION_SCALE_Q30,
    ATTENTION_SCORE_Q20_LAYOUT, ATTENTION_SCORE_Q20_PROFILE,
};
pub use attention_values::{
    attention_values_q10, AttentionValueQ10Error, AttentionValueQ10Layout, AttentionValueQ10Policy,
    AttentionValuesQ10, ATTENTION_VALUE_Q10_MAX_MULTIPLY_ACCUMULATES,
    ATTENTION_VALUE_Q10_MAX_OUTPUT_ELEMENTS, ATTENTION_VALUE_Q10_MAX_ROUNDING_ERROR_RAW,
    ATTENTION_VALUE_Q30_Q10_PROFILE,
};
pub use fixed_point::{
    gated_multiply_q7, multiply_q7, multiply_q7_tensor, rescale_q14_to_q10,
    rescale_q14_to_q10_centered_u32, rescale_q14_to_q7, rescale_q14_to_q7_tensor,
    Q14_TO_Q10_INPUT_MAX, Q14_TO_Q10_INPUT_MIN, Q14_TO_Q10_PROFILE,
};
pub use kernels::{Executor, Matrix};
pub use kv_cache::{BoundedKvCacheQ10, KvCacheError, Q10KvCache};
pub use rms_norm::{
    rms_norm_f32_direct, rms_norm_q10_direct, rms_normalizer_q30, RmsNormError, RmsNormQ10Error,
};
pub use rope::{
    rope_q10, rope_q30_coefficients, RopeError, RopeQ10Config, RopeQ30Coefficient,
    ROPE_Q10_COEFFICIENT_PROFILE, ROPE_Q10_PROFILE, ROPE_Q30_COEFFICIENT_PROFILE,
};
pub use softmax::{
    softmax_q20_to_q30, SoftmaxProbabilitiesQ30, SoftmaxQ30Error, SoftmaxQ30Policy,
    SOFTMAX_Q20_TO_Q30_PROFILE, SOFTMAX_Q30_MAX_DECODED_ERROR, SOFTMAX_Q30_MAX_ELEMENTS,
    SOFTMAX_Q30_MAX_ROW_LENGTH, SOFTMAX_Q30_ONE,
};
pub use tensor::add_wrap32;
pub use token_lookup::{
    token_lookup_q10, TokenEmbeddingsQ10, TokenLookupQ10Error, TokenLookupQ10Policy,
    TOKEN_LOOKUP_Q10_MAX_OUTPUT_ELEMENTS, TOKEN_LOOKUP_Q10_MAX_WEIGHT_ELEMENTS,
    TOKEN_LOOKUP_Q10_PROFILE,
};

#[cfg(test)]
mod tests;
