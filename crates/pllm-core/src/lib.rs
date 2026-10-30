//! Numeric primitives for private language model inference.
//!
//! This crate owns integer arithmetic, wire codecs, quantization, and mask
//! sampling. It has no Python dependency and does not implement an HE scheme.
//! Encryption and decryption remain the responsibility of the HE backend.
pub mod activation;
mod attention;
mod attention_values;
mod byte_dot;
pub mod cache_selection_reference;
pub mod ckks_layout;
pub mod codec;
pub mod coded_delegation_reference;
pub mod coded_linear;
pub mod coefficient_compression;
pub mod coefficient_conversion;
pub mod coefficient_extraction;
pub mod coefficient_layout;
pub mod coefficient_ntt;
pub mod coefficient_phase_reference;
pub mod coefficient_repacking;
pub mod coefficient_torus_conversion;
pub mod column_product_reference;
pub mod commutable_projection_reference;
pub mod segmented_affine_reference;
pub mod compact;
pub mod compact_product;
pub mod digit_gemm;
pub mod edge_contract;
pub mod field_float;
pub mod fixed_file;
pub mod fixed_point;
mod freivalds;
pub mod gkm;
pub mod head_index;
pub mod integer_bounds;
pub mod interleaved_kv_layout;
pub mod joint_gated_reference;
pub mod kernels;
pub mod kv_cache;
pub mod logrow_numeric;
pub mod mask_stream;
pub mod masked_aggregate;
pub mod maverick_batch;
pub mod offset_transport;
pub mod packed_embedding;
pub mod paged;
pub mod piecewise_gated_reference;
pub mod polynomial_softmax_reference;
pub mod prepared_stage;
pub mod progressive_head;
pub mod public_entropy;
pub mod public_file;
pub mod public_transforms;
pub mod puma;
pub mod residual_silu_reference;
pub mod retained_kv;
pub mod rms_norm;
pub mod rope;
pub mod row_memo;
pub mod row_fold;
pub mod row_normalization_reference;
pub mod slot_matrix_layout;
mod softmax;
pub mod streamed_matrix;
pub mod tensor;
mod token_lookup;
pub mod token_pruning;
pub mod torus_cmux;
pub mod wavelet_gated_reference;
pub use activation::{silu_quadratic_q7, silu_quadratic_q7_tensor};
pub use attention::{
    ATTENTION_Q20_MAX_DECODED_ERROR, ATTENTION_Q20_MAX_ENCODED_ERROR_RAW,
    ATTENTION_Q20_MAX_HEAD_DIM, ATTENTION_Q20_MAX_MULTIPLY_ACCUMULATES,
    ATTENTION_Q20_MAX_SCORE_ELEMENTS, ATTENTION_Q30_COEFFICIENT_MAX_ERROR_RAW, ATTENTION_SCALE_Q30,
    ATTENTION_SCORE_Q20_LAYOUT, ATTENTION_SCORE_Q20_PROFILE, AttentionScoreQ20Error,
    AttentionScoreQ20Layout, AttentionScoreQ20Policy, AttentionScoresQ20, attention_scale_q30,
    attention_scores_q20, attention_scores_window_q20,
};
pub use attention_values::{
    ATTENTION_VALUE_Q10_MAX_MULTIPLY_ACCUMULATES, ATTENTION_VALUE_Q10_MAX_OUTPUT_ELEMENTS,
    ATTENTION_VALUE_Q10_MAX_ROUNDING_ERROR_RAW, ATTENTION_VALUE_Q30_Q10_PROFILE,
    AttentionValueQ10Error, AttentionValueQ10Layout, AttentionValueQ10Policy, AttentionValuesQ10,
    attention_values_q10,
};
pub use coded_linear::{
    CODED_LINEAR_CHALLENGE_WEIGHT, CODED_LINEAR_FIELD, CODED_LINEAR_MAX_CLAIMS,
    CODED_LINEAR_REPETITIONS, CodedMatVecVerifier,
};
pub use compact::{CompactQ7Error, CompactQ7Piece, CompactQ7Profile, fit_compact_silu_q7};
pub use fixed_point::{
    Q14_TO_Q10_INPUT_MAX, Q14_TO_Q10_INPUT_MIN, Q14_TO_Q10_PROFILE, gated_multiply_q7, multiply_q7,
    multiply_q7_tensor, rescale_q14_to_q7, rescale_q14_to_q7_tensor, rescale_q14_to_q10,
    rescale_q14_to_q10_centered_u32, rescale_q14_to_q10_centered_u32_tensor,
};
pub use freivalds::{
    FREIVALDS_FIELD_MODULUS, FREIVALDS_MAX_BINDING_BYTES, FREIVALDS_MAX_CHECKS,
    FREIVALDS_MAX_PROCESS_SESSIONS, FREIVALDS_MAX_SOUNDNESS_BITS, FreivaldsError, FreivaldsPolicy,
    FreivaldsProjectionBatch, FreivaldsResourcePolicy, FreivaldsSession, FreivaldsVerifier,
    VerifiedAccumulatorBatch, import_freivalds_projections, prepare_freivalds_projections,
};
pub use kernels::{Executor, Matrix};
pub use kv_cache::{BoundedKvCacheQ10, KvCacheError, Q10KvCache};
pub use rms_norm::{
    RmsNormError, RmsNormQ10Error, rms_norm_f32_direct, rms_norm_q10_direct, rms_normalizer_q30,
};
pub use rope::{
    ROPE_Q10_COEFFICIENT_PROFILE, ROPE_Q10_PROFILE, ROPE_Q30_COEFFICIENT_PROFILE, RopeError,
    RopeQ10Config, RopeQ30Coefficient, rope_q10, rope_q30_coefficients,
};
pub use softmax::{
    SOFTMAX_Q20_TO_Q30_PROFILE, SOFTMAX_Q30_MAX_DECODED_ERROR, SOFTMAX_Q30_MAX_ELEMENTS,
    SOFTMAX_Q30_MAX_ROW_LENGTH, SOFTMAX_Q30_ONE, SoftmaxProbabilitiesQ30, SoftmaxQ30Error,
    SoftmaxQ30Policy, softmax_q20_to_q30,
};
pub use tensor::add_wrap32;
pub use token_lookup::{
    TOKEN_LOOKUP_Q10_MAX_OUTPUT_ELEMENTS, TOKEN_LOOKUP_Q10_MAX_WEIGHT_ELEMENTS,
    TOKEN_LOOKUP_Q10_PROFILE, TokenEmbeddingsQ10, TokenLookupQ10Error, TokenLookupQ10Policy,
    token_lookup_q10,
};

#[cfg(test)]
mod tests;

pub mod polynomial_schedule;
