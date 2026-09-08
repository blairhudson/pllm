#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

static inline uint64_t reciprocal_u64(uint32_t modulus) {
    return static_cast<uint64_t>((static_cast<__uint128_t>(1) << 64) / modulus);
}

static inline uint32_t reduce_unsigned_barrett(uint64_t value, uint32_t modulus, uint64_t reciprocal) {
    const uint64_t quotient = static_cast<uint64_t>((static_cast<__uint128_t>(value) * reciprocal) >> 64);
    uint64_t residue = value - quotient * static_cast<uint64_t>(modulus);
    // With floor(2^64 / modulus), the approximation is at most one quotient low
    // for our <2^63 accumulators. Keep the loop defensive for arbitrary callers.
    while (residue >= modulus) residue -= modulus;
    return static_cast<uint32_t>(residue);
}

static inline uint32_t reduce_signed_barrett(int64_t value, uint32_t modulus, uint64_t reciprocal) {
    if (value >= 0) {
        return reduce_unsigned_barrett(static_cast<uint64_t>(value), modulus, reciprocal);
    }
    const uint64_t magnitude = static_cast<uint64_t>(-static_cast<__int128_t>(value));
    const uint32_t residue = reduce_unsigned_barrett(magnitude, modulus, reciprocal);
    return residue == 0 ? 0 : modulus - residue;
}

static inline int64_t horizontal_sum_i32(__m256i value) {
    alignas(32) int32_t lanes[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), value);
    return static_cast<int64_t>(lanes[0]) + lanes[1] + lanes[2] + lanes[3]
         + lanes[4] + lanes[5] + lanes[6] + lanes[7];
}

static inline __m256i pack_u32_to_u16(__m256i low, __m256i high) {
    // AVX2 packs independently inside each 128-bit half. Reorder the 64-bit
    // groups so the final lanes retain input order 0..15.
    return _mm256_permute4x64_epi64(_mm256_packus_epi32(low, high), 0xD8);
}

static inline int64_t dot_i8_u32_limb12(
    const int8_t* weights,
    const uint32_t* inputs,
    size_t count
) {
    // A 4096-element flush keeps the worst-case 12-bit partial sum inside
    // signed int32 even for |w| <= 7. High limbs are smaller still.
    constexpr size_t kFlushElements = 4096;
    const __m256i mask = _mm256_set1_epi32(0xFFF);
    int64_t low_total = 0;
    int64_t high_total = 0;
    size_t index = 0;
    while (index + 16 <= count) {
        const size_t available = count - index;
        const size_t block_elements = std::min(kFlushElements, available - (available % 16));
        const size_t block_end = index + block_elements;
        __m256i low_sum = _mm256_setzero_si256();
        __m256i high_sum = _mm256_setzero_si256();
        for (; index < block_end; index += 16) {
            const __m128i w8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + index));
            const __m256i w16 = _mm256_cvtepi8_epi16(w8);
            const __m256i x0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(inputs + index));
            const __m256i x1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(inputs + index + 8));
            const __m256i low16 = pack_u32_to_u16(
                _mm256_and_si256(x0, mask), _mm256_and_si256(x1, mask)
            );
            const __m256i high16 = pack_u32_to_u16(
                _mm256_srli_epi32(x0, 12), _mm256_srli_epi32(x1, 12)
            );
            low_sum = _mm256_add_epi32(low_sum, _mm256_madd_epi16(w16, low16));
            high_sum = _mm256_add_epi32(high_sum, _mm256_madd_epi16(w16, high16));
        }
        low_total += horizontal_sum_i32(low_sum);
        high_total += horizontal_sum_i32(high_sum);
    }
    for (; index < count; ++index) {
        const int64_t coefficient = static_cast<int64_t>(weights[index]);
        low_total += coefficient * static_cast<int64_t>(inputs[index] & 0xFFFu);
        high_total += coefficient * static_cast<int64_t>(inputs[index] >> 12);
    }
    return low_total + high_total * 4096;
}

static inline int64_t dot_i8_u32(
    const int8_t* weights,
    const uint32_t* inputs,
    size_t count
) {
    __m256i sum_lo = _mm256_setzero_si256();
    __m256i sum_hi = _mm256_setzero_si256();
    size_t index = 0;
    for (; index + 8 <= count; index += 8) {
        const __m128i w8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(weights + index));
        const __m256i w32 = _mm256_cvtepi8_epi32(w8);
        const __m256i x32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(inputs + index));
        const __m256i products = _mm256_mullo_epi32(w32, x32);
        const __m128i products_low = _mm256_castsi256_si128(products);
        const __m128i products_high = _mm256_extracti128_si256(products, 1);
        sum_lo = _mm256_add_epi64(sum_lo, _mm256_cvtepi32_epi64(products_low));
        sum_hi = _mm256_add_epi64(sum_hi, _mm256_cvtepi32_epi64(products_high));
    }
    alignas(32) int64_t lanes_lo[4];
    alignas(32) int64_t lanes_hi[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes_lo), sum_lo);
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes_hi), sum_hi);
    int64_t total = lanes_lo[0] + lanes_lo[1] + lanes_lo[2] + lanes_lo[3]
                  + lanes_hi[0] + lanes_hi[1] + lanes_hi[2] + lanes_hi[3];
    for (; index < count; ++index) {
        total += static_cast<int64_t>(weights[index]) * static_cast<int64_t>(inputs[index]);
    }
    return total;
}

static inline int64_t dot_i8_i8(
    const int8_t* weights,
    const int8_t* inputs,
    size_t count
) {
    __m256i sum = _mm256_setzero_si256();
    size_t index = 0;
    for (; index + 32 <= count; index += 32) {
        const __m256i w8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights + index));
        const __m256i x8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(inputs + index));
        const __m256i w_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w8));
        const __m256i w_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w8, 1));
        const __m256i x_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(x8));
        const __m256i x_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x8, 1));
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(w_lo, x_lo));
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(w_hi, x_hi));
    }
    alignas(32) int32_t lanes[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), sum);
    int64_t total = static_cast<int64_t>(lanes[0]) + lanes[1] + lanes[2] + lanes[3]
                  + lanes[4] + lanes[5] + lanes[6] + lanes[7];
    for (; index < count; ++index) {
        total += static_cast<int64_t>(weights[index]) * static_cast<int64_t>(inputs[index]);
    }
    return total;
}

template <bool Modular>
void gemm_row_major_samples(
    const int8_t* weights,
    const uint32_t* inputs,
    uint32_t* outputs,
    size_t batch,
    size_t out_features,
    size_t in_features,
    uint32_t modulus,
    int threads
) {
    const uint64_t reciprocal = Modular ? reciprocal_u64(modulus) : 0;
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
    #pragma omp parallel for schedule(static) if(batch * out_features >= 256 && threads != 1)
#endif
    for (size_t row = 0; row < out_features; ++row) {
        const int8_t* weight = weights + row * in_features;
        for (size_t sample = 0; sample < batch; ++sample) {
            const int64_t total = dot_i8_u32(weight, inputs + sample * in_features, in_features);
            outputs[sample * out_features + row] = Modular
                ? reduce_signed_barrett(total, modulus, reciprocal)
                : static_cast<uint32_t>(total);
        }
    }
}

void gemm_row_major_limb12(
    const int8_t* weights,
    const uint32_t* inputs,
    uint32_t* outputs,
    size_t batch,
    size_t out_features,
    size_t in_features,
    uint32_t modulus,
    int threads
) {
    const uint64_t reciprocal = reciprocal_u64(modulus);
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
    #pragma omp parallel for schedule(static) if(batch * out_features >= 256 && threads != 1)
#endif
    for (size_t row = 0; row < out_features; ++row) {
        const int8_t* weight = weights + row * in_features;
        for (size_t sample = 0; sample < batch; ++sample) {
            const int64_t total = dot_i8_u32_limb12(
                weight, inputs + sample * in_features, in_features
            );
            outputs[sample * out_features + row] = reduce_signed_barrett(
                total, modulus, reciprocal
            );
        }
    }
}

void gemm_shared_weight_batch_modular(
    const int8_t* weights,
    const uint32_t* inputs,
    uint32_t* outputs,
    size_t batch,
    size_t out_features,
    size_t in_features,
    uint32_t modulus,
    int threads
) {
    const size_t blocks = (batch + 7) / 8;
    const size_t packed_batch = blocks * 8;
    std::vector<uint32_t> feature_major(packed_batch * in_features, 0);
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
    #pragma omp parallel for schedule(static) if(batch * in_features >= 65536 && threads != 1)
#endif
    for (size_t feature = 0; feature < in_features; ++feature) {
        uint32_t* destination = feature_major.data() + feature * packed_batch;
        for (size_t sample = 0; sample < batch; ++sample) {
            destination[sample] = inputs[sample * in_features + feature];
        }
    }

    const uint64_t reciprocal = reciprocal_u64(modulus);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(out_features >= 128 && threads != 1)
#endif
    for (size_t row = 0; row < out_features; ++row) {
        __m256i accum_even[8];
        __m256i accum_odd[8];
        for (size_t block = 0; block < blocks; ++block) {
            accum_even[block] = _mm256_setzero_si256();
            accum_odd[block] = _mm256_setzero_si256();
        }
        const int8_t* weight = weights + row * in_features;
        for (size_t feature = 0; feature < in_features; ++feature) {
            const __m256i coefficient = _mm256_set1_epi32(static_cast<int32_t>(weight[feature]));
            const uint32_t* values = feature_major.data() + feature * packed_batch;
            for (size_t block = 0; block < blocks; ++block) {
                const __m256i x = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(values + block * 8)
                );
                accum_even[block] = _mm256_add_epi64(
                    accum_even[block], _mm256_mul_epi32(x, coefficient)
                );
                const __m256i x_odd = _mm256_srli_epi64(x, 32);
                accum_odd[block] = _mm256_add_epi64(
                    accum_odd[block], _mm256_mul_epi32(x_odd, coefficient)
                );
            }
        }
        for (size_t block = 0; block < blocks; ++block) {
            alignas(32) int64_t even[4];
            alignas(32) int64_t odd[4];
            _mm256_store_si256(reinterpret_cast<__m256i*>(even), accum_even[block]);
            _mm256_store_si256(reinterpret_cast<__m256i*>(odd), accum_odd[block]);
            for (size_t lane = 0; lane < 4; ++lane) {
                const size_t even_sample = block * 8 + lane * 2;
                const size_t odd_sample = even_sample + 1;
                if (even_sample < batch) {
                    outputs[even_sample * out_features + row] = reduce_signed_barrett(
                        even[lane], modulus, reciprocal
                    );
                }
                if (odd_sample < batch) {
                    outputs[odd_sample * out_features + row] = reduce_signed_barrett(
                        odd[lane], modulus, reciprocal
                    );
                }
            }
        }
    }
}

void gemm_shared_weight_batch_wrap32(
    const int8_t* weights,
    const uint32_t* inputs,
    uint32_t* outputs,
    size_t batch,
    size_t out_features,
    size_t in_features,
    int threads
) {
    const size_t blocks = (batch + 7) / 8;
    const size_t packed_batch = blocks * 8;
    std::vector<uint32_t> feature_major(packed_batch * in_features, 0);
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
    #pragma omp parallel for schedule(static) if(batch * in_features >= 65536 && threads != 1)
#endif
    for (size_t feature = 0; feature < in_features; ++feature) {
        uint32_t* destination = feature_major.data() + feature * packed_batch;
        for (size_t sample = 0; sample < batch; ++sample) {
            destination[sample] = inputs[sample * in_features + feature];
        }
    }

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(out_features >= 128 && threads != 1)
#endif
    for (size_t row = 0; row < out_features; ++row) {
        __m256i accumulators[8];
        for (size_t block = 0; block < blocks; ++block) {
            accumulators[block] = _mm256_setzero_si256();
        }
        const int8_t* weight = weights + row * in_features;
        for (size_t feature = 0; feature < in_features; ++feature) {
            const __m256i coefficient = _mm256_set1_epi32(static_cast<int32_t>(weight[feature]));
            const uint32_t* values = feature_major.data() + feature * packed_batch;
            for (size_t block = 0; block < blocks; ++block) {
                const __m256i x = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(values + block * 8)
                );
                accumulators[block] = _mm256_add_epi32(
                    accumulators[block], _mm256_mullo_epi32(x, coefficient)
                );
            }
        }
        for (size_t block = 0; block < blocks; ++block) {
            alignas(32) uint32_t lanes[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), accumulators[block]);
            for (size_t lane = 0; lane < 8; ++lane) {
                const size_t sample = block * 8 + lane;
                if (sample < batch) outputs[sample * out_features + row] = lanes[lane];
            }
        }
    }
}

template <bool Modular>
void dispatch_gemm(
    const int8_t* weights,
    const uint32_t* inputs,
    uint32_t* outputs,
    size_t batch,
    size_t out_features,
    size_t in_features,
    uint32_t modulus,
    int threads
) {
    // For sustained service batches, reusing each weight across rows saves far
    // more memory traffic than the input transpose costs. Small batches retain
    // the lower-overhead AVX2 dot path.
    if (batch >= 8 && batch <= 64 && out_features >= 64) {
        if constexpr (Modular) {
            gemm_shared_weight_batch_modular(
                weights, inputs, outputs, batch, out_features, in_features, modulus, threads
            );
        } else {
            gemm_shared_weight_batch_wrap32(
                weights, inputs, outputs, batch, out_features, in_features, threads
            );
        }
    } else {
        gemm_row_major_samples<Modular>(
            weights, inputs, outputs, batch, out_features, in_features, modulus, threads
        );
    }
}

} // namespace

extern "C" void masked_gemm_i8_u32_mod(
    const int8_t* weights,
    const uint32_t* inputs,
    uint32_t* outputs,
    size_t batch,
    size_t out_features,
    size_t in_features,
    uint32_t modulus,
    int threads
) {
    dispatch_gemm<true>(weights, inputs, outputs, batch, out_features, in_features, modulus, threads);
}

// Backward-compatible symbol used by the earlier Round 8 wrapper.
extern "C" void masked_gemm_i8_mod_u32(
    const int8_t* weights,
    const uint32_t* inputs,
    uint32_t* outputs,
    size_t batch,
    size_t out_features,
    size_t in_features,
    uint32_t modulus,
    int threads
) {
    masked_gemm_i8_u32_mod(weights, inputs, outputs, batch, out_features, in_features, modulus, threads);
}

extern "C" void masked_gemm_i8_u32_mod_limb12(
    const int8_t* weights,
    const uint32_t* inputs,
    uint32_t* outputs,
    size_t batch,
    size_t out_features,
    size_t in_features,
    uint32_t modulus,
    int threads
) {
    gemm_row_major_limb12(
        weights, inputs, outputs, batch, out_features, in_features, modulus, threads
    );
}

extern "C" void masked_gemm_i8_u32_wrap(
    const int8_t* weights,
    const uint32_t* inputs,
    uint32_t* outputs,
    size_t batch,
    size_t out_features,
    size_t in_features,
    int threads
) {
    dispatch_gemm<false>(weights, inputs, outputs, batch, out_features, in_features, 0, threads);
}

extern "C" void clear_gemm_i8_i8_i32(
    const int8_t* weights,
    const int8_t* inputs,
    int32_t* outputs,
    size_t batch,
    size_t out_features,
    size_t in_features,
    int threads
) {
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
    #pragma omp parallel for schedule(static) if(batch * out_features >= 256 && threads != 1)
#endif
    for (size_t row = 0; row < out_features; ++row) {
        const int8_t* weight = weights + row * in_features;
        for (size_t sample = 0; sample < batch; ++sample) {
            outputs[sample * out_features + row] = static_cast<int32_t>(
                dot_i8_i8(weight, inputs + sample * in_features, in_features)
            );
        }
    }
}
