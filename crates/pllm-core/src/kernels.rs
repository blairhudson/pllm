//! Exact integer kernels. These do not implement an encryption scheme.
//! SIMD entry points have a runtime feature check and a scalar reference.
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

pub const MAX_THREADS: usize = 32;

pub struct Executor {
    pool: Option<ThreadPool>,
    threads: usize,
    simd: bool,
}

impl Executor {
    pub fn new(threads: usize, simd: bool) -> Result<Self, String> {
        if !(1..=MAX_THREADS).contains(&threads) {
            return Err("threads must be in 1..=32".into());
        }
        let pool = if threads == 1 {
            None
        } else {
            Some(
                ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .map_err(|e| e.to_string())?,
            )
        };
        Ok(Self {
            pool,
            threads,
            simd: simd && has_simd(),
        })
    }
    /// Number of workers requested for this persistent executor.
    pub fn threads(&self) -> usize {
        self.threads
    }
    /// Whether supported SIMD instructions are enabled on this host.
    pub fn simd(&self) -> bool {
        self.simd
    }
    fn rows<T: Send, F: Fn(usize, &mut [T]) + Send + Sync>(
        &self,
        data: &mut [T],
        width: usize,
        f: F,
    ) {
        if data.is_empty() {
            return;
        }
        if let Some(pool) = &self.pool {
            pool.install(|| {
                data.par_chunks_mut(width)
                    .enumerate()
                    .for_each(|(i, row)| f(i, row))
            });
        } else {
            for (i, row) in data.chunks_mut(width).enumerate() {
                f(i, row);
            }
        }
    }
}

pub fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

pub fn has_neon() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("neon")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}
pub fn has_simd() -> bool {
    has_avx2() || has_neon()
}

#[derive(Clone)]
pub struct Matrix {
    weights: Vec<i8>,
    rows: usize,
    cols: usize,
    max_weight: u64,
}
impl Matrix {
    pub fn new(bytes: &[u8], rows: usize, cols: usize) -> Result<Self, String> {
        if rows == 0 || cols == 0 || rows.checked_mul(cols) != Some(bytes.len()) {
            return Err("weights must be a nonempty [out,in] int8 matrix".into());
        }
        let mut weights = Vec::new();
        weights
            .try_reserve_exact(bytes.len())
            .map_err(|_| "matrix weight allocation failed")?;
        weights.extend(bytes.iter().map(|&value| value as i8));
        let max_weight = weights
            .iter()
            .map(|&x| (x as i16).unsigned_abs() as u64)
            .max()
            .unwrap_or(0);
        Ok(Self {
            weights,
            rows,
            cols,
            max_weight,
        })
    }
    /// Matrix dimensions in output, input order.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
    /// Resident bytes of immutable signed weights.
    pub fn weight_bytes(&self) -> usize {
        self.weights.len()
    }
    fn output_size(&self, input_len: usize, batch: usize) -> Result<usize, String> {
        if batch.checked_mul(self.cols) != Some(input_len) {
            return Err("input must have shape [batch,in]".into());
        }
        batch
            .checked_mul(self.rows)
            .ok_or_else(|| "output size overflow".into())
    }
    fn validate(&self, input_len: usize, batch: usize, max_input: u64) -> Result<usize, String> {
        let size = self.output_size(input_len, batch)?;
        if (self.cols as u128) * (self.max_weight as u128) * (max_input as u128) > i64::MAX as u128
        {
            return Err("dot product bound exceeds int64".into());
        }
        Ok(size)
    }
    /// Canonical residues in 2 <= p < 2^31. Multiplications widen BEFORE accumulating.
    pub fn modular(
        &self,
        executor: &Executor,
        x: &[u32],
        batch: usize,
        p: u32,
    ) -> Result<Vec<u32>, String> {
        if !(2..(1u32 << 31)).contains(&p) {
            return Err("modulus must satisfy 2 <= p < 2^31".into());
        }
        let size = self.validate(x.len(), batch, (p - 1) as u64)?;
        if x.iter().any(|&v| v >= p) {
            return Err("input contains a noncanonical residue".into());
        }
        if size == 0 {
            return Ok(Vec::new());
        }
        let mut transposed = zeroed_vec(size, "modular matrix output")?;
        let reducer = Reducer::new(p);
        executor.rows(&mut transposed, batch, |row, outputs| {
            let w = &self.weights[row * self.cols..(row + 1) * self.cols];
            for (b, output) in outputs.iter_mut().enumerate() {
                let input = &x[b * self.cols..(b + 1) * self.cols];
                let total = dot(w, input, p, executor.simd);
                *output = reducer.reduce(total);
            }
        });
        transpose_result(transposed, self.rows, batch)
    }
    pub fn wrap32(&self, executor: &Executor, x: &[u32], batch: usize) -> Result<Vec<u32>, String> {
        let size = self.output_size(x.len(), batch)?;
        if size == 0 {
            return Ok(Vec::new());
        }
        let mut transposed = zeroed_vec(size, "wrap32 matrix output")?;
        executor.rows(&mut transposed, batch, |row, outputs| {
            let w = &self.weights[row * self.cols..(row + 1) * self.cols];
            for (b, output) in outputs.iter_mut().enumerate() {
                let input = &x[b * self.cols..(b + 1) * self.cols];
                *output = dot_wrap32(w, input, executor.simd);
            }
        });
        transpose_result(transposed, self.rows, batch)
    }
    pub fn wrap64(&self, executor: &Executor, x: &[u64], batch: usize) -> Result<Vec<u64>, String> {
        let size = self.output_size(x.len(), batch)?;
        if size == 0 {
            return Ok(Vec::new());
        }
        let mut transposed = zeroed_vec(size, "wrap64 matrix output")?;
        executor.rows(&mut transposed, batch, |row, outputs| {
            let w = &self.weights[row * self.cols..(row + 1) * self.cols];
            for (b, output) in outputs.iter_mut().enumerate() {
                let input = &x[b * self.cols..(b + 1) * self.cols];
                *output = dot_wrap64(w, input, executor.simd);
            }
        });
        transpose_result(transposed, self.rows, batch)
    }
    pub fn clear(&self, executor: &Executor, x: &[i8], batch: usize) -> Result<Vec<i32>, String> {
        let size = self.validate(x.len(), batch, 128)?;
        if self.cols as u128 * self.max_weight as u128 * 128 > i32::MAX as u128 {
            return Err("clear int32 result cannot represent this matrix's worst case".into());
        }
        if size == 0 {
            return Ok(Vec::new());
        }
        let mut transposed = zeroed_vec(size, "clear matrix output")?;
        executor.rows(&mut transposed, batch, |row, outputs| {
            let w = &self.weights[row * self.cols..(row + 1) * self.cols];
            for (b, output) in outputs.iter_mut().enumerate() {
                let input = &x[b * self.cols..(b + 1) * self.cols];
                *output = dot_clear(w, input, executor.simd) as i32;
            }
        });
        transpose_result(transposed, self.rows, batch)
    }
    /// Coefficient matrix C = W A mod q. q is a ciphertext modulus, NOT p.
    /// Limb work is tiled to bound memory; all intermediates are integers.
    pub fn coefficients(
        &self,
        executor: &Executor,
        a: &[u64],
        columns: usize,
        q: u64,
    ) -> Result<Vec<u64>, String> {
        if !(2..(1u64 << 54)).contains(&q) {
            return Err("q must satisfy 2 <= q < 2^54".into());
        }
        if columns == 0 || self.cols.checked_mul(columns) != Some(a.len()) {
            return Err("coefficient matrix shape mismatch".into());
        }
        if a.iter().any(|&v| v >= q) {
            return Err("coefficient exceeds q".into());
        }
        if self.cols as u128 * self.max_weight as u128 * 128 > i32::MAX as u128 {
            return Err("signed limb dot product exceeds int32".into());
        }
        let mut out = vec![
            0u64;
            self.rows
                .checked_mul(columns)
                .ok_or("output size overflow")?
        ];
        let limbs = (64 - (q - 1).leading_zeros() as usize).div_ceil(8);
        // One signed digit matrix at a time, rather than seven full matrices.
        for limb in (0..limbs).rev() {
            let digits: Vec<i8> = a
                .iter()
                .map(|&v| (((v >> (8 * limb)) & 255) as i16 - 128) as i8)
                .collect();
            executor.rows(&mut out, columns, |row, output| {
                let w = &self.weights[row * self.cols..(row + 1) * self.cols];
                let correction: i64 = 128 * w.iter().map(|&v| v as i64).sum::<i64>();
                for start in (0..columns).step_by(64) {
                    let end = (start + 64).min(columns);
                    let mut sums = [0i32; 64];
                    for (i, &wi) in w.iter().enumerate() {
                        let input = &digits[i * columns + start..i * columns + end];
                        for (sum, &digit) in sums.iter_mut().zip(input) {
                            *sum += wi as i32 * digit as i32;
                        }
                    }
                    for (offset, value) in output[start..end].iter_mut().enumerate() {
                        let acc = *value as i128 * 256 + sums[offset] as i128 + correction as i128;
                        *value = acc.rem_euclid(q as i128) as u64;
                    }
                }
            });
        }
        Ok(out)
    }
}
fn zeroed_vec<T: Clone + Default>(len: usize, name: &str) -> Result<Vec<T>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| format!("{name} allocation failed"))?;
    values.resize(len, T::default());
    Ok(values)
}

fn transpose_result<T: Copy + Default>(
    input: Vec<T>,
    rows: usize,
    batch: usize,
) -> Result<Vec<T>, String> {
    if batch == 1 {
        return Ok(input);
    }
    let mut out = zeroed_vec(input.len(), "matrix transpose output")?;
    for i in 0..rows {
        for b in 0..batch {
            out[b * rows + i] = input[i * batch + b];
        }
    }
    Ok(out)
}
struct Reducer {
    p: u64,
    reciprocal: u64,
}
impl Reducer {
    fn new(p: u32) -> Self {
        Self {
            p: p as u64,
            reciprocal: ((1u128 << 64) / p as u128) as u64,
        }
    }
    fn reduce(&self, value: i64) -> u32 {
        let mag = value.unsigned_abs();
        let quotient = ((mag as u128 * self.reciprocal as u128) >> 64) as u64;
        let mut r = mag - quotient * self.p;
        if r >= self.p {
            r -= self.p;
        }
        if value < 0 && r != 0 {
            (self.p - r) as u32
        } else {
            r as u32
        }
    }
}
fn dot(w: &[i8], x: &[u32], p: u32, simd: bool) -> i64 {
    #[cfg(target_arch = "x86_64")]
    if simd {
        // SAFETY: caller selected AVX2 with is_x86_feature_detected; slice lengths match.
        return unsafe {
            if p <= (1 << 24) {
                x86::dot_limb12(w, x)
            } else {
                x86::dot_wide(w, x)
            }
        };
    }
    #[cfg(target_arch = "aarch64")]
    if simd {
        // SAFETY: the executor checked NEON availability and validated slice lengths.
        return unsafe { arm::dot_wide(w, x) };
    }
    let _ = (p, simd);
    w.iter()
        .zip(x)
        .map(|(&wi, &xi)| wi as i64 * xi as i64)
        .sum()
}
fn dot_clear(w: &[i8], x: &[i8], simd: bool) -> i64 {
    #[cfg(target_arch = "x86_64")]
    if simd {
        return unsafe { x86::dot_clear(w, x) };
    }
    #[cfg(target_arch = "aarch64")]
    if simd {
        // SAFETY: the executor checked NEON availability and validated slice lengths.
        return unsafe { arm::dot_clear(w, x) };
    }
    let _ = simd;
    w.iter()
        .zip(x)
        .map(|(&wi, &xi)| wi as i64 * xi as i64)
        .sum()
}
fn dot_wrap32(w: &[i8], x: &[u32], simd: bool) -> u32 {
    #[cfg(target_arch = "x86_64")]
    if simd {
        return unsafe { x86::dot_wrap32(w, x) };
    }
    #[cfg(target_arch = "aarch64")]
    if simd {
        return unsafe { arm::dot_wrap32(w, x) };
    }
    let _ = simd;
    w.iter().zip(x).fold(0u32, |acc, (&weight, &input)| {
        acc.wrapping_add((weight as i32 as u32).wrapping_mul(input))
    })
}
fn dot_wrap64(w: &[i8], x: &[u64], simd: bool) -> u64 {
    #[cfg(target_arch = "x86_64")]
    if simd {
        return unsafe { x86::dot_wrap64(w, x) };
    }
    // NEON has no packed u64 multiply. Its 32-bit limb emulation measures
    // slower than this scalar loop on Apple Silicon; AVX2 remains vectorized.
    let _ = simd;
    w.iter().zip(x).fold(0u64, |acc, (&weight, &input)| {
        acc.wrapping_add((weight as i64 as u64).wrapping_mul(input))
    })
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;
    #[target_feature(enable = "avx2")]
    unsafe fn sum32(v: __m256i) -> i64 {
        let mut lanes = [0i32; 8];
        _mm256_storeu_si256(lanes.as_mut_ptr().cast(), v);
        lanes.iter().map(|&v| v as i64).sum()
    }
    #[target_feature(enable = "avx2")]
    unsafe fn sum64(v: __m256i) -> i64 {
        let mut lanes = [0i64; 4];
        _mm256_storeu_si256(lanes.as_mut_ptr().cast(), v);
        lanes.iter().sum()
    }
    #[target_feature(enable = "avx2")]
    unsafe fn pack16(a: __m256i, b: __m256i) -> __m256i {
        _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(a, b))
    }
    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_limb12(w: &[i8], x: &[u32]) -> i64 {
        debug_assert_eq!(w.len(), x.len());
        let mask = _mm256_set1_epi32(4095);
        let (mut index, mut low, mut high) = (0, 0i64, 0i64);
        while index + 16 <= w.len() {
            // Flush before int32 lane sums can overflow, including w=-128.
            let end = (index + 2048).min(w.len() / 16 * 16);
            let mut lo = _mm256_setzero_si256();
            let mut hi = _mm256_setzero_si256();
            while index < end {
                let wi = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.as_ptr().add(index).cast()));
                let a = _mm256_loadu_si256(x.as_ptr().add(index).cast());
                let b = _mm256_loadu_si256(x.as_ptr().add(index + 8).cast());
                let lv = pack16(_mm256_and_si256(a, mask), _mm256_and_si256(b, mask));
                let hv = pack16(_mm256_srli_epi32::<12>(a), _mm256_srli_epi32::<12>(b));
                lo = _mm256_add_epi32(lo, _mm256_madd_epi16(wi, lv));
                hi = _mm256_add_epi32(hi, _mm256_madd_epi16(wi, hv));
                index += 16;
            }
            low += sum32(lo);
            high += sum32(hi);
        }
        for i in index..w.len() {
            low += w[i] as i64 * (x[i] & 4095) as i64;
            high += w[i] as i64 * (x[i] >> 12) as i64;
        }
        low + high * 4096
    }
    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_wide(w: &[i8], x: &[u32]) -> i64 {
        let (mut i, mut a, mut b) = (0, _mm256_setzero_si256(), _mm256_setzero_si256());
        while i + 8 <= w.len() {
            let wi = _mm256_cvtepi8_epi32(_mm_loadl_epi64(w.as_ptr().add(i).cast()));
            let xi = _mm256_loadu_si256(x.as_ptr().add(i).cast());
            // Inputs are <2^31; signed 32x32 -> 64 produces the full product.
            a = _mm256_add_epi64(a, _mm256_mul_epi32(wi, xi));
            b = _mm256_add_epi64(
                b,
                _mm256_mul_epi32(_mm256_srli_epi64::<32>(wi), _mm256_srli_epi64::<32>(xi)),
            );
            i += 8;
        }
        sum64(a)
            + sum64(b)
            + w[i..]
                .iter()
                .zip(&x[i..])
                .map(|(&wi, &xi)| wi as i64 * xi as i64)
                .sum::<i64>()
    }
    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_clear(w: &[i8], x: &[i8]) -> i64 {
        let (mut i, mut total) = (0, 0i64);
        while i + 32 <= w.len() {
            let end = (i + 2048).min(w.len() / 32 * 32);
            let mut sum = _mm256_setzero_si256();
            while i < end {
                let a = _mm256_loadu_si256(w.as_ptr().add(i).cast());
                let b = _mm256_loadu_si256(x.as_ptr().add(i).cast());
                sum = _mm256_add_epi32(
                    sum,
                    _mm256_madd_epi16(
                        _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a)),
                        _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b)),
                    ),
                );
                sum = _mm256_add_epi32(
                    sum,
                    _mm256_madd_epi16(
                        _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(a)),
                        _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(b)),
                    ),
                );
                i += 32;
            }
            total += sum32(sum);
        }
        total
            + w[i..]
                .iter()
                .zip(&x[i..])
                .map(|(&a, &b)| a as i64 * b as i64)
                .sum::<i64>()
    }
    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_wrap32(w: &[i8], x: &[u32]) -> u32 {
        debug_assert_eq!(w.len(), x.len());
        let (mut index, mut accum) = (0, _mm256_setzero_si256());
        while index + 8 <= w.len() {
            let weights = _mm256_cvtepi8_epi32(_mm_loadl_epi64(w.as_ptr().add(index).cast()));
            let inputs = _mm256_loadu_si256(x.as_ptr().add(index).cast());
            accum = _mm256_add_epi32(accum, _mm256_mullo_epi32(weights, inputs));
            index += 8;
        }
        let mut lanes = [0u32; 8];
        _mm256_storeu_si256(lanes.as_mut_ptr().cast(), accum);
        let mut total = lanes.into_iter().fold(0u32, u32::wrapping_add);
        for (&weight, &input) in w[index..].iter().zip(&x[index..]) {
            total = total.wrapping_add((weight as i32 as u32).wrapping_mul(input));
        }
        total
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_wrap64(w: &[i8], x: &[u64]) -> u64 {
        debug_assert_eq!(w.len(), x.len());
        let (mut index, mut accum) = (0, _mm256_setzero_si256());
        let zero = _mm256_setzero_si256();
        while index + 4 <= w.len() {
            let packed = std::ptr::read_unaligned(w.as_ptr().add(index).cast::<i32>());
            let weights = _mm256_cvtepi8_epi64(_mm_cvtsi32_si128(packed));
            let signs = _mm256_cmpgt_epi64(zero, weights);
            let magnitudes = _mm256_sub_epi64(_mm256_xor_si256(weights, signs), signs);
            let inputs = _mm256_loadu_si256(x.as_ptr().add(index).cast());
            let low = _mm256_mul_epu32(inputs, magnitudes);
            let high = _mm256_slli_epi64::<32>(_mm256_mul_epu32(
                _mm256_srli_epi64::<32>(inputs),
                magnitudes,
            ));
            let products = _mm256_add_epi64(low, high);
            let signed = _mm256_sub_epi64(_mm256_xor_si256(products, signs), signs);
            accum = _mm256_add_epi64(accum, signed);
            index += 4;
        }
        let mut lanes = [0u64; 4];
        _mm256_storeu_si256(lanes.as_mut_ptr().cast(), accum);
        let mut total = lanes.into_iter().fold(0u64, u64::wrapping_add);
        for (&weight, &input) in w[index..].iter().zip(&x[index..]) {
            total = total.wrapping_add((weight as i64 as u64).wrapping_mul(input));
        }
        total
    }
}

// Apple Silicon and Linux ARM64 use widening NEON products. No x86 assumptions
// enter a portable wheel, and the scalar path remains available for comparison.
#[cfg(target_arch = "aarch64")]
mod arm {
    use std::arch::aarch64::*;

    #[target_feature(enable = "neon")]
    pub unsafe fn dot_wide(w: &[i8], x: &[u32]) -> i64 {
        let mut index = 0;
        let mut accum = [vdupq_n_s64(0); 4];
        while index + 8 <= w.len() {
            let weights = vmovl_s8(vld1_s8(w.as_ptr().add(index)));
            let low = vmovl_s16(vget_low_s16(weights));
            let high = vmovl_s16(vget_high_s16(weights));
            // Input residues are below 2^31, so signed reinterpretation is exact.
            let a = vreinterpretq_s32_u32(vld1q_u32(x.as_ptr().add(index)));
            let b = vreinterpretq_s32_u32(vld1q_u32(x.as_ptr().add(index + 4)));
            accum[0] = vaddq_s64(accum[0], vmull_s32(vget_low_s32(low), vget_low_s32(a)));
            accum[1] = vaddq_s64(accum[1], vmull_s32(vget_high_s32(low), vget_high_s32(a)));
            accum[2] = vaddq_s64(accum[2], vmull_s32(vget_low_s32(high), vget_low_s32(b)));
            accum[3] = vaddq_s64(accum[3], vmull_s32(vget_high_s32(high), vget_high_s32(b)));
            index += 8;
        }
        let total: i64 = accum.iter().map(|&v| vaddvq_s64(v)).sum();
        total
            + w[index..]
                .iter()
                .zip(&x[index..])
                .map(|(&weight, &input)| weight as i64 * input as i64)
                .sum::<i64>()
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn dot_clear(w: &[i8], x: &[i8]) -> i64 {
        let mut index = 0;
        let mut total = 0i64;
        while index + 16 <= w.len() {
            let end = (index + 2048).min(w.len() / 16 * 16);
            let mut sum = vdupq_n_s32(0);
            while index < end {
                let a = vld1q_s8(w.as_ptr().add(index));
                let b = vld1q_s8(x.as_ptr().add(index));
                sum = vpadalq_s16(sum, vmull_s8(vget_low_s8(a), vget_low_s8(b)));
                sum = vpadalq_s16(sum, vmull_s8(vget_high_s8(a), vget_high_s8(b)));
                index += 16;
            }
            // At most 2048 products of magnitude 16384: horizontal sum fits i32.
            total += vaddvq_s32(sum) as i64;
        }
        total
            + w[index..]
                .iter()
                .zip(&x[index..])
                .map(|(&a, &b)| a as i64 * b as i64)
                .sum::<i64>()
    }
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_wrap32(w: &[i8], x: &[u32]) -> u32 {
        debug_assert_eq!(w.len(), x.len());
        let mut index = 0;
        let mut accum = vdupq_n_u32(0);
        while index + 8 <= w.len() {
            let weights = vmovl_s8(vld1_s8(w.as_ptr().add(index)));
            let low = vreinterpretq_u32_s32(vmovl_s16(vget_low_s16(weights)));
            let high = vreinterpretq_u32_s32(vmovl_s16(vget_high_s16(weights)));
            accum = vaddq_u32(accum, vmulq_u32(low, vld1q_u32(x.as_ptr().add(index))));
            accum = vaddq_u32(accum, vmulq_u32(high, vld1q_u32(x.as_ptr().add(index + 4))));
            index += 8;
        }
        let mut total = vaddvq_u32(accum);
        for (&weight, &input) in w[index..].iter().zip(&x[index..]) {
            total = total.wrapping_add((weight as i32 as u32).wrapping_mul(input));
        }
        total
    }
}
