/* Standalone resident test fixture only. No SDK registration or workspace build.
 * cc -O2 -dynamiclib fixture.c -framework Accelerate -o fixture.dylib
 */
#include <Accelerate/Accelerate.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

int limb_stationary_corrections(const int8_t *w, const uint32_t *x, uint32_t *out,
                               size_t m, size_t rows, size_t cols,
                               unsigned bits, int stationary) {
    const size_t block = 1023, tile = 1024, cap = 4u << 20;
    if (!w || !x || !out || !m || m > 4096 || !rows || !cols ||
        rows > cap / m || cols > cap / m ||
        rows > (16u << 20) / cols ||
        (bits != 16 && bits != 24 && bits != 32)) return 1;
    size_t kmax = cols < block ? cols : block;
    size_t nmax = rows < tile ? rows : tile;
    size_t d = bits / 8, slabs = stationary ? d : 1;
    size_t work = 4 * (m * rows + slabs * m * kmax + nmax * kmax + m * nmax);
    if (work > (64u << 20)) return 2;
    for (size_t i = 0; i < m * cols; i++)
        if (bits < 32 && x[i] >= (1u << bits)) return 3;
    /* All geometry/domain/work checks precede output initialization/allocation. */
    float *a = malloc(4 * slabs * m * kmax);
    float *b = malloc(4 * nmax * kmax);
    float *c = malloc(4 * m * nmax);
    if (!a || !b || !c) { free(a); free(b); free(c); return 4; }
    memset(out, 0, 4 * m * rows);
    int status = 0;
    for (size_t start = 0; start < cols; start += block) {
        size_t k = cols - start < block ? cols - start : block;
        if (stationary) {
            uint64_t carry = 0;
            for (size_t digit = 0; digit < d; digit++) {
                unsigned shift = 8 * digit;
                for (size_t r = 0; r < m; r++) for (size_t i = 0; i < k; i++) {
                    uint64_t rem = ((uint64_t)x[r * cols + start + i] + carry) >> shift;
                    a[digit * m * k + r * k + i] = (float)((int)((rem + 128) & 255) - 128);
                }
                carry += (uint64_t)128 << shift;
            }
        }
        for (size_t first = 0; first < rows; first += tile) {
            size_t n = rows - first < tile ? rows - first : tile;
            for (size_t j = 0; j < n; j++) for (size_t i = 0; i < k; i++)
                b[j * k + i] = (float)w[(first + j) * cols + start + i];
            uint64_t carry = 0;
            for (size_t digit = 0; digit < d; digit++) {
                unsigned shift = 8 * digit;
                float *slab = a + (stationary ? digit * m * k : 0);
                if (!stationary) {
                    for (size_t r = 0; r < m; r++) for (size_t i = 0; i < k; i++) {
                        uint64_t rem = ((uint64_t)x[r * cols + start + i] + carry) >> shift;
                        slab[r * k + i] = (float)((int)((rem + 128) & 255) - 128);
                    }
                }
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            (int)m, (int)n, (int)k, 1.0f, slab, (int)k,
                            b, (int)k, 0.0f, c, (int)n);
                for (size_t r = 0; r < m; r++) for (size_t j = 0; j < n; j++) {
                    float v = c[r * n + j];
                    if (!isfinite(v) || fabsf(v) >= 16777216.0f || truncf(v) != v) {
                        status = 5; goto done;
                    }
                    out[r * rows + first + j] += (uint32_t)(int32_t)v << shift;
                }
                carry += (uint64_t)128 << shift;
            }
        }
    }
    if (bits < 32) for (size_t i = 0; i < m * rows; i++) out[i] &= (1u << bits) - 1;
done:
    free(a); free(b); free(c);
    return status;
}
