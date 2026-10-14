/* Independently written research fixture. No SDK registration.
 * Parent only: clang -O2 -std=c11 -Wall -Wextra -Werror -dynamiclib
 *   -Wno-deprecated-declarations fixture.c -framework Accelerate -o fixture.dylib
 * LP64 CBLAS is deliberate, matching installed digit8's integer ABI.
 * Do not use -ffast-math. Snapshot must remain owned by this handle.
 */
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define CDB_WORK ((size_t)64 << 20)
#define CDB_ELEMENTS ((size_t)4 << 20)
#define CDB_WEIGHTS ((size_t)16 << 20)
#define CDB_L1 131071u

typedef struct { uint32_t first, last, start, end, l1, reserved; } cdb_record;
_Static_assert(sizeof(cdb_record) == 24, "certificate record ABI");
typedef struct {
    size_t rows, cols, batch, tile, count, kmax, reserved_bytes, workspace;
    unsigned bits;
    int certified;
    int8_t *weights;
    cdb_record *records;
} cdb_handle;

/* Status: 1 ABI/geometry; 2 workspace; 3 allocation; 4 source certificate;
 * 5 canonical input; 6 SGEMM integer check. Failed output is invalid.
 */
void cdb_destroy(void *opaque) {
    cdb_handle *h = opaque;
    if (h) { free(h->weights); free(h->records); free(h); }
}

static uint32_t magnitude(int8_t w) {
    int value = (int)w; /* abs(-128) must widen before negation. */
    return (uint32_t)(value < 0 ? -value : value);
}

static void append(cdb_handle *h, size_t first, size_t last, size_t start,
                   size_t end, const uint64_t *sums) {
    uint64_t maximum = 0;
    if (sums) for (size_t j = 0; j < last - first; ++j)
        if (sums[j] > maximum) maximum = sums[j];
    h->records[h->count++] = (cdb_record){(uint32_t)first, (uint32_t)last,
        (uint32_t)start, (uint32_t)end, (uint32_t)maximum, 0};
    if (end - start > h->kmax) h->kmax = end - start;
}

int cdb_create(const int8_t *source, size_t source_len, size_t rows, size_t cols,
               size_t batch, unsigned bits, int certified, void **result) {
    if (!result) return 1;
    *result = NULL;
    if (!source || !rows || !cols || rows > 16384 || cols > 16384 ||
        !batch || batch > 4096 || rows > CDB_WEIGHTS / cols ||
        rows * cols != source_len || rows > CDB_ELEMENTS / batch ||
        cols > CDB_ELEMENTS / batch ||
        (bits != 16 && bits != 24 && bits != 32) ||
        (certified != 0 && certified != 1)) return 1;
    size_t tile = rows < 1024 ? rows : 1024;
    size_t tiles = (rows + tile - 1) / tile;
    size_t metadata = 24 * cols * tiles + 32 * tile;
    size_t fixed = 4 * (batch * rows + batch * tile) + metadata;
    if (fixed >= CDB_WORK) return 2;
    size_t width = (CDB_WORK - fixed) / (4 * (batch + tile));
    if (width > cols) width = cols;
    if (!certified) width = cols < 1023 ? cols : 1023;
    if (!width || fixed + 4 * (batch + tile) * width > CDB_WORK) return 2;
    cdb_handle *h = calloc(1, sizeof(*h));
    if (!h) return 3;
    h->rows = rows; h->cols = cols; h->batch = batch; h->bits = bits;
    h->tile = tile; h->certified = certified; h->reserved_bytes = metadata;
    h->weights = malloc(source_len);
    h->records = malloc(24 * cols * tiles);
    uint64_t *sums = calloc(tile, sizeof(*sums));
    if (!h->weights || !h->records || !sums) {
        free(sums); cdb_destroy(h); return 3;
    }
    memcpy(h->weights, source, source_len);
    for (size_t first = 0; first < rows; first += tile) {
        size_t last = rows - first < tile ? rows : first + tile;
        if (!certified) {
            for (size_t start = 0; start < cols; start += width)
                append(h, first, last, start, cols - start < width ? cols : start + width, NULL);
            continue;
        }
        memset(sums, 0, tile * sizeof(*sums));
        size_t start = 0;
        for (size_t column = 0; column < cols; ++column) {
            int cut = column - start == width;
            for (size_t j = first; j < last; ++j)
                if (sums[j - first] + magnitude(h->weights[j * cols + column]) > CDB_L1) cut = 1;
            if (cut) {
                if (column == start) { free(sums); cdb_destroy(h); return 4; }
                append(h, first, last, start, column, sums);
                memset(sums, 0, tile * sizeof(*sums)); start = column;
            }
            for (size_t j = first; j < last; ++j)
                sums[j - first] += magnitude(h->weights[j * cols + column]);
        }
        append(h, first, last, start, cols, sums);
    }
    free(sums);
    /* Independent second traversal validates exact coverage and recomputes
     * each row's absolute sum. Greedy planner's stored maximum is not trusted.
     */
    size_t record = 0;
    for (size_t first = 0; first < rows; first += tile) {
        size_t last = rows - first < tile ? rows : first + tile;
        size_t next = 0;
        while (next < cols) {
            if (record >= h->count) { cdb_destroy(h); return 4; }
            cdb_record p = h->records[record++];
            if (p.first != first || p.last != last || p.start != next ||
                p.end <= next || p.end > cols || p.end - next > width || p.reserved) {
                cdb_destroy(h); return 4;
            }
            uint64_t maximum = 0;
            if (certified) {
                for (size_t j = first; j < last; ++j) {
                    uint64_t total = 0;
                    for (size_t i = next; i < p.end; ++i) total += magnitude(h->weights[j * cols + i]);
                    if (total > maximum) maximum = total;
                    if (total > CDB_L1) { cdb_destroy(h); return 4; }
                }
                if (maximum != p.l1) { cdb_destroy(h); return 4; }
            } else if ((uint64_t)(p.end - next) * 128 * 128 >= (1u << 24)) {
                cdb_destroy(h); return 4;
            }
            next = p.end;
        }
    }
    if (record != h->count) { cdb_destroy(h); return 4; }
    h->workspace = fixed + 4 * (batch + tile) * h->kmax;
    *result = h;
    return 0;
}

/* Stats ABI: calls, max width, workspace reservation, block records,
 * record/scratch reservation, dense products, source bytes, owner bytes.
 */
int cdb_stats(void *opaque, uint64_t *values, size_t count) {
    cdb_handle *h = opaque;
    if (!h || !values || count != 8) return 1;
    values[0] = h->count * (h->bits / 8); values[1] = h->kmax;
    values[2] = h->workspace; values[3] = h->count;
    values[4] = h->reserved_bytes;
    values[5] = (uint64_t)h->batch * h->rows * h->cols * (h->bits / 8);
    values[6] = h->rows * h->cols;
    values[7] = sizeof(*h) + h->rows * h->cols + 24 * h->cols * ((h->rows + h->tile - 1) / h->tile);
    return 0;
}

int cdb_records(void *opaque, uint32_t *values, size_t count) {
    cdb_handle *h = opaque;
    if (!h || !values || count != 6 * h->count) return 1;
    memcpy(values, h->records, count * sizeof(*values)); return 0;
}

int cdb_execute(void *opaque, const uint32_t *input, size_t input_len,
                const uint32_t *mask, size_t mask_len, uint32_t *output, size_t output_len) {
    cdb_handle *h = opaque;
    if (!h || !input || !mask || !output || input_len != h->batch * h->cols ||
        mask_len != h->batch * h->rows || output_len != mask_len ||
        output == input || output == mask) return 1;
    uint32_t narrow = h->bits == 32 ? UINT32_MAX : (1u << h->bits) - 1;
    for (size_t i = 0; i < input_len; ++i) if (input[i] > narrow) return 5;
    for (size_t i = 0; i < mask_len; ++i) if (mask[i] > narrow) return 5;
    /* Per-issuance allocations mirror kernel lifetime. Input/mask buffers and
     * caller's output are separate owners; immutable snapshot stays in handle.
     */
    float *a = malloc(4 * h->batch * h->kmax);
    float *b = malloc(4 * h->tile * h->kmax);
    float *c = malloc(4 * h->batch * h->tile);
    if (!a || !b || !c) { free(a); free(b); free(c); return 3; }
    memset(output, 0, 4 * output_len);
    int status = 0;
    /* Baseline records are ordered block-major below, matching native control.
     * Certified records are tile-major because each tile has different cuts.
     */
    size_t base_blocks = (h->cols + 1022) / 1023;
    size_t tiles = (h->rows + h->tile - 1) / h->tile;
    for (size_t visit = 0; visit < h->count; ++visit) {
        size_t index = h->certified ? visit : (visit % tiles) * base_blocks + visit / tiles;
        cdb_record p = h->records[index];
        size_t k = p.end - p.start, n = p.last - p.first;
        for (size_t j = 0; j < n; ++j) for (size_t i = 0; i < k; ++i)
            b[j * k + i] = (float)h->weights[((size_t)p.first + j) * h->cols + p.start + i];
        uint64_t carry = 0;
        for (unsigned shift = 0; shift < h->bits; shift += 8) {
            for (size_t r = 0; r < h->batch; ++r) for (size_t i = 0; i < k; ++i) {
                uint64_t remaining = ((uint64_t)input[r * h->cols + p.start + i] + carry) >> shift;
                int digit = (int)((remaining + 128) & 255) - 128;
                a[r * k + i] = (float)digit;
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        (int)h->batch, (int)n, (int)k, 1.0f, a, (int)k,
                        b, (int)k, 0.0f, c, (int)n);
            for (size_t r = 0; r < h->batch; ++r) for (size_t j = 0; j < n; ++j) {
                float v = c[r * n + j];
                if (!isfinite(v) || fabsf(v) >= 16777216.0f || truncf(v) != v) {
                    status = 6; goto cleanup;
                }
                /* Range check above proves safe i32 cast; u32 shift/add wrap. */
                output[r * h->rows + p.first + j] += (uint32_t)(int32_t)v << shift;
            }
            carry += (uint64_t)128 << shift;
        }
    }
    for (size_t i = 0; i < output_len; ++i) output[i] = (output[i] - mask[i]) & narrow;
cleanup:
    free(a); free(b); free(c); return status;
}
