#include <cstddef>
#include <cstdint>

extern "C" {
void pack_u24(const uint32_t* in, uint8_t* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        const uint32_t value = in[i];
        out[3 * i] = static_cast<uint8_t>(value);
        out[3 * i + 1] = static_cast<uint8_t>(value >> 8);
        out[3 * i + 2] = static_cast<uint8_t>(value >> 16);
    }
}

void unpack_u24(const uint8_t* in, uint32_t* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = static_cast<uint32_t>(in[3 * i])
            | (static_cast<uint32_t>(in[3 * i + 1]) << 8)
            | (static_cast<uint32_t>(in[3 * i + 2]) << 16);
    }
}
}
