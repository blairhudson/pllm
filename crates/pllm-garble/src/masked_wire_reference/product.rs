//! Negative-mask algebra shared by scalar and matrix masked products.
//! SIGMA §3.2.1: A=-RX, B=-RW, C=RX RW+RY. Public product belongs
//! exclusively to party 1; reconstruct only masked output, not differences.

pub(crate) fn negative_mask(value: u64) -> u64 {
    value.wrapping_neg()
}

pub(crate) fn correlation(rx: u64, rw: u64, ry: u64) -> u64 {
    rx.wrapping_mul(rw).wrapping_add(ry)
}

pub(crate) fn evaluate(party: u8, x: u64, w: u64, coefficients: &[u64; 3]) -> u64 {
    u64::from(party)
        .wrapping_mul(x)
        .wrapping_mul(w)
        .wrapping_add(coefficients[0].wrapping_mul(w))
        .wrapping_add(coefficients[1].wrapping_mul(x))
        .wrapping_add(coefficients[2])
}
