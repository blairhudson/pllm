//! One D21 trace/merge kernel. Legacy and explicit clear-reference gates differ;
//! operation order and coefficient algebra do not.
use super::LayoutError;

fn mul(a: u64, b: u64, p: u64) -> u64 {
    (u128::from(a) * u128::from(b) % u128::from(p)) as u64
}
fn sub(a: u64, b: u64, p: u64) -> u64 {
    if a >= b { a - b } else { p - (b - a) }
}
pub(crate) fn inverse_power_two(p: u64, divisor: usize) -> u64 {
    let mut inverse = 1;
    for _ in 0..divisor.trailing_zeros() {
        inverse = mul(inverse, p.div_ceil(2), p);
    }
    inverse
}
// Keep legacy and explicit-clear degree gates distinct without changing the
// extracted kernel's arithmetic or caller-owned input/output/scratch slices.
#[allow(clippy::too_many_arguments)]
pub(crate) fn interleave(
    n: usize,
    p: u64,
    max_degree: usize,
    input: &[u64],
    stride: usize,
    count: usize,
    output: &mut [u64],
    scratch: &mut [u64],
) -> Result<(), LayoutError> {
    if !n.is_power_of_two() || n > max_degree || n > 8192 {
        return Err(LayoutError::InvalidDegree);
    }
    if !(3..(1_u64 << 61)).contains(&p) || p & 1 == 0 {
        return Err(LayoutError::InvalidModulus);
    }
    if !stride.is_power_of_two() || stride > n || count == 0 || count > stride.min(256) {
        return Err(LayoutError::InvalidGeometry);
    }
    let padded = count.next_power_of_two();
    if input.len() != count * n || output.len() != n || scratch.len() != (padded + 1) * n {
        return Err(LayoutError::BufferLength);
    }
    if input.iter().any(|&x| x >= p) {
        return Err(LayoutError::NonCanonical);
    }
    // Preserve legacy repeated odd-ring inverse-of-two, including composites.
    let inverse = inverse_power_two(p, stride);
    let (values, temp) = scratch.split_at_mut(padded * n);
    values.fill(0);
    for (v, &a) in values.iter_mut().zip(input) {
        *v = mul(a, inverse, p);
    }
    for value in values.chunks_exact_mut(n) {
        let mut h = padded;
        while h < stride {
            temp.copy_from_slice(value);
            add_automorphism(n, p, value, temp, n / h + 1);
            h *= 2;
        }
    }
    let mut h = padded / 2;
    while h != 0 {
        let (left, right) = values.split_at_mut(h * n);
        for (a, b) in left.chunks_exact_mut(n).zip(right.chunks_exact(n)) {
            for i in 0..n {
                let shifted = if i >= h {
                    b[i - h]
                } else {
                    sub(0, b[n + i - h], p)
                };
                temp[i] = sub(a[i], shifted, p);
                a[i] = (a[i] + shifted) % p;
            }
            add_automorphism(n, p, a, temp, n / h + 1);
        }
        h /= 2;
    }
    output.copy_from_slice(&values[..n]);
    Ok(())
}
fn add_automorphism(n: usize, p: u64, output: &mut [u64], input: &[u64], element: usize) {
    for (i, &v) in input.iter().enumerate() {
        let at = i * element % (2 * n);
        let v = if at < n { v } else { sub(0, v, p) };
        output[at % n] = (output[at % n] + v) % p;
    }
}
