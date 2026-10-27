//! One D21 trace/merge kernel. Legacy and explicit clear-reference gates differ;
//! operation order and coefficient algebra do not.
use super::schedule::{InterleaveSchedule, InterleaveStep};
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
    // Preserve degree/modulus/geometry error precedence as well as arithmetic.
    if !n.is_power_of_two() || n > max_degree || n > 8192 {
        return Err(LayoutError::InvalidDegree);
    }
    if !(3..(1_u64 << 61)).contains(&p) || p & 1 == 0 {
        return Err(LayoutError::InvalidModulus);
    }
    let schedule = InterleaveSchedule::bounded(n, stride, count, max_degree)?;
    if input.len() != count * n || output.len() != n || scratch.len() != schedule.scratch_words() {
        return Err(LayoutError::BufferLength);
    }
    if input.iter().any(|&x| x >= p) {
        return Err(LayoutError::NonCanonical);
    }
    let (values, temp) = scratch.split_at_mut(schedule.padded() * n);
    for step in schedule.steps() {
        match step {
            InterleaveStep::Normalize { divisor, .. } => {
                // Same repeated odd-ring inverse, including composite moduli.
                let inverse = inverse_power_two(p, divisor);
                values.fill(0);
                for (v, &a) in values.iter_mut().zip(input) {
                    *v = mul(a, inverse, p);
                }
            }
            InterleaveStep::Trace { lane, element } => {
                let value = &mut values[lane * n..(lane + 1) * n];
                temp.copy_from_slice(value);
                add_automorphism(n, p, value, temp, element);
            }
            InterleaveStep::Merge {
                left,
                right,
                shift,
                element,
            } => {
                let (low, high) = values.split_at_mut(right * n);
                let a = &mut low[left * n..(left + 1) * n];
                let b = &high[..n];
                for i in 0..n {
                    let shifted = if i >= shift {
                        b[i - shift]
                    } else {
                        sub(0, b[n + i - shift], p)
                    };
                    temp[i] = sub(a[i], shifted, p);
                    a[i] = (a[i] + shifted) % p;
                }
                add_automorphism(n, p, a, temp, element);
            }
            InterleaveStep::Finish => output.copy_from_slice(&values[..n]),
        }
    }
    Ok(())
}
fn add_automorphism(n: usize, p: u64, output: &mut [u64], input: &[u64], element: usize) {
    for (i, &v) in input.iter().enumerate() {
        let at = i * element % (2 * n);
        let v = if at < n { v } else { sub(0, v, p) };
        output[at % n] = (output[at % n] + v) % p;
    }
}
