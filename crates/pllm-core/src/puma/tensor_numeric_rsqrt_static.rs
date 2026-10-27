//! Private scalar inference only: compile-time Bernstein enclosure, not energy.
//! This recipe is Q18: ONE, all four product shifts and the final raw shift in
//! the caller are fixed at 18. Other tensor scale recipes do not enter here.
use super::*;

const fn add(a: i128, b: i128) -> i128 {
    match a.checked_add(b) {
        Some(v) => v,
        None => panic!("rsqrt constant addition overflow"),
    }
}
const fn mul(a: i128, b: i128) -> i128 {
    match a.checked_mul(b) {
        Some(v) => v,
        None => panic!("rsqrt constant multiplication overflow"),
    }
}
const fn exact_div(a: i128, b: i128) -> i128 {
    assert!(b > 0 && a % b == 0);
    a / b
}
const fn choose(n: usize, k: usize) -> i128 {
    assert!(n <= 4 && k <= n);
    let mut value = 1;
    let mut i = 0;
    while i < k {
        value = exact_div(mul(value, (n - i) as i128), (i + 1) as i128);
        i += 1;
    }
    value
}
const fn power4(n: usize) -> i128 {
    let mut value = 1;
    let mut i = 0;
    while i < n {
        value = mul(value, 4);
        i += 1;
    }
    value
}

// f(t) is in integer output codes; t=(1+s)/4, s in [0,1]. D clears
// 4^i and every C(4,k). No division truncates a transformed coefficient.
// Degree-four Bernstein basis weights on [0,1] are nonnegative and sum to one:
// min(b_j) <= f((1+s)/4) <= max(b_j). Outward division happens only after the
// exact common-denominator controls have been minimized/maximized.
const DENOMINATOR: i128 = mul(power4(4), 12);
const _: () = assert!(ONE == 1 << 18);
const fn controls() -> [i128; 5] {
    let mut result = [0; 5];
    let mut j = 0;
    while j <= 4 {
        let mut k = 0;
        while k <= j {
            let mut i = k;
            while i <= 4 {
                let weight = mul(
                    mul(choose(i, k), choose(j, k)),
                    exact_div(DENOMINATOR, mul(power4(i), choose(4, k))),
                );
                result[j] = add(result[j], mul(RSQRT_COEFFICIENTS[i] as i128, weight));
                i += 1;
            }
            k += 1;
        }
        j += 1;
    }
    result
}
const CONTROLS: [i128; 5] = controls();

const fn magnitude(value: i64) -> i128 {
    let value = value as i128;
    if value < 0 { -value } else { value }
}
const fn ceil_div(n: i128, d: i128) -> i128 {
    assert!(d > 0);
    add(n.div_euclid(d), if n.rem_euclid(d) == 0 { 0 } else { 1 })
}
const ERROR: i128 = {
    // Actual dependency chain is U2=TR(U*U/Q), U3=TR(U2*U/Q),
    // U4=TR(U3*U/Q). For t<=1/2, absolute power errors are <1,
    // <3/2 and <7/4 codes. One final signed TR contributes <1.
    let q4 = mul(4, ONE as i128);
    ceil_div(
        add(
            add(
                mul(4, magnitude(RSQRT_COEFFICIENTS[2])),
                mul(6, magnitude(RSQRT_COEFFICIENTS[3])),
            ),
            add(mul(7, magnitude(RSQRT_COEFFICIENTS[4])), q4),
        ),
        q4,
    )
};
const LIMITS: (i64, i64) = {
    let mut lo = CONTROLS[0];
    let mut hi = CONTROLS[0];
    let mut i = 1;
    while i < 5 {
        if CONTROLS[i] < lo { lo = CONTROLS[i]; }
        if CONTROLS[i] > hi { hi = CONTROLS[i]; }
        i += 1;
    }
    let lo = add(lo.div_euclid(DENOMINATOR), -ERROR);
    let hi = add(ceil_div(hi, DENOMINATOR), ERROR);
    assert!(lo >= i64::MIN as i128 && hi <= i64::MAX as i128);
    (lo as i64, hi as i64)
};

fn transfer(ordinary_scaled: Interval, normalized: Interval) -> Result<Interval, Error> {
    if normalized.lower() < ONE / 4 || normalized.upper() > ONE / 2 {
        return Err(Error::Bounds);
    }
    // Replaces the old fixed (1,0,a0) affine transfer, not an additional pass.
    // Two checked additions + two comparisons for domain + min/max fit within
    // its original four i128 multiplications, four min/max, four checked sums
    // and two narrowing conversions. Same occupied-box/node census; no cells,
    // allocation, proof cache, certificate, native work or runtime instructions.
    let lower = ordinary_scaled.lower().checked_add(RSQRT_COEFFICIENTS[0])
        .ok_or(Error::Bounds)?.max(LIMITS.0);
    let upper = ordinary_scaled.upper().checked_add(RSQRT_COEFFICIENTS[0])
        .ok_or(Error::Bounds)?.min(LIMITS.1);
    Interval::new(lower, upper)
}

pub(super) fn finish<S: attention_work::SoftmaxSink>(
    b: &mut S, scaled: &Ranged, normalized: &Ranged,
) -> Result<Ranged, Error> {
    let mut boxes = [None; DOMAINS];
    for (j, slot) in boxes.iter_mut().enumerate() {
        if let Some(source) = scaled.boxes[j] {
            *slot = Some(transfer(source, normalized.boxes[j].ok_or(Error::Binding)?)?);
        }
    }
    // Exactly the old affine instruction and ordinary initial width inference;
    // only its subsequent public range refinement changes.
    let value = b.affine(scaled.value, None, affine(1, 0, RSQRT_COEFFICIENTS[0]))?;
    b.refine(value, hull(&boxes)?);
    Ok(Ranged { value, boxes })
}

#[cfg(test)]
#[path = "tensor_numeric_rsqrt_static_tests.rs"]
mod tests;
