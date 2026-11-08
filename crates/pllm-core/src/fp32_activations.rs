//! Public-source FP32 forward activations, not private integer kernels.
//!
//! GELU uses the erf mathematical form, with each stated operation in `f32`;
//! tanh uses `libm::tanhf`. Neither promises correctly rounded mathematical
//! results or bit identity with Torch. No fused-operation intent is implied.
//! These kernels allocate nothing and never mutate their input.

use std::fmt;

/// Buffer geometry or numeric failure in an FP32 activation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    Capacity { required: usize, available: usize },
    NonFiniteInput { index: usize },
    NonFiniteArithmetic { index: usize },
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Capacity { required, available } => write!(
                formatter,
                "FP32 activation requires {required} output elements, got {available}"
            ),
            Self::NonFiniteInput { index } => {
                write!(formatter, "FP32 activation input {index} is not finite")
            }
            Self::NonFiniteArithmetic { index } => write!(
                formatter,
                "FP32 activation arithmetic at {index} is not finite"
            ),
        }
    }
}

impl std::error::Error for Error {}

/// Write erf-form GELU to `output[..input.len()]` using FP32 operations.
///
/// Evaluation order: `c = f32::from_bits(0x3f3504f3)`, `u = x * c`,
/// `e = libm::erff(u)`, `s = 1.0f32 + e`, `h = 0.5f32 * x`, `y = h * s`.
/// This is not the tanh approximation of GELU.
///
/// Capacity is checked before reading input values or writing output. Capacity
/// refusal leaves all output untouched. After that check, any non-finite input
/// or arithmetic result clears the entire active output prefix to positive zero
/// before returning an error. The spare output tail is always untouched.
/// Empty input succeeds without touching output; it is not a useful model case.
pub fn gelu_erf_into(input: &[f32], output: &mut [f32]) -> Result<(), Error> {
    apply_into(input, output, gelu_erf)
}

/// Write `libm::tanhf(x)` to `output[..input.len()]`, preserving signed zero.
///
/// Capacity, failure clearing, spare-tail, and empty-input behavior are identical
/// to [`gelu_erf_into`]. This is a mathematical-form contract, not a correctly
/// rounded or Torch-bit-identity contract.
pub fn tanh_into(input: &[f32], output: &mut [f32]) -> Result<(), Error> {
    apply_into(input, output, |x| Some(libm::tanhf(x)))
}

fn gelu_erf(x: f32) -> Option<f32> {
    let c = f32::from_bits(0x3f3504f3);
    let u = x * c;
    let e = libm::erff(u);
    let s = 1.0f32 + e;
    let h = 0.5f32 * x;
    let y = h * s;
    if [u, e, s, h, y].iter().all(|value| value.is_finite()) {
        Some(y)
    } else {
        None
    }
}

fn apply_into(
    input: &[f32],
    output: &mut [f32],
    evaluate: fn(f32) -> Option<f32>,
) -> Result<(), Error> {
    if output.len() < input.len() {
        return Err(Error::Capacity {
            required: input.len(),
            available: output.len(),
        });
    }
    let active = &mut output[..input.len()];
    for (index, &x) in input.iter().enumerate() {
        if !x.is_finite() {
            active.fill(0.0);
            return Err(Error::NonFiniteInput { index });
        }
        match evaluate(x) {
            Some(y) if y.is_finite() => active[index] = y,
            _ => {
                active.fill(0.0);
                return Err(Error::NonFiniteArithmetic { index });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Declared before execution: scalar activation checks only, not an encoder
    // gate. References use independent f64 libm functions and mathematical forms.
    const ATOL: f64 = 2e-6;
    const RTOL: f64 = 2e-6;
    type Kernel = fn(&[f32], &mut [f32]) -> Result<(), Error>;
    const KERNELS: [Kernel; 2] = [gelu_erf_into, tanh_into];

    fn assert_close(actual: f32, expected: f64) {
        assert!(actual.is_finite());
        let error = (f64::from(actual) - expected).abs();
        assert!(
            error <= ATOL + RTOL * expected.abs(),
            "actual={actual:?}, expected={expected:?}, error={error:?}"
        );
    }

    fn check_references(x: f32) {
        let mut output = [0.0];
        let reference_x = f64::from(x);
        gelu_erf_into(&[x], &mut output).unwrap();
        let expected = (0.5 * reference_x)
            * (1.0 + libm::erf(reference_x / std::f64::consts::SQRT_2));
        assert_close(output[0], expected);
        tanh_into(&[x], &mut output).unwrap();
        assert_close(output[0], libm::tanh(reference_x));
    }

    #[test]
    fn dense_interval_matches_independent_f64_references() {
        for step in -10_000..=10_000 {
            check_references(step as f32 / 1000.0);
        }
    }

    #[test]
    fn signed_zeros_are_preserved() {
        for kernel in KERNELS {
            let input = [0.0, -0.0];
            let mut output = [1.0; 2];
            kernel(&input, &mut output).unwrap();
            assert_eq!(output.map(f32::to_bits), input.map(f32::to_bits));
        }
    }

    #[test]
    fn subnormals_match_references_and_tanh_preserves_their_bits() {
        for bits in [1, 2, 3, 0x0040_0000, 0x007f_ffff] {
            for sign in [0, 0x8000_0000] {
                let x = f32::from_bits(bits | sign);
                check_references(x);
                let mut output = [0.0];
                tanh_into(&[x], &mut output).unwrap();
                assert_eq!(output[0].to_bits(), x.to_bits());
            }
        }
    }

    #[test]
    fn saturation_and_largest_finite_inputs_remain_finite() {
        for x in [-f32::MAX, -100.0, -20.0, -10.0, 10.0, 20.0, 100.0, f32::MAX] {
            check_references(x);
        }
        let mut output = [0.0; 2];
        gelu_erf_into(&[-f32::MAX, f32::MAX], &mut output).unwrap();
        assert_eq!(output[0].to_bits(), (-0.0f32).to_bits());
        assert_eq!(output[1].to_bits(), f32::MAX.to_bits());
        tanh_into(&[-f32::MAX, f32::MAX], &mut output).unwrap();
        assert_eq!(output, [-1.0, 1.0]);
    }

    #[test]
    fn capacity_refusal_precedes_nonfinite_inspection_and_preserves_canaries() {
        for kernel in KERNELS {
            let input = [f32::from_bits(0x7fc0_1234), f32::INFINITY, -f32::INFINITY];
            let before_input = input.map(f32::to_bits);
            let mut output = [f32::from_bits(0x7fc0_5678), -0.0];
            let before_output = output.map(f32::to_bits);
            assert_eq!(
                kernel(&input, &mut output),
                Err(Error::Capacity { required: 3, available: 2 })
            );
            assert_eq!(output.map(f32::to_bits), before_output);
            assert_eq!(input.map(f32::to_bits), before_input);
            assert_eq!(
                kernel(&input, &mut []),
                Err(Error::Capacity { required: 3, available: 0 })
            );
        }
        // The common preflight must not invoke even an instrumented evaluator.
        assert_eq!(
            apply_into(&[1.0], &mut [], |_| panic!("capacity read input")),
            Err(Error::Capacity { required: 1, available: 0 })
        );
    }

    #[test]
    fn nonfinite_refusal_wipes_entire_active_prefix_and_preserves_tail_and_input() {
        for kernel in KERNELS {
            for bad in [f32::NAN, f32::INFINITY, -f32::INFINITY] {
                for index in 0..3 {
                    let mut input = [1.0, -2.0, 3.0];
                    input[index] = bad;
                    let before_input = input.map(f32::to_bits);
                    let mut output = [9.0, 8.0, 7.0, -0.0, f32::from_bits(0x7fc0_5678)];
                    let before_tail = [output[3].to_bits(), output[4].to_bits()];
                    assert_eq!(kernel(&input, &mut output), Err(Error::NonFiniteInput { index }));
                    assert!(output[..3].iter().all(|x| x.to_bits() == 0));
                    assert_eq!([output[3].to_bits(), output[4].to_bits()], before_tail);
                    assert_eq!(input.map(f32::to_bits), before_input);
                }
            }
        }
    }

    #[test]
    fn arithmetic_failure_wipes_active_prefix_without_touching_tail() {
        // Finite public inputs should not reach this branch with pinned libm;
        // inject evaluator failures to check its cleanup contract independently.
        for evaluate in [
            (|x| if x == 2.0 { None } else { Some(x) }) as fn(f32) -> Option<f32>,
            |x| Some(if x == 2.0 { f32::NAN } else { x }),
            |x| Some(if x == 2.0 { f32::INFINITY } else { x }),
        ] {
            let mut output = [9.0, 8.0, 7.0, -0.0];
            assert_eq!(
                apply_into(&[1.0, 2.0, 3.0], &mut output, evaluate),
                Err(Error::NonFiniteArithmetic { index: 1 })
            );
            assert!(output[..3].iter().all(|x| x.to_bits() == 0));
            assert_eq!(output[3].to_bits(), (-0.0f32).to_bits());
        }
    }

    #[test]
    fn successful_calls_preserve_input_and_spare_tail() {
        for kernel in KERNELS {
            let input = [-2.0, -0.0, 1.0];
            let before_input = input.map(f32::to_bits);
            let mut output = [99.0, 99.0, 99.0, -0.0, f32::from_bits(0x7fc0_5678)];
            let before_tail = [output[3].to_bits(), output[4].to_bits()];
            kernel(&input, &mut output).unwrap();
            assert!(output[..3].iter().all(|x| x.is_finite()));
            assert_eq!([output[3].to_bits(), output[4].to_bits()], before_tail);
            assert_eq!(input.map(f32::to_bits), before_input);
        }
    }

    #[test]
    fn empty_input_is_a_noop_even_with_nonfinite_output_canaries() {
        for kernel in KERNELS {
            assert_eq!(kernel(&[], &mut []), Ok(()));
            let mut output = [f32::from_bits(0x7fc0_5678), -0.0];
            let before = output.map(f32::to_bits);
            assert_eq!(kernel(&[], &mut output), Ok(()));
            assert_eq!(output.map(f32::to_bits), before);
        }
    }
}
