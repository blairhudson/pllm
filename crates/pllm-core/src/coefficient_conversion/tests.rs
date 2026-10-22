use super::*;

// Independent quotient/remainder control from Python test_shared_profile_oracle
// CoefficientH2AClearAlgebra.reference. Never calls production down/delta_encode.
fn reference(m: u64, error: i128, zero: i128, mask: u128, bits: u8, q: u128) -> (u64, u64) {
    let t = 1_u128 << bits;
    let nearest = |phase: i128| {
        let scaled = phase.rem_euclid(q as i128) as u128 * t;
        let (quotient, remainder) = (scaled / q, scaled % q);
        assert_ne!(2 * remainder, q);
        (quotient + u128::from(2 * remainder > q)) % t
    };
    let receiver = nearest((q / t * u128::from(m)) as i128 + error + zero + mask as i128);
    let sender = (t - nearest(mask as i128)) % t;
    (receiver as u64, sender as u64)
}

fn evaluate(
    domain: CoefficientConversion,
    m: u64,
    error: i128,
    zero: i128,
    mask: u128,
) -> (u64, u64) {
    let result = domain
        .evaluate(CoefficientPhaseInputs {
            delta_message: domain.delta_encode(m).unwrap(),
            error,
            fresh_zero_error: zero,
            mask,
        })
        .unwrap();
    (result.receiver, result.sender)
}

#[test]
fn python_derived_tiny_exhaustive_24604_cases() {
    let mut cases = 0;
    for bits in 1..=3 {
        let t = 1_u64 << bits;
        for q in ((u128::from(t) + 1)..18).step_by(2) {
            let domain = CoefficientConversion::new(bits, q).unwrap();
            let radius = 2.min((q / 2) as i128);
            for m in 0..t {
                for mask in 0..q {
                    for error in -radius..=radius {
                        for zero in -radius..=radius {
                            let actual = evaluate(domain, m, error, zero, mask);
                            assert_eq!(actual, reference(m, error, zero, mask, bits, q));
                            assert!(actual.0 < t && actual.1 < t);
                            cases += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(cases, 24604);
}

#[test]
fn python_derived_wrap_and_signed_phase_edges() {
    for (bits, q) in [(1, 3), (2, 7), (3, 9), (4, 255), (4, 257)] {
        let domain = CoefficientConversion::new(bits, q).unwrap();
        for m in [0, 1, (1 << bits) - 1] {
            for mask in [0, 1, q / 2, q - 2, q - 1] {
                for error in [-(q as i128 / 2), -1, 0, 1, q as i128 / 2] {
                    for zero in [-(q as i128 / 2), -1, 0, 1, q as i128 / 2] {
                        assert_eq!(
                            evaluate(domain, m, error, zero, mask),
                            reference(m, error, zero, mask, bits, q)
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn frozen_python_controls_distinguish_delta_from_lift_and_error_bound() {
    let domain = CoefficientConversion::new(2, 7).unwrap();
    assert_eq!(domain.delta_encode(3), Ok(3));
    // Python lift(3,2,7)=5, whereas Delta*3=3.
    assert_eq!(domain.down(5), Ok(3));
    assert_eq!(evaluate(domain, 3, 0, 0, 0), (2, 0));
    let domain = CoefficientConversion::new(3, 15).unwrap();
    assert_eq!(evaluate(domain, 0, 7, 0, 0), (4, 0)); // signed ring8: -4
}

#[test]
fn rejects_domains_and_noncanonical_or_out_of_range_phases() {
    use ConversionError::*;
    for bits in [0, 65, 255] {
        assert_eq!(CoefficientConversion::new(bits, 7), Err(InvalidBits));
    }
    for q in [0, 3, 4, 6] {
        assert_eq!(CoefficientConversion::new(2, q), Err(InvalidModulus));
    }
    let domain = CoefficientConversion::new(2, 11).unwrap();
    assert_eq!(domain.delta_encode(4), Err(NonCanonical));
    let valid = CoefficientPhaseInputs {
        delta_message: 6,
        error: 0,
        fresh_zero_error: 0,
        mask: 0,
    };
    for delta_message in [1, 8, u128::MAX] {
        assert_eq!(
            domain.evaluate(CoefficientPhaseInputs {
                delta_message,
                ..valid
            }),
            Err(InvalidDeltaMessage)
        );
    }
    assert_eq!(
        domain.evaluate(CoefficientPhaseInputs { mask: 11, ..valid }),
        Err(NonCanonical)
    );
    for error in [i128::MIN, -6, 6, i128::MAX] {
        assert_eq!(
            domain.evaluate(CoefficientPhaseInputs { error, ..valid }),
            Err(PhaseOutOfRange)
        );
        assert_eq!(
            domain.evaluate(CoefficientPhaseInputs {
                fresh_zero_error: error,
                ..valid
            }),
            Err(PhaseOutOfRange)
        );
    }
}

#[test]
fn exact_u128_capacity_boundary_and_explicit_wider_integer_gate() {
    use ConversionError::ArithmeticCapacity;
    // bits=1: worst numerator=5*q-4; u128::MAX is divisible by 5,
    // and q=MAX/5 is odd. Next odd modulus exceeds admitted capacity.
    let q = u128::MAX / 5;
    let domain = CoefficientConversion::new(1, q).unwrap();
    assert_eq!(domain.bits(), 1);
    assert_eq!(domain.odd_modulus(), q);
    assert_eq!(domain.down(q - 1), Ok(0));
    assert_eq!(domain.down(u128::MAX), Ok(0));
    let half = (q / 2) as i128;
    assert_eq!(evaluate(domain, 0, -half, half, 0), (0, 0));
    assert_eq!(
        CoefficientConversion::new(1, q + 2),
        Err(ArithmeticCapacity)
    );
    assert_eq!(
        CoefficientConversion::new(64, (1_u128 << 64) + 1),
        Err(ArithmeticCapacity)
    );
    assert_eq!(
        CoefficientConversion::new(1, u128::MAX),
        Err(ArithmeticCapacity)
    );
    // Largest target bit width with any admitted odd modulus; frozen exact control.
    let domain = CoefficientConversion::new(63, (1_u128 << 63) + 1).unwrap();
    assert_eq!(
        evaluate(domain, (1_u64 << 63) - 1, 0, 0, 0),
        ((1_u64 << 63) - 2, 0)
    );
}
