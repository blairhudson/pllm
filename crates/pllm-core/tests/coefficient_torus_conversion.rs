//! Public synthetic TEST secrets only; independent BigInt and quotient/remainder
//! controls. No encryption, RNG, decryption, model, or candidate roundtrip oracle.
use num_bigint::BigInt;
use pllm_core::coefficient_conversion::CoefficientConversion;
use pllm_core::coefficient_extraction::{Destination, KeyDomain, Layout, Limb, Scale, Sign};
use pllm_core::coefficient_torus_conversion::{Error, MAX_DIMENSION, Torus64Conversion};

// INFERRED 53-bit fixed odd test modulus, NOT a recovered source prime manifest.
const PROFILE_Q: u64 = (1_u64 << 53) - 111;
const T: u128 = 1_u128 << 64;

fn descriptor(n: usize, q: u64) -> Destination {
    Destination {
        degree: n, rank: 1, limb: Limb { id: 17, modulus: q },
        layout: Layout::ConstantBASecretOrder, sign: Sign::BPlusAS,
        scale: Scale { numerator: 1_u64 << 36, denominator: 1 },
        key: KeyDomain { label: 91, dimension: n },
    }
}

fn run(c: Torus64Conversion, input: &[u64], output: &mut [u64]) {
    let q = c.quote();
    let result = c.convert_into(input, output, q.word_ops, q.caller_bytes).unwrap();
    assert_eq!(result.source_descriptor(), c.source_descriptor());
    assert_eq!(result.words().len(), q.output_words);
}

// Different arithmetic structure: floor quotient plus remainder comparison.
fn nearest(x: u64, q: u64) -> u64 {
    let scaled = u128::from(x).checked_mul(T).unwrap();
    let q = u128::from(q);
    let twice_rem = (scaled % q).checked_mul(2).unwrap();
    assert_ne!(twice_rem, q, "odd q cannot give an exact half tie");
    ((scaled / q + u128::from(twice_rem > q)) % T) as u64
}

fn phase(output: &[u64], secret: &[u64]) -> u64 {
    output[1..].iter().zip(secret).fold(output[0], |p, (&a, &s)| {
        assert!(s <= 1);
        p.wrapping_sub(a.wrapping_mul(s))
    })
}

fn source_phase(input: &[u64], secret: &[u64], q: u64) -> u64 {
    input[1..].iter().zip(secret).fold(input[0], |p, (&a, &s)| {
        (u128::from(p).checked_add(u128::from(a).checked_mul(u128::from(s)).unwrap())
            .unwrap() % u128::from(q)) as u64
    })
}

// Exact circular distance numerator, denominator q*2^64 in normalized units.
// BigInt avoids overflowing q*2^64 and intermediate signed differences.
fn distance(torus_phase: u64, residue: u64, q: u64) -> BigInt {
    let modulus = BigInt::from(q) * BigInt::from(T);
    let raw = BigInt::from(torus_phase) * BigInt::from(q)
        - BigInt::from(residue) * BigInt::from(T);
    let forward = ((raw % &modulus) + &modulus) % &modulus;
    let backward = &modulus - &forward;
    forward.min(backward)
}

#[test]
fn independent_components_boundaries_sign_and_modular_wrap() {
    for q in [4097, 4099, 8191, PROFILE_Q, (1_u64 << 63) - 1] {
        let c = Torus64Conversion::new(descriptor(1, q)).unwrap();
        // Exhaust small moduli; large cases include nearest half-boundary neighbors.
        let half_neighbor = q / (2 * (T % u128::from(q)) as u64);
        let values: Vec<u64> = if q < 10000 { (0..q).collect() } else {
            vec![0, 1, 2, q / 2 - 1, q / 2, q / 2 + 1, q - 2, q - 1,
                half_neighbor - 1, half_neighbor, half_neighbor + 1]
        };
        for x in values {
            let mut output = [77; 2];
            run(c, &[x, x], &mut output);
            assert_eq!(output, [nearest(x, q), nearest(x, q).wrapping_neg()]);
            assert_eq!(phase(&output, &[1]), nearest(x, q).wrapping_mul(2));
        }
        let mut output = [77; 2];
        run(c, &[q - 1, 1], &mut output);
        assert_eq!(phase(&output, &[1]), 0);
        assert!(output[0] > u64::MAX / 2);
    }
}

#[test]
fn rational_phase_nonzero_errors_and_worst_weight_bound() {
    for q in [4097, 8191, PROFILE_Q] {
        for n in [1, 7, MAX_DIMENSION] {
            let c = Torus64Conversion::new(descriptor(n, q)).unwrap();
            let (num, den) = c.torus_scale();
            assert_eq!((num, den), (T.checked_mul(1_u128 << 36).unwrap(), u128::from(q)));
            for kind in 0..3 {
                let secret: Vec<u64> = (0..n).map(|i| match kind {
                    0 => 0, 1 => 1, _ => u64::from(i % 3 != 0),
                }).collect();
                let mut input = vec![0; n + 1];
                for (i, a) in input[1..].iter_mut().enumerate() {
                    *a = (17 * i as u64 + q / 3) % (q - 1) + 1;
                }
                let mask_phase = source_phase(&input, &secret, q);
                let ideal = q / 4;
                for error in [-3_i128, -1, 0, 1, 3] {
                    let noisy = (i128::from(ideal) + error).rem_euclid(i128::from(q)) as u64;
                    input[0] = ((u128::from(noisy) + u128::from(q)
                        - u128::from(mask_phase)) % u128::from(q)) as u64;
                    let mut output = vec![0; n + 1];
                    run(c, &input, &mut output);
                    let p = phase(&output, &secret);
                    assert_eq!(source_phase(&input, &secret, q), noisy);
                    let h: u64 = secret.iter().sum();
                    let d = distance(p, noisy, q);
                    assert!(&d * 2 <= BigInt::from(h + 1) * BigInt::from(q));
                    assert!(d * 2 <= BigInt::from(n + 1) * BigInt::from(q));
                    // Triangle bound includes scaled incoming E, not zero-error fiction.
                    assert!(distance(p, ideal, q) * 2 <=
                        BigInt::from(n + 1) * BigInt::from(q)
                        + BigInt::from(error.unsigned_abs()) * BigInt::from(T) * 2);
                    if error != 0 {
                        assert!(distance(p, ideal, q) > BigInt::from(0));
                    }
                }
            }
        }
    }
}

#[test]
fn exact_quote_and_atomic_work_storage_length_canonical_refusal() {
    let c = Torus64Conversion::new(descriptor(3, PROFILE_Q)).unwrap();
    let q = c.quote();
    assert_eq!((q.input_words, q.output_words, q.scratch_words), (4, 4, 0));
    assert_eq!(q.caller_bytes, 64);
    assert_eq!(q.word_ops, 64 * 4 + 128);
    assert!(q.descriptor_bytes >= std::mem::size_of_val(&c));
    for (work, bytes, input, len, expected) in [
        (q.word_ops - 1, q.caller_bytes, vec![0; 4], 4, Error::ResourceLimit),
        (q.word_ops, q.caller_bytes - 1, vec![0; 4], 4, Error::ResourceLimit),
        (q.word_ops, q.caller_bytes, vec![0; 3], 4, Error::BufferLength),
        (q.word_ops, q.caller_bytes, vec![0; 5], 4, Error::BufferLength),
        (q.word_ops, q.caller_bytes, vec![0; 4], 3, Error::BufferLength),
        (q.word_ops, q.caller_bytes, vec![0; 4], 5, Error::BufferLength),
    ] {
        let mut output = vec![0xdead; len];
        assert_eq!(c.convert_into(&input, &mut output, work, bytes).err(), Some(expected));
        assert!(output.iter().all(|&x| x == 0xdead));
    }
    for at in 0..4 {
        for bad in [PROFILE_Q, u64::MAX] {
            let mut input = [1; 4];
            input[at] = bad;
            let mut output = [0xdead; 4];
            assert_eq!(c.convert_into(&input, &mut output, q.word_ops, q.caller_bytes).err(),
                Some(Error::NonCanonical));
            assert_eq!(output, [0xdead; 4]);
        }
    }
    run(c, &[1, 2, 3, PROFILE_Q - 1], &mut [0; 4]);
}

#[test]
fn descriptors_dimension_limits_and_checked_quote_overflow() {
    assert_eq!(Torus64Conversion::new(descriptor(0, PROFILE_Q)), Err(Error::InvalidDegree));
    assert_eq!(Torus64Conversion::new(descriptor(MAX_DIMENSION + 1, PROFILE_Q)),
        Err(Error::InvalidDegree));
    for n in [usize::MAX, usize::MAX / 2] {
        assert_eq!(Torus64Conversion::new(descriptor(n, PROFILE_Q)), Err(Error::Overflow));
    }
    let max = Torus64Conversion::new(descriptor(MAX_DIMENSION, PROFILE_Q)).unwrap();
    assert_eq!(max.quote().input_words, MAX_DIMENSION + 1);
    for q in [0, 4095, 4096, 4098, 1_u64 << 63, u64::MAX] {
        assert_eq!(Torus64Conversion::new(descriptor(1, q)), Err(Error::InvalidModulus));
    }
    let d = descriptor(4, PROFILE_Q);
    for bad in [Destination { rank: 2, ..d }, Destination { degree: 3, ..d },
        Destination { layout: Layout::OrderedLimbBA, ..d },
        Destination { scale: Scale { numerator: 0, denominator: 1 }, ..d },
        Destination { scale: Scale { numerator: 1, denominator: 0 }, ..d }] {
        assert_eq!(Torus64Conversion::new(bad), Err(Error::InvalidGeometry));
    }
}

#[test]
fn fixed_public_work_no_secret_argument_or_new_key_authority() {
    let d = descriptor(3, PROFILE_Q);
    let c = Torus64Conversion::new(d).unwrap();
    for input in [[0; 4], [1, 2, 3, 4], [PROFILE_Q - 1; 4]] {
        let before = c.quote();
        let mut output = [0; 4];
        run(c, &input, &mut output);
        for secret in [[0; 3], [1; 3], [1, 0, 1]] {
            let _public_test_phase = phase(&output, &secret);
            assert_eq!(c.quote(), before);
        }
        assert_eq!(c.source_descriptor().key, d.key);
    }
    // Labels remain caller consistency, never authentication or admission.
    let relabeled = Destination { key: KeyDomain { label: 999, ..d.key }, ..d };
    assert_eq!(Torus64Conversion::new(relabeled).unwrap().quote(), c.quote());
}

#[test]
fn coarse_rotation_crosscheck_and_concrete_non_quant4096_contract() {
    let q = PROFILE_Q;
    let c = Torus64Conversion::new(descriptor(1, q)).unwrap();
    let down = CoefficientConversion::new(12, u128::from(q)).unwrap();
    let coarse = |x: u64| ((u128::from(x) * 4096 + T / 2) / T % 4096) as u64;
    for b in [0, 1, q / 4, q / 2, q - 1] {
        let mut output = [0; 2];
        run(c, &[b, 0], &mut output);
        assert_eq!(coarse(output[0]), down.down(u128::from(b)).unwrap());
    }
    // Rounding individual components straight into 4096 cells is NOT rounding
    // the whole phase. q/8192 is just below the first half-cell boundary.
    let a = q / 8192;
    let mut output = [0; 2];
    run(c, &[a, a], &mut output);
    let whole = down.down(u128::from(2 * a)).unwrap();
    let componentwise = (2 * down.down(u128::from(a)).unwrap()) % 4096;
    assert_eq!((componentwise, whole), (0, 1));
    assert_eq!(coarse(phase(&output, &[1])), whole);
    // This *coarse componentwise alternative* exceeds 1/8192, not the Torus64
    // rounding bound. Do not confuse these two distinct conversions.
    assert!(distance(0, 2 * a, q) * 8192 > BigInt::from(q) * BigInt::from(T));
    // Actual Torus64 phase crosses a coarse cell boundary too. Since
    // 2^64 = 2048*q + 227328, this mask rounds down by almost half a unit.
    // 128 such errors exceed the ideal phase's ~27.75-unit boundary margin.
    let n = 128;
    let fine = Torus64Conversion::new(descriptor(n, q)).unwrap();
    let mask = q / (2 * 227328);
    let residue = 1_u64 << 40;
    let mut input = vec![mask; n + 1];
    input[0] = ((u128::from(residue) + u128::from(q)
        - (u128::from(mask) * n as u128) % u128::from(q)) % u128::from(q)) as u64;
    let mut converted = vec![0; n + 1];
    run(fine, &input, &mut converted);
    let secret = vec![1; n];
    assert_eq!(source_phase(&input, &secret, q), residue);
    assert_eq!(down.down(u128::from(residue)), Ok(1));
    assert_eq!(coarse(phase(&converted, &secret)), 0);
    // Witness substantial accumulation, not merely a vacuous upper bound.
    let d = distance(phase(&converted, &secret), residue, q);
    assert!(&d * 4 > BigInt::from(n) * BigInt::from(q));
    assert!(d * 2 <= BigInt::from(n + 1) * BigInt::from(q));
    // No unconditional total-error bound: a nonzero incoming E is preserved,
    // scaled, not magically removed. Here ideal phase is zero, E=q/8192+1.
    run(c, &[q / 8192 + 1, 0], &mut output);
    assert!(distance(output[0], 0, q) * 8192 > BigInt::from(q) * BigInt::from(T));
    // Even this fine Torus64 kernel is approximate, not exact rational equality.
    run(c, &[1, 0], &mut output);
    assert!(distance(output[0], 1, q) > BigInt::from(0));
    // Nearest coarse rotation is not exact floor.
    run(c, &[q / 4096 - 1, 0], &mut output);
    assert_eq!(coarse(output[0]), 1);
    assert_eq!(u128::from(q / 4096 - 1) * 4096 / u128::from(q), 0);
}
