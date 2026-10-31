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

// Independent unbounded integer rational oracle: compare distances to the two
// neighboring integers, then select the even integer on equality. No production
// helper, fixed-width product, remainder doubling or candidate roundtrip used.
fn reverse_oracle(x: u64, q: u64) -> u64 {
    let numerator = BigInt::from(q) * BigInt::from(x);
    let denominator = BigInt::from(T);
    let lower = &numerator / &denominator;
    let upper = &lower + BigInt::from(1);
    let below = &numerator - &lower * &denominator;
    let above = &upper * &denominator - &numerator;
    let choose_lower = below < above
        || (below == above && &lower % BigInt::from(2) == BigInt::from(0));
    let rounded = if choose_lower { lower } else { upper };
    u64::try_from(rounded % BigInt::from(q)).unwrap()
}

fn reverse_expected(words: &[u64], q: u64) -> Vec<u64> {
    words.iter().enumerate().map(|(i, &x)| {
        let rounded = reverse_oracle(x, q);
        if i == 0 || rounded == 0 { rounded } else { q - rounded }
    }).collect()
}

#[test]
fn reverse_bigint_ties_neighbors_wrap_and_round_before_negation() {
    for q in [4097, 4099, 8191, PROFILE_Q, (1_u64 << 63) - 1] {
        let c = Torus64Conversion::new(descriptor(1, q)).unwrap();
        let f = c.quote();
        let r = c.reverse_quote();
        let mut backing = [0; 2];
        let mut view = c.convert_into(&[0; 2], &mut backing, f.word_ops, f.caller_bytes).unwrap();
        let half = 1_u64 << 63;
        for x in [0, 1, 2, half - 1, half, half + 1, u64::MAX - 1, u64::MAX] {
            view.words_mut().copy_from_slice(&[x, x]);
            let mut output = [0xdead; 2];
            let result = c.reverse_into(&view, &mut output, r.word_ops, r.caller_bytes).unwrap();
            assert_eq!(result.descriptor(), c.source_descriptor());
            assert_eq!(result.words(), reverse_expected(&[x, x], q));
            assert!(result.words().iter().all(|&word| word < q));
            if x == half {
                let even = if q % 4 == 1 { q / 2 } else { q / 2 + 1 };
                assert_eq!(result.words(), &[even, q - even]);
                // Torus negation fixes half, but odd-q negation of its even
                // rounded image does NOT: pre-negation differs by one q-unit.
                assert_eq!(x.wrapping_neg(), x);
                assert_ne!(result.words()[1], reverse_oracle(x.wrapping_neg(), q));
                assert_eq!(result.words()[1].abs_diff(even), 1);
                // Separate omitted-negation negative control.
                assert_ne!(result.words()[1], reverse_oracle(x, q));
            }
            if x == u64::MAX {
                assert_eq!(result.words(), &[0, 0], "rounded q reduces to zero");
            }
        }
        // Borrowed wrapper adds no implicit erasure to either caller buffer.
        drop(view);
        assert_eq!(backing, [u64::MAX; 2]);
    }
}

#[test]
fn unchanged_forward_reverse_recovers_canonical_components_not_arbitrary_torus() {
    for q in [4097, 4099, PROFILE_Q, (1_u64 << 63) - 1] {
        let c = Torus64Conversion::new(descriptor(1, q)).unwrap();
        let f = c.quote();
        let r = c.reverse_quote();
        let values: Vec<u64> = if q < 10000 { (0..q).collect() } else {
            vec![0, 1, 2, q / 2 - 1, q / 2, q / 2 + 1, q - 2, q - 1]
        };
        for x in values {
            let input = [x, q - 1 - x];
            let mut backing = [0; 2];
            let view = c.convert_into(&input, &mut backing, f.word_ops, f.caller_bytes).unwrap();
            // Oracle is independent of the roundtrip identity being tested.
            let expected = reverse_expected(view.words(), q);
            assert_eq!(expected, input);
            let mut recovered = [0xdead; 2];
            let result = c.reverse_into(&view, &mut recovered, r.word_ops, r.caller_bytes).unwrap();
            assert_eq!(result.words(), expected);
            assert_eq!(result.words(), input);
        }
        let mut backing = [0; 2];
        let mut view = c.convert_into(&[0; 2], &mut backing, f.word_ops, f.caller_bytes).unwrap();
        view.words_mut().copy_from_slice(&[1, u64::MAX]);
        let mut recovered = [0xdead; 2];
        let result = c.reverse_into(&view, &mut recovered, r.word_ops, r.caller_bytes).unwrap();
        assert_eq!(result.words(), &[0, 0]);
        let mut again = [0xdead; 2];
        let remapped = c.convert_into(result.words(), &mut again, f.word_ops, f.caller_bytes).unwrap();
        assert_eq!(remapped.words(), &[0, 0]);
        assert_ne!(remapped.words(), view.words());
    }
}

#[test]
fn reverse_binary_phase_bounds_incoming_error_composition_and_dimension_2048() {
    for q in [4097, 4099, PROFILE_Q, (1_u64 << 63) - 1] {
        for n in [4, MAX_DIMENSION] {
            let c = Torus64Conversion::new(descriptor(n, q)).unwrap();
            let f = c.quote();
            let r = c.reverse_quote();
            let mut backing = vec![0; n + 1];
            let input = vec![0; n + 1];
            let mut view = c.convert_into(&input, &mut backing, f.word_ops, f.caller_bytes).unwrap();
            let fixtures = [0, u64::MAX / 3, (1_u64 << 63) - 1, 1_u64 << 63,
                (1_u64 << 63) + 1, u64::MAX, 1];
            for (i, x) in view.words_mut().iter_mut().enumerate() {
                *x = fixtures[i % fixtures.len()];
            }
            let mut recovered = vec![0xdead; n + 1];
            let result = c.reverse_into(&view, &mut recovered, r.word_ops, r.caller_bytes).unwrap();
            assert_eq!(result.words(), reverse_expected(view.words(), q));
            let mut nonzero = false;
            let mut wrong_sign_exceeds_bound = false;
            // All small public binary vectors; large dimension checks zero,
            // full and patterned weights without truncating any coordinate.
            let patterns = if n == 4 { 16 } else { 3 };
            for bits in 0..patterns {
                let secret: Vec<u64> = (0..n).map(|i| if n == 4 {
                    ((bits >> i) & 1) as u64
                } else {
                    match bits { 0 => 0, 1 => 1, _ => u64::from(i % 3 != 0) }
                }).collect();
                let h: u64 = secret.iter().sum();
                let torus_phase = phase(view.words(), &secret);
                let residue = source_phase(result.words(), &secret, q);
                let d = distance(torus_phase, residue, q);
                // distance denominator T gives q-units (not torus units).
                assert!(&d * 2 <= BigInt::from(h + 1) * BigInt::from(T));
                nonzero |= d != BigInt::from(0);
                let wrong: Vec<u64> = view.words().iter().map(|&x| reverse_oracle(x, q)).collect();
                wrong_sign_exceeds_bound |= distance(torus_phase, source_phase(&wrong, &secret, q), q)
                    * 2 > BigInt::from(h + 1) * BigInt::from(T);
                for incoming in [-7_i64, -1, 0, 1, 7] {
                    let ideal_torus = torus_phase.wrapping_sub(incoming as u64);
                    assert!(distance(ideal_torus, residue, q) * 2 <=
                        BigInt::from(h + 1) * BigInt::from(T)
                        + BigInt::from(q) * BigInt::from(incoming.unsigned_abs()) * 2);
                }
            }
            assert!(nonzero);
            assert!(wrong_sign_exceeds_bound);
            // Canonical recovery control only: its zero composition distance
            // cannot discriminate bound coefficients. The boundary-adversary
            // test below independently exercises nonzero two-stage accounting.
            let canonical: Vec<u64> = (0..=n).map(|i| (i as u64 * 17 + 1) % q).collect();
            let mut forward_backing = vec![0; n + 1];
            let forward = c.convert_into(&canonical, &mut forward_backing, f.word_ops, f.caller_bytes).unwrap();
            let mut final_backing = vec![0xdead; n + 1];
            let final_view = c.reverse_into(&forward, &mut final_backing, r.word_ops, r.caller_bytes).unwrap();
            assert_eq!(final_view.words(), canonical);
            let secret = vec![1; n];
            let original_phase = source_phase(&canonical, &secret, q);
            let final_phase = source_phase(final_view.words(), &secret, q);
            let delta = (final_phase + q - original_phase) % q;
            let circular = delta.min(q - delta);
            assert!(BigInt::from(circular) * BigInt::from(T) * 2 <=
                BigInt::from(n + 1) * (BigInt::from(T) + BigInt::from(q)));
            assert_eq!(r.input_words, n + 1);
            assert_eq!(r.output_words, n + 1);
        }
    }
}

#[test]
fn reverse_exact_quotes_atomic_caps_lengths_and_original_descriptor_binding() {
    use pllm_core::coefficient_torus_conversion::{CoefficientCiphertext, Torus64Ciphertext};
    let d = descriptor(3, PROFILE_Q);
    let c = Torus64Conversion::new(d).unwrap();
    let f = c.quote();
    let r = c.reverse_quote();
    assert_eq!((r.input_words, r.output_words, r.scratch_words), (4, 4, 0));
    assert_eq!(r.caller_bytes, 64);
    assert_eq!(r.word_ops, 64 * 4 + 128);
    assert_eq!(r.descriptor_bytes, std::mem::size_of::<Torus64Conversion>()
        + std::mem::size_of::<Torus64Ciphertext<'static>>()
        + std::mem::size_of::<CoefficientCiphertext<'static>>());
    let mut backing = [0; 4];
    let view = c.convert_into(&[1, 2, 3, PROFILE_Q - 1], &mut backing, f.word_ops, f.caller_bytes).unwrap();
    let before = view.words().to_vec();
    for (work, bytes, len, error) in [
        (r.word_ops - 1, r.caller_bytes, 4, Error::ResourceLimit),
        (r.word_ops, r.caller_bytes - 1, 4, Error::ResourceLimit),
        (r.word_ops, r.caller_bytes, 0, Error::BufferLength),
        (r.word_ops, r.caller_bytes, 3, Error::BufferLength),
        (r.word_ops, r.caller_bytes, 5, Error::BufferLength),
    ] {
        let mut output = vec![0xdead; len];
        assert_eq!(c.reverse_into(&view, &mut output, work, bytes).err(), Some(error));
        assert!(output.iter().all(|&x| x == 0xdead));
        assert_eq!(view.words(), before);
    }
    for other in [
        Destination { key: KeyDomain { label: 92, ..d.key }, ..d },
        Destination { scale: Scale { numerator: 1, denominator: 2 }, ..d },
        Destination { limb: Limb { id: 18, ..d.limb }, ..d },
        Destination { limb: Limb { modulus: 4099, ..d.limb }, ..d },
        descriptor(2, PROFILE_Q),
    ] {
        let other = Torus64Conversion::new(other).unwrap();
        let mut output = [0xdead; 4];
        assert_eq!(other.reverse_into(&view, &mut output, usize::MAX, usize::MAX).err(),
            Some(Error::InvalidGeometry));
        assert_eq!(output, [0xdead; 4]);
    }
    let mut output = [0xdead; 4];
    let result = c.reverse_into(&view, &mut output, r.word_ops, r.caller_bytes).unwrap();
    assert_eq!(result.words(), &[1, 2, 3, PROFILE_Q - 1]);
    assert_eq!(result.descriptor(), d);
    assert_eq!(c.reverse_quote(), r);
    drop(result);
    assert_eq!(output, [1, 2, 3, PROFILE_Q - 1], "caller retains cleanup");
    assert_eq!(view.words(), before);
    let mut guarded = [0xfeed; 6];
    let exact_subslice = c.reverse_into(
        &view, &mut guarded[1..5], r.word_ops, r.caller_bytes,
    ).unwrap();
    assert_eq!(exact_subslice.words(), &[1, 2, 3, PROFILE_Q - 1]);
    drop(exact_subslice);
    assert_eq!(guarded, [0xfeed, 1, 2, 3, PROFILE_Q - 1, 0xfeed]);
    assert_eq!(c.reverse_into(
        &view, &mut guarded[1..1], r.word_ops, r.caller_bytes,
    ).err(), Some(Error::BufferLength));
    assert_eq!(guarded, [0xfeed, 1, 2, 3, PROFILE_Q - 1, 0xfeed]);
    assert_eq!(view.words(), before);
    // Hostile dimension arithmetic and malformed modulus/geometry are refused
    // by the shared constructor, before either directional API exists.
    for n in [usize::MAX, usize::MAX / 2] {
        assert_eq!(Torus64Conversion::new(descriptor(n, PROFILE_Q)), Err(Error::Overflow));
    }
}

// Unreduced signed integer phases, independent of wrapping u64 helpers. Retain
// these lifts when relating two stages: reducing first hides multiples of q*T.
fn integer_phase(words: &[u64], secret: &[u64], subtract_masks: bool) -> BigInt {
    assert_eq!(words.len(), secret.len() + 1);
    words[1..].iter().zip(secret).fold(BigInt::from(words[0]), |p, (&a, &s)| {
        assert!(s <= 1);
        let term = BigInt::from(a) * BigInt::from(s);
        if subtract_masks { p - term } else { p + term }
    })
}

fn integer_residue(value: &BigInt, modulus: &BigInt) -> BigInt {
    ((value % modulus) + modulus) % modulus
}

#[test]
fn independent_boundary_adversary_discriminates_two_stage_error_coefficient() {
    // Hand derivation, not a production conversion helper:
    // q=2^32+1, c=(q-1)/2, K=2^32-1, q*K=T-1.
    // Forward(c)=K*c=F; its error F-T*c/q=-c/q, almost -1/2
    // torus WORD per selected component (forward masks negate F).
    // x_minus=T/2+2^32-1: q*x_minus=(c+1)*T+T/2-1;
    // x_plus =T/2-2^32+1: q*x_plus =(c-1)*T+T/2+1.
    // Thus U(x_minus)=c+1, U(x_plus)=c. Reverse B error and
    // NEGATED-mask errors all equal -1/2+1/T q-units.
    // This public translation is solely an arithmetic boundary adversary, not
    // a qualified protected operation, HomComp encoding or scale authority.
    const Q: u64 = (1_u64 << 32) + 1;
    const C: u64 = 1_u64 << 31;
    const K: u64 = (1_u64 << 32) - 1;
    const F: u64 = K * C;
    const X_MINUS: u64 = (1_u64 << 63) + K;
    const X_PLUS: u64 = (1_u64 << 63) - K;
    let q = BigInt::from(Q);
    let t = BigInt::from(T);
    let c_word = BigInt::from(C);
    let one = BigInt::from(1);
    let two = BigInt::from(2);
    let three = BigInt::from(3);
    assert_eq!(&q * BigInt::from(K), &t - &one);
    assert_eq!(&q * BigInt::from(X_MINUS), (&c_word + &one) * &t + &t / &two - &one);
    assert_eq!(&q * BigInt::from(X_PLUS), (&c_word - &one) * &t + &t / &two + &one);
    for n in [4, MAX_DIMENSION] {
        let conversion = Torus64Conversion::new(descriptor(n, Q)).unwrap();
        let fq = conversion.quote();
        let rq = conversion.reverse_quote();
        let input = vec![C; n + 1];
        let secret = vec![1; n];
        let m = BigInt::from(n + 1);
        let n_big = BigInt::from(n);
        let source_raw = integer_phase(&input, &secret, false);
        assert_eq!(source_raw, &m * &c_word);
        assert!(source_raw > q, "force actual coefficient-domain wrap");
        let mut backing = vec![0; n + 1];
        let mut forward = conversion.convert_into(
            &input, &mut backing, fq.word_ops, fq.caller_bytes,
        ).unwrap();
        assert_eq!(forward.words()[0], F);
        assert!(forward.words()[1..].iter().all(|&a| a == F.wrapping_neg()));
        let forward_raw = integer_phase(forward.words(), &secret, true);
        assert_eq!(forward_raw, &m * BigInt::from(F) - &n_big * &t);
        assert!(forward_raw < BigInt::from(0), "force actual torus-domain wrap");
        let forward_residue = integer_residue(&forward_raw, &t);
        assert_eq!(BigInt::from(phase(forward.words(), &secret)), forward_residue);
        let source_residue = integer_residue(&source_raw, &q);
        assert_eq!(BigInt::from(source_phase(&input, &secret, Q)), source_residue);

        // Lift by the independently known n*T wraps. e_forward_numerator/q
        // is in torus WORD units; the SAME numerator/T is in q-units.
        let forward_lift = &forward_raw + &n_big * &t;
        let e_forward_numerator = &q * &forward_lift - &t * &source_raw;
        assert_eq!(e_forward_numerator, -(&m * &c_word));
        let forward_abs_numerator = -&e_forward_numerator;
        assert!(&forward_abs_numerator * 2 <= &m * &q);
        assert!(&forward_abs_numerator * 3 > &m * &q,
            "substantial near-half growth, not merely nonzero error");
        assert_eq!(distance(u64::try_from(forward_residue).unwrap(),
            u64::try_from(source_residue).unwrap(), Q), forward_abs_numerator);

        forward.words_mut()[0] = X_MINUS;
        forward.words_mut()[1..].fill(X_PLUS);
        let processed_raw = integer_phase(forward.words(), &secret, true);
        assert_eq!(processed_raw, BigInt::from(X_MINUS) - &n_big * BigInt::from(X_PLUS));
        assert!(processed_raw < BigInt::from(0));
        // Exact signed public phase translation; not an unknown noise budget.
        let translation = &processed_raw - &forward_raw;
        let mut output = vec![0xdead; n + 1];
        let reverse = conversion.reverse_into(
            &forward, &mut output, rq.word_ops, rq.caller_bytes,
        ).unwrap();
        assert!(reverse.words().iter().all(|&word| word == C + 1));
        assert_eq!(reverse.words(), reverse_expected(forward.words(), Q));
        let reverse_raw = integer_phase(reverse.words(), &secret, false);
        assert_eq!(reverse_raw, &m * (&c_word + &one));
        assert!(reverse_raw > q, "force output wrap too");
        // Reverse lift subtracts n*q because masks were canonically negated.
        // e_reverse_numerator/T is q-units. Every selected term aligns.
        let reverse_lift = &reverse_raw - &n_big * &q;
        let e_reverse_numerator = &t * &reverse_lift - &q * &processed_raw;
        assert_eq!(e_reverse_numerator, -(&m * (&t / &two - &one)));
        let reverse_abs_numerator = -&e_reverse_numerator;
        assert!(&reverse_abs_numerator * 2 <= &m * &t);
        assert!(&reverse_abs_numerator * 3 > &m * &t);
        assert_eq!(distance(phase(forward.words(), &secret),
            source_phase(reverse.words(), &secret, Q), Q), reverse_abs_numerator);

        // Ideal translated q-phase is source_raw + q*translation/T. Compose
        // using denominator T throughout; independent wrap multiples cancel.
        let final_numerator = &t * (&reverse_raw - &source_raw) - &q * &translation;
        assert_eq!(final_numerator, &e_forward_numerator + &e_reverse_numerator);
        let expected_magnitude = &m * (&t / &two + &c_word - &one);
        assert_eq!(final_numerator, -&expected_magnitude);
        let period = &q * &t;
        let positive = integer_residue(&final_numerator, &period);
        let final_distance = positive.clone().min(&period - &positive);
        assert_eq!(final_distance, expected_magnitude);
        assert!(&final_distance * 2 <= &m * (&t + &q));
        assert!(&final_distance * 2 > &m * &t,
            "reverse-only composition budget MUST fail this adversary");
        assert!(&final_distance * 4 > &m * (&t * 2 + &q),
            "even halving the q*m/(2*T) forward contribution MUST fail");
        // Strength limit: finite q leaves slack 3*m/(2*T). A multiplier alpha
        // on the forward budget fails exactly when alpha < 1-3/q here. This
        // rejects omitted/halved terms, NOT every smaller coefficient or every
        // possible implementation bug; it is no universal sharpness proof.
        assert_eq!(&m * (&t + &q) - &final_distance * &two, &m * &three);
    }
}

#[test]
fn nonzero_public_incoming_torus_error_survives_reverse_with_exact_phase_witness() {
    const Q: u64 = (1_u64 << 32) + 1;
    const C: u64 = 1_u64 << 31;
    const K: u64 = (1_u64 << 32) - 1;
    let n = 4;
    let conversion = Torus64Conversion::new(descriptor(n, Q)).unwrap();
    let fq = conversion.quote();
    let rq = conversion.reverse_quote();
    let input = [C; 5];
    let secret = [1; 4];
    let mut backing = [0; 5];
    let mut forward = conversion.convert_into(
        &input, &mut backing, fq.word_ops, fq.caller_bytes,
    ).unwrap();
    let before_raw = integer_phase(forward.words(), &secret, true);
    let incoming = 3 * K; // torus WORD units, q*incoming=3*(T-1).
    forward.words_mut()[0] += incoming;
    let after_raw = integer_phase(forward.words(), &secret, true);
    assert_eq!(after_raw - &before_raw, BigInt::from(incoming));
    let mut output = [0xdead; 5];
    let reverse = conversion.reverse_into(
        &forward, &mut output, rq.word_ops, rq.caller_bytes,
    ).unwrap();
    // q*(F+3K)/T=c+3-(c+3)/T, uniquely nearest to c+3.
    assert_eq!(reverse.words(), &[C + 3, C, C, C, C]);
    assert_eq!(reverse.words(), reverse_expected(forward.words(), Q));
    let original = integer_phase(&input, &secret, false);
    let actual = integer_phase(reverse.words(), &secret, false);
    assert_eq!(&actual - &original, BigInt::from(3));
    let residue_delta = integer_residue(&(&actual - &original), &BigInt::from(Q));
    let circular = residue_delta.clone().min(BigInt::from(Q) - residue_delta);
    assert_eq!(circular, BigInt::from(3), "nonzero incoming error not removed");
    let t = BigInt::from(T);
    let q = BigInt::from(Q);
    let m = BigInt::from(n + 1);
    // Full two-stage budget: reverse + rescaled forward + incoming torus error.
    assert!(&circular * &t * 2 <= &m * &t + &m * &q + &q * BigInt::from(incoming) * 2);
    assert!(&circular * &t * 2 > &m * (&t + &q),
        "dropping incoming term would fail, not silently pass with zero LHS");
}
