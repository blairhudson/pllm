//! Independent CRT and signed floor-division definitions. Source-only, UNRUN.
use super::*;
use crate::coefficient_layout::DegreeBound;
use num_bigint::BigInt;

const UNLIMITED: DropChainWork = DropChainWork {
    degree_checks: usize::MAX,
    pair_checks: usize::MAX,
    canonical_checks: usize::MAX,
    projections: usize::MAX,
    word_stores: usize::MAX,
};

fn rings(degree: usize, moduli: &[u64]) -> Vec<CoefficientRing> {
    moduli.iter().map(|&q| CoefficientRing::new(degree, q).unwrap()).collect()
}

fn pairs(rings: &[CoefficientRing]) -> Vec<RoundedLimbDrop> {
    let mut pairs = Vec::new();
    for j in 1..rings.len() {
        for i in 0..j {
            pairs.push(RoundedLimbDrop::new(rings[i], rings[j]).unwrap());
        }
    }
    pairs
}

fn positive_mod(value: BigInt, modulus: &BigInt) -> BigInt {
    ((value % modulus) + modulus) % modulus
}

fn word(value: BigInt) -> u64 {
    assert!(value >= BigInt::from(0));
    let (_, digits) = value.to_u64_digits();
    assert!(digits.len() <= 1);
    digits.first().copied().unwrap_or(0)
}

// BigInt / truncates toward zero. Correct its negative, nonintegral cases to
// mathematical floor before using nearest rounding for an odd positive divisor.
fn floor_div(value: BigInt, divisor: &BigInt) -> BigInt {
    let remainder = &value % divisor;
    let quotient = value / divisor;
    if remainder < BigInt::from(0) { quotient - 1 } else { quotient }
}

fn nearest(value: BigInt, divisor: &BigInt) -> BigInt {
    floor_div(value + (divisor - 1) / 2, divisor)
}

// Independent global CRT sum, not production pair projection or reconstruction.
fn centered_crt(residues: &[u64], moduli: &[u64]) -> BigInt {
    let modulus: BigInt = moduli.iter().map(|&q| BigInt::from(q)).product();
    let mut sum = BigInt::from(0);
    for (&residue, &q) in residues.iter().zip(moduli) {
        let q = BigInt::from(q);
        let cofactor = &modulus / &q;
        let inverse = cofactor.modinv(&q).unwrap();
        sum += BigInt::from(residue) * cofactor * inverse;
    }
    let x = positive_mod(sum, &modulus);
    if x > (&modulus - 1) / 2 { x - modulus } else { x }
}

fn oracle(input: &[u64], degree: usize, moduli: &[u64], steps: usize) -> Vec<u64> {
    let m = moduli.len();
    let r = m - steps;
    let divisor: BigInt = moduli[r..].iter().map(|&q| BigInt::from(q)).product();
    let mut out = vec![0; 2 * r * degree];
    for component in 0..2 {
        for k in 0..degree {
            let residues: Vec<_> = (0..m)
                .map(|i| input[component * m * degree + i * degree + k]).collect();
            let x = centered_crt(&residues, moduli);
            let aggregate = nearest(x.clone(), &divisor);
            let mut sequential = x;
            for &q in moduli[r..].iter().rev() {
                sequential = nearest(sequential, &BigInt::from(q));
            }
            // Odd-divisor offsets telescope exactly to (product-1)/2.
            assert_eq!(sequential, aggregate);
            for (i, &q) in moduli[..r].iter().enumerate() {
                out[component * r * degree + i * degree + k] =
                    word(positive_mod(aggregate.clone(), &BigInt::from(q)));
            }
        }
    }
    out
}

fn encode(values: &[BigInt], degree: usize, moduli: &[u64]) -> Vec<u64> {
    assert_eq!(values.len(), 2 * degree);
    let mut input = vec![0; 2 * moduli.len() * degree];
    for component in 0..2 {
        for (i, &q) in moduli.iter().enumerate() {
            for k in 0..degree {
                input[component * moduli.len() * degree + i * degree + k] =
                    word(positive_mod(values[component * degree + k].clone(), &BigInt::from(q)));
            }
        }
    }
    input
}

fn check_fixture(degree: usize, moduli: &[u64], steps: usize, values: &[BigInt]) {
    let rings = rings(degree, moduli);
    let pairs = pairs(&rings);
    let chain = RoundedDropChain::new(&rings, &pairs, steps, UNLIMITED).unwrap();
    let input = encode(values, degree, moduli);
    let mut output = vec![u64::MAX; chain.geometry().output_words];
    let mut scratch = vec![u64::MAX - 1; chain.geometry().scratch_words + 3];
    chain.apply_into(&input, &mut output, &mut scratch).unwrap();
    assert_eq!(output, oracle(&input, degree, moduli, steps));
    assert_eq!(&scratch[chain.geometry().scratch_words..], &[u64::MAX - 1; 3]);
}

#[test]
fn signed_floor_oracle_and_odd_rounding_boundaries() {
    let d = BigInt::from(9);
    for (x, expected) in [(-14, -2), (-13, -1), (-5, -1), (-4, 0), (4, 0), (5, 1)] {
        assert_eq!(nearest(BigInt::from(x), &d), BigInt::from(expected));
    }
    assert_eq!(floor_div(BigInt::from(-9), &d), BigInt::from(-1));
    assert_eq!(floor_div(BigInt::from(-10), &d), BigInt::from(-2));
}

#[test]
fn finite_exhaustion_two_and_three_limbs_both_components() {
    for moduli in [vec![9, 25], vec![3, 5, 7]] {
        let product: u64 = moduli.iter().product();
        for steps in 1..moduli.len() {
            for x in 0..product {
                check_fixture(1, &moduli, steps, &[BigInt::from(x), -BigInt::from(x)]);
            }
        }
    }
}

fn permutations(values: &mut [u64], at: usize, visit: &mut impl FnMut(&[u64])) {
    if at == values.len() {
        visit(values);
    } else {
        for i in at..values.len() {
            values.swap(at, i);
            permutations(values, at + 1, visit);
            values.swap(at, i);
        }
    }
}

#[test]
fn composite_two_to_four_limb_orders_center_wrap_and_boundaries() {
    for m in 2..=4 {
        let mut moduli = [9, 25, 49, 11];
        permutations(&mut moduli[..m], 0, &mut |order| {
            let product: u64 = order.iter().product();
            for steps in 1..m {
                let divisor: u64 = order[m - steps..].iter().product();
                let half = (divisor - 1) / 2;
                let values = [
                    BigInt::from(half), BigInt::from(half + 1),
                    -BigInt::from(half), -BigInt::from(half + 1),
                    BigInt::from(product - 1), BigInt::from(product / 2),
                    BigInt::from(product / 2 + 1), BigInt::from(0),
                ];
                check_fixture(4, order, steps, &values);
            }
        });
    }
}

#[test]
fn public_8192_sparse_chain_both_components_every_output_word() {
    let moduli = [9, 25, 49];
    let rings: Vec<_> = moduli.iter().map(|&q|
        CoefficientRing::bounded(8192, q, DegreeBound::Public8192).unwrap()).collect();
    let pairs = pairs(&rings);
    for steps in 1..3 {
        let chain = RoundedDropChain::new(&rings, &pairs, steps, UNLIMITED).unwrap();
        let mut values = vec![BigInt::from(0); 2 * 8192];
        for (at, x) in [(0, -612), (1, 613), (4096, -5512), (8191, 5512),
            (8192, 25), (8193, -26), (12288, 1225), (16383, -1226)]
        {
            values[at] = BigInt::from(x);
        }
        let input = encode(&values, 8192, &moduli);
        // Oracle only reconstructs sparse positions; all other expected words
        // remain zero, so this stays linear without dense BigInt CRT work.
        let mut expected = vec![0; chain.geometry().output_words];
        let r = moduli.len() - steps;
        for k in [0, 1, 4096, 8191] {
            let mut residues = Vec::new();
            for component in 0..2 {
                for i in 0..moduli.len() {
                    residues.push(input[component * moduli.len() * 8192 + i * 8192 + k]);
                }
            }
            let sparse = oracle(&residues, 1, &moduli, steps);
            for component in 0..2 {
                for i in 0..r {
                    expected[component * r * 8192 + i * 8192 + k] = sparse[component * r + i];
                }
            }
        }
        let mut output = vec![u64::MAX; expected.len()];
        let mut scratch = vec![u64::MAX; input.len() + 1];
        chain.apply_into(&input, &mut output, &mut scratch).unwrap();
        assert_eq!(output, expected);
        assert_eq!(scratch[input.len()], u64::MAX);
    }
}

#[test]
fn geometry_quotes_exact_payload_descriptors_and_semantic_work() {
    let rings = rings(4, &[9, 25, 49, 11]);
    let pairs = pairs(&rings);
    let chain = RoundedDropChain::new(&rings, &pairs, 2, UNLIMITED).unwrap();
    assert_eq!(chain.geometry(), DropChainGeometry {
        degree: 4, input_limbs: 4, retained_limbs: 2, steps: 2,
        input_words: 32, output_words: 16, scratch_words: 32,
        payload_bytes: 640,
        descriptor_bytes: 4 * size_of::<CoefficientRing>() + 6 * size_of::<RoundedLimbDrop>(),
    });
    assert_eq!(chain.work(), DropChainWork {
        degree_checks: 4, pair_checks: 6, canonical_checks: 32,
        projections: 40, word_stores: 88,
    });
}

#[test]
fn every_exact_and_one_below_semantic_ceiling() {
    let rings = rings(4, &[9, 25, 49, 11]);
    let pairs = pairs(&rings);
    for steps in 1..rings.len() {
        let work = RoundedDropChain::new(&rings, &pairs, steps, UNLIMITED).unwrap().work();
        assert!(RoundedDropChain::new(&rings, &pairs, steps, work).is_ok());
        for field in 0..5 {
            let mut ceiling = work;
            match field {
                0 => ceiling.degree_checks -= 1,
                1 => ceiling.pair_checks -= 1,
                2 => ceiling.canonical_checks -= 1,
                3 => ceiling.projections -= 1,
                _ => ceiling.word_stores -= 1,
            }
            assert_eq!(RoundedDropChain::new(&rings, &pairs, steps, ceiling),
                Err(LayoutError::ResourceLimit));
        }
    }
}

#[test]
fn geometry_then_ceiling_then_pair_length_then_degrees() {
    let mismatched = [CoefficientRing::new(1, 9).unwrap(), CoefficientRing::new(2, 25).unwrap()];
    let zero = DropChainWork { degree_checks: 0, pair_checks: 0, canonical_checks: 0,
        projections: 0, word_stores: 0 };
    assert_eq!(RoundedDropChain::new(&[], &[], 1, zero), Err(LayoutError::InvalidGeometry));
    assert_eq!(RoundedDropChain::new(&mismatched[..1], &[], 1, zero), Err(LayoutError::InvalidGeometry));
    for steps in [0, 2, usize::MAX] {
        assert_eq!(RoundedDropChain::new(&mismatched, &[], steps, zero), Err(LayoutError::InvalidGeometry));
    }
    assert_eq!(RoundedDropChain::new(&mismatched, &[], 1, zero), Err(LayoutError::ResourceLimit));
    assert_eq!(RoundedDropChain::new(&mismatched, &[], 1, UNLIMITED), Err(LayoutError::BufferLength));
    let valid = rings(1, &[9, 25]);
    let pairs = pairs(&valid);
    assert_eq!(RoundedDropChain::new(&mismatched, &pairs, 1, UNLIMITED), Err(LayoutError::InvalidDegree));
}

#[test]
fn exact_triangular_pair_length_and_all_ordered_identities() {
    let rings = rings(1, &[9, 25, 49, 11]);
    let pairs = pairs(&rings);
    assert_eq!(RoundedDropChain::new(&rings, &pairs[..5], 1, UNLIMITED), Err(LayoutError::BufferLength));
    let mut extra = pairs.clone();
    extra.push(pairs[0]);
    assert_eq!(RoundedDropChain::new(&rings, &extra, 1, UNLIMITED), Err(LayoutError::BufferLength));
    for j in 1..rings.len() {
        for i in 0..j {
            let mut wrong = pairs.clone();
            wrong[pair_row(j) + i] = RoundedLimbDrop::new(rings[j], rings[i]).unwrap();
            // Includes retained/retained pairs when only the last limb drops.
            assert_eq!(RoundedDropChain::new(&rings, &wrong, 1, UNLIMITED), Err(LayoutError::InvalidModulus));
        }
    }
    let mut reordered = rings.clone();
    reordered.swap(0, 1);
    assert_eq!(RoundedDropChain::new(&reordered, &pairs, 1, UNLIMITED), Err(LayoutError::InvalidModulus));
}

#[test]
fn descriptor_degree_identity_precedes_its_modulus_identity() {
    let rings = rings(1, &[9, 25, 49]);
    let mut pairs = pairs(&rings);
    pairs[0] = RoundedLimbDrop::new(CoefficientRing::new(2, 7).unwrap(),
        CoefficientRing::new(2, 11).unwrap()).unwrap();
    assert_eq!(RoundedDropChain::new(&rings, &pairs, 1, UNLIMITED), Err(LayoutError::InvalidDegree));
}

#[test]
fn pair_constructors_exclude_duplicate_and_non_coprime_composites() {
    for moduli in [[9, 25, 9], [9, 25, 15]] {
        let rings = rings(1, &moduli);
        assert_eq!(RoundedLimbDrop::new(rings[0], rings[2]), Err(LayoutError::InvalidModulus));
    }
    assert!(RoundedLimbDrop::new(rings(1, &[9])[0], rings(1, &[25])[0]).is_ok());
}

#[test]
fn all_length_refusals_preserve_entire_output_and_scratch() {
    let rings = rings(2, &[9, 25, 49]);
    let pairs = pairs(&rings);
    let chain = RoundedDropChain::new(&rings, &pairs, 1, UNLIMITED).unwrap();
    let g = chain.geometry();
    for ilen in [0, g.input_words - 1, g.input_words, g.input_words + 1] {
        for olen in [0, g.output_words - 1, g.output_words, g.output_words + 1] {
            for slen in [0, g.scratch_words - 1, g.scratch_words, g.scratch_words + 2] {
                if ilen == g.input_words && olen == g.output_words && slen >= g.scratch_words {
                    continue;
                }
                // Noncanonical input also proves length-error precedence.
                let input = vec![u64::MAX; ilen];
                let mut output = vec![u64::MAX - 1; olen];
                let mut scratch = vec![u64::MAX - 2; slen];
                assert_eq!(chain.apply_into(&input, &mut output, &mut scratch), Err(LayoutError::BufferLength));
                assert_eq!(output, vec![u64::MAX - 1; olen]);
                assert_eq!(scratch, vec![u64::MAX - 2; slen]);
            }
        }
    }
}

#[test]
fn every_limb_canonical_refusal_including_late_second_component_is_atomic() {
    let moduli = [9, 25, 49];
    let rings = rings(2, &moduli);
    let pairs = pairs(&rings);
    let chain = RoundedDropChain::new(&rings, &pairs, 2, UNLIMITED).unwrap();
    let g = chain.geometry();
    for component in 0..2 {
        for (i, &q) in moduli.iter().enumerate() {
            for bad in [q, u64::MAX] {
                let mut input = vec![0; g.input_words];
                input[component * 6 + i * 2 + 1] = bad;
                let mut output = vec![u64::MAX - 1; g.output_words];
                let mut scratch = vec![u64::MAX - 2; g.scratch_words + 3];
                assert_eq!(chain.apply_into(&input, &mut output, &mut scratch), Err(LayoutError::NonCanonical));
                assert_eq!(output, vec![u64::MAX - 1; g.output_words]);
                assert_eq!(scratch, vec![u64::MAX - 2; g.scratch_words + 3]);
            }
        }
    }
}

#[test]
fn reused_chain_quotes_per_call_and_preserves_scratch_tail() {
    let rings = rings(1, &[9, 25, 49, 11, 13]);
    let pairs = pairs(&rings);
    let chain = RoundedDropChain::new(&rings, &pairs, 2, UNLIMITED).unwrap();
    let work = chain.work();
    let mut output = vec![0; chain.geometry().output_words];
    let mut scratch = vec![123; chain.geometry().scratch_words + 4];
    for x in [-12345, 0, 4321] {
        let input = encode(&[BigInt::from(x), BigInt::from(-x)], 1, &[9, 25, 49, 11, 13]);
        chain.apply_into(&input, &mut output, &mut scratch).unwrap();
        assert_eq!(output, oracle(&input, 1, &[9, 25, 49, 11, 13], 2));
        assert_eq!(chain.work(), work);
        assert_eq!(&scratch[chain.geometry().scratch_words..], &[123; 4]);
    }
}

#[test]
fn checked_half_products_and_geometry_overflow_without_huge_slices() {
    assert_eq!(checked_half_product(usize::MAX, 2), Ok(usize::MAX));
    assert_eq!(checked_half_product(2, usize::MAX), Ok(usize::MAX));
    assert_eq!(checked_half_product(usize::MAX, usize::MAX - 1), Err(LayoutError::Overflow));
    for (m, degree, steps) in [
        (usize::MAX, 1, 1),                 // triangular count
        (2, usize::MAX, 1),                 // input words
        (4, usize::MAX / 12 + 1, 3),        // projections
        (2, usize::MAX / 7, 1),             // summed stores
        (2, usize::MAX / 16 + 1, 1),        // payload bytes
        (1_usize << (usize::BITS / 2 - 1), 1, 1), // descriptor bytes
    ] {
        assert_eq!(chain_quote(m, degree, steps), Err(LayoutError::Overflow));
    }
    assert_eq!(chain_quote(0, 1, 1), Err(LayoutError::InvalidGeometry));
    assert_eq!(chain_quote(2, 1, 0), Err(LayoutError::InvalidGeometry));
    assert_eq!(chain_quote(2, 1, 2), Err(LayoutError::InvalidGeometry));
}
