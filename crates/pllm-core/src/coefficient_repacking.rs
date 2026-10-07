//! Bounded public-ciphertext RNS scaling for coefficient trace/repacking.
//! Encryption, automorphisms and security parameter validation belong to SEAL.

/// Divide a size-two, coefficient-form ciphertext by a power of two modulo each
/// odd RNS prime. This is NOT division in the BFV plaintext ring. A complete trace
/// cancels the divisor; decrypting this intermediate does not implement division.
pub fn divide_power_two(
    words: &mut [u64],
    degree: usize,
    moduli: &[u64],
    divisor: usize,
) -> Result<(), String> {
    if degree != 4096
        || moduli.is_empty()
        || moduli.len() > 2
        || divisor == 0
        || !divisor.is_power_of_two()
        || divisor > degree
        || words.len() != 2 * degree * moduli.len()
        || moduli
            .iter()
            .any(|&q| !(3..(1 << 61)).contains(&q) || q & 1 == 0)
    {
        return Err("invalid bounded coefficient trace geometry".into());
    }
    for polynomial in words.chunks_exact(degree * moduli.len()) {
        for (limb, &q) in polynomial.chunks_exact(degree).zip(moduli) {
            if limb.iter().any(|&v| v >= q) {
                return Err("noncanonical public ciphertext coefficient".into());
            }
        }
    }
    for polynomial in words.chunks_exact_mut(degree * moduli.len()) {
        for (limb, &q) in polynomial.chunks_exact_mut(degree).zip(moduli) {
            let mut inverse = 1_u128;
            for _ in 0..divisor.trailing_zeros() {
                inverse = inverse * u128::from(q.div_ceil(2)) % u128::from(q);
            }
            for v in limb {
                *v = (u128::from(*v) * inverse % u128::from(q)) as u64;
            }
        }
    }
    Ok(())
}

/// Sample-extract the selected coefficient as (b, a_0, ..., a_{N-1}) in
/// each RNS limb, with decryption b + <a,s>. Inputs are public ciphertexts.
pub fn extract_lwe(words: &[u64], moduli: &[u64], coefficient: usize) -> Result<Vec<u64>, String> {
    if coefficient >= 4096
        || moduli.is_empty()
        || moduli.len() > 2
        || words.len() != 8192 * moduli.len()
    {
        return Err("LWE extraction coefficient exceeds degree".into());
    }
    // Validation is shared with modular scaling and completes before extraction.
    let mut checked = words.to_vec();
    divide_power_two(&mut checked, 4096, moduli, 1)?;
    let mut out = Vec::with_capacity(4097 * moduli.len());
    for (m, &q) in moduli.iter().enumerate() {
        out.push(words[m * 4096 + coefficient]);
        let a = &words[(moduli.len() + m) * 4096..(moduli.len() + m + 1) * 4096];
        for i in 0..4096 {
            let v = a[(coefficient + 4096 - i) % 4096];
            out.push(if i > coefficient && v != 0 { q - v } else { v });
        }
    }
    Ok(out)
}

/// Embed an extracted LWE in a size-two RLWE container. Only its constant
/// coefficient has the LWE's plaintext meaning; PackLWEs must trace away the rest.
pub fn embed_lwe(words: &[u64], moduli: &[u64]) -> Result<Vec<u64>, String> {
    if moduli.is_empty()
        || moduli.len() > 2
        || words.len() != moduli.len() * 4097
        || moduli
            .iter()
            .any(|&q| !(3..(1 << 61)).contains(&q) || q & 1 == 0)
        || words
            .chunks_exact(4097)
            .zip(moduli)
            .any(|(limb, &q)| limb.iter().any(|&v| v >= q))
    {
        return Err("noncanonical bounded extracted LWE".into());
    }
    let mut out = vec![0; 2 * 4096 * moduli.len()];
    for (m, (limb, &q)) in words.chunks_exact(4097).zip(moduli).enumerate() {
        out[m * 4096] = limb[0];
        out[(moduli.len() + m) * 4096] = limb[1];
        for j in 1..4096 {
            let v = limb[4097 - j];
            out[(moduli.len() + m) * 4096 + j] = if v == 0 { 0 } else { q - v };
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rns_scaling_is_exact_and_invalid_input_is_atomic() {
        for divisor in [1, 2, 64, 4096] {
            let moduli = [65537, 1125899906826241];
            let mut words = (0..16384).map(|i| i as u64).collect::<Vec<_>>();
            let original = words.clone();
            divide_power_two(&mut words, 4096, &moduli, divisor).unwrap();
            for (i, (&a, &b)) in words.iter().zip(&original).enumerate() {
                assert_eq!(
                    (u128::from(a) * divisor as u128 % u128::from(moduli[(i / 4096) % 2])) as u64,
                    b
                );
            }
            words[16383] = u64::MAX;
            let old = words.clone();
            assert!(divide_power_two(&mut words, 4096, &moduli, divisor).is_err());
            assert_eq!(words, old);
        }
    }

    #[test]
    fn lwe_extraction_and_embedding_preserve_signed_negacyclic_inner_products() {
        let moduli = [65537, 1125899906826241];
        let words = (0..16384)
            .map(|i| (i * 177 + 3) as u64 % moduli[(i / 4096) % 2])
            .collect::<Vec<_>>();
        for k in [0, 1, 127, 4095] {
            let extracted = extract_lwe(&words, &moduli, k).unwrap();
            let embedded = embed_lwe(&extracted, &moduli).unwrap();
            for (m, &q) in moduli.iter().enumerate() {
                let mut oracle = i128::from(words[m * 4096 + k]);
                let mut lwe = i128::from(extracted[m * 4097]);
                let mut ring = i128::from(embedded[m * 4096]);
                for i in 0..4096 {
                    let secret = (i as i128 % 3) - 1;
                    let a = i128::from(words[(2 + m) * 4096 + (k + 4096 - i) % 4096]);
                    oracle += if i <= k { a * secret } else { -a * secret };
                    lwe += i128::from(extracted[m * 4097 + i + 1]) * secret;
                    let b = i128::from(embedded[(2 + m) * 4096 + (4096 - i) % 4096]);
                    ring += if i == 0 { b * secret } else { -b * secret };
                }
                assert_eq!(oracle.rem_euclid(q.into()), lwe.rem_euclid(q.into()));
                assert_eq!(oracle.rem_euclid(q.into()), ring.rem_euclid(q.into()));
                assert!(embedded[m * 4096 + 1..(m + 1) * 4096]
                    .iter()
                    .all(|&v| v == 0));
            }
        }
        assert!(extract_lwe(&words, &moduli, 4096).is_err());
        assert!(embed_lwe(&[0; 4097], &[2]).is_err());
        assert!(embed_lwe(&[17; 4097], &[17]).is_err());
    }
}
