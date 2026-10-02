//! Domain-bound expansion of one worker's additive input share.
use aes::{
    cipher::{BlockEncrypt, KeyInit},
    Aes256,
};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

/// Expand a fresh seed, or subtract that expansion from a signed-i8 input.
/// The seed belongs only to the mask worker; the complement worker never gets it.
pub fn seeded_share(
    seed: &[u8; 32],
    context: &[u8; 32],
    count: usize,
    bits: u8,
    input: Option<&[u8]>,
) -> Result<Vec<u32>, String> {
    if !matches!(bits, 16 | 24 | 32)
        || !(1..=4_000_000).contains(&count)
        || input.is_some_and(|value| value.len() != count)
    {
        return Err("invalid seeded offset share dimensions or ring".into());
    }
    let mut hash = Sha256::new();
    hash.update(b"pllm/offset-input-share/aes256/v1\0");
    hash.update(seed);
    hash.update(context);
    hash.update((count as u64).to_le_bytes());
    hash.update([bits]);
    let key = Zeroizing::new(<[u8; 32]>::from(hash.finalize()));
    let cipher = Aes256::new_from_slice(key.as_ref()).map_err(|e| e.to_string())?;
    let mask = u32::MAX >> (32 - bits);
    let mut result = Vec::with_capacity(count);
    for counter in 0..count.div_ceil(4) {
        let mut block = aes::cipher::Block::<Aes256>::default();
        block[..8].copy_from_slice(&(counter as u64).to_le_bytes());
        cipher.encrypt_block(&mut block);
        for bytes in block.chunks_exact(4) {
            if result.len() == count {
                break;
            }
            let value = u32::from_le_bytes(bytes.try_into().unwrap()) & mask;
            result.push(match input {
                Some(input) => (input[result.len()] as i8 as i32 as u32).wrapping_sub(value) & mask,
                None => value,
            });
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_split_and_context_separation() {
        let input = [128, 255, 0, 1, 127, 3, 4];
        for bits in [16, 24, 32] {
            let a = seeded_share(&[3; 32], &[4; 32], input.len(), bits, Some(&input)).unwrap();
            let b = seeded_share(&[3; 32], &[4; 32], input.len(), bits, None).unwrap();
            for (i, (&a, &b)) in a.iter().zip(&b).enumerate() {
                assert_eq!(
                    a.wrapping_add(b) & (u32::MAX >> (32 - bits)),
                    (input[i] as i8 as i32 as u32) & (u32::MAX >> (32 - bits))
                );
            }
            assert_ne!(
                b,
                seeded_share(&[3; 32], &[5; 32], input.len(), bits, None).unwrap()
            );
        }
        assert!(seeded_share(&[0; 32], &[0; 32], 4_000_001, 32, None).is_err());
        assert!(seeded_share(&[0; 32], &[0; 32], 2, 32, Some(&input)).is_err());
    }
}
