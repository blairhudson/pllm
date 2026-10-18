use zeroize::Zeroizing;

use super::{Error, Result, FIELD};

pub(super) fn canonical(x: &[u32], length: usize) -> Result<()> {
    if x.len() != length {
        return Err(Error::Shape);
    }
    if x.iter().any(|&v| v >= FIELD) {
        return Err(Error::Field);
    }
    Ok(())
}

#[inline]
pub(super) fn add(a: u32, b: u32) -> u32 {
    ((u64::from(a) + u64::from(b)) % u64::from(FIELD)) as u32
}

#[inline]
pub(super) fn sub(a: u32, b: u32) -> u32 {
    ((u64::from(a) + u64::from(FIELD) - u64::from(b)) % u64::from(FIELD)) as u32
}

#[inline]
pub(super) fn mul(a: u32, b: u32) -> u32 {
    (u64::from(a) * u64::from(b) % u64::from(FIELD)) as u32
}

pub(super) fn dot(a: &[u32], b: &[u32]) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0, |acc, (&x, &y)| add(acc, mul(x, y)))
}

pub(super) trait RandomWords {
    fn word(&mut self) -> Result<u32>;
}

pub(super) struct OsRandom;

impl RandomWords for OsRandom {
    fn word(&mut self) -> Result<u32> {
        word_from_fill(|bytes| getrandom::fill(bytes).map_err(|_| Error::Randomness))
    }
}

pub(super) fn word_from_fill(fill: impl FnOnce(&mut [u8]) -> Result<()>) -> Result<u32> {
    // The guard also erases bytes partially written by an entropy provider that
    // subsequently fails. Never return or reuse that partial word.
    let mut bytes = Zeroizing::new([0; 4]);
    fill(bytes.as_mut())?;
    Ok(u32::from_le_bytes(*bytes))
}

pub(super) fn below(rng: &mut impl RandomWords, upper: u32) -> Result<u32> {
    if upper == 0 {
        return Err(Error::Randomness);
    }
    let range = 1_u64 << 32;
    let ceiling = range - range % u64::from(upper);
    loop {
        let x = u64::from(rng.word()?);
        if x < ceiling {
            return Ok((x % u64::from(upper)) as u32);
        }
    }
}
