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
        return Err(Error::Shape);
    }
    let range = 1_u64 << 32;
    let ceiling = range - range % u64::from(upper);
    // Bound rejection work without changing the uniform accepted domain.
    for _ in 0..32 {
        let x = u64::from(rng.word()?);
        if x < ceiling {
            return Ok((x % u64::from(upper)) as u32);
        }
    }
    Err(Error::Randomness)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CountedWords<'a> {
        words: &'a [Result<u32>],
        draws: usize,
    }

    impl RandomWords for CountedWords<'_> {
        fn word(&mut self) -> Result<u32> {
            let word = self.words.get(self.draws).copied();
            self.draws += 1;
            word.unwrap_or(Err(Error::Randomness))
        }
    }

    #[test]
    fn bounded_draw_accepts_draw_32_without_consuming_draw_33() {
        let mut words = [Ok(u32::MAX); 33];
        words[31] = Ok(9);
        words[32] = Err(Error::Field);
        let mut rng = CountedWords {
            words: &words,
            draws: 0,
        };
        assert_eq!(below(&mut rng, 5), Ok(4));
        assert_eq!(rng.draws, 32);
    }

    #[test]
    fn bounded_draw_rejection_cap_never_consumes_acceptable_draw_33() {
        let mut words = [Ok(u32::MAX); 33];
        words[32] = Ok(4);
        let mut rng = CountedWords {
            words: &words,
            draws: 0,
        };
        assert_eq!(below(&mut rng, 5), Err(Error::Randomness));
        assert_eq!(rng.draws, 32);
    }

    #[test]
    fn bounded_draw_propagates_errors_immediately_before_rejection_cap() {
        for rejected in 0..32 {
            let mut words = [Ok(u32::MAX); 33];
            words[rejected] = Err(Error::Field);
            words[rejected + 1] = Ok(4);
            let mut rng = CountedWords {
                words: &words,
                draws: 0,
            };
            assert_eq!(below(&mut rng, 5), Err(Error::Field));
            assert_eq!(rng.draws, rejected + 1);
        }
    }

    #[test]
    fn bounded_draw_zero_uses_no_entropy_and_one_uses_one_fresh_word() {
        let mut rng = CountedWords {
            words: &[Err(Error::Field), Ok(u32::MAX), Err(Error::Allocation)],
            draws: 0,
        };
        assert_eq!(below(&mut rng, 0), Err(Error::Shape));
        assert_eq!(rng.draws, 0);
        assert_eq!(below(&mut rng, 1), Err(Error::Field));
        assert_eq!(rng.draws, 1);
        assert_eq!(below(&mut rng, 1), Ok(0));
        assert_eq!(rng.draws, 2);
        assert_eq!(below(&mut rng, 1), Err(Error::Allocation));
        assert_eq!(rng.draws, 3);
    }
}
