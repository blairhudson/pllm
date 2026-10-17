use super::*;
use crate::gkm::tests::hex;
use sha2::{Digest, Sha256};

pub(crate) const LINEAR_KEY: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
pub(crate) const LOOKUP_KEY: [u8; 16] = [
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
];

// Independent public OpenSSL AES-128-ECB fixture, not upstream execution.
// Input block c = LE64(c)||LE64(c). Generation command is in the research note.
pub(crate) fn linear_fixture() -> Vec<u8> {
    hex(concat!(
        "c6a13b37878f5b826f4f8162a1c8d879",
        "57dfae4563c7d0579a1b58d216a7ab82",
        "6303bf72f17f57727d30d473fc66e549",
        "0eed7cd237b3697ff701e94010546b83",
        "1aebfc6cdf6b1ec18c629954ff68f153",
        "b6a8db33eb2a684496040109bb55f487",
    ))
}
pub(crate) fn lookup_fixture() -> Vec<u8> {
    hex("eda330f90eecd16c003e5fb09bcff358d0589492a3a1940fed5897c5fc9c1b8c")
}

#[test]
fn aes_nist_kat_source_counter_openssl_kat_and_zeroizing_cipher() {
    fn zeroizing<T: ZeroizeOnDrop>() {}
    zeroizing::<Aes128Enc>();
    zeroizing::<Stream>();
    // FIPS 197 Appendix C.1; independent of the stream/counter construction.
    let mut block = Block::<Aes128Enc>::clone_from_slice(&hex("00112233445566778899aabbccddeeff"));
    Aes128Enc::new((&LINEAR_KEY).into()).encrypt_block(&mut block);
    assert_eq!(block.as_slice(), hex("69c4e0d86a7b0430d8cdb78070b4c55a"));
    let mut out = [0; 96];
    Stream::new(&LINEAR_KEY, 96)
        .unwrap()
        .fill(&mut out)
        .unwrap();
    assert_eq!(out.as_slice(), linear_fixture());
    let mut out = [0; 32];
    Stream::new(&LOOKUP_KEY, 32)
        .unwrap()
        .fill(&mut out)
        .unwrap();
    assert_eq!(out.as_slice(), lookup_fixture());
}

#[test]
fn every_split_tail_and_source_prefetch_boundary_preserve_byte_continuity() {
    let expected = linear_fixture();
    for split in 0..=expected.len() {
        let mut stream = Stream::new(&LINEAR_KEY, 96).unwrap();
        let mut out = [0; 96];
        stream.fill(&mut out[..split]).unwrap();
        stream.fill(&mut []).unwrap();
        stream.fill(&mut out[split..]).unwrap();
        assert_eq!(out.as_slice(), expected);
        assert!(stream.complete());
    }
    let mut stream = Stream::new(&LINEAR_KEY, 8197).unwrap();
    let mut out = [0; 8197];
    let mut at = 0;
    for length in [1, 7, 4093, 8, 17, 4071] {
        stream.fill(&mut out[at..at + length]).unwrap();
        at += length;
    }
    assert_eq!(at, out.len());
    assert_eq!(
        Sha256::digest(out).as_slice(),
        hex("862d1180811883a8684e67feb5e4d31ba33bd549be2b484492ea37fe474e51ca")
    );
    assert!(stream.complete());
}

#[test]
fn byte_budget_and_last_counter_fail_closed_without_wrapping_or_partial_writes() {
    assert!(matches!(
        Stream::new(&LINEAR_KEY, Stream::DOMAIN_BYTES + 1),
        Err(Error::Limit)
    ));
    for split in [0, 8, 15] {
        let mut stream = Stream::new(&LINEAR_KEY, Stream::DOMAIN_BYTES).unwrap();
        // Test-only boundary setup, not a production cursor-import API.
        stream.position = Stream::DOMAIN_BYTES - 16;
        let mut last = [0; 16];
        stream.fill(&mut last[..split]).unwrap();
        stream.fill(&mut last[split..]).unwrap();
        assert_eq!(last.as_slice(), hex("3c441f32ce07822364d7a2990e50bb13"));
        assert!(stream.complete());
        let mut sentinel = [0xab; 17];
        assert_eq!(stream.fill(&mut sentinel), Err(Error::Limit));
        assert_eq!(sentinel, [0xab; 17]);
        assert_eq!(stream.fill(&mut []), Err(Error::Retired));
    }
    let mut stream = Stream::new(&LINEAR_KEY, 9).unwrap();
    stream.fill(&mut [0; 8]).unwrap();
    let mut untouched = [0x42; 2];
    assert_eq!(stream.fill(&mut untouched), Err(Error::Limit));
    assert_eq!(untouched, [0x42; 2]);
    assert_eq!(stream.position, 8);
    assert_eq!(stream.fill(&mut [0; 1]), Err(Error::Retired));
}
