use super::*;
use crate::compact_dpf::{BitCount, Key as LegacyKey, RandomField};

fn bytes<const N: usize>(hex: &str) -> [u8; N] {
    assert_eq!(hex.len(), 2 * N);
    std::array::from_fn(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap())
}

fn fixture(profile: Profile, point: u64, phase: u8, salt: u128) -> [GpuKey; 2] {
    let roots = [
        0x123456789abcdef0fedcba9876543210u128 ^ (salt << 1),
        ((0x7a65_4321_0fed_cba9_8765_4321_0abc_def0_u128 ^ salt.rotate_left(19)) & !1)
            | u128::from(phase),
    ];
    let mut calls = 0;
    let keys = GpuKey::issue_with(profile, point, |out| {
        calls += 1;
        assert_eq!(out.len(), 32);
        out[..16].copy_from_slice(&roots[0].to_le_bytes());
        out[16..].copy_from_slice(&roots[1].to_le_bytes());
        Ok(())
    })
    .unwrap();
    assert_eq!(calls, 1);
    assert_eq!((*keys[0].root & 1) as u8, 1 ^ phase);
    assert_eq!((*keys[1].root & 1) as u8, phase);
    keys
}

// Independent scalar AES oracle, not the production aes crate or upstream code.
// FIPS 197 GF(256) inverse/affine S-box, big-endian word schedule and rounds.
fn mul(mut a: u8, mut b: u8) -> u8 {
    let mut out = 0;
    for _ in 0..8 {
        out ^= a & 0u8.wrapping_sub(b & 1);
        a = (a << 1) ^ (0x1b & 0u8.wrapping_sub(a >> 7));
        b >>= 1;
    }
    out
}

fn sbox(value: u8) -> u8 {
    let (mut a, mut power, mut inv) = (value, 254u8, 1u8);
    while power != 0 {
        if power & 1 != 0 {
            inv = mul(inv, a);
        }
        a = mul(a, a);
        power >>= 1;
    }
    inv ^ inv.rotate_left(1) ^ inv.rotate_left(2) ^ inv.rotate_left(3) ^ inv.rotate_left(4) ^ 0x63
}

fn subword(word: u32) -> u32 {
    u32::from_be_bytes(word.to_be_bytes().map(sbox))
}

fn mix(word: u32) -> u32 {
    let [a, b, c, d] = word.to_be_bytes();
    u32::from_be_bytes([
        mul(a, 2) ^ mul(b, 3) ^ c ^ d,
        a ^ mul(b, 2) ^ mul(c, 3) ^ d,
        a ^ b ^ mul(c, 2) ^ mul(d, 3),
        mul(a, 3) ^ b ^ c ^ mul(d, 2),
    ])
}

fn aes_words(key: [u8; 16], mut state: [u32; 4]) -> [u32; 4] {
    let mut schedule = [0u32; 44];
    // Source reads little-endian u32 words, then reverseBytes swaps each word.
    for i in 0..4 {
        schedule[i] = u32::from_le_bytes(key[4 * i..4 * i + 4].try_into().unwrap()).swap_bytes();
    }
    let mut rcon = 1u8;
    for i in 4..44 {
        let mut t = schedule[i - 1];
        if i % 4 == 0 {
            t = subword(t.rotate_left(8)) ^ (u32::from(rcon) << 24);
            rcon = mul(rcon, 2);
        }
        schedule[i] = schedule[i - 4] ^ t;
    }
    for i in 0..4 {
        state[i] ^= schedule[i];
    }
    for round in 1..=10 {
        let shifted: [u32; 4] = std::array::from_fn(|c| {
            u32::from_be_bytes(std::array::from_fn(|r| {
                sbox(state[(c + r) % 4].to_be_bytes()[r])
            }))
        });
        for c in 0..4 {
            state[c] = if round == 10 {
                shifted[c]
            } else {
                mix(shifted[c])
            } ^ schedule[4 * round + c];
        }
    }
    state
}

fn source_word_oracle(packed: u128, branch: usize) -> u128 {
    // Source initializes zero words, then ((uint8_t*)ct)[3] = 2*branch.
    // On little-endian GPU that is numeric first word (2*branch)<<24.
    let words = aes_words(
        (packed & !1).to_le_bytes(),
        [(2 * branch as u32) << 24, 0, 0, 0],
    );
    let mut out = [0u8; 16];
    for i in 0..4 {
        // Source's final reverseBytes followed by little-endian block view.
        out[4 * i..4 * i + 4].copy_from_slice(&words[i].swap_bytes().to_le_bytes());
    }
    u128::from_le_bytes(out)
}

/// Independent per-terminal scalar source expansion. Only reads issued fields;
/// never calls production child/children/descend/point/less/all/visit.
pub(crate) fn source_bits(key: &GpuKey) -> Vec<u8> {
    let width = key.profile.bits;
    let ordinary = usize::from(width - 8);
    let mut output = Vec::new();
    for block in 0..1usize << (width - 7) {
        let mut state = *key.root;
        for level in 0..ordinary {
            let branch = (block >> (ordinary - level)) & 1;
            let seed = key.words[level] & !1;
            let control = if branch == 0 {
                key.words[level] & 1
            } else {
                u128::from(key.right[level])
            };
            let correction = if state & 1 == 1 { seed | control } else { 0 };
            state = source_word_oracle(state, branch) ^ correction;
        }
        let branch = block & 1;
        let word = source_word_oracle(state, branch)
            ^ if state & 1 == 1 { key.terminal[branch] } else { 0 };
        for offset in 0..128 {
            output.push(((word >> offset) & 1) as u8);
        }
    }
    output
}

#[test]
fn ascending_visit_independent_scalar_source_bits_exact_work_and_once_only() {
    for (width, visits, blocks) in [(8, 256, 2), (10, 1024, 14)] {
        for phase in [0, 1] {
            for point in [0, 127, 128, (visits - 1) as u64] {
                let keys = fixture(Profile::new(width).unwrap(), point, phase, 79);
                assert_eq!(*keys[1].root & 1, u128::from(phase));
                assert_eq!(*keys[0].root & 1, u128::from(phase ^ 1));
                let expected = [source_bits(&keys[0]), source_bits(&keys[1])];
                for (role, key) in keys.iter().enumerate() {
                    let mut seen = vec![0usize; visits];
                    let mut next = 0;
                    let before = measured_aes_blocks();
                    key.preparation_visit(|index, bit| {
                        assert_eq!(index, next);
                        next += 1;
                        seen[index] += 1;
                        assert_eq!(*bit.value(), expected[role][index]);
                    }).unwrap();
                    assert_eq!(measured_aes_blocks() - before, blocks);
                    assert_eq!(next, visits);
                    assert!(seen.iter().all(|&n| n == 1));
                    // API continuity, not the independent expected-bit proof.
                    assert_eq!(&*key.all(visits).unwrap(), &expected[role]);
                }
                for index in 0..visits {
                    assert_eq!(expected[0][index] ^ expected[1][index], u8::from(index as u64 == point));
                }
            }
        }
    }
}

#[test]
fn visit_wide_rejects_before_callback_or_aes_and_callback_unwind_erases_key_on_drop() {
    for width in [9, 11, 13, 63] {
        let [key, _] = fixture(Profile::new(width).unwrap(), 127, 0, 3);
        let before = measured_aes_blocks();
        assert_eq!(key.preparation_visit(|_, _| panic!("unsupported callback")), Err(Error::Resource));
        assert_eq!(measured_aes_blocks(), before);
    }
    for at in [0, 127, 128, 1023] {
        let [key, other] = fixture(Profile::new(10).unwrap(), 127, 1, 3);
        let before = ERASED_KEYS.with(|n| n.get());
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            key.preparation_visit(|index, _| if index == at { panic!("visit unwind") }).unwrap();
        }));
        assert!(outcome.is_err());
        // Borrowing traversal does not consume the key or give it to callback.
        assert_eq!(ERASED_KEYS.with(|n| n.get()), before);
        drop((key, other));
        assert_eq!(ERASED_KEYS.with(|n| n.get()), before + 2);
    }
}

#[test]
fn preparation_actual_borrowed_bit_owner_erases_on_return_and_callback_unwind() {
    for width in [8, 10] {
        for party in 0..2 {
            let keys = fixture(Profile::new(width).unwrap(), 127, party as u8, 3);
            let key = &keys[party];
            for at in [0, 127, 128, (1usize << width) - 1] {
                for unwind in [false, true] {
                    let mut observed = None;
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        key.preparation_visit(|index, bit| {
                            if index == at {
                                // Observes Drop of THIS actual walk's bit owner,
                                // not a global counter or fabricated static flag.
                                observed = Some(std::rc::Rc::clone(&bit.retired));
                                assert_eq!(bit.retired.get(), None);
                                if unwind { panic!("protected owner unwind"); }
                            }
                        }).unwrap();
                    }));
                    assert_eq!(result.is_err(), unwind);
                    assert_eq!(observed.unwrap().get(), Some(0));
                }
            }
        }
    }
    let [mut invalid, _] = fixture(Profile::new(8).unwrap(), 0, 0, 3);
    invalid.party = 2;
    let before = measured_aes_blocks();
    assert_eq!(invalid.preparation_visit(|_, _| panic!("invalid actual role")), Err(Error::Domain));
    assert_eq!(measured_aes_blocks(), before);
}

#[test]
fn preparation_real_key_mutations_each_width_party_have_non_target_scalar_witness() {
    // Finite deterministic fixture/counterexample search in TEST ONLY. Neither
    // production traversal nor preparation chooses a secret-dependent algorithm.
    for width in [8, 10] {
        for party in 0..2 {
            let kinds: &[&str] = if width == 8 {
                &["root-seed", "root-control", "terminal-left", "terminal-right"]
            } else {
                &["root-seed", "root-control", "terminal-left", "terminal-right",
                  "correction-seed", "correction-left-control", "correction-right-control"]
            };
            for &kind in kinds {
                let mut witness = None;
                for salt in 0..64 {
                    // This party's root control is one: width8 terminal changes
                    // must not be tested through an inactive root-control-zero key.
                    let mut keys = fixture(Profile::new(width).unwrap(), 127, party as u8, salt);
                    let original = source_bits(&keys[party]);
                    let sibling = source_bits(&keys[1 - party]);
                    let key = &mut keys[party];
                    match kind {
                        "root-seed" => *key.root ^= 2,
                        "root-control" => *key.root ^= 1,
                        "terminal-left" => key.terminal[0] ^= 1,
                        "terminal-right" => key.terminal[1] ^= 1,
                        "correction-seed" => key.words[0] ^= 2,
                        "correction-left-control" => key.words[0] ^= 1,
                        "correction-right-control" => key.right[0] ^= 1,
                        _ => unreachable!(),
                    }
                    let changed = source_bits(key);
                    let mut actual = Vec::new();
                    let before = measured_aes_blocks();
                    key.preparation_visit(|index, bit| {
                        assert_eq!(index, actual.len());
                        actual.push(*bit.value());
                    }).unwrap();
                    assert_eq!(actual, changed, "AES-correct mutated vector {width}/{party}/{kind}/{salt}");
                    assert_eq!(measured_aes_blocks() - before, if width == 8 { 2 } else { 14 });
                    if let Some(index) = (0..changed.len()).find(|&i| i != 127 && changed[i] != original[i]) {
                        assert_eq!(original[index] ^ sibling[index], 0);
                        assert_eq!(actual[index] ^ sibling[index], 1,
                            "explicit non-target {index}: width={width}, party={party}, kind={kind}, salt={salt}");
                        witness = Some((salt, index));
                        break;
                    }
                }
                assert!(witness.is_some(), "missing finite non-target witness: {width}/{party}/{kind}");
            }
        }
    }
}

#[test]
fn independent_word_mapping_nist_and_openssl_half_paired_vectors() {
    // FIPS 197 Appendix B/C AES-128 known answer; validates independent oracle.
    let key = bytes::<16>("000102030405060708090a0b0c0d0e0f");
    let pt = bytes::<16>("00112233445566778899aabbccddeeff");
    let words =
        std::array::from_fn(|i| u32::from_be_bytes(pt[4 * i..4 * i + 4].try_into().unwrap()));
    let ct = aes_words(key, words);
    let expected = bytes::<16>("69c4e0d86a7b0430d8cdb78070b4c55a");
    for i in 0..4 {
        assert_eq!(ct[i].to_be_bytes(), expected[4 * i..4 * i + 4]);
    }
    // Frozen standard OpenSSL AES-128-ECB, no padding, one block; no upstream
    // execution. Plaintexts are 00..00 and 02 followed by fifteen zero bytes.
    for (key, outputs) in [
        (
            "00000000000000000000000000000000",
            [
                "66e94bd4ef8a2c3b884cfa59ca342b2e",
                "bcf176a7eaad8085ebacea362462a281",
            ],
        ),
        (
            "000102030405060708090a0b0c0d0e0f",
            [
                "c6a13b37878f5b826f4f8162a1c8d879",
                "fb8ae31ba5db9cad97364d8722d47326",
            ],
        ),
        (
            "f0debc9a785634121032547698badcfe",
            [
                "bb546766054ba5420a3127ad56b9e91b",
                "67b86ceae1689c1fa4379c4da927d4dd",
            ],
        ),
    ] {
        let seed = u128::from_le_bytes(bytes::<16>(key));
        for phase in [0, 1] {
            let packed = seed | phase;
            let paired = children(packed);
            for branch in 0..2 {
                let expected = u128::from_le_bytes(bytes::<16>(outputs[branch]));
                assert_eq!(*child(packed, branch), expected);
                assert_eq!(paired[branch], expected);
                assert_eq!(source_word_oracle(packed, branch), expected);
            }
        }
    }
    for bit in 1..128 {
        let packed = 1u128 << bit;
        let paired = children(packed);
        for branch in 0..2 {
            assert_eq!(*child(packed, branch), paired[branch]);
        }
    }
}

#[test]
fn exhaustive_width8_width9_points_strict_less_both_root_phases_and_full_scan() {
    for bits in [8, 9] {
        let profile = Profile::new(bits).unwrap();
        for phase in [0, 1] {
            for point in 0..1u64 << bits {
                let keys = fixture(profile, point, phase, u128::from(point));
                let all = [
                    keys[0].all(1 << bits).unwrap(),
                    keys[1].all(1 << bits).unwrap(),
                ];
                // Independent truth table, then independently accumulated suffix
                // parity. No production traversal/control equation in this oracle.
                let oracle: Vec<u8> = (0..1u64 << bits).map(|x| u8::from(x == point)).collect();
                let mut suffix = 0;
                for x in (0..1u64 << bits).rev() {
                    let index = x as usize;
                    assert_eq!(all[0][index] ^ all[1][index], oracle[index]);
                    assert_eq!(
                        keys[0].point(x).unwrap() ^ keys[1].point(x).unwrap(),
                        oracle[index]
                    );
                    assert_eq!(keys[0].less(x).unwrap() ^ keys[1].less(x).unwrap(), suffix);
                    assert_eq!(suffix, u8::from(x < point));
                    for party in 0..2 {
                        assert_eq!(all[party][index], keys[party].point(x).unwrap());
                    }
                    suffix ^= oracle[index];
                }
                assert_eq!(suffix, 1);
            }
        }
    }
}

#[test]
fn sparse_wide_endpoints_widths_and_unsigned_comparison_contract() {
    for bits in [8, 9, 10, 11, 25, 40, 50, 63] {
        let profile = Profile::new(bits).unwrap();
        let max = (1u64 << bits) - 1;
        for phase in [0, 1] {
            for point in [
                0,
                1,
                126,
                127,
                128,
                129,
                (1 << (bits - 1)) - 1,
                1 << (bits - 1),
                max,
            ] {
                let keys = fixture(profile, point, phase, u128::from(point));
                for x in [
                    0,
                    1,
                    126,
                    127,
                    128,
                    129,
                    point.saturating_sub(1),
                    point,
                    (point + 1) & max,
                    max,
                ] {
                    assert_eq!(
                        keys[0].point(x).unwrap() ^ keys[1].point(x).unwrap(),
                        u8::from(x == point)
                    );
                    assert_eq!(
                        keys[0].less(x).unwrap() ^ keys[1].less(x).unwrap(),
                        u8::from(x < point)
                    );
                }
                assert_eq!(keys[0].point(max + 1), Err(Error::Domain));
                assert_eq!(keys[1].less(max + 1), Err(Error::Domain));
            }
        }
    }
}

#[test]
fn exact_geometry_work_formulas_and_full_expansion_preflight() {
    for bits in 8..=63 {
        let p = Profile::new(bits).unwrap();
        assert_eq!(p.bits(), bits);
        assert_eq!(Profile::DOMAIN, "pllm.compact-dpf.source-gpu127.v1");
        let d = usize::from(bits - 7);
        let cost = p.resources();
        assert_eq!(cost.logical_key_bits, 129 * d + 255);
        assert_eq!(cost.codec_bits, 8 * HEADER_BYTES + cost.logical_key_bits);
        assert_eq!(cost.codec_bytes, cost.codec_bits.div_ceil(8));
        assert_eq!(cost.key_owner_bytes, std::mem::size_of::<GpuKey>());
        assert_eq!(cost.pair_owner_bytes, 2 * cost.key_owner_bytes);
        assert!(cost.key_owner_bytes <= 2048 && cost.codec_bytes <= 953);
        AES_BLOCKS.with(|c| c.set(0));
        let keys = fixture(p, 127, 0, 0);
        assert_eq!(AES_BLOCKS.with(|c| c.get()), cost.paired_issue_aes_blocks);
        assert_eq!(cost.paired_issue_aes_blocks, 4 * d);
        for key in &keys {
            AES_BLOCKS.with(|c| c.set(0));
            key.point(127).unwrap();
            assert_eq!(AES_BLOCKS.with(|c| c.get()), d);
            AES_BLOCKS.with(|c| c.set(0));
            key.less(127).unwrap();
            assert_eq!(AES_BLOCKS.with(|c| c.get()), cost.point_or_less_aes_blocks);
            AES_BLOCKS.with(|c| c.set(0));
            if bits <= MAX_FULL_BITS {
                let count = 1usize << bits;
                let all = key.all(count).unwrap();
                assert_eq!(all.len(), count);
                assert_eq!(cost.full_expansion_bytes, Some(count));
                assert_eq!(AES_BLOCKS.with(|c| c.get()), 2 * ((1usize << d) - 1));
                assert_eq!(
                    cost.full_expansion_aes_blocks,
                    Some(2 * ((1usize << d) - 1))
                );
            } else {
                assert!(matches!(
                    key.all_with_reservation(usize::MAX, |_| panic!("wide allocation")),
                    Err(Error::Resource)
                ));
                assert_eq!(AES_BLOCKS.with(|c| c.get()), 0);
                assert_eq!(cost.full_expansion_bytes, None);
                assert_eq!(cost.full_expansion_aes_blocks, None);
            }
        }
    }
    for bits in (0..=7).chain(64..=255) {
        assert_eq!(Profile::new(bits), Err(Error::Domain));
    }
}

#[test]
fn frozen_source_gpu_codec_vector_and_canonical_roundtrip() {
    let p = Profile::new(8).unwrap();
    let keys = GpuKey::issue_with(p, 255, |out| {
        out.fill(0);
        for (i, byte) in out[16..].iter_mut().enumerate() {
            *byte = i as u8;
        }
        Ok(())
    })
    .unwrap();
    let expected = [
        bytes::<66>("504c4c4d2d4450462d47505531323701080001000000000000000000000000000000a04870e3680577b9e7037b3b6bfcf357477b95bc4f761c287c9aa7b106b6d127"),
        bytes::<66>("504c4c4d2d4450462d475055313237010801000102030405060708090a0b0c0d0e0fa04870e3680577b9e7037b3b6bfcf357477b95bc4f761c287c9aa7b106b6d127"),
    ];
    for (party, key) in keys.into_iter().enumerate() {
        let material = key.into_material(66).unwrap();
        assert_eq!(&material[..], &expected[party]);
        let key = GpuKey::decode(p, party as u8, &material).unwrap();
        assert_eq!(&*key.into_material(66).unwrap(), &*material);
    }
    for bits in [8, 9, 10, 50, 63] {
        let p = Profile::new(bits).unwrap();
        let cost = p.resources();
        for phase in [0, 1] {
            for (party, key) in fixture(p, 127, phase, 53).into_iter().enumerate() {
                let before: Vec<_> = [0, 126, 127, 128, (1u64 << bits) - 1]
                    .into_iter()
                    .map(|x| (x, key.point(x).unwrap(), key.less(x).unwrap()))
                    .collect();
                let mut count = BitCount::default();
                key.write(&mut count);
                assert_eq!(count.0, cost.codec_bits);
                let material = key.into_material(cost.codec_bytes).unwrap();
                assert_eq!(material.len(), cost.codec_bytes);
                let restored = GpuKey::decode(p, party as u8, &material).unwrap();
                for (x, point, less) in before {
                    assert_eq!(restored.point(x).unwrap(), point);
                    assert_eq!(restored.less(x).unwrap(), less);
                }
                assert_eq!(
                    &*restored.into_material(cost.codec_bytes).unwrap(),
                    &*material
                );
            }
        }
    }
}

#[test]
fn malformed_codec_profile_version_role_width_padding_truncation_and_substitution() {
    for bits in [8, 9, 10, 50, 63] {
        let p = Profile::new(bits).unwrap();
        let cost = p.resources();
        let [a, _b] = fixture(p, 127, 1, 0);
        let material = a.into_material(cost.codec_bytes).unwrap();
        for len in 0..material.len() {
            assert!(matches!(
                GpuKey::decode(p, 0, &material[..len]),
                Err(Error::Encoding)
            ));
        }
        let mut bad = Zeroizing::new(material.to_vec());
        bad.push(0);
        assert!(matches!(GpuKey::decode(p, 0, &bad), Err(Error::Encoding)));
        for index in 0..HEADER_BYTES {
            let mut bad = Zeroizing::new(material.to_vec());
            bad[index] ^= 0x80;
            assert!(matches!(GpuKey::decode(p, 0, &bad), Err(Error::Encoding)));
        }
        assert!(matches!(
            GpuKey::decode(p, 1, &material),
            Err(Error::Encoding)
        ));
        assert!(matches!(
            GpuKey::decode(p, 2, &material),
            Err(Error::Domain)
        ));
        let other = Profile::new(if bits == 63 { 62 } else { bits + 1 }).unwrap();
        assert!(matches!(
            GpuKey::decode(other, 0, &material),
            Err(Error::Encoding)
        ));
        if cost.codec_bits % 8 != 0 {
            for bit in cost.codec_bits % 8..8 {
                let mut bad = Zeroizing::new(material.to_vec());
                *bad.last_mut().unwrap() |= 1 << bit;
                assert!(matches!(GpuKey::decode(p, 0, &bad), Err(Error::Encoding)));
            }
        }
        // Arbitrary same-profile payloads remain parseable. Authentication and
        // session-bound substitution rejection are explicitly outer-owner duties.
        let mut changed = Zeroizing::new(material.to_vec());
        changed[HEADER_BYTES] ^= 2;
        assert!(GpuKey::decode(p, 0, &changed).is_ok());
        assert!(matches!(
            GpuKey::decode(p, 0, &[0; 49]),
            Err(Error::Encoding)
        ));
    }
}

#[test]
fn preflight_entropy_failure_allocation_hooks_and_consumed_owner_erasure() {
    let p = Profile::new(9).unwrap();
    assert!(matches!(
        GpuKey::issue_with(p, 512, |_| panic!("entropy before preflight")),
        Err(Error::Domain)
    ));
    assert!(matches!(
        GpuKey::issue_with(p, 127, |out| {
            out.fill(0xa5);
            Err(Error::Randomness)
        }),
        Err(Error::Randomness)
    ));
    let cost = p.resources();
    let [a, b] = fixture(p, 127, 0, 0);
    AES_BLOCKS.with(|c| c.set(0));
    assert!(matches!(
        a.all_with_reservation(511, |_| panic!("unadmitted output")),
        Err(Error::Resource)
    ));
    assert!(matches!(
        a.all_with_reservation(512, |bytes| {
            assert_eq!(bytes, 512);
            Err(Error::Allocation)
        }),
        Err(Error::Allocation)
    ));
    assert_eq!(AES_BLOCKS.with(|c| c.get()), 0);
    ERASED_KEYS.with(|c| c.set(0));
    assert!(matches!(
        a.into_material_with_reservation(cost.codec_bytes - 1, |_| panic!("unadmitted codec")),
        Err(Error::Resource)
    ));
    assert_eq!(ERASED_KEYS.with(|c| c.get()), 1);
    assert!(matches!(
        b.into_material_with_reservation(cost.codec_bytes, |bytes| {
            assert_eq!(bytes, cost.codec_bytes);
            Err(Error::Allocation)
        }),
        Err(Error::Allocation)
    ));
    assert_eq!(ERASED_KEYS.with(|c| c.get()), 2);
    let [a, b] = fixture(p, 127, 1, 2);
    let out = a
        .into_material_with_reservation(cost.codec_bytes, |bytes| {
            assert_eq!(bytes, cost.codec_bytes);
            Ok(())
        })
        .unwrap();
    assert_eq!(ERASED_KEYS.with(|c| c.get()), 3);
    assert_eq!(out.len(), cost.codec_bytes);
    drop(b);
    assert_eq!(ERASED_KEYS.with(|c| c.get()), 4);
}

#[test]
fn full_scan_allocator_checks_geometry_capacity_and_failure_before_aes() {
    let profile = Profile::new(10).unwrap();
    let [left, right] = fixture(profile, 1023, 1, 3);
    AES_BLOCKS.with(|count| count.set(0));
    assert!(matches!(
        left.all_with_allocator(1023, |_| panic!("unadmitted allocation")),
        Err(Error::Resource)
    ));
    assert!(matches!(
        left.all_with_allocator(1024, |count| {
            assert_eq!(count, 1024);
            Err(Error::Allocation)
        }),
        Err(Error::Allocation)
    ));
    for capacity in [0, 1023, 1025] {
        assert!(matches!(
            left.all_with_allocator(1024, |_| Ok(Vec::with_capacity(capacity))),
            Err(Error::Resource)
        ));
    }
    assert!(matches!(
        left.all_with_allocator(1024, |_| Ok(vec![1; 1024])),
        Err(Error::Resource)
    ));
    assert_eq!(AES_BLOCKS.with(|count| count.get()), 0);
    let mut calls = 0;
    let output = left
        .all_with_allocator(1024, |count| {
            calls += 1;
            Ok(Vec::with_capacity(count))
        })
        .unwrap();
    assert_eq!(calls, 1);
    assert_eq!(output.len(), 1024);
    assert_eq!(output.capacity(), 1024);
    assert_eq!(
        AES_BLOCKS.with(|count| count.get()),
        profile.resources().full_expansion_aes_blocks.unwrap()
    );
    let other = right.all(1024).unwrap();
    for (index, (&a, &b)) in output.iter().zip(other.iter()).enumerate() {
        assert_eq!(a ^ b, u8::from(index == 1023));
    }
    let [wide, _] = fixture(Profile::new(11).unwrap(), 0, 0, 0);
    assert!(matches!(
        wide.all_with_allocator(usize::MAX, |_| panic!("wide allocation")),
        Err(Error::Resource)
    ));
}

#[test]
fn fresh_os_material_and_negative_terminal_bit_witness_not_a_privacy_proof() {
    let p = Profile::new(8).unwrap();
    let a = GpuKey::issue(p, 127).unwrap();
    let b = GpuKey::issue(p, 127).unwrap();
    assert_ne!(*a[0].root & !1, *b[0].root & !1);
    assert_ne!(*a[1].root & !1, *b[1].root & !1);
    // Finite counterexample to CPU's deterministic endpoint-bit invariant only.
    // Both bit values must occur for a fixed point across valid independent seeds.
    for point in [0, 1, 127, 128, 255] {
        let mut seen = [[false; 2]; 2];
        for salt in 0..64 {
            let keys = fixture(p, point, (salt & 1) as u8, salt);
            for (branch, bits_seen) in seen.iter_mut().enumerate() {
                assert_eq!(keys[0].terminal[branch], keys[1].terminal[branch]);
                bits_seen[(keys[0].terminal[branch] & 1) as usize] = true;
            }
        }
        assert_eq!(seen, [[true, true]; 2]);
    }
}

#[test]
fn frozen_legacy_bytes_random_fields_and_draw_order_remain_unchanged() {
    let roots = [
        u128::from_le_bytes(std::array::from_fn(|i| i as u8)),
        u128::from_le_bytes(std::array::from_fn(|i| (16 + i) as u8)),
    ];
    let mut draws = Vec::new();
    let keys = LegacyKey::issue_with(8, 255, |party| {
        draws.push(party);
        Ok(roots[party])
    })
    .unwrap();
    assert_eq!(draws, [0, 1]);
    let expected = [
        bytes::<49>("000102030405060708090a0b0c0d0e0f2b020bce89638aee6f71ded23a072b218bbe7e6b33f524e61b65407da510fa8601"),
        bytes::<49>("101112131415161718191a1b1c1d1e1f2b020bce89638aee6f71ded23a072b218bbe7e6b33f524e61b65407da510fa8601"),
    ];
    struct Fields {
        bits: Bits,
        random: Vec<(usize, usize, bool)>,
    }
    impl BitSink for Fields {
        fn put(&mut self, value: u128, width: usize) {
            self.bits.put(value, width);
        }
        fn random(&mut self, value: u128, width: usize, field: RandomField) {
            self.random.push((
                self.bits.len,
                width,
                matches!(field, RandomField::BothParties),
            ));
            self.put(value, width);
        }
    }
    for (party, key) in keys.iter().enumerate() {
        let mut fields = Fields {
            bits: Bits::new(),
            random: Vec::new(),
        };
        key.encode(&mut fields);
        assert_eq!(fields.bits.len, 386);
        assert_eq!(&fields.bits.data[..], &expected[party]);
        assert_eq!(fields.random, [(0, 128, true)]);
    }
    for width in [1, 7] {
        draws.clear();
        let keys = LegacyKey::issue_with(width, 1, |party| {
            draws.push(party);
            Ok(roots[party])
        })
        .unwrap();
        assert_eq!(draws, [0]);
        for key in keys {
            let mut fields = Fields {
                bits: Bits::new(),
                random: Vec::new(),
            };
            key.encode(&mut fields);
            assert_eq!(fields.random, [(0, 1 << width, false)]);
        }
    }
}
