use super::*;

fn pair(point: u64, phase: u8) -> [GpuKey; 2] {
    let mut calls = 0;
    let keys = GpuKey::issue_with(Profile::new(13).unwrap(), point, |bytes| {
        calls += 1;
        assert_eq!(bytes.len(), 32);
        bytes[..16].copy_from_slice(&0x123456789abcdef0u128.to_le_bytes());
        bytes[16..].copy_from_slice(&(0xfedcba9876543210u128 | u128::from(phase)).to_le_bytes());
        Ok(())
    })
    .unwrap();
    assert_eq!(calls, 1);
    assert_eq!((*keys[0].root & 1) as u8, 1 ^ phase);
    assert_eq!((*keys[1].root & 1) as u8, phase);
    keys
}

#[test]
fn exact_resources_and_measured_original_work() {
    let p = Profile::new(13).unwrap();
    assert_eq!(p.depth(), 6);
    assert_eq!(p.ordinary(), 5);
    let q = p.layernorm13_resources().unwrap();
    assert_eq!(q.logical_key_bits, 1029);
    assert_eq!(q.codec_bytes, 147);
    assert_eq!(q.key_owner_bytes, std::mem::size_of::<GpuKey>());
    assert!(q.key_owner_bytes > q.codec_bytes);
    assert_eq!(q.full_expansion_bytes, Some(8192));
    assert_eq!(q.full_expansion_aes_blocks, Some(126));
    AES_BLOCKS.with(|c| c.set(0));
    let keys = pair(128, 1);
    assert_eq!(AES_BLOCKS.with(|c| c.get()), 24);
    for key in &keys {
        AES_BLOCKS.with(|c| c.set(0));
        key.point(8191).unwrap();
        assert_eq!(AES_BLOCKS.with(|c| c.get()), 6);
        AES_BLOCKS.with(|c| c.set(0));
        key.less(8191).unwrap();
        assert_eq!(AES_BLOCKS.with(|c| c.get()), 6);
        AES_BLOCKS.with(|c| c.set(0));
        let out = key
            .all_layernorm13_with_allocator(8192, |n| Ok(Vec::with_capacity(n)))
            .unwrap();
        assert_eq!(AES_BLOCKS.with(|c| c.get()), 126);
        assert_eq!((out.len(), out.capacity()), (8192, 8192));
    }
}

#[test]
fn selected_points_both_root_phases_full_one_hot_and_original_codec() {
    let p = Profile::new(13).unwrap();
    for point in [0, 127, 128, 255, 256, 4095, 4096, 8191] {
        for phase in [0, 1] {
            let keys = pair(point, phase);
            let all: Vec<_> = keys
                .iter()
                .map(|k| {
                    k.all_layernorm13_with_allocator(8192, |n| Ok(Vec::with_capacity(n)))
                        .unwrap()
                })
                .collect();
            for x in 0..8192 {
                assert!(all[0][x] <= 1 && all[1][x] <= 1);
                assert_eq!(all[0][x] ^ all[1][x], u8::from(x as u64 == point));
            }
            for (role, key) in keys.into_iter().enumerate() {
                let material = key.into_material(147).unwrap();
                assert_eq!(material.len(), 147);
                let decoded = GpuKey::decode(p, role as u8, &material).unwrap();
                let reconstructed = decoded
                    .all_layernorm13_with_allocator(8192, |n| Ok(Vec::with_capacity(n)))
                    .unwrap();
                assert_eq!(&reconstructed[..], &all[role][..]);
                assert_eq!(&decoded.into_material(147).unwrap()[..], &material[..]);
                assert!(GpuKey::decode(p, (1 - role) as u8, &material).is_err());
                assert!(GpuKey::decode(Profile::new(12).unwrap(), role as u8, &material).is_err());
                assert!(GpuKey::decode(p, 2, &material).is_err());
                let mut bad = material.to_vec();
                *bad.last_mut().unwrap() |= 0x80;
                assert!(GpuKey::decode(p, role as u8, &bad).is_err());
            }
        }
    }
}

#[test]
fn allocation_refusals_precede_aes_and_limit_precedes_callback() {
    let [key, _] = pair(0, 0);
    AES_BLOCKS.with(|c| c.set(0));
    assert!(matches!(
        key.all_layernorm13_with_allocator(8191, |_| panic!("short limit callback")),
        Err(Error::Resource)
    ));
    assert!(matches!(
        key.all_layernorm13_with_allocator(8192, |_| Err(Error::Allocation)),
        Err(Error::Allocation)
    ));
    for mode in 0..3 {
        assert!(matches!(
            key.all_layernorm13_with_allocator(8192, |n| {
                let mut out = Vec::with_capacity(if mode == 0 {
                    n - 1
                } else if mode == 1 {
                    n + 1
                } else {
                    n
                });
                if mode == 2 {
                    out.push(1);
                }
                Ok(out)
            }),
            Err(Error::Resource)
        ));
    }
    assert_eq!(AES_BLOCKS.with(|c| c.get()), 0);
    let before = ERASED_KEYS.with(|c| c.get());
    drop(key);
    assert_eq!(ERASED_KEYS.with(|c| c.get()), before + 1);
}

#[test]
fn private_profile_gate_never_changes_legacy_full_gate() {
    assert_eq!(MAX_FULL_BITS, 10);
    let p = Profile::new(13).unwrap();
    assert_eq!(p.resources().full_expansion_bytes, None);
    let [key, _] = pair(0, 0);
    assert!(key.all(8192).is_err());
    assert!(key
        .all_with_allocator(8192, |_| panic!("legacy callback"))
        .is_err());
    for bits in [8, 10, 12, 14, 63] {
        let p = Profile::new(bits).unwrap();
        assert!(matches!(p.layernorm13_resources(), Err(Error::Domain)));
        let [key, _] = GpuKey::issue_with(p, 0, |bytes| {
            bytes.fill(0);
            Ok(())
        })
        .unwrap();
        assert!(matches!(
            key.all_layernorm13_with_allocator(8192, |_| panic!("wrong profile callback")),
            Err(Error::Domain)
        ));
        assert_eq!(
            p.resources().full_expansion_bytes,
            if bits <= 10 {
                Some(1usize << bits)
            } else {
                None
            }
        );
    }
}
