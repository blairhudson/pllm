use super::*;

const COSINE_SENTINEL: i64 = i64::MIN + 17;
const SINE_SENTINEL: i64 = i64::MAX - 29;

#[test]
fn rotary_into_matches_legacy_for_every_width_and_public_parameter_edges() {
    let mut cosine = [COSINE_SENTINEL; MAX_TILE_LANES / 2 + 1];
    let mut sine = [SINE_SENTINEL; MAX_TILE_LANES / 2 + 1];
    for width in (2..=MAX_TILE_LANES).step_by(2) {
        let half = width / 2;
        for theta in [
            1.0,
            f32::from_bits(1.0_f32.to_bits() + 1),
            2.0,
            10000.0,
            1000000.0,
            f32::MAX,
        ] {
            for position in [0, 1, 2, 17, 127, 255, 511, 512] {
                let (old_cosine, old_sine) =
                    Program::rotary_coefficients(theta, position, width).unwrap();
                cosine.fill(COSINE_SENTINEL);
                sine.fill(SINE_SENTINEL);
                Program::rotary_coefficients_into(
                    theta,
                    position,
                    width,
                    &mut cosine[..half],
                    &mut sine[..half],
                )
                .unwrap();
                assert_eq!(&cosine[..half], old_cosine.as_slice());
                assert_eq!(&sine[..half], old_sine.as_slice());
                assert!(cosine[half..].iter().all(|&v| v == COSINE_SENTINEL));
                assert!(sine[half..].iter().all(|&v| v == SINE_SENTINEL));
            }
        }
    }
}

#[test]
fn rotary_into_frozen_binary32_q18_endpoints() {
    let mut cosine = [COSINE_SENTINEL; MAX_TILE_LANES / 2];
    let mut sine = [SINE_SENTINEL; MAX_TILE_LANES / 2];
    // Position zero remains exact even at the largest finite theta and width.
    Program::rotary_coefficients_into(f32::MAX, 0, MAX_TILE_LANES, &mut cosine, &mut sine).unwrap();
    assert_eq!(cosine, [262144; MAX_TILE_LANES / 2]);
    assert_eq!(sine, [0; MAX_TILE_LANES / 2]);

    // Theta one fixes every frequency to one. Negative coefficients also freeze
    // toward-zero, not floor, encoding; position 512 is the public endpoint.
    for (position, expected_cosine, expected_sine) in [
        (1, 141637, 220586),
        (4, -171348, -198391),
        (512, -261313, 20845),
    ] {
        Program::rotary_coefficients_into(1.0, position, MAX_TILE_LANES, &mut cosine, &mut sine)
            .unwrap();
        assert_eq!(cosine, [expected_cosine; MAX_TILE_LANES / 2]);
        assert_eq!(sine, [expected_sine; MAX_TILE_LANES / 2]);
    }

    // Non-unit theta freezes the width-dependent frequency ordering, including
    // the finite binary32 angle boundary before the Q18 toward-zero cast.
    Program::rotary_coefficients_into(10000.0, 512, 8, &mut cosine[..4], &mut sine[..4]).unwrap();
    assert_eq!(&cosine[..4], &[-261313, 155767, 103918, 228528]);
    assert_eq!(&sine[..4], &[20845, 210845, -240666, 128430]);
}

#[test]
fn rotary_into_invalid_geometry_preserves_both_destinations_and_legacy_errors() {
    for (theta, position, width) in [
        (f32::NAN, 0, 2),
        (f32::from_bits(0xffc00001), 0, 2),
        (f32::INFINITY, 0, 2),
        (f32::NEG_INFINITY, 0, 2),
        (-1.0, 0, 2),
        (-0.0, 0, 2),
        (0.0, 0, 2),
        (f32::MIN_POSITIVE, 0, 2),
        (f32::from_bits(1.0_f32.to_bits() - 1), 0, 2),
        (1.0, 513, 2),
        (1.0, u32::MAX, 2),
        (1.0, 0, 0),
        (1.0, 0, 1),
        (1.0, 0, 3),
        (1.0, 0, MAX_TILE_LANES + 1),
        (1.0, 0, MAX_TILE_LANES + 2),
        (1.0, 0, usize::MAX - 1),
        (1.0, 0, usize::MAX),
    ] {
        assert_eq!(
            Program::rotary_coefficients(theta, position, width),
            Err(Error::Bounds)
        );
        // Both exact and mismatched lengths leave both entire arrays intact.
        // Invalid geometry wins over destination shape, before any indexing.
        for (cosine_len, sine_len) in [(1, 1), (0, 0), (2, 1), (1, 2)] {
            let mut cosine = [COSINE_SENTINEL; 2];
            let mut sine = [SINE_SENTINEL; 2];
            assert_eq!(
                Program::rotary_coefficients_into(
                    theta,
                    position,
                    width,
                    &mut cosine[..cosine_len],
                    &mut sine[..sine_len],
                ),
                Err(Error::Bounds)
            );
            assert_eq!(cosine, [COSINE_SENTINEL; 2]);
            assert_eq!(sine, [SINE_SENTINEL; 2]);
        }
    }
}

#[test]
fn rotary_into_rejects_either_inexact_length_before_writing_either_destination() {
    const CAPACITY: usize = MAX_TILE_LANES / 2 + 1;
    for width in (2..=MAX_TILE_LANES).step_by(2) {
        let half = width / 2;
        for cosine_len in [0, half - 1, half, half + 1, CAPACITY] {
            for sine_len in [0, half - 1, half, half + 1, CAPACITY] {
                if cosine_len == half && sine_len == half {
                    continue;
                }
                let mut cosine = [COSINE_SENTINEL; CAPACITY];
                let mut sine = [SINE_SENTINEL; CAPACITY];
                assert_eq!(
                    Program::rotary_coefficients_into(
                        10000.0,
                        512,
                        width,
                        &mut cosine[..cosine_len],
                        &mut sine[..sine_len],
                    ),
                    Err(Error::Shape)
                );
                assert_eq!(cosine, [COSINE_SENTINEL; CAPACITY]);
                assert_eq!(sine, [SINE_SENTINEL; CAPACITY]);
            }
        }
    }
}

// Transcribed from parent Python reference_pairs(1_000_000, 64, 4).
// Exactly 160 rows, (p,j) == divmod(k,32), p-major/j-minor order.
// ROTARY_IDEAL_FREQ = first 32 row[2:5], invariant at each j over all p;
// ROTARY_IDEAL_GRID = all row[5:10]. These are independently correctly rounded
// individual-operation ideal-math predictions, NOT guaranteed libm outputs.
// First difference holds bit parity; never replace literals with observations.
const ROTARY_IDEAL_FREQ: [(u32, u32, u32); 32] = [
    (0x00000000, 0x3f800000, 0x3f800000),
    (0x3d000000, 0x3fc51c50, 0x3f263de0),
    (0x3d800000, 0x4017c496, 0x3ed7e89b),
    (0x3dc00000, 0x4069b621, 0x3e8c3504),
    (0x3e000000, 0x40b3f300, 0x3e361887),
    (0x3e200000, 0x410a8de6, 0x3dec7fd6),
    (0x3e400000, 0x41555d0a, 0x3d99940d),
    (0x3e600000, 0x41a44831, 0x3d47763f),
    (0x3e800000, 0x41fcfb72, 0x3d0186e3),
    (0x3e900000, 0x4242c979, 0x3ca8398b),
    (0x3ea00000, 0x4295fa95, 0x3c5a7bf2),
    (0x3eb00000, 0x42e6f4d6, 0x3c0de12d),
    (0x3ec00000, 0x4331d3f4, 0x3bb8449c),
    (0x3ed00000, 0x4388ebc5, 0x3b6f520e),
    (0x3ee00000, 0x43d2d927, 0x3b1b690d),
    (0x3ef00000, 0x4422586d, 0x3ac9d75c),
    (0x3f000000, 0x447a0000, 0x3a83126f),
    (0x3f080000, 0x44c07da6, 0x3a2a3b44),
    (0x3f100000, 0x451435fb, 0x39dd1725),
    (0x3f180000, 0x45643bdc, 0x398f9272),
    (0x3f200000, 0x45afbb4e, 0x393a7753),
    (0x3f280000, 0x46074e93, 0x38f22ce2),
    (0x3f300000, 0x46505cdb, 0x389d43a4),
    (0x3f380000, 0x46a06e80, 0x384c3fbe),
    (0x3f400000, 0x46f70d8e, 0x3804a2b2),
    (0x3f480000, 0x473e38c1, 0x37ac431d),
    (0x3f500000, 0x479276b6, 0x375fba4f),
    (0x3f580000, 0x47e18b19, 0x371148e3),
    (0x3f600000, 0x482da8fc, 0x36bcb0c1),
    (0x3f680000, 0x4885b63f, 0x36751070),
    (0x3f700000, 0x48cde810, 0x361f23e4),
    (0x3f780000, 0x491e8a5a, 0x35ceaf79),
];
const ROTARY_IDEAL_GRID: [(u32, u32, u32, i64, i64); 160] = [
    // position 0
    (0x00000000, 0x3f800000, 0x00000000, 262144, 0),
    (0x00000000, 0x3f800000, 0x00000000, 262144, 0),
    (0x00000000, 0x3f800000, 0x00000000, 262144, 0),
    (0x00000000, 0x3f800000, 0x00000000, 262144, 0),
    (0x00000000, 0x3f800000, 0x00000000, 262144, 0),
    (0x00000000, 0x3f800000, 0x00000000, 262144, 0),
    (0x00000000, 0x3f800000, 0x00000000, 262144, 0),
    (0x00000000, 0x3f800000, 0x00000000, 262144, 0),
    (0x00000000, 0x3f800000, 0x00000000, 262144, 0),
    (0x00000000, 0x3f800000, 0x00000000, 262144, 0),
    (0x00000000, 0x3f800000, 0x00000000, 262144, 0),
    (0x00000000, 0x3f800000, 0x00000000, 262144, 0),
    (0x00000000, 0x3f800000, 0x00000000, 262144, 0),
    (0x00000000, 0x3f800000, 0x00000000, 262144, 0),
    (0x00000000, 0x3f800000, 0x00000000, 262144, 0),
    (0x00000000, 0x3f800000, 0x00000000, 262144, 0),
    (0x00000000, 0x3f800000, 0x00000000, 262144, 0),
    (0x00000000, 0x3f800000, 0x00000000, 262144, 0),
    (0x00000000, 0x3f800000, 0x00000000, 262144, 0),
    (0x00000000, 0x3f800000, 0x00000000, 262144, 0),
    (0x00000000, 0x3f800000, 0x00000000, 262144, 0),
    (0x00000000, 0x3f800000, 0x00000000, 262144, 0),
    (0x00000000, 0x3f800000, 0x00000000, 262144, 0),
    (0x00000000, 0x3f800000, 0x00000000, 262144, 0),
    (0x00000000, 0x3f800000, 0x00000000, 262144, 0),
    (0x00000000, 0x3f800000, 0x00000000, 262144, 0),
    (0x00000000, 0x3f800000, 0x00000000, 262144, 0),
    (0x00000000, 0x3f800000, 0x00000000, 262144, 0),
    (0x00000000, 0x3f800000, 0x00000000, 262144, 0),
    (0x00000000, 0x3f800000, 0x00000000, 262144, 0),
    (0x00000000, 0x3f800000, 0x00000000, 262144, 0),
    (0x00000000, 0x3f800000, 0x00000000, 262144, 0),
    // position 1
    (0x3f800000, 0x3f0a5140, 0x3f576aa4, 141637, 220586),
    (0x3f263de0, 0x3f4be4aa, 0x3f1acd3a, 208786, 158516),
    (0x3ed7e89b, 0x3f6992c6, 0x3ed190f1, 239179, 107297),
    (0x3e8c3504, 0x3f767610, 0x3e8a7618, 252376, 70892),
    (0x3e361887, 0x3f7bf683, 0x3e35233a, 258010, 46371),
    (0x3dec7fd6, 0x3f7e4b84, 0x3debf95d, 260398, 30204),
    (0x3d99940d, 0x3f7f47d1, 0x3d996f36, 261407, 19639),
    (0x3d47763f, 0x3f7fb24f, 0x3d476211, 261833, 12760),
    (0x3d0186e3, 0x3f7fdf3c, 0x3d01815c, 262012, 8288),
    (0x3ca8398b, 0x3f7ff22f, 0x3ca83684, 262088, 5382),
    (0x3c5a7bf2, 0x3f7ffa2c, 0x3c5a7a4a, 262120, 3495),
    (0x3c0de12d, 0x3f7ffd8b, 0x3c0de0b9, 262134, 2270),
    (0x3bb8449c, 0x3f7ffef7, 0x3bb8445c, 262139, 1474),
    (0x3b6f520e, 0x3f7fff90, 0x3b6f51eb, 262142, 957),
    (0x3b1b690d, 0x3f7fffd1, 0x3b1b6903, 262143, 621),
    (0x3ac9d75c, 0x3f7fffec, 0x3ac9d757, 262143, 403),
    (0x3a83126f, 0x3f7ffff8, 0x3a83126e, 262143, 262),
    (0x3a2a3b44, 0x3f7ffffc, 0x3a2a3b43, 262143, 170),
    (0x39dd1725, 0x3f7fffff, 0x39dd1725, 262143, 110),
    (0x398f9272, 0x3f7fffff, 0x398f9272, 262143, 71),
    (0x393a7753, 0x3f800000, 0x393a7753, 262144, 46),
    (0x38f22ce2, 0x3f800000, 0x38f22ce2, 262144, 30),
    (0x389d43a4, 0x3f800000, 0x389d43a4, 262144, 19),
    (0x384c3fbe, 0x3f800000, 0x384c3fbe, 262144, 12),
    (0x3804a2b2, 0x3f800000, 0x3804a2b2, 262144, 8),
    (0x37ac431d, 0x3f800000, 0x37ac431d, 262144, 5),
    (0x375fba4f, 0x3f800000, 0x375fba4f, 262144, 3),
    (0x371148e3, 0x3f800000, 0x371148e3, 262144, 2),
    (0x36bcb0c1, 0x3f800000, 0x36bcb0c1, 262144, 1),
    (0x36751070, 0x3f800000, 0x36751070, 262144, 0),
    (0x361f23e4, 0x3f800000, 0x361f23e4, 262144, 0),
    (0x35ceaf79, 0x3f800000, 0x35ceaf79, 262144, 0),
    // position 2
    (0x40000000, 0xbed51133, 0x3f68c7b7, -109090, 238366),
    (0x3fa63de0, 0x3e8991c6, 0x3f769605, 70435, 252504),
    (0x3f57e89b, 0x3f2a3902, 0x3f3f3512, 174308, 195796),
    (0x3f0c3504, 0x3f5a8e3c, 0x3f054d57, 223800, 136501),
    (0x3eb61887, 0x3f6ffaa5, 0x3eb247f6, 245738, 91279),
    (0x3e6c7fd6, 0x3f7933e2, 0x3e6a6707, 255183, 60007),
    (0x3e19940d, 0x3f7d204f, 0x3e1900d3, 259201, 39168),
    (0x3dc7763f, 0x3f7ec96d, 0x3dc7258f, 260901, 25490),
    (0x3d8186e3, 0x3f7f7cf9, 0x3d8170c9, 261619, 16568),
    (0x3d28398b, 0x3f7fc8bc, 0x3d282d70, 261922, 10763),
    (0x3cda7bf2, 0x3f7fe8b1, 0x3cda7551, 262050, 6990),
    (0x3c8de12d, 0x3f7ff62c, 0x3c8ddf5c, 262104, 4539),
    (0x3c38449c, 0x3f7ffbdb, 0x3c38439d, 262127, 2948),
    (0x3bef520e, 0x3f7ffe41, 0x3bef5183, 262137, 1914),
    (0x3b9b690d, 0x3f7fff43, 0x3b9b68e7, 262141, 1243),
    (0x3b49d75c, 0x3f7fffb0, 0x3b49d747, 262142, 807),
    (0x3b03126f, 0x3f7fffde, 0x3b031269, 262143, 524),
    (0x3aaa3b44, 0x3f7ffff2, 0x3aaa3b41, 262143, 340),
    (0x3a5d1725, 0x3f7ffffa, 0x3a5d1723, 262143, 221),
    (0x3a0f9272, 0x3f7ffffd, 0x3a0f9272, 262143, 143),
    (0x39ba7753, 0x3f7fffff, 0x39ba7753, 262143, 93),
    (0x39722ce2, 0x3f800000, 0x39722ce2, 262144, 60),
    (0x391d43a4, 0x3f800000, 0x391d43a4, 262144, 39),
    (0x38cc3fbe, 0x3f800000, 0x38cc3fbe, 262144, 25),
    (0x3884a2b2, 0x3f800000, 0x3884a2b2, 262144, 16),
    (0x382c431d, 0x3f800000, 0x382c431d, 262144, 10),
    (0x37dfba4f, 0x3f800000, 0x37dfba4f, 262144, 6),
    (0x379148e3, 0x3f800000, 0x379148e3, 262144, 4),
    (0x373cb0c1, 0x3f800000, 0x373cb0c1, 262144, 2),
    (0x36f51070, 0x3f800000, 0x36f51070, 262144, 1),
    (0x369f23e4, 0x3f800000, 0x369f23e4, 262144, 1),
    (0x364eaf79, 0x3f800000, 0x364eaf79, 262144, 0),
    // position 3
    (0x40400000, 0xbf7d7026, 0x3e1081c3, -259520, 36993),
    (0x3ff95cd0, 0xbebca662, 0x3f6dfd34, -96588, 243700),
    (0x3fa1ee74, 0x3e9a1863, 0x3f742163, 78896, 249989),
    (0x3f524f86, 0x3f2e5ce1, 0x3f3b708b, 178547, 191938),
    (0x3f089265, 0x3f5c6d29, 0x3f022f6c, 225716, 133309),
    (0x3eb15fe0, 0x3f70ca76, 0x3eadd906, 246569, 89010),
    (0x3e665e14, 0x3f798c92, 0x3e646de2, 255538, 58477),
    (0x3e1598af, 0x3f7d45e5, 0x3e15109b, 259351, 38160),
    (0x3dc24a54, 0x3f7ed94f, 0x3dc1ffc1, 260965, 24831),
    (0x3d7c5650, 0x3f7f83ad, 0x3d7c2d75, 261646, 16139),
    (0x3d23dcf6, 0x3f7fcb90, 0x3d23d1c6, 261934, 10484),
    (0x3cd4d1c4, 0x3f7fe9e3, 0x3cd4cba3, 262055, 6809),
    (0x3c8a3375, 0x3f7ff6ad, 0x3c8a31c7, 262106, 4422),
    (0x3c337d8a, 0x3f7ffc11, 0x3c337c9f, 262128, 2871),
    (0x3be91d94, 0x3f7ffe57, 0x3be91d13, 262137, 1864),
    (0x3b976185, 0x3f7fff4d, 0x3b976162, 262141, 1211),
    (0x3b449ba6, 0x3f7fffb5, 0x3b449b93, 262142, 786),
    (0x3aff58e6, 0x3f7fffe0, 0x3aff58db, 262143, 510),
    (0x3aa5d15c, 0x3f7ffff3, 0x3aa5d159, 262143, 331),
    (0x3a575bab, 0x3f7ffffa, 0x3a575ba9, 262143, 215),
    (0x3a0bd97e, 0x3f7ffffe, 0x3a0bd97e, 262143, 139),
    (0x39b5a1aa, 0x3f7fffff, 0x39b5a1aa, 262143, 90),
    (0x396be576, 0x3f800000, 0x396be576, 262144, 58),
    (0x39192fce, 0x3f800000, 0x39192fce, 262144, 38),
    (0x38c6f40b, 0x3f800000, 0x38c6f40b, 262144, 24),
    (0x38813256, 0x3f800000, 0x38813256, 262144, 16),
    (0x3827cbbb, 0x3f800000, 0x3827cbbb, 262144, 10),
    (0x37d9ed54, 0x3f800000, 0x37d9ed54, 262144, 6),
    (0x378d8491, 0x3f800000, 0x378d8491, 262144, 4),
    (0x3737cc54, 0x3f800000, 0x3737cc54, 262144, 2),
    (0x36eeb5d6, 0x3f800000, 0x36eeb5d6, 262144, 1),
    (0x369b039b, 0x3f800000, 0x369b039b, 262144, 1),
    // position 4
    (0x40800000, 0xbf275530, 0xbf41bdcf, -171348, -198391),
    (0x40263de0, 0xbf5b0954, 0x3f0482b2, -224293, 135690),
    (0x3fd7e89b, 0xbded03c2, 0x3f7e47a5, -30337, 260382),
    (0x3f8c3504, 0x3eea5a32, 0x3f639bd8, 119988, 233071),
    (0x3f361887, 0x3f41ebed, 0x3f271fbc, 198575, 171134),
    (0x3eec7fd6, 0x3f652bf0, 0x3ee42db7, 234671, 116827),
    (0x3e99940d, 0x3f7491c1, 0x3e974920, 250439, 77458),
    (0x3e47763f, 0x3f7b28a5, 0x3e4633f5, 257186, 50739),
    (0x3e0186e3, 0x3f7df469, 0x3e012e89, 260049, 33070),
    (0x3da8398b, 0x3f7f2309, 0x3da80922, 261260, 21508),
    (0x3d5a7bf2, 0x3f7fa2ca, 0x3d5a616d, 261771, 13976),
    (0x3d0de12d, 0x3f7fd8b0, 0x3d0dd9ea, 261986, 9078),
    (0x3cb8449c, 0x3f7fef6c, 0x3cb840a2, 262077, 5896),
    (0x3c6f520e, 0x3f7ff902, 0x3c6f4fe0, 262116, 3828),
    (0x3c1b690d, 0x3f7ffd0d, 0x3c1b6874, 262132, 2486),
    (0x3bc9d75c, 0x3f7ffec2, 0x3bc9d708, 262139, 1614),
    (0x3b83126f, 0x3f7fff7a, 0x3b831258, 262141, 1048),
    (0x3b2a3b44, 0x3f7fffc7, 0x3b2a3b37, 262143, 680),
    (0x3add1725, 0x3f7fffe8, 0x3add171e, 262143, 442),
    (0x3a8f9272, 0x3f7ffff6, 0x3a8f9270, 262143, 287),
    (0x3a3a7753, 0x3f7ffffc, 0x3a3a7752, 262143, 186),
    (0x39f22ce2, 0x3f7ffffe, 0x39f22ce1, 262143, 121),
    (0x399d43a4, 0x3f7fffff, 0x399d43a4, 262143, 78),
    (0x394c3fbe, 0x3f800000, 0x394c3fbe, 262144, 51),
    (0x3904a2b2, 0x3f800000, 0x3904a2b2, 262144, 33),
    (0x38ac431d, 0x3f800000, 0x38ac431d, 262144, 21),
    (0x385fba4f, 0x3f800000, 0x385fba4f, 262144, 13),
    (0x381148e3, 0x3f800000, 0x381148e3, 262144, 9),
    (0x37bcb0c1, 0x3f800000, 0x37bcb0c1, 262144, 5),
    (0x37751070, 0x3f800000, 0x37751070, 262144, 3),
    (0x371f23e4, 0x3f800000, 0x371f23e4, 262144, 2),
    (0x36ceaf79, 0x3f800000, 0x36ceaf79, 262144, 1),
];

#[test]
fn rotary_ideal_vs_locked_binary32_finite_grid() {
    // Compare independent predictions, not native observations. Native exact
    // checks remain in rotary_locked_libm_{power,inverse_angle,trig}_grid.
    // Derivation: zip reference_pairs(1_000_000, 64, 4) from rotary_reference
    // with locked_trig_pairs() from test_rotary_locked_trig. Both use exact
    // rational arithmetic and explicit RN-even rounding; only the ideal side
    // uses mathematical power/trig. No expected word is retuned here.
    // Bit j of each mask denotes a discrepancy at that frequency. Frequency
    // stages are position-independent; later stages have one mask per p.
    const FREQUENCY_DIFFERENCES: u32 = 0x02400200; // j = 9, 22, 25
    const GRID_DIFFERENCES: [u32; 5] = [
        0x00000000, 0x02400200, 0x02400200, 0x00400200, 0x02400200,
    ]; // p=3,j=25 reconverges after the separately rounded angle product.

    let mut frequency_masks = [0u32; 3];
    let mut first_power_difference = None;
    for j in 0..32 {
        let (ideal_exponent, ideal_power, ideal_inverse) = ROTARY_IDEAL_FREQ[j];
        let (locked_exponent, locked_power) = ROTARY_LOCKED_POWERS[j];
        for (stage, ideal, locked) in [
            (0, ideal_exponent, locked_exponent),
            (1, ideal_power, locked_power),
            (2, ideal_inverse, ROTARY_LOCKED_INVERSES[j]),
        ] {
            if ideal != locked {
                frequency_masks[stage] |= 1u32 << j;
            }
        }
        if ideal_power != locked_power && first_power_difference.is_none() {
            first_power_difference = Some((
                j,
                ROTARY_LOCKED_THETA_BITS,
                locked_exponent,
                ideal_power,
                locked_power,
            ));
        }
    }
    for (stage, actual, expected) in [
        ("exponent", frequency_masks[0], 0),
        ("positive_power", frequency_masks[1], FREQUENCY_DIFFERENCES),
        ("reciprocal", frequency_masks[2], FREQUENCY_DIFFERENCES),
    ] {
        assert_eq!(actual, expected, "{stage} discrepancy mask");
    }
    assert_eq!(
        first_power_difference,
        Some((9, 0x49742400, 0x3e900000, 0x4242c979, 0x4242c97a)),
        "first positive-power witness (j, theta, exponent, ideal, locked)",
    );

    // Pin every differing power and reciprocal from the independent Python
    // references above. Masks alone would accept altered words that still
    // differ at the same j, including an ideal-only mutation at j=22.
    for (j, ideal_power, locked_power, ideal_inverse, locked_inverse) in [
        (9, 0x4242c979, 0x4242c97a, 0x3ca8398b, 0x3ca8398a),
        (22, 0x46505cdb, 0x46505cdc, 0x389d43a4, 0x389d43a3),
        (25, 0x473e38c1, 0x473e38c0, 0x37ac431d, 0x37ac431e),
    ] {
        for (stage, expected, actual) in [
            ("ideal positive_power", ideal_power, ROTARY_IDEAL_FREQ[j].1),
            ("locked positive_power", locked_power, ROTARY_LOCKED_POWERS[j].1),
            ("ideal reciprocal", ideal_inverse, ROTARY_IDEAL_FREQ[j].2),
            ("locked reciprocal", locked_inverse, ROTARY_LOCKED_INVERSES[j]),
        ] {
            assert!(
                actual == expected,
                "{stage} literal witness j={j}, theta=0x{ROTARY_LOCKED_THETA_BITS:08x}, expected=0x{expected:08x}, actual=0x{actual:08x}",
            );
        }
    }

    let mut grid_masks = [[0u32; 5]; 5];
    let q18 = |word| (f32::from_bits(word) * 262144.0_f32) as i64;
    for k in 0..160 {
        let (p, j) = (k / 32, k % 32);
        let (ideal_angle, ideal_cos, ideal_sin, cos_q18, sin_q18) = ROTARY_IDEAL_GRID[k];
        let (locked_cos, locked_sin) = ROTARY_LOCKED_TRIG[k];
        // Validate ideal words against their independent integer literals too;
        // a common-mode Q18 change cannot hide behind a zero difference mask.
        assert_eq!(q18(ideal_cos), cos_q18, "ideal cos_q18 p={p}, j={j}");
        assert_eq!(q18(ideal_sin), sin_q18, "ideal sin_q18 p={p}, j={j}");
        for (stage, differs) in [
            (0, ideal_angle != ROTARY_LOCKED_ANGLES[k]),
            (1, ideal_cos != locked_cos),
            (2, ideal_sin != locked_sin),
            (3, cos_q18 != q18(locked_cos)),
            (4, sin_q18 != q18(locked_sin)),
        ] {
            if differs {
                grid_masks[stage][p] |= 1u32 << j;
            }
        }
    }
    for (stage, actual, expected) in [
        ("angle", grid_masks[0], GRID_DIFFERENCES),
        ("cos", grid_masks[1], [0; 5]),
        ("sin", grid_masks[2], GRID_DIFFERENCES),
        ("cos_q18", grid_masks[3], [0; 5]),
        ("sin_q18", grid_masks[4], [0; 5]),
    ] {
        assert_eq!(actual, expected, "{stage} discrepancy masks by position");
    }
    assert_eq!(
        grid_masks[2].iter().map(|mask| mask.count_ones()).sum::<u32>(),
        11,
    );
}

// Generated before native comparison by tests/test_rotary_locked_power.py
// (_locked_power). This is the libm 0.2.16 source profile with RN-even
// binary32 per operator, without fusion/reassociation, not the ideal grid.
// A match is evidence only for the tested stages on the target/profile tested.
const ROTARY_LOCKED_THETA_BITS: u32 = 0x49742400;
const ROTARY_LOCKED_POWERS: [(u32, u32); 32] = [
    (0x00000000, 0x3f800000),
    (0x3d000000, 0x3fc51c50),
    (0x3d800000, 0x4017c496),
    (0x3dc00000, 0x4069b621),
    (0x3e000000, 0x40b3f300),
    (0x3e200000, 0x410a8de6),
    (0x3e400000, 0x41555d0a),
    (0x3e600000, 0x41a44831),
    (0x3e800000, 0x41fcfb72),
    (0x3e900000, 0x4242c97a),
    (0x3ea00000, 0x4295fa95),
    (0x3eb00000, 0x42e6f4d6),
    (0x3ec00000, 0x4331d3f4),
    (0x3ed00000, 0x4388ebc5),
    (0x3ee00000, 0x43d2d927),
    (0x3ef00000, 0x4422586d),
    (0x3f000000, 0x447a0000),
    (0x3f080000, 0x44c07da6),
    (0x3f100000, 0x451435fb),
    (0x3f180000, 0x45643bdc),
    (0x3f200000, 0x45afbb4e),
    (0x3f280000, 0x46074e93),
    (0x3f300000, 0x46505cdc),
    (0x3f380000, 0x46a06e80),
    (0x3f400000, 0x46f70d8e),
    (0x3f480000, 0x473e38c0),
    (0x3f500000, 0x479276b6),
    (0x3f580000, 0x47e18b19),
    (0x3f600000, 0x482da8fc),
    (0x3f680000, 0x4885b63f),
    (0x3f700000, 0x48cde810),
    (0x3f780000, 0x491e8a5a),
];

// Generated before native comparison with PYTHONPATH=tests .venv/bin/python -B:
// I[j] = round_f32(1 / f32_dyadic(_locked_power(j)))
// A[p,j] = round_f32(p * f32_dyadic(I[j])), p=0..4, j=0..31.
// rotary_reference supplies only exact dyadic decoding and RN-even rounding.
// Angles are p-major/j-minor, with +0 for every position-zero product.
const ROTARY_LOCKED_INVERSES: [u32; 32] = [
    0x3f800000, 0x3f263de0, 0x3ed7e89b, 0x3e8c3504, 0x3e361887, 0x3dec7fd6, 0x3d99940d, 0x3d47763f,
    0x3d0186e3, 0x3ca8398a, 0x3c5a7bf2, 0x3c0de12d, 0x3bb8449c, 0x3b6f520e, 0x3b1b690d, 0x3ac9d75c,
    0x3a83126f, 0x3a2a3b44, 0x39dd1725, 0x398f9272, 0x393a7753, 0x38f22ce2, 0x389d43a3, 0x384c3fbe,
    0x3804a2b2, 0x37ac431e, 0x375fba4f, 0x371148e3, 0x36bcb0c1, 0x36751070, 0x361f23e4, 0x35ceaf79,
];
const ROTARY_LOCKED_ANGLES: [u32; 160] = [
    // position 0
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    // position 1
    0x3f800000, 0x3f263de0, 0x3ed7e89b, 0x3e8c3504, 0x3e361887, 0x3dec7fd6, 0x3d99940d, 0x3d47763f,
    0x3d0186e3, 0x3ca8398a, 0x3c5a7bf2, 0x3c0de12d, 0x3bb8449c, 0x3b6f520e, 0x3b1b690d, 0x3ac9d75c,
    0x3a83126f, 0x3a2a3b44, 0x39dd1725, 0x398f9272, 0x393a7753, 0x38f22ce2, 0x389d43a3, 0x384c3fbe,
    0x3804a2b2, 0x37ac431e, 0x375fba4f, 0x371148e3, 0x36bcb0c1, 0x36751070, 0x361f23e4, 0x35ceaf79,
    // position 2
    0x40000000, 0x3fa63de0, 0x3f57e89b, 0x3f0c3504, 0x3eb61887, 0x3e6c7fd6, 0x3e19940d, 0x3dc7763f,
    0x3d8186e3, 0x3d28398a, 0x3cda7bf2, 0x3c8de12d, 0x3c38449c, 0x3bef520e, 0x3b9b690d, 0x3b49d75c,
    0x3b03126f, 0x3aaa3b44, 0x3a5d1725, 0x3a0f9272, 0x39ba7753, 0x39722ce2, 0x391d43a3, 0x38cc3fbe,
    0x3884a2b2, 0x382c431e, 0x37dfba4f, 0x379148e3, 0x373cb0c1, 0x36f51070, 0x369f23e4, 0x364eaf79,
    // position 3
    0x40400000, 0x3ff95cd0, 0x3fa1ee74, 0x3f524f86, 0x3f089265, 0x3eb15fe0, 0x3e665e14, 0x3e1598af,
    0x3dc24a54, 0x3d7c564f, 0x3d23dcf6, 0x3cd4d1c4, 0x3c8a3375, 0x3c337d8a, 0x3be91d94, 0x3b976185,
    0x3b449ba6, 0x3aff58e6, 0x3aa5d15c, 0x3a575bab, 0x3a0bd97e, 0x39b5a1aa, 0x396be574, 0x39192fce,
    0x38c6f40b, 0x38813256, 0x3827cbbb, 0x37d9ed54, 0x378d8491, 0x3737cc54, 0x36eeb5d6, 0x369b039b,
    // position 4
    0x40800000, 0x40263de0, 0x3fd7e89b, 0x3f8c3504, 0x3f361887, 0x3eec7fd6, 0x3e99940d, 0x3e47763f,
    0x3e0186e3, 0x3da8398a, 0x3d5a7bf2, 0x3d0de12d, 0x3cb8449c, 0x3c6f520e, 0x3c1b690d, 0x3bc9d75c,
    0x3b83126f, 0x3b2a3b44, 0x3add1725, 0x3a8f9272, 0x3a3a7753, 0x39f22ce2, 0x399d43a3, 0x394c3fbe,
    0x3904a2b2, 0x38ac431e, 0x385fba4f, 0x381148e3, 0x37bcb0c1, 0x37751070, 0x371f23e4, 0x36ceaf79,
];

fn check_locked_word(stage: &str, p: usize, j: usize, operands: [f32; 2], expected: u32, actual: f32) {
    let actual = actual.to_bits();
    assert!(
        expected == actual,
        "{stage} p={p}, j={j}, operands=[0x{:08x},0x{:08x}], expected=0x{expected:08x}, actual=0x{actual:08x}",
        operands[0].to_bits(),
        operands[1].to_bits(),
    );
}

fn checked_locked_powers() -> [f32; 32] {
    let theta = f32::from_bits(ROTARY_LOCKED_THETA_BITS);
    let denominator = 64.0_f32;
    let mut powers = [0.0_f32; 32];
    // Exponents, powers and inverses are position-independent; report p=0.
    for (j, &(expected_exponent, expected_power)) in ROTARY_LOCKED_POWERS.iter().enumerate() {
        let numerator = (2 * j) as f32;
        let exponent = numerator / denominator;
        check_locked_word(
            "exponent", 0, j, [numerator, denominator], expected_exponent, exponent,
        );
        let power = libm::powf(theta, exponent);
        check_locked_word("power", 0, j, [theta, exponent], expected_power, power);
        powers[j] = power;
    }
    powers
}

#[test]
fn rotary_locked_libm_power_grid() {
    let _ = checked_locked_powers();
}

fn checked_locked_angles() -> [f32; 160] {
    let powers = checked_locked_powers();
    let mut inverses = [0.0_f32; 32];
    // Check all separately rounded reciprocals before any angle product.
    for (j, &expected) in ROTARY_LOCKED_INVERSES.iter().enumerate() {
        let power = powers[j];
        let inverse = 1.0_f32 / power;
        check_locked_word("inverse", 0, j, [1.0_f32, power], expected, inverse);
        inverses[j] = inverse;
    }
    let mut angles = [0.0_f32; 160];
    for (k, &expected) in ROTARY_LOCKED_ANGLES.iter().enumerate() {
        let (p, j) = (k / 32, k % 32);
        let position = p as f32;
        let inverse = inverses[j];
        let angle = position * inverse;
        check_locked_word("angle", p, j, [position, inverse], expected, angle);
        angles[k] = angle;
    }
    angles
}

#[test]
fn rotary_locked_libm_inverse_angle_grid() {
    let _ = checked_locked_angles();
}

// Frozen before native comparison with PYTHONPATH=tests .venv/bin/python -B:
// [(r.cosine, r.sine) for r in locked_trig_pairs()] from test_rotary_locked_trig.
// SHA-256 of p-major/j-minor, big-endian u32 cos then sin (_word_digest):
// 03385e151098598cfa2914609194cb1c0c959939f15d524ec8fc7175864f53b8
const ROTARY_LOCKED_TRIG: [(u32, u32); 160] = [
    // position 0
    (0x3f800000, 0x00000000),
    (0x3f800000, 0x00000000),
    (0x3f800000, 0x00000000),
    (0x3f800000, 0x00000000),
    (0x3f800000, 0x00000000),
    (0x3f800000, 0x00000000),
    (0x3f800000, 0x00000000),
    (0x3f800000, 0x00000000),
    (0x3f800000, 0x00000000),
    (0x3f800000, 0x00000000),
    (0x3f800000, 0x00000000),
    (0x3f800000, 0x00000000),
    (0x3f800000, 0x00000000),
    (0x3f800000, 0x00000000),
    (0x3f800000, 0x00000000),
    (0x3f800000, 0x00000000),
    (0x3f800000, 0x00000000),
    (0x3f800000, 0x00000000),
    (0x3f800000, 0x00000000),
    (0x3f800000, 0x00000000),
    (0x3f800000, 0x00000000),
    (0x3f800000, 0x00000000),
    (0x3f800000, 0x00000000),
    (0x3f800000, 0x00000000),
    (0x3f800000, 0x00000000),
    (0x3f800000, 0x00000000),
    (0x3f800000, 0x00000000),
    (0x3f800000, 0x00000000),
    (0x3f800000, 0x00000000),
    (0x3f800000, 0x00000000),
    (0x3f800000, 0x00000000),
    (0x3f800000, 0x00000000),
    // position 1
    (0x3f0a5140, 0x3f576aa4),
    (0x3f4be4aa, 0x3f1acd3a),
    (0x3f6992c6, 0x3ed190f1),
    (0x3f767610, 0x3e8a7618),
    (0x3f7bf683, 0x3e35233a),
    (0x3f7e4b84, 0x3debf95d),
    (0x3f7f47d1, 0x3d996f36),
    (0x3f7fb24f, 0x3d476211),
    (0x3f7fdf3c, 0x3d01815c),
    (0x3f7ff22f, 0x3ca83683),
    (0x3f7ffa2c, 0x3c5a7a4a),
    (0x3f7ffd8b, 0x3c0de0b9),
    (0x3f7ffef7, 0x3bb8445c),
    (0x3f7fff90, 0x3b6f51eb),
    (0x3f7fffd1, 0x3b1b6903),
    (0x3f7fffec, 0x3ac9d757),
    (0x3f7ffff8, 0x3a83126e),
    (0x3f7ffffc, 0x3a2a3b43),
    (0x3f7fffff, 0x39dd1725),
    (0x3f7fffff, 0x398f9272),
    (0x3f800000, 0x393a7753),
    (0x3f800000, 0x38f22ce2),
    (0x3f800000, 0x389d43a3),
    (0x3f800000, 0x384c3fbe),
    (0x3f800000, 0x3804a2b2),
    (0x3f800000, 0x37ac431e),
    (0x3f800000, 0x375fba4f),
    (0x3f800000, 0x371148e3),
    (0x3f800000, 0x36bcb0c1),
    (0x3f800000, 0x36751070),
    (0x3f800000, 0x361f23e4),
    (0x3f800000, 0x35ceaf79),
    // position 2
    (0xbed51133, 0x3f68c7b7),
    (0x3e8991c6, 0x3f769605),
    (0x3f2a3902, 0x3f3f3512),
    (0x3f5a8e3c, 0x3f054d57),
    (0x3f6ffaa5, 0x3eb247f6),
    (0x3f7933e2, 0x3e6a6707),
    (0x3f7d204f, 0x3e1900d3),
    (0x3f7ec96d, 0x3dc7258f),
    (0x3f7f7cf9, 0x3d8170c9),
    (0x3f7fc8bc, 0x3d282d6f),
    (0x3f7fe8b1, 0x3cda7551),
    (0x3f7ff62c, 0x3c8ddf5c),
    (0x3f7ffbdb, 0x3c38439d),
    (0x3f7ffe41, 0x3bef5183),
    (0x3f7fff43, 0x3b9b68e7),
    (0x3f7fffb0, 0x3b49d747),
    (0x3f7fffde, 0x3b031269),
    (0x3f7ffff2, 0x3aaa3b41),
    (0x3f7ffffa, 0x3a5d1723),
    (0x3f7ffffd, 0x3a0f9272),
    (0x3f7fffff, 0x39ba7753),
    (0x3f800000, 0x39722ce2),
    (0x3f800000, 0x391d43a3),
    (0x3f800000, 0x38cc3fbe),
    (0x3f800000, 0x3884a2b2),
    (0x3f800000, 0x382c431e),
    (0x3f800000, 0x37dfba4f),
    (0x3f800000, 0x379148e3),
    (0x3f800000, 0x373cb0c1),
    (0x3f800000, 0x36f51070),
    (0x3f800000, 0x369f23e4),
    (0x3f800000, 0x364eaf79),
    // position 3
    (0xbf7d7026, 0x3e1081c3),
    (0xbebca662, 0x3f6dfd34),
    (0x3e9a1863, 0x3f742163),
    (0x3f2e5ce1, 0x3f3b708b),
    (0x3f5c6d29, 0x3f022f6c),
    (0x3f70ca76, 0x3eadd906),
    (0x3f798c92, 0x3e646de2),
    (0x3f7d45e5, 0x3e15109b),
    (0x3f7ed94f, 0x3dc1ffc1),
    (0x3f7f83ad, 0x3d7c2d74),
    (0x3f7fcb90, 0x3d23d1c6),
    (0x3f7fe9e3, 0x3cd4cba3),
    (0x3f7ff6ad, 0x3c8a31c7),
    (0x3f7ffc11, 0x3c337c9f),
    (0x3f7ffe57, 0x3be91d13),
    (0x3f7fff4d, 0x3b976162),
    (0x3f7fffb5, 0x3b449b93),
    (0x3f7fffe0, 0x3aff58db),
    (0x3f7ffff3, 0x3aa5d159),
    (0x3f7ffffa, 0x3a575ba9),
    (0x3f7ffffe, 0x3a0bd97e),
    (0x3f7fffff, 0x39b5a1aa),
    (0x3f800000, 0x396be574),
    (0x3f800000, 0x39192fce),
    (0x3f800000, 0x38c6f40b),
    (0x3f800000, 0x38813256),
    (0x3f800000, 0x3827cbbb),
    (0x3f800000, 0x37d9ed54),
    (0x3f800000, 0x378d8491),
    (0x3f800000, 0x3737cc54),
    (0x3f800000, 0x36eeb5d6),
    (0x3f800000, 0x369b039b),
    // position 4
    (0xbf275530, 0xbf41bdcf),
    (0xbf5b0954, 0x3f0482b2),
    (0xbded03c2, 0x3f7e47a5),
    (0x3eea5a32, 0x3f639bd8),
    (0x3f41ebed, 0x3f271fbc),
    (0x3f652bf0, 0x3ee42db7),
    (0x3f7491c1, 0x3e974920),
    (0x3f7b28a5, 0x3e4633f5),
    (0x3f7df469, 0x3e012e89),
    (0x3f7f2309, 0x3da80921),
    (0x3f7fa2ca, 0x3d5a616d),
    (0x3f7fd8b0, 0x3d0dd9ea),
    (0x3f7fef6c, 0x3cb840a2),
    (0x3f7ff902, 0x3c6f4fe0),
    (0x3f7ffd0d, 0x3c1b6874),
    (0x3f7ffec2, 0x3bc9d708),
    (0x3f7fff7a, 0x3b831258),
    (0x3f7fffc7, 0x3b2a3b37),
    (0x3f7fffe8, 0x3add171e),
    (0x3f7ffff6, 0x3a8f9270),
    (0x3f7ffffc, 0x3a3a7752),
    (0x3f7ffffe, 0x39f22ce1),
    (0x3f7fffff, 0x399d43a3),
    (0x3f800000, 0x394c3fbe),
    (0x3f800000, 0x3904a2b2),
    (0x3f800000, 0x38ac431e),
    (0x3f800000, 0x385fba4f),
    (0x3f800000, 0x381148e3),
    (0x3f800000, 0x37bcb0c1),
    (0x3f800000, 0x37751070),
    (0x3f800000, 0x371f23e4),
    (0x3f800000, 0x36ceaf79),
];

#[test]
fn rotary_locked_libm_trig_grid() {
    let angles = checked_locked_angles();
    let mut trig = [(0.0_f32, 0.0_f32); 160];
    // Check every native output word before performing any signed Q18 cast.
    for (k, &(cos_word, sin_word)) in ROTARY_LOCKED_TRIG.iter().enumerate() {
        let (p, j) = (k / 32, k % 32);
        let angle = angles[k];
        let cosine = libm::cosf(angle);
        let sine = libm::sinf(angle);
        for (stage, expected, result) in [("cos", cos_word, cosine), ("sin", sin_word, sine)] {
            let actual = result.to_bits();
            assert!(
                actual == expected,
                "{stage} p={p}, j={j}, input=0x{:08x}, expected=0x{expected:08x}, actual=0x{actual:08x}",
                angle.to_bits(),
            );
        }
        trig[k] = (cosine, sine);
    }

    let scale = 262144.0_f32;
    for (k, &(cosine, sine)) in trig.iter().enumerate() {
        let (p, j) = (k / 32, k % 32);
        let (_, _, _, cos_q18, sin_q18) = ROTARY_IDEAL_GRID[k];
        for (stage, expected, operand) in [("cos_q18", cos_q18, cosine), ("sin_q18", sin_q18, sine)] {
            let actual = (operand * scale) as i64;
            assert!(
                actual == expected,
                "{stage} p={p}, j={j}, operands=[0x{:08x},0x{:08x}], expected={expected}, actual={actual}",
                operand.to_bits(),
                scale.to_bits(),
            );
        }
    }

    let theta = f32::from_bits(ROTARY_LOCKED_THETA_BITS);
    let width = 64;
    // Each API is checked against the ideal integer literals, not native casts
    // or results from the other API.
    let check_api = |api: &str, p: usize, cosine: &[i64], sine: &[i64]| {
        assert!(
            cosine.len() == 32 && sine.len() == 32,
            "{api}.length p={p}, theta=0x{:08x}, width={width}, expected=(32,32), actual=({},{})",
            theta.to_bits(),
            cosine.len(),
            sine.len(),
        );
        for j in 0..32 {
            let (_, _, _, cos_q18, sin_q18) = ROTARY_IDEAL_GRID[p * 32 + j];
            for (stage, expected, actual) in [("cos", cos_q18, cosine[j]), ("sin", sin_q18, sine[j])] {
                assert!(
                    actual == expected,
                    "{api}.{stage} p={p}, j={j}, theta=0x{:08x}, width={width}, expected={expected}, actual={actual}",
                    theta.to_bits(),
                );
            }
        }
    };
    for p in 0..5 {
        let (cosine, sine) = Program::rotary_coefficients(theta, p as u32, width)
            .unwrap_or_else(|error| panic!(
                "rotary_coefficients.result p={p}, theta=0x{:08x}, width={width}, expected=Ok, actual=Err({error:?})",
                theta.to_bits(),
            ));
        check_api("rotary_coefficients", p, &cosine, &sine);

        let mut cosine = [COSINE_SENTINEL; 32];
        let mut sine = [SINE_SENTINEL; 32];
        Program::rotary_coefficients_into(theta, p as u32, width, &mut cosine, &mut sine)
            .unwrap_or_else(|error| panic!(
                "rotary_coefficients_into.result p={p}, theta=0x{:08x}, width={width}, expected=Ok, actual=Err({error:?})",
                theta.to_bits(),
            ));
        check_api("rotary_coefficients_into", p, &cosine, &sine);
    }
}
