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
