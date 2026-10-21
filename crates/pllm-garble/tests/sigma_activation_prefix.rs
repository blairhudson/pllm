//! Independent integer specification for the approved TR6 -> DReLU -> Select
//! prefix, plus public consuming/structural tests. Native numerical parity belongs
//! to region-private unit tests; public outputs and dealer masks stay opaque.

use pllm_core::edge_contract::{
    EdgeDescriptor, EdgeIdentity, Interpretation, Numeric, Representation, Shape, SourceIdentity,
};
use pllm_garble::masked_wire_reference::typed::{
    approved_implementation, ActivationPrefixParty, BoundRange, Budget, CommonMasked, DealerMask,
    Error, LocalDealer, PrefixFrame, PrefixSpec, PrefixStep, Ring, Step,
};

fn source() -> SourceIdentity {
    SourceIdentity::new([1; 32], [2; 32], approved_implementation())
}

fn edge(id: u8, bits: u8, fractional: u8, lanes: usize, predicate: bool) -> EdgeDescriptor {
    EdgeDescriptor::new(
        EdgeIdentity::new(source(), [3; 32], [id + 64; 32], [id; 32], 0),
        Shape::new(&[lanes]).unwrap(),
        Numeric::new(
            bits,
            fractional,
            if predicate {
                Interpretation::Predicate
            } else {
                Interpretation::Signed
            },
        )
        .unwrap(),
        if predicate {
            Representation::CommonXorMasked
        } else {
            Representation::CommonAdditiveMasked
        },
        None,
    )
    .unwrap()
}

struct Fixture<'b> {
    mask: DealerMask<'b, Ring>,
    parties: [ActivationPrefixParty<'b>; 2],
    inputs: [CommonMasked<'b, Ring>; 2],
    output: EdgeDescriptor,
}

fn fixture<'b>(dealer: &mut LocalDealer<'b>, bits: u8, lanes: usize) -> Fixture<'b> {
    let input = edge(5, bits, 12, lanes, false);
    let output = edge(8, bits - 6, 6, lanes, false);
    let half = 1i128 << (bits - 1);
    let samples = [
        -half,
        -half + 1,
        -129,
        -65,
        -64,
        -1,
        0,
        1,
        63,
        64,
        65,
        half - 1,
    ];
    let values: Vec<_> = (0..lanes)
        .map(|i| i64::try_from(samples[i % samples.len()]).unwrap())
        .collect();
    // Independently declared full signed domain, never sample-derived extrema.
    let (input_mask, inputs) = dealer
        .prepare_root(input, BoundRange::new(-half, half).unwrap(), [7; 32])
        .unwrap()
        .encode(&values)
        .unwrap();
    let spec = PrefixSpec::new(
        [7; 32],
        input,
        edge(6, bits - 6, 6, lanes, false),
        edge(7, 1, 0, lanes, true),
        output,
        [9; 32],
    )
    .unwrap();
    let (mask, parties) = dealer.activation_prefix(spec, input_mask).unwrap();
    Fixture {
        mask,
        parties,
        inputs,
        output,
    }
}

// Literal protocol expectations, not phase counts inferred from observed output.
const PREFIX_EXCHANGES: usize = 4;
const PREFIX_DIRECTIONAL_FRAMES: usize = 8;
const FRAME_HEADER_BYTES: usize = 76;
const FRAME_LANE_BYTES: usize = 8;

#[test]
fn public_prefix_consumes_original_inputs_and_hands_outputs_to_drelu() {
    for (bits, lanes) in [(12, 1), (16, 12), (32, 15), (64, 256)] {
        let budget = Budget::new(64 << 20).unwrap();
        let retained: Vec<[PrefixFrame<'_>; 2]>;
        {
            let mut dealer = LocalDealer::new(source(), [3; 32], [4; 32], &budget).unwrap();
            let Fixture {
                mask,
                parties: [mut left, mut right],
                inputs: [x0, x1],
                output,
            } = fixture(&mut dealer, bits, lanes);
            let mut frames = vec![[left.begin(x0).unwrap(), right.begin(x1).unwrap()]];
            // TR1 -> TR2 -> DReLU -> Select: only frames may emerge until
            // the fourth peer exchange. Original X capabilities moved above.
            for exchange in 1..PREFIX_EXCHANGES {
                let [a, b] = frames.last().unwrap();
                match (
                    left.advance(b.as_bytes()).unwrap(),
                    right.advance(a.as_bytes()).unwrap(),
                ) {
                    (PrefixStep::Frame(a), PrefixStep::Frame(b)) => frames.push([a, b]),
                    _ => panic!("prefix output before exchange four: {exchange}"),
                }
            }
            let [a, b] = frames.last().unwrap();
            let [r0, r1] = match (
                left.advance(b.as_bytes()).unwrap(),
                right.advance(a.as_bytes()).unwrap(),
            ) {
                (PrefixStep::Output(a), PrefixStep::Output(b)) => [a, b],
                _ => panic!("prefix must finish on exchange four"),
            };
            assert_eq!(frames.len(), PREFIX_EXCHANGES);
            assert_eq!(frames.iter().flatten().count(), PREFIX_DIRECTIONAL_FRAMES);
            let mut directed_bytes = 0;
            for frame in frames.iter().flatten() {
                let expected = FRAME_HEADER_BYTES + FRAME_LANE_BYTES * lanes;
                assert_eq!(frame.as_bytes().len(), expected);
                directed_bytes += frame.as_bytes().len();
            }
            assert_eq!(directed_bytes, 8 * (76 + 8 * lanes));
            assert_eq!(r0.descriptor(), output);
            assert_eq!(r1.descriptor(), output);
            assert_eq!(left.advance(&[]).err(), Some(Error::Consumed));
            assert_eq!(right.advance(&[]).err(), Some(Error::Consumed));
            drop((left, right));

            // Consume the actual opaque R capabilities, not independently
            // encoded substitutes. No mask inspection or public decoding.
            let predicate = edge(10, 1, 0, lanes, true);
            let (mask, [mut left, mut right]) =
                dealer.drelu([9; 32], mask, predicate, [11; 32]).unwrap();
            let a = left.begin(r0).unwrap();
            let b = right.begin(r1).unwrap();
            let outputs = match (
                left.advance(b.as_bytes()).unwrap(),
                right.advance(a.as_bytes()).unwrap(),
            ) {
                (Step::Output(a), Step::Output(b)) => [a, b],
                _ => panic!("downstream DReLU must finish in one exchange"),
            };
            for value in &outputs {
                assert_eq!(value.descriptor(), predicate);
            }
            drop((mask, outputs, a, b, left, right, dealer));
            retained = frames;
        }
        // Completed owners do not refund separately retained frame leases.
        assert!(budget.used_bytes() > 0);
        drop(retained);
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn public_prefix_rejects_wrong_role_or_context_and_cancels_sibling() {
    for failure in 0..3 {
        let budget = Budget::new(64 << 20).unwrap();
        {
            let mut dealer = LocalDealer::new(source(), [3; 32], [4; 32], &budget).unwrap();
            let Fixture {
                mask,
                parties: [mut left, mut right],
                inputs: [x0, x1],
                ..
            } = fixture(&mut dealer, 16, 1);
            match failure {
                0 => {
                    assert_eq!(left.begin(x1).err(), Some(Error::Context));
                    drop(x0);
                }
                1 => {
                    // Same public declaration, independently issued root:
                    // metadata cannot counterfeit this region's input binding.
                    let (other_mask, [other0, other1]) = dealer
                        .prepare_root(
                            edge(5, 16, 12, 1, false),
                            BoundRange::new(-32768, 32768).unwrap(),
                            [7; 32],
                        )
                        .unwrap()
                        .encode(&[-32768])
                        .unwrap();
                    assert_eq!(left.begin(other0).err(), Some(Error::Context));
                    drop((other_mask, other1, x0, x1));
                }
                2 => {
                    let a = left.begin(x0).unwrap();
                    let b = right.begin(x1).unwrap();
                    // Reflection has the wrong peer role, even in this context.
                    assert_eq!(left.advance(a.as_bytes()).err(), Some(Error::Context));
                    drop((a, b));
                }
                _ => unreachable!(),
            }
            assert_eq!(left.advance(&[]).err(), Some(Error::Consumed));
            // Failure is local; the coordinator must explicitly retire sibling.
            right.cancel();
            assert_eq!(right.advance(&[]).err(), Some(Error::Consumed));
            drop((left, right, mask, dealer));
        }
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn public_prefix_cancel_keeps_retained_frames_charged_until_drop() {
    let budget = Budget::new(64 << 20).unwrap();
    let frames;
    {
        let mut dealer = LocalDealer::new(source(), [3; 32], [4; 32], &budget).unwrap();
        let Fixture {
            mask,
            parties: [mut left, mut right],
            inputs: [x0, x1],
            ..
        } = fixture(&mut dealer, 12, 1);
        frames = [left.begin(x0).unwrap(), right.begin(x1).unwrap()];
        left.cancel();
        right.cancel();
        // Cancellation is idempotent and terminal, including with live frames.
        left.cancel();
        assert_eq!(
            left.advance(frames[1].as_bytes()).err(),
            Some(Error::Consumed)
        );
        assert_eq!(
            right.advance(frames[0].as_bytes()).err(),
            Some(Error::Consumed)
        );
        drop((mask, left, right, dealer));
    }
    assert!(budget.used_bytes() > 0);
    drop(frames);
    assert_eq!(budget.used_bytes(), 0);
}

#[derive(Debug, PartialEq, Eq)]
struct PrefixOracle {
    y: i128,
    d: bool,
    r: i128,
    y_residue: u64,
    r_residue: u64,
}

// Use i128 throughout: 2^64 and both signed ring64 endpoints fit exactly.
// In particular, never construct a modulus with a u64 shift by 64.
fn modulus(bits: u8) -> i128 {
    assert!((1..=64).contains(&bits));
    1i128 << bits
}

fn signed(residue: u64, bits: u8) -> i128 {
    let modulus = modulus(bits);
    let residue = i128::from(residue);
    assert!(residue < modulus, "noncanonical ring residue");
    if residue >= modulus / 2 {
        residue - modulus
    } else {
        residue
    }
}

fn encode(value: i128, bits: u8) -> u64 {
    u64::try_from(value.rem_euclid(modulus(bits))).unwrap()
}

fn prefix_oracle(residue: u64, bits: u8) -> PrefixOracle {
    // Q12 requires at least twelve ring bits under the existing Numeric contract.
    assert!((12..=64).contains(&bits));
    let x = signed(residue, bits);
    let y = x.div_euclid(64);
    let d = y >= 0; // DReLU treats zero as nonnegative.
    let r = if d { y } else { 0 };
    PrefixOracle {
        y,
        d,
        r,
        y_residue: encode(y, bits - 6),
        r_residue: encode(r, bits - 6),
    }
}

// Fixed vectors, not bounds inferred from private values. Every member fits
// signed n >= 12, including the most negative member, -129.
const FLOOR_BOUNDARIES: &[(i128, i128, bool, i128)] = &[
    (-129, -3, false, 0),
    (-128, -2, false, 0),
    (-127, -2, false, 0),
    (-65, -2, false, 0),
    (-64, -1, false, 0),
    (-63, -1, false, 0),
    (-1, -1, false, 0),
    (0, 0, true, 0),
    (1, 0, true, 0),
    (63, 0, true, 0),
    (64, 1, true, 1),
    (65, 1, true, 1),
    (127, 1, true, 1),
    (128, 2, true, 2),
    (129, 2, true, 2),
];

#[test]
fn frozen_signed_floor_boundaries_hold_at_every_input_width() {
    for bits in 12..=64 {
        for &(x, y, d, r) in FLOOR_BOUNDARIES {
            let actual = prefix_oracle(encode(x, bits), bits);
            assert_eq!((actual.y, actual.d, actual.r), (y, d, r));
            assert_eq!(signed(actual.y_residue, bits - 6), y);
            assert_eq!(signed(actual.r_residue, bits - 6), r);
        }
    }
}

#[test]
fn q12_metadata_and_prefix_oracle_reject_widths_nine_through_eleven() {
    use pllm_core::edge_contract::{Interpretation, Numeric};

    for bits in 9..=11 {
        assert!(Numeric::new(bits, 12, Interpretation::Signed).is_err());
        assert!(std::panic::catch_unwind(|| prefix_oracle(0, bits)).is_err());
    }
    for bits in [12, 64] {
        assert!(Numeric::new(bits, 12, Interpretation::Signed).is_ok());
        assert!(Numeric::new(bits - 6, 6, Interpretation::Signed).is_ok());
    }
}

#[test]
fn ring12_frozen_residues_include_signed_endpoints() {
    let cases = [
        (0x800, -32, false, 0),
        (0x801, -32, false, 0),
        (0xfbf, -2, false, 0),
        (0xfc0, -1, false, 0),
        (0xfff, -1, false, 0),
        (0, 0, true, 0),
        (63, 0, true, 0),
        (64, 1, true, 1),
        (0x7ff, 31, true, 31),
    ];
    for (residue, y, d, r) in cases {
        let actual = prefix_oracle(residue, 12);
        assert_eq!((actual.y, actual.d, actual.r), (y, d, r));
        assert_eq!(signed(actual.y_residue, 6), y);
        assert_eq!(signed(actual.r_residue, 6), r);
    }
}

#[test]
fn ring64_frozen_residues_include_signed_minimum_without_abs_or_negation() {
    // Residues and expected floors are literals independent of encode/modulus.
    let cases = [
        (0x8000_0000_0000_0000, -144_115_188_075_855_872, false, 0),
        (0x8000_0000_0000_0001, -144_115_188_075_855_872, false, 0),
        (0xffff_ffff_ffff_ffbf, -2, false, 0),
        (0xffff_ffff_ffff_ffc0, -1, false, 0),
        (0xffff_ffff_ffff_ffc1, -1, false, 0),
        (0xffff_ffff_ffff_ffff, -1, false, 0),
        (0, 0, true, 0),
        (63, 0, true, 0),
        (64, 1, true, 1),
        (
            0x7fff_ffff_ffff_ffff,
            144_115_188_075_855_871,
            true,
            144_115_188_075_855_871,
        ),
    ];
    for (residue, y, d, r) in cases {
        let actual = prefix_oracle(residue, 64);
        assert_eq!((actual.y, actual.d, actual.r), (y, d, r));
        assert_eq!(signed(actual.y_residue, 58), y);
        assert_eq!(signed(actual.r_residue, 58), r);
    }
}

#[test]
fn exhaustive_small_rings_match_unsigned_tr_then_reduced_ring_sign() {
    for bits in 12..=14 {
        for residue in 0..u64::try_from(modulus(bits)).unwrap() {
            let actual = prefix_oracle(residue, bits);
            // Independent TR characterization: unsigned floor, then interpret
            // that residue in the reduced signed ring. No arithmetic owner used.
            let unsigned_tr = residue / 64;
            assert_eq!(actual.y_residue, unsigned_tr);
            assert_eq!(actual.y, signed(unsigned_tr, bits - 6));
            assert_eq!(actual.r, actual.y.max(0));
        }
    }
}

#[test]
fn every_width_and_lane_count_preserves_exact_floor_and_output_domain() {
    for bits in 12..=64 {
        let half = modulus(bits) / 2;
        let output_half = modulus(bits - 6) / 2;
        let endpoints = [-half, -half + 1, half - 2, half - 1];
        for lanes in 1..=256 {
            for lane in 0..lanes {
                let index = lane % (endpoints.len() + FLOOR_BOUNDARIES.len());
                let x = if index < endpoints.len() {
                    endpoints[index]
                } else {
                    FLOOR_BOUNDARIES[index - endpoints.len()].0
                };
                let actual = prefix_oracle(encode(x, bits), bits);
                // Multiplication by 64 fits i128 even at signed ring64 limits.
                assert!(actual.y * 64 <= x && x < (actual.y + 1) * 64);
                assert!((-output_half..output_half).contains(&actual.y));
                assert!((0..output_half).contains(&actual.r));
                assert_eq!(actual.d, actual.y >= 0);
                assert_eq!(actual.r, if actual.d { actual.y } else { 0 });
            }
        }
    }
}

#[test]
fn schedule_sensitive_controls_distinguish_floor_and_selected_operand() {
    let negative = prefix_oracle(encode(-1, 16), 16);
    assert_eq!(negative.y, -1);
    assert_ne!(negative.y, -1i128 / 64, "TR is floor, not truncation");
    assert_ne!(negative.y.abs(), (-1i128).abs() / 64);

    let positive_but_floored_to_zero = prefix_oracle(63, 16);
    assert!(positive_but_floored_to_zero.d, "zero is nonnegative");
    assert_eq!(positive_but_floored_to_zero.r, 0);
    assert_eq!(prefix_oracle(64, 16).r, 1, "Select consumes Y, not X");
    // Sign(X) equals sign(floor(X/64)); final values alone cannot prove
    // operation order. The consuming region fixture must verify its schedule.
}
