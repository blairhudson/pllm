//! Shared masked-floor equations. No randomness, allocation or numeric rewrite.

pub(super) fn adjusted(input: u64, bits: u8, gap: bool) -> u64 {
    input.wrapping_add(if gap { 1 << (bits - 2) } else { 0 }) & super::mask(bits)
}

pub(super) fn finish(
    party: u8,
    input: u64,
    bits: u8,
    shift: u8,
    gap: bool,
    constants: &[u64; 3],
    opened_bit: u64,
) -> u64 {
    let input = adjusted(input, bits, gap);
    let carry = u64::from(party)
        .wrapping_mul(opened_bit)
        .wrapping_add(constants[0])
        .wrapping_sub(2u64.wrapping_mul(opened_bit).wrapping_mul(constants[0]));
    let wrap = constants[1].wrapping_mul(1 - (input >> (bits - 1)));
    u64::from(party)
        .wrapping_mul(input >> shift)
        .wrapping_add(constants[2])
        .wrapping_add(wrap)
        .wrapping_sub(carry)
        & super::mask(if gap { bits } else { bits - shift })
}
