//! Literal numeric vectors and lifecycle definitions; no model/profile adoption.
use pllm_core::field_float::{
    mul_bf16_work, silu_bf16_work, swiglu_bf16_into, swiglu_bf16_work, Error,
};

#[test]
fn intermediate_bf16_cast_is_observable() {
    // SiLU(1) rounds to BF16 187/256 = 0x3f3b. Multiplying by 9 gives
    // exactly 1683/256 = 6.57421875, below midpoint 6.578125, hence 0x40d2.
    // Omitting that intermediate cast gives ~6.5795272, hence 0x40d3.
    let mut output = [0x5555; 3];
    let mut scratch = [0x1234; 125];
    assert_eq!(swiglu_bf16_into(&[0x3f80], &[0x4110], &mut output,
        &mut scratch, swiglu_bf16_work(1).unwrap()), Ok(()));
    assert_eq!(output, [0x40d2, 0x5555, 0x5555]);
    assert_eq!(&scratch[..122], &[0; 122]);
    assert_eq!(&scratch[122..], &[0x1234; 3]);
}

#[test]
fn literal_signed_zero_and_subnormal_vectors() {
    // Tiny SiLU rounds its input half: BF16 2 -> 1, while 1 -> even zero.
    // Signs survive the cast and combine by XOR in multiplication.
    let gate = [0, 0x8000, 0x8000, 2, 0x8002, 1, 0x8001];
    let up = [0xbf80, 0x3f80, 0xbf80, 0x3f80, 0x3f80, 0x7f7f, 0x3f80];
    let mut output = [0x5555; 8];
    let mut scratch = [0x1234; 123];
    assert_eq!(swiglu_bf16_into(&gate, &up, &mut output, &mut scratch,
        swiglu_bf16_work(7).unwrap()), Ok(()));
    assert_eq!(output, [0x8000, 0x8000, 0, 1, 0x8001, 0, 0x8000, 0x5555]);
    assert_eq!(&scratch[..122], &[0; 122]);
    assert_eq!(scratch[122], 0x1234);
}

#[test]
fn late_failure_clears_entire_active_output_not_tails() {
    for (gate, up) in [
        ([0x3f80, 0x7f7f, 0x3f80], [0x3f80, 0x4000, 0x3f80]),
        ([0x3f80, 0x7fc0, 0x3f80], [0x3f80; 3]),
        ([0x3f80; 3], [0x3f80, 0x7f80, 0x3f80]),
    ] {
        let mut output = [0x5555; 5];
        let mut scratch = [0x1234; 124];
        assert_eq!(swiglu_bf16_into(&gate, &up, &mut output, &mut scratch,
            usize::MAX), Err(Error::NonFinite));
        assert_eq!(output, [0, 0, 0, 0x5555, 0x5555]);
        assert_eq!(&scratch[..122], &[0; 122]);
        assert_eq!(&scratch[122..], &[0x1234; 2]);
    }
}

#[test]
fn first_nonfinite_gate_preserves_scratch_but_up_error_follows_silu_wipe() {
    for (gate, up, wiped) in [(0x7f80, 0x3f80, false), (0x3f80, 0x7fc0, true)] {
        let mut output = [0x5555; 2];
        let mut scratch = [0x1234; 123];
        assert_eq!(swiglu_bf16_into(&[gate], &[up], &mut output, &mut scratch,
            usize::MAX), Err(Error::NonFinite));
        assert_eq!(output, [0, 0x5555]);
        assert_eq!(&scratch[..122], &[if wiped { 0 } else { 0x1234 }; 122]);
        assert_eq!(scratch[122], 0x1234);
    }
}

#[test]
fn preflight_precedence_and_one_short_preserve_all_buffers() {
    let required = swiglu_bf16_work(1).unwrap();
    for (up_len, out_len, scratch_len, limit, error) in [
        (0, 0, 0, 0, Error::Length),
        (1, 0, 0, 0, Error::Capacity),
        (1, 2, 121, 0, Error::Capacity),
        (1, 2, 123, required - 1, Error::WorkLimit),
    ] {
        let mut output = [0x5555; 2];
        let mut scratch = [0x1234; 123];
        assert_eq!(swiglu_bf16_into(&[0x7fc0], &([0x7f80; 1])[..up_len],
            &mut output[..out_len], &mut scratch[..scratch_len], limit), Err(error));
        assert_eq!(output, [0x5555; 2]);
        assert_eq!(scratch, [0x1234; 123]);
    }
}

#[test]
fn empty_still_requires_scratch_and_budget_without_mutation() {
    let mut output = [0x5555; 2];
    let mut scratch = [0x1234; 123];
    let work = swiglu_bf16_work(0).unwrap();
    assert_eq!(swiglu_bf16_into(&[], &[], &mut output, &mut scratch[..121],
        usize::MAX), Err(Error::Capacity));
    assert_eq!(swiglu_bf16_into(&[], &[], &mut output, &mut scratch,
        work - 1), Err(Error::WorkLimit));
    assert_eq!(swiglu_bf16_into(&[], &[], &mut output, &mut scratch, work), Ok(()));
    assert_eq!(output, [0x5555; 2]);
    assert_eq!(scratch, [0x1234; 123]);
}

#[test]
fn checked_quote_composes_existing_allowances_and_rejects_overflow() {
    for width in [0, 1, 2, 17] {
        assert_eq!(swiglu_bf16_work(width),
            Ok(width * silu_bf16_work().unwrap() + mul_bf16_work(width).unwrap()));
    }
    assert_eq!(swiglu_bf16_work(usize::MAX), Err(Error::WorkLimit));
    assert_eq!(swiglu_bf16_work(usize::MAX / silu_bf16_work().unwrap() + 1),
        Err(Error::WorkLimit));
}
