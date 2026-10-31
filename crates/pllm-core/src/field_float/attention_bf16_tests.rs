//! Exact expectations authored independently of production helpers. The fixture
//! documents hand derivations and has a separate integer/Fraction Python oracle;
//! neither expected bits nor errors are obtained from Rust arithmetic helpers.
use super::{
    context_row_bf16_into, context_row_bf16_work, geometry, score_row_bf16_into,
    score_row_bf16_work, Error, FINITE_R32_WORK,
};
use serde_json::Value;

const SENTINEL: u16 = 0x5555;

fn word(value: &Value) -> u32 {
    u32::from_str_radix(value.as_str().unwrap(), 16).unwrap()
}

fn words(case: &Value, name: &str) -> Vec<u16> {
    case[name].as_array().unwrap().iter()
        .map(|value| u16::try_from(word(value)).unwrap()).collect()
}

#[test]
fn independent_exact_fixture_bits_errors_and_spare_tail() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/field_float_attention.json"
    )).unwrap();
    assert_eq!(fixture["case_count"].as_u64(), Some(27));
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 27);
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (count, result, output) = match case["kind"].as_str().unwrap() {
            "score" => {
                let query = words(case, "query");
                let keys = words(case, "keys");
                let count = keys.len() / query.len();
                let mut output = vec![SENTINEL; count + 2];
                let work = score_row_bf16_work(count, query.len()).unwrap();
                let result = score_row_bf16_into(&query, &keys, word(&case["alpha"]), &mut output, work);
                (count, result, output)
            }
            "context" => {
                let probabilities = words(case, "probabilities");
                let values = words(case, "values");
                let count = values.len() / probabilities.len();
                let mut output = vec![SENTINEL; count + 2];
                let work = context_row_bf16_work(probabilities.len(), count).unwrap();
                let result = context_row_bf16_into(&probabilities, &values, &mut output, work);
                (count, result, output)
            }
            other => panic!("unknown fixture kind {other}"),
        };
        if let Some(error) = case.get("error") {
            assert_eq!(error.as_str(), Some("NonFinite"), "{name}");
            assert_eq!(result, Err(Error::NonFinite), "{name}");
            assert_eq!(&output[..count], vec![0; count], "{name}");
        } else {
            assert_eq!(result, Ok(()), "{name}");
            assert_eq!(&output[..count], words(case, "output"), "{name}");
        }
        assert_eq!(&output[count..], &[SENTINEL; 2], "{name}");
    }
}

#[test]
fn three_term_nonfused_witness_rejects_fma_mutation() {
    // Independent integer equations on the exact 2^-150 grid. F32 subnormal
    // quantum is 2 grid units; BF16 quantum is 2^17. No production arithmetic
    // helper and no host float/FMA participates in deriving expected results.
    let round_even_units = |magnitude: u64, quantum: u64| {
        let quotient = magnitude / quantum;
        let remainder = magnitude % quantum;
        quotient + u64::from(remainder > quantum / 2
            || (remainder == quantum / 2 && quotient % 2 == 1))
    };
    // Exact products: +65536, +2, -1 grid units. First two are exact F32.
    let before_last = 65536_u64 + 2;
    let separate_last_product_magnitude = round_even_units(1, 2);
    assert_eq!(separate_last_product_magnitude, 0); // Negative product -> -0.
    let nonfused_f32_units = round_even_units(
        before_last - 2 * separate_last_product_magnitude, 2,
    );
    let fused_f32_units = round_even_units(before_last - 1, 2);
    assert_eq!(nonfused_f32_units, 32769);
    assert_eq!(fused_f32_units, 32768); // Tie: 32768.5 -> even 32768.
    let nonfused_bf16 = round_even_units(2 * nonfused_f32_units, 1 << 17);
    let fused_bf16 = round_even_units(2 * fused_f32_units, 1 << 17);
    assert_eq!((nonfused_bf16, fused_bf16), (1, 0));

    let mut output = [SENTINEL; 2];
    let work = context_row_bf16_work(3, 1).unwrap();
    context_row_bf16_into(
        &[0x0001; 3], &[0x3f00, 0x3780, 0xb700], &mut output, work,
    ).unwrap();
    assert_eq!(output, [0x0001, SENTINEL]);
    assert_eq!(u64::from(output[0]), nonfused_bf16);
    assert_ne!(u64::from(output[0]), fused_bf16);
}

#[test]
fn checked_work_counts_every_primitive_and_cleanup_without_saturating() {
    for rows in [1, 2, 8, 128] {
        for width in [1, 2, 128] {
            // Reuse local-dot quote per output, plus scale only for scores.
            let score = rows * (width * (2 * FINITE_R32_WORK + 2048)
                + 4096 + FINITE_R32_WORK + 2048) + 4096;
            let context = width * (rows * (2 * FINITE_R32_WORK + 2048)
                + 4096 + 2048) + 4096;
            assert_eq!(score_row_bf16_work(rows, width), Ok(score));
            assert_eq!(context_row_bf16_work(rows, width), Ok(context));
        }
    }
    for (rows, width) in [(0, 0), (0, 1), (1, 0)] {
        assert_eq!(score_row_bf16_work(rows, width), Err(Error::Geometry));
        assert_eq!(context_row_bf16_work(rows, width), Err(Error::Geometry));
    }
    // Exercise geometry overflow separately from primitive/OTHER/outer quotes,
    // without allocating enormous slices or inventing invalid slice pointers.
    assert_eq!(geometry(usize::MAX, 2), Err(Error::WorkLimit));
    assert_eq!(geometry(2, usize::MAX), Err(Error::WorkLimit));
    assert_eq!(geometry(usize::MAX, 1), Ok(usize::MAX));
    for (rows, width) in [(usize::MAX, 1), (1, usize::MAX), (usize::MAX, 2), (2, usize::MAX)] {
        assert_eq!(score_row_bf16_work(rows, width), Err(Error::WorkLimit));
        assert_eq!(context_row_bf16_work(rows, width), Err(Error::WorkLimit));
    }
}

#[test]
fn score_preflight_is_element_blind_and_atomic() {
    let mut output = [SENTINEL; 3];
    assert_eq!(score_row_bf16_into(&[], &[0x7fc1], 0x7f80_0000, &mut output, usize::MAX), Err(Error::Geometry));
    assert_eq!(score_row_bf16_into(&[0x7fc1], &[], 0, &mut output, usize::MAX), Err(Error::Geometry));
    assert_eq!(score_row_bf16_into(&[0x7fc1; 2], &[0; 3], 0, &mut output, usize::MAX), Err(Error::Length));
    assert_eq!(score_row_bf16_into(&[0x7fc1], &[0; 4], 0, &mut output, usize::MAX), Err(Error::Capacity));
    let work = score_row_bf16_work(2, 1).unwrap();
    assert_eq!(score_row_bf16_into(&[0x7fc1], &[0; 2], 0x7f80_0000, &mut output, work - 1), Err(Error::WorkLimit));
    assert_eq!(output, [SENTINEL; 3]);
    let mut short = [SENTINEL; 1];
    assert_eq!(score_row_bf16_into(&[0x7fc1], &[0; 2], 0, &mut short, work), Err(Error::Capacity));
    assert_eq!(short, [SENTINEL]);
}

#[test]
fn context_preflight_is_element_blind_and_atomic() {
    let mut output = [SENTINEL; 3];
    assert_eq!(context_row_bf16_into(&[], &[0x7fc1], &mut output, usize::MAX), Err(Error::Geometry));
    assert_eq!(context_row_bf16_into(&[0x7fc1], &[], &mut output, usize::MAX), Err(Error::Geometry));
    assert_eq!(context_row_bf16_into(&[0x7fc1; 2], &[0; 3], &mut output, usize::MAX), Err(Error::Length));
    assert_eq!(context_row_bf16_into(&[0x7fc1], &[0; 4], &mut output, usize::MAX), Err(Error::Capacity));
    let work = context_row_bf16_work(1, 2).unwrap();
    assert_eq!(context_row_bf16_into(&[0x7fc1], &[0; 2], &mut output, work - 1), Err(Error::WorkLimit));
    assert_eq!(output, [SENTINEL; 3]);
    let mut short = [SENTINEL; 1];
    assert_eq!(context_row_bf16_into(&[0x7fc1], &[0; 2], &mut short, work), Err(Error::Capacity));
    assert_eq!(short, [SENTINEL]);
}

#[test]
fn exact_work_and_capacity_succeed_one_short_work_never_writes() {
    let mut scores = [SENTINEL; 2];
    let score_work = score_row_bf16_work(2, 1).unwrap();
    let mut context = [SENTINEL; 2];
    let context_work = context_row_bf16_work(1, 2).unwrap();
    assert_eq!(score_row_bf16_into(&[0x3f80], &[0x4000, 0x4040], 0x3f80_0000, &mut scores, score_work - 1), Err(Error::WorkLimit));
    assert_eq!(context_row_bf16_into(&[0x3f80], &[0x4000, 0x4040], &mut context, context_work - 1), Err(Error::WorkLimit));
    assert_eq!(scores, [SENTINEL; 2]);
    assert_eq!(context, [SENTINEL; 2]);
    score_row_bf16_into(&[0x3f80], &[0x4000, 0x4040], 0x3f80_0000, &mut scores, score_work).unwrap();
    context_row_bf16_into(&[0x3f80], &[0x4000, 0x4040], &mut context, context_work).unwrap();
    assert_eq!(scores, [0x4000, 0x4040]);
    assert_eq!(context, [0x4000, 0x4040]);
}

#[test]
fn score_late_failures_clear_written_and_unwritten_active_words_only() {
    for (query, keys, alpha) in [
        // First row succeeds, second fails; third remains unwritten.
        ([0x3f80, 0x3f80], [0x3f80, 0, 0x7fc1, 0, 0, 0], 0x3f80_0000),
        ([0x7f7f, 0x7b00], [0, 0, 0x4000, 0, 0, 0], 0x3f80_0000), // RN32 product overflow
        ([0x7f7f, 0x7b00], [0, 0, 0x3f80, 0x3f80, 0, 0], 0), // first R16 overflow despite alpha=0
        ([0x7f7f, 0], [0, 0, 0x3f80, 0, 0, 0], 0x3f80_8000), // final R16 overflow
        ([0x7f7f, 0], [0, 0, 0x3f80, 0, 0, 0], 0x4000_0000), // alpha RN32 overflow
    ] {
        let mut output = [SENTINEL; 5];
        assert_eq!(score_row_bf16_into(&query, &keys, alpha, &mut output, usize::MAX), Err(Error::NonFinite));
        assert_eq!(output, [0, 0, 0, SENTINEL, SENTINEL]);
    }
}

#[test]
fn context_late_failures_clear_written_and_unwritten_active_words_only() {
    for values in [
        [0x3f80, 0x7fc1, 0, 0x3f80, 0, 0],
        [0x3f80, 0x7f7f, 0, 0x3f80, 0x7f7f, 0], // RN32 sum overflow
        [0x3f80, 0x7f7f, 0, 0x3f80, 0x7b00, 0], // R16-only overflow
    ] {
        let mut output = [SENTINEL; 5];
        assert_eq!(context_row_bf16_into(&[0x3f80; 2], &values, &mut output, usize::MAX), Err(Error::NonFinite));
        assert_eq!(output, [0, 0, 0, SENTINEL, SENTINEL]);
    }
}

#[test]
fn all_nonfinite_word_classes_refuse_even_when_other_operand_is_zero() {
    for bad in [0x7f80, 0xff80, 0x7f81, 0xff81, 0x7fc0, 0xffc1] {
        for query_bad in [false, true] {
            let mut output = [SENTINEL; 2];
            let query = [if query_bad { bad } else { 0 }];
            let keys = [if query_bad { 0 } else { bad }];
            assert_eq!(score_row_bf16_into(&query, &keys, 0, &mut output, usize::MAX), Err(Error::NonFinite));
            assert_eq!(output, [0, SENTINEL]);
            let mut output = [SENTINEL; 2];
            assert_eq!(context_row_bf16_into(&query, &keys, &mut output, usize::MAX), Err(Error::NonFinite));
            assert_eq!(output, [0, SENTINEL]);
        }
    }
    for alpha in [0x7f80_0000, 0xff80_0000, 0x7f80_0001, 0xff80_0001, 0x7fc0_0000, 0xffc0_0001] {
        let mut output = [SENTINEL; 4];
        assert_eq!(score_row_bf16_into(&[0], &[0; 2], alpha, &mut output, usize::MAX), Err(Error::NonFinite));
        assert_eq!(output, [0, 0, SENTINEL, SENTINEL]);
    }
}
