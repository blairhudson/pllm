//! Source-authored clear numeric fixtures, not execution/qualification evidence.
//! Expected coefficients and replies below are literal dyadic/rational values;
//! modular sums use independently literal BabyBear arithmetic, never the new
//! implementation, `linear_bf16`, or `decode_reply_bf16` as an oracle.

use pllm_core::field_float::{
    import_bf16_q8_into,
    window_linear::{
        BandScratch, LinearSession, QuantizedInput, SourceWindow, WindowBuffers, WindowLinear,
    },
    Block128Geometry, Error,
};

const P: i128 = 2_013_265_921;
const FIELD_SENTINEL: u32 = 0xdead_beef;
const BF16_SENTINEL: u16 = 0xabcd;

fn window(row_start: usize, rows: usize, column_start: usize, columns: usize) -> SourceWindow {
    SourceWindow {
        row_start,
        rows,
        column_start,
        columns,
    }
}

fn modular_sum(weights: &[i64], input: &[i64]) -> u32 {
    assert_eq!(weights.len(), input.len());
    weights.iter().zip(input).fold(0_i128, |sum, (&w, &x)| {
        (sum + i128::from(w) * i128::from(x)).rem_euclid(P)
    }) as u32
}

struct Storage {
    input: Vec<u32>,
    sums: Vec<u32>,
    coefficients: Vec<i64>,
    field: Vec<u32>,
    bf16: Vec<u16>,
}

impl Storage {
    fn new(columns: usize, rows: usize, coefficients: usize) -> Self {
        Self {
            input: vec![91; columns],
            sums: vec![92; rows],
            coefficients: vec![93; coefficients],
            field: vec![FIELD_SENTINEL; rows],
            bf16: vec![BF16_SENTINEL; rows],
        }
    }

    fn owner(
        &mut self,
        geometry: Block128Geometry,
        row_start: usize,
        rows: usize,
        input: &[u16],
        work: usize,
    ) -> Result<WindowLinear<'_>, Error> {
        WindowLinear::new(
            geometry,
            row_start,
            rows,
            input,
            WindowBuffers {
                input: &mut self.input,
                sums: &mut self.sums,
                coefficients: &mut self.coefficients,
                field_output: &mut self.field,
                bf16_output: &mut self.bf16,
            },
            work,
        )
    }

    fn unpublished(&self) {
        assert!(self.field.iter().all(|&x| x == FIELD_SENTINEL));
        assert!(self.bf16.iter().all(|&x| x == BF16_SENTINEL));
    }

    fn scratch(&mut self) -> BandScratch<'_> {
        BandScratch {
            input: &mut self.input,
            sums: &mut self.sums,
            coefficients: &mut self.coefficients,
        }
    }

    fn cleared(&self) {
        assert!(self.input.iter().all(|&x| x == 0));
        assert!(self.sums.iter().all(|&x| x == 0));
    }
}

#[test]
fn global_scale_rectangle_crosses_both_axes_without_local_retiling() {
    let geometry = Block128Geometry::new(256, 256).unwrap();
    let region = window(127, 2, 127, 2);
    let mut output = [99; 5];
    geometry
        .import_q8_window(
            region,
            &[0x38; 4],
            &[0x3f80, 0x4000, 0x4040, 0x4080],
            &mut output,
            32,
        )
        .unwrap();
    // +1 * scales {1,2;3,4}, each multiplied by 256, exactly.
    assert_eq!(output, [256, 512, 768, 1024, 99]);
    // The same row/column origin lies inside a SINGLE global block.
    geometry
        .import_q8_window(
            window(129, 1, 129, 2),
            &[0x38, 0xb8],
            &[0x4080],
            &mut output,
            16,
        )
        .unwrap();
    assert_eq!(output, [1024, -1024, 768, 1024, 99]);
}

#[test]
fn fp8_subnormal_negative_and_q8_ties_have_literal_coefficients() {
    let geometry = Block128Geometry::new(128, 128).unwrap();
    let mut output = [99; 8];
    // FP8 subnormals are n/512: fixed-Q8 values n/2, ties to even.
    geometry
        .import_q8_window(
            window(0, 1, 0, 8),
            &[0x01, 0x03, 0x05, 0x81, 0x83, 0x85, 0x80, 0xb8],
            &[0x3f80],
            &mut output,
            64,
        )
        .unwrap();
    assert_eq!(output, [0, 2, 2, 0, -2, -2, 0, -256]);
    geometry
        .import_q8_window(window(0, 1, 0, 1), &[0x7e], &[1], &mut output, 8)
        .unwrap();
    assert_eq!(output[0], 0); // Positive BF16 subnormal scale, no FTZ policy.

    let mut head = [99; 7];
    import_bf16_q8_into(
        &[0x3b00, 0x3bc0, 0xbb00, 0xbbc0, 1, 0x8001, 0x8000],
        &mut head,
        14,
    )
    .unwrap();
    assert_eq!(head, [0, 2, 0, -2, 0, 0, 0]);
}

#[test]
fn import_preflight_and_numeric_errors_never_write() {
    let geometry = Block128Geometry::new(128, 128).unwrap();
    let region = window(0, 1, 0, 2);
    let mut output = [99; 2];
    assert_eq!(
        geometry.import_q8_window(region, &[0x38; 2], &[0x3f80], &mut output[..1], 16),
        Err(Error::Capacity)
    );
    // Capacity/work fail BEFORE inspecting even a NaN source.
    assert_eq!(
        geometry.import_q8_window(region, &[0x7f; 2], &[0], &mut output[..1], 16),
        Err(Error::Capacity)
    );
    assert_eq!(
        geometry.import_q8_window(region, &[0x7f; 2], &[0], &mut output, 15),
        Err(Error::WorkLimit)
    );
    for (bytes, scale, error) in [
        ([0x38, 0x7f], 0x3f80, Error::NonFinite),
        ([0x38, 0xff], 0x3f80, Error::NonFinite),
        ([0x38; 2], 0, Error::NonPositiveScale),
        ([0x38; 2], 0x8000, Error::NonPositiveScale),
        ([0x38; 2], 0xbf80, Error::NonPositiveScale),
        ([0x38; 2], 0x7fc1, Error::NonFinite),
        ([0x38; 2], 0x7f80, Error::NonFinite),
        ([0x7e; 2], 0x7f7f, Error::NonFinite),
        ([0x38; 2], 0x5b00, Error::IntegerRange),
    ] {
        assert_eq!(
            geometry.import_q8_window(region, &bytes, &[scale], &mut output, 16),
            Err(error)
        );
        assert_eq!(output, [99; 2]);
    }
    assert_eq!(
        geometry.import_q8_window(region, &[0x38], &[0x3f80], &mut output, 16),
        Err(Error::Length)
    );
    assert_eq!(
        geometry.import_q8_window(region, &[0x38; 2], &[], &mut output, 16),
        Err(Error::Length)
    );
    assert_eq!(
        geometry.import_q8_window(
            window(usize::MAX, 2, 0, 1),
            &[0x38; 2],
            &[0x3f80],
            &mut output,
            16
        ),
        Err(Error::Geometry)
    );
    assert_eq!(output, [99; 2]);

    let mut head = [99; 2];
    for (words, error) in [
        ([0x3f80, 0x7fc1], Error::NonFinite),
        ([0x3f80, 0x5b00], Error::IntegerRange),
    ] {
        assert_eq!(import_bf16_q8_into(&words, &mut head, 4), Err(error));
        assert_eq!(head, [99; 2]);
    }
    assert_eq!(
        import_bf16_q8_into(&[0x7fc1; 2], &mut head[..1], 4),
        Err(Error::Capacity)
    );
    assert_eq!(
        import_bf16_q8_into(&[0x7fc1; 2], &mut head, 3),
        Err(Error::WorkLimit)
    );
    assert_eq!(head, [99; 2]);
}

#[test]
fn complete_rows_preserve_global_offsets_and_both_scale_axes() {
    let geometry = Block128Geometry::new(256, 256).unwrap();
    let work = WindowLinear::required_work(geometry, 2).unwrap();
    let mut storage = Storage::new(256, 2, 2 * 127);
    let input = [0x3f80; 256];
    let mut owner = storage.owner(geometry, 127, 2, &input, work).unwrap();
    // Non-block-aligned partition includes both sides of the scale boundary.
    for (start, columns, scales) in [
        (0, 127, [0x3f80, 0x4040]),
        (127, 1, [0x3f80, 0x4040]),
        (128, 1, [0x4000, 0x4080]),
        (129, 127, [0x4000, 0x4080]),
    ] {
        owner = owner
            .append_fp8(
                window(127, 2, start, columns),
                &vec![0x38; 2 * columns],
                &scales,
            )
            .unwrap();
    }
    let result = owner.finish().unwrap();
    assert_eq!(result.row_start, 127);
    let row0: Vec<_> = (0..256).map(|c| if c < 128 { 256 } else { 512 }).collect();
    let row1: Vec<_> = (0..256).map(|c| if c < 128 { 768 } else { 1024 }).collect();
    assert_eq!(
        result.field,
        [
            modular_sum(&row0, &[256; 256]),
            modular_sum(&row1, &[256; 256])
        ]
    );
    assert_eq!(result.bf16, [0x43c0, 0x4460]); // Exact 384 and 896.
    storage.cleared();
    assert!(storage.coefficients.iter().all(|&x| x == 0));
}

#[test]
fn original_input_q8_tie_is_not_a_per_window_or_bf16_dot_substitute() {
    let geometry = Block128Geometry::new(128, 128).unwrap();
    let work = WindowLinear::required_work(geometry, 1).unwrap();
    let mut storage = Storage::new(128, 1, 127);
    // Input 3/512 -> Q8 2. Dot with 128 exact unit weights is 1, NOT 3/4.
    let owner = storage
        .owner(geometry, 0, 1, &[0x3bc0; 128], work)
        .unwrap()
        .append_bf16(window(0, 1, 0, 1), &[0x3f80])
        .unwrap()
        .append_bf16(window(0, 1, 1, 127), &[0x3f80; 127])
        .unwrap();
    let result = owner.finish().unwrap();
    assert_eq!(result.field, [modular_sum(&[256; 128], &[2; 128])]);
    assert_eq!(result.bf16, [0x3f80]);
}

#[test]
fn one_prepared_original_input_reuses_q8_across_row_bands() {
    let geometry = Block128Geometry::new(128, 128).unwrap();
    let work = LinearSession::required_work(geometry).unwrap();
    let mut prepared_scratch = [91; 129];
    let input = QuantizedInput::new(geometry, &[0x3bc0; 128], &mut prepared_scratch, 384).unwrap();
    let mut fields = [92; 128];
    let mut output = [FIELD_SENTINEL; 128];
    let mut bf16 = [BF16_SENTINEL; 128];
    let mut session =
        LinearSession::new(&input, &mut fields, &mut output, &mut bf16, work).unwrap();
    for row_start in 0..128 {
        let weight = if row_start == 127 { 0xbf80 } else { 0x3f80 };
        let mut storage = Storage::new(128, 1, 128);
        session = session
            .band(geometry, &input, row_start, 1, storage.scratch())
            .unwrap()
            .append_bf16(window(row_start, 1, 0, 63), &[weight; 63])
            .unwrap()
            .append_bf16(window(row_start, 1, 63, 65), &[weight; 65])
            .unwrap()
            .finish()
            .unwrap();
        storage.cleared();
        storage.unpublished();
    }
    let result = session.finish().unwrap();
    assert_eq!(result.row_start, 0);
    let expected = modular_sum(&[256; 128], &[2; 128]);
    assert_eq!(expected, 65_536);
    assert_eq!(result.field[..127], [expected; 127]);
    assert_eq!(result.field[127], (P - i128::from(expected)) as u32);
    assert_eq!(result.bf16[..127], [0x3f80; 127]);
    assert_eq!(result.bf16[127], 0xbf80);
    assert_eq!(fields, [0; 128]);
    drop(input);
    assert_eq!(prepared_scratch[..128], [0; 128]);
    assert_eq!(prepared_scratch[128], 91); // Spare storage is not charged/touched.
}

#[test]
fn prepared_input_preflight_and_failed_conversion_leave_no_partial_words() {
    let geometry = Block128Geometry::new(128, 128).unwrap();
    let mut scratch = [91; 128];
    assert_eq!(
        QuantizedInput::new(geometry, &[0x7fc1; 128], &mut scratch[..127], 384).err(),
        Some(Error::Capacity)
    );
    assert_eq!(scratch, [91; 128]);
    assert_eq!(
        QuantizedInput::new(geometry, &[0x7fc1; 128], &mut scratch, 383).err(),
        Some(Error::WorkLimit)
    );
    assert_eq!(scratch, [91; 128]);
    let mut invalid = [0x3f80; 128];
    invalid[127] = 0x7fc1;
    assert_eq!(
        QuantizedInput::new(geometry, &invalid, &mut scratch, 384).err(),
        Some(Error::NonFinite)
    );
    assert_eq!(scratch, [0; 128]);
    assert_eq!(
        QuantizedInput::new(geometry, &[], &mut scratch, 0).err(),
        Some(Error::Length)
    );
}

#[test]
fn session_rejects_same_columns_wrong_original_geometry_before_conversion() {
    let geometry = Block128Geometry::new(128, 128).unwrap();
    let work = LinearSession::required_work(geometry).unwrap();
    for wrong_rows in [256, 384] {
        let mut prepared = [91; 128];
        let input = QuantizedInput::new(geometry, &[0x3bc0; 128], &mut prepared, 384).unwrap();
        let mut fields = [92; 128];
        let mut output = [FIELD_SENTINEL; 128];
        let mut bf16 = [BF16_SENTINEL; 128];
        let mut scratch = Storage::new(128, 1, 128);
        let session = LinearSession::new(&input, &mut fields, &mut output, &mut bf16, work)
            .unwrap()
            .band(geometry, &input, 0, 1, scratch.scratch())
            .unwrap()
            .append_bf16(window(0, 1, 0, 128), &[0x3f80; 128])
            .unwrap()
            .finish()
            .unwrap();
        let wrong = Block128Geometry::new(wrong_rows, 128).unwrap();
        assert_eq!(
            session.band(wrong, &input, 1, 1, scratch.scratch()).err(),
            Some(Error::Geometry)
        );
        // The earlier completed row was held privately and erased on rejection.
        assert_eq!(fields, [0; 128]);
        assert_eq!(output, [FIELD_SENTINEL; 128]);
        assert_eq!(bf16, [BF16_SENTINEL; 128]);
        scratch.cleared();
        scratch.unpublished();
        assert_eq!(scratch.coefficients, [0; 128]);
        drop(input);
        assert_eq!(prepared, [0; 128]);
    }
}

#[test]
fn session_rejects_different_or_equal_but_distinct_input_cross_band() {
    let geometry = Block128Geometry::new(128, 128).unwrap();
    let work = LinearSession::required_work(geometry).unwrap();
    // Identity is borrowed ownership, not equality of words or a caller label.
    for other_word in [0x3bc0, 0x3f80] {
        let mut prepared = [91; 128];
        let mut other_prepared = [91; 128];
        let input = QuantizedInput::new(geometry, &[0x3bc0; 128], &mut prepared, 384).unwrap();
        let other =
            QuantizedInput::new(geometry, &[other_word; 128], &mut other_prepared, 384).unwrap();
        let mut fields = [92; 128];
        let mut output = [FIELD_SENTINEL; 128];
        let mut bf16 = [BF16_SENTINEL; 128];
        let mut scratch = Storage::new(128, 1, 128);
        let session = LinearSession::new(&input, &mut fields, &mut output, &mut bf16, work)
            .unwrap()
            .band(geometry, &input, 0, 1, scratch.scratch())
            .unwrap()
            .append_bf16(window(0, 1, 0, 128), &[0x3f80; 128])
            .unwrap()
            .finish()
            .unwrap();
        assert_eq!(
            session
                .band(geometry, &other, 1, 1, scratch.scratch())
                .err(),
            Some(Error::SessionMismatch)
        );
        assert_eq!(fields, [0; 128]);
        assert_eq!(output, [FIELD_SENTINEL; 128]);
        assert_eq!(bf16, [BF16_SENTINEL; 128]);
        scratch.cleared();
        assert_eq!(scratch.coefficients, [0; 128]);
        drop(input);
        drop(other);
        assert_eq!(prepared, [0; 128]);
        assert_eq!(other_prepared, [0; 128]);
    }
}

#[test]
fn session_incomplete_rows_columns_and_reordered_bands_publish_nothing() {
    let geometry = Block128Geometry::new(128, 128).unwrap();
    let work = LinearSession::required_work(geometry).unwrap();
    let mut prepared = [91; 128];
    let input = QuantizedInput::new(geometry, &[0x3bc0; 128], &mut prepared, 384).unwrap();
    for failure in 0..9 {
        let mut fields = [92; 128];
        let mut output = [FIELD_SENTINEL; 128];
        let mut bf16 = [BF16_SENTINEL; 128];
        let mut scratch = Storage::new(128, 1, 128);
        let session =
            LinearSession::new(&input, &mut fields, &mut output, &mut bf16, work).unwrap();
        if failure == 0 {
            assert_eq!(session.finish().err(), Some(Error::Incomplete));
        } else if failure == 1 {
            let band = session
                .band(geometry, &input, 0, 1, scratch.scratch())
                .unwrap()
                .append_bf16(window(0, 1, 0, 64), &[0x3f80; 64])
                .unwrap();
            assert_eq!(band.finish().err(), Some(Error::Incomplete));
            scratch.cleared();
        } else {
            let session = session
                .band(geometry, &input, 0, 1, scratch.scratch())
                .unwrap()
                .append_bf16(window(0, 1, 0, 128), &[0x3f80; 128])
                .unwrap()
                .finish()
                .unwrap();
            match failure {
                2 => assert_eq!(session.finish().err(), Some(Error::Incomplete)),
                3 => drop(session),
                4 => assert_eq!(
                    session
                        .band(geometry, &input, 0, 1, scratch.scratch())
                        .err(),
                    Some(Error::Coverage)
                ),
                5 => assert_eq!(
                    session
                        .band(geometry, &input, 2, 1, scratch.scratch())
                        .err(),
                    Some(Error::Coverage)
                ),
                6 => assert_eq!(
                    session
                        .band(geometry, &input, 1, 0, scratch.scratch())
                        .err(),
                    Some(Error::Geometry)
                ),
                7 => assert_eq!(
                    session
                        .band(geometry, &input, 1, 1, scratch.scratch())
                        .unwrap()
                        .append_bf16(window(1, 1, 0, 128), &[0x7fc1; 128])
                        .err(),
                    Some(Error::NonFinite)
                ),
                8 => assert_eq!(
                    session
                        .band(geometry, &input, 1, 1, scratch.scratch())
                        .unwrap()
                        .append_bf16(window(1, 1, 1, 127), &[0x3f80; 127])
                        .err(),
                    Some(Error::Coverage)
                ),
                _ => unreachable!(),
            }
            scratch.cleared();
        }
        assert_eq!(fields, [0; 128]);
        assert_eq!(output, [FIELD_SENTINEL; 128]);
        assert_eq!(bf16, [BF16_SENTINEL; 128]);
        scratch.unpublished();
    }
    drop(input);
    assert_eq!(prepared, [0; 128]);
}

#[test]
fn session_all_capacities_and_cumulative_work_preflight_before_writes() {
    let geometry = Block128Geometry::new(128, 128).unwrap();
    let work = LinearSession::required_work(geometry).unwrap();
    assert_eq!(work, 263_680); // Unchanged 16*rows*columns + 4*columns + 8*rows.
    let mut prepared = [91; 128];
    let input = QuantizedInput::new(geometry, &[0x3bc0; 128], &mut prepared, 384).unwrap();
    for failure in 0..4 {
        let mut fields = [92; 128];
        let mut output = [FIELD_SENTINEL; 128];
        let mut bf16 = [BF16_SENTINEL; 128];
        let error = LinearSession::new(
            &input,
            &mut fields[..if failure == 0 { 127 } else { 128 }],
            &mut output[..if failure == 1 { 127 } else { 128 }],
            &mut bf16[..if failure == 2 { 127 } else { 128 }],
            if failure == 3 { work - 1 } else { work },
        )
        .err();
        assert_eq!(
            error,
            Some(if failure == 3 {
                Error::WorkLimit
            } else {
                Error::Capacity
            })
        );
        assert_eq!(fields, [92; 128]);
        assert_eq!(output, [FIELD_SENTINEL; 128]);
        assert_eq!(bf16, [BF16_SENTINEL; 128]);
    }
    drop(input);
    assert_eq!(prepared, [0; 128]);
}

#[test]
fn centered_partial_crossing_and_complete_cancellation_stay_modular() {
    let geometry = Block128Geometry::new(128, 128).unwrap();
    let work = WindowLinear::required_work(geometry, 1).unwrap();
    let mut storage = Storage::new(128, 1, 126);
    assert!((1_i128 << 30) > (P - 1) / 2);
    let owner = storage
        .owner(geometry, 0, 1, &[0x3b80; 128], work)
        .unwrap()
        // Weight +2^22 -> Q8 +2^30, input 1/256 -> Q8 1.
        .append_bf16(window(0, 1, 0, 1), &[0x4a80])
        .unwrap()
        .append_bf16(window(0, 1, 1, 1), &[0xca80])
        .unwrap()
        .append_bf16(window(0, 1, 2, 126), &[0; 126])
        .unwrap();
    let result = owner.finish().unwrap();
    assert_eq!(result.field, [modular_sum(&[1 << 30, -(1 << 30)], &[1, 1])]);
    assert_eq!(result.bf16, [0]);
}

#[test]
fn complete_positive_and_negative_16842753_reject_double_rounding() {
    let geometry = Block128Geometry::new(128, 128).unwrap();
    let work = WindowLinear::required_work(geometry, 2).unwrap();
    let mut storage = Storage::new(128, 2, 252);
    let mut tail = [0_u16; 252];
    tail[0] = 0x3b80;
    tail[126] = 0xbb80;
    let owner = storage
        .owner(geometry, 0, 2, &[0x3b80; 128], work)
        .unwrap()
        .append_bf16(window(0, 2, 0, 1), &[0x4780, 0xc780])
        .unwrap()
        .append_bf16(window(0, 2, 1, 1), &[0x4380, 0xc380])
        .unwrap()
        .append_bf16(window(0, 2, 2, 126), &tail)
        .unwrap();
    let result = owner.finish().unwrap();
    let expected = modular_sum(&[16_777_216, 65_536, 1], &[1, 1, 1]);
    assert_eq!(expected, 16_842_753);
    assert_eq!(result.field, [expected, (P - i128::from(expected)) as u32]);
    // Literal exact rational RNE: ±258; the forbidden F32 bridge gives ±256.
    assert_eq!(result.bf16, [0x4381, 0xc381]);
}

#[test]
fn complete_reply_midpoints_round_to_even_on_both_signs() {
    let geometry = Block128Geometry::new(128, 128).unwrap();
    let work = WindowLinear::required_work(geometry, 4).unwrap();
    let mut storage = Storage::new(128, 4, 4 * 127);
    let mut tail = vec![0_u16; 4 * 127];
    for (row, word) in [0x4380, 0x4440, 0xc380, 0xc440].into_iter().enumerate() {
        tail[row * 127] = word;
    }
    let owner = storage
        .owner(geometry, 0, 4, &[0x3b80; 128], work)
        .unwrap()
        .append_bf16(window(0, 4, 0, 1), &[0x4780, 0x4780, 0xc780, 0xc780])
        .unwrap()
        .append_bf16(window(0, 4, 1, 127), &tail)
        .unwrap();
    let result = owner.finish().unwrap();
    let lower = modular_sum(&[16_777_216, 65_536], &[1, 1]);
    let upper = modular_sum(&[16_777_216, 196_608], &[1, 1]);
    assert_eq!(
        result.field,
        [
            lower,
            upper,
            (P - i128::from(lower)) as u32,
            (P - i128::from(upper)) as u32
        ]
    );
    // Exact ±257 and ±259: the adjacent even BF16 significands win.
    assert_eq!(result.bf16, [0x4380, 0x4382, 0xc380, 0xc382]);
}

#[test]
fn consuming_coverage_rejects_gap_duplicate_overlap_reorder_and_wrong_rows() {
    let geometry = Block128Geometry::new(128, 128).unwrap();
    let work = WindowLinear::required_work(geometry, 1).unwrap();
    for invalid in [
        window(0, 1, 65, 1), // Gap.
        window(0, 1, 0, 64), // Duplicate / reorder.
        window(0, 1, 63, 2), // Overlap.
        window(1, 1, 64, 1), // Wrong global rows.
        window(0, 2, 64, 1), // Wrong row extent.
    ] {
        let mut storage = Storage::new(128, 1, 64);
        let owner = storage
            .owner(geometry, 0, 1, &[0x3f80; 128], work)
            .unwrap()
            .append_bf16(window(0, 1, 0, 64), &[0x3f80; 64])
            .unwrap();
        assert_eq!(
            owner.append_bf16(invalid, &[0; 64]).err(),
            Some(Error::Coverage)
        );
        storage.unpublished();
        storage.cleared();
    }
    for invalid in [
        window(0, 1, 128, 1),
        window(0, 1, 0, 0),
        window(usize::MAX, 2, 0, 1),
    ] {
        let mut storage = Storage::new(128, 1, 128);
        assert_eq!(
            storage
                .owner(geometry, 0, 1, &[0; 128], work)
                .unwrap()
                .append_bf16(invalid, &[0])
                .err(),
            Some(Error::Geometry)
        );
        storage.unpublished();
        storage.cleared();
    }
}

#[test]
fn incomplete_abandoned_and_failed_owners_publish_nothing() {
    let geometry = Block128Geometry::new(128, 128).unwrap();
    let work = WindowLinear::required_work(geometry, 1).unwrap();
    for append in [false, true] {
        let mut storage = Storage::new(128, 1, 64);
        let mut owner = storage.owner(geometry, 0, 1, &[0x3f80; 128], work).unwrap();
        if append {
            owner = owner
                .append_bf16(window(0, 1, 0, 64), &[0x3f80; 64])
                .unwrap();
        }
        assert_eq!(owner.finish().err(), Some(Error::Incomplete));
        storage.unpublished();
        storage.cleared();
    }
    let mut storage = Storage::new(128, 1, 64);
    drop(
        storage
            .owner(geometry, 0, 1, &[0x3f80; 128], work)
            .unwrap()
            .append_bf16(window(0, 1, 0, 64), &[0x3f80; 64])
            .unwrap(),
    );
    storage.unpublished();
    storage.cleared();

    for fp8 in [false, true] {
        let mut storage = Storage::new(128, 1, 64);
        let owner = storage
            .owner(geometry, 0, 1, &[0x3f80; 128], work)
            .unwrap()
            .append_bf16(window(0, 1, 0, 64), &[0x3f80; 64])
            .unwrap();
        let error = if fp8 {
            let mut source = [0x38; 64];
            source[63] = 0x7f;
            owner
                .append_fp8(window(0, 1, 64, 64), &source, &[0x3f80])
                .err()
        } else {
            let mut source = [0x3f80; 64];
            source[63] = 0x7fc1;
            owner.append_bf16(window(0, 1, 64, 64), &source).err()
        };
        assert_eq!(error, Some(Error::NonFinite));
        storage.unpublished();
        storage.cleared();
        assert!(storage.coefficients.iter().all(|&x| x == 0));
    }
}

#[test]
fn every_capacity_is_preflighted_and_exact_fit_succeeds() {
    let geometry = Block128Geometry::new(128, 128).unwrap();
    let work = WindowLinear::required_work(geometry, 1).unwrap();
    for short in 0..5 {
        let mut storage = Storage::new(128, 1, 1);
        match short {
            0 => {
                storage.input.pop();
            }
            1 => {
                storage.sums.pop();
            }
            2 => {
                storage.coefficients.pop();
            }
            3 => {
                storage.field.pop();
            }
            _ => {
                storage.bf16.pop();
            }
        }
        assert_eq!(
            storage.owner(geometry, 0, 1, &[0x7fc1; 128], work).err(),
            Some(Error::Capacity)
        );
        storage.unpublished();
        assert!(storage.input.iter().all(|&x| x == 91));
        assert!(storage.sums.iter().all(|&x| x == 92));
    }
    // Construction fits, but a wider source window is one coefficient short.
    let mut storage = Storage::new(128, 1, 127);
    assert_eq!(
        storage
            .owner(geometry, 0, 1, &[0; 128], work)
            .unwrap()
            .append_bf16(window(0, 1, 0, 128), &[0x7fc1; 128])
            .err(),
        Some(Error::Capacity)
    );
    storage.unpublished();
    storage.cleared();

    let mut storage = Storage::new(128, 1, 1);
    let mut owner = storage.owner(geometry, 0, 1, &[0; 128], work).unwrap();
    for column in 0..128 {
        owner = owner
            .append_bf16(window(0, 1, column, 1), &[0x3f80])
            .unwrap();
    }
    let result = owner.finish().unwrap();
    assert_eq!(result.field, [0]);
    assert_eq!(result.bf16, [0]);
    storage.cleared();
}

#[test]
fn work_overflow_and_input_errors_fail_closed() {
    let geometry = Block128Geometry::new(128, 128).unwrap();
    let work = WindowLinear::required_work(geometry, 1).unwrap();
    let mut storage = Storage::new(128, 1, 1);
    assert_eq!(
        storage
            .owner(geometry, 0, 1, &[0x7fc1; 128], work - 1)
            .err(),
        Some(Error::WorkLimit)
    );
    storage.unpublished();
    assert!(storage.input.iter().all(|&x| x == 91));
    for (bad, error) in [(0x7fc1, Error::NonFinite), (0x5b00, Error::IntegerRange)] {
        let mut input = [0x3f80; 128];
        input[127] = bad;
        assert_eq!(
            storage.owner(geometry, 0, 1, &input, work).err(),
            Some(error)
        );
        storage.unpublished();
        storage.cleared();
    }
    assert_eq!(
        storage
            .owner(geometry, usize::MAX, 1, &[0; 128], work)
            .err(),
        Some(Error::Geometry)
    );
    assert_eq!(
        storage.owner(geometry, 0, 1, &[0; 127], work).err(),
        Some(Error::Length)
    );
    assert_eq!(
        WindowLinear::required_work(geometry, 0),
        Err(Error::Geometry)
    );
    let columns = (usize::MAX / 128 / 128) * 128;
    let enormous = Block128Geometry::new(128, columns).unwrap();
    assert_eq!(
        WindowLinear::required_work(enormous, 128),
        Err(Error::WorkLimit)
    );
    assert_eq!(
        LinearSession::required_work(enormous),
        Err(Error::WorkLimit)
    );
    assert_eq!(
        Block128Geometry::new(usize::MAX - 127, 128),
        Err(Error::Geometry)
    );
}
