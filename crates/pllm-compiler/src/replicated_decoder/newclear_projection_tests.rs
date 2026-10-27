//! Synthetic storage only; these fixtures authenticate neither publisher nor
//! raw production format/input. Tests exercise the real LinearSession adapter.
use super::*;

const P: i128 = 2_013_265_921;
const R: usize = 128;
const C: usize = 256;

fn plan() -> Vec<SourceWindow> {
    let mut result = Vec::new();
    for (row_start, rows) in [(0, 65), (65, 63)] {
        for (column_start, columns) in [(0, 127), (127, 2), (129, 127)] {
            result.push(SourceWindow { row_start, rows, column_start, columns });
        }
    }
    result
}
fn source() -> FixtureProjectionSource {
    let choices = [0x01, 0x08, 0x30, 0xb8, 0x7e, 0x80, 0x3f];
    let bytes: Vec<_> = (0..R*C).map(|i| choices[(i + i/C) % choices.len()]).collect();
    FixtureProjectionSource::fp8((7, 8), [R, C], &bytes, &[0x3f80, 0x3f40]).unwrap()
}
fn input_words() -> Vec<u16> {
    (0..C).map(|i| [0x3b80, 0x3f00, 0xbf80, 0x3f81][i%4]).collect()
}
struct Storage {
    input: Vec<u32>, sums: Vec<u32>, coefficients: Vec<i64>, session: Vec<u32>,
    field: Vec<u32>, bf16: Vec<u16>,
}
impl Storage {
    fn new(rows: usize, columns: usize, max_rows: usize, max_columns: usize) -> Self {
        Self { input: vec![19; columns+1], sums: vec![19; max_rows+1],
            coefficients: vec![19; max_rows*max_columns+1], session: vec![19; rows+1],
            field: vec![23; rows+1], bf16: vec![29; rows+1] }
    }
    fn buffers(&mut self) -> ClearBuffers<'_> {
        ClearBuffers { band_input: &mut self.input, band_sums: &mut self.sums,
            coefficients: &mut self.coefficients, session_sums: &mut self.session,
            field_output: &mut self.field, bf16_output: &mut self.bf16 }
    }
    fn untouched(&self) {
        assert!(self.field.iter().all(|&x| x == 23));
        assert!(self.bf16.iter().all(|&x| x == 29));
    }
    fn cleaned(&self) {
        assert!(self.input.iter().all(|&x| x == 0));
        assert!(self.sums.iter().all(|&x| x == 0));
        assert!(self.coefficients.iter().all(|&x| x == 0));
        assert!(self.session.iter().all(|&x| x == 0));
    }
}

// Independent exact dyadic/rational oracle: no core decode, quantize, field
// multiply or reply conversion supplies expected values.
fn round_ratio(n: i128, d: i128) -> i128 {
    assert!(n >= 0 && d > 0);
    let q = n/d; let r = n%d;
    q + i128::from(2*r > d || (2*r == d && q%2 != 0))
}
fn scaled_integer(n: i128, power: i32) -> i128 {
    if power >= 0 { n * (1i128 << power) }
    else if power < -120 { 0 }
    else { round_ratio(n, 1i128 << -power) }
}
fn bf16_parts(word: u16) -> (i128, i32) {
    let e = (word >> 7) & 255;
    assert_ne!(e, 255);
    let sign = if word & 0x8000 == 0 { 1 } else { -1 };
    if e == 0 { (sign*i128::from(word&127), -133) }
    else { (sign*(128+i128::from(word&127)), i32::from(e)-134) }
}
fn bf16_q8(word: u16) -> i128 {
    let (n,e) = bf16_parts(word);
    n.signum()*scaled_integer(n.abs(), e+8)
}
fn fp8_q8(byte: u8, scale: u16) -> i128 {
    let e = (byte >> 3)&15; let m = byte&7;
    assert!(!(e == 15 && m == 7));
    let (sn, se) = bf16_parts(scale);
    assert!(sn > 0);
    let (n,power) = if e == 0 { (i128::from(m), -9) }
        else { (8+i128::from(m), i32::from(e)-10) };
    let sign = if byte&128 == 0 { 1 } else { -1 };
    sign*scaled_integer(n*sn, power+se+8)
}
fn reply(residue: u32) -> u16 {
    let n = if i128::from(residue) <= (P-1)/2 { i128::from(residue) } else { i128::from(residue)-P };
    if n == 0 { return 0; }
    let magnitude = n.abs();
    let mut high = 0u32;
    while (1i128 << (high+1)) <= magnitude { high += 1; }
    let mut mantissa = if high > 7 { round_ratio(magnitude, 1i128 << (high-7)) }
        else { magnitude << (7-high) };
    let mut exponent = high as i32 - 16 + 127;
    if mantissa == 256 { mantissa = 128; exponent += 1; }
    (if n < 0 { 0x8000 } else { 0 }) | ((exponent as u16)<<7) | ((mantissa-128) as u16)
}
fn reference(source: &FixtureProjectionSource, input: &[u16]) -> (Vec<u32>, Vec<u16>) {
    let mut fields = Vec::new(); let mut words = Vec::new();
    let c = source.original.columns;
    for row in 0..source.original.rows {
        let mut sum = 0i128;
        for column in 0..c {
            let lane = row*c+column;
            let weight = match source.format {
                ClearFormat::E4m3fnBlock128 => {
                    let cell = (row/128)*(c/128)+column/128;
                    let scales = &source.scales.as_ref().unwrap().bytes;
                    let scale = u16::from_le_bytes([scales[cell*2], scales[cell*2+1]]);
                    fp8_q8(source.weights.bytes[lane], scale)
                }
                ClearFormat::Bf16 => bf16_q8(u16::from_le_bytes([
                    source.weights.bytes[lane*2], source.weights.bytes[lane*2+1]])),
            };
            sum = (sum + weight*bf16_q8(input[column])).rem_euclid(P);
        }
        fields.push(sum as u32); words.push(reply(sum as u32));
    }
    (fields, words)
}

#[test]
fn fp8_original_block_boundary_partial_bands_use_independent_rational_reference() {
    let source = source(); let words = input_words(); let partition = plan();
    let expected = reference(&source, &words);
    let ledger = ClearLedger::new(usize::MAX, usize::MAX);
    let mut input_scratch = vec![41; C];
    let input = ClearInput::new(&source, &words, &mut input_scratch, &ledger).unwrap();
    let mut storage = Storage::new(R,C,65,127);
    let projection = ClearProjection::bind(&source, &input, &partition).unwrap();
    let produced = projection.run(&source, &input, storage.buffers(), &ledger).unwrap();
    assert_eq!(produced.field(), expected.0); assert_eq!(produced.bf16(), expected.1);
    assert_eq!(source.weights.reads.get(), 3 * R);
    assert_eq!(source.scales.as_ref().unwrap().reads.get(), 8);
    drop(produced); storage.cleaned();
    assert_eq!(storage.field[R], 23); assert_eq!(storage.bf16[R], 29);
    drop(input); assert!(input_scratch.iter().all(|&x| x == 0));
    assert_eq!(ledger.resident.get(), 0);
}

#[test]
fn bf16_head_direct_centered_rounding_not_f32_double_rounding() {
    let mut weights = vec![0; 128*128];
    weights[..3].copy_from_slice(&[0x4380, 0x4000, 0x3b80]);
    let source = FixtureProjectionSource::bf16(7, [128,128], &weights).unwrap();
    let mut words = vec![0;128]; words[..3].copy_from_slice(&[0x4000,0x3f80,0x3b80]);
    let partition = vec![SourceWindow { row_start:0, rows:128, column_start:0, columns:128 }];
    let expected = reference(&source, &words);
    assert_eq!(expected.0[0], 33_685_505); assert_eq!(expected.1[0], 0x4401);
    assert_eq!(clear::f32_to_bf16((33_685_505_f64/65536.0) as f32).unwrap(), 0x4400);
    let ledger = ClearLedger::new(usize::MAX, usize::MAX);
    let mut scratch = vec![0;128];
    let input = ClearInput::new(&source,&words,&mut scratch,&ledger).unwrap();
    let mut storage = Storage::new(128,128,128,128);
    let output = ClearProjection::bind(&source,&input,&partition).unwrap()
        .run(&source,&input,storage.buffers(),&ledger).unwrap();
    assert_eq!(output.field(), expected.0); assert_eq!(output.bf16(), expected.1);
    drop(output); storage.cleaned();
}

#[test]
fn foreign_same_labels_content_and_prepared_input_rejected_before_reads() {
    let source = source(); let foreign = self::source();
    assert_eq!(source.tensor_id, foreign.tensor_id);
    assert_eq!(source.weights.bytes, foreign.weights.bytes);
    let words = input_words(); let partition = plan();
    let ledger = ClearLedger::new(usize::MAX,usize::MAX);
    let mut a = vec![0;C]; let mut b = vec![0;C]; let mut c = vec![0;C];
    let input = ClearInput::new(&source,&words,&mut a,&ledger).unwrap();
    let equal = ClearInput::new(&source,&words,&mut b,&ledger).unwrap();
    let foreign_input = ClearInput::new(&foreign,&words,&mut c,&ledger).unwrap();
    assert!(ClearProjection::bind(&foreign,&input,&partition).is_err());
    assert!(ClearProjection::bind(&source,&foreign_input,&partition).is_err());
    let mut storage = Storage::new(R,C,65,127);
    let p = ClearProjection::bind(&source,&input,&partition).unwrap();
    assert!(p.run(&source,&equal,storage.buffers(),&ledger).is_err());
    let p = ClearProjection::bind(&source,&input,&partition).unwrap();
    assert!(p.run(&foreign,&input,storage.buffers(),&ledger).is_err());
    assert_eq!(source.weights.reads.get(),0); assert_eq!(foreign.weights.reads.get(),0);
    storage.untouched();
}

#[test]
fn partitions_reject_gap_duplicate_reorder_overlap_incomplete_and_changed_band() {
    let source = source(); let words = input_words();
    let ledger = ClearLedger::new(usize::MAX,usize::MAX); let mut scratch = vec![0;C];
    let input = ClearInput::new(&source,&words,&mut scratch,&ledger).unwrap();
    let mut cases = Vec::new();
    let mut p = plan(); p[1].column_start += 1; cases.push(p);
    let mut p = plan(); p.insert(1,p[0]); cases.push(p);
    let mut p = plan(); p.swap(0,1); cases.push(p);
    let mut p = plan(); p[1].column_start -= 1; cases.push(p);
    let mut p = plan(); p.pop(); cases.push(p);
    let mut p = plan(); p[1].rows -= 1; cases.push(p);
    let mut p = plan(); p[3].row_start += 1; cases.push(p);
    let mut p = plan(); p[0].columns = usize::MAX; cases.push(p);
    for p in cases { assert!(ClearProjection::bind(&source,&input,&p).is_err()); }
    assert_eq!(source.weights.reads.get(),0);
}

#[test]
fn original_full_shape_not_partial_band_and_format_lengths_axes_are_checked() {
    for shape in [[65,256],[128,255],[0,128],[usize::MAX,128],[usize::MAX-127,128]] {
        assert!(FixtureProjectionSource::fp8((1,2),shape,&[],&[]).is_err());
    }
    assert!(FixtureProjectionSource::fp8((1,2),[R,C],&vec![0;R*C-1],&[0x3f80;2]).is_err());
    assert!(FixtureProjectionSource::fp8((1,2),[R,C],&vec![0;R*C],&[0x3f80;3]).is_err());
    assert!(FixtureProjectionSource::bf16(1,[128,128],&[0;127]).is_err());
    // Bad scale axes: extra row cells cannot be treated as local tiled scales.
    let mut source = source(); source.scales.as_mut().unwrap().bytes.extend_from_slice(&[0,0]);
    let ledger = ClearLedger::new(usize::MAX,usize::MAX); let mut scratch = vec![0;C];
    let words = input_words(); let input = ClearInput::new(&source,&words,&mut scratch,&ledger).unwrap();
    assert!(ClearProjection::bind(&source,&input,&plan()).is_err());
    assert_eq!(source.weights.reads.get(),0);
}

#[test]
fn late_invalid_coefficients_and_scale_values_never_publish_and_clear_scratch() {
    for bad_scale in [0,0x8000,0xbf80,0x7f80,0x7fc1,0x7f7f] {
        let source = FixtureProjectionSource::fp8((1,2),[R,C],&vec![0x38;R*C],&[0x3f80,bad_scale]).unwrap();
        failure_cleanup(&source);
    }
    let mut source = source(); source.weights.bytes[R*C-1] = 0x7f;
    failure_cleanup(&source);
    let mut weights = vec![0x3f80;128*128]; weights[128*128-1] = 0x7f80;
    let source = FixtureProjectionSource::bf16(1,[128,128],&weights).unwrap();
    let words = vec![0x3f80;128]; let partition = vec![SourceWindow { row_start:0,rows:128,column_start:0,columns:128 }];
    let ledger = ClearLedger::new(usize::MAX,usize::MAX); let mut scratch = vec![0;128];
    let input = ClearInput::new(&source,&words,&mut scratch,&ledger).unwrap();
    let before = ledger.resident.get(); let mut storage = Storage::new(128,128,128,128);
    assert!(ClearProjection::bind(&source,&input,&partition).unwrap().run(&source,&input,storage.buffers(),&ledger).is_err());
    storage.untouched(); storage.cleaned(); assert_eq!(ledger.resident.get(), before);
}
fn failure_cleanup(source: &FixtureProjectionSource) {
    let ledger = ClearLedger::new(usize::MAX,usize::MAX); let mut scratch = vec![0;C];
    let words = input_words(); let partition = plan();
    let input = ClearInput::new(source,&words,&mut scratch,&ledger).unwrap();
    let retained = ledger.resident.get(); let mut storage = Storage::new(R,C,65,127);
    assert!(ClearProjection::bind(source,&input,&partition).unwrap().run(source,&input,storage.buffers(),&ledger).is_err());
    storage.untouched(); storage.cleaned(); assert_eq!(ledger.resident.get(), retained);
    // Drop-time hooks assert both read and external scratch were cleared BEFORE
    // either execution or binding reservation could release resident admission.
    assert!(!ledger.pending_scratch.get()); assert!(!ledger.pending_read.get());
    assert_eq!(ledger.release_checks.get(), 2);
    assert!(ledger.work.get() > 3*C); // no per-window reset or failed-run refund
}

#[test]
fn late_read_failure_and_consuming_abandonment_not_async_cancellation_proof() {
    let source = source(); source.weights.fail_at.set(Some(200));
    failure_cleanup(&source); assert!(source.weights.reads.get() > 200);
    source.weights.fail_at.set(None);
    let ledger = ClearLedger::new(usize::MAX,usize::MAX); let mut scratch = vec![47;C];
    let words = input_words(); let partition = plan(); let reads = source.weights.reads.get();
    let input = ClearInput::new(&source,&words,&mut scratch,&ledger).unwrap();
    let p = ClearProjection::bind(&source,&input,&partition).unwrap();
    let work = ledger.work.get(); drop(p); drop(input);
    assert_eq!(ledger.work.get(),work); assert_eq!(ledger.resident.get(),0);
    assert_eq!(source.weights.reads.get(),reads); assert!(scratch.iter().all(|&x| x == 0));
}

fn limits() -> (usize,usize,usize) {
    let source = source(); let words = input_words(); let partition = plan();
    let ledger = ClearLedger::new(usize::MAX,usize::MAX); let mut scratch = vec![0;C];
    let input = ClearInput::new(&source,&words,&mut scratch,&ledger).unwrap();
    let p = ClearProjection::bind(&source,&input,&partition).unwrap();
    let mut storage = Storage::new(R,C,65,127);
    let (work,bytes) = p.quote(&storage.buffers()).unwrap();
    assert!(work >= 16*R*C+4*C+8*R);
    let both_windows = 4*p.max_lanes;
    assert!(bytes >= both_windows + 2*p.max_lanes + 2*p.max_scales);
    assert_eq!(source.weights.reads.get(),0);
    (ledger.work.get()+work,ledger.resident.get()+bytes,both_windows)
}
#[test]
fn exact_fit_one_short_work_residency_overlap_and_no_refund_before_cleanup() {
    let (work,bytes,overlap) = limits();
    for (work_cap,byte_cap,success) in [(work,bytes,true),(work-1,bytes,false),(work,bytes-1,false),(work,bytes-overlap,false)] {
        let source = source(); let words = input_words(); let partition = plan();
        let ledger = ClearLedger::new(work_cap,byte_cap); let mut scratch = vec![0;C];
        let input = ClearInput::new(&source,&words,&mut scratch,&ledger).unwrap();
        let p = ClearProjection::bind(&source,&input,&partition).unwrap();
        let retained = input._reservation.bytes; let mut storage = Storage::new(R,C,65,127);
        let result = p.run(&source,&input,storage.buffers(),&ledger);
        assert_eq!(result.is_ok(),success); drop(result);
        if success { storage.cleaned(); assert_eq!(ledger.work.get(),work); }
        else {
            storage.untouched(); assert_eq!(source.weights.reads.get(),0);
            assert!(storage.input.iter().all(|&x| x == 19));
            assert!(storage.sums.iter().all(|&x| x == 19));
            assert!(storage.coefficients.iter().all(|&x| x == 19));
            assert!(storage.session.iter().all(|&x| x == 19));
        }
        assert_eq!(ledger.resident.get(),retained);
        drop(input); assert_eq!(ledger.resident.get(),0); assert!(scratch.iter().all(|&x|x==0));
    }
}

#[test]
fn allocation_failure_before_session_creation_clears_full_sums_after_admission() {
    for ordinal in 0..4 {
        let source = source(); let words = input_words(); let partition = plan();
        let ledger = ClearLedger::new(usize::MAX,usize::MAX); let mut scratch = vec![41;C];
        let input = ClearInput::new(&source,&words,&mut scratch,&ledger).unwrap();
        let p = ClearProjection::bind(&source,&input,&partition).unwrap();
        let retained = input._reservation.bytes;
        let work = ledger.work.get(); let releases = ledger.release_checks.get();
        ledger.fail_read_allocation.set(Some(ordinal));
        let mut storage = Storage::new(R,C,65,127);
        assert!(matches!(p.run(&source,&input,storage.buffers(),&ledger), Err(Error::Allocation)));
        // No LinearSession exists yet. Full sums (including spare entry) must
        // already be zero; other/read guards run before both lease releases.
        storage.cleaned(); storage.untouched();
        assert_eq!(source.weights.reads.get(),0);
        assert_eq!(source.scales.as_ref().unwrap().reads.get(),0);
        assert!(!ledger.pending_scratch.get()); assert!(!ledger.pending_read.get());
        assert_eq!(ledger.release_checks.get(),releases+2);
        assert_eq!(ledger.resident.get(),retained); assert!(ledger.work.get() > work);
        drop(input); assert_eq!(ledger.resident.get(),0);
        assert!(scratch.iter().all(|&x| x == 0));
    }
}

#[test]
fn checked_input_admission_overflow_nonfinite_and_failed_retention() {
    let source = source(); let words = input_words(); let mut scratch = vec![59;C];
    let bytes = ClearInput::quote(&source,&words,&scratch).unwrap();
    for ledger in [ClearLedger::new(3*C-1,usize::MAX),ClearLedger::new(usize::MAX,bytes-1)] {
        assert!(ClearInput::new(&source,&words,&mut scratch,&ledger).is_err());
        assert!(scratch.iter().all(|&x|x==59)); assert_eq!(ledger.resident.get(),0);
    }
    let ledger = ClearLedger::new(usize::MAX,usize::MAX);
    ledger.work.set(usize::MAX);
    assert!(ClearInput::new(&source,&words,&mut scratch,&ledger).is_err());
    assert_eq!(ledger.resident.get(),0); assert_eq!(source.weights.reads.get(),0);
    ledger.work.set(0);
    ledger.resident.set(usize::MAX);
    assert!(ClearInput::new(&source,&words,&mut scratch,&ledger).is_err());
    assert!(scratch.iter().all(|&x|x==59));
    assert_eq!(ledger.resident.get(),usize::MAX);
    ledger.resident.set(0);
    let mut words = words; words[C-1] = 0x7fc1;
    assert!(ClearInput::new(&source,&words,&mut scratch,&ledger).is_err());
    assert!(scratch.iter().all(|&x|x==0)); assert_eq!(ledger.resident.get(),0);
    assert_eq!(ledger.work.get(),3*C);
}

#[test]
fn completed_values_outlive_source_prepared_input_and_plan_not_output_buffers() {
    let ledger = ClearLedger::new(usize::MAX,usize::MAX); let mut storage = Storage::new(R,C,65,127);
    let produced = {
        let source = source(); let words = input_words(); let partition = plan(); let mut scratch = vec![0;C];
        let input = ClearInput::new(&source,&words,&mut scratch,&ledger).unwrap();
        ClearProjection::bind(&source,&input,&partition).unwrap().run(&source,&input,storage.buffers(),&ledger).unwrap()
    };
    assert_eq!(produced.field().len(),R); assert_eq!(produced.bf16().len(),R);
    assert!(ledger.resident.get() > 0); drop(produced);
    storage.cleaned(); assert_eq!(ledger.resident.get(),0);
}

#[test]
fn original_row_scale_boundary_not_transposed_or_retiled_at_band_origin() {
    let raw: Vec<_> = (0..256*128).map(|i| [0x31,0xb8,0x01,0x7e][i%4]).collect();
    let source = FixtureProjectionSource::fp8((7,8),[256,128],&raw,&[0x3f80,0x3e80]).unwrap();
    let words = vec![0x3f81;128];
    let expected = reference(&source,&words);
    let partition: Vec<_> = [(0,127),(127,2),(129,127)].into_iter().map(|(row_start,rows)|
        SourceWindow { row_start,rows,column_start:0,columns:128 }).collect();
    let ledger = ClearLedger::new(usize::MAX,usize::MAX);
    let mut scratch = vec![0;128];
    let input = ClearInput::new(&source,&words,&mut scratch,&ledger).unwrap();
    let mut storage = Storage::new(256,128,127,128);
    let output = ClearProjection::bind(&source,&input,&partition).unwrap()
        .run(&source,&input,storage.buffers(),&ledger).unwrap();
    assert_eq!(output.field(),expected.0); assert_eq!(output.bf16(),expected.1);
    drop(output); storage.cleaned();
}

#[test]
fn binding_one_short_retained_metadata_preserves_prepared_owner_before_reads() {
    let source = source(); let words = input_words(); let partition = plan();
    let probe_ledger = ClearLedger::new(usize::MAX,usize::MAX);
    let mut probe_scratch = vec![0;C];
    let probe_input = ClearInput::new(&source,&words,&mut probe_scratch,&probe_ledger).unwrap();
    let binding = ClearProjection::bind(&source,&probe_input,&partition).unwrap();
    let exact_bytes = probe_ledger.resident.get();
    let exact_work = probe_ledger.work.get();
    drop(binding); drop(probe_input);
    for (work,bytes,ok) in [(exact_work,exact_bytes,true),
        (exact_work-1,exact_bytes,false),(exact_work,exact_bytes-1,false)] {
        let ledger = ClearLedger::new(work,bytes); let mut scratch = vec![0;C];
        let input = ClearInput::new(&source,&words,&mut scratch,&ledger).unwrap();
        let retained = ledger.resident.get();
        let result = ClearProjection::bind(&source,&input,&partition);
        assert_eq!(result.is_ok(),ok); drop(result);
        assert_eq!(ledger.resident.get(),retained);
        assert!(std::ptr::eq(input.source,&source));
        assert_eq!(input.original.rows,R); assert_eq!(input.original.columns,C);
        assert_eq!(source.weights.reads.get(),0);
        drop(input); assert!(scratch.iter().all(|&x|x==0));
    }
}
