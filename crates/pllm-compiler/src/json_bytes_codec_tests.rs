//! Source-authored cases, NOT a Python runtime oracle. All tests UNRUN.
use super::*;
use crate::{original_source, plan_input::json};
use pllm_garble::shared_allocation::{Budget, OwnedBudget, MAX_ALLOCATION_BYTES};
use pllm_models::{ConfigLowererLimits, DecoderExpansionBudget, DecoderWorkload};

// Genuine existing public factory path, following original_source_tests.rs.
// No sibling access to private root/ledger or production test constructor.
const CONFIG: &[u8] = br#"{"model_type":"qwen2","hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":7,"max_position_embeddings":8,"hidden_act":"silu","rms_norm_eps":0.000001,"rope_theta":10000,"tie_word_embeddings":true,"use_sliding_window":false}"#;
const ROOT_WORK: u64 = 1_000_000_000_000;

fn source_limits() -> original_source::Limits {
    original_source::Limits {
        expansion: DecoderExpansionBudget {
            max_layers: 8, max_feature_width: 128, max_attention_heads: 8,
            max_vocabulary: 128, max_tensor_elements: 32768,
        },
        factory: ConfigLowererLimits {
            max_operations_per_phase: 256, max_string_bytes: 1024,
            max_attribute_nodes: 32768, max_retained_strings: 1 << 20,
            max_factory_bytes: 64 << 20,
        },
        max_work: ROOT_WORK, max_working_payload: 128 << 20,
    }
}

fn limits() -> Limits {
    Limits { max_source_bytes: 4096, max_points: 4096, max_work: 1_000_000, max_working_payload: 8 << 20 }
}

fn raw(budget: &OwnedBudget, bytes: &[u8], spare: usize) -> OwnedCharged<Vec<u8>> {
    let capacity = bytes.len() + spare;
    budget.charge_external(capacity).unwrap().try_build(move |_| {
        let mut v = Vec::new();
        v.try_reserve_exact(capacity).map_err(|_| NativeError::Allocation)?;
        if v.capacity() != capacity { return Err(NativeError::Resource); }
        v.extend_from_slice(bytes);
        Ok(v)
    }).unwrap()
}

fn root(budget: &OwnedBudget) -> OriginalSourcePlan {
    let jl = json::Limits {
        max_source_bytes: 32768, max_nodes: 512, max_depth: 16,
        max_string_bytes: 16384, max_number_bytes: 64,
        max_number_work: 1_000_000, max_work: 100_000_000,
        max_payload_bytes: 4 << 20,
    };
    let imported = json::quote(
        raw(budget, CONFIG, 0), budget.charge_external(json::bootstrap_bytes().unwrap()).unwrap(), jl,
    ).unwrap().construct().unwrap();
    let sl = source_limits();
    let bootstrap = original_source::bootstrap_bytes(&imported, sl).unwrap();
    original_source::quote(imported, budget.charge_external(bootstrap).unwrap(),
        DecoderWorkload { batch: 1, max_input_tokens: 4, max_new_tokens: 2 }, sl).unwrap().construct().unwrap()
}

fn controller(budget: &OwnedBudget) -> OwnedExternalCharge {
    budget.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap()).unwrap()
}
fn quote(budget: &OwnedBudget, root: &OriginalSourcePlan, bytes: &[u8]) -> Result<CodecQuote, Error> {
    quote_codec(raw(budget, bytes, 7), budget.charge_external(codec_control_bytes().unwrap()).unwrap(),
        root, controller(budget), limits())
}
fn decoded(bytes: &[u8]) -> Result<Vec<u32>, Error> {
    let mut d = Decoder::new(bytes, detect(bytes));
    let mut out = Vec::new();
    while let Some(point) = d.next()? { out.push(point); }
    Ok(out)
}
fn failure(bytes: &[u8], kind: DecodeFailure, start: usize, end: usize, base: usize) {
    assert_eq!(decoded(bytes), Err(Error::Decode(DecodeError { kind, start, end, error_input_base: base })));
}

#[test]
fn pinned_detection_lengths_zero_through_four_and_all_zero_patterns() {
    use Encoding::*;
    for (bytes, enc, points) in [
        (&b""[..], Utf8, &[][..]), (&[0][..], Utf8, &[0][..]),
        (&[0, 0][..], Utf16Be, &[0][..]), (&[0, 0, 0][..], Utf8, &[0, 0, 0][..]),
        (&[0, 0, 0, 0][..], Utf32Be, &[0][..]),
        (&[0, 65][..], Utf16Be, &[65][..]), (&[65, 0][..], Utf16Le, &[65][..]),
        (&[65, 65][..], Utf8, &[65, 65][..]),
        (&[0, 65, 0][..], Utf8, &[0, 65, 0][..]),
        (&[65, 0, 0][..], Utf8, &[65, 0, 0][..]),
    ] {
        assert_eq!(detect(bytes).encoding, enc);
        assert_eq!(decoded(bytes).unwrap(), points);
    }
    // Literal rows from the six-pin detect_encoding source contract, NOT a
    // transcription of candidate branches. Bit 0 changes byte 0, bit 1 byte 1,
    // bit 2 byte 2, bit 3 byte 3; unset => 00, set => 41.
    let expected = [
        Utf32Be, Utf32Le, Utf16Be, Utf8,
        Utf32Be, Utf16Le, Utf16Be, Utf8,
        Utf32Be, Utf16Le, Utf16Be, Utf8,
        Utf32Be, Utf16Le, Utf16Be, Utf8,
    ];
    for bits in 0..16 {
        let p = [if bits & 1 == 0 { 0 } else { 65 }, if bits & 2 == 0 { 0 } else { 65 },
            if bits & 4 == 0 { 0 } else { 65 }, if bits & 8 == 0 { 0 } else { 65 }];
        assert_eq!(detect(&p).encoding, expected[bits]);
        assert_eq!(detect(&p).bom_bytes, 0);
    }
}

#[test]
fn pinned_bom_precedence_strip_one_repeated_and_interior() {
    use Encoding::*;
    for (bytes, enc, skip, expected) in [
        (&[0xff, 0xfe, 0, 0][..], Utf32Le, 4, &[][..]),
        (&[0, 0, 0xfe, 0xff][..], Utf32Be, 4, &[][..]),
        (&[0xff, 0xfe][..], Utf16Le, 2, &[][..]),
        (&[0xfe, 0xff][..], Utf16Be, 2, &[][..]),
        (&[0xef, 0xbb, 0xbf][..], Utf8, 3, &[][..]),
        (&[0xef, 0xbb, 0xbf, 0xef, 0xbb, 0xbf, 65, 0xef, 0xbb, 0xbf][..], Utf8, 3, &[0xfeff, 65, 0xfeff][..]),
        (&[0xff, 0xfe, 0xff, 0xfe, 65, 0, 0xff, 0xfe][..], Utf16Le, 2, &[0xfeff, 65, 0xfeff][..]),
        (&[0xfe, 0xff, 0xfe, 0xff, 0, 65, 0xfe, 0xff][..], Utf16Be, 2, &[0xfeff, 65, 0xfeff][..]),
        (&[0xff, 0xfe, 0, 0, 0xff, 0xfe, 0, 0, 65, 0, 0, 0, 0xff, 0xfe, 0, 0][..], Utf32Le, 4, &[0xfeff, 65, 0xfeff][..]),
        (&[0, 0, 0xfe, 0xff, 0, 0, 0xfe, 0xff, 0, 0, 0, 65, 0, 0, 0xfe, 0xff][..], Utf32Be, 4, &[0xfeff, 65, 0xfeff][..]),
    ] {
        assert_eq!(detect(bytes).encoding, enc);
        assert_eq!(detect(bytes).bom_bytes, skip);
        assert_eq!(decoded(bytes).unwrap(), expected);
    }
    // Same byte signature inside UTF-8 data has no detector authority.
    assert_eq!(decoded(b"a\xef\xbb\xbfb").unwrap(), [97, 0xfeff, 98]);
}

#[test]
fn pinned_utf8_surrogatepass_never_pairs_and_malformed_spans() {
    use DecodeFailure::*;
    assert_eq!(decoded(&[0x7f, 0xc2, 0x80, 0xdf, 0xbf, 0xe0, 0xa0, 0x80,
        0xed, 0x9f, 0xbf, 0xed, 0xa0, 0x80, 0xed, 0xbf, 0xbf,
        0xf0, 0x90, 0x80, 0x80, 0xf4, 0x8f, 0xbf, 0xbf]).unwrap(),
        [0x7f, 0x80, 0x7ff, 0x800, 0xd7ff, 0xd800, 0xdfff, 0x10000, 0x10ffff]);
    assert_eq!(decoded(&[0xed, 0xa0, 0x80, 0xed, 0xb0, 0x80]).unwrap(), [0xd800, 0xdc00]);
    for byte in [0x80, 0xbf, 0xc0, 0xc1, 0xf5, 0xff] { failure(&[byte], InvalidStart, 0, 1, 0); }
    for b in [&[0xe0, 0x80, 0x80][..], &[0xf0, 0x80, 0x80, 0x80][..],
        &[0xf4, 0x90, 0x80, 0x80][..], &[0xc2, 65][..], &[0xed, 0xa0][..], &[0xed, 0xa0, 65][..]] {
        failure(b, InvalidContinuation, 0, 1, 0);
    }
    failure(&[0xe2, 0x82, 65], InvalidContinuation, 0, 2, 0);
    failure(&[0xf0, 0x90, 0x80, 65], InvalidContinuation, 0, 3, 0);
    for b in [&[0xc2][..], &[0xe2, 0x82][..], &[0xf0, 0x90, 0x80][..]] {
        failure(b, UnexpectedEnd, 0, b.len(), 0);
    }
    failure(&[0xef, 0xbb, 0xbf, 0xe2, 0x82], UnexpectedEnd, 0, 2, 3);
    failure(&[0xef, 0xbb, 0xbf, 65, 0xff], InvalidStart, 1, 2, 3);
}

#[test]
fn pinned_utf16_both_orders_pairs_isolation_cached_unit_and_odd_tail() {
    for le in [true, false] {
        let mut bytes = if le { vec![0xff, 0xfe] } else { vec![0xfe, 0xff] };
        for unit in [0x0041u16, 0xd800, 0xdc00, 0xdbff, 0xdfff, 0xd800, 0xd801, 0x0042, 0xdc00] {
            bytes.extend_from_slice(&if le { unit.to_le_bytes() } else { unit.to_be_bytes() });
        }
        assert_eq!(decoded(&bytes).unwrap(), [65, 0x10000, 0x10ffff, 0xd800, 0xd801, 66, 0xdc00]);
        let bom = if le { [0xff, 0xfe] } else { [0xfe, 0xff] };
        let high = if le { [0, 0xd8] } else { [0xd8, 0] };
        let mut isolated = bom.to_vec(); isolated.extend_from_slice(&high);
        assert_eq!(decoded(&isolated).unwrap(), [0xd800]);
        isolated.push(65);
        failure(&isolated, DecodeFailure::TruncatedUnit, 4, 5, 0);
        failure(&[bom[0], bom[1], 65], DecodeFailure::TruncatedUnit, 2, 3, 0);
    }
}

#[test]
fn pinned_utf32_both_orders_surrogates_range_and_truncation() {
    for le in [true, false] {
        let bom = if le { [0xff, 0xfe, 0, 0] } else { [0, 0, 0xfe, 0xff] };
        let mut bytes = bom.to_vec();
        for point in [0u32, 65, 0xd800, 0xdc00, 0x10000, 0x10ffff] {
            bytes.extend_from_slice(&if le { point.to_le_bytes() } else { point.to_be_bytes() });
        }
        assert_eq!(decoded(&bytes).unwrap(), [0, 65, 0xd800, 0xdc00, 0x10000, 0x10ffff]);
        for point in [0x110000u32, 0xffffffff] {
            let mut bad = bom.to_vec();
            bad.extend_from_slice(&if le { point.to_le_bytes() } else { point.to_be_bytes() });
            failure(&bad, DecodeFailure::OutOfRange, 4, 8, 0);
        }
        for count in 1..4 {
            let mut bad = bom.to_vec(); bad.extend_from_slice(&[0, 0, 0][..count]);
            failure(&bad, DecodeFailure::TruncatedUnit, 4, 4 + count, 0);
        }
    }
}

#[test]
fn actual_owner_pointer_len_capacity_counter_and_retained_controls() {
    let b = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let r = root(&b);
    let baseline = b.budget().used_bytes();
    let owner = raw(&b, &[0xed, 0xa0, 0x80], 29);
    let ptr = owner.payload().as_ptr();
    let len = owner.payload().len();
    let cap = owner.payload().capacity();
    let q = quote_codec(owner, b.charge_external(codec_control_bytes().unwrap()).unwrap(), &r,
        controller(&b), limits()).unwrap();
    let expected = *q.resources();
    assert_eq!(expected.prepaid_work, 82); // literal three-byte debit
    let out = q.construct().unwrap();
    assert_eq!(out.payload().codepoints(), [0xd800]);
    assert_eq!(out.payload().original_owner().payload().as_ptr(), ptr);
    assert_eq!(out.payload().original_owner().payload().len(), len);
    assert_eq!(out.payload().original_owner().payload().capacity(), cap);
    assert!(std::ptr::eq(out.budget(), r.budget()));
    assert_eq!(out.payload().resources(), &expected);
    assert_eq!(out.charged_bytes(), expected.working_charge_bytes);
    assert_eq!(out.payload().controller.get().unwrap().charged_bytes(), expected.controller_charge_bytes);
    assert_eq!(b.budget().used_bytes(), baseline + expected.original_owner_bytes
        + expected.working_charge_bytes + expected.controller_charge_bytes);
    drop(out);
    assert_eq!(b.budget().used_bytes(), baseline);
}

#[test]
fn both_passes_empty_and_all_encoding_families() {
    let b = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let r = root(&b);
    for (bytes, expected) in [
        (&b""[..], &[][..]),
        (&[0xef, 0xbb, 0xbf][..], &[][..]),
        (&[0xed, 0xa0, 0x80, 0xed, 0xb0, 0x80][..], &[0xd800, 0xdc00][..]),
        (&[0xff, 0xfe, 0, 0xd8, 1, 0xd8, 0, 0xdc][..], &[0xd800, 0x10400][..]),
        (&[0xfe, 0xff, 0xd8, 0, 0xd8, 1, 0xdc, 0][..], &[0xd800, 0x10400][..]),
        (&[0xff, 0xfe, 0, 0, 0, 0xd8, 0, 0][..], &[0xd800][..]),
        (&[0, 0, 0xfe, 0xff, 0, 0, 0xd8, 0][..], &[0xd800][..]),
        (&[0, 0, 0, 65][..], &[65][..]),
        (&[65, 0, 0, 0][..], &[65][..]),
        (&[0, 65][..], &[65][..]),
        (&[65, 0][..], &[65][..]),
    ] {
        let out = quote(&b, &r, bytes).unwrap().construct().unwrap();
        assert_eq!(out.payload().codepoints(), expected);
        assert_eq!(out.payload().resources().point_capacity_bytes, 4 * expected.len());
        assert_eq!(out.payload().points.capacity(), expected.len());
        assert_eq!(out.payload().encoding(), detect(bytes).encoding);
        assert_eq!(out.payload().bom_bytes(), detect(bytes).bom_bytes);
    }
}

#[test]
fn metadata_refusals_precede_read_factory_and_root_burn() {
    let b = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let other = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let r = root(&b);
    let before = FACTORY_ENTRIES.with(|n| n.get());
    // Invalid bytes would otherwise burn after beginning the attempt.
    let undercharged = b.charge_external(0).unwrap().try_build(|_| Ok(vec![0xff; 32])).unwrap();
    assert!(matches!(quote_codec(undercharged, b.charge_external(codec_control_bytes().unwrap()).unwrap(),
        &r, controller(&b), limits()), Err(Error::Capacity)));
    assert!(matches!(quote_codec(raw(&other, &[0xff], 0), b.charge_external(codec_control_bytes().unwrap()).unwrap(),
        &r, controller(&b), limits()), Err(Error::Counter)));
    assert!(matches!(quote_codec(raw(&b, &[0xff], 0), other.charge_external(codec_control_bytes().unwrap()).unwrap(),
        &r, controller(&b), limits()), Err(Error::Counter)));
    assert!(matches!(quote_codec(raw(&b, &[0xff], 0), b.charge_external(codec_control_bytes().unwrap()).unwrap(),
        &r, controller(&other), limits()), Err(Error::Counter)));
    assert!(matches!(quote_codec(raw(&b, &[0xff], 0), b.charge_external(codec_control_bytes().unwrap() - 1).unwrap(),
        &r, controller(&b), limits()), Err(Error::Capacity)));
    assert!(matches!(quote_codec(raw(&b, &[0xff], 0), b.charge_external(codec_control_bytes().unwrap()).unwrap(),
        &r, b.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap() - 1).unwrap(), limits()), Err(Error::Capacity)));
    assert_eq!(FACTORY_ENTRIES.with(|n| n.get()), before);
    drop(quote(&b, &r, b"a").unwrap().construct().unwrap());
}

#[test]
fn exact_work_and_one_short_prepaid_before_factory() {
    for short in [false, true] {
        let b = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let r = root(&b);
        let before = FACTORY_ENTRIES.with(|n| n.get());
        let mut l = limits(); l.max_work = 68 - u64::from(short);
        let q = quote_codec(raw(&b, b"a", 0), b.charge_external(codec_control_bytes().unwrap()).unwrap(),
            &r, controller(&b), l);
        if short {
            assert!(matches!(q, Err(Error::Work)));
            assert_eq!(FACTORY_ENTRIES.with(|n| n.get()), before);
            assert!(matches!(quote(&b, &r, b"a"), Err(Error::Source(SourceError::Burned))));
        } else { drop(q.unwrap().construct().unwrap()); }
    }
}

#[test]
fn exact_working_payload_and_one_short_precede_factory() {
    for short in [false, true] {
        let b = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let r = root(&b);
        let before = FACTORY_ENTRIES.with(|n| n.get());
        let mut l = limits(); l.max_working_payload = codec_control_bytes().unwrap() + 4 - usize::from(short);
        let q = quote_codec(raw(&b, b"a", 0), b.charge_external(codec_control_bytes().unwrap()).unwrap(),
            &r, controller(&b), l);
        if short {
            assert!(matches!(q, Err(Error::Resource)));
            assert_eq!(FACTORY_ENTRIES.with(|n| n.get()), before);
        } else { drop(q.unwrap().construct().unwrap()); }
    }
}

#[test]
fn exact_counter_headroom_and_one_short_before_allocating_factory() {
    for short in [false, true] {
        let b = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let r = root(&b);
        let q = quote(&b, &r, b"a").unwrap();
        let initial = q.working.charged_bytes();
        let increment = q.resources().working_charge_bytes - initial;
        let headroom = increment - usize::from(short);
        let ballast = b.charge_external(MAX_ALLOCATION_BYTES - b.budget().used_bytes()
            - size_of::<OwnedExternalCharge>() - headroom).unwrap();
        let entries = FACTORY_ENTRIES.with(|n| n.get());
        let result = q.construct();
        if short {
            assert!(matches!(result, Err(Error::Native(NativeError::Resource))));
            assert_eq!(FACTORY_ENTRIES.with(|n| n.get()), entries);
            assert_eq!(r.lower_bootstrap_bytes(source_limits()), Err(SourceError::Burned));
        } else {
            let out = result.unwrap();
            assert_eq!(b.budget().used_bytes(), MAX_ALLOCATION_BYTES);
            assert_eq!(out.payload().codepoints(), [97]);
            drop(out);
        }
        drop(ballast);
    }
}

#[test]
fn actual_shared_cumulative_work_no_fresh_codec_quota_and_committed_survival() {
    let b = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let r = root(&b);
    let fee = 68; // literal one-byte debit, independent of production work()
    let remaining = ROOT_WORK - r.resources().prepaid_work;
    let guard = controller(&b);
    r.prepay_primitive_work(&guard, remaining - 2 * fee).unwrap();
    let first = quote(&b, &r, b"a").unwrap().construct().unwrap();
    let second = quote(&b, &r, b"b").unwrap().construct().unwrap();
    assert!(matches!(quote(&b, &r, b"c"), Err(Error::Source(SourceError::Work))));
    assert!(matches!(quote(&b, &r, b"d"), Err(Error::Source(SourceError::Burned))));
    assert_eq!(first.payload().codepoints(), [97]);
    assert_eq!(second.payload().codepoints(), [98]);
    assert!(first.payload().controller.get().is_some());
    assert!(second.payload().controller.get().is_some());
}

#[test]
fn abandonment_failure_and_partial_unwind_cleanup_burn_siblings() {
    for mode in 0..4 {
        let b = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let r = root(&b);
        let baseline = b.budget().used_bytes();
        let q = quote(&b, &r, b"ab").unwrap();
        match mode {
            0 => drop(q),
            1 => {
                FAIL_AFTER_POINT.with(|flag| flag.set(true));
                assert!(matches!(q.construct(), Err(Error::Native(NativeError::Resource))));
            }
            2 => {
                PANIC_AFTER_POINT.with(|flag| flag.set(true));
                assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| q.construct())).is_err());
            }
            _ => {
                PANIC_FACTORY.with(|flag| flag.set(true));
                assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| q.construct())).is_err());
            }
        }
        assert_eq!(b.budget().used_bytes(), baseline);
        assert!(matches!(quote(&b, &r, b"a"), Err(Error::Source(SourceError::Burned))));
    }
}

#[test]
fn decode_failure_burns_and_later_sibling_cannot_commit_or_clear_burn() {
    let b = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let r = root(&b);
    let good = quote(&b, &r, b"a").unwrap();
    let before = FACTORY_ENTRIES.with(|n| n.get());
    assert!(matches!(quote(&b, &r, &[0xff]), Err(Error::Decode(_))));
    assert_eq!(FACTORY_ENTRIES.with(|n| n.get()), before);
    assert!(matches!(good.construct(), Err(Error::Source(SourceError::Burned))));
    assert!(matches!(quote(&b, &r, b"a"), Err(Error::Source(SourceError::Burned))));
}

#[test]
fn literal_debits_and_same_root_remaining_ledger_exact_and_one_short() {
    // Expectations deliberately do not call work() or read FIXED_WORK. Original
    // factory/import work and the retained root allocation are real baselines.
    for (bytes, debit) in [(&b""[..], 61u64), (&b"a"[..], 68), (&b"abc"[..], 82)] {
        for short in [false, true] {
            let b = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
            let r = root(&b);
            let initial_work = r.resources().prepaid_work;
            assert!(initial_work > 0);
            assert!(b.budget().used_bytes() > 0);
            let guard = controller(&b);
            let baseline = b.budget().used_bytes();
            let available = debit - u64::from(short);
            r.prepay_primitive_work(&guard, ROOT_WORK - initial_work - available).unwrap();
            let entries = FACTORY_ENTRIES.with(|n| n.get());
            let result = quote(&b, &r, bytes);
            if short {
                assert!(matches!(result, Err(Error::Source(SourceError::Work))));
                assert_eq!(FACTORY_ENTRIES.with(|n| n.get()), entries);
                assert_eq!(b.budget().used_bytes(), baseline);
                assert_eq!(r.prepay_primitive_work(&guard, 1), Err(SourceError::Burned));
            } else {
                let q = result.unwrap();
                assert_eq!(q.resources().prepaid_work, debit);
                let out = q.construct().unwrap();
                assert_eq!(out.payload().resources().prepaid_work, debit);
                // Exactly exhausted SAME ledger; no local counter/refund/reset.
                assert_eq!(r.prepay_primitive_work(&guard, 1), Err(SourceError::Work));
                assert!(out.payload().controller.get().is_some());
                assert_eq!(out.payload().original_owner().payload().as_slice(), bytes);
                drop(out);
                assert_eq!(b.budget().used_bytes(), baseline);
            }
        }
    }
}

#[test]
fn pinned_malformed_quote_paths_use_literal_spans_and_genuine_root_per_input() {
    use DecodeFailure::*;
    // Final-decoder/surrogatepass expectations from the six immutable source
    // pins, not candidate detect/decoded/error helpers or an installed runtime.
    for (bytes, kind, start, end, base) in [
        (&[0xef, 0xbb, 0xbf, 0x41, 0xff][..], InvalidStart, 1, 2, 3),
        (&[0xff, 0xfe, 0x41, 0, 0x42][..], TruncatedUnit, 4, 5, 0),
        (&[0xff, 0xfe, 0, 0, 0x41, 0, 0, 0, 0x42][..], TruncatedUnit, 8, 9, 0),
        (&[0xff, 0xfe, 0, 0xd8, 1, 0xd8, 0, 0xdc, 0x41][..], TruncatedUnit, 8, 9, 0),
        (&[0xff, 0xfe, 0, 0, 0x41, 0, 0, 0, 0, 0, 0x11, 0][..], OutOfRange, 8, 12, 0),
    ] {
        let b = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let r = root(&b);
        let baseline = b.budget().used_bytes();
        let entries = FACTORY_ENTRIES.with(|n| n.get());
        let result = quote(&b, &r, bytes);
        assert_eq!(result.err(), Some(Error::Decode(DecodeError {
            kind, start, end, error_input_base: base,
        })));
        assert_eq!(FACTORY_ENTRIES.with(|n| n.get()), entries);
        assert_eq!(b.budget().used_bytes(), baseline);
        assert!(matches!(quote(&b, &r, b"a"), Err(Error::Source(SourceError::Burned))));
    }
}

#[test]
fn pinned_bom_only_quote_construct_retains_actual_original_owner() {
    use Encoding::*;
    let b = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let r = root(&b);
    for (bytes, encoding, skip) in [
        (&[0xef, 0xbb, 0xbf][..], Utf8, 3),
        (&[0xff, 0xfe][..], Utf16Le, 2),
        (&[0xfe, 0xff][..], Utf16Be, 2),
        (&[0xff, 0xfe, 0, 0][..], Utf32Le, 4),
        (&[0, 0, 0xfe, 0xff][..], Utf32Be, 4),
    ] {
        let baseline = b.budget().used_bytes();
        let owner = raw(&b, bytes, 11);
        let ptr = owner.payload().as_ptr();
        let cap = owner.payload().capacity();
        let original_charge = owner.charged_bytes();
        let q = quote_codec(owner, b.charge_external(codec_control_bytes().unwrap()).unwrap(),
            &r, controller(&b), limits()).unwrap();
        assert_eq!(q.resources().point_count, 0);
        let out = q.construct().unwrap();
        assert!(out.payload().codepoints().is_empty());
        assert_eq!(out.payload().encoding(), encoding);
        assert_eq!(out.payload().bom_bytes(), skip);
        let retained = out.payload().original_owner();
        assert_eq!(retained.payload().as_ptr(), ptr);
        assert_eq!(retained.payload().len(), bytes.len());
        assert_eq!(retained.payload().capacity(), cap);
        assert_eq!(retained.payload().as_slice(), bytes);
        assert_eq!(retained.charged_bytes(), original_charge);
        assert!(std::ptr::eq(retained.budget(), r.budget()));
        drop(out);
        assert_eq!(b.budget().used_bytes(), baseline);
    }
}

// Shared-owner seam coverage ONLY. A sentinel cannot be inserted into the
// codec's actual Vec<u32>/Vec<u8> elements. Concrete codec Vec retirement still
// rests on source field/local order, not these observations. All tests UNRUN.
struct ChargeDropSentinel<'a> {
    budget: &'a Budget,
    expected_during_drop: usize,
    observed: &'a std::cell::Cell<Option<usize>>,
}
impl Drop for ChargeDropSentinel<'_> {
    fn drop(&mut self) {
        let used = self.budget.used_bytes();
        self.observed.set(Some(used));
        assert_eq!(used, self.expected_during_drop);
    }
}

#[test]
fn shared_owner_seam_charge_visible_during_partial_unwind_and_payload_drop() {
    for unwind in [false, true] {
        let b = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let baseline = b.budget().used_bytes();
        let observed = std::cell::Cell::new(None);
        let charge = b.charge_external(128).unwrap();
        let inline = size_of::<OwnedCharged<ChargeDropSentinel<'_>>>()
            - size_of::<OwnedExternalCharge>();
        let expected = baseline + charge.charged_bytes() + inline;
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            charge.try_build(|native| {
                assert!(std::ptr::eq(native, b.budget()));
                assert_eq!(native.used_bytes(), expected);
                let sentinel = ChargeDropSentinel {
                    budget: b.budget(), expected_during_drop: expected, observed: &observed,
                };
                if unwind { panic!("shared-owner seam partial builder unwind"); }
                Ok(sentinel)
            })
        }));
        if unwind {
            assert!(outcome.is_err());
        } else {
            let out = outcome.unwrap().unwrap();
            assert_eq!(observed.get(), None);
            assert_eq!(b.budget().used_bytes(), expected);
            drop(out);
        }
        assert_eq!(observed.get(), Some(expected));
        assert_eq!(b.budget().used_bytes(), baseline);
    }
}
