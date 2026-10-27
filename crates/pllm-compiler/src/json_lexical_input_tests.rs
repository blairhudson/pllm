//! Source-authored production quote -> construct controls. ALL TESTS UNRUN.
//! No numeric/Unicode conversion qualification or original-source identity claim.

use super::*;
use crate::plan_input::{FACTORIES, NUMBER_CALLS};
use pllm_garble::shared_allocation::{OwnedBudget, MAX_ALLOCATION_BYTES};

fn limits() -> LexicalLimits {
    LexicalLimits {
        max_source_bytes: 1 << 20,
        max_nodes: 100_000,
        max_depth: 64,
        max_string_bytes: 1 << 16,
        max_number_bytes: 1 << 16,
        max_payload_bytes: MAX_ALLOCATION_BYTES,
        max_work: 1 << 30,
        work: LexicalWorkLimits {
            quotation: 1 << 30,
            replay: 1 << 30,
            decode: 1 << 30,
            classify: 1 << 30,
            deduplicate: 1 << 30,
            cleanup: 1 << 30,
        },
    }
}
fn incoming(root: &OwnedBudget, bytes: &[u8], spare: usize) -> OwnedCharged<Vec<u8>> {
    let capacity = bytes.len() + spare;
    root.charge_external(capacity).unwrap().try_build(|_| {
        let mut data = Vec::with_capacity(capacity);
        data.extend_from_slice(bytes);
        assert_eq!(data.capacity(), capacity);
        Ok(data)
    }).unwrap()
}
fn working(root: &OwnedBudget) -> OwnedExternalCharge {
    root.charge_external(lexical_bootstrap_bytes().unwrap()).unwrap()
}
fn reset() {
    FACTORIES.with(|state| state.set((0, None, None)));
    NUMBER_CALLS.with(|calls| calls.set(0));
    LEXICAL_RETIREMENTS.with(|events| events.borrow_mut().clear());
}
fn factories() -> usize {
    FACTORIES.with(|state| state.get().0)
}
fn lexical(root: &OwnedBudget, raw: &[u8]) -> OwnedCharged<LexicalValue> {
    quote_lexical(incoming(root, raw, 19), working(root), limits())
        .unwrap().construct().unwrap()
}
const FIXTURE: &[u8] = br#"{"z":[-0,-0.0,1E9],"a":{"x":"\u03bb","y":["one","two"]}}"#;

#[test]
fn lexical_byte_domain_and_nested_closing_replay_floor_literal_controls() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    reset();
    let raw = b"\"\xf0\x9f\xa6\x80\""; // quoted U+1F980, six UTF-8 bytes, three points
    let q = quote_lexical(incoming(&root, raw, 0), working(&root), limits()).unwrap();
    assert_eq!(q.counts().scan_work, 10); // 6 byte reads + 3 peeks + 1 emit
    assert_eq!(q.resources().work.quotation, 146); // 130 frames + 6 UTF-8 + 10
    assert_eq!(q.counts().decoded_string_bytes, 4); // NOT a point-count field
    let output = q.construct().unwrap();
    assert_eq!(output.payload().node(0).unwrap().raw_range(), 0..6);
    assert_eq!(output.payload().node(0).unwrap().decoded_string(), Some("🦀"));
    drop(output);
    let baseline = root.budget().used_bytes();
    // [[]]: 17 peek/read/emit events; two span slots; two closing updates.
    // Replay 17 + 2 init + 2 closing/finalization + 130 frames + 4 finish = 155.
    // Neither old 2S nor +4 may disappear into the measured scan counter.
    for (replay_cap, success) in [(155, true), (154, false)] {
        reset();
        let mut capped = limits();
        capped.work.replay = replay_cap;
        let result = quote_lexical(incoming(&root, b"[[]]", 0), working(&root), capped);
        if success {
            let q = result.unwrap();
            assert_eq!(q.counts().spans, 2);
            assert_eq!(q.counts().scan_work, 17);
            assert_eq!(q.resources().work.quotation, 151); // 130 + UTF-8 4 + 17
            assert_eq!(q.resources().work.replay, 155);
            let output = q.construct().unwrap();
            for (index, range, end, children) in [(0, 0..4, 2, 1), (1, 1..3, 2, 0)] {
                let node = output.payload().node(index).unwrap();
                assert_eq!(node.raw_range(), range);
                assert_eq!(node.subtree_end(), end);
                assert_eq!(node.immediate_child_count(), children);
            }
            drop(output);
        } else {
            assert!(matches!(result, Err(Error::WorkLimit)));
            assert_eq!(factories(), 0);
        }
        assert_eq!(root.budget().used_bytes(), baseline);
    }
}

#[test]
fn lexical_exact_numbers_and_ordered_subtrees() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    reset();
    let raw = br#"{"z":[-0,-0.0,1E9],"a":2}"#;
    let output = lexical(&root, raw);
    let value = output.payload();
    // Literal byte ranges, preorder and exclusive subtree ends, not a parser or
    // production planner used to construct expected results. z stays BEFORE a.
    let expected = [
        (LexicalKind::Object, 0..25, 8, 2),
        (LexicalKind::String, 1..4, 2, 0),
        (LexicalKind::Array, 5..18, 6, 3),
        (LexicalKind::Number, 6..8, 4, 0),
        (LexicalKind::Number, 9..13, 5, 0),
        (LexicalKind::Number, 14..17, 6, 0),
        (LexicalKind::String, 19..22, 7, 0),
        (LexicalKind::Number, 23..24, 8, 0),
    ];
    assert_eq!(value.counts().spans, expected.len());
    for (index, (kind, range, end, children)) in expected.into_iter().enumerate() {
        let node = value.node(index).unwrap();
        assert_eq!(node.kind(), kind);
        assert_eq!(node.raw_range(), range);
        assert_eq!(node.subtree_end(), end);
        assert_eq!(node.immediate_child_count(), children);
    }
    assert!(value.node(8).is_none());
    assert_eq!(value.node(1).unwrap().decoded_string(), Some("z"));
    assert_eq!(value.node(6).unwrap().decoded_string(), Some("a"));
    for (index, text, class) in [
        (3, "-0", LexicalNumberClass::Integer),
        (4, "-0.0", LexicalNumberClass::Float),
        (5, "1E9", LexicalNumberClass::Float),
        (7, "2", LexicalNumberClass::Integer),
    ] {
        assert_eq!(value.node(index).unwrap().number(), Some((text, class)));
        assert_eq!(value.node(index).unwrap().decoded_string(), None);
    }
    for (text, class) in [
        ("184467440737095516160000000000000000000", LexicalNumberClass::Integer),
        ("-92233720368547758090000000000000000000", LexicalNumberClass::Integer),
        ("1e99999999999999999999999", LexicalNumberClass::Float),
        ("-0.000000000000000000000001e-99999", LexicalNumberClass::Float),
    ] {
        let scalar = lexical(&root, text.as_bytes());
        assert_eq!(scalar.payload().node(0).unwrap().number(), Some((text, class)));
        assert_eq!(scalar.payload().raw(), text.as_bytes());
    }
    // No scalar serde numeric helper entered, even for non-f64 lexemes.
    assert_eq!(NUMBER_CALLS.with(|calls| calls.get()), 0);
    for (raw, kind, end) in [
        (b"null".as_slice(), LexicalKind::Null, 4),
        (b"true", LexicalKind::True, 4),
        (b"false", LexicalKind::False, 5),
        (b"[]", LexicalKind::Array, 2),
        (b"{}", LexicalKind::Object, 2),
    ] {
        let output = lexical(&root, raw);
        let node = output.payload().node(0).unwrap();
        assert_eq!(node.kind(), kind);
        assert_eq!(node.raw_range(), 0..end);
        assert_eq!(node.subtree_end(), 1);
        assert_eq!(node.immediate_child_count(), 0);
        assert_eq!(node.number(), None);
        assert_eq!(node.decoded_string(), None);
    }
    for malformed in [b"01".as_slice(), b"1e+", b"+1", b"NaN", b"[1,]"] {
        reset();
        assert!(matches!(quote_lexical(incoming(&root, malformed, 0), working(&root), limits()),
            Err(Error::Syntax)));
        assert_eq!(factories(), 0);
    }
}

#[test]
fn lexical_decoded_unicode_and_duplicate_aliases() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let raw = r#"["\u0061","a","\u03bb","\ud83e\udd80","λ","x\n"]"#.as_bytes();
    let output = lexical(&root, raw);
    for (index, expected) in ["a", "a", "λ", "🦀", "λ", "x\n"].into_iter().enumerate() {
        assert_eq!(output.payload().node(index + 1).unwrap().decoded_string(), Some(expected));
    }
    drop(output);
    let baseline = root.budget().used_bytes();
    for raw in [
        br#"{"a":1,"\u0061":2}"#.as_slice(),
        br#"{"a":{"nested":[1,2]},"\u0061":null}"#,
        r#"{"λ":0,"\u03bb":1}"#.as_bytes(),
        br#"{"\ud83e\udd80":0,"\ud83e\udd80":1}"#,
        br#"{"":0,"":1}"#,
    ] {
        reset();
        assert!(matches!(quote_lexical(incoming(&root, raw, 0), working(&root), limits())
            .unwrap().construct(), Err(Error::DuplicateField)));
        assert!(factories() > 0);
        assert_eq!(root.budget().used_bytes(), baseline);
    }
    // Keys in separate objects are NOT duplicates. Prefixes stay distinct.
    drop(lexical(&root, br#"{"a":{"a":1},"ab":2,"array":[{"a":3}]}"#));
    for raw in [br#""\ud800""#.as_slice(), br#""\udc00""#, &[b'"', 0xff, b'"']] {
        reset();
        assert!(matches!(quote_lexical(incoming(&root, raw, 0), working(&root), limits()),
            Err(Error::Syntax)));
        assert_eq!(factories(), 0);
        assert_eq!(root.budget().used_bytes(), baseline);
    }
}

#[test]
fn lexical_retains_original_allocation_capacity_and_counter() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let baseline = root.budget().used_bytes();
    assert_eq!(baseline, OwnedBudget::resources().unwrap().initial_reserved_bytes);
    let utf8 = incoming(&root, FIXTURE, 4096);
    let pointer = utf8.payload().as_ptr();
    let capacity = utf8.payload().capacity();
    let incoming_bytes = utf8.charged_bytes();
    let identity = std::ptr::from_ref(utf8.budget());
    let quote = quote_lexical(utf8, working(&root), limits()).unwrap();
    let resources = *quote.resources();
    let output = quote.construct().unwrap();
    assert_eq!(output.payload().raw().as_ptr(), pointer);
    assert_eq!(output.payload().raw_owner().payload().capacity(), capacity);
    assert_eq!(output.payload().raw_owner().charged_bytes(), incoming_bytes);
    assert_eq!(std::ptr::from_ref(output.budget()), identity);
    assert_eq!(std::ptr::from_ref(output.payload().raw_owner().budget()), identity);
    assert_eq!(resources.incoming_owner_bytes, incoming_bytes);
    assert_eq!(resources.working_charge_bytes, output.charged_bytes());
    assert_eq!(resources.original_owners_peak_bytes, incoming_bytes + output.charged_bytes());
    assert_eq!(root.budget().used_bytes(), baseline + resources.original_owners_peak_bytes);
    assert_eq!(root.budget().admitted_peak_bytes(), baseline + resources.original_owners_peak_bytes);
    drop(output);
    assert_eq!(root.budget().used_bytes(), baseline);

    // Actual guards keep original counter alive after root drops, not Rc-only custody.
    let output = lexical(&root, FIXTURE);
    drop(root);
    assert_eq!(std::ptr::from_ref(output.budget()), identity);
    assert_eq!(output.payload().raw(), FIXTURE);
    assert!(output.budget().used_bytes() >= output.payload().resources().original_owners_peak_bytes);
}

#[test]
fn lexical_snapshot_depth_16_17_32_64_and_scalar_boundary() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    for depth in [16, 17, 32, 64] {
        // Empty deepest container at value depth `depth`; root is depth zero.
        let raw = format!("{}{}", "[".repeat(depth + 1), "]".repeat(depth + 1));
        let output = lexical(&root, raw.as_bytes());
        assert_eq!(output.payload().counts().depth, depth);
        assert_eq!(output.payload().counts().spans, depth + 1);
        let leaf = output.payload().node(depth).unwrap();
        assert_eq!(leaf.kind(), LexicalKind::Array);
        assert_eq!(leaf.immediate_child_count(), 0);
        let scalar = format!("{}0{}", "[".repeat(depth), "]".repeat(depth));
        let output = lexical(&root, scalar.as_bytes());
        assert_eq!(output.payload().counts().depth, depth);
        assert_eq!(output.payload().node(depth).unwrap().number(),
            Some(("0", LexicalNumberClass::Integer)));
    }
    // Object keys at container depth 64 do not consume value-depth 65.
    let object = format!("{}{{}}{}", "{\"k\":".repeat(64), "}".repeat(64));
    assert_eq!(lexical(&root, object.as_bytes()).payload().counts().depth, 64);
    for raw in [
        format!("{}0{}", "[".repeat(65), "]".repeat(65)),
        format!("{}{}", "[".repeat(66), "]".repeat(66)),
        format!("{}0{}", "{\"k\":".repeat(65), "}".repeat(65)),
    ] {
        reset();
        assert!(matches!(quote_lexical(incoming(&root, raw.as_bytes(), 0), working(&root), limits()),
            Err(Error::DepthLimit)));
        assert_eq!(factories(), 0);
    }
    let exact = format!("{}0{}", "[".repeat(17), "]".repeat(17));
    assert!(matches!(quote_lexical(incoming(&root, exact.as_bytes(), 0), working(&root),
        LexicalLimits { max_depth: 16, ..limits() }), Err(Error::DepthLimit)));
}

#[test]
fn lexical_exact_capacity_and_one_short_before_factory() {
    let (exact, required, spans, strings) = {
        let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let baseline = root.budget().used_bytes();
        let quote = quote_lexical(incoming(&root, FIXTURE, 83), working(&root), limits()).unwrap();
        let counts = *quote.counts();
        let resources = *quote.resources();
        let required = lexical_bootstrap_bytes().unwrap()
            + size_of::<Span>() * counts.spans
            + size_of::<Option<String>>() * counts.spans
            + counts.decoded_string_bytes + counts.largest_string;
        assert_eq!(resources.required_working_payload, required);
        (baseline + resources.original_owners_peak_bytes, required, counts.spans,
            counts.decoded_string_bytes)
    };
    assert!(strings > 0 && spans > 0);
    for (limit, success) in [(exact, true), (exact - 1, false)] {
        let root = OwnedBudget::new(limit).unwrap();
        let baseline = root.budget().used_bytes();
        reset();
        let quote = quote_lexical(incoming(&root, FIXTURE, 83), working(&root), limits()).unwrap();
        let result = quote.construct();
        assert_eq!(result.is_ok(), success);
        if !success {
            assert_eq!(factories(), 0);
        }
        drop(result);
        assert_eq!(root.budget().used_bytes(), baseline);
    }
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    reset();
    assert!(matches!(quote_lexical(incoming(&root, FIXTURE, 83), working(&root),
        LexicalLimits { max_payload_bytes: required - 1, ..limits() }), Err(Error::Capacity)));
    assert_eq!(factories(), 0);
    // Retain excess working charge; no shrink/refund/reacquire on success.
    let quote = quote_lexical(incoming(&root, FIXTURE, 83),
        root.charge_external(required + 999).unwrap(), limits()).unwrap();
    assert_eq!(quote.resources().working_charge_bytes,
        required + 999 + size_of::<OwnedCharged<LexicalValue>>());
    let output = quote.construct().unwrap();
    assert_eq!(output.charged_bytes(), required + 999 + size_of::<OwnedCharged<LexicalValue>>());
}

#[test]
fn lexical_work_slices_refuse_before_factory() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let quote = quote_lexical(incoming(&root, FIXTURE, 0), working(&root), limits()).unwrap();
    let work = quote.resources().work;
    let counts = *quote.counts();
    // Independently written loop bounds, no production quotation helper oracle.
    let n = counts.spans as u128;
    let d = counts.decoded_string_bytes as u128;
    let l = counts.largest_string as u128;
    assert_eq!(work.replay, counts.scan_work + 2 * n + 134);
    assert_eq!(work.decode, 4 * FIXTURE.len() as u128 + 4 * d + 8 * n + 4);
    assert_eq!(work.classify, counts.number_bytes as u128 + 2 * n);
    assert_eq!(work.deduplicate, n + counts.map_squares * (l + 3) + 2 * counts.map_entries as u128);
    assert_eq!(work.cleanup, 2 * n + d + l + 138);
    assert_eq!(work.total, work.quotation + work.replay + work.decode
        + work.classify + work.deduplicate + work.cleanup);
    drop(quote);
    let baseline = root.budget().used_bytes();
    for slice in 0..7 {
        let mut changed = limits();
        match slice {
            0 => changed.work.quotation = work.quotation - 1,
            1 => changed.work.replay = work.replay - 1,
            2 => changed.work.decode = work.decode - 1,
            3 => changed.work.classify = work.classify - 1,
            4 => changed.work.deduplicate = work.deduplicate - 1,
            5 => changed.work.cleanup = work.cleanup - 1,
            6 => changed.max_work = work.total - 1,
            _ => unreachable!(),
        }
        reset();
        assert!(matches!(quote_lexical(incoming(&root, FIXTURE, 0), working(&root), changed),
            Err(Error::WorkLimit)));
        assert_eq!(factories(), 0);
        assert_eq!(root.budget().used_bytes(), baseline);
    }
    let exact = LexicalLimits {
        max_work: work.total,
        work: LexicalWorkLimits {
            quotation: work.quotation, replay: work.replay, decode: work.decode,
            classify: work.classify, deduplicate: work.deduplicate, cleanup: work.cleanup,
        },
        ..limits()
    };
    let output = quote_lexical(incoming(&root, FIXTURE, 0), working(&root), exact)
        .unwrap().construct().unwrap();
    assert_eq!(output.payload().resources().work, work);
    assert_eq!(NUMBER_CALLS.with(|calls| calls.get()), 0);
    for (raw, changed, expected) in [
        (b"[0,1]".as_slice(), LexicalLimits { max_nodes: 2, ..limits() }, Error::NodeLimit),
        (b"null", LexicalLimits { max_source_bytes: 3, ..limits() }, Error::SourceLimit),
        (br#""\u03bb""#, LexicalLimits { max_string_bytes: 1, ..limits() }, Error::StringLimit),
        (b"1e99", LexicalLimits { max_number_bytes: 3, ..limits() }, Error::NumberLimit),
    ] {
        reset();
        assert!(matches!(quote_lexical(incoming(&root, raw, 0), working(&root), changed),
            Err(error) if error == expected));
        assert_eq!(factories(), 0);
    }
}

#[test]
fn lexical_partial_error_unwind_drop_order() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let baseline = root.budget().used_bytes();
    assert_eq!(baseline, OwnedBudget::resources().unwrap().initial_reserved_bytes);
    reset();
    let quote = quote_lexical(incoming(&root, FIXTURE, 19), working(&root), limits()).unwrap();
    let held = baseline + quote.resources().original_owners_peak_bytes;
    let identity = std::ptr::from_ref(root.budget()) as usize;
    drop(quote.construct().unwrap());
    let sites = factories();
    // Exactly two backing factories plus one per decoded string. No public
    // probe can mint an allocation site or substitute a stand-in custody owner.
    assert_eq!(sites, 9);
    LEXICAL_RETIREMENTS.with(|events| assert_eq!(events.borrow().as_slice(), &[(identity, held)]));
    assert_eq!(root.budget().used_bytes(), baseline);
    for site in 1..=sites {
        for panic in [false, true] {
            reset();
            FACTORIES.with(|state| state.set((0,
                if panic { None } else { Some(site) },
                if panic { Some(site) } else { None })));
            if panic {
                assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _ = quote_lexical(incoming(&root, FIXTURE, 19), working(&root), limits())
                        .unwrap().construct();
                })).is_err());
            } else {
                assert!(matches!(quote_lexical(incoming(&root, FIXTURE, 19), working(&root), limits())
                    .unwrap().construct(), Err(Error::Factory)));
            }
            // Snapshots occur AFTER actual initialized fact buffers drop, while
            // BOTH actual owner guards still charge the exact original counter.
            LEXICAL_RETIREMENTS.with(|events| {
                for &(counter, used) in events.borrow().iter() {
                    assert_eq!((counter, used), (identity, held));
                }
                if site > 1 {
                    assert_eq!(events.borrow().len(), 1);
                }
            });
            // Actual guard retirement, including root's own charged baseline.
            assert_eq!(root.budget().used_bytes(), baseline);
        }
    }
    reset();
    let raw = br#"{"a":"first","\u0061":"second"}"#;
    let quote = quote_lexical(incoming(&root, raw, 19), working(&root), limits()).unwrap();
    let duplicate_held = baseline + quote.resources().original_owners_peak_bytes;
    assert!(matches!(quote.construct(), Err(Error::DuplicateField)));
    LEXICAL_RETIREMENTS.with(|events| assert_eq!(events.borrow().as_slice(), &[(identity, duplicate_held)]));
    assert_eq!(root.budget().used_bytes(), baseline);
    reset();
}

#[test]
fn lexical_foreign_counter_and_short_bootstrap() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let other = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let baseline = root.budget().used_bytes();
    let other_baseline = other.budget().used_bytes();
    reset();
    assert!(matches!(quote_lexical(incoming(&root, FIXTURE, 0), working(&other), limits()),
        Err(Error::Counter)));
    for bytes in [0, lexical_bootstrap_bytes().unwrap() - 1] {
        assert!(matches!(quote_lexical(incoming(&root, FIXTURE, 0),
            root.charge_external(bytes).unwrap(), limits()), Err(Error::Bootstrap)));
    }
    let invalid = root.charge_external(0).unwrap().try_build(|_| Ok(FIXTURE.to_vec())).unwrap();
    assert!(matches!(quote_lexical(invalid, working(&root), limits()), Err(Error::OriginalOwner)));
    assert_eq!(factories(), 0);
    assert_eq!(root.budget().used_bytes(), baseline);
    assert_eq!(other.budget().used_bytes(), other_baseline);
    drop(quote_lexical(incoming(&root, FIXTURE, 0), working(&root), limits()).unwrap());
    assert_eq!(root.budget().used_bytes(), baseline);
}

#[test]
fn lexical_c9_defaults_remain_unchanged() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let ordinary = Limits {
        max_source_bytes: 1 << 20, max_nodes: 100_000, max_depth: 32,
        max_string_bytes: 1 << 16, max_number_bytes: 128, max_number_work: 1 << 26,
        max_work: 1 << 30, max_payload_bytes: MAX_ALLOCATION_BYTES,
    };
    let raw = br#"{"a":1,"\u0061":2,"zero":-0,"float":-0.0}"#;
    reset();
    let value = quote(incoming(&root, raw, 0), root.charge_external(bootstrap_bytes().unwrap()).unwrap(),
        ordinary).unwrap().construct().unwrap();
    assert_eq!(value.payload().value()["a"].as_u64(), Some(2));
    // Ordinary serde's -0 remains f64 negative zero, NOT lexical Integer.
    assert!(value.payload().value()["zero"].is_f64());
    assert_eq!(value.payload().value()["zero"].as_f64().unwrap().to_bits(), (-0.0f64).to_bits());
    assert!(value.payload().value()["float"].is_f64());
    assert_eq!(NUMBER_CALLS.with(|calls| calls.get()), 4);
    for containers in [16, 17] {
        let raw = format!("{}0{}", "[".repeat(containers), "]".repeat(containers));
        let result = quote(incoming(&root, raw.as_bytes(), 0),
            root.charge_external(bootstrap_bytes().unwrap()).unwrap(), ordinary);
        assert_eq!(result.is_ok(), containers == 16);
        if let Ok(quote) = result {
            assert_eq!(quote.counts().depth, 16);
            drop(quote.construct().unwrap());
        }
    }
    // Default plan scanner keeps 32 container slots and its old container-count
    // semantics, independently of the whole-Value census's sixteen-level cap.
    for containers in [32, 33] {
        let raw = format!("{}0{}", "[".repeat(containers), "]".repeat(containers));
        let bytes = incoming(&root, raw.as_bytes(), 0);
        let working = root.charge_external(super::super::bootstrap_bytes().unwrap()).unwrap();
        let result = super::super::quote(&bytes, working, ordinary);
        assert_eq!(result.is_ok(), containers == 32);
        if let Ok(quote) = result {
            assert_eq!(quote.counts().depth, 32);
        }
    }
    reset();
    let quote = quote(incoming(&root, b"1e9999", 0),
        root.charge_external(bootstrap_bytes().unwrap()).unwrap(), ordinary).unwrap();
    assert!(matches!(quote.construct(), Err(Error::InvalidNumber)));
    assert_eq!(NUMBER_CALLS.with(|calls| calls.get()), 1);
}
