//! Authored source-only controls: ALL TESTS UNRUN; compilation unverified.
//! Static expected bits, never backend or CPython parsing as an expected oracle.

use super::*;
use crate::plan_input::json::{
    lexical_bootstrap_bytes, quote_lexical, LexicalLimits, LexicalWorkLimits,
};
use pllm_garble::shared_allocation::{OwnedBudget, MAX_ALLOCATION_BYTES};

fn limits() -> LexicalLimits {
    // These are existing lexical construction limits ONLY. They do not quote,
    // fund or bound the subsequent UNQUALIFIED std float parser's total work.
    LexicalLimits {
        max_source_bytes: 1 << 16,
        max_nodes: 1024,
        max_depth: 64,
        max_string_bytes: 1 << 16,
        max_number_bytes: 1 << 16,
        max_payload_bytes: MAX_ALLOCATION_BYTES,
        max_work: 1 << 24,
        work: LexicalWorkLimits {
            quotation: 1 << 24,
            replay: 1 << 24,
            decode: 1 << 24,
            classify: 1 << 24,
            deduplicate: 1 << 24,
            cleanup: 1 << 24,
        },
    }
}

fn original(root: &OwnedBudget, raw: &[u8], spare: usize) -> OwnedCharged<Vec<u8>> {
    let capacity = raw.len() + spare;
    root.charge_external(capacity).unwrap().try_build(|_| {
        let mut bytes = Vec::with_capacity(capacity);
        bytes.extend_from_slice(raw);
        assert_eq!(bytes.capacity(), capacity);
        Ok(bytes)
    }).unwrap()
}

fn lexical(root: &OwnedBudget, raw: &[u8]) -> OwnedCharged<LexicalValue> {
    // Real production quote -> construct. No manufactured lexical node/span,
    // borrowed byte surrogate, Rc-only custody or conversion-factory probe.
    quote_lexical(
        original(root, raw, 73),
        root.charge_external(lexical_bootstrap_bytes().unwrap()).unwrap(),
        limits(),
    ).unwrap().construct().unwrap()
}

#[test]
fn float_static_binary64_bits() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    for (lexeme, expected) in [
        ("0.0", 0x0000_0000_0000_0000u64),
        ("-0.0", 0x8000_0000_0000_0000),
        ("5e-324", 0x0000_0000_0000_0001),
        ("2.2250738585072014e-308", 0x0010_0000_0000_0000),
        ("1.7976931348623157e308", 0x7fef_ffff_ffff_ffff),
        ("9007199254740993.0", 0x4340_0000_0000_0000),
        ("9007199254740995.0", 0x4340_0000_0000_0002),
        ("1e309", 0x7ff0_0000_0000_0000),
        ("-1e309", 0xfff0_0000_0000_0000),
        ("-1e-4000", 0x8000_0000_0000_0000),
    ] {
        let owner = lexical(&root, lexeme.as_bytes());
        let converted = parse_unqualified(FloatNode::borrow(&owner, 0).unwrap()).unwrap();
        assert_eq!(*converted.bits(), expected, "{lexeme}");
        assert_eq!(converted.node().node().number(), Some((lexeme, LexicalNumberClass::Float)));
        assert!(std::ptr::eq(converted.node().owner(), &owner));
    }
}

#[test]
fn float_huge_exponent_and_long_mantissa_retain_exact_spans() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    for (raw, lexeme) in [
        (
            b" \t 1e9999999999999999999999999999999999999999 \r\n".as_slice(),
            "1e9999999999999999999999999999999999999999",
        ),
        (
            b" \t 0.123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890 \r\n",
            "0.123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890",
        ),
    ] {
        let owner = lexical(&root, raw);
        let borrowed = FloatNode::borrow(&owner, 0).unwrap();
        assert_eq!(borrowed.node().raw_range(), 3..3 + lexeme.len());
        assert_eq!(borrowed.node().number(), Some((lexeme, LexicalNumberClass::Float)));
        assert_eq!(*borrowed.resources().lexeme_bytes(), lexeme.len());
        let converted = parse_unqualified(borrowed).unwrap();
        assert_eq!(converted.node().node().raw_range(), 3..3 + lexeme.len());
        assert_eq!(converted.node().node().number(), Some((lexeme, LexicalNumberClass::Float)));
        assert_eq!(converted.node().owner().payload().raw(), raw);
        // No expected numeric result derived by another parser for these cases.
    }
}

#[test]
fn float_refuses_integers_non_numbers_and_missing_nodes() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let owner = lexical(&root,
        br#"[0,-0,1,9007199254740993,true,false,null,"1.0",[],{},-0.0]"#);
    // Root array, four Integers, booleans/null, string, empty array/object.
    // Literal node order, with no nested children in the empty containers.
    for index in 0..=10 {
        assert!(matches!(FloatNode::borrow(&owner, index), Err(FloatError::NotFloat)));
    }
    assert_eq!(owner.payload().node(2).unwrap().number(), Some(("-0", LexicalNumberClass::Integer)));
    let converted = parse_unqualified(FloatNode::borrow(&owner, 11).unwrap()).unwrap();
    assert_eq!(*converted.bits(), 0x8000_0000_0000_0000);
    for index in [12, usize::MAX] {
        assert!(matches!(FloatNode::borrow(&owner, index), Err(FloatError::MissingNode)));
    }
}

#[test]
fn float_special_tokens_refused_by_real_lexical_quote() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let baseline = root.budget().used_bytes();
    for raw in [b"NaN".as_slice(), b"Infinity", b"-Infinity", b"+Infinity", b"[NaN]"] {
        let result = quote_lexical(
            original(&root, raw, 73),
            root.charge_external(lexical_bootstrap_bytes().unwrap()).unwrap(),
            limits(),
        );
        assert!(matches!(result, Err(crate::plan_input::Error::Syntax)));
        assert_eq!(root.budget().used_bytes(), baseline);
    }
    // Overflowing valid Float lexemes in the static table are a DIFFERENT case;
    // their unqualified infinity bits do not admit special tokens or exports.
}

#[test]
fn float_actual_owner_capacity_counter_success_refusal_and_drop() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let baseline = root.budget().used_bytes();
    assert_eq!(baseline, OwnedBudget::resources().unwrap().initial_reserved_bytes);
    let owner = lexical(&root, b"[1.0,0]");
    let raw_pointer = owner.payload().raw().as_ptr();
    let capacity = owner.payload().raw_owner().payload().capacity();
    let identity = std::ptr::from_ref(root.budget());
    let retained = root.budget().used_bytes();
    assert_eq!(capacity, 7 + 73);
    for _ in 0..2 {
        // Repeated pure conversion is allowed; NOT a one-shot attempt/root gate.
        let borrowed = FloatNode::borrow(&owner, 1).unwrap();
        assert!(std::ptr::eq(borrowed.owner(), &owner));
        assert_eq!(*borrowed.resources().lexeme_bytes(), 3);
        assert!(borrowed.resources().total_work().is_none());
        assert!(borrowed.resources().total_auxiliary_bytes().is_none());
        assert!(borrowed.resources().total_stack_bytes().is_none());
        let converted = parse_unqualified(borrowed).unwrap();
        assert_eq!(*converted.bits(), 0x3ff0_0000_0000_0000);
        assert_eq!(converted.node().owner().payload().raw().as_ptr(), raw_pointer);
        assert_eq!(converted.node().owner().payload().raw_owner().payload().capacity(), capacity);
        assert_eq!(std::ptr::from_ref(converted.node().owner().budget()), identity);
        assert_eq!(std::ptr::from_ref(converted.node().owner().payload().raw_owner().budget()), identity);
        // Original logical charges stay intact; this does NOT observe or prove
        // anything about std-parser allocations, total work or native stack.
        assert_eq!(root.budget().used_bytes(), retained);
    }
    assert!(matches!(FloatNode::borrow(&owner, 2), Err(FloatError::NotFloat)));
    assert!(matches!(FloatNode::borrow(&owner, 3), Err(FloatError::MissingNode)));
    assert_eq!(owner.payload().raw().as_ptr(), raw_pointer);
    assert_eq!(owner.payload().raw_owner().payload().capacity(), capacity);
    assert_eq!(root.budget().used_bytes(), retained);
    drop(owner);
    assert_eq!(root.budget().used_bytes(), baseline);
}

#[test]
fn float_existing_scalar_value_profile_is_not_python_domain_approval() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let owner = lexical(&root, br#"{"s":"\ud83e\udd80","f":1.0}"#);
    // Existing paired-surrogate decoding is usable; no blanket rejection of all
    // surrogate escape bytes or claim about Python's original input domain.
    assert_eq!(owner.payload().node(2).unwrap().decoded_string(), Some("🦀"));
    let converted = parse_unqualified(FloatNode::borrow(&owner, 4).unwrap()).unwrap();
    assert_eq!(*converted.bits(), 0x3ff0_0000_0000_0000);
    for raw in [br#"["\ud800",1.0]"#.as_slice(), br#"["\udc00",1.0]"#] {
        assert!(matches!(quote_lexical(
            original(&root, raw, 73),
            root.charge_external(lexical_bootstrap_bytes().unwrap()).unwrap(),
            limits(),
        ), Err(crate::plan_input::Error::Syntax)));
    }
}
