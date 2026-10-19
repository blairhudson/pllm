use super::*;
use serde_json::Value;
use sha2::{Digest, Sha256};

const FIXTURE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../paper/reproduction-contracts/fusefss/gate-set/fixtures.json"
));
const SUPPLEMENT: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../paper/reproduction-contracts/fusefss/gate-set/source-tables-supplement.json"
));

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::new(), |mut s, b| {
        write!(s, "{b:02x}").unwrap();
        s
    })
}
fn digest(values: &[u64]) -> String {
    let mut hash = Sha256::new();
    for v in values {
        hash.update(v.to_le_bytes());
    }
    hex(&hash.finalize())
}
fn check(values: &[u64], expected: &Value, expected_hash: &Value) {
    assert_eq!(values.len(), expected.as_array().unwrap().len());
    for (actual, expected) in values.iter().zip(expected.as_array().unwrap()) {
        assert_eq!(Some(*actual), expected.as_u64());
    }
    assert_eq!(digest(values), expected_hash.as_str().unwrap());
}

#[test]
fn complete_native_source_tables_match_frozen_independent_values() {
    assert_eq!(
        hex(&Sha256::digest(FIXTURE)),
        "ae1d8ba29b2563761ce1362688c4ff893812294d76066b0f398fb3388ace2a87"
    );
    let fixture: Value = serde_json::from_slice(FIXTURE).unwrap();
    assert_eq!(fixture["source_lock_sha256"], SOURCE_LOCK_HEX);
    assert_eq!(hex(&source_digest()), SOURCE_LOCK_HEX);
    for (name, values) in [("silu", silu()), ("exp0", exp0()), ("exp1", exp1())] {
        check(
            values,
            &fixture["tables"][name],
            &fixture["sha256_u64le"][name],
        );
    }
    let budget = Budget::new(2 << 20).unwrap();
    for k in [1, 2, 3, 768, 1024] {
        let name = format!("reciprocal-k{k}");
        let values = reciprocal(k, &budget).unwrap();
        check(
            &values,
            &fixture["tables"][&name],
            &fixture["sha256_u64le"][&name],
        );
        assert_eq!(values.len(), 1usize << reciprocal_q(k).unwrap());
        drop(values);
        assert_eq!(budget.used_bytes(), 0);
    }
    for k in [4, 768] {
        let name = format!("rsqrt-k{k}");
        let values = rsqrt(k, &budget).unwrap();
        check(
            &values,
            &fixture["tables"][&name],
            &fixture["sha256_u64le"][&name],
        );
        drop(values);
        assert_eq!(budget.used_bytes(), 0);
    }
    let supplement: Value = serde_json::from_slice(SUPPLEMENT).unwrap();
    assert_eq!(supplement["name"], "rsqrt-k2");
    assert_eq!(supplement["source_lock_sha256"], SOURCE_LOCK_HEX);
    let values = rsqrt(2, &budget).unwrap();
    check(&values, &supplement["values"], &supplement["sha256_u64le"]);
    assert_eq!(values[141], 260_119);
}

#[test]
fn table_generation_preflights_exact_owners_and_retires_on_failure() {
    let bytes = 8192 * 8 + size_of::<Values<'_>>();
    let budget = Budget::new(bytes).unwrap();
    let values = rsqrt(4096, &budget).unwrap();
    assert_eq!(budget.used_bytes(), bytes);
    assert_eq!(values[0], 1 << 30);
    assert!(matches!(reciprocal(1, &budget), Err(Error::Resource)));
    assert_eq!(budget.used_bytes(), bytes);
    drop(values);
    assert_eq!(budget.used_bytes(), 0);
    let short = Budget::new(bytes - 1).unwrap();
    assert!(matches!(rsqrt(4096, &short), Err(Error::Resource)));
    assert_eq!(short.used_bytes(), 0);
    for invalid in [0, 4097, usize::MAX] {
        assert!(matches!(rsqrt(invalid, &budget), Err(Error::Domain)));
    }
    for invalid in [0, 1025, usize::MAX] {
        assert!(matches!(reciprocal(invalid, &budget), Err(Error::Domain)));
    }
    budget
        .fail_after
        .store(0, std::sync::atomic::Ordering::SeqCst);
    assert!(matches!(rsqrt(2, &budget), Err(Error::Allocation)));
    assert_eq!(budget.used_bytes(), 0);
}
