//! Definitions only; not executed. No funded permit fixture or cfg(test) issuer.
use super::*;

#[test]
fn unpriced_quotes_return_refusal() {
    assert!(matches!(original_source::quote_metadata_entry(), Err(original_source::Error::Quote)));
    assert!(pllm_models::source_work::extended_metadata_quote().is_err());
    // These assertions check refusal results only. Pre-issuance ordering is a
    // separate static-source finding; allocation/query/lifecycle behavior is not
    // instrumented here.
}

#[test]
fn unknown_complete_controls_returns_bounds() {
    assert!(matches!(metadata::complete_controls(), Err(Error::Bounds)));
}

// Deferred—not implemented or executed tests; genuine funded issuance and separate authorization required
// 1. Genuine Context mismatch despite equal counters: two original roots and a
//    genuinely issued permit, not a fabricated permit fixture.
// 2. Nonreissuance/reselection on the same root and cumulative-ledger observation.
// 3. Both checker quotes, caller pool initialization, copy and retirement prepaid:
//    independent SOURCEEVENT inventory and extended body price still missing.
// 4. Body-work/full-peak shortages refuse before allocation: meaningful boundary
//    injection is unavailable behind the unpriced entry.
// 5. Primary failure preservation and sticky charge/burn, including partial
//    Vec/String construction failures under genuine funded admission.
// 6. Records, then permit, then geometry retirement observations on genuine owners.
// 7. Borrowed-name copy, deduplication and conflict behavior through the candidate.
// 8. Unchanged default quotes, owner layout and refusal: existing
//    source_geometry_tests.rs owns actual default regression coverage; no new
//    default test or matching execution is supplied here.
