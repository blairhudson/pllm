//! Candidate mount smoke test; the full fixture is registered once, in lib tests.
use pllm_core::coefficient_phase_reference::{MAX_WORK, Quote};

#[test]
fn public_candidate_mount_has_bounded_qualification() {
    assert!(Quote::qualification().unwrap().arithmetic() <= MAX_WORK);
}
