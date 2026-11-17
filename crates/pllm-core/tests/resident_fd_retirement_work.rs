//! Quote arithmetic only: no registry preparation, lock, lease, or descriptor.
#![cfg(unix)]

use pllm_core::resident_fd::retirement_work::{quote, QuoteError, RetirementEntry, SourceProfile};

#[test]
fn unqualified_profile_always_rejected() {
    for entry in [RetirementEntry::Borrowed, RetirementEntry::Consuming] {
        assert_eq!(
            quote(SourceProfile::Unqualified, entry),
            Err(QuoteError::UnsupportedProfile)
        );
    }
}

#[test]
fn declared_profile_has_only_target_conditional_integer_envelope() {
    for entry in [RetirementEntry::Borrowed, RetirementEntry::Consuming] {
        let result = quote(SourceProfile::Rust1851Mac64WarmSystemNoRetag, entry);
        // Integration tests compile the library WITHOUT cfg(test). This checks
        // arithmetic under a declaration, not compiler/allocator/warmth proof.
        if cfg!(all(
            target_os = "macos",
            target_pointer_width = "64",
            not(miri)
        )) {
            let value = result.unwrap();
            assert!(value.helper_calls > 0 && value.helper_calls < 10_000);
            assert!(value.visits > 0 && value.visits < 100_000);
            assert!(value.potential_bytes > 0 && value.potential_bytes < 1_000_000);
            let total = value
                .helper_calls
                .checked_add(value.visits)
                .unwrap()
                .checked_add(value.potential_bytes)
                .unwrap();
            assert_eq!(value.total_work(), Ok(total));
        } else {
            assert_eq!(result, Err(QuoteError::UnsupportedProfile));
        }
    }
}
