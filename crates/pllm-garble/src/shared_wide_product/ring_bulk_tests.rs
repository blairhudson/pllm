use super::*;
use crate::{
    fusefss_backend::{conversions, Budget, Error as NativeError},
    shared_rescale_reference::Context,
};
use std::{cell::RefCell, panic::AssertUnwindSafe};

pub(super) fn issue_with(
    bits: u8,
    mut draw: impl FnMut() -> Result<u64, Error>,
) -> Result<[TripleShare; 2], Error> {
    issue_from_fill(bits, |bytes| {
        for chunk in bytes.chunks_exact_mut(8) {
            chunk.copy_from_slice(&draw()?.to_le_bytes());
        }
        Ok(())
    })
}

// Thread-local, test-only injection at the real OS source boundary. The guard
// restores the source even on unwind; other tests/threads keep using getrandom.
thread_local! {
    static SOURCE: RefCell<Option<Source>> = const { RefCell::new(None) };
}
struct Source {
    calls: Vec<usize>,
    fail_call: Option<usize>,
    prefix: usize,
    panic: bool,
}
struct SourceGuard;
impl SourceGuard {
    fn install(fail_call: Option<usize>, prefix: usize, panic: bool) -> Self {
        // Reject same-thread joint controllers BEFORE this source is modified.
        assert!(!crate::shared_affine_lookup_reference::joint_basis_test_hooks::active());
        assert!(!crate::shared_affine_lookup_reference::joint_basis::test_source_active());
        SOURCE.with(|source| {
            assert!(source.borrow().is_none());
            *source.borrow_mut() = Some(Source {
                calls: Vec::new(),
                fail_call,
                prefix,
                panic,
            });
        });
        Self
    }
    fn calls(&self) -> Vec<usize> {
        SOURCE.with(|source| source.borrow().as_ref().unwrap().calls.clone())
    }
}
impl Drop for SourceGuard {
    fn drop(&mut self) {
        SOURCE.with(|source| *source.borrow_mut() = None);
    }
}
pub(super) fn source_active() -> bool {
    SOURCE.with(|source| source.try_borrow().map_or(true, |source| source.is_some()))
}
pub(super) fn fill_override(bytes: &mut [u8]) -> Option<Result<(), Error>> {
    SOURCE.with(|source| {
        let mut source = source.borrow_mut();
        let source = source.as_mut()?;
        let call = source.calls.len();
        source.calls.push(bytes.len());
        assert!(bytes.iter().all(|&byte| byte == 0));
        if source.fail_call == Some(call) {
            bytes[..source.prefix].fill(0xa5);
            assert!(!source.panic, "injected partial-fill unwind");
            Some(Err(Error::Randomness))
        } else {
            // Exact test bytes, not a replacement RNG or seed expander.
            bytes.fill(0x39);
            Some(Ok(()))
        }
    })
}

fn encode(words: [u64; 5]) -> [u8; ENTROPY_BYTES] {
    let mut bytes = [0; ENTROPY_BYTES];
    for (chunk, word) in bytes.chunks_exact_mut(8).zip(words) {
        chunk.copy_from_slice(&word.to_le_bytes());
    }
    bytes
}

// Independent pre-change reference: five source calls, u128 modular arithmetic,
// no shared decoder, TripleShare operations, wrapping arithmetic or mask helper.
fn reference(bits: u8, mut fill: impl FnMut(&mut [u8]) -> Result<(), Error>) -> [[u64; 3]; 2] {
    let modulus = 1u128 << bits;
    let mut words = [0u128; 5];
    for word in &mut words {
        let mut bytes = [0; 8];
        fill(&mut bytes).unwrap();
        *word = u128::from(u64::from_le_bytes(bytes)) % modulus;
    }
    let [a, b, x, y, z] = words;
    [
        [x as u64, y as u64, z as u64],
        [
            ((a + modulus - x) % modulus) as u64,
            ((b + modulus - y) % modulus) as u64,
            (((a * b) % modulus + modulus - z) % modulus) as u64,
        ],
    ]
}

#[test]
fn five_source_calls_become_one_with_identical_material_and_party_transcripts() {
    let bytes = encode([
        0xfedc_ba98_7654_3210,
        0x0123_4567_89ab_cdef,
        0xffee_ddcc_bbaa_9988,
        0x7766_5544_3322_1100,
        0x89ab_cdef_0123_4567,
    ]);
    for bits in 1..=64 {
        let mut old_calls = Vec::new();
        let mut offset = 0;
        let expected = reference(bits, |out| {
            let len = out.len();
            old_calls.push(len);
            out.copy_from_slice(&bytes[offset..offset + len]);
            offset += len;
            Ok(())
        });
        let mut new_calls = Vec::new();
        let [left, right] = issue_from_fill(bits, |out| {
            new_calls.push(out.len());
            out.copy_from_slice(&bytes);
            Ok(())
        })
        .unwrap();
        assert_eq!(old_calls, [8; 5]);
        assert_eq!(new_calls, [40]);
        let modulus = 1u128 << bits;
        let operands = [[modulus - 1, modulus / 2], [modulus / 2, 1]];
        let openings: [[u64; 2]; 2] = std::array::from_fn(|party| {
            std::array::from_fn(|coordinate| {
                ((operands[party][coordinate] + modulus - u128::from(expected[party][coordinate]))
                    % modulus) as u64
            })
        });
        let d = (u128::from(openings[0][0]) + u128::from(openings[1][0])) % modulus;
        let e = (u128::from(openings[0][1]) + u128::from(openings[1][1])) % modulus;
        let mut outputs = [0; 2];
        for (party, triple) in [left, right].into_iter().enumerate() {
            let mut material = Vec::new();
            triple.write_material(&mut material);
            let expected_bytes: Vec<u8> = expected[party]
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect();
            assert_eq!(material, expected_bytes, "bits={bits}, party={party}");
            let own = triple
                .open(operands[party][0] as u64, operands[party][1] as u64)
                .unwrap();
            assert_eq!(*own, openings[party]);
            let expected_output = (u128::from(expected[party][2])
                + d * u128::from(expected[party][1]) % modulus
                + e * u128::from(expected[party][0]) % modulus
                + if party == 0 { d * e % modulus } else { 0 })
                % modulus;
            outputs[party] = triple
                .finish(party as u8, &own, &openings[1 - party])
                .unwrap();
            assert_eq!(u128::from(outputs[party]), expected_output);
        }
        let x = (operands[0][0] + operands[1][0]) % modulus;
        let y = (operands[0][1] + operands[1][1]) % modulus;
        assert_eq!(
            (u128::from(outputs[0]) + u128::from(outputs[1])) % modulus,
            x * y % modulus
        );
    }
}

#[test]
fn exhaustive_small_ring_joint_distribution_is_unchanged_for_each_party() {
    for bits in 1..=2 {
        let n = 1usize << bits;
        // Condition on a,b: every possible party-local triple occurs exactly
        // once among the n^3 independent left-share choices, for BOTH parties.
        for a in 0..n {
            for b in 0..n {
                let mut counts = [vec![0; n * n * n], vec![0; n * n * n]];
                for x in 0..n {
                    for y in 0..n {
                        for z in 0..n {
                            let bytes = encode([a as u64, b as u64, x as u64, y as u64, z as u64]);
                            let shares = issue_from_fill(bits, |out| {
                                out.copy_from_slice(&bytes);
                                Ok(())
                            })
                            .unwrap();
                            for (party, share) in shares.into_iter().enumerate() {
                                let [x, y, z] = *share.words;
                                counts[party][(x as usize * n + y as usize) * n + z as usize] += 1;
                            }
                        }
                    }
                }
                assert!(counts.iter().flatten().all(|&count| count == 1));
            }
        }
    }
}

fn spec(lanes: usize) -> conversions::Spec {
    conversions::Spec {
        input_bits: 50,
        output_bits: 50,
        lanes,
        operation: conversions::Operation::Multiply,
        source_digest: [1; 32],
        profile_digest: [2; 32],
    }
}
fn context() -> Context {
    Context {
        plan: [3; 32],
        session: [4; 32],
        operation: [5; 32],
        first_tensor_index: 0,
    }
}

#[test]
fn geometry_reservation_and_allocation_failures_precede_triple_entropy() {
    let source = SourceGuard::install(Some(0), 0, false);
    for bits in [0, 65, 255] {
        assert!(matches!(issue(bits), Err(Error::Domain)));
    }
    let valid = spec(1);
    let quote = conversions::resources(valid).unwrap();
    let budget = Budget::new(quote.combined_reservation_bytes - 1).unwrap();
    assert!(matches!(
        conversions::issue(spec(0), context(), &budget),
        Err(NativeError::Domain)
    ));
    let mut overflow = context();
    overflow.first_tensor_index = u64::MAX;
    assert!(matches!(
        conversions::issue(valid, overflow, &budget),
        Err(NativeError::Domain)
    ));
    assert!(matches!(
        conversions::issue(valid, context(), &budget),
        Err(NativeError::Resource)
    ));
    assert_eq!(budget.used_bytes(), 0);
    let budget = Budget::new(quote.combined_reservation_bytes).unwrap();
    for allocation in [0, 1] {
        budget
            .fail_after
            .store(allocation, std::sync::atomic::Ordering::SeqCst);
        assert!(matches!(
            conversions::issue(valid, context(), &budget),
            Err(NativeError::Allocation)
        ));
        assert_eq!(budget.used_bytes(), 0);
    }
    assert!(source.calls().is_empty());
}

#[test]
fn partial_fill_failure_and_unwind_burn_whole_issuance_and_refund_prior_lanes() {
    let valid = spec(2);
    let quote = conversions::resources(valid).unwrap();
    for panic in [false, true] {
        for prefix in 0..=ENTROPY_BYTES {
            // Fail on lane two: lane one's material must also be dropped.
            let source = SourceGuard::install(Some(1), prefix, panic);
            let budget = Budget::new(quote.combined_reservation_bytes).unwrap();
            let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                conversions::issue(valid, context(), &budget)
            }));
            if panic {
                assert!(result.is_err());
            } else {
                assert!(matches!(result, Ok(Err(NativeError::Randomness))));
            }
            assert_eq!(source.calls(), [40, 40]); // no retry or partial publication
            assert_eq!(budget.used_bytes(), 0);
        }
    }
}

#[test]
fn real_native_issuance_uses_one_fill_per_lane_and_retains_cancel_replay_burns() {
    let source = SourceGuard::install(None, 0, false);
    let valid = spec(2);
    let quote = conversions::resources(valid).unwrap();
    // Existing 4096-byte dealer scratch includes the entire 136-byte bulk
    // primitive overlap. No extra heap allocation, per-lane scratch multiplier,
    // cap relaxation, or change in persistent private material is needed.
    assert!(quote.combined_reservation_bytes >= 2 * quote.party_reservation_bytes + 4096);
    assert_eq!(ISSUANCE_SCRATCH_BYTES, 136);
    assert_eq!(quote.private_material_bytes_per_party, 2 * TRIPLE_BYTES);
    let budget = Budget::new(quote.combined_reservation_bytes).unwrap();
    let [mut left, mut right] = conversions::issue(valid, context(), &budget).unwrap();
    assert_eq!(source.calls(), [40, 40]);
    assert_eq!(budget.used_bytes(), 2 * quote.party_reservation_bytes);
    let own = left.start_product(&[1, 2], &[3, 4]).unwrap();
    let peer = right.start_product(&[5, 6], &[7, 8]).unwrap();
    // Self-role input burns left rather than accepting a second opening.
    assert!(matches!(
        left.finish(own.as_bytes()),
        Err(NativeError::Context)
    ));
    assert!(matches!(
        left.finish(peer.as_bytes()),
        Err(NativeError::Consumed)
    ));
    right.cancel();
    assert!(matches!(
        right.start_product(&[5, 6], &[7, 8]),
        Err(NativeError::Consumed)
    ));
    drop((own, peer, left, right));
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn joint_and_triple_controllers_reject_conflicts_before_modifying_either() {
    use crate::{shared_affine_lookup_reference::joint_basis_test_hooks as hooks,
        shared_allocation::{OwnedBudget, MAX_ALLOCATION_BYTES}};
    let owned = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let (identity, bootstrap) = owned.budget().test_snapshot();
    let weak = owned.test_counter_weak();
    let config = || hooks::Config { counter: owned.test_counter_weak(),
        expected_identity: identity, expected_charge: bootstrap,
        expected_updates: owned.budget().test_accounting_updates(), fault: None };
    {
        let triple = SourceGuard::install(None, 0, false);
        hooks::assert_install_conflict(std::panic::catch_unwind(AssertUnwindSafe(|| hooks::install(config()))),
            "assertion failed: !crate::shared_wide_product::ring::test_source_active()");
        assert!(!hooks::active());
        assert!(source_active());
        assert!(triple.calls().is_empty());
        // Existing triple source still works. No joint observation/admission
        // claim is made for this standalone primitive issuance.
        drop(issue(50).unwrap());
        assert_eq!(triple.calls(), [40]);
    }
    {
        let guard = hooks::install(config());
        hooks::assert_install_conflict(std::panic::catch_unwind(|| SourceGuard::install(None, 0, false)),
            "assertion failed: !crate::shared_affine_lookup_reference::joint_basis_test_hooks::active()");
        assert!(!source_active());
        assert!(hooks::active());
        assert!(!guard.snapshot().failed);
        assert_eq!(guard.snapshot().len, 0);
    }
    assert_eq!(owned.budget().used_bytes(), bootstrap);
    let quote = OwnedBudget::resources().unwrap();
    let fixed = bootstrap.checked_sub(quote.owner_metadata_bytes).unwrap();
    assert_eq!(fixed, quote.arc_backing_bytes.checked_add(quote.bootstrap_transition_bytes).unwrap());
    let observer = weak.upgrade().unwrap();
    drop(owned);
    assert_eq!(observer.used_bytes(), fixed);
    drop(observer);
    assert!(weak.upgrade().is_none());
}
