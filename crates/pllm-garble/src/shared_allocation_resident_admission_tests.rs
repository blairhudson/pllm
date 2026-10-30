//! Authored regression oracles only; execution/review remains a separate gate.
use super::*;
use std::cell::Cell;

fn baseline() -> usize {
    OwnedBudget::resources().unwrap().initial_reserved_bytes
}

fn snapshot(root: &OwnedBudget) -> (usize, usize, usize, usize) {
    (
        root.budget().used_bytes(),
        root.budget().admitted_peak_bytes(),
        root.budget().test_accounting_updates(),
        Arc::strong_count(&root.budget),
    )
}

fn resident<T>(slot: &ResidentAdmission<T>) -> (u8, usize, usize, usize, usize) {
    match &slot.state {
        ResidentAdmissionState::Empty => (0, 0, 0, 0, 0),
        ResidentAdmissionState::Pending { charge, backing_bytes, quoted_bytes } => (
            1, Arc::as_ptr(&charge.budget) as usize, charge.bytes,
            *backing_bytes, *quoted_bytes,
        ),
        ResidentAdmissionState::Admitted { charge, backing_bytes, quoted_bytes } => (
            2, Arc::as_ptr(&charge.budget) as usize, charge.bytes,
            *backing_bytes, *quoted_bytes,
        ),
        ResidentAdmissionState::Sealed => (3, 0, 0, 0, 0),
    }
}

#[test]
fn empty_and_abandoned_handle_do_not_touch_accounting() {
    let root = OwnedBudget::new(baseline() + 1024).unwrap();
    let before = snapshot(&root);
    let mut slot = ResidentAdmission::<u64>::empty();
    {
        let _attempt = slot.attempt();
    }
    assert_eq!(resident(&slot), (0, 0, 0, 0, 0));
    assert_eq!(snapshot(&root), before);
    drop(slot);
    assert_eq!(snapshot(&root), before);
}

#[test]
fn enter_moves_exact_guard_without_clone_and_abandonment_retains_it() {
    let root = OwnedBudget::new(baseline() + 1024).unwrap();
    let mut source = Some(root.charge_external(19).unwrap());
    let identity = Arc::as_ptr(&source.as_ref().unwrap().budget) as usize;
    let (s, q) = ResidentAdmission::<u64>::quote(19).unwrap();
    let before = snapshot(&root);
    let mut slot = ResidentAdmission::empty();
    ResidentAdmission::<u64>::enter_into(&mut source, 19, &mut slot).unwrap();
    assert!(source.is_none());
    let retained = (1, identity, s, 19, q);
    assert_eq!(resident(&slot), retained);
    {
        let _attempt = slot.attempt();
    }
    assert_eq!(resident(&slot), retained);
    assert_eq!(snapshot(&root), before);
    // No backing exists in this numeric-only test; retirement duty is vacuous.
    drop(slot);
    assert_eq!(root.budget().used_bytes(), baseline());
    assert_eq!(Arc::strong_count(&root.budget), 1);
}

#[test]
fn entry_rejections_preserve_sources_slot_and_all_counters() {
    let root = OwnedBudget::new(baseline() + 4096).unwrap();
    let mut slot = ResidentAdmission::<u64>::empty();
    let mut missing = None;
    let before = snapshot(&root);
    assert_eq!(ResidentAdmission::enter_into(&mut missing, 0, &mut slot), Err(Error::Resource));
    assert_eq!(snapshot(&root), before);
    // These entry calls prove rejection retains owners/accounting. Overflow
    // requests also mismatch these small guards, so they do NOT isolate quote
    // arithmetic or prove rejection precedence from the identical error alone.
    // Direct private-helper boundary tests below complement this retention test.
    for (actual, requested) in [(7, 8), (9, 8), (8, usize::MAX),
        (8, usize::MAX - size_of::<OwnedExternalCharge>())] {
        let mut source = Some(root.charge_external(actual).unwrap());
        let before = snapshot(&root);
        let bytes = source.as_ref().unwrap().bytes;
        let ptr = Arc::as_ptr(&source.as_ref().unwrap().budget);
        assert_eq!(ResidentAdmission::enter_into(&mut source, requested, &mut slot), Err(Error::Resource));
        assert_eq!(source.as_ref().unwrap().bytes, bytes);
        assert_eq!(Arc::as_ptr(&source.as_ref().unwrap().budget), ptr);
        assert_eq!(resident(&slot), (0, 0, 0, 0, 0));
        assert_eq!(snapshot(&root), before);
    }
}

#[test]
fn checked_quote_rejects_backing_plus_metadata_overflow() {
    let metadata = size_of::<OwnedExternalCharge>();
    assert!(metadata > 0);
    // Scalar arithmetic only: no guard, counter mutation or huge allocation.
    let largest_source_backing = usize::MAX - metadata;
    assert_eq!(largest_source_backing.checked_add(metadata), Some(usize::MAX));
    assert_eq!(
        ResidentAdmission::<()>::quote(largest_source_backing + 1),
        Err(Error::Resource),
    );
    assert_eq!(ResidentAdmission::<()>::quote(usize::MAX), Err(Error::Resource));
}

#[test]
fn checked_quote_inline_overflow_and_positive_scalar_boundaries() {
    type Payload = [u64; 4];
    let metadata = size_of::<OwnedExternalCharge>();
    let inline = size_of::<OwnedCharged<Payload>>().checked_sub(metadata).unwrap();
    assert!(inline > 0);
    let largest_source_backing = usize::MAX - metadata;
    assert_eq!(largest_source_backing.checked_add(metadata), Some(usize::MAX));
    // The first addition fits exactly; only adding inline storage overflows.
    assert_eq!(
        ResidentAdmission::<Payload>::quote(largest_source_backing),
        Err(Error::Resource),
    );
    let largest_full_backing = largest_source_backing - inline;
    assert!(largest_full_backing > 0);
    assert_eq!(
        ResidentAdmission::<Payload>::quote(largest_full_backing),
        Ok((usize::MAX - inline, usize::MAX)),
    );
    assert_eq!(
        ResidentAdmission::<Payload>::quote(largest_full_backing - 1),
        Ok((usize::MAX - inline - 1, usize::MAX - 1)),
    );
    assert_eq!(
        ResidentAdmission::<Payload>::quote(largest_full_backing + 1),
        Err(Error::Resource),
    );
    assert_eq!(
        ResidentAdmission::<Payload>::quote(1),
        Ok((metadata + 1, metadata + 1 + inline)),
    );
}

#[test]
fn occupied_pending_and_admitted_entry_never_replace_guard() {
    let root = OwnedBudget::new(baseline() + 4096).unwrap();
    let mut source = Some(root.charge_external(8).unwrap());
    let mut slot = ResidentAdmission::<u64>::empty();
    ResidentAdmission::enter_into(&mut source, 8, &mut slot).unwrap();
    for admitted in [false, true] {
        if admitted {
            slot.attempt().admit_inline().unwrap();
        }
        let mut other = Some(root.charge_external(11).unwrap());
        let before = snapshot(&root);
        let state = resident(&slot);
        let other_bytes = other.as_ref().unwrap().bytes;
        let other_counter = Arc::as_ptr(&other.as_ref().unwrap().budget);
        // Every non-state precondition is valid: removing the occupied-state
        // check would move this source and replace the existing resident guard.
        assert_eq!(ResidentAdmission::enter_into(&mut other, 11, &mut slot), Err(Error::Resource));
        assert_eq!(other.as_ref().unwrap().bytes, other_bytes);
        assert_eq!(Arc::as_ptr(&other.as_ref().unwrap().budget), other_counter);
        assert_eq!(resident(&slot), state);
        assert_eq!(snapshot(&root), before);
        // Additional combined-invalid retention checks, not order proofs.
        assert_eq!(ResidentAdmission::enter_into(&mut other, usize::MAX, &mut slot), Err(Error::Resource));
        assert_eq!(other.as_ref().unwrap().payload_bytes(), 11);
        assert_eq!(ResidentAdmission::enter_into(&mut source, usize::MAX, &mut slot), Err(Error::Resource));
        assert_eq!(resident(&slot), state);
        assert_eq!(snapshot(&root), before);
    }
}

#[test]
fn positive_inline_increment_is_once_and_admitted_abandonment_is_inert() {
    let root = OwnedBudget::new(baseline() + 4096).unwrap();
    let mut source = Some(root.charge_external(13).unwrap());
    let mut slot = ResidentAdmission::<[u64; 3]>::empty();
    let (s, q) = ResidentAdmission::<[u64; 3]>::quote(13).unwrap();
    assert!(q > s);
    ResidentAdmission::enter_into(&mut source, 13, &mut slot).unwrap();
    let before = snapshot(&root);
    slot.attempt().admit_inline().unwrap();
    let after = snapshot(&root);
    assert_eq!(after, (before.0 + q - s, before.0 + q - s, before.2 + 1, before.3));
    let state = resident(&slot);
    for _ in 0..3 {
        slot.attempt().admit_inline().unwrap();
    }
    assert_eq!(resident(&slot), state);
    assert_eq!(state.0, 2);
    assert_eq!(state.2, q);
    assert_eq!(snapshot(&root), after);
}

#[test]
fn failed_pending_reservation_retries_only_original_counter_after_real_release() {
    let (s, q) = ResidentAdmission::<[u64; 4]>::quote(3).unwrap();
    let delta = q - s;
    assert!(delta > 0);
    let root = OwnedBudget::new(baseline() + q).unwrap();
    let blocker = root.budget().reserve(delta).unwrap();
    let mut source = Some(root.charge_external(3).unwrap());
    let mut slot = ResidentAdmission::<[u64; 4]>::empty();
    ResidentAdmission::enter_into(&mut source, 3, &mut slot).unwrap();
    let state = resident(&slot);
    let before = snapshot(&root);
    for _ in 0..2 {
        assert_eq!(slot.attempt().admit_inline(), Err(Error::Resource));
        assert_eq!(resident(&slot), state);
        assert_eq!(snapshot(&root), before);
    }
    drop(blocker);
    let released = snapshot(&root);
    assert_eq!(released.0, baseline() + s);
    slot.attempt().admit_inline().unwrap();
    assert_eq!(resident(&slot), (2, state.1, q, 3, q));
    assert_eq!(snapshot(&root), (baseline() + q, before.1, released.2 + 1, before.3));
}

#[test]
fn empty_and_pending_seal_refuse_without_payload_or_output_mutation() {
    let root = OwnedBudget::new(baseline() + 4096).unwrap();
    let mut slot = ResidentAdmission::<String>::empty();
    let mut payload = Some(String::from("whole payload"));
    let ptr = payload.as_ref().unwrap().as_ptr();
    let mut output = None;
    let before = snapshot(&root);
    assert_eq!(slot.attempt().admit_inline(), Err(Error::Resource));
    assert_eq!(slot.attempt().seal_into(&mut payload, &mut output), Err(Error::Resource));
    assert_eq!(snapshot(&root), before);
    let b = payload.as_ref().unwrap().capacity();
    let mut source = Some(root.charge_external(b).unwrap());
    ResidentAdmission::enter_into(&mut source, b, &mut slot).unwrap();
    let state = resident(&slot);
    let before = snapshot(&root);
    assert_eq!(slot.attempt().seal_into(&mut payload, &mut output), Err(Error::Resource));
    assert_eq!(resident(&slot), state);
    assert_eq!(payload.as_ref().unwrap().as_ptr(), ptr);
    assert_eq!(payload.as_ref().unwrap(), "whole payload");
    assert!(output.is_none());
    assert_eq!(snapshot(&root), before);
    drop(payload); // Actual backing retires before resident guard.
    drop(slot);
}

#[test]
fn seal_rejects_occupied_output_and_missing_payload_without_mutation() {
    let root = OwnedBudget::new(baseline() + 4096).unwrap();
    let mut source = Some(root.charge_external(0).unwrap());
    let mut slot = ResidentAdmission::<u64>::empty();
    ResidentAdmission::enter_into(&mut source, 0, &mut slot).unwrap();
    slot.attempt().admit_inline().unwrap();
    let mut output = Some(root.charge_external(0).unwrap().try_build(|_| Ok(71u64)).unwrap());
    let mut payload = Some(23);
    let state = resident(&slot);
    let before = snapshot(&root);
    assert_eq!(slot.attempt().seal_into(&mut payload, &mut output), Err(Error::Resource));
    assert_eq!(payload, Some(23));
    assert_eq!(*output.as_ref().unwrap().payload(), 71);
    assert_eq!(resident(&slot), state);
    assert_eq!(snapshot(&root), before);
    drop(output.take());
    let mut missing = None;
    let before = snapshot(&root);
    assert_eq!(slot.attempt().seal_into(&mut missing, &mut output), Err(Error::Resource));
    assert!(missing.is_none() && output.is_none());
    assert_eq!(resident(&slot), state);
    assert_eq!(snapshot(&root), before);
}

#[test]
fn whole_nonclone_payload_seals_without_accounting_and_sealed_reentry_refuses() {
    struct Payload(Box<[u8]>); // Intentionally no Clone implementation.
    let root = OwnedBudget::new(baseline() + 4096).unwrap();
    // Persistent resident precedes caller backing: before successful sealing,
    // assertion unwind drops backing first. The API does not enforce this duty.
    let mut slot = ResidentAdmission::<Payload>::empty();
    let mut source = Some(root.charge_external(3).unwrap());
    ResidentAdmission::enter_into(&mut source, 3, &mut slot).unwrap();
    slot.attempt().admit_inline().unwrap();
    let mut payload = Some(Payload(vec![3, 5, 8].into_boxed_slice()));
    let ptr = payload.as_ref().unwrap().0.as_ptr();
    let mut output = None;
    let before = snapshot(&root);
    slot.attempt().seal_into(&mut payload, &mut output).unwrap();
    assert!(payload.is_none() && source.is_none());
    assert_eq!(resident(&slot), (3, 0, 0, 0, 0));
    assert_eq!(snapshot(&root), before);
    let owner = output.as_ref().unwrap();
    assert_eq!(owner.payload().0.as_ref(), &[3, 5, 8]);
    assert_eq!(owner.payload().0.as_ptr(), ptr);
    assert!(std::ptr::eq(owner.budget(), root.budget()));
    assert_eq!(owner.charged_bytes(), ResidentAdmission::<Payload>::quote(3).unwrap().1);
    assert_eq!(slot.attempt().admit_inline(), Err(Error::Resource));
    assert_eq!(snapshot(&root), before);
    // Combined-invalid retention case; the isolated Sealed cases follow.
    assert_eq!(slot.attempt().seal_into(&mut payload, &mut output), Err(Error::Resource));
    assert_eq!(ResidentAdmission::enter_into(&mut source, 3, &mut slot), Err(Error::Resource));
    assert_eq!(snapshot(&root), before);

    // New exact guard precedes its candidate backing, including on unwind.
    let mut new_source = Some(root.charge_external(3).unwrap());
    let source_bytes = new_source.as_ref().unwrap().bytes;
    let source_counter = Arc::as_ptr(&new_source.as_ref().unwrap().budget);
    let mut candidate = Some(Payload(vec![13, 21, 34].into_boxed_slice()));
    let candidate_ptr = candidate.as_ref().unwrap().0.as_ptr();
    let mut empty_output = None;
    let with_candidate = snapshot(&root);
    // Present payload and empty output isolate the Sealed-state refusal.
    assert_eq!(slot.attempt().seal_into(&mut candidate, &mut empty_output), Err(Error::Resource));
    assert!(empty_output.is_none());
    assert_eq!(candidate.as_ref().unwrap().0.as_ref(), &[13, 21, 34]);
    assert_eq!(candidate.as_ref().unwrap().0.as_ptr(), candidate_ptr);
    assert_eq!(resident(&slot), (3, 0, 0, 0, 0));
    assert_eq!(snapshot(&root), with_candidate);
    // Present exact source and valid B isolate the Sealed entry refusal.
    assert_eq!(ResidentAdmission::enter_into(&mut new_source, 3, &mut slot), Err(Error::Resource));
    assert_eq!(new_source.as_ref().unwrap().bytes, source_bytes);
    assert_eq!(Arc::as_ptr(&new_source.as_ref().unwrap().budget), source_counter);
    assert_eq!(resident(&slot), (3, 0, 0, 0, 0));
    assert_eq!(output.as_ref().unwrap().payload().0.as_ptr(), ptr);
    assert_eq!(snapshot(&root), with_candidate);
    drop(candidate);
    drop(new_source);
    assert_eq!(snapshot(&root).0, before.0);
    assert_eq!(snapshot(&root).1, with_candidate.1);
    assert_eq!(snapshot(&root).2, with_candidate.2 + 1);
    assert_eq!(snapshot(&root).3, before.3);
    let after_candidate = snapshot(&root);
    drop(slot);
    assert_eq!(snapshot(&root), after_candidate);
    drop(output);
    assert_eq!(root.budget().used_bytes(), baseline());
}

#[test]
fn normal_payload_drop_observes_full_reservation_before_guard_release() {
    struct Observe<'a> { budget: &'a Budget, seen: &'a Cell<usize> }
    impl Drop for Observe<'_> {
        fn drop(&mut self) { self.seen.set(self.budget.used_bytes()); }
    }
    let root = OwnedBudget::new(baseline() + 4096).unwrap();
    let seen = Cell::new(0);
    let mut slot = ResidentAdmission::<Observe<'_>>::empty();
    let mut source = Some(root.charge_external(0).unwrap());
    ResidentAdmission::enter_into(&mut source, 0, &mut slot).unwrap();
    slot.attempt().admit_inline().unwrap();
    let mut payload = Some(Observe { budget: root.budget(), seen: &seen });
    let mut output = None;
    slot.attempt().seal_into(&mut payload, &mut output).unwrap();
    assert_eq!(seen.get(), 0);
    let full = root.budget().used_bytes();
    let updates = root.budget().test_accounting_updates();
    drop(output);
    assert_eq!(seen.get(), full);
    assert_eq!(root.budget().used_bytes(), baseline());
    assert_eq!(root.budget().test_accounting_updates(), updates + 1);
}

#[test]
fn caller_unwind_retires_backing_before_resident_guard() {
    struct Observe<'a> {
        backing: Box<[u8]>,
        budget: &'a Budget,
        seen: &'a Cell<Option<(usize, usize)>>,
    }
    impl Drop for Observe<'_> {
        fn drop(&mut self) {
            // The Box field retires immediately after this observer returns,
            // while the actual original reservation remains outside unwind.
            self.seen.set(Some((self.budget as *const Budget as usize, self.budget.used_bytes())));
        }
    }
    let root = OwnedBudget::new(baseline() + 4096).unwrap();
    let seen = Cell::new(None);
    let mut slot = ResidentAdmission::<Observe<'_>>::empty();
    let mut source = Some(root.charge_external(3).unwrap());
    ResidentAdmission::enter_into(&mut source, 3, &mut slot).unwrap();
    slot.attempt().admit_inline().unwrap();
    let state = resident(&slot);
    let before = snapshot(&root);
    let counter = root.budget() as *const Budget as usize;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _attempt = slot.attempt();
        // Caller explicitly scopes backing inside unwind, guard outside it.
        // This arrangement is an oracle, not a type-enforced API guarantee.
        let payload = Observe {
            backing: vec![3, 5, 8].into_boxed_slice(),
            budget: root.budget(),
            seen: &seen,
        };
        assert_eq!(payload.backing.len(), 3);
        panic!("bounded resident admission caller unwind");
    }));
    assert!(result.is_err());
    assert_eq!(seen.get(), Some((counter, before.0)));
    assert_eq!(resident(&slot), state);
    assert_eq!(snapshot(&root), before);
    // Backing has retired; only now may the resident guard release its charge.
    drop(slot);
    assert_eq!(root.budget().used_bytes(), baseline());
    assert_eq!(root.budget().admitted_peak_bytes(), before.1);
    assert_eq!(root.budget().test_accounting_updates(), before.2 + 1);
    assert_eq!(Arc::strong_count(&root.budget), 1);
}

#[test]
fn plain_and_aligned_zsts_use_actual_wrapper_layout() {
    #[repr(align(64))]
    struct Aligned;
    let metadata = size_of::<OwnedExternalCharge>();
    assert_eq!(size_of::<Aligned>(), 0);
    for (s, q, inline) in [
        { let (s, q) = ResidentAdmission::<()>::quote(0).unwrap();
          (s, q, size_of::<OwnedCharged<()>>() - metadata) },
        { let (s, q) = ResidentAdmission::<Aligned>::quote(0).unwrap();
          (s, q, size_of::<OwnedCharged<Aligned>>() - metadata) },
    ] {
        assert_eq!(s, metadata);
        assert_eq!(q - s, inline);
    }
    let root = OwnedBudget::new(baseline() + 4096).unwrap();
    let mut plain = ResidentAdmission::<()>::empty();
    let mut source = Some(root.charge_external(0).unwrap());
    ResidentAdmission::enter_into(&mut source, 0, &mut plain).unwrap();
    let before = snapshot(&root);
    plain.attempt().admit_inline().unwrap();
    let delta = size_of::<OwnedCharged<()>>() - metadata;
    assert_eq!(snapshot(&root), (before.0 + delta, before.0 + delta,
        before.2 + usize::from(delta != 0), before.3));
    let mut aligned = ResidentAdmission::<Aligned>::empty();
    let mut source = Some(root.charge_external(0).unwrap());
    ResidentAdmission::enter_into(&mut source, 0, &mut aligned).unwrap();
    let before = snapshot(&root);
    aligned.attempt().admit_inline().unwrap();
    let delta = size_of::<OwnedCharged<Aligned>>() - metadata;
    // On platforms where metadata already meets this layout, no padding is due.
    if metadata < std::mem::align_of::<Aligned>() { assert!(delta > 0); }
    assert_eq!(snapshot(&root), (before.0 + delta, before.0 + delta,
        before.2 + usize::from(delta != 0), before.3));
}

#[test]
fn private_metadata_validation_rejects_without_moving_or_accounting() {
    let root = OwnedBudget::new(baseline() + 4096).unwrap();
    let mut source = Some(root.charge_external(7).unwrap());
    let mut slot = ResidentAdmission::<u64>::empty();
    ResidentAdmission::enter_into(&mut source, 7, &mut slot).unwrap();
    // Fault injection changes only scalar metadata, never the real guard debit.
    for mode in 0..3 {
        if let ResidentAdmissionState::Pending { backing_bytes, quoted_bytes, .. } = &mut slot.state {
            match mode {
                0 => *backing_bytes = 8,
                1 => *quoted_bytes += 1,
                _ => *backing_bytes = usize::MAX,
            }
        }
        let state = resident(&slot);
        let before = snapshot(&root);
        assert_eq!(slot.attempt().admit_inline(), Err(Error::Resource));
        assert_eq!(resident(&slot), state);
        assert_eq!(snapshot(&root), before);
        if let ResidentAdmissionState::Pending { backing_bytes, quoted_bytes, .. } = &mut slot.state {
            *backing_bytes = 7;
            *quoted_bytes = ResidentAdmission::<u64>::quote(7).unwrap().1;
        }
    }
    slot.attempt().admit_inline().unwrap();
    if let ResidentAdmissionState::Admitted { quoted_bytes, .. } = &mut slot.state {
        *quoted_bytes += 1;
    }
    let state = resident(&slot);
    let before = snapshot(&root);
    let mut payload = Some(41);
    let mut output = None;
    assert_eq!(slot.attempt().admit_inline(), Err(Error::Resource));
    assert_eq!(slot.attempt().seal_into(&mut payload, &mut output), Err(Error::Resource));
    assert_eq!(payload, Some(41));
    assert!(output.is_none());
    assert_eq!(resident(&slot), state);
    assert_eq!(snapshot(&root), before);
}

#[test]
fn checked_counter_overflow_rejects_without_disturbing_pending_guard() {
    let root = OwnedBudget::new(baseline() + 4096).unwrap();
    let mut source = Some(root.charge_external(1).unwrap());
    let mut slot = ResidentAdmission::<u64>::empty();
    ResidentAdmission::enter_into(&mut source, 1, &mut slot).unwrap();
    let state = resident(&slot);
    let before = snapshot(&root);
    assert_eq!(root.budget().reserve_bytes(usize::MAX), Err(Error::Resource));
    assert_eq!(resident(&slot), state);
    assert_eq!(snapshot(&root), before);
}
