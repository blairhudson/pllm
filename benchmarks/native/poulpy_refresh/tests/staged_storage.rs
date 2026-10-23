//! Ordinary Cargo integration tests. Parent compiles/runs serially; no HE keys,
//! ciphertexts, context solver, or altered cryptographic profile are instantiated.
#[path = "../src/bounded_allocator.rs"]
mod bounded_allocator;
#[path = "../src/staged_preparation.rs"]
mod staged_preparation;
#[path = "../src/staged_storage.rs"]
mod staged_storage;

use bounded_allocator::{BOOKKEEPING_TICKET, BoundedAllocator, Mapper, PAGE_BYTES, SystemMapper};
use poulpy_core::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWELayout, GetAutomorphismKey, Rank, TorusPrecision,
};
use poulpy_hal::AlignedBuf;
use staged_storage::{CheckedOwner, PhaseOwnerQuote, RotationPhase, StorageError, StorageQuote};
use std::alloc::{GlobalAlloc, Layout};

#[global_allocator]
static ALLOCATOR: BoundedAllocator = BoundedAllocator::qualification();

#[test]
fn real_upstream_aligned_owner_releases_original_capacity() {
    let mut buffer = AlignedBuf::zeroed(PAGE_BYTES, 64);
    buffer.truncate(64);
    let owner = CheckedOwner::aligned(buffer).unwrap();
    assert_eq!(owner.borrow().len(), 64);
    assert_eq!(owner.borrow().capacity(), PAGE_BYTES);
    let quote = owner.known_storage().unwrap();
    assert_eq!(quote.payload(), PAGE_BYTES);
    assert_eq!(quote.mapped(), 2 * PAGE_BYTES);
    assert_eq!(quote.charged(), 2 * PAGE_BYTES + BOOKKEEPING_TICKET);
    assert_eq!(quote.mappings(), 1);
    owner.release().unwrap();
}

#[test]
fn actual_phase_provider_rejects_missing_key_and_never_admits_execution() {
    let mut phase = RotationPhase::empty();
    {
        let view = phase.views();
        assert!(view.get_automorphism_key(-1, TorusPrecision(1427)).is_err());
    }
    assert_eq!(phase.clear(), Err(StorageError::AllocatorRejected));
    assert!(matches!(
        phase.storage_quote(),
        Err(StorageError::AllocatorRejected)
    ));
    assert_eq!(
        staged_storage::admit_encrypted_execution(),
        Err(StorageError::IncompleteQualificationBounds)
    );
}

#[test]
fn full_seven_row_shape_is_required_without_allocating_key_storage() {
    let mut shape = GGLWELayout {
        n: Degree(65536),
        base2k: Base2K(52),
        dnum: Dnum(7),
        dsize: Dsize(4),
        k_aux: TorusPrecision(224),
        rank_in: Rank(1),
        rank_out: Rank(1),
        stride: 1,
    };
    let layout = staged_storage::high_key_layout(&shape).unwrap();
    assert_eq!((layout.size(), layout.align()), (968_884_224, 64));
    // The generic raw-scratch shortcut's two rows are not this preset's key.
    shape.dnum = Dnum(2);
    assert_eq!(
        staged_storage::high_key_layout(&shape),
        Err(StorageError::Shape)
    );
}

// This body type-checks actual specialized upstream owners/views without ever
// constructing a 968-MiB key. Removing `drop(view)` then using view after clear
// would be rejected by Rust's ordinary borrow checker, not a Python model.
#[allow(dead_code)]
fn upstream_typed_views(phase: &mut RotationPhase) {
    let view = phase.views();
    let _ = view.lookup_automorphism_key(-1, TorusPrecision(1427));
    drop(view);
    let _ = phase.clear();
}

struct FailingUnmap;
unsafe impl Mapper for FailingUnmap {
    fn page_size(&self) -> usize {
        PAGE_BYTES
    }
    unsafe fn map(&self, length: usize) -> *mut u8 {
        unsafe { SystemMapper.map(length) }
    }
    unsafe fn unmap(&self, _address: *mut u8, _length: usize) -> i32 {
        -1
    }
}

struct DirectOwner<'a, M: Mapper> {
    allocator: &'a BoundedAllocator<M>,
    pointer: *mut u8,
    layout: Layout,
}

impl<M: Mapper> Drop for DirectOwner<'_, M> {
    fn drop(&mut self) {
        unsafe {
            self.allocator.dealloc(self.pointer, self.layout);
        }
    }
}

#[test]
fn actual_failed_unmap_cannot_pass_checked_drop() {
    let allocator = BoundedAllocator::new(FailingUnmap, 1 << 20, 4, BOOKKEEPING_TICKET);
    // PAGE alignment fixes user offset to PAGE, allowing test-only OS cleanup
    // without exposing private allocator headers or adding a release bypass.
    let layout = Layout::from_size_align(64, PAGE_BYTES).unwrap();
    let pointer = unsafe { allocator.alloc(layout) };
    assert!(!pointer.is_null());
    let before = allocator.snapshot();
    let owner = DirectOwner {
        allocator: &allocator,
        pointer,
        layout,
    };
    assert_eq!(
        staged_storage::drop_checked(&allocator, owner, layout),
        Err(StorageError::AllocatorRejected)
    );
    let after = allocator.snapshot();
    assert!(after.poisoned);
    assert_eq!(
        StorageQuote::default().check_additional_capacity(after, usize::MAX, usize::MAX),
        Err(StorageError::AllocatorRejected)
    );
    assert_eq!(
        (after.charged, after.mappings),
        (before.charged, before.mappings)
    );
    // Test-only cleanup. Poison/charges remain, with no production retry API.
    unsafe {
        assert_eq!(
            SystemMapper.unmap(pointer.sub(PAGE_BYTES), before.mapped),
            0
        );
    }
    assert_eq!(allocator.snapshot().charged, before.charged);
}

#[test]
fn absent_deallocation_is_not_observed_release() {
    let allocator = BoundedAllocator::new(SystemMapper, 1 << 20, 4, BOOKKEEPING_TICKET);
    let layout = Layout::from_size_align(64, 64).unwrap();
    let pointer = unsafe { allocator.alloc(layout) };
    assert!(!pointer.is_null());
    assert_eq!(
        staged_storage::drop_checked(&allocator, (), layout),
        Err(StorageError::ReleaseNotObserved)
    );
    unsafe {
        allocator.dealloc(pointer, layout);
    }
}

#[test]
fn bounded_native_record_reader_rejects_short_length_and_trailing_bytes() {
    use std::io::{Cursor, Read};
    for bytes in [b"abcd".as_slice(), b"abc".as_slice(), b"abcde".as_slice()] {
        let mut reader = Cursor::new(bytes);
        let result = staged_preparation::read_record(&mut reader, 4, |record| {
            let mut buffer = [0u8; 4];
            record.read_exact(&mut buffer)
        });
        assert_eq!(result.is_ok(), bytes.len() == 4);
    }
    let mut reader = Cursor::new(b"abcd".as_slice());
    assert!(
        staged_preparation::read_record(&mut reader, 4, |record| {
            let mut short = [0u8; 3];
            record.read_exact(&mut short)
        })
        .is_err()
    );
}

#[test]
fn failed_phase_cannot_be_reused_after_cleanup() {
    let mut phase = RotationPhase::empty();
    assert!(phase.take_for_refill(0).is_err());
    assert_eq!(phase.clear(), Err(StorageError::AllocatorRejected));
    assert!(
        phase
            .views()
            .lookup_automorphism_key(-1, TorusPrecision(1427))
            .is_err()
    );
}

#[test]
fn actual_preparation_scratch_preserves_original_capacity_after_truncation() {
    let mut scratch = staged_preparation::PreparationScratch {
        data: AlignedBuf::zeroed(4 << 20, 64),
        _phantom: std::marker::PhantomData,
    };
    scratch.data.truncate(2 << 20);
    let owner = CheckedOwner::preparation_scratch(scratch).unwrap();
    assert_eq!(owner.borrow().data.len(), 2 << 20);
    assert_eq!(owner.borrow().data.capacity(), 4 << 20);
    assert_eq!(owner.known_storage().unwrap().payload(), 4 << 20);
    owner.release().unwrap();
}

#[test]
fn preparation_scratch_exact_minimum_and_one_short_use_visible_length_not_capacity() {
    let make = |length| staged_preparation::PreparationScratch {
        data: AlignedBuf::zeroed(length, 64),
        _phantom: std::marker::PhantomData,
    };
    let exact = CheckedOwner::preparation_scratch(make(2 << 20)).unwrap();
    assert_eq!(exact.known_storage().unwrap().payload(), 2 << 20);
    exact.release().unwrap();
    assert!(matches!(
        CheckedOwner::preparation_scratch(make((2 << 20) - 64)),
        Err(StorageError::Capacity)
    ));
    let mut truncated = make(2 << 20);
    truncated.data.truncate((2 << 20) - 1);
    assert!(matches!(
        CheckedOwner::preparation_scratch(truncated),
        Err(StorageError::Capacity)
    ));
    assert!(matches!(
        CheckedOwner::preparation_scratch(staged_preparation::PreparationScratch {
            data: AlignedBuf::zeroed(2 << 20, 32),
            _phantom: std::marker::PhantomData,
        }),
        Err(StorageError::Shape)
    ));
}

#[test]
fn allocation_geometry_quotes_exact_fit_one_byte_short_and_mapping_capacity() {
    // Alignment larger than a page must retain maximum-offset overhead too.
    let layout = Layout::from_size_align(PAGE_BYTES + 1, 2 * PAGE_BYTES).unwrap();
    let quote = StorageQuote::allocation(layout).unwrap();
    assert_eq!(quote.payload(), PAGE_BYTES + 1);
    assert_eq!(quote.mapped(), 4 * PAGE_BYTES);
    assert_eq!(quote.charged(), 4 * PAGE_BYTES + BOOKKEEPING_TICKET);
    assert_eq!(quote.check_capacity(quote.charged(), 1), Ok(()));
    assert_eq!(
        quote.check_capacity(quote.charged() - 1, 1),
        Err(StorageError::Capacity)
    );
    assert_eq!(
        quote.check_capacity(quote.charged(), 0),
        Err(StorageError::Capacity)
    );
    assert_eq!(
        StorageQuote::allocation(Layout::from_size_align(0, 64).unwrap()),
        Err(StorageError::Shape)
    );
    assert_eq!(
        StorageQuote::allocation(Layout::from_size_align(isize::MAX as usize, 1).unwrap()),
        Err(StorageError::ArithmeticOverflow)
    );
    assert_eq!(
        quote.checked_repeat(usize::MAX),
        Err(StorageError::ArithmeticOverflow)
    );
    assert_eq!(quote.checked_repeat(0).unwrap(), StorageQuote::default());
    let huge = quote.checked_repeat(usize::MAX / quote.charged()).unwrap();
    assert_eq!(
        huge.checked_add(huge),
        Err(StorageError::ArithmeticOverflow)
    );
}

#[test]
fn borrowed_output_scratch_recovery_inventory_counts_original_capacities_together() {
    let output = CheckedOwner::aligned(AlignedBuf::zeroed(PAGE_BYTES, 64)).unwrap();
    let mut scratch_buffer = AlignedBuf::zeroed(2 * PAGE_BYTES, 64);
    scratch_buffer.truncate(64);
    let scratch = CheckedOwner::aligned(scratch_buffer).unwrap();
    let recovery = CheckedOwner::aligned(AlignedBuf::zeroed(3 * PAGE_BYTES, 64)).unwrap();
    let output_and_scratch = output
        .known_storage()
        .unwrap()
        .checked_add(scratch.known_storage().unwrap())
        .unwrap();
    let expected = output_and_scratch
        .checked_add(recovery.known_storage().unwrap())
        .unwrap();
    let mut phase = PhaseOwnerQuote::<3>::empty();
    phase.retain(&output).unwrap();
    phase.retain(&scratch).unwrap();
    phase.retain(&recovery).unwrap();
    assert_eq!(phase.known_storage().unwrap(), expected);
    assert_eq!(expected.payload(), 6 * PAGE_BYTES);
    assert_eq!(expected.mappings(), 3);
    assert_eq!(phase.check_capacity(expected.charged(), 3), Ok(()));
    // Recovery is not free merely because it is not the current output.
    assert_eq!(
        phase.check_capacity(output_and_scratch.charged(), 3),
        Err(StorageError::Capacity)
    );
    assert_eq!(
        phase.check_capacity(usize::MAX, usize::MAX),
        Err(StorageError::AllocatorRejected)
    );
    drop(phase);
    output.release().unwrap();
    scratch.release().unwrap();
    recovery.release().unwrap();
}

#[test]
fn phase_owner_quote_rejects_duplicate_and_full_inventory_stickily() {
    let first = CheckedOwner::aligned(AlignedBuf::zeroed(64, 64)).unwrap();
    let second = CheckedOwner::aligned(AlignedBuf::zeroed(64, 64)).unwrap();
    let mut duplicate = PhaseOwnerQuote::<2>::empty();
    duplicate.retain(&first).unwrap();
    assert_eq!(duplicate.retain(&first), Err(StorageError::Duplicate));
    assert_eq!(
        duplicate.retain(&second),
        Err(StorageError::AllocatorRejected)
    );
    assert_eq!(
        duplicate.known_storage(),
        Err(StorageError::AllocatorRejected)
    );
    drop(duplicate);
    let mut full = PhaseOwnerQuote::<1>::empty();
    full.retain(&first).unwrap();
    assert_eq!(full.retain(&second), Err(StorageError::Capacity));
    assert_eq!(full.known_storage(), Err(StorageError::AllocatorRejected));
    drop(full);
    first.release().unwrap();
    second.release().unwrap();
}

#[test]
fn standard_high_key_preparation_leaf_overlap_is_not_a_complete_phase_bound() {
    let shape = GGLWELayout {
        n: Degree(65536),
        base2k: Base2K(52),
        dnum: Dnum(7),
        dsize: Dsize(4),
        k_aux: TorusPrecision(224),
        rank_in: Rank(1),
        rank_out: Rank(1),
        stride: 1,
    };
    let quote = staged_storage::rotation_prepare_leaf_quote(&shape).unwrap();
    assert_eq!(quote.payload(), 1_215_299_584);
    assert_eq!(quote.mapped(), 1_215_365_120);
    assert_eq!(quote.charged(), 1_215_496_192);
    assert_eq!(quote.mappings(), 4); // raw + output + scratch + private Vec
    assert_eq!(quote.check_capacity(quote.charged(), 4), Ok(()));
    assert_eq!(
        quote.check_capacity(quote.charged() - 1, 4),
        Err(StorageError::Capacity)
    );
    assert_eq!(
        quote.check_capacity(quote.charged(), 3),
        Err(StorageError::Capacity)
    );
    let private = staged_storage::rotation_prepare_private_workspace().unwrap();
    assert_eq!(private.payload(), 2 << 20);
    assert_eq!(private.charged(), (2 << 20) + 49_152);
    assert_eq!(private.mappings(), 1); // not 462 simultaneously live blocks
    let recovery =
        StorageQuote::allocation(Layout::from_size_align(242_221_056, 64).unwrap()).unwrap();
    let with_recovery = quote.checked_add(recovery).unwrap();
    assert_eq!(with_recovery.payload(), 1_457_520_640);
    assert_eq!(with_recovery.charged(), 1_457_766_400);
    assert_eq!(with_recovery.mappings(), 5);
    assert_eq!(
        with_recovery.check_capacity(quote.charged(), 5),
        Err(StorageError::Capacity)
    );
    assert_eq!(staged_storage::UNCLOSED_BOUNDS.len(), 6);
    assert_eq!(
        staged_storage::require_rotation_preconstruction(&shape),
        Err(StorageError::IncompleteQualificationBounds)
    );
    assert_eq!(
        staged_storage::admit_encrypted_execution(),
        Err(StorageError::IncompleteQualificationBounds)
    );
}

#[test]
fn actual_allocator_preflight_keeps_output_scratch_and_recovery_until_release() {
    let layout = Layout::from_size_align(64, 64).unwrap();
    let leaf = StorageQuote::allocation(layout).unwrap();
    let overlap = leaf.checked_repeat(4).unwrap();
    let allocator = BoundedAllocator::new(SystemMapper, overlap.charged(), 4, BOOKKEEPING_TICKET);
    let allocate = || {
        let pointer = unsafe { allocator.alloc(layout) };
        assert!(!pointer.is_null());
        DirectOwner {
            allocator: &allocator,
            pointer,
            layout,
        }
    };
    let output = allocate();
    let scratch = allocate();
    let recovery = allocate();
    let before = allocator.snapshot();
    assert_eq!(before.charged, leaf.checked_repeat(3).unwrap().charged());
    assert_eq!(
        leaf.check_additional_capacity(before, overlap.charged(), 4),
        Ok(())
    );
    assert_eq!(
        leaf.check_additional_capacity(before, overlap.charged() - 1, 4),
        Err(StorageError::Capacity)
    );
    assert_eq!(
        leaf.check_additional_capacity(before, overlap.charged(), 3),
        Err(StorageError::Capacity)
    );
    // Preflight must neither reserve twice nor credit any still-live owner.
    assert_eq!(allocator.snapshot().charged, before.charged);
    let workspace = allocate();
    assert_eq!(allocator.snapshot().peak_charged, overlap.charged());
    staged_storage::drop_checked(&allocator, workspace, layout).unwrap();
    staged_storage::drop_checked(&allocator, scratch, layout).unwrap();
    staged_storage::drop_checked(&allocator, recovery, layout).unwrap();
    assert_eq!(allocator.snapshot().charged, leaf.charged()); // output still owned
    staged_storage::drop_checked(&allocator, output, layout).unwrap();
    assert_eq!(allocator.snapshot().charged, 0);
}

#[test]
fn quote_rejects_sticky_denial_after_all_successful_releases_and_checked_overflow() {
    let layout = Layout::from_size_align(64, 64).unwrap();
    let leaf = StorageQuote::allocation(layout).unwrap();
    let allocator = BoundedAllocator::new(SystemMapper, leaf.charged(), 1, BOOKKEEPING_TICKET);
    let pointer = unsafe { allocator.alloc(layout) };
    assert!(!pointer.is_null());
    assert!(unsafe { allocator.alloc(layout) }.is_null());
    unsafe { allocator.dealloc(pointer, layout) };
    let after = allocator.snapshot();
    assert_eq!(after.charged, 0);
    assert_eq!(after.denied, 1);
    assert_eq!(
        leaf.check_additional_capacity(after, usize::MAX, usize::MAX),
        Err(StorageError::AllocatorRejected)
    );
    let healthy = BoundedAllocator::new(SystemMapper, usize::MAX, usize::MAX, BOOKKEEPING_TICKET);
    for field in 0..3 {
        let mut snapshot = healthy.snapshot();
        match field {
            0 => snapshot.charged = usize::MAX,
            1 => snapshot.mappings = usize::MAX,
            _ => snapshot.mapped = usize::MAX,
        }
        assert_eq!(
            leaf.check_additional_capacity(snapshot, usize::MAX, usize::MAX),
            Err(StorageError::ArithmeticOverflow)
        );
    }
}

#[test]
fn canonical_deserialization_does_not_shrink_original_upstream_backing() {
    use poulpy_hal::layouts::{
        Backend, DataView, MatZnx, MatZnxToBackendRef, ReaderFrom, WriterTo,
    };
    use std::io::Cursor;

    // Tiny real upstream matrix serialization, not an HE key/profile. Two-row
    // backing accepts a one-row canonical record without releasing its tail.
    let canonical =
        MatZnx::<AlignedBuf, i64>::from_data(AlignedBuf::zeroed(128, 64), 8, 1, 1, 2, 1);
    let mut oversized =
        MatZnx::<AlignedBuf, i64>::from_data(AlignedBuf::zeroed(256, 64), 8, 2, 1, 2, 1);
    let mut wire = Vec::new();
    canonical.write_to(&mut wire).unwrap();
    oversized.read_from(&mut Cursor::new(wire)).unwrap();
    assert_eq!(oversized.shape(), canonical.shape());
    assert_eq!(DataView::data(&oversized).capacity(), 256);
    let backend =
        <MatZnx<AlignedBuf, i64> as MatZnxToBackendRef<staged_storage::Cpu>>::to_backend_ref(
            &oversized,
        );
    let span = staged_storage::Cpu::len_bytes_ref(DataView::data(&backend));
    assert_eq!(span, 256);
    assert_eq!(
        staged_preparation::raw_backing_layout(span, Layout::from_size_align(128, 64).unwrap()),
        Err(StorageError::Capacity)
    );
}

#[test]
fn matching_raw_span_cannot_certify_hidden_original_capacity() {
    use poulpy_hal::layouts::Backend;
    let expected = Layout::from_size_align(128, 64).unwrap();
    let mut oversized = AlignedBuf::zeroed(256, 64);
    oversized.truncate(128);
    let view = staged_storage::Cpu::view(&oversized);
    let span = staged_storage::Cpu::len_bytes_ref(&view);
    assert_eq!(span, expected.size());
    assert_eq!(oversized.capacity(), 256);
    assert_eq!(
        staged_preparation::raw_backing_layout(span, expected),
        Err(StorageError::UnprovenOwnerCapacity)
    );
    // Even an actually exact allocation cannot be certified from its span alone.
    let exact = AlignedBuf::zeroed(128, 64);
    assert_eq!(
        staged_preparation::raw_backing_layout(exact.len(), expected),
        Err(StorageError::UnprovenOwnerCapacity)
    );
}

#[test]
fn refill_drop_and_explicit_abort_burn_origin_after_checked_cleanup() {
    use std::cell::Cell;
    for explicit in [false, true] {
        let failed = Cell::new(false);
        let pending = Cell::new(false);
        let mut slots = [Some(
            CheckedOwner::aligned(AlignedBuf::zeroed(64, 64)).unwrap(),
        )];
        let guard = staged_storage::RefillGuard::begin(&mut slots, &failed, &pending, 0).unwrap();
        assert!(pending.get());
        if explicit {
            guard.abort().unwrap();
        } else {
            drop(guard);
        }
        assert!(failed.get());
        assert!(pending.get());
        assert!(slots[0].is_none());
        assert!(matches!(
            staged_storage::RefillGuard::begin(&mut slots, &failed, &pending, 0),
            Err(StorageError::AllocatorRejected)
        ));
    }
}

#[test]
fn detached_adoption_and_closed_preparation_errors_burn_origin() {
    use std::cell::Cell;
    fn reject_adoption(
        mut guard: staged_storage::RefillGuard<'_, AlignedBuf>,
    ) -> Result<(), StorageError> {
        let _original = guard.take_destination()?;
        Err(StorageError::UnprovenOwnerCapacity)
    }
    fn reject_preparation(
        mut guard: staged_storage::RefillGuard<'_, AlignedBuf>,
    ) -> Result<(), StorageError> {
        let original = guard.take_destination()?;
        staged_storage::admit_encrypted_execution()?;
        guard.commit_checked(original, |_| Ok(()), |a, b| a.len() == b.len())
    }
    for preparation in [false, true] {
        let failed = Cell::new(false);
        let pending = Cell::new(false);
        let mut slots = [Some(
            CheckedOwner::aligned(AlignedBuf::zeroed(64, 64)).unwrap(),
        )];
        let guard = staged_storage::RefillGuard::begin(&mut slots, &failed, &pending, 0).unwrap();
        let error = if preparation {
            reject_preparation(guard)
        } else {
            reject_adoption(guard)
        };
        assert_eq!(
            error,
            Err(if preparation {
                StorageError::IncompleteQualificationBounds
            } else {
                StorageError::UnprovenOwnerCapacity
            })
        );
        assert!(failed.get());
        assert!(pending.get());
        assert!(slots[0].is_none());
        assert!(matches!(
            staged_storage::RefillGuard::begin(&mut slots, &failed, &pending, 0),
            Err(StorageError::AllocatorRejected)
        ));
    }
}

#[test]
fn refill_checked_commit_restores_only_valid_unique_ready_owner() {
    use std::cell::Cell;
    for reject in [false, true] {
        let failed = Cell::new(false);
        let pending = Cell::new(false);
        let mut slots = [Some(
            CheckedOwner::aligned(AlignedBuf::zeroed(64, 64)).unwrap(),
        )];
        let mut guard =
            staged_storage::RefillGuard::begin(&mut slots, &failed, &pending, 0).unwrap();
        let original = guard.take_destination().unwrap();
        let result = guard.commit_checked(
            original,
            |_| {
                if reject {
                    Err(StorageError::Shape)
                } else {
                    Ok(())
                }
            },
            |a, b| a.len() == b.len(),
        );
        if reject {
            assert_eq!(result, Err(StorageError::Shape));
            assert!(failed.get());
            assert!(pending.get());
            assert!(slots[0].is_none());
        } else {
            result.unwrap();
            assert!(!failed.get());
            assert!(!pending.get());
            slots[0].take().unwrap().release().unwrap();
        }
    }
}

#[test]
fn refill_duplicate_commit_and_repeated_extraction_burn_origin() {
    use std::cell::Cell;
    let make = |identity| {
        let buffer = AlignedBuf::zeroed(64, 64);
        CheckedOwner::new((identity, buffer), Layout::from_size_align(64, 64).unwrap()).unwrap()
    };
    for repeated_take in [false, true] {
        let failed = Cell::new(false);
        let pending = Cell::new(false);
        let mut slots = [Some(make(1u8)), Some(make(2u8))];
        let mut guard =
            staged_storage::RefillGuard::begin(&mut slots, &failed, &pending, 0).unwrap();
        let mut original = guard.take_destination().unwrap();
        if repeated_take {
            assert!(matches!(
                guard.take_destination(),
                Err(StorageError::Capacity)
            ));
        } else {
            // A refill's new role identity conflicts with another live owner.
            original.borrow_mut().0 = 2;
        }
        let result = guard.commit_checked(original, |_| Ok(()), |a, b| a.0 == b.0);
        assert_eq!(
            result,
            Err(if repeated_take {
                StorageError::AllocatorRejected
            } else {
                StorageError::Duplicate
            })
        );
        assert!(failed.get());
        assert!(pending.get());
        assert!(slots[0].is_none());
        slots[1].take().unwrap().release().unwrap();
        assert!(matches!(
            staged_storage::RefillGuard::begin(&mut slots, &failed, &pending, 0),
            Err(StorageError::AllocatorRejected)
        ));
    }
}

// Parent compilation checks this same standard typed preflight without creating
// a Module, keys or scratch. The quote is not an admission/preparation token.
#[allow(dead_code)]
fn upstream_typed_preparation_quote<'a>(
    module: &poulpy_hal::layouts::Module<staged_storage::Cpu>,
    raw: &'a CheckedOwner<staged_preparation::RawRotation>,
    output: &'a CheckedOwner<staged_storage::Rotation>,
    scratch: &'a CheckedOwner<staged_preparation::PreparationScratch>,
    recovery: &'a CheckedOwner<staged_preparation::RawRotation>,
) -> Result<staged_storage::RotationPreparationQuote<'a>, StorageError> {
    let quote =
        staged_storage::quote_rotation_preparation(module, raw, output, scratch, Some(recovery))?;
    let _ = quote.known_storage()?;
    quote.check_workspace_capacity(
        bounded_allocator::CHARGE_CEILING,
        bounded_allocator::MAX_MAPPINGS,
    )?;
    Ok(quote)
}

// Parent's ordinary Cargo compilation type-checks the full actual upstream
// raw -> factory prepare -> checked loading release -> typed provider path.
// It is never invoked here: the full qualification-bound gate still rejects.
#[allow(dead_code)]
fn upstream_typed_preparation(
    module: &poulpy_hal::layouts::Module<staged_storage::Cpu>,
    phase: &mut RotationPhase,
    raw: staged_preparation::RawRotation,
    scratch: staged_preparation::PreparationScratch,
    expected_p: i64,
) -> Result<(), StorageError> {
    let refill = phase.take_for_refill(0)?;
    let loading =
        staged_preparation::RotationPreparation::adopt_local_raw(raw, refill, scratch, expected_p)?;
    loading.prepare_and_commit(module)
}
