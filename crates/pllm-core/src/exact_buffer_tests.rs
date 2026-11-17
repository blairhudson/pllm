use super::*;
use std::cell::Cell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;

thread_local! {
    static ALLOCATIONS: Cell<(usize, bool)> = const { Cell::new((0, false)) };
    static ZST_DROPS: Cell<usize> = const { Cell::new(0) };
}

pub(super) fn refuse_allocation() -> bool {
    ALLOCATIONS.with(|state| {
        let (entries, refuse) = state.get();
        state.set((entries + 1, refuse));
        refuse
    })
}

pub(crate) struct AllocationScope;
impl AllocationScope {
    pub(crate) fn new(refuse: bool) -> Self {
        ALLOCATIONS.with(|state| state.set((0, refuse)));
        Self
    }

    pub(crate) fn entries(&self) -> usize {
        ALLOCATIONS.with(|state| state.get().0)
    }
}
impl Drop for AllocationScope {
    fn drop(&mut self) {
        ALLOCATIONS.with(|state| state.set((0, false)));
    }
}

#[repr(align(256))]
#[derive(Debug)]
struct Aligned(u8);

#[repr(align(256))]
#[derive(Debug)]
struct DropZst;
impl Drop for DropZst {
    fn drop(&mut self) {
        ZST_DROPS.with(|drops| drops.set(drops.get() + 1));
    }
}

#[test]
fn zero_and_zst_skip_allocator_and_keep_aligned_exact_metadata() {
    let scope = AllocationScope::new(true);
    let empty = ExactBuffer::<Aligned>::new(0).unwrap();
    assert_eq!(empty.slots.len(), 0);
    assert_eq!(empty.slots.as_ptr().addr() % 256, 0);
    assert!(empty.finish().unwrap().is_empty());

    // Even a maximal logical ZST length has zero byte extent. Drop sees no live
    // values and does not iterate over uninitialized logical slots.
    let huge = ExactBuffer::<()>::new(usize::MAX).unwrap();
    assert_eq!(huge.slots.len(), usize::MAX);
    drop(huge);
    ZST_DROPS.with(|drops| drops.set(0));
    let mut partial = ExactBuffer::new(4).unwrap();
    partial.push(DropZst).unwrap();
    partial.push(DropZst).unwrap();
    assert_eq!(partial.slots.len(), 4);
    assert_eq!(partial.slots.as_ptr().addr() % 256, 0);
    drop(partial);
    ZST_DROPS.with(|drops| assert_eq!(drops.get(), 2));

    let mut complete = ExactBuffer::new(3).unwrap();
    for _ in 0..3 {
        complete.push(DropZst).unwrap();
    }
    let returned = complete.push(DropZst).unwrap_err();
    ZST_DROPS.with(|drops| assert_eq!(drops.get(), 2));
    drop(returned);
    let complete = complete.finish().unwrap();
    assert_eq!(complete.len(), 3);
    ZST_DROPS.with(|drops| assert_eq!(drops.get(), 3));
    drop(complete);
    ZST_DROPS.with(|drops| assert_eq!(drops.get(), 6));
    assert_eq!(scope.entries(), 0);
}

#[test]
fn layout_overflow_precedes_allocator_entry() {
    let scope = AllocationScope::new(true);
    for count in [isize::MAX as usize + 1, usize::MAX] {
        assert!(matches!(
            ExactBuffer::<u8>::new(count),
            Err(Error::Overflow)
        ));
    }
    assert!(matches!(
        ExactBuffer::<u64>::new(usize::MAX),
        Err(Error::Overflow)
    ));
    assert_eq!(layout_bytes::<u64>(usize::MAX), Err(Error::Overflow));
    assert_eq!(scope.entries(), 0);
}

#[test]
fn positive_allocation_refusal_then_guard_release_allows_same_thread_success() {
    let scope = AllocationScope::new(true);
    assert!(matches!(
        ExactBuffer::<u64>::new(3),
        Err(Error::Allocation)
    ));
    assert_eq!(scope.entries(), 1);
    assert!(matches!(
        ExactBuffer::<Aligned>::new(1),
        Err(Error::Allocation)
    ));
    assert_eq!(scope.entries(), 2);
    // Only the private seam returns null; no global allocator is replaced and
    // no forged pointer can become Box backing.
    drop(scope);

    // Construct a cleanup-only guard without calling new or resetting state:
    // success below must depend on the refusing guard's Drop, on this thread.
    let cleanup = AllocationScope;
    ALLOCATIONS.with(|state| assert_eq!(state.get(), (0, false)));
    let mut buffer = ExactBuffer::<u64>::new(2).unwrap();
    assert_eq!(cleanup.entries(), 1);
    buffer.push(17).unwrap();
    buffer.push(29).unwrap();
    let complete = buffer.finish().unwrap();
    assert_eq!(complete.as_ref(), &[17, 29]);
    assert_eq!(cleanup.entries(), 1);
    drop(complete);
    drop(cleanup);
    ALLOCATIONS.with(|state| assert_eq!(state.get(), (0, false)));
}

#[test]
fn positive_overaligned_backing_transfers_same_pointer_and_exact_length() {
    let scope = AllocationScope::new(false);
    let mut buffer = ExactBuffer::new(3).unwrap();
    let address = buffer.slots.as_ptr().cast::<Aligned>();
    assert_eq!(address.addr() % 256, 0);
    assert_eq!(buffer.slots.len(), 3);
    for value in [4, 7, 9] {
        buffer.push(Aligned(value)).unwrap();
    }
    let complete = buffer.finish().unwrap();
    assert_eq!(complete.as_ptr(), address);
    assert_eq!(complete.len(), 3);
    assert_eq!(
        complete.iter().map(|value| value.0).collect::<Vec<_>>(),
        [4, 7, 9]
    );
    assert_eq!(scope.entries(), 1);
    drop(complete);
}

#[derive(Debug)]
struct Probe {
    ordinal: usize,
    drops: Rc<[Cell<usize>; 5]>,
}
impl Drop for Probe {
    fn drop(&mut self) {
        self.drops[self.ordinal].set(self.drops[self.ordinal].get() + 1);
    }
}

fn probe(ordinal: usize, drops: &Rc<[Cell<usize>; 5]>) -> Probe {
    Probe {
        ordinal,
        drops: drops.clone(),
    }
}

fn counts(drops: &Rc<[Cell<usize>; 5]>) -> [usize; 5] {
    std::array::from_fn(|index| drops[index].get())
}

#[test]
fn capacity_failure_returns_owned_value_and_keeps_prefix() {
    let drops = Rc::new(std::array::from_fn(|_| Cell::new(0)));
    let mut buffer = ExactBuffer::new(1).unwrap();
    buffer.push(probe(0, &drops)).unwrap();
    let address = buffer.slots.as_ptr().cast::<Probe>();
    let incoming = probe(1, &drops);
    let incoming_owner = Rc::as_ptr(&incoming.drops);
    let returned = buffer.push(incoming).unwrap_err();
    assert_eq!(returned.ordinal, 1);
    assert_eq!(Rc::as_ptr(&returned.drops), incoming_owner);
    assert_eq!(counts(&drops), [0; 5]);
    assert_eq!(buffer.initialized, 1);
    drop(returned);
    assert_eq!(counts(&drops), [0, 1, 0, 0, 0]);
    let complete = buffer.finish().unwrap();
    assert_eq!(complete.as_ptr(), address);
    assert_eq!(counts(&drops), [0, 1, 0, 0, 0]);
    drop(complete);
    assert_eq!(counts(&drops), [1, 1, 0, 0, 0]);

    let mut empty = ExactBuffer::new(0).unwrap();
    let returned = empty.push(probe(2, &drops)).unwrap_err();
    assert_eq!(counts(&drops), [1, 1, 0, 0, 0]);
    drop(returned);
    assert!(empty.finish().unwrap().is_empty());
    assert_eq!(counts(&drops), [1, 1, 1, 0, 0]);
}

#[test]
fn incomplete_finish_consumes_and_drops_only_initialized_prefix() {
    let drops = Rc::new(std::array::from_fn(|_| Cell::new(0)));
    let mut buffer = ExactBuffer::new(5).unwrap();
    buffer.push(probe(0, &drops)).unwrap();
    buffer.push(probe(1, &drops)).unwrap();
    assert!(matches!(buffer.finish(), Err(Error::Incomplete)));
    assert_eq!(counts(&drops), [1, 1, 0, 0, 0]);
}

#[test]
fn initializer_panic_drops_exactly_the_live_prefix() {
    let drops = Rc::new(std::array::from_fn(|_| Cell::new(0)));
    let mut visits = Vec::new();
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut buffer = ExactBuffer::new(5).unwrap();
        for ordinal in 0..5 {
            let value = {
                visits.push(ordinal);
                assert_ne!(ordinal, 3, "initializer failure");
                probe(ordinal, &drops)
            };
            buffer.push(value).unwrap();
        }
    }));
    assert!(result.is_err());
    assert_eq!(visits, [0, 1, 2, 3]);
    assert_eq!(counts(&drops), [1, 1, 1, 0, 0]);
}

#[test]
fn destructor_unwind_drops_remaining_prefix_not_uninitialized_tail() {
    #[derive(Debug)]
    struct PanicOnce {
        probe: Probe,
        fired: Rc<Cell<bool>>,
    }
    impl Drop for PanicOnce {
        fn drop(&mut self) {
            if self.probe.ordinal == 1 && !self.fired.replace(true) {
                panic!("single destructor failure");
            }
        }
    }
    let drops = Rc::new(std::array::from_fn(|_| Cell::new(0)));
    let fired = Rc::new(Cell::new(false));
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut buffer = ExactBuffer::new(5).unwrap();
        for ordinal in 0..3 {
            buffer
                .push(PanicOnce {
                    probe: probe(ordinal, &drops),
                    fired: fired.clone(),
                })
                .unwrap();
        }
        drop(buffer);
    }));
    assert!(result.is_err());
    assert!(fired.get());
    assert_eq!(counts(&drops), [1, 1, 1, 0, 0]);
}

#[test]
fn into_initialized_vec_preserves_empty_partial_full_and_overaligned_backing() {
    for (requested, initialized) in [(0, 0), (1, 0), (4, 0), (4, 1), (4, 3), (4, 4)] {
        let scope = AllocationScope::new(false);
        let mut buffer = ExactBuffer::new(requested).unwrap();
        let pointer = buffer.slots.as_ptr().cast::<Aligned>();
        let bytes = layout_bytes::<Aligned>(requested).unwrap();
        for ordinal in 0..initialized {
            buffer.push(Aligned(ordinal as u8 + 17)).unwrap();
        }
        assert_eq!(buffer.initialized, initialized);
        let entries = scope.entries();
        assert_eq!(entries, usize::from(requested != 0));
        // Reuse the private refusal seam without resetting its entry count.
        // It observes ExactBuffer allocator entries, not every global allocation.
        ALLOCATIONS.with(|state| state.set((entries, true)));
        let values = buffer.into_initialized_vec();
        assert_eq!(scope.entries(), entries);
        assert_eq!(values.as_ptr(), pointer);
        assert_eq!(values.as_ptr().addr() % 256, 0);
        assert_eq!((values.len(), values.capacity()), (initialized, requested));
        assert_eq!(layout_bytes::<Aligned>(values.capacity()).unwrap(), bytes);
        for (ordinal, value) in values.iter().enumerate() {
            assert_eq!(value.0, ordinal as u8 + 17);
        }
        drop(values);
        assert_eq!(scope.entries(), entries);
    }
}

#[test]
fn into_initialized_vec_transfers_unique_live_owners_and_never_drops_unused_tail() {
    for initialized in [0, 1, 3, 5] {
        let scope = AllocationScope::new(false);
        let drops = Rc::new(std::array::from_fn(|_| Cell::new(0)));
        let mut buffer = ExactBuffer::new(5).unwrap();
        for ordinal in 0..initialized {
            buffer.push(probe(ordinal, &drops)).unwrap();
        }
        let pointer = buffer.slots.as_ptr().cast::<Probe>();
        assert_eq!(Rc::strong_count(&drops), initialized + 1);
        ALLOCATIONS.with(|state| state.set((scope.entries(), true)));
        let values = buffer.into_initialized_vec();
        assert_eq!(scope.entries(), 1);
        assert_eq!(values.as_ptr(), pointer);
        assert_eq!((values.len(), values.capacity()), (initialized, 5));
        assert_eq!(counts(&drops), [0; 5]);
        assert_eq!(Rc::strong_count(&drops), initialized + 1);
        for (ordinal, value) in values.iter().enumerate() {
            assert_eq!(value.ordinal, ordinal);
            assert!(Rc::ptr_eq(&value.drops, &drops));
        }
        // Probe is neither Copy nor Clone. No prefix owner was duplicated or
        // destroyed at transfer; only the returned Vec owns these live values.
        drop(values);
        let expected: [usize; 5] = std::array::from_fn(|i| usize::from(i < initialized));
        assert_eq!(counts(&drops), expected);
        assert_eq!(Rc::strong_count(&drops), 1);
        assert_eq!(scope.entries(), 1);
    }
}

#[test]
fn into_initialized_vec_after_full_push_refusal_preserves_incoming_owner() {
    let scope = AllocationScope::new(false);
    let drops = Rc::new(std::array::from_fn(|_| Cell::new(0)));
    let mut buffer = ExactBuffer::new(2).unwrap();
    buffer.push(probe(0, &drops)).unwrap();
    buffer.push(probe(1, &drops)).unwrap();
    let returned = buffer.push(probe(2, &drops)).unwrap_err();
    let pointer = buffer.slots.as_ptr().cast::<Probe>();
    ALLOCATIONS.with(|state| state.set((scope.entries(), true)));
    let values = buffer.into_initialized_vec();
    assert_eq!(values.as_ptr(), pointer);
    assert_eq!((values.len(), values.capacity()), (2, 2));
    assert_eq!(counts(&drops), [0; 5]);
    assert_eq!(returned.ordinal, 2);
    assert!(Rc::ptr_eq(&returned.drops, &drops));
    drop(returned);
    assert_eq!(counts(&drops), [0, 0, 1, 0, 0]);
    drop(values);
    assert_eq!(counts(&drops), [1, 1, 1, 0, 0]);
    assert_eq!(Rc::strong_count(&drops), 1);
    assert_eq!(scope.entries(), 1);
}

#[test]
fn into_initialized_vec_spare_capacity_accepts_moved_values_without_growth() {
    let scope = AllocationScope::new(false);
    let drops = Rc::new(std::array::from_fn(|_| Cell::new(0)));
    let mut buffer = ExactBuffer::new(5).unwrap();
    buffer.push(probe(0, &drops)).unwrap();
    buffer.push(probe(1, &drops)).unwrap();
    let pointer = buffer.slots.as_ptr().cast::<Probe>();
    ALLOCATIONS.with(|state| state.set((scope.entries(), true)));
    let mut values = buffer.into_initialized_vec();
    assert_eq!((values.len(), values.capacity()), (2, 5));
    for ordinal in 2..5 {
        assert!(values.len() < values.capacity());
        values.push(probe(ordinal, &drops));
        assert_eq!(values.as_ptr(), pointer);
        assert_eq!(values.capacity(), 5);
    }
    assert_eq!(values.len(), 5);
    assert_eq!(counts(&drops), [0; 5]);
    assert_eq!(Rc::strong_count(&drops), 6);
    drop(values);
    assert_eq!(counts(&drops), [1; 5]);
    assert_eq!(Rc::strong_count(&drops), 1);
    assert_eq!(scope.entries(), 1);
}

#[test]
fn into_initialized_vec_zst_capacity_differs_from_requested_slots_but_length_is_exact() {
    for (requested, initialized) in [(0, 0), (4, 0), (4, 2), (4, 4), (usize::MAX, 2)] {
        let scope = AllocationScope::new(true);
        ZST_DROPS.with(|drops| drops.set(0));
        let mut buffer = ExactBuffer::new(requested).unwrap();
        for _ in 0..initialized {
            buffer.push(DropZst).unwrap();
        }
        assert_eq!(buffer.slots.len(), requested);
        let pointer = buffer.slots.as_ptr().cast::<DropZst>();
        let values = buffer.into_initialized_vec();
        assert_eq!(values.as_ptr(), pointer);
        assert_eq!(values.as_ptr().addr() % 256, 0);
        assert_eq!((values.len(), values.capacity()), (initialized, usize::MAX));
        ZST_DROPS.with(|drops| assert_eq!(drops.get(), 0));
        drop(values);
        ZST_DROPS.with(|drops| assert_eq!(drops.get(), initialized));
        assert_eq!(scope.entries(), 0);
    }
}

#[test]
fn into_initialized_vec_destructor_unwind_cleans_remaining_live_prefix_once() {
    #[derive(Debug)]
    struct PanicOnce {
        probe: Probe,
        fired: Rc<Cell<bool>>,
    }
    impl Drop for PanicOnce {
        fn drop(&mut self) {
            if self.probe.ordinal == 1 && !self.fired.replace(true) {
                panic!("single transferred destructor failure");
            }
        }
    }
    for initialized in [3, 5] {
        let scope = AllocationScope::new(false);
        let drops = Rc::new(std::array::from_fn(|_| Cell::new(0)));
        let fired = Rc::new(Cell::new(false));
        let mut buffer = ExactBuffer::new(5).unwrap();
        for ordinal in 0..initialized {
            buffer.push(PanicOnce { probe: probe(ordinal, &drops), fired: fired.clone() }).unwrap();
        }
        let pointer = buffer.slots.as_ptr().cast::<PanicOnce>();
        ALLOCATIONS.with(|state| state.set((scope.entries(), true)));
        let values = buffer.into_initialized_vec();
        assert_eq!(values.as_ptr(), pointer);
        assert_eq!((values.len(), values.capacity()), (initialized, 5));
        assert_eq!(counts(&drops), [0; 5]);
        assert!(!fired.get());
        let result = catch_unwind(AssertUnwindSafe(|| drop(values)));
        assert!(result.is_err());
        assert!(fired.get());
        let expected: [usize; 5] = std::array::from_fn(|i| usize::from(i < initialized));
        assert_eq!(counts(&drops), expected);
        assert_eq!(Rc::strong_count(&drops), 1);
        assert_eq!(Rc::strong_count(&fired), 1);
        assert_eq!(scope.entries(), 1);
    }
}

#[test]
fn into_initialized_vec_zst_destructor_unwind_drops_length_not_capacity() {
    #[repr(align(256))]
    #[derive(Debug)]
    struct PanicZst;
    impl Drop for PanicZst {
        fn drop(&mut self) {
            let ordinal = ZST_DROPS.with(|drops| {
                let ordinal = drops.get();
                drops.set(ordinal + 1);
                ordinal
            });
            if ordinal == 1 {
                panic!("single transferred ZST destructor failure");
            }
        }
    }
    let scope = AllocationScope::new(true);
    ZST_DROPS.with(|drops| drops.set(0));
    let mut buffer = ExactBuffer::new(5).unwrap();
    for _ in 0..3 {
        buffer.push(PanicZst).unwrap();
    }
    let pointer = buffer.slots.as_ptr().cast::<PanicZst>();
    let values = buffer.into_initialized_vec();
    assert_eq!(values.as_ptr(), pointer);
    assert_eq!((values.len(), values.capacity()), (3, usize::MAX));
    ZST_DROPS.with(|drops| assert_eq!(drops.get(), 0));
    assert!(catch_unwind(AssertUnwindSafe(|| drop(values))).is_err());
    ZST_DROPS.with(|drops| assert_eq!(drops.get(), 3));
    assert_eq!(scope.entries(), 0);
}

#[test]
fn into_initialized_vec_keeps_zero_null_and_overflow_allocation_rules() {
    let scope = AllocationScope::new(true);
    let empty = ExactBuffer::<u64>::new(0).unwrap();
    let pointer = empty.slots.as_ptr().cast::<u64>();
    let empty = empty.into_initialized_vec();
    assert_eq!(empty.as_ptr(), pointer);
    assert_eq!((empty.len(), empty.capacity()), (0, 0));
    drop(empty);
    assert_eq!(scope.entries(), 0);
    assert!(matches!(ExactBuffer::<u64>::new(usize::MAX), Err(Error::Overflow)));
    assert!(matches!(ExactBuffer::<u8>::new(isize::MAX as usize + 1), Err(Error::Overflow)));
    assert_eq!(scope.entries(), 0);
    assert!(matches!(ExactBuffer::<Aligned>::new(3), Err(Error::Allocation)));
    assert!(matches!(ExactBuffer::<u64>::new(1), Err(Error::Allocation)));
    assert_eq!(scope.entries(), 2);
    ALLOCATIONS.with(|state| state.set((scope.entries(), false)));
    let mut buffer = ExactBuffer::new(3).unwrap();
    buffer.push(42u64).unwrap();
    let pointer = buffer.slots.as_ptr().cast::<u64>();
    assert_eq!(scope.entries(), 3);
    ALLOCATIONS.with(|state| state.set((scope.entries(), true)));
    let values = buffer.into_initialized_vec();
    assert_eq!(values.as_ptr(), pointer);
    assert_eq!((values.len(), values.capacity()), (1, 3));
    assert_eq!(values.as_slice(), &[42]);
    drop(values);
    assert_eq!(scope.entries(), 3);
}
