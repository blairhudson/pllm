use super::*;
use crate::exact_buffer::AllocationScope;
use std::cell::Cell;
use std::panic::{catch_unwind, AssertUnwindSafe};

thread_local! {
    static DROPS: Cell<usize> = const { Cell::new(0) };
    static CLONE_FAILURE: Cell<Option<usize>> = const { Cell::new(None) };
}
pub(super) fn record_drop() { DROPS.with(|n| n.set(n.get() + 1)); }
pub(super) fn clone_seam() {
    CLONE_FAILURE.with(|n| match n.get() {
        Some(0) => { n.set(None); panic!("partial linked clone unwind"); },
        Some(left) => n.set(Some(left - 1)), None => {},
    });
}

#[test]
fn linked_rows_every_exact_extent_and_no_conversion_allocation() {
    let scope = AllocationScope::new(false);
    let header = std::mem::size_of::<Record<u64, u8, u64, 1>>() - 8;
    for n in 1..=384 {
        let before = scope.entries();
        let mut link = allocate::<u64, u8, u64>(n, 99, 7).unwrap();
        assert_eq!(scope.entries(), before + 1);
        assert_eq!(link.rows().len(), n);
        assert_eq!(link.copied(), 0);
        assert_eq!(link.allocation_bytes(), n * std::mem::size_of::<u64>() + header);
        assert_eq!(layouts::<u64, u8, u64>(n).unwrap().backing, link.allocation_bytes());
        assert!(layouts::<u64, u8, u64>(n).unwrap().staging >= link.allocation_bytes());
        for i in 0..n { link.push(i as u64).unwrap(); }
        assert_eq!(link.copied(), n);
        assert!(link.push(0).is_err());
        for (i, value) in link.rows().iter().enumerate() { assert_eq!(*value, i as u64); }
        assert!(link.next().first().is_none());
    }
    let before = scope.entries();
    for n in [0, 385, 1024, usize::MAX] {
        assert!(allocate::<u64, u8, u64>(n, 0, 0).is_err());
        assert!(layouts::<u64, u8, u64>(n).is_err());
    }
    assert_eq!(scope.entries(), before);
}

#[test]
fn linked_rows_deep_clone_m_plus_q_and_partial_clone_iterative_cleanup() {
    let mut source = Chain::<u64, u8, u64>::empty();
    for id in 0..191_u8 {
        let mut record = allocate(1, 0, id).unwrap();
        record.push(u64::from(id)).unwrap();
        if id % 2 == 0 { *record.proofs_slot() = vec![u64::from(id)].into_boxed_slice(); }
        source.prepend(record);
    }
    let scope = AllocationScope::new(false);
    let mut cloned = source.clone();
    assert_eq!(scope.entries(), 191 + 96);
    for id in 0..191 {
        let a = source.find(id).unwrap();
        let b = cloned.find(id).unwrap();
        assert_eq!(a.rows(), b.rows());
        assert_eq!(a.proofs(), b.proofs());
        assert_ne!(a.rows().as_ptr(), b.rows().as_ptr());
        if !a.proofs().is_empty() { assert_ne!(a.proofs().as_ptr(), b.proofs().as_ptr()); }
    }
    cloned.find_mut(0).unwrap().rows_mut()[0] = 999;
    assert_eq!(source.find(0).unwrap().rows(), &[0]);
    let before = DROPS.with(Cell::get);
    CLONE_FAILURE.with(|n| n.set(Some(190)));
    assert!(catch_unwind(AssertUnwindSafe(|| source.clone())).is_err());
    assert_eq!(DROPS.with(Cell::get) - before, 191);
    assert_eq!(source.find(190).unwrap().rows(), &[190]);
    let before = DROPS.with(Cell::get);
    drop(cloned);
    drop(source);
    assert_eq!(DROPS.with(Cell::get) - before, 382);
}

#[test]
fn linked_rows_acquisition_refusal_has_no_partial_record() {
    let scope = AllocationScope::new(true);
    let drops = DROPS.with(Cell::get);
    assert!(matches!(allocate::<u64, u8, u64>(384, 0, 0), Err(Error::Allocation)));
    assert_eq!(scope.entries(), 1);
    assert_eq!(DROPS.with(Cell::get), drops);
}
