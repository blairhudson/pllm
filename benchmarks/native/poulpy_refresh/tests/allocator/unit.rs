//! Included only by rustc --test on the standalone allocator source.
//! Nested here (without main.rs) to avoid Cargo integration-test autodiscovery.
use super::*;

fn allocator(limit: usize, mappings: usize) -> BoundedAllocator {
    BoundedAllocator::new(SystemMapper, limit, mappings, BOOKKEEPING_TICKET)
}

#[test]
fn geometry_checks_zero_overflow_and_bookkeeping() {
    assert_eq!(HEADER_BYTES, 64);
    assert_eq!(std::mem::size_of::<BoundedAllocator>(), 88);
    assert!(geometry(Layout::from_size_align(0, 1).unwrap(), PAGE_BYTES, 0).is_none());
    assert!(geometry(Layout::from_size_align(1, 1).unwrap(), 0, 0).is_none());
    assert!(geometry(Layout::from_size_align(1, 1).unwrap(), 1000, 0).is_none());
    assert!(geometry(Layout::from_size_align(isize::MAX as usize, 1).unwrap(), PAGE_BYTES, 0).is_none());
    assert!(geometry(Layout::from_size_align(1, 1).unwrap(), PAGE_BYTES, usize::MAX).is_none());
    for size in [1, 63, 64, PAGE_BYTES - 64, PAGE_BYTES, 3 * PAGE_BYTES] {
        for exponent in 0..=18 {
            let layout = Layout::from_size_align(size, 1 << exponent).unwrap();
            let g = geometry(layout, PAGE_BYTES, BOOKKEEPING_TICKET).unwrap();
            assert_eq!(g.mapped % PAGE_BYTES, 0);
            assert_eq!(g.charge, g.mapped + BOOKKEEPING_TICKET);
            for base in [PAGE_BYTES, 3 * PAGE_BYTES, 64 * PAGE_BYTES] {
                let user = round_up(base + HEADER_BYTES, g.alignment).unwrap();
                assert!(user - base <= g.maximum_offset);
                assert!(user - base >= HEADER_BYTES);
                assert!(user - base + size <= g.mapped);
            }
        }
    }
}

#[test]
fn anonymous_zeroing_alignment_and_release() {
    let a = allocator(4 << 20, 8);
    assert_eq!(SystemMapper.page_size(), PAGE_BYTES);
    for alignment in [1, 16, 64, PAGE_BYTES, 4 * PAGE_BYTES] {
        let layout = Layout::from_size_align(PAGE_BYTES + 3, alignment).unwrap();
        let g = geometry(layout, PAGE_BYTES, BOOKKEEPING_TICKET).unwrap();
        unsafe {
            let p = a.alloc_zeroed(layout);
            assert!(!p.is_null());
            assert_eq!((p as usize) % alignment, 0);
            assert!(std::slice::from_raw_parts(p, layout.size()).iter().all(|&v| v == 0));
            p.write_bytes(0xa5, layout.size());
            assert_eq!(a.snapshot().charged, g.charge);
            a.dealloc(p, layout);
        }
        let s = a.snapshot();
        assert_eq!((s.charged, s.mapped, s.mappings), (0, 0, 0));
        assert!(!s.poisoned);
    }
}

#[test]
fn cap_is_inclusive_and_map_count_is_separate() {
    let l = Layout::from_size_align(20, 64).unwrap();
    let charge = geometry(l, PAGE_BYTES, BOOKKEEPING_TICKET).unwrap().charge;
    let a = allocator(charge, 2);
    unsafe {
        let p = a.alloc(l);
        assert!(!p.is_null());
        assert!(a.alloc(l).is_null());
        assert_eq!(a.snapshot().charged, charge);
        a.dealloc(p, l);
    }
    let b = allocator(2 * charge, 1);
    unsafe {
        let p = b.alloc(l);
        assert!(!p.is_null());
        assert!(b.alloc(l).is_null());
        b.dealloc(p, l);
    }
    assert_eq!(b.snapshot().mappings, 0);
    assert_eq!(a.snapshot().peak_charged, charge);
}

#[test]
fn realloc_counts_overlap_and_failure_preserves_original() {
    let l = Layout::from_size_align(128, 64).unwrap();
    let charge = geometry(l, PAGE_BYTES, BOOKKEEPING_TICKET).unwrap().charge;
    let a = allocator(charge, 4);
    unsafe {
        let p = a.alloc(l);
        assert!(!p.is_null());
        p.write_bytes(0x51, l.size());
        assert!(a.realloc(p, l, 256).is_null());
        assert!(a.realloc(p, l, 64).is_null());
        assert!(a.realloc(p, l, 0).is_null());
        assert!(a.realloc(p, l, usize::MAX).is_null());
        assert_eq!(a.realloc(p, l, l.size()), p);
        assert!(std::slice::from_raw_parts(p, l.size()).iter().all(|&v| v == 0x51));
        a.dealloc(p, l);
    }
    let b = allocator(2 * charge, 4);
    unsafe {
        let p = b.alloc(l);
        assert!(!p.is_null());
        p.write_bytes(0x72, l.size());
        let q = b.realloc(p, l, 256);
        assert!(!q.is_null());
        assert!(std::slice::from_raw_parts(q, l.size()).iter().all(|&v| v == 0x72));
        assert_eq!(b.snapshot().peak_charged, 2 * charge);
        assert_eq!(b.snapshot().charged, charge);
        let r = b.realloc(q, Layout::from_size_align(256, 64).unwrap(), 64);
        assert!(!r.is_null());
        assert!(std::slice::from_raw_parts(r, 64).iter().all(|&v| v == 0x72));
        b.dealloc(r, Layout::from_size_align(64, 64).unwrap());
    }
    assert_eq!(b.snapshot().charged, 0);
}

struct FailingMap;
unsafe impl Mapper for FailingMap {
    fn page_size(&self) -> usize { PAGE_BYTES }
    unsafe fn map(&self, _length: usize) -> *mut u8 { MAP_FAILED }
    unsafe fn unmap(&self, _address: *mut u8, _length: usize) -> i32 { -1 }
}

#[test]
fn failed_mapping_rolls_back_all_reservations() {
    let a = BoundedAllocator::new(FailingMap, 1 << 20, 4, BOOKKEEPING_TICKET);
    let layout = Layout::from_size_align(1, 1).unwrap();
    unsafe { assert!(a.alloc(layout).is_null()); }
    let s = a.snapshot();
    assert_eq!((s.charged, s.mapped, s.mappings), (0, 0, 0));
    assert_eq!((s.map_failures, s.denied), (1, 1));
}

struct WrongPage;
unsafe impl Mapper for WrongPage {
    fn page_size(&self) -> usize { 4096 }
    unsafe fn map(&self, _length: usize) -> *mut u8 { MAP_FAILED }
    unsafe fn unmap(&self, _address: *mut u8, _length: usize) -> i32 { -1 }
}

#[test]
fn wrong_page_and_impossible_budgets_reject_before_mapping() {
    let l = Layout::from_size_align(128, 64).unwrap();
    let wrong = BoundedAllocator::new(WrongPage, 1 << 20, 4, BOOKKEEPING_TICKET);
    unsafe { assert!(wrong.alloc(l).is_null()); }
    assert_eq!(wrong.snapshot().map_failures, 0);
    for (limit, maps) in [(0, 1), (1 << 20, 0), (1, 1)] {
        let a = BoundedAllocator::new(FailingMap, limit, maps, BOOKKEEPING_TICKET);
        unsafe { assert!(a.alloc(l).is_null()); }
        let s = a.snapshot();
        assert_eq!((s.charged, s.mappings, s.map_failures), (0, 0, 0));
    }
}

struct FailingUnmap;
unsafe impl Mapper for FailingUnmap {
    fn page_size(&self) -> usize { PAGE_BYTES }
    unsafe fn map(&self, length: usize) -> *mut u8 { unsafe { SystemMapper.map(length) } }
    unsafe fn unmap(&self, _address: *mut u8, _length: usize) -> i32 { -1 }
}

#[test]
fn failed_unmap_poison_retains_charge_without_allocating_diagnostic() {
    let a = BoundedAllocator::new(FailingUnmap, 1 << 20, 4, BOOKKEEPING_TICKET);
    let layout = Layout::from_size_align(1, 1).unwrap();
    unsafe {
        let p = a.alloc(layout);
        assert!(!p.is_null());
        let h = a.header(p, layout);
        a.dealloc(p, layout);
        let s = a.snapshot();
        assert!(s.poisoned);
        assert_eq!(s.unmap_failures, 1);
        assert_eq!((s.charged, s.mapped, s.mappings), (h.charge, h.mapped, 1));
        assert!(a.alloc(layout).is_null());
        // Test-only cleanup of the deliberate failed-unmap extent. The poisoned
        // allocator keeps its conservative charge; no production release bypass.
        assert_eq!(SystemMapper.unmap(h.base, h.mapped), 0);
    }
}

#[test]
fn repeated_private_vector_shape_has_no_retained_mapping_growth() {
    let a = allocator(2 << 20, 4);
    let l = Layout::from_size_align(32 << 10, 4).unwrap();
    let charge = geometry(l, PAGE_BYTES, BOOKKEEPING_TICKET).unwrap().charge;
    for _ in 0..462 {
        unsafe {
            let p = a.alloc_zeroed(l);
            assert!(!p.is_null());
            p.write(7);
            a.dealloc(p, l);
        }
        assert_eq!(a.snapshot().charged, 0);
    }
    assert_eq!(a.snapshot().peak_charged, charge);
}
