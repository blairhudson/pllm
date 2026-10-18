//! Tiny standalone installation check. No Poulpy crate or HE-sized allocations.
#[path = "../../src/bounded_allocator.rs"]
mod bounded_allocator;

use bounded_allocator::{BoundedAllocator, SystemMapper, BOOKKEEPING_TICKET};

#[global_allocator]
static ALLOCATOR: BoundedAllocator = BoundedAllocator::new(SystemMapper, 8 << 20, 128, BOOKKEEPING_TICKET);

fn main() {
    let before = ALLOCATOR.snapshot();
    {
        // Exercise the same global allocation entry points as Vec and AlignedBuf.
        let mut values = Vec::<u32>::with_capacity(1024);
        values.extend(0..1024);
        values.reserve_exact(1024);
        assert_eq!(values[1023], 1023);
        let layout = std::alloc::Layout::from_size_align(64 << 10, 64).unwrap();
        unsafe {
            let p = std::alloc::alloc_zeroed(layout);
            assert!(!p.is_null());
            assert_eq!((p as usize) % 64, 0);
            assert!(std::slice::from_raw_parts(p, layout.size()).iter().all(|&v| v == 0));
            std::alloc::dealloc(p, layout);
        }
    }
    let after = ALLOCATOR.snapshot();
    assert_eq!((after.charged, after.mapped, after.mappings), (before.charged, before.mapped, before.mappings));
    assert!(!after.poisoned);
    assert_eq!(after.denied, before.denied);
    println!("allocator-global-smoke: passed; no HE; peak tracked charge {}", after.peak_charged);
}
