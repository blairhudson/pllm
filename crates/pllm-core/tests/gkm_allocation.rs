//! Real allocator regression, isolated in this native test binary. Production
//! GKM code remains safe Rust; unsafe here is only the GlobalAlloc test oracle.
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    sync::Arc,
};

use pllm_core::gkm::*;

#[derive(Clone, Copy, Default)]
struct Trace {
    enabled: bool,
    fail_size: usize,
    failures: usize,
    live: isize,
    peak: isize,
    largest: usize,
}

thread_local! {
    static TRACE: Cell<Trace> = const { Cell::new(Trace {
        enabled: false, fail_size: 0, failures: 0, live: 0, peak: 0, largest: 0,
    }) };
}

fn reject(size: usize) -> bool {
    TRACE
        .try_with(|slot| {
            let mut trace = slot.get();
            if trace.enabled && trace.fail_size == size {
                trace.failures += 1;
                slot.set(trace);
                true
            } else {
                false
            }
        })
        .unwrap_or(false)
}

fn account(delta: isize, allocation: usize) {
    let _ = TRACE.try_with(|slot| {
        let mut trace = slot.get();
        if trace.enabled {
            trace.live += delta;
            trace.peak = trace.peak.max(trace.live);
            trace.largest = trace.largest.max(allocation);
            slot.set(trace);
        }
    });
}

struct Tracked;
// SAFETY: successful operations forward the original layout/pointer to System;
// injected failures return null without taking ownership or freeing anything.
unsafe impl GlobalAlloc for Tracked {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if reject(layout.size()) {
            return std::ptr::null_mut();
        }
        // SAFETY: GlobalAlloc supplies a valid nonzero allocation layout.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            account(layout.size() as isize, layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        account(-(layout.size() as isize), 0);
        // SAFETY: pointer/layout are unchanged from the caller's valid pair.
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if reject(size) {
            return std::ptr::null_mut();
        }
        // SAFETY: forward the caller's valid pointer/layout/new size unchanged.
        let next = unsafe { System.realloc(pointer, layout, size) };
        if !next.is_null() {
            account(size as isize - layout.size() as isize, size);
        }
        next
    }
}

#[global_allocator]
static ALLOCATOR: Tracked = Tracked;

struct Reset;
impl Drop for Reset {
    fn drop(&mut self) {
        TRACE.with(|slot| slot.set(Trace::default()));
    }
}

fn traced<T>(fail_size: usize, action: impl FnOnce() -> T) -> (T, Trace) {
    TRACE.with(|slot| {
        assert!(!slot.get().enabled);
        slot.set(Trace {
            enabled: true,
            fail_size,
            ..Trace::default()
        });
    });
    let reset = Reset;
    let result = action();
    let trace = TRACE.with(Cell::get);
    drop(reset);
    (result, trace)
}

fn allocation_error<T>((result, trace): (Result<T, Error>, Trace)) {
    assert!(matches!(result, Err(Error::Allocation)));
    assert_eq!(trace.failures, 1);
}

fn linear_window(client: &Client, matrix: &I16Matrix) -> Window {
    let pending = client
        .begin([0; 32], &[Step::Linear([1; 32])])
        .unwrap()
        .linear(&[5])
        .unwrap();
    let first = matrix.evaluate(&pending.share(Worker::Zero));
    let half = pending.freeze_first(first).unwrap();
    let second = matrix.evaluate(&half.remaining_share());
    half.freeze_second(second).unwrap()
}

#[test]
fn actual_null_payload_allocations_return_errors_and_import_peak_matches_receipt() {
    // The independent audit's previous four-byte allocation caused SIGABRT.
    allocation_error(traced(4, || FloatTable::from_le_bytes(1, 1, &[0; 4])));
    allocation_error(traced(2, || I16Matrix::from_encoded(1, 1, &[1], &[100])));
    allocation_error(traced(2, || I16Matrix::from_f64(1, 1, &[0.5], &[100])));

    let weights = vec![0.0; 256 * 129];
    let receipt = PublicationResources::for_shapes(&[(256, 129)], 1, 1).unwrap();
    let (imported, trace) = traced(0, || I16Matrix::from_f64(256, 129, &weights, &[1; 129]));
    let imported = imported.unwrap();
    assert_eq!(
        trace.peak - trace.live,
        receipt.max_import_workspace_bytes as isize
    );
    assert_eq!(trace.largest, 256 * 129 * 2); // 66048, never growth to 131072
    drop(imported);

    let matrix = I16Matrix::from_encoded(1, 1, &[1], &[100]).unwrap();
    let table = FloatTable::from_le_bytes(1, 1, &[0; 4]).unwrap();
    allocation_error(traced(8, || matrix.multiply_share(&[1])));
    allocation_error(traced(4, || table.xor_rows(&[1])));
    let stages = [Stage {
        id: [1; 32],
        matrix: &matrix,
    }];
    allocation_error(traced(32, || {
        Publication::publish([0; 32], &stages, &table)
    }));
    let publication = Arc::new(Publication::publish([0; 32], &stages, &table).unwrap());
    let client = Client::new(Arc::clone(&publication), publication.digest()).unwrap();

    // Avoid the documented fixed Arc handle allocation (32 bytes). The first
    // variable-size accumulator array has two Vecs and one usize per matrix.
    let accumulator_bytes = 2 * size_of::<Vec<u8>>() + size_of::<usize>();
    allocation_error(traced(accumulator_bytes, || {
        client.begin([0; 32], &[Step::Linear([1; 32])])
    }));
    assert_eq!(publication.live_windows(), 0);
    let window = client.begin([0; 32], &[Step::Linear([1; 32])]).unwrap();
    allocation_error(traced(8, || window.linear(&[5])));
    assert_eq!(publication.live_windows(), 0);

    for decode in [false, true] {
        let pending = client
            .begin([0; 32], &[Step::Linear([1; 32])])
            .unwrap()
            .linear(&[5])
            .unwrap();
        let share = pending.share(Worker::Zero);
        allocation_error(traced(8, || {
            if decode {
                LinearResponse::from_untrusted(&share, share.binding(), &[0])
            } else {
                matrix.evaluate(&share)
            }
        }));
        assert!(matches!(matrix.evaluate(&share), Err(Error::Retired)));
        drop(pending);
        assert_eq!(publication.live_windows(), 0);
    }

    let pending = client
        .begin([0; 32], &[Step::Linear([1; 32])])
        .unwrap()
        .linear(&[5])
        .unwrap();
    let first = matrix.evaluate(&pending.share(Worker::Zero));
    let half = pending.freeze_first(first).unwrap();
    let second = matrix.evaluate(&half.remaining_share());
    allocation_error(traced(8, || half.freeze_second(second)));
    assert_eq!(publication.live_windows(), 0);
    let window = linear_window(&client, &matrix);
    allocation_error(traced(32, || window.finish()));
    assert_eq!(publication.live_windows(), 0);

    let window = client.begin([0; 32], &[Step::Lookup]).unwrap();
    allocation_error(traced(1, || window.lookup(0)));
    for decode in [false, true] {
        let pending = client
            .begin([0; 32], &[Step::Lookup])
            .unwrap()
            .lookup(0)
            .unwrap();
        let share = pending.share(Worker::Zero);
        allocation_error(traced(4, || {
            if decode {
                LookupResponse::from_untrusted(&share, share.binding(), &[0; 4])
            } else {
                table.evaluate(&share)
            }
        }));
        drop(pending);
    }
    let pending = client
        .begin([0; 32], &[Step::Lookup])
        .unwrap()
        .lookup(0)
        .unwrap();
    let first = table.evaluate(&pending.share(Worker::Zero));
    let half = pending.freeze_first(first).unwrap();
    let second = table.evaluate(&half.remaining_share());
    allocation_error(traced(4, || half.freeze_second(second)));
    assert_eq!(publication.live_windows(), 0);
    assert!(linear_window(&client, &matrix).finish().is_ok());
}
