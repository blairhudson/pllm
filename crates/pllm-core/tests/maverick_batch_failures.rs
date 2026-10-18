//! Exhaustive dynamic allocation-index failures, using real GlobalAlloc nulls.
//! One test, one thread. Fixed Arc::new metadata allocations are NOT failed.
#[allow(dead_code, unused_imports)]
#[path = "../src/maverick_batch/mod.rs"]
mod maverick_batch;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use maverick_batch::testing::{self, Fixture, MemoryWatch};
use maverick_batch::*;

const BASE: usize = 2_695_168 + 262_144;
const WORD_OVERHEAD: usize = 128;
const VECTOR: usize = 64 * 4 + WORD_OVERHEAD;
const PRIMARY: usize = 2 * (2 * 31 * 4 + WORD_OVERHEAD);
const AUXILIARY: usize = 256 * 4 + WORD_OVERHEAD;

#[derive(Clone, Copy)]
struct Entry {
    pointer: usize,
    counter: usize,
    bytes: usize,
    charge: usize,
}
const EMPTY: Entry = Entry {
    pointer: 0,
    counter: 0,
    bytes: 0,
    charge: 0,
};
static LIVE: Mutex<[Entry; 128]> = Mutex::new([EMPTY; 128]);
static BAD: AtomicBool = AtomicBool::new(false);
static DESTRUCTIONS: AtomicUsize = AtomicUsize::new(0);
static ACTIVE: AtomicBool = AtomicBool::new(false);
static FAIL_AT: AtomicUsize = AtomicUsize::new(usize::MAX);
static INDEX: AtomicUsize = AtomicUsize::new(0);
static HIT: AtomicBool = AtomicBool::new(false);

fn fail_allocation() -> bool {
    if testing::allocation_context().is_some() && ACTIVE.load(Ordering::Relaxed) {
        let index = INDEX.fetch_add(1, Ordering::Relaxed);
        if index == FAIL_AT.load(Ordering::Relaxed) {
            HIT.store(true, Ordering::Relaxed);
            return true;
        }
    }
    false
}

fn record(pointer: *mut u8, layout: Layout) {
    if pointer.is_null() {
        return;
    }
    if let Some((counter, charge)) = testing::allocation_context() {
        let mut live = LIVE.lock().unwrap();
        let pending = live
            .iter()
            .filter(|e| e.counter == counter as usize)
            .map(|e| e.charge)
            .sum::<usize>();
        // SAFETY: the allocation scope holds the Memory Arc containing counter.
        let actual = unsafe { &*counter }.load(Ordering::Acquire);
        if charge != layout.size() + WORD_OVERHEAD || actual != BASE + pending + charge {
            BAD.store(true, Ordering::Relaxed);
        }
        if let Some(entry) = live.iter_mut().find(|e| e.pointer == 0) {
            *entry = Entry {
                pointer: pointer as usize,
                counter: counter as usize,
                bytes: layout.size(),
                charge,
            };
        } else {
            BAD.store(true, Ordering::Relaxed);
        }
    }
}

struct FailingAllocator;
// SAFETY: System receives original layouts/pointers. The only altered operation
// is returning null for a selected fallible Vec reservation. Inspection happens
// before System frees the initialized word buffer; observer counters are atomic
// and kept alive by that buffer's reservation. No allocator callback allocates.
unsafe impl GlobalAlloc for FailingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if fail_allocation() {
            return std::ptr::null_mut();
        }
        let pointer = unsafe { System.alloc(layout) };
        record(pointer, layout);
        pointer
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if fail_allocation() {
            return std::ptr::null_mut();
        }
        let pointer = unsafe { System.alloc_zeroed(layout) };
        record(pointer, layout);
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        {
            let mut live = LIVE.lock().unwrap();
            if let Some(index) = live.iter().position(|e| e.pointer == pointer as usize) {
                let entry = live[index];
                let pending = live
                    .iter()
                    .filter(|e| e.counter == entry.counter)
                    .map(|e| e.charge)
                    .sum::<usize>();
                // SAFETY: Words drops its data before its reservation/Memory Arc.
                let counter = unsafe { &*(entry.counter as *const AtomicUsize) };
                // Refunding before freeing, double refunds, and missing refunds
                // are observable here: all live reservations must still count.
                if counter.load(Ordering::Acquire) != BASE + pending || layout.size() != entry.bytes
                {
                    BAD.store(true, Ordering::Relaxed);
                }
                // SAFETY: successful Words construction initializes its entire
                // exact capacity; this callback runs before deallocation.
                let bytes = unsafe { std::slice::from_raw_parts(pointer, entry.bytes) };
                if bytes.iter().any(|&byte| byte != 0) {
                    BAD.store(true, Ordering::Relaxed);
                }
                DESTRUCTIONS.fetch_add(1, Ordering::Relaxed);
                live[index] = EMPTY;
            }
        }
        unsafe { System.dealloc(pointer, layout) };
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // Native Words never grows after its initial exact reservation.
        if testing::allocation_context().is_some() {
            BAD.store(true, Ordering::Relaxed);
            return std::ptr::null_mut();
        }
        unsafe { System.realloc(pointer, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: FailingAllocator = FailingAllocator;

struct Disarm;
impl Drop for Disarm {
    fn drop(&mut self) {
        ACTIVE.store(false, Ordering::Release);
    }
}

fn inject<T>(index: usize, operation: impl FnOnce() -> Result<T>) -> (Result<T>, usize) {
    INDEX.store(0, Ordering::Release);
    HIT.store(false, Ordering::Release);
    FAIL_AT.store(index, Ordering::Release);
    ACTIVE.store(true, Ordering::Release);
    let guard = Disarm;
    let result = operation();
    drop(guard);
    let calls = INDEX.load(Ordering::Acquire);
    if index == usize::MAX {
        assert!(!HIT.load(Ordering::Acquire));
        assert!(
            result.is_ok(),
            "successful control run: {:?}",
            result.as_ref().err()
        );
    } else {
        assert!(
            HIT.load(Ordering::Acquire),
            "allocation index {index} was not exercised"
        );
        assert_eq!(calls, index + 1);
        assert_eq!(result.as_ref().err(), Some(&Error::Allocation));
    }
    assert!(
        !BAD.load(Ordering::Acquire),
        "erasure/reservation/refund ordering violation"
    );
    (result, calls)
}

fn retired(watch: &MemoryWatch) {
    assert_eq!(watch.stats().charged_live_bytes, BASE);
    assert_eq!(
        watch.owner_count(),
        1,
        "only the non-secret observer may remain"
    );
    assert!(!BAD.load(Ordering::Acquire));
}

fn sweep(name: &str, mut run: impl FnMut(usize) -> usize) -> usize {
    let count = run(usize::MAX);
    assert!(count > 0);
    for index in 0..count {
        run(index);
    }
    println!("{name}: {count} actual dynamic allocation failures checked");
    count
}

#[derive(Clone, Copy)]
enum Online {
    IssueQuery,
    IssueBatch,
    Evaluate,
    FinishQuery,
    Begin,
    Project,
    FinishBatch,
}

fn online_case(
    fixture: &Fixture,
    parts: PublicationParts<'_>,
    phase: Online,
    fail: usize,
) -> usize {
    testing::capture_next_memory();
    let draft =
        UnvalidatedPublication::import(AllocationPlan::bounded().unwrap(), fixture.source(), parts)
            .unwrap();
    let watch = testing::take_memory_watch();
    let (mut client, mut server) = draft.validate(fixture.source()).unwrap().split().unwrap();
    let idle = client.memory_stats().charged_live_bytes;
    let count;
    match phase {
        Online::IssueQuery => {
            let (result, calls) = inject(fail, || client.issue_query(0, &fixture.inputs[0]));
            if let Ok((query, request)) = result {
                drop(query);
                assert_eq!(watch.stats().charged_live_bytes, idle + VECTOR);
                drop(request);
            }
            count = calls;
        }
        Online::IssueBatch => {
            let (result, calls) = inject(fail, || {
                client.issue_batch([&fixture.inputs[0], &fixture.inputs[1]])
            });
            if let Ok((batch, requests)) = result {
                drop(batch);
                assert_eq!(watch.stats().charged_live_bytes, idle + 2 * VECTOR);
                drop(requests);
            }
            count = calls;
        }
        Online::Evaluate | Online::FinishQuery => {
            let (query, request) = client.issue_query(0, &fixture.inputs[0]).unwrap();
            if matches!(phase, Online::Evaluate) {
                let (result, calls) = inject(fail, || server.evaluate(request.view()));
                drop(result);
                assert_eq!(server.evaluate(request.view()).err(), Some(Error::Replay));
                drop(query); // Client-side cancellation after server transport failure.
                assert_eq!(watch.stats().charged_live_bytes, idle + VECTOR);
                count = calls;
            } else {
                let response = server.evaluate(request.view()).unwrap();
                let (result, calls) = inject(fail, || client.finish_query(query, response.view()));
                drop(result);
                assert_eq!(watch.stats().charged_live_bytes, idle + 2 * VECTOR);
                drop(response);
                count = calls;
            }
            drop(request);
        }
        Online::Begin | Online::Project | Online::FinishBatch => {
            let (batch, requests) = client
                .issue_batch([&fixture.inputs[0], &fixture.inputs[1]])
                .unwrap();
            let replies = [
                server.evaluate(requests[0].view()).unwrap(),
                server.evaluate(requests[1].view()).unwrap(),
            ];
            if matches!(phase, Online::Begin) {
                let (result, calls) = inject(fail, || {
                    client.begin_verification(batch, &[replies[0].view(), replies[1].view()])
                });
                if let Ok((fixed, primary)) = result {
                    drop(fixed);
                    assert_eq!(
                        watch.stats().charged_live_bytes,
                        idle + 4 * VECTOR + PRIMARY
                    );
                    drop(primary);
                }
                count = calls;
            } else {
                let (fixed, primary) = client
                    .begin_verification(batch, &[replies[0].view(), replies[1].view()])
                    .unwrap();
                if matches!(phase, Online::Project) {
                    let (result, calls) = inject(fail, || server.project(primary.view()));
                    drop(result);
                    assert_eq!(server.project(primary.view()).err(), Some(Error::Replay));
                    drop(fixed); // The client owns and cancels the failed window.
                    count = calls;
                } else {
                    let auxiliary = server.project(primary.view()).unwrap();
                    let (result, calls) =
                        inject(fail, || client.finish_batch(fixed, auxiliary.view()));
                    drop(result);
                    assert_eq!(
                        watch.stats().charged_live_bytes,
                        idle + 4 * VECTOR + PRIMARY + AUXILIARY
                    );
                    drop(auxiliary);
                    count = calls;
                }
                assert_eq!(
                    watch.stats().charged_live_bytes,
                    idle + 4 * VECTOR + PRIMARY
                );
                drop(primary);
            }
            assert_eq!(watch.stats().charged_live_bytes, idle + 4 * VECTOR);
            drop(replies);
            drop(requests);
        }
    }
    assert!(!client.has_live_window());
    assert_eq!(client.attempts_issued(), 1);
    assert_eq!(watch.stats().charged_live_bytes, idle);
    // A failed attempt is retired, not a stuck lease; the next owner has a new ID.
    let (next, request) = client.issue_query(0, &fixture.inputs[0]).unwrap();
    assert_eq!(request.view().header.attempt, 1);
    drop(next);
    drop(request);
    assert!(!client.has_live_window());
    drop(client);
    drop(server);
    retired(&watch);
    count
}

#[test]
fn every_dynamic_allocation_index_fails_closed_with_exact_refunds_and_erasure() {
    let fixture = Fixture::new();
    let plan = AllocationPlan::bounded().unwrap();
    let master = preprocess(plan, fixture.source()).unwrap();
    let parts = master.view().derived;
    let mut total = sweep("import", |index| {
        testing::capture_next_memory();
        let (result, calls) = inject(index, || {
            UnvalidatedPublication::import(plan, fixture.source(), parts)
        });
        let watch = testing::take_memory_watch();
        drop(result);
        retired(&watch);
        calls
    });
    total += sweep("full P/Q/Qaux validation", |index| {
        testing::capture_next_memory();
        let draft = UnvalidatedPublication::import(plan, fixture.source(), parts).unwrap();
        let watch = testing::take_memory_watch();
        let (result, calls) = inject(index, || draft.validate(fixture.source()));
        drop(result);
        retired(&watch);
        calls
    });
    total += sweep("party split", |index| {
        testing::capture_next_memory();
        let draft = UnvalidatedPublication::import(plan, fixture.source(), parts).unwrap();
        let watch = testing::take_memory_watch();
        let validated = draft.validate(fixture.source()).unwrap();
        let (result, calls) = inject(index, || validated.split());
        drop(result);
        retired(&watch);
        calls
    });
    for (name, phase) in [
        ("query issuance", Online::IssueQuery),
        ("batch issuance", Online::IssueBatch),
        ("server evaluation", Online::Evaluate),
        ("Protocol 3 verification", Online::FinishQuery),
        ("original fixation / primary reveal", Online::Begin),
        ("server auxiliary response", Online::Project),
        (
            "auxiliary fixation / Protocol 2 verification",
            Online::FinishBatch,
        ),
    ] {
        total += sweep(name, |index| online_case(&fixture, parts, phase, index));
    }
    drop(master);
    assert!(LIVE.lock().unwrap().iter().all(|entry| entry.pointer == 0));
    assert!(!BAD.load(Ordering::Acquire));
    println!(
        "{total} injected failures; {} buffers checked erased-before-free and refund-after-free",
        DESTRUCTIONS.load(Ordering::Acquire)
    );
}
