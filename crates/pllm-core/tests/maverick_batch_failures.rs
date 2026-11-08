//! Exhaustive dynamic allocation-index failures, using real GlobalAlloc nulls.
//! Instrument only the sweep thread; embedded tests may run concurrently.
//! Fixed Arc::new metadata allocations are NOT failed.
#[allow(dead_code, unused_imports)]
#[path = "../src/maverick_batch/mod.rs"]
mod maverick_batch;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
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
static LEASE_DESTRUCTIONS: AtomicUsize = AtomicUsize::new(0);
static ACTIVE: AtomicBool = AtomicBool::new(false);
static FAIL_AT: AtomicUsize = AtomicUsize::new(usize::MAX);
static INDEX: AtomicUsize = AtomicUsize::new(0);
static HIT: AtomicBool = AtomicBool::new(false);

thread_local! {
    static AUDIT_OWNER: Cell<bool> = const { Cell::new(false) };
}

fn audit_owner() -> bool {
    AUDIT_OWNER.try_with(Cell::get).unwrap_or(false)
}

struct AuditOwner;
impl AuditOwner {
    fn start() -> Self {
        AUDIT_OWNER
            .try_with(|owner| {
                assert!(!owner.get(), "allocation audit owner must not nest");
                owner.set(true);
            })
            .expect("allocation audit owner TLS unavailable");
        Self
    }
}
impl Drop for AuditOwner {
    fn drop(&mut self) {
        let _ = AUDIT_OWNER.try_with(|owner| owner.set(false));
    }
}

fn fail_allocation() -> bool {
    if !audit_owner() {
        return false;
    }
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
    if !audit_owner() || pointer.is_null() {
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
        if audit_owner() {
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
                if let Some(live) = testing::observed_lease_is_live() {
                    if !live {
                        BAD.store(true, Ordering::Relaxed);
                    }
                    LEASE_DESTRUCTIONS.fetch_add(1, Ordering::Relaxed);
                }
                DESTRUCTIONS.fetch_add(1, Ordering::Relaxed);
                live[index] = EMPTY;
            }
        }
        unsafe { System.dealloc(pointer, layout) };
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // Native Words never grows after its initial exact reservation.
        if audit_owner() && testing::allocation_context().is_some() {
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

fn inject_client<T>(
    index: usize,
    client: &mut Client,
    operation: impl FnOnce(&mut Client) -> Result<T>,
) -> (Result<T>, usize) {
    let had_owner = client.has_live_window();
    let before = LEASE_DESTRUCTIONS.load(Ordering::Acquire);
    let result = inject(index, || testing::with_live_lease(client, operation));
    // An issuance failure at allocation zero has no successful buffer to free.
    // Every other consuming phase must actually exercise the live-flag probe.
    if had_owner || index != 0 {
        assert!(LEASE_DESTRUCTIONS.load(Ordering::Acquire) > before);
    }
    result
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
    IssueDependent,
    EvaluateDependentFirst,
    ContinueDependent,
    EvaluateDependentSecond,
    BeginDependent,
    ProjectDependent,
    FinishDependent,
}

fn dependent_case(
    client: &mut Client,
    server: &mut Server,
    fixture: &Fixture,
    phase: Online,
    fail: usize,
) -> usize {
    if matches!(phase, Online::IssueDependent) {
        let (result, count) = inject_client(fail, client, |client| {
            client.issue_dependent(&fixture.inputs[0])
        });
        if let Ok((owner, request)) = result {
            testing::with_live_lease(client, |_| drop(owner));
            drop(request);
        }
        return count;
    }
    let (owner, r0) = client.issue_dependent(&fixture.inputs[0]).unwrap();
    if matches!(phase, Online::EvaluateDependentFirst) {
        let (result, count) = inject(fail, || server.evaluate(r0.view()));
        drop(result);
        assert_eq!(server.evaluate(r0.view()).err(), Some(Error::Replay));
        testing::with_live_lease(client, |_| drop(owner)); // Transport cancellation.
        return count;
    }
    let y0 = server.evaluate(r0.view()).unwrap();
    if matches!(phase, Online::ContinueDependent) {
        let (result, count) = inject_client(fail, client, |client| {
            client.continue_dependent(owner, y0.view())
        });
        if let Ok((owner, request)) = result {
            testing::with_live_lease(client, |_| drop(owner));
            drop(request);
        }
        return count;
    }
    let (owner, r1) = client.continue_dependent(owner, y0.view()).unwrap();
    if matches!(phase, Online::EvaluateDependentSecond) {
        let (result, count) = inject(fail, || server.evaluate(r1.view()));
        drop(result);
        assert_eq!(server.evaluate(r1.view()).err(), Some(Error::Replay));
        testing::with_live_lease(client, |_| drop(owner));
        return count;
    }
    let y1 = server.evaluate(r1.view()).unwrap();
    if matches!(phase, Online::BeginDependent) {
        let (result, count) = inject_client(fail, client, |client| {
            client.begin_dependent_verification(owner, y1.view())
        });
        if let Ok((owner, primary)) = result {
            testing::with_live_lease(client, |_| drop(owner));
            drop(primary);
        }
        return count;
    }
    let (fixed, primary) = client
        .begin_dependent_verification(owner, y1.view())
        .unwrap();
    if matches!(phase, Online::ProjectDependent) {
        let (result, count) = inject(fail, || server.project(primary.view()));
        drop(result);
        assert_eq!(server.project(primary.view()).err(), Some(Error::Replay));
        testing::with_live_lease(client, |_| drop(fixed));
        return count;
    }
    let auxiliary = server.project(primary.view()).unwrap();
    let (result, count) = inject_client(fail, client, |client| {
        client.finish_batch(fixed, auxiliary.view())
    });
    drop(result);
    count
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
        Online::IssueDependent
        | Online::EvaluateDependentFirst
        | Online::ContinueDependent
        | Online::EvaluateDependentSecond
        | Online::BeginDependent
        | Online::ProjectDependent
        | Online::FinishDependent => {
            count = dependent_case(&mut client, &mut server, fixture, phase, fail);
        }
        Online::IssueQuery => {
            let (result, calls) = inject_client(fail, &mut client, |client| {
                client.issue_query(0, &fixture.inputs[0])
            });
            if let Ok((query, request)) = result {
                testing::with_live_lease(&mut client, |_| drop(query));
                assert_eq!(watch.stats().charged_live_bytes, idle + VECTOR);
                drop(request);
            }
            count = calls;
        }
        Online::IssueBatch => {
            let (result, calls) = inject_client(fail, &mut client, |client| {
                client.issue_batch([&fixture.inputs[0], &fixture.inputs[1]])
            });
            if let Ok((batch, requests)) = result {
                testing::with_live_lease(&mut client, |_| drop(batch));
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
                testing::with_live_lease(&mut client, |_| drop(query));
                assert_eq!(watch.stats().charged_live_bytes, idle + VECTOR);
                count = calls;
            } else {
                let response = server.evaluate(request.view()).unwrap();
                let (result, calls) = inject_client(fail, &mut client, |client| {
                    client.finish_query(query, response.view())
                });
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
                let (result, calls) = inject_client(fail, &mut client, |client| {
                    client.begin_verification(batch, &[replies[0].view(), replies[1].view()])
                });
                if let Ok((fixed, primary)) = result {
                    testing::with_live_lease(&mut client, |_| drop(fixed));
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
                    testing::with_live_lease(&mut client, |_| drop(fixed));
                    count = calls;
                } else {
                    let auxiliary = server.project(primary.view()).unwrap();
                    let (result, calls) = inject_client(fail, &mut client, |client| {
                        client.finish_batch(fixed, auxiliary.view())
                    });
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
    testing::with_live_lease(&mut client, |_| drop(next));
    drop(request);
    assert!(!client.has_live_window());
    drop(client);
    drop(server);
    retired(&watch);
    count
}

// Same allocator/erasure/reservation probe as the allocation-index sweep. These
// cases additionally inspect the real live-window flag, not a mirrored counter.
fn cancellation_rejection_and_partial_entropy_keep_lease_until_free(
    fixture: &Fixture,
    parts: PublicationParts<'_>,
) {
    testing::capture_next_memory();
    let draft =
        UnvalidatedPublication::import(AllocationPlan::bounded().unwrap(), fixture.source(), parts)
            .unwrap();
    let watch = testing::take_memory_watch();
    let (mut client, mut server) = draft.validate(fixture.source()).unwrap().split().unwrap();
    let idle = client.memory_stats().charged_live_bytes;
    let partial_privacy = 2 * PRIVACY_WEIGHT - 1; // 164 pairs + last position.
    let second_check = 4 * CHECK_WEIGHT - 1; // First 31 pairs, then 30 + position.
    let before = LEASE_DESTRUCTIONS.load(Ordering::Acquire);
    let failed = testing::with_live_lease(&mut client, |client| {
        testing::issue_dependent_rng_failure_after(client, &fixture.inputs[0], partial_privacy)
    });
    assert_eq!(failed.err(), Some(Error::Randomness));
    assert!(LEASE_DESTRUCTIONS.load(Ordering::Acquire) > before);
    assert!(!client.has_live_window());
    assert_eq!(client.attempts_issued(), 1);
    assert_eq!(watch.stats().charged_live_bytes, idle);

    for case in 0..8 {
        let before = LEASE_DESTRUCTIONS.load(Ordering::Acquire);
        {
            let (owner, r0) =
                testing::issue_dependent_fixed(&mut client, &fixture.inputs[0], &fixture.noise[0])
                    .unwrap();
            if case == 0 {
                testing::with_live_lease(&mut client, |_| drop(owner));
            } else {
                let y0 = server.evaluate(r0.view()).unwrap();
                let mut original = y0.view().values.to_vec();
                if case == 6 {
                    original[0] = (original[0] + 1) % FIELD;
                }
                let first = ReplyView {
                    header: y0.view().header,
                    values: &original,
                };
                if case == 1 {
                    let failed = testing::with_live_lease(&mut client, |client| {
                        testing::continue_dependent_rng_failure_after(
                            client,
                            owner,
                            first,
                            partial_privacy,
                        )
                    });
                    assert_eq!(failed.err(), Some(Error::Randomness));
                } else {
                    let (owner, r1) = testing::continue_dependent_fixed(
                        &mut client,
                        owner,
                        first,
                        &fixture.noise[1],
                    )
                    .unwrap();
                    let y1 = server.evaluate(r1.view()).unwrap();
                    if case == 2 {
                        testing::with_live_lease(&mut client, |_| drop(owner));
                    } else if case == 3 {
                        let failed = testing::with_live_lease(&mut client, |client| {
                            testing::begin_dependent_rng_failure_after(
                                client,
                                owner,
                                y1.view(),
                                second_check,
                            )
                        });
                        assert_eq!(failed.err(), Some(Error::Randomness));
                    } else {
                        let (fixed, primary) = testing::begin_dependent_fixed(
                            &mut client,
                            owner,
                            y1.view(),
                            [
                                &fixture.primary_challenges[0],
                                &fixture.primary_challenges[1],
                            ],
                        )
                        .unwrap();
                        if case == 4 {
                            testing::with_live_lease(&mut client, |_| drop(fixed));
                        } else {
                            let auxiliary = server.project(primary.view()).unwrap();
                            if case == 5 {
                                let failed = testing::with_live_lease(&mut client, |client| {
                                    testing::finish_batch_rng_failure_after(
                                        client,
                                        fixed,
                                        auxiliary.view(),
                                        second_check,
                                    )
                                });
                                assert_eq!(failed.err(), Some(Error::Randomness));
                            } else {
                                let mut values = auxiliary.view().values.to_vec();
                                if case == 7 {
                                    values[0] = (values[0] + 1) % FIELD;
                                }
                                let failed = testing::with_live_lease(&mut client, |client| {
                                    testing::finish_batch_fixed(
                                        client,
                                        fixed,
                                        AuxiliaryReplyView {
                                            header: auxiliary.view().header,
                                            values: &values,
                                        },
                                        [
                                            &fixture.secondary_challenges[0],
                                            &fixture.secondary_challenges[1],
                                        ],
                                    )
                                });
                                assert_eq!(failed.err(), Some(Error::Rejected));
                            }
                        }
                    }
                }
            }
        }
        assert!(
            LEASE_DESTRUCTIONS.load(Ordering::Acquire) > before,
            "case {case} observed no owner destruction"
        );
        assert!(!client.has_live_window());
        assert_eq!(client.attempts_issued(), case + 2);
        assert_eq!(watch.stats().charged_live_bytes, idle);
        assert!(!BAD.load(Ordering::Acquire));
    }
    let (owner, request) = client.issue_dependent(&fixture.inputs[0]).unwrap();
    assert_eq!(request.view().header.attempt, 9); // No failed attempt got recycled.
    testing::with_live_lease(&mut client, |_| drop(owner));
    drop(request);
    assert_eq!(watch.stats().charged_live_bytes, idle);
    drop(client);
    drop(server);
    retired(&watch);
}

#[test]
fn every_dynamic_allocation_index_fails_closed_with_exact_refunds_and_erasure() {
    let _audit_owner = AuditOwner::start();
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
        ("dependent issuance", Online::IssueDependent),
        ("dependent first evaluation", Online::EvaluateDependentFirst),
        (
            "dependent first fixation / private handoff",
            Online::ContinueDependent,
        ),
        (
            "dependent second evaluation",
            Online::EvaluateDependentSecond,
        ),
        (
            "dependent second fixation / primary reveal",
            Online::BeginDependent,
        ),
        ("dependent auxiliary response", Online::ProjectDependent),
        ("dependent final verification", Online::FinishDependent),
    ] {
        total += sweep(name, |index| online_case(&fixture, parts, phase, index));
    }
    cancellation_rejection_and_partial_entropy_keep_lease_until_free(&fixture, parts);
    drop(master);
    assert!(LIVE.lock().unwrap().iter().all(|entry| entry.pointer == 0));
    assert!(!BAD.load(Ordering::Acquire));
    assert!(LEASE_DESTRUCTIONS.load(Ordering::Acquire) > 0);
    println!(
        "{total} injected failures; {} buffers checked erased-before-free and refund-after-free; {} frees checked with original live lease",
        DESTRUCTIONS.load(Ordering::Acquire), LEASE_DESTRUCTIONS.load(Ordering::Acquire)
    );
}
