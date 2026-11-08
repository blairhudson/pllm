//! Thread-scoped allocator probe, separate from JSON/oracle reporters.
//! Path-imported unit tests may run concurrently without entering observation.
//! Actual null-return failure sweeps live in maverick_batch_failures.rs.
#[allow(dead_code, unused_imports)]
#[path = "../src/maverick_batch/mod.rs"]
mod maverick_batch;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use maverick_batch::testing::{self, Fixture};
use maverick_batch::*;

struct Tracking;
static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static CALLS: AtomicUsize = AtomicUsize::new(0);
static TRACK: AtomicBool = AtomicBool::new(false);
static PROBE: AtomicUsize = AtomicUsize::new(0);
static PROBE_BYTES: AtomicUsize = AtomicUsize::new(0);
static ERASED: AtomicBool = AtomicBool::new(false);

thread_local! {
    static OBSERVE: Cell<bool> = const { Cell::new(false) };
}

fn observing() -> bool {
    OBSERVE.try_with(Cell::get).unwrap_or(false)
}

struct Observation;

impl Observation {
    fn start() -> Self {
        assert!(!OBSERVE.with(|flag| flag.replace(true)));
        Self
    }
}

impl Drop for Observation {
    fn drop(&mut self) {
        OBSERVE.with(|flag| flag.set(false));
    }
}

fn allocated(size: usize) {
    if !observing() {
        return;
    }
    CALLS.fetch_add(1, Ordering::Relaxed);
    let current = CURRENT.fetch_add(size, Ordering::Relaxed) + size;
    if TRACK.load(Ordering::Relaxed) {
        PEAK.fetch_max(current, Ordering::Relaxed);
    }
}

// SAFETY: every request is forwarded unchanged to System. Accounting touches
// only a thread-local flag and atomics; the erasure probe inspects a still-live
// allocation before free.
unsafe impl GlobalAlloc for Tracking {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            allocated(layout.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            allocated(layout.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        if observing() {
            if PROBE
                .compare_exchange(p as usize, 0, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                let size = PROBE_BYTES.load(Ordering::Acquire);
                if size <= layout.size() {
                    // SAFETY: p remains allocated for layout.size() bytes here.
                    let bytes = unsafe { std::slice::from_raw_parts(p, size) };
                    ERASED.store(bytes.iter().all(|&b| b == 0), Ordering::Release);
                }
            }
            CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
        }
        unsafe { System.dealloc(p, layout) };
    }
    unsafe fn realloc(&self, p: *mut u8, old: Layout, new_size: usize) -> *mut u8 {
        let next = unsafe { System.realloc(p, old, new_size) };
        if !next.is_null() && observing() {
            CURRENT.fetch_sub(old.size(), Ordering::Relaxed);
            allocated(new_size);
        }
        next
    }
}

#[global_allocator]
static ALLOCATOR: Tracking = Tracking;

struct RetainedAttempt {
    requests: [Request; BATCH_SIZE],
    replies: [Reply; BATCH_SIZE],
    primary: PrimaryChallenge,
    auxiliary: AuxiliaryReply,
    accepted: AcceptedBatch,
}

fn word_charge(words: usize) -> usize {
    words * 4 + 128
}

// Ordinary regression inventory, not another admission policy. Payloads already
// charged by AllocationPlan stay separate from Words overhead and retained wires.
fn retention_inventory(plan: AllocationPlan) -> (usize, usize) {
    // Client: 2 M + 4 arrays for each of 4 codes + 2 P + 2 Q + Qaux.
    // Server: 2 M + 2 Q, with independent backing.
    let persistent_overhead = (2 + 4 * 4 + 2 + 2 + 1 + 2 + 2) * 128;
    let wires = MAX_ATTEMPTS as usize
        * (2 * BATCH_SIZE * word_charge(DIMENSION)
            + REPETITIONS * word_charge(2 * CHECK_WEIGHT)
            + word_charge(REPETITIONS * SECONDARY_DIMENSION));
    let accepted_overhead = MAX_ATTEMPTS as usize * BATCH_SIZE * 128;
    // Deliberately sum even disjoint online temporaries: both claims, originals,
    // encodings, secret primary/secondary checks, largest sampler pool, privacy
    // noise, private handoff, largest syndrome and encoder, auxiliary copy and
    // verification projection. Retained outbound copies are in wires above.
    let scratch = 4 * word_charge(DIMENSION)
        + 2 * word_charge(DIMENSION)
        + 2 * word_charge(LENGTH)
        + 2 * REPETITIONS * word_charge(2 * CHECK_WEIGHT)
        + word_charge(SECONDARY_LENGTH)
        + BATCH_SIZE * word_charge(2 * PRIVACY_WEIGHT)
        + word_charge(DIMENSION)
        + 2 * word_charge(SECONDARY_LENGTH)
        + word_charge(SECONDARY_DIMENSION)
        + word_charge(REPETITIONS * SECONDARY_DIMENSION)
        + 2 * word_charge(SECONDARY_LENGTH)
        + word_charge(DIMENSION);
    assert_eq!(persistent_overhead, 3_456);
    assert_eq!(wires, 55_040);
    assert_eq!(accepted_overhead, 4_096);
    assert_eq!(scratch, 34_736);
    assert!(scratch < 40 * 1024);
    let tail = persistent_overhead + wires + accepted_overhead + scratch;
    assert_eq!(tail, 97_328);
    assert!(tail < 103_552);
    assert!(tail < plan.transient_allowance_bytes());
    (persistent_overhead + wires + accepted_overhead, scratch)
}

#[test]
fn admission_and_all_retained_wire_output_owners_fit_exact_planned_ceiling() {
    let _observation = Observation::start();
    let calls = CALLS.load(Ordering::Acquire);
    let overflow = AllocationPlan::for_shape(usize::MAX, 64, 2, ALLOCATION_CEILING);
    let undersized = AllocationPlan::for_shape(64, 64, 2, 1024);
    let changed = AllocationPlan::for_shape(65, 64, 2, ALLOCATION_CEILING);
    let inflated_cap = AllocationPlan::for_shape(64, 64, 2, ALLOCATION_CEILING + 1);
    assert_eq!(CALLS.load(Ordering::Acquire), calls);
    assert_eq!(overflow.err(), Some(Error::Allocation));
    assert_eq!(undersized.err(), Some(Error::Allocation));
    assert_eq!(changed.err(), Some(Error::Shape));
    assert_eq!(inflated_cap.err(), Some(Error::Allocation));

    let fixture = Fixture::new();
    let mut bad = fixture.source();
    bad.primary_code.field = 2_147_483_659;
    let plan = AllocationPlan::bounded().unwrap();
    let calls = CALLS.load(Ordering::Acquire);
    let bad_field = preprocess(plan, bad);
    assert_eq!(CALLS.load(Ordering::Acquire), calls);
    assert_eq!(bad_field.err(), Some(Error::Field));
    assert_eq!(plan.public_payload_bytes(), 2_695_168);
    assert_eq!(plan.server_payload_bytes(), 294_912);
    assert_eq!(plan.retained_output_bytes(), 8192);
    assert_eq!(plan.planned_peak_bytes(), 6_479_872);
    let calls = CALLS.load(Ordering::Acquire);
    let one_short = AllocationPlan::for_shape(64, 64, 2, plan.planned_peak_bytes() - 1);
    let plan = AllocationPlan::for_shape(64, 64, 2, plan.planned_peak_bytes()).unwrap();
    assert_eq!(CALLS.load(Ordering::Acquire), calls);
    assert_eq!(one_short.err(), Some(Error::Allocation));
    let (retained_extra, scratch) = retention_inventory(plan);

    let baseline = CURRENT.load(Ordering::Acquire);
    PEAK.store(baseline, Ordering::Release);
    TRACK.store(true, Ordering::Release);
    let draft = preprocess(plan, fixture.source()).unwrap();
    let (mut client, mut server) = draft.validate(fixture.source()).unwrap().split().unwrap();
    testing::erase_probe(&client, |pointer, size| {
        PROBE_BYTES.store(size, Ordering::Release);
        PROBE.store(pointer as usize, Ordering::Release);
    })
    .unwrap();
    assert!(ERASED.load(Ordering::Acquire));

    let mut retained: [Option<RetainedAttempt>; MAX_ATTEMPTS as usize] =
        std::array::from_fn(|_| None);
    for (attempt, accepted) in retained.iter_mut().enumerate() {
        let (requests, replies, fixed, primary) = if attempt % 2 == 1 {
            let (owner, r0) = client.issue_dependent(&fixture.inputs[0]).unwrap();
            let y0 = server.evaluate(r0.view()).unwrap();
            let (owner, r1) = client.continue_dependent(owner, y0.view()).unwrap();
            let y1 = server.evaluate(r1.view()).unwrap();
            let (fixed, primary) = client
                .begin_dependent_verification(owner, y1.view())
                .unwrap();
            ([r0, r1], [y0, y1], fixed, primary)
        } else {
            let (batch, requests) = client
                .issue_batch([&fixture.inputs[0], &fixture.inputs[1]])
                .unwrap();
            let replies = [
                server.evaluate(requests[0].view()).unwrap(),
                server.evaluate(requests[1].view()).unwrap(),
            ];
            let (fixed, primary) = client
                .begin_verification(batch, &[replies[0].view(), replies[1].view()])
                .unwrap();
            (requests, replies, fixed, primary)
        };
        let auxiliary = server.project(primary.view()).unwrap();
        let output = client.finish_batch(fixed, auxiliary.view()).unwrap();
        *accepted = Some(RetainedAttempt {
            requests,
            replies,
            primary,
            auxiliary,
            accepted: output,
        });
    }
    for attempt in retained.iter().map(|attempt| attempt.as_ref().unwrap()) {
        assert_eq!(attempt.requests[0].view().values.len(), DIMENSION);
        assert_eq!(attempt.replies[1].view().values.len(), DIMENSION);
        assert_eq!(attempt.primary.view().entries[1].len(), 2 * CHECK_WEIGHT);
        assert_eq!(
            attempt.auxiliary.view().values.len(),
            REPETITIONS * SECONDARY_DIMENSION
        );
        assert_eq!(attempt.accepted.outputs()[0].values(), fixture.direct(0));
    }
    assert!(!client.has_live_window());
    assert_eq!(client.attempts_issued(), MAX_ATTEMPTS);
    let calls = CALLS.load(Ordering::Acquire);
    let exhausted = client.issue_dependent(&fixture.inputs[0]);
    assert_eq!(CALLS.load(Ordering::Acquire), calls);
    assert_eq!(exhausted.err(), Some(Error::Attempts));
    let stats = client.memory_stats();
    assert_eq!(stats.ceiling_bytes, plan.planned_peak_bytes());
    let retained_live = plan.external_payload_bytes()
        + plan.control_allowance_bytes()
        + plan.public_payload_bytes()
        + plan.server_payload_bytes()
        + plan.retained_output_bytes()
        + retained_extra;
    assert_eq!(retained_live, 6_018_176);
    assert_eq!(stats.charged_live_bytes, retained_live);
    assert!(stats.charged_peak_bytes <= retained_live + scratch);
    assert!(stats.charged_peak_bytes <= plan.planned_peak_bytes());
    assert_eq!(
        server.memory_stats().charged_peak_bytes,
        stats.charged_peak_bytes
    );
    drop(retained);
    drop(server);
    drop(client);
    TRACK.store(false, Ordering::Release);
    let incremental_peak = PEAK.load(Ordering::Acquire) - baseline;
    let conservative_peak =
        incremental_peak + plan.external_payload_bytes() + plan.control_allowance_bytes();
    assert!(
        conservative_peak <= plan.planned_peak_bytes(),
        "charged allocator peak {conservative_peak}"
    );
    assert!(conservative_peak < ALLOCATION_CEILING);
    assert_eq!(
        CURRENT.load(Ordering::Acquire),
        baseline,
        "no retained component allocations"
    );
    println!(
        "planned={} tracked={} allocator-plus-reserves={} cap={}",
        plan.planned_peak_bytes(),
        stats.charged_peak_bytes,
        conservative_peak,
        stats.ceiling_bytes
    );
}
