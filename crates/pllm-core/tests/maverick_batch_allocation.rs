//! Dedicated single-test allocator probe, separate from JSON/oracle reporters.
//! Actual null-return failure sweeps live in maverick_batch_failures.rs.
#[allow(dead_code, unused_imports)]
#[path = "../src/maverick_batch/mod.rs"]
mod maverick_batch;

use std::alloc::{GlobalAlloc, Layout, System};
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

fn allocated(size: usize) {
    CALLS.fetch_add(1, Ordering::Relaxed);
    let current = CURRENT.fetch_add(size, Ordering::Relaxed) + size;
    if TRACK.load(Ordering::Relaxed) {
        PEAK.fetch_max(current, Ordering::Relaxed);
    }
}

// SAFETY: every request is forwarded unchanged to System. Accounting touches
// only atomics; the erasure probe inspects a still-live allocation before free.
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
        unsafe { System.dealloc(p, layout) };
    }
    unsafe fn realloc(&self, p: *mut u8, old: Layout, new_size: usize) -> *mut u8 {
        let next = unsafe { System.realloc(p, old, new_size) };
        if !next.is_null() {
            CURRENT.fetch_sub(old.size(), Ordering::Relaxed);
            allocated(new_size);
        }
        next
    }
}

#[global_allocator]
static ALLOCATOR: Tracking = Tracking;

#[test]
fn admission_precedes_allocation_and_complete_capsule_lifetime_stays_bounded() {
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

    let mut retained: [Option<AcceptedBatch>; MAX_ATTEMPTS as usize] =
        std::array::from_fn(|_| None);
    for accepted in &mut retained {
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
        let auxiliary = server.project(primary.view()).unwrap();
        *accepted = Some(client.finish_batch(fixed, auxiliary.view()).unwrap());
    }
    assert!(!client.has_live_window());
    let stats = client.memory_stats();
    assert_eq!(stats.ceiling_bytes, ALLOCATION_CEILING);
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
        ALLOCATION_CEILING
    );
}
