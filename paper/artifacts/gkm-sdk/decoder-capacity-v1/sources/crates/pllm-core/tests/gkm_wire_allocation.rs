//! Independent requested-allocation oracle. Unsafe code is confined to this test
//! allocator; production wire owners remain safe Rust and use fallible payloads.
use pllm_core::gkm::*;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    sync::Arc,
};

#[derive(Clone, Copy, Default)]
struct Trace {
    enabled: bool,
    fail: usize,
    fail_nth: usize,
    matches: usize,
    failures: usize,
    calls: usize,
    live: isize,
    peak: isize,
}
thread_local! { static TRACE: Cell<Trace> = const { Cell::new(Trace {
    enabled: false, fail: 0, fail_nth: 0, matches: 0, failures: 0, calls: 0, live: 0, peak: 0,
}) }; }
fn reject(bytes: usize) -> bool {
    TRACE
        .try_with(|slot| {
            let mut trace = slot.get();
            if trace.enabled && trace.fail == bytes {
                trace.matches += 1;
                let rejected = trace.matches == trace.fail_nth;
                trace.failures += usize::from(rejected);
                slot.set(trace);
                rejected
            } else {
                false
            }
        })
        .unwrap_or(false)
}

#[derive(Clone, Copy)]
struct Releases {
    capacity: Capacity,
    sizes: [usize; 5],
    count: usize,
    minimum_wire_bytes: usize,
}
thread_local! { static RELEASES: Cell<Option<Releases>> = const { Cell::new(None) }; }

fn observe_release(bytes: usize) {
    let _ = RELEASES.try_with(|slot| {
        if let Some(mut releases) = slot.get() {
            if releases.sizes.contains(&bytes) {
                // Observe before System actually frees the payload. Looking only
                // after the constructor returns would miss an early permit drop.
                releases.count += 1;
                releases.minimum_wire_bytes = releases.minimum_wire_bytes.min(
                    WireResources::current_with_capacity(releases.capacity).process_live_bytes,
                );
                slot.set(Some(releases));
            }
        }
    });
}

struct ReleaseReset;
impl Drop for ReleaseReset {
    fn drop(&mut self) {
        RELEASES.with(|slot| slot.set(None));
    }
}

fn releases<T>(capacity: Capacity, sizes: [usize; 5], run: impl FnOnce() -> T) -> (T, Releases) {
    RELEASES.with(|slot| {
        assert!(slot.get().is_none());
        slot.set(Some(Releases {
            capacity,
            sizes,
            count: 0,
            minimum_wire_bytes: usize::MAX,
        }));
    });
    let reset = ReleaseReset;
    let result = run();
    let measured = RELEASES.with(|slot| slot.get().unwrap());
    drop(reset);
    (result, measured)
}
fn account(delta: isize, allocation: bool) {
    let _ = TRACE.try_with(|slot| {
        let mut trace = slot.get();
        if trace.enabled {
            trace.calls += usize::from(allocation);
            trace.live += delta;
            trace.peak = trace.peak.max(trace.live);
            slot.set(trace);
        }
    });
}
struct Allocator;
// SAFETY: unmodified valid pointer/layout pairs are forwarded to System;
// injected allocation failures return null without taking ownership.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if reject(layout.size()) {
            return std::ptr::null_mut();
        }
        // SAFETY: GlobalAlloc supplies a valid allocation layout.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            account(layout.size() as isize, true);
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        observe_release(layout.size());
        account(-(layout.size() as isize), false);
        // SAFETY: caller supplies the same live pointer/layout pair.
        unsafe {
            System.dealloc(ptr, layout);
        }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, bytes: usize) -> *mut u8 {
        if reject(bytes) {
            return std::ptr::null_mut();
        }
        // SAFETY: caller's valid pointer/layout/new size are forwarded unchanged.
        let next = unsafe { System.realloc(ptr, layout, bytes) };
        if !next.is_null() {
            account(bytes as isize - layout.size() as isize, true);
        }
        next
    }
}
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;
struct Reset;
impl Drop for Reset {
    fn drop(&mut self) {
        TRACE.with(|t| t.set(Trace::default()));
    }
}
fn trace<T>(fail: usize, run: impl FnOnce() -> T) -> (T, Trace) {
    trace_nth(fail, 1, run)
}
fn trace_nth<T>(fail: usize, fail_nth: usize, run: impl FnOnce() -> T) -> (T, Trace) {
    assert!(fail_nth > 0);
    TRACE.with(|t| {
        assert!(!t.get().enabled);
        t.set(Trace {
            enabled: true,
            fail,
            fail_nth,
            ..Trace::default()
        });
    });
    let reset = Reset;
    let value = run();
    let measured = TRACE.with(Cell::get);
    drop(reset);
    (value, measured)
}
fn failed<T>(result: (Result<T, Error>, Trace)) {
    assert!(matches!(result.0, Err(Error::Allocation)));
    assert_eq!(result.1.failures, 1);
}

#[test]
fn preallocation_rejection_real_null_failures_and_maximum_source_copy_accounting() {
    worker_constructor_admission_survives_actual_payload_deallocation();
    // Exact-size source buffers transfer into immutable numeric and worker
    // owners. Fail allocations of their payload sizes to detect any hidden copy.
    let owned_weights = vec![3i16; 11 * 13];
    let bounds = vec![100u64; 13];
    let (owned_matrix, measured) = trace(owned_weights.len() * 2, || {
        I16Matrix::from_encoded_owned(11, 13, owned_weights, bounds)
    });
    let owned_matrix = owned_matrix.unwrap();
    assert_eq!(measured.failures, 0);
    let raw = vec![0u8; 17 * 13 * 4];
    let (owned_table, measured) = trace(raw.len(), || FloatTable::from_le_bytes_owned(17, 13, raw));
    let owned_table = owned_table.unwrap();
    assert_eq!(measured.failures, 0);
    let public = Arc::new(
        Publication::publish(
            [2; 32],
            &[Stage {
                id: [1; 32],
                matrix: &owned_matrix,
            }],
            &owned_table,
        )
        .unwrap(),
    );
    let (worker, measured) = trace(17 * 13 * 4, || {
        WorkerWire::from_owned(
            Arc::clone(&public),
            public.digest(),
            Worker::Zero,
            &[[1; 32]],
            vec![owned_matrix],
            owned_table,
        )
    });
    assert_eq!(measured.failures, 0);
    drop(worker.unwrap());
    drop(public);
    assert_eq!(WireResources::current().process_live_bytes, 0);

    let matrix = I16Matrix::from_encoded(1, 1, &[3], &[100]).unwrap();
    let table = FloatTable::from_le_bytes(1, 1, &[0; 4]).unwrap();
    let stages = [Stage {
        id: [1; 32],
        matrix: &matrix,
    }];
    let public = Arc::new(Publication::publish([2; 32], &stages, &table).unwrap());
    let packet = public.export_wire().unwrap();
    for end in 0..packet.len() {
        let (result, measured) = trace(0, || {
            Publication::import_wire(&packet.as_bytes()[..end], public.digest())
        });
        assert!(result.is_err());
        assert_eq!(measured.calls, 0);
    }
    let (result, measured) = trace(0, || Publication::import_wire(packet.as_bytes(), [0; 32]));
    assert!(result.is_err());
    assert_eq!(measured.calls, 0);
    failed(trace(packet.len(), || public.export_wire()));
    failed(trace(32, || {
        Publication::import_wire(packet.as_bytes(), public.digest())
    }));
    let wire_before = WireResources::current().process_live_bytes;
    failed(trace(2, || {
        WorkerWire::new(
            Arc::clone(&public),
            public.digest(),
            Worker::Zero,
            &stages,
            &table,
        )
    }));
    assert_eq!(WireResources::current().process_live_bytes, wire_before);
    let client = Client::new(Arc::clone(&public), public.digest()).unwrap();
    let mut window = client
        .begin_wire([3; 32], &[Step::Linear([1; 32])])
        .unwrap();
    let starts = window.start_frames().unwrap();
    let mut worker = WorkerWire::new(
        Arc::clone(&public),
        public.digest(),
        Worker::Zero,
        &stages,
        &table,
    )
    .unwrap();
    let mut request_owner = worker.begin(starts[0].as_bytes()).unwrap();
    let requests = window.linear(&[7]).unwrap();
    failed(trace(304, || request_owner.execute(requests[0].as_bytes())));
    assert!(request_owner.execute(requests[0].as_bytes()).is_err());
    window.abort();
    drop((
        requests,
        starts,
        window,
        request_owner,
        worker,
        client,
        packet,
        public,
    ));
    assert_eq!(WireResources::current().process_live_bytes, 0);

    // Maximum original table is 4 MiB. Each party-local copy is separately
    // charged to the additional 8 MiB pool; two copies plus owners cannot fit in
    // one process. Separate actual worker processes have separate logical pools.
    let raw = vec![0u8; MAX_VOCABULARY * MAX_DIMENSION * 4];
    let table = FloatTable::from_le_bytes(MAX_VOCABULARY, MAX_DIMENSION, &raw).unwrap();
    let public = Arc::new(Publication::publish([2; 32], &stages, &table).unwrap());
    let before = WireResources::current().process_live_bytes;
    let (first, measured) = trace(0, || {
        WorkerWire::new(
            Arc::clone(&public),
            public.digest(),
            Worker::Zero,
            &stages,
            &table,
        )
    });
    let first = first.unwrap();
    let admitted = WireResources::current().process_live_bytes - before;
    assert!(measured.peak as usize + size_of::<WorkerWire>() <= admitted);
    assert!(admitted > raw.len());
    let (second, measured) = trace(0, || {
        WorkerWire::new(
            Arc::clone(&public),
            public.digest(),
            Worker::One,
            &stages,
            &table,
        )
    });
    assert!(matches!(second, Err(Error::Limit)));
    assert_eq!(measured.calls, 0); // capacity rejected before copying public source
    assert_eq!(
        WireResources::current().process_live_bytes,
        before + admitted
    );
    drop(first);
    assert_eq!(WireResources::current().process_live_bytes, before);

    // Reimporting always pays another complete native envelope. Exhaustion is
    // detected before allocation, and dropping imported owners restores room.
    let packet = public.export_wire().unwrap();
    let mut imports = Vec::new();
    loop {
        let (result, measured) = trace(0, || {
            Publication::import_wire(packet.as_bytes(), public.digest())
        });
        match result {
            Ok(imported) => imports.push(imported),
            Err(Error::Limit) => {
                assert_eq!(measured.calls, 0);
                break;
            }
            Err(error) => panic!("unexpected {error:?}"),
        }
    }
    assert!(!imports.is_empty());
    drop(imports);
    assert!(Publication::import_wire(packet.as_bytes(), public.digest()).is_ok());
    drop((packet, public, table, raw));
    maximum_full_window_peak();
    decoder_capacity_peak_failures_and_process_admission();
}

fn worker_constructor_admission_survives_actual_payload_deallocation() {
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        let matrix = || {
            I16Matrix::from_encoded_owned_with_capacity(
                11,
                13,
                vec![3; 11 * 13],
                vec![100; 13],
                capacity,
            )
            .unwrap()
        };
        let table = || {
            FloatTable::from_le_bytes_owned_with_capacity(17, 7, vec![0; 17 * 7 * 4], capacity)
                .unwrap()
        };
        let original_matrix = matrix();
        let original_table = table();
        let stages = [Stage {
            id: [1; 32],
            matrix: &original_matrix,
        }];
        let public = Arc::new(
            Publication::publish_with_capacity([2; 32], &stages, &original_table, capacity)
                .unwrap(),
        );
        let packet = public.export_wire().unwrap();
        let live = || WireResources::current_with_capacity(capacity).process_live_bytes;
        let before = live();
        let new_worker = || {
            WorkerWire::new(
                Arc::clone(&public),
                public.digest(),
                Worker::Zero,
                &stages,
                &original_table,
            )
        };
        // Distinct source allocation sizes: matrix owner array, weights, input
        // bounds, output bounds, and table. None coincide with fixed Arc handles.
        let sizes = [
            size_of::<I16Matrix>(),
            11 * 13 * 2,
            13 * 8,
            11 * 8,
            17 * 7 * 4,
        ];
        let worker = new_worker().unwrap();
        let admitted = live() - before;
        assert!(admitted > sizes.iter().sum());
        let ((), released) = releases(capacity, sizes, || drop(worker));
        assert_eq!(released.count, 5);
        assert!(released.minimum_wire_bytes >= before + admitted);
        assert_eq!(live(), before);

        let mut clients: Vec<_> = (0..MAX_CLIENTS_PER_PUBLICATION)
            .map(|_| Client::new(Arc::clone(&public), public.digest()).unwrap())
            .collect();
        let matrices = vec![matrix()];
        let owned_table = table();
        let ((result, measured), released) = releases(capacity, sizes, || {
            trace(0, || {
                WorkerWire::from_owned(
                    Arc::clone(&public),
                    public.digest(),
                    Worker::Zero,
                    &[[1; 32]],
                    matrices,
                    owned_table,
                )
            })
        });
        assert!(matches!(result, Err(Error::Limit)));
        assert_eq!(measured.calls, 0);
        assert_eq!(released.count, 5);
        assert!(
            released.minimum_wire_bytes >= before + admitted,
            "source permit refunded before payload deallocation: observed {}, required {}",
            released.minimum_wire_bytes,
            before + admitted
        );
        assert_eq!(live(), before);

        let ((result, measured), released) = releases(capacity, sizes, || trace(0, new_worker));
        assert!(matches!(result, Err(Error::Limit)));
        assert_eq!(measured.calls, 0); // Handle exhaustion precedes every copy/allocation.
        assert_eq!(released.count, 0);
        assert_eq!(live(), before);
        drop(clients.pop());
        drop(new_worker().unwrap()); // Failed admission did not consume the freed slot.
        drop(clients);

        // Recoverable failures after each copied payload keep the source permit
        // alive through destruction of all partially constructed owners.
        for (index, &fail) in sizes.iter().enumerate() {
            let (result, released) = releases(capacity, sizes, || trace(fail, new_worker));
            failed(result);
            assert_eq!(released.count, index);
            if index != 0 {
                assert!(released.minimum_wire_bytes >= before + admitted);
            }
            assert_eq!(live(), before);
        }

        // Binding validation is a preflight: borrowed sources are never copied,
        // moved invalid sources retire under their caller's existing admission,
        // and no publication-handle or wire reservation is consumed.
        let (result, measured) = trace(0, || {
            WorkerWire::new(
                Arc::clone(&public),
                [0; 32],
                Worker::Zero,
                &stages,
                &original_table,
            )
        });
        assert!(matches!(result, Err(Error::Binding)));
        assert_eq!(measured.calls, 0);
        let matrices = vec![matrix()];
        let owned_table = table();
        let ((result, measured), released) = releases(capacity, sizes, || {
            trace(0, || {
                WorkerWire::from_owned(
                    Arc::clone(&public),
                    public.digest(),
                    Worker::Zero,
                    &[[9; 32]],
                    matrices,
                    owned_table,
                )
            })
        });
        assert!(matches!(result, Err(Error::Binding)));
        assert_eq!(measured.calls, 0);
        assert_eq!(released.count, 5);
        assert_eq!(released.minimum_wire_bytes, before);
        assert_eq!(live(), before);
        let clients: Vec<_> = (0..MAX_CLIENTS_PER_PUBLICATION)
            .map(|_| Client::new(Arc::clone(&public), public.digest()).unwrap())
            .collect();
        drop((clients, packet, public));
        assert_eq!(
            WireResources::current_with_capacity(capacity).process_live_bytes,
            0
        );
    }
}

fn decoder_capacity_peak_failures_and_process_admission() {
    let cap = Capacity::Decoder;
    for (rows, columns) in [(1031, 3), (2, 1033)] {
        let weights = vec![1i16; rows * columns];
        let bounds = vec![8u64; columns];
        let (matrix, measured) = trace(weights.len() * 2, || {
            I16Matrix::from_encoded_owned_with_capacity(rows, columns, weights, bounds, cap)
        });
        let matrix = matrix.unwrap();
        assert_eq!(measured.failures, 0); // no hidden source-weight copy
        let table = FloatTable::from_le_bytes_owned_with_capacity(1, 1, vec![0; 4], cap).unwrap();
        let stages = [Stage {
            id: [1; 32],
            matrix: &matrix,
        }];
        let public =
            Arc::new(Publication::publish_with_capacity([2; 32], &stages, &table, cap).unwrap());
        let exported = public.export_wire().unwrap();
        failed(trace(rows * 8, || {
            Publication::import_wire_with_capacity(exported.as_bytes(), public.digest(), cap)
        }));
        let client = Client::new(Arc::clone(&public), public.digest()).unwrap();
        failed(trace(rows * 32, || {
            client.begin([3; 32], &[Step::Linear([1; 32])])
        }));
        assert_eq!(public.live_windows(), 0);

        // Every newly dynamic worker buffer, including encoded output, fails
        // recoverably and makes that role's one-use request unrecoverable.
        for (fail, nth) in [
            (columns * 8, 1),
            (rows * 8, 1),
            (rows * 8, 2),
            (296 + rows * 8, 1),
        ] {
            let mut worker = WorkerWire::new(
                Arc::clone(&public),
                public.digest(),
                Worker::Zero,
                &stages,
                &table,
            )
            .unwrap();
            let mut window = client
                .begin_wire([3; 32], &[Step::Linear([1; 32])])
                .unwrap();
            let start = window.start_frames().unwrap();
            let mut left = worker.begin(start[0].as_bytes()).unwrap();
            let input = vec![2; columns];
            let request = window.linear(&input).unwrap();
            let before = WireResources::current_with_capacity(cap).process_live_bytes;
            let result = trace_nth(fail, nth, || left.execute(request[0].as_bytes()));
            assert_eq!(result.1.matches, nth);
            if nth == 2 {
                // Input and u64 output both allocated successfully. Reject the
                // following equally sized encoded-u8 body, not multiply_share.
                assert_eq!(result.1.calls, 2);
            }
            failed(result);
            assert!(matches!(
                left.execute(request[0].as_bytes()),
                Err(Error::Retired)
            ));
            assert_eq!(
                WireResources::current_with_capacity(cap).process_live_bytes,
                before
            );
            drop((request, start, left, worker, window));
            assert_eq!(public.live_windows(), 0);
        }

        let before = WireResources::current_with_capacity(cap).process_live_bytes;
        let input = vec![2; columns];
        let mut wire_peak = 0;
        let ((), measured) = trace(0, || {
            let mut zero = WorkerWire::new(
                Arc::clone(&public),
                public.digest(),
                Worker::Zero,
                &stages,
                &table,
            )
            .unwrap();
            let mut one = WorkerWire::new(
                Arc::clone(&public),
                public.digest(),
                Worker::One,
                &stages,
                &table,
            )
            .unwrap();
            let mut window = client
                .begin_wire([3; 32], &[Step::Linear([1; 32]); 2])
                .unwrap();
            let start = window.start_frames().unwrap();
            let mut left = zero.begin(start[0].as_bytes()).unwrap();
            let mut right = one.begin(start[1].as_bytes()).unwrap();
            for _ in 0..2 {
                let request = window.linear(&input).unwrap();
                let a = left.execute(request[0].as_bytes()).unwrap();
                let b = right.execute(request[1].as_bytes()).unwrap();
                wire_peak = wire_peak
                    .max(WireResources::current_with_capacity(cap).process_live_bytes - before);
                let tentative = window.accept_replies(a.as_bytes(), b.as_bytes()).unwrap();
                assert!(tentative.signed.iter().all(|&v| v == 2 * columns as i64));
            }
            left.finish().unwrap();
            right.finish().unwrap();
            window.finish().unwrap();
        });
        assert!(
            measured.peak as usize <= 3 * public.resources().max_window_payload_bytes + wire_peak
        );
        assert_eq!(measured.live, 0);
        assert_eq!(public.live_windows(), 0);
    }
    assert_eq!(
        WireResources::current_with_capacity(cap).process_live_bytes,
        0
    );

    // Exhaust the 4 GiB logical envelope with tiny publications, not large
    // allocations. Each import reserves full maximum-window capacity first.
    let matrix = I16Matrix::from_encoded(1, 1, &[1], &[8]).unwrap();
    let table = FloatTable::from_le_bytes(1, 1, &[0; 4]).unwrap();
    let public = Publication::publish_with_capacity(
        [2; 32],
        &[Stage {
            id: [1; 32],
            matrix: &matrix,
        }],
        &table,
        cap,
    )
    .unwrap();
    let frame = public.export_wire().unwrap();
    let mut imports = Vec::new();
    loop {
        let (result, measured) = trace(0, || {
            Publication::import_wire_with_capacity(frame.as_bytes(), public.digest(), cap)
        });
        match result {
            Ok(imported) => imports.push(imported),
            Err(Error::Limit) => {
                assert_eq!(measured.calls, 0);
                break;
            }
            Err(other) => panic!("unexpected {other:?}"),
        }
    }
    assert_eq!(
        imports.len() + 1,
        cap.max_process_publication_bytes() / public.resources().admitted_payload_bytes
    );
    drop(imports);
    assert!(Publication::import_wire_with_capacity(frame.as_bytes(), public.digest(), cap).is_ok());
}

fn maximum_full_window_peak() {
    let mut raw = vec![0i16; MAX_DIMENSION * MAX_DIMENSION];
    for row in 0..MAX_DIMENSION {
        raw[row * MAX_DIMENSION + row] = row as i16 + 1;
    }
    let matrix =
        I16Matrix::from_encoded(MAX_DIMENSION, MAX_DIMENSION, &raw, &[128; MAX_DIMENSION]).unwrap();
    let table = FloatTable::from_le_bytes(1, 1, &[0; 4]).unwrap();
    let stages = [Stage {
        id: [1; 32],
        matrix: &matrix,
    }];
    let public = Arc::new(Publication::publish([2; 32], &stages, &table).unwrap());
    let original = public.resources().max_window_payload_bytes * 3;
    let input = [-123i64; MAX_DIMENSION];
    let mut admitted_wire_peak = 0;
    let ((), measured) = trace(0, || {
        let client = Client::new(Arc::clone(&public), public.digest()).unwrap();
        let mut zero = WorkerWire::new(
            Arc::clone(&public),
            public.digest(),
            Worker::Zero,
            &stages,
            &table,
        )
        .unwrap();
        let mut one = WorkerWire::new(
            Arc::clone(&public),
            public.digest(),
            Worker::One,
            &stages,
            &table,
        )
        .unwrap();
        let mut window = client
            .begin_wire([3; 32], &[Step::Linear([1; 32]); MAX_WINDOW_STEPS])
            .unwrap();
        let start = window.start_frames().unwrap();
        assert_eq!(start[0].len(), 296 + 33 * MAX_WINDOW_STEPS);
        let mut left = zero.begin(start[0].as_bytes()).unwrap();
        let mut right = one.begin(start[1].as_bytes()).unwrap();
        for _ in 0..MAX_WINDOW_STEPS {
            let request = window.linear(&input).unwrap();
            assert_eq!(request[0].len(), 296 + MAX_DIMENSION * 8);
            let a = left.execute(request[0].as_bytes()).unwrap();
            let b = right.execute(request[1].as_bytes()).unwrap();
            admitted_wire_peak =
                admitted_wire_peak.max(WireResources::current().process_live_bytes);
            let result = window.accept_replies(a.as_bytes(), b.as_bytes()).unwrap();
            for (row, (&signed, &float)) in result.signed.iter().zip(result.values).enumerate() {
                let exact = -123i128 * (row as i128 + 1);
                assert_eq!(signed as i128, exact);
                assert_eq!(
                    float.to_bits(),
                    ((exact as i64 as f32) / 4_194_304.0f32).to_bits()
                );
            }
        }
        left.finish().unwrap();
        right.finish().unwrap();
        assert_eq!(window.finish().unwrap().steps(), MAX_WINDOW_STEPS);
    });
    // Sources, native accumulator/pending/reconstruction, new party-local
    // schedules, retained start/request/reply buffers and final checks overlap.
    // New fixed owner sizes already occur in their explicit wire reservation.
    assert!(measured.peak as usize <= original + admitted_wire_peak);
    assert_eq!(measured.live, 0);
    assert_eq!(public.live_windows(), 0);
    assert_eq!(WireResources::current().process_live_bytes, 0);
}
