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

#[test]
fn seeded_null_payload_failures_and_full_window_peak_keep_all_owner_reservations() {
    let matrix = I16Matrix::from_encoded(1, 2, &[3, -2], &[100; 2]).unwrap();
    let table = FloatTable::from_le_bytes(1, 1, &(-0.0_f32).to_le_bytes()).unwrap();
    let publication = Arc::new(
        Publication::publish(
            [1; 32],
            &[Stage {
                id: [2; 32],
                matrix: &matrix,
            }],
            &table,
        )
        .unwrap(),
    );
    let client = Client::new(Arc::clone(&publication), publication.digest()).unwrap();
    let plan = [Step::Linear([2; 32])];
    let accumulator_bytes = 2 * size_of::<Vec<u8>>() + size_of::<usize>();
    allocation_error(traced(accumulator_bytes, || {
        client.begin_seeded([3; 32], &plan)
    }));
    assert_eq!(publication.live_windows(), 0);
    for size in [16, 8] {
        // local expanded input, then response allocation
        let (window, mut worker) = client.begin_seeded([3; 32], &plan).unwrap();
        let pending = window.linear(&[7, -11]).unwrap();
        let share = pending.share(Worker::One);
        allocation_error(traced(size, || worker.evaluate_linear(&matrix, &share)));
        assert!(matches!(
            worker.evaluate_linear(&matrix, &share),
            Err(Error::Retired)
        ));
        drop(pending);
        assert_eq!(publication.live_windows(), 1);
        drop(worker);
        assert_eq!(publication.live_windows(), 0);
    }
    for size in [1, 4] {
        // expanded packed selector, then exact float-byte response
        let (window, mut worker) = client.begin_seeded([3; 32], &[Step::Lookup]).unwrap();
        let pending = window.lookup(0).unwrap();
        let share = pending.share(Worker::One);
        allocation_error(traced(size, || worker.evaluate_lookup(&table, &share)));
        assert!(matches!(
            worker.evaluate_lookup(&table, &share),
            Err(Error::Retired)
        ));
        drop(pending);
        assert_eq!(publication.live_windows(), 1);
        drop(worker);
        assert_eq!(publication.live_windows(), 0);
    }
    let (window, worker) = client.begin_seeded([3; 32], &plan).unwrap();
    allocation_error(traced(16, || window.linear(&[7, -11])));
    assert_eq!(publication.live_windows(), 1);
    drop(worker);

    let ((accepted, worker), trace) = traced(0, || {
        let (mut window, mut worker) = client
            .begin_seeded([3; 32], &[Step::Lookup; MAX_WINDOW_STEPS])
            .unwrap();
        for _ in 0..MAX_WINDOW_STEPS {
            let pending = window.lookup(0).unwrap();
            let first = table.evaluate(&pending.share(Worker::Zero));
            let half = pending.freeze_first(first).unwrap();
            let second = worker.evaluate_lookup(&table, &half.remaining_share());
            window = half.freeze_second(second).unwrap();
        }
        (window.finish().unwrap(), worker)
    });
    assert!(trace.live > 0);
    assert_eq!(trace.failures, 0);
    // Include inline expanded keys/cursors, not just heap payloads. This is a
    // native capacity check, not allocator-overhead/RSS/performance qualification.
    assert!(
        trace.peak as usize + size_of::<Window>() + size_of::<SeededWorker>()
            <= publication.resources().max_window_payload_bytes
    );
    assert_eq!(accepted.values()[0].to_bits(), (-0.0_f32).to_bits());
    drop(accepted);
    assert_eq!(publication.live_windows(), 1);
    drop(worker);
    assert_eq!(publication.live_windows(), 0);
}

const LIVE: usize = MAX_LIVE_WINDOWS as usize;

// Fixed-size test holders avoid introducing an unpriced heap queue. Production
// source buffers, publication/client handles and every window allocation are
// created while the allocator trace is enabled and retained through its end.
struct PeakOwners {
    _matrix: I16Matrix,
    _table: FloatTable,
    client: Client,
    publication: Arc<Publication>,
    accepted: [Option<AcceptedWindow>; LIVE],
    workers: [Option<SeededWorker>; LIVE],
}

fn assert_full_peak_envelope(label: &str, trace: Trace, owners: &PeakOwners, inline: usize) {
    let resources = owners.publication.resources();
    assert_eq!(trace.failures, 0);
    assert!(trace.live > 0 && trace.peak >= trace.live);
    assert_eq!(
        resources.max_live_window_payload_bytes,
        LIVE * resources.max_window_payload_bytes
    );
    assert!(resources.admitted_payload_bytes <= MAX_PUBLICATION_BYTES);
    assert!(resources.admitted_payload_bytes <= MAX_PROCESS_PUBLICATION_BYTES);
    let total = trace.peak as usize + inline;
    assert!(
        total <= resources.admitted_payload_bytes,
        "{label}: heap {} + inline {inline} = {total}, admitted {}",
        trace.peak,
        resources.admitted_payload_bytes
    );
    eprintln!(
        "{label}: heap_peak={} inline_bound={inline} total={total} admitted={}",
        trace.peak, resources.admitted_payload_bytes
    );
}

fn retire_peak_owners(mut owners: PeakOwners) {
    assert_eq!(owners.publication.live_windows(), MAX_LIVE_WINDOWS);
    assert!(matches!(
        owners.client.begin([9; 32], &[Step::Lookup]),
        Err(Error::Limit)
    ));
    for worker in &mut owners.workers {
        drop(worker.take());
    }
    // Maximum-width accepted payloads alone keep all eight window reservations.
    assert_eq!(owners.publication.live_windows(), MAX_LIVE_WINDOWS);
    assert!(matches!(
        owners.client.begin_seeded([9; 32], &[Step::Lookup]),
        Err(Error::Limit)
    ));
    for (index, accepted) in owners.accepted.iter_mut().enumerate() {
        drop(accepted.take());
        assert_eq!(owners.publication.live_windows(), (LIVE - index - 1) as u32);
    }
}

#[test]
fn maximum_column_linear_peak_includes_complete_publication_and_eight_live_owners() {
    let resources =
        PublicationResources::for_shapes(&[(MAX_DIMENSION, MAX_DIMENSION)], 1, 1).unwrap();
    // Caller-owned oracle/import inputs are outside the native envelope/trace.
    // The owned imported matrix and all downstream native payloads are inside.
    let weights: Vec<_> = (0..MAX_DIMENSION)
        .flat_map(|row| {
            (0..MAX_DIMENSION).map(move |column| (row as i16 % 7 - 3) * (column as i16 % 5 - 2))
        })
        .collect();
    let input: [i64; MAX_DIMENSION] = std::array::from_fn(|column| column as i64 % 11 - 5);
    let expected: Vec<_> = weights
        .chunks_exact(MAX_DIMENSION)
        .map(|row| {
            let sum: i128 = row
                .iter()
                .zip(&input)
                .map(|(&weight, &x)| i128::from(weight) * i128::from(x))
                .sum();
            i64::try_from(sum).unwrap()
        })
        .collect();
    assert!(expected.iter().any(|&value| value != 0));
    let (owners, trace) = traced(0, || {
        let matrix =
            I16Matrix::from_encoded(MAX_DIMENSION, MAX_DIMENSION, &weights, &[8; MAX_DIMENSION])
                .unwrap();
        let table = FloatTable::from_le_bytes(1, 1, &[0; 4]).unwrap();
        let publication = Arc::new(
            Publication::publish(
                [1; 32],
                &[Stage {
                    id: [2; 32],
                    matrix: &matrix,
                }],
                &table,
            )
            .unwrap(),
        );
        let client = Client::new(Arc::clone(&publication), publication.digest()).unwrap();
        let mut pending: [Option<PendingLinear>; LIVE] = std::array::from_fn(|_| None);
        let mut workers: [Option<SeededWorker>; LIVE] = std::array::from_fn(|_| None);
        let mut first: [Option<LinearResponse>; LIVE] = std::array::from_fn(|_| None);
        let mut second: [Option<LinearResponse>; LIVE] = std::array::from_fn(|_| None);
        let mut accepted: [Option<AcceptedWindow>; LIVE] = std::array::from_fn(|_| None);
        for index in 0..LIVE {
            let (window, worker) = client
                .begin_seeded([3; 32], &[Step::Linear([2; 32])])
                .unwrap();
            workers[index] = Some(worker);
            pending[index] = Some(window.linear(&input).unwrap());
        }
        for index in 0..LIVE {
            let p = pending[index].as_ref().unwrap();
            assert_eq!(p.share(Worker::Zero).values().unwrap().len(), MAX_DIMENSION);
            assert!(p.share(Worker::One).values().is_none());
            first[index] = Some(matrix.evaluate(&p.share(Worker::Zero)).unwrap());
            second[index] = Some(
                workers[index]
                    .as_mut()
                    .unwrap()
                    .evaluate_linear(&matrix, &p.share(Worker::One))
                    .unwrap(),
            );
        }
        // All maximum-width inputs and both response buffers coexist before
        // freezing. Completing one window never refunds its accepted payload.
        assert_eq!(publication.live_windows(), MAX_LIVE_WINDOWS);
        for index in 0..LIVE {
            let half = pending[index]
                .take()
                .unwrap()
                .freeze_first(Ok(first[index].take().unwrap()))
                .unwrap();
            let window = half
                .freeze_second(Ok(second[index].take().unwrap()))
                .unwrap();
            accepted[index] = Some(window.finish().unwrap());
        }
        PeakOwners {
            publication,
            client,
            _matrix: matrix,
            _table: table,
            accepted,
            workers,
        }
    });
    assert_eq!(
        owners.publication.resources().admitted_payload_bytes,
        resources.admitted_payload_bytes
    );
    assert_eq!(trace.largest, MAX_DIMENSION * MAX_DIMENSION * 2);
    // Conservatively include every inline holder plus transient freeze/return
    // owners and views, even though not all contain live payloads simultaneously.
    // Arc heap allocations (including Publication) are already in the trace;
    // its extra inline size below covers the pre-Arc publication return value.
    let inline = size_of::<PeakOwners>()
        + size_of::<Publication>()
        + size_of::<[Option<PendingLinear>; LIVE]>()
        + 2 * size_of::<[Option<LinearResponse>; LIVE]>()
        + size_of::<HalfLinear>()
        + size_of::<Window>()
        + 2 * size_of::<LinearShare<'_>>();
    assert_full_peak_envelope("linear 256x256 / eight windows", trace, &owners, inline);
    for accepted in &owners.accepted {
        let accepted = accepted.as_ref().unwrap();
        assert_eq!(accepted.signed_linear_output(), expected);
        assert_eq!(accepted.values().len(), MAX_DIMENSION);
        for (&actual, &integer) in accepted.values().iter().zip(&expected) {
            assert_eq!(
                actual.to_bits(),
                ((integer as f32) / 4_194_304.0_f32).to_bits()
            );
        }
    }
    retire_peak_owners(owners);
}

#[test]
fn maximum_selector_and_row_width_lookup_peak_includes_all_live_owners() {
    let resources =
        PublicationResources::for_shapes(&[(1, 1)], MAX_VOCABULARY, MAX_DIMENSION).unwrap();
    let row_bytes = MAX_DIMENSION * 4;
    let original: Vec<_> = (0..MAX_VOCABULARY)
        .flat_map(|row| {
            (0..MAX_DIMENSION).flat_map(move |column| {
                // Every row differs; the last row also carries an exact negative zero.
                let bits = if row == MAX_VOCABULARY - 1 && column == 0 {
                    0x8000_0000u32
                } else {
                    0x3f00_0000 | (row * MAX_DIMENSION + column) as u32
                };
                bits.to_le_bytes()
            })
        })
        .collect();
    let tokens = [0, 1, 7, 8, 511, 512, 2048, MAX_VOCABULARY - 1];
    assert_eq!(tokens.len(), LIVE);
    let (owners, trace) = traced(0, || {
        let matrix = I16Matrix::from_encoded(1, 1, &[3], &[8]).unwrap();
        let table = FloatTable::from_le_bytes(MAX_VOCABULARY, MAX_DIMENSION, &original).unwrap();
        let publication = Arc::new(
            Publication::publish(
                [1; 32],
                &[Stage {
                    id: [2; 32],
                    matrix: &matrix,
                }],
                &table,
            )
            .unwrap(),
        );
        let client = Client::new(Arc::clone(&publication), publication.digest()).unwrap();
        let mut pending: [Option<PendingLookup>; LIVE] = std::array::from_fn(|_| None);
        let mut workers: [Option<SeededWorker>; LIVE] = std::array::from_fn(|_| None);
        let mut first: [Option<LookupResponse>; LIVE] = std::array::from_fn(|_| None);
        let mut second: [Option<LookupResponse>; LIVE] = std::array::from_fn(|_| None);
        let mut accepted: [Option<AcceptedWindow>; LIVE] = std::array::from_fn(|_| None);
        for index in 0..LIVE {
            let (window, worker) = client.begin_seeded([3; 32], &[Step::Lookup]).unwrap();
            workers[index] = Some(worker);
            pending[index] = Some(window.lookup(tokens[index]).unwrap());
        }
        for index in 0..LIVE {
            let p = pending[index].as_ref().unwrap();
            assert_eq!(
                p.share(Worker::Zero).selector().unwrap().len(),
                MAX_VOCABULARY / 8
            );
            assert!(p.share(Worker::One).selector().is_none());
            first[index] = Some(table.evaluate(&p.share(Worker::Zero)).unwrap());
            second[index] = Some(
                workers[index]
                    .as_mut()
                    .unwrap()
                    .evaluate_lookup(&table, &p.share(Worker::One))
                    .unwrap(),
            );
        }
        assert_eq!(publication.live_windows(), MAX_LIVE_WINDOWS);
        for index in 0..LIVE {
            let half = pending[index]
                .take()
                .unwrap()
                .freeze_first(Ok(first[index].take().unwrap()))
                .unwrap();
            let window = half
                .freeze_second(Ok(second[index].take().unwrap()))
                .unwrap();
            accepted[index] = Some(window.finish().unwrap());
        }
        PeakOwners {
            publication,
            client,
            _matrix: matrix,
            _table: table,
            accepted,
            workers,
        }
    });
    assert_eq!(
        owners.publication.resources().admitted_payload_bytes,
        resources.admitted_payload_bytes
    );
    assert_eq!(trace.largest, MAX_VOCABULARY * row_bytes);
    let inline = size_of::<PeakOwners>()
        + size_of::<Publication>()
        + size_of::<[Option<PendingLookup>; LIVE]>()
        + 2 * size_of::<[Option<LookupResponse>; LIVE]>()
        + size_of::<HalfLookup>()
        + size_of::<Window>()
        + 2 * size_of::<LookupShare<'_>>();
    assert_full_peak_envelope("lookup 4096x256 / eight windows", trace, &owners, inline);
    for (accepted, token) in owners.accepted.iter().zip(tokens) {
        let accepted = accepted.as_ref().unwrap();
        assert!(accepted.signed_linear_output().is_empty());
        assert_eq!(accepted.values().len(), MAX_DIMENSION);
        for (&value, bytes) in accepted
            .values()
            .iter()
            .zip(original[token * row_bytes..(token + 1) * row_bytes].chunks_exact(4))
        {
            assert_eq!(value.to_le_bytes().as_slice(), bytes);
        }
    }
    retire_peak_owners(owners);
}
