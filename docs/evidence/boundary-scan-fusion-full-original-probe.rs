//! Separately admitted full-head fixture; shares the original private test reference.
use super::*;
use std::process::Command;

const ROWS: usize = 151_936;
const COLS: usize = 896;
const RESERVE: u64 = 2 * 1024 * 1024 * 1024;
// Conservative aggregate live payload envelope, including source+native matrix,
// caller+two table scales, batch mask, all simultaneous result/oracle owners,
// input/entropy scratch, packets/decoded keys, ledgers and recursive scratch.
// Extra 32 MiB reserves metadata/executor/allocator uncertainty, not a RAM proof.
const ALLOCATION_BOUND: u64 =
    (2 * ROWS * COLS + 3 * 4 * ROWS + 16 * 4 * ROWS + 4 * ROWS + 32 * 4 * COLS + 32 * 1024 * 1024)
        as u64;

fn host_guard() -> (u64, u64) {
    assert_eq!(
        std::env::consts::OS,
        "macos",
        "native guard currently macOS only"
    );
    let run = |program: &str, args: &[&str]| {
        let out = Command::new(program)
            .args(args)
            .output()
            .expect("host guard command");
        assert!(out.status.success(), "host guard unavailable");
        String::from_utf8(out.stdout).unwrap()
    };
    let vm = run("/usr/bin/vm_stat", &[]);
    let page: u64 = vm
        .lines()
        .next()
        .unwrap()
        .split("page size of ")
        .nth(1)
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .parse()
        .unwrap();
    let pages = |name: &str| -> u64 {
        vm.lines()
            .find_map(|line| line.strip_prefix(name))
            .expect("vm_stat field")
            .trim()
            .trim_end_matches('.')
            .parse()
            .unwrap()
    };
    // Conservative immediately free + inactive pages; speculative not double-counted.
    let available = (pages("Pages free:") + pages("Pages inactive:")) * page;
    let swap = run("/usr/sbin/sysctl", &["-n", "vm.swapusage"]);
    let used = swap
        .split("used = ")
        .nth(1)
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap();
    assert!(used.ends_with('M'), "unexpected swap units");
    let swap_bytes = (used.trim_end_matches('M').parse::<f64>().unwrap() * 1048576.0) as u64;
    assert!(
        available >= RESERVE + ALLOCATION_BOUND,
        "native headroom reserve rejected"
    );
    (available, swap_bytes)
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
fn cpu_ns() -> u128 {
    #[repr(C)]
    struct Timespec {
        seconds: std::ffi::c_long,
        nanos: std::ffi::c_long,
    }
    unsafe extern "C" {
        fn clock_gettime(clock: std::ffi::c_int, time: *mut Timespec) -> std::ffi::c_int;
    }
    let mut time = Timespec {
        seconds: 0,
        nanos: 0,
    };
    let clock = if cfg!(target_os = "macos") { 12 } else { 2 };
    assert_eq!(unsafe { clock_gettime(clock, &mut time) }, 0);
    time.seconds as u128 * 1_000_000_000 + time.nanos as u128
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn cpu_ns() -> u128 {
    panic!("process CPU clock unsupported")
}

fn measured<T>(f: impl FnOnce() -> T) -> (T, serde_json::Value) {
    let cpu = cpu_ns();
    let wall = Instant::now();
    let value = f();
    let wall_ns = wall.elapsed().as_nanos();
    let process_cpu_ns = cpu_ns() - cpu;
    (
        value,
        json!({"wall_ns": wall_ns, "process_cpu_ns": process_cpu_ns}),
    )
}

fn run(rows: usize, columns: usize, repetitions: usize, guarded: bool) -> serde_json::Value {
    // Public geometry/work admission must precede every data/key allocation.
    assert!((rows == ROWS && columns == COLS) || (!guarded && rows == 129 && columns == 19));
    assert!((2..=16).contains(&repetitions) && repetitions % 2 == 0);
    let work_bound = rows
        .checked_mul(columns)
        .unwrap()
        .checked_mul(2 * repetitions * 12 + 8)
        .unwrap();
    assert!(work_bound <= ROWS * COLS * (2 * 16 * 12 + 8));
    assert!(ALLOCATION_BOUND < 384 * 1024 * 1024);
    let initial = if guarded { Some(host_guard()) } else { None };
    let setup = Instant::now();
    let weights: Vec<u8> = (0..rows * columns)
        .map(|i| (i * 131 + i / columns) as u8)
        .collect();
    let scales: Vec<u8> = (0..rows)
        .flat_map(|i| (0.001_f32 + i as f32 / 1024.0).to_le_bytes())
        .collect();
    let matrix = Arc::new(Matrix::new(&weights, rows, columns).unwrap());
    let mut fused: Vec<_> = (0..2)
        .map(|party| {
            let mut root = Server::from_matrix(Arc::clone(&matrix), &scales, party).unwrap();
            Fused::new(root.bind_context([173; 32]).unwrap())
        })
        .collect();
    let mut pairs: Vec<_> = fused.iter().map(|f| oracle(&f.server)).collect();
    let mut lookups: Vec<_> = fused.iter().map(|f| oracle(&f.server)).collect();
    let executor = Executor::new(1, true).unwrap();
    assert_eq!(Arc::strong_count(&matrix), 3);
    for p in 0..2 {
        assert!(Arc::ptr_eq(&pairs[p].data, &fused[p].server.data));
        assert!(Arc::ptr_eq(&lookups[p].data, &fused[p].server.data));
    }
    let q: Vec<_> = (0..columns)
        .map(|c| (c as i32 % 255 - 127) as u32)
        .collect();
    let expected_q = scalar(&matrix, &q);
    let setup_ns = setup.elapsed().as_nanos();
    let mut samples = Vec::new();
    let mut guards = Vec::new();
    for iteration in 0..repetitions {
        if guarded {
            let now = host_guard();
            assert_eq!(now.1, initial.unwrap().1, "native zero-swap guard");
            guards.push(json!({"available_bytes": now.0, "swap_used_bytes": now.1}));
        }
        let index = if iteration % 2 == 0 { 0 } else { rows - 1 };
        let (qa, qb, decoder) = issue_packed(fused[0].server.descriptor(), index).unwrap();
        let queries = [&qa, &qb];
        let mut replies = Vec::new();
        for party in 0..2 {
            let r = entropy_ring(columns);
            let id = ticket();
            let complement = Zeroizing::new(
                q.iter()
                    .zip(r.iter())
                    .map(|(&q, &r)| q.wrapping_sub(r))
                    .collect::<Vec<_>>(),
            );
            // Rotate every arm through every slot; reverse across the next repeat.
            // Only repeat/party controls order, never a mask or token value.
            let arms = [
                "lookup",
                "head",
                "original_pair",
                "fused_pair",
                "complementary_head",
            ];
            let shift = (iteration / 2 + party) % arms.len();
            let mut order: Vec<_> = (0..5).map(|i| arms[(i + shift) % 5]).collect();
            if iteration % 2 == 1 {
                order.reverse();
            }
            // Pair ordering itself is exactly balanced even with only two repeats.
            if (iteration + party) % 2 == 0 {
                let a = order.iter().position(|&s| s == "original_pair").unwrap();
                let b = order.iter().position(|&s| s == "fused_pair").unwrap();
                if a > b {
                    order.swap(a, b);
                }
            } else {
                let a = order.iter().position(|&s| s == "original_pair").unwrap();
                let b = order.iter().position(|&s| s == "fused_pair").unwrap();
                if b > a {
                    order.swap(a, b);
                }
            }
            let mut timings = serde_json::Map::new();
            let (mut lookup, mut head, mut original, mut candidate, mut other) =
                (None, None, None, None, None);
            for &arm in &order {
                let timing = match arm {
                    "lookup" => {
                        let (v, t) = measured(|| {
                            lookups[party]
                                .evaluate_batch(&[queries[party]])
                                .unwrap()
                                .remove(0)
                        });
                        lookup = Some(v);
                        t
                    }
                    "head" => {
                        let (v, t) = measured(|| matrix.wrap32(&executor, &r, 1).unwrap());
                        head = Some(v);
                        t
                    }
                    "original_pair" => {
                        let (v, t) = measured(|| {
                            (
                                pairs[party]
                                    .evaluate_batch(&[queries[party]])
                                    .unwrap()
                                    .remove(0),
                                matrix.wrap32(&executor, &r, 1).unwrap(),
                            )
                        });
                        original = Some(v);
                        t
                    }
                    "fused_pair" => {
                        let (v, t) = measured(|| {
                            fused[party].evaluate(queries[party], &r, id, None).unwrap()
                        });
                        candidate = Some(v);
                        t
                    }
                    "complementary_head" => {
                        let (v, t) = measured(|| matrix.wrap32(&executor, &complement, 1).unwrap());
                        other = Some(v);
                        t
                    }
                    _ => unreachable!(),
                };
                timings.insert(arm.into(), timing);
            }
            let candidate = candidate.unwrap();
            let original = original.unwrap();
            assert_eq!(candidate.0, original.0);
            assert_eq!(candidate.0, lookup.unwrap());
            assert_eq!(
                candidate.0,
                oracle(&fused[party].server)
                    .evaluate(queries[party])
                    .unwrap()
            );
            assert_eq!(candidate.1, original.1);
            assert_eq!(candidate.1, head.unwrap());
            assert_eq!(candidate.1, scalar(&matrix, &r));
            for ((&a, b), &expected) in candidate.1.iter().zip(other.unwrap()).zip(&expected_q) {
                assert_eq!(a.wrapping_add(b), expected);
            }
            assert_eq!(fused[party].last_work.weight_values, rows * columns);
            assert_eq!(fused[party].last_work.scale_bytes, 4 * rows);
            assert!(
                fused[party]
                    .evaluate(queries[party], &r, ticket(), None)
                    .is_err()
            );
            let sample = json!({"iteration": iteration, "party": party, "order": order,
                "timings": timings, "parity": true, "replay_rejected": true,
                "rows_visited": rows, "weight_values_visited": rows * columns, "scale_bytes_visited": rows * 4});
            println!("FUSION_FULL_PROGRESS {}", sample);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
            samples.push(sample);
            replies.push(candidate.0);
        }
        let mut expected = weights[index * columns..(index + 1) * columns].to_vec();
        expected.extend_from_slice(&scales[index * 4..index * 4 + 4]);
        assert_eq!(decoder.decode(&replies[0], &replies[1]).unwrap(), expected);
    }
    // Failure controls share the SAME matrix/table, not another fixture owner.
    // Full post-scan failure charges a full head; none are timed arm samples.
    for reference in &fused {
        failure_controls(&reference.server, columns, rows);
    }
    let final_guard = if guarded {
        let now = host_guard();
        assert_eq!(now.1, initial.unwrap().1, "native zero-swap guard");
        Some(json!({"available_bytes": now.0, "swap_used_bytes": now.1}))
    } else {
        None
    };
    json!({"schema": "boundary-scan-fusion-full-v1", "rows": rows, "columns": columns,
        "repetitions": repetitions, "samples": samples, "setup_wall_ns": setup_ns,
        "allocation_bound_bytes": ALLOCATION_BOUND, "work_bound_weight_values": work_bound,
        "native_initial_guard": initial.map(|g| json!({"available_bytes": g.0, "swap_used_bytes": g.1})),
        "native_guards": guards, "native_final_guard": final_guard,
        "reserve_bytes": RESERVE, "native_simd": executor.simd(),
        "matrix_owners": 1, "parties": 2, "correct": true, "fault_controls": true,
        "physical_memory_scans_measured": false, "sdk_selectable": false})
}

fn failure_controls(server: &Server, columns: usize, rows: usize) {
    let mut f = Fused::new(oracle(server));
    let query = |f: &Fused| {
        let (a, b, _) = issue_packed(f.server.descriptor(), rows - 1).unwrap();
        if f.server.party == 0 { a } else { b }
    };
    let first = query(&f);
    let r = entropy_ring(columns);
    let id = ticket();
    assert!(f.evaluate(&first, &r[..columns - 1], id, None).is_err());
    let mut wrong = first.clone();
    wrong[8] = 1 - f.server.party;
    assert!(f.evaluate(&wrong, &r, id, None).is_err());
    let mut wrong = first.clone();
    wrong[17] ^= 1;
    assert!(f.evaluate(&wrong, &r, id, None).is_err());
    assert!(f.evaluate(&first[..10], &r, id, None).is_err());
    assert!(f.server.burned.is_empty() && f.heads.is_empty());
    assert!(f.evaluate(&first, &r, id, Some(7)).is_err());
    assert_eq!(f.last_work.weight_values, 7 * columns);
    assert!(f.server.burned.len() == 1 && f.heads.contains(&id));
    assert!(f.evaluate(&first, &r, ticket(), None).is_err());
    let next = query(&f);
    let next_r = entropy_ring(columns);
    assert!(f.evaluate(&next, &next_r, id, None).is_err());
    assert_eq!(f.server.burned.len(), 1);
    let cancel_id = ticket();
    f.cancel(&next, &next_r, cancel_id).unwrap();
    assert!(f.evaluate(&next, &next_r, ticket(), None).is_err());
    let last = query(&f);
    let last_r = entropy_ring(columns);
    assert!(f.evaluate(&last, &last_r, cancel_id, None).is_err());
    let final_id = ticket();
    assert!(f.evaluate(&last, &last_r, final_id, Some(rows)).is_err());
    assert_eq!(f.last_work.rows, rows);
    assert!(f.heads.contains(&final_id));
    assert!(f.evaluate(&last, &last_r, ticket(), None).is_err());
    let fresh = query(&f);
    for i in 0..MAX_QUERIES {
        let mut id = [0; 16];
        id[..8].copy_from_slice(&(i as u64).to_le_bytes());
        f.heads.insert(id);
    }
    assert!(f.evaluate(&fresh, &last_r, ticket(), None).is_err());
    assert_eq!(f.last_work.rows, 0);
    let (a, b, _) = issue(f.server.descriptor(), rows - 1).unwrap();
    assert!(
        f.evaluate(
            if f.server.party == 0 { &a } else { &b },
            &last_r,
            ticket(),
            None
        )
        .is_err()
    );
}

#[test]
fn full_reference_small_correctness() {
    let result = run(129, 19, 2, false);
    assert_eq!(result["correct"], true);
}

#[test]
#[ignore = "parent-only serial full geometry; use guarded pinned launcher"]
fn full_fixture() {
    assert_eq!(
        std::env::var("PLLM_FUSION_FULL_ADMITTED").as_deref(),
        Ok("1")
    );
    let repetitions: usize = std::env::var("PLLM_FUSION_FULL_REPETITIONS")
        .unwrap()
        .parse()
        .unwrap();
    println!("{}", run(ROWS, COLS, repetitions, true));
}
