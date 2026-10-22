use super::*;

#[test]
fn allocation_failure_after_first_window_import_consumes_runtime_without_early_window_refund() {
    use crate::fusefss_backend::composition::provider::ReplyWindow;
    use std::sync::atomic::Ordering;
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let plan = kernels::floor(
        kernels::Binding {
            source_digest: [81; 32],
            contract_digest: [82; 32],
        },
        gates::InputRange {
            min: -100_000,
            max: 100_000,
        },
        12,
        &budget,
    )
    .unwrap()
    .0
    .repeat_tensor(2)
    .unwrap();
    let schedule = Schedule::new(&plan).unwrap();
    let mut runtime = schedule.runtime(0, context(), &[0; 2]).unwrap();
    assert!(matches!(
        runtime.poll().unwrap(),
        Event::NeedMaterial { .. }
    ));
    let required = ReplyWindow::<usize>::reservation_bytes().unwrap();
    let window_budget = Budget::new(required).unwrap();
    let mut window = ReplyWindow::new(&window_budget, 0usize).unwrap();
    assert!(runtime.pending_materials(window.pcs_mut()).unwrap() >= 2);
    let mut packets = Vec::new();
    for i in 0..2 {
        let pc = window.pc(i).unwrap();
        window
            .push(
                pc,
                plan.step_transport_resources(pc)
                    .unwrap()
                    .packet_bytes_bound,
            )
            .unwrap();
        let [a, b] = plan.issue_step(pc, context()).unwrap();
        drop(b);
        packets.push(a.export_packet().unwrap());
    }
    window
        .provide(|_, replies| {
            for (reply, packet) in replies.iter_mut().zip(&packets) {
                reply.write(packet.as_bytes())?;
            }
            Ok(())
        })
        .unwrap();
    drop(packets);
    let mut guard = ImportGuard::new_indexed(plan.resources.secret_steps.max(1), &budget).unwrap();
    let first = window.bytes(0).unwrap();
    let material = plan.import_step_bytes(
        window.pc(0).unwrap(),
        context(),
        0,
        first[48..80].try_into().unwrap(),
        first,
        &mut guard,
    );
    assert!(matches!(
        runtime.feed_result(material).unwrap(),
        Event::NeedMaterial { .. }
    ));
    budget.fail_after.store(0, Ordering::SeqCst);
    let second = window.bytes(1).unwrap();
    let material = plan.import_step_bytes(
        window.pc(1).unwrap(),
        context(),
        0,
        second[48..80].try_into().unwrap(),
        second,
        &mut guard,
    );
    assert!(runtime.feed_result(material).is_err());
    assert!(runtime.is_consumed());
    assert_eq!(window_budget.used_bytes(), required);
    drop(window);
    assert_eq!(window_budget.used_bytes(), 0);
    budget.fail_after.store(-1, Ordering::SeqCst);
    drop(runtime);
    drop(guard);
    drop(schedule);
    drop(plan);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn real_frozen_group_has_bounded_full_tail_windows_without_crossing_frame_barrier() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let plan = kernels::floor(
        kernels::Binding {
            source_digest: [71; 32],
            contract_digest: [72; 32],
        },
        gates::InputRange {
            min: -100_000,
            max: 100_000,
        },
        12,
        &budget,
    )
    .unwrap()
    .0
    .repeat_tensor(17)
    .unwrap();
    let schedule = Schedule::new(&plan).unwrap();
    let mut runtime = schedule.runtime(0, context(), &[0; 17]).unwrap();
    let mut guard = ImportGuard::new_indexed(plan.resources.secret_steps.max(1), &budget).unwrap();
    let mut event = runtime.poll().unwrap();
    let state = runtime.state.as_ref().unwrap();
    let expected: Vec<_> = (state.offset..state.end)
        .map(|i| schedule.step(i).unwrap().pc)
        .collect();
    assert!(expected.len() >= 17);
    let mut consumed = Vec::new();
    let mut counts = Vec::new();
    while matches!(event, Event::NeedMaterial { .. }) {
        let mut pcs = [usize::MAX; 8];
        let count = runtime.pending_materials(&mut pcs).unwrap();
        counts.push(count);
        for &pc in &pcs[..count] {
            assert!(matches!(event, Event::NeedMaterial { pc: actual } if actual == pc));
            let [material, other] = plan.issue_step(pc, context()).unwrap();
            drop(other);
            let packet = material.export_packet().unwrap();
            let material = plan.import_step_bytes(
                pc,
                context(),
                0,
                packet.as_bytes()[48..80].try_into().unwrap(),
                packet.as_bytes(),
                &mut guard,
            );
            event = runtime.feed_result(material).unwrap();
            consumed.push(pc);
        }
    }
    assert_eq!(consumed, expected);
    assert_eq!(counts[0], 8);
    assert_eq!(*counts.last().unwrap(), (expected.len() - 1) % 8 + 1);
    assert!(matches!(event, Event::Frame(_)));
    assert!(runtime.pending_materials(&mut [0; 8]).is_err());
    drop(event);
    runtime.cancel();
    drop(runtime);
    drop(guard);
    drop(schedule);
    drop(plan);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn compact_tensor_keeps_one_template_with_exact_large_lane_outputs() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let binding = kernels::Binding {
        source_digest: [21; 32],
        contract_digest: [22; 32],
    };
    let base = || {
        kernels::floor(
            binding,
            gates::InputRange {
                min: -100_000,
                max: 100_000,
            },
            12,
            &budget,
        )
        .unwrap()
        .0
    };
    let expanded = base().repeat(256).unwrap();
    let template = base().repeat_tensor(1024).unwrap();
    assert!(template.is_tensor());
    assert_eq!(template.code.len(), base().code.len());
    assert_eq!(template.input_bits().len(), 1024);
    assert_eq!(template.output_count(), 1024);
    assert!(
        template.resources().plan_reservation_bytes
            < expanded.resources().plan_reservation_bytes / 8
    );
    assert_eq!(
        template.resources().secret_steps,
        4 * expanded.resources().secret_steps
    );
    let values: Vec<i64> = (0..1024).map(|i| (i * 193 % 200_001) - 100_000).collect();
    let input: Vec<_> = values.iter().map(|&v| v as u64 & mask(50)).collect();
    let expected: Vec<_> = values
        .iter()
        .map(|&v| (v >> 12) as u64 & mask(50))
        .collect();
    assert_eq!(
        evaluate(&Schedule::new(&template).unwrap(), &input)
            .unwrap()
            .0,
        expected
    );
    let wave = Schedule::new(&template).unwrap();
    let single = base().repeat_tensor(1).unwrap();
    let single_wave = Schedule::new(&single).unwrap();
    assert_eq!(
        wave.resources().metadata_bytes,
        single_wave.resources().metadata_bytes
    );
    assert!(wave.resources().runtime_owner_bytes > single_wave.resources().runtime_owner_bytes);
    assert_eq!(wave.steps.len(), single_wave.steps.len());
    assert_eq!(wave.spans.len(), single_wave.spans.len());
    assert!(wave.step(template.resources().instructions).is_none());
    drop(single_wave);
    drop(single);
    assert!(wave.resources().rounds <= 4 * Schedule::new(&expanded).unwrap().resources().rounds);
    assert!(wave.resources().retained_party_bytes <= MAX_RETAINED_PARTY_BYTES);
    assert!(wave.resources().largest_frame_bytes <= MAX_FRAME_BYTES);
    assert!(template.instruction(1024).is_err());
    assert!(template.output_bits(1024).is_err());
    drop(wave);
    drop(template);
    drop(expanded);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn compact_tensor_preserves_mixed_domains_multi_outputs_and_ties() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let binding = kernels::Binding {
        source_digest: [23; 32],
        contract_digest: [24; 32],
    };
    let base = || {
        kernels::stable_max(
            binding,
            [gates::InputRange {
                min: -100,
                max: 100,
            }; 2],
            gates::InputRange { min: 0, max: 1023 },
            &budget,
        )
        .unwrap()
        .0
    };
    let expanded = base().repeat(17).unwrap();
    let tensor = base().repeat_tensor(17).unwrap();
    let expanded_wave = Schedule::new(&expanded).unwrap();
    let tensor_wave = Schedule::new(&tensor).unwrap();
    let projected: Vec<_> = (0..tensor.resources().instructions)
        .map(|i| tensor_wave.step(i).unwrap())
        .collect();
    assert_eq!(projected, expanded_wave.steps);
    assert_eq!(
        tensor_wave.resources().rounds,
        expanded_wave.resources().rounds
    );
    assert_eq!(
        tensor_wave.resources().frame_bytes_per_party,
        expanded_wave.resources().frame_bytes_per_party
    );
    assert_eq!(
        tensor_wave.material_steps().collect::<Vec<_>>(),
        expanded_wave.material_steps().collect::<Vec<_>>()
    );
    assert!(tensor_wave.resources().metadata_bytes < expanded_wave.resources().metadata_bytes);
    drop(expanded_wave);
    drop(tensor_wave);
    let mut input = Vec::new();
    let mut expected = Vec::new();
    for i in 0..17i64 {
        let a = i - 8;
        let b = if i % 3 == 0 { a } else { 8 - i };
        input.extend([
            a as u64 & mask(50),
            (2 * i) as u64,
            b as u64 & mask(50),
            (2 * i + 1) as u64,
        ]);
        let (value, index) = if a >= b { (a, 2 * i) } else { (b, 2 * i + 1) };
        expected.extend([value as u64 & mask(50), index as u64]);
    }
    assert_eq!(
        evaluate(&Schedule::new(&expanded).unwrap(), &input)
            .unwrap()
            .0,
        expected
    );
    assert_eq!(
        evaluate(&Schedule::new(&tensor).unwrap(), &input)
            .unwrap()
            .0,
        expected
    );
    assert_ne!(expanded.digest(), tensor.digest());
    assert_eq!(
        expanded.resources().private_material_bytes_total,
        tensor.resources().private_material_bytes_total
    );
    for pc in 0..expanded.resources().instructions {
        assert_eq!(
            format!("{:?}", expanded.instruction(pc).unwrap()),
            format!("{:?}", tensor.instruction(pc).unwrap())
        );
    }
    for i in 0..expected.len() {
        assert_eq!(
            expanded.output_bits(i).unwrap(),
            tensor.output_bits(i).unwrap()
        );
        assert_eq!(
            expanded.output_is_xor(i).unwrap(),
            tensor.output_is_xor(i).unwrap()
        );
    }
    drop(expanded);
    drop(tensor);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn compact_template_failure_sweep_retires_old_and_new_allocations() {
    use std::sync::atomic::Ordering;
    for failure in 0..=5 {
        let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
        let binding = kernels::Binding {
            source_digest: [25; 32],
            contract_digest: [26; 32],
        };
        let plan = kernels::floor(
            binding,
            gates::InputRange {
                min: -100,
                max: 100,
            },
            3,
            &budget,
        )
        .unwrap()
        .0;
        budget.fail_after.store(failure, Ordering::SeqCst);
        let plan = plan.repeat_tensor(1024);
        if failure < 5 {
            assert!(matches!(plan, Err(Error::Allocation)));
        } else {
            assert!(plan.is_ok());
        }
        drop(plan);
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn compact_schedule_failure_sweep_retires_depth_steps_and_spans() {
    use std::sync::atomic::Ordering;
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let binding = kernels::Binding {
        source_digest: [27; 32],
        contract_digest: [28; 32],
    };
    let plan = kernels::stable_max(
        binding,
        [gates::InputRange {
            min: -100,
            max: 100,
        }; 2],
        gates::InputRange { min: 0, max: 1023 },
        &budget,
    )
    .unwrap()
    .0
    .repeat_tensor(17)
    .unwrap();
    let baseline = budget.used_bytes();
    let mut passed = false;
    for count in 0..128 {
        budget.fail_after.store(count, Ordering::SeqCst);
        let result = Schedule::new(&plan);
        budget.fail_after.store(-1, Ordering::SeqCst);
        passed = result.is_ok();
        if !passed {
            assert!(matches!(result, Err(Error::Allocation)));
        }
        drop(result);
        assert_eq!(budget.used_bytes(), baseline, "allocation {count}");
        if passed {
            break;
        }
    }
    assert!(passed);
    drop(plan);
    assert_eq!(budget.used_bytes(), 0);
}
use crate::fusefss_backend::wire::ImportGuard;

const MASK: u64 = (1 << 50) - 1;
fn context() -> Context {
    Context {
        plan: [3; 32],
        session: [4; 32],
        operation: [5; 32],
        first_tensor_index: 0,
    }
}
fn product_plan(budget: &Budget, lanes: usize) -> Plan<'_> {
    let mut b = Builder::new(
        [1; 32],
        b"wavefront-test-product-floor",
        &vec![50; 2 * lanes],
        budget,
    )
    .unwrap();
    let mut out = Vec::new();
    for i in 0..lanes {
        let x = b.input(i).unwrap();
        let y = b.input(lanes + i).unwrap();
        let p = b.multiply(x, y).unwrap();
        out.push(
            b.convert(
                p,
                conversions::Operation::Ars {
                    shift: 12,
                    range: conversions::SignedRange::full_ring(50).unwrap(),
                },
                50,
            )
            .unwrap(),
        );
    }
    b.finish(&out).unwrap()
}

fn evaluate(schedule: &Schedule<'_, '_>, inputs: &[u64]) -> Result<(Vec<u64>, usize), Error> {
    let plan = schedule.plan;
    let left: Vec<_> = inputs
        .iter()
        .enumerate()
        .map(|(i, _)| (0x3857_fafb_9185 + (i as u64) * 37) & MASK)
        .collect();
    let right: Vec<_> = inputs
        .iter()
        .zip(&left)
        .map(|(x, l)| x.wrapping_sub(*l) & MASK)
        .collect();
    let mut a = schedule.runtime(0, context(), &left)?;
    let mut b = schedule.runtime(1, context(), &right)?;
    let mut ga = ImportGuard::new_indexed(plan.resources.secret_steps.max(1), plan.lease.budget)?;
    let mut gb = ImportGuard::new_indexed(plan.resources.secret_steps.max(1), plan.lease.budget)?;
    let mut events = (a.poll()?, b.poll()?);
    let mut rounds = 0;
    loop {
        events = match events {
            (Event::NeedMaterial { pc: x }, Event::NeedMaterial { pc: y }) => {
                assert_eq!(x, y);
                // Every existing scalar oracle also checks bounded observation
                // against the actual frozen frontier, never a future group.
                for runtime in [&a, &b] {
                    let mut pcs = [usize::MAX; 8];
                    let count = runtime.pending_materials(&mut pcs)?;
                    let state = runtime.state.as_ref().unwrap();
                    let first = state.offset + state.slots.len();
                    assert_eq!(count, 8.min(state.end - first));
                    for (index, pc) in pcs[..count].iter().enumerate() {
                        assert_eq!(*pc, schedule.step(first + index).unwrap().pc);
                    }
                    assert_eq!(pcs[0], x);
                }
                let [m0, m1] = plan.issue_step(x, context())?;
                let p0 = m0.export_packet()?;
                let p1 = m1.export_packet()?;
                let m0 = plan.import_step_bytes(
                    x,
                    context(),
                    0,
                    p0.as_bytes()[48..80].try_into().unwrap(),
                    p0.as_bytes(),
                    &mut ga,
                );
                let m1 = plan.import_step_bytes(
                    x,
                    context(),
                    1,
                    p1.as_bytes()[48..80].try_into().unwrap(),
                    p1.as_bytes(),
                    &mut gb,
                );
                (a.feed_result(m0)?, b.feed_result(m1)?)
            }
            (Event::Frame(x), Event::Frame(y)) => {
                assert!(a.pending_materials(&mut [0; 8]).is_err());
                assert!(b.pending_materials(&mut [0; 8]).is_err());
                rounds += 1;
                assert!(x.as_bytes().len() <= schedule.resources.largest_frame_bytes);
                (a.advance(y.as_bytes())?, b.advance(x.as_bytes())?)
            }
            (Event::Output(x), Event::Output(y)) => {
                assert!(a.is_consumed() && b.is_consumed());
                return Ok((
                    x.shares()
                        .iter()
                        .zip(y.shares())
                        .enumerate()
                        .map(|(i, (x, y))| {
                            if plan.output_is_xor(i).unwrap() {
                                x ^ y
                            } else {
                                x.wrapping_add(*y) & mask(plan.output_bits(i).unwrap())
                            }
                        })
                        .collect(),
                    rounds,
                ));
            }
            _ => panic!("roles diverged"),
        };
    }
}

#[test]
fn dependency_waves_preserve_complete_products_floors_and_bounded_tails() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    for lanes in [1, 2, 63, 64, 65, 129, 256, 257] {
        let plan = product_plan(&budget, lanes);
        let baseline = budget.used_bytes();
        let schedule = Schedule::new(&plan).unwrap();
        let values: Vec<i64> = (0..2 * lanes)
            .map(|i| {
                if i % 3 == 0 {
                    -(i as i64) * 31 - 1
                } else {
                    i as i64 * 19 + 7
                }
            })
            .collect();
        let inputs: Vec<_> = values.iter().map(|&x| x as u64 & MASK).collect();
        let oracle: Vec<_> = (0..lanes)
            .map(|i| (values[i] * values[lanes + i]).div_euclid(4096) as u64 & MASK)
            .collect();
        let (output, rounds) = evaluate(&schedule, &inputs).unwrap();
        assert_eq!(output, oracle);
        assert_eq!(rounds, 2 * lanes.div_ceil(MAX_ACTIVE));
        assert_eq!(schedule.resources.primitive_frames, 2 * lanes);
        assert_eq!(schedule.material_steps().count(), 2 * lanes);
        drop(schedule);
        assert_eq!(budget.used_bytes(), baseline);
    }
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn mixed_primitive_rounds_and_local_dependencies_match_integer_oracle() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut b = Builder::new([1; 32], b"wave-mixed", &[50; 8], &budget).unwrap();
    let zero = b.constant(50, 0).unwrap();
    let mut outputs = Vec::new();
    for i in 0..8 {
        let x = b.input(i).unwrap();
        let negative = b.lt_signed(x, 0).unwrap();
        let inverse = b.sub(zero, x).unwrap();
        outputs.push(b.select(negative, inverse, x).unwrap());
    }
    let plan = b.finish(&outputs).unwrap();
    let schedule = Schedule::new(&plan).unwrap();
    let input = [-99i64, -1, 0, 1, 55, -735, 4096, -4096];
    let (result, rounds) = evaluate(&schedule, &input.map(|x| x as u64 & MASK)).unwrap();
    assert_eq!(result, input.map(i64::unsigned_abs));
    assert!(rounds < plan.resources.frames_per_party);
}

#[test]
fn repeated_templates_keep_lane_domains_tables_and_fresh_material() {
    use super::super::{cache, gates, kernels};
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let binding = kernels::Binding {
        source_digest: [7; 32],
        contract_digest: [8; 32],
    };
    let range = gates::InputRange {
        min: -8192,
        max: 8192,
    };
    let cache = cache::SourceTableCache::new(
        [7; 32],
        cache::Limits {
            max_entries: cache::MAX_ENTRIES,
            max_retained_bytes: cache::MAX_RETAINED_BYTES,
        },
        &budget,
    )
    .unwrap();
    for kind in 0..3 {
        let make = || match kind {
            0 => kernels::floor(binding, range, 12, &budget).map(|v| v.0),
            1 => kernels::stable_max(
                binding,
                [range; 2],
                gates::InputRange { min: 0, max: 1023 },
                &budget,
            )
            .map(|v| v.0),
            _ => gates::silu_cached(range, &cache).map(|v| v.0),
        };
        let one = make().unwrap();
        let primitive_count = one.resources().secret_steps;
        let bound = one.repetition_bound();
        assert!(bound > 1);
        drop(one);
        for lanes in [1, 3, bound] {
            let mut inputs = Vec::new();
            let mut expected = Vec::new();
            for i in 0..lanes {
                let x: i64 = match i % 5 {
                    0 => -8192,
                    1 => -1,
                    2 => 0,
                    3 => 1,
                    _ => 8192,
                };
                let lane = if kind == 1 {
                    vec![x as u64 & MASK, i as u64, x as u64 & MASK, (i + 100) as u64]
                } else {
                    vec![x as u64 & MASK]
                };
                let p = make().unwrap();
                expected.extend(evaluate(&Schedule::new(&p).unwrap(), &lane).unwrap().0);
                inputs.extend(lane);
            }
            let plan = make().unwrap().repeat(lanes).unwrap();
            assert_eq!(plan.resources().secret_steps, primitive_count * lanes);
            let schedule = Schedule::new(&plan).unwrap();
            assert_eq!(evaluate(&schedule, &inputs).unwrap().0, expected);
            let tensor = make().unwrap().repeat_tensor(lanes).unwrap();
            let schedule = Schedule::new(&tensor).unwrap();
            assert_eq!(evaluate(&schedule, &inputs).unwrap().0, expected);
        }
        assert!(matches!(
            make().unwrap().repeat(bound + 1),
            Err(Error::Resource)
        ));
    }
    drop(cache);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn malformed_batches_and_wrong_material_burn_complete_frontier() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let plan = product_plan(&budget, 2);
    let schedule = Schedule::new(&plan).unwrap();
    let baseline = budget.used_bytes();
    for fault in 0..7 {
        let mut a = schedule.runtime(0, context(), &[1, 2, 3, 4]).unwrap();
        let mut b = schedule.runtime(1, context(), &[0; 4]).unwrap();
        let mut ea = a.poll().unwrap();
        let mut eb = b.poll().unwrap();
        while let (Event::NeedMaterial { pc: x }, Event::NeedMaterial { pc: y }) = (&ea, &eb) {
            assert_eq!(x, y);
            let [ma, mb] = plan.issue_step(*x, context()).unwrap();
            ea = a.feed(ma).unwrap();
            eb = b.feed(mb).unwrap();
        }
        let (Event::Frame(_fa), Event::Frame(fb)) = (ea, eb) else {
            panic!()
        };
        let mut frame = fb.as_bytes().to_vec();
        match fault {
            0 => {
                frame[0] ^= 1;
            }
            1 => {
                frame[4] = 0;
            }
            2 => {
                frame[8] = 1;
            }
            3 => {
                frame[16] = 1;
            }
            4 => {
                frame[20] ^= 1;
            }
            5 => {
                frame.pop();
            }
            _ => {
                frame.push(0);
            }
        }
        assert!(a.advance(&frame).is_err());
        assert!(a.is_consumed());
        assert!(matches!(a.poll(), Err(Error::Consumed)));
    }
    assert_eq!(budget.used_bytes(), baseline);
    let mut a = schedule.runtime(0, context(), &[1, 2, 3, 4]).unwrap();
    let Event::NeedMaterial { pc } = a.poll().unwrap() else {
        panic!()
    };
    let mut wrong = context();
    wrong.session = [8; 32];
    let [material, _] = plan.issue_step(pc, wrong).unwrap();
    assert!(a.feed(material).is_err());
    assert!(a.is_consumed());
}

#[test]
fn allocation_failure_refunds_only_after_erasing_frontiers() {
    use std::sync::atomic::Ordering;
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let plan = product_plan(&budget, 2);
    let schedule = Schedule::new(&plan).unwrap();
    let baseline = budget.used_bytes();
    let mut passed = false;
    for count in 0..512 {
        budget.fail_after.store(count, Ordering::SeqCst);
        let outcome = evaluate(&schedule, &[4096, 8192, 12288, 16384]);
        budget.fail_after.store(-1, Ordering::SeqCst);
        assert_eq!(budget.used_bytes(), baseline, "allocation {count}");
        if outcome.is_ok() {
            passed = true;
            break;
        }
    }
    assert!(passed);
}
