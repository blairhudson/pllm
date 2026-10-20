use super::*;
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
