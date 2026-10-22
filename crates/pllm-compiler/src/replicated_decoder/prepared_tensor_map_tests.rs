//! Closed factory parity through committed unrolled BODY Tensor execution.
//! No mapped Op, original Worker bridge, source identity or row-proof authority.
use super::*;
use crate::replicated_decoder::prepared_tensor_map::{
    native_attempts, observe_native_owners, take_native_owners, OneLoopScope, PreparedTensorMap,
};
use puma::{prss::Cursors, prss::PrssLimits, Party};

use std::mem::size_of_val;

const PARTIES: [Party; 3] = [Party::P0, Party::P1, Party::P2];

fn builder(lanes: usize) -> (Builder, Layout) {
    let mut b = Builder::new(
        WeightMode::Public,
        &[Register {
            lanes,
            bounds: Interval::new(-ONE, ONE).unwrap(),
        }],
    )
    .unwrap();
    let input = b.input(0).unwrap();
    (b, input)
}

// Existing unrolled Op::Tensor, with the SAME original prepared Arc owners.
// Oracle-only construction: no per-iteration native constructor or bound inference.
fn unrolled(input: &Layout, prepared: &PreparedTensorMap) -> Arc<Plan> {
    let (mut b, _) = builder(input.len());
    for p in prepared.programs() {
        let r = p.resources().unwrap();
        b.template_insert_preflight(&p.digest(), r, prepared.baseline_bytes().unwrap())
            .unwrap();
        b.templates.insert(p.digest(), p.clone());
        b.template_metadata
            .insert(p.digest(), r.immutable_template_bytes);
    }
    let stride = prepared.program_at(0).unwrap().lanes();
    let mut programs = b
        .new_template_list(
            input,
            prepared.invocations() as usize,
            prepared.baseline_bytes().unwrap(),
        )
        .unwrap();
    for offset in (0..input.len()).step_by(stride) {
        programs.push(prepared.program_at(offset).unwrap().clone());
    }
    let mut regions = b
        .region_buffer(1, prepared.baseline_bytes().unwrap())
        .unwrap();
    regions.push(Region {
        offset: 0,
        len: input.len(),
        bounds: prepared.output_bounds().unwrap(),
    });
    let output = b
        .push(
            Op::Tensor {
                input: input.clone(),
                stride,
                programs: programs.into_boxed_slice(),
            },
            input.len(),
            regions,
            prepared.work_budget().unwrap(),
            prepared.advances().unwrap(),
        )
        .unwrap();
    b.finish(&[output]).unwrap()
}

fn secret(party: Party, lanes: usize) -> Secret {
    let mut s = Secret::new(lanes).unwrap();
    for i in 0..lanes {
        let clear = ((i as i64 * 7919) % (2 * ONE) - ONE) as u64;
        // Nontrivial, consistent BOTH components, including wrapping ring shares.
        let a = 0xf123456789abcdef_u64.wrapping_add(i as u64 * 17);
        let b = 0x8123456789abcdef_u64.wrapping_sub(i as u64 * 23);
        let parts = [a, b, clear.wrapping_sub(a).wrapping_sub(b)];
        s.first[i] = parts[party as usize];
        s.second[i] = parts[party.next() as usize];
    }
    s
}

fn owners(plan: &Arc<Plan>, limits: PrssLimits) -> [Executor; 3] {
    // Explicit SHARED local test context, independent of the two representation
    // digests. Fresh owners each run; no canonical/full-decoder identity claim.
    let context = puma::Context::new([71; 32], [92; 32], 0).unwrap();
    puma::prss::insecure_test_support::fresh_compound_parties(context, limits)
        .unwrap()
        .map(|runtime| {
            let value = secret(runtime.party(), plan.input_specs()[0].lanes);
            Executor::new(plan.clone(), runtime, vec![value], None, None).unwrap()
        })
}

type Wire = Vec<(Party, Party, Vec<u8>)>;
type ExecutionResult = (Wire, [Vec<(u64, u64)>; 3], [Cursors; 3]);
fn drive(owners: &mut [Executor; 3]) -> Result<(Wire, u64), Error> {
    let mut wire = Vec::new();
    let mut advances = 0;
    while !owners.iter().all(Executor::complete) {
        assert!(!owners.iter().any(Executor::complete));
        advances += 1;
        assert!(advances <= MAX_COORDINATOR_ADVANCES);
        let mut queue = Vec::new();
        for (i, owner) in owners.iter_mut().enumerate() {
            queue.extend(
                owner
                    .advance()?
                    .into_iter()
                    .map(|(to, frame)| (PARTIES[i], to, frame)),
            );
        }
        let mut waves = 0;
        while !queue.is_empty() {
            waves += 1;
            assert!(waves <= 1024); // Existing global frozen-wave drain ceiling.
            let mut next = Vec::new();
            for (from, to, frame) in queue {
                // No sorting, truncation or header-only comparison.
                wire.push((from, to, frame.as_bytes().to_vec()));
                next.extend(
                    owners[to as usize]
                        .receive(frame.as_bytes(), from)?
                        .into_iter()
                        .map(|(recipient, frame)| (to, recipient, frame)),
                );
            }
            queue = next;
        }
    }
    Ok((wire, advances))
}

fn execute(plan: &Arc<Plan>, limits: PrssLimits) -> ExecutionResult {
    let mut owners = owners(plan, limits);
    let (wire, advances) = drive(&mut owners).unwrap();
    assert_eq!(advances, plan.coordinator_advances());
    let finished = owners.map(|o| o.finish().unwrap());
    let components = finished.each_ref().map(|(_, values)| {
        (0..values.output_len(0).unwrap())
            .map(|i| values.word(0, i).unwrap())
            .collect()
    });
    let cursors = finished.each_ref().map(|(runtime, _)| runtime.cursors());
    (wire, components, cursors)
}

#[test]
fn closed_factory_original_arcs_match_ordinary_body_wire_both_shares_and_cursors() {
    for (lanes, silu, permuted) in [
        (1, false, false),
        (128, false, false),
        (129, false, false),
        (257, false, false),
        (129, true, false),
        (129, false, true),
    ] {
        let (mut legacy, input) = builder(lanes);
        let input = if permuted {
            Layout::affine(0, 0, &[3, 43], &[1, 3]).unwrap()
        } else {
            input
        };
        let bounds = legacy.bounds(&input).unwrap();
        // Match the COMMITTED ordinary BODY factories, including Q18 rescale.
        // This local declaration is not a checkpoint/source certificate.
        let prepared = if silu {
            PreparedTensorMap::guarded_silu(lanes, bounds, 0, |_| Ok(())).unwrap()
        } else {
            PreparedTensorMap::rescale(lanes, bounds, 18, 0, |_| Ok(())).unwrap()
        };
        let original: Vec<_> = prepared.programs().map(Arc::downgrade).collect();
        let work = prepared.work_budget().unwrap();
        let candidate = unrolled(&input, &prepared);
        let output = if silu {
            legacy.silu(&input).unwrap()
        } else {
            legacy.rescale(&input).unwrap()
        };
        assert_eq!(
            legacy.bounds(&output).unwrap(),
            prepared.output_bounds().unwrap()
        );
        assert!(legacy.proofs.is_empty());
        let reference = legacy.finish(&[output]).unwrap();
        assert_eq!(candidate.work_budget(), reference.work_budget());
        assert_eq!(candidate.work_budget(), work);
        assert_eq!(
            candidate.coordinator_advances(),
            reference.coordinator_advances()
        );
        assert!(candidate
            .complete_row(candidate.output_layout(0).unwrap())
            .is_none());
        let Op::Tensor { programs, .. } = &candidate.ops[0] else {
            panic!("committed unrolled Tensor required")
        };
        for (offset, program) in (0..lanes).step_by(128).zip(programs) {
            assert!(Arc::ptr_eq(program, prepared.program_at(offset).unwrap()));
        }
        assert_eq!(programs.len(), prepared.invocations() as usize);
        assert_eq!(prepared.programs().count(), original.len());
        if lanes == 257 {
            assert_eq!(programs.len(), 3);
            assert_eq!(original.len(), 2);
        }
        drop(prepared);
        assert!(original.iter().all(|owner| owner.upgrade().is_some()));
        let actual = execute(&candidate, work.limits().unwrap());
        assert_eq!(actual, execute(&reference, work.limits().unwrap()));
        let budget = work.cursors();
        for actual in actual.2 {
            assert_eq!(actual.public_blocks, budget.public_blocks);
            assert_eq!(actual.pair_blocks, budget.pair_blocks);
            assert!(actual.private_blocks <= budget.private_blocks);
            assert!(actual.aes_work <= budget.aes_work);
        }
        drop((candidate, reference));
        assert!(original.iter().all(|owner| owner.upgrade().is_none()));
    }
}

#[test]
fn prepared_tensor_map_scope_is_exact_factory_provenance_and_preserves_original_arcs() {
    let input = Interval::new(-ONE, ONE).unwrap();
    for lanes in [1, 128, 129, 257] {
        // All valid native shifts still prepare; only shift 18 is in the
        // initial one-loop scope. No template/digest inspection grants scope.
        for shift in 1..=62 {
            let prepared = PreparedTensorMap::rescale(lanes, input, shift, 0, |_| Ok(())).unwrap();
            let originals: Vec<_> = prepared.programs().cloned().collect();
            assert_eq!(
                prepared.one_loop_scope(),
                (shift == 18).then_some(OneLoopScope::Rescale18)
            );
            prepared.validate().unwrap();
            for (actual, original) in prepared.programs().zip(&originals) {
                assert!(Arc::ptr_eq(actual, original));
            }
        }
        let prepared = PreparedTensorMap::guarded_silu(lanes, input, 0, |_| Ok(())).unwrap();
        let originals: Vec<_> = prepared.programs().cloned().collect();
        assert_eq!(prepared.one_loop_scope(), Some(OneLoopScope::GuardedSilu));
        prepared.validate().unwrap();
        for (actual, original) in prepared.programs().zip(&originals) {
            assert!(Arc::ptr_eq(actual, original));
        }
    }
}

#[test]
fn prepared_tensor_map_preparation_exact_one_short_and_preexisting_owner_baseline() {
    let caller = vec![0u8; 37];
    // The same callback TYPE/inline captures are used for measure/short/exact.
    // Its two borrowed log pointees are caller baseline, not erased behind F.
    fn run(
        silu: bool,
        baseline: usize,
        limit: usize,
        first: &std::cell::Cell<Option<usize>>,
        required: &std::cell::Cell<usize>,
    ) -> Result<PreparedTensorMap, Error> {
        let admit = move |bytes| {
            if first.get().is_none() {
                first.set(Some(bytes));
            }
            required.set(required.get().max(bytes));
            if bytes <= limit {
                Ok(())
            } else {
                Err(Error::Bounds)
            }
        };
        let input = Interval::new(-ONE, ONE).unwrap();
        if silu {
            PreparedTensorMap::guarded_silu(129, input, baseline, admit)
        } else {
            PreparedTensorMap::rescale(129, input, 3, baseline, admit)
        }
    }
    for silu in [false, true] {
        let first = std::cell::Cell::new(None);
        let required = std::cell::Cell::new(0);
        let baseline =
            size_of::<Vec<u8>>() + caller.capacity() + size_of_val(&first) + size_of_val(&required);
        let prepared = run(silu, baseline, usize::MAX, &first, &required).unwrap();
        let limit = required.get();
        assert_eq!(first.get(), Some(limit));
        drop(prepared);
        let before = native_attempts();
        first.set(None);
        required.set(0);
        let rejected = run(silu, baseline, limit - 1, &first, &required);
        assert!(matches!(rejected, Err(Error::Bounds)));
        assert_eq!(native_attempts(), before); // Rejected BEFORE either native factory.
        first.set(None);
        required.set(0);
        let accepted = run(silu, baseline, limit, &first, &required).unwrap();
        assert_eq!(native_attempts(), before + 2);
        assert_eq!(required.get(), limit);
        assert!(accepted.baseline_bytes().unwrap() + baseline <= limit);
    }
}

// Independent literal inventory: do not call the production assembly calculator.
// Its four usize fields/Result are quote controls, not native buffers. Native
// quotes already include their Program local and Arc publication; only the
// extra source Factory::make Result bridge is added below.
fn literal_preparation_slots(callback: usize) -> [usize; 4] {
    type Iter = std::iter::Chain<
        std::iter::Once<&'static Arc<tensor::Program>>,
        std::option::Iter<'static, Arc<tensor::Program>>,
    >;
    let common = 2 * callback
        + size_of::<&mut ()>()
        + size_of::<Result<(), Error>>()
        + size_of::<[usize; 4]>()
        + size_of::<Result<[usize; 4], Error>>()
        + size_of::<tensor::ConstructionQuote>()
        + size_of::<Option<tensor::ConstructionQuote>>()
        + 2 * size_of::<Result<PreparedTensorMap, Error>>();
    let quoting = 2 * size_of::<tensor::Construction>()
        + size_of::<Result<tensor::ConstructionQuote, tensor::ConstructionQuoteError>>()
        + 2 * size_of::<Result<tensor::ConstructionQuote, Error>>()
        + size_of::<Option<tensor::ConstructionQuote>>();
    let full = common
        + 2 * size_of::<Arc<tensor::Program>>()
        + size_of::<tensor::ConstructionQuote>()
        + size_of::<Result<tensor::Program, Error>>()
        + size_of::<Result<Arc<tensor::Program>, Error>>()
        + size_of::<tensor::Resources>()
        + size_of::<Result<tensor::Resources, Error>>();
    let tail = full
        + size_of::<tensor::ConstructionQuote>()
        + 2 * size_of::<Option<Arc<tensor::Program>>>();
    let visit = size_of::<std::iter::Once<&Arc<tensor::Program>>>()
        + size_of::<std::option::Iter<'_, Arc<tensor::Program>>>()
        + 3 * size_of::<Iter>()
        + size_of::<&Arc<tensor::Program>>()
        + (size_of::<tensor::Resources>() + size_of::<Result<tensor::Resources, Error>>())
            .max(size_of::<Option<usize>>());
    let geometry = 2 * size_of::<Option<&Arc<tensor::Program>>>()
        + size_of::<&usize>()
        + size_of::<&Arc<tensor::Program>>()
        + size_of::<Result<(), Error>>();
    let validation = geometry.max(visit + size_of::<Result<(), Error>>());
    let census = visit + size_of::<&mut Iter>() + 2 * size_of::<Result<usize, Error>>();
    let owner = common
        + size_of::<PreparedTensorMap>()
        + size_of::<Arc<tensor::Program>>()
        + size_of::<Option<Arc<tensor::Program>>>()
        + validation.max(census);
    [
        common + quoting.max(size_of::<tensor::ConstructionQuote>()),
        full,
        tail,
        owner,
    ]
}

#[test]
fn prepared_factory_literal_phase_inventory_and_owned_callback_captures() {
    fn check<const N: usize>(lanes: usize, silu: bool) -> (usize, usize) {
        let log = std::cell::Cell::new([0usize; 3]);
        let calls = std::cell::Cell::new(0usize);
        // Borrowed callback pointees are explicitly caller-owned baseline.
        let baseline = size_of_val(&log) + size_of_val(&calls);
        let capture = [0u8; N];
        let log_ref = &log;
        let calls_ref = &calls;
        let admit = move |bytes| {
            std::hint::black_box(&capture);
            let index = calls_ref.get();
            let mut values = log_ref.get();
            values[index] = bytes;
            log_ref.set(values);
            calls_ref.set(index + 1);
            Ok(())
        };
        let capture_bytes = size_of_val(&admit);
        let slots = literal_preparation_slots(capture_bytes);
        let quote = |n| {
            if silu {
                tensor::Construction::GuardedSilu { lanes: n }
            } else {
                tensor::Construction::Builder { lanes: n }
            }
            .quote()
            .unwrap()
        };
        let width = lanes.min(tensor::MAX_TILE_LANES);
        let remainder = lanes % width;
        let full_quote = quote(width);
        let tail_quote = (remainder != 0).then(|| quote(remainder));
        let mut expected = slots[0].max(slots[1] + full_quote.new_peak_bytes);
        let mut backing = full_quote.immutable_template_bytes;
        if let Some(tail) = tail_quote {
            expected =
                expected.max(slots[2] + full_quote.immutable_template_bytes + tail.new_peak_bytes);
            backing += tail.immutable_template_bytes;
        }
        expected = baseline + expected.max(slots[3] + backing);
        let input = Interval::new(-ONE, ONE).unwrap();
        let prepared = if silu {
            PreparedTensorMap::guarded_silu(lanes, input, baseline, admit)
        } else {
            PreparedTensorMap::rescale(lanes, input, 18, baseline, admit)
        }
        .unwrap();
        let values = log.get();
        assert_eq!(values[0], expected);
        assert_eq!(calls.get(), 2 + usize::from(remainder != 0));
        if let Some(tail) = tail_quote {
            assert_eq!(
                values[1],
                baseline
                    + slots[2]
                    + prepared
                        .program_at(0)
                        .unwrap()
                        .resources()
                        .unwrap()
                        .immutable_template_bytes
                    + tail.new_peak_bytes
            );
        }
        assert_eq!(
            values[calls.get() - 1],
            baseline + slots[3] + prepared.template_bytes().unwrap()
        );
        assert!(values[..calls.get()]
            .iter()
            .all(|bytes| *bytes <= values[0]));
        (values[0], capture_bytes)
    }
    for silu in [false, true] {
        for lanes in [1, 128, 129] {
            let (small, small_capture) = check::<0>(lanes, silu);
            let (large, large_capture) = check::<97>(lanes, silu);
            assert!(large_capture > small_capture);
            assert_eq!(large - small, 2 * (large_capture - small_capture));
        }
    }
}

#[test]
fn prepared_factory_invalid_parameters_and_baseline_overflow_precede_entry() {
    let input = Interval::new(-ONE, ONE).unwrap();
    let calls = std::cell::Cell::new(0usize);
    let mut admit = |_| {
        calls.set(calls.get() + 1);
        Ok(())
    };
    let before = native_attempts();
    for lanes in [0, crate::replicated_decoder::MAX_VALUES + 1, usize::MAX] {
        assert!(matches!(
            PreparedTensorMap::rescale(lanes, input, 18, 0, &mut admit),
            Err(Error::Bounds)
        ));
        assert!(matches!(
            PreparedTensorMap::guarded_silu(lanes, input, 0, &mut admit),
            Err(Error::Bounds)
        ));
    }
    assert_eq!(calls.get(), 0);
    for shift in [0, 63, u8::MAX] {
        assert!(matches!(
            PreparedTensorMap::rescale(129, input, shift, 0, &mut admit),
            Err(Error::Bounds)
        ));
    }
    assert!(matches!(
        PreparedTensorMap::rescale(129, input, 18, usize::MAX, &mut admit),
        Err(Error::Bounds)
    ));
    assert!(matches!(
        PreparedTensorMap::guarded_silu(129, input, usize::MAX, &mut admit),
        Err(Error::Bounds)
    ));
    assert_eq!(calls.get(), 3); // Invalid shifts retain first-callback precedence.
    assert_eq!(native_attempts(), before);
    assert!(matches!(
        PreparedTensorMap::rescale(129, input, 0, 0, |_| Err(Error::Binding)),
        Err(Error::Binding)
    ));
    assert!(std::panic::catch_unwind(|| {
        PreparedTensorMap::rescale(129, input, 0, 0, |_| {
            panic!("original admission precedence")
        })
    })
    .is_err());
    assert_eq!(native_attempts(), before);
}

#[test]
fn prepared_factory_callback_errors_and_unwinds_release_original_retained_owners() {
    for silu in [false, true] {
        for reject_at in [1, 2, 3] {
            for panic in [false, true] {
                // Observation only: Weak backing/trace allocations are test
                // instrumentation, not a claimed production admission term.
                observe_native_owners();
                let before = native_attempts();
                let mut calls = 0;
                let mut seen = Vec::new();
                let mut observed_live = false;
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let admit = |_| {
                        calls += 1;
                        if calls == reject_at {
                            seen = take_native_owners();
                            assert_eq!(seen.len(), reject_at - 1);
                            assert!(seen.iter().all(|owner| owner.strong_count() == 1));
                            observed_live = true;
                            if panic {
                                panic!("admission callback after original retention");
                            }
                            return Err(Error::Binding);
                        }
                        Ok(())
                    };
                    if silu {
                        PreparedTensorMap::guarded_silu(
                            129,
                            Interval::new(-ONE, ONE).unwrap(),
                            0,
                            admit,
                        )
                    } else {
                        PreparedTensorMap::rescale(
                            129,
                            Interval::new(-ONE, ONE).unwrap(),
                            18,
                            0,
                            admit,
                        )
                    }
                }));
                if panic {
                    assert!(result.is_err());
                } else {
                    assert!(matches!(result, Ok(Err(Error::Binding))));
                }
                assert_eq!(calls, reject_at);
                assert!(observed_live);
                assert_eq!(seen.len(), reject_at - 1);
                assert_eq!(native_attempts() - before, reject_at - 1);
                assert!(seen.iter().all(|owner| owner.upgrade().is_none()));
                assert!(take_native_owners().is_empty());
            }
        }
    }
}
