//! Closed factories and local BODY prepared-map execution.
//! No original Worker bridge, sealed source identity or row-proof authority.
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

fn drain_frames(
    roles: &mut [Executor; 3],
    mut queue: Vec<(Party, Party, crate::replicated_decoder::runtime::PeerBytes)>,
) {
    for _ in 0..1024 {
        if queue.is_empty() {
            return;
        }
        let mut next = Vec::new();
        for (from, to, frame) in queue {
            next.extend(
                roles[to as usize]
                    .receive(frame.as_bytes(), from)
                    .unwrap()
                    .into_iter()
                    .map(|(recipient, frame)| (to, recipient, frame)),
            );
        }
        queue = next;
    }
    panic!("frozen-wave drain exhausted");
}

fn prepare(b: &mut Builder, input: &Layout, silu: bool) -> PreparedTensorMap {
    if silu {
        b.prepare_guarded_silu_map(input, 0).unwrap()
    } else {
        b.prepare_rescale_map(input, 3, 0).unwrap()
    }
}

fn append_with_limit(limit: usize) -> (Result<Layout, Error>, Builder) {
    let (mut b, input) = builder(129);
    let prepared = prepare(&mut b, &input, false);
    b.proof_metadata_limit = limit;
    let result = b.prepared_tensor_map(&input, prepared, 0);
    (result, b)
}

#[derive(Clone, Copy)]
enum LaterPhase {
    Preparation,
    Append,
    Finish,
}

fn later_phase(limit: usize, phase: LaterPhase) -> bool {
    let (mut b, input) = builder(129);
    let first = prepare(&mut b, &input, false);
    let first_bytes = first.template_bytes().unwrap();
    let original: Vec<_> = first.programs().map(Arc::downgrade).collect();
    let first_output = b.prepared_tensor_map(&input, first, 0).unwrap();
    if matches!(phase, LaterPhase::Preparation) {
        b.proof_metadata_limit = limit;
    }
    let second = b.prepare_rescale_map(&first_output, 3, 0);
    let second = match second {
        Ok(p) => p,
        Err(Error::Bounds) => {
            assert!(matches!(b.input(0), Err(Error::Binding)));
            assert!(original.iter().all(|p| p.strong_count() == 1));
            return false;
        }
        Err(e) => panic!("unexpected preparation error: {e:?}"),
    };
    if matches!(phase, LaterPhase::Preparation) {
        return true;
    }
    let second_bytes = second.template_bytes().unwrap();
    if matches!(phase, LaterPhase::Append) {
        b.proof_metadata_limit = limit;
    }
    let output = match b.prepared_tensor_map(&first_output, second, 0) {
        Ok(output) => output,
        Err(Error::Bounds) => {
            assert_eq!(b.ops.len(), 1);
            assert!(original.iter().all(|p| p.strong_count() == 1));
            assert!(matches!(b.input(0), Err(Error::Binding)));
            return false;
        }
        Err(e) => panic!("unexpected append error: {e:?}"),
    };
    if matches!(phase, LaterPhase::Append) {
        return true;
    }
    b.proof_metadata_limit = limit;
    match b.finish(&[output]) {
        Ok(plan) => {
            assert!(plan.resources().immutable_metadata_bytes >= first_bytes + second_bytes);
            assert!(original.iter().all(|p| p.strong_count() == 1));
            true
        }
        Err(Error::Bounds) => {
            // Consuming failed finish drops Builder and its committed templates.
            assert!(original.iter().all(|p| p.upgrade().is_none()));
            false
        }
        Err(e) => panic!("unexpected finish error: {e:?}"),
    }
}

#[test]
fn prepared_tensor_map_original_arcs_flat_tail_wire_both_shares_and_cursors() {
    for (lanes, silu, permuted) in [
        (1, false, false),
        (128, false, false),
        (129, false, false),
        (257, false, false),
        (129, true, false),
        (129, false, true),
    ] {
        let (mut b, input) = builder(lanes);
        let input = if permuted {
            Layout::affine(0, 0, &[3, 43], &[1, 3]).unwrap()
        } else {
            input
        };
        let prepared = prepare(&mut b, &input, silu);
        let original: Vec<_> = prepared.programs().map(Arc::downgrade).collect();
        let reference = unrolled(&input, &prepared);
        let expected_work = prepared.work_budget().unwrap();
        let expected_bounds = prepared.output_bounds().unwrap();
        let output = b.prepared_tensor_map(&input, prepared, 0).unwrap();
        assert_eq!(b.bounds(&output).unwrap(), expected_bounds);
        assert!(b.proofs.is_empty());
        assert!(b.templates.is_empty()); // No substitution or hidden interner copies.
        let candidate = b.finish(&[output]).unwrap();
        assert_ne!(candidate.digest(), reference.digest());
        assert_eq!(candidate.work_budget(), reference.work_budget());
        assert_eq!(candidate.work_budget(), expected_work);
        assert_eq!(
            candidate.coordinator_advances(),
            reference.coordinator_advances()
        );
        assert_eq!(
            candidate.statistics().tensor_invocations,
            reference.statistics().tensor_invocations
        );
        assert_eq!(
            candidate.statistics().tensor_instructions,
            reference.statistics().tensor_instructions
        );
        assert!(candidate
            .complete_row(candidate.output_layout(0).unwrap())
            .is_none());
        let Op::PreparedTensorMap { prepared, .. } = &candidate.ops[0] else {
            panic!("map op")
        };
        assert_eq!(
            prepared.programs().count(),
            1 + usize::from(lanes > 128 && lanes % 128 != 0)
        );
        for (p, original) in prepared.programs().zip(original) {
            assert!(Arc::ptr_eq(p, &original.upgrade().unwrap()));
        }
        if lanes == 257 {
            let Op::Tensor { programs, .. } = &reference.ops[0] else {
                panic!("unrolled op")
            };
            assert_eq!(programs.len(), 3);
            assert_eq!(prepared.programs().count(), 2);
            assert_eq!(candidate.statistics().range_regions, 2); // Input + ONE map hull.
        }
        let limits = expected_work.limits().unwrap();
        let a = execute(&candidate, limits);
        let r = execute(&reference, limits);
        assert_eq!(a, r);
        let budget = expected_work.cursors();
        for actual in a.2 {
            assert_eq!(actual.public_blocks, budget.public_blocks);
            assert_eq!(actual.pair_blocks, budget.pair_blocks);
            assert!(actual.private_blocks <= budget.private_blocks);
            assert!(actual.aes_work <= budget.aes_work);
        }
    }
}

#[test]
fn prepared_tensor_map_local_declaration_subsets_never_mint_external_authority() {
    // These are compiler declarations, not checks of secret values or source
    // authority. A different local Builder may declare any accepted subset.
    for (lower, upper, accepted) in [
        (-ONE, ONE, true),
        (-ONE / 2, ONE / 2, true),
        (-ONE, -ONE, true),
        (ONE, ONE, true),
        (-ONE - 1, ONE, false),
        (-ONE, ONE + 1, false),
        (-2 * ONE, 2 * ONE, false),
        (ONE + 1, ONE + 2, false),
    ] {
        let (mut producer, x) = builder(129);
        let prepared = prepare(&mut producer, &x, false);
        let actual = Interval::new(lower, upper).unwrap();
        assert_eq!(prepared.accepts(129, actual), accepted);
        assert!(!prepared.accepts(128, actual));
        let mut b = Builder::new(
            WeightMode::Public,
            &[Register {
                lanes: 129,
                bounds: actual,
            }],
        )
        .unwrap();
        let input = b.input(0).unwrap();
        let result = b.prepared_tensor_map(&input, prepared, 0);
        assert_eq!(result.is_ok(), accepted);
        assert!(b.proofs.is_empty());
        if !accepted {
            assert!(matches!(result, Err(Error::Binding)));
            assert!(b.ops.is_empty());
            assert!(matches!(b.input(0), Err(Error::Binding)));
        }
    }
    let (mut producer, x) = builder(129);
    let prepared = prepare(&mut producer, &x, false);
    let (mut b, wrong_len) = builder(128);
    assert!(matches!(
        b.prepared_tensor_map(&wrong_len, prepared, 0),
        Err(Error::Binding)
    ));
    assert!(matches!(b.input(0), Err(Error::Binding)));

    let (mut b, input) = builder(129);
    let before = native_attempts();
    // Independent caller backing cannot be relabelled as an interner/native
    // credit. Rejection precedes both factories and burns the Builder.
    assert!(matches!(
        b.prepare_rescale_map(&input, 3, MAX_PROGRAM_BYTES),
        Err(Error::Bounds)
    ));
    assert_eq!(native_attempts(), before);
    assert!(matches!(b.input(0), Err(Error::Binding)));
}

#[test]
fn prepared_tensor_map_append_capacity_peak_exact_one_short_and_allocation_failure_burn() {
    // Search only the existing owner-defined byte gate, not an allocator/RSS
    // measurement. Each trial has fresh owners and original native templates.
    let (mut lo, mut hi) = (0, MAX_PROGRAM_BYTES);
    assert!(append_with_limit(hi).0.is_ok());
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if append_with_limit(mid).0.is_ok() {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    assert!(append_with_limit(lo).0.is_ok());
    let (result, mut rejected) = append_with_limit(lo - 1);
    assert!(matches!(result, Err(Error::Bounds)));
    assert_eq!(rejected.registers.len(), 1);
    assert!(rejected.ops.is_empty());
    assert!(matches!(rejected.input(0), Err(Error::Binding)));

    let (mut existing, input) = builder(129);
    let mut native = tensor::Builder::new(1, Interval::new(-ONE, ONE).unwrap()).unwrap();
    let output = native.rescale(native.input(), 3).unwrap();
    existing.intern(native.finish(output).unwrap()).unwrap();
    let prepared = prepare(&mut existing, &input, false);
    existing.proof_metadata_limit = lo;
    assert!(matches!(
        existing.prepared_tensor_map(&input, prepared, 0),
        Err(Error::Bounds)
    ));
    assert_eq!(existing.templates.len(), 1); // Existing interner remains priced.
    assert!(existing.ops.is_empty());
    assert!(matches!(existing.input(0), Err(Error::Binding)));

    let (mut b, input) = builder(129);
    let prepared = prepare(&mut b, &input, false);
    b.fail_builder_allocation = true;
    assert!(matches!(
        b.prepared_tensor_map(&input, prepared, 0),
        Err(Error::Allocation)
    ));
    assert!(b.ops.is_empty());
    assert!(matches!(b.input(0), Err(Error::Binding)));
}

#[test]
fn prepared_tensor_map_cancel_malformed_receive_and_premature_finish_retire_owners() {
    let (mut b, input) = builder(129);
    let prepared = prepare(&mut b, &input, false);
    let output = b.prepared_tensor_map(&input, prepared, 0).unwrap();
    let plan = b.finish(&[output]).unwrap();
    for active in [false, true] {
        let mut owners = owners(&plan, plan.work_budget().limits().unwrap());
        for owner in &mut owners {
            if active {
                drop(owner.advance().unwrap());
            }
            owner.cancel();
            assert_eq!(owner.live_bytes(), 0);
            assert!(!owner.complete());
            assert!(matches!(owner.advance(), Err(Error::Retired)));
            assert!(matches!(
                owner.receive(b"bad", Party::P0),
                Err(Error::Retired)
            ));
        }
    }
    let mut owners = owners(&plan, plan.work_budget().limits().unwrap());
    for (i, owner) in owners.iter_mut().enumerate() {
        drop(owner.advance().unwrap());
        assert!(owner.receive(b"bad", PARTIES[i].next().next()).is_err());
        assert_eq!(owner.live_bytes(), 0);
        assert!(matches!(owner.advance(), Err(Error::Retired)));
    }
    for owner in self::owners(&plan, plan.work_budget().limits().unwrap()) {
        assert!(matches!(owner.finish(), Err(Error::Phase)));
    }
}

#[test]
fn prepared_tensor_map_returns_original_context_and_next_ordinal_without_prss_reset() {
    let (mut b, input) = builder(129);
    let prepared = prepare(&mut b, &input, false);
    let output = b.prepared_tensor_map(&input, prepared, 0).unwrap();
    let plan = b.finish(&[output]).unwrap();
    let followup = WorkBudget::product(1, false).unwrap();
    let limits = plan
        .work_budget()
        .checked_add(followup)
        .unwrap()
        .limits()
        .unwrap();
    let mut roles = owners(&plan, limits);
    drive(&mut roles).unwrap();
    for owner in roles {
        let (mut runtime, values) = owner.finish().unwrap();
        assert_eq!(values.output_len(0).unwrap(), 129);
        assert!(runtime.matches_context(puma::Context::new([71; 32], [92; 32], 0).unwrap()));
        let cursors = runtime.cursors();
        let bounds = InputBounds {
            left: Interval::new(0, 1).unwrap(),
            right: Interval::new(0, 1).unwrap(),
        };
        let next = runtime.plan_ring_product(&[bounds]).unwrap();
        assert_eq!(next.sequence(), plan.work_budget().operations());
        assert_eq!(runtime.cursors(), cursors); // Planning did not reset/draw PRSS.
        runtime.cancel(); // Abandoned planned invocation must burn original owner.
        assert_eq!(runtime.status(), multiply::Status::Retired);
    }
}

#[test]
fn prepared_tensor_map_native_envelope_and_work_exact_one_short() {
    let (mut b, input) = builder(1);
    let prepared = prepare(&mut b, &input, true);
    let bytes = prepared
        .program_at(0)
        .unwrap()
        .resources()
        .unwrap()
        .shared_admitted_bytes;
    let output = b.prepared_tensor_map(&input, prepared, 0).unwrap();
    let plan = b.finish(&[output]).unwrap();
    let work = plan.work_budget();
    let aes = work.cursors().aes_work;
    let exact = PrssLimits::new(
        puma::Limits::new(128, work.operations(), bytes).unwrap(),
        aes,
    )
    .unwrap();
    execute(&plan, exact);
    for limits in [
        PrssLimits::new(
            puma::Limits::new(128, work.operations(), bytes - 1).unwrap(),
            aes,
        )
        .unwrap(),
        PrssLimits::program(work.operations() - 1, aes).unwrap(),
        PrssLimits::program(work.operations(), aes - 1).unwrap(),
    ] {
        let mut owners = owners(&plan, limits);
        for owner in &mut owners {
            assert!(matches!(owner.advance(), Err(Error::Bounds)));
            assert_eq!(owner.live_bytes(), 0);
            assert!(matches!(owner.advance(), Err(Error::Retired)));
        }
    }
}

#[test]
fn prepared_tensor_map_active_tail_fault_cancel_and_retained_real_frames() {
    let (mut b, input) = builder(129);
    let prepared = prepare(&mut b, &input, false);
    assert_eq!(prepared.program_at(0).unwrap().lanes(), 128);
    assert_eq!(prepared.program_at(0).unwrap().instructions(), 1);
    assert_eq!(prepared.program_at(128).unwrap().lanes(), 1);
    let output = b.prepared_tensor_map(&input, prepared, 0).unwrap();
    let plan = b.finish(&[output]).unwrap();
    for fault in [false, true] {
        let context = puma::Context::new([71; 32], [92; 32], 0).unwrap();
        let runtimes = puma::prss::insecure_test_support::fresh_compound_parties(
            context,
            plan.work_budget().limits().unwrap(),
        )
        .unwrap();
        let usage = runtimes.each_ref().map(|r| r.payload_usage());
        let mut roles = runtimes.map(|runtime| {
            let value = secret(runtime.party(), 129);
            Executor::new(plan.clone(), runtime, vec![value], None, None).unwrap()
        });
        let mut full = Vec::new();
        for (i, role) in roles.iter_mut().enumerate() {
            full.extend(
                role.advance()
                    .unwrap()
                    .into_iter()
                    .map(|(to, frame)| (PARTIES[i], to, frame)),
            );
        }
        // Rescale has one instruction. Draining completes full tile; following
        // advance stores it and starts the one-lane tail, not a fresh runtime.
        drain_frames(&mut roles, full);
        let mut retained = Vec::new();
        for (i, role) in roles.iter_mut().enumerate() {
            retained.extend(
                role.advance()
                    .unwrap()
                    .into_iter()
                    .map(|(to, frame)| (PARTIES[i], to, frame)),
            );
            assert!(!role.complete());
        }
        assert!(!retained.is_empty());
        let bytes: Vec<_> = retained
            .iter()
            .map(|(_, _, frame)| frame.as_bytes().to_vec())
            .collect();
        for (i, role) in roles.iter_mut().enumerate() {
            if fault {
                assert!(role.receive(b"bad", PARTIES[i].next()).is_err());
            } else {
                role.cancel();
            }
            assert_eq!(role.live_bytes(), 0);
            assert!(matches!(role.advance(), Err(Error::Retired)));
        }
        // Keep ORIGINAL PeerBytes, not reconstructed byte vectors, alive over
        // cancellation. Their ledger charges survive native secret retirement.
        assert!(usage.iter().any(|u| u.live_bytes() != 0));
        for ((_, _, frame), bytes) in retained.iter().zip(bytes) {
            assert_eq!(frame.as_bytes(), bytes);
        }
        drop(retained);
        assert!(usage.iter().all(|u| u.live_bytes() == 0));
    }
}

#[test]
fn prepared_tensor_map_values_retain_plan_and_original_templates_until_last_drop() {
    let (mut b, input) = builder(129);
    let prepared = prepare(&mut b, &input, false);
    let templates: Vec<_> = prepared.programs().map(Arc::downgrade).collect();
    let output = b.prepared_tensor_map(&input, prepared, 0).unwrap();
    let plan = b.finish(&[output]).unwrap();
    let weak_plan = Arc::downgrade(&plan);
    let mut roles = owners(&plan, plan.work_budget().limits().unwrap());
    drive(&mut roles).unwrap();
    let mut values: Vec<_> = roles
        .into_iter()
        .map(|owner| {
            let (mut runtime, values) = owner.finish().unwrap();
            runtime.cancel();
            values
        })
        .collect();
    let expected: Vec<_> = values.iter().map(|v| v.word(0, 128).unwrap()).collect();
    drop(plan);
    assert_eq!(weak_plan.strong_count(), 3);
    assert!(templates.iter().all(|p| p.strong_count() == 1));
    for (v, expected) in values.iter().zip(expected) {
        assert_eq!(v.output_len(0).unwrap(), 129);
        assert_eq!(v.word(0, 128).unwrap(), expected);
    }
    while values.len() > 1 {
        drop(values.pop());
        assert!(weak_plan.upgrade().is_some());
        assert!(templates.iter().all(|p| p.strong_count() == 1));
    }
    drop(values);
    assert!(weak_plan.upgrade().is_none());
    assert!(templates.iter().all(|p| p.upgrade().is_none()));
}

#[test]
fn prepared_tensor_map_same_digest_interner_never_substitutes_original_arcs() {
    let (mut b, input) = builder(129);
    let prepared = prepare(&mut b, &input, false);
    let original: Vec<_> = prepared.programs().map(Arc::downgrade).collect();
    for p in prepared.programs() {
        // Independent immutable allocation with identical numeric digest.
        let interned = b.intern((**p).clone()).unwrap();
        assert_eq!(interned.digest(), p.digest());
        assert!(!Arc::ptr_eq(&interned, p));
    }
    let output = b.prepared_tensor_map(&input, prepared, 0).unwrap();
    assert_eq!(b.templates.len(), 2);
    let Op::PreparedTensorMap { prepared, .. } = &b.ops[0] else {
        panic!("map")
    };
    for (p, weak) in prepared.programs().zip(&original) {
        assert!(Arc::ptr_eq(p, &weak.upgrade().unwrap()));
        assert!(!Arc::ptr_eq(p, b.templates.get(&p.digest()).unwrap()));
    }
    let plan = b.finish(&[output]).unwrap();
    let Op::PreparedTensorMap { prepared, .. } = &plan.ops[0] else {
        panic!("map")
    };
    for (p, weak) in prepared.programs().zip(original) {
        assert!(Arc::ptr_eq(p, &weak.upgrade().unwrap()));
    }
    execute(&plan, plan.work_budget().limits().unwrap());
}

#[test]
fn prepared_tensor_map_committed_backing_later_phases_exact_and_one_short() {
    for phase in [
        LaterPhase::Preparation,
        LaterPhase::Append,
        LaterPhase::Finish,
    ] {
        let (mut lo, mut hi) = (0, MAX_PROGRAM_BYTES);
        assert!(later_phase(hi, phase));
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if later_phase(mid, phase) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        assert!(later_phase(lo, phase));
        assert!(!later_phase(lo - 1, phase));
    }
}

#[test]
fn prepared_map_active_original_full_tail_arcs_release_on_all_exit_paths() {
    for exit in 0..4 {
        let (mut b, input) = builder(129);
        let prepared = prepare(&mut b, &input, false);
        assert_eq!(prepared.program_at(0).unwrap().instructions(), 1);
        let full = Arc::downgrade(prepared.program_at(0).unwrap());
        let tail = Arc::downgrade(prepared.program_at(128).unwrap());
        let output = b.prepared_tensor_map(&input, prepared, 0).unwrap();
        let plan = b.finish(&[output]).unwrap();
        let weak_plan = Arc::downgrade(&plan);
        assert_eq!((full.strong_count(), tail.strong_count()), (1, 1));
        let before = native_attempts();
        let mut roles = owners(&plan, plan.work_budget().limits().unwrap());
        let mut frames = Vec::new();
        for (i, role) in roles.iter_mut().enumerate() {
            frames.extend(
                role.advance()
                    .unwrap()
                    .into_iter()
                    .map(|(to, f)| (PARTIES[i], to, f)),
            );
        }
        // Plan owns each original once. Each active native executor must pin
        // that SAME full Arc: reverting to owned new() leaves this count at 1.
        assert_eq!((full.strong_count(), tail.strong_count()), (4, 1));
        drain_frames(&mut roles, frames);
        assert_eq!((full.strong_count(), tail.strong_count()), (4, 1));
        let mut retained = Vec::new();
        for (i, role) in roles.iter_mut().enumerate() {
            retained.extend(
                role.advance()
                    .unwrap()
                    .into_iter()
                    .map(|(to, f)| (PARTIES[i], to, f)),
            );
        }
        assert!(!retained.is_empty());
        assert_eq!((full.strong_count(), tail.strong_count()), (1, 4));
        if exit == 0 {
            drain_frames(&mut roles, retained);
            drive(&mut roles).unwrap();
            assert_eq!((full.strong_count(), tail.strong_count()), (1, 1));
            let finished = roles.map(|role| role.finish().unwrap());
            drop(plan);
            assert_eq!(weak_plan.strong_count(), 3);
            assert_eq!((full.strong_count(), tail.strong_count()), (1, 1));
            drop(finished);
        } else {
            if exit == 3 {
                for role in roles {
                    assert!(matches!(role.finish(), Err(Error::Phase)));
                }
            } else {
                for (i, role) in roles.iter_mut().enumerate() {
                    if exit == 1 {
                        role.cancel();
                    } else {
                        assert!(role.receive(b"bad", PARTIES[i].next()).is_err());
                    }
                    assert_eq!(role.live_bytes(), 0);
                }
                drop(roles);
            }
            assert_eq!((full.strong_count(), tail.strong_count()), (1, 1));
            drop(plan);
            // Original frames survive cancellation, but carry no hidden native
            // Program ownership. Their separate ledger leases are tested above.
            assert!(full.upgrade().is_none());
            assert!(tail.upgrade().is_none());
            drop(retained);
        }
        assert_eq!(native_attempts(), before); // No factory re-entry per tile.
        assert!(weak_plan.upgrade().is_none());
        assert!(full.upgrade().is_none());
        assert!(tail.upgrade().is_none());
    }
}

#[test]
fn prepared_map_actual_spare_op_capacity_and_retained_backing_are_charged_once() {
    for one_short in [false, true] {
        let (mut b, input) = builder(257); // Two full invocations, ONE full owner.
        let prepared = prepare(&mut b, &input, false);
        let templates = prepared.template_bytes().unwrap();
        assert_eq!(prepared.programs().count(), 2);
        let output = b.prepared_tensor_map(&input, prepared, 0).unwrap();
        assert_eq!(b.ops.len(), 1);
        assert!(b.ops.capacity() > b.ops.len());
        assert!(b.templates.is_empty() && b.template_metadata.is_empty());
        assert!(b.proofs.is_empty());
        let Op::PreparedTensorMap { input: stored, .. } = &b.ops[0] else {
            panic!("map required")
        };
        let input_metadata = stored.metadata_bytes();
        // Literal owner-local census: EVERY actual spare inline Op slot,
        // but retained native allocations once per initialized map, not per
        // spare slot or per full/tail invocation. This uses the NEW Op size;
        // no assumption of unchanged legacy enum layout/admission is made.
        let live = size_of::<Builder>()
            + 512
            + b.registers.capacity() * size_of::<Register>()
            + b.regions.capacity() * size_of::<Box<[Region]>>()
            + b.ops.capacity() * size_of::<Op>()
            + b.proofs.capacity() * size_of::<Proof>()
            + b.regions
                .iter()
                .map(|r| r.len() * size_of::<Region>() + 64)
                .sum::<usize>()
            + input_metadata
            + 64
            + templates;
        b.proof_metadata_limit = live - usize::from(one_short);
        if one_short {
            assert!(matches!(b.proof_preflight(0), Err(Error::Bounds)));
            assert!(matches!(b.input(0), Err(Error::Binding)));
            continue;
        }
        b.proof_preflight(0).unwrap();
        b.proof_metadata_limit = MAX_PROGRAM_BYTES;
        let plan = b.finish(&[output]).unwrap();
        assert_eq!(plan.ops.len(), 1);
        // Final boxed inline slots + two distinct layout owners + native
        // full/tail backing, once. Legacy retirement capacity policy unchanged.
        let other = size_of::<Plan>()
            + plan.registers.len() * (size_of::<Register>() + size_of::<Box<[Region]>>())
            + plan
                .regions
                .iter()
                .map(|r| r.len() * size_of::<Region>())
                .sum::<usize>()
            + plan.ops.len() * size_of::<Op>()
            + input_metadata
            + plan.outputs[0].metadata_bytes()
            + std::mem::size_of_val(plan.outputs.as_ref())
            + plan.retire.len() * size_of::<Vec<usize>>()
            + plan
                .retire
                .iter()
                .map(|r| r.capacity() * size_of::<usize>())
                .sum::<usize>()
            + plan.unused.len() * size_of::<usize>()
            + plan.proofs.len() * size_of::<Proof>();
        assert_eq!(plan.resources().immutable_metadata_bytes, other + templates);
        assert_eq!(
            plan.resources().primitive_ledger_allowance_bytes,
            2 * puma::MAX_ADMITTED_BYTES
        );
    }
}

// Oracle-only ordinary Tensor composition inside an existing Builder. These are
// the exact prepared Arc owners, not reconstructed kernels or per-tile bounds.
// This local protocol oracle does not qualify enclosing constructor admission.
fn append_original_unrolled(
    b: &mut Builder,
    input: &Layout,
    prepared: &PreparedTensorMap,
) -> Layout {
    assert!(prepared.accepts(input.len(), b.bounds(input).unwrap()));
    let baseline = prepared.baseline_bytes().unwrap();
    for program in prepared.programs() {
        let resources = program.resources().unwrap();
        if let Some(original) = b.templates.get(&program.digest()) {
            assert!(
                Arc::ptr_eq(original, program),
                "oracle must not substitute owners"
            );
        } else {
            b.template_insert_preflight(&program.digest(), resources, baseline)
                .unwrap();
            b.templates.insert(program.digest(), program.clone());
            b.template_metadata
                .insert(program.digest(), resources.immutable_template_bytes);
        }
    }
    let stride = prepared.program_at(0).unwrap().lanes();
    let mut programs = b
        .new_template_list(input, prepared.invocations() as usize, baseline)
        .unwrap();
    for offset in (0..input.len()).step_by(stride) {
        programs.push(prepared.program_at(offset).unwrap().clone());
    }
    let mut regions = b.region_buffer(1, baseline).unwrap();
    regions.push(Region {
        offset: 0,
        len: input.len(),
        bounds: prepared.output_bounds().unwrap(),
    });
    b.push(
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
    .unwrap()
}

type AllComponents = Vec<Vec<(u64, u64)>>;
type AllExecution = (Wire, [AllComponents; 3], [Cursors; 3]);

fn execute_all_outputs(plan: &Arc<Plan>, limits: PrssLimits) -> AllExecution {
    let mut owners = owners(plan, limits);
    let (wire, advances) = drive(&mut owners).unwrap();
    assert_eq!(advances, plan.coordinator_advances());
    let finished = owners.map(|owner| owner.finish().unwrap());
    let components = finished.each_ref().map(|(_, values)| {
        (0..plan.outputs.len())
            .map(|output| {
                (0..values.output_len(output).unwrap())
                    .map(|lane| values.word(output, lane).unwrap())
                    .collect()
            })
            .collect()
    });
    let cursors = finished.each_ref().map(|(runtime, _)| runtime.cursors());
    (wire, components, cursors)
}

// Literal original native work and sticky products, independent of the map's
// cached multiplicity/work methods. Not a canonical decoder identity or issuer.
struct OriginalLocalSchedule {
    work: WorkBudget,
    advances: u64,
    invocations: u64,
    instructions: u64,
    guard_tiles: u64,
}

impl OriginalLocalSchedule {
    fn native(&mut self, program: &tensor::Program) {
        self.work = self.work.checked_add(program.work_budget()).unwrap();
        self.invocations += 1;
        self.instructions += program.instructions() as u64;
        self.advances += program.instructions() as u64;
    }

    fn from_plan(plan: &Plan) -> Self {
        let mut original = Self {
            work: WorkBudget::default(),
            advances: 1, // Existing final BODY completion advance.
            invocations: 0,
            instructions: 0,
            guard_tiles: 0,
        };
        for op in plan.ops.iter() {
            match op {
                Op::PreparedTensorMap { input, prepared } => {
                    let stride = prepared.program_at(0).unwrap().lanes();
                    for offset in (0..input.len()).step_by(stride) {
                        original.native(prepared.program_at(offset).unwrap());
                    }
                }
                Op::Tensor { programs, .. } => {
                    for program in programs.iter() {
                        original.native(program);
                    }
                }
                Op::Check {
                    previous,
                    checked,
                    valid,
                    ..
                } => {
                    assert_eq!(checked.len(), valid.len());
                    for (tile, (check, all_valid)) in checked.iter().zip(valid.iter()).enumerate() {
                        original.native(check);
                        original.native(all_valid);
                        original.guard_tiles += 1;
                        if tile > 0 || previous.is_some() {
                            original.work = original
                                .work
                                .checked_add(WorkBudget::product(1, false).unwrap())
                                .unwrap();
                            original.advances += 1;
                        }
                    }
                }
                _ => panic!("fixture only admits maps, ordinary Tensor and original Check"),
            }
        }
        original
    }

    fn check(&self, plan: &Plan) {
        assert_eq!(self.work, plan.work_budget());
        assert_eq!(self.advances, plan.coordinator_advances());
        let statistics = plan.statistics();
        assert_eq!(self.invocations, statistics.tensor_invocations);
        assert_eq!(self.instructions, statistics.tensor_instructions);
        assert_eq!(self.guard_tiles, statistics.guard_tiles);
        assert_eq!(statistics.matrix_products, 0);
        assert_eq!(statistics.logical_public_file_read_bytes, 0);
        assert_eq!(statistics.public_ring64_macs, 0);
        assert_eq!(statistics.public_bias_vector_reads, 0);
    }
}

fn consecutive_original_pair(lanes: usize, guarded: bool) -> (Arc<Plan>, Arc<Plan>) {
    let (mut mapped, mut input) = builder(lanes);
    let (mut ordinary, mut reference_input) = builder(lanes);
    let (previous, reference_previous) = if guarded {
        let (safe, previous) = mapped.checked(&input, ONE / 2, None).unwrap();
        let (reference_safe, reference_previous) =
            ordinary.checked(&reference_input, ONE / 2, None).unwrap();
        input = safe;
        reference_input = reference_safe;
        assert!(previous.is_some() && reference_previous.is_some());
        (previous, reference_previous)
    } else {
        (None, None)
    };
    let mut outputs = Vec::new();
    let mut reference_outputs = Vec::new();
    // First is genuinely Q18. Different second shift prevents the oracle from
    // using a foreign same-digest interner owner; neither scope grants authority.
    for shift in [18, 3] {
        let prepared = mapped.prepare_rescale_map(&input, shift, 0).unwrap();
        let reference = append_original_unrolled(&mut ordinary, &reference_input, &prepared);
        input = mapped.prepared_tensor_map(&input, prepared, 0).unwrap();
        reference_input = reference;
        assert_eq!(
            mapped.bounds(&input).unwrap(),
            ordinary.bounds(&reference_input).unwrap()
        );
        outputs.push(input.clone());
        reference_outputs.push(reference_input.clone());
    }
    if guarded {
        let (safe, valid) = mapped.checked(&input, 0, previous.as_ref()).unwrap();
        let (reference_safe, reference_valid) = ordinary
            .checked(&reference_input, 0, reference_previous.as_ref())
            .unwrap();
        outputs.extend([safe, valid.unwrap(), previous.unwrap()]);
        reference_outputs.extend([
            reference_safe,
            reference_valid.unwrap(),
            reference_previous.unwrap(),
        ]);
    }
    assert!(mapped.proofs.is_empty() && ordinary.proofs.is_empty());
    let mapped = mapped.finish(&outputs).unwrap();
    let ordinary = ordinary.finish(&reference_outputs).unwrap();
    let map_start = usize::from(guarded);
    assert!(matches!(
        &mapped.ops[map_start],
        Op::PreparedTensorMap { .. }
    ));
    assert!(matches!(
        &mapped.ops[map_start + 1],
        Op::PreparedTensorMap { .. }
    ));
    (mapped, ordinary)
}

#[test]
fn prepared_map_consecutive_q18_and_mixed_sticky_sites_match_original_full_execution() {
    for (lanes, guarded) in [
        (1, false),
        (128, false),
        (129, false),
        (256, false), // EXACT two-full / no-tail case, not 257's tail path.
        (257, false),
        (1, true),
        (129, true),
        (256, true),
    ] {
        let (mapped, ordinary) = consecutive_original_pair(lanes, guarded);
        let original = OriginalLocalSchedule::from_plan(&ordinary);
        original.check(&mapped);
        original.check(&ordinary);
        let mapped_stats = mapped.statistics();
        let ordinary_stats = ordinary.statistics();
        assert_eq!(mapped_stats.semantic_sites, ordinary_stats.semantic_sites);
        assert_eq!(mapped_stats.register_owners, ordinary_stats.register_owners);
        assert_eq!(mapped_stats.range_regions, ordinary_stats.range_regions);
        assert!(mapped
            .outputs
            .iter()
            .all(|output| mapped.complete_row(output).is_none()));
        if guarded {
            assert_eq!(original.guard_tiles, 2 * lanes.div_ceil(128) as u64);
        } else {
            assert_eq!(original.invocations, 2 * lanes.div_ceil(128) as u64);
        }
        if lanes == 256 {
            for op in mapped.ops.iter() {
                if let Op::PreparedTensorMap { prepared, .. } = op {
                    assert_eq!(prepared.programs().count(), 1);
                    assert_eq!(prepared.invocations(), 2);
                    assert!(Arc::ptr_eq(
                        prepared.program_at(0).unwrap(),
                        prepared.program_at(128).unwrap()
                    ));
                }
            }
        }
        let before = native_attempts();
        let actual = execute_all_outputs(&mapped, original.work.limits().unwrap());
        let reference = execute_all_outputs(&ordinary, original.work.limits().unwrap());
        assert_eq!(actual, reference); // ALL bytes, ALL output components, all cursors.
        assert_eq!(native_attempts(), before); // No factory re-entry while executing.
        let budget = original.work.cursors();
        for cursors in actual.2 {
            assert_eq!(cursors.public_blocks, budget.public_blocks);
            assert_eq!(cursors.pair_blocks, budget.pair_blocks);
            assert!(cursors.private_blocks <= budget.private_blocks);
            assert!(cursors.aes_work <= budget.aes_work);
        }
        if guarded {
            // Known fixture x[0] == -ONE fails the first genuine half-ONE guard.
            // Sticky 0 survives both maps and the later real guard/AllValid.
            for output in [3, 4] {
                assert_eq!(
                    actual
                        .1
                        .iter()
                        .fold(0u64, |sum, party| sum.wrapping_add(party[output][0].0)),
                    0
                );
            }
        }
    }
}

#[test]
fn prepared_map_real_full_tail_native_shared_limits_and_immutable_failure_winner() {
    for silu in [false, true] {
        let (mut b, input) = builder(129);
        let prepared = if silu {
            b.prepare_guarded_silu_map(&input, 0).unwrap()
        } else {
            b.prepare_rescale_map(&input, 18, 0).unwrap()
        };
        let full = prepared.program_at(0).unwrap();
        let tail = prepared.program_at(128).unwrap();
        assert_eq!((full.lanes(), tail.lanes()), (128, 1));
        let full_bytes = full.resources().unwrap().shared_admitted_bytes;
        let tail_bytes = tail.resources().unwrap().shared_admitted_bytes;
        assert!(
            full_bytes > tail_bytes,
            "fixture requires the full envelope to dominate"
        );
        let width = full.lanes();
        // The closed Q18 factory emits one Rescale: admit(width), without the
        // Sign/FromBit expansion required by the original guarded SiLU kernel.
        let native_lanes = width * if silu { 2 } else { 1 };
        if !silu {
            assert_eq!(full.instructions(), 1);
        }
        let reference = unrolled(&input, &prepared);
        let output = b.prepared_tensor_map(&input, prepared, 0).unwrap();
        let plan = b.finish(&[output]).unwrap();
        let work = plan.work_budget();
        assert!((1..=puma::MAX_OPERATIONS).contains(&work.operations()));
        assert!((1..=puma::prss::MAX_AES_BLOCKS).contains(&work.cursors().aes_work));
        assert!(full_bytes <= puma::MAX_ADMITTED_BYTES);
        let primitive_floor = |lanes| puma::Limits::default().admitted_bytes(lanes).unwrap();
        // Both FULL exact and FULL one-short must be representable. No padding
        // to the primitive floor is allowed to masquerade as native exactness.
        assert!(
            full_bytes > primitive_floor(native_lanes),
            "silu={silu}, full={full_bytes}, tail={tail_bytes}, floor={}",
            primitive_floor(native_lanes)
        );
        let limits = |lanes, bytes| {
            PrssLimits::new(
                puma::Limits::new(lanes, work.operations(), bytes)
                    .expect("priced configuration must not unexpectedly reject"),
                work.cursors().aes_work,
            )
            .unwrap()
        };
        // Exact FULL shared reservation runs the real full + one-lane tail;
        // only guarded SiLU needs the twice-lane configuration.
        let exact = limits(native_lanes, full_bytes);
        assert_eq!(execute(&plan, exact), execute(&reference, exact));
        assert_eq!(
            plan.resources().primitive_ledger_allowance_bytes,
            2 * puma::MAX_ADMITTED_BYTES
        );
        let mut failing = vec![
            (native_lanes, full_bytes - 1),
            // Immutable limits cannot be lowered after FULL. Depending on its
            // actual quote this loses at configuration or earlier FULL entry,
            // never at the tail's native admission boundary.
            (native_lanes, tail_bytes - 1),
        ];
        if silu {
            // Payload envelope is exact; this failure is the real 2*lane guard
            // construction check, not a misleading byte-boundary assertion.
            failing.push((width, full_bytes));
        }
        for (lanes, bytes) in failing {
            let floor = primitive_floor(lanes);
            if bytes < floor {
                // Only this explicitly named tail candidate is allowed to be
                // unrepresentable; do not catch/skip arbitrary Bounds errors.
                assert_eq!((lanes, bytes), (native_lanes, tail_bytes - 1));
                assert_eq!(
                    puma::Limits::new(lanes, work.operations(), bytes),
                    Err(Error::Bounds),
                    "silu={silu}, tail-short={bytes}, configuration floor={floor}"
                );
                assert!(puma::Limits::new(lanes, work.operations(), floor).is_ok());
                continue; // No runtime exists: not a native rejection claim.
            }
            let context = puma::Context::new([71; 32], [92; 32], 0).unwrap();
            let runtimes = puma::prss::insecure_test_support::fresh_compound_parties(
                context,
                limits(lanes, bytes),
            )
            .unwrap();
            let usage = runtimes
                .each_ref()
                .map(multiply::PartyRuntime::payload_usage);
            let mut roles = runtimes.map(|runtime| {
                let value = secret(runtime.party(), input.len());
                Executor::new(plan.clone(), runtime, vec![value], None, None).unwrap()
            });
            assert_eq!(roles[0].advance().err(), Some(Error::Bounds));
            assert_eq!(roles[0].advance().err(), Some(Error::Retired));
            // The coordinator, not a new runtime authority, aborts siblings.
            for role in &mut roles {
                role.cancel();
                assert_eq!(role.live_bytes(), 0);
            }
            drop(roles);
            assert!(usage.iter().all(|usage| usage.live_bytes() == 0));
        }
    }
}

#[test]
fn prepared_map_bound_native_wrong_role_and_previous_full_replay_abort_siblings() {
    for replay in [false, true] {
        let (mut b, input) = builder(256);
        let prepared = b.prepare_rescale_map(&input, 18, 0).unwrap();
        assert_eq!(prepared.programs().count(), 1);
        let original = Arc::downgrade(prepared.program_at(0).unwrap());
        let output = b.prepared_tensor_map(&input, prepared, 0).unwrap();
        let plan = b.finish(&[output]).unwrap();
        let retained_plan = Arc::downgrade(&plan);
        let context = puma::Context::new([71; 32], [92; 32], 0).unwrap();
        let runtimes = puma::prss::insecure_test_support::fresh_compound_parties(
            context,
            plan.work_budget().limits().unwrap(),
        )
        .unwrap();
        let usage = runtimes
            .each_ref()
            .map(multiply::PartyRuntime::payload_usage);
        let mut roles = runtimes.map(|runtime| {
            let value = secret(runtime.party(), input.len());
            Executor::new(plan.clone(), runtime, vec![value], None, None).unwrap()
        });
        let start = |roles: &mut [Executor; 3]| {
            let mut frames = Vec::new();
            for (i, role) in roles.iter_mut().enumerate() {
                frames.extend(
                    role.advance()
                        .unwrap()
                        .into_iter()
                        .map(|(to, frame)| (PARTIES[i], to, frame)),
                );
            }
            frames
        };
        let first = start(&mut roles);
        assert!(!first.is_empty());
        let (from, to, frame) = &first[0];
        let (from, to, saved) = (*from, *to, frame.as_bytes().to_vec());
        let retained = if replay {
            // Drain the genuine FIRST full native invocation, then enter the
            // next full using the SAME original Arc/digest and native step.
            drain_frames(&mut roles, first);
            let second = start(&mut roles);
            // The public pivot may change channel directions. Outer native
            // program/invocation/step binding is shared by every party.
            let current = second.first().unwrap().2.as_bytes();
            assert_eq!(&saved[..36], &current[..36]);
            assert_eq!(&saved[44..48], &current[44..48]);
            assert_ne!(&saved[36..44], &current[36..44]); // Genuine invocation changed.
            assert_eq!(
                roles[to as usize].receive(&saved, from).err(),
                Some(Error::Binding)
            );
            second
        } else {
            // Unmodified, correctly bound native frame; ONLY authenticated
            // channel peer identity is wrong. No fabricated header or binding.
            assert_eq!(
                roles[to as usize].receive(&saved, from.next()).err(),
                Some(Error::Role)
            );
            first
        };
        assert_eq!(roles[to as usize].advance().err(), Some(Error::Retired));
        for role in &mut roles {
            role.cancel();
            assert_eq!(role.advance().err(), Some(Error::Retired));
            assert_eq!(role.live_bytes(), 0);
        }
        assert_eq!(original.strong_count(), 1); // Original Plan only, no active clone.
        drop((roles, plan));
        assert!(retained_plan.upgrade().is_none());
        assert!(original.upgrade().is_none());
        // Real queued encoded frames retain their OWN ledger leases after abort.
        assert!(usage.iter().any(|usage| usage.live_bytes() > 0));
        drop(retained);
        assert!(usage.iter().all(|usage| usage.live_bytes() == 0));
    }
}
