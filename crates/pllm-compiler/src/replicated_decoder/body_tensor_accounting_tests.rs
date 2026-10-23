//! Named adapter control slots and genuine-owner finalization boundaries.
//! These are logical owner/result-slot checks, not an ABI stack or RSS bound.
use super::*;

fn fixture(lanes: usize) -> (Arc<Program>, PreparedTensorMap, usize, usize) {
    super::super::program::prepared_rescale18_worker_fixture(
        lanes,
        Interval::new(-super::super::ONE, super::super::ONE).unwrap(),
    )
    .unwrap()
}

#[test]
fn named_control_slots_fit_already_priced_allowances_without_new_reserve() {
    // Large owning destinations are explicitly present in retained_baseline,
    // stage_slots, and Plan::resources. These lists cover adapter-owned controls,
    // including the handoff copy/export controls, NOT recursive native-helper
    // internals or ABI spill slots. Moves never duplicate an owning payload.
    let preparation = [
        ("binding", size_of::<Binding>()),
        ("binding result", size_of::<Result<Binding, Error>>()),
        ("phase census", size_of::<Phases>()),
        ("phase result", size_of::<Result<Phases, Error>>()),
        ("program argument", size_of::<Arc<Program>>()),
        ("plan argument", size_of::<Arc<Plan>>()),
        ("input declarations", size_of::<[body::Register; 1]>()),
        ("input layout", size_of::<Layout>()),
        ("output layout", size_of::<Layout>()),
        ("input layout result", size_of::<Result<Layout, Error>>()),
        ("output layout result", size_of::<Result<Layout, Error>>()),
        ("layout quote", size_of::<layer::LayoutQuote>()),
        (
            "layout quote result",
            size_of::<Result<layer::LayoutQuote, Error>>(),
        ),
        ("plan resource census", size_of::<body::Resources>()),
        ("checked output layout", size_of::<Result<&Layout, Error>>()),
        ("checked output row", size_of::<Option<layer::RowKey>>()),
        ("baseline/limit/quote", size_of::<[usize; 3]>()),
    ];
    let execution = [
        ("compiled argument", size_of::<Arc<Compiled>>()),
        ("input ids", size_of::<[usize; 1]>()),
        ("source contract arguments", size_of::<[usize; 3]>()),
        (
            "executor result discriminant/excess",
            size_of::<Result<Executor, Error>>().saturating_sub(size_of::<Executor>()),
        ),
        (
            "handoff copy secret result",
            size_of::<Result<Secret, Error>>(),
        ),
        (
            "advance/receive outgoing result",
            size_of::<Result<Vec<(puma::Party, PeerBytes)>, Error>>(),
        ),
        (
            "advance boundary result",
            size_of::<Result<BoundaryProgress, Error>>(),
        ),
    ];
    let restoration = [
        ("output ids", size_of::<[usize; 1]>()),
        ("export closure", size_of::<&Values>()),
        ("exported secret control", size_of::<Secret>()),
        ("exported secret result", size_of::<Result<Secret, Error>>()),
        (
            "exported word result",
            size_of::<Result<(u64, u64), Error>>(),
        ),
        ("export length result", size_of::<Result<usize, Error>>()),
        ("export iterator/index/word", size_of::<[usize; 5]>()),
    ];
    let preparation_bytes: usize = preparation.into_iter().map(|(_, bytes)| bytes).sum();
    // No Invocation exists during metadata preparation. Its single explicitly
    // retained-baseline-priced destination covers these preparation controls;
    // do not borrow a later execution-only allowance before it is admitted.
    assert!(preparation_bytes <= size_of::<Invocation>());
    let execution_bytes: usize = execution
        .into_iter()
        .chain(restoration)
        .map(|(_, bytes)| bytes)
        .sum();
    for lanes in [1, 128, 129, 257] {
        let (program, prepared, input, output) = fixture(lanes);
        let binding = Binding::validate(
            &program.nodes,
            program.inputs,
            WeightMode::Public,
            input,
            output,
            &prepared,
        )
        .unwrap();
        let baseline = retained_baseline(&program, &prepared).unwrap();
        let (builder, layout) = stage(binding, prepared, baseline).unwrap();
        let (plan, _) =
            finish_stage(builder, layout, baseline, super::super::MAX_PROGRAM_BYTES).unwrap();
        // retained_baseline prices Worker AND Invocation AND Result<Invocation>.
        // There is only ONE original Worker: it moves from begin's argument to
        // SuspendedWorker to Invocation.worker, then out at restore. Its control
        // fits the Invocation destination throughout. The separately charged
        // Worker slot therefore covers these small controls, even when the
        // Invocation result destination is also occupied. Never borrow ANY of
        // Plan's 65536/128/container allowance: existing users keep all of it.
        assert!(size_of::<Worker>() <= size_of::<Invocation>());
        assert!(execution_bytes <= size_of::<Worker>());
        // Values and Executor remain independently charged in the native plan;
        // finish's runtime/Values tuple is separately added by Phases::new.
        assert!(
            plan.resources().owner_and_arc_allowance_bytes
                >= size_of::<Executor>() + size_of::<Values>()
        );
        // Constructor tuples and output-array slots have their own explicit
        // quote, independent of the execution allowance tested above.
        assert!(stage_slots().unwrap() >= size_of::<Result<(Builder, Layout), Error>>());
        assert!(stage_slots().unwrap() >= size_of::<[Layout; 1]>());
    }
}

#[test]
fn genuine_retained_owner_baseline_is_added_before_finalizer_exact_and_one_short() {
    for lanes in [1, 128, 129, 257] {
        for one_short in [false, true] {
            let (program, prepared, input, output) = fixture(lanes);
            let binding = Binding::validate(
                &program.nodes,
                program.inputs,
                WeightMode::Public,
                input,
                output,
                &prepared,
            )
            .unwrap();
            // Original Program, capacity-bearing graph/retirement containers,
            // share slots and prepared owners remain independently live.
            let original_owners = program.payload_bytes()
                + size_of::<Program>()
                + size_of::<Node>() * program.nodes.capacity()
                + size_of::<Vec<usize>>() * program.retire.capacity()
                + program
                    .retire
                    .iter()
                    .map(|r| size_of::<usize>() * r.capacity())
                    .sum::<usize>()
                + size_of::<(usize, usize)>() * program.outputs.capacity()
                + size_of::<Option<Secret>>() * program.nodes.len()
                + prepared.baseline_bytes().unwrap();
            let baseline = retained_baseline(&program, &prepared).unwrap();
            let transition_excess = size_of::<
                Result<
                    (
                        SuspendedWorker,
                        super::super::multiply::PartyRuntime,
                        Vec<Secret>,
                    ),
                    Error,
                >,
            >()
            .saturating_sub(size_of::<Worker>())
                + size_of::<Result<Worker, Error>>().saturating_sub(size_of::<Worker>());
            let original_controls = size_of::<Worker>()
                + size_of::<Compiled>()
                + size_of::<Result<Arc<Compiled>, PreparationError>>()
                + size_of::<Invocation>()
                + size_of::<Result<Invocation, Error>>()
                + transition_excess
                + 3 * (2 * size_of::<usize>()); // Program, Plan, Compiled Arc headers.
            assert_eq!(baseline, original_owners + original_controls);
            let weak_program = Arc::downgrade(&program);
            let full = Arc::downgrade(prepared.program_at(0).unwrap());
            let (builder, layout) = stage(binding, prepared, baseline).unwrap();
            let construction = builder
                .proof_finish_quote(std::slice::from_ref(&layout), stage_slots().unwrap())
                .unwrap();
            let exact = baseline.checked_add(construction).unwrap();
            assert!(exact <= super::super::MAX_PROGRAM_BYTES);
            let before = FINISH_ENTRIES.with(std::cell::Cell::get);
            let result = finish_stage(builder, layout, baseline, exact - usize::from(one_short));
            if one_short {
                assert!(matches!(result, Err(Error::Bounds)));
                assert_eq!(FINISH_ENTRIES.with(std::cell::Cell::get), before);
            } else {
                let (plan, quoted) = result.unwrap();
                assert_eq!(quoted, construction);
                assert_eq!(FINISH_ENTRIES.with(std::cell::Cell::get), before + 1);
                drop(plan);
            }
            assert_eq!(weak_program.strong_count(), 1);
            assert_eq!(full.strong_count(), 1);
            drop(program);
            assert!(weak_program.upgrade().is_none());
            assert!(full.upgrade().is_none());
        }
    }
}
