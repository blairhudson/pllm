//! Literal logical adapter controls; not a complete ABI/RSS or request budget.
use super::*;
use crate::replicated_decoder::MAX_PROGRAM_BYTES;

#[test]
fn named_control_slots_fit_already_priced_allowances_without_new_reserve() {
    run_large_stack(named_controls);
}

fn named_controls() {
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
    {
        let manifest = manifest_prepared_fixture::public_manifest().unwrap();
        let (program, prepared, input, output) =
            manifest_prepared_fixture::compile_manifest_rescale18(
                &manifest,
                Q_PROJECTION_SOURCE_KEY,
            )
            .unwrap();
        let roles = imported_workers(&program, &manifest);
        let public = Arc::new(import_public_dispatch_source(&manifest));
        let registry = SourceRegistry::from_manifest(&manifest).unwrap();
        let source = SourceDispatch::new(
            &program,
            &roles[0],
            public,
            &registry,
            manifest_inline_baseline(&manifest),
        )
        .unwrap();
        let binding = Binding::validate(
            &program.nodes,
            program.inputs,
            WeightMode::Public,
            input,
            output,
            &prepared,
        )
        .unwrap();
        let baseline = add(
            retained_program_baseline(&program, &prepared).unwrap(),
            source.baseline_bytes,
        )
        .unwrap();
        let (builder, layout) = stage(binding, prepared, baseline).unwrap();
        let (plan, _) =
            finish_stage(builder, layout, baseline, MAX_PROGRAM_BYTES).unwrap();
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
