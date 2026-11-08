use super::*;
use pllm_garble::fusefss_backend::composition::{self, cache};

fn builder() -> import::SourceBuilder {
    let config = pllm_types::canonical_bytes(&serde_json::json!({
        "model_type":"gpt2", "n_embd":2, "n_head":1, "n_layer":1,
        "n_positions":4, "vocab_size":3, "n_inner":2
    }));
    import::SourceBuilder::new(&config, 1, 2, sdk::GPT2_NUMERIC).unwrap()
}

#[test]
fn backend_failure_is_inline_and_preserves_seal_identity() {
    assert!(std::mem::size_of::<AdmissionFailure<Error>>() <= 128);
    assert!(!std::mem::needs_drop::<AdmissionFailure<Error>>());
    let old = builder().seal_with_schedule(OnlineSchedule::TensorV1).err().unwrap();
    let contextual = builder().seal_with_schedule_context(OnlineSchedule::TensorV1).err().unwrap();
    assert_eq!(old, Error::Context);
    assert_eq!(contextual.error, old);
    let context = contextual.context.unwrap();
    assert_eq!(context.stage, AdmissionStage::SourceSeal);
    assert_eq!(context.gate, AdmissionGate::SourceAssembly);
    assert_eq!((context.required, context.used, context.limit), (None, None, None));

    let mut retired = builder();
    retired.cancel();
    let failure = retired.seal_with_schedule_context(OnlineSchedule::TensorV1).err().unwrap();
    assert_eq!(failure.error, Error::Consumed);
}

#[test]
fn compile_compatibility_uses_same_early_error_and_consumes_source() {
    fn imported() -> import::ImportedSource {
        let mut owner = builder();
        for (name, shape) in owner.admitted().unwrap().source_shapes.clone() {
            let count = shape.iter().product::<usize>();
            owner.append(&name, WeightDtype::F32, 0, &vec![0; count * 4]).unwrap();
        }
        owner.seal_source().unwrap()
    }
    let old = imported();
    let old = Program::compile_with_schedule(&old.plan, old.source, 0, Limits::default(), OnlineSchedule::TensorV1)
        .err().unwrap();
    let contextual = imported();
    let contextual = Program::compile_with_schedule_context(&contextual.plan, contextual.source, 0,
        Limits::default(), OnlineSchedule::TensorV1).err().unwrap();
    assert_eq!(old, Error::Unsupported);
    assert_eq!(contextual.error, old);
    assert_eq!(contextual.context.unwrap().gate, AdmissionGate::ProgramConstruction);
}

#[test]
fn quote_exact_one_short_preserves_counters_and_comparison_precedence() {
    fn finish(resources: &mut Resources, limits: Limits, contextual: bool) -> Result<(), AdmissionFailure<Error>> {
        let quote = resource_quote::Quote::begin(resources, limits, Capacity::Capsule,
            composition::Resources::default(), None, 0, false)?;
        if contextual { quote.finish_context(cache::Stats::default()) }
        else { quote.finish(cache::Stats::default()).map_err(AdmissionFailure::from) }
    }
    for short in [false, true] {
        let limits = Limits { max_work: 7 - usize::from(short), max_material_bytes: 0, ..Limits::default() };
        let seed = Resources { work: 7, material_bytes: usize::from(short), ..Resources::default() };
        let mut old = seed;
        let mut contextual = seed;
        let old_result = finish(&mut old, limits, false);
        let context_result = finish(&mut contextual, limits, true);
        assert_eq!(old_result.map_err(|failure| failure.error), context_result.map_err(|failure| failure.error));
        // All scalar counters, including non-refundable increments, stay equal.
        assert_eq!(format!("{old:?}"), format!("{contextual:?}"));
        assert_eq!(contextual.kernels, 1);
        assert_eq!(contextual.work, 7);
        if short {
            let failure = context_result.unwrap_err();
            assert_eq!(failure.error, Error::Resource);
            let context = failure.context.unwrap();
            assert_eq!(context.gate, AdmissionGate::Work); // Before material refusal.
            assert_eq!((context.required, context.used, context.limit), (Some(7), None, Some(6)));
        } else { assert!(context_result.is_ok()); }
    }
}

// Existing tiny importer: real decoded tensor payloads and the normal seal path
// (Source::new_with_capacity), not geometry-only admission or mock certificates.
fn diagnostic_imported() -> import::ImportedSource {
    let mut owner = builder();
    for (name, shape) in owner.admitted().unwrap().source_shapes.clone() {
        let count = shape.iter().product::<usize>();
        owner.append(&name, WeightDtype::F32, 0, &vec![0; count * 4]).unwrap();
    }
    owner.seal_source().unwrap()
}

#[test]
fn diagnostic_material_only_walk_matches_admitted_imported_source() {
    let imported = diagnostic_imported();
    let baseline = Program::compile(&imported.plan, imported.source, 2, Limits::default()).unwrap();
    assert!(baseline.resources().material_bytes > 0);
    let limits = Limits { max_material_bytes: 0, ..Limits::default() };
    let imported = diagnostic_imported();
    let normal = Program::compile_with_schedule_context(&imported.plan, imported.source, 2,
        limits, OnlineSchedule::Serial).err().unwrap();
    assert_eq!(normal.context.unwrap().gate, AdmissionGate::MaterialBytes);
    let imported = diagnostic_imported();
    let diagnostic = Program::diagnose_with_schedule(&imported.plan, imported.source, 2,
        limits, OnlineSchedule::Serial).unwrap();
    assert!(matches!(diagnostic.extent, runtime::DiagnosticExtent::Complete));
    assert_eq!(diagnostic.first_material, normal.context);
    assert!(diagnostic.first_material_selection.is_some());
    assert_eq!(format!("{:?}", diagnostic.resources), format!("{:?}", baseline.resources()));
    assert_eq!(diagnostic.material_wire_bytes, None); // Explicitly unsupported.
    // The descriptive record has no custody, destructor, or executable Program.
    assert!(!std::mem::needs_drop::<runtime::Diagnostic>());
}

#[test]
fn diagnostic_stops_on_same_kernel_frame_limit_with_attempted_prefix() {
    let imported = diagnostic_imported();
    let diagnostic = Program::diagnose_with_schedule(&imported.plan, imported.source, 2,
        Limits { max_material_bytes: 0, max_frame_bytes: 0, ..Limits::default() },
        OnlineSchedule::Serial).unwrap();
    let runtime::DiagnosticExtent::Partial(failure) = diagnostic.extent else { panic!("expected partial walk") };
    assert_eq!(failure.error, Error::Resource);
    assert_eq!(failure.context.unwrap().gate, AdmissionGate::FrameBytes);
    let first = diagnostic.first_material.unwrap();
    assert_eq!(first.kernel, Some(diagnostic.resources.kernels));
    assert_eq!(first.node, Some(diagnostic.node));
    assert_eq!(diagnostic.first_material_selection, Some(diagnostic.selection));
    assert!(diagnostic.resources.material_bytes > 0);
    assert!(diagnostic.resources.frame_bytes > 0);
}

#[test]
fn diagnostic_constructor_retains_hard_limits() {
    let hard = Limits::for_capacity(Capacity::Decoder);
    assert_eq!((hard.max_material_bytes, hard.max_sequence), (32usize << 30, 16));
    for limits in [
        Limits { max_sequence: 17, ..Limits::default() },
        Limits { max_material_bytes: Limits::default().max_material_bytes + 1, ..Limits::default() },
    ] {
        let imported = diagnostic_imported();
        let failure = Program::diagnose_with_schedule(&imported.plan, imported.source, 2,
            limits, OnlineSchedule::Serial).unwrap_err();
        assert_eq!(failure.error, Error::Unsupported);
        assert_eq!(failure.context.unwrap().gate, AdmissionGate::ProgramConstruction);
    }
}

#[test]
fn diagnostic_description_work_policy_cannot_admit_oversized_normal_program() {
    use import::SourceDescriptionWorkPolicy::{Default, MixedWork40};
    let hard = Limits::for_capacity(Capacity::Decoder);
    assert_eq!(hard.max_work, 1usize << 38);
    assert_eq!(composition::MAX_WORK, 1usize << 26); // Independent native Plan ceiling.
    let imported = diagnostic_imported();
    let limits = Limits { max_work: MixedWork40.ceiling(0).unwrap(), ..Limits::default() };
    let failure = Program::compile_with_schedule_context(&imported.plan, imported.source, 2,
        limits, OnlineSchedule::Serial).err().unwrap();
    assert_eq!(failure.error, Error::Unsupported);
    assert_eq!(failure.context.unwrap().gate, AdmissionGate::ProgramConstruction);
    let imported = diagnostic_imported();
    let failure = Program::diagnose_validated(&imported.plan, imported.source, 2,
        limits, OnlineSchedule::Serial, Default, |_| Ok(())).unwrap_err();
    assert_eq!(failure.error, Error::Unsupported);
    let imported = diagnostic_imported();
    let diagnostic = Program::diagnose_validated(&imported.plan, imported.source, 2,
        limits, OnlineSchedule::Serial, MixedWork40, |_| Ok(())).unwrap();
    assert!(matches!(diagnostic.extent, runtime::DiagnosticExtent::Complete));
    for limits in [
        Limits { max_work: limits.max_work + 1, ..limits },
        Limits { max_sequence: 17, ..limits },
        Limits { max_material_bytes: limits.max_material_bytes + 1, ..limits },
        Limits { max_frame_bytes: limits.max_frame_bytes + 1, ..limits },
        Limits { max_public_copy_bytes: limits.max_public_copy_bytes + 1, ..limits },
    ] {
        let imported = diagnostic_imported();
        let failure = Program::diagnose_validated(&imported.plan, imported.source, 2,
            limits, OnlineSchedule::Serial, MixedWork40, |_| Ok(())).unwrap_err();
        assert_eq!(failure.error, Error::Unsupported);
        assert_eq!(failure.context.unwrap().gate, AdmissionGate::ProgramConstruction);
    }
}

#[test]
fn diagnostic_description_work_policy_keeps_finite_same_ledger_and_other_refusals() {
    let ceiling = import::SourceDescriptionWorkPolicy::MixedWork40.ceiling(0).unwrap();
    let limits = Limits { max_work: ceiling, max_material_bytes: 0, ..Limits::default() };
    fn finish(resources: &mut Resources, limits: Limits, native: composition::Resources,
        first: &mut Option<AdmissionContext>) -> Result<(), AdmissionFailure<Error>> {
        resource_quote::Quote::begin(resources, limits, Capacity::Capsule,
            native, None, 0, false)?.finish_observing(cache::Stats::default(), first)
    }
    // Only the comparison ceiling changes. The exact ledger and spent increments
    // are identical at equality and at the first finite work refusal.
    for delta in [0, 1] {
        let mut resources = Resources { work: ceiling - 2 + delta, ..Resources::default() };
        let mut first = None;
        let result = finish(&mut resources, limits, composition::Resources {
            cumulative_work: 1, ..composition::Resources::default()
        }, &mut first);
        assert_eq!(resources.work, ceiling + delta);
        assert_eq!(resources.kernels, 1);
        if delta == 0 { result.unwrap(); } else {
            let context = result.unwrap_err().context.unwrap();
            assert_eq!(context.gate, AdmissionGate::Work);
            assert_eq!((context.required, context.limit), (Some(ceiling + 1), Some(ceiling)));
        }
        assert!(first.is_none());
    }
    let mut resources = Resources::default();
    let mut first = None;
    let failure = finish(&mut resources, Limits { max_frame_bytes: 0, ..limits },
        composition::Resources { private_material_bytes_total: 1, frame_bytes_total: 1,
            ..composition::Resources::default() }, &mut first).unwrap_err();
    assert_eq!(first.unwrap().gate, AdmissionGate::MaterialBytes);
    assert_eq!(failure.context.unwrap().gate, AdmissionGate::FrameBytes);
    let mut resources = Resources { work: usize::MAX, ..Resources::default() };
    let mut first = None;
    assert_eq!(finish(&mut resources, limits, composition::Resources {
        cumulative_work: 1, ..composition::Resources::default()
    }, &mut first).unwrap_err().error, Error::Resource);
    assert_eq!(resources.work, usize::MAX);
    assert!(first.is_none());
}

#[test]
fn diagnostic_quote_checks_native_peak_and_checked_overflow() {
    fn observed(resources: &mut Resources, native: composition::Resources,
        first: &mut Option<AdmissionContext>) -> Result<(), AdmissionFailure<Error>> {
        resource_quote::Quote::begin(resources,
            Limits { max_material_bytes: 0, ..Limits::default() },
            Capacity::Capsule, native, None, 0, false)?
            .finish_observing(cache::Stats::default(), first)
    }
    let native = composition::Resources {
        private_material_bytes_total: 2, combined_reservation_bytes: 1024,
        ..composition::Resources::default()
    };
    let mut resources = Resources::default();
    let mut first = None;
    observed(&mut resources, native, &mut first).unwrap();
    observed(&mut resources, native, &mut first).unwrap();
    assert_eq!(resources.material_bytes, 4);
    assert_eq!(resources.peak_native_bytes, 1024); // MAX, not cumulative addition.
    assert_eq!(first.unwrap().required, Some(2)); // First refusal retained.

    let mut resources = Resources::default();
    let mut first = None;
    let failure = observed(&mut resources, composition::Resources {
        combined_reservation_bytes: pllm_garble::fusefss_backend::MAX_ALLOCATION_BYTES + 1,
        ..native
    }, &mut first).unwrap_err();
    assert_eq!(failure.context.unwrap().gate, AdmissionGate::NativeBytes);
    assert_eq!(first.unwrap().gate, AdmissionGate::MaterialBytes);
    assert_eq!(resources.material_bytes, 2); // Rejected kernel already counted.

    let mut resources = Resources { material_bytes: usize::MAX, ..Resources::default() };
    let mut first = None;
    let failure = observed(&mut resources, native, &mut first).unwrap_err();
    assert_eq!(failure.error, Error::Resource);
    assert_eq!(failure.context.unwrap().gate, AdmissionGate::ResourceAccounting);
    assert_eq!(resources.material_bytes, usize::MAX);
    assert!(first.is_none()); // Arithmetic failure is never waived.

    // Late overflow preserves the earlier material mutation, not a coherent
    // completed-kernel prefix. The failed frame addition leaves its prior value.
    let mut resources = Resources {
        material_bytes: 7, frame_bytes: usize::MAX, ..Resources::default()
    };
    let mut first = None;
    let failure = observed(&mut resources, composition::Resources {
        frame_bytes_total: 1, ..native
    }, &mut first).unwrap_err();
    assert_eq!(failure.error, Error::Resource);
    assert_eq!(failure.context.unwrap().gate, AdmissionGate::ResourceAccounting);
    assert_eq!(resources.material_bytes, 9);
    assert_eq!(resources.frame_bytes, usize::MAX);
    assert!(first.is_none()); // Overflow precedes the material-limit comparison.
}

#[test]
fn diagnostic_frame_observation_preserves_first_origin_and_normal_hard_limit() {
    use import::SourceDescriptionWorkPolicy::{Default, MixedWork40};
    assert_eq!(Limits::for_capacity(Capacity::Decoder).max_frame_bytes, 8usize << 30);
    let imported = diagnostic_imported();
    let baseline = Program::compile(&imported.plan, imported.source, 2, Limits::default()).unwrap();
    for policy in [Default, MixedWork40] {
        let limits = Limits { max_work: policy.ceiling(Limits::default().max_work).unwrap(),
            max_frame_bytes: 0, ..Limits::default() };
        let imported = diagnostic_imported();
        let hard = Program::diagnose_validated_options(&imported.plan, imported.source, 2,
            limits, OnlineSchedule::Serial, policy, false, |_| Ok(())).unwrap();
        let runtime::DiagnosticExtent::Partial(failure) = hard.extent else { panic!("frame gate must remain hard") };
        let first = failure.context.unwrap();
        assert_eq!(first.gate, AdmissionGate::FrameBytes);
        assert!(hard.first_frame_refusal.is_none());
        assert!(hard.first_frame_selection.is_none());
        let imported = diagnostic_imported();
        let observed = Program::diagnose_validated_options(&imported.plan, imported.source, 2,
            limits, OnlineSchedule::Serial, policy, true, |_| Ok(())).unwrap();
        assert!(matches!(observed.extent, runtime::DiagnosticExtent::Complete));
        assert_eq!(observed.first_frame_refusal, Some(first));
        assert_eq!(observed.first_frame_selection, Some(hard.selection));
        assert_eq!(hard.selection, 0);
        assert_eq!(observed.selection, 1); // Later crossings must not replace origin.
        assert!(observed.resources.frame_bytes > usize::try_from(first.required.unwrap()).unwrap());
        assert_eq!(format!("{:?}", observed.resources), format!("{:?}", baseline.resources()));
        assert!(observed.material_wire_bytes.is_none());
        assert!(!std::mem::needs_drop::<runtime::Diagnostic>());
    }
    // Observation does not enlarge even the constructor's frame ceiling.
    let imported = diagnostic_imported();
    let failure = Program::diagnose_validated_options(&imported.plan, imported.source, 2,
        Limits { max_frame_bytes: Limits::default().max_frame_bytes + 1, ..Limits::default() },
        OnlineSchedule::Serial, Default, true, |_| Ok(())).unwrap_err();
    assert_eq!(failure.error, Error::Unsupported);
    assert_eq!(failure.context.unwrap().gate, AdmissionGate::ProgramConstruction);
}

#[test]
fn diagnostic_frame_observation_keeps_same_kernel_copy_native_and_work_failures() {
    fn finish(resources: &mut Resources, limits: Limits, native: composition::Resources,
        material: &mut Option<AdmissionContext>, frame: &mut Option<AdmissionContext>)
        -> Result<(), AdmissionFailure<Error>> {
        resource_quote::Quote::begin(resources, limits, Capacity::Capsule,
            native, None, 0, false)?.finish_observing_options(cache::Stats::default(), material, Some(frame))
    }
    let limits = Limits { max_material_bytes: 0, max_frame_bytes: 0, ..Limits::default() };
    let native = composition::Resources { private_material_bytes_total: 2, frame_bytes_total: 3,
        ..composition::Resources::default() };
    for gate in [AdmissionGate::PublicCopyBytes, AdmissionGate::NativeBytes] {
        let (limits, native) = if gate == AdmissionGate::PublicCopyBytes {
            (Limits { max_public_copy_bytes: 0, ..limits }, composition::Resources {
                public_table_copy_bytes_total: 1, ..native })
        } else {
            (limits, composition::Resources {
                combined_reservation_bytes: pllm_garble::fusefss_backend::MAX_ALLOCATION_BYTES + 1, ..native })
        };
        let mut resources = Resources::default();
        let (mut material, mut frame) = (None, None);
        let failure = finish(&mut resources, limits, native, &mut material, &mut frame).unwrap_err();
        assert_eq!(failure.context.unwrap().gate, gate);
        assert_eq!(material.unwrap().gate, AdmissionGate::MaterialBytes);
        let frame = frame.unwrap();
        assert_eq!(frame.gate, AdmissionGate::FrameBytes);
        assert_eq!((frame.required, frame.limit), (Some(3), Some(0)));
        assert_eq!((resources.material_bytes, resources.frame_bytes), (2, 3));
    }
    let mut resources = Resources::default();
    let (mut material, mut frame) = (None, None);
    finish(&mut resources, limits, native, &mut material, &mut frame).unwrap();
    let origin = frame;
    finish(&mut resources, limits, native, &mut material, &mut frame).unwrap();
    assert_eq!(frame, origin);
    assert_eq!((resources.material_bytes, resources.frame_bytes), (4, 6));
    let mut resources = Resources { work: limits.max_work + 1, ..Resources::default() };
    let (mut material, mut frame) = (None, None);
    let failure = finish(&mut resources, limits, native, &mut material, &mut frame).unwrap_err();
    assert_eq!(failure.context.unwrap().gate, AdmissionGate::Work);
    assert!(material.is_none() && frame.is_none()); // Comparison precedence retained.
}

#[test]
fn diagnostic_frame_observation_never_waives_counter_overflow() {
    let mut resources = Resources { material_bytes: 7, frame_bytes: usize::MAX, ..Resources::default() };
    let (mut material, mut frame) = (None, None);
    let failure = resource_quote::Quote::begin(&mut resources,
        Limits { max_material_bytes: 0, max_frame_bytes: 0, ..Limits::default() },
        Capacity::Capsule, composition::Resources {
            private_material_bytes_total: 2, frame_bytes_total: 1,
            ..composition::Resources::default()
        }, None, 0, false).unwrap().finish_observing_options(cache::Stats::default(),
            &mut material, Some(&mut frame)).unwrap_err();
    assert_eq!(failure.error, Error::Resource);
    assert_eq!(failure.context.unwrap().gate, AdmissionGate::ResourceAccounting);
    assert_eq!((resources.material_bytes, resources.frame_bytes), (9, usize::MAX));
    assert!(material.is_none() && frame.is_none());
}
