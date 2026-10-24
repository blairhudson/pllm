use super::*;

fn full_operations() -> [OperationCount; 126] {
    let mut result = [OperationCount {
        inputs: 1,
        output_rank: 2,
    }; 126];
    for start in [1, 65] {
        for stage in 0..10 {
            result[start + stage * 6 + 4].inputs = 2;
            result[start + stage * 6 + 5].inputs = 2;
        }
    }
    result[62].inputs = 2;
    result
}
fn backing_inventory(m: &PublicCiphertextConstructionManifest<'_>) -> Vec<Backing> {
    let mut result = Vec::new();
    inventory(m, |element, length, _cap| {
        let n = length.unwrap_or(8);
        result.push(Backing {
            element,
            length: n,
            capacity: Some(n),
            first_phase: 0,
            end_phase: m.phases,
        });
        Ok(())
    })
    .unwrap();
    result
}

#[test]
fn full_count_quote_missing_and_short_ambient_before_factories() {
    let operations = full_operations();
    let unary = [2];
    let binary = [2, 2];
    let capabilities = operations
        .iter()
        .map(|o| CapabilityCount {
            input_ranks: if o.inputs == 1 {
                &unary[..]
            } else {
                &binary[..]
            },
            output_rank: 2,
            execution: Execution::Ckks,
        })
        .collect::<Vec<_>>();
    let mut releases = [1usize; 126];
    releases[125] = 0;
    let owners = [
        OtherOwner {
            bytes: Some(262144),
            metadata: false,
            first_phase: 0,
            end_phase: 2,
        },
        OtherOwner {
            bytes: Some(5767168),
            metadata: false,
            first_phase: 0,
            end_phase: 2,
        },
        OtherOwner {
            bytes: Some(1572864),
            metadata: false,
            first_phase: 0,
            end_phase: 2,
        },
        OtherOwner {
            bytes: Some(25600 + 65536),
            metadata: false,
            first_phase: 0,
            end_phase: 2,
        },
    ];
    let mut manifest = PublicCiphertextConstructionManifest {
        owners: 149,
        persistent_owners: 22,
        input_ranks: &[2],
        operations: &operations,
        capabilities: &capabilities,
        outputs: 1,
        retained_inputs: 1,
        release_lengths: &releases,
        backings: &[],
        other_owners: &owners,
        phases: 2,
        manifest_original_bytes: Some(1 << 20),
    };
    let backings = backing_inventory(&manifest);
    manifest.backings = &backings;
    assert!(backings.len() < 8192);
    let mut work = WorkLedger::new(1 << 28, 1 << 20);
    let mut control = ControlLedger::new(&mut work);
    let quote = quote_public_ciphertext_construction(&manifest, &mut control).unwrap();
    assert_eq!(
        (
            quote.records,
            quote.trace_work,
            quote.validator_payload_bytes
        ),
        (7774, 30_068_627_962_880, 1_991_168)
    );
    assert!(quote.owner_peak < 64 << 20);
    assert!(quote.metadata_peak < 8 << 20);
    // These finite ambient limits predate quote and are deliberately too small.
    // No grant is ever initialized from the requirement or local ceiling.
    let ambient = PublicCiphertextTraceCapacity {
        max_records: 8000,
        max_work: 1 << 20,
        max_buffer_bytes: 4 << 20,
        max_live_owner_bytes: 32 << 20,
    };
    let mut ledger = MetadataLedger::new(1 << 20);
    let mut factory_entries = 0;
    let refused = quote.admit_trace(None, &mut ledger).err();
    if refused.is_none() {
        factory_entries += 1;
    }
    assert_eq!(refused, Some(Refusal::MissingAmbientTraceAdmission));
    let refused = quote.admit_trace(Some(ambient), &mut ledger).err();
    if refused.is_none() {
        factory_entries += 1;
    }
    assert_eq!(refused, Some(Refusal::MetadataWorkRefused));
    assert_eq!(factory_entries, 0);
    assert_eq!(ledger.remaining(), 1 << 20);
}

#[test]
fn nested_original_capacity_bounds_overlap_and_control_refuse() {
    let operations = [OperationCount {
        inputs: 1,
        output_rank: 2,
    }];
    let caps = [CapabilityCount {
        input_ranks: &[2],
        output_rank: 2,
        execution: Execution::Ckks,
    }];
    let mut m = PublicCiphertextConstructionManifest {
        owners: 2,
        persistent_owners: 0,
        input_ranks: &[2],
        operations: &operations,
        capabilities: &caps,
        outputs: 1,
        retained_inputs: 1,
        release_lengths: &[0],
        backings: &[],
        other_owners: &[],
        phases: 2,
        manifest_original_bytes: Some(65536),
    };
    let original = backing_inventory(&m);
    let mut work = WorkLedger::new(1 << 28, 1 << 20);
    let mut control = ControlLedger::new(&mut work);
    let mut b = original.clone();
    b[0].capacity = None;
    m.backings = &b;
    assert!(matches!(
        quote_public_ciphertext_construction(&m, &mut control),
        Err(Refusal::UnknownOriginalCapacity)
    ));
    let mut b = original.clone();
    b[0].capacity = Some(65);
    m.backings = &b;
    assert!(matches!(
        quote_public_ciphertext_construction(&m, &mut control),
        Err(Refusal::IncompleteConstructionManifest)
    ));
    m.backings = &original[..original.len() - 1];
    assert!(matches!(
        quote_public_ciphertext_construction(&m, &mut control),
        Err(Refusal::IncompleteConstructionManifest)
    ));
    m.backings = &original;
    let overlap = [
        OtherOwner {
            bytes: Some(40 << 20),
            metadata: false,
            first_phase: 0,
            end_phase: 2,
        },
        OtherOwner {
            bytes: Some(40 << 20),
            metadata: false,
            first_phase: 1,
            end_phase: 2,
        },
    ];
    m.other_owners = &overlap;
    assert!(matches!(
        quote_public_ciphertext_construction(&m, &mut control),
        Err(Refusal::MetadataStorageRefused)
    ));
    let phased = [
        OtherOwner {
            bytes: Some(40 << 20),
            metadata: false,
            first_phase: 0,
            end_phase: 1,
        },
        overlap[1],
    ];
    m.other_owners = &phased;
    assert!(quote_public_ciphertext_construction(&m, &mut control).is_ok());
    let mut exhausted_work = WorkLedger::new(1 << 28, 255);
    let mut exhausted = ControlLedger::new(&mut exhausted_work);
    assert!(matches!(
        quote_public_ciphertext_construction(&m, &mut exhausted),
        Err(Refusal::ControlWorkExhausted)
    ));
}

#[test]
fn shared_formula_exact_legacy_bound_and_cumulative_no_refund() {
    assert_eq!(
        trace_resource_bound(7774),
        Some((30_068_627_962_880, 1_991_168))
    );
    let operations = [OperationCount {
        inputs: 1,
        output_rank: 2,
    }];
    let caps = [CapabilityCount {
        input_ranks: &[2],
        output_rank: 2,
        execution: Execution::Ckks,
    }];
    let mut m = PublicCiphertextConstructionManifest {
        owners: 2,
        persistent_owners: 0,
        input_ranks: &[2],
        operations: &operations,
        capabilities: &caps,
        outputs: 1,
        retained_inputs: 1,
        release_lengths: &[0],
        backings: &[],
        other_owners: &[],
        phases: 1,
        manifest_original_bytes: Some(65536),
    };
    let b = backing_inventory(&m);
    m.backings = &b;
    let mut work = WorkLedger::new(1 << 28, 1 << 20);
    let mut control = ControlLedger::new(&mut work);
    let q = quote_public_ciphertext_construction(&m, &mut control).unwrap();
    // Independent finite allowance for small consistency cases, not full N-N.
    let ambient = PublicCiphertextTraceCapacity {
        max_records: 128,
        max_work: 100_000_000,
        max_buffer_bytes: 65536,
        max_live_owner_bytes: 1 << 20,
    };
    let mut ledger = MetadataLedger::new(100_000_000);
    q.admit_trace(Some(ambient), &mut ledger).unwrap();
    assert_eq!(ledger.remaining(), 100_000_000 - q.trace_work);
    let mut short = MetadataLedger::new(1);
    assert_eq!(
        q.admit_trace(Some(ambient), &mut short).err(),
        Some(Refusal::MetadataWorkRefused)
    );
}

#[test]
fn constructed_synthetic_only_uses_ordinary_ssa_and_rejects_unknown_error() {
    // Explicitly synthetic carrier/cost/backing assumptions, NOT real CKKS rows.
    let state = PublicCiphertextState {
        scheme: "synth000".into(),
        key_domain: "synthkey".into(),
        layout: "layout00".into(),
        scale: "scale000".into(),
        level: 0,
        shape: vec![32, 1024],
        error: PublicCiphertextError {
            model_identity: "syntherr".into(),
            bound: Some(1),
        },
    };
    let value = |id: &str, owner: &str| PublicCiphertextValue {
        identity: id.into(),
        state: state.clone(),
        owner: owner.into(),
    };
    let cost = PublicCost {
        identity: "cost0000".into(),
        units: Some(1),
    };
    let factory = || {
        let g = PublicCiphertextGraph {
            cost_model_identity: "synthmdl".into(),
            cost_unit: "synthunt".into(),
            maximum_level: 0,
            owners: vec![
                PublicCiphertextOwner {
                    identity: "owner000".into(),
                    bytes: Some(256),
                },
                PublicCiphertextOwner {
                    identity: "owner001".into(),
                    bytes: Some(256),
                },
            ],
            persistent_owners: vec![],
            inputs: vec![value("input000", "owner000")],
            operations: vec![PublicCiphertextGraphOperation {
                identity: "step0000".into(),
                kernel_identity: "square00".into(),
                primitive: PublicPrimitive::Arithmetic,
                inputs: vec!["input000".into()],
                output: value("output00", "owner001"),
            }],
            capabilities: vec![PublicCiphertextCapability {
                identity: "caps0000".into(),
                kernel_identity: "square00".into(),
                primitive: PublicPrimitive::Arithmetic,
                inputs: vec![state.clone()],
                output: state.clone(),
                join: PublicCiphertextJoin::Unary,
                execution: PublicCiphertextExecution::Ckks {
                    compute: PublicLevelCompute {
                        input_level: 0,
                        cost: cost.clone(),
                    },
                    depth: 0,
                    transition: PublicLevelTransition {
                        kind: PublicLevelTransitionKind::Preserve,
                        from_level: 0,
                        to_level: 0,
                        cost: PublicCost {
                            identity: "cost0000".into(),
                            units: Some(0),
                        },
                    },
                },
                error_assumption: Some("assume00".into()),
                scratch_bytes: Some(0),
                recovery_overlap_bytes: Some(0),
            }],
            outputs: vec!["output00".into()],
            retained_inputs: vec!["input000".into()],
        };
        let trace = vec![PublicCiphertextTraceStep {
            operation: "step0000".into(),
            capability: "caps0000".into(),
            release: vec![],
        }];
        (g, trace)
    };
    let operations = [OperationCount {
        inputs: 1,
        output_rank: 2,
    }];
    let caps = [CapabilityCount {
        input_ranks: &[2],
        output_rank: 2,
        execution: Execution::Ckks,
    }];
    let mut m = PublicCiphertextConstructionManifest {
        owners: 2,
        persistent_owners: 0,
        input_ranks: &[2],
        operations: &operations,
        capabilities: &caps,
        outputs: 1,
        retained_inputs: 1,
        release_lengths: &[0],
        backings: &[],
        other_owners: &[],
        phases: 1,
        manifest_original_bytes: Some(65536),
    };
    let b = backing_inventory(&m);
    m.backings = &b;
    let mut work = WorkLedger::new(1 << 28, 1 << 20);
    let mut control = ControlLedger::new(&mut work);
    let q = quote_public_ciphertext_construction(&m, &mut control).unwrap();
    let ambient = PublicCiphertextTraceCapacity {
        max_records: 128,
        max_work: 100_000_000,
        max_buffer_bytes: 65536,
        max_live_owner_bytes: 1 << 20,
    };
    // Independent literal one-invocation allowance: 64*72^3+256*72.
    let mut ledger = MetadataLedger::new(23_906_304);
    let admission = q.admit_trace(Some(ambient), &mut ledger).unwrap();
    assert_eq!(admission._ledger.remaining(), 0);
    let (mut g, trace) = factory();
    assert_eq!(q.check_constructed(&g, &trace, &mut control), Ok(()));
    assert_eq!(
        trace_preflight(&g, &trace, ambient).unwrap(),
        (q.trace_work(), q.validator_payload_bytes())
    );
    q.validate_constructed(admission, &g, &trace, &mut control)
        .unwrap();
    assert_eq!(ledger.remaining(), 0);
    assert_eq!(
        q.admit_trace(Some(ambient), &mut ledger).err(),
        Some(Refusal::MetadataWorkRefused)
    );
    let mut ledger = MetadataLedger::new(100_000_000);
    g.capabilities[0].output.error.bound = None;
    let admission = q.admit_trace(Some(ambient), &mut ledger).unwrap();
    assert_eq!(
        q.validate_constructed(admission, &g, &trace, &mut control),
        Err(Refusal::InvalidTrace)
    );
    assert_eq!(ledger.remaining(), 100_000_000 - q.trace_work);
    // Equal manifests/requirements are not the same immutable quote binding.
    let other = quote_public_ciphertext_construction(&m, &mut control).unwrap();
    let admission = q.admit_trace(Some(ambient), &mut ledger).unwrap();
    assert_eq!(
        other.validate_constructed(admission, &g, &trace, &mut control),
        Err(Refusal::LifetimeMismatch)
    );
    assert_eq!(ledger.remaining(), 100_000_000 - 2 * q.trace_work);
    let aborted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _admission = q.admit_trace(Some(ambient), &mut ledger).unwrap();
        panic!("public construction abort");
    }));
    assert!(aborted.is_err());
    assert_eq!(ledger.remaining(), 100_000_000 - 3 * q.trace_work);
    g.inputs[0].state.scheme.push('x');
    let admission = q.admit_trace(Some(ambient), &mut ledger).unwrap();
    assert_eq!(
        q.validate_constructed(admission, &g, &trace, &mut control),
        Err(Refusal::LifetimeMismatch)
    );
    assert_eq!(ledger.remaining(), 100_000_000 - 4 * q.trace_work);
}
