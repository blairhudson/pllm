use super::*;

#[test]
fn shared_phase_geometry_is_not_sdk_or_native_capacity_permission() {
    use super::super::{program::construction_probe, Error, Limits, Program, Source, Tensor, WeightDtype};
    use construction_probe::Boundary;

    let small = preflight_source(&canonical_bytes(&dense()), 2, 3, DENSE_ROPE_NUMERIC).unwrap();
    assert_eq!(small.maximum_sequence, 4);
    let geometry = sequence(2, 3).unwrap();
    assert_eq!(geometry.padded_rows(), 4);
    assert_eq!([geometry.active_rows(0), geometry.active_rows(1), geometry.active_rows(2)],
        [Some(2), Some(3), Some(4)]);
    for (input, outputs, padded) in [(30, 2, 31), (32, 2, 33), (64, 8, 71)] {
        assert_eq!(DecoderPlan::validated_phase_geometry(input, outputs).unwrap().padded_rows(), padded);
        assert!(sequence(input, outputs).is_err());
        for capacity in [Capacity::Capsule, Capacity::Decoder] {
            // Empty config is a later sentinel: native geometry refuses before
            // parsing, graph expansion, or any SourceBuilder payload opening.
            let error = preflight_source_with_capacity(b"", input, outputs,
                DENSE_ROPE_NUMERIC, capacity).err().unwrap();
            let message = if outputs > 4 { "request rows/selections exceed native Limits" }
                else { "complete-prefix sequence exceeds native Limits" };
            assert_eq!(error, reject(message));
        }
    }
    for (input, outputs) in [(0, 2), (2, 0), (usize::MAX, 2), (2, usize::MAX), (16, 2), (1, 5)] {
        assert!(sequence(input, outputs).is_err());
    }
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        let hard = Limits::for_capacity(capacity);
        assert_eq!((hard.max_sequence, hard.max_selections), (16, 4));
        for (input, outputs, max_sequence, expected, entries) in [
            (30, 2, 16, Error::Resource, [0; 4]),
            (32, 2, 16, Error::Resource, [0; 4]),
            (64, 8, 16, Error::Unsupported, [0; 4]),
            (16, 2, 16, Error::Resource, [0; 4]),
            (14, 4, 16, Error::Resource, [0; 4]),
            (1, 5, 16, Error::Unsupported, [0; 4]),
            (u64::MAX, 2, 16, Error::Resource, [0; 4]),
            (2, usize::MAX, 16, Error::Unsupported, [0; 4]),
            (0, 2, 0, Error::Resource, [0; 4]),
            (0, 1, 16, Error::Domain, [0; 4]),
            (2, 0, 16, Error::Unsupported, [0; 4]),
            // Positive scalar-boundary controls stop at validation. These do
            // not claim that edited graph metadata passes semantic validation.
            (16, 1, 16, Error::Unsupported, [1, 0, 0, 0]),
            (13, 4, 16, Error::Unsupported, [1, 0, 0, 0]),
            (2, 3, 16, Error::Unsupported, [1, 0, 0, 0]),
        ] {
            let mut plan = small.plan.clone();
            plan.prefill.query_sequence = input;
            // Deliberately small metadata, no large geometry-sized allocations.
            plan.decode.maximum_key_sequence = 16;
            let source = Source::new_with_capacity(vec![
                Tensor::from_ieee("geometry", &[1], WeightDtype::F32, &[0]).unwrap(),
            ], capacity).unwrap();
            let (result, actual_entries) = construction_probe::scoped(Some(Boundary::Validation), ||
                Program::compile(&plan, source, outputs, Limits { max_sequence, ..hard }));
            assert!(matches!(result, Err(error) if error == expected));
            assert_eq!(actual_entries, entries);
        }
    }
}

#[test]
fn construction_quote_preserves_sdk_preparation_weights_and_source_once() {
    use super::super::{energy, program::Node, resource_quote::Quote, Source, Tensor, WeightDtype};
    for (profile, capacity) in [
        (GPT2_NUMERIC, Capacity::Capsule),
        (DENSE_ROPE_NUMERIC, Capacity::Decoder),
        (DENSE_ROPE_CHECKED_NUMERIC, Capacity::Capsule),
    ] {
        let config = if profile == GPT2_NUMERIC { absolute() } else { dense() };
        let a = preflight_source_with_capacity(&canonical_bytes(&config), 2, 2, profile, capacity).unwrap();
        let proofs = capacity == Capacity::Decoder || profile == DENSE_ROPE_CHECKED_NUMERIC;
        let owned = Quote::plan_owned(&a.plan).unwrap();
        let external = 4096 + 2 * std::mem::size_of::<SourceExecutionBinding>();
        let count = a.plan.prefill.operations.len();
        let proof_bytes = if proofs { energy::reservation_bound(count).unwrap() } else { 0 };
        let expected = owned * 3 + super::super::MAX_BYTES
            + count * (std::mem::size_of::<Node>() + 4 * 8) + proof_bytes
            + MAX_NODES * 512 + MAX_CONFIG_BYTES * 3 + MAX_CONTRACT_BYTES + external
            + if proofs { pllm_garble::fusefss_backend::MAX_ALLOCATION_BYTES / 3 } else { 0 };
        assert_eq!(a.phase_admission.compile_prepare_peak_bound, expected);
        assert_eq!(Quote::preparation_peak(&a.plan, super::super::MAX_BYTES, proofs).unwrap(), expected);
        let source = Source::new_with_capacity(vec![
            Tensor::from_ieee("quote", &[1], WeightDtype::F32, &[0]).unwrap(),
        ], capacity).unwrap();
        // Same pricing, native uses actual custody rather than SDK worst-case
        // source bound. Source appears once, not once again as an extra owner.
        assert_eq!(Quote::preparation_peak(&a.plan, source.reservation_bytes(), proofs).unwrap(),
            expected - super::super::MAX_BYTES + source.reservation_bytes());
        assert_eq!(Quote::preparation_peak(&a.plan, usize::MAX, proofs), Err(super::super::Error::Resource));
    }
}

#[test]
fn construction_quote_cache_edge_overflow_and_cache_lease_release() {
    use super::super::{energy, program::Node, resource_quote::Quote, Error, Source, Tensor, WeightDtype};
    use pllm_garble::fusefss_backend::{composition::cache, Budget};
    let a = preflight_source_with_capacity(&canonical_bytes(&dense()), 2, 2,
        DENSE_ROPE_NUMERIC, Capacity::Decoder).unwrap();
    let source = Source::new_with_capacity(vec![
        Tensor::from_ieee("quote", &[1], WeightDtype::F32, &[0]).unwrap(),
    ], Capacity::Decoder).unwrap();
    let count = a.plan.prefill.operations.len();
    let base = source.reservation_bytes() + energy::reservation_bound(count).unwrap()
        + count * (std::mem::size_of::<Node>() + 2 * std::mem::size_of::<Vec<usize>>() + 10 * 8)
        + (count + a.plan.decode.operations.len()) * 4096;
    let limits = cache::Limits { max_entries: cache::MAX_ENTRIES, max_retained_bytes: cache::MAX_RETAINED_BYTES };
    let cache_bytes = cache::SourceTableCache::bootstrap_descriptor(source.digest(), limits).unwrap()
        .metadata_reservation_bytes;
    let cap = pllm_garble::fusefss_backend::MAX_ALLOCATION_BYTES;
    // Private arithmetic robustness case: these counters are not a claim that
    // native metadata validation accepts >1 MiB strings. Two-byte price steps.
    let strings = (cap / 3 - cache_bytes - base) / 2;
    let admitted = Quote::construction_limit(&a.plan, &source, 0, strings).unwrap();
    assert!(admitted >= cache_bytes && admitted <= cache_bytes + 1);
    let energy_owner = energy::Ledger::new(source.digest(), a.plan.digest().as_str(), count).unwrap();
    assert_eq!(energy_owner.resident_bytes(), energy::reservation_bound(count).unwrap());
    drop(energy_owner);
    assert_eq!(Quote::construction_limit(&a.plan, &source, 0, strings + 1), Err(Error::Resource));
    for (nodes, bytes) in [(usize::MAX, 0), (0, usize::MAX)] {
        assert_eq!(Quote::construction_limit(&a.plan, &source, nodes, bytes), Err(Error::Resource));
    }
    let budget = Budget::new(admitted).unwrap();
    let failed = (|| -> Result<(), Error> {
        let tables = cache::SourceTableCache::new(source.digest(), limits, &budget)?;
        assert_eq!(budget.used_bytes(), cache_bytes);
        drop(tables); // Same explicit retirement as Program, before preflight.
        assert_eq!(budget.used_bytes(), 0);
        let _tables = cache::SourceTableCache::new(source.digest(), limits, &budget)?;
        Err(Error::Domain) // Synthetic cache-scope failure, not Program cancellation.
    })();
    assert_eq!(failed, Err(Error::Domain));
    assert_eq!(budget.used_bytes(), 0);
    assert_eq!(source.reservation_bytes(), base - energy::reservation_bound(count).unwrap()
        - count * (std::mem::size_of::<Node>() + 2 * std::mem::size_of::<Vec<usize>>() + 10 * 8)
        - (count + a.plan.decode.operations.len()) * 4096);
}

#[test]
fn native_construction_refuses_borrowed_owner_peak_before_semantic_validation() {
    use super::super::{Error, Limits, Program, Source, Tensor, WeightDtype};
    use super::super::program::construction_probe;
    let a = preflight_source_with_capacity(&canonical_bytes(&dense()), 2, 2,
        DENSE_ROPE_NUMERIC, Capacity::Decoder).unwrap();
    let mut plan = a.plan;
    // Valid small custody, no checkpoint files. Deliberately invalid schema is
    // semantic-validation sentinel: without early admission this returns Domain.
    plan.schema_version = "invalid".into();
    let source = || Source::new_with_capacity(vec![
        Tensor::from_ieee("quote", &[1], WeightDtype::F32, &[0]).unwrap(),
    ], Capacity::Decoder).unwrap();
    let (result, entries) = construction_probe::scoped(None, ||
        Program::compile(&plan, source(), 2, Limits::for_capacity(Capacity::Decoder)));
    assert!(matches!(result, Err(Error::Domain)));
    assert_eq!(entries, [1, 0, 0, 0]);
    // Each sparse map entry retains the existing SDK 1024-byte owner weight.
    // Recursive metadata nodes/string bytes still fit native lexical caps.
    plan.prefill.operations[0].attributes = Value::Object((0..16000)
        .map(|i| (format!("inert-{i}"), Value::Null)).collect());
    let (result, entries) = construction_probe::scoped(None, ||
        Program::compile(&plan, source(), 2, Limits::for_capacity(Capacity::Decoder)));
    assert!(matches!(result, Err(Error::Resource)));
    assert_eq!(entries, [0; 4]);
}

// Public, validated deserialization preserves the supplied String's capacity.
// No private Digest layout assumptions, invalid digest, or production accessor.
fn digest_with_capacity(identity: &str, capacity: usize) -> pllm_types::Digest {
    let mut text = String::with_capacity(capacity);
    text.push_str(identity);
    let actual = text.capacity();
    let digest = <pllm_types::Digest as serde::Deserialize>::deserialize(
        serde::de::value::StringDeserializer::<serde::de::value::Error>::new(text),
    ).unwrap();
    assert_eq!(digest.backing_capacity_bytes(), actual);
    digest
}

#[test]
fn construction_census_retained_empty_methods_and_digest_backing_count_once() {
    use super::super::{resource_quote::Quote, Error};
    use pllm_models::{owned_backing::checked_array_bytes, AppliedMethod};
    let mut a = preflight_source(&canonical_bytes(&dense()), 2, 2, DENSE_ROPE_NUMERIC).unwrap();
    let base = Quote::plan_owned(&a.plan).unwrap();
    let peak = Quote::preparation_peak(&a.plan, 0, false).unwrap();
    let old_digest = a.plan.config_digest.backing_capacity_bytes();
    let digest = digest_with_capacity(a.plan.config_digest.as_str(), 4096);
    assert_eq!(digest.as_str(), a.plan.config_digest.as_str());
    let digest_delta = digest.backing_capacity_bytes() - old_digest;
    a.plan.config_digest = digest;
    assert!(a.plan.transformations.is_empty());
    let old_methods = checked_array_bytes::<AppliedMethod>(a.plan.transformations.capacity()).unwrap();
    a.plan.transformations.reserve_exact(128);
    let method_delta = checked_array_bytes::<AppliedMethod>(a.plan.transformations.capacity()).unwrap() - old_methods;
    assert!(a.plan.transformations.is_empty());
    assert_eq!(Quote::plan_owned(&a.plan).unwrap(), base + digest_delta + method_delta);
    assert_eq!(Quote::preparation_peak(&a.plan, 0, false).unwrap(), peak + 3 * (digest_delta + method_delta));

    // Existing source-byte arithmetic seam exercises exact global cap and +1,
    // without changing a native cap or manufacturing impossible Vec capacities.
    let cap = pllm_garble::fusefss_backend::MAX_ALLOCATION_BYTES;
    let floor = Quote::preparation_peak(&a.plan, 0, false).unwrap();
    assert_eq!(Quote::preparation_peak(&a.plan, cap - floor, false), Ok(cap));
    assert_eq!(Quote::preparation_peak(&a.plan, cap - floor + 1, false), Ok(cap + 1));
    assert_eq!(Quote::preparation_peak(&a.plan, usize::MAX, false), Err(Error::Resource));
    let method_size = std::mem::size_of::<AppliedMethod>();
    let last_capacity = usize::MAX / method_size;
    assert_eq!(checked_array_bytes::<AppliedMethod>(last_capacity).unwrap(), last_capacity * method_size);
    assert!(checked_array_bytes::<AppliedMethod>(last_capacity + 1).is_err());
    assert!(checked_array_bytes::<AppliedMethod>(usize::MAX).is_err());
}

#[test]
fn native_constructor_boundaries_reject_retained_empty_methods_and_digest_capacity() {
    use super::super::{Error, Limits, Program, Source, Tensor, WeightDtype};
    use super::super::program::construction_probe::{self, Boundary};
    let mut a = preflight_source_with_capacity(&canonical_bytes(&dense()), 2, 2,
        DENSE_ROPE_NUMERIC, Capacity::Decoder).unwrap();
    let source = || Source::new_with_capacity(vec![
        Tensor::from_ieee("quote", &[1], WeightDtype::F32, &[0]).unwrap(),
    ], Capacity::Decoder).unwrap();
    let compile = |plan: &DecoderPlan| Program::compile(plan, source(), 2, Limits::for_capacity(Capacity::Decoder));
    // Each positive control reaches its actual Program call site. Cache sentinel
    // stops before cache allocation / node lowering / full runtime preflight.
    for (stop, expected) in [
        (Boundary::Validation, [1, 0, 0, 0]),
        (Boundary::Vectors, [1, 1, 0, 0]),
        (Boundary::Energy, [1, 1, 1, 0]),
        (Boundary::Cache, [1, 1, 1, 1]),
    ] {
        let (result, entries) = construction_probe::scoped(Some(stop), || compile(&a.plan));
        assert!(matches!(result, Err(Error::Unsupported)));
        assert_eq!(entries, expected);
    }
    // Actual Program failure after all four entries: deliberately incomplete
    // source inventory fails first tensor lookup. Entry observations alone do
    // not assert lease retirement, backing destruction, or runtime cancellation.
    let (result, entries) = construction_probe::scoped(None, || compile(&a.plan));
    assert!(matches!(result, Err(Error::Domain)));
    assert_eq!(entries, [1; 4]);

    // No injectable native memory cap exists. AUTHORED ONLY under execution
    // hold: this case reserves approximately 22 MiB, not a tiny-cap surrogate.
    // Empty retained backing alone exceeds the existing 3*borrowed-plan tariff.
    let cap = pllm_garble::fusefss_backend::MAX_ALLOCATION_BYTES;
    a.plan.transformations.reserve_exact(cap / 3 / std::mem::size_of::<pllm_models::AppliedMethod>() + 1);
    assert!(a.plan.transformations.is_empty());
    let (result, entries) = construction_probe::scoped(None, || compile(&a.plan));
    assert!(matches!(result, Err(Error::Resource)));
    assert_eq!(entries, [0; 4]);
    // Independent nonempty-method rejection remains Unsupported, even with the
    // same over-cap backing, and still enters no allocating constructor.
    a.plan.transformations.push(pllm_models::AppliedMethod {
        component: "unsupported".into(), implementation: "unsupported".into(), method_id: "unsupported".into(),
        input_digest: a.plan.config_digest.clone(), configuration_digest: a.plan.config_digest.clone(),
    });
    let (result, entries) = construction_probe::scoped(None, || compile(&a.plan));
    assert!(matches!(result, Err(Error::Unsupported)));
    assert_eq!(entries, [0; 4]);
    a.plan.transformations = Vec::new(); // Retire the large test backing first.

    // Native three-byte tariff edge: <= cap, then one extra digest-capacity byte
    // exceeds cap. About 14 MiB retained here; do not run under execution hold.
    let identity = a.plan.config_digest.as_str().to_owned();
    let old_capacity = a.plan.config_digest.backing_capacity_bytes();
    let peak = super::super::resource_quote::Quote::preparation_peak(
        &a.plan, source().reservation_bytes(), true,
    ).unwrap();
    let edge_capacity = old_capacity + (cap - peak) / 3;
    a.plan.config_digest = digest_with_capacity(&identity, edge_capacity);
    let edge_peak = super::super::resource_quote::Quote::preparation_peak(
        &a.plan, source().reservation_bytes(), true,
    ).unwrap();
    assert!(edge_peak <= cap && cap - edge_peak < 3);
    let (result, entries) = construction_probe::scoped(Some(Boundary::Validation), || compile(&a.plan));
    assert!(matches!(result, Err(Error::Unsupported)));
    assert_eq!(entries, [1, 0, 0, 0]);
    a.plan.config_digest = digest_with_capacity(&identity, 64); // Retire large backing.
    a.plan.config_digest = digest_with_capacity(&identity, edge_capacity + 1);
    let (result, entries) = construction_probe::scoped(None, || compile(&a.plan));
    assert!(matches!(result, Err(Error::Resource)));
    assert_eq!(entries, [0; 4]);
}

#[test]
fn wavefront_online_schedule_is_checked_policy_and_preserves_numeric_source_contract() {
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        let config = dense();
        let mut experiment = experiment(DENSE_ROPE_NUMERIC);
        if capacity == Capacity::Decoder {
            experiment["pipeline"]["components"]["linear"]["params"]["capacity"] = json!("decoder");
        }
        let serial = contract(&experiment, &config, 3, 2).unwrap();
        let original: Value = serde_json::from_slice(&serial).unwrap();
        assert!(original.get("online_schedule").is_none());
        assert_eq!(
            contract_online_schedule(&serial).unwrap(),
            super::super::OnlineSchedule::Serial
        );
        let admitted = preflight_source_with_capacity(
            &canonical_bytes(&config),
            3,
            2,
            DENSE_ROPE_NUMERIC,
            capacity,
        )
        .unwrap();
        let binding = SourceExecutionBinding::new(&admitted, 2).unwrap();
        experiment["pipeline"]["components"]["linear"]["params"]["online_schedule"] =
            json!("wavefront-v1");
        let wave = contract(&experiment, &config, 3, 2).unwrap();
        assert_ne!(
            binding.checked_contract_digest(&serial).unwrap(),
            binding.checked_contract_digest(&wave).unwrap()
        );
        assert_eq!(
            contract_online_schedule(&wave).unwrap(),
            super::super::OnlineSchedule::WavefrontV1
        );
        let mut selected: Value = serde_json::from_slice(&wave).unwrap();
        for field in [
            "plan",
            "source_shapes",
            "numeric_profile",
            "source_q12_value_bytes",
        ] {
            assert_eq!(selected[field], original[field]);
        }
        for invalid in [
            json!("serial"),
            json!("wavefront-v2"),
            json!(true),
            json!(1),
            Value::Null,
        ] {
            experiment["pipeline"]["components"]["linear"]["params"]["online_schedule"] =
                invalid.clone();
            assert!(contract(&experiment, &config, 3, 2).is_err());
            selected["online_schedule"] = invalid;
            assert!(binding
                .checked_contract_digest(&canonical_bytes(&selected))
                .is_err());
        }
        experiment["pipeline"]["components"]["linear"]["params"]
            .as_object_mut()
            .unwrap()
            .remove("online_schedule");
        assert_eq!(contract(&experiment, &config, 3, 2).unwrap(), serial);
        experiment["pipeline"]["components"]["linear"]["params"]["online_schedule"] =
            json!("tensor-v1");
        let tensor = contract(&experiment, &config, 3, 2).unwrap();
        assert_eq!(
            contract_online_schedule(&tensor).unwrap(),
            super::super::OnlineSchedule::TensorV1
        );
        assert_ne!(
            binding.checked_contract_digest(&tensor).unwrap(),
            binding.checked_contract_digest(&wave).unwrap()
        );
        let tensor: Value = serde_json::from_slice(&tensor).unwrap();
        for field in [
            "plan",
            "source_shapes",
            "numeric_profile",
            "source_q12_value_bytes",
        ] {
            assert_eq!(tensor[field], original[field]);
        }
    }
}

#[test]
fn material_delivery_is_checked_contract_policy_and_default_identities_stay_omitted() {
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        let config = dense();
        let mut experiment = experiment(DENSE_ROPE_NUMERIC);
        if capacity == Capacity::Decoder {
            experiment["pipeline"]["components"]["linear"]["params"]["capacity"] = json!("decoder");
        }
        let single = contract(&experiment, &config, 3, 2).unwrap();
        let admitted = preflight_source_with_capacity(
            &canonical_bytes(&config),
            3,
            2,
            DENSE_ROPE_NUMERIC,
            capacity,
        )
        .unwrap();
        let binding = SourceExecutionBinding::new(&admitted, 2).unwrap();
        let single_digest = binding.checked_contract_digest(&single).unwrap();
        assert_eq!(
            contract_material_delivery(&single).unwrap(),
            MaterialDelivery::Single
        );
        let single_value: Value = serde_json::from_slice(&single).unwrap();
        assert!(single_value.get("material_delivery").is_none());

        experiment["pipeline"]["components"]["linear"]["params"]["material_delivery"] =
            json!("batch-v1");
        let batched = contract(&experiment, &config, 3, 2).unwrap();
        assert_eq!(
            contract_material_delivery(&batched).unwrap(),
            MaterialDelivery::BatchV1
        );
        assert_ne!(
            binding.checked_contract_digest(&batched).unwrap(),
            single_digest
        );
        let mut batch_value: Value = serde_json::from_slice(&batched).unwrap();
        assert_eq!(batch_value["material_delivery"], "batch-v1");
        for key in [
            "plan",
            "source_shapes",
            "numeric_profile",
            "source_q12_value_bytes",
        ] {
            assert_eq!(batch_value[key], single_value[key]);
        }
        // Default omission is canonical. Arbitrary or type-coerced selectors
        // cannot enter either source-contract generation or checked binding.
        for invalid in [
            json!("single"),
            json!("batch-v2"),
            json!(true),
            json!(1),
            Value::Null,
        ] {
            experiment["pipeline"]["components"]["linear"]["params"]["material_delivery"] =
                invalid.clone();
            assert!(contract(&experiment, &config, 3, 2).is_err());
            batch_value["material_delivery"] = invalid;
            assert!(binding
                .checked_contract_digest(&canonical_bytes(&batch_value))
                .is_err());
            assert!(contract_material_delivery(&canonical_bytes(&batch_value)).is_err());
        }
        experiment["pipeline"]["components"]["linear"]["params"]
            .as_object_mut()
            .unwrap()
            .remove("material_delivery");
        assert_eq!(contract(&experiment, &config, 3, 2).unwrap(), single);
    }
}

#[test]
fn explicit_source_capacity_is_bound_and_legacy_contract_has_no_capacity_fields() {
    let config = dense();
    let old = contract(&experiment(DENSE_ROPE_NUMERIC), &config, 3, 2).unwrap();
    let old_record: Value = serde_json::from_slice(&old).unwrap();
    for field in [
        "capacity",
        "source_disk_bytes",
        "source_resident_bound_bytes",
    ] {
        assert!(old_record.get(field).is_none());
    }
    let capsule = preflight(&config, DENSE_ROPE_NUMERIC).unwrap();
    let capsule_binding = SourceExecutionBinding::new(&capsule, 2).unwrap();
    capsule_binding.checked_contract_digest(&old).unwrap();
    let mut experiment = experiment(DENSE_ROPE_NUMERIC);
    experiment["pipeline"]["components"]["linear"]["params"]["capacity"] = json!("decoder");
    let encoded = contract(&experiment, &config, 3, 2).unwrap();
    let decoder = preflight_source_with_capacity(
        &canonical_bytes(&config),
        3,
        2,
        DENSE_ROPE_NUMERIC,
        Capacity::Decoder,
    )
    .unwrap();
    let binding = SourceExecutionBinding::new(&decoder, 2).unwrap();
    binding.checked_contract_digest(&encoded).unwrap();
    assert_ne!(binding.identity, capsule_binding.identity);
    assert!(binding.checked_contract_digest(&old).is_err());
    assert!(capsule_binding.checked_contract_digest(&encoded).is_err());
    let record: Value = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(record["capacity"], "decoder");
    assert_eq!(record["admission"]["numeric"], "not_performed");
    assert_eq!(record["admission"]["serving_qualified"], false);
    for (field, replacement) in [
        ("capacity", json!("capsule")),
        ("source_disk_bytes", json!(0)),
        ("source_resident_bound_bytes", json!(1)),
    ] {
        let mut changed = record.clone();
        changed[field] = replacement;
        assert!(binding
            .checked_contract_digest(&canonical_bytes(&changed))
            .is_err());
    }
}

fn experiment(profile: &str) -> Value {
    json!({"schema":"pllm.experiment.v2","name":"fss-metadata-only",
        "pipeline":{"model":{"source":"generated/public-source"},"components":{
            "linear":{"component":PROTOCOL,"params":{"numeric_profile":profile}},
            "topology":{"component":TOPOLOGY,"params":{}},
            "state":{"component":STATE,"params":{}},
            "kernels":{"component":"pllm/cpu","params":{"threads":1}}}},
        "deployment":{"kind":"local","root":"local://fss-metadata-only"},
        "budget":{"requests":1,"max_input_tokens":3,"max_new_tokens":2}})
}
fn document(value: &Value) -> ExperimentDocument {
    serde_json::from_value(value.clone()).unwrap()
}
fn absolute() -> Value {
    json!({"model_type":"gpt2","n_embd":8,"n_head":2,"n_layer":2,
        "n_positions":16,"vocab_size":17,"tie_word_embeddings":true})
}
fn dense() -> Value {
    json!({"model_type":"qwen2","hidden_size":8,"intermediate_size":16,"num_hidden_layers":2,
        "num_attention_heads":2,"num_key_value_heads":1,"vocab_size":17,
        "max_position_embeddings":32,"hidden_act":"silu","rms_norm_eps":0.000001,
        "rope_theta":10000,"tie_word_embeddings":true})
}
fn preflight(value: &Value, profile: &str) -> Result<SourcePreflight, String> {
    preflight_source(&canonical_bytes(value), 3, 2, profile)
}
fn contract(e: &Value, c: &Value, rows: usize, outputs: usize) -> Result<Vec<u8>, String> {
    source_contract(
        &canonical_bytes(e),
        &canonical_bytes(c),
        rows,
        outputs,
        &"a".repeat(64),
        &"b".repeat(64),
    )
}

#[test]
fn audit_metadata_amplification_and_late_string_configs_reject_before_expansion() {
    for layers in [1024, 36] {
        let malicious = json!({"model_type":"qwen2","hidden_size":2,"intermediate_size":2,
            "num_hidden_layers":layers,"num_attention_heads":1,"num_key_value_heads":1,
            "vocab_size":3,"max_position_embeddings":16,"hidden_act":"silu",
            "rope_theta":10000,"rms_norm_eps":format!("0.000001{}", "0".repeat(15000)),
            "tie_word_embeddings":true});
        let bytes = canonical_bytes(&malicious);
        assert!(bytes.len() <= MAX_CONFIG_BYTES);
        let error = match preflight_source(&bytes, 1, 1, DENSE_ROPE_NUMERIC) {
            Ok(_) => panic!("amplification accepted"),
            Err(e) => e,
        };
        assert!(error.contains("string"), "{error}");
        // The raw 16 KiB config is the only sizeable owner. Never lower these
        // 1024 layers in a regression merely to observe the old rejection.
        assert!(bounded_json_value(&bytes).is_err());
    }
    let mut large = dense();
    large["num_hidden_layers"] = json!(1024);
    let budget = DecoderExpansionBudget {
        max_layers: 1024,
        max_feature_width: 32768,
        max_attention_heads: 32768,
        max_vocabulary: 32768,
        max_tensor_elements: 32768,
    };
    assert!(budget
        .preflight_metadata(
            &large,
            DecoderWorkload {
                batch: 1,
                max_input_tokens: 1,
                max_new_tokens: 1
            },
            MAX_NODES,
            1024,
            32768,
            1 << 20
        )
        .is_err());
}

#[test]
fn bounded_canonical_counting_rejects_before_building_output_and_preserves_identity() {
    let value = json!({"z":[true,1,1.5,"escaped\n"],"a":{"b":null}});
    assert_eq!(
        bounded_canonical(&value, 1024).unwrap(),
        canonical_bytes(&value)
    );
    assert!(bounded_canonical(&value, 4).is_err());
    // No giant output allocation: the counting writer stops at the first limit.
    struct Oversize;
    impl Serialize for Oversize {
        fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            use serde::ser::SerializeSeq;
            let mut seq = s.serialize_seq(None)?;
            for _ in 0..2_000_000 {
                seq.serialize_element(&0u8)?;
            }
            seq.end()
        }
    }
    assert!(bounded_canonical(&Oversize, 1024).is_err());
    let mut large = dense();
    large["num_hidden_layers"] = json!(36);
    large["rms_norm_eps"] = json!(format!("0.000001{}", "0".repeat(1016)));
    assert!(preflight(&large, DENSE_ROPE_NUMERIC).is_err());
}

#[test]
fn immutable_execution_binding_rejects_contract_substitution_not_just_digest_mismatch() {
    for (config, profile) in [(absolute(), GPT2_NUMERIC), (dense(), DENSE_ROPE_NUMERIC)] {
        let admitted = preflight(&config, profile).unwrap();
        let binding = SourceExecutionBinding::new(&admitted, 2).unwrap();
        let bytes = contract(&experiment(profile), &config, 3, 2).unwrap();
        assert_eq!(
            binding.checked_contract_digest(&bytes).unwrap(),
            <[u8; 32]>::from(Sha256::digest(&bytes))
        );
        // Inert metadata leaves semantic lowering unchanged, but exact config
        // bytes are available to this native constructor and must also bind.
        let mut other_config = config.clone();
        other_config["_name_or_path"] = json!("different-inert-source-label");
        let other_contract = contract(&experiment(profile), &other_config, 3, 2).unwrap();
        assert!(binding.checked_contract_digest(&other_contract).is_err());
        assert!(binding.checked_contract_digest(b"x").is_err());
        assert!(binding.checked_contract_digest(b"{}").is_err());
        assert!(binding
            .checked_contract_digest(&vec![b'x'; MAX_CONTRACT_BYTES + 1])
            .is_err());
        assert!(binding
            .checked_contract_digest(
                &serde_json::to_vec_pretty(&serde_json::from_slice::<Value>(&bytes).unwrap())
                    .unwrap()
            )
            .is_err());
        for (path, value) in [
            ("numeric_profile", json!("wrong-profile")),
            ("input_rows", json!(2)),
            ("output_tokens", json!(1)),
            ("width", json!(999)),
            ("vocabulary", json!(999)),
            ("schema_version", json!("other")),
            ("state_policy", json!("kv-cache")),
            ("setup_owner", json!("client")),
            ("ring_bits", json!(64)),
            ("weights", json!("private")),
            ("source_float32_bytes", json!(1)),
        ] {
            let mut changed: Value = serde_json::from_slice(&bytes).unwrap();
            changed[path] = value;
            assert!(
                binding
                    .checked_contract_digest(&canonical_bytes(&changed))
                    .is_err(),
                "{path}"
            );
        }
        let mut changed: Value = serde_json::from_slice(&bytes).unwrap();
        changed["plan"]["config_digest"] = json!("f".repeat(64));
        assert!(binding
            .checked_contract_digest(&canonical_bytes(&changed))
            .is_err());
        let mut inventory: Value = serde_json::from_slice(&bytes).unwrap();
        *inventory["source_shapes"]
            .as_object_mut()
            .unwrap()
            .values_mut()
            .next()
            .unwrap() = json!([1]);
        assert!(binding
            .checked_contract_digest(&canonical_bytes(&inventory))
            .is_err());
        let mut rotary: Value = serde_json::from_slice(&bytes).unwrap();
        rotary["rotary_shapes"] = json!({"forged":[1,2]});
        assert!(binding
            .checked_contract_digest(&canonical_bytes(&rotary))
            .is_err());
        let mut changed: Value = serde_json::from_slice(&bytes).unwrap();
        let layout = changed["source_layouts"]
            .as_object_mut()
            .unwrap()
            .values_mut()
            .next()
            .unwrap();
        *layout = json!("wrong");
        assert!(binding
            .checked_contract_digest(&canonical_bytes(&changed))
            .is_err());
        let mut changed: Value = serde_json::from_slice(&bytes).unwrap();
        changed["unexpected"] = json!(true);
        assert!(binding
            .checked_contract_digest(&canonical_bytes(&changed))
            .is_err());
        // Pins are format-checked authenticated-caller claims, not proven by a
        // tensor stream. Their valid substitution needs caller authentication.
        changed.as_object_mut().unwrap().remove("unexpected");
        changed["source_lock"] = json!("c".repeat(64));
        assert!(binding
            .checked_contract_digest(&canonical_bytes(&changed))
            .is_ok());
    }
}

#[test]
fn exact_sdk_identities_and_compact_public_profile_params() {
    assert_eq!(PROTOCOL, "pllm/fusefss-source-decoder/v1");
    assert_eq!(TOPOLOGY, "pllm/trusted-dealer-two-share/v1");
    assert_eq!(STATE, "pllm/fixed-count-private-prefix/v1");
    for p in [GPT2_NUMERIC, DENSE_ROPE_NUMERIC] {
        let e = document(&experiment(p));
        assert!(selected(&e.pipeline));
        validate(&e).unwrap();
    }
    let mut old = experiment(GPT2_NUMERIC);
    old["pipeline"]["components"] =
        json!({"linear":{"component":"pllm/masked-linear","params":{}}});
    assert!(!selected(&document(&old).pipeline));
    for slot in ["linear", "topology", "state", "wrong-slot"] {
        let mut partial = old.clone();
        partial["pipeline"]["components"][slot] = json!({"component":STATE,"params":{}});
        assert!(selected(&document(&partial).pipeline));
        assert!(validate(&document(&partial)).is_err());
    }
}

#[test]
fn rejects_noncanonical_params_client_dealer_extra_slots_and_wrong_cpu() {
    for (slot, key, value) in [
        (
            "linear",
            "numeric_profile",
            json!("fusefss.source-derived-ring50-q12.v1"),
        ),
        ("linear", "numeric_profile", json!(false)),
        ("linear", "weights", json!("private")),
        ("linear", "weights", json!("public")), // SDK omits the immutable public default.
        ("linear", "range_policy", json!("unchecked")),
        ("topology", "setup_owner", json!("client")),
        ("topology", "client_is_dealer", json!(true)),
        ("state", "kv_cache", json!(true)),
        ("state", "streaming", json!(true)),
        ("state", "early_exit", json!(true)),
        ("state", "generation_count", json!(2)),
        ("kernels", "threads", json!(2)),
        ("kernels", "threads", json!(1.0)),
        ("kernels", "threads", json!(true)),
    ] {
        let mut e = experiment(GPT2_NUMERIC);
        e["pipeline"]["components"][slot]["params"][key] = value;
        assert!(validate(&document(&e)).is_err(), "{slot}:{key}");
    }
    for slot in [
        "cache",
        "verification",
        "preparation",
        "dealer",
        "nonlinear",
        "extra",
    ] {
        let mut e = experiment(GPT2_NUMERIC);
        e["pipeline"]["components"][slot] = json!({"component":STATE,"params":{}});
        assert!(validate(&document(&e)).is_err(), "{slot}");
    }
    for (slot, id) in [
        ("linear", "pllm/puma-three-party-decoder/v1"),
        ("topology", "pllm/two-online-offset-workers/v1"),
        ("state", "pllm/client-local-kv"),
        ("kernels", "pllm/apple-metal-int8/v1"),
    ] {
        let mut e = experiment(GPT2_NUMERIC);
        e["pipeline"]["components"][slot]["component"] = json!(id);
        assert!(validate(&document(&e)).is_err());
    }
    let mut missing = experiment(GPT2_NUMERIC);
    missing["pipeline"]["components"]["linear"]["params"] = json!({});
    assert!(validate(&document(&missing)).is_err());
}

#[test]
fn native_request_source_and_local_topology_envelope_is_bounded() {
    for kind in ["huggingface", "safetensors"] {
        let mut e = experiment(GPT2_NUMERIC);
        e["pipeline"]["model"]["kind"] = json!(kind);
        validate(&document(&e)).unwrap();
    }
    for kind in ["tiny", "vllm", "gguf", "mlx", "ollama"] {
        let mut e = experiment(GPT2_NUMERIC);
        e["pipeline"]["model"]["kind"] = json!(kind);
        assert!(validate(&document(&e)).is_err());
    }
    for (field, n) in [
        ("requests", 0),
        ("requests", 65),
        ("max_input_tokens", 0),
        ("max_input_tokens", 17),
        ("max_new_tokens", 0),
        ("max_new_tokens", 5),
        ("max_input_tokens", u64::MAX),
    ] {
        let mut e = experiment(GPT2_NUMERIC);
        e["budget"][field] = json!(n);
        assert!(validate(&document(&e)).is_err(), "{field}:{n}");
    }
    let mut max = experiment(GPT2_NUMERIC);
    max["budget"] = json!({"requests":64,"max_input_tokens":13,"max_new_tokens":4});
    validate(&document(&max)).unwrap();
    max["budget"]["max_input_tokens"] = json!(14);
    assert!(validate(&document(&max)).is_err());
    let mut e = experiment(GPT2_NUMERIC);
    e["schema"] = json!("pllm.experiment.v3");
    assert!(validate(&document(&e)).is_err());
    e["deployment"] = json!({"kind":"network","network_id":"fss-test-network","network_spec_digest":"a".repeat(64),"snapshot_digest":null});
    assert!(validate(&document(&e)).is_err());
    let mut e = experiment(GPT2_NUMERIC);
    e["pipeline"]["model"]["endpoint"] = json!("http://remote.invalid");
    assert!(validate(&document(&e)).is_err());
    let mut e = experiment(GPT2_NUMERIC);
    e["deployment"]["root"] = json!("");
    assert!(validate(&document(&e)).is_err());
}

#[test]
fn learned_positions_conv1d_bias_norm_and_tied_source_shapes_are_explicit() {
    for tied in [true, false] {
        let mut c = absolute();
        c["tie_word_embeddings"] = json!(tied);
        let p = preflight(&c, GPT2_NUMERIC).unwrap();
        assert_eq!((p.width, p.vocabulary, p.maximum_sequence), (8, 17, 4));
        assert_eq!(p.source_shapes["transformer.wte.weight"], vec![17, 8]);
        assert_eq!(p.source_shapes["transformer.wpe.weight"], vec![16, 8]);
        assert_eq!(
            p.source_shapes["transformer.h.0.attn.c_attn.weight"],
            vec![8, 24]
        );
        assert_eq!(
            p.source_shapes["transformer.h.0.attn.c_attn.bias"],
            vec![24]
        );
        assert_eq!(
            p.source_layouts["transformer.h.0.attn.c_attn.weight"],
            "input_output"
        );
        assert_eq!(
            p.source_shapes["transformer.h.0.mlp.c_fc.weight"],
            vec![8, 32]
        );
        assert_eq!(
            p.source_shapes["transformer.h.0.mlp.c_proj.weight"],
            vec![32, 8]
        );
        for name in [
            "transformer.h.0.ln_1.weight",
            "transformer.h.0.ln_1.bias",
            "transformer.h.0.ln_2.bias",
            "transformer.ln_f.bias",
        ] {
            assert_eq!(p.source_shapes[name], vec![8]);
        }
        assert_eq!(p.source_shapes.contains_key("lm_head.weight"), !tied);
        if !tied {
            assert_eq!(p.source_shapes["lm_head.weight"], vec![17, 8]);
            assert_eq!(p.source_layouts["lm_head.weight"], "output_input");
        }
        assert!(p.rotary_shapes.is_empty());
        assert_eq!(p.rotary_float32_bytes, 0);
        let values: usize = p
            .source_shapes
            .values()
            .map(|s| s.iter().product::<usize>())
            .sum();
        assert_eq!(p.source_float32_bytes, values * 4);
        assert_eq!(p.source_q12_value_bytes, values * 8);
        assert!(p.source_minimum_owner_bytes > p.source_q12_value_bytes);
    }
}

#[test]
fn dense_rope_gqa_qk_norm_and_bias_free_controls_use_semantic_capabilities() {
    let c = dense();
    let p = preflight(&c, DENSE_ROPE_NUMERIC).unwrap();
    assert_eq!(
        p.source_shapes["model.layers.0.self_attn.q_proj.weight"],
        vec![8, 8]
    );
    assert_eq!(
        p.source_shapes["model.layers.0.self_attn.k_proj.weight"],
        vec![4, 8]
    );
    assert_eq!(
        p.source_shapes["model.layers.0.self_attn.k_proj.bias"],
        vec![4]
    );
    assert_eq!(
        p.source_shapes["model.layers.0.input_layernorm.weight"],
        vec![8]
    );
    assert_eq!(
        p.source_layouts["model.layers.0.self_attn.k_proj.weight"],
        "output_input"
    );
    assert_eq!(p.rotary_shapes["layer.0.rope_q"], vec![4, 4]);
    assert_eq!(p.rotary_shapes["layer.0.rope_k"], vec![4, 4]);
    assert_eq!(p.rotary_float32_bytes, 4 * 4 * 4 * 2 * 4);
    let mut qk = c.clone();
    qk["model_type"] = json!("qwen3");
    for (key, value) in [
        ("head_dim", json!(4)),
        ("attention_bias", json!(false)),
        ("attention_dropout", json!(0.0)),
        ("rope_scaling", Value::Null),
        ("sliding_window", Value::Null),
        ("use_sliding_window", json!(false)),
        ("use_cache", json!(true)),
        ("max_window_layers", json!(2)),
    ] {
        qk[key] = value;
    }
    let p = preflight(&qk, DENSE_ROPE_NUMERIC).unwrap();
    assert_eq!(
        p.source_shapes["model.layers.0.self_attn.q_norm.weight"],
        vec![4]
    );
    assert_eq!(
        p.source_shapes["model.layers.0.self_attn.k_norm.weight"],
        vec![4]
    );
    assert!(!p
        .source_shapes
        .contains_key("model.layers.0.self_attn.q_proj.bias"));
    let mut bias_free = c;
    bias_free["model_type"] = json!("llama");
    let p = preflight(&bias_free, DENSE_ROPE_NUMERIC).unwrap();
    assert!(!p.source_shapes.keys().any(|s| s.ends_with(".bias")));
    assert!(!p.source_shapes.keys().any(|s| s.contains("q_norm")));
}

#[test]
fn rejects_profile_mismatch_unapproved_source_attributes_dtype_and_rotary() {
    assert!(preflight(&absolute(), DENSE_ROPE_NUMERIC).is_err());
    assert!(preflight(&dense(), GPT2_NUMERIC).is_err());
    assert!(preflight(&dense(), "fusefss.source-derived-ring50-q12.v1").is_err());
    for (key, value) in [
        ("rope_scaling", json!({"type":"linear","factor":2})),
        ("partial_rotary_factor", json!(0.5)),
        ("use_cache", json!(false)),
        ("attention_dropout", json!(0.1)),
        ("torch_dtype", json!("int8")),
        ("mlp_bias", json!(true)),
        ("extra_norm", json!(true)),
        ("attention_bias", json!(false)),
        ("head_dim", json!(8)),
        (
            "layer_types",
            json!(["sliding_attention", "full_attention"]),
        ),
    ] {
        let mut c = dense();
        c[key] = value;
        assert!(preflight(&c, DENSE_ROPE_NUMERIC).is_err(), "{key}");
    }
    let mut odd = dense();
    odd["hidden_size"] = json!(6);
    assert!(preflight(&odd, DENSE_ROPE_NUMERIC).is_err());
}

#[test]
fn tensor_geometry_total_source_owner_and_expanded_views_reject_before_reads() {
    assert!(preflight_source(&[], 3, 2, GPT2_NUMERIC).is_err());
    assert!(preflight_source(&vec![b' '; MAX_CONFIG_BYTES + 1], 3, 2, GPT2_NUMERIC).is_err());
    let mut exact = canonical_bytes(&absolute());
    exact.resize(MAX_CONFIG_BYTES, b' ');
    assert!(preflight_source(&exact, 3, 2, GPT2_NUMERIC).is_ok());
    let mut full_size = absolute();
    full_size["n_embd"] = json!(768);
    full_size["n_head"] = json!(12);
    full_size["n_layer"] = json!(12);
    full_size["vocab_size"] = json!(50257);
    full_size["n_positions"] = json!(1024);
    assert!(preflight(&full_size, GPT2_NUMERIC).is_err()); // Metadata only; no checkpoint reads.
    for (rows, outputs) in [(0, 1), (1, 0), (1, 5), (16, 2), (usize::MAX, 4)] {
        assert!(
            preflight_source(&canonical_bytes(&absolute()), rows, outputs, GPT2_NUMERIC).is_err()
        );
    }
    let mut c = absolute();
    c["n_embd"] = json!(128);
    c["n_head"] = json!(8);
    c["n_inner"] = json!(128);
    assert!(preflight(&c, GPT2_NUMERIC)
        .err()
        .unwrap()
        .contains("MAX_VALUES"));
    let mut c = dense();
    c["hidden_size"] = json!(80);
    c["num_attention_heads"] = json!(4);
    c["num_key_value_heads"] = json!(4);
    c["intermediate_size"] = json!(160);
    c["num_hidden_layers"] = json!(20);
    assert!(preflight(&c, DENSE_ROPE_NUMERIC)
        .err()
        .unwrap()
        .contains("MAX_BYTES"));
    let mut c = absolute();
    c["n_embd"] = json!(129);
    c["n_head"] = json!(129);
    c["n_inner"] = json!(8);
    c["n_layer"] = json!(1);
    assert!(preflight_source(&canonical_bytes(&c), 13, 4, GPT2_NUMERIC)
        .err()
        .unwrap()
        .contains("MAX_VALUES"));
    assert!(count(&[u64::MAX, 2]).is_err());
    let mut shapes = BTreeMap::new();
    let mut layouts = BTreeMap::new();
    bind(
        Capacity::Capsule,
        &mut shapes,
        &mut layouts,
        "shared",
        &[2, 4],
        "output_input",
    )
    .unwrap();
    assert!(bind(
        Capacity::Capsule,
        &mut shapes,
        &mut layouts,
        "shared",
        &[4, 2],
        "output_input"
    )
    .is_err());
    assert!(bind(
        Capacity::Capsule,
        &mut shapes,
        &mut layouts,
        "shared",
        &[2, 4],
        "input_output"
    )
    .is_err());
}

#[test]
fn mutated_semantic_ops_shapes_dtype_orientation_and_unapproved_numeric_attrs_reject() {
    let p = preflight(&absolute(), GPT2_NUMERIC).unwrap().plan;
    for (id, key, value) in [
        ("position_lookup", "weight_shape", json!([3, 8])),
        ("layer.0.qkv_linear", "weight_layout", json!("output_input")),
        ("layer.0.input_norm", "bias_shape", json!([7])),
        ("output_head", "tied", json!(false)),
        ("layer.0.qkv_linear", "compute_dtype", json!("int8")),
        ("layer.0.gelu", "numeric_profile", json!("other")),
    ] {
        let mut bad = p.clone();
        bad.prefill
            .operations
            .iter_mut()
            .find(|v| v.id == id)
            .unwrap()
            .attributes[key] = value;
        assert!(preflight_plan(bad, 2, GPT2_NUMERIC).is_err(), "{id}:{key}");
    }
    let mut bad = p;
    bad.prefill.operations[0].operator = Op::SecureGather;
    assert!(preflight_plan(bad, 2, GPT2_NUMERIC).is_err());
    let p = preflight(&dense(), DENSE_ROPE_NUMERIC).unwrap().plan;
    for (key, value) in [
        ("frequency_scaling", json!({"factor":2})),
        ("coefficient_profile", json!("arbitrary")),
        ("rotary_dimensions", json!(2)),
        ("pairing", json!("interleaved")),
    ] {
        let mut bad = p.clone();
        bad.prefill
            .operations
            .iter_mut()
            .find(|v| v.operator == Op::RotaryEmbedding)
            .unwrap()
            .attributes[key] = value;
        assert!(preflight_plan(bad, 2, DENSE_ROPE_NUMERIC).is_err());
    }
}

#[test]
fn canonical_metadata_binds_pins_composition_source_shapes_and_explicit_unknowns() {
    for (profile, c) in [(GPT2_NUMERIC, absolute()), (DENSE_ROPE_NUMERIC, dense())] {
        let e = experiment(profile);
        let bytes = contract(&e, &c, 3, 2).unwrap();
        let v: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(canonical_bytes(&v), bytes);
        assert_eq!(v["schema_version"], "pllm.fss.sdk-contract.v1");
        assert_eq!(v["numeric_profile"], profile);
        assert_eq!(v["weights"], "public");
        assert_eq!(v["setup_owner"], "dealer");
        assert_eq!(v["source_lock"], "a".repeat(64));
        assert_eq!(v["tokenizer"], "b".repeat(64));
        assert_eq!(
            v["composition"],
            serde_json::to_value(crate::pipeline_digest(&document(&e).pipeline)).unwrap()
        );
        assert_eq!(
            v["source_shapes"],
            serde_json::to_value(preflight(&c, profile).unwrap().source_shapes).unwrap()
        );
        assert_eq!(v["selection_prefix_rows"], json!([3, 4]));
        assert_eq!(v["output_tokens"], 2);
        assert_eq!(v["vocabulary"], 17);
        assert_eq!(v["width"], 8);
        assert_eq!(v["persistent_kv"], false);
        assert_eq!(v["admission"]["numeric"], "not_performed");
        assert_eq!(v["admission"]["serving_qualified"], false);
        for key in [
            "source_owner_reservation_bytes",
            "rotary_f32_payloads",
            "range_obligations",
            "domain_admission",
            "program_resources",
            "transport_resources",
            "complete_live_accounting",
        ] {
            assert_eq!(v["admission"][key], Value::Null);
        }
        assert_eq!(bytes, contract(&e, &c, 3, 2).unwrap());
        assert_ne!(
            bytes,
            source_contract(
                &canonical_bytes(&e),
                &canonical_bytes(&c),
                3,
                2,
                &"c".repeat(64),
                &"b".repeat(64)
            )
            .unwrap()
        );
        let mut changed = e;
        changed["name"] = json!("changed-configuration");
        assert_ne!(bytes, contract(&changed, &c, 3, 2).unwrap());
    }
}

#[test]
fn metadata_rejects_token_source_pins_noncanonical_json_and_oversized_saved_envelopes() {
    let e = experiment(GPT2_NUMERIC);
    let c = absolute();
    assert!(source_contract(
        &[],
        &canonical_bytes(&c),
        3,
        2,
        &"a".repeat(64),
        &"b".repeat(64)
    )
    .is_err());
    assert!(source_contract(
        &vec![b' '; MAX_EXPERIMENT_BYTES + 1],
        &canonical_bytes(&c),
        3,
        2,
        &"a".repeat(64),
        &"b".repeat(64)
    )
    .is_err());
    for bad in [
        "".to_owned(),
        "a".repeat(63),
        "A".repeat(64),
        "0".repeat(64),
        "g".repeat(64),
    ] {
        assert!(source_contract(
            &canonical_bytes(&e),
            &canonical_bytes(&c),
            3,
            2,
            &bad,
            &"b".repeat(64)
        )
        .is_err());
        assert!(source_contract(
            &canonical_bytes(&e),
            &canonical_bytes(&c),
            3,
            2,
            &"a".repeat(64),
            &bad
        )
        .is_err());
    }
    for (rows, outputs) in [(4, 2), (3, 3), (0, 1), (1, 0)] {
        assert!(contract(&e, &c, rows, outputs).is_err());
    }
    let mut bytes = canonical_bytes(&e);
    bytes.push(b'\n');
    assert!(source_contract(
        &bytes,
        &canonical_bytes(&c),
        3,
        2,
        &"a".repeat(64),
        &"b".repeat(64)
    )
    .is_err());
    let mut small_table = c;
    small_table["n_positions"] = json!(3);
    assert!(contract(&e, &small_table, 1, 1).is_err()); // Actual query fits; declared budget does not.
    assert!(contract(&e, &absolute(), 1, 1).is_ok());
}
