//! Source definitions, UNRUN. Synthetic cost/error/owner assumptions only.
use super::*;
use pllm_core::interleaved_kv_layout::RetainedCache;
use pllm_core::row_fold::WorkLedger;
use schedule::construction::{
    Backing, CapabilityCount, Element, Execution, OperationCount, OtherOwner,
    PublicCiphertextConstructionManifest, quote_public_ciphertext_construction,
    quote_public_ciphertext_dag_construction,
};

fn geometry() -> Geometry {
    Geometry::new(64, 2, 4, 33).unwrap()
}
fn state() -> PublicCiphertextState {
    PublicCiphertextState {
        scheme: "ckks".into(),
        key_domain: "synthetic_key".into(),
        layout: "synthetic_prepacked_delta".into(),
        scale: "synthetic_scale".into(),
        level: 2,
        shape: vec![64],
        error: PublicCiphertextError {
            model_identity: "synthetic_assumption".into(),
            bound: Some(1),
        },
    }
}
fn cost() -> PublicCost {
    PublicCost {
        identity: "synthetic_unit".into(),
        units: Some(1),
    }
}
fn val(id: String) -> PublicCiphertextValue {
    PublicCiphertextValue {
        owner: id.clone(),
        identity: id,
        state: state(),
    }
}
fn empty() -> PublicCiphertextGraph {
    PublicCiphertextGraph {
        cost_model_identity: "synthetic_only".into(),
        cost_unit: "synthetic_unit".into(),
        maximum_level: 2,
        owners: vec![],
        persistent_owners: vec![],
        inputs: vec![],
        operations: vec![],
        capabilities: vec![],
        outputs: vec![],
        retained_inputs: vec![],
    }
}
fn capacity() -> PublicEncryptedDagCapacity {
    PublicEncryptedDagCapacity {
        max_operations: 5,
        max_states: 326,
        max_frontier: 326,
        max_work: 10_000_000_000_000,
        max_buffer_bytes: 24 << 20,
        trace: PublicCiphertextTraceCapacity {
            max_records: 1024,
            max_work: 1_000_000_000_000,
            max_buffer_bytes: 2 << 20,
            max_live_owner_bytes: 1 << 20,
        },
    }
}
struct Fixture {
    original: PublicCiphertextGraph,
    keys: Vec<String>,
    values: Vec<String>,
    incoming: Vec<String>,
    outputs: Vec<PublicCiphertextValue>,
    owners: Vec<PublicCiphertextOwner>,
    caps: Vec<PublicCiphertextCapability>,
}
impl Fixture {
    fn new(active: usize) -> Self {
        let g = geometry();
        let mut original = empty();
        let keys = (0..g.key_layout().ciphertexts(active).unwrap())
            .map(|i| format!("old_k_{i}"))
            .collect::<Vec<_>>();
        let values = (0..g.value_layout().ciphertexts(active).unwrap())
            .map(|i| format!("old_v_{i}"))
            .collect::<Vec<_>>();
        let incoming = (0..5).map(|i| format!("delta_{i}")).collect::<Vec<_>>();
        for id in keys.iter().chain(&values).chain(&incoming) {
            original.inputs.push(val(id.clone()));
            original.owners.push(PublicCiphertextOwner {
                identity: id.clone(),
                bytes: Some(512),
            });
        }
        let outputs = (0..5).map(|i| val(format!("next_{i}"))).collect::<Vec<_>>();
        let owners = outputs
            .iter()
            .map(|v| PublicCiphertextOwner {
                identity: v.owner.clone(),
                bytes: Some(512),
            })
            .collect();
        let caps = (0..5)
            .map(|i| {
                let partial = if i == 0 {
                    active % 8 != 0
                } else {
                    active % 32 != 0
                };
                PublicCiphertextCapability {
                    identity: format!("append_cap_{i}"),
                    kernel_identity: format!("cachemir_append_v1:64:2:4:33:{active}:{i}"),
                    primitive: PublicPrimitive::Arithmetic,
                    inputs: vec![state(); 1 + usize::from(partial)],
                    output: state(),
                    join: if partial {
                        PublicCiphertextJoin::Aligned
                    } else {
                        PublicCiphertextJoin::Unary
                    },
                    execution: PublicCiphertextExecution::Ckks {
                        compute: PublicLevelCompute {
                            input_level: 2,
                            cost: cost(),
                        },
                        depth: 0,
                        transition: PublicLevelTransition {
                            kind: PublicLevelTransitionKind::Preserve,
                            from_level: 2,
                            to_level: 2,
                            // Canonical Preserve is explicitly zero; compute
                            // remains known/nonzero. Never erase unknown cost.
                            cost: PublicCost {
                                units: Some(0),
                                ..cost()
                            },
                        },
                    },
                    error_assumption: Some("explicit_synthetic_not_calibrated".into()),
                    scratch_bytes: Some(0),
                    recovery_overlap_bytes: Some(0),
                }
            })
            .collect();
        Self {
            original,
            keys,
            values,
            incoming,
            outputs,
            owners,
            caps,
        }
    }
    fn parents(&self) -> (Vec<&str>, Vec<&str>) {
        (
            self.keys.iter().map(String::as_str).collect(),
            self.values.iter().map(String::as_str).collect(),
        )
    }
    fn replacements(&self) -> Vec<Replacement<'_>> {
        (0..5)
            .map(|i| Replacement {
                incoming: &self.incoming[i],
                output: &self.outputs[i],
                owner: &self.owners[i],
                capability: &self.caps[i],
            })
            .collect()
    }
}
fn append_at(active: usize) -> AppendDescriptor {
    let g = geometry();
    let mut k = vec![0; g.key_layout().storage_words().unwrap()];
    let mut v = vec![0; g.value_layout().storage_words().unwrap()];
    let mut cache = RetainedCache::new(g, &mut k, &mut v).unwrap();
    for _ in 0..active {
        cache.append(&[0; 8], &[0; 8]).unwrap();
    }
    cache.append_descriptor().unwrap()
}

#[test]
fn real_geometry_append_derived_all_groups_against_independent_dense() {
    let g = geometry();
    let mut k = vec![0; g.key_layout().storage_words().unwrap()];
    let mut v = vec![0; g.value_layout().storage_words().unwrap()];
    let mut cache = RetainedCache::new(g, &mut k, &mut v).unwrap();
    let mut dense_k = Vec::new();
    let mut dense_v = Vec::new();
    for token in 0..33 {
        let before = cache.derived_view();
        let old_k = before.key_slots().to_vec();
        let old_v = before.value_slots().to_vec();
        let descriptor = cache.append_descriptor().unwrap();
        assert_eq!(descriptor.token, token);
        super::descriptor(g, descriptor).unwrap();
        let key = std::array::from_fn::<_, 8, _>(|i| (token as i64 + i as i64) % 7 - 3);
        let value = std::array::from_fn::<_, 8, _>(|i| (2 * token as i64 + i as i64) % 9 - 4);
        cache.append(&key, &value).unwrap();
        dense_k.push(key);
        dense_v.push(value);
        let view = cache.derived_view();
        for old in 0..token {
            for h in 0..2 {
                for f in 0..4 {
                    let a = g.key_layout().address(old, h, f).unwrap();
                    assert_eq!(
                        view.key_slots()[a.ciphertext * 64 + a.slot],
                        old_k[a.ciphertext * 64 + a.slot]
                    );
                    let a = g.value_layout().address(old, h, f).unwrap();
                    assert_eq!(
                        view.value_slots()[a.ciphertext * 64 + a.slot],
                        old_v[a.ciphertext * 64 + a.slot]
                    );
                }
            }
        }
        let query = [1, -2, 3, -1, 2, 1, -1, 3];
        let mut scores = vec![0; g.score_words()];
        view.query_key(&query, &mut scores).unwrap();
        let mut weights = vec![0; g.score_words()];
        for (t, key_row) in dense_k.iter().enumerate().take(token + 1) {
            for h in 0..2 {
                let a = g.score_address(t, h).unwrap();
                assert_eq!(
                    scores[a.ciphertext * 64 + a.slot],
                    (0..4)
                        .map(|f| query[h * 4 + f] * key_row[h * 4 + f])
                        .sum::<i64>()
                );
                weights[a.ciphertext * 64 + a.slot] = t as i64 % 3 - 1;
            }
        }
        let mut output = [0; 64];
        view.attention_value(&weights, &mut output).unwrap();
        for h in 0..2 {
            for f in 0..4 {
                assert_eq!(
                    output[g.output_slot(h, f).unwrap()],
                    (0..=token)
                        .map(|t| (t as i64 % 3 - 1) * dense_v[t][h * 4 + f])
                        .sum::<i64>()
                );
            }
        }
    }
    assert!(cache.append_descriptor().is_err());
    assert!(cache.append(&[0; 8], &[0; 8]).is_err());
}

#[test]
fn synthetic_graph_partial_replacement_new_groups_solver_and_trace() {
    // Component graph/solver qualification, NOT outer constructor admission.
    for active in [0, 1, 7, 8, 9, 31, 32] {
        let f = Fixture::new(active);
        let (k, v) = f.parents();
        let p = Parents {
            keys: &k,
            values: &v,
        };
        let r = f.replacements();
        let a = append_at(active);
        preflight(geometry(), a, &f.original, &p, &r).unwrap();
        let graph = emit_graph(geometry(), a, &f.original, &p, &r).unwrap();
        assert_eq!(graph.operations.len(), 5);
        assert_eq!(graph.outputs.len(), k.len() + v.len() + 5);
        for id in k.iter().chain(&v) {
            assert!(graph.outputs.iter().any(|s| s == id));
            assert!(graph.retained_inputs.iter().any(|s| s == id));
        }
        for (i, op) in graph.operations.iter().enumerate() {
            assert_eq!(op.inputs.last().unwrap(), &f.incoming[i]);
            assert_eq!(
                op.inputs.len(),
                1 + usize::from(if i == 0 {
                    active % 8 != 0
                } else {
                    active % 32 != 0
                })
            );
        }
        let c = capacity();
        let solution = solve_public_encrypted_dag(&graph, c).unwrap();
        let checked =
            validate_public_ciphertext_trace(&graph, &solution.ledger.trace, c.trace).unwrap();
        assert_eq!(checked, solution.trace_summary);
        assert!(
            serde_json::to_vec(&(&graph, &solution.ledger))
                .unwrap()
                .len()
                <= 1 << 20,
            "qualification output ceiling"
        );
        let mut owner_short = c;
        owner_short.trace.max_live_owner_bytes = checked.peak_live_owner_bytes - 1;
        assert!(solve_public_encrypted_dag(&graph, owner_short).is_err());
        let mut exhausted = c;
        exhausted.max_states = 1;
        assert!(solve_public_encrypted_dag(&graph, exhausted).is_err());
    }
}

#[test]
fn missing_parent_stale_graph_key_scheme_level_error_refuse() {
    for fault in 0..8 {
        let mut f = Fixture::new(7);
        match fault {
            0 => {
                f.original.inputs.remove(0);
            }
            1 => {
                f.caps[0].inputs[0].key_domain = "wrong".into();
            }
            2 => {
                f.caps[0].inputs[0].scheme = "wrong".into();
            }
            3 => {
                f.caps[0].inputs[0].level = 1;
            }
            4 => {
                f.outputs[0].state.error.bound = None;
            }
            5 => {
                f.owners[0].bytes = None;
            }
            6 => {
                f.caps[0].scratch_bytes = None;
            }
            _ => {
                f.caps[0].execution = PublicCiphertextExecution::View { cost: cost() };
            }
        }
        let (k, v) = f.parents();
        assert!(
            preflight(
                geometry(),
                append_at(7),
                &f.original,
                &Parents {
                    keys: &k,
                    values: &v
                },
                &f.replacements()
            )
            .is_err()
        );
    }
}

#[test]
fn forged_descriptor_and_partial_pair_refuse_without_graph_factory() {
    let f = Fixture::new(8);
    let (k, v) = f.parents();
    let p = Parents {
        keys: &k,
        values: &v,
    };
    let r = f.replacements();
    let mut a = append_at(8);
    a.new_key_ciphertexts = 0;
    assert_eq!(
        preflight(geometry(), a, &f.original, &p, &r),
        Err(Refusal::Geometry)
    );
    assert!(preflight(geometry(), append_at(8), &f.original, &p, &r[..4]).is_err());
}

#[test]
fn known_factory_overlap_real_counter_exact_short_and_unwind() {
    // Component ownership closure; outer manifest/trace admission tested later
    // with its independently acquired exact original-capacity fixture.
    let f = Fixture::new(7);
    let (k, v) = f.parents();
    let p = Parents {
        keys: &k,
        values: &v,
    };
    let r = f.replacements();
    let payload = factory_bytes(&f.original, &p, &r).unwrap();
    let bootstrap = OwnedBudget::resources().unwrap().initial_reserved_bytes;
    let exact = bootstrap + payload + size_of::<OwnedExternalCharge>();
    for short in [false, true] {
        let budget = OwnedBudget::new(exact - usize::from(short)).unwrap();
        let baseline = budget.budget().used_bytes();
        let charge = budget.charge_external(payload);
        if short {
            assert!(charge.is_err());
            assert_eq!(budget.budget().used_bytes(), baseline);
        } else {
            let charge = charge.unwrap();
            assert_eq!(budget.budget().used_bytes(), exact);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _charge = charge;
                let _graph = emit_graph(geometry(), append_at(7), &f.original, &p, &r).unwrap();
                panic!("analysis-only cancellation/unwind fixture");
            }));
            assert!(result.is_err());
            assert_eq!(budget.budget().used_bytes(), baseline);
        }
    }
}

#[test]
fn stale_capability_token_and_role_refuse_before_graph_emission() {
    let mut f = Fixture::new(8);
    f.caps[0].kernel_identity = "cachemir_append_v1:64:2:4:33:7:0".into();
    let (k, v) = f.parents();
    assert_eq!(
        preflight(
            geometry(),
            append_at(8),
            &f.original,
            &Parents {
                keys: &k,
                values: &v
            },
            &f.replacements()
        ),
        Err(Refusal::Capability)
    );
}

/// Independent declaration of the constructor's original capacities. This
/// builds NO expected graph and calls neither emit_graph nor the solver.
struct ManifestFixture {
    input_ranks: Vec<usize>,
    operations: Vec<OperationCount>,
    cap_ranks: Vec<Vec<usize>>,
    backings: Vec<Backing>,
    releases: Vec<usize>,
    other: Vec<OtherOwner>,
}
impl ManifestFixture {
    fn new(f: &Fixture, active: usize) -> Self {
        fn b(v: &mut Vec<Backing>, e: Element, n: usize) {
            v.push(Backing {
                element: e,
                length: n,
                capacity: Some(n),
                first_phase: 0,
                end_phase: 2,
            });
        }
        fn text(v: &mut Vec<Backing>, s: &str) {
            b(v, Element::Byte, s.len());
        }
        fn state(v: &mut Vec<Backing>, s: &PublicCiphertextState) {
            for s in [
                &s.scheme,
                &s.key_domain,
                &s.layout,
                &s.scale,
                &s.error.model_identity,
            ] {
                text(v, s);
            }
            b(v, Element::Shape, s.shape.len());
        }
        fn value(v: &mut Vec<Backing>, x: &PublicCiphertextValue) {
            text(v, &x.identity);
            text(v, &x.owner);
            state(v, &x.state);
        }
        let parents = f.keys.len() + f.values.len();
        let mut backings = Vec::new();
        text(&mut backings, &f.original.cost_model_identity);
        text(&mut backings, &f.original.cost_unit);
        for (e, n) in [
            (Element::Owner, f.original.owners.len() + 5),
            (Element::String, 0),
            (Element::Value, f.original.inputs.len()),
            (Element::Operation, 5),
            (Element::Capability, 5),
            (Element::String, parents + 5),
            (Element::String, parents),
            (Element::Step, 5),
        ] {
            b(&mut backings, e, n);
        }
        for o in f.original.owners.iter().chain(&f.owners) {
            text(&mut backings, &o.identity);
        }
        for x in &f.original.inputs {
            value(&mut backings, x);
        }
        for i in 0..5 {
            text(&mut backings, &f.outputs[i].identity);
            text(&mut backings, &f.caps[i].kernel_identity);
            let old = if i == 0 {
                if active % 8 != 0 {
                    Some(&f.keys[active / 8])
                } else {
                    None
                }
            } else if active % 32 != 0 {
                Some(&f.values[active / 32 * 4 + i - 1])
            } else {
                None
            };
            b(
                &mut backings,
                Element::String,
                1 + usize::from(old.is_some()),
            );
            if let Some(s) = old {
                text(&mut backings, s);
            }
            text(&mut backings, &f.incoming[i]);
            value(&mut backings, &f.outputs[i]);
        }
        for c in &f.caps {
            text(&mut backings, &c.identity);
            text(&mut backings, &c.kernel_identity);
            b(&mut backings, Element::State, c.inputs.len());
            for s in &c.inputs {
                state(&mut backings, s);
            }
            state(&mut backings, &c.output);
            text(&mut backings, c.error_assumption.as_ref().unwrap());
            let PublicCiphertextExecution::Ckks {
                compute,
                transition,
                ..
            } = &c.execution
            else {
                panic!("fixture declares preserve CKKS only")
            };
            text(&mut backings, &compute.cost.identity);
            text(&mut backings, &transition.cost.identity);
        }
        for s in f.keys.iter().chain(&f.values) {
            text(&mut backings, s);
        }
        for x in &f.outputs {
            text(&mut backings, &x.identity);
        }
        for s in f.keys.iter().chain(&f.values) {
            text(&mut backings, s);
        }
        // Unique kernel matches, equal costs: existing solver's documented
        // canonical tie order is operation indices 0..5. Each delta dies once.
        for i in 0..5 {
            text(&mut backings, &f.outputs[i].identity);
            text(&mut backings, &f.caps[i].identity);
            b(&mut backings, Element::String, 1);
            text(&mut backings, &f.incoming[i]);
        }
        Self {
            input_ranks: vec![1; f.original.inputs.len()],
            operations: f
                .caps
                .iter()
                .map(|c| OperationCount {
                    inputs: c.inputs.len(),
                    output_rank: 1,
                })
                .collect(),
            cap_ranks: f.caps.iter().map(|c| vec![1; c.inputs.len()]).collect(),
            backings,
            releases: vec![1; 5],
            other: Vec::new(),
        }
    }
    fn original_bytes(&self, capabilities: &Vec<CapabilityCount<'_>>) -> usize {
        size_of::<Self>()
            + size_of::<PublicCiphertextConstructionManifest<'_>>()
            + self.input_ranks.capacity() * size_of::<usize>()
            + self.operations.capacity() * size_of::<OperationCount>()
            + self.cap_ranks.capacity() * size_of::<Vec<usize>>()
            + self
                .cap_ranks
                .iter()
                .map(|r| r.capacity() * size_of::<usize>())
                .sum::<usize>()
            + self.backings.capacity() * size_of::<Backing>()
            + self.releases.capacity() * size_of::<usize>()
            + self.other.capacity() * size_of::<OtherOwner>()
            + capabilities.capacity() * size_of::<CapabilityCount<'_>>()
    }
}

fn whole_fixture(
    active: usize,
    run: impl FnOnce(
        &Fixture,
        &PublicCiphertextConstructionQuote<'_>,
        &OwnedBudget,
        &mut ControlLedger<'_>,
        &mut RetainedCache<'_>,
        AppendDescriptor,
    ),
) {
    whole_fixture_with_backing_count(active, |f, q, b, c, cache, a, _| run(f, q, b, c, cache, a));
}

// Expose only the ACTUAL independently declared backing census to boundary
// tests. Original setup, quote, source owner and clear-cache coverage unchanged.
fn whole_fixture_with_backing_count(
    active: usize,
    run: impl FnOnce(
        &Fixture,
        &PublicCiphertextConstructionQuote<'_>,
        &OwnedBudget,
        &mut ControlLedger<'_>,
        &mut RetainedCache<'_>,
        AppendDescriptor,
        usize,
    ),
) {
    let mut work = WorkLedger::new(1 << 28, 1 << 20);
    let mut control = ControlLedger::new(&mut work);
    // Independently granted upstream source/manifest region BEFORE acquisition.
    // Its full capacity, not just visible fixture lengths, stays charged. It is
    // not created from the constructor requirement or treated as extra credit.
    control.core_work().debit(0, 8192).unwrap();
    let budget = OwnedBudget::new(64 << 20).unwrap();
    let _upstream = budget.charge_external(128 << 10).unwrap();
    let f = Fixture::new(active);
    let mut mf = ManifestFixture::new(&f, active);
    let capabilities = mf
        .cap_ranks
        .iter()
        .map(|r| CapabilityCount {
            input_ranks: r,
            output_rank: 1,
            execution: Execution::Ckks,
        })
        .collect::<Vec<_>>();
    // Full upstream grant plus named fixture/control/header/copy slots. Query
    // backing census is charged as part of original manifest region below.
    mf.other.push(OtherOwner {
        bytes: Some(128 << 10),
        metadata: true,
        first_phase: 0,
        end_phase: 2,
    });
    let original_bytes = mf.original_bytes(&capabilities);
    let manifest = PublicCiphertextConstructionManifest {
        owners: f.original.owners.len() + 5,
        persistent_owners: 0,
        input_ranks: &mf.input_ranks,
        operations: &mf.operations,
        capabilities: &capabilities,
        outputs: f.keys.len() + f.values.len() + 5,
        retained_inputs: f.keys.len() + f.values.len(),
        release_lengths: &mf.releases,
        backings: &mf.backings,
        other_owners: &mf.other,
        phases: 2,
        manifest_original_bytes: Some(original_bytes),
    };
    assert!(
        matches!(
            quote_public_ciphertext_construction(&manifest, &mut control),
            Err(schedule::construction::Refusal::IncompleteConstructionManifest)
        ),
        "legacy entry remains unary"
    );
    let quote = quote_public_ciphertext_dag_construction(&manifest, &mut control).unwrap();
    let (k, v) = f.parents();
    let p = Parents {
        keys: &k,
        values: &v,
    };
    let replacements = f.replacements();
    let g = geometry();
    let mut keys = vec![0; g.key_layout().storage_words().unwrap()];
    let mut values = vec![0; g.value_layout().storage_words().unwrap()];
    let mut cache = RetainedCache::new(g, &mut keys, &mut values).unwrap();
    control.core_work().debit(active * 8 * 8, 0).unwrap();
    for token in 0..active {
        cache
            .append(&[token as i64 + 1; 8], &[token as i64 + 2; 8])
            .unwrap();
    }
    assert_eq!(cache.append_descriptor().unwrap(), append_at(active));
    let actual = sum([
        factory_bytes(&f.original, &p, &replacements).unwrap(),
        original_bytes,
        (g.key_layout().storage_words().unwrap() + g.value_layout().storage_words().unwrap())
            * size_of::<i64>(),
    ])
    .unwrap();
    assert!(
        actual <= 128 << 10,
        "upstream region includes ORIGINAL capacities/controls, no unpriced manifest acquisition"
    );
    run(
        &f,
        &quote,
        &budget,
        &mut control,
        &mut cache,
        append_at(active),
        manifest.backings.len(),
    );
}

#[test]
fn public_constructor_exact_total_control_succeeds_through_final_validation() {
    for active in [7, 8, 31, 32] {
        whole_fixture_with_backing_count(
            active,
            |f, quote, budget, control, _cache, a, backings| {
                let (k, v) = f.parents();
                let p = Parents {
                    keys: &k,
                    values: &v,
                };
                let r = f.replacements();
                let c = capacity();
                let construction =
                    control_allowance(&f.original, &p, &r, c.trace.max_records).unwrap();
                let validation = 1024usize
                    .checked_add(4usize.checked_mul(backings).unwrap())
                    .unwrap();
                let required = construction.checked_add(validation).unwrap();
                // Exact constructor allowance AFTER real source/quote acquisition.
                // Successful zero remainder proves actual consumption, not just a quote.
                // Simulated prior quota consumption, NOT actual CPU work.
                // Keep the original cumulative post-setup ledger; never reset it.
                let excess = control
                    .core_work()
                    .control_remaining()
                    .checked_sub(required)
                    .unwrap();
                control.core_work().debit(0, excess).unwrap();
                assert_eq!(control.core_work().control_remaining(), required);
                let initial_work = control.core_work().remaining();
                let mut metadata =
                    MetadataLedger::new(c.max_work.checked_add(quote.trace_work()).unwrap());
                let before = budget.budget().used_bytes();
                GRAPH_FACTORIES.with(|n| n.set(0));
                let analysis = analyze_append(
                    geometry(),
                    a,
                    &f.original,
                    &p,
                    &r,
                    quote,
                    budget,
                    &mut metadata,
                    control,
                    c,
                )
                .unwrap();
                assert_eq!(GRAPH_FACTORIES.with(|n| n.get()), 1);
                assert_eq!(control.core_work().control_remaining(), 0);
                assert_eq!(control.core_work().remaining(), initial_work - required);
                assert_eq!(metadata.remaining(), 0);
                assert!(std::ptr::eq(analysis.original(), &f.original));
                assert_eq!(analysis.graph.operations.len(), 5);
                for (i, op) in analysis.graph.operations.iter().enumerate() {
                    assert_eq!(op.inputs.last(), Some(&f.incoming[i]));
                }
                for id in k.iter().chain(&v) {
                    assert!(analysis.graph.retained_inputs.iter().any(|s|s==id));
                    assert!(analysis.graph.outputs.iter().any(|s|s==id));
                }
                assert_eq!(
                    validate_public_ciphertext_trace(
                        &analysis.graph,
                        &analysis.solution.ledger.trace,
                        c.trace
                    )
                    .unwrap(),
                    analysis.solution.trace_summary
                );
                assert!(budget.budget().used_bytes() > before);
                drop(analysis);
                assert_eq!(budget.budget().used_bytes(), before);
                // Successful cleanup does not replenish either cumulative work ledger.
                assert_eq!(control.core_work().control_remaining(), 0);
                assert_eq!(metadata.remaining(), 0);
            },
        );
    }
}

#[test]
fn public_constructor_total_control_one_short_refuses_late_without_publication() {
    for active in [7, 8, 31, 32] {
        whole_fixture_with_backing_count(
            active,
            |f, quote, budget, control, _cache, a, backings| {
                let (k, v) = f.parents();
                let p = Parents {
                    keys: &k,
                    values: &v,
                };
                let r = f.replacements();
                let c = capacity();
                let construction =
                    control_allowance(&f.original, &p, &r, c.trace.max_records).unwrap();
                let validation = 1024usize
                    .checked_add(4usize.checked_mul(backings).unwrap())
                    .unwrap();
                let required = construction.checked_add(validation).unwrap();
                // Simulated prior quota consumption on the SAME setup ledger;
                // no replacement meter, refund, or claim of actual CPU work.
                let excess = control
                    .core_work()
                    .control_remaining()
                    .checked_sub(required - 1)
                    .unwrap();
                control.core_work().debit(0, excess).unwrap();
                assert_eq!(control.core_work().control_remaining(), required - 1);
                let initial_work = control.core_work().remaining();
                let mut metadata =
                    MetadataLedger::new(c.max_work.checked_add(quote.trace_work()).unwrap());
                let before = budget.budget().used_bytes();
                GRAPH_FACTORIES.with(|n| n.set(0));
                assert!(matches!(
                    analyze_append(
                        geometry(),
                        a,
                        &f.original,
                        &p,
                        &r,
                        quote,
                        budget,
                        &mut metadata,
                        control,
                        c
                    ),
                    Err(Refusal::Trace)
                ));
                // Construction/search occurred. Validation's own control gate refused;
                // its rejected debit is atomic, not spent and not mislabelled pre-entry.
                assert_eq!(GRAPH_FACTORIES.with(|n| n.get()), 1);
                assert_eq!(control.core_work().control_remaining(), validation - 1);
                assert_eq!(control.core_work().remaining(), initial_work - construction);
                assert_eq!(metadata.remaining(), 0);
                assert_eq!(budget.budget().used_bytes(), before);
                // No AppendAnalysis returned; existing graph/solution-before-charge drop
                // order is source proof, not an empirical allocation/free observer here.
                assert_eq!(f.original.inputs.len(), k.len() + v.len() + 5);
            },
        );
    }
}

fn verify_clear_boundary(
    cache: &mut RetainedCache<'_>,
    a: AppendDescriptor,
    control: &mut ControlLedger<'_>,
) {
    let g = geometry();
    assert_eq!(cache.append_descriptor().unwrap(), a);
    control.core_work().debit(33 * 8 * 4 * 64, 0).unwrap();
    let old = cache.derived_view();
    let old_k = old.key_slots().to_vec();
    let old_v = old.value_slots().to_vec();
    cache
        .append(&[a.token as i64 + 1; 8], &[a.token as i64 + 2; 8])
        .unwrap();
    let view = cache.derived_view();
    for token in 0..a.token {
        for h in 0..2 {
            for f in 0..4 {
                let k = g.key_layout().address(token, h, f).unwrap();
                let v = g.value_layout().address(token, h, f).unwrap();
                assert_eq!(
                    view.key_slots()[k.ciphertext * 64 + k.slot],
                    old_k[k.ciphertext * 64 + k.slot]
                );
                assert_eq!(
                    view.value_slots()[v.ciphertext * 64 + v.slot],
                    old_v[v.ciphertext * 64 + v.slot]
                );
            }
        }
    }
    let mut scores = vec![0; g.score_words()];
    view.query_key(&[1; 8], &mut scores).unwrap();
    let mut weights = vec![0; g.score_words()];
    for token in 0..=a.token {
        for h in 0..2 {
            let address = g.score_address(token, h).unwrap();
            assert_eq!(
                scores[address.ciphertext * 64 + address.slot],
                4 * (token as i64 + 1)
            );
            weights[address.ciphertext * 64 + address.slot] = 1;
        }
    }
    let mut output = [0; 64];
    view.attention_value(&weights, &mut output).unwrap();
    for h in 0..2 {
        for f in 0..4 {
            assert_eq!(
                output[g.output_slot(h, f).unwrap()],
                (0..=a.token).map(|t| t as i64 + 2).sum::<i64>()
            );
        }
    }
}

#[test]
fn public_constructor_boundary_graph_solver_trace_and_actual_clear_descriptor() {
    for active in [7, 8, 31, 32] {
        whole_fixture(active, |f, quote, budget, control, cache, a| {
            let (k, v) = f.parents();
            let p = Parents {
                keys: &k,
                values: &v,
            };
            let r = f.replacements();
            let c = capacity();
            let total = c.max_work.checked_add(quote.trace_work()).unwrap();
            let mut metadata = MetadataLedger::new(total);
            let before = budget.budget().used_bytes();
            GRAPH_FACTORIES.with(|n| n.set(0));
            let analysis = analyze_append(
                geometry(),
                a,
                &f.original,
                &p,
                &r,
                quote,
                budget,
                &mut metadata,
                control,
                c,
            )
            .unwrap();
            assert!(std::ptr::eq(analysis.original(), &f.original));
            assert_eq!(metadata.remaining(), 0);
            assert_eq!(GRAPH_FACTORIES.with(|n| n.get()), 1);
            assert_eq!(analysis.graph.operations.len(), 5);
            for (i, op) in analysis.graph.operations.iter().enumerate() {
                assert_eq!(op.inputs.last().unwrap(), &f.incoming[i]);
                if op.inputs.len() == 2 {
                    assert_eq!(
                        &op.inputs[0],
                        if i == 0 {
                            &f.keys[a.key_tile]
                        } else {
                            &f.values[a.value_group * 4 + i - 1]
                        }
                    );
                }
            }
            for id in k.iter().chain(&v) {
                assert!(analysis.graph.retained_inputs.iter().any(|s| s == id));
                assert!(analysis.graph.outputs.iter().any(|s| s == id));
            }
            let summary = validate_public_ciphertext_trace(
                &analysis.graph,
                &analysis.solution.ledger.trace,
                c.trace,
            )
            .unwrap();
            assert_eq!(summary, analysis.solution.trace_summary);
            assert!(summary.final_live_backing_bytes >= ((k.len() + v.len() + 5) * 512) as u64);
            assert!(budget.budget().used_bytes() > before);
            drop(analysis);
            assert_eq!(budget.budget().used_bytes(), before);
            verify_clear_boundary(cache, a, control);
        });
    }
}

#[test]
fn public_constructor_independent_metadata_control_and_byte_one_short() {
    for fault in 0..3 {
        whole_fixture(8, |f, quote, budget, control, _cache, a| {
            let (k, v) = f.parents();
            let p = Parents {
                keys: &k,
                values: &v,
            };
            let r = f.replacements();
            let c = capacity();
            let total = c.max_work + quote.trace_work();
            let mut metadata = MetadataLedger::new(total - usize::from(fault == 0));
            // Reserve unrelated headroom, not a new counter/requirement-as-grant.
            // Source and complete manifest stay under ORIGINAL upstream reservation.
            let bytes = owner_bytes(quote, c, factory_bytes(&f.original, &p, &r).unwrap()).unwrap();
            let _headroom = if fault == 2 {
                Some(
                    budget
                        .charge_external(
                            (64 << 20)
                                - budget.budget().used_bytes()
                                - bytes
                                - 2 * size_of::<OwnedExternalCharge>()
                                + 1,
                        )
                        .unwrap(),
                )
            } else {
                None
            };
            let before = budget.budget().used_bytes();
            GRAPH_FACTORIES.with(|n| n.set(0));
            if fault == 1 {
                // Same core meter, consume actual remaining logical control down to
                // allowance-1; no ledger replacement/reset to fabricate the edge.
                let want = control_allowance(&f.original, &p, &r, c.trace.max_records).unwrap() - 1;
                let remaining = control.core_work().control_remaining();
                control.core_work().debit(0, remaining - want).unwrap();
            }
            assert!(
                analyze_append(
                    geometry(),
                    a,
                    &f.original,
                    &p,
                    &r,
                    quote,
                    budget,
                    &mut metadata,
                    control,
                    c
                )
                .is_err()
            );
            assert_eq!(GRAPH_FACTORIES.with(|n| n.get()), 0);
            assert_eq!(budget.budget().used_bytes(), before);
            if fault == 0 {
                assert_eq!(metadata.remaining(), quote.trace_work() - 1);
            }
            if fault == 1 {
                assert_eq!(metadata.remaining(), total);
            }
        });
    }
}

#[test]
fn public_constructor_same_metadata_ledger_repeated_calls_and_failed_search_keep_debit() {
    whole_fixture(31, |f, quote, budget, control, _cache, a| {
        let (k, v) = f.parents();
        let p = Parents {
            keys: &k,
            values: &v,
        };
        let r = f.replacements();
        let c = capacity();
        let total = c.max_work + quote.trace_work();
        let mut metadata = MetadataLedger::new(total * 2);
        for _ in 0..2 {
            let plan = analyze_append(
                geometry(),
                a,
                &f.original,
                &p,
                &r,
                quote,
                budget,
                &mut metadata,
                control,
                c,
            )
            .unwrap();
            drop(plan);
        }
        assert_eq!(metadata.remaining(), 0);
        GRAPH_FACTORIES.with(|n| n.set(0));
        assert!(
            analyze_append(
                geometry(),
                a,
                &f.original,
                &p,
                &r,
                quote,
                budget,
                &mut metadata,
                control,
                c
            )
            .is_err()
        );
        assert_eq!(GRAPH_FACTORIES.with(|n| n.get()), 0);
    });
    whole_fixture(32, |f, quote, budget, control, _cache, a| {
        let (k, v) = f.parents();
        let p = Parents {
            keys: &k,
            values: &v,
        };
        let r = f.replacements();
        let mut c = capacity();
        c.max_states = 1;
        let mut metadata = MetadataLedger::new(c.max_work + quote.trace_work());
        let before = budget.budget().used_bytes();
        GRAPH_FACTORIES.with(|n| n.set(0));
        assert!(matches!(
            analyze_append(
                geometry(),
                a,
                &f.original,
                &p,
                &r,
                quote,
                budget,
                &mut metadata,
                control,
                c
            ),
            Err(Refusal::Trace)
        ));
        assert_eq!(GRAPH_FACTORIES.with(|n| n.get()), 1);
        assert_eq!(metadata.remaining(), 0);
        assert_eq!(budget.budget().used_bytes(), before);
    });
}

#[test]
fn public_constructor_owner_exact_counter_and_whole_result_retirement() {
    whole_fixture(8, |f, quote, budget, control, _cache, a| {
        let (k, v) = f.parents();
        let p = Parents {
            keys: &k,
            values: &v,
        };
        let r = f.replacements();
        let c = capacity();
        let bytes = owner_bytes(quote, c, factory_bytes(&f.original, &p, &r).unwrap()).unwrap();
        // Independent ambient root stays 64 MiB. Reserved unrelated headroom
        // establishes the edge; 64 MiB is NOT this append's required footprint.
        let _unavailable = budget
            .charge_external(
                (64 << 20)
                    - budget.budget().used_bytes()
                    - bytes
                    - 2 * size_of::<OwnedExternalCharge>(),
            )
            .unwrap();
        let baseline = budget.budget().used_bytes();
        let mut metadata = MetadataLedger::new(c.max_work + quote.trace_work());
        let plan = analyze_append(
            geometry(),
            a,
            &f.original,
            &p,
            &r,
            quote,
            budget,
            &mut metadata,
            control,
            c,
        )
        .unwrap();
        assert_eq!(budget.budget().used_bytes(), 64 << 20);
        assert_eq!(plan.graph.operations.len(), 5);
        drop(plan);
        assert_eq!(budget.budget().used_bytes(), baseline);
    });
}

#[test]
fn public_constructor_unknown_cost_error_stale_pair_and_descriptor_burn_without_factory() {
    for fault in 0..4 {
        whole_fixture(7, |f, quote, budget, control, _cache, a| {
            let (mut k, v) = f.parents();
            if fault == 2 {
                k[0] = "not_a_parent";
            }
            let p = Parents {
                keys: &k,
                values: &v,
            };
            let mut bad = f.caps[0].clone();
            if fault == 0 {
                let PublicCiphertextExecution::Ckks { compute, .. } = &mut bad.execution else {
                    unreachable!()
                };
                compute.cost.units = None;
            } else if fault == 1 {
                bad.output.error.bound = None;
            }
            let mut r = f.replacements();
            if fault <= 1 {
                r[0].capability = &bad;
            }
            let mut a = a;
            if fault == 3 {
                a.key_lane += 1;
            }
            let c = capacity();
            let mut metadata = MetadataLedger::new(c.max_work + quote.trace_work());
            let before = budget.budget().used_bytes();
            GRAPH_FACTORIES.with(|n| n.set(0));
            assert!(
                analyze_append(
                    geometry(),
                    a,
                    &f.original,
                    &p,
                    &r,
                    quote,
                    budget,
                    &mut metadata,
                    control,
                    c
                )
                .is_err()
            );
            assert_eq!(GRAPH_FACTORIES.with(|n| n.get()), 0);
            assert_eq!(metadata.remaining(), 0);
            assert_eq!(budget.budget().used_bytes(), before);
        });
    }
}

#[test]
fn dag_quote_rejects_changed_capacity_missing_retained_backing_and_stale_binary_manifest() {
    for fault in 0..3 {
        whole_fixture(7, |f, quote, budget, control, _cache, a| {
            let (k, v) = f.parents();
            let p = Parents {
                keys: &k,
                values: &v,
            };
            let r = f.replacements();
            let c = capacity();
            let mut metadata = MetadataLedger::new(c.max_work + quote.trace_work());
            let plan = analyze_append(
                geometry(),
                a,
                &f.original,
                &p,
                &r,
                quote,
                budget,
                &mut metadata,
                control,
                c,
            )
            .unwrap();
            control
                .core_work()
                .debit(0, c.trace.max_records * 128)
                .unwrap();
            // Existing counter covers clone/capacity transition BEFORE copying.
            let _copy = budget
                .charge_external(factory_bytes(&f.original, &p, &r).unwrap())
                .unwrap();
            let mut changed = plan.graph.clone();
            match fault {
                0 => changed.inputs.try_reserve_exact(1).unwrap(),
                1 => {
                    changed.retained_inputs.pop();
                }
                // Genuine stale arity manifest, not an authenticated identity claim.
                _ => {
                    changed.operations[0].inputs.pop();
                }
            }
            assert!(
                quote
                    .check_constructed(&changed, &plan.solution.ledger.trace, control)
                    .is_err()
            );
            assert_eq!(plan.graph.retained_inputs.len(), k.len() + v.len());
            assert_eq!(plan.graph.operations[0].inputs.len(), 2);
        });
    }
}

#[test]
fn shared_validator_still_rejects_retained_parent_release_and_binary_capability_mismatch() {
    whole_fixture(7, |f, quote, budget, control, _cache, a| {
        let (k, v) = f.parents();
        let p = Parents {
            keys: &k,
            values: &v,
        };
        let r = f.replacements();
        let c = capacity();
        let mut metadata = MetadataLedger::new(c.max_work + quote.trace_work());
        let plan = analyze_append(
            geometry(),
            a,
            &f.original,
            &p,
            &r,
            quote,
            budget,
            &mut metadata,
            control,
            c,
        )
        .unwrap();
        control
            .core_work()
            .debit(0, c.trace.max_records * 128)
            .unwrap();
        let _copies = budget
            .charge_external(factory_bytes(&f.original, &p, &r).unwrap() * 2)
            .unwrap();
        let mut bad_trace = plan.solution.ledger.trace.clone();
        bad_trace[0].release.push(f.keys[0].clone());
        assert!(validate_public_ciphertext_trace(&plan.graph, &bad_trace, c.trace).is_err());
        let mut bad_graph = plan.graph.clone();
        bad_graph.capabilities[0].inputs.pop();
        assert!(
            validate_public_ciphertext_trace(&bad_graph, &plan.solution.ledger.trace, c.trace)
                .is_err()
        );
        assert_eq!(plan.graph.capabilities[0].inputs.len(), 2);
    });
}

#[test]
fn dag_entry_rejects_missing_retained_id_backing_in_independent_manifest() {
    whole_fixture(7, |f, _quote, budget, control, _cache, _a| {
        control.core_work().debit(0, 8192).unwrap();
        let _region = budget.charge_external(128 << 10).unwrap();
        let mut mf = ManifestFixture::new(f, 7);
        let parents = f.keys.len() + f.values.len();
        // Last twenty records are five (operation,capability,release-vector,id)
        // trace tuples; immediately preceding are retained-input id backings.
        let retained_start = mf.backings.len() - 20 - parents;
        assert!(matches!(mf.backings[retained_start].element, Element::Byte));
        mf.backings.remove(retained_start);
        let capabilities = mf
            .cap_ranks
            .iter()
            .map(|r| CapabilityCount {
                input_ranks: r,
                output_rank: 1,
                execution: Execution::Ckks,
            })
            .collect::<Vec<_>>();
        mf.other.push(OtherOwner {
            bytes: Some(128 << 10),
            metadata: true,
            first_phase: 0,
            end_phase: 2,
        });
        let manifest = PublicCiphertextConstructionManifest {
            owners: f.original.owners.len() + 5,
            persistent_owners: 0,
            input_ranks: &mf.input_ranks,
            operations: &mf.operations,
            capabilities: &capabilities,
            outputs: parents + 5,
            retained_inputs: parents,
            release_lengths: &mf.releases,
            backings: &mf.backings,
            other_owners: &mf.other,
            phases: 2,
            manifest_original_bytes: Some(mf.original_bytes(&capabilities)),
        };
        GRAPH_FACTORIES.with(|n| n.set(0));
        assert!(quote_public_ciphertext_dag_construction(&manifest, control).is_err());
        assert_eq!(GRAPH_FACTORIES.with(|n| n.get()), 0);
    });
}
