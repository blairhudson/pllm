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
    let key_words = g.key_layout().storage_words().unwrap();
    let value_words = g.value_layout().storage_words().unwrap();
    let slot_bytes = key_words
        .checked_add(value_words)
        .and_then(|words| words.checked_mul(size_of::<i64>()))
        .unwrap();
    let actual = factory_bytes(&f.original, &p, &replacements)
        .unwrap()
        .checked_add(original_bytes)
        .and_then(|bytes| bytes.checked_add(slot_bytes))
        .unwrap();
    assert!(
        actual <= 128 << 10,
        "upstream region includes ORIGINAL capacities/controls, no unpriced manifest acquisition"
    );
    let mut keys = vec![0; key_words];
    let mut values = vec![0; value_words];
    let mut cache = RetainedCache::new(g, &mut keys, &mut values).unwrap();
    control.core_work().debit(active * 8 * 8, 0).unwrap();
    for token in 0..active {
        cache
            .append(&[token as i64 + 1; 8], &[token as i64 + 2; 8])
            .unwrap();
    }
    assert_eq!(cache.append_descriptor().unwrap(), append_at(active));
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
                    assert!(analysis.graph.retained_inputs.iter().any(|s| s == id));
                    assert!(analysis.graph.outputs.iter().any(|s| s == id));
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

// AstraC41: source-only synthetic producer composition. No encrypted producer
// implementation, generated prefix activation or numeric certificate is implied.
fn produced_geometry() -> Geometry {
    Geometry::new(64, 1, 1, 65).unwrap()
}
fn produced_descriptor() -> AppendDescriptor {
    AppendDescriptor {
        token: 7,
        key_tile: 0,
        key_lane: 7,
        new_key_ciphertexts: 0,
        value_group: 0,
        new_value_ciphertexts: 0,
    }
}
fn produced_fixture() -> Fixture {
    let mut f = Fixture::new(7);
    f.keys.truncate(1);
    f.values.truncate(1);
    f.incoming.truncate(2);
    f.outputs.truncate(2);
    f.owners.truncate(2);
    f.caps.truncate(2);
    for (i, c) in f.caps.iter_mut().enumerate() {
        c.kernel_identity = format!("cachemir_append_v1:64:1:1:65:7:{i}");
    }
    let mut original = empty();
    original.inputs = vec![
        val(f.keys[0].clone()),
        val(f.values[0].clone()),
        val("seed".into()),
    ];
    for id in [
        &f.keys[0],
        &f.values[0],
        &"seed".to_string(),
        &f.incoming[0],
        &f.incoming[1],
    ] {
        original.owners.push(PublicCiphertextOwner {
            identity: id.clone(),
            bytes: Some(512),
        });
    }
    // Deliberately reverse producer declaration order. V depends on K, so DAG
    // must not treat either result as an already-live original input.
    for i in [1, 0] {
        let mut cap = f.caps[i].clone();
        cap.identity = format!("producer_cap_{i}");
        cap.kernel_identity = format!("synthetic_projection_{i}");
        cap.inputs = vec![state()];
        cap.join = PublicCiphertextJoin::Unary;
        original.operations.push(PublicCiphertextGraphOperation {
            identity: format!("produce_{i}"),
            kernel_identity: cap.kernel_identity.clone(),
            primitive: PublicPrimitive::Arithmetic,
            inputs: vec![if i == 0 {
                "seed".into()
            } else {
                f.incoming[0].clone()
            }],
            output: val(f.incoming[i].clone()),
        });
        original.capabilities.push(cap);
    }
    f.original = original;
    f
}
fn produced_replacements(f: &Fixture) -> Vec<Replacement<'_>> {
    (0..2)
        .map(|i| Replacement {
            incoming: &f.incoming[i],
            output: &f.outputs[i],
            owner: &f.owners[i],
            capability: &f.caps[i],
        })
        .collect()
}
fn produced_updates(f: &Fixture) -> Vec<ProducedUpdate<'_>> {
    [1, 0]
        .into_iter()
        .map(|i| ProducedUpdate {
            producer: &f.original.operations[i],
            capability: &f.original.capabilities[i],
        })
        .collect()
}

/// Hand-authored expected four-operation graph; never calls append emission.
fn produced_expected(f: &Fixture) -> PublicCiphertextGraph {
    let mut graph = f.original.clone();
    for i in 0..2 {
        graph.owners.push(f.owners[i].clone());
        graph.capabilities.push(f.caps[i].clone());
        graph.operations.push(PublicCiphertextGraphOperation {
            identity: f.outputs[i].identity.clone(),
            kernel_identity: f.caps[i].kernel_identity.clone(),
            primitive: PublicPrimitive::Arithmetic,
            inputs: vec![
                if i == 0 {
                    f.keys[0].clone()
                } else {
                    f.values[0].clone()
                },
                f.incoming[i].clone(),
            ],
            output: f.outputs[i].clone(),
        });
    }
    graph.outputs = vec![
        f.keys[0].clone(),
        f.values[0].clone(),
        f.outputs[0].identity.clone(),
        f.outputs[1].identity.clone(),
    ];
    graph.retained_inputs = vec![f.keys[0].clone(), f.values[0].clone()];
    graph.owners.shrink_to_fit();
    graph.operations.shrink_to_fit();
    graph.capabilities.shrink_to_fit();
    graph
}

// Independent source graph census, not production emission/search output. The
// expected trace is literal [K producer,V producer,K append,V append].
fn produced_manifest(f: &Fixture) -> ManifestFixture {
    fn backing(v: &mut Vec<Backing>, element: Element, length: usize) {
        v.push(Backing {
            element,
            length,
            capacity: Some(length),
            first_phase: 0,
            end_phase: 2,
        });
    }
    fn text(v: &mut Vec<Backing>, s: &str) {
        backing(v, Element::Byte, s.len());
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
        backing(v, Element::Shape, s.shape.len());
    }
    fn value(v: &mut Vec<Backing>, x: &PublicCiphertextValue) {
        text(v, &x.identity);
        text(v, &x.owner);
        state(v, &x.state);
    }
    let g = produced_expected(f);
    let mut b = Vec::new();
    text(&mut b, &g.cost_model_identity);
    text(&mut b, &g.cost_unit);
    for (e, n) in [
        (Element::Owner, 7),
        (Element::String, 0),
        (Element::Value, 3),
        (Element::Operation, 4),
        (Element::Capability, 4),
        (Element::String, 4),
        (Element::String, 2),
        (Element::Step, 4),
    ] {
        backing(&mut b, e, n);
    }
    for o in &g.owners {
        text(&mut b, &o.identity);
    }
    for x in &g.inputs {
        value(&mut b, x);
    }
    for o in &g.operations {
        text(&mut b, &o.identity);
        text(&mut b, &o.kernel_identity);
        backing(&mut b, Element::String, o.inputs.len());
        for s in &o.inputs {
            text(&mut b, s);
        }
        value(&mut b, &o.output);
    }
    for c in &g.capabilities {
        text(&mut b, &c.identity);
        text(&mut b, &c.kernel_identity);
        backing(&mut b, Element::State, c.inputs.len());
        for s in &c.inputs {
            state(&mut b, s);
        }
        state(&mut b, &c.output);
        text(&mut b, c.error_assumption.as_ref().unwrap());
        let PublicCiphertextExecution::Ckks {
            compute,
            transition,
            ..
        } = &c.execution
        else {
            unreachable!()
        };
        text(&mut b, &compute.cost.identity);
        text(&mut b, &transition.cost.identity);
    }
    for s in g.outputs.iter().chain(&g.retained_inputs) {
        text(&mut b, s);
    }
    for (op, cap, release) in [
        ("produce_0", "producer_cap_0", Some("seed")),
        ("produce_1", "producer_cap_1", None),
        ("next_0", "append_cap_0", Some("delta_0")),
        ("next_1", "append_cap_1", Some("delta_1")),
    ] {
        text(&mut b, op);
        text(&mut b, cap);
        backing(&mut b, Element::String, usize::from(release.is_some()));
        if let Some(s) = release {
            text(&mut b, s);
        }
    }
    ManifestFixture {
        input_ranks: vec![1; 3],
        operations: g
            .operations
            .iter()
            .map(|o| OperationCount {
                inputs: o.inputs.len(),
                output_rank: 1,
            })
            .collect(),
        cap_ranks: g
            .capabilities
            .iter()
            .map(|c| vec![1; c.inputs.len()])
            .collect(),
        backings: b,
        releases: vec![1, 0, 1, 1],
        other: vec![OtherOwner {
            bytes: Some(128 << 10),
            metadata: true,
            first_phase: 0,
            end_phase: 2,
        }],
    }
}
fn produced_capacity() -> PublicEncryptedDagCapacity {
    PublicEncryptedDagCapacity {
        max_operations: 4,
        max_states: 32,
        max_frontier: 32,
        // Independent census: complete trace 282, plus four unused release
        // slots in the DAG envelope (3 inputs + 4 outputs versus 3 releases).
        // This test-only ceiling is not a production cap. Keep the SAME original
        // cumulative ledger; do not increase its grant to fit repeated faults.
        trace: PublicCiphertextTraceCapacity {
            max_records: 286,
            ..capacity().trace
        },
        ..capacity()
    }
}
fn with_produced_quote(
    run: impl FnOnce(
        &Fixture,
        &PublicCiphertextConstructionQuote<'_>,
        &OwnedBudget,
        &mut ControlLedger<'_>,
        usize,
    ),
) {
    let mut work = WorkLedger::new(1 << 28, 1 << 20);
    let mut control = ControlLedger::new(&mut work);
    control.core_work().debit(0, 8192).unwrap();
    let budget = OwnedBudget::new(64 << 20).unwrap();
    let _source = budget.charge_external(128 << 10).unwrap();
    let f = produced_fixture();
    let mf = produced_manifest(&f);
    let caps = mf
        .cap_ranks
        .iter()
        .map(|r| CapabilityCount {
            input_ranks: r,
            output_rank: 1,
            execution: Execution::Ckks,
        })
        .collect::<Vec<_>>();
    let manifest = PublicCiphertextConstructionManifest {
        owners: 7,
        persistent_owners: 0,
        input_ranks: &mf.input_ranks,
        operations: &mf.operations,
        capabilities: &caps,
        outputs: 4,
        retained_inputs: 2,
        release_lengths: &mf.releases,
        backings: &mf.backings,
        other_owners: &mf.other,
        phases: 2,
        manifest_original_bytes: Some(mf.original_bytes(&caps)),
    };
    let (k, v) = f.parents();
    assert!(
        factory_bytes(
            &f.original,
            &Parents {
                keys: &k,
                values: &v
            },
            &produced_replacements(&f)
        )
        .unwrap()
            + mf.original_bytes(&caps)
            < 128 << 10
    );
    let quote = quote_public_ciphertext_dag_construction(&manifest, &mut control).unwrap();
    // Independently authored expanded census, not a production calculator:
    // root; owner/persistent/input/op/cap/output/retained lists; step fields;
    // releases; input values; unary/binary operations; unary/binary capabilities.
    let complete_records = 1 + 7 + 0 + 3 + 4 + 4 + 4 + 2 + 12 + 3
        + 3 * (9 + 1)
        + 2 * (3 + 1 + 9 + 1)
        + 2 * (3 + 2 + 9 + 1)
        + 2 * (12 + 1 + 9 + 1 + 9 + 1)
        + 2 * (12 + 2 + 9 + 1 + 2 * (9 + 1));
    assert_eq!(complete_records, 282);
    assert_eq!(quote.records(), 282);
    assert_eq!(complete_records - 3 + (3 + 4), 286);
    assert_eq!(produced_capacity().trace.max_records, 286);
    run(&f, &quote, &budget, &mut control, mf.backings.len());
}

#[test]
fn produced_quote_trace_record_boundary_282_admits_281_refuses_without_debit() {
    with_produced_quote(|_, quote, _, _, _| {
        for records in [281, 282] {
            let mut metadata = MetadataLedger::new(quote.trace_work());
            let trace = PublicCiphertextTraceCapacity {
                max_records: records,
                ..produced_capacity().trace
            };
            let result = quote.admit_trace(Some(trace), &mut metadata);
            if records == 281 {
                assert!(matches!(
                    result,
                    Err(schedule::construction::Refusal::MetadataWorkRefused)
                ));
            } else {
                assert!(result.is_ok());
            }
            drop(result);
            assert_eq!(
                metadata.remaining(),
                if records == 281 { quote.trace_work() } else { 0 }
            );
        }
    });
}

#[test]
fn produced_trace_record_281_refusal_prepays_search_without_matching_or_factory() {
    with_produced_quote(|f, quote, budget, control, _| {
        let (k, v) = f.parents();
        let p = Parents {
            keys: &k,
            values: &v,
        };
        let r = produced_replacements(f);
        let updates = produced_updates(f);
        let mut c = produced_capacity();
        c.trace.max_records = 281;
        let construction = control_allowance(&f.original, &p, &r, 281).unwrap();
        let mut binding = bind_produced_updates(
            produced_geometry(),
            produced_descriptor(),
            &f.original,
            p,
            &r,
            &updates,
        );
        let mut metadata = MetadataLedger::new(c.max_work + quote.trace_work());
        let before = budget.budget().used_bytes();
        let work_before = control.core_work().remaining();
        let control_before = control.core_work().control_remaining();
        GRAPH_FACTORIES.with(|n| n.set(0));
        MATCH_PREDICATES.with(|n| n.set(0));
        assert!(matches!(
            analyze_produced_append(&mut binding, quote, budget, &mut metadata, control, c),
            Err(Refusal::Work)
        ));
        assert!(binding.pending.is_none());
        assert_eq!(metadata.remaining(), quote.trace_work());
        assert_eq!(control.core_work().remaining(), work_before - construction);
        assert_eq!(
            control.core_work().control_remaining(),
            control_before - construction
        );
        assert_eq!(GRAPH_FACTORIES.with(|n| n.get()), 0);
        assert_eq!(MATCH_PREDICATES.with(|n| n.get()), 0);
        assert_eq!(budget.budget().used_bytes(), before);
        // A normal-capacity retry cannot recover the consumed binding or debit.
        assert!(matches!(
            analyze_produced_append(
                &mut binding,
                quote,
                budget,
                &mut metadata,
                control,
                produced_capacity()
            ),
            Err(Refusal::Consumed)
        ));
        assert_eq!(metadata.remaining(), quote.trace_work());
        assert_eq!(control.core_work().remaining(), work_before - construction);
        assert_eq!(
            control.core_work().control_remaining(),
            control_before - construction
        );
        assert_eq!(GRAPH_FACTORIES.with(|n| n.get()), 0);
        assert_eq!(MATCH_PREDICATES.with(|n| n.get()), 0);
        assert_eq!(budget.budget().used_bytes(), before);
    });
}

#[test]
fn produced_ambiguous_capability_refuses_before_factory_and_burns_binding() {
    // Authored regressions, UNRUN: equal execution/cost/state, distinct identity
    // in original inventory; then appended inventory, including repeated borrow.
    for (fault, short) in (0..3).flat_map(|fault| [false, true].map(|short| (fault, short))) {
        with_produced_quote(|_, quote, budget, control, _| {
            let mut f = produced_fixture();
            if fault == 0 {
                let mut competing = f.original.capabilities[1].clone();
                competing.identity = "competing_producer_cap_0".into();
                f.original.capabilities.push(competing);
            } else {
                // Same aligned producer geometry as K append: appended K cap
                // now matches producer too, without changing any numeric bits.
                let mut cap = f.caps[0].clone();
                cap.identity = f.original.capabilities[1].identity.clone();
                f.original.operations[1].kernel_identity = cap.kernel_identity.clone();
                f.original.operations[1].inputs = vec![f.keys[0].clone(), "seed".into()];
                f.original.capabilities[1] = cap;
            }
            let (k, v) = f.parents();
            let p = Parents {
                keys: &k,
                values: &v,
            };
            let mut r = produced_replacements(&f);
            let mut updates = produced_updates(&f);
            if fault == 0 {
                // Bind clone at index 2, not existing equal-cost match at 1.
                updates[0].capability = &f.original.capabilities[2];
            } else if fault == 2 {
                // Same pointer still emits a second capability occurrence.
                r[0].capability = updates[0].capability;
            }
            assert!(schedule::dag::matches(
                &f.original,
                updates[0].producer,
                updates[0].capability
            ));
            let competitor = if fault == 0 {
                &f.original.capabilities[1]
            } else {
                r[0].capability
            };
            assert!(schedule::dag::matches(
                &f.original,
                updates[0].producer,
                competitor
            ));
            assert_eq!(competitor.execution, updates[0].capability.execution);
            // Collision is not a stale/invalid replacement witness.
            preflight_sources(
                produced_geometry(),
                produced_descriptor(),
                &f.original,
                &p,
                &r,
                true,
            )
            .unwrap();
            let c = produced_capacity();
            let construction = control_allowance(&f.original, &p, &r, c.trace.max_records).unwrap();
            let mut binding = bind_produced_updates(
                produced_geometry(),
                produced_descriptor(),
                &f.original,
                p,
                &r,
                &updates,
            );
            let bound = binding.pending.as_ref().unwrap();
            produced_preflight(bound).unwrap();
            let matching = produced_matching_work(bound).unwrap();
            let required = construction + matching;
            // Same original setup ledger; exact Q reaches ambiguity, Q-1
            // refuses atomically before any shared matching predicate visit.
            let excess =
                control.core_work().control_remaining() - (required - usize::from(short));
            control.core_work().debit(0, excess).unwrap();
            let mut metadata = MetadataLedger::new(c.max_work + quote.trace_work());
            let before = budget.budget().used_bytes();
            let work_before = control.core_work().remaining();
            let control_before = control.core_work().control_remaining();
            GRAPH_FACTORIES.with(|n| n.set(0));
            MATCH_PREDICATES.with(|n| n.set(0));
            let refusal = if short {
                Refusal::Work
            } else {
                Refusal::Capability
            };
            let error =
                analyze_produced_append(&mut binding, quote, budget, &mut metadata, control, c)
                    .err()
                    .expect("ambiguous capability or short matching grant must refuse");
            assert_eq!(error, refusal, "fault={fault}, short={short}");
            let predicates = MATCH_PREDICATES.with(|n| n.get());
            assert_eq!(predicates == 0, short);
            let spent = construction + if short { 0 } else { matching };
            assert_eq!(GRAPH_FACTORIES.with(|n| n.get()), 0);
            assert_eq!(budget.budget().used_bytes(), before);
            assert_eq!(metadata.remaining(), 0);
            assert_eq!(control.core_work().remaining(), work_before - spent);
            assert_eq!(
                control.core_work().control_remaining(),
                control_before - spent
            );
            assert!(matches!(
                analyze_produced_append(&mut binding, quote, budget, &mut metadata, control, c),
                Err(Refusal::Consumed)
            ));
            assert_eq!(GRAPH_FACTORIES.with(|n| n.get()), 0);
            assert_eq!(budget.budget().used_bytes(), before);
            assert_eq!(metadata.remaining(), 0);
            assert_eq!(MATCH_PREDICATES.with(|n| n.get()), predicates);
            assert_eq!(control.core_work().remaining(), work_before - spent);
            assert_eq!(
                control.core_work().control_remaining(),
                control_before - spent
            );
        });
    }
}

#[test]
fn produced_unique_capability_selects_bound_original() {
    with_produced_quote(|f, quote, budget, control, backings| {
        let (k, v) = f.parents();
        let r = produced_replacements(f);
        let updates = produced_updates(f);
        let mut binding = bind_produced_updates(
            produced_geometry(),
            produced_descriptor(),
            &f.original,
            Parents {
                keys: &k,
                values: &v,
            },
            &r,
            &updates,
        );
        let c = produced_capacity();
        let bound = binding.pending.as_ref().unwrap();
        produced_preflight(bound).unwrap();
        let matching = produced_matching_work(bound).unwrap();
        assert_eq!(matching, 8720, "U=2, C=4, V=5, S=2");
        let construction =
            control_allowance(&f.original, &bound.parents, &r, c.trace.max_records).unwrap();
        let required = construction + matching + 1024 + 4 * backings;
        let work_before = control.core_work().remaining();
        let control_before = control.core_work().control_remaining();
        let mut metadata = MetadataLedger::new(c.max_work + quote.trace_work());
        let before = budget.budget().used_bytes();
        GRAPH_FACTORIES.with(|n| n.set(0));
        MATCH_PREDICATES.with(|n| n.set(0));
        let plan = analyze_produced_append(&mut binding, quote, budget, &mut metadata, control, c)
            .unwrap();
        assert_eq!(GRAPH_FACTORIES.with(|n| n.get()), 1);
        let predicates = MATCH_PREDICATES.with(|n| n.get());
        assert!(predicates > 0);
        assert_eq!(control.core_work().remaining(), work_before - required);
        assert_eq!(
            control.core_work().control_remaining(),
            control_before - required
        );
        assert!(std::ptr::eq(plan.original(), &f.original));
        for update in &updates {
            let operation_index = f
                .original
                .operations
                .iter()
                .position(|op| std::ptr::eq(op, update.producer))
                .unwrap();
            let capability_index = f
                .original
                .capabilities
                .iter()
                .position(|cap| std::ptr::eq(cap, update.capability))
                .unwrap();
            let selected = plan
                .solution()
                .ledger
                .selections
                .iter()
                .find(|selected| selected.operation_index == operation_index)
                .unwrap();
            assert_eq!(selected.capability_index, capability_index);
            assert_eq!(
                &plan.graph().operations[selected.operation_index],
                update.producer
            );
            assert_eq!(
                &plan.graph().capabilities[selected.capability_index],
                update.capability
            );
            assert_eq!(
                plan.graph().capabilities[selected.capability_index].identity,
                update.capability.identity
            );
        }
        drop(plan);
        assert_eq!(metadata.remaining(), 0);
        assert_eq!(budget.budget().used_bytes(), before);
        assert!(matches!(
            analyze_produced_append(&mut binding, quote, budget, &mut metadata, control, c),
            Err(Refusal::Consumed)
        ));
        assert_eq!(control.core_work().remaining(), work_before - required);
        assert_eq!(
            control.core_work().control_remaining(),
            control_before - required
        );
        assert_eq!(metadata.remaining(), 0);
        assert_eq!(MATCH_PREDICATES.with(|n| n.get()), predicates);
        assert_eq!(GRAPH_FACTORIES.with(|n| n.get()), 1);
        assert_eq!(budget.budget().used_bytes(), before);
    });
}

#[test]
fn produced_pair_public_analysis_matches_hand_graph_and_all_legal_schedules() {
    with_produced_quote(|f, quote, budget, control, _| {
        let (k, v) = f.parents();
        let r = produced_replacements(f);
        let updates = produced_updates(f);
        let mut binding = bind_produced_updates(
            produced_geometry(),
            produced_descriptor(),
            &f.original,
            Parents {
                keys: &k,
                values: &v,
            },
            &r,
            &updates,
        );
        let c = produced_capacity();
        let mut metadata = MetadataLedger::new(c.max_work + quote.trace_work());
        let before = budget.budget().used_bytes();
        let plan = analyze_produced_append(&mut binding, quote, budget, &mut metadata, control, c)
            .unwrap();
        assert_eq!(plan.graph(), &produced_expected(f));
        assert!(std::ptr::eq(plan.original(), &f.original));
        assert_eq!(
            plan.descriptor(),
            (produced_geometry(), produced_descriptor())
        );
        // All 24 distinct permutations reach the shared validator. Dependency
        // inequalities below describe an independent oracle, NEVER a prefilter.
        // Capabilities, edges and retirement names are hand-authored, not copied
        // from the solver trace. Both consumers must finish before K retirement.
        let expected = produced_expected(f);
        let trace_capacity = c.trace;
        let mut legal = Vec::new();
        let (mut submitted, mut rejected) = (0, 0);
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    for d in 0..4 {
                        let order = [a, b, c, d];
                        if (0..4).any(|i| order[..i].contains(&order[i])) {
                            continue;
                        }
                        let pos = |n| order.iter().position(|x| *x == n).unwrap();
                        let dependencies_hold =
                            pos(1) < pos(0) && pos(1) < pos(2) && pos(0) < pos(3);
                        let mut trace = Vec::new();
                        for op in order {
                            let (operation, capability) = [
                                ("produce_1", "producer_cap_1"),
                                ("produce_0", "producer_cap_0"),
                                ("next_0", "append_cap_0"),
                                ("next_1", "append_cap_1"),
                            ][op];
                            let mut release = Vec::new();
                            if op == 1 {
                                release.push("seed".into());
                            }
                            if op == 3 {
                                release.push("delta_1".into());
                            }
                            if (op == 0 && pos(2) < pos(0)) || (op == 2 && pos(0) < pos(2)) {
                                release.push("delta_0".into());
                            }
                            trace.push(PublicCiphertextTraceStep {
                                operation: operation.into(),
                                capability: capability.into(),
                                release,
                            });
                        }
                        submitted += 1;
                        let summary = match validate_public_ciphertext_trace(
                            &expected,
                            &trace,
                            trace_capacity,
                        ) {
                            Ok(summary) => {
                                assert!(
                                    dependencies_hold,
                                    "validator accepted dependency violation: {order:?}"
                                );
                                summary
                            }
                            Err(error) => {
                                assert!(
                                    !dependencies_hold,
                                    "validator rejected legal order {order:?}: {error}"
                                );
                                assert_eq!(
                                    error,
                                    "unavailable parent (cycle, early release or type mismatch)",
                                    "wrong rejection category for {order:?}"
                                );
                                rejected += 1;
                                continue;
                            }
                        };
                        let mut live = [true, true, true, false, false, false, false];
                        let mut peak = 1536;
                        for op in order {
                            let output = [4, 3, 5, 6][op];
                            live[output] = true;
                            peak = peak.max(live.iter().filter(|b| **b).count() * 512);
                            if op == 1 {
                                live[2] = false;
                            }
                            if op == 3 {
                                live[4] = false;
                            }
                            if (op == 0 && pos(2) < pos(0)) || (op == 2 && pos(0) < pos(2)) {
                                live[3] = false;
                            }
                            assert!(live[0] && live[1], "complete old parents never die");
                        }
                        assert_eq!(summary.peak_live_owner_bytes, peak as u64);
                        assert_eq!(summary.final_live_backing_bytes, 2048);
                        assert_eq!(summary.total_cost, 4);
                        legal.push((order, peak, summary.total_cost));
                    }
                }
            }
        }
        // Same objective (cost then lexical operation order), derived from all
        // validator-accepted traces, not hardcoded readiness or selected trace.
        legal.sort_by_key(|(order, _, cost)| (*cost, *order));
        assert_eq!(submitted, 24);
        assert_eq!(legal.len(), 3);
        assert_eq!(rejected, 21);
        let selected = plan
            .solution()
            .ledger
            .selections
            .iter()
            .map(|s| s.operation_index)
            .collect::<Vec<_>>();
        assert_eq!(selected, legal[0].0);
        assert_eq!(plan.solution().ledger.total_cost, legal[0].2);
        assert_eq!(
            plan.solution().trace_summary.peak_live_owner_bytes,
            legal[0].1 as u64
        );
        assert_eq!(plan.solution().trace_summary.final_live_backing_bytes, 2048);
        for step in &plan.solution().ledger.trace {
            assert!(
                !step.release.contains(&k[0].to_owned())
                    && !step.release.contains(&v[0].to_owned())
            );
        }
        drop(plan);
        assert_eq!(budget.budget().used_bytes(), before);
        assert_eq!(metadata.remaining(), 0);
        assert!(matches!(
            analyze_produced_append(&mut binding, quote, budget, &mut metadata, control, c),
            Err(Refusal::Consumed)
        ));
    });
}

#[test]
fn produced_binding_foreign_same_names_swaps_duplicates_stale_and_incomplete_refuse_before_factory()
{
    with_produced_quote(|f, quote, budget, control, _| {
        let foreign = produced_fixture();
        let (k, v) = f.parents();
        let r = produced_replacements(f);
        let c = produced_capacity();
        let mut metadata = MetadataLedger::new(6 * (c.max_work + quote.trace_work()));
        for fault in 0..6 {
            let mut updates = produced_updates(f);
            let mut descriptor = produced_descriptor();
            match fault {
                0 => updates[0].producer = &foreign.original.operations[1],
                1 => updates.swap(0, 1),
                2 => updates[1].producer = updates[0].producer,
                3 => descriptor.token += 1,
                4 => {
                    updates.pop();
                }
                _ => updates[0].capability = &foreign.original.capabilities[1],
            }
            let mut binding = bind_produced_updates(
                produced_geometry(),
                descriptor,
                &f.original,
                Parents {
                    keys: &k,
                    values: &v,
                },
                &r,
                &updates,
            );
            let before = budget.budget().used_bytes();
            GRAPH_FACTORIES.with(|n| n.set(0));
            assert!(
                analyze_produced_append(&mut binding, quote, budget, &mut metadata, control, c)
                    .is_err()
            );
            assert_eq!(GRAPH_FACTORIES.with(|n| n.get()), 0);
            assert_eq!(budget.budget().used_bytes(), before);
            assert!(matches!(
                analyze_produced_append(&mut binding, quote, budget, &mut metadata, control, c),
                Err(Refusal::Consumed)
            ));
        }
        assert_eq!(metadata.remaining(), 0);
    });
}

#[test]
fn produced_unknown_cost_error_missing_capability_ambiguous_output_and_parent_refuse_before_factory()
 {
    with_produced_quote(|_, quote, budget, control, _| {
        let c = produced_capacity();
        let mut metadata = MetadataLedger::new(8 * (c.max_work + quote.trace_work()));
        for fault in 0..8 {
            let mut f = produced_fixture();
            match fault {
                0 => {
                    let PublicCiphertextExecution::Ckks { compute, .. } =
                        &mut f.original.capabilities[0].execution
                    else {
                        unreachable!()
                    };
                    compute.cost.units = None;
                }
                1 => f.original.operations[0].output.state.error.bound = None,
                2 => f.original.capabilities[0].kernel_identity = "missing_capability".into(),
                3 => f
                    .original
                    .inputs
                    .push(f.original.operations[1].output.clone()),
                4 => {
                    f.values.clear();
                }
                5 => f.original.capabilities[0].inputs[0].key_domain = "wrong_domain".into(),
                6 => f.original.capabilities[0].join = PublicCiphertextJoin::Aligned,
                _ => f.original.owners[2].bytes = None,
            }
            let (k, v) = f.parents();
            let r = produced_replacements(&f);
            let updates = produced_updates(&f);
            let mut binding = bind_produced_updates(
                produced_geometry(),
                produced_descriptor(),
                &f.original,
                Parents {
                    keys: &k,
                    values: &v,
                },
                &r,
                &updates,
            );
            GRAPH_FACTORIES.with(|n| n.set(0));
            assert!(
                analyze_produced_append(&mut binding, quote, budget, &mut metadata, control, c)
                    .is_err()
            );
            assert_eq!(GRAPH_FACTORIES.with(|n| n.get()), 0);
        }
        assert_eq!(metadata.remaining(), 0);
    });
}

#[test]
fn produced_append_retains_complete_old_families_including_unchanged_tiles() {
    let mut f = produced_fixture();
    f.original.outputs.push(f.incoming[0].clone());
    f.original.retained_inputs.push("seed".into());
    for (names, id) in [(&mut f.keys, "old_k_1"), (&mut f.values, "old_v_1")] {
        names.push(id.into());
        f.original.inputs.push(val(id.into()));
        f.original.owners.push(PublicCiphertextOwner {
            identity: id.into(),
            bytes: Some(512),
        });
    }
    for (i, cap) in f.caps.iter_mut().enumerate() {
        cap.kernel_identity = format!("cachemir_append_v1:64:1:1:129:65:{i}");
    }
    let g = Geometry::new(64, 1, 1, 129).unwrap();
    let a = AppendDescriptor {
        token: 65,
        key_tile: 1,
        key_lane: 1,
        value_group: 1,
        new_key_ciphertexts: 0,
        new_value_ciphertexts: 0,
    };
    let (k, v) = f.parents();
    let r = produced_replacements(&f);
    let updates = produced_updates(&f);
    let bound = ProducedAppend {
        geometry: g,
        append: a,
        original: &f.original,
        parents: Parents {
            keys: &k,
            values: &v,
        },
        replacements: &r,
        updates: &updates,
    };
    produced_preflight(&bound).unwrap();
    assert!(
        preflight(g, a, &f.original, &bound.parents, &r).is_err(),
        "legacy prepacked-input API must not silently acquire producer authority"
    );
    let graph = emit_graph(g, a, &f.original, &bound.parents, &r).unwrap();
    assert_eq!(graph.operations[2].inputs[0], "old_k_1");
    assert_eq!(graph.operations[3].inputs[0], "old_v_1");
    // Independent expanded census for this larger fixture only. Nine owners,
    // five inputs, seven outputs, five retained inputs, and one actual release
    // (delta_1). DAG envelope allows all five input + four output releases.
    let complete_records = 1 + 9 + 0 + 5 + 4 + 4 + 7 + 5 + 12 + 1
        + 5 * (9 + 1)
        + 2 * (3 + 1 + 9 + 1)
        + 2 * (3 + 2 + 9 + 1)
        + 2 * (12 + 1 + 9 + 1 + 9 + 1)
        + 2 * (12 + 2 + 9 + 1 + 2 * (9 + 1));
    assert_eq!(complete_records, 310);
    assert_eq!(complete_records - 1 + (5 + 4), 318);
    let mut c = produced_capacity();
    c.trace.max_records = 318;
    let solution = solve_public_encrypted_dag(&graph, c).unwrap();
    let exact_trace = PublicCiphertextTraceCapacity {
        max_records: 310,
        ..c.trace
    };
    assert_eq!(
        validate_public_ciphertext_trace(&graph, &solution.ledger.trace, exact_trace).unwrap(),
        solution.trace_summary
    );
    assert!(
        validate_public_ciphertext_trace(
            &graph,
            &solution.ledger.trace,
            PublicCiphertextTraceCapacity {
                max_records: 309,
                ..exact_trace
            },
        )
        .is_err()
    );
    for id in k.iter().chain(&v) {
        assert!(graph.outputs.iter().any(|s| s == id));
        assert!(graph.retained_inputs.iter().any(|s| s == id));
        assert!(
            solution
                .ledger
                .trace
                .iter()
                .all(|s| !s.release.iter().any(|r| r == id))
        );
    }
    assert!(graph.outputs.contains(&f.incoming[0]));
    assert!(graph.retained_inputs.iter().any(|id| id == "seed"));
    assert!(
        solution.ledger.trace.iter().all(
            |s| !s.release.contains(&f.incoming[0]) && !s.release.iter().any(|id| id == "seed")
        )
    );
    assert_eq!(solution.trace_summary.final_live_backing_bytes, 8 * 512);
}

#[test]
fn produced_cumulative_admission_exact_fit_and_one_short_keep_original_ledgers() {
    for fault in 0..5 {
        with_produced_quote(|f, quote, budget, control, backings| {
            let (k, v) = f.parents();
            let p = Parents {
                keys: &k,
                values: &v,
            };
            let r = produced_replacements(f);
            let updates = produced_updates(f);
            let c = produced_capacity();
            let construction = control_allowance(&f.original, &p, &r, c.trace.max_records).unwrap();
            let validation = 1024 + 4 * backings;
            let bound = ProducedAppend {
                geometry: produced_geometry(),
                append: produced_descriptor(),
                original: &f.original,
                parents: Parents {
                    keys: &k,
                    values: &v,
                },
                replacements: &r,
                updates: &updates,
            };
            produced_preflight(&bound).unwrap();
            let matching = produced_matching_work(&bound).unwrap();
            let required = 2 * (construction + matching + validation);
            let remaining = control.core_work().control_remaining();
            control
                .core_work()
                .debit(0, remaining - required + usize::from(fault == 1))
                .unwrap();
            let total = c.max_work + quote.trace_work();
            let mut metadata = MetadataLedger::new(2 * total - usize::from(fault == 2));
            let bytes = owner_bytes(quote, c, factory_bytes(&f.original, &p, &r).unwrap()).unwrap();
            let _headroom = if fault >= 3 {
                Some(
                    budget
                        .charge_external(
                            (64 << 20)
                                - budget.budget().used_bytes()
                                - bytes
                                - 2 * size_of::<OwnedExternalCharge>()
                                + usize::from(fault == 4),
                        )
                        .unwrap(),
                )
            } else {
                None
            };
            let before = budget.budget().used_bytes();
            for attempt in 0..2 {
                let mut binding = bind_produced_updates(
                    produced_geometry(),
                    produced_descriptor(),
                    &f.original,
                    Parents {
                        keys: &k,
                        values: &v,
                    },
                    &r,
                    &updates,
                );
                GRAPH_FACTORIES.with(|n| n.set(0));
                MATCH_PREDICATES.with(|n| n.set(0));
                let result =
                    analyze_produced_append(&mut binding, quote, budget, &mut metadata, control, c);
                let fails = fault == 4 || attempt == 1 && (fault == 1 || fault == 2);
                assert_eq!(result.is_err(), fails);
                if fault == 3 {
                    assert_eq!(budget.budget().used_bytes(), 64 << 20);
                }
                if fault == 2 && attempt == 1 || fault == 4 {
                    assert_eq!(GRAPH_FACTORIES.with(|n| n.get()), 0);
                }
                if fault == 2 && attempt == 1 {
                    assert_eq!(MATCH_PREDICATES.with(|n| n.get()), 0);
                } else {
                    assert!(MATCH_PREDICATES.with(|n| n.get()) > 0);
                }
                if fault == 1 && attempt == 1 {
                    // Final validation refuses late, after emission/search.
                    assert!(matches!(result, Err(Refusal::Trace)));
                    assert_eq!(GRAPH_FACTORIES.with(|n| n.get()), 1);
                }
                drop(result);
                assert_eq!(budget.budget().used_bytes(), before);
                let metadata_after = metadata.remaining();
                let work_after = control.core_work().remaining();
                let control_after = control.core_work().control_remaining();
                let factories_after = GRAPH_FACTORIES.with(|n| n.get());
                let predicates_after = MATCH_PREDICATES.with(|n| n.get());
                assert!(matches!(
                    analyze_produced_append(&mut binding, quote, budget, &mut metadata, control, c),
                    Err(Refusal::Consumed)
                ));
                assert_eq!(budget.budget().used_bytes(), before);
                assert_eq!(metadata.remaining(), metadata_after);
                assert_eq!(control.core_work().remaining(), work_after);
                assert_eq!(control.core_work().control_remaining(), control_after);
                assert_eq!(GRAPH_FACTORIES.with(|n| n.get()), factories_after);
                assert_eq!(MATCH_PREDICATES.with(|n| n.get()), predicates_after);
            }
            if fault == 0 || fault == 3 {
                assert_eq!(metadata.remaining(), 0);
                assert_eq!(control.core_work().control_remaining(), 0);
            }
            if fault == 1 {
                assert_eq!(control.core_work().control_remaining(), validation - 1);
            }
            if fault == 2 {
                assert_eq!(metadata.remaining(), quote.trace_work() - 1);
            }
        });
    }
}

#[test]
fn produced_foreign_original_substitution_refuses_original_record_references_before_factory() {
    with_produced_quote(|f, quote, budget, control, _| {
        let foreign = produced_fixture();
        assert_eq!(
            foreign.original, f.original,
            "equal content, distinct graph allocation"
        );
        assert!(!std::ptr::eq(&foreign.original, &f.original));
        let (k, v) = f.parents();
        let r = produced_replacements(f);
        let updates = produced_updates(f);
        // Reverse direction of the foreign-record cases: substituted original,
        // but the producer AND capability references still belong to f.original.
        let mut binding = bind_produced_updates(
            produced_geometry(),
            produced_descriptor(),
            &foreign.original,
            Parents {
                keys: &k,
                values: &v,
            },
            &r,
            &updates,
        );
        let c = produced_capacity();
        let mut metadata = MetadataLedger::new(c.max_work + quote.trace_work());
        let _other_live_owner = budget.charge_external(4096).unwrap();
        let before = budget.budget().used_bytes();
        GRAPH_FACTORIES.with(|n| n.set(0));
        MATCH_PREDICATES.with(|n| n.set(0));
        assert!(matches!(
            analyze_produced_append(&mut binding, quote, budget, &mut metadata, control, c),
            Err(Refusal::StaleParents)
        ));
        assert_eq!(GRAPH_FACTORIES.with(|n| n.get()), 0);
        assert_eq!(MATCH_PREDICATES.with(|n| n.get()), 0);
        assert_eq!(budget.budget().used_bytes(), before);
        assert_eq!(metadata.remaining(), 0);
        let work_after = control.core_work().remaining();
        let control_after = control.core_work().control_remaining();
        assert!(matches!(
            analyze_produced_append(&mut binding, quote, budget, &mut metadata, control, c),
            Err(Refusal::Consumed)
        ));
        assert_eq!(budget.budget().used_bytes(), before);
        assert_eq!(metadata.remaining(), 0);
        assert_eq!(control.core_work().remaining(), work_after);
        assert_eq!(control.core_work().control_remaining(), control_after);
        assert_eq!(MATCH_PREDICATES.with(|n| n.get()), 0);
        assert_eq!(GRAPH_FACTORIES.with(|n| n.get()), 0);
        // Other guard remains in scope through every observation. Assertions
        // concern unchanged ownership only, not intermediate refund timing.
        drop(_other_live_owner);
    });
}

#[test]
fn produced_checked_header_overflow_spends_handle_without_factory_or_owner_changes() {
    with_produced_quote(|f, quote, budget, control, _| {
        let (k, v) = f.parents();
        let r = produced_replacements(f);
        let updates = produced_updates(f);
        let mut binding = bind_produced_updates(
            produced_geometry(),
            produced_descriptor(),
            &f.original,
            Parents {
                keys: &k,
                values: &v,
            },
            &r,
            &updates,
        );
        let normal = produced_capacity();
        let mut metadata = MetadataLedger::new(normal.max_work + quote.trace_work());
        let _other_live_owner = budget.charge_external(4096).unwrap();
        let before = budget.budget().used_bytes();
        let metadata_before = metadata.remaining();
        let work_before = control.core_work().remaining();
        let control_before = control.core_work().control_remaining();
        // Malformed boundary probe, NOT an admitted larger grant. No budget,
        // work allowance, or successful-path cap is increased. Header admission
        // must checked-reject R*128 overflow before debiting or visiting records.
        let malformed_records = usize::MAX;
        assert!(malformed_records.checked_mul(128).is_none());
        let malformed = PublicEncryptedDagCapacity {
            trace: PublicCiphertextTraceCapacity {
                max_records: malformed_records,
                ..normal.trace
            },
            ..normal
        };
        GRAPH_FACTORIES.with(|n| n.set(0));
        MATCH_PREDICATES.with(|n| n.set(0));
        assert!(matches!(
            analyze_produced_append(
                &mut binding,
                quote,
                budget,
                &mut metadata,
                control,
                malformed
            ),
            Err(Refusal::Capacity)
        ));
        assert_eq!(GRAPH_FACTORIES.with(|n| n.get()), 0);
        assert_eq!(budget.budget().used_bytes(), before);
        assert_eq!(metadata.remaining(), metadata_before);
        assert_eq!(control.core_work().remaining(), work_before);
        assert_eq!(control.core_work().control_remaining(), control_before);
        assert_eq!(MATCH_PREDICATES.with(|n| n.get()), 0);
        // Even retry with ordinary bounded inputs cannot replay failed handle.
        assert!(matches!(
            analyze_produced_append(&mut binding, quote, budget, &mut metadata, control, normal),
            Err(Refusal::Consumed)
        ));
        assert_eq!(GRAPH_FACTORIES.with(|n| n.get()), 0);
        assert_eq!(budget.budget().used_bytes(), before);
        assert_eq!(metadata.remaining(), metadata_before);
        assert_eq!(control.core_work().remaining(), work_before);
        assert_eq!(control.core_work().control_remaining(), control_before);
        assert_eq!(MATCH_PREDICATES.with(|n| n.get()), 0);
        drop(_other_live_owner);
    });
}
