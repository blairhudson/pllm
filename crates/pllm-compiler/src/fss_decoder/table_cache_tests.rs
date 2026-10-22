use super::*;
use pllm_models::{DecoderWorkload, ModelOperator};
use serde_json::json;

fn fixture() -> Program {
    let config = json!({"model_type":"gpt2","n_embd":2,"n_head":1,"n_layer":1,"n_positions":4,"vocab_size":3,"n_inner":2});
    let plan = pllm_models::lower_model_json(
        &serde_json::to_vec(&config).unwrap(),
        DecoderWorkload {
            batch: 1,
            max_input_tokens: 1,
            max_new_tokens: 2,
        },
    )
    .unwrap();
    let mut tensors: Vec<Tensor> = Vec::new();
    for op in &plan.prefill.operations {
        let Some(id) = op.attributes["weight"].as_str() else {
            continue;
        };
        if tensors.iter().any(|t| t.id() == id) {
            continue;
        }
        let shape: Vec<_> = op.attributes["weight_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let code = if op.operator == ModelOperator::LayerNorm {
            4096
        } else {
            0
        };
        tensors.push(Tensor::from_q12(id, &shape, &vec![code; shape.iter().product()]).unwrap());
        if let Some(id) = op.attributes["bias"].as_str() {
            let shape: Vec<_> = op.attributes["bias_shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            tensors.push(Tensor::from_q12(id, &shape, &vec![0; shape.iter().product()]).unwrap());
        }
    }
    Program::compile(&plan, Source::new(tensors).unwrap(), 2, Limits::default()).unwrap()
}
fn context(p: &Program) -> Context {
    Context {
        plan: p.digest(),
        session: [1; 32],
        operation: [2; 32],
        first_tensor_index: 0,
    }
}

#[test]
fn cached_kernel_charges_lowered_construction_metadata_per_process_once() {
    let program = fixture();
    let budget = request_budget(&program).unwrap();
    let tables = request_tables(&program, &budget).unwrap();
    let mut state = Schedule::new(&program, context(&program), 0, None, &tables).unwrap();
    let mut party_handles = 0;
    let before = state.resources.public_copy_bytes;
    drop(
        state
            .cached_kernel(&[0], 1, |tables| {
                let plan = gates::silu_cached(
                    gates::InputRange {
                        min: -4096,
                        max: 4096,
                    },
                    tables,
                )?
                .0;
                party_handles = plan.resources().public_table_copy_bytes_total;
                Ok(plan)
            })
            .unwrap(),
    );
    let first = tables.stats();
    assert_eq!(
        state.resources.public_copy_bytes - before,
        3 * first.table_payload_copy_bytes
            + 3 * first.lowered_construction_metadata_bytes
            + 24 * first.references
            + 512 * first.misses
            + 3 * std::mem::size_of::<cache::Stats>()
            + 2 * party_handles
    );
    let before = state.resources.public_copy_bytes;
    drop(
        state
            .cached_kernel(&[0], 1, |tables| {
                Ok(gates::silu_cached(
                    gates::InputRange {
                        min: -4096,
                        max: 4096,
                    },
                    tables,
                )?
                .0)
            })
            .unwrap(),
    );
    assert_eq!(
        tables.stats().lowered_construction_metadata_bytes,
        first.lowered_construction_metadata_bytes
    );
    assert_eq!(
        state.resources.public_copy_bytes - before,
        24 + 3 * std::mem::size_of::<cache::Stats>() + 2 * party_handles
    );
    drop(state);
    drop(tables);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn complete_schedule_reuses_real_tables_and_charges_only_real_misses() {
    let program = fixture();
    let budget = request_budget(&program).unwrap();
    let tables = request_tables(&program, &budget).unwrap();
    let initial = budget.used_bytes();
    let mut state = Schedule::new(&program, context(&program), 0, None, &tables).unwrap();
    drop(state.execute(&Zeroizing::new(vec![0])).unwrap());
    let stats = tables.stats();
    assert_eq!((stats.references, stats.misses, stats.hits), (24, 5, 19));
    assert_eq!(
        (
            stats.table_constructions,
            stats.table_hashes,
            stats.retained_entries
        ),
        (5, 5, 5)
    );
    assert_eq!(stats.generated_words, 8192 + 128 + 256);
    assert_eq!(
        stats.table_payload_copy_bytes,
        65536 + 4096 + 3072 + stats.lowered_payload_bytes
    );
    assert!(stats.lowered_chunk_constructions > stats.retained_entries);
    assert!(stats.lowered_retained_bytes > stats.lowered_payload_bytes);
    let a = state.resources;
    let b = program.resources();
    assert_eq!(
        (
            a.kernels,
            a.primitive_requests_per_party,
            a.work,
            a.material_bytes,
            a.frame_bytes,
            a.public_copy_bytes,
            a.metadata_bytes,
            a.live_array_bytes_per_party,
            a.peak_native_bytes,
            a.combined_reservation_bytes
        ),
        (
            b.kernels,
            b.primitive_requests_per_party,
            b.work,
            b.material_bytes,
            b.frame_bytes,
            b.public_copy_bytes,
            b.metadata_bytes,
            b.live_array_bytes_per_party,
            b.peak_native_bytes,
            b.combined_reservation_bytes
        )
    );
    assert_eq!(
        state.hash.clone().finalize().as_slice(),
        program.obligations
    );
    assert!(state.resources.work < 30_000_000);
    assert_eq!(budget.used_bytes(), initial + stats.retained_bytes);
    assert_eq!(b.source_table_cache_bytes_per_role, budget.used_bytes());
    println!("source-cache references={} misses={} hits={} generated_words={} hashes={} payload_copies={} retained={} work={} combined={}",stats.references,stats.misses,stats.hits,stats.generated_words,stats.table_hashes,stats.table_payload_copy_bytes,stats.retained_bytes,state.resources.work,state.resources.combined_reservation_bytes);
    drop(tables);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn transport_prices_live_request_cache_alongside_each_kernel() {
    use crate::fss_decoder::transport;
    let program = fixture();
    let context = context(&program);
    let admitted = transport::admit(&program, context).unwrap();
    let budget = request_budget(&program).unwrap();
    let tables = request_tables(&program, &budget).unwrap();
    struct Check<'a, 'b> {
        tables: &'a SourceTableCache<'b>,
        admission: transport::Admission,
        compiler_owners: usize,
        visits: usize,
    }
    impl MaterialVisitor for Check<'_, '_> {
        fn visit(&mut self, _: Context, plan: &Plan<'_>) -> Result<(), Error> {
            let r = plan.transport_resources_indexed(6, self.admission.guard_capacity())?;
            let cache = self.tables.stats();
            // Observe actual live cache leases while this Plan is borrowed.
            // Neither current Plan resources nor unrelated per-role cache
            // metadata may stand in for the other owner's reservation.
            let live_cache = cache.metadata_reservation_bytes + cache.retained_bytes;
            let live = self.compiler_owners
                + 3 * live_cache
                + 2 * r.combined_worker_reservation_bytes
                + r.combined_dealer_export_reservation_bytes
                + 16 * r.retained_frame_bytes_per_party
                + 64 * 1024;
            assert!(live <= self.admission.combined_reservation_bytes());
            self.visits += 1;
            Ok(())
        }
        fn cancel(&mut self) {}
    }
    let r = program.resources();
    let mut check = Check {
        tables: &tables,
        admission: admitted,
        compiler_owners: 3 * r.metadata_bytes + 2 * r.live_array_bytes_per_party,
        visits: 0,
    };
    let mut state = Schedule::new(&program, context, 0, None, &tables).unwrap();
    state.visitor = Some(&mut check);
    drop(state.execute(&Zeroizing::new(vec![0])).unwrap());
    assert_eq!(check.visits, admitted.kernels());
    assert!(tables.stats().hits > tables.stats().misses);
    assert_eq!(tables.stats().retained_entries, 5);
    drop(tables);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn partial_visit_releases_cache_and_rejects_namespace_substitution() {
    let program = fixture();
    let budget = request_budget(&program).unwrap();
    let tables = request_tables(&program, &budget).unwrap();
    struct Stop {
        cancelled: usize,
        visits: usize,
    }
    impl MaterialVisitor for Stop {
        fn visit(&mut self, _: Context, _: &Plan<'_>) -> Result<(), Error> {
            self.visits += 1;
            Err(Error::Context)
        }
        fn cancel(&mut self) {
            self.cancelled += 1;
        }
    }
    let mut stop = Stop {
        cancelled: 0,
        visits: 0,
    };
    assert!(program
        .visit_material(context(&program), &mut stop)
        .is_err());
    assert_eq!((stop.visits, stop.cancelled), (1, 1));
    let wrong = SourceTableCache::new(
        [9; 32],
        cache::Limits {
            max_entries: 8,
            max_retained_bytes: 1024 * 1024,
        },
        &budget,
    )
    .unwrap();
    assert!(matches!(
        Schedule::new(&program, context(&program), 0, None, &wrong),
        Err(Error::Context)
    ));
    drop(wrong);
    drop(tables);
    assert_eq!(budget.used_bytes(), 0);
}
