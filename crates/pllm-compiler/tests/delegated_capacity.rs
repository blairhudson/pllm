//! Geometry-only full-checkpoint admission and small Decoder transport fixtures.
//! No checkpoint files, model values, Python extension, or model jobs are used.
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

use pllm_compiler::{
    lower_decoder_matrix_inventory, resolve_experiment, GkmDecoderContract, GkmRuntimeContract,
    GkmTransportImport, GkmTransportWorker,
};
use pllm_core::gkm::{self, Capacity, Worker};
use pllm_models::{DecoderGraph, DecoderPlan, DecoderWorkload, ModelOperator};
use pllm_types::{canonical_bytes, Digest};
use serde_json::{json, Value};
use sha2::{Digest as _, Sha256};

static SERIAL: Mutex<()> = Mutex::new(());

// Qwen/Qwen2.5-0.5B-Instruct, revision 7ae557604adf67be50417f59c2c2f167def9a775;
// complete config.json Git blob 0dbb161213629a23f0fc00ef286e6b1e366d180f.
const PINNED_CONFIG: &[u8] = br#"{
  "architectures": ["Qwen2ForCausalLM"], "attention_dropout": 0.0,
  "bos_token_id": 151643, "eos_token_id": 151645, "hidden_act": "silu",
  "hidden_size": 896, "initializer_range": 0.02, "intermediate_size": 4864,
  "max_position_embeddings": 32768, "max_window_layers": 21, "model_type": "qwen2",
  "num_attention_heads": 14, "num_hidden_layers": 24, "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06, "rope_theta": 1000000.0, "sliding_window": 32768,
  "tie_word_embeddings": true, "torch_dtype": "bfloat16",
  "transformers_version": "4.43.1", "use_cache": true,
  "use_sliding_window": false, "vocab_size": 151936
}"#;

fn document(bytes: &[u8]) -> Value {
    serde_json::from_slice(bytes).unwrap()
}

fn number(value: &Value) -> u64 {
    value.as_u64().unwrap()
}

fn shape(value: &Value) -> Vec<usize> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|n| number(n) as usize)
        .collect()
}

fn tiny_config(vocabulary: usize) -> Value {
    json!({"model_type":"qwen2", "hidden_size":8, "intermediate_size":16,
        "num_hidden_layers":1, "num_attention_heads":2, "num_key_value_heads":1,
        "vocab_size":vocabulary, "max_position_embeddings":1024, "hidden_act":"silu",
        "rms_norm_eps":1e-6, "rope_theta":10000.0, "tie_word_embeddings":true})
}

fn experiment(decoder: bool, rows: usize, outputs: usize) -> Value {
    let mut value = json!({"schema":"pllm.experiment.v2", "name":"capacity-control",
        "pipeline":{"model":{"source":"fixture", "kind":"safetensors"}, "components":{
            "kernels":{"component":"pllm/cpu", "params":{"threads":1}},
            "linear":{"component":"pllm/open-weight-verifiable-outsourcing/v1",
                "params":{"sharing":"full", "generation":"fixed_length_greedy"}},
            "verification":{"component":"pllm/published-linear-verify/v1", "params":{"acceptance":"request"}},
            "topology":{"component":"pllm/two-online-delegation-workers/v1", "params":{}},
            "state":{"component":"pllm/client-local-kv", "params":{}}
        }}, "deployment":{"kind":"local", "root":"./capacity-fixture"},
        "budget":{"requests":1, "max_input_tokens":rows, "max_new_tokens":outputs}});
    if decoder {
        value["pipeline"]["components"]["linear"]["params"]["capacity"] = json!("decoder");
    }
    value
}

fn runtime(
    experiment: &Value,
    config: &[u8],
    rows: usize,
    outputs: usize,
) -> Result<GkmRuntimeContract, String> {
    GkmRuntimeContract::compile(
        &canonical_bytes(experiment),
        config,
        rows,
        outputs,
        Digest::from_sha256([1; 32]),
        Digest::from_sha256([2; 32]),
    )
}

// Independent interval oracle: allocation roots get inclusive birth/last-use
// intervals. Alias consumers extend whole backing-root intervals. No production
// remaining-use counter, resource estimator, or contract matrix list is reused.
fn interval_peak(graph: &DecoderGraph, inventory: &Value) -> usize {
    let ops: BTreeMap<_, _> = graph
        .operations
        .iter()
        .map(|op| (op.id.as_str(), op))
        .collect();
    let mut leaders = BTreeMap::new();
    let mut members = BTreeSet::new();
    for group in inventory["groups"].as_array().unwrap() {
        let ids: Vec<_> = group["members"]
            .as_array()
            .unwrap()
            .iter()
            .map(|member| member["operation_id"].as_str().unwrap())
            .collect();
        members.extend(ids.iter().copied());
        leaders.insert(ids[0], ids);
    }
    let mut steps = Vec::new();
    for op in &graph.operations {
        if let Some(group) = leaders.get(op.id.as_str()) {
            steps.push(group.iter().map(|id| ops[id]).collect::<Vec<_>>());
        } else if !members.contains(op.id.as_str()) {
            steps.push(vec![op]);
        }
    }
    let end = steps.len();
    let mut last_use = BTreeMap::new();
    for (index, step) in steps.iter().enumerate() {
        for input in &step[0].inputs {
            last_use.insert(input.as_str(), index);
        }
    }
    for op in &graph.operations {
        if op.operator == ModelOperator::GreedyTokenSelection {
            last_use.insert(op.inputs[0].as_str(), end);
        }
    }
    let mut intervals: Vec<(usize, usize, usize)> = Vec::new();
    let mut roots = BTreeMap::<&str, BTreeSet<usize>>::new();
    for (index, step) in steps.iter().enumerate() {
        let root = intervals.len();
        let bytes = step
            .iter()
            .map(|op| op.output_shape.iter().product::<u64>() as usize * 8)
            .sum();
        intervals.push((index, index, bytes));
        for op in step {
            let mut owners = BTreeSet::from([root]);
            if matches!(
                op.operator,
                ModelOperator::Reshape | ModelOperator::LastToken | ModelOperator::CacheSuffix
            ) {
                for input in &op.inputs {
                    if let Some(backing) = roots.get(input.as_str()) {
                        owners.extend(backing);
                    }
                }
            }
            let until = last_use.get(op.id.as_str()).copied().unwrap_or(end);
            for &owner in &owners {
                intervals[owner].1 = intervals[owner].1.max(until);
            }
            roots.insert(op.id.as_str(), owners);
        }
    }
    (0..end)
        .map(|index| {
            intervals
                .iter()
                .filter(|(birth, death, _)| *birth <= index && index <= *death)
                .map(|(_, _, bytes)| bytes)
                .sum()
        })
        .max()
        .unwrap()
}

fn independent_tensor_allowance(
    plan: &DecoderPlan,
    inventory: &Value,
    vocabulary: usize,
    decode_steps: usize,
) -> usize {
    let phase_peak = |graph: &DecoderGraph, inv: &Value| {
        // Independent caller-owned input vectors live alongside graph roots.
        interval_peak(graph, inv)
            + 8 * graph.batch as usize * (3 * graph.query_sequence as usize + 1)
    };
    let peak = phase_peak(&plan.prefill, &inventory["prefill"])
        .max(phase_peak(&plan.decode, &inventory["decode"]));
    let key_rows = plan.decode.maximum_key_sequence as usize;
    let allocated_rows = key_rows.max(64).next_power_of_two();
    let state_bytes: usize = plan
        .decode
        .state_outputs
        .iter()
        .map(|state| {
            assert_eq!(state.shape.len(), 4);
            assert_eq!(state.shape[2] as usize, key_rows);
            state.shape[0] as usize
                * state.shape[1] as usize
                * allocated_rows
                * state.shape[3] as usize
                * 8
        })
        .sum();
    6 * peak + 6 * state_bytes + vocabulary * 4 * (decode_steps + 3)
}

#[test]
fn pinned_qwen_short_150_and_288_row_contracts_price_independent_inventory_owners() {
    for (rows, decode_steps) in [(3, 1), (150, 31), (288, 31)] {
        let contract = GkmDecoderContract::compile_with_capacity(
            PINNED_CONFIG,
            rows,
            decode_steps,
            Capacity::Decoder,
        )
        .unwrap();
        assert_eq!(contract.capacity(), Capacity::Decoder);
        assert!(GkmDecoderContract::compile(PINNED_CONFIG, rows, decode_steps).is_err());
        let doc = document(&contract.canonical_bytes());
        let result_data_bytes =
            4 * 151_936 * (decode_steps + 1) + 4 * 2 * 24 * (rows + decode_steps) * 2 * 64;
        assert_eq!(
            number(&doc["decoder_resources"]["accepted_result_bytes"]),
            (result_data_bytes + (16 << 10)) as u64
        );
        if rows >= 150 {
            assert!(result_data_bytes > 16 << 20);
        }
        let plan = pllm_models::lower_model_json(
            PINNED_CONFIG,
            DecoderWorkload {
                batch: 1,
                max_input_tokens: rows as u64,
                max_new_tokens: decode_steps as u64 + 1,
            },
        )
        .unwrap();
        assert_eq!(doc["plan"], serde_json::to_value(&plan).unwrap());
        let inventory = lower_decoder_matrix_inventory(&plan).unwrap();
        let inventory = document(inventory.canonical_bytes());
        let groups = inventory["prefill"]["groups"].as_array().unwrap();
        assert_eq!(groups.len(), 98);
        let shapes: Vec<_> = groups
            .iter()
            .skip(1)
            .map(|group| {
                let shape = shape(&group["logical_shape"]);
                (shape[0], shape[1])
            })
            .collect();
        assert_eq!(shapes.len(), 97);
        for layer in 0..24 {
            assert_eq!(
                &shapes[layer * 4..layer * 4 + 4],
                &[(1152, 896), (896, 896), (9728, 896), (896, 4864)]
            );
        }
        assert_eq!(shapes[96], (151_936, 896));
        assert_eq!(
            number(&doc["window_steps"]),
            (97 * rows + 1 + 98 * decode_steps) as u64
        );
        let mut expected_sources = BTreeMap::new();
        let mut local = BTreeSet::new();
        for group in groups {
            for member in group["members"].as_array().unwrap() {
                expected_sources.insert(
                    member["weight_id"].as_str().unwrap().to_owned(),
                    shape(&member["source_shape"]),
                );
                if let Some(name) = member["bias_id"].as_str() {
                    expected_sources.insert(name.to_owned(), shape(&member["bias_shape"]));
                    local.insert(name.to_owned());
                }
            }
        }
        for op in &plan.prefill.operations {
            if op.operator == ModelOperator::RmsNorm {
                let name = op.attributes["weight"].as_str().unwrap().to_owned();
                expected_sources.insert(
                    name.clone(),
                    vec![*op.output_shape.last().unwrap() as usize],
                );
                local.insert(name);
            }
        }
        assert_eq!(contract.source_shapes(), &expected_sources);
        assert_eq!(expected_sources.len(), 290); // tied head/table is one source, two native artifacts
        assert_eq!(local.len(), 121); // 72 QKV biases + 49 normalization vectors
        let source_floats: usize = expected_sources
            .values()
            .map(|shape| shape.iter().product::<usize>() * 4)
            .sum();
        let local_bytes: usize = local
            .iter()
            .map(|name| expected_sources[name].iter().product::<usize>() * 4)
            .sum();
        assert_eq!(source_floats, 1_976_131_072);
        assert_eq!(local_bytes, 286_208);
        assert_eq!(number(&doc["source_float32_bytes"]), source_floats as u64);
        for (matrix, group) in doc["matrices"]
            .as_array()
            .unwrap()
            .iter()
            .zip(groups.iter().skip(1))
        {
            assert_eq!(
                json!([matrix["rows"], matrix["columns"]]),
                group["logical_shape"]
            );
            for (member, source) in matrix["members"]
                .as_array()
                .unwrap()
                .iter()
                .zip(group["members"].as_array().unwrap())
            {
                assert_eq!(member["weight"], source["weight_id"]);
                assert_eq!(member["bias"], source["bias_id"]);
                assert_eq!(member["rows"], source["output_width"]);
                assert_eq!(member["offset"], source["output_offset"]);
            }
        }
        let weights: usize = shapes.iter().map(|&(m, n)| m * n * 2).sum();
        let bounds: usize = shapes.iter().map(|&(m, n)| (m + n) * 8).sum();
        let table_bytes = 151_936 * 896 * 4;
        assert_eq!(weights + table_bytes, 1_532_461_056);
        assert_eq!(bounds, 5_105_664);
        let native = gkm::PublicationResources::for_shapes_with_capacity(
            &shapes,
            151_936,
            896,
            Capacity::Decoder,
        )
        .unwrap();
        assert_eq!(native.source_weight_bytes, weights);
        assert_eq!(native.source_table_bytes, table_bytes);
        assert_eq!(native.source_bound_bytes, bounds);
        assert_eq!(native.window_accumulator_bytes, 20_422_656);
        let resources = &doc["decoder_resources"];
        assert_eq!(resources["profile"], "pllm.gkm.decoder-capacity.v1");
        assert_eq!(
            number(&resources["source_import_bytes"]),
            (weights + table_bytes + bounds + local_bytes) as u64
        );
        assert_eq!(
            number(&resources["publication_payload_bytes"]),
            native.native_publication_payload_bytes as u64
        );
        assert_eq!(
            number(&resources["window_payload_bytes"]),
            native.max_window_payload_bytes as u64
        );
        assert_eq!(number(&resources["source_chunk_bytes"]), 64 << 10);
        assert_eq!(number(&resources["publication_frame_bytes"]), 32 << 20);
        assert_eq!(number(&resources["operation_frame_bytes"]), 4 << 20);
        let tensors = independent_tensor_allowance(&plan, &inventory, 151_936, decode_steps);
        // Additional conservatism is allowed; dropping a backing owner or a
        // state/snapshot allowance below this independent bound is not.
        let charged_tensors = number(&resources["client_tensor_bytes"]) as usize;
        assert!(charged_tensors >= tensors);
        // Includes the separately admitted tokenizer/index and graph metadata.
        let outer_frames_and_metadata = 4 * (32 << 20) + 8 * (4 << 20) + (384 << 20);
        let worker = native.admitted_payload_bytes + outer_frames_and_metadata + 2 * local_bytes;
        assert_eq!(number(&resources["worker_payload_bytes"]), worker as u64);
        assert_eq!(
            number(&resources["client_payload_bytes"]),
            (worker + charged_tensors) as u64
        );
        assert_eq!(
            doc["execution_payload_estimate_bytes"],
            resources["client_payload_bytes"]
        );
        assert!(worker + charged_tensors <= Capacity::Decoder.max_publication_bytes());
        let body_elements: usize = shapes[..96].iter().map(|&(m, n)| m * n).sum();
        let macs = body_elements as u64 * (rows + decode_steps) as u64
            + (151_936 * 896) as u64 * (decode_steps + 1) as u64;
        assert_eq!(number(&resources["request_linear_macs_per_worker"]), macs);
        assert!(macs <= 1 << 37);
    }
}

#[test]
fn sdk_capacity_is_explicit_and_bound_to_configuration_and_actual_schedule() {
    let config = canonical_bytes(&tiny_config(32));
    let capsule = runtime(&experiment(false, 2, 2), &config, 2, 2).unwrap();
    let decoder_experiment = experiment(true, 2, 2);
    let decoder = runtime(&decoder_experiment, &config, 2, 2).unwrap();
    assert_eq!(capsule.decoder().capacity(), Capacity::Capsule);
    assert_eq!(decoder.decoder().capacity(), Capacity::Decoder);
    let old = document(&capsule.canonical_bytes());
    let new = document(&decoder.canonical_bytes());
    assert!(old["decoder"].get("decoder_resources").is_none());
    assert_ne!(old["configuration"], new["configuration"]);
    assert_ne!(old["composition"], new["composition"]);
    assert_ne!(capsule.digest(), decoder.digest());
    assert_ne!(capsule.decoder().digest(), decoder.decoder().digest());
    assert_eq!(old["decoder"]["plan"], new["decoder"]["plan"]);
    let resolved = resolve_experiment(&canonical_bytes(&decoder_experiment)).unwrap();
    assert_eq!(new["configuration"], json!(resolved.configuration_digest()));
    assert_eq!(new["composition"], json!(resolved.composition_digest()));
    assert_ne!(
        runtime(&decoder_experiment, &config, 1, 1)
            .unwrap()
            .digest(),
        decoder.digest()
    );
    assert!(runtime(&experiment(false, 1, 1), PINNED_CONFIG, 1, 1).is_err());
    assert!(runtime(&experiment(true, 3, 2), PINNED_CONFIG, 3, 2).is_ok());
    assert!(runtime(
        &experiment(false, 1, 1),
        &canonical_bytes(&tiny_config(257)),
        1,
        1
    )
    .is_err());
    for invalid in [
        json!("capsule"),
        json!("Decoder"),
        json!(false),
        json!(null),
        json!({}),
    ] {
        let mut value = decoder_experiment.clone();
        value["pipeline"]["components"]["linear"]["params"]["capacity"] = invalid;
        assert!(runtime(&value, &config, 1, 1).is_err());
    }
}

#[test]
fn invalid_decoder_geometry_and_declared_envelopes_reject_without_source_values() {
    let config = canonical_bytes(&tiny_config(32));
    for (rows, decode) in [(0, 0), (513, 0), (1, 65), (usize::MAX, 0), (1, usize::MAX)] {
        assert!(GkmDecoderContract::compile_with_capacity(
            &config,
            rows,
            decode,
            Capacity::Decoder
        )
        .is_err());
    }
    for (key, value) in [
        ("num_hidden_layers", u64::MAX),
        ("hidden_size", 16_385),
        ("intermediate_size", u64::MAX),
        ("vocab_size", 262_145),
        ("num_attention_heads", 129),
        ("num_key_value_heads", u64::MAX),
    ] {
        let mut changed = tiny_config(32);
        changed[key] = json!(value);
        assert!(
            GkmDecoderContract::compile_with_capacity(
                &canonical_bytes(&changed),
                1,
                0,
                Capacity::Decoder
            )
            .is_err(),
            "{key}"
        );
    }
    // Individually bounded fields still exceed total matrix-count or source-byte capacity.
    let mut too_many = tiny_config(32);
    too_many["num_hidden_layers"] = json!(64);
    assert!(GkmDecoderContract::compile_with_capacity(
        &canonical_bytes(&too_many),
        1,
        0,
        Capacity::Decoder
    )
    .is_err());
    let mut too_large = tiny_config(65_536);
    for (key, value) in [
        ("hidden_size", 4096),
        ("intermediate_size", 8192),
        ("num_attention_heads", 64),
        ("num_key_value_heads", 2),
        ("num_hidden_layers", 3),
    ] {
        too_large[key] = json!(value);
    }
    assert!(GkmDecoderContract::compile_with_capacity(
        &canonical_bytes(&too_large),
        1,
        0,
        Capacity::Decoder
    )
    .is_err());
    assert!(
        GkmDecoderContract::compile_with_capacity(PINNED_CONFIG, 512, 64, Capacity::Decoder)
            .is_err()
    );
    // An actual tiny query must not conceal an oversized saved work envelope.
    assert!(runtime(&experiment(true, 512, 65), PINNED_CONFIG, 1, 1).is_err());
    assert!(runtime(&experiment(true, 513, 1), &config, 1, 1).is_err());
    assert!(runtime(&experiment(true, 1, 66), &config, 1, 1).is_err());
    for bytes in [&b""[..], &b"{"[..], &vec![b' '; (16 << 10) + 1]] {
        assert!(GkmDecoderContract::compile_with_capacity(bytes, 1, 0, Capacity::Decoder).is_err());
    }
}

fn small_source(contract: &GkmRuntimeContract) -> BTreeMap<String, Vec<u8>> {
    let pattern = [
        -0.0f32,
        1.0 / 4096.0,
        3.0 / 4096.0,
        -3.0 / 4096.0,
        0.15625,
        -0.125,
        1.0,
    ];
    contract
        .decoder()
        .source_shapes()
        .iter()
        .map(|(name, shape)| {
            (
                name.clone(),
                (0..shape.iter().product::<usize>())
                    .flat_map(|i| pattern[i % pattern.len()].to_le_bytes())
                    .collect(),
            )
        })
        .collect()
}

fn streamed(
    contract: GkmRuntimeContract,
    source: &BTreeMap<String, Vec<u8>>,
) -> GkmTransportImport {
    let mut importer = GkmTransportImport::new(contract).unwrap();
    for (name, bytes) in source {
        for (index, chunk) in bytes.chunks(28).enumerate() {
            importer.push(name, index * 28, chunk).unwrap();
        }
    }
    importer
}

fn quantized(value: f32) -> i128 {
    let x = value as f64 * 2048.0;
    let floor = x.floor();
    let part = x - floor;
    floor as i128 + i128::from(part > 0.5 || (part == 0.5 && floor as i128 % 2 != 0))
}

#[test]
fn streamed_wide_head_exact_integer_transcript_acceptance_and_failure_retirement() {
    let _serial = SERIAL.lock().unwrap();
    let before = gkm::WireResources::current_with_capacity(Capacity::Decoder).process_live_bytes;
    let config = canonical_bytes(&tiny_config(257));
    let experiment = experiment(true, 2, 2);
    let make = || runtime(&experiment, &config, 2, 2).unwrap();
    for corrupt in [false, true] {
        let contract = make();
        let sdk_pin = contract.digest();
        let decoder_pin = contract.decoder().digest();
        let doc = document(&contract.decoder().canonical_bytes());
        assert_eq!(doc["matrices"][4]["rows"], 257);
        let source = small_source(&contract);
        let mut client = streamed(contract, &source).client().unwrap();
        let public = client.publication_bytes().unwrap();
        let starts = client.starts().unwrap();
        assert_eq!(public.capacity(), Capacity::Decoder);
        let mut workers: [GkmTransportWorker; 2] = [Worker::Zero, Worker::One].map(|role| {
            streamed(make(), &source)
                .worker(
                    public.as_bytes(),
                    client.publication_digest(),
                    role,
                    starts[usize::from(role == Worker::One)].as_bytes(),
                )
                .unwrap()
        });
        let mut source_hash = Sha256::new();
        source_hash.update(b"pllm.gkm.decoder-source.v1");
        source_hash.update(decoder_pin.as_str());
        for (name, bytes) in &source {
            source_hash.update((name.len() as u64).to_le_bytes());
            source_hash.update(name);
            source_hash.update((bytes.len() as u64).to_le_bytes());
            source_hash.update(bytes);
        }
        let source_pin: [u8; 32] = source_hash.finalize().into();
        let mut context = Sha256::new();
        context.update(b"pllm.gkm.sdk-request-context.v1");
        context.update(sdk_pin.as_str());
        context.update(source_pin);
        let context: [u8; 32] = context.finalize().into();
        assert_eq!(&starts[0].as_bytes()[114..146], &context);
        let mut plan_hash = Sha256::new();
        plan_hash.update(b"pllm.gkm.window-plan.v1");
        plan_hash.update(client.publication_digest());
        plan_hash.update(context);
        plan_hash.update(number(&doc["window_steps"]).to_le_bytes());
        let mut ordinal = 0;
        for rows in [2, 1] {
            for token in (257 - rows)..257 {
                plan_hash.update([0]);
                let request = client.lookup(token).unwrap();
                let replies = [
                    workers[0].execute(request[0].as_bytes()).unwrap(),
                    workers[1].execute(request[1].as_bytes()).unwrap(),
                ];
                let actual = client
                    .accept(replies[0].as_bytes(), replies[1].as_bytes())
                    .unwrap();
                assert_eq!(
                    actual,
                    source[doc["table"].as_str().unwrap()][token * 32..(token + 1) * 32]
                );
                ordinal += 1;
            }
            for (stage, matrix) in doc["matrices"].as_array().unwrap().iter().enumerate() {
                for _ in 0..if stage == 4 { 1 } else { rows } {
                    plan_hash.update([1]);
                    plan_hash.update(Sha256::digest(matrix["id"].as_str().unwrap()));
                    let columns = number(&matrix["columns"]) as usize;
                    let pattern = [
                        0.25f32,
                        -0.125,
                        1.0 / 4096.0,
                        3.0 / 4096.0,
                        -3.0 / 4096.0,
                        -0.0,
                        1.25,
                    ];
                    let input: Vec<_> = (0..columns).map(|i| pattern[i % pattern.len()]).collect();
                    let input_bytes: Vec<_> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let request = client.linear(stage, &input_bytes).unwrap();
                    let replies = [
                        workers[0].execute(request[0].as_bytes()).unwrap(),
                        workers[1].execute(request[1].as_bytes()).unwrap(),
                    ];
                    let exact: Vec<i128> = matrix["members"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .flat_map(|member| {
                            source[member["weight"].as_str().unwrap()]
                                .chunks_exact(columns * 4)
                                .map(|row| {
                                    row.chunks_exact(4)
                                        .zip(&input)
                                        .map(|(w, &x)| {
                                            quantized(f32::from_le_bytes(w.try_into().unwrap()))
                                                * quantized(x)
                                        })
                                        .sum()
                                })
                        })
                        .collect();
                    let offset = replies[0].len() - exact.len() * 8;
                    for ((a, b), &expected) in replies[0].as_bytes()[offset..]
                        .chunks_exact(8)
                        .zip(replies[1].as_bytes()[offset..].chunks_exact(8))
                        .zip(&exact)
                    {
                        let actual = u64::from_le_bytes(a.try_into().unwrap())
                            .wrapping_add(u64::from_le_bytes(b.try_into().unwrap()))
                            as i64;
                        assert_eq!(actual as i128, expected);
                    }
                    ordinal += 1;
                    if corrupt && ordinal == number(&doc["window_steps"]) {
                        let mut changed = replies[0].as_bytes().to_vec();
                        let word =
                            u64::from_le_bytes(changed[offset..offset + 8].try_into().unwrap());
                        changed[offset..offset + 8]
                            .copy_from_slice(&word.wrapping_add(1).to_le_bytes());
                        client.accept(&changed, replies[1].as_bytes()).unwrap();
                    } else {
                        let expected: Vec<_> = exact
                            .iter()
                            .flat_map(|&v| ((v as i64 as f32) / 4_194_304.0).to_le_bytes())
                            .collect();
                        assert_eq!(
                            client
                                .accept(replies[0].as_bytes(), replies[1].as_bytes())
                                .unwrap(),
                            expected
                        );
                    }
                }
            }
        }
        assert_eq!(ordinal, number(&doc["window_steps"]));
        let expected_window: [u8; 32] = plan_hash.finalize().into();
        for worker in &mut workers {
            let receipt = document(&worker.finish().unwrap());
            assert_eq!(receipt["contract"], json!(sdk_pin));
            assert_eq!(receipt["decoder_contract"], json!(decoder_pin));
            assert_eq!(receipt["context"], json!(Digest::from_sha256(context)));
            assert!(receipt.get("acceptance").is_none());
            assert!(worker.finish().is_err());
            assert!(worker.execute(&[]).is_err());
        }
        if corrupt {
            assert!(client.finish().unwrap_err().contains("Verification"));
        } else {
            let bytes = client.finish().unwrap();
            let receipt = document(&bytes);
            assert_eq!(bytes, canonical_bytes(&receipt));
            assert_eq!(receipt["contract"], json!(sdk_pin));
            assert_eq!(receipt["decoder_contract"], json!(decoder_pin));
            assert_eq!(receipt["source"], json!(Digest::from_sha256(source_pin)));
            assert_eq!(
                receipt["window"],
                json!(Digest::from_sha256(expected_window))
            );
            assert_eq!(receipt["context"], json!(Digest::from_sha256(context)));
            assert_eq!(receipt["steps"], doc["window_steps"]);
        }
        assert!(client.local_tensors().is_empty());
        assert!(client.finish().is_err());
        assert!(client.lookup(0).is_err());
        assert!(client.publication_bytes().is_err());
    }
    assert_eq!(
        gkm::WireResources::current_with_capacity(Capacity::Decoder).process_live_bytes,
        before
    );
    let values = small_source(&make());
    let (name, bytes) = values.first_key_value().unwrap();
    let mut malformed = GkmTransportImport::new(make()).unwrap();
    assert!(malformed.push(name, 4, bytes).is_err());
    assert!(malformed.push(name, 0, bytes).is_err());
    assert!(malformed.client().is_err());
}

#[test]
fn decoder_worker_rejects_capacity_and_short_schedule_substitution() {
    let _serial = SERIAL.lock().unwrap();
    let config = canonical_bytes(&tiny_config(32));
    let make = |expanded| runtime(&experiment(expanded, 1, 1), &config, 1, 1).unwrap();
    let source = small_source(&make(true));
    let mut client = streamed(make(true), &source).client().unwrap();
    let public = client.publication_bytes().unwrap();
    let starts = client.starts().unwrap();
    // This source is small enough for either policy. Substituting a Capsule
    // contract must still fail: capacity participates in the source/SDK identity.
    let error = streamed(make(false), &source)
        .worker(
            public.as_bytes(),
            client.publication_digest(),
            Worker::Zero,
            starts[0].as_bytes(),
        )
        .err()
        .expect("capacity substitution accepted");
    assert!(error.contains("source mismatch"), "{error}");

    let publication = std::sync::Arc::new(
        gkm::Publication::import_wire_with_capacity(
            public.as_bytes(),
            client.publication_digest(),
            Capacity::Decoder,
        )
        .unwrap(),
    );
    let rogue = gkm::Client::new(publication, client.publication_digest()).unwrap();
    let context = starts[0].as_bytes()[114..146].try_into().unwrap();
    let mut short = rogue.begin_wire(context, &[gkm::Step::Lookup]).unwrap();
    let short_start = short.start_frames().unwrap();
    let error = streamed(make(true), &source)
        .worker(
            public.as_bytes(),
            client.publication_digest(),
            Worker::Zero,
            short_start[0].as_bytes(),
        )
        .err()
        .expect("short authenticated schedule accepted");
    assert!(error.contains("Binding"), "{error}");
    client.abort();
    assert!(client.local_tensors().is_empty());
    assert!(client.starts().is_err());
}
