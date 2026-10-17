use std::collections::BTreeMap;
use std::sync::Arc;

use pllm_compiler::{
    GkmDecoderSession, GkmRuntimeContract, GkmTransportClient, GkmTransportImport,
    GkmTransportWorker,
};
use pllm_core::gkm::{self, Publication, Step, WireBytes, Worker};
use pllm_types::{canonical_bytes, Digest};
use serde_json::{json, Value};
use sha2::{Digest as _, Sha256};

fn experiment() -> Value {
    json!({"schema":"pllm.experiment.v2", "name":"transport-control",
        "pipeline":{"model":{"source":"fixture", "kind":"safetensors"},
        "components":{
            "kernels":{"component":"pllm/cpu", "params":{"threads":1}},
            "linear":{"component":"pllm/open-weight-verifiable-outsourcing/v1",
                "params":{"sharing":"full", "generation":"fixed_length_greedy"}},
            "verification":{"component":"pllm/published-linear-verify/v1", "params":{"acceptance":"request"}},
            "topology":{"component":"pllm/two-online-delegation-workers/v1", "params":{}},
            "state":{"component":"pllm/client-local-kv", "params":{}}
        }}, "deployment":{"kind":"local", "root":"./transport-fixture"},
        "budget":{"requests":1, "max_input_tokens":2, "max_new_tokens":2}})
}

fn config(family: &str, tied: bool) -> Value {
    let mut value = json!({
        "model_type":family, "hidden_size":8, "intermediate_size":16,
        "num_hidden_layers":1, "num_attention_heads":2, "num_key_value_heads":1,
        "vocab_size":32, "max_position_embeddings":16, "hidden_act":"silu",
        "rms_norm_eps":1e-6, "rope_theta":10000.0, "tie_word_embeddings":tied
    });
    if family == "qwen3" {
        value["head_dim"] = json!(4);
        value["attention_bias"] = json!(false);
        value["attention_dropout"] = json!(0.0);
        value["rope_scaling"] = Value::Null;
        value["sliding_window"] = Value::Null;
        value["use_sliding_window"] = json!(false);
        value["use_cache"] = json!(true);
        value["max_window_layers"] = json!(1);
        value["layer_types"] = json!(["full_attention"]);
    }
    value
}

fn runtime(family: &str, tied: bool, rows: usize, outputs: usize) -> GkmRuntimeContract {
    GkmRuntimeContract::compile(
        &canonical_bytes(&experiment()),
        &canonical_bytes(&config(family, tied)),
        rows,
        outputs,
        Digest::from_sha256([1; 32]),
        Digest::from_sha256([2; 32]),
    )
    .unwrap()
}

fn tensors(contract: &GkmRuntimeContract) -> BTreeMap<String, Vec<u8>> {
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
                    .flat_map(|index| pattern[index % pattern.len()].to_le_bytes())
                    .collect(),
            )
        })
        .collect()
}

fn borrowed(source: &BTreeMap<String, Vec<u8>>) -> BTreeMap<String, &[u8]> {
    source
        .iter()
        .map(|(name, bytes)| (name.clone(), bytes.as_slice()))
        .collect()
}

fn streamed(
    contract: GkmRuntimeContract,
    values: &BTreeMap<String, Vec<u8>>,
) -> GkmTransportImport {
    let mut source = GkmTransportImport::new(contract).unwrap();
    for (name, bytes) in values {
        // Split within source rows and grouped member boundaries.
        for (index, chunk) in bytes.chunks(28).enumerate() {
            source.push(name, index * 28, chunk).unwrap();
        }
    }
    source
}

fn workers(
    client: &mut GkmTransportClient,
    tensors: &BTreeMap<String, &[u8]>,
    make: impl Fn() -> GkmRuntimeContract,
) -> [GkmTransportWorker; 2] {
    let publication = client.publication_bytes().unwrap();
    let starts = client.starts().unwrap();
    [Worker::Zero, Worker::One].map(|role| {
        let index = usize::from(role == Worker::One);
        GkmTransportWorker::new(
            make(),
            tensors,
            publication.as_bytes(),
            client.publication_digest(),
            role,
            starts[index].as_bytes(),
        )
        .unwrap()
    })
}

#[derive(Clone, Copy)]
enum Call {
    Lookup(usize),
    Linear(usize),
}

fn calls(doc: &Value) -> Vec<Call> {
    let mut calls = Vec::new();
    let mut token = 0;
    for (name, repeats) in [
        ("prefill", 1),
        ("decode", doc["decode_steps"].as_u64().unwrap()),
    ] {
        let phase = &doc[name];
        for _ in 0..repeats {
            let mut stage = 0;
            for step in phase["steps"].as_array().unwrap() {
                if step["executor"] == "client_local" {
                    continue;
                }
                let count = if step["operators"][0] == "output_head" {
                    1
                } else {
                    phase["query_sequence"].as_u64().unwrap()
                };
                for _ in 0..count {
                    if step["operators"][0] == "token_lookup" {
                        calls.push(Call::Lookup(token));
                        token += 1;
                    } else {
                        calls.push(Call::Linear(stage));
                    }
                }
                if step["operators"][0] != "token_lookup" {
                    stage += 1;
                }
            }
            assert_eq!(stage, doc["matrices"].as_array().unwrap().len());
        }
    }
    assert_eq!(calls.len() as u64, doc["window_steps"].as_u64().unwrap());
    calls
}

fn inputs(doc: &Value, stage: usize) -> Vec<u8> {
    let pattern = [
        0.25f32,
        -0.125,
        1.0 / 4096.0,
        3.0 / 4096.0,
        -3.0 / 4096.0,
        -0.0,
        1.25,
    ];
    (0..doc["matrices"][stage]["columns"].as_u64().unwrap() as usize)
        .flat_map(|i| pattern[i % pattern.len()].to_le_bytes())
        .collect()
}

// Independent exact-binary tie rule; the oracle calls no GKM numeric helper.
fn integer(value: f32) -> i128 {
    let scaled = value as f64 * 2048.0;
    let floor = scaled.floor();
    let fraction = scaled - floor;
    if fraction > 0.5 || (fraction == 0.5 && floor as i128 % 2 != 0) {
        floor as i128 + 1
    } else {
        floor as i128
    }
}

fn linear_oracle(
    doc: &Value,
    source: &BTreeMap<String, Vec<u8>>,
    stage: usize,
    input: &[u8],
) -> Vec<i128> {
    let columns = doc["matrices"][stage]["columns"].as_u64().unwrap() as usize;
    doc["matrices"][stage]["members"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|member| {
            source[member["weight"].as_str().unwrap()]
                .chunks_exact(columns * 4)
                .map(|row| {
                    row.chunks_exact(4)
                        .zip(input.chunks_exact(4))
                        .map(|(w, x)| {
                            integer(f32::from_le_bytes(w.try_into().unwrap()))
                                * integer(f32::from_le_bytes(x.try_into().unwrap()))
                        })
                        .sum()
                })
        })
        .collect()
}

fn exchange(workers: &mut [GkmTransportWorker; 2], request: &[WireBytes; 2]) -> [WireBytes; 2] {
    [
        workers[0].execute(request[0].as_bytes()).unwrap(),
        workers[1].execute(request[1].as_bytes()).unwrap(),
    ]
}

#[test]
fn generated_complete_schedules_match_integer_original_rows_and_seeded_reference() {
    complete_source_schedules(false);
}

#[test]
fn streamed_sources_preserve_original_digest_coefficients_lookup_and_integer_products() {
    complete_source_schedules(true);
}

fn complete_source_schedules(streaming: bool) {
    let mut saw_bias = false;
    for family in ["qwen2", "qwen3", "llama"] {
        for tied in [true, false] {
            let contract = runtime(family, tied, 2, 2);
            let sdk_digest = contract.digest();
            let decoder_digest = contract.decoder().digest();
            let expected_contract = contract.canonical_bytes();
            let doc: Value = serde_json::from_slice(&contract.decoder().canonical_bytes()).unwrap();
            let source = tensors(&contract);
            let borrowed = borrowed(&source);
            let reference_contract = pllm_compiler::GkmDecoderContract::compile(
                &canonical_bytes(&config(family, tied)),
                2,
                1,
            )
            .unwrap();
            let mut reference =
                GkmDecoderSession::new(reference_contract, &borrowed, true).unwrap();
            let mut client = if streaming {
                streamed(contract, &source).client().unwrap()
            } else {
                GkmTransportClient::new(contract, &borrowed).unwrap()
            };
            assert_eq!(client.contract_bytes(), expected_contract);
            assert_eq!(
                serde_json::from_slice::<Value>(&client.decoder_bytes()).unwrap(),
                doc
            );
            for (name, bytes) in client.local_tensors() {
                assert_eq!(bytes, &source[name]);
            }
            for matrix in doc["matrices"].as_array().unwrap() {
                for member in matrix["members"].as_array().unwrap() {
                    if let Some(bias) = member["bias"].as_str() {
                        saw_bias = true;
                        assert_eq!(client.local_tensors()[bias], source[bias]);
                    }
                }
            }
            let mut workers = if streaming {
                let public = client.publication_bytes().unwrap();
                let starts = client.starts().unwrap();
                [Worker::Zero, Worker::One].map(|role| {
                    streamed(runtime(family, tied, 2, 2), &source)
                        .worker(
                            public.as_bytes(),
                            client.publication_digest(),
                            role,
                            starts[usize::from(role == Worker::One)].as_bytes(),
                        )
                        .unwrap()
                })
            } else {
                workers(&mut client, &borrowed, || runtime(family, tied, 2, 2))
            };
            for call in calls(&doc) {
                match call {
                    Call::Lookup(token) => {
                        let requests = client.lookup(token).unwrap();
                        let replies = exchange(&mut workers, &requests);
                        let got = client
                            .accept(replies[0].as_bytes(), replies[1].as_bytes())
                            .unwrap();
                        let width = doc["width"].as_u64().unwrap() as usize * 4;
                        assert_eq!(
                            got,
                            source[doc["table"].as_str().unwrap()]
                                [token * width..(token + 1) * width]
                        );
                        assert_eq!(got, reference.lookup(token).unwrap());
                    }
                    Call::Linear(stage) => {
                        let input = inputs(&doc, stage);
                        let requests = client.linear(stage, &input).unwrap();
                        let replies = exchange(&mut workers, &requests);
                        let exact = linear_oracle(&doc, &source, stage, &input);
                        let width = exact.len() * 8;
                        let a = &replies[0].as_bytes()[replies[0].len() - width..];
                        let b = &replies[1].as_bytes()[replies[1].len() - width..];
                        for ((a, b), &expected) in
                            a.chunks_exact(8).zip(b.chunks_exact(8)).zip(&exact)
                        {
                            let signed = u64::from_le_bytes(a.try_into().unwrap())
                                .wrapping_add(u64::from_le_bytes(b.try_into().unwrap()))
                                as i64;
                            assert_eq!(signed as i128, expected);
                        }
                        let expected: Vec<_> = exact
                            .iter()
                            .flat_map(|&v| ((v as i64 as f32) / 4_194_304.0f32).to_le_bytes())
                            .collect();
                        let got = client
                            .accept(replies[0].as_bytes(), replies[1].as_bytes())
                            .unwrap();
                        assert_eq!(got, expected); // pre-bias, preserving the semantic executor boundary
                        assert_eq!(got, reference.linear(stage, &input).unwrap());
                    }
                }
            }
            for (index, worker) in workers.iter_mut().enumerate() {
                let bytes = worker.finish().unwrap();
                let receipt: Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(bytes, canonical_bytes(&receipt));
                assert_eq!(
                    receipt["schema_version"],
                    "pllm.gkm.sdk-worker-completion.v1"
                );
                assert_eq!(receipt["worker"], index);
                assert!(receipt.get("acceptance").is_none());
                assert_eq!(
                    receipt["request"],
                    json!(Digest::from_sha256(client.request_id()))
                );
                assert!(worker.finish().is_err());
            }
            let bytes = client.finish().unwrap();
            let receipt: Value = serde_json::from_slice(&bytes).unwrap();
            let legacy: Value = serde_json::from_slice(&reference.finish().unwrap()).unwrap();
            assert_eq!(bytes, canonical_bytes(&receipt));
            assert_eq!(receipt["schema_version"], "pllm.gkm.sdk-acceptance.v1");
            assert_eq!(receipt["contract"], json!(sdk_digest));
            assert_eq!(receipt["decoder_contract"], json!(decoder_digest));
            assert_eq!(receipt["numeric_profile"], gkm::NUMERIC_PROFILE);
            assert_eq!(receipt["source"], legacy["source"]);
            assert_eq!(receipt["steps"], doc["window_steps"]);
            assert_eq!(receipt["seeded"], false);
            assert!(client.local_tensors().is_empty());
            assert!(client.finish().is_err());
        }
    }
    assert!(saw_bias);
}

#[test]
fn source_import_order_offsets_numerics_and_finish_are_one_use() {
    let values = tensors(&runtime("qwen2", true, 2, 2));
    let (name, bytes) = values.first_key_value().unwrap();
    for (wrong_name, offset, data) in [
        ("wrong", 0, bytes.as_slice()),
        (name.as_str(), 4, bytes.as_slice()),
        (name.as_str(), 0, &[]),
        (name.as_str(), 0, &[0; 3]),
        (name.as_str(), 0, &f32::NAN.to_le_bytes()),
        (name.as_str(), 0, &f32::INFINITY.to_le_bytes()),
        (name.as_str(), 0, &f32::MAX.to_le_bytes()),
    ] {
        let mut importer = GkmTransportImport::new(runtime("qwen2", true, 2, 2)).unwrap();
        assert!(importer.push(wrong_name, offset, data).is_err());
        assert!(importer.push(name, 0, bytes).is_err());
    }
    let mut importer = GkmTransportImport::new(runtime("qwen2", true, 2, 2)).unwrap();
    importer.push(name, 0, &bytes[..4]).unwrap();
    assert!(importer.push(name, 0, &bytes[..4]).is_err());
    assert!(importer.push(name, 4, &bytes[4..]).is_err());
    let mut importer = GkmTransportImport::new(runtime("qwen2", true, 2, 2)).unwrap();
    assert!(importer
        .push(name, 0, &vec![0; pllm_compiler::GKM_SOURCE_CHUNK_BYTES + 4])
        .is_err());
    assert!(importer.client().is_err());
    assert!(GkmTransportImport::new(runtime("qwen2", true, 2, 2))
        .unwrap()
        .client()
        .is_err());
    let mut complete = streamed(runtime("qwen2", true, 2, 2), &values);
    assert!(complete.push(name, 0, bytes).is_err());
    assert!(complete.client().is_err());
    let mut aborted = streamed(runtime("qwen2", true, 2, 2), &values);
    aborted.abort();
    assert!(aborted.client().is_err());
}

#[test]
fn worker_rejects_actual_source_sdk_context_and_forged_short_schedule() {
    let contract = runtime("qwen2", true, 2, 2);
    let source = tensors(&contract);
    let doc: Value = serde_json::from_slice(&contract.decoder().canonical_bytes()).unwrap();
    let mut client = GkmTransportClient::new(contract, &borrowed(&source)).unwrap();
    let public = client.publication_bytes().unwrap();
    let starts = client.starts().unwrap();
    let pin = client.publication_digest();
    for field in ["configuration", "source_lock", "tokenizer", "actual_rows"] {
        let mut e = experiment();
        if field == "configuration" {
            e["name"] = json!("different-experiment");
        }
        let other = GkmRuntimeContract::compile(
            &canonical_bytes(&e),
            &canonical_bytes(&config("qwen2", true)),
            if field == "actual_rows" { 1 } else { 2 },
            2,
            Digest::from_sha256([if field == "source_lock" { 3 } else { 1 }; 32]),
            Digest::from_sha256([if field == "tokenizer" { 4 } else { 2 }; 32]),
        )
        .unwrap();
        assert!(
            GkmTransportWorker::new(
                other,
                &borrowed(&source),
                public.as_bytes(),
                pin,
                Worker::Zero,
                starts[0].as_bytes()
            )
            .is_err(),
            "{field}"
        );
    }
    let weight = doc["matrices"][0]["members"][0]["weight"].as_str().unwrap();
    let local = client.local_tensors().keys().next().unwrap();
    let table = doc["table"].as_str().unwrap();
    for name in [weight, local.as_str(), table] {
        let mut altered = source.clone();
        // -0.0 -> +0.0 preserves quantized integers, but not the actual source.
        altered.get_mut(name).unwrap()[..4].copy_from_slice(&0.0f32.to_le_bytes());
        assert!(
            GkmTransportWorker::new(
                runtime("qwen2", true, 2, 2),
                &borrowed(&altered),
                public.as_bytes(),
                pin,
                Worker::Zero,
                starts[0].as_bytes()
            )
            .is_err(),
            "{name}"
        );
    }
    assert!(GkmTransportWorker::new(
        runtime("qwen2", true, 2, 2),
        &borrowed(&source),
        public.as_bytes(),
        [0; 32],
        Worker::Zero,
        starts[0].as_bytes()
    )
    .is_err());
    assert!(GkmTransportWorker::new(
        runtime("qwen2", true, 2, 2),
        &borrowed(&source),
        public.as_bytes(),
        pin,
        Worker::One,
        starts[0].as_bytes()
    )
    .is_err());
    let publication = Arc::new(Publication::import_wire(public.as_bytes(), pin).unwrap());
    let mut hash = Sha256::new();
    hash.update(b"pllm.gkm.sdk-request-context.v1");
    hash.update(runtime("qwen2", true, 2, 2).digest().as_str().as_bytes());
    hash.update(publication.context());
    let context: [u8; 32] = hash.finalize().into();
    let rogue = gkm::Client::new(publication, pin).unwrap();
    let mut short = rogue.begin_wire(context, &[Step::Lookup]).unwrap();
    let short_starts = short.start_frames().unwrap();
    assert!(GkmTransportWorker::new(
        runtime("qwen2", true, 2, 2),
        &borrowed(&source),
        public.as_bytes(),
        pin,
        Worker::Zero,
        short_starts[0].as_bytes()
    )
    .is_err());
}

#[test]
fn final_corruption_never_becomes_a_client_acceptance_receipt() {
    let contract = runtime("qwen2", true, 1, 1);
    let source = tensors(&contract);
    let doc: Value = serde_json::from_slice(&contract.decoder().canonical_bytes()).unwrap();
    let mut client = GkmTransportClient::new(contract, &borrowed(&source)).unwrap();
    let mut workers = workers(&mut client, &borrowed(&source), || {
        runtime("qwen2", true, 1, 1)
    });
    let calls = calls(&doc);
    for (ordinal, call) in calls.iter().enumerate() {
        let request = match *call {
            Call::Lookup(token) => client.lookup(token),
            Call::Linear(stage) => client.linear(stage, &inputs(&doc, stage)),
        }
        .unwrap();
        let reply = exchange(&mut workers, &request);
        if ordinal + 1 == calls.len() {
            let Call::Linear(stage) = *call else {
                panic!("expected full head");
            };
            let width = doc["matrices"][stage]["rows"].as_u64().unwrap() as usize * 8;
            let mut corrupt = reply[0].as_bytes().to_vec();
            let offset = corrupt.len() - width;
            let value = u64::from_le_bytes(corrupt[offset..offset + 8].try_into().unwrap());
            corrupt[offset..offset + 8].copy_from_slice(&value.wrapping_add(1).to_le_bytes());
            assert!(client.accept(&corrupt, reply[1].as_bytes()).is_ok());
        } else {
            client
                .accept(reply[0].as_bytes(), reply[1].as_bytes())
                .unwrap();
        }
    }
    assert!(workers[0].finish().is_ok());
    assert!(workers[1].finish().is_ok());
    assert!(client.finish().unwrap_err().contains("Verification"));
    assert!(client.local_tensors().is_empty());
    assert!(client.finish().is_err());
    assert!(client.lookup(0).is_err());
}

#[test]
fn invalid_repeated_malformed_partial_and_cancelled_calls_retire_both_adapters() {
    for mode in 0..12 {
        let contract = runtime("qwen2", true, 1, 1);
        let source = tensors(&contract);
        let mut client = GkmTransportClient::new(contract, &borrowed(&source)).unwrap();
        let mut workers = workers(&mut client, &borrowed(&source), || {
            runtime("qwen2", true, 1, 1)
        });
        match mode {
            0 => {
                assert!(client.starts().is_err());
            }
            1 => {
                assert!(client.linear(0, &[0; 32]).is_err());
            }
            2 => {
                assert!(client.lookup(32).is_err());
            }
            3 => {
                client.lookup(0).unwrap();
                assert!(client.lookup(0).is_err());
            }
            4 => {
                assert!(client.accept(&[], &[]).is_err());
            }
            5 => {
                assert!(client.finish().is_err());
            }
            6 => {
                client.abort();
            }
            _ => {
                let requests = client.lookup(0).unwrap();
                let replies = exchange(&mut workers, &requests);
                if mode == 7 {
                    assert!(client.accept(replies[0].as_bytes(), &[]).is_err());
                } else {
                    client
                        .accept(replies[0].as_bytes(), replies[1].as_bytes())
                        .unwrap();
                    match mode {
                        8 => {
                            assert!(client.linear(0, &[0; 31]).is_err());
                        }
                        9 => {
                            assert!(client
                                .linear(
                                    0,
                                    &[f32::NAN; 8]
                                        .iter()
                                        .flat_map(|v| v.to_le_bytes())
                                        .collect::<Vec<_>>()
                                )
                                .is_err());
                        }
                        10 => {
                            assert!(client
                                .linear(
                                    0,
                                    &[1e9f32; 8]
                                        .iter()
                                        .flat_map(|v| v.to_le_bytes())
                                        .collect::<Vec<_>>()
                                )
                                .is_err());
                        }
                        _ => {
                            assert!(client
                                .accept(replies[0].as_bytes(), replies[1].as_bytes())
                                .is_err());
                        }
                    }
                }
            }
        }
        assert!(client.local_tensors().is_empty());
        assert!(client.lookup(0).is_err());
        assert!(client.finish().is_err());
        assert!(client.publication_bytes().is_err());
        for worker in &mut workers {
            worker.abort();
            assert!(worker.execute(&[]).is_err());
            assert!(worker.finish().is_err());
        }
    }
    for replay in [false, true] {
        let contract = runtime("qwen2", true, 1, 1);
        let source = tensors(&contract);
        let mut client = GkmTransportClient::new(contract, &borrowed(&source)).unwrap();
        let mut workers = workers(&mut client, &borrowed(&source), || {
            runtime("qwen2", true, 1, 1)
        });
        let requests = client.lookup(0).unwrap();
        if replay {
            workers[0].execute(requests[0].as_bytes()).unwrap();
            assert!(workers[0].execute(requests[0].as_bytes()).is_err());
        } else {
            assert!(workers[0]
                .execute(&requests[0].as_bytes()[..requests[0].len() - 1])
                .is_err());
        }
        assert!(workers[0].execute(requests[0].as_bytes()).is_err());
        assert!(workers[0].finish().is_err());
    }
}
