//! Test-only frozen-fixture adapter and diagnostic export. Never production code.
//!
//! Source bytes are imported through the real native IEEE decoder. Semantic IDs
//! are mapped explicitly here, not guessed inside production family dispatch.
//! An unavailable checkpoint is a blocker, never copied from expected.json.
use super::*;
use pllm_garble::{
    fusefss_backend::composition::{gates, kernels, Event, Plan},
    shared_rescale_reference::Context,
};
use pllm_models::{DecoderPlan, DecoderWorkload, ModelOperator as Op};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::PathBuf};
use std::{sync::mpsc, thread};
use zeroize::Zeroizing;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_owned()
}
fn evidence() -> PathBuf {
    std::env::var_os("PLLM_FSS_DECODER_EXPORT_DIR")
        .map(|path| root().join(path))
        .unwrap_or_else(|| root().join("target/fss-decoder-conformance-v2"))
}
fn fixture_dir() -> PathBuf {
    root().join("paper/reproduction-contracts/fusefss/decoder")
}
fn hex(bytes: impl AsRef<[u8]>) -> String {
    use std::fmt::Write;
    bytes.as_ref().iter().fold(String::new(), |mut out, value| {
        write!(out, "{value:02x}").unwrap();
        out
    })
}
fn hash(bytes: &[u8]) -> String {
    hex(Sha256::digest(bytes))
}
fn read_json(path: PathBuf) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}
fn write_json(name: &str, value: &Value) {
    fs::create_dir_all(evidence()).unwrap();
    let mut bytes = serde_json::to_vec(value).unwrap();
    bytes.push(b'\n');
    fs::write(evidence().join(name), bytes).unwrap();
}
fn snapshot() -> Value {
    let mut paths = vec![
        "Cargo.lock".to_owned(),
        "crates/pllm-compiler/src/fss_decoder.rs".to_owned(),
    ];
    for entry in fs::read_dir(root().join("crates/pllm-compiler/src/fss_decoder")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|s| s == "rs") {
            paths.push(
                path.strip_prefix(root())
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_owned(),
            );
        }
    }
    for name in [
        "inputs.json",
        "expected.json",
        "source-lock.json",
        "oracle.py",
        "check_outputs.py",
        "EXCHANGE.md",
        "semantic-graphs.json",
        "semantic-expected.json",
        "semantic-source-lock.json",
        "semantic_reference.py",
        "check_semantic.py",
        "freeze_semantic.py",
        "SEMANTIC_EXCHANGE.md",
    ] {
        paths.push(format!(
            "paper/reproduction-contracts/fusefss/decoder/{name}"
        ));
    }
    for directory in [
        "crates/pllm-garble/src/fusefss_backend",
        "crates/pllm-garble/src/fusefss_backend/composition",
        "crates/pllm-models/src",
    ] {
        for entry in fs::read_dir(root().join(directory)).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|suffix| suffix == "rs") {
                paths.push(
                    path.strip_prefix(root())
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .to_owned(),
                );
            }
        }
    }
    paths.sort();
    Value::Object(
        paths
            .into_iter()
            .map(|path| {
                let digest = hash(&fs::read(root().join(&path)).unwrap());
                (path, json!(digest))
            })
            .collect(),
    )
}
fn number(v: &Value, key: &str) -> usize {
    v[key].as_u64().unwrap() as usize
}
fn public_tensor(name: &str, v: &Value) -> Result<Tensor, Error> {
    let shape: Vec<_> = v["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    let words: Vec<_> = v["words"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| u32::from_str_radix(x.as_str().unwrap(), 16).unwrap())
        .collect();
    let dtype = match v["dtype"].as_str().unwrap() {
        "f32" => WeightDtype::F32,
        "f16" => WeightDtype::F16,
        "bf16" => WeightDtype::Bf16,
        _ => return Err(Error::Domain),
    };
    Tensor::from_ieee(name, &shape, dtype, &words)
}

fn semantic(c: &Value) -> DecoderPlan {
    let d = &c["dimensions"];
    let family = c["family"].as_str().unwrap();
    let config = if family == "gpt2" {
        json!({"model_type":family,"n_embd":d["hidden"],"n_head":d["heads"],"n_layer":d["layers"],
            "n_positions":d["max_positions"],"n_inner":d["intermediate"],"vocab_size":d["vocab"],
            "tie_word_embeddings":c["tied_head"]})
    } else {
        json!({"model_type":family,"hidden_size":d["hidden"],"intermediate_size":d["intermediate"],
            "num_hidden_layers":d["layers"],"num_attention_heads":d["heads"],"num_key_value_heads":d["kv_heads"],
            "head_dim":d["head_dim"],"vocab_size":d["vocab"],"max_position_embeddings":d["max_positions"],
            "hidden_act":"silu","rms_norm_eps":"0.000001","rope_theta":10000,
            "attention_bias":family=="qwen2",
            "attention_dropout":0,"use_sliding_window":false,"sliding_window":null,
            "rope_scaling":null,"use_cache":true,"max_window_layers":d["layers"],
            "tie_word_embeddings":c["tied_head"]})
    };
    pllm_models::lower_model_json(
        &serde_json::to_vec(&config).unwrap(),
        DecoderWorkload {
            batch: 1,
            max_input_tokens: c["prefix"].as_array().unwrap().len() as u64,
            max_new_tokens: number(c, "generation_count") as u64,
        },
    )
    .unwrap()
}

/// Test-side source-role mapping. No source ranges/digests are caller supplied.
fn public_role(id: &str, bias: bool) -> Option<String> {
    if id == "token_lookup" {
        return Some("embedding".into());
    }
    if id == "position_lookup" {
        return Some("positions".into());
    }
    if id == "final_norm" {
        return Some(format!("final.{}", if bias { "beta" } else { "gamma" }));
    }
    if id == "output_head" {
        return Some("head".into());
    }
    let (base, suffix) = id.rsplit_once('.')?;
    let role = match suffix {
        "input_norm" => "norm1",
        "post_norm" => "norm2",
        "q_norm" => "q_norm",
        "k_norm" => "k_norm",
        "q_linear" => "q",
        "k_linear" => "k",
        "v_linear" => "v",
        "o_proj" => "o",
        "up_proj" => "up",
        "down_proj" => "down",
        "gate_proj" => "gate",
        _ => return None,
    };
    let ending = if suffix.ends_with("norm") {
        if bias {
            "beta"
        } else {
            "gamma"
        }
    } else if bias {
        "bias"
    } else {
        "weight"
    };
    Some(format!("{base}.{role}.{ending}"))
}

fn source(c: &Value, semantic: &DecoderPlan) -> Result<Source, Error> {
    let imported: BTreeMap<String, Tensor> = c["tensors"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, value)| Ok((name.clone(), public_tensor(name, value)?)))
        .collect::<Result<_, Error>>()?;
    let mut tensors = BTreeMap::<String, Tensor>::new();
    let mut rotary = Vec::new();
    for op in &semantic.prefill.operations {
        if op.operator == Op::RotaryEmbedding {
            let f32s = |name: &str| -> Vec<f32> {
                c["tensors"][name]["words"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| f32::from_bits(u32::from_str_radix(v.as_str().unwrap(), 16).unwrap()))
                    .collect()
            };
            rotary.push(RotaryCoefficients::from_f32(
                &op.id,
                10000,
                [
                    number(&c["dimensions"], "max_positions"),
                    number(&c["dimensions"], "head_dim"),
                ],
                &f32s("rope.cos"),
                &f32s("rope.sin"),
            )?);
        }
        for (key, is_bias) in [("weight", false), ("bias", true)] {
            let Some(target) = op.attributes[key].as_str() else {
                continue;
            };
            if tensors.contains_key(target) {
                continue;
            }
            if op.id.ends_with(".qkv_linear") {
                let base = op.id.rsplit_once('.').unwrap().0;
                let mut codes = Vec::new();
                for role in ["q", "k", "v"] {
                    let name = format!("{base}.{role}.{}", if is_bias { "bias" } else { "weight" });
                    codes.extend_from_slice(
                        imported
                            .get(&name)
                            .ok_or(Error::Domain)?
                            .values()
                            .ok_or(Error::Context)?,
                    );
                }
                let h = number(&c["dimensions"], "hidden");
                let (shape, values) = if is_bias {
                    (vec![3 * h], codes)
                } else {
                    let mut transposed = vec![0; codes.len()];
                    for out in 0..3 * h {
                        for input in 0..h {
                            transposed[input * 3 * h + out] = codes[out * h + input];
                        }
                    }
                    (vec![h, 3 * h], transposed)
                };
                tensors.insert(target.into(), Tensor::from_q12(target, &shape, &values)?);
                continue;
            }
            let mut role = public_role(&op.id, is_bias).ok_or(Error::Unsupported)?;
            if op.operator == Op::OutputHead && c["tied_head"] == true {
                role = "embedding".into();
            }
            let t = imported.get(&role).ok_or(Error::Domain)?;
            let transpose = !is_bias && op.attributes["weight_layout"] == "input_output";
            let mut shape = t.shape().to_vec();
            let resident = t.values().ok_or(Error::Context)?;
            let mut values = resident.to_vec();
            if transpose {
                if shape.len() != 2 {
                    return Err(Error::Domain);
                }
                let (out, input) = (shape[0], shape[1]);
                for o in 0..out {
                    for i in 0..input {
                        values[i * out + o] = resident[o * input + i];
                    }
                }
                shape.swap(0, 1);
            }
            tensors.insert(target.into(), Tensor::from_q12(target, &shape, &values)?);
        }
    }
    let source = Source::new(tensors.into_values().collect())?;
    if rotary.is_empty() {
        Ok(source)
    } else {
        source.with_rotary(rotary)
    }
}

type KernelRecord = ([u8; 32], [u8; 32], [u8; 32], u64, [u8; 32]);
fn kernel_record(c: Context, plan: &Plan<'_>) -> KernelRecord {
    (
        c.plan,
        c.session,
        c.operation,
        c.first_tensor_index,
        plan.digest(),
    )
}
#[derive(Default)]
struct Visits {
    records: Vec<KernelRecord>,
    requests: usize,
    cancelled: bool,
}
impl MaterialVisitor for Visits {
    fn visit(&mut self, c: Context, plan: &Plan<'_>) -> Result<(), Error> {
        self.records.push(kernel_record(c, plan));
        self.requests += plan.resources().secret_steps;
        Ok(())
    }
    fn cancel(&mut self) {
        self.records.clear();
        self.cancelled = true;
    }
}
#[derive(Default, Clone)]
struct Actual {
    kernels: usize,
    issued_pairs: usize,
    frames: usize,
    frame_bytes: usize,
    material_logical_bytes: usize,
}
struct Observation {
    selection: usize,
    node: usize,
    shape: Vec<usize>,
    shares: Zeroizing<Vec<u64>>,
}
struct NativeSample {
    native: NativeOutput,
    shares: Zeroizing<Vec<u64>>,
}
struct Request {
    context: Context,
    digest: [u8; 32],
    input: Zeroizing<Vec<u64>>,
    reply: mpsc::SyncSender<Result<NativeOutput, Error>>,
}
struct Harness<'a> {
    role: u8,
    send: Option<mpsc::SyncSender<Request>>,
    receive: Option<mpsc::Receiver<Request>>,
    expected: &'a [KernelRecord],
    next: usize,
    observations: Vec<Observation>,
    actual: Actual,
    cancelled: bool,
}
/// Separate scoped dealer; only test coordinator ever drives two native roles.
/// Dealer receives only public pc/context, not private worker input or register.
fn native_pair(
    plan: &Plan<'_>,
    context: Context,
    a: &[u64],
    b: &[u64],
    actual: &mut Actual,
) -> Result<[NativeSample; 2], Error> {
    thread::scope(|scope| {
        let (requests, request_rx) = mpsc::sync_channel::<usize>(0);
        let (materials, material_rx) = mpsc::sync_channel(0);
        let dealer = scope.spawn(move || {
            let mut count = 0;
            while let Ok(pc) = request_rx.recv() {
                if materials.send(plan.issue_step(pc, context)).is_err() {
                    break;
                }
                count += 1;
            }
            count
        });
        let mut left = plan.runtime(0, context, a)?;
        let mut right = plan.runtime(1, context, b)?;
        let mut le = left.poll()?;
        let mut re = right.poll()?;
        let result = loop {
            match (le, re) {
                (Event::NeedMaterial { pc: a }, Event::NeedMaterial { pc: b }) => {
                    if a != b {
                        break Err(Error::Context);
                    }
                    requests.send(a).map_err(|_| Error::Context)?;
                    let [a, b] = material_rx.recv().map_err(|_| Error::Context)??;
                    le = left.feed(a)?;
                    re = right.feed(b)?;
                }
                (Event::Frame(a), Event::Frame(b)) => {
                    actual.frames += 2;
                    actual.frame_bytes += a.as_bytes().len() + b.as_bytes().len();
                    le = left.advance(b.as_bytes())?;
                    re = right.advance(a.as_bytes())?;
                }
                (Event::Output(a), Event::Output(b)) => {
                    let aa = Zeroizing::new(a.shares().to_vec());
                    let bb = Zeroizing::new(b.shares().to_vec());
                    break Ok([
                        NativeSample {
                            native: NativeOutput::from_native(a)?,
                            shares: aa,
                        },
                        NativeSample {
                            native: NativeOutput::from_native(b)?,
                            shares: bb,
                        },
                    ]);
                }
                _ => break Err(Error::Context),
            }
        };
        drop(requests);
        drop(material_rx);
        let issued = dealer.join().map_err(|_| Error::Context)?;
        actual.kernels += 1;
        actual.issued_pairs += issued;
        actual.material_logical_bytes += plan.resources().private_material_bytes_total;
        if result.is_ok() && issued != plan.resources().secret_steps {
            return Err(Error::Context);
        }
        result
    })
}
impl Driver for Harness<'_> {
    fn run(&mut self, c: Context, plan: &Plan<'_>, input: &[u64]) -> Result<NativeOutput, Error> {
        if self.expected.get(self.next) != Some(&kernel_record(c, plan)) {
            return Err(Error::Context);
        }
        self.next += 1;
        if self.role == 1 {
            let (reply, rx) = mpsc::sync_channel(0);
            self.send
                .as_ref()
                .unwrap()
                .send(Request {
                    context: c,
                    digest: plan.digest(),
                    input: Zeroizing::new(input.to_vec()),
                    reply,
                })
                .map_err(|_| Error::Context)?;
            rx.recv().map_err(|_| Error::Context)?
        } else {
            let request = self
                .receive
                .as_ref()
                .unwrap()
                .recv()
                .map_err(|_| Error::Context)?;
            if request.digest != plan.digest()
                || kernel_record(request.context, plan) != kernel_record(c, plan)
            {
                return Err(Error::Context);
            }
            match native_pair(plan, c, input, &request.input, &mut self.actual) {
                Ok([a, b]) => {
                    request
                        .reply
                        .send(Ok(b.native))
                        .map_err(|_| Error::Context)?;
                    Ok(a.native)
                }
                Err(e) => {
                    let _ = request.reply.send(Err(e));
                    Err(e)
                }
            }
        }
    }
    fn cancel(&mut self) {
        self.cancelled = true;
        self.send = None;
        self.receive = None;
    }
    fn observe(&mut self, selection: usize, node: usize, shape: &[usize], shares: &[u64]) {
        self.observations.push(Observation {
            selection,
            node,
            shape: shape.to_vec(),
            shares: Zeroizing::new(shares.to_vec()),
        });
    }
}
fn reconstruct(a: &[u64], b: &[u64]) -> Vec<i64> {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&a, &b)| {
            let w = a.wrapping_add(b) & MASK;
            if w & (1 << 49) == 0 {
                w as i64
            } else {
                w as i64 - (1 << 50)
            }
        })
        .collect()
}
type ClearObservation = (usize, usize, Vec<usize>, Vec<i64>);

fn execute(
    program: &Program,
    c: &Value,
    case_index: usize,
) -> (Vec<i64>, Vec<ClearObservation>, Actual) {
    let context = Context {
        plan: program.digest(),
        session: [(case_index + 17) as u8; 32],
        operation: [(case_index + 31) as u8; 32],
        first_tensor_index: 0,
    };
    let mut visits = Visits::default();
    program.visit_material(context, &mut visits).unwrap();
    assert!(!visits.cancelled);
    assert_eq!(
        visits.requests,
        program.resources().primitive_requests_per_party
    );
    assert_eq!(visits.records.len(), program.resources().kernels);
    let (send, receive) = mpsc::sync_channel(0);
    let records = visits.records.as_slice();
    let prefix: Vec<u64> = c["prefix"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    let left: Vec<u64> = (0..prefix.len())
        .map(|i| 123456789 + 71893 * i as u64)
        .collect();
    let right: Vec<u64> = prefix
        .iter()
        .zip(&left)
        .map(|(&v, &s)| v.wrapping_sub(s) & MASK)
        .collect();
    let (a, b) = thread::scope(|scope| {
        let worker = |role, input, send, receive| {
            let mut harness = Harness {
                role,
                send,
                receive,
                expected: records,
                next: 0,
                observations: Vec::new(),
                actual: Actual::default(),
                cancelled: false,
            };
            let result = Worker::new(program, role, context, Zeroizing::new(input))
                .unwrap()
                .execute(&mut harness);
            assert!(
                !harness.cancelled,
                "role {role}: {:?}",
                result.as_ref().err()
            );
            assert_eq!(harness.next, records.len());
            (result.unwrap(), harness.observations, harness.actual)
        };
        let a = scope.spawn(move || worker(0, left, None, Some(receive)));
        let b = scope.spawn(move || worker(1, right, Some(send), None));
        (a.join().unwrap(), b.join().unwrap())
    });
    assert_eq!(a.1.len(), b.1.len());
    if program.checked_ranges() {
        assert_eq!(
            a.0.validity_share()
                .unwrap()
                .wrapping_add(b.0.validity_share().unwrap())
                & MASK,
            1
        );
    } else {
        assert_eq!((a.0.validity_share(), b.0.validity_share()), (None, None));
    }
    let trace =
        a.1.into_iter()
            .zip(b.1)
            .map(|(a, b)| {
                assert_eq!(
                    (a.selection, a.node, &a.shape),
                    (b.selection, b.node, &b.shape)
                );
                (
                    a.selection,
                    a.node,
                    a.shape,
                    reconstruct(&a.shares, &b.shares),
                )
            })
            .collect();
    assert_eq!(a.2.issued_pairs, visits.requests);
    assert_eq!(a.2.frame_bytes, program.resources().frame_bytes);
    assert_eq!(
        a.2.material_logical_bytes,
        program.resources().material_bytes
    );
    (reconstruct(a.0.values(), b.0.values()), trace, a.2)
}

/// Crop fixed-capacity native node arrays to this selection's full prefix.
/// No arithmetic recomputation, interpolation, or expected-value substitution.
fn crop(shape: &[usize], values: &[i64], prefix: usize) -> Vec<i64> {
    match shape {
        [1, _, width] => values[..prefix * width].to_vec(),
        [1, heads, sequence, width] => {
            let mut out = Vec::new();
            for h in 0..*heads {
                for p in 0..prefix {
                    out.extend_from_slice(
                        &values[(h * sequence + p) * width..(h * sequence + p + 1) * width],
                    );
                }
            }
            out
        }
        _ => values.to_vec(),
    }
}
fn mapped_checkpoints(
    semantic: &DecoderPlan,
    c: &Value,
    trace: &[(usize, usize, Vec<usize>, Vec<i64>)],
) -> (BTreeMap<String, Vec<i64>>, Vec<Value>) {
    let initial = c["prefix"].as_array().unwrap().len();
    let mut checkpoints = BTreeMap::new();
    let mut map = Vec::new();
    let embedded_sum = semantic
        .prefill
        .operations
        .iter()
        .any(|op| op.id == "embedding_sum");
    for (selection, index, shape, values) in trace {
        let op = &semantic.prefill.operations[*index];
        let prefix = initial + selection;
        let base = format!("step.{selection}");
        let cropped = crop(shape, values, prefix);
        let mut emitted = Vec::<String>::new();
        let mut emit = |name: String, values: Vec<i64>| {
            emitted.push(name.clone());
            assert!(checkpoints.insert(name, values).is_none());
        };
        match op.id.as_str() {
            "token_lookup" => {
                emit(format!("{base}.token_embedding"), cropped.clone());
                if !embedded_sum {
                    emit(format!("{base}.embedding_output"), cropped.clone());
                }
            }
            "embedding_sum" => emit(format!("{base}.embedding_output"), cropped.clone()),
            "final_norm" => emit(format!("{base}.final_norm.output"), cropped.clone()),
            _ => {
                if let Some((layer, suffix)) = op.id.rsplit_once('.') {
                    let name = format!("{base}.{layer}");
                    let suffix_map = match suffix {
                        "input_norm" => Some("norm1.output"),
                        "post_norm" => Some("norm2.output"),
                        "q_linear" | "q_slice" => Some("q.output"),
                        "k_linear" | "k_slice" => Some("k.output"),
                        "v_linear" | "v_slice" => Some("v.output"),
                        "q_heads" => Some("q.physical"),
                        "k_heads" => Some("k.physical"),
                        "v_heads" => Some("v.physical"),
                        "rope_q" => Some("q_rope.output"),
                        "rope_k" => Some("k_rope.output"),
                        "o_proj" => Some("o.output"),
                        "up_proj" => Some("up.output"),
                        "gate_proj" => Some("gate.output"),
                        "down_proj" => Some("down.output"),
                        "gelu" => Some("gelu.output"),
                        "silu" => Some("silu.output"),
                        "gated_multiply" => Some("gated_output"),
                        "attention_values" => Some("attention.physical"),
                        "attention_hidden" => Some("attention.merged"),
                        "attention_residual" => Some("attention_residual"),
                        "mlp_residual" => Some("mlp_residual"),
                        _ => None,
                    };
                    if let Some(suffix) = suffix_map {
                        emit(format!("{name}.{suffix}"), cropped.clone());
                    }
                    if matches!(suffix, "q_norm" | "k_norm") {
                        let hd = shape[3];
                        for head in 0..shape[1] {
                            emit(
                                format!("{name}.{suffix}.head.{head}.output"),
                                cropped[head * prefix * hd..(head + 1) * prefix * hd].to_vec(),
                            );
                        }
                    }
                    if matches!(suffix, "attention_scores" | "attention_scale" | "softmax") {
                        let sequence = shape[2];
                        for head in 0..shape[1] {
                            let mut scores = Vec::new();
                            for p in 0..prefix {
                                let row = values[(head * sequence + p) * sequence
                                    ..(head * sequence + p) * sequence + prefix]
                                    .to_vec();
                                if suffix == "softmax" {
                                    emit(
                                        format!("{name}.head.{head}.softmax.{p}.output"),
                                        row.clone(),
                                    );
                                }
                                scores.extend(row);
                            }
                            if suffix != "softmax" {
                                emit(
                                    format!(
                                        "{name}.head.{head}.{}",
                                        if suffix == "attention_scores" {
                                            "qk_rescaled"
                                        } else {
                                            "scores"
                                        }
                                    ),
                                    scores,
                                );
                            }
                        }
                    }
                }
            }
        }
        map.push(
            json!({"selection":selection,"semantic_index":index,"semantic_id":op.id,
            "native_shape":shape,"active_prefix":prefix,"exchange_tensors":emitted}),
        );
    }
    (checkpoints, map)
}

fn canonical_digest(value: &Value) -> String {
    let mut bytes = serde_json::to_vec(value).unwrap();
    bytes.push(b'\n');
    hash(&bytes)
}

fn semantic_operator_map(semantic: &DecoderPlan) -> Vec<Value> {
    semantic
        .prefill
        .operations
        .iter()
        .enumerate()
        .map(|(index, op)| {
            json!({"semantic_index":index,"id":op.id,"operator":format!("{:?}",op.operator),
            "inputs":op.inputs,"declared_shape":op.output_shape,"attributes":op.attributes})
        })
        .collect()
}

/// Test-only complete semantic observations. Never reads semantic-expected.json
/// or fills values with an oracle/native clear recomputation. Frozen graph is
/// metadata only and must match ALL actually lowered operations before export.
fn semantic_native_result(
    semantic: &DecoderPlan,
    c: &Value,
    graph: &Value,
    tokens: &[i64],
    trace: &[(usize, usize, Vec<usize>, Vec<i64>)],
) -> Value {
    assert_eq!(graph["fixture_id"], c["id"]);
    assert_eq!(json!(semantic_operator_map(semantic)), graph["operators"]);
    let count = semantic.prefill.operations.len();
    let selections = number(c, "generation_count");
    let initial = c["prefix"].as_array().unwrap().len();
    let sequence = initial + selections - 1;
    assert_eq!(tokens.len(), selections);
    assert_eq!(trace.len(), count * selections);
    let mut nodes = Vec::new();
    for (ordinal, (selection, index, shape, values)) in trace.iter().enumerate() {
        assert_eq!((*selection, *index), (ordinal / count, ordinal % count));
        let op = &semantic.prefill.operations[*index];
        let kind = format!("{:?}", op.operator);
        let fractional = if matches!(kind.as_str(), "GreedyTokenSelection" | "TokenFeedback") {
            assert_eq!(values.as_slice(), &[tokens[*selection]]);
            0
        } else {
            12
        };
        assert_eq!(shape.iter().product::<usize>(), values.len());
        nodes.push(
            json!({"selection":selection,"semantic_index":index,"id":op.id,
            "shape":shape,"fractional":fractional,"values":values}),
        );
    }
    let mut public: BTreeMap<String, Value> = c["tensors"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();
    assert!(public
        .insert("attention_scale".into(), c["attention_scale"].clone())
        .is_none());
    let source_encodings: Vec<_> = public
        .iter()
        .map(|(name, value)| {
            // These values come from real native exact IEEE -> Q12 conversion,
            // not frozen expected JSON. IEEE words and shapes are public inputs.
            let native = public_tensor(name, value).unwrap();
            json!({"name":name,"dtype":value["dtype"],"shape":value["shape"],
            "ieee_words":value["words"],"q12_values":native.values()})
        })
        .collect();
    let mut prefix: Vec<i64> = c["prefix"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_i64().unwrap())
        .collect();
    let mut states = Vec::new();
    for (selection, token) in tokens.iter().enumerate() {
        let mut buffer = prefix.clone();
        buffer.resize(sequence, 0);
        let mut after = prefix.clone();
        after.push(*token);
        states.push(json!({"selection":selection,"active_prefix_ids_private":prefix,
            "input_token_buffer_private":buffer,"active_positions":(0..initial+selection).collect::<Vec<_>>(),
            "physical_positions_computed":(0..sequence).collect::<Vec<_>>(),"selected_id_private":token,
            "logical_prefix_after_selection_private":after,"persistent_kv":false,"released":false}));
        prefix = after;
    }
    json!({"schema":"pllm.fusefss.semantic-execution.v1","fixture_id":c["id"],"profile":c["profile"],
        "input_sha256":canonical_digest(c),"semantic_graph_sha256":canonical_digest(graph),
        "source_encodings":source_encodings,"nodes":nodes,"states":states,
        "final":{"recipient":"client-only","token_ids":prefix,"generated_ids":tokens,
            "selection_count":selections,"persistent_kv":false}})
}

#[path = "conformance_tests.rs"]
mod conformance_tests;
