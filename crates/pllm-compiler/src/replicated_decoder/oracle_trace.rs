//! TEST-ONLY native/independent decoder conformance capture owner.
//!
//! Integration snippets (no production fields, hashes, or feature flags):
//! runtime.rs, module scope:
//!   #[cfg(test)]
//!   #[path = "oracle_trace.rs"]
//!   pub(super) mod oracle_trace;
//! runtime.rs, FIRST line of Worker::retire_inputs():
//!   #[cfg(test)]
//!   oracle_trace::retiring(self);
//! program.rs, graph(), immediately BEFORE values.insert(op.id.clone(), output):
//!   #[cfg(test)]
//!   super::runtime::oracle_trace::boundary(
//!       site.selection, &op.id, &output.shape,
//!       output.words.iter().collect(),
//!   );
//! For the explicit physical-lane mapper, add before `let output = match ...`:
//!   #[cfg(test)] let oracle_start = self.nodes.len();
//! and after that match (before values.insert):
//!   #[cfg(test)]
//!   super::runtime::oracle_trace::span(
//!       site.selection, &op.id, oracle_start, self.nodes.len());
//! program.rs, selection loop BEFORE builder.graph(...):
//!   #[cfg(test)]
//!   super::runtime::oracle_trace::boundary(
//!       output, "input.tokens", &tokens.shape, tokens.words.iter().collect());
//! program.rs, AFTER tokens.shape = vec![1, tokens.words.len()]:
//!   #[cfg(test)]
//!   super::runtime::oracle_trace::boundary(
//!       output, "feedback.history", &tokens.shape, tokens.words.iter().collect());
//!
//! Start capture BEFORE compilation. Retirement copies only explicitly requested
//! coordinates, before native liveness retirement; no production owner survives
//! longer. drain() releases complete three-party checkpoints. A source-present
//! audit must drain/replay each bounded checkpoint, not collect its full trace.
//! Boundary metadata itself must also be publicly tiled for a large checkpoint.
//!
//! Actual truncation masks are NOT guessed from arithmetic outputs. Existing
//! private seed-exchange messages are copied ONLY inside this all-party unit
//! test. Outgoing TruncAPr frame headers identify real invocations; public source
//! cursors locate the source AES fills. Python's pinned independent AES fixture
//! reconstructs masks. No core secret/debug API or deterministic seed injection
//! is necessary. Never print or persist these records in benchmark artifacts.
//!
//! Generated conformance runs ordinary native workers and independently replays
//! every truncation lane in Python. Larger geometries must supply reviewed lane
//! maps; the fixture mapper explicitly rejects unsupported layouts. Checked
//! diagnostics are immutable public placements, not private-data-selected sites.

use super::*;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

const CAPTURE_WORDS: usize = 1 << 20;

#[derive(Clone)]
struct Boundary {
    selection: usize,
    id: String,
    shape: Vec<usize>,
    words: Vec<(usize, usize)>,
    oracle_name: Option<String>,
    batch_dimension: bool,
}

type ReadyShares = [Option<Vec<(u64, u64)>>; 3];

#[derive(Default)]
struct Capture {
    enabled: bool,
    boundaries: Vec<Boundary>,
    spans: Vec<serde_json::Value>,
    cache: [BTreeMap<(usize, usize), (u64, u64)>; 3],
    finished: [BTreeSet<usize>; 3],
    ready: BTreeMap<usize, ReadyShares>,
    words: usize,
    metadata_words: usize,
    range_context: Option<RangeContext>,
    guards: Vec<serde_json::Value>,
}

struct RangeContext {
    selection: usize,
    name: String,
    shape: Vec<usize>,
    original: Vec<(usize, usize)>,
    safe: Vec<(usize, usize)>,
    valid: Vec<(usize, usize)>,
    reductions: Vec<(usize, usize)>,
    sticky: Option<usize>,
    sticky_tiles: Vec<(usize, usize)>,
    bound: Option<i64>,
}

pub(in crate::replicated_decoder) fn range_begin(
    selection: usize,
    op: &str,
    shape: &[usize],
    original: Vec<(usize, usize)>,
) {
    let name = if op == "final_norm" {
        format!("s{selection}/final_norm/input")
    } else {
        let mut parts = op.rsplitn(2, '.');
        let kind = parts.next().unwrap();
        let layer = parts.next().expect("layer RMSNorm name");
        let kind = if kind == "post_norm" {
            "post_attention_norm"
        } else {
            kind
        };
        format!("s{selection}/{layer}/{kind}/input")
    };
    CAPTURE.with(|slot| {
        let mut state = slot.borrow_mut();
        assert!(state.range_context.is_none());
        state.range_context = Some(RangeContext {
            selection,
            name,
            shape: shape.to_vec(),
            original,
            safe: vec![],
            valid: vec![],
            reductions: vec![],
            sticky: None,
            sticky_tiles: vec![],
            bound: None,
        });
    });
}

pub(in crate::replicated_decoder) fn range_tile(
    bound: i64,
    owner: usize,
    lanes: usize,
    reduction: usize,
    sticky: usize,
) {
    CAPTURE.with(|slot| {
        let mut state = slot.borrow_mut();
        let context = state.range_context.as_mut().expect("RMSNorm guard context");
        context.bound = Some(bound);
        context.safe.extend((0..lanes).map(|i| (owner, i)));
        context.valid.extend((lanes..2 * lanes).map(|i| (owner, i)));
        context.reductions.push((reduction, 0));
        context.sticky = Some(sticky);
        context.sticky_tiles.push((sticky, 0));
    });
}

pub(in crate::replicated_decoder) fn range_end() {
    let context = CAPTURE.with(|slot| slot.borrow_mut().range_context.take().unwrap());
    let Some(bound) = context.bound else {
        return;
    }; // public static proof, no guard/work
    let s = context.selection;
    let prefix = context.name.clone() + "/checked_range/";
    diagnostic(s, &context.name, &context.shape, context.original);
    diagnostic(s, &(prefix.clone() + "safe"), &context.shape, context.safe);
    diagnostic(
        s,
        &(prefix.clone() + "valid"),
        &context.shape,
        context.valid,
    );
    let tiles = context.reductions.len();
    diagnostic(
        s,
        &(prefix.clone() + "all_valid"),
        if tiles == 1 {
            &[]
        } else {
            std::slice::from_ref(&tiles)
        },
        context.reductions,
    );
    diagnostic(
        s,
        &(prefix.clone() + "tile_sticky"),
        &[tiles],
        context.sticky_tiles,
    );
    diagnostic(
        s,
        &(prefix + "sticky"),
        &[],
        vec![(context.sticky.unwrap(), 0)],
    );
    CAPTURE.with(|slot| {
        slot.borrow_mut().guards.push(serde_json::json!({
            "name": context.name, "lower": -bound, "upper": bound, "fallback": 0,
        }))
    });
}

thread_local! {
    static CAPTURE: RefCell<Capture> = RefCell::new(Capture::default());
}

pub(in crate::replicated_decoder) fn enabled() -> bool {
    CAPTURE.with(|slot| slot.borrow().enabled)
}

struct Session;
impl Session {
    fn start() -> Self {
        CAPTURE.with(|slot| {
            assert!(!slot.borrow().enabled, "nested oracle capture");
            *slot.borrow_mut() = Capture {
                enabled: true,
                ..Capture::default()
            };
        });
        Self
    }
}
impl Drop for Session {
    fn drop(&mut self) {
        CAPTURE.with(|slot| {
            let mut state = slot.borrow_mut();
            for cache in &mut state.cache {
                for pair in cache.values_mut() {
                    *pair = (0, 0);
                }
            }
            for parties in state.ready.values_mut() {
                for pairs in parties.iter_mut().flatten() {
                    for pair in pairs {
                        *pair = (0, 0);
                    }
                }
            }
            *state = Capture::default();
        });
    }
}

pub(in crate::replicated_decoder) fn boundary(
    selection: usize,
    id: &str,
    shape: &[usize],
    words: Vec<(usize, usize)>,
) {
    CAPTURE.with(|slot| {
        let mut state = slot.borrow_mut();
        if !state.enabled {
            return;
        }
        assert!(!words.is_empty() && words.len() <= CAPTURE_WORDS);
        assert_eq!(shape.iter().product::<usize>(), words.len());
        assert!(!state
            .boundaries
            .iter()
            .any(|b| b.selection == selection && b.id == id));
        state.metadata_words = state.metadata_words.checked_add(words.len()).unwrap();
        assert!(
            state.metadata_words <= CAPTURE_WORDS,
            "bounded semantic coordinate metadata exhausted"
        );
        state.boundaries.push(Boundary {
            selection,
            id: id.to_owned(),
            shape: shape.to_vec(),
            words,
            oracle_name: None,
            batch_dimension: true,
        });
    });
}

/// Explicit diagnostic view, whose shape already omits batch. Compiler must
/// register actual native coordinates, never reconstructed/fabricated values.
/// Useful for guard input/safe/valid and final aggregate acceptance. Guard-tile
/// scalar reductions require explicit tile metadata; do not pretend a per-tile
/// all_valid scalar is the whole-site aggregate when there is more than one tile.
pub(in crate::replicated_decoder) fn diagnostic(
    selection: usize,
    oracle_name: &str,
    shape: &[usize],
    words: Vec<(usize, usize)>,
) {
    CAPTURE.with(|slot| {
        let mut state = slot.borrow_mut();
        if !state.enabled {
            return;
        }
        assert!(!words.is_empty() && words.len() <= CAPTURE_WORDS);
        assert_eq!(shape.iter().product::<usize>(), words.len());
        assert!(!state
            .boundaries
            .iter()
            .any(|b| b.selection == selection && b.id == oracle_name));
        state.metadata_words = state.metadata_words.checked_add(words.len()).unwrap();
        assert!(
            state.metadata_words <= CAPTURE_WORDS,
            "bounded diagnostic coordinate metadata exhausted"
        );
        state.boundaries.push(Boundary {
            selection,
            id: oracle_name.to_owned(),
            shape: shape.to_vec(),
            words,
            oracle_name: Some(oracle_name.to_owned()),
            batch_dimension: false,
        });
    });
}

pub(in crate::replicated_decoder) fn span(selection: usize, id: &str, start: usize, end: usize) {
    CAPTURE.with(|slot| {
        let mut state = slot.borrow_mut();
        if state.enabled {
            state.spans.push(
                serde_json::json!({"selection": selection, "id": id, "start": start, "end": end}),
            );
        }
    });
}

pub(super) fn retiring(worker: &Worker) {
    CAPTURE.with(|slot| {
        let mut state = slot.borrow_mut();
        if !state.enabled {
            return;
        }
        if matches!(
            worker.program.nodes[worker.step].code,
            Code::TensorLoop { .. }
        ) && worker.loop_offset < worker.program.nodes[worker.step].lanes
        {
            return;
        }
        let role = worker.party as usize;
        let requested: BTreeSet<_> = state
            .boundaries
            .iter()
            .flat_map(|b| b.words.iter().copied())
            .filter(|(owner, _)| *owner == worker.step)
            .collect();
        for coordinate @ (owner, index) in requested {
            let pair = worker
                .word(owner, index)
                .expect("native checkpoint coordinate unavailable");
            if state.cache[role].insert(coordinate, pair).is_none() {
                state.words += 2;
            }
        }
        for index in 0..state.boundaries.len() {
            if state.finished[role].contains(&index) {
                continue;
            }
            let words = &state.boundaries[index].words;
            if !words
                .iter()
                .all(|word| state.cache[role].contains_key(word))
            {
                continue;
            }
            let pairs: Vec<_> = words.iter().map(|word| state.cache[role][word]).collect();
            state.words += 2 * pairs.len();
            state
                .ready
                .entry(index)
                .or_insert_with(|| std::array::from_fn(|_| None))[role] = Some(pairs);
            state.finished[role].insert(index);
        }
        // A copied coordinate can retire once no unresolved boundary needs it.
        let needed: BTreeSet<_> = state
            .boundaries
            .iter()
            .enumerate()
            .filter(|(index, _)| !state.finished[role].contains(index))
            .flat_map(|(_, b)| b.words.iter().copied())
            .collect();
        let before = state.cache[role].len();
        state.cache[role].retain(|coordinate, _| needed.contains(coordinate));
        state.words -= 2 * (before - state.cache[role].len());
        assert!(
            state.words <= CAPTURE_WORDS,
            "drain bounded checkpoints sooner"
        );
    });
}

fn drain() -> Vec<serde_json::Value> {
    CAPTURE.with(|slot| {
        let mut state = slot.borrow_mut();
        let complete: Vec<_> = state
            .ready
            .iter()
            .filter(|(_, parties)| parties.iter().all(Option::is_some))
            .map(|(index, _)| *index)
            .collect();
        complete
            .into_iter()
            .map(|index| {
                let parties = state.ready.remove(&index).unwrap().map(Option::unwrap);
                let boundary = &state.boundaries[index];
                for role in 0..3 {
                    for lane in 0..boundary.words.len() {
                        assert_eq!(
                            parties[role][lane].1,
                            parties[(role + 1) % 3][lane].0,
                            "overlapping replicated components disagree"
                        );
                    }
                }
                let mut record = serde_json::json!({
                    "selection": boundary.selection, "id": boundary.id,
                    "shape": boundary.shape,
                    // Reconstruct this authorized public-fixture checkpoint now;
                    // never retain all three parties' histories in the record queue.
                    "values": (0..parties[0].len()).map(|lane|
                        parties.iter().fold(0u64, |sum, p| sum.wrapping_add(p[lane].0)) as i64
                    ).collect::<Vec<_>>(),
                });
                if let Some(name) = &boundary.oracle_name {
                    record["oracle_name"] = serde_json::json!(name);
                    record["batch_dimension"] = serde_json::json!(boundary.batch_dimension);
                }
                state.words -= 6 * parties[0].len();
                record
            })
            .collect()
    })
}

fn drain_into(records: &mut Vec<serde_json::Value>, words: &mut usize) {
    for record in drain() {
        *words += record["values"].as_array().unwrap().len();
        assert!(*words <= CAPTURE_WORDS,
            "bounded clear-fixture history exhausted; use incremental comparator instead of retaining history");
        records.push(record);
    }
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut text, byte| {
        write!(text, "{byte:02x}").unwrap();
        text
    })
}

/// Observe outgoing headers immediately after the originating native transition.
/// Do NOT observe queued frames later: the sender's cursors may have advanced.
fn truncation(worker: &Worker, frame: &PeerBytes) -> Option<serde_json::Value> {
    let raw = frame.as_bytes();
    if raw.starts_with(b"PUT2") {
        assert!(
            raw.len() >= 92 && &raw[48..52] == b"PUC1",
            "unsupported compact tensor format"
        );
        assert!(raw[48 + 38] < 10, "compact format cannot carry truncation");
        return None;
    }
    let (inner, tensor_step, invocation) = if raw.starts_with(b"PUT1") {
        assert!(raw.len() >= 92);
        (
            &raw[48..],
            Some(u32::from_le_bytes(raw[44..48].try_into().unwrap())),
            Some(u64::from_le_bytes(raw[36..44].try_into().unwrap())),
        )
    } else {
        (raw, None, None)
    };
    assert!(inner.starts_with(b"PUM1") && inner.len() >= 44);
    // Pinned wire.rs: MaskedInput=10; Correlation=11. Other protocols are NOT
    // truncations, even though they consume PRSS and change source cursors.
    let phase = inner[38];
    if phase != 10 && phase != 11 {
        return None;
    }
    let count = u32::from_le_bytes(inner[40..44].try_into().unwrap()) as usize;
    let lanes = if phase == 11 {
        assert_eq!(count % 2, 0);
        count / 2
    } else {
        count
    };
    let cursors = match &worker.active {
        Active::Tensor(executor) => executor.cursors(),
        Active::Product(runtime) | Active::Idle(runtime) => runtime.cursors(),
        _ => panic!("unexpected truncation owner"),
    };
    Some(serde_json::json!({
        "binding": hex(&inner[4..36]), "role": worker.party as u8, "to": inner[37],
        "phase": phase, "lanes": lanes, "node": worker.step,
        "loop_offset": worker.loop_offset, "tensor_step": tensor_step,
        "invocation": invocation, "public_end": cursors.public_blocks,
        "pair_end": cursors.pair_blocks,
    }))
}

fn make_workers(
    program: Arc<Program>,
    clear: &BTreeMap<String, Vec<i64>>,
    manifest: Option<&Manifest>,
) -> ([Worker; 3], zeroize::Zeroizing<[[u8; 16]; 3]>, u128) {
    use puma::prss::{PublicContribution, SeedMessage};
    let parties = [Party::P0, Party::P1, Party::P2];
    let mut setups: [_; 3] = std::array::from_fn(|i| program.setup([42; 32], parties[i]).unwrap());
    let public = setups.each_mut().map(|setup| setup.contribution().unwrap());
    let public_seed = public
        .iter()
        .fold(0u128, |sum, value| sum.wrapping_add(value.value()));
    for (i, setup) in setups.iter_mut().enumerate() {
        for j in 0..3 {
            if i != j {
                setup
                    .receive_public(
                        PublicContribution::decode(&public[j].encode(), parties[j]).unwrap(),
                    )
                    .unwrap();
            }
        }
    }
    let mut messages = setups
        .each_mut()
        .map(|setup| Some(setup.share_seed().unwrap().encode().unwrap()));
    let mut keys = zeroize::Zeroizing::new([[0u8; 16]; 3]);
    for role in 0..3 {
        let raw = messages[role].as_ref().unwrap().as_bytes();
        assert_eq!(raw.len(), 54);
        assert_eq!(&raw[..4], b"PUS1");
        assert_eq!(raw[36], role as u8);
        keys[role].copy_from_slice(&raw[38..54]);
    }
    for i in 0..3 {
        let from = (i + 1) % 3;
        let message = messages[from].take().unwrap();
        setups[i]
            .receive_seed(
                SeedMessage::decode(message.as_bytes(), parties[from], parties[i]).unwrap(),
            )
            .unwrap();
    }
    let mut workers = std::array::from_fn(|i| {
        let source = if let Some(manifest) = manifest {
            // Exercise the same coefficient conversion, ordered streaming,
            // source hashes and statistics recomputation as a real worker.
            let mut importer = PublicSourceImport::new(manifest).unwrap();
            for key in manifest.tensors.keys() {
                for (chunk, values) in clear[key].chunks(113).enumerate() {
                    let mut bytes = Vec::new();
                    for &value in values {
                        let raw = ((value as f64 / ONE as f64) as f32).to_le_bytes();
                        assert_eq!(super::super::source::encode_q18(&raw).unwrap(), value);
                        bytes.extend_from_slice(&raw);
                    }
                    importer.append(key, chunk * 113, &bytes).unwrap();
                }
            }
            Some(importer.finish().unwrap())
        } else {
            PublicSource::test_values(&program, clear)
        };
        Worker::new_with_source(
            program.clone(),
            setups[i].finish().unwrap(),
            [42; 32],
            source,
        )
        .unwrap()
    });
    for (key, lanes) in program.inputs() {
        let x = &clear[key];
        assert_eq!(x.len(), lanes);
        let first: Vec<_> = (0..lanes)
            .map(|i| 0x123456789abcde00u64.wrapping_add(i as u64))
            .collect();
        let second: Vec<_> = (0..lanes)
            .map(|i| 0x876543210fedcb00u64.wrapping_add(i as u64 * 7))
            .collect();
        let third: Vec<_> = x
            .iter()
            .zip(first.iter().zip(&second))
            .map(|(x, (a, b))| (*x as u64).wrapping_sub(*a).wrapping_sub(*b))
            .collect();
        let shares = [first, second, third];
        for i in 0..3 {
            workers[i]
                .import(key, &shares[i], &shares[(i + 1) % 3])
                .unwrap();
        }
    }
    (workers, keys, public_seed)
}

type Fixture = (
    Arc<Program>,
    BTreeMap<String, Vec<i64>>,
    serde_json::Value,
    Option<Manifest>,
);

fn generated(
    family: &str,
    mode: &str,
    tiled_lookup: bool,
    tiled_selection: bool,
    intermediate: usize,
) -> Result<Fixture, String> {
    generated_with_norms(
        family,
        mode,
        tiled_lookup,
        tiled_selection,
        intermediate,
        false,
    )
}

fn generated_with_norms(
    family: &str,
    mode: &str,
    tiled_lookup: bool,
    tiled_selection: bool,
    intermediate: usize,
    certified: bool,
) -> Result<Fixture, String> {
    let wide = matches!(intermediate, 131 | 133);
    let full_vocab = intermediate == 132;
    let hidden = if intermediate == 133 {
        128
    } else if wide {
        72
    } else {
        2
    };
    let vocab = if full_vocab {
        151936
    } else if wide {
        5
    } else {
        3
    };
    let config = serde_json::json!({
        "model_type": family, "hidden_size": hidden, "intermediate_size": if full_vocab { 2 } else { intermediate },
        "num_hidden_layers": if wide { 2 } else { 1 },
        "num_attention_heads": if wide { 4 } else { 1 }, "num_key_value_heads": if wide { 2 } else { 1 },
        "head_dim": if intermediate == 133 { 32 } else if wide { 18 } else { 2 }, "vocab_size": vocab, "tie_word_embeddings": false,
        "hidden_act": "silu", "attention_bias": family == "qwen2",
        "max_position_embeddings": 64, "rope_theta": if wide { 1 } else { 10000 },
        "rms_norm_eps": 0.000001, "attention_dropout": 0.0,
        "use_sliding_window": false, "use_cache": true, "max_window_layers": if wide { 2 } else { 1 },
        "layer_types": vec!["full_attention"; if wide { 2 } else { 1 }],
    });
    let raw = serde_json::to_vec(&config).unwrap();
    let shapes = Program::source_shapes_with_weights(&raw, 2, 2, WeightMode::Public)?;
    let mut clear = BTreeMap::new();
    let mut sources = BTreeMap::new();
    for (key, shape) in shapes {
        let n: usize = shape.iter().product();
        let values = if full_vocab && key == "model.embed_tokens.weight" {
            (0..n)
                .map(|i| match (i / hidden, i % hidden) {
                    (0, 0) => ONE / 4,
                    (0, _) => ONE / 8,
                    (1, 0) => -ONE / 4,
                    (1, _) => ONE / 16,
                    (row, 0) if row == vocab - 1 => ONE / 32,
                    (_, 0) => ONE / 16,
                    _ => -ONE / 8,
                })
                .collect()
        } else if full_vocab && key == "lm_head.weight" {
            // Dense small middle rows exercise real rounding across the whole
            // vocabulary, while keeping a decisive last-row/row-zero winner.
            let mut values: Vec<_> = (0..n)
                .map(|i| {
                    let row = i / hidden;
                    let coefficient = if i % hidden == 0 {
                        (row * 7 % 13) as i64 - 6
                    } else {
                        (row * 11 % 17) as i64 - 8
                    };
                    coefficient * (ONE / 8192)
                })
                .collect();
            values[0] = ONE / 32;
            values[1] = 0;
            values[(vocab - 1) * hidden] = -ONE / 16;
            values[(vocab - 1) * hidden + 1] = 0;
            values
        } else if wide && key == "model.embed_tokens.weight" {
            (0..n)
                .map(|i| {
                    let (row, col) = (i / hidden, i % hidden);
                    if col == 0 {
                        match row {
                            0 => ONE / 4,
                            1 => -ONE / 4,
                            2 if mode != "static" => 257 * ONE,
                            _ => ONE / 16,
                        }
                    } else {
                        (i as i64 % 3 - 1) * (ONE / 16)
                    }
                })
                .collect()
        } else if wide && key == "lm_head.weight" {
            let mut values = vec![0; n];
            values[0] = ONE / 32;
            values[if mode == "checked_invalid" {
                2 * hidden
            } else {
                hidden
            }] = -ONE / 32;
            values
        } else if key == "model.embed_tokens.weight" {
            vec![
                ONE / 4,
                ONE / 8,
                -ONE / 4,
                ONE / 16,
                if mode == "static" {
                    ONE / 16
                } else {
                    257 * ONE
                },
                -ONE / 8,
            ]
        } else if key == "lm_head.weight" {
            if mode == "checked_invalid" {
                vec![ONE / 32, 0, 0, 0, -ONE / 32, 0]
            } else if mode == "checked_valid" {
                vec![ONE / 32, 0, -ONE / 32, 0, 0, 0]
            } else {
                vec![ONE / 32, 0, -ONE / 32, ONE / 64, 0, ONE / 32]
            }
        } else if key.contains("norm") {
            vec![ONE; n]
        } else if wide {
            // Period 13 does not divide either head_dim=18 or hidden=72.
            // Unlike period 3, it distinguishes KV groups and query heads.
            let salt: usize = key.bytes().map(usize::from).sum();
            (0..n)
                .map(|i| (((i * 7 + salt) % 13) as i64 - 6) * (ONE / 1024))
                .collect()
        } else {
            (0..n).map(|i| (i as i64 % 3 - 1) * (ONE / 128)).collect()
        };
        let digest = Sha256::digest(
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .into();
        sources.insert(
            key.clone(),
            SourceTensor {
                shape,
                lower: *values.iter().min().unwrap(),
                upper: *values.iter().max().unwrap(),
                public_q18: key.contains("norm").then(|| values.clone()),
                digest,
                row_l1: None,
                row_squared_norm: None,
                row_l1_squared_sum: None,
            },
        );
        clear.insert(key, values);
    }
    clear.insert("input.tokens".into(), vec![0, 1]);
    let policy = ExecutionPolicy {
        range_policy: if mode == "static" {
            RangePolicy::Static
        } else {
            RangePolicy::Checked
        },
        lookup: if tiled_lookup {
            LookupPolicy::Tiled
        } else {
            LookupPolicy::PerToken
        },
        selection: if tiled_selection {
            super::super::policy::SelectionPolicy::Tiled
        } else {
            super::super::policy::SelectionPolicy::Expanded
        },
        ..ExecutionPolicy::default()
    };
    let mut norms = SourceNorms::new();
    if certified {
        let plan = super::super::program::lower_with_weights(&raw, 2, 2, WeightMode::Public)?;
        for (key, (rows, columns, block_rows)) in
            super::super::source_norms::requests(&plan.prefill).map_err(|e| e.to_string())?
        {
            let mut scan = super::super::source_norms::BlockScan::new(rows, columns, block_rows)
                .map_err(|e| e.to_string())?;
            for (offset, values) in clear[&key].chunks(17).enumerate() {
                scan.append(offset * 17, values)
                    .map_err(|e| e.to_string())?;
            }
            norms.insert(key, scan.finish().map_err(|e| e.to_string())?);
        }
    }
    let mut program =
        Program::compile_with_norms(&raw, 2, 2, &sources, WeightMode::Public, policy, &norms)?;
    if certified {
        assert!(!norms.is_empty());
        assert!(program
            .nodes
            .iter()
            .any(|node| matches!(node.code, Code::Product { norm: Some(_), .. })));
    }
    let manifest = if certified {
        program.bind([11; 32], [12; 32]);
        Some(Manifest {
            schema: "pllm.replicated-decoder.manifest.v1".into(),
            weights: WeightMode::Public,
            execution: policy,
            config: raw,
            rows: 2,
            outputs: 2,
            source: [11; 32],
            composition: [12; 32],
            session: [42; 32],
            tensors: sources,
            norms,
            program: program.digest(),
        })
    } else {
        None
    };
    Ok((Arc::new(program), clear, config, manifest))
}

#[test]
fn native_certified_attention_complete_tensor_oracle() {
    for family in ["qwen2", "qwen3", "llama"] {
        for mode in ["checked_valid", "checked_invalid"] {
            let _session = Session::start();
            // Distinct two-layer GQA weights; 18-wide heads and 131-wide MLP
            // include partial guard/activation tiles and every causal key.
            let fixture = generated_with_norms(family, mode, true, true, 131, true).unwrap();
            audit_fixture(fixture, mode, true, true);
        }
    }
}

fn audit_case(
    family: &str,
    mode: &str,
    tiled_lookup: bool,
    tiled_selection: bool,
    intermediate: usize,
) {
    let _session = Session::start();
    let fixture = generated(family, mode, tiled_lookup, tiled_selection, intermediate).unwrap();
    audit_fixture(fixture, mode, tiled_lookup, tiled_selection);
}

/// Focused compiler guard/energy regression. Reuses the ordinary native workers,
/// bounded checkpoint capture, actual seed/frame evidence and independent Python
/// RMSNorm/AES oracle. All-party evidence stays in the anonymous test-only pipe.
pub(in crate::replicated_decoder) fn audit_checked_row_energy(
    program: Arc<Program>,
    values: Vec<i64>,
    cap: i64,
    rms_node: usize,
    checkpoints: &[(&str, Vec<(usize, usize)>)],
    rejected: bool,
) -> Vec<[u64; 7]> {
    let _session = Session::start();
    for (name, words) in checkpoints {
        diagnostic(0, name, &[words.len()], words.clone());
    }
    let clear = BTreeMap::from([("x".to_owned(), values.clone())]);
    let (mut workers, keys, public_seed) = make_workers(program, &clear, None);
    let mut queue = Vec::new();
    let mut records = Vec::new();
    let mut record_words = 0;
    let mut events = Vec::new();
    for transition in 0..1_000_000 {
        if workers.iter().all(Worker::complete) {
            break;
        }
        if queue.is_empty() {
            for worker in &mut workers {
                if worker.complete() {
                    continue;
                }
                for (to, frame) in worker.advance().unwrap() {
                    if let Some(event) = truncation(worker, &frame) {
                        events.push(event);
                    }
                    queue.push((worker.party(), to, frame));
                }
            }
        } else {
            for (from, to, frame) in std::mem::take(&mut queue) {
                let worker = &mut workers[to as usize];
                for (next, frame) in worker.receive(frame.as_bytes(), from).unwrap() {
                    if let Some(event) = truncation(worker, &frame) {
                        events.push(event);
                    }
                    queue.push((to, next, frame));
                }
            }
        }
        assert!(queue.len() <= 2048 && events.len() <= CAPTURE_WORDS);
        drain_into(&mut records, &mut record_words);
        assert!(transition < 999_999, "bounded checked-row transitions");
    }
    assert!(queue.is_empty() && workers.iter().all(Worker::complete));
    drain_into(&mut records, &mut record_words);
    assert_eq!(records.len(), checkpoints.len());
    CAPTURE.with(|slot| {
        let state = slot.borrow();
        assert!(state.ready.is_empty() && state.words == 0);
    });
    // Guards, sticky products and argmax must add no truncation sites.
    assert!(events.iter().all(|e| e["node"] == rms_node));
    // Compare every event in observed order, including multiplicity and public
    // PRSS cursor endpoints. Fresh setup changes bindings and randomized pivot
    // role/routes, not these public schedule coordinates. Route completeness is
    // independently checked per invocation by native_mask_tapes below.
    let signature: Vec<_> = events
        .iter()
        .map(|e| {
            [
                e["node"].as_u64().unwrap(),
                e["tensor_step"].as_u64().unwrap(),
                e["loop_offset"].as_u64().unwrap(),
                e["invocation"].as_u64().unwrap(),
                e["lanes"].as_u64().unwrap(),
                e["public_end"].as_u64().unwrap(),
                e["pair_end"].as_u64().unwrap(),
            ]
        })
        .collect();
    assert_eq!(signature.len(), 40); // four directed frames per truncation
    assert_eq!(
        signature
            .iter()
            .map(|e| e[1])
            .collect::<BTreeSet<_>>()
            .len(),
        10
    );
    let final_components = workers.map(|worker| worker.finish().unwrap());
    let payload = zeroize::Zeroizing::new(
        serde_json::to_string(&serde_json::json!({
            "input": values, "cap": cap, "rejected": rejected, "records": records,
            "events": events, "pair_keys": keys.iter().map(|key| hex(key)).collect::<Vec<_>>(),
            "public_seed": hex(&public_seed.to_le_bytes()),
            "final_components": final_components.each_ref().map(|v| v.as_slice()),
        }))
        .unwrap(),
    );
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let script = r#"
import json, sys
sys.path.insert(0, sys.argv[1])
import replicated_decoder_oracle as o
b = json.load(sys.stdin)
x = o.array([b['input']])
c = b['cap']
safe = o.np.where((x >= -c) & (x <= c), x, 0)
valid = bool(o.np.all((x >= -c) & (x <= c)))
assert valid == (not b['rejected'])
events = b['events']
bindings = list(dict.fromkeys(e['binding'] for e in events))
physical = o.native_mask_tapes(events, [bytes.fromhex(k) for k in b['pair_keys']],
    int.from_bytes(bytes.fromhex(b['public_seed']), 'little'), dict.fromkeys(bindings, 18))
suffixes = ['squares', 'mean'] + ['rsqrt/' + s for s in
    ['u', 'u2', 'u3', 'u4', 'polynomial', 'out']] + ['normalize', 'gamma']
assert len(bindings) == len(suffixes) == 10
layout = {}
for binding, suffix in zip(bindings, suffixes):
    lanes = len(physical[binding]['masks'])
    vector = suffix in ('squares', 'normalize', 'gamma')
    assert lanes == (x.shape[-1] if vector else 1)
    layout['norm/' + suffix] = {'shift': 18, 'shape': [1, lanes] if vector else [1],
        'lanes': [[binding, i] for i in range(lanes)]}
tape = o.adapt_tapes(physical, layout)
rms = o.rmsnorm(safe, [o.Q] * x.shape[-1], tape, 'norm')
tape.finish()
later = o.np.where((safe >= -(c//2)) & (safe <= c//2), safe, 0)
sticky = valid and bool(o.np.all((safe >= -(c//2)) & (safe <= c//2)))
records = {r['id']: r['values'] for r in b['records']}
assert records['safe'] == safe.reshape(-1).tolist()
assert records['rms'] == rms.reshape(-1).tolist()
assert records['later_safe'] == later.reshape(-1).tolist()
assert records['first_valid'] == [int(valid)]
assert records['sticky'] == [int(sticky)]
token = max(range(x.shape[-1]), key=lambda i: int(rms[0, i]))
assert token != 0
assert records['selected'] == [token]
final = [sum(p[i] for p in b['final_components']) % (1 << 64) for i in range(2)]
assert final == [token if sticky else 0, int(sticky)]
"#;
    use std::io::Write;
    let mut child = std::process::Command::new(root.join(".venv/bin/python"))
        .arg("-c")
        .arg(script)
        .arg(root.join("tests"))
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("oracle Python environment");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(payload.as_bytes())
        .unwrap();
    let result = child.wait_with_output().unwrap();
    // Do not print captured operands, masks or Python assertion source.
    assert!(
        result.status.success(),
        "independent checked-row stochastic audit failed"
    );
    signature
}

fn audit_fixture(fixture: Fixture, mode: &str, tiled_lookup: bool, tiled_selection: bool) {
    let (program, clear, config, manifest) = fixture;
    CAPTURE.with(|slot| {
        assert!(
            !slot.borrow().boundaries.is_empty(),
            "semantic compile hook missing"
        )
    });
    let (mut workers, keys, public_seed) = make_workers(program, &clear, manifest.as_ref());
    let mut queue: Vec<(Party, Party, PeerBytes)> = Vec::new();
    let mut records = Vec::new();
    let mut record_words = 0;
    let mut events = Vec::new();
    for transition in 0..1_000_000 {
        if workers.iter().all(Worker::complete) {
            break;
        }
        let outgoing = if queue.is_empty() {
            let mut outgoing = Vec::new();
            for worker in &mut workers {
                if worker.complete() {
                    continue;
                }
                let frames = worker.advance().unwrap();
                for (to, frame) in frames {
                    if let Some(event) = truncation(worker, &frame) {
                        events.push(event);
                    }
                    outgoing.push((worker.party(), to, frame));
                }
            }
            outgoing
        } else {
            let mut outgoing = Vec::new();
            for (from, to, frame) in std::mem::take(&mut queue) {
                let worker = &mut workers[to as usize];
                for (next, frame) in worker.receive(frame.as_bytes(), from).unwrap() {
                    if let Some(event) = truncation(worker, &frame) {
                        events.push(event);
                    }
                    outgoing.push((to, next, frame));
                }
            }
            outgoing
        };
        queue = outgoing;
        assert!(queue.len() <= 2048, "bounded test peer queue");
        assert!(
            events.len() <= CAPTURE_WORDS,
            "bounded test truncation evidence"
        );
        drain_into(&mut records, &mut record_words);
        assert!(transition < 999_999, "bounded native transition limit");
    }
    assert!(queue.is_empty() && workers.iter().all(Worker::complete));
    drain_into(&mut records, &mut record_words);
    CAPTURE.with(|slot| {
        let state = slot.borrow();
        assert_eq!(records.len(), state.boundaries.len());
        assert!(state.ready.is_empty() && state.words == 0);
    });
    assert!(!events.is_empty());
    let final_components = workers.map(|worker| worker.finish().unwrap());
    for selection in 0..2 {
        let embedding = records
            .iter()
            .find(|r| r["selection"] == selection && r["id"] == "token_lookup")
            .unwrap();
        let native = embedding["values"][0].as_i64().unwrap();
        assert_eq!(native, clear["model.embed_tokens.weight"][0]);
        let corrupted = native.wrapping_add(1);
        assert_ne!(corrupted, clear["model.embed_tokens.weight"][0]);
    }
    // Private evidence crosses ONLY this unit-test subprocess's anonymous
    // stdin pipe. No files, test output, SDK methods, or benchmark records.
    let shapes = Program::source_shapes_with_weights(
        &serde_json::to_vec(&config).unwrap(),
        2,
        2,
        WeightMode::Public,
    )
    .unwrap();
    let sources: BTreeMap<_, _> = shapes
        .into_iter()
        .map(|(key, shape)| {
            let value = serde_json::json!({"shape": shape, "values": clear[&key]});
            (key, value)
        })
        .collect();
    let spans = CAPTURE.with(|slot| slot.borrow().spans.clone());
    let guards = CAPTURE.with(|slot| slot.borrow().guards.clone());
    assert!(!spans.is_empty(), "semantic span hook missing");
    let bundle = serde_json::json!({
        "config": config, "sources": sources, "input_ids": [0,1], "outputs": 2,
        "pair_keys_hex": keys.iter().map(|key| hex(key)).collect::<Vec<_>>(),
        "public_seed_le_hex": hex(&public_seed.to_le_bytes()),
        "checkpoints": records, "events": events, "spans": spans,
        "final_components": final_components.each_ref().map(|v| v.as_slice()),
        "checked_profile": mode != "static", "expected_accepted": mode != "checked_invalid",
        "guard_schedule": guards,
        "assert_distinct_gqa": config["num_hidden_layers"].as_u64().unwrap() >= 2,
        "fixture_mode": mode, "lookup_tiled": tiled_lookup, "selection_tiled": tiled_selection,
    });
    let payload = zeroize::Zeroizing::new(serde_json::to_string(&bundle).unwrap());
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let script = root.join("tests/replicated_decoder_oracle.py");
    let python = root.join(".venv/bin/python");
    use std::io::Write;
    let mut child = std::process::Command::new(python)
        .arg(script)
        .arg("--audit-native-capture")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("oracle Python environment");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(payload.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "native independent audit failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let receipt: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("public oracle case receipt");
    assert_eq!(receipt["mode"], mode);
    assert_eq!(receipt["lookup_tiled"], tiled_lookup);
    assert_eq!(receipt["selection_tiled"], tiled_selection);
    eprintln!("ORACLE_CASE {}", receipt);
}

#[test]
fn native_generated_capture_has_complete_checkpoints_and_actual_truncation_events() {
    let mut cases = 0;
    for family in ["qwen2", "qwen3", "llama"] {
        for mode in ["static", "checked_valid", "checked_invalid"] {
            for tiled_lookup in [false, true] {
                for tiled_selection in [false, true] {
                    audit_case(family, mode, tiled_lookup, tiled_selection, 2);
                    cases += 1;
                }
            }
        }
    }
    audit_case("llama", "static", true, true, 130);
    assert_eq!(cases + 1, 37);
}

#[test]
fn native_multilayer_gqa_guard_tiles_and_tensor_tails() {
    for family in ["qwen2", "qwen3", "llama"] {
        for mode in ["checked_valid", "checked_invalid"] {
            for lookup_tiled in [false, true] {
                for selection_tiled in [false, true] {
                    audit_case(family, mode, lookup_tiled, selection_tiled, 131);
                }
            }
        }
    }
    for mode in ["checked_valid", "checked_invalid"] {
        // 2*128 and 3*128 lanes: complete guard tiles, larger heads.
        audit_case("qwen3", mode, true, true, 133);
    }
}

#[test]
fn native_full_vocabulary_narrow_fixture_admission_and_audit() {
    let _session = Session::start();
    match generated("llama", "static", true, true, 132) {
        Ok(fixture) => {
            eprintln!(
                "generated full-vocabulary fixture admitted: ordinals={}, public compiler bytes={}",
                fixture.0.operation_limit(),
                fixture.0.payload_bytes()
            );
            audit_fixture(fixture, "static", true, true);
        }
        Err(reason) => {
            // Admission failure is a retained result, not a smaller vocabulary
            // rerun or a capacity bypass. Nothing executes after this rejection.
            eprintln!("generated full-vocabulary fixture REJECTED before workers: {reason}");
            assert!(
                reason.contains("bound") || reason.contains("limit") || reason.contains("capacity")
            );
        }
    }
}
