//! Opt-in finite complete-gate producer. Only this test harness holds both
//! evaluators and reconstructs final test outputs (including fixture diagnostics).
//! No expected fixture result enters lowering, issuance or runtime execution.
use super::{gates, norm, source_tables, Event, Plan};
use crate::fusefss_backend::{gate_conformance, Budget, MAX_ALLOCATION_BYTES};
use crate::shared_rescale_reference::Context;
use serde_json::{json, Value as J};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, io::Write, path::Path};
use zeroize::Zeroizing;

const MASK: u64 = (1u64 << 50) - 1;
const FIXTURE_HASH: &str = "ae1d8ba29b2563761ce1362688c4ff893812294d76066b0f398fb3388ace2a87";
const FIXTURE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../paper/reproduction-contracts/fusefss/gate-set/fixtures.json"
));
const SOURCES: &[(&str, &[u8])] = &[
    (
        "crates/pllm-garble/src/fixed_activation_tables.rs",
        include_bytes!("../../fixed_activation_tables.rs"),
    ),
    (
        "crates/pllm-garble/src/fixed_activation_table_values.rs",
        include_bytes!("../../fixed_activation_table_values.rs"),
    ),
    (
        "crates/pllm-garble/src/fusefss/typed_comparison.rs",
        include_bytes!("../../fusefss/typed_comparison.rs"),
    ),
    (
        "crates/pllm-garble/src/fusefss_backend/composition.rs",
        include_bytes!("../composition.rs"),
    ),
    (
        "crates/pllm-garble/src/fusefss_backend/composition/runtime.rs",
        include_bytes!("runtime.rs"),
    ),
    (
        "crates/pllm-garble/src/fusefss_backend/composition/gates.rs",
        include_bytes!("gates.rs"),
    ),
    (
        "crates/pllm-garble/src/fusefss_backend/composition/norm.rs",
        include_bytes!("norm.rs"),
    ),
    (
        "crates/pllm-garble/src/fusefss_backend/composition/source_tables.rs",
        include_bytes!("source_tables.rs"),
    ),
    (
        "crates/pllm-garble/src/fusefss_backend/composition/source_table_values.rs",
        include_bytes!("source_table_values.rs"),
    ),
    (
        "crates/pllm-garble/src/fusefss_backend/composition/conformance.rs",
        include_bytes!("conformance.rs"),
    ),
    (
        "paper/reproduction-contracts/fusefss/gate-set/generate-native-tables.py",
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../paper/reproduction-contracts/fusefss/gate-set/generate-native-tables.py"
        )),
    ),
    (
        "paper/reproduction-contracts/fusefss/gate-set/source-tables-supplement.json",
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../paper/reproduction-contracts/fusefss/gate-set/source-tables-supplement.json"
        )),
    ),
];

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::new(), |mut s, v| {
        write!(s, "{v:02x}").unwrap();
        s
    })
}
fn hash(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}
fn signed(word: u64) -> i64 {
    if word & (1u64 << 49) == 0 {
        word as i64
    } else {
        word as i64 - (1i64 << 50)
    }
}
fn manifest() -> BTreeMap<String, String> {
    let mut values = gate_conformance::source_manifest();
    for (name, bytes) in SOURCES {
        values.insert((*name).into(), hash(bytes));
    }
    values.insert(
        "paper/reproduction-contracts/fusefss/gate-set/fixtures.json".into(),
        hash(FIXTURE),
    );
    values
}
fn fingerprints() -> BTreeMap<String, String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let values = manifest();
    for (name, expected) in &values {
        assert_eq!(
            hash(&fs::read(root.join(name)).unwrap()),
            *expected,
            "compiled input changed: {name}"
        );
    }
    values
}
fn implementation() -> String {
    format!(
        "pllm-native-composed-f2-test-sha256:{}",
        hash(&serde_json::to_vec(&manifest()).unwrap())
    )
}
fn context(label: &str, plan: &Plan<'_>) -> Context {
    let digest: [u8; 32] = Sha256::digest(label.as_bytes()).into();
    Context {
        plan: plan.digest(),
        session: [0x49; 32],
        operation: digest,
        first_tensor_index: 0,
    }
}
fn shares(value: u64, bits: u8) -> [u64; 2] {
    let mut bytes = Zeroizing::new([0u8; 8]);
    getrandom::fill(&mut *bytes).unwrap();
    let mask = (1u64 << bits) - 1;
    let left = u64::from_le_bytes(*bytes) & mask;
    [left, value.wrapping_sub(left) & mask]
}

fn run_plan(plan: &Plan<'_>, words: &[u64], label: &str, budget: &Budget) -> (Vec<u64>, J) {
    assert_eq!(plan.input_bits().len(), words.len());
    let mut left = Zeroizing::new(Vec::with_capacity(words.len()));
    let mut right = Zeroizing::new(Vec::with_capacity(words.len()));
    for (&word, &bits) in words.iter().zip(plan.input_bits()) {
        let pair = Zeroizing::new(shares(word, bits));
        left.push(pair[0]);
        right.push(pair[1]);
    }
    let ctx = context(label, plan);
    let mut a = plan.runtime(0, ctx, &left).unwrap();
    let mut b = plan.runtime(1, ctx, &right).unwrap();
    let (mut ae, mut be) = (a.poll().unwrap(), b.poll().unwrap());
    let mut issuances = 0usize;
    let mut frames = 0usize;
    let mut frame_bytes = 0usize;
    let mut peak = budget.used_bytes();
    let outputs = loop {
        match (ae, be) {
            (Event::NeedMaterial { pc: ap }, Event::NeedMaterial { pc: bp }) => {
                assert_eq!(ap, bp);
                let [am, bm] = plan.issue_step(ap, ctx).unwrap();
                peak = peak.max(budget.used_bytes());
                issuances += 1;
                ae = a.feed(am).unwrap();
                be = b.feed(bm).unwrap();
            }
            (Event::Frame(af), Event::Frame(bf)) => {
                frames += 2;
                frame_bytes += af.as_bytes().len() + bf.as_bytes().len();
                ae = a.advance(bf.as_bytes()).unwrap();
                be = b.advance(af.as_bytes()).unwrap();
                peak = peak.max(budget.used_bytes());
                drop((af, bf));
            }
            (Event::Output(ao), Event::Output(bo)) => {
                assert!(a.is_consumed() && b.is_consumed());
                assert_eq!(ao.shares().len(), bo.shares().len());
                let outputs = ao
                    .shares()
                    .iter()
                    .zip(bo.shares())
                    .enumerate()
                    .map(|(i, (&x, &y))| {
                        let bits = plan.output_bits(i).unwrap();
                        if plan.output_is_xor(i).unwrap() {
                            x ^ y
                        } else {
                            x.wrapping_add(y) & ((1u64 << bits) - 1)
                        }
                    })
                    .collect::<Vec<_>>();
                drop((ao, bo));
                break outputs;
            }
            _ => panic!("asymmetric composed schedule"),
        }
        peak = peak.max(budget.used_bytes());
    };
    let r = plan.resources();
    assert_eq!(issuances, r.secret_steps);
    assert_eq!(frames, r.frames_total);
    assert_eq!(frame_bytes, r.frame_bytes_total);
    assert!(peak <= r.combined_reservation_bytes);
    drop((a, b));
    assert_eq!(budget.used_bytes(), r.plan_reservation_bytes);
    (
        outputs,
        json!({"plan_sha256":hex(&plan.digest()),"instructions":r.instructions,
        "registers":r.registers,"issuances":issuances,"lanes":issuances,
        "dcf_evaluations_per_party":r.dcf_evaluations_per_party,
        "dcf_keys_total":r.dcf_keys_total,"dcf_evaluations_total":r.dcf_evaluations_total,
        "triples_per_party":r.triples_per_party,"triples_total":r.triples_total,
        "private_material_bytes_total":r.private_material_bytes_total,
        "public_table_copy_bytes_total":r.public_table_copy_bytes_total,
        "cumulative_work":r.cumulative_work,
        "plan_reservation_bytes":r.plan_reservation_bytes,
        "public_table_reservation_bytes":r.public_table_reservation_bytes,
        "retained_frame_overlap_bytes":r.retained_frame_overlap_bytes,
        "combined_reservation_bytes":r.combined_reservation_bytes,
        "peak_observed_reserved_bytes":peak,"peer_frames_both_parties":frames,
        "peer_frame_bytes_both_parties":frame_bytes}),
    )
}
fn field<'a>(values: &'a [u64], layout: &gates::OutputLayout, name: &str) -> &'a [u64] {
    let range = &layout
        .intermediates
        .iter()
        .find(|(n, _)| *n == name)
        .unwrap()
        .1;
    &values[range.clone()]
}
fn integers(value: &J) -> Vec<i64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect()
}

fn run_case(case: &J) -> (J, J) {
    let gate = case["gate"].as_str().unwrap();
    let label = case["id"].as_str().unwrap();
    let args = case["args"].as_array().unwrap();
    if matches!(
        gate,
        "shift" | "b2a" | "compare" | "phi-select" | "external"
    ) {
        let (value, resources, detail) = gate_conformance::run_selected_case(case);
        return (
            value,
            json!({"producer":"shared-selected-primitive","resources":resources,"binding":detail}),
        );
    }
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let (value, resources) = match gate {
        "silu" | "nexp" | "reciprocal" => {
            let (plan, layout) = match gate {
                "silu" => gates::silu(
                    gates::InputRange {
                        min: -(1i64 << 37),
                        max: (1i64 << 37) - 1,
                    },
                    &budget,
                ),
                "nexp" => gates::nexp(gates::InputRange { min: 0, max: 65535 }, &budget),
                "reciprocal" => {
                    let k = args[1].as_u64().unwrap() as usize;
                    gates::reciprocal(
                        k,
                        gates::InputRange {
                            min: 64,
                            max: 64 * k as i64,
                        },
                        &budget,
                    )
                }
                _ => unreachable!(),
            }
            .unwrap();
            let word = (args[0].as_i64().unwrap() as u64) & MASK;
            let (values, resources) = run_plan(&plan, &[word], label, &budget);
            let output = values[layout.final_outputs.start];
            let result = if gate == "silu" {
                json!({"floor":signed(field(&values,&layout,"floor")[0]),
                    "index":field(&values,&layout,"index")[0],"output":output})
            } else {
                json!(output)
            };
            drop(plan);
            (result, resources)
        }
        "softmax" => {
            let row = integers(&args[0]);
            let active: Vec<bool> = if args[1].is_null() {
                vec![true; row.len()]
            } else {
                args[1]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() == 1)
                    .collect()
            };
            // Frozen, public bound covering every fixture row; no private-data fit.
            let ranges = vec![
                gates::InputRange {
                    min: -131072,
                    max: 131072
                };
                row.len()
            ];
            let (plan, layout) = gates::softmax(&ranges, &active, &budget).unwrap();
            let words: Vec<_> = row.iter().map(|&v| (v as u64) & MASK).collect();
            let (values, resources) = run_plan(&plan, &words, label, &budget);
            let output = json!({"exp":field(&values,&layout,"exp"),
                "denominator":field(&values,&layout,"denominator")[0],
                "reciprocal":field(&values,&layout,"reciprocal")[0],
                "output":&values[layout.final_outputs]});
            drop(plan);
            (output, resources)
        }
        "norm-index" => {
            let plan = norm::compile_index(source_tables::source_digest(), &budget).unwrap();
            let (values, resources) = run_plan(&plan, &[args[0].as_u64().unwrap()], label, &budget);
            assert_eq!(values.len(), 3);
            let output = json!({"exponent":values[0],"mantissa":values[1],"index":values[2]});
            drop(plan);
            (output, resources)
        }
        "norm" => {
            let row = integers(&args[0]);
            let gamma = if args[1].is_null() {
                vec![4096; row.len()]
            } else {
                integers(&args[1])
            };
            let beta = if args[2].is_null() {
                vec![0; row.len()]
            } else {
                integers(&args[2])
            };
            let spec = norm::Spec {
                kind: match args[3].as_str().unwrap() {
                    "layer" => norm::Kind::Layer,
                    "rms" => norm::Kind::Rms,
                    _ => panic!("unknown norm"),
                },
                input_min: -8192,
                input_max: 8192,
                gamma: &gamma,
                beta: &beta,
            };
            let plan =
                norm::compile_fixture(source_tables::source_digest(), &spec, &budget).unwrap();
            let words: Vec<_> = row.iter().map(|&v| (v as u64) & MASK).collect();
            let (values, resources) = run_plan(&plan, &words, label, &budget);
            assert_eq!(values.len(), 6 + row.len());
            let output = json!({"mean":signed(values[0]),"sum_squares_plus_one":values[1],
                "exponent":values[2],"mantissa":values[3],"index":values[4],"inverse":values[5],
                "output":values[6..].iter().map(|&v| signed(v)).collect::<Vec<_>>()});
            drop(plan);
            (output, resources)
        }
        _ => panic!("unknown complete gate"),
    };
    assert_eq!(budget.used_bytes(), 0, "unretired complete gate: {label}");
    (
        value,
        json!({"producer":"composed-party-local-native","resources":resources,"final_reserved_bytes":0}),
    )
}

fn write_line(file: &mut fs::File, value: &J) {
    let bytes = serde_json::to_vec(value).unwrap();
    assert!(bytes.len() < 256 << 10);
    file.write_all(&bytes).unwrap();
    file.write_all(b"\n").unwrap();
}

#[test]
#[ignore = "explicit release complete 275-case protected composition export"]
fn export_complete_gates() {
    assert!(!cfg!(debug_assertions));
    assert_eq!(
        std::env::var("PLLM_FUSEFSS_EXPORT_COMPOSITION").as_deref(),
        Ok("1")
    );
    assert_eq!(hash(FIXTURE), FIXTURE_HASH);
    let fixture: J = serde_json::from_slice(FIXTURE).unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 275);
    let before = fingerprints();
    let dir = std::env::var_os("PLLM_FUSEFSS_COMPOSITION_EXPORT_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/fusefss-composition-native-v1")
        });
    assert!(!dir.as_os_str().is_empty());
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("complete-gates.jsonl");
    let mut file = fs::File::create_new(&path).unwrap();
    write_line(
        &mut file,
        &json!({"schema":"pllm.fusefss.gate-set.v1/native-output",
        "fixture_sha256":FIXTURE_HASH,"execution":"protected-local-reconstruction","implementation":implementation()}),
    );
    let mut records = Vec::new();
    let mut coverage = BTreeMap::<String, usize>::new();
    for case in cases {
        let (output, record) = run_case(case);
        *coverage
            .entry(case["gate"].as_str().unwrap().into())
            .or_default() += 1;
        write_line(
            &mut file,
            &json!({"id":case["id"],"gate":case["gate"],"output":output}),
        );
        records.push(json!({"id":case["id"],"gate":case["gate"],"execution":record}));
    }
    file.sync_all().unwrap();
    drop(file);
    let after = fingerprints();
    assert_eq!(before, after);
    let receipt = json!({"schema":"pllm.fusefss.composition.native-export-receipt.v1",
        "coverage":"all-frozen-cases","case_count":275,"gate_counts":coverage,
        "fixture_sha256":FIXTURE_HASH,"source_lock_sha256":source_tables::SOURCE_LOCK_HEX,
        "implementation":implementation(),"source_sha256":manifest(),
        "source_input_fingerprints_before":before,"source_input_fingerprints_after":after,
        "source_inputs_unchanged":true,"output_sha256":hash(&fs::read(&path).unwrap()),"records":records,
        "final_reserved_bytes":0,"accounting":{
            "freshness":"production OS randomness per primitive; externally issued material; fresh input shares",
            "budget":"complete planned envelopes and sampled native reservations; observed samples can miss issuance scratch and are not a high-water mark, allocator RSS or process peak",
            "frames":"actual native peer frame lengths; no authenticated network transport",
            "excluded_test_owners":"fixture JSON, caller buffers, source snapshots, reconstruction, export buffers and filesystem I/O"},
        "limitations":["public input ranges are caller assumptions, not private membership proofs",
            "finite protected gate conformance, not complete decoder or SDK execution",
            "local trusted-dealer feed, not distributed preprocessing or independent security qualification"]});
    let mut report = fs::File::create_new(dir.join("complete-gates.receipt.json")).unwrap();
    serde_json::to_writer_pretty(&mut report, &receipt).unwrap();
    report.write_all(b"\n").unwrap();
}

#[test]
fn composed_producer_ignores_expected_fields_and_uses_final_diagnostic_shares() {
    let case = json!({"id":"silu-negative-floor","gate":"silu","args":[-1],
        "expected":{"floor":123,"index":456,"output":789}});
    let (output, record) = run_case(&case);
    assert_eq!(output, json!({"floor":-1,"index":1,"output":(1u64<<50)-31}));
    assert_eq!(record["final_reserved_bytes"], 0);
    assert!(record["resources"]["issuances"].as_u64().unwrap() > 0);
}
