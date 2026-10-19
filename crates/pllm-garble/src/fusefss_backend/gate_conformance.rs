//! Opt-in test exporters, not serving APIs, clear gate interpreters, or source/
//! model qualification. Expected fixture outputs are never execution inputs.
//! Fixture JSON, descriptor parsing, export buffers and filesystem I/O are test
//! harness owners outside the backend Budget; receipts say so explicitly.
use super::{conversions as cv, lookup, predicates, Budget, Output, MAX_ALLOCATION_BYTES};
use crate::{fusefss, shared_rescale_reference::Context};
use serde_json::{json, Value as J};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, io::Write, path::Path};
use zeroize::Zeroizing;

const FIXTURE: &[u8] =
    include_bytes!("../../../../paper/reproduction-contracts/fusefss/gate-set/fixtures.json");
const TABLE_FIXTURE: &[u8] =
    include_bytes!("../../../../paper/reproduction-contracts/fusefss/gate-set/table-fixtures.json");
const SOURCE_LOCK: &[u8] =
    include_bytes!("../../../../paper/reproduction-contracts/fusefss/gate-set/source-lock.json");
const FIXTURE_HASH: &str = "ae1d8ba29b2563761ce1362688c4ff893812294d76066b0f398fb3388ace2a87";
const TABLE_FIXTURE_HASH: &str = "161cb9531a63f5937f090494a5fc1a37a4d4ab8e34d22ed4f2a628f2081ae283";
const SOURCE_HASH: &str = "8080069e2edf58fb0b6b72a0a27bcbfa636db25018a0696681335a7dfe1ddcbd";
const SELECTED: [&str; 5] = ["shift", "b2a", "compare", "phi-select", "external"];
const TABLE_NAMES: [&str; 8] = [
    "silu",
    "exp0",
    "exp1",
    "reciprocal-k1",
    "reciprocal-k2",
    "reciprocal-k3",
    "rsqrt-k4",
    "rsqrt-k768",
];

// Compile-time content, not mtimes, caller-provided hashes or a git revision.
// Runtime fingerprints must still equal these exact compiled source snapshots.
const SOURCES: &[(&str, &[u8])] = &[
    (
        "crates/pllm-garble/src/fusefss_backend/gate_conformance_tests.rs",
        include_bytes!("gate_conformance_tests.rs"),
    ),
    (
        "crates/pllm-garble/src/fusefss_backend/gate_conformance.rs",
        include_bytes!("gate_conformance.rs"),
    ),
    (
        "crates/pllm-garble/src/fusefss_backend/mod.rs",
        include_bytes!("mod.rs"),
    ),
    (
        "crates/pllm-garble/src/fusefss_backend/predicates.rs",
        include_bytes!("predicates.rs"),
    ),
    (
        "crates/pllm-garble/src/fusefss_backend/conversions.rs",
        include_bytes!("conversions.rs"),
    ),
    (
        "crates/pllm-garble/src/fusefss_backend/lookup.rs",
        include_bytes!("lookup.rs"),
    ),
    (
        "crates/pllm-garble/src/fusefss/mod.rs",
        include_bytes!("../fusefss/mod.rs"),
    ),
    (
        "crates/pllm-garble/src/fusefss/compile.rs",
        include_bytes!("../fusefss/compile.rs"),
    ),
    (
        "crates/pllm-garble/src/fusefss/json.rs",
        include_bytes!("../fusefss/json.rs"),
    ),
    (
        "crates/pllm-garble/src/fusefss/fixture.rs",
        include_bytes!("../fusefss/fixture.rs"),
    ),
    (
        "crates/pllm-garble/src/fusefss/conformance.rs",
        include_bytes!("../fusefss/conformance.rs"),
    ),
    (
        "crates/pllm-garble/src/compact_dcf.rs",
        include_bytes!("../compact_dcf.rs"),
    ),
    (
        "crates/pllm-garble/src/interval_fss.rs",
        include_bytes!("../interval_fss.rs"),
    ),
    (
        "crates/pllm-garble/src/shared_affine_lookup_reference.rs",
        include_bytes!("../shared_affine_lookup_reference.rs"),
    ),
    (
        "crates/pllm-garble/src/shared_affine_lookup_reference/ring.rs",
        include_bytes!("../shared_affine_lookup_reference/ring.rs"),
    ),
    (
        "crates/pllm-garble/src/shared_wide_product.rs",
        include_bytes!("../shared_wide_product.rs"),
    ),
    (
        "crates/pllm-garble/src/shared_wide_product/ring.rs",
        include_bytes!("../shared_wide_product/ring.rs"),
    ),
    (
        "crates/pllm-garble/src/shared_circuit.rs",
        include_bytes!("../shared_circuit.rs"),
    ),
    (
        "crates/pllm-garble/src/shared_rescale_reference.rs",
        include_bytes!("../shared_rescale_reference.rs"),
    ),
    ("crates/pllm-garble/src/lib.rs", include_bytes!("../lib.rs")),
    (
        "crates/pllm-garble/Cargo.toml",
        include_bytes!("../../Cargo.toml"),
    ),
    ("Cargo.toml", include_bytes!("../../../../Cargo.toml")),
    ("Cargo.lock", include_bytes!("../../../../Cargo.lock")),
];

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::new(), |mut s, b| {
        write!(s, "{b:02x}").unwrap();
        s
    })
}
fn hash(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}
fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}
fn source_digest() -> [u8; 32] {
    digest(SOURCE_LOCK)
}
fn mask(bits: u8) -> u64 {
    (1u64 << bits) - 1
}
fn share(word: u64, bits: u8) -> [u64; 2] {
    assert!(word <= mask(bits));
    let left = super::random().unwrap() & mask(bits);
    [left, word.wrapping_sub(left) & mask(bits)]
}
fn bool_share(bit: u64) -> [u64; 2] {
    assert!(bit <= 1);
    let left = super::random().unwrap() & 1;
    [left, left ^ bit]
}
fn context(label: &str, offset: usize) -> Context {
    Context {
        plan: digest(FIXTURE),
        session: digest(b"pllm.fusefss.f2.native-local-conformance.v1"),
        operation: digest(label.as_bytes()),
        first_tensor_index: offset as u64,
    }
}

#[derive(Default, Clone)]
struct Stats {
    issuances: u64,
    lanes: u64,
    dcf_keys: u64,
    dcf_evals: u64,
    triples: u64,
    material: u64,
    public_table_copies: u64,
    frames: u64,
    frame_bytes: u64,
    peak: usize,
    observed: usize,
}
impl Stats {
    fn admission(&mut self, budget: &Budget, bytes: usize) {
        let peak = budget.used_bytes().checked_add(bytes).unwrap();
        assert!(peak <= MAX_ALLOCATION_BYTES);
        self.peak = self.peak.max(peak);
    }
    fn observe(&mut self, budget: &Budget) {
        self.observed = self.observed.max(budget.used_bytes());
        assert!(self.observed <= self.peak);
    }
    fn add(&mut self, other: &Self) {
        self.issuances += other.issuances;
        self.lanes += other.lanes;
        self.dcf_keys += other.dcf_keys;
        self.dcf_evals += other.dcf_evals;
        self.triples += other.triples;
        self.material += other.material;
        self.public_table_copies += other.public_table_copies;
        self.frames += other.frames;
        self.frame_bytes += other.frame_bytes;
        self.peak = self.peak.max(other.peak);
        self.observed = self.observed.max(other.observed);
    }
    fn json(&self) -> J {
        json!({"fresh_issuance_count":self.issuances,"fresh_lane_issuance_count":self.lanes,
            "dcf_keys_per_party":self.dcf_keys,"dcf_evaluations_per_party":self.dcf_evals,
            "triples_per_party":self.triples,"cumulative_private_material_bytes_both_parties":self.material,
            "cumulative_public_table_copy_bytes_both_parties":self.public_table_copies,
            "peer_frames_both_parties":self.frames,"peer_frame_bytes_both_parties":self.frame_bytes,
            "peak_logical_budget_bytes":self.peak,"max_observed_live_budget_bytes":self.observed,
            "final_reservation_bytes":0})
    }
}

fn spec(input_bits: u8, output_bits: u8, operation: cv::Operation) -> cv::Spec {
    cv::Spec {
        input_bits,
        output_bits,
        lanes: 1,
        operation,
        source_digest: source_digest(),
        profile_digest: digest(FIXTURE),
    }
}

// Return shares without reconstructing: Phi must receive B2A's actual outputs.
fn convert<'a>(
    s: cv::Spec,
    inputs: [u64; 2],
    label: &str,
    budget: &'a Budget,
    stats: &mut Stats,
) -> [Output<'a>; 2] {
    let r = cv::resources(s).unwrap();
    stats.admission(budget, r.combined_reservation_bytes);
    let [mut a, mut b] = cv::issue(s, context(label, 0), budget).unwrap();
    stats.observe(budget);
    let af = a.start(&[inputs[0]]).unwrap();
    let bf = b.start(&[inputs[1]]).unwrap();
    let actual = af.as_bytes().len() + bf.as_bytes().len();
    assert_eq!(actual, 2 * r.frame_bytes_per_party);
    let ao = a.finish(bf.as_bytes()).unwrap();
    let bo = b.finish(af.as_bytes()).unwrap();
    assert!(a.is_consumed() && b.is_consumed());
    stats.observe(budget);
    stats.issuances += 1;
    stats.lanes += 1;
    stats.dcf_keys += r.dcf_keys_per_party as u64;
    stats.dcf_evals += r.dcf_evaluations_per_party as u64;
    stats.triples += r.triples as u64;
    stats.material += r.private_material_bytes_total as u64;
    stats.frames += 2;
    stats.frame_bytes += actual as u64;
    [ao, bo]
}

fn final_word(outputs: &[Output<'_>; 2], bits: u8) -> u64 {
    outputs[0].shares()[0].wrapping_add(outputs[1].shares()[0]) & mask(bits)
}

fn comparison_descriptor(args: &[J]) -> fusefss::Descriptor {
    let bits = args[1].as_u64().unwrap() as u8;
    let expr = match args[3].as_str().unwrap() {
        "unsigned" => json!(["lt", args[2]]),
        "signed" => json!(["slt", args[2]]),
        "low" => json!(["low_lt", args[4], args[2]]),
        _ => panic!("unsupported frozen comparison"),
    };
    let public = json!({"schema":"pllm.fusefss.scalar-descriptor.v1",
        "profile":fusefss::SYNTHETIC_PROFILE,"gate":"f2-comparison-conformance",
        "source_lock_sha256":SOURCE_HASH,"input":{"domain":"A","bits":bits,"fractional":0,"signed":true},
        "output":{"arithmetic":"same-as-input","boolean_arity":1},
        "external_encoding":"nearest-ties-even-input-and-weight","internal_rounding":"signed-floor-only",
        "partition_order":"unsigned-canonical-half-open","degree":1,
        "coefficient_order":["constant","linear"],"coefficient_fractional":[0,0],
        "selector_semantics":"descriptor-defined","source_schedule":{},"phi":"identity",
        "boundaries":[0,1u64<<bits],"pieces":[{"coefficients":[0,0],"booleans":[expr]}]});
    let bytes = serde_json::to_vec(&public).unwrap();
    assert!(bytes.len() <= 4096); // tiny, bounded test public JSON; not serving admission
    fusefss::Descriptor::from_canonical_bytes(&bytes).unwrap()
}

fn compare(args: &[J], label: &str, budget: &Budget, stats: &mut Stats) -> (u64, J) {
    let d = comparison_descriptor(args);
    let r = predicates::resources(&d).unwrap();
    stats.admission(budget, r.combined_reservation_bytes);
    // Count public layout only, before any protected issuance, under a separate
    // compiler scratch reservation. Never evaluate the public zero-mask plan.
    let (key_bytes, bit_count) = {
        let _scratch = budget.reserve(r.compiler_overlap_bytes).unwrap();
        let plan = d.compile(0).unwrap();
        plan.with_dealer_handoff(|h| {
            let key_bytes = h
                .predicate_queries
                .iter()
                .map(|q| {
                    crate::compact_dcf::RingComparison::resources(q.bits(), 1)
                        .unwrap()
                        .encoded_key_bytes
                })
                .sum::<usize>();
            let bit_count = h
                .boolean_program
                .iter()
                .filter(|i| i.opcode() == fusefss::Opcode::Bit)
                .count();
            Ok((key_bytes, bit_count))
        })
        .unwrap()
    };
    let mut ctx = context(label, 0);
    ctx.plan = d.digest();
    let [mut a, mut b] = predicates::issue(&d, source_digest(), ctx, budget).unwrap();
    stats.observe(budget);
    let inputs = Zeroizing::new(share(args[0].as_u64().unwrap(), d.word().bits));
    let mut af = a.start(inputs[0]).unwrap();
    let mut bf = b.start(inputs[1]).unwrap();
    let mut frames = 2u64;
    let mut frame_bytes = (af.as_bytes().len() + bf.as_bytes().len()) as u64;
    let output = loop {
        let sa = a.advance(bf.as_bytes()).unwrap();
        let sb = b.advance(af.as_bytes()).unwrap();
        match (sa, sb) {
            (predicates::Step::Frame(next_a), predicates::Step::Frame(next_b)) => {
                frames += 2;
                frame_bytes += (next_a.as_bytes().len() + next_b.as_bytes().len()) as u64;
                af = next_a;
                bf = next_b;
            }
            (predicates::Step::Output(ao), predicates::Step::Output(bo)) => {
                assert!(a.is_consumed() && b.is_consumed());
                // The sole reconstruction is the testcase's final output.
                break ao.shares()[0] ^ bo.shares()[0];
            }
            _ => panic!("peer shape mismatch"),
        }
    };
    assert_eq!(frames, 2 * r.frames_per_party as u64);
    assert_eq!(frame_bytes, r.total_peer_frame_bytes as u64);
    stats.issuances += 1;
    stats.lanes += 1;
    stats.dcf_keys += r.dcf_keys_per_party as u64;
    stats.dcf_evals += r.dcf_evaluations_per_party as u64;
    stats.triples += r.triples_per_party as u64;
    // Scalar one-bit DCF payload encoding: 24 + 26*input_bits; Bit
    // operands are native u64 XOR shares; input-mask share is one u64.
    stats.material += 2 * (key_bytes + 8 + 8 * bit_count + 24 * r.triples_per_party) as u64;
    stats.frames += frames;
    stats.frame_bytes += frame_bytes;
    (
        output,
        json!({"descriptor_sha256":hex(&d.digest()),"bit_operands_per_party":bit_count,
        "descriptor_parser_and_borrowed_owner":"outside backend Budget; <=4096-byte test input"}),
    )
}

fn phi(args: &[J], label: &str, budget: &Budget, stats: &mut Stats) -> u64 {
    let bits = 50;
    let bs = Zeroizing::new(bool_share(args[0].as_u64().unwrap()));
    let bit = convert(
        spec(1, bits, cv::Operation::BooleanToArithmetic),
        *bs,
        &format!("{label}/b2a"),
        budget,
        stats,
    );
    let s = spec(
        bits,
        bits,
        cv::Operation::PhiSelect {
            add: 0,
            bit_contract: digest(format!("{label}/fresh-b2a-output-bit").as_bytes()),
        },
    );
    let r = cv::resources(s).unwrap();
    stats.admission(budget, r.combined_reservation_bytes);
    let [mut a, mut b] = cv::issue(s, context(&format!("{label}/phi"), 0), budget).unwrap();
    stats.observe(budget);
    let yes = Zeroizing::new(share(args[1].as_u64().unwrap(), bits));
    let no = Zeroizing::new(share(args[2].as_u64().unwrap(), bits));
    let af = a.start_phi(bit[0].shares(), &[yes[0]], &[no[0]]).unwrap();
    let bf = b.start_phi(bit[1].shares(), &[yes[1]], &[no[1]]).unwrap();
    let actual = af.as_bytes().len() + bf.as_bytes().len();
    assert_eq!(actual, 2 * r.frame_bytes_per_party);
    let out = [
        a.finish(bf.as_bytes()).unwrap(),
        b.finish(af.as_bytes()).unwrap(),
    ];
    assert!(a.is_consumed() && b.is_consumed());
    stats.observe(budget);
    stats.issuances += 1;
    stats.lanes += 1;
    stats.triples += r.triples as u64;
    stats.material += r.private_material_bytes_total as u64;
    stats.frames += 2;
    stats.frame_bytes += actual as u64;
    final_word(&out, bits)
}

fn validate_case(case: &J) -> Result<(), &'static str> {
    let id = case["id"].as_str().ok_or("case id")?;
    if id.is_empty() || id.len() > 96 || !id.is_ascii() {
        return Err("case id bound");
    }
    let gate = case["gate"].as_str().ok_or("gate label")?;
    let a = case["args"].as_array().ok_or("case args")?;
    let width = |i: usize| -> Result<u8, &'static str> {
        let v = a.get(i).and_then(J::as_u64).ok_or("width integer")?;
        if !(1..=50).contains(&v) {
            return Err("width range");
        }
        Ok(v as u8)
    };
    let word = |i: usize, bits: u8| -> Result<u64, &'static str> {
        a.get(i)
            .and_then(J::as_u64)
            .filter(|v| *v <= mask(bits))
            .ok_or("word range/type")
    };
    match gate {
        "shift" => {
            if a.len() != 4 {
                return Err("shift arity");
            }
            let bits = width(1)?;
            word(0, bits)?;
            if a[2].as_u64().filter(|v| *v < u64::from(bits)).is_none()
                || !matches!(a[3].as_str(), Some("TR" | "ARS"))
            {
                return Err("shift operation");
            }
        }
        "b2a" => {
            if a.len() != 3 {
                return Err("b2a arity");
            }
            word(0, 1)?;
            word(1, 1)?;
            width(2)?;
        }
        "compare" => {
            if a.len() != 5 {
                return Err("compare arity");
            }
            let bits = width(1)?;
            word(0, bits)?;
            match a[3].as_str() {
                Some("unsigned") => {
                    if !a[4].is_null() || a[2].as_u64().filter(|v| *v <= 1u64 << bits).is_none() {
                        return Err("unsigned comparison");
                    }
                }
                Some("signed") => {
                    let half = 1i64 << (bits - 1);
                    if !a[4].is_null()
                        || a[2]
                            .as_i64()
                            .filter(|v| (-half..=half).contains(v))
                            .is_none()
                    {
                        return Err("signed comparison");
                    }
                }
                Some("low") => {
                    let low = width(4)?;
                    if low > bits || a[2].as_u64().filter(|v| *v <= 1u64 << low).is_none() {
                        return Err("low comparison");
                    }
                }
                _ => return Err("comparison mode"),
            }
        }
        "phi-select" => {
            if a.len() != 3 {
                return Err("phi arity");
            }
            word(0, 1)?;
            word(1, 50)?;
            word(2, 50)?;
        }
        "external" => {
            if a.len() != 2 || a[0].as_i64().is_none() || a[1].as_i64().filter(|v| *v > 0).is_none()
            {
                return Err("external rational");
            }
        }
        _ => return Err("unselected gate"),
    }
    Ok(())
}

fn run_case(case: &J) -> (J, Stats, J) {
    validate_case(case).expect("invalid producer case");
    let gate = case["gate"].as_str().unwrap();
    let label = case["id"].as_str().unwrap();
    let args = case["args"].as_array().unwrap();
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut stats = Stats::default();
    let mut detail = json!({});
    let output = match gate {
        "shift" => {
            let bits = args[1].as_u64().unwrap() as u8;
            let shift = args[2].as_u64().unwrap() as u8;
            let range = cv::SignedRange::full_ring(bits).unwrap();
            let (out_bits, op) = match args[3].as_str().unwrap() {
                "TR" => (bits - shift, cv::Operation::Tr { shift, range }),
                "ARS" => (bits, cv::Operation::Ars { shift, range }),
                _ => panic!("unknown shift"),
            };
            let input = Zeroizing::new(share(args[0].as_u64().unwrap(), bits));
            let out = convert(spec(bits, out_bits, op), *input, label, &budget, &mut stats);
            json!({"bits":out_bits,"word":final_word(&out,out_bits)})
        }
        "b2a" => {
            let bits = args[2].as_u64().unwrap() as u8;
            // Fixture arguments specify XOR semantics; randomize local shares
            // anew rather than reusing the fixture's displayed share pair.
            let input = Zeroizing::new(bool_share(
                args[0].as_u64().unwrap() ^ args[1].as_u64().unwrap(),
            ));
            let out = convert(
                spec(1, bits, cv::Operation::BooleanToArithmetic),
                *input,
                label,
                &budget,
                &mut stats,
            );
            json!(final_word(&out, bits))
        }
        "compare" => {
            let (output, binding) = compare(args, label, &budget, &mut stats);
            detail = binding;
            json!(output)
        }
        "phi-select" => json!(phi(args, label, &budget, &mut stats)),
        "external" => {
            detail =
                json!({"execution":"local-exact-rational-native-frontend; no protected issuance"});
            let encoded = fusefss::conformance::encode_q12(
                args[0].as_i64().unwrap() as i128,
                args[1].as_i64().unwrap() as i128,
            )
            .unwrap();
            json!((encoded as u64) & mask(50))
        }
        _ => panic!("not a selected gate"),
    };
    assert_eq!(budget.used_bytes(), 0, "undrained testcase {label}");
    (output, stats, detail)
}

/// Shared finite primitive producer for the full composition exporter. The
/// selected subset retains its own original schema and receipt semantics.
pub(super) fn run_selected_case(case: &J) -> (J, J, J) {
    let (output, stats, detail) = run_case(case);
    (output, stats.json(), detail)
}

fn frozen() -> (J, J) {
    assert_eq!(hash(FIXTURE), FIXTURE_HASH);
    assert_eq!(hash(TABLE_FIXTURE), TABLE_FIXTURE_HASH);
    assert_eq!(hash(SOURCE_LOCK), SOURCE_HASH);
    let fixture: J = serde_json::from_slice(FIXTURE).unwrap();
    let tables: J = serde_json::from_slice(TABLE_FIXTURE).unwrap();
    assert_eq!(fixture["schema"], "pllm.fusefss.gate-set.v1");
    assert_eq!(fixture["profile"], fusefss::PROFILE);
    assert_eq!(tables["schema"], "pllm.fusefss.table-replay.v1");
    assert_eq!(tables["coverage"], "public-table-lookup-only");
    assert_eq!(fixture["source_lock_sha256"], SOURCE_HASH);
    assert_eq!(tables["gate_fixture_sha256"], FIXTURE_HASH);
    assert_eq!(tables["source_lock_sha256"], SOURCE_HASH);
    (fixture, tables)
}
fn selected(fixture: &J) -> Vec<&J> {
    fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|case| SELECTED.contains(&case["gate"].as_str().unwrap()))
        .collect()
}
fn gate_header() -> J {
    json!({"schema":"pllm.fusefss.gate-set.v2/selected-native-output",
        "fixture_sha256":FIXTURE_HASH,"execution":"protected-local-reconstruction",
        "implementation":implementation(),"selected_gates":SELECTED,"selected_case_count":66,
        "source_lock_sha256":SOURCE_HASH,"coverage":"selected-gates-only"})
}
fn table_header() -> J {
    json!({"schema":"pllm.fusefss.table-replay.v1/native-output",
        "fixture_sha256":TABLE_FIXTURE_HASH,"gate_fixture_sha256":FIXTURE_HASH,
        "source_lock_sha256":SOURCE_HASH,"execution":"protected-local-reconstruction",
        "implementation":implementation(),"coverage":"public-table-lookup-only",
        "table_count":8,"case_count":18560})
}
pub(super) fn source_manifest() -> BTreeMap<String, String> {
    SOURCES
        .iter()
        .map(|(name, bytes)| ((*name).into(), hash(bytes)))
        .collect()
}
fn implementation() -> String {
    let manifest = serde_json::to_vec(&source_manifest()).unwrap();
    format!("pllm-native-f2-test-sha256:{}", hash(&manifest))
}

fn fingerprints() -> BTreeMap<String, String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut snapshots = source_manifest();
    for (name, bytes) in [
        (
            "paper/reproduction-contracts/fusefss/gate-set/fixtures.json",
            FIXTURE,
        ),
        (
            "paper/reproduction-contracts/fusefss/gate-set/table-fixtures.json",
            TABLE_FIXTURE,
        ),
        (
            "paper/reproduction-contracts/fusefss/gate-set/source-lock.json",
            SOURCE_LOCK,
        ),
    ] {
        snapshots.insert(name.into(), hash(bytes));
    }
    let mut actual = BTreeMap::new();
    for (name, expected) in snapshots {
        let observed = hash(&fs::read(root.join(&name)).unwrap());
        assert_eq!(observed, expected, "compiled input changed: {name}");
        actual.insert(name, observed);
    }
    actual
}

fn write_line(file: &mut fs::File, value: &J) {
    let bytes = serde_json::to_vec(value).unwrap();
    assert!(bytes.len() < 256 << 10);
    file.write_all(&bytes).unwrap();
    file.write_all(b"\n").unwrap();
}
fn validate_opt_in(flag: Option<&str>, release: bool) -> Result<(), &'static str> {
    if !release {
        return Err("export requires cargo test --release");
    }
    if flag != Some("1") {
        return Err("explicit PLLM_FUSEFSS_EXPORT_GATE_SET=1 required");
    }
    Ok(())
}
fn output_dir() -> std::path::PathBuf {
    let flag = std::env::var("PLLM_FUSEFSS_EXPORT_GATE_SET").ok();
    validate_opt_in(flag.as_deref(), !cfg!(debug_assertions)).unwrap();
    let dir = std::env::var_os("PLLM_FUSEFSS_F2_EXPORT_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/fusefss-gates-native-v1")
        });
    assert!(!dir.as_os_str().is_empty());
    fs::create_dir_all(&dir).unwrap();
    dir
}
fn receipt(
    scope: &str,
    stats: &Stats,
    rows: Vec<J>,
    before: BTreeMap<String, String>,
    after: BTreeMap<String, String>,
    output: &Path,
) -> J {
    assert_eq!(before, after);
    json!({"schema":"pllm.fusefss.f2.native-export-receipt.v1","coverage":scope,
        "implementation":implementation(),"source_sha256":source_manifest(),
        "fixture_sha256":FIXTURE_HASH,"table_fixture_sha256":TABLE_FIXTURE_HASH,
        "source_lock_sha256":SOURCE_HASH,"source_input_fingerprints_before":before,
        "source_input_fingerprints_after":after,"source_inputs_unchanged":true,
        "output_sha256":hash(&fs::read(output).unwrap()),"totals":stats.json(),"records":rows,
        "accounting":{"material":"cumulative issued private payload layout, both parties; not allocator residency or transport",
            "budget":"peak exact reserved envelope derived from resources before each issuance, including live prior outputs/table; not RSS",
            "excluded_test_harness_owners":"fixture JSON, caller buffers, bounded descriptor parser/borrowed descriptor, export buffers, filesystem I/O",
            "freshness":"each issuance uses production OS randomness for masks, keys and correlations; input shares freshly randomized",
            "frames":"actual exported peer frame lengths; checked against resources"},
        "limitations":["finite test outputs only","no source/model/SDK completion claim","not independent security qualification"]})
}

#[test]
#[ignore = "opt-in release exporter; selected 66 cases only"]
fn export_selected_gates() {
    let (fixture, _) = frozen();
    let cases = selected(&fixture);
    assert_eq!(cases.len(), 66);
    let before = fingerprints();
    let dir = output_dir();
    let path = dir.join("selected-gates.jsonl");
    let mut output = fs::File::create_new(&path).unwrap();
    write_line(&mut output, &gate_header());
    let mut total = Stats::default();
    let mut rows = Vec::new();
    for case in cases {
        let (value, stats, detail) = run_case(case);
        write_line(
            &mut output,
            &json!({"id":case["id"],"gate":case["gate"],"output":value}),
        );
        rows.push(
            json!({"id":case["id"],"gate":case["gate"],"resources":stats.json(),"binding":detail}),
        );
        total.add(&stats);
    }
    assert_eq!(total.issuances, 62);
    assert_eq!(total.lanes, 62);
    output.sync_all().unwrap();
    drop(output);
    assert!(fs::metadata(&path).unwrap().len() <= 8 << 20);
    let report = receipt(
        "selected-gates-only",
        &total,
        rows,
        before,
        fingerprints(),
        &path,
    );
    let mut out = fs::File::create_new(dir.join("selected-gates.receipt.json")).unwrap();
    serde_json::to_writer_pretty(&mut out, &report).unwrap();
    out.write_all(b"\n").unwrap();
}

fn table_values(
    table: &lookup::Table<'_>,
    name: &str,
    count: usize,
    budget: &Budget,
    stats: &mut Stats,
) -> Vec<u64> {
    assert_eq!(
        count,
        table.entries(),
        "replay every entry of complete Table"
    );
    let mut values = Vec::with_capacity(count);
    for start in (0..count).step_by(4) {
        let lanes = (count - start).min(4);
        let r = table.resources(lanes).unwrap();
        stats.admission(budget, r.issuance_reservation_bytes);
        // Always issue the COMPLETE zero-extended Table: all public chunks,
        // including chunks that do not contain this test index, get fresh keys.
        let [mut a, mut b] = table.issue(lanes, context(name, start)).unwrap();
        stats.observe(budget);
        let mut left = Zeroizing::new(Vec::with_capacity(lanes));
        let mut right = Zeroizing::new(Vec::with_capacity(lanes));
        for index in start..start + lanes {
            let s = Zeroizing::new(share(index as u64, table.bits()));
            left.push(s[0]);
            right.push(s[1]);
        }
        let af = a.start(&left).unwrap();
        let bf = b.start(&right).unwrap();
        let bytes = af.as_bytes().len() + bf.as_bytes().len();
        assert_eq!(bytes, 2 * r.peer_frame_bytes_per_party);
        let ao = a.finish(bf.as_bytes()).unwrap();
        let bo = b.finish(af.as_bytes()).unwrap();
        assert!(a.is_consumed() && b.is_consumed());
        for (&x, &y) in ao.shares().iter().zip(bo.shares()) {
            values.push(x.wrapping_add(y) & mask(table.bits()));
        }
        stats.observe(budget);
        stats.issuances += 1;
        stats.lanes += lanes as u64;
        stats.dcf_keys += r.dcf_keys_per_party as u64;
        stats.dcf_evals += r.dcf_evaluations_per_party as u64;
        stats.material += 2 * r.private_key_bytes_per_party as u64;
        stats.public_table_copies += 2 * r.public_table_bytes_per_party as u64;
        stats.frames += 2;
        stats.frame_bytes += bytes as u64;
        drop((a, b, af, bf, ao, bo));
        assert_eq!(budget.used_bytes(), r.table_reservation_bytes);
    }
    values
}
fn words_hash(values: &[u64]) -> String {
    let mut h = Sha256::new();
    for x in values {
        h.update(x.to_le_bytes());
    }
    hex(&h.finalize())
}

fn validate_table(meta: &J, name: &str, values: &[u64]) -> Result<(), &'static str> {
    if !TABLE_NAMES.contains(&name)
        || values.is_empty()
        || values.len() > 8192
        || values.iter().any(|&v| v > mask(50))
    {
        return Err("table values/range");
    }
    if meta.as_object().map(|o| o.len()) != Some(7)
        || meta["name"] != name
        || meta["input_bits"] != 50
        || meta["output_bits"] != 50
        || meta["start"] != 0
        || meta["stop"].as_u64() != Some(values.len() as u64)
        || meta["count"].as_u64() != Some(values.len() as u64)
        || meta["sha256_u64le"].as_str() != Some(words_hash(values).as_str())
    {
        return Err("table metadata binding");
    }
    Ok(())
}

#[test]
#[ignore = "long full-table replay; parent review/explicit release opt-in required"]
fn export_tables() {
    let (fixture, tables) = frozen();
    let metadata = tables["tables"].as_array().unwrap();
    assert_eq!(metadata.len(), 8);
    let before = fingerprints();
    let dir = output_dir();
    let path = dir.join("tables.jsonl");
    let mut output = fs::File::create_new(&path).unwrap();
    write_line(&mut output, &table_header());
    let mut rows = Vec::new();
    let mut total = Stats::default();
    let mut outputs = 0;
    for (name, meta) in TABLE_NAMES.iter().zip(metadata) {
        assert_eq!(meta["name"], *name);
        let values: Vec<u64> = fixture["tables"][name]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        validate_table(meta, name, &values).unwrap();
        let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
        let table =
            lookup::Table::new_zero_extended(50, &values, source_digest(), &budget).unwrap();
        let binding = hex(&table.digest());
        let chunks = table.resources(4).unwrap().chunks_per_lane;
        let mut stats = Stats::default();
        let result = table_values(&table, name, values.len(), &budget, &mut stats);
        assert_eq!(result.len(), values.len());
        outputs += result.len();
        // Hash comes from the computed outputs, never from an expected field.
        write_line(
            &mut output,
            &json!({"table":name,"sha256_u64le":words_hash(&result),"outputs":result}),
        );
        drop(table);
        assert_eq!(budget.used_bytes(), 0);
        rows.push(
            json!({"table":name,"table_digest":binding,"outputs":values.len(),
            "chunks_per_lane":chunks,"chunk_entries":lookup::CHUNK_ENTRIES,
            "original_public_value_bytes":values.len()*8,
            "dcf_path_levels_per_party":stats.dcf_evals*50,
            "resources":stats.json()}),
        );
        total.add(&stats);
    }
    assert_eq!(outputs, 18560);
    assert_eq!(total.issuances, 4640);
    assert_eq!(total.lanes, 18560);
    output.sync_all().unwrap();
    drop(output);
    assert!(fs::metadata(&path).unwrap().len() <= 2 << 20);
    let report = receipt(
        "public-table-lookup-only",
        &total,
        rows,
        before,
        fingerprints(),
        &path,
    );
    let mut out = fs::File::create_new(dir.join("tables.receipt.json")).unwrap();
    serde_json::to_writer_pretty(&mut out, &report).unwrap();
    out.write_all(b"\n").unwrap();
}

#[path = "gate_conformance_tests.rs"]
mod tests;
