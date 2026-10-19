//! Explicitly opted-in, bounded, offline native conformance export.
//! Never compiled into production; fixture expected values are not consulted.
use super::*;
use crate::shared_rescale_reference::Context;
use serde_json::{json, Value as J};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
};

const FIXTURE: &[u8] = include_bytes!(
    "../../../../paper/reproduction-contracts/fusefss/native-conformance/fixtures.json"
);
const FIXTURE_HASH: &str = "64a2dfd493eab789a8aaca9fb12326affc7b12c09cc32ad3bcfbfeb8e3f7babd";
const SCHEMA: &str = "pllm.fusefss.independent-native-conformance.v1";
const MAX_OUTPUT_BYTES: usize = 32 << 20;

pub(crate) type ExportResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Supplied by the parent integration test, not independently authenticated by
/// this adapter. Sources are nonempty portable labels and lowercase SHA-256s.
pub(crate) struct ProtectedBackendIdentity<'a> {
    pub name: &'a str,
    pub sources: &'a [(&'a str, &'a str)],
}

/// Existing parent-approved material ceiling and its backend-specific estimate
/// for batches of 1,2,3,4 lanes. Not a new resource ceiling and not a measurement.
/// Parent remains responsible for live-owner, transport and issuance admission.
pub(crate) struct ProtectedWorkAdmission {
    pub material_bytes_by_lanes: [u64; 4],
    pub cumulative_material_limit: u64,
}

fn preflight_material(admission: &ProtectedWorkAdmission, batches: [u64; 4]) -> ExportResult<u64> {
    if admission.cumulative_material_limit == 0
        || admission.material_bytes_by_lanes.iter().any(|&v| v == 0)
    {
        return Err("positive parent material estimates/limit required".into());
    }
    let mut total = 0u64;
    for (&count, &bytes) in batches.iter().zip(&admission.material_bytes_by_lanes) {
        total = total
            .checked_add(count.checked_mul(bytes).ok_or("material work overflow")?)
            .ok_or("material total overflow")?;
    }
    if total > admission.cumulative_material_limit {
        return Err("protected replay exceeds parent material allowance".into());
    }
    Ok(total)
}

fn require_protected_opt_in(value: Option<&str>) -> ExportResult<()> {
    if value != Some("1") {
        return Err("set PLLM_FUSEFSS_EXPORT_PROTECTED_GELU=1 explicitly".into());
    }
    Ok(())
}

fn backend_identity(identity: &ProtectedBackendIdentity<'_>) -> ExportResult<J> {
    if identity.name.is_empty()
        || identity.name.len() > 64
        || !identity
            .name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c))
    {
        return Err("bounded simple backend name required".into());
    }
    if identity.sources.is_empty() || identity.sources.len() > 16 {
        return Err("one to sixteen backend source identities required".into());
    }
    let mut sources = BTreeMap::new();
    for &(label, digest) in identity.sources {
        if label.is_empty()
            || label.len() > 192
            || !label.bytes().all(|c| (33..=126).contains(&c))
            || label.starts_with('/')
            || label.split('/').any(|p| p == "..")
            || digest.len() != 64
            || !digest
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
            || sources.insert(label, digest).is_some()
        {
            return Err("invalid or duplicate backend source identity".into());
        }
    }
    Ok(
        json!({"name":identity.name,"source_sha256":sources,"binding":"parent-supplied; adapter does not authenticate backend implementation"}),
    )
}

fn load_fixture() -> ExportResult<(J, BTreeMap<String, Descriptor>)> {
    if FIXTURE.len() > 16 << 20 || hex(&Sha256::digest(FIXTURE)) != FIXTURE_HASH {
        return Err("fixture identity/byte cap".into());
    }
    let fixture: J = serde_json::from_slice(FIXTURE)?;
    if fixture["schema"] != SCHEMA {
        return Err("fixture schema".into());
    }
    let entries = fixture["descriptors"].as_array().ok_or("descriptor list")?;
    if entries.len() != 184 {
        return Err("descriptor count".into());
    }
    let mut descriptors = BTreeMap::new();
    for entry in entries {
        let bytes = serde_json::to_vec(&entry["descriptor"])?;
        let descriptor = Descriptor::from_canonical_bytes(&bytes)?;
        let id = hex(&descriptor.digest());
        if entry["id"] != id || descriptors.insert(id, descriptor).is_some() {
            return Err("descriptor identity/duplicate".into());
        }
    }
    if fixture["runs"].as_array().ok_or("run list")?.len() != 1552 {
        return Err("run count".into());
    }
    Ok((fixture, descriptors))
}

fn bind_run(descriptor: &Descriptor, run: &J) -> ExportResult<DealerPlan> {
    if run["descriptor_id"].as_str() != Some(hex(&descriptor.digest()).as_str()) {
        return Err("run descriptor substitution".into());
    }
    let mask = run["mask"].as_u64().ok_or("mask")?;
    let n = 1u64 << descriptor.word().bits;
    if mask >= n {
        return Err("mask range".into());
    }
    let words = run["words"].as_array().ok_or("words")?;
    let masked = run["masked_words"].as_array().ok_or("masked words")?;
    if words.is_empty() || words.len() > 100_000 || words.len() != masked.len() {
        return Err("word shape".into());
    }
    for (word, masked) in words.iter().zip(masked) {
        let word = word.as_u64().ok_or("word type")?;
        if word >= n || masked.as_u64() != Some((word + mask) % n) {
            return Err("masked word binding".into());
        }
    }
    let plan = descriptor.compile(mask)?;
    if plan.digest() != descriptor.digest() || plan.shape() != descriptor.shape() {
        return Err("compiled binding".into());
    }
    Ok(plan)
}

/// Parse the frozen fixture's public `SCHEMA/rNNNN` context, bind it to its run
/// ID, native descriptor, fixture identity and declared mask, and derive Context.
/// `plan` is the native descriptor digest; session/operation are domain-separated
/// hashes. No fresh-secret/security property follows from this public test context.
fn run_context(run: &J, descriptor: &Descriptor) -> ExportResult<Context> {
    let id = run["id"].as_str().ok_or("run id")?;
    if id.len() != 5
        || !id.starts_with('r')
        || !id.as_bytes()[1..].iter().all(u8::is_ascii_digit)
        || id[1..].parse::<usize>()? >= 1552
    {
        return Err("run id encoding".into());
    }
    let context = run["context"].as_str().ok_or("context")?;
    if context != format!("{SCHEMA}/{id}") {
        return Err("context/run binding".into());
    }
    let mask = run["mask"].as_u64().ok_or("mask")?;
    if run["descriptor_id"].as_str() != Some(hex(&descriptor.digest()).as_str())
        || mask >= 1u64 << descriptor.word().bits
    {
        return Err("context descriptor/mask binding".into());
    }
    let session = Sha256::new()
        .chain_update(b"PLLM-FuseFSS-conformance-session-v1\0")
        .chain_update(FIXTURE_HASH.as_bytes())
        .chain_update(context.as_bytes())
        .finalize()
        .into();
    let operation = Sha256::new()
        .chain_update(b"PLLM-FuseFSS-conformance-operation-v1\0")
        .chain_update(descriptor.digest())
        .chain_update(context.as_bytes())
        .chain_update(mask.to_le_bytes())
        .finalize()
        .into();
    Ok(Context {
        plan: descriptor.digest(),
        session,
        operation,
        first_tensor_index: 0,
    })
}

fn batch_context(base: Context, offset: usize, lanes: usize) -> ExportResult<Context> {
    if !(1..=4).contains(&lanes) {
        return Err("one to four lanes required".into());
    }
    let offset = u64::try_from(offset)?;
    let first = base
        .first_tensor_index
        .checked_add(offset)
        .ok_or("tensor index overflow")?;
    first
        .checked_add(u64::try_from(lanes - 1)?)
        .ok_or("last tensor index overflow")?;
    Ok(Context {
        first_tensor_index: first,
        ..base
    })
}

fn validate_batch_outputs(outputs: &[u64], lanes: usize, word: ArithmeticType) -> ExportResult<()> {
    if !(1..=4).contains(&lanes)
        || outputs.len() != lanes
        || outputs.iter().any(|&v| v >= 1u64 << word.bits)
    {
        return Err("callback output length/ring range".into());
    }
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut output, byte| {
        write!(output, "{byte:02x}").expect("String formatting");
        output
    })
}

/// Test-only exact finite signed rational encoding, checked throughout. This is
/// not an external tensor frontend and makes no binary-float conversion choice.
fn encode_q12(numerator: i128, denominator: i128) -> Result<i64> {
    if denominator <= 0 {
        return Err(Error::Range);
    }
    let scaled = numerator.checked_mul(4096).ok_or(Error::Range)?;
    let lower = scaled.div_euclid(denominator);
    let remainder = scaled.rem_euclid(denominator);
    // Compare r with d-r instead of potentially overflowing 2*r.
    let complement = denominator.checked_sub(remainder).ok_or(Error::Range)?;
    let increment = remainder > complement || (remainder == complement && lower.rem_euclid(2) == 1);
    let rounded = lower
        .checked_add(i128::from(increment))
        .ok_or(Error::Range)?;
    if !(-(1i128 << 49)..(1i128 << 49)).contains(&rounded) {
        return Err(Error::Range);
    }
    i64::try_from(rounded).map_err(|_| Error::Range)
}

struct BoundedWriter<W> {
    inner: W,
    bytes: usize,
    hash: Sha256,
}
impl<W: Write> Write for BoundedWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if self
            .bytes
            .checked_add(bytes.len())
            .filter(|&n| n <= MAX_OUTPUT_BYTES)
            .is_none()
        {
            return Err(std::io::Error::other("conformance output byte cap"));
        }
        let n = self.inner.write(bytes)?;
        self.bytes += n;
        self.hash.update(&bytes[..n]);
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

macro_rules! emit {
    ($out:expr,$value:expr) => {
        serde_json::to_writer($out, $value).map_err(std::io::Error::other)
    };
}

fn native_shape(s: Shape) -> J {
    json!({"source_intervals":s.source_intervals,"padded_intervals":s.padded_intervals,
        "boolean_outputs":s.boolean_outputs,"queries":s.queries,"degree":Shape::DEGREE,
        "coefficients_per_interval":Shape::COEFFICIENTS_PER_INTERVAL,"horner_products":Shape::HORNER_BEAVER_PRODUCTS})
}

fn diagnostics<W: Write>(out: &mut W, plan: &DealerPlan) -> Result<()> {
    // Only this cfg(test) adapter sees ordinals. Handoff remains trusted dealer.
    let (boundaries, ordinals) = plan.test_partition();
    plan.with_dealer_handoff(|h| {
        let write = |out: &mut W| -> std::io::Result<()> {
            out.write_all(b"{\"boundaries\":")?;
            emit!(&mut *out, &boundaries)?;
            out.write_all(b",\"ordinals\":")?;
            emit!(&mut *out, &ordinals)?;
            out.write_all(b",\"payloads\":")?;
            emit!(&mut *out, &h.coefficients)?;
            out.write_all(b",\"queries\":[")?;
            for (i, q) in h.predicate_queries.iter().enumerate() {
                if i != 0 {
                    out.write_all(b",")?;
                }
                emit!(&mut *out, &[u64::from(q.bits()), q.shift(), q.threshold()])?;
            }
            out.write_all(b"]}")
        };
        write(out).map_err(|_| Error::Encoding)
    })
}

fn write_header<W: Write>(
    out: &mut W,
    suite: &str,
    execution: &str,
    implementation: &str,
) -> ExportResult<()> {
    emit!(
        &mut *out,
        &json!({"schema":format!("{SCHEMA}/native-output"),"fixture_sha256":FIXTURE_HASH,
        "suite":suite,"execution":execution,"implementation":implementation})
    )?;
    out.write_all(b"\n")?;
    Ok(())
}

fn write_run_prefix<W: Write>(out: &mut W, run: &J, plan: &DealerPlan) -> ExportResult<()> {
    let native_bytes = plan.reference_record()?.len();
    out.write_all(b"{\"id\":")?;
    emit!(&mut *out, &run["id"])?;
    out.write_all(b",\"descriptor_id\":")?;
    emit!(&mut *out, &hex(&plan.digest()))?;
    out.write_all(b",\"mask\":")?;
    emit!(&mut *out, &run["mask"])?;
    out.write_all(b",\"context\":")?;
    emit!(&mut *out, &run["context"])?;
    out.write_all(b",\"shape\":")?;
    emit!(&mut *out, &native_shape(plan.shape()))?;
    out.write_all(b",\"native_bytes\":")?;
    emit!(&mut *out, &native_bytes)?;
    out.write_all(b",\"compiled\":")?;
    diagnostics(out, plan)?;
    out.write_all(b",\"outputs\":[")?;
    Ok(())
}

fn write_encodings_footer<W: Write>(out: &mut W, fixture: &J) -> ExportResult<usize> {
    let encodings = fixture["external_encoding"]
        .as_array()
        .ok_or("external encodings")?;
    if encodings.len() != 38 {
        return Err("encoding count".into());
    }
    out.write_all(b"{\"external_encoding\":[")?;
    for (i, e) in encodings.iter().enumerate() {
        let numerator = i128::from(e["numerator"].as_i64().ok_or("numerator")?);
        let denominator = i128::from(e["denominator"].as_i64().ok_or("denominator")?);
        if i != 0 {
            out.write_all(b",")?;
        }
        emit!(&mut *out, &encode_q12(numerator, denominator)?)?;
    }
    out.write_all(b"]}\n")?;
    Ok(encodings.len())
}

fn source_identity(execution: &str) -> J {
    let files: &[(&str, &[u8])] = &[
        (
            "crates/pllm-garble/src/fusefss/mod.rs",
            include_bytes!("mod.rs"),
        ),
        (
            "crates/pllm-garble/src/fusefss/json.rs",
            include_bytes!("json.rs"),
        ),
        (
            "crates/pllm-garble/src/fusefss/compile.rs",
            include_bytes!("compile.rs"),
        ),
        (
            "crates/pllm-garble/src/fusefss/fixture.rs",
            include_bytes!("fixture.rs"),
        ),
        (
            "crates/pllm-garble/src/fusefss/tests.rs",
            include_bytes!("tests.rs"),
        ),
        (
            "crates/pllm-garble/src/fusefss/conformance.rs",
            include_bytes!("conformance.rs"),
        ),
        (
            "crates/pllm-garble/src/fusefss/accounting.rs",
            include_bytes!("accounting.rs"),
        ),
    ];
    let sources: Vec<J> = files.iter().map(|(path,bytes)| json!({"path":path,"bytes":bytes.len(),"sha256":hex(&Sha256::digest(bytes))})).collect();
    json!({"package":env!("CARGO_PKG_NAME"),"version":env!("CARGO_PKG_VERSION"),
        "architecture":std::env::consts::ARCH,"os":std::env::consts::OS,"sources":sources,
        "gelu_descriptor_sha256":hex(&Descriptor::gelu().unwrap().digest()),
        "source_lock_sha256":hex(&Sha256::digest(include_bytes!("../../../../paper/reproduction-contracts/fusefss/source-lock.json"))),
        "execution":execution,
        "scope":"offline test-only native adapter; no backend authentication, model, or security qualification"})
}

fn write_clear_all<W: Write>(
    out: &mut W,
    fixture: &J,
    descriptors: &BTreeMap<String, Descriptor>,
) -> ExportResult<(usize, usize)> {
    write_header(
        &mut *out,
        "all",
        "compiler-clear",
        "pllm-garble native FuseFSS cfg(test) compiler exporter v1",
    )?;
    let mut evaluations = 0usize;
    for run in fixture["runs"].as_array().ok_or("run list")? {
        let descriptor = descriptors
            .get(run["descriptor_id"].as_str().ok_or("descriptor id")?)
            .ok_or("unknown descriptor")?;
        let words = run["words"].as_array().ok_or("words")?;
        let masked_words = run["masked_words"].as_array().ok_or("masked words")?;
        evaluations = evaluations
            .checked_add(words.len())
            .ok_or("evaluation overflow")?;
        if evaluations > 100_000 {
            return Err("evaluation cap".into());
        }
        let plan = bind_run(descriptor, run)?;
        write_run_prefix(&mut *out, run, &plan)?;
        for (i, masked) in masked_words.iter().enumerate() {
            let (arithmetic, bits) =
                plan.clear_interpret(masked.as_u64().ok_or("masked word")?, descriptor.digest())?;
            let mut bit_codes = [0u8; MAX_BOOLEAN_OUTPUTS];
            for (i, &b) in bits.iter().enumerate() {
                bit_codes[i] = u8::from(b);
            }
            if i != 0 {
                out.write_all(b",")?;
            }
            emit!(&mut *out, &(arithmetic, &bit_codes[..bits.len()]))?;
        }
        out.write_all(b"]}\n")?;
    }
    if evaluations != 58400 {
        return Err("all-suite evaluation count".into());
    }
    let encodings = write_encodings_footer(&mut *out, fixture)?;
    out.flush()?;
    Ok((evaluations, encodings))
}

/// Not run by default. Both the ignored-test name and the explicit environment
/// value are required; fixed repo-relative target path prevents accidental output
/// anywhere else. The fixture and complete output both have hard byte ceilings.
#[test]
#[ignore = "requires PLLM_FUSEFSS_EXPORT_COMPILER_CONFORMANCE=1; emits dealer-only target diagnostics"]
fn export_compiler_conformance_all() -> std::result::Result<(), Box<dyn std::error::Error>> {
    if std::env::var("PLLM_FUSEFSS_EXPORT_COMPILER_CONFORMANCE").as_deref() != Ok("1") {
        return Err("set PLLM_FUSEFSS_EXPORT_COMPILER_CONFORMANCE=1 explicitly".into());
    }
    let (fixture, descriptors) = load_fixture()?;
    let runs = fixture["runs"].as_array().ok_or("run list")?;
    assert_eq!(runs.len(), 1552);
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/fusefss-native-conformance-v1");
    std::fs::create_dir_all(&directory)?;
    let output = File::create(directory.join("compiler.jsonl"))?;
    let mut out = BoundedWriter {
        inner: BufWriter::with_capacity(8192, output),
        bytes: 0,
        hash: Sha256::new(),
    };
    let (evaluations, encoding_count) = write_clear_all(&mut out, &fixture, &descriptors)?;
    let receipt = json!({"schema":"pllm.fusefss.native-compiler-receipt.v1","suite":"all","execution":"compiler-clear",
        "fixture_sha256":FIXTURE_HASH,"descriptors":descriptors.len(),"runs":runs.len(),"gate_evaluations":evaluations,
        "external_encoding_cases":encoding_count,"output":{"path":"target/fusefss-native-conformance-v1/compiler.jsonl",
        "bytes":out.bytes,"sha256":hex(&out.hash.finalize())},"source_identity":source_identity("compiler-clear"),
        "independent_checker":"run check_native.py against output separately; this exporter does not certify itself"});
    serde_json::to_writer_pretty(
        File::create(directory.join("compiler-receipt.json"))?,
        &receipt,
    )?;
    serde_json::to_writer_pretty(
        File::create(directory.join("compiler-owner-receipt.json"))?,
        &super::accounting::gelu_owner_receipt(),
    )?;
    Ok(())
}

/// Reusable GELU protected-output exporter for a parent-owned ignored test.
/// Requires `PLLM_FUSEFSS_EXPORT_PROTECTED_GELU=1`; does not run from default tests.
///
/// `evaluate` receives 1..=4 ORIGINAL ring words, the actual immutable parsed
/// frozen descriptor, and a deterministic run-bound Context with checked lane
/// offset. It must return exactly one reconstructed arithmetic word per input.
/// It must use ordinary fresh-random issuance (e.g. CompiledGelu::issue), with
/// fresh private input masks, keys, triples and nonce for every batch/lane. It
/// receives NO fixture mask. Never derive crypto randomness from this public
/// Context, reuse material, call the clear interpreter, or return expected data.
///
/// JSONL `mask`/`compiled` are a SEPARATE real compiler diagnostic surface using
/// controlled PUBLIC fixture masks. They do not report protected runtime masks.
/// Only arithmetic outputs come from the callback. Header and receipt state this
/// separation. The independent checker checks both surfaces, not their runtime
/// mask equality, randomness, backend authenticity, or cryptographic security.
///
/// All 8 runs/43,544 words are mandatory. Full material work is checked against
/// the parent-provided existing allowance BEFORE file creation or any callback.
/// This helper does not admit backend live owners or raise resource ceilings.
/// Success emits only fixed target paths, bounded JSONL and an identity receipt;
/// no independent-checker success or measured-material claim is fabricated.
pub(crate) fn export_protected_gelu<F>(
    identity: ProtectedBackendIdentity<'_>,
    admission: ProtectedWorkAdmission,
    mut evaluate: F,
) -> ExportResult<J>
where
    F: FnMut(&[u64], &Descriptor, Context) -> ExportResult<Vec<u64>>,
{
    let directory =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/fusefss-native-protected-v1");
    // Invalidate prior success before any guard can reject this attempt. A
    // complete native run additionally requires its separate runtime receipt,
    // successful process exit and an independent checker result.
    let receipt_path = directory.join("protected-receipt.json");
    for receipt in [receipt_path.clone(), directory.join("runtime-receipt.json")] {
        match std::fs::remove_file(receipt) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
    }
    require_protected_opt_in(
        std::env::var("PLLM_FUSEFSS_EXPORT_PROTECTED_GELU")
            .ok()
            .as_deref(),
    )?;
    let backend = backend_identity(&identity)?;
    let (fixture, descriptors) = load_fixture()?;
    let descriptor = descriptors
        .get(GELU_DIGEST_HEX)
        .ok_or("frozen GELU missing")?;
    if descriptor.digest() != Descriptor::gelu()?.digest()
        || descriptor.shape().boolean_outputs != 0
    {
        return Err("frozen GELU substitution".into());
    }
    let selected: Vec<&J> = fixture["runs"]
        .as_array()
        .ok_or("runs")?
        .iter()
        .filter(|r| r["descriptor_id"].as_str() == Some(GELU_DIGEST_HEX))
        .collect();
    if selected.len() != 8 {
        return Err("GELU run count".into());
    }
    let mut words_total = 0usize;
    let mut batch_counts = [0u64; 4];
    // Validate all bindings and index/work bounds before any protected work.
    for run in &selected {
        let _diagnostic = bind_run(descriptor, run)?;
        let base = run_context(run, descriptor)?;
        let count = run["words"].as_array().ok_or("words")?.len();
        words_total = words_total
            .checked_add(count)
            .ok_or("gate count overflow")?;
        for offset in (0..count).step_by(4) {
            let lanes = (count - offset).min(4);
            batch_context(base, offset, lanes)?;
            batch_counts[lanes - 1] = batch_counts[lanes - 1]
                .checked_add(1)
                .ok_or("batch count overflow")?;
        }
    }
    if words_total != 43544 {
        return Err("GELU evaluation count".into());
    }
    let estimated_material = preflight_material(&admission, batch_counts)?;
    std::fs::create_dir_all(&directory)?;
    let mut out = BoundedWriter {
        inner: BufWriter::with_capacity(8192, File::create(directory.join("protected.jsonl"))?),
        bytes: 0,
        hash: Sha256::new(),
    };
    let implementation=format!("{}; protected outputs: fresh-random issuance callback; mask/compiled: separate public-fixture compiler diagnostics",identity.name);
    write_header(
        &mut out,
        "gelu",
        "protected-local-reconstruction",
        &implementation,
    )?;
    let mut actual_batches = 0u64;
    let mut actual_words = 0usize;
    let mut context_receipts = Vec::new();
    for run in selected {
        let plan = bind_run(descriptor, run)?;
        let base = run_context(run, descriptor)?;
        let words = run["words"].as_array().ok_or("words")?;
        write_run_prefix(&mut out, run, &plan)?;
        for offset in (0..words.len()).step_by(4) {
            let lanes = (words.len() - offset).min(4);
            let mut original = [0u64; 4];
            for i in 0..lanes {
                original[i] = words[offset + i].as_u64().ok_or("word")?;
            }
            let context = batch_context(base, offset, lanes)?;
            // The sole source of protected arithmetic results: parent callback.
            // No plan.clear_interpret, fixture expected, or fixture mask argument.
            let outputs = evaluate(&original[..lanes], descriptor, context)?;
            validate_batch_outputs(&outputs, lanes, descriptor.word())?;
            for (i, &value) in outputs.iter().enumerate() {
                if offset + i != 0 {
                    out.write_all(b",")?;
                }
                emit!(&mut out, &(value, [] as [u8; 0]))?;
            }
            actual_batches = actual_batches
                .checked_add(1)
                .ok_or("batch count overflow")?;
            actual_words = actual_words
                .checked_add(lanes)
                .ok_or("word count overflow")?;
        }
        out.write_all(b"]}\n")?;
        context_receipts.push(json!({"id":run["id"],"fixture_context":run["context"],"plan":hex(&base.plan),
            "session":hex(&base.session),"operation":hex(&base.operation),"first_tensor_index":base.first_tensor_index,
            "words":words.len(),"batch_offset_rule":"first_tensor_index + original-word offset; each batch at most four lanes"}));
    }
    if actual_words != words_total || actual_batches != batch_counts.iter().sum::<u64>() {
        return Err("incomplete protected callback coverage".into());
    }
    let encodings = write_encodings_footer(&mut out, &fixture)?;
    out.flush()?;
    let receipt = json!({"schema":"pllm.fusefss.native-protected-adapter-receipt.v1","suite":"gelu",
        "execution":"protected-local-reconstruction","fixture_sha256":FIXTURE_HASH,
        "descriptors":1,"runs":8,"gate_evaluations":actual_words,"callback_batches":actual_batches,
        "external_encoding_cases":encodings,"backend":backend,"source_identity":source_identity("protected-local-reconstruction"),
        "output":{"path":"target/fusefss-native-protected-v1/protected.jsonl","bytes":out.bytes,"sha256":hex(&out.hash.finalize())},
        "surfaces":{"mask_and_compiled":"actual native compiler diagnostics under controlled PUBLIC fixture masks; not protected runtime masks",
            "outputs":"parent callback reconstruction using ordinary fresh-random private input masks/keys/triples/nonce per batch/lane",
            "external_encoding":"native test-only checked i128 rational encoding, independent of protected callback"},
        "parent_declared_work_admission":{"batches_by_lanes_1_to_4":batch_counts,"estimated_material_bytes_by_lanes":admission.material_bytes_by_lanes,
            "estimated_cumulative_material_bytes":estimated_material,"existing_cumulative_limit_bytes":admission.cumulative_material_limit,
            "qualification":"parent estimates; not measured material, live owner, traffic or RSS receipt"},
        "contexts":context_receipts,"freshness":"callback obligation; helper does not inspect keys, masks, triples or nonces",
        "independent_checker":"run check_native.py --suite gelu separately; it verifies values/diagnostics, not randomness or runtime mask equality",
        "security_qualified":false,"whole_model_qualified":false});
    let encoded_receipt = serde_json::to_vec_pretty(&receipt)?;
    if encoded_receipt.len() > 64 << 10 {
        return Err("protected receipt byte cap".into());
    }
    let mut receipt_writer = BufWriter::with_capacity(8192, File::create(receipt_path)?);
    receipt_writer.write_all(&encoded_receipt)?;
    receipt_writer.write_all(b"\n")?;
    receipt_writer.flush()?;
    Ok(receipt)
}

#[test]
fn protected_export_opt_in_identity_work_and_output_guards() {
    for value in [None, Some(""), Some("0"), Some("true")] {
        assert!(require_protected_opt_in(value).is_err());
    }
    assert!(require_protected_opt_in(Some("1")).is_ok());
    let digest = "a".repeat(64);
    assert!(backend_identity(&ProtectedBackendIdentity {
        name: "parent-backend.v1",
        sources: &[("crates/backend.rs", &digest)]
    })
    .is_ok());
    for name in ["", "../escape", "name with space"] {
        assert!(backend_identity(&ProtectedBackendIdentity {
            name,
            sources: &[("backend.rs", &digest)]
        })
        .is_err());
    }
    assert!(backend_identity(&ProtectedBackendIdentity {
        name: "backend",
        sources: &[]
    })
    .is_err());
    assert!(backend_identity(&ProtectedBackendIdentity {
        name: "backend",
        sources: &[("../outside", &digest)]
    })
    .is_err());
    assert!(backend_identity(&ProtectedBackendIdentity {
        name: "backend",
        sources: &[("backend.rs", "wrong")]
    })
    .is_err());
    assert!(backend_identity(&ProtectedBackendIdentity {
        name: "backend",
        sources: &[("same", &digest), ("same", &digest)]
    })
    .is_err());
    let admission = ProtectedWorkAdmission {
        material_bytes_by_lanes: [10, 20, 30, 40],
        cumulative_material_limit: 100,
    };
    assert_eq!(preflight_material(&admission, [1, 1, 1, 1]).unwrap(), 100);
    assert!(preflight_material(&admission, [1, 1, 1, 2]).is_err());
    assert!(preflight_material(&admission, [u64::MAX, 0, 0, 0]).is_err());
    assert!(preflight_material(
        &ProtectedWorkAdmission {
            material_bytes_by_lanes: [0; 4],
            cumulative_material_limit: 100
        },
        [0; 4]
    )
    .is_err());
    let word = ArithmeticType {
        bits: 50,
        fractional: 12,
        signed: true,
    };
    assert!(validate_batch_outputs(&[0], 1, word).is_ok());
    assert!(validate_batch_outputs(&[], 1, word).is_err());
    assert!(validate_batch_outputs(&[0, 1], 1, word).is_err());
    assert!(validate_batch_outputs(&[1 << 50], 1, word).is_err());
    assert!(validate_batch_outputs(&[0; 5], 5, word).is_err());
    // No callback is executed and no file/receipt is produced by these guards.
}

#[test]
fn compiler_clear_stream_remains_identical_without_rewriting_historical_artifacts() {
    let (fixture, descriptors) = load_fixture().unwrap();
    let mut out = BoundedWriter {
        inner: std::io::sink(),
        bytes: 0,
        hash: Sha256::new(),
    };
    assert_eq!(
        write_clear_all(&mut out, &fixture, &descriptors).unwrap(),
        (58400, 38)
    );
    assert_eq!(out.bytes, 2494123);
    assert_eq!(
        hex(&out.hash.finalize()),
        "c2abac1dd01ae8a573940009511c3e7cdcdb577a6bc9789f36be6f68855720b6"
    );
}

#[test]
fn protected_fixture_binding_context_offsets_and_separate_diagnostics() {
    let descriptor = Descriptor::gelu().unwrap();
    let n = 1u64 << 50;
    let run = json!({"id":"r0001","context":format!("{SCHEMA}/r0001"),"descriptor_id":GELU_DIGEST_HEX,
        "mask":n-1,"words":[0,1,2,3,4],"masked_words":[n-1,0,1,2,3]});
    let plan = bind_run(&descriptor, &run).unwrap();
    let base = run_context(&run, &descriptor).unwrap();
    assert_eq!(base.plan, descriptor.digest());
    let batch = batch_context(base, 4, 1).unwrap();
    assert_eq!(batch.first_tensor_index, 4);
    assert_eq!(batch.plan, base.plan);
    assert_eq!(batch.session, base.session);
    assert_eq!(batch.operation, base.operation);
    assert!(batch_context(base, 0, 0).is_err());
    assert!(batch_context(base, 0, 5).is_err());
    assert!(batch_context(
        Context {
            first_tensor_index: u64::MAX,
            ..base
        },
        1,
        1
    )
    .is_err());
    assert!(batch_context(
        Context {
            first_tensor_index: u64::MAX,
            ..base
        },
        0,
        2
    )
    .is_err());
    for (field, value) in [
        ("id", json!("r0002")),
        ("id", json!("r9999")),
        ("context", json!("wrong")),
        ("descriptor_id", json!("0".repeat(64))),
        ("mask", json!(n)),
    ] {
        let mut bad = run.clone();
        bad[field] = value;
        assert!(run_context(&bad, &descriptor).is_err());
    }
    let mut next = run.clone();
    next["id"] = json!("r0002");
    next["context"] = json!(format!("{SCHEMA}/r0002"));
    let next_context = run_context(&next, &descriptor).unwrap();
    assert_ne!(next_context.session, base.session);
    assert_ne!(next_context.operation, base.operation);
    let mut changed = run.clone();
    changed["mask"] = json!(1);
    assert_ne!(
        run_context(&changed, &descriptor).unwrap().operation,
        base.operation
    );
    assert!(bind_run(&descriptor, &changed).is_err());
    let mut bad = run.clone();
    bad["descriptor_id"] = json!("0".repeat(64));
    assert!(bind_run(&descriptor, &bad).is_err());
    let mut out = Vec::new();
    write_header(
        &mut out,
        "gelu",
        "protected-local-reconstruction",
        "fresh-random callback outputs; separate public-fixture compiler diagnostics",
    )
    .unwrap();
    write_run_prefix(&mut out, &run, &plan).unwrap();
    let text = String::from_utf8(out).unwrap();
    assert!(text.contains("separate public-fixture compiler diagnostics"));
    assert!(text.contains(&format!("\"mask\":{}", n - 1)));
    assert!(text.ends_with("\"outputs\":["));
    // Prefix only: no fake protected result stream or receipt is generated.
}

#[test]
fn protected_full_work_preflight_without_evaluating_or_exporting() {
    let (fixture, descriptors) = load_fixture().unwrap();
    let descriptor = &descriptors[GELU_DIGEST_HEX];
    let mut runs = 0usize;
    let mut words = 0usize;
    let mut batches = [0u64; 4];
    for run in fixture["runs"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["descriptor_id"].as_str() == Some(GELU_DIGEST_HEX))
    {
        runs += 1;
        let base = run_context(run, descriptor).unwrap();
        let count = run["words"].as_array().unwrap().len();
        words += count;
        for offset in (0..count).step_by(4) {
            let lanes = (count - offset).min(4);
            assert_eq!(
                batch_context(base, offset, lanes)
                    .unwrap()
                    .first_tensor_index,
                offset as u64
            );
            batches[lanes - 1] += 1;
        }
    }
    assert_eq!((runs, words), (8, 43544));
    assert_eq!(batches.iter().sum::<u64>(), 10888);
    // Illustrative refusal only: no crypto/material is issued and no receipt
    // claims these artificial estimates as measured backend resource values.
    let excessive = ProtectedWorkAdmission {
        material_bytes_by_lanes: [10_000_000; 4],
        cumulative_material_limit: 64 << 20,
    };
    assert!(preflight_material(&excessive, batches).is_err());
}

#[test]
fn rational_encoding_checks_ties_sign_range_and_overflow() {
    for (numerator, expected) in [
        (-7, -4),
        (-5, -2),
        (-3, -2),
        (-1, 0),
        (1, 0),
        (3, 2),
        (5, 2),
        (7, 4),
    ] {
        assert_eq!(encode_q12(numerator, 8192).unwrap(), expected);
    }
    assert!(encode_q12(i128::MAX, 1).is_err());
    assert!(encode_q12(1, 0).is_err());
    assert!(encode_q12(1, -1).is_err());
    assert!(encode_q12(1i128 << 49, 4096).is_err());
    assert_eq!(encode_q12(-(1i128 << 49), 4096).unwrap(), -(1i64 << 49));
}
