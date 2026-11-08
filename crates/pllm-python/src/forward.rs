//! Public source-FP32 forward admission and bounded byte-only activation bridge.
use pyo3::exceptions::{PyMemoryError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict, PyInt, PyList, PySet, PyString, PyTuple, PyType};
use pllm_models::{ForwardDType, ForwardPlan, ForwardRequirement, ForwardResourceQuote};
use pllm_compiler::{ForwardMemoryInput, ForwardRuntimeSchedule, ForwardRuntimeStep,
    forward_input_preflight, forward_memory_input_from_plan, forward_memory_input_from_summary,
    lower_forward_runtime_schedule_with_limits, quote_forward_memory_input, validate_forward_composition};
use pllm_types::{canonical_bytes, digest_bytes};
use serde::Serialize;
use std::mem::size_of;
use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Serialize)]
struct Requirements {
    schema_version: &'static str,
    tensors: Vec<ForwardRequirement>,
    weight_bytes: u64,
}

/// Only this factory possesses the original config and admitted reconstruction.
/// No constructor, subclass, pickle, plan/schedule import or adoption entrypoint.
#[pyclass(frozen, name = "ForwardContract", module = "pllm._native")]
struct ForwardContract {
    source_config: Vec<u8>,
    composition: Vec<u8>,
    plan: ForwardPlan,
    schedule: ForwardRuntimeSchedule,
    requirements: Requirements,
    plan_canonical: Vec<u8>,
    schedule_canonical: Vec<u8>,
    requirements_canonical: Vec<u8>,
    metadata: u64,
    memory: u64,
}

fn positive_u64(value: &Bound<'_, PyAny>, name: &str) -> PyResult<u64> {
    use pyo3::types::PyInt;
    if !value.is_exact_instance_of::<PyInt>() {
        return Err(PyTypeError::new_err(format!("{name} must be an exact integer")));
    }
    let n = value.extract::<u64>()?;
    if n == 0 { return Err(PyValueError::new_err(format!("{name} must be positive"))); }
    Ok(n)
}

fn exact_bytes<'a, 'py>(value: &'a Bound<'py, PyAny>, name: &str) -> PyResult<&'a [u8]> {
    let bytes = exact_byte_type(value, name)?;
    forward_input_preflight(bytes).map_err(crate::invalid)?;
    Ok(bytes)
}

fn exact_byte_type<'a, 'py>(value: &'a Bound<'py, PyAny>, name: &str) -> PyResult<&'a [u8]> {
    let value = value.cast_exact::<PyBytes>()
        .map_err(|_| PyTypeError::new_err(format!("{name} must be exact bytes")))?;
    Ok(value.as_bytes())
}

fn add(a: usize, b: usize) -> PyResult<usize> {
    a.checked_add(b).ok_or_else(|| PyValueError::new_err("forward native metadata addition overflow"))
}
fn mul(a: usize, b: usize) -> PyResult<usize> {
    a.checked_mul(b).ok_or_else(|| PyValueError::new_err("forward native metadata multiplication overflow"))
}
fn sum(parts: &[usize]) -> PyResult<usize> {
    parts.iter().try_fold(0, |n, p| add(n, *p))
}
fn bytes_u64(n: usize) -> PyResult<u64> {
    u64::try_from(n).map_err(|_| PyValueError::new_err("forward native metadata address space"))
}
fn model_error(e: pllm_models::ModelError) -> PyErr { PyValueError::new_err(e.to_string()) }
fn tree<T>() -> PyResult<usize> {
    pllm_models::owned_backing::checked_tree_node_bytes::<T>().map_err(|e| crate::invalid(e.to_string()))
}

const QUOTE_KEYS: [&str; 7] = ["sequence_length", "input_bytes", "retained_tensor_bytes",
    "output_bytes", "workspace_bytes", "metadata_bytes", "required_bytes"];

fn quote_canonical_bound() -> PyResult<usize> {
    QUOTE_KEYS.iter().try_fold(2 + QUOTE_KEYS.len() - 1, |n, key| add(n, add(key.len(), 23)?))
}

const ABI_UNINITIALIZED: u8 = 0;
const ABI_SUPPORTED: u8 = 1;
const ABI_UNSUPPORTED: u8 = 2;
const ABI_INVALID_STARTUP: u8 = 3;
static PUBLICATION_ABI: AtomicU8 = AtomicU8::new(ABI_UNINITIALIZED);

fn publication_version_supported(bytes: &[u8], word: usize) -> bool {
    let supported = [b"3.11.".as_slice(), b"3.12.".as_slice(), b"3.13.".as_slice()]
        .iter().any(|prefix| bytes.starts_with(prefix));
    let token = bytes.split(|byte| byte.is_ascii_whitespace()).next().unwrap_or(&[]);
    let patch = token.get(5..).unwrap_or(&[]);
    let patch = patch.strip_suffix(b"+").unwrap_or(patch);
    let stable = !patch.is_empty() && patch.iter().all(u8::is_ascii_digit);
    word == 8 && supported && stable && ![b"PyPy".as_slice(), b"Graal".as_slice(), b"free-threading".as_slice()]
        .iter().any(|marker| bytes.windows(marker.len()).any(|part| part == *marker))
}

/// Pure descriptive classifier; synthetic test inputs cannot initialize ABI state.
fn classify_publication_startup(version: &[u8], word: usize, implementation: Option<&str>, flags: Option<&str>, debug: bool) -> u8 {
    if !publication_version_supported(version, word) { return ABI_UNSUPPORTED; }
    match (implementation, flags) {
        (Some("cpython"), Some("")) if !debug => ABI_SUPPORTED,
        (Some(_), Some(_)) => ABI_UNSUPPORTED,
        _ => ABI_INVALID_STARTUP,
    }
}

fn publication_status_result(status: u8) -> Result<(), &'static str> {
    match status {
        ABI_SUPPORTED => Ok(()),
        ABI_UNSUPPORTED => Err("SourceFloat32 publication requires 64-bit GIL CPython 3.11-3.13 release ABI"),
        ABI_UNINITIALIZED => Err("SourceFloat32 publication ABI was not initialized by the native module"),
        _ => Err("SourceFloat32 publication ABI startup metadata was invalid"),
    }
}

pub(crate) fn require_publication_abi() -> PyResult<()> {
    publication_status_result(PUBLICATION_ABI.load(Ordering::Acquire)).map_err(PyValueError::new_err)
}

fn capture_publication_abi(py: Python<'_>) -> PyResult<u8> {
    let version = unsafe { std::ffi::CStr::from_ptr(pyo3::ffi::Py_GetVersion()) };
    if !publication_version_supported(version.to_bytes(), size_of::<usize>()) { return Ok(ABI_UNSUPPORTED); }
    // Ordinary trusted startup metadata, OUTSIDE compiled-owner admission. Normal
    // imports/attribute access are allowed here, never in the per-compile guard.
    let sys = py.import("sys")?;
    let types = py.import("types")?;
    let implementation = sys.getattr("implementation")?;
    let simple_namespace = types.getattr("SimpleNamespace")?;
    if !implementation.get_type().is(&simple_namespace) { return Ok(ABI_INVALID_STARTUP); }
    let name = implementation.getattr("name")?;
    let flags = sys.getattr("abiflags")?;
    let name = name.cast_exact::<PyString>()?.to_str()?;
    let flags = flags.cast_exact::<PyString>()?.to_str()?;
    Ok(classify_publication_startup(version.to_bytes(), size_of::<usize>(), Some(name), Some(flags),
        sys.hasattr("getobjects")? || sys.hasattr("gettotalrefcount")?))
}

/// One process-ABI Rust fact, not a custody registry or security attestation of
/// pre-import host state. Relies on PyO3's existing interpreter-ID restriction;
/// adds no finalize/reinitialize or subinterpreter support. No Python references
/// are retained. Capture runs outside any Rust lock: concurrent losers discard
/// their local candidate and use the release/acquire-published winner. There is
/// no initializing state, waiting, spinning, reset or caller-supplied record.
pub(crate) fn initialize_publication_abi(py: Python<'_>) {
    if PUBLICATION_ABI.load(Ordering::Acquire) != ABI_UNINITIALIZED { return; }
    let mut candidate = match capture_publication_abi(py) {
        Ok(status) => status,
        Err(error) => { drop(error); ABI_INVALID_STARTUP }
    };
    // Capture failures must never leak a Python error into decoder/module setup.
    if PyErr::occurred(py) {
        drop(PyErr::fetch(py));
        candidate = ABI_INVALID_STARTUP;
    }
    let _ = PUBLICATION_ABI.compare_exchange(ABI_UNINITIALIZED, candidate, Ordering::AcqRel, Ordering::Acquire);
}

/// Bound the measurement operation BEFORE creating any of its Python inputs.
/// This is a finite CPython-layout proof for this exact probe, not a heap tariff.
/// PyObject=2 words; PyVarObject=3; release Unicode <=10 (3.11 largest);
/// GC=2 links. Dict with 7 insertions: 16 one-byte indices, <=10 generic
/// three-word entries; set with <=7 insertions: 32 two-word entries, with its
/// eight-entry inline table still in the object. List append(7): 8 pointers.
/// Known declarations are CPython 3.11--3.13 object.h, unicodeobject.h,
/// dictobject.h/pycore_dict.h, setobject.h, listobject.h and pycore_gc.h.
fn publication_probe_bound() -> PyResult<usize> {
    require_publication_abi()?;
    let w = size_of::<usize>();
    let gc = mul(2, w)?;
    let integer = add(mul(3, w)?, mul(3, size_of::<u32>())?)?; // ceil(64/30) digits
    let unicode = mul(10, w)?;
    let dict = sum(&[mul(6, w)?, gc, mul(4, w)?, 16, mul(10, mul(3, w)?)?])?;
    let set = sum(&[mul(25, w)?, gc, mul(32, mul(2, w)?)?])?;
    let list = sum(&[mul(5, w)?, gc, mul(8, w)?])?;
    let tuple2 = add(mul(5, w)?, gc)?;
    let text = quote_canonical_bound()?;
    // Retained inputs: one dictionary, pair list + 7 tuples, two sets +
    // alternatives tuple, one maximum u64. Keys: dict 7 + fields 6 + complete 7;
    // count 3 full key sets (21) to include all separately-created strings.
    let key_chars = QUOTE_KEYS.iter().try_fold(0usize, |n, key| add(n, add(key.len(), 1)?))?;
    let key_storage = mul(3, add(mul(QUOTE_KEYS.len(), unicode)?, key_chars)?)?;
    let retained = sum(&[dict, list, mul(7, tuple2)?, mul(2, set)?, tuple2, integer, key_storage])?;
    // At most one sampled bytes/unicode operand and its Rust repeat buffer are
    // temporary. Sum both operand alternatives rather than relying on exclusion.
    let operands = sum(&[add(mul(4, w)?, 1)?, unicode, add(text, 1)?,
        unicode, 65, size_of::<String>(), text])?;
    // Each size() chain has <=1 descriptor/bound callable, a call1 tuple, a
    // returned size integer, a flags integer and attribute-name temporaries.
    // Reserve two <=10-word descriptor objects (with GC), two 2-tuples and four
    // result integers, covering slot extraction and next-lookup overlap.
    let calls = sum(&[mul(2, add(mul(10, w)?, gc)?)?, mul(2, tuple2)?, mul(4, integer)?])?;
    let attributes = ["__sizeof__", "__flags__", "__basicsize__"]
        .iter().try_fold(0usize, |n, name| add(n, add(unicode, add(name.len(), 1)?)?))?;
    // Dict resize can keep its previous eight-slot keys table live while the
    // sixteen-slot table is built; list realloc can overlap four old pointers.
    let growth = sum(&[mul(4, w)?, 8, mul(5, mul(3, w)?)?, mul(4, w)?])?;
    // One additional current attribute name may overlap the interned names.
    // The per-call ABI guard is only an acquire load and scalar status decision:
    // no host metadata, key/type-name objects or dictionary control records.
    sum(&[retained, operands, calls, attributes, add(unicode, 17)?, growth])
}

fn prepay_publication_probe(q: &ForwardResourceQuote, source: usize, composition: usize, memory: u64) -> PyResult<usize> {
    let peak = sum(&[q.metadata_peak_bytes, mul(3, add(source, composition)?)?, publication_probe_bound()?])?;
    if bytes_u64(peak)? > memory {
        return Err(PyValueError::new_err("forward publication probe phase exceeds memory_bytes before probing"));
    }
    Ok(peak)
}

/// Narrow publication boundary, not a Python heap estimator. Large inspection
/// methods publish ONLY bytes; frontend DOMs must pre-admit independently. The
/// simultaneous publications are one of each getter, one quote, four digest
/// strings, two scalar getters plus their two compile arguments, and the opaque
/// owner. Unlimited caller-retained
/// repeats/clones are not promised. Measure actual interpreter layouts, never
/// infer dict/list/unicode backing from canonical byte multipliers.
fn publication_owners() -> PyResult<usize> {
    require_publication_abi()?;
    Python::attach(|py| {
        // Call immutable builtin type slots, not monkeypatchable sys.getsizeof.
        // Supported CPython ABI: GC prefix is two uintptr_t links. Non-GC bytes,
        // integers and unicode have no such prefix. __sizeof__ supplies actual
        // interpreter object/table/list capacities (including debug ABI headers).
        let gc_prefix = mul(2, size_of::<usize>())?;
        let size = |object: &Bound<'_, PyAny>| -> PyResult<usize> {
            let kind = object.get_type();
            let base = kind.getattr("__sizeof__")?.call1((object,))?.extract::<usize>()?;
            let flags = kind.getattr("__flags__")?.extract::<u64>()?;
            add(base, if flags & (pyo3::ffi::Py_TPFLAGS_HAVE_GC as u64) != 0 { gc_prefix } else { 0 })
        };
        let bytes_header = size(PyBytes::new(py, &[]).as_any())?;
        let integer = PyInt::new(py, u64::MAX);
        let integer_bytes = size(integer.as_any())?;
        let digest_bytes = size(PyString::new(py, &"f".repeat(64)).as_any())?;
        let dict = PyDict::new(py);
        let pairs = PyList::empty(py);
        let mut keys = 0;
        let mut tuples = 0;
        for key in QUOTE_KEYS {
            let key = PyString::new(py, key);
            keys = add(keys, size(key.as_any())?)?;
            dict.set_item(&key, &integer)?;
            let pair = PyTuple::new(py, [key.as_any(), integer.as_any()])?;
            tuples = add(tuples, size(pair.as_any())?)?;
            pairs.append(pair)?;
        }
        // Small frontend quote parse: result dict, pairs-hook list/tuples,
        // shared keys, seven distinct worst-width integers, ASCII decode string.
        // A second dict covers hook/validation overlap, not a large plan DOM.
        let quote_text = quote_canonical_bound()?;
        let decoded = size(PyString::new(py, &" ".repeat(quote_text)).as_any())?;
        let fields = PySet::new(py, QUOTE_KEYS.iter().copied().filter(|key| *key != "metadata_bytes"))?;
        let complete = PySet::new(py, QUOTE_KEYS)?;
        let alternatives = PyTuple::new(py, [fields.as_any(), complete.as_any()])?;
        let validation = sum(&[size(fields.as_any())?, mul(2, size(complete.as_any())?)?,
            size(alternatives.as_any())?, mul(2, keys)?])?;
        let quote_tree = sum(&[mul(2, size(dict.as_any())?)?, size(pairs.as_any())?, tuples,
            keys, mul(QUOTE_KEYS.len(), integer_bytes)?, decoded, validation])?;
        let key_bytes = QUOTE_KEYS.iter().try_fold(0usize, |n, key| add(n, key.len()))?;
        let native_quote_tree = sum(&[mul(QUOTE_KEYS.len(), tree::<(String, serde_json::Value)>()?)?,
            size_of::<serde_json::Value>(), key_bytes,
            mul(QUOTE_KEYS.len(), size_of::<(&String, &serde_json::Value)>())?,
            add(mul(2, add(QUOTE_KEYS.iter().map(|key| key.len()).max().unwrap_or(0), 2)?)?, 8)?])?;
        let native_quote_text = add(mul(2, quote_text)?, 8)?;
        let owner_type = py.get_type::<ForwardContract>();
        let owner_size = owner_type.getattr("__basicsize__")?.extract::<usize>()?;
        let wrapper = owner_size.checked_sub(size_of::<ForwardContract>())
            .ok_or_else(|| PyValueError::new_err("forward Python owner layout"))?;
        let flags = owner_type.getattr("__flags__")?.extract::<u64>()?;
        let gc = if flags & (pyo3::ffi::Py_TPFLAGS_HAVE_GC as u64) != 0 {
            gc_prefix
        } else { 0 };
        // Two incoming raw arguments, their two getter copies, three canonical
        // getter copies, and one quote: eight actual Python bytes headers.
        sum(&[mul(8, bytes_header)?, quote_text, quote_tree, native_quote_tree,
            native_quote_text, mul(4, digest_bytes)?, mul(4, integer_bytes)?, wrapper, gc])
    })
}

/// All terms are disjoint native owners. The model peak already includes the
/// original source Value, retained plan/requirements, one reconstruction and its
/// model canonical scratch. Its observational subtotals are NOT added again.
struct MetadataLedger {
    bytes: u64,
    plan_canonical: usize,
    schedule_canonical: usize,
    requirements_canonical: usize,
    schedule_owned: usize,
}

/// Composition is parsed sequentially, with source/plan holders retained but no
/// reconstruction live. The compiler enforces this complete raw-parser phase
/// budget (including decoding scratch) before allocating its Value. Model Value
/// lifecycle metadata does NOT stand in for raw-parser scratch.
/// At most one JSON token/member per source byte; concrete BTree/Value/String/Vec
/// records and decoded string growth bound all intermediate/typed owners. Four
/// independent representations cover parser Value, typed pipeline, canonical
/// Value and canonical output/sort/escape buffers. Not an architecture tariff.
fn composition_owners(bytes: usize) -> PyResult<usize> {
    pllm_compiler::forward_composition_memory_bytes(bytes).map_err(crate::invalid)
}

fn metadata_ledger(q: &ForwardResourceQuote, source: usize, composition: usize, memory: u64) -> PyResult<MetadataLedger> {
    // Probe ABI guard is only a cached scalar load. Pay all retained raw/model
    // holders plus the complete finite probe phase BEFORE its first Python input,
    // builtin-layout attribute lookup, descriptor call or returned integer.
    // Public compile's class was registered before entry. No startup metadata is
    // inspected here; missing/unsupported cached ABI status refuses immediately.
    let probe_peak = prepay_publication_probe(q, source, composition, memory)?;
    // Escaped dynamic text expands <=6x. Every stored dimension is >=8 bytes
    // versus <=20 decimal bytes (or ten for "sequence"). Fixed typed schema
    // keys/literals are charged separately per operation/root, never per BERT layer.
    let op_schema = b"{\"id\":,\"operator\":,\"layer\":,\"inputs\":[],\"output\":{\"dtype\":\"float32\",\"shape\":[]},\"attributes\":{}}".len();
    let root_schema = b"{\"schema_version\":,\"model_family\":,\"adapter\":,\"config_digest\":,\"max_sequence\":,\"numeric\":,\"output_contract\":,\"inputs\":{},\"operations\":[],\"outputs\":{}}".len();
    let plan_canonical = sum(&[mul(6, q.retained_plan_bytes)?, mul(q.operation_count, op_schema)?, root_schema])?;
    let step_schema = b"{\"executor\":\"client_local_source_f32_v1\",\"operation_id\":\"\"},".len();
    let schedule_root = b"{\"schema_version\":\"pllm.forward_runtime_schedule.v1\",\"plan_digest\":\"\",\"composition_digest\":\"\",\"numeric_digest\":\"\",\"max_sequence\":,\"complete\":true,\"protected_execution\":false,\"steps\":[]}".len();
    let schedule_canonical = sum(&[mul(6, q.operation_id_bytes)?, mul(q.operation_count, step_schema)?, schedule_root, 3 * 64, 20])?;
    // Admitted operations own at most two checkpoint records; shape vector
    // backing/names already included in the model's requirements subtotal.
    let records = mul(2, q.operation_count)?;
    let req_schema = b"{\"dtype\":\"float32\",\"name\":\"\",\"shape\":[]},".len();
    let requirements_canonical = sum(&[mul(6, q.retained_requirements_bytes)?, mul(records, req_schema)?,
        b"{\"schema_version\":\"pllm.forward_requirements.v1\",\"tensors\":[],\"weight_bytes\":}".len(), 20])?;
    // Compiler topological index contains borrowed values. Each input consumes
    // at least one tensor header in retained_plan_bytes; no fixed input count.
    let values = add(q.operation_count, q.retained_plan_bytes / size_of::<pllm_models::ForwardTensorSpec>())?;
    let schedule_owned = sum(&[size_of::<ForwardRuntimeSchedule>(),
        mul(q.operation_count, size_of::<ForwardRuntimeStep>())?, q.operation_id_bytes,
        mul(q.operation_count, "client_local_source_f32_v1".len())?,
        pllm_compiler::FORWARD_RUNTIME_SCHEDULE_SCHEMA_VERSION.len(), 3 * 64])?;
    let compiler = add(schedule_owned, mul(values, tree::<(&str, &pllm_models::ForwardTensorSpec)>()?)?)?;
    // Schedule and serialized requirements have independent JSON trees. Exact
    // typed schema member counts bound object/array slots, key text and sorting.
    let json_slots = add(mul(3, q.operation_count)?, add(mul(5, records)?, 16)?)?;
    let native_trees = sum(&[mul(json_slots, tree::<(String, serde_json::Value)>()?)?,
        mul(json_slots, size_of::<serde_json::Value>())?,
        mul(2, q.operation_id_bytes)?, mul(2, q.retained_requirements_bytes)?,
        mul(q.operation_count, step_schema)?, mul(records, req_schema)?, schedule_root])?;
    // Each cached canonical buffer may grow to <2x its encoded bound (plus Vec's
    // initial small allocation). Remaining copies price Python immutable result
    // publication and serialization/hash overlap. Model tree scratch is reused
    // sequentially; retained native bytes remain disjoint throughout validation.
    let canonical = add(mul(4, sum(&[plan_canonical, schedule_canonical, requirements_canonical])?)?, 3 * 8)?;
    // Inline model plan/requirement/schedule headers transfer into the native
    // owner; their original ledger slots must not be added a second time.
    let native_header = size_of::<ForwardContract>().checked_sub(sum(&[size_of::<ForwardPlan>(),
        size_of::<Vec<ForwardRequirement>>(), size_of::<ForwardRuntimeSchedule>()])?)
        .ok_or_else(|| PyValueError::new_err("forward native header accounting overflow"))?;
    let raw = mul(3, add(source, composition)?)?;
    let lifecycle = sum(&[q.metadata_peak_bytes, raw,
        composition_owners(composition)?, compiler, native_trees, canonical,
        native_header, size_of::<MetadataLedger>(), size_of::<ForwardResourceQuote>(),
        mul(2, size_of::<ForwardMemoryInput>())?, mul(2, size_of::<pllm_compiler::ForwardMemoryQuote>())?,
        publication_owners()?])?;
    // Source parsing is earlier than preflight; the same deterministic structural
    // budget is enforced there. This separate phase includes source Value plus
    // lexical/build/escaped scratch and raw holders. Composition parsing occurs
    // with model holders live: model peak + composition_owners already dominates
    // that disjoint phase in lifecycle. Caches are not yet constructed then.
    let source_parse_peak = add(raw, composition_owners(source)?)?;
    let total = lifecycle.max(source_parse_peak).max(probe_peak);
    Ok(MetadataLedger { bytes: bytes_u64(total)?, plan_canonical, schedule_canonical, requirements_canonical, schedule_owned })
}

fn requirement_weight_bytes(tensors: &[ForwardRequirement]) -> PyResult<u64> {
    let mut weight_bytes = 0_u64;
    for tensor in tensors {
        if tensor.dtype != ForwardDType::Float32 {
            return Err(PyValueError::new_err("forward source weights must be float32"));
        }
        let bytes = tensor.shape.iter().try_fold(4_u64, |n, d| n.checked_mul(*d))
            .ok_or_else(|| PyValueError::new_err("forward weight byte overflow"))?;
        weight_bytes = weight_bytes.checked_add(bytes)
            .ok_or_else(|| PyValueError::new_err("forward total weight byte overflow"))?;
    }
    Ok(weight_bytes)
}

fn reconcile_input(expected: &ForwardMemoryInput, actual: &ForwardMemoryInput) -> PyResult<()> {
    if expected != actual { return Err(PyValueError::new_err("forward source summary differs from expanded typed graph")); }
    Ok(())
}

fn compile_source(source: &[u8], composition: &[u8], max_sequence: u64, memory: u64) -> PyResult<ForwardContract> {
    compile_source_with(source, composition, max_sequence, memory, |preflight| preflight.construct())
}

fn universal_floor(max_sequence: u64, memory: u64) -> PyResult<pllm_compiler::ForwardMemoryQuote> {
    let minimum = quote_forward_memory_input(max_sequence, &ForwardMemoryInput::default(), 0).map_err(crate::invalid)?;
    if memory < minimum.required_bytes { return Err(PyValueError::new_err("forward memory_bytes below universal backend minimum")); }
    Ok(minimum)
}

fn compile_source_with<F>(source: &[u8], composition: &[u8], max_sequence: u64, memory: u64, construct: F) -> PyResult<ForwardContract>
where F: for<'a> FnOnce(pllm_models::ForwardPreflight<'a>) -> Result<ForwardPlan, pllm_models::ModelError> {
    #[cfg(test)]
    { compile_source_instrumented(source, composition, max_sequence, memory, construct, |_| {}) }
    #[cfg(not(test))]
    { compile_source_instrumented(source, composition, max_sequence, memory, construct) }
}

fn compile_source_instrumented<F>(source: &[u8], composition: &[u8], max_sequence: u64, memory: u64, construct: F,
    #[cfg(test)] census_fault: impl FnOnce(&mut ForwardMemoryInput),
) -> PyResult<ForwardContract>
where F: for<'a> FnOnce(pllm_models::ForwardPreflight<'a>) -> Result<ForwardPlan, pllm_models::ModelError> {
    // Allocation-free universal numerical floor; reject tiny caps before parsing
    // even malformed source/composition. No source-specific geometry here.
    let minimum = universal_floor(max_sequence, memory)?;
    let frontend = bytes_u64(sum(&[mul(3, add(source.len(), composition.len())?)?, composition_owners(composition.len())?])?)?;
    let remaining = memory.checked_sub(minimum.required_bytes).and_then(|n| n.checked_sub(frontend))
        .ok_or_else(|| PyValueError::new_err("forward frontend owners exceed memory_bytes before source parsing"))?;
    let mut limits = pllm_models::forward_limits_for_memory(remaining).map_err(model_error)?;
    // Explicit custom caller metadata policy, as in model custom-policy APIs;
    // default source/work ceilings stay unchanged. Never silently reapply 64MiB.
    limits.metadata.max_factory_bytes = usize::try_from(remaining)
        .map_err(|_| PyValueError::new_err("forward caller memory exceeds address space"))?;
    let mut parsing = limits;
    parsing.metadata.max_factory_bytes = parsing.metadata.max_factory_bytes.min(composition_owners(source.len())?);
    let document = pllm_models::parse_forward_source_json(source, parsing).map_err(model_error)?;
    let preflight = pllm_models::preflight_forward_value(&document, max_sequence, limits).map_err(model_error)?;
    let resource = *preflight.quote();
    let expected = forward_memory_input_from_summary(&resource.numerical).map_err(crate::invalid)?;
    let ledger = metadata_ledger(&resource, source.len(), composition.len(), memory)?;
    let prequote = quote_forward_memory_input(max_sequence, &expected, ledger.bytes).map_err(crate::invalid)?;
    if prequote.required_bytes > memory { return Err(PyValueError::new_err("forward complete storage exceeds memory_bytes before graph construction")); }
    // Bounded composition parser runs with original Value held. Its complete raw
    // parse phase, NOT just Value backing, is reserved disjointly above.
    validate_forward_composition(composition, limits).map_err(crate::invalid)?;
    let plan = construct(preflight).map_err(model_error)?;
    if plan.operations.len() != resource.operation_count || plan.operations.iter().try_fold(0usize, |n, op| add(n, op.id.len()))? != resource.operation_id_bytes {
        return Err(PyValueError::new_err("forward expanded operation resources differ from preflight"));
    }
    plan.checked_owned_backing_bytes(resource.retained_plan_bytes).map_err(model_error)?;
    // These calls retain their original reconstruction/validation checks. They
    // run sequentially and reuse the model's one-reconstruction reservation.
    let schedule = lower_forward_runtime_schedule_with_limits(&plan, composition, limits).map_err(crate::invalid)?;
    // Digests come only from the fixed-size compiler SHA256 encoders, not a
    // deserialized schedule. Other retained capacities are measured explicitly.
    let actual_schedule = schedule.steps.iter().try_fold(sum(&[size_of::<ForwardRuntimeSchedule>(),
        schedule.schema_version.capacity(), 3 * 64,
        mul(schedule.steps.capacity(), size_of::<ForwardRuntimeStep>())?])?, |n, step| {
        add(n, add(step.operation_id.capacity(), step.executor.capacity())?)
    })?;
    if actual_schedule > ledger.schedule_owned { return Err(PyValueError::new_err("forward schedule exceeds native metadata reservation")); }
    let tensors = pllm_models::forward_checkpoint_requirements_with_limits(&plan, limits).map_err(model_error)?;
    let actual_requirements = tensors.iter().try_fold(add(size_of::<Vec<ForwardRequirement>>(), mul(tensors.capacity(), size_of::<ForwardRequirement>())?)?, |n, tensor| {
        add(n, add(tensor.name.capacity(), mul(tensor.shape.capacity(), size_of::<u64>())?)?)
    })?;
    if actual_requirements > resource.retained_requirements_bytes { return Err(PyValueError::new_err("forward requirements exceed source metadata reservation")); }
    let weight_bytes = requirement_weight_bytes(&tensors)?;
    #[allow(unused_mut)]
    let mut actual = forward_memory_input_from_plan(&plan, max_sequence, weight_bytes).map_err(crate::invalid)?;
    #[cfg(test)]
    census_fault(&mut actual);
    reconcile_input(&expected, &actual)?;
    let postquote = quote_forward_memory_input(max_sequence, &actual, ledger.bytes).map_err(crate::invalid)?;
    if postquote != prequote { return Err(PyValueError::new_err("forward expanded quote differs from pre-expansion admission")); }
    let requirements = Requirements { schema_version: "pllm.forward_requirements.v1", tensors, weight_bytes };
    let plan_canonical = canonical_bytes(&plan);
    let schedule_canonical = canonical_bytes(&schedule);
    let requirements_canonical = canonical_bytes(&requirements);
    for (bytes, bound) in [(&plan_canonical, ledger.plan_canonical), (&schedule_canonical, ledger.schedule_canonical), (&requirements_canonical, ledger.requirements_canonical)] {
        if bytes.len() > bound || bytes.capacity() > add(mul(2, bound)?, 8)? {
            return Err(PyValueError::new_err("forward native canonical capacity differs from schema reservation"));
        }
    }
    Ok(ForwardContract {
        source_config: crate::bounded_copy(source, 1 << 20, "forward source config")?,
        composition: crate::bounded_copy(composition, 1 << 20, "forward composition")?,
        plan, schedule, memory, requirements, metadata: ledger.bytes,
        plan_canonical, schedule_canonical, requirements_canonical,
    })
}

#[pymethods]
impl ForwardContract {
    #[classmethod]
    #[pyo3(signature = (source_config, composition, *, max_sequence, memory_bytes))]
    fn compile(
        cls: &Bound<'_, PyType>, source_config: &Bound<'_, PyAny>, composition: &Bound<'_, PyAny>,
        max_sequence: &Bound<'_, PyAny>, memory_bytes: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        if !cls.is(&cls.py().get_type::<Self>()) {
            return Err(PyTypeError::new_err("ForwardContract requires exact native class"));
        }
        let max_sequence = positive_u64(max_sequence, "max_sequence")?;
        let memory = positive_u64(memory_bytes, "memory_bytes")?;
        let source = exact_byte_type(source_config, "source_config")?;
        let composition = exact_byte_type(composition, "composition")?;
        universal_floor(max_sequence, memory)?;
        forward_input_preflight(source).map_err(crate::invalid)?;
        forward_input_preflight(composition).map_err(crate::invalid)?;
        compile_source(source, composition, max_sequence, memory)
    }
    fn source_config_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> { PyBytes::new(py, &self.source_config) }
    fn composition_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> { PyBytes::new(py, &self.composition) }
    fn plan_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> { PyBytes::new(py, &self.plan_canonical) }
    fn schedule_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> { PyBytes::new(py, &self.schedule_canonical) }
    fn requirements_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> { PyBytes::new(py, &self.requirements_canonical) }
    fn plan_digest(&self) -> String { self.schedule.plan_digest.to_string() }
    fn schedule_digest(&self) -> String { digest_bytes(pllm_compiler::FORWARD_RUNTIME_SCHEDULE_SCHEMA_VERSION, &self.schedule_canonical).to_string() }
    fn composition_digest(&self) -> String { self.schedule.composition_digest.to_string() }
    fn numeric_digest(&self) -> String { self.schedule.numeric_digest.to_string() }
    fn max_sequence(&self) -> u64 { self.plan.max_sequence }
    fn memory_bytes(&self) -> u64 { self.memory }
    fn quote<'py>(&self, py: Python<'py>, sequence_length: &Bound<'_, PyAny>) -> PyResult<Bound<'py, PyBytes>> {
        let sequence = positive_u64(sequence_length, "sequence_length")?;
        let input = forward_memory_input_from_plan(&self.plan, sequence, self.requirements.weight_bytes).map_err(crate::invalid)?;
        let quote = quote_forward_memory_input(sequence, &input, self.metadata).map_err(crate::invalid)?;
        if quote.required_bytes > self.memory { return Err(PyValueError::new_err("forward complete storage exceeds memory_bytes")); }
        Ok(PyBytes::new(py, &canonical_bytes(&quote)))
    }
    fn __reduce__(&self) -> PyResult<()> { Err(PyTypeError::new_err("ForwardContract cannot be pickled")) }
    fn __reduce_ex__(&self, _protocol: &Bound<'_, PyAny>) -> PyResult<()> { Err(PyTypeError::new_err("ForwardContract cannot be pickled")) }
}

#[pyfunction]
#[pyo3(signature = (source_config, *, max_sequence))]
fn lower_forward_model<'py>(py: Python<'py>, source_config: &Bound<'_, PyAny>, max_sequence: &Bound<'_, PyAny>) -> PyResult<Bound<'py, PyBytes>> {
    let max_sequence = positive_u64(max_sequence, "max_sequence")?;
    let source = exact_bytes(source_config, "source_config")?;
    let plan = pllm_models::lower_forward_model(source, max_sequence)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyBytes::new(py, &canonical_bytes(&plan)))
}

const ACTIVATION_MAX_BYTES: usize = 1 << 20;

/// Largest simultaneously live numerical backing: borrowed input, decoded input,
/// native output and serialized Python output. No float-list or FP64 bridge.
const ACTIVATION_COPY_FACTOR: usize = 4;

#[pyfunction]
fn source_float32_activation<'py>(
    py: Python<'py>,
    kind: &str,
    input: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyBytes>> {
    use pllm_core::fp32_activations::{gelu_erf_into, tanh_into};
    let kernel = match kind {
        "gelu_erf" => gelu_erf_into,
        "tanh" => tanh_into,
        _ => return Err(PyValueError::new_err("unsupported source-FP32 activation")),
    };
    let input = input.cast_exact::<PyBytes>()
        .map_err(|_| PyTypeError::new_err("input must be exact bytes"))?;
    let bytes = input.as_bytes();
    if bytes.len() > ACTIVATION_MAX_BYTES || bytes.len() % 4 != 0 {
        return Err(PyValueError::new_err("activation requires at most 1 MiB aligned F32 bytes"));
    }
    let _peak = bytes.len().checked_mul(ACTIVATION_COPY_FACTOR)
        .ok_or_else(|| PyValueError::new_err("activation backing overflow"))?;
    let count = bytes.len() / 4;
    let mut decoded = Vec::<f32>::new();
    let mut output = Vec::<f32>::new();
    decoded.try_reserve_exact(count)
        .map_err(|_| PyMemoryError::new_err("activation input allocation failed"))?;
    output.try_reserve_exact(count)
        .map_err(|_| PyMemoryError::new_err("activation output allocation failed"))?;
    decoded.extend(bytes.chunks_exact(4).map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]])));
    output.resize(count, 0.0);
    kernel(&decoded, &mut output)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    PyBytes::new_with(py, bytes.len(), |serialized| {
        for (target, value) in serialized.chunks_exact_mut(4).zip(&output) {
            target.copy_from_slice(&value.to_le_bytes());
        }
        Ok(())
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ForwardContract>()?;
    module.add_function(wrap_pyfunction!(lower_forward_model, module)?)?;
    module.add_function(wrap_pyfunction!(source_float32_activation, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::cell::Cell;

    fn setup_publication_abi() {
        Python::initialize();
        // Embedded lib tests do not run the extension module initializer. Use
        // that very same production initializer, without any global Rust lock.
        Python::attach(initialize_publication_abi);
    }

    fn source() -> Vec<u8> {
        canonical_bytes(&json!({"model_type":"bert","architectures":["BertModel"],
            "vocab_size":17,"hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,
            "intermediate_size":12,"max_position_embeddings":16,"type_vocab_size":2,
            "hidden_act":"gelu","layer_norm_eps":"1e-12","position_embedding_type":"absolute",
            "torch_dtype":"float32"}))
    }
    fn composition() -> Vec<u8> {
        canonical_bytes(&json!({"model":{"source":"fixture"},"components":{
            "kernels":{"component":"pllm/cpu","params":{"threads":1}},
            "linear":{"component":"pllm/cleartext-linear","params":{}},
            "numeric":{"component":"pllm/source-float32/v1","params":{}},
            "topology":{"component":"pllm/client-only/v1","params":{}}}}))
    }
    #[test]
    fn complete_exact_cap_and_one_short_gate_before_consuming_preflight() {
        setup_publication_abi();
        let source = source();
        let composition = composition();
        let generous = compile_source(&source, &composition, 8, 1 << 30).unwrap();
        let input = forward_memory_input_from_plan(&generous.plan, 8, generous.requirements.weight_bytes).unwrap();
        let quote = quote_forward_memory_input(8, &input, generous.metadata).unwrap();
        assert!(quote.metadata_bytes > 0);
        let calls = Cell::new(0);
        let short = compile_source_with(&source, &composition, 8, quote.required_bytes - 1, |preflight| {
            calls.set(calls.get() + 1);
            preflight.construct()
        });
        assert!(short.is_err());
        assert_eq!(calls.get(), 0);
        let exact = compile_source_with(&source, &composition, 8, quote.required_bytes, |preflight| {
            calls.set(calls.get() + 1);
            preflight.construct()
        }).unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!(exact.metadata, generous.metadata);
        let more = compile_source(&source, &composition, 8, quote.required_bytes + 1).unwrap();
        assert_eq!(more.metadata, exact.metadata);
        for sequence in [1, 8] {
            let costs = forward_memory_input_from_plan(&exact.plan, sequence, exact.requirements.weight_bytes).unwrap();
            let quote = quote_forward_memory_input(sequence, &costs, exact.metadata).unwrap();
            assert_eq!(quote.metadata_bytes, generous.metadata);
        }
    }
    #[test]
    fn tiny_cap_refuses_before_source_parser_and_large_expansion_before_construct() {
        setup_publication_abi();
        let calls = Cell::new(0);
        let result = compile_source_with(b"not JSON", b"not JSON", 8, 1, |preflight| {
            calls.set(calls.get() + 1);
            preflight.construct()
        });
        let error = match result { Err(error) => error, Ok(_) => panic!("tiny cap admitted") };
        assert!(error.to_string().contains("universal backend minimum"));
        assert_eq!(calls.get(), 0);
        let mut config: serde_json::Value = serde_json::from_slice(&source()).unwrap();
        // Valid source geometry, deliberately unaffordable metadata expansion.
        config["num_hidden_layers"] = json!(100_000);
        let result = compile_source_with(&canonical_bytes(&config), &composition(), 8, 64 << 20, |preflight| {
            calls.set(calls.get() + 1);
            preflight.construct()
        });
        assert!(result.is_err());
        assert_eq!(calls.get(), 0);
    }
    #[test]
    fn descriptive_summary_mismatch_fails_closed_without_an_execution_owner() {
        setup_publication_abi();
        let expected = ForwardMemoryInput { weight_bytes: 4, ..ForwardMemoryInput::default() };
        assert!(reconcile_input(&expected, &ForwardMemoryInput::default()).is_err());
        assert!(reconcile_input(&expected, &expected).is_ok());
    }

    #[test]
    fn genuine_constructed_lifecycle_fault_is_refused_at_census_reconciliation() {
        setup_publication_abi();
        let constructed = Cell::new(0);
        let censused = Cell::new(0);
        let result = compile_source_instrumented(&source(), &composition(), 8, 1 << 30, |preflight| {
            let plan = preflight.construct()?;
            constructed.set(constructed.get() + 1);
            Ok(plan)
        }, |actual| {
            // Only cfg(test), after real graph/requirements/schedule derivation.
            // Never supplies a model, source owner, plan or opaque contract.
            censused.set(censused.get() + 1);
            actual.retained_tensor_bytes = actual.retained_tensor_bytes.checked_add(4).unwrap();
        });
        let error = match result { Err(error) => error, Ok(_) => panic!("fault published an opaque owner") };
        assert_eq!(constructed.get(), 1);
        assert_eq!(censused.get(), 1);
        assert!(error.to_string().contains("source summary differs from expanded typed graph"));
    }

    #[test]
    fn publication_abi_guard_refuses_unpriced_layout_families() {
        for version in [b"3.11.9 (release)".as_slice(), b"3.12.8 (release)", b"3.13.1 (release)"] {
            assert!(publication_version_supported(version, 8));
            assert_eq!(classify_publication_startup(version, 8, Some("cpython"), Some(""), false), ABI_SUPPORTED);
        }
        for version in [b"3.10.9".as_slice(), b"3.14.0", b"3.13.0rc1", b"3.11.9 [PyPy]", b"3.11.9 [GraalPy]",
            b"3.13.1 experimental free-threading build"] {
            assert!(!publication_version_supported(version, 8));
            assert_eq!(classify_publication_startup(version, 8, Some("cpython"), Some(""), false), ABI_UNSUPPORTED);
        }
        let version = b"3.13.1 (release)";
        assert!(!publication_version_supported(version, 4));
        assert_eq!(classify_publication_startup(version, 4, Some("cpython"), Some(""), false), ABI_UNSUPPORTED);
        for (name, flags, debug) in [(Some("pypy"), Some(""), false), (Some("cpython"), Some("d"), false),
            (Some("cpython"), Some("t"), false), (Some("cpython"), Some(""), true)] {
            assert_eq!(classify_publication_startup(version, 8, name, flags, debug), ABI_UNSUPPORTED);
        }
        for (name, flags) in [(None, Some("")), (Some("cpython"), None), (None, None)] {
            assert_eq!(classify_publication_startup(version, 8, name, flags, false), ABI_INVALID_STARTUP);
        }
    }

    #[test]
    fn publication_status_decision_refuses_uninitialized_and_failed_capture() {
        // Local scalar decisions only: never mutate or reset production state.
        assert_eq!(publication_status_result(ABI_SUPPORTED), Ok(()));
        assert!(publication_status_result(ABI_UNINITIALIZED).unwrap_err().contains("not initialized"));
        assert!(publication_status_result(ABI_UNSUPPORTED).unwrap_err().contains("release ABI"));
        assert!(publication_status_result(ABI_INVALID_STARTUP).unwrap_err().contains("startup metadata"));
        assert!(publication_status_result(u8::MAX).is_err());
    }

    #[test]
    fn finite_probe_phase_is_prepaid_before_python_publication_measurements() {
        setup_publication_abi();
        let raw = source();
        let pipeline = composition();
        let limits = pllm_models::default_forward_limits().unwrap();
        let document = pllm_models::parse_forward_source_json(&raw, limits).unwrap();
        let preflight = pllm_models::preflight_forward_value(&document, 8, limits).unwrap();
        let q = preflight.quote();
        let bootstrap = publication_probe_bound().unwrap();
        // Actual counted phase bound, not an assertion of unspecified spare
        // capacity: finite CPython object/table/growth/temporary formula above.
        assert!(bytes_u64(bootstrap).unwrap() < universal_floor(8, 8 << 20).unwrap().required_bytes);
        let peak = sum(&[q.metadata_peak_bytes, mul(3, add(raw.len(), pipeline.len()).unwrap()).unwrap(), bootstrap]).unwrap();
        assert!(prepay_publication_probe(q, raw.len(), pipeline.len(), bytes_u64(peak - 1).unwrap()).is_err());
        assert_eq!(prepay_publication_probe(q, raw.len(), pipeline.len(), bytes_u64(peak).unwrap()).unwrap(), peak);
        require_publication_abi().unwrap();
        let ledger = metadata_ledger(q, raw.len(), pipeline.len(), 1 << 30).unwrap();
        assert!(ledger.bytes >= bytes_u64(peak).unwrap());
        assert_eq!(ledger.bytes, metadata_ledger(q, raw.len(), pipeline.len(), 2 << 30).unwrap().bytes);
    }

    #[test]
    fn custom_policy_above_default_model_peak_survives_complete_native_lifecycle() {
        setup_publication_abi();
        let base = pllm_models::default_forward_limits().unwrap();
        let mut custom = base;
        custom.metadata.max_factory_bytes = base.metadata.max_factory_bytes.checked_mul(4).unwrap();
        let mut config: serde_json::Value = serde_json::from_slice(&source()).unwrap();
        // Same small geometry as model custom-policy regression, no coefficient
        // arrays. Find first layer count crossing default peak by scalar quotes
        // only, then construct that single case. At most 256 scalar preflights;
        // unlike fixed 641MiB, this adapts to real record layout and is minimal.
        config["architectures"] = json!(["BertForSequenceClassification"]);
        config["num_labels"] = json!(3);
        config["intermediate_size"] = json!(13);
        config["max_position_embeddings"] = json!(8);
        let mut selected = None;
        for layers in 1..=256 {
            config["num_hidden_layers"] = json!(layers);
            let resource = *pllm_models::preflight_forward_value(&config, 1, custom).unwrap().quote();
            if resource.metadata_peak_bytes > base.metadata.max_factory_bytes {
                selected = Some((layers, resource));
                break;
            }
        }
        let (layers, resource) = selected.expect("small geometry crosses default metadata peak");
        assert!(resource.metadata_peak_bytes > 64 << 20);
        assert!(pllm_models::preflight_forward_value(&config, 1, base).is_err());
        let raw = canonical_bytes(&config);
        let pipeline = composition();
        let budget = bytes_u64(base.metadata.max_factory_bytes.checked_mul(16).unwrap()).unwrap();
        let generous = compile_source(&raw, &pipeline, 1, budget).unwrap();
        assert_eq!(generous.plan.operations.len(), resource.operation_count);
        assert_eq!(generous.requirements.tensors.len(), 16 * layers + 9);
        let costs = forward_memory_input_from_plan(&generous.plan, 1, generous.requirements.weight_bytes).unwrap();
        let expected = quote_forward_memory_input(1, &costs, generous.metadata).unwrap();
        let constructed = Cell::new(0);
        let short = compile_source_with(&raw, &pipeline, 1, expected.required_bytes - 1, |preflight| {
            constructed.set(constructed.get() + 1); preflight.construct()
        });
        assert!(short.is_err());
        assert_eq!(constructed.get(), 0);
        let exact = compile_source_with(&raw, &pipeline, 1, expected.required_bytes, |preflight| {
            constructed.set(constructed.get() + 1); preflight.construct()
        }).unwrap();
        assert_eq!(constructed.get(), 1);
        assert_eq!(exact.metadata, generous.metadata);
        Python::attach(|py| {
            // Genuine private factory result, not adoption/deserialization.
            let owner = Bound::new(py, exact).unwrap();
            let owner = owner.borrow();
            let plan = owner.plan_bytes(py);
            let schedule = owner.schedule_bytes(py);
            let requirements = owner.requirements_bytes(py);
            assert_eq!(plan.as_bytes(), generous.plan_canonical.as_slice());
            assert_eq!(schedule.as_bytes(), generous.schedule_canonical.as_slice());
            assert_eq!(requirements.as_bytes(), generous.requirements_canonical.as_slice());
            let sequence = PyInt::new(py, 1);
            let quote = owner.quote(py, sequence.as_any()).unwrap();
            let quote: serde_json::Value = serde_json::from_slice(quote.as_bytes()).unwrap();
            assert_eq!(quote["metadata_bytes"], json!(generous.metadata));
            assert_eq!(quote["required_bytes"], json!(expected.required_bytes));
        });
    }
}
