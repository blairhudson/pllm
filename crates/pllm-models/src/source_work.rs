//! Opt-in logical work cover for the registered, closed Value source factories.
//!
//! This is NOT a CPU-time, instruction, allocator, stack/RSS, source-authentication
//! or Budget certificate. Units cover structural visits, potential byte traffic,
//! and explicitly named opaque helper invocations. The pinned default/std serde
//! closure is required; private serde/sort/hash/f64-parser/formatter algorithms and
//! their storage are unqualified. Descriptor limits alone are not a CPU budget.
//!
//! Prepay `bootstrap().quote_work` BEFORE constructing the fixed frame array or
//! calling `quote`. The first successful summary prepays BOTH whole quote passes
//! and one factory/cleanup/census. Do not refund slack or reset a global ledger.
//! Drop the first captured lowerer AND frame pool before moving its Value owner.
//! Requote inside the guarded construction scope; consume that captured lowerer.
//! Query and post-census frame pools are disjoint phases, not simultaneous arrays.
//! Inactive frames can retain safe earlier borrows until overwrite/drop.

use crate::owned_backing::bounded as census;
use crate::{
    ConfigLowererLimits, ConfigLowererQuote, DecoderExpansionBudget, DecoderWorkload, ModelError,
};
use pllm_types::bounded_canonical::borrowed::{self as json, Event, NativeFrame, Traversal};
use serde_json::Value;
use std::{fmt, mem::size_of};

/// Legacy source metadata depth 0..=16 corresponds to native JSON depth 1..=17.
pub const SOURCE_FRAME_SLOTS: usize = 17;

#[derive(Clone, Copy, Debug)]
pub struct SourceInputLimits {
    pub max_nodes: usize,
    /// Native traversal payload, including Number Display, not lexical JSON size.
    pub max_payload_bytes: usize,
    pub max_string_bytes: usize,
    pub max_key_bytes: usize,
    /// One-based, at most SOURCE_FRAME_SLOTS.
    pub max_depth: usize,
    pub max_traversal_work: usize,
    pub max_total_work: u64,
    pub max_workspace_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceWorkBootstrap {
    pub frame_slots: usize,
    /// Max of known query/census logical phases, including entire fixed pool.
    /// Caller Context, owner/counter/backing and factory storage remain separate.
    pub bootstrap_bytes: usize,
    pub quote_work: u64,
    pub requote_work: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceWorkSlices {
    pub quote: u64,
    pub requote: u64,
    pub factory: u64,
    pub cleanup: u64,
    /// Raw census C plus prepaid structural-header work 10*C + 16. The
    /// post-census cursor still receives precisely the original raw C cap.
    pub census: u64,
}

impl SourceWorkSlices {
    pub fn total_work(self) -> Result<u64, SourceWorkError> {
        sum(&[
            self.quote,
            self.requote,
            self.factory,
            self.cleanup,
            self.census,
        ])
    }
}

/// These are bounded calls/inputs, NOT a priced implementation of each helper.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OpaqueHelpers {
    pub typed_deserializations: u64,
    pub canonical_serializations: u64,
    pub sort_calls: u64,
    pub decimal_parse_calls: u64,
    pub decimal_parse_input_bytes: u64,
    pub number_format_calls: u64,
    pub hash_input_bytes: u64,
    pub typed_input_nodes: u64,
    pub typed_input_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct SourceWorkQuote {
    resources: SourceWorkBootstrap,
    slices: SourceWorkSlices,
    total: u64,
    input: json::Usage,
    helpers: OpaqueHelpers,
    post: census::Limits,
    post_resources: census::Resources,
}

impl SourceWorkQuote {
    pub fn resources(self) -> SourceWorkBootstrap {
        self.resources
    }
    pub fn slices(self) -> SourceWorkSlices {
        self.slices
    }
    pub fn total_work(self) -> u64 {
        self.total
    }
    pub fn input_usage(self) -> json::Usage {
        self.input
    }
    pub fn opaque_helpers(self) -> OpaqueHelpers {
        self.helpers
    }
    pub fn census_work(self) -> u64 {
        self.slices.census
    }
    pub fn post_census_limits(self) -> census::Limits {
        self.post
    }
    pub fn post_census_resources(self) -> census::Resources {
        self.post_resources
    }
}

#[derive(Debug)]
pub enum SourceWorkError {
    Overflow,
    InputLimit,
    Workspace,
    WorkLimit,
    Json(json::Error),
    Census(census::Error),
    Source(ModelError),
}

impl fmt::Display for SourceWorkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Overflow => "source work arithmetic overflow",
            Self::InputLimit => "source work input envelope invalid",
            Self::Workspace => "source work fixed workspace mismatch or limit",
            Self::WorkLimit => "source logical work limit exceeded",
            Self::Json(_) => "source work JSON traversal refused",
            Self::Census(_) => "source work census resources refused",
            Self::Source(_) => "original source quotation refused",
        })
    }
}
impl std::error::Error for SourceWorkError {}

fn number(n: usize) -> Result<u64, SourceWorkError> {
    u64::try_from(n).map_err(|_| SourceWorkError::Overflow)
}
fn add(a: u64, b: u64) -> Result<u64, SourceWorkError> {
    a.checked_add(b).ok_or(SourceWorkError::Overflow)
}
fn mul(a: u64, b: u64) -> Result<u64, SourceWorkError> {
    a.checked_mul(b).ok_or(SourceWorkError::Overflow)
}
fn header_work(census_work: u64) -> Result<u64, SourceWorkError> {
    // Per admitted pass: <=10 structural reads/writes per old charged step,
    // plus None/init/Some/take/extraction/retirement's fixed 16 accesses.
    add(mul(10, census_work)?, 16)
}
fn census_work_with_header(census_work: u64) -> Result<u64, SourceWorkError> {
    add(census_work, header_work(census_work)?)
}
fn sum(values: &[u64]) -> Result<u64, SourceWorkError> {
    values.iter().try_fold(0, |a, &b| add(a, b))
}
fn byte_sum(values: &[usize]) -> Result<usize, SourceWorkError> {
    values.iter().try_fold(0_usize, |a, &b| {
        a.checked_add(b).ok_or(SourceWorkError::Overflow)
    })
}

/// Pure quote BEFORE caller constructs `[const { NativeFrame::empty() }; 17]`.
/// Quotes/requotes retain their prepaid maxima even when actual visits are less.
pub fn bootstrap(
    input: SourceInputLimits,
    lowerer: ConfigLowererLimits,
) -> Result<SourceWorkBootstrap, SourceWorkError> {
    if input.max_nodes == 0 || input.max_depth == 0 || input.max_depth > SOURCE_FRAME_SLOTS {
        return Err(SourceWorkError::InputLimit);
    }
    let native =
        Traversal::resources_for_native_slots(SOURCE_FRAME_SLOTS).map_err(SourceWorkError::Json)?;
    let post = census::resources_for_slots(SOURCE_FRAME_SLOTS).map_err(SourceWorkError::Census)?;
    // Concrete consumer-owned logical argument, event, iterator, return and
    // retained summary slots. Native cursor/meter phases come from their owner.
    let slots = [
        size_of::<SourceInputLimits>(),
        size_of::<ConfigLowererLimits>(),
        size_of::<SourceWorkBootstrap>(),
        2 * size_of::<SourceWorkQuote>(),
        size_of::<json::Usage>(),
        size_of::<OpaqueHelpers>(),
        size_of::<SourceWorkSlices>(),
        2 * size_of::<ConfigLowererQuote<'_>>(),
        2 * size_of::<crate::source_budget::SourceMetadataEstimate>(),
        2 * size_of::<json::Limits>(),
        2 * size_of::<json::Resources>(),
        2 * size_of::<census::Resources>(),
        2 * size_of::<Result<(ConfigLowererQuote<'_>, SourceWorkQuote), SourceWorkError>>(),
        2 * size_of::<Result<Option<Event<'_>>, json::Error>>(),
        2 * size_of::<Event<'_>>(),
        size_of::<(
            &Value,
            DecoderWorkload,
            DecoderExpansionBudget,
            ConfigLowererLimits,
            SourceInputLimits,
            &mut [NativeFrame<'_>],
        )>(),
        // Fixed literal arithmetic-fold arrays (largest is the factory sum).
        size_of::<[u64; 8]>(),
        size_of::<std::slice::Iter<'_, u64>>(),
        size_of::<std::slice::Iter<'_, usize>>(),
        size_of::<Option<&u64>>(),
        size_of::<Option<&usize>>(),
        2 * size_of::<Result<u64, SourceWorkError>>(),
        size_of::<[usize; 2]>(),
        size_of::<&[usize]>(),
        size_of::<&[u64]>(),
        2 * size_of::<Result<usize, SourceWorkError>>(),
    ];
    let control = byte_sum(&[byte_sum(&slots)?, std::mem::size_of_val(&slots)])?;
    let query = byte_sum(&[native.logical_workspace_bytes, control])?;
    let post_phase = byte_sum(&[post.logical_workspace_bytes, size_of::<SourceWorkQuote>()])?;
    let bootstrap_bytes = query.max(post_phase);
    if bootstrap_bytes > input.max_workspace_bytes {
        return Err(SourceWorkError::Workspace);
    }
    let nodes = number(input.max_nodes)?;
    let payload = number(input.max_payload_bytes)?;
    // One metered native walk; metadata string scan + original backing census.
    // The registered source query has <=13 Value field lookups (keys <=32 B),
    // <=11 registry comparisons, and nine <=4-component dimension products.
    // Calls/visited inputs are logical units; std tree/numeric CPU is unqualified.
    let fields = mul(13, mul(add(nodes, 1)?, 33)?)?;
    let source_passes = add(mul(2, nodes)?, mul(2, payload)?)?;
    let scalar_calls = sum(&[11, 9 * 4, 16, 3, 5, 5])?;
    // Number backing census may format each scalar again. Error diagnostics can
    // copy/escape one source payload; none is mistaken for zero-cost cleanup.
    let diagnostics = add(mul(10, payload)?, mul(32, add(nodes, 1)?)?)?;
    let quote_work = sum(&[
        number(input.max_traversal_work)?,
        fields,
        source_passes,
        scalar_calls,
        diagnostics,
        number(lowerer.max_string_bytes.min(input.max_string_bytes))?,
    ])?;
    if mul(2, quote_work)? > input.max_total_work {
        return Err(SourceWorkError::WorkLimit);
    }
    Ok(SourceWorkBootstrap {
        frame_slots: SOURCE_FRAME_SLOTS,
        bootstrap_bytes,
        quote_work,
        requote_work: quote_work,
    })
}

/// Pure work/layout query, never invokes `construct` or creates a DecoderPlan.
/// Its returned original lowerer captures exactly this document/workload.
/// The Copy work summary is numeric, not a replacement for that original borrow:
///
/// ```compile_fail
/// use pllm_models::{ConfigLowererLimits, ConfigLowererQuote, DecoderExpansionBudget, DecoderWorkload};
/// use pllm_models::source_work::{quote, SourceInputLimits};
/// use pllm_types::bounded_canonical::borrowed::NativeFrame;
/// fn escape<'a>(workload: DecoderWorkload, expansion: DecoderExpansionBudget,
///     lowerer: ConfigLowererLimits, limits: SourceInputLimits,
///     frames: &mut [NativeFrame<'a>]) -> ConfigLowererQuote<'a> {
///     let temporary = serde_json::json!({});
///     quote(&temporary, workload, expansion, lowerer, limits, frames).unwrap().0
/// }
/// ```
pub fn quote<'a>(
    document: &'a Value,
    workload: DecoderWorkload,
    expansion: DecoderExpansionBudget,
    lowerer_limits: ConfigLowererLimits,
    input_limits: SourceInputLimits,
    frames: &mut [NativeFrame<'a>],
) -> Result<(ConfigLowererQuote<'a>, SourceWorkQuote), SourceWorkError> {
    let resources = bootstrap(input_limits, lowerer_limits)?;
    if frames.len() != resources.frame_slots {
        return Err(SourceWorkError::Workspace);
    }
    let native =
        Traversal::resources_for_native_slots(frames.len()).map_err(SourceWorkError::Json)?;
    let limits = json::Limits {
        max_nodes: input_limits.max_nodes,
        max_depth: input_limits.max_depth,
        max_bytes: input_limits.max_payload_bytes,
        max_string_bytes: input_limits.max_string_bytes,
        max_key_bytes: input_limits.max_key_bytes,
        max_work: input_limits.max_traversal_work,
        max_workspace_bytes: native.logical_workspace_bytes,
    };
    let mut walk =
        Traversal::new_native(document, frames, limits).map_err(SourceWorkError::Json)?;
    let mut max_text = 0_usize;
    while let Some(event) = walk.next_event().map_err(SourceWorkError::Json)? {
        match event {
            Event::Key { key, .. } => max_text = max_text.max(key.len()),
            Event::Value {
                value: Value::String(text),
                ..
            } => max_text = max_text.max(text.len()),
            _ => {}
        }
    }
    let input = walk.usage();
    let original = ConfigLowererQuote::new(document, workload, expansion, lowerer_limits)
        .map_err(SourceWorkError::Source)?;
    let metadata = original.metadata();
    let operations = mul(2, number(metadata.operations_per_phase)?)?;
    // Full original prefill has outputs; decode has inputs AND outputs.
    let states = mul(3, number(metadata.states_per_phase)?)?;
    let attributes = number(metadata.attribute_nodes)?;
    let strings = number(metadata.retained_string_bytes)?;
    let records = sum(&[operations, states, 4])?;
    let payload = add(strings, mul(32, attributes)?)?;
    // Registered dense factories have rank <=4, <=2 input IDs and <=16 DTO
    // field/shape slots per record. This is a conservative all-field/all-record
    // cover, not another operator recipe or a claim of one pass per operation.
    // It includes state joins, whole-list producer/consumer scans and key probes.
    let cells = add(attributes, mul(16, records)?)?;
    let pairs = mul(records, cells)?;
    let validation = mul(pairs, add(number(max_text.max(128))?, 1)?)?;
    // Typed source configs have <=32 fields, <=32-byte static keys, and at most
    // twelve bytes of default string values. Include the original input subtree
    // for optional Value fields even when validation subsequently rejects it.
    let typed_nodes = add(number(input.nodes)?, 33)?;
    let typed_bytes = sum(&[number(input.bytes)?, 32 * 32, 32 * 32, 12])?;
    let canonical_bytes = add(mul(6, typed_bytes)?, mul(3, typed_nodes)?)?;
    let helpers = OpaqueHelpers {
        typed_deserializations: 1,
        canonical_serializations: 1,
        sort_calls: typed_nodes,
        // At most one source validation + one per attribute/descriptor context;
        // calls/input bytes do not price Rust's private float parser internals.
        decimal_parse_calls: add(cells, 1)?,
        decimal_parse_input_bytes: mul(add(cells, 1)?, number(max_text.max(32))?)?,
        // Both query passes each format source Numbers twice; include typed
        // canonical Numbers, postcensus Numbers and generated decimal layer IDs.
        number_format_calls: sum(&[
            mul(4, number(input.nodes)?)?,
            typed_nodes,
            attributes,
            cells,
        ])?,
        hash_input_bytes: add(canonical_bytes, 65)?,
        typed_input_nodes: typed_nodes,
        typed_input_bytes: typed_bytes,
    };
    let helper_calls = sum(&[
        helpers.typed_deserializations,
        helpers.canonical_serializations,
        helpers.sort_calls,
        helpers.decimal_parse_calls,
        helpers.number_format_calls,
    ])?;
    // Potential initialized/copied bytes of the REGISTERED factory envelope are
    // charged conservatively; they are not measured writes or allocator time.
    let initialization = mul(2, number(original.factory_bytes())?)?;
    let factory = sum(&[
        cells,
        payload,
        validation,
        typed_nodes,
        typed_bytes,
        canonical_bytes,
        helpers.hash_input_bytes,
        initialization,
    ])?;
    let factory = sum(&[factory, helper_calls, helpers.decimal_parse_input_bytes])?;
    // Both success and partial-error paths retain cleanup allowance. This is
    // source-factory scratch/output cleanup, not a second credit for B's owner.
    let cleanup = sum(&[
        number(original.factory_bytes())?,
        cells,
        payload,
        typed_bytes,
    ])?;
    // Shared Census::plan: 10 header steps, <=5 per operation, and each state
    // once; native order <=5 transitions/candidates per node +2/root, plus
    // <=32 formatted bytes/Number. No sorted-selection quadratic work here.
    let dto = sum(&[10, mul(5, operations)?, states])?;
    let census_work = sum(&[
        dto,
        mul(5, attributes)?,
        mul(2, operations)?,
        mul(32, attributes)?,
    ])?;
    let slices = SourceWorkSlices {
        quote: resources.quote_work,
        requote: resources.requote_work,
        factory,
        cleanup,
        census: census_work_with_header(census_work)?,
    };
    let total = slices.total_work()?;
    if total > input_limits.max_total_work {
        return Err(SourceWorkError::WorkLimit);
    }
    let post_resources =
        census::resources_for_slots(SOURCE_FRAME_SLOTS).map_err(SourceWorkError::Census)?;
    let post = census::Limits {
        max_backing_bytes: original.factory_bytes(),
        max_work: usize::try_from(census_work).map_err(|_| SourceWorkError::Overflow)?,
        max_workspace_bytes: post_resources.logical_workspace_bytes,
        max_json_nodes: metadata.attribute_nodes,
        max_json_payload_bytes: usize::try_from(payload).map_err(|_| SourceWorkError::Overflow)?,
        max_json_string_bytes: metadata.retained_string_bytes,
        max_json_key_bytes: metadata.retained_string_bytes,
        max_json_depth: SOURCE_FRAME_SLOTS - 1,
    };
    Ok((
        original,
        SourceWorkQuote {
            resources,
            slices,
            total,
            input,
            helpers,
            post,
            post_resources,
        },
    ))
}

#[cfg(test)]
#[path = "source_work_tests.rs"]
mod tests;
