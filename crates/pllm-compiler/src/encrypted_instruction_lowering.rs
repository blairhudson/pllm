//! Bounded SOURCE authoring, not ciphertext issuance or Qwen adoption.
//!
//! The existing SSA has no plaintext/key/drop fields. Exact borrowed external
//! selector rows therefore remain attached to the result; a kernel label alone
//! is NEVER sufficient. No registration, allocator, solver or native path here.

use crate::encrypted_level_schedule::*;
use construction::{ControlLedger, MetadataLedger, PublicCiphertextConstructionManifest};
use pllm_core::{row_fold::InstructionPlan, slot_matrix_layout::BlockPcPhysicalPlan};
use pllm_garble::shared_allocation::{Budget, ExternalCharge};

pub(crate) const MAX_OPS: usize = 128;
const MAX_VALUES: usize = MAX_OPS + 4;
const NONE: usize = usize::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Refusal {
    Source,
    LegacyUnresolved,
    MissingOperandBinding,
    MissingKeyBinding,
    MissingDropBinding,
    MissingOverlapBinding,
    Binding,
    State,
    Capacity,
    Work,
    Overflow,
    Poisoned,
    Construction(construction::Refusal),
}
impl From<construction::Refusal> for Refusal {
    fn from(e: construction::Refusal) -> Self { Self::Construction(e) }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Branch { Support, Wrap, Remainder }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Arithmetic { Add, Subtract, Square }
/// Source coordinates are not friendly kernel names. A row is matched to one
/// EXACT capability index, never discovered by searching its kernel label.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Selector {
    Arithmetic(Arithmetic),
    RowVector(usize),
    RowRotation(isize),
    Weight { q: usize, a: usize, j: usize },
    Mask { q: usize, a: usize, branch: Branch },
    Rotation { q: usize, a: usize, shift: isize },
    Drop { q: usize, a: usize, boundary: u8, branch: Option<Branch> },
    Bias(usize),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Instruction {
    pub selector: Selector,
    pub parents: [usize; 2],
    pub arity: usize,
}
impl Instruction {
    pub(crate) fn unary(selector: Selector, parent: usize) -> Self {
        Self { selector, parents: [parent, NONE], arity: 1 }
    }
    pub(crate) fn add(left: usize, right: usize) -> Self {
        Self { selector: Selector::Arithmetic(Arithmetic::Add), parents: [left, right], arity: 2 }
    }
}

/// Full original owner custody, including invisible suffixes/allocator padding.
/// `original_bytes` is upstream evidence, not inferred from a view's length.
/// Guard and actual original backing remain borrowed through validation.
pub(crate) enum OriginalBacking<'a> { Bytes(&'a Vec<u8>), Floats(&'a Vec<f64>) }
impl OriginalBacking<'_> {
    fn range(&self, capacity: usize) -> Result<(usize, usize), Refusal> {
        let (start, visible) = match self {
            Self::Bytes(b) => (b.as_ptr() as usize, b.capacity()),
            Self::Floats(b) => (b.as_ptr() as usize, mul(b.capacity(), 8)?),
        };
        if visible == 0 || visible > capacity || capacity == 0 { return Err(Refusal::Capacity); }
        Ok((start, add(start, capacity)?))
    }
}
pub(crate) struct OriginalOwner<'a> {
    pub owner: &'a PublicCiphertextOwner,
    pub backing: OriginalBacking<'a>,
    pub original_bytes: Option<usize>,
    pub guard: &'a ExternalCharge<'a>,
    pub manifest_other: usize,
    /// Mandatory original concrete encoding state for public operand owners.
    /// This is a borrowed external declaration, NOT an encoding/key proof.
    pub parameter_state: Option<&'a PublicCiphertextState>,
}
/// Original source arrays are borrowed, never finite-scanned or materialized.
pub(crate) enum Source<'a> {
    Row { plan: &'a InstructionPlan, public: &'a Vec<f64>, original: &'a OriginalOwner<'a> },
    Pc { plan: &'a BlockPcPhysicalPlan, weights: &'a Vec<f64>, bias: &'a Vec<f64>,
         weights_owner: &'a OriginalOwner<'a>, bias_owner: &'a OriginalOwner<'a> },
}
impl Source<'_> {
    fn roots(&self) -> usize { match self { Self::Row { .. } => 1, Self::Pc { .. } => 4 } }
    fn count(&self) -> Result<usize, Refusal> {
        match self {
            Self::Row { plan, .. } => Ok(plan.instructions().len()),
            Self::Pc { plan, .. } => {
                // Only the sealed Scaled78 source variant has 78 stages. Avoid
                // copying the original inline plan through its by-value accessor.
                if plan.stages().len() != 78 {
                    return Err(Refusal::LegacyUnresolved);
                }
                Ok(88)
            }
        }
    }
    fn final_value(&self, ordinal: usize, count: usize) -> bool {
        match self { Self::Row { .. } => ordinal == count,
            Self::Pc { .. } => ordinal == count + 2 || ordinal == count + 3 }
    }
    fn cursor(&self) -> Cursor<'_> {
        match self {
            Self::Row { plan, .. } => Cursor::Row { plan, index: 0 },
            Self::Pc { plan, .. } => Cursor::Pc(crate::slot_pc_trace::Cursor::new(plan)),
        }
    }
}
enum Cursor<'a> {
    Row { plan: &'a InstructionPlan, index: usize },
    Pc(crate::slot_pc_trace::Cursor<'a>),
}
impl Cursor<'_> {
    fn next(&mut self) -> Result<Option<Instruction>, Refusal> {
        match self {
            Self::Row { plan, index } => {
                let op = crate::row_fold_ciphertext_lowering::instruction(plan, *index)?;
                *index += usize::from(op.is_some());
                Ok(op)
            }
            Self::Pc(c) => c.next(),
        }
    }
}

/// Explicit external encoded operand binding. Source owner and coordinate are
/// checked independently of encoded owner/capability identity. No encoding claim
/// is manufactured by this module; caller must supply the original contract.
pub(crate) enum Operand<'a> {
    RowVector { source: &'a Vec<f64>, vector: usize },
    Weight { source: &'a Vec<f64>, q: usize, a: usize, j: usize },
    Mask { plan: &'a BlockPcPhysicalPlan, q: usize, a: usize, branch: Branch },
    Bias { source: &'a Vec<f64>, q: usize },
}
pub(crate) enum Evidence<'a> {
    Arithmetic,
    Operand { source: Operand<'a>, encoded: &'a OriginalOwner<'a> },
    Rotation { signed_offset: isize, key: &'a OriginalOwner<'a>,
               key_domain: &'a str, layout: &'a str },
    /// Ordered concrete prime identities, not symbolic Delta/alpha/beta labels.
    /// `before` must be a prefix of `after`; exactly one new original prime.
    Drop { context: &'a OriginalOwner<'a>, before: &'a Vec<&'a OriginalOwner<'a>>,
           after: &'a Vec<&'a OriginalOwner<'a>>, input: &'a PublicCiphertextState,
           output: &'a PublicCiphertextState },
}
pub(crate) struct Binding<'a> {
    pub selector: Selector,
    pub capability: usize,
    pub evidence: Evidence<'a>,
}
pub(crate) struct Bindings<'a> {
    pub rows: &'a Vec<Binding<'a>>,
    pub capabilities: &'a Vec<PublicCiphertextCapability>,
    /// Full ordinal table: original roots, then fresh outputs in source order.
    pub values: &'a Vec<PublicCiphertextValue>,
    pub owners: &'a Vec<OriginalOwner<'a>>,
    /// Explicit externally encoded zero roots, in frame order; no synthesized zero.
    pub zeros: Option<[&'a OriginalOwner<'a>; 2]>,
    pub budget: &'a Budget,
    /// Aggregate ORIGINAL charge over distinct metadata allocations, NOT a graph
    /// backing identity or pretend packed persistent owner. Every original Vec is
    /// borrowed whole, so suffix capacities and nested allocations can be counted.
    pub metadata: MetadataCharge<'a>,
    /// Actual admitted original scratch/recovery pools, by source step. None is
    /// legal ONLY when every selected capability explicitly declares both zero.
    pub overlap: Option<&'a Vec<StepOverlap<'a>>>,
}
pub(crate) struct StepOverlap<'a> {
    /// Exact indexes into the borrowed ORIGINAL owner table; never kernel names.
    pub scratch: &'a Vec<usize>,
    pub recovery: &'a Vec<usize>,
}
pub(crate) struct MetadataCharge<'a> {
    pub guard: &'a ExternalCharge<'a>,
    pub original_bytes: Option<usize>,
    pub manifest_other: usize,
}

/// Pre-sized upstream-owned slots. No public partial graph getter. Failure or
/// unwind after filling starts poisons this storage; it cannot be qualified again.
pub(crate) struct Storage {
    graph: PublicCiphertextGraph,
    trace: Vec<PublicCiphertextTraceStep>,
    poisoned: bool,
}
impl Storage {
    pub(crate) fn new(graph: PublicCiphertextGraph, trace: Vec<PublicCiphertextTraceStep>) -> Self {
        Self { graph, trace, poisoned: false }
    }
}
/// Immutable checked metadata retaining original source, exact external rows,
/// all original guards, graph storage and BOTH exclusive cumulative ledgers.
/// This does not prove encryption, key validity or numerical/source adoption.
pub(crate) struct Checked<'a, 'w> {
    source: Source<'a>,
    bindings: &'a Bindings<'a>,
    manifest: &'a PublicCiphertextConstructionManifest<'a>,
    storage: &'a Storage,
    control: &'a mut ControlLedger<'w>,
    metadata: &'a mut MetadataLedger,
    summary: PublicCiphertextTraceSummary,
}
impl Checked<'_, '_> {
    pub(crate) fn graph(&self) -> &PublicCiphertextGraph { &self.storage.graph }
    pub(crate) fn trace(&self) -> &[PublicCiphertextTraceStep] { &self.storage.trace }
    pub(crate) fn summary(&self) -> PublicCiphertextTraceSummary { self.summary }
}

struct Tables {
    count: usize,
    roots: usize,
    capability: [usize; MAX_OPS],
    last: [usize; MAX_VALUES],
    releases: [[usize; 2]; MAX_OPS],
    release_len: [usize; MAX_OPS],
    parents: [[usize; 2]; MAX_OPS],
    arity: [usize; MAX_OPS],
}
fn add(a: usize, b: usize) -> Result<usize, Refusal> { a.checked_add(b).ok_or(Refusal::Overflow) }
fn mul(a: usize, b: usize) -> Result<usize, Refusal> { a.checked_mul(b).ok_or(Refusal::Overflow) }

/// Fixed bounded logical-event tariff, including terminating advances, two source
/// passes, fixed tables, byte comparisons/copies, backing scans and old+new slots.
/// Byte work and control work BOTH consume the original core ledger. No new grant.
fn prepay(source: &Source<'_>, b: &Bindings<'_>, m: &PublicCiphertextConstructionManifest<'_>,
          control: &mut ControlLedger<'_>) -> Result<usize, Refusal> {
    control.core_work().debit(0, 64).map_err(|_| Refusal::Work)?;
    let n = source.count()?;
    if n == 0 || n > MAX_OPS || b.values.len() != n + source.roots()
        || b.rows.len() > 128 || b.capabilities.len() > 128 || b.owners.len() > 8192
        || m.owners > 149 || m.persistent_owners > 22
        || m.backings.len() > 8192 || m.other_owners.len() > 8192
        || m.backings.len() + m.other_owners.len() > 8192 {
        return Err(Refusal::Capacity);
    }
    let pairs = add(add(add(mul(b.values.len(), b.values.len())?, mul(b.rows.len(), b.rows.len())?)?,
        mul(b.owners.len(), b.owners.len())?)?, mul(m.owners, m.owners)?)?;
    let traversal = add(mul(n + 1, 256)?, mul(b.capabilities.len() + m.backings.len() + 1, 64)?)?;
    // Additional final-owner membership comparisons and identity predicates,
    // plus constant final-scan allowance guard work, paid on ORIGINAL ledger.
    let identity_scan = add(64, mul(m.owners, add(b.values.len(), m.persistent_owners)?)?)?;
    let events = add(identity_scan, add(32768, add(traversal, add(mul(pairs, 4)?, mul(b.owners.len() + 1, 64)?)?)?)?)?;
    let bytes = mul(64, add(identity_scan, add(mul(pairs, 8)?, traversal)?)?)?;
    control.core_work().debit(bytes, events).map_err(|_| Refusal::Work)?;
    Ok(n)
}
fn bounded(s: &str) -> bool { s.len() <= 64 && pllm_types::valid_identity(s) }
fn state_ok(s: &PublicCiphertextState) -> bool {
    [&s.scheme, &s.key_domain, &s.layout, &s.scale, &s.error.model_identity]
        .iter().all(|s| bounded(s)) && (1..=2).contains(&s.shape.len())
        && s.shape.iter().all(|&d| d != 0) && s.error.bound.is_some()
}
fn shape_cells(s: &PublicCiphertextState) -> Result<usize, Refusal> {
    let mut cells=1usize;
    for &d in &s.shape { cells=mul(cells,usize::try_from(d).map_err(|_| Refusal::Overflow)?)?; }
    Ok(cells)
}
fn aligned(a: &PublicCiphertextState, b: &PublicCiphertextState) -> bool {
    a.scheme == b.scheme && a.key_domain == b.key_domain && a.layout == b.layout
        && a.scale == b.scale && a.level == b.level && a.shape == b.shape
        && a.error.model_identity == b.error.model_identity
}
fn same_domain_shape(a: &PublicCiphertextState, b: &PublicCiphertextState) -> bool {
    a.scheme == b.scheme && a.key_domain == b.key_domain && a.layout == b.layout
        && a.shape == b.shape && a.error.model_identity == b.error.model_identity
}
fn contains_owner(b: &Bindings<'_>, owner: &OriginalOwner<'_>) -> bool {
    b.owners.iter().any(|o| std::ptr::eq(o, owner))
}
fn source_operand(source: &Source<'_>, selector: Selector, operand: &Operand<'_>) -> bool {
    match (source, selector, operand) {
        (Source::Row { public, .. }, Selector::RowVector(v), Operand::RowVector { source, vector }) =>
            std::ptr::eq(*public, *source) && v == *vector,
        (Source::Pc { weights, .. }, Selector::Weight { q, a, j }, Operand::Weight { source, q: x, a: y, j: z }) =>
            std::ptr::eq(*weights, *source) && (q, a, j) == (*x, *y, *z),
        (Source::Pc { plan, .. }, Selector::Mask { q, a, branch }, Operand::Mask { plan: p, q: x, a: y, branch: z }) =>
            std::ptr::eq(*plan, *p) && (q, a, branch) == (*x, *y, *z),
        (Source::Pc { bias, .. }, Selector::Bias(q), Operand::Bias { source, q: x }) =>
            std::ptr::eq(*bias, *source) && q == *x,
        _ => false,
    }
}
fn evidence(source: &Source<'_>, b: &Bindings<'_>, row: &Binding<'_>, cap: &PublicCiphertextCapability,
            graph: &PublicCiphertextGraph)
    -> Result<(), Refusal> {
    match (row.selector, &row.evidence) {
        (Selector::Arithmetic(_), Evidence::Arithmetic) => Ok(()),
        (Selector::RowVector(_) | Selector::Weight { .. } | Selector::Mask { .. } | Selector::Bias(_),
         Evidence::Operand { source: operand, encoded }) => {
            let encoding=encoded.parameter_state.ok_or(Refusal::MissingOperandBinding)?;
            if !source_operand(source, row.selector, operand) || !contains_owner(b, encoded)
                || !matches!(encoded.backing, OriginalBacking::Bytes(_))
                || !graph.persistent_owners.contains(&encoded.owner.identity) {
                return Err(Refusal::MissingOperandBinding);
            }
            if !state_ok(encoding) || !same_domain_shape(encoding,&cap.inputs[0])
                || encoding.level!=cap.inputs[0].level
                || matches!(row.selector,Selector::Bias(_)) && encoding.scale!=cap.inputs[0].scale {
                return Err(Refusal::State);
            }
            Ok(())
        }
        (Selector::RowRotation(shift) | Selector::Rotation { shift, .. },
         Evidence::Rotation { signed_offset, key, key_domain, layout }) => {
            if shift != *signed_offset || !contains_owner(b, key)
                || !graph.persistent_owners.contains(&key.owner.identity)
                || *key_domain != cap.inputs[0].key_domain || *layout != cap.inputs[0].layout {
                return Err(Refusal::MissingKeyBinding);
            }
            if !aligned(&cap.inputs[0], &cap.output) { return Err(Refusal::State); }
            Ok(())
        }
        (Selector::Drop { boundary, .. }, Evidence::Drop { context, before, after, input, output }) => {
            if !contains_owner(b, context) || before.len() + 1 != after.len()
                || after.len() != usize::from(boundary) || after.len() > 2
                || !graph.persistent_owners.contains(&context.owner.identity)
                || before.capacity() > 2 || after.capacity() > 2
                || !after.iter().all(|o| contains_owner(b, o) && graph.persistent_owners.contains(&o.owner.identity))
                || !before.iter().zip(after.iter()).all(|(a, b)| std::ptr::eq(*a, *b))
                || (after.len() == 2 && std::ptr::eq(after[0], after[1]))
                || !std::ptr::eq(*input, &cap.inputs[0]) || !std::ptr::eq(*output, &cap.output) {
                return Err(Refusal::MissingDropBinding);
            }
            Ok(())
        }
        (Selector::Drop { .. }, _) => Err(Refusal::MissingDropBinding),
        (Selector::RowRotation(_) | Selector::Rotation { .. }, _) => Err(Refusal::MissingKeyBinding),
        _ => Err(Refusal::MissingOperandBinding),
    }
}

fn string_fits(dst: &String, src: &str) -> bool {
    bounded(src) && dst.len() == src.len() && dst.capacity() >= src.len() && dst.capacity() <= 64
}
fn state_fits(dst: &PublicCiphertextState, src: &PublicCiphertextState) -> bool {
    [(&dst.scheme, &src.scheme), (&dst.key_domain, &src.key_domain), (&dst.layout, &src.layout),
     (&dst.scale, &src.scale), (&dst.error.model_identity, &src.error.model_identity)]
        .iter().all(|(a, b)| string_fits(a, b))
        && dst.shape.len() == src.shape.len() && dst.shape.capacity() >= src.shape.len() && dst.shape.capacity() <= 2
}
fn copy_string(dst: &mut String, src: &str) { dst.clear(); dst.push_str(src); }
fn copy_state(dst: &mut PublicCiphertextState, src: &PublicCiphertextState) {
    copy_string(&mut dst.scheme, &src.scheme);
    copy_string(&mut dst.key_domain, &src.key_domain);
    copy_string(&mut dst.layout, &src.layout);
    copy_string(&mut dst.scale, &src.scale);
    copy_string(&mut dst.error.model_identity, &src.error.model_identity);
    dst.error.bound = src.error.bound;
    dst.level = src.level;
    dst.shape.clear(); dst.shape.extend_from_slice(&src.shape);
}

fn state_bytes(s: &PublicCiphertextState) -> Result<usize, Refusal> {
    if s.shape.capacity() > 2 { return Err(Refusal::Capacity); }
    let mut bytes = mul(s.shape.capacity(), std::mem::size_of::<u64>())?;
    for id in [&s.scheme, &s.key_domain, &s.layout, &s.scale, &s.error.model_identity] {
        if id.capacity() > 64 { return Err(Refusal::Capacity); }
        bytes = add(bytes, id.capacity())?;
    }
    Ok(bytes)
}
fn string_bytes(s: &String) -> Result<usize, Refusal> {
    if s.capacity() > 64 { Err(Refusal::Capacity) } else { Ok(s.capacity()) }
}
/// Inventory old source/operand/state/capability/control metadata PLUS destination
/// graph/trace allocations. Distinct allocations stay distinct in the manifest;
/// the original aggregate budget charge does not merge ciphertext/key backings.
fn metadata_bytes(source: &Source<'_>, b: &Bindings<'_>, m: &PublicCiphertextConstructionManifest<'_>)
    -> Result<usize, Refusal> {
    use construction::Element;
    if b.values.capacity() > MAX_VALUES || b.rows.capacity() > MAX_OPS
        || b.capabilities.capacity() > MAX_OPS || b.owners.capacity() > 8192 {
        return Err(Refusal::Capacity);
    }
    // Old metadata is an aggregate CHARGE, not one fictitious allocation. Count
    // its distinct bounded allocation slots as well as destination records. Alias
    // tables may be counted twice conservatively, never silently collapsed.
    let mut old_records=add(4,add(mul(b.values.len(),8)?,mul(b.owners.len(),2)?)?)?;
    let mut bytes = add(8192*10 + std::mem::size_of::<Tables>() + std::mem::size_of::<Cursor<'_>>()
        + std::mem::size_of::<Bindings<'_>>() + std::mem::size_of::<Source<'_>>()
        + std::mem::size_of::<Storage>() + std::mem::size_of::<ExternalCharge<'_>>()
        + std::mem::size_of::<ControlLedger<'_>>() + std::mem::size_of::<MetadataLedger>()
        + std::mem::size_of::<pllm_core::row_fold::WorkLedger>()
        + 4 * std::mem::size_of::<[usize; 257]>(), construction::VALIDATOR_BYTES_CEILING)?;
    bytes = add(bytes, match source { Source::Row { .. } => std::mem::size_of::<InstructionPlan>(),
        Source::Pc { .. } => std::mem::size_of::<BlockPcPhysicalPlan>() })?;
    for count in [mul(b.values.capacity(), std::mem::size_of::<PublicCiphertextValue>())?,
        mul(b.rows.capacity(), std::mem::size_of::<Binding<'_>>())?,
        mul(b.capabilities.capacity(), std::mem::size_of::<PublicCiphertextCapability>())?,
        mul(b.owners.capacity(), std::mem::size_of::<OriginalOwner<'_>>())?,
        m.manifest_original_bytes.ok_or(Refusal::Capacity)?] { bytes = add(bytes, count)?; }
    for v in b.values {
        bytes = add(bytes, add(string_bytes(&v.identity)?, add(string_bytes(&v.owner)?, state_bytes(&v.state)?)?)?)?;
    }
    for cap in b.capabilities {
        if cap.inputs.capacity() > 2 { return Err(Refusal::Capacity); }
        old_records=add(old_records,add(12,mul(cap.inputs.len(),6)?)?)?;
        bytes = add(bytes, mul(cap.inputs.capacity(), std::mem::size_of::<PublicCiphertextState>())?)?;
        bytes = add(bytes, add(string_bytes(&cap.identity)?, add(string_bytes(&cap.kernel_identity)?, state_bytes(&cap.output)?)?)?)?;
        for s in &cap.inputs { bytes = add(bytes, state_bytes(s)?)?; }
        let assumption = cap.error_assumption.as_ref().ok_or(Refusal::State)?;
        bytes = add(bytes, string_bytes(assumption)?)?;
        match &cap.execution {
            PublicCiphertextExecution::Ckks { compute, transition, .. } => {
                bytes = add(bytes, add(string_bytes(&compute.cost.identity)?, string_bytes(&transition.cost.identity)?)?)?;
            }
            _ => return Err(Refusal::State),
        }
    }
    for row in b.rows {
        if let Evidence::Drop { before, after, .. } = &row.evidence {
            if before.capacity() > 2 || after.capacity() > 2 { return Err(Refusal::Capacity); }
            old_records=add(old_records,2)?;
            bytes = add(bytes, add(2*std::mem::size_of::<Vec<&OriginalOwner<'_>>>(),
                mul(before.capacity() + after.capacity(), std::mem::size_of::<&OriginalOwner<'_>>())?)?)?;
        }
    }
    if let Some(overlap)=b.overlap {
        if overlap.capacity()>MAX_OPS { return Err(Refusal::Capacity); }
        old_records=add(old_records,1)?;
        bytes=add(bytes,mul(overlap.capacity(),std::mem::size_of::<StepOverlap<'_>>())?)?;
        for step in overlap {
            if step.scratch.capacity()>b.owners.len() || step.recovery.capacity()>b.owners.len() {
                return Err(Refusal::Capacity);
            }
            old_records=add(old_records,2)?;
            bytes=add(bytes,add(2*std::mem::size_of::<Vec<usize>>(),
                mul(step.scratch.capacity()+step.recovery.capacity(),std::mem::size_of::<usize>())?)?)?;
        }
    }
    if add(old_records,add(m.backings.len(),m.other_owners.len())?)?>8192 { return Err(Refusal::Capacity); }
    for owner in b.owners {
        bytes=add(bytes, add(std::mem::size_of::<PublicCiphertextOwner>()
            +std::mem::size_of::<Vec<u8>>() +std::mem::size_of::<ExternalCharge<'_>>(),
            string_bytes(&owner.owner.identity)?)?)?;
        if let Some(state)=owner.parameter_state {
            old_records=add(old_records,6)?;
            bytes=add(bytes,add(std::mem::size_of::<PublicCiphertextState>(),state_bytes(state)?)?)?;
        }
    }
    if add(old_records,add(m.backings.len(),m.other_owners.len())?)?>8192 { return Err(Refusal::Capacity); }
    for backing in m.backings {
        let size = match backing.element {
            Element::Byte => 1, Element::String => std::mem::size_of::<String>(),
            Element::Shape => std::mem::size_of::<u64>(), Element::Owner => std::mem::size_of::<PublicCiphertextOwner>(),
            Element::Value => std::mem::size_of::<PublicCiphertextValue>(), Element::Operation => std::mem::size_of::<PublicCiphertextGraphOperation>(),
            Element::Capability => std::mem::size_of::<PublicCiphertextCapability>(), Element::Step => std::mem::size_of::<PublicCiphertextTraceStep>(),
            Element::State => std::mem::size_of::<PublicCiphertextState>(),
        };
        bytes = add(bytes, mul(backing.capacity.ok_or(Refusal::Capacity)?, size)?)?;
    }
    Ok(bytes)
}

fn custody(source: &Source<'_>, b: &Bindings<'_>, m: &PublicCiphertextConstructionManifest<'_>) -> Result<(), Refusal> {
    if b.budget.used_bytes() > construction::OWNER_BYTES { return Err(Refusal::Capacity); }
    let metadata = m.other_owners.get(b.metadata.manifest_other).ok_or(Refusal::Capacity)?;
    let bytes = b.metadata.original_bytes.ok_or(Refusal::Capacity)?;
    if m.other_owners.len()!=b.owners.len()+1 || b.metadata.manifest_other>=8192
        || !metadata.metadata || metadata.bytes!=Some(bytes)
        || metadata.first_phase!=0 || metadata.end_phase!=m.phases
        || !b.metadata.guard.is_for(b.budget) || b.metadata.guard.payload_bytes()<bytes
        || b.owners.iter().any(|o|std::ptr::eq(o.guard,b.metadata.guard))
        || bytes<metadata_bytes(source,b,m)? { return Err(Refusal::Capacity); }
    let mut seen = [false; 8192];
    for (index, o) in b.owners.iter().enumerate() {
        let bytes = o.original_bytes.ok_or(Refusal::Capacity)?;
        let item = m.other_owners.get(o.manifest_other).ok_or(Refusal::Capacity)?;
        if !bounded(&o.owner.identity) || o.owner.bytes != u64::try_from(bytes).ok()
            || bytes > construction::OWNER_BYTES
            || !o.guard.is_for(b.budget) || o.guard.payload_bytes() < bytes
            || item.bytes != Some(bytes) || item.first_phase != 0 || item.end_phase != m.phases
            || seen[o.manifest_other] {
            return Err(Refusal::Capacity);
        }
        let (start, end) = o.backing.range(bytes)?;
        for previous in &b.owners[..index] {
            let (a, z) = previous.backing.range(previous.original_bytes.ok_or(Refusal::Capacity)?)?;
            if std::ptr::eq(previous.guard, o.guard) || previous.owner.identity == o.owner.identity
                || start < z && a < end { return Err(Refusal::Capacity); }
        }
        seen[o.manifest_other] = true;
    }
    if seen[b.metadata.manifest_other] { return Err(Refusal::Capacity); }
    match source {
        Source::Row { plan, public, original } => {
            let q = plan.geometry().quote().map_err(|_| Refusal::Source)?;
            let original_matches = matches!(&original.backing, OriginalBacking::Floats(v) if std::ptr::eq(*v, *public));
            if !contains_owner(b, original) || !original_matches || public.len() < q.public_cells
                || mul(public.len(), 8)? > original.original_bytes.ok_or(Refusal::Capacity)? {
                return Err(Refusal::MissingOperandBinding);
            }
        }
        Source::Pc { plan, weights, bias, weights_owner, bias_owner } => {
            let mut width=0;
            for frame in 0..2 { for lane in 0..4 {
                if let Some((_,feature))=plan.output_coordinate(frame,lane).map_err(|_|Refusal::Source)? {
                    width=width.max(feature+1);
                }
            }}
            if weights.len() < 32 || bias.len() < width || !contains_owner(b, weights_owner) || !contains_owner(b, bias_owner)
                || !matches!(&weights_owner.backing, OriginalBacking::Floats(v) if std::ptr::eq(*v, *weights))
                || !matches!(&bias_owner.backing, OriginalBacking::Floats(v) if std::ptr::eq(*v, *bias))
                || mul(weights.len(), 8)? > weights_owner.original_bytes.ok_or(Refusal::Capacity)?
                || mul(bias.len(), 8)? > bias_owner.original_bytes.ok_or(Refusal::Capacity)? {
                return Err(Refusal::MissingOperandBinding);
            }
            zero_states(b)?;
        }
    }
    Ok(())
}

fn zero_states(b: &Bindings<'_>) -> Result<(), Refusal> {
    let zeros = b.zeros.ok_or(Refusal::MissingOperandBinding)?;
    for (q, zero) in zeros.iter().enumerate() {
        let root = b.values.get(q + 2).ok_or(Refusal::MissingOperandBinding)?;
        if !contains_owner(b, zero) || root.owner != zero.owner.identity {
            return Err(Refusal::MissingOperandBinding);
        }
        let declared = zero.parameter_state.ok_or(Refusal::MissingOperandBinding)?;
        if declared != &root.state { return Err(Refusal::State); }
    }
    // External typed declaration only: this cannot establish encrypted zero.
    Ok(())
}

fn preflight(source: &Source<'_>, b: &Bindings<'_>, s: &Storage, n: usize) -> Result<Tables, Refusal> {
    let g = &s.graph;
    let roots = source.roots();
    if s.poisoned { return Err(Refusal::Poisoned); }
    if !bounded(&g.cost_model_identity) || !bounded(&g.cost_unit) { return Err(Refusal::State); }
    if b.overlap.is_some_and(|o|o.len()!=n || o.capacity()>MAX_OPS){return Err(Refusal::MissingOverlapBinding);}
    if g.operations.len() != n || s.trace.len() != n || g.inputs.len() != roots
        || g.capabilities.len() > 128 || g.owners.len() > 149 || g.persistent_owners.len() > 22
        || g.operations.capacity() > 128 || g.capabilities.capacity() > 128 || s.trace.capacity() > 128
        || g.owners.capacity() > 149 || g.persistent_owners.capacity() > 22
        || g.retained_inputs.len() != roots || g.outputs.len() != if roots == 1 { 1 } else { 2 } {
        return Err(Refusal::Capacity);
    }
    // Bound nested identities/shapes BEFORE equality or linear selector scans.
    for cap in b.capabilities.iter().chain(&g.capabilities) {
        if !bounded(&cap.identity) || !bounded(&cap.kernel_identity) || cap.inputs.len() > 2
            || cap.inputs.capacity() > 2 || cap.inputs.iter().any(|s| !state_ok(s)) || !state_ok(&cap.output)
            || !cap.error_assumption.as_ref().is_some_and(|s| bounded(s)) {
            return Err(Refusal::State);
        }
        let PublicCiphertextExecution::Ckks{compute,transition,..}=&cap.execution else {return Err(Refusal::State);};
        if !bounded(&compute.cost.identity) || !bounded(&transition.cost.identity)
            || compute.cost.units.is_none() || transition.cost.units.is_none() {return Err(Refusal::State);}
    }
    if g.capabilities != *b.capabilities { return Err(Refusal::Binding); }
    for (i,cap) in b.capabilities.iter().enumerate() {
        if b.capabilities[..i].iter().any(|old|old.identity==cap.identity) {return Err(Refusal::Binding);}
    }
    for (i, owner) in g.owners.iter().enumerate() {
        if !bounded(&owner.identity) || owner.bytes.is_none_or(|bytes|bytes==0)
            || g.owners[..i].iter().any(|old| old.identity == owner.identity) {
            return Err(Refusal::Capacity);
        }
    }
    for id in g.persistent_owners.iter().chain(&g.outputs).chain(&g.retained_inputs) {
        if !bounded(id) { return Err(Refusal::Capacity); }
    }
    for (i,id) in g.persistent_owners.iter().enumerate() {
        if g.persistent_owners[..i].contains(id) || !g.owners.iter().any(|o|o.identity==*id) {return Err(Refusal::Binding);}
    }
    for (i, v) in b.values.iter().enumerate() {
        if !bounded(&v.identity) || !bounded(&v.owner) || !state_ok(&v.state) || v.state.level>g.maximum_level
            || !g.owners.iter().any(|o| o.identity == v.owner && o.bytes.is_some())
            || g.persistent_owners.contains(&v.owner)
            || b.values[..i].iter().enumerate().any(|(j,old)| old.identity == v.identity
                || old.owner == v.owner && (i >= roots || j >= roots || old.state != v.state)) {
            return Err(Refusal::State);
        }
        if i < roots && (g.inputs[i] != *v || g.retained_inputs[i] != v.identity) { return Err(Refusal::Binding); }
    }
    for owner in &g.owners {
        if !b.values.iter().any(|v| v.owner == owner.identity)
            && !g.persistent_owners.contains(&owner.identity) { return Err(Refusal::Binding); }
    }
    if roots==4 { zero_states(b)?; }
    let mut final_index = 0;
    for (ordinal, v) in b.values.iter().enumerate() {
        if source.final_value(ordinal, n) {
            if g.outputs[final_index] != v.identity { return Err(Refusal::Binding); }
            final_index += 1;
        }
    }
    for (i, row) in b.rows.iter().enumerate() {
        if row.capability >= b.capabilities.len() || b.rows[..i].iter().any(|old|
            old.selector == row.selector && old.capability == row.capability
                || old.selector != row.selector && (old.capability == row.capability
                    || b.capabilities[old.capability].kernel_identity == b.capabilities[row.capability].kernel_identity)) {
            return Err(Refusal::Binding);
        }
    }
    // Concrete ordered chain must be the SAME original context/primes across
    // partials, frames and both branches. Independent labels cannot substitute.
    let mut canonical: Option<(&OriginalOwner<'_>, &Vec<&OriginalOwner<'_>>)> = None;
    for row in b.rows {
        if let Evidence::Drop { context, before, after, .. } = &row.evidence {
            if before.len() > 1 || after.is_empty() || after.len() > 2
                || before.capacity() > 2 || after.capacity() > 2 { return Err(Refusal::MissingDropBinding); }
            if let Some((original, primes))=canonical {
                if !std::ptr::eq(original,*context)
                    || !after.iter().zip(primes.iter()).all(|(a,b)|std::ptr::eq(*a,*b)) {
                    return Err(Refusal::MissingDropBinding);
                }
                if after.len()>primes.len(){canonical=Some((context,after));}
            } else {canonical=Some((context,after));}
        }
    }
    let mut t = Tables { count: n, roots, capability: [NONE; MAX_OPS], last: [NONE; MAX_VALUES],
        releases: [[NONE; 2]; MAX_OPS], release_len: [0; MAX_OPS],
        parents: [[NONE; 2]; MAX_OPS], arity: [0; MAX_OPS] };
    let mut overlap_seen=[NONE;8192];
    let mut overlap_eligible=[false;8192];
    for (i,owner) in b.owners.iter().enumerate(){
        overlap_eligible[i]=matches!(owner.backing,OriginalBacking::Bytes(_))
            && !g.owners.iter().any(|o|o.identity==owner.owner.identity);
    }
    let mut cursor = source.cursor();
    let mut execution_units=0u64;
    for i in 0..n {
        let ins = cursor.next()?.ok_or(Refusal::Source)?;
        let output = i + roots;
        if !(1..=2).contains(&ins.arity) || ins.parents[..ins.arity].iter().any(|&p| p >= output) {
            return Err(Refusal::Source);
        }
        let mut selected = None;
        for row in b.rows {
            if row.selector == ins.selector {
                let cap = &b.capabilities[row.capability];
                if cap.inputs.len() == ins.arity && cap.output == b.values[output].state
                    && cap.inputs.iter().zip(&ins.parents).all(|(s, &p)| *s == b.values[p].state) {
                    if selected.is_some() { return Err(Refusal::Binding); }
                    selected = Some(row);
                }
            }
        }
        let row = selected.ok_or(match ins.selector {
            Selector::Drop { .. } => Refusal::MissingDropBinding,
            Selector::RowRotation(_) | Selector::Rotation { .. } => Refusal::MissingKeyBinding,
            _ => Refusal::MissingOperandBinding,
        })?;
        let cap = &b.capabilities[row.capability];
        if cap.primitive != PublicPrimitive::Arithmetic || !bounded(&cap.identity) || !bounded(&cap.kernel_identity)
            || !cap.error_assumption.as_ref().is_some_and(|s| bounded(s))
            || cap.scratch_bytes.is_none() || cap.recovery_overlap_bytes.is_none()
            || cap.inputs.iter().any(|s| !state_ok(s)) || !state_ok(&cap.output)
            || !cap.inputs.iter().all(|s| same_domain_shape(s, &cap.output))
            || cap.join != if ins.arity == 2 { PublicCiphertextJoin::Aligned } else { PublicCiphertextJoin::Unary }
            || (ins.arity == 2 && !aligned(&cap.inputs[0], &cap.inputs[1])) {
            return Err(Refusal::State);
        }
        let PublicCiphertextExecution::Ckks { compute, depth, transition } = &cap.execution else { return Err(Refusal::State); };
        let is_drop = matches!(ins.selector, Selector::Drop { .. });
        if compute.cost.units.is_none() || transition.cost.units.is_none() || !bounded(&compute.cost.identity)
            || !bounded(&transition.cost.identity) || compute.input_level != cap.inputs[0].level
            || compute.input_level.checked_sub(*depth) != Some(transition.from_level)
            || transition.to_level != cap.output.level || cap.output.level > g.maximum_level
            || transition.kind != if is_drop { PublicLevelTransitionKind::Drop } else { PublicLevelTransitionKind::Preserve }
            || (is_drop && (*depth != 0 || transition.from_level.checked_sub(1) != Some(transition.to_level)
                || cap.inputs[0].scale == cap.output.scale))
            || (!is_drop && (transition.to_level != transition.from_level || transition.cost.units != Some(0))) {
            return Err(Refusal::State);
        }
        if matches!(ins.selector, Selector::Arithmetic(Arithmetic::Add | Arithmetic::Subtract) | Selector::Bias(_))
            && !aligned(&cap.inputs[0], &cap.output) { return Err(Refusal::State); }
        execution_units=execution_units.checked_add(compute.cost.units.ok_or(Refusal::State)?)
            .and_then(|v|v.checked_add(transition.cost.units?)).ok_or(Refusal::Overflow)?;
        evidence(source, b, row, cap, g)?;
        let scratch=usize::try_from(cap.scratch_bytes.ok_or(Refusal::MissingOverlapBinding)?).map_err(|_|Refusal::Overflow)?;
        let recovery=usize::try_from(cap.recovery_overlap_bytes.ok_or(Refusal::MissingOverlapBinding)?).map_err(|_|Refusal::Overflow)?;
        if let Some(overlap)=b.overlap {
            let step=&overlap[i];
            if step.scratch.len()>b.owners.len() || step.recovery.len()>b.owners.len(){return Err(Refusal::MissingOverlapBinding);}
            let mut scratch_bytes=0;let mut recovery_bytes=0;
            for (index,&owner_index) in step.scratch.iter().chain(step.recovery.iter()).enumerate() {
                if owner_index>=b.owners.len() || !overlap_eligible[owner_index] || overlap_seen[owner_index]==i {
                    return Err(Refusal::MissingOverlapBinding);
                }
                overlap_seen[owner_index]=i;
                let owner=&b.owners[owner_index];
                let bytes=owner.original_bytes.ok_or(Refusal::MissingOverlapBinding)?;
                if index<step.scratch.len(){scratch_bytes=add(scratch_bytes,bytes)?;}else{recovery_bytes=add(recovery_bytes,bytes)?;}
            }
            if scratch_bytes<scratch || recovery_bytes<recovery {return Err(Refusal::MissingOverlapBinding);}
        } else if scratch!=0 || recovery!=0 {return Err(Refusal::MissingOverlapBinding);}
        let op = &g.operations[i]; let step = &s.trace[i]; let v = &b.values[output];
        if !bounded(&op.identity) || !string_fits(&op.kernel_identity, &cap.kernel_identity)
            || op.inputs.len() != ins.arity || op.inputs.capacity() > 2
            || !op.inputs.iter().zip(&ins.parents).all(|(dst, &p)| string_fits(dst, &b.values[p].identity))
            || !string_fits(&op.output.identity, &v.identity) || !string_fits(&op.output.owner, &v.owner)
            || !state_fits(&op.output.state, &v.state) || !string_fits(&step.operation, &op.identity)
            || !string_fits(&step.capability, &cap.identity) || step.release.capacity() > 2 {
            return Err(Refusal::Capacity);
        }
        if g.operations[..i].iter().any(|old|old.identity==op.identity){return Err(Refusal::Binding);}
        t.capability[i] = row.capability;
        t.parents[i] = ins.parents; t.arity[i] = ins.arity;
        for &parent in &ins.parents[..ins.arity] { t.last[parent] = i; }
    }
    if cursor.next()?.is_some() { return Err(Refusal::Source); }
    // Do not defer malformed unused capabilities to post-factory validation.
    // This exact source table carries only the externally bound capabilities used.
    for index in 0..b.capabilities.len() {
        if !t.capability[..n].contains(&index) { return Err(Refusal::Binding); }
    }
    let mut releases = 0;
    for ordinal in roots..roots + n {
        if !source.final_value(ordinal, n) && t.last[ordinal] == NONE { return Err(Refusal::Source); }
    }
    // Parent order, not sorted ordinals: e.g. NEXUS subtraction releases [8,7].
    for i in 0..n {
        for parent in 0..t.arity[i] {
            let ordinal = t.parents[i][parent];
            if ordinal < roots || source.final_value(ordinal, n) || t.last[ordinal] != i
                || (parent == 1 && t.parents[i][0] == ordinal) { continue; }
            let slot = t.release_len[i];
            if slot == 2 { return Err(Refusal::Source); }
            t.releases[i][slot] = ordinal; t.release_len[i] += 1; releases += 1;
        }
    }
    if releases > 125 { return Err(Refusal::Capacity); }
    for (i, step) in s.trace.iter().enumerate() {
        if step.release.len() != t.release_len[i] || !step.release.iter().zip(t.releases[i]).all(|(dst, ordinal)|
            string_fits(dst, &b.values[ordinal].identity)) { return Err(Refusal::Capacity); }
    }
    Ok(t)
}
fn fill(source: &Source<'_>, b: &Bindings<'_>, s: &mut Storage, t: &Tables) {
    // All fallible conditions and nested capacities checked BEFORE any mutation.
    // Set before first write: failure/unwind cannot publish or retry this storage.
    s.poisoned = true;
    let mut cursor = source.cursor();
    for i in 0..t.count {
        let ins = cursor.next().expect("immutable preflight source").expect("exact count");
        let cap = &b.capabilities[t.capability[i]]; let v = &b.values[t.roots + i];
        let op = &mut s.graph.operations[i];
        op.primitive = PublicPrimitive::Arithmetic;
        copy_string(&mut op.kernel_identity, &cap.kernel_identity);
        for (dst, &p) in op.inputs.iter_mut().zip(&ins.parents) { copy_string(dst, &b.values[p].identity); }
        copy_string(&mut op.output.identity, &v.identity); copy_string(&mut op.output.owner, &v.owner);
        copy_state(&mut op.output.state, &v.state);
        let step = &mut s.trace[i]; copy_string(&mut step.operation, &op.identity);
        copy_string(&mut step.capability, &cap.identity);
        for (dst, ordinal) in step.release.iter_mut().zip(t.releases[i]) { copy_string(dst, &b.values[ordinal].identity); }
    }
}

fn final_control_cost(manifest: &PublicCiphertextConstructionManifest<'_>) -> Result<usize, Refusal> {
    add(1024, mul(4, manifest.backings.len())?)
}
fn guard_final_control(manifest: &PublicCiphertextConstructionManifest<'_>, control: &mut ControlLedger<'_>)
    -> Result<(), Refusal> {
    // Exact dependency: construction::PublicCiphertextConstructionQuote::check_constructed
    // charges 1024 + 4 * manifest.backings.len(). Guard does NOT debit/refund.
    let cost = final_control_cost(manifest)?;
    let work = control.core_work();
    if work.remaining() < cost || work.control_remaining() < cost {
        return Err(Refusal::Construction(construction::Refusal::ControlWorkExhausted));
    }
    Ok(())
}

pub(crate) fn lower<'a, 'w>(source: Source<'a>, bindings: &'a Bindings<'a>,
    storage: &'a mut Storage, manifest: &'a PublicCiphertextConstructionManifest<'a>,
    control: &'a mut ControlLedger<'w>, metadata: &'a mut MetadataLedger,
    ambient: Option<PublicCiphertextTraceCapacity>) -> Result<Checked<'a, 'w>, Refusal> {
    let n = prepay(&source, bindings, manifest, control)?;
    if storage.graph.owners.len()!=manifest.owners { return Err(Refusal::Capacity); }
    custody(&source, bindings, manifest)?;
    let tables = preflight(&source, bindings, storage, n)?;
    for owner in &storage.graph.owners {
        if !bindings.owners.iter().any(|original| original.owner == owner) {
            return Err(Refusal::Capacity);
        }
    }
    let slots=match &source { Source::Row{plan,..}=>plan.geometry().quote().map_err(|_|Refusal::Source)?.slots,
        Source::Pc{..}=>4096 };
    for root in &bindings.values[..tables.roots] {
        if shape_cells(&root.state)?!=slots { return Err(Refusal::State); }
    }
    if manifest.operations.len() != n || manifest.input_ranks.len() != tables.roots
        || manifest.release_lengths != &tables.release_len[..n] { return Err(Refusal::Capacity); }
    let quote = construction::quote_public_ciphertext_dag_construction(manifest, control)?;
    // Pre-sized string/vector LENGTHS also match final lengths. Check every
    // original nested capacity against the manifest BEFORE any write/admission.
    quote.check_constructed(&storage.graph, &storage.trace, control)?;
    guard_final_control(manifest, control)?;
    // Original control remains exclusively borrowed. Admission borrows metadata
    // ONLY; fixed private fill has no control consumer/callback. The next control
    // consumer is validate_constructed's existing check_constructed debit: second
    // actual scan, second charge, never a third charge or a transferable token.
    let admission = quote.admit_trace(ambient, metadata)?;
    fill(&source, bindings, storage, &tables);
    let summary = quote.validate_constructed(admission, &storage.graph, &storage.trace, control)?;
    Ok(Checked { source, bindings, manifest, storage, control, metadata, summary })
}

#[cfg(test)]
#[path = "encrypted_instruction_lowering_tests.rs"]
mod tests;
