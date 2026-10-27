//! CacheMir compiler ANALYSIS ONLY. No ciphertext, key, cache authentication,
//! encrypted append, calibrated error model, refresh or runtime issuer.
//! Supplied capabilities describe explicitly synthetic packed-update kernels.
//! Original graph stays borrowed; cloned records are metadata, never ciphertexts.
use crate::encrypted_level_schedule::{self as schedule, *};
use pllm_core::interleaved_kv_layout::{AppendDescriptor, Geometry};
use pllm_garble::shared_allocation::{OwnedBudget, OwnedExternalCharge};
use schedule::construction::{ControlLedger, MetadataLedger, PublicCiphertextConstructionQuote};
use schedule::dag::{
    PublicEncryptedDagCapacity, PublicEncryptedDagSolution, solve_public_encrypted_dag,
};
use std::mem::size_of;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    Geometry,
    StaleParents,
    Unknown,
    Capability,
    Capacity,
    Work,
    Allocation,
    Trace,
    Consumed,
}

/// Names already declared in ORIGINAL graph. Old arrays are complete, ordered
/// live K tiles / V groups, not merely the currently written tile.
pub struct Parents<'a> {
    pub keys: &'a [&'a str],
    pub values: &'a [&'a str],
}
/// Packed delta input is an ORIGINAL graph input, not clear bytes. Its packing
/// is an external synthetic capability assumption, NOT proved by this compiler.
/// Order is K replacement, then each of head_dim V replacements.
pub struct Replacement<'a> {
    pub incoming: &'a str,
    pub output: &'a PublicCiphertextValue,
    pub owner: &'a PublicCiphertextOwner,
    pub capability: &'a PublicCiphertextCapability,
}

/// Ordered synthetic producer declarations: K, then each V replacement. Both
/// references must be records INSIDE the same borrowed original graph. This is
/// graph membership, not evidence of packing, E3/RoPE/GQA or numeric correctness.
/// Role order is an explicit caller-supplied SYNTHETIC declaration. Neither an
/// operation label nor a method/kernel selector can certify semantic K/V roles.
pub struct ProducedUpdate<'a> {
    pub producer: &'a PublicCiphertextGraphOperation,
    pub capability: &'a PublicCiphertextCapability,
}

struct ProducedAppend<'a> {
    geometry: Geometry,
    append: AppendDescriptor,
    original: &'a PublicCiphertextGraph,
    parents: Parents<'a>,
    replacements: &'a [Replacement<'a>],
    updates: &'a [ProducedUpdate<'a>],
}

/// Non-cloneable, borrowed, one-attempt ANALYSIS metadata handle. Construction
/// performs no scans or allocations; analysis validates under existing ledgers.
/// Consumption prevents replay of THIS handle only. Callers can declare another
/// handle: no response/source/key-epoch/cancellation custody or logical commit.
/// Shared borrows prevent original graph/parent/capability mutation while bound.
///
/// ```compile_fail,E0599
/// use pllm_compiler::interleaved_cache_append::ProducedUpdateBinding;
/// fn duplicate(b: ProducedUpdateBinding<'_>) { let _second = b.clone(); }
/// ```
/// ```compile_fail,E0502
/// use pllm_compiler::{interleaved_cache_append::*, encrypted_level_schedule::*};
/// use pllm_core::interleaved_kv_layout::{Geometry, AppendDescriptor};
/// fn mutate(g: &mut PublicCiphertextGraph, geometry: Geometry, a: AppendDescriptor) {
///     let binding = bind_produced_updates(geometry, a, g,
///         Parents { keys: &[], values: &[] }, &[], &[]);
///     g.operations.clear();
///     drop(binding);
/// }
/// ```
/// Ordinary compile-pass control: the graph can be mutated after the borrowed
/// handle is dropped. This exercises borrow usage, not append qualification.
/// ```
/// use pllm_compiler::{interleaved_cache_append::*, encrypted_level_schedule::*};
/// use pllm_core::interleaved_kv_layout::{Geometry, AppendDescriptor};
/// fn release_then_mutate(g: &mut PublicCiphertextGraph, geometry: Geometry, a: AppendDescriptor) {
///     let binding = bind_produced_updates(geometry, a, g,
///         Parents { keys: &[], values: &[] }, &[], &[]);
///     drop(binding);
///     g.operations.clear();
/// }
/// ```
pub struct ProducedUpdateBinding<'a> {
    pending: Option<ProducedAppend<'a>>,
}

pub fn bind_produced_updates<'a>(
    geometry: Geometry,
    append: AppendDescriptor,
    original: &'a PublicCiphertextGraph,
    parents: Parents<'a>,
    replacements: &'a [Replacement<'a>],
    updates: &'a [ProducedUpdate<'a>],
) -> ProducedUpdateBinding<'a> {
    ProducedUpdateBinding {
        pending: Some(ProducedAppend {
            geometry,
            append,
            original,
            parents,
            replacements,
            updates,
        }),
    }
}

/// Payload first, guard last: all solution/graph storage dies before refund.
/// No partial plan getter or detached ciphertext handle exists.
pub struct AppendAnalysis<'a> {
    graph: PublicCiphertextGraph,
    solution: PublicEncryptedDagSolution,
    original: &'a PublicCiphertextGraph,
    geometry: Geometry,
    append: AppendDescriptor,
    _charge: OwnedExternalCharge,
}
impl AppendAnalysis<'_> {
    pub fn original(&self) -> &PublicCiphertextGraph {
        self.original
    }
    pub fn graph(&self) -> &PublicCiphertextGraph {
        &self.graph
    }
    pub fn solution(&self) -> &PublicEncryptedDagSolution {
        &self.solution
    }
    pub fn descriptor(&self) -> (Geometry, AppendDescriptor) {
        (self.geometry, self.append)
    }
}
fn add(a: usize, b: usize) -> Result<usize, Refusal> {
    a.checked_add(b).ok_or(Refusal::Capacity)
}
fn descriptor(g: Geometry, a: AppendDescriptor) -> Result<(), Refusal> {
    let t = g.tokens_per_key_tile();
    let v = g.tokens_per_value_group();
    if a.token >= g.capacity()
        || a.key_tile != a.token / t
        || a.key_lane != a.token % t
        || a.value_group != a.token / v
        || a.new_key_ciphertexts != usize::from(a.token % t == 0)
        || a.new_value_ciphertexts != if a.token % v == 0 { g.head_dim() } else { 0 }
    {
        return Err(Refusal::Geometry);
    }
    Ok(())
}
fn value<'a>(g: &'a PublicCiphertextGraph, id: &str) -> Result<&'a PublicCiphertextValue, Refusal> {
    let mut found = g
        .inputs
        .iter()
        .chain(g.operations.iter().map(|o| &o.output))
        .filter(|v| v.identity == id);
    let v = found.next().ok_or(Refusal::StaleParents)?;
    if found.next().is_some() {
        return Err(Refusal::StaleParents);
    }
    Ok(v)
}
fn known(state: &PublicCiphertextState) -> bool {
    state.error.bound.is_some()
        && !state.scheme.is_empty()
        && !state.key_domain.is_empty()
        && !state.layout.is_empty()
        && !state.scale.is_empty()
        && !state.error.model_identity.is_empty()
}
fn sum(parts: impl IntoIterator<Item = usize>) -> Result<usize, Refusal> {
    parts.into_iter().try_fold(0, add)
}
fn array<T>(n: usize) -> Result<usize, Refusal> {
    n.checked_mul(size_of::<T>()).ok_or(Refusal::Capacity)
}
fn state_bytes(s: &PublicCiphertextState) -> Result<usize, Refusal> {
    sum([
        s.scheme.capacity(),
        s.key_domain.capacity(),
        s.layout.capacity(),
        s.scale.capacity(),
        s.error.model_identity.capacity(),
        array::<u64>(s.shape.capacity())?,
    ])
}
fn value_bytes(v: &PublicCiphertextValue) -> Result<usize, Refusal> {
    sum([
        v.identity.capacity(),
        v.owner.capacity(),
        state_bytes(&v.state)?,
    ])
}
fn cap_bytes(c: &PublicCiphertextCapability) -> Result<usize, Refusal> {
    let mut n = sum([
        c.identity.capacity(),
        c.kernel_identity.capacity(),
        array::<PublicCiphertextState>(c.inputs.capacity())?,
        state_bytes(&c.output)?,
        c.error_assumption.as_ref().map_or(0, String::capacity),
    ])?;
    for s in &c.inputs {
        n = add(n, state_bytes(s)?)?;
    }
    let costs = match &c.execution {
        PublicCiphertextExecution::Ckks {
            compute,
            transition,
            ..
        } => sum([
            compute.cost.identity.capacity(),
            transition.cost.identity.capacity(),
        ])?,
        PublicCiphertextExecution::View { cost } => cost.identity.capacity(),
        PublicCiphertextExecution::Hybrid { round_trip } => sum([
            round_trip.identity.capacity(),
            round_trip.entry.identity.capacity(),
            round_trip.evaluate.identity.capacity(),
            round_trip.recover.identity.capacity(),
        ])?,
    };
    add(n, costs)
}
fn graph_bytes(g: &PublicCiphertextGraph) -> Result<usize, Refusal> {
    let mut n = sum([
        size_of::<PublicCiphertextGraph>(),
        g.cost_model_identity.capacity(),
        g.cost_unit.capacity(),
        array::<PublicCiphertextOwner>(g.owners.capacity())?,
        array::<PublicCiphertextValue>(g.inputs.capacity())?,
        array::<PublicCiphertextGraphOperation>(g.operations.capacity())?,
        array::<PublicCiphertextCapability>(g.capabilities.capacity())?,
        array::<String>(g.persistent_owners.capacity())?,
        array::<String>(g.outputs.capacity())?,
        array::<String>(g.retained_inputs.capacity())?,
    ])?;
    for o in &g.owners {
        n = add(n, o.identity.capacity())?;
    }
    for s in g
        .persistent_owners
        .iter()
        .chain(&g.outputs)
        .chain(&g.retained_inputs)
    {
        n = add(n, s.capacity())?;
    }
    for v in &g.inputs {
        n = add(n, value_bytes(v)?)?;
    }
    for op in &g.operations {
        n = add(
            n,
            sum([
                op.identity.capacity(),
                op.kernel_identity.capacity(),
                array::<String>(op.inputs.capacity())?,
                value_bytes(&op.output)?,
            ])?,
        )?;
        for s in &op.inputs {
            n = add(n, s.capacity())?;
        }
    }
    for c in &g.capabilities {
        n = add(n, cap_bytes(c)?)?;
    }
    Ok(n)
}
fn factory_bytes(
    g: &PublicCiphertextGraph,
    p: &Parents<'_>,
    r: &[Replacement<'_>],
) -> Result<usize, Refusal> {
    let mut n = graph_bytes(g)?;
    for id in p.keys.iter().chain(p.values.iter()) {
        n = add(n, add(size_of::<String>(), id.len())?)?;
        n = add(n, add(size_of::<String>(), id.len())?)?;
    }
    for x in r {
        n = add(
            n,
            sum([
                size_of::<PublicCiphertextGraphOperation>(),
                size_of::<PublicCiphertextOwner>(),
                size_of::<PublicCiphertextCapability>(),
                size_of::<String>(),
                size_of::<Replacement<'_>>(),
                size_of::<ProducedUpdate<'_>>(),
                x.owner.identity.capacity(),
                value_bytes(x.output)?,
                cap_bytes(x.capability)?,
                x.output.identity.capacity(),
                x.capability.kernel_identity.capacity(),
                array::<String>(2)?,
                x.incoming.len(),
                x.output.identity.capacity(),
            ])?,
        )?;
        // At most one retained parent name in addition to the incoming delta.
        n = add(
            n,
            g.inputs
                .iter()
                .chain(g.operations.iter().map(|o| &o.output))
                .map(|v| v.identity.capacity())
                .max()
                .unwrap_or(0),
        )?;
    }
    // Named phases: retained upstream original/binding inventory, destination
    // clone, and old/new outer Vec relocation overlap. No allocator/RSS claim.
    sum([
        n,
        n,
        n,
        size_of::<Result<AppendAnalysis<'_>, Refusal>>(),
        size_of::<OwnedExternalCharge>(),
        size_of::<PublicEncryptedDagSolution>(),
        size_of::<ProducedUpdateBinding<'_>>(),
        size_of::<ProducedAppend<'_>>(),
    ])
}
fn preserve(cap: &PublicCiphertextCapability) -> bool {
    match &cap.execution {
        PublicCiphertextExecution::Ckks {
            compute,
            depth,
            transition,
        } => {
            *depth == 0
                && compute.input_level == cap.output.level
                && compute.cost.units.is_some()
                && transition.kind == PublicLevelTransitionKind::Preserve
                && transition.from_level == cap.output.level
                && transition.to_level == cap.output.level
                && transition.cost.units == Some(0)
        }
        // Replacement is NOT a view, hybrid recovery, refresh or free alias.
        _ => false,
    }
}
fn kernel_geometry(name: &str, g: Geometry, a: AppendDescriptor, index: usize) -> bool {
    // Public external assumption naming, NOT a registered/implemented kernel.
    // Bind replacement role and exact logical insertion; don't reuse a stale
    // manifest merely because shape/level/scale happen to match.
    let mut parts = name.split(':');
    parts.next() == Some("cachemir_append_v1")
        && [
            g.slots(),
            g.heads(),
            g.head_dim(),
            g.capacity(),
            a.token,
            index,
        ]
        .into_iter()
        .all(|n| parts.next().and_then(|s| s.parse::<usize>().ok()) == Some(n))
        && parts.next().is_none()
}

/// Header-only bound, before nested visits/string comparisons/copying. R is an
/// independently supplied finite record ceiling, not a measured requirement.
/// The fixed linear term covers bounded strings/copy walks; quadratic header
/// pairs cover parent/identity joins. Ledger admission, not this number, permits
/// those logical control visits. It is not CPU fuel or the DAG work quantum.
fn control_allowance(
    old: &PublicCiphertextGraph,
    p: &Parents<'_>,
    r: &[Replacement<'_>],
    records: usize,
) -> Result<usize, Refusal> {
    let h = sum([
        old.owners.len(),
        old.persistent_owners.len(),
        old.inputs.len(),
        old.operations.len(),
        old.capabilities.len(),
        old.outputs.len(),
        old.retained_inputs.len(),
        p.keys.len(),
        p.values.len(),
        r.len(),
    ])?;
    if h > records || records == 0 {
        return Err(Refusal::Capacity);
    }
    sum([
        4096,
        records.checked_mul(128).ok_or(Refusal::Capacity)?,
        h.checked_mul(h)
            .and_then(|n| n.checked_mul(128))
            .ok_or(Refusal::Capacity)?,
    ])
}
fn bounded_metadata(
    old: &PublicCiphertextGraph,
    p: &Parents<'_>,
    r: &[Replacement<'_>],
    limit: usize,
) -> Result<(), Refusal> {
    fn id(s: &str) -> Result<(), Refusal> {
        if s.is_empty() || s.len() > 64 {
            Err(Refusal::Capacity)
        } else {
            Ok(())
        }
    }
    fn consume(n: &mut usize, count: usize) -> Result<(), Refusal> {
        *n = n.checked_sub(count).ok_or(Refusal::Capacity)?;
        Ok(())
    }
    fn state(s: &PublicCiphertextState, n: &mut usize) -> Result<(), Refusal> {
        consume(n, add(1, s.shape.len())?)?;
        if !(1..=2).contains(&s.shape.len()) {
            return Err(Refusal::Capacity);
        }
        for text in [
            &s.scheme,
            &s.key_domain,
            &s.layout,
            &s.scale,
            &s.error.model_identity,
        ] {
            id(text)?;
        }
        Ok(())
    }
    fn val(v: &PublicCiphertextValue, n: &mut usize) -> Result<(), Refusal> {
        id(&v.identity)?;
        id(&v.owner)?;
        state(&v.state, n)
    }
    fn cap(c: &PublicCiphertextCapability, n: &mut usize) -> Result<(), Refusal> {
        consume(n, add(1, c.inputs.len())?)?;
        id(&c.identity)?;
        id(&c.kernel_identity)?;
        for s in &c.inputs {
            state(s, n)?;
        }
        state(&c.output, n)?;
        if let Some(s) = &c.error_assumption {
            id(s)?;
        }
        match &c.execution {
            PublicCiphertextExecution::Ckks {
                compute,
                transition,
                ..
            } => {
                id(&compute.cost.identity)?;
                id(&transition.cost.identity)?;
            }
            PublicCiphertextExecution::Hybrid { round_trip: h } => {
                for s in [
                    &h.identity,
                    &h.entry.identity,
                    &h.evaluate.identity,
                    &h.recover.identity,
                ] {
                    id(s)?;
                }
            }
            PublicCiphertextExecution::View { cost } => id(&cost.identity)?,
        }
        Ok(())
    }
    let mut remaining = limit;
    consume(
        &mut remaining,
        sum([
            old.owners.len(),
            old.persistent_owners.len(),
            old.inputs.len(),
            old.operations.len(),
            old.capabilities.len(),
            old.outputs.len(),
            old.retained_inputs.len(),
            p.keys.len(),
            p.values.len(),
            r.len(),
        ])?,
    )?;
    id(&old.cost_model_identity)?;
    id(&old.cost_unit)?;
    for o in &old.owners {
        id(&o.identity)?;
    }
    for s in old
        .persistent_owners
        .iter()
        .chain(&old.outputs)
        .chain(&old.retained_inputs)
    {
        id(s)?;
    }
    for v in &old.inputs {
        val(v, &mut remaining)?;
    }
    for op in &old.operations {
        consume(&mut remaining, op.inputs.len())?;
        id(&op.identity)?;
        id(&op.kernel_identity)?;
        for s in &op.inputs {
            id(s)?;
        }
        val(&op.output, &mut remaining)?;
    }
    for c in &old.capabilities {
        cap(c, &mut remaining)?;
    }
    for s in p.keys.iter().chain(p.values.iter()) {
        id(s)?;
    }
    for x in r {
        id(x.incoming)?;
        id(&x.owner.identity)?;
        val(x.output, &mut remaining)?;
        cap(x.capability, &mut remaining)?;
    }
    Ok(())
}

/// Structural preflight, allocation-free. Descriptor is publicly forgeable
/// metadata and never becomes authenticated cache/source ownership.
fn preflight(
    g: Geometry,
    a: AppendDescriptor,
    old: &PublicCiphertextGraph,
    p: &Parents<'_>,
    r: &[Replacement<'_>],
) -> Result<(), Refusal> {
    preflight_sources(g, a, old, p, r, false)
}

fn preflight_sources(
    g: Geometry,
    a: AppendDescriptor,
    old: &PublicCiphertextGraph,
    p: &Parents<'_>,
    r: &[Replacement<'_>],
    produced: bool,
) -> Result<(), Refusal> {
    descriptor(g, a)?;
    if p.keys.len()
        != g.key_layout()
            .ciphertexts(a.token)
            .map_err(|_| Refusal::Geometry)?
        || p.values.len()
            != g.value_layout()
                .ciphertexts(a.token)
                .map_err(|_| Refusal::Geometry)?
        || r.len() != add(g.head_dim(), 1)?
    {
        return Err(Refusal::StaleParents);
    }
    for (i, id) in p.keys.iter().chain(p.values.iter()).enumerate() {
        if p.keys
            .iter()
            .chain(p.values.iter())
            .take(i)
            .any(|other| other == id)
        {
            return Err(Refusal::StaleParents);
        }
        let v = value(old, id)?;
        if !known(&v.state) || v.state.shape.as_slice() != [g.slots() as u64] {
            return Err(Refusal::Unknown);
        }
        if !old
            .owners
            .iter()
            .any(|o| o.identity == v.owner && o.bytes.is_some_and(|b| b > 0))
        {
            return Err(Refusal::Unknown);
        }
    }
    for (i, record) in r.iter().enumerate() {
        let incoming = value(old, record.incoming)?;
        if !produced && !old.inputs.iter().any(|v| std::ptr::eq(v, incoming)) {
            return Err(Refusal::StaleParents);
        }
        let parent = if i == 0 {
            if a.new_key_ciphertexts == 0 {
                Some(p.keys[a.key_tile])
            } else {
                None
            }
        } else if a.new_value_ciphertexts == 0 {
            Some(p.values[a.value_group * g.head_dim() + i - 1])
        } else {
            None
        };
        let cap = record.capability;
        if !known(&incoming.state)
            || !known(&record.output.state)
            || record.output.state.shape.as_slice() != [g.slots() as u64]
            || record.owner.bytes.is_none_or(|b| b == 0)
            || record.output.owner != record.owner.identity
        {
            return Err(Refusal::Unknown);
        }
        if cap.inputs.iter().any(|s| {
            s.scheme != cap.output.scheme
                || s.key_domain != cap.output.key_domain
                || s.scale != cap.output.scale
                || s.level != cap.output.level
                || s.layout != cap.output.layout
                || s.shape != cap.output.shape
                || s.error.model_identity != cap.output.error.model_identity
        }) || !kernel_geometry(&cap.kernel_identity, g, a, i)
            || cap.primitive != PublicPrimitive::Arithmetic
            || cap.output != record.output.state
            || !preserve(cap)
            || cap.scratch_bytes.is_none()
            || cap.recovery_overlap_bytes.is_none()
            || cap.error_assumption.as_ref().is_none_or(|s| s.is_empty())
            || cap.inputs.len() != 1 + usize::from(parent.is_some())
        {
            return Err(Refusal::Capability);
        }
        if let Some(id) = parent {
            if cap.join != PublicCiphertextJoin::Aligned
                || cap.inputs[0] != value(old, id)?.state
                || cap.inputs[1] != incoming.state
            {
                return Err(Refusal::Capability);
            }
        } else if cap.join != PublicCiphertextJoin::Unary || cap.inputs[0] != incoming.state {
            return Err(Refusal::Capability);
        }
        if old
            .owners
            .iter()
            .any(|o| o.identity == record.owner.identity)
            || old
                .inputs
                .iter()
                .chain(old.operations.iter().map(|o| &o.output))
                .any(|v| v.identity == record.output.identity)
            || r[..i].iter().any(|other| {
                other.owner.identity == record.owner.identity
                    || other.output.identity == record.output.identity
            })
        {
            return Err(Refusal::StaleParents);
        }
    }
    Ok(())
}

// Deliberately narrow synthetic producer qualification, not a second DAG
// planner: known preserving arithmetic CKKS, unary/aligned typed joins. Other
// execution families need their own explicit qualification, not free authority.
fn produced_preflight(bound: &ProducedAppend<'_>) -> Result<(), Refusal> {
    let old = bound.original;
    if bound.updates.len() != bound.replacements.len() {
        return Err(Refusal::StaleParents);
    }
    for (i, update) in bound.updates.iter().enumerate() {
        let op = update.producer;
        let cap = update.capability;
        if !old.operations.iter().any(|p| std::ptr::eq(p, op))
            || !old.capabilities.iter().any(|c| std::ptr::eq(c, cap))
            || bound.updates[..i]
                .iter()
                .any(|u| std::ptr::eq(u.producer, op))
            || op.output.identity != bound.replacements[i].incoming
            || !std::ptr::eq(value(old, &op.output.identity)?, &op.output)
            || old
                .operations
                .iter()
                .filter(|p| p.identity == op.identity)
                .count()
                != 1
        {
            return Err(Refusal::StaleParents);
        }
    }
    // Check complete upstream inventory before factory entry, including dormant
    // alternatives. The shared solver/validator remains final legality authority.
    for cap in &old.capabilities {
        if !known(&cap.output)
            || cap.inputs.iter().any(|s| !known(s))
            || cap.scratch_bytes.is_none()
            || cap.recovery_overlap_bytes.is_none()
            || cap.error_assumption.as_ref().is_none_or(|s| s.is_empty())
            || !preserve(cap)
        {
            return Err(Refusal::Unknown);
        }
        if cap.primitive != PublicPrimitive::Arithmetic
            || !(cap.join == PublicCiphertextJoin::Unary && cap.inputs.len() == 1
                || cap.join == PublicCiphertextJoin::Aligned && cap.inputs.len() >= 2)
            || cap.inputs.iter().any(|s| {
                s.scheme != cap.output.scheme
                    || s.key_domain != cap.output.key_domain
                    || s.scale != cap.output.scale
                    || s.layout != cap.output.layout
                    || s.shape != cap.output.shape
                    || s.level != cap.output.level
            })
        {
            return Err(Refusal::Capability);
        }
    }
    for declaration in old
        .inputs
        .iter()
        .chain(old.operations.iter().map(|o| &o.output))
    {
        let output = value(old, &declaration.identity)?;
        if !known(&output.state)
            || old
                .owners
                .iter()
                .filter(|o| o.identity == output.owner)
                .count()
                != 1
            || !old
                .owners
                .iter()
                .any(|o| o.identity == output.owner && o.bytes.is_some_and(|b| b > 0))
        {
            return Err(Refusal::Unknown);
        }
    }
    for op in &old.operations {
        let matches =
            |cap: &PublicCiphertextCapability| {
                cap.kernel_identity == op.kernel_identity
                    && cap.primitive == op.primitive
                    && cap.output == op.output.state
                    && cap.inputs.len() == op.inputs.len()
                    && op.inputs.iter().zip(&cap.inputs).all(|(id, s)| {
                        value(old, id).is_ok_and(|v| known(&v.state) && v.state == *s)
                    })
            };
        if !old.capabilities.iter().any(matches)
            || bound
                .updates
                .iter()
                .any(|u| std::ptr::eq(u.producer, op) && !matches(u.capability))
        {
            return Err(Refusal::Capability);
        }
    }
    preflight_sources(
        bound.geometry,
        bound.append,
        old,
        &bound.parents,
        bound.replacements,
        true,
    )
}

// Length-only quote AFTER structural membership/metadata checks. The baseline
// allowance covers this update-header walk; never inspect foreign headers here.
fn produced_matching_work(bound: &ProducedAppend<'_>) -> Result<usize, Refusal> {
    let mul = |a: usize, b: usize| a.checked_mul(b).ok_or(Refusal::Capacity);
    let u = bound.updates.len();
    let c = add(bound.original.capabilities.len(), bound.replacements.len())?;
    let v = add(bound.original.inputs.len(), bound.original.operations.len())?;
    let s = bound
        .updates
        .iter()
        .try_fold(0, |n, update| add(n, update.producer.inputs.len()))?;
    // T=65: length + 64 byte pairs; E=332: entry + five T + level,
    // shape (at most two entries) + error option. Source probe <=67;
    // each input <=336+67V; candidate overhead/output <=415. These
    // envelopes include chain/filter/zip visits and final pointer/next checks.
    add(
        mul(16, u)?,
        mul(c, add(mul(415, u)?, mul(add(336, mul(67, v)?)?, s)?)?)?,
    )
}

fn unique_produced_capabilities(bound: &ProducedAppend<'_>) -> Result<(), Refusal> {
    let old = bound.original;
    // Emission preserves original operations and their input states; the source
    // preflight above forbids replacement outputs from shadowing original values.
    // Match the solver's complete emitted inventory, including appended records.
    // Occurrences count separately even when replacement borrows the same cap.
    for update in bound.updates {
        let mut matching = old
            .capabilities
            .iter()
            .chain(bound.replacements.iter().map(|r| r.capability))
            .filter(|cap| {
                #[cfg(test)]
                MATCH_PREDICATES.with(|n| n.set(n.get() + 1));
                schedule::dag::matches(old, update.producer, cap)
            });
        if !matching
            .next()
            .is_some_and(|cap| std::ptr::eq(cap, update.capability))
            || matching.next().is_some()
        {
            return Err(Refusal::Capability);
        }
    }
    Ok(())
}

/// The manifest quote MUST cover original graph/binding/capability capacities,
/// clone/old-new/control slots and the actual final graph + chosen trace. It is
/// borrowed evidence, not allocator authority: the independent original Budget
/// charges the COMPLETE overlap before graph cloning or DAG search.
/// Search/trace envelopes are independently supplied finite grants; no limit is
/// treated as free memory or an actual kernel measurement. A short grant fails.
#[allow(clippy::too_many_arguments)]
pub fn analyze_append<'a>(
    geometry: Geometry,
    append: AppendDescriptor,
    original: &'a PublicCiphertextGraph,
    parents: &Parents<'_>,
    replacements: &[Replacement<'_>],
    quote: &PublicCiphertextConstructionQuote<'_>,
    budget: &OwnedBudget,
    metadata: &mut MetadataLedger,
    control: &mut ControlLedger<'_>,
    capacity: PublicEncryptedDagCapacity,
) -> Result<AppendAnalysis<'a>, Refusal> {
    analyze_append_inner(
        geometry,
        append,
        original,
        parents,
        replacements,
        None,
        quote,
        budget,
        metadata,
        control,
        capacity,
    )
}

/// Consume binding before ANY fallible admission. Failed attempts remain spent;
/// successful result borrows original graph, and drops graph/solution before its
/// overlap charge. Insertion is exactly [descriptor.token, descriptor.token+1).
pub fn analyze_produced_append<'a>(
    binding: &mut ProducedUpdateBinding<'a>,
    quote: &PublicCiphertextConstructionQuote<'_>,
    budget: &OwnedBudget,
    metadata: &mut MetadataLedger,
    control: &mut ControlLedger<'_>,
    capacity: PublicEncryptedDagCapacity,
) -> Result<AppendAnalysis<'a>, Refusal> {
    let bound = binding.pending.take().ok_or(Refusal::Consumed)?;
    analyze_append_inner(
        bound.geometry,
        bound.append,
        bound.original,
        &bound.parents,
        bound.replacements,
        Some(&bound),
        quote,
        budget,
        metadata,
        control,
        capacity,
    )
}

#[allow(clippy::too_many_arguments)]
fn analyze_append_inner<'a>(
    geometry: Geometry,
    append: AppendDescriptor,
    original: &'a PublicCiphertextGraph,
    parents: &Parents<'_>,
    replacements: &[Replacement<'_>],
    produced: Option<&ProducedAppend<'_>>,
    quote: &PublicCiphertextConstructionQuote<'_>,
    budget: &OwnedBudget,
    metadata: &mut MetadataLedger,
    control: &mut ControlLedger<'_>,
    capacity: PublicEncryptedDagCapacity,
) -> Result<AppendAnalysis<'a>, Refusal> {
    let visits = control_allowance(original, parents, replacements, capacity.trace.max_records)?;
    control
        .core_work()
        .debit(0, visits)
        .map_err(|_| Refusal::Work)?;
    if capacity.max_work == 0 {
        return Err(Refusal::Work);
    }
    // The SAME original cumulative metadata ledger prepays full caller search
    // allowance plus one final trace validation. Never create/reset a ledger.
    metadata
        .prepay_work(capacity.max_work)
        .map_err(|_| Refusal::Work)?;
    let trace_admission = quote
        .admit_trace(Some(capacity.trace), metadata)
        .map_err(|_| Refusal::Work)?;
    bounded_metadata(original, parents, replacements, capacity.trace.max_records)?;
    if let Some(bound) = produced {
        produced_preflight(bound)?;
        let matching_work = produced_matching_work(bound)?;
        control
            .core_work()
            .debit(0, matching_work)
            .map_err(|_| Refusal::Work)?;
        unique_produced_capabilities(bound)?;
    } else {
        preflight(geometry, append, original, parents, replacements)?;
    }
    let n = add(original.operations.len(), replacements.len())?;
    if n > capacity.max_operations || capacity.max_states == 0 || capacity.max_frontier == 0 {
        return Err(Refusal::Capacity);
    }
    let factory = factory_bytes(original, parents, replacements)?;
    let bytes = owner_bytes(quote, capacity, factory)?;
    let charge = budget
        .charge_external(bytes)
        .map_err(|_| Refusal::Capacity)?;
    let graph = emit_graph(geometry, append, original, parents, replacements)?;
    if graph_bytes(&graph)? > factory {
        return Err(Refusal::Capacity);
    }
    let solution = solve_public_encrypted_dag(&graph, capacity).map_err(|_| Refusal::Trace)?;
    // Shared checker verifies actual original capacities against the prepaid
    // immutable manifest, then invokes validate_public_ciphertext_trace itself.
    quote
        .validate_constructed(trace_admission, &graph, &solution.ledger.trace, control)
        .map_err(|_| Refusal::Trace)?;
    Ok(AppendAnalysis {
        graph,
        solution,
        original,
        geometry,
        append,
        _charge: charge,
    })
}
fn owner_bytes(
    quote: &PublicCiphertextConstructionQuote<'_>,
    capacity: PublicEncryptedDagCapacity,
    factory: usize,
) -> Result<usize, Refusal> {
    // Caller grants <=64 MiB WORKING ownership. 1 MiB output is a separate
    // qualification limit, not a substituted solver metadata-buffer ceiling.
    let mut bytes = add(quote.metadata_peak(), quote.owner_peak())?;
    bytes = add(bytes, factory)?;
    bytes = add(bytes, capacity.max_buffer_bytes)?;
    bytes = add(bytes, capacity.trace.max_buffer_bytes)?;
    bytes = add(
        bytes,
        usize::try_from(capacity.trace.max_live_owner_bytes).map_err(|_| Refusal::Capacity)?,
    )?;
    bytes = add(
        bytes,
        add(
            size_of::<AppendAnalysis<'_>>(),
            size_of::<PublicCiphertextGraph>(),
        )?,
    )?;
    if bytes > 64 << 20 {
        return Err(Refusal::Capacity);
    }
    Ok(bytes)
}

fn emit_graph(
    geometry: Geometry,
    append: AppendDescriptor,
    original: &PublicCiphertextGraph,
    parents: &Parents<'_>,
    replacements: &[Replacement<'_>],
) -> Result<PublicCiphertextGraph, Refusal> {
    #[cfg(test)]
    GRAPH_FACTORIES.with(|n| n.set(n.get() + 1));
    let mut graph = original.clone();
    graph
        .operations
        .try_reserve_exact(replacements.len())
        .map_err(|_| Refusal::Allocation)?;
    graph
        .owners
        .try_reserve_exact(replacements.len())
        .map_err(|_| Refusal::Allocation)?;
    graph
        .capabilities
        .try_reserve_exact(replacements.len())
        .map_err(|_| Refusal::Allocation)?;
    graph
        .outputs
        .try_reserve_exact(add(
            replacements.len(),
            add(parents.keys.len(), parents.values.len())?,
        )?)
        .map_err(|_| Refusal::Allocation)?;
    graph
        .retained_inputs
        .try_reserve_exact(add(parents.keys.len(), parents.values.len())?)
        .map_err(|_| Refusal::Allocation)?;
    // Retain BOTH complete parent families, including unchanged groups. Keeping
    // old outputs is a conservative analysis liveness constraint, not commit.
    for id in parents.keys.iter().chain(parents.values.iter()) {
        if original.inputs.iter().any(|v| v.identity == *id)
            && !graph.retained_inputs.iter().any(|v| v == id)
        {
            graph.retained_inputs.push((*id).to_owned());
        }
        if !graph.outputs.iter().any(|v| v == id) {
            graph.outputs.push((*id).to_owned());
        }
    }
    for (i, r) in replacements.iter().enumerate() {
        let parent = if i == 0 {
            if append.new_key_ciphertexts == 0 {
                Some(parents.keys[append.key_tile])
            } else {
                None
            }
        } else if append.new_value_ciphertexts == 0 {
            Some(parents.values[append.value_group * geometry.head_dim() + i - 1])
        } else {
            None
        };
        let mut inputs = Vec::new();
        inputs
            .try_reserve_exact(1 + usize::from(parent.is_some()))
            .map_err(|_| Refusal::Allocation)?;
        if let Some(parent) = parent {
            inputs.push(parent.to_owned());
        }
        inputs.push(r.incoming.to_owned());
        graph.owners.push(r.owner.clone());
        graph.capabilities.push(r.capability.clone());
        graph.operations.push(PublicCiphertextGraphOperation {
            identity: r.output.identity.clone(),
            kernel_identity: r.capability.kernel_identity.clone(),
            primitive: PublicPrimitive::Arithmetic,
            inputs,
            output: r.output.clone(),
        });
        graph.outputs.push(r.output.identity.clone());
    }
    Ok(graph)
}

#[cfg(test)]
thread_local! {static GRAPH_FACTORIES:std::cell::Cell<usize>=const{std::cell::Cell::new(0)};}
#[cfg(test)]
thread_local! {static MATCH_PREDICATES:std::cell::Cell<usize>=const{std::cell::Cell::new(0)};}
#[cfg(test)]
#[path = "interleaved_cache_append_tests.rs"]
mod tests;
