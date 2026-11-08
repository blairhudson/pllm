use super::*;
use pllm_models::{DecoderExpansionBudget, DecoderGraph, DecoderWorkload, ModelOperator as Op};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// Pure aggregate-energy algebra only. No production caller or certificate issuer.
#[allow(dead_code)]
#[path = "row_energy.rs"]
mod row_energy;

#[path = "private_matrix_driver.rs"]
pub(super) mod private_matrix_driver;

#[path = "graph.rs"]
mod graph;

#[path = "request_driver.rs"]
mod request_driver;

#[path = "request_acceptance.rs"]
mod request_acceptance;

#[allow(dead_code)]
#[path = "request_origin.rs"]
mod request_origin;

#[path = "source_entry.rs"]
pub(super) mod source_entry;

/// Only this module can create the actual token-input emission event.
pub(super) struct CreatedRequestInput {
    owner: usize,
    rows: usize,
    bounds: Interval,
}

impl CreatedRequestInput {
    pub(super) fn parts(self) -> (usize, usize, Interval) {
        (self.owner, self.rows, self.bounds)
    }
}

// Shared canonical vocabulary also serves sealed compact recipe publication.
// The candidate finalizer remains comparison-only: `allocated` is not the full
// retained-owner baseline. Production uses the original Builder::finish loop;
// candidate adoption still requires baseline, identity and new-owner admission.
#[allow(dead_code)]
#[path = "schedule_finalization.rs"]
mod schedule_finalization;

// Compiler-private shared numeric sinks and compact output publication.
// Used by the existing bounded sealed compact request route; this does not
// qualify separate publication drafts or establish complete-program admission.
#[allow(dead_code)]
#[path = "compact_lowering.rs"]
pub(super) mod compact_lowering;

#[cfg(test)]
#[path = "linear_tests.rs"]
mod linear_tests;

#[cfg(test)]
#[path = "program_graph_tests.rs"]
mod graph_tests;

#[cfg(test)]
#[path = "program_prepared_tensor_tests.rs"]
mod prepared_tensor_tests;

// Fixture-only exact Manifest::compile capture; no production numeric sink change.
#[cfg(test)]
#[path = "program_manifest_prepared_fixture.rs"]
pub(super) mod manifest_prepared_fixture;

/// Genuine expanded-worker fixture for staged, consuming one-loop adapters.
/// Only the closed rescale18 factory constructs templates; publication borrows
/// that same capability, and the ordinary finalizer owns identity and retirement.
/// No default graph lowering or production admission uses this seam.
#[cfg(test)]
pub(super) fn prepared_rescale18_worker_fixture(
    lanes: usize,
    bounds: Interval,
) -> Result<
    (
        Arc<Program>,
        super::prepared_tensor_map::PreparedTensorMap,
        usize,
        usize,
    ),
    Error,
> {
    use super::prepared_tensor_map::PreparedTensorMap;

    if !matches!(lanes, 1 | 128 | 129 | 257) {
        return Err(Error::Bounds);
    }
    let mut builder = Builder {
        public_weights: true,
        ..Builder::default()
    };
    let input = builder.push(Code::Input("local".into()), lanes, bounds)?;
    let imports = builder.nodes.len();
    let prepared = PreparedTensorMap::rescale(lanes, bounds, 18, builder.allocated, |peak| {
        if peak > MAX_PROGRAM_BYTES {
            Err(Error::Bounds)
        } else {
            Ok(())
        }
    })?;
    let output = builder.prepared_tensor_map(input, &prepared)?;
    // advance_before requires an existing original node at the interval end.
    // This ordinary local identity is outside the replaced loop. Its dead result
    // retires normally; ONLY the original loop output is retained for export.
    builder.affine(output, None, affine(1, 0, 0))?;
    let selected = (0..lanes).map(|index| (output, index)).collect();
    let program = builder.finish(
        imports,
        selected,
        &mut AdmissionSite::default(),
        |h| {
            h.update(b"pllm.replicated-decoder.prepared-rescale18-worker-fixture.v1");
            Ok(())
        },
        || source_binding(&BTreeMap::new()),
    )?;
    Ok((Arc::new(program), prepared, input, output))
}

#[cfg(test)]
type ProtectedPreparedFixture = (
    Arc<Program>,
    super::prepared_tensor_map::PreparedTensorMap,
    usize,
    usize,
    Option<usize>,
);

/// Extended qualification fixture only. Preserve the plain fixture above and
/// use the same original compiler publication/finalizer for every variant.
#[cfg(test)]
pub(super) fn prepared_rescale18_protected_worker_fixture(
    lanes: usize,
    bounds: Interval,
    retain_input: bool,
    sticky: bool,
) -> Result<ProtectedPreparedFixture, Error> {
    use super::prepared_tensor_map::PreparedTensorMap;
    if !matches!(lanes, 1 | 128 | 129 | 257) {
        return Err(Error::Bounds);
    }
    let mut builder = Builder {
        public_weights: true,
        ..Builder::default()
    };
    let input = builder.push(Code::Input("local".into()), lanes, bounds)?;
    let imports = builder.nodes.len();
    if sticky {
        // A real scalar prefix check creates original sticky validity. Keep the
        // map on the original input, so final masking cannot pass vacuously by
        // mapping only the prefix's zero fallback. No private branch/routing.
        builder.checked_bound = Some(ONE / 2);
        builder.checked_view(&[(input, 0)].into_iter().collect())?;
    }
    let validity = builder.validity;
    let prepared = PreparedTensorMap::rescale(lanes, bounds, 18, builder.allocated, |peak| {
        if peak > MAX_PROGRAM_BYTES {
            Err(Error::Bounds)
        } else {
            Ok(())
        }
    })?;
    let output = builder.prepared_tensor_map(input, &prepared)?;
    // Genuine ordinary protected suffix: must consume the restored source
    // ordinal/AES schedule, not merely run a local identity after restoration.
    let product = builder.product(output, output, None, false)?;
    let final_owner = if let Some(valid) = validity {
        let flags = builder.gather(&std::iter::repeat_n((valid, 0), lanes).collect())?;
        builder.product(product, flags, None, false)?
    } else {
        product
    };
    let mut selected: Vec<_> = (0..lanes).map(|i| (final_owner, i)).collect();
    if let Some(valid) = validity {
        selected.push((valid, 0));
    }
    if retain_input {
        selected.push((input, 0));
    }
    let program = builder.finish(
        imports,
        selected,
        &mut AdmissionSite::default(),
        |h| {
            h.update(b"pllm.replicated-decoder.prepared-rescale18-protected-worker-fixture.v1");
            h.update([u8::from(retain_input), u8::from(sticky)]);
            Ok(())
        },
        || source_binding(&BTreeMap::new()),
    )?;
    Ok((Arc::new(program), prepared, input, output, validity))
}

#[cfg(test)]
#[path = "body_tests.rs"]
mod body_tests;

#[cfg(test)]
#[path = "body_integration_tests.rs"]
mod body_integration_tests;

#[path = "ranges.rs"]
mod ranges;
#[cfg(test)]
#[path = "ranges_tests.rs"]
mod ranges_tests;
#[cfg(test)]
#[path = "range_view_metadata_tests.rs"]
mod range_view_metadata_tests;

#[path = "attention_energy.rs"]
mod attention_energy;
#[path = "lookup_builder.rs"]
mod lookup_builder;
#[cfg(test)]
#[path = "lookup_integration_tests.rs"]
mod lookup_integration_tests;
#[cfg(test)]
#[path = "selection_integration_tests.rs"]
mod selection_integration_tests;
#[cfg(test)]
#[path = "trie_integration_tests.rs"]
mod trie_integration_tests;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceTensor {
    pub shape: Vec<usize>,
    pub lower: i64,
    pub upper: i64,
    /// D10 explicitly publishes normalization coefficients. Other tensors stay
    /// party-local; the compiler rejects public bytes in their descriptors.
    pub public_q18: Option<Vec<i64>>,
    /// Commitment to the original encoded source bytes, checked by the issuer.
    pub digest: [u8; 32],
    /// Maximum exact encoded absolute row sum for a public matrix. Recomputed
    /// by every importing worker; absent from private-weight descriptors.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub row_l1: Option<u64>,
    /// Maximum sum of squared encoded coefficients in one physical matrix row.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub row_squared_norm: Option<u128>,
    /// Sum of squared absolute row sums over this complete public matrix.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub row_l1_squared_sum: Option<u128>,
}

/// Bind the validated source metadata without another serialized allocation.
/// Public coefficient bytes are already checked against their encoded digest.
pub(super) fn source_binding(sources: &BTreeMap<String, SourceTensor>) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"pllm.replicated-decoder.source-descriptors.v1");
    h.update((sources.len() as u64).to_le_bytes());
    for (key, source) in sources {
        h.update((key.len() as u64).to_le_bytes());
        h.update(key);
        h.update((source.shape.len() as u64).to_le_bytes());
        for dimension in &source.shape {
            h.update((*dimension as u64).to_le_bytes());
        }
        h.update(source.lower.to_le_bytes());
        h.update(source.upper.to_le_bytes());
        h.update(source.digest);
        h.update([
            source.public_q18.is_some() as u8,
            source.row_l1.is_some() as u8,
            source.row_squared_norm.is_some() as u8,
        ]);
        if let Some(sum) = source.row_l1 {
            h.update(sum.to_le_bytes());
        }
        if let Some(sum) = source.row_squared_norm {
            h.update(sum.to_le_bytes());
        }
        if let Some(sum) = source.row_l1_squared_sum {
            h.update(b"public-row-l1-energy/v1");
            h.update(sum.to_le_bytes());
        }
    }
    h.finalize().into()
}

pub(super) fn source_binding_with_norms(
    sources: &BTreeMap<String, SourceTensor>,
    norms: &SourceNorms,
) -> [u8; 32] {
    let original = source_binding(sources);
    if norms.is_empty() {
        return original;
    }
    let mut h = Sha256::new();
    h.update(b"pllm.replicated-decoder.norm-certified-source.v1");
    h.update(original);
    h.update(source_norms::digest(norms));
    h.finalize().into()
}

pub struct Program {
    pub(super) nodes: Vec<Node>,
    pub(super) inputs: usize,
    pub(super) outputs: Vec<(usize, usize)>,
    pub(super) retire: Vec<Vec<usize>>,
    digest: [u8; 32],
    payload_bytes: usize,
    work: WorkBudget,
    pub(super) source_binding: [u8; 32],
    // Consuming source/request custody follows the original Program allocation.
    pub(super) request_custody: Option<Box<request_origin::schedule::Custody>>,
}

#[path = "schedule_access.rs"]
pub(super) mod schedule_access;

// Storage-neutral schedule seam over expanded or sealed compact custody.
// Keep the Program and its retained owners authoritative; these borrows neither
// copy recipes nor mint hash/work authority or qualify unmounted drafts.
impl Program {
    /// Read-only backend witness. This reports the actual retained storage,
    /// never a requested selector, digest, or admission permission.
    #[doc(hidden)]
    pub fn storage_kind(&self) -> &'static str {
        if self.compact_storage().is_some() {
            "compact"
        } else if self
            .request_custody
            .as_deref()
            .and_then(request_origin::schedule::Custody::resident_compact)
            .is_some()
        {
            "compact"
        } else {
            "expanded"
        }
    }
    pub(super) fn compact_storage(
        &self,
    ) -> Option<&source_entry::compact_pending::sealed::SealedOwnedStorage> {
        self.request_custody
            .as_deref()
            .and_then(request_origin::schedule::Custody::compact)
    }
    pub(super) fn schedule_access(&self) -> schedule_access::PrivateScheduleRef<'_> {
        match self.compact_storage() {
            Some(owner) => schedule_access::PrivateScheduleRef::compact(owner),
            None => match self
                .request_custody
                .as_deref()
                .and_then(request_origin::schedule::Custody::resident_compact)
            {
                Some(owner) => schedule_access::PrivateScheduleRef::resident_compact(owner),
                None => schedule_access::PrivateScheduleRef::expanded(self),
            },
        }
    }

    /// Actual retained expanded owner census for the TEST-ONLY frozen oracle.
    /// Counts repeated native owners conservatively, without Arc alias credit.
    /// This is not a bounded producer or a replacement for request admission.
    #[cfg(test)]
    pub(super) fn frozen_oracle_metadata_bytes(&self) -> Result<usize, Error> {
        use std::mem::size_of;
        let add = |a: usize, b: usize| a.checked_add(b).ok_or(Error::Bounds);
        let mut bytes = add(
            size_of::<Self>() + 2 * size_of::<usize>(),
            self.nodes
                .capacity()
                .checked_mul(size_of::<Node>())
                .ok_or(Error::Bounds)?,
        )?;
        bytes = add(
            bytes,
            self.outputs
                .capacity()
                .checked_mul(size_of::<(usize, usize)>())
                .ok_or(Error::Bounds)?,
        )?;
        bytes = add(
            bytes,
            self.retire
                .capacity()
                .checked_mul(size_of::<Vec<usize>>())
                .ok_or(Error::Bounds)?,
        )?;
        for list in &self.retire {
            bytes = add(
                bytes,
                list.capacity()
                    .checked_mul(size_of::<usize>())
                    .ok_or(Error::Bounds)?,
            )?;
        }
        let native = |p: &Arc<tensor::Program>| {
            add(
                p.resources()?.immutable_template_bytes,
                2 * size_of::<usize>(),
            )
        };
        for node in &self.nodes {
            let extra = match &node.code {
                Code::Input(name) | Code::PublicInput { key: name, .. } => name.capacity(),
                Code::Gather(words) => words.metadata_bytes(),
                Code::PublicValues(values) => values
                    .capacity()
                    .checked_mul(size_of::<i64>())
                    .ok_or(Error::Bounds)?,
                Code::Affine { .. }
                | Code::Product { norm: None, .. }
                | Code::PublicProduct { .. } => 0,
                Code::Tensor { program, .. } => native(program)?,
                Code::TensorLoop { program, tail, .. } => {
                    add(native(program)?, tail.as_ref().map_or(Ok(0), native)?)?
                }
                _ => return Err(Error::Binding),
            };
            bytes = add(bytes, extra)?;
        }
        if let Some(custody) = &self.request_custody {
            bytes = add(bytes, custody.retained_bytes())?;
        }
        Ok(bytes)
    }

    /// Borrowed metadata only; original expanded identity, ownership and lifetime
    /// stay with this Program. No compact execution capability is manufactured.
    pub(super) fn read_recipe(&self, ordinal: usize) -> schedule_access::ReadNode<'_> {
        self.schedule_access()
            .node(ordinal)
            .expect("original schedule ordinal")
    }

    /// Source-schedule ordinals, independent of physical value storage.
    pub(super) fn logical_len(&self) -> usize {
        self.schedule_access().len()
    }

    /// Original ordered imports, including public source descriptors.
    pub(super) fn input_count(&self) -> usize {
        self.schedule_access()
            .input_count()
            .expect("original imports")
    }

    /// Return the logical owner and its borrowed original import recipe.
    // Legacy expanded Node lens retained for existing debug/test callers. New
    // startup/runtime access uses the storage-neutral recipe lens instead.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn input_at(&self, position: usize) -> (usize, &Node) {
        (position, &self.nodes[..self.inputs][position])
    }

    /// Borrow original geometry/code, including the original native kernel Arcs.
    pub(super) fn recipe(&self, ordinal: usize) -> &Node {
        &self.nodes[ordinal]
    }

    pub(super) fn recipe_at(&self, ordinal: usize) -> Option<&Node> {
        self.nodes.get(ordinal)
    }

    pub(super) fn value_slot_count(&self) -> usize {
        self.schedule_access()
            .value_slot_count()
            .expect("original slots")
    }

    /// Resolve a logical coordinate's owner; its lane index stays unchanged.
    /// Expanded storage is exactly identity-mapped, with caller-side indexing
    /// retaining its existing checked/unchecked behavior.
    pub(super) fn value_slot(&self, logical_owner: usize) -> usize {
        self.schedule_access()
            .value_slot(logical_owner)
            .expect("original slot mapping")
    }

    /// Original last-use list, in logical owner coordinates and original order.
    pub(super) fn retire_at(&self, ordinal: usize) -> &[usize] {
        self.schedule_access()
            .retirement(ordinal)
            .expect("original retirement ordinal")
    }

    pub(super) fn output_coordinates(&self) -> &[(usize, usize)] {
        self.schedule_access().outputs().expect("original outputs")
    }
}

/// Compiler-minted, borrowed descriptor validation. This is not source-byte
/// authentication or complete-program admission. Fields cannot be caller-built.
pub(super) fn source_shapes_from_plan(
    plan: &pllm_models::DecoderPlan,
    weights: WeightMode,
) -> Result<(BTreeMap<String, Vec<usize>>, usize), String> {
    let vocabulary = plan
        .prefill
        .operations
        .iter()
        .find(|op| op.operator == Op::OutputHead)
        .ok_or("missing complete head")?
        .output_shape[1] as usize;
    let mut shapes = BTreeMap::new();
    let mut known = BTreeMap::new();
    for op in &plan.prefill.operations {
        if let Some(name) = op.attributes.get("weight").and_then(|v| v.as_str()) {
            let shape = match op.operator {
                Op::GeluErf | Op::Tanh | Op::AttentionMask => return Err("source-FP32 operators have no replicated decoder source contract".into()),
                Op::TokenLookup => vec![vocabulary, *op.output_shape.last().unwrap() as usize],
                Op::RmsNorm => vec![*op.output_shape.last().unwrap() as usize],
                Op::Linear | Op::OutputHead => vec![
                    *op.output_shape.last().unwrap() as usize,
                    *known.get(&op.inputs[0]).ok_or("missing linear input")?,
                ],
                _ => return Err("unsupported replicated source operator".into()),
            };
            if shapes.get(name).is_some_and(|s| s != &shape) {
                return Err("conflicting replicated source shape".into());
            }
            shapes.insert(name.into(), shape);
        }
        if let Some(name) = op.attributes.get("bias").and_then(|v| v.as_str()) {
            shapes.insert(name.into(), vec![*op.output_shape.last().unwrap() as usize]);
        }
        known.insert(op.id.clone(), *op.output_shape.last().unwrap() as usize);
    }
    let count = shapes.values().try_fold(0usize, |total, shape| {
        let n = shape.iter().try_fold(1usize, |n, d| n.checked_mul(*d))?;
        total.checked_add(n)
    });
    if count.is_none_or(|n| {
        n > if weights == WeightMode::Public {
            MAX_PUBLIC_SOURCE_VALUES
        } else {
            MAX_VALUES
        }
    }) {
        return Err("replicated source payload limit".into());
    }
    Ok((shapes, vocabulary))
}

pub(super) struct ValidatedSourceDescriptors<'a> {
    sources: &'a BTreeMap<String, SourceTensor>,
    norms: &'a SourceNorms,
    weights: WeightMode,
    execution: ExecutionPolicy,
}

impl<'a> ValidatedSourceDescriptors<'a> {
    pub(super) fn public_records(
        self,
    ) -> Result<(&'a BTreeMap<String, SourceTensor>, &'a SourceNorms), Error> {
        if self.weights != WeightMode::Public {
            return Err(Error::Binding);
        }
        Ok((self.sources, self.norms))
    }
}

/// Shared source-contract checks, before any expanded execution nodes exist.
pub(super) fn validate_source_records<'a>(
    graph: &DecoderGraph,
    shapes: &BTreeMap<String, Vec<usize>>,
    sources: &'a BTreeMap<String, SourceTensor>,
    weights: WeightMode,
    execution: ExecutionPolicy,
    norms: &'a SourceNorms,
) -> Result<ValidatedSourceDescriptors<'a>, String> {
    if !norms.is_empty()
        && (weights != WeightMode::Public || execution.range_policy != RangePolicy::Checked)
    {
        return Err("public source norms require checked public execution".into());
    }
    source_norms::validate_set(graph, shapes, norms).map_err(|e| e.to_string())?;
    let public: std::collections::BTreeSet<_> = graph
        .operations
        .iter()
        .filter(|v| v.operator == Op::RmsNorm)
        .map(|op| {
            op.attributes
                .get("weight")
                .and_then(|v| v.as_str())
                .ok_or("missing norm source")
        })
        .collect::<Result<_, _>>()?;
    if shapes.len() != sources.len() || sources.iter().any(|(k, v)| shapes.get(k) != Some(&v.shape))
    {
        return Err("replicated source shape set mismatch".into());
    }
    if sources.iter().any(|(k, v)| {
        v.digest == [0; 32]
            || public.contains(k.as_str()) != v.public_q18.is_some()
            || v.public_q18.as_ref().is_some_and(|x| {
                x.len() != v.shape.iter().product::<usize>()
                    || x.iter().any(|n| *n < v.lower || *n > v.upper)
            })
    }) {
        return Err("replicated source commitment/public coefficient mismatch".into());
    }
    for source in sources.values() {
        if let Some(sum) = source.row_l1 {
            if weights != WeightMode::Public
                || source.shape.len() != 2
                || (source.shape[1] as u64)
                    .checked_mul(source.lower.unsigned_abs().max(source.upper.unsigned_abs()))
                    .is_none_or(|maximum| sum > maximum)
            {
                return Err("invalid public source row bound".into());
            }
        }
        if let Some(sum) = source.row_squared_norm {
            let magnitude =
                u128::from(source.lower.unsigned_abs().max(source.upper.unsigned_abs()));
            if weights != WeightMode::Public
                || source.shape.len() != 2
                || magnitude
                    .checked_mul(magnitude)
                    .and_then(|v| v.checked_mul(source.shape[1] as u128))
                    .is_none_or(|maximum| sum > maximum)
            {
                return Err("invalid public source row square bound".into());
            }
        }
        if let Some(sum) = source.row_l1_squared_sum {
            let maximum = source
                .row_l1
                .and_then(|row| u128::from(row).checked_mul(u128::from(row)))
                .and_then(|square| square.checked_mul(source.shape[0] as u128));
            if weights != WeightMode::Public
                || source.shape.len() != 2
                || maximum.is_none_or(|maximum| sum > maximum)
            {
                return Err("invalid public source row energy bound".into());
            }
        }
        if let Some(values) = &source.public_q18 {
            let mut h = Sha256::new();
            for value in values {
                h.update(value.to_le_bytes());
            }
            if <[u8; 32]>::from(h.finalize()) != source.digest {
                return Err("replicated public coefficient digest mismatch".into());
            }
        }
    }
    Ok(ValidatedSourceDescriptors {
        sources,
        norms,
        weights,
        execution,
    })
}

#[derive(Clone)]
struct View {
    words: Words,
    shape: Vec<usize>,
}

#[path = "original_import.rs"]
#[allow(dead_code)]
mod original_import;

#[path = "source_tensor_publication.rs"]
#[allow(dead_code)]
mod source_tensor_publication;

struct SourceView {
    owner: usize,
    shape: Vec<usize>,
}

/// Compiler-only public context. No request values or party-local data enter
/// admission diagnostics, and this record is not part of the execution schedule.
#[derive(Default)]
struct AdmissionSite {
    selection: usize,
    step: Option<usize>,
    operator: Option<Op>,
    layer: Option<u64>,
    inputs: [Option<(i64, i64)>; 2],
    complete_work: Option<(u64, u64)>,
}

impl std::fmt::Display for AdmissionSite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "selection: {}, step: {:?}, operator: {:?}, layer: {:?}, inputs: {:?}",
            self.selection, self.step, self.operator, self.layer, self.inputs
        )?;
        if let Some((ordinals, aes_blocks)) = self.complete_work {
            write!(
                f,
                "; complete-program ordinals: {ordinals}, per-party AES bound: {aes_blocks}"
            )?;
        }
        Ok(())
    }
}

/// Matrix coordinates are [output column, contraction index]. Source matrices
/// carry strides instead of one (node, index) pair per checkpoint value.
enum MatrixView<'a> {
    Values(&'a View),
    Source {
        owner: usize,
        columns: usize,
        inner: usize,
        column_stride: usize,
        inner_stride: usize,
    },
}

impl<'a> MatrixView<'a> {
    fn source(source: &'a SourceView, transposed: bool) -> Result<Self, Error> {
        let [rows, columns] = source.shape.as_slice() else {
            return Err(Error::Shape);
        };
        Ok(if transposed {
            Self::Source {
                owner: source.owner,
                columns: *columns,
                inner: *rows,
                column_stride: 1,
                inner_stride: *columns,
            }
        } else {
            Self::Source {
                owner: source.owner,
                columns: *rows,
                inner: *columns,
                column_stride: *columns,
                inner_stride: 1,
            }
        })
    }

    fn dimensions(&self) -> Result<(usize, usize), Error> {
        match self {
            Self::Source { columns, inner, .. } => Ok((*columns, *inner)),
            Self::Values(view) => {
                let [columns, inner] = view.shape.as_slice() else {
                    return Err(Error::Shape);
                };
                if columns.checked_mul(*inner) != Some(view.words.len()) {
                    return Err(Error::Shape);
                }
                Ok((*columns, *inner))
            }
        }
    }

    fn gather(
        &self,
        builder: &mut Builder,
        column: usize,
        columns: usize,
        start: usize,
        terms: usize,
    ) -> Result<usize, Error> {
        let (full_columns, inner) = self.dimensions()?;
        if column.checked_add(columns).is_none_or(|n| n > full_columns)
            || start.checked_add(terms).is_none_or(|n| n > inner)
            || terms
                .checked_mul(columns)
                .is_none_or(|n| n == 0 || n > puma::MAX_LANES)
        {
            return Err(Error::Shape);
        }
        match self {
            Self::Values(view) => {
                let mut items = Words::new();
                for k in start..start + terms {
                    for j in column..column + columns {
                        items.push(view.words.at(j * inner + k));
                    }
                }
                builder.gather(&items)
            }
            Self::Source {
                owner,
                column_stride,
                inner_stride,
                ..
            } => {
                let offset = column
                    .checked_mul(*column_stride)
                    .and_then(|v| v.checked_add(start.checked_mul(*inner_stride)?))
                    .ok_or(Error::Bounds)?;
                builder.source_gather(SourceGather {
                    owner: *owner,
                    offset,
                    rows: terms,
                    columns,
                    row_stride: *inner_stride,
                    column_stride: *column_stride,
                })
            }
        }
    }
}

#[derive(Default)]
struct Builder {
    nodes: Vec<Node>,
    // Empty on the legacy path. Only genuine opt-in input creation can append
    // these immutable records; ordinary coordinate-only Views cannot mint one.
    original_inputs: Vec<Arc<original_import::InputRecord>>,
    request_emission: Option<request_origin::Active>,
    allocated: usize,
    workspace_bytes: usize,
    tensors: BTreeMap<[u8; 32], Arc<tensor::Program>>,
    public_weights: bool,
    checked_bound: Option<i64>,
    validity: Option<usize>,
    lookup: policy::LookupPolicy,
    selection: SelectionPolicy,
    token_domains: BTreeMap<(usize, usize), (usize, runtime::trie::DomainOrigin)>,
    token_source_binding: [u8; 32],
    row_norms: Vec<(usize, u128)>,
    row_energies: Vec<(Words, u128)>,
    row_masses: Vec<(usize, u64)>,
    public_norms: BTreeMap<usize, BlockNorms>,
    attention_energies: Vec<(Words, u128)>,
    // Public admission diagnostics only; never hashed or executed.
    admission_stage: &'static str,
    admission_limit: Option<&'static str>,
    requested_bytes: Option<usize>,
    requested_lanes: Option<usize>,
    value_payload_bytes: usize,
    trie_workspace_bytes: usize,
}

fn construction_views(graph: &DecoderGraph, sources: usize) -> Result<(), String> {
    let failure = || "replicated compiler view payload limit".to_string();
    let mut last = BTreeMap::new();
    for (step, op) in graph.operations.iter().enumerate() {
        for input in &op.inputs {
            last.insert(input.as_str(), step);
        }
    }
    let mut live = BTreeMap::new();
    let mut retained = 0usize;
    for (step, op) in graph.operations.iter().enumerate() {
        let n = op
            .output_shape
            .iter()
            .try_fold(1usize, |n, width| {
                n.checked_mul(usize::try_from(*width).ok()?)
            })
            .ok_or_else(failure)?;
        if n > MAX_VALUES {
            return Err(failure());
        }
        // A maximally fragmented descriptor vector can have twice as much
        // capacity as length. Include construction copies and retained state.
        let frontier = live
            .values()
            .try_fold(retained, |a: usize, b: &usize| a.checked_add(*b))
            .and_then(|v| v.checked_add(n.checked_mul(2)?))
            .and_then(|v| v.checked_mul(64))
            .and_then(|v| v.checked_add(sources * (std::mem::size_of::<SourceView>() + 16)))
            .ok_or_else(failure)?;
        if frontier > MAX_PROGRAM_BYTES {
            return Err(failure());
        }
        if matches!(op.operator, Op::KvCacheAppend | Op::CacheSuffix) {
            retained = retained.checked_add(n).ok_or_else(failure)?;
        }
        live.insert(op.id.as_str(), n);
        live.retain(|id, _| last.get(id).is_some_and(|end| *end > step));
    }
    Ok(())
}

impl compact_lowering::SelectSink for Builder {
    type Owner = usize;
    fn bounds(&self, owner: usize) -> Interval {
        self.nodes[owner].bounds
    }
    fn affine(&mut self, left: usize, right: usize, coefficients: Affine) -> Result<usize, Error> {
        Builder::affine(self, left, Some(right), coefficients)
    }
    fn product(&mut self, left: usize, right: usize) -> Result<usize, Error> {
        Builder::product(self, left, right, None, false)
    }
    fn selected_affine(
        &mut self,
        no: usize,
        product: usize,
        bounds: Interval,
    ) -> Result<usize, Error> {
        let out = Builder::affine(self, no, Some(product), affine(1, 1, 0))?;
        self.nodes[out].bounds = bounds;
        Ok(out)
    }
}

struct ExpandedArgmax<'a> {
    builder: &'a mut Builder,
    input: &'a Words,
    values: Words,
    ids: Words,
    next_values: Words,
    next_ids: Words,
}

impl compact_lowering::ArgmaxSink for ExpandedArgmax<'_> {
    type Owner = usize;
    type Output = Words;
    fn indices(&mut self, lanes: usize) -> Result<(), Error> {
        let indices = self.builder.push(
            Code::PublicValues((0..lanes as i64).collect()),
            lanes,
            Interval::new(0, lanes as i64 - 1)?,
        )?;
        self.values = self.input.clone();
        self.ids = (0..lanes).map(|i| (indices, i)).collect();
        Ok(())
    }
    fn begin_round(&mut self, _round: usize, _lanes: usize) -> Result<(), Error> {
        self.next_values = Words::new();
        self.next_ids = Words::new();
        Ok(())
    }
    fn gather_pairs(
        &mut self,
        field: compact_lowering::ArgmaxField,
        start: usize,
        pairs: usize,
        parity: usize,
    ) -> Result<usize, Error> {
        let words = match field {
            compact_lowering::ArgmaxField::Values => &self.values,
            compact_lowering::ArgmaxField::Indices => &self.ids,
        };
        self.builder.gather(
            &(0..pairs)
                .map(|i| words.at(start + 2 * i + parity))
                .collect(),
        )
    }
    fn difference(&mut self, left: usize, right: usize) -> Result<usize, Error> {
        self.builder.affine(left, Some(right), affine(1, -1, 0))
    }
    fn negative(&mut self, difference: usize) -> Result<usize, Error> {
        self.builder.tensor(
            difference,
            tensor::Program::negative(
                self.builder.nodes[difference].lanes,
                self.builder.nodes[difference].bounds,
            )?,
        )
    }
    fn select(&mut self, bit: usize, yes: usize, no: usize) -> Result<usize, Error> {
        self.builder.select(bit, yes, no)
    }
    fn publish_pairs(&mut self, best: usize, index: usize, lanes: usize) -> Result<(), Error> {
        self.next_values.extend((0..lanes).map(|i| (best, i)));
        self.next_ids.extend((0..lanes).map(|i| (index, i)));
        Ok(())
    }
    fn carry(&mut self, index: usize) -> Result<(), Error> {
        self.next_values.push(self.values.at(index));
        self.next_ids.push(self.ids.at(index));
        Ok(())
    }
    fn end_round(&mut self) -> Result<(), Error> {
        self.values = std::mem::take(&mut self.next_values);
        self.ids = std::mem::take(&mut self.next_ids);
        Ok(())
    }
    fn finish(&mut self) -> Result<Words, Error> {
        Ok(std::mem::take(&mut self.ids))
    }
}

impl Builder {
    /// Mint only during actual Input creation under an already admitted permit
    /// factory. No existing View or ordinal is accepted by this endpoint.
    fn create_request_input(
        &mut self,
        mint: request_origin::InputMint,
    ) -> Result<Arc<original_import::InputRecord>, Error> {
        self.create_request_input_retained(mint, &mut None)
    }

    fn create_request_input_retained(
        &mut self,
        mint: request_origin::InputMint,
        pending: &mut Option<Code>,
    ) -> Result<Arc<original_import::InputRecord>, Error> {
        let (rows, bounds) = mint.parts();
        if self
            .nodes
            .iter()
            .any(|n| matches!(&n.code, Code::Input(key) if key == "input.tokens"))
        {
            return Err(Error::Binding);
        }
        let next = self
            .original_inputs
            .len()
            .checked_add(1)
            .ok_or(Error::Bounds)?;
        if self.original_inputs.capacity() < next {
            self.original_inputs
                .try_reserve_exact(1)
                .map_err(|_| Error::Allocation)?;
            if self.original_inputs.capacity() != next {
                return Err(Error::Allocation);
            }
        }
        if self.nodes.len() == self.nodes.capacity() {
            let next = self.nodes.len().checked_add(1).ok_or(Error::Bounds)?;
            self.nodes
                .try_reserve_exact(1)
                .map_err(|_| Error::Allocation)?;
            if self.nodes.capacity() != next {
                return Err(Error::Allocation);
            }
        }
        if pending.is_some() {
            return Err(Error::Binding);
        }
        *pending = Some(Code::Input("input.tokens".into()));
        let owner = self.push_retained(pending, rows, bounds)?;
        let record = original_import::request_record(CreatedRequestInput {
            owner,
            rows,
            bounds,
        })?;
        self.original_inputs.push(Arc::clone(&record));
        Ok(record)
    }

    fn admission_ledger(&self) -> String {
        format!(
            "stage: {}, limit: {:?}, nodes: {}/{MAX_NODES}, program_bytes: {}/{MAX_PROGRAM_BYTES}, requested_bytes: {:?}, requested_lanes: {:?}, value_limit: {MAX_VALUES}, public_value_limit: {MAX_PUBLIC_SOURCE_VALUES}, workspace_bytes: {}, trie_active_workspace_bytes: {}, value_payload_bytes: {}, cumulative_nonvalue_bytes: {}, tensor_templates: {}, row_certificates: {}",
            self.admission_stage,
            self.admission_limit,
            self.nodes.len(),
            self.allocated,
            self.requested_bytes,
            self.requested_lanes,
            self.workspace_bytes,
            self.trie_workspace_bytes,
            self.value_payload_bytes,
            self.allocated.saturating_sub(self.value_payload_bytes),
            self.tensors.len(),
            self.row_norms.len() + self.row_energies.len(),
        )
    }

    fn charge_program_bytes(&mut self, bytes: usize) -> Result<(), Error> {
        let requested = self.allocated.checked_add(bytes);
        self.requested_bytes = requested;
        let Some(total) = requested.filter(|n| *n <= MAX_PROGRAM_BYTES) else {
            self.admission_limit = Some("program_bytes");
            return Err(Error::Bounds);
        };
        self.allocated = total;
        Ok(())
    }

    fn push(&mut self, code: Code, lanes: usize, bounds: Interval) -> Result<usize, Error> {
        self.push_retained(&mut Some(code), lanes, bounds)
    }

    /// Keep caller's payload resident through validation, debit and reserve.
    fn push_retained(
        &mut self,
        pending: &mut Option<Code>,
        lanes: usize,
        bounds: Interval,
    ) -> Result<usize, Error> {
        let code = pending.as_ref().ok_or(Error::Binding)?;
        // A finished/reused Builder cannot relabel an old sealed declaration by
        // creating another Input at its recycled ordinal. Legacy builders have
        // no records and pay no new allocation or changed numeric schedule.
        if matches!(code, Code::Input(_))
            && self
                .original_inputs
                .iter()
                .any(|input| input.owner() == self.nodes.len())
        {
            return Err(Error::Binding);
        }
        let public_source = matches!(code, Code::PublicInput { .. });
        self.requested_lanes = Some(lanes);
        if lanes == 0
            || lanes
                > if public_source {
                    MAX_PUBLIC_SOURCE_VALUES
                } else {
                    MAX_VALUES
                }
        {
            self.admission_limit = Some(if public_source {
                "public_values"
            } else {
                "values"
            });
            return Err(Error::Bounds);
        }
        if self.nodes.len() >= MAX_NODES {
            self.admission_limit = Some("nodes");
            return Err(Error::Bounds);
        }
        let metadata = match code {
            Code::Product { norm: Some(_), .. } => {
                std::mem::size_of::<puma::multiply::norm::RowEnergyProof>()
                    + 2 * std::mem::size_of::<usize>()
            }
            Code::Gather(v) => v.metadata_bytes(),
            Code::Tensor { program, .. } if !self.tensors.contains_key(&program.digest()) => {
                // One conservative complete primitive allowance per immutable
                // program, plus the interning entry. Execution still uses fresh
                // party-local state for every invocation.
                program.admitted_bytes() + 1024
            }
            Code::TensorLoop { program, tail, .. } => {
                std::iter::once(program)
                    .chain(tail)
                    .filter(|p| !self.tensors.contains_key(&p.digest()))
                    .map(|p| p.admitted_bytes() + 1024)
                    .sum::<usize>()
                    + 4096
            }
            Code::Input(v) => v.capacity(),
            Code::PublicInput { key, .. } => key.capacity(),
            Code::PublicValues(values) => values.capacity() * 8,
            Code::PublicLookup { plan, .. } => {
                let (rows, _, width) = plan.geometry();
                if rows.checked_mul(width) != Some(lanes) {
                    return Err(Error::Shape);
                }
                let output_bytes = lanes.checked_mul(16).ok_or(Error::Bounds)?;
                // Immutable full/tail programs are shared with ordinary Tensor
                // nodes. Charge their Arc storage once/digest; each Plan Arc is
                // distinct. Registry/allocator allowance is separate from exact
                // payload/reference-counter sizes.
                let retained = plan
                    .programs()
                    .filter(|p| !self.tensors.contains_key(&p.digest()))
                    .try_fold(plan.shallow_metadata_bytes() + 1024, |n, p| {
                        n.checked_add(runtime::lookup::Plan::program_metadata_bytes(p))
                            .and_then(|n| n.checked_add(1024))
                            .ok_or(Error::Bounds)
                    })?;
                // Peak includes the active program clone, original primitive
                // workspace and overlapping selectors/temporaries/file buffer.
                // The accumulator becomes this node's output without a copy;
                // its ordinary 16*lanes graph charge below covers that owner.
                let workspace = plan
                    .peak_extra_bytes()?
                    .checked_sub(plan.metadata_bytes()?)
                    .and_then(|n| n.checked_sub(output_bytes))
                    .ok_or(Error::Bounds)?;
                let extra = workspace.saturating_sub(self.workspace_bytes);
                self.workspace_bytes = self.workspace_bytes.max(workspace);
                retained.checked_add(extra).ok_or(Error::Bounds)?
            }
            Code::TrieLookup {
                input,
                weight,
                plan,
            } => {
                let (rows, _, width) = plan.geometry();
                plan.validate_owners(*input, *weight, self.token_source_binding)?;
                if !self.public_weights
                    || !matches!(
                        self.nodes.get(*weight).map(|n| &n.code),
                        Some(Code::PublicInput { .. })
                    )
                {
                    return Err(Error::Binding);
                }
                if rows.checked_mul(width) != Some(lanes) {
                    return Err(Error::Shape);
                }
                let metadata = plan.metadata_bytes();
                // Input gather and final output each have ordinary graph charges.
                // All live selector levels, primitive state, copies and allocator
                // overlap remain in the peak owner allowance, never per-template.
                let workspace = plan
                    .peak_extra_bytes()?
                    .checked_sub(metadata)
                    .and_then(|n| n.checked_sub(lanes.checked_mul(16)?))
                    .and_then(|n| n.checked_sub(rows.checked_mul(16)?))
                    .ok_or(Error::Bounds)?;
                let extra = workspace.saturating_sub(self.workspace_bytes);
                self.workspace_bytes = self.workspace_bytes.max(workspace);
                self.trie_workspace_bytes = self.trie_workspace_bytes.max(workspace);
                metadata
                    .checked_add(1024)
                    .and_then(|n| n.checked_add(extra))
                    .ok_or(Error::Bounds)?
            }
            Code::Selection { plan, .. } => {
                let metadata = plan.metadata_bytes()?;
                // The dedicated full-head gather and original logits already
                // have distinct 16*lanes charges. Moving the gather into the
                // tournament allocates no second score array.
                let workspace = plan
                    .peak_extra_bytes()?
                    .checked_sub(metadata)
                    .and_then(|n| n.checked_add(plan.predicate_workspace_bytes()))
                    .ok_or(Error::Bounds)?;
                let extra = workspace.saturating_sub(self.workspace_bytes);
                self.workspace_bytes = self.workspace_bytes.max(workspace);
                metadata
                    .checked_add(1024)
                    .and_then(|n| n.checked_add(extra))
                    .ok_or(Error::Bounds)?
            }
            _ => 0,
        };
        self.charge_program_bytes(
            metadata + std::mem::size_of::<Node>() + if public_source { 0 } else { 16 * lanes },
        )?;
        if !public_source {
            // Already included in the admitted total above. Diagnostic subtotal
            // only; never an additional allowance or allocation.
            self.value_payload_bytes += 16 * lanes;
        }
        let id = self.nodes.len();
        self.nodes.try_reserve(1).map_err(|_| Error::Allocation)?;
        if let Code::Tensor { program, .. } = code {
            self.tensors
                .entry(program.digest())
                .or_insert_with(|| program.clone());
        }
        if let Code::TensorLoop { program, tail, .. } = code {
            for program in std::iter::once(program).chain(tail) {
                self.tensors
                    .entry(program.digest())
                    .or_insert_with(|| program.clone());
            }
        }
        if let Code::PublicLookup { plan, .. } = code {
            for program in plan.programs() {
                self.tensors
                    .entry(program.digest())
                    .or_insert_with(|| program.clone());
            }
        }
        self.nodes.push(Node {
            // Capacity secured above; no callback or fallibility after take.
            code: pending.take().expect("validated resident code"),
            lanes,
            bounds,
        });
        Ok(id)
    }
    fn bounds(&self, words: &Words) -> Result<Interval, Error> {
        let mut lo = i64::MAX;
        let mut hi = i64::MIN;
        for (id, index) in words.iter() {
            let node = self.nodes.get(id).ok_or(Error::Binding)?;
            if index >= node.lanes {
                return Err(Error::Shape);
            }
            lo = lo.min(node.bounds.lower());
            hi = hi.max(node.bounds.upper());
        }
        Interval::new(lo, hi)
    }
    fn gather(&mut self, words: &Words) -> Result<usize, Error> {
        let bound = self.bounds(words)?;
        self.push(Code::Gather(words.clone()), words.len(), bound)
    }
    fn source_gather(&mut self, view: SourceGather) -> Result<usize, Error> {
        let owner = self.nodes.get(view.owner).ok_or(Error::Binding)?;
        if !matches!(owner.code, Code::Input(_) | Code::PublicInput { .. }) {
            return Err(Error::Binding);
        }
        let lanes = view.validate(owner.lanes)?;
        self.push(Code::SourceGather(view), lanes, owner.bounds)
    }
    fn affine(&mut self, left: usize, right: Option<usize>, c: Affine) -> Result<usize, Error> {
        let a = &self.nodes[left];
        if right.is_some_and(|id| self.nodes[id].lanes != a.lanes) {
            return Err(Error::Shape);
        }
        let bounds = c.output_bounds(InputBounds {
            left: a.bounds,
            right: right.map_or(Interval::new(0, 0)?, |id| self.nodes[id].bounds),
        })?;
        self.push(
            Code::Affine {
                left,
                right,
                coefficients: c,
            },
            a.lanes,
            bounds,
        )
    }
    fn product(
        &mut self,
        left: usize,
        right: usize,
        shape: Option<MatrixShape>,
        rescale: bool,
    ) -> Result<usize, Error> {
        self.product_with_norm(left, right, shape, rescale, None)
    }

    fn product_with_norm(
        &mut self,
        left: usize,
        right: usize,
        shape: Option<MatrixShape>,
        rescale: bool,
        norm: Option<puma::multiply::norm::RowEnergyProof>,
    ) -> Result<usize, Error> {
        let a = &self.nodes[left];
        let b = &self.nodes[right];
        let (n, terms) = if let Some(s) = shape {
            if a.lanes != s.left_elements() || b.lanes != s.right_elements() {
                return Err(Error::Shape);
            }
            (s.output_elements(), s.dimensions()[1])
        } else {
            if a.lanes != b.lanes || a.lanes > puma::MAX_LANES {
                return Err(Error::Shape);
            }
            (a.lanes, 1)
        };
        let bounds = InputBounds {
            left: a.bounds,
            right: b.bounds,
        };
        let range =
            compact_lowering::product_bounds(bounds, terms, shape.is_some(), rescale, norm)?;
        self.push(
            Code::Product {
                left,
                right,
                shape,
                rescale,
                norm: norm.map(Arc::new),
            },
            n,
            range,
        )
    }
    fn tensor(&mut self, input: usize, program: tensor::Program) -> Result<usize, Error> {
        if self.nodes[input].lanes != program.lanes() {
            return Err(Error::Shape);
        }
        let n = program.output_lanes();
        let range = program.output_bounds();
        let program = self
            .tensors
            .get(&program.digest())
            .cloned()
            .unwrap_or_else(|| Arc::new(program));
        self.push(Code::Tensor { input, program }, n, range)
    }
    /// Original row lowering, shared by graph lowering and row regressions.
    /// Keep checked_rows outside: its original whole-input order/sticky owner
    /// must precede every gather here, never be silently bypassed by a sink.
    fn rms_norm_row(
        &mut self,
        row: &Words,
        width: usize,
        gamma: &SourceTensor,
    ) -> Result<usize, Error> {
        self.admission_stage = "rms_norm_input_gather";
        let input = self.gather(row)?;
        self.admission_stage = "rms_norm_numeric_constructor";
        let coefficients = gamma.public_q18.as_ref().ok_or(Error::Binding)?;
        let input_norm = if self.public_weights {
            self.row_norm(row, width)
        } else {
            None
        };
        let (program, squared) = rms_norm_program(
            self.public_weights,
            width,
            self.nodes[input].bounds,
            coefficients,
            input_norm,
        )?;
        self.admission_stage = "rms_norm_tensor_owner";
        let output = self.tensor(input, program)?;
        if let Some(squared) = squared {
            self.admission_stage = "rms_norm_output_certificate";
            self.charge_program_bytes(std::mem::size_of::<(usize, u128)>())?;
            self.row_norms
                .try_reserve_exact(1)
                .map_err(|_| Error::Allocation)?;
            self.row_norms.push((output, squared));
        }
        Ok(output)
    }

    fn view(&self, id: usize, shape: Vec<usize>) -> View {
        View {
            words: (0..self.nodes[id].lanes).map(|i| (id, i)).collect(),
            shape,
        }
    }

    fn tensor_loop(
        &mut self,
        input: LoopInput,
        lanes: usize,
        make: impl Fn(usize) -> Result<tensor::Program, Error>,
    ) -> Result<usize, Error> {
        let owner = self.nodes.get(input.owner()).ok_or(Error::Binding)?;
        if lanes == 0 || lanes > MAX_VALUES {
            return Err(Error::Bounds);
        }
        match input {
            LoopInput::Slice(_) if owner.lanes != lanes => return Err(Error::Shape),
            LoopInput::TokenOffsets { index, .. } if index >= owner.lanes => {
                return Err(Error::Shape);
            }
            _ => (),
        }
        let intern = |p: tensor::Program| {
            self.tensors
                .get(&p.digest())
                .cloned()
                .unwrap_or_else(|| Arc::new(p))
        };
        let width = lanes.min(128);
        let program = intern(make(width)?);
        let tail = if lanes % width == 0 {
            None
        } else {
            Some(intern(make(lanes % width)?))
        };
        self.publish_tensor_loop(input, lanes, program, tail)
    }

    /// Borrow a closed local-domain capability without replacing its original
    /// template owners. The caller may subsequently move it into the body sink.
    /// This staged seam is not used by default graph lowering.
    #[allow(dead_code)]
    fn prepared_tensor_map(
        &mut self,
        input: usize,
        prepared: &super::prepared_tensor_map::PreparedTensorMap,
    ) -> Result<usize, Error> {
        prepared.validate()?;
        let owner = self.nodes.get(input).ok_or(Error::Binding)?;
        if owner.lanes != prepared.lanes() {
            return Err(Error::Shape);
        }
        if !prepared.accepts(owner.lanes, owner.bounds) {
            return Err(Error::Binding);
        }
        // The existing census charges one backing per digest. It cannot account
        // for a second allocation, even if the immutable programs hash equally.
        for program in prepared.programs() {
            if self
                .tensors
                .get(&program.digest())
                .is_some_and(|existing| !Arc::ptr_eq(existing, program))
                || prepared
                    .programs()
                    .any(|other| other.digest() == program.digest() && !Arc::ptr_eq(other, program))
            {
                return Err(Error::Binding);
            }
        }
        let program = prepared.program_at(0)?.clone();
        let remainder = prepared.lanes() % program.lanes();
        let tail = if remainder == 0 {
            None
        } else {
            Some(prepared.program_at(prepared.lanes() - remainder)?.clone())
        };
        // push can fail reserving the node after charging bytes. Restore those
        // publication counters on failure; diagnostics may describe the attempt.
        // TensorLoop changes neither workspace nor interner before that reserve.
        let allocated = self.allocated;
        let value_payload_bytes = self.value_payload_bytes;
        let result =
            self.publish_tensor_loop(LoopInput::Slice(input), prepared.lanes(), program, tail);
        if result.is_err() {
            self.allocated = allocated;
            self.value_payload_bytes = value_payload_bytes;
        }
        result
    }

    // Shared post-factory publication only. Closure construction/interning above
    // intentionally retains its original order and owner selection.
    fn publish_tensor_loop(
        &mut self,
        input: LoopInput,
        lanes: usize,
        program: Arc<tensor::Program>,
        tail: Option<Arc<tensor::Program>>,
    ) -> Result<usize, Error> {
        if lanes == 0 || lanes > MAX_VALUES {
            return Err(Error::Bounds);
        }
        let width = lanes.min(128);
        let remainder = lanes % width;
        if program.lanes() != width
            || tail.is_some() != (remainder != 0)
            || tail.as_ref().is_some_and(|p| p.lanes() != remainder)
        {
            return Err(Error::Shape);
        }
        let mut bounds = program.output_bounds();
        for p in std::iter::once(&program).chain(&tail) {
            if p.output_lanes() != p.lanes() {
                return Err(Error::Shape);
            }
            bounds = Interval::new(
                bounds.lower().min(p.output_bounds().lower()),
                bounds.upper().max(p.output_bounds().upper()),
            )?;
        }
        self.push(
            Code::TensorLoop {
                input,
                program,
                tail,
            },
            lanes,
            bounds,
        )
    }
    fn constant(&mut self, anchor: usize, value: i64) -> Result<usize, Error> {
        self.affine(anchor, None, affine(0, 0, value))
    }
    fn select(&mut self, bit: usize, yes: usize, no: usize) -> Result<usize, Error> {
        compact_lowering::select(self, bit, yes, no)
    }

    fn argmax_tree(&mut self, logits: &View) -> Result<View, Error> {
        let n = logits.words.len();
        if n == 0 {
            return Err(Error::Shape);
        }
        if self.selection == SelectionPolicy::Tiled {
            if !self.public_weights {
                return Err(Error::Binding);
            }
            let plan = Arc::new(runtime::selection::Plan::new(
                n,
                self.bounds(&logits.words)?,
            )?);
            // Dedicated immediately preceding owner: only this selection node
            // consumes it. Original logits/views remain independently live.
            let input = self.gather(&logits.words)?;
            let output = self.push(
                Code::Selection { input, plan },
                1,
                Interval::new(0, n as i64 - 1)?,
            )?;
            return Ok(self.view(output, vec![1]));
        }
        let mut sink = ExpandedArgmax {
            builder: self,
            input: &logits.words,
            values: Words::new(),
            ids: Words::new(),
            next_values: Words::new(),
            next_ids: Words::new(),
        };
        Ok(View {
            words: compact_lowering::argmax_tree(&mut sink, n)?,
            shape: vec![1],
        })
    }

    fn linear(&mut self, input: &View, weight: &View, rescale: bool) -> Result<View, Error> {
        self.matrix(input, MatrixView::Values(weight), rescale)
    }

    fn matrix(
        &mut self,
        input: &View,
        weight: MatrixView<'_>,
        rescale: bool,
    ) -> Result<View, Error> {
        self.matrix_with_attention(input, weight, rescale, false)
    }

    fn matrix_with_attention(
        &mut self,
        input: &View,
        weight: MatrixView<'_>,
        rescale: bool,
        attention: bool,
    ) -> Result<View, Error> {
        let inner = *input.shape.last().ok_or(Error::Shape)?;
        let (columns, contraction) = weight.dimensions()?;
        if inner == 0
            || columns == 0
            || contraction != inner
            || input.words.is_empty()
            || input.words.len() % inner != 0
        {
            return Err(Error::Shape);
        }
        if let MatrixView::Source {
            owner,
            column_stride,
            inner_stride,
            ..
        } = &weight
        {
            if matches!(self.nodes[*owner].code, Code::PublicInput { .. }) {
                return self.public_matrix(
                    input,
                    *owner,
                    columns,
                    *column_stride == 1 && *inner_stride == columns,
                    rescale,
                );
            }
        }
        if inner > puma::MAX_LANES {
            return self.linear_chunks(input, &weight, rescale);
        }
        let rows = input.words.len() / inner;
        private_matrix_driver::expanded(
            self,
            input,
            &weight,
            [rows, inner, columns],
            rescale,
            attention,
        )
    }

    fn public_matrix(
        &mut self,
        input: &View,
        weight: usize,
        columns: usize,
        transposed: bool,
        rescale: bool,
    ) -> Result<View, Error> {
        let inner = *input.shape.last().ok_or(Error::Shape)?;
        let rows = input.words.len() / inner;
        let statistics = match self.nodes[weight].code {
            Code::PublicInput {
                row_l1,
                row_squared_norm,
                row_l1_squared_sum,
                ..
            } => PublicMatrixStatistics {
                row_l1,
                row_squared_norm,
                row_l1_squared_sum,
            },
            _ => PublicMatrixStatistics::default(),
        };
        let (bounds, energy) = public_matrix_contract(
            [rows, inner, columns],
            transposed,
            rescale,
            || {
                Ok(InputBounds {
                    left: self.bounds(&input.words)?,
                    right: self.nodes[weight].bounds,
                })
            },
            statistics,
            || self.row_norm(&input.words, inner),
        )?;
        let input = self.gather(&input.words)?;
        let raw = self.push(
            Code::PublicProduct {
                input,
                weight,
                rows,
                inner,
                columns,
                transposed,
            },
            rows.checked_mul(columns).ok_or(Error::Bounds)?,
            bounds,
        )?;
        let shape = vec![1, rows, columns];
        if !rescale {
            return Ok(self.view(raw, shape));
        }
        #[cfg(test)]
        let captured = manifest_prepared_fixture::capture_public_rescale(
            self,
            weight,
            raw,
            rows * columns,
            bounds,
        )?;
        #[cfg(test)]
        let id = if let Some(id) = captured {
            id
        } else {
            self.tensor_loop(LoopInput::Slice(raw), rows * columns, |n| {
                let mut b = tensor::Builder::new(n, bounds)?;
                let scaled = b.rescale(b.input(), 18)?;
                b.finish(scaled)
            })?
        };
        #[cfg(not(test))]
        let id = self.tensor_loop(LoopInput::Slice(raw), rows * columns, |n| {
            let mut b = tensor::Builder::new(n, bounds)?;
            let scaled = b.rescale(b.input(), 18)?;
            b.finish(scaled)
        })?;
        let output = self.view(id, shape);
        if let Some(energy) = energy {
            for row in output.words.chunks_exact(columns) {
                self.record_row_energy(&row, energy)?;
            }
        }
        Ok(output)
    }

    fn row_norm(&self, words: &Words, width: usize) -> Option<u128> {
        if width == 0 || words.is_empty() || words.len() % width != 0 {
            return None;
        }
        let mut maximum = 0;
        for row in words.chunks_exact(width) {
            let bound = self
                .row_certificate(&row, width, &self.row_norms)
                .or_else(|| {
                    self.row_energies
                        .iter()
                        .find(|(original, _)| {
                            original.len() == width && original.iter().eq(row.iter())
                        })
                        .map(|(_, bound)| *bound)
                })?;
            maximum = maximum.max(bound);
        }
        Some(maximum)
    }

    fn record_row_energy(&mut self, row: &Words, squared: u128) -> Result<(), Error> {
        let bytes = row
            .metadata_bytes()
            .checked_add(std::mem::size_of::<(Words, u128)>())
            .ok_or(Error::Bounds)?;
        self.charge_program_bytes(bytes)?;
        self.row_energies
            .try_reserve_exact(1)
            .map_err(|_| Error::Allocation)?;
        self.row_energies.push((row.clone(), squared));
        Ok(())
    }

    fn scalar_row_energy(&self, row: &Words) -> Result<u128, Error> {
        row.iter().try_fold(0_u128, |sum, (owner, _)| {
            let range = self.nodes.get(owner).ok_or(Error::Binding)?.bounds;
            let radius = u128::from(
                range
                    .lower()
                    .unsigned_abs()
                    .max(range.upper().unsigned_abs()),
            );
            sum.checked_add(radius * radius).ok_or(Error::Bounds)
        })
    }

    fn residual(&mut self, a: &View, b: &View) -> Result<Words, Error> {
        let words = self.binary(&a.words, &b.words, false)?;
        if self.public_weights {
            let width = *a.shape.last().ok_or(Error::Shape)?;
            if a.shape != b.shape || width == 0 || words.len() % width != 0 {
                return Err(Error::Shape);
            }
            for ((left, right), output) in a
                .words
                .chunks_exact(width)
                .zip(b.words.chunks_exact(width))
                .zip(words.chunks_exact(width))
            {
                let left = self
                    .row_norm(&left, width)
                    .map_or_else(|| self.scalar_row_energy(&left), Ok)?;
                let right = self
                    .row_norm(&right, width)
                    .map_or_else(|| self.scalar_row_energy(&right), Ok)?;
                // Public triangle inequality; the residual addition remains
                // the original exact ring64 operation and stays range-checked.
                self.record_row_energy(&output, residual_energy(left, right)?)?;
            }
        }
        Ok(words)
    }

    fn row_certificate<T: Copy + Ord>(
        &self,
        words: &Words,
        width: usize,
        certificates: &[(usize, T)],
    ) -> Option<T> {
        if width == 0 || words.is_empty() || words.len() % width != 0 {
            return None;
        }
        let mut bound: Option<T> = None;
        for row in words.chunks_exact(width) {
            let (owner, _) = row.iter().next()?;
            if self.nodes.get(owner)?.lanes != width
                || !row
                    .iter()
                    .enumerate()
                    .all(|(index, word)| word == (owner, index))
            {
                return None;
            }
            let current = certificates.iter().find(|(id, _)| *id == owner)?.1;
            bound = Some(bound.map_or(current, |previous| previous.max(current)));
        }
        bound
    }

    /// Keep every partial dot product in ring64. Only the complete accumulated
    /// dot is truncated; rounding individual chunks would change the profile.
    /// Each private chunk issues fresh product material through the existing
    /// bounded party-local owner. No expanded primitive allowance is required.
    fn linear_chunks(
        &mut self,
        input: &View,
        weight: &MatrixView<'_>,
        rescale: bool,
    ) -> Result<View, Error> {
        let inner = *input.shape.last().ok_or(Error::Shape)?;
        let (columns, _) = weight.dimensions()?;
        let rows = input.words.len() / inner;
        let mut words = Words::new();
        for row in input.words.chunks_exact(inner) {
            for column in 0..columns {
                let mut accumulator = None;
                for start in (0..inner).step_by(puma::MAX_LANES) {
                    let count = puma::MAX_LANES.min(inner - start);
                    let left = self.gather(&row.slice(start..start + count))?;
                    let right = weight.gather(self, column, 1, start, count)?;
                    let partial =
                        self.product(left, right, Some(MatrixShape::new(1, count, 1)?), false)?;
                    accumulator = Some(match accumulator {
                        None => partial,
                        Some(previous) => self.affine(previous, Some(partial), affine(1, 1, 0))?,
                    });
                }
                let mut output = accumulator.ok_or(Error::Shape)?;
                if rescale {
                    let mut b = tensor::Builder::new(1, self.nodes[output].bounds)?;
                    let scaled = b.rescale(b.input(), 18)?;
                    output = self.tensor(output, b.finish(scaled)?)?;
                }
                words.push((output, 0));
            }
        }
        Ok(View {
            words,
            shape: vec![1, rows, columns],
        })
    }

    fn binary(&mut self, a: &Words, b: &Words, product: bool) -> Result<Words, Error> {
        compact_lowering::binary(self, a, b, product)
    }
}

impl compact_lowering::BinarySink for Builder {
    type View = Words;
    type Owner = usize;
    type Output = Words;

    fn len(view: &Words) -> usize {
        view.len()
    }

    fn slice(view: &Words, start: usize, end: usize) -> Words {
        view.slice(start..end)
    }

    fn empty(&self) -> Words {
        Words::new()
    }

    fn gather(&mut self, view: &Words) -> Result<usize, Error> {
        Builder::gather(self, view)
    }

    fn combine(&mut self, left: usize, right: usize, product: bool) -> Result<usize, Error> {
        if product {
            self.product(left, right, None, true)
        } else {
            self.affine(left, Some(right), affine(1, 1, 0))
        }
    }

    fn append(&mut self, output: &mut Words, owner: usize, lanes: usize) -> Result<(), Error> {
        output.extend((0..lanes).map(|i| (owner, i)));
        Ok(())
    }
}

#[derive(Clone, Copy, Default)]
struct PublicMatrixStatistics {
    row_l1: Option<u64>,
    row_squared_norm: Option<u128>,
    row_l1_squared_sum: Option<u128>,
}

/// Original public_matrix numeric/certificate contract. Source statistics and
/// complete input-row norms must come from the owning compiler's source/row
/// ledger. Closures retain the original lazy validation/construction order.
fn public_matrix_contract(
    [rows, inner, columns]: [usize; 3],
    transposed: bool,
    rescale: bool,
    product: impl FnOnce() -> Result<InputBounds, Error>,
    statistics: PublicMatrixStatistics,
    input_norm: impl FnOnce() -> Option<u128>,
) -> Result<(Interval, Option<u128>), Error> {
    if (rows as u64)
        .checked_mul(inner as u64)
        .and_then(|n| n.checked_mul(columns as u64))
        .is_none_or(|n| n > 1 << 32)
    {
        return Err(Error::Bounds);
    }
    let product = product()?;
    let energy = match statistics.row_l1_squared_sum {
        Some(sum) if !transposed && rescale => {
            let magnitude = u128::from(
                product
                    .left
                    .lower()
                    .unsigned_abs()
                    .max(product.left.upper().unsigned_abs()),
            );
            let radius = magnitude
                .checked_mul(ceil_sqrt(sum))
                .ok_or(Error::Bounds)?
                .div_ceil(ONE as u128)
                .checked_add(ceil_sqrt(columns as u128))
                .ok_or(Error::Bounds)?;
            Some(radius.checked_mul(radius).ok_or(Error::Bounds)?)
        }
        _ => None,
    };
    let ordinary = product.product().and_then(|product| {
        affine(i64::try_from(inner).map_err(|_| Error::Bounds)?, 0, 0).output_bounds(InputBounds {
            left: product,
            right: Interval::new(0, 0)?,
        })
    });
    let bounds = match statistics.row_l1 {
        Some(sum) if !transposed => {
            let magnitude = product
                .left
                .lower()
                .unsigned_abs()
                .max(product.left.upper().unsigned_abs());
            let mut bound = u128::from(magnitude) * u128::from(sum);
            if let Some(squared) = statistics
                .row_squared_norm
                .and_then(|w| input_norm()?.checked_mul(w))
            {
                let root = squared.isqrt();
                bound = bound.min(root + u128::from(root * root != squared));
            }
            let bound = i64::try_from(bound).map_err(|_| Error::Bounds)?;
            if let Ok(ordinary) = ordinary {
                Interval::new(ordinary.lower().max(-bound), ordinary.upper().min(bound))?
            } else {
                Interval::new(-bound, bound)?
            }
        }
        _ => ordinary?,
    };
    Ok((bounds, energy))
}

/// Original RMSNorm factory selection. General complete-row energy may refine
/// this constructor; attention-only source/block energy must never enter here.
fn rms_norm_program(
    public: bool,
    width: usize,
    bounds: Interval,
    gamma: &[i64],
    input_norm: Option<u128>,
) -> Result<(tensor::Program, Option<u128>), Error> {
    if public {
        let (program, squared) = match input_norm {
            Some(norm) => tensor::Program::rms_norm_with_input_norm(1, width, bounds, gamma, norm)?,
            None => tensor::Program::rms_norm_certified(1, width, bounds, gamma)?,
        };
        Ok((program, Some(squared)))
    } else {
        Ok((tensor::Program::rms_norm(1, width, bounds, gamma)?, None))
    }
}

fn residual_energy(left: u128, right: u128) -> Result<u128, Error> {
    let radius = ceil_sqrt(left)
        .checked_add(ceil_sqrt(right))
        .ok_or(Error::Bounds)?;
    radius.checked_mul(radius).ok_or(Error::Bounds)
}

fn affine(left: i64, right: i64, constant: i64) -> Affine {
    Affine {
        left,
        right,
        constant,
    }
}

fn ceil_sqrt(value: u128) -> u128 {
    let root = value.isqrt();
    root + u128::from(root * root != value)
}

impl Program {
    /// Validate the original source contract without constructing expanded nodes.
    /// This deliberately retains existing source-shape/work preflights; it is not
    /// a compact complete-request admission path or a manifest program-digest check.
    /// Shapes, lowered graphs and validation maps are new construction temporaries,
    /// not borrowed baseline; their aggregate D26 peak remains unaudited here.
    pub(super) fn validated_source_descriptors(
        manifest: &Manifest,
    ) -> Result<ValidatedSourceDescriptors<'_>, String> {
        manifest.execution.validate(manifest.weights)?;
        let shapes = Self::source_shapes_with_policy(
            &manifest.config,
            manifest.rows,
            manifest.outputs,
            manifest.weights,
            manifest.execution,
        )?;
        let first = lower_with_weights(
            &manifest.config,
            manifest.rows,
            manifest.outputs,
            manifest.weights,
        )?;
        validate_source_records(
            &first.prefill,
            &shapes,
            &manifest.tensors,
            manifest.weights,
            manifest.execution,
            &manifest.norms,
        )
    }

    /// Bounded source-mapped metadata admission runs before graph expansion.
    pub fn source_shapes(
        config: &[u8],
        rows: usize,
        outputs: usize,
    ) -> Result<BTreeMap<String, Vec<usize>>, String> {
        Self::source_shapes_with_weights(config, rows, outputs, WeightMode::Private)
    }

    pub fn source_shapes_with_weights(
        config: &[u8],
        rows: usize,
        outputs: usize,
        weights: WeightMode,
    ) -> Result<BTreeMap<String, Vec<usize>>, String> {
        Self::source_shapes_with_policy(config, rows, outputs, weights, ExecutionPolicy::default())
    }

    pub fn source_shapes_with_policy(
        config: &[u8],
        rows: usize,
        outputs: usize,
        weights: WeightMode,
        execution: ExecutionPolicy,
    ) -> Result<BTreeMap<String, Vec<usize>>, String> {
        execution.validate(weights)?;
        let plan = lower_with_weights(config, rows, outputs, weights)?;
        let (shapes, vocabulary) = source_shapes_from_plan(&plan, weights)?;
        // Construction views retire at last semantic use. Preserve the original
        // final-prefix lower/admission on the legacy byte-config route only.
        let final_plan;
        let prefill = if outputs > 1 {
            final_plan = lower_with_weights(config, rows + outputs - 1, 1, weights)?;
            &final_plan.prefill
        } else {
            &plan.prefill
        };
        construction_views(prefill, shapes.len())?;
        if weights == WeightMode::Public {
            let lookup_work =
                public_lookup_work_floor_with_policy(rows, outputs, vocabulary, execution.lookup)
                    .map_err(|e| format!("replicated public lookup admission: {e}"))?;
            if lookup_work.operations() > puma::MAX_PROGRAM_OPERATIONS {
                return Err(format!(
                    "replicated public lookup operation count {} exceeds limit {}",
                    lookup_work.operations(),
                    puma::MAX_PROGRAM_OPERATIONS
                ));
            }
            if lookup_work.cursors().aes_work > puma::prss::MAX_PROGRAM_AES_BLOCKS {
                return Err(format!(
                    "replicated public lookup AES bound {} exceeds limit {}",
                    lookup_work.cursors().aes_work,
                    puma::prss::MAX_PROGRAM_AES_BLOCKS,
                ));
            }
        }
        Ok(shapes)
    }

    pub fn compile(
        config: &[u8],
        rows: usize,
        outputs: usize,
        sources: &BTreeMap<String, SourceTensor>,
    ) -> Result<Self, String> {
        Self::compile_with_weights(config, rows, outputs, sources, WeightMode::Private)
    }

    pub fn compile_with_weights(
        config: &[u8],
        rows: usize,
        outputs: usize,
        sources: &BTreeMap<String, SourceTensor>,
        weights: WeightMode,
    ) -> Result<Self, String> {
        Self::compile_with_policy(
            config,
            rows,
            outputs,
            sources,
            weights,
            ExecutionPolicy::default(),
        )
    }

    pub fn compile_with_policy(
        config: &[u8],
        rows: usize,
        outputs: usize,
        sources: &BTreeMap<String, SourceTensor>,
        weights: WeightMode,
        execution: ExecutionPolicy,
    ) -> Result<Self, String> {
        Self::compile_with_norms(
            config,
            rows,
            outputs,
            sources,
            weights,
            execution,
            &SourceNorms::new(),
        )
    }

    pub fn compile_with_norms(
        config: &[u8],
        rows: usize,
        outputs: usize,
        sources: &BTreeMap<String, SourceTensor>,
        weights: WeightMode,
        execution: ExecutionPolicy,
        norms: &SourceNorms,
    ) -> Result<Self, String> {
        execution.validate(weights)?;
        let shapes = Self::source_shapes_with_policy(config, rows, outputs, weights, execution)?;
        let first = lower_with_weights(config, rows, outputs, weights)?;
        validate_source_records(&first.prefill, &shapes, sources, weights, execution, norms)?;
        let mut site = AdmissionSite::default();
        let mut builder = Builder {
            public_weights: weights == WeightMode::Public,
            checked_bound: execution.checked_bound(),
            lookup: execution.lookup,
            selection: execution.selection,
            token_source_binding: if execution.lookup == LookupPolicy::Trie {
                source_binding_with_norms(sources, norms)
            } else {
                [0; 32]
            },
            admission_stage: "source_import_envelope",
            ..Builder::default()
        };
        let result = (|| -> Result<Self, Error> {
            // Retained certificates and source overlap keep their original cap.
            builder.allocated = source_norms::import_envelope(norms)?.total_bytes;
            builder.admission_stage = "source_inputs";
            let mut inputs = BTreeMap::new();
            for (key, source) in sources {
                let bounds = Interval::new(source.lower, source.upper)?;
                if source.lower < -(1 << 40) || source.upper > 1 << 40 {
                    return Err(Error::Bounds);
                }
                if source.public_q18.is_some() {
                    continue;
                }
                let id = builder.push(
                    if weights == WeightMode::Public {
                        Code::PublicInput {
                            key: key.clone(),
                            digest: source.digest,
                            row_l1: source.row_l1,
                            row_squared_norm: source.row_squared_norm,
                            row_l1_squared_sum: source.row_l1_squared_sum,
                        }
                    } else {
                        Code::Input(key.clone())
                    },
                    source.shape.iter().product(),
                    bounds,
                )?;
                inputs.insert(
                    key.clone(),
                    SourceView {
                        owner: id,
                        shape: source.shape.clone(),
                    },
                );
                if let Some(norms) = norms.get(key) {
                    builder.public_norms.insert(id, norms.clone());
                }
            }
            let first =
                lower_with_weights(config, rows, outputs, weights).map_err(|_| Error::Binding)?;
            let vocabulary = first
                .prefill
                .operations
                .iter()
                .find(|op| op.operator == Op::OutputHead)
                .ok_or(Error::Binding)?
                .output_shape[1] as usize;
            let input = builder.push(
                Code::Input("input.tokens".into()),
                rows,
                Interval::new(0, vocabulary as i64 - 1)?,
            )?;
            let count = builder.nodes.len();
            let mut tokens = builder.view(input, vec![1, rows]);
            if execution.lookup == LookupPolicy::Trie {
                builder.register_initial_tokens(&tokens.words, vocabulary)?;
            }
            let mut selected = request_driver::drive(
                &mut request_driver::Expanded {
                    builder: &mut builder,
                    inputs: &inputs,
                    sources,
                    vocabulary,
                },
                config,
                rows,
                outputs,
                weights,
                &mut tokens,
                &mut site,
            )?;
            if execution.range_policy == RangePolicy::Checked {
                builder.final_acceptance(input, outputs, &mut selected)?;
            }
            builder.finish(
                count,
                selected,
                &mut site,
                |h| {
                    h.update(b"pllm.replicated-decoder.source-q18.full-prefix-final-ids.v1");
                    if weights == WeightMode::Public {
                        h.update(b"public-source-paged-ring64.v1");
                    }
                    if !execution.is_default() {
                        h.update(b"explicit-public-execution-policy.v1");
                        h.update(serde_json::to_vec(&execution).map_err(|_| Error::Binding)?);
                    }
                    h.update(config);
                    h.update((rows as u64).to_le_bytes());
                    h.update((outputs as u64).to_le_bytes());
                    h.update(serde_json::to_vec(sources).map_err(|_| Error::Binding)?);
                    if !norms.is_empty() {
                        h.update(b"public-head-row-norms.v1");
                        h.update(source_norms::digest(norms));
                    }
                    Ok(())
                },
                || source_binding_with_norms(sources, norms),
            )
        })();
        result.map_err(|e| {
            format!(
                "replicated decoder admission: {e}; public context {site}; public admission ledger {}",
                builder.admission_ledger()
            )
        })
    }
}

impl Builder {
    /// Original expanded finalization, shared with compiler-private fixtures.
    /// The caller supplies only the identity prefix/source binding; ordered node
    /// encoding, work limits, output retention and last-use retirement stay here.
    fn finish(
        &mut self,
        count: usize,
        selected: Vec<(usize, usize)>,
        site: &mut AdmissionSite,
        identity: impl FnOnce(&mut Sha256) -> Result<(), Error>,
        source_binding: impl FnOnce() -> [u8; 32],
    ) -> Result<Program, Error> {
        let builder = self;
        let mut last: Vec<_> = (0..builder.nodes.len()).collect();
        site.step = None;
        site.operator = None;
        site.layer = None;
        site.inputs = [None; 2];
        builder.admission_stage = "complete_schedule_work";
        let work = schedule_work(&builder.nodes)?;
        let mut h = Sha256::new();
        identity(&mut h)?;
        #[cfg(test)]
        let finalization_prefix = h.clone();
        for (id, node) in builder.nodes.iter().enumerate() {
            for input in node.inputs() {
                last[input] = id;
            }
            hash_node_header(&mut h, node.lanes, node.bounds);
            match &node.code {
                Code::Input(key) => {
                    h.update([0]);
                    h.update((key.len() as u64).to_le_bytes());
                    h.update(key);
                }
                Code::PublicInput { key, digest, .. } => {
                    h.update([5]);
                    h.update((key.len() as u64).to_le_bytes());
                    h.update(key);
                    h.update(digest);
                }
                Code::PublicValues(values) => {
                    h.update([7]);
                    for n in values {
                        h.update(n.to_le_bytes());
                    }
                }
                Code::PublicProduct {
                    input,
                    weight,
                    rows,
                    inner,
                    columns,
                    transposed,
                } => {
                    hash_public_product(
                        &mut h,
                        *input,
                        *weight,
                        *rows,
                        *inner,
                        *columns,
                        *transposed,
                    );
                }
                Code::PublicLookup {
                    input,
                    weight,
                    plan,
                } => {
                    h.update([10]);
                    for n in [*input, *weight] {
                        h.update((n as u64).to_le_bytes());
                    }
                    h.update(plan.digest());
                }
                Code::Selection { input, plan } => {
                    h.update([11]);
                    h.update((*input as u64).to_le_bytes());
                    h.update(plan.digest());
                }
                Code::TrieLookup {
                    input,
                    weight,
                    plan,
                } => {
                    h.update([12]);
                    h.update((*input as u64).to_le_bytes());
                    h.update((*weight as u64).to_le_bytes());
                    h.update(plan.digest());
                }
                Code::CheckedRange { input, program } => {
                    h.update([9]);
                    h.update((*input as u64).to_le_bytes());
                    h.update(program.digest());
                }
                Code::Gather(items) => {
                    hash_gather(&mut h, items.iter());
                }
                Code::SourceGather(view) => {
                    // Hash the same logical gather as the original expanded
                    // list. Compact storage changes neither PRSS contexts nor
                    // the ordered primitive schedule.
                    hash_gather(&mut h, view.indices().map(|index| (view.owner, index)));
                }
                Code::Affine {
                    left,
                    right,
                    coefficients: c,
                } => {
                    hash_affine(&mut h, *left, *right, *c);
                }
                Code::Product {
                    left,
                    right,
                    shape,
                    rescale,
                    norm,
                } => {
                    hash_product(&mut h, *left, *right, *shape, *rescale, norm.as_deref());
                }
                Code::Tensor { input, program } => {
                    hash_tensor(&mut h, *input, program);
                }
                Code::TensorLoop {
                    input,
                    program,
                    tail,
                } => {
                    hash_tensor_loop(&mut h, *input, program, tail.as_ref());
                }
            }
        }
        if work.operations() == 0 {
            return Err(Error::Bounds);
        }
        site.complete_work = Some((work.operations(), work.cursors().aes_work));
        builder.admission_stage = "complete_work_limits";
        work.limits()?;
        for &(id, index) in &selected {
            last[id] = builder.nodes.len();
            h.update((id as u64).to_le_bytes());
            h.update((index as u64).to_le_bytes());
        }
        let mut retire = vec![vec![]; builder.nodes.len()];
        for (id, end) in last.into_iter().enumerate() {
            if end < retire.len() {
                retire[end].push(id);
            }
        }
        // Deliberately after original work/identity/limit diagnostics succeed.
        // This checks borrowed metadata only; no compact executable or global
        // memory admission is enabled, and original nodes/Arcs remain owners.
        #[cfg(test)]
        schedule_finalization::compare_original(
            &builder.nodes,
            &selected,
            &finalization_prefix,
            &h,
            work,
            &retire,
        );
        Ok(Program {
            nodes: std::mem::take(&mut builder.nodes),
            inputs: count,
            outputs: selected,
            retire,
            digest: h.finalize().into(),
            payload_bytes: builder.allocated,
            work,
            source_binding: source_binding(),
            request_custody: None,
        })
    }
}

// Pending E ACK: type re-export only, no constructor or field exposure.
pub(in crate::replicated_decoder) use request_origin::compact_storage::{ControllerAccounting, EmissionContext};

#[cfg(test)]
#[path = "prepared_tensor_map/prepared_tensor_request_rescale_tests.rs"]
pub(in crate::replicated_decoder) mod request_rescale_tests;

#[cfg(test)]
#[path = "prepared_tensor_map/prepared_tensor_request_silu_tests.rs"]
mod request_silu_tests;

// Canonical expanded encoding, shared with allocation-free candidate lowering.
// These helpers do not assign a new identity or admit a compact executable.
fn hash_node_header(h: &mut Sha256, lanes: usize, bounds: Interval) {
    h.update((lanes as u64).to_le_bytes());
    h.update(bounds.lower().to_le_bytes());
    h.update(bounds.upper().to_le_bytes());
}

fn hash_affine(h: &mut Sha256, left: usize, right: Option<usize>, c: Affine) {
    h.update([2]);
    h.update((left as u64).to_le_bytes());
    h.update(right.map_or(u64::MAX, |v| v as u64).to_le_bytes());
    for v in [c.left, c.right, c.constant] {
        h.update(v.to_le_bytes());
    }
}

fn hash_public_product(
    h: &mut Sha256,
    input: usize,
    weight: usize,
    rows: usize,
    inner: usize,
    columns: usize,
    transposed: bool,
) {
    h.update([6, u8::from(transposed)]);
    for n in [input, weight, rows, inner, columns] {
        h.update((n as u64).to_le_bytes());
    }
}

fn hash_tensor_loop(
    h: &mut Sha256,
    input: LoopInput,
    program: &Arc<tensor::Program>,
    tail: Option<&Arc<tensor::Program>>,
) {
    h.update([8]);
    match input {
        LoopInput::Slice(id) => {
            h.update([0]);
            h.update((id as u64).to_le_bytes());
        }
        LoopInput::TokenOffsets { owner, index } => {
            h.update([1]);
            h.update((owner as u64).to_le_bytes());
            h.update((index as u64).to_le_bytes());
        }
    }
    h.update(program.digest());
    h.update([tail.is_some() as u8]);
    if let Some(tail) = tail {
        h.update(tail.digest());
    }
}

fn hash_tensor(h: &mut Sha256, input: usize, program: &Arc<tensor::Program>) {
    h.update([4]);
    h.update((input as u64).to_le_bytes());
    h.update(program.digest());
}

fn hash_gather(h: &mut Sha256, words: impl IntoIterator<Item = (usize, usize)>) {
    h.update([1]);
    for (owner, index) in words {
        h.update((owner as u64).to_le_bytes());
        h.update((index as u64).to_le_bytes());
    }
}

fn hash_product(
    h: &mut Sha256,
    left: usize,
    right: usize,
    shape: Option<MatrixShape>,
    rescale: bool,
    norm: Option<&puma::multiply::norm::RowEnergyProof>,
) {
    // A certified complete dot binds its obligations as well as operand owners.
    h.update([if norm.is_some() { 13 } else { 3 }, u8::from(rescale)]);
    h.update((left as u64).to_le_bytes());
    h.update((right as u64).to_le_bytes());
    for n in shape.map_or([0, 0, 0], MatrixShape::dimensions) {
        h.update((n as u64).to_le_bytes());
    }
    if let Some(proof) = norm {
        h.update(proof.digest());
    }
}

impl Program {
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
    pub(super) fn bind(&mut self, source: [u8; 32], composition: [u8; 32]) {
        let mut h = Sha256::new();
        h.update(b"pllm.replicated-decoder.source-composition.v1");
        h.update(self.digest);
        h.update(source);
        h.update(composition);
        self.digest = h.finalize().into();
    }
    pub fn payload_bytes(&self) -> usize {
        self.payload_bytes
    }
    pub fn operation_limit(&self) -> u64 {
        self.work.operations()
    }
    pub fn work_budget(&self) -> WorkBudget {
        self.work
    }
    pub fn inputs(&self) -> impl Iterator<Item = (&str, usize)> {
        self.nodes[..self.inputs]
            .iter()
            .filter_map(|n| match &n.code {
                Code::Input(key) => Some((key.as_str(), n.lanes)),
                Code::PublicInput { .. } => None,
                _ => unreachable!(),
            })
    }
    pub fn setup(
        &self,
        session: [u8; 32],
        party: puma::Party,
    ) -> Result<multiply::SetupParty, Error> {
        let limits = self.work.limits()?;
        multiply::SetupParty::new(puma::Context::new(session, self.digest, 0)?, party, limits)
    }
}

/// Source-schedule operations and per-party AES upper bounds for lookup alone,
/// not timing or complete-decoder costs. D9 pays for every complete prefix.
/// Reject known impossible work before importing any checkpoint values.
fn public_lookup_work_floor_with_policy(
    rows: usize,
    outputs: usize,
    vocabulary: usize,
    lookup: LookupPolicy,
) -> Result<WorkBudget, Error> {
    if rows == 0 || outputs == 0 || vocabulary == 0 {
        return Err(Error::Shape);
    }
    if lookup == LookupPolicy::Trie {
        let mut work = WorkBudget::default();
        for selection in 0..outputs {
            let n = rows.checked_add(selection).ok_or(Error::Bounds)?;
            for start in (0..n).step_by(8) {
                work = work.checked_add(runtime::trie::Plan::work_for_geometry(
                    8.min(n - start),
                    vocabulary,
                )?)?;
            }
        }
        return Ok(work);
    }
    let bounds = Interval::new(-(vocabulary as i64 - 1), vocabulary as i64 - 1)?;
    let width = vocabulary.min(128);
    let equality = match lookup {
        LookupPolicy::PerToken => tensor::Program::equal_zero,
        LookupPolicy::Tiled => tensor::Program::equal_zero_bounded,
        LookupPolicy::Trie => unreachable!("trie work returned above"),
    };
    let program = equality(width, bounds)?;
    let mut charge = program
        .work_budget()
        .checked_mul((vocabulary / width) as u64)?;
    if vocabulary % width != 0 {
        charge = charge.checked_add(equality(vocabulary % width, bounds)?.work_budget())?;
    }
    let prefixes = (0..outputs)
        .try_fold(0usize, |n, i| n.checked_add(rows.checked_add(i)?))
        .ok_or(Error::Bounds)?;
    charge.checked_mul(prefixes as u64)
}

pub(super) fn lower_with_weights(
    config: &[u8],
    rows: usize,
    outputs: usize,
    weights: WeightMode,
) -> Result<pllm_models::DecoderPlan, String> {
    if config.is_empty()
        || config.len() > 16384
        || rows == 0
        || outputs == 0
        || rows.checked_add(outputs).is_none_or(|n| n > 40)
        || outputs > 8
    {
        return Err("replicated configuration/workload bound".into());
    }
    pllm_models::lower_model_json_with_budget(
        config,
        DecoderWorkload {
            batch: 1,
            max_input_tokens: rows as u64,
            max_new_tokens: outputs as u64,
        },
        DecoderExpansionBudget {
            max_layers: if weights == WeightMode::Public { 64 } else { 4 },
            max_feature_width: 8192,
            max_attention_heads: 16,
            max_vocabulary: if weights == WeightMode::Public {
                262144
            } else {
                4096
            },
            max_tensor_elements: if weights == WeightMode::Public {
                MAX_PUBLIC_SOURCE_VALUES
            } else {
                MAX_VALUES
            } as u64,
        },
    )
    .map_err(|e| e.to_string())
}
