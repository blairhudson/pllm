//! Original numeric lowering with expanded and compact publication sinks.
//!
//! This is not a second execution engine or complete-request admission. The
//! default Builder remains unchanged. CompactGraph publishes compiler-owned
//! projection/guarded-SiLU/binary views into the shared OrderedGraph frontier;
//! tiled outputs feed later projection/residual lowering without expanded refs.
//! Public PerToken lookup retains original equality-loop/public-matrix owners.
//! Last-row and token-feedback aliases retain original owner/history mappings;
//! feedback is the original View identity, not a semantic-shape rewrite.
//! Checked rank-one through rank-four dimensions and bounded lazy reshape
//! permutations preserve original ordered coordinates, without proof promotion.
//! Closed prepared capabilities retain their original Arc owners. Source/general
//! row/attention proof scopes remain separate. The earlier borrowed Multiply
//! candidate is retained only for numeric-sink qualification.
//! Public RMSNorm rows use the original shared semantic driver/native
//! factory. Sealed local checked rows retain original guard/sticky custody;
//! cross-prefix checked/sticky exports remain closed.
//! No compact runtime or complete-request admission is constructed here.
use super::*;

#[path = "compact_norm.rs"]
mod compact_norm;

#[path = "compact_validity.rs"]
mod compact_validity;

#[path = "compact_frozen_region.rs"]
pub(in crate::replicated_decoder) mod frozen_region;

/// Original three-node secret select. The final envelope is justified only by
/// the original native comparison bit, not by a caller-labelled [0,1] interval.
pub(super) trait SelectSink {
    type Owner: Copy;
    fn bounds(&self, owner: Self::Owner) -> Interval;
    fn affine(
        &mut self,
        left: Self::Owner,
        right: Self::Owner,
        coefficients: Affine,
    ) -> Result<Self::Owner, Error>;
    fn product(&mut self, left: Self::Owner, right: Self::Owner) -> Result<Self::Owner, Error>;
    fn selected_affine(
        &mut self,
        no: Self::Owner,
        product: Self::Owner,
        bounds: Interval,
    ) -> Result<Self::Owner, Error>;
}

pub(super) fn select<S: SelectSink>(
    sink: &mut S,
    bit: S::Owner,
    yes: S::Owner,
    no: S::Owner,
) -> Result<S::Owner, Error> {
    let yes_bounds = sink.bounds(yes);
    let no_bounds = sink.bounds(no);
    let bounds = Interval::new(
        yes_bounds.lower().min(no_bounds.lower()),
        yes_bounds.upper().max(no_bounds.upper()),
    )?;
    let difference = sink.affine(yes, no, affine(1, -1, 0))?;
    let product = sink.product(bit, difference)?;
    sink.selected_affine(no, product, bounds)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::replicated_decoder) enum ArgmaxField {
    Values,
    Indices,
}

pub(super) trait ArgmaxSink {
    type Owner: Copy;
    type Output;
    fn indices(&mut self, lanes: usize) -> Result<(), Error>;
    fn begin_round(&mut self, round: usize, lanes: usize) -> Result<(), Error>;
    fn gather_pairs(
        &mut self,
        field: ArgmaxField,
        start: usize,
        pairs: usize,
        parity: usize,
    ) -> Result<Self::Owner, Error>;
    fn difference(&mut self, left: Self::Owner, right: Self::Owner) -> Result<Self::Owner, Error>;
    fn negative(&mut self, difference: Self::Owner) -> Result<Self::Owner, Error>;
    fn select(
        &mut self,
        bit: Self::Owner,
        yes: Self::Owner,
        no: Self::Owner,
    ) -> Result<Self::Owner, Error>;
    fn publish_pairs(
        &mut self,
        best: Self::Owner,
        index: Self::Owner,
        lanes: usize,
    ) -> Result<(), Error>;
    fn carry(&mut self, index: usize) -> Result<(), Error>;
    fn end_round(&mut self) -> Result<(), Error>;
    fn finish(&mut self) -> Result<Self::Output, Error>;
}

/// Extracted original Builder::argmax_tree traversal. Pairing, 256-value chunks,
/// strict-negative comparison, BOTH select masks, and odd carries are shared.
/// The expanded sink retains Words; the compact sink retains no level lanes.
pub(super) fn argmax_tree<S: ArgmaxSink>(
    sink: &mut S,
    mut lanes: usize,
) -> Result<S::Output, Error> {
    if lanes == 0 {
        return Err(Error::Shape);
    }
    sink.indices(lanes)?;
    let mut round = 0;
    while lanes > 1 {
        sink.begin_round(round, lanes)?;
        for start in (0..lanes).step_by(256) {
            let length = (lanes - start).min(256);
            let pairs = length / 2;
            if pairs > 0 {
                let left = sink.gather_pairs(ArgmaxField::Values, start, pairs, 0)?;
                let right = sink.gather_pairs(ArgmaxField::Values, start, pairs, 1)?;
                let difference = sink.difference(left, right)?;
                let less = sink.negative(difference)?;
                let best = sink.select(less, right, left)?;
                let left_id = sink.gather_pairs(ArgmaxField::Indices, start, pairs, 0)?;
                let right_id = sink.gather_pairs(ArgmaxField::Indices, start, pairs, 1)?;
                let index = sink.select(less, right_id, left_id)?;
                sink.publish_pairs(best, index, pairs)?;
            }
            if length % 2 != 0 {
                sink.carry(start + length - 1)?;
            }
        }
        sink.end_round()?;
        lanes = lanes.div_ceil(2);
        round += 1;
    }
    sink.finish()
}

#[path = "compact_admission.rs"]
pub(in crate::replicated_decoder) mod compact_admission;

#[path = "compact_input_owner.rs"]
mod input_owner;

#[cfg(test)]
#[path = "compact_contraction_driver_tests.rs"]
mod contraction_driver_tests;

#[cfg(test)]
#[path = "compact_projection_bias_tests.rs"]
mod projection_bias_tests;

#[cfg(test)]
#[path = "compact_closed_factory_order_regressions.rs"]
mod closed_factory_order_regressions;

pub(super) trait BinarySink {
    type View;
    type Owner: Copy;
    type Output;

    fn len(view: &Self::View) -> usize;
    fn slice(view: &Self::View, start: usize, end: usize) -> Self::View;
    fn empty(&self) -> Self::Output;
    fn gather(&mut self, view: &Self::View) -> Result<Self::Owner, Error>;
    fn combine(
        &mut self,
        left: Self::Owner,
        right: Self::Owner,
        product: bool,
    ) -> Result<Self::Owner, Error>;
    fn append(
        &mut self,
        output: &mut Self::Output,
        owner: Self::Owner,
        lanes: usize,
    ) -> Result<(), Error>;
}

/// The original Builder::binary order, including both gathers before arithmetic,
/// independent 128-lane products and the final partial tile. No fusion, complete
/// row floor, regenerated tensor program or body-family substitute is permitted.
pub(super) fn binary<S: BinarySink>(
    sink: &mut S,
    a: &S::View,
    b: &S::View,
    product: bool,
) -> Result<S::Output, Error> {
    let len = S::len(a);
    if len != S::len(b) {
        return Err(Error::Shape);
    }
    let mut output = sink.empty();
    for start in (0..len).step_by(128) {
        let end = start + 128.min(len - start);
        let a = S::slice(a, start, end);
        let b = S::slice(b, start, end);
        let left = sink.gather(&a)?;
        let right = sink.gather(&b)?;
        let owner = sink.combine(left, right, product)?;
        sink.append(&mut output, owner, end - start)?;
    }
    Ok(output)
}

/// Extracted unchanged from product_with_norm. Compiler-owned complete-dot
/// proofs retain their input/shape checks; ordinary elementwise products retain
/// the original floor envelope, not a caller-selected or body-derived interval.
pub(super) fn product_bounds(
    bounds: InputBounds,
    terms: usize,
    shaped: bool,
    rescale: bool,
    norm: Option<puma::multiply::norm::RowEnergyProof>,
) -> Result<Interval, Error> {
    if let Some(proof) = norm {
        if !shaped || proof.inner() != terms || proof.input_bounds() != bounds {
            return Err(Error::Binding);
        }
        if rescale {
            proof.output_bounds()
        } else {
            Ok(proof.raw_bounds())
        }
    } else if rescale {
        bounds.dot_output(terms)
    } else {
        bounds.dot_product(terms)
    }
}

/// Borrowing the original node prevents the owner/bounds contract from changing
/// while a candidate exists. This is not an external source authentication proof.
#[derive(Clone, Copy)]
struct OriginalSlice<'a> {
    node: &'a Node,
    owner: usize,
    offset: usize,
    lanes: usize,
}

impl<'a> OriginalSlice<'a> {
    fn from_words(builder: &'a Builder, words: &Words) -> Result<Self, Error> {
        let (owner, offset) = words.iter().next().ok_or(Error::Shape)?;
        let node = builder.nodes.get(owner).ok_or(Error::Binding)?;
        if matches!(node.code, Code::PublicInput { .. }) {
            // File/import-owner baselines have no compact admission contract yet.
            return Err(Error::Binding);
        }
        if words.len() > MAX_VALUES
            || offset
                .checked_add(words.len())
                .is_none_or(|n| n > node.lanes)
            || !words
                .iter()
                .enumerate()
                .all(|(index, word)| word == (owner, offset + index))
        {
            return Err(Error::Shape);
        }
        Ok(Self {
            node,
            owner,
            offset,
            lanes: words.len(),
        })
    }
}

#[derive(Clone, Copy)]
struct Value {
    id: usize,
    lanes: usize,
    bounds: Interval,
    work: WorkBudget,
}

#[derive(Default)]
struct Summary {
    lanes: usize,
    tiles: usize,
    bounds: Option<Interval>,
    work: WorkBudget,
}

struct CompactSink<'a> {
    first: usize,
    next: usize,
    owners: std::marker::PhantomData<&'a Node>,
}

impl<'a> BinarySink for CompactSink<'a> {
    type View = OriginalSlice<'a>;
    type Owner = Value;
    type Output = Summary;

    fn len(view: &Self::View) -> usize {
        view.lanes
    }

    fn slice(view: &Self::View, start: usize, end: usize) -> Self::View {
        OriginalSlice {
            offset: view.offset + start,
            lanes: end - start,
            ..*view
        }
    }

    fn empty(&self) -> Summary {
        Summary::default()
    }

    fn gather(&mut self, view: &Self::View) -> Result<Value, Error> {
        let value = Value {
            id: self.next,
            lanes: view.lanes,
            bounds: view.node.bounds,
            work: WorkBudget::default(),
        };
        self.next = self.next.checked_add(1).ok_or(Error::Bounds)?;
        Ok(value)
    }

    fn combine(&mut self, left: Value, right: Value, product: bool) -> Result<Value, Error> {
        if left.lanes != right.lanes || left.lanes > puma::MAX_LANES {
            return Err(Error::Shape);
        }
        let inputs = InputBounds {
            left: left.bounds,
            right: right.bounds,
        };
        let bounds = if product {
            product_bounds(inputs, 1, false, true, None)?
        } else {
            affine(1, 1, 0).output_bounds(inputs)?
        };
        let value = Value {
            id: self.next,
            lanes: left.lanes,
            bounds,
            work: if product {
                WorkBudget::product(left.lanes, true)?
            } else {
                WorkBudget::default()
            },
        };
        self.next = self.next.checked_add(1).ok_or(Error::Bounds)?;
        Ok(value)
    }

    fn append(&mut self, output: &mut Summary, owner: Value, lanes: usize) -> Result<(), Error> {
        let expected = output
            .tiles
            .checked_mul(3)
            .and_then(|n| self.first.checked_add(n))
            .and_then(|n| n.checked_add(2))
            .ok_or(Error::Bounds)?;
        if owner.id != expected || owner.lanes != lanes {
            return Err(Error::Binding);
        }
        // Contiguous single-owner inputs have identical declared bounds in every
        // tile. Do not replace heterogeneous tile bounds with a wider envelope.
        if output.bounds.is_some_and(|bounds| bounds != owner.bounds) {
            return Err(Error::Binding);
        }
        output.work = output.work.checked_add(owner.work)?;
        output.bounds = Some(owner.bounds);
        output.lanes = output.lanes.checked_add(lanes).ok_or(Error::Bounds)?;
        output.tiles += 1;
        Ok(())
    }
}

/// Unadmitted, borrowed lowering result, NOT a Program/Plan or executable region.
/// Source/certificate ownership remains with the original Builder. No certificate
/// is created, rebound or inferred from a semantic label. Sticky validity, source
/// authentication, full-request lifetimes/work and routing remain enclosing gates.
pub(super) struct CompactBinary<'a> {
    left: OriginalSlice<'a>,
    right: OriginalSlice<'a>,
    first: usize,
    summary: Summary,
}

impl<'a> CompactBinary<'a> {
    pub(super) fn lower_unadmitted(
        builder: &'a Builder,
        op: &pllm_models::ModelOperation,
        a: &View,
        b: &View,
    ) -> Result<Self, Error> {
        if op.operator != Op::Multiply || op.inputs.len() != 2 {
            return Err(Error::Binding);
        }
        let lanes = op.output_shape.iter().try_fold(1usize, |n, dimension| {
            n.checked_mul(usize::try_from(*dimension).map_err(|_| Error::Bounds)?)
                .ok_or(Error::Bounds)
        })?;
        if lanes == 0 || lanes != a.words.len() || lanes != b.words.len() {
            return Err(Error::Shape);
        }
        let left = OriginalSlice::from_words(builder, &a.words)?;
        let right = OriginalSlice::from_words(builder, &b.words)?;
        let first = builder.nodes.len();
        let mut sink = CompactSink {
            first,
            next: first,
            owners: std::marker::PhantomData,
        };
        let summary = binary(&mut sink, &left, &right, true)?;
        Ok(Self {
            left,
            right,
            first,
            summary,
        })
    }

    pub(super) fn stored_descriptors(&self) -> usize {
        1
    }

    pub(super) fn expanded_nodes(&self) -> usize {
        3 * self.summary.tiles
    }

    /// Original virtual output IDs. No per-lane array or fragmented Words vector.
    pub(super) fn output_word(&self, index: usize) -> Result<(usize, usize), Error> {
        if index >= self.summary.lanes {
            return Err(Error::Shape);
        }
        Ok((self.first + 3 * (index / 128) + 2, index % 128))
    }

    pub(super) fn work_budget(&self) -> WorkBudget {
        self.summary.work
    }

    /// Append the original canonical node bytes, not a new execution identity.
    /// Original source ordinals, gather order and independent product floors stay
    /// unchanged. No Node/Code/Words or kernel is constructed while hashing.
    pub(super) fn hash_expanded_nodes(&self, h: &mut Sha256) {
        for start in (0..self.summary.lanes).step_by(128) {
            let lanes = 128.min(self.summary.lanes - start);
            let first = self.first + 3 * (start / 128);
            for input in [self.left, self.right] {
                hash_node_header(h, lanes, input.node.bounds);
                hash_gather(
                    h,
                    (0..lanes).map(|index| (input.owner, input.offset + start + index)),
                );
            }
            hash_node_header(
                h,
                lanes,
                self.summary.bounds.expect("nonempty checked lowering"),
            );
            hash_product(h, first, first + 1, None, true, None);
        }
    }
}

// The publication path below starts with compiler-owned IMPORT declarations,
// not an expanded Program or lowered arithmetic prefix. It is a region compiler:
// unsupported operators fail closed; no runtime/whole-request admission surface.
use super::super::prepared_tensor_map::PreparedTensorMap;

#[cfg(test)]
thread_local! {
    static RMS_NATIVE_ATTEMPTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mapping {
    Dense(usize),
    Products(usize),
    Rows { first: usize, width: usize },
    LookupRows { first: usize, width: usize },
}

#[derive(Clone)]
pub(in crate::replicated_decoder) struct PublishedView {
    origin: Arc<()>,
    slot: Option<usize>,
    mapping: Mapping,
    start: usize,
    lanes: usize,
    shape: [usize; 3],
    rank: u8,
    logical: ViewShape,
    order: LaneOrder,
    import_slot: Option<usize>,
    bounds: Interval,
}

/// Exact bounded semantic dimensions; the old three-axis physical envelope is
/// compatibility metadata only, never authority to treat KV heads as rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::replicated_decoder) struct ViewShape {
    dimensions: [usize; 4],
    rank: u8,
}

impl ViewShape {
    pub(super) fn new(dimensions: &[usize]) -> Result<Self, Error> {
        if !(1..=4).contains(&dimensions.len()) {
            return Err(Error::Shape);
        }
        let mut stored = [1; 4];
        let mut lanes = 1usize;
        for (index, &dimension) in dimensions.iter().enumerate() {
            if dimension == 0 {
                return Err(Error::Shape);
            }
            lanes = lanes.checked_mul(dimension).ok_or(Error::Bounds)?;
            if lanes > MAX_VALUES {
                return Err(Error::Bounds);
            }
            stored[index] = dimension;
        }
        Ok(Self {
            dimensions: stored,
            rank: dimensions.len() as u8,
        })
    }
    fn semantic(dimensions: &[u64]) -> Result<Self, Error> {
        if !(1..=4).contains(&dimensions.len()) {
            return Err(Error::Shape);
        }
        let mut stored = [1; 4];
        for (index, dimension) in dimensions.iter().enumerate() {
            stored[index] = usize::try_from(*dimension).map_err(|_| Error::Bounds)?;
        }
        Self::new(&stored[..dimensions.len()])
    }
    pub(in crate::replicated_decoder) fn dimensions(&self) -> &[usize] {
        &self.dimensions[..self.rank as usize]
    }
    fn lanes(self) -> Result<usize, Error> {
        self.dimensions()
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d).ok_or(Error::Bounds))
    }
    fn envelope(self) -> Result<[usize; 3], Error> {
        let d = self.dimensions;
        Ok(match self.rank {
            1 => [1, 1, d[0]],
            2 => [1, d[0], d[1]],
            3 => [d[0], d[1], d[2]],
            4 => [d[0], d[1].checked_mul(d[2]).ok_or(Error::Bounds)?, d[3]],
            _ => return Err(Error::Shape),
        })
    }
    fn row_width(self) -> usize {
        self.dimensions[self.rank as usize - 1]
    }
}

// Existing Words spans preserve exact order, but collecting a heads/sequence
// transpose grows a span table with geometry. Keep one bounded index transform
// over the existing Mapping/Word contract instead; inverse layouts cancel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LaneOrder {
    Linear,
    HeadsSequence {
        heads: usize,
        rows: usize,
        width: usize,
        to_heads: bool,
        base_start: usize,
    },
}

impl LaneOrder {
    fn index(self, index: usize) -> Result<usize, Error> {
        match self {
            Self::Linear => Ok(index),
            Self::HeadsSequence {
                heads,
                rows,
                width,
                to_heads,
                base_start,
            } => {
                let feature = index % width;
                let item = index / width;
                let mapped = if to_heads {
                    (item % rows)
                        .checked_mul(heads)
                        .and_then(|n| n.checked_add(item / rows))
                } else {
                    (item % heads)
                        .checked_mul(rows)
                        .and_then(|n| n.checked_add(item / heads))
                }
                .ok_or(Error::Bounds)?;
                mapped
                    .checked_mul(width)
                    .and_then(|n| n.checked_add(feature))
                    .and_then(|n| n.checked_add(base_start))
                    .ok_or(Error::Bounds)
            }
        }
    }
}

struct BoundImport<'g> {
    semantic: &'g str,
    view: PublishedView,
}

impl PublishedView {
    pub(super) fn word(&self, index: usize) -> Result<(usize, usize), Error> {
        if index >= self.lanes {
            return Err(Error::Shape);
        }
        let index = self
            .order
            .index(self.start.checked_add(index).ok_or(Error::Bounds)?)?;
        Ok(match self.mapping {
            Mapping::Dense(owner) => (owner, index),
            Mapping::Products(first) => (first + 3 * (index / 128) + 2, index % 128),
            Mapping::Rows { first, width } => (first + 2 * (index / width) + 1, index % width),
            Mapping::LookupRows { first, width } => {
                (first + 3 * (index / width) + 2, index % width)
            }
        })
    }

    fn words(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.lanes).map(|i| self.word(i).expect("checked compact layout"))
    }
}

struct PublicationSink {
    first: usize,
    next: usize,
}
impl BinarySink for PublicationSink {
    type View = PublishedView;
    type Owner = Value;
    type Output = Summary;
    fn len(v: &PublishedView) -> usize {
        v.lanes
    }
    fn slice(v: &PublishedView, start: usize, end: usize) -> PublishedView {
        let mut v = v.clone();
        v.start += start;
        v.lanes = end - start;
        v
    }
    fn empty(&self) -> Summary {
        Summary::default()
    }
    fn gather(&mut self, v: &PublishedView) -> Result<Value, Error> {
        let id = self.next;
        self.next = self.next.checked_add(1).ok_or(Error::Bounds)?;
        Ok(Value {
            id,
            lanes: v.lanes,
            bounds: v.bounds,
            work: WorkBudget::default(),
        })
    }
    fn combine(&mut self, left: Value, right: Value, product: bool) -> Result<Value, Error> {
        // Same original numeric sink, not the body family's binary constructor.
        let mut sink = CompactSink {
            first: self.first,
            next: self.next,
            owners: std::marker::PhantomData,
        };
        let value = sink.combine(left, right, product)?;
        self.next = sink.next;
        Ok(value)
    }
    fn append(&mut self, out: &mut Summary, owner: Value, lanes: usize) -> Result<(), Error> {
        let mut sink = CompactSink {
            first: self.first,
            next: self.next,
            owners: std::marker::PhantomData,
        };
        sink.append(out, owner, lanes)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MapKind {
    Rescale18,
    GuardedSilu,
}
enum MapStorage {
    Owned(PreparedTensorMap),
    Previous(usize),
}
struct MapOwner {
    kind: MapKind,
    input: Interval,
    lanes: usize,
    storage: MapStorage,
}

/// Original imported bias, not expanded row repetitions or a new certificate.
struct ProjectionBias<'a> {
    owner: usize,
    source: &'a SourceTensor,
    norms: Option<&'a BlockNorms>,
    floor_bounds: Interval,
    coefficients: Affine,
}

impl ProjectionBias<'_> {
    fn nodes(lanes: usize) -> Result<usize, Error> {
        lanes.div_ceil(128).checked_mul(3).ok_or(Error::Bounds)
    }
    fn bounds(&self) -> Result<Interval, Error> {
        Interval::new(self.source.lower, self.source.upper)
    }
}

enum PublishedKind<'a> {
    TokenLookup {
        weight: usize,
        source: &'a SourceTensor,
        plan: CompactTokenLookup,
    },
    Projection {
        weight: usize,
        source: &'a SourceTensor,
        norms: Option<&'a BlockNorms>,
        geometry: [usize; 3],
        raw_bounds: Interval,
        row_energy: Option<u128>,
        input_energy: Option<u128>,
        bias: Option<ProjectionBias<'a>>,
    },
    Silu,
    RmsNorm {
        gamma: &'a SourceTensor,
        input_energy: Option<u128>,
        output_energy: u128,
        program: Arc<tensor::Program>,
        owns_program: bool,
        checked: Option<compact_validity::CheckedRows<'a>>,
        row_bounds: Interval,
    },
    LastToken {
        row_energy: Option<u128>,
    },
    /// Original graph::TokenFeedback clones the input View, including its shape.
    /// Retain both immutable mappings in PublishedOp; no history lane array.
    TokenFeedback,
    Reshape,
    Argmax(CompactArgmax),
    Binary {
        right: PublishedView,
        product: bool,
        row_energy: Option<u128>,
    },
}

struct PublishedOp<'a> {
    first: usize,
    input: PublishedView,
    output: PublishedView,
    kind: PublishedKind<'a>,
    prepared: Option<MapOwner>,
}

/// Read-only physical layout/census for the separate admission owner. Virtual
/// IDs are original schedule coordinates, NOT allocated value/retirement slots.
#[derive(Clone, Copy)]
pub(in crate::replicated_decoder) enum PhysicalLayout {
    Dense {
        owner: usize,
    },
    BinaryTiles {
        first: usize,
        tile_lanes: usize,
        node_stride: usize,
        output_node: usize,
    },
    TensorRows {
        first: usize,
        row_width: usize,
        node_stride: usize,
        output_node: usize,
    },
}

#[derive(Clone, Copy)]
pub(in crate::replicated_decoder) struct PhysicalView {
    pub layout: PhysicalLayout,
    pub lanes: usize,
    /// Compatibility envelope only; use logical_shape for semantic dimensions.
    pub shape: [usize; 3],
    pub bounds: Interval,
    /// Physical three-axis normalization plus ORIGINAL semantic rank. A last-row
    /// alias retains its original mapping and records its logical lane offset.
    pub semantic_rank: u8,
    pub offset: usize,
    pub logical_shape: ViewShape,
    pub lane_order: PhysicalLaneOrder,
}

#[derive(Clone, Copy)]
pub(in crate::replicated_decoder) enum PhysicalLaneOrder {
    Linear,
    HeadsSequence {
        heads: usize,
        rows: usize,
        width: usize,
        to_heads: bool,
        base_start: usize,
    },
}

impl<'a> CompactGraph<'a, 'a> {
    /// Genuine ORIGINAL declaration -> native compact emission. The source
    /// context, not a recovered input View / digest / expanded executable,
    /// supplies the initial import, exact source registry and native admission.
    pub(in crate::replicated_decoder::program) fn from_source_request(
        imports: &'a Builder,
        context: &'a request_origin::compact_storage::EmissionContext<'a>,
        cap: usize,
    ) -> Result<Self, Error> {
        let mut attempt = context.attempt();
        context.check_imports(imports)?;
        let semantic = &context.plan().plan().prefill;
        let validated = ValidatedSourceDescriptors {
            sources: context.registry().source_records(),
            norms: context.registry().source_norms(),
            weights: WeightMode::Public,
            execution: context.manifest().execution,
        };
        let mut graph =
            Self::with_limit_and_admission(imports, validated, semantic, 0, cap, 0, Some(context))?;
        let (owner, rows, bounds) = context.initial_input();
        if imports.nodes.len() != add_bytes(owner, 1)? || !semantic.state_inputs.is_empty() {
            return Err(Error::Binding);
        }
        let logical = ViewShape::semantic(&[1, u64::try_from(rows).map_err(|_| Error::Bounds)?])?;
        let view = PublishedView {
            origin: graph.origin.clone(),
            slot: None,
            mapping: Mapping::Dense(owner),
            start: 0,
            lanes: rows,
            shape: logical.envelope()?,
            rank: logical.rank,
            logical,
            order: LaneOrder::Linear,
            import_slot: Some(0),
            bounds,
        };
        if graph.bound_imports.len() == graph.bound_imports.capacity() {
            return Err(Error::Bounds);
        }
        graph.bound_imports.push(BoundImport {
            semantic: "input.tokens",
            view: view.clone(),
        });
        graph.validate_view(&view)?;
        let mut prelude = None;
        // Same Begin/input binding as Builder::graph, ONE prequoted frontier.
        graph::dispatch_begin(
            &mut graph.frontier,
            &view,
            &mut BTreeMap::new(),
            |event, _| {
                if !matches!(event, graph::DispatchStep::Begin) {
                    return Err(Error::Binding);
                }
                prelude = Some(context.prelude()?);
                Ok(None)
            },
        )?;
        let prelude = prelude.ok_or(Error::Binding)?;
        if prelude.first() != imports.nodes.len() {
            return Err(Error::Binding);
        }
        graph.next = add_bytes(prelude.first(), 2)?;
        graph.request_prelude = Some(prelude);
        attempt.commit();
        Ok(graph)
    }
}

/// Source-neutral BODY bias mapping. Source authority is the retained original
/// import/source owner, never these numeric IDs, intervals or a copied digest.
#[derive(Clone, Copy)]
pub(in crate::replicated_decoder) struct ProjectionBiasLayout {
    pub source_owner: usize,
    pub width: usize,
    pub source_bounds: Interval,
    pub input_owner: usize,
    pub input_bounds: Interval,
    pub first: usize,
    pub coefficients: Affine,
}

pub(in crate::replicated_decoder) enum DescriptorKind {
    /// Original public PerToken recipe: equality loop, gather, transposed raw
    /// public product per token. Not a body emitter or token-domain certificate.
    TokenLookup {
        source_owner: usize,
        geometry: [usize; 3], // tokens, vocabulary, embedding width
    },
    Projection {
        source_owner: usize,
        geometry: [usize; 3],
        raw_owner: usize,
        raw_bounds: Interval,
        bias: Option<ProjectionBiasLayout>,
    },
    GuardedSilu,
    RmsNorm {
        rows: usize,
        width: usize,
        input_energy: Option<u128>,
        output_energy: u128,
        program_digest: [u8; 32],
        gamma_digest: [u8; 32],
    },
    LastToken,
    /// Zero-node identity. Output retains ORIGINAL input shape/rank, even when
    /// semantic operation declares [1, 1] for the selected rank-one token.
    /// Request history/export owners must alias this mapping, not allocate a
    /// second selection value or infer a new acceptance/proof producer.
    TokenFeedback,
    /// Original indexed view alias: no node, value allocation or numeric work.
    Reshape,
    Argmax {
        lanes: usize,
        rounds: usize,
        compare_blocks: usize,
        expanded_nodes: usize,
    },
    Binary {
        multiply: bool,
    },
}

pub(in crate::replicated_decoder) struct DescriptorCensus {
    pub virtual_start: usize,
    pub virtual_end: usize,
    pub kind: DescriptorKind,
    pub inputs: [Option<PhysicalView>; 2],
    pub output: PhysicalView,
}

impl PublishedOp<'_> {
    // Shared immutable recipe view: freezing changes ownership, not publication.
    fn descriptor_census(&self, virtual_end: usize) -> DescriptorCensus {
        DescriptorCensus {
            virtual_start: self.first,
            virtual_end,
            kind: match &self.kind {
                PublishedKind::TokenLookup { weight, plan, .. } => DescriptorKind::TokenLookup {
                    source_owner: *weight,
                    geometry: [self.input.lanes, plan.vocabulary, self.output.shape[2]],
                },
                PublishedKind::Projection {
                    weight,
                    geometry,
                    raw_bounds,
                    bias,
                    ..
                } => DescriptorKind::Projection {
                    source_owner: *weight,
                    geometry: *geometry,
                    raw_owner: self.first + 1,
                    raw_bounds: *raw_bounds,
                    bias: bias.as_ref().map(|bias| ProjectionBiasLayout {
                        source_owner: bias.owner,
                        width: bias.source.shape[0],
                        source_bounds: bias.bounds().expect("validated original bias"),
                        input_owner: self.first + 2,
                        input_bounds: bias.floor_bounds,
                        first: self.first + 3,
                        coefficients: bias.coefficients,
                    }),
                },
                PublishedKind::Silu => DescriptorKind::GuardedSilu,
                PublishedKind::RmsNorm {
                    gamma,
                    input_energy,
                    output_energy,
                    program,
                    ..
                } => DescriptorKind::RmsNorm {
                    rows: self.output.shape[1],
                    width: self.output.shape[2],
                    input_energy: *input_energy,
                    output_energy: *output_energy,
                    program_digest: program.digest(),
                    gamma_digest: gamma.digest,
                },
                PublishedKind::LastToken { .. } => DescriptorKind::LastToken,
                PublishedKind::TokenFeedback => DescriptorKind::TokenFeedback,
                PublishedKind::Reshape => DescriptorKind::Reshape,
                PublishedKind::Argmax(plan) => DescriptorKind::Argmax {
                    lanes: plan.geometry.lanes,
                    rounds: plan.geometry.rounds,
                    compare_blocks: plan.geometry.blocks,
                    expanded_nodes: plan.geometry.nodes().expect("admitted argmax geometry"),
                },
                PublishedKind::Binary { product, .. } => {
                    DescriptorKind::Binary { multiply: *product }
                }
            },
            inputs: [
                Some(self.input.physical()),
                match &self.kind {
                    PublishedKind::Binary { right, .. } => Some(right.physical()),
                    _ => None,
                },
            ],
            output: self.output.physical(),
        }
    }
}

/// Native row publication, not a producer-qualified body CompleteRow. The
/// mapping lane must bind the eventual complete execution/producer/export owner.
/// No caller-supplied interval or digest is accepted to manufacture this record.
pub(in crate::replicated_decoder) struct NativeRowCensus<'s, 'a> {
    pub descriptor: usize,
    pub row: usize,
    pub owner: usize,
    pub offset: usize,
    pub width: usize,
    pub energy: u128,
    pub program: &'s Arc<tensor::Program>,
    pub gamma: &'a SourceTensor,
    pub source_binding: [u8; 32],
}

pub(in crate::replicated_decoder) struct ContainerCensus {
    pub imported_descriptors: usize,
    pub imported_node_capacity: usize,
    pub imported_node_bytes: usize,
    pub imported_declared_bytes: usize,
    pub stored_operations: usize,
    pub descriptor_capacity: usize,
    pub descriptor_bytes: usize,
    pub inline_bytes: usize,
    pub origin_arc_bytes: usize,
    /// Conservative preadmitted map envelope, NOT measured map capacity/backing.
    pub frontier_storage_bound: usize,
    pub expanded_virtual_nodes: usize,
    pub partial_work: WorkBudget,
    pub import_binding_capacity: usize,
    pub import_binding_bytes: usize,
}

#[derive(Clone, Copy)]
pub(in crate::replicated_decoder) struct FailureSite<'g> {
    pub step: usize,
    pub operation: &'g str,
    pub operator: Op,
    pub input_owners: [Option<usize>; 2],
    pub source_owner: Option<usize>,
    pub next_virtual_owner: usize,
}

impl PublishedView {
    fn physical(&self) -> PhysicalView {
        PhysicalView {
            layout: match self.mapping {
                Mapping::Dense(owner) => PhysicalLayout::Dense { owner },
                Mapping::Products(first) => PhysicalLayout::BinaryTiles {
                    first,
                    tile_lanes: 128,
                    node_stride: 3,
                    output_node: 2,
                },
                Mapping::Rows { first, width } => PhysicalLayout::TensorRows {
                    first,
                    row_width: width,
                    node_stride: 2,
                    output_node: 1,
                },
                Mapping::LookupRows { first, width } => PhysicalLayout::TensorRows {
                    first,
                    row_width: width,
                    node_stride: 3,
                    output_node: 2,
                },
            },
            lanes: self.lanes,
            shape: self.shape,
            bounds: self.bounds,
            semantic_rank: self.rank,
            offset: self.start,
            logical_shape: self.logical,
            lane_order: match self.order {
                LaneOrder::Linear => PhysicalLaneOrder::Linear,
                LaneOrder::HeadsSequence {
                    heads,
                    rows,
                    width,
                    to_heads,
                    base_start,
                } => PhysicalLaneOrder::HeadsSequence {
                    heads,
                    rows,
                    width,
                    to_heads,
                    base_start,
                },
            },
        }
    }
}

/// Original expanded row coordinates, derived only by this compiler. Not a body
/// CompleteRow ticket: conversion would require the original producer/export
/// qualification. The original source descriptor remains borrowed alongside it.
pub(super) struct ProjectionRow<'a> {
    pub owner: usize,
    pub offset: usize,
    pub width: usize,
    pub energy: u128,
    pub source_owner: usize,
    pub source: &'a SourceTensor,
    pub source_binding: [u8; 32],
}

/// Owned compact output publication in the SAME OrderedGraph retirement/traversal.
/// The allocation gate covers new frontier/descriptors/prepared construction and
/// a conservative actual imports-Builder baseline. Borrowed semantic graph and
/// manifest backing, authenticated files, runtime values/slots/frames/exports and
/// execution scratch remain EXCLUDED. This is not D26 whole-owner admission.
pub(in crate::replicated_decoder) struct CompactGraph<'g, 'a> {
    imports: &'a Builder,
    sources: &'a BTreeMap<String, SourceTensor>,
    norms: &'a SourceNorms,
    binding: [u8; 32],
    frontier: graph::OrderedGraph<'g, PublishedView>,
    frontier_bytes: usize,
    ops: Vec<PublishedOp<'a>>,
    bound_imports: Vec<BoundImport<'g>>,
    origin: Arc<()>,
    next: usize,
    cursor: usize,
    work: WorkBudget,
    limit: usize,
    outside_owner_bytes: usize,
    request_admission: Option<&'a request_origin::compact_storage::EmissionContext<'a>>,
    request_prelude: Option<request_origin::compact_storage::Prelude>,
    prebound_inputs: Option<input_owner::InputOwner<'a>>,
    validity: compact_validity::RequestValidity<'a>,
    contraction_control_bytes: std::cell::Cell<usize>,
    construction_peak: std::cell::Cell<usize>,
    // Same historical arms, excluding only explicitly charged original source
    // subowners (RMSNorm's borrowed gamma). Imports/outside/frontier/recipes and
    // native construction remain included. Freeze adds full source/semantic
    // backing once; normalizing EACH arm preserves maxima when gamma differs.
    construction_peak_without_source: std::cell::Cell<usize>,
    failed: bool,
    failure: Option<FailureSite<'g>>,
}

fn add_bytes(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Bounds)
}
fn array_bytes<T>(n: usize) -> Result<usize, Error> {
    n.checked_mul(std::mem::size_of::<T>()).ok_or(Error::Bounds)
}

/// Existing floor contract, with the original tensor-loop call resets. Query
/// full and tail independently; combining lanes would change PRSS fill rounding.
fn contraction_rescale_work(lanes: usize) -> Result<WorkBudget, Error> {
    if lanes == 0 || lanes > MAX_VALUES {
        return Err(Error::Shape);
    }
    let full = lanes.min(128);
    let mut work = WorkBudget::rescale(full)?.checked_mul((lanes / full) as u64)?;
    if lanes % full != 0 {
        work = work.checked_add(WorkBudget::rescale(lanes % full)?)?;
    }
    Ok(work)
}

/// Known contraction work only, not a complete-decoder recipe or admission bit.
/// Original global token declaration supplies geometry/bounds. Native APIs own
/// equality/floor schedules; unsupported families are neither replaced nor priced.
fn pending_biased_projection_lanes(
    imports: &Builder,
    semantic: &DecoderGraph,
    op: &pllm_models::ModelOperation,
    sources: &BTreeMap<String, SourceTensor>,
) -> Result<Option<usize>, Error> {
    let Some(source) = op
        .attributes
        .get("weight")
        .and_then(|value| value.as_str())
        .and_then(|key| sources.get(key))
    else {
        return Ok(None);
    };
    let [columns, inner] = source.shape.as_slice() else {
        return Ok(None);
    };
    if *inner == 0 {
        return Ok(None);
    }
    let Some(input) = op.inputs.first() else {
        return Ok(None);
    };
    let lanes = if let Some(node) = imports
        .nodes
        .iter()
        .find(|node| matches!(&node.code, Code::Input(key) if key == input))
    {
        node.lanes
    } else if let Some(producer) = semantic
        .operations
        .iter()
        .find(|producer| producer.id == *input)
    {
        ViewShape::semantic(&producer.output_shape)?.lanes()?
    } else {
        return Ok(None);
    };
    if lanes % inner != 0 {
        return Ok(None);
    }
    Ok(Some(
        (lanes / inner)
            .checked_mul(*columns)
            .filter(|lanes| *lanes <= MAX_VALUES)
            .ok_or(Error::Bounds)?,
    ))
}

fn pending_contraction_work(
    imports: &Builder,
    semantic: &DecoderGraph,
    start: usize,
    sources: &BTreeMap<String, SourceTensor>,
) -> Result<WorkBudget, Error> {
    let tokens = imports
        .nodes
        .iter()
        .find(|node| matches!(&node.code, Code::Input(key) if key == "input.tokens"));
    let mut work = WorkBudget::default();
    for op in semantic.operations.get(start..).ok_or(Error::Binding)? {
        let pending = match op.operator {
            Op::TokenLookup
                if imports.public_weights && imports.lookup == LookupPolicy::PerToken =>
            {
                let Some(tokens) = tokens else {
                    continue;
                };
                let Some(source) = op
                    .attributes
                    .get("weight")
                    .and_then(|value| value.as_str())
                    .and_then(|key| sources.get(key))
                else {
                    continue;
                };
                let [vocabulary, _] = source.shape.as_slice() else {
                    continue;
                };
                CompactTokenLookup::pending_work(*vocabulary, tokens.lanes, tokens.bounds)?
            }
            Op::Linear | Op::OutputHead if imports.public_weights => {
                let declared = ViewShape::semantic(&op.output_shape)?.lanes()?;
                let lanes = if op
                    .attributes
                    .get("bias")
                    .is_some_and(|value| !value.is_null())
                {
                    // Bias validation follows the raw factory, even for malformed
                    // final shapes. Quote the actual original matrix geometry
                    // whenever its immutable source/input declaration is known.
                    pending_biased_projection_lanes(imports, semantic, op, sources)?
                        .unwrap_or(declared)
                } else {
                    declared
                };
                contraction_rescale_work(lanes)?
            }
            _ => continue,
        };
        work = work.checked_add(pending)?;
    }
    Ok(work)
}

fn native_construction_quote(
    construction: tensor::Construction,
) -> Result<tensor::ConstructionQuote, Error> {
    construction.quote().map_err(|error| match error {
        tensor::ConstructionQuoteError::Geometry(error) => error,
        tensor::ConstructionQuoteError::UncheckedRotaryCoefficientCapacity => Error::Bounds,
    })
}

/// Sequential native factories retain earlier Arc owners while constructing the
/// next one. The native owner defines every constructor/finish/publication arm;
/// only compiler quote-control and resource-inspection slots are added here.
/// This is a logical buffer envelope, not execution scratch or whole admission.
#[derive(Default)]
struct NativeConstructionPeak {
    retained: usize,
    peak: usize,
}

impl NativeConstructionPeak {
    fn include(&mut self, construction: tensor::Construction) -> Result<(), Error> {
        let quote = native_construction_quote(construction)?;
        self.peak = self
            .peak
            .max(add_bytes(self.retained, quote.new_peak_bytes)?);
        self.retained = add_bytes(self.retained, quote.immutable_template_bytes)?;
        Ok(())
    }

    fn bytes(&self) -> Result<usize, Error> {
        let mut bytes = add_bytes(self.peak, std::mem::size_of::<Self>())?;
        if self.retained != 0 {
            // Native quote already covers Builder/Program return and Arc
            // publication. Do not reconstruct or duplicate that inventory here.
            for slot in [
                std::mem::size_of::<tensor::Construction>(),
                std::mem::size_of::<tensor::ConstructionQuote>(),
                std::mem::size_of::<
                    Result<tensor::ConstructionQuote, tensor::ConstructionQuoteError>,
                >(),
                std::mem::size_of::<tensor::Resources>(),
                std::mem::size_of::<Result<tensor::Resources, Error>>(),
            ] {
                bytes = add_bytes(bytes, slot)?;
            }
        }
        Ok(bytes)
    }
}

struct EqualityProgram {
    bounds: Interval,
    program: Arc<tensor::Program>,
    owns: bool,
}

/// Only the original public PerToken kernel owners. No selector lanes, token
/// domain proof, source clone, rescale map or independently prepared substitute.
struct CompactTokenLookup {
    vocabulary: usize,
    full: EqualityProgram,
    tail: Option<EqualityProgram>,
    selector_bounds: Interval,
    work: WorkBudget, // one token; every invocation is charged by the graph
}

pub(in crate::replicated_decoder) struct TokenLookupCensus<'s, 'a> {
    pub descriptor: usize,
    pub first: usize,
    pub geometry: [usize; 3],
    pub source_owner: usize,
    pub source: &'a SourceTensor,
    pub source_binding: [u8; 32],
    pub program: &'s Arc<tensor::Program>,
    pub tail: Option<&'s Arc<tensor::Program>>,
}

#[cfg(test)]
thread_local! {
    static LOOKUP_NATIVE_ATTEMPTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static REQUEST_LOOKUP_FAILURE:std::cell::Cell<u8>=const{std::cell::Cell::new(0)};
}

#[cfg(test)]
pub(super) fn test_lookup_attempts() -> usize {
    LOOKUP_NATIVE_ATTEMPTS.with(|n| n.get())
}
#[cfg(test)]
pub(super) fn test_lookup_failure(stage: u8) {
    REQUEST_LOOKUP_FAILURE.with(|s| s.set(stage));
}

impl CompactTokenLookup {
    fn pending_work(
        vocabulary: usize,
        tokens: usize,
        input: Interval,
    ) -> Result<WorkBudget, Error> {
        if vocabulary == 0 || vocabulary > MAX_VALUES || tokens == 0 {
            return Err(Error::Shape);
        }
        let offsets = Interval::new(
            input
                .lower()
                .checked_sub(i64::try_from(vocabulary - 1).map_err(|_| Error::Bounds)?)
                .ok_or(Error::Bounds)?,
            input.upper(),
        )?;
        let full = vocabulary.min(128);
        let mut work = tensor::Program::equal_zero_work(full, offsets)?
            .checked_mul((vocabulary / full) as u64)?;
        if vocabulary % full != 0 {
            work = work.checked_add(tensor::Program::equal_zero_work(
                vocabulary % full,
                offsets,
            )?)?;
        }
        work.checked_mul(tokens as u64)
    }

    fn quote(
        vocabulary: usize,
        new_full: bool,
        new_tail: bool,
        baseline: usize,
    ) -> Result<usize, Error> {
        if vocabulary == 0 || vocabulary > MAX_VALUES {
            return Err(Error::Shape);
        }
        let lanes = vocabulary.min(128);
        let remainder = vocabulary % lanes;
        let mut native = NativeConstructionPeak::default();
        if new_full {
            native.include(tensor::Construction::EqualZero { lanes })?;
        }
        if new_tail && remainder != 0 {
            native.include(tensor::Construction::EqualZero { lanes: remainder })?;
        }
        let mut peak = add_bytes(baseline, native.bytes()?)?;
        for bytes in [
            2 * std::mem::size_of::<Result<Self, Error>>(),
            2 * std::mem::size_of::<Result<EqualityProgram, Error>>(),
        ] {
            peak = add_bytes(peak, bytes)?;
        }
        Ok(peak)
    }

    fn programs(&self) -> impl Iterator<Item = &EqualityProgram> {
        std::iter::once(&self.full).chain(self.tail.iter())
    }

    fn baseline(&self) -> Result<usize, Error> {
        self.programs()
            .filter(|program| program.owns)
            .try_fold(0, |bytes, program| {
                add_bytes(bytes, program.program.resources()?.immutable_template_bytes)
            })
    }

    fn lower(
        graph: &CompactGraph<'_, '_>,
        input: &PublishedView,
        vocabulary: usize,
        width: usize,
        _source: &SourceTensor,
    ) -> Result<Self, Error> {
        if vocabulary == 0 || vocabulary > MAX_VALUES || width == 0 {
            return Err(Error::Shape);
        }
        // Exact original graph::TokenLookup difference interval. This is the
        // TokenOffsets declaration, NOT a secret range check/domain certificate.
        let offsets = Interval::new(
            input
                .bounds
                .lower()
                .checked_sub(i64::try_from(vocabulary - 1).map_err(|_| Error::Bounds)?)
                .ok_or(Error::Bounds)?,
            input.bounds.upper(),
        )?;
        let lanes = vocabulary.min(128); // original Builder::tensor_loop width
        let remainder = vocabulary % lanes;
        // The shared request controller reserves query/ticket slots BEFORE the
        // original native work query. Isolated qualifiers retain their original
        // local checks and acquire no request authority through this branch.
        let admission = graph.request_admission;
        let query_bytes = if admission.is_some() {
            let r = tensor::WorkPreflightWorkspace::resources();
            [
                r.workspace_bytes,
                r.factory_bytes,
                r.ticket_bytes,
                r.admitted_schedule_bytes,
                r.admitted_part_bytes,
                r.summary_bytes,
                r.ledger_bytes,
                r.ticket_result_bytes,
                r.build_result_bytes,
            ]
            .into_iter()
            .try_fold(0usize, |n, v| {
                add_bytes(n, v.checked_mul(2).ok_or(Error::Bounds)?)
            })?
        } else {
            0
        };
        if let Some(controller) = admission {
            controller.admit_phase(
                add_bytes(query_bytes, graph.contraction_control_bytes.get())?,
                0,
            )?;
        }
        graph.admit_pending_work(Self::pending_work(vocabulary, input.lanes, input.bounds)?)?;
        let prior = |lanes| {
            graph
                .ops
                .iter()
                .filter_map(|op| match &op.kind {
                    PublishedKind::TokenLookup { plan, .. } => Some(plan),
                    _ => None,
                })
                .flat_map(|plan| plan.programs())
                .find(|program| program.program.lanes() == lanes && program.bounds == offsets)
        };
        // BOTH exact native full/tail paths, including EqualZero's owning shift
        // iterator, are admitted before the first call. Prior original Arcs are
        // already in the graph baseline and are never replaced or reallocated.
        let peak = Self::quote(
            vocabulary,
            prior(lanes).is_none(),
            remainder != 0 && prior(remainder).is_none(),
            graph.baseline()?,
        )?;
        graph.admit_local(peak)?;
        let (mut full_part, mut tail_part) = if let Some(controller) = admission {
            // Separate tickets: full-tail's native convenience API has ONE tail
            // invocation, whereas this original graph has one tail PER token.
            let mut workspace = tensor::WorkPreflightWorkspace::empty();
            let full_invocations = u64::try_from(vocabulary / lanes)
                .map_err(|_| Error::Bounds)?
                .checked_mul(u64::try_from(input.lanes).map_err(|_| Error::Bounds)?)
                .ok_or(Error::Bounds)?;
            let full = tensor::Construction::EqualZero { lanes }
                .bind(tensor::FactoryParameters::EqualZero { input: offsets })
                .and_then(|factory| factory.preflight(full_invocations, &mut workspace))
                .map_err(|_| Error::Bounds)?;
            let tail = if remainder != 0 {
                Some(
                    tensor::Construction::EqualZero { lanes: remainder }
                        .bind(tensor::FactoryParameters::EqualZero { input: offsets })
                        .and_then(|factory| factory.preflight(input.lanes as u64, &mut workspace))
                        .map_err(|_| Error::Bounds)?,
                )
            } else {
                None
            };
            // BOTH persistent debits finish before the first factory. A later
            // tail refusal/factory failure does not refund the full debit.
            let (full, _) = controller.admit_work(full)?.into_parts();
            let tail = tail
                .map(|ticket| controller.admit_work(ticket).map(|v| v.into_parts().0))
                .transpose()?;
            (Some(full), tail)
        } else {
            (None, None)
        };
        // Work workspace has retired. Admit the actual native full/tail overlap,
        // not the sum of query and construction maxima.
        if let Some(controller) = admission {
            let controls = add_bytes(
                2 * std::mem::size_of::<Option<tensor::AdmittedFactory<'_>>>(),
                2 * tensor::WorkPreflightWorkspace::resources().build_result_bytes,
            )?;
            controller.admit_phase(
                add_bytes(
                    Self::quote(
                        vocabulary,
                        prior(lanes).is_none(),
                        remainder != 0 && prior(remainder).is_none(),
                        0,
                    )?,
                    add_bytes(controls, graph.contraction_control_bytes.get())?,
                )?,
                0,
            )?;
        }
        let make =
            |lanes, part: Option<tensor::AdmittedFactory<'_>>| -> Result<EqualityProgram, Error> {
                if let Some(previous) = prior(lanes) {
                    return Ok(EqualityProgram {
                        bounds: offsets,
                        program: previous.program.clone(),
                        owns: false,
                    });
                }
                let quote = native_construction_quote(tensor::Construction::EqualZero { lanes })?;
                #[cfg(test)]
                LOOKUP_NATIVE_ATTEMPTS.with(|count| count.set(count.get() + 1));
                let program = if admission.is_some() {
                    let (program, certificate) = part.ok_or(Error::Binding)?.build()?.into_parts();
                    if certificate != tensor::ProducedCertificate::None {
                        return Err(Error::Binding);
                    }
                    program
                } else {
                    tensor::Program::equal_zero(lanes, offsets)?
                };
                #[cfg(test)]
                if admission.is_some()
                    && REQUEST_LOOKUP_FAILURE.with(|stage| {
                        if stage.get() == 1 {
                            stage.set(0);
                            true
                        } else {
                            false
                        }
                    })
                {
                    return Err(Error::Allocation);
                }
                let resources = program.resources()?;
                if resources.immutable_template_bytes != quote.immutable_template_bytes
                    || resources.shared_admitted_bytes > puma::MAX_ADMITTED_BYTES
                    || program.lanes() != lanes
                    || program.output_lanes() != lanes
                    || program.validity_lanes().is_some()
                {
                    return Err(Error::Binding);
                }
                // Original Builder interning only for a newly constructed native
                // program. Never replace an existing closed/prepared Arc owner.
                let existing = graph
                    .owned_native_programs()
                    .find(|old| old.digest() == program.digest())
                    .cloned();
                Ok(EqualityProgram {
                    bounds: offsets,
                    owns: existing.is_none(),
                    program: existing.unwrap_or_else(|| Arc::new(program)),
                })
            };
        let full = make(lanes, full_part.take())?;
        let tail = (remainder != 0)
            .then(|| make(remainder, tail_part.take()))
            .transpose()?;
        if let Some(controller) = admission {
            let retained = std::iter::once(&full)
                .chain(tail.iter())
                .filter(|p| p.owns)
                .try_fold(0usize, |n, p| {
                    add_bytes(
                        n,
                        add_bytes(
                            p.program.resources()?.immutable_template_bytes,
                            2 * std::mem::size_of::<usize>(),
                        )?,
                    )
                })?;
            controller.retain(retained)?;
        }
        let mut selector_bounds = full.program.output_bounds();
        let mut work = full
            .program
            .work_budget()
            .checked_mul((vocabulary / lanes) as u64)?;
        if let Some(tail) = &tail {
            let bounds = tail.program.output_bounds();
            selector_bounds = Interval::new(
                selector_bounds.lower().min(bounds.lower()),
                selector_bounds.upper().max(bounds.upper()),
            )?;
            work = work.checked_add(tail.program.work_budget())?;
        }
        // Matrix admission and original table-bound restoration belong to the
        // shared ContractionSink traversal, after each original selector view.
        if work.checked_mul(input.lanes as u64)?
            != Self::pending_work(vocabulary, input.lanes, input.bounds)?
        {
            return Err(Error::Binding);
        }
        Ok(Self {
            vocabulary,
            full,
            tail,
            selector_bounds,
            work,
        })
    }
}

type ContractionResult<'a> = (
    PublishedKind<'a>,
    Option<MapOwner>,
    Mapping,
    Interval,
    usize,
    WorkBudget,
);

#[derive(Clone, Copy, PartialEq, Eq)]
enum ContractionStage {
    Input,
    Selectors(usize),
    LookupRow(usize),
    Output,
}

struct ContractionView {
    view: PublishedView,
    bounds: std::cell::Cell<Interval>,
    restored: std::cell::Cell<bool>,
    stage: ContractionStage,
    valid: bool,
}

impl ContractionView {
    fn new(view: PublishedView, stage: ContractionStage) -> Self {
        Self {
            bounds: std::cell::Cell::new(view.bounds),
            restored: std::cell::Cell::new(false),
            view,
            stage,
            valid: true,
        }
    }
}

/// One bounded mapping across all original rows, never a selector/lane table.
#[derive(Default)]
struct ContractionWords {
    output: Option<ContractionView>,
    rows: usize,
}

struct ContractionPublication<'a> {
    kind: PublishedKind<'a>,
    prepared: Option<MapOwner>,
    nodes: usize,
    work: WorkBudget,
}

/// Creation custody, not a new label-based publication or compact frontier.
/// Ordinary finalized Program retains every original source/native allocation.
pub(in crate::replicated_decoder) struct CreationPublication {
    pub owner: request_origin::source_bias::Creation,
    pub census: DescriptorCensus,
    pub original_work: WorkBudget,
}

/// Rank-four original native rows. Mapping is (first + 2*row + 1, lane),
/// independent of physical Worker slots and of the source projection's tiles.
pub(in crate::replicated_decoder) struct RotaryCreationPublication {
    pub owner: request_origin::rotary_creation::Creation,
    pub work: WorkBudget,
    pub shape: [usize; 4],
}
pub(in crate::replicated_decoder) fn publish_rotary_creation(
    owner: request_origin::rotary_creation::Creation,
    program: &Program,
    registry: &super::super::source_registry::SourceRegistry<'_>,
    retained: usize,
    cap: usize,
) -> Result<RotaryCreationPublication, Error> {
    if add_bytes(retained, std::mem::size_of::<RotaryCreationPublication>())?
        > cap.min(MAX_PROGRAM_BYTES)
    {
        return Err(Error::Bounds);
    }
    owner.check(program, registry)?;
    let work = owner
        .rows
        .iter()
        .try_fold(WorkBudget::default(), |w, row| {
            w.checked_add(row.program.work_budget())
        })?;
    let emitted = (owner.first..owner.end).try_fold(WorkBudget::default(), |w, id| {
        w.checked_add(schedule_finalization::BorrowedNode::from(program.recipe(id)).work_budget()?)
    })?;
    if work != emitted {
        return Err(Error::Binding);
    }
    let shape = owner.shape();
    Ok(RotaryCreationPublication { owner, work, shape })
}

pub(in crate::replicated_decoder) fn creation_publication_bytes() -> Result<usize, Error> {
    [
        std::mem::size_of::<CreationPublication>(),
        std::mem::size_of::<ContractionPublication<'_>>(),
        std::mem::size_of::<PublishedOp<'_>>(),
        2 * std::mem::size_of::<PublishedView>(),
        2 * std::mem::size_of::<usize>(), // Local mapping Arc control, not authority.
        projection_bias_control_bytes()?,
    ]
    .into_iter()
    .try_fold(0, add_bytes)
}

/// Narrow freeze of ONE already created original projection. No native factory,
/// new semantic cursor, source allocation, or caller View can reach this entry.
pub(in crate::replicated_decoder) fn publish_creation(
    owner: request_origin::source_bias::Creation,
    program: &Program,
    registry: &super::super::source_registry::SourceRegistry<'_>,
    retained: usize,
    limit: usize,
) -> Result<CreationPublication, Error> {
    if add_bytes(retained, creation_publication_bytes()?)? > limit.min(MAX_PROGRAM_BYTES) {
        return Err(Error::Bounds);
    }
    owner.check(program, registry)?;
    let d = owner.description();
    let (_, _, weight, weight_norms) = registry.records().nth(d.weight).ok_or(Error::Binding)?;
    let (_, _, bias, bias_norms) = registry.records().nth(d.bias).ok_or(Error::Binding)?;
    let summary = projection_bias_summary(
        d.bias_start(),
        d.lanes(),
        d.floor_bounds,
        Interval::new(bias.lower, bias.upper)?,
    )?;
    if summary.bounds != Some(d.output_bounds) || summary.work != WorkBudget::default() {
        return Err(Error::Binding);
    }
    let publication = ContractionPublication {
        kind: PublishedKind::Projection {
            weight: d.weight,
            source: weight,
            norms: weight_norms,
            geometry: d.geometry,
            raw_bounds: d.raw_bounds,
            // Addition clears general row-energy authority; original graph's
            // attention-specific source-bound proof propagation remains there.
            row_energy: None,
            input_energy: None,
            bias: Some(ProjectionBias {
                owner: d.bias,
                source: bias,
                norms: bias_norms,
                floor_bounds: d.floor_bounds,
                coefficients: affine(1, 1, 0),
            }),
        },
        prepared: None, // SAME full/tail Arcs stay in creation owner + Program.
        nodes: d.end - d.first,
        // Transfer original accounting, not another debit or recomputed recipe.
        work: (d.first..d.end).try_fold(WorkBudget::default(), |work, id| {
            work.checked_add(program.recipe(id).work_budget()?)
        })?,
    };
    if publication.nodes != 3 + ProjectionBias::nodes(d.lanes())? {
        return Err(Error::Binding);
    }
    let origin = Arc::new(());
    let view = |mapping, shape: [usize; 3], bounds| -> Result<PublishedView, Error> {
        Ok(PublishedView {
            origin: Arc::clone(&origin),
            slot: None,
            mapping,
            start: 0,
            lanes: shape.iter().product(),
            shape,
            rank: 3,
            logical: ViewShape::new(&shape)?,
            order: LaneOrder::Linear,
            import_slot: None,
            bounds,
        })
    };
    let [rows, inner, columns] = d.geometry;
    let original_work = publication.work;
    let published = PublishedOp {
        first: d.first,
        input: view(Mapping::Dense(d.first), [1, rows, inner], d.input_bounds)?,
        output: view(
            Mapping::Products(d.bias_start()),
            [1, rows, columns],
            d.output_bounds,
        )?,
        kind: publication.kind,
        prepared: publication.prepared,
    };
    let census = published.descriptor_census(d.end);
    Ok(CreationPublication {
        owner,
        census,
        original_work,
    })
}

/// Binary traversal needs only uniform bounds and lane counts here. The exact
/// original gather coordinates remain in ProjectionBias and its frozen recipe.
#[derive(Clone, Copy)]
struct BiasOperand {
    lanes: usize,
    bounds: Interval,
}
struct BiasSink(PublicationSink);

fn projection_bias_summary(
    first: usize,
    lanes: usize,
    floor: Interval,
    bias: Interval,
) -> Result<Summary, Error> {
    binary(
        &mut BiasSink(PublicationSink { first, next: first }),
        &BiasOperand {
            lanes,
            bounds: floor,
        },
        &BiasOperand {
            lanes,
            bounds: bias,
        },
        false,
    )
}
impl BinarySink for BiasSink {
    type View = BiasOperand;
    type Owner = Value;
    type Output = Summary;
    fn len(view: &BiasOperand) -> usize {
        view.lanes
    }
    fn slice(view: &BiasOperand, start: usize, end: usize) -> BiasOperand {
        BiasOperand {
            lanes: end - start,
            bounds: view.bounds,
        }
    }
    fn empty(&self) -> Summary {
        self.0.empty()
    }
    fn gather(&mut self, view: &BiasOperand) -> Result<Value, Error> {
        let value = Value {
            id: self.0.next,
            lanes: view.lanes,
            bounds: view.bounds,
            work: WorkBudget::default(),
        };
        self.0.next = self.0.next.checked_add(1).ok_or(Error::Bounds)?;
        Ok(value)
    }
    fn combine(&mut self, left: Value, right: Value, product: bool) -> Result<Value, Error> {
        self.0.combine(left, right, product)
    }
    fn append(&mut self, output: &mut Summary, owner: Value, lanes: usize) -> Result<(), Error> {
        self.0.append(output, owner, lanes)
    }
}

fn projection_bias_control_bytes() -> Result<usize, Error> {
    let mut bytes = 0;
    for slot in [
        std::mem::size_of::<BiasSink>(),
        4 * std::mem::size_of::<BiasOperand>(),
        2 * std::mem::size_of::<Value>(),
        2 * std::mem::size_of::<Result<Value, Error>>(),
        2 * std::mem::size_of::<Summary>(),
        2 * std::mem::size_of::<Result<Summary, Error>>(),
        2 * std::mem::size_of::<ProjectionBias<'_>>(),
    ] {
        bytes = add_bytes(bytes, slot)?;
    }
    Ok(bytes)
}

#[cfg(test)]
thread_local! { static BIAS_RECIPE_ATTEMPTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

struct CompactContractionSink<'s, 'g, 'a> {
    graph: &'s CompactGraph<'g, 'a>,
    input: &'s PublishedView,
    declared: ViewShape,
    lookup_source: std::cell::Cell<Option<&'a SourceTensor>>,
    lookup: Option<CompactTokenLookup>,
    projection: Option<ContractionPublication<'a>>,
}

impl<'a> CompactContractionSink<'_, '_, 'a> {
    fn source(
        &self,
        source: &SourceTensor,
    ) -> Result<(usize, &'a SourceTensor, Option<&'a BlockNorms>), Error> {
        self.graph
            .sources
            .iter()
            .filter(|(_, source)| source.public_q18.is_none())
            .enumerate()
            .find(|(_, (_, original))| std::ptr::eq(*original, source))
            .map(|(owner, (key, original))| (owner, original, self.graph.norms.get(key)))
            .ok_or(Error::Binding)
    }

    fn output(
        &self,
        mapping: Mapping,
        logical: ViewShape,
        bounds: Interval,
        stage: ContractionStage,
    ) -> Result<ContractionView, Error> {
        Ok(ContractionView::new(
            PublishedView {
                origin: self.graph.origin.clone(),
                slot: Some(self.graph.ops.len()),
                mapping,
                start: 0,
                lanes: logical.lanes()?,
                shape: logical.envelope()?,
                rank: logical.rank,
                logical,
                order: LaneOrder::Linear,
                import_slot: None,
                bounds,
            },
            stage,
        ))
    }
}

impl graph::driver::ContractionSink for CompactContractionSink<'_, '_, '_> {
    type View = ContractionView;
    type Source = SourceTensor;
    type Words = ContractionWords;
    type Token = (usize, usize, usize); // logical index, ORIGINAL owner, lane

    fn source_shape(source: &SourceTensor) -> &[usize] {
        &source.shape
    }
    fn public_weights(&self) -> bool {
        self.graph.imports.public_weights
    }
    fn lookup_policy(&self) -> LookupPolicy {
        self.graph.imports.lookup
    }
    fn tokens(input: &ContractionView) -> impl Iterator<Item = Self::Token> + '_ {
        (0..input.view.lanes).map(move |index| {
            let (owner, lane) = input
                .view
                .word(index)
                .expect("validated original token view");
            (index, owner, lane)
        })
    }
    fn table_bounds(&self, table: &SourceTensor) -> Interval {
        let (_, original, _) = self
            .source(table)
            .expect("driver borrows original source map");
        self.lookup_source.set(Some(original));
        Interval::new(table.lower, table.upper).expect("validated source interval")
    }
    fn trie(
        &mut self,
        _: &ContractionView,
        _: &SourceTensor,
        _: &[usize],
    ) -> Result<ContractionView, Error> {
        Err(Error::Binding)
    }
    fn tiled(
        &mut self,
        _: &ContractionView,
        _: &SourceTensor,
        _: &[usize],
    ) -> Result<ContractionView, Error> {
        Err(Error::Binding)
    }
    fn private_selectors(
        &mut self,
        _: &mut ContractionWords,
        _: Self::Token,
        _: usize,
        _: usize,
    ) -> Result<(), Error> {
        Err(Error::Binding)
    }

    fn public_selectors(
        &mut self,
        token: Self::Token,
        vocabulary: usize,
    ) -> Result<ContractionWords, Error> {
        if self.input.word(token.0)? != (token.1, token.2) {
            return Err(Error::Binding);
        }
        let source = self.lookup_source.get().ok_or(Error::Binding)?;
        let [actual, width] = source.shape.as_slice() else {
            return Err(Error::Shape);
        };
        if *actual != vocabulary {
            return Err(Error::Binding);
        }
        if self.lookup.is_none() {
            self.graph.validate_lookup_tokens(self.input, vocabulary)?;
            self.lookup = Some(CompactTokenLookup::lower(
                self.graph, self.input, vocabulary, *width, source,
            )?);
        }
        let plan = self.lookup.as_ref().ok_or(Error::Binding)?;
        let first = token
            .0
            .checked_mul(3)
            .and_then(|n| self.graph.next.checked_add(n))
            .ok_or(Error::Bounds)?;
        Ok(ContractionWords {
            output: Some(self.output(
                Mapping::Dense(first),
                ViewShape::new(&[1, vocabulary])?,
                plan.selector_bounds,
                ContractionStage::Selectors(token.0),
            )?),
            rows: 0,
        })
    }

    fn selector_view(words: ContractionWords, _: usize) -> ContractionView {
        words
            .output
            .expect("nonempty original selector publication")
    }
    fn view(words: ContractionWords, shape: &[usize]) -> ContractionView {
        let mut output = words.output.expect("preflight rejects empty token input");
        Self::set_shape(&mut output, shape);
        output
    }
    fn matrix(
        &mut self,
        input: &ContractionView,
        source: &SourceTensor,
        transposed: bool,
        rescale: bool,
    ) -> Result<ContractionView, Error> {
        let (weight, source, norms) = self.source(source)?;
        let [a, b] = source.shape.as_slice() else {
            return Err(Error::Shape);
        };
        let (rows, inner, columns) = if transposed {
            (1, *a, *b)
        } else {
            (
                input.view.lanes.checked_div(*b).ok_or(Error::Shape)?,
                *b,
                *a,
            )
        };
        if inner == 0
            || columns == 0
            || input.view.logical.row_width() != inner
            || input.view.lanes != rows.checked_mul(inner).ok_or(Error::Bounds)?
        {
            return Err(Error::Shape);
        }
        let input_energy = if transposed {
            None
        } else {
            if input.stage != ContractionStage::Input || !rescale {
                return Err(Error::Binding);
            }
            self.graph.row_norm(&input.view)?
        };
        let (raw_bounds, row_energy) = public_matrix_contract(
            [rows, inner, columns],
            transposed,
            rescale,
            || {
                Ok(InputBounds {
                    left: input.bounds.get(),
                    right: Interval::new(source.lower, source.upper)?,
                })
            },
            PublicMatrixStatistics {
                row_l1: source.row_l1,
                row_squared_norm: source.row_squared_norm,
                row_l1_squared_sum: source.row_l1_squared_sum,
            },
            || input_energy,
        )?;
        if transposed {
            let ContractionStage::Selectors(index) = input.stage else {
                return Err(Error::Binding);
            };
            if rescale || !std::ptr::eq(self.lookup_source.get().ok_or(Error::Binding)?, source) {
                return Err(Error::Binding);
            }
            let first = index
                .checked_mul(3)
                .and_then(|n| self.graph.next.checked_add(n))
                .ok_or(Error::Bounds)?;
            self.output(
                Mapping::Dense(first.checked_add(2).ok_or(Error::Bounds)?),
                ViewShape::new(&[1, columns])?,
                raw_bounds,
                ContractionStage::LookupRow(index),
            )
        } else {
            let lanes = rows.checked_mul(columns).ok_or(Error::Bounds)?;
            let with_bias = self
                .graph
                .frontier
                .operations()
                .nth(self.graph.cursor)
                .is_some_and(|(_, op)| {
                    op.attributes
                        .get("bias")
                        .is_some_and(|value| !value.is_null())
                });
            if !with_bias
                && (self.declared.lanes()? != lanes
                    || self.declared.envelope()? != [1, rows, columns])
            {
                return Err(Error::Shape);
            }
            let pending = contraction_rescale_work(lanes)?;
            self.graph.admit_pending_work(pending)?;
            let prepared = self.graph.prepare(MapKind::Rescale18, lanes, raw_bounds)?;
            let map = self.graph.map(&prepared)?;
            let bounds = map.output_bounds()?;
            let work = map.work_budget()?;
            if work != pending {
                return Err(Error::Binding);
            }
            let mut dimensions = input.view.logical.dimensions;
            dimensions[input.view.logical.rank as usize - 1] = columns;
            let logical = ViewShape::new(&dimensions[..input.view.rank as usize])?;
            let output = self.output(
                Mapping::Dense(self.graph.next.checked_add(2).ok_or(Error::Bounds)?),
                logical,
                bounds,
                ContractionStage::Output,
            )?;
            self.projection = Some(ContractionPublication {
                kind: PublishedKind::Projection {
                    weight,
                    source,
                    norms,
                    geometry: [rows, inner, columns],
                    raw_bounds,
                    row_energy,
                    input_energy,
                    bias: None,
                },
                prepared: Some(prepared),
                nodes: 3,
                work,
            });
            Ok(output)
        }
    }
    fn restore_table_bounds(&mut self, row: &ContractionView, bounds: Interval) {
        row.bounds.set(bounds);
        row.restored.set(true);
    }
    fn append_row(output: &mut ContractionWords, mut row: ContractionView) {
        row.valid &= row.stage == ContractionStage::LookupRow(output.rows) && row.restored.get();
        row.view.bounds = row.bounds.get();
        let width = row.view.lanes;
        if let Some(accumulated) = &mut output.output {
            let Mapping::LookupRows {
                first,
                width: previous_width,
            } = accumulated.view.mapping
            else {
                accumulated.valid = false;
                return;
            };
            accumulated.valid &= row.valid
                && previous_width == width
                && row.view.mapping == Mapping::Dense(first + 3 * output.rows + 2)
                && accumulated.bounds.get() == row.bounds.get();
            accumulated.view.lanes += width;
        } else {
            let Mapping::Dense(owner) = row.view.mapping else {
                row.valid = false;
                output.output = Some(row);
                return;
            };
            row.view.mapping = Mapping::LookupRows {
                first: owner - 2,
                width,
            };
            row.stage = ContractionStage::Output;
            output.output = Some(row);
        }
        output.rows += 1;
    }
    fn len(output: &ContractionView) -> usize {
        output.view.lanes
    }
    fn bias(
        &mut self,
        output: &mut ContractionView,
        source: &SourceTensor,
        indices: graph::driver::BiasIndices,
    ) -> Result<(), Error> {
        let (owner, source, norms) = self.source(source)?;
        let [width] = source.shape.as_slice() else {
            return Err(Error::Shape);
        };
        if *width == 0 || output.stage != ContractionStage::Output || output.view.lanes % width != 0
        {
            return Err(Error::Shape);
        }
        let publication = self.projection.as_mut().ok_or(Error::Binding)?;
        let prepared = publication.prepared.as_ref().ok_or(Error::Binding)?;
        let map = self.graph.map(prepared)?;
        if !Arc::ptr_eq(&output.view.origin, &self.graph.origin)
            || output.view.slot != Some(self.graph.ops.len())
            || output.view.mapping
                != Mapping::Dense(self.graph.next.checked_add(2).ok_or(Error::Bounds)?)
            || output.view.start != 0
            || output.view.order != LaneOrder::Linear
            || output.view.lanes != map.lanes()
            || output.bounds.get() != map.output_bounds()?
        {
            return Err(Error::Binding);
        }
        let PublishedKind::Projection {
            bias, row_energy, ..
        } = &mut publication.kind
        else {
            return Err(Error::Binding);
        };
        if bias.is_some() || indices.size_hint() != (output.view.lanes, Some(output.view.lanes)) {
            return Err(Error::Binding);
        }
        // BODY::bias and original Builder::binary both use additive Affine and
        // zero work. No coefficient vector, repeated rows or layout heap exists.
        self.graph.admit_pending_work(publication.work)?;
        let native = match publication.prepared.as_ref().ok_or(Error::Binding)?.storage {
            MapStorage::Owned(ref map) => map.template_bytes()?,
            MapStorage::Previous(_) => 0,
        };
        self.graph
            .admit_local(add_bytes(self.graph.baseline()?, native)?)?;
        for (index, lane) in indices.enumerate() {
            if lane != index % width {
                return Err(Error::Binding);
            }
        }
        let floor_bounds = output.bounds.get();
        let first = self
            .graph
            .next
            .checked_add(publication.nodes)
            .ok_or(Error::Bounds)?;
        #[cfg(test)]
        BIAS_RECIPE_ATTEMPTS.with(|count| count.set(count.get() + 1));
        let summary = projection_bias_summary(
            first,
            output.view.lanes,
            floor_bounds,
            Interval::new(source.lower, source.upper)?,
        )?;
        if summary.work != WorkBudget::default() || summary.lanes != output.view.lanes {
            return Err(Error::Binding);
        }
        publication.nodes =
            add_bytes(publication.nodes, ProjectionBias::nodes(output.view.lanes)?)?;
        *bias = Some(ProjectionBias {
            owner,
            source,
            norms,
            floor_bounds,
            coefficients: affine(1, 1, 0),
        });
        // The original addition does NOT transfer the raw projection's general
        // complete-row ledger. Attention's separate source-block ledger follows.
        *row_energy = None;
        output.view.mapping = Mapping::Products(first);
        output.view.bounds = summary.bounds.ok_or(Error::Shape)?;
        output.bounds.set(output.view.bounds);
        Ok(())
    }
    fn set_shape(output: &mut ContractionView, shape: &[usize]) {
        if let Ok(logical) = ViewShape::new(shape) {
            output.valid &= logical.lanes().ok() == Some(output.view.lanes);
            output.view.logical = logical;
            output.view.rank = logical.rank;
            output.view.shape = logical.envelope().expect("checked rank envelope");
        } else {
            output.valid = false;
        }
    }
    fn projection_energies(
        &mut self,
        input: &ContractionView,
        output: &ContractionView,
        source: &SourceTensor,
        bias: Option<&SourceTensor>,
    ) -> Result<(), Error> {
        let (_, source, norms) = self.source(source)?;
        if let Some(norms) = norms {
            let (rows, columns, width) = norms.geometry();
            if source.shape != [rows, columns]
                || input.view.logical.row_width() != columns
                || output.view.logical.row_width() != rows
                || output.view.lanes / rows != input.view.lanes / columns
            {
                return Err(Error::Binding);
            }
            if let Some(energy) = self.graph.row_norm(&input.view)? {
                for start in (0..rows).step_by(width) {
                    let energy = norm_bounds::linear_energy(
                        energy,
                        norms
                            .block(start, width.min(rows - start))
                            .ok_or(Error::Binding)?,
                        18,
                    )?;
                    if let Some(bias) = bias {
                        if bias.shape != [rows] {
                            return Err(Error::Shape);
                        }
                        let (_, _, bias_norms) = self.source(bias)?;
                        let certificate = bias_norms
                            .and_then(|norms| norms.block(start, width.min(rows - start)))
                            .ok_or(Error::Binding)?;
                        norm_bounds::sum_energy(energy, certificate.frobenius_squared)?;
                    }
                }
            }
        }
        Ok(())
    }
}

struct ContractionControls<'a>(&'a std::cell::Cell<usize>);
impl Drop for ContractionControls<'_> {
    fn drop(&mut self) {
        self.0.set(0);
    }
}

impl<'g, 'a> CompactGraph<'g, 'a> {
    fn lower_contraction(
        &self,
        op: &pllm_models::ModelOperation,
        input: &PublishedView,
        logical: ViewShape,
    ) -> Result<ContractionResult<'a>, Error> {
        let original = ContractionView::new(input.clone(), ContractionStage::Input);
        let control =
            graph::driver::contraction_control::<CompactContractionSink<'_, '_, '_>>(&original)?;
        let mut controls = add_bytes(
            if op.operator == Op::TokenLookup {
                control.lookup_bytes
            } else {
                control.projection_bytes
            },
            add_bytes(
                std::mem::size_of::<CompactContractionSink<'_, '_, '_>>(),
                2 * std::mem::size_of::<Result<ContractionResult<'_>, Error>>(),
            )?,
        )?;
        if op.operator != Op::TokenLookup
            && op
                .attributes
                .get("bias")
                .is_some_and(|value| !value.is_null())
        {
            controls = add_bytes(controls, projection_bias_control_bytes()?)?;
        }
        self.admit_local(add_bytes(self.baseline()?, controls)?)?;
        if self.contraction_control_bytes.get() != 0 {
            return Err(Error::Binding);
        }
        self.contraction_control_bytes.set(controls);
        let _controls = ContractionControls(&self.contraction_control_bytes);
        let mut sink = CompactContractionSink {
            graph: self,
            input,
            declared: logical,
            lookup_source: std::cell::Cell::new(None),
            lookup: None,
            projection: None,
        };
        let output = if op.operator == Op::TokenLookup {
            let source = op
                .attributes
                .get("weight")
                .and_then(|value| value.as_str())
                .and_then(|key| self.sources.get(key))
                .ok_or(Error::Binding)?;
            let vocabulary = *source.shape.first().ok_or(Error::Shape)?;
            graph::driver::token_lookup(
                &mut sink,
                op,
                logical.dimensions(),
                &original,
                self.sources,
                vocabulary,
            )?
        } else {
            graph::driver::projection(
                &mut sink,
                op,
                logical.dimensions(),
                Some(&original),
                self.sources,
            )?
        };
        if !output.valid || output.view.logical != logical {
            return Err(Error::Shape);
        }
        let publication = if op.operator == Op::TokenLookup {
            let source = sink.lookup_source.get().ok_or(Error::Binding)?;
            let (weight, source, _) = sink.source(source)?;
            let plan = sink.lookup.take().ok_or(Error::Binding)?;
            let work = plan.work.checked_mul(input.lanes as u64)?;
            ContractionPublication {
                kind: PublishedKind::TokenLookup {
                    weight,
                    source,
                    plan,
                },
                prepared: None,
                nodes: input.lanes.checked_mul(3).ok_or(Error::Bounds)?,
                work,
            }
        } else {
            sink.projection.take().ok_or(Error::Binding)?
        };
        Ok((
            publication.kind,
            publication.prepared,
            output.view.mapping,
            output.bounds.get(),
            publication.nodes,
            publication.work,
        ))
    }

    fn admit_pending_work(&self, pending: WorkBudget) -> Result<(), Error> {
        let work = self.work.checked_add(pending)?;
        let outside = self
            .prebound_inputs
            .as_ref()
            .map_or(WorkBudget::default(), |owner| owner.work);
        outside.checked_add(work)?.limits()?;
        Ok(())
    }
    /// Match the ORIGINAL PerToken caller's two token producers before retaining
    /// graph.rs's table-element interval: compiler input.tokens declarations or
    /// the protected original argmax output, through unchanged view aliases.
    /// This is local recipe lineage, not a runtime range check, Trie domain
    /// ticket, source authentication proof or externally supplied certificate.
    fn validate_lookup_tokens<'s>(
        &'s self,
        mut input: &'s PublishedView,
        vocabulary: usize,
    ) -> Result<(), Error> {
        loop {
            let Some(slot) = input.slot else {
                let bound = self
                    .bound_imports
                    .get(input.import_slot.ok_or(Error::Binding)?)
                    .ok_or(Error::Binding)?;
                let Mapping::Dense(owner) = input.mapping else {
                    return Err(Error::Binding);
                };
                return if bound.semantic == "input.tokens"
                    && matches!(&self.imports.nodes[owner].code, Code::Input(key) if key == "input.tokens")
                    && input.bounds
                        == Interval::new(
                            0,
                            i64::try_from(vocabulary).map_err(|_| Error::Bounds)? - 1,
                        )?
                {
                    Ok(())
                } else {
                    Err(Error::Binding)
                };
            };
            let op = self.ops.get(slot).ok_or(Error::Binding)?;
            match &op.kind {
                PublishedKind::Reshape | PublishedKind::TokenFeedback => input = &op.input,
                PublishedKind::Argmax(plan) if plan.geometry.lanes == vocabulary => return Ok(()),
                _ => return Err(Error::Binding),
            }
        }
    }

    pub(in crate::replicated_decoder) fn import_binding_storage_bound(
        semantic: &DecoderGraph,
    ) -> Result<usize, Error> {
        let count = semantic
            .operations
            .iter()
            .try_fold(0usize, |n, op| add_bytes(n, op.inputs.len()))?;
        array_bytes::<BoundImport<'g>>(count)
    }
    pub(in crate::replicated_decoder) fn container_census(&self) -> Result<ContainerCensus, Error> {
        Ok(ContainerCensus {
            imported_descriptors: self.imports.nodes.len(),
            imported_node_capacity: self.imports.nodes.capacity(),
            imported_node_bytes: array_bytes::<Node>(self.imports.nodes.capacity())?,
            imported_declared_bytes: self.imports.allocated,
            stored_operations: self.ops.len(),
            descriptor_capacity: self.ops.capacity(),
            descriptor_bytes: array_bytes::<PublishedOp<'a>>(self.ops.capacity())?,
            inline_bytes: std::mem::size_of::<Self>(),
            origin_arc_bytes: 2 * std::mem::size_of::<usize>(),
            frontier_storage_bound: self.frontier_bytes,
            expanded_virtual_nodes: self.next,
            partial_work: self.work,
            import_binding_capacity: self.bound_imports.capacity(),
            import_binding_bytes: array_bytes::<BoundImport<'g>>(self.bound_imports.capacity())?,
        })
    }

    pub(in crate::replicated_decoder) fn descriptor_census(
        &self,
    ) -> impl Iterator<Item = DescriptorCensus> + '_ {
        self.ops.iter().enumerate().map(|(slot, op)| {
            op.descriptor_census(self.ops.get(slot + 1).map_or(self.next, |next| next.first))
        })
    }

    /// Original borrowed source and full/tail Arc owners, not digest authority.
    /// Frozen consumers need an equivalent read-only accessor before emitting
    /// this family; metadata retention itself moves these unchanged recipes.
    pub(in crate::replicated_decoder) fn token_lookup_census(
        &self,
    ) -> impl Iterator<Item = TokenLookupCensus<'_, 'a>> {
        self.ops.iter().enumerate().filter_map(|(descriptor, op)| {
            let PublishedKind::TokenLookup {
                weight,
                source,
                plan,
            } = &op.kind
            else {
                return None;
            };
            Some(TokenLookupCensus {
                descriptor,
                first: op.first,
                geometry: [op.input.lanes, plan.vocabulary, op.output.shape[2]],
                source_owner: *weight,
                source,
                source_binding: self.binding,
                program: &plan.full.program,
                tail: plan.tail.as_ref().map(|tail| &tail.program),
            })
        })
    }

    /// Actual immutable owners only. Reused capabilities do not invent another
    /// allocation; native resources/capacities remain inspectable on each Arc.
    pub(in crate::replicated_decoder) fn owned_prepared_programs(
        &self,
    ) -> impl Iterator<Item = &Arc<tensor::Program>> {
        self.owned_native_programs()
    }

    pub(in crate::replicated_decoder) fn owned_native_programs(
        &self,
    ) -> impl Iterator<Item = &Arc<tensor::Program>> {
        self.ops
            .iter()
            .filter_map(|op| match &op.prepared {
                Some(MapOwner {
                    storage: MapStorage::Owned(prepared),
                    ..
                }) => Some(prepared),
                _ => None,
            })
            .flat_map(|prepared| prepared.programs())
            .chain(self.ops.iter().filter_map(|op| match &op.kind {
                PublishedKind::RmsNorm {
                    program,
                    owns_program: true,
                    ..
                } => Some(program),
                _ => None,
            }))
            .chain(
                self.ops
                    .iter()
                    .filter_map(|op| match &op.kind {
                        PublishedKind::RmsNorm {
                            checked: Some(rows),
                            ..
                        } => Some(rows),
                        _ => None,
                    })
                    .flat_map(|rows| rows.programs()),
            )
            .chain(
                self.ops
                    .iter()
                    .filter_map(|op| match &op.kind {
                        PublishedKind::Argmax(plan) => Some(plan),
                        _ => None,
                    })
                    .flat_map(|plan| {
                        plan.programs
                            .iter()
                            .filter(|program| program.owns)
                            .map(|program| &program.program)
                    }),
            )
            .chain(self.ops.iter().flat_map(|op| {
                let plan = match &op.kind {
                    PublishedKind::TokenLookup { plan, .. } => Some(plan),
                    _ => None,
                };
                plan.into_iter()
                    .flat_map(|plan| plan.programs())
                    .filter(|program| program.owns)
                    .map(|program| &program.program)
            }))
    }

    pub(in crate::replicated_decoder) fn native_row_census(
        &self,
    ) -> impl Iterator<Item = NativeRowCensus<'_, 'a>> {
        self.ops
            .iter()
            .enumerate()
            .flat_map(move |(descriptor, op)| {
                let native = match &op.kind {
                    PublishedKind::RmsNorm {
                        gamma,
                        program,
                        output_energy,
                        ..
                    } => Some((*gamma, program, *output_energy)),
                    _ => None,
                };
                let width = op.output.shape[2];
                (0..if native.is_some() {
                    op.output.shape[1]
                } else {
                    0
                })
                    .map(move |row| {
                        let (gamma, program, energy) = native.expect("native row iterator");
                        NativeRowCensus {
                            descriptor,
                            row,
                            owner: match &op.kind {
                                PublishedKind::RmsNorm {
                                    checked: Some(rows),
                                    ..
                                } => rows.rows_first(),
                                _ => op.first,
                            } + 2 * row
                                + 1,
                            offset: 0,
                            width,
                            energy,
                            program,
                            gamma,
                            source_binding: self.binding,
                        }
                    })
            })
    }

    pub(in crate::replicated_decoder) fn borrowed_source_records(
        &self,
    ) -> &'a BTreeMap<String, SourceTensor> {
        self.sources
    }

    pub(in crate::replicated_decoder) fn argmax_census(
        &self,
    ) -> impl Iterator<Item = Result<ArgmaxCensus, Error>> + '_ {
        self.ops.iter().filter_map(|op| match &op.kind {
            PublishedKind::Argmax(plan) => Some(plan.census()),
            _ => None,
        })
    }

    /// Replay SAME shared lowering into read-only original primitive events.
    /// This allocates nothing, consumes no runtime/PRSS, and confers no dispatch
    /// or proof-map authority. The emitter must preadmit and map actual owners.
    pub(in crate::replicated_decoder) fn visit_argmax(
        &self,
        descriptor: usize,
        visitor: &mut dyn for<'e> FnMut(ArgmaxEvent<'e>) -> Result<(), Error>,
    ) -> Result<(), Error> {
        if self.failed {
            return Err(Error::Retired);
        }
        let op = self.ops.get(descriptor).ok_or(Error::Binding)?;
        let PublishedKind::Argmax(plan) = &op.kind else {
            return Err(Error::Binding);
        };
        plan.visit(&op.input, visitor)
    }
    pub(in crate::replicated_decoder) fn borrowed_source_norms(&self) -> &'a SourceNorms {
        self.norms
    }

    pub(super) fn new(
        imports: &'a Builder,
        validated: ValidatedSourceDescriptors<'a>,
        semantic: &'g DecoderGraph,
        start: usize,
    ) -> Result<Self, Error> {
        Self::with_limit(imports, validated, semantic, start, MAX_PROGRAM_BYTES)
    }

    /// Enclosing compiler's independently inventoried OUTSIDE owners. No credit
    /// for borrowed/live owners and no claim that this integer completes census.
    pub(super) fn with_owner_baseline(
        imports: &'a Builder,
        validated: ValidatedSourceDescriptors<'a>,
        semantic: &'g DecoderGraph,
        start: usize,
        outside_owner_bytes: usize,
    ) -> Result<Self, Error> {
        Self::with_limit_and_owner(
            imports,
            validated,
            semantic,
            start,
            MAX_PROGRAM_BYTES,
            outside_owner_bytes,
        )
    }

    pub(in crate::replicated_decoder) fn outside_owner_bytes(&self) -> usize {
        self.outside_owner_bytes
    }
    /// Greatest admitted construction quote, INCLUDING enclosing outside owners.
    /// Semantic/source maps remain excluded, except RMSNorm's explicit gamma
    /// subowner. Freeze uses the separately source-normalized historical maximum.
    pub(in crate::replicated_decoder) fn construction_peak_bytes(&self) -> usize {
        self.construction_peak.get()
    }

    fn admit_total(&self, total: usize) -> Result<(), Error> {
        self.admit_total_with_source(total, 0)
    }

    fn admit_total_with_source(&self, total: usize, included_source: usize) -> Result<(), Error> {
        if total < self.outside_owner_bytes {
            return Err(Error::Binding);
        }
        let admitted = if let Some(owner) = &self.prebound_inputs {
            add_bytes(
                total,
                owner
                    .borrowed_bytes
                    .checked_sub(included_source)
                    .ok_or(Error::Binding)?,
            )?
        } else {
            total
        };
        if admitted > self.limit {
            return Err(Error::Bounds);
        }
        let without_source = total.checked_sub(included_source).ok_or(Error::Binding)?;
        if without_source < self.outside_owner_bytes {
            return Err(Error::Binding);
        }
        self.construction_peak
            .set(self.construction_peak.get().max(admitted));
        self.construction_peak_without_source.set(
            self.construction_peak_without_source
                .get()
                .max(without_source),
        );
        Ok(())
    }

    fn admit_local(&self, local: usize) -> Result<(), Error> {
        self.admit_total(add_bytes(self.outside_owner_bytes, local)?)
    }

    pub(super) fn imports_bytes(imports: &Builder) -> Result<usize, Error> {
        let mut bytes = add_bytes(imports.allocated, std::mem::size_of::<Builder>())?;
        bytes = add_bytes(bytes, array_bytes::<Node>(imports.nodes.capacity())?)?;
        // Like nodes above, inventory the actual retained backing independently
        // of conservative allocation/history charges. Spare seal-record slots
        // can grow without changing allocated; never credit history against
        // that live owner. Construction and freeze share this one census, so
        // neither phase adds the same physical Vec backing a second time.
        bytes = add_bytes(
            bytes,
            array_bytes::<Arc<super::original_import::InputRecord>>(
                imports.original_inputs.capacity(),
            )?,
        )?;
        for node in &imports.nodes {
            match &node.code {
                Code::Input(key) | Code::PublicInput { key, .. } => {
                    bytes = add_bytes(bytes, key.capacity())?
                }
                _ => return Err(Error::Binding),
            }
        }
        for reserved in [
            array_bytes::<(usize, u128)>(imports.row_norms.capacity())?,
            array_bytes::<(Words, u128)>(imports.row_energies.capacity())?,
            array_bytes::<(usize, u64)>(imports.row_masses.capacity())?,
            array_bytes::<(Words, u128)>(imports.attention_energies.capacity())?,
        ] {
            bytes = add_bytes(bytes, reserved)?;
        }
        for norms in imports.public_norms.values() {
            bytes = add_bytes(bytes, norms.metadata_bytes()?)?;
            // Same conservative whole-node-per-entry policy as source norms.
            bytes = add_bytes(
                bytes,
                add_bytes(array_bytes::<(usize, BlockNorms)>(16)?, 128)?,
            )?;
        }
        Ok(bytes)
    }

    fn with_limit(
        imports: &'a Builder,
        validated: ValidatedSourceDescriptors<'a>,
        semantic: &'g DecoderGraph,
        start: usize,
        limit: usize,
    ) -> Result<Self, Error> {
        Self::with_limit_and_owner(imports, validated, semantic, start, limit, 0)
    }

    fn with_limit_and_owner(
        imports: &'a Builder,
        validated: ValidatedSourceDescriptors<'a>,
        semantic: &'g DecoderGraph,
        start: usize,
        limit: usize,
        outside_owner_bytes: usize,
    ) -> Result<Self, Error> {
        Self::with_limit_and_admission(
            imports,
            validated,
            semantic,
            start,
            limit,
            outside_owner_bytes,
            None,
        )
    }

    fn with_limit_and_admission(
        imports: &'a Builder,
        validated: ValidatedSourceDescriptors<'a>,
        semantic: &'g DecoderGraph,
        start: usize,
        limit: usize,
        outside_owner_bytes: usize,
        admission: Option<&'a request_origin::compact_storage::EmissionContext<'a>>,
    ) -> Result<Self, Error> {
        let execution = validated.execution;
        let (sources, norms) = validated.public_records()?;
        // No expanded arithmetic prefix, hidden validity or omitted certificates.
        if !imports.public_weights
            || imports.checked_bound != execution.checked_bound()
            || imports.lookup != execution.lookup
            || imports.selection != execution.selection
            || start >= semantic.operations.len()
            || imports
                .nodes
                .iter()
                .any(|n| !matches!(n.code, Code::Input(_) | Code::PublicInput { .. }))
            || !imports.tensors.is_empty()
            || imports.validity.is_some()
            || !imports.row_norms.is_empty()
            || !imports.row_energies.is_empty()
            || !imports.row_masses.is_empty()
            || !imports.attention_energies.is_empty()
            || !imports.token_domains.is_empty()
        {
            return Err(Error::Binding);
        }
        let count = sources.values().filter(|s| s.public_q18.is_none()).count();
        for (owner, (key, source)) in sources
            .iter()
            .filter(|(_, s)| s.public_q18.is_none())
            .enumerate()
        {
            let node = imports.nodes.get(owner).ok_or(Error::Binding)?;
            let Code::PublicInput {
                key: actual_key,
                digest,
                row_l1,
                row_squared_norm,
                row_l1_squared_sum,
            } = &node.code
            else {
                return Err(Error::Binding);
            };
            if actual_key != key
                || *digest != source.digest
                || node.lanes != source.shape.iter().product::<usize>()
                || node.bounds != Interval::new(source.lower, source.upper)?
                || *row_l1 != source.row_l1
                || *row_squared_norm != source.row_squared_norm
                || *row_l1_squared_sum != source.row_l1_squared_sum
                || imports.public_norms.get(&owner) != norms.get(key)
            {
                return Err(Error::Binding);
            }
        }
        if imports.nodes[count..]
            .iter()
            .any(|n| !matches!(n.code, Code::Input(_)))
            || imports.public_norms.len() != norms.len()
        {
            return Err(Error::Binding);
        }
        let capacity = semantic.operations.len() - start;
        if add_bytes(imports.nodes.len(), capacity)? > MAX_NODES {
            return Err(Error::Bounds);
        }
        let frontier_bytes = graph::OrderedGraph::<PublishedView>::storage_bound(semantic)?;
        let binding_capacity = semantic
            .operations
            .iter()
            .try_fold(0usize, |n, op| add_bytes(n, op.inputs.len()))?;
        let binding_bytes = Self::import_binding_storage_bound(semantic)?;
        let base = add_bytes(Self::imports_bytes(imports)?, std::mem::size_of::<Self>())?;
        // Arc control block, complete descriptor destination and construction
        // result slots are admitted BEFORE Arc/frontier/Vec allocation.
        let storage = add_bytes(
            add_bytes(frontier_bytes, array_bytes::<PublishedOp<'a>>(capacity)?)?,
            binding_bytes,
        )?;
        let slots = add_bytes(
            2 * std::mem::size_of::<Result<Self, Error>>(),
            2 * std::mem::size_of::<usize>(),
        )?;
        let limit = limit.min(MAX_PROGRAM_BYTES);
        let initial_peak = add_bytes(
            outside_owner_bytes,
            add_bytes(add_bytes(base, storage)?, slots)?,
        )?;
        if initial_peak > limit {
            return Err(Error::Bounds);
        }
        if let Some(controller) = admission {
            controller.admit_phase(
                add_bytes(add_bytes(std::mem::size_of::<Self>(), storage)?, slots)?,
                0,
            )?;
        }
        #[cfg(test)]
        input_owner::CONTAINER_FACTORIES.with(|n| n.set(n.get() + 1));
        let mut ops = Vec::new();
        ops.try_reserve_exact(capacity)
            .map_err(|_| Error::Allocation)?;
        if ops.capacity() != capacity {
            return Err(Error::Allocation);
        }
        let mut bound_imports = Vec::new();
        bound_imports
            .try_reserve_exact(binding_capacity)
            .map_err(|_| Error::Allocation)?;
        if bound_imports.capacity() != binding_capacity {
            return Err(Error::Allocation);
        }
        let origin = Arc::new(());
        let validity = compact_validity::RequestValidity::original(imports, origin.clone());
        if let Some(controller) = admission {
            controller.retain(add_bytes(
                add_bytes(std::mem::size_of::<Self>(), storage)?,
                slots,
            )?)?;
        }
        Ok(Self {
            imports,
            sources,
            norms,
            binding: source_binding_with_norms(sources, norms),
            frontier: graph::OrderedGraph::new(semantic),
            frontier_bytes,
            ops,
            bound_imports,
            origin,
            next: imports.nodes.len(),
            cursor: start,
            work: WorkBudget::default(),
            limit,
            outside_owner_bytes,
            request_admission: admission,
            request_prelude: None,
            prebound_inputs: None,
            validity,
            contraction_control_bytes: std::cell::Cell::new(0),
            construction_peak: std::cell::Cell::new(initial_peak),
            construction_peak_without_source: std::cell::Cell::new(initial_peak),
            failed: false,
            failure: None,
        })
    }

    fn baseline(&self) -> Result<usize, Error> {
        let mut bytes = add_bytes(
            Self::imports_bytes(self.imports)?,
            std::mem::size_of::<Self>(),
        )?;
        bytes = add_bytes(bytes, self.frontier_bytes)?;
        bytes = add_bytes(bytes, self.contraction_control_bytes.get())?;
        bytes = add_bytes(bytes, array_bytes::<PublishedOp<'a>>(self.ops.capacity())?)?;
        bytes = add_bytes(
            bytes,
            array_bytes::<BoundImport<'g>>(self.bound_imports.capacity())?,
        )?;
        bytes = add_bytes(
            bytes,
            2 * std::mem::size_of::<Result<Self, Error>>() + 2 * std::mem::size_of::<usize>(),
        )?;
        for op in &self.ops {
            if let Some(MapOwner {
                storage: MapStorage::Owned(prepared),
                ..
            }) = &op.prepared
            {
                bytes = add_bytes(bytes, prepared.baseline_bytes()?)?;
            }
            if let PublishedKind::RmsNorm {
                program,
                owns_program: true,
                ..
            } = &op.kind
            {
                bytes = add_bytes(bytes, program.resources()?.immutable_template_bytes)?;
            }
            if let PublishedKind::RmsNorm {
                checked: Some(rows),
                ..
            } = &op.kind
            {
                bytes = add_bytes(bytes, rows.baseline()?)?;
            }
            if let PublishedKind::Argmax(plan) = &op.kind {
                bytes = add_bytes(bytes, plan.baseline()?)?;
            }
            if let PublishedKind::TokenLookup { plan, .. } = &op.kind {
                bytes = add_bytes(bytes, plan.baseline()?)?;
            }
        }
        // Live lowering candidates and their Result/return overlaps. No refund
        // for retired frontier views, shared template backing or borrowed imports.
        add_bytes(bytes, 4 * std::mem::size_of::<PublishedOp<'a>>())
    }

    /// Actual prebound input/plan metadata is not an opaque outside allowance.
    /// Shared request census records it alongside the ordinary graph baseline.
    fn retained_metadata_bytes(&self) -> Result<usize, Error> {
        add_bytes(
            self.baseline()?,
            self.prebound_inputs
                .as_ref()
                .map_or(0, |owner| owner.metadata_bytes),
        )
    }

    fn validate_view(&self, view: &PublishedView) -> Result<(), Error> {
        if !Arc::ptr_eq(&self.origin, &view.origin) {
            return Err(Error::Binding);
        }
        if !(1..=4).contains(&view.logical.rank)
            || view.rank != view.logical.rank
            || ViewShape::new(view.logical.dimensions())? != view.logical
            || view.logical.lanes()? != view.lanes
            || view.logical.envelope()? != view.shape
        {
            return Err(Error::Shape);
        }
        if let Some(slot) = view.slot {
            let expected = &self.ops.get(slot).ok_or(Error::Binding)?.output;
            if expected.mapping != view.mapping
                || expected.lanes != view.lanes
                || expected.bounds != view.bounds
                || expected.shape != view.shape
                || expected.rank != view.rank
                || expected.start != view.start
                || expected.logical != view.logical
                || expected.order != view.order
                || expected.import_slot != view.import_slot
            {
                return Err(Error::Binding);
            }
        } else {
            let bound = &self
                .bound_imports
                .get(view.import_slot.ok_or(Error::Binding)?)
                .ok_or(Error::Binding)?
                .view;
            if bound.mapping != view.mapping
                || bound.start != view.start
                || bound.lanes != view.lanes
                || bound.bounds != view.bounds
                || bound.logical != view.logical
                || bound.shape != view.shape
                || bound.rank != view.rank
                || bound.order != view.order
                || bound.import_slot != view.import_slot
            {
                return Err(Error::Binding);
            }
            let Mapping::Dense(owner) = view.mapping else {
                return Err(Error::Binding);
            };
            let node = self.imports.nodes.get(owner).ok_or(Error::Binding)?;
            if !matches!(node.code, Code::Input(_))
                || view.start != 0
                || view.order != LaneOrder::Linear
                || node.lanes != view.lanes
                || node.bounds != view.bounds
                || view.logical.lanes()? != node.lanes
            {
                return Err(Error::Binding);
            }
        }
        Ok(())
    }

    pub(super) fn bind_import(
        &mut self,
        semantic: &str,
        input: &View,
    ) -> Result<PublishedView, Error> {
        if self.failed {
            return Err(Error::Retired);
        }
        if self.prebound_inputs.is_some() {
            // Opt-in authority comes only from the original creation seals.
            return Err(Error::Binding);
        }
        if !self.ops.is_empty()
            || self
                .frontier
                .operations()
                .skip(self.cursor)
                .any(|(_, op)| op.id == semantic)
        {
            return Err(Error::Binding);
        }
        let semantic = self
            .frontier
            .operations()
            .skip(self.cursor)
            .flat_map(|(_, op)| op.inputs.iter())
            .find(|id| id.as_str() == semantic)
            .map(String::as_str)
            .ok_or(Error::Binding)?;
        let logical = ViewShape::new(&input.shape)?;
        let original = OriginalSlice::from_words(self.imports, &input.words)?;
        if original.offset != 0
            || original.lanes != original.node.lanes
            || !matches!(original.node.code, Code::Input(_))
        {
            return Err(Error::Binding);
        }
        let declared = self.frontier.semantic_graph();
        let declaration = if semantic == "input.tokens" {
            Some(ViewShape::semantic(&[
                declared.batch,
                declared.query_sequence,
            ])?)
        } else {
            declared
                .state_inputs
                .iter()
                .find(|state| state.id == semantic)
                .map(|state| ViewShape::semantic(&state.shape))
                .transpose()?
        };
        if let Some(expected) = declaration {
            if logical != expected
                || !matches!(&original.node.code, Code::Input(key) if key == semantic)
            {
                return Err(Error::Binding);
            }
        }
        if logical.lanes()? != original.lanes {
            return Err(Error::Shape);
        }
        if let Some(bound) = self
            .bound_imports
            .iter()
            .find(|bound| bound.semantic == semantic)
        {
            if bound.view.mapping != Mapping::Dense(original.owner) || bound.view.logical != logical
            {
                return Err(Error::Binding);
            }
            return Ok(bound.view.clone());
        }
        if self.bound_imports.len() == self.bound_imports.capacity() {
            return Err(Error::Bounds);
        }
        self.admit_local(self.baseline()?)?;
        let view = PublishedView {
            origin: self.origin.clone(),
            slot: None,
            mapping: Mapping::Dense(original.owner),
            start: 0,
            lanes: original.lanes,
            shape: logical.envelope()?,
            rank: logical.rank,
            logical,
            order: LaneOrder::Linear,
            import_slot: Some(self.bound_imports.len()),
            bounds: original.node.bounds,
        };
        self.bound_imports.push(BoundImport {
            semantic,
            view: view.clone(),
        });
        self.validate_view(&view)?;
        // Only existing semantic IDs may bind: new key backing was preadmitted.
        self.frontier.bind(semantic.into(), view.clone());
        Ok(view)
    }

    fn kernel(&self, mut slot: usize) -> Result<&PreparedTensorMap, Error> {
        loop {
            let owner = self
                .ops
                .get(slot)
                .and_then(|op| op.prepared.as_ref())
                .ok_or(Error::Binding)?;
            match &owner.storage {
                MapStorage::Owned(prepared) => return Ok(prepared),
                MapStorage::Previous(previous) if *previous < slot => slot = *previous,
                _ => return Err(Error::Binding),
            }
        }
    }

    fn prepare(&self, kind: MapKind, lanes: usize, input: Interval) -> Result<MapOwner, Error> {
        for (slot, op) in self.ops.iter().enumerate() {
            if op
                .prepared
                .as_ref()
                .is_some_and(|p| p.kind == kind && p.lanes == lanes && p.input == input)
            {
                return Ok(MapOwner {
                    kind,
                    lanes,
                    input,
                    storage: MapStorage::Previous(slot),
                });
            }
        }
        let baseline = add_bytes(self.outside_owner_bytes, self.baseline()?)?;
        let admit = |peak| self.admit_total(peak);
        let prepared = match kind {
            MapKind::Rescale18 => PreparedTensorMap::rescale(lanes, input, 18, baseline, admit)?,
            MapKind::GuardedSilu => PreparedTensorMap::guarded_silu_with_prior_work(
                lanes,
                input,
                baseline,
                // Original outside request + already published graph work;
                // the factory gate adds the actual full/tail schedule before
                // entering either native constructor. It does not debit work.
                if let Some(owner) = &self.prebound_inputs {
                    owner.work.checked_add(self.work)?
                } else {
                    self.work
                },
                admit,
            )?,
        };
        // Never replace an original capability's backing with a digest match.
        // Until partial-template interning is extracted, reject such overlaps.
        for slot in 0..self.ops.len() {
            if self.ops[slot].prepared.is_none() {
                continue;
            }
            for old in self.kernel(slot)?.programs() {
                if prepared
                    .programs()
                    .any(|new| new.digest() == old.digest() && !Arc::ptr_eq(new, old))
                {
                    return Err(Error::Binding);
                }
            }
        }
        Ok(MapOwner {
            kind,
            lanes,
            input,
            storage: MapStorage::Owned(prepared),
        })
    }

    fn map<'s>(&'s self, owner: &'s MapOwner) -> Result<&'s PreparedTensorMap, Error> {
        match &owner.storage {
            MapStorage::Owned(p) => Ok(p),
            MapStorage::Previous(slot) => self.kernel(*slot),
        }
    }

    fn prepare_rms_norm(
        &self,
        width: usize,
        bounds: Interval,
        gamma: &'a SourceTensor,
        input_energy: Option<u128>,
    ) -> Result<(Arc<tensor::Program>, u128, bool), Error> {
        let coefficients = gamma.public_q18.as_ref().ok_or(Error::Binding)?;
        // Borrowed original backing stays live through both new construction
        // and reuse. Freeze normalizes this source subowner separately.
        let gamma_bytes = add_bytes(
            std::mem::size_of::<SourceTensor>(),
            add_bytes(
                array_bytes::<usize>(gamma.shape.capacity())?,
                array_bytes::<i64>(coefficients.capacity())?,
            )?,
        )?;
        for op in &self.ops {
            if let PublishedKind::RmsNorm {
                gamma: old_gamma,
                input_energy: old_energy,
                output_energy,
                program,
                row_bounds,
                ..
            } = &op.kind
            {
                if std::ptr::eq(*old_gamma, gamma)
                    && *old_energy == input_energy
                    && *row_bounds == bounds
                    && program.lanes() == width
                {
                    let slots = add_bytes(
                        compact_norm::driver_slots()?,
                        2 * std::mem::size_of::<Result<(Arc<tensor::Program>, u128, bool), Error>>(
                        ),
                    )?;
                    self.admit_total_with_source(
                        add_bytes(
                            self.outside_owner_bytes,
                            add_bytes(add_bytes(self.baseline()?, gamma_bytes)?, slots)?,
                        )?,
                        gamma_bytes,
                    )?;
                    return Ok((program.clone(), *output_energy, false));
                }
            }
        }
        let construction = if input_energy.is_some() {
            tensor::Construction::RmsNormWithInputNorm {
                rows: 1,
                width,
                gamma_len: coefficients.len(),
            }
        } else {
            tensor::Construction::RmsNormCertified {
                rows: 1,
                width,
                gamma_len: coefficients.len(),
            }
        };
        let quote = native_construction_quote(construction)?;
        // Native quote excludes borrowed public owners. Keep the ORIGINAL gamma
        // Vec/shape capacities plus declaration, not just its logical slice length.
        // This still excludes the enclosing manifest/map/files: no whole request
        // admission is inferred from this native-construction boundary.
        let slots = compact_norm::factory_slots()?;
        let baseline = add_bytes(add_bytes(self.baseline()?, gamma_bytes)?, slots)?;
        self.admit_total_with_source(
            add_bytes(
                self.outside_owner_bytes,
                add_bytes(baseline, quote.new_peak_bytes)?,
            )?,
            gamma_bytes,
        )?;
        #[cfg(test)]
        RMS_NATIVE_ATTEMPTS.with(|count| count.set(count.get() + 1));
        let (program, energy) = rms_norm_program(true, width, bounds, coefficients, input_energy)?;
        let resources = program.resources()?;
        if resources.immutable_template_bytes > quote.immutable_template_bytes
            || resources.shared_admitted_bytes > puma::MAX_ADMITTED_BYTES
            || program.output_lanes() != width
            || program.validity_lanes().is_some()
        {
            return Err(Error::Bounds);
        }
        let energy = energy.ok_or(Error::Binding)?;
        // Same original Builder::tensor interning rule for a freshly constructed
        // native Program. No existing closed PreparedTensorMap owner is replaced.
        let existing = self
            .owned_native_programs()
            .find(|old| old.digest() == program.digest())
            .cloned();
        let owns = existing.is_none();
        let program = existing.unwrap_or_else(|| Arc::new(program));
        self.admit_total_with_source(
            add_bytes(
                self.outside_owner_bytes,
                add_bytes(baseline, program.resources()?.immutable_template_bytes)?,
            )?,
            gamma_bytes,
        )?;
        Ok((program, energy, owns))
    }

    pub(super) fn complete_row(
        &self,
        view: &PublishedView,
        row: usize,
    ) -> Result<Option<ProjectionRow<'a>>, Error> {
        self.validate_view(view)?;
        let width = view.logical.row_width();
        if row >= view.lanes / width {
            return Err(Error::Shape);
        }
        let Some(slot) = view.slot else {
            return Ok(None);
        };
        if matches!(
            self.ops[slot].kind,
            PublishedKind::Reshape | PublishedKind::TokenFeedback | PublishedKind::LastToken { .. }
        ) {
            let Some((original, start)) = self.alias_range(view, row * width, width)? else {
                return Ok(None);
            };
            if width != original.logical.row_width() || start % width != 0 {
                return Ok(None);
            }
            return self.complete_row(original, start / width);
        }
        let PublishedKind::Projection {
            weight,
            source,
            row_energy: Some(energy),
            ..
        } = &self.ops[slot].kind
        else {
            return Ok(None);
        };
        let Mapping::Dense(owner) = view.mapping else {
            return Err(Error::Binding);
        };
        Ok(Some(ProjectionRow {
            owner,
            offset: row * view.shape[2],
            width: view.shape[2],
            energy: *energy,
            source_owner: *weight,
            source,
            source_binding: self.binding,
        }))
    }

    fn row_norm(&self, input: &PublishedView) -> Result<Option<u128>, Error> {
        self.validate_view(input)?;
        if let Some(slot) = input.slot {
            if matches!(self.ops[slot].kind, PublishedKind::TokenFeedback) {
                return self.row_norm(&self.ops[slot].input);
            }
            if matches!(self.ops[slot].kind, PublishedKind::Reshape) {
                let width = input.logical.row_width();
                let Some((original, offset)) = self.alias_range(input, 0, input.lanes)? else {
                    return Ok(None);
                };
                if original.logical.row_width() != width || offset % width != 0 {
                    return Ok(None);
                }
                return self.row_norm(original);
            }
            if let PublishedKind::Binary { row_energy, .. } = &self.ops[slot].kind {
                return Ok(*row_energy);
            }
            if let PublishedKind::RmsNorm { output_energy, .. } = &self.ops[slot].kind {
                return Ok(Some(*output_energy));
            }
            if let PublishedKind::LastToken { row_energy } = &self.ops[slot].kind {
                return Ok(*row_energy);
            }
        }
        let mut maximum = 0;
        for row in 0..input.lanes / input.logical.row_width() {
            let Some(proof) = self.complete_row(input, row)? else {
                return Ok(None);
            };
            maximum = maximum.max(proof.energy);
        }
        Ok(Some(maximum))
    }

    /// Transfer only exact ordered original coordinates, never semantic shape
    /// labels. No general-row certificate is inferred from an attention block.
    fn alias_range<'s>(
        &'s self,
        view: &'s PublishedView,
        start: usize,
        count: usize,
    ) -> Result<Option<(&'s PublishedView, usize)>, Error> {
        if count == 0 || start.checked_add(count).is_none_or(|end| end > view.lanes) {
            return Err(Error::Shape);
        }
        let mut original = view;
        while let Some(slot) = original.slot {
            let op = self.ops.get(slot).ok_or(Error::Binding)?;
            if !matches!(
                op.kind,
                PublishedKind::Reshape
                    | PublishedKind::TokenFeedback
                    | PublishedKind::LastToken { .. }
            ) {
                break;
            }
            original = &op.input;
        }
        if original.order != LaneOrder::Linear {
            return Err(Error::Binding);
        }
        let (owner, lane) = view.word(start)?;
        let absolute = match original.mapping {
            Mapping::Dense(id) if owner == id => Some(lane),
            Mapping::Products(first) => owner
                .checked_sub(first)
                .filter(|n| n % 3 == 2)
                .and_then(|n| (n / 3).checked_mul(128))
                .and_then(|n| n.checked_add(lane)),
            Mapping::Rows { first, width } => owner
                .checked_sub(first)
                .filter(|n| n % 2 == 1)
                .and_then(|n| (n / 2).checked_mul(width))
                .and_then(|n| n.checked_add(lane)),
            _ => None,
        };
        let Some(offset) = absolute.and_then(|n| n.checked_sub(original.start)) else {
            return Ok(None);
        };
        if offset
            .checked_add(count)
            .is_none_or(|end| end > original.lanes)
        {
            return Ok(None);
        }
        for i in 0..count {
            if view.word(start + i)? != original.word(offset + i)? {
                return Ok(None);
            }
        }
        Ok(Some((original, offset)))
    }

    /// Attention-only source-block mapping, never used as a general row norm.
    pub(super) fn attention_energy(
        &self,
        view: &PublishedView,
        start: usize,
        count: usize,
    ) -> Result<Option<u128>, Error> {
        self.validate_view(view)?;
        let Some(slot) = view.slot else {
            return Ok(None);
        };
        if matches!(
            self.ops[slot].kind,
            PublishedKind::LastToken { .. } | PublishedKind::TokenFeedback | PublishedKind::Reshape
        ) {
            let Some((original, start)) = self.alias_range(view, start, count)? else {
                return Ok(None);
            };
            return self.attention_energy(original, start, count);
        }
        let PublishedKind::Projection {
            norms: Some(norms),
            input_energy: Some(input),
            geometry: [_, _, columns],
            bias,
            ..
        } = &self.ops[slot].kind
        else {
            return Ok(None);
        };
        let row_start = start % columns;
        if start.checked_add(count).is_none_or(|end| end > view.lanes)
            || count == 0
            || row_start
                .checked_add(count)
                .is_none_or(|end| end > *columns)
        {
            return Err(Error::Shape);
        }
        let Some(certificate) = norms.block(row_start, count) else {
            return Ok(None);
        };
        let mut energy = norm_bounds::linear_energy(*input, certificate, 18)?;
        if let Some(bias) = bias {
            let certificate = bias
                .norms
                .and_then(|norms| norms.block(row_start, count))
                .ok_or(Error::Binding)?;
            energy = norm_bounds::sum_energy(energy, certificate.frobenius_squared)?;
        }
        Ok(Some(energy))
    }

    pub(super) fn lower_next(&mut self) -> Result<PublishedView, Error> {
        let mut attempt = self.request_admission.map(|context| context.attempt());
        if self.failed {
            return Err(Error::Retired);
        }
        let result = self.lower_next_inner();
        // lower_next_inner's contraction guard, factory parts, result wrappers
        // and publication validation have ALL retired before this boundary.
        if result.is_ok() {
            if let Some(controller) = self.request_admission {
                controller.retire_phase()?;
            }
        }
        if result.is_err() {
            self.failed = true;
            self.failure = self.frontier.operations().nth(self.cursor).map(|(_, op)| {
                let (a, b) = self.frontier.inputs(op);
                FailureSite {
                    step: self.cursor,
                    operation: &op.id,
                    operator: op.operator,
                    input_owners: [
                        a.and_then(|v| v.word(0).ok().map(|word| word.0)),
                        b.and_then(|v| v.word(0).ok().map(|word| word.0)),
                    ],
                    source_owner: op
                        .attributes
                        .get("weight")
                        .and_then(|value| value.as_str())
                        .and_then(|key| {
                            self.sources
                                .iter()
                                .filter(|(_, source)| source.public_q18.is_none())
                                .position(|(name, _)| name == key)
                        }),
                    next_virtual_owner: self.next,
                }
            });
        }
        if result.is_ok() {
            if let Some(attempt) = attempt.as_mut() {
                attempt.commit();
            }
        }
        result
    }

    fn lower_next_inner(&mut self) -> Result<PublishedView, Error> {
        let graph::DispatchStep::Operator {
            operation: op,
            left: a,
            right: b,
            ..
        } = graph::dispatch_operator(&self.frontier, self.cursor)?
        else {
            return Err(Error::Binding);
        };
        let input = if op.operator == Op::TokenLookup && self.prebound_inputs.is_some() {
            // The original driver receives the request's GLOBAL token view,
            // independently of the operation's first semantic input.
            self.bound_imports
                .iter()
                .find(|bound| bound.semantic == "input.tokens")
                .map(|bound| &bound.view)
                .ok_or(Error::Binding)?
                .clone()
        } else {
            // Keep the old isolated late-bind qualifier's caller-selected token
            // argument; it has no sealed aggregate-construction authority.
            a.ok_or(Error::Binding)?.clone()
        };
        let right = b.cloned();
        self.validate_view(&input)?;
        let logical = ViewShape::semantic(&op.output_shape)?;
        let shape = logical.envelope()?;
        let lanes = logical.lanes()?;
        if lanes == 0 || lanes > MAX_VALUES {
            return Err(Error::Bounds);
        }
        let first = self.next;
        let slot = self.ops.len();
        if slot == self.ops.capacity() || add_bytes(self.imports.nodes.len(), slot + 1)? > MAX_NODES
        {
            return Err(Error::Bounds);
        }
        self.admit_local(self.baseline()?)?;
        if !matches!(op.operator, Op::Reshape | Op::TokenFeedback)
            && (input.rank > 3 || logical.rank > 3 || input.shape[0] != 1 || shape[0] != 1)
        {
            return Err(Error::Shape);
        }
        let reshape = if op.operator == Op::Reshape {
            Some(reshape_order(
                &input,
                logical,
                op.attributes.get("layout").and_then(|v| v.as_str()),
            )?)
        } else {
            None
        };
        let (kind, prepared, mapping, bounds, nodes, work) = match op.operator {
            Op::TokenLookup if op.inputs.len() == 1 => {
                if self.imports.lookup != LookupPolicy::PerToken {
                    return Err(Error::Binding);
                }
                if !matches!(input.rank, 1 | 2) || logical.rank != 3 || shape[1] != input.lanes {
                    return Err(Error::Shape);
                }
                self.lower_contraction(op, &input, logical)?
            }
            Op::Reshape if op.inputs.len() == 1 && lanes == input.lanes => (
                PublishedKind::Reshape,
                None,
                input.mapping,
                input.bounds,
                0,
                WorkBudget::default(),
            ),
            Op::GreedyTokenSelection if op.inputs.len() == 1 && op.output_shape == [1] => {
                if self.imports.selection != SelectionPolicy::Expanded
                    || self.imports.lookup == LookupPolicy::Trie
                {
                    // Tiled has a different original recipe. Trie additionally
                    // registers selected-token source proofs in lookup_builder;
                    // neither can be silently replaced by this expanded tree.
                    return Err(Error::Binding);
                }
                let plan = CompactArgmax::lower(self, &input, first)?;
                let bounds = Interval::new(0, input.lanes as i64 - 1)?;
                let mapping = Mapping::Dense(plan.output.0);
                if plan.output.1 != 0 {
                    return Err(Error::Binding);
                }
                let nodes = plan.geometry.nodes()?;
                let work = plan.work;
                (
                    PublishedKind::Argmax(plan),
                    None,
                    mapping,
                    bounds,
                    nodes,
                    work,
                )
            }
            Op::RmsNorm
                if op.inputs.len() == 1
                    && logical == input.logical
                    && input.rank == 3
                    && op.output_shape.len() == 3 =>
            {
                let width = shape[2];
                let publication = self.lower_rms_norm(op, &input, &shape)?;
                (
                    publication.kind,
                    None,
                    Mapping::Rows {
                        first: publication.rows_first,
                        width,
                    },
                    publication.bounds,
                    publication.nodes,
                    publication.work,
                )
            }
            Op::Linear | Op::OutputHead if op.inputs.len() == 1 && matches!(input.rank, 2 | 3) => {
                self.lower_contraction(op, &input, logical)?
            }
            Op::Silu
                if op.inputs.len() == 1
                    && logical == input.logical
                    && op.output_shape.len() == input.rank as usize =>
            {
                let prepared = self.prepare(MapKind::GuardedSilu, lanes, input.bounds)?;
                let map = self.map(&prepared)?;
                let bounds = map.output_bounds()?;
                let work = map.work_budget()?;
                (
                    PublishedKind::Silu,
                    Some(prepared),
                    Mapping::Dense(first.checked_add(1).ok_or(Error::Bounds)?),
                    bounds,
                    2,
                    work,
                )
            }
            Op::Multiply | Op::ResidualAdd if op.inputs.len() == 2 => {
                let right = right.ok_or(Error::Binding)?;
                self.validate_view(&right)?;
                if logical != input.logical
                    || logical != right.logical
                    || input.rank != right.rank
                    || input.rank as usize != op.output_shape.len()
                {
                    return Err(Error::Shape);
                }
                let mut sink = PublicationSink { first, next: first };
                let product = op.operator == Op::Multiply;
                let summary = binary(&mut sink, &input, &right, product)?;
                let row_energy = if product {
                    None
                } else {
                    let scalar = |view: &PublishedView| -> Result<u128, Error> {
                        let magnitude = u128::from(
                            view.bounds
                                .lower()
                                .unsigned_abs()
                                .max(view.bounds.upper().unsigned_abs()),
                        );
                        magnitude
                            .checked_mul(magnitude)
                            .and_then(|n| n.checked_mul(view.shape[2] as u128))
                            .ok_or(Error::Bounds)
                    };
                    let left_energy = self.row_norm(&input)?.map_or_else(|| scalar(&input), Ok)?;
                    let right_energy = self.row_norm(&right)?.map_or_else(|| scalar(&right), Ok)?;
                    Some(residual_energy(left_energy, right_energy)?)
                };
                (
                    PublishedKind::Binary {
                        right,
                        product,
                        row_energy,
                    },
                    None,
                    Mapping::Products(first),
                    summary.bounds.ok_or(Error::Shape)?,
                    summary.tiles * 3,
                    summary.work,
                )
            }
            Op::LastToken
                if matches!(op.inputs.len(), 1 | 2)
                    && input.rank == 3
                    && op.output_shape.len() == 2
                    && shape == [1, 1, input.shape[2]] =>
            {
                let row_energy = self.row_norm(&input)?;
                (
                    PublishedKind::LastToken { row_energy },
                    None,
                    input.mapping,
                    input.bounds,
                    0,
                    WorkBudget::default(),
                )
            }
            Op::TokenFeedback if op.inputs.len() == 1 && lanes == input.lanes => (
                PublishedKind::TokenFeedback,
                None,
                input.mapping,
                input.bounds,
                0,
                WorkBudget::default(),
            ),
            _ => return Err(Error::Binding),
        };
        let next = first.checked_add(nodes).ok_or(Error::Bounds)?;
        let work = self.work.checked_add(work)?;
        work.limits()?;
        if let Some(owner) = &self.prebound_inputs {
            owner.work.checked_add(work)?.limits()?;
        }
        let output_start = match kind {
            PublishedKind::LastToken { .. } => input
                .start
                .checked_add(input.lanes.checked_sub(lanes).ok_or(Error::Shape)?)
                .ok_or(Error::Bounds)?,
            PublishedKind::TokenFeedback => input.start,
            PublishedKind::Reshape => reshape.ok_or(Error::Binding)?.1,
            _ => 0,
        };
        // The expanded graph checks the declared lane count, then publishes the
        // cloned input View. In particular, feedback is NOT a reshape to [1, 1].
        let (output_shape, output_rank) = if matches!(kind, PublishedKind::TokenFeedback) {
            (input.shape, input.rank)
        } else {
            (shape, op.output_shape.len() as u8)
        };
        let output_logical = if matches!(kind, PublishedKind::TokenFeedback) {
            input.logical
        } else {
            logical
        };
        let output_order = match kind {
            PublishedKind::LastToken { .. } | PublishedKind::TokenFeedback => input.order,
            PublishedKind::Reshape => reshape.ok_or(Error::Binding)?.0,
            _ => LaneOrder::Linear,
        };
        let output = PublishedView {
            origin: self.origin.clone(),
            slot: Some(slot),
            mapping,
            start: output_start,
            lanes,
            shape: output_shape,
            rank: output_rank,
            logical: output_logical,
            order: output_order,
            import_slot: None,
            bounds,
        };
        self.ops.push(PublishedOp {
            first,
            input,
            output: output.clone(),
            kind,
            prepared,
        });
        #[cfg(test)]
        if self.request_admission.is_some()
            && REQUEST_LOOKUP_FAILURE.with(|stage| {
                if stage.get() == 2 {
                    stage.set(0);
                    true
                } else {
                    false
                }
            })
        {
            return Err(Error::Allocation);
        }
        self.frontier.finish(op, output.clone())?;
        if let PublishedKind::RmsNorm {
            checked: Some(rows),
            ..
        } = &self.ops[slot].kind
        {
            self.validity.commit(rows)?;
        }
        self.next = next;
        self.work = work;
        self.cursor += 1;
        Ok(output)
    }

    pub(super) fn work_budget(&self) -> WorkBudget {
        self.work
    }

    #[cfg(test)]
    pub(super) fn test_lower_limit(&mut self, limit: usize) {
        assert!(limit <= self.limit);
        self.limit = limit;
    }

    pub(in crate::replicated_decoder) fn failure_site(&self) -> Option<FailureSite<'g>> {
        self.failure
    }
    pub(super) fn stored_descriptors(&self) -> usize {
        self.imports.nodes.len() + self.ops.len()
    }

    pub(super) fn hash_expanded_nodes(&self, hash: &mut Sha256) -> Result<(), Error> {
        for (slot, op) in self.ops.iter().enumerate() {
            match &op.kind {
                PublishedKind::TokenLookup { weight, plan, .. } => {
                    let width = op.output.shape[2];
                    for token in 0..op.input.lanes {
                        let first = op.first + 3 * token;
                        let (owner, index) = op.input.word(token)?;
                        hash_node_header(hash, plan.vocabulary, plan.selector_bounds);
                        hash_tensor_loop(
                            hash,
                            LoopInput::TokenOffsets { owner, index },
                            &plan.full.program,
                            plan.tail.as_ref().map(|tail| &tail.program),
                        );
                        hash_node_header(hash, plan.vocabulary, plan.selector_bounds);
                        hash_gather(hash, (0..plan.vocabulary).map(|lane| (first, lane)));
                        // graph::TokenLookup replaces the raw product interval
                        // AFTER public_matrix_contract succeeds, without rescale.
                        hash_node_header(hash, width, op.output.bounds);
                        hash_public_product(
                            hash,
                            first + 1,
                            *weight,
                            1,
                            plan.vocabulary,
                            width,
                            true,
                        );
                    }
                }
                PublishedKind::Projection {
                    weight,
                    geometry: [rows, inner, columns],
                    raw_bounds,
                    bias,
                    ..
                } => {
                    hash_node_header(hash, op.input.lanes, op.input.bounds);
                    hash_gather(hash, op.input.words());
                    hash_node_header(hash, op.output.lanes, *raw_bounds);
                    hash_public_product(hash, op.first, *weight, *rows, *inner, *columns, false);
                    hash_node_header(
                        hash,
                        op.output.lanes,
                        bias.as_ref()
                            .map_or(op.output.bounds, |bias| bias.floor_bounds),
                    );
                    hash_prepared_loop(hash, op.first + 1, self.kernel(slot)?)?;
                    if let Some(bias) = bias {
                        for start in (0..op.output.lanes).step_by(128) {
                            let lanes = 128.min(op.output.lanes - start);
                            let first = op.first + 3 + 3 * (start / 128);
                            hash_node_header(hash, lanes, bias.floor_bounds);
                            hash_gather(
                                hash,
                                (start..start + lanes).map(|lane| (op.first + 2, lane)),
                            );
                            hash_node_header(hash, lanes, bias.bounds()?);
                            hash_gather(
                                hash,
                                (start..start + lanes).map(|lane| (bias.owner, lane % columns)),
                            );
                            hash_node_header(hash, lanes, op.output.bounds);
                            hash_affine(hash, first, Some(first + 1), bias.coefficients);
                        }
                    }
                }
                PublishedKind::Silu => {
                    hash_node_header(hash, op.input.lanes, op.input.bounds);
                    hash_gather(hash, op.input.words());
                    hash_node_header(hash, op.output.lanes, op.output.bounds);
                    hash_prepared_loop(hash, op.first, self.kernel(slot)?)?;
                }
                PublishedKind::RmsNorm {
                    program,
                    checked,
                    row_bounds,
                    ..
                } => {
                    let width = op.output.shape[2];
                    if let Some(rows) = checked {
                        rows.visit(&mut |event| {
                            use compact_validity::{GuardCode, GuardGather};
                            hash_node_header(hash, event.lanes, event.bounds);
                            match event.code {
                                GuardCode::Gather(GuardGather::Original { start }) => hash_gather(
                                    hash,
                                    (start..start + event.lanes)
                                        .map(|i| op.input.word(i).expect("original guard input")),
                                ),
                                GuardCode::Gather(GuardGather::Native { owner, start }) => {
                                    hash_gather(
                                        hash,
                                        (start..start + event.lanes).map(|i| (owner, i)),
                                    )
                                }
                                GuardCode::Checked { input, program } => {
                                    hash.update([9]);
                                    hash.update((input as u64).to_le_bytes());
                                    hash.update(program.digest());
                                }
                                GuardCode::AllValid { input, program } => {
                                    hash_tensor(hash, input, program)
                                }
                                GuardCode::Product { left, right } => {
                                    hash_product(hash, left, right, None, false, None)
                                }
                            }
                            Ok(())
                        })?;
                    }
                    let rows_first = checked.as_ref().map_or(op.first, |rows| rows.rows_first());
                    let bounds = *row_bounds;
                    for row in 0..op.output.shape[1] {
                        let first = rows_first + 2 * row;
                        hash_node_header(hash, width, bounds);
                        hash_gather(
                            hash,
                            (row * width..(row + 1) * width).map(|i| {
                                checked
                                    .as_ref()
                                    .map_or_else(|| op.input.word(i), |rows| rows.word(i))
                                    .expect("checked RMS row")
                            }),
                        );
                        hash_node_header(hash, width, program.output_bounds());
                        hash_tensor(hash, first, program);
                    }
                }
                PublishedKind::LastToken { .. }
                | PublishedKind::TokenFeedback
                | PublishedKind::Reshape => {}
                PublishedKind::Argmax(plan) => {
                    plan.visit(&op.input, &mut |event| hash_argmax_event(hash, event))?;
                }
                PublishedKind::Binary { right, product, .. } => {
                    for start in (0..op.output.lanes).step_by(128) {
                        let lanes = 128.min(op.output.lanes - start);
                        let first = op.first + 3 * (start / 128);
                        for input in [&op.input, right] {
                            hash_node_header(hash, lanes, input.bounds);
                            hash_gather(
                                hash,
                                (start..start + lanes)
                                    .map(|i| input.word(i).expect("checked compact view")),
                            );
                        }
                        hash_node_header(hash, lanes, op.output.bounds);
                        if *product {
                            hash_product(hash, first, first + 1, None, true, None);
                        } else {
                            hash_affine(hash, first, Some(first + 1), affine(1, 1, 0));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct ArgmaxGeometry {
    lanes: usize,
    rounds: usize,
    blocks: usize,
    kernels: usize,
}

impl ArgmaxGeometry {
    fn new(lanes: usize) -> Result<Self, Error> {
        if lanes == 0 || lanes > MAX_VALUES {
            return Err(Error::Shape);
        }
        // One scalar bit per possible native pair count; no shape/lane array
        // needs constructing before the enclosing allocation gate.
        let mut seen = 0_u128;
        let (mut length, mut rounds, mut blocks) = (lanes, 0, 0usize);
        while length > 1 {
            blocks = add_bytes(blocks, length / 256 + usize::from(length % 256 >= 2))?;
            if length >= 256 {
                seen |= 1_u128 << 127;
            }
            if length % 256 >= 2 {
                seen |= 1_u128 << ((length % 256) / 2 - 1);
            }
            length = length.div_ceil(2);
            rounds += 1;
        }
        Ok(Self {
            lanes,
            rounds,
            blocks,
            kernels: seen.count_ones() as usize,
        })
    }
    fn nodes(self) -> Result<usize, Error> {
        self.blocks
            .checked_mul(12)
            .and_then(|n| n.checked_add(1))
            .ok_or(Error::Bounds)
    }
    fn level(self, round: usize) -> Result<(usize, usize), Error> {
        if round > self.rounds {
            return Err(Error::Shape);
        }
        let (mut length, mut blocks) = (self.lanes, 0usize);
        for _ in 0..round {
            blocks = add_bytes(blocks, length / 256 + usize::from(length % 256 >= 2))?;
            length = length.div_ceil(2);
        }
        Ok((length, blocks))
    }
    fn word(
        self,
        input: &PublishedView,
        first: usize,
        mut round: usize,
        field: ArgmaxField,
        mut index: usize,
    ) -> Result<(usize, usize), Error> {
        if index >= self.level(round)?.0 {
            return Err(Error::Shape);
        }
        while round > 0 {
            let (previous, blocks) = self.level(round - 1)?;
            if previous % 2 != 0 && index == previous / 2 {
                index = previous - 1;
                round -= 1;
                continue;
            }
            let block = add_bytes(blocks, index / 128)?;
            let owner = first
                .checked_add(1)
                .and_then(|n| n.checked_add(block.checked_mul(12)?))
                .and_then(|n| n.checked_add(if field == ArgmaxField::Values { 6 } else { 11 }))
                .ok_or(Error::Bounds)?;
            return Ok((owner, index % 128));
        }
        match field {
            ArgmaxField::Values => input.word(index),
            ArgmaxField::Indices => Ok((first, index)),
        }
    }
}

struct NegativeProgram {
    lanes: usize,
    bounds: Interval,
    program: Arc<tensor::Program>,
    owns: bool,
}

/// Read-only ORIGINAL primitive publication. Visitors may consume immediately;
/// no retained expanded Nodes, per-lane refs, public-iota array or runtime exists.
pub(in crate::replicated_decoder) struct ArgmaxEvent<'s> {
    pub owner: usize,
    pub lanes: usize,
    pub bounds: Interval,
    pub code: ArgmaxCode<'s>,
}

pub(in crate::replicated_decoder) enum ArgmaxCode<'s> {
    PublicIndices,
    Gather(ArgmaxGather<'s>),
    Affine {
        left: usize,
        right: usize,
        coefficients: Affine,
    },
    Product {
        left: usize,
        right: usize,
    },
    Tensor {
        input: usize,
        program: &'s Arc<tensor::Program>,
    },
}

pub(in crate::replicated_decoder) struct ArgmaxGather<'s> {
    input: &'s PublishedView,
    geometry: ArgmaxGeometry,
    first: usize,
    round: usize,
    field: ArgmaxField,
    start: usize,
    pairs: usize,
    parity: usize,
}

#[derive(Clone, Copy)]
pub(in crate::replicated_decoder) struct ArgmaxGatherLayout {
    pub input: PhysicalView,
    pub first: usize,
    pub input_lanes: usize,
    pub round: usize,
    pub field: ArgmaxField,
    pub start: usize,
    pub pairs: usize,
    pub parity: usize,
}

impl ArgmaxGather<'_> {
    /// Keep the indexed layout when emitting; do not retain words().collect()
    /// merely to adapt this view back into an expanded per-lane consumer.
    pub(in crate::replicated_decoder) fn layout(&self) -> ArgmaxGatherLayout {
        ArgmaxGatherLayout {
            input: self.input.physical(),
            first: self.first,
            input_lanes: self.geometry.lanes,
            round: self.round,
            field: self.field,
            start: self.start,
            pairs: self.pairs,
            parity: self.parity,
        }
    }

    pub(in crate::replicated_decoder) fn words(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.pairs).map(|i| {
            self.geometry
                .word(
                    self.input,
                    self.first,
                    self.round,
                    self.field,
                    self.start + 2 * i + self.parity,
                )
                .expect("checked original argmax gather")
        })
    }
}

#[derive(Clone, Copy)]
pub(in crate::replicated_decoder) struct ArgmaxCensus {
    pub first: usize,
    pub lanes: usize,
    pub rounds: usize,
    pub compare_blocks: usize,
    pub expanded_nodes: usize,
    pub kernel_count: usize,
    pub kernel_capacity: usize,
    pub kernel_metadata_bytes: usize,
    /// Inclusive local + independently supplied enclosing outside baseline.
    pub construction_peak_bytes: usize,
}

struct CompactArgmax {
    first: usize,
    geometry: ArgmaxGeometry,
    programs: Vec<NegativeProgram>,
    output: (usize, usize),
    work: WorkBudget,
    peak: usize,
}

impl CompactArgmax {
    /// Known logical Count/traversal/control slots only. Native-query slots are
    /// owned by the original native API, not reconstructed from its recipe here.
    /// Compiled/dependency callframes, execution scratch and RSS remain unquoted.
    fn work_phase_bytes(geometry: ArgmaxGeometry) -> Result<usize, Error> {
        let native = tensor::Program::negative_work_resources();
        let mut bytes = if geometry.blocks == 0 {
            0
        } else {
            native.known_peak_bytes
        };
        for slot in [
            // new() argument/return bridge, then the bound Count state (the
            // latter is already part of the common phase below).
            2 * std::mem::size_of::<Result<ArgmaxNumeric<'_, '_>, Error>>(),
            std::mem::size_of::<NegativePrograms<'_>>(),
            2 * std::mem::size_of::<ArgmaxGeometry>(),
            // Eight tree owners, five select owners, three affine/product
            // argument/result owners. These are scalar handles, not lane refs.
            array_bytes::<Value>(16)?,
            array_bytes::<usize>(6)?,
            4 * std::mem::size_of::<Result<Value, Error>>(),
            2 * std::mem::size_of::<Result<(usize, usize), Error>>(),
            3 * std::mem::size_of::<Interval>(),
            2 * std::mem::size_of::<Result<Interval, Error>>(),
            // Negative result/record sum; graph sum; outside sum; limits
            // receiver/argument moves. No request or graph ledger mutation.
            6 * std::mem::size_of::<WorkBudget>(),
            4 * std::mem::size_of::<Result<WorkBudget, Error>>(),
            2 * std::mem::size_of::<Result<puma::prss::PrssLimits, Error>>(),
            2 * std::mem::size_of::<tensor::NegativeWorkResources>(),
            std::mem::size_of::<Result<(), Error>>(),
        ] {
            bytes = add_bytes(bytes, slot)?;
        }
        Ok(bytes)
    }

    fn quote(lanes: usize, baseline: usize) -> Result<(ArgmaxGeometry, usize), Error> {
        let geometry = ArgmaxGeometry::new(lanes)?;
        let mut native = NativeConstructionPeak::default();
        let mut seen = 0_u128;
        let mut include = |pairs: usize| -> Result<(), Error> {
            let bit = 1_u128 << (pairs - 1);
            if seen & bit == 0 {
                native.include(tensor::Construction::Negative { lanes: pairs })?;
                seen |= bit;
            }
            Ok(())
        };
        // Original 256-value rounds construct one native lane per pair, never a
        // whole-logit tile. Quote each distinct full/tail pair count in call order.
        let mut length = lanes;
        while length > 1 {
            if length >= 256 {
                include(128)?;
            }
            let tail = (length % 256) / 2;
            if tail != 0 {
                include(tail)?;
            }
            length = length.div_ceil(2);
        }
        let mut common = baseline;
        for bytes in [
            2 * std::mem::size_of::<Result<Self, Error>>(),
            std::mem::size_of::<ArgmaxNumeric<'_, '_>>(),
            std::mem::size_of::<Result<Value, Error>>(),
            std::mem::size_of::<ArgmaxEvent<'_>>(),
            // Retain the pending work report across later native construction
            // so actual emission can be checked without a second debit.
            std::mem::size_of::<WorkBudget>(),
        ] {
            common = add_bytes(common, bytes)?;
        }
        let construction = add_bytes(
            add_bytes(common, array_bytes::<NegativeProgram>(geometry.kernels)?)?,
            native.bytes()?,
        )?;
        let counting = add_bytes(common, Self::work_phase_bytes(geometry)?)?;
        // Count is dropped BEFORE Vec/native construction. No summing disjoint
        // work/native phases and no fictional native backing for the count pass.
        let peak = construction.max(counting);
        Ok((geometry, peak))
    }

    fn lower(
        graph: &CompactGraph<'_, '_>,
        input: &PublishedView,
        first: usize,
    ) -> Result<Self, Error> {
        let (geometry, local_peak) = Self::quote(input.lanes, graph.baseline()?)?;
        graph.admit_local(local_peak)?; // BEFORE metadata reservation or ANY native factory
                                        // Replay the SAME complete tree with scalar work state only. No native
                                        // Program, instruction recipe, lane table or interner owner is created.
        let pending = {
            let mut sink =
                ArgmaxNumeric::new(input, first, geometry, NegativePrograms::Count, None)?;
            let output = argmax_tree(&mut sink, input.lanes)?;
            if output != geometry.word(input, first, geometry.rounds, ArgmaxField::Indices, 0)?
                || sink.next.checked_sub(first) != Some(geometry.nodes()?)
            {
                return Err(Error::Binding);
            }
            sink.work
        };
        {
            let total = graph.work.checked_add(pending)?;
            total.limits()?;
            if let Some(owner) = &graph.prebound_inputs {
                owner.work.checked_add(total)?.limits()?;
            }
        }
        let mut programs = Vec::new();
        programs
            .try_reserve_exact(geometry.kernels)
            .map_err(|_| Error::Allocation)?;
        if programs.capacity() != geometry.kernels {
            return Err(Error::Allocation);
        }
        let prior = |digest: [u8; 32]| {
            graph
                .owned_native_programs()
                .find(|program| program.digest() == digest)
                .cloned()
        };
        let (output, work) = {
            let mut sink = ArgmaxNumeric::new(
                input,
                first,
                geometry,
                NegativePrograms::Build {
                    programs: &mut programs,
                    prior: &prior,
                },
                None,
            )?;
            let output = argmax_tree(&mut sink, input.lanes)?;
            let work = sink.work;
            if sink.next.checked_sub(first) != Some(geometry.nodes()?) || work != pending {
                return Err(Error::Binding);
            }
            (output, work)
        };
        Ok(Self {
            first,
            geometry,
            programs,
            output,
            work,
            peak: add_bytes(graph.outside_owner_bytes, local_peak)?,
        })
    }

    fn census(&self) -> Result<ArgmaxCensus, Error> {
        Ok(ArgmaxCensus {
            first: self.first,
            lanes: self.geometry.lanes,
            rounds: self.geometry.rounds,
            compare_blocks: self.geometry.blocks,
            expanded_nodes: self.geometry.nodes()?,
            kernel_count: self.programs.len(),
            kernel_capacity: self.programs.capacity(),
            kernel_metadata_bytes: array_bytes::<NegativeProgram>(self.programs.capacity())?,
            construction_peak_bytes: self.peak,
        })
    }

    fn baseline(&self) -> Result<usize, Error> {
        let mut bytes = array_bytes::<NegativeProgram>(self.programs.capacity())?;
        for program in &self.programs {
            if program.owns {
                bytes = add_bytes(bytes, program.program.resources()?.immutable_template_bytes)?;
            }
        }
        Ok(bytes)
    }

    fn visit(
        &self,
        input: &PublishedView,
        visitor: &mut dyn for<'e> FnMut(ArgmaxEvent<'e>) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let mut sink = ArgmaxNumeric::new(
            input,
            self.first,
            self.geometry,
            NegativePrograms::Borrowed(&self.programs),
            Some(visitor),
        )?;
        let output = argmax_tree(&mut sink, input.lanes)?;
        if output != self.output
            || sink.work != self.work
            || sink.next.checked_sub(self.first) != Some(self.geometry.nodes()?)
        {
            return Err(Error::Binding);
        }
        Ok(())
    }
}

#[cfg(test)]
thread_local! { static ARGMAX_NATIVE_ATTEMPTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

enum NegativePrograms<'s> {
    /// Work-only traversal, never an executable native owner or publication.
    Count,
    Build {
        programs: &'s mut Vec<NegativeProgram>,
        prior: &'s dyn Fn([u8; 32]) -> Option<Arc<tensor::Program>>,
    },
    Borrowed(&'s [NegativeProgram]),
}

impl NegativePrograms<'_> {
    fn get(&mut self, lanes: usize, bounds: Interval) -> Result<&Arc<tensor::Program>, Error> {
        match self {
            Self::Count => Err(Error::Binding),
            Self::Borrowed(programs) => programs
                .iter()
                .find(|program| program.lanes == lanes && program.bounds == bounds)
                .map(|program| &program.program)
                .ok_or(Error::Binding),
            Self::Build { programs, prior } => {
                if let Some(index) = programs
                    .iter()
                    .position(|program| program.lanes == lanes && program.bounds == bounds)
                {
                    return Ok(&programs[index].program);
                }
                if programs.len() == programs.capacity() {
                    return Err(Error::Bounds);
                }
                let quote = native_construction_quote(tensor::Construction::Negative { lanes })?;
                #[cfg(test)]
                ARGMAX_NATIVE_ATTEMPTS.with(|count| count.set(count.get() + 1));
                let program = tensor::Program::negative(lanes, bounds)?;
                let resources = program.resources()?;
                if resources.immutable_template_bytes != quote.immutable_template_bytes
                    || resources.shared_admitted_bytes > puma::MAX_ADMITTED_BYTES
                    || program.lanes() != lanes
                    || program.output_lanes() != lanes
                    || program.output_bounds() != Interval::new(0, 1)?
                    || program.validity_lanes().is_some()
                {
                    return Err(Error::Binding);
                }
                let existing = prior(program.digest());
                let owns = existing.is_none();
                programs.push(NegativeProgram {
                    lanes,
                    bounds,
                    owns,
                    program: existing.unwrap_or_else(|| Arc::new(program)),
                });
                Ok(&programs.last().ok_or(Error::Binding)?.program)
            }
        }
    }
}

type ArgmaxVisitor<'v> = &'v mut (dyn for<'e> FnMut(ArgmaxEvent<'e>) -> Result<(), Error> + 'v);

struct ArgmaxNumeric<'s, 'v> {
    input: &'s PublishedView,
    geometry: ArgmaxGeometry,
    first: usize,
    next: usize,
    round: usize,
    length: usize,
    index_bounds: Interval,
    programs: NegativePrograms<'s>,
    visitor: Option<ArgmaxVisitor<'v>>,
    work: WorkBudget,
}

impl<'s, 'v> ArgmaxNumeric<'s, 'v> {
    fn new(
        input: &'s PublishedView,
        first: usize,
        geometry: ArgmaxGeometry,
        programs: NegativePrograms<'s>,
        visitor: Option<ArgmaxVisitor<'v>>,
    ) -> Result<Self, Error> {
        if matches!(&programs, NegativePrograms::Count) && visitor.is_some() {
            return Err(Error::Binding);
        }
        Ok(Self {
            input,
            geometry,
            first,
            next: first,
            round: 0,
            length: input.lanes,
            index_bounds: Interval::new(0, input.lanes as i64 - 1)?,
            programs,
            visitor,
            work: WorkBudget::default(),
        })
    }
    fn emit(
        &mut self,
        lanes: usize,
        bounds: Interval,
        code: ArgmaxCode<'_>,
        work: WorkBudget,
    ) -> Result<Value, Error> {
        let output = self.record(lanes, bounds, work)?;
        if let Some(visitor) = &mut self.visitor {
            visitor(ArgmaxEvent {
                owner: output.id,
                lanes,
                bounds,
                code,
            })?;
        }
        Ok(output)
    }
    fn record(&mut self, lanes: usize, bounds: Interval, work: WorkBudget) -> Result<Value, Error> {
        let id = self.next;
        self.next = self.next.checked_add(1).ok_or(Error::Bounds)?;
        self.work = self.work.checked_add(work)?;
        Ok(Value {
            id,
            lanes,
            bounds,
            work,
        })
    }
    fn affine_value(
        &mut self,
        left: Value,
        right: Value,
        coefficients: Affine,
        selected: Option<Interval>,
    ) -> Result<Value, Error> {
        if left.lanes != right.lanes {
            return Err(Error::Shape);
        }
        let raw = coefficients.output_bounds(InputBounds {
            left: left.bounds,
            right: right.bounds,
        })?;
        self.emit(
            left.lanes,
            selected.unwrap_or(raw),
            ArgmaxCode::Affine {
                left: left.id,
                right: right.id,
                coefficients,
            },
            WorkBudget::default(),
        )
    }
}

impl SelectSink for ArgmaxNumeric<'_, '_> {
    type Owner = Value;
    fn bounds(&self, owner: Value) -> Interval {
        owner.bounds
    }
    fn affine(&mut self, left: Value, right: Value, coefficients: Affine) -> Result<Value, Error> {
        self.affine_value(left, right, coefficients, None)
    }
    fn product(&mut self, left: Value, right: Value) -> Result<Value, Error> {
        if left.lanes != right.lanes || left.lanes > puma::MAX_LANES {
            return Err(Error::Shape);
        }
        let bounds = product_bounds(
            InputBounds {
                left: left.bounds,
                right: right.bounds,
            },
            1,
            false,
            false,
            None,
        )?;
        self.emit(
            left.lanes,
            bounds,
            ArgmaxCode::Product {
                left: left.id,
                right: right.id,
            },
            WorkBudget::product(left.lanes, false)?,
        )
    }
    fn selected_affine(
        &mut self,
        no: Value,
        product: Value,
        bounds: Interval,
    ) -> Result<Value, Error> {
        self.affine_value(no, product, affine(1, 1, 0), Some(bounds))
    }
}

impl ArgmaxSink for ArgmaxNumeric<'_, '_> {
    type Owner = Value;
    type Output = (usize, usize);
    fn indices(&mut self, lanes: usize) -> Result<(), Error> {
        if lanes != self.geometry.lanes {
            return Err(Error::Binding);
        }
        self.emit(
            lanes,
            self.index_bounds,
            ArgmaxCode::PublicIndices,
            WorkBudget::default(),
        )?;
        Ok(())
    }
    fn begin_round(&mut self, round: usize, lanes: usize) -> Result<(), Error> {
        if self.geometry.level(round)?.0 != lanes {
            return Err(Error::Binding);
        }
        self.round = round;
        self.length = lanes;
        Ok(())
    }
    fn gather_pairs(
        &mut self,
        field: ArgmaxField,
        start: usize,
        pairs: usize,
        parity: usize,
    ) -> Result<Value, Error> {
        if pairs == 0
            || pairs > 128
            || parity > 1
            || start
                .checked_add(2 * pairs)
                .is_none_or(|end| end > self.length)
        {
            return Err(Error::Shape);
        }
        let bounds = if field == ArgmaxField::Values {
            self.input.bounds
        } else {
            self.index_bounds
        };
        self.emit(
            pairs,
            bounds,
            ArgmaxCode::Gather(ArgmaxGather {
                input: self.input,
                geometry: self.geometry,
                first: self.first,
                round: self.round,
                field,
                start,
                pairs,
                parity,
            }),
            WorkBudget::default(),
        )
    }
    fn difference(&mut self, left: Value, right: Value) -> Result<Value, Error> {
        self.affine_value(left, right, affine(1, -1, 0), None)
    }
    fn negative(&mut self, difference: Value) -> Result<Value, Error> {
        if matches!(&self.programs, NegativePrograms::Count) {
            let work = tensor::Program::negative_work(difference.lanes, difference.bounds)?;
            // Native Negative's arithmetic-bit envelope, used only by this
            // private counting traversal. No proof or executable owner is minted.
            return self.record(difference.lanes, Interval::new(0, 1)?, work);
        }
        // Split borrows: the event borrows the ORIGINAL stored Arc, not a clone
        // or reconstructed equal-digest template. Frozen visitation allocates none.
        let id = self.next;
        let program = self.programs.get(difference.lanes, difference.bounds)?;
        let bounds = program.output_bounds();
        let work = program.work_budget();
        self.work = self.work.checked_add(work)?;
        self.next = self.next.checked_add(1).ok_or(Error::Bounds)?;
        if let Some(visitor) = &mut self.visitor {
            visitor(ArgmaxEvent {
                owner: id,
                lanes: difference.lanes,
                bounds,
                code: ArgmaxCode::Tensor {
                    input: difference.id,
                    program,
                },
            })?;
        }
        Ok(Value {
            id,
            lanes: difference.lanes,
            bounds,
            work,
        })
    }
    fn select(&mut self, bit: Value, yes: Value, no: Value) -> Result<Value, Error> {
        select(self, bit, yes, no)
    }
    fn publish_pairs(&mut self, best: Value, index: Value, lanes: usize) -> Result<(), Error> {
        if best.lanes != lanes
            || index.lanes != lanes
            || best.bounds != self.input.bounds
            || index.bounds != self.index_bounds
            || best.id.checked_add(6) != Some(self.next)
            || index.id.checked_add(1) != Some(self.next)
            || self
                .next
                .checked_sub(self.first)
                .and_then(|n| n.checked_sub(1))
                .is_none_or(|n| n % 12 != 0)
        {
            return Err(Error::Binding);
        }
        Ok(())
    }
    fn carry(&mut self, index: usize) -> Result<(), Error> {
        if self.length % 2 == 0 || index + 1 != self.length {
            return Err(Error::Binding);
        }
        Ok(())
    }
    fn end_round(&mut self) -> Result<(), Error> {
        let blocks = self.geometry.level(self.round + 1)?.1;
        let expected = self
            .first
            .checked_add(1)
            .and_then(|n| n.checked_add(blocks.checked_mul(12)?))
            .ok_or(Error::Bounds)?;
        if self.next != expected {
            return Err(Error::Binding);
        }
        Ok(())
    }
    fn finish(&mut self) -> Result<(usize, usize), Error> {
        self.geometry.word(
            self.input,
            self.first,
            self.geometry.rounds,
            ArgmaxField::Indices,
            0,
        )
    }
}

fn hash_argmax_event(hash: &mut Sha256, event: ArgmaxEvent<'_>) -> Result<(), Error> {
    hash_node_header(hash, event.lanes, event.bounds);
    match event.code {
        ArgmaxCode::PublicIndices => {
            hash.update([7]);
            for i in 0..event.lanes {
                hash.update((i as i64).to_le_bytes());
            }
        }
        ArgmaxCode::Gather(gather) => hash_gather(hash, gather.words()),
        ArgmaxCode::Affine {
            left,
            right,
            coefficients,
        } => hash_affine(hash, left, Some(right), coefficients),
        ArgmaxCode::Product { left, right } => hash_product(hash, left, right, None, false, None),
        ArgmaxCode::Tensor { input, program } => hash_tensor(hash, input, program),
    }
    Ok(())
}

fn reshape_order(
    input: &PublishedView,
    output: ViewShape,
    layout: Option<&str>,
) -> Result<(LaneOrder, usize), Error> {
    let (heads, rows, width, to_heads) =
        match (layout, input.logical.dimensions(), output.dimensions()) {
            (
                Some("batch_heads_sequence_feature"),
                [1, input_rows, hidden],
                [1, heads, rows, width],
            ) if input_rows == rows && heads.checked_mul(*width) == Some(*hidden) => {
                (*heads, *rows, *width, true)
            }
            (Some("batch_sequence_hidden"), [1, heads, rows, width], [1, output_rows, hidden])
                if output_rows == rows && heads.checked_mul(*width) == Some(*hidden) =>
            {
                (*heads, *rows, *width, false)
            }
            _ => return Err(Error::Binding),
        };
    if output.lanes()? != input.lanes {
        return Err(Error::Shape);
    }
    if heads == 1 || rows == 1 {
        return Ok((input.order, input.start));
    }
    match input.order {
        LaneOrder::Linear => Ok((
            LaneOrder::HeadsSequence {
                heads,
                rows,
                width,
                to_heads,
                base_start: input.start,
            },
            0,
        )),
        LaneOrder::HeadsSequence {
            heads: old_heads,
            rows: old_rows,
            width: old_width,
            to_heads: old_direction,
            base_start,
        } if old_heads == heads
            && old_rows == rows
            && old_width == width
            && old_direction != to_heads
            && input.start == 0 =>
        {
            Ok((LaneOrder::Linear, base_start))
        }
        // A repartitioned non-inverse chain needs another indexed contract; do
        // not substitute a flat identity or retain a per-lane composition table.
        _ => Err(Error::Binding),
    }
}

fn hash_prepared_loop(
    hash: &mut Sha256,
    input: usize,
    prepared: &PreparedTensorMap,
) -> Result<(), Error> {
    let full = prepared.program_at(0)?;
    let remainder = prepared.lanes() % full.lanes();
    let tail = if remainder == 0 {
        None
    } else {
        Some(prepared.program_at(prepared.lanes() - remainder)?)
    };
    hash_tensor_loop(hash, LoopInput::Slice(input), full, tail);
    Ok(())
}

#[cfg(test)]
#[path = "compact_lowering_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "compact_token_lookup_tests.rs"]
mod token_lookup_tests;

#[cfg(test)]
#[path = "compact_norm_tests.rs"]
mod norm_tests;

#[cfg(test)]
mod constructor_quote_tests {
    use super::*;
    use std::mem::size_of;

    // Independent caller-only inventory. Native Builder/Program/finish slots
    // must come from Construction, never a compiler reconstruction of that path.
    fn native_slots() -> usize {
        size_of::<NativeConstructionPeak>()
            + size_of::<tensor::Construction>()
            + size_of::<tensor::ConstructionQuote>()
            + size_of::<Result<tensor::ConstructionQuote, tensor::ConstructionQuoteError>>()
            + size_of::<tensor::Resources>()
            + size_of::<Result<tensor::Resources, Error>>()
    }

    // Literal caller inventory for these import-only, late-bound fixtures. Keep
    // allocation/history charges AND actual capacities: no retrospective credit.
    // The frontier's existing opaque-map bound is an owner contract, not an
    // invented exact B-tree layout. Borrowed semantic/source backing stays out.
    fn initial_baseline(graph: &CompactGraph<'_, '_>) -> usize {
        let imports = graph.imports;
        assert!(graph.ops.is_empty());
        assert!(graph.prebound_inputs.is_none());
        assert_eq!(graph.contraction_control_bytes.get(), 0);
        assert!(imports.tensors.is_empty());
        assert!(imports.public_norms.is_empty());
        assert!(imports.original_inputs.is_empty());
        let keys = imports
            .nodes
            .iter()
            .map(|node| match &node.code {
                Code::Input(key) | Code::PublicInput { key, .. } => key.capacity(),
                _ => panic!("fixture must retain only original imports"),
            })
            .sum::<usize>();
        imports.allocated
            + size_of::<Builder>()
            + size_of::<Node>() * imports.nodes.capacity()
            + size_of::<Arc<super::super::original_import::InputRecord>>()
                * imports.original_inputs.capacity()
            + keys
            + size_of::<(usize, u128)>() * imports.row_norms.capacity()
            + size_of::<(Words, u128)>() * imports.row_energies.capacity()
            + size_of::<(usize, u64)>() * imports.row_masses.capacity()
            + size_of::<(Words, u128)>() * imports.attention_energies.capacity()
            + size_of::<CompactGraph<'_, '_>>()
            + graph.frontier_bytes
            + size_of::<PublishedOp<'_>>() * graph.ops.capacity()
            + size_of::<BoundImport<'_>>() * graph.bound_imports.capacity()
            + 2 * size_of::<Result<CompactGraph<'_, '_>, Error>>()
            + 2 * size_of::<usize>()
            + 4 * size_of::<PublishedOp<'_>>()
    }

    // Independent lookup-driver slot enumeration. Do not call the production
    // contraction_control quote to manufacture its own expected total. Only the
    // closed sink's concrete borrowed iterator layout comes from its type owner.
    fn lookup_controls(input: &PublishedView) -> usize {
        let original = ContractionView::new(input.clone(), ContractionStage::Input);
        let iterator =
            <CompactContractionSink<'_, '_, '_> as graph::driver::ContractionSink>::tokens(
                &original,
            );
        2 * size_of::<ContractionWords>()
            + 2 * size_of::<ContractionView>()
            + 2 * size_of::<Result<ContractionView, Error>>()
            + 2 * size_of::<Result<ContractionWords, Error>>()
            + 2 * size_of::<Result<(), Error>>()
            + 2 * size_of::<Result<&SourceTensor, Error>>()
            + 2 * size_of::<Result<&usize, Error>>()
            + 2 * std::mem::size_of_val(&iterator)
            + 2 * size_of::<Option<(usize, usize, usize)>>()
            + size_of::<(usize, usize, usize)>()
            + 2 * size_of::<std::iter::StepBy<std::ops::Range<usize>>>()
            + size_of::<Interval>()
            + size_of::<(
                &mut CompactContractionSink<'_, '_, '_>,
                &pllm_models::ModelOperation,
                &[usize],
                &ContractionView,
                &BTreeMap<String, SourceTensor>,
                usize,
            )>()
            + size_of::<CompactContractionSink<'_, '_, '_>>()
            + 2 * size_of::<Result<ContractionResult<'_>, Error>>()
    }

    fn lookup_recipe_inventory(
        vocabulary: usize,
        new_full: bool,
        new_tail: bool,
        baseline: usize,
    ) -> usize {
        let full = new_full.then(|| {
            tensor::Construction::EqualZero {
                lanes: vocabulary.min(128),
            }
            .quote()
            .unwrap()
        });
        let tail = (new_tail && vocabulary > 128 && vocabulary % 128 != 0).then(|| {
            tensor::Construction::EqualZero {
                lanes: vocabulary % 128,
            }
            .quote()
            .unwrap()
        });
        let retained = full.map_or(0, |quote| quote.immutable_template_bytes);
        let peak = full
            .map_or(0, |quote| quote.new_peak_bytes)
            .max(tail.map_or(0, |quote| retained + quote.new_peak_bytes));
        let slots = if full.is_some() || tail.is_some() {
            native_slots()
        } else {
            size_of::<NativeConstructionPeak>()
        };
        baseline
            + peak
            + slots
            + 2 * size_of::<Result<CompactTokenLookup, Error>>()
            + 2 * size_of::<Result<EqualityProgram, Error>>()
    }

    // Native cost/query storage belongs to core; independently enumerate only
    // the compiler's Count state, owner handles, arguments and Result transitions.
    fn argmax_count_inventory(geometry: ArgmaxGeometry) -> usize {
        (if geometry.blocks == 0 {
            0
        } else {
            tensor::Program::negative_work_resources().known_peak_bytes
        }) + 2 * size_of::<Result<ArgmaxNumeric<'_, '_>, Error>>()
            + size_of::<NegativePrograms<'_>>()
            + 2 * size_of::<ArgmaxGeometry>()
            + size_of::<[Value; 16]>()
            + size_of::<[usize; 6]>()
            + 4 * size_of::<Result<Value, Error>>()
            + 2 * size_of::<Result<(usize, usize), Error>>()
            + 3 * size_of::<Interval>()
            + 2 * size_of::<Result<Interval, Error>>()
            + 6 * size_of::<WorkBudget>()
            + 4 * size_of::<Result<WorkBudget, Error>>()
            + 2 * size_of::<Result<puma::prss::PrssLimits, Error>>()
            + 2 * size_of::<tensor::NegativeWorkResources>()
            + size_of::<Result<(), Error>>()
    }

    fn argmax_recipe_inventory(lanes: usize, baseline: usize) -> usize {
        let geometry = ArgmaxGeometry::new(lanes).unwrap();
        let (mut retained, mut native_peak, mut kernels) = (0_usize, 0_usize, 0_usize);
        let mut seen = [false; 129];
        let mut length = lanes;
        while length > 1 {
            for pairs in [if length >= 256 { 128 } else { 0 }, (length % 256) / 2] {
                if pairs != 0 && !seen[pairs] {
                    let quote = tensor::Construction::Negative { lanes: pairs }
                        .quote()
                        .unwrap();
                    native_peak = native_peak.max(retained + quote.new_peak_bytes);
                    retained += quote.immutable_template_bytes;
                    kernels += 1;
                    seen[pairs] = true;
                }
            }
            length = length.div_ceil(2);
        }
        assert_eq!(kernels, geometry.kernels);
        let slots = if kernels == 0 {
            size_of::<NativeConstructionPeak>()
        } else {
            native_slots()
        };
        let common = baseline
            + 2 * size_of::<Result<CompactArgmax, Error>>()
            + size_of::<ArgmaxNumeric<'_, '_>>()
            + size_of::<Result<Value, Error>>()
            + size_of::<ArgmaxEvent<'_>>()
            + size_of::<WorkBudget>();
        // Counting ends before native Vec/Program construction. Max, not sum.
        (common + size_of::<NegativeProgram>() * kernels + native_peak + slots)
            .max(common + argmax_count_inventory(geometry))
    }

    #[test]
    fn native_predicate_inventory_matches_owner_quotes_and_actual_templates() {
        let bounds = Interval::new(-257, 257).unwrap();
        for lanes in [1, 3, 127, 128] {
            let negative = tensor::Construction::Negative { lanes }.quote().unwrap();
            let equality = tensor::Construction::EqualZero { lanes }.quote().unwrap();
            let shifts = size_of::<[u8; 6]>() + size_of::<std::array::IntoIter<u8, 6>>();
            assert_eq!(negative.constructor_buffer_bytes, 0);
            assert_eq!(equality.constructor_buffer_bytes, shifts);
            assert_eq!(equality.new_peak_bytes - negative.new_peak_bytes, shifts);
            for (construction, program, quote) in [
                (
                    tensor::Construction::Negative { lanes },
                    tensor::Program::negative(lanes, bounds).unwrap(),
                    negative,
                ),
                (
                    tensor::Construction::EqualZero { lanes },
                    tensor::Program::equal_zero(lanes, bounds).unwrap(),
                    equality,
                ),
            ] {
                let resources = program.resources().unwrap();
                assert_eq!(
                    resources.immutable_template_bytes,
                    quote.immutable_template_bytes
                );
                assert!(resources.shared_admitted_bytes <= puma::MAX_ADMITTED_BYTES);
                assert_eq!(program.lanes(), lanes);
                assert_eq!(program.output_lanes(), lanes);
                assert_eq!(program.output_bounds(), Interval::new(0, 1).unwrap());
                assert!(program.validity_lanes().is_none());
                let mut inventory = NativeConstructionPeak::default();
                inventory.include(construction).unwrap();
                assert_eq!(inventory.retained, resources.immutable_template_bytes);
                assert_eq!(inventory.peak, quote.new_peak_bytes);
                assert_eq!(
                    inventory.bytes().unwrap(),
                    quote.new_peak_bytes + native_slots()
                );
            }
        }
        for lanes in [0, 129, usize::MAX] {
            assert!(matches!(
                native_construction_quote(tensor::Construction::Negative { lanes }),
                Err(Error::Bounds)
            ));
            assert!(matches!(
                native_construction_quote(tensor::Construction::EqualZero { lanes }),
                Err(Error::Bounds)
            ));
        }
    }

    struct Fixture {
        full: DecoderGraph,
        region: DecoderGraph,
        sources: BTreeMap<String, SourceTensor>,
        norms: SourceNorms,
        imports: Builder,
        input: View,
    }

    impl Fixture {
        fn new(vocabulary: usize, selection: bool, repeat: bool) -> Self {
            let config = serde_json::to_vec(&serde_json::json!({
                "model_type": "llama", "hidden_size": 2, "intermediate_size": 2,
                "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1,
                "head_dim": 2, "vocab_size": vocabulary, "tie_word_embeddings": false,
                "hidden_act": "silu", "attention_bias": false, "max_position_embeddings": 64,
                "rope_theta": 10000, "rms_norm_eps": 0.000001, "attention_dropout": 0.0,
                "use_sliding_window": false, "use_cache": true, "max_window_layers": 1
            }))
            .unwrap();
            let full = lower_with_weights(&config, 3, 1, WeightMode::Public)
                .unwrap()
                .prefill;
            let operator = if selection {
                Op::GreedyTokenSelection
            } else {
                Op::TokenLookup
            };
            let mut operation = full
                .operations
                .iter()
                .find(|op| op.operator == operator)
                .unwrap()
                .clone();
            if selection {
                operation.inputs = vec!["scores".into()];
            }
            let mut region = full.clone();
            region.operations = vec![operation.clone()];
            if repeat {
                operation.id.push_str(".again");
                region.operations.push(operation.clone());
            }
            region.output = operation.id;
            let sources: BTreeMap<_, _> =
                Program::source_shapes_with_weights(&config, 3, 1, WeightMode::Public)
                    .unwrap()
                    .into_iter()
                    .map(|(key, shape)| {
                        let public = key.contains("norm");
                        let mut hash = Sha256::new();
                        for value in std::iter::repeat_n(
                            if public { ONE } else { ONE / 64 },
                            shape.iter().product(),
                        ) {
                            hash.update(value.to_le_bytes());
                        }
                        let source = SourceTensor {
                            public_q18: public.then(|| vec![ONE; shape.iter().product()]),
                            shape,
                            lower: if public { ONE } else { -ONE / 64 },
                            upper: if public { ONE } else { ONE / 64 },
                            digest: hash.finalize().into(),
                            row_l1: None,
                            row_squared_norm: None,
                            row_l1_squared_sum: None,
                        };
                        (key, source)
                    })
                    .collect();
            let mut imports = Builder {
                public_weights: true,
                ..Builder::default()
            };
            for (key, source) in sources
                .iter()
                .filter(|(_, source)| source.public_q18.is_none())
            {
                imports
                    .push(
                        Code::PublicInput {
                            key: key.clone(),
                            digest: source.digest,
                            row_l1: None,
                            row_squared_norm: None,
                            row_l1_squared_sum: None,
                        },
                        source.shape.iter().product(),
                        Interval::new(source.lower, source.upper).unwrap(),
                    )
                    .unwrap();
            }
            let lanes = if selection { vocabulary } else { 3 };
            let bounds = if selection {
                Interval::new(-ONE / 4, ONE / 4).unwrap()
            } else {
                Interval::new(0, vocabulary as i64 - 1).unwrap()
            };
            let owner = imports
                .push(
                    Code::Input(region.operations[0].inputs[0].clone()),
                    lanes,
                    bounds,
                )
                .unwrap();
            let input = imports.view(owner, vec![1, lanes]);
            Self {
                full,
                region,
                sources,
                norms: SourceNorms::new(),
                imports,
                input,
            }
        }

        fn graph(&self, outside: usize) -> CompactGraph<'_, '_> {
            let shapes = self
                .sources
                .iter()
                .map(|(key, source)| (key.clone(), source.shape.clone()))
                .collect();
            let validated = validate_source_records(
                &self.full,
                &shapes,
                &self.sources,
                WeightMode::Public,
                ExecutionPolicy::default(),
                &self.norms,
            )
            .unwrap();
            let mut graph = CompactGraph::with_owner_baseline(
                &self.imports,
                validated,
                &self.region,
                0,
                outside,
            )
            .unwrap();
            graph
                .bind_import(&self.region.operations[0].inputs[0], &self.input)
                .unwrap();
            graph
        }
    }

    #[test]
    fn argmax_quote_matches_actual_distinct_pair_templates_and_original_arc_interning() {
        for lanes in [1, 2, 3, 127, 128, 129, 255, 256, 257, 258, 513, 1025] {
            let fixture = Fixture::new(lanes, true, true);
            let mut graph = fixture.graph(37);
            let baseline = initial_baseline(&graph);
            assert_eq!(graph.baseline().unwrap(), baseline);
            let output = graph.lower_next().unwrap();
            let PublishedKind::Argmax(plan) = &graph.ops[0].kind else {
                panic!("argmax")
            };
            let (mut retained, mut native_peak) = (0, 0usize);
            for native in &plan.programs {
                let quote = tensor::Construction::Negative {
                    lanes: native.lanes,
                }
                .quote()
                .unwrap();
                let original = tensor::Program::negative(native.lanes, native.bounds).unwrap();
                assert_eq!(native.program.digest(), original.digest());
                assert_eq!(
                    native.program.resources().unwrap().immutable_template_bytes,
                    quote.immutable_template_bytes
                );
                native_peak = native_peak.max(retained + quote.new_peak_bytes);
                retained += quote.immutable_template_bytes;
            }
            let native_bytes = native_peak
                + if plan.programs.is_empty() {
                    size_of::<NativeConstructionPeak>()
                } else {
                    native_slots()
                };
            let common = baseline
                + 2 * size_of::<Result<CompactArgmax, Error>>()
                + size_of::<ArgmaxNumeric<'_, '_>>()
                + size_of::<Result<Value, Error>>()
                + size_of::<ArgmaxEvent<'_>>()
                + size_of::<WorkBudget>();
            let construction =
                common + size_of::<NegativeProgram>() * plan.geometry.kernels + native_bytes;
            let counting = common + argmax_count_inventory(plan.geometry);
            let expected = construction.max(counting);
            assert_eq!(CompactArgmax::quote(lanes, baseline).unwrap().1, expected);
            assert_eq!(plan.peak, 37 + expected);
            assert_eq!(plan.programs.len(), plan.geometry.kernels);
            let arcs: Vec<_> = plan
                .programs
                .iter()
                .map(|native| native.program.clone())
                .collect();
            let work = graph.work_budget();
            let nodes = plan.geometry.nodes().unwrap();
            let attempts = ARGMAX_NATIVE_ATTEMPTS.with(|count| count.get());
            let second = graph.lower_next().unwrap();
            let PublishedKind::Argmax(plan) = &graph.ops[1].kind else {
                panic!("argmax")
            };
            // Original Builder constructs then interns. Do not skip a factory or
            // replace an already-owned Arc merely because its digest is known.
            assert_eq!(
                ARGMAX_NATIVE_ATTEMPTS.with(|count| count.get()) - attempts,
                arcs.len()
            );
            for (old, native) in arcs.iter().zip(&plan.programs) {
                assert!(Arc::ptr_eq(old, &native.program));
                assert!(!native.owns);
            }
            assert_eq!(graph.owned_native_programs().count(), arcs.len());
            assert_eq!(graph.work_budget(), work.checked_mul(2).unwrap());
            assert_eq!(second.word(0).unwrap().0 - output.word(0).unwrap().0, nodes);
        }
    }

    #[test]
    fn lookup_quote_matches_actual_full_tail_templates_work_and_reused_arcs() {
        for vocabulary in [1, 127, 128, 129, 255, 256, 257] {
            let fixture = Fixture::new(vocabulary, false, true);
            let mut graph = fixture.graph(53);
            let baseline = initial_baseline(&graph);
            assert_eq!(graph.baseline().unwrap(), baseline);
            let input = graph
                .frontier
                .value(&fixture.region.operations[0].inputs[0])
                .unwrap();
            let controls = lookup_controls(input);
            let output = graph.lower_next().unwrap();
            let PublishedKind::TokenLookup { plan, .. } = &graph.ops[0].kind else {
                panic!("lookup")
            };
            let (mut retained, mut native_peak) = (0, 0usize);
            for native in plan.programs() {
                let lanes = native.program.lanes();
                let quote = tensor::Construction::EqualZero { lanes }.quote().unwrap();
                let original = tensor::Program::equal_zero(lanes, native.bounds).unwrap();
                assert_eq!(native.program.digest(), original.digest());
                assert_eq!(
                    native.program.resources().unwrap().immutable_template_bytes,
                    quote.immutable_template_bytes
                );
                native_peak = native_peak.max(retained + quote.new_peak_bytes);
                retained += quote.immutable_template_bytes;
            }
            let assembly = 2 * size_of::<Result<CompactTokenLookup, Error>>()
                + 2 * size_of::<Result<EqualityProgram, Error>>();
            let expected = baseline + native_peak + native_slots() + assembly;
            assert_eq!(
                CompactTokenLookup::quote(vocabulary, true, true, baseline).unwrap(),
                expected
            );
            // Either original owner can already exist independently. A borrowed
            // full owner stays in baseline; only a new tail gets construction
            // scratch (and conversely). No phantom tail for exact multiples.
            for new_full in [false, true] {
                for new_tail in [false, true] {
                    let full = new_full.then(|| {
                        tensor::Construction::EqualZero {
                            lanes: vocabulary.min(128),
                        }
                        .quote()
                        .unwrap()
                    });
                    let tail = (new_tail && vocabulary > 128 && vocabulary % 128 != 0).then(|| {
                        tensor::Construction::EqualZero {
                            lanes: vocabulary % 128,
                        }
                        .quote()
                        .unwrap()
                    });
                    let retained_full = full.map_or(0, |quote| quote.immutable_template_bytes);
                    let phase = full
                        .map_or(0, |quote| quote.new_peak_bytes)
                        .max(tail.map_or(0, |quote| retained_full + quote.new_peak_bytes));
                    let slots = if full.is_some() || tail.is_some() {
                        native_slots()
                    } else {
                        size_of::<NativeConstructionPeak>()
                    };
                    assert_eq!(
                        CompactTokenLookup::quote(vocabulary, new_full, new_tail, baseline)
                            .unwrap(),
                        baseline + phase + slots + assembly
                    );
                }
            }
            // The isolated kernel quote above excludes its caller. Shared
            // ContractionSink/driver/return controls remain live during lookup.
            assert_eq!(
                graph.construction_peak_bytes(),
                53 + controls + expected,
                "vocabulary={vocabulary}, live lookup controls={controls}"
            );
            assert_eq!(graph.contraction_control_bytes.get(), 0);
            assert_eq!(plan.baseline().unwrap(), retained);
            assert_eq!(plan.full.program.lanes(), vocabulary.min(128));
            assert_eq!(
                plan.tail.as_ref().map(|tail| tail.program.lanes()),
                (vocabulary > 128 && vocabulary % 128 != 0).then_some(vocabulary % 128)
            );
            let arcs: Vec<_> = plan
                .programs()
                .map(|native| native.program.clone())
                .collect();
            let work = plan
                .full
                .program
                .work_budget()
                .checked_mul((vocabulary / vocabulary.min(128)) as u64)
                .unwrap()
                .checked_add(
                    plan.tail
                        .as_ref()
                        .map_or(WorkBudget::default(), |tail| tail.program.work_budget()),
                )
                .unwrap()
                .checked_mul(3)
                .unwrap();
            assert_eq!(graph.work_budget(), work);
            assert_eq!(
                output.word(0).unwrap(),
                (fixture.imports.nodes.len() + 2, 0)
            );
            let baseline = graph.baseline().unwrap();
            assert_eq!(
                CompactTokenLookup::quote(vocabulary, false, false, baseline).unwrap(),
                baseline + size_of::<NativeConstructionPeak>() + assembly
            );
            let attempts = LOOKUP_NATIVE_ATTEMPTS.with(|count| count.get());
            let second = graph.lower_next().unwrap();
            assert_eq!(LOOKUP_NATIVE_ATTEMPTS.with(|count| count.get()), attempts);
            let PublishedKind::TokenLookup { plan, .. } = &graph.ops[1].kind else {
                panic!("lookup")
            };
            for (old, native) in arcs.iter().zip(plan.programs()) {
                assert!(Arc::ptr_eq(old, &native.program));
                assert!(!native.owns);
            }
            assert_eq!(graph.owned_native_programs().count(), arcs.len());
            assert_eq!(graph.work_budget(), work.checked_mul(2).unwrap());
            assert_eq!(second.word(0).unwrap().0 - output.word(0).unwrap().0, 9);
        }
    }

    fn unchanged_state(graph: &CompactGraph<'_, '_>) -> ([usize; 12], WorkBudget, [u8; 32]) {
        let mut hash = Sha256::new();
        graph.hash_expanded_nodes(&mut hash).unwrap();
        (
            [
                graph.next,
                graph.cursor,
                graph.ops.len(),
                graph.ops.capacity(),
                graph.bound_imports.len(),
                graph.bound_imports.capacity(),
                graph.owned_native_programs().count(),
                graph.imports.nodes.len(),
                graph.imports.nodes.capacity(),
                graph.imports.allocated,
                graph.imports.tensors.len(),
                graph.outside_owner_bytes(),
            ],
            graph.work_budget(),
            hash.finalize().into(),
        )
    }

    #[test]
    fn predicate_construction_exact_cap_and_one_short_preserve_unpublished_state() {
        for (vocabulary, selection) in [
            (1, true),
            (3, true),
            (257, true),
            (1, false),
            (128, false),
            (129, false),
            (257, false),
        ] {
            let fixture = Fixture::new(vocabulary, selection, false);
            let probe = fixture.graph(0);
            let baseline = initial_baseline(&probe);
            assert_eq!(probe.baseline().unwrap(), baseline);
            let input = probe
                .frontier
                .value(&fixture.region.operations[0].inputs[0])
                .unwrap();
            let recipe = if selection {
                argmax_recipe_inventory(vocabulary, baseline)
            } else {
                lookup_recipe_inventory(vocabulary, true, true, baseline)
            };
            let local = recipe + if selection { 0 } else { lookup_controls(input) };
            assert!(local >= probe.construction_peak_bytes());
            drop(probe);
            let outside = MAX_PROGRAM_BYTES.checked_sub(local).unwrap();
            let mut exact = fixture.graph(outside);
            exact.lower_next().unwrap();
            assert_eq!(exact.limit, MAX_PROGRAM_BYTES);
            assert_eq!(exact.construction_peak_bytes(), MAX_PROGRAM_BYTES);
            assert_eq!(exact.outside_owner_bytes(), outside);
            drop(exact);

            // Direct lower() does NOT own the wrapper's contraction controls.
            // Give that narrower recipe its own independently enumerated cap;
            // using the whole-operation cap leaves those bytes as spare room.
            let direct_outside = MAX_PROGRAM_BYTES.checked_sub(recipe).unwrap();
            let mut short = fixture.graph(direct_outside);
            short.limit = MAX_PROGRAM_BYTES - 1;
            let state = unchanged_state(&short);
            let attempts = (
                ARGMAX_NATIVE_ATTEMPTS.with(|count| count.get()),
                LOOKUP_NATIVE_ATTEMPTS.with(|count| count.get()),
            );
            let input_id = &fixture.region.operations[0].inputs[0];
            let input = short.frontier.value(input_id).unwrap().clone();
            let peak = short.construction_peak_bytes();
            // Direct recipe gate must not even reserve argmax metadata or update
            // admission history when its complete native envelope does not fit.
            let rejected = if selection {
                CompactArgmax::lower(&short, &input, short.next).map(|_| ())
            } else {
                let key = fixture.region.operations[0].attributes["weight"]
                    .as_str()
                    .unwrap();
                CompactTokenLookup::lower(&short, &input, vocabulary, 2, &fixture.sources[key])
                    .map(|_| ())
            };
            assert!(
                matches!(rejected, Err(Error::Bounds)),
                "direct recipe: selection={selection}, vocabulary={vocabulary}, envelope={recipe}"
            );
            assert_eq!(short.construction_peak_bytes(), peak);
            assert_eq!(unchanged_state(&short), state);
            assert!(!short.failed);
            assert_eq!(
                attempts,
                (
                    ARGMAX_NATIVE_ATTEMPTS.with(|count| count.get()),
                    LOOKUP_NATIVE_ATTEMPTS.with(|count| count.get()),
                )
            );
            drop(input);
            drop(short);
            // Separately test the public wrapper at ITS exact one-short cap,
            // not the direct recipe's larger outside-owner baseline.
            let mut short = fixture.graph(outside);
            short.limit = MAX_PROGRAM_BYTES - 1;
            let state = unchanged_state(&short);
            let peak = short.construction_peak_bytes();
            let input = short.frontier.value(input_id).unwrap().clone();
            // Admitted transient controls may advance peak diagnostics before
            // the native phase refuses; published owners/work stay unchanged.
            assert!(
                matches!(short.lower_next(), Err(Error::Bounds)),
                "whole operation: selection={selection}, vocabulary={vocabulary}, envelope={local}"
            );
            assert_eq!(unchanged_state(&short), state);
            assert!(short.construction_peak_bytes() >= peak);
            assert!(short.construction_peak_bytes() <= short.limit);
            assert_eq!(
                attempts,
                (
                    ARGMAX_NATIVE_ATTEMPTS.with(|count| count.get()),
                    LOOKUP_NATIVE_ATTEMPTS.with(|count| count.get())
                )
            );
            assert!(short.failed);
            assert_eq!(short.failure_site().unwrap().next_virtual_owner, short.next);
            assert_eq!(
                short.frontier.value(input_id).unwrap().word(0).unwrap(),
                input.word(0).unwrap()
            );
            assert!(short.frontier.value(&fixture.region.output).is_none());
            assert!(matches!(short.lower_next(), Err(Error::Retired)));
        }
    }

    #[test]
    fn rejected_second_recipe_preserves_existing_native_arc_owners_and_work() {
        for selection in [false, true] {
            let fixture = Fixture::new(257, selection, true);
            let mut graph = fixture.graph(71);
            graph.lower_next().unwrap();
            let old: Vec<_> = graph.owned_native_programs().cloned().collect();
            let baseline = graph.baseline().unwrap();
            let local = if selection {
                CompactArgmax::quote(257, baseline).unwrap().1
            } else {
                CompactTokenLookup::quote(257, false, false, baseline).unwrap()
            };
            graph.limit = graph.outside_owner_bytes() + local - 1;
            let state = unchanged_state(&graph);
            let attempts = (
                ARGMAX_NATIVE_ATTEMPTS.with(|count| count.get()),
                LOOKUP_NATIVE_ATTEMPTS.with(|count| count.get()),
            );
            assert!(matches!(graph.lower_next(), Err(Error::Bounds)));
            assert_eq!(unchanged_state(&graph), state);
            for (old, retained) in old.iter().zip(graph.owned_native_programs()) {
                assert!(Arc::ptr_eq(old, retained));
            }
            assert_eq!(
                attempts,
                (
                    ARGMAX_NATIVE_ATTEMPTS.with(|count| count.get()),
                    LOOKUP_NATIVE_ATTEMPTS.with(|count| count.get())
                )
            );
            assert_eq!(graph.failure_site().unwrap().step, 1);
        }
        assert!(matches!(
            CompactArgmax::quote(257, usize::MAX),
            Err(Error::Bounds)
        ));
        assert!(matches!(
            CompactTokenLookup::quote(129, true, true, usize::MAX),
            Err(Error::Bounds)
        ));
    }
}
