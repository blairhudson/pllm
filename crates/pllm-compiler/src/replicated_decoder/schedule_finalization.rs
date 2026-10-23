//! Borrowed original-schedule finalization, with preconstruction admission.
//!
//! Declared test-only INSIDE `program` to reuse its canonical encoding helpers.
//! Default production routing is unchanged. Numeric lowering and original
//! Arc owners remain untouched; no dependency Vec or serialized hash buffer is
//! constructed. This helper admits ONLY its new finalization owners, added to an
//! enclosing compiler's actual retained baseline. It is not request admission.
//!
//! The identity prefix is an already initialized public SHA state. Its original
//! source/policy/config serialization must be preadmitted by the parent, before
//! allocating those buffers. This helper neither reconstructs that identity nor
//! changes source_binding. Caller retains nodes, selected Vec/capacity, prefix,
//! source files/certificates and all other outside owners in its baseline.
//!
//! Integration gate: Builder::allocated is NOT that baseline. Its logical
//! charges do not census all live graph/source/view/map/vector capacities or
//! closure captures. Production integration must admit those owners, Builder,
//! caller Program/Result slots, and prefix serialization peak BEFORE identity(),
//! then add this helper's last/count/retire/hash/result overlap before finalize().
//! Preserve original complete_schedule_work/complete_work_limits diagnostic
//! ordering separately: this helper intentionally rejects work limits earlier.
//! Until that parent contract exists, compare_original is a test-only seam and
//! the duplicated original finish loop must remain the production authority.
use super::*;
use std::mem::size_of;

/// Immutable logical lane references. Compact owners may compute coordinates on
/// demand; no expanded Words/Vec is required. Count, coordinates and errors must
/// remain stable for the entire schedule borrow, including quote/finalize passes.
/// This lends coordinates only, never work, canonical bytes or request authority.
pub(super) trait BorrowedLaneRefs {
    fn len(&self) -> usize;
    fn lane_ref(&self, index: usize) -> Result<(usize, usize), Error>;
}

impl BorrowedLaneRefs for Words {
    fn len(&self) -> usize {
        self.len()
    }
    fn lane_ref(&self, index: usize) -> Result<(usize, usize), Error> {
        if index >= self.len() {
            return Err(Error::Binding);
        }
        Ok(self.at(index))
    }
}

mod public_constants {
    use super::*;

    // No user callback, arbitrary generator, digest or work supplier. Only
    // compiler-owned recipe owners can back these deterministic public values.
    trait Owner {}
    impl Owner for compact_lowering::frozen_region::FrozenRegion<'_> {}

    #[derive(Clone, Copy)]
    enum Sequence {
        Iota { lanes: usize },
    }

    /// Sealed public-constant recipe loan. The original argmax compiler derives
    /// both its range and identity; there is no newly fabricated source tensor
    /// or public-input owner, and no expanded i64 array is retained.
    #[derive(Clone, Copy)]
    pub(in super::super) struct GeneratedPublicValues<'a> {
        _owner: &'a dyn Owner,
        sequence: Sequence,
    }

    impl<'a> GeneratedPublicValues<'a> {
        pub(in super::super) fn argmax_indices(
            owner: &'a compact_lowering::frozen_region::FrozenRegion<'_>,
            descriptor: usize,
        ) -> Result<Self, Error> {
            let census = owner
                .descriptor_census()
                .nth(descriptor)
                .ok_or(Error::Binding)?;
            let compact_lowering::DescriptorKind::Argmax { lanes, .. } = census.kind else {
                return Err(Error::Binding);
            };
            let values = Self {
                _owner: owner,
                sequence: Sequence::Iota { lanes },
            };
            values.validate(
                lanes,
                Interval::new(0, i64::try_from(lanes).map_err(|_| Error::Bounds)? - 1)?,
            )?;
            Ok(values)
        }

        fn len(self) -> usize {
            match self.sequence {
                Sequence::Iota { lanes } => lanes,
            }
        }

        /// Read-only execution loan from this exact original frozen owner. No
        /// caller sequence, source declaration, or replacement owner is accepted.
        pub(in super::super) fn checked_len(
            self,
            owner: &compact_lowering::frozen_region::FrozenRegion<'_>,
            lanes: usize,
            bounds: Interval,
        ) -> Result<usize, Error> {
            if !std::ptr::addr_eq(self._owner, owner as &dyn Owner) {
                return Err(Error::Binding);
            }
            self.validate(lanes, bounds)?;
            Ok(self.len())
        }

        pub(in super::super) fn checked_element(
            self,
            owner: &compact_lowering::frozen_region::FrozenRegion<'_>,
            lanes: usize,
            bounds: Interval,
            index: usize,
        ) -> Result<i64, Error> {
            let len = self.checked_len(owner, lanes, bounds)?;
            if index >= len {
                return Err(Error::Shape);
            }
            match self.sequence {
                Sequence::Iota { .. } => i64::try_from(index).map_err(|_| Error::Bounds),
            }
        }

        pub(super) fn validate(self, lanes: usize, bounds: Interval) -> Result<(), Error> {
            let n = self.len();
            if n == 0 || n > MAX_VALUES || lanes != n {
                return Err(Error::Shape);
            }
            let last = i64::try_from(n - 1).map_err(|_| Error::Bounds)?;
            if bounds != Interval::new(0, last)? {
                return Err(Error::Shape);
            }
            Ok(())
        }

        // Only called after validate by quote/finalize. Bounds make this cast
        // exact, and the public sequence remains stable throughout its loan.
        pub(super) fn values(self) -> impl Iterator<Item = i64> {
            (0..self.len()).map(|index| index as i64)
        }
    }

    #[cfg(test)]
    pub(super) fn invalid_geometry_fixture<'a>(
        owner: &'a compact_lowering::frozen_region::FrozenRegion<'_>,
        lanes: usize,
    ) -> GeneratedPublicValues<'a> {
        // Negative geometry corruption only; even here retain a genuine source
        // owner rather than manufacturing a new provenance implementation.
        GeneratedPublicValues {
            _owner: owner,
            sequence: Sequence::Iota { lanes },
        }
    }
}

pub(super) use public_constants::GeneratedPublicValues;

/// Closed original recipe vocabulary. All programs/plans/proofs remain borrowed
/// native owners, not supplied digests or callbacks. This is finalization metadata
/// only: it does not validate numeric lowering, source provenance or execution.
#[derive(Clone, Copy)]
pub(super) enum BorrowedRecipe<'a> {
    Input(&'a str),
    PublicInput {
        key: &'a str,
        digest: &'a [u8; 32],
    },
    PublicValues(&'a [i64]),
    GeneratedPublicValues(GeneratedPublicValues<'a>),
    Gather(&'a dyn BorrowedLaneRefs),
    SourceGather(SourceGather),
    Affine {
        left: usize,
        right: Option<usize>,
        coefficients: Affine,
    },
    Product {
        left: usize,
        right: usize,
        shape: Option<MatrixShape>,
        rescale: bool,
        norm: Option<&'a puma::multiply::norm::RowEnergyProof>,
    },
    PublicProduct {
        input: usize,
        weight: usize,
        rows: usize,
        inner: usize,
        columns: usize,
        transposed: bool,
    },
    Tensor {
        input: usize,
        program: &'a Arc<tensor::Program>,
    },
    TensorLoop {
        input: LoopInput,
        program: &'a Arc<tensor::Program>,
        tail: Option<&'a Arc<tensor::Program>>,
    },
    CheckedRange {
        input: usize,
        program: &'a Arc<tensor::Program>,
    },
    PublicLookup {
        input: usize,
        weight: usize,
        plan: &'a runtime::lookup::Plan,
    },
    TrieLookup {
        input: usize,
        weight: usize,
        plan: &'a runtime::trie::Plan,
    },
    Selection {
        input: usize,
        plan: &'a runtime::selection::Plan,
    },
}

#[derive(Clone, Copy)]
pub(super) struct BorrowedNode<'a> {
    pub lanes: usize,
    pub bounds: Interval,
    pub recipe: BorrowedRecipe<'a>,
}

impl<'a> From<&'a Node> for BorrowedNode<'a> {
    fn from(node: &'a Node) -> Self {
        let recipe = match &node.code {
            Code::Input(key) => BorrowedRecipe::Input(key),
            Code::PublicInput { key, digest, .. } => BorrowedRecipe::PublicInput { key, digest },
            Code::PublicValues(values) => BorrowedRecipe::PublicValues(values),
            Code::Gather(words) => BorrowedRecipe::Gather(words),
            Code::SourceGather(view) => BorrowedRecipe::SourceGather(*view),
            Code::Affine {
                left,
                right,
                coefficients,
            } => BorrowedRecipe::Affine {
                left: *left,
                right: *right,
                coefficients: *coefficients,
            },
            Code::Product {
                left,
                right,
                shape,
                rescale,
                norm,
            } => BorrowedRecipe::Product {
                left: *left,
                right: *right,
                shape: *shape,
                rescale: *rescale,
                norm: norm.as_deref(),
            },
            Code::PublicProduct {
                input,
                weight,
                rows,
                inner,
                columns,
                transposed,
            } => BorrowedRecipe::PublicProduct {
                input: *input,
                weight: *weight,
                rows: *rows,
                inner: *inner,
                columns: *columns,
                transposed: *transposed,
            },
            Code::Tensor { input, program } => BorrowedRecipe::Tensor {
                input: *input,
                program,
            },
            Code::TensorLoop {
                input,
                program,
                tail,
            } => BorrowedRecipe::TensorLoop {
                input: *input,
                program,
                tail: tail.as_ref(),
            },
            Code::CheckedRange { input, program } => BorrowedRecipe::CheckedRange {
                input: *input,
                program,
            },
            Code::PublicLookup {
                input,
                weight,
                plan,
            } => BorrowedRecipe::PublicLookup {
                input: *input,
                weight: *weight,
                plan,
            },
            Code::TrieLookup {
                input,
                weight,
                plan,
            } => BorrowedRecipe::TrieLookup {
                input: *input,
                weight: *weight,
                plan,
            },
            Code::Selection { input, plan } => BorrowedRecipe::Selection {
                input: *input,
                plan,
            },
        };
        Self {
            lanes: node.lanes,
            bounds: node.bounds,
            recipe,
        }
    }
}

impl BorrowedNode<'_> {
    /// Same primitive work rules as Node::work_budget; no caller-supplied totals.
    /// Dependency validation checks full/tail geometry before the division.
    fn work_budget(self) -> Result<WorkBudget, Error> {
        match self.recipe {
            BorrowedRecipe::Product { rescale, .. } => WorkBudget::product(self.lanes, rescale),
            BorrowedRecipe::Tensor { program, .. }
            | BorrowedRecipe::CheckedRange { program, .. } => Ok(program.work_budget()),
            BorrowedRecipe::PublicLookup { plan, .. } => plan.work_budget(),
            BorrowedRecipe::TrieLookup { plan, .. } => Ok(plan.work_budget()),
            BorrowedRecipe::Selection { plan, .. } => Ok(plan.work_budget()),
            BorrowedRecipe::TensorLoop { program, tail, .. } => program
                .work_budget()
                .checked_mul((self.lanes / program.lanes()) as u64)?
                .checked_add(tail.map_or(WorkBudget::default(), |p| p.work_budget())),
            _ => Ok(WorkBudget::default()),
        }
    }
}

/// Compiler implementations keep count, ordinal mapping and borrowed recipes
/// stable. Expanded slices lend original owners unchanged; retained compact
/// recipes can return the same view without constructing Node or lane arrays.
pub(super) trait BorrowedOrderedNodes {
    fn node_count(&self) -> usize;
    fn node(&self, ordinal: usize) -> Result<BorrowedNode<'_>, Error>;
}

impl BorrowedOrderedNodes for [Node] {
    fn node_count(&self) -> usize {
        self.len()
    }
    fn node(&self, ordinal: usize) -> Result<BorrowedNode<'_>, Error> {
        self.get(ordinal)
            .map(BorrowedNode::from)
            .ok_or(Error::Binding)
    }
}

/// Backing ownership is transferred into Program by its original compiler.
/// Selected coordinates remain borrowed inputs here, never copied or rebound.
pub(super) struct FinalizedSchedule {
    pub digest: [u8; 32],
    pub work: WorkBudget,
    pub retire: Vec<Vec<usize>>,
}

struct Workspace {
    last: Vec<usize>,
    counts: Vec<usize>,
    retire: Vec<Vec<usize>>,
    hash: Sha256,
}

/// Native logical capacities/inline overlap, not allocator headers or RSS.
/// Each retirement bucket reserves its EXACT count before push; no growth
/// factor, empirical reserve or MAX_NODES-sized speculative scratch is used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FinalizationQuote {
    pub retained_baseline_bytes: usize,
    pub node_count: usize,
    pub retained_output_owners: usize,
    pub last_use_bytes: usize,
    pub retirement_count_bytes: usize,
    pub retirement_container_bytes: usize,
    pub retirement_entries_bytes: usize,
    pub workspace_inline_bytes: usize,
    pub hash_finish_bytes: usize,
    pub result_slot_bytes: usize,
    pub peak_bytes: usize,
    pub work: WorkBudget,
}

impl FinalizationQuote {
    pub(super) fn admit(&self, cap: usize) -> Result<(), Error> {
        if self.peak_bytes > cap.min(MAX_PROGRAM_BYTES) {
            Err(Error::Bounds)
        } else {
            Ok(())
        }
    }
}

fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Bounds)
}
fn array<T>(n: usize) -> Result<usize, Error> {
    n.checked_mul(size_of::<T>()).ok_or(Error::Bounds)
}

fn storage_quote(
    node_count: usize,
    retained_output_owners: usize,
    retained_baseline_bytes: usize,
    work: WorkBudget,
) -> Result<FinalizationQuote, Error> {
    let retired = node_count
        .checked_sub(retained_output_owners)
        .ok_or(Error::Binding)?;
    let last_use_bytes = array::<usize>(node_count)?;
    let retirement_count_bytes = array::<usize>(node_count)?;
    let retirement_container_bytes = array::<Vec<usize>>(node_count)?;
    let retirement_entries_bytes = array::<usize>(retired)?;
    let workspace_inline_bytes = size_of::<Workspace>();
    // Consuming SHA finalization/result conversion can overlap the workspace
    // state and both GenericArray/[u8;32] return slots. No hash heap buffer.
    let hash_finish_bytes = add(size_of::<Sha256>(), array::<[u8; 32]>(2)?)?;
    let mut result_slot_bytes = 0;
    for bytes in [
        size_of::<FinalizedSchedule>(),
        size_of::<Result<FinalizedSchedule, Error>>(),
        size_of::<FinalizationQuote>(),
        size_of::<Result<FinalizationQuote, Error>>(),
        size_of::<Result<Workspace, Error>>(),
        size_of::<Result<Vec<usize>, Error>>(),
        size_of::<Result<Vec<Vec<usize>>, Error>>(),
        size_of::<Result<(), std::collections::TryReserveError>>(),
        // Current recipe, dependency source and returned views can overlap.
        array::<BorrowedNode<'_>>(4)?,
        size_of::<Result<BorrowedNode<'_>, Error>>(),
        size_of::<Result<(usize, usize), Error>>(),
        size_of::<Option<Error>>(),
        size_of::<Result<WorkBudget, Error>>(),
        size_of::<Result<(), Error>>(),
    ] {
        result_slot_bytes = add(result_slot_bytes, bytes)?;
    }
    let mut peak_bytes = retained_baseline_bytes;
    for bytes in [
        last_use_bytes,
        retirement_count_bytes,
        retirement_container_bytes,
        retirement_entries_bytes,
        workspace_inline_bytes,
        hash_finish_bytes,
        result_slot_bytes,
    ] {
        peak_bytes = add(peak_bytes, bytes)?;
    }
    Ok(FinalizationQuote {
        retained_baseline_bytes,
        node_count,
        retained_output_owners,
        last_use_bytes,
        retirement_count_bytes,
        retirement_container_bytes,
        retirement_entries_bytes,
        workspace_inline_bytes,
        hash_finish_bytes,
        result_slot_bytes,
        peak_bytes,
        work,
    })
}

/// Validate dependency ordinals/coordinates BEFORE any vector allocation or hash
/// append. A dependency must name an earlier owner; a gather/selected coordinate
/// must fit that original owner. Duplicate dependencies are harmless last uses.
fn dependency<S: BorrowedOrderedNodes + ?Sized>(
    schedule: &S,
    ordinal: usize,
    owner: usize,
    index: Option<usize>,
    visit: &mut impl FnMut(usize) -> Result<(), Error>,
) -> Result<(), Error> {
    if owner >= ordinal {
        return Err(Error::Binding);
    }
    let source = schedule.node(owner)?;
    if index.is_some_and(|index| index >= source.lanes) {
        return Err(Error::Shape);
    }
    visit(owner)
}

fn dependencies<S: BorrowedOrderedNodes + ?Sized>(
    schedule: &S,
    ordinal: usize,
    mut visit: impl FnMut(usize) -> Result<(), Error>,
) -> Result<(), Error> {
    let node = schedule.node(ordinal)?;
    if node.lanes == 0 {
        return Err(Error::Shape);
    }
    match node.recipe {
        BorrowedRecipe::Input(_) | BorrowedRecipe::PublicInput { .. } => (),
        BorrowedRecipe::PublicValues(values) => {
            if values.len() != node.lanes {
                return Err(Error::Shape);
            }
        }
        BorrowedRecipe::GeneratedPublicValues(values) => {
            values.validate(node.lanes, node.bounds)?
        }
        BorrowedRecipe::Gather(words) => {
            if words.len() != node.lanes {
                return Err(Error::Shape);
            }
            for lane in 0..words.len() {
                let (id, index) = words.lane_ref(lane)?;
                dependency(schedule, ordinal, id, Some(index), &mut visit)?;
            }
        }
        BorrowedRecipe::SourceGather(view) => {
            dependency(schedule, ordinal, view.owner, None, &mut visit)?;
            if view.validate(schedule.node(view.owner)?.lanes)? != node.lanes {
                return Err(Error::Shape);
            }
        }
        BorrowedRecipe::Affine { left, right, .. } => {
            dependency(schedule, ordinal, left, None, &mut visit)?;
            if let Some(right) = right {
                dependency(schedule, ordinal, right, None, &mut visit)?;
            }
        }
        BorrowedRecipe::Product { left, right, .. } => {
            dependency(schedule, ordinal, left, None, &mut visit)?;
            dependency(schedule, ordinal, right, None, &mut visit)?;
        }
        BorrowedRecipe::PublicProduct { input, weight, .. }
        | BorrowedRecipe::PublicLookup { input, weight, .. }
        | BorrowedRecipe::TrieLookup { input, weight, .. } => {
            dependency(schedule, ordinal, input, None, &mut visit)?;
            dependency(schedule, ordinal, weight, None, &mut visit)?;
        }
        BorrowedRecipe::Tensor { input, program }
        | BorrowedRecipe::CheckedRange { input, program } => {
            dependency(schedule, ordinal, input, None, &mut visit)?;
            // Builder::tensor requires the complete native input width and
            // derives the output width from Program, never caller metadata.
            if schedule.node(input)?.lanes != program.lanes() {
                return Err(Error::Shape);
            }
            let output_lanes = match node.recipe {
                // checked_view retains safe words followed by validity words
                // in one owner (ranges.rs), unlike ordinary one-output Tensor.
                BorrowedRecipe::CheckedRange { .. } => program
                    .output_lanes()
                    .checked_add(program.validity_lanes().ok_or(Error::Shape)?)
                    .ok_or(Error::Bounds)?,
                _ => program.output_lanes(),
            };
            if node.lanes != output_lanes {
                return Err(Error::Shape);
            }
        }
        BorrowedRecipe::Selection { input, .. } => {
            dependency(schedule, ordinal, input, None, &mut visit)?;
        }
        BorrowedRecipe::TensorLoop {
            input,
            program,
            tail,
        } => {
            dependency(schedule, ordinal, input.owner(), None, &mut visit)?;
            let source = schedule.node(input.owner())?;
            match input {
                LoopInput::Slice(_) if node.lanes > source.lanes => return Err(Error::Shape),
                LoopInput::TokenOffsets { index, .. } if index >= source.lanes => {
                    return Err(Error::Shape);
                }
                _ => (),
            }
            // Check before work_budget's full-tile division. A tail may
            // neither disappear nor contribute extra work to an exact full tile.
            if program.lanes() == 0 {
                return Err(Error::Shape);
            }
            let remainder = node.lanes % program.lanes();
            match (remainder, tail) {
                (0, None) => (),
                (0, Some(_)) | (_, None) => return Err(Error::Shape),
                (n, Some(tail)) if tail.lanes() == n => (),
                _ => return Err(Error::Shape),
            }
        }
    }
    Ok(())
}

/// Allocation-free complete ORIGINAL schedule census. No outside work is
/// supplied or inferred; every recipe contributes its native primitive work once.
pub(super) fn quote<S: BorrowedOrderedNodes + ?Sized>(
    schedule: &S,
    selected: &[(usize, usize)],
    retained_baseline_bytes: usize,
) -> Result<FinalizationQuote, Error> {
    let n = schedule.node_count();
    let mut work = WorkBudget::default();
    for ordinal in 0..n {
        dependencies(schedule, ordinal, |_| Ok(()))?;
        work = work.checked_add(schedule.node(ordinal)?.work_budget()?)?;
    }
    if work.operations() == 0 {
        return Err(Error::Bounds);
    }
    work.limits()?;
    let mut retained = 0usize;
    for (position, &(owner, index)) in selected.iter().enumerate() {
        if owner >= n {
            return Err(Error::Binding);
        }
        if index >= schedule.node(owner)?.lanes {
            return Err(Error::Shape);
        }
        if !selected[..position]
            .iter()
            .any(|&(prior, _)| prior == owner)
        {
            retained = retained.checked_add(1).ok_or(Error::Bounds)?;
        }
    }
    storage_quote(n, retained, retained_baseline_bytes, work)
}

/// Canonical original bytes, using the SAME program helpers as Builder::finish.
/// Borrow programs/proofs/plans directly; never clone or substitute Arc owners.
fn hash_node(hash: &mut Sha256, node: BorrowedNode<'_>) -> Result<(), Error> {
    hash_node_header(hash, node.lanes, node.bounds);
    match node.recipe {
        BorrowedRecipe::Input(key) => {
            hash.update([0]);
            hash.update((key.len() as u64).to_le_bytes());
            hash.update(key);
        }
        BorrowedRecipe::PublicInput { key, digest } => {
            hash.update([5]);
            hash.update((key.len() as u64).to_le_bytes());
            hash.update(key);
            hash.update(digest);
        }
        BorrowedRecipe::PublicValues(values) => {
            hash_public_values(hash, values.iter().copied());
        }
        BorrowedRecipe::GeneratedPublicValues(values) => hash_public_values(hash, values.values()),
        BorrowedRecipe::PublicProduct {
            input,
            weight,
            rows,
            inner,
            columns,
            transposed,
        } => {
            hash_public_product(hash, input, weight, rows, inner, columns, transposed);
        }
        BorrowedRecipe::PublicLookup {
            input,
            weight,
            plan,
        } => {
            hash.update([10]);
            hash.update((input as u64).to_le_bytes());
            hash.update((weight as u64).to_le_bytes());
            hash.update(plan.digest());
        }
        BorrowedRecipe::Selection { input, plan } => {
            hash.update([11]);
            hash.update((input as u64).to_le_bytes());
            hash.update(plan.digest());
        }
        BorrowedRecipe::TrieLookup {
            input,
            weight,
            plan,
        } => {
            hash.update([12]);
            hash.update((input as u64).to_le_bytes());
            hash.update((weight as u64).to_le_bytes());
            hash.update(plan.digest());
        }
        BorrowedRecipe::CheckedRange { input, program } => {
            hash.update([9]);
            hash.update((input as u64).to_le_bytes());
            hash.update(program.digest());
        }
        BorrowedRecipe::Gather(words) => {
            // Use the original streaming encoder, with fallible coordinate
            // recovery. Never unwrap, truncate successfully or allocate lane refs.
            // A failed replay discards this private hash state with the result.
            let mut error = None;
            hash_gather(
                hash,
                (0..words.len()).scan((), |_, lane| match words.lane_ref(lane) {
                    Ok(word) => Some(word),
                    Err(failure) => {
                        error = Some(failure);
                        None
                    }
                }),
            );
            if let Some(error) = error {
                return Err(error);
            }
        }
        BorrowedRecipe::SourceGather(view) => {
            hash_gather(hash, view.indices().map(|index| (view.owner, index)));
        }
        BorrowedRecipe::Affine {
            left,
            right,
            coefficients,
        } => hash_affine(hash, left, right, coefficients),
        BorrowedRecipe::Product {
            left,
            right,
            shape,
            rescale,
            norm,
        } => {
            hash_product(hash, left, right, shape, rescale, norm);
        }
        BorrowedRecipe::Tensor { input, program } => hash_tensor(hash, input, program),
        BorrowedRecipe::TensorLoop {
            input,
            program,
            tail,
        } => {
            hash_tensor_loop(hash, input, program, tail);
        }
    }
    Ok(())
}

fn hash_public_values(hash: &mut Sha256, values: impl Iterator<Item = i64>) {
    hash.update([7]);
    for value in values {
        hash.update(value.to_le_bytes());
    }
}

fn reserved<T>(capacity: usize) -> Result<Vec<T>, Error> {
    let mut values = Vec::new();
    #[cfg(test)]
    if capacity != 0 {
        FINALIZATION_RESERVATIONS.with(|n| n.set(n.get() + 1));
    }
    values
        .try_reserve_exact(capacity)
        .map_err(|_| Error::Allocation)?;
    if values.capacity() != capacity {
        return Err(Error::Allocation);
    }
    Ok(values)
}

#[cfg(test)]
thread_local! {
    static FINALIZATION_FACTORY_ENTRIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    // Every positive-capacity helper allocation funnels through reserved().
    // This observes reserve attempts, not process RSS or outside fixture owners.
    static FINALIZATION_RESERVATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Borrowed original-recipe helper. `prefix` remains unchanged. Only after
/// validating/quoting/admitting do we allocate last-use, count and retire vectors.
/// No complete compact request API or compact execution admission is implied.
pub(super) fn finalize<S: BorrowedOrderedNodes + ?Sized>(
    schedule: &S,
    selected: &[(usize, usize)],
    prefix: &Sha256,
    retained_baseline_bytes: usize,
    cap: usize,
) -> Result<FinalizedSchedule, Error> {
    let quote = quote(schedule, selected, retained_baseline_bytes)?;
    quote.admit(cap)?;
    #[cfg(test)]
    FINALIZATION_FACTORY_ENTRIES.with(|n| n.set(n.get() + 1));
    let mut workspace = Workspace {
        last: reserved(quote.node_count)?,
        counts: reserved(quote.node_count)?,
        retire: reserved(quote.node_count)?,
        hash: prefix.clone(),
    };
    workspace.last.extend(0..quote.node_count);
    workspace.counts.resize(quote.node_count, 0);
    workspace.retire.resize_with(quote.node_count, Vec::new);
    for ordinal in 0..quote.node_count {
        dependencies(schedule, ordinal, |owner| {
            *workspace.last.get_mut(owner).ok_or(Error::Binding)? = ordinal;
            Ok(())
        })?;
        hash_node(&mut workspace.hash, schedule.node(ordinal)?)?;
    }
    for &(owner, index) in selected {
        *workspace.last.get_mut(owner).ok_or(Error::Binding)? = quote.node_count;
        workspace.hash.update((owner as u64).to_le_bytes());
        workspace.hash.update((index as u64).to_le_bytes());
    }
    for &end in &workspace.last {
        if end < quote.node_count {
            let count = workspace.counts.get_mut(end).ok_or(Error::Binding)?;
            *count = count.checked_add(1).ok_or(Error::Bounds)?;
        }
    }
    for (bucket, &count) in workspace.retire.iter_mut().zip(&workspace.counts) {
        *bucket = reserved(count)?;
    }
    for (owner, &end) in workspace.last.iter().enumerate() {
        if end < quote.node_count {
            workspace
                .retire
                .get_mut(end)
                .ok_or(Error::Binding)?
                .push(owner);
        }
    }
    Ok(FinalizedSchedule {
        digest: workspace.hash.finalize().into(),
        work: quote.work,
        retire: workspace.retire,
    })
}

/// Compare only after original Builder::finish has passed its own diagnostics.
/// Zero baseline deliberately scopes this check to helper-owned storage; it is
/// NOT a claim that the original compiler's simultaneously live owners fit.
/// Prefix includes the original source/policy/config bytes, without rebinding.
#[cfg(test)]
pub(super) fn compare_original(
    nodes: &[Node],
    selected: &[(usize, usize)],
    prefix: &Sha256,
    original_hash: &Sha256,
    original_work: WorkBudget,
    original_retire: &[Vec<usize>],
) {
    let quote = quote(nodes, selected, 0).expect("borrowed original finalization quote");
    let actual = finalize(nodes, selected, prefix, 0, quote.peak_bytes)
        .expect("borrowed original finalization comparison");
    let original_digest: [u8; 32] = original_hash.clone().finalize().into();
    assert_eq!(
        actual.digest, original_digest,
        "original canonical identity"
    );
    assert_eq!(actual.work, original_work, "original complete work");
    assert_eq!(
        actual.retire, original_retire,
        "original ordered retirement"
    );
}

#[cfg(test)]
#[path = "schedule_finalization_tests.rs"]
mod tests;
