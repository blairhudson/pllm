//! Storage-neutral, borrowed schedule reads; never an executable admission.
//!
//! The existing canonical BorrowedRecipe remains the only recipe vocabulary.
//! Kind/getters expose its read fields across the compiler/runtime privacy
//! boundary, without constructing Code, Node, Words, a new Program or kernels.
//! A frozen loan does NOT supply a physical slot/retirement map, source registry
//! qualification, owning lookup plans, completion, or router-drain authority.
use super::schedule_finalization::{BorrowedNode, BorrowedOrderedNodes, BorrowedRecipe};
use super::*;
#[cfg(test)]
use compact_lowering::frozen_region::recipe::FrozenSchedule;
use compact_lowering::frozen_region::FrozenRegion;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::replicated_decoder) enum Kind {
    Input,
    PublicInput,
    PublicValues,
    GeneratedPublicValues,
    Gather,
    SourceGather,
    Affine,
    Product,
    PublicProduct,
    Tensor,
    TensorLoop,
    CheckedRange,
    PublicLookup,
    TrieLookup,
    Selection,
}

#[derive(Clone, Copy)]
enum Storage<'s> {
    Expanded(&'s Program),
    Compact(&'s source_entry::compact_pending::sealed::SealedOwnedStorage),
    ResidentCompact(&'s source_entry::compact_pending::sealed::ResidentSealedOwnedStorage),
    #[allow(dead_code)] // FrozenSchedule is currently a test-only metadata loan.
    Frozen(&'s dyn BorrowedOrderedNodes),
    // Installed only by consuming test custody from an internally created loan.
    #[allow(dead_code)] // No production issuer; frozen_owned is test-only.
    OwnedFrozen(&'s dyn BorrowedOrderedNodes, &'s FrozenRegion<'s>),
}

/// No self-reference or owned/erased source. Only actual compiler owners can
/// construct this loan; frozen construction accepts the existing schedule loan.
#[derive(Clone, Copy)]
pub(in crate::replicated_decoder) struct PrivateScheduleRef<'s>(Storage<'s>);

#[allow(dead_code)]
impl<'s> PrivateScheduleRef<'s> {
    pub(super) fn expanded(program: &'s Program) -> Self {
        Self(Storage::Expanded(program))
    }
    pub(super) fn compact(
        owner: &'s source_entry::compact_pending::sealed::SealedOwnedStorage,
    ) -> Self {
        Self(Storage::Compact(owner))
    }
    pub(super) fn resident_compact(
        owner: &'s source_entry::compact_pending::sealed::ResidentSealedOwnedStorage,
    ) -> Self {
        Self(Storage::ResidentCompact(owner))
    }
    #[cfg(test)]
    pub(in crate::replicated_decoder) fn frozen(schedule: &'s FrozenSchedule<'_, '_>) -> Self {
        Self(Storage::Frozen(schedule))
    }
    #[cfg(test)]
    pub(in crate::replicated_decoder) fn frozen_owned(
        schedule: &'s FrozenSchedule<'_, '_>,
        owner: &'s FrozenRegion<'s>,
    ) -> Self {
        Self(Storage::OwnedFrozen(schedule, owner))
    }
    pub(in crate::replicated_decoder) fn generated_len(
        self,
        ordinal: usize,
    ) -> Result<usize, Error> {
        if let Storage::Compact(owner) = self.0 {
            return owner.generated_len(ordinal);
        }
        if let Storage::ResidentCompact(owner) = self.0 {
            return owner.generated_len(ordinal);
        }
        let Storage::OwnedFrozen(_, owner) = self.0 else {
            return Err(Error::Binding);
        };
        let node = self.node(ordinal)?.0;
        let BorrowedRecipe::GeneratedPublicValues(values) = node.recipe else {
            return Err(Error::Binding);
        };
        values.checked_len(owner, node.lanes, node.bounds)
    }
    pub(in crate::replicated_decoder) fn generated_element(
        self,
        ordinal: usize,
        index: usize,
    ) -> Result<i64, Error> {
        if let Storage::Compact(owner) = self.0 {
            return owner.generated_element(ordinal, index);
        }
        if let Storage::ResidentCompact(owner) = self.0 {
            return owner.generated_element(ordinal, index);
        }
        let Storage::OwnedFrozen(_, owner) = self.0 else {
            return Err(Error::Binding);
        };
        let node = self.node(ordinal)?.0;
        let BorrowedRecipe::GeneratedPublicValues(values) = node.recipe else {
            return Err(Error::Binding);
        };
        values.checked_element(owner, node.lanes, node.bounds, index)
    }
    #[cfg(test)]
    pub(in crate::replicated_decoder) fn exact_oracle_recipe(
        self,
        ordinal: usize,
        original: ReadNode<'_>,
    ) -> Result<bool, Error> {
        let node = self.node(ordinal)?;
        if node.kind() != Kind::GeneratedPublicValues {
            return Ok(node.exact_oracle_recipe(original));
        }
        let len = self.generated_len(ordinal)?;
        if original.lanes() != len || original.bounds() != node.bounds() {
            return Ok(false);
        }
        let values = original.public_values()?;
        if values.len() != len {
            return Ok(false);
        }
        for (index, &value) in values.iter().enumerate() {
            if self.generated_element(ordinal, index)? != value {
                return Ok(false);
            }
        }
        Ok(true)
    }
    pub(in crate::replicated_decoder) fn len(self) -> usize {
        match self.0 {
            Storage::Expanded(p) => p.nodes.len(),
            Storage::Compact(p) => p.logical_len(),
            Storage::ResidentCompact(p) => p.logical_len(),
            Storage::Frozen(s) | Storage::OwnedFrozen(s, _) => s.node_count(),
        }
    }
    pub(in crate::replicated_decoder) fn node(self, ordinal: usize) -> Result<ReadNode<'s>, Error> {
        let node = match self.0 {
            Storage::Expanded(p) => p.nodes.as_slice().node(ordinal)?,
            Storage::Compact(p) => p.read_recipe(ordinal)?,
            Storage::ResidentCompact(p) => p.read_recipe(ordinal)?,
            Storage::Frozen(s) | Storage::OwnedFrozen(s, _) => s.node(ordinal)?,
        };
        Ok(ReadNode(node))
    }
    pub(in crate::replicated_decoder) fn input_count(self) -> Result<usize, Error> {
        match self.0 {
            Storage::Expanded(p) => Ok(p.inputs),
            Storage::Compact(p) => Ok(p.input_count()),
            Storage::ResidentCompact(p) => Ok(p.input_count()),
            Storage::Frozen(_) | Storage::OwnedFrozen(_, _) => Err(Error::Binding),
        }
    }
    pub(in crate::replicated_decoder) fn public_input_count(self) -> Result<usize, Error> {
        let mut count = 0;
        for ordinal in 0..self.input_count()? {
            count += usize::from(self.node(ordinal)?.kind() == Kind::PublicInput);
        }
        Ok(count)
    }
    pub(in crate::replicated_decoder) fn public_prefix(self, count: usize) -> Result<bool, Error> {
        for ordinal in 0..count {
            if self.node(ordinal)?.kind() != Kind::PublicInput {
                return Ok(false);
            }
        }
        Ok(true)
    }
    pub(in crate::replicated_decoder) fn value_slot_count(self) -> Result<usize, Error> {
        match self.0 {
            Storage::Expanded(p) => Ok(p.nodes.len()),
            Storage::Compact(p) => Ok(p.value_slot_count()),
            Storage::ResidentCompact(p) => Ok(p.value_slot_count()),
            Storage::Frozen(_) | Storage::OwnedFrozen(_, _) => Err(Error::Binding),
        }
    }
    pub(in crate::replicated_decoder) fn value_slot(self, logical: usize) -> Result<usize, Error> {
        match self.0 {
            // Preserve the legacy identity mapping, including caller-side bounds
            // checks. Never infer this map for virtual frozen recipe ordinals.
            Storage::Expanded(_) => Ok(logical),
            Storage::Compact(p) => p.value_slot(logical),
            Storage::ResidentCompact(p) => p.value_slot(logical),
            Storage::Frozen(_) | Storage::OwnedFrozen(_, _) => Err(Error::Binding),
        }
    }
    pub(in crate::replicated_decoder) fn retirement(
        self,
        ordinal: usize,
    ) -> Result<&'s [usize], Error> {
        match self.0 {
            Storage::Compact(p) => p.retirement(ordinal),
            Storage::ResidentCompact(p) => p.retirement(ordinal),
            Storage::Expanded(p) => p
                .retire
                .get(ordinal)
                .map(Vec::as_slice)
                .ok_or(Error::Binding),
            Storage::Frozen(_) | Storage::OwnedFrozen(_, _) => Err(Error::Binding),
        }
    }
    pub(in crate::replicated_decoder) fn outputs(self) -> Result<&'s [(usize, usize)], Error> {
        match self.0 {
            Storage::Expanded(p) => Ok(&p.outputs),
            Storage::Compact(p) => Ok(p.outputs()),
            Storage::ResidentCompact(p) => Ok(p.outputs()),
            Storage::Frozen(_) | Storage::OwnedFrozen(_, _) => Err(Error::Binding),
        }
    }
    pub(in crate::replicated_decoder) fn source_binding(self) -> Result<[u8; 32], Error> {
        match self.0 {
            Storage::Expanded(p) => Ok(p.source_binding),
            Storage::Compact(p) => Ok(p.source_binding()),
            Storage::ResidentCompact(p) => Ok(p.source_binding()),
            Storage::Frozen(_) | Storage::OwnedFrozen(_, _) => Err(Error::Binding),
        }
    }
    // BorrowedRecipe deliberately lends these plans by &Plan, not &Arc<Plan>.
    // Keep the genuine expanded owners; never synthesize an Arc from a digest,
    // cloned plan or unrelated frozen source record.
    pub(in crate::replicated_decoder) fn trie_owner(
        self,
        ordinal: usize,
    ) -> Result<&'s Arc<runtime::trie::Plan>, Error> {
        match self.0 {
            Storage::Compact(p) => p.trie_owner(ordinal),
            Storage::ResidentCompact(p) => p.trie_owner(ordinal),
            Storage::Expanded(p) => match &p.nodes.get(ordinal).ok_or(Error::Binding)?.code {
                Code::TrieLookup { plan, .. } => Ok(plan),
                _ => Err(Error::Binding),
            },
            Storage::Frozen(_) | Storage::OwnedFrozen(_, _) => Err(Error::Binding),
        }
    }
    pub(in crate::replicated_decoder) fn selection_owner(
        self,
        ordinal: usize,
    ) -> Result<&'s Arc<runtime::selection::Plan>, Error> {
        match self.0 {
            Storage::Compact(p) => p.selection_owner(ordinal),
            Storage::ResidentCompact(p) => p.selection_owner(ordinal),
            Storage::Expanded(p) => match &p.nodes.get(ordinal).ok_or(Error::Binding)?.code {
                Code::Selection { plan, .. } => Ok(plan),
                _ => Err(Error::Binding),
            },
            Storage::Frozen(_) | Storage::OwnedFrozen(_, _) => Err(Error::Binding),
        }
    }
}

#[derive(Clone, Copy)]
pub(in crate::replicated_decoder) struct ReadNode<'s>(BorrowedNode<'s>);

type ProductFields<'r, 's> = (
    &'r usize,
    &'r usize,
    &'r Option<MatrixShape>,
    &'r bool,
    &'r Option<&'s multiply::norm::RowEnergyProof>,
);

#[allow(dead_code)]
impl<'s> ReadNode<'s> {
    /// Test-only consuming-custody check, not digest equivalence or a producer
    /// seal. Native handles must be the very same Arcs. Unsupported families
    /// cannot enter the initial execution oracle.
    #[cfg(test)]
    pub(in crate::replicated_decoder) fn exact_oracle_recipe(self, other: ReadNode<'_>) -> bool {
        if self.lanes() != other.lanes() || self.bounds() != other.bounds() {
            return false;
        }
        match (self.0.recipe, other.0.recipe) {
            (BorrowedRecipe::Input(a), BorrowedRecipe::Input(b)) => a == b,
            (
                BorrowedRecipe::PublicInput { key: a, digest: b },
                BorrowedRecipe::PublicInput { key: x, digest: y },
            ) => a == x && b == y,
            (
                BorrowedRecipe::PublicProduct {
                    input: a,
                    weight: b,
                    rows: c,
                    inner: d,
                    columns: e,
                    transposed: f,
                },
                BorrowedRecipe::PublicProduct {
                    input: u,
                    weight: v,
                    rows: w,
                    inner: x,
                    columns: y,
                    transposed: z,
                },
            ) => (a, b, c, d, e, f) == (u, v, w, x, y, z),
            (BorrowedRecipe::PublicValues(a), BorrowedRecipe::PublicValues(b)) => a == b,
            (BorrowedRecipe::Gather(a), BorrowedRecipe::Gather(b)) => {
                a.len() == b.len()
                    && (0..a.len())
                        .all(|i| matches!((a.lane_ref(i),b.lane_ref(i)),(Ok(x),Ok(y)) if x==y))
            }
            (
                BorrowedRecipe::Affine {
                    left: a,
                    right: b,
                    coefficients: c,
                },
                BorrowedRecipe::Affine {
                    left: x,
                    right: y,
                    coefficients: z,
                },
            ) => {
                a == x
                    && b == y
                    && c.left == z.left
                    && c.right == z.right
                    && c.constant == z.constant
            }
            (
                BorrowedRecipe::Product {
                    left: a,
                    right: b,
                    shape: c,
                    rescale: d,
                    norm: None,
                },
                BorrowedRecipe::Product {
                    left: x,
                    right: y,
                    shape: z,
                    rescale: w,
                    norm: None,
                },
            ) => a == x && b == y && c == z && d == w,
            (
                BorrowedRecipe::Tensor {
                    input: a,
                    program: b,
                },
                BorrowedRecipe::Tensor {
                    input: x,
                    program: y,
                },
            ) => a == x && Arc::ptr_eq(b, y),
            (
                BorrowedRecipe::TensorLoop {
                    input: a,
                    program: b,
                    tail: c,
                },
                BorrowedRecipe::TensorLoop {
                    input: x,
                    program: y,
                    tail: z,
                },
            ) => {
                (match (a, x) {
                    (LoopInput::Slice(a), LoopInput::Slice(x)) => a == x,
                    (
                        LoopInput::TokenOffsets { owner: a, index: b },
                        LoopInput::TokenOffsets { owner: x, index: y },
                    ) => a == x && b == y,
                    _ => false,
                }) && Arc::ptr_eq(b, y)
                    && match (c, z) {
                        (None, None) => true,
                        (Some(c), Some(z)) => Arc::ptr_eq(c, z),
                        _ => false,
                    }
            }
            _ => false,
        }
    }
    pub(in crate::replicated_decoder) fn lanes(self) -> usize {
        self.0.lanes
    }
    pub(in crate::replicated_decoder) fn bounds(self) -> Interval {
        self.0.bounds
    }
    pub(in crate::replicated_decoder) fn kind(self) -> Kind {
        match self.0.recipe {
            BorrowedRecipe::Input(_) => Kind::Input,
            BorrowedRecipe::PublicInput { .. } => Kind::PublicInput,
            BorrowedRecipe::PublicValues(_) => Kind::PublicValues,
            BorrowedRecipe::GeneratedPublicValues(_) => Kind::GeneratedPublicValues,
            BorrowedRecipe::Gather(_) => Kind::Gather,
            BorrowedRecipe::SourceGather(_) => Kind::SourceGather,
            BorrowedRecipe::Affine { .. } => Kind::Affine,
            BorrowedRecipe::Product { .. } => Kind::Product,
            BorrowedRecipe::PublicProduct { .. } => Kind::PublicProduct,
            BorrowedRecipe::Tensor { .. } => Kind::Tensor,
            BorrowedRecipe::TensorLoop { .. } => Kind::TensorLoop,
            BorrowedRecipe::CheckedRange { .. } => Kind::CheckedRange,
            BorrowedRecipe::PublicLookup { .. } => Kind::PublicLookup,
            BorrowedRecipe::TrieLookup { .. } => Kind::TrieLookup,
            BorrowedRecipe::Selection { .. } => Kind::Selection,
        }
    }
    pub(in crate::replicated_decoder) fn input_name(self) -> Result<&'s str, Error> {
        match self.0.recipe {
            BorrowedRecipe::Input(key) => Ok(key),
            _ => Err(Error::Binding),
        }
    }
    pub(in crate::replicated_decoder) fn public_input(
        self,
    ) -> Result<(&'s str, &'s [u8; 32]), Error> {
        match self.0.recipe {
            BorrowedRecipe::PublicInput { key, digest } => Ok((key, digest)),
            _ => Err(Error::Binding),
        }
    }
    pub(in crate::replicated_decoder) fn public_values(self) -> Result<&'s [i64], Error> {
        match self.0.recipe {
            BorrowedRecipe::PublicValues(values) => Ok(values),
            // Generated values require the separately bound original owner.
            _ => Err(Error::Binding),
        }
    }
    pub(in crate::replicated_decoder) fn lane_refs(self) -> Result<LaneRefs<'s>, Error> {
        match self.0.recipe {
            BorrowedRecipe::Gather(refs) => Ok(LaneRefs {
                recipe: self.0.recipe,
                len: refs.len(),
                next: 0,
            }),
            BorrowedRecipe::SourceGather(view) => Ok(LaneRefs {
                recipe: self.0.recipe,
                len: view.elements()?,
                next: 0,
            }),
            _ => Err(Error::Binding),
        }
    }
    pub(in crate::replicated_decoder) fn gather_output_lanes(self) -> Result<usize, Error> {
        match self.0.recipe {
            BorrowedRecipe::Gather(refs) => Ok(refs.len()),
            BorrowedRecipe::SourceGather(_) => Ok(self.0.lanes),
            _ => Err(Error::Binding),
        }
    }
    pub(in crate::replicated_decoder) fn inputs(self) -> BorrowedInputs<'s> {
        BorrowedInputs {
            node: self,
            next: 0,
        }
    }
    pub(in crate::replicated_decoder) fn affine(
        &self,
    ) -> Result<(&usize, &Option<usize>, &Affine), Error> {
        match &self.0.recipe {
            BorrowedRecipe::Affine {
                left,
                right,
                coefficients,
            } => Ok((left, right, coefficients)),
            _ => Err(Error::Binding),
        }
    }
    pub(in crate::replicated_decoder) fn public_product(
        &self,
    ) -> Result<(&usize, &usize, &usize, &usize, &usize, &bool), Error> {
        match &self.0.recipe {
            BorrowedRecipe::PublicProduct {
                input,
                weight,
                rows,
                inner,
                columns,
                transposed,
            } => Ok((input, weight, rows, inner, columns, transposed)),
            _ => Err(Error::Binding),
        }
    }
    pub(in crate::replicated_decoder) fn product(&self) -> Result<ProductFields<'_, 's>, Error> {
        match &self.0.recipe {
            BorrowedRecipe::Product {
                left,
                right,
                shape,
                rescale,
                norm,
            } => Ok((left, right, shape, rescale, norm)),
            _ => Err(Error::Binding),
        }
    }
    pub(in crate::replicated_decoder) fn tensor(
        &self,
    ) -> Result<(&usize, &'s Arc<tensor::Program>), Error> {
        match &self.0.recipe {
            BorrowedRecipe::Tensor { input, program }
            | BorrowedRecipe::CheckedRange { input, program } => Ok((input, *program)),
            _ => Err(Error::Binding),
        }
    }
    pub(in crate::replicated_decoder) fn loop_input(&self) -> Result<&LoopInput, Error> {
        match &self.0.recipe {
            BorrowedRecipe::TensorLoop { input, .. } => Ok(input),
            _ => Err(Error::Binding),
        }
    }
    pub(in crate::replicated_decoder) fn loop_programs(
        self,
    ) -> Result<(&'s Arc<tensor::Program>, Option<&'s Arc<tensor::Program>>), Error> {
        match self.0.recipe {
            BorrowedRecipe::TensorLoop { program, tail, .. } => Ok((program, tail)),
            _ => Err(Error::Binding),
        }
    }
    /// Same original dispatch error order: offset bounds, exhausted phase, then
    /// required tail. No factory, native clone or full/tail recombination.
    pub(in crate::replicated_decoder) fn loop_tile(
        self,
        offset: usize,
    ) -> Result<&'s Arc<tensor::Program>, Error> {
        let BorrowedRecipe::TensorLoop { program, tail, .. } = self.0.recipe else {
            return Err(Error::Binding);
        };
        let remaining = self.0.lanes.checked_sub(offset).ok_or(Error::Bounds)?;
        if remaining == 0 {
            return Err(Error::Phase);
        }
        if remaining < program.lanes() {
            tail.ok_or(Error::Shape)
        } else {
            Ok(program)
        }
    }
    pub(in crate::replicated_decoder) fn public_lookup(
        &self,
    ) -> Result<(&usize, &usize, &'s runtime::lookup::Plan), Error> {
        match &self.0.recipe {
            BorrowedRecipe::PublicLookup {
                input,
                weight,
                plan,
            } => Ok((input, weight, *plan)),
            _ => Err(Error::Binding),
        }
    }
    pub(in crate::replicated_decoder) fn trie(
        &self,
    ) -> Result<(&usize, &usize, &'s runtime::trie::Plan), Error> {
        match &self.0.recipe {
            BorrowedRecipe::TrieLookup {
                input,
                weight,
                plan,
            } => Ok((input, weight, *plan)),
            _ => Err(Error::Binding),
        }
    }
    pub(in crate::replicated_decoder) fn selection(
        &self,
    ) -> Result<(&usize, &'s runtime::selection::Plan), Error> {
        match &self.0.recipe {
            BorrowedRecipe::Selection { input, plan } => Ok((input, *plan)),
            _ => Err(Error::Binding),
        }
    }
}

/// Ordered logical coordinates, not a deduplicated dependency/retirement set.
/// Error terminates iteration; no expanded lane Vec is ever constructed.
pub(in crate::replicated_decoder) struct LaneRefs<'s> {
    recipe: BorrowedRecipe<'s>,
    len: usize,
    next: usize,
}
impl Iterator for LaneRefs<'_> {
    type Item = Result<(usize, usize), Error>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.len {
            return None;
        }
        let i = self.next;
        self.next += 1;
        let value = match self.recipe {
            BorrowedRecipe::Gather(refs) => refs.lane_ref(i),
            BorrowedRecipe::SourceGather(view) => (i / view.columns)
                .checked_mul(view.row_stride)
                .and_then(|n| {
                    (i % view.columns)
                        .checked_mul(view.column_stride)
                        .and_then(|col| n.checked_add(col))
                })
                .and_then(|n| view.offset.checked_add(n))
                .map(|index| (view.owner, index))
                .ok_or(Error::Bounds),
            _ => Err(Error::Binding),
        };
        if value.is_err() {
            self.next = self.len;
        }
        Some(value)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.len - self.next;
        (0, Some(n))
    }
}

/// Lazy original dependency reads. Repeated gather owners are intentionally not
/// sorted/deduplicated; this is not a last-use map or release authorization.
pub(in crate::replicated_decoder) struct BorrowedInputs<'s> {
    node: ReadNode<'s>,
    next: usize,
}
impl Iterator for BorrowedInputs<'_> {
    type Item = Result<usize, Error>;
    fn next(&mut self) -> Option<Self::Item> {
        let i = self.next;
        self.next = self.next.saturating_add(1);
        let pair = |a, b| {
            if i == 0 {
                Some(Ok(a))
            } else if i == 1 {
                Some(Ok(b))
            } else {
                None
            }
        };
        match self.node.0.recipe {
            BorrowedRecipe::Gather(refs) if i < refs.len() => Some(refs.lane_ref(i).map(|r| r.0)),
            BorrowedRecipe::SourceGather(view) if i == 0 => Some(Ok(view.owner)),
            BorrowedRecipe::Affine { left, right, .. } => {
                if i == 0 {
                    Some(Ok(left))
                } else if i == 1 {
                    right.map(Ok)
                } else {
                    None
                }
            }
            BorrowedRecipe::Product { left, right, .. } => pair(left, right),
            BorrowedRecipe::PublicProduct { input, weight, .. }
            | BorrowedRecipe::PublicLookup { input, weight, .. }
            | BorrowedRecipe::TrieLookup { input, weight, .. } => pair(input, weight),
            BorrowedRecipe::Tensor { input, .. }
            | BorrowedRecipe::CheckedRange { input, .. }
            | BorrowedRecipe::Selection { input, .. }
                if i == 0 =>
            {
                Some(Ok(input))
            }
            BorrowedRecipe::TensorLoop { input, .. } if i == 0 => Some(Ok(input.owner())),
            _ => None,
        }
    }
}

/// Explicit original frozen export mapping, separate from schedule ordinals.
/// Checked by FrozenRegion::word/resolve; does not mint physical runtime slots.
#[allow(dead_code)]
pub(in crate::replicated_decoder) struct FrozenExportRefs<'r, 'a> {
    owner: &'r FrozenRegion<'a>,
    view: &'r compact_lowering::PublishedView,
    len: usize,
    next: usize,
}
#[allow(dead_code)]
impl<'r, 'a> FrozenExportRefs<'r, 'a> {
    pub(super) fn new(
        owner: &'r FrozenRegion<'a>,
        view: &'r compact_lowering::PublishedView,
    ) -> Result<Self, Error> {
        let len = owner.resolve(view)?.lanes;
        Ok(Self {
            owner,
            view,
            len,
            next: 0,
        })
    }
}
impl Iterator for FrozenExportRefs<'_, '_> {
    type Item = Result<(usize, usize), Error>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.len {
            return None;
        }
        let i = self.next;
        self.next += 1;
        Some(self.owner.word(self.view, i))
    }
}

/// Fixed typed read/control inventory, above the caller's existing retained
/// owners. Reader performs no allocation/factory. This quote is NOT whole-Worker
/// admission; source-qualified publication must compose it before factories.
/// PrivateScheduleRef's actual enum extent includes both concrete compact loans;
/// each thin dispatch enters the same C-preserving storage read implementation.
pub(in crate::replicated_decoder) const CONTROL_BYTES: usize = std::mem::size_of::<PrivateScheduleRef<'_>>()
        + std::mem::size_of::<Arc<Program>>()
        + 2 * std::mem::size_of::<ReadNode<'_>>()
        + 2 * std::mem::size_of::<Result<ReadNode<'_>, Error>>()
        + std::mem::size_of::<LaneRefs<'_>>()
        + std::mem::size_of::<BorrowedInputs<'_>>()
        + std::mem::size_of::<FrozenExportRefs<'_, '_>>()
        + std::mem::size_of::<Result<&Arc<tensor::Program>, Error>>()
        // Named generated owner/query and optional finish-reader positions.
        // Not compiled private callframes, allocator overhead or RSS.
        + 3 * std::mem::size_of::<super::schedule_finalization::GeneratedPublicValues<'_>>()
        + 2 * std::mem::size_of::<Result<usize, Error>>()
        + 2 * std::mem::size_of::<Result<i64, Error>>()
        + 2 * std::mem::size_of::<Option<PrivateScheduleRef<'_>>>()
        + std::mem::size_of::<Result<(u64, u64), Error>>()
        + 2 * std::mem::size_of::<(u64, u64)>();
#[allow(dead_code)]
pub(in crate::replicated_decoder) fn control_preflight(
    retained: usize,
    limit: usize,
) -> Result<usize, Error> {
    retained
        .checked_add(CONTROL_BYTES)
        .filter(|n| *n <= limit.min(MAX_PROGRAM_BYTES))
        .ok_or(Error::Bounds)
}

#[cfg(test)]
#[path = "schedule_access_tests.rs"]
mod tests;

/// Geometry only, not an execution/source issuer. The initial execution oracle
/// retains the genuine expanded Program and validates its retirement authority.
/// No production Worker can install this test-only layout.
#[cfg(test)]
pub(in crate::replicated_decoder) mod physical {
    use super::*;
    use std::mem::size_of;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(in crate::replicated_decoder) struct Span {
        pub location: Location,
        pub lanes: usize,
        pub last_use: usize,
    }
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(in crate::replicated_decoder) enum Location {
        Stored { slot: usize, offset: usize },
        GeneratedPublic,
    }
    impl Span {
        pub(in crate::replicated_decoder) fn stored(self) -> Result<(usize, usize), Error> {
            match self.location {
                Location::Stored { slot, offset } => Ok((slot, offset)),
                Location::GeneratedPublic => Err(Error::Binding),
            }
        }
    }
    #[derive(Clone, Copy, Debug)]
    pub(in crate::replicated_decoder) struct Owner {
        pub lanes: usize,
        pub last_use: usize,
    }
    pub(in crate::replicated_decoder) struct Layout {
        pub spans: Vec<Span>,
        pub owners: Vec<Owner>,
    }
    #[derive(Clone, Copy, Debug)]
    pub(in crate::replicated_decoder) struct Quote {
        pub spans: usize,
        pub owners: usize,
        pub metadata: usize,
        pub words: usize,
        pub scratch_words: usize,
    }
    fn add(a: usize, b: usize) -> Result<usize, Error> {
        a.checked_add(b).ok_or(Error::Bounds)
    }
    fn array<T>(n: usize) -> Result<usize, Error> {
        n.checked_mul(size_of::<T>()).ok_or(Error::Bounds)
    }
    fn affine(node: ReadNode<'_>) -> bool {
        // One dense affine owner includes full vector chunks AND scalar tails.
        // Intervening gather/native nodes do not imply an affine-owner release.
        node.kind() == Kind::Affine
    }
    pub(in crate::replicated_decoder) fn quote(
        reader: PrivateScheduleRef<'_>,
    ) -> Result<Quote, Error> {
        let mut owners = 0;
        let mut packed = false;
        let mut packed_lanes = 0usize;
        let mut words = 0;
        let mut scratch_words = 0;
        for id in 0..reader.len() {
            let node = reader.node(id)?;
            if node.lanes() == 0 || (node.kind() != Kind::PublicInput && node.lanes() > MAX_VALUES)
            {
                return Err(Error::Bounds);
            }
            if node.kind() == Kind::GeneratedPublicValues {
                reader.generated_len(id)?;
                continue; // No secret owner, payload, scratch or slot for iota.
            }
            let next = affine(node);
            if !next || !packed {
                owners = add(owners, 1)?;
            }
            if next {
                packed = true;
                packed_lanes = add(packed_lanes, node.lanes())?;
                if packed_lanes > MAX_VALUES {
                    return Err(Error::Bounds);
                }
            }
            if node.kind() != Kind::PublicInput {
                words = add(words, node.lanes())?;
                scratch_words = scratch_words.max(node.lanes());
            }
        }
        let mut metadata = add(
            size_of::<Layout>() + 2 * size_of::<usize>(),
            array::<Span>(reader.len())?,
        )?;
        metadata = add(metadata, array::<Owner>(owners)?)?;
        // Three distinct role-local initialized-prefix owners; no lane bitmap.
        metadata = add(
            metadata,
            array::<usize>(owners)?
                .checked_mul(3)
                .ok_or(Error::Bounds)?,
        )?;
        metadata = add(
            metadata,
            3 * size_of::<Vec<usize>>() + 2 * size_of::<Quote>() + CONTROL_BYTES,
        )?;
        for control in [
            size_of::<Result<Layout, Error>>(),
            size_of::<Result<Quote, Error>>(),
            size_of::<Result<Vec<Span>, Error>>(),
            size_of::<Result<Vec<Owner>, Error>>(),
            size_of::<Result<Vec<usize>, Error>>(),
            size_of::<std::collections::TryReserveError>(),
            2 * size_of::<Span>(),
            2 * size_of::<Owner>(),
            size_of::<Result<Span, Error>>(),
        ] {
            metadata = add(metadata, control)?;
        }
        Ok(Quote {
            spans: reader.len(),
            owners,
            metadata,
            words,
            scratch_words,
        })
    }
    fn reserve<T>(n: usize) -> Result<Vec<T>, Error> {
        let mut v = Vec::new();
        v.try_reserve_exact(n).map_err(|_| Error::Allocation)?;
        if v.capacity() != n {
            return Err(Error::Allocation);
        }
        Ok(v)
    }
    impl Layout {
        pub(in crate::replicated_decoder) fn build(
            reader: PrivateScheduleRef<'_>,
            retirement: &[Vec<usize>],
            outputs: &[(usize, usize)],
            retained: usize,
            cap: usize,
        ) -> Result<Self, Error> {
            let q = quote(reader)?;
            if add(retained, q.metadata)? > cap.min(MAX_PROGRAM_BYTES)
                || retirement.len() != reader.len()
            {
                return Err(Error::Bounds);
            }
            let mut spans = reserve(q.spans)?;
            let mut owners: Vec<Owner> = reserve(q.owners)?;
            let mut packed: Option<usize> = None;
            for id in 0..reader.len() {
                let node = reader.node(id)?;
                if node.kind() == Kind::GeneratedPublicValues {
                    reader.generated_len(id)?;
                    spans.push(Span {
                        location: Location::GeneratedPublic,
                        lanes: node.lanes(),
                        last_use: id,
                    });
                    continue;
                }
                let next = affine(node);
                let (slot, offset) = if let Some(slot) = packed.filter(|_| next) {
                    let owner = owners.get_mut(slot).ok_or(Error::Binding)?;
                    let offset = owner.lanes;
                    owner.lanes = add(owner.lanes, node.lanes())?;
                    if owner.lanes > MAX_VALUES {
                        return Err(Error::Bounds);
                    }
                    (slot, offset)
                } else {
                    owners.push(Owner {
                        lanes: node.lanes(),
                        last_use: id,
                    });
                    let slot = owners.len() - 1;
                    if next {
                        packed = Some(slot);
                    }
                    (slot, 0)
                };
                spans.push(Span {
                    location: Location::Stored { slot, offset },
                    lanes: node.lanes(),
                    last_use: id,
                });
            }
            // Independently derive logical liveness from lazy dependency reads.
            // This is storage geometry, not a replacement for source qualification.
            for id in 0..reader.len() {
                for input in reader.node(id)?.inputs() {
                    let input = input?;
                    if input >= id {
                        return Err(Error::Binding);
                    }
                    spans[input].last_use = id;
                }
            }
            for &(id, index) in outputs {
                let span = spans.get_mut(id).ok_or(Error::Binding)?;
                if index >= span.lanes {
                    return Err(Error::Shape);
                }
                span.last_use = reader.len();
            }
            // Compare the actual original retirement lists, including order,
            // duplicates and output pins. No hash/ID match authorizes release.
            let mut retired = 0usize;
            for (at, list) in retirement.iter().enumerate() {
                let mut previous = None;
                for &id in list {
                    if spans.get(id).ok_or(Error::Binding)?.last_use != at
                        || previous.is_some_and(|p| p >= id)
                    {
                        return Err(Error::Binding);
                    }
                    previous = Some(id);
                    retired = add(retired, 1)?;
                }
            }
            if retired
                != spans
                    .iter()
                    .filter(|span| span.last_use < reader.len())
                    .count()
            {
                return Err(Error::Binding);
            }
            for span in &spans {
                if let Location::Stored { slot, .. } = span.location {
                    let owner = &mut owners[slot];
                    owner.last_use = owner.last_use.max(span.last_use);
                }
            }
            Ok(Self { spans, owners })
        }
        pub(in crate::replicated_decoder) fn span(&self, id: usize) -> Result<Span, Error> {
            self.spans.get(id).copied().ok_or(Error::Binding)
        }
        pub(in crate::replicated_decoder) fn address(
            &self,
            id: usize,
            index: usize,
        ) -> Result<(usize, usize), Error> {
            let s = self.span(id)?;
            if index >= s.lanes {
                return Err(Error::Shape);
            }
            let (slot, offset) = s.stored()?;
            Ok((slot, add(offset, index)?))
        }
    }
}

#[cfg(test)]
#[path = "schedule_execution_tests.rs"]
mod execution_tests;

#[cfg(test)]
#[path = "schedule_iota_execution_tests.rs"]
mod iota_execution_tests;

#[cfg(test)]
#[path = "schedule_source_execution_tests.rs"]
mod source_execution_tests;
