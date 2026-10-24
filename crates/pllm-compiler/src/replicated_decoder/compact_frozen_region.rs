//! Consuming recipe retention only. No semantic graph lifetime, execution,
//! request completion, certificate publication or finalization authority.
use super::*;
use std::mem::size_of;

#[path = "compact_prefix_custody.rs"]
pub(in crate::replicated_decoder) mod prefix_custody;

// Shared finalization remains a test-only seam. This does not open production
// routing, complete-request admission, dispatch or completion authority.
#[cfg(test)]
#[path = "compact_frozen_recipe.rs"]
pub(in crate::replicated_decoder) mod recipe;

#[derive(Clone, Copy)]
enum Name {
    Import(usize),
    Recipe(usize),
}

struct Import {
    name: String,
    view: PublishedView,
}

struct Frontier {
    name: Name,
    view: PublishedView,
}

/// A recomputable resource snapshot, never a caller-supplied admission ticket.
/// Outside owners remain charged. Semantic/frontier backing overlaps new names
/// and mappings during transition; recipes/exports move without another buffer.
/// retained_bytes describes the frozen owner, not the caller's subsequent phase:
/// any still-live semantic graph must remain in that phase's outside inventory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::replicated_decoder) struct FreezeResources {
    pub borrowed_owner_bytes: usize,
    pub borrowed_semantic_bytes: usize,
    pub recipe_capacity: usize,
    pub recipe_bytes: usize,
    pub native_owner_bytes: usize,
    pub export_capacity: usize,
    pub export_bytes: usize,
    pub import_count: usize,
    pub recipe_names: usize,
    pub frontier_count: usize,
    pub name_bytes: usize,
    pub new_backing_bytes: usize,
    pub retained_bytes: usize,
    pub transition_bytes: usize,
    pub historical_construction_bytes: usize,
    pub peak_bytes: usize,
}

impl FreezeResources {
    pub(in crate::replicated_decoder) fn admit(&self, cap: usize) -> Result<(), Error> {
        if self.peak_bytes > cap.min(MAX_PROGRAM_BYTES) {
            Err(Error::Bounds)
        } else {
            Ok(())
        }
    }
}

/// Owns the SAME recipes, prepared/native Arcs, provenance domain and export Vec
/// as its consumed graph. Imports/source/certificates remain borrowed for 'a.
/// Names and live frontier mappings are bounded metadata, never expanded lanes.
pub(in crate::replicated_decoder) struct FrozenRegion<'a> {
    imports_owner: &'a Builder,
    sources: &'a BTreeMap<String, SourceTensor>,
    norms: &'a SourceNorms,
    binding: [u8; 32],
    ops: Vec<PublishedOp<'a>>,
    imports: Vec<Import>,
    recipe_names: Vec<String>,
    frontier: Vec<Frontier>,
    exports: Vec<PublishedView>,
    output: Option<usize>,
    origin: Arc<()>,
    next: usize,
    request_prelude: Option<request_origin::compact_storage::Prelude>,
    work: WorkBudget,
    resources: FreezeResources,
    // Retain exact input seals and original plan/source owner borrows. A consuming
    // freeze is not permission to drop a still-borrowed original DecoderPlan.
    _prebound_inputs: Option<input_owner::InputOwner<'a>>,
    // External genuine consumed request context, never a borrowed lens into
    // this owner. Keeps exact source/plan/import reservation custody live.
    source_context: Option<&'a request_origin::compact_storage::EmissionContext<'a>>,
    validity: compact_validity::RequestValidity<'a>,
}

#[cfg(test)]
impl FrozenRegion<'_> {
    // Only Qualified captures this from the region whose actual schedule/native
    // Arc identities it checks. Possessing the Arc alone issues no capability.
    pub(in crate::replicated_decoder) fn execution_origin(&self) -> &Arc<()> {
        &self.origin
    }
    pub(in crate::replicated_decoder) fn execution_retained_bytes(&self) -> usize {
        self.resources.retained_bytes
    }
}

fn same_view(a: &PublishedView, b: &PublishedView) -> bool {
    Arc::ptr_eq(&a.origin, &b.origin)
        && a.slot == b.slot
        && a.mapping == b.mapping
        && a.start == b.start
        && a.lanes == b.lanes
        && a.shape == b.shape
        && a.rank == b.rank
        && a.logical == b.logical
        && a.order == b.order
        && a.import_slot == b.import_slot
        && a.bounds == b.bounds
}

fn reserve<T>(count: usize) -> Result<Vec<T>, Error> {
    #[cfg(test)]
    before_backing()?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| Error::Allocation)?;
    if values.capacity() != count {
        return Err(Error::Allocation);
    }
    Ok(values)
}

fn copy_name(name: &str) -> Result<String, Error> {
    #[cfg(test)]
    before_backing()?;
    let mut value = String::new();
    value
        .try_reserve_exact(name.len())
        .map_err(|_| Error::Allocation)?;
    if value.capacity() != name.len() {
        return Err(Error::Allocation);
    }
    value.push_str(name);
    Ok(value)
}

impl<'a> CompactGraph<'_, 'a> {
    /// No allocation or new owner. Exports must be exact original publications;
    /// a foreign same-digest graph or substituted shape/coordinate is rejected.
    /// Freeze recomputes this quote, so mutation cannot reuse an earlier snapshot.
    /// Only this transition is quoted, not earlier graph/source/export creation.
    pub(in crate::replicated_decoder) fn freeze_quote(
        &self,
        exports: &Vec<PublishedView>,
    ) -> Result<FreezeResources, Error> {
        if self.failed {
            return Err(Error::Retired);
        }
        if exports.len() > MAX_NODES || self.ops.len() > self.cursor {
            return Err(Error::Bounds);
        }
        for view in exports {
            self.validate_view(view)?;
        }
        for bound in &self.bound_imports {
            self.validate_view(&bound.view)?;
        }
        for op in &self.ops {
            self.validate_view(&op.input)?;
            self.validate_view(&op.output)?;
            if let PublishedKind::Binary { right, .. } = &op.kind {
                self.validate_view(right)?;
            }
        }
        let mut sticky = self.imports.validity;
        for op in &self.ops {
            if let PublishedKind::RmsNorm {
                checked: Some(rows),
                ..
            } = &op.kind
            {
                self.validity.check(self)?;
                rows.validate(
                    self.imports,
                    &op.input,
                    op.first,
                    op.output.shape[2],
                    sticky,
                )?;
                sticky = rows.outgoing;
            }
        }
        if sticky != self.validity.current() {
            return Err(Error::Binding);
        }
        let semantic = self.frontier.semantic_graph();
        let first = self
            .cursor
            .checked_sub(self.ops.len())
            .ok_or(Error::Binding)?;
        let operations = semantic
            .operations
            .get(first..self.cursor)
            .ok_or(Error::Binding)?;
        let mut names = 0;
        let mut frontier_count = 0;
        for (index, bound) in self.bound_imports.iter().enumerate() {
            if self.bound_imports[..index]
                .iter()
                .any(|prior| prior.semantic == bound.semantic)
                || operations.iter().any(|op| op.id == bound.semantic)
            {
                return Err(Error::Binding);
            }
            names = add_bytes(names, bound.semantic.len())?;
            if let Some(view) = self.frontier.value(bound.semantic) {
                if !same_view(view, &bound.view) {
                    return Err(Error::Binding);
                }
                frontier_count = add_bytes(frontier_count, 1)?;
            }
        }
        for (index, op) in operations.iter().enumerate() {
            if operations[..index].iter().any(|prior| prior.id == op.id) {
                return Err(Error::Binding);
            }
            names = add_bytes(names, op.id.len())?;
            if let Some(view) = self.frontier.value(&op.id) {
                if !same_view(view, &self.ops[index].output) {
                    return Err(Error::Binding);
                }
                frontier_count = add_bytes(frontier_count, 1)?;
            }
        }
        let mut native = 0;
        for program in self.owned_native_programs() {
            native = add_bytes(
                native,
                add_bytes(
                    program.resources()?.immutable_template_bytes,
                    2 * size_of::<usize>(),
                )?,
            )?;
        }
        // The bounded argmax kernel tables are retained separately from native
        // template Arcs. PreparedTensorMap itself has only inline fields/Arcs.
        for census in self.argmax_census() {
            native = add_bytes(native, census?.kernel_metadata_bytes)?;
        }
        for op in &self.ops {
            if let PublishedKind::Rotary(rows) = &op.kind {
                native = add_bytes(
                    native,
                    array_bytes::<compact_rotary::NativeRow>(rows.native.capacity())?,
                )?;
            }
            if let PublishedKind::RmsNorm {
                checked: Some(rows),
                ..
            } = &op.kind
            {
                native = add_bytes(native, rows.metadata_bytes()?)?;
            }
        }
        let recipe_bytes = array_bytes::<PublishedOp<'a>>(self.ops.capacity())?;
        let export_bytes = array_bytes::<PublishedView>(exports.capacity())?;
        let mut new_backing = names;
        for bytes in [
            array_bytes::<Import>(self.bound_imports.len())?,
            array_bytes::<String>(self.ops.len())?,
            array_bytes::<Frontier>(frontier_count)?,
        ] {
            new_backing = add_bytes(new_backing, bytes)?;
        }
        let source_bytes =
            super::super::super::source_registry::source_metadata_bytes(self.sources, self.norms)?;
        let borrowed = add_bytes(
            self.outside_owner_bytes,
            add_bytes(Self::imports_bytes(self.imports)?, source_bytes)?,
        )?;
        let semantic_bytes =
            super::super::super::source_registry::semantic_metadata_bytes(semantic)?;
        // Historical native-construction arms already contain outside owners,
        // original imports, frontier/recipe capacities and native scratch. They
        // omit the borrowed semantic graph and source/norm maps. RMSNorm alone
        // explicitly charges a gamma subowner, removed per arm by the original
        // admission recorder before taking this source-excluded maximum. Add
        // ONLY the two missing borrowed owners: not `borrowed` (which would
        // duplicate outside/import owners), nor the present transition's backing.
        let historical = add_bytes(
            add_bytes(self.construction_peak_without_source.get(), source_bytes)?,
            semantic_bytes,
        )?;
        let mut retained = add_bytes(borrowed, size_of::<FrozenRegion<'a>>())?;
        if self.prebound_inputs.is_some() {
            // Its plan remainder is already outside_owner_bytes; the exact graph
            // field remains borrowed too, unlike the legacy semantic-free path.
            retained = add_bytes(retained, semantic_bytes)?;
        }
        for bytes in [
            recipe_bytes,
            native,
            export_bytes,
            new_backing,
            2 * size_of::<usize>(),
        ] {
            retained = add_bytes(retained, bytes)?;
        }
        // Original graph containers overlap the frozen destination. Reuse their
        // capacity census; count moved recipes/native backing once, not once for
        // each inline Vec/Arc handle. The old frontier/import map remains live
        // until all replacement names/mappings have been constructed.
        let containers = self.container_census()?;
        let mut transition = borrowed;
        for bytes in [
            containers.inline_bytes,
            containers.frontier_storage_bound,
            containers.descriptor_bytes,
            containers.import_binding_bytes,
            containers.origin_arc_bytes,
            semantic_bytes,
            native,
            export_bytes,
            new_backing,
            size_of::<FrozenRegion<'a>>(),
            2 * size_of::<Result<FrozenRegion<'a>, Error>>(),
            2 * size_of::<FreezeResources>(),
            size_of::<Result<FreezeResources, Error>>(),
            size_of::<ContainerCensus>(),
            size_of::<Result<ContainerCensus, Error>>(),
            size_of::<Result<String, Error>>(),
            size_of::<String>(),
            size_of::<Result<Vec<Import>, Error>>(),
            size_of::<Result<Vec<String>, Error>>(),
            size_of::<Result<Vec<Frontier>, Error>>(),
            size_of::<std::collections::TryReserveError>(),
            size_of::<Import>(),
            size_of::<Frontier>(),
            2 * size_of::<PublishedView>(),
        ] {
            transition = add_bytes(transition, bytes)?;
        }
        transition = transition.max(retained);
        Ok(FreezeResources {
            borrowed_owner_bytes: borrowed,
            borrowed_semantic_bytes: semantic_bytes,
            recipe_capacity: self.ops.capacity(),
            recipe_bytes,
            native_owner_bytes: native,
            export_capacity: exports.capacity(),
            export_bytes,
            import_count: self.bound_imports.len(),
            recipe_names: self.ops.len(),
            frontier_count,
            name_bytes: names,
            new_backing_bytes: new_backing,
            retained_bytes: retained,
            transition_bytes: transition,
            historical_construction_bytes: historical,
            peak_bytes: transition.max(historical),
        })
    }

    /// Consumes graph AND supplied exports even on failure. No retry can publish
    /// a partial transition. Every new allocation follows recomputed admission;
    /// a borrowed quote or caller work/digest cannot authorize this operation.
    pub(in crate::replicated_decoder) fn freeze(
        self,
        exports: Vec<PublishedView>,
        cap: usize,
    ) -> Result<FrozenRegion<'a>, Error> {
        let resources = self.freeze_quote(&exports)?;
        resources.admit(cap.min(self.limit))?;
        #[cfg(test)]
        FREEZE_ALLOCATIONS.with(|count| count.set(count.get() + 1));
        let mut imports = reserve(self.bound_imports.len())?;
        let mut recipe_names = reserve(self.ops.len())?;
        let mut frontier = reserve(resources.frontier_count)?;
        let semantic = self.frontier.semantic_graph();
        let first = self.cursor - self.ops.len();
        for (index, bound) in self.bound_imports.iter().enumerate() {
            imports.push(Import {
                name: copy_name(bound.semantic)?,
                view: bound.view.clone(),
            });
            if let Some(view) = self.frontier.value(bound.semantic) {
                frontier.push(Frontier {
                    name: Name::Import(index),
                    view: view.clone(),
                });
            }
        }
        for (index, op) in semantic.operations[first..self.cursor].iter().enumerate() {
            recipe_names.push(copy_name(&op.id)?);
            if let Some(view) = self.frontier.value(&op.id) {
                frontier.push(Frontier {
                    name: Name::Recipe(index),
                    view: view.clone(),
                });
            }
        }
        let name = |name: Name| match name {
            Name::Import(i) => imports[i].name.as_str(),
            Name::Recipe(i) => recipe_names[i].as_str(),
        };
        let output = frontier
            .iter()
            .position(|entry| name(entry.name) == semantic.output);
        Ok(FrozenRegion {
            imports_owner: self.imports,
            sources: self.sources,
            norms: self.norms,
            binding: self.binding,
            ops: self.ops,
            imports,
            recipe_names,
            frontier,
            exports,
            output,
            origin: self.origin,
            next: self.next,
            request_prelude: self.request_prelude,
            work: self.work,
            resources,
            _prebound_inputs: self.prebound_inputs,
            source_context: self.request_admission,
            validity: self.validity,
        })
    }
}

impl<'a> FrozenRegion<'a> {
    /// Retained compiler-local sticky coordinate, not acceptance/export authority.
    pub(in crate::replicated_decoder) fn local_validity_owner(&self) -> Option<usize> {
        self.validity.current()
    }
    /// Borrow the original public PerToken recipes and their unchanged full/tail
    /// Arc owners. No kernel construction, lane expansion or new proof authority.
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

    pub(in crate::replicated_decoder) fn descriptor_census(
        &self,
    ) -> impl Iterator<Item = DescriptorCensus> + '_ {
        self.ops.iter().enumerate().map(|(slot, op)| {
            op.descriptor_census(self.ops.get(slot + 1).map_or(self.next, |next| next.first))
        })
    }

    pub(in crate::replicated_decoder) fn resources(&self) -> FreezeResources {
        self.resources
    }
    pub(in crate::replicated_decoder) fn work_budget(&self) -> WorkBudget {
        self.work
    }
    pub(in crate::replicated_decoder) fn next_virtual_owner(&self) -> usize {
        self.next
    }
    pub(in crate::replicated_decoder) fn exports(&self) -> &[PublishedView] {
        &self.exports
    }
    pub(in crate::replicated_decoder) fn output(&self) -> Option<&PublishedView> {
        self.output.map(|index| &self.frontier[index].view)
    }
    pub(in crate::replicated_decoder) fn recipe_name(&self, index: usize) -> Option<&str> {
        self.recipe_names.get(index).map(String::as_str)
    }
    pub(in crate::replicated_decoder) fn borrowed_source_records(
        &self,
    ) -> &'a BTreeMap<String, SourceTensor> {
        self.sources
    }
    pub(in crate::replicated_decoder) fn borrowed_source_norms(&self) -> &'a SourceNorms {
        self.norms
    }
    pub(in crate::replicated_decoder) fn source_binding(&self) -> [u8; 32] {
        self.binding
    }

    /// Borrow the unchanged original import recipe, not a virtual output node.
    pub(in crate::replicated_decoder) fn imported_recipe(&self, owner: usize) -> Option<&Node> {
        self.imports_owner.nodes.get(owner)
    }

    pub(in crate::replicated_decoder) fn named_view(&self, name: &str) -> Option<&PublishedView> {
        self.frontier
            .iter()
            .find(|entry| match entry.name {
                Name::Import(index) => self.imports[index].name == name,
                Name::Recipe(index) => self.recipe_names[index] == name,
            })
            .map(|entry| &entry.view)
    }

    pub(in crate::replicated_decoder) fn resolve(
        &self,
        view: &PublishedView,
    ) -> Result<PhysicalView, Error> {
        if !Arc::ptr_eq(&self.origin, &view.origin) {
            return Err(Error::Binding);
        }
        let original = if let Some(slot) = view.slot {
            &self.ops.get(slot).ok_or(Error::Binding)?.output
        } else {
            &self
                .imports
                .get(view.import_slot.ok_or(Error::Binding)?)
                .ok_or(Error::Binding)?
                .view
        };
        if !same_view(original, view) {
            return Err(Error::Binding);
        }
        Ok(view.physical())
    }

    pub(in crate::replicated_decoder) fn word(
        &self,
        view: &PublishedView,
        index: usize,
    ) -> Result<(usize, usize), Error> {
        self.resolve(view)?;
        view.word(index)
    }

    /// Existing immutable kernel capability, not a regenerated Program or proof.
    pub(in crate::replicated_decoder) fn prepared(
        &self,
        mut descriptor: usize,
    ) -> Result<&PreparedTensorMap, Error> {
        loop {
            let owner = self
                .ops
                .get(descriptor)
                .and_then(|op| op.prepared.as_ref())
                .ok_or(Error::Binding)?;
            match &owner.storage {
                MapStorage::Owned(prepared) => return Ok(prepared),
                MapStorage::Previous(previous) if *previous < descriptor => descriptor = *previous,
                _ => return Err(Error::Binding),
            }
        }
    }

    pub(in crate::replicated_decoder) fn native_row_program(
        &self,
        descriptor: usize,
    ) -> Option<&Arc<tensor::Program>> {
        match &self.ops.get(descriptor)?.kind {
            PublishedKind::RmsNorm { program, .. } => Some(program),
            _ => None,
        }
    }

    pub(in crate::replicated_decoder) fn visit_argmax(
        &self,
        descriptor: usize,
        visitor: &mut dyn for<'e> FnMut(ArgmaxEvent<'e>) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let op = self.ops.get(descriptor).ok_or(Error::Binding)?;
        let PublishedKind::Argmax(plan) = &op.kind else {
            return Err(Error::Binding);
        };
        plan.visit(&op.input, visitor)
    }
}

#[cfg(test)]
thread_local! {
    static FREEZE_ALLOCATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static FREEZE_FAIL_AFTER: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn before_backing() -> Result<(), Error> {
    FREEZE_FAIL_AFTER.with(|count| match count.get() {
        Some(0) => {
            count.set(None);
            Err(Error::Allocation)
        }
        Some(remaining) => {
            count.set(Some(remaining - 1));
            Ok(())
        }
        None => Ok(()),
    })
}

#[cfg(test)]
#[path = "compact_frozen_region_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "compact_frozen_lookup_tests.rs"]
mod lookup_tests;
