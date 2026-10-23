//! Original input creation, not retroactive certification of coordinate Views.
//! Source records are validated descriptors, NOT authenticated file/row proofs.
use super::compact_lowering::{ViewShape, compact_admission};
use super::*;
use crate::replicated_decoder::source_registry::{self, SourceRegistry};
use pllm_models::DecoderPlan;
use std::mem::size_of;

/// The actual immutable declaration retained by its creating Builder. Arc identity
/// identifies this declaration, not a random tag, digest, execution or proof owner.
pub(super) struct InputRecord {
    owner: usize,
    logical: ViewShape,
    bounds: Interval,
}

impl InputRecord {
    pub(super) fn owner(&self) -> usize {
        self.owner
    }
}

/// Only program's genuine token Input push can create this consuming event.
/// No existing ordinal/View or source hash is accepted as a creation substitute.
pub(super) fn request_record(
    created: super::CreatedRequestInput,
) -> Result<Arc<InputRecord>, Error> {
    let (owner, rows, bounds) = created.parts();
    Ok(Arc::new(InputRecord {
        owner,
        logical: ViewShape::new(&[1, rows])?,
        bounds,
    }))
}

/// Only declare_original_input can mint this view, at the original Input push.
/// No constructor accepts a preexisting View, owner ordinal, digest or baseline.
pub(super) struct OriginalImportedView<'s> {
    record: Arc<InputRecord>,
    registry: &'s SourceRegistry<'s>,
    plan: &'s DecoderPlan,
    graph: &'s DecoderGraph,
    semantic: &'s str,
}

impl<'s> OriginalImportedView<'s> {
    pub(super) fn owner(&self) -> usize {
        self.record.owner
    }
    pub(super) fn logical(&self) -> ViewShape {
        self.record.logical
    }
    pub(super) fn bounds(&self) -> Interval {
        self.record.bounds
    }
    pub(super) fn semantic(&self) -> &'s str {
        self.semantic
    }
    pub(super) fn registry(&self) -> &'s SourceRegistry<'s> {
        self.registry
    }
    pub(super) fn plan(&self) -> &'s DecoderPlan {
        self.plan
    }
    pub(super) fn graph(&self) -> &'s DecoderGraph {
        self.graph
    }
    /// Original creation-record identity, never coordinates/digest equivalence.
    pub(super) fn same_declaration(&self, other: &OriginalImportedView<'_>) -> bool {
        Arc::ptr_eq(&self.record, &other.record)
    }

    pub(super) fn check(
        &self,
        builder: &Builder,
        registry: &SourceRegistry<'_>,
        plan: &DecoderPlan,
        graph: &DecoderGraph,
    ) -> Result<(), Error> {
        if !std::ptr::eq(self.plan, plan)
            || !std::ptr::eq(self.graph, graph)
            || (!std::ptr::eq(graph, &plan.prefill) && !std::ptr::eq(graph, &plan.decode))
            || !std::ptr::eq(self.registry.source_records(), registry.source_records())
            || !std::ptr::eq(self.registry.source_norms(), registry.source_norms())
            || !builder
                .original_inputs
                .iter()
                .any(|record| Arc::ptr_eq(record, &self.record))
        {
            return Err(Error::Binding);
        }
        let node = builder.nodes.get(self.owner()).ok_or(Error::Binding)?;
        if !matches!(&node.code, Code::Input(key) if key == self.semantic)
            || node.bounds != self.bounds()
            || node.lanes != self.logical().dimensions().iter().product::<usize>()
        {
            return Err(Error::Binding);
        }
        Ok(())
    }
}

fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Bounds)
}
fn array<T>(n: usize) -> Result<usize, Error> {
    n.checked_mul(size_of::<T>()).ok_or(Error::Bounds)
}

impl Builder {
    /// Opt-in original declaration. Bounds declare an Input; they do not prove
    /// runtime range, producer completion, row energy or token-domain membership.
    /// Caller owns existing plan/registry; all other owners use the SAME enclosing
    /// admission ledger as compact construction. No arbitrary approval callback.
    /// Outside inventory excludes this Builder, complete plan, registry and its
    /// source maps: their actual owners are inventoried by this method itself.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn declare_original_input<'s>(
        &mut self,
        registry: &'s SourceRegistry<'s>,
        plan: &'s DecoderPlan,
        graph: &'s DecoderGraph,
        semantic: &str,
        bounds: Interval,
        prior: &compact_admission::RequestWork,
        owners: &compact_admission::RequestOwners<'_>,
        cap: usize,
    ) -> Result<OriginalImportedView<'s>, Error> {
        // This is an input declaration, never an incoming arithmetic-prefix export.
        if !std::ptr::eq(graph, &plan.prefill) && !std::ptr::eq(graph, &plan.decode) {
            return Err(Error::Binding);
        }
        let (semantic, logical) = if semantic == "input.tokens" {
            let name = graph
                .operations
                .iter()
                .flat_map(|op| &op.inputs)
                .find(|name| name.as_str() == semantic)
                .ok_or(Error::Binding)?;
            let vocabulary = graph
                .operations
                .iter()
                .find(|op| op.operator == Op::OutputHead)
                .and_then(|op| op.output_shape.last())
                .copied()
                .ok_or(Error::Binding)?;
            if bounds
                != Interval::new(0, i64::try_from(vocabulary).map_err(|_| Error::Bounds)? - 1)?
            {
                return Err(Error::Binding);
            }
            (
                name.as_str(),
                ViewShape::new(&[
                    usize::try_from(graph.batch).map_err(|_| Error::Bounds)?,
                    usize::try_from(graph.query_sequence).map_err(|_| Error::Bounds)?,
                ])?,
            )
        } else {
            let state = graph
                .state_inputs
                .iter()
                .find(|state| state.id == semantic)
                .ok_or(Error::Binding)?;
            let mut shape = [1usize; 4];
            if !(1..=4).contains(&state.shape.len()) {
                return Err(Error::Shape);
            }
            for (out, dimension) in shape.iter_mut().zip(&state.shape) {
                *out = usize::try_from(*dimension).map_err(|_| Error::Bounds)?;
            }
            (
                state.id.as_str(),
                ViewShape::new(&shape[..state.shape.len()])?,
            )
        };
        if self
            .nodes
            .iter()
            .any(|node| matches!(&node.code, Code::Input(key) if key == semantic))
        {
            return Err(Error::Binding);
        }
        let lanes = logical
            .dimensions()
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d).ok_or(Error::Bounds))?;
        let (outside, _, descriptors) =
            compact_admission::input_construction_baseline(prior, owners)?;
        if add(add(descriptors, self.nodes.len())?, 1)? > MAX_NODES {
            return Err(Error::Bounds);
        }
        // Read actual import capacities; arithmetic, validity and foreign source
        // nodes are not reclassified as declarations by this opt-in path.
        let existing = compact_lowering::CompactGraph::imports_bytes(self)?;
        let next = add(self.original_inputs.len(), 1)?;
        let record_bytes = add(size_of::<InputRecord>(), array::<usize>(2)?)?;
        let record_destination = if next > self.original_inputs.capacity() {
            array::<Arc<InputRecord>>(next)?
        } else {
            0
        };
        let node_destination = if self.nodes.len() == self.nodes.capacity() {
            array::<Node>(add(self.nodes.len(), 1)?)?
        } else {
            0
        };
        let retained_new = add(record_bytes, record_destination)?;
        let input_charge = add(
            add(size_of::<Node>(), semantic.len())?,
            array::<u64>(add(lanes, lanes)?)?,
        )?;
        let mut peak = add(outside, existing)?;
        for bytes in [
            source_registry::source_metadata_bytes(
                registry.source_records(),
                registry.source_norms(),
            )?,
            source_registry::decoder_plan_metadata_bytes(plan, cap)?,
            size_of::<SourceRegistry<'_>>(),
            retained_new,
            input_charge,
            node_destination,
            2 * size_of::<OriginalImportedView<'_>>(),
            2 * size_of::<Result<OriginalImportedView<'_>, Error>>(),
            size_of::<InputRecord>(),
            size_of::<Result<usize, Error>>(),
        ] {
            peak = add(peak, bytes)?;
        }
        if peak.max(owners.prior_request_peak_bytes) > cap.min(MAX_PROGRAM_BYTES) {
            return Err(Error::Bounds);
        }
        // Charge new identity/view ownership BEFORE either destination or Arc.
        // Keep conservative failed-construction/history charges; never refund.
        self.charge_program_bytes(retained_new)?;
        #[cfg(test)]
        INPUT_FACTORIES.with(|n| n.set(n.get() + 1));
        if record_destination != 0 {
            self.original_inputs
                .try_reserve_exact(1)
                .map_err(|_| Error::Allocation)?;
            if self.original_inputs.capacity() != next {
                return Err(Error::Allocation);
            }
        }
        if node_destination != 0 {
            self.nodes
                .try_reserve_exact(1)
                .map_err(|_| Error::Allocation)?;
            if self.nodes.capacity() != self.nodes.len() + 1 {
                return Err(Error::Allocation);
            }
        }
        let owner = self.push(Code::Input(semantic.to_owned()), lanes, bounds)?;
        let record = Arc::new(InputRecord {
            owner,
            logical,
            bounds,
        });
        self.original_inputs.push(record.clone());
        Ok(OriginalImportedView {
            record,
            registry,
            plan,
            graph,
            semantic,
        })
    }
}

#[cfg(test)]
thread_local! {
    static INPUT_FACTORIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
#[path = "original_import_tests.rs"]
pub(super) mod tests;

#[cfg(test)]
#[path = "original_import_owner_phase_regressions.rs"]
mod owner_phase_regressions;
