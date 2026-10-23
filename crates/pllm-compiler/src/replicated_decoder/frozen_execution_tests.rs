//! TEST-ONLY consuming execution custody, not a bounded production issuer.
//!
//! Retains genuine original expanded Program/retirement/source authority. The
//! frozen loan is created here from its actual region, never paired by a caller.
//! Source-free recipes and qualified streamed-source contractions execute here.
//! Sealed argmax iota is lazy; source+iota/guard/proof combinations stay closed.
//! No Compiled or production route is minted.
use super::*;
use crate::replicated_decoder::program::compact_lowering::{
    frozen_region::FrozenRegion, DescriptorKind,
};
use crate::replicated_decoder::program::schedule_access::{physical, CONTROL_BYTES};
use crate::replicated_decoder::source_registry::SourceRegistry;
use std::mem::size_of;

#[path = "frozen_source_execution.rs"]
mod sources;

pub(in crate::replicated_decoder) const QUEUE_FRAMES: usize = 64;

#[path = "frozen_prefix_handoff.rs"]
pub(in crate::replicated_decoder) mod prefix_handoff;

pub(super) struct Values {
    pub(super) layout: Arc<physical::Layout>,
    pub(super) initialized: Vec<usize>,
}

/// Private test token borrows an internally created matching frozen schedule.
/// Possession alone is insufficient: every use checks actual Program/layout Arcs.
pub(in crate::replicated_decoder) struct Qualified<'s> {
    reader: PrivateScheduleRef<'s>,
    program: Arc<Program>,
    layout: Arc<physical::Layout>,
    origin: Arc<()>,
    admitted_peak: usize,
    session: [u8; 32],
    registry: Option<&'s SourceRegistry<'s>>,
}
fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Bounds)
}
fn mul(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_mul(b).ok_or(Error::Bounds)
}

fn ready(worker: &Worker, program: &Arc<Program>) -> Result<(), Error> {
    worker.live()?;
    if !Arc::ptr_eq(&worker.program, program)
        || worker.oracle_values.is_some()
        || worker.started
        || worker.step != 0
        || worker.loop_offset != 0
        || worker.imported != program.input_count()
        || worker.imported_words != 0
        || !worker.boundary_ready()
    {
        return Err(Error::Binding);
    }
    // Storage transition is only allowed before dispatch, with genuine imports.
    if worker
        .values
        .iter()
        .skip(program.input_count())
        .any(Option::is_some)
    {
        return Err(Error::Binding);
    }
    Ok(())
}

/// New-owner quote BEFORE the frozen loan, layout, initialized-prefix Vecs, or
/// any changed value storage/native invocation. Existing expanded construction
/// is NOT retroactively admitted. All six original Workers/Programs remain
/// charged; no Arc alias credit, caller baseline, cap upgrade or source guess.
fn preflight(
    control: &[Worker; 3],
    candidate: &[Worker; 3],
    region: &FrozenRegion<'_>,
    registry: Option<&SourceRegistry<'_>>,
    cap: usize,
) -> Result<usize, Error> {
    let program = &candidate[0].program;
    program.work_budget().limits()?;
    if region.exports().len() != 1 || !region.borrowed_source_norms().is_empty() {
        return Err(Error::Binding);
    }
    for census in region.descriptor_census() {
        if !matches!(
            census.kind,
            DescriptorKind::GuardedSilu
                | DescriptorKind::Binary { .. }
                | DescriptorKind::Reshape
                | DescriptorKind::LastToken
                | DescriptorKind::Argmax { .. }
                | DescriptorKind::Projection { .. }
                | DescriptorKind::TokenLookup { .. }
        ) {
            return Err(Error::Binding);
        }
    }
    for worker in control.iter().chain(candidate) {
        ready(worker, program)?;
    }
    let source_bytes = if let Some(registry) = registry {
        sources::preflight(control, candidate, region, registry)?
    } else {
        if !region.borrowed_source_records().is_empty()
            || control.iter().chain(candidate).any(|w| w.public.is_some())
        {
            return Err(Error::Binding);
        }
        0
    };
    let q = physical::quote(program.schedule_access())?;
    let recipe = region.schedule_recipe_quote()?;
    let mut peak = add(add(recipe.peak_bytes, q.metadata)?, source_bytes)?;
    // Six retained expanded owners plus conservative six in-flight native
    // template clones. Repeated templates are counted, never deduplicated.
    peak = add(peak, mul(program.frozen_oracle_metadata_bytes()?, 12)?)?;
    peak = add(
        peak,
        6 * size_of::<Worker>()
            + size_of::<Qualified<'_>>()
            + size_of::<Values>() * 3
            + CONTROL_BYTES,
    )?;
    for worker in control.iter().chain(candidate) {
        peak = add(
            peak,
            mul(worker.values.capacity(), size_of::<Option<Secret>>())?,
        )?;
        for value in worker.values.iter().flatten() {
            peak = add(
                peak,
                mul(
                    add(value.first.capacity(), value.second.capacity())?,
                    size_of::<u64>(),
                )?,
            )?;
        }
    }
    // All role-local physical owners plus old/new temporary output overlap.
    peak = add(
        peak,
        mul(add(q.words, q.scratch_words)?, 6 * 2 * size_of::<u64>())?,
    )?;
    // Original primitive ledgers plus emitted/encoded routing overlap, for both
    // isolated graphs. This is a conservative cap inventory, not RSS evidence.
    peak = add(peak, 12 * puma::MAX_ADMITTED_BYTES)?;
    // Two old/new pairs of capped routing Vecs, including exact encoded-frame
    // bounds. Tests compare every byte in flight instead of retaining a trace.
    peak = add(
        peak,
        4 * QUEUE_FRAMES * (size_of::<(Party, Party, PeerBytes)>() + puma::wire::MAX_FRAME_BYTES)
            + 4 * size_of::<Vec<(Party, Party, PeerBytes)>>(),
    )?;
    peak = add(
        peak,
        mul(add(73, mul(program.output_coordinates().len(), 8)?)?, 2)?,
    )?;
    if peak > cap.min(MAX_PROGRAM_BYTES) {
        return Err(Error::Bounds);
    }
    Ok(peak)
}

pub(in crate::replicated_decoder) fn quote(
    control: &[Worker; 3],
    candidate: &[Worker; 3],
    region: &FrozenRegion<'_>,
) -> Result<usize, Error> {
    preflight(control, candidate, region, None, MAX_PROGRAM_BYTES)
}

pub(in crate::replicated_decoder) fn source_quote(
    control: &[Worker; 3],
    candidate: &[Worker; 3],
    region: &FrozenRegion<'_>,
    registry: &SourceRegistry<'_>,
) -> Result<usize, Error> {
    preflight(
        control,
        candidate,
        region,
        Some(registry),
        MAX_PROGRAM_BYTES,
    )
}

pub(in crate::replicated_decoder) fn with_oracle<R>(
    control: [Worker; 3],
    candidate: [Worker; 3],
    region: &FrozenRegion<'_>,
    cap: usize,
    run: impl for<'s> FnOnce([Worker; 3], [Worker; 3], &Qualified<'s>) -> Result<R, Error>,
) -> Result<R, Error> {
    with_sources(control, candidate, region, None, cap, run)
}

pub(in crate::replicated_decoder) fn with_sources<R>(
    control: [Worker; 3],
    mut candidate: [Worker; 3],
    region: &FrozenRegion<'_>,
    registry: Option<&SourceRegistry<'_>>,
    cap: usize,
    run: impl for<'s> FnOnce([Worker; 3], [Worker; 3], &Qualified<'s>) -> Result<R, Error>,
) -> Result<R, Error> {
    let admitted_peak = preflight(&control, &candidate, region, registry, cap)?;
    let program = candidate[0].program.clone();
    let loan = region.borrow_schedule(cap)?;
    let reader = PrivateScheduleRef::frozen_owned(&loan, region);
    if reader.len() != program.logical_len()
        || region.source_binding() != program.source_binding
        || loan.resources().work != program.work_budget()
    {
        return Err(Error::Binding);
    }
    for id in 0..reader.len() {
        if !reader.exact_oracle_recipe(id, program.read_recipe(id))? {
            return Err(Error::Binding);
        }
    }
    let output = &region.exports()[0];
    let lanes = region.resolve(output)?.lanes;
    if lanes != program.output_coordinates().len() {
        return Err(Error::Binding);
    }
    for (i, &word) in program.output_coordinates().iter().enumerate() {
        if region.word(output, i)? != word {
            return Err(Error::Binding);
        }
    }
    let layout = Arc::new(physical::Layout::build(
        reader,
        &program.retire,
        program.output_coordinates(),
        region.schedule_recipe_quote()?.peak_bytes,
        cap,
    )?);
    for worker in &mut candidate {
        let mut initialized = Vec::new();
        initialized
            .try_reserve_exact(layout.owners.len())
            .map_err(|_| Error::Allocation)?;
        if initialized.capacity() != layout.owners.len() {
            return Err(Error::Allocation);
        }
        initialized.resize(layout.owners.len(), 0);
        for (id, initialized) in initialized[..program.input_count()].iter_mut().enumerate() {
            if reader.node(id)?.kind() == Kind::PublicInput {
                if worker.values[id].is_some() {
                    return Err(Error::Binding);
                }
                continue;
            }
            let span = layout.span(id)?;
            if span.stored()? != (id, 0) {
                return Err(Error::Binding);
            }
            let value = worker.values[id].as_ref().ok_or(Error::Binding)?;
            if value.first.len() != span.lanes || value.second.len() != span.lanes {
                return Err(Error::Shape);
            }
            *initialized = span.lanes;
        }
        // Reuse the exact original storage Vec and imported Secret owners. Its
        // larger retained capacity is priced above; no new private input exists.
        worker.values.truncate(layout.owners.len());
        worker.oracle_values = Some(Values {
            layout: layout.clone(),
            initialized,
        });
    }
    let custody = Qualified {
        reader,
        program,
        layout,
        origin: region.execution_origin().clone(),
        admitted_peak,
        session: candidate[0].session,
        registry,
    };
    run(control, candidate, &custody)
    // Both worker arrays cancel/drop on return, error or unwind. No borrowed
    // recipe can escape and no replacement Worker/PartyRuntime is constructed.
}

impl Qualified<'_> {
    /// Consumes an EXISTING all-role/router capability, never local-ready alone.
    /// Kept private to this module and its copy implementation. Production has
    /// no coordinator issuer; Program/layout/native Arc checks remain unchanged.
    fn consume_for_prefix(
        &self,
        drained: crate::replicated_decoder::body_tensor_integration::CoordinatorDrained<Worker>,
        region: &FrozenRegion<'_>,
    ) -> Result<Worker, Error> {
        let worker = drained.into_frozen_prefix_worker();
        self.check(&worker)?;
        if !Arc::ptr_eq(&self.origin, region.execution_origin())
            || !worker.boundary_ready()
            || worker.loop_offset != 0
        {
            return Err(Error::Binding);
        }
        Ok(worker)
    }
    fn check(&self, worker: &Worker) -> Result<(), Error> {
        worker.live()?;
        let state = worker.oracle_values.as_ref().ok_or(Error::Binding)?;
        if !Arc::ptr_eq(&worker.program, &self.program)
            || !Arc::ptr_eq(&state.layout, &self.layout)
            || worker.session != self.session
        {
            return Err(Error::Binding);
        }
        if let Some(registry) = self.registry {
            if worker.public.is_none() || registry.binding() != self.program.source_binding {
                return Err(Error::Binding);
            }
        }
        Ok(())
    }
    pub(in crate::replicated_decoder) fn advance(
        &self,
        worker: &mut Worker,
        before: Option<usize>,
    ) -> Result<Vec<(Party, PeerBytes)>, Error> {
        let result = (|| {
            self.check(worker)?;
            worker.advance_reader(before, self.reader)
        })();
        worker.checked(result)
    }
    pub(in crate::replicated_decoder) fn has_packed_offsets(&self) -> bool {
        self.layout.spans.iter().any(|span| {
            matches!(span.location, physical::Location::Stored { offset, .. } if offset != 0)
        })
    }
    pub(in crate::replicated_decoder) fn finish_frame(
        &self,
        worker: Worker,
    ) -> Result<Zeroizing<Vec<u8>>, Error> {
        self.check(&worker)?;
        worker.finish_frame_reader(Some(self.reader))
    }
    pub(in crate::replicated_decoder) fn address(
        &self,
        id: usize,
        index: usize,
    ) -> Result<(usize, usize), Error> {
        self.layout.address(id, index)
    }
    pub(in crate::replicated_decoder) fn weak_layout(&self) -> std::sync::Weak<physical::Layout> {
        Arc::downgrade(&self.layout)
    }
    /// Observation only, never source qualification or an origin seal.
    pub(in crate::replicated_decoder) fn source_backing(
        &self,
        worker: &Worker,
        owner: usize,
    ) -> Result<Option<pllm_core::public_file::FileBacking>, Error> {
        self.check(worker)?;
        let (key, _) = self.reader.node(owner)?.public_input()?;
        worker
            .public
            .as_ref()
            .ok_or(Error::Binding)?
            .file(key)?
            .backing()
            .map_err(|_| Error::Binding)
    }
    pub(in crate::replicated_decoder) fn assert_same_state(
        &self,
        control: &Worker,
        candidate: &Worker,
    ) {
        assert_eq!(
            (
                control.party,
                control.pid,
                control.step,
                control.loop_offset,
                control.imported,
                control.imported_words,
                control.session,
                control.started
            ),
            (
                candidate.party,
                candidate.pid,
                candidate.step,
                candidate.loop_offset,
                candidate.imported,
                candidate.imported_words,
                candidate.session,
                candidate.started
            )
        );
        assert_eq!(
            std::mem::discriminant(&control.active),
            std::mem::discriminant(&candidate.active)
        );
        assert_eq!(
            control.inbox.each_ref().map(Option::is_some),
            candidate.inbox.each_ref().map(Option::is_some)
        );
        let cursors = |w: &Worker| match &w.active {
            Active::Idle(r) | Active::Product(r) => Some(r.cursors()),
            Active::Tensor(e) => Some(e.cursors()),
            Active::Retired => None,
            _ => panic!("unsupported oracle active owner"),
        };
        assert_eq!(cursors(control), cursors(candidate));
        assert_eq!(control.complete(), candidate.complete());
        assert_eq!(control.retired(), candidate.retired());
        assert_eq!(control.public.is_some(), candidate.public.is_some());
        if candidate.retired() {
            assert!(candidate.values.is_empty());
            assert!(candidate.oracle_values.is_none());
            assert!(candidate.public.is_none());
            return;
        }
        let state = candidate.oracle_values.as_ref().unwrap();
        for id in 0..self.reader.len() {
            let old = control.values[id].as_ref();
            let span = self.layout.span(id).unwrap();
            if span.location == physical::Location::GeneratedPublic {
                assert!(candidate.local_slices(id).is_err());
                assert!(self.layout.address(id, 0).is_err());
                if let Some(old) = old {
                    for index in 0..span.lanes {
                        let value = self.reader.generated_element(id, index).unwrap() as u64;
                        let (first, second) = public_components(candidate.party, value);
                        assert_eq!((old.first[index], old.second[index]), (first, second));
                    }
                }
                continue;
            }
            let (slot, offset) = span.stored().unwrap();
            if let Some(old) = old {
                let new = candidate.values[slot]
                    .as_ref()
                    .expect("live physical owner");
                let initialized = state.initialized[slot]
                    .saturating_sub(offset)
                    .min(span.lanes);
                assert_eq!(
                    &old.first[..initialized],
                    &new.first[offset..offset + initialized]
                );
                assert_eq!(
                    &old.second[..initialized],
                    &new.second[offset..offset + initialized]
                );
                if initialized < span.lanes {
                    assert!(candidate.local_slices(id).is_err());
                } else {
                    let (a, b) = candidate.local_slices(id).unwrap();
                    assert_eq!(a, old.first.as_slice());
                    assert_eq!(b, old.second.as_slice());
                }
            } else {
                assert!(candidate.local_slices(id).is_err());
                if let Some(value) = &candidate.values[slot] {
                    if span.last_use < candidate.step {
                        assert!(value.first[offset..offset + span.lanes]
                            .iter()
                            .all(|x| *x == 0));
                        assert!(value.second[offset..offset + span.lanes]
                            .iter()
                            .all(|x| *x == 0));
                    }
                }
            }
        }
    }
}
