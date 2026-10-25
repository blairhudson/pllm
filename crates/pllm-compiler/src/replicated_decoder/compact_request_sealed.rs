//! Actual consuming canonical finalization of genuine two-epoch history.
//! The original finish Attempt stays armed through private Program publication
//! and startup checks. No OPEN inspection loan is retained in this owner.
use super::*;
use crate::replicated_decoder::program::compact_lowering::frozen_region::owning::recipe::{
    HistorySchedule, RetainedRecipes,
};
use crate::replicated_decoder::program::schedule_finalization::{self, BorrowedOrderedNodes};
use std::alloc::Layout;

type History = request_origin::compact_storage::owning::OwnedEmission<
    request_origin::compact_storage::compact_history::CompletionLoan,
>;
type Root = SourceCustody<History>;
#[cfg(test)]
type PublicationObserver = Box<dyn FnOnce(usize)>;
#[cfg(test)]
thread_local! {static PUBLICATION_OBSERVER:std::cell::RefCell<Option<PublicationObserver>>=const{std::cell::RefCell::new(None)};}
#[cfg(test)]
pub(in crate::replicated_decoder::program) fn observe_publication_once(
    observer: impl FnOnce(usize) + 'static,
) {
    PUBLICATION_OBSERVER.with(|slot| {
        let mut slot = slot.borrow_mut();
        assert!(slot.is_none());
        *slot = Some(Box::new(observer));
    });
}

/// Borrow-free canonical owner. Private Program publication consumes it and only
/// then commits the SAME finish Attempt. Abandonment burns COMPLETE. All arrays and
/// original charged payloads die before finalization's covering lease.
pub(in crate::replicated_decoder) struct SealedOwnedStorage {
    attempt: request_origin::schedule::Attempt,
    recipes: RetainedRecipes,
    owner: Arc<FinalizedRoot>,
    _charge: OwnedExternalCharge,
}
// This actual root is born ONLY after the shared finalizer succeeds. Retained
// recipe descriptors share it, not the outer storage: no cycle/self-reference.
pub(in crate::replicated_decoder::program) struct FinalizedRoot {
    slots: Vec<usize>,
    finalized: schedule_finalization::FinalizedSchedule,
    root: Arc<Root>,
    outputs: [(usize, usize); 3],
    inputs: usize,
    slot_count: usize,
}

impl FinalizedRoot {
    pub(in crate::replicated_decoder::program) fn custody(&self) -> &Root {
        &self.root
    }
    pub(in crate::replicated_decoder::program) fn imports(&self) -> Result<&Builder, Error> {
        self.root.produced.completed_imports()
    }
}

fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Bounds)
}
fn array<T>(n: usize) -> Result<usize, Error> {
    n.checked_mul(size_of::<T>()).ok_or(Error::Bounds)
}
fn reserve<T>(n: usize) -> Result<Vec<T>, Error> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| Error::Allocation)?;
    if v.capacity() != n {
        return Err(Error::Bounds);
    }
    Ok(v)
}

// Actual finalizer intervals issue slots, not logical-id equality or a borrowed
// test layout. Allocate the output BEFORE releasing inputs at the same ordinal.
// Prefix imports are all installed before execution; never reuse their slots
// based on retirement buckets which the original Worker does not execute.
fn slots(retire: &[Vec<usize>], inputs: usize) -> Result<(Vec<usize>, usize), Error> {
    let n = retire.len();
    if inputs > n {
        return Err(Error::Binding);
    }
    let mut map = reserve(n)?;
    let mut free = reserve(n)?;
    let mut retired = reserve::<u8>(n)?;
    retired.resize(n, 0);
    let mut count = 0;
    for (ordinal, bucket) in retire.iter().enumerate() {
        let slot = if ordinal >= inputs { free.pop() } else { None };
        map.push(match slot {
            Some(slot) => slot,
            None => {
                let slot = count;
                count = add(count, 1)?;
                slot
            }
        });
        if ordinal >= inputs {
            for &logical in bucket {
                if logical > ordinal || retired.get(logical) != Some(&0) {
                    return Err(Error::Binding);
                }
                retired[logical] = 1;
                free.push(*map.get(logical).ok_or(Error::Binding)?);
            }
        }
    }
    Ok((map, count))
}

fn grow(guard: &mut OwnedExternalCharge, root: &Root, additional: usize) -> Result<(), Error> {
    let budget = root.produced().source_plan().budget();
    if !guard.is_for(budget) {
        return Err(Error::Binding);
    }
    let outside = budget
        .used_bytes()
        .checked_sub(guard.payload_bytes())
        .ok_or(Error::Binding)?;
    let payload = add(guard.payload_bytes(), additional)?;
    if add(outside, payload)? > root.produced().program_cap() {
        return Err(Error::Bounds);
    }
    guard.resize_payload(payload).map_err(|_| Error::Bounds)
}

impl Root {
    #[cfg(test)]
    pub(in crate::replicated_decoder::program) fn outside_lease_for_tests(
        &mut self,
    ) -> Result<OwnedExternalCharge, Error> {
        self.produced.finish_lease()
    }
    /// Only the consuming genuine source/history owner can call this issuer.
    /// Source plans, imports and both selected producers are never reconstructed.
    pub(in crate::replicated_decoder::program) fn seal_compact(
        mut self,
    ) -> Result<SealedOwnedStorage, Error> {
        self.check_emission(&self.produced)?;
        let (layout, _) = Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
            .extend(Layout::new::<Root>())
            .map_err(|_| Error::Bounds)?;
        let controls = add(
            layout.pad_to_align().size(),
            add(
                2 * size_of::<Result<SealedOwnedStorage, Error>>(),
                2 * size_of::<SealedOwnedStorage>(),
            )?,
        )?;
        let mut charge = self.produced.finish_lease()?;
        grow(&mut charge, &self, controls)?;
        // Reserve acceptance/Attempt controls under the ORIGINAL lease as well;
        // no state change until all OPEN-only reader preparation is finished.
        let finishing = self
            .produced
            .prepare_compact_history(size_of::<request_origin::schedule::Attempt>())?;
        let root = Arc::new(self);
        // Failure burns before root payload/leases retire (reverse local order).
        let attempt = finishing;
        let prefix = root.compact_canonical_prefix()?;
        let reader = HistorySchedule::new(root.as_ref())?;
        reader.prepay_finish()?;
        let resolutions = root
            .produced
            .produced
            .epochs()
            .iter()
            .try_fold(0usize, |n, e| {
                add(n, e.recipe_count().checked_mul(4).ok_or(Error::Bounds)?)
            })?;
        root.prepay_retained_resolution(resolutions)?;
        let outputs = root.produced.produced.outputs();
        let inputs = root.produced.checked_epoch_imports_for_finish()?;
        // Actual prefix declarations only; no synthesized expanded nodes.
        for ordinal in 0..reader.node_count() {
            let node = reader.node(ordinal)?;
            let declaration = matches!(
                node.recipe,
                schedule_finalization::BorrowedRecipe::Input(_)
                    | schedule_finalization::BorrowedRecipe::PublicInput { .. }
            );
            if declaration != (ordinal < inputs) {
                return Err(Error::Binding);
            }
        }
        let baseline = root.produced.source_plan().budget().used_bytes();
        let q = schedule_finalization::quote(&reader, &outputs, baseline)?;
        let cap = root.produced.program_cap();
        q.admit(cap)?;
        let phase = q.peak_bytes.checked_sub(baseline).ok_or(Error::Binding)?;
        let map_bytes = add(
            array::<usize>(reader.node_count())?
                .checked_mul(2)
                .ok_or(Error::Bounds)?,
            array::<u8>(reader.node_count())?,
        )?;
        let (owner_layout, _) = Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
            .extend(Layout::new::<FinalizedRoot>())
            .map_err(|_| Error::Bounds)?;
        let retained_reader = reader.retained_bytes()?;
        grow(
            &mut charge,
            &root,
            add(
                add(phase, map_bytes)?,
                add(retained_reader, owner_layout.pad_to_align().size())?,
            )?,
        )?;
        let visits = reader.node_count().checked_mul(32).ok_or(Error::Bounds)?;
        root.produced.prepay_import_metadata(visits)?;
        // Original ordering: authenticated acceptance COMPLETE, THEN canonical
        // finalization under the original armed Attempt, THEN publication later.
        root.produced.complete_compact_history(&attempt)?;
        let finalized = schedule_finalization::finalize(&reader, &outputs, &prefix, baseline, cap)?;
        if finalized.work != root.produced.work() {
            return Err(Error::Binding);
        }
        let (slots, slot_count) = slots(&finalized.retire, inputs)?;
        let owner = Arc::new(FinalizedRoot {
            slots,
            finalized,
            root: Arc::clone(&root),
            outputs,
            inputs,
            slot_count,
        });
        let recipes = reader.into_retained(Arc::clone(&owner))?;
        drop(root);
        Ok(SealedOwnedStorage {
            attempt,
            recipes,
            owner,
            _charge: charge,
        })
    }
}

impl SealedOwnedStorage {
    pub(in crate::replicated_decoder) fn snapshot_directory(
        &self,
    ) -> Result<&Arc<source_entry::snapshot_directory::OwnedSnapshotDirectory>, Error> {
        self.check_completed()?;
        self.owner.root.directory()
    }
    #[cfg(test)]
    pub(in crate::replicated_decoder::program) fn test_argmax_reuse(
        &self,
    ) -> Result<(usize, usize), Error> {
        let [first, second] = self.owner.root.produced.produced.epochs() else {
            return Err(Error::Binding);
        };
        let mut reused = 0;
        for (lanes, bounds, program) in second.test_negative_programs() {
            if !Arc::ptr_eq(
                first
                    .previous_negative(lanes, bounds)
                    .ok_or(Error::Binding)?,
                program,
            ) {
                return Err(Error::Binding);
            }
            reused += 1;
        }
        if second.work().operations() <= first.work().operations() {
            return Err(Error::Binding);
        }
        let expected = second.work().checked_add(WorkBudget::product(2, false)?)?;
        if expected != self.work() {
            return Err(Error::Binding);
        }
        Ok((first.test_negative_programs().count(), reused))
    }
    #[cfg(test)]
    pub(in crate::replicated_decoder::program) fn test_epoch_boundary(
        &self,
        index: usize,
    ) -> Result<(usize, Option<usize>), Error> {
        Ok(self
            .owner
            .root
            .produced
            .produced
            .epochs()
            .get(index)
            .ok_or(Error::Binding)?
            .test_boundary())
    }
    pub(in crate::replicated_decoder::program) fn generated_len(
        &self,
        ordinal: usize,
    ) -> Result<usize, Error> {
        let node = self.read_recipe(ordinal)?;
        let schedule_finalization::BorrowedRecipe::GeneratedPublicValues(values) = node.recipe
        else {
            return Err(Error::Binding);
        };
        values.retained_len(node.lanes, node.bounds)
    }
    pub(in crate::replicated_decoder::program) fn generated_element(
        &self,
        ordinal: usize,
        index: usize,
    ) -> Result<i64, Error> {
        let node = self.read_recipe(ordinal)?;
        let schedule_finalization::BorrowedRecipe::GeneratedPublicValues(values) = node.recipe
        else {
            return Err(Error::Binding);
        };
        values.retained_element(node.lanes, node.bounds, index)
    }
    pub(in crate::replicated_decoder) fn public(&self) -> &Arc<PublicSource> {
        &self.owner.root.public.public
    }
    pub(in crate::replicated_decoder) fn retained_bytes(&self) -> usize {
        self.owner.root.produced.source_plan().budget().used_bytes()
    }
    pub(in crate::replicated_decoder) fn check_completed(&self) -> Result<(), Error> {
        self.owner
            .root
            .produced
            .check_completed_attempt(&self.attempt)
    }
    pub(in crate::replicated_decoder::program) fn into_program(
        self,
    ) -> Result<Arc<Program>, Error> {
        self.publish(false)
    }
    pub(in crate::replicated_decoder) fn into_source_program(self) -> Result<Arc<Program>, Error> {
        self.publish(true)
    }
    pub(in crate::replicated_decoder) fn collection_phase(
        &self,
        guard: &mut OwnedExternalCharge,
        bytes: usize,
        visits: u64,
    ) -> Result<(), Error> {
        self.check_completed()?;
        crate::replicated_decoder::program::source_entry::ingress::resize(
            &self.owner.root.original,
            guard,
            bytes,
            visits,
        )
    }
    pub(in crate::replicated_decoder) fn source_manifest(
        &self,
    ) -> Result<&crate::replicated_decoder::Manifest, Error> {
        self.check_completed()?;
        Ok(self.owner.root.manifest.payload())
    }
    fn publish(mut self, source_bound: bool) -> Result<Arc<Program>, Error> {
        self.check_completed()?;
        let declared = self.owner.root.manifest.payload().program;
        let source = self.owner.root.manifest.payload().source;
        let composition = self.owner.root.manifest.payload().composition;
        let (layout, _) = Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
            .extend(Layout::new::<Program>())
            .map_err(|_| Error::Bounds)?;
        let controls = add(
            layout.pad_to_align().size(),
            add(
                size_of::<request_origin::schedule::Custody>(),
                add(
                    array::<(usize, usize)>(3)?,
                    2 * size_of::<Result<Arc<Program>, Error>>(),
                )?,
            )?,
        )?;
        #[cfg(test)]
        PUBLICATION_OBSERVER.with(|slot| {
            if let Some(observer) = slot.borrow_mut().take() {
                let required = self
                    ._charge
                    .budget()
                    .used_bytes()
                    .checked_add(controls)
                    .expect("test publication quote overflow");
                observer(required);
            }
        });
        grow(&mut self._charge, self.owner.custody(), controls)?;
        self.owner
            .custody()
            .produced
            .prepay_publication(&self.attempt, 64)?;
        self.owner
            .custody()
            .prepay_retained_resolution(add(self.input_count(), 1)?)?;
        let mut outputs = reserve(3)?;
        outputs.extend_from_slice(self.outputs());
        let expected_len = self.logical_len();
        let inputs = self.input_count();
        let digest = self.digest();
        let work = self.work();
        let binding = self.source_binding();
        let payload_bytes = self.retained_bytes();
        let custody = Box::new(request_origin::schedule::Custody::Compact(self));
        let mut program = Arc::new(Program {
            nodes: Vec::new(),
            inputs,
            outputs,
            retire: Vec::new(),
            digest,
            payload_bytes,
            work,
            source_binding: binding,
            request_custody: Some(custody),
        });
        if source_bound {
            Arc::get_mut(&mut program)
                .ok_or(Error::Binding)?
                .bind(source, composition);
        }
        if declared != [0; 32] && declared != program.digest() {
            return Err(Error::Binding);
        }
        // Adapter must read this actual owner before startup/publication. Until
        // its private delegation is mounted this is a typed refusal, not fallback.
        if program.schedule_access().len() != expected_len {
            return Err(Error::Binding);
        }
        program
            .request_custody
            .as_ref()
            .ok_or(Error::Binding)?
            .public()
            .check(&program)?;
        let program_mut = Arc::get_mut(&mut program).ok_or(Error::Binding)?;
        let request_origin::schedule::Custody::Compact(storage) = program_mut
            .request_custody
            .as_mut()
            .ok_or(Error::Binding)?
            .as_mut()
        else {
            return Err(Error::Binding);
        };
        storage.attempt.commit_publication();
        Ok(program)
    }
    pub(in crate::replicated_decoder::program) fn input_count(&self) -> usize {
        self.owner.inputs
    }
    pub(in crate::replicated_decoder::program) fn logical_len(&self) -> usize {
        self.owner.slots.len()
    }
    pub(in crate::replicated_decoder::program) fn value_slot_count(&self) -> usize {
        self.owner.slot_count
    }
    pub(in crate::replicated_decoder::program) fn value_slot(
        &self,
        logical: usize,
    ) -> Result<usize, Error> {
        self.owner.slots.get(logical).copied().ok_or(Error::Binding)
    }
    pub(in crate::replicated_decoder::program) fn retirement(
        &self,
        ordinal: usize,
    ) -> Result<&[usize], Error> {
        self.owner
            .finalized
            .retire
            .get(ordinal)
            .map(Vec::as_slice)
            .ok_or(Error::Binding)
    }
    pub(in crate::replicated_decoder::program) fn outputs(&self) -> &[(usize, usize)] {
        &self.owner.outputs
    }
    pub(in crate::replicated_decoder::program) fn source_binding(&self) -> [u8; 32] {
        self.owner.root.binding
    }
    pub(in crate::replicated_decoder::program) fn digest(&self) -> [u8; 32] {
        self.owner.finalized.digest
    }
    pub(in crate::replicated_decoder::program) fn work(&self) -> WorkBudget {
        self.owner.finalized.work
    }
    pub(in crate::replicated_decoder::program) fn read_recipe(
        &self,
        ordinal: usize,
    ) -> Result<schedule_finalization::BorrowedNode<'_>, Error> {
        self.check_completed()?;
        self.recipes.node(ordinal)
    }
    pub(in crate::replicated_decoder::program) fn selection_owner(
        &self,
        _ordinal: usize,
    ) -> Result<&Arc<runtime::selection::Plan>, Error> {
        // Current source Argmax retains negative tensor kernels, NOT Selection.
        Err(Error::Binding)
    }
    pub(in crate::replicated_decoder::program) fn trie_owner(
        &self,
        _ordinal: usize,
    ) -> Result<&Arc<runtime::trie::Plan>, Error> {
        // Original source lookup uses EqualZero/PublicProduct, never a trie.
        Err(Error::Binding)
    }
}
