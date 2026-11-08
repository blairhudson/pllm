//! PRIVATE, INACTIVE ASTRA SOURCE-REVIEW CANDIDATE: primary 65, independent 68.
//!
//! This file is deliberately absent from every mod/include/feature/test graph.
//! It is not compile-ready, an integration proposal, or permission to activate
//! a route. Existing dirty source, quotes, caps, tests and native HOLD are untouched.
//! Names below identify actual source owners, including inaccessible private
//! types. No production visibility changes are made to make this file work.
//!
//! BODY COMPLETENESS
//! Implemented here: detached/root/loan/context shapes; bounded private controller
//! callback; root-derived view; OPEN/COMPLETE association checks; import binding
//! and inspection admission; ingress/canonical prefix; finish_lease;
//! complete_compact_history. These are
//! source bodies, not compiler/lifetime evidence. Constructor, conversion, history
//! migration, prepare_compact_history, driver and seal are BLOCKED as described
//! below and intentionally have no callable substitute, empty body or stub.
//! No method manufactures an old OwnedEmission by temporarily attaching a guard.
//!
//! FREEZE / REQUIRED EXTRACTIONS (not performed)
//! 1. source_entry.rs Pending (689-701), CreatedInputs (324-331), create_inputs,
//!    quote_imports and PublicSnapshotOwner; compact_pending consume_transition
//!    (303-459). Author the real constructor in that source-owning privacy scope.
//!    Keep partially consumed Pending.controller in its ORIGINAL field while
//!    registry, boxed manifest, source scope and actual CreatedInputs are alive.
//!    Keep ImportStage's importer-before-import-charge retirement. No callback may
//!    capture payload owners merely to simulate field order. Authentic CreatedInputs
//!    must still be minted only by create_inputs; never reconstruct its marker.
//!    Preserve map-address checks, public.check_registry, quote/build/prepay,
//!    import resizing and create_inputs order. Before moving owners into Root,
//!    end all lexical loans and move the same controller, without refund/rebuy.
//! 2. compact_request_storage.rs EmissionContext/EmissionScope/check_initial_parts
//!    and compact_request_history.rs PendingCompactHistory must share the actual
//!    controller-loan context, not accept a new adapter trait. LoanContext below
//!    replaces the owned guard only; every check, prepay, retention, resize,
//!    admission, work debit and EmissionAttempt burn/commit stays in original order.
//!    EmissionAttempt currently names EmissionContext: it also needs extraction.
//!    CompactGraph, SourceScope::check_context, EpochAssociation::check_context,
//!    completed_feedback and dependent lowering signatures require the same
//!    concrete context migration. No new issuer or same-budget pairing API.
//! 3. compact_request_owned.rs into_owned (399-505): extract the owning payload
//!    conversion at its original privacy owner. Preserve checks at 411/414,
//!    admit_phase then retain then convert then check_imports (429-434).
//!    Destructure actual RequestInputPermit/History/RequestSources as at 460-478;
//!    history_public still drops there. Move all actual fields to DetachedEmission
//!    in order below. End the ControllerLoan instead of guard.into_inner(). No
//!    final detached result may contain loan, context, registry or manifest borrow.
//!    Do not first create old OwnedEmission, then detach its original controller.
//! 4. compact_owned_epoch.rs 407-417 accepts only old OwnedEmission<R>.
//!    Extract a concrete authenticated view check at the epoch's privacy owner:
//!    SAME EpochAssociation origin/plan/epoch/prefix-allocation checks via
//!    checked_epoch_imports AND SAME self.instance == instance check. Merely
//!    checking epoch numbers or budget identity is not equivalent. prepare below
//!    cannot call this old method with EmissionView. Do not fake a bridge trait.
//! 5. compact_owned_epoch.rs 98-116/139-227/291-311/315 onward: migrate concrete
//!    HistoryCustody, InspectionRoot and InspectionLoan, preserving source-root,
//!    instance, map identity, import key/shape/digest/bounds/statistics, prepay and
//!    inspection-attempt semantics. Returned loans must use Root's lifetime, not
//!    the lifetime of a temporary EmissionView. Retain pure recipe traversal.
//!    source_entry_compact_handles.rs SourceScope::new/check_owned/quote are
//!    parent-private; SourceCustody's source resolution implementations there
//!    need extraction for this actual root, not caller-selected manifests.
//! 6. compact_owned_recipe.rs HistorySchedule (450-454,619-772), RetainedEpoch,
//!    RetainedRecipes and FinalizedRoot are concretely old-root typed. Migrate
//!    their source-owning definitions together; no parallel recipe interpreter.
//!    Preserve into_retained 645 pointer equality of reader.custody and
//!    owner.custody(), then source/key pointer checks at 674-695. SAME allocation
//!    Arc<Root> must reach reader, FinalizedRoot and retained recipes. Keep actual
//!    Arc::clone points: seal owner root; reader.into_retained owner; per-retained
//!    epoch owner and per-gather epoch. No reconstructed Root/View as authority.
//!    schedule_finalization.rs 72-103 also seals generated-value issuance to
//!    concrete RetainedEpoch/InspectionLoan types. That dependency needs concrete
//!    migration, not field-visibility widening or issuance promotion.
//! 7. compact_request_sealed.rs SealedOwnedStorage's private owner field is
//!    Arc<production FinalizedRoot>; that FinalizedRoot.root is Arc<OLD Root>.
//!    Therefore this new Root CANNOT honestly return the existing actual
//!    Result<sealed::SealedOwnedStorage, Error> without production extraction.
//!    No replacement storage type, unchecked constructor, attach, infallible
//!    commit, or placeholder seal is provided. FinalizedRoot field order must
//!    remain slots, finalized, root, outputs, inputs, slot_count; storage order
//!    must remain attempt, recipes, owner, _charge. Boundary ends at storage
//!    return, not into_program. Nevertheless preserve completed checks at sealed
//!    317-321 and publication prepay at 381-385 when later extracting; COMPLETE
//!    checks must never call OPEN-only inspection or reopen the origin.
//!
//! FROZEN BOUNDED DRIVER / SEAL BODY DEPENDENCY MAP
//! Driver return: Result<sealed::SealedOwnedStorage, Error>, not a new authority.
//! source_entry_compact_pending.rs 249-267 remains the exact bounded itinerary:
//! limits from genuine pending; consume actual input/loan callback; construct
//! loan-backed scope; first graph from actual imports/context; lower exactly the
//! original prefill.operations; freeze_owned(Vec::new(), source); continue_original
//! with first epoch and limits; checked_graph_parts count; second graph from
//! actual pending history; lower exactly count; freeze_owned; history.finish into
//! detached history; then custody.seal_compact. No caller supplies view/controller.
//! Constructor and both histories are BLOCKED by 1-5; no driver body is claimed.
//!
//! Root.prepare_compact_history(&mut self, bytes) -> Result<schedule::Attempt,
//! Error> is BLOCKED by 4 and Attempt field privacy. Exact mutable body to extract
//! from compact_request_owned.rs 313-380: check_open; checked epochs*4+token visits;
//! prepay; epoch/rows/output/epoch-count/token-shape/import-state checks; genuine
//! epoch association checks; selected-word checks; final sticky; epoch[1] work +
//! Product2 equality; used-minus-original-controller payload; checked payload +
//! bytes and cap check; resize ORIGINAL controller; create original armed Attempt.
//! Field splitting is internal only. finish_lease and this resize precede Arc::new.
//!
//! seal_compact(self) must extract compact_request_sealed.rs 139-234 unchanged in
//! semantic order: check_ingress plus check_source_custody; Root Arc layout and
//! whole Result/storage controls; finish_lease; first grow of separate finish
//! lease; prepare history; Arc::new(self); move finishing to attempt AFTER root;
//! canonical prefix; HistorySchedule::new(&same root); prepay_finish; recipe*4
//! resolution prepay; actual outputs/import-count; prefix declaration loop;
//! baseline; quote/admit; phase/map/layout/retained-reader controls; second grow
//! of EXISTING separate finish lease; node_count*32 visits; COMPLETE before
//! finalize; work equality; actual slots allocation/validation; owner allocation
//! with Arc::clone(&root); reader.into_retained(Arc::clone(&owner)); drop(root);
//! Ok(SealedOwnedStorage { attempt, recipes, owner, _charge: charge }). Every
//! fallible seal operation must retain this same complete Root (before Arc as
//! self, afterwards via that Arc); no source payload leaves it. Only COMPLETE
//! mutates origin after Arc creation. Later grow never mutates originalcontroller.
//! This itinerary is NOT a completed seal body: blockers 4-7 prevent one here.
//!
//! QUOTE / TYPE ASSIGNMENT: NO NUMERIC OR FUNDING CLAIM
//! P0: pending_controller_bytes' old three Pending envelopes are ALL assigned:
//! ingress staging, pending assembly/transfer, destination Pending. None spare
//! for ControllerLoan, CreatedInputs forwarding, loan-scope constructor results,
//! detached returns, Root results, view or driver. Original root handle is not
//! funding. No bootstrap allowance, padding, free credit or attach expression.
//! P1: old context_bytes' 3*EmissionContext + Result<EmissionContext,Error> becomes
//! 3*LoanContext + Result<LoanContext,Error> only in a future actual extraction.
//! Concrete slots: context construction, scope-contained context, context move;
//! whole constructor Result. This is a logical phase assignment, not stack proof.
//! Existing separate OwnedExternalCharge and dispatch/map/PublishedView terms
//! keep their named roles; do not relabel them as unproved loan/view coverage.
//! P2: into_owned's 2*OwnedEmission<R> + Builder + 2*Box<Builder> becomes
//! 2*DetachedEmission<R> + Builder + 2*Box<Builder>: conversion local/destination,
//! builder source and box forwarding/destination. Whole
//! Result<DetachedEmission<CompletionLoan>,Error> from history.finish and its
//! callback forwarding are UNRESOLVED, not sums of success/error variants.
//! P3: SourceCatalog::quote<R> has ONE type parameter, not a new generic arity.
//! Current production history instantiates R as OwnedEmission<CompletionLoan>.
//! Detached History is ONLY the proposed candidate instantiation, not current R
//! (nor Result<History,Error> or an invented history tuple). quote<R> still sizes
//! production SourceCustody<R>. Substitution with candidate
//! SourceCustody<History,OwnedExternalCharge> is not implemented and needs approval.
//! Merely quote::<History> today still sizes the OLD root and is wrong.
//! Old 2*SourceCustody are local/destination Root
//! envelopes. Old 2*Controls<R> remain catalog quote/build traversal envelopes;
//! Controls.custody_move is PhantomData<R>, so it funds no R or Root Result.
//! Result<Root,Error> consume_transition/consume_compact forwarding: UNRESOLVED.
//! P4: seal controls' Root Arc backing occurs ONCE (atomic header + actual Root
//! layout). 2*Result<SealedOwnedStorage,Error> and 2*SealedOwnedStorage retain
//! seal-return/driver-forwarding and assembly/destination roles. Actual backing
//! changes Root size. FinalizedRoot layout and retained_bytes' whole
//! Result<RetainedRecipes,Error>, retained epoch layouts and lens/gather arrays
//! retain original phase roles, but must use migrated actual concrete types.
//! No extra root allocation is inferred from Arc clones.
//! P5: exact additional occurrences needing assignment, all UNRESOLVED:
//! ControllerLoan construction/callback move/LoanContext destination (P1);
//! actual CreatedInputs argument/constructor forwarding and
//! Result<loan-backed EmissionScope,Error> (P1); for<'s> callback's whole
//! Result<History,Error> plus conversion/history finish forwarding (P2);
//! consume_transition and consume_compact whole Result<Root,Error> (P3);
//! limits/closure driver controls and graph/history Result carriers (P1/P2/P4);
//! Root.emission temporary at every OPEN/import/inspection/completion/prepay
//! call and checked borrowed-return Result carriers (P2/P4); concrete migrated
//! InspectionLoan, InspectionRoot, OwnedSchedule, HistorySchedule constructors,
//! their forwarding whole Results and retained-reader adaptation (P4).
//! RetainedGather, SourceLenses/generated-value Result carriers, and helper
//! InspectionAttempt/ExternalCharge whole Results also remain UNRESOLVED (P4).
//! Naming these carriers does not establish coverage.
//! Per-call occurrence counts remain UNRESOLVED until those actual bodies freeze;
//! this list is not an exhaustive carrier census or simultaneous-peak proof.
//! No production quote is edited. Unchanged formulas do NOT imply unchanged
//! numerical amounts: context/conversion/catalog/seal/recipe depend on actual
//! size_of types. Old attach sizing is invalid and RETIRED, not reusable credit.
//!
//! LIMIT: Root's covered payload SLOTS precede originalcontroller. Root.root is
//! deliberately AFTER controller. This does NOT claim all owners outlive that
//! controller. Arc-pointee escape, unwind and whole-lifetime behavior are outside
//! this contract. Type/body freeze is not census exhaustiveness, compilation,
//! lifetimes, funding, metric, physical-resource or execution evidence. No tests,
//! parser, formatter, build, sizing, native command or network was run.

use crate::replicated_decoder::program::{self, request_origin, source_entry};
use request_origin::compact_storage::{compact_history, owning};
use request_origin::{HistoryOrigin, Resources};
use source_entry::compact_pending::handles::SourceCatalog;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;

// These aliases deliberately name actual private source types. Their privacy is
// an extraction blocker, not a request to widen production visibility.
use program::{Builder, Error, Manifest, OriginalSourcePlan, PublicSource, SourceView, View, Words};
use request_origin::{Fragment, RequestInputPermit};
use source_entry::{OwnedBudget, OwnedCharged, OwnedExternalCharge, PublicSnapshotOwner, SourceIngressRaw};
use crate::replicated_decoder::source_registry;
use request_origin::rotary_creation;
use pllm_core::puma::WorkBudget;

// EXACT current OwnedEmission 68-88 payload/history/import-charge order, deleting
// only guard at line 87. No invented tuple replaces actual history fields.
struct DetachedEmission<R> {
    produced: R,
    imports: Box<Builder>,
    inputs: BTreeMap<String, SourceView>,
    epochs: Vec<Arc<OriginalSourcePlan>>,
    fragments: Vec<Fragment>,
    projections: Vec<Option<rotary_creation::CreatedOperator>>,
    scratch: Words,
    origin: HistoryOrigin,
    public: Arc<PublicSource>,
    resources: Resources,
    sticky: Option<usize>,
    tokens: View,
    plan: Arc<OriginalSourcePlan>,
    epoch: usize,
    rows: usize,
    cap: usize,
    historical_peak: usize,
    work: WorkBudget,
    history_charge: OwnedExternalCharge,
    imports_charge: OwnedExternalCharge,
}

type History = DetachedEmission<compact_history::CompletionLoan>;
type Root = SourceCustody<History, OwnedExternalCharge>;

// Preserve production SourceCustody's produced-first payload order, including
// HistoryOrigin destruction/burn order. DetachedEmission contains no controller;
// originalcontroller follows the existing charges, immediately before root last.
struct SourceCustody<R, C> {
    detached: R,
    original: Arc<OriginalSourcePlan>,
    public: PublicSnapshotOwner,
    manifest: Box<OwnedCharged<Manifest>>,
    raw: SourceIngressRaw,
    directory: Option<Arc<source_entry::snapshot_directory::OwnedSnapshotDirectory>>,
    catalog: SourceCatalog,
    instance: source_registry::Instance,
    binding: [u8; 32],
    registry_charge: OwnedExternalCharge,
    manifest_box_charge: OwnedExternalCharge,
    catalog_charge: OwnedExternalCharge,
    originalcontroller: C,
    root: OwnedBudget,
}

// Private loan is the only cell containing the original mutable reference.
// No cell/guard accessor, extraction, replace, split or mutable public API.
struct ControllerLoan<'s> {
    controller: RefCell<&'s mut OwnedExternalCharge>,
}

// HRTB prevents only fresh controller-loan escape through R. It does not prove
// scope lifetimes for the absent CreatedInputs/registry/manifest constructor.
// That real constructor still needs the source-scope boundary described above.
// Provenance here is the actual pending.controller field borrow, not an extra
// policy check or a pair of independently supplied budget references.
fn with_pending_controller<R>(
    pending: &mut source_entry::Pending,
    callback: impl for<'s> FnOnce(ControllerLoan<'s>) -> Result<R, Error>,
) -> Result<R, Error> {
    callback(ControllerLoan {
        controller: RefCell::new(&mut pending.controller),
    })
}

// Concrete current EmissionContext fields, with controller loan replacing owned
// guard. No scope/history ever owns originalcontroller. Methods cannot delegate
// to production EmissionContext without the concrete extraction above.
struct LoanContext<'s> {
    manifest: &'s Manifest,
    registry: &'s source_registry::SourceRegistry<'s>,
    public: Arc<PublicSource>,
    plan: Arc<OriginalSourcePlan>,
    origin: Arc<OwnedCharged<request_origin::Origin>>,
    epoch: usize,
    rows: usize,
    prefix_allocation: usize,
    historical_peak: Cell<usize>,
    retained_payload: Cell<usize>,
    registered_builder: Cell<usize>,
    cap: usize,
    history_input: Option<compact_history::HistoryInput>,
    work: RefCell<WorkBudget>,
    request: RequestInputPermit<'s>,
    loan: ControllerLoan<'s>,
}

struct EmissionView<'r, R> {
    payload: &'r DetachedEmission<R>,
    controller: &'r OwnedExternalCharge,
}

impl<R> SourceCustody<DetachedEmission<R>, OwnedExternalCharge> {
    // This method derives both references from the same authentic root.
    // EmissionView fields have module privacy, not exclusive-constructor
    // enforcement against other code in this module.
    fn emission(&self) -> EmissionView<'_, R> {
        EmissionView {
            payload: &self.detached,
            controller: &self.originalcontroller,
        }
    }
}

impl<'r, R> EmissionView<'r, R> {
    fn check_open(&self) -> Result<(), Error> {
        let p = self.payload;
        if p.origin.payload().state.load(Ordering::Acquire) != request_origin::OPEN {
            return Err(Error::Binding);
        }
        if !Arc::ptr_eq(&p.public, &p.origin.payload().public)
            || p.epochs.get(p.epoch).is_none_or(|plan| !Arc::ptr_eq(plan, &p.plan))
            || !self.controller.is_for(p.plan.budget())
            || !p.history_charge.is_for(p.plan.budget())
            || !p.imports_charge.is_for(p.plan.budget())
        {
            return Err(Error::Binding);
        }
        Ok(())
    }

    // Explicit 'r keeps returned imports tied to Root, not temporary &self.
    fn checked_imports(&self, association: &owning::EpochAssociation) -> Result<&'r Builder, Error> {
        self.check_open()?;
        let p = self.payload;
        let origin = p.origin.payload();
        if !Arc::ptr_eq(&association.origin, &p.origin.0)
            || p.epochs.get(association.epoch).is_none_or(|plan| !Arc::ptr_eq(&association.plan, plan))
            || association.prefix_allocation != p.imports.nodes.as_ptr() as usize
            || !p.plan.same_origin(&origin.source)
            || !p.imports.original_inputs.iter().any(|input| Arc::ptr_eq(input, &origin.input))
        {
            return Err(Error::Binding);
        }
        let input = p.imports.nodes.get(origin.input.owner()).ok_or(Error::Binding)?;
        if !matches!(&input.code, program::Code::Input(key) if key == "input.tokens")
            || input.lanes != p.resources.rows
            || input.bounds != origin.bounds
            || p.imports.nodes.len() != origin.input.owner().checked_add(1).ok_or(Error::Bounds)?
        {
            return Err(Error::Binding);
        }
        Ok(&p.imports)
    }

    fn checked_epoch_imports(&self, association: &owning::EpochAssociation, epoch: usize) -> Result<&'r Builder, Error> {
        if association.epoch != epoch {
            return Err(Error::Binding);
        }
        self.checked_imports(association)
    }

    fn check_source_custody(&self, manifest: &Manifest, public: &Arc<PublicSource>, instance: source_registry::Instance) -> Result<(), Error> {
        self.check_open()?;
        let origin = self.payload.origin.payload();
        if origin.registry_identity != instance
            || !Arc::ptr_eq(public, &self.payload.public)
            || origin.source_records != std::ptr::from_ref(&manifest.tensors) as usize
        {
            return Err(Error::Binding);
        }
        Ok(())
    }

    fn prepay_import_metadata(&self, visits: usize) -> Result<(), Error> {
        self.check_open()?;
        self.payload.plan.prepay_compiler_metadata(
            self.controller,
            u64::try_from(visits).map_err(|_| Error::Bounds)?,
        ).map_err(|_| Error::Bounds)
    }

    fn checked_import_binding(&self, owner: usize) -> Result<(&'r str, &'r [usize]), Error> {
        self.check_open()?;
        self.prepay_import_metadata(self.payload.inputs.len().checked_add(1).ok_or(Error::Bounds)?)?;
        let mut bindings = self.payload.inputs.iter().filter(|(_, view)| view.owner == owner);
        let (key, view) = bindings.next().ok_or(Error::Binding)?;
        if bindings.next().is_some() {
            return Err(Error::Binding);
        }
        Ok((key.as_str(), view.shape.as_slice()))
    }

    fn inspection_attempt(&self) -> Result<owning::InspectionAttempt<'r>, Error> {
        self.check_open()?;
        Ok(owning::InspectionAttempt {
            state: &self.payload.origin.payload().state,
            committed: false,
        })
    }

    fn charge_inspection(&self, bytes: usize, visits: u64) -> Result<pllm_garble::shared_allocation::ExternalCharge<'r>, Error> {
        let mut attempt = self.inspection_attempt()?;
        let p = self.payload;
        let total = p.plan.budget().used_bytes().checked_add(bytes)
            .and_then(|n| n.checked_add(std::mem::size_of::<pllm_garble::shared_allocation::ExternalCharge<'_>>()))
            .ok_or(Error::Bounds)?;
        if total > p.cap.min(program::MAX_PROGRAM_BYTES) {
            return Err(Error::Bounds);
        }
        p.plan.prepay_compiler_metadata(self.controller, visits).map_err(|_| Error::Bounds)?;
        let charge = p.plan.budget().charge_external(bytes).map_err(|_| Error::Bounds)?;
        attempt.commit();
        Ok(charge)
    }
}

impl<'r> EmissionView<'r, compact_history::CompletionLoan> {
    fn completed_imports(&self) -> Result<&'r Builder, Error> {
        let p = self.payload;
        if p.origin.payload().state.load(Ordering::Acquire) != request_origin::COMPLETE
            || !Arc::ptr_eq(&p.public, &p.origin.payload().public)
            || p.epochs.get(p.epoch).is_none_or(|plan| !Arc::ptr_eq(plan, &p.plan))
            || !self.controller.is_for(p.plan.budget())
            || !p.imports_charge.is_for(p.plan.budget())
            || !p.history_charge.is_for(p.plan.budget())
        {
            return Err(Error::Binding);
        }
        Ok(&p.imports)
    }

    fn check_completed_attempt(&self, attempt: &request_origin::schedule::Attempt) -> Result<(), Error> {
        if !Arc::ptr_eq(&attempt.origin, &self.payload.origin.0) {
            return Err(Error::Binding);
        }
        self.completed_imports()?;
        Ok(())
    }

    fn prepay_publication(&self, attempt: &request_origin::schedule::Attempt, visits: u64) -> Result<(), Error> {
        self.check_completed_attempt(attempt)?;
        self.payload.plan.prepay_compiler_metadata(self.controller, visits).map_err(|_| Error::Bounds)
    }
}

impl Root {
    fn check_ingress(&self, instance: source_registry::Instance, binding: [u8; 32]) -> Result<(), Error> {
        let budget = self.root.budget();
        if self.instance != instance
            || self.binding != binding
            || !std::ptr::eq(self.original.budget(), budget)
            || !std::ptr::eq(self.manifest.budget(), budget)
            || !std::ptr::eq(self.raw.budget(), budget)
            || !self.public.charge.guard.is_for(budget)
            || !self.registry_charge.is_for(budget)
            || !self.manifest_box_charge.is_for(budget)
            || !self.catalog_charge.is_for(budget)
        {
            return Err(Error::Binding);
        }
        self.catalog.check_owned(self.manifest.payload(), &self.original, &self.catalog_charge)
    }

    fn check_emission(&self) -> Result<(), Error> {
        self.check_ingress(self.instance, self.binding)?;
        self.emission().check_source_custody(self.manifest.payload(), &self.public.public, self.instance)
    }

    fn compact_canonical_prefix(&self) -> Result<sha2::Sha256, Error> {
        self.check_emission()?;
        request_origin::schedule::canonical_prefix(
            &self.original,
            self.detached.resources.rows,
            self.detached.resources.outputs,
            self.manifest.payload(),
        )
    }

    fn finish_lease(&mut self) -> Result<OwnedExternalCharge, Error> {
        self.emission().check_open()?;
        self.originalcontroller.split_owned(0).map_err(|_| Error::Bounds)
    }

    fn complete_compact_history(&self, attempt: &request_origin::schedule::Attempt) -> Result<(), Error> {
        self.emission().check_open()?;
        if !Arc::ptr_eq(&attempt.origin, &self.detached.origin.0) || attempt.committed {
            return Err(Error::Binding);
        }
        self.detached.origin.payload().state.compare_exchange(
            request_origin::OPEN,
            request_origin::COMPLETE,
            Ordering::AcqRel,
            Ordering::Acquire,
        ).map_err(|_| Error::Binding)?;
        Ok(())
    }
}
