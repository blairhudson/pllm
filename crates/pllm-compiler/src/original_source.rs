//! Actual config-factory provenance; not checkpoint authentication or execution.
//!
//! Configured authority is born only after the registered model factory succeeds
//! under its original counter. Neither a DecoderPlan, a digest, nor a parsed plan
//! can be upgraded here. The generic JSON frontend supplies its intact original
//! byte/Value owner; it does not certify a publisher, weights, tokens or feedback.
//! Those existing frontend/import/acceptance gates remain independent.
//! Work covers the qualified default/std logical slices, not complete CPU fuel.
//! Opaque numeric/formatter internals, dependency/compiled callframes, allocator
//! overhead and RSS remain unqualified. This owner does not promote their quote
//! into whole-request, source-authentication, runtime or drain admission.
//!
//! Source-plan metadata and configuration are immutable. The root's sole mutable
//! state after one-shot configuration is its cumulative construction-work ledger: independent reservations
//! are prepaid, never refunded or lent. Failure/unwind burns further lowering.
//! A derived plan retains the original root (including its first actual plan),
//! conservatively retaining all original guard excess rather than taking credit.

use crate::plan_input::json::{self, ImportedValue};
use pllm_garble::shared_allocation::{
    Budget, Error as NativeError, OwnedCharged, OwnedExternalCharge, MAX_ALLOCATION_BYTES,
};
use pllm_models::{
    owned_backing::bounded as census,
    source_work::{self, SourceInputLimits, SourceWorkQuote},
};
use pllm_models::{
    ConfigLowererLimits, ConfigLowererQuote, DecoderExpansionBudget, DecoderPlan, DecoderWorkload,
};
use pllm_types::bounded_canonical::borrowed::NativeFrame;
use std::{
    alloc::Layout,
    mem::size_of,
    sync::{atomic::AtomicUsize, Arc, LockResult, Mutex, MutexGuard, OnceLock},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    Counter,
    Capacity,
    Overflow,
    Work,
    Quote,
    Census,
    Burned,
    Factory,
    Native(NativeError),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("original source construction refused")
    }
}
impl std::error::Error for Error {}

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub expansion: DecoderExpansionBudget,
    pub factory: ConfigLowererLimits,
    /// Includes the original JSON owner's complete independently prepaid work.
    /// Later lowering cannot raise the original root's limit.
    pub max_work: u64,
    pub max_working_payload: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Resources {
    /// Actual incoming Value and raw-byte guards, not a caller baseline.
    pub original_owner_bytes: usize,
    pub required_working_payload: usize,
    /// Working charge plus try_build's independently admitted inline/padding.
    pub working_charge_bytes: usize,
    /// Named original owners only; unrelated siblings remain on the same Budget.
    pub original_owners_peak_bytes: usize,
    pub model_factory_bytes: usize,
    pub model_retained_bytes: usize,
    pub prepaid_work: u64,
}

enum Configuration {
    Original(OwnedCharged<ImportedValue>),
    // Always the ORIGINAL context, not an arbitrary prior plan or source hash.
    OriginalContext(Arc<OwnedCharged<Context>>),
}

#[derive(Clone, Copy)]
struct Ledger {
    used: u64,
    limit: u64,
    burned: bool,
}

struct Configured {
    // Actual plan retires before its configuration/source owners and charge.
    plan: DecoderPlan,
    // Moved from the genuine factory's successful bounded Census, never caller
    // claims. Plain structural data; Context/source custody remains authority.
    header: census::Header,
    // Captured at the genuine factory call, never reconstructed from graph data.
    // Context inline/padding and controller quotes use size_of::<Context>(), so
    // this retained field is admitted before both root and derived factories.
    workload: DecoderWorkload,
    configuration: Configuration,
    resources: Resources,
}

// The actual root exists before configuration quotation. Original Configured
// owns only the imported value, never a back-reference to this cell/root.
struct Context {
    configured: OnceLock<OwnedCharged<Configured>>,
    ledger: Option<Mutex<Ledger>>,
}

impl std::ops::Deref for Context {
    type Target = Configured;
    fn deref(&self) -> &Configured {
        self.configured.get().expect("private configured plan invariant").payload()
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // Retire configured model/raw owners before the enclosing root's
        // OwnedCharged lease. No callback, parser, lock or publication work.
        drop(self.configured.take());
        #[cfg(test)]
        observe_context_retirement();
    }
}

/// Immutable actual factory result. No public Clone, detach, mutable plan, or
/// constructor accepting a pre-existing plan/identity/digest is provided.
pub struct OriginalSourcePlan {
    inner: Arc<OwnedCharged<Context>>,
}

/// Single-use, funded primitive transition. Source custody comes only from the
/// actual original root, never the funding controller. No public construction,
/// cloning, ledger access, reset, or adoption path exists.
pub(crate) struct PrimitiveAttempt {
    root: Arc<OwnedCharged<Context>>,
    controller: Option<OwnedExternalCharge>,
    committed: bool,
}

impl PrimitiveAttempt {
    fn check_usable(&self) -> Result<(), Error> {
        let ledger = self.root.payload().ledger.as_ref().ok_or(Error::Burned)?
            .lock().map_err(|_| Error::Burned)?;
        if ledger.burned { return Err(Error::Burned); }
        Ok(())
    }
    fn restrict_limit(&self, limit: u64) -> Result<(), Error> {
        let mut ledger = self.root.payload().ledger.as_ref().ok_or(Error::Burned)?
            .lock().map_err(|_| Error::Burned)?;
        if ledger.burned { return Err(Error::Burned); }
        ledger.limit = ledger.limit.min(limit);
        if ledger.used > ledger.limit { ledger.burned = true; return Err(Error::Work); }
        Ok(())
    }
    fn debit(&self, visits: u64) -> Result<u64, Error> {
        let mut ledger = self.root.payload().ledger.as_ref().ok_or(Error::Burned)?
            .lock().map_err(|_| Error::Burned)?;
        if ledger.burned { return Err(Error::Burned); }
        ledger.burned = true;
        let used = ledger.used.checked_add(visits)
            .filter(|used| *used <= ledger.limit).ok_or(Error::Work)?;
        ledger.used = used;
        ledger.burned = false;
        Ok(used)
    }

    // Installation and final sticky check share one critical section. No
    // parser, factory, policy callback or fallible publication prep runs here.
    fn install(mut self, configured: OwnedCharged<Configured>) -> Result<OriginalSourcePlan, Error> {
        let ledger = self.root.payload().ledger.as_ref().ok_or(Error::Burned)?
            .lock().map_err(|_| Error::Burned)?;
        if ledger.burned || self.root.payload().configured.get().is_some() {
            drop(ledger);
            return Err(Error::Burned);
        }
        if let Err(configured) = self.root.payload().configured.set(configured) {
            drop(ledger);
            drop(configured);
            return Err(Error::Burned);
        }
        self.committed = true;
        let result = OriginalSourcePlan { inner: Arc::clone(&self.root) };
        drop(ledger);
        Ok(result)
    }

    /// Call only AFTER all fallible publication preparation. A sibling failure
    /// cannot be cleared here. Previously committed outputs are not revoked.
    pub(crate) fn commit(mut self) -> Result<OwnedExternalCharge, Error> {
        let ledger = self.root.payload().ledger.as_ref().ok_or(Error::Burned)?
            .lock().map_err(|_| Error::Burned)?;
        if ledger.burned {
            return Err(Error::Burned);
        }
        // Taking the original controller and marking success are one atomic
        // decision with respect to source-wide failures; no external work here.
        let controller = self.controller.take().ok_or(Error::Burned)?;
        self.committed = true;
        // Never change ledger.burned, including on successful commit.
        drop(ledger);
        Ok(controller)
    }
}

impl Drop for PrimitiveAttempt {
    fn drop(&mut self) {
        if !self.committed {
            if let Some(ledger) = self.root.payload().ledger.as_ref() {
                // Poison is itself sticky refusal. Recover only to record burn;
                // never panic in a destructor or clear mutex poison/burn state.
                let mut ledger = match ledger.lock() {
                    Ok(ledger) => ledger,
                    Err(poisoned) => poisoned.into_inner(),
                };
                ledger.burned = true;
            }
        }
        // Lock retires before controller/root fields and their backing owners.
    }
}

pub(crate) const BOOTSTRAP_SOURCE_WORK: u64 = 1_000_000_000_000;
pub(crate) const BOOTSTRAP_CONFIG_BYTES: usize = 16 << 10;

/// Native-only unconfigured custody. Not source authority, input authentication,
/// or permission to read Python/checkpoint bytes. No Clone or ledger adoption.
/// Its private C9 input unit is NOT compatibility with the ordinary-serde SDK
/// configuration domain (numeric/duplicate semantics are not established).
pub(crate) struct SourceAdmission {
    root: Arc<OwnedCharged<Context>>,
    workload: DecoderWorkload,
    reserved: bool,
}

pub(crate) struct ReserveBootstrapPhase {
    attempt: PrimitiveAttempt,
    workload: DecoderWorkload,
}

/// Payment/custody proof for the private byte codec, not configured authority.
/// Only quote_decoding constructs it, after debiting this exact phase's ledger.
/// No extraction, commit, install, or conversion to prepared C9 input exists.
pub(crate) struct ReservedDecodeFunding {
    phase: ReserveBootstrapPhase,
}

impl ReservedDecodeFunding {
    pub(crate) fn codec_attempt(&self) -> &PrimitiveAttempt { &self.phase.attempt }
    pub(crate) fn check_usable(&self) -> Result<(), Error> { self.phase.attempt.check_usable() }
}

/// Only the admission-owned C9 path below can construct this type. The root in
/// its existing guard is admission identity, not a counter/byte equivalence.
pub(crate) struct PrivatePreparedConfiguration {
    source: OwnedCharged<ImportedValue>,
    attempt: PrimitiveAttempt,
    workload: DecoderWorkload,
}

fn context_control_bytes() -> Result<usize, Error> {
    let slots = [
        3 * size_of::<Context>(),
        3 * size_of::<OnceLock<OwnedCharged<Configured>>>(),
        3 * size_of::<OwnedCharged<Context>>(),
        3 * size_of::<Arc<OwnedCharged<Context>>>(),
        3 * size_of::<SourceAdmission>(),
        3 * size_of::<ReserveBootstrapPhase>(),
        3 * size_of::<PrivatePreparedConfiguration>(),
        3 * size_of::<Result<SourceAdmission, Error>>(),
        3 * size_of::<Result<ReserveBootstrapPhase, Error>>(),
        3 * size_of::<Result<PrivatePreparedConfiguration, Error>>(),
        3 * size_of::<Result<OwnedCharged<Configured>, OwnedCharged<Configured>>>(),
        3 * size_of::<json::Limits>(),
        3 * size_of::<Result<json::ValueQuote, json::Error>>(),
        3 * size_of::<json::ValueQuote>(),
        3 * size_of::<Result<OwnedCharged<ImportedValue>, json::Error>>(),
        3 * size_of::<OwnedCharged<Vec<u8>>>(),
        3 * size_of::<Result<OwnedCharged<Context>, NativeError>>(),
        3 * size_of::<Result<Arc<OwnedCharged<Context>>, Error>>(),
        3 * size_of::<Option<Ledger>>(),
        3 * size_of::<Mutex<Ledger>>(),
        3 * size_of::<Result<u64, std::num::TryFromIntError>>(),
        3 * size_of::<JsonPayment>(),
    ];
    let mut iter = slots.into_iter();
    let bytes = add(std::mem::size_of_val(&slots), std::mem::size_of_val(&iter))?;
    iter.try_fold(add(bytes, arc_header_bytes::<OwnedCharged<Context>>()?)?, add)
}

fn new_context(
    controller: OwnedExternalCharge,
    ledger: Option<Ledger>,
) -> Result<Arc<OwnedCharged<Context>>, Error> {
    if controller.payload_bytes() < context_control_bytes()? { return Err(Error::Capacity); }
    let charged = controller.try_build(|_| Ok(Context {
        configured: OnceLock::new(),
        ledger: ledger.map(Mutex::new),
    })).map_err(Error::Native)?;
    Ok(Arc::new(charged))
}

impl SourceAdmission {
    pub(crate) fn root_control_bytes() -> Result<usize, Error> { context_control_bytes() }

    pub(crate) fn new(controller: OwnedExternalCharge, workload: DecoderWorkload) -> Result<Self, Error> {
        // Closed request envelope, no parsed configuration dimensions needed.
        if workload.batch != 1 || workload.max_input_tokens == 0
            || !(1..=8).contains(&workload.max_new_tokens)
            || workload.max_input_tokens.checked_add(workload.max_new_tokens).is_none_or(|n| n > 40)
        { return Err(Error::Capacity); }
        Ok(Self {
            root: new_context(controller, Some(Ledger { used: 0, limit: BOOTSTRAP_SOURCE_WORK, burned: false }))?,
            workload,
            reserved: false,
        })
    }

    pub(crate) fn reserve_bootstrap_phase(
        &mut self, controller: OwnedExternalCharge, visits: u64,
    ) -> Result<ReserveBootstrapPhase, Error> {
        if self.reserved || self.root.payload().configured.get().is_some() { return Err(Error::Burned); }
        let attempt = begin_attempt(&self.root, controller, visits)?;
        self.reserved = true;
        Ok(ReserveBootstrapPhase { attempt, workload: self.workload })
    }

    pub(crate) fn check_prepared(&self, prepared: &PrivatePreparedConfiguration) -> Result<(), Error> {
        if !Arc::ptr_eq(&self.root, &prepared.attempt.root) { return Err(Error::Counter); }
        Ok(())
    }

    pub(crate) fn configure(
        self, prepared: PrivatePreparedConfiguration, working: OwnedExternalCharge, limits: Limits,
    ) -> Result<OriginalSourcePlan, Error> {
        self.check_prepared(&prepared)?;
        if limits.max_work > BOOTSTRAP_SOURCE_WORK || limits.max_working_payload > MAX_ALLOCATION_BYTES {
            return Err(Error::Capacity);
        }
        let PrivatePreparedConfiguration { source, attempt, .. } = prepared;
        quote_on_root(source, working, self.workload, limits, attempt, JsonPayment::Prepaid)?.construct()
    }
}

impl Drop for SourceAdmission {
    fn drop(&mut self) {
        if self.root.payload().configured.get().is_none() {
            if let Some(ledger) = self.root.payload().ledger.as_ref() {
                let mut ledger = match ledger.lock() {
                    Ok(ledger) => ledger,
                    Err(poisoned) => poisoned.into_inner(),
                };
                ledger.burned = true;
            }
        }
    }
}

impl ReserveBootstrapPhase {
    /// Inert preconfiguration step. Retains this SAME uncommitted phase; the
    /// already-paid reservation fee is neither repeated nor refunded. Errors,
    /// quote abandonment and pending-output abandonment burn the actual root.
    pub(crate) fn quote_decoding(
        self, raw: OwnedCharged<Vec<u8>>, working: OwnedExternalCharge,
        limits: crate::json_bytes_codec::Limits,
    ) -> Result<crate::json_bytes_codec::ReservedCodecQuote, crate::json_bytes_codec::Error> {
        use crate::json_bytes_codec::{self as codec, Error as CodecError};
        // Aggregate custody is bound before any fallible check. Raw retires
        // before the existing attempt/controller, all before working coverage.
        struct Incoming {
            raw: OwnedCharged<Vec<u8>>,
            phase: ReserveBootstrapPhase,
            working: OwnedExternalCharge,
        }
        let incoming = Incoming { raw, phase: self, working };
        // This new inert step cannot enlarge the closed bootstrap envelope.
        let limits = codec::Limits {
            max_source_bytes: limits.max_source_bytes.min(BOOTSTRAP_CONFIG_BYTES),
            max_work: limits.max_work.min(BOOTSTRAP_SOURCE_WORK),
            max_working_payload: limits.max_working_payload.min(MAX_ALLOCATION_BYTES),
            ..limits
        };
        let controller = incoming.phase.attempt.controller.as_ref()
            .ok_or(CodecError::Source(Error::Burned))?;
        if !controller.is_for(incoming.phase.attempt.root.budget()) {
            return Err(CodecError::Counter);
        }
        if controller.payload_bytes() < OriginalSourcePlan::primitive_attempt_control_bytes()
            .map_err(CodecError::Source)? { return Err(CodecError::Capacity); }
        let admission = codec::reserve_decode_admission(
            &incoming.raw, &incoming.working, incoming.phase.attempt.root.budget(),
            controller.charged_bytes(),
        )?;
        // BEFORE detection/content reads. Existing phase visits/fee remain in
        // Ledger.used; this adds only codec visits and new transition controls.
        incoming.phase.attempt.debit(admission.visits()).map_err(CodecError::Source)?;
        let Incoming { raw, phase, working } = incoming;
        codec::quote_reserved(raw, working, ReservedDecodeFunding { phase }, admission, limits)
    }

    pub(crate) fn prepare_configuration(
        self, raw: OwnedCharged<Vec<u8>>, working: OwnedExternalCharge,
    ) -> Result<PrivatePreparedConfiguration, Error> {
        if !working.is_for(self.attempt.root.budget())
            || !std::ptr::eq(raw.budget(), self.attempt.root.budget()) { return Err(Error::Counter); }
        if raw.payload().is_empty() || raw.payload().len() > BOOTSTRAP_CONFIG_BYTES {
            return Err(Error::Capacity);
        }
        let limits = bootstrap_json_limits();
        // Conservative C9 allowance paid BEFORE quote/parser; reported actual
        // work is deliberately NOT charged a second time after construction.
        self.attempt.debit(u64::try_from(limits.max_work).map_err(|_| Error::Work)?)?;
        let source = json::quote(raw, working, limits).map_err(|_| Error::Quote)?
            .construct().map_err(|_| Error::Factory)?;
        Ok(PrivatePreparedConfiguration { source, attempt: self.attempt, workload: self.workload })
    }
}

pub(crate) fn bootstrap_json_limits() -> json::Limits {
    json::Limits {
        max_source_bytes: BOOTSTRAP_CONFIG_BYTES,
        max_nodes: 65536,
        max_depth: 16,
        max_string_bytes: 1 << 20,
        max_number_bytes: 64,
        max_number_work: 100_000_000,
        max_work: 1_000_000_000,
        max_payload_bytes: 8 << 20,
    }
}

impl PrivatePreparedConfiguration {
    pub(crate) fn workload(&self) -> DecoderWorkload { self.workload }
    pub(crate) fn configuration(&self) -> &OwnedCharged<ImportedValue> { &self.source }
    pub(crate) fn prepay_policy(&self, visits: u64) -> Result<(), Error> {
        self.attempt.debit(visits).map(|_| ())
    }
    pub(crate) fn bootstrap_bytes(&self, limits: Limits) -> Result<usize, Error> {
        let ledger = self.attempt.root.payload().ledger.as_ref().ok_or(Error::Burned)?
            .lock().map_err(|_| Error::Burned)?;
        if ledger.burned { return Err(Error::Burned); }
        let remaining = ledger.limit.min(limits.max_work).checked_sub(ledger.used).ok_or(Error::Work)?;
        drop(ledger);
        let input = input_limits(&self.source, limits, remaining)?;
        let bootstrap = source_work::bootstrap(input, limits.factory).map_err(|_| Error::Quote)?;
        add(control_bytes()?, bootstrap.bootstrap_bytes)
    }
}

enum JsonPayment { LegacyReported, Prepaid }

#[derive(Clone, Copy)]
struct Prepared {
    input: SourceInputLimits,
    work: SourceWorkQuote,
    model_factory_bytes: usize,
    model_retained_bytes: usize,
}

struct Quoting {
    source: OwnedCharged<ImportedValue>,
    working: OwnedExternalCharge,
    attempt: PrimitiveAttempt,
}

/// Owns the original imported configuration and real working reservation.
/// No borrowed quote/frame survives its preparation or movement into Context.
pub struct SourceQuote {
    source: OwnedCharged<ImportedValue>,
    working: OwnedExternalCharge,
    limits: Limits,
    workload: DecoderWorkload,
    prepared: Prepared,
    resources: Resources,
    attempt: PrimitiveAttempt,
}

impl SourceQuote {
    pub fn resources(&self) -> Resources {
        self.resources
    }

    pub fn construct(mut self) -> Result<OriginalSourcePlan, Error> {
        self.attempt.check_usable()?;
        // Preserve aggregate drop order if growth refuses before the builder
        // owns source: imported/raw payload first, phase/root controls last.
        self.working.resize_payload(self.working.payload_bytes().max(self.resources.required_working_payload))
            .map_err(Error::Native)?;
        let Self {
            source,
            working,
            limits,
            workload,
            prepared,
            resources,
            attempt,
        } = self;
        let mut failure = None;
        // try_build prices Configured inline/padding BEFORE the second quote or
        // original factory. The SAME charge subsequently lives INSIDE the cell.
        let charged = working
            .try_build(|_| {
                let result = (|| {
                    // Same-counter working/inline reservation includes ALL
                    // source owners and construction scratch before the factory.
                    source
                        .budget()
                        .admit_peak(source.budget().used_bytes())
                        .map_err(Error::Native)?;
                    let (plan, header) = build_plan(source.payload(), workload, limits, prepared, &attempt)?;
                    Ok(Configured {
                        plan,
                        header,
                        workload,
                        configuration: Configuration::Original(source),
                        resources,
                    })
                })();
                result.map_err(|error| {
                    failure = Some(error);
                    NativeError::Resource
                })
            })
            .map_err(|native| failure.unwrap_or(Error::Native(native)))?;
        #[cfg(test)]
        {
            // The model factory and census completed successfully. Release the
            // slot borrow before abandoning a genuine same-root sibling.
            let sibling = PREINSTALL_SIBLING_BURN.with(|slot| slot.borrow_mut().take());
            drop(sibling);
        }
        attempt.install(charged)
    }
}

impl OriginalSourcePlan {
    /// Conservative named logical control/Arc/move overlap for begin, commit,
    /// and abandonment. Existing root allocation stays under its original guard;
    /// its Arc header is additionally covered here, not newly allocated/adopted.
    /// Not a physical stack, allocator, parser, factory, or CPU-fuel quotation.
    pub(crate) fn primitive_attempt_control_bytes() -> Result<usize, Error> {
        // This inventory is needed even BEFORE entering a funded attempt. Its
        // walk must therefore be const evaluation, not lazy/runtime quotation.
        // Preserve the former array/iterator/AttemptDebit allowances exactly;
        // removing runtime visits does not reduce any caller's capacity quote.
        const BASE: Result<usize, Error> = {
            let slots = [
                4 * size_of::<PrimitiveAttempt>(),
                3 * size_of::<Result<PrimitiveAttempt, Error>>(),
                4 * size_of::<OwnedExternalCharge>(),
                2 * size_of::<Option<OwnedExternalCharge>>(),
                2 * size_of::<Result<OwnedExternalCharge, Error>>(),
                3 * size_of::<Arc<OwnedCharged<Context>>>(),
                3 * size_of::<&Arc<OwnedCharged<Context>>>(),
                2 * size_of::<&OriginalSourcePlan>(),
                2 * size_of::<&Mutex<Ledger>>(),
                size_of::<Option<&Mutex<Ledger>>>(),
                2 * size_of::<MutexGuard<'_, Ledger>>(),
                size_of::<LockResult<MutexGuard<'_, Ledger>>>(),
                size_of::<std::sync::PoisonError<MutexGuard<'_, Ledger>>>(),
                size_of::<Result<(), Error>>(),
                3 * size_of::<Result<u64, Error>>(),
                2 * size_of::<Option<u64>>(),
                4 * size_of::<u64>(),
                4 * size_of::<usize>(),
                2 * size_of::<Result<usize, Error>>(),
                2 * size_of::<Option<usize>>(),
                2 * size_of::<bool>(),
            ];
            let mut bytes = size_of::<[usize; 21]>()
                .checked_add(size_of::<std::array::IntoIter<usize, 21>>());
            bytes = match bytes {
                Some(n) => n.checked_add(3 * size_of::<AttemptDebit>()),
                None => None,
            };
            let mut i = 0;
            while i < 21 {
                bytes = match bytes {
                    Some(n) => n.checked_add(slots[i]),
                    None => None,
                };
                i += 1;
            }
            match bytes { Some(n) => Ok(n), None => Err(Error::Overflow) }
        };
        add(BASE?, arc_header_bytes::<OwnedCharged<Context>>()?)
    }

    /// Debit visits plus a positive one-unit attempt fee under the original
    /// cumulative work ceiling BEFORE any caller factory/publication work.
    /// Counter/capacity refusals do not enter or burn the source ledger. Once
    /// admitted, overflow/exhaustion and later abandonment/error/unwind burn all
    /// siblings; no work refund, fresh budget, or renewable attempt counter.
    pub(crate) fn begin_primitive_attempt(
        &self,
        controller: OwnedExternalCharge,
        visits: u64,
    ) -> Result<PrimitiveAttempt, Error> {
        if !controller.is_for(self.budget()) {
            return Err(Error::Counter);
        }
        if controller.payload_bytes() < Self::primitive_attempt_control_bytes()? {
            return Err(Error::Capacity);
        }
        begin_attempt(self.root(), controller, visits)
    }

    /// Known logical slots for compiler metadata's shared-ledger transition.
    /// No allocation/factory/CPU-fuel/physical-callframe authority follows.
    pub(crate) fn compiler_metadata_control_bytes() -> Result<usize, Error> {
        metadata_control_bytes()
    }
}

#[cfg(test)]
#[test]
fn primitive_attempt_control_bytes_matches_old_layout_inventory() {
    // Independent literal old inventory and runtime old fold, TEST ONLY.
    // Production BASE is not inspected/reused by this comparison. Tests UNRUN.
    let slots = [
        4 * size_of::<PrimitiveAttempt>(),
        3 * size_of::<Result<PrimitiveAttempt, Error>>(),
        4 * size_of::<OwnedExternalCharge>(),
        2 * size_of::<Option<OwnedExternalCharge>>(),
        2 * size_of::<Result<OwnedExternalCharge, Error>>(),
        3 * size_of::<Arc<OwnedCharged<Context>>>(),
        3 * size_of::<&Arc<OwnedCharged<Context>>>(),
        2 * size_of::<&OriginalSourcePlan>(),
        2 * size_of::<&Mutex<Ledger>>(),
        size_of::<Option<&Mutex<Ledger>>>(),
        2 * size_of::<MutexGuard<'_, Ledger>>(),
        size_of::<LockResult<MutexGuard<'_, Ledger>>>(),
        size_of::<std::sync::PoisonError<MutexGuard<'_, Ledger>>>(),
        size_of::<Result<(), Error>>(),
        3 * size_of::<Result<u64, Error>>(),
        2 * size_of::<Option<u64>>(),
        4 * size_of::<u64>(),
        4 * size_of::<usize>(),
        2 * size_of::<Result<usize, Error>>(),
        2 * size_of::<Option<usize>>(),
        2 * size_of::<bool>(),
    ];
    let mut iter = slots.into_iter();
    let bytes = add(std::mem::size_of_val(&slots), std::mem::size_of_val(&iter)).unwrap();
    let bytes = add(bytes, arc_header_bytes::<OwnedCharged<Context>>().unwrap()).unwrap();
    let bytes = add(bytes, 3 * size_of::<AttemptDebit>()).unwrap();
    let old = iter.try_fold(bytes, add);
    assert_eq!(OriginalSourcePlan::primitive_attempt_control_bytes(), old);
}

fn begin_attempt(
    root: &Arc<OwnedCharged<Context>>,
    controller: OwnedExternalCharge,
    visits: u64,
) -> Result<PrimitiveAttempt, Error> {
    enter_attempt(root, controller, AttemptDebit::Primitive(visits))
}

// Existing native source construction/lowering already pays source_work's quote
// fee. Preserve legacy debit exactly; new primitive/bootstrap adds visits+1.
enum AttemptDebit { Primitive(u64), ExistingSource }

fn enter_attempt(
    root: &Arc<OwnedCharged<Context>>, controller: OwnedExternalCharge, debit: AttemptDebit,
) -> Result<PrimitiveAttempt, Error> {
        if !controller.is_for(root.budget()) { return Err(Error::Counter); }
        if controller.payload_bytes() < OriginalSourcePlan::primitive_attempt_control_bytes()? {
            return Err(Error::Capacity);
        }
        let mut ledger = root.payload().ledger.as_ref().ok_or(Error::Burned)?
            .lock().map_err(|_| Error::Burned)?;
        if ledger.burned {
            return Err(Error::Burned);
        }
        ledger.burned = true;
        let debit = match debit {
            AttemptDebit::Primitive(visits) => visits.checked_add(1).ok_or(Error::Work)?,
            AttemptDebit::ExistingSource => 0,
        };
        let used = ledger.used.checked_add(debit)
            .filter(|used| *used <= ledger.limit).ok_or(Error::Work)?;
        ledger.used = used;
        let attempt = PrimitiveAttempt {
            root: Arc::clone(root),
            controller: Some(controller),
            committed: false,
        };
        ledger.burned = false;
        drop(ledger);
        // No ledger mutex crosses into external/parser/read/writer work.
        Ok(attempt)
}

fn metadata_control_bytes() -> Result<usize, Error> {
        type Guard<'a> = std::sync::MutexGuard<'a, Ledger>;
        let mut bytes = size_of::<Guard<'_>>();
        for n in [
            size_of::<Result<Guard<'_>, std::sync::PoisonError<Guard<'_>>>>(),
            size_of::<&OwnedExternalCharge>(),
            size_of::<&Arc<OwnedCharged<Context>>>(),
            size_of::<Result<u64, Error>>(),
            size_of::<Option<u64>>(),
            3 * size_of::<u64>(),
        ] {
            bytes = add(bytes, n)?;
        }
        Ok(bytes)
}

impl OriginalSourcePlan {
    /// Compiler-private bounded metadata visits share the original source's
    /// cumulative quota. Not native WorkBudget, an input/proof issuer or CPU fuel.
    /// An admitted transition error burns all lowering siblings; no refund/reset.
    pub(crate) fn prepay_compiler_metadata(
        &self,
        controller: &OwnedExternalCharge,
        visits: u64,
    ) -> Result<(), Error> {
        self.prepay_primitive_work(controller, visits)
    }

    /// Method-neutral logical primitive work shares the SAME original source
    /// ledger as lowering and compiler metadata. No independent quota, numeric
    /// profile, source authority, CPU fuel, retry credit or refund is created.
    /// Caller reserves transition controls and prepays before doing the work.
    /// Failed cumulative admission burns every sibling exactly as lowering does.
    pub(crate) fn prepay_primitive_work(
        &self,
        controller: &OwnedExternalCharge,
        visits: u64,
    ) -> Result<(), Error> {
        if !controller.is_for(self.budget()) {
            return Err(Error::Counter);
        }
        if controller.payload_bytes() < Self::compiler_metadata_control_bytes()? {
            return Err(Error::Capacity);
        }
        let mut ledger = self
            .root()
            .payload()
            .ledger
            .as_ref()
            .ok_or(Error::Burned)?
            .lock()
            .map_err(|_| Error::Burned)?;
        if ledger.burned {
            return Err(Error::Burned);
        }
        ledger.burned = true;
        let used = ledger
            .used
            .checked_add(visits)
            .filter(|used| *used <= ledger.limit)
            .ok_or(Error::Work)?;
        ledger.used = used;
        ledger.burned = false;
        Ok(())
    }

    pub fn workload(&self) -> DecoderWorkload {
        self.inner.payload().workload
    }

    /// Allocation-free bootstrap query for SAME-source lowering. Read the actual
    /// shared remaining ledger, not this handle's historical resource summary.
    /// This returns required reservation bytes, NOT a permit or source/epoch
    /// authority. `lower` must still receive a real same-counter controller and
    /// atomically recheck/debit work before any query pool/factory allocation.
    pub fn lower_bootstrap_bytes(&self, limits: Limits) -> Result<usize, Error> {
        let ledger = self
            .root()
            .payload()
            .ledger
            .as_ref()
            .ok_or(Error::Burned)?
            .lock()
            .map_err(|_| Error::Burned)?;
        if ledger.burned {
            return Err(Error::Burned);
        }
        let remaining = ledger
            .limit
            .min(limits.max_work)
            .checked_sub(ledger.used)
            .filter(|n| *n != 0)
            .ok_or(Error::Work)?;
        drop(ledger);
        let input = input_limits(self.configuration(), limits, remaining)?;
        let bootstrap = source_work::bootstrap(input, limits.factory).map_err(|_| Error::Quote)?;
        let phase = add(control_bytes()?, bootstrap.bootstrap_bytes)?;
        add(phase, add(Self::primitive_attempt_control_bytes()?, size_of::<OwnedExternalCharge>())?)
    }

    pub fn plan(&self) -> &DecoderPlan {
        &self.inner.payload().plan
    }
    pub fn header(&self) -> &census::Header {
        &self.inner.payload().header
    }
    pub fn budget(&self) -> &Budget {
        self.inner.budget()
    }
    pub fn resources(&self) -> Resources {
        self.inner.payload().resources
    }

    fn root(&self) -> &Arc<OwnedCharged<Context>> {
        match &self.inner.payload().configuration {
            Configuration::Original(_) => &self.inner,
            Configuration::OriginalContext(root) => root,
        }
    }

    fn configuration(&self) -> &OwnedCharged<ImportedValue> {
        match &self.root().payload().configuration {
            Configuration::Original(value) => value,
            Configuration::OriginalContext(_) => unreachable!("private original context invariant"),
        }
    }

    pub fn raw_configuration(&self) -> &[u8] {
        self.configuration().payload().raw()
    }

    /// Reference identity of actual factory/source custody, never digest equality.
    pub fn same_origin(&self, other: &Self) -> bool {
        Arc::ptr_eq(self.root(), other.root())
    }
    pub fn same_plan(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Reinvoke the SAME original config factory for a new workload. This links
    /// source provenance only: no request epoch/token lineage or import issuer.
    /// Source-wide construction quota cannot fork, reset, or refund via plans.
    pub fn lower(
        &self,
        mut working: OwnedExternalCharge,
        workload: DecoderWorkload,
        limits: Limits,
    ) -> Result<Self, Error> {
        let source = self.configuration();
        check_counter(source, &working)?;
        // Every concurrent caller supplies its own real controller reservation
        // BEFORE creating lock/result slots. A foreign/unadmitted caller never
        // enters the source transition; no sibling's control allowance is lent.
        if working.payload_bytes() < control_bytes()? {
            return Err(Error::Capacity);
        }
        let root = self.root();
        let attempt = enter_attempt(root,
            working.split_owned(Self::primitive_attempt_control_bytes()?).map_err(Error::Native)?,
            AttemptDebit::ExistingSource)?;
        let ledger = root.payload().ledger.as_ref().ok_or(Error::Burned)?
            .lock().map_err(|_| Error::Burned)?;
        if ledger.burned { return Err(Error::Burned); }
        let limit = ledger.limit.min(limits.max_work);
        let remaining = limit.checked_sub(ledger.used).ok_or(Error::Work)?;
        drop(ledger);
        let input = input_limits(source, limits, remaining)?;
        let bootstrap = source_work::bootstrap(input, limits.factory).map_err(|_| Error::Quote)?;
        check_bootstrap(&working, bootstrap.bootstrap_bytes)?;
        let first = bootstrap.quote_work;
        attempt.debit(first)?;
        let prepared = prepare(source.payload(), workload, limits, input)?;
        let remainder = prepared
            .work
            .total_work()
            .checked_sub(first)
            .ok_or(Error::Work)?;
        let total = attempt.debit(remainder)?;
        if total > limit { return Err(Error::Work); }
        // Pay ALL pending qualified work before growth, original factory or Arc.
        let incoming = add(
            add(add(root.charged_bytes(), root.payload().configured.get().ok_or(Error::Burned)?.charged_bytes())?, source.charged_bytes())?,
            add(source.payload().raw_owner().charged_bytes(), attempt.controller.as_ref().ok_or(Error::Burned)?.charged_bytes())?,
        )?;
        // Both Context and its cell are admitted/constructed before the model
        // factory. Derived Configured retains the actual original root only.
        let context = new_context(
            working.split_owned(context_control_bytes()?).map_err(Error::Native)?, None,
        )?;
        let resources = resources(add(incoming, context.charged_bytes())?, &working, prepared, total, limits)?;
        working.resize_payload(working.payload_bytes().max(resources.required_working_payload))
            .map_err(Error::Native)?;
        let mut failure = None;
        let charged = working
            .try_build(|_| {
                if let Err(error) = self.budget().admit_peak(self.budget().used_bytes()) {
                    failure = Some(Error::Native(error));
                    return Err(NativeError::Resource);
                }
                let result =
                    build_plan(source.payload(), workload, limits, prepared, &attempt).map(|(plan, header)| Configured {
                        plan,
                        header,
                        workload,
                        configuration: Configuration::OriginalContext(Arc::clone(root)),
                        resources,
                    });
                result.map_err(|error| {
                    failure = Some(error);
                    NativeError::Resource
                })
            })
            .map_err(|native| failure.unwrap_or(Error::Native(native)))?;
        if context.payload().configured.set(charged).is_err() { return Err(Error::Burned); }
        let result = Self { inner: context };
        drop(attempt.commit()?);
        Ok(result)
    }
}

fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Overflow)
}

fn arc_header_bytes<T>() -> Result<usize, Error> {
    // Same named Arc control/padding policy as OwnedBudget::resources. Not an
    // observation of allocator overhead or a compiled/native stack quotation.
    let (layout, offset) = Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<T>())
        .map_err(|_| Error::Overflow)?;
    add(
        offset,
        layout
            .pad_to_align()
            .size()
            .checked_sub(add(offset, size_of::<T>())?)
            .ok_or(Error::Overflow)?,
    )
}

/// Must be present on the genuine same-counter working guard BEFORE quotation.
/// Named logical controls only; dependency/native frames and RSS remain unknown.
fn control_bytes() -> Result<usize, Error> {
    let slots = [
        size_of::<Limits>(),
        size_of::<Resources>(),
        size_of::<Context>(),
        4 * size_of::<Configured>(),
        4 * size_of::<OwnedCharged<Configured>>(),
        4 * size_of::<Result<Configured, Error>>(),
        4 * size_of::<Result<OwnedCharged<Configured>, NativeError>>(),
        4 * size_of::<OnceLock<OwnedCharged<Configured>>>(),
        context_control_bytes()?,
        OriginalSourcePlan::primitive_attempt_control_bytes()?,
        size_of::<OwnedExternalCharge>(),
        size_of::<OriginalSourcePlan>(),
        size_of::<ConfigLowererQuote<'_>>(),
        // build_plan's plan binding/published tuple operand and root/derived
        // Context field operands. Complete containing types, not a naked Header
        // increment to an old plan quote.
        4 * size_of::<DecoderPlan>(),
        size_of::<(DecoderPlan, census::Header)>(),
        // Producer/return plus root and derived call/propagation pairs.
        6 * size_of::<Result<(DecoderPlan, census::Header), Error>>(),
        2 * size_of::<Result<DecoderPlan, pllm_models::ModelError>>(),
        // build_plan's extracted header, tuple field operand, root/derived
        // destructuring bindings and their Context initializer field operands.
        6 * size_of::<census::Header>(),
        // Root/derived tuple-destructure producer slots and DecoderPlan bindings.
        2 * size_of::<(DecoderPlan, census::Header)>(),
        2 * size_of::<DecoderPlan>(),
        size_of::<OwnedExternalCharge>(),
        size_of::<OwnedCharged<ImportedValue>>(),
        size_of::<Result<ConfigLowererQuote<'_>, pllm_models::ModelError>>(),
        size_of::<Result<OriginalSourcePlan, Error>>(),
        size_of::<Ledger>(),
        size_of::<Option<Error>>(),
        size_of::<Prepared>(),
        size_of::<SourceQuote>(),
        3 * size_of::<Quoting>(),
        3 * size_of::<PrimitiveAttempt>(),
        3 * size_of::<Result<SourceQuote, Error>>(),
        3 * size_of::<AttemptDebit>(),
        size_of::<Result<Prepared, Error>>(),
        size_of::<Result<Context, Error>>(),
        size_of::<Result<OwnedCharged<Context>, NativeError>>(),
        size_of::<Arc<OwnedCharged<Context>>>(),
        size_of::<DecoderWorkload>(),
        size_of::<SourceInputLimits>(),
        size_of::<source_work::SourceWorkBootstrap>(),
        size_of::<Result<source_work::SourceWorkBootstrap, source_work::SourceWorkError>>(),
        size_of::<MutexGuard<'_, Ledger>>(),
        size_of::<LockResult<MutexGuard<'_, Ledger>>>(),
        size_of::<Option<&Mutex<Ledger>>>(),
        // Fixed preflight diagnostic text; errors are erased under this charge.
        256,
    ];
    // The expanded literal and its owning IntoIter are known simultaneous
    // quotation slots too; do not hide them behind the typed owner increment.
    let mut iter = slots.into_iter();
    let bytes = add(std::mem::size_of_val(&slots), std::mem::size_of_val(&iter))?;
    iter.try_fold(add(bytes, size_of::<Option<usize>>())?, add)
}

fn working_payload(prepared: Prepared) -> Result<usize, Error> {
    // Query and construction/census scopes are disjoint. During post-census the
    // actual generated plan remains, not the constructor's retired temporaries.
    let phase = prepared.model_factory_bytes.max(add(
        prepared.model_retained_bytes,
        prepared.work.resources().bootstrap_bytes,
    )?);
    add(
        add(control_bytes()?, phase)?,
        arc_header_bytes::<OwnedCharged<Context>>()?,
    )
}

fn working_charge_bytes(payload: usize) -> Result<usize, Error> {
    add(payload, size_of::<OwnedCharged<Configured>>())
}

fn input_limits(
    source: &OwnedCharged<ImportedValue>,
    limits: Limits,
    remaining: u64,
) -> Result<SourceInputLimits, Error> {
    let value = source.payload();
    let counts = value.counts();
    // B's original cursor-work and payload-byte caps have DIFFERENT units. Long
    // string payloads do not imply equally many traversal events. Reuse the
    // original owner's payload formula, never its work count or lexical length.
    // These are CEILINGS; native traversal independently refuses false caps.
    let traversal = usize::try_from(value.resources().work.census).map_err(|_| Error::Overflow)?;
    Ok(SourceInputLimits {
        max_nodes: counts.spans,
        max_payload_bytes: value
            .traversal_payload_bound()
            .map_err(|_| Error::Overflow)?,
        max_string_bytes: counts.largest_string,
        max_key_bytes: counts.largest_string,
        max_depth: add(counts.depth, 1)?,
        max_traversal_work: traversal,
        max_total_work: remaining,
        max_workspace_bytes: limits.max_working_payload,
    })
}

fn check_counter(
    source: &OwnedCharged<ImportedValue>,
    working: &OwnedExternalCharge,
) -> Result<(), Error> {
    if !working.is_for(source.budget()) || !std::ptr::eq(source.budget(), source.payload().budget())
    {
        return Err(Error::Counter);
    }
    Ok(())
}

fn check_bootstrap(working: &OwnedExternalCharge, model_bootstrap: usize) -> Result<(), Error> {
    if working.payload_bytes() < add(control_bytes()?, model_bootstrap)? {
        return Err(Error::Capacity);
    }
    Ok(())
}

/// Pure layout computation before caller reserves bootstrap or creates frames.
pub fn bootstrap_bytes(
    source: &OwnedCharged<ImportedValue>,
    limits: Limits,
) -> Result<usize, Error> {
    let prior = u64::try_from(source.payload().resources().work.total).map_err(|_| Error::Work)?;
    let input = input_limits(
        source,
        limits,
        limits.max_work.checked_sub(prior).ok_or(Error::Work)?,
    )?;
    let model = source_work::bootstrap(input, limits.factory).map_err(|_| Error::Quote)?;
    legacy_bootstrap_bytes(model.bootstrap_bytes)
}

fn legacy_bootstrap_bytes(model: usize) -> Result<usize, Error> {
    let root = add(context_control_bytes()?, size_of::<OwnedExternalCharge>())?;
    let attempt = add(OriginalSourcePlan::primitive_attempt_control_bytes()?, size_of::<OwnedExternalCharge>())?;
    add(add(add(control_bytes()?, model)?, root)?, attempt)
}

fn prepare(
    source: &ImportedValue,
    workload: DecoderWorkload,
    limits: Limits,
    input: SourceInputLimits,
) -> Result<Prepared, Error> {
    let bootstrap = source_work::bootstrap(input, limits.factory).map_err(|_| Error::Quote)?;
    if bootstrap.frame_slots != 17 {
        return Err(Error::Capacity);
    }
    // Bootstrap and global first-pass fee were checked before this actual pool.
    let mut frames = [const { NativeFrame::empty() }; 17];
    let (model, work) = source_work::quote(
        source.value(),
        workload,
        limits.expansion,
        limits.factory,
        input,
        &mut frames,
    )
    .map_err(|_| Error::Quote)?;
    Ok(Prepared {
        input,
        work,
        model_factory_bytes: model.factory_bytes(),
        model_retained_bytes: model.metadata().retained_metadata_bytes,
    })
    // Both captured model and complete frame pool drop before source can move.
}

fn resources(
    incoming: usize,
    working: &OwnedExternalCharge,
    prepared: Prepared,
    total: u64,
    limits: Limits,
) -> Result<Resources, Error> {
    let required = working_payload(prepared)?;
    let actual_payload = required.max(working.payload_bytes());
    if actual_payload > limits.max_working_payload {
        return Err(Error::Capacity);
    }
    let charge = working_charge_bytes(actual_payload)?;
    Ok(Resources {
        original_owner_bytes: incoming,
        required_working_payload: required,
        working_charge_bytes: charge,
        original_owners_peak_bytes: add(incoming, charge)?,
        model_factory_bytes: prepared.model_factory_bytes,
        model_retained_bytes: prepared.model_retained_bytes,
        prepaid_work: total,
    })
}

/// Consumes genuine original input owners. No unpriced lowerer, parsed-plan
/// conversion, caller capacity baseline or subsequent "approval" callback.
pub fn quote(
    source: OwnedCharged<ImportedValue>,
    mut working: OwnedExternalCharge,
    workload: DecoderWorkload,
    limits: Limits,
) -> Result<SourceQuote, Error> {
    check_counter(&source, &working)?;
    // Legacy already-imported input remains supported, but it uses this SAME
    // root/cell/attempt machinery; no second ledger is created at construct.
    let bootstrap = bootstrap_bytes(&source, limits)?;
    if working.payload_bytes() < bootstrap { return Err(Error::Capacity); }
    let root = new_context(
        working.split_owned(context_control_bytes()?).map_err(Error::Native)?,
        Some(Ledger { used: 0, limit: limits.max_work, burned: false }),
    )?;
    let attempt = enter_attempt(&root,
        working.split_owned(OriginalSourcePlan::primitive_attempt_control_bytes()?).map_err(Error::Native)?, AttemptDebit::ExistingSource)?;
    quote_on_root(source, working, workload, limits, attempt, JsonPayment::LegacyReported)
}

fn quote_on_root(
    source: OwnedCharged<ImportedValue>, working: OwnedExternalCharge,
    workload: DecoderWorkload, limits: Limits, attempt: PrimitiveAttempt, payment: JsonPayment,
) -> Result<SourceQuote, Error> {
    let pending = Quoting { source, working, attempt };
    check_counter(&pending.source, &pending.working)?;
    if !std::ptr::eq(pending.source.budget(), pending.attempt.root.budget()) { return Err(Error::Counter); }
    pending.attempt.restrict_limit(limits.max_work)?;
    if let JsonPayment::LegacyReported = payment {
        pending.attempt.debit(u64::try_from(pending.source.payload().resources().work.total).map_err(|_| Error::Work)?)?;
    }
    let ledger = pending.attempt.root.payload().ledger.as_ref().ok_or(Error::Burned)?
        .lock().map_err(|_| Error::Burned)?;
    if ledger.burned { return Err(Error::Burned); }
    let remaining = ledger.limit.min(limits.max_work).checked_sub(ledger.used).ok_or(Error::Work)?;
    drop(ledger);
    let input = input_limits(&pending.source, limits, remaining)?;
    let bootstrap = source_work::bootstrap(input, limits.factory).map_err(|_| Error::Quote)?;
    check_bootstrap(&pending.working, bootstrap.bootstrap_bytes)?;
    pending.attempt.debit(bootstrap.quote_work)?;
    let prepared = prepare(pending.source.payload(), workload, limits, input)?;
    let total = pending.attempt.debit(prepared.work.total_work().checked_sub(bootstrap.quote_work).ok_or(Error::Work)?)?;
    if total > limits.max_work { return Err(Error::Work); }
    let incoming = add(
        add(add(pending.source.charged_bytes(), pending.attempt.root.charged_bytes())?, pending.attempt.controller.as_ref().ok_or(Error::Burned)?.charged_bytes())?,
        pending.source.payload().raw_owner().charged_bytes(),
    )?;
    let resources = resources(incoming, &pending.working, prepared, total, limits)?;
    let Quoting { source, working, attempt } = pending;
    Ok(SourceQuote {
        source,
        working,
        limits,
        workload,
        prepared,
        resources,
        attempt,
    })
}

fn build_plan(
    source: &ImportedValue,
    workload: DecoderWorkload,
    limits: Limits,
    prepared: Prepared,
    attempt: &PrimitiveAttempt,
) -> Result<(DecoderPlan, census::Header), Error> {
    let (model, work) = {
        // Fresh, disjoint second pass; its WHOLE allowance was already prepaid.
        let bootstrap =
            source_work::bootstrap(prepared.input, limits.factory).map_err(|_| Error::Quote)?;
        if bootstrap.frame_slots != 17 {
            return Err(Error::Capacity);
        }
        let mut frames = [const { NativeFrame::empty() }; 17];
        source_work::quote(
            source.value(),
            workload,
            limits.expansion,
            limits.factory,
            prepared.input,
            &mut frames,
        )
        .map_err(|_| Error::Quote)?
    }; // Query frames actually retire before the original factory begins.
    if work.total_work() != prepared.work.total_work()
        || model.factory_bytes() != prepared.model_factory_bytes
        || model.metadata().retained_metadata_bytes != prepared.model_retained_bytes
    {
        return Err(Error::Quote);
    }
    attempt.check_usable()?;
    observe_factory();
    let plan = model.construct().map_err(|_| Error::Factory)?;
    let header = {
        let work = prepared.work;
        let resources = work.post_census_resources();
        let limits = work.post_census_limits();
        resources.check(limits).map_err(|_| Error::Census)?;
        let mut frames = [const { NativeFrame::empty() }; 17];
        let mut cursor = census::Census::new(&mut frames, limits).map_err(|_| Error::Census)?;
        let measured = cursor.plan(&plan).map_err(|_| Error::Census)?;
        if measured.backing_bytes() > prepared.model_retained_bytes {
            return Err(Error::Capacity);
        }
        measured.into_header()
    }; // Report, Census and native frame borrows end before tuple publication.
    Ok((plan, header))
}

#[cfg(not(test))]
fn observe_factory() {}

#[cfg(test)]
thread_local! { static FACTORIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

#[cfg(test)]
thread_local! { static PANIC_FACTORY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }

#[cfg(test)]
thread_local! {
    static PREINSTALL_SIBLING_BURN: std::cell::RefCell<Option<PrimitiveAttempt>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
thread_local! {
    static RETIREMENT_BUDGET: std::cell::RefCell<Option<pllm_garble::shared_allocation::OwnedBudget>> = const { std::cell::RefCell::new(None) };
    static RETIRED_BYTES: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn observe_context_retirement() {
    RETIREMENT_BUDGET.with(|slot| {
        if let Ok(slot) = slot.try_borrow() {
            if let Some(budget) = slot.as_ref() {
                RETIRED_BYTES.with(|value| value.set(Some(budget.budget().used_bytes())));
            }
        }
    });
}

#[cfg(test)]
fn observe_factory() {
    FACTORIES.with(|n| n.set(n.get() + 1));
    PANIC_FACTORY.with(|p| {
        if p.replace(false) {
            panic!("test original source factory unwind");
        }
    });
}

#[cfg(test)]
#[path = "original_source_tests.rs"]
mod tests;
