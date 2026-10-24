//! Actual config-factory provenance; not checkpoint authentication or execution.
//!
//! An origin is born only after the registered original model factory succeeds
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
//! field is its cumulative construction-work ledger: independent reservations
//! are prepaid, never refunded or lent. Failure/unwind burns further lowering.
//! A derived plan retains the original root (including its first actual plan),
//! conservatively retaining all original guard excess rather than taking credit.

use crate::plan_input::json::ImportedValue;
use pllm_garble::shared_allocation::{
    Budget, Error as NativeError, OwnedCharged, OwnedExternalCharge,
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
    sync::{atomic::AtomicUsize, Arc, LockResult, Mutex, MutexGuard},
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

struct Context {
    // Actual plan retires before its configuration/source owners and charge.
    plan: DecoderPlan,
    // Captured at the genuine factory call, never reconstructed from graph data.
    // Context inline/padding and controller quotes use size_of::<Context>(), so
    // this retained field is admitted before both root and derived factories.
    workload: DecoderWorkload,
    configuration: Configuration,
    resources: Resources,
    ledger: Option<Mutex<Ledger>>,
}

/// Immutable actual factory result. No public Clone, detach, mutable plan, or
/// constructor accepting a pre-existing plan/identity/digest is provided.
pub struct OriginalSourcePlan {
    inner: Arc<OwnedCharged<Context>>,
}

#[derive(Clone, Copy)]
struct Prepared {
    input: SourceInputLimits,
    work: SourceWorkQuote,
    model_factory_bytes: usize,
    model_retained_bytes: usize,
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
}

impl SourceQuote {
    pub fn resources(&self) -> Resources {
        self.resources
    }

    pub fn construct(self) -> Result<OriginalSourcePlan, Error> {
        let Self {
            source,
            mut working,
            limits,
            workload,
            prepared,
            resources,
        } = self;
        working
            .resize_payload(
                working
                    .payload_bytes()
                    .max(resources.required_working_payload),
            )
            .map_err(Error::Native)?;
        let mut failure = None;
        // try_build prices Context inline/padding BEFORE the second quote or
        // original factory. The SAME charge subsequently lives INSIDE the Arc.
        let charged = working
            .try_build(|_| {
                let result = (|| {
                    // Same-counter working/inline reservation includes ALL
                    // source owners and construction scratch before the factory.
                    source
                        .budget()
                        .admit_peak(source.budget().used_bytes())
                        .map_err(Error::Native)?;
                    let plan = build_plan(source.payload(), workload, limits, prepared)?;
                    Ok(Context {
                        plan,
                        workload,
                        configuration: Configuration::Original(source),
                        resources,
                        ledger: Some(Mutex::new(Ledger {
                            used: resources.prepaid_work,
                            limit: limits.max_work,
                            burned: false,
                        })),
                    })
                })();
                result.map_err(|error| {
                    failure = Some(error);
                    NativeError::Resource
                })
            })
            .map_err(|native| failure.unwrap_or(Error::Native(native)))?;
        // Actual allocation header/control was quoted before try_build. Stable
        // Arc OOM may abort, as in OwnedBudget; this is not recoverable RSS.
        Ok(OriginalSourcePlan {
            inner: Arc::new(charged),
        })
    }
}

impl OriginalSourcePlan {
    /// Known logical slots for compiler metadata's shared-ledger transition.
    /// No allocation/factory/CPU-fuel/physical-callframe authority follows.
    pub(crate) fn compiler_metadata_control_bytes() -> Result<usize, Error> {
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

    /// Compiler-private bounded metadata visits share the original source's
    /// cumulative quota. Not native WorkBudget, an input/proof issuer or CPU fuel.
    /// An admitted transition error burns all lowering siblings; no refund/reset.
    pub(crate) fn prepay_compiler_metadata(
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
        let input = input_limits(self.configuration(), limits, remaining)?;
        let bootstrap = source_work::bootstrap(input, limits.factory).map_err(|_| Error::Quote)?;
        add(control_bytes()?, bootstrap.bootstrap_bytes)
    }

    pub fn plan(&self) -> &DecoderPlan {
        &self.inner.payload().plan
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
        let mut ledger = root
            .payload()
            .ledger
            .as_ref()
            .ok_or(Error::Burned)?
            .lock()
            .map_err(|_| Error::Burned)?;
        if ledger.burned {
            return Err(Error::Burned);
        }
        // Any error/cancellation/unwind AFTER admitted entry burns future
        // lowering through ALL siblings. Immutable plans retain provenance only.
        ledger.burned = true;
        let limit = ledger.limit.min(limits.max_work);
        let remaining = limit.checked_sub(ledger.used).ok_or(Error::Work)?;
        let input = input_limits(source, limits, remaining)?;
        let bootstrap = source_work::bootstrap(input, limits.factory).map_err(|_| Error::Quote)?;
        check_bootstrap(&working, bootstrap.bootstrap_bytes)?;
        let first = bootstrap.quote_work;
        ledger.used = ledger
            .used
            .checked_add(first)
            .filter(|n| *n <= limit)
            .ok_or(Error::Work)?;
        let prepared = prepare(source.payload(), workload, limits, input)?;
        let remainder = prepared
            .work
            .total_work()
            .checked_sub(first)
            .ok_or(Error::Work)?;
        let total = ledger
            .used
            .checked_add(remainder)
            .filter(|n| *n <= limit)
            .ok_or(Error::Work)?;
        // Pay ALL pending qualified work before growth, original factory or Arc.
        ledger.used = total;
        let incoming = add(
            add(root.charged_bytes(), source.charged_bytes())?,
            source.payload().raw_owner().charged_bytes(),
        )?;
        let resources = resources(incoming, &working, prepared, total, limits)?;
        working
            .resize_payload(
                working
                    .payload_bytes()
                    .max(resources.required_working_payload),
            )
            .map_err(Error::Native)?;
        let mut failure = None;
        let charged = working
            .try_build(|_| {
                if let Err(error) = self.budget().admit_peak(self.budget().used_bytes()) {
                    failure = Some(Error::Native(error));
                    return Err(NativeError::Resource);
                }
                let result =
                    build_plan(source.payload(), workload, limits, prepared).map(|plan| Context {
                        plan,
                        workload,
                        configuration: Configuration::OriginalContext(Arc::clone(root)),
                        resources,
                        ledger: None,
                    });
                result.map_err(|error| {
                    failure = Some(error);
                    NativeError::Resource
                })
            })
            .map_err(|native| failure.unwrap_or(Error::Native(native)))?;
        let result = Self {
            inner: Arc::new(charged),
        };
        ledger.burned = false;
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
    [
        size_of::<Limits>(),
        size_of::<Resources>(),
        size_of::<Context>(),
        size_of::<OriginalSourcePlan>(),
        size_of::<ConfigLowererQuote<'_>>(),
        size_of::<OwnedExternalCharge>(),
        size_of::<OwnedCharged<ImportedValue>>(),
        size_of::<Result<ConfigLowererQuote<'_>, pllm_models::ModelError>>(),
        size_of::<Result<OriginalSourcePlan, Error>>(),
        size_of::<Ledger>(),
        size_of::<Option<Error>>(),
        size_of::<Prepared>(),
        size_of::<SourceQuote>(),
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
    ]
    .into_iter()
    .try_fold(0, add)
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
    add(payload, size_of::<OwnedCharged<Context>>())
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
    add(control_bytes()?, model.bootstrap_bytes)
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
    working: OwnedExternalCharge,
    workload: DecoderWorkload,
    limits: Limits,
) -> Result<SourceQuote, Error> {
    check_counter(&source, &working)?;
    let prior = u64::try_from(source.payload().resources().work.total).map_err(|_| Error::Work)?;
    let remaining = limits.max_work.checked_sub(prior).ok_or(Error::Work)?;
    let input = input_limits(&source, limits, remaining)?;
    let bootstrap = source_work::bootstrap(input, limits.factory).map_err(|_| Error::Quote)?;
    check_bootstrap(&working, bootstrap.bootstrap_bytes)?;
    prior
        .checked_add(bootstrap.quote_work)
        .filter(|n| *n <= limits.max_work)
        .ok_or(Error::Work)?;
    let prepared = prepare(source.payload(), workload, limits, input)?;
    let total = prior
        .checked_add(prepared.work.total_work())
        .filter(|n| *n <= limits.max_work)
        .ok_or(Error::Work)?;
    let incoming = add(
        source.charged_bytes(),
        source.payload().raw_owner().charged_bytes(),
    )?;
    let resources = resources(incoming, &working, prepared, total, limits)?;
    Ok(SourceQuote {
        source,
        working,
        limits,
        workload,
        prepared,
        resources,
    })
}

fn build_plan(
    source: &ImportedValue,
    workload: DecoderWorkload,
    limits: Limits,
    prepared: Prepared,
) -> Result<DecoderPlan, Error> {
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
    observe_factory();
    let plan = model.construct().map_err(|_| Error::Factory)?;
    {
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
    } // Native pools/report retire BEFORE actual plan or config owner moves.
    Ok(plan)
}

#[cfg(not(test))]
fn observe_factory() {}

#[cfg(test)]
thread_local! { static FACTORIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

#[cfg(test)]
thread_local! { static PANIC_FACTORY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }

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
