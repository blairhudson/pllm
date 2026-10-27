//! Model-neutral specialization of the original-byte parser, NOT source trust.
//!
//! Uses the parent's scanner, exact Builder::json/string/array, Number parser,
//! opaque-tree policy and iterative model census. No unbounded serde Value parse.
//! Default/std scalar Number + B-tree Map qualification and native/dependency
//! stack/allocator/RSS exclusions are IDENTICAL to `plan_input`.
//!
//! `quote` consumes genuine original owners. Callers hold bootstrap bytes on the
//! original working charge BEFORE entry. Construction grows that same charge
//! before any factory; no refund/reacquire. The complete construction reservation
//! and original byte owner remain in the returned owner. Work.total includes
//! spent quotation and ALL independently prepaid future slices. Never lend unused
//! numeric/census/build allowances to another pass or report them as zero work.
//! Opt-in whole-Value depth is at most sixteen containers; PlanImport is unchanged.

use super::{
    active_frames, add, census, check_original, json_cursor_work, json_frame_slots, layout_bytes,
    number_resources, numeric_work, replay, scan_original, shared_build_work, source_backing,
    wide_add, Builder, CensusPool, ExactBuffer, Meter, NativeError, NativeFrame, NumberResources,
    OwnedCharged, OwnedExternalCharge, Scanner, Span, CENSUS_SLOTS, NUMBER_DISPLAY_BYTES,
};
pub use super::{Error, Limits, SourceCounts};
use pllm_garble::shared_allocation::Budget;
use serde_json::Value;
use std::{cell::Cell, mem::size_of};

/// Disjoint logical work slices, not CPU fuel or refundable credit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Work {
    pub quotation: u128,
    pub construction: u128,
    pub numeric: u128,
    pub census: u128,
    pub total: u128,
}

/// These TWO original owners only. Root/siblings remain independently charged.
/// Payload omits try_build's separately admitted final inline return/padding;
/// working_charge_bytes includes that, retained excess and charge controls.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Resources {
    pub incoming_owner_bytes: usize,
    pub required_working_payload: usize,
    pub working_charge_bytes: usize,
    pub original_owners_peak_bytes: usize,
    pub work: Work,
}

/// Immutable parsed data AND ORIGINAL bytes. No detach, mutable view, or source
/// authority. Value drops before raw source; OwnedCharged drops both before the
/// working guard. Moving this whole owner into a preadmitted source-context Arc
/// preserves original counter and lineage without self-reference.
///
/// Value references cannot outlive the actual public owning type (not privacy):
/// ```compile_fail
/// use pllm_compiler::plan_input::json::ImportedValue;
/// use pllm_garble::shared_allocation::OwnedCharged;
/// fn escape<'a>(owner: OwnedCharged<ImportedValue>) -> &'a serde_json::Value {
///     owner.payload().value()
/// }
/// ```
/// Nor can downstream code mutate the data while claiming original lineage:
/// ```compile_fail
/// use pllm_compiler::plan_input::json::ImportedValue;
/// use pllm_garble::shared_allocation::OwnedCharged;
/// fn mutate(owner: &OwnedCharged<ImportedValue>) {
///     owner.payload().value()["model_type"] = serde_json::Value::Null;
/// }
/// ```
pub struct ImportedValue {
    value: Value,
    resources: Resources,
    counts: SourceCounts,
    original: OwnedCharged<Vec<u8>>,
}
impl ImportedValue {
    pub fn value(&self) -> &Value {
        &self.value
    }
    pub fn raw(&self) -> &[u8] {
        self.original.payload()
    }
    pub fn raw_owner(&self) -> &OwnedCharged<Vec<u8>> {
        &self.original
    }
    /// Counter identity only, not a new reservation or source certificate.
    pub fn budget(&self) -> &Budget {
        self.original.budget()
    }
    pub fn resources(&self) -> Resources {
        self.resources
    }
    pub fn counts(&self) -> SourceCounts {
        self.counts
    }

    /// Same raw-payload ceiling used by this owner's bounded census. Work units
    /// count cursor events separately and cannot substitute for this byte cap.
    /// Reuses the original pinned scalar Number policy, not lexical JSON length.
    pub fn traversal_payload_bound(&self) -> Result<usize, Error> {
        traversal_payload_bound(self.counts)
    }
}

/// Consumes both ORIGINAL owners; cannot be rebound to substitute bytes/baseline.
pub struct ValueQuote {
    original: OwnedCharged<Vec<u8>>,
    charge: OwnedExternalCharge,
    limits: Limits,
    counts: SourceCounts,
    resources: Resources,
}
impl ValueQuote {
    pub fn counts(&self) -> SourceCounts {
        self.counts
    }
    pub fn resources(&self) -> Resources {
        self.resources
    }

    /// Every factory and partial/error/unwind cleanup runs under the original
    /// working charge; the raw owner remains live until those borrows retire.
    /// Errors are fixed Copy codes, never a formatted serde diagnostic.
    pub fn construct(self) -> Result<OwnedCharged<ImportedValue>, Error> {
        let Self {
            original,
            mut charge,
            limits,
            counts,
            resources,
        } = self;
        // Independent slices already checked as ONE total at quote. Admit them
        // again before construction (no work/credit is reset or refunded).
        let mut meter = Meter {
            used: resources.work.quotation,
            limit: limits.max_work,
        };
        meter.pay(resources.work.construction)?;
        meter.pay(resources.work.numeric)?;
        meter.pay(resources.work.census)?;
        charge
            .resize_payload(
                charge
                    .payload_bytes()
                    .max(resources.required_working_payload),
            )
            .map_err(Error::Native)?;
        let mut failure = None;
        // Captures original by VALUE: on failure its source guard retires only
        // after Run's initialized Value/frames/partial buffers have retired.
        let result = charge.try_build(|_| {
            match (Run {
                original: &original,
                limits,
                counts,
                work: resources.work,
            })
            .run()
            {
                Ok(value) => Ok(ImportedValue {
                    value,
                    original,
                    counts,
                    resources,
                }),
                Err(error) => {
                    failure = Some(error);
                    Err(NativeError::Resource)
                }
            }
        });
        result.map_err(|native| failure.unwrap_or(Error::Native(native)))
    }
}

/// Required real working payload BEFORE fixed scanner/traversal controls exist.
pub fn bootstrap_bytes() -> Result<usize, Error> {
    let resources = census::resources_for_slots(CENSUS_SLOTS).map_err(|_| Error::Overflow)?;
    [
        size_of::<Scanner<'_, '_, '_>>(),
        resources.logical_workspace_bytes,
        size_of::<ValueQuote>(),
        size_of::<Run<'_>>(),
        size_of::<ImportedValue>(),
        size_of::<Limits>(),
        size_of::<SourceCounts>(),
        size_of::<Meter>(),
        size_of::<Builder<'_>>(),
        size_of::<Resources>(),
        size_of::<Work>(),
        size_of::<Result<ValueQuote, Error>>(),
        size_of::<Result<ImportedValue, NativeError>>(),
        size_of::<Result<OwnedCharged<ImportedValue>, NativeError>>(),
        size_of::<Result<OwnedCharged<ImportedValue>, Error>>(),
        size_of::<Result<Value, Error>>(),
        size_of::<Option<Error>>(),
        size_of::<OwnedCharged<ImportedValue>>(),
        size_of::<OwnedCharged<Vec<u8>>>(),
        size_of::<OwnedExternalCharge>(),
        size_of::<ExactBuffer<Span>>(),
        size_of::<Vec<Span>>(),
        size_of::<Result<Vec<Span>, Error>>(),
        size_of::<Result<(SourceCounts, Meter), Error>>(),
        size_of::<Result<SourceCounts, Error>>(),
        size_of::<Result<&str, std::str::Utf8Error>>(),
        size_of::<census::Resources>(),
        size_of::<NumberResources>(),
        size_of::<[u8; 4]>(),
    ]
    .into_iter()
    .try_fold(0, add)
}

fn payload(counts: SourceCounts) -> Result<usize, Error> {
    [
        bootstrap_bytes()?,
        layout_bytes::<Span>(counts.spans).map_err(|_| Error::Overflow)?,
        source_backing(counts, size_of::<Value>())?,
        active_frames(counts, json_frame_slots()?)?,
        number_resources()?.total()?,
    ]
    .into_iter()
    .try_fold(0, add)
}

fn traversal_payload_bound(counts: SourceCounts) -> Result<usize, Error> {
    add(
        counts.decoded_string_bytes,
        super::mul(counts.spans, NUMBER_DISPLAY_BYTES)?,
    )
}

fn census_limits(
    limits: Limits,
    counts: SourceCounts,
    work: u128,
) -> Result<census::Limits, Error> {
    let resources = census::resources_for_slots(CENSUS_SLOTS).map_err(|_| Error::Census)?;
    let limits = census::Limits {
        max_backing_bytes: limits.max_payload_bytes,
        max_work: usize::try_from(work).map_err(|_| Error::Overflow)?,
        max_workspace_bytes: resources.logical_workspace_bytes,
        max_json_nodes: limits.max_nodes,
        max_json_payload_bytes: traversal_payload_bound(counts)?,
        max_json_string_bytes: limits.max_string_bytes,
        max_json_key_bytes: limits.max_string_bytes,
        max_json_depth: CENSUS_SLOTS - 1,
    };
    resources.check(limits).map_err(|_| Error::Census)?;
    Ok(limits)
}

pub fn quote(
    original: OwnedCharged<Vec<u8>>,
    working: OwnedExternalCharge,
    limits: Limits,
) -> Result<ValueQuote, Error> {
    check_original(&original, &working)?;
    let (counts, meter) = scan_original(&original, limits, bootstrap_bytes()?, &working)?;
    // Whole Value, unlike Plan's separately visited attributes. Restrict BEFORE
    // factories rather than discover the native workspace/depth gate post-build.
    if counts.depth >= CENSUS_SLOTS {
        return Err(Error::DepthLimit);
    }
    let required = payload(counts)?;
    if required > limits.max_payload_bytes {
        return Err(Error::Capacity);
    }
    let construction = shared_build_work(counts, original.payload().len(), 3)?;
    let numeric = numeric_work(counts, limits)?;
    // Two Census DTO steps + Native root reset, then the SAME cursor bound.
    let census = wide_add(3, json_cursor_work(counts)?)?;
    let _ = census_limits(limits, counts, census)?; // no frame pool yet
    let work = Work {
        quotation: meter.used,
        construction,
        numeric,
        census,
        total: wide_add(
            meter.used,
            wide_add(construction, wide_add(numeric, census)?)?,
        )?,
    };
    let mut check = meter;
    check.pay(construction)?;
    check.pay(numeric)?;
    check.pay(census)?;
    let working_charge_bytes = add(
        required.max(working.payload_bytes()),
        size_of::<OwnedCharged<ImportedValue>>(),
    )?;
    let resources = Resources {
        incoming_owner_bytes: original.charged_bytes(),
        required_working_payload: required,
        working_charge_bytes,
        original_owners_peak_bytes: add(original.charged_bytes(), working_charge_bytes)?,
        work,
    };
    Ok(ValueQuote {
        original,
        charge: working,
        limits,
        counts,
        resources,
    })
}

struct Run<'a> {
    original: &'a OwnedCharged<Vec<u8>>,
    limits: Limits,
    counts: SourceCounts,
    work: Work,
}
impl Run<'_> {
    fn run(self) -> Result<Value, Error> {
        let spans = replay(self.original, self.counts, self.limits)?;
        let builder = Builder {
            raw: self.original.payload(),
            spans: &spans,
            numeric: Cell::new(Meter {
                used: 0,
                limit: self.work.numeric,
            }),
        };
        let value = builder.json(0)?;
        let limits = census_limits(self.limits, self.counts, self.work.census)?;
        let backing = {
            let mut frames = CensusPool([const { NativeFrame::empty() }; CENSUS_SLOTS]);
            let mut session =
                census::Census::new(&mut frames.0, limits).map_err(|_| Error::Census)?;
            session
                .json(&value)
                .map_err(|_| Error::Census)?
                .backing_bytes()
        }; // erase ALL frame references before Value can move/drop
        if backing > source_backing(self.counts, size_of::<Value>())? {
            return Err(Error::Capacity);
        }
        Ok(value)
    }
}

#[cfg(test)]
#[path = "json_value_input_tests.rs"]
mod tests;

// Inert lexical snapshot seam. No caller/config path selects this policy yet.
// This is UTF-8 lexical ownership, NOT original-source identity or root admission.
// UTF-16/32 admission/decoding, Python strings (including lone surrogates), numeric
// conversion, export, normalization, dtype and source binding remain upstream or
// future gates. In particular no f64 or arbitrary integer is constructed here.
// Existing shared string decoding refuses lone surrogate escapes, just as the
// ordinary serde String pipeline does; this is NOT a claim to accept all Python
// strings. No new Unicode codec or decimal conversion primitive is introduced.

const LEXICAL_SLOTS: usize = 65;

/// Independent prepaid loop slices, never CPU fuel, credit lending or refunds.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct LexicalWork {
    pub quotation: u128,
    pub replay: u128,
    pub decode: u128,
    pub classify: u128,
    pub deduplicate: u128,
    pub cleanup: u128,
    pub total: u128,
}

/// Every slice has a ceiling as well as the aggregate ceiling in LexicalLimits.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LexicalWorkLimits {
    pub quotation: u128,
    pub replay: u128,
    pub decode: u128,
    pub classify: u128,
    pub deduplicate: u128,
    pub cleanup: u128,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct LexicalLimits {
    pub max_source_bytes: usize,
    pub max_nodes: usize,
    /// Value depth, root zero, at most 64; object keys do not increase depth.
    pub max_depth: usize,
    pub max_string_bytes: usize,
    pub max_number_bytes: usize,
    pub max_payload_bytes: usize,
    pub max_work: u128,
    pub work: LexicalWorkLimits,
}
impl LexicalLimits {
    fn scanner(self) -> Limits {
        Limits {
            max_source_bytes: self.max_source_bytes,
            max_nodes: self.max_nodes,
            max_depth: self.max_depth,
            max_string_bytes: self.max_string_bytes,
            max_number_bytes: self.max_number_bytes,
            // Lexical scanner never invokes numeric helpers or their limits.
            max_number_work: 0,
            max_work: self.max_work.min(self.work.quotation),
            max_payload_bytes: self.max_payload_bytes,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct LexicalResources {
    pub incoming_owner_bytes: usize,
    pub required_working_payload: usize,
    pub working_charge_bytes: usize,
    pub original_owners_peak_bytes: usize,
    pub work: LexicalWork,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LexicalKind {
    Object,
    Array,
    String,
    Number,
    True,
    False,
    Null,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LexicalNumberClass {
    Integer,
    Float,
}

/// Only borrowed facts; indices/ranges are descriptive, never detachable source
/// authority. Span preorder includes object keys, followed by their values.
/// Construction/retirement work is quoted here; repeated downstream traversal
/// (including borrowed UTF-8 lexeme checks) needs its own funded phase.
pub(crate) struct LexicalNodeRef<'a> {
    value: &'a LexicalValue,
    index: usize,
}
impl LexicalNodeRef<'_> {
    pub(crate) fn kind(&self) -> LexicalKind {
        lexical_kind(self.value.spans[self.index].kind)
    }
    pub(crate) fn raw_range(&self) -> std::ops::Range<usize> {
        let span = self.value.spans[self.index];
        span.start..span.end
    }
    /// Exclusive preorder span index, not a byte offset.
    pub(crate) fn subtree_end(&self) -> usize {
        self.value.spans[self.index].next
    }
    /// Object entries or array elements; scalars and keys have zero children.
    pub(crate) fn immediate_child_count(&self) -> usize {
        self.value.spans[self.index].children
    }
    pub(crate) fn decoded_string(&self) -> Option<&str> {
        self.value.strings[self.index].as_deref()
    }
    pub(crate) fn number(&self) -> Option<(&str, LexicalNumberClass)> {
        let span = self.value.spans[self.index];
        if span.kind != super::Kind::Number {
            return None;
        }
        // Immutable scanner proved UTF-8 and byte boundaries. No conversion.
        let raw = std::str::from_utf8(&self.value.raw()[span.start..span.end]).ok()?;
        Some((raw, if span.decoded == 0 {
            LexicalNumberClass::Integer
        } else {
            LexicalNumberClass::Float
        }))
    }
}

/// Declaration order: owned facts BEFORE actual UTF-8 owner, before outer guard.
/// Full original allocation (including spare capacity) is retained unchanged.
pub(crate) struct LexicalValue {
    spans: Vec<Span>,
    strings: Vec<Option<String>>,
    counts: SourceCounts,
    resources: LexicalResources,
    utf8: OwnedCharged<Vec<u8>>,
}
impl LexicalValue {
    pub(crate) fn raw(&self) -> &[u8] {
        self.utf8.payload()
    }
    pub(crate) fn raw_owner(&self) -> &OwnedCharged<Vec<u8>> {
        &self.utf8
    }
    pub(crate) fn counts(&self) -> &SourceCounts {
        &self.counts
    }
    pub(crate) fn resources(&self) -> &LexicalResources {
        &self.resources
    }
    pub(crate) fn node(&self, index: usize) -> Option<LexicalNodeRef<'_>> {
        self.spans.get(index)?;
        Some(LexicalNodeRef { value: self, index })
    }
}
impl Drop for LexicalValue {
    fn drop(&mut self) {
        // Explicit destruction also gives tests an observation AFTER actual fact
        // retirement while both actual original guards still cover the backing.
        drop(std::mem::take(&mut self.strings));
        drop(std::mem::take(&mut self.spans));
        #[cfg(test)]
        lexical_retired(self.utf8.budget());
        // utf8 then drops normally; OwnedCharged's working charge drops last.
    }
}

/// Owns actual inputs, not a borrowed byte surrogate or caller budget identity.
/// Field order also handles refusal before construction or quote cancellation.
pub(crate) struct LexicalQuote {
    utf8: OwnedCharged<Vec<u8>>,
    limits: LexicalLimits,
    counts: SourceCounts,
    resources: LexicalResources,
    charge: OwnedExternalCharge,
}
impl LexicalQuote {
    pub(crate) fn counts(&self) -> &SourceCounts {
        &self.counts
    }
    pub(crate) fn resources(&self) -> &LexicalResources {
        &self.resources
    }
    pub(crate) fn construct(mut self) -> Result<OwnedCharged<LexicalValue>, Error> {
        lexical_check_work(self.resources.work, self.limits)?;
        // Admit every future slice before growth/factories; no new ledger/root.
        let mut prepaid = Meter { used: 0, limit: self.limits.max_work };
        prepaid.pay(self.resources.work.total)?;
        self.charge.resize_payload(self.charge.payload_bytes().max(
            self.resources.required_working_payload,
        )).map_err(Error::Native)?;
        let Self { utf8, limits, counts, resources, charge } = self;
        let mut failure = None;
        // By-value capture keeps incoming owner in try_build even if its inline
        // admission fails. Partial facts retire inside run before captured utf8;
        // try_build retires its closure/locals before its covering charge.
        let result = charge.try_build(|_| {
            match (LexicalRun { utf8: &utf8, limits, counts }).run() {
                Ok((spans, strings)) => Ok(LexicalValue {
                    spans, strings, counts, resources, utf8,
                }),
                Err(error) => {
                    failure = Some(error);
                    Err(NativeError::Resource)
                }
            }
        });
        result.map_err(|native| failure.unwrap_or(Error::Native(native)))
    }
}

/// Fixed controls, scanner workspace, string partials and result-move overlap.
/// Must already be funded on the SAME actual charge before quote enters scan.
/// Logical payload/backing only: no native stack, allocator, CPU or RSS claim.
pub(crate) fn lexical_bootstrap_bytes() -> Result<usize, Error> {
    let controls = [
        size_of::<Scanner<'_, '_, '_, LEXICAL_SLOTS, true>>(),
        size_of::<LexicalQuote>(),
        size_of::<LexicalRun<'_>>(),
        size_of::<LexicalPartial<'_>>(),
        size_of::<LexicalValue>(),
        size_of::<LexicalLimits>(),
        size_of::<LexicalWorkLimits>(),
        size_of::<SourceCounts>(),
        size_of::<LexicalResources>(),
        size_of::<LexicalWork>(),
        size_of::<Meter>(),
        size_of::<Builder<'_>>(),
        size_of::<OwnedCharged<Vec<u8>>>(),
        size_of::<OwnedExternalCharge>(),
        size_of::<ExactBuffer<Span>>(),
        size_of::<ExactBuffer<Option<String>>>(),
        size_of::<ExactBuffer<u8>>(),
        size_of::<Vec<Span>>(),
        size_of::<Vec<Option<String>>>(),
        size_of::<Vec<u8>>(),
        size_of::<String>(),
        size_of::<std::string::FromUtf8Error>(),
        size_of::<Result<String, Error>>(),
        size_of::<Result<Vec<u8>, Error>>(),
        size_of::<Result<Vec<Span>, Error>>(),
        size_of::<Result<Vec<Option<String>>, Error>>(),
        size_of::<Result<(Vec<Span>, Vec<Option<String>>), Error>>(),
        size_of::<Result<(SourceCounts, Meter), Error>>(),
        size_of::<Result<SourceCounts, Error>>(),
        size_of::<Result<LexicalQuote, Error>>(),
        size_of::<Result<LexicalValue, NativeError>>(),
        size_of::<Result<OwnedCharged<LexicalValue>, NativeError>>(),
        size_of::<Result<OwnedCharged<LexicalValue>, Error>>(),
        size_of::<Result<&str, std::str::Utf8Error>>(),
        size_of::<Option<Error>>(),
        size_of::<Option<String>>(),
        size_of::<std::str::Bytes<'_>>(),
        size_of::<std::slice::Iter<'_, u8>>(),
        size_of::<std::ops::Range<usize>>(),
        size_of::<(usize, usize, usize, usize, Span, Span)>(),
        size_of::<[u8; 4]>(),
    ].into_iter().try_fold(0, add)?;
    // Full origin/destination control sets, not backing copies or ABI estimates.
    super::mul(controls, 2)
}

fn lexical_payload(counts: SourceCounts) -> Result<usize, Error> {
    [
        lexical_bootstrap_bytes()?,
        layout_bytes::<Span>(counts.spans).map_err(|_| Error::Overflow)?,
        layout_bytes::<Option<String>>(counts.spans).map_err(|_| Error::Overflow)?,
        counts.decoded_string_bytes,
        // Conservative complete partial String/error overlap; no retirement credit.
        counts.largest_string,
    ].into_iter().try_fold(0, add)
}

fn lexical_work(counts: SourceCounts, bytes: usize, quotation: u128) -> Result<LexicalWork, Error> {
    let n = counts.spans as u128;
    let b = bytes as u128;
    let d = counts.decoded_string_bytes as u128;
    let l = counts.largest_string as u128;
    // Replay: measured peek/read/emit events + N initialization visits, fixed
    // scanner init/drop and N control/finish visits. No numeric helper work.
    let replay = wide_add(counts.scan_work, wide_add(super::wide_mul(n, 2)?,
        (2 * LEXICAL_SLOTS + 4) as u128)?)?;
    // Decode: complete raw escape/hex reads (<=B), writes (D), String UTF-8
    // validation (D), and per-span initialize/dispatch/push/return controls.
    let decode = [super::wide_mul(b, 4)?, super::wide_mul(d, 4)?,
        super::wide_mul(n, 8)?, 4].into_iter().try_fold(0, wide_add)?;
    // Every numeric byte at most once; every span dispatched once plus return.
    let classify = wide_add(counts.number_bytes as u128, super::wide_mul(n, 2)?)?;
    // Per object m*(m-1)/2 comparisons <= m^2. Span.next skips whole values;
    // key equality <= largest_string bytes + length check, plus two skip visits.
    let deduplicate = wide_add(n, wide_add(
        super::wide_mul(counts.map_squares, wide_add(l, 3)?)?,
        super::wide_mul(counts.map_entries as u128, 2)?,
    )?)?;
    // Full or partial strings/spans, decoded/partial byte retirement, controls.
    // Deallocation/allocator internals are excluded; no buffer erasure is claimed.
    let cleanup = [super::wide_mul(n, 2)?, d, l,
        (2 * LEXICAL_SLOTS + 8) as u128].into_iter().try_fold(0, wide_add)?;
    let total = [quotation, replay, decode, classify, deduplicate, cleanup]
        .into_iter().try_fold(0, wide_add)?;
    Ok(LexicalWork { quotation, replay, decode, classify, deduplicate, cleanup, total })
}

fn lexical_check_work(work: LexicalWork, limits: LexicalLimits) -> Result<(), Error> {
    let caps = limits.work;
    for (used, cap) in [
        (work.quotation, caps.quotation), (work.replay, caps.replay),
        (work.decode, caps.decode), (work.classify, caps.classify),
        (work.deduplicate, caps.deduplicate), (work.cleanup, caps.cleanup),
        (work.total, limits.max_work),
    ] {
        if used > cap {
            return Err(Error::WorkLimit);
        }
    }
    Ok(())
}

pub(crate) fn quote_lexical(
    utf8: OwnedCharged<Vec<u8>>,
    charge: OwnedExternalCharge,
    limits: LexicalLimits,
) -> Result<LexicalQuote, Error> {
    // Establish deterministic backing-before-working-charge retirement even on
    // early refusal. No count-dependent factory or decoded owner exists yet.
    let mut quote = LexicalQuote {
        utf8, charge, limits, counts: SourceCounts::default(),
        resources: LexicalResources::default(),
    };
    check_original(&quote.utf8, &quote.charge)?;
    let (counts, meter) = super::scan_original_policy::<LEXICAL_SLOTS, true>(
        &quote.utf8, limits.scanner(), lexical_bootstrap_bytes()?, &quote.charge,
    )?;
    let required = lexical_payload(counts)?;
    if required > limits.max_payload_bytes {
        return Err(Error::Capacity);
    }
    let work = lexical_work(counts, quote.utf8.payload().len(), meter.used)?;
    lexical_check_work(work, limits)?;
    let working_charge_bytes = add(required.max(quote.charge.payload_bytes()),
        size_of::<OwnedCharged<LexicalValue>>())?;
    quote.counts = counts;
    quote.resources = LexicalResources {
        incoming_owner_bytes: quote.utf8.charged_bytes(),
        required_working_payload: required,
        working_charge_bytes,
        original_owners_peak_bytes: add(quote.utf8.charged_bytes(), working_charge_bytes)?,
        work,
    };
    Ok(quote)
}

struct LexicalPartial<'a> {
    strings: Vec<Option<String>>,
    spans: Vec<Span>,
    budget: &'a Budget,
}
impl Drop for LexicalPartial<'_> {
    fn drop(&mut self) {
        let had_facts = !self.spans.is_empty();
        drop(std::mem::take(&mut self.strings));
        drop(std::mem::take(&mut self.spans));
        #[cfg(test)]
        if had_facts {
            lexical_retired(self.budget);
        }
        #[cfg(not(test))]
        let _ = (had_facts, self.budget);
    }
}

struct LexicalRun<'a> {
    utf8: &'a OwnedCharged<Vec<u8>>,
    limits: LexicalLimits,
    counts: SourceCounts,
}
impl LexicalRun<'_> {
    fn run(self) -> Result<(Vec<Span>, Vec<Option<String>>), Error> {
        let mut partial = LexicalPartial {
            strings: Vec::new(),
            spans: super::replay_policy::<LEXICAL_SLOTS, true>(
                self.utf8, self.counts, self.limits.scanner(),
            )?,
            budget: self.utf8.budget(),
        };
        let mut strings = super::exact::<Option<String>>(self.counts.spans)?;
        // Same string primitive and immutable replay facts, NEVER Builder::number.
        let builder = Builder {
            raw: self.utf8.payload(), spans: &partial.spans,
            numeric: Cell::new(Meter { used: 0, limit: 0 }),
        };
        for index in 0..partial.spans.len() {
            let string = if partial.spans[index].kind == super::Kind::String {
                Some(builder.string(index)?)
            } else {
                None
            };
            super::push(&mut strings, string)?;
        }
        partial.strings = super::finish(strings)?;
        lexical_classify(&mut partial.spans, self.utf8.payload(), None)?;
        lexical_deduplicate(&partial.spans, &partial.strings, None)?;
        Ok((std::mem::take(&mut partial.spans), std::mem::take(&mut partial.strings)))
    }
}

fn lexical_classify<T: super::ScanUnit>(spans: &mut [Span], raw: &[T], mut meter: Option<&mut Meter>) -> Result<(), Error> {
    for span in spans {
        if let Some(m) = meter.as_deref_mut() { m.pay(1)?; }
        if span.kind == super::Kind::Number {
            span.decoded = 0;
            for unit in &raw[span.start..span.end] {
                if let Some(m) = meter.as_deref_mut() { m.pay(1)?; }
                if matches!(unit.point(), 0x2e | 0x65 | 0x45) { span.decoded = 1; break; }
            }
        }
    }
    Ok(())
}
fn lexical_deduplicate<T: PartialEq>(spans: &[Span], strings: &[Option<T>], mut payment: Option<(&mut Meter, usize)>) -> Result<(), Error> {
    for (object, span) in spans.iter().enumerate() {
        if let Some((meter, _)) = &mut payment { meter.pay(1)?; }
        if span.kind != super::Kind::Object {
            continue;
        }
        let mut left = object + 1;
        while left < span.next {
            if let Some((meter, _)) = &mut payment { meter.pay(2)?; }
            let left_value = spans[left].next;
            let mut right = spans[left_value].next;
            while right < span.next {
                if let Some((meter, largest)) = &mut payment { meter.pay(wide_add(*largest as u128, 3)?)?; }
                if strings[left] == strings[right] {
                    return Err(Error::DuplicateField);
                }
                right = spans[spans[right].next].next;
            }
            left = spans[left_value].next;
        }
    }
    Ok(())
}

// Private point lexical bridge. Not selected by any parser/configuration path.
// Explicit domains: capacities below are raw BYTES, source/string/number POINTS;
// Span ranges and number spellings below always index immutable decoded points.
use crate::json_bytes_codec::PendingDecodedCustody;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PointError { Source(crate::original_source::Error), Input(Error) }
impl From<Error> for PointError { fn from(error: Error) -> Self { Self::Input(error) } }

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct PointWork {
    pub quotation: u128, pub replay: u128, pub strings: u128,
    pub classify: u128, pub deduplicate: u128, pub cleanup: u128,
    pub transition: u128, pub total: u128,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct PointLexicalLimits {
    pub max_raw_capacity_bytes: usize, pub max_points: usize, pub max_nodes: usize,
    pub max_depth: usize, pub max_string_points: usize, pub max_total_string_points: usize,
    pub max_number_points: usize, pub max_payload_bytes: usize, pub max_work: u128,
    /// Each field independently caps its named slice; total is also capped.
    pub work: PointWork,
}
impl PointLexicalLimits {
    fn scanner(self) -> super::PointScanLimits {
        super::PointScanLimits { nodes: self.max_nodes, depth: self.max_depth,
            string_points: self.max_string_points, number_points: self.max_number_points }
    }
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct PointResources {
    pub points: usize, pub spans: usize, pub string_points: usize, pub largest_string_points: usize,
    pub number_points: usize, pub pending_owner_bytes: usize, pub raw_owner_bytes: usize,
    pub decoded_capacity_bytes: usize, pub required_working_payload: usize,
    pub working_charge_bytes: usize, pub work: PointWork,
}

// E01 incoming move; E02 pending coverage; E03 point coverage; E04 raw coverage;
// E05 root/controller identity; E06 working identity; E07 controls floor;
// E08 explicit length/limit checks; E09 checked Q/layout/u64 preparation;
// E10 original debit; E11 quote move; E12 scanner entry; E13 scanner return;
// E14 payload quote; E15 future-work quote/caps; E16 resources finalization;
// E17 quote return; E18 incoming retirement; E19 quote retirement; E20 working
// retirement. Success pays ALL 20; pre-debit refusal has ONLY the explicitly
// approved bounded metadata envelope, no fictional charge or content access.
const POINT_ENTRY_WORK: u128 = 20;
// C01 sticky check; C02 slice/aggregate check; C03 future debit; C04 growth;
// C05 capture move; C06 inline/factory entry; C07 run entry; C08 run return;
// C09 output assembly; C10 sticky check; C11 outcome move; C12 capture retirement.
// C13 closure-layout validation: BOTH size and alignment against the single
// admitted PointCapture slot. One qualified layout operation group, NOT CPU ops;
// prepaid separately rather than silently folding this check into C05 or old 12.
const POINT_TRANSITION_WORK: u128 = 13;
// R01 partial, R02 run, R03 quote, R04 output, R05 retained-owner wrapper,
// R06 working retirement. S string slots + S spans, NO D-point erase traversal.
const POINT_RETIRE_WORK: u128 = 6;

struct PointIncoming {
    pending: OwnedCharged<PendingDecodedCustody>,
    working: OwnedExternalCharge,
}
pub(crate) struct PointLexicalQuote {
    incoming: PointIncoming,
    limits: PointLexicalLimits,
    counts: SourceCounts, // private policy units: decoded_string_bytes means POINTS
    resources: PointResources,
}
struct PointFacts { strings: Vec<Option<Vec<u32>>>, spans: Vec<Span> }
struct PointPartial<'a> { facts: PointFacts, budget: &'a Budget }
impl Drop for PointPartial<'_> {
    fn drop(&mut self) {
        #[cfg(test)]
        let had_facts = !self.facts.spans.is_empty();
        drop(std::mem::take(&mut self.facts.strings));
        drop(std::mem::take(&mut self.facts.spans));
        #[cfg(test)]
        if had_facts { lexical_retired(self.budget); }
        #[cfg(not(test))]
        let _ = self.budget;
    }
}
/// Facts retire before the WHOLE pending owner, its raw/point backing, original
/// phase/controller and codec guard. The enclosing lexical guard retires last.
/// Borrowed views confer no source authority. Subsequent traversal/consumption
/// needs its own admission; construction funding is not reusable downstream fuel.
pub(crate) struct PointLexicalCustody {
    facts: PointFacts,
    counts: SourceCounts,
    resources: PointResources,
    pending: OwnedCharged<PendingDecodedCustody>,
}
impl Drop for PointLexicalCustody {
    fn drop(&mut self) {
        drop(std::mem::take(&mut self.facts.strings));
        drop(std::mem::take(&mut self.facts.spans));
        #[cfg(test)]
        lexical_retired(self.pending.budget());
    }
}
pub(crate) struct PointNodeRef<'a> { owner: &'a PointLexicalCustody, index: usize }
impl PointNodeRef<'_> {
    pub(crate) fn point_range(&self) -> std::ops::Range<usize> {
        let span = self.owner.facts.spans[self.index]; span.start..span.end
    }
    pub(crate) fn kind(&self) -> LexicalKind { lexical_kind(self.owner.facts.spans[self.index].kind) }
    pub(crate) fn subtree_end(&self) -> usize { self.owner.facts.spans[self.index].next }
    pub(crate) fn immediate_child_count(&self) -> usize { self.owner.facts.spans[self.index].children }
    pub(crate) fn string_points(&self) -> Option<&[u32]> { self.owner.facts.strings[self.index].as_deref() }
    /// Scanner established ASCII JSON spelling, returned without conversion or
    /// revalidation traversal. Downstream repeated use needs its own payment.
    pub(crate) fn number_points(&self) -> Option<(&[u32], LexicalNumberClass)> {
        let span = self.owner.facts.spans[self.index];
        if span.kind != super::Kind::Number { return None; }
        Some((&self.owner.points()[span.start..span.end], if span.decoded == 0 {
            LexicalNumberClass::Integer
        } else { LexicalNumberClass::Float }))
    }
}
fn lexical_kind(kind: super::Kind) -> LexicalKind {
    match kind { super::Kind::Object => LexicalKind::Object, super::Kind::Array => LexicalKind::Array,
        super::Kind::String => LexicalKind::String, super::Kind::Number => LexicalKind::Number,
        super::Kind::True => LexicalKind::True, super::Kind::False => LexicalKind::False,
        super::Kind::Null => LexicalKind::Null }
}
impl PointLexicalCustody {
    pub(crate) fn points(&self) -> &[u32] { self.pending.payload().decoded().codepoints() }
    pub(crate) fn raw_owner(&self) -> &OwnedCharged<Vec<u8>> { self.pending.payload().decoded().original_owner() }
    pub(crate) fn decoded_owner(&self) -> &OwnedCharged<PendingDecodedCustody> { &self.pending }
    pub(crate) fn resources(&self) -> &PointResources { &self.resources }
    pub(crate) fn node(&self, index: usize) -> Option<PointNodeRef<'_>> {
        self.facts.spans.get(index)?; Some(PointNodeRef { owner: self, index })
    }
}

struct PointRun<'a> {
    pending: &'a OwnedCharged<PendingDecodedCustody>, counts: SourceCounts,
    limits: PointLexicalLimits, work: PointWork,
}
struct PointFactory {
    pending: OwnedCharged<PendingDecodedCustody>, counts: SourceCounts,
    limits: PointLexicalLimits, resources: PointResources,
}
struct PointCapture<'a> { factory: PointFactory, failure: &'a Cell<Option<Error>> }
impl PointCapture<'_> {
    fn build(self) -> Result<PointLexicalCustody, NativeError> {
        let result = (PointRun { pending: &self.factory.pending, counts: self.factory.counts,
            limits: self.factory.limits, work: self.factory.resources.work }).run();
        match result {
            Ok(facts) => Ok(PointLexicalCustody { facts, counts: self.factory.counts,
                resources: self.factory.resources, pending: self.factory.pending }),
            Err(error) => { self.failure.set(Some(error)); Err(NativeError::Resource) }
        }
    }
}

/// Named inventory, evaluated at compile time: no runtime inventory traversal.
/// Each entry is an actual body/return/capture slot, not a blanket multiplier.
/// Backings are separate. Physical stack/allocator internals remain excluded.
pub(crate) fn point_lexical_control_bytes() -> Result<usize, Error> {
    const BYTES: Option<usize> = {
        let slots = [
            size_of::<PointIncoming>(), // E01 entry owner
            size_of::<OwnedCharged<PendingDecodedCustody>>(), // argument move
            size_of::<OwnedExternalCharge>(), // working argument move
            size_of::<PointLexicalLimits>(), // argument
            size_of::<PointLexicalQuote>(), // E11 quote construction
            size_of::<Result<PointLexicalQuote, PointError>>(), // E17 result move
            size_of::<PointLexicalQuote>(), // C01 consuming self
            size_of::<PointFactory>(), // C05 factory destination
            size_of::<PointCapture<'_>>(), // C05 capture origin
            size_of::<PointCapture<'_>>(), // C06 single-by-value closure slot; checked
            size_of::<PointRun<'_>>(), // C07 run frame
            size_of::<PointPartial<'_>>(), // R01 partial
            size_of::<PointFacts>(), // C08 return facts
            size_of::<PointLexicalCustody>(), // C09 output construction
            size_of::<Result<PointLexicalCustody, NativeError>>(), // factory result
            size_of::<OwnedCharged<PointLexicalCustody>>(), // C11 output owner
            size_of::<Result<OwnedCharged<PointLexicalCustody>, NativeError>>(),
            size_of::<Result<OwnedCharged<PointLexicalCustody>, PointError>>(),
            size_of::<Scanner<'_, '_, '_, 65, true, u32>>(), // shared sequential passes
            size_of::<PointLexicalLimits>(), // run/quote overlap
            size_of::<PointResources>(), // E16 resource origin
            size_of::<PointWork>(), // E15 work origin
            size_of::<SourceCounts>(), // count return
            size_of::<Result<SourceCounts, Error>>(),
            size_of::<ExactBuffer<Span>>(), // replay slots
            size_of::<ExactBuffer<Option<Vec<u32>>>>(), // string slots
            size_of::<Vec<u32>>(), // ONE partial string
            size_of::<Result<Vec<u32>, Error>>(), // partial string move
            size_of::<Result<PointFacts, Error>>(),
            size_of::<Cell<Option<Error>>>(), // failure channel
            size_of::<(Meter, Meter, Meter, Meter, Meter)>(), // quote/replay/string/classify/dedup
            size_of::<(&[u32], &[Span], &[Option<Vec<u32>>], &Budget)>(), // run borrows
            size_of::<(&mut usize, &mut Meter, Option<&mut Vec<u32>>)>(), // string call
            size_of::<(usize, usize, usize, usize, Span, Span)>(), // dedup cursor
            size_of::<(usize, usize, usize, u32, u32, u32)>(), // string/probe cursor
            size_of::<(usize, usize, usize, u128, u128, u128, u128)>(), // quote/layout
            size_of::<Result<u128, Error>>(), size_of::<Result<usize, Error>>(),
            size_of::<Result<(), crate::original_source::Error>>(),
            size_of::<std::ops::Range<usize>>(), // number cursor
        ];
        let mut total = Some(0usize);
        let mut i = 0;
        while i < slots.len() {
            total = match total { Some(n) => n.checked_add(slots[i]), None => None };
            i += 1;
        }
        total
    };
    BYTES.ok_or(Error::Overflow)
}

fn point_q(p: usize) -> Result<u128, Error> {
    let p = p as u128;
    let u = wide_add(p, 1)?;
    wide_add(130, wide_add(super::wide_mul(2, p)?, wide_add(super::wide_mul(16, u)?,
        wide_add(1, super::wide_mul(6, p / 6)?)?)?)?)
}
fn point_work(counts: SourceCounts, p: usize, quotation: u128) -> Result<PointWork, Error> {
    let s = counts.spans as u128;
    let replay = wide_add(counts.scan_work, wide_add(super::wide_mul(2, s)?, 134)?)?;
    let strings = wide_add(p as u128, wide_add(super::wide_mul(6, (p / 6) as u128)?,
        wide_add(counts.decoded_string_bytes as u128, wide_add(super::wide_mul(4, s)?, 2)?)?)?)?;
    let classify = wide_add(counts.number_bytes as u128, s)?;
    let deduplicate = wide_add(s, wide_add(super::wide_mul(counts.map_squares,
        wide_add(counts.largest_string as u128, 3)?)?, super::wide_mul(2, counts.map_entries as u128)?)?)?;
    let cleanup = wide_add(super::wide_mul(2, s)?, POINT_RETIRE_WORK)?;
    let transition = POINT_TRANSITION_WORK;
    let total = wide_add(quotation, wide_add(replay, wide_add(strings,
        wide_add(classify, wide_add(deduplicate, wide_add(cleanup, transition)?)?)?)?)?)?;
    Ok(PointWork { quotation, replay, strings, classify, deduplicate, cleanup, transition, total })
}
fn point_check(work: PointWork, limits: PointLexicalLimits) -> Result<(), Error> {
    // Seven named slice caps + two aggregate caps, no runtime inventory/fold.
    // This scalar admission
    // group is E15/C02, NOT an extra unpriced span/content traversal.
    let cap = limits.work;
    if work.quotation > cap.quotation || work.replay > cap.replay || work.strings > cap.strings
        || work.classify > cap.classify || work.deduplicate > cap.deduplicate
        || work.cleanup > cap.cleanup || work.transition > cap.transition
        || work.total > cap.total || work.total > limits.max_work { return Err(Error::WorkLimit); }
    Ok(())
}
fn point_payload(counts: SourceCounts) -> Result<usize, Error> {
    let n = add(point_lexical_control_bytes()?, layout_bytes::<Span>(counts.spans).map_err(|_| Error::Overflow)?)?;
    let n = add(n, layout_bytes::<Option<Vec<u32>>>(counts.spans).map_err(|_| Error::Overflow)?)?;
    let n = add(n, layout_bytes::<u32>(counts.decoded_string_bytes).map_err(|_| Error::Overflow)?)?;
    add(n, layout_bytes::<u32>(counts.largest_string).map_err(|_| Error::Overflow)?)
}

pub(crate) fn quote_pending_lexical(pending: OwnedCharged<PendingDecodedCustody>,
    working: OwnedExternalCharge, limits: PointLexicalLimits) -> Result<PointLexicalQuote, PointError> {
    let incoming = PointIncoming { pending, working }; // E01; fixed refusal envelope
    let pending = incoming.pending.payload();
    let decoded = pending.decoded();
    let raw = decoded.original_owner();
    // E02-E07 metadata ONLY; no points or bytes inspected.
    if incoming.pending.charged_bytes() < decoded.resources().working_charge_bytes {
        return Err(Error::OriginalOwner.into());
    }
    let capacity_bytes = layout_bytes::<u32>(pending.point_capacity()).map_err(|_| Error::Overflow)?;
    if capacity_bytes > decoded.resources().point_capacity_bytes
        || raw.charged_bytes() < add(size_of::<OwnedCharged<Vec<u8>>>(), raw.payload().capacity())? {
        return Err(Error::OriginalOwner.into());
    }
    pending.lexical_identity(incoming.pending.budget()).map_err(PointError::Source)?;
    if !std::ptr::eq(raw.budget(), incoming.pending.budget())
        || !incoming.working.is_for(incoming.pending.budget()) { return Err(Error::Counter.into()); }
    let controls = point_lexical_control_bytes()?;
    if incoming.working.payload_bytes() < controls { return Err(Error::Bootstrap.into()); }
    let p = decoded.codepoints().len(); // slice length ONLY, never content
    if raw.payload().capacity() > limits.max_raw_capacity_bytes || p > limits.max_points {
        return Err(Error::SourceLimit.into());
    }
    if incoming.working.payload_bytes() > limits.max_payload_bytes { return Err(Error::Capacity.into()); }
    let q = point_q(p)?;
    let quotation = wide_add(q, POINT_ENTRY_WORK)?;
    if quotation > limits.work.quotation || quotation > limits.max_work || quotation > limits.work.total {
        return Err(Error::WorkLimit.into());
    }
    // E08-E10: ONE original attempt debit; refusal consumes custody, no retry.
    pending.lexical_debit(quotation).map_err(PointError::Source)?;
    // All scanner controls were admitted and success entry funded BEFORE here.
    let mut quote = PointLexicalQuote { incoming, limits, counts: SourceCounts::default(), resources: PointResources::default() };
    let mut meter = Meter { used: 0, limit: q };
    meter.pay(130)?; // 65 init + actual Drop's 65 retirement visits
    #[cfg(test)]
    POINT_SCANS.with(|n| n.set(n.get() + 1));
    let counts = Scanner::<65, true, u32> {
        raw: quote.incoming.pending.payload().decoded().codepoints(), position: 0,
        stack: [super::Frame::EMPTY; 65], active: 0, stats: SourceCounts::default(),
        meter: &mut meter, output: None, limits: limits.scanner(),
    }.run()?;
    if counts.decoded_string_bytes > limits.max_total_string_points { return Err(Error::StringLimit.into()); }
    let required = point_payload(counts)?;
    let actual_payload = required.max(quote.incoming.working.payload_bytes());
    if actual_payload > limits.max_payload_bytes { return Err(Error::Capacity.into()); }
    let work = point_work(counts, p, quotation)?;
    point_check(work, limits)?;
    quote.counts = counts;
    quote.resources = PointResources { points: p, spans: counts.spans,
        string_points: counts.decoded_string_bytes, largest_string_points: counts.largest_string,
        number_points: counts.number_bytes, pending_owner_bytes: quote.incoming.pending.charged_bytes(),
        raw_owner_bytes: quote.incoming.pending.payload().decoded().original_owner().charged_bytes(),
        decoded_capacity_bytes: capacity_bytes, required_working_payload: required,
        working_charge_bytes: add(actual_payload, size_of::<OwnedCharged<PointLexicalCustody>>())?, work };
    Ok(quote)
}
impl PointLexicalQuote {
    pub(crate) fn resources(&self) -> &PointResources { &self.resources }
    pub(crate) fn construct(mut self) -> Result<OwnedCharged<PointLexicalCustody>, PointError> {
        // Same bounded consuming metadata-only refusal rule as quote entry.
        // C01-C03 are paid on SUCCESS, never fictitiously on terminal refusal.
        self.incoming.pending.payload().lexical_identity(self.incoming.pending.budget()).map_err(PointError::Source)?;
        point_check(self.resources.work, self.limits)?;
        let future = self.resources.work.total.checked_sub(self.resources.work.quotation).ok_or(Error::Overflow)?;
        // ALL future slices and cleanup BEFORE growth/capture/inline/factory.
        self.incoming.pending.payload().lexical_debit(future).map_err(PointError::Source)?;
        let payload = self.resources.required_working_payload.max(self.incoming.working.payload_bytes());
        self.incoming.working.resize_payload(payload).map_err(Error::Native)?;
        let Self { incoming: PointIncoming { pending, working }, counts, limits, resources } = self;
        let failure = Cell::new(None);
        let capture = PointCapture { factory: PointFactory { pending, counts, limits, resources }, failure: &failure };
        let build = move |_: &Budget| capture.build();
        // Actual closure has ONE by-value named capture. Refuse if compiler's
        // concrete closure exceeds its admitted named slot; no ABI claim.
        if std::mem::size_of_val(&build) > size_of::<PointCapture<'_>>()
            || std::mem::align_of_val(&build) > std::mem::align_of::<PointCapture<'_>>() {
            drop(build); return Err(Error::Capacity.into());
        }
        let output = working.try_build(build).map_err(|native| PointError::Input(failure.get().unwrap_or(Error::Native(native))))?;
        output.payload().pending.payload().lexical_identity(output.budget()).map_err(PointError::Source)?;
        Ok(output)
    }
}

impl PointRun<'_> {
    fn run(self) -> Result<PointFacts, Error> {
        #[cfg(test)]
        point_factory_entry()?;
        let points = self.pending.payload().decoded().codepoints();
        let mut partial = PointPartial { facts: PointFacts { strings: Vec::new(), spans: Vec::new() }, budget: self.pending.budget() };
        let mut spans = super::exact::<Span>(self.counts.spans)?;
        for _ in 0..self.counts.spans { super::push(&mut spans, Span::EMPTY)?; }
        partial.facts.spans = super::finish(spans)?;
        let mut meter = Meter { used: 0, limit: self.work.replay };
        meter.pay(wide_add(super::wide_mul(2, self.counts.spans as u128)?, 134)?)?;
        #[cfg(test)]
        POINT_SCANS.with(|n| n.set(n.get() + 1));
        let replay = Scanner::<65, true, u32> {
            raw: points, position: 0, stack: [super::Frame::EMPTY; 65], active: 0,
            stats: SourceCounts::default(), meter: &mut meter,
            output: Some(&mut partial.facts.spans), limits: self.limits.scanner(),
        }.run()?;
        if replay.spans != self.counts.spans || replay.scan_work != self.counts.scan_work { return Err(Error::Capacity); }
        let mut strings = super::exact::<Option<Vec<u32>>>(self.counts.spans)?;
        let string_limit = self.work.strings;
        #[cfg(test)]
        let string_limit = point_test_pass_limit(PointPass::Strings, string_limit);
        let mut meter = Meter { used: 0, limit: string_limit };
        for _ in 0..self.counts.spans { meter.pay(1)?; super::push(&mut strings, None)?; }
        partial.facts.strings = super::finish(strings)?;
        meter.pay(2)?; // string-array entry/finish controls
        for index in 0..partial.facts.spans.len() {
            meter.pay(1)?; // dispatch (plus S init above)
            let span = partial.facts.spans[index];
            if span.kind == super::Kind::String {
                meter.pay(2)?; // partial-string init/return
                let mut string = Vec::new();
                #[cfg(test)]
                point_allocation_boundary()?;
                string.try_reserve_exact(span.decoded).map_err(|_| Error::Native(NativeError::Allocation))?;
                if string.capacity() != span.decoded { return Err(Error::Capacity); }
                let mut position = span.start;
                let decoded = super::point_string(points, &mut position, &mut meter, Some(&mut string), self.limits.max_string_points)?;
                if decoded != span.decoded || position != span.end { return Err(Error::Capacity); }
                partial.facts.strings[index] = Some(string);
                #[cfg(test)]
                point_after_string()?;
            }
        }
        let classify_limit = self.work.classify;
        #[cfg(test)]
        let classify_limit = point_test_pass_limit(PointPass::Classify, classify_limit);
        let mut classify = Meter { used: 0, limit: classify_limit };
        lexical_classify(&mut partial.facts.spans, points, Some(&mut classify))?;
        let dedup_limit = self.work.deduplicate;
        #[cfg(test)]
        let dedup_limit = point_test_pass_limit(PointPass::Deduplicate, dedup_limit);
        let mut dedup = Meter { used: 0, limit: dedup_limit };
        lexical_deduplicate(&partial.facts.spans, &partial.facts.strings,
            Some((&mut dedup, self.counts.largest_string)))?;
        Ok(PointFacts { strings: std::mem::take(&mut partial.facts.strings), spans: std::mem::take(&mut partial.facts.spans) })
    }
}

#[cfg(test)]
thread_local! {
    static POINT_SCANS: Cell<usize> = const { Cell::new(0) };
    static POINT_FACTORIES: Cell<usize> = const { Cell::new(0) };
    static POINT_REFUSAL: Cell<u8> = const { Cell::new(0) };
    static POINT_PASS_LIMIT: Cell<Option<(PointPass, u128)>> = const { Cell::new(None) };
}
#[cfg(test)]
#[derive(Clone, Copy, Eq, PartialEq)]
pub(crate) enum PointPass { Strings, Classify, Deduplicate }
#[cfg(test)]
pub(crate) struct PointPassLimit(Option<(PointPass, u128)>, std::marker::PhantomData<std::rc::Rc<()>>);
#[cfg(test)]
impl PointPassLimit {
    // Only lowers an existing local paid meter; NEVER changes quote prices,
    // root debits, pending identity, resources or production ceilings.
    pub(crate) fn arm(pass: PointPass, limit: u128) -> Self {
        Self(POINT_PASS_LIMIT.with(|f| f.replace(Some((pass, limit)))), std::marker::PhantomData)
    }
}
#[cfg(test)]
impl Drop for PointPassLimit {
    fn drop(&mut self) { POINT_PASS_LIMIT.with(|f| f.set(self.0)); }
}
#[cfg(test)]
fn point_test_pass_limit(pass: PointPass, paid: u128) -> u128 {
    POINT_PASS_LIMIT.with(|f| match f.get() {
        Some((selected, limit)) if selected == pass => paid.min(limit), _ => paid,
    })
}
#[cfg(test)]
pub(crate) struct PointRefusal(u8, std::marker::PhantomData<std::rc::Rc<()>>);
#[cfg(test)]
impl PointRefusal {
    // 1 injected allocation, 2 partial string Err, 3 partial panic, 4 factory panic,
    // 5 partial replay Err, 6 partial replay panic, 7 factory refusal.
    // Thread-scoped, not allocator proof.
    pub(crate) fn arm(mode: u8) -> Self { Self(POINT_REFUSAL.with(|f| f.replace(mode)), std::marker::PhantomData) }
}
#[cfg(test)]
impl Drop for PointRefusal { fn drop(&mut self) { POINT_REFUSAL.with(|f| f.set(self.0)); } }
#[cfg(test)]
pub(crate) fn point_test_counts() -> (usize, usize) { (POINT_SCANS.with(Cell::get), POINT_FACTORIES.with(Cell::get)) }
#[cfg(test)]
pub(crate) fn point_test_retirement() -> Option<(usize, usize)> { LEXICAL_RETIREMENTS.with(|v| v.borrow().last().copied()) }
#[cfg(test)]
fn point_factory_entry() -> Result<(), Error> {
    POINT_FACTORIES.with(|n| n.set(n.get() + 1));
    if POINT_REFUSAL.with(Cell::get) == 4 { panic!("point factory injected panic"); }
    if POINT_REFUSAL.with(Cell::get) == 7 { return Err(Error::Factory); }
    Ok(())
}
#[cfg(test)]
fn point_allocation_boundary() -> Result<(), Error> {
    if POINT_REFUSAL.with(Cell::get) == 1 { return Err(Error::Native(NativeError::Allocation)); }
    Ok(())
}
#[cfg(test)]
fn point_after_string() -> Result<(), Error> {
    match POINT_REFUSAL.with(Cell::get) {
        2 => Err(Error::Native(NativeError::Resource)), 3 => panic!("point partial injected panic"), _ => Ok(())
    }
}
#[cfg(test)]
pub(super) fn point_replay_emitted() -> Result<(), Error> {
    match POINT_REFUSAL.with(Cell::get) {
        5 => Err(Error::Native(NativeError::Resource)), 6 => panic!("point replay injected panic"), _ => Ok(())
    }
}

#[cfg(test)]
thread_local! {
    // Observes actual fact retirement while ACTUAL guards remain live. No Rc
    // stand-in, injectable production callback or extra allocation factory slot.
    static LEXICAL_RETIREMENTS: std::cell::RefCell<Vec<(usize, usize)>> = const {
        std::cell::RefCell::new(Vec::new())
    };
}
#[cfg(test)]
fn lexical_retired(budget: &Budget) {
    LEXICAL_RETIREMENTS.with(|events| events.borrow_mut().push((
        std::ptr::from_ref(budget) as usize, budget.used_bytes(),
    )));
}

#[cfg(test)]
#[path = "json_lexical_input_tests.rs"]
mod lexical_tests;
