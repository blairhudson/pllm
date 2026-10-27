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
        match self.value.spans[self.index].kind {
            super::Kind::Object => LexicalKind::Object,
            super::Kind::Array => LexicalKind::Array,
            super::Kind::String => LexicalKind::String,
            super::Kind::Number => LexicalKind::Number,
            super::Kind::True => LexicalKind::True,
            super::Kind::False => LexicalKind::False,
            super::Kind::Null => LexicalKind::Null,
        }
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
        for span in &mut partial.spans {
            if span.kind == super::Kind::Number {
                // For lexical NUMBER spans only, decoded is the immutable class
                // bit. Strings continue to use it as their exact decoded length.
                span.decoded = usize::from(self.utf8.payload()[span.start..span.end]
                    .iter().any(|byte| matches!(*byte, b'.' | b'e' | b'E')));
            }
        }
        lexical_deduplicate(&partial.spans, &partial.strings)?;
        Ok((std::mem::take(&mut partial.spans), std::mem::take(&mut partial.strings)))
    }
}

fn lexical_deduplicate(spans: &[Span], strings: &[Option<String>]) -> Result<(), Error> {
    for (object, span) in spans.iter().enumerate() {
        if span.kind != super::Kind::Object {
            continue;
        }
        let mut left = object + 1;
        while left < span.next {
            let left_value = spans[left].next;
            let mut right = spans[left_value].next;
            while right < span.next {
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
