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
