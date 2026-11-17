//! Unissued length-only envelope for ordinary UTF-8 ImportedValue intake.
//!
//! No source argument, owner, payer, factory, allocation or startup caller.
//! Logical component bounds are NOT a payable total or a closed terminal tariff.
//! Equal lengths produce identical metadata, including for failing prefixes.
//! Qualified only for Scanner<32, false, u8> and immutable ordinary replay;
//! neither lexical/point policies nor Experiment schema validity are implied.

use super::{bootstrap_bytes, Error, ImportedValue, Work};
use super::super::{
    add, json_frame_slots, layout_bytes, number_resources, object_members_from_byte_limit,
    wide_add, wide_mul, OwnedCharged, OwnedExternalCharge, Span, CENSUS_SLOTS, DEPTH, NUMBER_DISPLAY_BYTES,
};
use serde_json::Value;
use std::mem::size_of;

const SOURCE_CAP: usize = 32 << 10;

/// Independent completed-scan ceilings, never fabricated SourceCounts. Object
/// squares include all factory history, including overwritten nested children.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LogicalBounds {
    spans: usize,
    objects: usize,
    array_children: usize,
    map_entries: usize,
    numbers: usize,
    numeric_bytes: usize,
    decoded_bytes: usize,
    longest_string: usize,
    container_depth: usize,
    members_per_object: usize,
    map_squares: u128,
}

/// Prefix-safe charged events, not identities requiring a completed scan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ScanBounds {
    cursor_positions: u128,
    byte_attempts: u128,
    peek_attempts: u128,
    emits: u128,
    scan_work: u128,
    quotation_frame_visits: u128,
    utf8_allowance: u128,
}

/// K = helper applications, V = specified visits, P = logical payload traffic.
/// Conditional on reaching the named site, not universal success assertions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CopyComponents {
    reserve_requests_on_acquisition: u128,
    capacity_checks_after_reserve_success: u128,
    copy_calls_on_exact_capacity: u128,
    initialized_bytes_on_copy: u128,
    payload_reads_on_copy: u128,
    payload_writes_on_copy: u128,
    payload_traffic_on_copy: u128,
    copy_calls_on_capacity_mismatch: u128,
    initialized_bytes_on_capacity_mismatch: u128,
    byte_element_destructors: u128,
    byte_wipe_visits: u128,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SuccessSites {
    utf8_calls_at_quote: u128,
    scans_if_utf8_passes: u128,
    number_calls_at_quote: u128,
    immutable_replays: u128,
    scanner_frame_visits: u128,
    span_initializations_max: usize,
    replay_emits_max: usize,
    number_calls_max: usize,
    census_calls: u128,
    working_grow_checks: u128,
    outer_try_build_calls: u128,
}

/// Bytes, not work/traffic or measured allocator capacities. Only fresh exact
/// raw acquisition plus an EXACT bootstrap working charge fit this descriptor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BackingBounds {
    bootstrap: usize,
    spans: usize,
    array_slots: usize,
    decoded_strings: usize,
    opaque_trees: usize,
    active_result_moves: usize,
    numeric_transition_and_error: usize,
    required_unclipped: usize,
    admitted_required: usize,
    fresh_raw_owner: usize,
    working_at_return: usize,
    retained_two_owner_peak: usize,
    /// Raw reservation plus prospective raw copy, WITH working payload AND
    /// actual OwnedExternalCharge metadata live (before try_build conversion).
    /// Add actual external live counter usage separately; no guessed baseline.
    copy_window_with_bootstrap: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Outcome {
    BeforeRawReserve,
    BeforeCopy,
    AfterCopyBeforeScan,
    ScanRefusal,
    QuoteCancellation,
    GrowOrInlineRefusal,
    PartialReplayValueOrNumberError,
    PartialCensus,
    SuccessFinalDrop,
}

/// Whole-history ceilings, not final retained Map widths. Each initialized
/// child retires at most once (replacement, error/unwind OR final drop).
/// No cleanup credit, byte erasure pass or separate Work cleanup slice exists.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RetirementBounds {
    span_slots: usize,
    value_nodes: usize,
    decoded_key_or_string_owners: usize,
    array_element_slots: usize,
    entry_insertions: usize,
    object_headers: usize,
    scanner_frames_per_created_scanner: usize,
    census_frames_if_created: usize,
    raw_vec_release_endpoints_if_created: usize,
    working_guard_release_endpoints_if_created: usize,
    native_and_json_fixed_error_slots: [usize; 2],
    caller_result_tail: usize,
    outcomes: [Outcome; 9],
}

/// Explicitly unresolved endpoints, never interpreted as zero or converted to
/// a payable total. No source custody exported by fixture construction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Held {
    RootedFixedCoverTariffAndOriginalDebit,
    AdmissionTerminalAndSiblingLifetimes,
    RawReserveFailureCapacityAndRelease,
    SourceIngressNoCopyCounterBootstrapAndUnwindProbes,
    RetainedExcessRequiresActualFundedOwners,
    StdBTreeAndImplicitDropGlueVisits,
    SharedAllocationArcLockRetryAndAllocatorInternals,
    StdUtf8MemcpyAndMapInternals,
    PinnedSerdeNumericErrorFormatterBeyondDelegatedPolicy,
    NativeStackCpuAndRss,
    WrapperCallReturnCleanupAndDescriptorEvaluationAdmission,
    PythonExtractionFormattingPanicAndUnqualifiedSerdeFeatures,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct IntakeEnvelope {
    source_bytes: usize,
    logical: LogicalBounds,
    scan: ScanBounds,
    copy: CopyComponents,
    /// Unclipped requirements: a quote can compute then refuse these.
    requirements: Work,
    /// Independent ceilings for actual admitted disjoint slices. Their sum can
    /// exceed total: total is ONE aggregate gate, not transferable slice credit.
    admitted: Work,
    backing: BackingBounds,
    success_sites: SuccessSites,
    retirement: RetirementBounds,
    held: [Held; 12],
}

fn sum(terms: &[u128]) -> Result<u128, Error> {
    terms.iter().copied().try_fold(0, wide_add)
}

fn bytes(wide: u128) -> Result<usize, Error> {
    usize::try_from(wide).map_err(|_| Error::Overflow)
}

fn experiment_intake_envelope(e: usize) -> Result<IntakeEnvelope, Error> {
    // Fixed refusal BEFORE multiplication, helpers, limits or layouts.
    if e == 0 || e > SOURCE_CAP {
        return Err(Error::SourceLimit);
    }
    let limits = crate::original_source::byte_ingress_json_limits(SOURCE_CAP);
    let x = e as u128;
    let u = wide_add(x, 1)?;
    let peek_attempts = wide_mul(32, u)?;
    // At each cursor position whitespace/digit loops consume or stop. Number
    // sign/fraction/exponent probes run forward once, not per container. Outer
    // close/comma/key/value dispatch consumes or refuses before another loop.
    // No rewinds; hex/surrogate reads use byte(). Includes final probes at EOF.
    let g = sum(&[u, x, peek_attempts])?;
    let frame_visits = wide_mul(2, DEPTH as u128)?;
    let q = sum(&[frame_visits, x, g])?;
    let k = object_members_from_byte_limit(e);
    let squares = wide_mul(k as u128, x)?;
    // Monotonic substitution into shared_build_work(..., 3), number_work and
    // json_cursor_work. No synthetic SourceCounts passed as observed facts.
    let b = sum(&[
        g, wide_mul(3, x)?, wide_mul(3, x)?, wide_mul(8, x)?, wide_mul(2, x)?,
        wide_mul(squares, wide_add(x, 3)?)?,
        wide_mul(2, wide_add(DEPTH as u128, CENSUS_SLOTS as u128)?)?,
    ])?;
    let scaling = (f64::MAX_EXP - f64::MIN_EXP) as u128
        + u128::from(f64::MANTISSA_DIGITS) + 4;
    let n = wide_add(wide_mul(6, x)?, wide_mul(x, scaling)?)?;
    let c = sum(&[3, wide_mul(2 + NUMBER_DISPLAY_BYTES as u128, x)?, x, wide_mul(3, x)?])?;
    let total = sum(&[q, b, n, c])?;
    let requirements = Work { quotation: q, construction: b, numeric: n, census: c, total };
    let admitted = Work {
        quotation: q,
        construction: b.min(limits.max_work),
        numeric: n.min(limits.max_number_work),
        census: c,
        total: total.min(limits.max_work),
    };
    let h = e.min(limits.max_depth);
    let bootstrap = bootstrap_bytes()?; // Actual 29-term target-local inventory.
    let spans = layout_bytes::<Span>(e).map_err(|_| Error::Overflow)?;
    let array_slots = layout_bytes::<Value>(e).map_err(|_| Error::Overflow)?;
    let tree_node = pllm_models::owned_backing::checked_tree_node_bytes::<(String, Value)>()
        .map_err(|_| Error::Overflow)?;
    let trees = bytes(wide_mul(wide_mul(2, x)?, tree_node as u128)?)?;
    let moves = bytes(wide_mul(wide_mul(2, wide_add(h as u128, 2)?)?, json_frame_slots()? as u128)?)?;
    let numeric = number_resources()?.total()?;
    let required = bytes(sum(&[bootstrap as u128, spans as u128, array_slots as u128,
        x, trees as u128, moves as u128, numeric as u128])?)?;
    let admitted_required = required.min(limits.max_payload_bytes);
    let raw = add(size_of::<OwnedCharged<Vec<u8>>>(), e)?;
    let working = add(admitted_required.max(bootstrap), size_of::<OwnedCharged<ImportedValue>>())?;
    let backing = BackingBounds {
        bootstrap, spans, array_slots, decoded_strings: e, opaque_trees: trees,
        active_result_moves: moves, numeric_transition_and_error: numeric,
        required_unclipped: required, admitted_required,
        fresh_raw_owner: raw, working_at_return: working,
        retained_two_owner_peak: add(raw, working)?,
        copy_window_with_bootstrap: bytes(sum(&[bootstrap as u128,
            size_of::<OwnedExternalCharge>() as u128,
            size_of::<OwnedCharged<Vec<u8>>>() as u128, wide_mul(2, x)?])?)?,
    };
    Ok(IntakeEnvelope {
        source_bytes: e,
        logical: LogicalBounds {
            spans: e, objects: e, array_children: e, map_entries: e, numbers: e,
            numeric_bytes: e, decoded_bytes: e, longest_string: e, container_depth: h,
            members_per_object: k, map_squares: squares,
        },
        scan: ScanBounds {
            cursor_positions: u, byte_attempts: u, peek_attempts, emits: x,
            scan_work: g, quotation_frame_visits: frame_visits, utf8_allowance: x,
        },
        copy: CopyComponents {
            reserve_requests_on_acquisition: 1, capacity_checks_after_reserve_success: 1,
            copy_calls_on_exact_capacity: 1, initialized_bytes_on_copy: x,
            payload_reads_on_copy: x, payload_writes_on_copy: x,
            payload_traffic_on_copy: wide_mul(2, x)?, copy_calls_on_capacity_mismatch: 0,
            initialized_bytes_on_capacity_mismatch: 0, byte_element_destructors: 0,
            byte_wipe_visits: 0,
        },
        requirements, admitted, backing,
        success_sites: SuccessSites {
            utf8_calls_at_quote: 1, scans_if_utf8_passes: 1, number_calls_at_quote: 0,
            immutable_replays: 1, scanner_frame_visits: wide_mul(2, frame_visits)?,
            span_initializations_max: e, replay_emits_max: e, number_calls_max: e,
            census_calls: 1, working_grow_checks: 1, outer_try_build_calls: 1,
        },
        retirement: RetirementBounds {
            span_slots: e, value_nodes: e, decoded_key_or_string_owners: e,
            array_element_slots: e, entry_insertions: e, object_headers: e,
            scanner_frames_per_created_scanner: DEPTH, census_frames_if_created: CENSUS_SLOTS,
            raw_vec_release_endpoints_if_created: 1, working_guard_release_endpoints_if_created: 1,
            native_and_json_fixed_error_slots: [size_of::<super::NativeError>(), size_of::<Error>()],
            caller_result_tail: size_of::<Result<OwnedCharged<ImportedValue>, Error>>(),
            outcomes: [Outcome::BeforeRawReserve, Outcome::BeforeCopy, Outcome::AfterCopyBeforeScan,
                Outcome::ScanRefusal, Outcome::QuoteCancellation, Outcome::GrowOrInlineRefusal,
                Outcome::PartialReplayValueOrNumberError, Outcome::PartialCensus, Outcome::SuccessFinalDrop],
        },
        held: [Held::RootedFixedCoverTariffAndOriginalDebit, Held::AdmissionTerminalAndSiblingLifetimes,
            Held::RawReserveFailureCapacityAndRelease, Held::SourceIngressNoCopyCounterBootstrapAndUnwindProbes,
            Held::RetainedExcessRequiresActualFundedOwners, Held::StdBTreeAndImplicitDropGlueVisits,
            Held::SharedAllocationArcLockRetryAndAllocatorInternals, Held::StdUtf8MemcpyAndMapInternals,
            Held::PinnedSerdeNumericErrorFormatterBeyondDelegatedPolicy, Held::NativeStackCpuAndRss,
            Held::WrapperCallReturnCleanupAndDescriptorEvaluationAdmission,
            Held::PythonExtractionFormattingPanicAndUnqualifiedSerdeFeatures],
    })
}

// ImportedValue retires Value before raw, outer working last. Partial strings,
// arrays and Maps retire under that guard, including duplicate replacements.
// Census borrows end before Value move/drop. Fixed errors and caller result tail
// above name endpoints only: terminal/caller backing is still Held, not bought.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan_input as parser;
    use parser::{FACTORIES, NUMBER_CALLS};
    use pllm_garble::shared_allocation::{OwnedBudget, MAX_ALLOCATION_BYTES};
    use std::alloc::Layout;

    // Independent target-local inventory, not compute_bootstrap_bytes's array.
    fn bootstrap_oracle() -> usize {
        use super::super::{Builder, ExactBuffer, Limits, Meter, NativeError, NumberResources,
            OwnedExternalCharge, Resources, Run, Scanner, SourceCounts, ValueQuote};
        let workspace = parser::census::resources_for_slots(17).unwrap().logical_workspace_bytes;
        let terms = [size_of::<Scanner<'_, '_, '_>>(), workspace, size_of::<ValueQuote>(),
            size_of::<Run<'_>>(), size_of::<ImportedValue>(), size_of::<Limits>(),
            size_of::<SourceCounts>(), size_of::<Meter>(), size_of::<Builder<'_>>(),
            size_of::<Resources>(), size_of::<Work>(), size_of::<Result<ValueQuote, Error>>(),
            size_of::<Result<ImportedValue, NativeError>>(),
            size_of::<Result<OwnedCharged<ImportedValue>, NativeError>>(),
            size_of::<Result<OwnedCharged<ImportedValue>, Error>>(), size_of::<Result<Value, Error>>(),
            size_of::<Option<Error>>(), size_of::<OwnedCharged<ImportedValue>>(),
            size_of::<OwnedCharged<Vec<u8>>>(), size_of::<OwnedExternalCharge>(),
            size_of::<ExactBuffer<Span>>(), size_of::<Vec<Span>>(), size_of::<Result<Vec<Span>, Error>>(),
            size_of::<Result<(SourceCounts, Meter), Error>>(), size_of::<Result<SourceCounts, Error>>(),
            size_of::<Result<&str, std::str::Utf8Error>>(), size_of::<parser::census::Resources>(),
            size_of::<NumberResources>(), size_of::<[u8; 4]>()];
        terms.into_iter().try_fold(0usize, usize::checked_add).unwrap()
    }

    fn lanes(w: Work) -> [u128; 4] { [w.quotation, w.construction, w.numeric, w.census] }

    // Test-only inequality oracle: no new operational admission API.
    fn fits(actual: Work, caps: Work) -> bool {
        lanes(actual).into_iter().zip(lanes(caps)).all(|(a, c)| a <= c)
            && actual.total <= caps.total
    }

    #[test]
    fn intake_all_lengths_independent_components_and_target_layouts() {
        let bootstrap = bootstrap_oracle();
        // Bounded real-counter observation, not fabricated source admission.
        // Keep this actual working guard live throughout the copy-window oracle.
        let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let baseline = root.budget().used_bytes();
        let working = root.charge_external(bootstrap).unwrap();
        let working_delta = root.budget().used_bytes().checked_sub(baseline).unwrap();
        assert_eq!(working_delta, working.charged_bytes());
        assert_eq!(working_delta, bootstrap + size_of::<OwnedExternalCharge>());
        let tree = pllm_models::owned_backing::checked_tree_node_bytes::<(String, Value)>().unwrap();
        let mut previous: Option<IntakeEnvelope> = None;
        let mut saw_work_cap = false;
        let mut saw_payload_cap = false;
        for e in 1..=32768 {
            let got = experiment_intake_envelope(e).unwrap();
            assert_eq!(got.source_bytes, e);
            let x = e as u128;
            let k = (x - 1) / 5;
            let g = 34 * x + 33;
            let q = 35 * x + 97;
            let b = g + 16 * x + k * x * (x + 3) + 98;
            let n = x * (6 + (f64::MAX_EXP - f64::MIN_EXP) as u128
                + u128::from(f64::MANTISSA_DIGITS) + 4);
            let c = 30 * x + 3;
            assert_eq!(got.logical, LogicalBounds { spans: e, objects: e, array_children: e,
                map_entries: e, numbers: e, numeric_bytes: e, decoded_bytes: e,
                longest_string: e, container_depth: e.min(16), members_per_object: k as usize,
                map_squares: k * x });
            assert_eq!(got.scan, ScanBounds { cursor_positions: x + 1, byte_attempts: x + 1,
                peek_attempts: 32 * (x + 1), emits: x, scan_work: g,
                quotation_frame_visits: 64, utf8_allowance: x });
            assert_eq!(got.requirements, Work { quotation: q, construction: b, numeric: n,
                census: c, total: q + b + n + c });
            assert_eq!(got.admitted, Work { quotation: q, construction: b.min(1_000_000_000),
                numeric: n.min(100_000_000), census: c, total: (q+b+n+c).min(1_000_000_000) });
            let p = got.backing;
            assert_eq!(p.bootstrap, bootstrap);
            assert_eq!(p.spans, Layout::array::<Span>(e).unwrap().size());
            assert_eq!(p.array_slots, Layout::array::<Value>(e).unwrap().size());
            assert_eq!(p.decoded_strings, e);
            assert_eq!(p.opaque_trees as u128, 2 * x * tree as u128);
            assert_eq!(p.active_result_moves as u128,
                2 * (e.min(16) as u128 + 2) * parser::json_frame_slots().unwrap() as u128);
            assert_eq!(p.numeric_transition_and_error, parser::number_resources().unwrap().total().unwrap());
            let required = bootstrap as u128 + p.spans as u128 + x * size_of::<Value>() as u128
                + x + 2 * x * tree as u128 + p.active_result_moves as u128
                + p.numeric_transition_and_error as u128;
            assert_eq!(p.required_unclipped as u128, required);
            assert_eq!(p.admitted_required as u128, required.min(8 << 20));
            assert_eq!(p.fresh_raw_owner, size_of::<OwnedCharged<Vec<u8>>>() + e);
            assert_eq!(p.working_at_return, p.admitted_required.max(bootstrap)
                + size_of::<OwnedCharged<ImportedValue>>());
            assert_eq!(p.retained_two_owner_peak, p.fresh_raw_owner + p.working_at_return);
            assert_eq!(p.copy_window_with_bootstrap,
                working_delta + size_of::<OwnedCharged<Vec<u8>>>() + 2*e);
            assert_eq!(got.copy, CopyComponents { reserve_requests_on_acquisition: 1,
                capacity_checks_after_reserve_success: 1, copy_calls_on_exact_capacity: 1,
                initialized_bytes_on_copy: x, payload_reads_on_copy: x, payload_writes_on_copy: x,
                payload_traffic_on_copy: 2*x, copy_calls_on_capacity_mismatch: 0,
                initialized_bytes_on_capacity_mismatch: 0, byte_element_destructors: 0, byte_wipe_visits: 0 });
            assert_eq!(got.success_sites, SuccessSites { utf8_calls_at_quote: 1, scans_if_utf8_passes: 1,
                number_calls_at_quote: 0, immutable_replays: 1, scanner_frame_visits: 128,
                span_initializations_max: e, replay_emits_max: e, number_calls_max: e,
                census_calls: 1, working_grow_checks: 1, outer_try_build_calls: 1 });
            assert_eq!([got.retirement.span_slots, got.retirement.value_nodes,
                got.retirement.decoded_key_or_string_owners, got.retirement.array_element_slots,
                got.retirement.entry_insertions, got.retirement.object_headers], [e; 6]);
            assert_eq!(got.retirement, RetirementBounds {
                span_slots: e, value_nodes: e, decoded_key_or_string_owners: e,
                array_element_slots: e, entry_insertions: e, object_headers: e,
                scanner_frames_per_created_scanner: 32, census_frames_if_created: 17,
                raw_vec_release_endpoints_if_created: 1, working_guard_release_endpoints_if_created: 1,
                native_and_json_fixed_error_slots: [size_of::<parser::NativeError>(), size_of::<Error>()],
                caller_result_tail: size_of::<Result<OwnedCharged<ImportedValue>, Error>>(),
                outcomes: [Outcome::BeforeRawReserve, Outcome::BeforeCopy, Outcome::AfterCopyBeforeScan,
                    Outcome::ScanRefusal, Outcome::QuoteCancellation, Outcome::GrowOrInlineRefusal,
                    Outcome::PartialReplayValueOrNumberError, Outcome::PartialCensus, Outcome::SuccessFinalDrop],
            });
            assert_eq!(got.held, [Held::RootedFixedCoverTariffAndOriginalDebit,
                Held::AdmissionTerminalAndSiblingLifetimes, Held::RawReserveFailureCapacityAndRelease,
                Held::SourceIngressNoCopyCounterBootstrapAndUnwindProbes,
                Held::RetainedExcessRequiresActualFundedOwners, Held::StdBTreeAndImplicitDropGlueVisits,
                Held::SharedAllocationArcLockRetryAndAllocatorInternals, Held::StdUtf8MemcpyAndMapInternals,
                Held::PinnedSerdeNumericErrorFormatterBeyondDelegatedPolicy, Held::NativeStackCpuAndRss,
                Held::WrapperCallReturnCleanupAndDescriptorEvaluationAdmission,
                Held::PythonExtractionFormattingPanicAndUnqualifiedSerdeFeatures]);
            if let Some(old) = previous {
                assert!(lanes(old.requirements).into_iter().zip(lanes(got.requirements)).all(|(a,b)| a<=b));
                assert!(old.admitted.total <= got.admitted.total);
                assert!(old.backing.required_unclipped <= p.required_unclipped);
                assert!(old.backing.retained_two_owner_peak <= p.retained_two_owner_peak);
            }
            saw_work_cap |= b > 1_000_000_000;
            saw_payload_cap |= required > 8 << 20;
            previous = Some(got);
        }
        assert!(saw_work_cap);
        // Whether payload clamp switches depends on actual target layouts.
        assert_eq!(saw_payload_cap, previous.unwrap().backing.required_unclipped > 8 << 20);
        let cap = previous.unwrap();
        assert_eq!((cap.scan.scan_work, cap.requirements.quotation, cap.requirements.census),
            (1_114_145, 1_146_977, 983_043));
        assert_eq!(cap.copy.payload_traffic_on_copy, 65536);
        assert_eq!(cap.logical.members_per_object, 6553);
        drop(working);
        assert_eq!(root.budget().used_bytes(), baseline);
    }

    #[test]
    fn intake_refuses_lengths_before_arithmetic_and_never_calls_factory() {
        FACTORIES.with(|s| s.set((0, None, None)));
        NUMBER_CALLS.with(|s| s.set(0));
        let signature: fn(usize) -> Result<IntakeEnvelope, Error> = experiment_intake_envelope;
        for e in [0, 32769, usize::MAX] { assert_eq!(signature(e), Err(Error::SourceLimit)); }
        for e in 1..=32768 { assert_eq!(signature(e), signature(e)); }
        assert_eq!(FACTORIES.with(|s| s.get().0), 0);
        assert_eq!(NUMBER_CALLS.with(|s| s.get()), 0);
    }

    #[test]
    fn intake_disjoint_lanes_one_short_aggregate_and_swapped_equal_total() {
        for e in 1..=32768 {
            let w = experiment_intake_envelope(e).unwrap().requirements;
            assert!(fits(w, w));
            for lane in 0..4 {
                let mut caps = Work { quotation: u128::MAX, construction: u128::MAX,
                    numeric: u128::MAX, census: u128::MAX, total: u128::MAX };
                match lane { 0 => caps.quotation=w.quotation-1, 1 => caps.construction=w.construction-1,
                    2 => caps.numeric=w.numeric-1, _ => caps.census=w.census-1 }
                assert!(!fits(w, caps));
            }
            assert!(!fits(w, Work { total: w.total-1, ..w }));
            let swapped = Work { quotation: w.census, census: w.quotation, ..w };
            assert_eq!(swapped.total, w.total);
            assert!(!fits(w, swapped));
        }
    }

    fn actual_quote(root: &OwnedBudget, raw: &[u8], limits: super::super::Limits)
        -> Result<super::super::ValueQuote, Error>
    {
        // Real ordinary raw helper and its actual funded original counter. No
        // fake phase, Context, source identity or fabricated ImportedValue.
        let working = root.charge_external(bootstrap_bytes().unwrap()).unwrap();
        let original = crate::original_source::acquire_raw(root, raw).unwrap();
        super::super::quote(original, working, limits)
    }

    #[test]
    fn intake_existing_quote_construct_differential_and_fixed_refusals() {
        let mut fixtures: Vec<(Vec<u8>, Option<Error>)> = vec![
            (b"null".to_vec(), None), (b"1.25".to_vec(), None),
            (br#""a\n\u03bb\ud83e\udd80""#.to_vec(), None),
            (br#""\ud800""#.to_vec(), Some(Error::Syntax)),
            (br#"{"a":{"discard":[1,2]},"\u0061":[{"x":"last"}]}"#.to_vec(), None),
            (b"[0,1,2]".to_vec(), None), (b"[0,1,2}".to_vec(), Some(Error::Syntax)),
            (vec![b'1';64], None), (vec![b'1';65], Some(Error::NumberLimit)),
            (b"1e9999".to_vec(), Some(Error::InvalidNumber)),
            (b"1e-400".to_vec(), None),
        ];
        let mut padded = vec![b' '; 32768];
        padded[32764..].copy_from_slice(b"null");
        fixtures.push((padded.clone(), None));
        padded[32767] = b'x';
        fixtures.push((padded.clone(), Some(Error::Syntax)));
        padded[32767] = 0xff;
        fixtures.push((padded, Some(Error::Syntax)));
        for depth in [16,17] {
            let text = format!("{}0{}", "[".repeat(depth), "]".repeat(depth));
            fixtures.push((text.into_bytes(), if depth==17 {Some(Error::DepthLimit)} else {None}));
        }
        fixtures.push((format!("[{}]", vec!["0";512].join(",")).into_bytes(), None));
        fixtures.push((format!("{{{}}}", (0..48).map(|i| format!("\"same-prefix-{i:03}\":{i}"))
            .collect::<Vec<_>>().join(",")).into_bytes(), None));
        for (raw, refusal) in fixtures {
            FACTORIES.with(|s| s.set((0, None, None)));
            NUMBER_CALLS.with(|s| s.set(0));
            let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
            let baseline = root.budget().used_bytes();
            let envelope = experiment_intake_envelope(raw.len()).unwrap();
            let limits = crate::original_source::byte_ingress_json_limits(32768);
            let quoted = actual_quote(&root, &raw, limits);
            if raw.as_slice() == b"1e9999" {
                assert!(quoted.is_ok(), "numeric range refusal must occur in construction");
            }
            assert_eq!(FACTORIES.with(|s| s.get().0), 0);
            assert_eq!(NUMBER_CALLS.with(|s| s.get()), 0);
            match quoted {
                Err(error) => assert_eq!(Some(error), refusal),
                Ok(quote) => {
                    // Syntax/depth/lexeme-length refusals belong to quotation,
                    // not a later factory. Numeric range refusal is delegated.
                    assert!(matches!(refusal, None | Some(Error::InvalidNumber)));
                    let counts = quote.counts();
                    let resources = quote.resources();
                    assert!(fits(resources.work, envelope.admitted));
                    assert!(counts.scan_work <= envelope.scan.scan_work);
                    assert!(counts.map_squares <= envelope.logical.map_squares);
                    for count in [counts.spans,counts.objects,counts.array_children,counts.map_entries,
                        counts.numbers,counts.number_bytes,counts.decoded_string_bytes,counts.largest_string] {
                        assert!(count <= raw.len());
                    }
                    assert!(counts.depth <= envelope.logical.container_depth);
                    assert!(parser::layout_bytes::<Span>(counts.spans).unwrap() <= envelope.backing.spans);
                    assert!(parser::layout_bytes::<Value>(counts.array_children).unwrap()
                        <= envelope.backing.array_slots);
                    assert!(counts.decoded_string_bytes <= envelope.backing.decoded_strings);
                    let tree = pllm_models::owned_backing::checked_tree_node_bytes::<(String,Value)>().unwrap();
                    assert!((counts.objects + counts.map_entries) * tree <= envelope.backing.opaque_trees);
                    assert!(parser::active_frames(counts, parser::json_frame_slots().unwrap()).unwrap()
                        <= envelope.backing.active_result_moves);
                    assert_eq!(parser::number_resources().unwrap().total().unwrap(),
                        envelope.backing.numeric_transition_and_error);
                    assert_eq!(resources.work.construction,
                        parser::shared_build_work(counts, raw.len(), 3).unwrap());
                    assert_eq!(resources.work.numeric,
                        parser::number_work(counts.number_bytes, counts.numbers).unwrap());
                    assert_eq!(resources.work.census, 3 + parser::json_cursor_work(counts).unwrap());
                    assert!(resources.required_working_payload <= envelope.backing.admitted_required);
                    assert_eq!(resources.incoming_owner_bytes, envelope.backing.fresh_raw_owner);
                    assert!(resources.working_charge_bytes <= envelope.backing.working_at_return);
                    assert!(resources.original_owners_peak_bytes <= envelope.backing.retained_two_owner_peak);
                    match quote.construct() {
                        Ok(owner) => {
                            assert_eq!(refusal, None);
                            assert_eq!(owner.payload().value(), &serde_json::from_slice::<Value>(&raw).unwrap());
                            assert_eq!(owner.payload().raw(), raw.as_slice());
                            assert_eq!(NUMBER_CALLS.with(|s| s.get()), counts.numbers);
                            drop(owner);
                        }
                        Err(error) => assert_eq!(Some(error), refusal),
                    }
                }
            }
            assert_eq!(root.budget().used_bytes(), baseline);
        }
    }

    #[test]
    fn intake_existing_scanner_failing_prefix_bound_not_completed_identities() {
        // Ordinary scanner only, borrowed fixtures only. This observation is
        // never used to authorize its own first read or fabricate source custody.
        let limits = crate::original_source::byte_ingress_json_limits(32768);
        for raw in [b"[".as_slice(), b"{\"a\":", b"[1e+", b"[0,1,2}",
            br#""\ud800""#, b"-", b"1.0e-", b"   "] {
            let envelope = experiment_intake_envelope(raw.len()).unwrap();
            let mut meter = parser::Meter { used: 0, limit: envelope.requirements.quotation };
            meter.pay(64).unwrap();
            meter.pay(raw.len() as u128).unwrap();
            std::str::from_utf8(raw).unwrap();
            let result = parser::Scanner::<32, false, u8> {
                raw, position: 0, stack: [parser::Frame::EMPTY; 32], active: 0,
                stats: parser::SourceCounts::default(), meter: &mut meter, output: None, limits,
            }.run();
            assert!(matches!(result, Err(Error::Syntax)));
            assert!(meter.used - 64 - raw.len() as u128 <= envelope.scan.scan_work);
        }
    }

    #[test]
    fn intake_existing_quote_one_short_and_cancellation_no_factory() {
        let raw = br#"{"a":[1,2,"x"]}"#;
        let limits = crate::original_source::byte_ingress_json_limits(32768);
        let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let baseline = root.budget().used_bytes();
        let quote = actual_quote(&root, raw, limits).unwrap();
        let r = quote.resources();
        drop(quote);
        assert_eq!(root.budget().used_bytes(), baseline);
        for (changed, expected) in [
            (super::super::Limits { max_work: r.work.total-1, ..limits }, Error::WorkLimit),
            (super::super::Limits { max_number_work: r.work.numeric-1, ..limits }, Error::NumberWorkLimit),
            (super::super::Limits { max_payload_bytes: r.required_working_payload-1, ..limits }, Error::Capacity),
        ] {
            FACTORIES.with(|s| s.set((0, None, None)));
            NUMBER_CALLS.with(|s| s.set(0));
            assert!(matches!(actual_quote(&root, raw, changed), Err(error) if error==expected));
            assert_eq!(FACTORIES.with(|s| s.get().0), 0);
            assert_eq!(NUMBER_CALLS.with(|s| s.get()), 0);
            assert_eq!(root.budget().used_bytes(), baseline);
        }
    }
}
