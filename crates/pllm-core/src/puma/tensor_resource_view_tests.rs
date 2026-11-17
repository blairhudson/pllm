//! Opt-in Core TEST CONTRACT only. These sponsors are NOT compiler SOURCE
//! fixtures, native WorkBudget payers, permits or evidence of caller migration.
use super::*;
use std::mem::size_of;
use std::panic::{catch_unwind, AssertUnwindSafe};
use super::super::attention_work::SoftmaxSink;

fn input() -> Interval { Interval::new(-32, 32).unwrap() }
fn output() -> FactoryOutput {
    build_core(&Construction::AllValid { lanes: 1 }, &FactoryParameters::AllValid).unwrap()
}

#[test]
fn checked_rows_resource_view_rejects_genuine_closed_producer_without_body_entry() {
    let full = Interval::new(i64::MIN, i64::MAX).unwrap();
    for (rows, width) in [(2, 8), (3, 128), (4, 128)] {
        let output = build_core(&Construction::CheckedRangeRows { rows, width },
            &FactoryParameters::CheckedRangeRows { incoming: full, accepted: full, fallback: 0 }).unwrap();
        assert_eq!(output.producer, ResourceProducer::CheckedRangeRows);
        assert_eq!(output.certificate(), ProducedCertificate::None);
        let before = RESOURCE_BODY_ENTRIES.with(|n| n.get());
        let descriptors = predicate_provenance::access_counts();
        let mut slots = ResourceViewSlots::new(&output);
        assert!(slots.view().is_none());
        assert_eq!(query_resource_view(&mut slots), Err(WorkPreflightError::Unsupported));
        assert!(slots.view.is_none());
        assert!(slots.view().is_none());
        assert_eq!(RESOURCE_BODY_ENTRIES.with(|n| n.get()), before);
        assert_eq!(predicate_provenance::access_counts(), descriptors);
        assert_eq!(query_resource_view(&mut slots), Err(WorkPreflightError::WorkspacePoisoned));
        assert!(output.program.resources().is_ok());
    }
}

#[test]
fn checked_rows_logical_hash_rows_are_additive_and_keep_full_node_scans() {
    // Independent old table-free rows, not a call to another constructor's quote.
    let base = LogicalEvents { writes: 192 * 14 + 21,
        validation: 192 * 192 * 3 + 192 * (64 + 64 + 64 + 64) + 21,
        transitions: 7, allocation_attempts: 0, cleanup_dispatches: 3 };
    // Split source conversion chains get their own K; never borrow legacy slack.
    // Resident: 5+3+5+4+1+2+2+1=23 (last row is second validator BODY),
    // exclusive failure=46; quote/selection=6, installed selection=2 =>54.
    let fixed = LogicalEvents { writes: 54 * 64, validation: 54 * 64,
        transitions: 54 * 8, allocation_attempts: 0, cleanup_dispatches: 54 };
    let hash = LogicalEvents { writes: 2 * 1718 + 32, validation: 64 * (275 + 2),
        transitions: 275 + 2, allocation_attempts: 0, cleanup_dispatches: 1 };
    let expected = base.checked_add(fixed).unwrap().checked_add(hash).unwrap();
    for (rows, width) in [(1, 1), (2, 64), (3, 128), (4, 128)] {
        assert_eq!(Construction::CheckedRangeRows { rows, width }.logical_events().unwrap(), expected);
    }
    // Old 21-control and per-node/N^2 rows remain literal, not raised for identity.
    assert_eq!(Construction::CheckedRange { lanes: 128 }.logical_events().unwrap(), base);
    let resource = checked_rows_resource_events().unwrap();
    assert_eq!(resource, LogicalEvents { writes: 8 * 64,
        validation: 806_701_248 + 8 * 64, transitions: 8 * 8,
        allocation_attempts: 0, cleanup_dispatches: 8 });
}

#[test]
fn checked_rows_nominal_controls_use_actual_types_and_do_not_double_embedded_identity() {
    fn conversion_slots<From, To>() -> usize {
        // Independent literal row inventory, including ownership transfer slots;
        // do NOT obtain expected bytes by calling the production row helper.
        let operands = size_of::<From>() + size_of::<From>();
        let raw_return = size_of::<Result<To, std::num::TryFromIntError>>();
        let mapper_receiver = size_of::<Result<To, std::num::TryFromIntError>>();
        let mapper = size_of::<std::num::TryFromIntError>() + size_of::<Error>();
        let mapped_return = size_of::<Result<To, Error>>();
        let try_receiver = size_of::<Result<To, Error>>();
        let branch_return = size_of::<std::ops::ControlFlow<Result<std::convert::Infallible, Error>, To>>();
        let branch_match = size_of::<std::ops::ControlFlow<Result<std::convert::Infallible, Error>, To>>();
        let residual = size_of::<Result<std::convert::Infallible, Error>>();
        let residual_formal = size_of::<Result<std::convert::Infallible, Error>>();
        operands + raw_return + mapper_receiver + mapper + mapped_return + try_receiver
            + branch_return + branch_match + residual + residual_formal + size_of::<Error>()
    }
    let usize_site = conversion_slots::<u64, usize>();
    let u64_site = conversion_slots::<usize, u64>();
    assert_eq!(checked_rows_conversion_control_bytes::<u64, usize>(), usize_site);
    assert_eq!(checked_rows_conversion_control_bytes::<usize, u64>(), u64_site);
    let validation = size_of::<CheckedRowsIdentity>()
        + usize_site + usize_site // validator's two distinct source sites
        + 6 * size_of::<usize>() + 2 * size_of::<Result<usize, Error>>()
        + size_of::<Option<usize>>() + 2 * size_of::<Result<(), Error>>()
        + size_of::<Option<Value>>() + size_of::<&'static Program>()
        + size_of::<&'static CheckedRowsIdentity>()
        + size_of::<&'static predicate_provenance::Extension>() + 2 * size_of::<&'static str>();
    assert_eq!(checked_rows_validation_control_bytes(), validation);
    let constructor = size_of::<resident_parameters::CheckedRowsArgs>()
        + size_of::<CheckedRowsIdentity>() + u64_site + u64_site
        + size_of::<Result<usize, Error>>() + 6 * size_of::<usize>()
        + size_of::<Option<usize>>() + 2 * size_of::<Interval>() + size_of::<i64>()
        + size_of::<Builder>() + size_of::<Result<Builder, Error>>()
        + size_of::<CheckedValue>() + size_of::<Result<CheckedValue, Error>>()
        + size_of::<Value>() + size_of::<ProducedCertificate>() + size_of::<ResourceProducer>()
        + size_of::<FactoryOutput>() + size_of::<Option<FactoryOutput>>()
        + 2 * size_of::<&'static mut Builder>() + size_of::<Result<Program, Error>>()
        + size_of::<Result<(), Error>>();
    assert_eq!(checked_rows_constructor_control_bytes(), constructor);
    let quote = Construction::CheckedRangeRows { rows: 4, width: 128 }.quote().unwrap();
    assert_eq!(quote.constructor_buffer_bytes, constructor);
    // Positive nominal delta on the compiling target, not a guessed ABI delta
    // or evidence that a new fixed reservation changes the enclosing peak.
    assert!(usize_site > size_of::<Result<usize, std::num::TryFromIntError>>());
    assert!(u64_site > size_of::<Result<u64, std::num::TryFromIntError>>());
    let hash = size_of::<CheckedRowsIdentity>() + 2 * size_of::<[u8; 8]>()
        + 3 * size_of::<&'static [u8]>() + size_of::<&'static mut Sha256>()
        + size_of::<&'static Program>() + size_of::<&'static CheckedRowsIdentity>()
        + size_of::<[u8; 32]>() + size_of::<sha2::digest::Output<Sha256>>()
        + 2 * size_of::<Sha256>() + size_of::<&'static predicate_provenance::Extension>()
        + 3 * size_of::<()>();
    assert_eq!(checked_rows_hash_control_bytes(), hash);
    let full_calculation = reconstructed_controls()
        - (size_of::<ResourceViewSlots<'static>>()
        + 2 * size_of::<Result<ProofFreeOutput<'static>, WorkPreflightError>>()
        + size_of::<ProofFreeOutput<'static>>() + size_of::<WorkPreflightError>()
        + size_of::<Error>() + size_of::<Resources>() + size_of::<Result<(), WorkPreflightError>>()
        + size_of::<bool>() + size_of::<&'static mut ResourceViewSlots<'static>>()
        + size_of::<&'static ResourceViewQuery<'static>>() + size_of::<&'static ProofFreeOutput<'static>>()
        + size_of::<&'static mut ResourceBodySlots>() + size_of::<ResourceBodySlots>()
        + 3 * size_of::<Result<Resources, Error>>());
    assert_eq!(checked_rows_resource_control_bytes(), full_calculation + validation + size_of::<Result<Resources, Error>>());
    // Existing TEST sponsor only: independently reconstructed controls and the
    // installed-resident envelope must pass at exact capacity, fail one short.
    // No Rows resource-view entry, native admission, paid work, or fit is implied.
    for controls in [constructor, validation, checked_rows_resident_control_bytes(),
        full_calculation + validation + size_of::<Result<Resources, Error>>()] {
        assert!(controls > 0);
        let mut exact = TestSponsor::new(0, controls);
        assert_eq!(exact.enter(0, controls, 0), Ok(()));
        let mut one_short = TestSponsor::new(0, controls - 1);
        assert_eq!(one_short.enter(0, controls, 0), Err(()));
        assert_eq!(one_short.phase, 3);
        assert!(one_short.enter(0, controls, 0).is_err());
    }
    assert_eq!(size_of::<Builder>(), size_of::<Program>());
    assert!(size_of::<ProgramOwner>() >= size_of::<Program>());
    assert!(size_of::<Executor>() >= size_of::<ProgramOwner>());
    assert!(size_of::<FactoryOutput>() >= size_of::<Program>());
    assert!(WorkPreflightWorkspace::resources().build_result_bytes >= size_of::<Result<FactoryOutput, Error>>());
    // No ABI literal (including a guessed 616-byte enum) is an acceptance gate.
    assert!(size_of::<predicate_provenance::Extension>() >= size_of::<CheckedRowsIdentity>());
}

// Independently enumerated actual logical controls, not a pricing-helper call,
// pointer-width guess, compiler callframe or owner/backing capacity deduction.
fn reconstructed_controls() -> usize {
    let wrapper = size_of::<ResourceViewSlots<'static>>()
        + 2 * size_of::<Result<ProofFreeOutput<'static>, WorkPreflightError>>()
        + size_of::<ProofFreeOutput<'static>>() + size_of::<WorkPreflightError>()
        + size_of::<Error>() + size_of::<Resources>()
        + size_of::<Result<(), WorkPreflightError>>() + size_of::<bool>()
        + size_of::<&'static mut ResourceViewSlots<'static>>()
        + size_of::<&'static ResourceViewQuery<'static>>()
        + size_of::<&'static ProofFreeOutput<'static>>()
        + size_of::<&'static mut ResourceBodySlots>()
        + size_of::<ResourceBodySlots>() + 3 * size_of::<Result<Resources, Error>>();
    let calculation = size_of::<ResourceScalars>() + size_of::<ResourceAdd>()
        + size_of::<ResourceTableFold<'static>>() + size_of::<WorkingOuter>()
        + size_of::<WorkingStep<'static>>() + size_of::<ResourceIntermediates>()
        + 4 * size_of::<Result<usize, Error>>()
        + size_of::<Result<usize, crate::exact_buffer::Error>>()
        + size_of::<Result<std::alloc::Layout, std::alloc::LayoutError>>() + size_of::<std::alloc::Layout>()
        + size_of::<Node>() + 12 * size_of::<usize>()
        // 2 add arguments; owned/shared; step/i; bits/n; immutable tables;
        // exact-layout count; first outer operand; extension backing return.
        + size_of::<&'static Box<[Affine]>>() + size_of::<Option<&'static Box<[Affine]>>>()
        + 2 * size_of::<Option<usize>>() + size_of::<Resources>()
        + 4 * size_of::<&'static Program>()
        + 2 * size_of::<&'static predicate_provenance::Extension>()
        + size_of::<&'static ResourceAdd>()
        + 2 * size_of::<Value>() + size_of::<Option<Value>>() + 2 * size_of::<bool>();
    wrapper.checked_add(calculation).unwrap()
}

// Frozen pre-census equation, evaluated at the SAME current ABI as the complete
// equation. Not a literal pre-representation wrapper size or a quoted discount.
fn reconstructed_pre_census_controls() -> usize {
    let wrapper = size_of::<ResourceViewSlots<'static>>()
        + 2 * size_of::<Result<ProofFreeOutput<'static>, WorkPreflightError>>()
        + size_of::<ProofFreeOutput<'static>>() + size_of::<WorkPreflightError>()
        + size_of::<Error>() + size_of::<Resources>()
        + size_of::<Result<(), WorkPreflightError>>() + size_of::<bool>()
        + size_of::<&'static mut ResourceViewSlots<'static>>()
        + size_of::<&'static ResourceViewQuery<'static>>()
        + size_of::<&'static ProofFreeOutput<'static>>()
        + size_of::<&'static mut ResourceBodySlots>()
        + size_of::<ResourceBodySlots>() + 3 * size_of::<Result<Resources, Error>>();
    wrapper + size_of::<ResourceScalars>() + size_of::<ResourceAdd>()
        + size_of::<ResourceTableFold<'static>>() + size_of::<WorkingOuter>()
        + size_of::<WorkingStep<'static>>() + size_of::<ResourceIntermediates>()
        + 4 * size_of::<Result<usize, Error>>()
        + size_of::<Result<usize, crate::exact_buffer::Error>>()
        + size_of::<Result<std::alloc::Layout, std::alloc::LayoutError>>()
        + size_of::<std::alloc::Layout>() + size_of::<Node>() + 8 * size_of::<usize>()
        + size_of::<&'static Box<[Affine]>>() + size_of::<Option<&'static Box<[Affine]>>>()
        + 2 * size_of::<Option<usize>>() + size_of::<Resources>()
}

struct TestSponsor {
    work: u64,
    burned: u64,
    controls: usize,
    body_controls: usize,
    phase: u8,
}
impl TestSponsor {
    fn new(work: u64, controls: usize) -> Self {
        Self { work, burned: 0, controls, body_controls: controls, phase: 0 }
    }
    fn enter(&mut self, work: u64, controls: usize, expected_phase: u8) -> Result<(), ()> {
        // Original test sponsor admits controls and debits BEFORE any leaf entry.
        if self.phase != expected_phase { return Err(()); }
        self.phase += 1;
        let capacity = if expected_phase == 0 { self.controls } else { self.body_controls };
        if capacity < controls { self.phase = 3; return Err(()); }
        if self.work < work { self.phase = 3; self.burned += self.work; self.work = 0; return Err(()); }
        self.work -= work;
        self.burned += work;
        Ok(())
    }
}
fn sponsored_query<'p>(sponsor: &mut TestSponsor, output: &'p FactoryOutput)
    -> Result<ResourceViewSlots<'p>, ()> {
    sponsor.enter(256, reconstructed_controls(), 0)?;
    let mut slots = ResourceViewSlots::new(output);
    query_resource_view(&mut slots).map_err(|_| ())?;
    Ok(slots)
}
fn sponsored_body(sponsor: &mut TestSponsor, view: &ProofFreeOutput<'_>)
    -> Result<ResourceBodySlots, ()> {
    sponsor.enter(1_198_656, reconstructed_controls(), 1)?;
    let mut slots = ResourceBodySlots::empty();
    resources_proof_free(view, &mut slots).map_err(|_| ())?;
    Ok(slots)
}

#[test]
fn resource_view_independent_event_rows_and_concrete_control_layout() {
    // Each named row's source block has <=8 reads/tests/writes/transitions,
    // with a second 32-event exclusive failure block, not guessed spare margin.
    let query_rows = ["phase burn", "closed producer", "Absent match", "resident result/retire"];
    let body_rows = ["endpoint entry", "checker bypass", "table backing", "scratch/max",
        "body phase burn", "body result/retire"];
    let outer_nodes = (0..192u64).count() as u64;
    let liveness_candidates: u64 = (0..192u64).map(|i| (0..=i).count() as u64).sum();
    assert_eq!((outer_nodes, liveness_candidates), (192, 18_528));
    let base = (3 + outer_nodes + liveness_candidates) * 64;
    let query = query_rows.len() as u64 * (32 + 32);
    let body = base + body_rows.len() as u64 * (32 + 32);
    assert_eq!((query, body, query + body), (256, 1_198_656, 1_198_912));
    assert_eq!((RESOURCE_VIEW_QUERY, RESOURCE_VIEW_BODY, RESOURCE_VIEW_TOTAL), (query, body, query + body));
    // Init + push*192 + finish (now also scalar/None cache setup) + header +
    // rule/extra outer transition*192 + six publication rows + drop. Existing
    // dependency/hash body work is moved, not repeated. No cache walk in Absent.
    let maintenance = 64 + 128 * 192 + 128 + 64 + 64 * 192 + 6 * 64 + 64;
    assert_eq!(maintenance, 37_568);
    assert_eq!(PROOF_FREE_CONSTRUCTION_MAINTENANCE, maintenance);
    assert_eq!(PROOF_FREE_CONSTRUCTION_EVENTS, LogicalEvents { writes: 1, validation: maintenance,
        transitions: 0, allocation_attempts: 0, cleanup_dispatches: 1 });
    assert_eq!(Program::full_resource_validation_visits().unwrap(), 806_701_248);
    assert_eq!(resource_view_control_bytes(), reconstructed_controls());
    assert_eq!(RESOURCE_VIEW_CONTROL_BYTES, reconstructed_controls());
    assert_eq!(WorkPreflightWorkspace::resources().build_result_bytes,
        size_of::<Result<FactoryOutput, Error>>());
    assert!(size_of::<Program>() >= size_of::<predicate_provenance::Extension>());
    // No assumption that Absent shrinks the physical Program.
    let p = output();
    assert_eq!(p.program.immutable_template_bytes().unwrap(),
        size_of::<Program>() + 2 * size_of::<usize>());
}

// Independent typed census: do not call the production accessor layout helper.
fn reconstructed_accessor_controls() -> usize {
    let receiver = size_of::<&'static ResourceViewSlots<'static>>();
    let complete_operand = size_of::<ResourcePhase>();
    let comparison_operands = size_of::<&'static ResourcePhase>()
        + size_of::<&'static ResourcePhase>();
    let comparison_formals = size_of::<&'static ResourcePhase>()
        + size_of::<&'static ResourcePhase>();
    let equality_result = size_of::<bool>();
    let ne_result = size_of::<bool>();
    let branch_operand = size_of::<bool>();
    let option_borrow = size_of::<&'static Option<ProofFreeOutput<'static>>>();
    let match_binding = size_of::<&'static ProofFreeOutput<'static>>();
    let early_none = size_of::<Option<&'static ProofFreeOutput<'static>>>();
    let matched_some = size_of::<Option<&'static ProofFreeOutput<'static>>>();
    let matched_none = size_of::<Option<&'static ProofFreeOutput<'static>>>();
    let returned = size_of::<Option<&'static ProofFreeOutput<'static>>>();
    receiver + complete_operand + comparison_operands + comparison_formals
        + equality_result + ne_result + branch_operand + option_borrow + match_binding
        + early_none + matched_some + matched_none + returned
}

#[test]
fn resource_view_accessor_independent_128_event_rows_and_typed_controls() {
    // Separate source rows: derived ne/equality-negation is a typed logical
    // primitive, not a descriptor classifier or an unpriced helper walk.
    let phase_gate = 32 + 32; // operands/formals, Complete, ne, bool branch, early None/return
    let extraction = 32 + 32; // Option borrow/discriminant/branch/binding, constructors/return/retire
    assert_eq!((phase_gate, extraction, phase_gate + extraction), (64, 64, 128));
    assert_eq!(RESOURCE_VIEW_ACCESSOR, 128);
    let bytes = reconstructed_accessor_controls();
    assert_eq!(RESOURCE_VIEW_ACCESSOR_CONTROL_BYTES, bytes);
    assert_eq!(resource_view_accessor_control_bytes(), bytes);
    // Local test admission only. Caller-held Option/match and admission controls
    // are external; this never claims Compiler SOURCE funding.
    let actual = output();
    let mut query = ResourceViewSlots::new(&actual);
    query_resource_view(&mut query).unwrap();
    let before = (predicate_provenance::access_counts(),
        RESOURCE_QUERY_ENTRIES.with(|n| n.get()), RESOURCE_BODY_ENTRIES.with(|n| n.get()));
    for (work, capacity, accepted) in [(127, bytes, false), (128, bytes - 1, false), (128, bytes, true)] {
        let mut sponsor = TestSponsor::new(work, capacity);
        let admitted = sponsor.enter(128, bytes, 0);
        assert_eq!(admitted.is_ok(), accepted);
        if admitted.is_ok() { assert!(query.view().is_some()); }
        else { assert!(sponsor.enter(128, bytes, 0).is_err()); }
    }
    assert_eq!((predicate_provenance::access_counts(),
        RESOURCE_QUERY_ENTRIES.with(|n| n.get()), RESOURCE_BODY_ENTRIES.with(|n| n.get())), before);
}

#[test]
fn resource_view_accessor_complete_gate_and_refused_reentry_are_sticky() {
    let actual = output();
    let mut query = ResourceViewSlots::new(&actual);
    assert!(query.view().is_none());
    // An internal resident view alone must never bypass the phase gate.
    query.view = Some(ProofFreeOutput { program: &actual.program });
    assert!(query.view().is_none());
    query.query.phase = ResourcePhase::Burned;
    assert!(query.view().is_none());
    query.query.phase = ResourcePhase::Complete;
    assert!(query.view().is_some());
    query.view = None;
    assert!(query.view().is_none()); // completed-but-empty defensive match arm

    let mut query = ResourceViewSlots::new(&actual);
    query_resource_view(&mut query).unwrap();
    let endpoint = query.view().unwrap().program as *const Program;
    let result = query.result;
    assert_eq!(query_resource_view(&mut query), Err(WorkPreflightError::WorkspacePoisoned));
    assert!(query.query.phase == ResourcePhase::Complete);
    assert_eq!(query.result, result);
    assert_eq!(query.view().unwrap().program as *const Program, endpoint);
}

#[test]
fn resource_view_checked_rows_extension_rejects_independently_of_accepted_producer() {
    // Test-private corruption, NOT genuine Rows producer rejection above.
    // Every accepted tag must still refuse either valid-looking or adversarial
    // CheckedRowsIdentity without reading its descriptor or entering BODY.
    for producer in [ResourceProducer::Negative, ResourceProducer::EqualZero,
        ResourceProducer::Rescale, ResourceProducer::Scale18, ResourceProducer::CheckedRange,
        ResourceProducer::AllValid, ResourceProducer::RmsCertified,
        ResourceProducer::RmsWithInputNorm, ResourceProducer::Rotary] {
        for (rows, width) in [(1, 1), (0, 0), (u64::MAX, u64::MAX)] {
            let mut actual = output();
            actual.producer = producer;
            actual.program.extension = predicate_provenance::Extension::CheckedRowsIdentity(
                CheckedRowsIdentity { rows, width });
            let before = predicate_provenance::access_counts();
            let body_entries = RESOURCE_BODY_ENTRIES.with(|n| n.get());
            let mut query = ResourceViewSlots::new(&actual);
            assert!(query.view().is_none());
            assert_eq!(query_resource_view(&mut query), Err(WorkPreflightError::Unsupported));
            assert!(query.view().is_none());
            assert_eq!(query.result, Err(WorkPreflightError::Unsupported));
            assert_eq!(query_resource_view(&mut query), Err(WorkPreflightError::WorkspacePoisoned));
            assert!(query.view().is_none());
            assert_eq!(query.result, Err(WorkPreflightError::Unsupported));
            assert_eq!(predicate_provenance::access_counts(), before);
            assert_eq!(RESOURCE_BODY_ENTRIES.with(|n| n.get()), body_entries);
        }
    }
}

#[test]
fn resource_view_test_sponsor_exact_and_one_short_debits_precede_both_entries() {
    let output = output();
    let layout = reconstructed_controls();
    let query_entries = || RESOURCE_QUERY_ENTRIES.with(|n| n.get());
    let body_entries = || RESOURCE_BODY_ENTRIES.with(|n| n.get());
    for (work, bytes) in [(255, layout), (256, layout - 1)] {
        let before = query_entries();
        let mut sponsor = TestSponsor::new(work, bytes);
        assert!(sponsored_query(&mut sponsor, &output).is_err());
        assert_eq!(query_entries(), before);
        assert!(sponsored_query(&mut sponsor, &output).is_err());
    }
    let mut sponsor = TestSponsor::new(256, layout);
    let before = query_entries();
    let mut query = sponsored_query(&mut sponsor, &output).ok().unwrap();
    assert_eq!(query_entries(), before + 1);
    assert_eq!((sponsor.work, sponsor.burned), (0, 256));
    for work in [1_198_655, 1_198_656] {
        let mut sponsor = TestSponsor::new(256 + work, layout);
        // A separately bounded BODY control admission; original query controls
        // stay retained, never retired/reused to finance a later phase.
        if work == 1_198_656 { sponsor.body_controls = layout - 1; }
        let body_query = sponsored_query(&mut sponsor, &output).ok().unwrap();
        let before = body_entries();
        assert!(sponsored_body(&mut sponsor, body_query.view.as_ref().unwrap()).is_err());
        assert_eq!(body_entries(), before);
        assert!(sponsored_body(&mut sponsor, body_query.view.as_ref().unwrap()).is_err());
    }
    let mut sponsor = TestSponsor::new(1_198_912, layout);
    let body_query = sponsored_query(&mut sponsor, &output).ok().unwrap();
    let before = body_entries();
    let mut body = sponsored_body(&mut sponsor, body_query.view.as_ref().unwrap()).ok().unwrap();
    let resources = body.resources.unwrap();
    assert_eq!(body_entries(), before + 1);
    assert_eq!(resources, output.program.resources().unwrap());
    assert_eq!(body.resources, Some(resources));
    assert_eq!((sponsor.work, sponsor.burned), (0, 1_198_912));
    assert_eq!(resources_proof_free(body_query.view.as_ref().unwrap(), &mut body), Err(Error::Phase));
    assert_eq!(query_resource_view(&mut query), Err(WorkPreflightError::WorkspacePoisoned));
}

#[test]
fn resource_view_all_closed_producers_and_resident_issuers_preserve_resource_only_identity() {
    let gamma = [262_144; 2];
    let cosine = [262_144];
    let sine = [0];
    let cases = [
        (Construction::Negative { lanes: 2 }, FactoryParameters::Negative { input: input() }, ResourceProducer::Negative),
        (Construction::EqualZero { lanes: 2 }, FactoryParameters::EqualZero { input: input() }, ResourceProducer::EqualZero),
        (Construction::Builder { lanes: 2 }, FactoryParameters::Rescale { input: input(), shift: 5 }, ResourceProducer::Rescale),
        (Construction::Builder { lanes: 2 }, FactoryParameters::Scale18 { input: input(), coefficient: 262_144 }, ResourceProducer::Scale18),
        (Construction::CheckedRange { lanes: 2 }, FactoryParameters::CheckedRange { incoming: input(), accepted: input(), fallback: 0 }, ResourceProducer::CheckedRange),
        (Construction::AllValid { lanes: 2 }, FactoryParameters::AllValid, ResourceProducer::AllValid),
        (Construction::RmsNormCertified { rows: 1, width: 2, gamma_len: 2 }, FactoryParameters::RmsNormCertified { input: input(), gamma: &gamma }, ResourceProducer::RmsCertified),
        (Construction::RmsNormWithInputNorm { rows: 1, width: 2, gamma_len: 2 }, FactoryParameters::RmsNormWithInputNorm { input: Interval::new(0, 0).unwrap(), gamma: &gamma, squared_norm: 0 }, ResourceProducer::RmsWithInputNorm),
        (Construction::RotaryHalf { rows: 1, width: 2, cosine_len: 1, sine_len: 1 }, FactoryParameters::RotaryHalf { input: input(), cosine: &cosine, sine: &sine }, ResourceProducer::Rotary),
    ];
    for (construction, parameters, producer) in cases {
        let direct = build_core(&construction, &parameters).unwrap();
        assert_eq!(direct.producer, producer);
        let before = (direct.program.digest(), direct.program.work_budget(), direct.certificate());
        let mut query = ResourceViewSlots::new(&direct);
        query_resource_view(&mut query).unwrap();
        let mut body = ResourceBodySlots::empty();
        assert_eq!(resources_proof_free(query.view().unwrap(), &mut body).unwrap(), direct.program.resources().unwrap());
        assert_eq!(before, (direct.program.digest(), direct.program.work_budget(), direct.certificate()));
        // Actual resident issuer path, including BOTH sealed RMS origins and
        // rotary's separate finish literal. Native construction is already paid
        // by the existing test WorkBudget protocol; not SOURCE fee evidence.
        let mut source = Some(construction.resident_parameters(parameters));
        let mut resident = WorkPreflightWorkspace::resident_schedule();
        resident.install(&mut source, None, 1).unwrap();
        resident.bind().unwrap();
        resident.preflight(&mut WorkPreflightWorkspace::empty()).unwrap();
        let mut work = WorkBudget::default();
        resident.admit(&mut CallerWorkLedger::new(&mut work, u64::MAX, u64::MAX).unwrap()).unwrap();
        let mut actual = None;
        resident.build_into(&mut actual, &mut None).unwrap();
        let actual = actual.unwrap();
        assert_eq!(actual.producer, producer);
        assert_eq!((actual.program.digest(), actual.program.work_budget(), actual.certificate()), before);
        let mut query = ResourceViewSlots::new(&actual);
        query_resource_view(&mut query).unwrap();
        assert!(query.view().is_some());
        let mut body = ResourceBodySlots::empty();
        assert_eq!(resources_proof_free(query.view().unwrap(), &mut body).unwrap(), actual.program.resources().unwrap());
    }
}

#[test]
fn resource_view_resident_full_zero_tail_schedules_postcheck_each_output_before_take() {
    fn postcheck_then_consume(output: FactoryOutput) -> Resources {
        // Existing local TEST sponsors only: not SOURCE funding, admission of
        // the enclosing caller frame, or a memory-fit claim.
        let resources = {
            let mut leaf = TestSponsor::new(1_198_912, reconstructed_controls());
            let query = sponsored_query(&mut leaf, &output).ok().unwrap();
            let mut accessor = TestSponsor::new(128, reconstructed_accessor_controls());
            accessor.enter(128, reconstructed_accessor_controls(), 0).unwrap();
            let view = query.view().unwrap();
            let body = sponsored_body(&mut leaf, view).ok().unwrap();
            let resources = body.resources.unwrap();
            assert_eq!(query.result, Ok(()));
            assert_eq!(body.result, Ok(()));
            // Compare against THIS original output, while all controls/results
            // and the original affine owner remain resident, before take.
            assert_eq!(resources, output.program().resources().unwrap());
            assert_eq!((leaf.work, leaf.burned), (0, 1_198_912));
            assert_eq!((accessor.work, accessor.burned), (0, 128));
            resources
        }; // body/result, view, query/result retire before output consumption
        let (program, certificate) = output.into_parts();
        assert_eq!(resources, program.resources().unwrap());
        assert!(matches!(certificate, ProducedCertificate::GeneralOutputRowSquared(_)));
        resources
    }

    let gamma = [262_144; 2];
    // Expected counts are literal per-template outputs, NOT invocation count
    // (two full invocations still build/query one full template).
    for (count, has_tail, expected_entries) in [(2, false, 1), (0, false, 0), (2, true, 2)] {
        let before = (RESOURCE_QUERY_ENTRIES.with(|n| n.get()),
            RESOURCE_BODY_ENTRIES.with(|n| n.get()));
        let rows_before = row_entries();
        let mut full_source = Some(Construction::RmsNormCertified {
            rows: 2, width: 2, gamma_len: 2,
        }.resident_parameters(FactoryParameters::RmsNormCertified { input: input(), gamma: &gamma }));
        // Tail uses the other genuine sealed RMS origin and its own geometry
        // and input-norm obligation; never substitute full's returned Resources.
        let mut tail_source = has_tail.then(|| Construction::RmsNormWithInputNorm {
            rows: 1, width: 2, gamma_len: 2,
        }.resident_parameters(FactoryParameters::RmsNormWithInputNorm {
            input: Interval::new(0, 0).unwrap(), gamma: &gamma, squared_norm: 0,
        }));
        let mut resident = WorkPreflightWorkspace::resident_schedule();
        resident.install(&mut full_source,
            if has_tail { Some(&mut tail_source) } else { None }, count).unwrap();
        assert!(full_source.is_none() && tail_source.is_none());
        resident.bind().unwrap();
        let mut workspace = WorkPreflightWorkspace::empty();
        let mut work = WorkBudget::default();
        let mut full = None;
        let mut tail = None;
        if count == 0 {
            assert_eq!(resident.preflight(&mut workspace), Err(WorkPreflightError::EmptySchedule));
            assert!(workspace.is_poisoned());
            assert!(resident.summary().is_none());
            assert_eq!(resident.admit(&mut CallerWorkLedger::new(&mut work, u64::MAX, u64::MAX).unwrap()),
                Err(WorkPreflightError::Native(Error::Phase)));
            assert_eq!(resident.build_into(&mut full, &mut tail), Err(Error::Phase));
            assert!(full.is_none() && tail.is_none());
            assert_eq!(work, WorkBudget::default());
            assert_eq!(row_entries(), rows_before);
        } else {
            resident.preflight(&mut workspace).unwrap();
            let summary = resident.summary().unwrap();
            assert_eq!(summary.full.invocations, 2);
            assert_eq!(summary.tail.is_some(), has_tail);
            let expected_work = match summary.tail {
                Some(tail) => {
                    assert_eq!(tail.invocations, 1);
                    summary.full.unit_work.checked_mul(2).unwrap().checked_add(tail.unit_work).unwrap()
                },
                None => summary.full.unit_work.checked_mul(2).unwrap(),
            };
            assert_eq!(summary.work, expected_work);
            resident.admit(&mut CallerWorkLedger::new(&mut work, u64::MAX, u64::MAX).unwrap()).unwrap();
            resident.build_into(&mut full, &mut tail).unwrap();
            assert_eq!(work, expected_work);
            // Actual schedule completion is sticky; no second publication.
            assert_eq!(resident.build_into(&mut None, &mut None), Err(Error::Phase));
            assert_eq!(tail.is_some(), has_tail);
            let full = full.unwrap();
            assert_eq!(full.producer, ResourceProducer::RmsCertified);
            assert_eq!(full.program().public_affines[0].len(), 4);
            assert_eq!(full.program().work_budget(), summary.full.unit_work);
            let full_resources = postcheck_then_consume(full);
            assert_eq!((RESOURCE_QUERY_ENTRIES.with(|n| n.get()) - before.0,
                RESOURCE_BODY_ENTRIES.with(|n| n.get()) - before.1), (1, 1));
            if let Some(tail) = tail {
                assert_eq!(tail.producer, ResourceProducer::RmsWithInputNorm);
                assert_eq!(tail.program().public_affines[0].len(), 2);
                assert_eq!(tail.program().work_budget(), summary.tail.unwrap().unit_work);
                let tail_resources = postcheck_then_consume(tail);
                assert_ne!(tail_resources.public_table_bytes, full_resources.public_table_bytes);
            }
        }
        assert_eq!((RESOURCE_QUERY_ENTRIES.with(|n| n.get()) - before.0,
            RESOURCE_BODY_ENTRIES.with(|n| n.get()) - before.1), (expected_entries, expected_entries));
    }
}

#[test]
fn resource_view_wrong_producer_and_indexed_refuse_without_descriptor_or_checker_access() {
    for (construction, parameters) in [
        (Construction::GuardedSilu { lanes: 2 }, FactoryParameters::GuardedSilu { input: input() }),
        (Construction::SoftmaxCertified { rows: 1, width: 2 }, FactoryParameters::SoftmaxCertified { input: input() }),
    ] {
        let mut actual = build_core(&construction, &parameters).unwrap();
        // Numeric profile/certificate cannot change the genuine producer tag.
        actual.program.numeric_profile = RMS_NORM_PROFILE;
        actual.certificate = ProducedCertificate::GeneralOutputRowSquared(0);
        let mut sponsor = TestSponsor::new(256, reconstructed_controls());
        sponsor.enter(256, reconstructed_controls(), 0).unwrap();
        let before = predicate_provenance::access_counts();
        let body_entries = RESOURCE_BODY_ENTRIES.with(|n| n.get());
        let mut query = ResourceViewSlots::new(&actual);
        assert!(query.view().is_none());
        assert_eq!(query_resource_view(&mut query), Err(WorkPreflightError::Unsupported));
        assert_eq!(query.result, Err(WorkPreflightError::Unsupported));
        assert!(query.view.is_none());
        assert!(query.view().is_none());
        assert_eq!(predicate_provenance::access_counts(), before);
        assert_eq!(RESOURCE_BODY_ENTRIES.with(|n| n.get()), body_entries);
        assert_eq!(query_resource_view(&mut query), Err(WorkPreflightError::WorkspacePoisoned));
        assert_eq!(query.result, Err(WorkPreflightError::Unsupported));
        assert_eq!((sponsor.work, sponsor.burned), (0, 256));
        assert!(sponsored_query(&mut sponsor, &actual).is_err());
    }
    // Test-private adversarial publication: Indexed with empty owners, bad
    // capacities, nonzero backing, rules, tables, proofs all refuse in QUERY.
    // Genuine production cannot replace FactoryOutput.program like this.
    for corruption in 0..7 {
        let mut actual = output();
        let mut indexed = Builder::indexed(1, input()).unwrap();
        let owners = indexed.0.extension.owners_mut().unwrap();
        match corruption {
            0 => {},
            1 => owners.capacity = 1,
            2 => owners.backing_bytes = usize::MAX,
            3 => owners.rules[0] = Some(predicate_provenance::Rule::NestedDifference { packed: Value(255) }),
            4 => owners.insert_record_for_test(Value(0), &[IndexedAffineRow { constant: 0, len: 255,
                terms: [super::super::super::arithmetic::IndexedTerm::default(); 2] }]),
            5 => owners.check_visits = u64::MAX,
            _ => {},
        }
        if corruption == 6 {
            use predicate_provenance::IndexedSink;
            use super::super::super::arithmetic::IndexedTerm;
            let row = |constant, a, b, len| IndexedAffineRow { constant, len,
                terms: [IndexedTerm { lane: a, coefficient: 1 }, IndexedTerm { lane: b, coefficient: -1 }] };
            let offsets = [1, 0, -1].map(|constant| IndexedAffineRow { constant, len: 1,
                terms: [IndexedTerm { lane: 0, coefficient: 1 }, IndexedTerm::default()] });
            let offset = indexed.indexed_affine(Value(0), &offsets).unwrap();
            let sign = indexed.sign(offset).unwrap();
            let packed = indexed.from_bit(sign).unwrap();
            indexed.nested_difference(packed, &[row(0, 2, 0, 2)]).unwrap();
            assert!(!indexed.0.extension.owners().unwrap().proofs(4).is_empty());
        }
        actual.program = indexed.0;
        let before = predicate_provenance::access_counts();
        let body_entries = RESOURCE_BODY_ENTRIES.with(|n| n.get());
        let mut query = ResourceViewSlots::new(&actual);
        assert_eq!(query_resource_view(&mut query), Err(WorkPreflightError::Unsupported));
        assert!(query.view.is_none());
        assert!(query.view().is_none());
        assert_eq!(predicate_provenance::access_counts(), before);
        assert_eq!(RESOURCE_BODY_ENTRIES.with(|n| n.get()), body_entries);
    }
}

#[test]
fn resource_view_absent_native_metadata_clone_and_drop_have_no_indexed_owners() {
    let before = predicate_provenance::access_counts();
    {
        let original = output();
        let clone = original.program.clone();
        assert!(original.program.extension.absent() && clone.extension.absent());
        original.program.resources().unwrap();
        let mut query = ResourceViewSlots::new(&original);
        query_resource_view(&mut query).unwrap();
        resources_proof_free(query.view.as_ref().unwrap(), &mut ResourceBodySlots::empty()).unwrap();
        let mut scalar = SoftmaxWorkWorkspace::empty();
        scalar.reset(1, input());
        let instruction = Instruction::IndexedAffine { input: Value(0), table: 1, output_lanes: 1 };
        assert!(scalar.push(instruction, input(), false).is_err());
        let mut b = Builder::new(1, input()).unwrap();
        assert!(b.push(instruction, input(), false).is_err());
        assert_eq!(b.indexed_with_pending(Value(255), &[], false, &mut predicate_provenance::Pending::empty()), Err(Error::Phase));
        use predicate_provenance::IndexedSink;
        assert_eq!(b.masked_operand(Value(255), Value(255)), Err(Error::Phase));
        // Ordinary Builder errors stay nonpoisoning.
        let value = b.rescale(Value(0), 1).unwrap();
        assert!(b.finish(value).unwrap().extension.absent());
    }
    assert_eq!(predicate_provenance::access_counts(), before);
    let before = predicate_provenance::access_counts();
    {
        let b = Builder::indexed(1, input()).unwrap();
        let clone = b.0.clone();
        assert!(!clone.extension.absent());
    }
    let after = predicate_provenance::access_counts();
    assert_eq!((after[0] - before[0], after[1] - before[1], after[2] - before[2]), (1, 1, 2));
    assert_eq!(&after[3..], &before[3..]);
    // Legacy row bound is distinct from the indexed input tile bound.
    assert!(Builder::row(MAX_TILE_LANES + 1, input()).is_ok());
    assert!(Builder::indexed(MAX_TILE_LANES + 1, input()).is_err());
}

#[test]
fn resource_view_err_unwind_results_and_original_affines_remain_until_actual_destruction() {
    let cosine = [262_144];
    let sine = [0];
    let mut actual = build_core(&Construction::RotaryHalf { rows: 1, width: 2, cosine_len: 1, sine_len: 1 },
        &FactoryParameters::RotaryHalf { input: input(), cosine: &cosine, sine: &sine }).unwrap();
    rms_resident::watch_resource_output(&actual);
    let addresses = actual.program.public_affines.each_ref().map(|v| v.as_ptr());
    for stage in [ResourceLeafFault::QueryBurn, ResourceLeafFault::QueryInstall, ResourceLeafFault::QueryInstalled] {
        let mut query = ResourceViewSlots::new(&actual);
        assert!(query.view().is_none());
        RESOURCE_LEAF_FAULT.with(|f| f.set(Some(stage)));
        assert!(catch_unwind(AssertUnwindSafe(|| query_resource_view(&mut query))).is_err());
        assert!(query.query.phase == ResourcePhase::Burned);
        assert_eq!(query_resource_view(&mut query), Err(WorkPreflightError::WorkspacePoisoned));
        assert_eq!(query.view.is_some(), stage == ResourceLeafFault::QueryInstalled);
        // Installed success is NOT publication: phase remains Burned on unwind.
        assert!(query.view().is_none());
        assert_eq!(query.result.is_ok(), stage == ResourceLeafFault::QueryInstalled);
        assert_eq!(rms_resident::resource_output_drops(), (0, 0));
    }
    for stage in [ResourceLeafFault::BodyBurn, ResourceLeafFault::BodyInstall, ResourceLeafFault::BodyInstalled] {
        let mut query = ResourceViewSlots::new(&actual);
        query_resource_view(&mut query).unwrap();
        let mut body = ResourceBodySlots::empty();
        RESOURCE_LEAF_FAULT.with(|f| f.set(Some(stage)));
        assert!(catch_unwind(AssertUnwindSafe(|| resources_proof_free(query.view.as_ref().unwrap(), &mut body))).is_err());
        assert!(body.phase == ResourcePhase::Burned);
        assert_eq!(body.resources.is_some(), stage == ResourceLeafFault::BodyInstalled);
        assert_eq!(body.result.is_ok(), stage == ResourceLeafFault::BodyInstalled);
        let retained = body.resources;
        assert_eq!(resources_proof_free(query.view.as_ref().unwrap(), &mut body), Err(Error::Phase));
        assert_eq!(body.resources, retained);
        assert_eq!(rms_resident::resource_output_drops(), (0, 0));
    }
    actual.program.register_count = usize::MAX;
    {
        let mut query = ResourceViewSlots::new(&actual);
        query_resource_view(&mut query).unwrap();
        let mut body = ResourceBodySlots::empty();
        assert_eq!(resources_proof_free(query.view.as_ref().unwrap(), &mut body), Err(Error::Bounds));
        assert_eq!(body.result, Err(Error::Bounds));
        assert!(body.resources.is_none());
        assert_eq!(resources_proof_free(query.view.as_ref().unwrap(), &mut body), Err(Error::Phase));
        assert_eq!(body.result, Err(Error::Bounds));
    } // every view/result retired BEFORE the original output, then its guard
    assert_eq!(actual.program.public_affines.each_ref().map(|v| v.as_ptr()), addresses);
    assert_eq!(rms_resident::resource_output_drops(), (0, 0));
    struct OriginalGuard;
    impl Drop for OriginalGuard {
        fn drop(&mut self) { assert_eq!(rms_resident::resource_output_drops(), (1, 1)); }
    }
    let guard = OriginalGuard;
    drop(actual);
    assert_eq!(rms_resident::resource_output_drops(), (1, 1));
    drop(guard);
    rms_resident::unwatch_resource_output();
}

#[test]
fn resource_view_max_nodes_and_short_graphs_use_identical_resource_rows() {
    for nodes in [3, 6, 192] {
        let mut b = Builder::new(1, input()).unwrap();
        let sign = b.sign(b.input()).unwrap();
        let mut value = b.from_bit(sign).unwrap();
        for _ in 3..nodes { value = b.affine(value, None, Affine { left: 1, right: 0, constant: 0 }).unwrap(); }
        // Exercise the SAME calculation at its node ceiling without inventing
        // a closed producer or issuing any view for this plain Builder result.
        let program = b.finish(value).unwrap();
        let resources = program.resource_calculation().unwrap();
        assert_eq!(resources, program.resources().unwrap());
        assert!(matches!(program.nodes[2].unwrap().instruction, Instruction::FromBit(_)));
        // Sign: two input words + two output words + 34 scratch words. Its
        // peak dominates FromBit's (4+26)*8 and subsequent affine's (4+6)*8.
        assert_eq!(resources.working_bytes, 304);
        assert_eq!(program.work_budget().operations(), (nodes - 1) as u64);
    }
}

#[test]
fn resource_view_negative_exact_controls_include_from_bit_and_table_success_binding() {
    let layout = reconstructed_controls();
    let old = reconstructed_pre_census_controls();
    let delta = 4 * size_of::<usize>() + 4 * size_of::<&'static Program>()
        + 2 * size_of::<&'static predicate_provenance::Extension>()
        + size_of::<&'static ResourceAdd>() + size_of::<Option<Value>>()
        + 2 * size_of::<Value>() + 2 * size_of::<bool>();
    assert_eq!(layout, old + delta);
    assert!(old + 2 * size_of::<usize>() < layout);
    for lanes in [1, 128] {
        let actual = build_core(&Construction::Negative { lanes },
            &FactoryParameters::Negative { input: input() }).unwrap();
        assert_eq!(actual.producer, ResourceProducer::Negative);
        assert!(actual.program.nodes.iter().flatten().any(|n|
            matches!(n.instruction, Instruction::FromBit(_))));
        for bytes in [layout - 1, old, old + 2 * size_of::<usize>()] {
            let before = RESOURCE_QUERY_ENTRIES.with(|n| n.get());
            let mut sponsor = TestSponsor::new(1_198_912, bytes);
            assert!(sponsored_query(&mut sponsor, &actual).is_err());
            assert_eq!(RESOURCE_QUERY_ENTRIES.with(|n| n.get()), before);

            let mut sponsor = TestSponsor::new(1_198_912, layout);
            sponsor.body_controls = bytes;
            let query = sponsored_query(&mut sponsor, &actual).ok().unwrap();
            let before = RESOURCE_BODY_ENTRIES.with(|n| n.get());
            assert!(sponsored_body(&mut sponsor, query.view.as_ref().unwrap()).is_err());
            assert_eq!(RESOURCE_BODY_ENTRIES.with(|n| n.get()), before);
            assert_eq!(sponsor.burned, 256);
        }
        let mut sponsor = TestSponsor::new(1_198_912, layout);
        let query = sponsored_query(&mut sponsor, &actual).ok().unwrap();
        let body = sponsored_body(&mut sponsor, query.view.as_ref().unwrap()).ok().unwrap();
        let resources = body.resources.unwrap();
        assert_eq!(resources, actual.program.resources().unwrap());
        assert_eq!(resources.working_bytes, 304 * lanes);
        assert_eq!((sponsor.work, sponsor.burned), (0, 1_198_912));
    }
}
