//! D29 private clear-root owning region. No inherited/masked import entry.
//! Source labels remain local declarations, not publisher authentication.
//! Canonical records/owner inventory require review before any execution.
//! Logical capacity only: dependency stacks, allocator/RSS/OOM are not qualified.
use super::super::activation::certified_exp_lookup as certified;
use super::super::{
    nexp_product as consuming, nexp_stage_bounds as bounds, role_coordination as custody,
};
use super::*;
use crate::masked_wire_reference::mask as ring_mask;
use crate::masked_wire_reference::{elementwise_product as product, source_lookup as lookup};
use crate::shared_allocation::{OwnedCharged, MAX_ALLOCATION_BYTES};
use crate::{exp_floor, fixed_exp_tables as tables};
use pllm_core::edge_contract::{ConsumerUse, EdgeContractWorkspace};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use zeroize::Zeroize;

const N: usize = bounds::READ_COUNT;
const ROUNDS: usize = 11;
const MAP: [usize; N] = bounds::EDGE_BY_USE;
const DOMAIN: &[u8] = super::REGION_DOMAIN;
const COORD: &[u8] = super::COORDINATE_DOMAIN;
const EQUATION: &[u8] = super::EQUATION;
pub type Input = OwnedCharged<Zeroizing<Vec<i64>>>;
type Slot<C> = custody::RoleSlotFor<C, State<C>>;
type Shared<C> = custody::CoordinationFor<C, State<C>>;
pub type Created<C> = (Mask<C>, Coordinator<C>, [Role<C>; 2]);
type ArithmeticIssued<C> =
    Result<(arithmetic::OutputMasksFor<C>, [arithmetic::PartyFor<C>; 2]), Error>;
type LookupIssued<C> = Result<(lookup::OutputMasksFor<C>, [lookup::PartyFor<C>; 2]), Error>;

#[derive(Clone, Copy)]
pub struct Spec {
    // X,X40,A,H,B,W,V,C,J0,J1,U0,U1,P,E, in this EXACT order.
    pub(super) edges: [EdgeDescriptor; 14],
    pub(super) operation: [u8; 32],
    pub(super) next: [u8; 32],
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum BuildError {
    Native(Error),
    Floor(tables::Error),
    Quote(exp_floor::Error),
    Certified(certified::Error),
}
impl From<certified::Error> for BuildError {
    fn from(e: certified::Error) -> Self {
        match e {
            certified::Error::Native(native) => Self::Native(native),
            other => Self::Certified(other),
        }
    }
}
impl From<Error> for BuildError {
    fn from(e: Error) -> Self {
        Self::Native(e)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Quote {
    pub(super) incoming_owner_bytes: usize,
    pub(super) child_bytes: [usize; 8],
    pub(super) table_bytes: usize,
    pub(super) state_bytes: usize,
    pub(super) slot_bytes: usize,
    pub(super) control_bytes: usize,
    pub(super) total: usize,
    pub(super) peer_bytes: usize,
    pub(super) work: WorkQuote,
}

/// Separate deterministic categories, NOT CPU instructions/entropy/privacy.
/// Includes both lookup invocations, all floor entries and all native gates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkQuote {
    /// One checked declaration and the standard public query/issuance route.
    /// Extra caller-initiated queries/callback work are outside this invocation.
    pub public_declaration_descriptor_checks: usize,
    pub public_resource_query_passes: usize,
    pub public_resource_descriptor_checks: usize,
    pub input_validation_lanes: usize,
    pub input_retirement_words: usize,
    pub root_validation_lanes: usize,
    pub root_initialization_words: usize,
    pub root_encoding_word_writes: usize,
    pub root_mask_generation_words: usize,
    pub root_retirement_words: usize,
    pub ordered_use_checks: usize,
    pub public_table_range_checks: usize,
    pub stage_bound_descriptor_checks: usize,
    pub stage_bound_interval_writes: usize,
    pub stage_bound_metadata_queries: usize,
    pub stage_bound_table_metadata_passes: usize,
    pub buffer_initialization_words: usize,
    pub buffer_erasure_words: usize,
    pub projection_lane_visits: usize,
    pub floor_taylor: usize,
    pub floor_squarings: usize,
    pub floor_multiply: usize,
    pub floor_divide: usize,
    pub floor_remainder: usize,
    pub floor_add: usize,
    pub floor_shift: usize,
    pub paired_native_issue_aes: usize,
    pub native_evaluate_aes_per_role: usize,
    pub random_byte_requests: usize,
}

fn sum(parts: &[usize]) -> Result<usize, Error> {
    parts
        .iter()
        .try_fold(0usize, |n, b| n.checked_add(*b).ok_or(Error::Resource))
}

pub(super) fn preflight<C: AllocationContext>(
    dealer: &LocalDealerFor<C>,
    s: &Spec,
) -> Result<usize, Error> {
    let expect = [
        (50, 12, Interpretation::Signed),
        (40, 12, Interpretation::Signed),
        (40, 12, Interpretation::Signed),
        (1, 0, Interpretation::Predicate),
        (1, 0, Interpretation::Predicate),
        (16, 12, Interpretation::Unsigned),
        (16, 12, Interpretation::Unsigned),
        (16, 12, Interpretation::Unsigned),
        (8, 0, Interpretation::Unsigned),
        (8, 0, Interpretation::Unsigned),
        (50, 12, Interpretation::Signed),
        (50, 12, Interpretation::Signed),
        (50, 24, Interpretation::Signed),
        (50, 12, Interpretation::Signed),
    ];
    let first = s.edges[0];
    let lanes = first.shape().elements();
    if !(1..=256).contains(&lanes) {
        return Err(Error::Resource);
    }
    // Fourteen independent bindings, including the consuming product's P.
    // Refuse exhaustion before floor/hash/factory/entropy or partial issuance.
    dealer.counter.checked_add(14).ok_or(Error::Resource)?;
    first
        .identity()
        .first_tensor_index()
        .checked_add(lanes as u64)
        .ok_or(Error::Resource)?;
    for (i, edge) in s.edges.iter().enumerate() {
        let numeric = edge.numeric();
        let (bits, scale, interpretation) = expect[i];
        if numeric.ring_bits() != bits
            || numeric.fractional_bits() != scale
            || numeric.interpretation() != interpretation
            || edge.shape() != first.shape()
            || edge.identity().source() != dealer.source
            || edge.identity().plan() != dealer.plan
            || edge.identity().first_tensor_index() != first.identity().first_tensor_index()
            || edge.weight_visibility() == Some(WeightVisibility::Public)
            || edge.representation()
                != if i == 3 || i == 4 {
                    Representation::CommonXorMasked
                } else {
                    Representation::CommonAdditiveMasked
                }
        {
            return Err(Error::Domain);
        }
        if s.edges[..i]
            .iter()
            .any(|old| old.identity().wire() == edge.identity().wire())
        {
            return Err(Error::Context);
        }
    }
    Ok(lanes)
}

fn arithmetic_spec(lanes: usize, bits: u8, gate: arithmetic::Gate) -> arithmetic::Spec {
    arithmetic::Spec { bits, lanes, gate }
}
fn lookup_spec(lanes: usize) -> lookup::Spec {
    lookup::Spec {
        index_bits: 8,
        output_bits: 50,
        lanes,
        boundary: selection::Boundary::Reconstructed,
    }
}

// Quote slots exist before resources is called; caller must already hold their
// real original-counter reservation. No numeric limit acts as Budget authority.
pub(super) fn bootstrap_bytes<C: AllocationContext>() -> usize {
    size_of::<Spec>()
        + size_of::<Quote>()
        + size_of::<Result<Quote, BuildError>>()
        + size_of::<[usize; 16]>()
        + size_of::<LeaseFor<C>>()
        + size_of::<ConstructionSlots<'_, C>>()
        + bounds::validation_slots_bytes()
}

#[allow(dead_code)] // Concrete source-level old/new/call/return inventory.
struct ConstructionSlots<'a, C: AllocationContext> {
    public_calls: super::clear_root::CallSlots<'a, C>,
    spec: [Spec; 2],
    quote: Quote,
    quote_result: Result<Quote, BuildError>,
    proof_quote: tables::Quote,
    certified_quote: certified::Quote,
    certified_quote_result: Result<certified::Quote, certified::Error>,
    proof_result: Result<tables::Verified, tables::Error>,
    // Caller-held proof object/argument/result overlap is additional to the
    // certified helper's own quoted callee phases; no retirement credit here.
    proof_workspace: exp_floor::Workspace,
    proof_argument: &'a mut exp_floor::Workspace,
    preparation_result: Result<(tables::Verified, [[u8; 32]; 2]), certified::Error>,
    verified: tables::Verified,
    verified_argument: &'a tables::Verified,
    declarations: [Declaration; 2],
    coordinate_declarations: [CoordinateDeclaration; 2],
    provenances: [CoordinateProvenance; 6],
    root_range: [BoundRange<LocalAssumption>; 2],
    root_range_result: Result<BoundRange<LocalAssumption>, Error>,
    intervals: [Option<Interval>; 14],
    bound_input: &'a EdgeDescriptor,
    bound_values: &'a [i64],
    bound_reads: [&'a EdgeDescriptor; N],
    bound_returned: Result<bounds::BindingIntervals, Error>,
    preflight_types: [(u8, u8, Interpretation); 14],
    original_input: &'a Input,
    original_values: &'a Zeroizing<Vec<i64>>,
    binding: [Binding; 14],
    binding_results: [Result<Binding, Error>; 14],
    uses: [ConsumerUse; N],
    ledgers: [internal_use::Ledger<N>; 3],
    hash: [Sha256; 4],
    hash_results: HashSlots,
    coordinate_parts: [&'a [u8]; 11],
    region_parts: [&'a [u8]; 8],
    provenance_parts: [&'a [u8]; 6],
    edge_workspace: EdgeContractWorkspace,
    arithmetic_costs: [arithmetic::Resources; 3],
    select_cost: selection::Resources,
    lookup_cost: lookup::Resources,
    product_cost: product::Resources,
    arithmetic_cost_result: Result<arithmetic::Resources, Error>,
    select_cost_result: Result<selection::Resources, Error>,
    lookup_cost_result: Result<lookup::Resources, Error>,
    product_cost_result: Result<product::Resources, Error>,
    root: Result<RootInputFor<C>, Error>,
    root_return: Result<EncodedRootFor<C>, Error>,
    arithmetic: [ArithmeticIssued<C>; 3],
    select: Result<(selection::OutputMasksFor<C>, [selection::PartyFor<C>; 2]), Error>,
    lookup: [LookupIssued<C>; 2],
    product: Result<(product::MasksFor<C>, [product::PartyFor<C>; 2]), Error>,
    product_declaration: Option<consuming::DeclarationFor<C>>,
    product_declaration_result: Result<consuming::DeclarationFor<C>, Error>,
    lookup_masks: [consuming::LookupMaskFor<C>; 2],
    consuming_product: Result<(consuming::ProductMaskFor<C>, [consuming::PartyFor<C>; 2]), Error>,
    state: [Option<State<C>>; 2],
    state_result: Result<[State<C>; 2], Error>,
    slots: [Arc<Slot<C>>; 2],
    shared: Option<Arc<Shared<C>>>,
    return_origin: Created<C>,
    returned: Result<Created<C>, BuildError>,
    outer_returned: Result<Created<C>, BuildError>,
    vector: Vec<u64>,
    boxed: Box<[u64]>,
    buffer_result: Result<Zeroizing<Box<[u64]>>, Error>,
    source: Option<Arc<Tables<C>>>,
    leases: [Option<LeaseFor<C>>; 14],
    contexts: [arithmetic::Context; 3],
    select_context: selection::Context,
    lookup_context: [lookup::Context; 2],
    product_context: product::Context,
    nonce: Zeroizing<[u8; 32]>,
    word: Zeroizing<[u8; 8]>,
    control_guard: custody::OperationFor<'a, C, State<C>>,
}

#[allow(dead_code)]
struct HashSlots {
    tables: [[u8; 32]; 2],
    declaration: [u8; 32],
    coordinates: [[u8; 32]; 2],
    ledger: [[u8; 32]; 3],
    region: [u8; 32],
    stage: [u8; 32],
    input_bindings: [[u8; 32]; 2],
    output_binding: [u8; 32],
}

#[allow(dead_code)]
struct RuntimeSlots<C: AllocationContext> {
    beginning: Result<Body<C>, Error>,
    advancing: Result<Option<Body<C>>, Error>,
    outgoing: Result<Frame<C>, Error>,
    step: Result<Step<C>, Error>,
    handoff: Result<Output<C>, Error>,
    native_arithmetic: Result<arithmetic::StepFor<C>, Error>,
    native_select: Result<selection::MaskedOutputFor<C>, Error>,
    native_lookup: Result<lookup::StepFor<C>, Error>,
    native_product: Result<product::OutputFor<C>, Error>,
    product_values: [consuming::LookupValueFor<C>; 2],
    product_output: Result<consuming::ProductValueFor<C>, Error>,
    output_origin: CommonMaskedFor<C, Ring>,
    descriptor: Result<EdgeDescriptor, Error>,
    bytes: Result<&'static [u8], Error>,
    fixed: Result<(), Error>,
    word: u64,
    slot: Arc<Slot<C>>,
    shared: Arc<Shared<C>>,
}

pub(super) fn resources<C: AllocationContext>(
    dealer: &LocalDealerFor<C>,
    s: &Spec,
    input: &Input,
) -> Result<Quote, BuildError> {
    let n = preflight(dealer, s)?;
    if !std::ptr::eq(input.budget(), dealer.budget.budget()) {
        return Err(Error::Resource.into());
    }
    let incoming_minimum = sum(&[
        size_of::<Input>(),
        input
            .payload()
            .capacity()
            .checked_mul(size_of::<i64>())
            .ok_or(Error::Resource)?,
    ])?;
    if input.charged_bytes() < incoming_minimum {
        return Err(Error::Resource.into());
    }
    let root = dealer.root_reservation_bytes(s.edges[0], &BoundRange::new(0, 1i128 << 38)?)?;
    let d =
        arithmetic::resources_for_context::<C>(arithmetic_spec(n, 40, arithmetic::Gate::Drelu))?;
    let v = selection::resources_reconstructed_for_context::<C>(selection::Spec {
        bits: 16,
        lanes: n,
    })?;
    let t = arithmetic::resources_for_context::<C>(arithmetic_spec(
        n,
        16,
        arithmetic::Gate::Tr { shift: 8 },
    ))?;
    let l = lookup::resources_for_context::<C>(lookup_spec(n))?;
    let p = product::resources_for::<C>(n)?;
    let product_bytes = sum(&[p.combined_bytes, consuming::wrapper_bytes::<C>()?])?;
    let g = arithmetic::resources_for_context::<C>(arithmetic_spec(
        n,
        50,
        arithmetic::Gate::GapArs { shift: 12 },
    ))?;
    let pinned = certified::quote::<C>(n)?;
    let proof = &pinned.floors;
    let table_bytes = sum(&[
        pinned.preparation_peak_bytes,
        proof.selected_literal_bytes,
        custody::arc_bytes::<Tables<C>>()?,
        size_of::<Result<tables::Quote, exp_floor::Error>>(),
    ])?;
    let state_bytes = sum(&[
        5 * size_of::<u64>() * n,
        11 * size_of::<Frame<C>>(),
        11 * size_of::<Step<C>>(),
        size_of::<CommonMaskedFor<C, Ring>>(),
        size_of::<Role<C>>(),
        size_of::<Output<C>>(),
        2 * size_of::<consuming::LookupValueFor<C>>(),
    ])?;
    let slot_bytes = custody::arc_bytes::<Slot<C>>()?;
    let control_bytes = sum(&[
        bootstrap_bytes::<C>(),
        size_of::<ConstructionSlots<'_, C>>(),
        custody::arc_bytes::<Shared<C>>()?,
        3 * size_of::<u64>() * n,
        size_of::<[Zeroizing<Box<[u64]>>; 3]>(),
        size_of::<Coordinator<C>>(),
        size_of::<[Role<C>; 2]>(),
        size_of::<Result<Created<C>, BuildError>>(),
        size_of::<custody::CallbackPollFor<'_, C, State<C>>>(),
        size_of::<std::sync::MutexGuard<'_, State<C>>>(),
        2 * size_of::<RuntimeSlots<C>>(),
        pinned.native_query_phase_bytes,
        2 * pinned.issuance_adapter_bytes,
        super::nexp_declaration_resources(n)?.metadata_bytes,
        2 * size_of::<consuming::LookupMaskFor<C>>(),
        size_of::<Mask<C>>(),
        super::super::activation::arithmetic_resource_calculation_bytes()?,
        super::super::activation::selection_resource_calculation_bytes()?,
        super::super::activation::lookup_resource_calculation_bytes()?,
        2 * C::TRANSIENT_LINK_BYTES,
    ])?;
    let child_bytes = [
        root,
        d.combined_reservation_bytes,
        v.combined_reservation_bytes,
        t.combined_reservation_bytes,
        l.combined_reservation_bytes,
        l.combined_reservation_bytes,
        product_bytes,
        g.combined_reservation_bytes,
    ];
    let total = sum(&[
        sum(&child_bytes)?,
        table_bytes,
        2 * state_bytes,
        2 * slot_bytes,
        control_bytes,
        mask_charge_for::<C>(),
    ])?;
    // Actual ORIGINAL charge, including excess; never substitute a minimum
    // source-capacity estimate for already-retained authority.
    let incoming_owner_bytes = input.charged_bytes();
    if sum(&[total, incoming_owner_bytes])? > MAX_ALLOCATION_BYTES {
        return Err(Error::Resource.into());
    }
    let peer_bytes = sum(&[
        d.total_peer_frame_bytes,
        v.total_peer_frame_bytes,
        t.total_peer_frame_bytes,
        2 * l.total_peer_frame_bytes,
        p.total_peer_bytes,
        g.total_peer_frame_bytes,
    ])?;
    if peer_bytes != 1688 + 208 * n {
        return Err(Error::Context.into());
    }
    let work = WorkQuote {
        public_declaration_descriptor_checks: 14,
        public_resource_query_passes: 2,
        public_resource_descriptor_checks: 2 * 14,
        // Original whole-root validation, then the checked stage preflight.
        input_validation_lanes: 2 * n,
        // Pinned zeroize Vec visits initialized len, clears, then full capacity.
        input_retirement_words: input
            .payload()
            .capacity()
            .checked_add(input.payload().len())
            .ok_or(Error::Resource)?,
        root_validation_lanes: n,
        root_initialization_words: 3 * n,
        root_encoding_word_writes: 2 * n,
        root_mask_generation_words: n,
        root_retirement_words: 6 * n,
        ordered_use_checks: 3 * N,
        public_table_range_checks: bounds::TABLE_WORDS,
        stage_bound_descriptor_checks: bounds::READ_COUNT,
        stage_bound_interval_writes: bounds::READ_COUNT + 1,
        // certified_ranges: two table bounds, two product operands, product and
        // shifted-result expectations, four coordinate/Gap numeric queries;
        // then one per descriptor and one per interval assignment. Fixed scalar
        // checks inside each query are not a CPU-instruction/RSS claim.
        stage_bound_metadata_queries: 2 + 2 + 1 + 1 + 4 + 2 * bounds::READ_COUNT,
        stage_bound_table_metadata_passes: 2,
        // Dealer r40/r16/r8 and five private buffers for EACH role.
        buffer_initialization_words: (3 + 2 * 5) * n,
        // Dealer Drop; each role early r40 erase, cancellation, then Drop.
        buffer_erasure_words: (3 + 2 * (1 + 5 + 5)) * n,
        // Three dealer projections; each role X40,A40,B,W16,C16,J0.
        projection_lane_visits: (3 + 2 * 6) * n,
        floor_taylor: proof.proof.taylor_steps,
        floor_squarings: proof.proof.squarings,
        floor_multiply: proof.proof.checked_multiplications,
        floor_divide: proof.proof.checked_divisions,
        floor_remainder: proof.proof.checked_remainders,
        floor_add: proof.proof.checked_additions,
        floor_shift: proof.proof.checked_shifts,
        paired_native_issue_aes: sum(&[
            d.paired_issue_aes_blocks,
            v.paired_issue_aes_blocks,
            t.paired_issue_aes_blocks,
            2 * l.paired_issue_aes_blocks,
            g.paired_issue_aes_blocks,
        ])?,
        native_evaluate_aes_per_role: sum(&[
            d.evaluation_aes_blocks_per_party,
            v.evaluation_aes_blocks_per_party,
            t.evaluation_aes_blocks_per_party,
            2 * l.evaluation_aes_blocks_per_party,
            g.evaluation_aes_blocks_per_party,
        ])?,
        random_byte_requests: sum(&[
            8 * n,
            32,
            d.random_bytes,
            v.random_bytes,
            t.random_bytes,
            2 * l.random_bytes,
            p.random_bytes,
            g.random_bytes,
        ])?,
    };
    Ok(Quote {
        incoming_owner_bytes,
        child_bytes,
        table_bytes,
        state_bytes,
        slot_bytes,
        control_bytes,
        total,
        peer_bytes,
        work,
    })
}

struct Tables<C: AllocationContext> {
    checked: tables::Verified,
    hashes: [[u8; 32]; 2],
    declaration: Declaration,
    // Actual immutable original input and its ORIGINAL charged owner, not a copy.
    input: Input,
    _lease: LeaseFor<C>,
}

fn usage(s: &Spec) -> [ConsumerUse; N] {
    std::array::from_fn(|i| {
        ConsumerUse::new(
            s.edges[MAP[i]],
            if i == 17 { s.next } else { s.operation },
            u16::from(i == 7 || i == 15),
            i as u32,
        )
    })
}

// Explicit canonical preimage protocol. See independent record vectors in tests.
fn spec_digest(
    s: &Spec,
    hashes: [[u8; 32]; 2],
    session: [u8; 32],
    uses: &[ConsumerUse; N],
) -> [u8; 32] {
    let declaration = declaration_record(s, hashes);
    let mut h = Sha256::new();
    let mut part = |p: &[u8]| {
        h.update((p.len() as u64).to_le_bytes());
        h.update(p);
    };
    part(DOMAIN);
    part(b"declaration.v1");
    part(EQUATION);
    part(tables::PAPER_SHA256.as_bytes());
    part(tables::SOURCE_COMMIT.as_bytes());
    let source = declaration.source;
    part(&source.checkpoint());
    part(&source.numeric_profile());
    part(&source.implementation());
    part(&s.edges[0].identity().plan());
    part(&session);
    part(&declaration.operation);
    part(&declaration.next);
    part(&0i128.to_le_bytes());
    part(&(1i128 << 38).to_le_bytes());
    part(&0i128.to_le_bytes());
    part(&4097i128.to_le_bytes());
    part(&4096u64.to_le_bytes());
    part(&declaration.table_hashes[0]);
    part(&declaration.table_hashes[1]);
    part(&[12, 4, 50, 12, 16, 8, 0, 50, 24]);
    part(&(s.edges[0].shape().elements() as u64).to_le_bytes());
    part(&s.edges[0].identity().first_tensor_index().to_le_bytes());
    for edge in &s.edges {
        part(&edge.digest());
    }
    for use_ in uses {
        part(&use_.digest());
    }
    h.finalize().into()
}

fn declaration_record(s: &Spec, hashes: [[u8; 32]; 2]) -> Declaration {
    Declaration {
        source: s.edges[0].identity().source(),
        input: s.edges[0],
        output: s.edges[13],
        operation: s.operation,
        next: s.next,
        table_hashes: hashes,
        coordinates: std::array::from_fn(|index| CoordinateDeclaration {
            parent: s.edges[7],
            coordinate: s.edges[8 + index],
            table_hash: hashes[index],
            high: index == 1,
            first: s.edges[0].identity().first_tensor_index(),
            lanes: s.edges[0].shape().elements() as u64,
        }),
        uses: usage(s),
    }
}

fn coordinate(s: &Spec, hashes: [[u8; 32]; 2], index: usize) -> [u8; 32] {
    let record = declaration_record(s, hashes);
    let coordinate = &record.coordinates[index];
    digest(&[
        COORD,
        b"declaration.v1",
        if !coordinate.high {
            b"low8.v1"
        } else {
            b"high8-tr16-8.v1"
        },
        &coordinate.parent.digest(),
        &coordinate.coordinate.digest(),
        &coordinate.table_hash,
        &coordinate.first.to_le_bytes(),
        &coordinate.lanes.to_le_bytes(),
        &[8, 0],
        &0u16.to_le_bytes(),
        &256u16.to_le_bytes(),
    ])
}
fn provenance(declaration: [u8; 32], bindings: &[Binding; 14], index: usize, role: u8) -> [u8; 32] {
    let mut record = CoordinateProvenance {
        declaration,
        parent: bindings[7],
        coordinate: bindings[8 + index],
        role,
        digest: [0; 32],
    };
    record.digest = digest(&[
        COORD,
        b"runtime.v1",
        &record.declaration,
        &record.parent.digest(),
        &record.coordinate.digest(),
        &[record.role],
    ]);
    record.digest
}
fn stage(region: [u8; 32], ordinal: u16) -> [u8; 32] {
    digest(&[&region, b"stage.v1", &ordinal.to_le_bytes()])
}
fn ledger_digest(
    coordinates: [[u8; 32]; 2],
    bindings: &[Binding; 14],
    ledger: &internal_use::Ledger<N>,
) -> [u8; 32] {
    digest(&[
        &ledger.digest_with_domains(b"pllm.sigma.nexp-use.v1", b"pllm.sigma.nexp-ledger.v1"),
        &provenance(coordinates[0], bindings, 0, ledger.role),
        &provenance(coordinates[1], bindings, 1, ledger.role),
    ])
}
fn region_digest(
    declaration: [u8; 32],
    bindings: &[Binding; 14],
    ledgers: [[u8; 32]; 3],
    nonce: &[u8; 32],
) -> [u8; 32] {
    digest(&[
        DOMAIN,
        &declaration,
        &bindings[0].digest(),
        &bindings[13].digest(),
        &ledgers[0],
        &ledgers[1],
        &ledgers[2],
        nonce,
    ])
}
fn box_zeros<C: AllocationContext>(
    n: usize,
    guard: &LeaseFor<C>,
) -> Result<Zeroizing<Box<[u64]>>, Error> {
    let mut values = crate::shared_allocation::allocate(n, guard.budget.budget())?;
    values.resize(n, 0);
    Ok(Zeroizing::new(values.into_boxed_slice()))
}

/// Consumes genuine ORIGINAL charged immutable clear storage. Range validation
/// checks EVERY original integer before proof/hash/secret allocation or entropy.
/// Upstream input creation is not retroactively admitted. Working is an exact
/// already-live original-counter lease, never a numeric limit/adoption token.
pub(super) fn construct<C: AllocationContext>(
    dealer: &mut LocalDealerFor<C>,
    s: Spec,
    input: Input,
    working: LeaseFor<C>,
) -> Result<Created<C>, BuildError> {
    construct_with(dealer, s, input, working, |bytes| {
        getrandom::fill(bytes).map_err(|_| Error::Randomness)
    })
}

#[allow(clippy::arc_with_non_send_sync)]
fn construct_with<C: AllocationContext>(
    dealer: &mut LocalDealerFor<C>,
    s: Spec,
    input: Input,
    mut working: LeaseFor<C>,
    mut fill: impl FnMut(&mut [u8]) -> Result<(), Error>,
) -> Result<Created<C>, BuildError> {
    if working.bytes < bootstrap_bytes::<C>()
        || !std::ptr::eq(input.budget(), dealer.budget.budget())
        || !std::ptr::eq(working.budget.budget(), input.budget())
    {
        return Err(Error::Resource.into());
    }
    let q = resources(dealer, &s, &input)?;
    if working.bytes != q.total || input.charged_bytes() < q.incoming_owner_bytes {
        return Err(Error::Resource.into());
    }
    super::validate_nexp_clear_root(&s.edges[0], input.payload())?;
    let n = s.edges[0].shape().elements();
    let (checked, hashes) = {
        let mut scratch = exp_floor::Workspace::default();
        certified::prepare(&mut scratch)?
    };
    // Plain checked metadata from the ORIGINAL clear input and actual verified
    // static rows, before any root/private material or entropy. No caller range
    // assertion/certificate is accepted, and no new authority is minted here.
    let intervals = bounds::preflight(
        &s.edges[0],
        input.payload(),
        &checked,
        MAP.map(|i| &s.edges[i]),
    )?;
    let uses = usage(&s);
    let declaration = spec_digest(&s, hashes, dealer.session, &uses);
    let table_lease = working.split(q.table_bytes)?;
    dealer.budget.budget().allocation()?;
    let tables = Arc::new(Tables {
        checked,
        hashes,
        declaration: declaration_record(&s, hashes),
        input,
        _lease: table_lease,
    });
    let mut child = [None, None, None, None, None, None, None, None];
    for (slot, bytes) in child.iter_mut().zip(q.child_bytes) {
        *slot = Some(working.split(bytes)?);
    }
    let left_lease = working.split(q.state_bytes)?;
    let right_lease = working.split(q.state_bytes)?;
    let left_slot = working.split(q.slot_bytes)?;
    let right_slot = working.split(q.slot_bytes)?;
    let mask_lease = working.split(mask_charge_for::<C>())?;
    let prepared = dealer.prepare_root_reserved_with(
        s.edges[0],
        BoundRange::new(0, 1i128 << 38)?,
        s.operation,
        child[0].take().ok_or(Error::Consumed)?,
        &mut fill,
    )?;
    let mut bindings = [prepared.dealer.binding; 14];
    for i in 1..12 {
        bindings[i] = dealer.binding(s.edges[i], s.operation, intervals[i])?;
    }
    // Real reservation/declaration, before region hashing. P is NOT minted by
    // parallel binding logic; consuming owner validates both genuine U bindings.
    let product_declaration = consuming::declare_reserved(
        dealer,
        s.operation,
        [bindings[10], bindings[11]],
        s.edges[12],
        s.operation,
        child[6].take().ok_or(Error::Consumed)?,
    )?;
    bindings[12] = product_declaration.output_binding();
    bindings[13] = dealer.binding(s.edges[13], s.next, intervals[13])?;
    let c0 = coordinate(&s, hashes, 0);
    let c1 = coordinate(&s, hashes, 1);
    let make_ledger = |role| internal_use::Ledger {
        uses,
        bindings: std::array::from_fn(|i| bindings[MAP[i]]),
        role,
        next: 0,
    };
    let [mut dealer_ledger, left_ledger, right_ledger] =
        [make_ledger(2), make_ledger(0), make_ledger(1)];
    // Enter with the ORIGINAL prepared root binding, before encoding reads the
    // protected clear input. Root preparation/draws already occurred as before;
    // neither declaration nor this ledger burn issues new material or entropy.
    dealer_ledger.burn(0, prepared.dealer.binding, 2)?;
    #[cfg(test)]
    tests::observe_entry_encode(&dealer_ledger, prepared.dealer.binding);
    let (original, [root0, root1]) = prepared.encode(tables.input.payload())?;
    let ledger_hashes = [
        ledger_digest([c0, c1], &bindings, &left_ledger),
        ledger_digest([c0, c1], &bindings, &right_ledger),
        ledger_digest([c0, c1], &bindings, &dealer_ledger),
    ];
    let mut nonce = Zeroizing::new([0u8; 32]);
    fill(&mut *nonce)?;
    let region = region_digest(declaration, &bindings, ledger_hashes, &nonce);
    let mut r40 = box_zeros(n, &working)?;
    let mut r16 = box_zeros(n, &working)?;
    let mut r8 = box_zeros(n, &working)?;
    dealer_ledger.burn(1, bindings[0], 2)?;
    for i in 0..n {
        r40[i] = original.backing.values()[i] & ring_mask(40);
    }
    dealer_ledger.burn(2, bindings[1], 2)?; // subtracting K does not change the mask
    let arithmetic_context = |ordinal, a: usize, b: usize| arithmetic::Context {
        plan: dealer.plan,
        session: dealer.session,
        operation: stage(region, ordinal),
        input_wire: bindings[a].digest(),
        output_wire: bindings[b].digest(),
        first_tensor_index: s.edges[0].identity().first_tensor_index(),
    };
    dealer_ledger.burn(3, bindings[2], 2)?;
    let (h, [d0, d1]) = arithmetic::issue_reserved_with(
        arithmetic_spec(n, 40, arithmetic::Gate::Drelu),
        arithmetic_context(1, 2, 3),
        &r40,
        child[1].take().ok_or(Error::Consumed)?,
        &mut fill,
    )?;
    dealer_ledger.burn(4, bindings[3], 2)?; // public XOR 1 preserves the predicate mask
    dealer_ledger.burn(5, bindings[1], 2)?;
    for i in 0..n {
        r16[i] = original.backing.values()[i] & 65535;
    }
    dealer_ledger.burn(6, bindings[4], 2)?;
    dealer_ledger.burn(7, bindings[5], 2)?;
    let (v, [v0, v1]) = selection::issue_reserved_with(
        selection::Spec { bits: 16, lanes: n },
        selection::Context {
            plan: dealer.plan,
            session: dealer.session,
            operation: stage(region, 2),
            predicate_wire: bindings[4].digest(),
            input_wire: bindings[5].digest(),
            output_wire: bindings[6].digest(),
            first_tensor_index: s.edges[0].identity().first_tensor_index(),
        },
        h.values(),
        &r16,
        selection::Boundary::Reconstructed,
        child[2].take().ok_or(Error::Consumed)?,
        &mut fill,
    )?;
    dealer_ledger.burn(8, bindings[6], 2)?; // +65535 leaves arithmetic mask unchanged
    dealer_ledger.burn(9, bindings[7], 2)?;
    for i in 0..n {
        r8[i] = v.values()[i] & 255;
    }
    dealer_ledger.burn(10, bindings[7], 2)?;
    let (j1, [t0, t1]) = arithmetic::issue_reserved_with(
        arithmetic_spec(n, 16, arithmetic::Gate::Tr { shift: 8 }),
        arithmetic_context(3, 7, 9),
        v.values(),
        child[3].take().ok_or(Error::Consumed)?,
        &mut fill,
    )?;
    dealer_ledger.burn(11, bindings[9], 2)?;
    let lookup_context = |index: usize| lookup::Context {
        plan: dealer.plan,
        session: dealer.session,
        operation: stage(region, if index == 0 { 5 } else { 7 }),
        input_wire: bindings[8 + index].digest(),
        output_wire: bindings[10 + index].digest(),
        arithmetic_wire: digest(&[&region, b"lookup-arithmetic.v1", &[index as u8]]),
        correction_wire: digest(&[&region, b"lookup-correction.v1", &[index as u8]]),
        select_wire: digest(&[&region, b"lookup-select.v1", &[index as u8]]),
        numeric_contract: digest(&[
            COORD,
            &declaration,
            if index == 0 { &c0 } else { &c1 },
            &bindings[7].digest(),
            &bindings[8 + index].digest(),
        ]),
        first_tensor_index: s.edges[0].identity().first_tensor_index(),
    };
    dealer_ledger.burn(12, bindings[8], 2)?;
    let (u0, [l00, l01]) = certified::issue(
        &tables.checked,
        tables::Table::LowByte,
        lookup_context(0),
        &r8,
        child[4].take().ok_or(Error::Consumed)?,
        &mut fill,
    )?;
    dealer_ledger.burn(13, bindings[9], 2)?;
    let (u1, [l10, l11]) = certified::issue(
        &tables.checked,
        tables::Table::HighByte,
        lookup_context(1),
        j1.values(),
        child[5].take().ok_or(Error::Consumed)?,
        &mut fill,
    )?;
    dealer_ledger.burn(14, bindings[10], 2)?;
    dealer_ledger.burn(15, bindings[11], 2)?;
    let inputs = [
        consuming::LookupMaskFor {
            native: u0,
            binding: bindings[10],
            lease: working.split(size_of::<consuming::LookupMaskFor<C>>())?,
        },
        consuming::LookupMaskFor {
            native: u1,
            binding: bindings[11],
            lease: working.split(size_of::<consuming::LookupMaskFor<C>>())?,
        },
    ];
    let (p, [p0, p1]) =
        product_declaration.issue_with(region, stage(region, 9), inputs, &mut fill)?;
    dealer_ledger.burn(16, bindings[12], 2)?;
    let (e, [g0, g1]) = arithmetic::issue_reserved_with(
        arithmetic_spec(n, 50, arithmetic::Gate::GapArs { shift: 12 }),
        arithmetic_context(10, 12, 13),
        p.native.values(),
        child[7].take().ok_or(Error::Consumed)?,
        &mut fill,
    )?;
    dealer_ledger.burn(17, bindings[13], 2)?;
    let make_state = |root,
                      ledger,
                      role,
                      drelu,
                      select,
                      tr,
                      low,
                      high,
                      product,
                      gap,
                      lease: LeaseFor<C>|
     -> Result<State<C>, Error> {
        let work = [
            box_zeros(n, &lease)?,
            box_zeros(n, &lease)?,
            box_zeros(n, &lease)?,
            box_zeros(n, &lease)?,
            box_zeros(n, &lease)?,
        ];
        Ok(State {
            root: Some(root),
            bindings,
            ledger,
            role,
            phase: 0,
            drelu: Some(drelu),
            select: Some(select),
            tr: Some(tr),
            low: Some(low),
            high: Some(high),
            product: Some(product),
            gap: Some(gap),
            work,
            h: None,
            v: None,
            j1: None,
            u0: None,
            u1: None,
            p: None,
            output: None,
            tables: Arc::clone(&tables),
            lease,
            _local: PhantomData,
        })
    };
    let left = make_state(
        root0,
        left_ledger,
        0,
        d0,
        v0,
        t0,
        l00,
        l10,
        p0,
        g0,
        left_lease,
    )?;
    let right = make_state(
        root1,
        right_ledger,
        1,
        d1,
        v1,
        t1,
        l01,
        l11,
        p1,
        g1,
        right_lease,
    )?;
    dealer.budget.budget().allocation()?;
    let left = Arc::new(custody::RoleSlotFor {
        state: Mutex::new(left),
        _lease: left_slot,
    });
    dealer.budget.budget().allocation()?;
    let right = Arc::new(custody::RoleSlotFor {
        state: Mutex::new(right),
        _lease: right_slot,
    });
    dealer.budget.budget().allocation()?;
    let shared = Arc::new(custody::CoordinationFor {
        control: Mutex::new(custody::SplitControl::default()),
        slots: [Arc::clone(&left), Arc::clone(&right)],
        _original_run: None,
        _additional: working,
    });
    Ok((
        Mask {
            native: DealerMaskFor {
                backing: MaskBackingFor::Arithmetic(e),
                binding: bindings[13],
                _kind: PhantomData,
                _lease: mask_lease,
            },
            shared: Arc::clone(&shared),
        },
        Coordinator {
            shared: Arc::clone(&shared),
        },
        [
            Role {
                slot: left,
                shared: Arc::clone(&shared),
                role: 0,
                handed_off: false,
            },
            Role {
                slot: right,
                shared,
                role: 1,
                handed_off: false,
            },
        ],
    ))
}

struct State<C: AllocationContext> {
    root: Option<CommonMaskedFor<C, Ring>>,
    bindings: [Binding; 14],
    ledger: internal_use::Ledger<N>,
    role: u8,
    phase: u8,
    drelu: Option<arithmetic::PartyFor<C>>,
    select: Option<selection::PartyFor<C>>,
    tr: Option<arithmetic::PartyFor<C>>,
    low: Option<lookup::PartyFor<C>>,
    high: Option<lookup::PartyFor<C>>,
    product: Option<consuming::PartyFor<C>>,
    gap: Option<arithmetic::PartyFor<C>>,
    work: [Zeroizing<Box<[u64]>>; 5],
    h: Option<arithmetic::MaskedOutputFor<C>>,
    v: Option<selection::MaskedOutputFor<C>>,
    j1: Option<arithmetic::MaskedOutputFor<C>>,
    u0: Option<consuming::LookupValueFor<C>>,
    u1: Option<consuming::LookupValueFor<C>>,
    p: Option<consuming::ProductValueFor<C>>,
    output: Option<CommonMaskedFor<C, Ring>>,
    tables: Arc<Tables<C>>,
    _local: PhantomData<Rc<()>>,
    lease: LeaseFor<C>,
}
impl<C: AllocationContext> custody::Retire for State<C> {
    fn retire(&mut self) {
        self.output = None;
        self.cancel();
    }
}
impl<C: AllocationContext> State<C> {
    fn burn(&mut self, i: usize) -> Result<(), Error> {
        self.ledger.burn(i as u8, self.bindings[MAP[i]], self.role)
    }
    fn cancel(&mut self) {
        if self.phase == 255 {
            return;
        }
        self.phase = 255;
        self.ledger.next = N as u8;
        self.root = None;
        self.drelu = None;
        self.select = None;
        self.tr = None;
        self.low = None;
        self.high = None;
        self.product = None;
        self.gap = None;
        self.h = None;
        self.v = None;
        self.j1 = None;
        self.u0 = None;
        self.u1 = None;
        self.p = None;
        for work in &mut self.work {
            work.zeroize();
        }
    }
    fn begin(&mut self) -> Result<Body<C>, Error> {
        if self.phase != 0 {
            return Err(Error::Consumed);
        }
        self.burn(0)?;
        self.burn(1)?;
        let root = self.root.as_ref().ok_or(Error::Consumed)?;
        for i in 0..self.work[0].len() {
            self.work[0][i] = root.backing.values()[i] & ring_mask(40);
        }
        self.burn(2)?;
        for i in 0..self.work[0].len() {
            self.work[1][i] = self.work[0][i].wrapping_sub(65536) & ring_mask(40);
        }
        self.burn(3)?;
        let frame = self
            .drelu
            .as_mut()
            .ok_or(Error::Consumed)?
            .begin(&self.work[1])?;
        self.phase = 1;
        Ok(Body::Arithmetic(frame))
    }
    fn advance(&mut self, peer: &[u8]) -> Result<Option<Body<C>>, Error> {
        let body = match self.phase {
            1 => {
                let arithmetic::StepFor::Output(h) =
                    self.drelu.as_mut().ok_or(Error::Consumed)?.advance(peer)?
                else {
                    return Err(Error::Context);
                };
                self.h = Some(h);
                self.burn(4)?;
                for i in 0..self.work[0].len() {
                    self.work[2][i] = self.h.as_ref().ok_or(Error::Consumed)?.values()[i] ^ 1;
                }
                self.burn(5)?;
                for i in 0..self.work[0].len() {
                    self.work[1][i] = self.work[0][i].wrapping_sub(65535) & 65535;
                }
                self.burn(6)?;
                self.burn(7)?;
                let frame = self
                    .select
                    .as_mut()
                    .ok_or(Error::Consumed)?
                    .begin(&self.work[2], &self.work[1])?;
                self.root = None;
                self.work[0].zeroize();
                Body::Select(frame)
            }
            2 => {
                let v = self.select.as_mut().ok_or(Error::Consumed)?.finish(peer)?;
                self.burn(8)?;
                for i in 0..self.work[3].len() {
                    self.work[3][i] = v.values()[i].wrapping_add(65535) & 65535;
                }
                self.v = Some(v);
                self.burn(9)?;
                for i in 0..self.work[4].len() {
                    self.work[4][i] = self.work[3][i] & 255;
                }
                self.burn(10)?;
                let frame = self
                    .tr
                    .as_mut()
                    .ok_or(Error::Consumed)?
                    .begin(&self.work[3])?;
                self.h = None;
                Body::Arithmetic(frame)
            }
            3 => {
                let arithmetic::StepFor::Frame(frame) =
                    self.tr.as_mut().ok_or(Error::Consumed)?.advance(peer)?
                else {
                    return Err(Error::Context);
                };
                Body::Arithmetic(frame)
            }
            4 => {
                let arithmetic::StepFor::Output(j1) =
                    self.tr.as_mut().ok_or(Error::Consumed)?.advance(peer)?
                else {
                    return Err(Error::Context);
                };
                self.j1 = Some(j1);
                self.burn(11)?;
                self.burn(12)?;
                let frame = self
                    .low
                    .as_mut()
                    .ok_or(Error::Consumed)?
                    .begin(&self.work[4])?;
                self.v = None;
                self.work[3].zeroize();
                Body::Lookup(frame)
            }
            5 => {
                let lookup::StepFor::Frame(frame) =
                    self.low.as_mut().ok_or(Error::Consumed)?.advance(peer)?
                else {
                    return Err(Error::Context);
                };
                Body::Lookup(frame)
            }
            6 => {
                let lookup::StepFor::Reconstructed(value) =
                    self.low.as_mut().ok_or(Error::Consumed)?.advance(peer)?
                else {
                    return Err(Error::Context);
                };
                self.u0 = Some(consuming::LookupValueFor {
                    native: value,
                    binding: self.bindings[10],
                    role: self.role,
                    lease: self
                        .lease
                        .split(size_of::<consuming::LookupValueFor<C>>())?,
                });
                self.burn(13)?;
                Body::Lookup(
                    self.high
                        .as_mut()
                        .ok_or(Error::Consumed)?
                        .begin(self.j1.as_ref().ok_or(Error::Consumed)?.values())?,
                )
            }
            7 => {
                let lookup::StepFor::Frame(frame) =
                    self.high.as_mut().ok_or(Error::Consumed)?.advance(peer)?
                else {
                    return Err(Error::Context);
                };
                Body::Lookup(frame)
            }
            8 => {
                let lookup::StepFor::Reconstructed(value) =
                    self.high.as_mut().ok_or(Error::Consumed)?.advance(peer)?
                else {
                    return Err(Error::Context);
                };
                self.u1 = Some(consuming::LookupValueFor {
                    native: value,
                    binding: self.bindings[11],
                    role: self.role,
                    lease: self
                        .lease
                        .split(size_of::<consuming::LookupValueFor<C>>())?,
                });
                self.burn(14)?;
                self.burn(15)?;
                Body::Product(self.product.as_mut().ok_or(Error::Consumed)?.begin([
                    self.u0.take().ok_or(Error::Consumed)?,
                    self.u1.take().ok_or(Error::Consumed)?,
                ])?)
            }
            9 => {
                let p = self.product.as_mut().ok_or(Error::Consumed)?.finish(peer)?;
                self.p = Some(p);
                self.burn(16)?;
                let frame = self
                    .gap
                    .as_mut()
                    .ok_or(Error::Consumed)?
                    .begin(self.p.as_ref().ok_or(Error::Consumed)?.native.values())?;
                self.u0 = None;
                self.u1 = None;
                Body::Arithmetic(frame)
            }
            10 => {
                let arithmetic::StepFor::Frame(frame) =
                    self.gap.as_mut().ok_or(Error::Consumed)?.advance(peer)?
                else {
                    return Err(Error::Context);
                };
                Body::Arithmetic(frame)
            }
            11 => {
                let arithmetic::StepFor::Output(value) =
                    self.gap.as_mut().ok_or(Error::Consumed)?.advance(peer)?
                else {
                    return Err(Error::Context);
                };
                self.burn(17)?;
                let lease = self.lease.split(common_charge_for::<C>())?;
                self.output = Some(CommonMaskedFor {
                    backing: CommonBackingFor::Arithmetic(value),
                    binding: self.bindings[13],
                    role: self.role,
                    _kind: PhantomData,
                    _lease: lease,
                });
                self.cancel();
                return Ok(None);
            }
            _ => return Err(Error::Consumed),
        };
        self.phase += 1;
        Ok(Some(body))
    }
}

#[allow(clippy::large_enum_variant)]
enum Body<C: AllocationContext> {
    Arithmetic(arithmetic::FrameFor<C>),
    Select(selection::FrameFor<C>),
    Lookup(lookup::FrameFor<C>),
    Product(product::FrameFor<C>),
}
impl<C: AllocationContext> Body<C> {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Arithmetic(x) => x.as_bytes(),
            Self::Select(x) => x.as_bytes(),
            Self::Lookup(x) => x.as_bytes(),
            Self::Product(x) => x.as_bytes(),
        }
    }
}
pub struct Coordinator<C: AllocationContext> {
    shared: Arc<Shared<C>>,
}
/// Dealer output mask remains local/opaque and retains original source/range
/// custody, even after all role/frame handles are gone. No raw detach/Clone.
pub struct Mask<C: AllocationContext> {
    native: DealerMaskFor<C, Ring>,
    shared: Arc<Shared<C>>,
}
impl<C: AllocationContext> Mask<C> {
    pub fn descriptor(&self) -> Result<EdgeDescriptor, Error> {
        self.shared
            .operation(|_| Ok(self.native.binding.descriptor))
    }
}
impl<C: AllocationContext> Drop for Mask<C> {
    fn drop(&mut self) {
        self.shared.abort();
    }
}
impl<C: AllocationContext> Coordinator<C> {
    pub fn cancel(&mut self) {
        self.shared.abort();
    }
}
impl<C: AllocationContext> Drop for Coordinator<C> {
    fn drop(&mut self) {
        self.shared.abort();
    }
}
pub struct Role<C: AllocationContext> {
    slot: Arc<Slot<C>>,
    shared: Arc<Shared<C>>,
    role: u8,
    handed_off: bool,
}
pub struct Frame<C: AllocationContext> {
    native: Body<C>,
    shared: Arc<Shared<C>>,
}
impl<C: AllocationContext> Frame<C> {
    /// Borrow the original local peer codec only while the region is live.
    /// Already borrowed bytes cannot be revoked; this is not transport auth.
    pub fn bytes(&self) -> Result<&[u8], Error> {
        self.shared.operation(|_| Ok(self.native.bytes()))
    }
}
#[allow(clippy::large_enum_variant)]
pub enum Step<C: AllocationContext> {
    Frame(Frame<C>),
    Ready,
}
pub struct Output<C: AllocationContext> {
    slot: Arc<Slot<C>>,
    shared: Arc<Shared<C>>,
    role: u8,
}
impl<C: AllocationContext> Output<C> {
    pub fn role(&self) -> u8 {
        self.role
    }

    pub fn descriptor(&self) -> Result<EdgeDescriptor, Error> {
        self.shared.operation(|_| {
            self.slot
                .state
                .lock()
                .map_err(|_| Error::Consumed)?
                .output
                .as_ref()
                .map(CommonMaskedFor::descriptor)
                .ok_or(Error::Consumed)
        })
    }
}
impl<C: AllocationContext> Drop for Output<C> {
    fn drop(&mut self) {
        self.shared.abort();
    }
}
impl<C: AllocationContext> Role<C> {
    pub fn role(&self) -> u8 {
        self.role
    }

    pub fn begin(&mut self, cancelled: &mut impl FnMut() -> bool) -> Result<Frame<C>, Error> {
        self.shared.poll_cancelled(cancelled)?;
        self.shared.operation(|_| {
            let native = self
                .slot
                .state
                .lock()
                .map_err(|_| Error::Consumed)?
                .begin()?;
            Ok(Frame {
                native,
                shared: Arc::clone(&self.shared),
            })
        })
    }
    pub fn advance(
        &mut self,
        peer: &[u8],
        cancelled: &mut impl FnMut() -> bool,
    ) -> Result<Step<C>, Error> {
        self.shared.poll_cancelled(cancelled)?;
        self.shared.operation(|control| {
            let mut state = self.slot.state.lock().map_err(|_| Error::Consumed)?;
            match state.advance(peer)? {
                Some(native) => Ok(Step::Frame(Frame {
                    native,
                    shared: Arc::clone(&self.shared),
                })),
                None => {
                    control.mark_ready(self.role);
                    Ok(Step::Ready)
                }
            }
        })
    }
    pub fn finish(mut self) -> Result<Output<C>, Error> {
        let output = self.shared.operation(|control| {
            control.require_both_ready()?;
            if self
                .slot
                .state
                .lock()
                .map_err(|_| Error::Consumed)?
                .output
                .is_none()
            {
                return Err(Error::Consumed);
            }
            Ok(Output {
                slot: Arc::clone(&self.slot),
                shared: Arc::clone(&self.shared),
                role: self.role,
            })
        });
        self.handed_off = output.is_ok();
        output
    }
}
impl<C: AllocationContext> Drop for Role<C> {
    fn drop(&mut self) {
        if !self.handed_off {
            self.shared.abort();
        }
    }
}

#[cfg(test)]
#[path = "nexp_region_tests.rs"]
mod tests;

#[path = "nexp_integer_oracle.rs"]
mod integer_oracle;
