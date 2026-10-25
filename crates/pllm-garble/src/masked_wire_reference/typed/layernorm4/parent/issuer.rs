//! D43 local clear ROOT issuance. No runtime completion/public source issuer.
use super::super::super::super::{
    elementwise_product as product,
    source_lookup::{interval50 as paired, norm13},
};
use super::super::super::role_coordination::{
    self, CoordinationFor, Retire, RoleSlotFor, SplitControl,
};
use super::coordinate;
use super::*;
use zeroize::Zeroize;
#[path = "runtime.rs"]
mod runtime;
use crate::shared_allocation::MAX_ALLOCATION_BYTES;
use pllm_core::edge_contract::{EdgeIdentity, Shape};
use std::sync::Mutex;

const INPUT_SLOTS: [usize; USES] = [
    0, 3, 0, 4, 5, 5, 6, 7, 7, 7, 7, 9, 10, 11, 12, 8, 13, 5, 14, 15, 16, 1, 17, 2,
];
const SHAPES: [usize; SLOTS] = [4, 4, 4, 1, 1, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4];
const FRACTIONS: [u8; SLOTS] = [
    12, 12, 12, 24, 12, 12, 24, 24, 0, 0, 0, 0, 0, 0, 12, 24, 12, 24,
];
const BITS: [u8; SLOTS] = [
    50, 50, 50, 50, 50, 50, 50, 50, 13, 50, 50, 8, 7, 13, 50, 50, 50, 50,
];
fn checked_sum(a: &[usize]) -> Result<usize, Error> {
    a.iter()
        .try_fold(0usize, |v, x| v.checked_add(*x).ok_or(Error::Resource))
}
fn gap(lanes: usize) -> arithmetic::Spec {
    arithmetic::Spec {
        bits: 50,
        lanes,
        gate: arithmetic::Gate::GapArs { shift: 12 },
    }
}
fn arc<T>() -> Result<usize, Error> {
    role_coordination::arc_bytes::<T>()?
        .checked_add(64)
        .ok_or(Error::Resource)
}
fn promise() -> Result<BoundRange<LocalAssumption>, Error> {
    BoundRange::new(-4096, 4096)
}

/// No caller-selected profile, table, descriptor, intermediate or source seal.
#[allow(dead_code)] // Closed local-row ingress; no SDK/default activation.
pub(super) struct LocalClearRootSpec4<'a> {
    codes: [&'a [i64; 4]; 3],
}
#[allow(dead_code)] // Real consuming ingress retained private pending final review.
impl<'a> LocalClearRootSpec4<'a> {
    pub(super) fn new(
        x: &'a [i64; 4],
        gamma: &'a [i64; 4],
        beta: &'a [i64; 4],
    ) -> Result<Self, Error> {
        if [x, gamma, beta]
            .iter()
            .any(|v| v.iter().any(|x| !(-4096..=4095).contains(x)))
        {
            return Err(Error::Domain);
        }
        Ok(Self {
            codes: [x, gamma, beta],
        })
    }
}

struct Metadata {
    descriptors: [EdgeDescriptor; SLOTS],
    kinds: [ValueKind; SLOTS],
    uses: [ConsumerUse; USES],
    output: EdgeDescriptor,
    identity: SourceIdentity,
    profile: [u8; 32],
    operation: [u8; 32],
}
fn metadata() -> Result<Metadata, Error> {
    // This names the LOCAL constructor, NOT checkpoint/source authentication.
    let profile = digest(&[b"pllm.sigma.layernorm4.localclear.paper.v1"]);
    let operation = digest(&[b"pllm.sigma.layernorm4.localclear.consumer.v1"]);
    let identity = SourceIdentity::new(profile, profile, approved_implementation());
    let edge = |i: usize, lanes, bits, fraction, interpretation| -> Result<EdgeDescriptor, Error> {
        let mut wire = [0; 32];
        wire[..8].copy_from_slice(&(i as u64).to_le_bytes());
        EdgeDescriptor::new(
            EdgeIdentity::new(identity, profile, profile, wire, 0),
            Shape::new(&[lanes]).map_err(|_| Error::Domain)?,
            Numeric::new(bits, fraction, interpretation).map_err(|_| Error::Domain)?,
            Representation::CommonAdditiveMasked,
            Some(WeightVisibility::Private),
        )
        .map_err(|_| Error::Domain)
    };
    let mut descriptors = [edge(0, 4, 50, 12, Interpretation::Signed)?; SLOTS];
    let mut kinds = [ValueKind::SignedFixed {
        bits: 50,
        fraction: 12,
    }; SLOTS];
    for i in 0..SLOTS {
        let residue = (8..=13).contains(&i);
        descriptors[i] = edge(
            i,
            SHAPES[i],
            BITS[i],
            FRACTIONS[i],
            if residue {
                Interpretation::Unsigned
            } else {
                Interpretation::Signed
            },
        )?;
        kinds[i] = if residue {
            ValueKind::UnsignedResidue { bits: BITS[i] }
        } else {
            ValueKind::SignedFixed {
                bits: 50,
                fraction: FRACTIONS[i],
            }
        };
    }
    let operands: [u16; USES] = [
        0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1,
    ];
    let uses = std::array::from_fn(|i| {
        ConsumerUse::new(
            descriptors[INPUT_SLOTS[i]],
            operation,
            operands[i],
            i as u32,
        )
    });
    Ok(Metadata {
        descriptors,
        kinds,
        uses,
        output: edge(18, 4, 50, 12, Interpretation::Signed)?,
        identity,
        profile,
        operation,
    })
}

/// Every native child and original mask is retained, even after its numeric use.
/// No source/coordination backlink exists from this keeper.
#[allow(dead_code)] // Retains ALL original masks/dealer/source/table cover; no early phase credit.
pub(super) struct RowMaterial<C: AllocationContext> {
    source: Arc<SourceFor<C>>,
    masks: [Option<ProducedMaskFor<C>>; SLOTS],
    modular: coordinate::ModularMask<C>,
    coordinate: coordinate::CoordinateMask<C>,
    terminal_mask: DealerMaskFor<C, Ring>,
    dealer: LocalDealerFor<C>,
    bindings: [Binding; SLOTS],
    leases: [LeaseFor<C>; 4],
    issuance_work: IssuanceWork,
}
#[allow(clippy::large_enum_variant)] // Actual inline native owners are explicitly priced.
enum FrameOwner<C: AllocationContext> {
    Arithmetic(arithmetic::FrameFor<C>),
    Product(product::FrameFor<C>),
    Interval(paired::FramesFor<C>),
    Inverse(norm13::FrameFor<C>),
    Modular(coordinate::ModularFrame<C>),
}
struct InitializerState<C: AllocationContext> {
    values: [Option<ProducedValueFor<C>>; SLOTS],
    ledger: internal_use::Ledger<USES>,
    mean: arithmetic::PartyFor<C>,
    square: product::PartyFor<C>,
    interval: paired::PartyFor<C>,
    modular: coordinate::ModularParty<C>,
    coordinate: coordinate::CoordinateParty<C>,
    #[cfg(test)]
    retirement_probe: RetirementProbe<C>,
    inverse: norm13::PartyFor<C>,
    normalize: product::PartyFor<C>,
    floor: arithmetic::PartyFor<C>,
    affine: product::PartyFor<C>,
    terminal: Option<gapars_terminal::PartyFor<C>>,
    frames: [Option<FrameOwner<C>>; 17],
    share: Option<gapars_terminal::ShareFor<C>>,
    phase: u8,
    // Original scalar broadcasts are MASKS, never clear T. Role value broadcast
    // buffers are allocated before issuance and populated only during dispatch.
    broadcast: Zeroizing<Vec<u64>>,
    local_buffers: [Zeroizing<Vec<u64>>; 3],
    local_payloads: [Option<LeaseFor<C>>; 3],
    #[allow(dead_code)] // Immutable prepaid summary, never reset by a runtime transition.
    work: Work,
    runtime_lease: LeaseFor<C>,
    // MUST follow every dependent native/table/source/frame/output owner.
    keeper: Arc<RowMaterial<C>>,
}
#[cfg(test)]
struct RetirementProbe<C: AllocationContext> {
    material: std::sync::Weak<RowMaterial<C>>,
    observed: Option<std::rc::Rc<std::cell::Cell<usize>>>,
}
#[cfg(test)]
impl<C: AllocationContext> Drop for RetirementProbe<C> {
    fn drop(&mut self) {
        if let Some(n) = &self.observed {
            assert!(
                self.material.strong_count() > 0,
                "material/table allocation cover retired before native inverse"
            );
            n.set(n.get() + 1)
        }
    }
}
impl<C: AllocationContext> Retire for InitializerState<C> {
    fn retire(&mut self) {
        if self.phase == u8::MAX {
            return;
        }
        self.phase = u8::MAX;
        self.mean.cancel();
        self.square.cancel();
        self.interval.cancel();
        self.modular.cancel();
        self.coordinate.cancel();
        self.inverse.cancel();
        self.normalize.cancel();
        self.floor.cancel();
        self.affine.cancel();
        if let Some(t) = &mut self.terminal {
            t.cancel()
        }
        self.share = None;
        for v in &mut self.values {
            *v = None;
        }
        self.broadcast.iter_mut().for_each(|v| *v = 0);
        for v in &mut self.local_buffers {
            v.zeroize()
        }
    }
}
pub(super) struct PartyInitializer<C: AllocationContext> {
    coordination: Arc<CoordinationFor<C, InitializerState<C>>>,
    role: u8,
    handed_off: bool,
}
impl<C: AllocationContext> Drop for PartyInitializer<C> {
    fn drop(&mut self) {
        if !self.handed_off {
            self.coordination.abort();
        }
    }
}
#[allow(dead_code)] // Closed component handoff; no public/default activation.
pub(super) struct IssuedRow<C: AllocationContext> {
    pub material: Arc<RowMaterial<C>>,
    pub parties: [PartyInitializer<C>; 2],
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)] // Named finite native work inventory; not private implementation CPU qualification.
pub(super) struct Work {
    pub consumer_uses: usize,
    pub table_certificates: usize,
    pub table_integer_steps: usize,
    pub interval_less_queries: usize,
    pub native_factories: usize,
    pub entropy_bytes: usize,
    pub issue_aes_blocks: usize,
    pub evaluator_aes_blocks_per_party: usize,
    pub product_lanes: usize,
    pub frames_per_party: usize,
}
// Noncloneable whole-row prepayment, shared across all constructor fills; no
// child resets it. Native CPU/private compiled-stack bounds remain unqualified.
struct IssuanceWork {
    remaining_entropy: usize,
    spent_entropy: usize,
}
pub(super) struct Resources {
    pub total_bytes: usize,
    #[allow(dead_code)] // Audit-visible included original LESS/AES allowance, not extra credit.
    pub included_original_lookup_workspace_bytes: usize,
    pub work: Work,
    root: usize,
    dealer: usize,
    source: usize,
    table: usize,
    keeper: usize,
    party: usize,
    coordination: usize,
    scratch: usize,
    wrappers: usize,
    mean: usize,
    square: usize,
    interval: usize,
    modular: usize,
    coordinate: usize,
    inverse: usize,
    normalize: usize,
    floor: usize,
    affine: usize,
    terminal: usize,
    clear_copy: usize,
    runtime: usize,
}
#[allow(dead_code)] // Concrete simultaneous issuance/layout slots.
struct IssuanceControls<'a, C: AllocationContext> {
    metadata: Metadata,
    resources: Resources,
    spec: LocalClearRootSpec4<'a>,
    clear: Zeroizing<[[i64; 4]; 3]>,
    encoded: Option<EncodedRootFor<C>>,
    root: Option<RootInputFor<C>>,
    material: Option<RowMaterial<C>>,
    returned: Result<IssuedRow<C>, Error>,
    parties: Option<[InitializerState<C>; 2]>,
    masks: [Option<ProducedMaskFor<C>>; SLOTS],
    values: [[Option<ProducedValueFor<C>>; SLOTS]; 2],
    ledger: internal_use::Ledger<USES>,
    bindings: [Binding; SLOTS],
    promises: [Interval; SLOTS],
    native_product: Result<(product::MasksFor<C>, [product::PartyFor<C>; 2]), Error>,
    native_arithmetic: Result<(arithmetic::OutputMasksFor<C>, [arithmetic::PartyFor<C>; 2]), Error>,
    native_interval: Result<(paired::MasksFor<C>, [paired::PartyFor<C>; 2]), Error>,
    native_inverse: Result<(norm13::MasksFor<C>, [norm13::PartyFor<C>; 2]), Error>,
    modular: Result<(coordinate::ModularMask<C>, [coordinate::ModularParty<C>; 2]), Error>,
    coordinate: Result<
        (
            coordinate::CoordinateMask<C>,
            [coordinate::CoordinateParty<C>; 2],
        ),
        Error,
    >,
    terminal: gapars_terminal::RetainedResult<C>,
    leases: [Result<LeaseFor<C>, Error>; 24],
    source_arc: Arc<SourceFor<C>>,
    table_arc: Arc<table::CertifiedTableFor<C>>,
    keeper_arc: Arc<RowMaterial<C>>,
    slots: [Arc<RoleSlotFor<C, InitializerState<C>>>; 2],
    mask_copies: Zeroizing<[[u64; 4]; 3]>,
    work_ledger: IssuanceWork,
}
struct FrameHandle<C: AllocationContext> {
    coordination: Arc<CoordinationFor<C, InitializerState<C>>>,
    role: u8,
    slot: u8,
}
struct ShareHandle<C: AllocationContext> {
    coordination: Arc<CoordinationFor<C, InitializerState<C>>>,
    role: u8,
}
#[allow(dead_code)] // Owning frame/Ready result for the closed component only.
enum RowStep<C: AllocationContext> {
    Frame(FrameHandle<C>),
    Ready,
}
#[allow(dead_code)] // Runtime return/loan/move slots included in per-party admission.
struct TransitionControls<'a, C: AllocationContext> {
    operation: role_coordination::OperationFor<'a, C, InitializerState<C>>,
    state: std::sync::MutexGuard<'a, InitializerState<C>>,
    begin: Result<FrameHandle<C>, Error>,
    advance: Result<RowStep<C>, Error>,
    inverse_loan: Norm13ValueLoan<'a, C>,
    energy_loan: EnergyValueLoan<'a, C>,
    moving_share: Option<gapars_terminal::ShareFor<C>>,
    share_return: Result<ShareHandle<C>, Error>,
    frame_loan: runtime::FrameLoan<'a, C>,
    share_loan: runtime::ShareLoan<'a, C>,
    product_output: Result<product::OutputFor<C>, Error>,
    arithmetic_step: Result<arithmetic::StepFor<C>, Error>,
    inverse_step: Result<norm13::Step<C>, Error>,
    paired_step: Result<paired::Step<C>, Error>,
    native_frame: Result<Option<FrameOwner<C>>, Error>,
    local_result: Zeroizing<Vec<u64>>,
    beta_input: Option<ProducedValueFor<C>>,
    native_scalar: Zeroizing<u64>,
}

/// Allocation-free actual-layout aggregate. Native child factories consume
/// these exact slices; no child independently reserves/refunds the full cap.
#[allow(dead_code)] // Closed native component query; no outward registration.
pub(super) fn resources<C: AllocationContext>() -> Result<Resources, Error> {
    let root = root_payload_bytes::<C>(4)?;
    let dealer = size_of::<LocalDealerFor<C>>()
        .checked_add(4 << 10)
        .ok_or(Error::Resource)?;
    let source = arc::<SourceFor<C>>()?;
    let table = checked_sum(&[
        table::reservation_bytes::<C>()?,
        arc::<table::CertifiedTableFor<C>>()?,
    ])?;
    let keeper = checked_sum(&[
        arc::<RowMaterial<C>>()?,
        size_of::<IssuanceControls<'_, C>>(),
        C::TRANSIENT_LINK_BYTES
            .checked_mul(32)
            .ok_or(Error::Resource)?,
    ])?;
    let party = checked_sum(&[
        arc::<RoleSlotFor<C, InitializerState<C>>>()?,
        4 * 8 + 64,
        size_of::<PartyInitializer<C>>(),
        size_of::<runtime::Role<C>>(),
        size_of::<TransitionControls<'_, C>>(),
        17usize
            .checked_mul(size_of::<FrameHandle<C>>())
            .ok_or(Error::Resource)?,
        size_of::<ShareHandle<C>>(),
    ])?;
    let coordination = arc::<CoordinationFor<C, InitializerState<C>>>()?;
    let scratch = 8 * 4 + 64; // Original dealer's scalar-T broadcast mask only.
                              // Each record's actual inline body, vector allocator and typed native
                              // wrapper overlap stays admitted; unused shares are never credited back.
    let wrappers = checked_sum(&[
        size_of::<ProducedMaskFor<C>>(),
        2 * size_of::<ProducedValueFor<C>>(),
        mask_charge_for::<C>(),
        2 * common_charge_for::<C>(),
        3 * (4 * 8 + 64),
    ])?
    .checked_mul(SLOTS)
    .ok_or(Error::Resource)?;
    let mean = arithmetic::resources_for_context::<C>(gap(1))?.combined_reservation_bytes;
    let square = product::resources_for::<C>(4)?.combined_bytes;
    let interval = paired::resources::<C>()?.combined;
    let structural = coordinate::resources::<C>()?;
    let inverse = norm13::resources::<C>()?.combined_reservation_bytes;
    let normalize = product::resources_for::<C>(4)?.combined_bytes;
    let floor = arithmetic::resources_for_context::<C>(gap(4))?.combined_reservation_bytes;
    let affine = normalize;
    let terminal = gapars_terminal::resources::<C>()?.reservation_bytes;
    let clear_copy = checked_sum(&[
        size_of::<Zeroizing<[[i64; 4]; 3]>>(),
        size_of::<LeaseFor<C>>(),
    ])?;
    let runtime = checked_sum(&[
        SLOTS
            .checked_mul(size_of::<ProducedValueFor<C>>())
            .ok_or(Error::Resource)?,
        3 * common_charge_for::<C>(),
        6 * 8 + 3 * 64,
        4 * size_of::<LeaseFor<C>>(),
    ])?;
    let aq = [
        arithmetic::resources_for_context::<C>(gap(1))?,
        arithmetic::resources_for_context::<C>(gap(4))?,
        arithmetic::resources_for_context::<C>(arithmetic::Spec {
            bits: 50,
            lanes: 1,
            gate: arithmetic::Gate::Tr { shift: 42 },
        })?,
        gapars_terminal::resources::<C>()?.native,
    ];
    let pq = [
        product::resources_for::<C>(4)?,
        product::resources_for::<C>(4)?,
        product::resources_for::<C>(4)?,
        product::resources_for::<C>(1)?,
    ];
    let iw = paired::resources::<C>()?.work_counts()?;
    let nq = norm13::resources::<C>()?;
    let entropy_bytes = checked_sum(&[
        32,
        3 * 4 * 8,
        aq.iter().try_fold(0usize, |n, q| {
            n.checked_add(q.random_bytes).ok_or(Error::Resource)
        })?,
        pq.iter().try_fold(0usize, |n, q| {
            n.checked_add(q.random_bytes).ok_or(Error::Resource)
        })?,
        iw[0],
        nq.random_bytes,
        gapars_terminal::resources::<C>()?.extra_random_bytes,
    ])?;
    let issue_aes_blocks = checked_sum(&[
        aq.iter().try_fold(0usize, |n, q| {
            n.checked_add(q.paired_issue_aes_blocks)
                .ok_or(Error::Resource)
        })?,
        iw[1],
        nq.paired_issue_aes_blocks,
    ])?;
    let evaluator_aes_blocks_per_party = checked_sum(&[
        aq.iter().try_fold(0usize, |n, q| {
            n.checked_add(q.evaluation_aes_blocks_per_party)
                .ok_or(Error::Resource)
        })?,
        iw[2],
        nq.evaluation_aes_blocks_per_party,
    ])?;
    let mut frames_per_party = checked_sum(&[
        aq[0].frames_per_party,
        aq[1].frames_per_party,
        aq[2].frames_per_party,
        paired::resources::<C>()?.frames_per_party()?,
        nq.frames_per_party,
        gapars_terminal::resources::<C>()?.terminal_frames_per_party,
    ])?;
    for p in &pq {
        let both = p.frame_bytes.checked_mul(2).ok_or(Error::Resource)?;
        if both == 0 || p.total_peer_bytes % both != 0 {
            return Err(Error::Resource);
        }
        frames_per_party = frames_per_party
            .checked_add(p.total_peer_bytes / both)
            .ok_or(Error::Resource)?;
    }
    if frames_per_party != 17 {
        return Err(Error::Unsupported);
    }
    let total_bytes = checked_sum(&[
        root.checked_mul(3).ok_or(Error::Resource)?,
        dealer,
        source,
        table,
        keeper,
        party.checked_mul(2).ok_or(Error::Resource)?,
        coordination,
        scratch,
        wrappers,
        mean,
        square,
        interval,
        structural.product_bytes,
        structural.coordinate_bytes,
        inverse,
        normalize,
        floor,
        affine,
        terminal,
        clear_copy,
        runtime.checked_mul(2).ok_or(Error::Resource)?,
    ])?;
    if total_bytes > MAX_ALLOCATION_BYTES {
        return Err(Error::Resource);
    }
    Ok(Resources {
        total_bytes,
        included_original_lookup_workspace_bytes: paired::resources::<C>()?
            .original_native_workspace_bytes()?,
        work: Work {
            consumer_uses: USES,
            table_certificates: 8192,
            table_integer_steps: 8192 * 26,
            interval_less_queries: 2 * 51,
            native_factories: 10,
            entropy_bytes,
            issue_aes_blocks,
            evaluator_aes_blocks_per_party,
            frames_per_party,
            product_lanes: pq
                .iter()
                .try_fold(0usize, |n, q| n.checked_add(q.lanes).ok_or(Error::Resource))?,
        },
        root,
        dealer,
        source,
        table,
        keeper,
        party,
        coordination,
        scratch,
        wrappers,
        mean,
        square,
        interval,
        modular: structural.product_bytes,
        coordinate: structural.coordinate_bytes,
        inverse,
        normalize,
        floor,
        affine,
        terminal,
        clear_copy,
        runtime,
    })
}

fn vector<C: AllocationContext>(
    n: usize,
    lease: &LeaseFor<C>,
) -> Result<Zeroizing<Vec<u64>>, Error> {
    let mut v = crate::shared_allocation::allocate(n, lease.budget.budget())?;
    v.resize(n, 0);
    Ok(Zeroizing::new(v))
}
fn original_mask<C: AllocationContext>(
    native: product::MasksFor<C>,
    binding: Binding,
    source: &Arc<SourceFor<C>>,
    lease: &mut LeaseFor<C>,
) -> Result<ProducedMaskFor<C>, Error> {
    Ok(ProducedMaskFor {
        backing: MaskBacking::Original(DealerMaskFor {
            backing: MaskBackingFor::Elementwise(native),
            binding,
            _kind: PhantomData,
            _lease: lease.split(mask_charge_for::<C>())?,
        }),
        binding,
        source: Arc::clone(source),
        lease: lease.split(size_of::<ProducedMaskFor<C>>())?,
    })
}
fn arithmetic_context(
    source: &SourceFor<impl AllocationContext>,
    input: usize,
    output: usize,
) -> arithmetic::Context {
    arithmetic::Context {
        plan: source.profile,
        session: source.session,
        operation: source.operation,
        input_wire: source.descriptors[input].digest(),
        output_wire: source.descriptors[output].digest(),
        first_tensor_index: 0,
    }
}
fn product_context<C: AllocationContext>(
    source: &SourceFor<C>,
    left_use: usize,
    right_use: usize,
    output: usize,
) -> product::Context {
    product::Context {
        region: source.profile,
        stage: source.operation,
        inputs: [
            source.uses[left_use].digest(),
            source.uses[right_use].digest(),
        ],
        output: source.descriptors[output].digest(),
        first: 0,
    }
}
fn ledger<C: AllocationContext>(
    source: &SourceFor<C>,
    bindings: &[Binding; SLOTS],
    role: u8,
) -> internal_use::Ledger<USES> {
    internal_use::Ledger {
        uses: source.uses,
        bindings: std::array::from_fn(|i| bindings[INPUT_SLOTS[i]]),
        role,
        next: 0,
    }
}
fn local<C: AllocationContext>(
    source: &Arc<SourceFor<C>>,
    slot: usize,
    binding: Binding,
    lease: &mut LeaseFor<C>,
) -> Result<ProducedMaskFor<C>, Error> {
    Ok(ProducedMaskFor {
        backing: MaskBacking::Local(vector(SHAPES[slot], lease)?),
        binding,
        source: Arc::clone(source),
        lease: lease.split(size_of::<ProducedMaskFor<C>>() + SHAPES[slot] * 8 + 64)?,
    })
}
fn mutable<C: AllocationContext>(mask: &mut ProducedMaskFor<C>) -> Result<&mut [u64], Error> {
    match &mut mask.backing {
        MaskBacking::Local(v) => Ok(v),
        _ => Err(Error::Context),
    }
}
fn get<C: AllocationContext>(
    masks: &[Option<ProducedMaskFor<C>>; SLOTS],
    slot: usize,
) -> Result<&ProducedMaskFor<C>, Error> {
    masks[slot].as_ref().ok_or(Error::Consumed)
}

#[allow(dead_code)] // Full native issuer intentionally private pending independent review.
pub(super) fn issue_reserved_with<C: AllocationContext>(
    spec: LocalClearRootSpec4<'_>,
    session: [u8; 32],
    mut reservation: LeaseFor<C>,
    fill: impl FnMut(&mut [u8]) -> Result<(), Error>,
) -> Result<IssuedRow<C>, Error> {
    let q = resources::<C>()?;
    if reservation.bytes != q.total_bytes {
        return Err(Error::Resource);
    }
    let _clear_guard = reservation.split(q.clear_copy)?;
    let clear = Zeroizing::new([*spec.codes[0], *spec.codes[1], *spec.codes[2]]);
    let m = metadata()?;
    // All child slices AND every containing/native transition are admitted now.
    // This first entropy precedes no reservation or unpriced source factory.
    let mut issuance_work = IssuanceWork {
        remaining_entropy: q
            .work
            .entropy_bytes
            .checked_sub(32)
            .ok_or(Error::Resource)?,
        spent_entropy: 32,
    };
    let mut supplied_fill = fill;
    let mut fill = |bytes: &mut [u8]| -> Result<(), Error> {
        issuance_work.remaining_entropy = issuance_work
            .remaining_entropy
            .checked_sub(bytes.len())
            .ok_or(Error::Resource)?;
        issuance_work.spent_entropy = issuance_work
            .spent_entropy
            .checked_add(bytes.len())
            .ok_or(Error::Resource)?;
        supplied_fill(bytes)
    };
    let mut dealer = LocalDealerFor::from_reserved_namespace(
        m.identity,
        m.profile,
        session,
        reservation.split(q.dealer)?,
    )?;
    let mut wrappers = reservation.split(q.wrappers)?;
    let keeper_lease = reservation.split(q.keeper)?;
    let left_lease = reservation.split(q.party)?;
    let right_lease = reservation.split(q.party)?;
    let runtime_left = reservation.split(q.runtime)?;
    let runtime_right = reservation.split(q.runtime)?;
    let coord_lease = reservation.split(q.coordination)?;
    let scratch_lease = reservation.split(q.scratch)?;
    let mut table_lease = reservation.split(q.table)?;
    let table_value = table::construct(table_lease.split(table::reservation_bytes::<C>()?)?)?;
    reservation.budget.budget().allocation()?;
    let certified = Arc::new(table_value);
    let source_lease = reservation.split(q.source)?;
    reservation.budget.budget().allocation()?;
    let source = Arc::new(SourceFor {
        profile: m.profile,
        session,
        operation: m.operation,
        descriptors: m.descriptors,
        kinds: m.kinds,
        uses: m.uses,
        table: certified,
        lease: source_lease,
    });
    let mut masks: [Option<ProducedMaskFor<C>>; SLOTS] = std::array::from_fn(|_| None);
    let mut values: [[Option<ProducedValueFor<C>>; SLOTS]; 2] =
        std::array::from_fn(|_| std::array::from_fn(|_| None));
    for i in 0..3 {
        let root = dealer.prepare_root_reserved_with(
            m.descriptors[i],
            promise()?,
            m.operation,
            reservation.split(q.root)?,
            &mut fill,
        )?;
        let (mask, [a, b]) = root.encode(&clear[i])?;
        let binding = mask.binding;
        masks[i] = Some(ProducedMaskFor {
            backing: MaskBacking::Original(mask),
            binding,
            source: Arc::clone(&source),
            lease: wrappers.split(size_of::<ProducedMaskFor<C>>())?,
        });
        for (role, value) in [a, b].into_iter().enumerate() {
            values[role][i] = Some(ProducedValueFor {
                binding,
                role: role as u8,
                backing: ValueBacking::Original(value),
                source: Arc::clone(&source),
                lease: wrappers.split(size_of::<ProducedValueFor<C>>())?,
            });
        }
    }
    // Exact root-derived promises. No clear normalization or value-dependent
    // descriptor selection is performed by the dealer.
    let mut bindings = [get(&masks, 0)?.binding; SLOTS];
    for (i, binding) in bindings.iter_mut().enumerate().take(3) {
        *binding = get(&masks, i)?.binding;
    }
    for (i, binding) in bindings.iter_mut().enumerate().skip(3) {
        let range = match i {
            3 | 15 | 17 => Some(Interval {
                lower: -(1i128 << 39),
                upper: 1i128 << 39,
            }),
            4 | 5 | 16 => Some(Interval {
                lower: -(1i128 << 27),
                upper: 1i128 << 27,
            }),
            6 => Some(Interval {
                lower: 0,
                upper: 1i128 << 28,
            }),
            7 => Some(Interval {
                lower: 1,
                upper: 1i128 << 29,
            }),
            14 => Some(Interval {
                lower: 0,
                upper: (1i128 << 25) + 1,
            }),
            _ => None,
        };
        *binding = dealer.binding(m.descriptors[i], m.operation, range)?;
    }
    let mut uses = ledger(&source, &bindings, 0);
    masks[3] = Some(local(&source, 3, bindings[3], &mut wrappers)?);
    let mut copy = Zeroizing::new([0u64; 4]);
    copy.copy_from_slice(get(&masks, 0)?.backing.values());
    LocalMap::ScaledMean.apply(
        &*copy,
        &[],
        mutable(masks[3].as_mut().ok_or(Error::Consumed)?)?,
        true,
    )?;
    uses.burn(0, bindings[0], 0)?;
    uses.burn(1, bindings[3], 0)?;
    let (mean, [mean0, mean1]) = arithmetic::issue_reserved_with(
        gap(1),
        arithmetic_context(&source, 3, 4),
        get(&masks, 3)?.backing.values(),
        reservation.split(q.mean)?,
        &mut fill,
    )?;
    masks[4] = Some(ProducedMaskFor {
        backing: MaskBacking::Arithmetic(mean),
        binding: bindings[4],
        source: Arc::clone(&source),
        lease: wrappers.split(size_of::<ProducedMaskFor<C>>())?,
    });
    masks[5] = Some(local(&source, 5, bindings[5], &mut wrappers)?);
    let mu = Zeroizing::new([get(&masks, 4)?.backing.values()[0]]);
    LocalMap::Center.apply(
        &*copy,
        &*mu,
        mutable(masks[5].as_mut().ok_or(Error::Consumed)?)?,
        true,
    )?;
    uses.burn(2, bindings[0], 0)?;
    uses.burn(3, bindings[4], 0)?;
    uses.burn(4, bindings[5], 0)?;
    uses.burn(5, bindings[5], 0)?;
    // Distinct ConsumerUse IDs, SAME genuine Z owner on both operands. Original
    // Product algebra/mask draws are unchanged; no host square substitutes it.
    let z = get(&masks, 5)?.backing.values();
    let (square, [square0, square1]) = product::issue_reserved(
        product_context(&source, 4, 5, 6),
        z,
        z,
        reservation.split(q.square)?,
        &mut fill,
    )?;
    masks[6] = Some(original_mask(square, bindings[6], &source, &mut wrappers)?);
    masks[7] = Some(local(&source, 7, bindings[7], &mut wrappers)?);
    // Dealer transforms masks WITHOUT the +1 public value constant.
    copy.copy_from_slice(get(&masks, 6)?.backing.values());
    LocalMap::Energy.apply(
        &*copy,
        &[],
        mutable(masks[7].as_mut().ok_or(Error::Consumed)?)?,
        true,
    )?;
    uses.burn(6, bindings[6], 0)?;
    let loan = EnergyMaskLoan::take(&source, &mut uses, &masks)?;
    let (interval, [interval0, interval1]) =
        paired::issue_reserved_with(loan, reservation.split(q.interval)?, &mut fill)?;
    let (modular, [modular0, modular1]) = coordinate::issue_modular(
        &source,
        &mut uses,
        masks[7].take().ok_or(Error::Consumed)?,
        interval,
        reservation.split(q.modular)?,
        &mut fill,
    )?;
    let (mut coordinate, [coordinate0, coordinate1]) = coordinate::issue_coordinate(
        &modular,
        &mut uses,
        reservation.split(q.coordinate)?,
        &mut fill,
    )?;
    masks[13] = Some(coordinate.take_index()?);
    let loan = Norm13MaskLoan::take(&source, &mut uses, &masks)?;
    let (inverse, [inverse0, inverse1]) =
        norm13::issue_reserved_with(loan, reservation.split(q.inverse)?, &mut fill)?;
    masks[14] = Some(ProducedMaskFor {
        backing: MaskBacking::Inverse(inverse),
        binding: bindings[14],
        source: Arc::clone(&source),
        lease: wrappers.split(size_of::<ProducedMaskFor<C>>())?,
    });
    let mut broadcast = vector(4, &scratch_lease)?;
    broadcast.fill(get(&masks, 14)?.backing.values()[0]);
    uses.burn(17, bindings[5], 0)?;
    uses.burn(18, bindings[14], 0)?;
    let (normalize, [normalize0, normalize1]) = product::issue_reserved(
        product_context(&source, 17, 18, 15),
        get(&masks, 5)?.backing.values(),
        &broadcast,
        reservation.split(q.normalize)?,
        &mut fill,
    )?;
    masks[15] = Some(original_mask(
        normalize,
        bindings[15],
        &source,
        &mut wrappers,
    )?);
    uses.burn(19, bindings[15], 0)?;
    let (floor, [floor0, floor1]) = arithmetic::issue_reserved_with(
        gap(4),
        arithmetic_context(&source, 15, 16),
        get(&masks, 15)?.backing.values(),
        reservation.split(q.floor)?,
        &mut fill,
    )?;
    masks[16] = Some(ProducedMaskFor {
        backing: MaskBacking::Arithmetic(floor),
        binding: bindings[16],
        source: Arc::clone(&source),
        lease: wrappers.split(size_of::<ProducedMaskFor<C>>())?,
    });
    uses.burn(20, bindings[16], 0)?;
    uses.burn(21, bindings[1], 0)?;
    let (affine, [affine0, affine1]) = product::issue_reserved(
        product_context(&source, 20, 21, 17),
        get(&masks, 16)?.backing.values(),
        get(&masks, 1)?.backing.values(),
        reservation.split(q.affine)?,
        &mut fill,
    )?;
    masks[17] = Some(original_mask(affine, bindings[17], &source, &mut wrappers)?);
    uses.burn(22, bindings[17], 0)?;
    uses.burn(23, bindings[2], 0)?;
    let as_original = |slot| -> Result<&DealerMaskFor<C, Ring>, Error> {
        match &get(&masks, slot)?.backing {
            MaskBacking::Original(v) => Ok(v),
            _ => Err(Error::Context),
        }
    };
    let (terminal_mask, [terminal0, terminal1]) = gapars_terminal::issue_retained_with(
        &mut dealer,
        m.operation,
        as_original(17)?,
        as_original(2)?,
        m.output,
        m.operation,
        reservation.split(q.terminal)?,
        &mut fill,
    )?;
    if uses.next != USES as u8 || reservation.bytes != 0 {
        return Err(Error::Resource);
    }
    if issuance_work.remaining_entropy != 0 {
        return Err(Error::Resource);
    }
    // Unused wrapper slots/constructor slack are retained, never refunded while
    // any root/mask/role/source payload can remain live.
    keeper_lease.budget.budget().allocation()?;
    let material = Arc::new(RowMaterial {
        source: Arc::clone(&source),
        masks,
        modular,
        coordinate,
        terminal_mask,
        dealer,
        bindings,
        leases: [keeper_lease, wrappers, scratch_lease, table_lease],
        issuance_work,
    });
    let [v0, v1] = values;
    let make = |role,
                values,
                mean,
                square,
                interval,
                modular,
                coordinate,
                inverse,
                normalize,
                floor,
                affine,
                terminal,
                runtime_lease: LeaseFor<C>,
                lease: LeaseFor<C>|
     -> Result<Arc<RoleSlotFor<C, InitializerState<C>>>, Error> {
        let broadcast = vector(4, &lease)?;
        let local_buffers = [
            vector(1, &runtime_lease)?,
            vector(4, &runtime_lease)?,
            vector(1, &runtime_lease)?,
        ];
        lease.budget.budget().allocation()?;
        Ok(Arc::new(RoleSlotFor {
            state: Mutex::new(InitializerState {
                values,
                keeper: Arc::clone(&material),
                ledger: ledger(&source, &bindings, role),
                mean,
                square,
                interval,
                modular,
                coordinate,
                inverse,
                normalize,
                floor,
                affine,
                terminal: Some(terminal),
                frames: std::array::from_fn(|_| None),
                share: None,
                phase: 0,
                broadcast,
                work: q.work,
                #[cfg(test)]
                retirement_probe: RetirementProbe {
                    material: Arc::downgrade(&material),
                    observed: None,
                },
                local_buffers,
                local_payloads: [None, None, None],
                runtime_lease,
            }),
            _lease: lease,
        }))
    };
    let a = make(
        0,
        v0,
        mean0,
        square0,
        interval0,
        modular0,
        coordinate0,
        inverse0,
        normalize0,
        floor0,
        affine0,
        terminal0,
        runtime_left,
        left_lease,
    )?;
    let b = make(
        1,
        v1,
        mean1,
        square1,
        interval1,
        modular1,
        coordinate1,
        inverse1,
        normalize1,
        floor1,
        affine1,
        terminal1,
        runtime_right,
        right_lease,
    )?;
    coord_lease.budget.budget().allocation()?;
    let coordination = Arc::new(CoordinationFor {
        control: Mutex::new(SplitControl::default()),
        slots: [a, b],
        _original_run: None,
        _additional: coord_lease,
    });
    Ok(IssuedRow {
        material,
        parties: [
            PartyInitializer {
                coordination: Arc::clone(&coordination),
                role: 0,
                handed_off: false,
            },
            PartyInitializer {
                coordination,
                role: 1,
                handed_off: false,
            },
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn spec() -> LocalClearRootSpec4<'static> {
        LocalClearRootSpec4::new(&[-1, 0, 0, 0], &[-4096, -1, 1, 4095], &[4095, -4096, 1, -1])
            .unwrap()
    }
    #[test]
    fn every_child_is_issued_under_exact_original_reservation() {
        let q = resources::<OwnedBudgetContext>().unwrap();
        println!(
            "D43 row issuer total_bytes={} uses={} factories={}",
            q.total_bytes, q.work.consumer_uses, q.work.native_factories
        );
        println!(
            "work={:?}; included original interval LESS/AES allowance={} bytes",
            q.work, q.included_original_lookup_workspace_bytes
        );
        let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let before = budget.budget().used_bytes();
        let blocker = budget
            .reserve_native(MAX_ALLOCATION_BYTES - before - q.total_bytes)
            .unwrap();
        let mut draws = 0;
        let row = issue_reserved_with(
            spec(),
            [71; 32],
            budget.reserve_native(q.total_bytes).unwrap(),
            |bytes| {
                draws += 1;
                assert_eq!(budget.budget().used_bytes(), MAX_ALLOCATION_BYTES);
                getrandom::fill(bytes).map_err(|_| Error::Randomness)
            },
        )
        .unwrap();
        assert!(draws > 0);
        assert_eq!(row.material.issuance_work.remaining_entropy, 0);
        assert_eq!(
            row.material.issuance_work.spent_entropy,
            q.work.entropy_bytes
        );
        assert_eq!(
            budget.budget().used_bytes(),
            MAX_ALLOCATION_BYTES - q.clear_copy
        );
        assert_eq!(row.material.source.table.values().len(), 8192);
        for role in 0..2 {
            let c = &row.parties[role].coordination;
            assert!(!c.control().aborted);
            let state = c.slots[role].state.lock().unwrap();
            assert_eq!(state.ledger.next, 0);
            assert_eq!(state.ledger.role, role as u8);
            assert!(Arc::ptr_eq(&state.keeper, &row.material));
            assert_eq!(state.work.consumer_uses, 24);
            assert_eq!(state.frames.len(), 17);
            for slot in 0..3 {
                assert_eq!(
                    state.values[slot].as_ref().unwrap().binding.namespace,
                    row.material.bindings[slot].namespace
                );
            }
        }
        drop(row);
        drop(blocker);
        assert_eq!(budget.budget().used_bytes(), before);
    }
    #[test]
    fn one_short_refuses_before_any_child_entropy() {
        let q = resources::<OwnedBudgetContext>().unwrap();
        let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let before = budget.budget().used_bytes();
        let mut draws = 0;
        assert!(matches!(
            issue_reserved_with(
                spec(),
                [1; 32],
                budget.reserve_native(q.total_bytes - 1).unwrap(),
                |_| {
                    draws += 1;
                    Ok(())
                }
            ),
            Err(Error::Resource)
        ));
        assert_eq!(draws, 0);
        assert_eq!(budget.budget().used_bytes(), before);
    }
    #[test]
    fn partial_issuance_entropy_failure_erases_and_releases() {
        let q = resources::<OwnedBudgetContext>().unwrap();
        for fail in [0, 4, 15, 40] {
            let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
            let before = budget.budget().used_bytes();
            let mut draws = 0;
            let result = issue_reserved_with(
                spec(),
                [2; 32],
                budget.reserve_native(q.total_bytes).unwrap(),
                |bytes| {
                    let n = draws;
                    draws += 1;
                    if n == fail {
                        return Err(Error::Randomness);
                    }
                    getrandom::fill(bytes).map_err(|_| Error::Randomness)
                },
            );
            assert!(matches!(result, Err(Error::Randomness)), "fail={fail}");
            assert_eq!(budget.budget().used_bytes(), before);
        }
    }
    #[test]
    fn root_domain_does_not_adopt_default_gamma4096() {
        assert!(matches!(
            LocalClearRootSpec4::new(&[0; 4], &[4096; 4], &[0; 4]),
            Err(Error::Domain)
        ));
        assert!(LocalClearRootSpec4::new(&[-4096; 4], &[4095; 4], &[-4096; 4]).is_ok());
    }
    #[test]
    fn initializer_drop_burns_sibling_and_keeper_covers_last_initializer() {
        let q = resources::<OwnedBudgetContext>().unwrap();
        let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let before = budget.budget().used_bytes();
        let IssuedRow {
            material,
            parties: [a, b],
        } = issue_reserved_with(
            spec(),
            [5; 32],
            budget.reserve_native(q.total_bytes).unwrap(),
            |bytes| getrandom::fill(bytes).map_err(|_| Error::Randomness),
        )
        .unwrap();
        drop(material);
        drop(a);
        assert!(b.coordination.control().aborted);
        for slot in &b.coordination.slots {
            let state = slot.state.lock().unwrap();
            assert_eq!(state.phase, u8::MAX);
            assert!(state.values.iter().all(Option::is_none));
        }
        assert!(budget.budget().used_bytes() > before + 65536);
        drop(b);
        assert_eq!(budget.budget().used_bytes(), before);
    }
    #[test]
    fn issuance_callback_unwind_erases_partial_roots_and_children() {
        let q = resources::<OwnedBudgetContext>().unwrap();
        let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let before = budget.budget().used_bytes();
        let mut calls = 0;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = issue_reserved_with(
                spec(),
                [6; 32],
                budget.reserve_native(q.total_bytes).unwrap(),
                |bytes| {
                    calls += 1;
                    if calls == 20 {
                        panic!("native entropy injection")
                    }
                    getrandom::fill(bytes).map_err(|_| Error::Randomness)
                },
            );
        }));
        assert!(result.is_err());
        assert_eq!(budget.budget().used_bytes(), before);
    }
    #[test]
    fn material_cover_survives_final_native_dependency_drop() {
        let q = resources::<OwnedBudgetContext>().unwrap();
        let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let before = budget.budget().used_bytes();
        let IssuedRow {
            material,
            parties: [a, b],
        } = issue_reserved_with(
            spec(),
            [83; 32],
            budget.reserve_native(q.total_bytes).unwrap(),
            |bytes| getrandom::fill(bytes).map_err(|_| Error::Randomness),
        )
        .unwrap();
        let observed = std::rc::Rc::new(std::cell::Cell::new(0));
        for slot in &a.coordination.slots {
            slot.state.lock().unwrap().retirement_probe.observed =
                Some(std::rc::Rc::clone(&observed));
        }
        drop(material);
        drop(a);
        assert_eq!(observed.get(), 0);
        drop(b);
        assert_eq!(observed.get(), 2);
        assert_eq!(budget.budget().used_bytes(), before);
    }
}
