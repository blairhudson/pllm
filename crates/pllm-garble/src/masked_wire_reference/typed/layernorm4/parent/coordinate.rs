//! Closed D43 S*u -> unsigned TR50,42 -> ring7/ring13 coordinate.
//! No signed/GapARS certificate exists on this path. Inputs are original owning
//! energy and paired-correction records; no scalar/mask/Binding importer.
use super::super::super::super::{
    arithmetic, elementwise_product as product, source_lookup::interval50 as interval,
};
use super::*;
use crate::shared_allocation::MAX_ALLOCATION_BYTES;
use std::alloc::Layout;
use std::sync::atomic::AtomicUsize;

fn add(values: &[usize]) -> Result<usize, Error> {
    values
        .iter()
        .try_fold(0usize, |a, b| a.checked_add(*b).ok_or(Error::Resource))
}
fn arc_bytes<T>() -> Result<usize, Error> {
    Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<T>())
        .map_err(|_| Error::Resource)?
        .0
        .pad_to_align()
        .size()
        .checked_add(64)
        .ok_or(Error::Resource)
}
fn same_budget<C: AllocationContext>(lease: &LeaseFor<C>, source: &SourceFor<C>) -> bool {
    std::ptr::eq(lease.budget.budget(), source.lease.budget.budget())
}
fn tr_spec() -> arithmetic::Spec {
    arithmetic::Spec {
        bits: 50,
        lanes: 1,
        gate: arithmetic::Gate::Tr { shift: 42 },
    }
}
fn binding_at<C: AllocationContext>(
    source: &SourceFor<C>,
    ledger: &internal_use::Ledger<USES>,
    use_index: usize,
    slot: usize,
    namespace: [u8; 32],
) -> Result<Binding, Error> {
    let binding = ledger.bindings[use_index];
    if binding.descriptor != source.descriptors[slot]
        || binding.namespace != namespace
        || ledger.uses[use_index] != source.uses[use_index]
    {
        return Err(Error::Context);
    }
    Ok(binding)
}

// Original masks and both correction residuals are retained by MOVE here,
// through all product/TR frames and outputs. No native body is extracted.
struct ModularCreation<C: AllocationContext> {
    native: product::MasksFor<C>,
    energy: ProducedMaskFor<C>,
    interval: interval::MasksFor<C>,
    source: Arc<SourceFor<C>>,
    output: Binding,
    #[allow(dead_code)]
    // Immutable declaration/source allocation cover, retained through final borrower.
    lease: LeaseFor<C>,
}
struct ModularInputs<C: AllocationContext> {
    energy: ProducedValueFor<C>,
    interval: interval::ValuesFor<C>,
    #[allow(dead_code)] // Covers original moved inputs until last frame/value.
    lease: LeaseFor<C>,
}
pub(super) struct ModularMask<C: AllocationContext> {
    creation: Arc<ModularCreation<C>>,
}
pub(super) struct ModularParty<C: AllocationContext> {
    native: product::PartyFor<C>,
    creation: Arc<ModularCreation<C>>,
    inputs: Option<Arc<ModularInputs<C>>>,
    role: u8,
    phase: u8,
    lease: LeaseFor<C>,
}
#[allow(dead_code)] // Native frame plus creation/input leases are deliberately retained owners.
pub(super) struct ModularFrame<C: AllocationContext> {
    native: product::FrameFor<C>,
    creation: Arc<ModularCreation<C>>,
    inputs: Arc<ModularInputs<C>>,
    lease: LeaseFor<C>,
}
pub(super) struct ModularValue<C: AllocationContext> {
    native: product::OutputFor<C>,
    creation: Arc<ModularCreation<C>>,
    inputs: Arc<ModularInputs<C>>,
    role: u8,
    #[allow(dead_code)] // Actual immutable wrapper allocation cover.
    lease: LeaseFor<C>,
}
#[allow(dead_code)] // All original TR/material/structural bindings retained to outer retirement.
pub(super) struct CoordinateMask<C: AllocationContext> {
    index: Option<ProducedMaskFor<C>>,
    native: arithmetic::OutputMasksFor<C>,
    modular: Arc<ModularCreation<C>>,
    quotient: Binding,
    mantissa: Binding,
    lease: LeaseFor<C>,
}
pub(super) struct CoordinateParty<C: AllocationContext> {
    native: arithmetic::PartyFor<C>,
    modular: Arc<ModularCreation<C>>,
    input: Option<ModularValue<C>>,
    destination: Zeroizing<Vec<u64>>,
    bindings: [Binding; 3],
    role: u8,
    phase: u8,
    lease: LeaseFor<C>,
    tr_output: Option<arithmetic::MaskedOutputFor<C>>,
}
#[allow(clippy::large_enum_variant)] // Original inline producer layout is prequoted.
pub(super) enum CoordinateStep<C: AllocationContext> {
    Frame(arithmetic::FrameFor<C>),
    Index(ProducedValueFor<C>),
}
impl<C: AllocationContext> CoordinateMask<C> {
    pub(super) fn take_index(&mut self) -> Result<ProducedMaskFor<C>, Error> {
        self.index.take().ok_or(Error::Consumed)
    }
}

#[allow(dead_code)] // Simultaneous issuance/move-result layout slots priced by resources.
struct ProductIssueControls<C: AllocationContext> {
    energy: ProducedMaskFor<C>,
    interval: interval::MasksFor<C>,
    context: product::Context,
    query: product::Resources,
    native_result: Result<(product::MasksFor<C>, [product::PartyFor<C>; 2]), Error>,
    declaration: ModularCreation<C>,
    arc: Arc<ModularCreation<C>>,
    result: Result<(ModularMask<C>, [ModularParty<C>; 2]), Error>,
    splits: [Result<LeaseFor<C>, Error>; 5],
}
#[allow(dead_code)] // Real evaluator guard/return/move layout, not synthetic execution.
struct ProductEvaluatorControls<'a, C: AllocationContext> {
    guard: ProductAbort<'a, C>,
    energy: ProducedValueFor<C>,
    interval: interval::ValuesFor<C>,
    inputs: ModularInputs<C>,
    arc: Arc<ModularInputs<C>>,
    native_frame: Result<product::FrameFor<C>, Error>,
    native_output: Result<product::OutputFor<C>, Error>,
    begin_result: Result<ModularFrame<C>, Error>,
    finish_result: Result<ModularValue<C>, Error>,
    ledger: &'a mut internal_use::Ledger<USES>,
    split: Result<LeaseFor<C>, Error>,
}
#[allow(dead_code)] // Concrete native TR issuance slots included before factories.
struct CoordinateIssueControls<C: AllocationContext> {
    context: arithmetic::Context,
    query: arithmetic::Resources,
    native: Result<(arithmetic::OutputMasksFor<C>, [arithmetic::PartyFor<C>; 2]), Error>,
    mask: CoordinateMask<C>,
    parties: [CoordinateParty<C>; 2],
    vectors: [Zeroizing<Vec<u64>>; 3],
    splits: [Result<LeaseFor<C>, Error>; 5],
    result: Result<(CoordinateMask<C>, [CoordinateParty<C>; 2]), Error>,
}
#[allow(dead_code)] // Concrete guarded TR/coordinate result slots included per party.
struct CoordinateEvaluatorControls<'a, C: AllocationContext> {
    guard: CoordinateAbort<'a, C>,
    input: ModularValue<C>,
    native_result: Result<arithmetic::StepFor<C>, Error>,
    begin_result: Result<arithmetic::FrameFor<C>, Error>,
    advance_result: Result<CoordinateStep<C>, Error>,
    output: ProducedValueFor<C>,
    ledger: &'a mut internal_use::Ledger<USES>,
}
pub(super) struct Resources {
    pub product_bytes: usize,
    pub coordinate_bytes: usize,
    product_party: usize,
    product_dealer: usize,
    coordinate_party: usize,
    coordinate_dealer: usize,
}

// Candidate additive upper-bound storage, NOT runtime owners or a separated ABI.
// No resource query, issuance path or evaluator constructs/references these records.
pub(in crate::masked_wire_reference::typed::layernorm4) mod retained_layout {
    use super::*;
    use crate::masked_wire_reference::typed::layernorm4::retained_layout::{
        ProducedMaskLayout, ProducedValueLayout,
    };

    // FUTURE order only: validate the enlarged original aggregate; split and
    // establish incremental outer anchor FIRST, original modular/dealer-cover
    // outer anchor SECOND, covered secret locals AFTER both. Both outer anchors
    // stay UNMOVED through errors/unwind; a late cell shares both, never supplies
    // sole early cover. Existing equality/second dealer split needs later repair.
    // No construction, cloning, splitting or cleanup is implemented here.
    pub(in crate::masked_wire_reference::typed::layernorm4) struct LateCell<C: AllocationContext> {
        source: Arc<SourceFor<C>>,
        declaration: interval::DeclarationHandleFor<C>,
        energy: Binding,
        output: Binding,
        original_cover: Arc<LeaseFor<C>>,
        incremental_cover: Arc<LeaseFor<C>>,
    }

    // Normal completion field order only, not erasure on panic/abort or physical
    // through-deallocation coverage. No arbitrary-B constructor or methods.
    pub(in crate::masked_wire_reference::typed::layernorm4) struct Retained<B, C: AllocationContext, R> {
        backing: B,
        retention: R,
        control: LeaseFor<C>,
    }
    pub(in crate::masked_wire_reference::typed::layernorm4) type RetainedFrame<C> =
        Retained<arithmetic::FrameFor<C>, C, Arc<LateCell<C>>>;
    pub(in crate::masked_wire_reference::typed::layernorm4) type RetainedOutput<C> =
        Retained<arithmetic::MaskedOutputFor<C>, C, Arc<LateCell<C>>>;
    pub(in crate::masked_wire_reference::typed::layernorm4) type RetainedOutputMasks<C> =
        Retained<arithmetic::OutputMasksFor<C>, C, Arc<LateCell<C>>>;
    pub(in crate::masked_wire_reference::typed::layernorm4) type RetainedLocal<C> =
        Retained<Zeroizing<Vec<u64>>, C, Arc<LateCell<C>>>;

    // Each original owner occurs once in its inventory. Arc fields name separate
    // pointee storage, not another inline body/allocation. Creation is the dealer
    // attachment; ModularMask's creation handle does not add a second cell edge.
    pub(in crate::masked_wire_reference::typed::layernorm4::parent) struct ModularCreationLayout<C: AllocationContext> {
        original: ModularCreation<C>,
        retention: Arc<LateCell<C>>,
        control: LeaseFor<C>,
    }
    pub(in crate::masked_wire_reference::typed::layernorm4::parent) struct ModularMaskLayout<C: AllocationContext> {
        original: ModularMask<C>,
        creation: Arc<ModularCreationLayout<C>>,
    }
    pub(in crate::masked_wire_reference::typed::layernorm4::parent) struct ModularInputsLayout<C: AllocationContext> {
        original: ModularInputs<C>,
        retention: Arc<LateCell<C>>,
        control: LeaseFor<C>,
    }
    pub(in crate::masked_wire_reference::typed::layernorm4::parent) struct ModularPartyLayout<C: AllocationContext> {
        original: ModularParty<C>,
        retention: Arc<LateCell<C>>,
        control: LeaseFor<C>,
        inputs: Option<Arc<ModularInputsLayout<C>>>,
    }
    pub(in crate::masked_wire_reference::typed::layernorm4::parent) struct ModularFrameLayout<C: AllocationContext> {
        original: ModularFrame<C>,
        inputs: Arc<ModularInputsLayout<C>>,
        retention: Arc<LateCell<C>>,
        control: LeaseFor<C>,
    }
    pub(in crate::masked_wire_reference::typed::layernorm4::parent) struct ModularValueLayout<C: AllocationContext> {
        original: ModularValue<C>,
        inputs: Arc<ModularInputsLayout<C>>,
        retention: Arc<LateCell<C>>,
        control: LeaseFor<C>,
    }
    pub(in crate::masked_wire_reference::typed::layernorm4::parent) struct CoordinateMaskLayout<C: AllocationContext> {
        original: CoordinateMask<C>,
        retention: Arc<LateCell<C>>,
        control: LeaseFor<C>,
        dealer_index: Option<ProducedMaskLayout<C>>,
    }
    pub(in crate::masked_wire_reference::typed::layernorm4::parent) struct CoordinatePartyLayout<C: AllocationContext> {
        original: CoordinateParty<C>,
        input: Option<ModularValueLayout<C>>,
        retained_tr: Option<RetainedOutput<C>>,
        retention: Arc<LateCell<C>>,
        control: LeaseFor<C>,
    }
    pub(in crate::masked_wire_reference::typed::layernorm4::parent) enum CoordinateStepLayout<C: AllocationContext> {
        Frame(RetainedFrame<C>),
        Index(ProducedValueLayout<C>),
    }

    // Controls are named storage inventories, not coexistent runtime aggregates.
    // Existing five split Results remain in original. Extra slots below model
    // only the distinct future early-anchor sequence; no hidden split reuse.
    pub(in crate::masked_wire_reference::typed::layernorm4::parent) struct ProductIssueLayout<C: AllocationContext> {
        original: ProductIssueControls<C>,
        incremental_split: Result<LeaseFor<C>, Error>,
        modular_aggregate_split: Result<LeaseFor<C>, Error>,
        original_dealer_split: Result<LeaseFor<C>, Error>,
        incremental_anchor: Arc<LeaseFor<C>>,
        original_anchor: Arc<LeaseFor<C>>,
        incremental_anchor_move: Result<Arc<LeaseFor<C>>, Error>,
        original_anchor_move: Result<Arc<LeaseFor<C>>, Error>,
        declaration_clone: interval::DeclarationHandleFor<C>,
        cell_inline: LateCell<C>,
        cell_result: Result<Arc<LateCell<C>>, Error>,
        cell_move: Arc<LateCell<C>>,
        creation: ModularCreationLayout<C>,
        creation_result: Result<Arc<ModularCreationLayout<C>>, Error>,
        mask: ModularMaskLayout<C>,
        parties: [ModularPartyLayout<C>; 2],
        result: Result<(ModularMaskLayout<C>, [ModularPartyLayout<C>; 2]), Error>,
    }
    pub(in crate::masked_wire_reference::typed::layernorm4::parent) struct CoordinateIssueLayout<C: AllocationContext> {
        original: CoordinateIssueControls<C>,
        mask: CoordinateMaskLayout<C>,
        parties: [CoordinatePartyLayout<C>; 2],
        index: Option<ProducedMaskLayout<C>>,
        index_move: Result<ProducedMaskLayout<C>, Error>,
        result: Result<(CoordinateMaskLayout<C>, [CoordinatePartyLayout<C>; 2]), Error>,
    }
    // Begin and finish are different calls; their Result/move slots are explicit
    // alternatives for later peak review. Native/input owners stay in original.
    pub(in crate::masked_wire_reference::typed::layernorm4::parent) struct ProductEvaluatorLayout<'a, C: AllocationContext> {
        original: ProductEvaluatorControls<'a, C>,
        inputs: ModularInputsLayout<C>,
        inputs_arc: Arc<ModularInputsLayout<C>>,
        inputs_result: Result<Arc<ModularInputsLayout<C>>, Error>,
        begin: Result<ModularFrameLayout<C>, Error>,
        frame_move: ModularFrameLayout<C>,
        finish: Result<ModularValueLayout<C>, Error>,
        value_move: ModularValueLayout<C>,
    }
    pub(in crate::masked_wire_reference::typed::layernorm4::parent) struct CoordinateEvaluatorLayout<'a, C: AllocationContext> {
        original: CoordinateEvaluatorControls<'a, C>,
        input: ModularValueLayout<C>,
        begin: Result<RetainedFrame<C>, Error>,
        frame_move: RetainedFrame<C>,
        step: Result<CoordinateStepLayout<C>, Error>,
        retained_tr: Option<RetainedOutput<C>>,
        index_split: Result<LeaseFor<C>, Error>,
        index: Result<ProducedValueLayout<C>, Error>,
        index_move: ProducedValueLayout<C>,
    }
    // TR is installed before the fallible index split. Cancellation does not
    // retire inputs/TR. Those owners can overlap inverse/failure controls.
}

pub(super) fn resources<C: AllocationContext>() -> Result<Resources, Error> {
    let p = product::resources_for::<C>(1)?;
    let t = arithmetic::resources_for_context::<C>(tr_spec())?;
    let product_dealer = add(&[
        arc_bytes::<ModularCreation<C>>()?,
        size_of::<ProductIssueControls<C>>(),
        size_of::<ModularMask<C>>(),
        C::TRANSIENT_LINK_BYTES,
    ])?;
    let product_party = add(&[
        size_of::<ModularParty<C>>(),
        arc_bytes::<ModularInputs<C>>()?,
        size_of::<ProductEvaluatorControls<'_, C>>(),
        size_of::<ModularFrame<C>>(),
        size_of::<ModularValue<C>>(),
        C::TRANSIENT_LINK_BYTES,
    ])?;
    let coordinate_dealer = add(&[
        size_of::<CoordinateMask<C>>(),
        size_of::<CoordinateIssueControls<C>>(),
        8 + 64,
        C::TRANSIENT_LINK_BYTES,
    ])?;
    let coordinate_party = add(&[
        size_of::<CoordinateParty<C>>(),
        size_of::<CoordinateEvaluatorControls<'_, C>>(),
        size_of::<ProducedValueFor<C>>(),
        8 + 64,
        C::TRANSIENT_LINK_BYTES,
    ])?;
    let product_bytes = add(&[
        p.combined_bytes,
        product_dealer,
        product_party.checked_mul(2).ok_or(Error::Resource)?,
    ])?;
    let coordinate_bytes = add(&[
        t.combined_reservation_bytes,
        coordinate_dealer,
        coordinate_party.checked_mul(2).ok_or(Error::Resource)?,
    ])?;
    if product_bytes > MAX_ALLOCATION_BYTES || coordinate_bytes > MAX_ALLOCATION_BYTES {
        return Err(Error::Resource);
    }
    Ok(Resources {
        product_bytes,
        coordinate_bytes,
        product_party,
        product_dealer,
        coordinate_party,
        coordinate_dealer,
    })
}

pub(super) fn issue_modular<C: AllocationContext>(
    source: &Arc<SourceFor<C>>,
    ledger: &mut internal_use::Ledger<USES>,
    energy: ProducedMaskFor<C>,
    interval: interval::MasksFor<C>,
    mut lease: LeaseFor<C>,
    fill: impl FnMut(&mut [u8]) -> Result<(), Error>,
) -> Result<(ModularMask<C>, [ModularParty<C>; 2]), Error> {
    ledger.burn(10, energy.binding, ledger.role)?;
    let u = binding_at(source, ledger, 11, 9, energy.binding.namespace)?;
    ledger.burn(11, u, ledger.role)?;
    let output = binding_at(source, ledger, 12, 10, energy.binding.namespace)?;
    let q = resources::<C>()?;
    if lease.bytes != q.product_bytes
        || !same_budget(&lease, source)
        || !Arc::ptr_eq(source, &energy.source)
        || !interval.source_is(source)
        || energy.binding.descriptor != source.descriptors[7]
        || source.kinds[7]
            != (ValueKind::SignedFixed {
                bits: 50,
                fraction: 24,
            })
        || source.kinds[9] != (ValueKind::UnsignedResidue { bits: 50 })
        || source.kinds[10] != (ValueKind::UnsignedResidue { bits: 50 })
        || energy.backing.values().len() != 1
    {
        return Err(Error::Context);
    }
    let left = lease.split(q.product_party)?;
    let right = lease.split(q.product_party)?;
    let creation_lease = lease.split(q.product_dealer)?;
    let p = product::resources_for::<C>(1)?;
    let context = product::Context {
        region: source.profile,
        stage: digest(&[
            b"pllm.sigma.layernorm4.mod50-tr42-coordinate.v1",
            &source.operation,
        ]),
        inputs: [
            source.descriptors[7].digest(),
            source.descriptors[9].digest(),
        ],
        output: source.descriptors[10].digest(),
        first: source.descriptors[10].identity().first_tensor_index(),
    };
    let (native, [a, b]) = product::issue_reserved(
        context,
        energy.backing.values(),
        interval.output(1)?,
        lease.split(p.combined_bytes)?,
        fill,
    )?;
    // All original input masks/native correction owners remain inside the SAME
    // creation record. Binding is captured here, not adopted from a later value.
    lease.budget.budget().allocation()?;
    let creation = Arc::new(ModularCreation {
        native,
        energy,
        interval,
        source: Arc::clone(source),
        output,
        lease: creation_lease,
    });
    // No remaining capacity credit: retain construction reservation in creation.
    if lease.bytes != 0 {
        return Err(Error::Resource);
    }
    let make = |native, role, lease| ModularParty {
        native,
        creation: Arc::clone(&creation),
        inputs: None,
        role,
        phase: 0,
        lease,
    };
    Ok((
        ModularMask {
            creation: Arc::clone(&creation),
        },
        [make(a, 0, left), make(b, 1, right)],
    ))
}
struct ProductAbort<'a, C: AllocationContext> {
    party: &'a mut ModularParty<C>,
    armed: bool,
}
impl<C: AllocationContext> Drop for ProductAbort<'_, C> {
    fn drop(&mut self) {
        if self.armed {
            self.party.cancel();
        }
    }
}
impl<C: AllocationContext> ModularParty<C> {
    pub(super) fn cancel(&mut self) {
        self.phase = u8::MAX;
        self.native.cancel();
    }
    pub(super) fn begin(
        &mut self,
        ledger: &mut internal_use::Ledger<USES>,
        energy: ProducedValueFor<C>,
        interval: interval::ValuesFor<C>,
    ) -> Result<ModularFrame<C>, Error> {
        let mut guard = ProductAbort {
            party: self,
            armed: true,
        };
        let p = &mut *guard.party;
        ledger.burn(10, energy.binding, p.role)?;
        let u = binding_at(
            &p.creation.source,
            ledger,
            11,
            9,
            p.creation.energy.binding.namespace,
        )?;
        ledger.burn(11, u, p.role)?;
        if p.phase != 0
            || energy.role != p.role
            || energy.binding != p.creation.energy.binding
            || !Arc::ptr_eq(&energy.source, &p.creation.source)
            || !interval.is_for(&p.creation.interval, p.role)
        {
            return Err(Error::Context);
        }
        p.phase = 1;
        let lease = p.lease.split(arc_bytes::<ModularInputs<C>>()?)?;
        p.lease.budget.budget().allocation()?;
        let inputs = Arc::new(ModularInputs {
            energy,
            interval,
            lease,
        });
        let frame = p.native.begin(
            inputs.energy.backing.values(),
            inputs.interval.output(&p.creation.source, 1)?.values(),
        )?;
        let lease = p.lease.split(size_of::<ModularFrame<C>>())?;
        p.inputs = Some(Arc::clone(&inputs));
        let output = ModularFrame {
            native: frame,
            creation: Arc::clone(&p.creation),
            inputs,
            lease,
        };
        guard.armed = false;
        Ok(output)
    }
    pub(super) fn finish(&mut self, peer: &[u8]) -> Result<ModularValue<C>, Error> {
        let mut guard = ProductAbort {
            party: self,
            armed: true,
        };
        let p = &mut *guard.party;
        if p.phase != 1 {
            return Err(Error::Consumed);
        }
        let native = p.native.finish(peer)?;
        p.phase = 2;
        let inputs = p.inputs.take().ok_or(Error::Consumed)?;
        let output = ModularValue {
            native,
            creation: Arc::clone(&p.creation),
            inputs,
            role: p.role,
            lease: p.lease.split(size_of::<ModularValue<C>>())?,
        };
        guard.armed = false;
        Ok(output)
    }
}
impl<C: AllocationContext> ModularFrame<C> {
    pub(super) fn as_bytes(&self) -> &[u8] {
        self.native.as_bytes()
    }
}

pub(super) fn issue_coordinate<C: AllocationContext>(
    mask: &ModularMask<C>,
    ledger: &mut internal_use::Ledger<USES>,
    mut lease: LeaseFor<C>,
    fill: impl FnMut(&mut [u8]) -> Result<(), Error>,
) -> Result<(CoordinateMask<C>, [CoordinateParty<C>; 2]), Error> {
    let source = &mask.creation.source;
    ledger.burn(12, mask.creation.output, ledger.role)?;
    let quotient = binding_at(source, ledger, 13, 11, mask.creation.output.namespace)?;
    let mantissa = binding_at(source, ledger, 14, 12, mask.creation.output.namespace)?;
    let exponent = binding_at(source, ledger, 15, 8, mask.creation.output.namespace)?;
    // Private local reductions are precommitted once; native runtime output,
    // never a supplied unsigned tuple, will supply their actual input.
    ledger.burn(13, quotient, ledger.role)?;
    ledger.burn(14, mantissa, ledger.role)?;
    ledger.burn(15, exponent, ledger.role)?;
    let index = binding_at(source, ledger, 16, 13, mask.creation.output.namespace)?;
    let q = resources::<C>()?;
    if lease.bytes != q.coordinate_bytes
        || !same_budget(&lease, source)
        || source.kinds[10] != (ValueKind::UnsignedResidue { bits: 50 })
        || source.kinds[11] != (ValueKind::UnsignedResidue { bits: 8 })
        || source.kinds[12] != (ValueKind::UnsignedResidue { bits: 7 })
        || source.kinds[13] != (ValueKind::UnsignedResidue { bits: 13 })
    {
        return Err(Error::Context);
    }
    let mut left = lease.split(q.coordinate_party)?;
    let mut right = lease.split(q.coordinate_party)?;
    let mut dealer = lease.split(q.coordinate_dealer)?;
    // Native TR accepts the unsigned modular residue even when bit49==1.
    // There is deliberately no signed-positive or quarter-ring promise.
    let context = arithmetic::Context {
        plan: source.descriptors[10].identity().plan(),
        session: source.session,
        operation: digest(&[
            b"pllm.sigma.layernorm4.mod50-tr42-coordinate.v1",
            &source.profile,
        ]),
        input_wire: source.descriptors[10].digest(),
        output_wire: source.descriptors[11].digest(),
        first_tensor_index: source.descriptors[11].identity().first_tensor_index(),
    };
    let native_q = arithmetic::resources_for_context::<C>(tr_spec())?;
    let (native, [a, b]) = arithmetic::issue_reserved_with(
        tr_spec(),
        context,
        mask.creation.native.values(),
        lease.split(native_q.combined_reservation_bytes)?,
        fill,
    )?;
    if lease.bytes != 0 {
        return Err(Error::Resource);
    }
    let allocate = |owner: &mut LeaseFor<C>| -> Result<Zeroizing<Vec<u64>>, Error> {
        let mut v = crate::shared_allocation::allocate(1, owner.budget.budget())?;
        v.push(0);
        Ok(Zeroizing::new(v))
    };
    let mut index_mask = allocate(&mut dealer)?;
    index_mask[0] = ((native.values()[0] & 127) * 64 + mask.creation.interval.output(0)?[0]) & 8191;
    let va = allocate(&mut left)?;
    let vb = allocate(&mut right)?;
    let produced = ProducedMaskFor {
        backing: MaskBacking::Local(index_mask),
        binding: index,
        source: Arc::clone(source),
        lease: dealer,
    };
    let mask = CoordinateMask {
        index: Some(produced),
        native,
        modular: Arc::clone(&mask.creation),
        quotient,
        mantissa,
        lease,
    };
    let make = |native, destination, role, lease| CoordinateParty {
        native,
        modular: Arc::clone(&mask.modular),
        input: None,
        destination,
        bindings: [quotient, mantissa, index],
        role,
        phase: 0,
        lease,
        tr_output: None,
    };
    let parties = [make(a, va, 0, left), make(b, vb, 1, right)];
    Ok((mask, parties))
}
struct CoordinateAbort<'a, C: AllocationContext> {
    party: &'a mut CoordinateParty<C>,
    armed: bool,
}
impl<C: AllocationContext> Drop for CoordinateAbort<'_, C> {
    fn drop(&mut self) {
        if self.armed {
            self.party.cancel();
        }
    }
}
impl<C: AllocationContext> CoordinateParty<C> {
    pub(super) fn cancel(&mut self) {
        self.phase = u8::MAX;
        self.native.cancel();
    }
    pub(super) fn begin(
        &mut self,
        ledger: &mut internal_use::Ledger<USES>,
        input: ModularValue<C>,
    ) -> Result<arithmetic::FrameFor<C>, Error> {
        let mut guard = CoordinateAbort {
            party: self,
            armed: true,
        };
        let p = &mut *guard.party;
        ledger.burn(12, input.creation.output, p.role)?;
        if p.phase != 0 || input.role != p.role || !Arc::ptr_eq(&p.modular, &input.creation) {
            return Err(Error::Context);
        }
        let frame = p.native.begin(input.native.values())?;
        p.input = Some(input);
        p.phase = 1;
        guard.armed = false;
        Ok(frame)
    }
    pub(super) fn advance(
        &mut self,
        ledger: &mut internal_use::Ledger<USES>,
        peer: &[u8],
    ) -> Result<CoordinateStep<C>, Error> {
        let mut guard = CoordinateAbort {
            party: self,
            armed: true,
        };
        let p = &mut *guard.party;
        if !matches!(p.phase, 1 | 2) {
            return Err(Error::Consumed);
        }
        let output = match (p.native.advance(peer)?, p.phase) {
            (arithmetic::StepFor::Frame(frame), 1) => {
                p.phase = 2;
                CoordinateStep::Frame(frame)
            }
            (arithmetic::StepFor::Output(value), 2) => {
                ledger.burn(13, p.bindings[0], p.role)?;
                ledger.burn(14, p.bindings[1], p.role)?;
                let input = p.input.as_ref().ok_or(Error::Consumed)?;
                let exponent =
                    binding_at(&p.modular.source, ledger, 15, 8, p.modular.output.namespace)?;
                ledger.burn(15, exponent, p.role)?;
                if !input.inputs.interval.is_for(&p.modular.interval, p.role) {
                    return Err(Error::Context);
                }
                let e = input.inputs.interval.output(&p.modular.source, 0)?.values();
                p.destination[0] = ((value.values()[0] & 127) * 64 + e[0]) & 8191;
                // Original TR output stays owned, not discarded after reading.
                // CoordinateParty retains all upstream product/input custody.
                p.tr_output = Some(value);
                p.phase = 3;
                CoordinateStep::Index(ProducedValueFor {
                    backing: ValueBacking::Local(std::mem::replace(
                        &mut p.destination,
                        Zeroizing::new(Vec::new()),
                    )),
                    binding: p.bindings[2],
                    role: p.role,
                    source: Arc::clone(&p.modular.source),
                    lease: p.lease.split(size_of::<ProducedValueFor<C>>() + 8 + 64)?,
                })
            }
            _ => return Err(Error::Context),
        };
        guard.armed = false;
        Ok(output)
    }
}
