//! Actual D36 native dispatcher. Parent polls user cancellation outside BOTH
//! locks. Child calls are sealed internal calls and never receive user code.
use super::super::activation::{certified_exp_lookup as exp, certified_softmax_inverse as inverse};
use super::super::nexp::region::{self as child, inherited};
use super::super::{role_coordination as custody, softmax_coordinate as coordinate};
use super::*;
use crate::fixed_inverse_table;
use crate::masked_wire_reference::{elementwise_product as product, source_lookup as lookup};
use ledger::{Ledger, Read};
use std::rc::Rc;
use std::sync::Mutex;

pub(super) type Slot<C> = custody::RoleSlotFor<C, State<C>>;
pub(super) type Shared<C> = custody::CoordinationFor<C, State<C>>;

#[derive(Debug, PartialEq, Eq)]
pub struct BuildError(Failure);
#[derive(Debug, PartialEq, Eq)]
enum Failure {
    Native(Error),
    Exponential(exp::Error),
    Inverse(inverse::Error),
}
impl From<Error> for BuildError {
    fn from(e: Error) -> Self {
        Self(Failure::Native(e))
    }
}
impl From<exp::Error> for BuildError {
    fn from(e: exp::Error) -> Self {
        Self(Failure::Exponential(e))
    }
}
impl From<inverse::Error> for BuildError {
    fn from(e: inverse::Error) -> Self {
        Self(Failure::Inverse(e))
    }
}
impl BuildError {
    pub(super) fn native(&self) -> Option<Error> {
        match self.0 {
            Failure::Native(e) => Some(e),
            _ => None,
        }
    }
    pub(super) fn unresolved_floor(&self) -> Option<(u64, u64)> {
        match &self.0 {
            Failure::Exponential(exp::Error::Floor(crate::fixed_exp_tables::Error::Floor(
                crate::exp_floor::Error::Unresolved {
                    lower_floor,
                    upper_floor,
                },
            ))) => Some((*lower_floor, *upper_floor)),
            _ => None,
        }
    }
}
pub(super) fn native_inverse_error(_: inverse::Error) -> Error {
    Error::Domain
}
pub(super) fn native_exp_error(_: exp::Error) -> Error {
    Error::Domain
}

/// No dealer masks/keys in the evaluator registry. Root source and the child's
/// cancellation coordinator are shared; dealer material stays in Mask below.
pub(super) struct KeepaliveFor<C: AllocationContext> {
    source: Arc<SourceFor<C>>,
    child: Option<child::Coordinator<C>>,
    _verified: Option<fixed_inverse_table::Verified>,
    _lease: LeaseFor<C>,
}
#[derive(Clone, Copy)]
struct Bindings {
    root: Binding,
    maximum: Binding,
    scores: [Binding; 3],
    difference: Binding,
    exponent: Binding,
    sum: Binding,
    denominator: Binding,
    coordinate: Binding,
    reciprocal: Binding,
    product: Binding,
    _gap: Binding,
    output: Binding,
}
pub(super) struct State<C: AllocationContext> {
    maximum: Option<maximum::PartyFor<C>>,
    child: Option<child::Role<C>>,
    exponent: Option<inherited::EValueFor<C>>,
    coordinate: Option<coordinate::PartyFor<C>>,
    coordinate_value: Option<coordinate::OutputFor<C>>,
    inverse: Option<lookup::PartyFor<C>>,
    reciprocal: Option<lookup::MaskedOutputFor<C>>,
    product: Option<product::PartyFor<C>>,
    product_value: Option<product::OutputFor<C>>,
    gap: Option<arithmetic::PartyFor<C>>,
    output: Option<arithmetic::MaskedOutputFor<C>>,
    public: Option<[u64; 3]>,
    broadcast: Zeroizing<Box<[u64]>>,
    phase: u8,
    role: u8,
    bindings: Bindings,
    ledger: Option<Ledger>,
    keepalive: Arc<KeepaliveFor<C>>,
    lease: LeaseFor<C>,
    _affinity: PhantomData<Rc<()>>,
}
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum Phase {
    Start = 0,
    Maximum = 1,
    Child = 2,
    Coordinate = 3,
    Inverse = 4,
    Product = 5,
    Gap = 6,
    Ready = 7,
    Burnt = 8,
}

#[allow(clippy::large_enum_variant)]
enum Body<C: AllocationContext> {
    Maximum(maximum::Frame<C>),
    Child(child::Frame<C>),
    Arithmetic(arithmetic::FrameFor<C>),
    Lookup(lookup::FrameFor<C>),
    Product(product::FrameFor<C>),
}
impl<C: AllocationContext> Body<C> {
    fn bytes(&self) -> Result<&[u8], Error> {
        Ok(match self {
            Self::Maximum(f) => f.bytes(),
            Self::Child(f) => return f.bytes(),
            Self::Arithmetic(f) => f.as_bytes(),
            Self::Lookup(f) => f.as_bytes(),
            Self::Product(f) => f.as_bytes(),
        })
    }
}
pub struct Frame<C: AllocationContext> {
    native: Body<C>,
    shared: Arc<Shared<C>>,
}
pub struct Coordinator<C: AllocationContext> {
    shared: Arc<Shared<C>>,
}
pub struct Role<C: AllocationContext> {
    slot: Arc<Slot<C>>,
    shared: Arc<Shared<C>>,
    role: u8,
    handed_off: bool,
}
pub struct Output<C: AllocationContext> {
    slot: Arc<Slot<C>>,
    shared: Arc<Shared<C>>,
    role: u8,
}
pub struct Mask<C: AllocationContext> {
    #[allow(dead_code)] // Original material custody; these are not metadata-only masks.
    native: Option<arithmetic::OutputMasksFor<C>>,
    #[allow(dead_code)] // Original maximum material custody.
    maximum: Option<maximum::DealerFor<C>>,
    #[allow(dead_code)] // Original child mask must live through the outer mask.
    child: Option<child::Mask<C>>,
    #[allow(dead_code)] // Original coordinate material custody.
    coordinate: Option<coordinate::MaskFor<C>>,
    #[allow(dead_code)] // Original inverse material custody.
    inverse: Option<lookup::OutputMasksFor<C>>,
    #[allow(dead_code)] // Original fresh product material custody.
    product: Option<product::MasksFor<C>>,
    binding: Binding,
    shared: Arc<Shared<C>>,
    _lease: LeaseFor<C>,
}
pub type Created<C> = (Mask<C>, Coordinator<C>, [Role<C>; 2]);
#[allow(clippy::large_enum_variant)]
pub enum Step<C: AllocationContext> {
    Frame(Frame<C>),
    Ready,
}

impl<C: AllocationContext> State<C> {
    fn burn(&mut self, read: Read, binding: Binding) -> Result<(), Error> {
        self.ledger
            .as_mut()
            .ok_or(Error::Consumed)?
            .burn(read, binding, self.role)
    }
    fn begin(&mut self) -> Result<Option<Body<C>>, Error> {
        if self.phase != Phase::Start as u8 {
            return Err(Error::Consumed);
        }
        if self.public.is_some() {
            self.phase = Phase::Ready as u8;
            return Ok(None);
        }
        self.burn(Read::Entry, self.bindings.root)?;
        let frame = self
            .maximum
            .as_mut()
            .ok_or(Error::Consumed)?
            .begin(self.ledger.as_mut().ok_or(Error::Consumed)?)?;
        self.phase = Phase::Maximum as u8;
        Ok(Some(Body::Maximum(frame)))
    }
    fn differences(&mut self) -> Result<Body<C>, Error> {
        let source = Arc::clone(&self.keepalive.source);
        let s = source.declaration;
        let mut values = exact_words(usize::from(s.count), self.lease.budget.budget())?;
        for (lane, &position) in s.active_positions().iter().enumerate() {
            // M/X read identities are the fixed template's actual bindings.
            let ledger = self.ledger.as_mut().ok_or(Error::Consumed)?;
            let (m, x) = (self.bindings.maximum, self.bindings.scores[lane]);
            ledger.burn(
                Read::Difference {
                    lane: lane as u8,
                    operand: 0,
                },
                m,
                self.role,
            )?;
            ledger.burn(
                Read::Difference {
                    lane: lane as u8,
                    operand: 1,
                },
                x,
                self.role,
            )?;
            let maximum = self.maximum.as_ref().ok_or(Error::Consumed)?;
            values[lane] =
                maximum.maximum().wrapping_sub(maximum.original(position)) & ((1u64 << 50) - 1);
        }
        self.burn(Read::ChildInput, self.bindings.difference)?;
        let lease = self.lease.split(add(
            size_of::<DifferenceValueFor<C>>(),
            mul(values.len(), 8)?,
        )?)?;
        let value = DifferenceValueFor {
            binding: self.bindings.difference,
            role: self.role,
            words: values,
            source,
            lease,
        };
        self.phase = Phase::Child as u8;
        inherited::begin(self.child.as_mut().ok_or(Error::Consumed)?, value).map(Body::Child)
    }
    fn denominator(&mut self) -> Result<Body<C>, Error> {
        self.burn(Read::ChildOutput, self.bindings.exponent)?;
        let mut words = exact_words(1, self.lease.budget.budget())?;
        let source = Arc::clone(&self.keepalive.source);
        {
            let loan = self.exponent.as_ref().ok_or(Error::Consumed)?.loan()?;
            if loan.binding() != self.bindings.exponent
                || loan.role() != self.role
                || loan.values().len() != usize::from(source.declaration.count)
            {
                return Err(Error::Context);
            }
            for lane in 0..loan.values().len() {
                self.ledger.as_mut().ok_or(Error::Consumed)?.burn(
                    Read::Sum { lane: lane as u8 },
                    self.bindings.exponent,
                    self.role,
                )?;
                words[0] = words[0].wrapping_add(loan.values()[lane]) & ((1u64 << 50) - 1);
            }
        }
        self.burn(Read::Project, self.bindings.sum)?;
        words[0] &= (1u64 << 14) - 1;
        self.burn(Read::Truncate, self.bindings.denominator)?;
        let lease = self
            .lease
            .split(add(size_of::<DenominatorValueFor<C>>(), 8)?)?;
        let denominator = DenominatorValueFor {
            binding: self.bindings.denominator,
            role: self.role,
            words,
            source,
            lease,
        };
        self.phase = Phase::Coordinate as u8;
        self.coordinate
            .as_mut()
            .ok_or(Error::Consumed)?
            .begin(denominator)
            .map(Body::Arithmetic)
    }
    fn product(&mut self) -> Result<Body<C>, Error> {
        let k = usize::from(self.keepalive.source.declaration.count);
        let loan = self.exponent.as_ref().ok_or(Error::Consumed)?.loan()?;
        if loan.binding() != self.bindings.exponent || loan.values().len() != k {
            return Err(Error::Context);
        }
        let reciprocal = self.reciprocal.as_ref().ok_or(Error::Consumed)?;
        for lane in 0..k {
            self.ledger.as_mut().ok_or(Error::Consumed)?.burn(
                Read::Product {
                    lane: lane as u8,
                    operand: 0,
                },
                self.bindings.exponent,
                self.role,
            )?;
            self.ledger.as_mut().ok_or(Error::Consumed)?.burn(
                Read::Product {
                    lane: lane as u8,
                    operand: 1,
                },
                self.bindings.reciprocal,
                self.role,
            )?;
            self.broadcast[lane] = reciprocal.values()[0];
        }
        let frame = self
            .product
            .as_mut()
            .ok_or(Error::Consumed)?
            .begin(loan.values(), &self.broadcast)?;
        self.phase = Phase::Product as u8;
        Ok(Body::Product(frame))
    }
    fn advance(&mut self, peer: &[u8]) -> Result<Option<Body<C>>, Error> {
        match self.phase {
            x if x == Phase::Maximum as u8 => {
                match self
                    .maximum
                    .as_mut()
                    .ok_or(Error::Consumed)?
                    .advance(peer, self.ledger.as_mut().ok_or(Error::Consumed)?)?
                {
                    Some(frame) => Ok(Some(Body::Maximum(frame))),
                    None => self.differences().map(Some),
                }
            }
            x if x == Phase::Child as u8 => {
                match inherited::advance(self.child.as_mut().ok_or(Error::Consumed)?, peer)? {
                    child::Step::Frame(frame) => Ok(Some(Body::Child(frame))),
                    child::Step::Ready => {
                        self.exponent = Some(inherited::accept_ready(
                            self.child.take().ok_or(Error::Consumed)?,
                        )?);
                        self.denominator().map(Some)
                    }
                }
            }
            x if x == Phase::Coordinate as u8 => {
                match self
                    .coordinate
                    .as_mut()
                    .ok_or(Error::Consumed)?
                    .advance(peer)?
                {
                    coordinate::StepFor::Frame(frame) => Ok(Some(Body::Arithmetic(frame))),
                    coordinate::StepFor::Output(value) => {
                        self.coordinate_value = Some(value);
                        self.burn(Read::Coordinate, self.bindings.coordinate)?;
                        self.burn(Read::Inverse, self.bindings.coordinate)?;
                        let view = self
                            .coordinate_value
                            .as_mut()
                            .ok_or(Error::Consumed)?
                            .coordinate_once()?;
                        if view.binding() != self.bindings.coordinate {
                            return Err(Error::Context);
                        }
                        let frame = self
                            .inverse
                            .as_mut()
                            .ok_or(Error::Consumed)?
                            .begin(view.values())?;
                        self.phase = Phase::Inverse as u8;
                        Ok(Some(Body::Lookup(frame)))
                    }
                }
            }
            x if x == Phase::Inverse as u8 => {
                match self
                    .inverse
                    .as_mut()
                    .ok_or(Error::Consumed)?
                    .advance(peer)?
                {
                    lookup::StepFor::Frame(frame) => Ok(Some(Body::Lookup(frame))),
                    lookup::StepFor::Reconstructed(value) => {
                        self.reciprocal = Some(value);
                        self.product().map(Some)
                    }
                    _ => Err(Error::Context),
                }
            }
            x if x == Phase::Product as u8 => {
                self.product_value =
                    Some(self.product.as_mut().ok_or(Error::Consumed)?.finish(peer)?);
                self.burn(Read::Gap, self.bindings.product)?;
                let frame = self
                    .gap
                    .as_mut()
                    .ok_or(Error::Consumed)?
                    .begin(self.product_value.as_ref().ok_or(Error::Consumed)?.values())?;
                self.phase = Phase::Gap as u8;
                Ok(Some(Body::Arithmetic(frame)))
            }
            x if x == Phase::Gap as u8 => {
                match self.gap.as_mut().ok_or(Error::Consumed)?.advance(peer)? {
                    arithmetic::StepFor::Frame(frame) => Ok(Some(Body::Arithmetic(frame))),
                    arithmetic::StepFor::Output(value) => {
                        self.burn(Read::Output, self.bindings.output)?;
                        if !self.ledger.as_ref().ok_or(Error::Consumed)?.complete() {
                            return Err(Error::Consumed);
                        }
                        self.output = Some(value);
                        self.phase = Phase::Ready as u8;
                        self.retire_continuations();
                        Ok(None)
                    }
                }
            }
            _ => Err(Error::Consumed),
        }
    }
    fn retire_continuations(&mut self) {
        if let Some(mut x) = self.maximum.take() {
            x.cancel();
        }
        self.child = None;
        self.exponent = None;
        if let Some(x) = &mut self.coordinate {
            x.cancel();
        }
        if let Some(x) = &mut self.inverse {
            x.cancel();
        }
        if let Some(x) = &mut self.product {
            x.cancel();
        }
        if let Some(x) = &mut self.gap {
            x.cancel();
        }
        self.coordinate_value = None;
        self.reciprocal = None;
        self.product_value = None;
        // A second abort/retry cannot rescan a retired source buffer. Replacing
        // with the zero-capacity Default does not allocate; old Zeroizing drops
        // BEFORE its original state charge can retire.
        drop(std::mem::take(&mut self.broadcast));
    }
}
impl<C: AllocationContext> custody::Retire for State<C> {
    fn retire(&mut self) {
        if self.phase == Phase::Burnt as u8 {
            return;
        }
        self.phase = Phase::Burnt as u8;
        // Abnormal teardown ONLY. Receipts have non-aborting ordinary Drop;
        // child Role may already be consumed. The strictly downstream child
        // registry never holds this parent coordinator or callbacks.
        if let Some(child) = &self.keepalive.child {
            inherited::abort(child);
        }
        self.output = None;
        self.public = None;
        self.retire_continuations();
    }
}
impl<C: AllocationContext> Frame<C> {
    pub fn bytes(&self) -> Result<&[u8], Error> {
        self.shared.operation(|_| self.native.bytes())
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
impl<C: AllocationContext> Drop for Mask<C> {
    fn drop(&mut self) {
        self.shared.abort();
    }
}
impl<C: AllocationContext> Mask<C> {
    pub fn descriptor(&self) -> Result<EdgeDescriptor, Error> {
        self.shared.operation(|_| Ok(self.binding.descriptor))
    }
}
impl<C: AllocationContext> Output<C> {
    pub fn role(&self) -> u8 {
        self.role
    }
    pub fn descriptor(&self) -> Result<EdgeDescriptor, Error> {
        self.shared.operation(|_| {
            let state = self.slot.state.lock().map_err(|_| Error::Consumed)?;
            if state.output.is_none() && state.public.is_none() {
                return Err(Error::Consumed);
            }
            Ok(state.bindings.output.descriptor)
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
    pub fn begin(&mut self, cancelled: &mut impl FnMut() -> bool) -> Result<Step<C>, Error> {
        self.shared.poll_cancelled(cancelled)?;
        self.shared.operation(|control| {
            let body = self
                .slot
                .state
                .lock()
                .map_err(|_| Error::Consumed)?
                .begin()?;
            match body {
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
    pub fn advance(
        &mut self,
        peer: &[u8],
        cancelled: &mut impl FnMut() -> bool,
    ) -> Result<Step<C>, Error> {
        self.shared.poll_cancelled(cancelled)?;
        self.shared.operation(|control| {
            let body = self
                .slot
                .state
                .lock()
                .map_err(|_| Error::Consumed)?
                .advance(peer)?;
            match body {
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
        let result = self.shared.operation(|control| {
            control.require_both_ready()?;
            let state = self.slot.state.lock().map_err(|_| Error::Consumed)?;
            if state.phase != Phase::Ready as u8 || state.output.is_none() && state.public.is_none()
            {
                return Err(Error::Consumed);
            }
            Ok(Output {
                slot: Arc::clone(&self.slot),
                shared: Arc::clone(&self.shared),
                role: self.role,
            })
        });
        self.handed_off = result.is_ok();
        result
    }
}
impl<C: AllocationContext> Drop for Role<C> {
    fn drop(&mut self) {
        if !self.handed_off {
            self.shared.abort();
        }
    }
}

/// Same child declaration, no surrogate clear root and no table parameters.
fn child_edges(s: &CausalSoftmaxSpec) -> Result<[EdgeDescriptor; 14], Error> {
    let k = usize::from(s.count);
    let descriptions = [
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
    let mut edges = [s.input; 14];
    for (i, (bits, scale, kind)) in descriptions.into_iter().enumerate() {
        edges[i] = s.wire(100 + i as u16, k, bits, scale, kind)?;
    }
    Ok(edges)
}
pub(super) fn child_spec(s: &CausalSoftmaxSpec) -> Result<child::Spec, Error> {
    Ok(inherited::spec(
        child_edges(s)?,
        digest(&[spec::DOMAIN, &s.operation, b"child"]),
        s.operation,
    ))
}

fn edge_for(
    s: &CausalSoftmaxSpec,
    r: Read,
    child_edges: &[EdgeDescriptor; 14],
) -> Result<EdgeDescriptor, Error> {
    let k = usize::from(s.count);
    match r {
        Read::Entry => Ok(s.input),
        Read::Max { update, operand } => {
            let base = 10 + u16::from(update) * 8;
            let (slot, bits, scale, kind) = match operand {
                0 => {
                    if update == 0 {
                        (1, 50, 12, Interpretation::Signed)
                    } else {
                        (15, 50, 12, Interpretation::Signed)
                    }
                }
                1 | 6 => (2 + u16::from(update), 50, 12, Interpretation::Signed),
                2 | 5 => (base + 1, 50, 12, Interpretation::Signed),
                3 => (base + 2, 39, 12, Interpretation::Signed),
                4 => (base + 3, 1, 0, Interpretation::Predicate),
                7 => (base + 4, 50, 12, Interpretation::Signed),
                _ => return Err(Error::Domain),
            };
            s.wire(slot, 1, bits, scale, kind)
        }
        Read::Difference { lane, operand } => {
            if operand == 0 {
                s.wire(
                    10 + (k - 2) as u16 * 8 + 5,
                    1,
                    50,
                    12,
                    Interpretation::Signed,
                )
            } else {
                s.wire(1 + u16::from(lane), 1, 50, 12, Interpretation::Signed)
            }
        }
        Read::ChildInput => Ok(child_edges[0]),
        Read::ChildOutput | Read::Sum { .. } => Ok(child_edges[13]),
        Read::Project => s.wire(205, 1, 50, 12, Interpretation::Unsigned),
        Read::Truncate => s.wire(200, 1, 14, 12, Interpretation::Unsigned),
        Read::Coordinate | Read::Inverse => s.wire(201, 1, 8, 0, Interpretation::Unsigned),
        Read::Product { operand: 0, .. } => Ok(child_edges[13]),
        Read::Product { .. } => s.wire(202, 1, 50, 12, Interpretation::Signed),
        Read::Gap => s.wire(203, k, 50, 24, Interpretation::Signed),
        Read::Output => Ok(s.output),
    }
}
fn uses(
    s: &CausalSoftmaxSpec,
    edges: &[EdgeDescriptor; 14],
) -> Result<[pllm_core::edge_contract::ConsumerUse; spec::MAX_USES], Error> {
    use pllm_core::edge_contract::ConsumerUse;
    let (reads, n) = ledger::template(usize::from(s.count))?;
    let mut result = [ConsumerUse::new(s.output, s.next, 0, 0); spec::MAX_USES];
    for i in 0..n {
        result[i] = ConsumerUse::new(
            edge_for(s, reads[i], edges)?,
            if reads[i] == Read::Output {
                s.next
            } else {
                s.operation
            },
            0,
            i as u32,
        );
    }
    Ok(result)
}

/// Concrete named call/Result/old-new populations, not opaque node guesses.
#[allow(dead_code)]
struct TransitionSlots<C: AllocationContext> {
    owned_working: crate::shared_allocation::OwnedExternalCharge,
    original_input: CausalSoftmaxInput,
    // Outer record hash, nested numeric/context hash, and core's simultaneous
    // use/edge workspace. Dependency-private SHA stack remains excluded.
    hashes: [Sha256; 2],
    core_hash: pllm_core::edge_contract::EdgeContractWorkspace,
    spec: CausalSoftmaxSpec,
    quote: CausalSoftmaxResources,
    working: LeaseFor<C>,
    source: Arc<SourceFor<C>>,
    child_spec: child::Spec,
    child_quote: Result<child::Quote, Error>,
    child_created: Result<child::Created<C>, child::BuildError>,
    max_created: Result<(maximum::DealerFor<C>, [maximum::PartyFor<C>; 2]), Error>,
    coordinate_declaration: Result<coordinate::DeclarationFor<C>, Error>,
    coordinate_created: Result<(coordinate::MaskFor<C>, [coordinate::PartyFor<C>; 2]), Error>,
    inverse_created: Result<(lookup::OutputMasksFor<C>, [lookup::PartyFor<C>; 2]), inverse::Error>,
    product_created: Result<(product::MasksFor<C>, [product::PartyFor<C>; 2]), Error>,
    gap_created: Result<(arithmetic::OutputMasksFor<C>, [arithmetic::PartyFor<C>; 2]), Error>,
    e_receipt: Result<inherited::EValueFor<C>, Error>,
    bindings: Bindings,
    max_bindings: maximum::Bindings,
    bindings_result: Result<Binding, Error>,
    uses: [pllm_core::edge_contract::ConsumerUse; spec::MAX_USES],
    ledgers: [Ledger; 3],
    row_scratch: [u64; 3],
    mask_scratch: [Zeroizing<Box<[u64]>>; 3],
    mask_origin: Mask<C>,
    mask_result: Result<Mask<C>, Error>,
    created: Created<C>,
    created_result: Result<Created<C>, BuildError>,
    state_origin: State<C>,
    state_result: Result<State<C>, Error>,
    native_body: Body<C>,
    frame_origin: Frame<C>,
    frame_result: Result<Frame<C>, Error>,
    step_origin: Step<C>,
    step_result: Result<Step<C>, Error>,
    output_origin: Output<C>,
    output_result: Result<Output<C>, Error>,
    hash: Sha256,
    hash_args: [&'static [u8]; 32],
    hash_output: sha2::digest::Output<Sha256>,
    role_array: [Role<C>; 2],
    slot_array: [Arc<Slot<C>>; 2],
    shared: Arc<Shared<C>>,
    coordinate_quote: coordinate::Resources,
    inverse_quote: inverse::Quote,
    max_native: Result<(maximum::DealerFor<C>, [maximum::PartyFor<C>; 2]), Error>,
    coord_native: Result<(coordinate::MaskFor<C>, [coordinate::PartyFor<C>; 2]), Error>,
    inverse_native: Result<(lookup::OutputMasksFor<C>, [lookup::PartyFor<C>; 2]), inverse::Error>,
    product_native: Result<(product::MasksFor<C>, [product::PartyFor<C>; 2]), Error>,
    gap_native: Result<(arithmetic::OutputMasksFor<C>, [arithmetic::PartyFor<C>; 2]), Error>,
}
pub(super) fn transition_bytes<C: AllocationContext>() -> Result<usize, Error> {
    sum(&[
        size_of::<TransitionSlots<C>>(),
        size_of::<custody::CallbackPollFor<'_, C, State<C>>>(),
        size_of::<custody::OperationFor<'_, C, State<C>>>(),
        size_of::<std::sync::MutexGuard<'_, State<C>>>(),
        size_of::<coordinate::MaskCoordinate<'_, C>>(),
        size_of::<coordinate::ValueCoordinate<'_, C>>(),
        size_of::<Result<coordinate::ValueCoordinate<'_, C>, Error>>(),
        2 * C::TRANSIENT_LINK_BYTES,
    ])
}
pub(super) fn record_bytes(s: &CausalSoftmaxSpec) -> Result<usize, Error> {
    // Length framing and actual fixed source/edge/use encodings; used only as a
    // conservative bound, never as asserted serialized output or golden hash.
    let fixed = [
        spec::DOMAIN.len(),
        b"declaration.v1".len(),
        spec::EQUATION.len(),
        32,
        32,
        32,
        32,
        32,
        32,
        32,
        64,
        40,
        32,
        32,
        8,
        1,
        usize::from(s.count),
        16,
        16,
        12,
        32,
        32,
        32,
        32,
    ];
    let mut total = 0;
    for length in fixed {
        total = add(total, add(8, length)?)?;
    }
    add(total, mul(s.outer_uses(), 40)?)
}
pub(super) fn zero_child_work() -> child::WorkQuote {
    child::WorkQuote {
        public_declaration_descriptor_checks: 0,
        public_resource_query_passes: 0,
        public_resource_descriptor_checks: 0,
        input_validation_lanes: 0,
        input_retirement_words: 0,
        root_validation_lanes: 0,
        root_initialization_words: 0,
        root_encoding_word_writes: 0,
        root_mask_generation_words: 0,
        root_retirement_words: 0,
        ordered_use_checks: 0,
        public_table_range_checks: 0,
        stage_bound_descriptor_checks: 0,
        stage_bound_interval_writes: 0,
        stage_bound_metadata_queries: 0,
        stage_bound_table_metadata_passes: 0,
        buffer_initialization_words: 0,
        buffer_erasure_words: 0,
        projection_lane_visits: 0,
        floor_taylor: 0,
        floor_squarings: 0,
        floor_multiply: 0,
        floor_divide: 0,
        floor_remainder: 0,
        floor_add: 0,
        floor_shift: 0,
        paired_native_issue_aes: 0,
        native_evaluate_aes_per_role: 0,
        random_byte_requests: 0,
    }
}

/// Already-live original-counter authority is required even for preflight
/// controls. These are concrete source-level logical slots, not stack/RSS.
pub(super) fn bootstrap_bytes<C: AllocationContext>() -> Result<usize, Error> {
    sum(&[
        size_of::<CausalSoftmaxSpec>(),
        size_of::<CausalSoftmaxResources>(),
        size_of::<Result<CausalSoftmaxResources, Error>>(),
        size_of::<LeaseFor<C>>(),
        transition_bytes::<C>()?,
        inherited::bootstrap_bytes::<C>(),
    ])
}

pub(super) fn construct<C: AllocationContext>(
    dealer: &mut LocalDealerFor<C>,
    s: CausalSoftmaxSpec,
    input: CausalSoftmaxInput,
    working: LeaseFor<C>,
    limit: CausalSoftmaxWorkLimit,
) -> Result<Created<C>, BuildError> {
    construct_with(dealer, s, input, working, limit, |bytes| {
        getrandom::fill(bytes).map_err(|_| Error::Randomness)
    })
}

#[allow(clippy::arc_with_non_send_sync)]
fn construct_with<C: AllocationContext>(
    dealer: &mut LocalDealerFor<C>,
    s: CausalSoftmaxSpec,
    input: CausalSoftmaxInput,
    mut working: LeaseFor<C>,
    limit: CausalSoftmaxWorkLimit,
    mut fill: impl FnMut(&mut [u8]) -> Result<(), Error>,
) -> Result<Created<C>, BuildError> {
    if working.bytes < bootstrap_bytes::<C>()?
        || limit.0 == 0
        || !std::ptr::eq(working.budget.budget(), dealer.budget.budget())
        || !std::ptr::eq(input.budget(), dealer.budget.budget())
    {
        return Err(Error::Resource.into());
    }
    let q = quote::resources(dealer, &s, &input)?;
    if working.bytes != q.total_reserved_bytes || limit.0 < q.work.total {
        return Err(Error::Resource.into());
    }
    if s.count == 1 {
        return public_singleton(dealer, s, input, working, q);
    }
    // All source integers were scanned by resources BEFORE proof/hash/RNG.
    // Source-defined static evidence is arithmetic only, never source authority.
    let (_exp_verified, exp_hashes) = {
        let mut workspace = crate::exp_floor::Workspace::default();
        exp::prepare(&mut workspace)?
    };
    let (verified, inverse_hash) = inverse::prepare()?;
    let hashes = [exp_hashes[0], exp_hashes[1], inverse_hash];
    let edges = child_edges(&s)?;
    let uses = uses(&s, &edges)?;
    let declaration = records::declaration(&s, dealer.session, hashes, &uses[..s.outer_uses()]);
    let numeric = records::numeric(dealer.source, hashes);
    // Every native lease is split from the original aggregate before its
    // corresponding factory. No fresh reserve/refund/reacquire operation.
    let root_lease = working.split(q.original_root_bytes)?;
    let child_lease = working.split(q.child_bytes)?;
    let coord_lease = working.split(q.coordinate_bytes)?;
    let inverse_lease = working.split(q.inverse_bytes)?;
    let product_lease = working.split(q.product_bytes)?;
    let gap_lease = working.split(q.gap_bytes)?;
    let [left_lease, right_lease] = [working.split(q.state_bytes)?, working.split(q.state_bytes)?];
    let [left_slot, right_slot] = [working.split(q.slot_bytes)?, working.split(q.slot_bytes)?];
    let coordination_lease = working.split(q.coordination_bytes)?;
    let (comparison_cost, selection_cost) = maximum::costs::<C>()?;
    let mut comparison_leases = [None, None];
    let mut selection_leases = [None, None];
    for i in 0..usize::from(s.count) - 1 {
        comparison_leases[i] = Some(working.split(comparison_cost.combined_reservation_bytes)?);
        selection_leases[i] = Some(working.split(selection_cost.combined_reservation_bytes)?);
    }
    let source_lease = working.split(q.source_bytes)?;
    dealer.budget.budget().allocation()?;
    let source = Arc::new(SourceFor {
        input,
        declaration: s,
        lease: source_lease,
    });
    let prepared = dealer.prepare_root_reserved_with(
        s.input,
        BoundRange::new(-spec::SCORE_LIMIT, spec::SCORE_LIMIT)?,
        s.operation,
        root_lease,
        &mut fill,
    )?;
    let root_binding = prepared.dealer.binding;
    let maximum_bindings = maximum::declare(dealer, &s, root_binding)?;
    let difference = dealer.binding(
        edges[0],
        digest(&[spec::DOMAIN, &s.operation, b"child"]),
        Some(Interval {
            lower: 0,
            upper: 1i128 << 38,
        }),
    )?;
    let sum_binding = dealer.binding(
        s.wire(205, 1, 50, 12, Interpretation::Unsigned)?,
        s.operation,
        Some(Interval {
            lower: 4096,
            upper: 4096 * i128::from(s.count) + 1,
        }),
    )?;
    let denominator = dealer.binding(
        s.wire(200, 1, 14, 12, Interpretation::Unsigned)?,
        s.operation,
        Some(Interval {
            lower: 4096,
            upper: 4096 * i128::from(s.count) + 1,
        }),
    )?;
    let quotient = s.wire(201, 1, 8, 6, Interpretation::Unsigned)?;
    let coordinate_edge = s.wire(201, 1, 8, 0, Interpretation::Unsigned)?;
    let coordinate_declaration = coordinate::declare_reserved(
        dealer,
        denominator,
        coordinate::Spec {
            operation: s.operation,
            quotient,
            coordinate: coordinate_edge,
            next: s.operation,
        },
        coord_lease,
    )?;
    let coordinate_binding = coordinate_declaration.output_binding();
    let reciprocal = dealer.binding(
        s.wire(202, 1, 50, 12, Interpretation::Signed)?,
        s.operation,
        Some(Interval {
            lower: 1365,
            upper: 4097,
        }),
    )?;
    let product_binding = dealer.binding(
        s.wire(203, usize::from(s.count), 50, 24, Interpretation::Signed)?,
        s.operation,
        Some(Interval {
            lower: 0,
            upper: (1i128 << 24) + 1,
        }),
    )?;
    let gap_binding = dealer.binding(
        s.wire(204, usize::from(s.count), 50, 12, Interpretation::Signed)?,
        s.operation,
        Some(Interval {
            lower: 0,
            upper: 4097,
        }),
    )?;
    let output = dealer.binding(
        s.output,
        s.next,
        Some(Interval {
            lower: 0,
            upper: 4097,
        }),
    )?;
    let mut dealer_ledger = Ledger::new(
        usize::from(s.count),
        2,
        uses,
        [root_binding; spec::MAX_USES],
    )?;
    // Genuine original-root burn BEFORE protected clear read/encoding.
    dealer_ledger.burn(Read::Entry, root_binding, 2)?;
    let root = prepared.encode(source.input.payload())?;
    let (maximum, [max0, max1]) = maximum::issue(
        &s,
        declaration,
        maximum_bindings.clone(),
        root,
        comparison_leases,
        selection_leases,
        &mut dealer_ledger,
        &mut fill,
    )?;
    let mut differences = exact_words(usize::from(s.count), working.budget.budget())?;
    for (lane, &position) in s.active_positions().iter().enumerate() {
        dealer_ledger.burn(
            Read::Difference {
                lane: lane as u8,
                operand: 0,
            },
            maximum_bindings.maxima[usize::from(s.count) - 2],
            2,
        )?;
        dealer_ledger.burn(
            Read::Difference {
                lane: lane as u8,
                operand: 1,
            },
            maximum_bindings.scores[lane],
            2,
        )?;
        differences[lane] = maximum
            .maximum_mask
            .wrapping_sub(maximum.original_mask(position))
            & ((1u64 << 50) - 1);
    }
    let difference_lease = working.split(add(
        size_of::<DifferenceMaskFor<C>>(),
        mul(usize::from(s.count), 8)?,
    )?)?;
    let difference_mask = DifferenceMaskFor {
        binding: difference,
        words: differences,
        source: Arc::clone(&source),
        lease: difference_lease,
    };
    dealer_ledger.burn(Read::ChildInput, difference, 2)?;
    let (child_mask, child_coord, [child0, child1]) = inherited::construct_with(
        dealer,
        child_spec(&s)?,
        difference_mask,
        child_lease,
        &mut fill,
    )?;
    let mut denominator_words = exact_words(1, working.budget.budget())?;
    let exponent;
    {
        let loan = inherited::loan_mask(&child_mask)?;
        exponent = loan.binding();
        if exponent.descriptor != edges[13] || loan.values().len() != usize::from(s.count) {
            return Err(Error::Context.into());
        }
        dealer_ledger.burn(Read::ChildOutput, exponent, 2)?;
        for lane in 0..loan.values().len() {
            dealer_ledger.burn(Read::Sum { lane: lane as u8 }, exponent, 2)?;
            denominator_words[0] =
                denominator_words[0].wrapping_add(loan.values()[lane]) & ((1u64 << 50) - 1);
        }
    }
    dealer_ledger.burn(Read::Project, sum_binding, 2)?;
    denominator_words[0] &= (1u64 << 14) - 1;
    let denominator_lease = working.split(add(size_of::<DenominatorMaskFor<C>>(), 8)?)?;
    let denominator_mask = DenominatorMaskFor {
        binding: denominator,
        words: denominator_words,
        source: Arc::clone(&source),
        lease: denominator_lease,
    };
    dealer_ledger.burn(Read::Truncate, denominator, 2)?;
    let (mut coordinate_mask, [coordinate0, coordinate1]) =
        coordinate_declaration.issue_with(denominator_mask, &mut fill)?;
    dealer_ledger.burn(Read::Coordinate, coordinate_binding, 2)?;
    dealer_ledger.burn(Read::Inverse, coordinate_binding, 2)?;
    let (inverse_mask, [inverse0, inverse1]) = {
        let view = coordinate_mask.coordinate_once()?;
        if view.binding() != coordinate_binding
            || view.reachable_inverse()
                != (Interval {
                    lower: 1365,
                    upper: 4097,
                })
        {
            return Err(Error::Context.into());
        }
        inverse::issue(
            &verified,
            lookup::Context {
                plan: dealer.plan,
                session: dealer.session,
                operation: records::stage(declaration, 2 * (u16::from(s.count) - 1) + 2),
                input_wire: coordinate_binding.digest(),
                output_wire: reciprocal.digest(),
                arithmetic_wire: digest(&[&declaration, b"inverse.arithmetic"]),
                correction_wire: digest(&[&declaration, b"inverse.correction"]),
                select_wire: digest(&[&declaration, b"inverse.select"]),
                numeric_contract: numeric,
                first_tensor_index: s.input.identity().first_tensor_index(),
            },
            view.values(),
            inverse_lease,
            &mut fill,
        )?
    };
    let mut broadcast = exact_words(usize::from(s.count), working.budget.budget())?;
    let (product_mask, [product0, product1]) = {
        let loan = inherited::loan_mask(&child_mask)?;
        for lane in 0..usize::from(s.count) {
            dealer_ledger.burn(
                Read::Product {
                    lane: lane as u8,
                    operand: 0,
                },
                exponent,
                2,
            )?;
            dealer_ledger.burn(
                Read::Product {
                    lane: lane as u8,
                    operand: 1,
                },
                reciprocal,
                2,
            )?;
            broadcast[lane] = inverse_mask.values()[0];
        }
        product::issue_reserved(
            product::Context {
                region: declaration,
                stage: records::stage(declaration, 2 * (u16::from(s.count) - 1) + 3),
                inputs: [exponent.digest(), reciprocal.digest()],
                output: product_binding.digest(),
                first: s.input.identity().first_tensor_index(),
            },
            loan.values(),
            &broadcast,
            product_lease,
            &mut fill,
        )?
    };
    dealer_ledger.burn(Read::Gap, product_binding, 2)?;
    let (gap_mask, [gap0, gap1]) = arithmetic::issue_reserved_with(
        arithmetic::Spec {
            bits: 50,
            lanes: usize::from(s.count),
            gate: arithmetic::Gate::GapArs { shift: 12 },
        },
        arithmetic::Context {
            plan: dealer.plan,
            session: dealer.session,
            operation: records::stage(declaration, 2 * (u16::from(s.count) - 1) + 4),
            input_wire: product_binding.digest(),
            output_wire: gap_binding.digest(),
            first_tensor_index: s.input.identity().first_tensor_index(),
        },
        product_mask.values(),
        gap_lease,
        &mut fill,
    )?;
    dealer_ledger.burn(Read::Output, output, 2)?;
    let actual_bindings = dealer_ledger.frozen()?;
    let bindings = Bindings {
        root: root_binding,
        maximum: maximum_bindings.maxima[usize::from(s.count) - 2],
        scores: maximum_bindings.scores,
        difference,
        exponent,
        sum: sum_binding,
        denominator,
        coordinate: coordinate_binding,
        reciprocal,
        product: product_binding,
        _gap: gap_binding,
        output,
    };
    dealer.budget.budget().allocation()?;
    let keepalive = Arc::new(KeepaliveFor {
        source,
        child: Some(child_coord),
        _verified: Some(verified),
        _lease: working.split(0)?,
    });
    let make = |role,
                maximum,
                child,
                coordinate,
                inverse,
                product,
                gap,
                lease|
     -> Result<State<C>, Error> {
        Ok(State {
            maximum: Some(maximum),
            child: Some(child),
            exponent: None,
            coordinate: Some(coordinate),
            coordinate_value: None,
            inverse: Some(inverse),
            reciprocal: None,
            product: Some(product),
            product_value: None,
            gap: Some(gap),
            output: None,
            public: None,
            broadcast: exact_words(usize::from(s.count), working.budget.budget())?,
            phase: Phase::Start as u8,
            role,
            bindings,
            ledger: Some(Ledger::new(
                usize::from(s.count),
                role,
                uses,
                actual_bindings,
            )?),
            keepalive: Arc::clone(&keepalive),
            lease,
            _affinity: PhantomData,
        })
    };
    let states = [
        make(
            0,
            max0,
            child0,
            coordinate0,
            inverse0,
            product0,
            gap0,
            left_lease,
        )?,
        make(
            1,
            max1,
            child1,
            coordinate1,
            inverse1,
            product1,
            gap1,
            right_lease,
        )?,
    ];
    publish(
        states,
        [left_slot, right_slot],
        working,
        coordination_lease,
        bindings.output,
        Some(gap_mask),
        Some(maximum),
        Some(child_mask),
        Some(coordinate_mask),
        Some(inverse_mask),
        Some(product_mask),
    )
}

#[allow(clippy::arc_with_non_send_sync)]
#[allow(clippy::too_many_arguments)] // Explicit original material/lease transfers; no new allocation.
fn publish<C: AllocationContext>(
    states: [State<C>; 2],
    slot_leases: [LeaseFor<C>; 2],
    mut working: LeaseFor<C>,
    coordination_lease: LeaseFor<C>,
    binding: Binding,
    native: Option<arithmetic::OutputMasksFor<C>>,
    maximum: Option<maximum::DealerFor<C>>,
    child: Option<child::Mask<C>>,
    coordinate: Option<coordinate::MaskFor<C>>,
    inverse: Option<lookup::OutputMasksFor<C>>,
    product: Option<product::MasksFor<C>>,
) -> Result<Created<C>, BuildError> {
    let allocation = working.budget.share();
    let budget = allocation.budget();
    let mask_lease = working.split(size_of::<Mask<C>>())?;
    let [left, right] = states;
    let [left_lease, right_lease] = slot_leases;
    budget.allocation()?;
    let left = Arc::new(Slot {
        state: Mutex::new(left),
        _lease: left_lease,
    });
    budget.allocation()?;
    let right = Arc::new(Slot {
        state: Mutex::new(right),
        _lease: right_lease,
    });
    budget.allocation()?;
    let shared = Arc::new(Shared {
        control: Mutex::new(custody::SplitControl::default()),
        slots: [left, right],
        _original_run: Some(working),
        _additional: coordination_lease,
    });
    let roles = std::array::from_fn(|role| Role {
        slot: Arc::clone(&shared.slots[role]),
        shared: Arc::clone(&shared),
        role: role as u8,
        handed_off: false,
    });
    Ok((
        Mask {
            native,
            maximum,
            child,
            coordinate,
            inverse,
            product,
            binding,
            shared: Arc::clone(&shared),
            _lease: mask_lease,
        },
        Coordinator { shared },
        roles,
    ))
}

#[allow(clippy::arc_with_non_send_sync)]
fn public_singleton<C: AllocationContext>(
    dealer: &mut LocalDealerFor<C>,
    s: CausalSoftmaxSpec,
    input: CausalSoftmaxInput,
    mut working: LeaseFor<C>,
    q: CausalSoftmaxResources,
) -> Result<Created<C>, BuildError> {
    let source_lease = working.split(q.source_bytes)?;
    let [left_lease, right_lease] = [working.split(q.state_bytes)?, working.split(q.state_bytes)?];
    let slots = [working.split(q.slot_bytes)?, working.split(q.slot_bytes)?];
    let coordination_lease = working.split(q.coordination_bytes)?;
    dealer.budget.budget().allocation()?;
    let source = Arc::new(SourceFor {
        input,
        declaration: s,
        lease: source_lease,
    });
    let output = dealer.binding(
        s.output,
        s.next,
        Some(Interval {
            lower: 0,
            upper: 4097,
        }),
    )?;
    let bindings = Bindings {
        root: output,
        maximum: output,
        scores: [output; 3],
        difference: output,
        exponent: output,
        sum: output,
        denominator: output,
        coordinate: output,
        reciprocal: output,
        product: output,
        _gap: output,
        output,
    };
    dealer.budget.budget().allocation()?;
    let keepalive = Arc::new(KeepaliveFor {
        source,
        child: None,
        _verified: None,
        _lease: working.split(0)?,
    });
    let mut codes = [0; 3];
    codes[usize::from(s.active[0])] = 4096;
    let make = |role, lease| -> Result<State<C>, Error> {
        Ok(State {
            maximum: None,
            child: None,
            exponent: None,
            coordinate: None,
            coordinate_value: None,
            inverse: None,
            reciprocal: None,
            product: None,
            product_value: None,
            gap: None,
            output: None,
            public: Some(codes),
            broadcast: exact_words(0, working.budget.budget())?,
            phase: Phase::Start as u8,
            role,
            bindings,
            ledger: None,
            keepalive: Arc::clone(&keepalive),
            lease,
            _affinity: PhantomData,
        })
    };
    let states = [make(0, left_lease)?, make(1, right_lease)?];
    publish(
        states,
        slots,
        working,
        coordination_lease,
        output,
        None,
        None,
        None,
        None,
        None,
        None,
    )
}

impl<C: AllocationContext> Drop for State<C> {
    fn drop(&mut self) {
        custody::Retire::retire(self);
    }
}

#[cfg(test)]
#[path = "causal_softmax_canonical_tests.rs"]
mod canonical_tests;
#[cfg(test)]
#[path = "causal_softmax_outer_abort_regressions.rs"]
mod outer_abort_regressions;
#[cfg(test)]
#[path = "causal_softmax_tests.rs"]
mod tests;
