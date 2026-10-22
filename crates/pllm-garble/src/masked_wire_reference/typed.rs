//! Bounded local SIGMA ownership/provenance bridge, not a compiler or SDK.
//!
//! Public edge descriptors are declarations, never capabilities or authenticated
//! checkpoint identities. `LocalAssumption` records developer premises checked
//! by the trusted local root encoder; it is not a compiler-admitted proof.
//! Only the approved GPU127/D17 profile is accepted here. Legacy profiles and
//! their identities are untouched. No FuseFSS, public coefficients, additive
//! shares, general imports, decoder, transport, or independent security claim.
//!
//! Fresh matrix material alone supports one consuming, same-budget local handoff.
//! Its opaque packet can only be restored by its original nonserializable permit;
//! descriptors are inspectable declarations, never owners or authorization. This
//! preserves local issuance continuity, not authentication, durable replay
//! protection, checkpoint loading, or external authorization. Original typed
//! input capabilities remain necessary; metadata never reconstructs them.
//!
//! Single-consumer slice: each root supplies exactly one capability per party;
//! begin consumes those capabilities. No cloning, fanout or remasking API exists.
//! The fixed [`ActivationPrefixParty`] alone retains original backing for its
//! precommitted TR6/DReLU/Select uses; no intermediate capability is exported.
//! Every two-element owner array is ordered `[party0, party1]`. Consumer tags
//! identify the predeclared consumer operation/edge; issuing that operation
//! additionally binds its complete input and output descriptor digests.
//! On failure, callers must explicitly cancel the sibling party. Native output
//! backing owners (including matrix singleton guards) remain intact until drop.
//! Charges are logical capacities, not RSS bounds. Borrowed caller clear input
//! memory is never retained and remains the caller's responsibility.

use super::{arithmetic, matrix, selection};
pub use crate::shared_allocation::{Budget, Error};
use crate::{
    compact_dpf::gpu127::Profile,
    shared_allocation::{allocate, Lease},
};
use pllm_core::edge_contract::{
    EdgeDescriptor, Interpretation, Numeric, Representation, SourceIdentity, WeightVisibility,
};
use sha2::{Digest, Sha256};
use std::{marker::PhantomData, mem::size_of};
use zeroize::Zeroizing;

mod activation;
mod internal_use;
mod linear_activation;
mod material;
mod region;
mod selection_bridge;
pub use activation::{
    activation_resources, ActivationFrame, ActivationIssued, ActivationKind,
    ActivationMetadataResources, ActivationMetadataWorkspace, ActivationParty,
    ActivationResourceCalculationResources, ActivationResources, ActivationSpec, ActivationStep,
};
pub use linear_activation::{
    linear_activation_resources as local_linear_activation_resources,
    LinearActivationFrame as LocalLinearActivationFrame,
    LinearActivationMetadataResources as LocalLinearActivationMetadataResources,
    LinearActivationResourceCalculationResources as LocalLinearActivationResourceCalculationResources,
    LinearActivationResources as LocalLinearActivationResources,
    LinearActivationRun as LocalLinearActivationRun,
    LinearActivationSpec as LocalLinearActivationSpec,
    LinearActivationStep as LocalLinearActivationStep,
};
pub use material::{
    matrix_material_resources, MatrixHandoff, MatrixImportPermit, MatrixMaterialDescriptor,
    MatrixMaterialPacket, MatrixMaterialProvenance, MatrixMaterialResources,
};
pub use region::{ActivationPrefixParty, PrefixFrame, PrefixSpec, PrefixStep};
pub use selection_bridge::SelectionParty;

const ALLOCATOR: usize = 64;

/// Retirement erases dynamic state, but not a caller-retained inline wrapper.
/// Preserve its original charge until Drop; never refund and reacquire it.
fn retain_wrapper_metadata(lease: &mut Option<Lease<'_>>, bytes: usize) {
    if let Some(lease) = lease.as_mut() {
        let refund = lease.bytes.saturating_sub(bytes);
        drop(lease.split(refund));
    }
}

/// Arithmetic and predicate owners deliberately have different Rust types.
pub enum Ring {}
pub enum Predicate {}
/// A developer premise, never an independently established range proof.
pub enum LocalAssumption {}

/// Half-open integer interval, declared independently of private input values.
/// Binding to a full edge descriptor is performed only by the local dealer.
pub struct BoundRange<P> {
    lower: i128,
    upper: i128,
    _premise: PhantomData<P>,
}
impl BoundRange<LocalAssumption> {
    pub fn new(lower: i128, upper: i128) -> Result<Self, Error> {
        if lower >= upper {
            return Err(Error::Domain);
        }
        Ok(Self {
            lower,
            upper,
            _premise: PhantomData,
        })
    }
    pub fn endpoints(&self) -> (i128, i128) {
        (self.lower, self.upper)
    }
}

/// Explicitly distinguish modular multiplication from an unwrapped promise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixRange {
    /// Deliberately discard unwrapped provenance; cannot qualify GapARS later.
    RingOnly,
    /// Require checked interval products/sums to fit the signed output ring.
    RequireSignedNoWrap,
}

/// Implementation declaration required by this adapter. This digest does not
/// authenticate the supplied checkpoint or numeric-profile declaration.
pub fn approved_implementation() -> [u8; 32] {
    digest(&[
        b"pllm.sigma.typed.gpu127-d17.v1\0",
        arithmetic::DOMAIN.as_bytes(),
        Profile::DOMAIN.as_bytes(),
    ])
}

fn digest(parts: &[&[u8]]) -> [u8; 32] {
    let mut h = Sha256::new();
    for part in parts {
        h.update((part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.finalize().into()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Interval {
    lower: i128,
    upper: i128,
}
impl Interval {
    fn fits(self, bits: u8) -> bool {
        let half = 1i128 << (bits - 1);
        self.lower >= -half && self.upper <= half && self.lower < self.upper
    }
    fn shift(self, shift: u8) -> Self {
        let divisor = 1i128 << shift;
        Self {
            lower: self.lower.div_euclid(divisor),
            upper: (self.upper - 1).div_euclid(divisor) + 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Binding {
    descriptor: EdgeDescriptor,
    session: [u8; 32],
    namespace: [u8; 32],
    issuance: u64,
    mask_id: [u8; 32],
    consumer: [u8; 32],
    // This interval is always about exactly `descriptor`, not a reusable token.
    interval: Option<Interval>,
}
impl Binding {
    fn digest(self) -> [u8; 32] {
        let (tag, lower, upper) = match self.interval {
            Some(i) => (1u8, i.lower, i.upper),
            None => (0u8, 0, 0),
        };
        digest(&[
            b"pllm.sigma.typed.binding.v1\0",
            &self.descriptor.digest(),
            &self.session,
            &self.namespace,
            &self.issuance.to_le_bytes(),
            &self.mask_id,
            &self.consumer,
            &[tag],
            &lower.to_le_bytes(),
            &upper.to_le_bytes(),
        ])
    }
}

// Complete native backing owners are moved into these variants. Never detach
// their leases/guards, reconstruct a Vec from their slices, or copy their payload.
enum CommonBacking<'b> {
    Root(Zeroizing<Vec<u64>>),
    Arithmetic(arithmetic::MaskedOutput<'b>),
    Matrix(matrix::party::MaskedOutput<'b>),
    Selection(selection::MaskedOutput<'b>),
}
impl CommonBacking<'_> {
    fn values(&self) -> &[u64] {
        match self {
            Self::Root(v) => v,
            Self::Arithmetic(v) => v.values(),
            Self::Matrix(v) => v.values(),
            Self::Selection(v) => v.values(),
        }
    }
}
enum MaskBacking<'b> {
    Fresh(Zeroizing<Vec<u64>>),
    Arithmetic(arithmetic::OutputMasks<'b>),
    Selection(selection::OutputMasks<'b>),
}
impl MaskBacking<'_> {
    fn values(&self) -> &[u64] {
        match self {
            Self::Fresh(v) => v,
            Self::Arithmetic(v) => v.values(),
            Self::Selection(v) => v.values(),
        }
    }
}

/// Noncloneable evaluator capability. No raw constructor, export, reimport,
/// mask getter or into_inner. Predicate instances are common-XOR-masked.
///
/// ```compile_fail
/// use pllm_garble::masked_wire_reference::typed::{ArithmeticParty, CommonXorMasked, Predicate, Ring};
/// fn cannot_feed_predicate<'a>(p: &mut ArithmeticParty<'a, Ring>, x: CommonXorMasked<'a, Predicate>) {
///     p.begin(x); // An XOR predicate is not an arithmetic-ring capability.
/// }
/// ```
///
/// ```compile_fail
/// use pllm_garble::masked_wire_reference::typed::{CommonMasked, Ring};
/// fn duplicate<'a>(x: CommonMasked<'a, Ring>) -> CommonMasked<'a, Ring> {
///     x.clone()
/// }
/// ```
///
/// Public metadata cannot reconstruct an evaluator capability:
///
/// ```compile_fail
/// use pllm_core::edge_contract::EdgeDescriptor;
/// use pllm_garble::masked_wire_reference::typed::{CommonMasked, Ring};
/// fn reconstruct<'a>(descriptor: EdgeDescriptor) -> CommonMasked<'a, Ring> {
///     descriptor.into()
/// }
/// ```
pub struct CommonMasked<'b, K> {
    backing: CommonBacking<'b>,
    binding: Binding,
    role: u8,
    _kind: PhantomData<K>,
    // Always last: payload zeroizes and deallocates before budget refund.
    _lease: Lease<'b>,
}
pub type CommonXorMasked<'b, K> = CommonMasked<'b, K>;
impl<K> CommonMasked<'_, K> {
    pub fn descriptor(&self) -> EdgeDescriptor {
        self.binding.descriptor
    }
}

/// Trusted dealer-only mask capability. Operations take it by value. Never
/// supplied to an evaluator; there is no public access to its masks.
///
/// ```compile_fail
/// use pllm_garble::masked_wire_reference::typed::{DealerMask, Ring};
/// fn duplicate<'a>(mask: DealerMask<'a, Ring>) -> DealerMask<'a, Ring> {
///     mask.clone()
/// }
/// ```
///
/// ```compile_fail
/// use pllm_core::edge_contract::EdgeDescriptor;
/// use pllm_garble::masked_wire_reference::typed::{DealerMask, Ring};
/// fn reconstruct<'a>(descriptor: EdgeDescriptor) -> DealerMask<'a, Ring> {
///     descriptor.into()
/// }
/// ```
///
/// A predicate mask cannot enter a ring operation, even with valid metadata:
///
/// ```compile_fail
/// use pllm_core::edge_contract::EdgeDescriptor;
/// use pllm_garble::masked_wire_reference::typed::{DealerMask, LocalDealer, Predicate};
/// fn feed<'a>(dealer: &mut LocalDealer<'a>, mask: DealerMask<'a, Predicate>, output: EdgeDescriptor) {
///     dealer.tr([1; 32], mask, 1, output, [2; 32]);
/// }
/// ```
pub struct DealerMask<'b, K> {
    backing: MaskBacking<'b>,
    binding: Binding,
    _kind: PhantomData<K>,
    _lease: Lease<'b>,
}

/// Trusted local dealer. A source identity is a developer declaration, not
/// checkpoint authentication. Namespace is fresh OS randomness per instance.
///
/// Native local API example (no decoding, transport or SDK implied):
///
/// ```
/// use pllm_core::edge_contract::{EdgeDescriptor, EdgeIdentity, Interpretation,
///     Numeric, Representation, Shape, SourceIdentity};
/// use pllm_garble::masked_wire_reference::typed::{approved_implementation,
///     BoundRange, Budget, Error, LocalDealer};
/// # fn example() -> Result<(), Error> {
/// let budget = Budget::new(1 << 20)?;
/// let source = SourceIdentity::new([1; 32], [2; 32], approved_implementation());
/// let plan = [3; 32];
/// let descriptor = |wire, bits, f| EdgeDescriptor::new(
///     EdgeIdentity::new(source, plan, [4; 32], [wire; 32], 0),
///     Shape::new(&[1]).unwrap(),
///     Numeric::new(bits, f, Interpretation::Signed).unwrap(),
///     Representation::CommonAdditiveMasked, None,
/// ).unwrap();
/// let mut dealer = LocalDealer::new(source, plan, [5; 32], &budget)?;
/// // Promise is source/edge-bound before any private value enters the API.
/// let prepared = dealer.prepare_root(descriptor(6, 16, 4),
///     BoundRange::new(-32, 32)?, [7; 32])?;
/// let (mask, [x0, x1]) = prepared.encode(&[-1])?;
/// let (_next_mask, [mut p0, mut p1]) = dealer.tr([7; 32], mask, 2,
///     descriptor(8, 14, 2), [9; 32])?;
/// let _frame0 = p0.begin(x0)?;
/// let _frame1 = p1.begin(x1)?;
/// // Enclosing coordinator owns peer delivery and explicit sibling retirement.
/// p0.cancel();
/// p1.cancel();
/// # Ok(()) }
/// # example().unwrap();
/// ```
pub struct LocalDealer<'b> {
    source: SourceIdentity,
    plan: [u8; 32],
    session: [u8; 32],
    namespace: [u8; 32],
    counter: u64,
    budget: &'b Budget,
    _lease: Lease<'b>,
}

/// Prepared source-bound promise and fresh masks, created before receiving
/// private values. Failure in encode burns all three owners.
pub struct RootInput<'b> {
    dealer: DealerMask<'b, Ring>,
    common: [CommonMasked<'b, Ring>; 2],
}

fn zeros(lanes: usize, budget: &Budget) -> Result<Zeroizing<Vec<u64>>, Error> {
    let mut v = Zeroizing::new(allocate(lanes, budget)?);
    v.resize(lanes, 0);
    Ok(v)
}
fn fill_masks_with(
    values: &mut [u64],
    bits: u8,
    mut fill: impl FnMut(&mut [u8]) -> Result<(), Error>,
) -> Result<(), Error> {
    for value in values {
        let mut bytes = Zeroizing::new([0; 8]);
        fill(&mut *bytes)?;
        *value = u64::from_le_bytes(*bytes) & super::mask(bits);
    }
    Ok(())
}
fn common_charge() -> usize {
    size_of::<CommonMasked<'static, Ring>>()
}
fn mask_charge() -> usize {
    size_of::<DealerMask<'static, Ring>>()
}

impl<'b> LocalDealer<'b> {
    pub fn new(
        source: SourceIdentity,
        plan: [u8; 32],
        session: [u8; 32],
        budget: &'b Budget,
    ) -> Result<Self, Error> {
        if source.implementation() != approved_implementation() {
            return Err(Error::Unsupported);
        }
        let lease = budget.reserve(size_of::<Self>() + (4 << 10))?;
        let mut namespace = [0; 32];
        getrandom::fill(&mut namespace).map_err(|_| Error::Randomness)?;
        Ok(Self {
            source,
            plan,
            session,
            namespace,
            counter: 0,
            budget,
            _lease: lease,
        })
    }

    fn validate(&self, descriptor: EdgeDescriptor, predicate: bool) -> Result<(), Error> {
        if descriptor.identity().source() != self.source
            || descriptor.identity().plan() != self.plan
        {
            return Err(Error::Context);
        }
        if descriptor.shape().elements() > 256 {
            return Err(Error::Resource);
        }
        let numeric = descriptor.numeric();
        if predicate {
            if numeric.interpretation() != Interpretation::Predicate
                || descriptor.representation() != Representation::CommonXorMasked
            {
                return Err(Error::Unsupported);
            }
        } else if numeric.interpretation() != Interpretation::Signed
            || descriptor.representation() != Representation::CommonAdditiveMasked
        {
            return Err(Error::Unsupported);
        }
        if descriptor.weight_visibility() == Some(WeightVisibility::Public) {
            return Err(Error::Unsupported);
        }
        Ok(())
    }

    fn binding(
        &mut self,
        descriptor: EdgeDescriptor,
        consumer: [u8; 32],
        interval: Option<Interval>,
    ) -> Result<Binding, Error> {
        let issuance = self.counter;
        self.counter = self.counter.checked_add(1).ok_or(Error::Resource)?;
        let mask_id = digest(&[
            b"pllm.sigma.typed.mask-id.v1\0",
            &self.namespace,
            &issuance.to_le_bytes(),
            &descriptor.digest(),
        ]);
        Ok(Binding {
            descriptor,
            session: self.session,
            namespace: self.namespace,
            issuance,
            mask_id,
            consumer,
            interval,
        })
    }

    fn input(&self, binding: Binding, operation: [u8; 32]) -> Result<(), Error> {
        self.validate(binding.descriptor, false)?;
        if binding.session != self.session
            || binding.namespace != self.namespace
            || binding.consumer != operation
        {
            return Err(Error::Context);
        }
        Ok(())
    }

    /// Public-only preflight includes exact root 3×8L payload, three allocator
    /// allowances and explicit wrapper metadata. No private input is accepted.
    pub fn prepare_root(
        &mut self,
        descriptor: EdgeDescriptor,
        promise: BoundRange<LocalAssumption>,
        consumer: [u8; 32],
    ) -> Result<RootInput<'b>, Error> {
        let bytes = self.root_reservation_bytes(descriptor, &promise)?;
        let reservation = self.budget.reserve(bytes)?;
        self.prepare_root_reserved_with(descriptor, promise, consumer, reservation, |bytes| {
            getrandom::fill(bytes).map_err(|_| Error::Randomness)
        })
    }

    /// Public-only accounting for composition. No allocation or entropy occurs.
    fn root_reservation_bytes(
        &self,
        descriptor: EdgeDescriptor,
        promise: &BoundRange<LocalAssumption>,
    ) -> Result<usize, Error> {
        self.validate(descriptor, false)?;
        let interval = Interval {
            lower: promise.lower,
            upper: promise.upper,
        };
        if !interval.fits(descriptor.numeric().ring_bits()) {
            return Err(Error::Domain);
        }
        let lanes = descriptor.shape().elements();
        let common_bytes = 8 * lanes + ALLOCATOR + common_charge();
        let dealer_bytes = 8 * lanes + ALLOCATOR + mask_charge() + size_of::<RootInput<'static>>();
        Ok(2 * common_bytes + dealer_bytes)
    }

    /// Consume an exact original aggregate slice, without refund/reacquire.
    /// Callers accepting clear inputs validate them before invoking this helper.
    fn prepare_root_reserved_with(
        &mut self,
        descriptor: EdgeDescriptor,
        promise: BoundRange<LocalAssumption>,
        consumer: [u8; 32],
        mut lease: Lease<'b>,
        fill: impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<RootInput<'b>, Error> {
        let bytes = self.root_reservation_bytes(descriptor, &promise)?;
        if lease.bytes != bytes || !std::ptr::eq(lease.budget, self.budget) {
            return Err(Error::Resource);
        }
        let interval = Interval {
            lower: promise.lower,
            upper: promise.upper,
        };
        let lanes = descriptor.shape().elements();
        let common_bytes = 8 * lanes + ALLOCATOR + common_charge();
        let left_lease = lease.split(common_bytes)?;
        let right_lease = lease.split(common_bytes)?;
        let binding = self.binding(descriptor, consumer, Some(interval))?;
        let mut masks = zeros(lanes, self.budget)?;
        let left = zeros(lanes, self.budget)?;
        let right = zeros(lanes, self.budget)?;
        fill_masks_with(&mut masks, descriptor.numeric().ring_bits(), fill)?;
        Ok(RootInput {
            dealer: DealerMask {
                backing: MaskBacking::Fresh(masks),
                binding,
                _kind: PhantomData,
                _lease: lease,
            },
            common: [
                CommonMasked {
                    backing: CommonBacking::Root(left),
                    binding,
                    role: 0,
                    _kind: PhantomData,
                    _lease: left_lease,
                },
                CommonMasked {
                    backing: CommonBacking::Root(right),
                    binding,
                    role: 1,
                    _kind: PhantomData,
                    _lease: right_lease,
                },
            ],
        })
    }
}

impl<'b> RootInput<'b> {
    /// Validate against the predeclared interval, never derive public metadata
    /// from values. Input is borrowed transiently, not copied or retained clear.
    pub fn encode(
        mut self,
        values: &[i64],
    ) -> Result<(DealerMask<'b, Ring>, [CommonMasked<'b, Ring>; 2]), Error> {
        let binding = self.dealer.binding;
        let interval = binding.interval.ok_or(Error::Domain)?;
        if values.len() != binding.descriptor.shape().elements()
            || values
                .iter()
                .any(|&v| i128::from(v) < interval.lower || i128::from(v) >= interval.upper)
        {
            return Err(Error::Domain);
        }
        let bits = binding.descriptor.numeric().ring_bits();
        for common in &mut self.common {
            let CommonBacking::Root(out) = &mut common.backing else {
                return Err(Error::Domain);
            };
            for ((out, &value), &mask) in
                out.iter_mut().zip(values).zip(self.dealer.backing.values())
            {
                *out = (value as u64).wrapping_add(mask) & super::mask(bits);
            }
        }
        Ok((self.dealer, self.common))
    }
}

/// Party-local matrix owner. Each instance contains only its own native keys.
pub struct MatrixParty<'b> {
    native: Option<matrix::party::Party<'b>>,
    inputs: [Binding; 2],
    output: Binding,
    operation: [u8; 32],
    role: u8,
    imported: bool,
    lease: Option<Lease<'b>>,
    // Additional handoff metadata charge follows native secrets and original lease.
    material_lease: Option<Lease<'b>>,
}

fn matrix_interval(x: Interval, w: Interval, inner: usize, bits: u8) -> Result<Interval, Error> {
    let mut lower = i128::MAX;
    let mut upper = i128::MIN;
    for a in [x.lower, x.upper - 1] {
        for b in [w.lower, w.upper - 1] {
            let p = a.checked_mul(b).ok_or(Error::Domain)?;
            lower = lower.min(p);
            upper = upper.max(p);
        }
    }
    let count = i128::try_from(inner).map_err(|_| Error::Resource)?;
    let result = Interval {
        lower: lower.checked_mul(count).ok_or(Error::Domain)?,
        upper: upper
            .checked_mul(count)
            .and_then(|v| v.checked_add(1))
            .ok_or(Error::Domain)?,
    };
    if !result.fits(bits) {
        return Err(Error::Domain);
    }
    Ok(result)
}

/// Exact incremental admission; input backing and dealer remain separate owners.
pub(crate) fn matrix_reservation_bytes(spec: matrix::Spec) -> Result<usize, Error> {
    let native = matrix::party::resources(spec)?;
    Ok(native.reservation_bytes
        + 2 * (size_of::<MatrixParty<'static>>() + common_charge())
        + 8 * native.output_elements
        + ALLOCATOR
        + mask_charge())
}

/// Exact incremental admission for typed arithmetic, excluding original input.
pub(crate) fn arithmetic_reservation_bytes<K>(spec: arithmetic::Spec) -> Result<usize, Error> {
    Ok(arithmetic::resources(spec)?.combined_reservation_bytes
        + 2 * (size_of::<ArithmeticParty<'static, K>>() + common_charge())
        + mask_charge())
}

impl<'b> LocalDealer<'b> {
    /// Private-weight matrix only. `output` Q must equal fx+fw exactly. Native
    /// singleton/256-element/2^20 complete-MAC admission is preserved unchanged.
    // Keep operation, operands, result edge and range policy explicit at issuance.
    #[allow(clippy::too_many_arguments)]
    pub fn matrix(
        &mut self,
        operation: [u8; 32],
        spec: matrix::Spec,
        x: DealerMask<'b, Ring>,
        w: DealerMask<'b, Ring>,
        output: EdgeDescriptor,
        next_consumer: [u8; 32],
        range: MatrixRange,
    ) -> Result<(DealerMask<'b, Ring>, [MatrixParty<'b>; 2]), Error> {
        self.matrix_with(
            operation,
            spec,
            x,
            w,
            output,
            next_consumer,
            range,
            None,
            &mut |bytes| getrandom::fill(bytes).map_err(|_| Error::Randomness),
        )
    }

    /// Move exact `matrix_reservation_bytes(spec)` on this dealer's budget.
    /// Never refunds/reacquires native capacity. Inputs are consumed on failure.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn matrix_reserved_with(
        &mut self,
        operation: [u8; 32],
        spec: matrix::Spec,
        x: DealerMask<'b, Ring>,
        w: DealerMask<'b, Ring>,
        output: EdgeDescriptor,
        next_consumer: [u8; 32],
        range: MatrixRange,
        reservation: Lease<'b>,
        fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<(DealerMask<'b, Ring>, [MatrixParty<'b>; 2]), Error> {
        self.matrix_with(
            operation,
            spec,
            x,
            w,
            output,
            next_consumer,
            range,
            Some(reservation),
            fill,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn matrix_with(
        &mut self,
        operation: [u8; 32],
        spec: matrix::Spec,
        x: DealerMask<'b, Ring>,
        w: DealerMask<'b, Ring>,
        output: EdgeDescriptor,
        next_consumer: [u8; 32],
        range: MatrixRange,
        reservation: Option<Lease<'b>>,
        fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<(DealerMask<'b, Ring>, [MatrixParty<'b>; 2]), Error> {
        self.matrix_with_ticket(
            operation,
            spec,
            x,
            w,
            output,
            next_consumer,
            range,
            reservation,
            None,
            fill,
        )
    }

    /// Fixed regions acquire native singleton and issuance authority before
    /// drawing either root. Move that same authority into native material.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn matrix_ticket_reserved_with(
        &mut self,
        operation: [u8; 32],
        spec: matrix::Spec,
        x: DealerMask<'b, Ring>,
        w: DealerMask<'b, Ring>,
        output: EdgeDescriptor,
        next_consumer: [u8; 32],
        range: MatrixRange,
        reservation: Lease<'b>,
        ticket: matrix::party::IssuanceTicket,
        fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<(DealerMask<'b, Ring>, [MatrixParty<'b>; 2]), Error> {
        self.matrix_with_ticket(
            operation,
            spec,
            x,
            w,
            output,
            next_consumer,
            range,
            Some(reservation),
            Some(ticket),
            fill,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn matrix_with_ticket(
        &mut self,
        operation: [u8; 32],
        spec: matrix::Spec,
        x: DealerMask<'b, Ring>,
        w: DealerMask<'b, Ring>,
        output: EdgeDescriptor,
        next_consumer: [u8; 32],
        range: MatrixRange,
        reservation: Option<Lease<'b>>,
        ticket: Option<matrix::party::IssuanceTicket>,
        fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<(DealerMask<'b, Ring>, [MatrixParty<'b>; 2]), Error> {
        self.input(x.binding, operation)?;
        self.input(w.binding, operation)?;
        self.validate(output, false)?;
        let xd = x.binding.descriptor;
        let wd = w.binding.descriptor;
        if wd.weight_visibility() != Some(WeightVisibility::Private) {
            return Err(Error::Unsupported);
        }
        let native_cost = matrix::party::resources(spec)?;
        let weight_dimensions = match spec.layout {
            matrix::MatrixLayout::InputOutput => [spec.inner, spec.columns],
            matrix::MatrixLayout::OutputInput => [spec.columns, spec.inner],
        };
        if xd.shape().dimensions() != [spec.rows, spec.inner]
            || wd.shape().dimensions() != weight_dimensions
            || output.shape().dimensions() != [spec.rows, spec.columns]
            || xd.numeric().ring_bits() != spec.bits
            || wd.numeric().ring_bits() != spec.bits
            || output.numeric().ring_bits() != spec.bits
            || output.numeric().fractional_bits()
                != xd
                    .numeric()
                    .fractional_bits()
                    .checked_add(wd.numeric().fractional_bits())
                    .ok_or(Error::Domain)?
        {
            return Err(Error::Domain);
        }
        if [xd.identity().wire(), wd.identity().wire()].contains(&output.identity().wire())
            || xd.identity().wire() == wd.identity().wire()
        {
            return Err(Error::Context);
        }
        let interval = match range {
            MatrixRange::RingOnly => None,
            MatrixRange::RequireSignedNoWrap => Some(matrix_interval(
                x.binding.interval.ok_or(Error::Domain)?,
                w.binding.interval.ok_or(Error::Domain)?,
                spec.inner,
                spec.bits,
            )?),
        };
        let party_bytes = size_of::<MatrixParty<'static>>() + common_charge();
        let mask_bytes = 8 * native_cost.output_elements + ALLOCATOR + mask_charge();
        let bytes = matrix_reservation_bytes(spec)?;
        let mut reservation = match reservation {
            Some(lease) if lease.bytes == bytes && std::ptr::eq(lease.budget, self.budget) => lease,
            Some(_) => return Err(Error::Resource),
            None => self.budget.reserve(bytes)?,
        };
        let left_lease = reservation.split(party_bytes)?;
        let right_lease = reservation.split(party_bytes)?;
        let mask_lease = reservation.split(mask_bytes)?;
        let binding = self.binding(output, next_consumer, interval)?;
        let mut masks = zeros(native_cost.output_elements, self.budget)?;
        fill_masks_with(&mut masks, spec.bits, &mut *fill)?;
        let op = digest(&[
            b"pllm.sigma.typed.matrix.v1\0",
            &operation,
            &x.binding.digest(),
            &w.binding.digest(),
            &binding.digest(),
        ]);
        let context = matrix::Context {
            plan: self.plan,
            session: self.session,
            operation: op,
            input_wire: xd.digest(),
            weight_wire: wd.digest(),
            output_wire: output.digest(),
        };
        let inputs = [x.backing.values(), w.backing.values(), &masks];
        let [left, right] = match ticket {
            Some(ticket) => matrix::party::issue_reserved_ticket_with(
                spec,
                context,
                inputs,
                reservation,
                ticket,
                fill,
                &mut || false,
            ),
            None => matrix::party::issue_reserved_with(
                spec,
                context,
                inputs,
                reservation,
                fill,
                &mut || false,
            ),
        }?;
        let inputs = [x.binding, w.binding];
        Ok((
            DealerMask {
                backing: MaskBacking::Fresh(masks),
                binding,
                _kind: PhantomData,
                _lease: mask_lease,
            },
            [
                MatrixParty {
                    native: Some(left),
                    inputs,
                    output: binding,
                    operation,
                    role: 0,
                    imported: false,
                    lease: Some(left_lease),
                    material_lease: None,
                },
                MatrixParty {
                    native: Some(right),
                    inputs,
                    output: binding,
                    operation,
                    role: 1,
                    imported: false,
                    lease: Some(right_lease),
                    material_lease: None,
                },
            ],
        ))
    }
}

impl<'b> MatrixParty<'b> {
    pub fn cancel(&mut self) {
        self.native = None;
        retain_wrapper_metadata(&mut self.lease, size_of::<Self>());
        self.material_lease = None;
    }
    /// Consumes both evaluator capabilities, even on wrong role/mask/context.
    pub fn begin(
        &mut self,
        x: CommonMasked<'b, Ring>,
        w: CommonMasked<'b, Ring>,
        cancelled: &mut impl FnMut() -> bool,
    ) -> Result<matrix::party::Frame<'b>, Error> {
        let result = (|| {
            if x.binding != self.inputs[0]
                || w.binding != self.inputs[1]
                || x.role != self.role
                || w.role != self.role
            {
                return Err(Error::Context);
            }
            self.native.as_mut().ok_or(Error::Consumed)?.begin(
                x.backing.values(),
                w.backing.values(),
                cancelled,
            )
        })();
        if result.is_err() {
            self.cancel();
        }
        result
    }
    pub fn finish(&mut self, peer: &[u8]) -> Result<CommonMasked<'b, Ring>, Error> {
        let result = (|| {
            let output = self.native.as_mut().ok_or(Error::Consumed)?.finish(peer)?;
            let lease = self.lease.as_mut().ok_or(Error::Consumed)?;
            let lease = lease.split(lease.bytes - size_of::<Self>())?;
            // Native completed owner retains its own inline metadata charge.
            self.material_lease = None;
            Ok(CommonMasked {
                backing: CommonBacking::Matrix(output),
                binding: self.output,
                role: self.role,
                _kind: PhantomData,
                _lease: lease,
            })
        })();
        if result.is_err() {
            self.cancel();
        }
        result
    }
}

pub struct ArithmeticParty<'b, K> {
    native: Option<arithmetic::Party<'b>>,
    input: Binding,
    output: Binding,
    role: u8,
    _kind: PhantomData<K>,
    lease: Option<Lease<'b>>,
}
// Keep native output ownership inline: boxing adds an unpriced, infallible allocation.
#[allow(clippy::large_enum_variant)]
pub enum Step<'b, K> {
    Frame(arithmetic::Frame<'b>),
    Output(CommonMasked<'b, K>),
}

impl<'b> LocalDealer<'b> {
    pub fn tr(
        &mut self,
        operation: [u8; 32],
        input: DealerMask<'b, Ring>,
        shift: u8,
        output: EdgeDescriptor,
        next_consumer: [u8; 32],
    ) -> Result<(DealerMask<'b, Ring>, [ArithmeticParty<'b, Ring>; 2]), Error> {
        self.arithmetic(
            operation,
            input,
            arithmetic::Gate::Tr { shift },
            output,
            next_consumer,
            false,
        )
    }
    pub fn gap_ars(
        &mut self,
        operation: [u8; 32],
        input: DealerMask<'b, Ring>,
        shift: u8,
        output: EdgeDescriptor,
        next_consumer: [u8; 32],
    ) -> Result<(DealerMask<'b, Ring>, [ArithmeticParty<'b, Ring>; 2]), Error> {
        self.arithmetic(
            operation,
            input,
            arithmetic::Gate::GapArs { shift },
            output,
            next_consumer,
            false,
        )
    }
    /// XOR-masked [signed input >= 0]. Zero is true. Output cannot enter an
    /// arithmetic operation: those functions require DealerMask<Ring>.
    pub fn drelu(
        &mut self,
        operation: [u8; 32],
        input: DealerMask<'b, Ring>,
        output: EdgeDescriptor,
        next_consumer: [u8; 32],
    ) -> Result<
        (
            DealerMask<'b, Predicate>,
            [ArithmeticParty<'b, Predicate>; 2],
        ),
        Error,
    > {
        self.arithmetic(
            operation,
            input,
            arithmetic::Gate::Drelu,
            output,
            next_consumer,
            true,
        )
    }

    fn arithmetic<K>(
        &mut self,
        operation: [u8; 32],
        input: DealerMask<'b, Ring>,
        gate: arithmetic::Gate,
        output: EdgeDescriptor,
        next_consumer: [u8; 32],
        predicate: bool,
    ) -> Result<(DealerMask<'b, K>, [ArithmeticParty<'b, K>; 2]), Error> {
        self.arithmetic_with(
            operation,
            input,
            gate,
            output,
            next_consumer,
            predicate,
            None,
            &mut |bytes| getrandom::fill(bytes).map_err(|_| Error::Randomness),
        )
    }

    /// Move exact `arithmetic_reservation_bytes::<Ring>(spec)` on dealer budget.
    /// Original input backing is consumed, never copied or re-encoded.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gap_ars_reserved_with(
        &mut self,
        operation: [u8; 32],
        input: DealerMask<'b, Ring>,
        shift: u8,
        output: EdgeDescriptor,
        next_consumer: [u8; 32],
        reservation: Lease<'b>,
        fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<(DealerMask<'b, Ring>, [ArithmeticParty<'b, Ring>; 2]), Error> {
        self.arithmetic_reserved_with(
            operation,
            input,
            arithmetic::Gate::GapArs { shift },
            output,
            next_consumer,
            false,
            reservation,
            fill,
        )
    }

    /// Exact original admission for typed TR; same quote contract as GapARS.
    // Shared reserved adapter retained for fixed-region refactors. D22 uses
    // GapARS and unchanged D20 issues its TR directly from its own aggregate.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tr_reserved_with(
        &mut self,
        operation: [u8; 32],
        input: DealerMask<'b, Ring>,
        shift: u8,
        output: EdgeDescriptor,
        next_consumer: [u8; 32],
        reservation: Lease<'b>,
        fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<(DealerMask<'b, Ring>, [ArithmeticParty<'b, Ring>; 2]), Error> {
        self.arithmetic_reserved_with(
            operation,
            input,
            arithmetic::Gate::Tr { shift },
            output,
            next_consumer,
            false,
            reservation,
            fill,
        )
    }

    /// Exact `arithmetic_reservation_bytes::<Predicate>(spec)` original admission.
    // Shared reserved adapter retained for fixed-region refactors. Unchanged D20
    // issues DReLU directly from its aggregate rather than this typed adapter.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn drelu_reserved_with(
        &mut self,
        operation: [u8; 32],
        input: DealerMask<'b, Ring>,
        output: EdgeDescriptor,
        next_consumer: [u8; 32],
        reservation: Lease<'b>,
        fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<
        (
            DealerMask<'b, Predicate>,
            [ArithmeticParty<'b, Predicate>; 2],
        ),
        Error,
    > {
        self.arithmetic_reserved_with(
            operation,
            input,
            arithmetic::Gate::Drelu,
            output,
            next_consumer,
            true,
            reservation,
            fill,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn arithmetic_reserved_with<K>(
        &mut self,
        operation: [u8; 32],
        input: DealerMask<'b, Ring>,
        gate: arithmetic::Gate,
        output: EdgeDescriptor,
        next_consumer: [u8; 32],
        predicate: bool,
        reservation: Lease<'b>,
        fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<(DealerMask<'b, K>, [ArithmeticParty<'b, K>; 2]), Error> {
        self.arithmetic_with(
            operation,
            input,
            gate,
            output,
            next_consumer,
            predicate,
            Some(reservation),
            fill,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn arithmetic_with<K>(
        &mut self,
        operation: [u8; 32],
        input: DealerMask<'b, Ring>,
        gate: arithmetic::Gate,
        output: EdgeDescriptor,
        next_consumer: [u8; 32],
        predicate: bool,
        reservation: Option<Lease<'b>>,
        fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<(DealerMask<'b, K>, [ArithmeticParty<'b, K>; 2]), Error> {
        self.input(input.binding, operation)?;
        self.validate(output, predicate)?;
        let descriptor = input.binding.descriptor;
        if descriptor.shape() != output.shape()
            || descriptor.identity().wire() == output.identity().wire()
            || descriptor.identity().first_tensor_index() != output.identity().first_tensor_index()
        {
            return Err(Error::Context);
        }
        let numeric = descriptor.numeric();
        let spec = arithmetic::Spec {
            bits: numeric.ring_bits(),
            lanes: descriptor.shape().elements(),
            gate,
        };
        arithmetic::resources(spec)?;
        let (expected, interval) = match gate {
            arithmetic::Gate::Tr { shift } => {
                let f = numeric
                    .fractional_bits()
                    .checked_sub(shift)
                    .ok_or(Error::Unsupported)?;
                (
                    Numeric::new(spec.bits - shift, f, Interpretation::Signed)
                        .map_err(|_| Error::Domain)?,
                    input.binding.interval.map(|i| i.shift(shift)),
                )
            }
            arithmetic::Gate::GapArs { shift } => {
                let f = numeric
                    .fractional_bits()
                    .checked_sub(shift)
                    .ok_or(Error::Unsupported)?;
                let range = input.binding.interval.ok_or(Error::Domain)?;
                let quarter = 1i128 << (spec.bits - 2);
                // Full half-open range, not absolute-value or masked-value test.
                if range.lower < -quarter || range.upper > quarter {
                    return Err(Error::Domain);
                }
                (
                    Numeric::new(spec.bits, f, Interpretation::Signed)
                        .map_err(|_| Error::Domain)?,
                    Some(range.shift(shift)),
                )
            }
            arithmetic::Gate::Drelu => (
                Numeric::new(1, 0, Interpretation::Predicate).map_err(|_| Error::Domain)?,
                None,
            ),
        };
        if output.numeric() != expected {
            return Err(Error::Domain);
        }
        let party_bytes = size_of::<ArithmeticParty<'static, K>>() + common_charge();
        let bytes = arithmetic_reservation_bytes::<K>(spec)?;
        let mut reservation = match reservation {
            Some(lease) if lease.bytes == bytes && std::ptr::eq(lease.budget, self.budget) => lease,
            Some(_) => return Err(Error::Resource),
            None => self.budget.reserve(bytes)?,
        };
        let left_lease = reservation.split(party_bytes)?;
        let right_lease = reservation.split(party_bytes)?;
        let mask_lease = reservation.split(mask_charge())?;
        let binding = self.binding(output, next_consumer, interval)?;
        let op = digest(&[
            b"pllm.sigma.typed.arithmetic.v1\0",
            &operation,
            &input.binding.digest(),
            &binding.digest(),
        ]);
        let context = arithmetic::Context {
            plan: self.plan,
            session: self.session,
            operation: op,
            input_wire: descriptor.digest(),
            output_wire: output.digest(),
            first_tensor_index: descriptor.identity().first_tensor_index(),
        };
        let (masks, [left, right]) = arithmetic::issue_reserved_with(
            spec,
            context,
            input.backing.values(),
            reservation,
            fill,
        )?;
        Ok((
            DealerMask {
                backing: MaskBacking::Arithmetic(masks),
                binding,
                _kind: PhantomData,
                _lease: mask_lease,
            },
            [
                ArithmeticParty {
                    native: Some(left),
                    input: input.binding,
                    output: binding,
                    role: 0,
                    _kind: PhantomData,
                    lease: Some(left_lease),
                },
                ArithmeticParty {
                    native: Some(right),
                    input: input.binding,
                    output: binding,
                    role: 1,
                    _kind: PhantomData,
                    lease: Some(right_lease),
                },
            ],
        ))
    }
}

impl<'b, K> ArithmeticParty<'b, K> {
    pub fn cancel(&mut self) {
        self.native = None;
        retain_wrapper_metadata(&mut self.lease, size_of::<Self>());
    }
    pub fn begin(&mut self, input: CommonMasked<'b, Ring>) -> Result<arithmetic::Frame<'b>, Error> {
        let result = (|| {
            if input.binding != self.input || input.role != self.role {
                return Err(Error::Context);
            }
            self.native
                .as_mut()
                .ok_or(Error::Consumed)?
                .begin(input.backing.values())
        })();
        if result.is_err() {
            self.cancel();
        }
        result
    }
    pub fn advance(&mut self, peer: &[u8]) -> Result<Step<'b, K>, Error> {
        let result = (|| match self.native.as_mut().ok_or(Error::Consumed)?.advance(peer)? {
            arithmetic::Step::Frame(frame) => Ok(Step::Frame(frame)),
            arithmetic::Step::Output(output) => {
                let lease = self.lease.as_mut().ok_or(Error::Consumed)?;
                let lease = lease.split(lease.bytes - size_of::<Self>())?;
                // Keep completed native owner charged until wrapper Drop.
                Ok(Step::Output(CommonMasked {
                    backing: CommonBacking::Arithmetic(output),
                    binding: self.output,
                    role: self.role,
                    _kind: PhantomData,
                    _lease: lease,
                }))
            }
        })();
        if result.is_err() {
            self.cancel();
        }
        result
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod reserved_adapter_tests {
    use super::*;
    use pllm_core::edge_contract::{EdgeIdentity, Shape};

    fn source() -> SourceIdentity {
        SourceIdentity::new([11; 32], [12; 32], approved_implementation())
    }
    fn edge(id: u8, f: u8, private: bool) -> EdgeDescriptor {
        EdgeDescriptor::new(
            EdgeIdentity::new(source(), [1; 32], [id + 64; 32], [id; 32], 0),
            Shape::new(&[1, 1]).unwrap(),
            Numeric::new(50, f, Interpretation::Signed).unwrap(),
            Representation::CommonAdditiveMasked,
            private.then_some(WeightVisibility::Private),
        )
        .unwrap()
    }
    fn root<'b>(
        d: &mut LocalDealer<'b>,
        id: u8,
        private: bool,
        f: u8,
        value: i64,
    ) -> (DealerMask<'b, Ring>, [CommonMasked<'b, Ring>; 2]) {
        d.prepare_root(
            edge(id, f, private),
            BoundRange::new(-8192, 8192).unwrap(),
            [9; 32],
        )
        .unwrap()
        .encode(&[value])
        .unwrap()
    }
    fn spec() -> matrix::Spec {
        matrix::Spec {
            bits: 50,
            rows: 1,
            inner: 1,
            columns: 1,
            layout: matrix::MatrixLayout::InputOutput,
        }
    }

    #[test]
    fn typed_reserved_matrix_exact_short_foreign_and_fault_admission() {
        let _guard = matrix::TEST_SERIAL.lock().unwrap();
        let cost = matrix_reservation_bytes(spec()).unwrap();
        for delta in [-1isize, 0, 1, 2, 3] {
            let budget = Budget::new(64 << 20).unwrap();
            let foreign = Budget::new(cost).unwrap();
            let mut d = LocalDealer::new(source(), [1; 32], [2; 32], &budget).unwrap();
            let baseline = budget.used_bytes();
            let (x, xc) = root(&mut d, 3, false, 12, -4096);
            let (w, wc) = root(&mut d, 4, true, 12, 4096);
            let owner_budget = if delta == 2 { &foreign } else { &budget };
            let bytes = if delta == -1 {
                cost - 1
            } else if delta == 1 {
                cost + 1
            } else {
                cost
            };
            let lease = owner_budget.reserve(bytes).unwrap();
            // Occupy every spare byte: issuance must use original exact slice.
            let blocker = if delta == 0 {
                Some(budget.reserve((64 << 20) - budget.used_bytes()).unwrap())
            } else {
                None
            };
            let admitted = budget.used_bytes();
            let mut draws = 0;
            let result = d.matrix_reserved_with(
                [9; 32],
                spec(),
                x,
                w,
                edge(5, 24, false),
                [10; 32],
                MatrixRange::RequireSignedNoWrap,
                lease,
                &mut |bytes| {
                    assert_eq!(budget.used_bytes(), admitted);
                    draws += 1;
                    if delta == 3 {
                        return Err(Error::Randomness);
                    }
                    bytes.fill(7);
                    Ok(())
                },
            );
            if delta == 0 {
                let (mask, [mut a, mut b]) = result.unwrap();
                assert_eq!(draws, 5); // Output mask, nonce, A/B/C split draws.
                let [x0, x1] = xc;
                let [w0, w1] = wc;
                let af = a.begin(x0, w0, &mut || false).unwrap();
                let bf = b.begin(x1, w1, &mut || false).unwrap();
                let ao = a.finish(bf.as_bytes()).unwrap();
                let bo = b.finish(af.as_bytes()).unwrap();
                let expected = (-(4096i64 * 4096)) as u64 & super::super::mask(50);
                assert_eq!(
                    ao.backing.values()[0].wrapping_sub(mask.backing.values()[0])
                        & super::super::mask(50),
                    expected
                );
                assert_eq!(ao.backing.values(), bo.backing.values());
                drop((mask, ao, bo, bf));
                drop((a, b));
                // Actual frame alone retains native singleton and charge.
                assert!(matrix::party::issue_local(
                    spec(),
                    matrix::Context {
                        plan: [1; 32],
                        session: [2; 32],
                        operation: [3; 32],
                        input_wire: [4; 32],
                        weight_wire: [5; 32],
                        output_wire: [6; 32],
                    },
                    &[0],
                    &[0],
                    &[0],
                    &budget
                )
                .is_err());
                assert!(budget.used_bytes() > baseline);
                drop(af);
            } else {
                assert!(matches!(
                    result,
                    Err(Error::Resource) | Err(Error::Randomness)
                ));
                assert_eq!(draws, usize::from(delta == 3));
                drop((xc, wc));
            }
            assert_eq!(foreign.used_bytes(), 0);
            drop(blocker);
            assert_eq!(budget.used_bytes(), baseline);
        }
    }

    #[test]
    fn typed_reserved_arithmetic_exact_nonexact_and_retained_metadata() {
        let spec = arithmetic::Spec {
            bits: 50,
            lanes: 1,
            gate: arithmetic::Gate::GapArs { shift: 12 },
        };
        let cost = arithmetic_reservation_bytes::<Ring>(spec).unwrap();
        for delta in [-1isize, 0, 1, 2, 3] {
            let budget = Budget::new(64 << 20).unwrap();
            let foreign = Budget::new(cost).unwrap();
            let mut d = LocalDealer::new(source(), [1; 32], [2; 32], &budget).unwrap();
            let baseline = budget.used_bytes();
            let (mask, caps) = root(&mut d, 3, false, 24, -4097);
            let owner_budget = if delta == 2 { &foreign } else { &budget };
            let bytes = if delta == -1 {
                cost - 1
            } else if delta == 1 {
                cost + 1
            } else {
                cost
            };
            let lease = owner_budget.reserve(bytes).unwrap();
            let blocker = if delta == 0 {
                Some(budget.reserve((64 << 20) - budget.used_bytes()).unwrap())
            } else {
                None
            };
            let admitted = budget.used_bytes();
            let mut draws = 0;
            let result = d.gap_ars_reserved_with(
                [9; 32],
                mask,
                12,
                edge(5, 12, false),
                [10; 32],
                lease,
                &mut |bytes| {
                    assert_eq!(budget.used_bytes(), admitted);
                    draws += 1;
                    if delta == 3 {
                        return Err(Error::Randomness);
                    }
                    bytes.fill(7);
                    Ok(())
                },
            );
            if delta == 0 {
                let (mask, [mut a, mut b]) = result.unwrap();
                let [x, y] = caps;
                let af = a.begin(x).unwrap();
                let bf = b.begin(y).unwrap();
                let Step::Frame(af2) = a.advance(bf.as_bytes()).unwrap() else {
                    panic!()
                };
                let Step::Frame(bf2) = b.advance(af.as_bytes()).unwrap() else {
                    panic!()
                };
                let Step::Output(ao) = a.advance(bf2.as_bytes()).unwrap() else {
                    panic!()
                };
                let Step::Output(bo) = b.advance(af2.as_bytes()).unwrap() else {
                    panic!()
                };
                assert_eq!(
                    ao.backing.values()[0].wrapping_sub(mask.backing.values()[0])
                        & super::super::mask(50),
                    super::super::mask(50) - 1
                );
                assert_eq!(ao.backing.values(), bo.backing.values());
                drop((mask, ao, bo, af, bf, af2, bf2));
                assert!(
                    budget.used_bytes() - blocker.as_ref().map_or(0, |lease| lease.bytes)
                        >= baseline + 2 * size_of::<ArithmeticParty<'static, Ring>>()
                );
                a.cancel();
                b.cancel();
                assert_eq!(
                    budget.used_bytes() - blocker.as_ref().map_or(0, |lease| lease.bytes),
                    baseline + 2 * size_of::<ArithmeticParty<'static, Ring>>()
                );
                drop((a, b));
            } else {
                assert!(matches!(
                    result,
                    Err(Error::Resource) | Err(Error::Randomness)
                ));
                assert_eq!(draws, usize::from(delta == 3));
                drop(caps);
            }
            assert_eq!(foreign.used_bytes(), 0);
            drop(blocker);
            assert_eq!(budget.used_bytes(), baseline);
        }
    }
}
