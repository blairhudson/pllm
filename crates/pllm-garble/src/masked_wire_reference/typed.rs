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

// No production caller until enclosing state/source/handle admission exists.
#[allow(dead_code)]
pub(super) mod gapars_terminal;
pub use crate::shared_allocation::{Budget, Error};
use crate::{
    compact_dpf::gpu127::Profile,
    shared_allocation::{
        allocate, AllocationContext, Lease, LeaseFor, OwnedBudget, OwnedBudgetContext,
    },
};
use pllm_core::edge_contract::{
    EdgeDescriptor, Interpretation, Numeric, Representation, SourceIdentity, WeightVisibility,
};
use sha2::{Digest, Sha256};
use std::{
    marker::PhantomData,
    mem::{align_of, size_of},
};
use zeroize::Zeroizing;

mod activation;
mod causal_softmax;
mod softmax_coordinate;
pub use causal_softmax::{
    causal_softmax_bootstrap_bytes, causal_softmax_from_reserved, causal_softmax_resources,
    CausalSoftmaxCoordinatorFor, CausalSoftmaxCreatedFor, CausalSoftmaxError,
    CausalSoftmaxFrameFor, CausalSoftmaxInput, CausalSoftmaxMaskFor, CausalSoftmaxOutputFor,
    CausalSoftmaxPublicFloorWork, CausalSoftmaxResources, CausalSoftmaxRoleFor, CausalSoftmaxSpec,
    CausalSoftmaxStepFor, CausalSoftmaxWork, CausalSoftmaxWorkLimit,
};
mod internal_use;
pub(in crate::masked_wire_reference) mod layernorm4;
// D29's conditional implementation is not an activated/public product API.
mod nexp;
#[allow(dead_code)]
mod nexp_product;
mod nexp_stage_bounds;
pub use nexp::{
    clear_nexp_bootstrap_bytes, clear_nexp_from_reserved, clear_nexp_resources,
    ClearNexpCoordinatorFor, ClearNexpCreatedFor, ClearNexpError, ClearNexpFrameFor,
    ClearNexpInput, ClearNexpMaskFor, ClearNexpOutputFor, ClearNexpResources, ClearNexpRoleFor,
    ClearNexpSpec, ClearNexpStepFor, ClearNexpWork,
};
pub use nexp::{nexp_declaration_resources, validate_nexp_clear_root, NexpDeclarationResources};
mod linear_activation;
mod material;
mod region;
mod role_coordination;
mod selection_bridge;
pub use activation::{
    activation_resources, activation_resources_for, ActivationFrame, ActivationFrameFor,
    ActivationIssued, ActivationIssuedFor, ActivationKind, ActivationMetadataResources,
    ActivationMetadataWorkspace, ActivationParty, ActivationPartyFor,
    ActivationResourceCalculationResources, ActivationResources, ActivationSpec, ActivationStep,
    ActivationStepFor,
};
pub use linear_activation::{
    linear_activation_resources as local_linear_activation_resources,
    linear_activation_resources_for as local_linear_activation_resources_for,
    role_split_resources as local_linear_activation_role_split_resources,
    role_split_resources_for as local_linear_activation_role_split_resources_for,
    LinearActivationFrame as LocalLinearActivationFrame,
    LinearActivationFrameFor as LocalLinearActivationFrameFor,
    LinearActivationMetadataResources as LocalLinearActivationMetadataResources,
    LinearActivationResourceCalculationResources as LocalLinearActivationResourceCalculationResources,
    LinearActivationResources as LocalLinearActivationResources,
    LinearActivationRole as LocalLinearActivationRole,
    LinearActivationRoleCoordinator as LocalLinearActivationRoleCoordinator,
    LinearActivationRoleCoordinatorFor as LocalLinearActivationRoleCoordinatorFor,
    LinearActivationRoleFor as LocalLinearActivationRoleFor,
    LinearActivationRoleFrame as LocalLinearActivationRoleFrame,
    LinearActivationRoleFrameFor as LocalLinearActivationRoleFrameFor,
    LinearActivationRoleOutput as LocalLinearActivationRoleOutput,
    LinearActivationRoleOutputFor as LocalLinearActivationRoleOutputFor,
    LinearActivationRoleSplitResources as LocalLinearActivationRoleSplitResources,
    LinearActivationRoleStep as LocalLinearActivationRoleStep,
    LinearActivationRoleStepFor as LocalLinearActivationRoleStepFor,
    LinearActivationRun as LocalLinearActivationRun,
    LinearActivationRunFor as LocalLinearActivationRunFor,
    LinearActivationSpec as LocalLinearActivationSpec,
    LinearActivationStep as LocalLinearActivationStep,
    LinearActivationStepFor as LocalLinearActivationStepFor,
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
fn retain_wrapper_metadata<C: AllocationContext>(lease: &mut Option<LeaseFor<C>>, bytes: usize) {
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
enum CommonBackingFor<C: AllocationContext> {
    Root(Zeroizing<Vec<u64>>),
    Arithmetic(arithmetic::MaskedOutputFor<C>),
    Matrix(matrix::party::MaskedOutputFor<C>),
    Selection(selection::MaskedOutputFor<C>),
    Elementwise(super::elementwise_product::OutputFor<C>),
}
impl<C: AllocationContext> CommonBackingFor<C> {
    fn values(&self) -> &[u64] {
        match self {
            Self::Root(v) => v,
            Self::Arithmetic(v) => v.values(),
            Self::Matrix(v) => v.values(),
            Self::Selection(v) => v.values(),
            Self::Elementwise(v) => v.values(),
        }
    }
}
enum MaskBackingFor<C: AllocationContext> {
    Fresh(Zeroizing<Vec<u64>>),
    Arithmetic(arithmetic::OutputMasksFor<C>),
    Selection(selection::OutputMasksFor<C>),
    Elementwise(super::elementwise_product::MasksFor<C>),
}
impl<C: AllocationContext> MaskBackingFor<C> {
    fn values(&self) -> &[u64] {
        match self {
            Self::Fresh(v) => v,
            Self::Arithmetic(v) => v.values(),
            Self::Selection(v) => v.values(),
            Self::Elementwise(v) => v.values(),
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
pub struct CommonMaskedFor<C: AllocationContext, K> {
    backing: CommonBackingFor<C>,
    binding: Binding,
    role: u8,
    _kind: PhantomData<K>,
    // Always last: payload zeroizes and deallocates before budget refund.
    _lease: LeaseFor<C>,
}
// Private finished capability; no constructor/accessor, runtime integration or bridge.
// Native fields erase/refund/release before the original typed lease is refunded.
struct OpaqueTyped<C: AllocationContext> {
    native: matrix::party::OpaqueNative<C>,
    binding: Binding,
    role: u8,
    kind: PhantomData<Ring>,
    typed_lease: LeaseFor<C>,
}

impl<C: AllocationContext, K> CommonMaskedFor<C, K> {
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
pub struct DealerMaskFor<C: AllocationContext, K> {
    backing: MaskBackingFor<C>,
    binding: Binding,
    _kind: PhantomData<K>,
    _lease: LeaseFor<C>,
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
pub struct LocalDealerFor<C: AllocationContext> {
    source: SourceIdentity,
    plan: [u8; 32],
    session: [u8; 32],
    namespace: [u8; 32],
    counter: u64,
    budget: C,
    _lease: LeaseFor<C>,
}

/// Prepared source-bound promise and fresh masks, created before receiving
/// private values. Failure in encode burns all three owners.
pub struct RootInputFor<C: AllocationContext> {
    dealer: DealerMaskFor<C, Ring>,
    common: [CommonMaskedFor<C, Ring>; 2],
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
fn common_charge_for<C: AllocationContext>() -> usize {
    size_of::<CommonMaskedFor<C, Ring>>()
}
fn mask_charge_for<C: AllocationContext>() -> usize {
    size_of::<DealerMaskFor<C, Ring>>()
}

impl<C: AllocationContext> LocalDealerFor<C> {
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
    ) -> Result<RootInputFor<C>, Error> {
        let bytes = self.root_reservation_bytes(descriptor, &promise)?;
        let reservation = LeaseFor::reserve(&self.budget, bytes)?;
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
        root_geometry_bytes::<C>(descriptor, promise)
    }

    /// Consume an exact original aggregate slice, without refund/reacquire.
    /// Callers accepting clear inputs validate them before invoking this helper.
    fn prepare_root_reserved_with(
        &mut self,
        descriptor: EdgeDescriptor,
        promise: BoundRange<LocalAssumption>,
        consumer: [u8; 32],
        mut lease: LeaseFor<C>,
        fill: impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<RootInputFor<C>, Error> {
        let bytes = self.root_reservation_bytes(descriptor, &promise)?;
        if lease.bytes != bytes || !std::ptr::eq(lease.budget.budget(), self.budget.budget()) {
            return Err(Error::Resource);
        }
        let interval = Interval {
            lower: promise.lower,
            upper: promise.upper,
        };
        let lanes = descriptor.shape().elements();
        let common_bytes = 8 * lanes + ALLOCATOR + common_charge_for::<C>();
        let left_lease = lease.split(common_bytes)?;
        let right_lease = lease.split(common_bytes)?;
        let binding = self.binding(descriptor, consumer, Some(interval))?;
        let mut masks = zeros(lanes, self.budget.budget())?;
        let left = zeros(lanes, self.budget.budget())?;
        let right = zeros(lanes, self.budget.budget())?;
        fill_masks_with(&mut masks, descriptor.numeric().ring_bits(), fill)?;
        Ok(RootInputFor {
            dealer: DealerMaskFor {
                backing: MaskBackingFor::Fresh(masks),
                binding,
                _kind: PhantomData,
                _lease: lease,
            },
            common: [
                CommonMaskedFor {
                    backing: CommonBackingFor::Root(left),
                    binding,
                    role: 0,
                    _kind: PhantomData,
                    _lease: left_lease,
                },
                CommonMaskedFor {
                    backing: CommonBackingFor::Root(right),
                    binding,
                    role: 1,
                    _kind: PhantomData,
                    _lease: right_lease,
                },
            ],
        })
    }
}

// Original geometry quotation, shared with pre-namespace aggregate admission.
fn root_geometry_bytes<C: AllocationContext>(
    descriptor: EdgeDescriptor,
    promise: &BoundRange<LocalAssumption>,
) -> Result<usize, Error> {
    if !(Interval {
        lower: promise.lower,
        upper: promise.upper,
    })
    .fits(descriptor.numeric().ring_bits())
    {
        return Err(Error::Domain);
    }
    root_payload_bytes::<C>(descriptor.shape().elements())
}
// Original storage formula also usable BEFORE source/namespace/hash creation.
// It is a byte quote, never a descriptor/source/input capability constructor.
fn root_payload_bytes<C: AllocationContext>(lanes: usize) -> Result<usize, Error> {
    let payload = lanes
        .checked_mul(8)
        .and_then(|v| v.checked_add(ALLOCATOR))
        .ok_or(Error::Resource)?;
    let common = payload
        .checked_add(common_charge_for::<C>())
        .ok_or(Error::Resource)?;
    let dealer = payload
        .checked_add(mask_charge_for::<C>())
        .and_then(|v| v.checked_add(size_of::<RootInputFor<C>>()))
        .ok_or(Error::Resource)?;
    common
        .checked_mul(2)
        .and_then(|v| v.checked_add(dealer))
        .ok_or(Error::Resource)
}
type EncodedRootFor<C> = (DealerMaskFor<C, Ring>, [CommonMaskedFor<C, Ring>; 2]);
type MatrixIssuedFor<C> = (DealerMaskFor<C, Ring>, [MatrixPartyFor<C>; 2]);
type ArithmeticIssuedFor<C, K> = (DealerMaskFor<C, K>, [ArithmeticPartyFor<C, K>; 2]);

impl<C: AllocationContext> RootInputFor<C> {
    /// Validate against the predeclared interval, never derive public metadata
    /// from values. Input is borrowed transiently, not copied or retained clear.
    pub fn encode(mut self, values: &[i64]) -> Result<EncodedRootFor<C>, Error> {
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
            let CommonBackingFor::Root(out) = &mut common.backing else {
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
/// Owning accounting does not authorize the borrowed local material codec.
///
/// ```compile_fail
/// use pllm_garble::{masked_wire_reference::typed::MatrixPartyFor,
///     shared_allocation::OwnedBudgetContext};
/// fn no_owned_codec(party: MatrixPartyFor<OwnedBudgetContext>) {
///     let _ = party.export_material();
/// }
/// ```
pub struct MatrixPartyFor<C: AllocationContext> {
    native: Option<matrix::party::PartyFor<C>>,
    inputs: [Binding; 2],
    output: Binding,
    operation: [u8; 32],
    role: u8,
    imported: bool,
    lease: Option<LeaseFor<C>>,
    // Additional handoff metadata charge follows native secrets and original lease.
    material_lease: Option<LeaseFor<C>>,
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
pub(crate) fn matrix_reservation_bytes_for<C: AllocationContext>(
    spec: matrix::Spec,
) -> Result<usize, Error> {
    let native = match matrix::party::resources_for_context::<C>(spec) {
        Ok(value) => value,
        Err(error) => return Err(error),
    };
    let dt = match MatrixPartyFor::<C>::FINISHED_TRANSFER_BYTES {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let t00 = match size_of::<MatrixPartyFor<C>>().checked_add(size_of::<CommonMaskedFor<C, Ring>>()) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let t01 = match t00.checked_add(dt) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let t02 = match 2usize.checked_mul(t01) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let t03 = match native.reservation_bytes.checked_add(t02) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let t04 = match 8usize.checked_mul(native.output_elements) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let t05 = match t03.checked_add(t04) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let t06 = match t05.checked_add(ALLOCATOR) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    let t07 = match t06.checked_add(size_of::<DealerMaskFor<C, Ring>>()) {
        Some(value) => value,
        None => return Err(Error::Resource),
    };
    Ok(t07)
}

/// Exact incremental admission for typed arithmetic, excluding original input.
pub(crate) fn arithmetic_reservation_bytes_for<C: AllocationContext, K>(
    spec: arithmetic::Spec,
) -> Result<usize, Error> {
    Ok(
        arithmetic::resources_for_context::<C>(spec)?.combined_reservation_bytes
            + 2 * (size_of::<ArithmeticPartyFor<C, K>>() + common_charge_for::<C>())
            + mask_charge_for::<C>(),
    )
}

impl<C: AllocationContext> LocalDealerFor<C> {
    /// Private-weight matrix only. `output` Q must equal fx+fw exactly. Native
    /// singleton/256-element/2^20 complete-MAC admission is preserved unchanged.
    // Keep operation, operands, result edge and range policy explicit at issuance.
    #[allow(clippy::too_many_arguments)]
    pub fn matrix(
        &mut self,
        operation: [u8; 32],
        spec: matrix::Spec,
        x: DealerMaskFor<C, Ring>,
        w: DealerMaskFor<C, Ring>,
        output: EdgeDescriptor,
        next_consumer: [u8; 32],
        range: MatrixRange,
    ) -> Result<MatrixIssuedFor<C>, Error> {
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
        x: DealerMaskFor<C, Ring>,
        w: DealerMaskFor<C, Ring>,
        output: EdgeDescriptor,
        next_consumer: [u8; 32],
        range: MatrixRange,
        reservation: LeaseFor<C>,
        fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<MatrixIssuedFor<C>, Error> {
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
        x: DealerMaskFor<C, Ring>,
        w: DealerMaskFor<C, Ring>,
        output: EdgeDescriptor,
        next_consumer: [u8; 32],
        range: MatrixRange,
        reservation: Option<LeaseFor<C>>,
        fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<MatrixIssuedFor<C>, Error> {
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
        x: DealerMaskFor<C, Ring>,
        w: DealerMaskFor<C, Ring>,
        output: EdgeDescriptor,
        next_consumer: [u8; 32],
        range: MatrixRange,
        reservation: LeaseFor<C>,
        ticket: matrix::party::IssuanceTicket,
        fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<MatrixIssuedFor<C>, Error> {
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
        x: DealerMaskFor<C, Ring>,
        w: DealerMaskFor<C, Ring>,
        output: EdgeDescriptor,
        next_consumer: [u8; 32],
        range: MatrixRange,
        reservation: Option<LeaseFor<C>>,
        ticket: Option<matrix::party::IssuanceTicket>,
        fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<MatrixIssuedFor<C>, Error> {
        self.input(x.binding, operation)?;
        self.input(w.binding, operation)?;
        self.validate(output, false)?;
        let xd = x.binding.descriptor;
        let wd = w.binding.descriptor;
        if wd.weight_visibility() != Some(WeightVisibility::Private) {
            return Err(Error::Unsupported);
        }
        let native_cost = matrix::party::resources_for_context::<C>(spec)?;
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
        let bytes = matrix_reservation_bytes_for::<C>(spec)?;
        // Same enlarged birth lease follows ordinary G or opaque T without a split
        // at conversion. The total quote has already checked every addition.
        let dt = match MatrixPartyFor::<C>::FINISHED_TRANSFER_BYTES {
            Some(value) => value,
            None => return Err(Error::Resource),
        };
        let party_bytes = size_of::<MatrixPartyFor<C>>() + common_charge_for::<C>() + dt;
        let mask_bytes = 8 * native_cost.output_elements + ALLOCATOR + mask_charge_for::<C>();
        let mut reservation = match reservation {
            Some(lease)
                if lease.bytes == bytes
                    && std::ptr::eq(lease.budget.budget(), self.budget.budget()) =>
            {
                lease
            }
            Some(_) => return Err(Error::Resource),
            None => LeaseFor::reserve(&self.budget, bytes)?,
        };
        let left_lease = reservation.split(party_bytes)?;
        let right_lease = reservation.split(party_bytes)?;
        let mask_lease = reservation.split(mask_bytes)?;
        let binding = self.binding(output, next_consumer, interval)?;
        let mut masks = zeros(native_cost.output_elements, self.budget.budget())?;
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
            DealerMaskFor {
                backing: MaskBackingFor::Fresh(masks),
                binding,
                _kind: PhantomData,
                _lease: mask_lease,
            },
            [
                MatrixPartyFor {
                    native: Some(left),
                    inputs,
                    output: binding,
                    operation,
                    role: 0,
                    imported: false,
                    lease: Some(left_lease),
                    material_lease: None,
                },
                MatrixPartyFor {
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

impl<C: AllocationContext> MatrixPartyFor<C> {
    // Checked, layout-aware, compile-time slot inventories; no accounting cycle.
    // Four Result<G, E> slots: closure, named result, return, opaque receiver.
    // Separate H(N) prices the named predecessor, not the N embedded in H(T).
    const FINISHED_TRANSFER_BYTES: Option<usize> = 'inventory: {
        let padding0 = match align_of::<Result<CommonMaskedFor<C, Ring>, Error>>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h0 = match size_of::<Result<CommonMaskedFor<C, Ring>, Error>>().checked_add(padding0) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let p0 = match h0.checked_mul(4) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s0 = p0;
        let padding1 = match align_of::<CommonMaskedFor<C, Ring>>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h1 = match size_of::<CommonMaskedFor<C, Ring>>().checked_add(padding1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s1 = match s0.checked_add(h1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let padding2 = match align_of::<CommonBackingFor<C>>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h2 = match size_of::<CommonBackingFor<C>>().checked_add(padding2) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s2 = match s1.checked_add(h2) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let padding3 = match align_of::<matrix::party::MaskedOutputFor<C>>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h3 = match size_of::<matrix::party::MaskedOutputFor<C>>().checked_add(padding3) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s3 = match s2.checked_add(h3) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let padding4 = match align_of::<matrix::party::OpaqueNative<C>>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h4 = match size_of::<matrix::party::OpaqueNative<C>>().checked_add(padding4) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s4 = match s3.checked_add(h4) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let padding5 = match align_of::<OpaqueTyped<C>>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h5 = match size_of::<OpaqueTyped<C>>().checked_add(padding5) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let p5 = match h5.checked_mul(2) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s5 = match s4.checked_add(p5) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let padding6 = match align_of::<Result<OpaqueTyped<C>, Error>>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h6 = match size_of::<Result<OpaqueTyped<C>, Error>>().checked_add(padding6) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let p6 = match h6.checked_mul(2) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s6 = match s5.checked_add(p6) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let padding7 = match align_of::<Binding>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h7 = match size_of::<Binding>().checked_add(padding7) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s7 = match s6.checked_add(h7) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let padding8 = match align_of::<u8>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h8 = match size_of::<u8>().checked_add(padding8) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s8 = match s7.checked_add(h8) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let padding9 = match align_of::<PhantomData<Ring>>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h9 = match size_of::<PhantomData<Ring>>().checked_add(padding9) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s9 = match s8.checked_add(h9) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let padding10 = match align_of::<LeaseFor<C>>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h10 = match size_of::<LeaseFor<C>>().checked_add(padding10) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s10 = match s9.checked_add(h10) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let padding11 = match align_of::<Error>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h11 = match size_of::<Error>().checked_add(padding11) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s11 = match s10.checked_add(h11) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let padding12 = match align_of::<&mut MatrixPartyFor<C>>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h12 = match size_of::<&mut MatrixPartyFor<C>>().checked_add(padding12) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s12 = match s11.checked_add(h12) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let padding13 = match align_of::<&[u8]>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h13 = match size_of::<&[u8]>().checked_add(padding13) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s13 = match s12.checked_add(h13) {
            Some(value) => value,
            None => break 'inventory None,
        };
        Some(s13)
    };
    // 8 nodes: 16 operand U + 8 named U + dt = 25 U; 8 Options + read = 9;
    // 8 node errors + dt read + native forwarding = 10.
    const TYPED_CALC: Option<usize> = 'inventory: {
        let s0 = match matrix::party::PartyFor::<C>::NATIVE_CALC { Some(value) => value, None => break 'inventory None };
        let padding1 = match align_of::<matrix::Spec>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h1 = match size_of::<matrix::Spec>().checked_add(padding1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let p1 = match h1.checked_mul(2) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s1 = match s0.checked_add(p1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let padding2 = match align_of::<matrix::party::Resources>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h2 = match size_of::<matrix::party::Resources>().checked_add(padding2) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s2 = match s1.checked_add(h2) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let padding3 = match align_of::<usize>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h3 = match size_of::<usize>().checked_add(padding3) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let p3 = match h3.checked_mul(25) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s3 = match s2.checked_add(p3) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let padding4 = match align_of::<Option<usize>>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h4 = match size_of::<Option<usize>>().checked_add(padding4) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let p4 = match h4.checked_mul(9) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s4 = match s3.checked_add(p4) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let padding5 = match align_of::<Error>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h5 = match size_of::<Error>().checked_add(padding5) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let p5 = match h5.checked_mul(10) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s5 = match s4.checked_add(p5) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let padding6 = match align_of::<Result<usize, Error>>().checked_sub(1) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let h6 = match size_of::<Result<usize, Error>>().checked_add(padding6) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let p6 = match h6.checked_mul(4) {
            Some(value) => value,
            None => break 'inventory None,
        };
        let s6 = match s5.checked_add(p6) {
            Some(value) => value,
            None => break 'inventory None,
        };
        Some(s6)
    };

    fn finish_opaque(&mut self, peer: &[u8]) -> Result<OpaqueTyped<C>, Error> {
        let common: CommonMaskedFor<C, Ring> = match self.finish(peer) {
            Ok(value) => value,
            Err(error) => return Err(error),
        };
        let CommonMaskedFor { backing, binding, role, _kind: kind, _lease: typed_lease } = common;
        match backing {
            CommonBackingFor::Matrix(original) => {
                let native: matrix::party::OpaqueNative<C> = original.into_opaque();
                let out: OpaqueTyped<C> = OpaqueTyped { native, binding, role, kind, typed_lease };
                Ok(out)
            }
            _ => {
                // Complete secret-bearing owner retires before typed refund.
                drop(backing);
                drop(typed_lease);
                Err(Error::Context)
            }
        }
    }

    pub fn cancel(&mut self) {
        self.native = None;
        retain_wrapper_metadata(&mut self.lease, size_of::<Self>());
        self.material_lease = None;
    }
    /// Consumes both evaluator capabilities, even on wrong role/mask/context.
    pub fn begin(
        &mut self,
        x: CommonMaskedFor<C, Ring>,
        w: CommonMaskedFor<C, Ring>,
        cancelled: &mut impl FnMut() -> bool,
    ) -> Result<matrix::party::FrameFor<C>, Error> {
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
    pub fn finish(&mut self, peer: &[u8]) -> Result<CommonMaskedFor<C, Ring>, Error> {
        let result = (|| {
            let output = self.native.as_mut().ok_or(Error::Consumed)?.finish(peer)?;
            let lease = self.lease.as_mut().ok_or(Error::Consumed)?;
            let lease = lease.split(lease.bytes - size_of::<Self>())?;
            // Native completed owner retains its own inline metadata charge.
            self.material_lease = None;
            Ok(CommonMaskedFor {
                backing: CommonBackingFor::Matrix(output),
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

pub struct ArithmeticPartyFor<C: AllocationContext, K> {
    native: Option<arithmetic::PartyFor<C>>,
    input: Binding,
    output: Binding,
    role: u8,
    _kind: PhantomData<K>,
    lease: Option<LeaseFor<C>>,
}
// Keep native output ownership inline: boxing adds an unpriced, infallible allocation.
#[allow(clippy::large_enum_variant)]
pub enum StepFor<C: AllocationContext, K> {
    Frame(arithmetic::FrameFor<C>),
    Output(CommonMaskedFor<C, K>),
}

impl<C: AllocationContext> LocalDealerFor<C> {
    pub fn tr(
        &mut self,
        operation: [u8; 32],
        input: DealerMaskFor<C, Ring>,
        shift: u8,
        output: EdgeDescriptor,
        next_consumer: [u8; 32],
    ) -> Result<ArithmeticIssuedFor<C, Ring>, Error> {
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
        input: DealerMaskFor<C, Ring>,
        shift: u8,
        output: EdgeDescriptor,
        next_consumer: [u8; 32],
    ) -> Result<ArithmeticIssuedFor<C, Ring>, Error> {
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
        input: DealerMaskFor<C, Ring>,
        output: EdgeDescriptor,
        next_consumer: [u8; 32],
    ) -> Result<ArithmeticIssuedFor<C, Predicate>, Error> {
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
        input: DealerMaskFor<C, Ring>,
        gate: arithmetic::Gate,
        output: EdgeDescriptor,
        next_consumer: [u8; 32],
        predicate: bool,
    ) -> Result<ArithmeticIssuedFor<C, K>, Error> {
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
        input: DealerMaskFor<C, Ring>,
        shift: u8,
        output: EdgeDescriptor,
        next_consumer: [u8; 32],
        reservation: LeaseFor<C>,
        fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<ArithmeticIssuedFor<C, Ring>, Error> {
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
        input: DealerMaskFor<C, Ring>,
        shift: u8,
        output: EdgeDescriptor,
        next_consumer: [u8; 32],
        reservation: LeaseFor<C>,
        fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<ArithmeticIssuedFor<C, Ring>, Error> {
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
        input: DealerMaskFor<C, Ring>,
        output: EdgeDescriptor,
        next_consumer: [u8; 32],
        reservation: LeaseFor<C>,
        fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<ArithmeticIssuedFor<C, Predicate>, Error> {
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
        input: DealerMaskFor<C, Ring>,
        gate: arithmetic::Gate,
        output: EdgeDescriptor,
        next_consumer: [u8; 32],
        predicate: bool,
        reservation: LeaseFor<C>,
        fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<ArithmeticIssuedFor<C, K>, Error> {
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
        input: DealerMaskFor<C, Ring>,
        gate: arithmetic::Gate,
        output: EdgeDescriptor,
        next_consumer: [u8; 32],
        predicate: bool,
        reservation: Option<LeaseFor<C>>,
        fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<ArithmeticIssuedFor<C, K>, Error> {
        self.arithmetic_with_borrowed(
            operation,
            &input,
            gate,
            output,
            next_consumer,
            predicate,
            reservation,
            fill,
        )
    }

    // D43 retains original producer masks through outer retirement. The legacy
    // consuming entry above still drops its input at precisely the old return.
    #[allow(clippy::too_many_arguments)]
    fn arithmetic_with_borrowed<K>(
        &mut self,
        operation: [u8; 32],
        input: &DealerMaskFor<C, Ring>,
        gate: arithmetic::Gate,
        output: EdgeDescriptor,
        next_consumer: [u8; 32],
        predicate: bool,
        reservation: Option<LeaseFor<C>>,
        fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<ArithmeticIssuedFor<C, K>, Error> {
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
        arithmetic::resources_for_context::<C>(spec)?;
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
        let party_bytes = size_of::<ArithmeticPartyFor<C, K>>() + common_charge_for::<C>();
        let bytes = arithmetic_reservation_bytes_for::<C, K>(spec)?;
        let mut reservation = match reservation {
            Some(lease)
                if lease.bytes == bytes
                    && std::ptr::eq(lease.budget.budget(), self.budget.budget()) =>
            {
                lease
            }
            Some(_) => return Err(Error::Resource),
            None => LeaseFor::reserve(&self.budget, bytes)?,
        };
        let left_lease = reservation.split(party_bytes)?;
        let right_lease = reservation.split(party_bytes)?;
        let mask_lease = reservation.split(mask_charge_for::<C>())?;
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
            DealerMaskFor {
                backing: MaskBackingFor::Arithmetic(masks),
                binding,
                _kind: PhantomData,
                _lease: mask_lease,
            },
            [
                ArithmeticPartyFor {
                    native: Some(left),
                    input: input.binding,
                    output: binding,
                    role: 0,
                    _kind: PhantomData,
                    lease: Some(left_lease),
                },
                ArithmeticPartyFor {
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

impl<C: AllocationContext, K> ArithmeticPartyFor<C, K> {
    pub fn cancel(&mut self) {
        self.native = None;
        retain_wrapper_metadata(&mut self.lease, size_of::<Self>());
    }
    pub fn begin(
        &mut self,
        input: CommonMaskedFor<C, Ring>,
    ) -> Result<arithmetic::FrameFor<C>, Error> {
        self.begin_borrowed(&input)
    }
    // Existing native entry; the closed parent keeps the actual producer owner.
    fn begin_borrowed(
        &mut self,
        input: &CommonMaskedFor<C, Ring>,
    ) -> Result<arithmetic::FrameFor<C>, Error> {
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
    pub fn advance(&mut self, peer: &[u8]) -> Result<StepFor<C, K>, Error> {
        let result = (|| match self.native.as_mut().ok_or(Error::Consumed)?.advance(peer)? {
            arithmetic::StepFor::Frame(frame) => Ok(StepFor::Frame(frame)),
            arithmetic::StepFor::Output(output) => {
                let lease = self.lease.as_mut().ok_or(Error::Consumed)?;
                let lease = lease.split(lease.bytes - size_of::<Self>())?;
                // Keep completed native owner charged until wrapper Drop.
                Ok(StepFor::Output(CommonMaskedFor {
                    backing: CommonBackingFor::Arithmetic(output),
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

type CommonBacking<'b> = CommonBackingFor<&'b Budget>;
type MaskBacking<'b> = MaskBackingFor<&'b Budget>;
pub type CommonMasked<'b, K> = CommonMaskedFor<&'b Budget, K>;
pub type CommonXorMasked<'b, K> = CommonMasked<'b, K>;
pub type DealerMask<'b, K> = DealerMaskFor<&'b Budget, K>;
pub type LocalDealer<'b> = LocalDealerFor<&'b Budget>;
pub type RootInput<'b> = RootInputFor<&'b Budget>;
pub type MatrixParty<'b> = MatrixPartyFor<&'b Budget>;
pub type ArithmeticParty<'b, K> = ArithmeticPartyFor<&'b Budget, K>;
pub type Step<'b, K> = StepFor<&'b Budget, K>;
pub type OwnedLocalDealer = LocalDealerFor<OwnedBudgetContext>;
pub type OwnedDealerMask<K> = DealerMaskFor<OwnedBudgetContext, K>;
pub type OwnedCommonMasked<K> = CommonMaskedFor<OwnedBudgetContext, K>;

fn common_charge() -> usize {
    common_charge_for::<&Budget>()
}
fn mask_charge() -> usize {
    mask_charge_for::<&Budget>()
}
#[cfg(test)]
pub(crate) fn matrix_reservation_bytes(spec: matrix::Spec) -> Result<usize, Error> {
    matrix_reservation_bytes_for::<&Budget>(spec)
}
#[cfg(test)]
pub(crate) fn arithmetic_reservation_bytes<K>(spec: arithmetic::Spec) -> Result<usize, Error> {
    arithmetic_reservation_bytes_for::<&Budget, K>(spec)
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
        Self::from_reserved_namespace(
            source,
            plan,
            session,
            budget.reserve(size_of::<Self>() + (4 << 10))?,
        )
    }
}
impl OwnedLocalDealer {
    pub fn new(
        source: SourceIdentity,
        plan: [u8; 32],
        session: [u8; 32],
        budget: &OwnedBudget,
    ) -> Result<Self, Error> {
        if source.implementation() != approved_implementation() {
            return Err(Error::Unsupported);
        }
        Self::from_reserved_namespace(
            source,
            plan,
            session,
            budget.reserve_native(size_of::<Self>() + (4 << 10))?,
        )
    }
}
impl<C: AllocationContext> LocalDealerFor<C> {
    fn from_reserved_namespace(
        source: SourceIdentity,
        plan: [u8; 32],
        session: [u8; 32],
        lease: LeaseFor<C>,
    ) -> Result<Self, Error> {
        let mut namespace = [0; 32];
        getrandom::fill(&mut namespace).map_err(|_| Error::Randomness)?;
        Ok(Self {
            source,
            plan,
            session,
            namespace,
            counter: 0,
            budget: lease.budget.share(),
            _lease: lease,
        })
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

    #[test]
    fn owned_opaque_typed_survives_root_retirement() {
        let _serial = matrix::TEST_SERIAL.lock().unwrap();
        let budget = OwnedBudget::new(64 << 20).unwrap();
        let weak = budget.test_counter_weak();
        let mut dealer = OwnedLocalDealer::new(source(), [1; 32], [2; 32], &budget).unwrap();
        let (x, [x0, x1]) = dealer
            .prepare_root(
                edge(3, 12, false),
                BoundRange::new(-8192, 8192).unwrap(),
                [9; 32],
            )
            .unwrap()
            .encode(&[-4096])
            .unwrap();
        let (w, [w0, w1]) = dealer
            .prepare_root(
                edge(4, 12, true),
                BoundRange::new(-8192, 8192).unwrap(),
                [9; 32],
            )
            .unwrap()
            .encode(&[4096])
            .unwrap();
        let original = budget
            .reserve_native(matrix_reservation_bytes_for::<OwnedBudgetContext>(spec()).unwrap())
            .unwrap();
        let (mask, [mut a, mut b]) = dealer
            .matrix_reserved_with(
                [9; 32], spec(), x, w, edge(5, 24, false), [10; 32],
                MatrixRange::RequireSignedNoWrap, original,
                &mut |bytes| { bytes.fill(7); Ok(()) },
            )
            .unwrap();
        let af = a.begin(x0, w0, &mut || false).unwrap();
        let bf = b.begin(x1, w1, &mut || false).unwrap();
        let root_counter = budget.budget() as *const Budget;
        let typed_counter = a.lease.as_ref().unwrap().budget.budget() as *const Budget;
        let original_bytes = a.lease.as_ref().unwrap().bytes;
        let binding = a.output;
        let role = a.role;
        assert_eq!(typed_counter, root_counter);
        let opaque = a.finish_opaque(bf.as_bytes()).unwrap();
        let ordinary = b.finish(af.as_bytes()).unwrap();
        assert_eq!(opaque.typed_lease.budget.budget() as *const Budget, typed_counter);
        assert_eq!(
            opaque.typed_lease.bytes,
            original_bytes - size_of::<MatrixPartyFor<OwnedBudgetContext>>(),
        );
        assert_eq!(opaque.binding, binding);
        assert_eq!(opaque.role, role);

        drop(dealer);
        drop(budget);
        drop((mask, a, b, af, bf, ordinary));
        let observer = weak.upgrade().unwrap();
        assert!(std::ptr::eq(opaque.typed_lease.budget.budget(), observer.as_ref()));
        assert_eq!(opaque.typed_lease.budget.budget() as *const Budget, root_counter);
        drop(observer);
        drop(opaque);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn opaque_finish_retains_original_typed_lease_and_forwards_failure() {
        let _serial = matrix::TEST_SERIAL.lock().unwrap();
        for invalid in [false, true] {
            let budget = Budget::new(64 << 20).unwrap();
            let mut dealer = LocalDealer::new(source(), [1; 32], [2; 32], &budget).unwrap();
            let baseline = budget.used_bytes();
            let (x, [x0, x1]) = root(&mut dealer, 3, false, 12, -4096);
            let (w, [w0, w1]) = root(&mut dealer, 4, true, 12, 4096);
            let original = budget.reserve(matrix_reservation_bytes(spec()).unwrap()).unwrap();
            let (mask, [mut a, mut b]) = dealer.matrix_reserved_with(
                [9; 32], spec(), x, w, edge(5, 24, false), [10; 32],
                MatrixRange::RequireSignedNoWrap, original,
                &mut |bytes| { bytes.fill(7); Ok(()) },
            ).unwrap();
            let af = a.begin(x0, w0, &mut || false).unwrap();
            let bf = b.begin(x1, w1, &mut || false).unwrap();
            let expected_lease = a.lease.as_ref().unwrap().bytes - size_of::<MatrixParty<'_>>();
            let expected_binding = a.output;
            let used = budget.used_bytes();
            // Any attempted allocation after begin would fail.
            budget.fail_after.store(0, std::sync::atomic::Ordering::SeqCst);
            if invalid {
                assert!(matches!(a.finish_opaque(&[]), Err(Error::Domain)));
                assert!(a.native.is_none());
                assert_eq!(a.lease.as_ref().unwrap().bytes, size_of::<MatrixParty<'_>>());
                b.cancel();
            } else {
                let opaque = a.finish_opaque(bf.as_bytes()).unwrap();
                assert_eq!(opaque.typed_lease.bytes, expected_lease);
                assert!(std::ptr::eq(opaque.typed_lease.budget, &budget));
                assert_eq!(opaque.binding, expected_binding);
                assert_eq!(opaque.role, 0);
                assert_eq!(budget.used_bytes(), used);
                let ordinary = b.finish(af.as_bytes()).unwrap();
                assert_eq!(ordinary._lease.bytes, expected_lease);
                drop((opaque, ordinary));
            }
            drop((a, b, af, bf, mask));
            assert_eq!(budget.used_bytes(), baseline);
        }
    }
}


#[cfg(test)]
mod opaque_finished_tests {
    use super::*;

    #[test]
    fn current_typed_slot_funding() {
        use crate::shared_allocation::MAX_ALLOCATION_BYTES;
        let _serial = matrix::TEST_SERIAL.lock().unwrap();
        fn layout<T>(label: &str) -> usize {
            println!("{label}: {} size={} align={}", std::any::type_name::<T>(),
                size_of::<T>(), align_of::<T>());
            size_of::<T>()
        }
        #[derive(Default)]
        struct Ledger { bytes: usize }
        impl Ledger {
            fn bytes(&mut self, label: &str, bytes: usize) {
                println!("slot {label}: {bytes}");
                self.bytes = self.bytes.checked_add(bytes).expect(label);
            }
            fn slot<T>(&mut self, label: &str) {
                self.slots::<T>(label, 1);
            }
            fn slots<T>(&mut self, label: &str, count: usize) {
                let size = layout::<T>(label);
                let slack = align_of::<T>().checked_sub(1).unwrap();
                let slot = size.checked_add(slack).expect(label);
                self.bytes(label, slot.checked_mul(count).expect(label));
            }
            fn raw<T>(&mut self, label: &str) { self.bytes(label, layout::<T>(label)); }
        }
        fn reference_controller_existing_layout_terms() {
            // PARTIAL target-layout observations, not controller funding certification.
            // Each ledger is ONLY the explicitly labeled worksheet terms below.
            // No sums across domains, ABI/stack claim, backing proof or byte tariff.
            // Existing expected/transfer tests below retain their genuine contracts.
            type C = OwnedBudgetContext;
            let mut takes = Ledger::default();
            // startup-design.md A09: complete selected caller-take value terms.
            takes.slots::<Option<LocalDealerFor<C>>>("A09/dealer-result-receiver", 2);
            takes.slot::<LocalDealerFor<C>>("A09/dealer-local");
            takes.slots::<Option<DealerMaskFor<C, Ring>>>("A09/x-result-receiver", 2);
            takes.slot::<DealerMaskFor<C, Ring>>("A09/x-local");
            takes.slots::<Option<DealerMaskFor<C, Ring>>>("A09/w-result-receiver", 2);
            takes.slot::<DealerMaskFor<C, Ring>>("A09/w-local");
            println!("A09 selected value terms: {}", takes.bytes);

            let mut issuance = Ledger::default();
            // A11 exact existing matrix_with_ticket signature, except fill:
            // its selected controller closure expression does not exist: BLOCKED.
            issuance.slots::<&mut LocalDealerFor<C>>("A11/self-formation-parameter", 2);
            issuance.slots::<[u8; 32]>("A11/operation-formation-parameter", 2);
            issuance.slots::<matrix::Spec>("A11/spec-formation-parameter", 2);
            issuance.slots::<DealerMaskFor<C, Ring>>("A11/x-formation-parameter", 2);
            issuance.slots::<DealerMaskFor<C, Ring>>("A11/w-formation-parameter", 2);
            issuance.slots::<EdgeDescriptor>("A11/output-formation-parameter", 2);
            issuance.slots::<[u8; 32]>("A11/consumer-formation-parameter", 2);
            issuance.slots::<MatrixRange>("A11/range-formation-parameter", 2);
            issuance.slots::<Option<LeaseFor<C>>>("A11/reservation-formation-parameter", 2);
            issuance.slots::<Option<matrix::party::IssuanceTicket>>("A11/ticket-formation-parameter", 2);
            issuance.slot::<MatrixIssuedFor<C>>("A11/local-tuple");
            issuance.slots::<Result<MatrixIssuedFor<C>, Error>>("A11/return-receiver", 2);
            issuance.slot::<DealerMaskFor<C, Ring>>("A11/destructured-mask");
            issuance.slots::<MatrixPartyFor<C>>("A11/destructured-parties", 2);
            // Separate explicit drain boundaries; matched moves are already above.
            issuance.slots::<&mut Option<LeaseFor<C>>>("A11/drain-original-reference", 2);
            issuance.slots::<Option<LeaseFor<C>>>("A11/drain-original-return-receiver", 2);
            issuance.slots::<&mut Option<DealerMaskFor<C, Ring>>>("A11/drain-x-w-references", 4);
            issuance.slots::<Option<DealerMaskFor<C, Ring>>>("A11/drain-x-w-return-receiver", 4);
            println!("A11 selected non-fill terms ONLY: {}", issuance.bytes);

            let mut association = Ledger::default();
            // fixed-host-design.md native counter association: five grouped terms.
            association.slots::<&OwnedBudget>("host-association/budget-reference", 4);
            association.slots::<&std::sync::Arc<Budget>>("host-association/arc-reference", 4);
            association.slots::<&Budget>("host-association/counter-reference", 8);
            association.slots::<*const Budget>("host-association/pointer", 4);
            association.slots::<bool>("host-association/boolean", 3);
            println!("host association selected transits: {}", association.bytes);

            let mut validation = Ledger::default();
            // validation-census.md DimensionsBody: selected non-closure terms ONLY.
            validation.slot::<std::ops::RangeInclusive<u8>>("V/DimensionsBody/bits-range");
            validation.slots::<&std::ops::RangeInclusive<u8>>("V/DimensionsBody/bits-range-reference", 2);
            validation.slots::<&u8>("V/DimensionsBody/bits-reference", 2);
            validation.slots::<std::slice::Iter<'_, usize>>("V/DimensionsBody/iter-return-receiver", 2);
            validation.slots::<&mut std::slice::Iter<'_, usize>>("V/DimensionsBody/any-iterator-reference", 2);
            // L: actual array and iterator layouts, NOT component-size sums.
            validation.slots::<[i128; 2]>("V/L/endpoint-arrays", 3);
            validation.slots::<[i128; 2]>("V/L/into-iter-array-boundaries", 6);
            validation.slots::<std::array::IntoIter<i128, 2>>("V/L/into-iter-return-receiver", 6);
            validation.slots::<std::array::IntoIter<i128, 2>>("V/L/named-loop-iterators", 3);
            validation.slots::<&mut std::array::IntoIter<i128, 2>>("V/L/next-receiver-boundaries", 18);
            validation.slots::<Option<i128>>("V/L/next-return-receiver", 18);
            println!("validation selected range/iterator terms ONLY: {}", validation.bytes);
            // Other validation terms, exact source closures and proposed controller/
            // payer/quote/root records remain unmeasured here; no stand-ins allowed.
            // Vec/Arc header layouts never establish original backing capacities.
        }
        fn sum(values: &[usize]) -> usize {
            values.iter().try_fold(0usize, |n, &v| n.checked_add(v)).unwrap()
        }
        fn node(ledger: &mut Ledger, label: &str) {
            println!("checked node {label}");
            ledger.slot::<usize>("operand/left");
            ledger.slot::<usize>("operand/right");
            ledger.slot::<usize>("node/named-success");
            ledger.slot::<Option<usize>>("node/checked-result");
            ledger.slot::<Error>("error/node-resource");
        }
        fn native_calculation() -> usize {
            // This independent calculation ledger needs no N/Live private fields.
            type Q = (usize, usize, usize, usize);
            type Geometry = pllm_core::streamed_matrix::MatrixGeometry;
            type GeometryError = pllm_core::streamed_matrix::Error;
            let mut geometry = Ledger::default();
            geometry.raw::<matrix::Spec>("helper/geometry-input");
            geometry.raw::<Result<Geometry, GeometryError>>("producer/core-geometry-result");
            geometry.raw::<Result<Geometry, Error>>("producer/mapped-geometry-result");
            layout::<Geometry>("geometry/record");
            layout::<GeometryError>("geometry/error");
            let dimensions_phase = layout::<[usize; 3]>("helper/dimensions-elements")
                .max(geometry.bytes);
            let mut ledger = Ledger::default();
            ledger.raw::<matrix::Spec>("helper/dimensions-input");
            ledger.raw::<Result<Q, Error>>("producer/dimensions-result");
            ledger.bytes("helper/dimensions-phase-maximum", dimensions_phase);
            ledger.slot::<matrix::Spec>("helper/public-resources-input");
            ledger.slot::<matrix::Spec>("helper/context-resources-input");
            ledger.slot::<Result<Q, Error>>("receiver/dimensions_result");
            ledger.slot::<Q>("receiver/dimensions_tuple");
            ledger.slot::<usize>("destructure/a");
            ledger.slot::<usize>("destructure/b");
            ledger.slot::<usize>("destructure/c");
            ledger.slot::<usize>("destructure/d");
            ledger.slot::<usize>("receiver/dn");
            ledger.slot::<Option<usize>>("producer/dn-allowance-read");
            ledger.slot::<Error>("error/dn-read-resource");
            ledger.slot::<Error>("error/dimensions-forward");
            node(&mut ledger, "n00:a+b");
            node(&mut ledger, "n01:n00+c");
            node(&mut ledger, "n02:8*n01");
            node(&mut ledger, "n03:8*c");
            node(&mut ledger, "n04:header+n03");
            node(&mut ledger, "n05:party-owner+frame-owner");
            node(&mut ledger, "n06:n05+output-owner");
            node(&mut ledger, "n07:n06+dn");
            node(&mut ledger, "n08:n02+n03");
            node(&mut ledger, "n09:n08+n04");
            node(&mut ledger, "n10:n09+n07");
            node(&mut ledger, "n11:party-allocators");
            node(&mut ledger, "n12:n10+n11");
            node(&mut ledger, "n13:n12+party-stack");
            node(&mut ledger, "n14:lease+live");
            node(&mut ledger, "n15:n14+temporary-link");
            node(&mut ledger, "n16:n03+n15");
            node(&mut ledger, "n17:n16+dealer-allocator");
            node(&mut ledger, "n18:n17+dealer-stack");
            node(&mut ledger, "n19:both-parties");
            node(&mut ledger, "n20:n19+n18");
            node(&mut ledger, "n21:bits*n01");
            node(&mut ledger, "n22:party-zero-macs");
            node(&mut ledger, "n23:party-one-macs");
            node(&mut ledger, "n24:complete-macs");
            node(&mut ledger, "n25:nonce+n02");
            node(&mut ledger, "n26:both-frames");
            node(&mut ledger, "n27:borrowed-operands");
            node(&mut ledger, "n28:both-party-metadata");
            node(&mut ledger, "n29:n28+n15");
            node(&mut ledger, "n30:all-allocators");
            ledger.slot::<Error>("error/combined-cap");
            ledger.slot::<matrix::party::Resources>("receiver/named-out");
            ledger.slot::<Result<matrix::party::Resources, Error>>("producer/context-resources-return");
            ledger.slot::<Result<matrix::party::Resources, Error>>("receiver/public-forwarding");
            ledger.slot::<Result<matrix::party::Resources, Error>>("producer/public-resources-return");
            ledger.slot::<Result<matrix::party::Resources, Error>>("receiver/public-caller");
            ledger.slot::<[usize; 2]>("producer/party-macs-array");
            ledger.slot::<bool>("operand/cap-comparison");
            ledger.bytes
        }
        fn typed_calculation() -> usize {
            let mut ledger = Ledger { bytes: native_calculation() };
            ledger.slot::<matrix::Spec>("helper/public-matrix-reservation-input");
            ledger.slot::<matrix::Spec>("helper/context-matrix-reservation-input");
            ledger.slot::<matrix::party::Resources>("receiver/native-resources");
            ledger.slot::<usize>("receiver/dt");
            ledger.slot::<Option<usize>>("producer/dt-allowance-read");
            ledger.slot::<Error>("error/dt-read-resource");
            ledger.slot::<Error>("error/native-resources-forward");
            node(&mut ledger, "t00:matrix-party+common-output");
            node(&mut ledger, "t01:t00+dt");
            node(&mut ledger, "t02:both-parties");
            node(&mut ledger, "t03:native+t02");
            node(&mut ledger, "t04:output-mask-payload");
            node(&mut ledger, "t05:t03+t04");
            node(&mut ledger, "t06:t05+mask-allocator");
            node(&mut ledger, "t07:t06+mask-owner");
            ledger.slot::<Result<usize, Error>>("producer/context-reservation-return");
            ledger.slot::<Result<usize, Error>>("receiver/public-forwarding");
            ledger.slot::<Result<usize, Error>>("producer/public-reservation-return");
            ledger.slot::<Result<usize, Error>>("receiver/public-caller");
            ledger.bytes
        }
        fn transfer_slots<C: AllocationContext>() -> usize {
            let mut ledger = Ledger::default();
            ledger.slot::<Result<CommonMaskedFor<C, Ring>, Error>>("producer/finish-closure-return");
            ledger.slot::<Result<CommonMaskedFor<C, Ring>, Error>>("receiver/finish-named-result");
            ledger.slot::<Result<CommonMaskedFor<C, Ring>, Error>>("producer/finish-function-return");
            ledger.slot::<Result<CommonMaskedFor<C, Ring>, Error>>("receiver/finish_opaque");
            ledger.slot::<CommonMaskedFor<C, Ring>>("receiver/common");
            ledger.slot::<CommonBackingFor<C>>("destructure/backing");
            ledger.slot::<matrix::party::MaskedOutputFor<C>>("destructure/matrix-original");
            ledger.slot::<matrix::party::OpaqueNative<C>>("receiver/native-predecessor");
            ledger.slot::<OpaqueTyped<C>>("receiver/named-typed-out");
            ledger.slot::<OpaqueTyped<C>>("producer/returned-typed-out");
            ledger.slot::<Result<OpaqueTyped<C>, Error>>("producer/finish_opaque-return");
            ledger.slot::<Result<OpaqueTyped<C>, Error>>("receiver/finish_opaque-caller");
            ledger.slot::<Binding>("destructure/binding");
            ledger.slot::<u8>("destructure/role");
            ledger.slot::<PhantomData<Ring>>("destructure/kind");
            ledger.slot::<LeaseFor<C>>("destructure/typed-lease");
            ledger.slot::<Error>("error/wrong-variant-context");
            ledger.slot::<&mut MatrixPartyFor<C>>("receiver/self-reference");
            ledger.slot::<&[u8]>("receiver/peer-reference");
            ledger.bytes
        }
        fn spec() -> matrix::Spec {
            matrix::Spec { bits: 50, rows: 1, inner: 1, columns: 1,
                layout: matrix::MatrixLayout::InputOutput }
        }
        fn source() -> SourceIdentity {
            SourceIdentity::new([11; 32], [12; 32], approved_implementation())
        }
        fn edge(id: u8, f: u8, private: bool) -> EdgeDescriptor {
            use pllm_core::edge_contract::{EdgeIdentity, Shape};
            EdgeDescriptor::new(
                EdgeIdentity::new(source(), [1; 32], [id + 64; 32], [id; 32], 0),
                Shape::new(&[1, 1]).unwrap(),
                Numeric::new(50, f, Interpretation::Signed).unwrap(),
                Representation::CommonAdditiveMasked,
                private.then_some(WeightVisibility::Private),
            ).unwrap()
        }
        fn root<C: AllocationContext>(d: &mut LocalDealerFor<C>, id: u8, private: bool, value: i64)
            -> (DealerMaskFor<C, Ring>, [CommonMaskedFor<C, Ring>; 2])
        {
            d.prepare_root(edge(id, 12, private), BoundRange::new(-8192, 8192).unwrap(), [9; 32])
                .unwrap().encode(&[value]).unwrap()
        }
        #[derive(Clone, Copy)]
        struct Expected { party: usize, retained: usize, native_retained: usize, total: usize }
        fn expected<C: AllocationContext>() -> Expected {
            println!("typed context {}", std::any::type_name::<C>());
            layout::<CommonMaskedFor<C, Ring>>("typed/G");
            layout::<CommonBackingFor<C>>("typed/B");
            layout::<MatrixPartyFor<C>>("typed/MatrixParty");
            layout::<OpaqueTyped<C>>("typed/T");
            let dt = transfer_slots::<C>();
            assert_eq!(MatrixPartyFor::<C>::FINISHED_TRANSFER_BYTES, Some(dt));
            assert_eq!(matrix::party::PartyFor::<C>::NATIVE_CALC, Some(native_calculation()));
            assert_eq!(MatrixPartyFor::<C>::TYPED_CALC, Some(typed_calculation()));
            let retained = sum(&[size_of::<CommonMaskedFor<C, Ring>>(), dt]);
            let party = sum(&[size_of::<MatrixPartyFor<C>>(), retained]);
            // Native quote is a composed component, independently validated by
            // current_native_slot_funding. This test cannot inspect N's fields.
            let native = matrix::party::resources_for_context::<C>(spec()).unwrap();
            let frame = sum(&[native.frame_bytes, size_of::<matrix::party::FrameFor<C>>(), 64]);
            let native_retained = native.party_reservation_bytes.checked_sub(frame).unwrap()
                .checked_sub(size_of::<matrix::party::PartyFor<C>>()).unwrap();
            let mut incremental = Ledger::default();
            incremental.bytes("owner/left-typed-party-and-slots", party);
            incremental.bytes("owner/right-typed-party-and-slots", party);
            incremental.raw::<u64>("payload/output-dealer-mask");
            incremental.bytes("allocator/output-mask", 64);
            incremental.raw::<DealerMaskFor<C, Ring>>("owner/output-dealer-mask");
            let total = sum(&[native.reservation_bytes, incremental.bytes]);
            assert_eq!(matrix_reservation_bytes_for::<C>(spec()).unwrap(), total);
            Expected { party, retained, native_retained, total }
        }
        fn transfer<C: AllocationContext>(
            mask: DealerMaskFor<C, Ring>, [mut a, mut b]: [MatrixPartyFor<C>; 2],
            [x0, x1]: [CommonMaskedFor<C, Ring>; 2],
            [w0, w1]: [CommonMaskedFor<C, Ring>; 2], expected: Expected,
        ) -> OpaqueTyped<C> {
            assert_eq!(a.lease.as_ref().unwrap().bytes, expected.party);
            assert_eq!(b.lease.as_ref().unwrap().bytes, expected.party);
            let binding = a.output;
            let role = a.role;
            let counter = a.lease.as_ref().unwrap().budget.budget() as *const Budget;
            let af = a.begin(x0, w0, &mut || false).unwrap();
            let bf = b.begin(x1, w1, &mut || false).unwrap();
            assert_eq!(a.lease.as_ref().unwrap().bytes, expected.party);
            a.lease.as_ref().unwrap().budget.budget().fail_after.store(0, std::sync::atomic::Ordering::SeqCst);
            let used = a.lease.as_ref().unwrap().budget.budget().used_bytes();
            let updates = a.lease.as_ref().unwrap().budget.budget().test_accounting_updates();
            let opaque = a.finish_opaque(bf.as_bytes()).unwrap();
            assert_eq!(opaque.typed_lease.budget.budget().used_bytes(), used);
            assert_eq!(opaque.typed_lease.budget.budget().test_accounting_updates(), updates);
            assert_eq!(opaque.binding, binding);
            assert_eq!(opaque.role, role);
            assert_eq!(opaque.typed_lease.budget.budget() as *const Budget, counter);
            assert_eq!(opaque.typed_lease.bytes, expected.retained);
            assert_eq!(a.lease.as_ref().unwrap().bytes, size_of::<MatrixPartyFor<C>>());
            let ordinary = b.finish(af.as_bytes()).unwrap();
            assert_eq!(ordinary._lease.bytes, expected.retained);
            assert_eq!(b.lease.as_ref().unwrap().bytes, size_of::<MatrixPartyFor<C>>());
            drop((mask, a, b, af, bf, ordinary));
            opaque
        }
        reference_controller_existing_layout_terms();
        layout::<material::MatrixImportPermit<'static>>("borrowed-only/typed-material-permit");
        layout::<material::MatrixMaterialPacket<'static>>("borrowed-only/typed-material-packet");
        layout::<material::MatrixHandoff<'static>>("borrowed-only/typed-material-handoff");
        layout::<material::MatrixMaterialDescriptor>("borrowed-only/typed-material-descriptor");
        layout::<material::MatrixMaterialResources>("borrowed-only/typed-material-resources");
        let borrowed = expected::<&Budget>();
        let owned = expected::<OwnedBudgetContext>();
        let bootstrap = OwnedBudget::resources().unwrap().initial_reserved_bytes;
        let fixed = bootstrap.checked_sub(size_of::<OwnedBudget>()).unwrap();
        for short in [true, false] {
            let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
            let mut dealer = LocalDealer::new(source(), [1; 32], [2; 32], &budget).unwrap();
            let (x, xc) = root(&mut dealer, 3, false, -4096);
            let (w, wc) = root(&mut dealer, 4, true, 4096);
            let original = budget.reserve(borrowed.total.checked_sub(usize::from(short)).unwrap()).unwrap();
            // Fill all remaining capacity, proving exact admission uses only the
            // original slice; this blocker is test-only, never workload authority.
            let blocker = budget.reserve(MAX_ALLOCATION_BYTES.checked_sub(budget.used_bytes()).unwrap()).unwrap();
            if short { budget.fail_after.store(0, std::sync::atomic::Ordering::SeqCst); }
            let mut draws = 0;
            let result = dealer.matrix_reserved_with([9; 32], spec(), x, w, edge(5, 24, false),
                [10; 32], MatrixRange::RequireSignedNoWrap, original,
                &mut |bytes| { draws += 1; bytes.fill(7); Ok(()) });
            drop(blocker);
            if short {
                assert!(matches!(result, Err(Error::Resource)));
                assert_eq!(draws, 0);
                assert_eq!(budget.fail_after.load(std::sync::atomic::Ordering::SeqCst), 0);
                drop((xc, wc, dealer));
                assert_eq!(budget.used_bytes(), 0);
            } else {
                assert_eq!(draws, 5);
                let (mask, parties) = result.unwrap();
                let opaque = transfer(mask, parties, xc, wc, borrowed);
                assert!(std::ptr::eq(opaque.typed_lease.budget, &budget));
                drop(dealer);
                assert_eq!(budget.used_bytes(), sum(&[borrowed.retained, borrowed.native_retained]));
                drop(opaque);
                assert_eq!(budget.used_bytes(), 0);
            }
        }
        for short in [true, false] {
            let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
            let weak = budget.test_counter_weak();
            let counter = budget.budget() as *const Budget;
            let mut dealer = OwnedLocalDealer::new(source(), [1; 32], [2; 32], &budget).unwrap();
            let (x, xc) = root(&mut dealer, 3, false, -4096);
            let (w, wc) = root(&mut dealer, 4, true, 4096);
            let original = budget.reserve_native(owned.total.checked_sub(usize::from(short)).unwrap()).unwrap();
            let blocker = budget.reserve_native(MAX_ALLOCATION_BYTES.checked_sub(budget.budget().used_bytes()).unwrap()).unwrap();
            if short { budget.budget().fail_after.store(0, std::sync::atomic::Ordering::SeqCst); }
            let mut draws = 0;
            let result = dealer.matrix_reserved_with([9; 32], spec(), x, w, edge(5, 24, false),
                [10; 32], MatrixRange::RequireSignedNoWrap, original,
                &mut |bytes| { draws += 1; bytes.fill(7); Ok(()) });
            drop(blocker);
            if short {
                assert!(matches!(result, Err(Error::Resource)));
                assert_eq!(draws, 0);
                assert_eq!(budget.budget().fail_after.load(std::sync::atomic::Ordering::SeqCst), 0);
                drop((xc, wc, dealer));
                assert_eq!(budget.budget().used_bytes(), bootstrap);
            } else {
                assert_eq!(draws, 5);
                let (mask, parties) = result.unwrap();
                let opaque = transfer(mask, parties, xc, wc, owned);
                assert_eq!(opaque.typed_lease.budget.budget() as *const Budget, counter);
                drop((dealer, budget));
                let observer = weak.upgrade().unwrap();
                assert_eq!(observer.used_bytes(), sum(&[fixed, owned.retained, owned.native_retained]));
                drop(opaque);
                assert_eq!(observer.used_bytes(), fixed);
                drop(observer);
                assert!(weak.upgrade().is_none());
            }
        }
        let mut overflow = spec();
        overflow.rows = usize::MAX;
        overflow.inner = 2;
        assert!(matches!(matrix_reservation_bytes_for::<&Budget>(overflow), Err(Error::Resource)));
        assert!(matches!(matrix_reservation_bytes_for::<OwnedBudgetContext>(overflow), Err(Error::Resource)));
        {
            let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
            let mut dealer = LocalDealer::new(source(), [1; 32], [2; 32], &budget).unwrap();
            let (x, xc) = root(&mut dealer, 3, false, -4096);
            let (w, wc) = root(&mut dealer, 4, true, 4096);
            let original = budget.reserve(borrowed.total).unwrap();
            budget.fail_after.store(0, std::sync::atomic::Ordering::SeqCst);
            let mut draws = 0;
            assert!(matches!(dealer.matrix_reserved_with([9; 32], overflow, x, w,
                edge(5, 24, false), [10; 32], MatrixRange::RequireSignedNoWrap, original,
                &mut |bytes| { draws += bytes.len(); bytes.fill(7); Ok(()) }), Err(Error::Resource)));
            assert_eq!(draws, 0);
            assert_eq!(budget.fail_after.load(std::sync::atomic::Ordering::SeqCst), 0);
            drop((dealer, xc, wc));
            assert_eq!(budget.used_bytes(), 0);
        }
        {
            let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
            let mut dealer = OwnedLocalDealer::new(source(), [1; 32], [2; 32], &budget).unwrap();
            let (x, xc) = root(&mut dealer, 3, false, -4096);
            let (w, wc) = root(&mut dealer, 4, true, 4096);
            let original = budget.reserve_native(owned.total).unwrap();
            budget.budget().fail_after.store(0, std::sync::atomic::Ordering::SeqCst);
            let mut draws = 0;
            assert!(matches!(dealer.matrix_reserved_with([9; 32], overflow, x, w,
                edge(5, 24, false), [10; 32], MatrixRange::RequireSignedNoWrap, original,
                &mut |bytes| { draws += bytes.len(); bytes.fill(7); Ok(()) }), Err(Error::Resource)));
            assert_eq!(draws, 0);
            assert_eq!(budget.budget().fail_after.load(std::sync::atomic::Ordering::SeqCst), 0);
            drop((dealer, xc, wc));
            assert_eq!(budget.budget().used_bytes(), bootstrap);
        }
        // Authentic finish cannot produce the wrong backing variant. Its drop
        // ordering remains source-only; no copied branch or fake authority here.
        // Weak disappearance is not zeroization/deallocation/refund-arithmetic proof.
    }

    #[test]
    fn opaque_auto_traits_and_original_birth_quotes() {
        fn send_sync<T: Send + Sync>() {}
        send_sync::<OpaqueTyped<OwnedBudgetContext>>();
        fn check<C: AllocationContext>() {
            let spec = matrix::Spec {
                bits: 50, rows: 1, inner: 1, columns: 1,
                layout: matrix::MatrixLayout::OutputInput,
            };
            let native = matrix::party::resources_for_context::<C>(spec).unwrap();
            let dt = MatrixPartyFor::<C>::FINISHED_TRANSFER_BYTES.unwrap();
            assert_eq!(matrix_reservation_bytes_for::<C>(spec).unwrap(),
                native.reservation_bytes
                + 2 * (size_of::<MatrixPartyFor<C>>() + size_of::<CommonMaskedFor<C, Ring>>() + dt)
                + 8 * native.output_elements + ALLOCATOR + size_of::<DealerMaskFor<C, Ring>>());
            assert!(MatrixPartyFor::<C>::TYPED_CALC.unwrap()
                > matrix::party::PartyFor::<C>::NATIVE_CALC.unwrap());
        }
        check::<&Budget>();
        check::<OwnedBudgetContext>();
    }
}
