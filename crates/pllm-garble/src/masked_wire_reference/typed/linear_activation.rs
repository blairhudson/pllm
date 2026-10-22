//! D22 explicitly LOCAL conditional primitive: complete dot -> GapARS12 -> D20.
//!
//! Public descriptors and operand domains are caller declarations. Source labels
//! are compared for consistency, never authenticated. This API establishes only
//! a conditional integer contract, not compiler-certified native provenance.
//! A source-certified facade must validate sources and own coefficients entirely
//! on the compiler side; calling this primitive does not perform those checks.
//! No LocalAssumption elevation or imported RingOnly capability ingress exists.
//! No decoder, transport, bias, per-term shift, or alternative activation profile.

use super::internal_use::Ledger;
use super::*;
use crate::shared_allocation::ExternalCharge;
use pllm_core::edge_contract::{ConsumerUse, EdgeContractResourceQuote};

const USES: usize = 4;
const EXCHANGES: usize = 11; // matrix, two GapARS rounds, unchanged eight D20 rounds

/// Local declaration of checked geometry and conditional domains, never source
/// authentication, a compiler certificate, or permission to run another owner.
/// Edge order: input, private weights, complete Q24 sum, Q12 rescale, D20 output.
/// Operation order: matrix, rescale, activation, final consumer.
/// Domains are half-open encoded-integer intervals, independent of fixture values.
#[derive(Clone, Copy)]
pub struct LinearActivationSpec {
    matrix: matrix::Spec,
    edges: [EdgeDescriptor; 5],
    operations: [[u8; 32]; 4],
    operands: [Interval; 2],
    sum: Interval,
    rescaled: Interval,
    activation: ActivationSpec,
}

// Actual constructor input owner. Caller-retained copies are not this slot.
struct LinearActivationConstructionInput {
    matrix: matrix::Spec,
    edges: [EdgeDescriptor; 5],
    operations: [[u8; 32]; 4],
    kind: ActivationKind,
    operand_domains: [(i128, i128); 2],
}

struct LinearActivationGeometry {
    input: [usize; 2],
    weight: [usize; 2],
    output: [usize; 2],
}

struct LinearActivationIntervals {
    operands: [Interval; 2],
    sum: Interval,
    rescaled: Interval,
}

/// Preconstruction logical metadata only, available without a declaration,
/// table validation, hashing, allocation, or entropy. Each phase is additional
/// to `input_bytes`; phases are sequential, not quantities to sum together.
/// Results include their payload and error discriminant/padding. The return
/// phase covers the origin declaration and destination Result explicitly.
///
/// Caller-retained descriptors/specs/domain arrays, existing dealer, borrowed
/// data/static tables, scalar evaluation locals, references, ABI-only copies,
/// dependency-private SHA scratch, native stack and RSS are excluded. This is
/// a conservative logical slot reservation, not exact optimized stack usage.
/// Native operation reservations do NOT include this preconstruction quote:
/// an enclosing caller must reserve this overlap before calling `new`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearActivationMetadataResources {
    pub input_bytes: usize,
    pub matrix_resources_phase_bytes: usize,
    pub geometry_phase_bytes: usize,
    pub numeric_validation_phase_bytes: usize,
    pub implementation_validation_phase_bytes: usize,
    pub interval_phase_bytes: usize,
    pub activation_phase_bytes: usize,
    pub return_phase_bytes: usize,
    pub peak_bytes: usize,
}

fn metadata_sum(parts: &[usize]) -> Result<usize, Error> {
    parts.iter().try_fold(0usize, |total, &bytes| {
        total.checked_add(bytes).ok_or(Error::Resource)
    })
}

impl LinearActivationSpec {
    /// Owner-defined quote; independent of geometry, acceptance and activation
    /// kind. Never constructs a spec merely to discover constructor resources.
    pub fn checked_logical_resources() -> Result<LinearActivationMetadataResources, Error> {
        let core = EdgeContractResourceQuote::checked().map_err(|_| Error::Resource)?;
        let activation = ActivationSpec::checked_logical_resources()?;
        let input_bytes = size_of::<LinearActivationConstructionInput>();
        // Only the resource record is owned here, NOT the native payload or
        // stack/allocator allowances described by values inside that record.
        // Matrix resource calculation first validates dimensions/geometry. Its
        // returned record and the later element-cap array are sequential with
        // those helper Results; no native reservation value is charged here.
        let matrix_resources_phase_bytes = metadata_sum(&[
            size_of::<matrix::party::Resources>(), // named cost destination
            matrix_resource_calculation_bytes()?.max(size_of::<[usize; 3]>()), // later element-cap check
        ])?;
        let geometry_phase_bytes = size_of::<LinearActivationGeometry>();
        let numeric_validation_phase_bytes = core
            .numeric_construction_bytes
            .max(size_of::<Result<Numeric, Error>>());
        // approved_implementation -> digest: explicit slice table, SHA state,
        // encoded length and digest result. Source/identity getters have no
        // separately retained owner in this phase.
        let implementation_validation_phase_bytes = metadata_sum(&[
            size_of::<[&[u8]; 3]>(),
            size_of::<Sha256>(),
            size_of::<[u8; 8]>(),
            size_of::<[u8; 32]>(),
        ])?;
        // matrix_interval's two endpoint IntoIters coexist, as do its copied
        // interval arguments and returned Result. Other arithmetic locals are
        // scalars, not owned logical arrays. The original operand state stays.
        // Once traversal ends, the iterator slots cover the helper's named
        // interval result before its Result destination; those phases are serial.
        let interval_scratch = metadata_sum(&[
            size_of::<[Interval; 2]>(),
            size_of::<std::array::IntoIter<i128, 2>>(),
            size_of::<std::array::IntoIter<i128, 2>>(),
            size_of::<Result<Interval, Error>>(),
        ])?;
        let interval_phase_bytes = metadata_sum(&[
            size_of::<LinearActivationIntervals>(),
            // Domain conversion and complete-dot endpoint traversal are serial.
            size_of::<[Interval; 2]>().max(interval_scratch),
        ])?;
        let activation_phase_bytes = metadata_sum(&[
            size_of::<LinearActivationIntervals>(),
            activation.constructor_peak_bytes,
        ])?;
        let return_phase_bytes = metadata_sum(&[
            size_of::<LinearActivationIntervals>(),
            size_of::<ActivationSpec>(), // named activation origin before Self
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
        ])?;
        let peak = [
            matrix_resources_phase_bytes,
            geometry_phase_bytes,
            numeric_validation_phase_bytes,
            implementation_validation_phase_bytes,
            interval_phase_bytes,
            activation_phase_bytes,
            return_phase_bytes,
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        Ok(LinearActivationMetadataResources {
            input_bytes,
            matrix_resources_phase_bytes,
            geometry_phase_bytes,
            numeric_validation_phase_bytes,
            implementation_validation_phase_bytes,
            interval_phase_bytes,
            activation_phase_bytes,
            return_phase_bytes,
            peak_bytes: metadata_sum(&[input_bytes, peak])?,
        })
    }

    /// Check the fixed numerical contract and descriptor consistency only.
    /// Source identities are unauthenticated declarations. No bias is supported.
    pub fn new(
        matrix: matrix::Spec,
        edges: [EdgeDescriptor; 5],
        operations: [[u8; 32]; 4],
        kind: ActivationKind,
        operand_domains: [(i128, i128); 2],
    ) -> Result<Self, Error> {
        Self::from_construction_input(LinearActivationConstructionInput {
            matrix,
            edges,
            operations,
            kind,
            operand_domains,
        })
    }

    fn from_construction_input(input: LinearActivationConstructionInput) -> Result<Self, Error> {
        let matrix = &input.matrix;
        let edges = &input.edges;
        let operations = &input.operations;
        {
            let cost = matrix::party::resources(*matrix)?;
            if matrix.bits != 50
                || cost.complete_macs > 1 << 20
                || [
                    cost.input_elements,
                    cost.weight_elements,
                    cost.output_elements,
                ]
                .iter()
                .any(|&n| !(1..=256).contains(&n))
            {
                return Err(Error::Resource);
            }
        }
        // Borrow rather than retain five extra copied descriptors.
        let [x, w, s, z, out] = edges;
        {
            let geometry = LinearActivationGeometry {
                input: [matrix.rows, matrix.inner],
                weight: match matrix.layout {
                    matrix::MatrixLayout::InputOutput => [matrix.inner, matrix.columns],
                    matrix::MatrixLayout::OutputInput => [matrix.columns, matrix.inner],
                },
                output: [matrix.rows, matrix.columns],
            };
            if x.shape().dimensions() != geometry.input
                || w.shape().dimensions() != geometry.weight
                || s.shape().dimensions() != geometry.output
                || s.shape() != z.shape()
                || s.shape() != out.shape()
            {
                return Err(Error::Domain);
            }
        }
        for (i, edge) in edges.iter().enumerate() {
            if edge.numeric()
                != Numeric::new(50, if i == 2 { 24 } else { 12 }, Interpretation::Signed)
                    .map_err(|_| Error::Domain)?
                || edge.representation() != Representation::CommonAdditiveMasked
                || edge.weight_visibility()
                    != if i == 1 {
                        Some(WeightVisibility::Private)
                    } else {
                        None
                    }
            {
                return Err(Error::Unsupported);
            }
            if edge.identity().source() != x.identity().source()
                || edge.identity().plan() != x.identity().plan()
                || edges[..i]
                    .iter()
                    .any(|old| old.identity().wire() == edge.identity().wire())
            {
                return Err(Error::Context);
            }
        }
        if x.identity().source().implementation() != approved_implementation()
            || s.identity().first_tensor_index() != z.identity().first_tensor_index()
            || z.identity().first_tensor_index() != out.identity().first_tensor_index()
            || operations
                .iter()
                .enumerate()
                .any(|(i, op)| operations[..i].contains(op))
        {
            return Err(Error::Context);
        }
        let mut intervals = LinearActivationIntervals {
            // Index the retained input; no array-map adapter owns another
            // domain array while constructing the interval backing.
            operands: [
                Interval {
                    lower: input.operand_domains[0].0,
                    upper: input.operand_domains[0].1,
                },
                Interval {
                    lower: input.operand_domains[1].0,
                    upper: input.operand_domains[1].1,
                },
            ],
            sum: Interval { lower: 0, upper: 0 },
            rescaled: Interval { lower: 0, upper: 0 },
        };
        if intervals.operands.iter().any(|range| !range.fits(50)) {
            return Err(Error::Domain);
        }
        // Endpoint products times COMPLETE inner dimension. Never truncate terms.
        intervals.sum = matrix_interval(
            intervals.operands[0],
            intervals.operands[1],
            matrix.inner,
            50,
        )?;
        let quarter = 1i128 << 48;
        if intervals.sum.lower < -quarter || intervals.sum.upper > quarter {
            return Err(Error::Domain);
        }
        intervals.rescaled = intervals.sum.shift(12);
        if intervals.rescaled.lower < -(1i128 << 36) || intervals.rescaled.upper > 1i128 << 36 {
            return Err(Error::Domain);
        }
        let activation = ActivationSpec::new(input.kind, operations[2], *z, *out, operations[3])?;
        Ok(Self {
            matrix: *matrix,
            edges: *edges,
            operations: *operations,
            operands: intervals.operands,
            sum: intervals.sum,
            rescaled: intervals.rescaled,
            activation,
        })
    }

    fn arithmetic(&self) -> arithmetic::Spec {
        arithmetic::Spec {
            bits: 50,
            lanes: self.edges[2].shape().elements(),
            gate: arithmetic::Gate::GapArs { shift: 12 },
        }
    }

    fn uses(self) -> [ConsumerUse; USES] {
        [
            ConsumerUse::new(self.edges[0], self.operations[0], 0, 0),
            ConsumerUse::new(self.edges[1], self.operations[0], 1, 1),
            ConsumerUse::new(self.edges[2], self.operations[1], 0, 2),
            ConsumerUse::new(self.edges[3], self.operations[2], 0, 3),
        ]
    }
}

/// Immutable local logical-capacity estimate, not RSS or source authority.
/// Excludes existing LocalDealer and borrowed caller clear slices. Includes
/// native frames/output/transition overlap; does not reserve or mint capacity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearActivationResources {
    reservation_bytes: usize,
    root_bytes: [usize; 2],
    matrix_bytes: usize,
    rescale_bytes: usize,
    activation_bytes: usize,
    wrapper_bytes: usize,
    run_metadata_bytes: usize,
    party_metadata_bytes: usize,
    frame_wrapper_bytes: usize,
    step_wrapper_bytes: usize,
    transition_workspace_bytes: usize,
}

impl LinearActivationResources {
    /// Entire additional native operation reservation on the dealer's Budget.
    pub fn reservation_bytes(self) -> usize {
        self.reservation_bytes
    }

    /// Root x, root w, typed matrix, typed rescale, unchanged D20, new wrappers.
    /// Components sum to `reservation_bytes`; none is a reservation capability.
    pub fn component_bytes(self) -> [usize; 6] {
        [
            self.root_bytes[0],
            self.root_bytes[1],
            self.matrix_bytes,
            self.rescale_bytes,
            self.activation_bytes,
            self.wrapper_bytes,
        ]
    }

    /// New run metadata, party metadata, frame wrappers, step wrappers, bounded
    /// transition workspace. Existing embedded owners are not charged twice.
    pub fn wrapper_component_bytes(self) -> [usize; 5] {
        [
            self.run_metadata_bytes,
            self.party_metadata_bytes,
            self.frame_wrapper_bytes,
            self.step_wrapper_bytes,
            self.transition_workspace_bytes,
        ]
    }
}

/// PRECALCULATION metadata for `local_linear_activation_resources` (the local
/// public alias of `linear_activation_resources`), NOT `new` or issuance/run.
/// All phases are additional to input plus the retained two-root size array.
/// Native quote *records*, their calculation scratch, revalidation and return
/// origins/destinations are included; capacities described inside records are
/// not metadata. No allocation, hashing, entropy or declaration is needed here.
///
/// Same exclusions as `ActivationResourceCalculationResources`: caller originals
/// and retained quote records must be added separately; borrowed data, reference
/// and scalar/control locals, ABI-only moves, dependency-private SHA scratch,
/// machine stack/RSS and existing execution/dealer owners are excluded. This
/// does not establish source provenance or open the certified facade.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearActivationResourceCalculationResources {
    pub input_bytes: usize,
    pub root_array_bytes: usize,
    pub matrix_phase_bytes: usize,
    pub rescale_phase_bytes: usize,
    pub activation_phase_bytes: usize,
    pub sum_phase_bytes: usize,
    pub return_phase_bytes: usize,
    pub peak_bytes: usize,
}

// Calculation-only inventory, including core dimensions -> geometry. These
// are explicit public record/array/Result slots, not native stack allowances.
fn matrix_resource_calculation_bytes() -> Result<usize, Error> {
    let geometry = metadata_sum(&[
        size_of::<matrix::Spec>(), // geometry's Copy input
        size_of::<
            Result<pllm_core::streamed_matrix::MatrixGeometry, pllm_core::streamed_matrix::Error>,
        >(),
        size_of::<Result<pllm_core::streamed_matrix::MatrixGeometry, Error>>(),
    ])?;
    let dimensions = metadata_sum(&[
        size_of::<matrix::Spec>(), // dimensions' Copy input
        size_of::<Result<(usize, usize, usize, usize), Error>>(),
        size_of::<[usize; 3]>().max(geometry),
    ])?;
    metadata_sum(&[
        size_of::<matrix::Spec>(), // party::resources input
        size_of::<Result<matrix::party::Resources, Error>>(),
        dimensions,
    ])
}

impl LinearActivationSpec {
    /// Fixed PRECALCULATION quote, separate from `checked_logical_resources`.
    /// A caller retaining its original spec adds that owner to `peak_bytes`.
    pub fn checked_resource_calculation_resources(
    ) -> Result<LinearActivationResourceCalculationResources, Error> {
        let activation = ActivationSpec::checked_resource_calculation_resources()?;
        let input_bytes = size_of::<Self>();
        let root_array_bytes = size_of::<[usize; 2]>();
        let matrix_phase_bytes = metadata_sum(&[
            size_of::<matrix::Spec>(), // matrix_reservation_bytes Copy input
            size_of::<matrix::party::Resources>(), // named native destination
            size_of::<Result<usize, Error>>(),
            matrix_resource_calculation_bytes()?,
        ])?;
        let rescale_phase_bytes = metadata_sum(&[
            size_of::<arithmetic::Spec>(), // arithmetic_reservation_bytes input
            size_of::<Result<usize, Error>>(),
            activation::arithmetic_resource_calculation_bytes()?,
        ])?;
        let activation_phase_bytes = metadata_sum(&[
            size_of::<ActivationResources>(), // named activation destination
            activation.peak_bytes,
        ])?;
        // This iterator OWNS its six-element array; unlike borrowed native-cost
        // traversal, its IntoIter slot must be charged through try_fold.
        let sum_phase_bytes = size_of::<std::array::IntoIter<usize, 6>>();
        let return_phase_bytes = metadata_sum(&[
            size_of::<LinearActivationResources>(), // named resources origin
            size_of::<Result<LinearActivationResources, Error>>(),
        ])?;
        Ok(LinearActivationResourceCalculationResources {
            input_bytes,
            root_array_bytes,
            matrix_phase_bytes,
            rescale_phase_bytes,
            activation_phase_bytes,
            sum_phase_bytes,
            return_phase_bytes,
            peak_bytes: metadata_sum(&[
                input_bytes,
                root_array_bytes,
                matrix_phase_bytes
                    .max(rescale_phase_bytes)
                    .max(activation_phase_bytes)
                    .max(sum_phase_bytes)
                    .max(return_phase_bytes),
            ])?,
        })
    }
}

impl LinearActivationResourceCalculationResources {
    /// Add actual retained owners; exact-fit/overflow arithmetic only, no lease.
    pub fn peak_with_retained_bytes(self, retained: usize) -> Result<usize, Error> {
        metadata_sum(&[retained, self.peak_bytes])
    }
}

fn root_bytes(lanes: usize) -> usize {
    2 * (8 * lanes + ALLOCATOR + common_charge())
        + 8 * lanes
        + ALLOCATOR
        + mask_charge()
        + size_of::<RootInput<'static>>()
}
fn frame_charge(body: &FrameBody<'_>) -> usize {
    // Inner frame metadata already belongs to its original native quote.
    let inner = match body {
        FrameBody::Matrix(_) => size_of::<matrix::party::Frame<'static>>(),
        FrameBody::Arithmetic(_) => size_of::<arithmetic::Frame<'static>>(),
        FrameBody::Activation(_) => size_of::<ActivationFrame<'static>>(),
    };
    size_of::<LinearActivationFrame<'static>>() - inner
}
fn party_metadata_bytes() -> usize {
    // Existing quotes own inline matrix/rescale/D20 wrappers, two root common
    // wrappers and the retained dot wrapper. Count only new container/ledger
    // metadata, including Option tags/padding and this party's lease.
    size_of::<LinearParty<'static>>()
        - size_of::<MatrixParty<'static>>()
        - size_of::<ArithmeticParty<'static, Ring>>()
        - size_of::<ActivationParty<'static>>()
        - 3 * common_charge()
}
fn party_frame_bytes() -> usize {
    let outer = size_of::<LinearActivationFrame<'static>>();
    (outer - size_of::<matrix::party::Frame<'static>>())
        + 2 * (outer - size_of::<arithmetic::Frame<'static>>())
        + 8 * (outer - size_of::<ActivationFrame<'static>>())
}
fn step_charge(output: bool) -> usize {
    // Exactly one pair container per returned step; inner owners are already
    // charged, not two complete Step enums charged again on each frame.
    size_of::<LinearActivationStep<'static>>()
        - if output {
            2 * common_charge()
        } else {
            2 * size_of::<LinearActivationFrame<'static>>()
        }
}
fn party_charge() -> usize {
    party_metadata_bytes() + party_frame_bytes()
}
fn transition_workspace_bytes() -> usize {
    // Explicit bounded issuance/transition scratch: declarations, bindings,
    // ticket and at most twelve simultaneously live split-lease temporaries.
    // No copied payloads, extra tables, output backing or heap allocations.
    size_of::<LinearActivationSpec>()
        + size_of::<LinearActivationResources>()
        + 4 * size_of::<Binding>()
        + size_of::<matrix::party::IssuanceTicket>()
        + 12 * size_of::<Lease<'static>>()
        + 2 * (size_of::<PartyStep<'static>>()
            - size_of::<LinearActivationFrame<'static>>().min(common_charge()))
}

/// Quote the local conditional primitive. No allocation, entropy, source
/// authentication or issuance authority. Existing dealer/caller owners excluded.
pub fn linear_activation_resources(
    spec: LinearActivationSpec,
) -> Result<LinearActivationResources, Error> {
    let root_bytes = [
        root_bytes(spec.edges[0].shape().elements()),
        root_bytes(spec.edges[1].shape().elements()),
    ];
    let matrix_bytes = matrix_reservation_bytes(spec.matrix)?;
    let rescale_bytes = arithmetic_reservation_bytes::<Ring>(spec.arithmetic())?;
    let activation_bytes = {
        let activation = activation_resources(spec.activation)?;
        activation.combined_reservation_bytes
    };
    let run_metadata_bytes =
        size_of::<LinearActivationRun<'static>>() - 2 * size_of::<LinearParty<'static>>();
    let party_metadata_bytes = 2 * party_metadata_bytes();
    let frame_wrapper_bytes = 2 * party_frame_bytes();
    let step_wrapper_bytes = (EXCHANGES - 1) * step_charge(false) + step_charge(true);
    let transition_workspace_bytes = transition_workspace_bytes();
    let wrapper_bytes = run_metadata_bytes
        + party_metadata_bytes
        + frame_wrapper_bytes
        + step_wrapper_bytes
        + transition_workspace_bytes;
    let reservation_bytes = [
        root_bytes[0],
        root_bytes[1],
        matrix_bytes,
        rescale_bytes,
        activation_bytes,
        wrapper_bytes,
    ]
    .into_iter()
    .try_fold(0usize, |total, n| total.checked_add(n))
    .ok_or(Error::Resource)?;
    if reservation_bytes > crate::shared_allocation::MAX_ALLOCATION_BYTES {
        return Err(Error::Resource);
    }
    let resources = LinearActivationResources {
        reservation_bytes,
        root_bytes,
        matrix_bytes,
        rescale_bytes,
        activation_bytes,
        wrapper_bytes,
        run_metadata_bytes,
        party_metadata_bytes,
        frame_wrapper_bytes,
        step_wrapper_bytes,
        transition_workspace_bytes,
    };
    Ok(resources)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Fresh,
    Matrix,
    Rescale,
    Activation,
    Dead,
}

/// Separate party-local owners; only orchestration owns both and handles faults.
struct LinearParty<'b> {
    matrix: MatrixParty<'b>,
    rescale: ArithmeticParty<'b, Ring>,
    activation: ActivationParty<'b>,
    roots: Option<[CommonMasked<'b, Ring>; 2]>,
    // Actual original matrix output, including its native singleton guard.
    dot: Option<CommonMasked<'b, Ring>>,
    ledger: Ledger<USES>,
    phase: Phase,
    // Original root/dot inline metadata charges move here at issuance. Their
    // payload leases/backing stay intact; vacated Option storage remains charged
    // until this containing struct drops, even after begin/cancel/completion.
    _slot_metadata: [Lease<'b>; 3],
    lease: Option<Lease<'b>>,
}

enum FrameBody<'b> {
    Matrix(matrix::party::Frame<'b>),
    Arithmetic(arithmetic::Frame<'b>),
    Activation(ActivationFrame<'b>),
}
/// Opaque unauthenticated local peer frame. May outlive the run; original native
/// backing/metadata charges and any actual matrix singleton guard remain owned.
pub struct LinearActivationFrame<'b> {
    body: FrameBody<'b>,
    _lease: Lease<'b>,
}
impl LinearActivationFrame<'_> {
    pub fn as_bytes(&self) -> &[u8] {
        match &self.body {
            FrameBody::Matrix(f) => f.as_bytes(),
            FrameBody::Arithmetic(f) => f.as_bytes(),
            FrameBody::Activation(f) => f.as_bytes(),
        }
    }
}
#[allow(clippy::large_enum_variant)]
enum StepBody<'b> {
    Frames([LinearActivationFrame<'b>; 2], StepMetadata<'b>),
    Output([CommonMasked<'b, Ring>; 2], StepMetadata<'b>),
}
/// Opaque local transition result, either a pair of peer frames or final typed
/// masked outputs. No intermediate values, masks, reservation or proof escape.
pub struct LinearActivationStep<'b> {
    body: StepBody<'b>,
}
impl<'b> LinearActivationStep<'b> {
    /// Borrow outgoing frames ordered [party0, party1], without copying payloads.
    pub fn frames(&self) -> Option<&[LinearActivationFrame<'b>; 2]> {
        match &self.body {
            StepBody::Frames(frames, _) => Some(frames),
            StepBody::Output(_, _) => None,
        }
    }

    /// Consume a frame step. Returns the original opaque step if output-ready.
    // Preserve the admitted inline owner; boxing would add a fallible allocation.
    #[allow(clippy::result_large_err)]
    pub fn into_frames(self) -> Result<[LinearActivationFrame<'b>; 2], Self> {
        match self.body {
            StepBody::Frames(frames, _metadata) => Ok(frames),
            body @ StepBody::Output(_, _) => Err(Self { body }),
        }
    }

    /// Consume final outputs ordered [party0, party1]. These remain opaque typed
    /// capabilities; there is no decoding or source-certified provenance claim.
    /// Returns the original step if more peer exchanges are required.
    // Preserve the admitted inline owner; boxing would add a fallible allocation.
    #[allow(clippy::result_large_err)]
    pub fn into_output(self) -> Result<[CommonMasked<'b, Ring>; 2], Self> {
        match self.body {
            StepBody::Output(output, _metadata) => Ok(output),
            body @ StepBody::Frames(_, _) => Err(Self { body }),
        }
    }
}
/// Opaque charge for the pair container, not a reservation minting interface.
struct StepMetadata<'b> {
    _lease: Lease<'b>,
}

/// Local conditional consuming run, never source authentication, a compiler
/// certificate, decoder or transport. No party escape hatch: every returned
/// error/cancellation burns both owned peers. Final outputs remain masked.
pub struct LinearActivationRun<'b> {
    parties: [LinearParty<'b>; 2],
    lease: Option<Lease<'b>>,
}

impl<'b> LocalDealer<'b> {
    /// Execute the explicitly LOCAL conditional D22 primitive. Source labels and
    /// intervals are declarations, not checkpoint authentication or compiler-
    /// certified native provenance. A compiler facade must perform its own
    /// source checks and supply its own certificate-owned coefficients.
    ///
    /// Checks all clear integers against declared domains and admits the entire
    /// native inventory on this dealer's original Budget before any root draws.
    /// Both x and w are borrowed transiently; no clear copies are retained. No
    /// bias, decoding, transport, arbitrary imports or capability elevation.
    pub fn local_linear_activation_from_values(
        &mut self,
        spec: LinearActivationSpec,
        x: &[i64],
        w: &[i64],
    ) -> Result<(DealerMask<'b, Ring>, LinearActivationRun<'b>), Error> {
        self.linear_activation_from_values_with(spec, x, w, &mut |bytes| {
            getrandom::fill(bytes).map_err(|_| Error::Randomness)
        })
    }

    /// Execute the same LOCAL ASSUMPTION primitive using already-charged native
    /// capacity. This grants no compiler/source authority. `charge` must belong
    /// to this exact Budget and contain the full native quote as payload, beyond
    /// any retained caller storage. Validation and singleton admission precede
    /// transfer and entropy. Transfer never releases/reacquires capacity.
    ///
    /// Success or any post-transfer error consumes exactly the native quote from
    /// `charge`; native owners release their slices on drop. The source guard,
    /// its inline metadata and remaining payload stay alive and unchanged.
    pub fn local_linear_activation_from_reserved_values(
        &mut self,
        spec: LinearActivationSpec,
        x: &[i64],
        w: &[i64],
        charge: &mut ExternalCharge<'b>,
    ) -> Result<(DealerMask<'b, Ring>, LinearActivationRun<'b>), Error> {
        self.linear_activation_from_reserved_values_with(spec, x, w, charge, &mut |bytes| {
            getrandom::fill(bytes).map_err(|_| Error::Randomness)
        })
    }

    /// Test only whether this dealer uses that exact Budget object. This is not
    /// an admission, reservation, source check or issuance capability.
    pub fn is_budget(&self, budget: &Budget) -> bool {
        std::ptr::eq(self.budget, budget)
    }

    #[cfg(test)]
    pub(crate) fn linear_activation_from_values(
        &mut self,
        spec: LinearActivationSpec,
        x: &[i64],
        w: &[i64],
    ) -> Result<(DealerMask<'b, Ring>, LinearActivationRun<'b>), Error> {
        self.local_linear_activation_from_values(spec, x, w)
    }

    fn linear_activation_from_values_with(
        &mut self,
        spec: LinearActivationSpec,
        x: &[i64],
        w: &[i64],
        fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<(DealerMask<'b, Ring>, LinearActivationRun<'b>), Error> {
        let cost = self.linear_activation_preflight(spec, x, w)?;
        let aggregate = self.budget.reserve(cost.reservation_bytes)?;
        let ticket = matrix::party::IssuanceTicket::acquire()?;
        self.linear_activation_reserved_with(spec, x, w, cost, aggregate, ticket, fill)
    }

    fn linear_activation_from_reserved_values_with(
        &mut self,
        spec: LinearActivationSpec,
        x: &[i64],
        w: &[i64],
        charge: &mut ExternalCharge<'b>,
        fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<(DealerMask<'b, Ring>, LinearActivationRun<'b>), Error> {
        let cost = self.linear_activation_preflight(spec, x, w)?;
        if !charge.is_for(self.budget) || charge.payload_bytes() < cost.reservation_bytes {
            return Err(Error::Resource);
        }
        let ticket = matrix::party::IssuanceTicket::acquire()?;
        let aggregate = charge.split_payload(self.budget, cost.reservation_bytes)?;
        self.linear_activation_reserved_with(spec, x, w, cost, aggregate, ticket, fill)
    }

    fn linear_activation_preflight(
        &self,
        spec: LinearActivationSpec,
        x: &[i64],
        w: &[i64],
    ) -> Result<LinearActivationResources, Error> {
        let cost = linear_activation_resources(spec)?;
        for edge in spec.edges {
            self.validate(edge, false)?;
        }
        for (i, values) in [x, w].into_iter().enumerate() {
            let interval = spec.operands[i];
            if values.len() != spec.edges[i].shape().elements()
                || values
                    .iter()
                    .any(|&v| i128::from(v) < interval.lower || i128::from(v) >= interval.upper)
            {
                return Err(Error::Domain);
            }
        }
        // Two roots, matrix, rescale, seventeen D20 internal wires and output.
        self.counter.checked_add(22).ok_or(Error::Resource)?;
        Ok(cost)
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_activation_reserved_with(
        &mut self,
        spec: LinearActivationSpec,
        x: &[i64],
        w: &[i64],
        cost: LinearActivationResources,
        mut aggregate: Lease<'b>,
        ticket: matrix::party::IssuanceTicket,
        fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<(DealerMask<'b, Ring>, LinearActivationRun<'b>), Error> {
        // Own real singleton plus issuance number before ANY root/output mask
        // entropy. Error Drop releases this same guard; native issuance consumes
        // it continuously instead of checking metadata then racing to reacquire.
        let mut wrapper = aggregate.split(cost.wrapper_bytes)?;
        let left_lease = wrapper.split(party_charge())?;
        let right_lease = wrapper.split(party_charge())?;
        let mut roots = |index: usize, values: &[i64]| {
            let interval = spec.operands[index];
            self.prepare_root_reserved_with(
                spec.edges[index],
                BoundRange::new(interval.lower, interval.upper)?,
                spec.operations[0],
                aggregate.split(cost.root_bytes[index])?,
                &mut *fill,
            )?
            .encode(values)
        };
        let (xm, [x0, x1]) = roots(0, x)?;
        let (wm, [w0, w1]) = roots(1, w)?;
        let (sum_mask, [m0, m1]) = self.matrix_ticket_reserved_with(
            spec.operations[0],
            spec.matrix,
            xm,
            wm,
            spec.edges[2],
            spec.operations[1],
            MatrixRange::RequireSignedNoWrap,
            aggregate.split(cost.matrix_bytes)?,
            ticket,
            fill,
        )?;
        if sum_mask.binding.interval != Some(spec.sum) {
            return Err(Error::Context);
        }
        let (z_mask, [r0, r1]) = self.gap_ars_reserved_with(
            spec.operations[1],
            sum_mask,
            12,
            spec.edges[3],
            spec.operations[2],
            aggregate.split(cost.rescale_bytes)?,
            fill,
        )?;
        if z_mask.binding.interval != Some(spec.rescaled) {
            return Err(Error::Context);
        }
        let bindings = [m0.inputs[0], m0.inputs[1], m0.output, r0.output];
        // Requires only pub(super) visibility on the existing reserved D20 helper.
        // Do not replace this with activation(), which would reacquire capacity.
        let (output, [a0, a1]) = self.activation_reserved_with(
            spec.activation,
            z_mask,
            aggregate.split(cost.activation_bytes)?,
            fill,
        )?;
        if aggregate.bytes != 0 {
            return Err(Error::Resource);
        }
        let make = |mut matrix: MatrixParty<'b>,
                    rescale,
                    activation,
                    mut roots: [CommonMasked<'b, Ring>; 2],
                    role,
                    lease|
         -> Result<LinearParty<'b>, Error> {
            let slot_metadata = [
                roots[0]._lease.split(common_charge())?,
                roots[1]._lease.split(common_charge())?,
                matrix
                    .lease
                    .as_mut()
                    .ok_or(Error::Consumed)?
                    .split(common_charge())?,
            ];
            Ok(LinearParty {
                matrix,
                rescale,
                activation,
                roots: Some(roots),
                dot: None,
                ledger: Ledger {
                    uses: spec.uses(),
                    bindings,
                    role,
                    next: 0,
                },
                phase: Phase::Fresh,
                _slot_metadata: slot_metadata,
                lease: Some(lease),
            })
        };
        Ok((
            output,
            LinearActivationRun {
                parties: [
                    make(m0, r0, a0, [x0, w0], 0, left_lease)?,
                    make(m1, r1, a1, [x1, w1], 1, right_lease)?,
                ],
                lease: Some(wrapper),
            },
        ))
    }
}

impl<'b> LinearParty<'b> {
    fn cancel(&mut self) {
        self.phase = Phase::Dead;
        self.ledger.next = USES as u8;
        self.roots = None;
        self.dot = None;
        // Keep cancelled sub-wrapper metadata charged until this struct drops.
        self.matrix.cancel();
        self.rescale.cancel();
        self.activation.cancel();
        retain_wrapper_metadata(&mut self.lease, party_metadata_bytes());
    }
    fn frame(&mut self, body: FrameBody<'b>) -> Result<LinearActivationFrame<'b>, Error> {
        let bytes = frame_charge(&body);
        Ok(LinearActivationFrame {
            body,
            _lease: self.lease.as_mut().ok_or(Error::Consumed)?.split(bytes)?,
        })
    }
    fn begin(
        &mut self,
        cancelled: &mut impl FnMut() -> bool,
    ) -> Result<LinearActivationFrame<'b>, Error> {
        if self.phase != Phase::Fresh {
            return Err(Error::Consumed);
        }
        let [x, w] = self.roots.take().ok_or(Error::Consumed)?;
        self.ledger.burn(0, x.binding, x.role)?;
        self.ledger.burn(1, w.binding, w.role)?;
        let frame = self.matrix.begin(x, w, cancelled)?;
        self.phase = Phase::Matrix;
        self.frame(FrameBody::Matrix(frame))
    }
    fn advance(&mut self, peer: &[u8]) -> Result<PartyStep<'b>, Error> {
        let body = match self.phase {
            Phase::Matrix => {
                let dot = self.matrix.finish(peer)?;
                self.ledger.burn(2, dot.binding, dot.role)?;
                if dot.binding != self.rescale.input || dot.role != self.rescale.role {
                    return Err(Error::Context);
                }
                self.dot = Some(dot);
                // Borrow from actual guarded backing, not a reconstructed copy.
                // Retain it through completion of its consuming rescale operation.
                let frame = self
                    .rescale
                    .native
                    .as_mut()
                    .ok_or(Error::Consumed)?
                    .begin(self.dot.as_ref().ok_or(Error::Consumed)?.backing.values())?;
                self.phase = Phase::Rescale;
                FrameBody::Arithmetic(frame)
            }
            Phase::Rescale => match self.rescale.advance(peer)? {
                Step::Frame(frame) => FrameBody::Arithmetic(frame),
                Step::Output(z) => {
                    self.ledger.burn(3, z.binding, z.role)?;
                    let frame = self.activation.begin(z)?;
                    self.dot = None; // Last matrix-backed use has completed.
                    self.matrix.cancel();
                    self.rescale.cancel();
                    self.phase = Phase::Activation;
                    FrameBody::Activation(frame)
                }
            },
            Phase::Activation => match self.activation.advance(peer)? {
                ActivationStep::Frame(frame) => FrameBody::Activation(frame),
                ActivationStep::Output(output) => {
                    self.cancel();
                    return Ok(PartyStep::Output(output));
                }
            },
            Phase::Fresh | Phase::Dead => return Err(Error::Consumed),
        };
        Ok(PartyStep::Frame(self.frame(body)?))
    }
}
#[allow(clippy::large_enum_variant)]
enum PartyStep<'b> {
    Frame(LinearActivationFrame<'b>),
    Output(CommonMasked<'b, Ring>),
}

impl<'b> LinearActivationRun<'b> {
    /// Burn both owned peers. Retained external frames keep their own charges
    /// and native guards; cancelled wrapper metadata remains charged until drop.
    pub fn cancel(&mut self) {
        for party in &mut self.parties {
            party.cancel();
        }
        retain_wrapper_metadata(
            &mut self.lease,
            size_of::<Self>() - 2 * size_of::<LinearParty<'static>>(),
        );
    }
    /// Begin both party-local evaluators. Returned frames are [party0, party1].
    /// Any failure or cancellation consumes both peers; no retry is available.
    pub fn begin(
        &mut self,
        cancelled: &mut impl FnMut() -> bool,
    ) -> Result<[LinearActivationFrame<'b>; 2], Error> {
        let result = (|| {
            if cancelled() {
                return Err(Error::Consumed);
            }
            let left = self.parties[0].begin(cancelled)?;
            let right = self.parties[1].begin(cancelled)?;
            Ok([left, right])
        })();
        if result.is_err() {
            self.cancel();
        }
        result
    }
    /// Peer order is [frame from party1, frame from party0]. Frames remain caller
    /// owned; retained matrix frames independently retain their ORIGINAL guard.
    pub fn advance(
        &mut self,
        peers: [&[u8]; 2],
        cancelled: &mut impl FnMut() -> bool,
    ) -> Result<LinearActivationStep<'b>, Error> {
        let result = (|| {
            if cancelled() {
                return Err(Error::Consumed);
            }
            let left = self.parties[0].advance(peers[0])?;
            if cancelled() {
                return Err(Error::Consumed);
            }
            let right = self.parties[1].advance(peers[1])?;
            match (left, right) {
                (PartyStep::Frame(a), PartyStep::Frame(b)) => {
                    let metadata = StepMetadata {
                        _lease: self
                            .lease
                            .as_mut()
                            .ok_or(Error::Consumed)?
                            .split(step_charge(false))?,
                    };
                    Ok(LinearActivationStep {
                        body: StepBody::Frames([a, b], metadata),
                    })
                }
                (PartyStep::Output(a), PartyStep::Output(b)) => {
                    let metadata = StepMetadata {
                        _lease: self
                            .lease
                            .as_mut()
                            .ok_or(Error::Consumed)?
                            .split(step_charge(true))?,
                    };
                    self.cancel();
                    Ok(LinearActivationStep {
                        body: StepBody::Output([a, b], metadata),
                    })
                }
                _ => Err(Error::Context),
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
