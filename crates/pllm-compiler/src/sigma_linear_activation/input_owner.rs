//! Compiler-only typed input ownership for the existing D22/D27 facade.
//! Typed owners do not retroactively admit incoming creation. Charged Rust bytes
//! use the shared admitted parser below; no Python buffer is acquired here.
//! Publisher trust is the existing explicit trust assumption;
//! supplied source bytes still require the ordinary checked import.

use super::*;
use crate::decoder_matrix_inventory::{
    admit_inventory_for_original_owner, owned_inventory_bootstrap_bytes,
};
use pllm_garble::masked_wire_reference::typed;
use pllm_garble::shared_allocation::{
    Error as NativeError, OwnedBudget, OwnedCharged, OwnedExternalCharge,
};
use pllm_types::{Digest, NamedDigest};

/// Incoming, already charged publisher declaration. The upstream typed builder
/// must reserve BEFORE constructing its strings/vectors. This factory checks
/// actual backing again; it cannot retroactively admit their creation.
pub struct PublisherDeclaration {
    pub(crate) artifact: NamedDigest,
    pub(crate) selection: CheckpointTensorSelection,
    pub(crate) expected_config: Digest,
    pub(crate) expected_semantic_plan: Digest,
    pub(crate) selected_bytes: usize,
    pub(crate) dtype: SourceDtype,
    pub(crate) endian: SourceEndian,
    pub(crate) tensor_offset: u64,
}

impl PublisherDeclaration {
    /// Typed Rust handoff only. The real guard must already cover argument
    /// backing while those arguments are created; this does not admit parsing.
    #[allow(clippy::too_many_arguments)]
    pub fn from_reserved(
        guard: OwnedExternalCharge,
        artifact: NamedDigest,
        selection: CheckpointTensorSelection,
        expected_config: Digest,
        expected_semantic_plan: Digest,
        selected_bytes: usize,
        dtype: SourceDtype,
        endian: SourceEndian,
        tensor_offset: u64,
    ) -> Result<OwnedCharged<Self>, InputError> {
        let backing = add(
            selection
                .checked_metadata_backing_bytes()
                .map_err(InputError::Source)?,
            add(
                artifact.id.capacity(),
                add(
                    artifact.digest.backing_capacity_bytes(),
                    add(
                        expected_config.backing_capacity_bytes(),
                        expected_semantic_plan.backing_capacity_bytes(),
                    )?,
                )?,
            )?,
        )?;
        if guard.payload_bytes() < backing {
            return Err(InputError::PublisherOwner);
        }
        guard
            .try_build(|_| {
                Ok(Self {
                    artifact,
                    selection,
                    expected_config,
                    expected_semantic_plan,
                    selected_bytes,
                    dtype,
                    endian,
                    tensor_offset,
                })
            })
            .map_err(InputError::Inventory)
    }
    pub(crate) fn backing_bytes(&self) -> Result<usize, InputError> {
        add(
            self.selection
                .checked_metadata_backing_bytes()
                .map_err(InputError::Source)?,
            add(
                self.artifact.id.capacity(),
                add(
                    self.artifact.digest.backing_capacity_bytes(),
                    add(
                        self.expected_config.backing_capacity_bytes(),
                        self.expected_semantic_plan.backing_capacity_bytes(),
                    )?,
                )?,
            )?,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputError {
    BudgetIdentity,
    Resource,
    PublisherOwner,
    PublisherIdentity,
    Inventory(NativeError),
    Source(&'static str),
    Execution(SigmaExecutionError),
    Import(crate::plan_input::Error),
    Work,
}

fn add(a: usize, b: usize) -> Result<usize, InputError> {
    a.checked_add(b).ok_or(InputError::Resource)
}

/// Every field is independently owning. There is no inventory borrowing a plan
/// stored beside it, no native extraction and no detachable charge accessor.
/// Input/pin owners retire before the last original construction reservation.
pub struct OwnedSigmaInput {
    pin: ExpectedSigmaSourcePin,
    selection: CheckpointTensorSelection,
    semantic: OwnedPlanInventory,
    publisher: OwnedCharged<PublisherDeclaration>,
    _charge: OwnedExternalCharge,
}

/// Opaque original input custody, retained by external Rust session owners.
/// No pin/plan/source/guard extraction or lifetime extension is available.
pub struct OwnedSigmaInputRetention {
    semantic: OwnedPlanInventory,
    publisher: OwnedCharged<PublisherDeclaration>,
    charge: OwnedExternalCharge,
}

impl OwnedSigmaInputRetention {
    pub fn preparation_work(&self) -> Option<PreparationWork> {
        self.semantic.preparation_work
    }

    pub fn is_budget(&self, budget: &pllm_garble::shared_allocation::Budget) -> bool {
        self.charge.is_for(budget)
            && std::ptr::eq(self.semantic.budget(), budget)
            && std::ptr::eq(self.publisher.budget(), budget)
            && self
                .semantic
                .raw
                .as_ref()
                .is_none_or(|raw| std::ptr::eq(raw.budget(), budget))
    }
}

struct OwnedPlanInventory {
    inventory: OwnedCharged<DecoderMatrixInventory>,
    input: OwnedCharged<DecoderPlan>,
    // Original immutable bytes, not a reconstructed/re-encoded surrogate.
    // Kept through the existing session residue with all original charges.
    raw: Option<OwnedCharged<Vec<u8>>>,
    preparation_work: Option<PreparationWork>,
}

impl OwnedPlanInventory {
    fn inventory(&self) -> &DecoderMatrixInventory {
        self.inventory.payload()
    }
    fn plan(&self) -> &DecoderPlan {
        self.input.payload()
    }
    fn budget(&self) -> &pllm_garble::shared_allocation::Budget {
        self.input.budget()
    }
}

pub fn inventory_bootstrap_bytes() -> Result<usize, InputError> {
    owned_inventory_bootstrap_bytes().map_err(InputError::Inventory)
}

struct Admission {
    pin: Option<ExpectedSigmaSourcePin>,
    selection: Option<CheckpointTensorSelection>,
    semantic: OwnedPlanInventory,
    publisher: OwnedCharged<PublisherDeclaration>,
    charge: OwnedExternalCharge,
}

#[derive(Clone, Copy, Debug)]
struct Quote {
    publisher_owner_bytes: usize,
    extra_bytes: usize,
    payload_bytes: usize,
}

// Scalar sizing only: caller already holds a real working reservation before
// constructing this function's quote/argument/result population.
pub fn bootstrap_bytes() -> Result<usize, InputError> {
    let mut n = size_of::<Admission>();
    n = add(n, size_of::<OwnedSigmaInput>())?;
    n = add(n, size_of::<Quote>())?;
    n = add(n, size_of::<Result<Quote, InputError>>())?;
    n = add(n, size_of::<SigmaSourceConstructionResources>())?;
    n = add(
        n,
        size_of::<Result<SigmaSourceConstructionResources, &'static str>>(),
    )?;
    n = add(n, size_of::<Result<OwnedSigmaInput, InputError>>())?;
    n = add(
        n,
        size_of::<Result<OwnedCharged<DecoderMatrixInventory>, NativeError>>(),
    )?;
    n = add(n, size_of::<NamedDigest>())?;
    n = add(n, size_of::<CheckpointTensorSelection>())?;
    n = add(
        n,
        size_of::<Result<CheckpointTensorSelection, InputError>>(),
    )?;
    n = add(n, size_of::<Result<ExpectedSigmaSourcePin, &'static str>>())?;
    n = add(n, size_of::<Box<str>>())?;
    n = add(n, size_of::<String>())?;
    n = add(n, size_of::<Result<String, InputError>>())?;
    n = add(n, size_of::<Box<[usize]>>())?;
    n = add(n, size_of::<Vec<usize>>())?;
    n = add(n, size_of::<Result<Vec<usize>, InputError>>())?;
    n = add(n, size_of::<Result<(), NativeError>>())?;
    n = add(n, size_of::<[u8; 32]>())?;
    n = add(n, size_of::<Box<[std::mem::MaybeUninit<u8>]>>())?;
    n = add(n, size_of::<Box<[u8]>>())?;
    n = add(n, size_of::<Vec<u8>>())?;
    n = add(n, size_of::<Operation<'static>>())?;
    n = add(n, size_of::<OwnedSigmaInputRetention>())?;
    n = add(
        n,
        size_of::<(
            OwnedSigmaExecutionMask,
            OwnedSigmaExecutionRun,
            OwnedSigmaInputRetention,
        )>(),
    )?;
    n = add(
        n,
        size_of::<
            Result<
                (
                    OwnedSigmaExecutionMask,
                    OwnedSigmaExecutionRun,
                    OwnedSigmaInputRetention,
                ),
                InputError,
            >,
        >(),
    )?;
    n = add(n, size_of::<SigmaLinearActivationRequest<'static>>())?;
    n = add(n, size_of::<SigmaSourceBytes<'static>>())?;
    n = add(
        n,
        size_of::<Result<(OwnedSigmaExecutionMask, OwnedSigmaExecutionRun), SigmaExecutionError>>(),
    )?;
    n = add(
        n,
        size_of::<Result<(OwnedSigmaExecutionMask, OwnedSigmaExecutionRun), InputError>>(),
    )?;
    n = add(n, size_of::<std::array::IntoIter<usize, 8>>())?;
    n = add(n, size_of::<[usize; 8]>())?;
    n = add(n, size_of::<PlanInputLimits>())?;
    n = add(n, size_of::<PreparationWork>())?;
    n = add(n, size_of::<Result<PreparationWork, InputError>>())?;
    n = add(n, size_of::<OwnedCharged<Vec<u8>>>())?;
    n = add(n, size_of::<Option<OwnedCharged<Vec<u8>>>>())?;
    n = add(n, size_of::<crate::plan_input::ImportQuote<'_>>())?;
    n = add(
        n,
        size_of::<Result<crate::plan_input::ImportQuote<'_>, crate::plan_input::Error>>(),
    )?;
    n = add(
        n,
        size_of::<Result<OwnedCharged<DecoderPlan>, crate::plan_input::Error>>(),
    )?;
    n = add(
        n,
        size_of::<Result<OwnedCharged<DecoderPlan>, NativeError>>(),
    )?;
    n = add(n, size_of::<[&str; 7]>())?;
    n = add(n, size_of::<std::array::IntoIter<&str, 7>>())?;
    n = add(n, size_of::<[&[usize]; 3]>())?;
    n = add(n, size_of::<std::array::IntoIter<&[usize], 3>>())?;
    n = add(n, size_of::<[&str; 3]>())?;
    n = add(n, size_of::<[u128; 7]>())?;
    n = add(n, size_of::<std::array::IntoIter<u128, 7>>())?;
    n = add(n, size_of::<Result<u128, InputError>>())?;
    add(n, size_of::<crate::plan_input::Limits>())
}

/// Disjoint preparation slices, not a Budget or a source trust declaration.
/// The inventory slice is prepaid IN FULL, including unused credit; consumed
/// work is never guessed as zero and no later stage can borrow that allowance.
#[derive(Clone, Copy, Debug)]
pub struct PlanInputLimits {
    pub parse: crate::plan_input::Limits,
    pub inventory_work: u128,
    pub max_work: u128,
}

/// Read-only prepaid accounting, retained with the original plan/byte owners.
/// Units are those of the parser/inventory plus the explicitly enumerated
/// metadata span/coordinate/copy visits below, NOT native CPU instructions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreparationWork {
    parse: u128,
    inventory: u128,
    metadata: u128,
    total: u128,
}
impl PreparationWork {
    pub fn parse_prepaid(self) -> u128 {
        self.parse
    }
    pub fn inventory_prepaid(self) -> u128 {
        self.inventory
    }
    pub fn metadata_prepaid(self) -> u128 {
        self.metadata
    }
    pub fn total_prepaid(self) -> u128 {
        self.total
    }
}

fn work_add(a: u128, b: u128) -> Result<u128, InputError> {
    a.checked_add(b).ok_or(InputError::Work)
}
fn work_mul(a: u128, b: u128) -> Result<u128, InputError> {
    a.checked_mul(b).ok_or(InputError::Work)
}

fn coordinate_work(len: usize) -> Result<u128, InputError> {
    let n = len as u128;
    // coordinates / SelectedSourceTuple::checked_count: outer visits, range
    // comparisons and worst-case prefix contains comparisons; no tree guesses.
    work_add(work_mul(n, 2)?, work_mul(n, n.saturating_sub(1))? / 2)
}

fn metadata_work(p: &PublisherDeclaration) -> Result<u128, InputError> {
    // This quotation reads lengths ONLY. Fixed layout/table arithmetic is not
    // work fuel, exactly as in the underlying parser and inventory contracts.
    let s = &p.selection;
    let text = [
        s.checkpoint.as_str(),
        s.immutable_revision.as_str(),
        s.tensor.as_str(),
        s.input_tensor.as_str(),
        s.weight_layout.as_str(),
        s.input_layout.as_str(),
        s.encoding.as_str(),
    ];
    let mut text_bytes = 0;
    for t in text {
        text_bytes = work_add(text_bytes, t.len() as u128)?;
    }
    let mut coords = 0;
    let mut coord_checks = 0;
    for c in [
        &s.output_coordinates[..],
        &s.inner_coordinates[..],
        &s.input_row_coordinates[..],
    ] {
        coords = work_add(coords, c.len() as u128)?;
        coord_checks = work_add(coord_checks, coordinate_work(c.len())?)?;
    }
    // geometry: seven text-table visits, three shape-product-table visits,
    // layout/encoding comparisons, coordinate checks, full-range eq traversals
    // (both iterators, comparison, terminating probe). BOTH construction quotes.
    let geometry = work_add(
        7 + 3 + 2 * 3,
        work_add(text_bytes, work_add(coord_checks, work_mul(coords, 3)?)?)?,
    )?;
    let geometry = work_mul(geometry, 2)?;
    // Two source construction quotes decode three digests each; admitted pin
    // initialization decodes two more, shared publisher constructor one more.
    // Bundle identity check decodes two digests and compares two raw identities.
    let identities = work_add(work_mul(3 + 3 + 2 + 1 + 2, 64)?, 2 * 32)?;
    // exact_digest: decode, hex encoding, owned validation, eventual retirement.
    let artifact_digest = work_mul(4, 64)?;
    // Two complete selection copies plus their retirement; tuple's three strings
    // and two coordinate vectors get a third independent copy and retirement.
    let selection_bytes = work_add(text_bytes, work_mul(coords, size_of::<usize>() as u128)?)?;
    let tuple_bytes = work_add(
        work_add(
            s.checkpoint.len() as u128,
            work_add(s.immutable_revision.len() as u128, s.tensor.len() as u128)?,
        )?,
        work_mul(
            work_add(
                s.output_coordinates.len() as u128,
                s.inner_coordinates.len() as u128,
            )?,
            size_of::<usize>() as u128,
        )?,
    )?;
    let copies = work_add(
        work_mul(4, selection_bytes)?,
        work_mul(2, work_add(tuple_bytes, p.artifact.id.len() as u128)?)?,
    )?;
    // Shared selected-source validation: revision bytes, two coordinate checks
    // and their outer table visits. Existing <=256 checks are NOT relaxed.
    let selected = work_add(
        s.immutable_revision.len() as u128,
        work_add(
            3 + 2,
            work_add(
                coordinate_work(s.output_coordinates.len())?,
                coordinate_work(s.inner_coordinates.len())?,
            )?,
        )?,
    )?;
    // Exact helper/factory/capacity-check events: two selections (seven strings,
    // three vectors each), artifact string+digest, tuple three strings+two vectors.
    let helpers = work_mul(3, 2 * (7 + 3) + 2 + 3 + 2)?;
    [
        geometry,
        identities,
        artifact_digest,
        copies,
        selected,
        helpers,
        2,
    ]
    .into_iter()
    .try_fold(0, work_add)
}

/// Consuming Rust-only serialized-input handoff. Incoming byte/declaration
/// creation remains separately admitted by their original owners. All three
/// working charges ALREADY cover their bootstrap; no Python exporter is read.
/// Preparation credit is partitioned before the first allocation: metadata and
/// FULL inventory ceiling first, then parser scan/build/census/numeric envelope.
/// Source verification still occurs only in execute_retained's existing facade.
pub fn prepare_from_plan_bytes(
    raw: OwnedCharged<Vec<u8>>,
    publisher: OwnedCharged<PublisherDeclaration>,
    parse_working: OwnedExternalCharge,
    source_working: OwnedExternalCharge,
    inventory_working: OwnedExternalCharge,
    limits: PlanInputLimits,
) -> Result<OwnedSigmaInput, InputError> {
    if !std::ptr::eq(publisher.budget(), raw.budget())
        || !parse_working.is_for(raw.budget())
        || !source_working.is_for(raw.budget())
        || !inventory_working.is_for(raw.budget())
    {
        return Err(InputError::BudgetIdentity);
    }
    if source_working.payload_bytes() < bootstrap_bytes()?
        || inventory_working.payload_bytes() < inventory_bootstrap_bytes()?
        || parse_working.payload_bytes()
            < crate::plan_input::bootstrap_bytes().map_err(InputError::Import)?
    {
        return Err(InputError::Resource);
    }
    if add(
        size_of::<OwnedCharged<PublisherDeclaration>>(),
        publisher.payload().backing_bytes()?,
    )? > publisher.charged_bytes()
    {
        return Err(InputError::PublisherOwner);
    }
    let metadata = metadata_work(publisher.payload())?;
    let reserved = work_add(metadata, limits.inventory_work)?;
    let remaining = limits
        .max_work
        .checked_sub(reserved)
        .ok_or(InputError::Work)?;
    let mut parse_limits = limits.parse;
    parse_limits.max_work = parse_limits.max_work.min(remaining);
    let quote =
        crate::plan_input::quote(&raw, parse_working, parse_limits).map_err(InputError::Import)?;
    let parse = quote.total_work_bound().map_err(InputError::Import)?;
    let total = work_add(reserved, parse)?;
    if total > limits.max_work {
        return Err(InputError::Work);
    }
    let preparation_work = PreparationWork {
        parse,
        inventory: limits.inventory_work,
        metadata,
        total,
    };
    let plan = quote.construct().map_err(InputError::Import)?;
    let mut prepared = prepare(
        plan,
        publisher,
        source_working,
        inventory_working,
        limits.inventory_work,
    )?;
    prepared.semantic.raw = Some(raw);
    prepared.semantic.preparation_work = Some(preparation_work);
    Ok(prepared)
}

fn quote(publisher: &OwnedCharged<PublisherDeclaration>) -> Result<Quote, InputError> {
    let p = publisher.payload();
    let publisher_owner_bytes = add(
        size_of::<OwnedCharged<PublisherDeclaration>>(),
        p.backing_bytes()?,
    )?;
    if publisher_owner_bytes > publisher.charged_bytes() {
        return Err(InputError::PublisherOwner);
    }
    // Pin quote over ORIGINAL capacities conservatively covers exact cloned
    // artifact/selection arguments. Original declaration remains independently
    // charged; expected digest arguments borrow it, never clone its strings.
    let pin = ExpectedSigmaSourcePin::checked_construction_quote(
        &p.artifact,
        &p.selection,
        &p.expected_config,
        &p.expected_semantic_plan,
        0,
    )
    .map_err(InputError::Source)?;
    let extra_bytes = add(
        bootstrap_bytes()?,
        p.selection
            .checked_metadata_backing_bytes()
            .map_err(InputError::Source)?,
    )?;
    let payload_bytes = add(extra_bytes, pin.peak_bytes)?;
    Ok(Quote {
        publisher_owner_bytes,
        extra_bytes,
        payload_bytes,
    })
}

/// Consumes genuine original owning inputs. Both working guards must ALREADY
/// cover their bootstrap envelopes. Source peak is reserved before calling the
/// inventory factory; that factory separately prequotes/grows before allocating.
/// No caller-supplied baseline substitutes for incoming backing or authority.
// Typed owner handoff; serialized input MUST use prepare_from_plan_bytes.
#[allow(dead_code)]
pub fn prepare(
    plan: OwnedCharged<DecoderPlan>,
    publisher: OwnedCharged<PublisherDeclaration>,
    mut source_working: OwnedExternalCharge,
    inventory_working: OwnedExternalCharge,
    max_inventory_work: u128,
) -> Result<OwnedSigmaInput, InputError> {
    if !source_working.is_for(plan.budget())
        || !inventory_working.is_for(plan.budget())
        || !std::ptr::eq(publisher.budget(), plan.budget())
    {
        return Err(InputError::BudgetIdentity);
    }
    if source_working.payload_bytes() < bootstrap_bytes()? {
        return Err(InputError::Resource);
    }
    if inventory_working.payload_bytes() < inventory_bootstrap_bytes()? {
        return Err(InputError::Resource);
    }
    let resources = quote(&publisher)?;
    if source_working.payload_bytes() < resources.payload_bytes {
        source_working
            .resize_payload(resources.payload_bytes)
            .map_err(|_| InputError::Resource)?;
    }
    if resources.publisher_owner_bytes > publisher.charged_bytes() {
        return Err(InputError::PublisherOwner);
    }
    let inventory =
        admit_inventory_for_original_owner(&plan, inventory_working, max_inventory_work)
            .map_err(InputError::Inventory)?;
    let semantic = OwnedPlanInventory {
        input: plan,
        inventory,
        raw: None,
        preparation_work: None,
    };
    let mut admitted = Admission {
        pin: None,
        selection: None,
        semantic,
        publisher,
        charge: source_working,
    };
    let p = admitted.publisher.payload();
    if p.expected_config.as_sha256() != admitted.semantic.inventory().config_identity()
        || p.expected_semantic_plan.as_sha256()
            != admitted.semantic.inventory().semantic_plan_identity()
    {
        return Err(InputError::PublisherIdentity);
    }
    // The first allocating copy occurs only after BOTH complete peaks have
    // their real original-counter reservations and genuine plan census passed.
    let artifact = NamedDigest {
        id: exact_string(&p.artifact.id)?,
        digest: exact_digest(&p.artifact.digest)?,
    };
    let pin_selection = exact_selection(&p.selection)?;
    admitted.selection = Some(exact_selection(&p.selection)?);
    admitted.pin = Some(
        ExpectedSigmaSourcePin::from_trusted_publisher_admitted(
            artifact,
            p.selected_bytes,
            pin_selection,
            p.dtype,
            p.endian,
            p.tensor_offset,
            &p.expected_config,
            &p.expected_semantic_plan,
            resources.extra_bytes,
            admitted.charge.payload_bytes(),
        )
        .map_err(InputError::Source)?,
    );
    Ok(OwnedSigmaInput {
        pin: admitted.pin.take().ok_or(InputError::Resource)?,
        selection: admitted.selection.take().ok_or(InputError::Resource)?,
        semantic: admitted.semantic,
        publisher: admitted.publisher,
        _charge: admitted.charge,
    })
}

fn copy_factory() -> Result<(), InputError> {
    #[cfg(test)]
    COPY_FACTORIES.with(|state| {
        let (count, fail, panic) = state.get();
        if panic == Some(count) {
            panic!("SIGMA input copy panic");
        }
        if fail == Some(count) {
            return Err(InputError::Source("injected input copy failure"));
        }
        state.set((count + 1, fail, panic));
        Ok(())
    })?;
    Ok(())
}

fn exact_string(value: &str) -> Result<String, InputError> {
    copy_factory()?;
    let result = String::from(Box::<str>::from(value));
    if result.capacity() != value.len() {
        return Err(InputError::Resource);
    }
    Ok(result)
}

fn exact_coordinates(value: &[usize]) -> Result<Vec<usize>, InputError> {
    copy_factory()?;
    let result = Box::<[usize]>::from(value).into_vec();
    if result.capacity() != value.len() {
        return Err(InputError::Resource);
    }
    Ok(result)
}

fn exact_digest(value: &Digest) -> Result<Digest, InputError> {
    copy_factory()?;
    // Shared digest constructor uses its exact 64-byte boxed backing policy.
    Ok(Digest::from_sha256(value.as_sha256()))
}

fn exact_selection(s: &CheckpointTensorSelection) -> Result<CheckpointTensorSelection, InputError> {
    Ok(CheckpointTensorSelection {
        checkpoint: exact_string(&s.checkpoint)?,
        immutable_revision: exact_string(&s.immutable_revision)?,
        content_sha256: s.content_sha256,
        tensor: exact_string(&s.tensor)?,
        original_weight_shape: s.original_weight_shape,
        weight_layout: exact_string(&s.weight_layout)?,
        output_coordinates: exact_coordinates(&s.output_coordinates)?,
        inner_coordinates: exact_coordinates(&s.inner_coordinates)?,
        input_tensor: exact_string(&s.input_tensor)?,
        original_input_shape: s.original_input_shape,
        input_layout: exact_string(&s.input_layout)?,
        input_row_coordinates: exact_coordinates(&s.input_row_coordinates)?,
        encoding: exact_string(&s.encoding)?,
    })
}

/// Request data whose plan/selection are supplied only from the owning bundle.
/// Caller-owned typed edges/slices keep the existing facade's import/admission
/// contract. This is not admission of their Python/serialized origins.
pub struct Operation<'a> {
    pub mode: DecoderMode,
    pub linear_operation: &'a str,
    pub activation_operation: &'a str,
    pub next_operation: &'a str,
    pub source: SourceIdentity,
    pub input: EdgeDescriptor,
    pub private_weights: EdgeDescriptor,
    pub complete_sum: EdgeDescriptor,
    pub rescaled: EdgeDescriptor,
    pub activation_output: EdgeDescriptor,
    pub rescale_operation: [u8; 32],
    pub next_use: ConsumerUse,
    pub input_domain: IntegerInterval,
    pub source_coefficients: &'a [f64],
    pub encoded_private_coefficients: &'a [i64],
}

impl OwnedSigmaInput {
    pub fn preparation_work(&self) -> Option<PreparationWork> {
        self.semantic.preparation_work
    }
    /// D27 remains the existing returned run's into_roles path. No native handle,
    /// pin, inventory, source declaration or reservation is extractable here.
    #[allow(dead_code)]
    pub(crate) fn execute(
        self,
        supplied: SigmaSourceBytes<'_>,
        operation: Operation<'_>,
        root: &OwnedBudget,
        dealer: &mut typed::OwnedLocalDealer,
        input: &[i64],
    ) -> Result<(OwnedSigmaExecutionMask, OwnedSigmaExecutionRun), InputError> {
        self.execute_retained(supplied, operation, root, dealer, input)
            .map(|(mask, run, _retention)| (mask, run))
    }

    pub fn is_budget(&self, budget: &pllm_garble::shared_allocation::Budget) -> bool {
        self._charge.is_for(budget)
            && std::ptr::eq(self.semantic.budget(), budget)
            && std::ptr::eq(self.publisher.budget(), budget)
            && self
                .semantic
                .raw
                .as_ref()
                .is_none_or(|raw| std::ptr::eq(raw.budget(), budget))
    }

    /// Session handoff preserving original charged plan/declaration and complete
    /// construction reservation. The opaque residue contains no native handle.
    pub fn execute_retained(
        self,
        supplied: SigmaSourceBytes<'_>,
        operation: Operation<'_>,
        root: &OwnedBudget,
        dealer: &mut typed::OwnedLocalDealer,
        input: &[i64],
    ) -> Result<
        (
            OwnedSigmaExecutionMask,
            OwnedSigmaExecutionRun,
            OwnedSigmaInputRetention,
        ),
        InputError,
    > {
        if !std::ptr::eq(root.budget(), self.semantic.budget()) {
            return Err(InputError::BudgetIdentity);
        }
        let Self {
            pin,
            selection,
            semantic,
            publisher,
            _charge: charge,
        } = self;
        let request = SigmaLinearActivationRequest {
            plan: semantic.plan(),
            selection,
            mode: operation.mode,
            linear_operation: operation.linear_operation,
            activation_operation: operation.activation_operation,
            next_operation: operation.next_operation,
            source: operation.source,
            input: operation.input,
            private_weights: operation.private_weights,
            complete_sum: operation.complete_sum,
            rescaled: operation.rescaled,
            activation_output: operation.activation_output,
            rescale_operation: operation.rescale_operation,
            next_use: operation.next_use,
            input_domain: operation.input_domain,
            source_coefficients: operation.source_coefficients,
            encoded_private_coefficients: operation.encoded_private_coefficients,
        };
        let result = execute_owned_sigma_linear_activation(
            pin,
            supplied,
            request,
            semantic.inventory(),
            root,
            dealer,
            input,
        )
        .map_err(InputError::Execution);
        result.map(|(mask, run)| {
            (
                mask,
                run,
                OwnedSigmaInputRetention {
                    semantic,
                    publisher,
                    charge,
                },
            )
        })
    }
}

#[cfg(test)]
thread_local! {
    static COPY_FACTORIES: std::cell::Cell<(usize, Option<usize>, Option<usize>)> = const { std::cell::Cell::new((0, None, None)) };
}

#[cfg(test)]
#[path = "input_owner_tests.rs"]
mod tests;
