//! Borrow-free recipe data. This is NOT complete request Storage or execution
//! authority; original context/history and charged sources must accompany it.
use super::super::compact_attention_softmax::owning::{Conversion, Invocations};
use super::*;
use crate::replicated_decoder::program::source_entry::compact_pending::handles::{
    NormHandle, SourceScope, TensorHandle,
};

#[path = "compact_owned_epoch/compact_owned_recipe.rs"]
pub(in crate::replicated_decoder::program) mod recipe;

struct Bias {
    owner: usize,
    source: TensorHandle,
    norms: Option<NormHandle>,
    floor_bounds: Interval,
    coefficients: Affine,
}
enum Kind {
    Values(super::super::compact_attention_values::OwnedValues),
    Softmax,
    Scale(Arc<super::super::attention_scale::Rows>),
    Scores(super::super::compact_attention_scores::Scores),
    CausalMask,
    CacheSuffix,
    KvInitialize {
        layer: u64,
        state: pllm_models::StateKind,
    },
    Rotary(super::super::compact_rotary::Rows),
    Lookup {
        weight: usize,
        source: TensorHandle,
        plan: CompactTokenLookup,
    },
    Projection {
        weight: usize,
        source: TensorHandle,
        norms: Option<NormHandle>,
        geometry: [usize; 3],
        raw_bounds: Interval,
        row_energy: Option<u128>,
        input_energy: Option<u128>,
        bias: Option<Bias>,
    },
    Silu,
    Rms {
        gamma: TensorHandle,
        input_energy: Option<u128>,
        output_energy: u128,
        program: Arc<tensor::Program>,
        owns_program: bool,
        checked: Option<compact_validity::OwnedCheckedRows>,
        row_bounds: Interval,
    },
    LastToken {
        row_energy: Option<u128>,
    },
    Feedback,
    Reshape,
    Argmax(CompactArgmax),
    Binary {
        right: PublishedView,
        product: bool,
        row_energy: Option<u128>,
    },
}
struct Op {
    first: usize,
    input: PublishedView,
    output: PublishedView,
    kind: Kind,
    prepared: Option<MapOwner>,
}

pub(in crate::replicated_decoder::program) struct OwnedEpoch {
    association: request_origin::compact_storage::owning::EpochAssociation,
    instance: source_registry::Instance,
    ops: Vec<Op>,
    invocations: Invocations,
    imports: Vec<Import>,
    recipe_names: Vec<String>,
    frontier: Vec<Frontier>,
    exports: Vec<PublishedView>,
    output: Option<usize>,
    origin: Arc<()>,
    next: usize,
    request_prelude: Option<request_origin::compact_storage::Prelude>,
    binding: [u8; 32],
    work: WorkBudget,
    sticky: Option<usize>,
    validity_epoch: usize,
}

/// Sole combined inspection entry. Private fields prevent constructing a loan
/// from independently selected imports, source custody or epoch data.
type EpochCustody =
    crate::replicated_decoder::program::source_entry::compact_pending::SourceCustody<
        request_origin::compact_storage::owning::OwnedEmission<OwnedEpoch>,
    >;
type HistoryCustody =
    crate::replicated_decoder::program::source_entry::compact_pending::SourceCustody<
        request_origin::compact_storage::owning::OwnedEmission<
            request_origin::compact_storage::compact_history::CompletionLoan,
        >,
    >;
#[derive(Clone, Copy)]
enum InspectionRoot<'r> {
    Epoch(&'r EpochCustody),
    History(&'r HistoryCustody),
}
pub(in crate::replicated_decoder::program) struct InspectionLoan<'r> {
    custody: InspectionRoot<'r>,
    imports: &'r Builder,
    epoch: &'r OwnedEpoch,
}

/// Loan of the completed ORIGINAL source graph's feedback, not an arbitrary
/// exported scalar. Creation is sealed by full source traversal and association.
pub(in crate::replicated_decoder::program) struct FeedbackLoan<'r> {
    epoch: &'r OwnedEpoch,
    word: (usize, usize),
    bounds: Interval,
}
impl FeedbackLoan<'_> {
    pub(in crate::replicated_decoder::program) fn word(&self) -> (usize, usize) {
        self.word
    }
    pub(in crate::replicated_decoder::program) fn sticky(&self) -> Option<usize> {
        self.epoch.sticky
    }
    pub(in crate::replicated_decoder::program) fn next(&self) -> usize {
        self.epoch.next
    }
    pub(in crate::replicated_decoder::program) fn bounds(&self) -> Interval {
        self.bounds
    }
}
impl<'r> InspectionLoan<'r> {
    /// Authenticated ORIGINAL key -> import owner -> same owned source tensor.
    /// Digest/statistics are consistency checks, never the binding authority.
    pub(in crate::replicated_decoder::program) fn checked_import_source(
        &self,
        owner: usize,
    ) -> Result<(&'r str, &'r SourceTensor), Error> {
        let mut attempt = self.attempt()?;
        let (key, shape, source) = match self.custody {
            InspectionRoot::Epoch(c) => {
                let (key, shape) = c.produced().checked_import_binding(owner)?;
                (key, shape, c.resolve_file_import(key)?)
            }
            InspectionRoot::History(c) => {
                let (key, shape) = c.produced().checked_import_binding(owner)?;
                (key, shape, c.resolve_file_import(key)?)
            }
        };
        let node = self.imports.nodes.get(owner).ok_or(Error::Binding)?;
        let Code::PublicInput {
            key: node_key,
            digest,
            row_l1,
            row_squared_norm,
            row_l1_squared_sum,
        } = &node.code
        else {
            return Err(Error::Binding);
        };
        let visits = key
            .len()
            .checked_add(node_key.len())
            .and_then(|n| n.checked_add(shape.len()))
            .and_then(|n| n.checked_add(1))
            .ok_or(Error::Bounds)?;
        match self.custody {
            InspectionRoot::Epoch(c) => c.produced().prepay_import_metadata(visits)?,
            InspectionRoot::History(c) => c.produced().prepay_import_metadata(visits)?,
        };
        let lanes = shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d).ok_or(Error::Bounds))?;
        if key != node_key.as_str()
            || shape != source.shape.as_slice()
            || node.lanes != lanes
            || node.bounds != Interval::new(source.lower, source.upper)?
            || digest != &source.digest
            || row_l1 != &source.row_l1
            || row_squared_norm != &source.row_squared_norm
            || row_l1_squared_sum != &source.row_l1_squared_sum
        {
            return Err(Error::Binding);
        }
        attempt.commit();
        Ok((node_key.as_str(), source))
    }
    pub(in crate::replicated_decoder::program) fn imports(&self) -> &'r Builder {
        self.imports
    }
    pub(in crate::replicated_decoder::program) fn epoch(&self) -> &'r OwnedEpoch {
        self.epoch
    }
    pub(in crate::replicated_decoder::program) fn sources(
        &self,
        index: usize,
    ) -> Result<RecipeSources<'r>, Error> {
        match self.custody {
            InspectionRoot::Epoch(c) => resolve_recipe_sources(c, self.epoch, index),
            InspectionRoot::History(c) => resolve_recipe_sources(c, self.epoch, index),
        }
    }
    pub(in crate::replicated_decoder::program) fn attempt(
        &self,
    ) -> Result<request_origin::compact_storage::owning::InspectionAttempt<'r>, Error> {
        match self.custody {
            InspectionRoot::Epoch(c) => c.produced().inspection_attempt(),
            InspectionRoot::History(c) => c.produced().inspection_attempt(),
        }
    }
    pub(in crate::replicated_decoder::program) fn charge(
        &self,
        bytes: usize,
        visits: u64,
    ) -> Result<pllm_garble::shared_allocation::ExternalCharge<'r>, Error> {
        match self.custody {
            InspectionRoot::Epoch(c) => c.produced().charge_inspection(bytes, visits),
            InspectionRoot::History(c) => c.produced().charge_inspection(bytes, visits),
        }
    }
}

/// On-demand lexical source operands for the canonical recipe reader. Nothing
/// is copied or rebound; all references are tied to the whole retained custody.
#[derive(Clone, Copy)]
pub(in crate::replicated_decoder::program) struct RecipeSources<'r> {
    primary: Option<&'r SourceTensor>,
    norms: Option<&'r crate::replicated_decoder::BlockNorms>,
    bias: Option<&'r SourceTensor>,
    bias_norms: Option<&'r crate::replicated_decoder::BlockNorms>,
}
impl<'r> RecipeSources<'r> {
    pub(in crate::replicated_decoder::program) fn primary(&self) -> Option<&'r SourceTensor> {
        self.primary
    }
    pub(in crate::replicated_decoder::program) fn norms(
        &self,
    ) -> Option<&'r crate::replicated_decoder::BlockNorms> {
        self.norms
    }
    pub(in crate::replicated_decoder::program) fn bias(&self) -> Option<&'r SourceTensor> {
        self.bias
    }
    pub(in crate::replicated_decoder::program) fn bias_norms(
        &self,
    ) -> Option<&'r crate::replicated_decoder::BlockNorms> {
        self.bias_norms
    }
}
impl
    crate::replicated_decoder::program::source_entry::compact_pending::SourceCustody<
        request_origin::compact_storage::owning::OwnedEmission<OwnedEpoch>,
    >
{
    pub(in crate::replicated_decoder::program) fn inspect_epoch(
        &self,
    ) -> Result<InspectionLoan<'_>, Error> {
        let emission = self.produced();
        self.check_emission(emission)?;
        let epoch = &emission.produced;
        self.check_source_epoch(epoch.instance, emission.source_plan())?;
        let imports = emission.checked_imports(&epoch.association)?;
        if epoch.work != emission.work() {
            return Err(Error::Binding);
        }
        Ok(InspectionLoan {
            custody: InspectionRoot::Epoch(self),
            imports,
            epoch,
        })
    }
    pub(in crate::replicated_decoder::program) fn borrow_recipe_sources(
        &self,
        index: usize,
    ) -> Result<RecipeSources<'_>, Error> {
        let emission = self.produced();
        emission.check_open()?;
        let epoch = &emission.produced;
        self.check_source_epoch(epoch.instance, emission.source_plan())?;
        resolve_recipe_sources(self, epoch, index)
    }
}

impl HistoryCustody {
    pub(in crate::replicated_decoder::program) fn inspect_history_epoch(
        &self,
        index: usize,
    ) -> Result<InspectionLoan<'_>, Error> {
        let emission = self.produced();
        self.check_emission(emission)?;
        let epoch = emission
            .produced
            .epochs()
            .get(index)
            .ok_or(Error::Binding)?;
        self.check_source_epoch(epoch.instance, emission.source_plan())?;
        let imports = emission.checked_epoch_imports(&epoch.association, index)?;
        Ok(InspectionLoan {
            custody: InspectionRoot::History(self),
            imports,
            epoch,
        })
    }
}

// Shared pure source resolution. Callers enter through the exact combined
// custody loan above; no independent Manifest/epoch constructor is exposed.
fn resolve_recipe_sources<'r, R>(
    custody: &'r crate::replicated_decoder::program::source_entry::compact_pending::SourceCustody<
        R,
    >,
    epoch: &'r OwnedEpoch,
    index: usize,
) -> Result<RecipeSources<'r>, Error> {
    let op = epoch.ops.get(index).ok_or(Error::Binding)?;
    let mut result = RecipeSources {
        primary: None,
        norms: None,
        bias: None,
        bias_norms: None,
    };
    match &op.kind {
        Kind::Lookup { source, .. } => result.primary = Some(custody.resolve_tensor(source)?),
        Kind::Projection {
            source,
            norms,
            bias,
            ..
        } => {
            result.primary = Some(custody.resolve_tensor(source)?);
            result.norms = norms
                .as_ref()
                .map(|n| custody.resolve_norm(n))
                .transpose()?;
            if let Some(bias) = bias {
                result.bias = Some(custody.resolve_tensor(&bias.source)?);
                result.bias_norms = bias
                    .norms
                    .as_ref()
                    .map(|n| custody.resolve_norm(n))
                    .transpose()?;
            }
        }
        Kind::Rms { gamma, .. } => result.primary = Some(custody.resolve_tensor(gamma)?),
        _ => {}
    }
    Ok(result)
}
impl OwnedEpoch {
    #[cfg(test)]
    pub(in crate::replicated_decoder::program) fn test_negative_programs(
        &self,
    ) -> impl Iterator<Item = (usize, Interval, &Arc<tensor::Program>)> {
        self.ops
            .iter()
            .flat_map(|op| match &op.kind {
                Kind::Argmax(p) => p.programs.as_slice(),
                _ => &[],
            })
            .map(|p| (p.lanes, p.bounds, &p.program))
    }
    pub(in crate::replicated_decoder::program) fn previous_negative(
        &self,
        lanes: usize,
        bounds: Interval,
    ) -> Option<&Arc<tensor::Program>> {
        self.ops.iter().find_map(|op| match &op.kind {
            Kind::Argmax(plan) => plan
                .programs
                .iter()
                .find(|p| p.lanes == lanes && p.bounds == bounds)
                .map(|p| &p.program),
            _ => None,
        })
    }
    #[cfg(test)]
    pub(in crate::replicated_decoder::program) fn test_boundary(&self) -> (usize, Option<usize>) {
        (self.next, self.sticky)
    }
    fn read_start(&self, imports: &Builder) -> Result<(usize, Option<usize>), Error> {
        let first = self
            .request_prelude
            .as_ref()
            .map(|p| p.first())
            .unwrap_or(imports.nodes.len());
        let mut histories = self.imports.iter().filter_map(|i| match i.view.mapping {
            Mapping::History(h) => Some(h),
            _ => None,
        });
        let history = histories.next();
        if histories.next().is_some() {
            return Err(Error::Binding);
        }
        match (self.association.epoch(), history) {
            (0, None) if first == imports.nodes.len() => Ok((first, imports.validity)),
            (1, Some(history)) if first == history.first() => Ok((first, history.sticky())),
            _ => Err(Error::Binding),
        }
    }
    pub(in crate::replicated_decoder::program) fn check_owned_emission<R>(
        &self,
        emission: &request_origin::compact_storage::owning::OwnedEmission<R>,
        epoch: usize,
        instance: source_registry::Instance,
    ) -> Result<(), Error> {
        emission.checked_epoch_imports(&self.association, epoch)?;
        if self.instance != instance {
            return Err(Error::Binding);
        }
        Ok(())
    }
    pub(in crate::replicated_decoder::program) fn feedback_control_bytes() -> Result<usize, Error> {
        std::mem::size_of::<FeedbackLoan>()
            .checked_mul(2)
            .and_then(|n| n.checked_add(std::mem::size_of::<Result<FeedbackLoan, Error>>()))
            .ok_or(Error::Bounds)
    }
    pub(in crate::replicated_decoder::program) fn completed_feedback(
        &self,
        context: &request_origin::compact_storage::EmissionContext<'_>,
    ) -> Result<FeedbackLoan<'_>, Error> {
        self.association.check_context(context)?;
        if self.instance != context.registry().instance()
            || self.work != context.work()
            || self.ops.len() != context.plan().plan().prefill.operations.len()
        {
            return Err(Error::Binding);
        }
        let view = &self
            .frontier
            .get(self.output.ok_or(Error::Binding)?)
            .ok_or(Error::Binding)?
            .view;
        let op = self
            .ops
            .get(view.slot.ok_or(Error::Binding)?)
            .ok_or(Error::Binding)?;
        if !matches!(&op.kind, Kind::Feedback) || view.lanes != 1 {
            return Err(Error::Binding);
        }
        let word = view.word(0)?;
        let mut selected = self
            .ops
            .iter()
            .filter(|op| matches!(&op.kind, Kind::Argmax(_)));
        if selected.next().ok_or(Error::Binding)?.output.word(0)? != word
            || selected.next().is_some()
        {
            return Err(Error::Binding);
        }
        Ok(FeedbackLoan {
            epoch: self,
            word,
            bounds: view.bounds,
        })
    }
    pub(in crate::replicated_decoder::program) fn argmax_lanes(
        &self,
        index: usize,
    ) -> Result<usize, Error> {
        let op = self.ops.get(index).ok_or(Error::Binding)?;
        if !matches!(&op.kind, Kind::Argmax(_)) {
            return Err(Error::Binding);
        }
        Ok(op.input.lanes)
    }
    #[cfg(test)]
    pub(in crate::replicated_decoder::program) fn softmax_programs(
        &self,
    ) -> impl Iterator<Item = &Arc<tensor::Program>> {
        self.invocations.programs()
    }
    pub(in crate::replicated_decoder::program) fn work(&self) -> WorkBudget {
        self.work
    }
    pub(in crate::replicated_decoder::program) fn recipe_count(&self) -> usize {
        self.ops.len()
    }
}

impl CompactGraph<'_, '_> {
    /// Consumes the actual recipe prefix. Not a publication of partial Storage:
    /// this data has no Worker/Program constructor or completion capability.
    pub(in crate::replicated_decoder::program) fn freeze_owned(
        self,
        exports: Vec<PublishedView>,
        scope: &SourceScope<'_>,
    ) -> Result<OwnedEpoch, Error> {
        let context = self.request_admission.ok_or(Error::Binding)?;
        let mut attempt = context.attempt();
        context.check_imports(self.imports)?;
        let q = self.freeze_quote(&exports)?;
        let new = add_bytes(
            q.new_backing_bytes,
            2 * size_of::<Result<FrozenRegion<'_>, Error>>(),
        )?;
        context.admit_phase(new, 0)?;
        context.retain(new)?;
        let frozen = self.freeze(exports, MAX_PROGRAM_BYTES)?;
        let owned = frozen.into_owned(context, scope)?;
        attempt.commit();
        Ok(owned)
    }
}

impl<'a> FrozenRegion<'a> {
    fn into_owned(
        self,
        context: &request_origin::compact_storage::EmissionContext<'a>,
        scope: &SourceScope<'_>,
    ) -> Result<OwnedEpoch, Error> {
        if !self
            .source_context
            .is_some_and(|actual| std::ptr::eq(actual, context))
            || self._prebound_inputs.is_some()
        {
            return Err(Error::Binding);
        }
        context.check_imports(self.imports_owner)?;
        self.validity
            .check_frozen_owner(self.imports_owner, &self.origin)?;
        scope.check_context(context)?;
        let retained = add_bytes(
            array_bytes::<Op>(self.ops.len())?,
            add_bytes(
                2 * size_of::<OwnedEpoch>(),
                2 * size_of::<RecipeSources<'_>>(),
            )?,
        )?;
        let conversion_live = add_bytes(
            retained,
            array_bytes::<(usize, super::super::compact_attention_softmax::Rows<'a>)>(
                self.ops.len(),
            )?,
        )?;
        #[cfg(test)]
        conversion_probe::outer(context, conversion_live)?;
        context.admit_phase(conversion_live, 0)?;
        // Nested converters add new payload above this complete live stage,
        // including groups and move/result controls. Keep it through transfer;
        // no retirement credit is inferred from an independent phase maximum.
        context.retain(conversion_live)?;
        let conversion = Conversion::new(context, &self.ops)?;
        let mut ops = reserve(self.ops.len())?;
        let mut softmax = reserve(self.ops.len())?;
        let sticky = self.validity.current();
        let validity_epoch = self.validity.epoch();
        for (producer, op) in self.ops.into_iter().enumerate() {
            let kind = match op.kind {
                PublishedKind::AttentionValues(v) => {
                    Kind::Values(v.into_owned(context, &conversion)?)
                }
                PublishedKind::AttentionSoftmax(rows) => {
                    softmax.push((producer, rows));
                    Kind::Softmax
                }
                PublishedKind::AttentionScale(rows) => Kind::Scale(rows),
                PublishedKind::AttentionScores(rows) => Kind::Scores(rows),
                PublishedKind::CausalMask => Kind::CausalMask,
                PublishedKind::CacheSuffix => Kind::CacheSuffix,
                PublishedKind::KvInitialize { layer, state } => Kind::KvInitialize { layer, state },
                PublishedKind::Rotary(rows) => Kind::Rotary(rows),
                PublishedKind::TokenLookup {
                    weight,
                    source,
                    plan,
                } => Kind::Lookup {
                    weight,
                    source: scope.tensor(source)?,
                    plan,
                },
                PublishedKind::Projection {
                    weight,
                    source,
                    norms,
                    geometry,
                    raw_bounds,
                    row_energy,
                    input_energy,
                    bias,
                } => {
                    let source_handle = scope.tensor(source)?;
                    let norms = norms.map(|norm| scope.norm(source, norm)).transpose()?;
                    let bias = bias
                        .map(|b| {
                            Ok::<_, Error>(Bias {
                                owner: b.owner,
                                source: scope.tensor(b.source)?,
                                norms: b
                                    .norms
                                    .map(|norm| scope.norm(b.source, norm))
                                    .transpose()?,
                                floor_bounds: b.floor_bounds,
                                coefficients: b.coefficients,
                            })
                        })
                        .transpose()?;
                    Kind::Projection {
                        weight,
                        source: source_handle,
                        norms,
                        geometry,
                        raw_bounds,
                        row_energy,
                        input_energy,
                        bias,
                    }
                }
                PublishedKind::Silu => Kind::Silu,
                PublishedKind::RmsNorm {
                    gamma,
                    input_energy,
                    output_energy,
                    program,
                    owns_program,
                    checked,
                    row_bounds,
                } => Kind::Rms {
                    gamma: scope.tensor(gamma)?,
                    input_energy,
                    output_energy,
                    program,
                    owns_program,
                    checked: checked
                        .map(|rows| rows.into_owned(self.imports_owner))
                        .transpose()?,
                    row_bounds,
                },
                PublishedKind::LastToken { row_energy } => Kind::LastToken { row_energy },
                PublishedKind::TokenFeedback => Kind::Feedback,
                PublishedKind::Reshape => Kind::Reshape,
                PublishedKind::Argmax(argmax) => Kind::Argmax(argmax),
                PublishedKind::Binary {
                    right,
                    product,
                    row_energy,
                } => Kind::Binary {
                    right,
                    product,
                    row_energy,
                },
            };
            ops.push(Op {
                first: op.first,
                input: op.input,
                output: op.output,
                kind,
                prepared: op.prepared,
            });
        }
        // All downstream RowView aliases have retired; only original row owners
        // and conversion's lookup aliases remain. finish drops lookup first.
        let invocations = conversion.finish(softmax)?;
        let owned = OwnedEpoch {
            association: context.epoch_association()?,
            instance: context.registry().instance(),
            ops,
            invocations,
            imports: self.imports,
            recipe_names: self.recipe_names,
            frontier: self.frontier,
            exports: self.exports,
            output: self.output,
            origin: self.origin,
            next: self.next,
            request_prelude: self.request_prelude,
            binding: self.binding,
            work: self.work,
            sticky,
            validity_epoch,
        };
        fn send_sync<T: Send + Sync>() {}
        send_sync::<OwnedEpoch>();
        send_sync::<request_origin::compact_storage::owning::OwnedEmission<OwnedEpoch>>();
        send_sync::<Program>();
        Ok(owned)
    }
}

/// Test-only outside-live pressure at the actual nested destination boundary.
/// Expected coverage follows named simultaneous containers, independently of
/// `retain` calls, so retaining only either destination fails this regression.
#[cfg(test)]
pub(in crate::replicated_decoder::program) mod conversion_probe {
    use super::*;
    use std::cell::RefCell;
    struct Probe {
        short: bool,
        expected: usize,
        allocations: usize,
        target: usize,
        entered: bool,
    }
    thread_local! { static STATE:RefCell<Option<Probe>>=const{RefCell::new(None)}; }
    pub(in crate::replicated_decoder::program) fn start(short: bool) {
        STATE.with(|s| {
            assert!(s.borrow().is_none());
            *s.borrow_mut() = Some(Probe {
                short,
                expected: 0,
                allocations: 0,
                target: 0,
                entered: false,
            });
        });
    }
    pub(in crate::replicated_decoder::program) fn finish() -> (usize, usize, bool) {
        STATE.with(|s| {
            let p = s.borrow_mut().take().unwrap();
            (p.allocations, p.target, p.entered)
        })
    }
    pub(in crate::replicated_decoder::program) fn outer(
        context: &request_origin::compact_storage::EmissionContext<'_>,
        live: usize,
    ) -> Result<(), Error> {
        STATE.with(|s| {
            if let Some(p) = s.borrow_mut().as_mut() {
                p.expected = add_bytes(context.test_retained_payload(), live)?;
            }
            Ok(())
        })
    }
    pub(in crate::replicated_decoder::program) fn nested(live: usize) -> Result<(), Error> {
        STATE.with(|s| {
            if let Some(p) = s.borrow_mut().as_mut() {
                p.expected = add_bytes(p.expected, live)?;
            }
            Ok(())
        })
    }
    pub(in crate::replicated_decoder::program) fn pressure<'c>(
        context: &'c request_origin::compact_storage::EmissionContext<'_>,
        live: usize,
    ) -> Result<Option<pllm_garble::shared_allocation::ExternalCharge<'c>>, Error> {
        let selected = STATE.with(|s| {
            let mut s = s.borrow_mut();
            let Some(p) = s.as_mut() else {
                return Ok::<_, Error>(None);
            };
            if p.entered {
                return Ok(None);
            }
            p.entered = true;
            let payload = add_bytes(p.expected, live)?.max(context.reservation().payload_bytes());
            let outside = context
                .plan()
                .budget()
                .used_bytes()
                .checked_sub(context.reservation().payload_bytes())
                .ok_or(Error::Bounds)?;
            p.target = add_bytes(outside, payload)?;
            Ok(Some((p.short, p.target)))
        })?;
        let Some((short, total)) = selected else {
            return Ok(None);
        };
        // Same real counter, unchanged global cap. Include this pressure guard's
        // own header; no source-constructor floor or historic peak is modified.
        let bytes = MAX_PROGRAM_BYTES
            .checked_sub(total)
            .and_then(|n| n.checked_add(usize::from(short)))
            .ok_or(Error::Bounds)?;
        let payload = bytes
            .checked_sub(size_of::<pllm_garble::shared_allocation::ExternalCharge<'_>>())
            .ok_or(Error::Bounds)?;
        context
            .plan()
            .budget()
            .charge_external(payload)
            .map(Some)
            .map_err(|_| Error::Bounds)
    }
    pub(in crate::replicated_decoder::program) fn allocation() {
        STATE.with(|s| {
            if let Some(p) = s.borrow_mut().as_mut() {
                p.allocations += 1;
            }
        });
    }
}
