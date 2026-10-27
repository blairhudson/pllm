//! Compiler-created, storage-neutral public numeric recipes and ordered outputs.
//! No constructor certifies an existing View, lane array, digest or caller ID.
//! Creation is conditional on the original declared input domains. It does NOT
//! prove runtime range, complete rows, sticky validity or request admission.
use super::*;
use crate::replicated_decoder::{
    program::compact_lowering::compact_admission, source_registry::SourceRegistry,
};
use pllm_models::ModelOperation;
pub(in crate::replicated_decoder::program) use super::custody::{
    BindingRef, Geometry, OrderedCoordinate, OriginalInputBinding,
};

const MAX_PERMUTATIONS: usize = 8;

#[cfg(test)]
thread_local! {
    static FACTORY_ENTRIES: Cell<usize> = const { Cell::new(0) };
}
fn entering_factory() {
    #[cfg(test)]
    FACTORY_ENTRIES.with(|entries| entries.set(entries.get() + 1));
}

#[derive(Clone, Copy)]
struct Permutation {
    input: Geometry,
    output: Geometry,
    axes: [usize; 4],
}
impl Permutation {
    fn inverse(self, mut index: usize) -> Result<usize, Error> {
        let mut input = [0usize; 4];
        for axis in (0..self.output.rank).rev() {
            input[self.axes[axis]] = index % self.output.dimensions[axis];
            index /= self.output.dimensions[axis];
        }
        input[..self.input.rank]
            .iter()
            .zip(self.input.dimensions())
            .try_fold(0usize, |index, (coordinate, dimension)| {
                index
                    .checked_mul(*dimension)
                    .and_then(|index| index.checked_add(*coordinate))
                    .ok_or(Error::Bounds)
            })
    }
}

// Fixed-rank alias controls deliberately stay inline and are quoted as such;
// boxing this variant would add a separate unquoted factory/allocation.
#[allow(clippy::large_enum_variant)]
enum Parent<'s> {
    Imported(OriginalImportedView<'s>),
    Produced(OriginalProducedView<'s>),
}
impl<'s> Parent<'s> {
    fn binding(&self) -> BindingRef<'_, 's> {
        match self {
            Self::Imported(input) => input.binding(),
            Self::Produced(input) => input.binding(),
        }
    }
    fn root(&self) -> &OriginalImportedView<'s> {
        match self {
            Self::Imported(input) => input,
            Self::Produced(input) => input.record.parent.root(),
        }
    }
}
impl<'b, 's> BindingRef<'b, 's> {
    pub(in crate::replicated_decoder::program) fn coordinate(
        self,
        index: usize,
    ) -> Result<OrderedCoordinate<'b, 's>, Error> {
        match self {
            Self::Imported(input) => {
                if index >= input.logical().dimensions().iter().product::<usize>() {
                    return Err(Error::Shape);
                }
                Ok(OrderedCoordinate {
                    owner: super::custody::CoordinateOwner::Imported(input),
                    lane: index,
                })
            }
            Self::Produced(input) => input.coordinate(index),
        }
    }
    pub(super) fn root(&self) -> &OriginalImportedView<'s> {
        match self {
            Self::Imported(input) => input,
            Self::Produced(input) => input.record.parent.root(),
        }
    }
    pub(in crate::replicated_decoder::program) fn geometry(&self) -> Result<Geometry, Error> {
        match self {
            Self::Imported(input) => Geometry::new(input.logical().dimensions()),
            Self::Produced(input) => Ok(input.geometry),
        }
    }
    pub(in crate::replicated_decoder::program) fn semantic(&self) -> &'s str {
        match self {
            Self::Imported(input) => input.semantic(),
            Self::Produced(input) => input.record.operation.id.as_str(),
        }
    }
    pub(in crate::replicated_decoder::program) fn bounds(&self) -> Interval {
        match self {
            Self::Imported(input) => input.bounds(),
            Self::Produced(input) => input.record.bounds,
        }
    }
    fn retain(&self, builder: &Builder) -> Result<Parent<'s>, Error> {
        match self {
            Self::Imported(input) => Ok(Parent::Imported(input.retain_checked(builder)?)),
            Self::Produced(input) => Ok(Parent::Produced((*input).clone())),
        }
    }
    pub(super) fn depth(&self) -> usize {
        match self {
            Self::Imported(_) => 0,
            Self::Produced(input) => input.record.depth,
        }
    }
    fn native_work(&self) -> WorkBudget {
        match self {
            Self::Imported(_) => WorkBudget::default(),
            Self::Produced(input) => input.record.native_work,
        }
    }
    fn prior_peak(&self) -> usize {
        match self {
            Self::Imported(_) => 0,
            Self::Produced(input) => input.record.peak,
        }
    }
    fn retained_bytes(&self) -> Result<usize, Error> {
        match self {
            Self::Imported(_) => Ok(0), // actual record backing already in original Builder census
            Self::Produced(input) => plus(
                input.record.parent.binding().retained_bytes()?,
                plus(
                    plus(size_of::<Producer<'_>>(), array::<usize>(2)?)?,
                    input.record.recipe.template_bytes()?,
                )?,
            ),
        }
    }
}

enum Recipe {
    Lookup {
        full: Arc<tensor::Program>,
        tail: Option<Arc<tensor::Program>>,
        vocabulary: usize,
    },
    Rms {
        program: Arc<tensor::Program>,
        width: usize,
    },
    Projection {
        map: PreparedTensorMap,
        inner: usize,
        columns: usize,
        raw: Interval,
    },
    Silu(PreparedTensorMap),
}
impl Recipe {
    fn template_bytes(&self) -> Result<usize, Error> {
        match self {
            Self::Lookup { full, tail, .. } => std::iter::once(full)
                .chain(tail)
                .try_fold(0, |bytes, program| {
                    plus(bytes, program.resources()?.immutable_template_bytes)
                }),
            Self::Rms { program, .. } => Ok(program.resources()?.immutable_template_bytes),
            Self::Projection { map, .. } | Self::Silu(map) => map.template_bytes(),
        }
    }
}
struct Producer<'s> {
    // Native/public recipe drops while source and complete origin chain remain.
    recipe: Recipe,
    source: Arc<PublicSource>,
    operation: &'s ModelOperation,
    bounds: Interval,
    native_work: WorkBudget, // cumulative NATIVE work only, not whole-request work
    depth: usize,
    peak: usize,
    geometry: Geometry,
    parent: Parent<'s>,
}

/// Alias transforms never mint a new producer. Fixed depth/rank metadata only;
/// no Words, per-lane coordinates, ExpandedNode list or private values.
#[derive(Clone)]
pub(in crate::replicated_decoder::program) struct OriginalProducedView<'s> {
    record: Arc<Producer<'s>>,
    geometry: Geometry,
    permutations: [Option<Permutation>; MAX_PERMUTATIONS],
    count: usize,
}

/// Borrowed immutable public recipes. Original native Arc handles remain intact.
/// No private runtime extraction, proof, executable route or producer-ID input.
pub(in crate::replicated_decoder::program) enum RecipeRef<'a> {
    Lookup {
        full: &'a Arc<tensor::Program>,
        tail: Option<&'a Arc<tensor::Program>>,
        vocabulary: usize,
    },
    Rms {
        program: &'a Arc<tensor::Program>,
        width: usize,
    },
    Projection {
        map: &'a PreparedTensorMap,
        inner: usize,
        columns: usize,
        raw: Interval,
    },
    Silu(&'a PreparedTensorMap),
}

/// A declaration of STILL REQUIRED evidence, never a discharged proof. Keeping
/// this typed report with every recipe prevents source identity from being
/// mistaken for a range/general-row/sticky certificate by later emitters.
pub(in crate::replicated_decoder::program) struct UnresolvedRequirements {
    pub original_input: usize,
    pub declared_input_domain: Interval,
    pub complete_row_width: usize,
}

struct Preflight {
    baseline: usize,
    outside_work: WorkBudget,
    peak: Cell<usize>,
    limit: usize,
}
impl Preflight {
    fn admit(&self, bytes: usize) -> Result<(), Error> {
        if bytes > self.limit {
            return Err(Error::Bounds);
        }
        self.peak.set(self.peak.get().max(bytes));
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn preflight<'s>(
    builder: &Builder,
    input: &BindingRef<'_, 's>,
    source: &Arc<PublicSource>,
    operation: &'s ModelOperation,
    prior: &compact_admission::RequestWork,
    owners: &compact_admission::RequestOwners<'_>,
    cap: usize,
) -> Result<Preflight, Error> {
    let root = input.root();
    input.check_identity(builder, source, operation)?;
    let (outside, outside_work, descriptors) =
        compact_admission::input_construction_baseline(prior, owners)?;
    if plus(
        descriptors,
        plus(builder.nodes.len(), plus(input.depth(), 1)?)?,
    )? > MAX_NODES
    {
        return Err(Error::Bounds);
    }
    let mut baseline = plus(
        outside,
        compact_lowering::CompactGraph::imports_bytes(builder)?,
    )?;
    for bytes in [
        decoder_plan_metadata_bytes(root.plan(), cap)?,
        source_metadata_bytes(root.registry().source_records(), root.registry().source_norms())?,
        source.resident_metadata_bytes()?,
        array::<usize>(2)?,
        input.retained_bytes()?,
        size_of::<Producer<'_>>(),
        array::<usize>(2)?,      // NEW producer Arc backing
        size_of::<Parent<'_>>(), // parent-retain move
        size_of::<Result<Parent<'_>, Error>>(),
        size_of::<OriginalImportedView<'_>>(),
        size_of::<Result<OriginalImportedView<'_>, Error>>(),
        size_of::<OriginalProducedView<'_>>(),
        size_of::<Result<OriginalProducedView<'_>, Error>>(),
        size_of::<Result<OriginalProducedView<'_>, Error>>(),
        size_of::<Recipe>(),
        size_of::<Preflight>(),
        size_of::<Result<Preflight, Error>>(),
        size_of::<Geometry>(),
        size_of::<RequiredChecks>(),
        size_of::<SourceRegistry<'_>>(),
        size_of::<BindingRef<'_, '_>>(),
        size_of::<Result<Recipe, Error>>(),
        size_of::<Arc<tensor::Program>>(),
        size_of::<Arc<tensor::Program>>(),
        size_of::<Option<Arc<tensor::Program>>>(),
    ] {
        baseline = plus(baseline, bytes)?;
    }
    let quote = Preflight {
        baseline,
        outside_work,
        peak: Cell::new(owners.prior_request_peak_bytes.max(input.prior_peak())),
        limit: cap.min(MAX_PROGRAM_BYTES),
    };
    quote.admit(baseline.max(quote.peak.get()))?;
    Ok(quote)
}

fn native_quote(construction: tensor::Construction) -> Result<tensor::ConstructionQuote, Error> {
    construction.quote().map_err(|error| match error {
        tensor::ConstructionQuoteError::Geometry(error) => error,
        tensor::ConstructionQuoteError::UncheckedRotaryCoefficientCapacity => Error::Bounds,
    })
}

fn source_record<'a>(
    input: &BindingRef<'_, 'a>,
    op: &ModelOperation,
) -> Result<&'a SourceTensor, Error> {
    let key = op
        .attributes
        .get("weight")
        .and_then(|v| v.as_str())
        .ok_or(Error::Binding)?;
    input
        .root()
        .registry()
        .source_records()
        .get(key)
        .ok_or(Error::Binding)
}

impl<'s> OriginalProducedView<'s> {
    #[allow(clippy::too_many_arguments)]
    fn finish(
        builder: &Builder,
        input: BindingRef<'_, 's>,
        operation: &'s ModelOperation,
        source: &Arc<PublicSource>,
        recipe: Recipe,
        bounds: Interval,
        geometry: Geometry,
        own_work: WorkBudget,
        quote: Preflight,
    ) -> Result<Self, Error> {
        let native_work = input.native_work().checked_add(own_work)?;
        quote.outside_work.checked_add(native_work)?.limits()?;
        quote.admit(plus(quote.baseline, recipe.template_bytes()?)?)?;
        Ok(Self {
            record: Arc::new(Producer {
                parent: input.retain(builder)?,
                operation,
                source: source.clone(),
                recipe,
                bounds,
                native_work,
                depth: input.depth() + 1,
                peak: quote.peak.get(),
                geometry,
            }),
            geometry,
            permutations: [None; MAX_PERMUTATIONS],
            count: 0,
        })
    }

    /// Original PerToken full-ring selectors followed by raw public-table dot.
    /// Tiled/Trie policies require their own genuine source/token contracts.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::replicated_decoder::program) fn lookup(
        builder: &Builder,
        binding: &impl OriginalInputBinding<'s>,
        source: &Arc<PublicSource>,
        operation: &'s ModelOperation,
        prior: &compact_admission::RequestWork,
        owners: &compact_admission::RequestOwners<'_>,
        cap: usize,
    ) -> Result<Self, Error> {
        let input = binding.binding();
        let quote = preflight(builder, &input, source, operation, prior, owners, cap)?;
        if operation.operator != Op::TokenLookup
            || input.semantic() != "input.tokens"
            || builder.lookup != policy::LookupPolicy::PerToken
        {
            return Err(Error::Binding);
        }
        let table = source_record(&input, operation)?;
        let [vocabulary, width] = table.shape.as_slice() else {
            return Err(Error::Shape);
        };
        let shape = Geometry::operation(operation)?;
        let tokens = input.geometry()?.lanes();
        if shape.width() != *width
            || tokens.checked_mul(*width) != Some(shape.lanes())
            || input.bounds()
                != Interval::new(
                    0,
                    i64::try_from(*vocabulary).map_err(|_| Error::Bounds)? - 1,
                )?
            || table.public_q18.is_some()
        {
            return Err(Error::Shape);
        }
        source.file(
            operation.attributes["weight"]
                .as_str()
                .ok_or(Error::Binding)?,
        )?;
        let bounds = Interval::new(
            input
                .bounds()
                .lower()
                .checked_sub(*vocabulary as i64 - 1)
                .ok_or(Error::Bounds)?,
            input.bounds().upper(),
        )?;
        let width = (*vocabulary).min(tensor::MAX_TILE_LANES);
        let remainder = *vocabulary % width;
        let full_quote = native_quote(tensor::Construction::EqualZero { lanes: width })?;
        let tail_quote = if remainder == 0 {
            None
        } else {
            Some(native_quote(tensor::Construction::EqualZero {
                lanes: remainder,
            })?)
        };
        let mut peak = full_quote.new_peak_bytes;
        if let Some(tail) = tail_quote {
            peak = peak.max(plus(
                full_quote.immutable_template_bytes,
                tail.new_peak_bytes,
            )?);
        }
        peak = peak.max(plus(
            full_quote.immutable_template_bytes,
            tail_quote.map_or(0, |quote| quote.immutable_template_bytes),
        )?);
        quote.admit(plus(
            quote.baseline,
            plus(
                peak,
                plus(
                    size_of::<tensor::ConstructionQuote>(),
                    size_of::<Option<tensor::ConstructionQuote>>(),
                )?,
            )?,
        )?)?;
        entering_factory();
        let full = Arc::new(tensor::Program::equal_zero(width, bounds)?);
        let tail = if let Some(tail_quote) = tail_quote {
            quote.admit(plus(
                quote.baseline,
                plus(
                    full.resources()?.immutable_template_bytes,
                    tail_quote.new_peak_bytes,
                )?,
            )?)?;
            entering_factory();
            Some(Arc::new(tensor::Program::equal_zero(remainder, bounds)?))
        } else {
            None
        };
        for (program, native) in
            std::iter::once((&full, full_quote)).chain(tail.as_ref().zip(tail_quote))
        {
            if program.resources()?.immutable_template_bytes > native.immutable_template_bytes
                || program.resources()?.shared_admitted_bytes > puma::MAX_ADMITTED_BYTES
            {
                return Err(Error::Bounds);
            }
        }
        let full_count = *vocabulary / width;
        let row_work = full
            .work_budget()
            .checked_mul(full_count as u64)?
            .checked_add(
                tail.as_ref()
                    .map_or(WorkBudget::default(), |p| p.work_budget()),
            )?;
        let work = row_work.checked_mul(tokens as u64)?;
        Self::finish(
            builder,
            input,
            operation,
            source,
            Recipe::Lookup {
                full,
                tail,
                vocabulary: *vocabulary,
            },
            Interval::new(table.lower, table.upper)?,
            shape,
            work,
            quote,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::replicated_decoder::program) fn rms_norm(
        builder: &Builder,
        binding: &impl OriginalInputBinding<'s>,
        source: &Arc<PublicSource>,
        operation: &'s ModelOperation,
        prior: &compact_admission::RequestWork,
        owners: &compact_admission::RequestOwners<'_>,
        cap: usize,
    ) -> Result<Self, Error> {
        let input = binding.binding();
        let quote = preflight(builder, &input, source, operation, prior, owners, cap)?;
        if operation.operator != Op::RmsNorm
            || operation
                .attributes
                .get("weight_offset")
                .and_then(|v| v.as_i64())
                != Some(0)
            || builder.checked_bound.is_some()
        {
            return Err(Error::Binding);
        }
        // Checked-input/general-row producer is NOT supplied by an interval.
        let shape = Geometry::operation(operation)?;
        if shape != input.geometry()? {
            return Err(Error::Shape);
        }
        let gamma = source_record(&input, operation)?;
        if gamma.shape != [shape.width()] {
            return Err(Error::Shape);
        }
        let values = gamma.public_q18.as_deref().ok_or(Error::Binding)?;
        let native = native_quote(tensor::Construction::RmsNormCertified {
            rows: 1,
            width: shape.width(),
            gamma_len: values.len(),
        })?;
        quote.admit(plus(
            quote.baseline,
            plus(
                native.new_peak_bytes,
                plus(
                    size_of::<tensor::ConstructionQuote>(),
                    size_of::<Result<(tensor::Program, Option<u128>), Error>>(),
                )?,
            )?,
        )?)?;
        entering_factory();
        let (program, _factory_output_row_bound) =
            rms_norm_program(true, shape.width(), input.bounds(), values, None)?;
        // Keep actual certified numeric factory, but do not issue/copy a general
        // row/sticky certificate into any compiler ledger from this identity.
        if program.resources()?.immutable_template_bytes > native.immutable_template_bytes
            || program.resources()?.shared_admitted_bytes > puma::MAX_ADMITTED_BYTES
        {
            return Err(Error::Bounds);
        }
        let bounds = program.output_bounds();
        let work = program
            .work_budget()
            .checked_mul((shape.lanes() / shape.width()) as u64)?;
        Self::finish(
            builder,
            input,
            operation,
            source,
            Recipe::Rms {
                program: Arc::new(program),
                width: shape.width(),
            },
            bounds,
            shape,
            work,
            quote,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::replicated_decoder::program) fn projection(
        builder: &Builder,
        binding: &impl OriginalInputBinding<'s>,
        source: &Arc<PublicSource>,
        operation: &'s ModelOperation,
        prior: &compact_admission::RequestWork,
        owners: &compact_admission::RequestOwners<'_>,
        cap: usize,
    ) -> Result<Self, Error> {
        let checked = super::custody::ProjectionBinding::check(builder, binding, source, operation)?;
        let input = checked.input;
        let shape = checked.shape;
        let weight = checked.weight;
        let inner = checked.inner;
        let columns = checked.columns;
        let prior_shape = input.geometry()?;
        let quote = preflight(builder, &input, source, operation, prior, owners, cap)?;
        let (raw, _unissued_row_energy) = public_matrix_contract(
            [prior_shape.lanes() / inner, inner, columns],
            false,
            true,
            || {
                Ok(InputBounds {
                    left: input.bounds(),
                    right: Interval::new(weight.lower, weight.upper)?,
                })
            },
            PublicMatrixStatistics {
                row_l1: weight.row_l1,
                row_squared_norm: weight.row_squared_norm,
                row_l1_squared_sum: weight.row_l1_squared_sum,
            },
            || None,
        )?;
        let map = PreparedTensorMap::rescale(shape.lanes(), raw, 18, quote.baseline, |bytes| {
            quote.admit(bytes)
        })?;
        let bounds = map.output_bounds()?;
        let work = map.work_budget()?;
        Self::finish(
            builder,
            input,
            operation,
            source,
            Recipe::Projection {
                map,
                inner,
                columns,
                raw,
            },
            bounds,
            shape,
            work,
            quote,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::replicated_decoder::program) fn activation(
        builder: &Builder,
        binding: &impl OriginalInputBinding<'s>,
        source: &Arc<PublicSource>,
        operation: &'s ModelOperation,
        prior: &compact_admission::RequestWork,
        owners: &compact_admission::RequestOwners<'_>,
        cap: usize,
    ) -> Result<Self, Error> {
        let input = binding.binding();
        let quote = preflight(builder, &input, source, operation, prior, owners, cap)?;
        if operation.operator != Op::Silu {
            return Err(Error::Binding);
        }
        let shape = Geometry::operation(operation)?;
        if shape != input.geometry()? {
            return Err(Error::Shape);
        }
        let map = PreparedTensorMap::guarded_silu(
            shape.lanes(),
            input.bounds(),
            quote.baseline,
            |bytes| quote.admit(bytes),
        )?;
        let bounds = map.output_bounds()?;
        let work = map.work_budget()?;
        Self::finish(
            builder,
            input,
            operation,
            source,
            Recipe::Silu(map),
            bounds,
            shape,
            work,
            quote,
        )
    }

    pub(in crate::replicated_decoder::program) fn geometry(&self) -> Geometry {
        self.geometry
    }
    pub(super) fn same_producer(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.record, &other.record)
    }
    pub(super) fn canonical_geometry(&self) -> bool {
        self.geometry == self.record.geometry
    }
    pub(in crate::replicated_decoder::program) fn operation(&self) -> &'s ModelOperation {
        self.record.operation
    }
    pub(in crate::replicated_decoder::program) fn source(&self) -> &Arc<PublicSource> {
        &self.record.source
    }
    pub(in crate::replicated_decoder::program) fn input_binding(&self) -> BindingRef<'_, 's> {
        self.record.parent.binding()
    }
    pub(in crate::replicated_decoder::program) fn preparation_peak(&self) -> usize {
        self.record.peak
    }
    pub(in crate::replicated_decoder::program) fn unresolved_requirements(
        &self,
    ) -> UnresolvedRequirements {
        let root = self.record.parent.root();
        UnresolvedRequirements {
            original_input: root.owner(),
            declared_input_domain: root.bounds(),
            complete_row_width: self.geometry.width(),
        }
    }
    pub(in crate::replicated_decoder::program) fn recipe(&self) -> RecipeRef<'_> {
        match &self.record.recipe {
            Recipe::Lookup {
                full,
                tail,
                vocabulary,
            } => RecipeRef::Lookup {
                full,
                tail: tail.as_ref(),
                vocabulary: *vocabulary,
            },
            Recipe::Rms { program, width } => RecipeRef::Rms {
                program,
                width: *width,
            },
            Recipe::Projection {
                map,
                inner,
                columns,
                raw,
            } => RecipeRef::Projection {
                map,
                inner: *inner,
                columns: *columns,
                raw: *raw,
            },
            Recipe::Silu(map) => RecipeRef::Silu(map),
        }
    }
    pub(in crate::replicated_decoder::program) fn coordinate(
        &self,
        mut index: usize,
    ) -> Result<OrderedCoordinate<'_, 's>, Error> {
        if index >= self.geometry.lanes() {
            return Err(Error::Shape);
        }
        for permutation in self.permutations[..self.count].iter().rev().flatten() {
            index = permutation.inverse(index)?;
        }
        Ok(OrderedCoordinate {
            owner: super::custody::CoordinateOwner::Produced(self),
            lane: index,
        })
    }
    fn alias_quote(&self, cap: usize) -> Result<(), Error> {
        let bytes = plus(
            self.record.peak,
            plus(
                size_of::<Self>(),
                plus(size_of::<Self>(), size_of::<Result<Self, Error>>())?,
            )?,
        )?;
        if bytes > cap.min(MAX_PROGRAM_BYTES) {
            return Err(Error::Bounds);
        }
        Ok(())
    }
    pub(in crate::replicated_decoder::program) fn reshape(
        &self,
        dimensions: &[usize],
        cap: usize,
    ) -> Result<Self, Error> {
        let geometry = Geometry::new(dimensions)?;
        if geometry.lanes() != self.geometry.lanes() {
            return Err(Error::Shape);
        }
        self.alias_quote(cap)?;
        let mut result = self.clone();
        result.geometry = geometry;
        Ok(result)
    }
    pub(in crate::replicated_decoder::program) fn permute(
        &self,
        axes: &[usize],
        cap: usize,
    ) -> Result<Self, Error> {
        if axes.len() != self.geometry.rank || self.count == MAX_PERMUTATIONS {
            return Err(Error::Shape);
        }
        let mut dimensions = [1; 4];
        let mut mapping = [0; 4];
        let mut seen = [false; 4];
        for (out, axis) in axes.iter().copied().enumerate() {
            if axis >= axes.len() || seen[axis] {
                return Err(Error::Shape);
            }
            seen[axis] = true;
            mapping[out] = axis;
            dimensions[out] = self.geometry.dimensions[axis];
        }
        self.alias_quote(cap)?;
        let geometry = Geometry::new(&dimensions[..axes.len()])?;
        let mut result = self.clone();
        result.geometry = geometry;
        result.permutations[self.count] = Some(Permutation {
            input: self.geometry,
            output: geometry,
            axes: mapping,
        });
        result.count += 1;
        Ok(result)
    }
}

#[cfg(test)]
#[path = "original_produced_view_tests.rs"]
mod tests;
