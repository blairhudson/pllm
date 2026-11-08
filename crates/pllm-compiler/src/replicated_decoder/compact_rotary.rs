//! Canonical E6 owning-row child, with bounded R child repairs. This postimage
//! is for E's compact_rotary.rs mount; no shared E glue is changed here.
use super::super::graph::attention_driver::{self as driver, RotarySink};
use super::*;
use super::ControllerAccounting;
use std::mem::{size_of, size_of_val};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EnergyFamily {
    General,
    Attention,
}

/// Compiler-private ancestry of the checked bound, never an issuer accepting a
/// bare number. Output is attention-only irrespective of the input family.
#[derive(Clone, Copy)]
struct RotaryEnergyPremise {
    family: EnergyFamily,
    producer_slot: usize,
    start: usize,
    first_word: (usize, usize),
    width: usize,
    input_squared: u128,
    attention_squared: u128,
}

pub(super) struct NativeRow {
    pub(super) position: u32,
    pub(super) program: Arc<tensor::Program>,
    pub(super) incoming: Interval,
    pub(super) bounds: Interval,
    /// Attention-only; never a GENERAL complete-row certificate.
    pub(super) attention_energy: Option<u128>,
    premise: Option<RotaryEnergyPremise>,
}
pub(super) struct Rows {
    pub(super) native: NativeRows,
    pub(super) width: usize,
    pub(super) work: WorkBudget,
    pub(super) bounds: Interval,
}
struct Spec<'a> {
    position: u32,
    bounds: Interval,
    cosine: tensor::ExactStorage<i64>,
    sine: tensor::ExactStorage<i64>,
    cosine_box: Box<[i64]>,
    sine_box: Box<[i64]>,
    source: Option<tensor::ResidentFactorySource<'a>>,
    schedule: tensor::ResidentNativeSchedule<'a>,
    full: Option<tensor::FactoryOutput>,
    tail: Option<tensor::FactoryOutput>,
    program: Option<Arc<tensor::Program>>,
    coefficients_installed: bool,
    premise: Option<RotaryEnergyPremise>,
}
struct NativeStorage { native: tensor::ExactStorage<NativeRow> }
#[derive(Clone)]
pub(super) struct NativeRows(Arc<NativeStorage>);
impl NativeRows {
    fn empty() -> Self { Self(Arc::new(NativeStorage { native: tensor::ExactStorage::empty() })) }
    pub(super) fn capacity(&self) -> usize { self.0.native.extent() }
    pub(super) fn container_bytes(&self) -> Result<usize, Error> {
        add_bytes(array_bytes::<NativeRow>(self.capacity())?,
            add_bytes(size_of::<NativeStorage>(), 2 * size_of::<usize>())?)
    }
    pub(super) fn shares(&self, other: &Self) -> bool { Arc::ptr_eq(&self.0, &other.0) }
}
impl std::ops::Deref for NativeRows {
    type Target = [NativeRow];
    fn deref(&self) -> &[NativeRow] {
        // Only a complete owner is ever returned/published. Incomplete storage
        // remains private graph pending and is never consumed to fabricate a slice.
        self.0.native.ready().expect("published rotary rows must be complete")
    }
}
impl<'s> IntoIterator for &'s NativeRows {
    type Item = &'s NativeRow;
    type IntoIter = std::slice::Iter<'s, NativeRow>;
    fn into_iter(self) -> Self::IntoIter { self.iter() }
}
pub(super) struct Pending<'a> {
    specs: tensor::ExactStorage<Option<Spec<'a>>>,
    native: Option<NativeRows>,
    width: usize,
}
impl<'a> Pending<'a> {
    pub(super) fn extent(&self) -> usize { self.specs.extent() }
    fn empty(width: usize) -> Self { Self { specs: tensor::ExactStorage::empty(), native: None, width } }
    pub(super) fn native(&self) -> Option<&NativeRows> { self.native.as_ref() }
    pub(super) fn backing_bytes(&self) -> Result<usize, Error> {
        let mut bytes = array_bytes::<Option<Spec<'a>>>(self.specs.extent())?;
        if let Some(native) = &self.native { bytes = add_bytes(bytes, native.container_bytes()?)?; }
        let specs = match &self.specs { tensor::ExactStorage::Ready(v) => &v[..], _ => &[] };
        for spec in specs.iter().filter_map(Option::as_ref) {
            let count = if spec.coefficients_installed { self.width } else {
                add_bytes(add_bytes(spec.cosine.extent(), spec.sine.extent())?,
                    add_bytes(spec.cosine_box.len(), spec.sine_box.len())?)?
            };
            bytes = add_bytes(bytes, array_bytes::<i64>(count)?)?;
        }
        Ok(bytes)
    }
    pub(super) fn programs(&self) -> impl Iterator<Item = &tensor::Program> {
        let specs = match &self.specs { tensor::ExactStorage::Ready(v) => &v[..], _ => &[] };
        specs.iter().filter_map(Option::as_ref).flat_map(|s| [s.program.as_deref(),
            s.full.as_ref().map(tensor::FactoryOutput::program),
            s.tail.as_ref().map(tensor::FactoryOutput::program)].into_iter().flatten())
    }
}
struct MappedRow {
    input: PublishedView,
    start: usize,
}
struct Sink<'s, 'a> {
    pending: &'s mut Pending<'a>,
    width: usize,
    sequence: usize,
    theta: f32,
    work: WorkBudget,
    bounds: Option<Interval>,
    expected: PublishedView,
    quote: tensor::ConstructionQuote,
}

/// Lazy selection only; the private row constructor below binds the actual
/// producer/ordered coordinates. Errors are never converted into absence.
fn choose_energy(
    public_norms: bool,
    general: impl FnOnce() -> Result<Option<u128>, Error>,
    attention: impl FnOnce() -> Result<Option<u128>, Error>,
) -> Result<Option<(EnergyFamily, u128)>, Error> {
    if !public_norms {
        return Ok(None);
    }
    if let Some(bound) = general()? {
        return Ok(Some((EnergyFamily::General, bound)));
    }
    Ok(attention()?.map(|bound| (EnergyFamily::Attention, bound)))
}

fn premise<G: ControllerAccounting>(
    graph: &CompactGraph<'_, '_, G>,
    input: &PublishedView,
    row: usize,
    width: usize,
    cosine: &[i64],
    sine: &[i64],
) -> Result<Option<RotaryEnergyPremise>, Error> {
    let start = row.checked_mul(width).ok_or(Error::Bounds)?;
    let Some((family, input_squared)) = choose_energy(
        !graph.norms.is_empty(),
        || compact_validity::row_energy(graph, input, row),
        || graph.attention_energy(input, start, width),
    )?
    else {
        return Ok(None);
    };
    // A logical row may cross physical product tiles (e.g. 108..144).
    // Authenticate against its actual ordered publication, not one value slot.
    let (producer, original_start) = graph
        .alias_range(input, start, width)?
        .ok_or(Error::Binding)?;
    let producer_slot = producer.slot.ok_or(Error::Binding)?;
    let first_word = input.word(start)?;
    Ok(Some(RotaryEnergyPremise {
        family,
        producer_slot,
        start: original_start,
        first_word,
        width,
        input_squared,
        attention_squared: norm_bounds::rotary_energy(input_squared, cosine, sine, 18)?,
    }))
}

impl RotarySink for Sink<'_, '_> {
    type View = PublishedView;
    type Row = MappedRow;
    type Owner = ();
    type Output = usize;
    type Produced = Rows;
    fn begin(
        &mut self,
        _: &pllm_models::ModelOperation,
        shape: &[usize],
        input: &PublishedView,
        position: usize,
        sequence: usize,
        theta: f32,
    ) -> Result<(), Error> {
        if position != 0
            || sequence != self.sequence
            || theta.to_bits() != self.theta.to_bits()
            || shape != &input.logical.dimensions[..4]
            || !Arc::ptr_eq(&input.origin, &self.expected.origin)
            || input.slot != self.expected.slot
            || input.mapping != self.expected.mapping
            || input.order != self.expected.order
            || input.start != self.expected.start
        {
            return Err(Error::Binding);
        }
        Ok(())
    }
    fn rows<'a>(input: &'a PublishedView, width: usize) -> impl Iterator<Item = MappedRow> + 'a
    where
        Self::Row: 'a,
    {
        (0..input.lanes / width).map(move |row| MappedRow {
            input: input.clone(),
            start: row * width,
        })
    }
    fn emit(
        &mut self,
        row: &MappedRow,
        theta: f32,
        position: u32,
        width: usize,
    ) -> Result<(), Error> {
        let index = self.pending.native.as_ref().ok_or(Error::Binding)?.0.native.len();
        let spec = self.pending.specs.ready_mut()?.get_mut(index).and_then(Option::as_mut).ok_or(Error::Binding)?;
        if width != self.width
            || theta.to_bits() != self.theta.to_bits()
            || position != spec.position
            || row.start != index * width
            || !Arc::ptr_eq(&row.input.origin, &self.expected.origin)
            || row.input.slot != self.expected.slot
            || row.input.mapping != self.expected.mapping
            || row.input.order != self.expected.order
            || row.input.start != self.expected.start
        {
            return Err(Error::Binding);
        }
        #[cfg(test)]
        FACTORIES.with(|n| n.set(n.get() + 1));
        spec.schedule.build_into(&mut spec.full, &mut spec.tail).map_err(|_| Error::Bounds)?;
        let output = spec.full.as_ref().ok_or(Error::Binding)?;
        if output.certificate() != tensor::ProducedCertificate::None || spec.tail.is_some() {
            return Err(Error::Binding);
        }
        let resources = output.program().resources()?;
        if resources.immutable_template_bytes > self.quote.immutable_template_bytes
            || resources.shared_admitted_bytes > puma::MAX_ADMITTED_BYTES
        {
            return Err(Error::Bounds);
        }
        let (program, _) = spec.full.take().ok_or(Error::Binding)?.into_parts();
        spec.program = Some(Arc::new(program));
        let program = spec.program.as_ref().ok_or(Error::Binding)?;
        #[cfg(test)]
        OWNERS.with(|owners| owners.borrow_mut().push(Arc::downgrade(program)));
        let bounds = program.output_bounds();
        self.work = self.work.checked_add(program.work_budget())?;
        self.bounds = Some(match self.bounds {
            Some(b) => Interval::new(b.lower().min(bounds.lower()), b.upper().max(bounds.upper()))?,
            None => bounds,
        });
        Arc::get_mut(&mut self.pending.native.as_mut().ok_or(Error::Binding)?.0)
            .ok_or(Error::Binding)?.native.push(NativeRow {
            position,
            program: program.clone(),
            incoming: spec.bounds,
            bounds,
            attention_energy: spec.premise.map(|p| p.attention_squared),
            premise: spec.premise,
        })?;
        #[cfg(test)]
        if FAIL_AFTER_ROW.with(|flag| {
            if flag.get() == Some(index) {
                flag.set(None);
                true
            } else {
                false
            }
        }) {
            return Err(Error::Allocation);
        }
        Ok(())
    }
    fn append(output: &mut usize, _: (), width: usize) {
        *output += width
    }
    fn finish(&mut self, output: usize, shape: &[usize]) -> Result<Rows, Error> {
        if self.pending.native.as_ref().ok_or(Error::Binding)?.0.native.len() != self.pending.specs.len()
            || output != self.expected.lanes
            || shape != &self.expected.logical.dimensions[..4]
        {
            return Err(Error::Binding);
        }
        Arc::get_mut(&mut self.pending.native.as_mut().ok_or(Error::Binding)?.0)
            .ok_or(Error::Binding)?.native.seal()?;
        Ok(Rows {
            native: self.pending.native.as_ref().ok_or(Error::Binding)?.clone(),
            width: self.width,
            work: self.work,
            bounds: self.bounds.ok_or(Error::Shape)?,
        })
    }
}

/// Existing canonical two phases, augmented with actual local/control/return
/// owners. These are logical source-level bounds, NOT complete stack/RSS proof.
/// Compiler owners only; no native table/payload double count.
pub(super) fn compiler_events(rows: usize, width: usize) -> Result<tensor::LogicalEvents, Error> {
    let r = u64::try_from(rows).map_err(|_| Error::Bounds)?;
    let w = u64::try_from(width).map_err(|_| Error::Bounds)?;
    let schedules = crate::replicated_decoder::program::request_origin::compact_storage::resident_schedule_events()
        .checked_mul(r)?;
    let row = tensor::LogicalEvents {
        // Spec None/prefix(2), thirteen fields/Some overwrite(14), NativeRow
        // six fields/destination/prefix(8), source transfers/flag(5), output
        // transfer/Arc-control/program slot/row setup/clone(6+4+1+1+1).
        // Both coefficient halves total W: zero+prefix+generated overwrite=3W.
        writes: w.checked_mul(3).and_then(|n| n.checked_add(2 + 14 + 8 + 5 + 6 + 4 + 1 + 1 + 1)).ok_or(Error::Bounds)?,
        // Generation/finite checks, two binding sweeps <=4W; six independent
        // local descriptor/shape/immutable-header 64-event blocks. Not J.
        validation: w.checked_mul(4).and_then(|n| n.checked_add(6 * 64)).ok_or(Error::Bounds)?,
        transitions: w.checked_add(6 + 5).ok_or(Error::Bounds)?,
        allocation_attempts: 2 + 1, // coefficient attempts + Program Arc new
        // Each coefficient: max(Building owner+release, Ready Box+release)=2,
        // not their sum. Spec/NativeRow(2), Program Arc refs/control(3).
        // Schedule/body dispatch lives in schedules; native Program separately.
        cleanup_dispatches: 4 + 2 + 3,
    }.checked_mul(r)?;
    let fixed = tensor::LogicalEvents {
        writes: 2 + 2 + 2, validation: 2 * 64, transitions: 1 + 2,
        allocation_attempts: 1 + 1 + 1, // specs, rows backing, row-container Arc
        // Two exact owner/release pairs(4), row Arc refs/control(3), root(1).
        cleanup_dispatches: 4 + 3 + 1,
    };
    schedules.checked_add(row)?.checked_add(fixed)
}

fn control_bytes<G: ControllerAccounting>(rows: usize, lanes: usize, attempt: usize) -> Result<(usize, usize), Error> {
    let r = tensor::WorkPreflightWorkspace::resources();
    let mut bytes = 0;
    macro_rules! slot {
        ($n:expr) => {
            bytes = add_bytes(bytes, $n)?;
        };
    }
    macro_rules! pair {
        ($n:expr) => {
            slot!(($n).checked_mul(2).ok_or(Error::Bounds)?);
        };
    }
    slot!(pending_walk_control_bytes()?);
    slot!(array_bytes::<Option<Spec<'_>>>(rows)?);
    slot!(array_bytes::<NativeRow>(rows)?);
    slot!(size_of::<NativeStorage>() + 2 * size_of::<usize>());
    slot!(array_bytes::<i64>(lanes)?); // both exact half-width coefficient buffers
    pair!(size_of::<std::cell::RefMut<'_, Option<Pending<'_>>>>());
    pair!(size_of::<Spec<'_>>());
    pair!(size_of::<NativeRow>());
    pair!(size_of::<MappedRow>());
    pair!(size_of::<PublishedView>());
    slot!(size_of::<Sink<'_, '_>>());
    pair!(size_of::<Rows>());
    pair!(size_of::<Result<Rows, Error>>());
    pair!(size_of::<tensor::ConstructionQuote>());
    slot!(size_of::<NativeConstructionPeak>());
    pair!(size_of::<tensor::WorkPreflightResources>());
    pair!(size_of::<Option<RotaryEnergyPremise>>());
    pair!(size_of::<Result<Option<RotaryEnergyPremise>, Error>>());
    pair!(size_of::<Result<Option<(&PublishedView, usize)>, Error>>());
    slot!(size_of::<(&CompactGraph<'_, '_, G>, &PublishedView, &usize)>()); // GENERAL closure
    slot!(size_of::<(
        &CompactGraph<'_, '_, G>,
        &PublishedView,
        &usize,
        &usize
    )>()); // attention closure
    pair!(size_of::<Result<Option<(EnergyFamily, u128)>, Error>>());
    pair!(size_of::<Option<tensor::ResidentFactorySource<'_>>>());
    slot!(size_of::<std::slice::Iter<'_, Option<Spec<'_>>>>());
    slot!(size_of::<(tensor::Program, tensor::ProducedCertificate)>());
    slot!(size_of::<Result<tensor::FactoryOutput, Error>>());
    pair!(size_of::<Arc<tensor::Program>>());
    pair!(size_of::<Result<tensor::ResidentFactorySource<'_>, tensor::WorkPreflightError>>());
    pair!(r.build_result_bytes);
    pair!(r.summary_bytes);
    pair!(r.ledger_bytes);
    // Outer lower_next, lower_rotary and nested work-admission attempts.
    slot!(attempt.checked_mul(3).ok_or(Error::Bounds)?);
    Ok((add_bytes(bytes, r.workspace_bytes)?, bytes))
}

impl<'a, G: ControllerAccounting> CompactGraph<'_, 'a, G> {
    fn rotary_admission_peak(
        &self,
        rows: usize,
        width: usize,
        attempt: usize,
    ) -> Result<usize, Error> {
        let construction = tensor::Construction::RotaryHalf {
            rows: 1,
            width,
            cosine_len: width / 2,
            sine_len: width / 2,
        };
        let (quote, peak) = construction.quote_for_installed_resident().map_err(|error| match error {
            tensor::ConstructionQuoteError::Geometry(error) => error,
            tensor::ConstructionQuoteError::UncheckedRotaryCoefficientCapacity => Error::Bounds,
        })?;
        let retained = quote.immutable_template_bytes.checked_mul(rows.saturating_sub(1)).ok_or(Error::Bounds)?;
        let (query, controls) =
            control_bytes::<G>(rows, rows.checked_mul(width).ok_or(Error::Bounds)?, attempt)?;
        let build = add_bytes(controls, add_bytes(retained, peak)?)?;
        let graph = add_bytes(self.outside_owner_bytes, self.baseline()?)?;
        add_bytes(
            graph,
            add_bytes(query.max(build), self.contraction_control_bytes.get())?,
        )
    }

    pub(super) fn lower_rotary(
        &self,
        op: &pllm_models::ModelOperation,
        input: &PublishedView,
    ) -> Result<Rows, Error> {
        let context = self.request_admission.ok_or(Error::Binding)?;
        let mut attempt = context.attempt();
        context.check_open()?;
        self.validate_view(input)?;
        context.check_final_work(self.work)?;
        let shape = &input.logical.dimensions[..input.rank as usize];
        let (width, theta) = driver::rotary_geometry(op, shape)?;
        let sequence = usize::try_from(self.frontier.semantic_graph().query_sequence)
            .map_err(|_| Error::Bounds)?;
        if input.rank != 4
            || shape[0] != 1
            || shape[2] != sequence
            || width == 0
            || width % 2 != 0
            || sequence == 0
            || sequence > 513
            || input.lanes % width != 0
        {
            return Err(Error::Shape);
        }
        let rows = input.lanes / width;
        if rows == 0 {
            return Err(Error::Shape);
        }
        let construction = tensor::Construction::RotaryHalf {
            rows: 1,
            width,
            cosine_len: width / 2,
            sine_len: width / 2,
        };
        let peak = self.rotary_admission_peak(rows, width, size_of_val(&attempt))?;
        self.admit_total(peak)?;
        context.admit_phase(peak, 0)?;
        let depth = add_bytes(self.ops.len(), 1)?;
        let mapping_visits = input
            .lanes
            .checked_mul(depth)
            .and_then(|n| n.checked_mul(depth))
            .and_then(|n| n.checked_mul(16))
            .ok_or(Error::Bounds)?;
        // Additional alias_range pass: <= depth alias links per row, two
        // ordered coordinate resolutions per lane. Pay explicitly, rather than
        // assuming spare allowance in the existing proof/mapping census.
        let alias_visits = rows
            .checked_mul(depth)
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| {
                input
                    .lanes
                    .checked_mul(32)
                    .and_then(|lanes| n.checked_add(lanes))
            })
            .ok_or(Error::Bounds)?;
        let visits = rows
            .checked_mul(tensor::MAX_NODES * 36)
            .and_then(|n| n.checked_add(mapping_visits))
            .and_then(|n| n.checked_add(alias_visits))
            .ok_or(Error::Bounds)?;
        context.prepay_metadata(u64::try_from(visits).map_err(|_| Error::Bounds)?)?;
        // Pending identity scans are paid per baseline invocation by H/B.
        let row_count = u64::try_from(rows).map_err(|_| Error::Bounds)?;
        let events = construction.logical_events()?.checked_add(tensor::LogicalEvents {
            // Exactly one full resources postcheck/row, independent of mapping
            // and original protocol/AES work. Retention uses only header bytes.
            validation: tensor::Program::full_resource_validation_visits()?,
            ..tensor::LogicalEvents::default()
        })?.checked_mul(row_count)?.checked_add(compiler_events(rows, width)?)?.total()?;
        context.prepay_primitive(events)?;
        let mut owner = self.rotary_pending.borrow_mut();
        if owner.is_some() { return Err(Error::Binding); }
        *owner = Some(Pending::empty(width));
        let pending_owner = owner.as_mut().ok_or(Error::Binding)?;
        pending_owner.native = Some(NativeRows::empty());
        Arc::get_mut(&mut pending_owner.native.as_mut().ok_or(Error::Binding)?.0)
            .ok_or(Error::Binding)?.native.acquire(rows)?;
        pending_owner.specs.acquire_slots(rows)?;
        for row in 0..rows {
            let start = row * width;
            let position = (row % sequence) as u32;
            let bounds = compact_validity::input_bounds(self, input, start, width)?;
            let slot = pending_owner.specs.ready_mut()?.get_mut(row).ok_or(Error::Binding)?;
            *slot = Some(Spec { position, bounds,
                cosine: tensor::ExactStorage::empty(), sine: tensor::ExactStorage::empty(),
                cosine_box: Box::default(), sine_box: Box::default(), source: None,
                schedule: tensor::WorkPreflightWorkspace::resident_schedule(), full: None, tail: None,
                program: None, coefficients_installed: false, premise: None });
            let spec = slot.as_mut().ok_or(Error::Binding)?;
            spec.cosine.acquire(width / 2)?;
            for _ in 0..width / 2 { spec.cosine.push(0)?; }
            spec.cosine.seal()?;
            spec.sine.acquire(width / 2)?;
            for _ in 0..width / 2 { spec.sine.push(0)?; }
            spec.sine.seal()?;
            tensor::Program::rotary_coefficients_into(
                theta,
                position,
                width,
                spec.cosine.ready_mut()?,
                spec.sine.ready_mut()?,
            )?;
            spec.premise = premise(self, input, row, width, spec.cosine.ready()?, spec.sine.ready()?)?;
            spec.cosine.transfer(&mut spec.cosine_box)?;
            spec.sine.transfer(&mut spec.sine_box)?;
            construction.install_exact_rotary_source(&mut spec.source, bounds, &mut spec.cosine_box, &mut spec.sine_box)
                .map_err(|_| Error::Bounds)?;
            spec.coefficients_installed = true;
            spec.schedule.install(&mut spec.source, None, 1).map_err(|_| Error::Bounds)?;
        }
        let mut pending = WorkBudget::default();
        {
            let mut workspace = tensor::WorkPreflightWorkspace::empty();
            for spec in pending_owner.specs.ready_mut()?.iter_mut().filter_map(Option::as_mut) {
                spec.schedule.bind().and_then(|_| spec.schedule.preflight(&mut workspace)).map_err(|_| Error::Bounds)?;
                pending = pending.checked_add(spec.schedule.summary().ok_or(Error::Binding)?.work)?;
                context.admit_resident(&mut spec.schedule)?;
            }
        }
        self.admit_pending_work(pending)?;
        #[cfg(test)]
        if FAIL_AFTER_DEBIT.with(|f| f.replace(false)) {
            return Err(Error::Allocation);
        }
        let mut sink = Sink {
            pending: pending_owner,
            width,
            sequence,
            theta,
            work: WorkBudget::default(),
            bounds: None,
            expected: input.clone(),
            quote: native_construction_quote(construction)?,
        };
        let result = driver::emit_rotary(&mut sink, op, shape, Some(input), 0, sequence)?;
        if result.work != pending {
            return Err(Error::Binding);
        }
        let mut retained = result.native.container_bytes()?;
        for row in &result.native {
            retained = add_bytes(
                retained,
                row.program.immutable_template_bytes()?,
            )?;
        }
        context.retain(retained)?;
        #[cfg(test)]
        if FAIL_PUBLICATION.with(|f| f.replace(false)) {
            return Err(Error::Allocation);
        }
        // Specs/schedules/coefficients and all Arc clones stay graph-owned until
        // successful outer retirement; any error/unwind preserves them instead.
        attempt.commit();
        Ok(result)
    }
}

#[cfg(test)]
thread_local! {
    static FAIL_AFTER_DEBIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static FAIL_PUBLICATION: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static FAIL_AFTER_ROW: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    static FACTORIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static OWNERS: std::cell::RefCell<Vec<std::sync::Weak<tensor::Program>>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
#[path = "compact_lowering/compact_rotary_tests.rs"]
mod tests;
