//! Canonical E6 owning-row child, with bounded R child repairs. This postimage
//! is for E's compact_rotary.rs mount; no shared E glue is changed here.
use super::super::graph::attention_driver::{self as driver, RotarySink};
use super::*;
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
    pub(super) native: Vec<NativeRow>,
    pub(super) width: usize,
    pub(super) work: WorkBudget,
    pub(super) bounds: Interval,
}
struct Spec {
    position: u32,
    bounds: Interval,
    cosine: Vec<i64>,
    sine: Vec<i64>,
    premise: Option<RotaryEnergyPremise>,
}
struct MappedRow {
    input: PublishedView,
    start: usize,
}
struct Sink<'s> {
    specs: &'s [Spec],
    tickets: std::vec::IntoIter<tensor::AdmittedFactory<'s>>,
    native: Vec<NativeRow>,
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

fn premise(
    graph: &CompactGraph<'_, '_>,
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

impl RotarySink for Sink<'_> {
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
        let index = self.native.len();
        let spec = self.specs.get(index).ok_or(Error::Binding)?;
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
        let factory = self.tickets.next().ok_or(Error::Binding)?;
        #[cfg(test)]
        FACTORIES.with(|n| n.set(n.get() + 1));
        let (program, certificate) = factory.build()?.into_parts();
        if certificate != tensor::ProducedCertificate::None {
            return Err(Error::Binding);
        }
        let resources = program.resources()?;
        if resources.immutable_template_bytes > self.quote.immutable_template_bytes
            || resources.shared_admitted_bytes > puma::MAX_ADMITTED_BYTES
        {
            return Err(Error::Bounds);
        }
        let program = Arc::new(program);
        #[cfg(test)]
        OWNERS.with(|owners| owners.borrow_mut().push(Arc::downgrade(&program)));
        let bounds = program.output_bounds();
        self.work = self.work.checked_add(program.work_budget())?;
        self.bounds = Some(match self.bounds {
            Some(b) => Interval::new(b.lower().min(bounds.lower()), b.upper().max(bounds.upper()))?,
            None => bounds,
        });
        if self.native.len() == self.native.capacity() {
            return Err(Error::Bounds);
        }
        self.native.push(NativeRow {
            position,
            program,
            incoming: spec.bounds,
            bounds,
            attention_energy: spec.premise.map(|p| p.attention_squared),
            premise: spec.premise,
        });
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
        if self.tickets.next().is_some()
            || self.native.len() != self.specs.len()
            || output != self.expected.lanes
            || shape != &self.expected.logical.dimensions[..4]
        {
            return Err(Error::Binding);
        }
        Ok(Rows {
            native: std::mem::take(&mut self.native),
            width: self.width,
            work: self.work,
            bounds: self.bounds.ok_or(Error::Shape)?,
        })
    }
}

/// Existing canonical two phases, augmented with actual local/control/return
/// owners. These are logical source-level bounds, NOT complete stack/RSS proof.
fn control_bytes(rows: usize, lanes: usize, attempt: usize) -> Result<(usize, usize), Error> {
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
    slot!(array_bytes::<Spec>(rows)?);
    slot!(array_bytes::<NativeRow>(rows)?);
    slot!(array_bytes::<tensor::AdmittedFactory<'_>>(rows)?);
    slot!(array_bytes::<i64>(lanes)?); // both exact half-width coefficient buffers
    pair!(size_of::<Vec<Spec>>());
    pair!(size_of::<Vec<NativeRow>>());
    pair!(size_of::<Vec<tensor::AdmittedFactory<'_>>>());
    pair!(size_of::<Spec>());
    pair!(size_of::<NativeRow>());
    pair!(size_of::<MappedRow>());
    pair!(size_of::<PublishedView>());
    slot!(size_of::<Sink<'_>>());
    pair!(size_of::<Rows>());
    pair!(size_of::<Result<Rows, Error>>());
    pair!(size_of::<tensor::ConstructionQuote>());
    slot!(size_of::<NativeConstructionPeak>());
    pair!(size_of::<tensor::WorkPreflightResources>());
    pair!(size_of::<Option<RotaryEnergyPremise>>());
    pair!(size_of::<Result<Option<RotaryEnergyPremise>, Error>>());
    pair!(size_of::<Result<Option<(&PublishedView, usize)>, Error>>());
    slot!(size_of::<(&CompactGraph<'_, '_>, &PublishedView, &usize)>()); // GENERAL closure
    slot!(size_of::<(
        &CompactGraph<'_, '_>,
        &PublishedView,
        &usize,
        &usize
    )>()); // attention closure
    pair!(size_of::<Result<Option<(EnergyFamily, u128)>, Error>>());
    slot!(size_of::<std::vec::IntoIter<tensor::AdmittedFactory<'_>>>());
    slot!(size_of::<std::slice::Iter<'_, Spec>>());
    slot!(size_of::<(tensor::Program, tensor::ProducedCertificate)>());
    slot!(size_of::<Result<tensor::FactoryOutput, Error>>());
    pair!(size_of::<Arc<tensor::Program>>());
    pair!(r.factory_bytes);
    pair!(r.ticket_bytes);
    pair!(r.admitted_schedule_bytes);
    pair!(r.admitted_part_bytes);
    pair!(r.ticket_result_bytes);
    pair!(r.build_result_bytes);
    pair!(r.summary_bytes);
    pair!(r.ledger_bytes);
    // Outer lower_next, lower_rotary and nested work-admission attempts.
    slot!(attempt.checked_mul(3).ok_or(Error::Bounds)?);
    Ok((add_bytes(bytes, r.workspace_bytes)?, bytes))
}

impl CompactGraph<'_, '_> {
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
        let mut native = NativeConstructionPeak::default();
        native.include_repeated(construction, rows)?;
        let (query, controls) =
            control_bytes(rows, rows.checked_mul(width).ok_or(Error::Bounds)?, attempt)?;
        let arc_headers = array_bytes::<usize>(rows.checked_mul(2).ok_or(Error::Bounds)?)?;
        let build = add_bytes(add_bytes(controls, native.bytes()?)?, arc_headers)?;
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
        context
            .plan()
            .prepay_compiler_metadata(
                &context.reservation(),
                u64::try_from(visits).map_err(|_| Error::Bounds)?,
            )
            .map_err(|_| Error::Bounds)?;
        let mut specs = Vec::new();
        specs
            .try_reserve_exact(rows)
            .map_err(|_| Error::Allocation)?;
        let mut programs = Vec::new();
        programs
            .try_reserve_exact(rows)
            .map_err(|_| Error::Allocation)?;
        let mut tickets = Vec::new();
        tickets
            .try_reserve_exact(rows)
            .map_err(|_| Error::Allocation)?;
        if specs.capacity() != rows || programs.capacity() != rows || tickets.capacity() != rows {
            return Err(Error::Allocation);
        }
        for row in 0..rows {
            let start = row * width;
            let position = (row % sequence) as u32;
            let bounds = compact_validity::input_bounds(self, input, start, width)?;
            let mut cosine = Vec::new();
            let mut sine = Vec::new();
            cosine
                .try_reserve_exact(width / 2)
                .map_err(|_| Error::Allocation)?;
            sine.try_reserve_exact(width / 2)
                .map_err(|_| Error::Allocation)?;
            if cosine.capacity() != width / 2 || sine.capacity() != width / 2 {
                return Err(Error::Allocation);
            }
            cosine.resize(width / 2, 0);
            sine.resize(width / 2, 0);
            tensor::Program::rotary_coefficients_into(
                theta,
                position,
                width,
                &mut cosine,
                &mut sine,
            )?;
            let premise = premise(self, input, row, width, &cosine, &sine)?;
            specs.push(Spec {
                position,
                bounds,
                cosine,
                sine,
                premise,
            });
        }
        let mut pending = WorkBudget::default();
        {
            let mut workspace = tensor::WorkPreflightWorkspace::empty();
            for spec in &specs {
                let ticket = construction
                    .bind(tensor::FactoryParameters::RotaryHalf {
                        input: spec.bounds,
                        cosine: &spec.cosine,
                        sine: &spec.sine,
                    })
                    .and_then(|f| f.preflight(1, &mut workspace))
                    .map_err(|_| Error::Bounds)?;
                pending = pending.checked_add(ticket.summary().work)?;
                tickets.push(context.admit_work(ticket)?.into_parts().0);
            }
        }
        self.admit_pending_work(pending)?;
        #[cfg(test)]
        if FAIL_AFTER_DEBIT.with(|f| f.replace(false)) {
            return Err(Error::Allocation);
        }
        let mut sink = Sink {
            specs: &specs,
            tickets: tickets.into_iter(),
            native: programs,
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
        let mut retained = array_bytes::<NativeRow>(result.native.capacity())?;
        for row in &result.native {
            retained = add_bytes(
                retained,
                add_bytes(
                    row.program.resources()?.immutable_template_bytes,
                    2 * size_of::<usize>(),
                )?,
            )?;
        }
        context.retain(retained)?;
        #[cfg(test)]
        if FAIL_PUBLICATION.with(|f| f.replace(false)) {
            return Err(Error::Allocation);
        }
        // Specs/factory parts/control slots retire on RETURN, never retain().
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
