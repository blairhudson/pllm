//! Owning CERTIFIED public-weight softmax component, not a semantic publisher.
//! E owns dispatch/mappings/readers. Exact Scale18/Scores publication is required
//! before whole-attention qualification; primitive input tests are NOT Scale.
//! Named logical resource inventory only; opaque dependency callframes/allocator
//! internals/physical memory and enclosing constructor completion are UNKNOWN.
use super::super::graph::attention_driver::{self as driver, AttentionSink};
use super::*;
use crate::replicated_decoder::program::EmissionContext;
use std::mem::size_of;
use std::rc::Rc;

struct Certified {
    program: Arc<tensor::Program>,
    input: Interval,
    width: usize,
    // ONLY the actual ProducedCertificate::AttentionMass constructor below.
    mass: u64,
}

/// One original INVOCATION, not one interned native Program. Fields stay private.
pub(super) struct OwnedRow<'s> {
    predecessor: PublishedView,
    predecessor_native: SoftmaxPredecessor,
    prefix: usize,
    width: usize,
    gather: usize,
    tensor: usize,
    output_start: usize,
    keys: usize,
    zero: (usize, usize),
    native: Arc<Certified>,
    context: &'s EmissionContext<'s>,
}

/// Closed local mapping; E must resolve its published aliases to these actual
/// records, never attach a record by shape/digest. No public constructor.
#[derive(Clone)]
pub(super) struct RowView<'s> {
    owner: Rc<OwnedRow<'s>>,
    start: usize,
    lanes: usize,
    reversed: bool,
}
impl RowView<'_> {
    pub(super) fn predecessor_programs(&self) -> impl Iterator<Item = &Arc<tensor::Program>> {
        self.owner.predecessor_programs()
    }
    pub(super) fn word(&self, index: usize) -> Result<(usize, usize), Error> {
        self.owner.context.check_open()?;
        if index >= self.lanes {
            return Err(Error::Shape);
        }
        let index = self
            .start
            .checked_add(if self.reversed {
                self.lanes - 1 - index
            } else {
                index
            })
            .ok_or(Error::Bounds)?;
        if index >= self.owner.keys {
            return Err(Error::Shape);
        }
        Ok(if index < self.owner.width {
            (self.owner.tensor, index)
        } else {
            self.owner.zero
        })
    }
}
impl OwnedRow<'_> {
    pub(super) fn predecessor(&self) -> &PublishedView {
        &self.predecessor
    }
    pub(super) fn predecessor_programs(&self) -> impl Iterator<Item = &Arc<tensor::Program>> {
        self.predecessor_native.programs()
    }
    pub(super) fn prefix_offset(&self) -> usize {
        self.prefix
    }
    pub(super) fn program(&self) -> &Arc<tensor::Program> {
        &self.native.program
    }
    pub(super) fn native_width(&self) -> usize {
        self.width
    }
    pub(super) fn input_bounds(&self) -> Interval {
        self.native.input
    }
    pub(super) fn output_bounds(&self) -> Interval {
        self.native.program.output_bounds()
    }
    pub(super) fn mass_bound(&self) -> u64 {
        self.native.mass
    }
    pub(super) fn gather_ordinal(&self) -> usize {
        self.gather
    }
    pub(super) fn tensor_ordinal(&self) -> usize {
        self.tensor
    }
    pub(super) fn output_start(&self) -> usize {
        self.output_start
    }
    pub(super) fn keys(&self) -> usize {
        self.keys
    }
    pub(super) fn zero_word(&self) -> (usize, usize) {
        self.zero
    }
    pub(super) fn input_word(&self, index: usize) -> Result<(usize, usize), Error> {
        self.context.check_open()?;
        if index >= self.width {
            return Err(Error::Shape);
        }
        self.predecessor.word(add_bytes(self.prefix, index)?)
    }
}

pub(super) struct Rows<'s> {
    native: Vec<Rc<OwnedRow<'s>>>,
    first: usize,
    cursor: usize,
    work: WorkBudget,
    retained: usize,
}
impl<'s> Rows<'s> {
    pub(super) fn programs(&self) -> impl Iterator<Item = &Arc<tensor::Program>> {
        self.native.iter().map(|row| row.program())
    }
    pub(super) fn len(&self) -> usize {
        self.native.len()
    }
    pub(super) fn first(&self) -> usize {
        self.first
    }
    pub(super) fn cursor(&self) -> usize {
        self.cursor
    }
    pub(super) fn work(&self) -> WorkBudget {
        self.work
    }
    pub(super) fn retained_bytes(&self) -> usize {
        self.retained
    }
    pub(super) fn metadata_bytes(&self) -> Result<usize, Error> {
        fn shared_bytes<T, H>() -> Result<usize, Error> {
            let (layout, _) = std::alloc::Layout::new::<H>()
                .extend(std::alloc::Layout::new::<T>())
                .map_err(|_| Error::Bounds)?;
            Ok(layout.pad_to_align().size())
        }
        // Metadata only. Native Program allocations are enumerated separately.
        // Certified metadata is conservatively allowed once per invocation even
        // where the same certified Arc is reused; no digest-based shared credit.
        add_bytes(
            add_bytes(
                size_of::<Self>(),
                self.native
                    .capacity()
                    .checked_mul(size_of::<Rc<OwnedRow<'_>>>())
                    .ok_or(Error::Bounds)?,
            )?,
            self.native
                .len()
                .checked_mul(add_bytes(
                    shared_bytes::<OwnedRow<'_>, [std::cell::Cell<usize>; 2]>()?,
                    shared_bytes::<Certified, [std::sync::atomic::AtomicUsize; 2]>()?,
                )?)
                .ok_or(Error::Bounds)?,
        )
    }
    pub(super) fn bounds(&self) -> Result<Interval, Error> {
        let mut bounds = None::<Interval>;
        for row in &self.native {
            let mut b = row.output_bounds();
            if row.width < row.keys {
                b = Interval::new(b.lower().min(0), b.upper().max(0))?;
            }
            bounds = Some(match bounds {
                None => b,
                Some(a) => Interval::new(a.lower().min(b.lower()), a.upper().max(b.upper()))?,
            });
        }
        bounds.ok_or(Error::Binding)
    }
    pub(super) fn row(&self, index: usize) -> Result<&OwnedRow<'s>, Error> {
        let row = self.native.get(index).ok_or(Error::Shape)?;
        row.context.check_open()?;
        Ok(row)
    }
    pub(super) fn prefix(&self, index: usize) -> Result<RowView<'s>, Error> {
        let row = self.row(index)?;
        Ok(RowView {
            owner: Rc::clone(&self.native[index]),
            start: 0,
            lanes: row.width,
            reversed: false,
        })
    }
    pub(super) fn padded(&self, index: usize) -> Result<RowView<'s>, Error> {
        let mut view = self.prefix(index)?;
        view.lanes = view.owner.keys;
        Ok(view)
    }
    /// ACTUAL complete native prefix only (Astra319). Padding is not silently
    /// trimmed. Equal Program Arcs do not identify different invocations.
    pub(super) fn lookup<'r>(
        &'r self,
        view: &RowView<'s>,
        inner: usize,
    ) -> Result<Option<&'r OwnedRow<'s>>, Error> {
        view.owner.context.check_open()?;
        let Some(row) = self.native.iter().find(|r| Rc::ptr_eq(r, &view.owner)) else {
            return Err(Error::Binding);
        };
        if inner != row.width || view.lanes != inner || view.start != 0 || view.reversed {
            return Err(Error::Binding);
        }
        for lane in 0..inner {
            if view.word(lane)? != (row.tensor, lane) {
                return Err(Error::Binding);
            }
        }
        if row.native.width != inner
            || row.native.program.output_lanes() != inner
            || row.native.program.validity_lanes().is_some()
        {
            return Err(Error::Binding);
        }
        Ok(Some(row))
    }
    /// Separate publication geometry check, NOT a native row certificate.
    pub(super) fn check_padding(&self, index: usize, view: &RowView<'s>) -> Result<(), Error> {
        let row = self.row(index)?;
        if !Rc::ptr_eq(&self.native[index], &view.owner)
            || view.start != 0
            || view.lanes != row.keys
            || view.reversed
        {
            return Err(Error::Binding);
        }
        for lane in 0..row.keys {
            let expected = if lane < row.width {
                (row.tensor, lane)
            } else {
                row.zero
            };
            if view.word(lane)? != expected {
                return Err(Error::Binding);
            }
        }
        Ok(())
    }
    /// E's consumer entry AFTER resolving a real published alias. Known native
    /// Tensor + broken prefix is an error; genuine absence remains None. IDs
    /// only select a creation-issued record, never mint one. The receiving Rows
    /// must be the ACTUAL child stored in the producing publication.
    pub(super) fn lookup_published(
        &self,
        view: &PublishedView,
        inner: usize,
    ) -> Result<Option<&OwnedRow<'s>>, Error> {
        let first = self.native.first().ok_or(Error::Binding)?;
        first.context.check_open()?;
        if !Arc::ptr_eq(&first.predecessor.origin, &view.origin) {
            return Err(Error::Binding);
        }
        if view.lanes == 0 {
            return Ok(None);
        }
        let head = view.word(0)?;
        let Some(row) = self.native.iter().find(|r| r.tensor == head.0) else {
            return Ok(None);
        };
        if inner != row.width || view.lanes != inner {
            return Err(Error::Binding);
        }
        for lane in 0..inner {
            if view.word(lane)? != (row.tensor, lane) {
                return Err(Error::Binding);
            }
        }
        Ok(Some(row))
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Resources {
    pub(super) records: usize,
    pub(super) tickets: usize,
    pub(super) prefixes: usize,
    pub(super) query_controls: usize,
    pub(super) build_controls: usize,
    pub(super) retained: usize,
    pub(super) peak: usize,
}
fn arc_bytes<T>() -> Result<usize, Error> {
    let a = std::mem::align_of::<T>();
    let header = add_bytes(2 * size_of::<usize>(), a - 1)? / a * a;
    add_bytes(header, size_of::<T>())
}
fn geometry(
    input: &PublishedView,
    sequence: usize,
    position: usize,
) -> Result<(usize, usize), Error> {
    let keys = add_bytes(position, sequence)?;
    if sequence == 0
        || keys == 0
        || input.lanes == 0
        || input.lanes % keys != 0
        || input.lanes / keys % sequence != 0
    {
        return Err(Error::Shape);
    }
    Ok((keys, input.lanes / keys))
}

// Native bounds come from actual prepared owner(s), NEVER a widened whole-view
// estimate. Ordinary original imports have one actual Input node interval. The
// existing full/tail map is supported; unknown Scale/alias kinds fail closed.
fn prefix_bounds(
    graph: &CompactGraph<'_, '_>,
    input: &PublishedView,
    start: usize,
    width: usize,
) -> Result<Interval, Error> {
    graph.validate_view(input)?;
    if width == 0 || add_bytes(start, width)? > input.lanes {
        return Err(Error::Shape);
    }
    if let Some(slot) = input.slot {
        if matches!(
            graph.softmax_predecessor(input)?,
            SoftmaxPredecessor::Scale(_)
        ) {
            return graph.ordered_chunk_bounds(input, start, width);
        }
        if input.start != 0 || input.order != LaneOrder::Linear {
            return Err(Error::Binding);
        }
        let map = graph.kernel(slot)?;
        if map.lanes() != input.lanes {
            return Err(Error::Binding);
        }
        // Only the actual two-node native-map output, NOT a projection's later
        // bias-add/other arithmetic with a retained prepared predecessor. Such
        // unknown producers require their own exact creation-bound loan in E.
        let Mapping::Rows { first, width: tile } = input.mapping else {
            return Err(Error::Binding);
        };
        if tile == 0 {
            return Err(Error::Binding);
        }
        let mut result = None::<Interval>;
        for lane in start..start + width {
            let program = map.program_at((lane / tile) * tile)?;
            let native = add_bytes(first, (lane / tile).checked_mul(2).ok_or(Error::Bounds)?)?;
            if input.word(lane)? != (add_bytes(native, 1)?, lane % tile)
                || program.output_lanes() != (input.lanes - (lane / tile) * tile).min(tile)
                || program.lanes() != program.output_lanes()
                || program.validity_lanes().is_some()
            {
                return Err(Error::Binding);
            }
            let b = program.output_bounds();
            result = Some(match result {
                None => b,
                Some(a) => Interval::new(a.lower().min(b.lower()), a.upper().max(b.upper()))?,
            });
        }
        result.ok_or(Error::Binding)
    } else {
        Ok(input.bounds)
    }
}

pub(super) fn quote(
    graph: &CompactGraph<'_, '_>,
    input: &PublishedView,
    sequence: usize,
    position: usize,
) -> Result<Resources, Error> {
    let context = graph.request_admission.ok_or(Error::Binding)?;
    let mut attempt = context.attempt();
    context.check_open()?;
    require_public_weights(graph.imports.public_weights)?;
    graph.validate_view(input)?;
    let (keys, count) = geometry(input, sequence, position)?;
    if add_bytes(graph.next, count.checked_mul(2).ok_or(Error::Bounds)?)? > MAX_NODES {
        return Err(Error::Bounds);
    }
    let r = tensor::WorkPreflightWorkspace::resources();
    let records = count
        .checked_mul(size_of::<Rc<OwnedRow<'_>>>())
        .ok_or(Error::Bounds)?;
    let tickets = count
        .checked_mul(size_of::<tensor::AdmittedFactory<'static>>())
        .ok_or(Error::Bounds)?;
    let prefixes = count
        .checked_mul(size_of::<Interval>())
        .ok_or(Error::Bounds)?;
    let common = [
        size_of::<Rows<'_>>(),
        size_of::<Result<Rows<'_>, Error>>(),
        size_of::<Sink<'_>>(),
        size_of::<Resources>(),
        size_of::<WalkView>(),
        4 * size_of::<WalkWords>(),
        4 * size_of::<RowView<'_>>(),
        2 * size_of::<PublishedView>(),
        size_of::<std::vec::IntoIter<tensor::AdmittedFactory<'static>>>(),
        4 * size_of::<PublishedOp<'_>>(),
        size_of::<Result<tensor::FactoryOutput, Error>>(),
        size_of::<(tensor::Program, tensor::ProducedCertificate)>(),
        size_of::<Result<(tensor::Program, tensor::ProducedCertificate), Error>>(),
        4 * size_of::<request_origin::compact_storage::EmissionAttempt<'_, '_>>(),
        size_of::<Result<Option<&OwnedRow<'_>>, Error>>(),
        size_of::<Option<Interval>>(),
        2 * size_of::<OwnedRow<'_>>(),
        2 * size_of::<Certified>(),
        size_of::<Result<Rc<OwnedRow<'_>>, Error>>(),
    ]
    .into_iter()
    .try_fold(0, add_bytes)?;
    // Explicit original-prelude loan/result positions and returned native query.
    let common = [
        common,
        size_of::<request_origin::compact_storage::Prelude>(),
        size_of::<Result<request_origin::compact_storage::Prelude, Error>>(),
        size_of::<tensor::ConstructionQuote>(),
        size_of::<Result<tensor::ConstructionQuote, tensor::ConstructionQuoteError>>(),
    ]
    .into_iter()
    .try_fold(0, add_bytes)?;
    let query_controls = [
        common,
        r.workspace_bytes,
        r.factory_bytes,
        r.ticket_bytes,
        r.admitted_schedule_bytes,
        r.admitted_part_bytes,
        r.summary_bytes,
        r.ledger_bytes,
        r.ticket_result_bytes,
        r.build_result_bytes,
    ]
    .into_iter()
    .try_fold(0, add_bytes)?;
    let query_controls = [
        query_controls,
        size_of::<Vec<Interval>>(),
        size_of::<Result<Vec<Interval>, Error>>(),
        size_of::<Vec<tensor::AdmittedFactory<'static>>>(),
        size_of::<Result<Vec<tensor::AdmittedFactory<'static>>, Error>>(),
        size_of::<Vec<Rc<OwnedRow<'_>>>>(),
        size_of::<Result<Vec<Rc<OwnedRow<'_>>>, Error>>(),
    ]
    .into_iter()
    .try_fold(0, add_bytes)?;
    let build_controls = add_bytes(common, size_of::<Arc<Certified>>())?;
    let mut retained = add_bytes(size_of::<Rows<'_>>(), records)?;
    let mut native_peak = retained;
    for row in 0..count {
        let width = add_bytes(position, row % sequence + 1)?;
        let q = tensor::Construction::SoftmaxCertified { rows: 1, width }
            .quote()
            .map_err(|_| Error::Bounds)?;
        let publishing = add_bytes(arc_bytes::<Certified>()?, arc_bytes::<OwnedRow<'_>>()?)?;
        native_peak = native_peak.max(add_bytes(
            add_bytes(retained, q.new_peak_bytes)?,
            publishing,
        )?);
        retained = [
            retained,
            q.immutable_template_bytes,
            arc_bytes::<Certified>()?,
            arc_bytes::<OwnedRow<'_>>()?,
        ]
        .into_iter()
        .try_fold(0, add_bytes)?;
    }
    let baseline = add_bytes(graph.outside_owner_bytes, graph.baseline()?)?;
    // Disjoint query/build phases MAX, not their sum. No inner retirement/refund.
    let query = add_bytes(
        add_bytes(add_bytes(records, tickets)?, prefixes)?,
        query_controls,
    )?;
    let build = add_bytes(
        add_bytes(add_bytes(native_peak.max(retained), tickets)?, prefixes)?,
        build_controls,
    )?;
    let peak = add_bytes(baseline, query.max(build))?;
    let _ = keys;
    graph.admit_total(peak)?;
    context.admit_phase(peak, 0)?;
    attempt.commit();
    Ok(Resources {
        records,
        tickets,
        prefixes,
        query_controls,
        build_controls,
        retained,
        peak,
    })
}

#[derive(Clone, Default)]
struct WalkWords {
    input: Option<PublishedView>,
    start: usize,
    len: usize,
    valid: bool,
}
#[derive(Clone)]
struct WalkView {
    words: WalkWords,
    shape: ViewShape,
}
struct Sink<'s> {
    context: &'s EmissionContext<'s>,
    input: PublishedView,
    tickets: std::vec::IntoIter<tensor::AdmittedFactory<'static>>,
    rows: Vec<Rc<OwnedRow<'s>>>,
    first: usize,
    keys: usize,
    sequence: usize,
    position: usize,
    zero: (usize, usize),
    work: WorkBudget,
    predecessor: SoftmaxPredecessor,
    inputs: Vec<Interval>,
}
impl AttentionSink for Sink<'_> {
    type View = WalkView;
    type Words = WalkWords;
    type Word = (usize, usize);
    type Owner = usize;
    fn shape(v: &WalkView) -> &[usize] {
        v.shape.dimensions()
    }
    fn words(v: &WalkView) -> &WalkWords {
        &v.words
    }
    fn len(w: &WalkWords) -> usize {
        w.len
    }
    fn slice(w: &WalkWords, r: std::ops::Range<usize>) -> WalkWords {
        WalkWords {
            input: w.input.clone(),
            start: w.start + r.start,
            len: r.end - r.start,
            valid: w.valid && r.start <= r.end && r.end <= w.len,
        }
    }
    fn rows(w: &WalkWords, width: usize) -> impl Iterator<Item = WalkWords> + '_ {
        (0..w.len / width).map(move |i| Self::slice(w, i * width..(i + 1) * width))
    }
    fn chunks(w: &WalkWords, width: usize) -> impl Iterator<Item = WalkWords> + '_ {
        (0..w.len.div_ceil(width))
            .map(move |i| Self::slice(w, i * width..((i + 1) * width).min(w.len)))
    }
    fn at(w: &WalkWords, i: usize) -> (usize, usize) {
        w.input
            .as_ref()
            .expect("input walk")
            .word(w.start + i)
            .expect("checked walk")
    }
    fn push(w: &mut WalkWords, _: (usize, usize)) {
        w.len += 1;
    }
    fn extend(w: &mut WalkWords, more: WalkWords) {
        w.len += more.len;
    }
    fn repeat(w: &mut WalkWords, _: (usize, usize), n: usize) {
        w.len += n;
    }
    fn append_owner(w: &mut WalkWords, _: usize, n: usize) {
        w.len += n;
    }
    fn view(words: WalkWords, shape: &[usize]) -> WalkView {
        WalkView {
            words,
            shape: ViewShape::new(shape).expect("prevalidated shape"),
        }
    }
    fn rotary_row(&mut self, _: &WalkWords, _: f32, _: u32, _: usize) -> Result<usize, Error> {
        Err(Error::Binding)
    }
    fn scores(&mut self, _: &WalkView, _: &WalkView) -> Result<WalkWords, Error> {
        Err(Error::Binding)
    }
    fn values(&mut self, _: &WalkView, _: &WalkView) -> Result<WalkWords, Error> {
        Err(Error::Binding)
    }
    fn scale_coefficient(_: u64) -> i64 {
        unreachable!("softmax-only adapter")
    }
    fn scale_chunk(&mut self, _: &WalkWords, _: i64) -> Result<usize, Error> {
        Err(Error::Binding)
    }
    fn softmax_row(&mut self, row: &WalkWords, width: usize) -> Result<usize, Error> {
        let i = self.rows.len();
        if !row.valid
            || row.start != i * self.keys
            || row.len != self.keys
            || width != self.position + i % self.sequence + 1
        {
            return Err(Error::Binding);
        }
        let source = row.input.as_ref().ok_or(Error::Binding)?;
        if !Arc::ptr_eq(&source.origin, &self.input.origin)
            || source.slot != self.input.slot
            || source.mapping != self.input.mapping
            || source.start != self.input.start
            || source.order != self.input.order
        {
            return Err(Error::Binding);
        }
        let admitted = self.tickets.next().ok_or(Error::Binding)?;
        // Input parameter is fixed by the admitted ticket. Read from the actual
        // predecessor already validated before tickets, not a caller interval.
        let input = *self.inputs.get(i).ok_or(Error::Binding)?;
        let native = if let Some(old) = self
            .rows
            .iter()
            .find(|r| r.width == width && r.native.input == input)
        {
            let _admitted_use = admitted;
            Arc::clone(&old.native)
        } else {
            #[cfg(test)]
            FACTORIES.with(|n| n.set(n.get() + 1));
            let (program, certificate) = admitted.build()?.into_parts();
            let tensor::ProducedCertificate::AttentionMass(mass) = certificate else {
                return Err(Error::Binding);
            };
            if program.lanes() != width
                || program.output_lanes() != width
                || program.validity_lanes().is_some()
            {
                return Err(Error::Binding);
            }
            Arc::new(Certified {
                program: Arc::new(program),
                input,
                width,
                mass,
            })
        };
        #[cfg(test)]
        PANIC_AFTER_NATIVE.with(|flag| assert!(!flag.replace(false), "softmax after native"));
        self.work = self.work.checked_add(native.program.work_budget())?;
        let gather = add_bytes(self.first, i.checked_mul(2).ok_or(Error::Bounds)?)?;
        self.rows.push(Rc::new(OwnedRow {
            predecessor: self.input.clone(),
            predecessor_native: self.predecessor.clone(),
            prefix: row.start,
            width,
            gather,
            tensor: add_bytes(gather, 1)?,
            output_start: i * self.keys,
            keys: self.keys,
            zero: self.zero,
            native,
            context: self.context,
        }));
        Ok(i)
    }
}

fn reserve<T>(count: usize) -> Result<Vec<T>, Error> {
    let mut v = Vec::new();
    v.try_reserve_exact(count).map_err(|_| Error::Allocation)?;
    if v.capacity() != count {
        return Err(Error::Bounds);
    }
    Ok(v)
}

pub(super) fn prepare<'s>(
    graph: &CompactGraph<'_, 's>,
    input: &PublishedView,
    sequence: usize,
    position: usize,
) -> Result<Rows<'s>, Error> {
    let context = graph.request_admission.ok_or(Error::Binding)?;
    let mut attempt = context.attempt();
    context.check_open()?;
    let resources = quote(graph, input, sequence, position)?;
    let (keys, count) = geometry(input, sequence, position)?;
    let end = add_bytes(graph.next, count.checked_mul(2).ok_or(Error::Bounds)?)?;
    if end > MAX_NODES {
        return Err(Error::Bounds);
    }
    // All declared buffers and named controls are admitted BEFORE reservation.
    let mut inputs = reserve(count)?;
    let mut tickets = reserve(count)?;
    let rows = reserve(count)?;
    let predecessor = graph.softmax_predecessor(input)?;
    let mut pending = WorkBudget::default();
    {
        let mut workspace = tensor::WorkPreflightWorkspace::empty();
        for row in 0..count {
            let width = add_bytes(position, row % sequence + 1)?;
            let bounds = prefix_bounds(graph, input, row * keys, width)?;
            inputs.push(bounds);
            let ticket = tensor::Construction::SoftmaxCertified { rows: 1, width }
                .bind(tensor::FactoryParameters::SoftmaxCertified { input: bounds })
                .map_err(|_| Error::Bounds)?
                .preflight(1, &mut workspace)
                .map_err(|_| Error::Bounds)?;
            pending = pending.checked_add(ticket.summary().work)?;
            let (part, tail) = context.admit_work(ticket)?.into_parts();
            if tail.is_some() {
                return Err(Error::Binding);
            }
            tickets.push(part);
        }
    } // Query workspace is retired; no counter/phase refund.
    let zero = (add_bytes(context.prelude()?.first(), 1)?, 0);
    let mut sink = Sink {
        context,
        input: input.clone(),
        tickets: tickets.into_iter(),
        rows,
        first: graph.next,
        keys,
        sequence,
        position,
        zero,
        work: WorkBudget::default(),
        predecessor,
        inputs,
    };
    let view = WalkView {
        words: WalkWords {
            input: Some(input.clone()),
            start: 0,
            len: input.lanes,
            valid: true,
        },
        shape: input.logical,
    };
    let output = driver::softmax(
        &mut sink,
        input.logical.dimensions(),
        Some(&view),
        sequence,
        position,
        keys,
        zero,
    )?;
    if output.words.len != input.lanes
        || sink.rows.len() != count
        || sink.tickets.next().is_some()
        || sink.work != pending
    {
        return Err(Error::Binding);
    }
    let result = Rows {
        native: sink.rows,
        first: graph.next,
        cursor: graph.cursor,
        work: pending,
        retained: resources.retained,
    };
    // Numeric resources and certificate geometry are native-owned, never guessed.
    for row in &result.native {
        let q = tensor::Construction::SoftmaxCertified {
            rows: 1,
            width: row.width,
        }
        .quote()
        .map_err(|_| Error::Bounds)?;
        if row.native.program.resources()?.immutable_template_bytes > q.immutable_template_bytes {
            return Err(Error::Bounds);
        }
    }
    context.retain(resources.retained)?;
    attempt.commit();
    Ok(result)
}

#[cfg(test)]
thread_local! { static FACTORIES:std::cell::Cell<usize>=const {std::cell::Cell::new(0)}; }
#[cfg(test)]
thread_local! { static PANIC_AFTER_NATIVE:std::cell::Cell<bool>=const {std::cell::Cell::new(false)}; }
fn require_public_weights(public: bool) -> Result<(), Error> {
    if public {
        Ok(())
    } else {
        Err(Error::Binding)
    }
}
#[cfg(test)]
#[path = "compact_attention_softmax_tests.rs"]
mod tests;
