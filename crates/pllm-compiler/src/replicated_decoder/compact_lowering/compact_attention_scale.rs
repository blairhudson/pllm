//! Original scale-only traversal. Publication/mapping/reader glue stays outside.
//! No GENERAL/attention/validity certificate or probability-mass owner is issued.
use super::*;
use crate::replicated_decoder::program::EmissionContext;
use graph::attention_driver::{self, ScaleDeclaration, ScaleSink};
use pllm_garble::shared_allocation::OwnedExternalCharge;
use crate::replicated_decoder::program::request_origin::compact_storage::ControllerAccounting;
use std::mem::{size_of, size_of_val};

pub(super) struct NativeChunk {
    pub(super) start: usize,
    pub(super) lanes: usize,
    pub(super) incoming: Interval,
    pub(super) program: Arc<tensor::Program>,
}
pub(super) struct Rows {
    pub(super) native: Vec<NativeChunk>,
    pub(super) coefficient: i64,
    pub(super) work: WorkBudget,
    pub(super) bounds: Interval,
}
impl Rows {
    pub(super) fn metadata_bytes(&self) -> Result<usize, Error> {
        let (layout, _) = std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
            .extend(std::alloc::Layout::new::<Self>())
            .map_err(|_| Error::Bounds)?;
        add_bytes(
            layout.pad_to_align().size(),
            array_bytes::<NativeChunk>(self.native.capacity())?,
        )
    }
    pub(super) fn virtual_nodes(&self) -> Result<usize, Error> {
        self.native.len().checked_mul(2).ok_or(Error::Bounds)
    }
    pub(super) fn retained_bytes(&self) -> Result<usize, Error> {
        self.native
            .iter()
            .try_fold(self.metadata_bytes()?, |sum, chunk| {
                add_bytes(sum, chunk.program.resources()?.immutable_template_bytes)
            })
    }
}
struct Pending {
    start: usize,
    lanes: usize,
    incoming: Interval,
    part: Option<tensor::AdmittedFactory<'static>>,
    work: WorkBudget,
}
struct Item<T> {
    // Static ScaleSink::append cannot access sink-owned output storage. First
    // item transfers an ALREADY admitted/allocated Vec; never grow on append.
    buffer: Option<Vec<T>>,
    value: T,
}
fn append<T>(output: &mut Vec<T>, item: Item<T>) {
    if let Some(buffer) = item.buffer {
        assert!(output.is_empty() && output.capacity() == 0);
        *output = buffer;
    }
    assert!(output.len() < output.capacity(), "closed scale capacity");
    output.push(item.value);
}
fn reserve<T>(count: usize) -> Result<Vec<T>, Error> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| Error::Allocation)?;
    // Logical requested Layout, not allocator-private overhead/RSS. Do not use
    // spare capacity to admit extra descriptors or a larger semantic geometry.
    if values.capacity() != count {
        return Err(Error::Bounds);
    }
    Ok(values)
}
struct Chunk {
    input: PublishedView,
    start: usize,
    lanes: usize,
}
struct Chunks<'a> {
    input: &'a PublishedView,
    next: usize,
    width: usize,
}
impl Iterator for Chunks<'_> {
    type Item = Chunk;
    fn next(&mut self) -> Option<Chunk> {
        if self.next == self.input.lanes {
            return None;
        }
        let start = self.next;
        let lanes = self.width.min(self.input.lanes - start);
        self.next += lanes;
        Some(Chunk {
            input: self.input.clone(),
            start,
            lanes,
        })
    }
}
fn chunks(input: &PublishedView, width: usize) -> Chunks<'_> {
    Chunks {
        input,
        next: 0,
        width,
    }
}

struct Preparation<'s, 'g, 'a, G: ControllerAccounting = OwnedExternalCharge> {
    graph: &'s CompactGraph<'g, 'a, G>,
    workspace: &'s mut tensor::WorkPreflightWorkspace,
    buffer: Option<Vec<Pending>>,
}
struct Emission<'s, 'g, 'a, G: ControllerAccounting = OwnedExternalCharge> {
    graph: &'s CompactGraph<'g, 'a, G>,
    pending: Vec<Pending>,
    buffer: Option<Vec<NativeChunk>>,
    index: usize,
    coefficient: i64,
}

// Supplement the existing native ConstructionQuote with concrete lexical
// owners/move positions. No generic callback heap, ABI/call-stack or RSS bound.
fn controls<G: ControllerAccounting>(attempt: usize) -> Result<usize, Error> {
    let r = tensor::WorkPreflightWorkspace::resources();
    let mut total = attempt.checked_mul(5).ok_or(Error::Bounds)?;
    macro_rules! pair {
        ($n:expr) => {
            total = add_bytes(total, $n.checked_mul(2).ok_or(Error::Bounds)?)?;
        };
    }
    pair!(r.workspace_bytes);
    pair!(r.factory_bytes);
    pair!(r.ticket_bytes);
    pair!(r.admitted_schedule_bytes);
    pair!(r.admitted_part_bytes);
    pair!(r.summary_bytes);
    pair!(r.ledger_bytes);
    pair!(r.ticket_result_bytes);
    pair!(r.build_result_bytes);
    pair!(size_of_val(&r));
    pair!(size_of::<Preparation<'_, '_, '_, G>>());
    pair!(size_of::<Emission<'_, '_, '_, G>>());
    pair!(size_of::<Chunk>());
    pair!(size_of::<Option<Chunk>>());
    pair!(size_of::<Chunks<'_>>());
    pair!(size_of::<ScaleDeclaration>());
    pair!(size_of::<Pending>());
    pair!(size_of::<Item<Pending>>());
    pair!(size_of::<NativeChunk>());
    pair!(size_of::<Item<NativeChunk>>());
    pair!(size_of::<Rows>());
    pair!(size_of::<Result<Rows, Error>>());
    pair!(size_of::<Vec<Pending>>());
    pair!(size_of::<Result<Vec<Pending>, Error>>());
    pair!(size_of::<Vec<NativeChunk>>());
    pair!(size_of::<Result<Vec<NativeChunk>, Error>>());
    pair!(size_of::<tensor::ConstructionQuote>());
    pair!(size_of::<
        Result<tensor::ConstructionQuote, tensor::ConstructionQuoteError>,
    >());
    pair!(size_of::<tensor::FactoryOutput>());
    pair!(size_of::<Result<tensor::FactoryOutput, Error>>());
    pair!(size_of::<tensor::Resources>());
    pair!(size_of::<Result<tensor::Resources, Error>>());
    pair!(size_of::<Arc<tensor::Program>>());
    pair!(size_of::<(tensor::Program, tensor::ProducedCertificate)>());
    pair!(size_of::<WorkBudget>());
    pair!(size_of::<Result<WorkBudget, Error>>());
    pair!(size_of::<Interval>());
    pair!(size_of::<Result<Interval, Error>>());
    pair!(size_of::<ViewShape>());
    pair!(size_of::<Result<ViewShape, Error>>());
    pair!(size_of::<Result<(), Error>>());
    pair!(size_of::<&PublishedView>());
    pair!(size_of::<Option<Interval>>());
    pair!(size_of::<Result<(usize, usize), Error>>());
    pair!(size_of::<Option<&Node>>());
    pair!(size_of::<&Node>());
    pair!(size_of::<std::ops::Range<usize>>());
    pair!(size_of::<std::slice::Iter<'_, Pending>>());
    pair!(size_of::<std::slice::Iter<'_, NativeChunk>>());
    pair!(size_of::<Option<tensor::ConstructionQuote>>());
    pair!(size_of::<Result<(), std::collections::TryReserveError>>());
    pair!(size_of::<
        Result<tensor::ClosedFactory<'static>, tensor::WorkPreflightError>,
    >());
    // Original Gather and Tensor are two virtual nodes for EVERY chunk,
    // including one-word/single-chunk inputs. No expanded Node/Words allocation.
    Ok(total)
}
#[cfg(test)]
fn storage(lanes: usize, attempt: usize) -> Result<usize, Error> {
    storage_for::<OwnedExternalCharge>(lanes, attempt)
}
fn storage_for<G: ControllerAccounting>(lanes: usize, attempt: usize) -> Result<usize, Error> {
    if lanes == 0 || lanes > MAX_VALUES {
        return Err(Error::Bounds);
    }
    let full_lanes = lanes.min(128);
    let count = lanes.div_ceil(128);
    let full = tensor::Construction::Builder { lanes: full_lanes }
        .quote()
        .map_err(|_| Error::Bounds)?;
    let tail = if lanes % full_lanes != 0 {
        Some(
            tensor::Construction::Builder {
                lanes: lanes % full_lanes,
            }
            .quote()
            .map_err(|_| Error::Bounds)?,
        )
    } else {
        None
    };
    let templates = full
        .immutable_template_bytes
        .checked_mul(lanes / full_lanes)
        .ok_or(Error::Bounds)?;
    let templates = add_bytes(templates, tail.map_or(0, |q| q.immutable_template_bytes))?;
    let construction = full
        .new_peak_bytes
        .max(tail.map_or(0, |q| q.new_peak_bytes));
    // Conservative coexistence: all ticket descriptors, all output descriptors,
    // all immutable native owners PLUS one in-flight factory. No retired credit.
    add_bytes(
        add_bytes(
            array_bytes::<Pending>(count)?,
            array_bytes::<NativeChunk>(count)?,
        )?,
        add_bytes(add_bytes(templates, construction)?, controls::<G>(attempt)?)?,
    )
}

impl<G: ControllerAccounting> CompactGraph<'_, '_, G> {
    // Exact existing imported-node bounds, in original word order. A GLOBAL
    // PublishedView interval cannot stand in for per-chunk owner bounds.
    // Published-score/residual ancestry requires E's mapping-side resolver;
    // unresolved owners fail closed until that genuine mapping is available.
    fn scale_incoming(&self, chunk: &Chunk) -> Result<Interval, Error> {
        self.ordered_chunk_bounds(&chunk.input, chunk.start, chunk.lanes)
    }

    pub(super) fn lower_attention_scale(
        &self,
        op: &pllm_models::ModelOperation,
        input: &PublishedView,
        shape: &[usize],
    ) -> Result<Rows, Error> {
        let context = self.request_admission.ok_or(Error::Binding)?;
        let mut attempt = context.attempt();
        context.check_open()?;
        let baseline = add_bytes(self.outside_owner_bytes, self.baseline()?)?;
        let peak = add_bytes(baseline, storage_for::<G>(input.lanes, size_of_val(&attempt))?)?;
        self.admit_total(peak)?;
        context.admit_phase(peak, 0)?;
        let declaration = attention_driver::scale_declaration(op)?;
        let count = input.lanes.div_ceil(128);
        if self
            .next
            .checked_add(count.checked_mul(2).ok_or(Error::Bounds)?)
            .ok_or(Error::Bounds)?
            > MAX_NODES
        {
            return Err(Error::Bounds);
        }
        let pending_buffer = reserve(count)?;
        let native_buffer = reserve(count)?;
        // Workspace/pool actually ends before emission. Reservations do NOT.
        let pending = {
            let mut workspace = tensor::WorkPreflightWorkspace::empty();
            let mut sink = Preparation {
                graph: self,
                workspace: &mut workspace,
                buffer: Some(pending_buffer),
            };
            attention_driver::scale_chunks(&mut sink, &declaration, shape, Some(input))?
        };
        let pending_work = pending
            .iter()
            .try_fold(WorkBudget::default(), |w, p| w.checked_add(p.work))?;
        self.admit_pending_work(pending_work)?;
        #[cfg(test)]
        if FAIL_AFTER_DEBIT.with(|flag| flag.replace(false)) {
            return Err(Error::Allocation);
        }
        let mut sink = Emission {
            graph: self,
            pending,
            buffer: Some(native_buffer),
            index: 0,
            coefficient: declaration.coefficient(),
        };
        let rows = attention_driver::scale_chunks(&mut sink, &declaration, shape, Some(input))?;
        if rows.work != pending_work {
            return Err(Error::Binding);
        }
        context.retain(rows.retained_bytes()?)?;
        attempt.commit();
        Ok(rows)
    }
}
fn begin(input: &PublishedView, shape: &[usize]) -> Result<(), Error> {
    if ViewShape::new(shape)?.lanes()? != input.lanes {
        return Err(Error::Shape);
    }
    Ok(())
}
fn query<G: ControllerAccounting>(
    context: &EmissionContext<'_, G>,
    start: usize,
    lanes: usize,
    incoming: Interval,
    declaration: &ScaleDeclaration,
    workspace: &mut tensor::WorkPreflightWorkspace,
) -> Result<Pending, Error> {
    let ticket = tensor::Construction::Builder { lanes }
        .bind(tensor::FactoryParameters::Scale18 {
            input: incoming,
            coefficient: declaration.coefficient(),
        })
        .map_err(|_| Error::Bounds)?
        .preflight(1, workspace)
        .map_err(|_| Error::Bounds)?;
    let work = ticket.summary().work;
    let (part, tail) = context.admit_work(ticket)?.into_parts();
    if tail.is_some() {
        return Err(Error::Binding);
    }
    Ok(Pending {
        start,
        lanes,
        incoming,
        part: Some(part),
        work,
    })
}
fn build(p: &mut Pending) -> Result<NativeChunk, Error> {
    let part = p.part.take().ok_or(Error::Binding)?;
    #[cfg(test)]
    BUILD_ENTRIES.with(|n| n.set(n.get() + 1));
    let (program, certificate) = part.build()?.into_parts();
    if certificate != tensor::ProducedCertificate::None
        || program.output_lanes() != p.lanes
        || program.validity_lanes().is_some()
        || program.work_budget() != p.work
    {
        return Err(Error::Binding);
    }
    let quote = tensor::Construction::Builder { lanes: p.lanes }
        .quote()
        .map_err(|_| Error::Bounds)?;
    if program.resources()?.immutable_template_bytes > quote.immutable_template_bytes {
        return Err(Error::Bounds);
    }
    Ok(NativeChunk {
        start: p.start,
        lanes: p.lanes,
        incoming: p.incoming,
        program: Arc::new(program),
    })
}
impl<G: ControllerAccounting> ScaleSink for Preparation<'_, '_, '_, G> {
    type View = PublishedView;
    type Chunk = Chunk;
    type Owner = Item<Pending>;
    type Output = Vec<Pending>;
    type Produced = Vec<Pending>;
    fn begin(
        &mut self,
        input: &PublishedView,
        shape: &[usize],
        _: &ScaleDeclaration,
    ) -> Result<(), Error> {
        self.graph.validate_view(input)?;
        begin(input, shape)
    }
    fn chunks<'a>(input: &'a PublishedView, width: usize) -> impl Iterator<Item = Chunk> + 'a
    where
        Chunk: 'a,
    {
        chunks(input, width)
    }
    fn len(chunk: &Chunk) -> usize {
        chunk.lanes
    }
    fn native_chunk(
        &mut self,
        chunk: Chunk,
        declaration: &ScaleDeclaration,
    ) -> Result<Self::Owner, Error> {
        let incoming = self.graph.scale_incoming(&chunk)?;
        let context = self.graph.request_admission.ok_or(Error::Binding)?;
        let value = query(
            context,
            chunk.start,
            chunk.lanes,
            incoming,
            declaration,
            self.workspace,
        )?;
        Ok(Item {
            buffer: self.buffer.take(),
            value,
        })
    }
    fn append(output: &mut Vec<Pending>, owner: Self::Owner, lanes: usize) {
        assert_eq!(owner.value.lanes, lanes);
        append(output, owner);
    }
    fn finish(&mut self, output: Vec<Pending>, _: &[usize]) -> Result<Vec<Pending>, Error> {
        if output.is_empty() {
            return Err(Error::Shape);
        }
        Ok(output)
    }
}
impl<G: ControllerAccounting> ScaleSink for Emission<'_, '_, '_, G> {
    type View = PublishedView;
    type Chunk = Chunk;
    type Owner = Item<NativeChunk>;
    type Output = Vec<NativeChunk>;
    type Produced = Rows;
    fn begin(
        &mut self,
        input: &PublishedView,
        shape: &[usize],
        declaration: &ScaleDeclaration,
    ) -> Result<(), Error> {
        self.graph.validate_view(input)?;
        if declaration.coefficient() != self.coefficient {
            return Err(Error::Binding);
        }
        begin(input, shape)
    }
    fn chunks<'a>(input: &'a PublishedView, width: usize) -> impl Iterator<Item = Chunk> + 'a
    where
        Chunk: 'a,
    {
        chunks(input, width)
    }
    fn len(chunk: &Chunk) -> usize {
        chunk.lanes
    }
    fn native_chunk(&mut self, chunk: Chunk, _: &ScaleDeclaration) -> Result<Self::Owner, Error> {
        let incoming = self.graph.scale_incoming(&chunk)?;
        let p = self.pending.get_mut(self.index).ok_or(Error::Binding)?;
        if (p.start, p.lanes, p.incoming) != (chunk.start, chunk.lanes, incoming) {
            return Err(Error::Binding);
        }
        self.index += 1;
        let value = build(p)?;
        Ok(Item {
            buffer: self.buffer.take(),
            value,
        })
    }
    fn append(output: &mut Vec<NativeChunk>, owner: Self::Owner, lanes: usize) {
        assert_eq!(owner.value.lanes, lanes);
        append(output, owner);
    }
    fn finish(&mut self, native: Vec<NativeChunk>, _: &[usize]) -> Result<Rows, Error> {
        if self.index != self.pending.len() || self.pending.iter().any(|p| p.part.is_some()) {
            return Err(Error::Binding);
        }
        let mut work = WorkBudget::default();
        let mut bounds: Option<Interval> = None;
        for chunk in &native {
            work = work.checked_add(chunk.program.work_budget())?;
            let b = chunk.program.output_bounds();
            bounds = Some(match bounds {
                Some(old) => Interval::new(old.lower().min(b.lower()), old.upper().max(b.upper()))?,
                None => b,
            });
        }
        Ok(Rows {
            native,
            coefficient: self.coefficient,
            work,
            bounds: bounds.ok_or(Error::Shape)?,
        })
    }
}

#[cfg(test)]
thread_local! {
    static BUILD_ENTRIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static FAIL_AFTER_DEBIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
#[path = "compact_attention_scale_tests.rs"]
mod tests;
