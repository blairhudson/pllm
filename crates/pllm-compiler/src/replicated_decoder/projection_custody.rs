//! Shared declaration/recipe custody. Q18 authority and supplied CLEAR numeric
//! storage are deliberately separate contracts. No raw-format source authority.
use super::*;
use super::produced::OriginalProducedView;
use pllm_models::ModelOperation;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::replicated_decoder::program) struct Geometry {
    pub(super) dimensions: [usize; 4],
    pub(super) rank: usize,
    lanes: usize,
}
impl Geometry {
    pub(super) fn new(dimensions: &[usize]) -> Result<Self, Error> {
        if !(1..=4).contains(&dimensions.len()) { return Err(Error::Shape); }
        let mut result = Self { dimensions: [1; 4], rank: dimensions.len(), lanes: 1 };
        for (out, dimension) in result.dimensions.iter_mut().zip(dimensions) {
            result.lanes = result.lanes.checked_mul(*dimension).ok_or(Error::Bounds)?;
            if *dimension == 0 || result.lanes > MAX_VALUES { return Err(Error::Shape); }
            *out = *dimension;
        }
        Ok(result)
    }
    pub(super) fn operation(op: &ModelOperation) -> Result<Self, Error> {
        if !(1..=4).contains(&op.output_shape.len()) { return Err(Error::Shape); }
        let mut dimensions = [1; 4];
        for (out, dimension) in dimensions.iter_mut().zip(&op.output_shape) {
            *out = usize::try_from(*dimension).map_err(|_| Error::Bounds)?;
        }
        Self::new(&dimensions[..op.output_shape.len()])
    }
    pub(in crate::replicated_decoder::program) fn dimensions(&self) -> &[usize] {
        &self.dimensions[..self.rank]
    }
    pub(in crate::replicated_decoder::program) fn lanes(self) -> usize { self.lanes }
    pub(super) fn width(self) -> usize { self.dimensions[self.rank - 1] }
}

mod sealed { pub trait Sealed {} }
/// Declaration identity, not runtime-buffer custody or a numeric certificate.
pub(in crate::replicated_decoder::program) trait OriginalInputBinding<'s>: sealed::Sealed {
    fn binding(&self) -> BindingRef<'_, 's>;
    fn ordered_coordinate(&self, index: usize) -> Result<OrderedCoordinate<'_, 's>, Error> {
        self.binding().coordinate(index)
    }
}
pub(in crate::replicated_decoder::program) enum BindingRef<'b, 's> {
    Imported(&'b OriginalImportedView<'s>),
    Produced(&'b OriginalProducedView<'s>),
}
impl sealed::Sealed for OriginalImportedView<'_> {}
impl<'s> OriginalInputBinding<'s> for OriginalImportedView<'s> {
    fn binding(&self) -> BindingRef<'_, 's> { BindingRef::Imported(self) }
}
impl sealed::Sealed for OriginalProducedView<'_> {}
impl<'s> OriginalInputBinding<'s> for OriginalProducedView<'s> {
    fn binding(&self) -> BindingRef<'_, 's> { BindingRef::Produced(self) }
}
pub(super) enum CoordinateOwner<'a, 's> {
    Imported(&'a OriginalImportedView<'s>),
    Produced(&'a OriginalProducedView<'s>),
}
pub(in crate::replicated_decoder::program) struct OrderedCoordinate<'a, 's> {
    pub(super) owner: CoordinateOwner<'a, 's>,
    pub(super) lane: usize,
}
impl OrderedCoordinate<'_, '_> {
    pub(in crate::replicated_decoder::program) fn same(&self, other: &Self) -> bool {
        let same = match (&self.owner, &other.owner) {
            (CoordinateOwner::Imported(a), CoordinateOwner::Imported(b)) => a.same_declaration(b),
            (CoordinateOwner::Produced(a), CoordinateOwner::Produced(b)) => a.same_producer(b),
            _ => false,
        };
        same && self.lane == other.lane
    }
    pub(in crate::replicated_decoder::program) fn lane(&self) -> usize { self.lane }
}

impl BindingRef<'_, '_> {
    pub(super) fn check_identity(
        &self, builder: &Builder, source: &Arc<PublicSource>, operation: &ModelOperation,
    ) -> Result<(), Error> {
        let root = self.root();
        root.check(builder, root.registry(), root.plan(), root.graph())?;
        if self.depth() >= 64 || !builder.public_weights
            || !root.graph().operations.iter().any(|op| std::ptr::eq(op, operation))
            || operation.inputs.first().map(String::as_str) != Some(self.semantic()) {
            return Err(Error::Binding);
        }
        if let Self::Produced(input) = self {
            if !Arc::ptr_eq(input.source(), source) { return Err(Error::Binding); }
            if !input.canonical_geometry()
                || (0..input.geometry().lanes()).any(|index| match input.coordinate(index) {
                    Ok(coordinate) => coordinate.lane() != index,
                    Err(_) => true,
                }) { return Err(Error::Shape); }
        }
        source.check_registry(root.registry())
    }
}

/// Checked Q18 projection declaration. Never a raw FP8/BF16/scale contract.
pub(super) struct ProjectionBinding<'b, 's> {
    pub(super) input: BindingRef<'b, 's>,
    pub(super) shape: Geometry,
    pub(super) weight: &'s SourceTensor,
    pub(super) inner: usize,
    pub(super) columns: usize,
}
impl<'b, 's> ProjectionBinding<'b, 's> {
    pub(super) fn check(
        builder: &Builder, input: &'b impl OriginalInputBinding<'s>,
        source: &'b Arc<PublicSource>, operation: &'s ModelOperation,
    ) -> Result<Self, Error> {
        let input = input.binding();
        input.check_identity(builder, source, operation)?;
        if !matches!(operation.operator, Op::Linear | Op::OutputHead)
            || operation.attributes.get("bias").is_some_and(|v| !v.is_null()) {
            return Err(Error::Binding);
        }
        let key = operation.attributes.get("weight").and_then(|v| v.as_str()).ok_or(Error::Binding)?;
        let weight = input.root().registry().source_records().get(key).ok_or(Error::Binding)?;
        if weight.row_squared_norm.is_some() { return Err(Error::Binding); }
        let [columns, inner] = weight.shape.as_slice() else { return Err(Error::Shape); };
        let prior = input.geometry()?;
        let shape = Geometry::operation(operation)?;
        if prior.width() != *inner || shape.width() != *columns
            || (prior.lanes() / inner).checked_mul(*columns) != Some(shape.lanes()) {
            return Err(Error::Shape);
        }
        source.file(key)?;
        Ok(Self { input, shape, weight, inner: *inner, columns: *columns })
    }
}

// Storage custody only. Implementing a reader never establishes a coefficient
// format, source provenance, authenticated tensor ID, or input declaration.
mod storage_sealed { pub trait Sealed {} }
trait WindowRead: storage_sealed::Sealed {
    fn len(&self) -> usize;
    fn retained_bytes(&self) -> Result<usize, Error>;
    fn read_exact(&self, offset: usize, dst: &mut [u8]) -> Result<(), Error>;
}
impl storage_sealed::Sealed for pllm_core::public_file::PublicFile {}
impl WindowRead for pllm_core::public_file::PublicFile {
    fn len(&self) -> usize { usize::try_from(self.size()).unwrap_or(usize::MAX) }
    fn retained_bytes(&self) -> Result<usize, Error> {
        pllm_core::public_file::backing_bytes().map_err(|_| Error::Bounds)
    }
    fn read_exact(&self, offset: usize, dst: &mut [u8]) -> Result<(), Error> {
        self.read(u64::try_from(offset).map_err(|_| Error::Bounds)?, dst)
            .map_err(|_| Error::Binding)
    }
}

use pllm_core::field_float::{self as clear, window_linear::{
    BandScratch, CompletedRows, LinearSession, QuantizedInput, SourceWindow,
}, Block128Geometry};

#[derive(Clone, Copy)]
enum ClearFormat { E4m3fnBlock128, Bf16 }
/// Construct opaque core geometry and original dimensions TOGETHER from one
/// checked source shape, never from a partition or a caller's later label.
#[derive(Clone, Copy)]
struct ClearOriginal { geometry: Block128Geometry, rows: usize, columns: usize }
impl ClearOriginal {
    fn from_shape(shape: &[usize]) -> Result<Self, Error> {
        let [rows, columns] = shape else { return Err(Error::Shape); };
        let geometry = Block128Geometry::new(*rows, *columns).map_err(clear_error)?;
        product(*rows, *columns)?;
        Ok(Self { geometry, rows: *rows, columns: *columns })
    }
}
/// No production constructor here. D55 raw registration/custody is a separate
/// gate; PublicSource's i64-LE Q18 files cannot construct this owner.
struct ClearSource<S> {
    tensor_id: u64,
    scale_id: Option<u64>,
    original: ClearOriginal,
    format: ClearFormat,
    // Immutable ORIGINAL row-major weights; FP8 scale axes are exactly
    // [original.rows / 128, original.columns / 128], never window-local tiles.
    weights: S,
    scales: Option<S>,
}

/// Scalar visits and native resident payload only; not OS/allocator accounting.
/// Work is cumulative and never refunded, including failures. Resident admission
/// is held until cleanup (and, on success, until immutable outputs are released).
struct ClearLedger {
    work_limit: usize,
    resident_limit: usize,
    work: Cell<usize>,
    resident: Cell<usize>,
    #[cfg(test)]
    pending_scratch: Cell<bool>,
    #[cfg(test)]
    pending_read: Cell<bool>,
    #[cfg(test)]
    release_checks: Cell<usize>,
    #[cfg(test)]
    fail_read_allocation: Cell<Option<usize>>,
}
impl ClearLedger {
    // Local limits are fixture accounting, never original-source authority.
    // Production admission requires a typed reservation from the raw owner;
    // copied original bounds must not instantiate a replacement work ledger.
    #[cfg(test)]
    fn new(work_limit: usize, resident_limit: usize) -> Self {
        Self { work_limit, resident_limit, work: Cell::new(0), resident: Cell::new(0),
            #[cfg(test)] pending_scratch: Cell::new(false),
            #[cfg(test)] pending_read: Cell::new(false),
            #[cfg(test)] release_checks: Cell::new(0),
            #[cfg(test)] fail_read_allocation: Cell::new(None), }
    }
    fn reserve(&self, work: usize, bytes: usize) -> Result<ClearReservation<'_>, Error> {
        let total_work = plus(self.work.get(), work)?;
        let total_bytes = plus(self.resident.get(), bytes)?;
        if total_work > self.work_limit || total_bytes > self.resident_limit {
            return Err(Error::Bounds);
        }
        self.work.set(total_work);
        self.resident.set(total_bytes);
        Ok(ClearReservation { ledger: self, bytes })
    }
}
struct ClearReservation<'l> { ledger: &'l ClearLedger, bytes: usize }
impl Drop for ClearReservation<'_> {
    fn drop(&mut self) {
        #[cfg(test)] {
            assert!(!self.ledger.pending_scratch.get());
            assert!(!self.ledger.pending_read.get());
            self.ledger.release_checks.set(self.ledger.release_checks.get() + 1);
        }
        self.ledger.resident.set(self.ledger.resident.get() - self.bytes);
    }
}

/// Supplied clear BF16 input only. Never implements OriginalInputBinding, even
/// when supplied by another clear output. QuantizedInput clears before release.
struct ClearInput<'a, 'l, 's, S> {
    quantized: QuantizedInput<'a>,
    source: &'s ClearSource<S>,
    original: ClearOriginal,
    _reservation: ClearReservation<'l>,
}
impl<'a, 'l, 's, S: WindowRead> ClearInput<'a, 'l, 's, S> {
    fn quote(source: &ClearSource<S>, input: &[u16], scratch: &[u32]) -> Result<usize, Error> {
        // Borrowed slices charge their entire exposed retention. Any enclosing
        // Vec's unexposed spare capacity remains the caller's outside-owner
        // obligation, not an alias credit or a physical allocator certificate.
        let mut bytes = source.retained_bytes()?;
        for n in [array::<u16>(input.len())?, array::<u32>(scratch.len())?,
            size_of::<Self>(), size_of::<Result<Self, Error>>(),
            size_of::<Result<QuantizedInput<'_>, clear::Error>>(),
            size_of::<ClearReservation<'_>>(), size_of::<Result<ClearReservation<'_>, Error>>(),
            size_of::<ClearOriginal>(), size_of::<ClearLedger>()] { bytes = plus(bytes, n)?; }
        Ok(bytes)
    }
    fn new(source: &'s ClearSource<S>, input: &[u16], scratch: &'a mut [u32], ledger: &'l ClearLedger)
        -> Result<Self, Error> {
        let original = source.original;
        if input.len() != original.columns || scratch.len() < original.columns { return Err(Error::Shape); }
        let work = product(original.columns, 3)?;
        let reservation = ledger.reserve(work, Self::quote(source, input, scratch)?)?;
        let quantized = QuantizedInput::new(original.geometry, input, scratch, work).map_err(clear_error)?;
        Ok(Self { quantized, source, original, _reservation: reservation })
    }
}

struct ClearBuffers<'a> {
    band_input: &'a mut [u32],
    band_sums: &'a mut [u32],
    coefficients: &'a mut [i64],
    session_sums: &'a mut [u32],
    field_output: &'a mut [u32],
    bf16_output: &'a mut [u16],
}
struct ClearScratch<'a, 'l> {
    band_input: &'a mut [u32], band_sums: &'a mut [u32],
    coefficients: &'a mut [i64],
    ledger: &'l ClearLedger,
}
impl Drop for ClearScratch<'_, '_> {
    fn drop(&mut self) {
        self.band_input.fill(0); self.band_sums.fill(0);
        self.coefficients.fill(0);
        #[cfg(test)] self.ledger.pending_scratch.set(false);
    }
}
struct ReadScratch<'l> { old: Vec<u8>, next: Vec<u8>, words: Vec<u16>, scales: Vec<u16>, ledger: &'l ClearLedger }
impl Drop for ReadScratch<'_> {
    fn drop(&mut self) {
        self.old.fill(0); self.next.fill(0); self.words.fill(0); self.scales.fill(0);
        #[cfg(test)] self.ledger.pending_read.set(false);
    }
}

/// Once-consuming binding of actual source/input references and immutable full
/// ordered partition. No ID/content-equivalence substitution; IDs are metadata.
/// No Clone/replay/attach operation. Output lifetime is buffers', not source's.
struct ClearProjection<'b, 'i, 'l, 's, S> {
    source: &'b ClearSource<S>,
    input: &'b ClearInput<'i, 'l, 's, S>,
    plan: &'b Vec<SourceWindow>,
    max_lanes: usize,
    max_scales: usize,
    provider_work: usize,
    _binding: ClearReservation<'l>,
}
struct ClearProduced<'a, 'l> {
    rows: CompletedRows<'a>,
    _reservation: ClearReservation<'l>,
}
impl ClearProduced<'_, '_> {
    fn field(&self) -> &[u32] { self.rows.field }
    fn bf16(&self) -> &[u16] { self.rows.bf16 }
}

fn clear_error(_: clear::Error) -> Error { Error::Binding }
fn product(a: usize, b: usize) -> Result<usize, Error> { a.checked_mul(b).ok_or(Error::Bounds) }
fn scale_shape(window: SourceWindow) -> (usize, usize) {
    ((window.row_start + window.rows - 1) / 128 - window.row_start / 128 + 1,
     (window.column_start + window.columns - 1) / 128 - window.column_start / 128 + 1)
}

impl<S: WindowRead> ClearSource<S> {
    fn retained_bytes(&self) -> Result<usize, Error> {
        // Reader inline storage lives in Self; provider quotes only its backing.
        plus(size_of::<Self>(), plus(self.weights.retained_bytes()?, match &self.scales {
            Some(scales) => scales.retained_bytes()?, None => 0,
        })?)
    }
}
impl<'b, 'i, 'l, 's, S: WindowRead> ClearProjection<'b, 'i, 'l, 's, S> {
    fn bind(source: &'b ClearSource<S>, input: &'b ClearInput<'i, 'l, 's, S>,
        plan: &'b Vec<SourceWindow>) -> Result<Self, Error> {
        if !std::ptr::eq(source, input.source) { return Err(Error::Binding); }
        // Admit retained source/plan capacities and validation controls BEFORE
        // scanning metadata. These immutable references cannot be replaced.
        // No alias credit against the prepared owner's source retention.
        let mut controls = plus(source.retained_bytes()?, plus(
            array::<SourceWindow>(plan.capacity())?, plus(size_of::<Vec<SourceWindow>>(),
                 plus(size_of::<Self>(), size_of::<Result<Self, Error>>())?)?)?)?;
        for n in [size_of::<Result<Self, Error>>(), size_of::<ClearReservation<'_>>(),
            size_of::<Result<ClearReservation<'_>, Error>>(), size_of::<ClearOriginal>()] {
            controls = plus(controls, n)?;
        }
        let binding = input._reservation.ledger.reserve(plus(product(plan.len(), 16)?, 16)?, controls)?;
        let geometry = input.original;
        // Overflow and structural validation precede reads, factories, or scratch.
        let lanes = product(geometry.rows, geometry.columns)?;
        let scale_rows = geometry.rows / 128;
        let scale_columns = geometry.columns / 128;
        let expected = match source.format { ClearFormat::Bf16 => product(lanes, 2)?, _ => lanes };
        if source.weights.len() != expected { return Err(Error::Shape); }
        match source.format {
            ClearFormat::E4m3fnBlock128 => {
                if source.scale_id.is_none() || source.scales.as_ref().map(WindowRead::len)
                    != Some(product(product(scale_rows, scale_columns)?, 2)?) { return Err(Error::Shape); }
            }
            ClearFormat::Bf16 if source.scales.is_some() || source.scale_id.is_some() => return Err(Error::Binding),
            _ => {}
        }
        let mut next_row = 0;
        let mut next_column = 0;
        let mut band_rows = 0;
        let mut max_lanes = 0;
        let mut max_scales = 0;
        // Explicit division: core's 16RC+4C+8R covers numeric validation,
        // conversion, multiply, band/session copying and core cleanup. Additional
        // provider budget covers plan/metadata validation, bounded reads, byte
        // copies, u16 decoding, and clearing BOTH old/new read windows. No second
        // charge for core import_q8_window's hidden validation/conversion visits.
        let mut provider_work = 1;
        for &w in plan {
            let row_end = plus(w.row_start, w.rows)?;
            let col_end = plus(w.column_start, w.columns)?;
            if w.rows == 0 || w.columns == 0 || w.row_start != next_row
                || w.column_start != next_column || row_end > geometry.rows || col_end > geometry.columns {
                return Err(Error::Shape);
            }
            if next_column == 0 { band_rows = w.rows; }
            if w.rows != band_rows { return Err(Error::Shape); }
            let length = product(w.rows, w.columns)?;
            let (sr, sc) = scale_shape(w);
            let scales = if matches!(source.format, ClearFormat::E4m3fnBlock128) { product(sr, sc)? } else { 0 };
            max_lanes = max_lanes.max(length); max_scales = max_scales.max(scales);
            provider_work = plus(provider_work, plus(product(length, 12)?, product(scales, 12)?)?)?;
            next_column = col_end;
            if next_column == geometry.columns { next_row = row_end; next_column = 0; }
        }
        if next_row != geometry.rows || next_column != 0 { return Err(Error::Shape); }
        // Covers allocation initialization and final spare-window clear visits,
        // even when last window is smaller than previous window.
        let window_cleanup = plus(product(max_lanes, 12)?, product(max_scales, 4)?)?;
        provider_work = plus(provider_work, product(plus(plan.len(), 1)?, window_cleanup)?)?;
        Ok(Self { source, input, plan, max_lanes, max_scales, provider_work, _binding: binding })
    }

    fn quote(&self, buffers: &ClearBuffers<'_>) -> Result<(usize, usize), Error> {
        let core = LinearSession::required_work(self.source.original.geometry).map_err(clear_error)?;
        // Core cleanup is inside its unchanged bound. Our wrapper additionally
        // clears entire supplied scratch slices (including spare entries):
        // session sums before allocation, other scratch through its drop guard.
        let wrapper_cleanup = plus(plus(buffers.band_input.len(), buffers.band_sums.len())?,
            plus(buffers.coefficients.len(), buffers.session_sums.len())?)?;
        let work = plus(core, plus(self.provider_work, wrapper_cleanup)?)?;
        let mut bytes = self.source.retained_bytes()?;
        for n in [
            size_of::<Vec<SourceWindow>>(), array::<SourceWindow>(self.plan.capacity())?,
            array::<u32>(buffers.band_input.len())?, array::<u32>(buffers.band_sums.len())?,
            array::<i64>(buffers.coefficients.len())?, array::<u32>(buffers.session_sums.len())?,
            array::<u32>(buffers.field_output.len())?, array::<u16>(buffers.bf16_output.len())?,
            // Both actual retained read windows; no old/new alias credit.
            product(self.max_lanes, 4)?, array::<u16>(self.max_lanes)?, array::<u16>(self.max_scales)?,
            size_of::<Self>(), size_of::<ClearBuffers<'_>>(), size_of::<ClearScratch<'_, '_>>(),
            size_of::<ReadScratch<'_>>(), size_of::<LinearSession<'_, '_>>(),
            size_of::<clear::window_linear::WindowLinear<'_>>(),
            size_of::<Result<clear::window_linear::WindowLinear<'_>, clear::Error>>(),
            size_of::<BandScratch<'_>>(), size_of::<Result<(), clear::Error>>(),
            size_of::<clear::window_linear::SessionBand<'_, '_, '_>>(),
            size_of::<Result<clear::window_linear::SessionBand<'_, '_, '_>, clear::Error>>(),
            size_of::<Result<LinearSession<'_, '_>, clear::Error>>(),
            size_of::<ClearProduced<'_, '_>>(), size_of::<Result<ClearProduced<'_, '_>, Error>>(),
            size_of::<Result<CompletedRows<'_>, clear::Error>>(), size_of::<ClearReservation<'_>>(),
            size_of::<Result<ClearReservation<'_>, Error>>(),
            size_of::<Result<(), Error>>(), // allocation result
            size_of::<Result<(), std::collections::TryReserveError>>(),
            size_of::<ClearLedger>(), size_of::<SourceWindow>(), size_of::<ClearOriginal>(),
            array::<usize>(16)?, // ordered driver offsets, counts, and loop controls
            size_of::<[u8; 2]>(), size_of::<Result<(), Error>>(),
            size_of::<(usize, usize)>(), size_of::<Result<(usize, usize), Error>>(),
        ] { bytes = plus(bytes, n)?; }
        Ok((work, bytes))
    }

    fn run<'a>(self, source: &ClearSource<S>, input: &ClearInput<'_, '_, '_, S>,
        buffers: ClearBuffers<'a>, ledger: &'l ClearLedger) -> Result<ClearProduced<'a, 'l>, Error> {
        // Check actual prepared instance BEFORE any provider read, not labels or
        // equal words. The core also checks exactly this prepared object per band.
        if !std::ptr::eq(self.source, source) || !std::ptr::eq(self.input, input) { return Err(Error::Binding); }
        if !std::ptr::eq(ledger, self.input._reservation.ledger) { return Err(Error::Binding); }
        let geometry = self.input.original;
        if buffers.band_input.len() < geometry.columns || buffers.session_sums.len() < geometry.rows
            || buffers.field_output.len() < geometry.rows || buffers.bf16_output.len() < geometry.rows
            || buffers.band_sums.len() < self.plan.iter().map(|w| w.rows).max().unwrap_or(0)
            || buffers.coefficients.len() < self.max_lanes { return Err(Error::Shape); }
        let (work, bytes) = self.quote(&buffers)?;
        // Reservation declared BEFORE scratch: reverse local drop order clears
        // source windows and external active scratch before releasing reservation.
        let reservation = ledger.reserve(work, bytes)?;
        // Admission precedes every clear. Clear even spare session entries before
        // any fallible allocation; thereafter core owns active-sum cleanup.
        // Do not put this slice in ClearScratch: core ties its borrow to outputs,
        // so that guard cannot own/reborrow it and then drop before publication.
        buffers.session_sums.fill(0);
        #[cfg(test)] ledger.pending_scratch.set(true);
        let scratch = ClearScratch { band_input: buffers.band_input, band_sums: buffers.band_sums,
            coefficients: buffers.coefficients, ledger };
        #[cfg(test)] ledger.pending_read.set(true);
        let mut read = ReadScratch { old: Vec::new(), next: Vec::new(), words: Vec::new(), scales: Vec::new(), ledger };
        let read_len = product(self.max_lanes, 2)?;
        fn allocate<T: Default + Clone>(out: &mut Vec<T>, n: usize,
            _ledger: &ClearLedger, _ordinal: usize) -> Result<(), Error> {
            #[cfg(test)]
            if _ledger.fail_read_allocation.get() == Some(_ordinal) { return Err(Error::Allocation); }
            out.try_reserve_exact(n).map_err(|_| Error::Allocation)?;
            if out.capacity() != n { return Err(Error::Allocation); }
            out.resize(n, T::default()); Ok(())
        }
        allocate(&mut read.old, read_len, ledger, 0)?; allocate(&mut read.next, read_len, ledger, 1)?;
        allocate(&mut read.words, self.max_lanes, ledger, 2)?; allocate(&mut read.scales, self.max_scales, ledger, 3)?;
        let mut session = LinearSession::new(&self.input.quantized, buffers.session_sums,
            buffers.field_output, buffers.bf16_output,
            LinearSession::required_work(geometry.geometry).map_err(clear_error)?).map_err(clear_error)?;
        let mut index = 0;
        while index < self.plan.len() {
            let first = self.plan[index];
            let mut band = session.band(geometry.geometry, &self.input.quantized, first.row_start, first.rows,
                BandScratch { input: &mut *scratch.band_input, sums: &mut *scratch.band_sums,
                    coefficients: &mut *scratch.coefficients }).map_err(clear_error)?;
            loop {
                let w = self.plan[index];
                let length = product(w.rows, w.columns)?;
                let width = if matches!(self.source.format, ClearFormat::Bf16) { 2 } else { 1 };
                // Bounded row reads preserve ORIGINAL global row/column offsets.
                for row in 0..w.rows {
                    let offset = product(plus(product(w.row_start + row, geometry.columns)?, w.column_start)?, width)?;
                    let start = product(product(row, w.columns)?, width)?;
                    read_bounded(&self.source.weights, offset, &mut read.next[start..start + w.columns * width])?;
                }
                match self.source.format {
                    ClearFormat::Bf16 => {
                        for (out, pair) in read.words[..length].iter_mut().zip(read.next[..2 * length].chunks_exact(2)) {
                            *out = u16::from_le_bytes([pair[0], pair[1]]);
                        }
                        band = band.append_bf16(w, &read.words[..length]).map_err(clear_error)?;
                    }
                    ClearFormat::E4m3fnBlock128 => {
                        let (sr, sc) = scale_shape(w);
                        let scales = self.source.scales.as_ref().ok_or(Error::Binding)?;
                        // Reuse old window only AFTER zeroing it. Its allocation
                        // remains retained alongside next throughout execution.
                        read.old.fill(0);
                        for row in 0..sr {
                            let offset = product(plus(product(w.row_start / 128 + row, geometry.columns / 128)?, w.column_start / 128)?, 2)?;
                            // Scale rectangle can exceed a narrow weight window;
                            // decode one cell at a time, no unquoted temporary Vec.
                            for column in 0..sc {
                                let mut pair = [0u8; 2];
                                read_bounded(scales, plus(offset, product(column, 2)?)?, &mut pair)?;
                                read.scales[row * sc + column] = u16::from_le_bytes(pair);
                                pair.fill(0);
                            }
                        }
                        band = band.append_fp8(w, &read.next[..length], &read.scales[..sr * sc]).map_err(clear_error)?;
                    }
                }
                read.old.fill(0); read.words.fill(0); read.scales.fill(0);
                std::mem::swap(&mut read.old, &mut read.next);
                index += 1;
                if w.column_start + w.columns == geometry.columns { break; }
            }
            session = band.finish().map_err(clear_error)?;
        }
        let rows = session.finish().map_err(clear_error)?;
        drop(read); drop(scratch);
        Ok(ClearProduced { rows, _reservation: reservation })
    }
}
fn read_bounded(source: &impl WindowRead, mut offset: usize, dst: &mut [u8]) -> Result<(), Error> {
    for chunk in dst.chunks_mut(pllm_core::public_file::MAX_CHUNK_BYTES) {
        source.read_exact(offset, chunk)?;
        offset = plus(offset, chunk.len())?;
    }
    Ok(())
}

#[cfg(test)]
struct FixtureBytes { bytes: Vec<u8>, reads: Cell<usize>, fail_at: Cell<Option<usize>> }
#[cfg(test)]
impl storage_sealed::Sealed for FixtureBytes {}
#[cfg(test)]
impl WindowRead for FixtureBytes {
    fn len(&self) -> usize { self.bytes.len() }
    fn retained_bytes(&self) -> Result<usize, Error> { Ok(self.bytes.capacity()) }
    fn read_exact(&self, offset: usize, dst: &mut [u8]) -> Result<(), Error> {
        let count = self.reads.get(); self.reads.set(plus(count, 1)?);
        if self.fail_at.get() == Some(count) { return Err(Error::Binding); }
        let end = plus(offset, dst.len())?;
        dst.copy_from_slice(self.bytes.get(offset..end).ok_or(Error::Bounds)?); Ok(())
    }
}
#[cfg(test)]
type FixtureProjectionSource = ClearSource<FixtureBytes>;
#[cfg(test)]
impl FixtureProjectionSource {
    /// UNAUTHENTICATED supplied fixture storage. IDs never authorize production.
    fn fp8(ids: (u64, u64), shape: [usize; 2], raw: &[u8], scales: &[u16]) -> Result<Self, Error> {
        let original = ClearOriginal::from_shape(&shape)?;
        if raw.len() != product(original.rows, original.columns)?
            || scales.len() != product(original.rows / 128, original.columns / 128)? { return Err(Error::Shape); }
        Ok(Self { tensor_id: ids.0, scale_id: Some(ids.1), original, format: ClearFormat::E4m3fnBlock128,
            weights: FixtureBytes::new(raw.to_vec()),
            scales: Some(FixtureBytes::new(scales.iter().flat_map(|x| x.to_le_bytes()).collect())) })
    }
    fn bf16(id: u64, shape: [usize; 2], words: &[u16]) -> Result<Self, Error> {
        let original = ClearOriginal::from_shape(&shape)?;
        if words.len() != product(original.rows, original.columns)? { return Err(Error::Shape); }
        Ok(Self { tensor_id: id, scale_id: None, original, format: ClearFormat::Bf16,
            weights: FixtureBytes::new(words.iter().flat_map(|x| x.to_le_bytes()).collect()), scales: None })
    }
}
#[cfg(test)]
impl FixtureBytes {
    fn new(bytes: Vec<u8>) -> Self { Self { bytes, reads: Cell::new(0), fail_at: Cell::new(None) } }
}
#[cfg(test)]
#[path = "newclear_projection_tests.rs"]
mod clear_projection_tests;
