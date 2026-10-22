//! Bounded affine tensor layouts. Concatenation retains owners, never values.
use super::*;

/// Layout-owned backing only, not a whole-Builder or whole-request admission.
/// Counts Arc headers/span payloads and explicit construction Vec backing.
/// Caller containers, values, sources, transport and allocator bookkeeping need
/// their own owner quotes. Shared input Arcs are conservatively counted per use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayoutQuote {
    pub retained_backing_bytes: usize,
    pub destination_backing_bytes: usize,
    pub span_buffer_bytes: usize,
    pub construction_peak_bytes: usize,
}

impl LayoutQuote {
    fn new(retained: usize, spans: usize, capacity: usize) -> Result<Self, Error> {
        if spans == 0 || spans > MAX_NODES || (capacity != 0 && capacity < spans) {
            return Err(Error::Bounds);
        }
        let destination_backing_bytes = spans
            .checked_mul(size_of::<Span>())
            .and_then(|n| n.checked_add(2 * size_of::<usize>()))
            .ok_or(Error::Bounds)?;
        let span_buffer_bytes = capacity
            .checked_mul(size_of::<Span>())
            .ok_or(Error::Bounds)?;
        let construction_peak_bytes = retained
            .checked_add(destination_backing_bytes)
            .and_then(|n| n.checked_add(span_buffer_bytes))
            .ok_or(Error::Bounds)?;
        Ok(Self {
            retained_backing_bytes: retained,
            destination_backing_bytes,
            span_buffer_bytes,
            construction_peak_bytes,
        })
    }
    fn admit(self, limit: usize) -> Result<(), Error> {
        if self.construction_peak_bytes > limit.min(MAX_PROGRAM_BYTES) {
            Err(Error::Bounds)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
thread_local! {
    static ALLOCATION_ATTEMPTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn observe_allocation() {
    ALLOCATION_ATTEMPTS.with(|n| n.set(n.get() + 1));
}

#[derive(Clone)]
struct Span {
    owner: usize,
    offset: usize,
    shape: [usize; 4],
    strides: [usize; 4],
    rank: usize,
    start: usize,
    len: usize,
}

#[derive(Clone)]
pub struct Layout(Arc<[Span]>);

impl Layout {
    /// Validate geometry and quote the direct one-span Arc before allocation.
    pub fn quote_contiguous(owner: usize, len: usize) -> Result<LayoutQuote, Error> {
        Self::quote_affine(owner, 0, &[len], &[1])
    }
    pub fn quote_affine(
        owner: usize,
        offset: usize,
        shape: &[usize],
        strides: &[usize],
    ) -> Result<LayoutQuote, Error> {
        Span::affine(owner, offset, shape, strides)?;
        LayoutQuote::new(0, 1, 0)
    }
    pub fn contiguous(owner: usize, len: usize) -> Result<Self, Error> {
        Self::affine(owner, 0, &[len], &[1])
    }
    pub fn affine(
        owner: usize,
        offset: usize,
        shape: &[usize],
        strides: &[usize],
    ) -> Result<Self, Error> {
        Self::affine_admitted(owner, offset, shape, strides, MAX_PROGRAM_BYTES)
    }
    fn affine_admitted(
        owner: usize,
        offset: usize,
        shape: &[usize],
        strides: &[usize],
        limit: usize,
    ) -> Result<Self, Error> {
        let span = Span::affine(owner, offset, shape, strides)?;
        LayoutQuote::new(0, 1, 0)?.admit(limit)?;
        #[cfg(test)]
        observe_allocation();
        // Allocate Arc<[Span; 1]> directly, then unsize. No hidden Vec/Box owner.
        let spans: Arc<[Span]> = Arc::new([span]);
        Ok(Self(spans))
    }
    pub fn quote_concat(parts: &[Self]) -> Result<LayoutQuote, Error> {
        let count = parts
            .iter()
            .try_fold(0usize, |n, p| n.checked_add(p.0.len()).ok_or(Error::Bounds))?;
        if count == 0 || count > MAX_NODES {
            return Err(Error::Bounds);
        }
        let len = parts.iter().try_fold(0usize, |n, p| {
            n.checked_add(p.len())
                .filter(|&n| n <= MAX_VALUES)
                .ok_or(Error::Bounds)
        })?;
        if len == 0 {
            return Err(Error::Shape);
        }
        let retained = parts.iter().try_fold(0usize, |n, p| {
            n.checked_add(p.backing_bytes()?).ok_or(Error::Bounds)
        })?;
        LayoutQuote::new(retained, count, count)
    }
    pub fn concat(parts: &[Self]) -> Result<Self, Error> {
        Self::concat_admitted(parts, MAX_PROGRAM_BYTES)
    }
    fn concat_admitted(parts: &[Self], limit: usize) -> Result<Self, Error> {
        let quote = Self::quote_concat(parts)?;
        quote.admit(limit)?;
        let count = quote.span_buffer_bytes / size_of::<Span>();
        let mut spans = Vec::new();
        #[cfg(test)]
        observe_allocation();
        spans
            .try_reserve_exact(count)
            .map_err(|_| Error::Allocation)?;
        Self::check_span_buffer(&spans, count, quote, limit)?;
        for part in parts {
            spans.extend(part.0.iter().cloned());
        }
        #[cfg(test)]
        observe_allocation();
        Ok(Self(spans.into()))
    }
    pub fn quote_slice(&self, start: usize, len: usize) -> Result<LayoutQuote, Error> {
        let end = start
            .checked_add(len)
            .filter(|&end| end <= self.len())
            .ok_or(Error::Shape)?;
        if len == 0 {
            return Err(Error::Shape);
        }
        let mut offset = 0usize;
        let mut count = 0usize;
        for span in self.0.iter() {
            let next = offset.checked_add(span.len).ok_or(Error::Bounds)?;
            if start.max(offset) < end.min(next) {
                count = count.checked_add(1).ok_or(Error::Bounds)?;
            }
            offset = next;
        }
        LayoutQuote::new(self.backing_bytes()?, count, count)
    }
    pub fn slice(&self, start: usize, len: usize) -> Result<Self, Error> {
        self.slice_admitted(start, len, MAX_PROGRAM_BYTES)
    }
    fn slice_admitted(&self, start: usize, len: usize, limit: usize) -> Result<Self, Error> {
        let quote = self.quote_slice(start, len)?;
        quote.admit(limit)?;
        let end = start.checked_add(len).ok_or(Error::Bounds)?;
        let count = quote.span_buffer_bytes / size_of::<Span>();
        let mut spans = Vec::new();
        #[cfg(test)]
        observe_allocation();
        spans
            .try_reserve_exact(count)
            .map_err(|_| Error::Allocation)?;
        Self::check_span_buffer(&spans, count, quote, limit)?;
        let mut offset = 0;
        for span in self.0.iter() {
            let lo = start.max(offset);
            let hi = end.min(offset + span.len);
            if lo < hi {
                let mut span = span.clone();
                span.start += lo - offset;
                span.len = hi - lo;
                spans.push(span);
            }
            offset += span.len;
        }
        #[cfg(test)]
        observe_allocation();
        Ok(Self(spans.into()))
    }
    fn check_span_buffer(
        spans: &Vec<Span>,
        required: usize,
        quote: LayoutQuote,
        limit: usize,
    ) -> Result<(), Error> {
        // Quote actual reported Vec capacity; never silently refund spare backing.
        let actual = LayoutQuote::new(quote.retained_backing_bytes, required, spans.capacity())?;
        actual.admit(limit)?;
        if spans.capacity() != required || actual != quote {
            return Err(Error::Bounds);
        }
        Ok(())
    }
    pub fn backing_bytes(&self) -> Result<usize, Error> {
        Ok(LayoutQuote::new(0, self.0.len(), 0)?.destination_backing_bytes)
    }
    pub fn len(&self) -> usize {
        self.0.iter().map(|s| s.len).sum()
    }
    pub fn word(&self, mut index: usize) -> Result<(usize, usize), Error> {
        for span in self.0.iter() {
            if index < span.len {
                return Ok((span.owner, span.index(index)));
            }
            index -= span.len;
        }
        Err(Error::Shape)
    }
    pub(super) fn validate(&self, registers: &[Register]) -> Result<(), Error> {
        if self.len() == 0 || self.len() > MAX_VALUES {
            return Err(Error::Shape);
        }
        for span in self.0.iter() {
            if span.maximum()? >= registers.get(span.owner).ok_or(Error::Binding)?.lanes {
                return Err(Error::Shape);
            }
        }
        Ok(())
    }
    pub(super) fn owners(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter().map(|s| s.owner)
    }
    pub(super) fn metadata_bytes(&self) -> usize {
        self.backing_bytes()
            .expect("validated bounded layout backing")
    }
    pub(super) fn identity(&self) -> usize {
        Arc::as_ptr(&self.0) as *const Span as usize
    }
    pub(super) fn hash(&self, hash: &mut Sha256) {
        hash.update((self.0.len() as u64).to_le_bytes());
        for span in self.0.iter() {
            for n in [span.owner, span.offset, span.rank, span.start, span.len]
                .into_iter()
                .chain(span.shape)
                .chain(span.strides)
            {
                hash.update((n as u64).to_le_bytes());
            }
        }
    }
    /// Certificates apply only to exact complete contiguous rows. A strided or
    /// shortened alias cannot acquire a certificate just by sharing an owner.
    pub fn row_key(&self, width: usize) -> Option<RowKey> {
        if self.len() != width || width == 0 {
            return None;
        }
        let (owner, offset) = self.word(0).ok()?;
        for i in 1..width {
            if self.word(i).ok()? != (owner, offset.checked_add(i)?) {
                return None;
            }
        }
        Some(RowKey {
            owner,
            offset,
            width,
        })
    }
}
impl Span {
    fn affine(
        owner: usize,
        offset: usize,
        shape: &[usize],
        strides: &[usize],
    ) -> Result<Self, Error> {
        if shape.is_empty() || shape.len() > 4 || shape.len() != strides.len() {
            return Err(Error::Shape);
        }
        let mut span = Self {
            owner,
            offset,
            shape: [1; 4],
            strides: [0; 4],
            rank: shape.len(),
            start: 0,
            len: 1,
        };
        for (i, (&n, &stride)) in shape.iter().zip(strides).enumerate() {
            if n == 0 {
                return Err(Error::Shape);
            }
            span.len = span
                .len
                .checked_mul(n)
                .filter(|&n| n <= MAX_VALUES)
                .ok_or(Error::Bounds)?;
            span.shape[i] = n;
            span.strides[i] = stride;
        }
        span.maximum()?;
        Ok(span)
    }
    fn maximum(&self) -> Result<usize, Error> {
        (0..self.rank).try_fold(self.offset, |n, axis| {
            n.checked_add(
                (self.shape[axis] - 1)
                    .checked_mul(self.strides[axis])
                    .ok_or(Error::Bounds)?,
            )
            .ok_or(Error::Bounds)
        })
    }
    fn index(&self, i: usize) -> usize {
        let mut index = self.start + i;
        let mut offset = self.offset;
        for axis in (0..self.rank).rev() {
            offset += (index % self.shape[axis]) * self.strides[axis];
            index /= self.shape[axis];
        }
        offset
    }
}

#[cfg(test)]
#[path = "body_layout_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "body_input_admission_tests.rs"]
mod input_admission_tests;
