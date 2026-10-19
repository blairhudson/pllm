//! Bounded affine tensor layouts. Concatenation retains owners, never values.
use super::*;

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
    pub fn contiguous(owner: usize, len: usize) -> Result<Self, Error> {
        Self::affine(owner, 0, &[len], &[1])
    }
    pub fn affine(
        owner: usize,
        offset: usize,
        shape: &[usize],
        strides: &[usize],
    ) -> Result<Self, Error> {
        if shape.is_empty() || shape.len() > 4 || shape.len() != strides.len() {
            return Err(Error::Shape);
        }
        let mut span = Span {
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
        Ok(Self(Arc::from([span])))
    }
    pub fn concat(parts: &[Self]) -> Result<Self, Error> {
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
        let mut spans = Vec::new();
        spans
            .try_reserve_exact(count)
            .map_err(|_| Error::Allocation)?;
        for part in parts {
            spans.extend(part.0.iter().cloned());
        }
        Ok(Self(spans.into()))
    }
    pub fn slice(&self, start: usize, len: usize) -> Result<Self, Error> {
        let end = start
            .checked_add(len)
            .filter(|&end| end <= self.len())
            .ok_or(Error::Shape)?;
        if len == 0 {
            return Err(Error::Shape);
        }
        let mut spans = Vec::new();
        let mut offset = 0;
        for span in self.0.iter() {
            let lo = start.max(offset);
            let hi = end.min(offset + span.len);
            if lo < hi {
                let mut span = span.clone();
                span.start += lo - offset;
                span.len = hi - lo;
                spans.try_reserve(1).map_err(|_| Error::Allocation)?;
                spans.push(span);
            }
            offset += span.len;
        }
        Ok(Self(spans.into()))
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
        2 * std::mem::size_of::<usize>() + self.0.len() * std::mem::size_of::<Span>()
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
