//! Local material-provider ownership, independent of model and transport wire.
//! Scalar replies retain their original one-write semantics. A window owns its
//! entire charge before factories/packaging and never exposes detachable replies.
use super::{Budget, Error, Lease, MAX_ALLOCATION_BYTES};
use std::mem::size_of;
use zeroize::Zeroizing;

pub const MATERIAL_WINDOW_RECORDS: usize = 8;
pub const MATERIAL_WINDOW_BYTES: usize = 64 << 10;
// Canonical fixed public request metadata, serializer/bridge staging. Payload,
// result and Python owners are separate. Not allocator/stack/RSS accounting.
const ADAPTER_SCRATCH_BYTES: usize = 32 << 10;

pub struct Reply {
    bytes: Zeroizing<Vec<u8>>,
    bound: usize,
    state: u8,
}
impl Reply {
    fn new(bound: usize) -> Result<Self, Error> {
        if bound == 0 || bound > MAX_ALLOCATION_BYTES {
            return Err(Error::Resource);
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(bound)
            .map_err(|_| Error::Allocation)?;
        if bytes.capacity() != bound {
            return Err(Error::Resource);
        }
        Ok(Self {
            bytes: Zeroizing::new(bytes),
            bound,
            state: 0,
        })
    }
    /// Scalar factory: the caller retains the original scalar transport quote.
    /// No owned Reply constructor/extractor is exposed across the crate boundary.
    pub fn with_bound<T>(
        bound: usize,
        use_reply: impl FnOnce(&mut Self) -> Result<T, Error>,
    ) -> Result<T, Error> {
        use_reply(&mut Self::new(bound)?)
    }
    pub fn bound(&self) -> usize {
        self.bound
    }
    pub fn write(&mut self, bytes: &[u8]) -> Result<(), Error> {
        let fresh = self.state == 0;
        self.state = 2;
        if !fresh {
            return Err(Error::Consumed);
        }
        if bytes.is_empty() || bytes.len() > self.bound {
            return Err(Error::Resource);
        }
        self.bytes.extend_from_slice(bytes);
        self.state = 1;
        Ok(())
    }
    pub fn bytes(&self) -> Result<&[u8], Error> {
        if self.state != 1 {
            return Err(Error::Consumed);
        }
        Ok(&self.bytes)
    }
}

/// A batch provider can fill but cannot swap out a charged Reply. In particular,
/// nesting the scalar factory cannot move a window's buffer past its lease.
///
/// ```compile_fail
/// use pllm_garble::fusefss_backend::{Budget, composition::provider::ReplyWindow};
/// let budget = Budget::new(ReplyWindow::<usize>::reservation_bytes().unwrap()).unwrap();
/// let mut window = ReplyWindow::new(&budget, 0usize).unwrap();
/// window.push(0, 8).unwrap();
/// let mut escaped = None;
/// window.provide(|_, replies| { escaped = Some(&mut replies[0]); Ok(()) }).unwrap();
/// drop(window);
/// escaped.unwrap().write(&[1]).unwrap();
/// ```
pub struct ReplyView<'a> {
    reply: &'a mut Reply,
}
impl ReplyView<'_> {
    pub fn bound(&self) -> usize {
        self.reply.bound()
    }
    pub fn write(&mut self, bytes: &[u8]) -> Result<(), Error> {
        self.reply.write(bytes)
    }
}

pub struct ReplyWindow<'a, R: Copy> {
    replies: Vec<Reply>,
    requests: [R; MATERIAL_WINDOW_RECORDS],
    pcs: [usize; MATERIAL_WINDOW_RECORDS],
    payload: usize,
    state: u8, // 0 filling; 1 attempted/burned; 2 complete, immutable replies.
    budget: &'a Budget,
    // Last: every payload/descriptor backing is dropped before final refund.
    _lease: Lease<'a>,
}
impl<'a, R: Copy> ReplyWindow<'a, R> {
    pub fn reservation_bytes() -> Result<usize, Error> {
        MATERIAL_WINDOW_BYTES
            .checked_add(ADAPTER_SCRATCH_BYTES)
            // Factory/result inline transition plus initial request and scalar
            // Reply factory staging; no extra payload backing is created.
            .and_then(|n| n.checked_add(2usize.checked_mul(size_of::<Self>())?))
            .and_then(|n| n.checked_add(size_of::<R>()))
            .and_then(|n| n.checked_add(size_of::<Reply>()))
            .and_then(|n| n.checked_add(size_of::<Vec<ReplyView<'_>>>() + size_of::<Vec<u8>>()))
            .and_then(|n| n.checked_add(size_of::<Budget>()))
            .and_then(|n| n.checked_add(MATERIAL_WINDOW_RECORDS.checked_mul(size_of::<Reply>())?))
            .and_then(|n| {
                n.checked_add(MATERIAL_WINDOW_RECORDS.checked_mul(size_of::<ReplyView<'_>>())?)
            })
            .ok_or(Error::Resource)
    }
    pub fn new(budget: &'a Budget, initial: R) -> Result<Self, Error> {
        let lease = budget.reserve(Self::reservation_bytes()?)?;
        let mut result = Self {
            replies: Vec::new(),
            requests: [initial; MATERIAL_WINDOW_RECORDS],
            pcs: [0; MATERIAL_WINDOW_RECORDS],
            payload: 0,
            state: 0,
            budget,
            _lease: lease,
        };
        budget.allocation()?;
        result
            .replies
            .try_reserve_exact(MATERIAL_WINDOW_RECORDS)
            .map_err(|_| Error::Allocation)?;
        if result.replies.capacity() != MATERIAL_WINDOW_RECORDS {
            return Err(Error::Resource);
        }
        Ok(result)
    }
    pub fn pcs_mut(&mut self) -> &mut [usize] {
        &mut self.pcs
    }
    pub fn template(&self) -> R {
        self.requests[0]
    }
    pub fn pc(&self, index: usize) -> Result<usize, Error> {
        self.pcs.get(index).copied().ok_or(Error::Context)
    }
    pub fn push(&mut self, request: R, bound: usize) -> Result<(), Error> {
        if self.state != 0 {
            self.state = 1;
            return Err(Error::Consumed);
        }
        let result = self.push_once(request, bound);
        if result.is_err() {
            self.state = 1;
        }
        result
    }
    fn push_once(&mut self, request: R, bound: usize) -> Result<(), Error> {
        let total = self.payload.checked_add(bound).ok_or(Error::Resource)?;
        if self.replies.len() == MATERIAL_WINDOW_RECORDS || total > MATERIAL_WINDOW_BYTES {
            return Err(Error::Resource);
        }
        self.budget.allocation()?;
        let reply = Reply::new(bound)?; // Exact actual capacity, not just length.
        self.requests[self.replies.len()] = request;
        self.replies.push(reply);
        self.payload = total;
        Ok(())
    }
    pub fn requests(&self) -> &[R] {
        &self.requests[..self.replies.len()]
    }
    pub fn provide(
        &mut self,
        provider: impl FnOnce(&[R], &mut [ReplyView<'_>]) -> Result<(), Error>,
    ) -> Result<(), Error> {
        if self.state != 0 {
            self.state = 1;
            return Err(Error::Consumed);
        }
        self.state = 1; // Includes callback panic and allocation failure; no retry.
        if self.replies.is_empty() {
            return Err(Error::Context);
        }
        // This view array's backing is included in the complete initial charge.
        self.budget.allocation()?;
        let mut views = Vec::new();
        views
            .try_reserve_exact(self.replies.len())
            .map_err(|_| Error::Allocation)?;
        if views.capacity() != self.replies.len() {
            return Err(Error::Resource);
        }
        for reply in &mut self.replies {
            views.push(ReplyView { reply });
        }
        provider(&self.requests[..views.len()], &mut views)?;
        drop(views);
        // Validate every write before the first importing/consuming caller step.
        for reply in &self.replies {
            reply.bytes()?;
        }
        self.state = 2;
        Ok(())
    }
    pub fn bytes(&self, index: usize) -> Result<&[u8], Error> {
        if self.state != 2 {
            return Err(Error::Consumed);
        }
        self.replies.get(index).ok_or(Error::Context)?.bytes()
    }
}

#[cfg(test)]
#[path = "provider_tests.rs"]
mod tests;

#[cfg(any(test, doctest))]
#[path = "provider_equivalence_tests.rs"]
mod equivalence_tests;
