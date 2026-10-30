//! Shared logical capacity budget; not an allocator/RSS bound.
use std::{
    alloc::Layout,
    mem::size_of,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

pub const MAX_ALLOCATION_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    Domain,
    Unsupported,
    Resource,
    Allocation,
    Randomness,
    Context,
    Consumed,
}

/// Borrowed by every lease, including exported output owners.
pub struct Budget {
    limit: usize,
    used: AtomicUsize,
    admitted_peak: AtomicUsize,
    #[cfg(test)]
    pub(crate) fail_after: std::sync::atomic::AtomicIsize,
    #[cfg(test)]
    accounting_updates: AtomicUsize,
}
impl Budget {
    pub fn new(limit: usize) -> Result<Self, Error> {
        if limit == 0 || limit > MAX_ALLOCATION_BYTES {
            return Err(Error::Resource);
        }
        Ok(Self {
            limit,
            used: AtomicUsize::new(0),
            admitted_peak: AtomicUsize::new(0),
            #[cfg(test)]
            fail_after: std::sync::atomic::AtomicIsize::new(-1),
            #[cfg(test)]
            accounting_updates: AtomicUsize::new(0),
        })
    }
    pub fn used_bytes(&self) -> usize {
        self.used.load(Ordering::SeqCst)
    }
    #[cfg(test)]
    pub(crate) fn test_snapshot(&self) -> (usize, usize) {
        (self as *const Self as usize, self.used_bytes())
    }
    #[cfg(test)]
    pub(crate) fn test_accounting_updates(&self) -> usize {
        self.accounting_updates.load(Ordering::SeqCst)
    }

    /// Highest successfully admitted numeric total, including reservations.
    ///
    /// Numeric admission evidence only, not measured storage or RSS. This does
    /// not establish completeness of an inventory, source/provenance authority,
    /// or work admission. Callers must inventory all owners and reserve actual
    /// storage separately. Releases and transfers never lower this value.
    pub fn admitted_peak_bytes(&self) -> usize {
        self.admitted_peak.load(Ordering::SeqCst)
    }
    /// Check a caller-inventoried complete phase total against this limit and
    /// monotonically retain it. Does not reserve/refund bytes or debit work.
    /// Rejection leaves both usage and admitted peak unchanged.
    pub fn admit_peak(&self, bytes: usize) -> Result<(), Error> {
        if bytes > self.limit {
            return Err(Error::Resource);
        }
        self.admitted_peak.fetch_max(bytes, Ordering::SeqCst);
        Ok(())
    }
    /// Charge caller-owned payload and the guard's inline metadata to this budget.
    ///
    /// This is logical accounting only: the guard does not allocate, own, or erase
    /// the payload, and grants no native execution or provenance authority. The
    /// caller must retain it until the actual payload owner has been dropped.
    /// Owner wrappers should enforce that ordering with private fields.
    pub fn charge_external(&self, payload_bytes: usize) -> Result<ExternalCharge<'_>, Error> {
        let bytes = payload_bytes
            .checked_add(std::mem::size_of::<ExternalCharge<'_>>())
            .ok_or(Error::Resource)?;
        Ok(ExternalCharge {
            lease: self.reserve(bytes)?,
        })
    }
    pub(crate) fn reserve(&self, bytes: usize) -> Result<Lease<'_>, Error> {
        self.reserve_bytes(bytes)?;
        Ok(Lease {
            budget: self,
            bytes,
        })
    }
    fn reserve_bytes(&self, bytes: usize) -> Result<(), Error> {
        let previous = self
            .used
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |used| {
                used.checked_add(bytes).filter(|next| *next <= self.limit)
            })
            .map_err(|_| Error::Resource)?;
        // The successful update checked this exact sum. Do not reload `used`:
        // another owner may already have released storage before this point.
        self.admitted_peak
            .fetch_max(previous + bytes, Ordering::SeqCst);
        #[cfg(test)]
        self.accounting_updates.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn release_bytes(&self, bytes: usize) {
        #[cfg(test)]
        self.accounting_updates.fetch_add(1, Ordering::SeqCst);
        self.used.fetch_sub(bytes, Ordering::SeqCst);
    }
    pub(crate) fn allocation(&self) -> Result<(), Error> {
        #[cfg(test)]
        {
            let n = self.fail_after.load(Ordering::SeqCst);
            if n == 0 {
                return Err(Error::Allocation);
            }
            if n > 0 {
                self.fail_after.fetch_sub(1, Ordering::SeqCst);
            }
        }
        Ok(())
    }
}

/// Non-cloneable logical accounting for caller-owned external storage.
///
/// Includes its own inline metadata in the same budget as the payload. It does
/// not own or erase caller buffers, certify their contents, or grant native
/// execution authority. Retain it through the actual payload owner's drop;
/// cancellation of work alone is not grounds for releasing a retained payload's
/// charge. Private owner-wrapper fields must enforce payload-before-guard drop.
///
/// ```compile_fail
/// use pllm_garble::shared_allocation::Budget;
/// let budget = Budget::new(1024).unwrap();
/// let charge = budget.charge_external(8).unwrap();
/// let duplicate = charge.clone();
/// ```
pub struct ExternalCharge<'a> {
    lease: Lease<'a>,
}

impl<'a> ExternalCharge<'a> {
    pub fn payload_bytes(&self) -> usize {
        self.lease.bytes - std::mem::size_of::<Self>()
    }

    pub fn charged_bytes(&self) -> usize {
        self.lease.bytes
    }

    /// Whether this guard charges this exact budget, not an equal-sized budget.
    pub fn is_for(&self, budget: &Budget) -> bool {
        std::ptr::eq(self.lease.budget, budget)
    }

    /// Move already-charged payload into a native owner without touching `used`.
    /// The source keeps its inline metadata and any untransferred payload. This
    /// is accounting only, never provenance or permission to execute.
    pub(crate) fn split_payload(
        &mut self,
        budget: &Budget,
        bytes: usize,
    ) -> Result<Lease<'a>, Error> {
        if !self.is_for(budget) || bytes > self.payload_bytes() {
            return Err(Error::Resource);
        }
        self.lease.split(bytes)
    }

    /// Change the logical payload charge, retaining the inline metadata charge.
    ///
    /// Growth reserves the positive increment atomically before changing this
    /// guard; failure leaves its original charge intact. Call before growing the
    /// actual payload. Shrink only after the corresponding storage is released.
    pub fn resize_payload(&mut self, payload_bytes: usize) -> Result<(), Error> {
        let bytes = payload_bytes
            .checked_add(std::mem::size_of::<Self>())
            .ok_or(Error::Resource)?;
        if bytes > self.lease.bytes {
            let mut increment = self.lease.budget.reserve(bytes - self.lease.bytes)?;
            self.lease.bytes = bytes;
            // Transfer accounting into this guard without releasing the increment.
            increment.bytes = 0;
        } else {
            let released = self.lease.bytes - bytes;
            self.lease.bytes = bytes;
            #[cfg(test)]
            self.lease
                .budget
                .accounting_updates
                .fetch_add(1, Ordering::SeqCst);
            self.lease.budget.used.fetch_sub(released, Ordering::SeqCst);
        }
        Ok(())
    }
}

mod context_seal {
    pub trait Sealed {}
}

/// Allocation ownership only, never numeric, source, material or transport
/// authority. Implementations are sealed to the original borrowed counter and
/// its owning strong link. Native containers price these links in their layout.
#[doc(hidden)]
pub trait AllocationContext: context_seal::Sealed + Sized {
    /// A named temporary strong link used during reserved construction. Ordinary
    /// borrowed views remain excluded from logical owned-storage inventories.
    const TRANSIENT_LINK_BYTES: usize;
    fn budget(&self) -> &Budget;
    #[doc(hidden)]
    fn share(&self) -> Self;
}

impl context_seal::Sealed for &Budget {}
impl AllocationContext for &Budget {
    const TRANSIENT_LINK_BYTES: usize = 0;
    fn budget(&self) -> &Budget {
        self
    }
    fn share(&self) -> Self {
        self
    }
}

/// An owning accounting link inside an already-admitted native container. Not
/// publicly constructible/cloneable and not a replacement native lifetime tag.
#[doc(hidden)]
pub struct OwnedBudgetContext {
    budget: Arc<Budget>,
}
impl context_seal::Sealed for OwnedBudgetContext {}
impl AllocationContext for OwnedBudgetContext {
    const TRANSIENT_LINK_BYTES: usize = size_of::<Self>();
    fn budget(&self) -> &Budget {
        &self.budget
    }
    fn share(&self) -> Self {
        Self {
            budget: Arc::clone(&self.budget),
        }
    }
}

// The borrowed specialization retains both its field type and two-word layout.
// Unrelated native users may continue to use `.budget: &Budget` unchanged.
pub(crate) type Lease<'a> = LeaseFor<&'a Budget>;
pub(crate) struct LeaseFor<C: AllocationContext> {
    pub(crate) budget: C,
    pub(crate) bytes: usize,
}
impl<C: AllocationContext> LeaseFor<C> {
    pub(crate) fn reserve(context: &C, bytes: usize) -> Result<Self, Error> {
        context.budget().reserve_bytes(bytes)?;
        Ok(Self {
            budget: context.share(),
            bytes,
        })
    }
    pub(crate) fn split(&mut self, bytes: usize) -> Result<Self, Error> {
        self.bytes = self.bytes.checked_sub(bytes).ok_or(Error::Resource)?;
        Ok(Self {
            budget: self.budget.share(),
            bytes,
        })
    }
}
impl<C: AllocationContext> Drop for LeaseFor<C> {
    fn drop(&mut self) {
        self.budget.budget().release_bytes(self.bytes);
    }
}

/// Logical bootstrap inventory, checked before allocating the owning counter.
/// The original inline Budget and named construction slots are conservatively
/// retained in the fixed charge until the last strong owner retires. Arc backing
/// includes both counters and alignment padding, not allocator headers or RSS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OwnedBudgetResources {
    pub arc_backing_bytes: usize,
    pub bootstrap_transition_bytes: usize,
    pub owner_metadata_bytes: usize,
    pub initial_reserved_bytes: usize,
}

/// Lifetime-independent accounting root, not a native lifetime bridge.
///
/// Only a NEW budget can be created here. No existing Budget, borrowed lease or
/// native owner is migrated, cloned or assigned a different accounting identity.
/// All handles and charges borrow the same stable original Budget through Arc.
/// Native owners obtained from `budget()` still cannot outlive that borrow.
/// Ownership-parametric native constructors instead consume an already-admitted
/// strong context; they never upgrade an existing borrowed owner.
///
/// Construction checks the complete bootstrap quote against the limit before
/// Arc allocation. Standard Arc allocation can abort on allocator OOM on stable
/// Rust; this Result is logical admission, not recoverable allocator admission.
pub struct OwnedBudget {
    budget: Arc<Budget>,
}

impl OwnedBudget {
    pub(crate) fn reserve_native(
        &self,
        bytes: usize,
    ) -> Result<LeaseFor<OwnedBudgetContext>, Error> {
        self.budget.reserve_bytes(bytes)?;
        Ok(LeaseFor {
            budget: OwnedBudgetContext {
                budget: Arc::clone(&self.budget),
            },
            bytes,
        })
    }
    pub fn resources() -> Result<OwnedBudgetResources, Error> {
        let arc_backing_bytes = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<Budget>())
            .map_err(|_| Error::Resource)?
            .0
            .pad_to_align()
            .size();
        // Explicit origin Budget as well as its destination in Arc backing.
        // These are named logical slots, not an ABI-stack measurement.
        let bootstrap_transition_bytes = [
            size_of::<Budget>(),
            size_of::<Result<Budget, Error>>(),
            size_of::<OwnedBudgetResources>(),
            size_of::<Result<OwnedBudgetResources, Error>>(),
            size_of::<Result<OwnedBudget, Error>>(),
            size_of::<Arc<Budget>>(),
            size_of::<Result<(), Error>>(),
        ]
        .into_iter()
        .try_fold(0usize, |sum, bytes| sum.checked_add(bytes))
        .ok_or(Error::Resource)?;
        let owner_metadata_bytes = size_of::<Self>();
        let initial_reserved_bytes = arc_backing_bytes
            .checked_add(bootstrap_transition_bytes)
            .and_then(|bytes| bytes.checked_add(owner_metadata_bytes))
            .ok_or(Error::Resource)?;
        Ok(OwnedBudgetResources {
            arc_backing_bytes,
            bootstrap_transition_bytes,
            owner_metadata_bytes,
            initial_reserved_bytes,
        })
    }

    pub fn new(limit: usize) -> Result<Self, Error> {
        Self::from_fresh_budget(Budget::new(limit)?)
    }

    fn from_fresh_budget(budget: Budget) -> Result<Self, Error> {
        let quote = Self::resources()?;
        if budget.used_bytes() != 0 {
            return Err(Error::Resource);
        }
        // No borrowed guard is detached or refunded. The counter and its fixed
        // bootstrap reservation move together into their admitted stable backing.
        budget.reserve_bytes(quote.initial_reserved_bytes)?;
        budget.allocation()?;
        Ok(Self {
            budget: Arc::new(budget),
        })
    }

    pub fn budget(&self) -> &Budget {
        &self.budget
    }

    #[cfg(test)]
    pub(crate) fn test_counter_weak(&self) -> std::sync::Weak<Budget> {
        Arc::downgrade(&self.budget)
    }

    /// Each additional root handle pays for its inline Arc pointer before clone.
    /// Arc's existing control/backing charge stays resident, without a new heap
    /// allocation, counter replacement, refund or reacquisition.
    pub fn try_clone(&self) -> Result<Self, Error> {
        self.budget.reserve_bytes(size_of::<Self>())?;
        Ok(Self {
            budget: Arc::clone(&self.budget),
        })
    }

    pub fn charge_external(&self, payload_bytes: usize) -> Result<OwnedExternalCharge, Error> {
        let bytes = payload_bytes
            .checked_add(size_of::<OwnedExternalCharge>())
            .ok_or(Error::Resource)?;
        self.budget.reserve_bytes(bytes)?;
        Ok(OwnedExternalCharge {
            budget: Arc::clone(&self.budget),
            bytes,
        })
    }
}

impl Drop for OwnedBudget {
    fn drop(&mut self) {
        // Fixed backing/bootstrap remains charged even if only charges survive.
        self.budget.release_bytes(size_of::<Self>());
    }
}

/// Non-cloneable owned logical reservation, including its inline Arc/byte count.
/// The original counter/backing remains alive even after all root handles drop.
/// This guard neither owns nor erases external payload. Keep it until that actual
/// payload retires, or use `try_build` to enforce direct payload-before-guard drop.
/// No raw Arc/Weak handle or budget adoption/migration API is exposed.
///
/// ```compile_fail
/// use pllm_garble::shared_allocation::OwnedBudget;
/// let budget = OwnedBudget::new(4096).unwrap();
/// let charge = budget.charge_external(8).unwrap();
/// let duplicate = charge.clone();
/// ```
pub struct OwnedExternalCharge {
    budget: Arc<Budget>,
    bytes: usize,
}

impl OwnedExternalCharge {
    pub fn budget(&self) -> &Budget {
        &self.budget
    }

    pub fn payload_bytes(&self) -> usize {
        self.bytes - size_of::<Self>()
    }

    pub fn charged_bytes(&self) -> usize {
        self.bytes
    }

    pub fn is_for(&self, budget: &Budget) -> bool {
        std::ptr::eq(self.budget(), budget)
    }

    /// Transfer an exact native inventory, including its already-quoted lease
    /// metadata, from admitted payload. Retain this guard's own metadata. Neither
    /// the counter nor material is recreated; zero-sized residual leases remain
    /// valid. Only native container implementations may consume this seam.
    pub(crate) fn split_native(
        &mut self,
        budget: &Budget,
        bytes: usize,
    ) -> Result<LeaseFor<OwnedBudgetContext>, Error> {
        if !self.is_for(budget) || bytes > self.payload_bytes() {
            return Err(Error::Resource);
        }
        let lease = LeaseFor {
            budget: OwnedBudgetContext {
                budget: Arc::clone(&self.budget),
            },
            bytes,
        };
        self.bytes -= bytes;
        Ok(lease)
    }

    /// Growth reserves only the positive increment before changing this guard.
    /// Shrink only after retiring the corresponding external payload storage.
    pub fn resize_payload(&mut self, payload_bytes: usize) -> Result<(), Error> {
        let bytes = payload_bytes
            .checked_add(size_of::<Self>())
            .ok_or(Error::Resource)?;
        if bytes > self.bytes {
            self.budget.reserve_bytes(bytes - self.bytes)?;
        } else {
            self.budget.release_bytes(self.bytes - bytes);
        }
        self.bytes = bytes;
        Ok(())
    }

    /// Carve a child and its metadata from already-reserved payload. No counter
    /// update or allocation occurs; both independent guards retain one identity.
    pub fn split_owned(&mut self, payload_bytes: usize) -> Result<Self, Error> {
        let bytes = payload_bytes
            .checked_add(size_of::<Self>())
            .filter(|bytes| *bytes <= self.payload_bytes())
            .ok_or(Error::Resource)?;
        let child = Self {
            budget: Arc::clone(&self.budget),
            bytes,
        };
        self.bytes -= bytes;
        Ok(child)
    }

    /// Transfer a child and its inline metadata into an already-empty caller slot.
    ///
    /// The slot must already reside outside the operation that may fail or unwind,
    /// and the child must remain there until its associated backing retires. This
    /// guard is capacity/accounting only, not proof of actual backing custody or
    /// whole-payload retention. Persistent owners should use this owned variant.
    /// The parent retains its own metadata and any residual payload; exact fit is
    /// allowed, but even a zero-payload child needs its own metadata capacity.
    ///
    /// An occupied slot (even a guard on this same counter) rejects first, then
    /// payload-plus-child-metadata overflow, then insufficient parent payload.
    /// Every rejection leaves both guards, usage, peak and update count unchanged.
    /// Success transfers existing capacity without reserving or releasing bytes.
    /// The existing Arc is cloned before subtraction; strong-count overflow may
    /// abort separately. This is not an allocator-failure rollback guarantee.
    pub fn split_owned_into(
        &mut self,
        payload_bytes: usize,
        dst: &mut Option<Self>,
    ) -> Result<(), Error> {
        if dst.is_some() {
            return Err(Error::Resource);
        }
        let bytes = payload_bytes
            .checked_add(size_of::<Self>())
            .ok_or(Error::Resource)?;
        if bytes > self.payload_bytes() {
            return Err(Error::Resource);
        }
        let child = Self {
            budget: Arc::clone(&self.budget),
            bytes,
        };
        self.bytes -= bytes;
        *dst = Some(child);
        Ok(())
    }

    /// Scoped capacity/accounting transfer into an already-empty caller slot.
    ///
    /// The slot must already reside outside the failing/unwinding operation and
    /// outlive the associated backing. This does not establish backing custody or
    /// whole-payload retention. Retire backing before dropping the child charge.
    /// For persistent owners use `split_owned_into` instead.
    ///
    /// Rejection order is occupied slot (including same-counter guards), checked
    /// payload-plus-ExternalCharge-metadata overflow, then insufficient parent
    /// payload. Rejection changes neither guard nor usage, peak or update count.
    /// Success keeps the original counter without reservation/release; the parent
    /// retains its metadata and any residual payload, including an exact-fit zero
    /// residual. A zero-payload child still consumes its own metadata capacity.
    ///
    /// The mutable parent borrow lasts while the child or a derived lease remains
    /// active: the parent cannot be resized, dropped, moved or split again then.
    ///
    /// ```compile_fail
    /// use pllm_garble::shared_allocation::OwnedBudget;
    /// let root = OwnedBudget::new(4096).unwrap();
    /// let mut parent = root.charge_external(256).unwrap();
    /// let mut child = None;
    /// parent.split_external_into(8, &mut child).unwrap();
    /// parent.resize_payload(0).unwrap();
    /// assert_eq!(child.as_ref().unwrap().payload_bytes(), 8);
    /// ```
    ///
    /// ```compile_fail
    /// use pllm_garble::shared_allocation::OwnedBudget;
    /// let root = OwnedBudget::new(4096).unwrap();
    /// let mut parent = root.charge_external(256).unwrap();
    /// let mut child = None;
    /// parent.split_external_into(8, &mut child).unwrap();
    /// drop(parent);
    /// assert_eq!(child.as_ref().unwrap().payload_bytes(), 8);
    /// ```
    ///
    /// ```compile_fail
    /// use pllm_garble::shared_allocation::OwnedBudget;
    /// let root = OwnedBudget::new(4096).unwrap();
    /// let mut parent = root.charge_external(256).unwrap();
    /// let mut first = None;
    /// let mut second = None;
    /// parent.split_external_into(8, &mut first).unwrap();
    /// parent.split_external_into(8, &mut second).unwrap();
    /// assert_eq!(first.as_ref().unwrap().payload_bytes(), 8);
    /// ```
    pub fn split_external_into<'a>(
        &'a mut self,
        payload_bytes: usize,
        dst: &mut Option<ExternalCharge<'a>>,
    ) -> Result<(), Error> {
        if dst.is_some() {
            return Err(Error::Resource);
        }
        let bytes = payload_bytes
            .checked_add(size_of::<ExternalCharge<'a>>())
            .ok_or(Error::Resource)?;
        if bytes > self.payload_bytes() {
            return Err(Error::Resource);
        }
        let child = ExternalCharge {
            lease: Lease {
                budget: &self.budget,
                bytes,
            },
        };
        self.bytes -= bytes;
        *dst = Some(child);
        Ok(())
    }

    /// Scoped handoff to the unchanged borrowed API, including destination guard
    /// metadata in the transferred payload. Source metadata stays charged. The
    /// result cannot escape its original owning counter's borrow.
    pub fn split_external(&mut self, payload_bytes: usize) -> Result<ExternalCharge<'_>, Error> {
        let bytes = payload_bytes
            .checked_add(size_of::<ExternalCharge<'_>>())
            .filter(|bytes| *bytes <= self.payload_bytes())
            .ok_or(Error::Resource)?;
        self.bytes -= bytes;
        Ok(ExternalCharge {
            lease: Lease {
                budget: &self.budget,
                bytes,
            },
        })
    }

    /// Admit inline payload storage/padding BEFORE invoking the allocating
    /// builder. The original payload reservation covers external backing; the
    /// returned wrapper additionally owns T inline. Errors/unwind drop builder
    /// locals before this guard. This does not enable self-referential returns.
    pub fn try_build<T>(
        mut self,
        build: impl FnOnce(&Budget) -> Result<T, Error>,
    ) -> Result<OwnedCharged<T>, Error> {
        let inline_bytes = size_of::<OwnedCharged<T>>()
            .checked_sub(size_of::<Self>())
            .ok_or(Error::Resource)?;
        let payload_bytes = self
            .payload_bytes()
            .checked_add(inline_bytes)
            .ok_or(Error::Resource)?;
        self.resize_payload(payload_bytes)?;
        let payload = build(self.budget())?;
        Ok(OwnedCharged {
            payload,
            charge: self,
        })
    }
}

impl Drop for OwnedExternalCharge {
    fn drop(&mut self) {
        self.budget.release_bytes(self.bytes);
    }
}

/// Private field order enforces direct payload destruction before its charge.
/// No extraction, mutable replacement, Clone or charge-detachment API exists.
pub struct OwnedCharged<T> {
    payload: T,
    charge: OwnedExternalCharge,
}

impl<T> OwnedCharged<T> {
    pub fn payload(&self) -> &T {
        &self.payload
    }

    pub fn budget(&self) -> &Budget {
        self.charge.budget()
    }

    /// Includes inline T/padding admitted by try_build as well as external backing.
    pub fn charged_bytes(&self) -> usize {
        self.charge.charged_bytes()
    }
}

// No custom Drop: ordinary enum destruction releases its one resident guard.
enum ResidentAdmissionState {
    Empty,
    Pending {
        charge: OwnedExternalCharge,
        backing_bytes: usize,
        quoted_bytes: usize,
    },
    Admitted {
        charge: OwnedExternalCharge,
        backing_bytes: usize,
        quoted_bytes: usize,
    },
    Sealed,
}

/// Resident reservation for an eventual `OwnedCharged<T>`, not backing custody.
///
/// Keep this slot outside failing work. Abandoning an attempt only ends its
/// exclusive borrow; the slot retains the actual guard. Caller-owned backing
/// must retire BEFORE this slot drops unless successfully sealed into the output.
/// The exact guard proves a budget reservation, not ownership of that backing.
/// Caller attempt policy must independently forbid unauthorized retries; Pending
/// admission may retry on the same counter after legitimate capacity release.
/// No root immortality, global burn, work payment or execution authority follows.
///
/// The type parameter is invariant, but the marker neither stores T nor binds
/// backing auto traits. Slot/state/handle/Option storage, move overlap and
/// cumulative work are not included in the payload quote or funded here.
///
/// ```compile_fail
/// use pllm_garble::shared_allocation::ResidentAdmission;
/// fn shorten<'a>(slot: ResidentAdmission<&'static str>, _: &'a str)
///     -> ResidentAdmission<&'a str> { slot }
/// ```
pub struct ResidentAdmission<T> {
    state: ResidentAdmissionState,
    invariant: std::marker::PhantomData<fn(T) -> T>,
}

impl<T> ResidentAdmission<T> {
    /// Construct an empty slot without allocating or changing accounting.
    pub fn empty() -> Self {
        Self {
            state: ResidentAdmissionState::Empty,
            invariant: std::marker::PhantomData,
        }
    }

    /// Borrow this slot exclusively; creating or dropping the handle is inert.
    pub fn attempt(&mut self) -> AdmissionAttempt<'_, T> {
        AdmissionAttempt { resident: self }
    }

    // B includes conservative external excess, but excludes the new inline T
    // increment. Full wrapper layout includes padding, even for aligned ZSTs.
    fn quote(backing_bytes: usize) -> Result<(usize, usize), Error> {
        let metadata = size_of::<OwnedExternalCharge>();
        let inline = size_of::<OwnedCharged<T>>()
            .checked_sub(metadata)
            .ok_or(Error::Resource)?;
        let source = metadata.checked_add(backing_bytes).ok_or(Error::Resource)?;
        let quoted = source.checked_add(inline).ok_or(Error::Resource)?;
        Ok((source, quoted))
    }

    /// Move an exact existing reservation into an empty resident slot.
    ///
    /// Rejection order: occupied state, missing source, checked quote, exact
    /// source bytes (positive excess also rejects). No fallible work follows take;
    /// no accounting, Arc clone, allocation or payload operation occurs.
    pub fn enter_into(
        source: &mut Option<OwnedExternalCharge>,
        backing_bytes: usize,
        dst: &mut Self,
    ) -> Result<(), Error> {
        if !matches!(&dst.state, ResidentAdmissionState::Empty) {
            return Err(Error::Resource);
        }
        let charge = source.as_ref().ok_or(Error::Resource)?;
        let (source_bytes, quoted_bytes) = Self::quote(backing_bytes)?;
        if charge.bytes != source_bytes {
            return Err(Error::Resource);
        }
        // Exclusive prechecks and no intervening callbacks prove Some. The
        // defensive None arm owns no charge and cannot discard an owner.
        match source.take() {
            Some(charge) => {
                dst.state = ResidentAdmissionState::Pending {
                    charge,
                    backing_bytes,
                    quoted_bytes,
                };
            }
            None => {}
        }
        Ok(())
    }
}

/// Exclusive borrowed operations on a resident reservation; no owned guard,
/// counter clone, permission, attempt token or Drop accounting lives here.
///
/// ```compile_fail
/// use pllm_garble::shared_allocation::ResidentAdmission;
/// let mut slot = ResidentAdmission::<u64>::empty();
/// let mut first = slot.attempt();
/// let mut second = slot.attempt();
/// first.admit_inline().unwrap();
/// second.admit_inline().unwrap();
/// ```
///
/// ```compile_fail
/// use pllm_garble::shared_allocation::{AdmissionAttempt, ResidentAdmission};
/// fn escape<'a>() -> AdmissionAttempt<'a, u64> {
///     let mut slot = ResidentAdmission::empty();
///     slot.attempt()
/// }
/// ```
pub struct AdmissionAttempt<'a, T> {
    resident: &'a mut ResidentAdmission<T>,
}

impl<T> AdmissionAttempt<'_, T> {
    /// Reserve only the positive inline increment on the guard's original
    /// counter. Failure retains Pending and its unchanged guard/metadata.
    /// Repeated valid Admitted calls perform no accounting update.
    pub fn admit_inline(&mut self) -> Result<(), Error> {
        match &mut self.resident.state {
            ResidentAdmissionState::Pending {
                charge,
                backing_bytes,
                quoted_bytes,
            } => {
                let (source, quoted) = ResidentAdmission::<T>::quote(*backing_bytes)?;
                if charge.bytes != source || *quoted_bytes != quoted {
                    return Err(Error::Resource);
                }
                let delta = quoted.checked_sub(charge.bytes).ok_or(Error::Resource)?;
                if delta != 0 {
                    charge.budget.reserve_bytes(delta)?;
                }
                // No fallibility remains after successful reservation.
                charge.bytes = quoted;
            }
            ResidentAdmissionState::Admitted {
                charge,
                backing_bytes,
                quoted_bytes,
            } => {
                let (_, quoted) = ResidentAdmission::<T>::quote(*backing_bytes)?;
                if charge.bytes != quoted || *quoted_bytes != quoted {
                    return Err(Error::Resource);
                }
                return Ok(());
            }
            ResidentAdmissionState::Empty | ResidentAdmissionState::Sealed => {
                return Err(Error::Resource);
            }
        }
        let old = std::mem::replace(&mut self.resident.state, ResidentAdmissionState::Empty);
        self.resident.state = match old {
            ResidentAdmissionState::Pending {
                charge,
                backing_bytes,
                quoted_bytes,
            } => ResidentAdmissionState::Admitted {
                charge,
                backing_bytes,
                quoted_bytes,
            },
            other => other,
        };
        Ok(())
    }

    /// Seal the whole payload and actual guard into an empty output slot.
    /// Rejection leaves all owners unchanged. Prechecks precede both moves;
    /// commit has no allocation, callback, fallible operation or owner drop.
    /// Output's existing field order destroys payload before releasing charge.
    pub fn seal_into(
        &mut self,
        payload: &mut Option<T>,
        dst: &mut Option<OwnedCharged<T>>,
    ) -> Result<(), Error> {
        if dst.is_some() || payload.is_none() {
            return Err(Error::Resource);
        }
        match &self.resident.state {
            ResidentAdmissionState::Admitted {
                charge,
                backing_bytes,
                quoted_bytes,
            } => {
                let (_, quoted) = ResidentAdmission::<T>::quote(*backing_bytes)?;
                if charge.bytes != quoted || *quoted_bytes != quoted {
                    return Err(Error::Resource);
                }
            }
            _ => return Err(Error::Resource),
        }
        let old = std::mem::replace(&mut self.resident.state, ResidentAdmissionState::Sealed);
        // Private exclusive preconditions prove the first arm. The defensive
        // fallback restores BOTH complete owners into Sealed/None slots.
        match (old, payload.take()) {
            (ResidentAdmissionState::Admitted { charge, .. }, Some(value)) => {
                *dst = Some(OwnedCharged {
                    payload: value,
                    charge,
                });
            }
            (state, value) => {
                self.resident.state = state;
                *payload = value;
            }
        }
        Ok(())
    }
}

pub(crate) fn allocate<T>(capacity: usize, budget: &Budget) -> Result<Vec<T>, Error> {
    budget.allocation()?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| Error::Allocation)?;
    exact_capacity(values, capacity)
}
pub(crate) fn exact_capacity<T>(values: Vec<T>, capacity: usize) -> Result<Vec<T>, Error> {
    if values.capacity() != capacity {
        return Err(Error::Resource);
    }
    Ok(values)
}

#[cfg(test)]
#[path = "shared_allocation_peak_tests.rs"]
mod peak_tests;

#[cfg(test)]
#[path = "shared_allocation_split_into_tests.rs"]
mod split_into_tests;

#[cfg(test)]
#[path = "shared_allocation_resident_admission_tests.rs"]
mod resident_admission_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn borrowed_lease_alias_preserves_fields_layout_and_identity() {
        let budget = Budget::new(1024).unwrap();
        let mut lease: Lease<'_> = budget.reserve(31).unwrap();
        let original: &Budget = lease.budget;
        assert!(std::ptr::eq(original, &budget));
        assert_eq!(size_of::<Lease<'_>>(), size_of::<(&Budget, usize)>());
        assert_eq!(
            std::mem::align_of::<Lease<'_>>(),
            std::mem::align_of::<(&Budget, usize)>()
        );
        let child = lease.split(7).unwrap();
        assert!(std::ptr::eq(child.budget, original));
        assert_eq!(budget.used_bytes(), 31);
        drop(child);
        drop(lease);
        assert_eq!(budget.used_bytes(), 0);
    }

    #[test]
    fn owned_native_transfer_keeps_full_original_counter_without_updates() {
        let baseline = OwnedBudget::resources().unwrap().initial_reserved_bytes;
        let metadata = size_of::<OwnedExternalCharge>();
        let owned = OwnedBudget::new(baseline + metadata + 31).unwrap();
        let foreign = Budget::new(baseline + metadata + 31).unwrap();
        let mut source = owned.charge_external(31).unwrap();
        let updates = owned.budget().accounting_updates.load(Ordering::SeqCst);
        for (target, bytes) in [(&foreign, 0), (&foreign, 1), (owned.budget(), 32)] {
            assert!(source.split_native(target, bytes).is_err());
            assert_eq!(source.payload_bytes(), 31);
        }
        let mut native = source.split_native(owned.budget(), 31).unwrap();
        let zero = native.split(0).unwrap();
        let child = native.split(31).unwrap();
        assert_eq!(source.payload_bytes(), 0);
        assert_eq!(native.bytes, 0);
        assert!(std::ptr::eq(child.budget.budget(), owned.budget()));
        assert_eq!(
            owned.budget().accounting_updates.load(Ordering::SeqCst),
            updates
        );
        assert_eq!(owned.budget().used_bytes(), baseline + metadata + 31);
        assert!(owned.budget().reserve(1).is_err());
        drop((zero, native, source));
        assert_eq!(child.budget.budget().used_bytes(), baseline + 31);
        drop(owned);
        assert_eq!(
            child.budget.budget().used_bytes(),
            baseline - size_of::<OwnedBudget>() + 31
        );
        drop(child);
    }

    #[test]
    fn owned_bootstrap_quotes_original_budget_and_checks_before_allocation() {
        let quote = OwnedBudget::resources().unwrap();
        assert_eq!(
            quote.initial_reserved_bytes,
            quote.arc_backing_bytes + quote.bootstrap_transition_bytes + quote.owner_metadata_bytes
        );
        assert!(quote.arc_backing_bytes >= size_of::<Budget>() + 2 * size_of::<AtomicUsize>());
        assert!(quote.bootstrap_transition_bytes >= size_of::<Budget>());
        assert_eq!(quote.owner_metadata_bytes, size_of::<OwnedBudget>());
        for limit in [
            0,
            quote.initial_reserved_bytes - 1,
            MAX_ALLOCATION_BYTES + 1,
            usize::MAX,
        ] {
            assert!(matches!(OwnedBudget::new(limit), Err(Error::Resource)));
        }
        for short in [true, false] {
            let budget = Budget::new(quote.initial_reserved_bytes - usize::from(short)).unwrap();
            budget.fail_after.store(0, Ordering::SeqCst);
            // An insufficient limit rejects BEFORE the allocation hook. Exact
            // fit reaches that hook, which rejects before any Arc is allocated.
            let expected = if short {
                Error::Resource
            } else {
                Error::Allocation
            };
            assert!(matches!(
                OwnedBudget::from_fresh_budget(budget),
                Err(error) if error == expected
            ));
        }
        let owned = OwnedBudget::new(quote.initial_reserved_bytes).unwrap();
        assert_eq!(owned.budget().used_bytes(), quote.initial_reserved_bytes);
        assert!(owned.try_clone().is_err());
        assert!(owned.charge_external(0).is_err());
        assert!(owned.budget().charge_external(0).is_err());
    }

    #[test]
    fn owned_and_borrowed_handles_share_original_identity_and_accounting() {
        let quote = OwnedBudget::resources().unwrap();
        let limit = quote.initial_reserved_bytes
            + size_of::<OwnedBudget>()
            + size_of::<OwnedExternalCharge>()
            + 7
            + size_of::<ExternalCharge<'_>>()
            + 3;
        let owned = OwnedBudget::new(limit).unwrap();
        let sibling = owned.try_clone().unwrap();
        let charge = owned.charge_external(7).unwrap();
        assert!(std::ptr::eq(owned.budget(), sibling.budget()));
        assert!(charge.is_for(owned.budget()));
        let foreign = Budget::new(limit).unwrap();
        assert!(!charge.is_for(&foreign));
        let borrowed = sibling.budget().charge_external(3).unwrap();
        assert!(borrowed.is_for(charge.budget()));
        assert_eq!(owned.budget().used_bytes(), limit);
        assert!(sibling.try_clone().is_err());
        drop(owned);
        assert_eq!(
            charge.budget().used_bytes(),
            limit - size_of::<OwnedBudget>()
        );
        drop(borrowed);
        drop(sibling);
        // The owned charge alone keeps the original Budget and fixed backing
        // charged. Original root-wrapper metadata is no longer resident.
        let fixed = quote.initial_reserved_bytes - quote.owner_metadata_bytes;
        assert_eq!(charge.budget().used_bytes(), fixed + charge.charged_bytes());
        let borrowed = charge.budget().charge_external(0).unwrap();
        assert!(borrowed.is_for(charge.budget()));
        drop(borrowed);
        drop(charge);
    }

    #[test]
    fn owned_growth_reserves_only_positive_increment_and_failure_is_atomic() {
        let baseline = OwnedBudget::resources().unwrap().initial_reserved_bytes;
        let metadata = size_of::<OwnedExternalCharge>();
        let owned = OwnedBudget::new(baseline + metadata + 20).unwrap();
        let mut charge = owned.charge_external(4).unwrap();
        let updates = owned.budget().accounting_updates.load(Ordering::SeqCst);
        charge.resize_payload(20).unwrap();
        assert_eq!(owned.budget().used_bytes(), baseline + metadata + 20);
        assert_eq!(
            owned.budget().accounting_updates.load(Ordering::SeqCst),
            updates + 1
        );
        let updates = owned.budget().accounting_updates.load(Ordering::SeqCst);
        for payload in [21, usize::MAX] {
            assert!(charge.resize_payload(payload).is_err());
            assert!(owned.charge_external(payload).is_err());
            assert_eq!(charge.payload_bytes(), 20);
            assert_eq!(owned.budget().used_bytes(), baseline + metadata + 20);
            assert_eq!(
                owned.budget().accounting_updates.load(Ordering::SeqCst),
                updates
            );
        }
        charge.resize_payload(3).unwrap();
        assert_eq!(owned.budget().used_bytes(), baseline + metadata + 3);
        drop(charge);
        assert_eq!(owned.budget().used_bytes(), baseline);
    }

    #[test]
    fn owned_splits_transfer_metadata_without_any_accounting_update() {
        let baseline = OwnedBudget::resources().unwrap().initial_reserved_bytes;
        let metadata = size_of::<OwnedExternalCharge>();
        for payload in [0, 1, 31] {
            for source_first in [false, true] {
                let limit = baseline + 2 * metadata + payload;
                let owned = OwnedBudget::new(limit).unwrap();
                let mut source = owned.charge_external(metadata + payload).unwrap();
                let updates = owned.budget().accounting_updates.load(Ordering::SeqCst);
                assert!(source.split_owned(usize::MAX).is_err());
                assert!(source.split_owned(payload + 1).is_err());
                let child = source.split_owned(payload).unwrap();
                assert_eq!(
                    owned.budget().accounting_updates.load(Ordering::SeqCst),
                    updates
                );
                assert_eq!(owned.budget().used_bytes(), limit);
                assert_eq!(source.payload_bytes(), 0);
                assert_eq!(child.payload_bytes(), payload);
                assert!(child.is_for(source.budget()));
                assert!(owned.budget().reserve(1).is_err());
                assert!(source.split_owned(0).is_err());
                if source_first {
                    drop(source);
                    assert_eq!(
                        owned.budget().used_bytes(),
                        baseline + child.charged_bytes()
                    );
                    drop(child);
                } else {
                    drop(child);
                    assert_eq!(owned.budget().used_bytes(), baseline + metadata);
                    drop(source);
                }
                assert_eq!(owned.budget().used_bytes(), baseline);
            }
        }
    }

    #[test]
    fn owned_reservation_hands_off_to_unchanged_borrowed_leases_without_refund() {
        let baseline = OwnedBudget::resources().unwrap().initial_reserved_bytes;
        let owned_metadata = size_of::<OwnedExternalCharge>();
        let borrowed_metadata = size_of::<ExternalCharge<'_>>();
        let limit = baseline + owned_metadata + borrowed_metadata + 31;
        let owned = OwnedBudget::new(limit).unwrap();
        let mut source = owned.charge_external(borrowed_metadata + 31).unwrap();
        let updates = owned.budget().accounting_updates.load(Ordering::SeqCst);
        assert!(source.split_external(usize::MAX).is_err());
        assert!(source.split_external(32).is_err());
        let mut borrowed = source.split_external(31).unwrap();
        assert!(borrowed.is_for(owned.budget()));
        let lease = borrowed.split_payload(owned.budget(), 31).unwrap();
        assert_eq!(
            owned.budget().accounting_updates.load(Ordering::SeqCst),
            updates
        );
        assert_eq!(owned.budget().used_bytes(), limit);
        assert!(owned.budget().reserve(1).is_err());
        drop(borrowed);
        assert_eq!(owned.budget().used_bytes(), baseline + owned_metadata + 31);
        drop(lease);
        assert_eq!(source.payload_bytes(), 0);
        drop(source);
        assert_eq!(owned.budget().used_bytes(), baseline);
    }

    #[test]
    fn owned_builder_admits_backing_and_inline_storage_before_allocation() {
        let baseline = OwnedBudget::resources().unwrap().initial_reserved_bytes;
        let metadata = size_of::<OwnedExternalCharge>();
        let inline = size_of::<OwnedCharged<Vec<u8>>>() - metadata;
        for short in [true, false] {
            let limit = baseline + metadata + 17 + inline - usize::from(short);
            let owned = OwnedBudget::new(limit).unwrap();
            let charge = owned.charge_external(17).unwrap();
            let called = Cell::new(false);
            let result = charge.try_build(|budget| {
                called.set(true);
                assert_eq!(budget.used_bytes(), limit);
                assert!(std::ptr::eq(budget, owned.budget()));
                allocate::<u8>(17, budget)
            });
            assert_eq!(called.get(), !short);
            if short {
                assert!(matches!(result, Err(Error::Resource)));
            } else {
                let payload = result.unwrap();
                assert_eq!(payload.payload().capacity(), 17);
                assert_eq!(payload.charged_bytes(), metadata + 17 + inline);
                assert!(std::ptr::eq(payload.budget(), owned.budget()));
                drop(payload);
            }
            assert_eq!(owned.budget().used_bytes(), baseline);
        }
    }

    struct ObserveDrop<'a> {
        budget: &'a Budget,
        observed: &'a Cell<usize>,
    }
    impl Drop for ObserveDrop<'_> {
        fn drop(&mut self) {
            self.observed.set(self.budget.used_bytes());
        }
    }

    #[test]
    fn owned_payload_drops_before_charge_on_success_error_and_unwind() {
        let baseline = OwnedBudget::resources().unwrap().initial_reserved_bytes;
        let metadata = size_of::<OwnedExternalCharge>();
        let inline = size_of::<OwnedCharged<ObserveDrop<'_>>>() - metadata;
        let limit = baseline + metadata + inline;
        for case in 0..3 {
            let owned = OwnedBudget::new(limit).unwrap();
            let observed = Cell::new(0);
            let charge = owned.charge_external(0).unwrap();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let payload = charge.try_build(|_| {
                    let payload = ObserveDrop {
                        budget: owned.budget(),
                        observed: &observed,
                    };
                    match case {
                        1 => Err(Error::Allocation),
                        2 => panic!("owned builder unwind"),
                        _ => Ok(payload),
                    }
                });
                match case {
                    0 => drop(payload.unwrap()),
                    1 => assert!(matches!(payload, Err(Error::Allocation))),
                    _ => unreachable!(),
                }
            }));
            assert_eq!(result.is_err(), case == 2);
            assert_eq!(observed.get(), limit);
            assert_eq!(owned.budget().used_bytes(), baseline);
        }
    }

    #[test]
    fn owned_builder_allocation_rejection_retires_the_consumed_charge() {
        let baseline = OwnedBudget::resources().unwrap().initial_reserved_bytes;
        let owned = OwnedBudget::new(baseline + 1024).unwrap();
        let charge = owned.charge_external(17).unwrap();
        owned.budget().fail_after.store(0, Ordering::SeqCst);
        assert!(matches!(
            charge.try_build(|budget| allocate::<u8>(17, budget)),
            Err(Error::Allocation)
        ));
        assert_eq!(owned.budget().used_bytes(), baseline);
    }

    #[test]
    fn payload_transfer_keeps_exact_full_budget_and_inline_guard() {
        let metadata = std::mem::size_of::<ExternalCharge<'_>>();
        let budget = Budget::new(metadata + 31).unwrap();
        let mut charge = budget.charge_external(31).unwrap();
        let updates = budget.accounting_updates.load(Ordering::SeqCst);
        let lease = charge.split_payload(&budget, 31).unwrap();
        // Unlike a final used-byte comparison, this catches a transient
        // decrement/reacquire window even if nobody wins the race for capacity.
        assert_eq!(budget.accounting_updates.load(Ordering::SeqCst), updates);
        assert_eq!(lease.bytes, 31);
        assert_eq!(charge.payload_bytes(), 0);
        assert_eq!(charge.charged_bytes(), metadata);
        assert_eq!(budget.used_bytes(), metadata + 31);
        // Transfer did not make even one byte available to a competing owner.
        assert!(matches!(budget.reserve(1), Err(Error::Resource)));
        drop(charge);
        assert_eq!(budget.used_bytes(), 31);
        drop(lease);
        assert_eq!(budget.used_bytes(), 0);
    }

    #[test]
    fn payload_transfer_failure_is_atomic_and_partial_slices_drop_independently() {
        let metadata = std::mem::size_of::<ExternalCharge<'_>>();
        let budget = Budget::new(metadata + 31).unwrap();
        let foreign = Budget::new(metadata + 31).unwrap();
        let mut charge = budget.charge_external(31).unwrap();
        for (target, bytes) in [(&foreign, 1), (&foreign, 0), (&budget, 32)] {
            assert!(matches!(
                charge.split_payload(target, bytes),
                Err(Error::Resource)
            ));
            assert_eq!(charge.payload_bytes(), 31);
            assert_eq!(budget.used_bytes(), metadata + 31);
            assert_eq!(foreign.used_bytes(), 0);
        }
        let first = charge.split_payload(&budget, 12).unwrap();
        let second = charge.split_payload(&budget, 19).unwrap();
        drop(first);
        assert_eq!(budget.used_bytes(), metadata + 19);
        drop(second);
        assert_eq!(budget.used_bytes(), metadata);
        charge.resize_payload(31).unwrap();
        drop(charge);
        assert_eq!(budget.used_bytes(), 0);
    }

    #[test]
    fn zero_and_one_byte_transfers_preserve_metadata_in_either_drop_order() {
        let metadata = std::mem::size_of::<ExternalCharge<'_>>();
        for bytes in [0, 1] {
            for retained in [0, 1] {
                for source_first in [false, true] {
                    let budget = Budget::new(metadata + bytes + retained).unwrap();
                    let mut source = budget.charge_external(bytes + retained).unwrap();
                    let updates = budget.accounting_updates.load(Ordering::SeqCst);
                    let native = source.split_payload(&budget, bytes).unwrap();
                    assert_eq!(budget.accounting_updates.load(Ordering::SeqCst), updates);
                    assert_eq!(source.payload_bytes(), retained);
                    assert_eq!(source.charged_bytes(), metadata + retained);
                    assert_eq!(budget.used_bytes(), metadata + bytes + retained);
                    if source_first {
                        drop(source);
                        assert_eq!(budget.used_bytes(), bytes);
                        drop(native);
                    } else {
                        drop(native);
                        assert_eq!(budget.used_bytes(), metadata + retained);
                        drop(source);
                    }
                    assert_eq!(budget.used_bytes(), 0);
                }
            }
        }
    }
}
