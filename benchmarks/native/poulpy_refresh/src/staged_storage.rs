//! Typed storage/lifetime mechanics for the approved staged qualification design.
//!
//! No constructor here allocates a key, context, or ciphertext. Adopting existing
//! storage is NOT key-authentication or execution admission. The caller must
//! install `crate::ALLOCATOR` as its global BoundedAllocator::qualification().
//! This module is linked only by the ordinary staged integration tests for now;
//! the rejected eager admission executable is unchanged.
//!
//! All methods are serial. Views borrow the real Poulpy owners, so a live view
//! prevents consuming/releasing its owner or mutating its phase. No raw pointer,
//! cross-role cast, self-referential map, or fabricated key reference is used.
//!
//! Ownership audit: adoption, phase views and same-role refill retain original
//! leaves; import retains a full raw leaf and may overlap a retained original.
//! ReaderFrom's ComponentNoise::read_optional also creates a Vec then an Arc:
//! bounded wire bytes do not supply their complete capacity/overlap sizing API.
//! Standard typed prepare retains raw + output + original scratch capacity, plus
//! one serial private VMP Vec; raw/scratch release must precede ready-output use.
//! Clear grants credit only after observed unmap, never by changing a role tag.
//! Setup/context/solver, other owners, runtime, kernel and I/O remain explicitly
//! unclosed below; neither a leaf quote nor allocator headroom admits execution.
//! Opaque key adoption additionally rejects without authoritative original buffer
//! capacity/alignment. A canonical logical shape or backend span is insufficient;
//! original-capacity quotes remain supported for directly owned aligned scratch.

use std::{alloc::Layout, cell::Cell, marker::PhantomData, rc::Rc};

use poulpy_core::{
    error::CoreError,
    layouts::{
        GGLWEInfos, GGLWEPreparedToBackendRef, GLWEAutomorphismKeyPrepared,
        GLWEAutomorphismKeyPreparedBackendRef, GLWEAutomorphismKeyPreparedToBackendRef,
        GLWETensorKeyPrepared, GetAutomorphismKey, GetGaloisElement, LWEInfos, TorusPrecision,
    },
};
use poulpy_cpu_portable::NTT4x30Portable;
use poulpy_hal::{
    AlignedBuf,
    layouts::{Backend, DataView, PrepareHint, Standard},
};

use crate::bounded_allocator::{
    BOOKKEEPING_TICKET, BoundedAllocator, Mapper, PAGE_BYTES, Snapshot, geometry,
};

pub type Cpu = NTT4x30Portable<Standard>;
pub type Rotation = GLWEAutomorphismKeyPrepared<AlignedBuf, Cpu>;
pub type Tensor = GLWETensorKeyPrepared<AlignedBuf, Cpu>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageError {
    Shape,
    Capacity,
    Duplicate,
    AllocatorRejected,
    ReleaseNotObserved,
    ArithmeticOverflow,
    IncompleteQualificationBounds,
    UnprovenOwnerCapacity,
}

pub(crate) fn healthy(snapshot: Snapshot) -> bool {
    !snapshot.poisoned
        && snapshot.denied == 0
        && snapshot.map_failures == 0
        && snapshot.unmap_failures == 0
}

/// Source-proved high-key coefficient buffer only; metadata remains additional.
pub(crate) fn high_key_layout(key: &impl GGLWEInfos) -> Result<Layout, StorageError> {
    if key.n().as_usize() != 65536
        || key.base2k().as_usize() != 52
        || key.rank_in().as_usize() != 1
        || key.rank_out().as_usize() != 1
        || key.dnum().as_usize() != 7
        || key.dsize().as_usize() != 4
        || key.k_aux().as_usize() != 224
        || key.k().as_usize() != 1680
        || key.max_size() != 33
        || key.stride() != 1
    {
        return Err(StorageError::Shape);
    }
    // Same authoritative backend sizing contract as the eager admission probe;
    // all operands were checked above, and querying size allocates no storage.
    let bytes = Cpu::bytes_of_vmp_pmat(65536, 7, 1, 2, 33, PrepareHint::Reuse);
    if bytes != 968_884_224 {
        return Err(StorageError::Shape);
    }
    // Pinned prepared factory -> VmpPMatAlloc -> AlignedBuf, DEFAULTALIGN=64.
    Layout::from_size_align(bytes, 64).map_err(|_| StorageError::Shape)
}

/// Whole backend span is useful negative evidence, but cannot prove original
/// allocation capacity/alignment. Opaque specialized keys expose only borrowed
/// buffers; arbitrary raw AND prepared factory implementations can retain extra
/// capacity behind a matching span. Reject rather than fabricate an exact saved
/// Layout. This restriction does not apply to directly owned AlignedBuf/scratch.
pub(crate) fn require_original_layout(
    span: usize,
    expected: Layout,
) -> Result<Layout, StorageError> {
    if span != expected.size() {
        return Err(StorageError::Capacity);
    }
    Err(StorageError::UnprovenOwnerCapacity)
}

/// Exact geometry of the nominated allocation leaves, NOT a complete owner or
/// process bound. In particular, a key's noise Arc is not its coefficient leaf.
/// There is deliberately no subtraction/release-credit or admission operation.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct StorageQuote {
    payload: usize,
    mapped: usize,
    charged: usize,
    mappings: usize,
}

impl StorageQuote {
    pub fn allocation(layout: Layout) -> Result<Self, StorageError> {
        if layout.size() == 0 {
            return Err(StorageError::Shape);
        }
        let g = geometry(layout, PAGE_BYTES, BOOKKEEPING_TICKET)
            .ok_or(StorageError::ArithmeticOverflow)?;
        Ok(Self {
            payload: layout.size(),
            mapped: g.mapped,
            charged: g.charge,
            mappings: 1,
        })
    }

    pub fn checked_add(self, other: Self) -> Result<Self, StorageError> {
        let add = |a: usize, b| a.checked_add(b).ok_or(StorageError::ArithmeticOverflow);
        Ok(Self {
            payload: add(self.payload, other.payload)?,
            mapped: add(self.mapped, other.mapped)?,
            charged: add(self.charged, other.charged)?,
            mappings: add(self.mappings, other.mappings)?,
        })
    }

    /// Simultaneously live leaves, not cumulative requests over serial loops.
    pub fn checked_repeat(self, count: usize) -> Result<Self, StorageError> {
        let mul = |a: usize| a.checked_mul(count).ok_or(StorageError::ArithmeticOverflow);
        Ok(Self {
            payload: mul(self.payload)?,
            mapped: mul(self.mapped)?,
            charged: mul(self.charged)?,
            mappings: mul(self.mappings)?,
        })
    }

    pub fn payload(self) -> usize {
        self.payload
    }

    pub fn mapped(self) -> usize {
        self.mapped
    }

    pub fn charged(self) -> usize {
        self.charged
    }

    pub fn mappings(self) -> usize {
        self.mappings
    }

    /// Check this subset alone. A positive result is not execution admission.
    pub fn check_capacity(self, ceiling: usize, max_mappings: usize) -> Result<(), StorageError> {
        if self.charged > ceiling || self.mappings > max_mappings {
            return Err(StorageError::Capacity);
        }
        Ok(())
    }

    /// Additional allocations overlap ALL currently charged owners, including
    /// output, scratch, recovery and failed-release mappings. Never subtract a
    /// nominated owner from the snapshot or add an already-live leaf twice.
    /// This is a serial preflight, not a reservation or a concurrent guarantee.
    pub fn check_additional_capacity(
        self,
        current: Snapshot,
        ceiling: usize,
        max_mappings: usize,
    ) -> Result<(), StorageError> {
        if !healthy(current) {
            return Err(StorageError::AllocatorRejected);
        }
        let charged = current
            .charged
            .checked_add(self.charged)
            .ok_or(StorageError::ArithmeticOverflow)?;
        let mappings = current
            .mappings
            .checked_add(self.mappings)
            .ok_or(StorageError::ArithmeticOverflow)?;
        current
            .mapped
            .checked_add(self.mapped)
            .ok_or(StorageError::ArithmeticOverflow)?;
        if charged > ceiling || mappings > max_mappings {
            return Err(StorageError::Capacity);
        }
        Ok(())
    }
}

/// Fixed-capacity borrowed inventory of original owners. The borrowed Layout is
/// the saved allocation identity, not a caller-supplied logical length. Keeping
/// this inventory live prevents releasing/refilling its owners. Duplicate and
/// capacity failures burn the inventory; cleanup cannot rehabilitate it.
pub struct PhaseOwnerQuote<'a, const N: usize> {
    layouts: [Option<&'a Layout>; N],
    failed: Cell<bool>,
    _serial: PhantomData<Rc<()>>,
}

impl<'a, const N: usize> PhaseOwnerQuote<'a, N> {
    pub fn empty() -> Self {
        Self {
            layouts: [None; N],
            failed: Cell::new(false),
            _serial: PhantomData,
        }
    }

    pub fn retain<T>(&mut self, owner: &'a CheckedOwner<T>) -> Result<(), StorageError> {
        if self.failed.get() || !healthy(crate::ALLOCATOR.snapshot()) {
            self.failed.set(true);
            return Err(StorageError::AllocatorRejected);
        }
        if self
            .layouts
            .iter()
            .flatten()
            .any(|layout| std::ptr::eq(*layout, &owner.layout))
        {
            self.failed.set(true);
            return Err(StorageError::Duplicate);
        }
        let Some(slot) = self.layouts.iter_mut().find(|slot| slot.is_none()) else {
            self.failed.set(true);
            return Err(StorageError::Capacity);
        };
        *slot = Some(&owner.layout);
        Ok(())
    }

    pub fn known_storage(&self) -> Result<StorageQuote, StorageError> {
        if self.failed.get() || !healthy(crate::ALLOCATOR.snapshot()) {
            self.failed.set(true);
            return Err(StorageError::AllocatorRejected);
        }
        let result = self
            .layouts
            .iter()
            .flatten()
            .try_fold(StorageQuote::default(), |total, layout| {
                total.checked_add(StorageQuote::allocation(**layout)?)
            });
        if result.is_err() {
            self.failed.set(true);
        }
        result
    }

    pub fn check_capacity(&self, ceiling: usize, max_mappings: usize) -> Result<(), StorageError> {
        let result = self.known_storage()?.check_capacity(ceiling, max_mappings);
        if result.is_err() {
            self.failed.set(true);
        }
        result
    }
}

/// Bounds absent from the pinned public APIs. These are unknown, never zero or
/// caller-approved allowances. None of the storage quotes closes these gates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnclosedBound {
    /// Module::new, BootstrappingContext::{compile,compile_unprepared} and
    /// remez::{minimax_with,minimax_multi_interval_with} have no complete peak
    /// allocation query. BootstrappingContextUnprepared::prepare_tmp_bytes sizes
    /// matrix preparation scratch AFTER generation, not setup/solver heaps.
    /// Dashu parse/clone/arithmetic/conversion capacities and realloc overlap
    /// are not bounded by result precision.
    SetupAndSolver,
    /// *_bytes_of and *_tmp_bytes cover data/scratch, not the complete key graph,
    /// noise Arc/Vec capacity, context containers, codecs, secrets or client.
    MetadataAndRemainingOwners,
    /// vmp_prepare_tmp_bytes omits vec![0u32; 8*n]. The source below closes that
    /// ONE leaf only, not private heap sizing for every refresh/evaluation call.
    RemainingPrivateHeaps,
    /// GlobalAlloc cannot size dyld/libSystem, thread stacks or CommonCrypto.
    Runtime,
    /// No upstream/OS query bounds XNU VM bookkeeping/page tables/retention.
    Kernel,
    /// ReaderFrom/WriterTo/Read/Write/Seek expose no peak buffering, attributable
    /// file-cache, kernel-I/O or deadline bound, even with a bounded wire record.
    Io,
}

pub const UNCLOSED_BOUNDS: [UnclosedBound; 6] = [
    UnclosedBound::SetupAndSolver,
    UnclosedBound::MetadataAndRemainingOwners,
    UnclosedBound::RemainingPrivateHeaps,
    UnclosedBound::Runtime,
    UnclosedBound::Kernel,
    UnclosedBound::Io,
];

/// Actual GLWEAutomorphismKeyPreparedFactory path: gglwe_prepare -> VmpPrepare
/// -> NTT4x30VmpDefault -> ntt4x30_vmp_prepare_portable. Pinned kernel vmp.rs:
/// 53-54 sizes caller scratch; 98-127 creates/drops ONE private Vec<u32> per
/// iteration, 7*1*2*33 = 462 sequential iterations. Neither 462 simultaneously
/// live Vecs nor zero private heap is a valid quote. Allocation geometry includes
/// headers, rounding and tickets; cumulative request history is not peak charge.
pub fn rotation_prepare_private_workspace() -> Result<StorageQuote, StorageError> {
    let bytes = 8usize
        .checked_mul(65536)
        .and_then(|n| n.checked_mul(std::mem::size_of::<u32>()))
        .ok_or(StorageError::ArithmeticOverflow)?;
    StorageQuote::allocation(
        Layout::from_size_align(bytes, std::mem::align_of::<u32>())
            .map_err(|_| StorageError::Shape)?,
    )
}

/// Source-sized leaves for the frozen high-key shape; no Module/key allocation.
/// The scratch value mirrors ntt4x30_vmp_prepare_tmp_bytes_portable (4*n*u64),
/// not a new general backend contract. The typed quote below also checks the
/// actual public factory query on BOTH operands before accepting existing scratch.
pub fn rotation_prepare_leaf_quote(shape: &impl GGLWEInfos) -> Result<StorageQuote, StorageError> {
    let output = high_key_layout(shape)?;
    let raw = Layout::from_size_align(Cpu::bytes_of_mat_znx(65536, 7, 1, 2, 33), 64)
        .map_err(|_| StorageError::Shape)?;
    if raw.size() != 242_221_056 {
        return Err(StorageError::Shape);
    }
    let scratch = Layout::from_size_align(4 * 65536 * std::mem::size_of::<u64>(), 64)
        .map_err(|_| StorageError::Shape)?;
    StorageQuote::allocation(output)?
        .checked_add(StorageQuote::allocation(raw)?)?
        .checked_add(StorageQuote::allocation(scratch)?)?
        .checked_add(rotation_prepare_private_workspace()?)
}

/// Earliest allocation-free candidate-construction seam. Prices only NEW raw,
/// output, scratch and private workspace; already-live recovery is in the real
/// snapshot. Not for same-role refill (whose output already exists). Module,
/// solver, metadata and runtime/kernel/I/O remain unbounded, so even ample known
/// leaf headroom cannot authorize constructing any of them. No callback runs.
pub fn require_rotation_preconstruction(shape: &impl GGLWEInfos) -> Result<(), StorageError> {
    rotation_prepare_leaf_quote(shape)?.check_additional_capacity(
        crate::ALLOCATOR.snapshot(),
        crate::bounded_allocator::CHARGE_CEILING,
        crate::bounded_allocator::MAX_MAPPINGS,
    )?;
    admit_encrypted_execution()
}

pub struct RotationPreparationQuote<'a> {
    owners: PhaseOwnerQuote<'a, 4>,
    workspace: StorageQuote,
}

impl RotationPreparationQuote<'_> {
    /// Only the enumerated leaves: optional original/recovery raw owner remains
    /// charged alongside loading raw, prepared output, scratch and private Vec.
    pub fn known_storage(&self) -> Result<StorageQuote, StorageError> {
        let result = self.owners.known_storage()?.checked_add(self.workspace);
        if result.is_err() {
            self.owners.failed.set(true);
        }
        result
    }

    /// Owners in this quote ALREADY exist. Only the private Vec is additional to
    /// a snapshot taken now; metadata/other native allocations remain unbounded.
    pub fn check_workspace_capacity(
        &self,
        ceiling: usize,
        max_mappings: usize,
    ) -> Result<(), StorageError> {
        self.owners.known_storage()?;
        let result = self.workspace.check_additional_capacity(
            crate::ALLOCATOR.snapshot(),
            ceiling,
            max_mappings,
        );
        if result.is_err() {
            self.owners.failed.set(true);
        }
        result
    }
}

/// Pure preflight over the real standard typed operands. No factory invocation,
/// allocation, conversion, owner transfer, release credit or admission token.
/// Module setup itself is deliberately outside this partial storage quote.
pub fn quote_rotation_preparation<'a>(
    module: &poulpy_hal::layouts::Module<Cpu>,
    raw: &'a CheckedOwner<crate::staged_preparation::RawRotation>,
    destination: &'a CheckedOwner<Rotation>,
    scratch: &'a CheckedOwner<crate::staged_preparation::PreparationScratch>,
    recovery: Option<&'a CheckedOwner<crate::staged_preparation::RawRotation>>,
) -> Result<RotationPreparationQuote<'a>, StorageError> {
    use poulpy_core::layouts::{GLWEAutomorphismKeyPreparedFactory, GetDegree};

    if !healthy(crate::ALLOCATOR.snapshot()) {
        return Err(StorageError::AllocatorRejected);
    }
    high_key_layout(raw.borrow())?;
    high_key_layout(destination.borrow())?;
    if module.ring_degree().as_usize() != 65536 {
        return Err(StorageError::Shape);
    }
    // The real factory checks the destination's query, not only the source's.
    let required = module.glwe_automorphism_key_prepare_tmp_bytes(destination.borrow());
    if required != 2 << 20
        || module.glwe_automorphism_key_prepare_tmp_bytes(raw.borrow()) != required
    {
        return Err(StorageError::Shape);
    }
    if scratch.borrow().data.len() < required {
        return Err(StorageError::Capacity);
    }
    if scratch.borrow().data.align() < Cpu::SCRATCH_ALIGN {
        return Err(StorageError::Shape);
    }
    let mut owners = PhaseOwnerQuote::empty();
    owners.retain(raw)?;
    owners.retain(destination)?;
    owners.retain(scratch)?;
    if let Some(recovery) = recovery {
        high_key_layout(recovery.borrow())?;
        owners.retain(recovery)?;
    }
    Ok(RotationPreparationQuote {
        owners,
        workspace: rotation_prepare_private_workspace()?,
    })
}

/// An actual typed owner plus a minimum known leaf-release charge. Not Clone.
/// Rc marker prevents moving/sharing this serial release check across threads.
pub struct CheckedOwner<T> {
    owner: T,
    layout: Layout,
    _serial: PhantomData<Rc<()>>,
}

impl<T> CheckedOwner<T> {
    pub(crate) fn new(owner: T, layout: Layout) -> Result<Self, StorageError> {
        if !healthy(crate::ALLOCATOR.snapshot()) {
            return Err(StorageError::AllocatorRejected);
        }
        Ok(Self {
            owner,
            layout,
            _serial: PhantomData,
        })
    }

    pub fn borrow(&self) -> &T {
        &self.owner
    }

    pub(crate) fn borrow_mut(&mut self) -> &mut T {
        &mut self.owner
    }

    /// Original capacity and alignment, never a truncated logical length. This
    /// quotes the minimum known leaf only; it does not certify extra metadata.
    pub fn known_storage(&self) -> Result<StorageQuote, StorageError> {
        if !healthy(crate::ALLOCATOR.snapshot()) {
            return Err(StorageError::AllocatorRejected);
        }
        StorageQuote::allocation(self.layout)
    }

    /// Consuming self is illegal while a returned owner/backend view is live.
    /// A successful check is only a serial allocator observation, never a full
    /// process/RSS bound. All extra metadata frees may release MORE than one leaf.
    pub fn release(self) -> Result<(), StorageError> {
        drop_checked(&crate::ALLOCATOR, self.owner, self.layout)
    }
}

impl CheckedOwner<AlignedBuf> {
    pub fn aligned(buffer: AlignedBuf) -> Result<Self, StorageError> {
        // Original capacity, not truncated logical length.
        let layout = Layout::from_size_align(buffer.capacity(), buffer.align())
            .map_err(|_| StorageError::Shape)?;
        if layout.size() == 0 {
            return Err(StorageError::Shape);
        }
        Self::new(buffer, layout)
    }
}

impl CheckedOwner<Rotation> {
    pub fn rotation(key: Rotation) -> Result<Self, StorageError> {
        let expected = high_key_layout(&key)?;
        let backend = <Rotation as GGLWEPreparedToBackendRef<Cpu>>::to_backend_ref(&key);
        let layout =
            require_original_layout(Cpu::len_bytes_ref(DataView::data(backend.data())), expected)?;
        Self::new(key, layout)
    }
}

impl CheckedOwner<Tensor> {
    pub fn tensor(key: Tensor) -> Result<Self, StorageError> {
        let expected = high_key_layout(&key)?;
        let backend = <Tensor as GGLWEPreparedToBackendRef<Cpu>>::to_backend_ref(&key);
        let layout =
            require_original_layout(Cpu::len_bytes_ref(DataView::data(backend.data())), expected)?;
        Self::new(key, layout)
    }
}

/// No release credit from a drop alone: observe charged/mapped/count deltas AND
/// the sticky failure counters. Only internal audited owning types use this in
/// production. The generic form permits tiny deterministic fault-injection tests.
pub(crate) fn drop_checked<T, M: Mapper>(
    allocator: &BoundedAllocator<M>,
    owner: T,
    layout: Layout,
) -> Result<(), StorageError> {
    let expected = geometry(layout, PAGE_BYTES, BOOKKEEPING_TICKET);
    let before = allocator.snapshot();
    drop(owner);
    let after = allocator.snapshot();
    if !healthy(before) || !healthy(after) {
        return Err(StorageError::AllocatorRejected);
    }
    let expected = expected.ok_or(StorageError::ArithmeticOverflow)?;
    let charge = before.charged.checked_sub(expected.charge);
    let mapped = before.mapped.checked_sub(expected.mapped);
    let count = before.mappings.checked_sub(1);
    if !charge.is_some_and(|limit| after.charged <= limit)
        || !mapped.is_some_and(|limit| after.mapped <= limit)
        || !count.is_some_and(|limit| after.mappings <= limit)
    {
        return Err(StorageError::ReleaseNotObserved);
    }
    Ok(())
}

/// Seven real specialized owners, without a second heap-allocated key-view map.
/// No ability to allocate/load/preprocess keys or bypass unclosed setup gates.
pub struct RotationPhase {
    owners: [Option<CheckedOwner<Rotation>>; 7],
    failed: Cell<bool>,
    pending: Cell<bool>,
}

/// Consuming serial refill lease. Owns the removed original destination and
/// borrows the originating slots/failure state, not a detached failure copy.
/// No public destination extraction or generic commit permits bypassing the
/// preparation transaction. The generic core supports tiny non-HE regressions.
#[must_use = "commit through preparation or checked-abort; dropping burns the originating phase"]
pub struct RefillGuard<'a, T> {
    slots: &'a mut [Option<CheckedOwner<T>>],
    failed: &'a Cell<bool>,
    pending: &'a Cell<bool>,
    index: usize,
    destination: Option<CheckedOwner<T>>,
    committed: bool,
}

impl<'a, T> RefillGuard<'a, T> {
    pub(crate) fn begin(
        slots: &'a mut [Option<CheckedOwner<T>>],
        failed: &'a Cell<bool>,
        pending: &'a Cell<bool>,
        index: usize,
    ) -> Result<Self, StorageError> {
        if failed.get() || pending.get() || !healthy(crate::ALLOCATOR.snapshot()) {
            failed.set(true);
            return Err(StorageError::AllocatorRejected);
        }
        let Some(destination) = slots.get_mut(index).and_then(Option::take) else {
            failed.set(true);
            return Err(StorageError::Capacity);
        };
        // A forgotten lease cannot rehabilitate the phase either: pending stays
        // set until successful commit, independently of whether Drop runs.
        pending.set(true);
        Ok(Self {
            slots,
            failed,
            pending,
            index,
            destination: Some(destination),
            committed: false,
        })
    }

    pub(crate) fn take_destination(&mut self) -> Result<CheckedOwner<T>, StorageError> {
        if self.failed.get() || !healthy(crate::ALLOCATOR.snapshot()) {
            self.failed.set(true);
            return Err(StorageError::AllocatorRejected);
        }
        self.destination.take().ok_or_else(|| {
            self.failed.set(true);
            StorageError::Capacity
        })
    }

    /// Checked cleanup only: successful abort does not restore phase service.
    pub fn abort(mut self) -> Result<(), StorageError> {
        self.failed.set(true);
        if let Some(destination) = self.destination.take() {
            destination.release()?;
        }
        Ok(())
    }

    pub(crate) fn commit_checked(
        mut self,
        owner: CheckedOwner<T>,
        validate: impl FnOnce(&T) -> Result<(), StorageError>,
        same_identity: impl Fn(&T, &T) -> bool,
    ) -> Result<(), StorageError> {
        if self.failed.get() || !self.pending.get() || !healthy(crate::ALLOCATOR.snapshot()) {
            return Err(StorageError::AllocatorRejected);
        }
        if self.destination.is_some() || self.slots[self.index].is_some() {
            return Err(StorageError::Capacity);
        }
        validate(owner.borrow())?;
        if self
            .slots
            .iter()
            .flatten()
            .any(|other| same_identity(other.borrow(), owner.borrow()))
        {
            return Err(StorageError::Duplicate);
        }
        if !healthy(crate::ALLOCATOR.snapshot()) {
            return Err(StorageError::AllocatorRejected);
        }
        *self
            .slots
            .get_mut(self.index)
            .ok_or(StorageError::Capacity)? = Some(owner);
        self.committed = true;
        self.pending.set(false);
        Ok(())
    }
}

impl RefillGuard<'_, Rotation> {
    pub(crate) fn commit_rotation(self, owner: CheckedOwner<Rotation>) -> Result<(), StorageError> {
        self.commit_checked(
            owner,
            |key| high_key_layout(key).map(|_| ()),
            |a, b| a.p() == b.p(),
        )
    }
}

impl<T> Drop for RefillGuard<'_, T> {
    fn drop(&mut self) {
        if !self.committed {
            self.failed.set(true);
            if let Some(destination) = self.destination.take() {
                // Abort never grants a new-role allocation credit or a ready
                // owner. Failure remains sticky even after successful cleanup.
                let _ = destination.release();
            }
        }
    }
}

impl RotationPhase {
    pub fn empty() -> Self {
        Self {
            owners: std::array::from_fn(|_| None),
            failed: Cell::new(false),
            pending: Cell::new(false),
        }
    }

    pub fn adopt(&mut self, key: Rotation) -> Result<(), StorageError> {
        let owner = match CheckedOwner::rotation(key) {
            Ok(owner) => owner,
            Err(error) => {
                self.failed.set(true);
                return Err(error);
            }
        };
        self.adopt_checked(owner)
    }

    pub fn adopt_checked(&mut self, key: CheckedOwner<Rotation>) -> Result<(), StorageError> {
        if self.failed.get() || self.pending.get() || !healthy(crate::ALLOCATOR.snapshot()) {
            self.failed.set(true);
            return Err(StorageError::AllocatorRejected);
        }
        if self
            .owners
            .iter()
            .flatten()
            .any(|owner| owner.borrow().p() == key.borrow().p())
        {
            self.failed.set(true);
            return Err(StorageError::Duplicate);
        }
        let Some(slot) = self.owners.iter_mut().find(|owner| owner.is_none()) else {
            self.failed.set(true);
            return Err(StorageError::Capacity);
        };
        *slot = Some(key);
        Ok(())
    }

    /// Lease the actual owner for same-role preparation after every old view
    /// ends. Only checked preparation/commit can restore service. Abandoning the
    /// lease or any adoption/preparation error burns this originating phase.
    pub fn take_for_refill(
        &mut self,
        index: usize,
    ) -> Result<RefillGuard<'_, Rotation>, StorageError> {
        RefillGuard::begin(&mut self.owners, &self.failed, &self.pending, index)
    }

    pub fn views(&self) -> RotationViews<'_> {
        RotationViews { phase: self }
    }

    /// Actual occupied slots only. A destination taken for refill is no longer
    /// counted here; add it to the preparation inventory, never both inventories.
    pub fn storage_quote(&self) -> Result<PhaseOwnerQuote<'_, 7>, StorageError> {
        if self.failed.get() || self.pending.get() || !healthy(crate::ALLOCATOR.snapshot()) {
            self.failed.set(true);
            return Err(StorageError::AllocatorRejected);
        }
        let mut quote = PhaseOwnerQuote::empty();
        for owner in self.owners.iter().flatten() {
            quote.retain(owner)?;
        }
        Ok(quote)
    }

    /// &mut self cannot coexist with a live phase/backend view. Successfully
    /// unmap old owners before any caller attempts next-role key allocation.
    pub fn clear(&mut self) -> Result<(), StorageError> {
        for slot in &mut self.owners {
            if let Some(owner) = slot.take() {
                if let Err(error) = owner.release() {
                    self.failed.set(true);
                    return Err(error);
                }
            }
        }
        if self.failed.get() || self.pending.get() || !healthy(crate::ALLOCATOR.snapshot()) {
            self.failed.set(true);
            return Err(StorageError::AllocatorRejected);
        }
        Ok(())
    }
}

pub struct RotationViews<'a> {
    phase: &'a RotationPhase,
}

impl GetAutomorphismKey<Cpu> for RotationViews<'_> {
    fn lookup_automorphism_key(
        &self,
        p: i64,
        _k: TorusPrecision,
    ) -> Result<GLWEAutomorphismKeyPreparedBackendRef<'_, Cpu>, CoreError> {
        // String::new is allocation-free, including after allocator poison.
        let error = |op| CoreError::GGLWEKeyUse {
            op,
            detail: String::new(),
        };
        if self.phase.failed.get()
            || self.phase.pending.get()
            || !healthy(crate::ALLOCATOR.snapshot())
        {
            self.phase.failed.set(true);
            return Err(error("staged_storage_allocator_rejected"));
        }
        let owner = self
            .phase
            .owners
            .iter()
            .flatten()
            .find(|owner| owner.borrow().p() == p);
        match owner {
            Some(owner) => Ok(GLWEAutomorphismKeyPreparedToBackendRef::to_backend_ref(
                owner.borrow(),
            )),
            None => {
                self.phase.failed.set(true);
                Err(error("staged_storage_missing_automorphism"))
            }
        }
    }
}

/// Deliberately no positive admission token: source-proved key leaves do not
/// close solver/setup, private heaps, runtime/kernel/I/O, or numeric re-entry.
pub fn admit_encrypted_execution() -> Result<(), StorageError> {
    Err(StorageError::IncompleteQualificationBounds)
}
