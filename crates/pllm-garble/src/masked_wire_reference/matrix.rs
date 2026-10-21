//! Bounded native masked matrix product, SIGMA §3.2.1 / §4.1.1.
//!
//! Single-process, coupled two-share harness, not a party-local security boundary.
//! Dealer issuance takes only masks and public shape; each evaluator uses one
//! A/B/C share and the common masked operands, but both materials remain in the
//! same operation owner. Outputs reconstruct XW+RY in Z/(2^n), with no truncation,
//! clipping, scalar difference openings, or transport claim. Separate party-owned
//! material, authenticated import/export and transport lifecycle are not provided.
//! Caller owns operation control on one thread. Borrowed handles keep its full
//! reservation alive, including through exported-output erase/deallocation.
use super::product;
use crate::shared_allocation::{allocate, Lease, MAX_ALLOCATION_BYTES};
pub use crate::shared_allocation::{Budget, Error};
pub use pllm_core::streamed_matrix::MatrixLayout;
use pllm_core::streamed_matrix::{accumulate_resident, MatrixGeometry};
use sha2::{Digest, Sha256};
use std::{
    cell::{Cell, RefCell},
    marker::PhantomData,
    mem::size_of,
    rc::Rc,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
};
use zeroize::Zeroizing;

pub const MAX_ELEMENTS: usize = 256;
pub const MAX_COMPLETE_MACS: usize = 1 << 20;
static LIVE: AtomicBool = AtomicBool::new(false);
static ISSUANCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub bits: u8,
    pub rows: usize,
    pub inner: usize,
    pub columns: usize,
    pub layout: MatrixLayout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Context {
    pub plan: [u8; 32],
    pub session: [u8; 32],
    pub operation: [u8; 32],
    pub input_wire: [u8; 32],
    pub weight_wire: [u8; 32],
    pub output_wire: [u8; 32],
}
impl Context {
    fn validate(self) -> Result<(), Error> {
        if self.input_wire == self.weight_wire
            || self.input_wire == self.output_wire
            || self.weight_wire == self.output_wire
        {
            return Err(Error::Context);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resources {
    pub input_elements: usize,
    pub weight_elements: usize,
    pub output_elements: usize,
    pub party_key_bits: usize,
    pub dealer_macs: usize,
    pub party_macs: [usize; 2],
    pub complete_macs: usize,
    /// 8*(6a+6b+7c), including admitted caller masks, masked operands,
    /// retained clear inputs, and every simultaneously live private capacity.
    pub overlap_payload_bytes: usize,
    /// Caller-owned minimum within overlap: masks (a+b+c), common masked
    /// operands (a+b), retained clear inputs (a+b). Actual caller backing
    /// capacities/aliases beyond these admitted slices are ADDITIONAL owners;
    /// caller must price those separately. Slice APIs cannot inspect capacity.
    pub borrowed_caller_minimum_bytes: usize,
    pub owner_metadata_bytes: usize,
    /// Logical allowance for bounded call frames; not measured process stack.
    pub stack_allowance_bytes: usize,
    /// Per-allocation allowance, not a claim about system allocator internals.
    pub allocator_allowance_bytes: usize,
    pub reservation_bytes: usize,
}

pub fn resources(spec: Spec) -> Result<Resources, Error> {
    if !(1..=64).contains(&spec.bits) {
        return Err(Error::Domain);
    }
    let a = spec.rows.checked_mul(spec.inner).ok_or(Error::Resource)?;
    let b = spec
        .inner
        .checked_mul(spec.columns)
        .ok_or(Error::Resource)?;
    let c = spec.rows.checked_mul(spec.columns).ok_or(Error::Resource)?;
    if [a, b, c].iter().any(|n| !(1..=MAX_ELEMENTS).contains(n)) {
        return Err(Error::Resource);
    }
    let d = a.checked_mul(spec.columns).ok_or(Error::Resource)?;
    let complete = d.checked_mul(6).ok_or(Error::Resource)?;
    if complete > MAX_COMPLETE_MACS {
        return Err(Error::Resource);
    }
    geometry(spec)?;
    let payload = 8 * (6 * a + 6 * b + 7 * c);
    let metadata = size_of::<Operation<'static>>()
        + 2 * size_of::<Party<'static, 'static>>()
        + size_of::<MaskedInputs<'static, 'static>>()
        + 2 * size_of::<OutputShare<'static, 'static>>()
        + size_of::<MaskedMatrix<'static, 'static>>();
    let stack = 16 << 10;
    let allocator = 16 * 64;
    let reservation = payload + metadata + stack + allocator;
    if reservation > MAX_ALLOCATION_BYTES {
        return Err(Error::Resource);
    }
    Ok(Resources {
        input_elements: a,
        weight_elements: b,
        output_elements: c,
        party_key_bits: usize::from(spec.bits) * (a + b + c),
        dealer_macs: d,
        party_macs: [2 * d, 3 * d],
        complete_macs: complete,
        overlap_payload_bytes: payload,
        borrowed_caller_minimum_bytes: 8 * (3 * a + 3 * b + c),
        owner_metadata_bytes: metadata,
        stack_allowance_bytes: stack,
        allocator_allowance_bytes: allocator,
        reservation_bytes: reservation,
    })
}

fn geometry(spec: Spec) -> Result<MatrixGeometry, Error> {
    MatrixGeometry::new(spec.rows, spec.inner, spec.columns, spec.layout)
        .map_err(|_| Error::Resource)
}
fn mask(bits: u8) -> u64 {
    u64::MAX >> (64 - bits)
}
fn validate(values: &[u64], len: usize, bits: u8) -> Result<(), Error> {
    if values.len() != len || values.iter().any(|v| v & !mask(bits) != 0) {
        return Err(Error::Domain);
    }
    Ok(())
}
fn zeros(len: usize, budget: &Budget) -> Result<Zeroizing<Vec<u64>>, Error> {
    let mut v = allocate(len, budget)?;
    v.resize(len, 0);
    Ok(Zeroizing::new(v))
}
fn random(bytes: &mut [u8]) -> Result<(), Error> {
    getrandom::fill(bytes).map_err(|_| Error::Randomness)
}
fn split(
    value: u64,
    bits: u8,
    fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
) -> Result<[u64; 2], Error> {
    let mut bytes = Zeroizing::new([0; 8]);
    fill(&mut *bytes)?;
    let left = u64::from_le_bytes(*bytes) & mask(bits);
    Ok([left, value.wrapping_sub(left) & mask(bits)])
}

struct Live;
impl Live {
    fn acquire() -> Result<Self, Error> {
        LIVE.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .map_err(|_| Error::Resource)?;
        Ok(Self)
    }
}
impl Drop for Live {
    fn drop(&mut self) {
        LIVE.store(false, Ordering::SeqCst);
    }
}
struct Material {
    a: Zeroizing<Vec<u64>>,
    b: Zeroizing<Vec<u64>>,
    c: Zeroizing<Vec<u64>>,
}

/// Dealer-only entry point. No online handle retains RX/RW/RY.
pub struct Dealer;
impl Dealer {
    pub fn issue<'b>(
        context: Context,
        spec: Spec,
        rx: &[u64],
        rw: &[u64],
        ry: &[u64],
        budget: &'b Budget,
    ) -> Result<Operation<'b>, Error> {
        Self::issue_with(
            context,
            spec,
            [rx, rw, ry],
            budget,
            &mut random,
            &mut || false,
        )
    }

    /// Cancellation polled at least every 256 MACs, and every random draw.
    pub fn issue_cancellable<'b>(
        context: Context,
        spec: Spec,
        rx: &[u64],
        rw: &[u64],
        ry: &[u64],
        budget: &'b Budget,
        cancelled: &mut impl FnMut() -> bool,
    ) -> Result<Operation<'b>, Error> {
        Self::issue_with(context, spec, [rx, rw, ry], budget, &mut random, cancelled)
    }

    fn issue_with<'b>(
        context: Context,
        spec: Spec,
        [rx, rw, ry]: [&[u64]; 3],
        budget: &'b Budget,
        fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
        cancelled: &mut impl FnMut() -> bool,
    ) -> Result<Operation<'b>, Error> {
        // All public geometry, canonical masks, and total admission checked
        // before RNG or allocation. Borrowed caller buffers are priced above;
        // unrelated caller memory/aliases remain additional, not process-bounded.
        let cost = resources(spec)?;
        context.validate()?;
        validate(rx, cost.input_elements, spec.bits)?;
        validate(rw, cost.weight_elements, spec.bits)?;
        validate(ry, cost.output_elements, spec.bits)?;
        let lease = budget.reserve(cost.reservation_bytes)?;
        let live = Live::acquire()?;
        let issuance = ISSUANCE
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_add(1))
            .map_err(|_| Error::Resource)?;
        if cancelled() {
            return Err(Error::Consumed);
        }
        let mut nonce = [0; 32];
        fill(&mut nonce)?;
        let binding = binding(context, spec, issuance, &nonce);
        let mut materials = [
            Material {
                a: zeros(cost.input_elements, budget)?,
                b: zeros(cost.weight_elements, budget)?,
                c: zeros(cost.output_elements, budget)?,
            },
            Material {
                a: zeros(cost.input_elements, budget)?,
                b: zeros(cost.weight_elements, budget)?,
                c: zeros(cost.output_elements, budget)?,
            },
        ];
        let mut c = zeros(cost.output_elements, budget)?;
        c.copy_from_slice(ry);
        accumulate_resident(rw, &[rx], &mut [&mut c], geometry(spec)?, cancelled)
            .map_err(|_| Error::Consumed)?;
        for (kind, values) in [(0, rx), (1, rw), (2, &*c)] {
            for (i, &value) in values.iter().enumerate() {
                if cancelled() {
                    return Err(Error::Consumed);
                }
                let value = if kind < 2 {
                    product::negative_mask(value)
                } else {
                    value
                };
                let shares = Zeroizing::new(split(value, spec.bits, fill)?);
                for (party, material) in materials.iter_mut().enumerate() {
                    let output = match kind {
                        0 => &mut material.a,
                        1 => &mut material.b,
                        _ => &mut material.c,
                    };
                    output[i] = shares[party];
                }
            }
        }
        Ok(Operation {
            context,
            spec,
            binding,
            materials: RefCell::new(materials.map(Some)),
            retired: Cell::new(false),
            parties_taken: Cell::new(false),
            inputs_taken: Cell::new(false),
            reconstructed: Cell::new(false),
            budget,
            _thread: PhantomData,
            _lease: lease,
            _live: live,
        })
    }
}

fn binding(context: Context, spec: Spec, issuance: u64, nonce: &[u8; 32]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"pllm.masked.matrix.v1\0");
    for id in [
        context.plan,
        context.session,
        context.operation,
        context.input_wire,
        context.weight_wire,
        context.output_wire,
    ] {
        hash.update(id);
    }
    hash.update([spec.bits]);
    for dim in [spec.rows, spec.inner, spec.columns] {
        hash.update((dim as u64).to_le_bytes());
    }
    hash.update([match spec.layout {
        MatrixLayout::InputOutput => 0,
        MatrixLayout::OutputInput => 1,
    }]);
    hash.update(issuance.to_le_bytes());
    hash.update(nonce);
    hash.finalize().into()
}

/// Stack-owned coupled control. No Clone/Debug/Send/Sync. Drop only after all
/// borrowed workers, inputs, output shares and masked outputs have dropped.
/// Full reservation deliberately remains charged until this owner drops.
pub struct Operation<'b> {
    context: Context,
    spec: Spec,
    binding: [u8; 32],
    materials: RefCell<[Option<Material>; 2]>,
    retired: Cell<bool>,
    parties_taken: Cell<bool>,
    inputs_taken: Cell<bool>,
    reconstructed: Cell<bool>,
    budget: &'b Budget,
    _thread: PhantomData<Rc<()>>,
    // Keep lease/live guard after every secret-containing field.
    _lease: Lease<'b>,
    _live: Live,
}
impl<'b> Operation<'b> {
    pub fn cancel(&self) {
        self.retired.set(true);
        self.materials.replace([None, None]);
    }
    pub fn parties(&self) -> Result<[Party<'_, 'b>; 2], Error> {
        if self.retired.get() || self.parties_taken.replace(true) {
            self.cancel();
            return Err(Error::Consumed);
        }
        Ok([0, 1].map(|role| Party {
            operation: self,
            role,
            active: true,
        }))
    }

    /// Import canonical common masked operands, once. Context, exact public
    /// geometry/layout and all three wire identities must match issuance.
    pub fn masked_inputs(
        &self,
        context: Context,
        spec: Spec,
        xhat: &[u64],
        what: &[u64],
    ) -> Result<MaskedInputs<'_, 'b>, Error> {
        let result = (|| {
            if self.retired.get() || self.inputs_taken.replace(true) {
                return Err(Error::Consumed);
            }
            if context != self.context || spec != self.spec {
                return Err(Error::Context);
            }
            let cost = resources(spec)?;
            validate(xhat, cost.input_elements, spec.bits)?;
            validate(what, cost.weight_elements, spec.bits)?;
            let mut x = zeros(xhat.len(), self.budget)?;
            let mut w = zeros(what.len(), self.budget)?;
            x.copy_from_slice(xhat);
            w.copy_from_slice(what);
            Ok(MaskedInputs {
                x,
                w,
                operation: self,
            })
        })();
        if result.is_err() {
            self.cancel();
        }
        result
    }
}
impl Drop for Operation<'_> {
    fn drop(&mut self) {
        self.cancel();
    }
}

pub struct MaskedInputs<'o, 'b> {
    x: Zeroizing<Vec<u64>>,
    w: Zeroizing<Vec<u64>>,
    operation: &'o Operation<'b>,
}
pub struct Party<'o, 'b> {
    operation: &'o Operation<'b>,
    role: u8,
    active: bool,
}
impl<'o, 'b> Party<'o, 'b> {
    pub fn role(&self) -> u8 {
        self.role
    }
    pub fn cancel(self) {
        self.operation.cancel();
    }
    pub fn evaluate(
        mut self,
        input: &MaskedInputs<'_, '_>,
        cancelled: &mut impl FnMut() -> bool,
    ) -> Result<OutputShare<'o, 'b>, Error> {
        let op = self.operation;
        if op.retired.get() {
            return Err(Error::Consumed);
        }
        if !std::ptr::eq(op, input.operation) || self.role > 1 {
            return Err(Error::Context);
        }
        let material = op.materials.borrow_mut()[usize::from(self.role)]
            .take()
            .ok_or(Error::Consumed)?;
        let cost = resources(op.spec)?;
        // Revalidate every length/residue before allocating or mutating output.
        validate(&input.x, cost.input_elements, op.spec.bits)?;
        validate(&input.w, cost.weight_elements, op.spec.bits)?;
        validate(&material.a, cost.input_elements, op.spec.bits)?;
        validate(&material.b, cost.weight_elements, op.spec.bits)?;
        validate(&material.c, cost.output_elements, op.spec.bits)?;
        let mut output = zeros(cost.output_elements, op.budget)?;
        output.copy_from_slice(&material.c);
        let mut stop = || op.retired.get() || cancelled();
        let geometry = geometry(op.spec)?;
        // Exactly 2d or 3d MACs; complete ring contraction, no termwise floor.
        accumulate_resident(
            &input.w,
            &[&material.a],
            &mut [&mut output],
            geometry,
            &mut stop,
        )
        .map_err(|_| Error::Consumed)?;
        accumulate_resident(
            &material.b,
            &[&input.x],
            &mut [&mut output],
            geometry,
            &mut stop,
        )
        .map_err(|_| Error::Consumed)?;
        if self.role == 1 {
            accumulate_resident(
                &input.w,
                &[&input.x],
                &mut [&mut output],
                geometry,
                &mut stop,
            )
            .map_err(|_| Error::Consumed)?;
        }
        for value in output.iter_mut() {
            *value &= mask(op.spec.bits);
        }
        if stop() {
            return Err(Error::Consumed);
        }
        self.active = false;
        Ok(OutputShare {
            values: output,
            operation: op,
            binding: op.binding,
            role: self.role,
            active: true,
        })
    }
}
impl Drop for Party<'_, '_> {
    fn drop(&mut self) {
        if self.active {
            self.operation.cancel();
        }
    }
}

pub struct OutputShare<'o, 'b> {
    values: Zeroizing<Vec<u64>>,
    operation: &'o Operation<'b>,
    binding: [u8; 32],
    role: u8,
    active: bool,
}
impl Drop for OutputShare<'_, '_> {
    fn drop(&mut self) {
        if self.active {
            self.operation.cancel();
        }
    }
}
pub struct MaskedMatrix<'o, 'b> {
    values: Zeroizing<Vec<u64>>,
    operation: &'o Operation<'b>,
}
impl MaskedMatrix<'_, '_> {
    pub fn values(&self) -> &[u64] {
        &self.values
    }
    pub fn spec(&self) -> Spec {
        self.operation.spec
    }
    pub fn context(&self) -> Context {
        self.operation.context
    }
}

/// Consume complementary shares. Invalid attempts retire both operations.
pub fn reconstruct<'o, 'b>(
    mut left: OutputShare<'o, 'b>,
    mut right: OutputShare<'_, '_>,
) -> Result<MaskedMatrix<'o, 'b>, Error> {
    let op = left.operation;
    if op.retired.get() || op.reconstructed.replace(true) {
        return Err(Error::Consumed);
    }
    if !std::ptr::eq(op, right.operation)
        || left.binding != op.binding
        || right.binding != op.binding
        || left.role > 1
        || right.role != 1 - left.role
    {
        return Err(Error::Context);
    }
    let cost = resources(op.spec)?;
    validate(&left.values, cost.output_elements, op.spec.bits)?;
    validate(&right.values, cost.output_elements, op.spec.bits)?;
    let mut values = zeros(cost.output_elements, op.budget)?;
    for ((out, x), y) in values
        .iter_mut()
        .zip(left.values.iter())
        .zip(right.values.iter())
    {
        *out = x.wrapping_add(*y) & mask(op.spec.bits);
    }
    left.active = false;
    right.active = false;
    Ok(MaskedMatrix {
        values,
        operation: op,
    })
}

#[cfg(test)]
mod tests;
