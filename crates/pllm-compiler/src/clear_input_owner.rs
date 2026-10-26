//! Owned ordinary clear fixtures, NOT checkpoint/import/source authority.
//!
//! Reuses ModelOperation/ModelOperator and shared opaque JSON tree accounting.
//! replicated_decoder::SourceTensor is deliberately NOT fabricated: its digest,
//! encoded Q18 norms and import contracts do not describe these f64/u64 fixtures.
//! No public adoption, callbacks, mutable payload, guard or ledger extraction.
//! Logical capacity/work accounting is not allocator/RSS or physical erasure.
//!
//! MODULE-SPECIFIC CONTRACT: tensor names/shapes, operation IDs/inputs/layer and
//! ALL attributes (including numeric attributes) are PUBLIC semantic IR only.
//! Never put secret parameters in descriptor JSON; use the typed tensor pools.
//! This does not reclassify ModelOperation globally or assert other producers
//! already obey this contract. Only fully initialized numeric tensor storage,
//! including unused admitted tails, is zeroized; no blanket metadata wipe claim.
use pllm_core::row_fold::WorkLedger;
use pllm_garble::shared_allocation::{ExternalCharge, OwnedBudget, OwnedCharged};
use pllm_models::{owned_backing::checked_tree_node_bytes, ModelOperation, ModelOperator};
use serde_json::{Map, Number, Value};
use std::{
    cell::{Cell, Ref, RefCell},
    mem::size_of,
    ops::Range,
};
use zeroize::Zeroize;

pub const MAX_WORK: usize = 65536;
const NAME: usize = 256;
const KEY: usize = 64;
const RANK: usize = 4;
const INPUTS: usize = 2;
const ATTRS: usize = 8;
const MAX_ATTEMPTS: usize = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    Resource,
    Work,
    Shape,
    NonFinite,
    Burned,
}

#[derive(Clone, Copy, Debug)]
pub struct Caps {
    pub f64_values: usize,
    pub u64_values: usize,
    pub tensors: usize,
    pub operations: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Resources {
    pub charged_owner_bytes: usize,
    pub minimum_fresh_budget_bytes: usize,
    /// Prepaid whole-capacity initialization and retirement, even on failure.
    pub construction_work: usize,
    backing_bytes: usize,
}

/// Concrete bounded attributes, materialized in the EXISTING ModelOperation
/// schema. Not a second JSON specification or a claim about model provenance.
pub enum Attribute<'a> {
    Text(&'a str),
    Unsigned(u64),
    Real(f64),
    Bool(bool),
    Shape(&'a [u64]),
}

enum Storage {
    F64(Range<usize>),
    U64(Range<usize>),
}
pub struct ClearTensor {
    name: String,
    shape: Vec<usize>,
    storage: Storage,
}
impl ClearTensor {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}
struct Data {
    floats: Vec<f64>,
    words: Vec<u64>,
    tensors: Vec<ClearTensor>,
    ops: Vec<ModelOperation>,
    float_used: usize,
    word_used: usize,
    tensor_used: usize,
    op_used: usize,
}
impl Data {
    fn wipe(&mut self) {
        // Length equals admitted capacity for numeric pools, including unused tail.
        self.floats.as_mut_slice().zeroize();
        self.words.as_mut_slice().zeroize();
    }
}
impl Drop for Data {
    fn drop(&mut self) {
        self.wipe();
    }
}
struct Payload {
    data: RefCell<Data>,
    ledger: RefCell<WorkLedger>,
    caps: Caps,
    poisoned: Cell<bool>,
    attempts: Cell<usize>,
    work_limit: usize,
}
pub struct ClearInputBuilder {
    owned: OwnedCharged<Payload>,
}
pub struct ClearInputs {
    owned: OwnedCharged<Payload>,
}

fn allocated<T>(n: usize) -> Result<Vec<T>, Error> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| Error::Resource)?;
    if v.capacity() != n {
        return Err(Error::Resource);
    }
    Ok(v)
}
fn text(s: &str, cap: usize) -> Result<String, Error> {
    if s.is_empty() || s.len() > cap {
        return Err(Error::Shape);
    }
    let mut out = String::new();
    out.try_reserve_exact(cap).map_err(|_| Error::Resource)?;
    if out.capacity() != cap {
        return Err(Error::Resource);
    }
    out.push_str(s);
    Ok(out)
}
fn debit(p: &Payload, clear: usize, control: usize) -> Result<(), Error> {
    p.ledger
        .borrow_mut()
        .debit(clear, control)
        .map_err(|_| Error::Work)
}

impl ClearInputBuilder {
    pub fn resources(c: Caps) -> Result<Resources, Error> {
        if c.f64_values > 8192
            || c.u64_values > 8192
            || c.tensors > 16
            || c.operations > 8
            || c.tensors == 0
            || c.f64_values + c.u64_values == 0
        {
            return Err(Error::Resource);
        }
        // All slots, bounded names/inputs/shapes, opaque object nodes, and one
        // transient attribute value/name are reserved BEFORE any allocating builder.
        let tree = checked_tree_node_bytes::<(String, Value)>().map_err(|_| Error::Resource)?;
        let tensor = size_of::<ClearTensor>() + NAME + RANK * size_of::<usize>();
        let op = size_of::<ModelOperation>()
            + NAME
            + INPUTS * (size_of::<String>() + NAME)
            + RANK * 8
            + ATTRS * (tree + KEY + NAME + RANK * size_of::<Value>());
        let backing_bytes =
            4096 + 8 * (c.f64_values + c.u64_values) + c.tensors * tensor + c.operations * op;
        let charged_owner_bytes = backing_bytes + size_of::<OwnedCharged<Payload>>();
        if charged_owner_bytes > 1 << 20 {
            return Err(Error::Resource);
        }
        let construction_work = 512
            + 3 * (c.f64_values + c.u64_values)
            + c.tensors * (NAME + RANK + 16)
            + c.operations * (NAME + INPUTS * NAME + RANK + ATTRS * (KEY + NAME + RANK) + 64);
        if construction_work > MAX_WORK {
            return Err(Error::Work);
        }
        let minimum_fresh_budget_bytes = OwnedBudget::resources()
            .map_err(|_| Error::Resource)?
            .initial_reserved_bytes
            + charged_owner_bytes;
        Ok(Resources {
            charged_owner_bytes,
            minimum_fresh_budget_bytes,
            construction_work,
            backing_bytes,
        })
    }
    /// Consumes an existing owned accounting root. Fresh input may pay fresh work;
    /// the SAME retained input can never replace/refill its original ledger.
    pub fn new(budget: OwnedBudget, caps: Caps, work_limit: usize) -> Result<Self, Error> {
        let q = Self::resources(caps)?;
        if work_limit > MAX_WORK || work_limit < q.construction_work {
            return Err(Error::Work);
        }
        let mut ledger = WorkLedger::new(work_limit, work_limit);
        ledger
            .debit(q.construction_work, 0)
            .map_err(|_| Error::Work)?;
        let charge = budget
            .charge_external(q.backing_bytes)
            .map_err(|_| Error::Resource)?;
        let owned = charge
            .try_build(|_| {
                let build = || -> Result<Payload, Error> {
                    let mut floats = allocated(caps.f64_values)?;
                    floats.resize(caps.f64_values, 0.0);
                    let mut words = allocated(caps.u64_values)?;
                    words.resize(caps.u64_values, 0);
                    // Install numeric pools into their cleanup owner BEFORE any later
                    // descriptor allocation can fail. All error paths retire via Drop.
                    let mut data = Data {
                        floats,
                        words,
                        tensors: Vec::new(),
                        ops: Vec::new(),
                        float_used: 0,
                        word_used: 0,
                        tensor_used: 0,
                        op_used: 0,
                    };
                    data.tensors = allocated(caps.tensors)?;
                    data.ops = allocated(caps.operations)?;
                    for _ in 0..caps.tensors {
                        data.tensors.push(ClearTensor {
                            name: text("_", NAME)?,
                            shape: allocated(RANK)?,
                            storage: Storage::F64(0..0),
                        });
                    }
                    for _ in 0..caps.operations {
                        data.ops.push(ModelOperation {
                            id: text("_", NAME)?,
                            operator: ModelOperator::Linear,
                            layer: None,
                            state_kind: None,
                            inputs: allocated(INPUTS)?,
                            output_shape: allocated(RANK)?,
                            attributes: Value::Object(Map::new()),
                        });
                    }
                    Ok(Payload {
                        data: RefCell::new(data),
                        ledger: RefCell::new(ledger),
                        caps,
                        poisoned: Cell::new(false),
                        attempts: Cell::new(0),
                        work_limit,
                    })
                };
                build().map_err(|_| pllm_garble::shared_allocation::Error::Resource)
            })
            .map_err(|_| Error::Resource)?;
        Ok(Self { owned })
    }
    fn start_write(&self, work: usize) -> Result<&Payload, Error> {
        let p = self.owned.payload();
        if p.poisoned.replace(true) {
            return Err(Error::Burned);
        }
        debit(p, work, 0)?;
        Ok(p)
    }
    fn tensor(&self, name: &str, shape: &[usize], len: usize, words: bool) -> Result<usize, Error> {
        let p = self.start_write(
            512 + 2 * len.min(8192) + name.len().min(NAME) * self.owned.payload().caps.tensors,
        )?;
        if name.is_empty()
            || name.len() > NAME
            || shape.is_empty()
            || shape.len() > RANK
            || shape.iter().any(|&d| d == 0)
            || shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d)) != Some(len)
        {
            return Err(Error::Shape);
        }
        let d = p.data.borrow();
        let used = if words { d.word_used } else { d.float_used };
        let cap = if words {
            p.caps.u64_values
        } else {
            p.caps.f64_values
        };
        if len > cap.saturating_sub(used)
            || d.tensor_used == p.caps.tensors
            || d.tensors[..d.tensor_used].iter().any(|t| t.name == name)
        {
            return Err(Error::Shape);
        }
        Ok(d.tensor_used)
    }
    pub fn add_f64_tensor(
        &mut self,
        name: &str,
        shape: &[usize],
        values: &[f64],
    ) -> Result<usize, Error> {
        let id = self.tensor(name, shape, values.len(), false)?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(Error::NonFinite);
        }
        let p = self.owned.payload();
        let mut d = p.data.borrow_mut();
        let start = d.float_used;
        let end = start + values.len();
        d.floats[start..end].copy_from_slice(values);
        d.float_used = end;
        d.tensors[id].name.clear();
        d.tensors[id].name.push_str(name);
        d.tensors[id].shape.extend_from_slice(shape);
        d.tensors[id].storage = Storage::F64(start..end);
        d.tensor_used += 1;
        p.poisoned.set(false);
        Ok(id)
    }
    pub fn add_u64_tensor(
        &mut self,
        name: &str,
        shape: &[usize],
        values: &[u64],
    ) -> Result<usize, Error> {
        let id = self.tensor(name, shape, values.len(), true)?;
        let p = self.owned.payload();
        let mut d = p.data.borrow_mut();
        let start = d.word_used;
        let end = start + values.len();
        d.words[start..end].copy_from_slice(values);
        d.word_used = end;
        d.tensors[id].name.clear();
        d.tensors[id].name.push_str(name);
        d.tensors[id].shape.extend_from_slice(shape);
        d.tensors[id].storage = Storage::U64(start..end);
        d.tensor_used += 1;
        p.poisoned.set(false);
        Ok(id)
    }
    pub fn add_operation(
        &mut self,
        id: &str,
        operator: ModelOperator,
        layer: Option<u64>,
        inputs: &[&str],
        shape: &[u64],
    ) -> Result<usize, Error> {
        let p = self.start_write(
            1536 + id.len().min(NAME) * (1 + self.owned.payload().caps.operations)
                + inputs
                    .iter()
                    .take(INPUTS)
                    .map(|s| s.len().min(NAME))
                    .sum::<usize>(),
        )?;
        if id.is_empty()
            || id.len() > NAME
            || inputs.len() > INPUTS
            || shape.is_empty()
            || shape.len() > RANK
            || shape.contains(&0)
        {
            return Err(Error::Shape);
        }
        let mut d = p.data.borrow_mut();
        let n = d.op_used;
        if n == p.caps.operations || d.ops[..n].iter().any(|o| o.id == id) {
            return Err(Error::Shape);
        }
        let op = &mut d.ops[n];
        op.id.clear();
        op.id.push_str(id);
        op.operator = operator;
        op.layer = layer;
        for input in inputs {
            op.inputs.push(text(input, NAME)?);
        }
        op.output_shape.extend_from_slice(shape);
        d.op_used += 1;
        p.poisoned.set(false);
        Ok(n)
    }
    pub fn set_attribute(
        &mut self,
        operation: usize,
        key: &str,
        value: Attribute<'_>,
    ) -> Result<(), Error> {
        let p = self.start_write(1024)?;
        if key.is_empty() || key.len() > KEY {
            return Err(Error::Shape);
        }
        let mut d = p.data.borrow_mut();
        if operation >= d.op_used {
            return Err(Error::Shape);
        }
        let attrs = d.ops[operation]
            .attributes
            .as_object_mut()
            .ok_or(Error::Shape)?;
        if attrs.len() == ATTRS || attrs.contains_key(key) {
            return Err(Error::Shape);
        }
        // Allocate the bounded PUBLIC key before any Value/shape allocation.
        // Failure drops public descriptor temporaries normally, not as secrets.
        let key = text(key, KEY)?;
        let value = match value {
            Attribute::Text(s) => Value::String(text(s, NAME)?),
            Attribute::Unsigned(n) => Value::from(n),
            Attribute::Real(n) => Value::Number(Number::from_f64(n).ok_or(Error::NonFinite)?),
            Attribute::Bool(b) => Value::Bool(b),
            Attribute::Shape(s) => {
                if s.len() > RANK || s.contains(&0) {
                    return Err(Error::Shape);
                }
                let mut v = allocated(RANK)?;
                v.extend(s.iter().copied().map(Value::from));
                Value::Array(v)
            }
        };
        attrs.insert(key, value);
        p.poisoned.set(false);
        Ok(())
    }
    pub fn finish(self) -> Result<ClearInputs, Error> {
        if self.owned.payload().poisoned.get() {
            return Err(Error::Burned);
        }
        Ok(ClearInputs { owned: self.owned })
    }
}

impl ClearInputs {
    pub fn remaining_work(&self) -> usize {
        self.owned.payload().ledger.borrow().remaining()
    }
    pub fn consumed_work(&self) -> usize {
        self.owned.payload().work_limit - self.remaining_work()
    }
    pub fn charged_bytes(&self) -> usize {
        self.owned.charged_bytes()
    }
    pub fn tensors(&self) -> Ref<'_, [ClearTensor]> {
        Ref::map(self.owned.payload().data.borrow(), |d| {
            &d.tensors[..d.tensor_used]
        })
    }
    pub fn operations(&self) -> Ref<'_, [ModelOperation]> {
        Ref::map(self.owned.payload().data.borrow(), |d| &d.ops[..d.op_used])
    }
    pub fn f64_tensor(&self, id: usize) -> Result<Ref<'_, [f64]>, Error> {
        let d = self.owned.payload().data.borrow();
        let t = d
            .tensors
            .get(id)
            .filter(|_| id < d.tensor_used)
            .ok_or(Error::Shape)?;
        let Storage::F64(r) = &t.storage else {
            return Err(Error::Shape);
        };
        let r = r.clone();
        Ok(Ref::map(d, |d| &d.floats[r]))
    }
    pub fn u64_tensor(&self, id: usize) -> Result<Ref<'_, [u64]>, Error> {
        let d = self.owned.payload().data.borrow();
        let t = d
            .tensors
            .get(id)
            .filter(|_| id < d.tensor_used)
            .ok_or(Error::Shape)?;
        let Storage::U64(r) = &t.storage else {
            return Err(Error::Shape);
        };
        let r = r.clone();
        Ok(Ref::map(d, |d| &d.words[r]))
    }
    /// Ticket belongs to THIS retained owner. No ledger/guard argument or reset.
    /// Public API/type positive control (also detects missing module registration).
    /// ```
    /// use pllm_compiler::clear_input_owner::{Caps, ClearInputBuilder, ClearInputs, Attempt, BorrowedOutput};
    /// use pllm_garble::shared_allocation::OwnedBudget;
    /// fn construct(budget: OwnedBudget) -> ClearInputs {
    ///     let caps = Caps { f64_values: 2, u64_values: 0, tensors: 1, operations: 0 };
    ///     let mut b = ClearInputBuilder::new(budget, caps, 65536).unwrap();
    ///     b.add_f64_tensor("x", &[2], &[1.0, 2.0]).unwrap();
    ///     b.finish().unwrap()
    /// }
    /// fn retain(input: &ClearInputs) -> Attempt<'_> { input.begin_attempt().unwrap() }
    /// fn inspect(out: &BorrowedOutput<'_, [f64; 1]>) { let _ = out.values(); }
    /// ```
    /// Fresh-ledger injection is not a callable API.
    /// ```compile_fail,E0061
    /// use pllm_compiler::clear_input_owner::ClearInputs;
    /// use pllm_core::row_fold::WorkLedger;
    /// fn reset(input: &ClearInputs) {
    ///     let mut fresh = WorkLedger::new(65536, 65536);
    ///     let _ = input.begin_attempt(&mut fresh);
    /// }
    /// ```
    /// An unrelated reservation cannot be injected either.
    /// ```compile_fail,E0061
    /// use pllm_compiler::clear_input_owner::ClearInputs;
    /// use pllm_garble::shared_allocation::ExternalCharge;
    /// fn replace(input: &ClearInputs, guard: &ExternalCharge<'_>) {
    ///     let _ = input.begin_attempt(guard);
    /// }
    /// ```
    /// Original cannot drop while its real public ticket remains live.
    /// ```compile_fail,E0505
    /// use pllm_compiler::clear_input_owner::ClearInputs;
    /// fn retire(input: ClearInputs) {
    ///     let ticket = input.begin_attempt().unwrap();
    ///     drop(input);
    ///     std::hint::black_box(ticket);
    /// }
    /// ```
    /// No replacement behind a retained immutable numeric view.
    /// ```compile_fail,E0506
    /// use pllm_compiler::clear_input_owner::ClearInputs;
    /// fn replace(input: &mut ClearInputs, replacement: ClearInputs) {
    ///     let view = input.f64_tensor(0).unwrap();
    ///     *input = replacement;
    ///     std::hint::black_box(view);
    /// }
    /// ```
    pub fn begin_attempt(&self) -> Result<Attempt<'_>, Error> {
        let p = self.owned.payload();
        let used = p.attempts.get();
        if used == MAX_ATTEMPTS {
            return Err(Error::Burned);
        }
        p.attempts.set(used + 1); // burn before any fallible validation/debit
        debit(p, 32, 0)?;
        Ok(Attempt { owner: self })
    }
}

pub struct Attempt<'a> {
    owner: &'a ClearInputs,
}
impl<'a> Attempt<'a> {
    /// Narrow TRUSTED compiler adapter seam, not a public callback/proof issuer.
    /// A transient ControlLedger can borrow this SAME original ledger inside the
    /// callback. Returned core prepayments must not borrow the ledger. RefMut ends
    /// here, BEFORE the adapter borrows immutable source buffers. Ticket consumed
    /// even if bounded validation or the original ledger refuses.
    /// Trusted adapters MUST preflight/debit before expensive traversal, NEVER
    /// replace/reset this original ledger, and perform no unpaid allocation.
    /// These obligations require actual consumer-path review; this callback and
    /// its documentation do not prove a consumer has prepaid correctly.
    pub(crate) fn with_ledger<R>(
        self,
        quote: impl FnOnce(&mut WorkLedger) -> Result<R, Error>,
    ) -> Result<(R, Prepared<'a>), Error> {
        let result = quote(&mut self.owner.owned.payload().ledger.borrow_mut())?;
        Ok((result, Prepared { owner: self.owner }))
    }
    /// Internal numeric adapters derive costs from bounded kernel quotes. This is
    /// accounting, NOT proof of their arithmetic or permission to execute code.
    /// Consumes ticket before fallible quote/debit/reservation. No caller ledger.
    #[cfg(test)]
    pub(crate) fn admit<T: Zeroize>(
        self,
        clear: usize,
        control: usize,
        external_bytes: usize,
    ) -> Result<Phase<'a, T>, Error> {
        let (_, prepared) =
            self.with_ledger(|ledger| ledger.debit(clear, control).map_err(|_| Error::Work))?;
        prepared.reserve(external_bytes)
    }
}
pub(crate) struct Prepared<'a> {
    owner: &'a ClearInputs,
}
impl<'a> Prepared<'a> {
    /// Reserve BEFORE allocating workspace/output at their FULL retained
    /// capacities. Adapter must already have debited initialization, execution,
    /// oracle and cleanup work on the same ledger. Every fallible construction
    /// path needs zeroizing RAII declared AFTER the phase, so Result failure,
    /// unwind/cancellation/drop retires buffers BEFORE releasing this charge.
    pub(crate) fn reserve<T: Zeroize>(self, external_bytes: usize) -> Result<Phase<'a, T>, Error> {
        let bytes = external_bytes
            .checked_add(size_of::<BorrowedOutput<'a, T>>())
            .and_then(|b| b.checked_add(size_of::<Self>()))
            .ok_or(Error::Resource)?;
        if bytes > 1 << 20 {
            return Err(Error::Resource);
        }
        let charge = self
            .owner
            .owned
            .budget()
            .charge_external(bytes)
            .map_err(|_| Error::Resource)?;
        Ok(Phase {
            owner: self.owner,
            _charge: charge,
            _type: std::marker::PhantomData,
        })
    }
}
pub(crate) struct Phase<'a, T: Zeroize> {
    owner: &'a ClearInputs,
    _charge: ExternalCharge<'a>,
    _type: std::marker::PhantomData<T>,
}
impl<'a, T: Zeroize> Phase<'a, T> {
    /// Adapter constructs bounded payload ONLY AFTER admit succeeds. No public
    /// callback/adoption path; adapter must inventory all nested capacities.
    /// Buffers need zeroizing RAII from their FIRST allocation, declared AFTER
    /// the phase; calling finish later cannot protect earlier failure paths.
    /// T: Zeroize does NOT guarantee nonpanicking cleanup or that interior borrows
    /// permit a wipe. Consumer review must verify actual all-path cleanup; these
    /// docs and this bound are not a prepayment/erasure proof.
    pub(crate) fn finish(self, payload: T) -> BorrowedOutput<'a, T> {
        BorrowedOutput {
            payload,
            phase: self,
        }
    }
}
/// Public read-only result type; constructors remain trusted-adapter-private.
/// Its lifetime retains the actual original owner, not a caller guard label.
/// ```compile_fail,E0505
/// use pllm_compiler::clear_input_owner::{ClearInputs, BorrowedOutput};
/// fn retire(input: ClearInputs,
///     execute: for<'a> fn(&'a ClearInputs) -> BorrowedOutput<'a, [f64; 1]>) {
///     let output = execute(&input);
///     drop(input);
///     std::hint::black_box(output);
/// }
/// ```
pub struct BorrowedOutput<'a, T: Zeroize> {
    payload: T,
    phase: Phase<'a, T>,
}
impl<T: Zeroize> BorrowedOutput<'_, T> {
    pub fn values(&self) -> &T {
        &self.payload
    }
    pub fn original(&self) -> &ClearInputs {
        self.phase.owner
    }
    #[cfg(test)]
    pub(crate) fn phase_bytes(&self) -> usize {
        self.phase._charge.charged_bytes()
    }
}
impl<T: Zeroize> Drop for BorrowedOutput<'_, T> {
    fn drop(&mut self) {
        self.payload.zeroize();
    }
}

#[cfg(test)]
#[path = "clear_input_owner_tests.rs"]
mod tests;
