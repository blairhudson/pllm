use super::program::{graph_elements, Code, Node};
use super::*;
use pllm_garble::{
    fusefss_backend::{
        composition::{
            cache::{self, SourceTableCache},
            gates, kernels, norm, wavefront, Plan,
        },
        Budget,
    },
    shared_rescale_reference::Context,
};
use sha2::{Digest, Sha256};
use zeroize::{Zeroize, Zeroizing};

#[path = "resource_quote.rs"]
mod resource_quote;

/// External, role-local native executor. Implementations must return exactly one
/// party's Plan output, drive consuming Runtime through externally supplied fresh
/// material and peer frames, and abort their transport on any error. Never issue
/// material or reconstruct inside this callback. Context binds program/source,
/// session, fixed selection, node and monotonically ordered kernel invocation.
pub trait Driver {
    fn run(
        &mut self,
        context: Context,
        plan: &Plan<'_>,
        input: &[u64],
    ) -> Result<NativeOutput, Error>;
    fn cancel(&mut self);
    /// Test-only party-local operator export. Never present in production builds.
    #[cfg(test)]
    fn observe(&mut self, _selection: usize, _node: usize, _shape: &[usize], _shares: &[u64]) {}
}

/// Trusted-preprocessing-only traversal of the public immutable kernel schedule.
/// The visitor receives no private input, registers, selection result, or output.
/// A trusted dealer may issue/export each bounded Plan's material externally.
/// A borrow cannot escape a visit; retained material/transport owners are the
/// visitor's responsibility and must be retired by cancel() on any failure.
pub trait MaterialVisitor {
    fn visit(&mut self, context: Context, plan: &Plan<'_>) -> Result<(), Error>;
    fn cancel(&mut self);
}

impl Program {
    pub fn visit_material(
        &self,
        context: Context,
        visitor: &mut impl MaterialVisitor,
    ) -> Result<(), Error> {
        if context.plan != self.digest
            || context.session == [0; 32]
            || context.operation == [0; 32]
            || context.first_tensor_index == u64::MAX
        {
            visitor.cancel();
            return Err(Error::Context);
        }
        let budget = match request_budget(self) {
            Ok(b) => b,
            Err(e) => {
                visitor.cancel();
                return Err(e);
            }
        };
        let tables = match request_tables(self, &budget) {
            Ok(t) => t,
            Err(e) => {
                visitor.cancel();
                return Err(e);
            }
        };
        let mut state = match Schedule::new(self, context, 0, None, &tables) {
            Ok(s) => s,
            Err(e) => {
                visitor.cancel();
                return Err(e);
            }
        };
        state.visitor = Some(visitor);
        let result = (|| {
            let input = Zeroizing::new(zeros(self.initial)?);
            drop(state.execute(&input)?);
            if state.hash.clone().finalize().as_slice() != self.obligations {
                return Err(Error::Context);
            }
            Ok(())
        })();
        if result.is_err() {
            state.cancel();
        }
        result
    }
}

/// Opaque single-party native result. There is deliberately no constructor from
/// caller-supplied words, digest, claimed range, or reconstructed clear values.
/// Consuming backend Output retires its native lease after a bounded erasing copy.
pub struct NativeOutput {
    values: Zeroizing<Vec<u64>>,
}
impl NativeOutput {
    pub fn from_native(output: pllm_garble::fusefss_backend::Output<'_>) -> Result<Self, Error> {
        if output.shares().is_empty()
            || output.shares().len() > MAX_VALUES
            || output.shares().iter().any(|&v| v > MASK)
        {
            return Err(Error::Domain);
        }
        let values = Zeroizing::new(copied(output.shares())?);
        Ok(Self { values })
    }

    /// A large output is admitted only by the concrete native Plan, never by an
    /// unchecked caller count. Ordinary scalar plans retain their own limits.
    pub fn from_native_plan(
        output: pllm_garble::fusefss_backend::Output<'_>,
        plan: &Plan<'_>,
    ) -> Result<Self, Error> {
        let n = output.shares().len();
        if n == 0 || n > pllm_garble::fusefss_backend::one_hot::MAX_VOCABULARY {
            return Err(Error::Domain);
        }
        for (i, &value) in output.shares().iter().enumerate() {
            if value > MASK || plan.output_is_xor(i)? || plan.output_bits(i)? != 50 {
                return Err(Error::Domain);
            }
        }
        if plan.output_bits(n).is_ok() {
            return Err(Error::Domain);
        }
        let values = Zeroizing::new(copied(output.shares())?);
        Ok(Self { values })
    }
}

/// Final client-only additive token shares. No intermediate retrieval API.
pub struct TokenShares {
    values: Zeroizing<Vec<u64>>,
    validity: Zeroizing<Option<u64>>,
    program: [u8; 32],
    role: u8,
}
impl TokenShares {
    pub fn values(&self) -> &[u64] {
        &self.values
    }
    pub fn program_digest(&self) -> [u8; 32] {
        self.program
    }
    pub fn role(&self) -> u8 {
        self.role
    }
    pub fn validity_share(&self) -> Option<u64> {
        *self.validity
    }
}

/// Consuming one-role execution. Initial input has exactly the public initial
/// length, canonical A50 shares of client-validated IDs in [0,vocabulary). Public
/// masks are generated here; private padding/length inputs are not supported.
pub struct Worker<'p> {
    program: &'p Program,
    role: u8,
    context: Context,
    input: Zeroizing<Vec<u64>>,
}
impl<'p> Worker<'p> {
    pub fn new(
        program: &'p Program,
        role: u8,
        context: Context,
        input: Zeroizing<Vec<u64>>,
    ) -> Result<Self, Error> {
        if role > 1
            || input.len() != program.initial
            || input.iter().any(|&v| v > MASK)
            || context.session == [0; 32]
            || context.operation == [0; 32]
            || context.first_tensor_index == u64::MAX
            || context.plan != program.digest
        {
            return Err(Error::Context);
        }
        if input.capacity() > MAX_VALUES {
            return Err(Error::Resource);
        }
        // Deserializers commonly over-allocate small vectors. Canonicalize a
        // bounded incoming owner fallibly, then erase/retire its old capacity.
        // The compiler's charged gather workspace covers this transient overlap.
        let input = if input.capacity() == input.len() {
            input
        } else {
            let canonical = Zeroizing::new(copied(&input)?);
            drop(input);
            canonical
        };
        Ok(Self {
            program,
            role,
            context,
            input,
        })
    }
    pub fn execute(self, driver: &mut impl Driver) -> Result<TokenShares, Error> {
        let budget = match request_budget(self.program) {
            Ok(b) => b,
            Err(e) => {
                driver.cancel();
                return Err(e);
            }
        };
        let tables = match request_tables(self.program, &budget) {
            Ok(t) => t,
            Err(e) => {
                driver.cancel();
                return Err(e);
            }
        };
        let mut state = match Schedule::new(self.program, self.context, self.role, None, &tables) {
            Ok(s) => s,
            Err(e) => {
                driver.cancel();
                return Err(e);
            }
        };
        state.driver = Some(driver);
        let result = state.execute(&self.input);
        match result {
            Ok(values) if state.hash.clone().finalize().as_slice() == self.program.obligations => {
                Ok(TokenShares {
                    values,
                    validity: Zeroizing::new(
                        self.program.checked_ranges().then_some(*state.validity),
                    ),
                    program: self.program.digest,
                    role: self.role,
                })
            }
            Ok(_) => {
                state.cancel();
                Err(Error::Context)
            }
            Err(e) => {
                state.cancel();
                Err(e)
            }
        }
    }
}

pub(super) fn preflight(program: &mut Program) -> Result<(), Error> {
    let context = Context {
        plan: program.digest,
        session: [1; 32],
        operation: [2; 32],
        first_tensor_index: 0,
    };
    let budget = request_budget(program)?;
    let tables = request_tables(program, &budget)?;
    let mut state = Schedule::new(program, context, 0, None, &tables)?;
    let input = Zeroizing::new(zeros(program.initial)?);
    // This walk has only zero placeholders, no Driver, and no material visitor.
    // Diagnostics belong here, never on the live private execution path.
    let admitted = state.execute(&input);
    if let Err(error) = &admitted {
        if std::env::var_os("PLLM_FSS_ADMISSION_DEBUG").is_some() {
            eprintln!(
                "FSS public kernel admission failed: selection={} node={} code={:?} error={error:?} resources={:?} limits={:?} native_budget_used={} cache={:?}",
                state.selection,
                state.node,
                program.nodes.get(state.node).map(|node| node.code),
                state.resources,
                program.limits,
                budget.used_bytes(),
                tables.stats(),
            );
        }
    }
    drop(admitted?);
    let resources = state.resources;
    let obligations = state.hash.clone().finalize().into();
    program.resources = resources;
    program.obligations = obligations;
    Ok(())
}

fn zeros(n: usize) -> Result<Vec<u64>, Error> {
    let mut v = vector(n)?;
    v.resize(n, 0);
    Ok(v)
}
fn ranges(n: usize, r: Range) -> Result<Vec<gates::InputRange>, Error> {
    let mut v = vector(n)?;
    v.resize(n, r.gate());
    Ok(v)
}
fn checked_add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Resource)
}
fn word(v: i64) -> u64 {
    v as u64 & MASK
}

// Stack-bound request owners: Budget outlives cache and every short-lived Plan.
// No self-reference, global cache, private material reuse, or lifetime extension.
fn request_budget(program: &Program) -> Result<Budget, Error> {
    Budget::new(resource_quote::Quote::available_per_role(
        program.resources.metadata_bytes, program.resources.live_array_bytes_per_party,
    )?)
}

struct RequestTables<'b> {
    cache: SourceTableCache<'b>,
    // One preconstruction ledger seed, retained across the constructor boundary.
    // Schedule copies it once; no retrospective cache-stat repricing/reset.
    bootstrap: Resources,
}
impl<'b> std::ops::Deref for RequestTables<'b> {
    type Target = SourceTableCache<'b>;
    fn deref(&self) -> &Self::Target { &self.cache }
}

trait ScheduleTables<'b> {
    fn cache(&self) -> &SourceTableCache<'b>;
    fn bootstrap(&self) -> Result<Resources, Error>;
}
impl<'b> ScheduleTables<'b> for RequestTables<'b> {
    fn cache(&self) -> &SourceTableCache<'b> { &self.cache }
    fn bootstrap(&self) -> Result<Resources, Error> { Ok(self.bootstrap) }
}
// Existing negative namespace test supplies a raw cache. It must never become
// an unadmitted fixture path for constructing a valid Schedule.
#[cfg(test)]
impl<'b> ScheduleTables<'b> for SourceTableCache<'b> {
    fn cache(&self) -> &SourceTableCache<'b> { self }
    fn bootstrap(&self) -> Result<Resources, Error> { Err(Error::Context) }
}

// Thread-scoped constructor-boundary instrumentation only. Never production
// admission authority; the guard cannot escape or migrate between test threads.
#[cfg(test)]
mod cache_constructor_probe {
    use super::Error;

    thread_local! {
        static ENTRIES: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    }

    struct Reset {
        previous: Option<usize>,
        _thread: std::marker::PhantomData<std::rc::Rc<()>>,
    }
    impl Drop for Reset {
        fn drop(&mut self) {
            ENTRIES.with(|entries| entries.set(self.previous));
        }
    }

    pub(super) fn scoped<T>(f: impl FnOnce() -> T) -> (T, usize) {
        let _reset = Reset {
            previous: ENTRIES.with(|entries| entries.replace(Some(0))),
            _thread: std::marker::PhantomData,
        };
        let result = f();
        let count = ENTRIES.with(|entries| entries.get().expect("scoped constructor probe"));
        (result, count)
    }

    pub(super) fn enter() -> Result<(), Error> {
        ENTRIES.with(|entries| match entries.get() {
            None => Ok(()),
            Some(count) => {
                entries.set(Some(count.checked_add(1).expect("constructor probe count")));
                // Deliberate test sentinel, distinct from Resource and Domain.
                // Stop before constructor reservation, Vec allocation or hash.
                Err(Error::Unsupported)
            }
        })
    }
}

// Reuse the private thread-scoped boundary trap. Wave tests construct request
// tables before entering the scope, so only this later boundary is observed.
#[cfg(test)]
use cache_constructor_probe as wave_constructor_probe;

fn request_tables<'b>(
    program: &Program,
    budget: &'b Budget,
) -> Result<RequestTables<'b>, Error> {
    let limits = cache::Limits {
        max_entries: cache::MAX_ENTRIES,
        max_retained_bytes: cache::MAX_RETAINED_BYTES,
    };
    let descriptor = SourceTableCache::bootstrap_descriptor(program.digest, limits)?;
    // Source::reservation_bytes is an already admitted resident-owner count
    // (including bounded file-read workspace), not disk payload or a new walk.
    resource_quote::Quote::bootstrap(
        program.resources.metadata_bytes,
        program.resources.live_array_bytes_per_party,
        program.source.reservation_bytes(),
        descriptor,
        program.limits,
        |bootstrap| {
            #[cfg(test)]
            cache_constructor_probe::enter()?;
            Ok(RequestTables {
                cache: SourceTableCache::new(program.digest, limits, budget)?,
                bootstrap,
            })
        },
    )
}

struct Schedule<'a, 'd, 'c, 'b> {
    program: &'a Program,
    context: Context,
    role: u8,
    driver: Option<&'d mut dyn Driver>,
    visitor: Option<&'d mut dyn MaterialVisitor>,
    tables: &'c SourceTableCache<'b>,
    accounted_tables: cache::Stats,
    resources: Resources,
    hash: Sha256,
    selection: usize,
    node: usize,
    validity: Zeroizing<u64>,
}
impl<'a, 'd, 'c, 'b> Schedule<'a, 'd, 'c, 'b> {
    fn new(
        program: &'a Program,
        context: Context,
        role: u8,
        driver: Option<&'d mut dyn Driver>,
        owner: &'c impl ScheduleTables<'b>,
    ) -> Result<Self, Error> {
        let tables = owner.cache();
        let mut hash = Sha256::new();
        hash.update(b"pllm.fss-decoder.kernel-obligations.v1\0");
        hash.update(program.digest);
        if tables.namespace() != program.digest {
            return Err(Error::Context);
        }
        let resources = owner.bootstrap()?;
        let table_stats = tables.stats();
        hash.update(b"request-local-shared-source-tables.v1\0");
        hash.update(tables.numeric_revision());
        Ok(Self {
            program,
            context,
            role,
            driver,
            visitor: None,
            tables,
            accounted_tables: table_stats,
            resources,
            hash,
            selection: 0,
            node: 0,
            validity: Zeroizing::new(u64::from(role == 0)),
        })
    }
    fn cancel(&mut self) {
        if let Some(driver) = self.driver.as_mut() {
            driver.cancel();
        }
        if let Some(visitor) = self.visitor.as_mut() {
            visitor.cancel();
        }
    }
    fn binding(&self) -> kernels::Binding {
        kernels::Binding {
            source_digest: self.program.source.digest(),
            contract_digest: self.program.digest,
        }
    }
    fn local(&mut self, n: usize) -> Result<(), Error> {
        self.resources.work = checked_add(
            self.resources.work,
            n.checked_mul(2).ok_or(Error::Resource)?,
        )?;
        if self.resources.work > self.program.limits.max_work {
            return Err(Error::Resource);
        }
        Ok(())
    }
    fn public_copy(&mut self, bytes: usize) -> Result<(), Error> {
        self.resources.public_copy_bytes = checked_add(self.resources.public_copy_bytes, bytes)?;
        if self.resources.public_copy_bytes > self.program.limits.max_public_copy_bytes {
            return Err(Error::Resource);
        }
        Ok(())
    }
    fn run(
        &mut self,
        plan: &Plan<'_>,
        input: &[u64],
        outputs: usize,
        cached_tables: bool,
    ) -> Result<Zeroizing<Vec<u64>>, Error> {
        if input.len() != plan.input_bits().len() || input.iter().any(|&v| v > MASK) {
            return Err(Error::Domain);
        }
        for i in 0..outputs {
            if plan.output_is_xor(i)? || plan.output_bits(i)? != 50 {
                return Err(Error::Domain);
            }
        }
        if plan.output_bits(outputs).is_ok() {
            return Err(Error::Domain);
        }
        let r = plan.resources();
        let wave = if self.program.online.uses_wavefront() && !plan.is_atomic_vector() {
            let backing_work = resource_quote::Quote::wave_backing(
                &mut self.resources, self.program.limits, plan,
            )?;
            #[cfg(test)]
            wave_constructor_probe::enter()?;
            let schedule = wavefront::Schedule::new(plan)?;
            self.hash.update(schedule.digest());
            Some((schedule.resources(), backing_work))
        } else {
            None
        };
        let mut quote = resource_quote::Quote::begin(
            &mut self.resources, self.program.limits, self.program.capacity(),
            r, wave, input.len(), cached_tables,
        )?;
        let tables = self.tables.stats();
        let delta = |now: usize, before: usize| now.checked_sub(before).ok_or(Error::Context);
        let references = delta(tables.references, self.accounted_tables.references)?;
        let misses = delta(tables.misses, self.accounted_tables.misses)?;
        let retained = delta(tables.retained_bytes, self.accounted_tables.retained_bytes)?;
        let copied = delta(
            tables.table_payload_copy_bytes,
            self.accounted_tables.table_payload_copy_bytes,
        )?;
        let lowered_metadata = delta(
            tables.lowered_construction_metadata_bytes,
            self.accounted_tables.lowered_construction_metadata_bytes,
        )?
        .checked_mul(3)
        .ok_or(Error::Resource)?;
        if !cached_tables && references != 0 {
            return Err(Error::Context);
        }
        if cached_tables && (references == 0 || r.public_table_reservation_bytes == 0) {
            return Err(Error::Context);
        }
        quote.tables(tables, references, misses, retained, copied, lowered_metadata)?;
        for count in [
            tables.references,
            tables.hits,
            tables.misses,
            tables.table_constructions,
            tables.table_hashes,
            tables.table_hash_bytes,
            tables.table_payload_copy_bytes,
            tables.lowered_chunk_constructions,
            tables.lowered_payload_bytes,
            tables.lowered_construction_metadata_bytes,
            tables.lowered_retained_bytes,
            tables.generated_words,
            tables.retained_bytes,
            tables.retained_entries,
        ] {
            self.hash.update((count as u64).to_le_bytes());
        }
        self.accounted_tables = tables;
        quote.finish(tables)?;
        self.hash.update((self.selection as u64).to_le_bytes());
        self.hash.update((self.node as u64).to_le_bytes());
        self.hash.update(plan.digest());
        let operation = Sha256::new()
            .chain_update(b"pllm.fss-decoder.invocation.v1\0")
            .chain_update(self.program.digest)
            .chain_update(self.context.operation)
            .chain_update((self.selection as u64).to_le_bytes())
            .chain_update((self.node as u64).to_le_bytes())
            .chain_update((self.resources.kernels as u64).to_le_bytes())
            .finalize()
            .into();
        let context = Context {
            plan: self.program.digest,
            operation,
            ..self.context
        };
        if let Some(visitor) = self.visitor.as_mut() {
            visitor.visit(context, plan)?;
        }
        let out = if let Some(driver) = self.driver.as_mut() {
            driver.run(context, plan, input)?.values
        } else {
            Zeroizing::new(zeros(outputs)?)
        };
        if out.len() != outputs || out.capacity() != out.len() || out.iter().any(|&v| v > MASK) {
            return Err(Error::Domain);
        }
        Ok(out)
    }
    // Every short-lived Plan borrows the external request Budget. The immutable
    // cache survives across kernels; only table owners, never material, persist.
    fn kernel<F>(
        &mut self,
        input: &[u64],
        outputs: usize,
        build: F,
    ) -> Result<Zeroizing<Vec<u64>>, Error>
    where
        F: for<'kernel> FnOnce(&'kernel Budget) -> Result<Plan<'kernel>, Error>,
    {
        let plan = build(self.tables.budget())?;
        let plan = if self.program.online == OnlineSchedule::TensorV1
            && !plan.is_atomic_vector()
            && !plan.is_tensor()
        {
            plan.repeat_tensor(1)?
        } else {
            plan
        };
        self.run(&plan, input, outputs, false)
    }
    fn cached_kernel<F>(
        &mut self,
        input: &[u64],
        outputs: usize,
        build: F,
    ) -> Result<Zeroizing<Vec<u64>>, Error>
    where
        F: FnOnce(&SourceTableCache<'b>) -> Result<Plan<'b>, Error>,
    {
        let plan = build(self.tables)?;
        let plan = if self.program.online == OnlineSchedule::TensorV1
            && !plan.is_atomic_vector()
            && !plan.is_tensor()
        {
            plan.repeat_tensor(1)?
        } else {
            plan
        };
        self.run(&plan, input, outputs, true)
    }

    fn lanes<F, I, O>(
        &mut self,
        count: usize,
        cached: bool,
        mut build: F,
        mut gather: I,
        mut store: O,
    ) -> Result<(), Error>
    where
        F: FnMut(&SourceTableCache<'b>) -> Result<Plan<'b>, Error>,
        I: FnMut(usize, &mut Vec<u64>) -> Result<(), Error>,
        O: FnMut(usize, &[u64]) -> Result<(), Error>,
    {
        let mut start = 0;
        while start < count {
            let template = build(self.tables)?;
            let inputs = template.input_bits().len();
            let outputs = template.output_count();
            let lanes = (count - start)
                .min(self.program.online.lane_bound(&template))
                .min(MAX_VALUES / outputs);
            if lanes == 0 {
                return Err(Error::Resource);
            }
            let plan = self.program.online.repeat(template, lanes)?;
            let mut values = Zeroizing::new(vector(inputs * lanes)?);
            for lane in 0..lanes {
                gather(start + lane, &mut values)?;
                if values.len() != (lane + 1) * inputs {
                    return Err(Error::Context);
                }
            }
            let result = self.run(&plan, &values, outputs * lanes, cached)?;
            for lane in 0..lanes {
                store(start + lane, &result[lane * outputs..(lane + 1) * outputs])?;
            }
            start += lanes;
        }
        Ok(())
    }

    fn floor_values(&mut self, values: &mut [u64], range: Range) -> Result<(), Error> {
        let binding = self.binding();
        let online = self.program.online;
        let chunk_size = if online == OnlineSchedule::TensorV1 {
            wavefront::MAX_TENSOR_ACTIVE
        } else {
            wavefront::MAX_ACTIVE
        };
        for chunk in values.chunks_mut(chunk_size) {
            let mut inputs = Zeroizing::new(vector(chunk.len())?);
            inputs.extend(chunk.iter().map(|&v| v & MASK));
            let n = chunk.len();
            let output = self.kernel(&inputs, n, |b| {
                online.repeat(kernels::floor(binding, range.gate(), 12, b)?.0, n)
            })?;
            chunk.copy_from_slice(&output);
        }
        Ok(())
    }

    fn dot_lanes<I, O>(
        &mut self,
        count: usize,
        inner: usize,
        a: Range,
        b: Range,
        gather: I,
        store: O,
    ) -> Result<(), Error>
    where
        I: FnMut(usize, &mut Vec<u64>) -> Result<(), Error>,
        O: FnMut(usize, &[u64]) -> Result<(), Error>,
    {
        let left = ranges(inner, a)?;
        let right = ranges(inner, b)?;
        let binding = self.binding();
        let proof = self.program.dot_proof(self.node)?;
        if proof.is_some_and(|p| p.inner != inner) {
            return Err(Error::Context);
        }
        self.lanes(
            count,
            false,
            |tables| {
                if let Some(p) = proof {
                    let (plan, metadata) = kernels::private_dot_with_row_energy(
                        binding,
                        &left,
                        &right,
                        [p.left, p.right],
                        tables.budget(),
                    )?;
                    if metadata.raw_range != Some(p.raw.gate()) {
                        return Err(Error::Context);
                    }
                    Ok(plan)
                } else {
                    Ok(kernels::private_dot(binding, &left, &right, tables.budget())?.0)
                }
            },
            gather,
            store,
        )
    }

    fn tree_argmax(&mut self, values: &[u64], range: Range) -> Result<u64, Error> {
        if values.len() == 1 {
            return Ok(0);
        }
        let binding = self.binding();
        let ids = Range::new(0, values.len() as i128 - 1)?.gate();
        let mut frontier = Zeroizing::new(zeros(2 * values.len().div_ceil(2))?);
        let role = self.role;
        // Leaves begin in original vocabulary order. Every level merges
        // adjacent subtrees and carries an odd tail without reordering it.
        // Thus every left leaf precedes every right leaf, making left-on-tie
        // exact without comparing private index payloads again.
        let select = if self.program.online == OnlineSchedule::TensorV1 {
            kernels::max_left_on_tie
        } else {
            kernels::stable_max
        };
        self.lanes(
            values.len() / 2,
            false,
            |t| Ok(select(binding, [range.gate(); 2], ids, t.budget())?.0),
            |i, v| {
                v.extend_from_slice(&[
                    values[2 * i],
                    if role == 0 { 2 * i as u64 } else { 0 },
                    values[2 * i + 1],
                    if role == 0 { (2 * i + 1) as u64 } else { 0 },
                ]);
                Ok(())
            },
            |i, v| {
                frontier[2 * i..2 * i + 2].copy_from_slice(v);
                Ok(())
            },
        )?;
        if values.len() % 2 != 0 {
            let i = values.len() - 1;
            let last = frontier.len() - 2;
            frontier[last] = values[i];
            frontier[last + 1] = if role == 0 { i as u64 } else { 0 };
        }
        while frontier.len() > 2 {
            let nodes = frontier.len() / 2;
            let mut pair = 0;
            while pair < nodes / 2 {
                let template = select(binding, [range.gate(); 2], ids, self.tables.budget())?.0;
                let count = (nodes / 2 - pair).min(self.program.online.lane_bound(&template));
                let plan = self.program.online.repeat(template, count)?;
                let input = Zeroizing::new(copied(&frontier[4 * pair..4 * (pair + count)])?);
                let output = self.run(&plan, &input, 2 * count, false)?;
                frontier[2 * pair..2 * (pair + count)].copy_from_slice(&output);
                pair += count;
            }
            if nodes % 2 != 0 {
                frontier.copy_within(2 * (nodes - 1)..2 * nodes, 2 * (nodes / 2));
            }
            let retained = 2 * nodes.div_ceil(2);
            frontier[retained..].zeroize();
            frontier.truncate(retained);
        }
        Ok(frontier[1])
    }
    fn public_dot(
        &mut self,
        x: &[u64],
        range: Range,
        weights: &[i64],
        bias: i64,
    ) -> Result<u64, Error> {
        // Each role gathers one public coefficient row while the immutable
        // checkpoint owner remains live. Account cumulative copies as well as
        // the already-reserved bounded gather workspace.
        self.public_copy(weights.len().checked_mul(16).ok_or(Error::Resource)?)?;
        let rs = ranges(x.len(), range)?;
        let lo = *weights.iter().min().ok_or(Error::Domain)?;
        let hi = *weights.iter().max().ok_or(Error::Domain)?;
        let binding = self.binding();
        let out = self.kernel(x, 1, |b| {
            Ok(kernels::public_dot(
                binding,
                &rs,
                weights,
                gates::InputRange { min: lo, max: hi },
                b,
            )?
            .0)
        })?;
        self.local(1)?;
        Ok(out[0].wrapping_add(if self.role == 0 { word(bias) } else { 0 }) & MASK)
    }
    fn private_dot(
        &mut self,
        left: &[u64],
        right: &[u64],
        a: Range,
        b: Range,
    ) -> Result<u64, Error> {
        let lr = ranges(left.len(), a)?;
        let rr = ranges(right.len(), b)?;
        let mut input = Zeroizing::new(vector(left.len() + right.len())?);
        input.extend_from_slice(left);
        input.extend_from_slice(right);
        let binding = self.binding();
        let proof = self.program.dot_proof(self.node)?;
        if proof.is_some_and(|proof| proof.inner != left.len() || left.len() != right.len()) {
            return Err(Error::Context);
        }
        Ok(self.kernel(&input, 1, |budget| {
            if let Some(proof) = proof {
                let (plan, metadata) = kernels::private_dot_with_row_energy(
                    binding,
                    &lr,
                    &rr,
                    [proof.left, proof.right],
                    budget,
                )?;
                if metadata.raw_range != Some(proof.raw.gate()) {
                    return Err(Error::Context);
                }
                Ok(plan)
            } else {
                Ok(kernels::private_dot(binding, &lr, &rr, budget)?.0)
            }
        })?[0])
    }
    fn execute(&mut self, input: &[u64]) -> Result<Zeroizing<Vec<u64>>, Error> {
        let p = self.program;
        let geometry = pllm_models::DecoderPlan::validated_phase_geometry(p.initial, p.selections)
            .ok_or(Error::Domain)?;
        if geometry.padded_rows() != p.sequence {
            return Err(Error::Domain);
        }
        let mut tokens = Zeroizing::new(zeros(p.sequence)?);
        tokens[..p.initial].copy_from_slice(input);
        let mut finals = Zeroizing::new(vector(p.selections)?);
        for selection in 0..p.selections {
            self.selection = selection;
            let prefix = geometry.active_rows(selection).ok_or(Error::Domain)?;
            let mut arrays = vector::<Option<Zeroizing<Vec<u64>>>>(p.nodes.len())?;
            arrays.resize_with(p.nodes.len(), || None);
            for (index, node) in p.nodes.iter().enumerate() {
                self.node = index;
                let values = self.node(node, &arrays, &tokens, prefix)?;
                if values.len() != graph_elements(p.capacity(), &node.shape)? {
                    return Err(Error::Domain);
                }
                #[cfg(test)]
                if let Some(driver) = self.driver.as_mut() {
                    driver.observe(selection, index, &node.shape, &values);
                }
                arrays[index] = Some(values);
                for (j, array) in arrays.iter_mut().enumerate().take(index) {
                    if p.nodes[j].last_use == index {
                        *array = None;
                    }
                }
            }
            let token = Zeroizing::new(
                arrays
                    .last()
                    .and_then(|a| a.as_ref())
                    .ok_or(Error::Domain)?[0],
            );
            finals.push(*token);
            if selection + 1 < p.selections {
                tokens[prefix] = *token;
            }
        }
        if p.checked_ranges() {
            // Full private feedback has already finished. One final, unscaled
            // mask protects EVERY selection, including outputs before a failure.
            self.node = p.nodes.len();
            let mut input = Zeroizing::new(vector(finals.len() + 1)?);
            input.extend_from_slice(&finals);
            input.push(*self.validity);
            let rs = ranges(finals.len(), Range::new(0, p.vocabulary as i128 - 1)?)?;
            let binding = self.binding();
            let masked = self.kernel(&input, finals.len(), |budget| {
                Ok(kernels::mask_by_validity(binding, &rs, budget)?.0)
            })?;
            finals.copy_from_slice(&masked);
        }
        Ok(finals)
    }
    fn node(
        &mut self,
        node: &Node,
        arrays: &[Option<Zeroizing<Vec<u64>>>],
        tokens: &[u64],
        prefix: usize,
    ) -> Result<Zeroizing<Vec<u64>>, Error> {
        let p = self.program;
        let t = p.sequence;
        let get = |i: usize| -> Result<&[u64], Error> {
            arrays
                .get(i)
                .and_then(|v| v.as_ref())
                .map(|v| v.as_slice())
                .ok_or(Error::Domain)
        };
        let mut out = Zeroizing::new(zeros(graph_elements(p.capacity(), &node.shape)?)?);
        use Code::*;
        match node.code {
            Embedding { weight, positions } => {
                let w = &p.source.tensors[weight];
                let columns = w.shape[1];
                let rows = w.shape[0];
                if positions {
                    if self.role == 0 {
                        if w.is_file_backed() {
                            // Only this role copies position rows; unused table
                            // rows and the other holder cause no coefficient read.
                            self.public_copy(
                                t.checked_mul(columns)
                                    .and_then(|n| n.checked_mul(8))
                                    .ok_or(Error::Resource)?,
                            )?;
                        }
                        let mut coefficients = Zeroizing::new(vector::<i64>(columns)?);
                        coefficients.resize(columns, 0);
                        for i in 0..t {
                            w.read_row(i, 0, &mut coefficients)?;
                            for c in 0..columns {
                                out[i * columns + c] = word(coefficients[c]);
                            }
                        }
                    }
                } else {
                    let binding = self.binding();
                    let id_range = Range::new(0, p.vocabulary as i128 - 1)?.gate();
                    if let Some(file) = w.fixed_file() {
                        let batch = p.embedding_batch_rows()?;
                        // One complete table scan serves all padded rows in each
                        // bounded batch, on both holders. Price the whole node
                        // BEFORE any kernel/read, including dry/dealer walks.
                        self.public_copy(
                            t.div_ceil(batch)
                                .checked_mul(rows)
                                .and_then(|n| n.checked_mul(columns))
                                .and_then(|n| n.checked_mul(16))
                                .ok_or(Error::Resource)?,
                        )?;
                        // Only workers evaluate the public contraction. Dealer
                        // and dry admission retain no multi-row selector owner;
                        // their identical fresh kernel walk still prices both
                        // workers' actual gather/copy work below.
                        let mut selectors = if self.driver.is_some() {
                            Some(Zeroizing::new(zeros(
                                batch.checked_mul(rows).ok_or(Error::Resource)?,
                            )?))
                        } else {
                            None
                        };
                        for start in (0..t).step_by(batch) {
                            let count = batch.min(t - start);
                            for i in 0..count {
                                // Identical order and fresh invocation to the
                                // single-row schedule; no key/material reuse.
                                let token = start + i;
                                let hot =
                                    self.kernel(&tokens[token..token + 1], rows, |budget| {
                                        Ok(kernels::compact_token_one_hot(
                                            binding, id_range, rows, budget,
                                        )?
                                        .0)
                                    })?;
                                if let Some(selectors) = selectors.as_mut() {
                                    selectors[i * rows..(i + 1) * rows].copy_from_slice(&hot);
                                }
                            }
                            if let Some(selectors) = selectors.as_ref() {
                                file.accumulate_rows_ring64(
                                    &[&selectors[..count * rows]],
                                    &mut [&mut out[start * columns..(start + count) * columns]],
                                    count,
                                    0,
                                    rows,
                                )
                                .map_err(|_| Error::Context)?;
                                for value in &mut out[start * columns..(start + count) * columns] {
                                    *value &= MASK;
                                }
                            }
                        }
                        // Gathering private selector words is real local copy
                        // work in addition to all original multiply/add work.
                        self.local(t.checked_mul(rows).ok_or(Error::Resource)?)?;
                    } else {
                        for i in 0..t {
                            let hot = self.kernel(&tokens[i..i + 1], rows, |budget| {
                                Ok(if p.capacity() == Capacity::Decoder {
                                    kernels::compact_token_one_hot(binding, id_range, rows, budget)?
                                        .0
                                } else {
                                    kernels::token_one_hot(binding, id_range, rows, budget)?.0
                                })
                            })?;
                            for row in 0..rows {
                                for c in 0..columns {
                                    let v =
                                        hot[row].wrapping_mul(word(w.values[row * columns + c]));
                                    out[i * columns + c] =
                                        out[i * columns + c].wrapping_add(v) & MASK;
                                }
                            }
                        }
                    }
                }
                self.local(
                    t.checked_mul(rows)
                        .and_then(|n| n.checked_mul(columns))
                        .ok_or(Error::Resource)?,
                )?;
            }
            Linear {
                input,
                weight,
                bias,
                input_output,
            } => {
                let x = get(input)?;
                let w = &p.source.tensors[weight];
                let inner = *p.nodes[input].shape.last().unwrap();
                let columns = *node.shape.last().unwrap();
                if p.online.uses_wavefront() {
                    let mut raw_range: Option<Range> = None;
                    if let Some(file) = w.fixed_file() {
                        if input_output {
                            return Err(Error::Unsupported);
                        }
                        self.public_copy(checked_add(
                            w.len().checked_mul(16).ok_or(Error::Resource)?,
                            columns
                                .checked_mul(3 * pllm_core::fixed_file::ROW_STATISTICS_RECORD_BYTES)
                                .ok_or(Error::Resource)?,
                        )?)?;
                        for col in 0..columns {
                            let r = p.linear_raw_range(self.node, col)?;
                            raw_range = Some(raw_range.map_or(r, |v| Range {
                                lo: v.lo.min(r.lo),
                                hi: v.hi.max(r.hi),
                            }));
                        }
                        if self.driver.is_some() {
                            file.multiply_ring64(
                                &[x],
                                &mut [out.as_mut_slice()],
                                x.len() / inner,
                                pllm_core::streamed_matrix::MatrixLayout::OutputInput,
                            )
                            .map_err(|_| Error::Context)?;
                        }
                    } else {
                        self.public_copy(
                            w.len()
                                .checked_mul(8 * (2 * x.len() / inner + 3))
                                .ok_or(Error::Resource)?,
                        )?;
                        for col in 0..columns {
                            let mut r = Range::new(0, 0)?;
                            for k in 0..inner {
                                let coefficient = w.values[if input_output {
                                    k * columns + col
                                } else {
                                    col * inner + k
                                }];
                                r = r.add(p.nodes[input].range.scale(coefficient)?)?;
                                if self.driver.is_some() {
                                    for row in 0..x.len() / inner {
                                        let index = row * columns + col;
                                        out[index] = out[index].wrapping_add(
                                            x[row * inner + k].wrapping_mul(word(coefficient)),
                                        ) & MASK;
                                    }
                                }
                            }
                            raw_range = Some(raw_range.map_or(r, |v| Range {
                                lo: v.lo.min(r.lo),
                                hi: v.hi.max(r.hi),
                            }));
                        }
                    }
                    self.local(x.len().checked_mul(columns).ok_or(Error::Resource)?)?;
                    self.floor_values(&mut out, raw_range.ok_or(Error::Domain)?)?;
                    if let Some(b) = bias {
                        let bias = p.source.tensors[b]
                            .resident_values()
                            .ok_or(Error::Context)?;
                        if self.role == 0 {
                            for (i, v) in out.iter_mut().enumerate() {
                                *v = v.wrapping_add(word(bias[i % columns])) & MASK;
                            }
                        }
                        self.local(out.len())?;
                    }
                    return Ok(out);
                }
                if let Some(file) = w.fixed_file() {
                    use pllm_core::streamed_matrix::MatrixLayout;
                    if input_output {
                        // Stored row certificates must never be treated as
                        // certificates for the transposed contraction columns.
                        return Err(Error::Unsupported);
                    }
                    // One coefficient scan per holder serves ALL input rows.
                    // One immutable certificate per output column serves all
                    // rows on both holders AND the dealer. Its raw interval is
                    // compiler-bound to this input's complete-row proof, which
                    // holds uniformly for every token row. Each output retains
                    // its independent fresh floor invocation.
                    // Source-import/hash and compiler-only range reads are not
                    // execution reads and must not be multiplied by selections.
                    let coefficients = w.len().checked_mul(16).ok_or(Error::Resource)?;
                    let statistics = columns
                        .checked_mul(pllm_core::fixed_file::ROW_STATISTICS_RECORD_BYTES)
                        .and_then(|n| n.checked_mul(3))
                        .ok_or(Error::Resource)?;
                    self.public_copy(checked_add(coefficients, statistics)?)?;
                    if self.driver.is_some() {
                        file.multiply_ring64(
                            &[x],
                            &mut [out.as_mut_slice()],
                            x.len() / inner,
                            MatrixLayout::OutputInput,
                        )
                        .map_err(|_| Error::Context)?;
                    }
                    self.local(x.len().checked_mul(columns).ok_or(Error::Resource)?)?;
                    let binding = self.binding();
                    for col in 0..columns {
                        let raw_range = p.linear_raw_range(self.node, col)?;
                        let bias = if let Some(b) = bias {
                            *p.source.tensors[b]
                                .resident_values()
                                .and_then(|v| v.get(col))
                                .ok_or(Error::Domain)?
                        } else {
                            0
                        };
                        for row in 0..x.len() / inner {
                            let index = row * columns + col;
                            let raw = [out[index] & MASK];
                            let value = self.kernel(&raw, 1, |budget| {
                                Ok(kernels::floor(binding, raw_range.gate(), 12, budget)?.0)
                            })?[0];
                            out[index] =
                                value.wrapping_add(if self.role == 0 { word(bias) } else { 0 })
                                    & MASK;
                        }
                    }
                    return Ok(out);
                }
                let mut coefficients = Zeroizing::new(vector(inner)?);
                for row in 0..x.len() / inner {
                    for col in 0..columns {
                        coefficients.clear();
                        for k in 0..inner {
                            coefficients.push(
                                w.values[if input_output {
                                    k * columns + col
                                } else {
                                    col * inner + k
                                }],
                            );
                        }
                        out[row * columns + col] = self.public_dot(
                            &x[row * inner..(row + 1) * inner],
                            p.nodes[input].range,
                            &coefficients,
                            bias.map_or(0, |b| p.source.tensors[b].values[col]),
                        )?;
                    }
                }
            }
            Norm {
                input,
                gamma,
                beta,
                layer,
            } => {
                let x = get(input)?;
                let gamma = &p.source.tensors[gamma].values;
                let width = gamma.len();
                let zeros = Zeroizing::new({
                    let mut v = vector(width)?;
                    v.resize(width, 0i64);
                    v
                });
                let beta = beta.map_or(zeros.as_slice(), |b| p.source.tensors[b].values.as_slice());
                let original_range = p.nodes[input].range;
                let range = if p.checked_ranges() {
                    super::energy::rms_guard(width)?
                } else {
                    original_range
                };
                let source = p.source.digest();
                let input_energy = p.norm_row_energy(self.node)?;
                if p.online.uses_wavefront() && !p.checked_ranges() {
                    let spec = norm::Spec {
                        kind: if layer {
                            norm::Kind::Layer
                        } else {
                            norm::Kind::Rms
                        },
                        input_min: range.lo,
                        input_max: range.hi,
                        gamma,
                        beta,
                    };
                    self.lanes(
                        x.len() / width,
                        true,
                        |tables| {
                            if let Some(energy) = input_energy {
                                norm::compile_cached_with_row_energy(source, &spec, energy, tables)
                            } else {
                                norm::compile_cached(source, &spec, tables)
                            }
                        },
                        |row, values| {
                            values.extend_from_slice(&x[row * width..(row + 1) * width]);
                            Ok(())
                        },
                        |row, values| {
                            out[row * width..(row + 1) * width].copy_from_slice(values);
                            Ok(())
                        },
                    )?;
                    return Ok(out);
                }
                for row in 0..x.len() / width {
                    let guarded = if p.checked_ranges() {
                        let binding = self.binding();
                        let rs = ranges(width, original_range)?;
                        let values =
                            self.kernel(&x[row * width..(row + 1) * width], width + 1, |budget| {
                                let check = if p.online.uses_wavefront() {
                                    kernels::checked_range_balanced
                                } else {
                                    kernels::checked_range_with_fallback
                                };
                                Ok(check(
                                    binding,
                                    &rs,
                                    range.gate(),
                                    kernels::RangeFallback::WholeRow,
                                    budget,
                                )?
                                .0)
                            })?;
                        let pair = Zeroizing::new([*self.validity, values[width]]);
                        let combined = self.kernel(&pair[..], 1, |budget| {
                            Ok(kernels::validity_and(binding, budget)?.0)
                        })?;
                        *self.validity = combined[0];
                        Some(values)
                    } else {
                        None
                    };
                    let row_input = guarded
                        .as_ref()
                        .map_or(&x[row * width..(row + 1) * width], |v| &v[..width]);
                    let spec = norm::Spec {
                        kind: if layer {
                            norm::Kind::Layer
                        } else {
                            norm::Kind::Rms
                        },
                        input_min: range.lo,
                        input_max: range.hi,
                        gamma,
                        beta,
                    };
                    let y =
                        self.cached_kernel(row_input, width, |tables| {
                            let plan = if let Some(energy) = input_energy {
                                norm::compile_cached_with_row_energy(source, &spec, energy, tables)
                            } else {
                                norm::compile_cached(source, &spec, tables)
                            };
                            if plan.is_err() && std::env::var_os("PLLM_FSS_ADMISSION_DEBUG").is_some() {
                                eprintln!("fss norm rejection: width={width} input={range:?} row_energy={input_energy:?} layer={layer}");
                            }
                            plan
                        })?;
                    out[row * width..(row + 1) * width].copy_from_slice(&y);
                }
            }
            Add(a, b) => {
                let a = get(a)?;
                let b = get(b)?;
                for i in 0..out.len() {
                    out[i] = a[i].wrapping_add(b[i]) & MASK;
                }
                self.local(out.len())?;
            }
            Product(a, b) => {
                let x = get(a)?;
                let y = get(b)?;
                if p.online.uses_wavefront() {
                    self.dot_lanes(
                        out.len(),
                        1,
                        p.nodes[a].range,
                        p.nodes[b].range,
                        |i, v| {
                            v.extend_from_slice(&[x[i], y[i]]);
                            Ok(())
                        },
                        |i, v| {
                            out[i] = v[0];
                            Ok(())
                        },
                    )?;
                    return Ok(out);
                }
                for i in 0..out.len() {
                    out[i] = self.private_dot(
                        &x[i..i + 1],
                        &y[i..i + 1],
                        p.nodes[a].range,
                        p.nodes[b].range,
                    )?;
                }
            }
            Activation { input, silu } => {
                let x = get(input)?;
                let range = p.nodes[input].range.gate();
                let binding = self.binding();
                if p.online.uses_wavefront() {
                    self.lanes(
                        x.len(),
                        silu,
                        |tables| {
                            if silu {
                                Ok(gates::silu_cached(range, tables)?.0)
                            } else {
                                Ok(kernels::gelu(binding, range, tables.budget())?.0)
                            }
                        },
                        |i, v| {
                            v.push(x[i]);
                            Ok(())
                        },
                        |i, v| {
                            out[i] = v[0];
                            Ok(())
                        },
                    )?;
                    return Ok(out);
                }
                for i in 0..x.len() {
                    out[i] = if silu {
                        self.cached_kernel(&x[i..i + 1], 1, |tables| {
                            Ok(gates::silu_cached(range, tables)?.0)
                        })?
                    } else {
                        self.kernel(&x[i..i + 1], 1, |budget| {
                            Ok(kernels::gelu(binding, range, budget)?.0)
                        })?
                    }[0];
                }
            }
            Slice { input, start } => {
                let x = get(input)?;
                let width = *p.nodes[input].shape.last().unwrap();
                let columns = *node.shape.last().unwrap();
                for row in 0..out.len() / columns {
                    out[row * columns..(row + 1) * columns]
                        .copy_from_slice(&x[row * width + start..row * width + start + columns]);
                }
                self.local(out.len())?;
            }
            Heads { input, split } => {
                let x = get(input)?;
                let shape = if split {
                    &node.shape
                } else {
                    &p.nodes[input].shape
                };
                let heads = shape[1];
                let d = shape[3];
                for h in 0..heads {
                    for s in 0..t {
                        for c in 0..d {
                            let head = (h * t + s) * d + c;
                            let flat = (s * heads + h) * d + c;
                            if split {
                                out[head] = x[flat];
                            } else {
                                out[flat] = x[head];
                            }
                        }
                    }
                }
                self.local(out.len())?;
            }
            Rotary { input, descriptor } => {
                let x = get(input)?;
                let d = node.shape[3];
                let r = &p.source.rotary[descriptor];
                let original = p.nodes[input].range;
                let binding = self.binding();
                if p.online.uses_wavefront() {
                    let pairs = x.len() / 2;
                    for start in (0..pairs).step_by(wavefront::MAX_ACTIVE / 4) {
                        let count = (pairs - start).min(wavefront::MAX_ACTIVE / 4);
                        let mut terms = Zeroizing::new(vector(4 * count)?);
                        let mut bound: Option<Range> = None;
                        for pair in start..start + count {
                            let c = pair % (d / 2);
                            let row = pair / (d / 2);
                            let s = row % t;
                            let i = row * d + c;
                            let j = i + d / 2;
                            let cosine = r.cos[s * d + c];
                            let sine = r.sin[s * d + c];
                            for (value, coefficient) in
                                [(x[i], cosine), (x[j], -sine), (x[j], cosine), (x[i], sine)]
                            {
                                let domain = original.scale(coefficient)?;
                                bound = Some(bound.map_or(domain, |b| Range {
                                    lo: b.lo.min(domain.lo),
                                    hi: b.hi.max(domain.hi),
                                }));
                                terms.push(value.wrapping_mul(word(coefficient)) & MASK);
                            }
                        }
                        self.floor_values(&mut terms, bound.ok_or(Error::Domain)?)?;
                        for (lane, pair) in (start..start + count).enumerate() {
                            let i = pair / (d / 2) * d + pair % (d / 2);
                            out[i] = terms[4 * lane].wrapping_add(terms[4 * lane + 1]) & MASK;
                            out[i + d / 2] =
                                terms[4 * lane + 2].wrapping_add(terms[4 * lane + 3]) & MASK;
                        }
                    }
                    self.local(3 * x.len())?;
                    self.public_copy(16 * pairs)?;
                    return Ok(out);
                }
                for h in 0..node.shape[1] {
                    for s in 0..t {
                        for c in 0..d / 2 {
                            let i = (h * t + s) * d + c;
                            let j = i + d / 2;
                            let pair = Zeroizing::new([x[i], x[j]]);
                            let cosine = r.cos[s * d + c];
                            let sine = r.sin[s * d + c];
                            let y = self.kernel(&pair[..], 2, |budget| {
                                Ok(kernels::rope(
                                    binding,
                                    [original.gate(); 2],
                                    cosine,
                                    sine,
                                    budget,
                                )?
                                .0)
                            })?;
                            out[i] = y[0];
                            out[j] = y[1];
                        }
                    }
                }
                self.local(out.len())?;
            }
            Identity(input) | Mask(input) => {
                out.copy_from_slice(get(input)?);
                self.local(out.len())?;
            }
            Scores { left, right, group } => {
                let a = get(left)?;
                let b = get(right)?;
                let d = p.nodes[left].shape[3];
                let heads = node.shape[1];
                if p.online.uses_wavefront() {
                    self.dot_lanes(
                        out.len(),
                        d,
                        p.nodes[left].range,
                        p.nodes[right].range,
                        |i, v| {
                            let h = i / (t * t);
                            let q = i / t % t;
                            let k = i % t;
                            let ai = (h * t + q) * d;
                            let bi = (h / group * t + k) * d;
                            v.extend_from_slice(&a[ai..ai + d]);
                            v.extend_from_slice(&b[bi..bi + d]);
                            Ok(())
                        },
                        |i, v| {
                            out[i] = v[0];
                            Ok(())
                        },
                    )?;
                    return Ok(out);
                }
                for h in 0..heads {
                    for q in 0..t {
                        for k in 0..t {
                            let ai = (h * t + q) * d;
                            let bi = (h / group * t + k) * d;
                            out[(h * t + q) * t + k] = self.private_dot(
                                &a[ai..ai + d],
                                &b[bi..bi + d],
                                p.nodes[left].range,
                                p.nodes[right].range,
                            )?;
                        }
                    }
                }
            }
            Scale { input, coefficient } => {
                let x = get(input)?;
                if p.online.uses_wavefront() {
                    let binding = self.binding();
                    let r = gates::InputRange {
                        min: coefficient,
                        max: coefficient,
                    };
                    self.lanes(
                        x.len(),
                        false,
                        |t| {
                            Ok(kernels::public_scale_q12(
                                binding,
                                p.nodes[input].range.gate(),
                                coefficient,
                                r,
                                t.budget(),
                            )?
                            .0)
                        },
                        |i, v| {
                            v.push(x[i]);
                            Ok(())
                        },
                        |i, v| {
                            out[i] = v[0];
                            Ok(())
                        },
                    )?;
                    return Ok(out);
                }
                for i in 0..out.len() {
                    out[i] =
                        self.public_dot(&x[i..i + 1], p.nodes[input].range, &[coefficient], 0)?;
                }
            }
            Softmax(input) => {
                let x = get(input)?;
                let heads = node.shape[1];
                let rs = ranges(t, p.nodes[input].range)?;
                let mut active = vector(t)?;
                if p.online.uses_wavefront() {
                    for q in 0..t {
                        active.clear();
                        active.extend((0..t).map(|k| k <= q && k < prefix));
                        self.lanes(
                            heads,
                            true,
                            |tables| Ok(gates::softmax_cached(&rs, &active, tables)?.0),
                            |head, values| {
                                let start = (head * t + q) * t;
                                values.extend_from_slice(&x[start..start + t]);
                                Ok(())
                            },
                            |head, values| {
                                let start = (head * t + q) * t;
                                out[start..start + t].copy_from_slice(values);
                                Ok(())
                            },
                        )?;
                    }
                    return Ok(out);
                }
                for h in 0..heads {
                    for q in 0..t {
                        active.clear();
                        active.extend((0..t).map(|k| k <= q && k < prefix));
                        let start = (h * t + q) * t;
                        let y = self.cached_kernel(&x[start..start + t], t, |tables| {
                            Ok(gates::softmax_cached(&rs, &active, tables)?.0)
                        })?;
                        out[start..start + t].copy_from_slice(&y);
                    }
                }
            }
            Values { left, right, group } => {
                let a = get(left)?;
                let b = get(right)?;
                let heads = node.shape[1];
                let d = node.shape[3];
                if p.online.uses_wavefront() {
                    self.dot_lanes(
                        out.len(),
                        t,
                        p.nodes[left].range,
                        p.nodes[right].range,
                        |i, v| {
                            let h = i / (t * d);
                            let q = i / d % t;
                            let c = i % d;
                            v.extend_from_slice(&a[(h * t + q) * t..(h * t + q + 1) * t]);
                            for k in 0..t {
                                v.push(b[(h / group * t + k) * d + c]);
                            }
                            Ok(())
                        },
                        |i, v| {
                            out[i] = v[0];
                            Ok(())
                        },
                    )?;
                    return Ok(out);
                }
                let mut column = Zeroizing::new(zeros(t)?);
                for h in 0..heads {
                    for q in 0..t {
                        for c in 0..d {
                            for k in 0..t {
                                column[k] = b[(h / group * t + k) * d + c];
                            }
                            out[(h * t + q) * d + c] = self.private_dot(
                                &a[(h * t + q) * t..(h * t + q + 1) * t],
                                &column,
                                p.nodes[left].range,
                                p.nodes[right].range,
                            )?;
                        }
                    }
                }
            }
            Last(input) => {
                let x = get(input)?;
                let width = out.len();
                out.copy_from_slice(&x[(prefix - 1) * width..prefix * width]);
                self.local(width)?;
            }
            Argmax(input) => {
                let x = get(input)?;
                if p.online.uses_wavefront() {
                    out[0] = self.tree_argmax(x, p.nodes[input].range)?;
                    return Ok(out);
                }
                let binding = self.binding();
                if p.capacity() == Capacity::Decoder {
                    let range = p.nodes[input].range.gate();
                    let ids = Range::new(0, x.len() as i128 - 1)?.gate();
                    let mut winner = Zeroizing::new([x[0], 0]);
                    for (index, &value) in x.iter().enumerate().skip(1) {
                        let input = Zeroizing::new([
                            winner[0],
                            winner[1],
                            value,
                            if self.role == 0 { index as u64 } else { 0 },
                        ]);
                        let selected = self.kernel(input.as_slice(), 2, |budget| {
                            Ok(kernels::stable_max(binding, [range; 2], ids, budget)?.0)
                        })?;
                        winner.copy_from_slice(&selected);
                    }
                    out[0] = winner[1];
                } else {
                    let rs = ranges(x.len(), p.nodes[input].range)?;
                    out[0] = self
                        .kernel(x, 1, |budget| Ok(kernels::argmax(binding, &rs, budget)?.0))?[0];
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

#[cfg(test)]
#[path = "table_cache_tests.rs"]
mod table_cache_tests;

#[cfg(test)]
#[path = "runtime_accounting_tests.rs"]
mod runtime_accounting_tests;
