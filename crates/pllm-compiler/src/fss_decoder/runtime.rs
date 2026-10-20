use super::program::{Code, Node};
use super::*;
use pllm_garble::{
    fusefss_backend::{
        composition::{
            cache::{self, SourceTableCache},
            gates, kernels, norm, Plan,
        },
        Budget,
    },
    shared_rescale_reference::Context,
};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

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
}

/// Final client-only additive token shares. No intermediate retrieval API.
pub struct TokenShares {
    values: Zeroizing<Vec<u64>>,
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
    drop(state.execute(&input)?);
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
    let compiler = checked_add(
        program
            .resources
            .metadata_bytes
            .checked_mul(3)
            .ok_or(Error::Resource)?,
        program
            .resources
            .live_array_bytes_per_party
            .checked_mul(2)
            .ok_or(Error::Resource)?,
    )?;
    let available = pllm_garble::fusefss_backend::MAX_ALLOCATION_BYTES
        .checked_sub(compiler)
        .filter(|&n| n > 0)
        .ok_or(Error::Resource)?;
    Budget::new(available / 3)
}
fn request_tables<'b>(
    program: &Program,
    budget: &'b Budget,
) -> Result<SourceTableCache<'b>, Error> {
    SourceTableCache::new(
        program.digest,
        cache::Limits {
            max_entries: cache::MAX_ENTRIES,
            max_retained_bytes: cache::MAX_RETAINED_BYTES,
        },
        budget,
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
}
impl<'a, 'd, 'c, 'b> Schedule<'a, 'd, 'c, 'b> {
    fn new(
        program: &'a Program,
        context: Context,
        role: u8,
        driver: Option<&'d mut dyn Driver>,
        tables: &'c SourceTableCache<'b>,
    ) -> Result<Self, Error> {
        let compiler = checked_add(
            program
                .resources
                .metadata_bytes
                .checked_mul(3)
                .ok_or(Error::Resource)?,
            program
                .resources
                .live_array_bytes_per_party
                .checked_mul(2)
                .ok_or(Error::Resource)?,
        )?;
        let available = pllm_garble::fusefss_backend::MAX_ALLOCATION_BYTES
            .checked_sub(compiler)
            .filter(|&n| n > 0)
            .ok_or(Error::Resource)?;
        let mut hash = Sha256::new();
        hash.update(b"pllm.fss-decoder.kernel-obligations.v1\0");
        hash.update(program.digest);
        let _ = Budget::new(available / 3)?;
        if tables.namespace() != program.digest {
            return Err(Error::Context);
        }
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
            resources: Resources {
                work: program
                    .resources
                    .metadata_bytes
                    .checked_mul(3)
                    .and_then(|n| n.checked_add(3 * table_stats.metadata_reservation_bytes))
                    .and_then(|n| n.checked_add(3 * table_stats.numeric_revision_hash_bytes))
                    .ok_or(Error::Resource)?,
                metadata_bytes: program.resources.metadata_bytes,
                live_array_bytes_per_party: program.resources.live_array_bytes_per_party,
                source_table_cache_bytes_per_role: table_stats.metadata_reservation_bytes,
                public_copy_bytes: program
                    .source
                    .reservation_bytes()
                    .checked_mul(3)
                    .ok_or(Error::Resource)?,
                ..Resources::default()
            },
            hash,
            selection: 0,
            node: 0,
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
        self.resources.kernels = checked_add(self.resources.kernels, 1)?;
        self.resources.primitive_requests_per_party =
            checked_add(self.resources.primitive_requests_per_party, r.secret_steps)?;
        self.resources.work = checked_add(
            self.resources.work,
            r.cumulative_work.checked_mul(2).ok_or(Error::Resource)?,
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
        if !cached_tables && references != 0 {
            return Err(Error::Context);
        }
        if cached_tables && (references == 0 || r.public_table_reservation_bytes == 0) {
            return Err(Error::Context);
        }
        // Only actual newly constructed owners receive generation/validation/
        // content-hash cost. Cached references still charge bounded structural
        // identity checks, Arc/metadata copies, Plan digest updates, and resource
        // geometry walks on both workers and dealer. Actual primitive work and
        // per-party lookup copies remain unchanged above/below this accounting.
        let table_lowering = if cached_tables {
            let chunks = r.public_table_reservation_bytes.div_ceil(256 * 8);
            let per_reference = 4096usize
                .checked_add(
                    tables
                        .retained_entries
                        .checked_mul(64)
                        .ok_or(Error::Resource)?,
                )
                .ok_or(Error::Resource)?;
            retained
                .checked_mul(48)
                .and_then(|n| n.checked_add(references.checked_mul(per_reference)?.checked_mul(3)?))
                .and_then(|n| n.checked_add(chunks.checked_mul(192)?))
                .ok_or(Error::Resource)?
        } else {
            r.public_table_reservation_bytes
                .checked_mul(48)
                .ok_or(Error::Resource)?
        };
        let lowering = table_lowering
            .checked_add(plan.input_bits().len() * 32)
            .and_then(|n| n.checked_add(r.instructions * 4))
            // Cache stats snapshot/deltas and ten counter commitments are also
            // performed on kernels with no table references; never free work.
            .and_then(|n| n.checked_add(1024))
            .ok_or(Error::Resource)?;
        self.resources.work = checked_add(self.resources.work, lowering)?;
        let metadata_copies = references
            .checked_mul(24)
            .and_then(|n| n.checked_add(misses.checked_mul(512)?))
            .and_then(|n| n.checked_add(3 * std::mem::size_of::<cache::Stats>()))
            .ok_or(Error::Resource)?;
        self.resources.public_copy_bytes = checked_add(
            self.resources.public_copy_bytes,
            copied
                .checked_mul(3)
                .and_then(|n| n.checked_add(metadata_copies))
                .ok_or(Error::Resource)?,
        )?;
        for count in [
            tables.references,
            tables.hits,
            tables.misses,
            tables.table_constructions,
            tables.table_hashes,
            tables.table_hash_bytes,
            tables.table_payload_copy_bytes,
            tables.generated_words,
            tables.retained_bytes,
            tables.retained_entries,
        ] {
            self.hash.update((count as u64).to_le_bytes());
        }
        self.accounted_tables = tables;
        self.resources.source_table_cache_bytes_per_role =
            checked_add(tables.retained_bytes, tables.metadata_reservation_bytes)?;
        self.resources.material_bytes = checked_add(
            self.resources.material_bytes,
            r.private_material_bytes_total,
        )?;
        self.resources.frame_bytes = checked_add(self.resources.frame_bytes, r.frame_bytes_total)?;
        self.resources.public_copy_bytes = checked_add(
            self.resources.public_copy_bytes,
            r.public_table_copy_bytes_total,
        )?;
        // Two independent worker Plans and a separate dealer Plan may coexist.
        // The primitive combined envelope already includes one Plan, two native
        // role registers, paired issuance and retained frame overlap. Add the
        // other two immutable Plans explicitly; no shared-address-space discount.
        self.resources.peak_native_bytes = self.resources.peak_native_bytes.max(
            r.combined_reservation_bytes
                .checked_add(
                    r.plan_reservation_bytes
                        .checked_mul(2)
                        .ok_or(Error::Resource)?,
                )
                .and_then(|n| {
                    n.checked_add(
                        tables
                            .retained_bytes
                            .checked_add(tables.metadata_reservation_bytes)?
                            .checked_mul(3)?,
                    )
                })
                .ok_or(Error::Resource)?,
        );
        // Conservatively retain referenced table reservations inside each Plan's
        // envelope as well as the complete request cache. This covers miss-time
        // generated-value/Table overlap without a shared-address-space discount.
        self.resources.combined_reservation_bytes = checked_add(
            checked_add(
                3 * self.resources.metadata_bytes,
                2 * self.resources.live_array_bytes_per_party,
            )?,
            self.resources.peak_native_bytes,
        )?;
        let l = self.program.limits;
        if self.resources.kernels > 32768
            || self.resources.work > l.max_work
            || self.resources.material_bytes > l.max_material_bytes
            || self.resources.frame_bytes > l.max_frame_bytes
            || self.resources.public_copy_bytes > l.max_public_copy_bytes
            || self.resources.combined_reservation_bytes
                > pllm_garble::fusefss_backend::MAX_ALLOCATION_BYTES
        {
            return Err(Error::Resource);
        }
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
        self.run(&plan, input, outputs, true)
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
        self.resources.public_copy_bytes = checked_add(
            self.resources.public_copy_bytes,
            weights.len().checked_mul(16).ok_or(Error::Resource)?,
        )?;
        if self.resources.public_copy_bytes > self.program.limits.max_public_copy_bytes {
            return Err(Error::Resource);
        }
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
        Ok(self.kernel(&input, 1, |budget| {
            Ok(kernels::private_dot(binding, &lr, &rr, budget)?.0)
        })?[0])
    }
    fn execute(&mut self, input: &[u64]) -> Result<Zeroizing<Vec<u64>>, Error> {
        let p = self.program;
        let mut tokens = Zeroizing::new(zeros(p.sequence)?);
        tokens[..p.initial].copy_from_slice(input);
        let mut finals = Zeroizing::new(vector(p.selections)?);
        for selection in 0..p.selections {
            self.selection = selection;
            let prefix = p.initial + selection;
            let mut arrays = vector::<Option<Zeroizing<Vec<u64>>>>(p.nodes.len())?;
            arrays.resize_with(p.nodes.len(), || None);
            for (index, node) in p.nodes.iter().enumerate() {
                self.node = index;
                let values = self.node(node, &arrays, &tokens, prefix)?;
                if values.len() != elements(&node.shape)? {
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
        let mut out = Zeroizing::new(zeros(elements(&node.shape)?)?);
        use Code::*;
        match node.code {
            Embedding { weight, positions } => {
                let w = &p.source.tensors[weight];
                let columns = w.shape[1];
                let rows = w.shape[0];
                if positions {
                    if self.role == 0 {
                        for i in 0..t {
                            for c in 0..columns {
                                out[i * columns + c] = word(w.values[i * columns + c]);
                            }
                        }
                    }
                } else {
                    let binding = self.binding();
                    let id_range = Range::new(0, p.vocabulary as i128 - 1)?.gate();
                    for i in 0..t {
                        let hot = self.kernel(&tokens[i..i + 1], rows, |budget| {
                            Ok(kernels::token_one_hot(binding, id_range, rows, budget)?.0)
                        })?;
                        for row in 0..rows {
                            for c in 0..columns {
                                let v = hot[row].wrapping_mul(word(w.values[row * columns + c]));
                                out[i * columns + c] = out[i * columns + c].wrapping_add(v) & MASK;
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
                let range = p.nodes[input].range;
                let source = p.source.digest();
                for row in 0..x.len() / width {
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
                        self.cached_kernel(&x[row * width..(row + 1) * width], width, |tables| {
                            norm::compile_cached(source, &spec, tables)
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
                let rs = ranges(x.len(), p.nodes[input].range)?;
                let binding = self.binding();
                out[0] =
                    self.kernel(x, 1, |budget| Ok(kernels::argmax(binding, &rs, budget)?.0))?[0];
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
