use super::*;
#[cfg(test)]
#[path = "oracle_trace.rs"]
pub(super) mod oracle_trace;
use multiply::{Operand, PartyRuntime, Status};
use puma::{Context, Message, MessagePhase, Operation, Party, ReplicatedShare};

// A reusable execution unit, not yet a selectable whole-decoder policy. The
// enclosing Body compiler must bind its digest and aggregate its complete work.
#[allow(dead_code)]
#[path = "body.rs"]
pub(super) mod body;

#[path = "lookup.rs"]
pub(super) mod lookup;

#[path = "selection.rs"]
pub(super) mod selection;

#[path = "trie.rs"]
pub(super) mod trie;

pub enum PeerBytes {
    Arithmetic(puma::wire::Encoded<Message>),
    Tensor(puma::wire::Encoded<tensor::Frame>),
}

/// Local bounded-driver result. Empty frames do not imply a reached boundary.
/// `Reached` is not proof that the router has drained the other two roles.
pub enum BoundaryProgress {
    Frames(Vec<(Party, PeerBytes)>),
    Reached,
}
impl PeerBytes {
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Arithmetic(v) => v.as_bytes(),
            Self::Tensor(v) => v.as_bytes(),
        }
    }
}

// Keep bounded protocol transitions inline; boxing would add an unchecked heap
// allocation at each tensor handoff. The enclosing worker prices this fixed owner.
#[allow(clippy::large_enum_variant)]
enum Active {
    Idle(PartyRuntime),
    Product(PartyRuntime),
    Tensor(tensor::Executor),
    Lookup(lookup::BulkLookup),
    Trie(trie::TrieLookup),
    Selection(selection::StableArgmax),
    Retired,
}

/// One native role, with no other party's secrets or reconstructed tensor.
/// Public graph metadata and persistent shares have a separate admitted owner;
/// each primitive retains its original 256 KiB ledger.
pub struct Worker {
    program: Arc<Program>,
    party: Party,
    values: Vec<Option<Secret>>,
    active: Active,
    inbox: [Option<Message>; 2],
    step: usize,
    loop_offset: usize,
    started: bool,
    imported: usize,
    imported_words: usize,
    session: [u8; 32],
    pid: u32,
    public: Option<PublicSource>,
}

#[cfg(test)]
type BoundaryState<I> = Result<(usize, puma::prss::Cursors, I), Error>;

impl Worker {
    pub fn new(
        program: Arc<Program>,
        runtime: PartyRuntime,
        session: [u8; 32],
    ) -> Result<Self, Error> {
        Self::new_with_source(program, runtime, session, None)
    }

    pub fn new_with_source(
        program: Arc<Program>,
        runtime: PartyRuntime,
        session: [u8; 32],
        public: Option<PublicSource>,
    ) -> Result<Self, Error> {
        if runtime.status() != Status::Ready
            || !runtime.matches_context(Context::new(session, program.digest(), 0)?)
        {
            return Err(Error::Binding);
        }
        let public_count = program.nodes[..program.inputs]
            .iter()
            .filter(|n| matches!(n.code, Code::PublicInput { .. }))
            .count();
        if let Some(source) = &public {
            source.check(&program)?;
        }
        if public.is_some() != (public_count > 0)
            || program.nodes[..public_count]
                .iter()
                .any(|n| !matches!(n.code, Code::PublicInput { .. }))
        {
            return Err(Error::Binding);
        }
        let party = runtime.party();
        let mut values = Vec::new();
        values
            .try_reserve_exact(program.nodes.len())
            .map_err(|_| Error::Allocation)?;
        values.resize_with(program.nodes.len(), || None);
        Ok(Self {
            program,
            party,
            values,
            active: Active::Idle(runtime),
            inbox: [None, None],
            step: 0,
            loop_offset: 0,
            started: false,
            imported: public_count,
            imported_words: 0,
            session,
            pid: std::process::id(),
            public,
        })
    }

    pub fn party(&self) -> Party {
        self.party
    }
    pub fn complete(&self) -> bool {
        self.step == self.program.nodes.len() && !self.retired()
    }
    pub fn retired(&self) -> bool {
        matches!(self.active, Active::Retired)
    }
    #[cfg(test)]
    pub(super) fn loop_test_state(&self) -> (usize, usize) {
        (
            self.loop_offset,
            self.values.iter().filter(|v| v.is_some()).count(),
        )
    }
    /// Borrow local components only at an idle, locally quiescent test boundary.
    /// Caller still proves router drain. The iterator cannot outlive this borrow
    /// or duplicate a consuming runtime.
    #[cfg(test)]
    pub(super) fn test_boundary_state(
        &self,
    ) -> BoundaryState<impl Iterator<Item = Option<(&[u64], &[u64])>> + '_> {
        self.live()?;
        if self.imported != self.program.inputs || !self.boundary_ready() {
            return Err(Error::Phase);
        }
        let Active::Idle(runtime) = &self.active else {
            return Err(Error::Phase);
        };
        Ok((
            self.step,
            runtime.cursors(),
            self.values.iter().map(|value| {
                value
                    .as_ref()
                    .map(|value| (value.first.as_slice(), value.second.as_slice()))
            }),
        ))
    }
    #[cfg(test)]
    pub(super) fn selection_test_phase(&self) -> Option<u8> {
        match &self.active {
            Active::Selection(selection) => Some(selection.test_phase()),
            _ => None,
        }
    }
    pub fn cancel(&mut self) {
        self.active = Active::Retired;
        self.inbox = [None, None];
        self.values.clear();
        self.public = None;
        self.loop_offset = 0;
    }
    fn checked<T>(&mut self, value: Result<T, Error>) -> Result<T, Error> {
        if value.is_err() {
            self.cancel();
        }
        value
    }
    fn live(&self) -> Result<(), Error> {
        if self.retired() || self.pid != std::process::id() {
            Err(Error::Retired)
        } else {
            Ok(())
        }
    }

    /// Ordered original model/token shares. The enclosing publisher checks the
    /// native source/range commitment before transmission; no private value is
    /// reconstructed here to validate a claimed range.
    pub fn import(&mut self, key: &str, first: &[u64], second: &[u64]) -> Result<(), Error> {
        self.import_chunk(key, 0, first, second)
    }

    pub fn import_chunk(
        &mut self,
        key: &str,
        offset: usize,
        first: &[u64],
        second: &[u64],
    ) -> Result<(), Error> {
        let result = (|| {
            self.live()?;
            if self.started || self.imported >= self.program.inputs {
                return Err(Error::Phase);
            }
            let node = &self.program.nodes[self.imported];
            if !matches!(&node.code, Code::Input(name) if name == key)
                || first.is_empty()
                || first.len() != second.len()
                || offset != self.imported_words
                || offset.checked_add(first.len()).ok_or(Error::Bounds)? > node.lanes
            {
                return Err(Error::Shape);
            }
            if offset == 0 {
                self.values[self.imported] = Some(Secret::new(node.lanes)?);
            }
            let value = self.values[self.imported].as_mut().ok_or(Error::Phase)?;
            value.first[offset..offset + first.len()].copy_from_slice(first);
            value.second[offset..offset + second.len()].copy_from_slice(second);
            self.imported_words += first.len();
            if self.imported_words == node.lanes {
                self.imported += 1;
                self.imported_words = 0;
            }
            Ok(())
        })();
        self.checked(result)
    }

    fn share(&self, input: usize, context: Context) -> Result<ReplicatedShare, Error> {
        let source = self
            .values
            .get(input)
            .and_then(Option::as_ref)
            .ok_or(Error::Binding)?;
        let op = Operation::new(
            context,
            0,
            Party::P0,
            source.first.len(),
            1,
            puma::Limits::default(),
        )?;
        ReplicatedShare::from_local(
            op,
            self.party,
            &source.first,
            &source.second,
            puma::Limits::default(),
        )
    }

    fn save(&mut self, value: ReplicatedShare) -> Result<(), Error> {
        if matches!(self.program.nodes[self.step].code, Code::TensorLoop { .. }) {
            let output = self.values[self.step].as_mut().ok_or(Error::Phase)?;
            let end = self
                .loop_offset
                .checked_add(value.first().len())
                .ok_or(Error::Bounds)?;
            output
                .first
                .get_mut(self.loop_offset..end)
                .ok_or(Error::Shape)?
                .copy_from_slice(value.first());
            output
                .second
                .get_mut(self.loop_offset..end)
                .ok_or(Error::Shape)?
                .copy_from_slice(value.second());
            self.loop_offset = end;
            return Ok(());
        }
        let mut secret = Secret::new(value.first().len())?;
        secret.first.copy_from_slice(value.first());
        secret.second.copy_from_slice(value.second());
        self.values[self.step] = Some(secret);
        Ok(())
    }

    fn retire_inputs(&mut self) {
        #[cfg(test)]
        oracle_trace::retiring(self);
        if matches!(self.program.nodes[self.step].code, Code::TensorLoop { .. })
            && self.loop_offset < self.program.nodes[self.step].lanes
        {
            return;
        }
        for &id in &self.program.retire[self.step] {
            self.values[id] = None;
        }
        self.step += 1;
        self.loop_offset = 0;
    }

    fn loop_share(
        &self,
        input: LoopInput,
        n: usize,
        context: Context,
    ) -> Result<ReplicatedShare, Error> {
        let op = Operation::new(context, 0, Party::P0, n, 1, puma::Limits::default())?;
        match input {
            LoopInput::Slice(owner) => {
                let source = self.values[owner].as_ref().ok_or(Error::Binding)?;
                let end = self.loop_offset.checked_add(n).ok_or(Error::Bounds)?;
                ReplicatedShare::from_local(
                    op,
                    self.party,
                    source
                        .first
                        .get(self.loop_offset..end)
                        .ok_or(Error::Shape)?,
                    source
                        .second
                        .get(self.loop_offset..end)
                        .ok_or(Error::Shape)?,
                    puma::Limits::default(),
                )
            }
            LoopInput::TokenOffsets { owner, index } => {
                let (a, b) = self.word(owner, index)?;
                let mut input = Secret::new(n)?;
                for i in 0..n {
                    let offset = (self.loop_offset + i) as u64;
                    input.first[i] =
                        a.wrapping_sub(if self.party == Party::P0 { offset } else { 0 });
                    input.second[i] = b.wrapping_sub(if self.party.next() == Party::P0 {
                        offset
                    } else {
                        0
                    });
                }
                ReplicatedShare::from_local(
                    op,
                    self.party,
                    &input.first,
                    &input.second,
                    puma::Limits::default(),
                )
            }
        }
    }

    fn word(&self, owner: usize, index: usize) -> Result<(u64, u64), Error> {
        if let Code::PublicInput { key, .. } = &self.program.nodes[owner].code {
            let mut raw = [0u8; 8];
            self.public
                .as_ref()
                .ok_or(Error::Binding)?
                .file(key)?
                .read(index as u64 * 8, &mut raw)
                .map_err(|_| Error::Binding)?;
            let value = u64::from_le_bytes(raw);
            Ok((
                if self.party == Party::P0 { value } else { 0 },
                if self.party.next() == Party::P0 {
                    value
                } else {
                    0
                },
            ))
        } else {
            let input = self
                .values
                .get(owner)
                .and_then(Option::as_ref)
                .ok_or(Error::Binding)?;
            Ok((
                *input.first.get(index).ok_or(Error::Shape)?,
                *input.second.get(index).ok_or(Error::Shape)?,
            ))
        }
    }

    /// Public quiescence barrier: finish the previous primitive/tensor step and
    /// start the next. The authenticated router drains every emitted peer batch
    /// on all three roles before invoking this barrier again.
    pub fn advance(&mut self) -> Result<Vec<(Party, PeerBytes)>, Error> {
        let result = self.advance_inner(None);
        self.checked(result)
    }

    /// Advance strictly before an existing original node, finishing every prefix
    /// primitive and every TensorLoop tile. Backward, out-of-range and already
    /// active target nodes are rejected and retire this worker.
    ///
    /// The caller must drain emitted frames across all three roles before each
    /// advance and before handoff; local `Reached` cannot establish global drain.
    pub fn advance_before(&mut self, target: usize) -> Result<BoundaryProgress, Error> {
        let result = (|| {
            self.live()?;
            if target >= self.program.nodes.len() || target < self.step {
                return Err(Error::Bounds);
            }
            if target == self.step && !matches!(self.active, Active::Idle(_)) {
                return Err(Error::Phase);
            }
            let frames = self.advance_inner(Some(target))?;
            if self.step == target {
                if !frames.is_empty() || !self.boundary_ready() {
                    return Err(Error::Phase);
                }
                Ok(BoundaryProgress::Reached)
            } else {
                Ok(BoundaryProgress::Frames(frames))
            }
        })();
        self.checked(result)
    }

    fn boundary_ready(&self) -> bool {
        matches!(&self.active, Active::Idle(runtime) if runtime.status() == Status::Ready)
            && self.inbox.iter().all(Option::is_none)
            && self.loop_offset == 0
    }

    fn advance_inner(&mut self, before: Option<usize>) -> Result<Vec<(Party, PeerBytes)>, Error> {
        self.live()?;
        if self.complete()
            || self.imported != self.program.inputs
            || self.inbox.iter().any(Option::is_some)
        {
            return Err(Error::Phase);
        }
        let active = std::mem::replace(&mut self.active, Active::Retired);
        self.active = match active {
            Active::Product(mut runtime) => {
                let output = runtime.take_output()?;
                self.save(output)?;
                self.retire_inputs();
                Active::Idle(runtime)
            }
            Active::Tensor(mut executor) => {
                executor.finish_step()?;
                if !executor.complete() {
                    let frames = executor.start_step()?;
                    self.active = Active::Tensor(executor);
                    return tensor_frames(frames);
                }
                let runtime = if matches!(
                    self.program.nodes[self.step].code,
                    Code::CheckedRange { .. }
                ) {
                    let (runtime, output, valid) = executor.finish_checked()?;
                    let n = output.first().len();
                    if valid.first().len() != n || self.program.nodes[self.step].lanes != 2 * n {
                        return Err(Error::Shape);
                    }
                    let mut both = Secret::new(2 * n)?;
                    both.first[..n].copy_from_slice(output.first());
                    both.second[..n].copy_from_slice(output.second());
                    both.first[n..].copy_from_slice(valid.first());
                    both.second[n..].copy_from_slice(valid.second());
                    self.values[self.step] = Some(both);
                    runtime
                } else {
                    let (runtime, output) = executor.finish()?;
                    self.save(output)?;
                    runtime
                };
                self.retire_inputs();
                Active::Idle(runtime)
            }
            Active::Lookup(mut bulk) => {
                let Code::PublicLookup { weight, .. } = &self.program.nodes[self.step].code else {
                    return Err(Error::Binding);
                };
                let Code::PublicInput { key, .. } = &self.program.nodes[*weight].code else {
                    return Err(Error::Binding);
                };
                let file = self.public.as_ref().ok_or(Error::Binding)?.file(key)?;
                bulk.finish_step(file)?;
                if !bulk.complete() {
                    let frames = bulk.start_step()?;
                    self.active = Active::Lookup(bulk);
                    return tensor_frames(frames);
                }
                let (runtime, output) = bulk.finish()?;
                if output.first.len() != self.program.nodes[self.step].lanes {
                    return Err(Error::Shape);
                }
                self.values[self.step] = Some(output);
                self.retire_inputs();
                Active::Idle(runtime)
            }
            Active::Trie(mut lookup) => {
                let Code::TrieLookup { weight, .. } = &self.program.nodes[self.step].code else {
                    return Err(Error::Binding);
                };
                let Code::PublicInput { key, .. } = &self.program.nodes[*weight].code else {
                    return Err(Error::Binding);
                };
                let file = self.public.as_ref().ok_or(Error::Binding)?.file(key)?;
                lookup.finish_step(*weight, file)?;
                if !lookup.complete() {
                    let frames = lookup.start_step()?;
                    self.active = Active::Trie(lookup);
                    return Ok(frames);
                }
                let (runtime, output) = lookup.finish()?;
                if output.first.len() != self.program.nodes[self.step].lanes {
                    return Err(Error::Shape);
                }
                self.values[self.step] = Some(output);
                self.retire_inputs();
                Active::Idle(runtime)
            }
            Active::Selection(mut selection) => {
                selection.finish_step()?;
                if !selection.complete() {
                    let frames = selection.start_step()?;
                    self.active = Active::Selection(selection);
                    return Ok(frames);
                }
                let (runtime, output) = selection.finish()?;
                if output.first.len() != self.program.nodes[self.step].lanes {
                    return Err(Error::Shape);
                }
                self.values[self.step] = Some(output);
                self.retire_inputs();
                Active::Idle(runtime)
            }
            Active::Idle(runtime) => Active::Idle(runtime),
            Active::Retired => return Err(Error::Retired),
        };
        self.started = true;
        while self.step < before.unwrap_or(self.program.nodes.len()) {
            let node = &self.program.nodes[self.step];
            match &node.code {
                Code::Input(_) | Code::PublicInput { .. } => {
                    self.retire_inputs();
                }
                Code::PublicValues(values) => {
                    let mut out = Secret::new(values.len())?;
                    for (i, value) in values.iter().enumerate() {
                        if self.party == Party::P0 {
                            out.first[i] = *value as u64;
                        }
                        if self.party.next() == Party::P0 {
                            out.second[i] = *value as u64;
                        }
                    }
                    self.values[self.step] = Some(out);
                    self.retire_inputs();
                }
                Code::Gather(items) => {
                    let mut output = Secret::new(items.len())?;
                    for (i, (owner, index)) in items.iter().enumerate() {
                        (output.first[i], output.second[i]) = self.word(owner, index)?;
                    }
                    self.values[self.step] = Some(output);
                    self.retire_inputs();
                }
                Code::SourceGather(view) => {
                    let mut output = Secret::new(node.lanes)?;
                    for (i, index) in view.indices().enumerate() {
                        (output.first[i], output.second[i]) = self.word(view.owner, index)?;
                    }
                    self.values[self.step] = Some(output);
                    self.retire_inputs();
                }
                Code::PublicProduct {
                    input,
                    weight,
                    rows,
                    inner,
                    columns,
                    transposed,
                } => {
                    let Code::PublicInput { key, .. } = &self.program.nodes[*weight].code else {
                        return Err(Error::Binding);
                    };
                    let file = self.public.as_ref().ok_or(Error::Binding)?.file(key)?;
                    let input = self.values[*input].as_ref().ok_or(Error::Binding)?;
                    self.values[self.step] = Some(super::public_source::multiply(
                        file,
                        input,
                        *rows,
                        *inner,
                        *columns,
                        *transposed,
                    )?);
                    self.retire_inputs();
                }
                Code::Affine {
                    left,
                    right,
                    coefficients: c,
                } => {
                    let left = self.values[*left].as_ref().ok_or(Error::Binding)?;
                    let right = right
                        .map(|id| self.values[id].as_ref().ok_or(Error::Binding))
                        .transpose()?;
                    let mut output = Secret::new(node.lanes)?;
                    for i in 0..node.lanes {
                        output.first[i] = left.first[i]
                            .wrapping_mul(c.left as u64)
                            .wrapping_add(
                                right.map_or(0, |v| v.first[i]).wrapping_mul(c.right as u64),
                            )
                            .wrapping_add(if self.party == Party::P0 {
                                c.constant as u64
                            } else {
                                0
                            });
                        output.second[i] = left.second[i]
                            .wrapping_mul(c.left as u64)
                            .wrapping_add(
                                right
                                    .map_or(0, |v| v.second[i])
                                    .wrapping_mul(c.right as u64),
                            )
                            .wrapping_add(if self.party.next() == Party::P0 {
                                c.constant as u64
                            } else {
                                0
                            });
                    }
                    self.values[self.step] = Some(output);
                    self.retire_inputs();
                }
                Code::Selection { input, plan } => {
                    // Compiler emits a dedicated immediately preceding gather.
                    // Last-use retirement proves no live consumer/output is
                    // invalidated when the tournament compacts that owner.
                    if self.step.checked_sub(1) != Some(*input)
                        || !matches!(self.program.nodes[*input].code, Code::Gather(_))
                        || !self.program.retire[self.step].contains(input)
                        || node.lanes != 1
                    {
                        return Err(Error::Binding);
                    }
                    let Active::Idle(runtime) =
                        std::mem::replace(&mut self.active, Active::Retired)
                    else {
                        return Err(Error::Phase);
                    };
                    let scores = self.values[*input].take().ok_or(Error::Binding)?;
                    let mut selection =
                        selection::StableArgmax::new(runtime, scores, plan.clone())?;
                    if selection.complete() {
                        let (runtime, output) = selection.finish()?;
                        self.values[self.step] = Some(output);
                        self.retire_inputs();
                        self.active = Active::Idle(runtime);
                    } else {
                        let frames = selection.start_step()?;
                        self.active = Active::Selection(selection);
                        return Ok(frames);
                    }
                }
                Code::TrieLookup {
                    input,
                    weight,
                    plan,
                } => {
                    plan.validate_owners(*input, *weight, self.program.source_binding)?;
                    if self.step.checked_sub(1) != Some(*input)
                        || !matches!(self.program.nodes[*input].code, Code::Gather(_))
                        || !self.program.retire[self.step].contains(input)
                    {
                        return Err(Error::Binding);
                    }
                    let Code::PublicInput { key, .. } = &self.program.nodes[*weight].code else {
                        return Err(Error::Binding);
                    };
                    let file = self.public.as_ref().ok_or(Error::Binding)?.file(key)?;
                    let context = Context::new(self.session, self.program.digest(), 0)?;
                    let Active::Idle(runtime) =
                        std::mem::replace(&mut self.active, Active::Retired)
                    else {
                        return Err(Error::Phase);
                    };
                    let tokens = self.values[*input].take().ok_or(Error::Binding)?;
                    let mut lookup = trie::TrieLookup::new(
                        runtime,
                        tokens,
                        plan.clone(),
                        context,
                        *input,
                        *weight,
                        file,
                    )?;
                    let frames = lookup.start_step()?;
                    self.active = Active::Trie(lookup);
                    return Ok(frames);
                }
                Code::PublicLookup { input, plan, .. } => {
                    let Active::Idle(runtime) =
                        std::mem::replace(&mut self.active, Active::Retired)
                    else {
                        return Err(Error::Phase);
                    };
                    let source = self.values[*input].as_ref().ok_or(Error::Binding)?;
                    let mut tokens = Secret::new(source.first.len())?;
                    tokens.first.copy_from_slice(&source.first);
                    tokens.second.copy_from_slice(&source.second);
                    let mut bulk = lookup::BulkLookup::new(runtime, tokens, (**plan).clone())?;
                    let frames = bulk.start_step()?;
                    self.active = Active::Lookup(bulk);
                    return tensor_frames(frames);
                }
                Code::Tensor { input, program } | Code::CheckedRange { input, program } => {
                    let Active::Idle(runtime) =
                        std::mem::replace(&mut self.active, Active::Retired)
                    else {
                        return Err(Error::Phase);
                    };
                    let input = self.share(*input, runtime.context())?;
                    let mut executor = tensor::Executor::new(runtime, (**program).clone(), input)?;
                    let out = executor.start_step()?;
                    self.active = Active::Tensor(executor);
                    return tensor_frames(out);
                }
                Code::TensorLoop {
                    input,
                    program,
                    tail,
                } => {
                    let Active::Idle(runtime) =
                        std::mem::replace(&mut self.active, Active::Retired)
                    else {
                        return Err(Error::Phase);
                    };
                    let remaining = node
                        .lanes
                        .checked_sub(self.loop_offset)
                        .ok_or(Error::Bounds)?;
                    if remaining == 0 {
                        return Err(Error::Phase);
                    }
                    let selected = if remaining < program.lanes() {
                        tail.as_ref().ok_or(Error::Shape)?
                    } else {
                        program
                    };
                    if self.loop_offset == 0 {
                        self.values[self.step] = Some(Secret::new(node.lanes)?);
                    }
                    let input = self.loop_share(*input, selected.lanes(), runtime.context())?;
                    // Each iteration constructs a fresh consuming executor. Its
                    // invocation is bound to the runtime's monotonic PRSS cursor,
                    // so frames and randomness cannot be reused across tiles.
                    let mut executor = tensor::Executor::new(runtime, (**selected).clone(), input)?;
                    let out = executor.start_step()?;
                    self.active = Active::Tensor(executor);
                    return tensor_frames(out);
                }
                Code::Product {
                    left,
                    right,
                    shape,
                    rescale,
                    norm,
                } => {
                    let Active::Idle(mut runtime) =
                        std::mem::replace(&mut self.active, Active::Retired)
                    else {
                        return Err(Error::Phase);
                    };
                    let bounds = InputBounds {
                        left: self.program.nodes[*left].bounds,
                        right: self.program.nodes[*right].bounds,
                    };
                    let plan = if let Some(proof) = norm {
                        if proof.input_bounds() != bounds {
                            return Err(Error::Binding);
                        }
                        let shape = shape.ok_or(Error::Binding)?;
                        if *rescale {
                            runtime.plan_norm_matmul(shape, **proof)?
                        } else {
                            runtime.plan_ring_norm_matmul(shape, **proof)?
                        }
                    } else {
                        match (shape, rescale) {
                            (Some(shape), true) => runtime.plan_matmul(*shape, bounds)?,
                            (Some(shape), false) => runtime.plan_ring_matmul(*shape, bounds)?,
                            (None, true) => runtime.plan(&vec![bounds; node.lanes])?,
                            (None, false) => {
                                runtime.plan_ring_product(&vec![bounds; node.lanes])?
                            }
                        }
                    };
                    let a = self.values[*left].as_ref().ok_or(Error::Binding)?;
                    let b = self.values[*right].as_ref().ok_or(Error::Binding)?;
                    let a = plan.share_operand(
                        Operand::Left,
                        self.party,
                        &a.first,
                        &a.second,
                        puma::Limits::default(),
                    )?;
                    let b = plan.share_operand(
                        Operand::Right,
                        self.party,
                        &b.first,
                        &b.second,
                        puma::Limits::default(),
                    )?;
                    let out = runtime.begin(a, b)?;
                    self.active = Active::Product(runtime);
                    return arithmetic_frames([Some(out), None, None]);
                }
            }
        }
        Ok(vec![])
    }

    pub fn receive(&mut self, raw: &[u8], sender: Party) -> Result<Vec<(Party, PeerBytes)>, Error> {
        let result = (|| {
            self.live()?;
            match &mut self.active {
                Active::Tensor(executor) => tensor_frames(executor.receive_encoded(raw, sender)?),
                Active::Lookup(bulk) => tensor_frames(bulk.receive_encoded(raw, sender)?),
                Active::Trie(lookup) => lookup.receive_encoded(raw, sender),
                Active::Selection(selection) => selection.receive_encoded(raw, sender),
                Active::Product(runtime) => {
                    let message = Message::decode(raw, sender, self.party)?;
                    let mut out = [None, None, None];
                    match runtime.status() {
                        Status::AwaitingRotate => {
                            let [a, b] = runtime.rotate(message)?;
                            out = [a, b, None];
                        }
                        Status::AwaitingOpen => {
                            let slot = match message.header().phase {
                                MessagePhase::MaskedInput => 0,
                                MessagePhase::Correlation => 1,
                                _ => return Err(Error::Phase),
                            };
                            if self.inbox[slot].is_some() {
                                return Err(Error::Binding);
                            }
                            self.inbox[slot] = Some(message);
                            if self.inbox.iter().all(Option::is_some) {
                                out[0] = Some(runtime.open(
                                    self.inbox[0].take().unwrap(),
                                    self.inbox[1].take().unwrap(),
                                )?);
                            }
                        }
                        Status::AwaitingReshare => runtime.reshare(message)?,
                        _ => return Err(Error::Phase),
                    }
                    arithmetic_frames(out)
                }
                _ => Err(Error::Phase),
            }
        })();
        self.checked(result)
    }

    /// Consuming final-only disclosure of this role's first replicated component.
    /// No logits, activations, lookup values or intermediate IDs are exported.
    pub fn finish(mut self) -> Result<Zeroizing<Vec<u64>>, Error> {
        self.live()?;
        if !self.complete() {
            return Err(Error::Phase);
        }
        let mut out = Zeroizing::new(vec_zero(self.program.outputs.len())?);
        for (slot, &(id, index)) in out.iter_mut().zip(&self.program.outputs) {
            *slot = self.values[id].as_ref().ok_or(Error::Binding)?.first[index];
        }
        self.cancel();
        Ok(out)
    }

    pub fn finish_frame(self) -> Result<Zeroizing<Vec<u8>>, Error> {
        let session = self.session;
        let program = self.program.digest();
        let party = self.party;
        let words = self.finish()?;
        let mut frame = Zeroizing::new(vec_zero(73 + 8 * words.len())?);
        frame[..4].copy_from_slice(b"PUR1");
        frame[4..36].copy_from_slice(&session);
        frame[36..68].copy_from_slice(&program);
        frame[68] = party as u8;
        frame[69..73].copy_from_slice(&(words.len() as u32).to_le_bytes());
        for (out, value) in frame[73..].chunks_exact_mut(8).zip(words.iter()) {
            out.copy_from_slice(&value.to_le_bytes());
        }
        Ok(frame)
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.cancel();
    }
}

fn arithmetic_frames(frames: [Option<Message>; 3]) -> Result<Vec<(Party, PeerBytes)>, Error> {
    frames
        .into_iter()
        .flatten()
        .map(|v| Ok((v.header().to, PeerBytes::Arithmetic(v.encode()?))))
        .collect()
}
fn tensor_frames(frames: [Option<tensor::Frame>; 3]) -> Result<Vec<(Party, PeerBytes)>, Error> {
    frames
        .into_iter()
        .flatten()
        .map(|v| Ok((v.recipient(), PeerBytes::Tensor(v.encode()?))))
        .collect()
}
