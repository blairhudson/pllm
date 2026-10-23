use super::program::schedule_access::{Kind, PrivateScheduleRef};
use super::*;
#[cfg(test)]
#[path = "frozen_execution_tests.rs"]
pub(in crate::replicated_decoder) mod frozen_execution_tests;
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
    // Logical source-schedule ordinal, never a physical value slot.
    step: usize,
    loop_offset: usize,
    started: bool,
    imported: usize,
    imported_words: usize,
    session: [u8; 32],
    pid: u32,
    public: Option<PublicSource>,
    // Test-only consuming custody. Production has no frozen constructor/route.
    #[cfg(test)]
    oracle_values: Option<frozen_execution_tests::Values>,
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
        let public_count = program.schedule_access().public_input_count()?;
        if let Some(source) = &public {
            source.check(&program)?;
        }
        if public.is_some() != (public_count > 0)
            || !program.schedule_access().public_prefix(public_count)?
        {
            return Err(Error::Binding);
        }
        let party = runtime.party();
        let mut values = Vec::new();
        values
            .try_reserve_exact(program.value_slot_count())
            .map_err(|_| Error::Allocation)?;
        values.resize_with(program.value_slot_count(), || None);
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
            #[cfg(test)]
            oracle_values: None,
        })
    }

    pub fn party(&self) -> Party {
        self.party
    }
    pub fn complete(&self) -> bool {
        self.step == self.program.logical_len() && !self.retired()
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
        if self.imported != self.program.input_count() || !self.boundary_ready() {
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
        #[cfg(test)]
        {
            self.oracle_values = None;
        }
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
            if self.started || self.imported >= self.program.input_count() {
                return Err(Error::Phase);
            }
            let owner = self.imported;
            let node = self.program.read_recipe(owner);
            let slot = self.program.value_slot(owner);
            if node.input_name().ok() != Some(key)
                || first.is_empty()
                || first.len() != second.len()
                || offset != self.imported_words
                || offset.checked_add(first.len()).ok_or(Error::Bounds)? > node.lanes()
            {
                return Err(Error::Shape);
            }
            if offset == 0 {
                self.values[slot] = Some(Secret::new(node.lanes())?);
            }
            let value = self.values[slot].as_mut().ok_or(Error::Phase)?;
            value.first[offset..offset + first.len()].copy_from_slice(first);
            value.second[offset..offset + second.len()].copy_from_slice(second);
            self.imported_words += first.len();
            if self.imported_words == node.lanes() {
                self.imported += 1;
                self.imported_words = 0;
            }
            Ok(())
        })();
        self.checked(result)
    }

    fn share(&self, input: usize, context: Context) -> Result<ReplicatedShare, Error> {
        let (first, second) = self.local_slices(input)?;
        let op = Operation::new(
            context,
            0,
            Party::P0,
            first.len(),
            1,
            puma::Limits::default(),
        )?;
        ReplicatedShare::from_local(op, self.party, first, second, puma::Limits::default())
    }

    fn local_slices(&self, logical: usize) -> Result<(&[u64], &[u64]), Error> {
        #[cfg(test)]
        if let Some(state) = &self.oracle_values {
            let span = state.layout.span(logical)?;
            let (slot, offset) = span.stored()?;
            let end = offset.checked_add(span.lanes).ok_or(Error::Bounds)?;
            if self.step > span.last_use || end > state.initialized[slot] {
                return Err(Error::Binding);
            }
            let value = self
                .values
                .get(slot)
                .and_then(Option::as_ref)
                .ok_or(Error::Binding)?;
            return Ok((
                value.first.get(offset..end).ok_or(Error::Shape)?,
                value.second.get(offset..end).ok_or(Error::Shape)?,
            ));
        }
        let value = self
            .values
            .get(self.program.value_slot(logical))
            .and_then(Option::as_ref)
            .ok_or(Error::Binding)?;
        Ok((&value.first, &value.second))
    }

    fn ensure_local(&mut self, logical: usize, lanes: usize) -> Result<(), Error> {
        #[cfg(test)]
        if let Some(state) = &self.oracle_values {
            let span = state.layout.span(logical)?;
            let (slot, offset) = span.stored()?;
            if span.lanes != lanes {
                return Err(Error::Shape);
            }
            if self.values[slot].is_none() {
                if offset != 0 || state.initialized[slot] != 0 {
                    return Err(Error::Binding);
                }
                self.values[slot] = Some(Secret::new(state.layout.owners[slot].lanes)?);
            }
            return Ok(());
        }
        self.values[self.program.value_slot(logical)] = Some(Secret::new(lanes)?);
        Ok(())
    }

    #[cfg(test)]
    fn write_local(
        &mut self,
        logical: usize,
        offset: usize,
        first: &[u64],
        second: &[u64],
    ) -> Result<(), Error> {
        let state = self.oracle_values.as_mut().ok_or(Error::Binding)?;
        let span = state.layout.span(logical)?;
        let (slot, physical_offset) = span.stored()?;
        let end = offset.checked_add(first.len()).ok_or(Error::Bounds)?;
        if first.len() != second.len() || end > span.lanes {
            return Err(Error::Shape);
        }
        let start = physical_offset.checked_add(offset).ok_or(Error::Bounds)?;
        if state.initialized[slot] != start {
            return Err(Error::Phase);
        }
        let stop = start.checked_add(first.len()).ok_or(Error::Bounds)?;
        let output = self.values[slot].as_mut().ok_or(Error::Phase)?;
        output
            .first
            .get_mut(start..stop)
            .ok_or(Error::Shape)?
            .copy_from_slice(first);
        output
            .second
            .get_mut(start..stop)
            .ok_or(Error::Shape)?
            .copy_from_slice(second);
        state.initialized[slot] = stop;
        Ok(())
    }

    fn store_local(&mut self, logical: usize, value: Secret) -> Result<(), Error> {
        #[cfg(test)]
        if self.oracle_values.is_some() {
            let span = self
                .oracle_values
                .as_ref()
                .ok_or(Error::Binding)?
                .layout
                .span(logical)?;
            if span.lanes != value.first.len() {
                return Err(Error::Shape);
            }
            self.ensure_local(logical, span.lanes)?;
            return self.write_local(logical, 0, &value.first, &value.second);
        }
        self.values[self.program.value_slot(logical)] = Some(value);
        Ok(())
    }

    fn save(
        &mut self,
        value: ReplicatedShare,
        reader: PrivateScheduleRef<'_>,
    ) -> Result<(), Error> {
        if reader.node(self.step)?.kind() == Kind::TensorLoop {
            #[cfg(test)]
            if self.oracle_values.is_some() {
                let end = self
                    .loop_offset
                    .checked_add(value.first().len())
                    .ok_or(Error::Bounds)?;
                self.write_local(self.step, self.loop_offset, value.first(), value.second())?;
                self.loop_offset = end;
                return Ok(());
            }
            let output = self.values[self.program.value_slot(self.step)]
                .as_mut()
                .ok_or(Error::Phase)?;
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
        self.store_local(self.step, secret)
    }

    fn retire_inputs(&mut self) {
        let program = self.program.clone();
        self.retire_reader(program.schedule_access())
            .expect("original retirement geometry");
    }

    fn retire_reader(&mut self, reader: PrivateScheduleRef<'_>) -> Result<(), Error> {
        #[cfg(test)]
        if self.oracle_values.is_none() {
            oracle_trace::retiring(self);
        }
        if reader.node(self.step)?.kind() == Kind::TensorLoop
            && self.loop_offset < reader.node(self.step)?.lanes()
        {
            return Ok(());
        }
        for &id in self.program.retire_at(self.step) {
            #[cfg(test)]
            if let Some(state) = &mut self.oracle_values {
                let span = state.layout.span(id)?;
                if span.location
                    == super::program::schedule_access::physical::Location::GeneratedPublic
                {
                    // Logical last-use was checked against the original Program.
                    // This public recipe never owned a secret slot to retire.
                    reader.generated_len(id)?;
                    continue;
                }
                let (slot, offset) = span.stored()?;
                if let Some(value) = &mut self.values[slot] {
                    use zeroize::Zeroize;
                    value.first[offset..offset + span.lanes].zeroize();
                    value.second[offset..offset + span.lanes].zeroize();
                }
                if state.layout.owners[slot].last_use == self.step {
                    self.values[slot] = None;
                    state.initialized[slot] = 0;
                }
                continue;
            }
            self.values[self.program.value_slot(id)] = None;
        }
        self.step += 1;
        self.loop_offset = 0;
        Ok(())
    }

    fn loop_share(
        &self,
        input: LoopInput,
        n: usize,
        context: Context,
        reader: PrivateScheduleRef<'_>,
    ) -> Result<ReplicatedShare, Error> {
        let op = Operation::new(context, 0, Party::P0, n, 1, puma::Limits::default())?;
        match input {
            LoopInput::Slice(owner) => {
                let (first, second) = self.local_slices(owner)?;
                let end = self.loop_offset.checked_add(n).ok_or(Error::Bounds)?;
                ReplicatedShare::from_local(
                    op,
                    self.party,
                    first.get(self.loop_offset..end).ok_or(Error::Shape)?,
                    second.get(self.loop_offset..end).ok_or(Error::Shape)?,
                    puma::Limits::default(),
                )
            }
            LoopInput::TokenOffsets { owner, index } => {
                let (a, b) = self.word(owner, index, reader)?;
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

    fn word(
        &self,
        owner: usize,
        index: usize,
        reader: PrivateScheduleRef<'_>,
    ) -> Result<(u64, u64), Error> {
        let node = reader.node(owner)?;
        if node.kind() == Kind::GeneratedPublicValues {
            // Only a loan bound to the exact sealed frozen owner can resolve this.
            let value = reader.generated_element(owner, index)? as u64;
            return Ok(public_components(self.party, value));
        }
        if let Ok((key, _digest)) = node.public_input() {
            // Frozen source custody supports complete streamed contractions,
            // never a coefficient gather or unqualified bias fallback.
            #[cfg(test)]
            if self.oracle_values.is_some() {
                return Err(Error::Binding);
            }
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
            let (first, second) = self.local_slices(owner)?;
            Ok((
                *first.get(index).ok_or(Error::Shape)?,
                *second.get(index).ok_or(Error::Shape)?,
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
            if target >= self.program.logical_len() || target < self.step {
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
        #[cfg(test)]
        if self.oracle_values.is_some() {
            return Err(Error::Binding);
        }
        let program = self.program.clone();
        self.advance_reader(before, program.schedule_access())
    }

    // Both storage forms use exactly this transition machine. No new runtime,
    // setup, PID, namespace, PRSS cursor, native frame or completion issuer.
    fn advance_reader(
        &mut self,
        before: Option<usize>,
        reader: PrivateScheduleRef<'_>,
    ) -> Result<Vec<(Party, PeerBytes)>, Error> {
        self.live()?;
        if self.complete()
            || self.imported != self.program.input_count()
            || self.inbox.iter().any(Option::is_some)
        {
            return Err(Error::Phase);
        }
        let active = std::mem::replace(&mut self.active, Active::Retired);
        self.active = match active {
            Active::Product(mut runtime) => {
                let output = runtime.take_output()?;
                self.save(output, reader)?;
                self.retire_reader(reader)?;
                Active::Idle(runtime)
            }
            Active::Tensor(mut executor) => {
                executor.finish_step()?;
                if !executor.complete() {
                    let frames = executor.start_step()?;
                    self.active = Active::Tensor(executor);
                    return tensor_frames(frames);
                }
                let runtime = if matches!(reader.node(self.step)?.kind(), Kind::CheckedRange) {
                    let (runtime, output, valid) = executor.finish_checked()?;
                    let n = output.first().len();
                    if valid.first().len() != n || reader.node(self.step)?.lanes() != 2 * n {
                        return Err(Error::Shape);
                    }
                    let mut both = Secret::new(2 * n)?;
                    both.first[..n].copy_from_slice(output.first());
                    both.second[..n].copy_from_slice(output.second());
                    both.first[n..].copy_from_slice(valid.first());
                    both.second[n..].copy_from_slice(valid.second());
                    self.store_local(self.step, both)?;
                    runtime
                } else {
                    let (runtime, output) = executor.finish()?;
                    self.save(output, reader)?;
                    runtime
                };
                self.retire_reader(reader)?;
                Active::Idle(runtime)
            }
            Active::Lookup(mut bulk) => {
                let node = reader.node(self.step)?;
                let (_, weight, _) = node.public_lookup()?;
                let (key, _digest) = reader.node(*weight)?.public_input()?;
                let file = self.public.as_ref().ok_or(Error::Binding)?.file(key)?;
                bulk.finish_step(file)?;
                if !bulk.complete() {
                    let frames = bulk.start_step()?;
                    self.active = Active::Lookup(bulk);
                    return tensor_frames(frames);
                }
                let (runtime, output) = bulk.finish()?;
                if output.first.len() != reader.node(self.step)?.lanes() {
                    return Err(Error::Shape);
                }
                self.store_local(self.step, output)?;
                self.retire_reader(reader)?;
                Active::Idle(runtime)
            }
            Active::Trie(mut lookup) => {
                let node = reader.node(self.step)?;
                let (_, weight, _) = node.trie()?;
                let (key, _digest) = reader.node(*weight)?.public_input()?;
                let file = self.public.as_ref().ok_or(Error::Binding)?.file(key)?;
                lookup.finish_step(*weight, file)?;
                if !lookup.complete() {
                    let frames = lookup.start_step()?;
                    self.active = Active::Trie(lookup);
                    return Ok(frames);
                }
                let (runtime, output) = lookup.finish()?;
                if output.first.len() != reader.node(self.step)?.lanes() {
                    return Err(Error::Shape);
                }
                self.store_local(self.step, output)?;
                self.retire_reader(reader)?;
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
                if output.first.len() != reader.node(self.step)?.lanes() {
                    return Err(Error::Shape);
                }
                self.store_local(self.step, output)?;
                self.retire_reader(reader)?;
                Active::Idle(runtime)
            }
            Active::Idle(runtime) => Active::Idle(runtime),
            Active::Retired => return Err(Error::Retired),
        };
        self.started = true;
        while self.step < before.unwrap_or(reader.len()) {
            let node = reader.node(self.step)?;
            match node.kind() {
                Kind::Input | Kind::PublicInput => {
                    self.retire_reader(reader)?;
                }
                Kind::PublicValues => {
                    let values = node.public_values()?;
                    let mut out = Secret::new(values.len())?;
                    for (i, value) in values.iter().enumerate() {
                        (out.first[i], out.second[i]) =
                            public_components(self.party, *value as u64);
                    }
                    self.store_local(self.step, out)?;
                    self.retire_reader(reader)?;
                }
                Kind::Gather | Kind::SourceGather => {
                    let refs = node.lane_refs()?;
                    let mut output = Secret::new(node.gather_output_lanes()?)?;
                    for (i, coordinate) in refs.enumerate() {
                        let (owner, index) = coordinate?;
                        (output.first[i], output.second[i]) = self.word(owner, index, reader)?;
                    }
                    self.store_local(self.step, output)?;
                    self.retire_reader(reader)?;
                }
                Kind::PublicProduct => {
                    let (input, weight, rows, inner, columns, transposed) =
                        node.public_product()?;
                    let (key, _digest) = reader.node(*weight)?.public_input()?;
                    let file = self.public.as_ref().ok_or(Error::Binding)?.file(key)?;
                    let (first, second) = self.local_slices(*input)?;
                    let output = super::public_source::multiply_components(
                        file,
                        [first, second],
                        *rows,
                        *inner,
                        *columns,
                        *transposed,
                    )?;
                    self.store_local(self.step, output)?;
                    self.retire_reader(reader)?;
                }
                Kind::Affine => {
                    let (left, right, c) = node.affine()?;
                    let (left_first, left_second) = self.local_slices(*left)?;
                    let right = right.map(|id| self.local_slices(id)).transpose()?;
                    let mut output = Secret::new(node.lanes())?;
                    for i in 0..node.lanes() {
                        output.first[i] = left_first[i]
                            .wrapping_mul(c.left as u64)
                            .wrapping_add(right.map_or(0, |v| v.0[i]).wrapping_mul(c.right as u64))
                            .wrapping_add(if self.party == Party::P0 {
                                c.constant as u64
                            } else {
                                0
                            });
                        output.second[i] = left_second[i]
                            .wrapping_mul(c.left as u64)
                            .wrapping_add(right.map_or(0, |v| v.1[i]).wrapping_mul(c.right as u64))
                            .wrapping_add(if self.party.next() == Party::P0 {
                                c.constant as u64
                            } else {
                                0
                            });
                    }
                    self.store_local(self.step, output)?;
                    self.retire_reader(reader)?;
                }
                Kind::Selection => {
                    let (input, _) = node.selection()?;
                    // Compiler emits a dedicated immediately preceding gather.
                    // Last-use retirement proves no live consumer/output is
                    // invalidated when the tournament compacts that owner.
                    if self.step.checked_sub(1) != Some(*input)
                        || reader.node(*input)?.kind() != Kind::Gather
                        || !self.program.retire_at(self.step).contains(input)
                        || node.lanes() != 1
                    {
                        return Err(Error::Binding);
                    }
                    let Active::Idle(runtime) =
                        std::mem::replace(&mut self.active, Active::Retired)
                    else {
                        return Err(Error::Phase);
                    };
                    let scores = self.values[self.program.value_slot(*input)]
                        .take()
                        .ok_or(Error::Binding)?;
                    let mut selection = selection::StableArgmax::new(
                        runtime,
                        scores,
                        self.program
                            .schedule_access()
                            .selection_owner(self.step)?
                            .clone(),
                    )?;
                    if selection.complete() {
                        let (runtime, output) = selection.finish()?;
                        self.store_local(self.step, output)?;
                        self.retire_reader(reader)?;
                        self.active = Active::Idle(runtime);
                    } else {
                        let frames = selection.start_step()?;
                        self.active = Active::Selection(selection);
                        return Ok(frames);
                    }
                }
                Kind::TrieLookup => {
                    let (input, weight, plan) = node.trie()?;
                    plan.validate_owners(
                        *input,
                        *weight,
                        self.program.schedule_access().source_binding()?,
                    )?;
                    if self.step.checked_sub(1) != Some(*input)
                        || reader.node(*input)?.kind() != Kind::Gather
                        || !self.program.retire_at(self.step).contains(input)
                    {
                        return Err(Error::Binding);
                    }
                    let (key, _digest) = reader.node(*weight)?.public_input()?;
                    let file = self.public.as_ref().ok_or(Error::Binding)?.file(key)?;
                    let context = Context::new(self.session, self.program.digest(), 0)?;
                    let Active::Idle(runtime) =
                        std::mem::replace(&mut self.active, Active::Retired)
                    else {
                        return Err(Error::Phase);
                    };
                    let tokens = self.values[self.program.value_slot(*input)]
                        .take()
                        .ok_or(Error::Binding)?;
                    let mut lookup = trie::TrieLookup::new(
                        runtime,
                        tokens,
                        self.program
                            .schedule_access()
                            .trie_owner(self.step)?
                            .clone(),
                        context,
                        *input,
                        *weight,
                        file,
                    )?;
                    let frames = lookup.start_step()?;
                    self.active = Active::Trie(lookup);
                    return Ok(frames);
                }
                Kind::PublicLookup => {
                    let (input, _, plan) = node.public_lookup()?;
                    let Active::Idle(runtime) =
                        std::mem::replace(&mut self.active, Active::Retired)
                    else {
                        return Err(Error::Phase);
                    };
                    let (first, second) = self.local_slices(*input)?;
                    let mut tokens = Secret::new(first.len())?;
                    tokens.first.copy_from_slice(first);
                    tokens.second.copy_from_slice(second);
                    let mut bulk = lookup::BulkLookup::new(runtime, tokens, plan.clone())?;
                    let frames = bulk.start_step()?;
                    self.active = Active::Lookup(bulk);
                    return tensor_frames(frames);
                }
                Kind::Tensor | Kind::CheckedRange => {
                    let (input, program) = node.tensor()?;
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
                Kind::TensorLoop => {
                    let input = node.loop_input()?;
                    let Active::Idle(runtime) =
                        std::mem::replace(&mut self.active, Active::Retired)
                    else {
                        return Err(Error::Phase);
                    };
                    let selected = node.loop_tile(self.loop_offset)?;
                    if self.loop_offset == 0 {
                        self.ensure_local(self.step, node.lanes())?;
                    }
                    let input =
                        self.loop_share(*input, selected.lanes(), runtime.context(), reader)?;
                    // Each iteration constructs a fresh consuming executor. Its
                    // invocation is bound to the runtime's monotonic PRSS cursor,
                    // so frames and randomness cannot be reused across tiles.
                    let mut executor = tensor::Executor::new(runtime, (**selected).clone(), input)?;
                    let out = executor.start_step()?;
                    self.active = Active::Tensor(executor);
                    return tensor_frames(out);
                }
                Kind::Product => {
                    let (left, right, shape, rescale, norm) = node.product()?;
                    let Active::Idle(mut runtime) =
                        std::mem::replace(&mut self.active, Active::Retired)
                    else {
                        return Err(Error::Phase);
                    };
                    let bounds = InputBounds {
                        left: reader.node(*left)?.bounds(),
                        right: reader.node(*right)?.bounds(),
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
                            (None, true) => runtime.plan(&vec![bounds; node.lanes()])?,
                            (None, false) => {
                                runtime.plan_ring_product(&vec![bounds; node.lanes()])?
                            }
                        }
                    };
                    let (a_first, a_second) = self.local_slices(*left)?;
                    let (b_first, b_second) = self.local_slices(*right)?;
                    let a = plan.share_operand(
                        Operand::Left,
                        self.party,
                        a_first,
                        a_second,
                        puma::Limits::default(),
                    )?;
                    let b = plan.share_operand(
                        Operand::Right,
                        self.party,
                        b_first,
                        b_second,
                        puma::Limits::default(),
                    )?;
                    let out = runtime.begin(a, b)?;
                    self.active = Active::Product(runtime);
                    return arithmetic_frames([Some(out), None, None]);
                }
                Kind::GeneratedPublicValues => {
                    // Owner validation, but no materialization or fake value slot.
                    reader.generated_len(self.step)?;
                    self.retire_reader(reader)?;
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
    pub fn finish(self) -> Result<Zeroizing<Vec<u64>>, Error> {
        self.finish_reader(None)
    }

    fn finish_reader(
        mut self,
        reader: Option<PrivateScheduleRef<'_>>,
    ) -> Result<Zeroizing<Vec<u64>>, Error> {
        self.live()?;
        if !self.complete() {
            return Err(Error::Phase);
        }
        let mut out = Zeroizing::new(vec_zero(self.program.output_coordinates().len())?);
        for (slot, &(id, index)) in out.iter_mut().zip(self.program.output_coordinates()) {
            *slot = match reader {
                Some(reader) => self.word(id, index, reader)?.0,
                None => self.local_slices(id)?.0[index],
            };
        }
        self.cancel();
        Ok(out)
    }

    pub fn finish_frame(self) -> Result<Zeroizing<Vec<u8>>, Error> {
        self.finish_frame_reader(None)
    }

    fn finish_frame_reader(
        self,
        reader: Option<PrivateScheduleRef<'_>>,
    ) -> Result<Zeroizing<Vec<u8>>, Error> {
        let session = self.session;
        let program = self.program.digest();
        let party = self.party;
        let words = self.finish_reader(reader)?;
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

// Same role-specific public sharing used by the original PublicValues arm.
// No randomness, private input, source declaration or runtime reset is involved.
fn public_components(party: Party, value: u64) -> (u64, u64) {
    (
        if party == Party::P0 { value } else { 0 },
        if party.next() == Party::P0 { value } else { 0 },
    )
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

#[cfg(test)]
mod schedule_access_tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn expanded_accessors_borrow_original_recipes_coordinates_and_native_owners() {
        for lanes in [1, 128, 129, 257] {
            let (program, prepared, input, output) = program::prepared_rescale18_worker_fixture(
                lanes,
                Interval::new(-ONE, ONE).unwrap(),
            )
            .unwrap();
            let native_before = super::super::prepared_tensor_map::native_attempts();
            let owners_before = Arc::strong_count(&program);
            assert_eq!(program.logical_len(), program.nodes.len());
            assert_eq!(program.value_slot_count(), program.nodes.len());
            assert_eq!(program.input_count(), program.inputs);
            for position in 0..program.input_count() {
                let (ordinal, recipe) = program.input_at(position);
                assert_eq!(ordinal, position);
                assert!(std::ptr::eq(recipe, &program.nodes[position]));
            }
            for ordinal in 0..program.logical_len() {
                assert_eq!(program.value_slot(ordinal), ordinal);
                assert!(std::ptr::eq(
                    program.recipe(ordinal),
                    &program.nodes[ordinal]
                ));
                assert!(std::ptr::eq(
                    program.recipe_at(ordinal).unwrap(),
                    &program.nodes[ordinal]
                ));
                assert!(std::ptr::eq(
                    program.retire_at(ordinal),
                    program.retire[ordinal].as_slice()
                ));
            }
            assert!(program.recipe_at(program.logical_len()).is_none());
            assert!(program.recipe_at(usize::MAX).is_none());
            assert!(std::ptr::eq(
                program.output_coordinates(),
                program.outputs.as_slice()
            ));
            assert_eq!(program.input_at(0).0, input);
            let Code::TensorLoop {
                program: full,
                tail,
                ..
            } = &program.recipe(output).code
            else {
                panic!("original loop recipe required")
            };
            let mut originals = prepared.programs();
            for stored in std::iter::once(full).chain(tail) {
                assert!(Arc::ptr_eq(stored, originals.next().unwrap()));
            }
            assert!(originals.next().is_none());
            assert_eq!(Arc::strong_count(&program), owners_before);
            assert_eq!(
                super::super::prepared_tensor_map::native_attempts(),
                native_before
            );
        }
    }

    #[test]
    fn expanded_worker_keeps_program_owner_and_original_full_tail_results() {
        for lanes in [1, 128, 129, 257] {
            let (program, _, _, _) = program::prepared_rescale18_worker_fixture(
                lanes,
                Interval::new(-ONE, ONE).unwrap(),
            )
            .unwrap();
            // Exact multiples avoid relying on probabilistic truncation error.
            let clear: Vec<_> = (0..lanes).map(|i| (i as i64 % 3 - 1) * ONE).collect();
            let expected: Vec<_> = clear.iter().map(|value| (value / ONE) as u64).collect();
            let inputs = BTreeMap::from([("local".to_owned(), clear)]);
            let workers = super::super::tests::workers(program.clone(), &inputs);
            for worker in &workers {
                assert!(Arc::ptr_eq(&worker.program, &program));
                assert_eq!(worker.values.len(), program.value_slot_count());
                assert_eq!(worker.imported, program.input_count());
            }
            assert_eq!(super::super::tests::execute(workers), expected);
        }
    }
}
