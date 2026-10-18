use super::*;
use multiply::{Operand, PartyRuntime, Status};
use puma::{Context, Message, MessagePhase, Operation, Party, ReplicatedShare};

pub enum PeerBytes {
    Arithmetic(puma::wire::Encoded<Message>),
    Tensor(puma::wire::Encoded<tensor::Frame>),
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
    started: bool,
    imported: usize,
    imported_words: usize,
    session: [u8; 32],
    pid: u32,
    public: Option<PublicSource>,
}

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
    pub fn cancel(&mut self) {
        self.active = Active::Retired;
        self.inbox = [None, None];
        self.values.clear();
        self.public = None;
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
        let mut secret = Secret::new(value.first().len())?;
        secret.first.copy_from_slice(value.first());
        secret.second.copy_from_slice(value.second());
        self.values[self.step] = Some(secret);
        Ok(())
    }

    fn retire_inputs(&mut self) {
        for &id in &self.program.retire[self.step] {
            self.values[id] = None;
        }
        self.step += 1;
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
        let result = self.advance_inner();
        self.checked(result)
    }
    fn advance_inner(&mut self) -> Result<Vec<(Party, PeerBytes)>, Error> {
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
                let (runtime, output) = executor.finish()?;
                self.save(output)?;
                self.retire_inputs();
                Active::Idle(runtime)
            }
            Active::Idle(runtime) => Active::Idle(runtime),
            Active::Retired => return Err(Error::Retired),
        };
        self.started = true;
        while self.step < self.program.nodes.len() {
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
                    for (i, &(owner, index)) in items.iter().enumerate() {
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
                Code::Tensor { input, program } => {
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
                Code::Product {
                    left,
                    right,
                    shape,
                    rescale,
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
                    let plan = match (shape, rescale) {
                        (Some(shape), true) => runtime.plan_matmul(*shape, bounds)?,
                        (Some(shape), false) => runtime.plan_ring_matmul(*shape, bounds)?,
                        (None, true) => runtime.plan(&vec![bounds; node.lanes])?,
                        (None, false) => runtime.plan_ring_product(&vec![bounds; node.lanes])?,
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
