//! Bounded mixed-mode selection of complete ring32 records.
//!
//! Comparisons use fresh dealer-aided Yao conversions. Conditional record moves
//! use fresh Beaver products. Each primitive party owns only its own shares.
//! This in-process harness supplies correlations serially; it is not a provider
//! state owner, distributed setup, authenticated transport or malicious protocol.
use crate::boolean::{BooleanCircuitBuilder as Builder, BooleanWire};
use crate::shared_arithmetic_reference as mul;
use crate::shared_circuit as gc;
use crate::shared_rescale_reference::{Context, Error};
use crate::token_pruning::Counts;
use sha2::{Digest, Sha256};
use std::time::Instant;
use zeroize::{Zeroize, Zeroizing};

mod batch;

pub const MAX_ALLOCATION: usize = 64 << 20;
const MAX_PRODUCTS: usize = 8 << 20;
const MAX_MATERIAL: u64 = 512 << 20;
const PREDICATE_WIRES: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Selection {
    /// Stable descending nonnegative finite float32 scores, with mandatory recent
    /// rows. Returns selected records in their original order.
    TopK { keep: usize, recent: usize },
    /// Exact unsigned score sums. Discloses retained count, then retained-order
    /// degree bits only when some rows were removed.
    Threshold { threshold: u32, reduction: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub rows: usize,
    pub words: usize,
    pub batch_size: usize,
    /// Use proven score/index/priority bounds in top-k predicate conversions.
    pub bounded_metadata: bool,
    pub selection: Selection,
}

#[derive(Clone, Copy)]
enum Predicate {
    Rank(usize),
    Order(usize),
    Greater(u32),
}

impl Predicate {
    fn words(self) -> usize {
        match self {
            Self::Rank(_) => 6,
            Self::Order(_) => 2,
            Self::Greater(_) => 1,
        }
    }
    fn widths(self) -> Vec<usize> {
        match self {
            Self::Rank(32) => vec![32; 6],
            Self::Rank(index) => vec![31, index, 2, 31, index, 2],
            Self::Order(index) => vec![index; 2],
            Self::Greater(_) => vec![32],
        }
    }
    fn reduce(self, values: &mut [u64]) {
        for (value, bits) in values.iter_mut().zip(self.widths()) {
            *value &= gc::mask(bits);
        }
    }
    fn counts(self) -> Counts {
        let words = self.words() as u64;
        let count = self.widths().iter().sum::<usize>() as u64;
        let width = match self {
            Self::Rank(_) | Self::Order(_) => count / 2,
            Self::Greater(_) => 32,
        };
        Counts {
            input_bits: 2 * count,
            output_bits: 32,
            constants: words
                + 36
                + if matches!(self, Self::Greater(_)) {
                    32
                } else {
                    0
                },
            nots: 64 + 2 * width,
            xors: 3 * (count + 32) + 2 * width,
            ands: 2 * (count + 32) + 3 * width,
        }
    }
    fn resources(self) -> gc::Resources {
        let c = self.counts();
        let count = self.widths().iter().sum::<usize>();
        gc::Resources {
            key_bytes: [
                96 * count + 8,
                17 * count + 2048 + c.evaluator_bytes() as usize,
            ],
            phase_bytes: [
                [gc::HEADER, gc::HEADER + count],
                [gc::HEADER + 48 * count, gc::HEADER],
            ],
            ciphertext_bytes: 32 * c.ands as usize,
            constant_bytes: 16 * c.constants as usize,
            public_metadata_bytes: c.metadata_bytes() as usize + 2 * gc::HEADER,
            allocation_bytes: c.wires() as usize * 384 + 1024 * (count + 32) + (1 << 20),
        }
    }
    fn build(
        self,
        b: &mut Builder,
        x: &[Vec<BooleanWire>],
    ) -> Result<Vec<Vec<BooleanWire>>, String> {
        let bit = match self {
            Self::Rank(_) => {
                let a = [&x[1][..], &x[0][..], &x[2][..]].concat();
                let c = [&x[4][..], &x[3][..], &x[5][..]].concat();
                b.compare_unsigned(&a, &c)?.0
            }
            Self::Order(_) => b.compare_unsigned(&x[0], &x[1])?.0,
            Self::Greater(threshold) => {
                let t = b.constant_word(u128::from(threshold), 32)?;
                b.compare_unsigned(&t, &x[0])?.0
            }
        };
        let zero = b.constant(false)?;
        let mut result = vec![zero; 32];
        result[0] = bit;
        Ok(vec![result])
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Resources {
    pub key_bytes: [u64; 2],
    pub frame_bytes: [u64; 2],
    pub framing_bytes: u64,
    pub rounds: usize,
    pub predicates: usize,
    pub products: usize,
    pub ciphertext_bytes: u64,
    pub constant_bytes: u64,
    pub public_metadata_bytes: u64,
    pub phases: Vec<[usize; 2]>,
}

impl Resources {
    fn predicate_group(&mut self, p: Predicate, count: usize) {
        self.predicate(p, count);
        self.rounds -= 2 * (count - 1);
        self.framing_bytes += (4 * batch::overhead(count)) as u64;
        for bytes in &mut self.frame_bytes {
            *bytes += (2 * batch::overhead(count)) as u64;
        }
    }
    fn movement_group(&mut self, stride: usize, count: usize) {
        self.movement(stride, count);
        let chunks = stride.div_ceil(mul::MAX_LANES);
        self.rounds -= chunks * (count - 1);
        self.framing_bytes += (2 * chunks * batch::overhead(count)) as u64;
        for bytes in &mut self.frame_bytes {
            *bytes += (chunks * batch::overhead(count)) as u64;
        }
    }
    fn predicate(&mut self, p: Predicate, count: usize) {
        let r = p.resources();
        for (out, value) in self.key_bytes.iter_mut().zip(r.key_bytes) {
            *out += value as u64 * count as u64;
        }
        self.ciphertext_bytes += r.ciphertext_bytes as u64 * count as u64;
        self.constant_bytes += r.constant_bytes as u64 * count as u64;
        self.public_metadata_bytes += r.public_metadata_bytes as u64 * count as u64;
        self.predicates += count;
        for phase in r.phase_bytes {
            self.phase(phase, count);
        }
    }
    fn phase(&mut self, phase: [usize; 2], count: usize) {
        self.framing_bytes += (2 * gc::HEADER * count) as u64;
        for (out, value) in self.frame_bytes.iter_mut().zip(phase) {
            *out += value as u64 * count as u64;
        }
        self.rounds += count;
    }
    fn movement(&mut self, stride: usize, swaps: usize) {
        self.framing_bytes +=
            (2 * mul::HEADER_BYTES * stride.div_ceil(mul::MAX_LANES) * swaps) as u64;
        self.products += stride * swaps;
        for value in &mut self.key_bytes {
            *value += (12 * stride * swaps) as u64;
        }
        let chunks = stride.div_ceil(mul::MAX_LANES);
        let bytes = chunks * mul::HEADER_BYTES + 8 * stride;
        for value in &mut self.frame_bytes {
            *value += (bytes * swaps) as u64;
        }
        self.rounds += chunks * swaps;
    }
}

impl Plan {
    fn index_bits(self) -> usize {
        if self.bounded_metadata {
            // Low bits of u32::MAX-row must stay strictly positive even for the
            // last real row; padding uses zero. In particular 256 rows need 9.
            (usize::BITS - self.rows.leading_zeros()).max(2) as usize
        } else {
            32
        }
    }
    fn validate(self) -> Result<(), Error> {
        if !(1..=256).contains(&self.rows)
            || !(1..=8192).contains(&self.words)
            || !matches!(self.batch_size, 1 | 4 | 16)
            || (self.bounded_metadata && !matches!(self.selection, Selection::TopK { .. }))
        {
            return Err(Error::Resource);
        }
        match self.selection {
            Selection::TopK { keep, recent } if keep == 0 || keep > self.rows || recent > keep => {
                Err(Error::Domain)
            }
            Selection::Threshold {
                threshold,
                reduction,
            } if threshold >= reduction => Err(Error::Domain),
            _ => Ok(()),
        }
    }
    fn stride(self) -> usize {
        self.words
            + if matches!(self.selection, Selection::TopK { .. }) {
                3
            } else {
                2
            }
    }

    /// Exact public-geometry projection, including all conversion and movement
    /// material. A projection does not authorize execution.
    pub fn project(self, retained: usize) -> Result<Resources, Error> {
        self.validate()?;
        if retained > self.rows {
            return Err(Error::Domain);
        }
        let mut r = Resources::default();
        match self.selection {
            Selection::TopK { keep, .. } => {
                if retained != keep {
                    return Err(Error::Domain);
                }
                if keep != self.rows {
                    for (rows, p) in [
                        (
                            self.rows.next_power_of_two(),
                            Predicate::Rank(self.index_bits()),
                        ),
                        (
                            keep.next_power_of_two(),
                            Predicate::Order(self.index_bits()),
                        ),
                    ] {
                        for layer in batch::sort_layers(rows) {
                            for pairs in layer.chunks(self.batch_size) {
                                r.predicate_group(p, pairs.len());
                                r.movement_group(self.stride(), pairs.len());
                            }
                        }
                    }
                }
            }
            Selection::Threshold {
                threshold,
                reduction,
            } => {
                for start in (0..self.rows).step_by(self.batch_size) {
                    r.predicate_group(
                        Predicate::Greater(threshold),
                        (self.rows - start).min(self.batch_size),
                    );
                }
                r.phase([gc::HEADER + 4; 2], 1);
                if retained != 0 && retained != self.rows {
                    if self.batch_size == 1 {
                        let swaps = pllm_core::token_pruning::swaps(self.rows, retained)
                            .map_err(|_| Error::Domain)?;
                        r.movement(self.stride(), swaps);
                    } else {
                        for layer in batch::compact_layers(self.rows, retained) {
                            for pairs in layer.chunks(self.batch_size) {
                                r.movement_group(self.stride(), pairs.len());
                            }
                        }
                    }
                    for start in (0..retained).step_by(self.batch_size) {
                        r.predicate_group(
                            Predicate::Greater(reduction),
                            (retained - start).min(self.batch_size),
                        );
                    }
                    r.phase([gc::HEADER + 4 * retained; 2], 1);
                }
            }
        }
        Ok(r)
    }
    /// All possible public counts are priced before sharing inputs or disclosing
    /// the first count. Only one bounded independent batch is live at once.
    pub fn admit(self) -> Result<usize, Error> {
        self.validate()?;
        let mut max_rounds = 0;
        let counts = match self.selection {
            Selection::TopK { keep, .. } => keep..=keep,
            _ => 0..=self.rows,
        };
        for count in counts {
            let r = self.project(count)?;
            if r.products > MAX_PRODUCTS || r.key_bytes.iter().sum::<u64>() > MAX_MATERIAL {
                return Err(Error::Resource);
            }
            max_rounds = max_rounds.max(r.rounds);
        }
        // Caller copies, both share owners, padded sort/return overlap, transcript
        // accounting, the largest predicate circuit and bounded product scratch.
        let bytes = 64 * self.rows.next_power_of_two() * self.stride()
            + 64 * max_rounds
            + PREDICATE_WIRES * 384
            + (8 << 20)
            + (self.batch_size - 1)
                * (PREDICATE_WIRES * 384 + 1024 * (192 + 32) + (1 << 20) + mul::MAX_LANES * 80);
        if bytes > MAX_ALLOCATION {
            return Err(Error::Resource);
        }
        Ok(bytes)
    }
    fn binding(self, context: Context) -> Context {
        let mut hash = Sha256::new();
        hash.update(b"pllm.shared_selection.v1\0");
        hash.update(context.operation);
        hash.update((self.rows as u64).to_le_bytes());
        hash.update((self.words as u64).to_le_bytes());
        if self.batch_size != 1 {
            hash.update(b"batched.v1");
            hash.update((self.batch_size as u64).to_le_bytes());
        }
        if self.bounded_metadata {
            hash.update(b"bounded_metadata.v1");
        }
        match self.selection {
            Selection::TopK { keep, recent } => {
                hash.update([0]);
                hash.update((keep as u64).to_le_bytes());
                hash.update((recent as u64).to_le_bytes());
            }
            Selection::Threshold {
                threshold,
                reduction,
            } => {
                hash.update([1]);
                hash.update(threshold.to_le_bytes());
                hash.update(reduction.to_le_bytes());
            }
        }
        Context {
            operation: hash.finalize().into(),
            ..context
        }
    }
}

/// One party's opaque record owner; never cloned. Errors and cancellation drop
/// the complete remaining owner rather than returning partly selected records.
struct Owner {
    records: Zeroizing<Vec<u32>>,
}

enum DisclosureState {
    Ready(Zeroizing<Vec<u32>>),
    Waiting(Zeroizing<Vec<u32>>),
}

struct Disclosure {
    id: u8,
    binding: [u8; 32],
    nonce: [u8; 32],
    state: Option<DisclosureState>,
}

impl Disclosure {
    fn pair(values: [Zeroizing<Vec<u32>>; 2], context: Context) -> Result<[Self; 2], Error> {
        if values[0].len() != values[1].len() || !(1..=256).contains(&values[0].len()) {
            return Err(Error::Domain);
        }
        let mut hash = Sha256::new();
        hash.update(b"pllm.shared_selection.disclosure.v1\0");
        hash.update(context.plan);
        hash.update(context.session);
        hash.update(context.operation);
        hash.update(context.first_tensor_index.to_le_bytes());
        let binding: [u8; 32] = hash.finalize().into();
        let mut nonce = [0; 32];
        getrandom::fill(&mut nonce).map_err(|_| Error::Randomness)?;
        let [left, right] = values;
        Ok([(0, left), (1, right)].map(|(id, values)| Self {
            id,
            binding,
            nonce,
            state: Some(DisclosureState::Ready(values)),
        }))
    }
    fn start(&mut self) -> Result<Zeroizing<Vec<u8>>, Error> {
        let Some(DisclosureState::Ready(values)) = self.state.take() else {
            return Err(Error::Consumed);
        };
        let mut frame = Zeroizing::new(Vec::with_capacity(gc::HEADER + 4 * values.len()));
        frame.extend(b"PLLMSD01");
        frame.extend(self.binding);
        frame.extend(self.nonce);
        frame.extend([0, self.id]);
        frame.extend((values.len() as u32).to_le_bytes());
        for value in values.iter() {
            frame.extend(value.to_le_bytes());
        }
        self.state = Some(DisclosureState::Waiting(values));
        Ok(frame)
    }
    fn finish(&mut self, frame: &[u8]) -> Result<Vec<u32>, Error> {
        let Some(DisclosureState::Waiting(values)) = self.state.take() else {
            return Err(Error::Consumed);
        };
        if frame.len() != gc::HEADER + 4 * values.len()
            || &frame[..8] != b"PLLMSD01"
            || frame[8..40] != self.binding
            || frame[40..72] != self.nonce
            || frame[72] != 0
            || frame[73] != 1 - self.id
            || u32::from_le_bytes(frame[74..78].try_into().expect("count")) as usize != values.len()
        {
            return Err(Error::Context);
        }
        Ok(values
            .iter()
            .zip(frame[gc::HEADER..].chunks_exact(4))
            .map(|(&own, peer)| {
                own.wrapping_add(u32::from_le_bytes(peer.try_into().expect("word")))
            })
            .collect())
    }
    fn cancel(&mut self) {
        self.state = None;
    }
}

struct Execution {
    owners: [Owner; 2],
    context: Context,
    counter: u64,
    stride: usize,
    batch_size: usize,
    index_bits: usize,
    resources: Resources,
    issuance_seconds: f64,
}

impl Execution {
    fn context(&mut self) -> Result<Context, Error> {
        let first_tensor_index = self
            .context
            .first_tensor_index
            .checked_add(self.counter)
            .ok_or(Error::Domain)?;
        self.counter = self.counter.checked_add(1).ok_or(Error::Domain)?;
        Ok(Context {
            first_tensor_index,
            ..self.context
        })
    }
    fn predicate(
        &mut self,
        p: Predicate,
        mut input: [Zeroizing<Vec<u64>>; 2],
    ) -> Result<[u32; 2], Error> {
        let context = self.context()?;
        let start = Instant::now();
        let ([mut a, mut b], resources) =
            gc::issue_widths(&p.widths(), 32, 1, PREDICATE_WIRES, context, |b, x| {
                p.build(b, x)
            })?;
        self.issuance_seconds += start.elapsed().as_secs_f64();
        if resources != p.resources() {
            return Err(Error::Resource);
        }
        self.resources.predicate(p, 1);
        for values in &mut input {
            p.reduce(values);
        }
        let result = (|| {
            let mut frames = [a.start(&input[0])?, b.start(&input[1])?];
            for expected in resources.phase_bytes {
                if frames.each_ref().map(|f| f.len()) != expected {
                    return Err(Error::Context);
                }
                self.resources.phases.push(expected);
                let left = a.advance(&frames[1])?;
                let right = b.advance(&frames[0])?;
                match (left, right) {
                    (gc::Progress::Open(left), gc::Progress::Open(right)) => frames = [left, right],
                    (gc::Progress::Complete(left), gc::Progress::Complete(right)) => {
                        return Ok([left[0] as u32, right[0] as u32])
                    }
                    _ => return Err(Error::Context),
                }
            }
            Err(Error::Context)
        })();
        a.cancel();
        b.cancel();
        result
    }
    fn swap(&mut self, left: usize, right: usize, flag: [u32; 2]) -> Result<(), Error> {
        for start in (0..self.stride).step_by(mul::MAX_LANES) {
            let len = (self.stride - start).min(mul::MAX_LANES);
            let context = self.context()?;
            let clock = Instant::now();
            let [mut a, mut b] = mul::issue_reference(32, len, context)?;
            self.issuance_seconds += clock.elapsed().as_secs_f64();
            let inputs: [Zeroizing<Vec<u32>>; 2] = std::array::from_fn(|id| {
                Zeroizing::new(
                    (start..start + len)
                        .map(|i| {
                            self.owners[id].records[right * self.stride + i]
                                .wrapping_sub(self.owners[id].records[left * self.stride + i])
                        })
                        .collect(),
                )
            });
            let result = (|| {
                let left_frame = a.start(&Zeroizing::new(vec![flag[0]; len]), &inputs[0])?;
                let right_frame = b.start(&Zeroizing::new(vec![flag[1]; len]), &inputs[1])?;
                self.resources
                    .phases
                    .push([left_frame.len(), right_frame.len()]);
                Ok::<_, Error>([a.finish(&right_frame)?, b.finish(&left_frame)?])
            })();
            a.cancel();
            b.cancel();
            let movement = result?;
            for (owner, movement) in self.owners.iter_mut().zip(movement) {
                for (i, &delta) in (start..start + len).zip(movement.iter()) {
                    let a = left * self.stride + i;
                    let b = right * self.stride + i;
                    owner.records[a] = owner.records[a].wrapping_add(delta);
                    owner.records[b] = owner.records[b].wrapping_sub(delta);
                }
            }
        }
        self.resources.movement(self.stride, 1);
        Ok(())
    }
    fn predicates(
        &mut self,
        p: Predicate,
        mut inputs: Vec<[Zeroizing<Vec<u64>>; 2]>,
    ) -> Result<Vec<[u32; 2]>, Error> {
        if inputs.len() == 1 {
            return Ok(vec![self.predicate(p, inputs.into_iter().next().unwrap())?]);
        }
        if inputs.is_empty() || inputs.len() > self.batch_size {
            return Err(Error::Resource);
        }
        let r = p.resources();
        let mut parties = batch::Predicates(Vec::new());
        for _ in &inputs {
            let context = self.context()?;
            let start = Instant::now();
            let (pair, actual) =
                gc::issue_widths(&p.widths(), 32, 1, PREDICATE_WIRES, context, |b, x| {
                    p.build(b, x)
                })?;
            self.issuance_seconds += start.elapsed().as_secs_f64();
            parties.0.push(pair);
            if actual != r {
                return Err(Error::Resource);
            }
        }
        self.resources.predicate_group(p, inputs.len());
        for input in &mut inputs {
            for values in input {
                p.reduce(values);
            }
        }
        let mut frames = [Vec::new(), Vec::new()];
        for (pair, input) in parties.0.iter_mut().zip(&inputs) {
            for id in 0..2 {
                frames[id].push(pair[id].start(&input[id])?);
            }
        }
        for (phase, sizes) in r.phase_bytes.into_iter().enumerate() {
            let packets = [batch::pack(&frames[0])?, batch::pack(&frames[1])?];
            let decoded = [
                batch::unpack(&packets[0], &vec![sizes[0]; inputs.len()])?,
                batch::unpack(&packets[1], &vec![sizes[1]; inputs.len()])?,
            ];
            self.resources
                .phases
                .push(packets.each_ref().map(|p| p.len()));
            let mut next = [Vec::new(), Vec::new()];
            let mut output = Vec::new();
            for (i, pair) in parties.0.iter_mut().enumerate() {
                match (
                    pair[0].advance(decoded[1][i])?,
                    pair[1].advance(decoded[0][i])?,
                ) {
                    (gc::Progress::Open(a), gc::Progress::Open(b)) if phase == 0 => {
                        next[0].push(a);
                        next[1].push(b);
                    }
                    (gc::Progress::Complete(a), gc::Progress::Complete(b)) if phase == 1 => {
                        output.push([a[0] as u32, b[0] as u32]);
                    }
                    _ => return Err(Error::Context),
                }
            }
            if phase == 1 {
                return Ok(output);
            }
            frames = next;
        }
        Err(Error::Context)
    }

    fn swaps(&mut self, pairs: &[(usize, usize)], flags: &[[u32; 2]]) -> Result<(), Error> {
        if pairs.len() != flags.len() || pairs.is_empty() || pairs.len() > self.batch_size {
            return Err(Error::Resource);
        }
        // Reject a dependent batch before any correlated material is issued.
        let mut seen = std::collections::BTreeSet::new();
        for &(a, b) in pairs {
            if !seen.insert(a) || !seen.insert(b) {
                return Err(Error::Context);
            }
        }
        if pairs.len() == 1 {
            return self.swap(pairs[0].0, pairs[0].1, flags[0]);
        }
        for start in (0..self.stride).step_by(mul::MAX_LANES) {
            let len = (self.stride - start).min(mul::MAX_LANES);
            let mut parties = batch::Products(Vec::new());
            let mut frames = [Vec::new(), Vec::new()];
            for (&(left, right), flag) in pairs.iter().zip(flags) {
                let context = self.context()?;
                let clock = Instant::now();
                parties.0.push(mul::issue_reference(32, len, context)?);
                self.issuance_seconds += clock.elapsed().as_secs_f64();
                let pair = parties.0.last_mut().unwrap();
                for id in 0..2 {
                    let input = Zeroizing::new(
                        (start..start + len)
                            .map(|i| {
                                self.owners[id].records[right * self.stride + i]
                                    .wrapping_sub(self.owners[id].records[left * self.stride + i])
                            })
                            .collect::<Vec<_>>(),
                    );
                    frames[id].push(pair[id].start(&Zeroizing::new(vec![flag[id]; len]), &input)?);
                }
            }
            let packets = [batch::pack(&frames[0])?, batch::pack(&frames[1])?];
            let sizes = vec![mul::HEADER_BYTES + 8 * len; pairs.len()];
            let decoded = [
                batch::unpack(&packets[0], &sizes)?,
                batch::unpack(&packets[1], &sizes)?,
            ];
            self.resources
                .phases
                .push(packets.each_ref().map(|p| p.len()));
            for (i, (&(left, right), pair)) in pairs.iter().zip(&mut parties.0).enumerate() {
                let movement = [
                    pair[0].finish(decoded[1][i])?,
                    pair[1].finish(decoded[0][i])?,
                ];
                for (owner, movement) in self.owners.iter_mut().zip(movement) {
                    for (column, &delta) in (start..start + len).zip(movement.iter()) {
                        let a = left * self.stride + column;
                        let b = right * self.stride + column;
                        owner.records[a] = owner.records[a].wrapping_add(delta);
                        owner.records[b] = owner.records[b].wrapping_sub(delta);
                    }
                }
            }
        }
        self.resources.movement_group(self.stride, pairs.len());
        Ok(())
    }

    fn sort(&mut self, rows: usize, rank: bool) -> Result<(), Error> {
        for layer in batch::sort_layers(rows) {
            for pairs in layer.chunks(self.batch_size) {
                let mut inputs = Vec::new();
                for &(a, b) in pairs {
                    let input = std::array::from_fn(|id| {
                        let owner = &self.owners[id].records;
                        Zeroizing::new(if rank {
                            owner[a * self.stride..a * self.stride + 3]
                                .iter()
                                .chain(&owner[b * self.stride..b * self.stride + 3])
                                .map(|&v| u64::from(v))
                                .collect()
                        } else {
                            vec![
                                u64::from(owner[a * self.stride + 1]),
                                u64::from(owner[b * self.stride + 1]),
                            ]
                        })
                    });
                    inputs.push(input);
                }
                let flags = self.predicates(
                    if rank {
                        Predicate::Rank(self.index_bits)
                    } else {
                        Predicate::Order(self.index_bits)
                    },
                    inputs,
                )?;
                self.swaps(pairs, &flags)?;
            }
        }
        Ok(())
    }
    fn disclose(&mut self, values: [Zeroizing<Vec<u32>>; 2]) -> Result<Vec<u32>, Error> {
        let context = self.context()?;
        let [mut a, mut b] = Disclosure::pair(values, context)?;
        let result = (|| {
            let frames = [a.start()?, b.start()?];
            let lengths = frames.each_ref().map(|f| f.len());
            self.resources.phase(lengths, 1);
            self.resources.phases.push(lengths);
            let left = a.finish(&frames[1])?;
            let right = b.finish(&frames[0])?;
            if left != right {
                return Err(Error::Context);
            }
            Ok(left)
        })();
        a.cancel();
        b.cancel();
        result
    }
}

pub struct Measurement {
    pub output: Zeroizing<Vec<u32>>,
    pub retained: usize,
    pub high_degree: Option<Vec<bool>>,
    pub resources: Resources,
    pub allocation_bytes: usize,
    pub sharing_seconds: f64,
    pub issuance_seconds: f64,
    pub online_seconds: f64,
}

/// In-process benchmark harness. The original plaintexts belong to the caller;
/// intermediate decisions and moved records are never reconstructed. Only the
/// authorized disclosure and the final independent oracle boundary are opened.
pub fn run_reference(
    plan: Plan,
    scores: &[u32],
    words: &[u32],
    context: Context,
) -> Result<Measurement, Error> {
    let allocation_bytes = plan.admit()?;
    if scores.len() != plan.rows || words.len() != plan.rows * plan.words {
        return Err(Error::Domain);
    }
    if matches!(plan.selection, Selection::TopK { .. }) && scores.iter().any(|&x| x > 0x7f7f_ffff) {
        return Err(Error::Domain);
    }
    context
        .first_tensor_index
        .checked_add(1 << 24)
        .ok_or(Error::Domain)?;
    let start = Instant::now();
    let mut nonce = [0; 32];
    getrandom::fill(&mut nonce).map_err(|_| Error::Randomness)?;
    let mut hash = Sha256::new();
    hash.update(b"pllm.shared_selection.attempt.v1\0");
    hash.update(context.session);
    hash.update(nonce);
    let context = Context {
        session: hash.finalize().into(),
        ..context
    };
    let stride = plan.stride();
    let rows = if matches!(plan.selection, Selection::TopK { .. }) {
        plan.rows.next_power_of_two()
    } else {
        plan.rows
    };
    let mut owners: [Owner; 2] = std::array::from_fn(|_| Owner {
        records: Zeroizing::new(Vec::with_capacity(rows * stride)),
    });
    for row in 0..rows {
        let mut record = Zeroizing::new(vec![0; stride]);
        if row < plan.rows {
            let offset = match plan.selection {
                Selection::TopK { recent, .. } => {
                    record[0] = scores[row];
                    record[1] = u32::MAX - row as u32;
                    record[2] = 1 + u32::from(row >= plan.rows - recent);
                    3
                }
                Selection::Threshold { .. } => {
                    record[1] = scores[row];
                    2
                }
            };
            record[offset..].copy_from_slice(&words[row * plan.words..(row + 1) * plan.words]);
        }
        for &value in record.iter() {
            let share = gc::random_word()? as u32;
            owners[0].records.push(share);
            owners[1].records.push(value.wrapping_sub(share));
        }
    }
    let sharing_seconds = start.elapsed().as_secs_f64();
    let mut execution = Execution {
        owners,
        context: plan.binding(context),
        counter: 0,
        stride,
        batch_size: plan.batch_size,
        index_bits: plan.index_bits(),
        resources: Resources::default(),
        issuance_seconds: 0.0,
    };
    let mut high_degree = None;
    let retained = match plan.selection {
        Selection::TopK { keep, .. } => {
            if keep != plan.rows {
                execution.sort(rows, true)?;
                for owner in &mut execution.owners {
                    owner.records[keep * stride..].zeroize();
                    owner.records.truncate(keep * stride);
                    owner.records.resize(keep.next_power_of_two() * stride, 0);
                }
                execution.sort(keep.next_power_of_two(), false)?;
            }
            keep
        }
        Selection::Threshold {
            threshold,
            reduction,
        } => {
            let mut sums = [0_u32; 2];
            for start in (0..rows).step_by(plan.batch_size) {
                let end = (start + plan.batch_size).min(rows);
                let inputs = (start..end)
                    .map(|row| {
                        std::array::from_fn(|id| {
                            Zeroizing::new(vec![u64::from(
                                execution.owners[id].records[row * stride + 1],
                            )])
                        })
                    })
                    .collect();
                let flags = execution.predicates(Predicate::Greater(threshold), inputs)?;
                for (row, flag) in (start..end).zip(flags) {
                    for id in 0..2 {
                        execution.owners[id].records[row * stride] = flag[id];
                        sums[id] = sums[id].wrapping_add(flag[id]);
                    }
                }
            }
            let retained = execution.disclose(sums.map(|v| Zeroizing::new(vec![v])))?[0] as usize;
            if retained > rows {
                return Err(Error::Domain);
            }
            if retained < rows {
                high_degree = Some(Vec::new());
            }
            if retained > 0 && retained < rows {
                let layers = if plan.batch_size == 1 {
                    (0..rows - retained)
                        .flat_map(|pass| (0..rows - pass - 1).map(|row| vec![(row, row + 1)]))
                        .collect()
                } else {
                    batch::compact_layers(rows, retained)
                };
                for layer in layers {
                    for pairs in layer.chunks(plan.batch_size) {
                        let flags: Vec<_> = pairs
                            .iter()
                            .map(|&(row, _)| {
                                [
                                    1_u32.wrapping_sub(execution.owners[0].records[row * stride]),
                                    0_u32.wrapping_sub(execution.owners[1].records[row * stride]),
                                ]
                            })
                            .collect();
                        execution.swaps(pairs, &flags)?;
                    }
                }
                let mut degrees: [Zeroizing<Vec<u32>>; 2] =
                    std::array::from_fn(|_| Zeroizing::new(Vec::new()));
                for start in (0..retained).step_by(plan.batch_size) {
                    let inputs = (start..(start + plan.batch_size).min(retained))
                        .map(|row| {
                            std::array::from_fn(|id| {
                                Zeroizing::new(vec![u64::from(
                                    execution.owners[id].records[row * stride + 1],
                                )])
                            })
                        })
                        .collect();
                    for bit in execution.predicates(Predicate::Greater(reduction), inputs)? {
                        for id in 0..2 {
                            degrees[id].push(bit[id]);
                        }
                    }
                }
                let values = execution.disclose(degrees)?;
                if values.iter().any(|&v| v > 1) {
                    return Err(Error::Domain);
                }
                high_degree = Some(values.into_iter().map(|v| v == 1).collect());
            }
            retained
        }
    };
    let expected = plan.project(retained)?;
    let phases = std::mem::take(&mut execution.resources.phases);
    if execution.resources != expected
        || phases.len() != expected.rounds
        || (0..2)
            .any(|id| phases.iter().map(|p| p[id] as u64).sum::<u64>() != expected.frame_bytes[id])
    {
        return Err(Error::Resource);
    }
    execution.resources.phases = phases;
    let offset = stride - plan.words;
    let mut output = Zeroizing::new(Vec::with_capacity(retained * plan.words));
    for row in 0..retained {
        for i in offset..stride {
            output.push(
                execution.owners[0].records[row * stride + i]
                    .wrapping_add(execution.owners[1].records[row * stride + i]),
            );
        }
    }
    Ok(Measurement {
        output,
        retained,
        high_degree,
        allocation_bytes,
        sharing_seconds,
        online_seconds: (start.elapsed().as_secs_f64()
            - sharing_seconds
            - execution.issuance_seconds)
            .max(0.0),
        issuance_seconds: execution.issuance_seconds,
        resources: execution.resources,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn context() -> Context {
        Context {
            plan: [1; 32],
            session: [2; 32],
            operation: [3; 32],
            first_tensor_index: 0,
        }
    }

    #[test]
    fn exact_resource_projection_matches_each_issued_predicate() {
        for p in [
            Predicate::Rank(32),
            Predicate::Rank(2),
            Predicate::Rank(9),
            Predicate::Order(32),
            Predicate::Order(2),
            Predicate::Order(9),
            Predicate::Greater(0),
            Predicate::Greater(u32::MAX),
        ] {
            let (_, actual) =
                gc::issue_widths(&p.widths(), 32, 1, PREDICATE_WIRES, context(), |b, x| {
                    p.build(b, x)
                })
                .unwrap();
            assert_eq!(actual, p.resources());
        }
    }
    #[test]
    fn stable_top_k_recent_ties_padding_and_complete_words_match_sort() {
        for rows in [1_usize, 3, 8, 17] {
            for keep in [1, rows.div_ceil(2), rows] {
                for recent in [0, keep.min(2)] {
                    let scores: Vec<_> =
                        (0..rows).map(|i| ((i * 7 % 5) as f32).to_bits()).collect();
                    let words: Vec<_> = (0..rows * 5)
                        .map(|i| (i as u32).wrapping_mul(0x8765_4321))
                        .collect();
                    let p = Plan {
                        rows,
                        words: 5,
                        batch_size: 1,
                        bounded_metadata: false,
                        selection: Selection::TopK { keep, recent },
                    };
                    let mut indices: Vec<_> = (0..rows).collect();
                    indices.sort_by_key(|&i| {
                        (
                            std::cmp::Reverse(i >= rows - recent),
                            std::cmp::Reverse(scores[i]),
                            i,
                        )
                    });
                    indices.truncate(keep);
                    indices.sort_unstable();
                    let expected: Vec<_> = indices
                        .into_iter()
                        .flat_map(|i| words[i * 5..i * 5 + 5].iter().copied())
                        .collect();
                    let actual = run_reference(p, &scores, &words, context()).unwrap();
                    assert_eq!(*actual.output, expected);
                    assert_eq!(actual.high_degree, None);
                }
            }
        }
    }
    #[test]
    fn every_small_keep_mask_is_stable_with_exact_disclosed_degree() {
        for mask in 0..32 {
            let scores: Vec<_> = (0..5)
                .map(|i| {
                    if mask & (1 << i) == 0 {
                        10
                    } else {
                        11 + i * 10
                    }
                })
                .collect();
            let words: Vec<_> = (0..15_u32).map(|i| i.wrapping_mul(0xaaaa_aaaa)).collect();
            let p = Plan {
                rows: 5,
                words: 3,
                batch_size: 1,
                bounded_metadata: false,
                selection: Selection::Threshold {
                    threshold: 10,
                    reduction: 25,
                },
            };
            let selected: Vec<_> = (0..5).filter(|&i| scores[i] > 10).collect();
            let expected: Vec<_> = selected
                .iter()
                .flat_map(|&i| words[i * 3..i * 3 + 3].iter().copied())
                .collect();
            let actual = run_reference(p, &scores, &words, context()).unwrap();
            assert_eq!(*actual.output, expected);
            assert_eq!(
                actual.high_degree,
                (selected.len() < 5).then(|| selected.iter().map(|&i| scores[i] > 25).collect())
            );
        }
    }
    #[test]
    fn full_width_movement_and_worst_case_admission_precede_issuance() {
        let p = Plan {
            rows: 4,
            words: 1025,
            batch_size: 1,
            bounded_metadata: false,
            selection: Selection::TopK { keep: 2, recent: 0 },
        };
        let scores = [0., 3., 1., 2.].map(f32::to_bits);
        let words: Vec<_> = (0..4100_u32).map(|i| i.wrapping_mul(0x8000_1234)).collect();
        let actual = run_reference(p, &scores, &words, context()).unwrap();
        assert_eq!(
            *actual.output,
            [&words[1025..2050], &words[3075..]].concat()
        );
        assert!(Plan {
            rows: 256,
            words: 8192,
            batch_size: 1,
            bounded_metadata: false,
            selection: Selection::Threshold {
                threshold: 1,
                reduction: 2
            }
        }
        .admit()
        .is_err());
        assert!(run_reference(p, &[f32::NAN.to_bits(); 4], &words, context()).is_err());
        assert!(run_reference(p, &scores, &words[..9], context()).is_err());
        assert!(run_reference(
            p,
            &scores,
            &words,
            Context {
                first_tensor_index: u64::MAX,
                ..context()
            }
        )
        .is_err());
    }
    #[test]
    fn batched_sort_and_wavefront_preserve_complete_words_and_material() {
        for selection in [
            Selection::TopK { keep: 7, recent: 2 },
            Selection::Threshold {
                threshold: 2,
                reduction: 5,
            },
        ] {
            let scores: Vec<_> = (0..17)
                .map(|i| {
                    if matches!(selection, Selection::TopK { .. }) {
                        ((i * 7 % 9) as f32).to_bits()
                    } else {
                        i * 7 % 9
                    }
                })
                .collect();
            let words: Vec<_> = (0..17 * 5_u32)
                .map(|i| i.wrapping_mul(0xabcd_8123))
                .collect();
            let p = Plan {
                rows: 17,
                words: 5,
                batch_size: 1,
                bounded_metadata: false,
                selection,
            };
            let serial = run_reference(p, &scores, &words, context()).unwrap();
            for batch_size in [4, 16] {
                let batched =
                    run_reference(Plan { batch_size, ..p }, &scores, &words, context()).unwrap();
                assert_eq!(serial.output, batched.output);
                assert_eq!(serial.high_degree, batched.high_degree);
                assert_eq!(serial.resources.key_bytes, batched.resources.key_bytes);
                assert!(batched.resources.rounds < serial.resources.rounds);
                assert_eq!(serial.resources.products, batched.resources.products);
                assert_eq!(
                    serial.resources.frame_bytes.iter().sum::<u64>()
                        - serial.resources.framing_bytes,
                    batched.resources.frame_bytes.iter().sum::<u64>()
                        - batched.resources.framing_bytes
                );
            }
        }
        // Every five-row keep mask, including zero/all and ties at threshold.
        for mask in 0..32 {
            let scores: Vec<_> = (0..5)
                .map(|i| if mask & (1 << i) == 0 { 10 } else { 20 })
                .collect();
            let words = (0..15).collect::<Vec<_>>();
            let p = Plan {
                rows: 5,
                words: 3,
                batch_size: 16,
                bounded_metadata: false,
                selection: Selection::Threshold {
                    threshold: 10,
                    reduction: 15,
                },
            };
            let actual = run_reference(p, &scores, &words, context()).unwrap();
            let expected: Vec<_> = (0..5)
                .filter(|&i| mask & (1 << i) != 0)
                .flat_map(|i| words[i * 3..i * 3 + 3].iter().copied())
                .collect();
            assert_eq!(*actual.output, expected);
        }
        let p = Plan {
            rows: 256,
            words: 257,
            batch_size: 16,
            bounded_metadata: false,
            selection: Selection::TopK {
                keep: 128,
                recent: 2,
            },
        };
        assert!(p.admit().unwrap() <= MAX_ALLOCATION);
        assert_eq!(p.project(128).unwrap().rounds, 1200);
        assert!(Plan {
            batch_size: 17,
            ..p
        }
        .admit()
        .is_err());
    }

    #[test]
    fn bounded_metadata_preserves_extreme_scores_padding_ties_and_full_words() {
        for rows in [1_usize, 3, 16, 17, 256] {
            let scores: Vec<_> = (0..rows)
                .map(|i| [0, 1, f32::MAX.to_bits(), 1.0_f32.to_bits()][i % 4])
                .collect();
            let words: Vec<_> = (0..rows * 3)
                .map(|i| (i as u32).wrapping_mul(0x8000_1234))
                .collect();
            for batch_size in [1, 16] {
                let keep = rows.div_ceil(2);
                let recent = keep.min(2);
                let p = Plan {
                    rows,
                    words: 3,
                    batch_size,
                    bounded_metadata: true,
                    selection: Selection::TopK { keep, recent },
                };
                let mut order: Vec<_> = (0..rows).collect();
                order.sort_by_key(|&i| {
                    (
                        std::cmp::Reverse(i >= rows - recent),
                        std::cmp::Reverse(scores[i]),
                        i,
                    )
                });
                order.truncate(keep);
                order.sort_unstable();
                let expected: Vec<_> = order
                    .iter()
                    .flat_map(|&i| words[3 * i..3 * i + 3].iter().copied())
                    .collect();
                let actual = run_reference(p, &scores, &words, context()).unwrap();
                assert_eq!(*actual.output, expected);
                let mut measured = actual.resources.clone();
                measured.phases.clear();
                assert_eq!(measured, p.project(keep).unwrap());
                let full = Plan {
                    bounded_metadata: false,
                    ..p
                };
                assert_ne!(
                    p.binding(context()).operation,
                    full.binding(context()).operation
                );
                if keep < rows {
                    assert!(
                        actual.resources.key_bytes.iter().sum::<u64>()
                            < full.project(keep).unwrap().key_bytes.iter().sum::<u64>()
                    );
                }
                for invalid in [
                    (-0.0_f32).to_bits(),
                    (-1.0_f32).to_bits(),
                    f32::INFINITY.to_bits(),
                    f32::NAN.to_bits(),
                ] {
                    assert!(run_reference(p, &vec![invalid; rows], &words, context()).is_err());
                }
            }
        }
    }

    #[test]
    fn batch_envelopes_reject_lengths_and_children_reject_permutation() {
        let mut guard = batch::Products(Vec::new());
        let mut frames = Vec::new();
        for _ in 0..4 {
            let mut pair = mul::issue_reference(32, 2, context()).unwrap();
            frames.push(pair[0].start(&[1, 2], &[3, 4]).unwrap());
            pair[1].start(&[9, 8], &[7, 6]).unwrap();
            guard.0.push(pair);
        }
        let packet = batch::pack(&frames).unwrap();
        let sizes = vec![mul::HEADER_BYTES + 16; 4];
        assert_eq!(
            batch::unpack(&packet, &sizes).unwrap(),
            frames.iter().map(Vec::as_slice).collect::<Vec<_>>()
        );
        for byte in [0, 8, 12, 16] {
            let mut bad = packet.clone();
            bad[byte] ^= 1;
            assert!(batch::unpack(&bad, &sizes).is_err());
        }
        assert!(batch::unpack(&packet[..packet.len() - 1], &sizes).is_err());
        // Same caller context still gets fresh issuance nonces in every child.
        assert!(guard.0[1][1].finish(&frames[0]).is_err());
        assert!(guard.0[1][1].finish(&frames[1]).is_err());
    }

    #[test]
    fn disclosure_frames_bind_context_role_shape_and_burn_every_attempt() {
        for byte in [0, 8, 40, 72, 73, 74, gc::HEADER + 4] {
            let mut parties = Disclosure::pair(
                [Zeroizing::new(vec![u32::MAX]), Zeroizing::new(vec![3])],
                context(),
            )
            .unwrap();
            let mut frame = parties[0].start().unwrap();
            parties[1].start().unwrap();
            if byte == frame.len() {
                frame.push(0);
            } else {
                frame[byte] ^= 1;
            }
            assert!(parties[1].finish(&frame).is_err());
            assert!(parties[1].finish(&frame).is_err());
            parties[0].cancel();
            assert!(parties[0].start().is_err());
        }
        let [mut a, mut b] = Disclosure::pair(
            [Zeroizing::new(vec![u32::MAX]), Zeroizing::new(vec![3])],
            context(),
        )
        .unwrap();
        let left = a.start().unwrap();
        let right = b.start().unwrap();
        assert_eq!(a.finish(&right).unwrap(), vec![2]);
        assert_eq!(b.finish(&left).unwrap(), vec![2]);
        assert!(a.finish(&right).is_err());
        assert!(b.start().is_err());
        let [mut a, mut b] = Disclosure::pair(
            [Zeroizing::new(vec![0]), Zeroizing::new(vec![2])],
            context(),
        )
        .unwrap();
        a.start().unwrap();
        assert!(a.start().is_err());
        assert!(a.finish(&right).is_err());
        b.start().unwrap();
        assert!(b.finish(&left).is_err());
    }
}
