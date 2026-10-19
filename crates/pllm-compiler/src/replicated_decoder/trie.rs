//! Shared-prefix public lookup, enabled only through explicit parent integration.
//! The compiler MUST authenticate token-domain and checked-source provenance.
//! This module cannot validate those facts by observing private token components.
use super::PeerBytes;
use crate::replicated_decoder::*;
use multiply::{boolean, boolean_to_arithmetic as b2a, conversion, Operand, PartyRuntime, Status};
use pllm_core::public_file::PublicFile;
use puma::{Message, MessagePhase, Party};

pub(in crate::replicated_decoder) const PROFILE: &str =
    "pllm.replicated-decoder.lookup.trie-prefix.raw-pum1.ring64.v1";
const ROWS: usize = 8;
const LANES: usize = 128;
const MAX_DEPTH: usize = 18;
const MAX_VOCABULARY: usize = 1 << MAX_DEPTH;

/// These are public proof ROUTE commitments, NEVER hashes of private ID values.
/// Neither variant is evidence by itself: the trusted model compiler constructs
/// it only after validated initial-ID import or native full-head stable selection.
#[derive(Clone, Copy)]
pub(in crate::replicated_decoder) enum DomainOrigin {
    ValidatedInitialIds([u8; 32]),
    NativeStableSelection([u8; 32]),
}
impl DomainOrigin {
    fn parts(self) -> (u8, [u8; 32]) {
        match self {
            Self::ValidatedInitialIds(witness) => (0, witness),
            Self::NativeStableSelection(witness) => (1, witness),
        }
    }
}

/// The caller has already run PublicSource::check against this source manifest.
/// Owner is the original immutable PublicInput node, NOT an activation/view.
#[derive(Clone, Copy)]
pub(in crate::replicated_decoder) struct SourceBinding {
    pub owner: usize,
    pub source_commitment: [u8; 32],
    pub file_digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, Default)]
struct Level {
    parents: u32,
    branches: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::replicated_decoder) struct LogicalCost {
    pub secret_products: u64,
    /// Two component subtractions per product plus the root complement.
    pub sibling_component_subtractions: u64,
    pub forced_component_copies: u64,
    pub selector_component_writes: u64,
    pub public_ring64_macs: u64,
    pub table_bytes: u64,
    /// All parties/directions; no HTTP/MessagePack framing implied.
    pub a2b_frames: u64,
    pub b2a_frames: u64,
    pub product_frames: u64,
    /// Primitive completion barriers plus bounded public streaming tiles.
    pub quiescence_barriers: u64,
}

pub(in crate::replicated_decoder) struct Plan {
    rows: usize,
    vocabulary: usize,
    width: usize,
    depth: usize,
    input_owner: usize,
    source: SourceBinding,
    proofs: [Option<DomainOrigin>; ROWS],
    levels: [Level; MAX_DEPTH],
    work: WorkBudget,
    logical: LogicalCost,
}
impl Plan {
    /// Public-geometry estimate, also used before any source import. No proof or
    /// source is manufactured to obtain this work-only lower bound.
    pub fn work_for_geometry(rows: usize, vocabulary: usize) -> Result<WorkBudget, Error> {
        if rows == 0 || rows > ROWS || !(2..=MAX_VOCABULARY).contains(&vocabulary) {
            return Err(Error::Bounds);
        }
        let depth = (usize::BITS - (vocabulary - 1).leading_zeros()) as usize;
        let mut work = WorkBudget::conversion_bounded(rows, depth as u8)?;
        for start in (0..rows * depth).step_by(LANES) {
            let n = LANES.min(rows * depth - start);
            work = work
                .checked_add(WorkBudget::local())?
                .checked_add(WorkBudget::from_boolean(n, 1)?)?;
        }
        for d in 1..depth {
            let parents = vocabulary.div_ceil(1 << (depth - d));
            let branches = vocabulary.div_ceil(1 << (depth - d - 1)) - parents;
            for start in (0..rows * branches).step_by(LANES) {
                work = work.checked_add(WorkBudget::product(
                    LANES.min(rows * branches - start),
                    false,
                )?)?;
            }
        }
        Ok(work)
    }
    /// PRECONDITION: proofs are issued by the validated model compiler for the
    /// exact gathered input and checked source. Never infer this from components.
    /// Loose intervals, unknown origins, V=1 and unsupported shapes are denied.
    pub fn new(
        rows: usize,
        vocabulary: usize,
        width: usize,
        input_owner: usize,
        domain: Interval,
        proofs: &[DomainOrigin],
        source: SourceBinding,
    ) -> Result<Self, Error> {
        if rows == 0
            || rows > ROWS
            || !(2..=MAX_VOCABULARY).contains(&vocabulary)
            || width == 0
            || width > 8192
            || rows * width > super::lookup::ACCUMULATOR_WORDS
            || domain.lower() != 0
            || domain.upper() != vocabulary as i64 - 1
            || proofs.len() != rows
            || input_owner == source.owner
            || source.source_commitment == [0; 32]
            || source.file_digest == [0; 32]
            || proofs.iter().any(|p| p.parts().1 == [0; 32])
            || (rows as u64) * vocabulary as u64 * width as u64 > 1 << 32
        {
            return Err(Error::Bounds);
        }
        let depth = (usize::BITS - (vocabulary - 1).leading_zeros()) as usize;
        let mut plan = Self {
            rows,
            vocabulary,
            width,
            depth,
            input_owner,
            source,
            proofs: [None; ROWS],
            levels: [Level::default(); MAX_DEPTH],
            work: Self::work_for_geometry(rows, vocabulary)?,
            logical: LogicalCost {
                secret_products: 0,
                sibling_component_subtractions: 0,
                forced_component_copies: 0,
                selector_component_writes: 0,
                public_ring64_macs: 2 * rows as u64 * vocabulary as u64 * width as u64,
                table_bytes: 8 * vocabulary as u64 * width as u64,
                a2b_frames: 3 * (depth.ilog2() as u64 + u64::from(!depth.is_power_of_two()) + 2),
                b2a_frames: 0,
                product_frames: 0,
                quiescence_barriers: 1 + vocabulary.div_ceil(LANES) as u64,
            },
        };
        for (dst, src) in plan.proofs.iter_mut().zip(proofs) {
            *dst = Some(*src);
        }
        for _ in (0..rows * depth).step_by(LANES) {
            plan.logical.b2a_frames += 4;
            plan.logical.quiescence_barriers += 1;
        }
        for d in 0..depth {
            let parents = vocabulary.div_ceil(1 << (depth - d));
            let children = vocabulary.div_ceil(1 << (depth - d - 1));
            let branches = children - parents;
            plan.levels[d] = Level {
                parents: parents as u32,
                branches: branches as u32,
            };
            plan.logical.selector_component_writes += 2 * rows as u64 * children as u64;
            plan.logical.forced_component_copies += 2 * rows as u64 * (parents - branches) as u64;
            if d == 0 {
                continue;
            } // public-one root: b and 1-b, no product/mask.
            plan.logical.secret_products += (rows * branches) as u64;
            for _ in (0..rows * branches).step_by(LANES) {
                plan.logical.product_frames += 3;
                plan.logical.quiescence_barriers += 1;
            }
        }
        plan.logical.sibling_component_subtractions =
            2 * (plan.logical.secret_products + rows as u64);
        plan.work.limits()?;
        if plan.peak_extra_bytes()? > MAX_PROGRAM_BYTES {
            return Err(Error::Bounds);
        }
        Ok(plan)
    }
    pub fn digest(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        for profile in [
            PROFILE,
            conversion::PROFILE,
            b2a::PROFILE,
            multiply::RAW_PRODUCT_PROFILE,
            "low-bits.nominal-xor-domain.compact-wire.v1",
            multiply::arithmetic::BOOLEAN_BIT_SLICE_PROFILE,
            "siblings.local-ring64.no-native-affine.v1",
        ] {
            h.update((profile.len() as u64).to_le_bytes());
            h.update(profile.as_bytes());
        }
        for n in [
            self.rows,
            self.vocabulary,
            self.width,
            self.depth,
            self.input_owner,
            self.source.owner,
            LANES,
            ROWS,
        ] {
            h.update((n as u64).to_le_bytes());
        }
        h.update(0_i64.to_le_bytes());
        h.update((self.vocabulary as i64 - 1).to_le_bytes());
        h.update(self.source.source_commitment);
        h.update(self.source.file_digest);
        for proof in self.proofs.iter().flatten() {
            let (kind, witness) = proof.parts();
            h.update([kind]);
            h.update(witness);
        }
        for l in &self.levels[..self.depth] {
            h.update(l.parents.to_le_bytes());
            h.update(l.branches.to_le_bytes());
        }
        let c = self.work.cursors();
        for n in [
            self.work.operations(),
            c.public_blocks,
            c.pair_blocks,
            c.private_blocks,
            c.aes_work,
        ] {
            h.update(n.to_le_bytes());
        }
        h.finalize().into()
    }
    pub fn work_budget(&self) -> WorkBudget {
        self.work
    }
    pub fn geometry(&self) -> (usize, usize, usize) {
        (self.rows, self.vocabulary, self.width)
    }
    pub fn validate_owners(
        &self,
        input: usize,
        weight: usize,
        source: [u8; 32],
    ) -> Result<(), Error> {
        if input != self.input_owner
            || weight != self.source.owner
            || source != self.source.source_commitment
        {
            return Err(Error::Binding);
        }
        Ok(())
    }
    pub fn logical_cost(&self) -> LogicalCost {
        self.logical
    }
    pub fn metadata_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + 2 * std::mem::size_of::<usize>()
    }
    fn children(&self, d: usize) -> usize {
        self.levels[d].parents as usize + self.levels[d].branches as usize
    }
    pub fn peak_selector_bytes(&self) -> usize {
        (1..self.depth)
            .map(|d| 16 * self.rows * (self.levels[d].parents as usize + self.children(d)))
            .max()
            .unwrap_or(16 * self.rows * self.vocabulary)
    }
    /// Conservative complete overlap. Supplied tokens are INCLUDED. Primitive
    /// input/output/masks stay under their separate original 256 KiB ledger.
    /// 64 KiB transport allowance covers per-step encoded copies, not a backlog.
    pub fn peak_extra_bytes(&self) -> Result<usize, Error> {
        self.peak_selector_bytes()
            .checked_add(self.metadata_bytes())
            .and_then(|n| n.checked_add(std::mem::size_of::<TrieLookup>()))
            .and_then(|n| n.checked_add(2 * self.rows * std::mem::size_of::<Secret>()))
            .and_then(|n| n.checked_add(16 * self.rows * (1 + self.depth + self.width + LANES)))
            .and_then(|n| {
                n.checked_add(
                    32 * LANES + (64 << 10) + puma::MAX_ADMITTED_BYTES + (64 << 10) + (64 << 10),
                )
            }) // transport copies + allocator/slab allowance
            .ok_or(Error::Bounds)
    }
    fn check_file(&self, source_owner: usize, file: &PublicFile) -> Result<(), Error> {
        if source_owner != self.source.owner
            || file.digest() != self.source.file_digest
            || file.size() != self.logical.table_bytes
        {
            return Err(Error::Binding);
        }
        Ok(())
    }
}

/// Injection covers this owner's fallible allocations, not the core allocator.
/// Native allocation failures remain under native payload-ledger tests.
#[derive(Default)]
struct Allocations {
    #[cfg(test)]
    remaining: Option<usize>,
    #[cfg(test)]
    calls: usize,
}
impl Allocations {
    fn checkpoint(&mut self) -> Result<(), Error> {
        #[cfg(test)]
        {
            self.calls += 1;
            if let Some(left) = &mut self.remaining {
                if *left == 0 {
                    return Err(Error::Allocation);
                }
                *left -= 1;
            }
        }
        Ok(())
    }
    fn secret(&mut self, n: usize) -> Result<Secret, Error> {
        self.checkpoint()?;
        let value = Secret::new(n)?;
        if value.first.capacity() != n || value.second.capacity() != n {
            return Err(Error::Allocation);
        }
        Ok(value)
    }
    fn rows(&mut self, rows: usize, n: usize) -> Result<Vec<Secret>, Error> {
        self.checkpoint()?;
        let mut out = Vec::new();
        out.try_reserve_exact(rows).map_err(|_| Error::Allocation)?;
        if out.capacity() != rows {
            return Err(Error::Allocation);
        }
        for _ in 0..rows {
            out.push(self.secret(n)?);
        }
        Ok(out)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    ReadyA2b,
    A2b,
    ReadyBits,
    B2a { start: usize, n: usize },
    ReadyProduct,
    Product { n: usize },
    ReadyStream,
    Stream,
    Done,
    Retired,
}

pub(in crate::replicated_decoder) struct TrieLookup {
    plan: Arc<Plan>,
    runtime: Option<PartyRuntime>,
    tokens: Option<Secret>,
    packed: Option<boolean::Share>,
    bits: Option<Secret>,
    parents: Vec<Secret>,
    children: Vec<Secret>,
    output: Option<Secret>,
    inbox: [Option<Message>; 3],
    step: Step,
    bit_offset: usize,
    level: usize,
    offset: usize,
    stream_offset: usize,
    pid: u32,
    alloc: Allocations,
    #[cfg(test)]
    peak_selectors: usize,
}
impl TrieLookup {
    /// Caller supplies the authenticated worker context and the dedicated input
    /// owner ID. Caller retains the checked PublicSource; we borrow its file.
    pub fn new(
        runtime: PartyRuntime,
        tokens: Secret,
        plan: Arc<Plan>,
        context: puma::Context,
        input_owner: usize,
        source_owner: usize,
        file: &PublicFile,
    ) -> Result<Self, Error> {
        Self::construct(
            runtime,
            tokens,
            plan,
            context,
            input_owner,
            source_owner,
            file,
            Allocations::default(),
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn construct(
        runtime: PartyRuntime,
        tokens: Secret,
        plan: Arc<Plan>,
        context: puma::Context,
        input_owner: usize,
        source_owner: usize,
        file: &PublicFile,
        alloc: Allocations,
    ) -> Result<Self, Error> {
        let mut owner = Self {
            plan,
            runtime: Some(runtime),
            tokens: Some(tokens),
            packed: None,
            bits: None,
            parents: Vec::new(),
            children: Vec::new(),
            output: None,
            inbox: [None, None, None],
            step: Step::ReadyA2b,
            bit_offset: 0,
            level: 0,
            offset: 0,
            stream_offset: 0,
            pid: std::process::id(),
            alloc,
            #[cfg(test)]
            peak_selectors: 0,
        };
        let result = (|| {
            let r = owner.runtime.as_ref().ok_or(Error::Retired)?;
            if !r.matches_context(context)
                || r.status() != Status::Ready
                || input_owner != owner.plan.input_owner
            {
                return Err(Error::Binding);
            }
            owner.plan.check_file(source_owner, file)?;
            let t = owner.tokens.as_ref().ok_or(Error::Retired)?;
            if t.first.len() != owner.plan.rows || t.second.len() != owner.plan.rows {
                return Err(Error::Shape);
            }
            if t.first.capacity() != owner.plan.rows || t.second.capacity() != owner.plan.rows {
                return Err(Error::Bounds);
            }
            if owner.plan.peak_extra_bytes()? > MAX_PROGRAM_BYTES {
                return Err(Error::Bounds);
            }
            owner.bits = Some(owner.alloc.secret(owner.plan.rows * owner.plan.depth)?);
            owner.output = Some(owner.alloc.secret(owner.plan.rows * owner.plan.width)?);
            Ok(())
        })();
        owner.checked(result)?;
        Ok(owner)
    }
    pub fn complete(&self) -> bool {
        self.step == Step::Done
    }
    pub fn retired(&self) -> bool {
        self.step == Step::Retired
    }
    pub fn cancel(&mut self) {
        if let Some(mut r) = self.runtime.take() {
            r.cancel();
        }
        self.tokens = None;
        self.packed = None;
        self.bits = None;
        self.parents = Vec::new();
        self.children = Vec::new();
        self.output = None;
        self.inbox = [None, None, None];
        self.step = Step::Retired;
    }
    fn live(&self) -> Result<(), Error> {
        if self.retired() || self.pid != std::process::id() {
            Err(Error::Retired)
        } else {
            Ok(())
        }
    }
    fn checked<T>(&mut self, result: Result<T, Error>) -> Result<T, Error> {
        if result.is_err() {
            self.cancel();
        }
        result
    }
    fn frames(&mut self, messages: [Option<Message>; 2]) -> Result<Vec<(Party, PeerBytes)>, Error> {
        self.alloc.checkpoint()?;
        let mut out = Vec::new();
        out.try_reserve_exact(2).map_err(|_| Error::Allocation)?;
        if out.capacity() != 2 {
            return Err(Error::Allocation);
        }
        for message in messages.into_iter().flatten() {
            out.push((
                message.header().to,
                PeerBytes::Arithmetic(message.encode()?),
            ));
        }
        Ok(out)
    }
    pub fn start_step(&mut self) -> Result<Vec<(Party, PeerBytes)>, Error> {
        let result = self.start_inner();
        self.checked(result)
    }
    fn start_inner(&mut self) -> Result<Vec<(Party, PeerBytes)>, Error> {
        self.live()?;
        match self.step {
            Step::ReadyA2b => {
                let r = self.runtime.as_mut().ok_or(Error::Retired)?;
                let plan = r.plan_conversion(
                    self.plan.rows,
                    conversion::Kind::LowBits(self.plan.depth as u8),
                )?;
                let t = self.tokens.as_ref().ok_or(Error::Retired)?;
                let input =
                    plan.share_input(r.party(), &t.first, &t.second, puma::Limits::default())?;
                let message = r.begin_conversion(input)?;
                self.step = Step::A2b;
                self.frames([Some(message), None])
            }
            Step::ReadyBits => {
                let n = LANES.min(self.plan.rows * self.plan.depth - self.bit_offset);
                let r = self.runtime.as_mut().ok_or(Error::Retired)?;
                // REQUIRED CORE HOOK. Borrowed packed owner remains ledger-charged.
                let bits = r.boolean_bit_slice(
                    self.packed.as_ref().ok_or(Error::Retired)?,
                    self.plan.depth as u8,
                    self.bit_offset,
                    n,
                )?;
                let plan = r.plan_boolean_to_arithmetic(&bits)?;
                if plan.bits() != 1
                    || plan.lanes() != n
                    || plan.pair_blocks() != 3 * (n as u64).div_ceil(2)
                {
                    return Err(Error::Binding);
                }
                let messages = r.begin_boolean_to_arithmetic(bits)?;
                self.step = Step::B2a {
                    start: self.bit_offset,
                    n,
                };
                self.frames(messages)
            }
            Step::ReadyProduct => {
                if self.children.is_empty() {
                    let level = self.plan.levels[self.level];
                    self.children = self
                        .alloc
                        .rows(self.plan.rows, self.plan.children(self.level))?;
                    // Only the last parent can be unary for contiguous [0,V).
                    // Its bit is forced zero under the authenticated domain proof.
                    if level.parents > level.branches {
                        let last = level.parents as usize - 1;
                        for row in 0..self.plan.rows {
                            self.children[row].first[2 * last] = self.parents[row].first[last];
                            self.children[row].second[2 * last] = self.parents[row].second[last];
                        }
                    }
                    #[cfg(test)]
                    {
                        self.peak_selectors = self.peak_selectors.max(
                            self.parents
                                .iter()
                                .chain(&self.children)
                                .map(|s| 8 * (s.first.capacity() + s.second.capacity()))
                                .sum(),
                        );
                    }
                }
                let branches = self.plan.levels[self.level].branches as usize;
                let n = LANES.min(self.plan.rows * branches - self.offset);
                let mut left = self.alloc.secret(n)?;
                let mut right = self.alloc.secret(n)?;
                let bits = self.bits.as_ref().ok_or(Error::Retired)?;
                for i in 0..n {
                    let index = self.offset + i;
                    let row = index / branches;
                    let parent = index % branches;
                    left.first[i] = self.parents[row].first[parent];
                    left.second[i] = self.parents[row].second[parent];
                    let bit = row * self.plan.depth + self.plan.depth - 1 - self.level;
                    right.first[i] = bits.first[bit];
                    right.second[i] = bits.second[bit];
                }
                let r = self.runtime.as_mut().ok_or(Error::Retired)?;
                let bound = multiply::InputBounds {
                    left: Interval::new(0, 1)?,
                    right: Interval::new(0, 1)?,
                };
                let plan = r.plan_ring_product(&[bound; LANES][..n])?;
                let a = plan.share_operand(
                    Operand::Left,
                    r.party(),
                    &left.first,
                    &left.second,
                    puma::Limits::default(),
                )?;
                let b = plan.share_operand(
                    Operand::Right,
                    r.party(),
                    &right.first,
                    &right.second,
                    puma::Limits::default(),
                )?;
                drop(left);
                drop(right);
                let message = r.begin(a, b)?;
                self.step = Step::Product { n };
                self.frames([Some(message), None])
            }
            Step::ReadyStream => {
                self.step = Step::Stream;
                self.frames([None, None])
            }
            _ => Err(Error::Phase),
        }
    }
    pub fn receive_encoded(
        &mut self,
        raw: &[u8],
        sender: Party,
    ) -> Result<Vec<(Party, PeerBytes)>, Error> {
        let result = self.receive_inner(raw, sender);
        self.checked(result)
    }
    fn receive_inner(
        &mut self,
        raw: &[u8],
        sender: Party,
    ) -> Result<Vec<(Party, PeerBytes)>, Error> {
        self.live()?;
        let r = self.runtime.as_mut().ok_or(Error::Retired)?;
        let message = Message::decode(raw, sender, r.party())?;
        if message.words().len() > 2 * LANES {
            return Err(Error::Bounds);
        }
        let replies = match self.step {
            Step::A2b => [r.advance_conversion(message)?, None],
            Step::Product { .. } if r.status() == Status::AwaitingRotate => r.rotate(message)?,
            Step::B2a { .. } => {
                let slot = match (r.status(), message.header().phase) {
                    (Status::AwaitingOpen, MessagePhase::BooleanOtMask) => 0,
                    (Status::AwaitingOpen, MessagePhase::BooleanOtZero) => 1,
                    (Status::AwaitingOpen, MessagePhase::BooleanOtOne) => 2,
                    (Status::AwaitingReshare, MessagePhase::BooleanOtReshare) => 0,
                    _ => return Err(Error::Phase),
                };
                if self.inbox[slot].is_some() {
                    return Err(Error::Phase);
                }
                self.inbox[slot] = Some(message);
                let ready = if r.status() == Status::AwaitingReshare {
                    self.inbox[0].is_some()
                } else {
                    self.inbox.iter().all(Option::is_some)
                };
                if ready {
                    let inbox = std::mem::take(&mut self.inbox);
                    [r.receive_boolean_to_arithmetic(inbox)?, None]
                } else {
                    [None, None]
                }
            }
            _ => return Err(Error::Phase),
        };
        self.frames(replies)
    }
    /// Call only at the ordinary globally drained quiescence barrier. During
    /// streaming, one public vocabulary tile is read per barrier (bounded cancel).
    pub fn finish_step(&mut self, source_owner: usize, file: &PublicFile) -> Result<(), Error> {
        let result = self.finish_inner(source_owner, file);
        self.checked(result)
    }
    fn finish_inner(&mut self, source_owner: usize, file: &PublicFile) -> Result<(), Error> {
        self.live()?;
        self.plan.check_file(source_owner, file)?;
        let r = self.runtime.as_mut().ok_or(Error::Retired)?;
        match self.step {
            Step::A2b => {
                self.packed = Some(r.take_boolean_output()?);
                self.tokens = None;
                self.step = Step::ReadyBits;
            }
            Step::B2a { start, n } => {
                let output = r.take_output()?;
                let bits = self.bits.as_mut().ok_or(Error::Retired)?;
                bits.first[start..start + n].copy_from_slice(output.first());
                bits.second[start..start + n].copy_from_slice(output.second());
                self.bit_offset += n;
                if self.bit_offset != self.plan.rows * self.plan.depth {
                    self.step = Step::ReadyBits;
                } else {
                    self.packed = None;
                    self.parents = self.alloc.rows(self.plan.rows, 2)?;
                    for row in 0..self.plan.rows {
                        let bit = row * self.plan.depth + self.plan.depth - 1;
                        let b0 = bits.first[bit];
                        let b1 = bits.second[bit];
                        self.parents[row].first[0] =
                            u64::from(r.party() == Party::P0).wrapping_sub(b0);
                        self.parents[row].second[0] =
                            u64::from(r.party().next() == Party::P0).wrapping_sub(b1);
                        self.parents[row].first[1] = b0;
                        self.parents[row].second[1] = b1;
                    }
                    self.level = 1;
                    #[cfg(test)]
                    {
                        self.peak_selectors = 16 * self.plan.rows * 2;
                    }
                    if self.level == self.plan.depth {
                        self.bits = None;
                        self.step = Step::ReadyStream;
                    } else {
                        self.step = Step::ReadyProduct;
                    }
                }
            }
            Step::Product { n } => {
                let output = r.take_output()?;
                let branches = self.plan.levels[self.level].branches as usize;
                for i in 0..n {
                    let index = self.offset + i;
                    let row = index / branches;
                    let p = index % branches;
                    let a = output.first()[i];
                    let b = output.second()[i];
                    // Sealed invariant: p and b are Boolean arithmetic shares;
                    // these children are p*(1-b) and p*b in the COMPLETE ring.
                    self.children[row].first[2 * p] = self.parents[row].first[p].wrapping_sub(a);
                    self.children[row].second[2 * p] = self.parents[row].second[p].wrapping_sub(b);
                    self.children[row].first[2 * p + 1] = a;
                    self.children[row].second[2 * p + 1] = b;
                }
                self.offset += n;
                if self.offset == self.plan.rows * branches {
                    self.parents = std::mem::take(&mut self.children);
                    self.offset = 0;
                    self.level += 1;
                }
                if self.level == self.plan.depth {
                    self.bits = None;
                    self.step = Step::ReadyStream;
                } else {
                    self.step = Step::ReadyProduct;
                }
            }
            Step::Stream => {
                let n = LANES.min(self.plan.vocabulary - self.stream_offset);
                let mut selectors = self.alloc.secret(self.plan.rows * n)?;
                for row in 0..self.plan.rows {
                    selectors.first[row * n..(row + 1) * n].copy_from_slice(
                        &self.parents[row].first[self.stream_offset..self.stream_offset + n],
                    );
                    selectors.second[row * n..(row + 1) * n].copy_from_slice(
                        &self.parents[row].second[self.stream_offset..self.stream_offset + n],
                    );
                }
                public_source::accumulate_rows(
                    file,
                    &selectors,
                    self.output.as_mut().ok_or(Error::Retired)?,
                    self.plan.rows,
                    self.plan.vocabulary,
                    self.plan.width,
                    self.stream_offset,
                    n,
                )?;
                self.stream_offset += n;
                if self.stream_offset == self.plan.vocabulary {
                    self.parents = Vec::new();
                    self.step = Step::Done;
                } else {
                    self.step = Step::ReadyStream;
                }
            }
            _ => return Err(Error::Phase),
        }
        Ok(())
    }
    pub fn finish(mut self) -> Result<(PartyRuntime, Secret), Error> {
        let result = (|| {
            self.live()?;
            if !self.complete() {
                return Err(Error::Phase);
            }
            Ok((
                self.runtime.take().ok_or(Error::Retired)?,
                self.output.take().ok_or(Error::Retired)?,
            ))
        })();
        self.checked(result)
    }
}
impl Drop for TrieLookup {
    fn drop(&mut self) {
        self.cancel();
    }
}

#[cfg(test)]
#[path = "trie_tests.rs"]
mod tests;
