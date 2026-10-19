//! Fixed full-vocabulary stable argmax, with no per-comparison graph nodes.
//! Scores compact in place; only a replicated original-index array is added.
#![allow(dead_code)] // Diagnostic accessors are also used by embedding clients.
use super::PeerBytes;
use crate::replicated_decoder::*;
use multiply::{Operand, PartyRuntime, Status};
use puma::{Message, Operation, Party, ReplicatedShare};

pub(in crate::replicated_decoder) const COMPARISON_TILE: usize = 64;
const MAX_VOCABULARY: usize = 262144;
const MAX_EXTRA_BYTES: usize = 8 << 20;
const TILE_SCRATCH_BYTES: usize = 16 << 10;

pub(in crate::replicated_decoder) struct Plan {
    vocabulary: usize,
    difference: Interval,
    programs: Vec<tensor::Program>,
    work: WorkBudget,
    tiles: u64,
}

impl Plan {
    pub fn new(vocabulary: usize, scores: Interval) -> Result<Self, Error> {
        if vocabulary == 0 || vocabulary > MAX_VOCABULARY {
            return Err(Error::Bounds);
        }
        let span = i128::from(scores.upper()) - i128::from(scores.lower());
        // Signed subtraction, sign extraction and selected products all retain
        // quarter-gap domains. Original scores need not be centered or narrowed.
        if span >= i128::from(puma::INPUT_MAX_EXCLUSIVE) {
            return Err(Error::Bounds);
        }
        let span = i64::try_from(span).map_err(|_| Error::Bounds)?;
        let difference = Interval::new(-span, span)?;
        let levels = (usize::BITS - (vocabulary - 1).leading_zeros()) as usize;
        // Preflight the conservative metadata/index/scratch ceiling before any
        // native owner allocation. At most one full width plus one tail/level.
        let preflight = (levels + 1)
            .checked_mul(std::mem::size_of::<tensor::Program>())
            .and_then(|n| {
                n.checked_add(std::mem::size_of::<Self>() + 2 * std::mem::size_of::<usize>())
            })
            .and_then(|n| n.checked_add(vocabulary.checked_mul(16)?))
            .and_then(|n| n.checked_add(TILE_SCRATCH_BYTES + std::mem::size_of::<StableArgmax>()))
            .ok_or(Error::Bounds)?;
        if preflight > MAX_EXTRA_BYTES {
            return Err(Error::Bounds);
        }
        let mut programs: Vec<tensor::Program> = Vec::new();
        programs
            .try_reserve_exact(levels + 1)
            .map_err(|_| Error::Allocation)?;
        let mut work = WorkBudget::default();
        let mut tiles = 0u64;
        let mut width = vocabulary;
        while width > 1 {
            let pairs = width / 2;
            for (n, count) in [
                (COMPARISON_TILE, pairs / COMPARISON_TILE),
                (pairs % COMPARISON_TILE, 1),
            ] {
                if n == 0 || count == 0 {
                    continue;
                }
                if !programs.iter().any(|p| p.lanes() == n) {
                    programs.push(tensor::Program::negative(n, difference)?);
                }
                let program = programs
                    .iter()
                    .find(|p| p.lanes() == n)
                    .ok_or(Error::Binding)?;
                work = work.checked_add(
                    program
                        .work_budget()
                        .checked_add(WorkBudget::product(2 * n, false)?)?
                        .checked_mul(count as u64)?,
                )?;
                tiles = tiles.checked_add(count as u64).ok_or(Error::Bounds)?;
            }
            width = width.div_ceil(2);
        }
        let plan = Self {
            vocabulary,
            difference,
            programs,
            work,
            tiles,
        };
        if plan.peak_extra_bytes()? > MAX_EXTRA_BYTES {
            return Err(Error::Bounds);
        }
        Ok(plan)
    }

    pub fn vocabulary(&self) -> usize {
        self.vocabulary
    }
    pub fn digest(&self) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(b"pllm.replicated-decoder.stable-argmax.comparisons64.ring64.v1");
        hash.update((self.vocabulary as u64).to_le_bytes());
        hash.update(self.difference.lower().to_le_bytes());
        hash.update(self.difference.upper().to_le_bytes());
        for program in &self.programs {
            hash.update(program.digest());
        }
        hash.finalize().into()
    }
    pub fn work_budget(&self) -> WorkBudget {
        self.work
    }
    /// (comparisons, exact selected products, primitive tile pairs).
    pub fn logical_cost(&self) -> (u64, u64, u64) {
        let comparisons = (self.vocabulary - 1) as u64;
        (comparisons, 2 * comparisons, self.tiles)
    }
    /// Arc payload/header plus fallibly reserved inline tensor-program storage.
    /// Negative programs have no heap-backed public affine coefficient arrays.
    pub fn metadata_bytes(&self) -> Result<usize, Error> {
        self.programs
            .capacity()
            .checked_mul(std::mem::size_of::<tensor::Program>())
            .and_then(|n| {
                n.checked_add(std::mem::size_of::<Self>() + 2 * std::mem::size_of::<usize>())
            })
            .ok_or(Error::Bounds)
    }
    /// Excludes caller's full logits and original primitive heap workspace.
    /// Includes shared plan metadata once, fixed owner, indices and tile copies.
    pub fn peak_extra_bytes(&self) -> Result<usize, Error> {
        self.vocabulary
            .checked_mul(16)
            .and_then(|n| n.checked_add(TILE_SCRATCH_BYTES))
            .and_then(|n| n.checked_add(std::mem::size_of::<StableArgmax>()))
            .and_then(|n| n.checked_add(self.metadata_bytes().ok()?))
            .ok_or(Error::Bounds)
    }
    pub fn predicate_workspace_bytes(&self) -> usize {
        self.programs
            .iter()
            .map(|p| p.admitted_bytes())
            .max()
            .unwrap_or(0)
    }
    fn program(&self, pairs: usize) -> Result<&tensor::Program, Error> {
        self.programs
            .iter()
            .find(|p| p.lanes() == pairs)
            .ok_or(Error::Binding)
    }
}

#[allow(clippy::large_enum_variant)]
enum Active {
    Ready(PartyRuntime),
    Negative(tensor::Executor),
    Product(PartyRuntime),
    Retired,
}

pub(in crate::replicated_decoder) struct StableArgmax {
    plan: Arc<Plan>,
    scores: Option<Secret>,
    indices: Option<Secret>,
    selector: Option<ReplicatedShare>,
    active: Active,
    width: usize,
    offset: usize,
    pid: u32,
}

impl StableArgmax {
    /// Takes ownership of the caller's complete logits, avoiding a second full
    /// score allocation. Main must not copy logits before calling this API.
    pub fn new(runtime: PartyRuntime, scores: Secret, plan: Arc<Plan>) -> Result<Self, Error> {
        if runtime.status() != Status::Ready
            || scores.first.len() != plan.vocabulary
            || scores.second.len() != plan.vocabulary
        {
            return Err(Error::Shape);
        }
        // Plan's checked peak/work arithmetic precedes the index allocation.
        if plan.peak_extra_bytes()? > MAX_EXTRA_BYTES {
            return Err(Error::Bounds);
        }
        let mut indices = Secret::new(plan.vocabulary)?;
        let party = runtime.party();
        for i in 0..plan.vocabulary {
            if party == Party::P0 {
                indices.first[i] = i as u64;
            }
            if party.next() == Party::P0 {
                indices.second[i] = i as u64;
            }
        }
        let width = plan.vocabulary;
        Ok(Self {
            plan,
            scores: Some(scores),
            indices: Some(indices),
            selector: None,
            active: Active::Ready(runtime),
            width,
            offset: 0,
            pid: std::process::id(),
        })
    }
    pub fn complete(&self) -> bool {
        self.width == 1 && matches!(self.active, Active::Ready(_)) && self.selector.is_none()
    }
    pub fn retired(&self) -> bool {
        matches!(self.active, Active::Retired)
    }
    #[cfg(test)]
    pub(super) fn test_phase(&self) -> u8 {
        match self.active {
            Active::Ready(_) => 0,
            Active::Negative(_) => 1,
            Active::Product(_) => 2,
            Active::Retired => 3,
        }
    }
    pub fn cancel(&mut self) {
        self.active = Active::Retired;
        self.selector = None;
        self.scores = None;
        self.indices = None;
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
    fn pairs(&self) -> usize {
        COMPARISON_TILE.min(self.width / 2 - self.offset)
    }

    pub fn start_step(&mut self) -> Result<Vec<(Party, PeerBytes)>, Error> {
        let result = (|| {
            self.live()?;
            if self.complete() {
                return Err(Error::Phase);
            }
            let active = std::mem::replace(&mut self.active, Active::Retired);
            match active {
                Active::Negative(mut executor) => {
                    let frames = tensor_frames(executor.start_step()?)?;
                    self.active = Active::Negative(executor);
                    Ok(frames)
                }
                Active::Ready(mut runtime) => {
                    let n = self.pairs();
                    let scores = self.scores.as_ref().ok_or(Error::Retired)?;
                    if let Some(selector) = self.selector.take() {
                        let indices = self.indices.as_ref().ok_or(Error::Retired)?;
                        let mut flags = Secret::new(2 * n)?;
                        let mut differences = Secret::new(2 * n)?;
                        for i in 0..n {
                            let left = 2 * (self.offset + i);
                            let right = left + 1;
                            for (lane, values) in [(i, scores), (n + i, indices)] {
                                flags.first[lane] = selector.first()[i];
                                flags.second[lane] = selector.second()[i];
                                differences.first[lane] =
                                    values.first[right].wrapping_sub(values.first[left]);
                                differences.second[lane] =
                                    values.second[right].wrapping_sub(values.second[left]);
                            }
                        }
                        // End tensor output's ledger charge before product copies.
                        drop(selector);
                        let score_bounds = multiply::InputBounds {
                            left: Interval::new(0, 1)?,
                            right: self.plan.difference,
                        };
                        let radius = (self.plan.vocabulary - 1) as i64;
                        let index_bounds = multiply::InputBounds {
                            left: Interval::new(0, 1)?,
                            right: Interval::new(-radius, radius)?,
                        };
                        let mut bounds = [score_bounds; 2 * COMPARISON_TILE];
                        bounds[n..2 * n].fill(index_bounds);
                        let plan = runtime.plan_ring_product(&bounds[..2 * n])?;
                        let party = runtime.party();
                        let left = plan.share_operand(
                            Operand::Left,
                            party,
                            &flags.first,
                            &flags.second,
                            puma::Limits::default(),
                        )?;
                        let right = plan.share_operand(
                            Operand::Right,
                            party,
                            &differences.first,
                            &differences.second,
                            puma::Limits::default(),
                        )?;
                        let frames = arithmetic_frames([Some(runtime.begin(left, right)?), None])?;
                        self.active = Active::Product(runtime);
                        Ok(frames)
                    } else {
                        let mut difference = Secret::new(n)?;
                        for i in 0..n {
                            let left = 2 * (self.offset + i);
                            difference.first[i] =
                                scores.first[left].wrapping_sub(scores.first[left + 1]);
                            difference.second[i] =
                                scores.second[left].wrapping_sub(scores.second[left + 1]);
                        }
                        let op = Operation::new(
                            runtime.context(),
                            0,
                            Party::P0,
                            n,
                            1,
                            puma::Limits::default(),
                        )?;
                        let input = ReplicatedShare::from_local(
                            op,
                            runtime.party(),
                            &difference.first,
                            &difference.second,
                            puma::Limits::default(),
                        )?;
                        let mut executor =
                            tensor::Executor::new(runtime, self.plan.program(n)?.clone(), input)?;
                        let frames = tensor_frames(executor.start_step()?)?;
                        self.active = Active::Negative(executor);
                        Ok(frames)
                    }
                }
                _ => Err(Error::Phase),
            }
        })();
        self.checked(result)
    }

    pub fn receive_encoded(
        &mut self,
        raw: &[u8],
        sender: Party,
    ) -> Result<Vec<(Party, PeerBytes)>, Error> {
        let result = (|| {
            self.live()?;
            match &mut self.active {
                Active::Negative(executor) => tensor_frames(executor.receive_encoded(raw, sender)?),
                Active::Product(runtime) if runtime.status() == Status::AwaitingRotate => {
                    let message = Message::decode(raw, sender, runtime.party())?;
                    arithmetic_frames(runtime.rotate(message)?)
                }
                _ => Err(Error::Phase),
            }
        })();
        self.checked(result)
    }

    pub fn finish_step(&mut self) -> Result<(), Error> {
        let result = (|| {
            self.live()?;
            let active = std::mem::replace(&mut self.active, Active::Retired);
            self.active = match active {
                Active::Negative(mut executor) => {
                    executor.finish_step()?;
                    if executor.complete() {
                        let (runtime, selector) = executor.finish()?;
                        self.selector = Some(selector);
                        Active::Ready(runtime)
                    } else {
                        Active::Negative(executor)
                    }
                }
                Active::Product(mut runtime) => {
                    let product = runtime.take_output()?;
                    let n = self.pairs();
                    let scores = self.scores.as_mut().ok_or(Error::Retired)?;
                    let indices = self.indices.as_mut().ok_or(Error::Retired)?;
                    for i in 0..n {
                        let out = self.offset + i;
                        let left = 2 * out;
                        scores.first[out] = scores.first[left].wrapping_add(product.first()[i]);
                        scores.second[out] = scores.second[left].wrapping_add(product.second()[i]);
                        indices.first[out] =
                            indices.first[left].wrapping_add(product.first()[n + i]);
                        indices.second[out] =
                            indices.second[left].wrapping_add(product.second()[n + i]);
                    }
                    self.offset += n;
                    if self.offset == self.width / 2 {
                        if self.width % 2 != 0 {
                            let tail = self.width - 1;
                            scores.first[self.offset] = scores.first[tail];
                            scores.second[self.offset] = scores.second[tail];
                            indices.first[self.offset] = indices.first[tail];
                            indices.second[self.offset] = indices.second[tail];
                        }
                        self.width = self.width.div_ceil(2);
                        self.offset = 0;
                    }
                    Active::Ready(runtime)
                }
                _ => return Err(Error::Phase),
            };
            Ok(())
        })();
        self.checked(result)
    }

    pub fn finish(mut self) -> Result<(PartyRuntime, Secret), Error> {
        self.live()?;
        if !self.complete() {
            return Err(Error::Phase);
        }
        let mut result = Secret::new(1)?;
        let indices = self.indices.as_ref().ok_or(Error::Retired)?;
        result.first[0] = indices.first[0];
        result.second[0] = indices.second[0];
        let Active::Ready(runtime) = std::mem::replace(&mut self.active, Active::Retired) else {
            return Err(Error::Phase);
        };
        Ok((runtime, result))
    }
}
impl Drop for StableArgmax {
    fn drop(&mut self) {
        self.cancel();
    }
}

fn tensor_frames(frames: [Option<tensor::Frame>; 3]) -> Result<Vec<(Party, PeerBytes)>, Error> {
    let mut out = Vec::new();
    out.try_reserve_exact(3).map_err(|_| Error::Allocation)?;
    for frame in frames.into_iter().flatten() {
        out.push((frame.recipient(), PeerBytes::Tensor(frame.encode()?)));
    }
    Ok(out)
}
fn arithmetic_frames(frames: [Option<Message>; 2]) -> Result<Vec<(Party, PeerBytes)>, Error> {
    let mut out = Vec::new();
    out.try_reserve_exact(2).map_err(|_| Error::Allocation)?;
    for frame in frames.into_iter().flatten() {
        out.push((frame.header().to, PeerBytes::Arithmetic(frame.encode()?)));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use puma::{
        prss::{PublicContribution, SeedMessage},
        Context,
    };

    fn runtimes(work: WorkBudget) -> [PartyRuntime; 3] {
        let parties = [Party::P0, Party::P1, Party::P2];
        let context = Context::new([81; 32], [92; 32], 0).unwrap();
        let mut setup =
            parties.map(|p| multiply::SetupParty::new(context, p, work.limits().unwrap()).unwrap());
        let public = setup.each_mut().map(|s| s.contribution().unwrap().encode());
        for (i, s) in setup.iter_mut().enumerate() {
            for j in 0..3 {
                if i != j {
                    s.receive_public(PublicContribution::decode(&public[j], parties[j]).unwrap())
                        .unwrap();
                }
            }
        }
        let mut seeds = setup
            .each_mut()
            .map(|s| Some(s.share_seed().unwrap().encode().unwrap()));
        for (i, s) in setup.iter_mut().enumerate() {
            let j = (i + 1) % 3;
            s.receive_seed(
                SeedMessage::decode(seeds[j].take().unwrap().as_bytes(), parties[j], parties[i])
                    .unwrap(),
            )
            .unwrap();
        }
        setup.each_mut().map(|s| s.finish().unwrap())
    }

    fn owners(scores: &[i64], bounds: Interval) -> ([StableArgmax; 3], Arc<Plan>) {
        let plan = Arc::new(Plan::new(scores.len(), bounds).unwrap());
        let shares: [Vec<u64>; 3] = std::array::from_fn(|party| {
            scores
                .iter()
                .enumerate()
                .map(|(i, &score)| {
                    let a = 0xc751f00d12345678u64
                        .wrapping_add((i as u64).wrapping_mul(0xabcdef9988776655));
                    let b = 0x254277811928f93du64
                        .wrapping_add((i as u64).wrapping_mul(0xffffabcd76543211));
                    match party {
                        0 => a,
                        1 => b,
                        _ => (score as u64).wrapping_sub(a).wrapping_sub(b),
                    }
                })
                .collect()
        });
        let mut index = 0;
        let owners = runtimes(plan.work_budget()).map(|runtime| {
            let i = index;
            index += 1;
            let mut secret = Secret::new(scores.len()).unwrap();
            secret.first.copy_from_slice(&shares[i]);
            secret.second.copy_from_slice(&shares[(i + 1) % 3]);
            StableArgmax::new(runtime, secret, plan.clone()).unwrap()
        });
        (owners, plan)
    }

    fn step(owners: &mut [StableArgmax; 3]) {
        let parties = [Party::P0, Party::P1, Party::P2];
        let mut queue = Vec::new();
        for (i, owner) in owners.iter_mut().enumerate() {
            queue.extend(
                owner
                    .start_step()
                    .unwrap()
                    .into_iter()
                    .map(|(to, frame)| (parties[i], to, frame)),
            );
        }
        while !queue.is_empty() {
            for (sender, recipient, frame) in std::mem::take(&mut queue) {
                queue.extend(
                    owners[recipient as usize]
                        .receive_encoded(frame.as_bytes(), sender)
                        .unwrap()
                        .into_iter()
                        .map(|(to, frame)| (recipient, to, frame)),
                );
            }
        }
        for owner in owners {
            owner.finish_step().unwrap();
        }
    }

    fn test_stack(test: fn()) {
        std::thread::Builder::new()
            .stack_size(16 << 20)
            .spawn(test)
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn stable_selection_independent_signed_oracle_ties_tails_and_exact_work() {
        test_stack(oracle);
    }

    fn oracle() {
        let mut fixtures = vec![
            vec![i64::MIN],
            vec![i64::MAX, i64::MAX - 1, i64::MAX],
            vec![i64::MIN, i64::MIN + 1, i64::MIN + 2],
            vec![i64::MIN, i64::MIN + (1i64 << 62) - 1],
            vec![i64::MAX - ((1i64 << 62) - 1), i64::MAX],
            vec![-((1i64 << 61) - 1), (1i64 << 61) - 1, 0, (1i64 << 61) - 1],
        ];
        for n in [2, 3, 63, 64, 65, 127, 128, 129, 257, 513] {
            let mut values: Vec<_> = (0..n)
                .map(|i| -(((i * 7919 + 17) % 100003) as i64))
                .collect();
            values[n - 1] = 500;
            fixtures.push(values.clone());
            values[0] = 500;
            fixtures.push(values);
            fixtures.push(vec![-7; n]);
        }
        for values in fixtures {
            let bounds =
                Interval::new(*values.iter().min().unwrap(), *values.iter().max().unwrap())
                    .unwrap();
            // Independent full-array signed oracle, stable first occurrence.
            let mut expected = 0;
            for i in 1..values.len() {
                if values[i] > values[expected] {
                    expected = i;
                }
            }
            let (mut owners, plan) = owners(&values, bounds);
            let usages = owners.each_ref().map(|owner| match &owner.active {
                Active::Ready(runtime) => runtime.payload_usage(),
                _ => unreachable!(),
            });
            while !owners[0].complete() {
                step(&mut owners);
            }
            let winning_score = owners.iter().fold(0u64, |sum, owner| {
                sum.wrapping_add(owner.scores.as_ref().unwrap().first[0])
            });
            assert_eq!(winning_score as i64, values[expected]);
            let outputs = owners.map(|owner| owner.finish().unwrap());
            let actual = outputs
                .iter()
                .fold(0u64, |sum, o| sum.wrapping_add(o.1.first[0]));
            assert_eq!(actual, expected as u64, "n={}", values.len());
            for i in 0..3 {
                assert_eq!(outputs[i].1.second[0], outputs[(i + 1) % 3].1.first[0]);
                let actual = outputs[i].0.cursors();
                let bound = plan.work_budget().cursors();
                assert_eq!(actual.public_blocks, bound.public_blocks);
                assert_eq!(actual.pair_blocks, bound.pair_blocks);
                assert!(
                    actual.private_blocks <= bound.private_blocks
                        && actual.aes_work <= bound.aes_work
                );
                assert_eq!(usages[i].live_bytes(), 0);
            }
        }
    }

    #[test]
    fn stable_selection_geometry_costs_and_admission() {
        assert!(Plan::new(0, Interval::new(0, 1).unwrap()).is_err());
        assert!(Plan::new(MAX_VOCABULARY + 1, Interval::new(0, 1).unwrap()).is_err());
        assert!(Plan::new(3, Interval::new(-(1i64 << 61), 1i64 << 61).unwrap()).is_err());
        for n in [1, 151936, MAX_VOCABULARY] {
            let plan = Plan::new(n, Interval::new(-1000000, 1000000).unwrap()).unwrap();
            assert!(plan.peak_extra_bytes().unwrap() <= MAX_EXTRA_BYTES);
            assert_eq!(plan.logical_cost().0, (n - 1) as u64);
            assert_eq!(plan.logical_cost().1, (2 * (n - 1)) as u64);
            eprintln!("stable argmax V={n}: work={:?}; logical={:?}; metadata={}; peak_extra={}; primitive_workspace={}",
                plan.work_budget(), plan.logical_cost(), plan.metadata_bytes().unwrap(), plan.peak_extra_bytes().unwrap(), plan.predicate_workspace_bytes());
        }
    }

    #[test]
    fn stable_selection_tensor_and_product_cancel_malformed_and_replay_burn_owners() {
        test_stack(failures);
    }

    fn failures() {
        for product_phase in [false, true] {
            for attack in 0..3 {
                let (mut owners, _) = owners(&[3, 8, 8], Interval::new(3, 8).unwrap());
                let usages = owners.each_ref().map(|owner| match &owner.active {
                    Active::Ready(runtime) => runtime.payload_usage(),
                    _ => unreachable!(),
                });
                if product_phase {
                    while owners[0].selector.is_none() {
                        step(&mut owners);
                    }
                }
                let parties = [Party::P0, Party::P1, Party::P2];
                let mut frames = Vec::new();
                for (i, owner) in owners.iter_mut().enumerate() {
                    frames.extend(
                        owner
                            .start_step()
                            .unwrap()
                            .into_iter()
                            .map(|(to, frame)| (parties[i], to, frame)),
                    );
                }
                assert!(owners.iter().all(|o| if product_phase {
                    matches!(o.active, Active::Product(_))
                } else {
                    matches!(o.active, Active::Negative(_))
                }));
                if attack == 1 {
                    assert!(owners[0].receive_encoded(b"malformed", Party::P1).is_err());
                    assert!(owners[0].retired());
                } else if attack == 2 {
                    let (from, to, frame) = frames.first().expect("phase emits frames");
                    let raw = frame.as_bytes().to_vec();
                    let recipient = &mut owners[*to as usize];
                    drop(recipient.receive_encoded(&raw, *from).unwrap());
                    assert!(recipient.receive_encoded(&raw, *from).is_err());
                    assert!(recipient.retired());
                }
                drop(frames);
                for (i, owner) in owners.iter_mut().enumerate() {
                    owner.cancel();
                    assert!(owner.retired());
                    assert!(
                        owner.scores.is_none()
                            && owner.indices.is_none()
                            && owner.selector.is_none()
                    );
                    assert_eq!(usages[i].live_bytes(), 0);
                    assert!(owner.start_step().is_err());
                }
            }
        }
    }
}
