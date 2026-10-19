//! Bounded full-vocabulary public-weight lookup. Compiler integration must bind
//! geometry, token interval, source owner and work budget into the graph digest.
//! This owner is per selection, never a prefix cache. No new primitive limits.
use crate::replicated_decoder::*;
use multiply::PartyRuntime;
use pllm_core::public_file::PublicFile;
use puma::{Operation, Party, ReplicatedShare};
use std::collections::BTreeMap;

pub(in crate::replicated_decoder) const ROW_TILE: usize = 8;
pub(in crate::replicated_decoder) const VOCAB_TILE: usize = 128;
pub(in crate::replicated_decoder) const ACCUMULATOR_WORDS: usize = 65536;

#[derive(Clone)]
pub(in crate::replicated_decoder) struct Plan {
    rows: usize,
    vocabulary: usize,
    width: usize,
    full: Arc<tensor::Program>,
    tail: Option<Arc<tensor::Program>>,
}

impl Plan {
    /// Immutable programs only. Each invocation still constructs a fresh native
    /// executor with the consuming runtime's monotonic PRSS cursor.
    pub fn programs(&self) -> impl Iterator<Item = &Arc<tensor::Program>> {
        std::iter::once(&self.full).chain(self.tail.iter())
    }

    pub fn reuse_programs(
        &mut self,
        programs: &BTreeMap<[u8; 32], Arc<tensor::Program>>,
    ) -> Result<(), Error> {
        for program in std::iter::once(&mut self.full).chain(self.tail.iter_mut()) {
            let digest = program.digest();
            if let Some(shared) = programs.get(&digest) {
                if shared.digest() != digest {
                    return Err(Error::Binding);
                }
                *program = shared.clone();
            }
        }
        Ok(())
    }

    /// Bind geometry, bounded quotient-ring/compact-wire primitive identities,
    /// and their source-fill work budgets. Pointer interning cannot change it.
    pub fn digest(&self) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(b"pllm.replicated-decoder.public-lookup.bounded-compact.v2");
        for n in [self.rows, self.vocabulary, self.width, ROW_TILE, VOCAB_TILE] {
            hash.update((n as u64).to_le_bytes());
        }
        hash.update([self.tail.is_some() as u8]);
        for program in self.programs() {
            hash.update(program.digest());
            let profile = program.numeric_profile().as_bytes();
            hash.update((profile.len() as u64).to_le_bytes());
            hash.update(profile);
            let work = program.work_budget();
            let cursors = work.cursors();
            for count in [
                work.operations(),
                cursors.public_blocks,
                cursors.pair_blocks,
                cursors.private_blocks,
                cursors.aes_work,
            ] {
                hash.update(count.to_le_bytes());
            }
        }
        hash.finalize().into()
    }
    pub fn geometry(&self) -> (usize, usize, usize) {
        (self.rows, self.vocabulary, self.width)
    }

    /// Each Code::PublicLookup retains one Arc<Plan>. Its inner Arc handles are
    /// inline here, but their pointees must be charged separately, once/digest.
    pub fn shallow_metadata_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + 2 * std::mem::size_of::<usize>()
    }

    /// Bounded equality programs have no heap public-affine arrays: this is the
    /// complete Arc payload and reference counters, excluding allocator slop or
    /// the compiler's BTreeMap entry (priced by the enclosing registry).
    pub fn program_metadata_bytes(program: &tensor::Program) -> usize {
        std::mem::size_of_val(program) + 2 * std::mem::size_of::<usize>()
    }

    /// Standalone deep metadata ownership. Across plans, charge shallow bytes
    /// per plan and program_metadata_bytes only for unseen programs() digests.
    pub fn metadata_bytes(&self) -> Result<usize, Error> {
        self.programs()
            .try_fold(self.shallow_metadata_bytes(), |n, p| {
                n.checked_add(Self::program_metadata_bytes(p))
                    .ok_or(Error::Bounds)
            })
    }

    /// Conservative serving overlap, including standalone deep metadata, fixed
    /// native owner, both accumulator/selector/token components, tile copies,
    /// 64 KiB read buffer and the original active tensor workspace/clone.
    /// Source files and caller's original token owner are separate. The output
    /// accumulator is INCLUDED: do not accidentally omit it when subtracting a
    /// separately priced graph output. Buffer and tensor scratch are summed
    /// conservatively although their highest-use phases do not coincide.
    pub fn peak_extra_bytes(&self) -> Result<usize, Error> {
        let tile = VOCAB_TILE.min(self.vocabulary);
        self.metadata_bytes()?
            .checked_add(std::mem::size_of::<BulkLookup>())
            .and_then(|n| n.checked_add(self.rows.checked_mul(16)?))
            .and_then(|n| n.checked_add(self.rows.checked_mul(tile)?.checked_mul(16)?))
            .and_then(|n| n.checked_add(self.rows.checked_mul(self.width)?.checked_mul(16)?))
            .and_then(|n| n.checked_add(tile.checked_mul(32)?))
            .and_then(|n| n.checked_add(65536))
            .and_then(|n| n.checked_add(self.predicate_workspace_bytes()))
            .ok_or(Error::Bounds)
    }

    /// Conservative original primitive workspace; full and tail never overlap.
    pub fn predicate_workspace_bytes(&self) -> usize {
        self.full
            .admitted_bytes()
            .max(self.tail.as_ref().map_or(0, |p| p.admitted_bytes()))
    }

    pub fn new(
        rows: usize,
        vocabulary: usize,
        width: usize,
        tokens: Interval,
    ) -> Result<Self, Error> {
        if rows == 0
            || rows > ROW_TILE
            || vocabulary == 0
            || width == 0
            || rows
                .checked_mul(width)
                .is_none_or(|n| n > ACCUMULATOR_WORDS)
            || (rows as u64)
                .checked_mul(vocabulary as u64)
                .and_then(|n| n.checked_mul(width as u64))
                .is_none_or(|n| n > 1 << 32)
        {
            return Err(Error::Bounds);
        }
        let last = i64::try_from(vocabulary - 1).map_err(|_| Error::Bounds)?;
        let difference = Interval::new(
            tokens.lower().checked_sub(last).ok_or(Error::Bounds)?,
            tokens.upper(),
        )?;
        // Retain quarter-gap admission. The bounded equality constructor proves
        // no nonzero multiple of its public power-of-two modulus lies in this
        // difference interval; modular shares themselves remain full ring64.
        if difference.lower() < puma::INPUT_MIN || difference.upper() >= puma::INPUT_MAX_EXCLUSIVE {
            return Err(Error::Bounds);
        }
        let full = Arc::new(tensor::Program::equal_zero_bounded(
            VOCAB_TILE.min(vocabulary),
            difference,
        )?);
        let tail = if vocabulary > VOCAB_TILE && vocabulary % VOCAB_TILE != 0 {
            Some(Arc::new(tensor::Program::equal_zero_bounded(
                vocabulary % VOCAB_TILE,
                difference,
            )?))
        } else {
            None
        };
        Ok(Self {
            rows,
            vocabulary,
            width,
            full,
            tail,
        })
    }

    pub fn work_budget(&self) -> Result<WorkBudget, Error> {
        let tiles = self.vocabulary / VOCAB_TILE;
        let row = if tiles == 0 {
            self.full.work_budget()
        } else {
            self.full
                .work_budget()
                .checked_mul(tiles as u64)?
                .checked_add(
                    self.tail
                        .as_ref()
                        .map_or(WorkBudget::default(), |p| p.work_budget()),
                )?
        };
        row.checked_mul(self.rows as u64)
    }

    /// Logical counts, not physical disk traffic or measured elapsed costs.
    pub fn logical_cost(&self) -> Result<(u64, u64), Error> {
        let words = (self.vocabulary as u64)
            .checked_mul(self.width as u64)
            .ok_or(Error::Bounds)?;
        Ok((
            words.checked_mul(8).ok_or(Error::Bounds)?,
            words
                .checked_mul(self.rows as u64)
                .and_then(|n| n.checked_mul(2))
                .ok_or(Error::Bounds)?,
        ))
    }

    fn count(&self, start: usize) -> usize {
        VOCAB_TILE.min(self.vocabulary - start)
    }
    fn program(&self, start: usize) -> &tensor::Program {
        if self.count(start) < self.full.lanes() {
            self.tail.as_ref().unwrap()
        } else {
            &self.full
        }
    }
}

/// Primitive executor owns its original ledger. Additional native owners: two
/// token components (<=128 B), selectors (<=16 KiB), accumulators (<=1 MiB), and
/// transient 64 KiB public buffer. Compiler must price their overlap separately.
pub(in crate::replicated_decoder) struct BulkLookup {
    plan: Plan,
    tokens: Option<Secret>,
    selectors: Option<Secret>,
    output: Option<Secret>,
    runtime: Option<PartyRuntime>,
    executor: Option<tensor::Executor>,
    start: usize,
    row: usize,
    retired: bool,
}

impl BulkLookup {
    pub fn new(runtime: PartyRuntime, tokens: Secret, plan: Plan) -> Result<Self, Error> {
        if runtime.status() != multiply::Status::Ready {
            return Err(Error::Phase);
        }
        if tokens.first.len() != plan.rows || tokens.second.len() != plan.rows {
            return Err(Error::Shape);
        }
        let selectors = Secret::new(plan.rows * plan.count(0))?;
        let output = Secret::new(plan.rows * plan.width)?;
        Ok(Self {
            plan,
            tokens: Some(tokens),
            selectors: Some(selectors),
            output: Some(output),
            runtime: Some(runtime),
            executor: None,
            start: 0,
            row: 0,
            retired: false,
        })
    }

    pub fn complete(&self) -> bool {
        !self.retired && self.start == self.plan.vocabulary
    }
    pub fn cancel(&mut self) {
        if let Some(executor) = &mut self.executor {
            executor.cancel();
        }
        self.executor = None;
        self.runtime = None;
        self.tokens = None;
        self.selectors = None;
        self.output = None;
        self.retired = true;
    }
    fn checked<T>(&mut self, result: Result<T, Error>) -> Result<T, Error> {
        if result.is_err() {
            self.cancel();
        }
        result
    }

    pub fn start_step(&mut self) -> Result<[Option<tensor::Frame>; 3], Error> {
        let result =
            (|| {
                if self.retired || self.complete() {
                    return Err(Error::Phase);
                }
                if self.executor.is_none() {
                    let runtime = self.runtime.take().ok_or(Error::Phase)?;
                    let n = self.plan.count(self.start);
                    let party = runtime.party();
                    let tokens = self.tokens.as_ref().ok_or(Error::Retired)?;
                    let mut shifted = Secret::new(n)?;
                    for i in 0..n {
                        let public = (self.start + i) as u64;
                        shifted.first[i] = tokens.first[self.row]
                            .wrapping_sub(if party == Party::P0 { public } else { 0 });
                        shifted.second[i] = tokens.second[self.row]
                            .wrapping_sub(if party.next() == Party::P0 { public } else { 0 });
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
                        party,
                        &shifted.first,
                        &shifted.second,
                        puma::Limits::default(),
                    )?;
                    // Fresh consuming executor uses the monotonic runtime PRSS cursor.
                    self.executor = Some(tensor::Executor::new(
                        runtime,
                        self.plan.program(self.start).clone(),
                        input,
                    )?);
                }
                self.executor.as_mut().ok_or(Error::Phase)?.start_step()
            })();
        self.checked(result)
    }

    pub fn receive_encoded(
        &mut self,
        raw: &[u8],
        sender: Party,
    ) -> Result<[Option<tensor::Frame>; 3], Error> {
        let result = self
            .executor
            .as_mut()
            .ok_or(Error::Phase)
            .and_then(|e| e.receive_encoded(raw, sender));
        self.checked(result)
    }

    pub fn finish_step(&mut self, file: &PublicFile) -> Result<(), Error> {
        let result = (|| {
            let executor = self.executor.as_mut().ok_or(Error::Phase)?;
            executor.finish_step()?;
            if !executor.complete() {
                return Ok(());
            }
            let (runtime, result) = self.executor.take().ok_or(Error::Phase)?.finish()?;
            self.runtime = Some(runtime);
            let n = self.plan.count(self.start);
            let selectors = self.selectors.as_mut().ok_or(Error::Retired)?;
            selectors.first[self.row * n..(self.row + 1) * n].copy_from_slice(result.first());
            selectors.second[self.row * n..(self.row + 1) * n].copy_from_slice(result.second());
            self.row += 1;
            if self.row == self.plan.rows {
                public_source::accumulate_rows(
                    file,
                    selectors,
                    self.output.as_mut().ok_or(Error::Retired)?,
                    self.plan.rows,
                    self.plan.vocabulary,
                    self.plan.width,
                    self.start,
                    n,
                )?;
                self.start += n;
                self.row = 0;
                // Reuse selector capacity for full tiles. Tail needs exact shape.
                if !self.complete() && self.plan.count(self.start) != n {
                    self.selectors = None;
                    self.selectors =
                        Some(Secret::new(self.plan.rows * self.plan.count(self.start))?);
                }
            }
            Ok(())
        })();
        self.checked(result)
    }

    pub fn finish(mut self) -> Result<(PartyRuntime, Secret), Error> {
        if !self.complete() {
            return Err(Error::Phase);
        }
        Ok((
            self.runtime.take().ok_or(Error::Phase)?,
            self.output.take().ok_or(Error::Phase)?,
        ))
    }
}

impl Drop for BulkLookup {
    fn drop(&mut self) {
        self.cancel();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pllm_core::public_file::PublicFileWriter;
    use puma::{
        prss::{PrssLimits, PublicContribution, SeedMessage},
        Context,
    };

    #[test]
    fn bulk_lookup_owner_layout() {
        eprintln!(
            "lookup Plan={} BulkLookup={}",
            std::mem::size_of::<Plan>(),
            std::mem::size_of::<BulkLookup>()
        );
        assert!(std::mem::size_of::<Plan>() < 128);
        assert!(std::mem::size_of::<BulkLookup>() < 20000);
    }

    #[test]
    fn bulk_lookup_compact_identity_work_and_program_reuse_are_bound() {
        let bounds = Interval::new(0, 130).unwrap();
        let plan = Plan::new(8, 131, 71, bounds).unwrap();
        let difference = Interval::new(-130, 130).unwrap();
        let expected =
            [128, 3].map(|n| tensor::Program::equal_zero_bounded(n, difference).unwrap());
        assert_eq!(plan.programs().count(), 2);
        for (actual, expected) in plan.programs().zip(&expected) {
            assert_eq!(actual.digest(), expected.digest());
            assert_eq!(actual.numeric_profile(), tensor::BOUNDED_EQUALITY_PROFILE);
            assert_ne!(
                actual.digest(),
                tensor::Program::equal_zero(actual.lanes(), difference)
                    .unwrap()
                    .digest()
            );
        }
        assert_eq!(
            plan.metadata_bytes().unwrap(),
            plan.shallow_metadata_bytes()
                + 2 * (std::mem::size_of::<tensor::Program>() + 2 * std::mem::size_of::<usize>())
        );
        assert_eq!(
            plan.peak_extra_bytes().unwrap(),
            plan.metadata_bytes().unwrap()
                + std::mem::size_of::<BulkLookup>()
                + 8 * 16
                + 8 * 128 * 16
                + 8 * 71 * 16
                + 128 * 32
                + 65536
                + plan.predicate_workspace_bytes()
        );
        let maximum = Plan::new(8, 131, ACCUMULATOR_WORDS / 8, bounds).unwrap();
        assert!(maximum.peak_extra_bytes().unwrap() < 2 << 20);
        eprintln!(
            "lookup shared metadata={}, peak_extra={}, maximum_accumulator_peak={}",
            plan.metadata_bytes().unwrap(),
            plan.peak_extra_bytes().unwrap(),
            maximum.peak_extra_bytes().unwrap()
        );
        let expected_work = expected[0]
            .work_budget()
            .checked_add(expected[1].work_budget())
            .unwrap()
            .checked_mul(8)
            .unwrap();
        assert_eq!(plan.work_budget().unwrap(), expected_work);
        assert_eq!(expected_work.operations(), 240);
        assert_eq!(expected_work.cursors().public_blocks, 16);
        assert_eq!(expected_work.cursors().pair_blocks, 2352);

        let cache: BTreeMap<_, _> = plan.programs().map(|p| (p.digest(), p.clone())).collect();
        let mut repeated = Plan::new(8, 131, 71, bounds).unwrap();
        let digest = repeated.digest();
        assert!(!Arc::ptr_eq(&repeated.full, &plan.full));
        repeated.reuse_programs(&cache).unwrap();
        assert_eq!(repeated.digest(), digest);
        assert_eq!(repeated.digest(), plan.digest());
        for (actual, original) in repeated.programs().zip(plan.programs()) {
            assert!(Arc::ptr_eq(actual, original));
        }
        for (rows, width, bounds) in [
            (1, 71, bounds),
            (8, 72, bounds),
            (8, 71, Interval::new(0, 131).unwrap()),
        ] {
            assert_ne!(
                Plan::new(rows, 131, width, bounds).unwrap().digest(),
                plan.digest()
            );
        }
        let legacy = Arc::new(tensor::Program::equal_zero(128, difference).unwrap());
        let forged = BTreeMap::from([(repeated.full.digest(), legacy)]);
        assert_eq!(repeated.reuse_programs(&forged), Err(Error::Binding));
    }

    fn runtimes(work: WorkBudget) -> [PartyRuntime; 3] {
        let parties = [Party::P0, Party::P1, Party::P2];
        let context = Context::new([61; 32], [72; 32], 0).unwrap();
        let limits = PrssLimits::program(work.operations(), work.cursors().aes_work).unwrap();
        let mut setup = parties.map(|p| multiply::SetupParty::new(context, p, limits).unwrap());
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

    fn file(weights: &[u64]) -> PublicFile {
        let bytes: Vec<_> = weights.iter().flat_map(|w| w.to_le_bytes()).collect();
        let digest = Sha256::digest(&bytes).into();
        let mut writer = PublicFileWriter::new(bytes.len() as u64, digest).unwrap();
        for chunk in bytes.chunks(65536) {
            writer.write(chunk).unwrap();
        }
        writer.finish().unwrap()
    }

    fn owners(ids: &[u64], plan: &Plan) -> [BulkLookup; 3] {
        let shares = [
            vec![0xc751f00d12345678u64; ids.len()],
            vec![0x254277811928f93du64; ids.len()],
            ids.iter()
                .map(|id| {
                    id.wrapping_sub(0xc751f00d12345678)
                        .wrapping_sub(0x254277811928f93d)
                })
                .collect(),
        ];
        let mut index = 0;
        runtimes(plan.work_budget().unwrap()).map(|runtime| {
            let i = index;
            index += 1;
            let mut tokens = Secret::new(ids.len()).unwrap();
            tokens.first.copy_from_slice(&shares[i]);
            tokens.second.copy_from_slice(&shares[(i + 1) % 3]);
            BulkLookup::new(runtime, tokens, plan.clone()).unwrap()
        })
    }

    fn step(owners: &mut [BulkLookup; 3], file: &PublicFile) -> (usize, usize) {
        let parties = [Party::P0, Party::P1, Party::P2];
        let mut queue = Vec::new();
        let mut counts = (0, 0);
        for (i, owner) in owners.iter_mut().enumerate() {
            queue.extend(
                owner
                    .start_step()
                    .unwrap()
                    .into_iter()
                    .flatten()
                    .map(|f| (parties[i], f)),
            );
        }
        while !queue.is_empty() {
            let current = std::mem::take(&mut queue);
            for (sender, frame) in current {
                let recipient = frame.recipient() as usize;
                let raw = frame.encode().unwrap();
                counts.0 += 1;
                counts.1 += raw.as_bytes().len();
                assert_eq!(&raw.as_bytes()[..4], b"PUT2");
                assert_eq!(&raw.as_bytes()[48..52], b"PUC1");
                queue.extend(
                    owners[recipient]
                        .receive_encoded(raw.as_bytes(), sender)
                        .unwrap()
                        .into_iter()
                        .flatten()
                        .map(|f| (parties[recipient], f)),
                );
            }
        }
        for owner in owners {
            owner.finish_step(file).unwrap();
        }
        counts
    }

    #[test]
    fn bulk_lookup_width18_compact_tile_has_exact_wire_cost_and_retires_partial_owner() {
        let plan = Plan::new(1, 151936, 1, Interval::new(0, 151935).unwrap()).unwrap();
        let file = file(&vec![9; 151936]); // Generated 1.2 MiB scalar table, no checkpoint.
        let mut owners = owners(&[0], &plan);
        let usages = owners
            .each_ref()
            .map(|o| o.runtime.as_ref().unwrap().payload_usage());
        let mut counts = (0, 0);
        while owners[0].start == 0 {
            let current = step(&mut owners, &file);
            counts.0 += current.0;
            counts.1 += current.1;
        }
        assert_eq!(counts, (40, 19920));
        assert!(owners.iter().all(|o| o.start == 128 && !o.complete()));
        let partial = owners.iter().fold(0u64, |sum, o| {
            sum.wrapping_add(o.output.as_ref().unwrap().first[0])
        });
        assert_eq!(partial, 9);
        for (i, owner) in owners.iter_mut().enumerate() {
            let actual = owner.runtime.as_ref().unwrap().cursors();
            assert_eq!(actual.public_blocks, 1);
            assert_eq!(actual.pair_blocks, 648);
            assert!(actual.aes_work <= 1169);
            owner.cancel();
            assert_eq!(usages[i].live_bytes(), 0);
        }
    }

    #[test]
    fn bulk_lookup_independent_full_vocabulary_oracle_and_fresh_work() {
        independent_oracle();
    }

    fn independent_oracle() {
        for (vocabulary, width, ids) in [
            (3, 2, vec![2, 0, 2]),
            (128, 65, vec![127, 0]),
            (131, 71, vec![130, 128, 127, 0, 1, 130, 64, 17]),
        ] {
            let plan = Plan::new(
                ids.len(),
                vocabulary,
                width,
                Interval::new(0, vocabulary as i64 - 1).unwrap(),
            )
            .unwrap();
            let weights: Vec<_> = (0..vocabulary * width)
                .map(|i| {
                    (i as u64)
                        .wrapping_mul(0xfedcba9876543211)
                        .wrapping_add(0x8000000000000000)
                })
                .collect();
            let file = file(&weights);
            let mut owners = owners(&ids, &plan);
            let began = std::time::Instant::now();
            let usages = owners
                .each_ref()
                .map(|o| o.runtime.as_ref().unwrap().payload_usage());
            while !owners[0].complete() {
                step(&mut owners, &file);
            }
            let outputs = owners.map(|o| o.finish().unwrap());
            eprintln!("bulk lookup rows={} vocabulary={vocabulary} width={width}: elapsed={:?}; operations_bound={}; cursors={:?}; logical={:?}",
                ids.len(), began.elapsed(), plan.work_budget().unwrap().operations(),
                outputs.each_ref().map(|o| o.0.cursors()), plan.logical_cost().unwrap());
            for i in 0..3 {
                let actual = outputs[i].0.cursors();
                let bound = plan.work_budget().unwrap().cursors();
                assert_eq!(actual.public_blocks, bound.public_blocks);
                assert_eq!(actual.pair_blocks, bound.pair_blocks);
                assert_eq!(actual.private_blocks, bound.private_blocks);
                assert!(actual.aes_work <= bound.aes_work);
                assert_eq!(usages[i].live_bytes(), 0);
                assert_eq!(&outputs[i].1.second[..], &outputs[(i + 1) % 3].1.first[..]);
            }
            // Independent direct-index oracle, never production matrix helpers.
            for (row, id) in ids.iter().enumerate() {
                for c in 0..width {
                    let j = row * width + c;
                    let clear = outputs
                        .iter()
                        .fold(0u64, |sum, o| sum.wrapping_add(o.1.first[j]));
                    assert_eq!(clear, weights[*id as usize * width + c]);
                }
            }
            assert_eq!(
                plan.logical_cost().unwrap(),
                (
                    (vocabulary * width * 8) as u64,
                    (2 * ids.len() * vocabulary * width) as u64
                )
            );
        }
    }

    #[test]
    fn bulk_lookup_caps_domain_and_cancellation() {
        caps_and_cancellation();
    }

    fn caps_and_cancellation() {
        assert!(Plan::new(9, 3, 2, Interval::new(0, 2).unwrap()).is_err());
        assert!(Plan::new(8, 1 << 30, 2, Interval::new(0, 2).unwrap()).is_err());
        assert!(Plan::new(8, 3, 8193, Interval::new(0, 2).unwrap()).is_err());
        assert!(Plan::new(1, 3, 2, Interval::new(i64::MIN, i64::MAX).unwrap()).is_err());
        let plan = Plan::new(2, 131, 2, Interval::new(0, 130).unwrap()).unwrap();
        for mut owner in owners(&[0, 130], &plan) {
            let usage = owner.runtime.as_ref().unwrap().payload_usage();
            owner.start_step().unwrap();
            owner.cancel();
            assert_eq!(usage.live_bytes(), 0);
            assert!(owner.tokens.is_none() && owner.selectors.is_none() && owner.output.is_none());
            assert!(owner.start_step().is_err());
        }
        for mut owner in owners(&[0, 130], &plan) {
            let usage = owner.runtime.as_ref().unwrap().payload_usage();
            owner.start_step().unwrap();
            assert!(owner.receive_encoded(b"bad", Party::P0).is_err());
            assert!(owner.retired);
            assert_eq!(usage.live_bytes(), 0);
        }
        let file = file(&vec![17; 131 * 2]);
        let mut active = owners(&[0, 130], &plan);
        while active[0].start == 0 {
            step(&mut active, &file);
        }
        assert_eq!(active[0].start, 128);
        for owner in &mut active {
            let usage = owner.runtime.as_ref().unwrap().payload_usage();
            owner.cancel();
            assert_eq!(usage.live_bytes(), 0);
            assert!(owner.output.is_none() && owner.selectors.is_none());
        }
        let mut raw = [0u8; 8];
        file.read(0, &mut raw).unwrap(); // Cancellation never consumes immutable source.
        assert_eq!(u64::from_le_bytes(raw), 17);
    }

    #[test]
    fn bulk_lookup_row_range_matches_independent_ring_products() {
        let (rows, vocabulary, width) = (3, 137, 71);
        let weights: Vec<_> = (0..vocabulary * width)
            .map(|i| (i as u64).wrapping_mul(u64::MAX / 3))
            .collect();
        let file = file(&weights);
        let mut result = Secret::new(rows * width).unwrap();
        for (start, count) in [(0, 128), (128, 9)] {
            let mut selectors = Secret::new(rows * count).unwrap();
            for r in 0..rows {
                for k in 0..count {
                    selectors.first[r * count + k] =
                        (r * vocabulary + start + k) as u64 ^ 0xfedcba9876543210;
                    selectors.second[r * count + k] = !(selectors.first[r * count + k]);
                }
            }
            public_source::accumulate_rows(
                &file,
                &selectors,
                &mut result,
                rows,
                vocabulary,
                width,
                start,
                count,
            )
            .unwrap();
        }
        for r in 0..rows {
            for c in 0..width {
                let (mut a, mut b) = (0u64, 0u64);
                for k in 0..vocabulary {
                    let selector = (r * vocabulary + k) as u64 ^ 0xfedcba9876543210;
                    a = a.wrapping_add(selector.wrapping_mul(weights[k * width + c]));
                    b = b.wrapping_add((!selector).wrapping_mul(weights[k * width + c]));
                }
                assert_eq!(result.first[r * width + c], a);
                assert_eq!(result.second[r * width + c], b);
            }
        }
        let selectors = Secret::new(3).unwrap();
        assert!(public_source::accumulate_rows(
            &file,
            &selectors,
            &mut result,
            rows,
            vocabulary,
            width,
            vocabulary,
            1
        )
        .is_err());
    }
}
