//! Seed-only designated-worker integration with the existing typed transaction.
use super::*;

impl Client {
    /// Admit a complete computational-privacy AES window. Deliver only the
    /// returned worker owner to worker One; worker Zero uses explicit shares.
    /// This returns bounded in-process ownership. `begin_seeded_wire` performs
    /// the separately bound, consuming pre-expansion transport transfer.
    /// Both purpose keys are fresh OS draws once per window. Verifier challenges
    /// continue to use independent OS field-scalar draws after response freeze.
    pub fn begin_seeded(
        &self,
        context: [u8; 32],
        plan: &[Step],
    ) -> Result<(Window, SeededWorker), Error> {
        self.begin_seeded_with(context, plan, &mut OsEntropy)
    }

    fn begin_seeded_with(
        &self,
        context: [u8; 32],
        plan: &[Step],
        rng: &mut impl Entropy,
    ) -> Result<(Window, SeededWorker), Error> {
        let mut window = self.begin_profile(context, plan, SharingProfile::Aes128WindowV1)?;
        let mut linear_bytes = 0usize;
        let mut lookup_bytes = 0usize;
        for step in plan {
            match step {
                Step::Lookup => {
                    lookup_bytes = lookup_bytes
                        .checked_add(self.publication.row_hashes.len().div_ceil(8))
                        .ok_or(Error::Limit)?;
                }
                Step::Linear(id) => {
                    let matrix = &self.publication.matrices[self.publication.matrix_index(id)?];
                    linear_bytes = linear_bytes
                        .checked_add(matrix.columns * 8)
                        .ok_or(Error::Limit)?;
                }
            }
        }
        let mut linear_key = Zeroizing::new([0; 16]);
        let mut lookup_key = Zeroizing::new([0; 16]);
        rng.fill(Purpose::LinearSeed, &mut *linear_key)?;
        rng.fill(Purpose::LookupSeed, &mut *lookup_key)?;
        window.streams = Some(Streams::new(
            &linear_key,
            &lookup_key,
            linear_bytes,
            lookup_bytes,
        )?);
        let worker = SeededWorker {
            streams: Streams::new(&linear_key, &lookup_key, linear_bytes, lookup_bytes)?,
            export_keys: Some(Zeroizing::new([*linear_key, *lookup_key])),
            plan: Arc::clone(&window.plan),
            base: window.binding(Worker::One)?,
            ordinal: 0,
            _lease: Arc::clone(&window._lease),
        };
        Ok((window, worker))
    }
}

/// One opaque designated-worker owner for one exact seeded window. No Clone,
/// Debug, seed/cursor export, serialization, rewind, or resumption. Dropping it
/// early retires the transaction. The worker's intrinsic knowledge of its seeds
/// is not an adaptive-erasure security guarantee.
///
/// ```compile_fail
/// use pllm_core::gkm::SeededWorker;
/// fn clone_worker(worker: &SeededWorker) { let _: SeededWorker = worker.clone(); }
/// ```
pub struct SeededWorker {
    streams: Streams,
    // One consuming, pre-expansion transfer only. Never exposed by a reusable getter.
    export_keys: Option<Zeroizing<[[u8; 16]; 2]>>,
    plan: Arc<Vec<Step>>,
    base: Binding,
    ordinal: usize,
    // Last: key/cursor and shared full-window plan must drop before reservation.
    _lease: Arc<Lease>,
}

impl SeededWorker {
    pub(super) fn transfer_keys(
        mut self,
        lease: &Arc<Lease>,
    ) -> Result<Zeroizing<[[u8; 16]; 2]>, Error> {
        self.check(self.base, lease)?;
        if self.ordinal != 0 {
            return Err(Error::Retired);
        }
        let keys = self.export_keys.take().ok_or(Error::Retired)?;
        // Transfer retires this owner, not the destination transaction.
        self.ordinal = self.plan.len();
        Ok(keys)
    }

    pub(super) fn from_wire(
        keys: &[[u8; 16]; 2],
        plan: Arc<Vec<Step>>,
        base: Binding,
        lease: Arc<Lease>,
    ) -> Result<Self, Error> {
        if base.worker != Worker::One || base.profile != SharingProfile::Aes128WindowV1 {
            return Err(Error::Binding);
        }
        let publication = &lease.budget.publication;
        let mut linear = 0usize;
        let mut lookup = 0usize;
        for step in plan.iter() {
            match step {
                Step::Lookup => {
                    lookup = lookup
                        .checked_add(publication.row_hashes.len().div_ceil(8))
                        .ok_or(Error::Limit)?
                }
                Step::Linear(id) => {
                    linear = linear
                        .checked_add(
                            publication.matrices[publication.matrix_index(id)?].columns * 8,
                        )
                        .ok_or(Error::Limit)?
                }
            }
        }
        Ok(Self {
            streams: Streams::new(&keys[0], &keys[1], linear, lookup)?,
            export_keys: None,
            plan,
            base,
            ordinal: 0,
            _lease: lease,
        })
    }

    fn check(&self, binding: Binding, lease: &Arc<Lease>) -> Result<(), Error> {
        self._lease.ensure_open()?;
        if !Arc::ptr_eq(lease, &self._lease) {
            return Err(Error::Binding);
        }
        let step = *self.plan.get(self.ordinal).ok_or(Error::Schedule)?;
        let publication = &self._lease.budget.publication;
        let artifact = match step {
            Step::Lookup => publication.table_digest,
            Step::Linear(id) => publication.matrices[publication.matrix_index(&id)?].digest,
        };
        let expected = Binding {
            ordinal: self.ordinal as u32,
            step,
            artifact,
            ..self.base
        };
        if binding != expected {
            return Err(Error::Binding);
        }
        Ok(())
    }

    fn complete_step(&mut self) -> Result<(), Error> {
        self.commit_rows(1)
    }

    pub(super) fn commit_rows(&mut self, rows: usize) -> Result<(), Error> {
        self._lease.ensure_open()?;
        self.export_keys = None;
        self.ordinal = self
            .ordinal
            .checked_add(rows)
            .filter(|&end| end <= self.plan.len())
            .ok_or(Error::Schedule)?;
        if self.ordinal == self.plan.len() && !self.streams.complete() {
            return Err(Error::Incomplete);
        }
        self._lease
            .seeded_completed
            .store(self.ordinal as u32, Ordering::Release);
        Ok(())
    }

    fn check_rows(&self, binding: Binding, lease: &Arc<Lease>, rows: usize) -> Result<(), Error> {
        self.check(binding, lease)?;
        if rows == 0 {
            return Err(Error::Shape);
        }
        let end = self.ordinal.checked_add(rows).ok_or(Error::Limit)?;
        if self
            .plan
            .get(self.ordinal..end)
            .ok_or(Error::Schedule)?
            .iter()
            .any(|step| *step != binding.step)
        {
            return Err(Error::Binding);
        }
        Ok(())
    }

    pub(super) fn fill_linear_rows(
        &mut self,
        binding: Binding,
        lease: &Arc<Lease>,
        rows: usize,
        output: &mut [u64],
    ) -> Result<(), Error> {
        self.check_rows(binding, lease, rows)?;
        let Step::Linear(id) = binding.step else {
            return Err(Error::Binding);
        };
        let public = &lease.budget.publication;
        let columns = public.matrices[public.matrix_index(&id)?].columns;
        if output.len() != rows.checked_mul(columns).ok_or(Error::Limit)? {
            return Err(Error::Shape);
        }
        let mut word = Zeroizing::new([0u8; 8]);
        for value in output {
            self.streams.linear.fill(&mut *word)?;
            *value = u64::from_le_bytes(*word);
        }
        Ok(())
    }

    pub(super) fn fill_lookup_rows(
        &mut self,
        binding: Binding,
        lease: &Arc<Lease>,
        rows: usize,
        output: &mut [u8],
    ) -> Result<(), Error> {
        self.check_rows(binding, lease, rows)?;
        if binding.step != Step::Lookup {
            return Err(Error::Binding);
        }
        let vocabulary = lease.budget.publication.row_hashes.len();
        let stride = vocabulary.div_ceil(8);
        if output.len() != rows.checked_mul(stride).ok_or(Error::Limit)? {
            return Err(Error::Shape);
        }
        for selector in output.chunks_exact_mut(stride) {
            self.streams.lookup.fill(selector)?;
            if vocabulary % 8 != 0 {
                selector[stride - 1] &= (1u8 << (vocabulary % 8)) - 1;
            }
        }
        Ok(())
    }

    /// Expands exactly this step locally, then evaluates the immutable matrix.
    /// The ordinary response slot is spent before any allocation or expansion.
    pub fn evaluate_linear(
        &mut self,
        matrix: &I16Matrix,
        share: &LinearShare<'_>,
    ) -> Result<LinearResponse, Error> {
        let result = share.slot.produce(share.lease, || {
            self.check(share.binding, share.lease)?;
            if share.values.is_some() || share.binding.artifact != matrix.digest {
                return Err(Error::Binding);
            }
            let mut mask = Zeroizing::new(payload::filled(matrix.columns, 0u64)?);
            self.fill_linear_rows(share.binding, share.lease, 1, &mut mask)?;
            let values = matrix.multiply_share(&mask)?;
            self.complete_step()?;
            Ok(LinearResponse {
                binding: share.binding,
                values,
                _lease: Arc::clone(share.lease),
            })
        });
        if result.is_err() {
            // A foreign/failed slot also retires this worker's original owner.
            self._lease.poisoned.store(true, Ordering::Release);
        }
        result
    }

    pub fn evaluate_lookup(
        &mut self,
        table: &FloatTable,
        share: &LookupShare<'_>,
    ) -> Result<LookupResponse, Error> {
        let result = share.slot.produce(share.lease, || {
            self.check(share.binding, share.lease)?;
            if share.selector.is_some() || share.binding.artifact != table.digest {
                return Err(Error::Binding);
            }
            let mut selector = Zeroizing::new(payload::filled(table.vocabulary.div_ceil(8), 0)?);
            self.fill_lookup_rows(share.binding, share.lease, 1, &mut selector)?;
            let bytes = table.xor_rows(&selector)?;
            self.complete_step()?;
            Ok(LookupResponse {
                binding: share.binding,
                bytes,
                _lease: Arc::clone(share.lease),
            })
        });
        if result.is_err() {
            self._lease.poisoned.store(true, Ordering::Release);
        }
        result
    }
}

impl Drop for SeededWorker {
    fn drop(&mut self) {
        if self.ordinal != self.plan.len() {
            self._lease.poisoned.store(true, Ordering::Release);
        }
        zeroize::Zeroize::zeroize(&mut self.ordinal);
    }
}

#[cfg(test)]
#[path = "seeded_window_tests.rs"]
mod tests;
