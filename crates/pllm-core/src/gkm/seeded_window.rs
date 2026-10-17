//! Seed-only designated-worker integration with the existing typed transaction.
use super::*;

impl Client {
    /// Admit a complete computational-privacy AES window. Deliver only the
    /// returned worker owner to worker One; worker Zero uses explicit shares.
    /// This is bounded/in-process ownership, not a transport or seed-export API.
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
    plan: Arc<Vec<Step>>,
    base: Binding,
    ordinal: usize,
    // Last: key/cursor and shared full-window plan must drop before reservation.
    _lease: Arc<Lease>,
}

impl SeededWorker {
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
        self.ordinal += 1;
        if self.ordinal == self.plan.len() && !self.streams.complete() {
            return Err(Error::Incomplete);
        }
        self._lease
            .seeded_completed
            .store(self.ordinal as u32, Ordering::Release);
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
            let mut word = Zeroizing::new([0; 8]);
            for value in mask.iter_mut() {
                self.streams.linear.fill(&mut *word)?;
                *value = u64::from_le_bytes(*word);
            }
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
            self.streams.lookup.fill(&mut selector)?;
            if table.vocabulary % 8 != 0 {
                let last = selector.len() - 1;
                selector[last] &= (1u8 << (table.vocabulary % 8)) - 1;
            }
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
