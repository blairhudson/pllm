//! Arithmetic for public resource records and fixed cache bootstrap facts.
//! Bootstrap admits only an initial floor, not complete construction/peak cost.
//! Schedule phases borrow the live counters; no second ledger or Plan builder.
//! Logical accounting only, not physical-memory or full-capacity authority.
use super::{cache, wavefront, Capacity, Error, Limits, Resources};
use pllm_garble::fusefss_backend::composition;

fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Resource)
}
fn mul(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_mul(b).ok_or(Error::Resource)
}

pub(super) struct Quote<'a> {
    resources: &'a mut Resources,
    limits: Limits,
    capacity: Capacity,
    native: composition::Resources,
    wave: Option<wavefront::Resources>,
    inputs: usize,
    cached: bool,
}

impl<'a> Quote<'a> {
    pub(super) fn compiler_reservation(metadata: usize, live: usize) -> Result<usize, Error> {
        add(mul(metadata, 3)?, mul(live, 2)?)
    }

    // Borrow actual validated Program geometry and its last-use peak. This is
    // only the existing per-party live-owner tariff, not cumulative admission.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn live_array_reservation(
        peak: usize, sequence: usize, selections: usize, vocabulary: usize,
        batch_rows: usize, nodes: usize, capacity: Capacity, wavefront: bool,
    ) -> Result<usize, Error> {
        let output = mul(mul(8, super::MAX_VALUES)?, 8)?;
        let gather = if capacity == Capacity::Decoder {
            // Complete compact one-hot batch plus the two shared-reader buffers;
            // produced kernel/import owners remain in the native reservation.
            output.max(add(mul(mul(batch_rows, vocabulary)?, 8)?,
                mul(2, super::capacity::SOURCE_READ_BYTES)?)?)
        } else {
            output
        };
        let backing = mul(nodes,
            std::mem::size_of::<Option<zeroize::Zeroizing<Vec<u64>>>>())?;
        let live = add(add(add(peak, mul(8, add(sequence, selections)?)?)?, gather)?, backing)?;
        if wavefront {
            // Full tournament frontier and bounded lane-major gather/output.
            add(live, add(mul(8, add(vocabulary, 1)?)?,
                mul(mul(2, 8)?, composition::MAX_INPUTS)?)?)
        } else {
            Ok(live)
        }
    }

    pub(super) fn available_per_role(metadata: usize, live: usize) -> Result<usize, Error> {
        let compiler = Self::compiler_reservation(metadata, live)?;
        let available = pllm_garble::fusefss_backend::MAX_ALLOCATION_BYTES
            .checked_sub(compiler).filter(|&n| n > 0).ok_or(Error::Resource)?;
        let per_role = available / 3;
        if per_role == 0 { return Err(Error::Resource); }
        Ok(per_role)
    }

    // All inputs are already retained scalars or allocation-free cache facts.
    // The closure is the private construction boundary: refusal never calls it.
    pub(super) fn bootstrap<T>(
        metadata: usize,
        live: usize,
        source_reservation: usize,
        descriptor: cache::BootstrapDescriptor,
        limits: Limits,
        factory: impl FnOnce(Resources) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let cache = descriptor.metadata_reservation_bytes;
        let work = add(add(mul(metadata, 3)?, mul(cache, 3)?)?,
            mul(descriptor.numeric_revision_hash_bytes, 3)?)?;
        let copies = mul(source_reservation, 3)?;
        let compiler = Self::compiler_reservation(metadata, live)?;
        let floor = add(compiler, mul(cache, 3)?)?;
        let per_role = Self::available_per_role(metadata, live)?;
        if work > limits.max_work || copies > limits.max_public_copy_bytes
            || floor > pllm_garble::fusefss_backend::MAX_ALLOCATION_BYTES
            || cache > per_role
        {
            return Err(Error::Resource);
        }
        factory(Resources {
            work,
            metadata_bytes: metadata,
            live_array_bytes_per_party: live,
            source_table_cache_bytes_per_role: cache,
            public_copy_bytes: copies,
            // A bootstrap floor is deliberately not the full native peak.
            ..Resources::default()
        })
    }

    // Mutations and early copy-limit checks deliberately precede runtime's
    // cache-delta validation, just as before extraction. Failure is not rollback.
    pub(super) fn begin(
        resources: &'a mut Resources,
        limits: Limits,
        capacity: Capacity,
        native: composition::Resources,
        wave: Option<wavefront::Resources>,
        inputs: usize,
        cached: bool,
    ) -> Result<Self, Error> {
        let mut quote = Self { resources, limits, capacity, native, wave, inputs, cached };
        let r = &mut *quote.resources;
        r.kernels = add(r.kernels, 1)?;
        r.primitive_requests_per_party = add(r.primitive_requests_per_party, native.secret_steps)?;
        r.work = add(r.work, mul(native.cumulative_work, 2)?)?;
        if native.template_copy_bytes != 0 {
            let copies = mul(native.template_copy_bytes, 3)?;
            quote.public_copy(copies)?;
            quote.resources.work = add(quote.resources.work, copies)?;
        }
        if let Some(wave) = wave {
            // Three roles, each with obligation-walk AND concrete-owner schedule.
            quote.resources.work = add(quote.resources.work, mul(wave.construction_work, 6)?)?;
            quote.public_copy(mul(wave.metadata_bytes, 6)?)?;
        }
        Ok(quote)
    }

    fn public_copy(&mut self, bytes: usize) -> Result<(), Error> {
        self.resources.public_copy_bytes = add(self.resources.public_copy_bytes, bytes)?;
        if self.resources.public_copy_bytes > self.limits.max_public_copy_bytes {
            return Err(Error::Resource);
        }
        Ok(())
    }

    // Deltas are validated by runtime against its accounted_tables snapshot.
    // lowered_metadata is already multiplied by three there, before the cache
    // context checks; preserve that validation/error ordering and do not reprice.
    pub(super) fn tables(
        &mut self,
        tables: cache::Stats,
        references: usize,
        misses: usize,
        retained: usize,
        copied: usize,
        lowered_metadata: usize,
    ) -> Result<(), Error> {
        let r = self.native;
        let table_lowering = if self.cached {
            let chunks = r.public_table_reservation_bytes.div_ceil(256 * 8);
            let per_reference = add(4096, mul(tables.retained_entries, 64)?)?;
            add(add(mul(retained, 48)?, mul(mul(references, per_reference)?, 3)?)?,
                mul(chunks, 192)?)?
        } else {
            mul(r.public_table_reservation_bytes, 48)?
        };
        let lowering = add(add(add(table_lowering, mul(self.inputs, 32)?)?,
            mul(r.instructions, 4)?)?, 1024)?;
        self.resources.work = add(self.resources.work, lowering)?;
        self.resources.work = add(self.resources.work, lowered_metadata)?;
        // Miss allowance covers raw identity metadata only. Chunk construction
        // metadata, payload copies, and each statistics commitment remain priced.
        let metadata_copies = add(add(add(mul(references, 24)?, mul(misses, 512)?)?,
            lowered_metadata)?, mul(3, std::mem::size_of::<cache::Stats>())?)?;
        self.resources.public_copy_bytes = add(self.resources.public_copy_bytes,
            add(mul(copied, 3)?, metadata_copies)?)?;
        Ok(())
    }

    // Runtime hashes table counters and advances accounted_tables BEFORE this
    // phase, and hashes selection/node/Plan only AFTER this phase succeeds.
    pub(super) fn finish(self, tables: cache::Stats) -> Result<(), Error> {
        let r = self.native;
        let resources = self.resources;
        resources.source_table_cache_bytes_per_role =
            add(tables.retained_bytes, tables.metadata_reservation_bytes)?;
        resources.material_bytes = add(resources.material_bytes, r.private_material_bytes_total)?;
        let frames = match self.wave {
            Some(wave) => mul(2, wave.frame_bytes_per_party)?,
            None => r.frame_bytes_total,
        };
        resources.frame_bytes = add(resources.frame_bytes, frames)?;
        // Cached handle writes occur at issuance AND receiving-role import.
        resources.public_copy_bytes = add(resources.public_copy_bytes,
            mul(r.public_table_copy_bytes_total, if self.cached { 2 } else { 1 })?)?;
        // Independent worker Plans and dealer Plan: combined already contains
        // one Plan, both registers, paired issuance and retained frame overlap.
        let wave_extra = self.wave.map_or(0, |w| w.additional_reservation_bytes);
        let native = add(add(r.combined_reservation_bytes, wave_extra)?,
            mul(r.plan_reservation_bytes, 2)?)?;
        let native = add(native.max(mul(3, r.template_construction_bytes)?),
            mul(add(tables.retained_bytes, tables.metadata_reservation_bytes)?, 3)?)?;
        resources.peak_native_bytes = resources.peak_native_bytes.max(native);
        // Full request cache PLUS referenced Plan table reservations: no sharing
        // discount; miss-time generated-value/Table overlap remains covered.
        resources.combined_reservation_bytes = add(
            add(mul(3, resources.metadata_bytes)?, mul(2, resources.live_array_bytes_per_party)?)?,
            resources.peak_native_bytes,
        )?;
        let l = self.limits;
        let kernel_limit = match self.capacity {
            Capacity::Capsule => 32768,
            Capacity::Decoder => 1 << 22,
        };
        if resources.kernels > kernel_limit
            || resources.work > l.max_work
            || resources.material_bytes > l.max_material_bytes
            || resources.frame_bytes > l.max_frame_bytes
            || resources.public_copy_bytes > l.max_public_copy_bytes
            || resources.combined_reservation_bytes
                > pllm_garble::fusefss_backend::MAX_ALLOCATION_BYTES
        {
            return Err(Error::Resource);
        }
        Ok(())
    }
}
