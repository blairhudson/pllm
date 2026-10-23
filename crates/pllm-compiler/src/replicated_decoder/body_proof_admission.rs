//! Owner-local container, post-construction template, proof and finalization
//! admission, not whole-Builder or whole-request admission. Native Program and
//! other family constructors may allocate before reaching these boundaries.
//! Census deliberately overcounts shared layouts/sources; never refunds owners.
use super::*;

fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Bounds)
}
fn array<T>(n: usize) -> Result<usize, Error> {
    n.checked_mul(size_of::<T>()).ok_or(Error::Bounds)
}
// Conservative allocator/node allowance, matching the body's existing policy.
// A B-tree node holds at most 11 entries/12 edges. Charge one full node per
// entry, including empty-map/root and allocator/header allowance separately.
fn tree_entry<K, V>() -> Result<usize, Error> {
    add(array::<(K, V)>(11)?, add(array::<usize>(16)?, 128)?)
}
fn layout_bytes(layout: &Layout) -> Result<usize, Error> {
    add(layout.metadata_bytes(), 64)
}
fn growth_capacity(len: usize, capacity: usize) -> Result<usize, Error> {
    if len >= MAX_VALUES || capacity > MAX_VALUES || len > capacity {
        return Err(Error::Bounds);
    }
    if len < capacity {
        Ok(0)
    } else {
        Ok(capacity
            .checked_mul(2)
            .ok_or(Error::Bounds)?
            .clamp(4, MAX_VALUES))
    }
}
fn owner_growth_capacity(len: usize, capacity: usize) -> Result<usize, Error> {
    if len >= MAX_NODES || capacity > MAX_NODES || len > capacity {
        return Err(Error::Bounds);
    }
    if len < capacity {
        Ok(0)
    } else {
        Ok(capacity
            .checked_mul(2)
            .ok_or(Error::Bounds)?
            .clamp(4, MAX_NODES))
    }
}

/// Closed destination import, not a row coordinate or numeric certificate.
/// The mutable borrow pins the ACTUAL Builder; moving/replacing a Builder or
/// selecting an equal-looking consumer cannot substitute another destination.
///
/// No production issuer exists yet. bind_complete_row_import deliberately
/// returns Binding, including for a structurally valid, fully admitted request.
/// Only a compiler-owned compact import/export mapping may eventually issue
/// this value. In particular ConsumerObligation metadata is not such authority.
pub(in crate::replicated_decoder) struct AuthenticatedRowEdge<'a> {
    destination: &'a mut Builder,
    original_producer: &'a Arc<Plan>,
    producer: &'a Arc<Plan>,
    export: usize,
    start: usize,
    source_row: &'a Layout,
    input_row: &'a Layout,
    retained_baseline: usize,
}

impl Plan {
    /// Borrow the exact native program owning a complete certified Tensor row.
    /// Returning an Arc reference preserves actual ownership, not digest reuse.
    pub(in crate::replicated_decoder) fn complete_row_prepared_owner(
        &self,
        row: &Layout,
    ) -> Result<&Arc<tensor::Program>, Error> {
        let native = self.complete_row(row).ok_or(Error::Binding)?;
        let key = native.key();
        let op = key.owner.checked_sub(self.inputs).ok_or(Error::Binding)?;
        let Op::Tensor {
            stride, programs, ..
        } = self.ops.get(op).ok_or(Error::Binding)?
        else {
            return Err(Error::Binding);
        };
        if *stride == 0 || key.width != *stride || key.offset % *stride != 0 {
            return Err(Error::Binding);
        }
        programs.get(key.offset / *stride).ok_or(Error::Binding)
    }

    /// Check the ACTUAL native producer of this complete row, not a digest or
    /// an independently prepared replacement. This is a read-only necessary
    /// check; it does not authenticate any consumer or runtime value owner.
    /// Only a single complete native Tensor row is supported. General residual,
    /// source-only and attention-only coordinates cannot name a prepared owner.
    pub(in crate::replicated_decoder) fn check_original_prepared_row_owner(
        &self,
        row: &Layout,
        original: &Arc<tensor::Program>,
        candidate: &Arc<tensor::Program>,
    ) -> Result<(), Error> {
        if !Arc::ptr_eq(original, candidate) {
            return Err(Error::Binding);
        }
        let actual = self.complete_row_prepared_owner(row)?;
        if !Arc::ptr_eq(actual, original) {
            return Err(Error::Binding);
        }
        Ok(())
    }
}

impl Builder {
    pub(super) fn input_metadata_bytes(count: usize) -> Result<usize, Error> {
        let mut bytes = add(size_of::<Self>(), 512)?;
        bytes = add(bytes, array::<Register>(count)?)?;
        bytes = add(bytes, array::<Box<[Region]>>(count)?)?;
        add(
            bytes,
            count
                .checked_mul(add(size_of::<Region>(), 64)?)
                .ok_or(Error::Bounds)?,
        )
    }
    fn push_storage_bytes(
        &self,
        op: &Op,
        regions: &Vec<Region>,
        temporary: usize,
    ) -> Result<usize, Error> {
        // Incoming Op/Region owners are not yet in the live census. Price them
        // as well as the boxed destination, all replacement Vecs and output Arc.
        let mut extra = add(self.proof_op_metadata(op)?, temporary)?;
        extra = add(extra, add(array::<Region>(regions.capacity())?, 64)?)?;
        extra = add(extra, add(array::<Region>(regions.len())?, 64)?)?;
        // A contiguous output has one span; every validated input has >=1.
        extra = add(
            extra,
            Self::proof_row_scratch(op.first_layout().ok_or(Error::Shape)?)?,
        )?;
        for n in [
            array::<Register>(owner_growth_capacity(
                self.registers.len(),
                self.registers.capacity(),
            )?)?,
            array::<Box<[Region]>>(owner_growth_capacity(
                self.regions.len(),
                self.regions.capacity(),
            )?)?,
            array::<Op>(owner_growth_capacity(self.ops.len(), self.ops.capacity())?)?,
        ] {
            extra = add(extra, add(n, 64)?)?;
        }
        Ok(extra)
    }
    pub(super) fn push_storage_admitted(
        &mut self,
        op: &Op,
        regions: &Vec<Region>,
        temporary: usize,
    ) -> Result<(), Error> {
        let extra = self.push_storage_bytes(op, regions, temporary)?;
        self.proof_preflight(extra)?;
        #[cfg(test)]
        if self.fail_builder_allocation {
            return Err(Error::Allocation);
        }
        let rc = owner_growth_capacity(self.registers.len(), self.registers.capacity())?;
        let gc = owner_growth_capacity(self.regions.len(), self.regions.capacity())?;
        let oc = owner_growth_capacity(self.ops.len(), self.ops.capacity())?;
        let mut registers: Vec<Register> = Vec::new();
        let mut region_owners: Vec<Box<[Region]>> = Vec::new();
        let mut ops: Vec<Op> = Vec::new();
        // Allocate every replacement before moving any existing owner. A fault
        // leaves all original lengths/capacities intact; caller retires Builder.
        registers
            .try_reserve_exact(rc)
            .map_err(|_| Error::Allocation)?;
        region_owners
            .try_reserve_exact(gc)
            .map_err(|_| Error::Allocation)?;
        ops.try_reserve_exact(oc).map_err(|_| Error::Allocation)?;
        if registers.capacity() != rc || region_owners.capacity() != gc || ops.capacity() != oc {
            return Err(Error::Bounds);
        }
        if rc != 0 {
            registers.append(&mut self.registers);
            self.registers = registers;
        }
        if gc != 0 {
            region_owners.append(&mut self.regions);
            self.regions = region_owners;
        }
        if oc != 0 {
            ops.append(&mut self.ops);
            self.ops = ops;
        }
        Ok(())
    }
    fn template_insert_bytes(
        &self,
        digest: &[u8; 32],
        resources: tensor::Resources,
        temporary: usize,
    ) -> Result<usize, Error> {
        let mut extra = add(temporary, add(resources.immutable_template_bytes, 64)?)?;
        if !self.templates.contains_key(digest) {
            // Owner's incoming Program quote plus the Arc destination coexist.
            // Full post-insertion map envelopes cover root/path splits while
            // both old trees remain live; no expected-amortized node allowance.
            extra = add(extra, add(resources.immutable_template_bytes, 64)?)?;
            extra = add(
                extra,
                add(self.templates.len(), 1)?
                    .checked_mul(tree_entry::<[u8; 32], Arc<tensor::Program>>()?)
                    .ok_or(Error::Bounds)?,
            )?;
            extra = add(
                extra,
                add(self.template_metadata.len(), 1)?
                    .checked_mul(tree_entry::<[u8; 32], usize>()?)
                    .ok_or(Error::Bounds)?,
            )?;
        }
        Ok(extra)
    }
    pub(super) fn template_insert_preflight(
        &mut self,
        digest: &[u8; 32],
        resources: tensor::Resources,
        temporary: usize,
    ) -> Result<(), Error> {
        let extra = self.template_insert_bytes(digest, resources, temporary)?;
        self.proof_preflight(extra)
    }
    pub(super) fn template_call_temporary(
        input: &Layout,
        programs: &Vec<Arc<tensor::Program>>,
        other: usize,
    ) -> Result<usize, Error> {
        add(
            add(
                array::<Arc<tensor::Program>>(programs.capacity())?,
                Self::proof_row_scratch(input)?,
            )?,
            other,
        )
    }
    pub(super) fn new_template_list(
        &mut self,
        input: &Layout,
        count: usize,
        temporary: usize,
    ) -> Result<Vec<Arc<tensor::Program>>, Error> {
        self.proof_preflight(add(
            add(
                array::<Arc<tensor::Program>>(count)?,
                Self::proof_row_scratch(input)?,
            )?,
            temporary,
        )?)?;
        let mut programs = Vec::new();
        if programs.try_reserve_exact(count).is_err() {
            self.proof_admission_retired = true;
            return Err(Error::Allocation);
        }
        if programs.capacity() != count {
            self.proof_admission_retired = true;
            return Err(Error::Bounds);
        }
        Ok(programs)
    }
    pub(super) fn region_buffer(
        &mut self,
        count: usize,
        temporary: usize,
    ) -> Result<Vec<Region>, Error> {
        self.proof_preflight(add(add(array::<Region>(count)?, 64)?, temporary)?)?;
        let mut regions = Vec::new();
        if regions.try_reserve_exact(count).is_err() {
            self.proof_admission_retired = true;
            return Err(Error::Allocation);
        }
        if regions.capacity() != count {
            self.proof_admission_retired = true;
            return Err(Error::Bounds);
        }
        Ok(regions)
    }
    pub(super) fn matrix_row_buffer(
        &mut self,
        left: &Layout,
        right: &Layout,
        count: usize,
    ) -> Result<Vec<MatrixRow>, Error> {
        let slices = add(
            Self::proof_row_scratch(left)?,
            Self::proof_row_scratch(right)?,
        )?;
        let extra = add(
            array::<MatrixRow>(count)?,
            count.checked_mul(slices).ok_or(Error::Bounds)?,
        )?;
        self.proof_preflight(extra)?;
        let mut rows = Vec::new();
        if rows.try_reserve_exact(count).is_err() {
            self.proof_admission_retired = true;
            return Err(Error::Allocation);
        }
        if rows.capacity() != count {
            self.proof_admission_retired = true;
            return Err(Error::Bounds);
        }
        Ok(rows)
    }
    pub(super) fn matrix_construction_bytes(
        rows: &Vec<MatrixRow>,
        tiles: usize,
    ) -> Result<usize, Error> {
        let mut bytes = add(array::<MatrixRow>(rows.capacity())?, 64)?;
        let mut scratch = 0;
        for row in rows {
            bytes = add(
                bytes,
                add(layout_bytes(&row.left)?, layout_bytes(&row.right)?)?,
            )?;
            bytes = add(bytes, add(array::<InputBounds>(row.bounds.len())?, 64)?)?;
            scratch = scratch.max(add(
                Self::proof_row_scratch(&row.left)?,
                Self::proof_row_scratch(&row.right)?,
            )?);
        }
        // All bound destinations coexist with per-row Vec/boxing overlap.
        bytes = add(
            bytes,
            array::<InputBounds>(tiles.checked_mul(2).ok_or(Error::Bounds)?)?,
        )?;
        bytes = add(bytes, array::<MatrixRow>(rows.len())?)?;
        add(
            add(bytes, scratch)?,
            rows.len().checked_mul(192).ok_or(Error::Bounds)?,
        )
    }
    pub(super) fn proof_admission_live(&self) -> Result<(), Error> {
        if self.proof_admission_retired {
            Err(Error::Binding)
        } else {
            Ok(())
        }
    }
    fn proof_limit(&self) -> usize {
        #[cfg(test)]
        return self.proof_metadata_limit.min(MAX_PROGRAM_BYTES);
        #[cfg(not(test))]
        MAX_PROGRAM_BYTES
    }
    /// No temporary collections: capacity, not length, for every live Vec.
    /// Repeated Arc references overcount backing intentionally.
    fn proof_live_metadata(&self) -> Result<usize, Error> {
        let mut bytes = add(size_of::<Self>(), 512)?;
        for n in [
            array::<Register>(self.registers.capacity())?,
            array::<Box<[Region]>>(self.regions.capacity())?,
            array::<Op>(self.ops.capacity())?,
            array::<Proof>(self.proofs.capacity())?,
        ] {
            bytes = add(bytes, n)?;
        }
        for regions in &self.regions {
            bytes = add(bytes, add(array::<Region>(regions.len())?, 64)?)?;
        }
        for op in &self.ops {
            bytes = add(bytes, self.proof_op_metadata(op)?)?;
        }
        for metadata in self.template_metadata.values() {
            bytes = add(bytes, add(*metadata, 64)?)?;
        }
        for _ in &self.templates {
            bytes = add(bytes, tree_entry::<[u8; 32], Arc<tensor::Program>>()?)?;
        }
        for _ in &self.template_metadata {
            bytes = add(bytes, tree_entry::<[u8; 32], usize>()?)?;
        }
        Ok(bytes)
    }
    fn proof_op_metadata(&self, op: &Op) -> Result<usize, Error> {
        let source = |s: &Source| add(add(size_of::<Source>(), s.key.capacity())?, 192);
        match op {
            Op::Binary { left, right, .. } => add(layout_bytes(left)?, layout_bytes(right)?),
            Op::PreparedTensorMap { input, prepared } => {
                add(layout_bytes(input)?, prepared.template_bytes()?)
            }
            Op::Tensor {
                input, programs, ..
            } => add(
                layout_bytes(input)?,
                add(array::<Arc<tensor::Program>>(programs.len())?, 64)?,
            ),
            Op::Matrix { rows, .. } => {
                let mut n = add(array::<MatrixRow>(rows.len())?, 64)?;
                for row in rows.iter() {
                    n = add(n, layout_bytes(&row.left)?)?;
                    n = add(n, layout_bytes(&row.right)?)?;
                    n = add(n, add(array::<InputBounds>(row.bounds.len())?, 64)?)?;
                }
                Ok(n)
            }
            Op::PublicMatrix {
                input, source: s, ..
            }
            | Op::Bias {
                input, source: s, ..
            } => add(layout_bytes(input)?, source(s)?),
            Op::Check {
                input,
                previous,
                checked,
                valid,
                ..
            } => {
                let mut n = layout_bytes(input)?;
                if let Some(p) = previous {
                    n = add(n, layout_bytes(p)?)?;
                }
                add(
                    n,
                    add(
                        array::<Arc<tensor::Program>>(add(checked.len(), valid.len())?)?,
                        128,
                    )?,
                )
            }
        }
    }
    pub(super) fn proof_preflight(&mut self, extra: usize) -> Result<(), Error> {
        self.proof_admission_live()?;
        let result = self.proof_live_metadata().and_then(|n| add(n, extra));
        match result {
            Ok(n) if n <= self.proof_limit() => Ok(()),
            _ => {
                self.proof_admission_retired = true;
                Err(Error::Bounds)
            }
        }
    }
    /// `temporary` accounts caller-owned producer buffers still live here.
    /// Replacing rather than reallocating makes old/new overlap explicit.
    pub(super) fn record_proof(&mut self, proof: Proof, temporary: usize) -> Result<(), Error> {
        let result = self.record_proof_inner(proof, temporary);
        if result.is_err() {
            self.proof_admission_retired = true;
        }
        result
    }
    /// The same capacity/overlap quote is used by native record_proof and the
    /// read-only external import quote. Existing live capacity is in the census.
    fn record_proof_growth(&self, temporary: usize) -> Result<(usize, usize), Error> {
        let capacity = growth_capacity(self.proofs.len(), self.proofs.capacity())?;
        let extra = add(temporary, add(array::<Proof>(capacity)?, 64)?)?;
        Ok((capacity, extra))
    }
    fn record_proof_inner(&mut self, proof: Proof, temporary: usize) -> Result<(), Error> {
        let (capacity, extra) = self.record_proof_growth(temporary)?;
        let grow = capacity != 0;
        self.proof_preflight(extra)?;
        if grow {
            #[cfg(test)]
            if self.fail_proof_allocation {
                self.proof_admission_retired = true;
                return Err(Error::Allocation);
            }
            let mut replacement = Vec::new();
            if replacement.try_reserve_exact(capacity).is_err() {
                self.proof_admission_retired = true;
                return Err(Error::Allocation);
            }
            // Vec's requested allocation is charged before reserve. Refuse an
            // implementation-reported excess capacity before publishing proof.
            if replacement.capacity() != capacity {
                self.proof_admission_retired = true;
                return Err(Error::Bounds);
            }
            replacement.extend_from_slice(&self.proofs);
            self.proofs = replacement;
        }
        self.proofs.push(proof);
        Ok(())
    }

    /// Necessary structural checks only, never destination dataflow authority.
    /// Lookup is in the native GENERAL row ledger. Source statistics, compact
    /// projection coordinates and attention-only bounds are not accepted here.
    #[allow(clippy::too_many_arguments)]
    fn checked_row_import(
        &self,
        original_producer: &Arc<Plan>,
        producer: &Arc<Plan>,
        export: usize,
        start: usize,
        source_row: &Layout,
        input_row: &Layout,
    ) -> Result<(CompleteRow, RowKey), Error> {
        self.proof_admission_live()?;
        if !Arc::ptr_eq(original_producer, producer) {
            return Err(Error::Binding);
        }
        let native = producer.complete_row(source_row).ok_or(Error::Binding)?;
        if native.producer() != producer.digest() {
            return Err(Error::Binding);
        }
        let output = producer.output_layout(export)?;
        if start.checked_add(source_row.len()).ok_or(Error::Bounds)? > output.len() {
            return Err(Error::Shape);
        }
        let input = input_row.row_key(source_row.len()).ok_or(Error::Binding)?;
        if input.owner >= self.inputs {
            return Err(Error::Binding);
        }
        input_row.validate(&self.registers)?;
        // Preserve the actual exported scalar hull, not caller-selected bounds.
        // Read the native region ledger without making row slices or buffers.
        let (mut lower, mut upper) = (i64::MAX, i64::MIN);
        for index in 0..source_row.len() {
            let word = source_row.word(index)?;
            if word != output.word(start + index)? {
                return Err(Error::Binding);
            }
            let regions = producer.regions.get(word.0).ok_or(Error::Binding)?;
            let slot = regions.partition_point(|r| r.offset <= word.1);
            let region = slot
                .checked_sub(1)
                .and_then(|slot| regions.get(slot))
                .filter(|r| word.1 - r.offset < r.len)
                .ok_or(Error::Binding)?;
            lower = lower.min(region.bounds.lower());
            upper = upper.max(region.bounds.upper());
        }
        if self.registers[input.owner].bounds != Interval::new(lower, upper)? {
            return Err(Error::Binding);
        }
        // Return the ORIGINAL producer obligation and a checked coordinate,
        // never a rebound destination Proof before consumer authentication.
        Ok((native, input))
    }

    fn row_import_owners(
        producer: &Arc<Plan>,
        source_row: &Layout,
        input_row: &Layout,
        retained_baseline: usize,
    ) -> Result<usize, Error> {
        [
            // Conservative full native admission, never a template-sharing credit.
            producer.resources().admitted_bytes,
            2 * size_of::<usize>(),
            source_row.backing_bytes()?,
            input_row.backing_bytes()?,
            2 * size_of::<Layout>(),
            2 * size_of::<AuthenticatedRowEdge<'_>>(),
            size_of::<Result<AuthenticatedRowEdge<'_>, Error>>(),
            2 * size_of::<Proof>(),
            size_of::<CompleteRow>(),
        ]
        .into_iter()
        .try_fold(retained_baseline, add)
    }

    /// Read-only, allocation-free TOTAL for current Builder live capacities,
    /// producer/query owners, temporary buffers and exact record_proof growth.
    /// A successful quote DOES NOT authenticate a producer-to-consumer edge.
    /// ADD enclosing original Program/Worker/import-map/source-file owners in
    /// retained_baseline; they are not implied by native Plan admitted bytes.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::replicated_decoder) fn complete_row_import_quote(
        &self,
        original_producer: &Arc<Plan>,
        producer: &Arc<Plan>,
        export: usize,
        start: usize,
        source_row: &Layout,
        input_row: &Layout,
        retained_baseline: usize,
        temporary: usize,
    ) -> Result<usize, Error> {
        self.checked_row_import(
            original_producer,
            producer,
            export,
            start,
            source_row,
            input_row,
        )?;
        let owners = Self::row_import_owners(producer, source_row, input_row, retained_baseline)?;
        let (_, extra) = self.record_proof_growth(add(owners, temporary)?)?;
        add(self.proof_live_metadata()?, extra)
    }

    /// Read-only admission under the SAME actual owner limit as record_proof.
    /// A checked copy may use this before retaining handles/allocating payloads;
    /// it still confers no executable consumer or certificate import authority.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::replicated_decoder) fn complete_row_import_preflight(
        &self,
        original_producer: &Arc<Plan>,
        producer: &Arc<Plan>,
        export: usize,
        start: usize,
        source_row: &Layout,
        input_row: &Layout,
        retained_baseline: usize,
        temporary: usize,
    ) -> Result<(), Error> {
        let quote = self.complete_row_import_quote(
            original_producer,
            producer,
            export,
            start,
            source_row,
            input_row,
            retained_baseline,
            temporary,
        )?;
        if quote > self.proof_limit() {
            Err(Error::Bounds)
        } else {
            Ok(())
        }
    }

    /// Fail-closed issuer until the MAIN EMITTER owns a closed executable import
    /// mapping. That mapping must prove: original producer Arc + exact export
    /// segment is copied, in order, into this exact Builder input row, under the
    /// unchanged original source/validity/lifecycle contract. Metadata alone,
    /// including a successful quote, cannot prove that copy will actually occur.
    /// No flags, callbacks, caller labels or test-only issuer bypass this gate.
    /// Original bridge exports and their real single-use value copies are owned
    /// by body_tensor_integration's compact_row_proof child. Even that custody
    /// cannot open this metadata-only constructor: its consuming bind_import
    /// still needs the main emitter's final executable destination/input owner.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::replicated_decoder) fn bind_complete_row_import<'a>(
        &'a mut self,
        original_producer: &'a Arc<Plan>,
        producer: &'a Arc<Plan>,
        export: usize,
        start: usize,
        source_row: &'a Layout,
        input_row: &'a Layout,
        retained_baseline: usize,
        temporary: usize,
    ) -> Result<AuthenticatedRowEdge<'a>, Error> {
        let quote = self.complete_row_import_quote(
            original_producer,
            producer,
            export,
            start,
            source_row,
            input_row,
            retained_baseline,
            temporary,
        )?;
        if quote > self.proof_limit() {
            return Err(Error::Bounds);
        }
        // Do NOT replace this with a struct literal from these arguments. The
        // emitter must first supply a closed binding derived from its actual
        // executable copy/import map, and this issuer must check that binding.
        Err(Error::Binding)
    }

    /// Consuming, destination-pinned publication. No production edge can be
    /// issued yet. Once the original emitter can authenticate the copy mapping,
    /// this path rechecks all binding/owner/capacity facts and uses the ordinary
    /// admitted record_proof growth. No separate attention/source ledger exists
    /// here, and neither can be substituted for the producer's native proof.
    pub(in crate::replicated_decoder) fn import_complete_row(
        edge: AuthenticatedRowEdge<'_>,
        temporary: usize,
    ) -> Result<(), Error> {
        let destination = edge.destination;
        let result = (|| {
            let (native, input) = destination.checked_row_import(
                edge.original_producer,
                edge.producer,
                edge.export,
                edge.start,
                edge.source_row,
                edge.input_row,
            )?;
            let owners = Self::row_import_owners(
                edge.producer,
                edge.source_row,
                edge.input_row,
                edge.retained_baseline,
            )?;
            destination.record_proof(
                Proof {
                    key: input,
                    energy: native.energy(),
                    mass: native.mass(),
                },
                add(owners, temporary)?,
            )
        })();
        if result.is_err() {
            destination.proof_admission_retired = true;
        }
        result
    }
    /// Bound slice Vec growth plus old/new Arc conversion overlap, including
    /// small-Vec minimum capacity. Used before proof producers make row slices.
    pub(super) fn proof_row_scratch(input: &Layout) -> Result<usize, Error> {
        add(
            input.metadata_bytes().checked_mul(8).ok_or(Error::Bounds)?,
            4096,
        )
    }
    pub(super) fn proof_producer_buffers(
        &mut self,
        input: &Layout,
        width: usize,
        programs: &mut Vec<Arc<tensor::Program>>,
        certificates: &mut Vec<(usize, Option<u128>, Option<u64>)>,
        enclosing_temporary: usize,
    ) -> Result<(), Error> {
        let rows = input.len() / width;
        let backing = add(
            array::<Arc<tensor::Program>>(rows)?,
            array::<(usize, Option<u128>, Option<u64>)>(rows)?,
        )?;
        self.proof_preflight(add(
            add(backing, Self::proof_row_scratch(input)?)?,
            enclosing_temporary,
        )?)?;
        if programs.try_reserve_exact(rows).is_err()
            || certificates.try_reserve_exact(rows).is_err()
        {
            self.proof_admission_retired = true;
            return Err(Error::Allocation);
        }
        if programs.capacity() != rows || certificates.capacity() != rows {
            self.proof_admission_retired = true;
            return Err(Error::Bounds);
        }
        Ok(())
    }
    pub(super) fn proof_producer_row_preflight(
        &mut self,
        input: &Layout,
        programs: &Vec<Arc<tensor::Program>>,
        certificates: &Vec<(usize, Option<u128>, Option<u64>)>,
        enclosing_temporary: usize,
    ) -> Result<(), Error> {
        let extra =
            Self::proof_producer_temporary(input, programs, certificates, enclosing_temporary)?;
        self.proof_preflight(extra)
    }
    pub(super) fn proof_producer_temporary(
        input: &Layout,
        programs: &Vec<Arc<tensor::Program>>,
        certificates: &Vec<(usize, Option<u128>, Option<u64>)>,
        enclosing_temporary: usize,
    ) -> Result<usize, Error> {
        let buffers = add(
            array::<Arc<tensor::Program>>(programs.capacity())?,
            array::<(usize, Option<u128>, Option<u64>)>(certificates.capacity())?,
        )?;
        // Interning earlier rows changes live template ownership; repeat census
        // before each new row slice rather than reusing the initial estimate.
        add(
            add(buffers, Self::proof_row_scratch(input)?)?,
            enclosing_temporary,
        )
    }
    /// Complete ordered row without allocating a temporary Layout::slice.
    pub(super) fn proof_row_key(layout: &Layout, start: usize, width: usize) -> Option<RowKey> {
        if width == 0 || start.checked_add(width)? > layout.len() {
            return None;
        }
        let (owner, offset) = layout.word(start).ok()?;
        for i in 1..width {
            if layout.word(start + i).ok()? != (owner, offset.checked_add(i)?) {
                return None;
            }
        }
        Some(RowKey {
            owner,
            offset,
            width,
        })
    }
    pub(super) fn proof_at(&self, layout: &Layout, start: usize, width: usize) -> Option<Proof> {
        let key = Self::proof_row_key(layout, start, width)?;
        self.proofs.iter().copied().find(|p| p.key == key)
    }
    /// Conservative preflight for all finish-only scratch and destination boxes.
    /// Includes simultaneous old capacity + new boxed payload, last-use table,
    /// retirement vectors/growth overlap, temporary op.layouts(), and B-tree sets.
    fn proof_finish_bytes(&self, outputs: &[Layout]) -> Result<usize, Error> {
        let mut extra = add(size_of::<Plan>(), 256)?;
        let count = self.registers.len();
        let mut layout_refs = outputs.len();
        let mut max_refs = 2;
        for op in &self.ops {
            let refs = match op {
                Op::Matrix { rows, .. } => rows.len().checked_mul(2).ok_or(Error::Bounds)?,
                Op::Binary { .. } | Op::Check { .. } => 2,
                _ => 1,
            };
            layout_refs = add(layout_refs, refs)?;
            max_refs = max_refs.max(refs);
        }
        for output in outputs {
            extra = add(extra, layout_bytes(output)?)?;
        }
        for n in [
            array::<Register>(count)?,
            array::<Box<[Region]>>(self.regions.len())?,
            array::<Op>(self.ops.len())?,
            array::<Proof>(self.proofs.len())?,
            array::<Layout>(outputs.len())?,
            // Capacity doubling, old/new growth, box conversion, and last/unused.
            array::<usize>(count.checked_mul(12).ok_or(Error::Bounds)?)?,
            array::<Vec<usize>>(self.ops.len().checked_mul(2).ok_or(Error::Bounds)?)?,
            array::<&Layout>(max_refs.checked_mul(4).ok_or(Error::Bounds)?)?,
            layout_refs
                .checked_mul(tree_entry::<usize, ()>()?)
                .ok_or(Error::Bounds)?,
            self.ops
                .len()
                .checked_mul(tree_entry::<usize, ()>()?)
                .ok_or(Error::Bounds)?,
            // Per retirement allocation, destination box and scratch allowance.
            count.checked_mul(512).ok_or(Error::Bounds)?,
        ] {
            extra = add(extra, n)?;
        }
        add(self.proof_live_metadata()?, extra)
    }

    /// Allocation-free FINALIZATION-phase quote, including the current live
    /// Builder capacity census and the SAME finish scratch/destination-overlap
    /// policy used by proof_finish_preflight. This is a total, not an increment
    /// to add to proof_live_metadata again. Output layouts are charged by that
    /// policy, even when their backing aliases a Builder-owned input layout.
    ///
    /// ADD `retained_baseline` for independently retained external owners:
    /// original Program/Worker/source owners, caller containers and layout
    /// handles/backing not already covered by the output-layout policy, and
    /// enclosing transition/result/control slots. No shared-template discount.
    /// This quote excludes earlier construction peaks, secret input/export
    /// copies, execution workspace, transport and later restoration. It does
    /// not admit the total; compose phase maxima or call the baseline preflight.
    /// Work/geometry validation mirrors Plan::finish before its allocations;
    /// outputs are validated without constructing op.layouts() or other Vecs.
    pub(in crate::replicated_decoder) fn proof_finish_quote(
        &self,
        outputs: &[Layout],
        retained_baseline: usize,
    ) -> Result<usize, Error> {
        self.proof_admission_live()?;
        self.work.limits()?;
        if self.advances > MAX_COORDINATOR_ADVANCES {
            return Err(Error::Bounds);
        }
        if self.ops.is_empty() || outputs.is_empty() || outputs.len() > MAX_NODES {
            return Err(Error::Shape);
        }
        outputs.iter().try_fold(0usize, |n, layout| {
            n.checked_add(layout.len())
                .filter(|&n| n <= MAX_VALUES)
                .ok_or(Error::Bounds)
        })?;
        for output in outputs {
            output.validate(&self.registers)?;
        }
        add(retained_baseline, self.proof_finish_bytes(outputs)?)
    }

    /// Read-only admission of external retained baseline + finalization total
    /// against the EXISTING owner limit (and unchanged MAX_PROGRAM_BYTES).
    /// Rejection neither allocates nor retires/mutates this borrowed Builder.
    pub(in crate::replicated_decoder) fn proof_finish_preflight_with_baseline(
        &self,
        outputs: &[Layout],
        retained_baseline: usize,
    ) -> Result<(), Error> {
        if self.proof_finish_quote(outputs, retained_baseline)? > self.proof_limit() {
            return Err(Error::Bounds);
        }
        Ok(())
    }

    pub(super) fn proof_finish_preflight(&self, outputs: &[Layout]) -> Result<(), Error> {
        self.proof_admission_live()?;
        if self.proof_finish_bytes(outputs)? > self.proof_limit() {
            return Err(Error::Bounds);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "body_proof_admission_tests.rs"]
mod tests;
