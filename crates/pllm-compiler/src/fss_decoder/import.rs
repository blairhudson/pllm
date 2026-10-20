//! Ordered, bounded public checkpoint ingestion. Source geometry is lowered before
//! any payload is accepted; malformed input consumes the complete pending owner.
//! The caller authenticates checkpoint files and prices its read buffers separately.
use super::source::TensorImport;
use super::{
    sdk, vector, Capacity, Error, Limits, Program, RotaryCoefficients, Source, Tensor, WeightDtype,
};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

pub const MAX_FRAME_BYTES: usize = 64 * 1024;

pub struct SourceBuilder {
    pending: Option<Pending>,
}

struct Pending {
    admitted: sdk::SourcePreflight,
    outputs: usize,
    tensors: Vec<Tensor>,
    rotary: Vec<RotaryCoefficients>,
    tensor: Option<TensorImport>,
    sine: Zeroizing<Vec<f32>>,
    cosine: Zeroizing<Vec<f32>>,
    dtype: Option<WeightDtype>,
    input_hash: Sha256,
}

/// Both encoded source and original IEEE stream identities remain available.
/// Equal Q12 codes do not authenticate distinct external checkpoint bytes.
pub struct ImportedProgram {
    pub program: Program,
    pub input_digest: [u8; 32],
    pub execution_binding: sdk::SourceExecutionBinding,
    pub ownership: CompileOwnership,
}

/// Source-only completion. This is not complete-program or transport admission.
/// The immutable binding and actual certificates were minted by this importer.
pub struct ImportedSource {
    pub source: Source,
    pub plan: pllm_models::DecoderPlan,
    pub input_digest: [u8; 32],
    pub execution_binding: sdk::SourceExecutionBinding,
    pub phases: sdk::SourcePhaseAdmission,
    pub outputs: usize,
    pub numeric_profile: &'static str,
}

/// Logical phase/overlap proof. The schedule fields partition the *existing*
/// Program 3*metadata allowance; they are not a second reservation or RSS claim.
#[derive(Clone, Copy, Debug)]
pub struct CompileOwnership {
    pub phases: sdk::SourcePhaseAdmission,
    pub schedule_metadata_credit_floor: usize,
    pub schedule_retained_owner_bound: usize,
}

fn compile_ownership(
    plan: &pllm_models::DecoderPlan,
    source: &Source,
    phases: sdk::SourcePhaseAdmission,
) -> Result<CompileOwnership, Error> {
    let floor =
        sdk::compile_credit_floor(plan, source.reservation_bytes()).map_err(|_| Error::Resource)?;
    let credit = floor.checked_mul(3).ok_or(Error::Resource)?;
    let proofs = if source.capacity() == Capacity::Decoder || source.checked_ranges() {
        super::energy::reservation_bound(plan.prefill.operations.len())?
    } else {
        0
    };
    let program = source.reservation_bytes()
        + proofs
        + std::mem::size_of::<Program>()
        + plan.prefill.operations.len() * (std::mem::size_of::<super::program::Node>() + 4 * 8);
    let retained = program
        .checked_add(phases.compile_external_bytes)
        .ok_or(Error::Resource)?;
    // Inventories have retired. Binding uses fixed-size digests, never a second
    // retained plan. ids/originals retire in Program::compile before its schedule;
    // canonical/validation copies belong only to the separately priced prepare
    // phase. Party arrays and primitive/cache owners stay charged by core.
    if retained > credit
        || phases.compile_prepare_peak_bound > pllm_garble::fusefss_backend::MAX_ALLOCATION_BYTES
    {
        return Err(Error::Resource);
    }
    Ok(CompileOwnership {
        phases,
        schedule_metadata_credit_floor: floor,
        schedule_retained_owner_bound: retained,
    })
}

fn elements(shape: &[usize]) -> Result<usize, Error> {
    shape.iter().try_fold(1usize, |n, &d| {
        n.checked_mul(d)
            .filter(|&v| v <= super::MAX_VALUES)
            .ok_or(Error::Resource)
    })
}

fn dtype_id(dtype: WeightDtype) -> u8 {
    match dtype {
        WeightDtype::F32 => 0,
        WeightDtype::F16 => 1,
        WeightDtype::Bf16 => 2,
    }
}

impl SourceBuilder {
    pub fn new(
        config: &[u8],
        rows: usize,
        outputs: usize,
        numeric_profile: &str,
    ) -> Result<Self, String> {
        Self::new_with_capacity(config, rows, outputs, numeric_profile, Capacity::Capsule)
    }
    pub fn new_with_capacity(
        config: &[u8],
        rows: usize,
        outputs: usize,
        numeric_profile: &str,
        capacity: Capacity,
    ) -> Result<Self, String> {
        let admitted =
            sdk::preflight_source_with_capacity(config, rows, outputs, numeric_profile, capacity)?;
        let tensors = vector(admitted.source_shapes.len())
            .map_err(|_| "FuseFSS source tensor-owner allocation rejected".to_string())?;
        let rotary = vector(admitted.rotary_shapes.len())
            .map_err(|_| "FuseFSS rotary-owner allocation rejected".to_string())?;
        let mut input_hash = Sha256::new();
        input_hash.update(b"pllm.fss.original-ieee-stream.v1\0");
        Ok(Self {
            pending: Some(Pending {
                admitted,
                outputs,
                tensors,
                rotary,
                tensor: None,
                sine: Zeroizing::new(Vec::new()),
                cosine: Zeroizing::new(Vec::new()),
                dtype: None,
                input_hash,
            }),
        })
    }

    pub fn admitted(&self) -> Result<&sdk::SourcePreflight, Error> {
        self.pending
            .as_ref()
            .map(|p| &p.admitted)
            .ok_or(Error::Consumed)
    }

    /// `offset` counts source elements; payload is canonical little-endian IEEE.
    /// Frame boundaries do not affect the original-byte commitment.
    pub fn append(
        &mut self,
        name: &str,
        dtype: WeightDtype,
        offset: usize,
        payload: &[u8],
    ) -> Result<(), Error> {
        let mut pending = self.pending.take().ok_or(Error::Consumed)?;
        pending.append(name, dtype, offset, payload)?;
        self.pending = Some(pending);
        Ok(())
    }

    /// Pinned public F32 rotary tables use the same ordered element cursor for
    /// cosine and sine. Theta and geometry come from native source lowering.
    pub fn append_rotary(
        &mut self,
        operation: &str,
        offset: usize,
        cosine: &[u8],
        sine: &[u8],
    ) -> Result<(), Error> {
        let mut pending = self.pending.take().ok_or(Error::Consumed)?;
        pending.append_rotary(operation, offset, cosine, sine)?;
        self.pending = Some(pending);
        Ok(())
    }

    pub fn seal_source(mut self) -> Result<ImportedSource, Error> {
        let pending = self.pending.take().ok_or(Error::Consumed)?;
        if pending.tensors.len() != pending.admitted.source_shapes.len()
            || pending.rotary.len() != pending.admitted.rotary_shapes.len()
            || pending.tensor.is_some()
            || !pending.cosine.is_empty()
            || !pending.sine.is_empty()
            || pending.dtype.is_some()
        {
            return Err(Error::Context);
        }
        let input_digest = pending.input_hash.finalize().into();
        let execution_binding =
            sdk::SourceExecutionBinding::new(&pending.admitted, pending.outputs)
                .map_err(|_| Error::Resource)?;
        let mut source = Source::new_with_capacity(pending.tensors, pending.admitted.capacity)?;
        if pending.admitted.numeric_profile == sdk::DENSE_ROPE_CHECKED_NUMERIC {
            source = source.with_checked_ranges();
        }
        if !pending.rotary.is_empty() {
            source = source.with_rotary(pending.rotary)?;
        }
        let plan = pending.admitted.plan;
        // Only the plan and fixed-size execution commitment cross compile.
        // Ingestion inventories are not native Program inputs and must retire.
        drop(pending.admitted.source_shapes);
        drop(pending.admitted.source_layouts);
        drop(pending.admitted.rotary_shapes);
        drop(pending.admitted.source_config_sha256);
        Ok(ImportedSource {
            source,
            plan,
            input_digest,
            execution_binding,
            phases: pending.admitted.phase_admission,
            outputs: pending.outputs,
            numeric_profile: pending.admitted.numeric_profile,
        })
    }

    pub fn seal(self) -> Result<ImportedProgram, Error> {
        self.seal_with_schedule(super::OnlineSchedule::Serial)
    }

    pub fn seal_with_schedule(
        self,
        online: super::OnlineSchedule,
    ) -> Result<ImportedProgram, Error> {
        let imported = self.seal_source()?;
        let ownership = compile_ownership(&imported.plan, &imported.source, imported.phases)?;
        let limits = Limits::for_capacity(imported.source.capacity());
        let program = Program::compile_with_schedule(
            &imported.plan,
            imported.source,
            imported.outputs,
            limits,
            online,
        )?;
        // Compatibility assertion, not a late primary gate: the pre-call floor
        // above follows the current core formula and is tested against resources.
        if program.resources().metadata_bytes < ownership.schedule_metadata_credit_floor {
            return Err(Error::Context);
        }
        if program.profile() != imported.numeric_profile {
            return Err(Error::Context);
        }
        Ok(ImportedProgram {
            program,
            input_digest: imported.input_digest,
            execution_binding: imported.execution_binding,
            ownership,
        })
    }

    pub fn cancel(&mut self) {
        self.pending.take();
    }
}

impl Pending {
    fn append(
        &mut self,
        name: &str,
        dtype: WeightDtype,
        offset: usize,
        payload: &[u8],
    ) -> Result<(), Error> {
        let (expected, shape) = self
            .admitted
            .source_shapes
            .iter()
            .nth(self.tensors.len())
            .ok_or(Error::Context)?;
        let word_bytes = if dtype_id(dtype) == 0 { 4 } else { 2 };
        let length = self.admitted.capacity.source_elements(shape)?;
        if name != expected
            || payload.is_empty()
            || payload.len() > MAX_FRAME_BYTES
            || payload.len() % word_bytes != 0
            || offset != self.tensor.as_ref().map_or(0, TensorImport::written)
            || offset
                .checked_add(payload.len() / word_bytes)
                .is_none_or(|end| end > length)
            || self
                .dtype
                .is_some_and(|previous| dtype_id(previous) != dtype_id(dtype))
        {
            return Err(Error::Context);
        }
        if self.dtype.is_none() {
            self.tensor = Some(TensorImport::new(name, shape, self.admitted.capacity)?);
            self.dtype = Some(dtype);
            self.input_hash.update((name.len() as u64).to_le_bytes());
            self.input_hash.update(name.as_bytes());
            self.input_hash.update([dtype_id(dtype)]);
            self.input_hash.update((shape.len() as u64).to_le_bytes());
            for &dimension in shape {
                self.input_hash.update((dimension as u64).to_le_bytes());
            }
        }
        let tensor = self.tensor.as_mut().ok_or(Error::Context)?;
        tensor.append(dtype, payload)?;
        self.input_hash.update(payload);
        if tensor.complete() {
            self.tensors
                .push(self.tensor.take().ok_or(Error::Context)?.finish()?);
            self.dtype = None;
        }
        Ok(())
    }

    fn append_rotary(
        &mut self,
        operation: &str,
        offset: usize,
        cosine: &[u8],
        sine: &[u8],
    ) -> Result<(), Error> {
        let (expected, shape) = self
            .admitted
            .rotary_shapes
            .iter()
            .nth(self.rotary.len())
            .ok_or(Error::Context)?;
        let length = elements(shape)?;
        if self.tensors.len() != self.admitted.source_shapes.len()
            || self.dtype.is_some()
            || operation != expected
            || cosine.is_empty()
            || cosine.len() > MAX_FRAME_BYTES
            || cosine.len() != sine.len()
            || cosine.len() % 4 != 0
            || offset != self.cosine.len()
            || offset
                .checked_add(cosine.len() / 4)
                .is_none_or(|end| end > length)
        {
            return Err(Error::Context);
        }
        let op = self
            .admitted
            .plan
            .prefill
            .operations
            .iter()
            .find(|op| op.id == operation)
            .ok_or(Error::Context)?;
        let theta = op.attributes["theta"].as_u64().ok_or(Error::Domain)?;
        if offset == 0 {
            self.cosine = Zeroizing::new(vector(length)?);
            self.sine = Zeroizing::new(vector(length)?);
            self.input_hash
                .update((operation.len() as u64).to_le_bytes());
            self.input_hash.update(operation.as_bytes());
            self.input_hash.update(theta.to_le_bytes());
            for &dimension in shape {
                self.input_hash.update((dimension as u64).to_le_bytes());
            }
        }
        for (c, s) in cosine.chunks_exact(4).zip(sine.chunks_exact(4)) {
            let c = f32::from_le_bytes(c.try_into().map_err(|_| Error::Context)?);
            let s = f32::from_le_bytes(s.try_into().map_err(|_| Error::Context)?);
            let index = self.cosine.len();
            let width = shape[1];
            if !c.is_finite()
                || !s.is_finite()
                || c.abs() > 1.0
                || s.abs() > 1.0
                || (index < width
                    && (c.to_bits() != 1f32.to_bits() || s.to_bits() != 0f32.to_bits()))
                || (index % width >= width / 2
                    && (c.to_bits() != self.cosine[index - width / 2].to_bits()
                        || s.to_bits() != self.sine[index - width / 2].to_bits()))
            {
                return Err(Error::Domain);
            }
            self.cosine.push(c);
            self.sine.push(s);
            // Interleaving preserves identity across different chunk boundaries.
            self.input_hash.update(c.to_le_bytes());
            self.input_hash.update(s.to_le_bytes());
        }
        if self.cosine.len() == length {
            let shape: [usize; 2] = shape.as_slice().try_into().map_err(|_| Error::Domain)?;
            self.rotary.push(RotaryCoefficients::from_f32(
                operation,
                theta,
                shape,
                &self.cosine,
                &self.sine,
            )?);
            self.cosine = Zeroizing::new(Vec::new());
            self.sine = Zeroizing::new(Vec::new());
        }
        Ok(())
    }
}

/// Domain-only validation, without changing Tensor::from_ieee conversion order.
/// All finite F16 values fit signed50/Q12. For F32/BF16 the representable spacing
/// next to 2^37 is much greater than half a Q12 unit: positive 2^37 is excluded,
/// negative -2^37 included. Integer bit comparison also excludes NaN/infinity.
#[cfg(test)]
fn validate_word(dtype: WeightDtype, bits: u32) -> Result<(), Error> {
    let full = match dtype {
        WeightDtype::F32 => bits,
        WeightDtype::Bf16 if bits <= 65535 => bits << 16,
        WeightDtype::F16 if bits <= 65535 && bits & 0x7c00 != 0x7c00 => return Ok(()),
        _ => return Err(Error::Domain),
    };
    let magnitude = full & 0x7fff_ffff;
    let boundary = (127u32 + 37) << 23;
    if magnitude > boundary || (magnitude == boundary && full >> 31 == 0) {
        return Err(Error::Domain);
    }
    Ok(())
}

#[cfg(test)]
#[path = "import_tests.rs"]
mod tests;
