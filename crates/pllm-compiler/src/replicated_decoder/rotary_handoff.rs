//! Consuming original-row execution. No native factory, protocol translation,
//! primitive loan, second Worker, or production router-drain issuer.
use super::super::super::compact_lowering::{self, RotaryCreationPublication};
use super::*;
use crate::replicated_decoder::runtime::{BoundaryProgress, body};
use body::handoff::{SuspendedWorker, TRANSITION_INLINE_BYTES, owned_restore_inline_bytes};
use layer::{Builder as BodyBuilder, Executor, Layout, Plan, Values};

/// Kept closed exactly like source bias: only this module's actual three-role
/// router tests can construct it. Local readiness is never global custody.
pub(in crate::replicated_decoder) struct CoordinatorDrained<T> {
    owner: T,
}
struct RestoreOutputs {
    values: Values,
    #[cfg(test)]
    _erased: body::integration::restoration_tests::ErasureProbe,
}

pub(in crate::replicated_decoder) struct Compiled {
    plan: Arc<Plan>,
    publication: RotaryCreationPublication,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    resources: Resources,
    // Last: retained original schedule/source guards outlive all native handles.
    program: Arc<Program>,
}
#[derive(Clone, Copy, Debug)]
pub(in crate::replicated_decoder) struct Resources {
    pub retained_owner_bytes: usize,
    pub control_and_mapping_bytes: usize,
    pub import_and_restore_overlap_bytes: usize,
    pub construction_bytes: usize,
    pub execution_bytes: usize,
    pub admitted_bytes: usize,
}
fn admit(n: usize, cap: usize) -> Result<(), Error> {
    if n > cap.min(MAX_PROGRAM_BYTES) {
        Err(Error::Bounds)
    } else {
        Ok(())
    }
}

/// Named live control + map/Vec/Arc/copy population, not allocator/RSS estimates.
fn controls(program: &Program, owner: &Creation) -> Result<usize, Error> {
    let lanes = owner.input.len();
    let rows = owner.rows.len();
    let span = Layout::quote_affine(0, 0, &[1], &[1])?;
    let layouts = span
        .destination_backing_bytes
        .checked_mul(4)
        .and_then(|n| n.checked_mul(lanes))
        .ok_or(Error::Bounds)?;
    [
        bytes::<Compiled>(2)?,
        size_of::<Invocation>(),
        size_of::<Worker>(),
        size_of::<BodyBuilder>(),
        size_of::<Executor>(),
        size_of::<Plan>(),
        size_of::<Values>(),
        size_of::<RotaryCreationPublication>(),
        size_of::<Result<Invocation, Error>>(),
        size_of::<Result<Worker, Error>>(),
        size_of::<Result<Arc<Compiled>, Error>>(),
        size_of::<Result<(multiply::PartyRuntime, Values), Error>>(),
        size_of::<RestoreOutputs>(),
        owned_restore_inline_bytes::<RestoreOutputs>()?,
        TRANSITION_INLINE_BYTES,
        bytes::<Option<Secret>>(program.value_slot_count())?,
        // Worst case every mapped word has a distinct original live owner. Keep
        // whole imported owners, never refund unused lanes of their Vec backing.
        bytes::<usize>(lanes.checked_mul(2).ok_or(Error::Bounds)?)?,
        bytes::<body::Register>(lanes)?,
        bytes::<Secret>(lanes)?,
        bytes::<Layout>(lanes.checked_mul(2).ok_or(Error::Bounds)?)?,
        layouts,
        bytes::<usize>(rows.checked_mul(2).ok_or(Error::Bounds)?)?,
        bytes::<Arc<tensor::Program>>(rows.checked_mul(2).ok_or(Error::Bounds)?)?,
        bytes::<Vec<usize>>(2)?,
        size_of::<Vec<body::Register>>(),
        size_of::<Vec<Layout>>(),
        size_of::<Vec<Secret>>(),
        bytes::<usize>(4)?,
        puma::MAX_ADMITTED_BYTES,
        size_of::<[Option<puma::Message>; 2]>(),
    ]
    .into_iter()
    .try_fold(0, add)
}
impl Compiled {
    pub(in crate::replicated_decoder) fn admitted_bytes(&self) -> usize {
        self.resources.admitted_bytes
    }
    pub(in crate::replicated_decoder) fn prepare(
        program: Arc<Program>,
        owner: Creation,
        registry: &SourceRegistry<'_>,
        cap: usize,
    ) -> Result<Arc<Self>, Error> {
        owner.check(&program, registry)?;
        let custody = program.request_custody.as_ref().ok_or(Error::Binding)?;
        let retained_owner_bytes = custody.retained_bytes();
        let control_and_mapping_bytes = controls(&program, &owner)?;
        let baseline = add(retained_owner_bytes, control_and_mapping_bytes)?;
        // Count full import payload BEFORE map/row/layout/container allocation.
        let mut imported = 0;
        for (at, (id, _)) in owner.input.iter().enumerate() {
            if !owner.input.iter().take(at).any(|(prior, _)| prior == id) {
                imported = add(
                    imported,
                    bytes::<u64>(
                        program
                            .recipe_at(id)
                            .ok_or(Error::Binding)?
                            .lanes
                            .checked_mul(2)
                            .ok_or(Error::Bounds)?,
                    )?,
                )?;
            }
        }
        let copies = add(
            imported,
            bytes::<u64>(owner.input.len().checked_mul(4).ok_or(Error::Bounds)?)?,
        )?;
        let baseline = add(baseline, copies)?;
        admit(baseline, cap)?;
        let visits = owner
            .input
            .len()
            .checked_mul(owner.input.len())
            .and_then(|n| {
                n.checked_add(program.logical_len().checked_mul(owner.end - owner.first)?)
            })
            .ok_or(Error::Bounds)?;
        custody.prepay_bias_metadata(visits)?; // Same model-neutral original metadata ledger.
        // Every interior gather must die inside this original operator. Otherwise
        // output-only restore would discard a genuine later use: fail closed.
        for row in &owner.rows {
            if program
                .output_coordinates()
                .iter()
                .any(|&(id, _)| id == row.input)
                || (owner.end..program.logical_len())
                    .any(|id| body::integration::uses(program.recipe(id), row.input))
            {
                return Err(Error::Binding);
            }
        }
        let mut inputs = reserve(owner.input.len())?;
        let mut specs = reserve(owner.input.len())?;
        for (id, _) in owner.input.iter() {
            if !inputs.contains(&id) {
                inputs.push(id);
                let node = program.recipe_at(id).ok_or(Error::Binding)?;
                specs.push(body::Register {
                    lanes: node.lanes,
                    bounds: node.bounds,
                });
            }
        }
        let mut builder =
            BodyBuilder::new_with_baseline(WeightMode::Public, &specs, baseline, cap)?;
        let mut parts = reserve(owner.input.len())?;
        for (id, lane) in owner.input.iter() {
            let register = inputs.iter().position(|&v| v == id).ok_or(Error::Binding)?;
            parts.push(Layout::affine(register, lane, &[1], &[1])?);
        }
        admit(
            add(
                baseline,
                Layout::quote_concat(&parts)?.construction_peak_bytes,
            )?,
            cap,
        )?;
        let input = Layout::concat(&parts)?;
        let output = builder.retained_native_rows(
            &input,
            owner.shape()[3],
            owner.rows.iter().map(|row| &row.program),
        )?;
        let views = [output];
        let construction = builder.proof_finish_quote(&views, size_of::<[Layout; 1]>())?;
        admit(add(baseline, construction)?, cap)?;
        let plan = builder.finish(&views)?;
        let original = owner
            .rows
            .iter()
            .try_fold(WorkBudget::default(), |w, row| {
                w.checked_add(row.program.work_budget())
            })?;
        if plan.work_budget() != original
            || plan.input_count() != inputs.len()
            || plan.output_count() != 1
        {
            return Err(Error::Binding);
        }
        // Native backing is already retained by exact original row Arcs, never
        // by digest sharing. BODY reports its own workspace and metadata here.
        let execution_bytes = plan.resources().admitted_bytes;
        let admitted = add(baseline, construction.max(execution_bytes))?;
        admit(admitted, cap)?;
        let mut outputs = reserve(owner.rows.len())?;
        outputs.extend(owner.rows.iter().map(|row| row.output));
        let publication =
            compact_lowering::publish_rotary_creation(owner, &program, registry, baseline, cap)?;
        if publication.work != original
            || publication.shape.iter().product::<usize>() != plan.output_layout(0)?.len()
        {
            return Err(Error::Binding);
        }
        let mut compiled = Arc::new(Self {
            plan,
            publication,
            inputs,
            outputs,
            resources: Resources {
                retained_owner_bytes,
                control_and_mapping_bytes,
                import_and_restore_overlap_bytes: copies,
                construction_bytes: construction,
                execution_bytes,
                admitted_bytes: admitted,
            },
            program,
        });
        Arc::get_mut(&mut compiled)
            .ok_or(Error::Binding)?
            .publication
            .owner
            .published();
        Ok(compiled)
    }
    pub(in crate::replicated_decoder) fn advance_to_start(
        &self,
        worker: &mut Worker,
    ) -> Result<BoundaryProgress, Error> {
        SuspendedWorker::advance_to_start(&self.program, self.publication.owner.first, worker)
    }
    pub(in crate::replicated_decoder) fn begin(
        self: Arc<Self>,
        drained: CoordinatorDrained<Worker>,
        registry: &SourceRegistry<'_>,
    ) -> Result<Invocation, Error> {
        admit(self.resources.admitted_bytes, MAX_PROGRAM_BYTES)?;
        self.publication.owner.check(&self.program, registry)?;
        SuspendedWorker::check_shared_source(&drained.owner, self.publication.owner.public())?;
        let (mut worker, runtime, inputs) = SuspendedWorker::begin(
            drained.owner,
            &self.program,
            self.publication.owner.first,
            &self.inputs,
        )?;
        match Executor::new_with_registry(
            Arc::clone(&self.plan),
            runtime,
            inputs,
            Some(Arc::clone(self.publication.owner.public())),
            registry,
        ) {
            Ok(executor) => Ok(Invocation {
                executor: Some(executor),
                worker,
                compiled: self,
            }),
            Err(e) => {
                worker.cancel();
                Err(e)
            }
        }
    }
}
pub(in crate::replicated_decoder) struct Invocation {
    // Same ordered drop as shared restoration: secrets before original runtime,
    // then Compiled/native/source owners. Cancellation never returns a Worker.
    executor: Option<Executor>,
    worker: SuspendedWorker,
    compiled: Arc<Compiled>,
}
impl Invocation {
    pub(in crate::replicated_decoder) fn complete(&self) -> bool {
        self.executor.as_ref().is_some_and(Executor::complete)
    }
    pub(in crate::replicated_decoder) fn party(&self) -> puma::Party {
        self.worker.party()
    }
    pub(in crate::replicated_decoder) fn cancel(&mut self) {
        self.executor = None;
        self.worker.cancel()
    }
    pub(in crate::replicated_decoder) fn advance(
        &mut self,
    ) -> Result<Vec<(puma::Party, PeerBytes)>, Error> {
        let result = self.executor.as_mut().ok_or(Error::Retired)?.advance();
        if result.is_err() {
            self.cancel()
        }
        result
    }
    pub(in crate::replicated_decoder) fn receive(
        &mut self,
        bytes: &[u8],
        from: puma::Party,
    ) -> Result<Vec<(puma::Party, PeerBytes)>, Error> {
        let result = self
            .executor
            .as_mut()
            .ok_or(Error::Retired)?
            .receive(bytes, from);
        if result.is_err() {
            self.cancel()
        }
        result
    }
}
impl CoordinatorDrained<Invocation> {
    pub(in crate::replicated_decoder) fn restore(self) -> Result<Worker, Error> {
        let mut invocation = self.owner;
        #[cfg(test)]
        body::integration::restoration_tests::extracting();
        let (runtime, outputs) = invocation.executor.take().ok_or(Error::Retired)?.finish()?;
        let outputs = RestoreOutputs {
            values: outputs,
            #[cfg(test)]
            _erased: body::integration::restoration_tests::ErasureProbe::take(),
        };
        let width = invocation.compiled.publication.shape[3];
        invocation.worker.restore_owned(
            runtime,
            invocation.compiled.publication.owner.end,
            &invocation.compiled.outputs,
            outputs,
            |outputs, row, lane| {
                #[cfg(test)]
                body::integration::restoration_tests::reading()?;
                outputs.values.word(0, row * width + lane)
            },
        )
    }
}

#[cfg(test)]
#[path = "rotary_handoff_tests.rs"]
pub(in crate::replicated_decoder) mod tests;
