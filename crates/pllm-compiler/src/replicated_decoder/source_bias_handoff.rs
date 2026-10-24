//! ONE public bias site: original projection/rescale -> BODY -> original consumer.
//! No default route, primitive loan, second Worker, or production drain issuer.
use super::super::super::compact_lowering::{self, CreationPublication, DescriptorKind};
use super::*;
use crate::replicated_decoder::runtime::{BoundaryProgress, body};
use body::handoff::{SuspendedWorker, TRANSITION_INLINE_BYTES, owned_restore_inline_bytes};
use layer::{Builder as BodyBuilder, Executor, Layout, Plan, Values};

struct RestoreOutputs {
    values: Values,
    #[cfg(test)]
    _erased: body::integration::restoration_tests::ErasureProbe,
}

#[cfg(test)]
thread_local! { static CONSTRUCTOR_ENTRIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
#[cfg(test)]
pub(in crate::replicated_decoder) fn constructor_entries() -> usize {
    CONSTRUCTOR_ENTRIES.with(std::cell::Cell::get)
}

/// Only authenticated coordinator custody can enter a consuming transition.
/// Intentionally no production issuer: local readiness/completion proves no drain.
pub(in crate::replicated_decoder) struct CoordinatorDrained<T> {
    owner: T,
}

pub(in crate::replicated_decoder) struct Compiled {
    plan: Arc<Plan>,
    publication: CreationPublication,
    outputs: Box<[usize]>,
    admitted_bytes: usize,
    // LAST: original guards outlive body metadata and creation's native handles.
    program: Arc<Program>,
}

fn admit(bytes: usize, cap: usize) -> Result<(), Error> {
    if bytes > cap.min(MAX_PROGRAM_BYTES) {
        Err(Error::Bounds)
    } else {
        Ok(())
    }
}

/// Additive named controls. Original full plan/source/history/templates remain
/// charged by custody's real finalization peak, not a digest-based alias credit.
fn controls(program: &Program, d: Description) -> Result<usize, Error> {
    [
        bytes::<Compiled>(2)?, // Construction candidate and final Arc payload.
        size_of::<Invocation>(),
        size_of::<Worker>(),
        size_of::<Result<Invocation, Error>>(),
        size_of::<Result<Worker, Error>>(),
        size_of::<Result<Arc<Compiled>, Error>>(),
        size_of::<BodyBuilder>(),
        bytes::<Description>(3)?,
        size_of::<[body::Register; 1]>(),
        bytes::<Result<Layout, Error>>(2)?,
        size_of::<Result<Arc<Plan>, Error>>(),
        size_of::<Layout>(),
        size_of::<[Layout; 1]>(),
        size_of::<Arc<Plan>>(),
        size_of::<Plan>(),
        size_of::<Executor>(),
        size_of::<Result<(multiply::PartyRuntime, Values), Error>>(),
        size_of::<RestoreOutputs>(),
        owned_restore_inline_bytes::<RestoreOutputs>()?,
        TRANSITION_INLINE_BYTES,
        bytes::<Option<Secret>>(program.logical_len())?,
        bytes::<usize>(d.lanes().div_ceil(128))?,
        bytes::<usize>(d.lanes().div_ceil(128))?, // Vec-to-box construction overlap.
        size_of::<Vec<usize>>(),
        size_of::<Vec<Secret>>(),
        size_of::<Secret>(),
        bytes::<usize>(4)?, // Compiled + Plan Arc controls.
        compact_lowering::creation_publication_bytes()?,
        // Original idle runtime/transport still retained. BODY's own workspace
        // is charged independently through Plan::resources before extraction.
        puma::MAX_ADMITTED_BYTES,
        size_of::<[Option<puma::Message>; 2]>(),
    ]
    .into_iter()
    .try_fold(0, add)
}

impl Compiled {
    pub(in crate::replicated_decoder) fn admitted_bytes(&self) -> usize {
        self.admitted_bytes
    }
    pub(in crate::replicated_decoder) fn prepare(
        program: Arc<Program>,
        owner: Creation,
        registry: &SourceRegistry<'_>,
        cap: usize,
    ) -> Result<Arc<Self>, Error> {
        owner.check(&program, registry)?;
        let d = owner.description();
        let retained = program
            .request_custody
            .as_ref()
            .ok_or(Error::Binding)?
            .retained_bytes();
        let baseline = add(retained, controls(&program, d)?)?;
        admit(baseline, cap)?;
        let metadata = add(
            (d.end - d.bias_start())
                .checked_mul(program.logical_len())
                .ok_or(Error::Bounds)?,
            add(
                d.lanes().checked_mul(8).ok_or(Error::Bounds)?,
                source_registry::source_metadata_bytes(
                    registry.source_records(),
                    registry.source_norms(),
                )?,
            )?,
        )?;
        program
            .request_custody
            .as_ref()
            .ok_or(Error::Binding)?
            .prepay_bias_metadata(metadata)?;
        // No later user of a gather may be silently discarded. This borrows the
        // existing dependency representation; Node::inputs() would allocate.
        for step in (d.bias_start()..d.end).step_by(3) {
            for gather in [step, step + 1] {
                if program
                    .output_coordinates()
                    .iter()
                    .any(|&(id, _)| id == gather)
                    || (d.end..program.logical_len())
                        .any(|id| body::integration::uses(program.recipe(id), gather))
                {
                    return Err(Error::Binding);
                }
            }
        }
        // Public bias has no independent secret input and no primitive debit.
        let specs = [body::Register {
            lanes: d.lanes(),
            bounds: d.floor_bounds,
        }];
        #[cfg(test)]
        CONSTRUCTOR_ENTRIES.with(|n| n.set(n.get() + 1));
        let mut builder =
            BodyBuilder::new_with_baseline(WeightMode::Public, &specs, baseline, cap)?;
        let input = builder.input(0)?;
        let output = builder.bias(&input, Arc::clone(owner.source()), d.geometry[2])?;
        let views = [output];
        let construction = builder.proof_finish_quote(&views, size_of::<[Layout; 1]>())?;
        admit(add(baseline, construction)?, cap)?;
        let plan = builder.finish(&views)?;
        if plan.work_budget() != WorkBudget::default()
            || plan.input_count() != 1
            || plan.output_count() != 1
        {
            return Err(Error::Binding);
        }
        // Simultaneous original live values, imported copy, body retained outputs
        // and copied exports. Never refund original work or retained Arc backing.
        let copies = d
            .lanes()
            .checked_mul(16)
            .and_then(|n| n.checked_mul(3))
            .ok_or(Error::Bounds)?;
        let execution = add(plan.resources().admitted_bytes, copies)?;
        let admitted_bytes = add(baseline, construction.max(execution))?;
        admit(admitted_bytes, cap)?;
        let publication =
            compact_lowering::publish_creation(owner, &program, registry, baseline, cap)?;
        let original_work = (d.first..d.end).try_fold(WorkBudget::default(), |work, id| {
            work.checked_add(program.recipe(id).work_budget()?)
        })?;
        if publication.original_work != original_work {
            return Err(Error::Binding);
        }
        let DescriptorKind::Projection {
            bias: Some(bias),
            geometry,
            ..
        } = &publication.census.kind
        else {
            return Err(Error::Binding);
        };
        if *geometry != d.geometry
            || bias.first != d.bias_start()
            || bias.input_owner != d.first + 2
        {
            return Err(Error::Binding);
        }
        let count = d.lanes().div_ceil(128);
        let mut outputs = Vec::new();
        outputs
            .try_reserve_exact(count)
            .map_err(|_| Error::Allocation)?;
        if outputs.capacity() != count {
            return Err(Error::Allocation);
        }
        outputs.extend((d.bias_start()..d.end).step_by(3).map(|id| id + 2));
        let mut compiled = Arc::new(Self {
            plan,
            program,
            outputs: outputs.into_boxed_slice(),
            admitted_bytes,
            publication,
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
        SuspendedWorker::advance_to_start(
            &self.program,
            self.publication.owner.description().bias_start(),
            worker,
        )
    }

    pub(in crate::replicated_decoder) fn begin(
        self: Arc<Self>,
        drained: CoordinatorDrained<Worker>,
        registry: &SourceRegistry<'_>,
    ) -> Result<Invocation, Error> {
        let worker = drained.owner;
        admit(self.admitted_bytes, MAX_PROGRAM_BYTES)?;
        self.publication.owner.check(&self.program, registry)?;
        SuspendedWorker::check_shared_source(&worker, self.publication.owner.public())?;
        let d = self.publication.owner.description();
        let (mut suspended, runtime, inputs) =
            SuspendedWorker::begin(worker, &self.program, d.bias_start(), &[d.first + 2])?;
        match Executor::new_with_registry(
            Arc::clone(&self.plan),
            runtime,
            inputs,
            Some(Arc::clone(self.publication.owner.public())),
            registry,
        ) {
            Ok(executor) => Ok(Invocation {
                executor: Some(executor),
                worker: suspended,
                compiled: self,
            }),
            Err(error) => {
                suspended.cancel();
                Err(error)
            }
        }
    }
}

// Drop order is deliberate: BODY secret outputs erase before original Worker
// cancellation, then source/native/parameter owners release with Compiled last.
pub(in crate::replicated_decoder) struct Invocation {
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
        self.worker.cancel();
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
        raw: &[u8],
        from: puma::Party,
    ) -> Result<Vec<(puma::Party, PeerBytes)>, Error> {
        let result = self
            .executor
            .as_mut()
            .ok_or(Error::Retired)?
            .receive(raw, from);
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
        let limit = body::integration::restoration_tests::limit();
        #[cfg(not(test))]
        let limit = MAX_PROGRAM_BYTES;
        admit(invocation.compiled.admitted_bytes, limit)?;
        #[cfg(test)]
        body::integration::restoration_tests::extracting();
        let (runtime, values) = invocation.executor.take().ok_or(Error::Retired)?.finish()?;
        let d = invocation.compiled.publication.owner.description();
        let outputs = RestoreOutputs {
            values,
            #[cfg(test)]
            _erased: body::integration::restoration_tests::ErasureProbe::take(),
        };
        invocation.worker.restore_owned(
            runtime,
            d.end,
            &invocation.compiled.outputs,
            outputs,
            |outputs, tile, lane| {
                #[cfg(test)]
                body::integration::restoration_tests::reading()?;
                outputs.values.word(0, tile * 128 + lane)
            },
        )
    }
}

#[cfg(test)]
#[path = "source_bias_route_tests.rs"]
pub(in crate::replicated_decoder) mod route_tests;
