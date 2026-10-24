//! Request-only closed original guarded-SiLU construction. No row/validity proof.
use super::*;
use crate::replicated_decoder::program::EmissionContext;
use std::mem::size_of_val;

struct Parts {
    full: tensor::AdmittedFactory<'static>,
    tail: Option<tensor::AdmittedFactory<'static>>,
}

// Same concrete query/controller positions as the scalar request rescale seam.
// No credit for aliasing full/tail parameters or compiler-elided moves.
fn controls(attempt_bytes: usize) -> Result<usize, Error> {
    let r = tensor::WorkPreflightWorkspace::resources();
    let mut bytes = add(size_of_val(&r), size_of_val(&r))?;
    bytes = add(bytes, attempt_bytes.checked_mul(5).ok_or(Error::Bounds)?)?;
    macro_rules! slot {
        ($n:expr) => {
            bytes = add(bytes, $n)?;
        };
    }
    macro_rules! pair {
        ($n:expr) => {
            slot!($n.checked_mul(2).ok_or(Error::Bounds)?);
        };
    }
    pair!(r.workspace_bytes);
    pair!(r.factory_bytes);
    pair!(r.ticket_bytes);
    pair!(r.admitted_schedule_bytes);
    pair!(r.admitted_part_bytes);
    pair!(r.summary_bytes);
    pair!(r.ledger_bytes);
    pair!(r.ticket_result_bytes);
    pair!(r.build_result_bytes);
    pair!(size_of::<Parts>());
    slot!(size_of::<Result<Parts, Error>>());
    slot!(size_of::<Option<tensor::AdmittedFactory<'static>>>());
    slot!(size_of::<Result<tensor::FactoryOutput, Error>>());
    slot!(size_of::<(tensor::Program, tensor::ProducedCertificate)>());
    slot!(size_of::<
        Result<(tensor::Program, tensor::ProducedCertificate), Error>,
    >());
    slot!(size_of::<&EmissionContext<'static>>());
    Ok(bytes)
}

fn peak<F>(lanes: usize, _: &F, attempt_bytes: usize) -> Result<usize, Error> {
    if lanes == 0 || lanes > MAX_VALUES {
        return Err(Error::Bounds);
    }
    let width = lanes.min(tensor::MAX_TILE_LANES);
    let remainder = lanes % width;
    let full = Factory::GuardedSilu.quote(width)?;
    let tail = (remainder != 0)
        .then(|| Factory::GuardedSilu.quote(remainder))
        .transpose()?;
    let a = AssemblyInventory::for_callback::<F>()?;
    let mut templates = full.immutable_template_bytes;
    let mut bytes = a.quote.max(add(a.full, full.new_peak_bytes)?);
    if let Some(tail) = tail {
        bytes = bytes.max(add(a.tail, add(templates, tail.new_peak_bytes)?)?);
        templates = add(templates, tail.immutable_template_bytes)?;
    }
    bytes = bytes.max(add(a.owner, templates)?);
    add(bytes, controls(attempt_bytes)?)
}

impl PreparedTensorMap {
    pub(in crate::replicated_decoder) fn request_silu_peak(
        context: &EmissionContext<'_>,
        lanes: usize,
    ) -> Result<usize, Error> {
        let mut attempt = context.attempt();
        context.check_open()?;
        let admit = |bytes| context.admit_phase(bytes, 0);
        let result = peak(lanes, &admit, size_of_val(&attempt));
        if result.is_ok() {
            attempt.commit();
        }
        result
    }

    fn request_silu_parts(
        context: &EmissionContext<'_>,
        lanes: usize,
        input: Interval,
    ) -> Result<Parts, Error> {
        let mut attempt = context.attempt();
        context.check_open()?;
        context.admit_phase(controls(size_of_val(&attempt))?, 0)?;
        context.admit_phase(Self::request_silu_peak(context, lanes)?, 0)?;
        let width = lanes.min(tensor::MAX_TILE_LANES);
        let remainder = lanes % width;
        let mut workspace = tensor::WorkPreflightWorkspace::empty();
        let full = tensor::Construction::GuardedSilu { lanes: width }
            .bind(tensor::FactoryParameters::GuardedSilu { input })
            .map_err(|_| Error::Bounds)?;
        let ticket = if remainder != 0 {
            let tail = tensor::Construction::GuardedSilu { lanes: remainder }
                .bind(tensor::FactoryParameters::GuardedSilu { input })
                .map_err(|_| Error::Bounds)?;
            full.preflight_full_tail(
                u64::try_from(lanes / width).map_err(|_| Error::Bounds)?,
                tail,
                &mut workspace,
            )
        } else {
            full.preflight(
                u64::try_from(lanes / width).map_err(|_| Error::Bounds)?,
                &mut workspace,
            )
        }
        .map_err(|_| Error::Bounds)?;
        // One persistent debit covers ALL invocations BEFORE either factory.
        let (full, tail) = context.admit_work(ticket)?.into_parts();
        attempt.commit();
        Ok(Parts { full, tail })
    }

    pub(in crate::replicated_decoder) fn admit_request_silu_reuse(
        context: &EmissionContext<'_>,
        lanes: usize,
        input: Interval,
    ) -> Result<(), Error> {
        let _parts = Self::request_silu_parts(context, lanes, input)?;
        Ok(())
    }

    pub(in crate::replicated_decoder) fn silu_request(
        context: &EmissionContext<'_>,
        lanes: usize,
        input: Interval,
    ) -> Result<Self, Error> {
        let mut attempt = context.attempt();
        let Parts { full, tail } = Self::request_silu_parts(context, lanes, input)?;
        let width = lanes.min(tensor::MAX_TILE_LANES);
        let full = Self::request_silu_build(full, Factory::GuardedSilu.quote(width)?)?;
        #[cfg(test)]
        PANIC_AFTER_FULL.with(|flag| assert!(!flag.replace(false), "request silu after full"));
        let tail = match tail {
            Some(part) => Some(Self::request_silu_build(
                part,
                Factory::GuardedSilu.quote(lanes % width)?,
            )?),
            None => None,
        };
        let prepared = Self {
            lanes,
            input,
            factory: Factory::GuardedSilu,
            full,
            tail,
        };
        prepared.validate()?; // Includes exact output width and no validity owner.
        context.retain(prepared.baseline_bytes()?)?;
        attempt.commit();
        Ok(prepared)
    }

    fn request_silu_build(
        part: tensor::AdmittedFactory<'_>,
        quote: tensor::ConstructionQuote,
    ) -> Result<Arc<tensor::Program>, Error> {
        #[cfg(test)]
        NATIVE_ATTEMPTS.with(|n| n.set(n.get() + 1));
        let (program, certificate) = part.build()?.into_parts();
        if certificate != tensor::ProducedCertificate::None {
            return Err(Error::Binding);
        }
        let resources = program.resources()?;
        if resources.immutable_template_bytes > quote.immutable_template_bytes
            || resources.shared_admitted_bytes > puma::MAX_ADMITTED_BYTES
        {
            return Err(Error::Bounds);
        }
        let owner = Arc::new(program);
        #[cfg(test)]
        NATIVE_OWNERS.with(|owners| {
            if let Some(owners) = &mut *owners.borrow_mut() {
                owners.push(Arc::downgrade(&owner));
            }
        });
        Ok(owner)
    }
}

#[cfg(test)]
thread_local! { static PANIC_AFTER_FULL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
#[cfg(test)]
pub(in crate::replicated_decoder) fn panic_after_full() {
    PANIC_AFTER_FULL.with(|flag| flag.set(true));
}
