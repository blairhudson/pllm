//! Request-only capture of EXISTING checked-row/RMS native factory tickets.
//! No GENERAL premise is invented: None selects the ordinary certified factory.
use super::*;
use std::mem::size_of;

pub(super) struct GuardTicket<'a> {
    pub(super) range: Option<tensor::AdmittedFactory<'a>>,
    pub(super) valid: Option<tensor::AdmittedFactory<'a>>,
}
pub(super) struct Tickets<'a> {
    pub(super) guards: Vec<GuardTicket<'a>>,
    pub(super) rms: Option<tensor::AdmittedFactory<'a>>,
}
pub(super) fn slots(tiles: usize) -> Result<usize, Error> {
    let r = tensor::WorkPreflightWorkspace::resources();
    let mut total = array_bytes::<GuardTicket<'_>>(tiles)?;
    for n in [
        2 * size_of::<Tickets<'_>>(),
        2 * size_of::<Result<Option<Tickets<'_>>, Error>>(),
        3 * size_of::<Option<tensor::AdmittedFactory<'_>>>(),
        2 * r.factory_bytes,
        2 * r.ticket_bytes,
        2 * r.admitted_schedule_bytes,
        2 * r.ticket_result_bytes,
        r.summary_bytes,
        r.ledger_bytes,
        r.build_result_bytes,
    ] {
        total = add_bytes(total, n)?;
    }
    Ok(total)
}

pub(super) struct RequestNorm<'a> {
    pub(super) width: usize,
    pub(super) gamma: &'a SourceTensor,
    pub(super) accepted: Interval,
    pub(super) bounds: Interval,
    pub(super) energy: Option<u128>,
    pub(super) tiles: usize,
    pub(super) controls: usize,
}
pub(super) fn prepare<'a>(
    graph: &CompactGraph<'_, 'a>,
    input: &PublishedView,
    request: RequestNorm<'a>,
) -> Result<Option<Tickets<'a>>, Error> {
    let RequestNorm {
        width,
        gamma,
        accepted,
        bounds,
        energy,
        tiles,
        controls,
    } = request;
    let Some(context) = graph.request_admission else {
        return Ok(None);
    };
    context.check_final_work(graph.work)?;
    let r = tensor::WorkPreflightWorkspace::resources();
    let live = add_bytes(
        add_bytes(controls, graph.contraction_control_bytes.get())?,
        add_bytes(
            tensor::RmsWorkWorkspace::workspace_bytes(),
            r.workspace_bytes,
        )?,
    )?;
    context.admit_phase(live, 0)?;
    let visits = tiles
        .checked_mul(2)
        .and_then(|n| n.checked_add(1))
        .and_then(|n| n.checked_mul(tensor::MAX_NODES * 36))
        .ok_or(Error::Bounds)?;
    context
        .plan()
        .prepay_compiler_metadata(
            &context.reservation(),
            u64::try_from(visits).map_err(|_| Error::Bounds)?,
        )
        .map_err(|_| Error::Bounds)?;
    // Ticket storage and BOTH workspaces were admitted before either allocation.
    let mut guards = Vec::new();
    guards
        .try_reserve_exact(tiles)
        .map_err(|_| Error::Allocation)?;
    if guards.capacity() != tiles {
        return Err(Error::Allocation);
    }
    let mut workspace = tensor::WorkPreflightWorkspace::empty();
    for tile in 0..tiles {
        let lanes = (input.lanes - tile * 128).min(128);
        let incoming = input_bounds(graph, input, tile * 128, lanes)?;
        let range = tensor::Construction::CheckedRange { lanes }
            .bind(tensor::FactoryParameters::CheckedRange {
                incoming,
                accepted,
                fallback: 0,
            })
            .and_then(|factory| factory.preflight(1, &mut workspace))
            .map_err(|_| Error::Bounds)?;
        let range = context.admit_work(range)?.into_parts().0;
        let valid = tensor::Construction::AllValid { lanes }
            .bind(tensor::FactoryParameters::AllValid)
            .and_then(|factory| factory.preflight(1, &mut workspace))
            .map_err(|_| Error::Bounds)?;
        let valid = context.admit_work(valid)?.into_parts().0;
        guards.push(GuardTicket {
            range: Some(range),
            valid: Some(valid),
        });
    }
    let coefficients = gamma.public_q18.as_deref().ok_or(Error::Binding)?;
    let (construction, parameters) = if let Some(squared_norm) = energy {
        (
            tensor::Construction::RmsNormWithInputNorm {
                rows: 1,
                width,
                gamma_len: coefficients.len(),
            },
            tensor::FactoryParameters::RmsNormWithInputNorm {
                input: bounds,
                gamma: coefficients,
                squared_norm,
            },
        )
    } else {
        (
            tensor::Construction::RmsNormCertified {
                rows: 1,
                width,
                gamma_len: coefficients.len(),
            },
            tensor::FactoryParameters::RmsNormCertified {
                input: bounds,
                gamma: coefficients,
            },
        )
    };
    let rows = u64::try_from(input.lanes / width).map_err(|_| Error::Bounds)?;
    let rms = construction
        .bind(parameters)
        .and_then(|factory| factory.preflight(rows, &mut workspace))
        .map_err(|_| Error::Bounds)?;
    let rms = context.admit_work(rms)?.into_parts().0;
    let products = if tiles == 0 {
        0
    } else {
        tiles - usize::from(graph.validity.current().is_none())
    };
    context.admit_products(products)?;
    Ok(Some(Tickets {
        guards,
        rms: Some(rms),
    }))
}
