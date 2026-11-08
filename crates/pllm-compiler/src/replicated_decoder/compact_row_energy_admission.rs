//! INACTIVE fixed-arithmetic admission ingredient, NOT a traversal quote or
//! proof issuer. No production caller. Quote construction itself, caller locals,
//! descriptor lookup/new_source, row_norm/alias walks, publication, Vec growth,
//! error cleanup and retained-root overlap need separately admitted inventory.
//! Missing that inventory must keep propagation unapplied, not mint a ledger.
use super::*;
use super::super::row_energy::{self, Activation, Arithmetic, Product, Projection, RowEnergyWorkspace};
use std::mem::{align_of, size_of};

/// Equality predicate only, NOT a validator/certificate: the caller still must
/// validate the retained publication against its original graph/source binding.
/// In particular equal shape, bounds or digest alone cannot establish custody.
pub(super) fn same_publication(a: &PublishedView, b: &PublishedView) -> bool {
    Arc::ptr_eq(&a.origin, &b.origin) && a.slot == b.slot
        && a.mapping == b.mapping && a.start == b.start && a.lanes == b.lanes
        && a.shape == b.shape && a.rank == b.rank && a.logical == b.logical
        && a.order == b.order && a.import_slot == b.import_slot && a.bounds == b.bounds
}

/// Named logical slots only; not a claim about physical stack/RSS or CPU time.
/// Sum rather than max: no unproved sequential scratch reuse. Per-type alignment
/// slack covers independently laid out logical slots, including borrowed refs.
fn slots<T>(count: usize) -> Result<usize, Error> {
    size_of::<T>().checked_add(align_of::<T>() - 1)
        .and_then(|n| n.checked_mul(count)).ok_or(Error::Bounds)
}

struct FixedQuote {
    bytes: usize,
    visits: u64,
}

fn fixed_quote(a: &Arithmetic) -> Result<FixedQuote, Error> {
    let mut bytes = 0;
    for n in [
        slots::<RowEnergyWorkspace>(1)?,
        slots::<Projection>(2)?, slots::<Activation>(2)?, slots::<Product>(2)?,
        slots::<&RowEnergyWorkspace>(2)?, slots::<&Projection>(2)?,
        slots::<&Activation>(2)?, slots::<&Product>(2)?, slots::<&Arithmetic>(2)?,
        slots::<Option<u128>>(8)?, slots::<Result<Option<u128>, Error>>(3)?,
        // residual arguments, destructured energy options, mul/add operands and
        // checked return temporaries; call/return slots intentionally overlap.
        slots::<u128>(20)?, slots::<Result<u128, Error>>(4)?,
        slots::<Result<(), Error>>(3)?, slots::<Error>(3)?,
        slots::<FixedQuote>(2)?, slots::<Result<FixedQuote, Error>>(2)?,
        slots::<usize>(8)?, slots::<u64>(4)?, slots::<bool>(4)?,
        crate::original_source::OriginalSourcePlan::compiler_metadata_control_bytes()
            .map_err(|_| Error::Bounds)?,
    ] { bytes = add_bytes(bytes, n)?; }
    // Arithmetic-only census. Structural call/return/move/quote/debit operations
    // are NOT hidden in a guessed constant: they remain an integration gate.
    let visits = [a.mul, a.add, a.min, a.roots, a.div, a.rem,
        a.comparisons, a.conversions, a.checked_refusals, a.proof_options]
        .into_iter().try_fold(0usize, add_bytes)?;
    Ok(FixedQuote { bytes, visits: u64::try_from(visits).map_err(|_| Error::Bounds)? })
}

/// Not usable as live admission: private, unused and deliberately arithmetic-
/// only. A future approved caller must pre-admit quote construction and supply
/// proven total caller/contraction/publication overlap before reaching here.
/// This does demonstrate the SAME authentic cumulative nonrefundable work root;
/// admit_local/admit_total are never treated as an original-work debit.
fn prepay_arithmetic<G: ControllerAccounting>(
    graph: &CompactGraph<'_, '_, G>, a: &Arithmetic, separately_quoted_live: usize,
) -> Result<(), Error> {
    let context = graph.request_admission.ok_or(Error::Binding)?;
    let quote = fixed_quote(a)?;
    context.admit_phase(add_bytes(separately_quoted_live, quote.bytes)?, 0)?;
    context.prepay_metadata(quote.visits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_frame_is_not_an_energy_option_or_a_complete_admission() {
        for a in [&row_energy::PROJECTION, &row_energy::ACTIVATION,
            &row_energy::PRODUCT, &row_energy::RESIDUAL] {
            let quote = fixed_quote(a).unwrap();
            assert!(quote.bytes > size_of::<RowEnergyWorkspace>());
            assert!(quote.bytes > size_of::<Option<u128>>());
            assert!(quote.visits > 0);
        }
        assert!(slots::<RowEnergyWorkspace>(usize::MAX).is_err());
    }

    #[test]
    fn exact_publication_rejects_foreign_owner_offsets_width_order_and_origin() {
        let original = PublishedView {
            origin: Arc::new(()), slot: Some(0), mapping: Mapping::Dense(7),
            start: 0, lanes: 2, shape: [1, 1, 2], rank: 3,
            logical: ViewShape::new(&[1, 1, 2]).unwrap(), order: LaneOrder::Linear,
            import_slot: None, bounds: Interval::new(-1, 1).unwrap(),
        };
        assert!(same_publication(&original, &original.clone()));
        let mut wrong = original.clone(); wrong.origin = Arc::new(());
        assert!(!same_publication(&original, &wrong));
        let mut wrong = original.clone(); wrong.mapping = Mapping::Dense(8);
        assert!(!same_publication(&original, &wrong));
        let mut wrong = original.clone(); wrong.start = 1;
        assert!(!same_publication(&original, &wrong));
        let mut wrong = original.clone(); wrong.lanes = 1;
        assert!(!same_publication(&original, &wrong));
        let mut wrong = original.clone(); wrong.shape = [1, 2, 1];
        assert!(!same_publication(&original, &wrong));
        let mut wrong = original.clone(); wrong.rank = 2;
        assert!(!same_publication(&original, &wrong));
        let mut wrong = original.clone(); wrong.import_slot = Some(0);
        assert!(!same_publication(&original, &wrong));
        let mut wrong = original.clone(); wrong.bounds = Interval::new(0, 1).unwrap();
        assert!(!same_publication(&original, &wrong));
        let mut wrong = original.clone(); wrong.logical = ViewShape::new(&[1, 2, 1]).unwrap();
        assert!(!same_publication(&original, &wrong));
        let mut wrong = original.clone(); wrong.slot = Some(1);
        assert!(!same_publication(&original, &wrong));
        let mut wrong = original.clone();
        wrong.order = LaneOrder::HeadsSequence {
            heads: 1, rows: 1, width: 2, to_heads: true, base_start: 0,
        };
        assert!(!same_publication(&original, &wrong));
        // Identical origin/shape still does not validate a SourceTensor pointer,
        // original source_binding, guard/fallback energy, or a transposed matrix.
    }
}
