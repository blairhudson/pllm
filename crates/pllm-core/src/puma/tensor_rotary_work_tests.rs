use super::numeric::affine;
use super::rotary_work::factory_attempts;
use super::work_preflight::row_entries;
use super::*;

const ONE: i64 = 1 << 18;

// Literal ORIGINAL factory composition, independent of rotary_work's geometry,
// coefficient iterator and recipe. No expected work is derived from the query.
fn original(
    rows: usize,
    width: usize,
    input: Interval,
    cosine: &[i64],
    sine: &[i64],
) -> Result<Program, Error> {
    let lanes = rows.checked_mul(width).ok_or(Error::Bounds)?;
    if rows == 0
        || width == 0
        || width % 2 != 0
        || cosine.len() != lanes / 2
        || sine.len() != lanes / 2
    {
        return Err(Error::Shape);
    }
    let mut b = Builder::new(lanes, input)?;
    if cosine.iter().chain(sine).any(|c| !(-ONE..=ONE).contains(c)) {
        return Err(Error::Bounds);
    }
    let x = b.input();
    let half = width / 2;
    let mut coefficients = [affine(0, 0, 0); MAX_TILE_LANES];
    for (i, c) in coefficients[..lanes].iter_mut().enumerate() {
        *c = affine(cosine[(i / width) * half + i % half], 0, 0);
    }
    let cosine_product = b.affine_lanes(x, &coefficients[..lanes])?;
    let cosine_product = b.rescale(cosine_product, 18)?;
    let swapped = b.row_map(x, width, RowMap::Rotate(half))?;
    for (i, c) in coefficients[..lanes].iter_mut().enumerate() {
        let s = sine[(i / width) * half + i % half];
        *c = affine(if i % width < half { -s } else { s }, 0, 0);
    }
    let sine_product = b.affine_lanes(swapped, &coefficients[..lanes])?;
    let sine_product = b.rescale(sine_product, 18)?;
    let output = b.affine(cosine_product, Some(sine_product), affine(1, 1, 0))?;
    b.finish(output)
}

fn same(a: &Program, b: &Program) {
    assert_eq!(a.nodes, b.nodes);
    assert_eq!(a.len, b.len);
    assert_eq!(a.lanes(), b.lanes());
    assert_eq!(a.digest(), b.digest());
    assert_eq!(a.numeric_profile(), b.numeric_profile());
    assert_eq!(a.output_bounds(), b.output_bounds());
    assert_eq!(a.work_budget(), b.work_budget());
    assert_eq!(a.work_budget().cursors(), b.work_budget().cursors());
    assert_eq!(a.bit_widths, b.bit_widths);
    assert_eq!(a.last_use, b.last_use);
    assert_eq!(a.slots, b.slots);
    assert_eq!(a.register_count, b.register_count);
    assert_eq!(a.public_affines, b.public_affines);
    assert_eq!(a.public_affine_count, b.public_affine_count);
    assert_eq!(a.checked_validity, b.checked_validity);
    assert_eq!(a.resources().unwrap(), b.resources().unwrap());
}

fn closed<'a>(
    rows: usize,
    width: usize,
    input: Interval,
    cosine: &'a [i64],
    sine: &'a [i64],
) -> ClosedFactory<'a> {
    Construction::RotaryHalf {
        rows,
        width,
        cosine_len: cosine.len(),
        sine_len: sine.len(),
    }
    .bind(FactoryParameters::RotaryHalf {
        input,
        cosine,
        sine,
    })
    .unwrap()
}

#[test]
fn rotary_work_matches_independent_original_every_even_tile_width_rows_and_domains() {
    let _quote = Program::rotary_half_work_resources();
    let mut workspace = SoftmaxWorkWorkspace::empty();
    for width in (2..=MAX_TILE_LANES).step_by(2) {
        for rows in [1, 2, 3, MAX_TILE_LANES / width] {
            if rows * width > MAX_TILE_LANES {
                continue;
            }
            let cosine: Vec<_> = (0..rows * width / 2)
                .map(|i| [ONE, -ONE, 0, ONE / 2][i % 4])
                .collect();
            let sine: Vec<_> = (0..cosine.len())
                .map(|i| [-ONE / 2, ONE, 0, -ONE][i % 4])
                .collect();
            for input in [
                Interval::new(-32, 32).unwrap(),
                Interval::new(0, 0).unwrap(),
                Interval::new(-ONE, ONE).unwrap(),
            ] {
                let before = (factory_attempts(), row_entries());
                let work =
                    Program::rotary_half_work(rows, width, input, &cosine, &sine, &mut workspace)
                        .unwrap();
                assert_eq!((factory_attempts(), row_entries()), before);
                let expected = original(rows, width, input, &cosine, &sine).unwrap();
                assert_eq!(work, expected.work_budget());
                workspace.assert_native_metadata(&expected);
                let actual = Program::rotary_half(rows, width, input, &cosine, &sine).unwrap();
                same(&actual, &expected);
            }
        }
    }
}

#[test]
fn rotary_signed_quarter_ring_endpoints_keep_original_validation_and_tables() {
    let mut workspace = SoftmaxWorkWorkspace::empty();
    let input = Interval::new(-(1_i64 << 44), (1_i64 << 44) - 1).unwrap();
    let cosine = [ONE; MAX_TILE_LANES / 2];
    let sine = [0; MAX_TILE_LANES / 2];
    let work = Program::rotary_half_work(1, 128, input, &cosine, &sine, &mut workspace).unwrap();
    let expected = original(1, 128, input, &cosine, &sine).unwrap();
    let actual = Program::rotary_half(1, 128, input, &cosine, &sine).unwrap();
    assert_eq!(work, expected.work_budget());
    same(&actual, &expected);
    for input in [
        Interval::new(-(1_i64 << 44) - 1, 0).unwrap(),
        Interval::new(0, 1_i64 << 44).unwrap(),
    ] {
        assert_eq!(
            Program::rotary_half_work(1, 128, input, &cosine, &sine, &mut workspace),
            Err(Error::Bounds)
        );
        assert_eq!(
            original(1, 128, input, &cosine, &sine).err(),
            Some(Error::Bounds)
        );
    }
}

#[test]
fn rotary_invalid_geometry_lengths_domains_and_competing_error_winners_match_literal() {
    fn check(
        rows: usize,
        width: usize,
        input: Interval,
        cosine: &[i64],
        sine: &[i64],
        error: Error,
    ) {
        let mut workspace = SoftmaxWorkWorkspace::empty();
        let before = (factory_attempts(), row_entries());
        assert_eq!(
            Program::rotary_half_work(rows, width, input, cosine, sine, &mut workspace),
            Err(error)
        );
        assert_eq!((factory_attempts(), row_entries()), before);
        assert_eq!(
            original(rows, width, input, cosine, sine).err(),
            Some(error)
        );
        assert_eq!(
            Program::rotary_half(rows, width, input, cosine, sine).err(),
            Some(error)
        );
    }
    let input = Interval::new(-32, 32).unwrap();
    check(usize::MAX, 2, input, &[], &[], Error::Bounds); // overflow before short slices
    check(0, 2, input, &[], &[], Error::Shape);
    check(1, 0, input, &[], &[], Error::Shape);
    check(1, 3, input, &[ONE + 1], &[0], Error::Shape); // odd before domain
    check(1, 2, input, &[], &[ONE + 1], Error::Shape); // short before domain
    check(1, 2, input, &[ONE], &[], Error::Shape);
    check(1, 2, input, &[ONE, ONE], &[0], Error::Shape);
    check(1, 130, input, &[ONE + 1; 65], &[0; 65], Error::Bounds); // tile before domain
    check(1, 2, input, &[ONE + 1], &[0], Error::Bounds);
    check(1, 2, input, &[ONE], &[-ONE - 1], Error::Bounds);
    check(1, 2, input, &[i64::MIN], &[0], Error::Bounds); // never negate MIN
    check(
        1,
        2,
        Interval::new(i64::MIN, i64::MAX).unwrap(),
        &[ONE],
        &[0],
        Error::Bounds,
    );

    let mut workspace = WorkPreflightWorkspace::empty();
    let before = (factory_attempts(), row_entries());
    assert_eq!(
        closed(0, 2, input, &[], &[])
            .preflight(1, &mut workspace)
            .err(),
        Some(WorkPreflightError::Native(Error::Shape))
    );
    assert_eq!(
        Construction::RotaryHalf {
            rows: 0,
            width: 2,
            cosine_len: 0,
            sine_len: 0
        }
        .quote()
        .err(),
        Some(ConstructionQuoteError::Geometry(Error::Bounds))
    );
    assert_eq!((factory_attempts(), row_entries()), before);
}

#[test]
fn rotary_ticket_full_tail_binds_original_coefficients_and_counts_each_native_fill() {
    let input = Interval::new(-32, 32).unwrap();
    let cosine = [ONE, ONE / 2, -ONE / 2, 0];
    let sine = [0, ONE / 4, ONE / 2, -ONE];
    let tail_cosine = [ONE];
    let tail_sine = [ONE / 2];
    let _quote = WorkPreflightWorkspace::resources();
    let mut workspace = WorkPreflightWorkspace::empty();
    let before = (factory_attempts(), row_entries());
    let ticket = closed(4, 2, input, &cosine, &sine)
        .preflight_full_tail(
            3,
            closed(1, 2, input, &tail_cosine, &tail_sine),
            &mut workspace,
        )
        .unwrap();
    let summary = ticket.summary();
    assert_eq!((factory_attempts(), row_entries()), before);
    let mut total = WorkBudget::local().checked_mul(13).unwrap();
    let prior = total;
    let complete = prior.checked_add(summary.work).unwrap();
    let admitted = CallerWorkLedger::new(
        &mut total,
        complete.operations(),
        complete.cursors().aes_work,
    )
    .unwrap()
    .admit(ticket)
    .unwrap();
    assert_eq!(total, complete);
    assert_eq!((factory_attempts(), row_entries()), before);
    let (full, tail) = admitted.into_parts();
    let full = full.build().unwrap();
    let tail = tail.unwrap().build().unwrap();
    assert_eq!(
        (factory_attempts(), row_entries()),
        (before.0 + 2, before.1 + 2)
    );
    let expected_full = original(4, 2, input, &cosine, &sine).unwrap();
    let expected_tail = original(1, 2, input, &tail_cosine, &tail_sine).unwrap();
    same(full.program(), &expected_full);
    same(tail.program(), &expected_tail);
    assert_eq!(full.certificate(), ProducedCertificate::None);
    assert_eq!(tail.certificate(), ProducedCertificate::None);
    let expected = expected_full
        .work_budget()
        .checked_mul(3)
        .unwrap()
        .checked_add(expected_tail.work_budget())
        .unwrap();
    assert_eq!(summary.work, expected);
    assert_eq!(
        summary.full.construction.borrowed_public_bytes,
        (cosine.len() + sine.len()) * std::mem::size_of::<i64>()
    );
    // Fresh rescale public draws are not merged across repeated/tail calls.
    let merged = original(13, 2, input, &[ONE; 13], &[0; 13]).unwrap();
    assert_ne!(summary.work.cursors(), merged.work_budget().cursors());
}

#[test]
fn rotary_whole_pending_prior_exact_one_short_and_overflow_refuse_before_real_factory_entry() {
    let input = Interval::new(-32, 32).unwrap();
    let cosine = [ONE; 4];
    let sine = [0; 4];
    let mut workspace = WorkPreflightWorkspace::empty();
    let before = (factory_attempts(), row_entries());
    let work = closed(4, 2, input, &cosine, &sine)
        .preflight(3, &mut workspace)
        .unwrap()
        .summary()
        .work;
    let prior = WorkBudget::local().checked_mul(13).unwrap();
    let complete = prior.checked_add(work).unwrap();
    for (max_ops, max_aes) in [
        (complete.operations() - 1, complete.cursors().aes_work),
        (complete.operations(), complete.cursors().aes_work - 1),
    ] {
        let ticket = closed(4, 2, input, &cosine, &sine)
            .preflight(3, &mut workspace)
            .unwrap();
        let mut total = prior;
        let mut ledger = CallerWorkLedger::new(&mut total, max_ops, max_aes).unwrap();
        assert_eq!(
            ledger.admit(ticket).err(),
            Some(WorkPreflightError::Native(Error::Bounds))
        );
        assert_eq!(ledger.total(), prior);
        assert_eq!((factory_attempts(), row_entries()), before);
    }
    for count in [0, u64::MAX] {
        workspace.reset();
        let error = closed(4, 2, input, &cosine, &sine)
            .preflight(count, &mut workspace)
            .err();
        assert_eq!(
            error,
            Some(if count == 0 {
                WorkPreflightError::EmptySchedule
            } else {
                WorkPreflightError::Native(Error::Bounds)
            })
        );
        assert!(workspace.is_poisoned());
        assert_eq!((factory_attempts(), row_entries()), before);
    }
    workspace.reset();
    let ticket = closed(4, 2, input, &cosine, &sine)
        .preflight(1, &mut workspace)
        .unwrap();
    let mut zero = WorkBudget::default();
    assert_eq!(
        CallerWorkLedger::new(&mut zero, 0, 0)
            .unwrap()
            .admit(ticket)
            .err(),
        Some(WorkPreflightError::Native(Error::Bounds))
    );
    assert_eq!(zero, WorkBudget::default());
    assert_eq!((factory_attempts(), row_entries()), before);
}

#[test]
fn rotary_direct_metadata_reuse_and_ticket_poison_reset_keep_owner_parameters() {
    let input = Interval::new(-32, 32).unwrap();
    let cosine = [ONE];
    let sine = [0];
    let before = (factory_attempts(), row_entries());
    let mut direct = SoftmaxWorkWorkspace::empty();
    assert_eq!(
        Program::rotary_half_work(1, 2, input, &[ONE + 1], &sine, &mut direct),
        Err(Error::Bounds)
    );
    let good = Program::rotary_half_work(1, 2, input, &cosine, &sine, &mut direct).unwrap();
    let mut workspace = WorkPreflightWorkspace::empty();
    assert_eq!(
        closed(1, 2, input, &[ONE + 1], &sine)
            .preflight(1, &mut workspace)
            .err(),
        Some(WorkPreflightError::Native(Error::Bounds))
    );
    assert!(workspace.is_poisoned());
    assert_eq!(
        closed(1, 2, input, &cosine, &sine)
            .preflight(1, &mut workspace)
            .err(),
        Some(WorkPreflightError::WorkspacePoisoned)
    );
    workspace.reset();
    assert_eq!(
        closed(1, 2, input, &cosine, &sine)
            .preflight(1, &mut workspace)
            .unwrap()
            .summary()
            .work,
        good
    );
    assert_eq!(
        Construction::RotaryHalf {
            rows: 1,
            width: 2,
            cosine_len: 2,
            sine_len: 1
        }
        .bind(FactoryParameters::RotaryHalf {
            input,
            cosine: &cosine,
            sine: &sine
        })
        .err(),
        Some(WorkPreflightError::ParameterBinding)
    );
    assert_eq!((factory_attempts(), row_entries()), before);
}

#[test]
fn rotary_consuming_build_reads_borrowed_owner_without_coefficient_substitution() {
    let input = Interval::new(-32, 32).unwrap();
    let mut rows = 1;
    let mut width = 4;
    let mut cosine = vec![ONE, -ONE / 2];
    let mut sine = vec![ONE / 4, -ONE];
    let mut workspace = WorkPreflightWorkspace::empty();
    let factory = closed(rows, width, input, &cosine, &sine);
    rows = 2;
    width = 2; // Same lane count, different row/half mapping: not a replacement.
    let ticket = factory.preflight(1, &mut workspace).unwrap();
    let summary = ticket.summary();
    let mut total = WorkBudget::default();
    let admitted = CallerWorkLedger::new(
        &mut total,
        summary.work.operations(),
        summary.work.cursors().aes_work,
    )
    .unwrap()
    .admit(ticket)
    .unwrap();
    let actual = admitted.into_parts().0.build().unwrap();
    let expected = original(1, 4, input, &cosine, &sine).unwrap();
    same(actual.program(), &expected);
    assert_ne!(
        actual.program().digest(),
        original(rows, width, input, &cosine, &sine)
            .unwrap()
            .digest()
    );
    // Borrow ends at consuming build; original vectors can now change/drop.
    cosine.fill(0);
    sine.fill(0);
    let foreign = original(1, 4, input, &cosine, &sine).unwrap();
    assert_ne!(actual.program().digest(), foreign.digest());
    drop(cosine);
    drop(sine);
    same(actual.program(), &expected);
}

#[test]
fn rotary_resources_quote_actual_fixed_workspace_and_literal_control_before_construction() {
    use std::mem::size_of;
    let before = (factory_attempts(), row_entries());
    let quote = Program::rotary_half_work_resources();
    assert_eq!(quote.workspace_bytes, size_of::<SoftmaxWorkWorkspace>());
    assert_eq!(
        quote.controller_bytes,
        size_of::<(&mut SoftmaxWorkWorkspace, usize)>()
    );
    let entry = 2 * size_of::<usize>()
        + size_of::<Interval>()
        + 2 * size_of::<&[i64]>()
        + size_of::<&mut SoftmaxWorkWorkspace>()
        + 2 * size_of::<Result<usize, Error>>()
        + 2 * size_of::<Result<(), Error>>()
        + 2 * size_of::<Result<Value, Error>>()
        + 2 * size_of::<Result<WorkBudget, Error>>();
    assert_eq!(quote.entry_bytes, entry);
    let borrowed_selection = size_of::<(&[i64], usize, usize, usize, bool)>();
    assert!(
        quote.coefficient_bytes
            >= borrowed_selection + size_of::<InputBounds>() + size_of::<Option<Affine>>()
    );
    assert!(
        quote.known_peak_bytes
            >= size_of::<SoftmaxWorkWorkspace>()
                + size_of::<(&mut SoftmaxWorkWorkspace, usize)>()
                + entry
                + borrowed_selection
    );
    let native = Construction::RotaryHalf {
        rows: 1,
        width: 128,
        cosine_len: 64,
        sine_len: 64,
    }
    .quote()
    .unwrap();
    assert!(native.public_table_bytes > 0); // native backing remains separately priced
    assert_eq!((factory_attempts(), row_entries()), before);
    // Caller can reject numeric byte ceiling BEFORE constructing fixed metadata.
    let cap = quote.known_peak_bytes - 1;
    assert!(quote.known_peak_bytes > cap);
    let mut workspace = SoftmaxWorkWorkspace::empty();
    let work = Program::rotary_half_work(
        1,
        2,
        Interval::new(-32, 32).unwrap(),
        &[ONE],
        &[0],
        &mut workspace,
    )
    .unwrap();
    assert!(work.operations() > 0);
    assert_eq!((factory_attempts(), row_entries()), before);
}
