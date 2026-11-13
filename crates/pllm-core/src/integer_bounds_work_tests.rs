//! Source-only regression definitions. Adding these does not claim execution or
//! compilation, nor prove the manual source inventory complete.
use super::*;
use std::cell::Cell;
use std::ops::ControlFlow;

// Test-only future-caller arithmetic; no ledger, scan hook or production funding.
fn quote(
    rates: MatrixScanWorkRates,
    entries: u64,
    attempted_values: u64,
    completed_rows: u64,
    failures: u64,
    retirements: u64,
) -> Result<u64, Error> {
    let mut total = 0u64;
    for (count, rate) in [
        (entries, rates.append_entry),
        (attempted_values, rates.value),
        (completed_rows, rates.completed_row),
        (failures, rates.terminal_error),
        (retirements, rates.retire),
    ] {
        total = total
            .checked_add(count.checked_mul(rate).ok_or(Error::Bounds)?)
            .ok_or(Error::Bounds)?;
    }
    Ok(total)
}

#[test]
fn rates_are_repeatable_without_scan_state() {
    let before = MatrixScan::work_rates().unwrap();
    assert_eq!(MatrixScan::work_rates(), Ok(before));
    let bytes = MatrixScan::row_energy_payload_bytes(1, 2).unwrap();
    let mut scan = MatrixScan::new_row_energy(1, 2, bytes).unwrap();
    assert_eq!(MatrixScan::work_rates(), Ok(before));
    scan.append(0, &[-3, 4]).unwrap();
    assert_eq!(MatrixScan::work_rates(), Ok(before));
    assert_eq!(scan.finish_row_energy().unwrap().maximum_row_squared_energy(), 25);
    assert_eq!(MatrixScan::work_rates(), Ok(before));
}

#[test]
fn private_slot_inventory_covers_source_sites() {
    // Independent MANUAL inventory, not a sum of the implementation's arrays.
    // Extraction/local/restore: Pending twice, Option three times; extraction
    // Result and ControlFlow separately. End arithmetic has its own Option,
    // Result and ControlFlow. No public-layout or wrapper monotonicity shortcut.
    type Residual = Result<std::convert::Infallible, Error>;
    let entry = 2 * size_of::<Pending>()
        + 3 * size_of::<Option<Pending>>()
        + size_of::<Result<Pending, Error>>()
        + size_of::<ControlFlow<Residual, Pending>>()
        + size_of::<Option<usize>>()
        + size_of::<Result<usize, Error>>()
        + size_of::<ControlFlow<Residual, usize>>()
        + 2 * size_of::<Residual>()
        + 2 * size_of::<Error>()
        // Two len transfers; checked end; geometry product; ordering;
        // empty/end checks; final-length check: six four-slot envelopes.
        + 24 * size_of::<usize>()
        + 8 * size_of::<bool>()
        + size_of::<&mut MatrixScan>()
        + size_of::<&mut Option<Pending>>()
        + size_of::<Option<i64>>() // loop's final None
        + size_of::<Result<(), Error>>()
        + size_of::<()>()
        + 8 + 8 + 8 + 12 + 4 + 4 + 8;
    // One squared helper and four accumulations; helper result crosses caller
    // boundary, hence six Results but five Option and caller propagation sites.
    let value = 2 * size_of::<i64>()
        + 2 * size_of::<u64>()
        + 32 * size_of::<u128>()
        + 5 * size_of::<Option<u128>>()
        + 6 * size_of::<Result<u128, Error>>()
        + 5 * size_of::<ControlFlow<Residual, u128>>()
        + 5 * size_of::<Residual>()
        + 5 * size_of::<Error>()
        // Guard, column remainder, two index projections, increment, row test.
        + 24 * size_of::<usize>()
        + 8 * size_of::<bool>()
        + size_of::<Option<i64>>() // Some and value binding stay core-owned
        + 2 * size_of::<&Vec<u128>>()
        + size_of::<&mut Vec<u128>>()
        + size_of::<&u128>()
        + size_of::<&mut u128>()
        + 8 + 8 + 12 + 32 + 4 + 12 + 8;
    let row = 12 * size_of::<u128>() + 8 + 4; // two maxima and two resets
    let extraction_failure = size_of::<Result<Pending, Error>>()
        + size_of::<ControlFlow<Residual, Pending>>()
        + size_of::<Residual>() + 8;
    let end_failure = size_of::<Option<usize>>()
        + size_of::<Result<usize, Error>>()
        + size_of::<ControlFlow<Residual, usize>>()
        + size_of::<Residual>() + 8;
    let helper_failure = size_of::<Option<u128>>()
        + 2 * size_of::<Result<u128, Error>>()
        + size_of::<ControlFlow<Residual, u128>>()
        + size_of::<Residual>()
        + 2 * size_of::<Error>() + 12;
    let error = size_of::<Result<(), Error>>()
        + size_of::<Error>()
        + size_of::<()>()
        + 2 * size_of::<bool>() + 4
        + extraction_failure.max(end_failure).max(helper_failure).max(4);
    let retire = size_of::<Pending>()
        + size_of::<Option<Pending>>()
        + size_of::<Vec<u128>>()
        + size_of::<&mut Vec<u128>>()
        + 3 * size_of::<usize>()
        + 2 * size_of::<bool>() + 4 + 4 + 1;
    assert_eq!(MatrixScan::work_rates(), Ok(MatrixScanWorkRates {
        append_entry: u64::try_from(entry).unwrap(),
        value: u64::try_from(value).unwrap(),
        completed_row: u64::try_from(row).unwrap(),
        terminal_error: u64::try_from(error).unwrap(),
        retire: u64::try_from(retire).unwrap(),
    }));
}

#[test]
fn checked_rate_arithmetic_rejects_overflow() {
    assert_eq!(scan_work_sum(&[(usize::MAX, 2)], 0), Err(Error::Bounds));
    assert_eq!(scan_work_sum(&[(1, usize::MAX)], 1), Err(Error::Bounds));
    assert_eq!(scan_work_sum(&[(0, usize::MAX)], 0), Ok(0));
    assert_eq!(scan_work_sum(&[(1, usize::MAX)], 0), Ok(usize::MAX));
    assert_eq!(scan_work_rate(scan_work_sum(&[(1, usize::MAX)], 0)),
        u64::try_from(usize::MAX).map_err(|_| Error::Bounds));
    assert_eq!(scan_work_rate(Err(Error::Bounds)), Err(Error::Bounds));
    let rates = MatrixScan::work_rates().unwrap();
    assert_eq!(quote(rates, u64::MAX, 0, 0, 0, 0), Err(Error::Bounds));
    // Addition overflow independently of multiplication overflow.
    let large = MatrixScanWorkRates { append_entry: u64::MAX, value: 1, ..rates };
    assert_eq!(quote(large, 1, 1, 0, 0, 0), Err(Error::Bounds));
}

#[test]
fn zero_invocations_differ_from_empty_append() {
    let rates = MatrixScan::work_rates().unwrap();
    assert_eq!(quote(rates, 0, 0, 0, 0, 0), Ok(0));
    let bytes = MatrixScan::payload_bytes(1, 1).unwrap();
    let mut scan = MatrixScan::new(1, 1, bytes).unwrap();
    assert!(scan.0.is_some()); // no invocation leaves owner live
    assert_eq!(scan.append(0, &[]), Err(Error::Shape));
    assert!(scan.0.is_none()); // empty append took state, did not restore
    assert_eq!(quote(rates, 1, 0, 0, 1, 1),
        Ok(rates.append_entry + rates.terminal_error + rates.retire));
    assert_eq!(scan.append(0, &[]), Err(Error::Retired));
    // Two failed entries, still only ONE live-owner retirement.
    assert_eq!(quote(rates, 2, 0, 0, 2, 1),
        Ok(2 * rates.append_entry + 2 * rates.terminal_error + rates.retire));
    assert_eq!(scan.finish(), Err(Error::Retired));
}

#[test]
fn batch_decomposition_counts_entries_and_rows() {
    let rates = MatrixScan::work_rates().unwrap();
    let values = [1, -2, 3, -4, 0, 5];
    let mut oracle = None;
    for chunk in [1, 2, 3, 4, 6] {
        let bytes = MatrixScan::payload_bytes(2, 3).unwrap();
        let mut scan = MatrixScan::new(2, 3, bytes).unwrap();
        let mut entries = 0u64;
        for (index, batch) in values.chunks(chunk).enumerate() {
            scan.append(index * chunk, batch).unwrap();
            entries += 1;
        }
        // Expected semantic trace, not an instrumentation/count-completeness
        // claim: six accepted values, two rows even across append boundaries.
        assert_eq!(quote(rates, entries, 6, 2, 0, 1),
            Ok(entries * rates.append_entry + 6 * rates.value
                + 2 * rates.completed_row + rates.retire));
        let certificate = scan.finish().unwrap();
        assert_eq!(certificate.maximum_row_l1(), 9);
        assert_eq!(certificate.maximum_column_l1(), 8);
        assert_eq!(certificate.frobenius_squared(), 55);
        assert_eq!(certificate.maximum_row_squared_energy(), 41);
        match oracle {
            None => oracle = Some(certificate),
            Some(expected) => assert_eq!(certificate, expected),
        }
    }
}

struct LyingIterator<'a> {
    values: std::slice::Iter<'a, i64>,
    advertised: usize,
    attempted: &'a Cell<u64>,
    lengths: &'a Cell<u64>,
}

impl Iterator for LyingIterator<'_> {
    type Item = i64;

    fn next(&mut self) -> Option<Self::Item> {
        let value = self.values.next().copied();
        if value.is_some() {
            self.attempted.set(self.attempted.get() + 1);
        }
        value
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.advertised, Some(self.advertised))
    }
}

impl ExactSizeIterator for LyingIterator<'_> {
    fn len(&self) -> usize {
        self.lengths.set(self.lengths.get() + 1);
        self.advertised
    }
}

#[test]
fn iterator_length_mismatch_retires_scan() {
    let rates = MatrixScan::work_rates().unwrap();
    for (advertised, actual, expected_attempts) in [
        (1, &[1, 2][..], 2), // second yielded item rejected BEFORE arithmetic
        (2, &[1][..], 1), // final length check fails after early None
    ] {
        let attempts = Cell::new(0u64);
        let lengths = Cell::new(0u64);
        let values = LyingIterator {
            values: actual.iter(), advertised, attempted: &attempts, lengths: &lengths,
        };
        let mut scan = MatrixScan::new(1, 2,
            MatrixScan::payload_bytes(1, 2).unwrap()).unwrap();
        assert_eq!(scan.append_iter(0, values), Err(Error::Shape));
        assert_eq!(attempts.get(), expected_attempts);
        assert_eq!(lengths.get(), 2);
        assert!(scan.0.is_none());
        assert_eq!(quote(rates, 1, attempts.get(), 0, 1, 1),
            Ok(rates.append_entry + expected_attempts * rates.value
                + rates.terminal_error + rates.retire));
        assert_eq!(scan.append(0, &[1, 2]), Err(Error::Retired));
        assert_eq!(scan.finish(), Err(Error::Retired));
    }
}

// Counts are semantic caller assumptions, NOT observed ledger/deallocation
// events. Lifecycle order: attempts, initialized columns, finish attempts,
// visited columns, constructor failures, finish failures. Append order matches
// quote above; its last count is actual live-Pending retirements, exactly once.
fn lifecycle_quote(
    rates: MatrixScanLifecycleRates,
    lifecycle: [u64; 6],
    append: [u64; 5],
) -> Result<u64, Error> {
    let mut total = quote(rates.append, append[0], append[1], append[2],
        append[3], append[4])?;
    for (count, rate) in lifecycle.into_iter().zip([
        rates.construct, rates.initialize_column, rates.finish,
        rates.finish_column, rates.construct_error, rates.finish_error,
    ]) {
        total = total.checked_add(count.checked_mul(rate).ok_or(Error::Bounds)?)
            .ok_or(Error::Bounds)?;
    }
    Ok(total)
}

#[test]
fn lifecycle_overflow() {
    assert_eq!(scan_work_sum(&[(usize::MAX, 2)], 0), Err(Error::Bounds));
    assert_eq!(scan_work_sum(&[(1, usize::MAX)], 1), Err(Error::Bounds));
    assert_eq!(scan_work_rate(Err(Error::Bounds)), Err(Error::Bounds));
    let rates = MatrixScan::lifecycle_work_rates().unwrap();
    // Every new term uses checked multiplication, not a wrapping total.
    for index in 0..6 {
        let mut counts = [0; 6];
        counts[index] = u64::MAX;
        assert_eq!(lifecycle_quote(rates, counts, [0; 5]), Err(Error::Bounds));
    }
    // Distinct addition overflow: each individual multiplication fits.
    let large = MatrixScanLifecycleRates {
        construct: u64::MAX, initialize_column: 1, ..rates
    };
    assert_eq!(lifecycle_quote(large, [1, 1, 0, 0, 0, 0], [0; 5]),
        Err(Error::Bounds));
    assert_eq!(lifecycle_quote(large, [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1]),
        Err(Error::Bounds));
}

#[test]
fn lifecycle_legacy_append_identical() {
    let old = MatrixScan::work_rates().unwrap();
    let rates = MatrixScan::lifecycle_work_rates().unwrap();
    assert_eq!(rates.append, old); // exact nested five-field DTO
    assert_eq!(lifecycle_quote(rates, [0; 6], [2, 7, 3, 1, 1]),
        quote(old, 2, 7, 3, 1, 1));
    let bytes = MatrixScan::payload_bytes(1, 1).unwrap();
    let mut scan = MatrixScan::new(1, 1, bytes).unwrap();
    assert_eq!(MatrixScan::lifecycle_work_rates(), Ok(rates));
    scan.append(0, &[2]).unwrap();
    scan.finish().unwrap();
    assert_eq!(MatrixScan::lifecycle_work_rates(), Ok(rates));
    assert_eq!(MatrixScan::work_rates(), Ok(old));
}

#[test]
fn lifecycle_zero() {
    let rates = MatrixScan::lifecycle_work_rates().unwrap();
    assert_eq!(lifecycle_quote(rates, [0; 6], [0; 5]), Ok(0));
    // Pure zero arithmetic even with arbitrary unauthenticated DTO values.
    let arbitrary = MatrixScanLifecycleRates {
        append: MatrixScanWorkRates {
            append_entry: u64::MAX, value: u64::MAX,
            completed_row: u64::MAX, terminal_error: u64::MAX, retire: u64::MAX,
        },
        construct: u64::MAX, initialize_column: u64::MAX, finish: u64::MAX,
        finish_column: u64::MAX, construct_error: u64::MAX, finish_error: u64::MAX,
    };
    assert_eq!(lifecycle_quote(arbitrary, [0; 6], [0; 5]), Ok(0));
    assert_eq!(scan_work_sum(&[(0, usize::MAX)], 0), Ok(0));
}

#[test]
fn lifecycle_construct_refusal() {
    let rates = MatrixScan::lifecycle_work_rates().unwrap();
    // Geometry still precedes payload refusal; errors return no scan/Pending.
    for (rows, columns) in [(0, 1), (1, 0), (MAX_ROWS + 1, 1), (1, MAX_COLUMNS + 1)] {
        assert!(matches!(MatrixScan::new(rows, columns, 0), Err(Error::Shape)));
    }
    for (rows, columns, error) in [
        (0, usize::MAX, Error::Shape),
        (usize::MAX, 0, Error::Shape),
        (usize::MAX, 2, Error::Bounds),
    ] {
        assert!(matches!(MatrixScan::new_row_energy(rows, columns, 0),
            Err(actual) if actual == error));
    }
    let bytes = MatrixScan::payload_bytes(1, 1).unwrap();
    assert!(matches!(MatrixScan::new(1, 1, bytes - 1), Err(Error::Bounds)));
    let row_bytes = MatrixScan::row_energy_payload_bytes(1, MAX_COLUMNS + 1).unwrap();
    assert!(matches!(MatrixScan::new_row_energy(1, MAX_COLUMNS + 1, row_bytes - 1),
        Err(Error::Bounds)));
    // Refused before allocation/start: no initialized columns or live retirement.
    assert_eq!(lifecycle_quote(rates, [1, 0, 0, 0, 1, 0], [0; 5]),
        Ok(rates.construct + rates.construct_error));
    // Reserve failure is inventoried, not deterministically induced here.
    assert!(scan_work_sum(SCAN_CONSTRUCT_RESERVE_FAILURE, 8 + 4 + 1).unwrap()
        >= size_of::<std::collections::TryReserveError>() + size_of::<Vec<u128>>());
}

#[test]
fn lifecycle_no_live_finish() {
    let rates = MatrixScan::lifecycle_work_rates().unwrap();
    assert_eq!(MatrixScan(None).finish(), Err(Error::Retired));
    assert_eq!(MatrixScan(None).finish_row_energy(), Err(Error::Retired));
    assert_eq!(lifecycle_quote(rates, [0, 0, 1, 0, 0, 1], [0; 5]),
        Ok(rates.finish + rates.finish_error));
    // Incomplete live scans fail before traversal but still retire one owner.
    let bytes = MatrixScan::payload_bytes(1, 2).unwrap();
    let mut scan = MatrixScan::new(1, 2, bytes).unwrap();
    scan.append(0, &[1]).unwrap();
    assert!(scan.0.is_some());
    assert_eq!(scan.finish(), Err(Error::Shape));
    let bytes = MatrixScan::row_energy_payload_bytes(1, 2).unwrap();
    let mut row = MatrixScan::new_row_energy(1, 2, bytes).unwrap();
    row.append(0, &[1]).unwrap();
    assert_eq!(row.finish_row_energy(), Err(Error::Shape));
    assert_eq!(lifecycle_quote(rates, [0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 1]),
        Ok(rates.finish + rates.finish_error + rates.append.retire));
}

#[test]
fn lifecycle_column_cases() {
    let rates = MatrixScan::lifecycle_work_rates().unwrap();
    // Independent semantic extents, not fabricated std resize/next internals.
    assert_eq!(rates.initialize_column,
        u64::try_from(size_of::<u128>() + size_of::<usize>() + 2).unwrap());
    assert_eq!(rates.finish_column,
        u64::try_from(3 * size_of::<u128>() + size_of::<bool>()
            + size_of::<usize>() + 4).unwrap());
    for columns in [1, 3, MAX_COLUMNS] {
        let bytes = MatrixScan::payload_bytes(1, columns).unwrap();
        let mut scan = MatrixScan::new(1, columns, bytes).unwrap();
        assert_eq!(scan.0.as_ref().unwrap().columns.len(), columns);
        assert!(scan.0.as_ref().unwrap().columns.iter().all(|value| *value == 0));
        scan.append(0, &vec![1; columns]).unwrap();
        assert_eq!(scan.finish().unwrap().maximum_column_l1(), 1);
        let n = u64::try_from(columns).unwrap();
        assert_eq!(lifecycle_quote(rates, [1, n, 1, n, 0, 0], [1, n, 1, 0, 1]),
            Ok(rates.construct + n * rates.initialize_column + rates.finish
                + n * rates.finish_column + rates.append.append_entry
                + n * rates.append.value + rates.append.completed_row + rates.append.retire));
    }
    let width = MAX_COLUMNS + 1;
    let bytes = MatrixScan::row_energy_payload_bytes(1, width).unwrap();
    let mut scan = MatrixScan::new_row_energy(1, width, bytes).unwrap();
    assert_eq!(scan.0.as_ref().unwrap().columns.len(), 0);
    scan.append(0, &vec![1; width]).unwrap();
    assert_eq!(scan.finish_row_energy().unwrap().maximum_row_squared_energy(), width as u128);
    // Wide row-energy width does not become initialized/visited column backing.
    assert_eq!(lifecycle_quote(rates, [1, 0, 1, 0, 0, 0], [0; 5]),
        Ok(rates.construct + rates.finish));
}

#[test]
fn lifecycle_once_retire() {
    let rates = MatrixScan::lifecycle_work_rates().unwrap();
    let bytes = MatrixScan::payload_bytes(1, 1).unwrap();
    // Success transfers backing to IntoIter: still one live owner retirement.
    let mut success = MatrixScan::new(1, 1, bytes).unwrap();
    success.append(0, &[1]).unwrap();
    success.finish().unwrap();
    assert_eq!(lifecycle_quote(rates, [1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 1]),
        Ok(rates.construct + rates.initialize_column + rates.finish + rates.finish_column
            + rates.append.append_entry + rates.append.value
            + rates.append.completed_row + rates.append.retire));
    // Failed append retires; later failed finish must not retire again.
    let mut failed = MatrixScan::new(1, 1, bytes).unwrap();
    assert_eq!(failed.append(0, &[]), Err(Error::Shape));
    assert!(failed.0.is_none());
    assert_eq!(failed.finish(), Err(Error::Retired));
    assert_eq!(lifecycle_quote(rates, [1, 1, 1, 0, 0, 1], [1, 0, 0, 1, 1]),
        Ok(rates.construct + rates.initialize_column + rates.finish + rates.finish_error
            + rates.append.append_entry + rates.append.terminal_error + rates.append.retire));
    // Cancellation retires once; repeated cancellation has no further live owner.
    let mut cancelled = MatrixScan::new(1, 1, bytes).unwrap();
    assert!(cancelled.0.is_some());
    cancelled.cancel();
    assert!(cancelled.0.is_none());
    cancelled.cancel();
    assert_eq!(cancelled.finish(), Err(Error::Retired));
    assert_eq!(lifecycle_quote(rates, [1, 1, 1, 0, 0, 1], [0, 0, 0, 0, 1]),
        Ok(rates.construct + rates.initialize_column + rates.finish
            + rates.finish_error + rates.append.retire));
    // Live finish failure: complete row-energy scan cannot mint matrix bounds.
    let row_bytes = MatrixScan::row_energy_payload_bytes(1, 1).unwrap();
    let mut row = MatrixScan::new_row_energy(1, 1, row_bytes).unwrap();
    row.append(0, &[1]).unwrap();
    assert_eq!(row.finish(), Err(Error::Shape));
    assert_eq!(lifecycle_quote(rates, [1, 0, 1, 0, 0, 1], [1, 1, 1, 0, 1]),
        Ok(rates.construct + rates.finish + rates.finish_error
            + rates.append.append_entry + rates.append.value
            + rates.append.completed_row + rates.append.retire));
}
