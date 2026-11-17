//! Exact integer matrix and row-energy bounds, independent of numeric protocols.
//!
//! Certificates describe ordinary signed coefficients, never ring residues or
//! secret observations. Source authentication, certificate provenance and no-wrap
//! proofs belong to callers. Rounding envelopes are explicit absolute errors in
//! output-code units; an envelope of one covers either signed floor or ceiling.
use std::mem::{size_of, size_of_val};

pub const MAX_ROWS: usize = 16_384;
pub const MAX_COLUMNS: usize = 16_384;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    Shape,
    Bounds,
    Allocation,
    Retired,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Shape => "integer bound shape or ordered scan length differs",
            Self::Bounds => "integer bound arithmetic or payload limit exceeded",
            Self::Allocation => "integer bound column allocation failed",
            Self::Retired => "integer bound scan is retired",
        })
    }
}
impl std::error::Error for Error {}

fn shape(rows: usize, columns: usize) -> Result<(), Error> {
    if rows == 0 || rows > MAX_ROWS || columns == 0 || columns > MAX_COLUMNS {
        return Err(Error::Shape);
    }
    rows.checked_mul(columns).ok_or(Error::Bounds)?;
    Ok(())
}

/// Read-only complete-matrix bounds. A scan computes exact statistics; imported
/// bounds require independent caller-owned provenance and authentication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatrixCertificate {
    rows: usize,
    columns: usize,
    maximum_row_l1: u128,
    maximum_column_l1: u128,
    frobenius_squared: u128,
    maximum_row_squared_energy: u128,
}

impl MatrixCertificate {
    /// Check geometry for caller-authenticated bounds. This does not certify that
    /// the supplied statistics describe any particular source matrix. Without an
    /// exact row scan, the Frobenius bound also bounds every complete row's energy.
    pub fn from_bounds(
        rows: usize,
        columns: usize,
        maximum_row_l1: u128,
        maximum_column_l1: u128,
        frobenius_squared: u128,
    ) -> Result<Self, Error> {
        shape(rows, columns)?;
        Ok(Self {
            rows,
            columns,
            maximum_row_l1,
            maximum_column_l1,
            frobenius_squared,
            maximum_row_squared_energy: frobenius_squared,
        })
    }

    pub fn rows(self) -> usize {
        self.rows
    }
    pub fn columns(self) -> usize {
        self.columns
    }
    pub fn maximum_row_l1(self) -> u128 {
        self.maximum_row_l1
    }
    pub fn maximum_column_l1(self) -> u128 {
        self.maximum_column_l1
    }
    pub fn frobenius_squared(self) -> u128 {
        self.frobenius_squared
    }
    /// Exact for scanned matrices; conservatively Frobenius for `from_bounds`.
    pub fn maximum_row_squared_energy(self) -> u128 {
        self.maximum_row_squared_energy
    }

    /// ||W||_2^2 <= min(||W||_infinity * ||W||_1, ||W||_F^2).
    pub fn operator_squared(self) -> Result<u128, Error> {
        shape(self.rows, self.columns)?;
        // An overflowing product exceeds every representable Frobenius bound.
        Ok(self
            .maximum_row_l1
            .checked_mul(self.maximum_column_l1)
            .map_or(self.frobenius_squared, |bound| {
                bound.min(self.frobenius_squared)
            }))
    }
}

/// Exact maximum energy of a complete scanned row. No caller-supplied energy
/// constructor exists; only a completed ordered scan can create this value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RowEnergyCertificate {
    rows: usize,
    columns: usize,
    maximum_row_squared_energy: u128,
}

impl RowEnergyCertificate {
    pub fn rows(self) -> usize {
        self.rows
    }
    pub fn columns(self) -> usize {
        self.columns
    }
    pub fn maximum_row_squared_energy(self) -> u128 {
        self.maximum_row_squared_energy
    }
}

struct Pending {
    columns: Vec<u128>,
    width: usize,
    rows: usize,
    offset: usize,
    current_row: u128,
    maximum_row: u128,
    frobenius_squared: u128,
    current_row_squared: u128,
    maximum_row_squared: u128,
}

impl Pending {
    fn complete(&self) -> Result<(), Error> {
        if self.offset != self.rows * self.width
            || self.current_row != 0
            || self.current_row_squared != 0
        {
            return Err(Error::Shape);
        }
        Ok(())
    }
}

/// Ordered row-major public scan retaining at most one u128 total per column.
/// Row-energy-only mode retains constant state and never sums across rows.
/// Every append error drops its state. Finish consumes the scan, and incomplete
/// scans cannot mint certificates. Dropping the owner releases all payload.
pub struct MatrixScan(Option<Pending>);

/// Borrowed-owner observation, not an allocation promise or admission receipt.
/// Bytes exclude the inline Pending/Vec headers already inside MatrixScan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ColumnBacking {
    pub initialized_columns: usize,
    pub capacity: usize,
    pub bytes: usize,
}

/// Separate exact-constructor/query rows. Ordinary lifecycle rates are unchanged.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExactScanWorkRates {
    pub construct: u64,
    pub initialize_column: u64,
    pub observe: u64,
    pub inline_bytes: usize,
}

// The selected tier counts logical typed endpoints, not compiler stack copies.
// A call owns actual/formal arguments and returned/received results. A mapped
// propagation owns its Result, Try classification and residual separately. Plain
// returns and iterator Options do not acquire fictional Result/Try endpoints.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
struct ExactRow {
    key: &'static str,
    arguments: &'static str,
    output: &'static str,
    bytes: usize,
    calls: u64,
    controls: u64,
}

impl ExactRow {
    fn work(&self) -> Result<u64, Error> {
        let b = u64::try_from(self.bytes).map_err(|_| Error::Bounds)?;
        let e = self.calls.checked_add(self.controls).ok_or(Error::Bounds)?;
        e.checked_add(2u64.checked_mul(b).and_then(|n| n.checked_mul(e.checked_add(1)?))
            .ok_or(Error::Bounds)?).ok_or(Error::Bounds)
    }
}

fn exact_cells<A, R>(key: &'static str, calls: u64, controls: u64) -> ExactRow {
    ExactRow { key, arguments: std::any::type_name::<A>(), output: std::any::type_name::<R>(),
        bytes: 2 * size_of::<A>() + 2 * size_of::<R>(), calls, controls }
}

fn exact_item1<A, R, F: FnOnce(A) -> R>(key: &'static str, f: F) -> ExactRow {
    let mut row = exact_cells::<A, R>(key, 1, 0);
    row.bytes += 3 * size_of_val(&f); // item expression, argument and consumed self
    row
}
fn exact_item2<A, B, R, F: FnOnce(A, B) -> R>(key: &'static str, f: F) -> ExactRow {
    let mut row = exact_cells::<(A, B), R>(key, 1, 0);
    row.bytes += 3 * size_of_val(&f);
    row
}
fn exact_item3<A, B, C, R, F: FnOnce(A, B, C) -> R>(key: &'static str, f: F) -> ExactRow {
    let mut row = exact_cells::<(A, B, C), R>(key, 1, 0);
    row.bytes += 3 * size_of_val(&f);
    row
}
fn exact_try<T>(key: &'static str) -> ExactRow {
    use std::{convert::Infallible, ops::ControlFlow};
    ExactRow { key, arguments: std::any::type_name::<Result<T, Error>>(),
        output: std::any::type_name::<ControlFlow<Result<Infallible, Error>, T>>(),
        bytes: size_of::<Result<T, Error>>()
            + size_of::<ControlFlow<Result<Infallible, Error>, T>>()
            + size_of::<Result<Infallible, Error>>(), calls: 0, controls: 3 }
}

fn exact_constructor_rows() -> [ExactRow; 21] {
    use crate::exact_buffer::{layout_bytes, ExactBuffer};
    use std::ops::Range;
    [
        exact_item2("payload", MatrixScan::payload_bytes),
        exact_item2("shape", shape),
        exact_try::<()>("shape-try"),
        exact_try::<usize>("shape-product-try"),
        // shape/payload local checked products, multiply/add/and_then, predicates,
        // explicit Err/Ok and return. Each of <=32 source controls owns this pool;
        // helper endpoints above and caller propagation below are not in it.
        exact_cells::<(usize, usize, bool, Option<usize>, Error), Result<usize, Error>>(
            "preflight-local", 0, 32),
        exact_try::<usize>("payload-try"),
        exact_cells::<(usize, usize, bool), Result<MatrixScan, Error>>("limit-return", 0, 3),
        exact_item1("layout-raw", layout_bytes::<u128>),
        exact_item1("layout-map", exact_error),
        exact_try::<usize>("layout-try"),
        exact_item1("acquire-raw", ExactBuffer::<u128>::new),
        exact_item1("acquire-map", exact_error),
        exact_try::<ExactBuffer<u128>>("acquire-try"),
        exact_cells::<(usize, usize), Range<usize>>("range", 0, 1),
        exact_item1("terminal-next", <Range<usize> as Iterator>::next),
        exact_cells::<Option<usize>, ()>("terminal-none", 0, 1),
        exact_item1("finish-raw", ExactBuffer::<u128>::finish),
        exact_item1("finish-map", exact_error),
        exact_try::<Box<[u128]>>("finish-try"),
        // Three calls to the same named mapper; each owns its match/return
        // controls here, separately from the mapper's typed call endpoints.
        exact_cells::<(crate::exact_buffer::Error, Error), Error>("exact-map-bodies", 0, 12),
        // Single constructor failure-prefix reserve. ExactBuffer/std cleanup
        // implementations remain imports; these are moved local owner headers.
        exact_cells::<(ExactBuffer<u128>, Box<[u128]>, Vec<u128>, Error), Result<MatrixScan, Error>>(
            "exact-constructor-error", 0, 8),
    ]
}

fn exact_constructor_tail() -> [ExactRow; 3] {
    [exact_item1::<Box<[u128]>, Vec<u128>, _>("box-into-vec", <[u128]>::into_vec),
        exact_item3("start", MatrixScan::start),
        // start's nine field installations, Pending/Some/Self and final Ok.
        exact_cells::<(Pending, Option<Pending>, MatrixScan), Result<MatrixScan, Error>>(
            "install-return", 0, 13)]
}

fn exact_initialize_rows() -> [ExactRow; 6] {
    use crate::exact_buffer::ExactBuffer;
    [exact_item1("some-next", <std::ops::Range<usize> as Iterator>::next),
        exact_cells::<Option<usize>, usize>("some-bind", 0, 1),
        exact_item2("push-raw", ExactBuffer::<u128>::push),
        exact_item1("push-map", exact_column_push_error),
        exact_try::<()>("push-try"),
        exact_cells::<(u128, Error), Error>("push-map-body", 0, 2)]
}

fn exact_observe_rows() -> [ExactRow; 8] {
    [exact_item1("observe", MatrixScan::pending_backing),
        exact_item1("column-bytes", column_backing_bytes),
        exact_try::<usize>("column-bytes-try"),
        exact_item1("initialized-length", Vec::<u128>::len),
        exact_item1("capacity-field", Vec::<u128>::capacity),
        exact_item1("capacity-bytes-argument", Vec::<u128>::capacity),
        exact_cells::<(usize, Option<usize>, Error), Result<usize, Error>>(
            "capacity-arithmetic", 0, 6),
        exact_cells::<(&Option<Pending>, Option<ColumnBacking>, ColumnBacking), Result<Option<ColumnBacking>, Error>>(
            "observation-branches", 0, 6)]
}

fn exact_column_layout_rows() -> [ExactRow; 4] {
    [exact_item1("column-layout-raw", crate::exact_buffer::layout_bytes::<u128>),
        exact_item1("column-layout-map", exact_error),
        exact_cells::<Result<usize, crate::exact_buffer::Error>, Result<usize, Error>>(
            "column-layout-mapped-return", 0, 1),
        exact_cells::<(crate::exact_buffer::Error, Error), Error>("column-layout-map-body", 0, 4)]
}

fn exact_rows_work(rows: impl IntoIterator<Item = ExactRow>) -> Result<u64, Error> {
    rows.into_iter().try_fold(0u64, |n, row| n.checked_add(row.work()?).ok_or(Error::Bounds))
}
fn exact_rows_inline(rows: impl IntoIterator<Item = ExactRow>) -> Result<usize, Error> {
    rows.into_iter().try_fold(0usize, |n, row| n.checked_add(row.bytes).ok_or(Error::Bounds))
}

/// Inert, conservative logical source-slot rates for PARTIAL append/retire work.
/// Units are named slot bytes plus scalar-control allowances, not MIR copies,
/// physical storage, RSS, CPU fuel, allocator internals, payment or refunds.
/// This is NOT a full `MatrixScan` lifecycle quote or an execution capability.
///
/// A future caller may use checked arithmetic for
/// `entries*append_entry + attempted_values*value + completed_rows*completed_row
/// + failures*terminal_error + retirements*retire`. Reserve retirement ONCE per
/// live Pending owner, for success, failure OR cancellation, not once per failure
/// AND again per owner. An already retired append has entry/error work but no live
/// deallocation. Zero invocations cost zero; one empty append takes then drops
/// live state and needs entry + terminal_error + that owner's reserved retirement.
///
/// `attempted_values` bounds ACTUAL loop iterations, including an extra item
/// rejected by the overproduction guard; advertised length is not that bound.
/// `append_iter` accepts `ExactSizeIterator<Item = i64>`, not `IntoIterator`.
/// Caller separately prices its concrete iterator construction, both `len` calls,
/// `next`, helpers, storage/borrows and destruction (possibly arbitrary CPU).
/// Core loop classification, bindings and branches remain in these rates.
/// Rates do not bound arbitrary iterator fuel, panics or unwinding.
///
/// Column initialization, construction, block transitions, complete/finish
/// traversal, certification and formatting remain outside these rates. Retirement
/// covers only Pending/Option/Vec HEADER slots, drop control and one named
/// deallocation invocation; NOT allocator implementation work. `Vec<u128>` is
/// not Zeroizing: no element Drop or wipe is claimed. Finish's traversal and
/// iterator ownership transitions need separate pricing even when retirement
/// was reserved. Quote calculation itself must be separately prepaid by a future
/// caller; this API establishes no already-funded prefix or downstream admission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatrixScanWorkRates {
    pub append_entry: u64,
    pub value: u64,
    pub completed_row: u64,
    pub terminal_error: u64,
    pub retire: u64,
}

/// Plain, unauthenticated lifecycle metadata, not funding or execution authority.
/// Units combine logical scalar/site allowances and typed-byte extents; NOT CPU,
/// RSS, physical copies, or opaque allocator implementation prices. Both full
/// matrix and row-energy constructors/finishers fit the fixed envelopes.
///
/// Future callers must sponsor quote computation itself, then use CHECKED sums:
/// `construct_attempts*construct + actual_initialized_columns*initialize_column
/// + finish_attempts*finish + actual_visited_columns*finish_column
/// + constructor_failures*construct_error + finish_failures*finish_error`, plus
/// existing append terms and `live_retirements*append.retire`. Zero invocations
/// cost zero. Column counts mean ACTUAL Vec backing length: full scans <=16384,
/// row-energy scans zero even when their logical width exceeds MAX_COLUMNS.
/// No full-matrix shape limit is added to row-energy mode.
///
/// Reserve append.retire ONCE per actual live Pending, including backing moved
/// into IntoIter on successful finish. No second backing retirement is charged.
/// Constructor failure before start has no Pending: its local Vec cleanup is
/// covered separately by construct_error. Already-retired finish needs fixed
/// finish/error terms only. Iterator headers belong to finish; opaque allocation,
/// deallocation, panics/unwinding and formatting remain outside these rates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatrixScanLifecycleRates {
    pub append: MatrixScanWorkRates,
    pub construct: u64,
    pub initialize_column: u64,
    pub finish: u64,
    pub finish_column: u64,
    pub construct_error: u64,
    pub finish_error: u64,
}

// These are actual private layouts, not public stand-ins or assumptions that
// Option/Result/ControlFlow sizes are monotone. Sequential source sites receive
// separate logical allowances; overlapping payloads are deliberately retained.
type ScanResidual = Result<std::convert::Infallible, Error>;
type ScanFlow<T> = std::ops::ControlFlow<ScanResidual, T>;

const SCAN_ENTRY_SLOTS: &[(usize, usize)] = &[
    (2, size_of::<Pending>()), // take's local owner; successful restore operand
    (3, size_of::<Option<Pending>>()), // self.0, take output, Some restore
    (1, size_of::<Result<Pending, Error>>()), // take().ok_or
    (1, size_of::<ScanFlow<Pending>>()), // extraction ? classification
    (1, size_of::<Option<usize>>()), // checked end
    (1, size_of::<Result<usize, Error>>()), // end.ok_or
    (1, size_of::<ScanFlow<usize>>()), // end ? classification
    (2, size_of::<ScanResidual>()), // two ? residual sites
    (2, size_of::<Error>()), // Retired and Shape ok_or operands
    // Six groups of four operand/result/binding slots: two len results/transfers,
    // checked end, geometry product, order comparison, empty/end comparisons,
    // final length comparison. Conservative spare slots within each group.
    (24, size_of::<usize>()),
    // Four comparisons, two short-circuit continuations, loop None/exit control.
    (8, size_of::<bool>()),
    (1, size_of::<&mut MatrixScan>()), // append_iter receiver
    (1, size_of::<&mut Option<Pending>>()), // take/restore owner borrow
    (1, size_of::<Option<i64>>()), // final next None classification, not next work
    (1, size_of::<Result<(), Error>>()), // successful return
    (1, size_of::<()>()), // Ok payload
];
// Beyond slots: take/ok_or/? 8; checked-end/ok_or/? 8; two len
// call-boundary bindings/branches 2*4; short-circuit geometry/order 12;
// loop final None 4; final-length branch 4; restore/Ok 8. Not sizeof-only pricing.
const SCAN_ENTRY_CONTROL: usize = 8 + 8 + 2 * 4 + 12 + 4 + 4 + 8;

const SCAN_VALUE_SLOTS: &[(usize, usize)] = &[
    (2, size_of::<i64>()), // loop binding and unsigned_abs receiver
    (2, size_of::<u64>()), // unsigned_abs result and conversion operand
    // Eight groups of four operands/result/binding: conversion, squared helper,
    // four checked accumulations, magnitude binding, square binding.
    (32, size_of::<u128>()),
    (5, size_of::<Option<u128>>()), // squared checked_mul + FOUR checked_add
    (6, size_of::<Result<u128, Error>>()), // helper return/caller + four adds
    (5, size_of::<ScanFlow<u128>>()), // squared caller ? + four adds ?
    (5, size_of::<ScanResidual>()), // same five caller propagation sites
    (5, size_of::<Error>()), // helper + four ok_or operands
    // Six four-slot groups: overproduction guard, column remainder, read index,
    // write index, offset increment, row-boundary remainder/test.
    (24, size_of::<usize>()),
    // Some classification, guard, empty test, two index checks, row test/branches.
    (8, size_of::<bool>()),
    (1, size_of::<Option<i64>>()), // next Some classification, not iterator internals
    (2, size_of::<&Vec<u128>>()), // is_empty and column read
    (1, size_of::<&mut Vec<u128>>()), // column write
    (1, size_of::<&u128>()), // indexed read projection
    (1, size_of::<&mut u128>()), // indexed write projection
];
// Some/bind/guard 8; unsigned_abs/conversion/bind 8; squared helper's
// checked_mul/ok_or/return AND caller propagation 12; four accumulations 4*8;
// empty branch 4; remainder + two index checks 12; increment/row test 8.
// Completed-row maxima/resets deliberately NOT included here.
const SCAN_VALUE_CONTROL: usize = 8 + 8 + 12 + 4 * 8 + 4 + 12 + 8;

const SCAN_ROW_SLOTS: &[(usize, usize)] = &[
    (12, size_of::<u128>()), // two maxima * four slots; two resets * two slots
];
const SCAN_ROW_CONTROL: usize = 2 * 4 + 2 * 2; // TWO max/commit; TWO zero/reset

const SCAN_ERROR_SLOTS: &[(usize, usize)] = &[
    (1, size_of::<Result<(), Error>>()), // failing append return
    (1, size_of::<Error>()), // returned Shape/Bounds/Retired
    (1, size_of::<()>()), // unit alternative, conservatively retained
    (2, size_of::<bool>()), // failing classification and return branch
];
const SCAN_ERROR_CONTROL: usize = 4; // construct/transfer Err and return
const SCAN_TAKE_FAILURE: &[(usize, usize)] = &[
    (1, size_of::<Result<Pending, Error>>()),
    (1, size_of::<ScanFlow<Pending>>()),
    (1, size_of::<ScanResidual>()),
]; // extraction ok_or/? failure
const SCAN_END_FAILURE: &[(usize, usize)] = &[
    (1, size_of::<Option<usize>>()),
    (1, size_of::<Result<usize, Error>>()),
    (1, size_of::<ScanFlow<usize>>()),
    (1, size_of::<ScanResidual>()),
]; // checked-end ok_or/? failure
const SCAN_NUMERIC_FAILURE: &[(usize, usize)] = &[
    (1, size_of::<Option<u128>>()),
    (2, size_of::<Result<u128, Error>>()),
    (1, size_of::<ScanFlow<u128>>()),
    (1, size_of::<ScanResidual>()),
    (2, size_of::<Error>()),
]; // MAX nested squared checked_mul/ok_or/return + caller ?; adds are shorter
// Terminal rate is common return + MAX single failing exit, NOT sum of exits.
// Explicit Shape exits have no helper propagation. Success-site allowances
// prepaid above may overlap failure slots/control: conservative EXTRA logical
// allowance, not exact disjoint physical CPU work. Retirement is separate.

const SCAN_RETIRE_SLOTS: &[(usize, usize)] = &[
    (1, size_of::<Pending>()), // live owner being released
    (1, size_of::<Option<Pending>>()), // taken/cleared owner container
    (1, size_of::<Vec<u128>>()), // column header drop operand (not elements)
    (1, size_of::<&mut Vec<u128>>()), // named Vec drop invocation receiver
    (3, size_of::<usize>()), // pointer extent/length/capacity control allowances
    (2, size_of::<bool>()), // live-Option and allocated-Vec drop branches
];
const SCAN_RETIRE_CONTROL: usize = 4 + 4 + 1; // owner/drop branches + named dealloc

const fn scan_work_sum(slots: &[(usize, usize)], control: usize) -> Result<usize, Error> {
    let mut total = control;
    let mut index = 0;
    while index < slots.len() {
        let (count, bytes) = slots[index];
        let product = match count.checked_mul(bytes) {
            Some(product) => product,
            None => return Err(Error::Bounds),
        };
        total = match total.checked_add(product) {
            Some(total) => total,
            None => return Err(Error::Bounds),
        };
        index += 1;
    }
    Ok(total)
}

// Inventory walks happen at constant evaluation, never against runtime state.
const SCAN_ENTRY_WORK: Result<usize, Error> = scan_work_sum(SCAN_ENTRY_SLOTS, SCAN_ENTRY_CONTROL);
const SCAN_VALUE_WORK: Result<usize, Error> = scan_work_sum(SCAN_VALUE_SLOTS, SCAN_VALUE_CONTROL);
const SCAN_ROW_WORK: Result<usize, Error> = scan_work_sum(SCAN_ROW_SLOTS, SCAN_ROW_CONTROL);
const SCAN_ERROR_WORK: Result<usize, Error> = scan_work_sum(SCAN_ERROR_SLOTS, SCAN_ERROR_CONTROL);
const SCAN_TAKE_FAILURE_WORK: Result<usize, Error> = scan_work_sum(SCAN_TAKE_FAILURE, 8);
const SCAN_END_FAILURE_WORK: Result<usize, Error> = scan_work_sum(SCAN_END_FAILURE, 8);
const SCAN_NUMERIC_FAILURE_WORK: Result<usize, Error> = scan_work_sum(SCAN_NUMERIC_FAILURE, 12);
const SCAN_RETIRE_WORK: Result<usize, Error> = scan_work_sum(SCAN_RETIRE_SLOTS, SCAN_RETIRE_CONTROL);

fn scan_work_rate(work: Result<usize, Error>) -> Result<u64, Error> {
    u64::try_from(work?).map_err(|_| Error::Bounds)
}

// Lifecycle reconciliation, exclusively against this file's source owners:
// new -> payload_bytes -> shape; new_row_energy -> row_energy_payload_bytes;
// both -> start. Full-path inventory plus row-only validation is deliberately
// additive (not a multiplier). Four-slot scalar groups allow operands, result,
// binding/transfer. No inventory purports to expose standard-library internals.
const SCAN_CONSTRUCT_SLOTS: &[(usize, usize)] = &[
    // shape: four bounds comparisons, checked product; payload: checked multiply
    // and checked add; limit comparison; row-only: two zeros, product, fixed size.
    (4 * (4 + 1 + 2 + 1 + 2 + 1 + 1), size_of::<usize>()),
    // Full path boundary arguments: constructor 3, payload 2, shape 2,
    // reserve/resize extents 2, start geometry 2 (row-only path is shorter).
    (11, size_of::<usize>()),
    (4 + 3 + 1 + 2 + 1 + 1, size_of::<bool>()), // comparisons + OR continuations
    (5, size_of::<Option<usize>>()), // products, payload multiply/add + and_then
    // shape product 1; full payload return/caller 2; row product 1;
    // row payload return/caller 2. Alternatives deliberately overlap.
    (6, size_of::<Result<usize, Error>>()),
    (4, size_of::<ScanFlow<usize>>()), // both products + both constructor payload ?
    (2, size_of::<Result<(), Error>>()), // shape return; reserve mapped result
    (2, size_of::<ScanFlow<()>>()), // payload's shape ?; reserve ?
    (6, size_of::<ScanResidual>()), // products 2, payload ? 2, shape ?, reserve ?
    (8, size_of::<Error>()), // shape/product 2, payload 1, row 2, reserve 1, limits 2
    (1, size_of::<Result<(), std::collections::TryReserveError>>()),
    (1, size_of::<&mut Vec<u128>>()), // reserve boundary
    (1, size_of::<&mut Vec<u128>>()), // resize boundary
    (4, size_of::<Vec<u128>>()), // new/local, start argument, Pending field
    (3, size_of::<usize>()), // start width/rows/offset fields
    (6, size_of::<u128>()), // resize zero operand + start's five zero fields
    (1, size_of::<Pending>()), // all nine fields assembled
    (1, size_of::<Option<Pending>>()), // Some
    (2, size_of::<MatrixScan>()), // start Self and constructor return operand
    (1, size_of::<Result<MatrixScan, Error>>()),
    (1, size_of::<()>()),
];
// shape: 4 comparisons + 3 OR + checked product/ok_or + return/propagation;
// payload: checked multiply/add + and_then/ok_or + caller propagation + limit;
// reserve: Vec.new + reserve/map_err/? + resize boundary; start: 9 fields,
// Pending/Some/Self/Ok; row-only: 2 comparisons/OR/product/ok_or/return/limit.
const SCAN_CONSTRUCT_CONTROL: usize =
    4 + 3 + 8 + 4 + 8 + 8 + 4 + 1 + 8 + 1 + 9 + 4 + 2 + 1 + 8 + 4;
const SCAN_INITIALIZE_COLUMN_SLOTS: &[(usize, usize)] = &[
    (1, size_of::<u128>()), // semantic zero-write extent, not resize copies
    (1, size_of::<usize>()), // logical initialized-column progress
];
const SCAN_INITIALIZE_COLUMN_CONTROL: usize = 2; // write + logical progress

// finish -> Pending.complete -> Vec.into_iter.max -> from_bounds -> shape.
// No core norm_bounds helper exists; compiler norm_bounds is downstream.
const SCAN_FINISH_SLOTS: &[(usize, usize)] = &[
    (2, size_of::<MatrixScan>()), // consuming receiver + cleared owner drop
    (1, size_of::<&mut Option<Pending>>()), // take
    (2, size_of::<Option<Pending>>()), // cleared field + take output
    (1, size_of::<Pending>()),
    (1, size_of::<Result<Pending, Error>>()),
    (1, size_of::<ScanFlow<Pending>>()),
    (1, size_of::<&Pending>()), // complete receiver
    (3, size_of::<Result<(), Error>>()), // complete, shape, from_bounds shape ?
    (2, size_of::<ScanFlow<()>>()), // complete ? and from_bounds shape ?
    // complete: product + offset comparison; shape: four comparisons + product.
    (4 * (1 + 1 + 4 + 1), size_of::<usize>()),
    (8, size_of::<u128>()), // complete's two accumulator comparisons, four slots each
    (3 + 2 + 1 + 4 + 3, size_of::<bool>()), // includes empty test
    (1, size_of::<&Vec<u128>>()),
    (2, size_of::<Vec<u128>>()), // field transfer + into_iter receiver
    (3, size_of::<std::vec::IntoIter<u128>>()), // creation, max receiver, header retirement
    (1, size_of::<Option<u128>>()), // max result, NOT hidden next slots
    (1, size_of::<Result<u128, Error>>()),
    (1, size_of::<ScanFlow<u128>>()),
    (1, size_of::<Option<usize>>()), // shape checked product
    (1, size_of::<Result<usize, Error>>()), // shape ok_or
    (1, size_of::<ScanFlow<usize>>()),
    (6, size_of::<ScanResidual>()), // take, complete, max, from_bounds, shape, product
    (4, size_of::<Error>()), // take/max/product operands + explicit Shape
    (6, size_of::<usize>()), // from_bounds arguments + both certificates' geometry fields
    (9, size_of::<u128>()), // 4 matrix fields, energy assignment pair, row field + transfers
    (2, size_of::<MatrixCertificate>()), // from_bounds construct + final operand
    (3, size_of::<Result<MatrixCertificate, Error>>()), // helper/?, final Ok
    (1, size_of::<ScanFlow<MatrixCertificate>>()),
    (1, size_of::<RowEnergyCertificate>()), // alternate three-field result
    (1, size_of::<Result<RowEnergyCertificate, Error>>()),
    (1, size_of::<()>()),
];
// take/ok_or/? 8; complete product/3 comparisons/2 OR/return/? 12;
// empty 4; into_iter/header transfers/max invocation/header retirement 6;
// max ok_or/? 8; shape 4 comparisons/3 OR/product/ok_or/return/? 19;
// from_bounds six fields/Ok/caller ? 11; energy assignment/final Ok 4;
// alternate row certificate three fields/Ok 4. No backing retirement here.
const SCAN_FINISH_CONTROL: usize = 8 + 12 + 4 + 6 + 8 + 19 + 11 + 4 + 4;
const SCAN_FINISH_COLUMN_SLOTS: &[(usize, usize)] = &[
    (3, size_of::<u128>()), // visited element, logical transfer, selected value
    (1, size_of::<bool>()), // at most one comparison/selection per visited column
    (1, size_of::<usize>()), // logical visited-column progress
];
// n visits and <=n comparisons conservatively cover max's n-1 comparisons.
// Semantic traversal only: no invented std::Iterator::next Option/source slots.
const SCAN_FINISH_COLUMN_CONTROL: usize = 4; // visit/transfer/selection/progress

const SCAN_CONSTRUCT_ERROR_SLOTS: &[(usize, usize)] = &[
    (1, size_of::<Result<MatrixScan, Error>>()),
    (1, size_of::<Error>()),
    (2, size_of::<bool>()),
];
const SCAN_CONSTRUCT_SHAPE_FAILURE: &[(usize, usize)] = &[
    (1, size_of::<Option<usize>>()),
    (1, size_of::<Result<usize, Error>>()), // shape product ok_or
    (1, size_of::<ScanFlow<usize>>()),
    (1, size_of::<Result<(), Error>>()), // shape return
    (1, size_of::<ScanFlow<()>>()), // payload's shape ?
    (1, size_of::<Result<usize, Error>>()), // payload return
    (1, size_of::<ScanFlow<usize>>()), // constructor's payload ?
    (3, size_of::<ScanResidual>()),
    (3, size_of::<Error>()),
];
const SCAN_CONSTRUCT_PAYLOAD_FAILURE: &[(usize, usize)] = &[
    (3, size_of::<Option<usize>>()), // multiply/add/and_then
    (1, size_of::<Result<usize, Error>>()),
    (1, size_of::<ScanFlow<usize>>()),
    (1, size_of::<ScanResidual>()),
    (1, size_of::<Error>()),
];
const SCAN_CONSTRUCT_RESERVE_FAILURE: &[(usize, usize)] = &[
    (1, size_of::<Result<(), std::collections::TryReserveError>>()),
    (1, size_of::<std::collections::TryReserveError>()), // map_err input/drop
    (1, size_of::<Error>()), // mapped core error
    (1, size_of::<Result<(), Error>>()),
    (1, size_of::<ScanFlow<()>>()),
    (1, size_of::<ScanResidual>()),
    // Local Vec cleanup BEFORE Pending exists, never a ghost live retirement.
    (1, size_of::<Vec<u128>>()),
    (1, size_of::<&mut Vec<u128>>()),
    (3, size_of::<usize>()),
    (1, size_of::<bool>()),
];
const SCAN_FINISH_ERROR_SLOTS: &[(usize, usize)] = &[
    // Both return layouts explicitly, without assuming Result-size monotonicity.
    (1, size_of::<Result<MatrixCertificate, Error>>()),
    (1, size_of::<Result<RowEnergyCertificate, Error>>()),
    (1, size_of::<Error>()),
    (2, size_of::<bool>()),
];
const SCAN_FINISH_COMPLETE_FAILURE: &[(usize, usize)] = &[
    (1, size_of::<Error>()),
    (1, size_of::<Result<(), Error>>()),
    (1, size_of::<ScanFlow<()>>()),
    (1, size_of::<ScanResidual>()),
];
const SCAN_FINISH_MAX_FAILURE: &[(usize, usize)] = &[
    (1, size_of::<Option<u128>>()),
    (1, size_of::<Result<u128, Error>>()),
    (1, size_of::<ScanFlow<u128>>()),
    (1, size_of::<ScanResidual>()),
    (1, size_of::<Error>()),
];
const SCAN_FINISH_SHAPE_FAILURE: &[(usize, usize)] = &[
    (1, size_of::<Option<usize>>()),
    (1, size_of::<Result<usize, Error>>()),
    (1, size_of::<ScanFlow<usize>>()),
    (1, size_of::<Result<(), Error>>()),
    (1, size_of::<ScanFlow<()>>()),
    (1, size_of::<Result<MatrixCertificate, Error>>()),
    (1, size_of::<ScanFlow<MatrixCertificate>>()),
    (3, size_of::<ScanResidual>()),
    (3, size_of::<Error>()),
];

impl MatrixScan {
    /// Body of exact_column_bytes, excluding its caller-owned call endpoints.
    /// Imported ExactBuffer layout internals remain their qualified contract.
    pub fn exact_column_work_rates() -> Result<(u64, usize), Error> {
        Ok((exact_rows_work(exact_column_layout_rows())?, exact_rows_inline(exact_column_layout_rows())?))
    }
    pub fn exact_work_rates() -> Result<ExactScanWorkRates, Error> {
        use crate::exact_buffer::ExactBuffer;
        let construct = exact_rows_work(exact_constructor_rows().into_iter().chain(exact_constructor_tail()))?;
        // Conservative simultaneous union of logical endpoint pools; reuse per
        // loop site, never columns times a pool. This is not physical stack ABI.
        let endpoints = exact_rows_inline(exact_constructor_rows().into_iter()
            .chain(exact_constructor_tail()).chain(exact_initialize_rows()).chain(exact_observe_rows()))?;
        Ok(ExactScanWorkRates {
            construct,
            initialize_column: exact_rows_work(exact_initialize_rows())?,
            observe: exact_rows_work(exact_observe_rows())?,
            inline_bytes: size_of::<ExactBuffer<u128>>() + size_of::<Box<[u128]>>()
                + size_of::<Vec<u128>>() + 2 * size_of::<Self>()
                + size_of::<Result<Self, Error>>() + size_of::<ColumnBacking>() + endpoints,
        })
    }
    /// Pure metadata calculation: no state reads, dummy scan or allocation.
    /// Only fixed const inventories; future caller sponsors this calculation.
    pub fn lifecycle_work_rates() -> Result<MatrixScanLifecycleRates, Error> {
        // Common Err + MAX single propagation chain, never sum failing exits.
        // Explicit shape/limit returns need only their common return + 4 control.
        let construct_failure = scan_work_sum(SCAN_CONSTRUCT_SHAPE_FAILURE, 3 * 8)?
            .max(scan_work_sum(SCAN_CONSTRUCT_PAYLOAD_FAILURE, 12)?)
            .max(scan_work_sum(SCAN_CONSTRUCT_RESERVE_FAILURE, 8 + 4 + 1)?)
            .max(4);
        let construct_error = scan_work_sum(SCAN_CONSTRUCT_ERROR_SLOTS, 4)?
            .checked_add(construct_failure).ok_or(Error::Bounds)?;
        let finish_failure = SCAN_TAKE_FAILURE_WORK?
            .max(scan_work_sum(SCAN_FINISH_COMPLETE_FAILURE, 8)?)
            .max(scan_work_sum(SCAN_FINISH_MAX_FAILURE, 8)?)
            .max(scan_work_sum(SCAN_FINISH_SHAPE_FAILURE, 3 * 8)?)
            .max(4);
        let finish_error = scan_work_sum(SCAN_FINISH_ERROR_SLOTS, 4)?
            .checked_add(finish_failure).ok_or(Error::Bounds)?;
        Ok(MatrixScanLifecycleRates {
            append: Self::work_rates()?,
            construct: scan_work_rate(scan_work_sum(SCAN_CONSTRUCT_SLOTS, SCAN_CONSTRUCT_CONTROL))?,
            initialize_column: scan_work_rate(scan_work_sum(
                SCAN_INITIALIZE_COLUMN_SLOTS, SCAN_INITIALIZE_COLUMN_CONTROL))?,
            finish: scan_work_rate(scan_work_sum(SCAN_FINISH_SLOTS, SCAN_FINISH_CONTROL))?,
            finish_column: scan_work_rate(scan_work_sum(
                SCAN_FINISH_COLUMN_SLOTS, SCAN_FINISH_COLUMN_CONTROL))?,
            construct_error: scan_work_rate(Ok(construct_error))?,
            finish_error: scan_work_rate(Ok(finish_error))?,
        })
    }

    /// Calculate inert rates without scan state, traversal, allocation or ledger.
    /// Only constant private-type inventories and checked fixed-size arithmetic
    /// are evaluated. The calculation's own work is outside the returned rates.
    pub fn work_rates() -> Result<MatrixScanWorkRates, Error> {
        let failing_exit = SCAN_TAKE_FAILURE_WORK?
            .max(SCAN_END_FAILURE_WORK?)
            .max(SCAN_NUMERIC_FAILURE_WORK?)
            .max(4); // explicit Shape branch/return, no helper slots
        let terminal = SCAN_ERROR_WORK?
            .checked_add(failing_exit)
            .ok_or(Error::Bounds)?;
        Ok(MatrixScanWorkRates {
            append_entry: scan_work_rate(SCAN_ENTRY_WORK)?,
            value: scan_work_rate(SCAN_VALUE_WORK)?,
            completed_row: scan_work_rate(SCAN_ROW_WORK)?,
            terminal_error: u64::try_from(terminal).map_err(|_| Error::Bounds)?,
            retire: scan_work_rate(SCAN_RETIRE_WORK)?,
        })
    }

    /// Preflight column payload plus owner/working-state sizes, excluding
    /// allocator bookkeeping and caller-owned input slices. No allocation occurs.
    pub fn payload_bytes(rows: usize, columns: usize) -> Result<usize, Error> {
        shape(rows, columns)?;
        columns
            .checked_mul(size_of::<u128>())
            .and_then(|n| n.checked_add(size_of::<Self>() + size_of::<Pending>()))
            .ok_or(Error::Bounds)
    }

    /// The caller reserves the explicit payload limit before this allocation.
    pub fn new(rows: usize, columns: usize, payload_limit: usize) -> Result<Self, Error> {
        if Self::payload_bytes(rows, columns)? > payload_limit {
            return Err(Error::Bounds);
        }
        let mut sums = Vec::new();
        sums.try_reserve_exact(columns)
            .map_err(|_| Error::Allocation)?;
        sums.resize(columns, 0);
        Ok(Self::start(rows, columns, sums))
    }

    /// Exact logical column acquisition on the same, untagged scan type.
    /// The caller must already cover this constructor and all failure cleanup.
    pub fn new_exact(rows: usize, columns: usize, payload_limit: usize) -> Result<Self, Error> {
        if Self::payload_bytes(rows, columns)? > payload_limit {
            return Err(Error::Bounds);
        }
        crate::exact_buffer::layout_bytes::<u128>(columns).map_err(exact_error)?;
        let mut sums = crate::exact_buffer::ExactBuffer::new(columns).map_err(exact_error)?;
        for _ in 0..columns {
            sums.push(0u128).map_err(exact_column_push_error)?;
        }
        let sums = sums.finish().map_err(exact_error)?.into_vec();
        Ok(Self::start(rows, columns, sums))
    }

    /// Observe actual capacity, including ordinary-created spare capacity.
    /// None means retired; a live row-energy scan has Some with zero bytes.
    pub fn pending_backing(&self) -> Result<Option<ColumnBacking>, Error> {
        match &self.0 {
            None => Ok(None),
            Some(pending) => Ok(Some(ColumnBacking {
                initialized_columns: pending.columns.len(),
                capacity: pending.columns.capacity(),
                bytes: column_backing_bytes(pending.columns.capacity())?,
            })),
        }
    }

    /// Exact acquisition layout exported by the actual column owner.
    pub fn exact_column_bytes(columns: usize) -> Result<usize, Error> {
        crate::exact_buffer::layout_bytes::<u128>(columns).map_err(exact_error)
    }

    /// Constant-state row-energy preflight. Unlike a full matrix certificate,
    /// this supports any nonempty checked geometry, including large vocabularies.
    /// The enclosing producer is responsible for its own total byte/work limits.
    pub fn row_energy_payload_bytes(rows: usize, columns: usize) -> Result<usize, Error> {
        if rows == 0 || columns == 0 {
            return Err(Error::Shape);
        }
        rows.checked_mul(columns).ok_or(Error::Bounds)?;
        Ok(size_of::<Self>() + size_of::<Pending>())
    }

    pub fn new_row_energy(
        rows: usize,
        columns: usize,
        payload_limit: usize,
    ) -> Result<Self, Error> {
        if Self::row_energy_payload_bytes(rows, columns)? > payload_limit {
            return Err(Error::Bounds);
        }
        Ok(Self::start(rows, columns, Vec::new()))
    }

    fn start(rows: usize, width: usize, columns: Vec<u128>) -> Self {
        Self(Some(Pending {
            columns,
            width,
            rows,
            offset: 0,
            current_row: 0,
            maximum_row: 0,
            frobenius_squared: 0,
            current_row_squared: 0,
            maximum_row_squared: 0,
        }))
    }

    pub fn append(&mut self, offset: usize, values: &[i64]) -> Result<(), Error> {
        self.append_iter(offset, values.iter().copied())
    }

    // FixedFile decodes canonical byte chunks through this same scan without an
    // extra decoded buffer. Check iterator length too, before restoring state.
    pub(crate) fn append_iter(
        &mut self,
        offset: usize,
        values: impl ExactSizeIterator<Item = i64>,
    ) -> Result<(), Error> {
        let mut pending = self.0.take().ok_or(Error::Retired)?;
        let end = offset.checked_add(values.len()).ok_or(Error::Shape)?;
        if values.len() == 0 || offset != pending.offset || end > pending.rows * pending.width {
            return Err(Error::Shape);
        }
        for value in values {
            if pending.offset == end {
                return Err(Error::Shape);
            }
            let magnitude = u128::from(value.unsigned_abs());
            let square = squared(magnitude)?;
            pending.current_row_squared = pending
                .current_row_squared
                .checked_add(square)
                .ok_or(Error::Bounds)?;
            if !pending.columns.is_empty() {
                let column = pending.offset % pending.width;
                pending.columns[column] = pending.columns[column]
                    .checked_add(magnitude)
                    .ok_or(Error::Bounds)?;
                pending.current_row = pending
                    .current_row
                    .checked_add(magnitude)
                    .ok_or(Error::Bounds)?;
                pending.frobenius_squared = pending
                    .frobenius_squared
                    .checked_add(square)
                    .ok_or(Error::Bounds)?;
            }
            pending.offset += 1;
            if pending.offset % pending.width == 0 {
                pending.maximum_row = pending.maximum_row.max(pending.current_row);
                pending.current_row = 0;
                pending.maximum_row_squared =
                    pending.maximum_row_squared.max(pending.current_row_squared);
                pending.current_row_squared = 0;
            }
        }
        if pending.offset != end {
            return Err(Error::Shape);
        }
        self.0 = Some(pending);
        Ok(())
    }

    pub fn cancel(&mut self) {
        self.0 = None;
    }

    pub fn finish(mut self) -> Result<MatrixCertificate, Error> {
        let pending = self.0.take().ok_or(Error::Retired)?;
        pending.complete()?;
        // Row-only scans cannot fabricate column statistics or Frobenius sums.
        if pending.columns.is_empty() {
            return Err(Error::Shape);
        }
        let mut certificate = MatrixCertificate::from_bounds(
            pending.rows,
            pending.width,
            pending.maximum_row,
            pending.columns.into_iter().max().ok_or(Error::Shape)?,
            pending.frobenius_squared,
        )?;
        certificate.maximum_row_squared_energy = pending.maximum_row_squared;
        Ok(certificate)
    }

    pub fn finish_row_energy(mut self) -> Result<RowEnergyCertificate, Error> {
        let pending = self.0.take().ok_or(Error::Retired)?;
        pending.complete()?;
        Ok(RowEnergyCertificate {
            rows: pending.rows,
            columns: pending.width,
            maximum_row_squared_energy: pending.maximum_row_squared,
        })
    }
}

fn column_backing_bytes(capacity: usize) -> Result<usize, Error> {
    capacity.checked_mul(size_of::<u128>()).ok_or(Error::Bounds)
}

fn exact_error(error: crate::exact_buffer::Error) -> Error {
    match error {
        crate::exact_buffer::Error::Allocation => Error::Allocation,
        crate::exact_buffer::Error::Overflow | crate::exact_buffer::Error::Incomplete => Error::Bounds,
    }
}
fn exact_column_push_error(_: u128) -> Error { Error::Bounds }

#[cfg(test)]
mod exact_column_tests {
    use super::*;

    #[test]
    fn exact_columns_actual_capacity_numeric_and_retirement() {
        let mut exact = MatrixScan::new_exact(2, 3, usize::MAX).unwrap();
        assert_eq!(exact.pending_backing().unwrap(), Some(ColumnBacking {
            initialized_columns: 3, capacity: 3, bytes: 48,
        }));
        let mut ordinary = MatrixScan::new(2, 3, usize::MAX).unwrap();
        for (offset, values) in [(0, &[1, -2][..]), (2, &[3, -4, 0, 5][..])] {
            exact.append(offset, values).unwrap();
            ordinary.append(offset, values).unwrap();
        }
        assert_eq!(exact.finish().unwrap(), ordinary.finish().unwrap());
        let mut columns = Vec::with_capacity(11);
        columns.resize(3, 0);
        let mut larger = MatrixScan::start(1, 3, columns);
        assert_eq!(larger.pending_backing().unwrap().unwrap().bytes,
            larger.0.as_ref().unwrap().columns.capacity() * 16);
        assert!(larger.pending_backing().unwrap().unwrap().capacity >= 11);
        larger.cancel();
        assert_eq!(larger.pending_backing().unwrap(), None);
        let zero = MatrixScan::new_row_energy(1, 3, usize::MAX).unwrap();
        assert_eq!(zero.pending_backing().unwrap().unwrap().bytes, 0);
        assert_eq!(column_backing_bytes(usize::MAX), Err(Error::Bounds));
        assert!(matches!(MatrixScan::new_exact(0, 1, usize::MAX), Err(Error::Shape)));
        assert!(matches!(MatrixScan::new_exact(1, 1, 0), Err(Error::Bounds)));
    }

    #[test]
    fn exact_column_source_rows_have_independent_typed_u128_oracle() {
        use std::{any::TypeId, convert::Infallible, ops::ControlFlow};
        use crate::exact_buffer::ExactBuffer;
        fn actual_output<F, T: 'static>(_: F) -> TypeId
        where F: for<'a> FnOnce(&'a MatrixScan) -> Result<T, Error>
        {
            TypeId::of::<T>()
        }
        use crate::exact_buffer::Error as Raw;
        use std::{any::type_name, ops::Range};
        // Handwritten endpoint roster. No production row generator supplies
        // expected types, keys, counts, extents or u128 arithmetic.
        fn call<A, R>(key: &'static str) -> ExactRow {
            ExactRow { key, arguments: type_name::<A>(), output: type_name::<R>(),
                bytes: 2 * size_of::<A>() + 2 * size_of::<R>(), calls: 1, controls: 0 }
        }
        fn local<A, R>(key: &'static str, controls: u64) -> ExactRow {
            let mut row = call::<A, R>(key);
            row.calls = 0; row.controls = controls; row
        }
        fn propagation<T>(key: &'static str) -> ExactRow {
            ExactRow { key, arguments: type_name::<Result<T, Error>>(),
                output: type_name::<ControlFlow<Result<Infallible, Error>, T>>(),
                bytes: size_of::<Result<T, Error>>()
                    + size_of::<ControlFlow<Result<Infallible, Error>, T>>()
                    + size_of::<Result<Infallible, Error>>(), calls: 0, controls: 3 }
        }
        fn oracle(rows: &[ExactRow]) -> u128 {
            rows.iter().map(|r| { let e = u128::from(r.calls) + u128::from(r.controls);
                e + 2 * r.bytes as u128 * (e + 1) }).sum()
        }
        let rows = [
            call::<(usize, usize), Result<usize, Error>>("payload"),
            call::<(usize, usize), Result<(), Error>>("shape"), propagation::<()>("shape-try"),
            propagation::<usize>("shape-product-try"),
            local::<(usize, usize, bool, Option<usize>, Error), Result<usize, Error>>("preflight-local", 32),
            propagation::<usize>("payload-try"),
            local::<(usize, usize, bool), Result<MatrixScan, Error>>("limit-return", 3),
            call::<usize, Result<usize, Raw>>("layout-raw"), call::<Raw, Error>("layout-map"),
            propagation::<usize>("layout-try"),
            call::<usize, Result<ExactBuffer<u128>, Raw>>("acquire-raw"),
            call::<Raw, Error>("acquire-map"), propagation::<ExactBuffer<u128>>("acquire-try"),
            local::<(usize, usize), Range<usize>>("range", 1),
            call::<&mut Range<usize>, Option<usize>>("terminal-next"),
            local::<Option<usize>, ()>("terminal-none", 1),
            call::<ExactBuffer<u128>, Result<Box<[u128]>, Raw>>("finish-raw"),
            call::<Raw, Error>("finish-map"), propagation::<Box<[u128]>>("finish-try"),
            local::<(Raw, Error), Error>("exact-map-bodies", 12),
            local::<(ExactBuffer<u128>, Box<[u128]>, Vec<u128>, Error), Result<MatrixScan, Error>>(
                "exact-constructor-error", 8),
            call::<Box<[u128]>, Vec<u128>>("box-into-vec"),
            call::<(usize, usize, Vec<u128>), MatrixScan>("start"),
            local::<(Pending, Option<Pending>, MatrixScan), Result<MatrixScan, Error>>("install-return", 13),
        ];
        let initialize = [call::<&mut Range<usize>, Option<usize>>("some-next"),
            local::<Option<usize>, usize>("some-bind", 1),
            call::<(&mut ExactBuffer<u128>, u128), Result<(), u128>>("push-raw"),
            call::<u128, Error>("push-map"), propagation::<()>("push-try"),
            local::<(u128, Error), Error>("push-map-body", 2)];
        let observe = [call::<&MatrixScan, Result<Option<ColumnBacking>, Error>>("observe"),
            call::<usize, Result<usize, Error>>("column-bytes"), propagation::<usize>("column-bytes-try"),
            call::<&Vec<u128>, usize>("initialized-length"),
            call::<&Vec<u128>, usize>("capacity-field"),
            call::<&Vec<u128>, usize>("capacity-bytes-argument"),
            local::<(usize, Option<usize>, Error), Result<usize, Error>>("capacity-arithmetic", 6),
            local::<(&Option<Pending>, Option<ColumnBacking>, ColumnBacking), Result<Option<ColumnBacking>, Error>>(
                "observation-branches", 6)];
        let column_layout = [call::<usize, Result<usize, Raw>>("column-layout-raw"),
            call::<Raw, Error>("column-layout-map"),
            local::<Result<usize, Raw>, Result<usize, Error>>("column-layout-mapped-return", 1),
            local::<(Raw, Error), Error>("column-layout-map-body", 4)];
        assert_eq!(exact_column_layout_rows(), column_layout);
        let layout_quote = MatrixScan::exact_column_work_rates().unwrap();
        assert_eq!(u128::from(layout_quote.0), oracle(&column_layout));
        assert_eq!(layout_quote.1, column_layout.iter().map(|r| r.bytes).sum::<usize>());
        assert_eq!(exact_constructor_rows().into_iter().chain(exact_constructor_tail()).collect::<Vec<_>>(), rows);
        assert_eq!(exact_initialize_rows(), initialize);
        assert_eq!(exact_observe_rows(), observe);
        let expected = oracle(&rows);
        let rates = MatrixScan::exact_work_rates().unwrap();
        assert_eq!(rates.construct as u128, expected);
        assert_eq!(rates.initialize_column as u128, oracle(&initialize));
        assert_eq!(rates.observe as u128, oracle(&observe));
        assert_eq!(rates.inline_bytes as u128,
            (size_of::<ExactBuffer<u128>>() + size_of::<Box<[u128]>>() + size_of::<Vec<u128>>()
                + 2 * size_of::<MatrixScan>() + size_of::<Result<MatrixScan, Error>>()
                + size_of::<ColumnBacking>()) as u128
                + rows.iter().chain(&initialize).chain(&observe).map(|r| r.bytes as u128).sum::<u128>());
        for index in 0..rows.len() { // includes every mapper and terminal next/None
            let mut missing = rows.to_vec(); missing.remove(index);
            let mut duplicate = rows.to_vec(); duplicate.push(rows[index].clone());
            assert_ne!(expected, oracle(&missing)); assert_ne!(expected, oracle(&duplicate));
        }
        for index in 0..initialize.len() {
            let mut missing = initialize.to_vec(); missing.remove(index);
            assert_ne!(oracle(&initialize), oracle(&missing));
        }
        let mut wrong = rows.clone();
        wrong[7] = call::<usize, Result<usize, Error>>("layout-raw");
        assert_ne!(wrong, rows); // nominal error identity, even if layouts coincide
        wrong = rows.clone(); wrong[21] = call::<Box<[u128]>, Result<Vec<u128>, Error>>("box-into-vec");
        assert_ne!(wrong, rows);
        #[repr(transparent)]
        struct WrongRaw(Raw);
        #[repr(transparent)]
        struct WrongVec(Vec<u128>);
        let wrong_map = call::<WrongRaw, Error>("layout-map");
        assert_eq!(wrong_map.bytes, rows[8].bytes);
        assert_ne!(wrong_map, rows[8]);
        let wrong_plain = call::<Box<[u128]>, WrongVec>("box-into-vec");
        assert_eq!(wrong_plain.bytes, rows[21].bytes);
        assert_ne!(wrong_plain, rows[21]);
        let mut wrong_push = initialize.clone();
        wrong_push[2] = call::<(&mut ExactBuffer<u128>, u128), Result<(), Error>>("push-raw");
        assert_ne!(wrong_push, initialize);
        // Same-layout semantic mutations need nominal evidence, not an ABI
        // inequality. Infer T from the actual method item, without a pointer
        // cast, replacement closure, invocation or guessed callable layout.
        let observed = actual_output(MatrixScan::pending_backing);
        assert_eq!(observed, TypeId::of::<Option<ColumnBacking>>());
        assert_ne!(observed, TypeId::of::<ColumnBacking>());
    }
}

/// Exact ceiling of the square root over the entire u128 domain.
pub fn ceil_sqrt(value: u128) -> u128 {
    let floor = value.isqrt();
    floor + u128::from(floor * floor != value)
}

fn squared(radius: u128) -> Result<u128, Error> {
    radius.checked_mul(radius).ok_or(Error::Bounds)
}

/// Exact squared energy of ordinary signed codes, checked even for i64::MIN.
/// An empty vector has zero energy; no scan state or allocation is involved.
pub fn row_energy(values: &[i64]) -> Result<u128, Error> {
    values.iter().try_fold(0u128, |sum, value| {
        sum.checked_add(squared(u128::from(value.unsigned_abs()))?)
            .ok_or(Error::Bounds)
    })
}

/// Bound a row after division by 2^shift and a per-coordinate absolute error of
/// at most `rounding_units`. Shift is 0..=127. Zero error represents exact scaling;
/// one conservatively covers both signed floor and signed ceiling (<1 error).
pub fn rescaled_energy(
    raw_squared: u128,
    coordinates: usize,
    shift: u32,
    rounding_units: u128,
) -> Result<u128, Error> {
    if coordinates == 0 || coordinates > MAX_COLUMNS {
        return Err(Error::Shape);
    }
    if shift >= 128 {
        return Err(Error::Bounds);
    }
    let rounding = ceil_sqrt(coordinates as u128)
        .checked_mul(rounding_units)
        .ok_or(Error::Bounds)?;
    squared(
        ceil_sqrt(raw_squared)
            .div_ceil(1u128 << shift)
            .checked_add(rounding)
            .ok_or(Error::Bounds)?,
    )
}

/// Bound a complete output row of a matrix stored as [output, input], with one
/// caller-specified final-coordinate rounding envelope after the complete sum.
pub fn linear_energy(
    input_squared: u128,
    matrix: MatrixCertificate,
    shift: u32,
    rounding_units: u128,
) -> Result<u128, Error> {
    let raw = input_squared
        .checked_mul(matrix.operator_squared()?)
        .ok_or(Error::Bounds)?;
    rescaled_energy(raw, matrix.rows, shift, rounding_units)
}

/// Minkowski bound for addition of two equal-width rows.
pub fn sum_energy(left: u128, right: u128) -> Result<u128, Error> {
    squared(
        ceil_sqrt(left)
            .checked_add(ceil_sqrt(right))
            .ok_or(Error::Bounds)?,
    )
}

/// Bound half-split two-coordinate rotations with operator norm
/// sqrt(c^2+s^2)/2^shift. The envelope is per FINAL coordinate: use two for two
/// separately floored/ceiled product terms, or one for a single final truncation.
pub fn rotary_energy(
    input_squared: u128,
    cos: &[i64],
    sin: &[i64],
    shift: u32,
    rounding_units: u128,
) -> Result<u128, Error> {
    if cos.is_empty() || cos.len() != sin.len() || cos.len() > MAX_ROWS / 2 {
        return Err(Error::Shape);
    }
    let mut maximum = 0u128;
    for (&c, &s) in cos.iter().zip(sin) {
        maximum = maximum.max(
            squared(u128::from(c.unsigned_abs()))?
                .checked_add(squared(u128::from(s.unsigned_abs()))?)
                .ok_or(Error::Bounds)?,
        );
    }
    let raw = input_squared.checked_mul(maximum).ok_or(Error::Bounds)?;
    rescaled_energy(raw, 2 * cos.len(), shift, rounding_units)
}

/// Cauchy--Schwarz radius for a complete untruncated dot. Ring/no-wrap admission
/// remains a separate caller obligation.
pub fn dot_radius(left_squared: u128, right_squared: u128) -> Result<u128, Error> {
    Ok(ceil_sqrt(
        left_squared
            .checked_mul(right_squared)
            .ok_or(Error::Bounds)?,
    ))
}

/// Absolute update-error codes, each representing its value divided by 2^f.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReciprocalStepError {
    pub delta_d: u128,
    pub delta_n: u128,
}

/// Public reference envelope only: f=0..=127, m=0..=8 are not evaluated settings.
/// All numeric codes represent value / 2^fraction_bits. Every inactive step must
/// have both fields zero; this also applies to all eight steps when m=0.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReciprocalBoundSpec {
    pub fraction_bits: u8,
    pub iterations: u8,
    pub xmin: u128,
    pub xmax: u128,
    pub e0: u128,
    pub w0: u128,
    pub steps: [ReciprocalStepError; 8],
}

/// Conditional nonstrict absolute error bound, not an execution capability.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReciprocalBound {
    code: u128,
    fraction_bits: u8,
}

impl ReciprocalBound {
    pub fn code(self) -> u128 {
        self.code
    }

    pub fn fraction_bits(self) -> u8 {
        self.fraction_bits
    }
}

// Private caller contract: divisor > 0. Quotient/remainder avoid n+d-1 overflow.
// Increment cannot overflow: remainder!=0 implies divisor>=2 and q<u128::MAX;
// divisor=1 instead gives remainder=0. Keep the increment checked nonetheless.
fn reciprocal_ceil(numerator: u128, divisor: u128) -> Result<u128, Error> {
    let quotient = numerator / divisor;
    let remainder = numerator % divisor;
    quotient
        .checked_add(u128::from(remainder != 0))
        .ok_or(Error::Bounds)
}

/// Bound simultaneous old-state Goldschmidt updates for a fixed positive x:
/// d' = d*(2-d)+eps_d, n' = n*(2-d)+eps_n, |eps_*| <= delta_*/S.
/// The caller must independently justify xmin/S <= x <= xmax/S,
/// |1-d0| <= e0/S and |d0-x*n0| <= w0/S, where S=2^fraction_bits.
/// For OLD E,W, propagate E'=ceil(E*E/S)+delta_d and
/// W'=ceil((S+E)*W/S)+delta_d+ceil(xmax*delta_n/S).
/// The returned B/S bounds |n-1/x| nonstrictly, B=ceil((E+W)*S/xmin).
/// Indeed 1-d'=(1-d)^2-eps_d and
/// d'-x*n'=(2-d)*(d-x*n)+eps_d-x*eps_n, with |2-d|<=1+E/S.
/// Finally |n-1/x|<=((E+W)/S)/x<=(E+W)/xmin before outward rounding.
/// Supplied premises are not authenticated: this API has no x,d0,n0 inputs.
/// Refresh/gain are excluded; a post-refresh invocation needs independently
/// justified new initial premises, never an automatic error reset. No convergence,
/// tolerance, paper iteration count or numerical profile is selected.
///
/// Every invalid domain, nonzero inactive field or intermediate overflow returns
/// Error::Bounds, even when a reduced answer would fit. Only complete success
/// constructs a bound; the borrowed specification is immutable on every path.
///
/// External admission, as with MatrixScan: BEFORE entry the caller must fund
/// validation, full work, input/original backing and output/error overlap. This
/// helper neither authenticates funding/premises nor quotes/adopts a runtime.
/// There is no heap allocation, history, callback or imported capability.
///
/// Source-level work inventory (successful m-step path; errors short-circuit):
/// checked multiply 3m+1; division AND remainder each 3m+1; checked ceiling
/// increment 3m+1; other checked add 4m+1; checked shift 1. Scalar validation
/// has <=4 comparisons, then <=8 borrowed step reads/classifications, <=8 index
/// validity comparisons and <=16 zero comparisons (short-circuit OR). Validation
/// has <=9 loop termination checks; recurrence <=8 iterations/9 termination checks.
/// A single prefix slice bounds check is justified by m<=8. No recurrence indexing.
/// Each step has 7 checked-arithmetic Option->Result/? sites and 3 helper /?
/// sites; finalization has 2 arithmetic sites and 1 helper /?, plus entry shift.
/// Thus main has 10m+4 propagated Result sites on success; each of 3m+1 helper
/// calls adds one Option->Result return. Two explicit validation Err returns and
/// one final Ok return exist; helper returns its checked increment Result.
/// Branches also include short-circuit validation and helper remainder!=0; each
/// checked operation may exit early. Local E,W update only after all step checks.
/// Named assignments: entry binds S,E,W; each recurrence binds nine magnitudes
/// then assigns E,W; finalization binds sum,numerator,code. Each helper binds q,r.
///
/// Remaining source-control inventory (successful path):
/// - Two slice-iterator constructions, one enumerate construction; validation
///   next 9 / recurrence next m+1, separately Some/None classifications 9 / m+1;
///   bind 8 validation pairs and m steps.
/// - Scalar OR 1..4 comparisons, <=3 OR continuations. Validation 8 index
///   comparisons / AND RHS 8-m, inactive fields 2(8-m) comparisons on success
///   and short-circuit OR on error. Helper: one remainder comparison and u128
///   Bool conversion per call.
/// - usize::from 9, u32::from 1; prefix bounds check 1. Iterator internals are
///   not additional SOURCE indexing.
/// - Arithmetic Options 7m+3 main, 3m+1 helpers classified by ok_or. Main
///   10m+4 propagated Result classifications; helper returns Result without ?;
///   iterator Option classifications are separate.
/// - Bind S/E/W once, nine magnitudes/step plus 2 commits, 3 final magnitudes,
///   helper q/r. Two explicit validation Err sites, propagation exits, helper
///   returns and final Ok are distinct. Accessors return one copied field.
///
/// All error prefixes use independent maximum m=8: validation next <=9,
/// recurrence next <=9, 8 pair/step bindings, 4 scalar / 8 index / 16 inactive
/// comparisons. Invalid supplied m never determines work. Arithmetic follows
/// validation; errors omit suffixes.
///
/// Conservative LOGICAL source-slot expression; aliases are NOT new Rust types
/// or API. With sz<T> = size_of::<T>(), the logical envelope L is:
///
/// ```text
/// I = std::slice::Iter<'a, ReciprocalStepError>
/// J = std::iter::Enumerate<I>
/// P = (usize, &'a ReciprocalStepError)
/// A = Option<u128>
/// R = Result<u128, Error>
/// O = Result<ReciprocalBound, Error>
/// L = sz<&ReciprocalBoundSpec>
///   + sz<&[ReciprocalStepError; 8]> + sz<&[ReciprocalStepError]>
///   + 3*sz<I> + sz<J> + sz<P> + sz<Option<P>>
///   + 2*sz<&ReciprocalStepError> + sz<Option<&ReciprocalStepError>>
///   + sz<std::ops::RangeTo<usize>> + 3*sz<usize>
///   + 4*sz<u8> + sz<u32> + 13*sz<bool>
///   + 76*sz<u128> + 11*sz<A> + 15*sz<R>
///   + 17*sz<O> + 27*sz<Error> + sz<ReciprocalBound>
/// ```
///
/// The 76 magnitudes cover 20 named main/helper argument/local slots, 3 operand/
/// result slots for each of 16 conservative expression sites, and 8 extra helper
/// call argument slots. The 16 sites are 11 distinct checked-arithmetic
/// expressions, 4 ceiling-call source sites and 1 final bound construction:
/// source-audited conservative operand/result allowance, NOT a new fee. Sum-type
/// allowances cover checked operations, helper returns and propagated error
/// outputs, intentionally conservatively overlapping payloads. Sequential
/// iterations/helper calls reuse SOURCE-SITE slots.
///
/// Spec view extent size_of::<ReciprocalBoundSpec>() is contained in, NOT billed
/// additionally over, caller-charged FULL original backing; borrow references
/// are separate logical slots. Caller entry/work/original backing must be funded
/// externally BEFORE validation. Source-call slots EXCLUDE private standard
/// library internals; fixed width/bounded iteration do not establish physical
/// costs. No stack/RSS/payment/tariff/quote API/admitted-execution claim follows.
/// Independent caller binding is still required.
pub fn reciprocal_bound(spec: &ReciprocalBoundSpec) -> Result<ReciprocalBound, Error> {
    if spec.fraction_bits > 127 || spec.iterations > 8 || spec.xmin == 0 || spec.xmin > spec.xmax {
        return Err(Error::Bounds);
    }
    for (index, step) in spec.steps.iter().enumerate() {
        if index >= usize::from(spec.iterations) && (step.delta_d != 0 || step.delta_n != 0) {
            return Err(Error::Bounds);
        }
    }
    // This checked shift cannot fail after f<=127; S and xmin are positive.
    let scale = 1u128
        .checked_shl(u32::from(spec.fraction_bits))
        .ok_or(Error::Bounds)?;
    let mut e = spec.e0;
    let mut w = spec.w0;
    for step in &spec.steps[..usize::from(spec.iterations)] {
        let square = e.checked_mul(e).ok_or(Error::Bounds)?;
        let next_e = reciprocal_ceil(square, scale)?
            .checked_add(step.delta_d)
            .ok_or(Error::Bounds)?;
        // After E^2 succeeds, E<=2^64-1, so S+E cannot overflow for S<=2^127.
        let gain = scale.checked_add(e).ok_or(Error::Bounds)?;
        let product = gain.checked_mul(w).ok_or(Error::Bounds)?;
        let propagated_w = reciprocal_ceil(product, scale)?;
        let with_delta = propagated_w
            .checked_add(step.delta_d)
            .ok_or(Error::Bounds)?;
        let numerator_error = spec
            .xmax
            .checked_mul(step.delta_n)
            .ok_or(Error::Bounds)?;
        let scaled_error = reciprocal_ceil(numerator_error, scale)?;
        let next_w = with_delta.checked_add(scaled_error).ok_or(Error::Bounds)?;
        e = next_e;
        w = next_w;
    }
    let sum = e.checked_add(w).ok_or(Error::Bounds)?;
    let numerator = sum.checked_mul(scale).ok_or(Error::Bounds)?;
    let code = reciprocal_ceil(numerator, spec.xmin)?;
    Ok(ReciprocalBound {
        code,
        fraction_bits: spec.fraction_bits,
    })
}

#[cfg(test)]
mod reciprocal_tests;

#[cfg(test)]
#[path = "integer_bounds_work_tests.rs"]
mod integer_bounds_work_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigUint;

    fn scan(rows: usize, columns: usize, values: &[i64], chunk: usize) -> MatrixCertificate {
        let bytes = MatrixScan::payload_bytes(rows, columns).unwrap();
        let mut scan = MatrixScan::new(rows, columns, bytes).unwrap();
        for (i, values) in values.chunks(chunk).enumerate() {
            scan.append(i * chunk, values).unwrap();
        }
        scan.finish().unwrap()
    }

    fn energy(values: &[i64]) -> u128 {
        values
            .iter()
            .map(|v| (*v as i128 * *v as i128) as u128)
            .sum()
    }

    #[test]
    fn matrix_statistics_are_exact_across_chunks_and_extended_geometry() {
        let values = [1, -2, 3, -4, 0, 5];
        let certificate = scan(2, 3, &values, 4);
        assert_eq!((certificate.rows(), certificate.columns()), (2, 3));
        assert_eq!(certificate.maximum_row_l1(), 9);
        assert_eq!(certificate.maximum_column_l1(), 8);
        assert_eq!(certificate.frobenius_squared(), 55);
        assert_eq!(certificate.maximum_row_squared_energy(), 41);
        assert_eq!(certificate.operator_squared(), Ok(55));
        for chunk in [1, 2, 3, 6] {
            assert_eq!(scan(2, 3, &values, chunk), certificate);
        }
        assert_eq!(scan(2, 3, &values.map(|v| -v), 5), certificate);
        let tall = scan(MAX_ROWS, 1, &vec![1; MAX_ROWS], 127);
        assert_eq!(tall.maximum_column_l1(), MAX_ROWS as u128);
        assert_eq!(tall.operator_squared(), Ok(MAX_ROWS as u128));
        assert_eq!(tall.maximum_row_squared_energy(), 1);
        let declared = MatrixCertificate::from_bounds(2, 3, 9, 8, 55).unwrap();
        assert_eq!(declared.maximum_row_squared_energy(), 55);
        assert_eq!(declared.operator_squared(), certificate.operator_squared());
        assert_eq!(
            MatrixCertificate::from_bounds(1, 1, u128::MAX, 2, 13)
                .unwrap()
                .operator_squared(),
            Ok(13)
        );
    }

    #[test]
    fn row_only_scan_is_constant_payload_and_does_not_require_total_square_sum() {
        let bytes = MatrixScan::row_energy_payload_bytes(MAX_ROWS + 1, 2).unwrap();
        assert_eq!(bytes, MatrixScan::row_energy_payload_bytes(1, 1).unwrap());
        assert!(matches!(
            MatrixScan::new_row_energy(2, 2, bytes - 1),
            Err(Error::Bounds)
        ));
        assert_eq!(
            MatrixScan::row_energy_payload_bytes(usize::MAX, 2),
            Err(Error::Bounds)
        );
        let mut rows = MatrixScan::new_row_energy(2, 2, bytes).unwrap();
        rows.append(0, &[i64::MIN; 3]).unwrap();
        rows.append(3, &[i64::MIN]).unwrap();
        let certificate = rows.finish_row_energy().unwrap();
        assert_eq!((certificate.rows(), certificate.columns()), (2, 2));
        assert_eq!(certificate.maximum_row_squared_energy(), 1 << 127);
        let mut matrix = MatrixScan::new(2, 2, MatrixScan::payload_bytes(2, 2).unwrap()).unwrap();
        assert_eq!(matrix.append(0, &[i64::MIN; 4]), Err(Error::Bounds));
        assert_eq!(matrix.finish(), Err(Error::Retired));
        let mut rows = MatrixScan::new_row_energy(1, 1, bytes).unwrap();
        rows.append(0, &[1]).unwrap();
        assert_eq!(rows.finish(), Err(Error::Shape));

        let mut large = MatrixScan::new_row_energy(MAX_ROWS + 1, 2, bytes).unwrap();
        for row in 0..MAX_ROWS + 1 {
            large.append(row * 2, &[-3, 4]).unwrap();
        }
        assert_eq!(
            large
                .finish_row_energy()
                .unwrap()
                .maximum_row_squared_energy(),
            25
        );
    }

    #[test]
    fn requested_row_energy_overflow_order_errors_and_cancellation_retire_scan() {
        let bytes = MatrixScan::row_energy_payload_bytes(1, 4).unwrap();
        let mut scan = MatrixScan::new_row_energy(1, 4, bytes).unwrap();
        scan.append(0, &[i64::MIN; 3]).unwrap();
        assert_eq!(scan.append(3, &[i64::MIN]), Err(Error::Bounds));
        assert_eq!(scan.finish_row_energy(), Err(Error::Retired));
        let mut scan = MatrixScan::new_row_energy(1, 4, bytes).unwrap();
        scan.append(0, &[1]).unwrap();
        assert_eq!(scan.append(0, &[2]), Err(Error::Shape));
        assert_eq!(scan.finish_row_energy(), Err(Error::Retired));
        let mut scan = MatrixScan::new_row_energy(1, 4, bytes).unwrap();
        scan.append(0, &[1]).unwrap();
        assert_eq!(scan.finish_row_energy(), Err(Error::Shape));
        let mut scan = MatrixScan::new_row_energy(1, 4, bytes).unwrap();
        scan.cancel();
        assert_eq!(scan.append(0, &[1]), Err(Error::Retired));
        assert_eq!(scan.finish_row_energy(), Err(Error::Retired));
    }

    #[test]
    fn scans_preflight_payload_and_consume_every_malformed_or_overflowed_append() {
        let bytes = MatrixScan::payload_bytes(MAX_ROWS, MAX_COLUMNS).unwrap();
        assert!(bytes < (257 << 10));
        assert!(matches!(
            MatrixScan::new(MAX_ROWS, MAX_COLUMNS, bytes - 1),
            Err(Error::Bounds)
        ));
        for shape in [
            (0, 1),
            (1, 0),
            (MAX_ROWS + 1, 1),
            (1, MAX_COLUMNS + 1),
            (usize::MAX, 1),
        ] {
            assert_eq!(
                MatrixScan::payload_bytes(shape.0, shape.1),
                Err(Error::Shape)
            );
        }
        for (offset, values) in [
            (1, vec![0]),
            (0, vec![]),
            (0, vec![0; 5]),
            (usize::MAX, vec![0]),
            (0, vec![i64::MIN; 4]),
        ] {
            let mut owner = MatrixScan::new(2, 2, bytes).unwrap();
            assert!(owner.append(offset, &values).is_err());
            assert_eq!(owner.append(0, &[0; 4]), Err(Error::Retired));
            assert_eq!(owner.finish(), Err(Error::Retired));
        }
        let mut owner = MatrixScan::new(2, 2, bytes).unwrap();
        owner.append(0, &[1, -1, 2]).unwrap();
        assert_eq!(owner.finish(), Err(Error::Shape));
        let mut owner = MatrixScan::new(1, 1, bytes).unwrap();
        owner.append(0, &[0]).unwrap();
        assert_eq!(owner.append(1, &[0]), Err(Error::Shape));
        assert_eq!(owner.finish(), Err(Error::Retired));
    }

    #[test]
    fn roots_and_row_energy_cover_full_integer_domain_and_fail_closed_on_overflow() {
        for value in [
            0,
            1,
            2,
            3,
            4,
            15,
            16,
            17,
            (1 << 126) - 1,
            1 << 126,
            u128::MAX,
        ] {
            let root = ceil_sqrt(value);
            let big = BigUint::from(value);
            assert!(BigUint::from(root).pow(2) >= big);
            if root != 0 {
                assert!(BigUint::from(root - 1).pow(2) < big);
            }
        }
        assert_eq!(ceil_sqrt(u128::MAX), 1 << 64);
        assert_eq!(row_energy(&[]), Ok(0));
        assert_eq!(row_energy(&[i64::MIN]), Ok(1 << 126));
        assert_eq!(row_energy(&[i64::MIN; 4]), Err(Error::Bounds));
        assert_eq!(dot_radius(u128::MAX, 2), Err(Error::Bounds));
        assert_eq!(sum_energy(u128::MAX, 1), Err(Error::Bounds));
        assert_eq!(rescaled_energy(1, 1, 128, 0), Err(Error::Bounds));
        assert_eq!(rescaled_energy(1, 0, 1, 1), Err(Error::Shape));
        assert_eq!(rescaled_energy(1, 2, 1, u128::MAX), Err(Error::Bounds));
        assert_eq!(
            rotary_energy(u128::MAX, &[2], &[0], 18, 2),
            Err(Error::Bounds)
        );
    }

    #[test]
    fn independent_signed_floor_and_ceiling_matrix_outputs_obey_shared_envelope() {
        for code in 0..81usize {
            let mut code = code;
            let mut weights = [0i64; 4];
            for weight in &mut weights {
                *weight = (code % 3) as i64 - 1;
                code /= 3;
            }
            let certificate = scan(2, 2, &weights, 3);
            for x in -2..=2i64 {
                for y in -2..=2i64 {
                    let raw = [
                        weights[0] * x + weights[1] * y,
                        weights[2] * x + weights[3] * y,
                    ];
                    let input = energy(&[x, y]);
                    assert!(energy(&raw) <= linear_energy(input, certificate, 0, 0).unwrap());
                    for shift in [1, 3] {
                        let bound = linear_energy(input, certificate, shift, 1).unwrap();
                        for choices in 0..4 {
                            let output = std::array::from_fn::<_, 2, _>(|i| {
                                raw[i].div_euclid(1 << shift)
                                    + i64::from(
                                        choices & (1 << i) != 0
                                            && raw[i].rem_euclid(1 << shift) != 0,
                                    )
                            });
                            assert!(energy(&output) <= bound);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn rotary_final_and_separate_product_rounding_envelopes_and_dot_bounds() {
        for x in -3..=3i64 {
            for y in -3..=3i64 {
                for (c, s) in [(8, 0), (0, 8), (5, 6), (-5, 6)] {
                    let input = energy(&[x, y]);
                    let final_floor =
                        [(x * c - y * s).div_euclid(8), (y * c + x * s).div_euclid(8)];
                    assert!(
                        energy(&final_floor) <= rotary_energy(input, &[c], &[s], 3, 1).unwrap()
                    );
                    let bound = rotary_energy(input, &[c], &[s], 3, 2).unwrap();
                    for choices in 0..16 {
                        let raw = [x * c, -y * s, y * c, x * s];
                        let terms = std::array::from_fn::<_, 4, _>(|i| {
                            raw[i].div_euclid(8)
                                + i64::from(choices & (1 << i) != 0 && raw[i].rem_euclid(8) != 0)
                        });
                        let out = [terms[0] + terms[1], terms[2] + terms[3]];
                        assert!(energy(&out) <= bound);
                        let bias = [-2, 3];
                        assert!(
                            energy(&[out[0] - 2, out[1] + 3])
                                <= sum_energy(bound, energy(&bias)).unwrap()
                        );
                        let dot = out[0] as i128 * -2 + out[1] as i128 * 3;
                        assert!(dot.unsigned_abs() <= dot_radius(bound, energy(&bias)).unwrap());
                    }
                }
            }
        }
    }
}
