//! Bounded clear source views and complete-row modular linear accumulation.
//!
//! Supplied views establish no authenticated source authority, head tying,
//! no-wrap, protocol, transport, privacy or trained-model qualification.
//! Original geometry and global offsets survive partitioning. Only complete
//! rows can publish field/BF16 outputs; no partial centering or F32 bridge.

use super::{
    bf16_q8, decode_reply_bf16, field_add, field_mul, fp8_q8, import_bf16_q8_into,
    positive_bf16_scale, q8_to_field, Block128Geometry, Error,
};

/// A tightly packed row-major view in the original, unpadded matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceWindow {
    pub row_start: usize,
    pub rows: usize,
    pub column_start: usize,
    pub columns: usize,
}

impl SourceWindow {
    fn checked(self, geometry: Block128Geometry) -> Result<usize, Error> {
        if self.rows == 0
            || self.columns == 0
            || self
                .row_start
                .checked_add(self.rows)
                .ok_or(Error::Geometry)?
                > geometry.rows
            || self
                .column_start
                .checked_add(self.columns)
                .ok_or(Error::Geometry)?
                > geometry.columns
        {
            return Err(Error::Geometry);
        }
        self.rows.checked_mul(self.columns).ok_or(Error::Geometry)
    }

    fn scale_columns(self) -> usize {
        (self.column_start + self.columns - 1) / 128 - self.column_start / 128 + 1
    }

    fn scale_count(self) -> usize {
        let rows = (self.row_start + self.rows - 1) / 128 - self.row_start / 128 + 1;
        rows * self.scale_columns()
    }

    fn scale_index(self, index: usize) -> usize {
        let row = self.row_start + index / self.columns;
        let column = self.column_start + index % self.columns;
        (row / 128 - self.row_start / 128) * self.scale_columns() + column / 128
            - self.column_start / 128
    }
}

impl Block128Geometry {
    /// Import a supplied clear FP8 window. `scales` is the tightly packed
    /// row-major rectangle of ORIGINAL 128x128 scale cells touched by `window`,
    /// including partial boundary cells, never retiled at local offset zero.
    /// Exact capacity is sufficient; spare output entries remain unchanged.
    /// All source validation precedes writes, with no matrix-sized allocation.
    pub fn import_q8_window(
        self,
        window: SourceWindow,
        weights: &[u8],
        scales: &[u16],
        output: &mut [i64],
        work_limit: usize,
    ) -> Result<(), Error> {
        let length = window.checked(self)?;
        if weights.len() != length || scales.len() != window.scale_count() {
            return Err(Error::Length);
        }
        if output.len() < length {
            return Err(Error::Capacity);
        }
        // Eight scalar visits per coefficient conservatively cover scales,
        // both conversion passes and writes (scale_count <= length).
        let work = length.checked_mul(8).ok_or(Error::WorkLimit)?;
        if work > work_limit {
            return Err(Error::WorkLimit);
        }
        for &scale in scales {
            positive_bf16_scale(scale)?;
        }
        let convert = |index: usize, byte: u8| fp8_q8(byte, scales[window.scale_index(index)]);
        for (index, &byte) in weights.iter().enumerate() {
            convert(index, byte)?;
        }
        for (index, (&byte, slot)) in weights.iter().zip(output).enumerate() {
            *slot = convert(index, byte)?;
        }
        Ok(())
    }
}

/// Caller owns all storage. Active input/sum scratch is exclusively borrowed
/// and cleared on success, failure or abandoned coverage. Imported coefficient
/// slices are cleared immediately after use; spare capacity is never touched.
/// Output is untouched until `finish`.
/// Minimum lengths: original columns, selected rows, largest supplied window,
/// selected rows, selected rows, respectively. No whole-matrix buffer required.
pub struct WindowBuffers<'a> {
    pub input: &'a mut [u32],
    pub sums: &'a mut [u32],
    pub coefficients: &'a mut [i64],
    pub field_output: &'a mut [u32],
    pub bf16_output: &'a mut [u16],
}

/// Prepare the ORIGINAL geometry and full input once, then borrow this immutable
/// instance for a `LinearSession`. Neither geometry nor words can be replaced.
/// Storage remains caller-owned.
/// No authenticated input/source authority is established by this clear value.
pub struct QuantizedInput<'a> {
    geometry: Block128Geometry,
    words: &'a mut [u32],
}

impl<'a> QuantizedInput<'a> {
    pub fn new(
        geometry: Block128Geometry,
        input: &[u16],
        scratch: &'a mut [u32],
        work_limit: usize,
    ) -> Result<Self, Error> {
        if input.len() != geometry.columns {
            return Err(Error::Length);
        }
        if scratch.len() < input.len() {
            return Err(Error::Capacity);
        }
        let work = input.len().checked_mul(3).ok_or(Error::WorkLimit)?;
        if work > work_limit {
            return Err(Error::WorkLimit);
        }
        let owner = Self {
            geometry,
            words: &mut scratch[..input.len()],
        };
        for (&word, slot) in input.iter().zip(owner.words.iter_mut()) {
            *slot = q8_to_field(bf16_q8(word)?);
        }
        Ok(owner)
    }
}

impl Drop for QuantizedInput<'_> {
    fn drop(&mut self) {
        self.words.fill(0);
    }
}

enum Input<'a> {
    Bf16(&'a [u16]),
    Prepared(&'a [u32]),
}

impl Input<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Bf16(words) => words.len(),
            Self::Prepared(words) => words.len(),
        }
    }
}

/// Independent clear row sample, NOT a multi-band logical-output session.
/// Use `LinearSession` for a whole original matvec with shared input and exhaustive
/// row ownership. Standalone results cannot be attached to a session.
/// Append errors destroy this owner; there are no partial-output/state getters.
#[must_use = "complete every original column and finish, or discard without output"]
pub struct WindowLinear<'a> {
    geometry: Block128Geometry,
    row_start: usize,
    rows: usize,
    next_column: usize,
    input: &'a mut [u32],
    sums: &'a mut [u32],
    coefficients: &'a mut [i64],
    output: Option<(&'a mut [u32], &'a mut [u16])>,
}

/// Complete original rows from one independent sample, or every original row
/// from one `LinearSession`. Not an attachable band token, authenticated claim
/// or honest-integer no-wrap certificate.
pub struct CompletedRows<'a> {
    pub row_start: usize,
    pub field: &'a [u32],
    pub bf16: &'a [u16],
}

/// Scratch for one band. No band output buffer: tentative rows belong solely
/// to the consuming session, and cannot be detached or attached elsewhere.
pub struct BandScratch<'a> {
    pub input: &'a mut [u32],
    pub sums: &'a mut [u32],
    pub coefficients: &'a mut [i64],
}

/// One logical clear matvec, borrowing one immutable geometry/input instance.
/// All original rows and columns must be covered in order. Only `finish`
/// publishes output. Reference identity below is typed, process-local ownership,
/// NOT source authentication, a caller label, a digest or a numeric certificate.
#[must_use = "finish every original row and column, or discard without output"]
pub struct LinearSession<'s, 'a> {
    input: &'s QuantizedInput<'s>,
    next_row: usize,
    fields: &'a mut [u32],
    output: Option<(&'a mut [u32], &'a mut [u16])>,
    work: usize,
}

impl<'s, 'a> LinearSession<'s, 'a> {
    /// Reserve a cumulative scalar-visit budget, not a fresh cap per band.
    /// Each band's coefficient/row allowance is deducted before reading source;
    /// final-conversion work remains reserved for session publication. Ordered
    /// coverage bounds total coefficient visits by rows*columns. The unchanged
    /// 16-per-coefficient allowance also covers prepared-input copy/clear visits
    /// (at most 2*rows*columns). Row scratch is O(rows), never O(rows*columns).
    /// Input preparation has its separately checked 3*columns.
    pub fn required_work(geometry: Block128Geometry) -> Result<usize, Error> {
        WindowLinear::required_work(geometry, geometry.rows)
    }

    pub fn new(
        input: &'s QuantizedInput<'s>,
        field_scratch: &'a mut [u32],
        field_output: &'a mut [u32],
        bf16_output: &'a mut [u16],
        work_limit: usize,
    ) -> Result<Self, Error> {
        let rows = input.geometry.rows;
        if field_scratch.len() < rows || field_output.len() < rows || bf16_output.len() < rows {
            return Err(Error::Capacity);
        }
        let work = Self::required_work(input.geometry)?;
        if work > work_limit {
            return Err(Error::WorkLimit);
        }
        let owner = Self {
            input,
            next_row: 0,
            fields: &mut field_scratch[..rows],
            output: Some((field_output, bf16_output)),
            work,
        };
        owner.fields.fill(0);
        Ok(owner)
    }

    /// Explicitly check the supplied original geometry and typed prepared input
    /// before even reading band scratch or converting source coefficients.
    /// No substitute prepared input, including numerically equal words, may
    /// continue this session. The band retains the consuming session itself.
    pub fn band<'b>(
        mut self,
        geometry: Block128Geometry,
        input: &QuantizedInput<'_>,
        row_start: usize,
        rows: usize,
        scratch: BandScratch<'b>,
    ) -> Result<SessionBand<'s, 'a, 'b>, Error> {
        if geometry != self.input.geometry {
            return Err(Error::Geometry);
        }
        if !std::ptr::eq(self.input, input) {
            return Err(Error::SessionMismatch);
        }
        if row_start != self.next_row {
            return Err(Error::Coverage);
        }
        let band_bound = WindowLinear::required_work(geometry, rows)?;
        // The standalone 4*columns input allowance is reserved globally once.
        // Prepared-input copies for bands fit the coefficient allowance above;
        // no repeated BF16 conversion, renewed cap or raised ceiling is needed.
        let input_allowance = geometry.columns.checked_mul(4).ok_or(Error::WorkLimit)?;
        let charge = band_bound
            .checked_sub(input_allowance)
            .ok_or(Error::WorkLimit)?;
        let remaining = self.work.checked_sub(charge).ok_or(Error::WorkLimit)?;
        let linear = WindowLinear::build(
            geometry,
            row_start,
            rows,
            Input::Prepared(&*self.input.words),
            scratch,
            None,
            band_bound,
        )?;
        self.work = remaining;
        Ok(SessionBand {
            session: self,
            linear,
        })
    }

    pub fn finish(mut self) -> Result<CompletedRows<'a>, Error> {
        if self.next_row != self.input.geometry.rows {
            return Err(Error::Incomplete);
        }
        for &sum in self.fields.iter() {
            decode_reply_bf16(sum)?;
        }
        let (field, bf16) = self.output.take().ok_or(Error::Incomplete)?;
        for ((&sum, field), bf16) in self
            .fields
            .iter()
            .zip(field.iter_mut())
            .zip(bf16.iter_mut())
        {
            *field = sum;
            *bf16 = decode_reply_bf16(sum)?;
        }
        Ok(CompletedRows {
            row_start: 0,
            field: &field[..self.next_row],
            bf16: &bf16[..self.next_row],
        })
    }
}

impl Drop for LinearSession<'_, '_> {
    fn drop(&mut self) {
        self.fields.fill(0);
    }
}

/// Not interchangeable with another session's band: the original consuming
/// session is held privately, with no attach/accept-completed-band operation.
#[must_use = "complete this band's original columns, then finish the band"]
pub struct SessionBand<'s, 'a, 'b> {
    session: LinearSession<'s, 'a>,
    linear: WindowLinear<'b>,
}

impl<'s, 'a, 'b> SessionBand<'s, 'a, 'b> {
    pub fn append_fp8(
        self,
        window: SourceWindow,
        weights: &[u8],
        scales: &[u16],
    ) -> Result<Self, Error> {
        let Self { session, linear } = self;
        let linear = linear.append_fp8(window, weights, scales)?;
        Ok(Self { session, linear })
    }

    pub fn append_bf16(self, window: SourceWindow, words: &[u16]) -> Result<Self, Error> {
        let Self { session, linear } = self;
        let linear = linear.append_bf16(window, words)?;
        Ok(Self { session, linear })
    }

    pub fn finish(self) -> Result<LinearSession<'s, 'a>, Error> {
        let Self {
            mut session,
            linear,
        } = self;
        if linear.next_column != linear.geometry.columns {
            return Err(Error::Incomplete);
        }
        let end = linear.row_start + linear.rows;
        session.fields[linear.row_start..end].copy_from_slice(&*linear.sums);
        session.next_row = end;
        Ok(session)
    }
}

impl<'a> WindowLinear<'a> {
    /// Conservative scalar-visit bound for the entire operation, including
    /// construction, exhaustive windows, final conversion and scratch clearing.
    /// Assumes only disjoint ordered windows, so each coefficient is visited a
    /// bounded number of times regardless of partition. Not a global admission
    /// or physical memory/time certificate. Overflow fails closed.
    pub fn required_work(geometry: Block128Geometry, rows: usize) -> Result<usize, Error> {
        if rows == 0 || rows > geometry.rows {
            return Err(Error::Geometry);
        }
        rows.checked_mul(geometry.columns)
            .and_then(|n| n.checked_mul(16))
            .and_then(|n| {
                geometry
                    .columns
                    .checked_mul(4)
                    .and_then(|x| n.checked_add(x))
            })
            .and_then(|n| rows.checked_mul(8).and_then(|x| n.checked_add(x)))
            .ok_or(Error::WorkLimit)
    }

    pub fn new(
        geometry: Block128Geometry,
        row_start: usize,
        rows: usize,
        input: &[u16],
        buffers: WindowBuffers<'a>,
        work_limit: usize,
    ) -> Result<Self, Error> {
        Self::build(
            geometry,
            row_start,
            rows,
            Input::Bf16(input),
            BandScratch {
                input: buffers.input,
                sums: buffers.sums,
                coefficients: buffers.coefficients,
            },
            Some((buffers.field_output, buffers.bf16_output)),
            work_limit,
        )
    }

    fn build(
        geometry: Block128Geometry,
        row_start: usize,
        rows: usize,
        input: Input<'_>,
        buffers: BandScratch<'a>,
        output: Option<(&'a mut [u32], &'a mut [u16])>,
        work_limit: usize,
    ) -> Result<Self, Error> {
        SourceWindow {
            row_start,
            rows,
            column_start: 0,
            columns: geometry.columns,
        }
        .checked(geometry)?;
        if input.len() != geometry.columns {
            return Err(Error::Length);
        }
        if buffers.input.len() < geometry.columns
            || buffers.sums.len() < rows
            || buffers.coefficients.len() < rows
            || output
                .as_ref()
                .is_some_and(|(field, bf16)| field.len() < rows || bf16.len() < rows)
        {
            return Err(Error::Capacity);
        }
        if Self::required_work(geometry, rows)? > work_limit {
            return Err(Error::WorkLimit);
        }
        // Borrow only active scratch; do not touch or clear uncharged spare
        // capacity. Coefficients are cleared after each supplied window.
        let owner = Self {
            geometry,
            row_start,
            rows,
            next_column: 0,
            input: &mut buffers.input[..geometry.columns],
            sums: &mut buffers.sums[..rows],
            coefficients: buffers.coefficients,
            output,
        };
        owner.sums.fill(0);
        // Exactly one Q8 conversion for each original BF16 input, not per tile.
        // Failure drops and clears even a partially prepared input.
        match input {
            Input::Bf16(words) => {
                for (&word, slot) in words.iter().zip(owner.input.iter_mut()) {
                    *slot = q8_to_field(bf16_q8(word)?);
                }
            }
            Input::Prepared(words) => owner.input.copy_from_slice(words),
        }
        Ok(owner)
    }

    fn check_window(&self, window: SourceWindow) -> Result<usize, Error> {
        let length = window.checked(self.geometry)?;
        if window.row_start != self.row_start
            || window.rows != self.rows
            || window.column_start != self.next_column
        {
            return Err(Error::Coverage);
        }
        if self.coefficients.len() < length {
            return Err(Error::Capacity);
        }
        Ok(length)
    }

    pub fn append_fp8(
        mut self,
        window: SourceWindow,
        weights: &[u8],
        scales: &[u16],
    ) -> Result<Self, Error> {
        let length = self.check_window(window)?;
        let work = length.checked_mul(8).ok_or(Error::WorkLimit)?;
        self.geometry.import_q8_window(
            window,
            weights,
            scales,
            &mut self.coefficients[..length],
            work,
        )?;
        self.accumulate(window, length);
        Ok(self)
    }

    /// BF16 head/embedding-derived coefficients use the SAME Q8 importer as
    /// `import_bf16_q8`, not a BF16 dot or a new head/tie/source contract.
    pub fn append_bf16(mut self, window: SourceWindow, words: &[u16]) -> Result<Self, Error> {
        let length = self.check_window(window)?;
        if words.len() != length {
            return Err(Error::Length);
        }
        let work = length.checked_mul(2).ok_or(Error::WorkLimit)?;
        import_bf16_q8_into(words, &mut self.coefficients[..length], work)?;
        self.accumulate(window, length);
        Ok(self)
    }

    fn accumulate(&mut self, window: SourceWindow, length: usize) {
        for (row, coefficients) in self.coefficients[..length]
            .chunks_exact(window.columns)
            .enumerate()
        {
            for (&weight, &input) in coefficients
                .iter()
                .zip(&self.input[window.column_start..window.column_start + window.columns])
            {
                self.sums[row] = field_add(self.sums[row], field_mul(q8_to_field(weight), input));
            }
        }
        self.coefficients[..length].fill(0);
        self.next_column += window.columns;
    }

    pub fn finish(mut self) -> Result<CompletedRows<'a>, Error> {
        if self.next_column != self.geometry.columns {
            return Err(Error::Incomplete);
        }
        // Read-before-write validation; canonical sums make conversion
        // infallible under the existing scalar contract.
        for &sum in self.sums.iter() {
            decode_reply_bf16(sum)?;
        }
        let (field, bf16) = self.output.take().ok_or(Error::Incomplete)?;
        for ((&sum, field), bf16) in self.sums.iter().zip(field.iter_mut()).zip(bf16.iter_mut()) {
            *field = sum;
            *bf16 = decode_reply_bf16(sum)?;
        }
        Ok(CompletedRows {
            row_start: self.row_start,
            field: &field[..self.rows],
            bf16: &bf16[..self.rows],
        })
    }
}

impl Drop for WindowLinear<'_> {
    fn drop(&mut self) {
        self.input.fill(0);
        self.sums.fill(0);
        // Each successfully imported window clears its active coefficient
        // slice immediately. Atomic import errors never write coefficients.
    }
}
