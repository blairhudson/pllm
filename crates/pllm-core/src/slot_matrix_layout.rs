//! Checked public diagonal address maps and clear-slot algebra, not HE execution.
//!
//! Source: `papers/thor.pdf`, SHA-256
//! `ddb57004b56ee8e00beffc4f563fd0921242753fda6c704065d4900b792439cd`,
//! frozen by `paper/methods/source-lock.v1.json`. Definitions 3.1, 3.3,
//! Corollary 3.8 / Eq. (10), Algorithm 1's block decomposition, Appendix B.2–3.
//! `ckks_layout::diagonals` is a distinct bounded matrix-vector layout; its
//! cryptographic admission and numeric restrictions are deliberately not reused
//! or widened here. No backend, ciphertext, scale, tolerance, or admission claim.
//!
//! Slots are frame-major, then diagonal, time, head (Eq. 7). Valid rectangles
//! are zero-extended into divisible physical dimensions. Every input padding lane
//! is ignored, even if it contains garbage. Output layout padding is always zero;
//! caller-buffer suffix beyond the required extent is untouched. `frame_slots`
//! is the logical payload extent, NOT a backend CKKS slot count. In particular,
//! the source's H=12,n=128,c=16 payload has 24576 lanes, not 32768. Mapping its
//! unused backend-slot tail and cyclic rotations remains a backend-layout gate.
//! Arithmetic is ordinary f64, with no rounding/tolerance or CKKS simulation;
//! finite inputs may overflow to IEEE infinity. All fallible checks precede writes.
//!
//! Algorithm 2 uses the paper-derived full-block routing correction proved in
//! `docs/research/thor-multiblock-indexing-audit.md`, not an author erratum.
//! Source gaps: Appendix B.2/B.3 printed mask indexing (use definition-based
//! clear transforms, not those mask schedules); packed lower-lower complexification
//! (only the fully specified Proposition 3.6 control below is implemented). Declared
//! schedules exclude replication/transpose and input preparation. Rescale fields
//! declare product-sum boundaries, not a complete level/scale DAG: the source
//! does not freeze that DAG or mask scales. No full-source execution admission.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Diagonal {
    Upper,
    Lower,
}

/// Local clear-algorithm limits supplied by caller; not encrypted admission.
/// Work bounds count visited buffer elements and contraction terms, including
/// validation/initialization passes, not CPU instructions or cryptographic work.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_buffer_elements: usize,
    pub max_work: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Geometry,
    Overflow,
    Buffer,
    WorkLimit,
    NonFinite,
    /// Historical printed Algorithm 2 index gate; retained for API compatibility.
    /// Full logical blocks now use the explicit paper-derived routing correction.
    /// This does not admit physical CKKS slot tails.
    OptimizedMultiBlockIndexGate,
}

fn mul(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_mul(b).ok_or(Error::Overflow)
}
fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Overflow)
}
fn work(n: usize, limits: Limits) -> Result<(), Error> {
    if n > limits.max_work {
        Err(Error::WorkLimit)
    } else {
        Ok(())
    }
}
fn buffer(actual: usize, required: usize, limits: Limits) -> Result<(), Error> {
    if actual < required || actual > limits.max_buffer_elements {
        Err(Error::Buffer)
    } else {
        Ok(())
    }
}
fn finite(values: &[f64]) -> Result<(), Error> {
    if values.iter().any(|v| !v.is_finite()) {
        Err(Error::NonFinite)
    } else {
        Ok(())
    }
}
// Modular sum/difference without overflowing even near usize::MAX.
fn plus(a: usize, b: usize, n: usize) -> usize {
    let b = b % n;
    if a >= n - b { a - (n - b) } else { a + b }
}
fn minus(a: usize, b: usize, n: usize) -> usize {
    let b = b % n;
    if a >= b { a - b } else { n - (b - a) }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    rows: usize,
    columns: usize,
    valid_rows: usize,
    valid_columns: usize,
    heads: usize,
    capacity: usize,
    diagonal: Diagonal,
    length: usize,
    diagonals: usize,
    frame_slots: usize,
    elements: usize,
    dense_elements: usize,
}

impl Layout {
    /// Rectangular diagonal vectors are bijective only for these divisible
    /// dimensions. Partial final frames are an explicit zero-padding extension.
    // Keep physical/valid geometry and caller limits explicit at construction.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rows: usize,
        columns: usize,
        valid_rows: usize,
        valid_columns: usize,
        heads: usize,
        capacity: usize,
        diagonal: Diagonal,
        limits: Limits,
    ) -> Result<Self, Error> {
        let length = rows.max(columns);
        let diagonals = rows.min(columns);
        if diagonals == 0
            || heads == 0
            || capacity == 0
            || capacity > diagonals
            || length % diagonals != 0
            || valid_rows > rows
            || valid_columns > columns
        {
            return Err(Error::Geometry);
        }
        let frame_slots = mul(mul(capacity, length)?, heads)?;
        let frames = diagonals / capacity + usize::from(diagonals % capacity != 0);
        let elements = mul(frames, frame_slots)?;
        let dense_elements = mul(mul(rows, columns)?, heads)?;
        buffer(elements, elements, limits)?;
        buffer(dense_elements, dense_elements, limits)?;
        work(elements, limits)?;
        Ok(Self {
            rows,
            columns,
            valid_rows,
            valid_columns,
            heads,
            capacity,
            diagonal,
            length,
            diagonals,
            frame_slots,
            elements,
            dense_elements,
        })
    }
    pub fn rows(self) -> usize {
        self.rows
    }
    pub fn columns(self) -> usize {
        self.columns
    }
    pub fn heads(self) -> usize {
        self.heads
    }
    pub fn valid_shape(self) -> (usize, usize) {
        (self.valid_rows, self.valid_columns)
    }
    pub fn capacity(self) -> usize {
        self.capacity
    }
    pub fn diagonal_length(self) -> usize {
        self.length
    }
    pub fn diagonal_count(self) -> usize {
        self.diagonals
    }
    pub fn elements(self) -> usize {
        self.elements
    }
    pub fn dense_elements(self) -> usize {
        self.dense_elements
    }
    pub fn frame_slots(self) -> usize {
        self.frame_slots
    }
    pub fn frames(self) -> usize {
        self.elements / self.frame_slots
    }

    fn raw(self, d: usize, t: usize, h: usize) -> usize {
        (d / self.capacity) * self.frame_slots
            + ((d % self.capacity) * self.length + t) * self.heads
            + h
    }
    fn coordinates(self, d: usize, t: usize) -> (usize, usize) {
        match self.diagonal {
            Diagonal::Upper => (t % self.rows, plus(t % self.columns, d, self.columns)),
            Diagonal::Lower => (plus(t % self.rows, d, self.rows), t % self.columns),
        }
    }
    /// Includes physical coordinates outside valid rectangle. No HE allocation.
    pub fn address(self, head: usize, row: usize, column: usize) -> Result<usize, Error> {
        if head >= self.heads || row >= self.rows || column >= self.columns {
            return Err(Error::Geometry);
        }
        let (d, t) = match self.diagonal {
            Diagonal::Upper if self.rows >= self.columns => (minus(column, row, self.columns), row),
            Diagonal::Upper => {
                let d = minus(column % self.rows, row, self.rows);
                (d, minus(column, d, self.columns))
            }
            Diagonal::Lower if self.columns >= self.rows => (minus(row, column, self.rows), column),
            Diagonal::Lower => {
                let d = minus(row % self.columns, column, self.columns);
                (d, minus(row, d, self.rows))
            }
        };
        Ok(self.raw(d, t, head))
    }
    pub fn lane(self, index: usize) -> Result<Option<(usize, usize, usize)>, Error> {
        if index >= self.elements {
            return Err(Error::Buffer);
        }
        let frame = index / self.frame_slots;
        let local = index % self.frame_slots;
        let d = frame * self.capacity + local / (self.length * self.heads);
        if d >= self.diagonals {
            return Ok(None);
        }
        let h = local % self.heads;
        let t = (local / self.heads) % self.length;
        let (r, c) = self.coordinates(d, t);
        Ok(if r < self.valid_rows && c < self.valid_columns {
            Some((h, r, c))
        } else {
            None
        })
    }
    fn live(self, d: usize, t: usize) -> bool {
        let (r, c) = self.coordinates(d, t);
        r < self.valid_rows && c < self.valid_columns
    }
    fn read(self, slots: &[f64], d: usize, t: usize, h: usize) -> f64 {
        if self.live(d, t) {
            slots[self.raw(d, t, h)]
        } else {
            0.0
        }
    }
    fn validate(self, slots: &[f64], limits: Limits) -> Result<(), Error> {
        buffer(slots.len(), self.elements, limits)?;
        // Padding is intentionally not an operand (including NaN padding).
        for (i, v) in slots[..self.elements].iter().enumerate() {
            if self.lane(i)?.is_some() && !v.is_finite() {
                return Err(Error::NonFinite);
            }
        }
        Ok(())
    }
    pub fn pack(self, dense: &[f64], output: &mut [f64], limits: Limits) -> Result<(), Error> {
        buffer(dense.len(), self.dense_elements, limits)?;
        buffer(output.len(), self.elements, limits)?;
        work(mul(2, self.elements)?, limits)?;
        for h in 0..self.heads {
            for r in 0..self.valid_rows {
                for c in 0..self.valid_columns {
                    if !dense[(h * self.rows + r) * self.columns + c].is_finite() {
                        return Err(Error::NonFinite);
                    }
                }
            }
        }
        for (i, v) in output[..self.elements].iter_mut().enumerate() {
            *v = match self.lane(i).expect("validated index") {
                Some((h, r, c)) => dense[(h * self.rows + r) * self.columns + c],
                None => 0.0,
            };
        }
        Ok(())
    }
    pub fn unpack(self, slots: &[f64], output: &mut [f64], limits: Limits) -> Result<(), Error> {
        buffer(slots.len(), self.elements, limits)?;
        buffer(output.len(), self.dense_elements, limits)?;
        work(add(mul(2, self.elements)?, self.dense_elements)?, limits)?;
        self.validate(slots, limits)?;
        output[..self.dense_elements].fill(0.0);
        for (i, value) in slots[..self.elements].iter().enumerate() {
            if let Some((h, r, c)) = self.lane(i).expect("validated index") {
                output[(h * self.rows + r) * self.columns + c] = *value;
            }
        }
        Ok(())
    }
}

/// Transpose via Definition 3.1 / Lemma 3.3, not ambiguous printed mask indices
/// of Algorithm 5. Supports tall, wide, partial frames and either orientation.
pub fn transpose(
    source: Layout,
    input: &[f64],
    target: Layout,
    output: &mut [f64],
    limits: Limits,
) -> Result<(), Error> {
    if source.rows != target.columns
        || source.columns != target.rows
        || source.valid_rows != target.valid_columns
        || source.valid_columns != target.valid_rows
        || source.heads != target.heads
    {
        return Err(Error::Geometry);
    }
    buffer(input.len(), source.elements, limits)?;
    buffer(output.len(), target.elements, limits)?;
    work(add(source.elements, target.elements)?, limits)?;
    source.validate(input, limits)?;
    for (i, v) in output[..target.elements].iter_mut().enumerate() {
        *v = match target.lane(i).expect("validated index") {
            Some((h, r, c)) => input[source.address(h, c, r).expect("checked shapes")],
            None => 0.0,
        };
    }
    Ok(())
}

/// Left rotation independently within each interlaced diagonal; never crosses
/// head or frame boundaries. Mask both source and destination valid lanes.
pub fn internal_rotate(
    layout: Layout,
    input: &[f64],
    shift: usize,
    output: &mut [f64],
    limits: Limits,
) -> Result<(), Error> {
    buffer(input.len(), layout.elements, limits)?;
    buffer(output.len(), layout.elements, limits)?;
    work(mul(3, layout.elements)?, limits)?;
    layout.validate(input, limits)?;
    output[..layout.elements].fill(0.0);
    for d in 0..layout.diagonals {
        for t in 0..layout.length {
            if layout.live(d, t) {
                for h in 0..layout.heads {
                    output[layout.raw(d, t, h)] =
                        layout.read(input, d, plus(t, shift, layout.length), h);
                }
            }
        }
    }
    Ok(())
}

/// Output has one frame per actual diagonal, containing `capacity` copies.
/// Definition of replicated vectors in §4.3.1; avoids Appendix B.2's ambiguous
/// printed nu_k index (k ranges over c/2 but mask text says n/c).
pub fn replicate(
    layout: Layout,
    input: &[f64],
    output: &mut [f64],
    limits: Limits,
) -> Result<(), Error> {
    let required = mul(layout.diagonals, layout.frame_slots)?;
    buffer(input.len(), layout.elements, limits)?;
    buffer(output.len(), required, limits)?;
    work(add(required, layout.elements)?, limits)?;
    layout.validate(input, limits)?;
    for d in 0..layout.diagonals {
        for copy in 0..layout.capacity {
            for t in 0..layout.length {
                for h in 0..layout.heads {
                    output
                        [d * layout.frame_slots + (copy * layout.length + t) * layout.heads + h] =
                        layout.read(input, d, t, h);
                }
            }
        }
    }
    Ok(())
}

/// Declared primitive schedule, NOT timings and NOT actual HE calls. Positions
/// are represented explicitly by separate stages. Clear gathers fuse rotations
/// and masks; scalar work admission is separate from these SIMD counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Schedule {
    pub tensor_products: usize,
    pub plaintext_products: usize,
    pub input_rotations: usize,
    pub output_rotations: usize,
    pub conjugations: usize,
    pub rescale_after_product_sums: usize,
    pub relinearize_after_product_sums: usize,
    pub relinearize_before_output_rotation: usize,
}

/// Source Eq. (10) basic lower-lower evaluation with already replicated RHS.
/// Four masks/two rotations per nonzero l; no BSGS or complexification claim.
pub fn lower_schedule(left: Layout) -> Result<Schedule, Error> {
    if left.diagonal != Diagonal::Lower
        || left.rows > left.columns
        || left.diagonals % left.capacity != 0
    {
        return Err(Error::Geometry);
    }
    let pairs = mul(left.frames(), left.columns - 1)?;
    Ok(Schedule {
        tensor_products: mul(left.frames(), left.columns)?,
        plaintext_products: mul(4, pairs)?,
        input_rotations: mul(2, pairs)?,
        rescale_after_product_sums: left.frames(),
        relinearize_after_product_sums: left.frames(),
        ..Schedule::default()
    })
}

/// Algorithm 2 with paper-derived lines 16–17 frame routing correction:
/// `(j-floor(l/c)-1) mod M`, `(j-floor(l/c)) mod M`, M=m/c.
/// Line 7 rotations are unchanged. Full logical frames only, not CKKS tails.
pub fn optimized_lower_schedule(left: Layout) -> Result<Schedule, Error> {
    let _ = lower_schedule(left)?;
    let frames = left.frames();
    let pairs = mul(frames, left.columns - 1)?;
    let delayed = if left.columns > 1 { frames } else { 0 };
    Ok(Schedule {
        tensor_products: mul(frames, left.columns)?,
        plaintext_products: mul(3, pairs)?,
        input_rotations: pairs,
        output_rotations: delayed,
        rescale_after_product_sums: add(frames, delayed)?,
        relinearize_after_product_sums: frames,
        relinearize_before_output_rotation: delayed,
        ..Schedule::default()
    })
}

/// Executable Corollary 3.8 / Eq. (10), over caller packed buffers. This is
/// basic lower-lower algebra, not execution of optimized Algorithm 2.
/// RHS is ordinary multi-diagonal packed; replicated accesses are fused, so no
/// allocated replicated owner or implicit HE replication cost is claimed.
pub fn lower_mm(
    left: Layout,
    a: &[f64],
    right: Layout,
    b: &[f64],
    out: Layout,
    output: &mut [f64],
    limits: Limits,
) -> Result<(), Error> {
    if left.diagonal != Diagonal::Lower
        || right.diagonal != Diagonal::Lower
        || out.diagonal != Diagonal::Lower
        || left.rows > left.columns
        || right.rows != left.columns
        || right.columns != left.columns
        || out.rows != left.rows
        || out.columns != left.columns
        || left.heads != right.heads
        || left.heads != out.heads
        || left.valid_columns != right.valid_rows
        || out.valid_rows != left.valid_rows
        || out.valid_columns != right.valid_columns
    {
        return Err(Error::Geometry);
    }
    buffer(a.len(), left.elements, limits)?;
    buffer(b.len(), right.elements, limits)?;
    buffer(output.len(), out.elements, limits)?;
    work(
        add(
            add(left.elements, right.elements)?,
            mul(out.elements, add(left.columns, 1)?)?,
        )?,
        limits,
    )?;
    left.validate(a, limits)?;
    right.validate(b, limits)?;
    output[..out.elements].fill(0.0);
    for r in 0..out.diagonals {
        for t in 0..out.length {
            if out.live(r, t) {
                for h in 0..out.heads {
                    let mut sum = 0.0;
                    for l in 0..left.columns {
                        sum += left.read(a, minus(r, l, left.rows), plus(t, l, left.columns), h)
                            * right.read(b, l, t, h);
                    }
                    output[out.raw(r, t, h)] = sum;
                }
            }
        }
    }
    Ok(())
}

/// Executable Algorithm 2 for c|m|n, with paper-derived frame routing correction
/// (see `optimized_lower_schedule`) and delayed rotation of each second sum.
/// Scratch is two packed extents (2*left.elements()), owned by caller; no
/// products, replicated RHS or masks are allocated. Clear gathers fuse masks
/// and rotations, as in the single-frame subset; primitive counts are declared
/// SIMD operations, not scalar instructions. Physical CKKS tails unsupported.
/// RHS is packed normally; repeated lower diagonal reads fuse replication.
// Explicit immutable layouts, borrowed operands and separate caller-owned
// scratch/output buffers preserve ownership and pre-mutation validation order.
#[allow(clippy::too_many_arguments)]
pub fn optimized_lower_mm(
    left: Layout,
    a: &[f64],
    right: Layout,
    b: &[f64],
    out: Layout,
    scratch: &mut [f64],
    output: &mut [f64],
    limits: Limits,
) -> Result<(), Error> {
    let _ = optimized_lower_schedule(left)?;
    if right.rows != left.columns
        || right.columns != left.columns
        || right.diagonal != Diagonal::Lower
        || out != left
        || right.heads != left.heads
        || left.valid_columns != right.valid_rows
        || out.valid_columns != right.valid_columns
    {
        return Err(Error::Geometry);
    }
    let scratch_len = mul(2, left.elements)?;
    buffer(a.len(), left.elements, limits)?;
    buffer(b.len(), right.elements, limits)?;
    buffer(scratch.len(), scratch_len, limits)?;
    buffer(output.len(), out.elements, limits)?;
    work(
        add(
            add(left.elements, right.elements)?,
            mul(mul(4, out.elements)?, left.columns)?,
        )?,
        limits,
    )?;
    left.validate(a, limits)?;
    right.validate(b, limits)?;
    scratch[..scratch_len].fill(0.0);
    let extent = left.elements;
    let s = left.frame_slots;
    let block = left.length * left.heads;
    // At destination x, first partial sum uses product x. Wrapped time lanes
    // use product y=x-nH, followed by the single final rotation by -nH.
    // Both are selected from Rot(A;(-n*(l%c)+l)*H), Algorithm 2 line 7.
    for r in 0..left.diagonals {
        for t in 0..left.length {
            for h in 0..left.heads {
                let x = left.raw(r, t, h);
                let frame = r / left.capacity;
                let local = x % s;
                for l in 0..left.columns {
                    let wrapped = t >= left.length - l;
                    let y = if wrapped {
                        minus(local, block, s)
                    } else {
                        local
                    };
                    // Both products are strictly below checked frame size s.
                    let shift = minus(l * left.heads, (l % left.capacity) * block, s);
                    // Correct product-frame routing only; do not also relabel
                    // line 7's input rotation. Reduce q modulo M for l>=m.
                    let routed = minus(frame, l / left.capacity, left.frames());
                    let routed = minus(
                        routed,
                        usize::from(r % left.capacity < l % left.capacity),
                        left.frames(),
                    );
                    let src = routed * s + plus(y, shift, s);
                    let av = if left.lane(src).expect("validated index").is_some() {
                        a[src]
                    } else {
                        0.0
                    };
                    let product = av * right.read(b, l, t, h);
                    if wrapped {
                        scratch[extent + frame * s + y] += product;
                    } else {
                        scratch[x] += product;
                    }
                }
            }
        }
    }
    for x in 0..extent {
        output[x] = if out.lane(x).expect("validated index").is_some() {
            scratch[x] + scratch[extent + (x / s) * s + minus(x % s, block, s)]
        } else {
            0.0
        };
    }
    Ok(())
}

/// Algorithm 1 declared non-complexified diagonal-block schedule. Plaintexts
/// include masks. Lazy rescale follows each block product sum before alignment;
/// no ciphertext tensor product or relinearization occurs.
pub fn block_pc_schedule(layout: Layout) -> Result<Schedule, Error> {
    if layout.rows != layout.columns
        || layout.diagonal != Diagonal::Lower
        || layout.valid_rows != layout.rows
        || layout.diagonals % layout.capacity != 0
    {
        return Err(Error::Geometry);
    }
    let nc = layout.frames();
    let align = mul(layout.heads - 1, nc)?;
    Ok(Schedule {
        plaintext_products: add(
            mul(mul(layout.heads, nc)?, layout.rows)?,
            mul(layout.heads, nc)?,
        )?,
        input_rotations: mul(nc, layout.capacity - 1)?,
        output_rotations: mul(2, align)?,
        rescale_after_product_sums: mul(layout.heads, nc)?,
        ..Schedule::default()
    })
}

/// Algorithm 1 block decomposition C_i = sum_k A_ik B_k, evaluating Eq. (2)
/// inside each block. Input/output heads here mean block rows, NOT independent
/// attention heads. A is row-major d*d, d=n*heads. Clear slot accesses fuse
/// plaintext preparation and masked block alignment; no temporary allocations.
/// Only full physical n*n blocks supported; padding in B still zero-extended.
pub fn block_pc_mm(
    layout: Layout,
    weights: &[f64],
    b: &[f64],
    output: &mut [f64],
    limits: Limits,
) -> Result<(), Error> {
    let _ = block_pc_schedule(layout)?;
    let d = mul(layout.rows, layout.heads)?;
    let weights_len = mul(d, d)?;
    buffer(weights.len(), weights_len, limits)?;
    buffer(b.len(), layout.elements, limits)?;
    buffer(output.len(), layout.elements, limits)?;
    work(
        add(
            add(weights_len, mul(2, layout.elements)?)?,
            mul(layout.elements, d)?,
        )?,
        limits,
    )?;
    finite(&weights[..weights_len])?;
    layout.validate(b, limits)?;
    // B/output have full physical row support; valid columns are preserved.
    output[..layout.elements].fill(0.0);
    for r in 0..layout.diagonals {
        for t in 0..layout.length {
            if t < layout.valid_columns {
                for i in 0..layout.heads {
                    let row = plus(t, r, layout.rows);
                    let mut sum = 0.0;
                    for k in 0..layout.heads {
                        for l in 0..layout.rows {
                            let inner = plus(t, l, layout.rows);
                            sum += weights[(i * layout.rows + row) * d + k * layout.rows + inner]
                                * layout.read(b, l, t, k);
                        }
                    }
                    output[layout.raw(r, t, i)] = sum;
                }
            }
        }
    }
    Ok(())
}

/// Source §4.4 / p.9 complexification control, restricted to Proposition 3.6:
/// A is n*m upper, B is m*n lower, n divisible by even m. One diagonal per
/// frame, independent heads. No claim about unspecified packed lower-lower
/// complexification. X=sum (a0-i*a1)/2 * (b0+i*b1), C=X+Conj(X).
/// `imaginary` receives Im(X), exposing the nontrivial conjugation control.
/// This is real f64 component arithmetic, not a CKKS complex tolerance.
/// Its declared core schedule is `paired_upper_lower_schedule`; /2 is input
/// preparation, with no inferred rescale or relinearization placement.
// Keep immutable operand/output layouts and both caller-owned result buffers
// explicit; imaginary control output must not be hidden in allocated state.
#[allow(clippy::too_many_arguments)]
pub fn paired_upper_lower_mm(
    left: Layout,
    a: &[f64],
    right: Layout,
    b: &[f64],
    out: Layout,
    imaginary: &mut [f64],
    output: &mut [f64],
    limits: Limits,
) -> Result<(), Error> {
    let n = left.rows;
    let m = left.columns;
    if n < m
        || m % 2 != 0
        || left.diagonal != Diagonal::Upper
        || right.diagonal != Diagonal::Lower
        || out.diagonal != Diagonal::Lower
        || right.rows != m
        || right.columns != n
        || out.rows != n
        || out.columns != n
        || left.heads != right.heads
        || left.heads != out.heads
        || left.capacity != 1
        || right.capacity != 1
        || out.capacity != 1
        || left.valid_columns != right.valid_rows
        || out.valid_rows != left.valid_rows
        || out.valid_columns != right.valid_columns
    {
        return Err(Error::Geometry);
    }
    buffer(a.len(), left.elements, limits)?;
    buffer(b.len(), right.elements, limits)?;
    buffer(output.len(), out.elements, limits)?;
    buffer(imaginary.len(), out.elements, limits)?;
    work(
        add(
            add(left.elements, right.elements)?,
            mul(mul(8, out.elements)?, m)?,
        )?,
        limits,
    )?;
    left.validate(a, limits)?;
    right.validate(b, limits)?;
    output[..out.elements].fill(0.0);
    imaginary[..out.elements].fill(0.0);
    for r in 0..n {
        for t in 0..n {
            if out.live(r, t) {
                for h in 0..out.heads {
                    let mut re = 0.0;
                    let mut im = 0.0;
                    for l in 0..m / 2 {
                        let a0 = left.read(a, minus(l, r, m), plus(t, r, n), h) * 0.5;
                        let a1 = left.read(a, minus(l + m / 2, r, m), plus(t, r, n), h) * 0.5;
                        let b0 = right.read(b, l, t, h);
                        let b1 = right.read(b, l + m / 2, t, h);
                        re += a0 * b0 + a1 * b1;
                        im += a0 * b1 - a1 * b0;
                    }
                    let x = out.raw(r, t, h);
                    imaginary[x] = im;
                    // Real part of X+Conj(X); imaginary terms cancel exactly by sign.
                    output[x] = re + re;
                }
            }
        }
    }
    Ok(())
}

/// Proposition 3.6 core only. Scale/level/relinearization placement is not
/// specified by this control and deliberately not inferred from Algorithm 2.
pub fn paired_upper_lower_schedule(left: Layout) -> Result<Schedule, Error> {
    if left.rows < left.columns
        || left.columns % 2 != 0
        || left.capacity != 1
        || left.diagonal != Diagonal::Upper
    {
        return Err(Error::Geometry);
    }
    Ok(Schedule {
        tensor_products: mul(left.rows, left.columns / 2)?,
        input_rotations: mul(left.rows - 1, left.columns / 2)?,
        conjugations: left.rows,
        ..Schedule::default()
    })
}

#[cfg(test)]
#[path = "slot_matrix_layout_tests.rs"]
mod tests;
