//! Owning causal score recipes over original mapped Q/K views. No expanded
//! Words/Nodes, arithmetic implementation, mass issuer or runtime work here.
use super::super::graph::attention_driver::{self as attention, AttentionSink};
use super::super::private_matrix_driver::{self as matrix, Sink as MatrixSink};
use super::*;
use puma::multiply::norm::RowEnergyProof;
use std::mem::{size_of, size_of_val};

pub(super) struct ScoreRow {
    pub(super) gather: ScoreGather,
    pub(super) query_start: usize,
    pub(super) key_start: usize,
    pub(super) prefix: usize,
    pub(super) suffix: usize,
    pub(super) node_offset: usize,
    pub(super) tile_start: usize,
    pub(super) tile_count: usize,
    pub(super) query_bounds: Interval,
}
pub(super) struct ScoreTile {
    pub(super) gather: ScoreGather,
    pub(super) column_start: usize,
    pub(super) count: usize,
    pub(super) right_offset: usize,
    pub(super) right_start: usize,
    pub(super) left_offset: usize,
    pub(super) shape: MatrixShape,
    pub(super) inputs: InputBounds,
    pub(super) proof: Option<RowEnergyProof>,
    pub(super) rescale: bool,
    pub(super) ordinary_bounds: Interval,
    pub(super) bounds: Interval,
}
pub(super) struct Scores {
    pub(super) query: PublishedView,
    pub(super) keys: PublishedView,
    pub(super) rows: Vec<ScoreRow>,
    pub(super) tiles: Vec<ScoreTile>,
    pub(super) width: usize,
    pub(super) columns: usize,
    pub(super) zero: (usize, usize),
    pub(super) nodes: usize,
    pub(super) work: WorkBudget,
    pub(super) bounds: Interval,
}
/// Checked causal mapping, not uniform Products. Copyable public geometry only;
/// proof/source ownership remains in Scores. Constructed only from that owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ScoreLayout {
    rows: usize,
    sequence: usize,
    columns: usize,
    tile: usize,
    zero: (usize, usize),
}
fn tile_sum(n: usize, tile: usize) -> Result<u128, Error> {
    if tile == 0 {
        return Err(Error::Shape);
    }
    let (q, r) = ((n / tile) as u128, (n % tile) as u128);
    (tile as u128)
        .checked_mul(q)
        .and_then(|v| v.checked_mul(q + 1))
        .map(|v| v / 2)
        .and_then(|v| r.checked_mul(q + 1).and_then(|tail| v.checked_add(tail)))
        .ok_or(Error::Bounds)
}
impl ScoreLayout {
    fn nodes_prefix(self, rows: usize) -> Result<usize, Error> {
        let past = self.columns - self.sequence;
        let tiles = tile_sum(add_bytes(past, rows)?, self.tile)?
            .checked_sub(tile_sum(past, self.tile)?)
            .ok_or(Error::Bounds)?;
        usize::try_from(
            (rows as u128)
                .checked_add(tiles.checked_mul(2).ok_or(Error::Bounds)?)
                .ok_or(Error::Bounds)?,
        )
        .map_err(|_| Error::Bounds)
    }
    pub(super) fn word(self, first: usize, index: usize) -> Result<(usize, usize), Error> {
        let row = index / self.columns;
        let col = index % self.columns;
        if row >= self.rows {
            return Err(Error::Shape);
        }
        let local = row % self.sequence;
        if col > self.columns - self.sequence + local {
            return Ok(self.zero);
        }
        let head = (row / self.sequence)
            .checked_mul(self.nodes_prefix(self.sequence)?)
            .ok_or(Error::Bounds)?;
        let row_start = add_bytes(head, self.nodes_prefix(local)?)?;
        let product = add_bytes(
            add_bytes(row_start, 2)?,
            (col / self.tile).checked_mul(2).ok_or(Error::Bounds)?,
        )?;
        Ok((add_bytes(first, product)?, col % self.tile))
    }
}
/// No absolute publication ordinal duplicated in retained rows/tiles.
impl Scores {
    pub(super) fn layout(&self) -> Result<ScoreLayout, Error> {
        let sequence = add_bytes(self.rows.first().ok_or(Error::Shape)?.suffix, 1)?;
        if self.width == 0
            || self.width > puma::MAX_LANES
            || self.columns < sequence
            || self.rows.len() % sequence != 0
        {
            return Err(Error::Shape);
        }
        let layout = ScoreLayout {
            rows: self.rows.len(),
            sequence,
            columns: self.columns,
            tile: (puma::MAX_LANES / self.width).min(128),
            zero: self.zero,
        };
        for (row_index, row) in self.rows.iter().enumerate() {
            if row.node_offset
                != add_bytes(
                    (row_index / sequence)
                        .checked_mul(layout.nodes_prefix(sequence)?)
                        .ok_or(Error::Bounds)?,
                    layout.nodes_prefix(row_index % sequence)?,
                )?
                || row.prefix != self.columns - sequence + row_index % sequence + 1
            {
                return Err(Error::Binding);
            }
        }
        Ok(layout)
    }
    pub(super) fn word(&self, first: usize, index: usize) -> Result<(usize, usize), Error> {
        let row = self.rows.get(index / self.columns).ok_or(Error::Shape)?;
        let column = index % self.columns;
        if column >= row.prefix {
            return Ok(self.zero);
        }
        let tiles = &self.tiles[row.tile_start..row.tile_start + row.tile_count];
        let tile = tiles
            .iter()
            .find(|t| column >= t.column_start && column - t.column_start < t.count)
            .ok_or(Error::Binding)?;
        Ok((
            add_bytes(first, add_bytes(tile.right_offset, 1)?)?,
            column - tile.column_start,
        ))
    }
    pub(super) fn retained_bytes(&self) -> Result<usize, Error> {
        add_bytes(
            array_bytes::<ScoreRow>(self.rows.capacity())?,
            array_bytes::<ScoreTile>(self.tiles.capacity())?,
        )
    }
    pub(super) fn gather(&self, ordinal: usize) -> Result<&ScoreGather, Error> {
        if let Some(row) = self.rows.iter().find(|r| r.node_offset == ordinal) {
            return Ok(&row.gather);
        }
        let tile = self
            .tiles
            .iter()
            .find(|t| t.right_offset == ordinal)
            .ok_or(Error::Binding)?;
        Ok(&tile.gather)
    }
}
/// Reader adapter may borrow this descriptor; never materializes lane arrays.
pub(super) struct ScoreGather {
    view: PublishedView,
    start: usize,
    rows: usize,
    columns: usize,
    row_stride: usize,
    column_stride: usize,
}
impl ScoreGather {
    pub(super) fn new(
        view: PublishedView,
        start: usize,
        rows: usize,
        columns: usize,
        row_stride: usize,
        column_stride: usize,
    ) -> Result<Self, Error> {
        let lanes = rows.checked_mul(columns).ok_or(Error::Bounds)?;
        if rows == 0 || columns == 0 || lanes > puma::MAX_LANES {
            return Err(Error::Shape);
        }
        let end = add_bytes(
            add_bytes(
                start,
                (rows - 1).checked_mul(row_stride).ok_or(Error::Bounds)?,
            )?,
            (columns - 1)
                .checked_mul(column_stride)
                .ok_or(Error::Bounds)?,
        )?;
        if end >= view.lanes {
            return Err(Error::Shape);
        }
        Ok(Self {
            view,
            start,
            rows,
            columns,
            row_stride,
            column_stride,
        })
    }
    pub(super) fn len(&self) -> usize {
        self.rows * self.columns
    }
    pub(super) fn lane_ref(&self, index: usize) -> Result<(usize, usize), Error> {
        if index >= self.len() {
            return Err(Error::Binding);
        }
        let offset = add_bytes(
            (index / self.columns)
                .checked_mul(self.row_stride)
                .ok_or(Error::Bounds)?,
            (index % self.columns)
                .checked_mul(self.column_stride)
                .ok_or(Error::Bounds)?,
        )?;
        self.view.word(add_bytes(self.start, offset)?)
    }
}
/// Private reusable recorder inputs. These are mapped coordinates/bounds, NOT
/// source/proof issuers; each caller binds authentic inputs before recording.
pub(super) struct RightGather {
    pub(super) gather: ScoreGather,
    pub(super) offset: usize,
    pub(super) bounds: Interval,
    pub(super) column_start: usize,
    pub(super) count: usize,
}
pub(super) fn record_gather(
    next: &mut usize,
    gather: ScoreGather,
    bounds: Interval,
) -> Result<(usize, ScoreGather, Interval), Error> {
    let offset = *next;
    *next = add_bytes(offset, 1)?;
    Ok((offset, gather, bounds))
}
#[allow(clippy::too_many_arguments)]
pub(super) fn record_product(
    records: &mut Vec<ScoreTile>,
    next: &mut usize,
    work: &mut WorkBudget,
    left: (usize, &ScoreGather, Interval),
    right: RightGather,
    shape: MatrixShape,
    rescale: bool,
    proof: Option<RowEnergyProof>,
    ordinary: Interval,
) -> Result<usize, Error> {
    let (left, left_gather, left_bounds) = left;
    if shape.dimensions()[0] != 1
        || shape.left_elements() != left_gather.len()
        || left_gather.rows != 1
        || left_gather.columns != shape.dimensions()[1]
        || shape.right_elements() != right.gather.len()
        || shape.output_elements() != right.count
        || right.gather.rows != shape.dimensions()[1]
        || right.gather.columns != shape.dimensions()[2]
        || left >= right.offset
        || *next != add_bytes(right.offset, 1)?
    {
        return Err(Error::Binding);
    }
    let inputs = InputBounds {
        left: left_bounds,
        right: right.bounds,
    };
    if ordinary != product_bounds(inputs, shape.dimensions()[1], true, rescale, proof)? {
        return Err(Error::Binding);
    }
    if records.len() == records.capacity() {
        return Err(Error::Bounds);
    }
    let updated = work.checked_add(WorkBudget::product(right.count, rescale)?)?;
    let output = *next;
    let after = add_bytes(output, 1)?;
    records.push(ScoreTile {
        right_start: right.gather.start,
        gather: right.gather,
        column_start: right.column_start,
        count: right.count,
        right_offset: right.offset,
        left_offset: left,
        shape,
        inputs,
        proof,
        rescale,
        ordinary_bounds: ordinary,
        bounds: ordinary,
    });
    *next = after;
    *work = updated;
    Ok(output)
}
/// Bounds-only ORIGINAL mass refinement. Caller must authenticate exact native
/// probability-prefix owner/ordinal/order/width BEFORE supplying its mass.
/// This function issues no certificate and changes no recipe or work counter.
pub(super) fn refine_product_mass(
    records: &mut [ScoreTile],
    product: usize,
    right: usize,
    mass: u64,
) -> Result<(), Error> {
    let tile = records
        .iter_mut()
        .find(|t| t.right_offset.checked_add(1) == Some(product))
        .ok_or(Error::Binding)?;
    if tile.right_offset != right {
        return Err(Error::Binding);
    }
    tile.bounds = matrix::with_mass(tile.ordinary_bounds, tile.inputs.right, mass, tile.rescale)?;
    Ok(())
}
impl super::super::schedule_finalization::BorrowedLaneRefs for ScoreGather {
    fn len(&self) -> usize {
        ScoreGather::len(self)
    }
    fn lane_ref(&self, index: usize) -> Result<(usize, usize), Error> {
        ScoreGather::lane_ref(self, index)
    }
}

#[derive(Clone, Copy)]
struct Span {
    kind: u8,
    start: usize,
    len: usize,
    valid: bool,
}
impl Default for Span {
    fn default() -> Self {
        Self {
            kind: 2,
            start: 0,
            len: 0,
            valid: true,
        }
    }
}
#[derive(Clone)]
struct View {
    words: Span,
    shape: [usize; 4],
    rank: usize,
}
struct Sink<'a, 'g, 's> {
    graph: &'a CompactGraph<'g, 's>,
    result: Scores,
    pending: Option<RightGather>,
    query_start: usize,
    key_start: usize,
    prefix: usize,
    row: usize,
    sequence: usize,
}
fn score_energy(
    graph: &CompactGraph<'_, '_>,
    view: &PublishedView,
    start: usize,
    width: usize,
) -> Result<Option<u128>, Error> {
    if let Some(bound) = graph.attention_energy(view, start, width)? {
        return Ok(Some(bound));
    }
    // Genuine absence only; errors propagate. GENERAL remains an authenticated
    // input obligation, never a GENERAL certificate for the score output.
    if width == 0 || view.rank == 0 || view.logical.dimensions[view.rank as usize - 1] != width {
        return Err(Error::Shape);
    }
    if start % width != 0 {
        return Ok(None);
    }
    compact_validity::row_energy(graph, view, start / width)
}
impl MatrixSink for Sink<'_, '_, '_> {
    type Owner = usize;
    type Mass = u64;
    type Output = usize;
    fn empty(&self) -> usize {
        0
    }
    fn begin_row(&mut self, row: usize) -> Result<(), Error> {
        if row != 0 {
            Err(Error::Shape)
        } else {
            Ok(())
        }
    }
    fn mass(&self, _: usize) -> Result<Option<u64>, Error> {
        // Source imports reject row_masses and no currently supported compact
        // prefix produces one (Softmax is unsupported). Revisit with that family.
        Ok(None)
    }
    fn left(&mut self, _: usize) -> Result<usize, Error> {
        Ok(self.result.rows[self.row].node_offset)
    }
    fn right(&mut self, start: usize, count: usize) -> Result<usize, Error> {
        if self.pending.is_some() {
            return Err(Error::Binding);
        }
        let width = self.result.width;
        let key_start = add_bytes(
            self.key_start,
            start.checked_mul(width).ok_or(Error::Bounds)?,
        )?;
        let right = compact_validity::input_bounds(
            self.graph,
            &self.result.keys,
            key_start,
            count.checked_mul(width).ok_or(Error::Bounds)?,
        )?;
        let gather = ScoreGather::new(self.result.keys.clone(), key_start, width, count, 1, width)?;
        let (offset, gather, bounds) = record_gather(&mut self.result.nodes, gather, right)?;
        self.pending = Some(RightGather {
            gather,
            offset,
            bounds,
            column_start: start,
            count,
        });
        Ok(offset)
    }
    fn proof(
        &self,
        _: usize,
        start: usize,
        count: usize,
        left: usize,
        right: usize,
    ) -> Result<Option<RowEnergyProof>, Error> {
        let tile = self.pending.as_ref().ok_or(Error::Binding)?;
        if tile.column_start != start
            || tile.count != count
            || right != tile.offset
            || left != self.result.rows[self.row].node_offset
        {
            return Err(Error::Binding);
        }
        let Some(query) = score_energy(
            self.graph,
            &self.result.query,
            self.query_start,
            self.result.width,
        )?
        else {
            return Ok(None);
        };
        let mut keys = 0;
        for key in 0..count {
            let begin = add_bytes(
                tile.gather.start,
                key.checked_mul(self.result.width).ok_or(Error::Bounds)?,
            )?;
            let Some(energy) =
                score_energy(self.graph, &self.result.keys, begin, self.result.width)?
            else {
                return Ok(None);
            };
            keys = keys.max(energy);
        }
        Ok(Some(RowEnergyProof::new(
            self.result.width,
            InputBounds {
                left: self.result.rows[self.row].query_bounds,
                right: tile.bounds,
            },
            query,
            keys,
        )?))
    }
    fn product(
        &mut self,
        left: usize,
        right: usize,
        shape: MatrixShape,
        rescale: bool,
        proof: Option<RowEnergyProof>,
    ) -> Result<usize, Error> {
        let tile = self.pending.take().ok_or(Error::Binding)?;
        if !rescale
            || left != self.result.rows[self.row].node_offset
            || right != tile.offset
            || shape != MatrixShape::new(1, self.result.width, tile.count)?
        {
            return Err(Error::Binding);
        }
        let bounds = product_bounds(
            InputBounds {
                left: self.result.rows[self.row].query_bounds,
                right: tile.bounds,
            },
            self.result.width,
            true,
            rescale,
            proof,
        )?;
        let first = self.result.tiles.is_empty();
        let output = record_product(
            &mut self.result.tiles,
            &mut self.result.nodes,
            &mut self.result.work,
            (
                left,
                &self.result.rows[self.row].gather,
                self.result.rows[self.row].query_bounds,
            ),
            tile,
            shape,
            rescale,
            proof,
            bounds,
        )?;
        self.result.bounds = if first {
            bounds
        } else {
            Interval::new(
                self.result.bounds.lower().min(bounds.lower()),
                self.result.bounds.upper().max(bounds.upper()),
            )?
        };
        Ok(output)
    }
    fn refine_mass(
        &mut self,
        output: usize,
        right: usize,
        mass: u64,
        rescale: bool,
    ) -> Result<(), Error> {
        // Current scores mass() is None; no mass evidence is issued here.
        if !rescale {
            return Err(Error::Binding);
        }
        refine_product_mass(&mut self.result.tiles, output, right, mass)
    }
    fn append(&mut self, output: &mut usize, _: usize, lanes: usize) {
        *output += lanes;
    }
}
impl AttentionSink for Sink<'_, '_, '_> {
    type View = View;
    type Words = Span;
    type Word = ();
    type Owner = usize;
    fn shape(view: &View) -> &[usize] {
        &view.shape[..view.rank]
    }
    fn words(view: &View) -> &Span {
        &view.words
    }
    fn len(words: &Span) -> usize {
        words.len
    }
    fn slice(words: &Span, range: std::ops::Range<usize>) -> Span {
        Span {
            kind: words.kind,
            start: words.start.saturating_add(range.start),
            len: range.end.saturating_sub(range.start),
            valid: words.valid
                && range.start <= range.end
                && range.end <= words.len
                && words.start.checked_add(range.start).is_some(),
        }
    }
    fn rows(words: &Span, width: usize) -> impl Iterator<Item = Span> + '_ {
        (0..words.len)
            .step_by(width.max(1))
            .map(move |start| Self::slice(words, start..(start + width).min(words.len)))
    }
    fn chunks(words: &Span, width: usize) -> impl Iterator<Item = Span> + '_ {
        Self::rows(words, width)
    }
    fn at(_: &Span, _: usize) {
        unreachable!("closed scores-only adapter")
    }
    fn push(_: &mut Span, _: ()) {
        unreachable!("closed scores-only adapter")
    }
    fn append_owner(_: &mut Span, _: usize, _: usize) {
        unreachable!("closed scores-only adapter")
    }
    fn extend(words: &mut Span, more: Span) {
        words.valid &= more.valid
            && words.kind == 2
            && more.kind == 2
            && words.len.checked_add(more.len).is_some();
        words.len = words.len.saturating_add(more.len);
    }
    fn repeat(words: &mut Span, _: (), count: usize) {
        words.valid &= words.len.checked_add(count).is_some();
        words.len = words.len.saturating_add(count);
    }
    fn view(words: Span, shape: &[usize]) -> View {
        let mut dims = [0; 4];
        let rank = shape.len().min(4);
        dims[..rank].copy_from_slice(&shape[..rank]);
        View {
            words: Span {
                valid: words.valid && shape.len() <= 4,
                ..words
            },
            shape: dims,
            rank,
        }
    }
    fn rotary_row(&mut self, _: &Span, _: f32, _: u32, _: usize) -> Result<usize, Error> {
        Err(Error::Binding)
    }
    fn values(&mut self, _: &View, _: &View) -> Result<Span, Error> {
        Err(Error::Binding)
    }
    fn scale_coefficient(_: u64) -> i64 {
        unreachable!("closed scores-only adapter")
    }
    fn scale_chunk(&mut self, _: &Span, _: i64) -> Result<usize, Error> {
        Err(Error::Binding)
    }
    fn softmax_row(&mut self, _: &Span, _: usize) -> Result<usize, Error> {
        Err(Error::Binding)
    }
    fn scores(&mut self, query: &View, keys: &View) -> Result<Span, Error> {
        let width = self.result.width;
        if !query.words.valid
            || !keys.words.valid
            || query.words.kind != 0
            || keys.words.kind != 1
            || Self::shape(query) != [1, width]
            || keys.rank != 2
            || keys.shape[1] != width
            || query.words.len != width
            || keys.words.len != keys.shape[0].checked_mul(width).ok_or(Error::Bounds)?
        {
            return Err(Error::Shape);
        }
        self.query_start = query.words.start;
        self.key_start = keys.words.start;
        self.prefix = keys.shape[0];
        self.row = self.result.rows.len();
        let r = self.row % self.sequence;
        if self.prefix + self.sequence - r - 1 != self.result.columns
            || self.result.rows.len() == self.result.rows.capacity()
        {
            return Err(Error::Shape);
        }
        let bounds = compact_validity::input_bounds(
            self.graph,
            &self.result.query,
            self.query_start,
            width,
        )?;
        let gather = ScoreGather::new(
            self.result.query.clone(),
            self.query_start,
            1,
            width,
            width,
            1,
        )?;
        let (node_offset, gather, bounds) = record_gather(&mut self.result.nodes, gather, bounds)?;
        self.result.rows.push(ScoreRow {
            gather,
            query_start: self.query_start,
            key_start: self.key_start,
            prefix: self.prefix,
            suffix: self.sequence - r - 1,
            node_offset,
            tile_start: self.result.tiles.len(),
            tile_count: 0,
            query_bounds: bounds,
        });
        let prefix = self.prefix;
        let count = matrix::lower(self, 1, width, prefix, true)?;
        if count != self.prefix || self.pending.is_some() {
            return Err(Error::Binding);
        }
        self.result.rows[self.row].tile_count =
            self.result.tiles.len() - self.result.rows[self.row].tile_start;
        Ok(Span {
            kind: 2,
            start: 0,
            len: count,
            valid: true,
        })
    }
}

pub(super) fn geometry(
    query: &PublishedView,
    keys: &PublishedView,
    shape: &[usize],
    group: usize,
) -> Result<(usize, usize, usize, usize, usize), Error> {
    let q = &query.logical.dimensions;
    let k = &keys.logical.dimensions;
    if query.rank != 4
        || keys.rank != 4
        || q[0] != 1
        || k[0] != 1
        || group == 0
        || q[1] / group != k[1]
        || q[3] != k[3]
        || q[3] == 0
        || q[3] > puma::MAX_LANES
        || q[2] == 0
        || k[2] < q[2]
        || shape != [1, q[1], q[2], k[2]]
    {
        return Err(Error::Shape);
    }
    let tile = (puma::MAX_LANES / q[3]).min(128);
    let mut tiles = 0usize;
    for row in 0..q[2] {
        let prefix = k[2] - q[2] + row + 1;
        tiles = add_bytes(tiles, prefix.div_ceil(tile))?;
    }
    Ok((
        q[1].checked_mul(q[2]).ok_or(Error::Bounds)?,
        tiles.checked_mul(q[1]).ok_or(Error::Bounds)?,
        q[3],
        q[2],
        k[2],
    ))
}
fn same_input(a: &PublishedView, b: &PublishedView) -> bool {
    Arc::ptr_eq(&a.origin, &b.origin)
        && a.slot == b.slot
        && a.mapping == b.mapping
        && a.order == b.order
        && a.start == b.start
        && a.lanes == b.lanes
        && a.rank == b.rank
        && a.logical == b.logical
        && a.bounds == b.bounds
}
impl CompactGraph<'_, '_> {
    fn score_peak(&self, rows: usize, tiles: usize, attempt: usize) -> Result<usize, Error> {
        let backing = add_bytes(
            array_bytes::<ScoreRow>(rows)?,
            array_bytes::<ScoreTile>(tiles)?,
        )?;
        let mut peak = add_bytes(
            add_bytes(self.baseline()?, self.outside_owner_bytes)?,
            backing,
        )?;
        macro_rules! slot {
            ($bytes:expr) => {
                peak = add_bytes(peak, $bytes)?;
            };
        }
        slot!(size_of::<Sink<'_, '_, '_>>());
        slot!(2 * size_of::<Scores>());
        slot!(2 * size_of::<Result<Scores, Error>>());
        slot!(2 * size_of::<Vec<ScoreRow>>());
        slot!(2 * size_of::<Vec<ScoreTile>>());
        slot!(2 * size_of::<ScoreRow>());
        slot!(2 * size_of::<ScoreTile>());
        // Two source views, two row views, constructor/return transition slots.
        slot!(6 * size_of::<View>());
        slot!(2 * size_of::<Result<View, Error>>());
        slot!(2 * size_of::<ScoreLayout>());
        slot!(2 * size_of::<Option<RightGather>>());
        slot!(2 * size_of::<RightGather>());
        slot!(2 * size_of::<(usize, &ScoreGather, Interval)>());
        slot!(2 * size_of::<Option<RowEnergyProof>>());
        slot!(2 * size_of::<Result<Option<RowEnergyProof>, Error>>());
        slot!(2 * size_of::<ScoreGather>());
        slot!(3 * size_of::<WorkBudget>());
        slot!(3 * size_of::<MatrixShape>());
        slot!(2 * size_of::<InputBounds>());
        slot!(2 * size_of::<Result<Interval, Error>>());
        slot!(attempt.checked_mul(3).ok_or(Error::Bounds)?);
        // baseline() already retains the caller publication controls once.
        Ok(peak)
    }
    pub(super) fn lower_attention_scores(
        &self,
        op: &pllm_models::ModelOperation,
        shape: &[usize],
        query: &PublishedView,
        keys: &PublishedView,
    ) -> Result<Scores, Error> {
        let context = self.request_admission.ok_or(Error::Binding)?;
        let mut attempt = context.attempt();
        context.check_open()?;
        context.check_final_work(self.work)?;
        self.validate_view(query)?;
        self.validate_view(keys)?;
        if op.operator != Op::AttentionScores {
            return Err(Error::Binding);
        }
        let original = self
            .frontier
            .operations()
            .nth(self.cursor)
            .ok_or(Error::Binding)?
            .1;
        if original.operator != op.operator || original.attributes != op.attributes {
            return Err(Error::Binding);
        }
        let (expected_query, expected_keys) = self.frontier.inputs(original);
        if !same_input(query, expected_query.ok_or(Error::Binding)?)
            || !same_input(keys, expected_keys.ok_or(Error::Binding)?)
        {
            return Err(Error::Binding);
        }
        let group = usize::try_from(
            op.attributes
                .get("group_size")
                .and_then(|v| v.as_u64())
                .ok_or(Error::Binding)?,
        )
        .map_err(|_| Error::Bounds)?;
        let (rows, tiles, width, sequence, columns) = geometry(query, keys, shape, group)?;
        let peak = self.score_peak(rows, tiles, size_of_val(&attempt))?;
        self.admit_total(peak)?;
        context.admit_phase(peak, 0)?;
        // Finite mapped source/proof walks, including per-key intact row checks.
        let depth = add_bytes(self.ops.len(), 1)?;
        let visits = rows
            .checked_mul(columns)
            .and_then(|n| n.checked_mul(width))
            .and_then(|n| n.checked_mul(depth))
            .and_then(|n| n.checked_mul(depth))
            .and_then(|n| n.checked_mul(32))
            .ok_or(Error::Bounds)?;
        context
            .plan()
            .prepay_compiler_metadata(
                &context.reservation(),
                u64::try_from(visits).map_err(|_| Error::Bounds)?,
            )
            .map_err(|_| Error::Bounds)?;
        let mut row_records = Vec::new();
        row_records
            .try_reserve_exact(rows)
            .map_err(|_| Error::Allocation)?;
        let mut tile_records = Vec::new();
        tile_records
            .try_reserve_exact(tiles)
            .map_err(|_| Error::Allocation)?;
        if row_records.capacity() != rows || tile_records.capacity() != tiles {
            return Err(Error::Allocation);
        }
        let zero = (
            add_bytes(
                self.request_prelude.as_ref().ok_or(Error::Binding)?.first(),
                1,
            )?,
            0,
        );
        let result = Scores {
            query: query.clone(),
            keys: keys.clone(),
            rows: row_records,
            tiles: tile_records,
            width,
            columns,
            zero,
            nodes: 0,
            work: WorkBudget::default(),
            bounds: Interval::new(0, 0)?,
        };
        let mut sink = Sink {
            graph: self,
            result,
            pending: None,
            query_start: 0,
            key_start: 0,
            prefix: 0,
            row: 0,
            sequence,
        };
        let q = Sink::view(
            Span {
                kind: 0,
                start: 0,
                len: query.lanes,
                valid: true,
            },
            &query.logical.dimensions[..4],
        );
        let k = Sink::view(
            Span {
                kind: 1,
                start: 0,
                len: keys.lanes,
                valid: true,
            },
            &keys.logical.dimensions[..4],
        );
        let output = attention::scores(
            &mut sink,
            op,
            shape,
            Some(&q),
            Some(&k),
            sequence,
            columns - sequence,
            columns,
            (),
        )?;
        if !output.words.valid
            || output.words.len != rows.checked_mul(columns).ok_or(Error::Bounds)?
            || sink.result.rows.len() != rows
            || sink.result.tiles.len() != tiles
        {
            return Err(Error::Binding);
        }
        self.admit_pending_work(sink.result.work)?;
        sink.result.layout()?;
        if sequence > 1 {
            sink.result.bounds = Interval::new(
                sink.result.bounds.lower().min(0),
                sink.result.bounds.upper().max(0),
            )?;
        }
        for tile in &sink.result.tiles {
            // Original driver summary, actual full/tail output count. Existing
            // persistent compiler protocol ledger, not a tensor-ticket substitute.
            context.admit_protocol_work(WorkBudget::product(tile.count, true)?)?;
        }
        context.check_final_work(self.work.checked_add(sink.result.work)?)?;
        #[cfg(test)]
        if FAIL_AFTER_DEBIT.with(|f| f.replace(false)) {
            return Err(Error::Allocation);
        }
        context.retain(sink.result.retained_bytes()?)?;
        #[cfg(test)]
        if FAIL_PUBLICATION.with(|f| f.replace(false)) {
            return Err(Error::Allocation);
        }
        attempt.commit();
        Ok(sink.result)
    }
}
#[cfg(test)]
thread_local! {
    static FAIL_AFTER_DEBIT:std::cell::Cell<bool>=const{std::cell::Cell::new(false)};
    static FAIL_PUBLICATION:std::cell::Cell<bool>=const{std::cell::Cell::new(false)};
}
#[cfg(test)]
#[path = "compact_attention_scores_tests.rs"]
mod tests;
