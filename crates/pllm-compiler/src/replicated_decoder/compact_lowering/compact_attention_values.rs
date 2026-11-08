//! Original small-inner Values traversal over owning mapped records. Parent
//! publication/reader glue is E-owned. No mass issuer or expanded entry owner.
use super::super::graph::attention_driver::{self as attention, AttentionSink};
use super::super::private_matrix_driver::{self as matrix, Sink as MatrixSink};
use super::compact_attention_scores::{
    record_gather, record_product, refine_product_mass, RightGather, ScoreGather, ScoreTile,
};
use super::compact_attention_softmax::{RowView, Rows as SoftmaxRows};
use super::*;
use crate::replicated_decoder::program::request_origin::compact_storage::ControllerAccounting;
use puma::multiply::norm::RowEnergyProof;
use pllm_garble::shared_allocation::OwnedExternalCharge;
use std::mem::{size_of, size_of_val};

/// Private creation-bound evidence. No constructor accepts a mass integer.
/// RowView owns the SAME original invocation Arc (including its certificate).
pub(super) struct Probability<'s, G = OwnedExternalCharge> {
    native: RowView<'s, G>,
    predecessor: PublishedView,
    program: Arc<tensor::Program>,
    prefix_offset: usize,
    tensor: usize,
    gather: usize,
    width: usize,
    mass: u64,
    input: Interval,
    output: Interval,
}
impl<'s, G: ControllerAccounting> Probability<'s, G> {
    fn from_rows(
        rows: &SoftmaxRows<'s, G>,
        prefix: &PublishedView,
        inner: usize,
    ) -> Result<Self, Error> {
        let actual = rows
            .lookup_published(prefix, inner)?
            .ok_or(Error::Binding)?;
        let index = (0..rows.len())
            .find_map(|i| match rows.row(i) {
                Ok(row) if std::ptr::eq(row, actual) => Some(Ok(i)),
                Ok(_) => None,
                Err(e) => Some(Err(e)),
            })
            .ok_or(Error::Binding)??;
        let native = rows.prefix(index)?;
        let borrowed = rows.lookup(&native, inner)?.ok_or(Error::Binding)?;
        if !std::ptr::eq(borrowed, actual)
            || actual.native_width() != inner
            || actual.program().output_lanes() != inner
        {
            return Err(Error::Binding);
        }
        Ok(Self {
            native,
            predecessor: actual.predecessor().clone(),
            program: Arc::clone(actual.program()),
            prefix_offset: actual.prefix_offset(),
            tensor: actual.tensor_ordinal(),
            gather: actual.gather_ordinal(),
            width: inner,
            mass: actual.mass_bound(),
            input: actual.input_bounds(),
            output: actual.output_bounds(),
        })
    }
    fn check(&self) -> Result<(), Error> {
        for k in 0..self.width {
            if self.native.word(k)? != (self.tensor, k) {
                return Err(Error::Binding);
            }
        }
        Ok(())
    }
    pub(super) fn program(&self) -> &Arc<tensor::Program> {
        &self.program
    }
    pub(super) fn predecessor(&self) -> &PublishedView {
        &self.predecessor
    }
    pub(super) fn tensor_ordinal(&self) -> usize {
        self.tensor
    }
    pub(super) fn gather_ordinal(&self) -> usize {
        self.gather
    }
    pub(super) fn input_bounds(&self) -> Interval {
        self.input
    }
    pub(super) fn prefix_offset(&self) -> usize {
        self.prefix_offset
    }
    pub(super) fn predecessor_programs(&self) -> impl Iterator<Item = &Arc<tensor::Program>> {
        self.native.predecessor_programs()
    }
}
pub(super) struct ValueRow<'s, G = OwnedExternalCharge> {
    pub(super) gather: ScoreGather,
    pub(super) gather_offset: usize,
    pub(super) input_bounds: Interval,
    pub(super) inner: usize,
    pub(super) probability_start: usize,
    pub(super) value_start: usize,
    pub(super) tile_start: usize,
    pub(super) tile_count: usize,
    pub(super) probability: Option<Probability<'s, G>>,
}
pub(super) struct Values<'s, G = OwnedExternalCharge> {
    pub(super) probabilities: PublishedView,
    pub(super) values: PublishedView,
    pub(super) rows: Vec<ValueRow<'s, G>>,
    pub(super) tiles: Vec<ScoreTile>,
    pub(super) width: usize,
    pub(super) keys: usize,
    pub(super) sequence: usize,
    pub(super) nodes: usize,
    pub(super) work: WorkBudget,
    pub(super) bounds: Interval,
}

pub(super) struct OwnedValueRow {
    pub(super) gather: ScoreGather,
    pub(super) gather_offset: usize,
    pub(super) input_bounds: Interval,
    pub(super) inner: usize,
    pub(super) probability_start: usize,
    pub(super) value_start: usize,
    pub(super) tile_start: usize,
    pub(super) tile_count: usize,
    pub(super) probability: Option<super::compact_attention_softmax::owning::RowIndex>,
}
pub(super) struct OwnedValues {
    pub(super) probabilities: PublishedView,
    pub(super) values: PublishedView,
    pub(super) rows: Vec<OwnedValueRow>,
    pub(super) tiles: Vec<ScoreTile>,
    pub(super) width: usize,
    pub(super) keys: usize,
    pub(super) sequence: usize,
    pub(super) nodes: usize,
    pub(super) work: WorkBudget,
    pub(super) bounds: Interval,
}
impl<'s, G: ControllerAccounting> Values<'s, G> {
    pub(super) fn into_owned(
        self,
        context: &EmissionContext<'s, G>,
        conversion: &super::compact_attention_softmax::owning::Conversion<'_, 's, G>,
    ) -> Result<OwnedValues, Error> {
        let retained = add_bytes(
            array_bytes::<OwnedValueRow>(self.rows.len())?,
            2 * size_of::<OwnedValues>(),
        )?;
        #[cfg(test)]
        let _outside_pressure =
            super::frozen_region::owning::conversion_probe::pressure(context, retained)?;
        context.admit_phase(retained, 0)?;
        context.retain(retained)?;
        let visits = self
            .rows
            .iter()
            .try_fold(self.rows.len(), |n, row| add_bytes(n, row.inner))?;
        context.prepay_metadata(u64::try_from(visits.checked_mul(8).ok_or(Error::Bounds)?).map_err(|_| Error::Bounds)?)?;
        let mut rows = Vec::new();
        #[cfg(test)]
        super::frozen_region::owning::conversion_probe::allocation();
        rows.try_reserve_exact(self.rows.len())
            .map_err(|_| Error::Allocation)?;
        if rows.capacity() != self.rows.len() {
            return Err(Error::Bounds);
        }
        for row in self.rows {
            let probability = match row.probability {
                None => None,
                Some(probability) => {
                    probability.check()?;
                    Some(conversion.index(probability.native)?)
                }
            };
            rows.push(OwnedValueRow {
                gather: row.gather,
                gather_offset: row.gather_offset,
                input_bounds: row.input_bounds,
                inner: row.inner,
                probability_start: row.probability_start,
                value_start: row.value_start,
                tile_start: row.tile_start,
                tile_count: row.tile_count,
                probability,
            });
        }
        Ok(OwnedValues {
            probabilities: self.probabilities,
            values: self.values,
            rows,
            tiles: self.tiles,
            width: self.width,
            keys: self.keys,
            sequence: self.sequence,
            nodes: self.nodes,
            work: self.work,
            bounds: self.bounds,
        })
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ValueLayout {
    rows: usize,
    width: usize,
    keys: usize,
    sequence: usize,
}
impl ValueLayout {
    fn before(self, rows: usize) -> Result<usize, Error> {
        let mut nodes = 0;
        for r in 0..rows {
            let inner = self.keys - self.sequence + r + 1;
            let tile = (puma::MAX_LANES / inner).min(128);
            nodes = add_bytes(
                nodes,
                add_bytes(
                    1,
                    self.width
                        .div_ceil(tile)
                        .checked_mul(2)
                        .ok_or(Error::Bounds)?,
                )?,
            )?;
        }
        Ok(nodes)
    }
    pub(super) fn word(self, first: usize, index: usize) -> Result<(usize, usize), Error> {
        let row = index / self.width;
        let column = index % self.width;
        if row >= self.rows {
            return Err(Error::Shape);
        }
        let r = row % self.sequence;
        let tile = (puma::MAX_LANES / (self.keys - self.sequence + r + 1)).min(128);
        let start = add_bytes(
            (row / self.sequence)
                .checked_mul(self.before(self.sequence)?)
                .ok_or(Error::Bounds)?,
            self.before(r)?,
        )?;
        Ok((
            add_bytes(
                first,
                add_bytes(
                    start,
                    add_bytes(2, (column / tile).checked_mul(2).ok_or(Error::Bounds)?)?,
                )?,
            )?,
            column % tile,
        ))
    }
}
impl<G: ControllerAccounting> Values<'_, G> {
    pub(super) fn layout(&self) -> Result<ValueLayout, Error> {
        if self.sequence == 0
            || self.keys < self.sequence
            || self.keys > puma::MAX_LANES
            || self.width == 0
            || self.rows.len() % self.sequence != 0
        {
            return Err(Error::Shape);
        }
        let layout = ValueLayout {
            rows: self.rows.len(),
            width: self.width,
            keys: self.keys,
            sequence: self.sequence,
        };
        for (index, row) in self.rows.iter().enumerate() {
            let expected = add_bytes(
                (index / self.sequence)
                    .checked_mul(layout.before(self.sequence)?)
                    .ok_or(Error::Bounds)?,
                layout.before(index % self.sequence)?,
            )?;
            if row.gather_offset != expected
                || row.inner != self.keys - self.sequence + index % self.sequence + 1
            {
                return Err(Error::Binding);
            }
        }
        Ok(layout)
    }
    pub(super) fn word(&self, first: usize, index: usize) -> Result<(usize, usize), Error> {
        let row = self.rows.get(index / self.width).ok_or(Error::Shape)?;
        let column = index % self.width;
        let tile = self.tiles[row.tile_start..row.tile_start + row.tile_count]
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
            array_bytes::<ValueRow<'_, G>>(self.rows.capacity())?,
            array_bytes::<ScoreTile>(self.tiles.capacity())?,
        )
    }
    pub(super) fn gather(&self, offset: usize) -> Result<&ScoreGather, Error> {
        if let Some(row) = self.rows.iter().find(|r| r.gather_offset == offset) {
            return Ok(&row.gather);
        }
        Ok(&self
            .tiles
            .iter()
            .find(|t| t.right_offset == offset)
            .ok_or(Error::Binding)?
            .gather)
    }
}

// A single coordinate at a time, not a Words/Node array. Selected spans verify
// the ORIGINAL d-outer/k-inner loop before accepting its affine descriptor.
#[derive(Clone, Copy)]
struct Word {
    index: usize,
    width: usize,
    valid: bool,
}
#[derive(Clone, Copy)]
struct Span {
    kind: u8,
    start: usize,
    len: usize,
    width: usize,
    inner: usize,
    valid: bool,
}
impl Default for Span {
    fn default() -> Self {
        Self {
            kind: 2,
            start: 0,
            len: 0,
            width: 0,
            inner: 0,
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
struct SinkFor<'a, 'g, 's, G: ControllerAccounting = OwnedExternalCharge> {
    graph: &'a CompactGraph<'g, 's, G>,
    result: Values<'s, G>,
    pending: Option<RightGather>,
    current: usize,
}
#[cfg(test)]
type Sink<'a, 'g, 's> = SinkFor<'a, 'g, 's, OwnedExternalCharge>;
impl<G: ControllerAccounting> MatrixSink for SinkFor<'_, '_, '_, G> {
    type Owner = usize;
    type Mass = usize;
    type Output = usize;
    fn empty(&self) -> usize {
        0
    }
    fn begin_row(&mut self, row: usize) -> Result<(), Error> {
        if row == 0 {
            Ok(())
        } else {
            Err(Error::Shape)
        }
    }
    fn mass(&self, _: usize) -> Result<Option<usize>, Error> {
        // Copyable invocation index, NOT caller-supplied numerical authority.
        if let Some(proof) = &self.result.rows[self.current].probability {
            proof.check()?;
            Ok(Some(self.current))
        } else {
            Ok(None)
        }
    }
    fn left(&mut self, _: usize) -> Result<usize, Error> {
        Ok(self.result.rows[self.current].gather_offset)
    }
    fn right(&mut self, start: usize, count: usize) -> Result<usize, Error> {
        if self.pending.is_some() {
            return Err(Error::Binding);
        }
        let row = &self.result.rows[self.current];
        let base = add_bytes(row.value_start, start)?;
        let mut bounds = None::<Interval>;
        for k in 0..row.inner {
            let offset = add_bytes(base, k.checked_mul(self.result.width).ok_or(Error::Bounds)?)?;
            let b = compact_validity::input_bounds(self.graph, &self.result.values, offset, count)?;
            bounds = Some(match bounds {
                None => b,
                Some(a) => Interval::new(a.lower().min(b.lower()), a.upper().max(b.upper()))?,
            });
        }
        let gather = ScoreGather::new(
            self.result.values.clone(),
            base,
            row.inner,
            count,
            self.result.width,
            1,
        )?;
        let (offset, gather, bounds) =
            record_gather(&mut self.result.nodes, gather, bounds.ok_or(Error::Shape)?)?;
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
        _: usize,
        _: usize,
        _: usize,
        _: usize,
    ) -> Result<Option<RowEnergyProof>, Error> {
        // Original Values private-matrix path has attention=false. This is NOT
        // a fallback for failed native probability/mass association.
        Ok(None)
    }
    fn product(
        &mut self,
        left: usize,
        right: usize,
        shape: MatrixShape,
        rescale: bool,
        proof: Option<RowEnergyProof>,
    ) -> Result<usize, Error> {
        let pending = self.pending.take().ok_or(Error::Binding)?;
        let row = &self.result.rows[self.current];
        if left != row.gather_offset
            || right != pending.offset
            || shape != MatrixShape::new(1, row.inner, pending.count)?
            || !rescale
            || proof.is_some()
        {
            return Err(Error::Binding);
        }
        // GENERAL/ordinary Product MUST succeed before any mass refinement.
        let ordinary = product_bounds(
            InputBounds {
                left: row.input_bounds,
                right: pending.bounds,
            },
            row.inner,
            true,
            true,
            None,
        )?;
        record_product(
            &mut self.result.tiles,
            &mut self.result.nodes,
            &mut self.result.work,
            (left, &row.gather, row.input_bounds),
            pending,
            shape,
            rescale,
            None,
            ordinary,
        )
    }
    fn refine_mass(
        &mut self,
        product: usize,
        right: usize,
        row_index: usize,
        rescale: bool,
    ) -> Result<(), Error> {
        if row_index != self.current || !rescale {
            return Err(Error::Binding);
        }
        let row = &self.result.rows[row_index];
        let proof = row.probability.as_ref().ok_or(Error::Binding)?;
        proof.check()?;
        let tile = self.result.tiles.last().ok_or(Error::Binding)?;
        if tile.left_offset != row.gather_offset
            || tile.right_offset != right
            || tile.right_offset.checked_add(1) != Some(product)
            || tile.proof.is_some()
            || !tile.rescale
        {
            return Err(Error::Binding);
        }
        refine_product_mass(&mut self.result.tiles, product, right, proof.mass)
    }
    fn append(&mut self, out: &mut usize, _: usize, lanes: usize) {
        *out += lanes;
    }
}
impl<G: ControllerAccounting> AttentionSink for SinkFor<'_, '_, '_, G> {
    type View = View;
    type Words = Span;
    type Word = Word;
    type Owner = usize;
    fn shape(v: &View) -> &[usize] {
        &v.shape[..v.rank]
    }
    fn words(v: &View) -> &Span {
        &v.words
    }
    fn len(w: &Span) -> usize {
        w.len
    }
    fn slice(w: &Span, range: std::ops::Range<usize>) -> Span {
        Span {
            start: w.start.saturating_add(range.start),
            len: range.end.saturating_sub(range.start),
            valid: w.valid
                && range.start <= range.end
                && range.end <= w.len
                && w.start.checked_add(range.start).is_some(),
            ..*w
        }
    }
    fn rows(w: &Span, width: usize) -> impl Iterator<Item = Span> + '_ {
        (0..w.len)
            .step_by(width.max(1))
            .map(move |s| Self::slice(w, s..(s + width).min(w.len)))
    }
    fn chunks(w: &Span, width: usize) -> impl Iterator<Item = Span> + '_ {
        Self::rows(w, width)
    }
    fn at(w: &Span, index: usize) -> Word {
        Word {
            index: w.start.saturating_add(index),
            width: w.width,
            valid: w.valid && w.kind == 1 && index < w.len && w.start.checked_add(index).is_some(),
        }
    }
    fn push(w: &mut Span, word: Word) {
        w.valid &= word.valid && word.width > 0;
        if w.len == 0 {
            w.kind = 3;
            w.start = word.index;
            w.width = word.width;
        }
        w.valid &= w.kind == 3 && w.width == word.width;
        if w.len > 0 && w.inner == 0 {
            let expected = w
                .len
                .checked_mul(w.width)
                .and_then(|n| w.start.checked_add(n));
            if expected != Some(word.index) {
                if w.start.checked_add(1) == Some(word.index) {
                    w.inner = w.len;
                } else {
                    w.valid = false;
                }
            }
        }
        if w.inner != 0 {
            let expected = (w.len % w.inner)
                .checked_mul(w.width)
                .and_then(|n| n.checked_add(w.len / w.inner))
                .and_then(|n| w.start.checked_add(n));
            w.valid &= expected == Some(word.index);
        }
        w.valid &= w.len.checked_add(1).is_some();
        w.len = w.len.saturating_add(1);
    }
    fn extend(w: &mut Span, more: Span) {
        w.valid &= more.valid
            && (w.kind == 2 || w.kind == 4)
            && more.kind == 4
            && w.len.checked_add(more.len).is_some();
        w.kind = 4;
        w.len = w.len.saturating_add(more.len);
    }
    fn repeat(_: &mut Span, _: Word, _: usize) {
        unreachable!("closed Values adapter")
    }
    fn append_owner(_: &mut Span, _: usize, _: usize) {
        unreachable!("closed Values adapter")
    }
    fn view(mut words: Span, shape: &[usize]) -> View {
        let mut dimensions = [0; 4];
        let rank = shape.len().min(4);
        dimensions[..rank].copy_from_slice(&shape[..rank]);
        words.valid &= shape.len() <= 4;
        if words.kind == 3 {
            words.valid &= shape.len() == 2
                && shape[0] == words.width
                && shape[0].checked_mul(shape[1]) == Some(words.len)
                && (words.inner == shape[1] || (words.width == 1 && words.inner == 0));
            words.inner = shape.get(1).copied().unwrap_or(0);
        }
        View {
            words,
            shape: dimensions,
            rank,
        }
    }
    fn rotary_row(&mut self, _: &Span, _: f32, _: u32, _: usize) -> Result<usize, Error> {
        Err(Error::Binding)
    }
    fn scores(&mut self, _: &View, _: &View) -> Result<Span, Error> {
        Err(Error::Binding)
    }
    fn scale_coefficient(_: u64) -> i64 {
        unreachable!("closed Values adapter")
    }
    fn scale_chunk(&mut self, _: &Span, _: i64) -> Result<usize, Error> {
        Err(Error::Binding)
    }
    fn softmax_row(&mut self, _: &Span, _: usize) -> Result<usize, Error> {
        Err(Error::Binding)
    }
    fn values(&mut self, probability: &View, values: &View) -> Result<Span, Error> {
        self.current = self.result.rows.len();
        let width = self.result.width;
        let sequence = self.result.sequence;
        let keys = self.result.keys;
        let row = self.current % sequence;
        let head = self.current / sequence;
        let inner = keys - sequence + row + 1;
        let op = self
            .graph
            .frontier
            .operations()
            .nth(self.graph.cursor)
            .ok_or(Error::Binding)?
            .1;
        let group = usize::try_from(
            op.attributes
                .get("group_size")
                .and_then(|n| n.as_u64())
                .ok_or(Error::Binding)?,
        )
        .map_err(|_| Error::Bounds)?;
        let pstart = self.current.checked_mul(keys).ok_or(Error::Bounds)?;
        let vstart = (head / group)
            .checked_mul(keys)
            .and_then(|n| n.checked_mul(width))
            .ok_or(Error::Bounds)?;
        if !probability.words.valid
            || !values.words.valid
            || probability.words.kind != 0
            || probability.words.start != pstart
            || Self::shape(probability) != [1, inner]
            || probability.words.len != inner
            || values.words.kind != 3
            || values.words.start != vstart
            || Self::shape(values) != [width, inner]
            || values.words.inner != inner
            || self.result.rows.len() == self.result.rows.capacity()
        {
            return Err(Error::Binding);
        }
        // Exact original driver-selected prefix, NOT trimming arbitrary zeroes.
        let end = add_bytes(pstart, inner)?;
        if end > self.result.probabilities.lanes {
            return Err(Error::Shape);
        }
        let prefix = PublicationSink::slice(&self.result.probabilities, pstart, end);
        let evidence = match self.graph.softmax_rows_for(&self.result.probabilities)? {
            None => None,
            Some(rows) => Some(Probability::from_rows(rows, &prefix, inner)?),
        };
        let bounds = match &evidence {
            Some(proof) => proof.output,
            None => compact_validity::input_bounds(
                self.graph,
                &self.result.probabilities,
                pstart,
                inner,
            )?,
        };
        let gather = ScoreGather::new(
            self.result.probabilities.clone(),
            pstart,
            1,
            inner,
            inner,
            1,
        )?;
        let (offset, gather, input_bounds) = record_gather(&mut self.result.nodes, gather, bounds)?;
        let begin = self.result.tiles.len();
        self.result.rows.push(ValueRow {
            gather,
            gather_offset: offset,
            input_bounds,
            inner,
            probability_start: pstart,
            value_start: vstart,
            tile_start: begin,
            tile_count: 0,
            probability: evidence,
        });
        let produced = matrix::lower(self, 1, inner, width, true)?;
        if produced != width || self.pending.is_some() {
            return Err(Error::Binding);
        }
        self.result.rows[self.current].tile_count = self.result.tiles.len() - begin;
        Ok(Span {
            kind: 4,
            start: 0,
            len: width,
            width: 0,
            inner: 0,
            valid: true,
        })
    }
}

pub(super) fn geometry(
    p: &PublishedView,
    v: &PublishedView,
    shape: &[usize],
    group: usize,
) -> Result<(usize, usize, usize, usize, usize), Error> {
    let a = &p.logical.dimensions;
    let b = &v.logical.dimensions;
    if p.rank != 4
        || v.rank != 4
        || a[0] != 1
        || b[0] != 1
        || group == 0
        || a[1] / group != b[1]
        || a[2] == 0
        || a[3] != b[2]
        || a[3] < a[2]
        || a[3] > puma::MAX_LANES
        || b[3] == 0
        || shape != [1, a[1], a[2], b[3]]
    {
        return Err(Error::Shape);
    }
    let mut tiles = 0;
    for r in 0..a[2] {
        let inner = a[3] - a[2] + r + 1;
        let tile = (puma::MAX_LANES / inner).min(128);
        tiles = add_bytes(tiles, b[3].div_ceil(tile))?;
    }
    Ok((
        a[1].checked_mul(a[2]).ok_or(Error::Bounds)?,
        tiles.checked_mul(a[1]).ok_or(Error::Bounds)?,
        b[3],
        a[2],
        a[3],
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
impl<'s, G: ControllerAccounting> CompactGraph<'_, 's, G> {
    fn values_peak(&self, rows: usize, tiles: usize, attempt: usize) -> Result<usize, Error> {
        let backing = add_bytes(
            array_bytes::<ValueRow<'s, G>>(rows)?,
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
        slot!(size_of::<SinkFor<'_, '_, '_, G>>());
        slot!(2 * size_of::<Values<'s, G>>());
        slot!(2 * size_of::<Result<Values<'s, G>, Error>>());
        slot!(2 * size_of::<Vec<ValueRow<'s, G>>>());
        slot!(2 * size_of::<Vec<ScoreTile>>());
        slot!(2 * size_of::<ValueRow<'s, G>>());
        slot!(2 * size_of::<ScoreTile>());
        slot!(2 * size_of::<RightGather>());
        slot!(2 * size_of::<Option<RightGather>>());
        slot!(2 * size_of::<Option<Probability<'s, G>>>());
        slot!(2 * size_of::<Result<Probability<'s, G>, Error>>());
        slot!(2 * size_of::<RowView<'s, G>>());
        slot!(
            2 * size_of::<Result<Option<&super::compact_attention_softmax::OwnedRow<'s, G>>, Error>>()
        );
        slot!(6 * size_of::<View>());
        slot!(2 * size_of::<Result<View, Error>>());
        slot!(2 * size_of::<Word>());
        slot!(2 * size_of::<PublishedView>());
        slot!(2 * size_of::<ScoreGather>());
        slot!(2 * size_of::<(usize, &ScoreGather, Interval)>());
        slot!(2 * size_of::<ValueLayout>());
        slot!(3 * size_of::<WorkBudget>());
        slot!(3 * size_of::<MatrixShape>());
        slot!(2 * size_of::<InputBounds>());
        slot!(2 * size_of::<Result<Interval, Error>>());
        slot!(attempt.checked_mul(3).ok_or(Error::Bounds)?);
        // baseline() already accounts caller publication controls once.
        Ok(peak)
    }
    pub(super) fn lower_attention_values(
        &self,
        op: &pllm_models::ModelOperation,
        shape: &[usize],
        probabilities: &PublishedView,
        values: &PublishedView,
    ) -> Result<Values<'s, G>, Error> {
        let context = self.request_admission.ok_or(Error::Binding)?;
        let mut attempt = context.attempt();
        context.check_open()?;
        context.check_final_work(self.work)?;
        self.validate_view(probabilities)?;
        self.validate_view(values)?;
        let original = self
            .frontier
            .operations()
            .nth(self.cursor)
            .ok_or(Error::Binding)?
            .1;
        let (p, v) = self.frontier.inputs(original);
        if op.operator != Op::AttentionValues
            || original.operator != op.operator
            || original.attributes != op.attributes
            || !same_input(probabilities, p.ok_or(Error::Binding)?)
            || !same_input(values, v.ok_or(Error::Binding)?)
        {
            return Err(Error::Binding);
        }
        let group = usize::try_from(
            op.attributes
                .get("group_size")
                .and_then(|n| n.as_u64())
                .ok_or(Error::Binding)?,
        )
        .map_err(|_| Error::Bounds)?;
        let (rows, tiles, width, sequence, keys) = geometry(probabilities, values, shape, group)?;
        let peak = self.values_peak(rows, tiles, size_of_val(&attempt))?;
        self.admit_total(peak)?;
        context.admit_phase(peak, 0)?;
        let depth = add_bytes(self.ops.len(), 1)?;
        let visits = rows
            .checked_mul(keys)
            .and_then(|n| n.checked_mul(width))
            .and_then(|n| n.checked_mul(depth))
            .and_then(|n| n.checked_mul(depth))
            .and_then(|n| n.checked_add(rows.checked_mul(rows)?))
            .and_then(|n| n.checked_mul(32))
            .ok_or(Error::Bounds)?;
        context.prepay_metadata(u64::try_from(visits).map_err(|_| Error::Bounds)?)?;
        let mut records = Vec::new();
        records
            .try_reserve_exact(rows)
            .map_err(|_| Error::Allocation)?;
        let mut products = Vec::new();
        products
            .try_reserve_exact(tiles)
            .map_err(|_| Error::Allocation)?;
        if records.capacity() != rows || products.capacity() != tiles {
            return Err(Error::Allocation);
        }
        let result = Values {
            probabilities: probabilities.clone(),
            values: values.clone(),
            rows: records,
            tiles: products,
            width,
            keys,
            sequence,
            nodes: 0,
            work: WorkBudget::default(),
            bounds: Interval::new(0, 0)?,
        };
        let mut sink = SinkFor {
            graph: self,
            result,
            pending: None,
            current: 0,
        };
        let p = SinkFor::<G>::view(
            Span {
                kind: 0,
                start: 0,
                len: probabilities.lanes,
                width: keys,
                inner: 0,
                valid: true,
            },
            &probabilities.logical.dimensions[..4],
        );
        let v = SinkFor::<G>::view(
            Span {
                kind: 1,
                start: 0,
                len: values.lanes,
                width,
                inner: 0,
                valid: true,
            },
            &values.logical.dimensions[..4],
        );
        let output = attention::values(
            &mut sink,
            op,
            shape,
            Some(&p),
            Some(&v),
            sequence,
            keys - sequence,
            keys,
        )?;
        if !output.words.valid
            || output.words.len != rows.checked_mul(width).ok_or(Error::Bounds)?
            || sink.result.rows.len() != rows
            || sink.result.tiles.len() != tiles
        {
            return Err(Error::Binding);
        }
        let mut bounds = None::<Interval>;
        for tile in &sink.result.tiles {
            bounds = Some(match bounds {
                None => tile.bounds,
                Some(a) => Interval::new(
                    a.lower().min(tile.bounds.lower()),
                    a.upper().max(tile.bounds.upper()),
                )?,
            });
        }
        sink.result.bounds = bounds.ok_or(Error::Binding)?;
        sink.result.layout()?;
        self.admit_pending_work(sink.result.work)?;
        for tile in &sink.result.tiles {
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
#[path = "compact_attention_values_tests.rs"]
mod tests;
