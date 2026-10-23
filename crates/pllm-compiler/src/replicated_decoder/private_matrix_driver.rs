//! Storage-neutral ORIGINAL small-inner private matrix traversal.
//!
//! This is not an input/source/proof issuer. Original Builder retains its native
//! owners; compact sinks retain their actual sealed inputs and quote their own
//! bounded metadata before calling. Public and large-inner dispatch stay with
//! matrix_with_attention. No new numeric profile or runtime is selected here.
use super::*;
use puma::multiply::norm::RowEnergyProof;

pub(super) trait Sink {
    type Owner: Copy;
    type Mass: Copy;
    type Output;
    fn begin_row(&mut self, _: usize) -> Result<(), Error> {
        Ok(())
    }
    fn empty(&self) -> Self::Output;
    fn mass(&self, row: usize) -> Result<Option<Self::Mass>, Error>;
    fn left(&mut self, row: usize) -> Result<Self::Owner, Error>;
    fn right(&mut self, start: usize, count: usize) -> Result<Self::Owner, Error>;
    fn proof(
        &self,
        row: usize,
        start: usize,
        count: usize,
        left: Self::Owner,
        right: Self::Owner,
    ) -> Result<Option<RowEnergyProof>, Error>;
    fn product(
        &mut self,
        left: Self::Owner,
        right: Self::Owner,
        shape: MatrixShape,
        rescale: bool,
        proof: Option<RowEnergyProof>,
    ) -> Result<Self::Owner, Error>;
    fn refine_mass(
        &mut self,
        output: Self::Owner,
        right: Self::Owner,
        mass: Self::Mass,
        rescale: bool,
    ) -> Result<(), Error>;
    fn append(&mut self, output: &mut Self::Output, owner: Self::Owner, lanes: usize);
}

pub(super) fn lower<S: Sink>(
    sink: &mut S,
    rows: usize,
    inner: usize,
    columns: usize,
    rescale: bool,
) -> Result<S::Output, Error> {
    if inner == 0 || inner > puma::MAX_LANES {
        return Err(Error::Shape);
    }
    let tile = (puma::MAX_LANES / inner).min(128);
    let mut words = sink.empty();
    for row in 0..rows {
        sink.begin_row(row)?;
        let mass = sink.mass(row)?;
        let left = sink.left(row)?;
        for start in (0..columns).step_by(tile) {
            let count = tile.min(columns - start);
            let right = sink.right(start, count)?;
            let proof = sink.proof(row, start, count, left, right)?;
            let output = sink.product(
                left,
                right,
                MatrixShape::new(1, inner, count)?,
                rescale,
                proof,
            )?;
            if let Some(mass) = mass {
                sink.refine_mass(output, right, mass, rescale)?;
            }
            sink.append(&mut words, output, count);
        }
    }
    Ok(words)
}

struct Expanded<'b, 'v, 'w> {
    builder: &'b mut Builder,
    input: &'v View,
    weight: &'v MatrixView<'w>,
    inner: usize,
    attention: bool,
    row: Words,
}

impl Sink for Expanded<'_, '_, '_> {
    type Owner = usize;
    type Mass = u64;
    type Output = Words;
    fn empty(&self) -> Words {
        Words::new()
    }
    fn begin_row(&mut self, row: usize) -> Result<(), Error> {
        let start = row * self.inner;
        self.row = Words::new();
        self.row = self.input.words.slice(start..start + self.inner);
        Ok(())
    }
    fn mass(&self, _: usize) -> Result<Option<u64>, Error> {
        Ok(self
            .builder
            .row_certificate(&self.row, self.inner, &self.builder.row_masses))
    }
    fn left(&mut self, _: usize) -> Result<usize, Error> {
        self.builder.gather(&self.row)
    }
    fn right(&mut self, start: usize, count: usize) -> Result<usize, Error> {
        self.weight
            .gather(self.builder, start, count, 0, self.inner)
    }
    fn proof(
        &self,
        _: usize,
        start: usize,
        count: usize,
        left: usize,
        right: usize,
    ) -> Result<Option<RowEnergyProof>, Error> {
        if !self.attention || self.builder.attention_energies.is_empty() {
            return Ok(None);
        }
        self.builder.attention_dot(
            &self.row,
            self.weight,
            start,
            count,
            self.inner,
            left,
            right,
        )
    }
    fn product(
        &mut self,
        left: usize,
        right: usize,
        shape: MatrixShape,
        rescale: bool,
        proof: Option<RowEnergyProof>,
    ) -> Result<usize, Error> {
        self.builder
            .product_with_norm(left, right, Some(shape), rescale, proof)
    }
    fn refine_mass(
        &mut self,
        output: usize,
        right: usize,
        mass: u64,
        rescale: bool,
    ) -> Result<(), Error> {
        let values = self.builder.nodes[right].bounds;
        let ordinary = self.builder.nodes[output].bounds;
        self.builder.nodes[output].bounds = with_mass(ordinary, values, mass, rescale)?;
        Ok(())
    }
    fn append(&mut self, output: &mut Words, owner: usize, lanes: usize) {
        output.extend((0..lanes).map(|index| (owner, index)));
    }
}

pub(super) fn expanded(
    builder: &mut Builder,
    input: &View,
    weight: &MatrixView<'_>,
    [rows, inner, columns]: [usize; 3],
    rescale: bool,
    attention: bool,
) -> Result<View, Error> {
    let words = lower(
        &mut Expanded {
            builder,
            input,
            weight,
            inner,
            attention,
            row: Words::new(),
        },
        rows,
        inner,
        columns,
        rescale,
    )?;
    Ok(View {
        words,
        shape: vec![1, rows, columns],
    })
}

/// The unchanged ORIGINAL softmax-mass intersection, after ordinary Product
/// admission. A caller numeric value is not an owner token for compact sinks.
pub(super) fn with_mass(
    ordinary: Interval,
    values: Interval,
    mass: u64,
    rescale: bool,
) -> Result<Interval, Error> {
    let bound =
        |value: i64| i64::try_from(i128::from(value) * i128::from(mass)).map_err(|_| Error::Bounds);
    let mut range = Interval::new(bound(values.lower().min(0))?, bound(values.upper().max(0))?)?;
    if rescale {
        range = Interval::new(
            range.lower().div_euclid(ONE),
            -(-range.upper()).div_euclid(ONE),
        )?;
    }
    Interval::new(
        ordinary.lower().max(range.lower()),
        ordinary.upper().min(range.upper()),
    )
}

#[cfg(test)]
#[path = "private_matrix_driver_tests.rs"]
mod tests;
