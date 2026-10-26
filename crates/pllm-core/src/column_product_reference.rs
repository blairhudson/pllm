//! NEXUS Algorithm 5, one-head CLEAR reference for Q K^T.
//! Logical column vectors, NOT physical CKKS packing, HE authority, numeric
//! profile, source authentication, or full NEXUS adoption. Ascending j,
//! zero initialization, separate binary64 multiply/add. Key columns rotate LEFT
//! after EVERY diagonal, INCLUDING the final iteration.
//! q_j[t]=Q[t,j], r_i[t]=sum_j Q[t,j]*K[(t+i)%m,j],
//! C[t,(t+i)%m]=r_i[t]. Caller owns all disjoint storage before entry;
//! suffixes are rejected, not silently excluded from admission/cleanup.
use crate::row_fold::{State, WorkLedger};
pub use crate::slot_matrix_layout::{Error as LayoutError, Limits};
pub const MAX_BYTES: usize = 64 << 20;
pub const MAX_VISITS: usize = 1 << 28;
pub const CONTROL_BYTES: usize = 4096;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Layout(LayoutError),
    Work(crate::row_fold::Error),
    Poisoned,
}
impl From<LayoutError> for Error {
    fn from(e: LayoutError) -> Self {
        Self::Layout(e)
    }
}
impl From<crate::row_fold::Error> for Error {
    fn from(e: crate::row_fold::Error) -> Self {
        Self::Work(e)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    rows: usize,
    width: usize,
    cells: usize,
    scores: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Quote {
    pub input_cells: usize,
    pub workspace_cells: usize,
    pub output_cells: usize,
    pub bytes: usize,
    /// Conservative visits: validation, packing, initialization, arithmetic,
    /// all rotations, scatter, and full cleanup even on cancellation.
    pub visits: usize,
}
fn add(a: usize, b: usize) -> Result<usize, LayoutError> {
    a.checked_add(b).ok_or(LayoutError::Overflow)
}
fn mul(a: usize, b: usize) -> Result<usize, LayoutError> {
    a.checked_mul(b).ok_or(LayoutError::Overflow)
}
impl Geometry {
    pub fn new(rows: usize, width: usize) -> Result<Self, Error> {
        if rows == 0 || width == 0 {
            return Err(LayoutError::Geometry.into());
        }
        let g = Self {
            rows,
            width,
            cells: mul(rows, width)?,
            scores: mul(rows, rows)?,
        };
        g.quote(Limits {
            max_buffer_elements: MAX_BYTES / 8,
            max_work: MAX_VISITS,
        })?;
        Ok(g)
    }
    pub fn rows(self) -> usize {
        self.rows
    }
    pub fn width(self) -> usize {
        self.width
    }
    pub fn quote(self, limits: Limits) -> Result<Quote, Error> {
        let workspace_cells = add(mul(2, self.cells)?, self.rows)?;
        let total = add(add(mul(2, self.cells)?, workspace_cells)?, self.scores)?;
        let bytes = add(mul(total, 8)?, CONTROL_BYTES)?;
        if bytes > MAX_BYTES
            || [self.cells, workspace_cells, self.scores]
                .iter()
                .any(|&n| n > limits.max_buffer_elements)
        {
            return Err(LayoutError::Buffer.into());
        }
        let visits = add(
            add(mul(16, mul(self.rows, self.cells)?)?, mul(16, self.cells)?)?,
            add(mul(8, self.scores)?, 1024)?,
        )?;
        if visits > MAX_VISITS.min(limits.max_work) {
            return Err(LayoutError::WorkLimit.into());
        }
        Ok(Quote {
            input_cells: self.cells,
            workspace_cells,
            output_cells: self.scores,
            bytes,
            visits,
        })
    }
    pub fn scatter_address(self, diagonal: usize, row: usize) -> Result<usize, Error> {
        if diagonal >= self.rows || row >= self.rows {
            return Err(LayoutError::Geometry.into());
        }
        Ok(row * self.rows + (row + diagonal) % self.rows)
    }
}
/// Opaque, non-cloneable complete-work prepayment. No factory/callback runs here.
pub struct Prepayment {
    geometry: Geometry,
    quote: Quote,
}
pub fn prepay(
    g: Geometry,
    limits: Limits,
    ledger: &mut WorkLedger,
    control: usize,
) -> Result<Prepayment, Error> {
    let quote = g.quote(limits)?;
    ledger.debit(quote.visits, control)?;
    Ok(Prepayment { geometry: g, quote })
}
/// Original input borrows remain immutable through retained output lifetime.
///
/// ```compile_fail
/// use pllm_core::{column_product_reference::*, row_fold::{State, WorkLedger}};
/// let g=Geometry::new(1,1).unwrap();
/// let mut query=[1.0]; let key=[2.0]; let mut workspace=[0.0;3];
/// let mut scores=[0.0]; let mut state=State::Fresh;
/// let mut work=WorkLedger::new(MAX_VISITS,0);
/// let p=prepay(g,Limits{max_buffer_elements:100,max_work:MAX_VISITS},&mut work,0).unwrap();
/// let output=Reference::new_prepaid(p,&query,&key,&mut workspace,&mut scores,&mut state).unwrap().execute().unwrap();
/// query[0]=4.0;
/// assert_eq!(output.values()[0],2.0);
/// ```
pub struct Reference<'a> {
    geometry: Geometry,
    query: &'a [f64],
    key: &'a [f64],
    workspace: &'a mut [f64],
    output: &'a mut [f64],
    state: &'a mut State,
}
pub struct Output<'a> {
    reference: Reference<'a>,
}
impl Output<'_> {
    pub fn values(&self) -> &[f64] {
        self.reference.output
    }
    /// Key columns after m left rotations: restored to initial logical packing.
    pub fn final_key_columns(&self) -> &[f64] {
        let n = self.reference.geometry.cells;
        &self.reference.workspace[n..2 * n]
    }
}
impl<'a> Reference<'a> {
    pub fn new_prepaid(
        payment: Prepayment,
        query: &'a [f64],
        key: &'a [f64],
        workspace: &'a mut [f64],
        output: &'a mut [f64],
        state: &'a mut State,
    ) -> Result<Self, Error> {
        if *state != State::Fresh {
            return Err(Error::Poisoned);
        }
        let q = payment.quote;
        if query.len() != q.input_cells
            || key.len() != q.input_cells
            || workspace.len() != q.workspace_cells
            || output.len() != q.output_cells
        {
            return Err(LayoutError::Buffer.into());
        }
        *state = State::Poisoned;
        let run = Self {
            geometry: payment.geometry,
            query,
            key,
            workspace,
            output,
            state,
        };
        if run.query.iter().chain(run.key).any(|x| !x.is_finite()) {
            return Err(LayoutError::NonFinite.into());
        }
        run.workspace.fill(0.0);
        run.output.fill(0.0);
        Ok(run)
    }
    pub fn execute(self) -> Result<Output<'a>, Error> {
        let g = self.geometry;
        let (q, rest) = self.workspace.split_at_mut(g.cells);
        let (k, r) = rest.split_at_mut(g.cells);
        for j in 0..g.width {
            for t in 0..g.rows {
                q[j * g.rows + t] = self.query[t * g.width + j];
                k[j * g.rows + t] = self.key[t * g.width + j];
            }
        }
        for i in 0..g.rows {
            r.fill(0.0);
            for j in 0..g.width {
                for t in 0..g.rows {
                    let product = q[j * g.rows + t] * k[j * g.rows + t];
                    let sum = r[t] + product;
                    if !product.is_finite() || !sum.is_finite() {
                        return Err(LayoutError::NonFinite.into());
                    }
                    r[t] = sum;
                }
            }
            for (t, &value) in r.iter().enumerate() {
                self.output[g.scatter_address(i, t)?] = value;
            }
            for column in k.chunks_exact_mut(g.rows) {
                column.rotate_left(1);
            }
        }
        *self.state = State::Complete;
        Ok(Output { reference: self })
    }
}
impl Drop for Reference<'_> {
    fn drop(&mut self) {
        self.workspace.fill(0.0);
        self.output.fill(0.0);
    }
}
#[cfg(test)]
#[path = "column_product_reference_tests.rs"]
mod tests;
