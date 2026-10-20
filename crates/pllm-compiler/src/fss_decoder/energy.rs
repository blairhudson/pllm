//! Decoder-only complete-row premises. Shared integer arithmetic establishes
//! bounds; this ledger binds their provenance to source, semantic plan, node and
//! physical last-axis rows. No public caller can author or import a certificate.
use super::*;
use pllm_core::integer_bounds as integer;
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug)]
struct Entry {
    code: program::Code,
    shape: [usize; 4],
    rank: usize,
    scalar: Range,
    squared: u128,
    dot: Option<Dot>,
    affine: Option<AffineLinear>,
    seal: [u8; 32],
}

/// Energy premises for the TWO actual inner-axis vectors, including a gathered
/// column when required. Raw bounds are recomputed by the shared kernel too.
#[derive(Clone, Copy, Debug)]
pub(super) struct Dot {
    pub inner: usize,
    pub left: u128,
    pub right: u128,
    pub raw: Range,
}

/// A latent pre-affine normalized row, tied to one immediately preceding norm
/// and its exact immutable affine/weight tensors. Only a public proof is folded.
#[derive(Clone, Copy, Debug)]
pub(super) struct AffineLinear {
    norm: usize,
    gamma: usize,
    beta: Option<usize>,
    normalized: u128,
    weighted: source::WeightedColumns,
    original: integer::MatrixCertificate,
}
impl AffineLinear {
    pub(super) fn derive(
        source: &Source,
        norm: usize,
        node: &program::Node,
        weight: usize,
        scratch_limit: usize,
    ) -> Result<Option<Self>, Error> {
        let program::Code::Norm { gamma, beta, .. } = node.code else {
            return Ok(None);
        };
        let width = *node.shape.last().ok_or(Error::Context)?;
        let weight = source.tensors.get(weight).ok_or(Error::Context)?;
        if weight.shape().get(1) != Some(&width) {
            return Err(Error::Context);
        }
        let Some(original) = weight.matrix_certificate()? else {
            return Ok(None);
        };
        let Some(weighted) = weight.weighted_columns(
            &source.tensors[gamma],
            beta.map(|i| &source.tensors[i]),
            scratch_limit,
        )?
        else {
            return Ok(None);
        };
        let normalized =
            pllm_garble::fusefss_backend::composition::norm::bounds::source_bounds(width)?
                .normalized_row_energy();
        // If shared integer arithmetic cannot represent this optional proof,
        // retain the independently admitted ordinary interval/energy proof.
        if integer::dot_radius(normalized, weighted.matrix.maximum_row_squared_energy()).is_err() {
            return Ok(None);
        }
        Ok(Some(Self {
            norm,
            gamma,
            beta,
            normalized,
            weighted,
            original,
        }))
    }
    fn seal(self, hash: &mut Sha256) {
        for v in [
            self.norm as u64,
            self.gamma as u64,
            self.beta.map_or(u64::MAX, |n| n as u64),
        ] {
            hash.update(v.to_le_bytes());
        }
        hash.update(self.normalized.to_le_bytes());
        for matrix in [self.weighted.matrix, self.original] {
            hash.update((matrix.rows() as u64).to_le_bytes());
            hash.update((matrix.columns() as u64).to_le_bytes());
            for v in [
                matrix.maximum_row_l1(),
                matrix.maximum_column_l1(),
                matrix.frobenius_squared(),
                matrix.maximum_row_squared_energy(),
            ] {
                hash.update(v.to_le_bytes());
            }
        }
        hash.update(self.weighted.beta_min.to_le_bytes());
        hash.update(self.weighted.beta_max.to_le_bytes());
        hash.update(self.weighted.beta_prefix_abs.to_le_bytes());
        hash.update(self.weighted.beta_squared.to_le_bytes());
    }
    pub(super) fn matches(self, norm: usize, node: &program::Node) -> bool {
        matches!(node.code, program::Code::Norm { gamma, beta, .. }
            if norm == self.norm && gamma == self.gamma && beta == self.beta
                && node.shape.last() == Some(&self.original.columns()))
    }
    /// x_j=(q_j*gamma_j-r_j)/Q+beta_j, with 0<=r_j<=Q-1 for
    /// BOTH signs. Q*(W*x)=Wdiag(gamma)*q-W*r+Q*Wbeta.
    fn raw_bounds(
        self,
        row: pllm_core::fixed_file::RowStatistics,
        prefix: bool,
    ) -> Result<(i128, i128), Error> {
        let radius = i128::try_from(
            integer::dot_radius(
                self.normalized,
                self.weighted.matrix.maximum_row_squared_energy(),
            )
            .map_err(|_| Error::Domain)?,
        )
        .map_err(|_| Error::Domain)?;
        let positive = row.positive_sum.checked_mul(Q - 1).ok_or(Error::Domain)?;
        let negative = row.negative_sum.checked_mul(Q - 1).ok_or(Error::Domain)?;
        let (beta_lo, beta_hi) = if prefix {
            let radius =
                i128::try_from(self.weighted.beta_prefix_abs).map_err(|_| Error::Domain)?;
            (-radius, radius)
        } else {
            (self.weighted.beta_min, self.weighted.beta_max)
        };
        // ceil((-radius-positive)/Q) is -floor((radius+positive)/Q).
        let lo = -radius
            .checked_add(positive)
            .ok_or(Error::Domain)?
            .div_euclid(Q);
        let hi = radius
            .checked_sub(negative)
            .ok_or(Error::Domain)?
            .div_euclid(Q);
        Ok((
            lo.checked_add(beta_lo).ok_or(Error::Domain)?,
            hi.checked_add(beta_hi).ok_or(Error::Domain)?,
        ))
    }
    pub(super) fn output_energy(self, bias: Option<&Tensor>) -> Result<Option<u128>, Error> {
        let matrix = self.weighted.matrix;
        let result = (|| {
            let main = self
                .normalized
                .checked_mul(matrix.operator_squared().ok()?)?;
            let error = (matrix.columns() as u128)
                .checked_mul((Q as u128 - 1).pow(2))?
                .checked_mul(self.original.operator_squared().ok()?)?;
            let main = integer::rescaled_energy(main, matrix.rows(), 12, 0).ok()?;
            let error = integer::rescaled_energy(error, matrix.rows(), 12, 0).ok()?;
            let raw = integer::sum_energy(main, error).ok()?;
            let raw = integer::sum_energy(raw, self.weighted.beta_squared).ok()?;
            integer::rescaled_energy(raw, matrix.rows(), 12, 1).ok()
        })();
        let Some(mut result) = result else {
            return Ok(None);
        };
        if let Some(bias) = bias {
            if bias.shape() != [matrix.rows()] {
                return Err(Error::Context);
            }
            result = integer::sum_energy(
                result,
                integer::row_energy(bias.resident_values().ok_or(Error::Context)?)
                    .map_err(|_| Error::Domain)?,
            )
            .map_err(|_| Error::Domain)?;
        }
        Ok(Some(result))
    }
}

pub(super) struct Ledger {
    source: [u8; 32],
    context: [u8; 32],
    entries: Vec<Entry>,
}

pub(super) fn reservation_bound(nodes: usize) -> Result<usize, Error> {
    nodes
        .checked_mul(std::mem::size_of::<Entry>())
        .and_then(|n| n.checked_add(std::mem::size_of::<Ledger>()))
        .ok_or(Error::Resource)
}

/// D14's frozen public predicate. The +1 denominator and strict signed50 upper
/// boundary require sum(x^2) <= 2^49-2. No checkpoint/private calibration.
pub(super) fn rms_guard(width: usize) -> Result<Range, Error> {
    if !(1..=4096).contains(&width) {
        return Err(Error::Domain);
    }
    let b = ((HALF as u128 - 2) / width as u128).isqrt() as i128;
    Range::new(-b, b)
}

impl Ledger {
    pub(super) fn new(source: [u8; 32], semantic: &str, nodes: usize) -> Result<Self, Error> {
        if source == [0; 32] || semantic.is_empty() || nodes == 0 || nodes > MAX_NODES {
            return Err(Error::Context);
        }
        let mut hash = Sha256::new();
        hash.update(b"pllm.fss.complete-row-provenance.ring50-q12.v1\0");
        hash.update(source);
        hash.update(semantic.as_bytes());
        Ok(Self {
            source,
            context: hash.finalize().into(),
            entries: vector(nodes)?,
        })
    }
    fn seal(&self, index: usize, e: &Entry) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(self.context);
        hash.update((index as u64).to_le_bytes());
        hash.update((e.rank as u64).to_le_bytes());
        use program::Code::*;
        let code: [u64; 5] = match e.code {
            Embedding { weight, positions } => [0, weight as u64, positions as u64, 0, 0],
            Linear {
                input,
                weight,
                bias,
                input_output,
            } => [
                1,
                input as u64,
                weight as u64,
                bias.map_or(u64::MAX, |i| i as u64),
                input_output as u64,
            ],
            Norm {
                input,
                gamma,
                beta,
                layer,
            } => [
                2,
                input as u64,
                gamma as u64,
                beta.map_or(u64::MAX, |i| i as u64),
                layer as u64,
            ],
            Add(a, b) => [3, a as u64, b as u64, 0, 0],
            Product(a, b) => [4, a as u64, b as u64, 0, 0],
            Activation { input, silu } => [5, input as u64, silu as u64, 0, 0],
            Slice { input, start } => [6, input as u64, start as u64, 0, 0],
            Heads { input, split } => [7, input as u64, split as u64, 0, 0],
            Rotary { input, descriptor } => [8, input as u64, descriptor as u64, 0, 0],
            Identity(i) => [9, i as u64, 0, 0, 0],
            Scores { left, right, group } => [10, left as u64, right as u64, group as u64, 0],
            Scale { input, coefficient } => [11, input as u64, coefficient as u64, 0, 0],
            Mask(i) => [12, i as u64, 0, 0, 0],
            Softmax(i) => [13, i as u64, 0, 0, 0],
            Values { left, right, group } => [14, left as u64, right as u64, group as u64, 0],
            Last(i) => [15, i as u64, 0, 0, 0],
            Argmax(i) => [16, i as u64, 0, 0, 0],
        };
        for word in code {
            hash.update(word.to_le_bytes());
        }
        for &d in &e.shape {
            hash.update((d as u64).to_le_bytes());
        }
        hash.update(e.scalar.lo.to_le_bytes());
        hash.update(e.scalar.hi.to_le_bytes());
        hash.update(e.squared.to_le_bytes());
        hash.update([u8::from(e.dot.is_some())]);
        if let Some(dot) = e.dot {
            hash.update((dot.inner as u64).to_le_bytes());
            hash.update(dot.left.to_le_bytes());
            hash.update(dot.right.to_le_bytes());
            hash.update(dot.raw.lo.to_le_bytes());
            hash.update(dot.raw.hi.to_le_bytes());
        }
        hash.update([u8::from(e.affine.is_some())]);
        if let Some(affine) = e.affine {
            affine.seal(&mut hash);
        }
        hash.finalize().into()
    }
    fn checked(
        &self,
        source: [u8; 32],
        index: usize,
        node: &program::Node,
    ) -> Result<&Entry, Error> {
        let e = self.entries.get(index).ok_or(Error::Context)?;
        if e.rank == 0
            || e.rank > 4
            || self.source != source
            || e.rank != node.shape.len()
            || e.shape[..e.rank] != node.shape
            || e.scalar != node.range
            || e.code != node.code
            || self.seal(index, e) != e.seal
        {
            return Err(Error::Context);
        }
        Ok(e)
    }
    pub(super) fn row(
        &self,
        source: [u8; 32],
        index: usize,
        node: &program::Node,
    ) -> Result<u128, Error> {
        Ok(self.checked(source, index, node)?.squared)
    }
    pub(super) fn dot(
        &self,
        source: [u8; 32],
        index: usize,
        node: &program::Node,
    ) -> Result<Option<Dot>, Error> {
        Ok(self.checked(source, index, node)?.dot)
    }
    pub(super) fn affine(
        &self,
        source: [u8; 32],
        index: usize,
        node: &program::Node,
    ) -> Result<Option<AffineLinear>, Error> {
        Ok(self.checked(source, index, node)?.affine)
    }
    pub(super) fn record(
        &mut self,
        code: program::Code,
        shape: &[usize],
        scalar: Range,
        squared: Option<u128>,
        dot: Option<Dot>,
        affine: Option<AffineLinear>,
    ) -> Result<Range, Error> {
        if shape.is_empty()
            || shape.len() > 4
            || shape.contains(&0)
            || self.entries.len() == self.entries.capacity()
        {
            return Err(Error::Context);
        }
        let width = *shape.last().ok_or(Error::Domain)?;
        let scalar_squared = scalar_energy(scalar, width)?;
        let squared = squared.map_or(scalar_squared, |e| e.min(scalar_squared));
        if minimum_energy(scalar, width)? > squared {
            return Err(Error::Domain);
        }
        let radius = squared.isqrt() as i128;
        let scalar = Range::new(
            i128::from(scalar.lo).max(-radius),
            i128::from(scalar.hi).min(radius),
        )?;
        let mut dims = [0; 4];
        dims[..shape.len()].copy_from_slice(shape);
        let mut e = Entry {
            code,
            shape: dims,
            rank: shape.len(),
            scalar,
            squared,
            dot,
            affine,
            seal: [0; 32],
        };
        e.seal = self.seal(self.entries.len(), &e);
        self.entries.push(e);
        Ok(scalar)
    }
    pub(super) fn digest(&self) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(self.context);
        for e in &self.entries {
            hash.update(e.seal);
        }
        hash.finalize().into()
    }
    pub(super) fn resident_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.entries.capacity() * std::mem::size_of::<Entry>()
    }
}

pub(super) fn scalar_energy(r: Range, width: usize) -> Result<u128, Error> {
    if width == 0 {
        return Err(Error::Domain);
    }
    let radius = u128::from(r.lo.unsigned_abs().max(r.hi.unsigned_abs()));
    radius
        .checked_mul(radius)
        .and_then(|n| n.checked_mul(width as u128))
        .ok_or(Error::Domain)
}
fn minimum_energy(r: Range, width: usize) -> Result<u128, Error> {
    let radius = if r.lo > 0 {
        r.lo as u128
    } else if r.hi < 0 {
        r.hi.unsigned_abs() as u128
    } else {
        0
    };
    radius
        .checked_mul(radius)
        .and_then(|n| n.checked_mul(width as u128))
        .ok_or(Error::Domain)
}
fn intersect(lo: i128, hi: i128, radius: Option<u128>) -> Result<Range, Error> {
    let (lo, hi) = radius.map_or((lo, hi), |r| (lo.max(-(r as i128)), hi.min(r as i128)));
    Range::new(lo, hi)
}

pub(super) fn dot_bounds(
    a: Range,
    b: Range,
    inner: usize,
    energies: Option<(u128, u128)>,
) -> Result<(Range, Option<Dot>), Error> {
    if inner == 0 {
        return Err(Error::Domain);
    }
    if let Some((left, right)) = energies {
        if minimum_energy(a, inner)? > left || minimum_energy(b, inner)? > right {
            return Err(Error::Domain);
        }
    }
    // Overflow means this optional proof is unavailable, not permission to relax
    // the independently checked scalar proof.
    let proof = energies.and_then(|(left, right)| {
        integer::dot_radius(left, right)
            .ok()
            .map(|r| (left, right, r))
    });
    let radius = proof.map(|(_, _, r)| r);
    let products = [
        a.lo as i128 * b.lo as i128,
        a.lo as i128 * b.hi as i128,
        a.hi as i128 * b.lo as i128,
        a.hi as i128 * b.hi as i128,
    ];
    let lo = *products.iter().min().ok_or(Error::Domain)?;
    let hi = *products.iter().max().ok_or(Error::Domain)?;
    intersect(lo, hi, radius)?; // Every individual multiplication must fit too.
    let lo = lo.checked_mul(inner as i128).ok_or(Error::Domain)?;
    let hi = hi.checked_mul(inner as i128).ok_or(Error::Domain)?;
    intersect(lo.min(0), hi.max(0), radius)?; // Every complete-dot prefix.
    let raw = intersect(lo, hi, radius)?;
    Ok((
        raw,
        proof.map(|(left, right, _)| Dot {
            inner,
            left,
            right,
            raw,
        }),
    ))
}

pub(super) fn public_raw_with_affine(
    input: Range,
    row: pllm_core::fixed_file::RowStatistics,
    energies: Option<(u128, u128)>,
    affine: Option<AffineLinear>,
) -> Result<Range, Error> {
    if row.positive_sum < 0 || row.negative_sum > 0 {
        return Err(Error::Domain);
    }
    let radius = energies.and_then(|(a, b)| integer::dot_radius(a, b).ok());
    let sum = |a: i128, b: i128| {
        a.checked_mul(row.positive_sum)
            .and_then(|x| x.checked_add(b.checked_mul(row.negative_sum)?))
            .ok_or(Error::Domain)
    };
    let (lo, hi) = (input.lo as i128, input.hi as i128);
    let (mut pl, mut ph) = (sum(lo.min(0), hi.max(0))?, sum(hi.max(0), lo.min(0))?);
    let (mut low, mut high) = (sum(lo, hi)?, sum(hi, lo)?);
    if let Some(affine) = affine {
        let (a, b) = affine.raw_bounds(row, true)?;
        pl = pl.max(a);
        ph = ph.min(b);
        let (a, b) = affine.raw_bounds(row, false)?;
        low = low.max(a);
        high = high.min(b);
    }
    intersect(pl, ph, radius)?;
    intersect(low, high, radius)
}

/// Full output-row matrix bound. Transposition swaps induced norms; Frobenius is
/// unchanged. The certificate is read only from this exact immutable tensor.
pub(super) fn linear(
    input: u128,
    weight: &Tensor,
    input_output: bool,
    bias: Option<&Tensor>,
) -> Result<Option<u128>, Error> {
    let Some(mut matrix) = weight.matrix_certificate()? else {
        return Ok(None);
    };
    if weight.shape() != [matrix.rows(), matrix.columns()] {
        return Err(Error::Context);
    }
    if input_output {
        matrix = integer::MatrixCertificate::from_bounds(
            matrix.columns(),
            matrix.rows(),
            matrix.maximum_column_l1(),
            matrix.maximum_row_l1(),
            matrix.frobenius_squared(),
        )
        .map_err(|_| Error::Domain)?;
    }
    let Some(mut squared) = integer::linear_energy(input, matrix, 12, 1).ok() else {
        return Ok(None);
    };
    if let Some(bias) = bias {
        if bias.shape() != [matrix.rows()] {
            return Err(Error::Context);
        }
        let bias = integer::row_energy(bias.resident_values().ok_or(Error::Context)?)
            .map_err(|_| Error::Domain)?;
        squared = integer::sum_energy(squared, bias).map_err(|_| Error::Domain)?;
    }
    Ok(Some(squared))
}

pub(super) fn affine(
    normalized: u128,
    gamma: &Tensor,
    beta: Option<&Tensor>,
) -> Result<Option<u128>, Error> {
    let values = gamma.resident_values().ok_or(Error::Context)?;
    let width = values.len();
    let maximum = values
        .iter()
        .map(|v| u128::from(v.unsigned_abs()))
        .max()
        .ok_or(Error::Domain)?;
    let Some(raw) = maximum
        .checked_mul(maximum)
        .and_then(|n| n.checked_mul(normalized))
    else {
        return Ok(None);
    };
    let Some(mut squared) = integer::rescaled_energy(raw, width, 12, 1).ok() else {
        return Ok(None);
    };
    if let Some(beta) = beta {
        if beta.shape() != [width] {
            return Err(Error::Context);
        }
        squared = integer::sum_energy(
            squared,
            integer::row_energy(beta.resident_values().ok_or(Error::Context)?)
                .map_err(|_| Error::Domain)?,
        )
        .map_err(|_| Error::Domain)?;
    }
    Ok(Some(squared))
}

/// A normalization consequence cannot enter the ledger until the actual native
/// input/mean/square/+1/product/affine obligations have been admitted. Compile
/// and immediately retire the public Plan; no private material or values exist.
pub(super) fn admit_norm(
    source: [u8; 32],
    input: Range,
    input_energy: u128,
    gamma: &Tensor,
    beta: Option<&Tensor>,
    layer: bool,
    tables: &pllm_garble::fusefss_backend::composition::cache::SourceTableCache<'_>,
) -> Result<pllm_garble::fusefss_backend::composition::norm::bounds::SourceBounds, Error> {
    use pllm_garble::fusefss_backend::composition::norm;
    let gamma = gamma.resident_values().ok_or(Error::Context)?;
    if gamma.is_empty() || gamma.len() > 4096 {
        return Err(Error::Domain);
    }
    // Native norm admission also charges borrowed affine payloads to its Budget.
    let zero = [0i64; 4096];
    let beta = match beta {
        Some(beta) => beta.resident_values().ok_or(Error::Context)?,
        None => &zero[..gamma.len()],
    };
    let spec = norm::Spec {
        kind: if layer {
            norm::Kind::Layer
        } else {
            norm::Kind::Rms
        },
        input_min: input.lo,
        input_max: input.hi,
        gamma,
        beta,
    };
    let plan = if layer {
        norm::compile_cached(source, &spec, tables)?
    } else {
        norm::compile_cached_with_row_energy(source, &spec, input_energy, tables)?
    };
    drop(plan);
    norm::bounds::source_bounds(gamma.len())
}

/// Pointwise multiplication uses matching coordinates, not every Cartesian pair:
/// sum(a_i^2 b_i^2) <= min(Ea*Eb, Ea*max(b_i^2), Eb*max(a_i^2)).
/// The existing scalar intervals remain independently binding. Only the one
/// original final floor is applied by the shared rescaling bound.
pub(super) fn product_energy(
    left: Range,
    left_energy: u128,
    right: Range,
    right_energy: u128,
    width: usize,
) -> Result<Option<u128>, Error> {
    if width == 0
        || minimum_energy(left, width)? > left_energy
        || minimum_energy(right, width)? > right_energy
    {
        return Err(Error::Domain);
    }
    let raw = [
        left_energy.checked_mul(right_energy),
        left_energy.checked_mul(scalar_energy(right, 1)?),
        right_energy.checked_mul(scalar_energy(left, 1)?),
    ]
    .into_iter()
    .flatten()
    .min();
    Ok(raw.and_then(|raw| integer::rescaled_energy(raw, width, 12, 1).ok()))
}

pub(super) fn rotary(
    input: u128,
    descriptor: &RotaryCoefficients,
    sequence: usize,
) -> Result<Option<u128>, Error> {
    let width = descriptor.dimensions;
    if width == 0 || width % 2 != 0 || sequence > descriptor.positions {
        return Err(Error::Context);
    }
    let mut maximum = 0;
    for row in 0..sequence {
        let start = row * width;
        let Some(bound) = integer::rotary_energy(
            input,
            &descriptor.cos[start..start + width / 2],
            &descriptor.sin[start..start + width / 2],
            12,
            2,
        )
        .ok() else {
            return Ok(None);
        };
        maximum = maximum.max(bound);
    }
    Ok(Some(maximum))
}

/// A column gathers one coordinate from EACH distinct physical value row.
/// Its energy may be S times the per-row energy; treating it as one row is wrong.
pub(super) fn gathered_column(
    row_energy: u128,
    scalar: Range,
    positions: usize,
) -> Result<u128, Error> {
    if positions == 0 {
        return Err(Error::Domain);
    }
    row_energy
        .min(scalar_energy(scalar, 1)?)
        .checked_mul(positions as u128)
        .ok_or(Error::Domain)
}

pub(super) fn values_output(
    probabilities: u128,
    value_row: u128,
    positions: usize,
    width: usize,
) -> Option<u128> {
    // ||V^T p||^2 <= ||V||_F^2 ||p||^2 <= S*E_v*E_p.
    let raw = value_row
        .checked_mul(positions as u128)?
        .checked_mul(probabilities)?;
    integer::rescaled_energy(raw, width, 12, 1).ok()
}

#[cfg(test)]
#[path = "energy_tests.rs"]
mod tests;
