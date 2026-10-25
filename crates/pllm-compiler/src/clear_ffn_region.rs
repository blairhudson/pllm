//! Explicit artifact-adapted CLEAR reference only. No default GeluTanh override,
//! evaluated-author profile, encrypted capability, bootstrap or residual step.
//! Original immutable Vec owners and their upstream same-counter guard are
//! retained; neither pointer/name checks nor that guard authenticate checkpoints.
//! Cached call source: crypto-starlab/THOR@b513c06aaabccc04b560721e35e9a98008428c25,
//! nonlinear/polynomial.py grouping; cached reconciliation A[gelu], not printed P.
//! Bounds below cover named logical owners/work, NOT allocator/callframes/RSS.
//! New float64, flat-attribute reference regions only; no automatic migration of
//! existing float32 regions and no evaluated-run identity is inferred.
use pllm_core::{polynomial_schedule as poly, row_fold::WorkLedger, slot_matrix_layout as matrix};
use pllm_garble::shared_allocation::{Budget, ExternalCharge};
use pllm_models::{owned_backing::checked_json_backing_bytes, ModelOperation, ModelOperator};
use std::{mem::size_of, ptr};
use zeroize::{Zeroize, Zeroizing};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Region,
    Shape,
    Policy,
    Source,
    Resource,
    Work,
    NonFinite,
    Burned,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Fresh,
    Spent,
}
#[derive(Clone, Copy)]
pub struct Limits {
    pub max_owner_bytes: usize,
    pub max_phase_bytes: usize,
    pub max_work: usize,
}
/// Bounded metadata preflight visits/bytes, not a CPU-instruction estimate.
/// Includes fixed schema checks, names <=256, keys <=64 and shared opaque-map
/// census (only already-validated depth-two public attributes are accepted).
pub const PREFLIGHT_CONTROL: usize = 3 * (6 * 256 + 7 * (3 * 64 + 2 * 256) + 128) + 2 * 256;
// Logical visits/arithmetic/initialization/cleanup, NOT cycles: geometry24;
// packs96; corrected matrix validation/products/scratch/output2*(152+16+8);
// unpacks48; bias16 arithmetic+checks; reversal32 + eight outer read/halves56;
// polynomial
// arithmetic+finite checks2*8*(74+65); polynomial inputs/coefficients8*62;
// four wrapper arithmetic+checks per scalar; fixed float owner init+cleanup362.
pub const CLEAR_WORK: usize =
    24 + 96 + 2 * (152 + 16 + 8) + 48 + 32 + 32 + 8 * 56 + 2 * 8 * (74 + 65) + 8 * 62 + 8 * 8 + 362;

pub struct Source<'a> {
    pub operation: &'a ModelOperation,
    pub weight_name: &'a String,
    pub bias_name: &'a String,
    pub weights: &'a Vec<f64>,
    pub bias: &'a Vec<f64>,
}
pub struct OriginalInputs<'a> {
    pub operations: &'a Vec<ModelOperation>,
    pub input_name: &'a String,
    pub input: &'a Vec<f64>,
    pub sources: [Source<'a>; 2],
    /// Upstream ORIGINAL owner allocation obligation, not slice-sized credit.
    pub original_guard: &'a ExternalCharge<'a>,
}
pub struct ClearReferencePolicy<'a> {
    original: OriginalInputs<'a>,
}
impl<'a> ClearReferencePolicy<'a> {
    /// Explicitly selects the complete cached A[gelu] call, NOT GeluTanh math.
    /// Precision is binary64/non-fused; ordinary float32 model ops are refused.
    pub fn artifact_call_gelu(original: OriginalInputs<'a>) -> Self {
        Self { original }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Quote {
    pub original_bytes: usize,
    pub phase_bytes: usize,
    pub clear_work: usize,
    pub schedules: [matrix::Schedule; 2],
}
struct Scratch {
    a: [f64; 8],
    w1: [f64; 16],
    w2: [f64; 16],
    mm: [f64; 16],
    first_packed: [f64; 8],
    first: [f64; 8],
    activation: [f64; 8],
    activation_packed: [f64; 8],
    final_packed: [f64; 8],
    output: [f64; 8],
    inner: [f64; 32],
    outer: [f64; 28],
    poly: poly::Workspace,
}
impl Scratch {
    fn empty() -> Self {
        Self {
            a: [0.; 8],
            w1: [0.; 16],
            w2: [0.; 16],
            mm: [0.; 16],
            first_packed: [0.; 8],
            first: [0.; 8],
            activation: [0.; 8],
            activation_packed: [0.; 8],
            final_packed: [0.; 8],
            output: [0.; 8],
            inner: [0.; 32],
            outer: [0.; 28],
            poly: poly::Workspace::empty(),
        }
    }
}
impl Zeroize for Scratch {
    fn zeroize(&mut self) {
        self.a.zeroize();
        self.w1.zeroize();
        self.w2.zeroize();
        self.mm.zeroize();
        self.first_packed.zeroize();
        self.first.zeroize();
        self.activation.zeroize();
        self.activation_packed.zeroize();
        self.final_packed.zeroize();
        self.output.zeroize();
        self.inner.zeroize();
        self.outer.zeroize();
        self.poly.zeroize();
    }
}
/// Original owners cannot be changed while a result borrows them.
/// ```compile_fail
/// use pllm_compiler::clear_ffn_region::Output;
/// fn mutate<'a>(v: &'a mut Vec<f64>, run: fn(&'a Vec<f64>) -> Output<'a>) {
///     let output = run(v);
///     v[0] = 7.0;
///     drop(output);
/// }
/// ```
pub struct Output<'a> {
    scratch: Zeroizing<Scratch>,
    _original: ClearReferencePolicy<'a>,
    // Payload drops before its charge; original guard still borrowed throughout.
    _charge: ExternalCharge<'a>,
}
impl Output<'_> {
    pub fn values(&self) -> &[f64; 8] {
        &self.scratch.output
    }
    pub fn first_affine(&self) -> &[f64; 8] {
        &self.scratch.first
    }
    pub fn activated(&self) -> &[f64; 8] {
        &self.scratch.activation
    }
}
fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Resource)
}
fn finite(x: f64) -> Result<f64, Error> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(Error::NonFinite)
    }
}
fn mat(e: matrix::Error) -> Error {
    if e == matrix::Error::NonFinite {
        Error::NonFinite
    } else {
        Error::Resource
    }
}
fn limits() -> matrix::Limits {
    matrix::Limits {
        max_buffer_elements: 16,
        max_work: 2048,
    }
}
fn layouts() -> Result<(matrix::Layout, matrix::Layout), Error> {
    Ok((
        matrix::Layout::new(2, 4, 2, 4, 1, 1, matrix::Diagonal::Lower, limits()).map_err(mat)?,
        matrix::Layout::new(4, 4, 4, 4, 1, 1, matrix::Diagonal::Lower, limits()).map_err(mat)?,
    ))
}
fn name(s: &str) -> Result<(), Error> {
    if s.is_empty() || s.len() > 256 {
        Err(Error::Policy)
    } else {
        Ok(())
    }
}

fn original_bytes(policy: &ClearReferencePolicy<'_>, cap: usize) -> Result<usize, Error> {
    let o = &policy.original;
    let ops = o.operations;
    if ops.len() != 3 {
        return Err(Error::Region);
    }
    if [ops[0].operator, ops[1].operator, ops[2].operator]
        != [
            ModelOperator::Linear,
            ModelOperator::GeluTanh,
            ModelOperator::Linear,
        ]
    {
        return Err(Error::Region);
    }
    name(o.input_name)?;
    if o.input.len() != 8 {
        return Err(Error::Shape);
    }
    let mut total = add(
        size_of::<Vec<ModelOperation>>(),
        ops.capacity()
            .checked_mul(size_of::<ModelOperation>())
            .ok_or(Error::Resource)?,
    )?;
    total = add(
        total,
        add(
            size_of::<Vec<f64>>(),
            o.input.capacity().checked_mul(8).ok_or(Error::Resource)?,
        )?,
    )?;
    total = add(total, add(size_of::<String>(), o.input_name.capacity())?)?;
    for (i, op) in ops.iter().enumerate() {
        name(&op.id)?;
        if op.state_kind.is_some() || op.output_shape != [2, 4] || op.inputs.len() != 1 {
            return Err(Error::Shape);
        }
        if op.layer != ops[0].layer || ops.iter().take(i).any(|other| other.id == op.id) {
            return Err(Error::Region);
        }
        name(&op.inputs[0])?;
        if op.inputs[0].as_str() != (if i == 0 { o.input_name } else { &ops[i - 1].id }).as_str() {
            return Err(Error::Source);
        }
        total = add(total, op.id.capacity())?;
        total = add(
            total,
            op.inputs
                .capacity()
                .checked_mul(size_of::<String>())
                .ok_or(Error::Resource)?,
        )?;
        total = add(total, op.inputs[0].capacity())?;
        total = add(
            total,
            op.output_shape
                .capacity()
                .checked_mul(size_of::<u64>())
                .ok_or(Error::Resource)?,
        )?;
        let attrs = op.attributes.as_object().ok_or(Error::Policy)?;
        if i == 1 {
            if attrs.len() != 1
                || attrs.get("compute_dtype").and_then(|v| v.as_str()) != Some("float64")
            {
                return Err(Error::Policy);
            }
        } else {
            let source = &o.sources[usize::from(i == 2)];
            if !ptr::eq(op, source.operation) {
                return Err(Error::Source);
            }
            name(source.weight_name)?;
            name(source.bias_name)?;
            if source.weights.len() != 16 || source.bias.len() != 4 {
                return Err(Error::Shape);
            }
            if attrs.len() != 7 {
                return Err(Error::Policy);
            }
            for key in attrs.keys() {
                if key.len() > 64 {
                    return Err(Error::Policy);
                }
            }
            if attrs.get("weight").and_then(|v| v.as_str()) != Some(source.weight_name.as_str())
                || attrs.get("bias").and_then(|v| v.as_str()) != Some(source.bias_name.as_str())
                || attrs.get("compute_dtype").and_then(|v| v.as_str()) != Some("float64")
                || attrs.get("weight_layout").and_then(|v| v.as_str()) != Some("input_output")
                || attrs.get("formula").and_then(|v| v.as_str())
                    != Some("x_matmul_weight_plus_bias")
            {
                return Err(Error::Policy);
            }
            for (key, expected) in [
                ("weight_shape", &[4u64, 4][..]),
                ("bias_shape", &[4u64][..]),
            ] {
                let arr = attrs
                    .get(key)
                    .and_then(|v| v.as_array())
                    .ok_or(Error::Policy)?;
                if arr.len() != expected.len()
                    || arr
                        .iter()
                        .zip(expected)
                        .any(|(a, b)| a.as_u64() != Some(*b))
                {
                    return Err(Error::Shape);
                }
            }
            for v in [source.weights, source.bias] {
                total = add(
                    total,
                    add(
                        size_of::<Vec<f64>>(),
                        v.capacity().checked_mul(8).ok_or(Error::Resource)?,
                    )?,
                )?;
            }
            for s in [source.weight_name, source.bias_name] {
                total = add(total, add(size_of::<String>(), s.capacity())?)?;
            }
        }
        // Shared conservative opaque BTree/empty-root policy, NOT exact node count.
        total = add(
            total,
            checked_json_backing_bytes(&op.attributes, cap).map_err(|_| Error::Resource)?,
        )?;
    }
    if total > cap {
        return Err(Error::Resource);
    }
    Ok(total)
}

/// Query consumes bounded shared control work. It does not allocate or execute
/// matrix/polynomial kernels. Original input metadata is borrowed, never parsed.
pub fn quote(
    policy: &ClearReferencePolicy<'_>,
    cap: Limits,
    work: &mut WorkLedger,
) -> Result<Quote, Error> {
    work.debit(0, PREFLIGHT_CONTROL).map_err(|_| Error::Work)?;
    let original_bytes = original_bytes(policy, cap.max_owner_bytes)?;
    if CLEAR_WORK > cap.max_work {
        return Err(Error::Work);
    }
    let (left, _) = layouts()?;
    let schedule = matrix::optimized_lower_schedule(left).map_err(mat)?;
    // Concrete retained payload + old/new scratch/result moves + original
    // borrowed controls, layout/schedule/Result and polynomial helper overlap.
    let phase_bytes = [
        size_of::<Output<'_>>(),
        size_of::<Output<'_>>(),
        size_of::<Result<Output<'_>, Error>>(),
        size_of::<Scratch>(),
        size_of::<Zeroizing<Scratch>>(),
        size_of::<Quote>(),
        size_of::<(matrix::Layout, matrix::Layout)>(),
        size_of::<[matrix::Schedule; 2]>(),
        size_of::<(
            ClearReferencePolicy<'_>,
            Limits,
            &Budget,
            &mut State,
            &mut WorkLedger,
        )>(),
        size_of::<(usize, std::ops::Range<usize>, Option<usize>)>() * 4,
        size_of::<Result<f64, Error>>() * 8,
        size_of::<[f64; 8]>(),
        poly::resources().control_bytes,
        size_of::<[f64; 60]>(), // Original immutable coefficient-table obligation.
        size_of::<[usize; 15]>()
            + size_of::<std::array::IntoIter<usize, 15>>()
            + 2 * size_of::<Result<Quote, Error>>()
            + 3 * size_of::<Result<usize, Error>>()
            + size_of::<ExternalCharge<'_>>()
            + size_of::<Result<ExternalCharge<'_>, pllm_garble::shared_allocation::Error>>(),
    ]
    .into_iter()
    .try_fold(0usize, add)?;
    if phase_bytes > cap.max_phase_bytes {
        return Err(Error::Resource);
    }
    Ok(Quote {
        original_bytes,
        phase_bytes,
        clear_work: CLEAR_WORK,
        schedules: [schedule, schedule],
    })
}

pub fn execute<'a>(
    policy: ClearReferencePolicy<'a>,
    cap: Limits,
    budget: &'a Budget,
    state: &mut State,
    work: &mut WorkLedger,
) -> Result<Output<'a>, Error> {
    if *state != State::Fresh {
        return Err(Error::Burned);
    }
    *state = State::Spent; // Every admitted failure/unwind burns this region.
    let q = quote(&policy, cap, work)?;
    if !policy.original.original_guard.is_for(budget)
        || policy.original.original_guard.payload_bytes() < q.original_bytes
    {
        return Err(Error::Source);
    }
    // Admit COMPLETE cumulative clear work before scratch initialization/kernels.
    work.debit(q.clear_work, 0).map_err(|_| Error::Work)?;
    let total = add(
        budget.used_bytes(),
        add(q.phase_bytes, size_of::<ExternalCharge<'_>>())?,
    )?;
    budget.admit_peak(total).map_err(|_| Error::Resource)?;
    let charge = budget
        .charge_external(q.phase_bytes)
        .map_err(|_| Error::Resource)?;
    let mut s = Zeroizing::new(Scratch::empty());
    let o = &policy.original;
    let (left, right) = layouts()?;
    left.pack(o.input, &mut s.a, limits()).map_err(mat)?;
    right
        .pack(o.sources[0].weights, &mut s.w1, limits())
        .map_err(mat)?;
    right
        .pack(o.sources[1].weights, &mut s.w2, limits())
        .map_err(mat)?;
    // Split fixed fields explicitly: no products/masks/replicated RHS allocation.
    let p = &mut *s;
    matrix::optimized_lower_mm(
        left,
        &p.a,
        right,
        &p.w1,
        left,
        &mut p.mm,
        &mut p.first_packed,
        limits(),
    )
    .map_err(mat)?;
    left.unpack(&p.first_packed, &mut p.first, limits())
        .map_err(mat)?;
    for i in 0..8 {
        p.first[i] = finite(p.first[i] + o.sources[0].bias[i % 4])?;
    }
    // Pinned cached A tuples converted to binary64 by Rust literals. Reverse
    // BEFORE evaluation, halve OUTER coefficients BEFORE its polynomial call.
    for i in 0..32 {
        p.inner[i] = A_FIRST[31 - i];
    }
    for i in 0..8 {
        let u = finite(p.first[i] / 64.)?;
        let inner = poly::stockmeyer(&p.inner, u, &mut p.poly).map_err(|_| Error::NonFinite)?;
        for j in 0..28 {
            p.outer[j] = A_SECOND[27 - j] * 0.5;
        }
        let helper =
            poly::stockmeyer(&p.outer, inner, &mut p.poly).map_err(|_| Error::NonFinite)?;
        let recovered = finite(64. * u)?;
        let factor = finite(0.5 + helper)?;
        p.activation[i] = finite(recovered * factor)?;
    }
    left.pack(&p.activation, &mut p.activation_packed, limits())
        .map_err(mat)?;
    matrix::optimized_lower_mm(
        left,
        &p.activation_packed,
        right,
        &p.w2,
        left,
        &mut p.mm,
        &mut p.final_packed,
        limits(),
    )
    .map_err(mat)?;
    left.unpack(&p.final_packed, &mut p.output, limits())
        .map_err(mat)?;
    for i in 0..8 {
        p.output[i] = finite(p.output[i] + o.sources[1].bias[i % 4])?;
    }
    Ok(Output {
        scratch: s,
        _original: policy,
        _charge: charge,
    })
}

// Cached benchmarks/research/thor_literal_reconciliation.py A["gelu"].
// Full precision strings, NOT printed paper P, source tuple order retained.
const A_FIRST: [f64; 32] = [
    -1.06240033e-05,
    1.64454894e-04,
    -5.83533517e-04,
    -3.80912692e-04,
    2.24431193e-03,
    8.92295204e-03,
    -1.05277477e-02,
    -1.91827040e-02,
    -2.04634786e-01,
    4.54014410e-01,
    -5.40759203e-01,
    5.67745523e+00,
    -1.36433727e+01,
    1.82574621e+01,
    -8.48849601e+01,
    1.28686741e+02,
    3.66720281e+02,
    -1.01400159e+03,
    -1.26278856e+02,
    2.21728878e+03,
    -9.95421415e+02,
    -2.31059465e+03,
    1.73583957e+03,
    1.27394360e+03,
    -1.27836230e+03,
    -3.66781716e+02,
    4.79663919e+02,
    4.94610178e+01,
    -9.06754761e+01,
    -2.36515790e+00,
    8.74311855e+00,
    1.62838703e-02,
];
const A_SECOND: [f64; 28] = [
    -1.70270667e+02,
    6.81076279e+01,
    1.79197364e+03,
    -6.81621043e+02,
    -8.49256169e+03,
    3.05629446e+03,
    2.39579397e+04,
    -8.10435126e+03,
    -4.48145152e+04,
    1.41297616e+04,
    5.86197512e+04,
    -1.70371505e+04,
    -5.51326382e+04,
    1.45532495e+04,
    3.77866438e+04,
    -8.87673890e+03,
    -1.89514802e+04,
    3.84972853e+03,
    6.94169727e+03,
    -1.16901058e+03,
    -1.84658407e+03,
    2.41693754e+02,
    3.54452276e+02,
    -3.24499570e+01,
    -4.91918227e+01,
    2.58122977e+00,
    5.78392852e+00,
    -9.45171527e-02,
];

#[cfg(test)]
#[path = "clear_ffn_region_tests.rs"]
mod tests;
