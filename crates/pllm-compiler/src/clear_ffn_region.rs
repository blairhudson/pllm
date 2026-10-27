//! Explicit artifact-adapted CLEAR reference only. No default GeluTanh override,
//! evaluated-author profile, encrypted capability, bootstrap or residual step.
//! ClearInputs retains actual metadata, tensors, budget and cumulative ledger.
//! Ordinary fixture custody does not authenticate checkpoints or source artifacts.
//! Cached call source: crypto-starlab/THOR@b513c06aaabccc04b560721e35e9a98008428c25,
//! nonlinear/polynomial.py grouping; cached reconciliation A[gelu], not printed P.
//! Bounds below cover named logical owners/work, NOT allocator/callframes/RSS.
//! New float64, flat-attribute reference regions only; no automatic migration of
//! existing float32 regions and no evaluated-run identity is inferred.
use crate::clear_input_owner::{self as owner, BorrowedOutput, ClearInputs, Phase, Prepared};
use crate::clear_value_binding::{BoundValue, RetainedProducer, ValueEdge};
use pllm_core::{polynomial_schedule as poly, slot_matrix_layout as matrix};
use pllm_models::{ModelOperation, ModelOperator};
use std::{cell::Ref, mem::size_of};
use zeroize::{Zeroize, Zeroizing};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Owner(owner::Error),
    Region,
    Shape,
    Policy,
    Source,
    Resource,
    Work,
    NonFinite,
}
impl From<owner::Error> for Error {
    fn from(error: owner::Error) -> Self {
        Self::Owner(error)
    }
}
#[derive(Clone, Copy)]
pub struct Limits {
    pub max_owner_bytes: usize,
    pub max_phase_bytes: usize,
    pub max_work: usize,
}
/// Bounded metadata preflight visits/bytes, not a CPU-instruction estimate.
/// Includes three fixed schemas, names <=256, keys <=64, shape/selection and
/// bounded name census. Original builder already paid opaque metadata storage;
/// cross-namespace comparisons are debited separately before their traversal.
pub const PREFLIGHT_CONTROL: usize = 12 * 1024;
// Logical visits/arithmetic/initialization/cleanup, NOT cycles: geometry24;
// packs96; corrected matrix validation/products/scratch/output2*(152+16+8);
// unpacks48; bias16 arithmetic+checks; reversal32 + eight outer read/halves56;
// polynomial
// arithmetic+finite checks2*8*(74+65); polynomial inputs/coefficients8*62;
// four wrapper arithmetic+checks per scalar; 197 float cells across all three
// payloads, eight visits each for initialization/moves/redundant all-path wipes.
pub const CLEAR_WORK: usize = 24
    + 96
    + 2 * (152 + 16 + 8)
    + 48
    + 32
    + 32
    + 8 * 56
    + 2 * 8 * (74 + 65)
    + 8 * 62
    + 8 * 8
    + 8 * 197;

/// Select ORIGINAL descriptors and tensors, never caller-provided numeric views.
#[derive(Clone, Copy)]
pub struct Selection {
    pub operations: [usize; 3],
    /// Input, first weight/bias, second weight/bias.
    pub tensors: [usize; 5],
}
#[derive(Clone, Copy)]
pub struct ClearReferencePolicy {
    _explicit: (),
}
impl ClearReferencePolicy {
    /// Explicitly selects the complete cached A[gelu] call, NOT GeluTanh math.
    /// Precision is binary64/non-fused; ordinary float32 model ops are refused.
    pub fn artifact_call_gelu() -> Self {
        Self { _explicit: () }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Quote {
    pub original_bytes: usize,
    pub phase_bytes: usize,
    pub clear_work: usize,
    pub additional_work: usize,
    pub cumulative_work: usize,
    pub schedules: [matrix::Schedule; 2],
}
struct Affine {
    a: [f64; 8],
    w: [f64; 16],
    mm: [f64; 16],
    packed: [f64; 8],
    output: [f64; 8],
}
impl Affine {
    fn empty() -> Self {
        Self {
            a: [0.; 8],
            w: [0.; 16],
            mm: [0.; 16],
            packed: [0.; 8],
            output: [0.; 8],
        }
    }
}
impl Zeroize for Affine {
    fn zeroize(&mut self) {
        #[cfg(test)]
        let partial = self
            .a
            .iter()
            .chain(&self.w)
            .chain(&self.mm)
            .chain(&self.packed)
            .chain(&self.output)
            .any(|&x| x != 0.);
        self.a.zeroize();
        self.w.zeroize();
        self.mm.zeroize();
        self.packed.zeroize();
        self.output.zeroize();
        #[cfg(test)]
        observe::wipe(
            partial,
            self.a
                .iter()
                .chain(&self.w)
                .chain(&self.mm)
                .chain(&self.packed)
                .chain(&self.output),
        );
    }
}
struct Activation {
    output: [f64; 8],
    inner: [f64; 32],
    outer: [f64; 28],
    poly: poly::Workspace,
}
impl Activation {
    fn empty() -> Self {
        Self {
            output: [0.; 8],
            inner: [0.; 32],
            outer: [0.; 28],
            poly: poly::Workspace::empty(),
        }
    }
}
impl Zeroize for Activation {
    fn zeroize(&mut self) {
        #[cfg(test)]
        let partial = self
            .output
            .iter()
            .chain(&self.inner)
            .chain(&self.outer)
            .any(|&x| x != 0.);
        self.output.zeroize();
        self.inner.zeroize();
        self.outer.zeroize();
        self.poly.zeroize();
        #[cfg(test)]
        observe::wipe(
            partial,
            self.output.iter().chain(&self.inner).chain(&self.outer),
        );
    }
}
struct First<'a> {
    inner: BorrowedOutput<'a, Zeroizing<Affine>>,
    operation: usize,
}
impl RetainedProducer for First<'_> {
    fn original(&self) -> &ClearInputs {
        self.inner.original()
    }
    fn operation_index(&self) -> usize {
        self.operation
    }
}
struct Activated<'a> {
    inner: BorrowedOutput<'a, Zeroizing<Activation>>,
    binding: BoundValue<'a, First<'a>>,
}
impl RetainedProducer for Activated<'_> {
    fn original(&self) -> &ClearInputs {
        self.inner.original()
    }
    fn operation_index(&self) -> usize {
        self.binding.consumer()
    }
}
/// Immutable final output retains both actual intermediate producers. Consumer
/// payload/charge retires before upstream payloads. No caller guard/ledger/state.
/// ```
/// use pllm_compiler::{clear_input_owner::ClearInputs, clear_ffn_region::*};
/// fn run(input: &ClearInputs, ids: Selection, cap: Limits) -> Output<'_> {
///     execute(quote(input, ids, ClearReferencePolicy::artifact_call_gelu(), cap).unwrap()).unwrap()
/// }
/// ```
/// ```compile_fail,E0505
/// use pllm_compiler::{clear_input_owner::ClearInputs, clear_ffn_region::*};
/// fn retire(input: ClearInputs, ids: Selection, cap: Limits) {
///     let out = execute(quote(&input, ids, ClearReferencePolicy::artifact_call_gelu(), cap).unwrap()).unwrap();
///     drop(input);
///     std::hint::black_box(out.values());
/// }
/// ```
/// ```compile_fail,E0594
/// use pllm_compiler::clear_ffn_region::Output;
/// fn mutate(out: &Output<'_>) { out.values()[0] = 7.0; }
/// ```
pub struct Output<'a> {
    inner: BorrowedOutput<'a, Zeroizing<Affine>>,
    binding: BoundValue<'a, Activated<'a>>,
}
impl Output<'_> {
    pub fn values(&self) -> &[f64; 8] {
        &self.inner.values().output
    }
    pub fn first_affine(&self) -> &[f64; 8] {
        &self
            .binding
            .producer()
            .binding
            .producer()
            .inner
            .values()
            .output
    }
    pub fn activated(&self) -> &[f64; 8] {
        &self.binding.producer().inner.values().output
    }
    pub fn original(&self) -> &ClearInputs {
        self.inner.original()
    }
    pub fn operation(&self) -> Ref<'_, ModelOperation> {
        Ref::map(self.original().operations(), |ops| {
            &ops[self.binding.consumer()]
        })
    }
}
impl RetainedProducer for Output<'_> {
    fn original(&self) -> &ClearInputs {
        self.inner.original()
    }
    fn operation_index(&self) -> usize {
        self.binding.consumer()
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
fn validate(input: &ClearInputs, ids: Selection) -> Result<[ValueEdge<'_>; 2], Error> {
    let all = input.operations();
    let tensors = input.tensors();
    let [up, activation, down] = ids.operations;
    if !(up < activation && activation < down)
        || ids
            .tensors
            .iter()
            .enumerate()
            .any(|(i, t)| ids.tensors[..i].contains(t))
        || all
            .iter()
            .any(|op| tensors.iter().any(|t| t.name() == op.id))
    {
        return Err(Error::Region);
    }
    let ops = [all.get(up), all.get(activation), all.get(down)].map(|op| op.ok_or(Error::Region));
    let [a, b, c] = ops;
    let ops = [a?, b?, c?];
    if [ops[0].operator, ops[1].operator, ops[2].operator]
        != [
            ModelOperator::Linear,
            ModelOperator::GeluTanh,
            ModelOperator::Linear,
        ]
    {
        return Err(Error::Region);
    }
    for (i, id) in ids.tensors.into_iter().enumerate() {
        let t = tensors.get(id).ok_or(Error::Shape)?;
        let shape: &[usize] = match i {
            0 => &[2, 4],
            1 | 3 => &[4, 4],
            _ => &[4],
        };
        if t.shape() != shape || input.f64_tensor(id)?.len() != shape.iter().product::<usize>() {
            return Err(Error::Shape);
        }
    }
    let input_name = tensors[ids.tensors[0]].name();
    for (i, op) in ops.iter().enumerate() {
        if op.state_kind.is_some() || op.output_shape != [2, 4] || op.inputs.len() != 1 {
            return Err(Error::Shape);
        }
        if op.layer != ops[0].layer || ops.iter().take(i).any(|other| other.id == op.id) {
            return Err(Error::Region);
        }
        if op.inputs[0] != (if i == 0 { input_name } else { &ops[i - 1].id }) {
            return Err(Error::Source);
        }
        let attrs = op.attributes.as_object().ok_or(Error::Policy)?;
        if i == 1 {
            if attrs.len() != 1
                || attrs.get("compute_dtype").and_then(|v| v.as_str()) != Some("float64")
            {
                return Err(Error::Policy);
            }
        } else {
            let offset = if i == 0 { 1 } else { 3 };
            let weight = tensors[ids.tensors[offset]].name();
            let bias = tensors[ids.tensors[offset + 1]].name();
            if attrs.len() != 7 {
                return Err(Error::Policy);
            }
            for key in attrs.keys() {
                if key.len() > 64 {
                    return Err(Error::Policy);
                }
            }
            if attrs.get("weight").and_then(|v| v.as_str()) != Some(weight)
                || attrs.get("bias").and_then(|v| v.as_str()) != Some(bias)
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
        }
    }
    Ok([
        ValueEdge::new(input, up, activation).ok_or(Error::Source)?,
        ValueEdge::new(input, activation, down).ok_or(Error::Source)?,
    ])
}

/// One-use whole-region admission; dropping refunds bytes, NEVER original work.
/// All three LIVE phases are reserved before any numeric factory. No output is
/// published until both private operation bindings and all kernels finish.
pub struct Admitted<'a> {
    edges: [ValueEdge<'a>; 2],
    ids: Selection,
    resources: Quote,
    first: Phase<'a, Zeroizing<Affine>>,
    activation: Phase<'a, Zeroizing<Activation>>,
    final_affine: Phase<'a, Zeroizing<Affine>>,
}
impl Admitted<'_> {
    pub fn resources(&self) -> Quote {
        self.resources
    }
}

// Retained payload plus constructing/moving old/new payload and Result overlap.
// Metadata is an explicit upper bound checked against concrete wrappers below;
// coefficient table and shared polynomial controls are separately inventoried.
const METADATA: usize = 4096;
fn affine_bytes() -> usize {
    4 * size_of::<Affine>() + METADATA
}
fn activation_bytes() -> usize {
    4 * size_of::<Activation>()
        + METADATA
        + size_of::<[f64; 60]>()
        + poly::resources().control_bytes
}
fn charged<T: Zeroize>(bytes: usize) -> Result<usize, Error> {
    add(
        add(
            add(bytes, size_of::<BorrowedOutput<'_, T>>())?,
            size_of::<Prepared<'_>>(),
        )?,
        size_of::<pllm_garble::shared_allocation::ExternalCharge<'_>>(),
    )
}

// Shared contract with ClearInputs::begin_attempt's fixed 32-unit debit. The
// owner burns before debit; this adapter must refuse insufficient work first.
const ATTEMPT_WORK: usize = 32;

fn check_work(input: &ClearInputs, cap: Limits, amount: usize) -> Result<(), Error> {
    let cumulative = input
        .consumed_work()
        .checked_add(amount)
        .ok_or(Error::Work)?;
    if cumulative > cap.max_work.min(owner::MAX_WORK) {
        return Err(Error::Work);
    }
    if input.remaining_work() < amount {
        return Err(Error::Owner(owner::Error::Work));
    }
    Ok(())
}

/// Replaces borrowed-Vec/caller-State API: compatibility would bypass custody.
/// Adapter prechecks 32 units before each owner ticket; admitted tickets remain
/// burned on any later failure, with no work refund or whole-region rollback.
pub fn quote(
    input: &ClearInputs,
    ids: Selection,
    _policy: ClearReferencePolicy,
    cap: Limits,
) -> Result<Admitted<'_>, Error> {
    let before = input.consumed_work();
    check_work(input, cap, ATTEMPT_WORK)?;
    let ticket = input.begin_attempt()?;
    check_work(input, cap, ATTEMPT_WORK)?;
    let activation_ticket = input.begin_attempt()?;
    check_work(input, cap, ATTEMPT_WORK)?;
    let final_ticket = input.begin_attempt()?;
    check_work(input, cap, PREFLIGHT_CONTROL)?;
    let (_, prepared) = ticket.with_ledger(|ledger| {
        ledger
            .debit(PREFLIGHT_CONTROL, 0)
            .map_err(|_| owner::Error::Work)
    })?;
    // The first callback has released its original-ledger borrow. Prepaid
    // bounded census precedes the full checked cross-namespace/numeric debit.
    let names = input.tensors().iter().try_fold(0usize, |sum, t| {
        sum.checked_add(1)
            .and_then(|n| n.checked_add(t.name().len()))
            .ok_or(Error::Work)
    })?;
    let remaining = input
        .operations()
        .len()
        .checked_mul(names)
        .and_then(|census| census.checked_add(CLEAR_WORK))
        .ok_or(Error::Work)?;
    check_work(input, cap, remaining)?;
    let (_, activation_prepared) = activation_ticket.with_ledger(|ledger| {
        ledger
            .debit(remaining, 0)
            .map_err(|_| owner::Error::Work)
    })?;
    let (_, final_prepared) = final_ticket.with_ledger(|_| Ok(()))?;
    let edges = validate(input, ids)?;
    if input.charged_bytes() > cap.max_owner_bytes {
        return Err(Error::Resource);
    }
    // All numeric initialization/kernel/retirement work was paid on the SAME
    // original ledger before any descriptor validation or numeric factory.
    let (left, _) = layouts()?;
    let schedule = matrix::optimized_lower_schedule(left).map_err(mat)?;
    let metadata = size_of::<Admitted<'_>>()
        + 2 * size_of::<Output<'_>>()
        + size_of::<Result<Output<'_>, Error>>()
        + size_of::<Selection>()
        + size_of::<(matrix::Layout, matrix::Layout)>()
        + 8 * size_of::<(usize, std::ops::Range<usize>, Option<usize>)>()
        + 8 * size_of::<Result<f64, Error>>();
    if metadata > 3 * METADATA {
        return Err(Error::Resource);
    }
    let phase_bytes = add(
        2 * charged::<Zeroizing<Affine>>(affine_bytes())?,
        charged::<Zeroizing<Activation>>(activation_bytes())?,
    )?;
    if phase_bytes > cap.max_phase_bytes {
        return Err(Error::Resource);
    }
    let resources = Quote {
        original_bytes: input.charged_bytes(),
        phase_bytes,
        clear_work: CLEAR_WORK,
        additional_work: input.consumed_work() - before,
        cumulative_work: input.consumed_work(),
        schedules: [schedule, schedule],
    };
    let first = prepared.reserve(affine_bytes())?;
    let activation = activation_prepared.reserve(activation_bytes())?;
    let final_affine = final_prepared.reserve(affine_bytes())?;
    Ok(Admitted {
        edges,
        ids,
        resources,
        first,
        activation,
        final_affine,
    })
}

fn affine(x: &[f64], weights: &[f64], bias: &[f64], p: &mut Affine) -> Result<(), Error> {
    let (left, right) = layouts()?;
    left.pack(x, &mut p.a, limits()).map_err(mat)?;
    right.pack(weights, &mut p.w, limits()).map_err(mat)?;
    // Split fixed fields explicitly: no products/masks/replicated RHS allocation.
    matrix::optimized_lower_mm(
        left,
        &p.a,
        right,
        &p.w,
        left,
        &mut p.mm,
        &mut p.packed,
        limits(),
    )
    .map_err(mat)?;
    left.unpack(&p.packed, &mut p.output, limits())
        .map_err(mat)?;
    for i in 0..8 {
        p.output[i] = finite(p.output[i] + bias[i % 4])?;
    }
    Ok(())
}

fn activate(x: &[f64; 8], p: &mut Activation) -> Result<(), Error> {
    // Pinned cached A tuples converted to binary64 by Rust literals. Reverse
    // BEFORE evaluation, halve OUTER coefficients BEFORE its polynomial call.
    for i in 0..32 {
        p.inner[i] = A_FIRST[31 - i];
    }
    for i in 0..8 {
        let u = finite(x[i] / 64.)?;
        let inner = poly::stockmeyer(&p.inner, u, &mut p.poly).map_err(|_| Error::NonFinite)?;
        for j in 0..28 {
            p.outer[j] = A_SECOND[27 - j] * 0.5;
        }
        let helper =
            poly::stockmeyer(&p.outer, inner, &mut p.poly).map_err(|_| Error::NonFinite)?;
        let recovered = finite(64. * u)?;
        let factor = finite(0.5 + helper)?;
        p.output[i] = finite(recovered * factor)?;
    }
    Ok(())
}

/// Admission is consumed once, including failure/unwind; no tentative prefix
/// escapes. Zeroizing payloads are declared AFTER every live phase reservation.
/// ```compile_fail,E0382
/// use pllm_compiler::clear_ffn_region::{Admitted, execute};
/// fn replay(admitted: Admitted<'_>) {
///     let _ = execute(admitted);
///     let _ = execute(admitted);
/// }
/// ```
pub fn execute(admitted: Admitted<'_>) -> Result<Output<'_>, Error> {
    let Admitted {
        edges: [first_edge, final_edge],
        ids,
        first,
        activation,
        final_affine,
        ..
    } = admitted;
    let input = first_edge.original();
    let x = input.f64_tensor(ids.tensors[0])?;
    let w1 = input.f64_tensor(ids.tensors[1])?;
    let b1 = input.f64_tensor(ids.tensors[2])?;
    let w2 = input.f64_tensor(ids.tensors[3])?;
    let b2 = input.f64_tensor(ids.tensors[4])?;
    #[cfg(test)]
    observe::factory();
    let mut up = Zeroizing::new(Affine::empty());
    affine(&x, &w1, &b1, &mut up)?;
    let up = First {
        inner: first.finish(up),
        operation: ids.operations[0],
    };
    let binding = first_edge.bind(up).map_err(|_| Error::Source)?;
    #[cfg(test)]
    observe::factory();
    let mut active = Zeroizing::new(Activation::empty());
    activate(&binding.producer().inner.values().output, &mut active)?;
    let active = Activated {
        inner: activation.finish(active),
        binding,
    };
    let binding = final_edge.bind(active).map_err(|_| Error::Source)?;
    #[cfg(test)]
    observe::factory();
    let mut down = Zeroizing::new(Affine::empty());
    affine(
        &binding.producer().inner.values().output,
        &w2,
        &b2,
        &mut down,
    )?;
    Ok(Output {
        inner: final_affine.finish(down),
        binding,
    })
}

// Fixed-size test observation on actual factory/wipe paths, not authority or a
// receipt log. Shared Workspace's private fields are wiped by its unchanged
// Zeroize implementation, not inspected by this adapter's observation hook.
#[cfg(test)]
mod observe {
    use std::cell::Cell;
    #[derive(Clone, Copy, Default, Debug)]
    pub(super) struct Counts {
        pub factories: usize,
        pub wipes: usize,
        pub partial: usize,
        pub dirty: usize,
    }
    thread_local! {
        static COUNTS: Cell<Counts> = Cell::new(Counts::default());
        static PANIC_AT: Cell<usize> = const { Cell::new(0) };
    }
    pub(super) fn snapshot() -> Counts {
        COUNTS.with(Cell::get)
    }
    pub(super) fn panic_at(factory: usize) {
        PANIC_AT.with(|p| p.set(factory));
    }
    pub(super) fn factory() {
        let mut c = snapshot();
        c.factories += 1;
        COUNTS.with(|v| v.set(c));
        PANIC_AT.with(|p| {
            if p.get() == c.factories {
                p.set(0);
                panic!("FFN factory cancellation");
            }
        });
    }
    pub(super) fn wipe<'a>(partial: bool, mut values: impl Iterator<Item = &'a f64>) {
        let mut c = snapshot();
        c.wipes += 1;
        c.partial += usize::from(partial);
        c.dirty += usize::from(values.any(|&v| v != 0.));
        COUNTS.with(|v| v.set(c));
    }
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
// Replaces legacy borrowed-Vec test entrypoints, not shared compiler callers.
#[path = "clear_ffn_retained_tests.rs"]
mod tests;
