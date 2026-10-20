use super::*;
use pllm_models::{DecoderPlan, ModelOperation, ModelOperator as Op};
use sha2::{Digest, Sha256};

/// Admission ceilings may only reduce the fixed native envelope.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_sequence: usize,
    pub max_selections: usize,
    pub max_work: usize,
    pub max_material_bytes: usize,
    pub max_frame_bytes: usize,
    pub max_public_copy_bytes: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            max_sequence: 16,
            max_selections: 4,
            max_work: 1 << 28,
            max_material_bytes: 1 << 30,
            max_frame_bytes: 1 << 28,
            max_public_copy_bytes: 1 << 30,
        }
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub struct Resources {
    pub kernels: usize,
    /// Exact native NeedMaterial requests for either role across all selections.
    /// Dealer issues one fresh paired material owner for each such request.
    pub primitive_requests_per_party: usize,
    pub work: usize,
    pub material_bytes: usize,
    pub frame_bytes: usize,
    pub public_copy_bytes: usize,
    pub metadata_bytes: usize,
    pub live_array_bytes_per_party: usize,
    /// Request-long cache metadata plus all retained immutable source tables.
    /// Referenced Plan tables remain separately priced for miss-time staging.
    pub source_table_cache_bytes_per_role: usize,
    pub peak_native_bytes: usize,
    pub combined_reservation_bytes: usize,
}

pub(super) enum Code {
    Embedding {
        weight: usize,
        positions: bool,
    },
    Linear {
        input: usize,
        weight: usize,
        bias: Option<usize>,
        input_output: bool,
    },
    Norm {
        input: usize,
        gamma: usize,
        beta: Option<usize>,
        layer: bool,
    },
    Add(usize, usize),
    Product(usize, usize),
    Activation {
        input: usize,
        silu: bool,
    },
    Slice {
        input: usize,
        start: usize,
    },
    Heads {
        input: usize,
        split: bool,
    },
    Rotary {
        input: usize,
        descriptor: usize,
    },
    Identity(usize),
    Scores {
        left: usize,
        right: usize,
        group: usize,
    },
    Scale {
        input: usize,
        coefficient: i64,
    },
    Mask(usize),
    Softmax(usize),
    Values {
        left: usize,
        right: usize,
        group: usize,
    },
    Last(usize),
    Argmax(usize),
}
pub(super) struct Node {
    pub code: Code,
    pub shape: Vec<usize>,
    pub range: Range,
    pub last_use: usize,
}
impl Node {
    pub fn operands(&self) -> [Option<usize>; 2] {
        use Code::*;
        match self.code {
            Embedding { .. } => [None, None],
            Add(a, b)
            | Product(a, b)
            | Scores {
                left: a, right: b, ..
            }
            | Values {
                left: a, right: b, ..
            } => [Some(a), Some(b)],
            Linear { input, .. }
            | Norm { input, .. }
            | Activation { input, .. }
            | Slice { input, .. }
            | Heads { input, .. }
            | Rotary { input, .. }
            | Scale { input, .. }
            | Identity(input)
            | Mask(input)
            | Softmax(input)
            | Last(input)
            | Argmax(input) => [Some(input), None],
        }
    }
}

/// Immutable instruction templates; no expanded per-scalar graph or public
/// constructor for ranges, output obligations, or program digests.
pub struct Program {
    pub(super) source: Source,
    pub(super) nodes: Vec<Node>,
    pub(super) initial: usize,
    pub(super) selections: usize,
    pub(super) sequence: usize,
    pub(super) vocabulary: usize,
    pub(super) digest: [u8; 32],
    pub(super) obligations: [u8; 32],
    pub(super) resources: Resources,
    pub(super) limits: Limits,
}

fn dimensions(op: &ModelOperation) -> Result<Vec<usize>, Error> {
    let mut out = vector(op.output_shape.len())?;
    for &n in &op.output_shape {
        out.push(usize::try_from(n).map_err(|_| Error::Resource)?);
    }
    elements(&out)?;
    Ok(out)
}
fn field<'a>(op: &'a ModelOperation, key: &str) -> Result<&'a str, Error> {
    op.attributes[key]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or(Error::Domain)
}
fn number(op: &ModelOperation, key: &str) -> Result<usize, Error> {
    usize::try_from(op.attributes[key].as_u64().ok_or(Error::Domain)?).map_err(|_| Error::Resource)
}
fn tensor(
    source: &Source,
    op: &ModelOperation,
    key: &str,
    shape: &[usize],
) -> Result<usize, Error> {
    source.find(field(op, key)?, shape)
}
fn optional_tensor(
    source: &Source,
    op: &ModelOperation,
    key: &str,
    shape: &[usize],
) -> Result<Option<usize>, Error> {
    if op.attributes[key].is_null() {
        Ok(None)
    } else {
        Ok(Some(tensor(source, op, key, shape)?))
    }
}
fn attr_shape(op: &ModelOperation, key: &str, shape: &[usize]) -> Result<(), Error> {
    let a = op.attributes[key].as_array().ok_or(Error::Domain)?;
    if a.len() != shape.len()
        || a.iter()
            .zip(shape)
            .any(|(a, &b)| a.as_u64() != Some(b as u64))
    {
        return Err(Error::Domain);
    }
    Ok(())
}
fn union(a: Range, b: Range) -> Range {
    Range {
        lo: a.lo.min(b.lo),
        hi: a.hi.max(b.hi),
    }
}
pub(super) fn public_dot_range(
    x: Range,
    weights: impl Iterator<Item = i64>,
    bias: i64,
) -> Result<Range, Error> {
    let mut sum = Range::new(0, 0)?;
    for w in weights {
        sum = sum.add(x.scale(w)?)?;
    }
    sum.floor()?.add(Range::new(bias as i128, bias as i128)?)
}
pub(super) fn dot_range(a: Range, b: Range, n: usize) -> Result<Range, Error> {
    let p = a.product(b)?;
    Range::new(p.lo as i128 * n as i128, p.hi as i128 * n as i128)?.floor()
}

/// Split-half source order: floor(l*c)+floor((-r)*s), then
/// floor(r*c)+floor(l*s). Negating the public coefficient proves the exact raw
/// signed product before its floor; subtracting an already floored r*s is wrong.
pub(super) fn rotary_output_range(input: Range, cosine: i64, sine: i64) -> Result<Range, Error> {
    let cosine = input.scale(cosine)?.floor()?;
    let negative = input
        .scale(sine.checked_neg().ok_or(Error::Domain)?)?
        .floor()?;
    let positive = input.scale(sine)?.floor()?;
    Ok(union(cosine.add(negative)?, cosine.add(positive)?))
}

fn metadata(
    value: &serde_json::Value,
    depth: usize,
    nodes: &mut usize,
    bytes: &mut usize,
) -> Result<(), Error> {
    *nodes = nodes.checked_add(1).ok_or(Error::Resource)?;
    if depth > 16 || *nodes > 32768 {
        return Err(Error::Resource);
    }
    match value {
        serde_json::Value::String(s) => {
            if s.capacity() > 1024 {
                return Err(Error::Resource);
            }
            *bytes = bytes.checked_add(s.capacity()).ok_or(Error::Resource)?;
        }
        serde_json::Value::Array(a) => {
            if a.capacity() > 32768 {
                return Err(Error::Resource);
            }
            for v in a {
                metadata(v, depth + 1, nodes, bytes)?;
            }
        }
        serde_json::Value::Object(a) => {
            for (k, v) in a {
                if k.len() > 1024 {
                    return Err(Error::Resource);
                }
                *bytes = bytes.checked_add(k.len()).ok_or(Error::Resource)?;
                metadata(v, depth + 1, nodes, bytes)?;
            }
        }
        _ => {}
    }
    if *bytes > 1 << 20 {
        return Err(Error::Resource);
    }
    Ok(())
}

impl Program {
    pub fn compile(
        plan: &DecoderPlan,
        source: Source,
        selections: usize,
        limits: Limits,
    ) -> Result<Self, Error> {
        let hard = Limits::default();
        if plan.schema_version.capacity() > 1024
            || plan.model_family.capacity() > 1024
            || plan.adapter.capacity() > 1024
        {
            return Err(Error::Resource);
        }
        if selections == 0
            || selections > limits.max_selections
            || limits.max_selections > hard.max_selections
            || limits.max_sequence > hard.max_sequence
            || limits.max_work > hard.max_work
            || limits.max_material_bytes > hard.max_material_bytes
            || limits.max_frame_bytes > hard.max_frame_bytes
            || limits.max_public_copy_bytes > hard.max_public_copy_bytes
            || !plan.transformations.is_empty()
        {
            return Err(Error::Unsupported);
        }
        let initial = usize::try_from(plan.prefill.query_sequence).map_err(|_| Error::Resource)?;
        let sequence = initial
            .checked_add(selections - 1)
            .filter(|&n| n <= limits.max_sequence)
            .ok_or(Error::Resource)?;
        if initial == 0
            || plan.prefill.batch != 1
            || plan.decode.batch != 1
            || plan.decode.maximum_key_sequence < sequence as u64
        {
            return Err(Error::Domain);
        }
        let mut meta_nodes = 0;
        let mut meta_bytes = 0usize;
        for graph in [&plan.prefill, &plan.decode] {
            if graph.operations.capacity() > MAX_NODES
                || graph.state_inputs.capacity() > MAX_NODES
                || graph.state_outputs.capacity() > MAX_NODES
                || graph.output.capacity() > 1024
                || graph.query_sequence == 0
                || graph.query_sequence > 16
                || graph.maximum_key_sequence > 16
            {
                return Err(Error::Resource);
            }
            for state in graph.state_inputs.iter().chain(&graph.state_outputs) {
                if state.id.capacity() > 1024
                    || state.shape.capacity() > 4
                    || state.maximum_sequence > 16
                {
                    return Err(Error::Resource);
                }
                let mut shape = vector(state.shape.len())?;
                for &n in &state.shape {
                    shape.push(usize::try_from(n).map_err(|_| Error::Resource)?);
                }
                elements(&shape)?;
            }
            for op in &graph.operations {
                if op.id.is_empty()
                    || op.id.capacity() > 1024
                    || op.inputs.capacity() > 8
                    || op.output_shape.capacity() > 4
                    || op.inputs.iter().any(|s| s.capacity() > 1024)
                {
                    return Err(Error::Resource);
                }
                dimensions(op)?;
                metadata(&op.attributes, 0, &mut meta_nodes, &mut meta_bytes)?;
            }
        }
        plan.validate().map_err(|_| Error::Domain)?;
        let mut ids = vector::<&str>(plan.prefill.operations.len())?;
        let mut originals = vector::<Vec<usize>>(plan.prefill.operations.len())?;
        let mut nodes: Vec<Node> = vector(plan.prefill.operations.len())?;
        let mut vocabulary = None;
        for op in &plan.prefill.operations {
            if op.operator == Op::OutputHead {
                let d = dimensions(op)?;
                if d.len() != 2 || d[0] != 1 || vocabulary.replace(d[1]).is_some() {
                    return Err(Error::Domain);
                }
            }
        }
        let mut head = None;
        for (index, op) in plan.prefill.operations.iter().enumerate() {
            if ids.contains(&op.id.as_str()) {
                return Err(Error::Domain);
            }
            let declared = dimensions(op)?;
            let operand = |i: usize| -> Result<usize, Error> {
                let id = op.inputs.get(i).ok_or(Error::Domain)?;
                ids.iter().position(|s| *s == id).ok_or(Error::Domain)
            };
            let unary = || -> Result<(usize, &Node, &Vec<usize>), Error> {
                let i = operand(0)?;
                Ok((i, &nodes[i], &originals[i]))
            };
            let same = |expected: &[usize]| -> Result<(), Error> {
                if declared == expected {
                    Ok(())
                } else {
                    Err(Error::Domain)
                }
            };
            let (code, shape, range) = match op.operator {
                Op::TokenLookup => {
                    if op.inputs.len() != 1 || declared.len() != 3 || declared[..2] != [1, initial]
                    {
                        return Err(Error::Domain);
                    }
                    let positions = matches!(
                        op.inputs[0].as_str(),
                        "input.positions" | "position_ids" | "input.position_ids"
                    );
                    if !positions && op.inputs[0] != "input.tokens" {
                        return Err(Error::Unsupported);
                    }
                    if positions
                        && (op.attributes["position_policy"] != "sequential_absolute"
                            || op.attributes["out_of_bounds"] != "reject")
                    {
                        return Err(Error::Unsupported);
                    }
                    let rows = if let Some(a) = op.attributes["weight_shape"].as_array() {
                        if a.len() != 2 || a[1].as_u64() != Some(declared[2] as u64) {
                            return Err(Error::Domain);
                        }
                        usize::try_from(a[0].as_u64().ok_or(Error::Domain)?)
                            .map_err(|_| Error::Resource)?
                    } else if !positions && op.attributes["weight_shape"].is_null() {
                        vocabulary.ok_or(Error::Domain)?
                    } else {
                        return Err(Error::Domain);
                    };
                    if positions && rows < sequence {
                        return Err(Error::Domain);
                    }
                    if !positions {
                        if vocabulary.is_some_and(|v| v != rows) {
                            return Err(Error::Domain);
                        }
                        vocabulary = Some(rows);
                    }
                    let weight = tensor(&source, op, "weight", &[rows, declared[2]])?;
                    (
                        Code::Embedding { weight, positions },
                        copied(&[1, sequence, declared[2]])?,
                        source.tensors[weight].range,
                    )
                }
                Op::Linear | Op::OutputHead => {
                    let (input, n, original) = unary()?;
                    if original.len() < 2
                        || declared.len() != original.len()
                        || declared[..declared.len() - 1] != original[..original.len() - 1]
                    {
                        return Err(Error::Domain);
                    }
                    let inner = *original.last().unwrap();
                    let columns = *declared.last().unwrap();
                    let input_output = match op.attributes["weight_layout"].as_str() {
                        Some("input_output") => true,
                        Some("output_input") | None => false,
                        _ => return Err(Error::Unsupported),
                    };
                    let ws = if input_output {
                        [inner, columns]
                    } else {
                        [columns, inner]
                    };
                    if !op.attributes["weight_shape"].is_null() {
                        attr_shape(op, "weight_shape", &ws)?;
                    }
                    let weight = tensor(&source, op, "weight", &ws)?;
                    let bias = optional_tensor(&source, op, "bias", &[columns])?;
                    if bias.is_some() && !op.attributes["bias_shape"].is_null() {
                        attr_shape(op, "bias_shape", &[columns])?;
                    }
                    let mut r = None;
                    for col in 0..columns {
                        let b = bias.map_or(0, |i| source.tensors[i].values[col]);
                        let row = public_dot_range(
                            n.range,
                            (0..inner).map(|k| {
                                source.tensors[weight].values[if input_output {
                                    k * columns + col
                                } else {
                                    col * inner + k
                                }]
                            }),
                            b,
                        )?;
                        r = Some(r.map_or(row, |r| union(r, row)));
                    }
                    let mut shape = copied(&n.shape)?;
                    *shape.last_mut().unwrap() = columns;
                    if op.operator == Op::OutputHead
                        && (shape != [1, columns] || vocabulary != Some(columns))
                    {
                        return Err(Error::Domain);
                    }
                    if op.operator == Op::OutputHead && head.replace(index).is_some() {
                        return Err(Error::Domain);
                    }
                    (
                        Code::Linear {
                            input,
                            weight,
                            bias,
                            input_output,
                        },
                        shape,
                        r.ok_or(Error::Domain)?,
                    )
                }
                Op::LayerNorm | Op::RmsNorm => {
                    let (input, n, original) = unary()?;
                    same(original)?;
                    if !op.attributes["axis"].is_null() && op.attributes["axis"] != -1 {
                        return Err(Error::Unsupported);
                    }
                    if !op.attributes["weight_offset"].is_null()
                        && op.attributes["weight_offset"] != 0
                    {
                        return Err(Error::Unsupported);
                    }
                    let width = *n.shape.last().unwrap();
                    if width > 4096 {
                        return Err(Error::Resource);
                    }
                    let gamma = tensor(&source, op, "weight", &[width])?;
                    let beta = optional_tensor(&source, op, "bias", &[width])?;
                    let layer = op.operator == Op::LayerNorm;
                    if layer
                        && (op.attributes["variance"] != "centered_mean_square"
                            || op.attributes["epsilon_placement"] != "inside_sqrt"
                            || op.attributes["affine_order"] != "normalize_multiply_add")
                    {
                        return Err(Error::Unsupported);
                    }
                    // Source fixed +1 in Q24 replaces source FP32 epsilon; this
                    // is the explicit D8 norm schedule, not a float emulation.
                    // For index bins, denominator >= (128/129)*128*S.
                    // Hence |z*inverse/4096| <= 4096*sqrt(129*k/128).
                    // Smallest B with 128*B^2 >= 2^24*129*k encloses both
                    // positive products and signed negative floors exactly.
                    let target = (1u128 << 24) * 129 * width as u128;
                    let (mut lo, mut hi) = (0u64, 1u64 << 30);
                    while lo < hi {
                        let mid = lo + (hi - lo) / 2;
                        if (mid as u128) * (mid as u128) * 128 >= target {
                            hi = mid;
                        } else {
                            lo = mid + 1;
                        }
                    }
                    let normalized = Range::new(-(lo as i128), lo as i128)?;
                    let mut range = None;
                    for c in 0..width {
                        let b = beta.map_or(0, |i| source.tensors[i].values[c]);
                        let r = normalized
                            .scale(source.tensors[gamma].values[c])?
                            .floor()?
                            .add(Range::new(b as i128, b as i128)?)?;
                        range = Some(range.map_or(r, |a| union(a, r)));
                    }
                    (
                        Code::Norm {
                            input,
                            gamma,
                            beta,
                            layer,
                        },
                        copied(&n.shape)?,
                        range.unwrap(),
                    )
                }
                Op::ResidualAdd | Op::Multiply => {
                    let a = operand(0)?;
                    let b = operand(1)?;
                    if op.inputs.len() != 2
                        || nodes[a].shape != nodes[b].shape
                        || originals[a] != originals[b]
                    {
                        return Err(Error::Domain);
                    }
                    same(&originals[a])?;
                    if op.operator == Op::ResidualAdd {
                        (
                            Code::Add(a, b),
                            copied(&nodes[a].shape)?,
                            nodes[a].range.add(nodes[b].range)?,
                        )
                    } else {
                        (
                            Code::Product(a, b),
                            copied(&nodes[a].shape)?,
                            nodes[a].range.product(nodes[b].range)?.floor()?,
                        )
                    }
                }
                Op::GeluTanh | Op::Silu => {
                    let (input, n, original) = unary()?;
                    same(original)?;
                    if n.range.lo < -(1i64 << 37) || n.range.hi >= 1i64 << 37 {
                        return Err(Error::Domain);
                    }
                    let range = if op.operator == Op::GeluTanh {
                        // Existing frozen descriptor proves this PUBLIC output
                        // envelope. No copied table or caller's proof is used.
                        let compiler_bytes = 3
                            * (source.reservation_bytes()
                                + meta_nodes * 128
                                + meta_bytes * 2
                                + plan.prefill.operations.len() * 4096
                                + plan.decode.operations.len() * 4096);
                        let available = pllm_garble::fusefss_backend::MAX_ALLOCATION_BYTES
                            .checked_sub(compiler_bytes)
                            .ok_or(Error::Resource)?;
                        let budget = pllm_garble::fusefss_backend::Budget::new(available / 3)?;
                        let binding = pllm_garble::fusefss_backend::composition::kernels::Binding {
                            source_digest: source.digest(),
                            contract_digest: source.digest(),
                        };
                        let (_, meta) = pllm_garble::fusefss_backend::composition::kernels::gelu(
                            binding,
                            n.range.gate(),
                            &budget,
                        )?;
                        Range::new(meta.range.min as i128, meta.range.max as i128)?
                    } else {
                        // Source SiLU is ReLU(original)-unsigned table correction;
                        // the frozen correction payload is in [0,65536].
                        Range::new(-65536, n.range.hi.max(0) as i128)?
                    };
                    (
                        Code::Activation {
                            input,
                            silu: op.operator == Op::Silu,
                        },
                        copied(&n.shape)?,
                        range,
                    )
                }
                Op::Slice => {
                    let (input, n, original) = unary()?;
                    if op.attributes["axis"] != -1 || op.attributes["squeeze"] != false {
                        return Err(Error::Unsupported);
                    }
                    let start = number(op, "start")?;
                    let end = number(op, "end")?;
                    let width = *original.last().unwrap();
                    if start >= end || end > width {
                        return Err(Error::Domain);
                    }
                    let mut expected = copied(original)?;
                    *expected.last_mut().unwrap() = end - start;
                    same(&expected)?;
                    let mut shape = copied(&n.shape)?;
                    *shape.last_mut().unwrap() = end - start;
                    (Code::Slice { input, start }, shape, n.range)
                }
                Op::Reshape => {
                    let (input, n, original) = unary()?;
                    let split = op.attributes["layout"] == "batch_heads_sequence_feature"
                        && (op.attributes["input_layout"] == "batch_sequence_hidden"
                            || op.attributes["input_layout"].is_null());
                    let merge = op.attributes["layout"] == "batch_sequence_hidden"
                        && (op.attributes["input_layout"] == "batch_heads_sequence_feature"
                            || op.attributes["input_layout"].is_null());
                    if split {
                        if original.len() != 3
                            || declared.len() != 4
                            || declared[0] != 1
                            || declared[2] != initial
                            || declared[1].checked_mul(declared[3]) != Some(original[2])
                        {
                            return Err(Error::Domain);
                        }
                        (
                            Code::Heads { input, split: true },
                            copied(&[1, declared[1], sequence, declared[3]])?,
                            n.range,
                        )
                    } else if merge {
                        if original.len() != 4
                            || declared != [1, initial, original[1] * original[3]]
                        {
                            return Err(Error::Domain);
                        }
                        (
                            Code::Heads {
                                input,
                                split: false,
                            },
                            copied(&[1, sequence, declared[2]])?,
                            n.range,
                        )
                    } else {
                        return Err(Error::Unsupported);
                    }
                }
                Op::RotaryEmbedding => {
                    let (input, n, original) = unary()?;
                    same(original)?;
                    if n.shape.len() != 4
                        || n.shape[3] % 2 != 0
                        || op.inputs.len() != 2
                        || !matches!(
                            op.inputs[1].as_str(),
                            "input.positions" | "position_ids" | "input.position_ids"
                        )
                        || op.attributes["pairing"] != "split_half"
                        || op.attributes["position_policy"] != "sequential_absolute"
                        || op.attributes["rotary_dimensions"].as_u64() != Some(n.shape[3] as u64)
                        || op.attributes["input_layout"] != "batch_heads_sequence_feature"
                        || op.attributes["output_layout"] != "batch_heads_sequence_feature"
                        || op.attributes["tail_policy"] != "unchanged"
                        || op.attributes["coefficient_profile"] != "pllm.numeric.rope.q30.libm.v1"
                        || !op.attributes["frequency_scaling"].is_null()
                    {
                        return Err(Error::Unsupported);
                    }
                    let descriptor = source
                        .rotary
                        .iter()
                        .position(|r| r.operation == op.id)
                        .ok_or(Error::Unsupported)?;
                    let r = &source.rotary[descriptor];
                    if r.dimensions != n.shape[3]
                        || r.positions < sequence
                        || op.attributes["theta"].as_u64() != Some(r.theta)
                    {
                        return Err(Error::Domain);
                    }
                    let mut range = None;
                    for (&c, &s) in r.cos.iter().zip(r.sin.iter()) {
                        let x = rotary_output_range(n.range, c, s)?;
                        range = Some(range.map_or(x, |a| union(a, x)));
                    }
                    (
                        Code::Rotary { input, descriptor },
                        copied(&n.shape)?,
                        range.unwrap(),
                    )
                }
                Op::KvCacheAppend | Op::CacheSuffix => {
                    let (input, n, original) = unary()?;
                    if original.len() != 4
                        || declared.len() != 4
                        || declared[0] != 1
                        || declared[1] != original[1]
                        || declared[3] != original[3]
                    {
                        return Err(Error::Domain);
                    }
                    if op.operator == Op::KvCacheAppend {
                        if op.attributes["mode"] != "initialize" || declared[2] < sequence {
                            return Err(Error::Unsupported);
                        }
                    } else if op.attributes["axis"] != 2
                        || op.attributes["semantics"] != "visible_valid_prefix"
                        || declared[2] != initial
                    {
                        return Err(Error::Unsupported);
                    }
                    (Code::Identity(input), copied(&n.shape)?, n.range)
                }
                Op::AttentionScores | Op::AttentionValues => {
                    let left = operand(0)?;
                    let right = operand(1)?;
                    let a = &nodes[left];
                    let b = &nodes[right];
                    if a.shape.len() != 4
                        || b.shape.len() != 4
                        || a.shape[0] != 1
                        || b.shape[0] != 1
                    {
                        return Err(Error::Domain);
                    }
                    let group = number(op, "group_size")?;
                    if group == 0 || a.shape[1] != b.shape[1] * group {
                        return Err(Error::Domain);
                    }
                    if op.operator == Op::AttentionScores {
                        if a.shape[2] != sequence
                            || b.shape[2] != sequence
                            || a.shape[3] != b.shape[3]
                        {
                            return Err(Error::Domain);
                        }
                        same(&[1, a.shape[1], initial, initial])?;
                        (
                            Code::Scores { left, right, group },
                            copied(&[1, a.shape[1], sequence, sequence])?,
                            dot_range(a.range, b.range, a.shape[3])?,
                        )
                    } else {
                        if a.shape[2] != sequence
                            || a.shape[3] != sequence
                            || b.shape[2] != sequence
                        {
                            return Err(Error::Domain);
                        }
                        same(&[1, a.shape[1], initial, b.shape[3]])?;
                        (
                            Code::Values { left, right, group },
                            copied(&[1, a.shape[1], sequence, b.shape[3]])?,
                            dot_range(a.range, b.range, sequence)?,
                        )
                    }
                }
                Op::AttentionScale => {
                    let (input, n, original) = unary()?;
                    same(original)?;
                    let d = number(op, "head_dim")?;
                    if d == 0 || d > 4096 {
                        return Err(Error::Domain);
                    }
                    let coefficient =
                        ((1.0f32 / (d as f32).sqrt()) as f64 * 4096.0).round_ties_even() as i64;
                    (
                        Code::Scale { input, coefficient },
                        copied(&n.shape)?,
                        n.range.scale(coefficient)?.floor()?,
                    )
                }
                Op::CausalMask => {
                    let (input, n, original) = unary()?;
                    same(original)?;
                    if n.shape.len() != 4
                        || n.shape[2] != sequence
                        || n.shape[3] != sequence
                        || (!op.attributes["masked_value"].is_null()
                            && op.attributes["masked_value"] != "negative_infinity")
                    {
                        return Err(Error::Unsupported);
                    }
                    (Code::Mask(input), copied(&n.shape)?, n.range)
                }
                Op::Softmax => {
                    let (input, n, original) = unary()?;
                    same(original)?;
                    if !matches!(n.code, Code::Mask(_)) || op.attributes["axis"] != -1 {
                        return Err(Error::Unsupported);
                    }
                    (
                        Code::Softmax(input),
                        copied(&n.shape)?,
                        Range::new(0, 4096)?,
                    )
                }
                Op::LastToken => {
                    let (input, n, original) = unary()?;
                    if original.len() != 3
                        || declared != [1, original[2]]
                        || op.attributes["axis"] != 1
                    {
                        return Err(Error::Domain);
                    }
                    (Code::Last(input), copied(&declared)?, n.range)
                }
                Op::GreedyTokenSelection => {
                    let (input, _n, original) = unary()?;
                    let stable = op.attributes["tie_break"] == "lowest_index"
                        || (op.attributes["tie_break"].is_null()
                            && op.attributes["policy"] == "pllm.greedy.v1");
                    if head != Some(input)
                        || declared != [1]
                        || original.as_slice() != [1, vocabulary.ok_or(Error::Domain)?]
                        || !stable
                    {
                        return Err(Error::Unsupported);
                    }
                    (
                        Code::Argmax(input),
                        copied(&[1])?,
                        Range::new(0, vocabulary.unwrap() as i128 - 1)?,
                    )
                }
                Op::TokenFeedback => {
                    let (input, n, _) = unary()?;
                    if !matches!(n.code, Code::Argmax(_))
                        || index + 1 != plan.prefill.operations.len()
                        || declared != [1, 1]
                        || plan.prefill.output != op.id
                    {
                        return Err(Error::Domain);
                    }
                    (Code::Identity(input), copied(&[1, 1])?, n.range)
                }
                _ => return Err(Error::Unsupported),
            };
            elements(&shape)?;
            ids.push(&op.id);
            originals.push(declared);
            nodes.push(Node {
                code,
                shape,
                range,
                last_use: index,
            });
        }
        if nodes.is_empty() || !matches!(nodes.last().unwrap().code, Code::Identity(_)) {
            return Err(Error::Domain);
        }
        if nodes
            .iter()
            .filter(|n| matches!(n.code, Code::Rotary { .. }))
            .count()
            != source.rotary.len()
        {
            // Every descriptor must bind exactly one actual semantic operator;
            // no ignored/misspelled IDs can silently enter the source contract.
            return Err(Error::Domain);
        }
        for i in 0..nodes.len() {
            for j in nodes[i].operands().into_iter().flatten() {
                nodes[j].last_use = nodes[j].last_use.max(i);
            }
        }
        let vocabulary = vocabulary.ok_or(Error::Domain)?;
        if vocabulary == 0 || vocabulary > MAX_VALUES {
            return Err(Error::Resource);
        }
        let mut hash = Sha256::new();
        hash.update(b"pllm.fss-decoder.program.fixed-T.full-prefix.v2\0rotary=floor(l*c)+floor((-r)*s),floor(r*c)+floor(l*s)\0");
        hash.update(source.digest());
        hash.update(plan.digest().as_str().as_bytes());
        for n in [initial, selections, sequence, vocabulary] {
            hash.update((n as u64).to_le_bytes());
        }
        let mut program = Self {
            source,
            nodes,
            initial,
            selections,
            sequence,
            vocabulary,
            digest: hash.finalize().into(),
            obligations: [0; 32],
            resources: Resources::default(),
            limits,
        };
        // Include admitted input metadata and public validation/compiler scratch,
        // even though the finished Program retains neither graph nor its names.
        program.resources.metadata_bytes = program.source.reservation_bytes()
            + std::mem::size_of::<Self>()
            + program.nodes.capacity() * std::mem::size_of::<Node>()
            + program
                .nodes
                .iter()
                .map(|n| n.shape.capacity() * std::mem::size_of::<usize>())
                .sum::<usize>()
            + meta_nodes * 128
            + meta_bytes * 2
            + plan.prefill.operations.len() * 4096
            + plan.decode.operations.len() * 4096;
        drop(ids);
        drop(originals);
        // Two party arrays, prefix/selection owners, and bounded gather/output
        // workspace; all native Plan/material/frame owners are added below.
        let mut live = 0usize;
        let mut peak = 0usize;
        for i in 0..program.nodes.len() {
            live = live
                .checked_add(elements(&program.nodes[i].shape)? * 8)
                .ok_or(Error::Resource)?;
            peak = peak.max(live);
            for j in 0..i {
                if program.nodes[j].last_use == i {
                    live -= elements(&program.nodes[j].shape)? * 8;
                }
            }
        }
        program.resources.live_array_bytes_per_party = peak
            + 8 * (sequence + selections)
            + 8 * MAX_VALUES * 8
            + program.nodes.len() * std::mem::size_of::<Option<zeroize::Zeroizing<Vec<u64>>>>();
        if 3 * program.resources.metadata_bytes + 2 * program.resources.live_array_bytes_per_party
            > pllm_garble::fusefss_backend::MAX_ALLOCATION_BYTES
        {
            return Err(Error::Resource);
        }
        super::runtime::preflight(&mut program)?;
        Ok(program)
    }
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
    pub fn source_digest(&self) -> [u8; 32] {
        self.source.digest()
    }
    pub fn resources(&self) -> Resources {
        self.resources
    }
    pub fn initial_tokens(&self) -> usize {
        self.initial
    }
    pub fn selections(&self) -> usize {
        self.selections
    }
    pub fn vocabulary(&self) -> usize {
        self.vocabulary
    }
    pub fn profile(&self) -> &'static str {
        if self
            .nodes
            .iter()
            .any(|n| matches!(n.code, Code::Rotary { .. }))
        {
            "fusefss.dense-rope-ring50-q12.v1"
        } else {
            "fusefss.gpt2-complete-ring50-q12.v1"
        }
    }
}
