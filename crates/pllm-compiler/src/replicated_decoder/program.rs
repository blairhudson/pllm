use super::*;
use pllm_models::{DecoderExpansionBudget, DecoderGraph, DecoderWorkload, ModelOperator as Op};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceTensor {
    pub shape: Vec<usize>,
    pub lower: i64,
    pub upper: i64,
    /// D10 explicitly publishes normalization coefficients. Other tensors stay
    /// party-local; the compiler rejects public bytes in their descriptors.
    pub public_q18: Option<Vec<i64>>,
    /// Commitment to the original encoded source bytes, checked by the issuer.
    pub digest: [u8; 32],
}

pub struct Program {
    pub(super) nodes: Vec<Node>,
    pub(super) inputs: usize,
    pub(super) outputs: Vec<(usize, usize)>,
    pub(super) retire: Vec<Vec<usize>>,
    digest: [u8; 32],
    payload_bytes: usize,
    operation_limit: u64,
}

#[derive(Clone)]
struct View {
    words: Vec<(usize, usize)>,
    shape: Vec<usize>,
}

struct Builder {
    nodes: Vec<Node>,
    allocated: usize,
}

impl Builder {
    fn push(&mut self, code: Code, lanes: usize, bounds: Interval) -> Result<usize, Error> {
        if lanes == 0 || lanes > MAX_VALUES || self.nodes.len() >= MAX_NODES {
            return Err(Error::Bounds);
        }
        let metadata = match &code {
            Code::Gather(v) => v.capacity() * std::mem::size_of::<(usize, usize)>(),
            Code::Tensor { program, .. } => program.admitted_bytes(),
            Code::Input(v) => v.capacity(),
            _ => 0,
        };
        self.allocated = self
            .allocated
            .checked_add(metadata + std::mem::size_of::<Node>() + 16 * lanes)
            .ok_or(Error::Bounds)?;
        if self.allocated > MAX_PROGRAM_BYTES {
            return Err(Error::Bounds);
        }
        let id = self.nodes.len();
        self.nodes.try_reserve(1).map_err(|_| Error::Allocation)?;
        self.nodes.push(Node {
            code,
            lanes,
            bounds,
        });
        Ok(id)
    }
    fn bounds(&self, words: &[(usize, usize)]) -> Result<Interval, Error> {
        let mut lo = i64::MAX;
        let mut hi = i64::MIN;
        for &(id, index) in words {
            let node = self.nodes.get(id).ok_or(Error::Binding)?;
            if index >= node.lanes {
                return Err(Error::Shape);
            }
            lo = lo.min(node.bounds.lower());
            hi = hi.max(node.bounds.upper());
        }
        Interval::new(lo, hi)
    }
    fn gather(&mut self, words: &[(usize, usize)]) -> Result<usize, Error> {
        let bound = self.bounds(words)?;
        self.push(Code::Gather(words.to_vec()), words.len(), bound)
    }
    fn affine(&mut self, left: usize, right: Option<usize>, c: Affine) -> Result<usize, Error> {
        let a = &self.nodes[left];
        if right.is_some_and(|id| self.nodes[id].lanes != a.lanes) {
            return Err(Error::Shape);
        }
        let bounds = c.output_bounds(InputBounds {
            left: a.bounds,
            right: right.map_or(Interval::new(0, 0)?, |id| self.nodes[id].bounds),
        })?;
        self.push(
            Code::Affine {
                left,
                right,
                coefficients: c,
            },
            a.lanes,
            bounds,
        )
    }
    fn product(
        &mut self,
        left: usize,
        right: usize,
        shape: Option<MatrixShape>,
        rescale: bool,
    ) -> Result<usize, Error> {
        let a = &self.nodes[left];
        let b = &self.nodes[right];
        let (n, terms) = if let Some(s) = shape {
            if a.lanes != s.left_elements() || b.lanes != s.right_elements() {
                return Err(Error::Shape);
            }
            (s.output_elements(), s.dimensions()[1])
        } else {
            if a.lanes != b.lanes || a.lanes > puma::MAX_LANES {
                return Err(Error::Shape);
            }
            (a.lanes, 1)
        };
        let bounds = InputBounds {
            left: a.bounds,
            right: b.bounds,
        };
        let range = if rescale {
            bounds.dot_output(terms)?
        } else {
            bounds.dot_product(terms)?
        };
        self.push(
            Code::Product {
                left,
                right,
                shape,
                rescale,
            },
            n,
            range,
        )
    }
    fn tensor(&mut self, input: usize, program: tensor::Program) -> Result<usize, Error> {
        if self.nodes[input].lanes != program.lanes() {
            return Err(Error::Shape);
        }
        let n = program.output_lanes();
        let range = program.output_bounds();
        self.push(
            Code::Tensor {
                input,
                program: Arc::new(program),
            },
            n,
            range,
        )
    }
    fn view(&self, id: usize, shape: Vec<usize>) -> View {
        View {
            words: (0..self.nodes[id].lanes).map(|i| (id, i)).collect(),
            shape,
        }
    }
    fn constant(&mut self, anchor: usize, value: i64) -> Result<usize, Error> {
        self.affine(anchor, None, affine(0, 0, value))
    }
    fn select(&mut self, bit: usize, yes: usize, no: usize) -> Result<usize, Error> {
        let bounds = Interval::new(
            self.nodes[yes]
                .bounds
                .lower()
                .min(self.nodes[no].bounds.lower()),
            self.nodes[yes]
                .bounds
                .upper()
                .max(self.nodes[no].bounds.upper()),
        )?;
        let difference = self.affine(yes, Some(no), affine(1, -1, 0))?;
        let product = self.product(bit, difference, None, false)?;
        let out = self.affine(no, Some(product), affine(1, 1, 0))?;
        // Only constructed for the native comparison circuit's arithmetic bit.
        self.nodes[out].bounds = bounds;
        Ok(out)
    }

    fn linear(&mut self, input: &View, weight: &View, rescale: bool) -> Result<View, Error> {
        let inner = *input.shape.last().ok_or(Error::Shape)?;
        let columns = *weight.shape.first().ok_or(Error::Shape)?;
        if weight.shape != [columns, inner] || inner > puma::MAX_LANES {
            return Err(Error::Shape);
        }
        let rows = input.words.len() / inner;
        let tile = (puma::MAX_LANES / inner).min(128);
        let mut words = Vec::new();
        for row in input.words.chunks_exact(inner) {
            let left = self.gather(row)?;
            for start in (0..columns).step_by(tile) {
                let n = tile.min(columns - start);
                let mut w = Vec::with_capacity(inner * n);
                for k in 0..inner {
                    for j in start..start + n {
                        w.push(weight.words[j * inner + k]);
                    }
                }
                let right = self.gather(&w)?;
                let output =
                    self.product(left, right, Some(MatrixShape::new(1, inner, n)?), rescale)?;
                words.extend((0..n).map(|i| (output, i)));
            }
        }
        Ok(View {
            words,
            shape: vec![1, rows, columns],
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn graph(
        &mut self,
        graph: &DecoderGraph,
        tokens: &View,
        sources: &BTreeMap<String, View>,
        public: &BTreeMap<String, SourceTensor>,
        vocabulary: usize,
        position: usize,
        state: &mut BTreeMap<(u64, pllm_models::StateKind), View>,
    ) -> Result<View, Error> {
        let sequence = tokens.words.len();
        let keys = position.checked_add(sequence).ok_or(Error::Bounds)?;
        if sequence != graph.query_sequence as usize || keys > graph.maximum_key_sequence as usize {
            return Err(Error::Shape);
        }
        let mut values = BTreeMap::<String, View>::new();
        for input in &graph.state_inputs {
            let key = (input.layer.ok_or(Error::Binding)?, input.kind);
            values.insert(
                input.id.clone(),
                state.get(&key).ok_or(Error::Binding)?.clone(),
            );
        }
        values.insert("input.tokens".into(), tokens.clone());
        let anchor = self.gather(&tokens.words[..1])?;
        let zero = self.constant(anchor, 0)?;
        for op in &graph.operations {
            let mut shape: Vec<usize> = op.output_shape.iter().map(|v| *v as usize).collect();
            if matches!(op.operator, Op::KvCacheAppend | Op::CacheSuffix) {
                shape[2] = keys;
            }
            if matches!(
                op.operator,
                Op::AttentionScores | Op::AttentionScale | Op::CausalMask | Op::Softmax
            ) {
                *shape.last_mut().ok_or(Error::Shape)? = keys;
            }
            let source = |name: &str| -> Result<&View, Error> {
                sources
                    .get(
                        op.attributes
                            .get(name)
                            .and_then(|v| v.as_str())
                            .ok_or(Error::Binding)?,
                    )
                    .ok_or(Error::Binding)
            };
            let a = op.inputs.first().and_then(|id| values.get(id));
            let b = op.inputs.get(1).and_then(|id| values.get(id));
            let output = match op.operator {
                Op::TokenLookup => {
                    let table = source("weight")?;
                    if table.shape != [vocabulary, *shape.last().ok_or(Error::Shape)?] {
                        return Err(Error::Shape);
                    }
                    let width = table.shape[1];
                    let table_bound = self.bounds(&table.words)?;
                    let mut words = vec![];
                    for &token in &tokens.words {
                        let mut one_hot = Vec::new();
                        for start in (0..vocabulary).step_by(128) {
                            let n = 128.min(vocabulary - start);
                            let x = self.gather(&vec![token; n])?;
                            let mut builder = tensor::Builder::new(n, self.nodes[x].bounds)?;
                            let coefficients: Vec<_> = (start..start + n)
                                .map(|i| affine(1, 0, -(i as i64)))
                                .collect();
                            let shifted = builder.affine_lanes(builder.input(), &coefficients)?;
                            let program = builder.finish(shifted)?;
                            let difference = self.tensor(x, program)?;
                            let program =
                                tensor::Program::equal_zero(n, self.nodes[difference].bounds)?;
                            let equal = self.tensor(difference, program)?;
                            one_hot.extend((0..n).map(|i| (equal, i)));
                        }
                        let selectors = View {
                            words: one_hot,
                            shape: vec![1, vocabulary],
                        };
                        let transposed = View {
                            words: (0..width)
                                .flat_map(|j| {
                                    (0..vocabulary).map(move |i| table.words[i * width + j])
                                })
                                .collect(),
                            shape: vec![width, vocabulary],
                        };
                        let row = self.linear(&selectors, &transposed, false)?;
                        // Source input IDs and protected argmax establish exactly
                        // one selector. The output is an original table element.
                        for &(id, _) in &row.words {
                            self.nodes[id].bounds = table_bound;
                        }
                        words.extend(row.words);
                    }
                    View {
                        words,
                        shape: shape.clone(),
                    }
                }
                Op::Linear | Op::OutputHead => {
                    let mut output =
                        self.linear(a.ok_or(Error::Binding)?, source("weight")?, true)?;
                    if op.attributes.get("bias").is_some_and(|v| !v.is_null()) {
                        let bias = source("bias")?;
                        if bias.shape != [*shape.last().ok_or(Error::Shape)?] {
                            return Err(Error::Shape);
                        }
                        let repeated: Vec<_> = (0..output.words.len())
                            .map(|i| bias.words[i % bias.words.len()])
                            .collect();
                        output.words = self.binary(&output.words, &repeated, false)?;
                    }
                    output.shape = shape.clone();
                    output
                }
                Op::RmsNorm => {
                    if op.attributes.get("weight_offset").and_then(|v| v.as_i64()) != Some(0) {
                        return Err(Error::Binding);
                    }
                    let gamma = public
                        .get(
                            op.attributes
                                .get("weight")
                                .and_then(|v| v.as_str())
                                .ok_or(Error::Binding)?,
                        )
                        .ok_or(Error::Binding)?;
                    let width = *shape.last().ok_or(Error::Shape)?;
                    if gamma.shape != [width] {
                        return Err(Error::Shape);
                    }
                    let mut words = vec![];
                    for row in a.ok_or(Error::Binding)?.words.chunks_exact(width) {
                        let input = self.gather(row)?;
                        let program = tensor::Program::rms_norm(
                            1,
                            width,
                            self.nodes[input].bounds,
                            gamma.public_q18.as_ref().ok_or(Error::Binding)?,
                        )?;
                        let output = self.tensor(input, program)?;
                        words.extend((0..width).map(|i| (output, i)));
                    }
                    View {
                        words,
                        shape: shape.clone(),
                    }
                }
                Op::Silu => {
                    let mut words = vec![];
                    for chunk in &a
                        .ok_or(Error::Binding)?
                        .words
                        .chunks(128)
                        .collect::<Vec<_>>()
                    {
                        let input = self.gather(chunk)?;
                        let program = tensor::Program::activation(
                            tensor::Activation::Silu,
                            chunk.len(),
                            self.nodes[input].bounds,
                        )?;
                        let output = self.tensor(input, program)?;
                        words.extend((0..chunk.len()).map(|i| (output, i)));
                    }
                    View {
                        words,
                        shape: shape.clone(),
                    }
                }
                Op::ResidualAdd | Op::Multiply => View {
                    words: self.binary(
                        &a.ok_or(Error::Binding)?.words,
                        &b.ok_or(Error::Binding)?.words,
                        op.operator == Op::Multiply,
                    )?,
                    shape: shape.clone(),
                },
                Op::Reshape => {
                    let a = a.ok_or(Error::Binding)?;
                    let words = match op.attributes.get("layout").and_then(|v| v.as_str()) {
                        Some("batch_heads_sequence_feature") if shape.len() == 4 => {
                            let (heads, rows, width) = (shape[1], shape[2], shape[3]);
                            (0..heads)
                                .flat_map(|h| {
                                    (0..rows).flat_map(move |r| {
                                        (0..width)
                                            .map(move |d| a.words[(r * heads + h) * width + d])
                                    })
                                })
                                .collect()
                        }
                        Some("batch_sequence_hidden") if a.shape.len() == 4 => {
                            let (heads, rows, width) = (a.shape[1], a.shape[2], a.shape[3]);
                            (0..rows)
                                .flat_map(|r| {
                                    (0..heads).flat_map(move |h| {
                                        (0..width).map(move |d| a.words[(h * rows + r) * width + d])
                                    })
                                })
                                .collect()
                        }
                        _ => return Err(Error::Binding),
                    };
                    View {
                        words,
                        shape: shape.clone(),
                    }
                }
                Op::RotaryEmbedding => {
                    if shape.len() != 4
                        || shape[3] > 128
                        || op.attributes.get("frequency_scaling").is_some()
                        || op.attributes.get("pairing").and_then(|v| v.as_str())
                            != Some("split_half")
                    {
                        return Err(Error::Binding);
                    }
                    let theta = op
                        .attributes
                        .get("theta")
                        .and_then(|v| v.as_u64())
                        .ok_or(Error::Binding)? as f32;
                    let width = shape[3];
                    let mut words = vec![];
                    for (index, row) in a
                        .ok_or(Error::Binding)?
                        .words
                        .chunks_exact(width)
                        .enumerate()
                    {
                        let (cos, sin) = tensor::Program::rotary_coefficients(
                            theta,
                            (position + index % sequence) as u32,
                            width,
                        )?;
                        let input = self.gather(row)?;
                        let program = tensor::Program::rotary_half(
                            1,
                            width,
                            self.nodes[input].bounds,
                            &cos,
                            &sin,
                        )?;
                        let output = self.tensor(input, program)?;
                        words.extend((0..width).map(|i| (output, i)));
                    }
                    View {
                        words,
                        shape: shape.clone(),
                    }
                }
                Op::KvCacheAppend => {
                    let key = (
                        op.layer.ok_or(Error::Binding)?,
                        op.state_kind.ok_or(Error::Binding)?,
                    );
                    let mode = op
                        .attributes
                        .get("mode")
                        .and_then(|v| v.as_str())
                        .ok_or(Error::Binding)?;
                    let input = if position == 0 {
                        if mode != "initialize" {
                            return Err(Error::Binding);
                        }
                        a
                    } else {
                        if mode != "append" {
                            return Err(Error::Binding);
                        }
                        b
                    }
                    .ok_or(Error::Binding)?;
                    let (heads, width) = (input.shape[1], input.shape[3]);
                    let mut words = vec![];
                    for head in 0..heads {
                        if position > 0 {
                            let previous = state.get(&key).ok_or(Error::Binding)?;
                            if previous.shape != [1, heads, position, width] {
                                return Err(Error::Shape);
                            }
                            words.extend_from_slice(
                                &previous.words
                                    [head * position * width..(head + 1) * position * width],
                            );
                        }
                        words.extend_from_slice(
                            &input.words[head * sequence * width..(head + 1) * sequence * width],
                        );
                    }
                    let out = View {
                        words,
                        shape: vec![1, heads, keys, width],
                    };
                    state.insert(key, out.clone());
                    out
                }
                Op::CacheSuffix => {
                    if op.attributes.get("semantics").and_then(|v| v.as_str())
                        != Some("visible_valid_prefix")
                    {
                        return Err(Error::Binding);
                    }
                    a.ok_or(Error::Binding)?.clone()
                }
                Op::AttentionScores => {
                    let q = a.ok_or(Error::Binding)?;
                    let k = b.ok_or(Error::Binding)?;
                    let (heads, width) = (q.shape[1], q.shape[3]);
                    let group = op
                        .attributes
                        .get("group_size")
                        .and_then(|v| v.as_u64())
                        .ok_or(Error::Binding)? as usize;
                    if group == 0 || heads / group != k.shape[1] {
                        return Err(Error::Shape);
                    }
                    let mut words = vec![];
                    for h in 0..heads {
                        for r in 0..sequence {
                            let query = View {
                                words: q.words
                                    [(h * sequence + r) * width..(h * sequence + r + 1) * width]
                                    .to_vec(),
                                shape: vec![1, width],
                            };
                            let key_rows = View {
                                words: k.words[(h / group) * keys * width
                                    ..((h / group) * keys + position + r + 1) * width]
                                    .to_vec(),
                                shape: vec![position + r + 1, width],
                            };
                            words.extend(self.linear(&query, &key_rows, true)?.words);
                            words.extend(std::iter::repeat_n((zero, 0), sequence - r - 1));
                        }
                    }
                    View {
                        words,
                        shape: shape.clone(),
                    }
                }
                Op::AttentionScale => {
                    let width = op
                        .attributes
                        .get("head_dim")
                        .and_then(|v| v.as_u64())
                        .ok_or(Error::Binding)?;
                    let coefficient = ((1.0_f32 / (width as f32).sqrt()) * ONE as f32) as i64;
                    let mut words = vec![];
                    for chunk in a.ok_or(Error::Binding)?.words.chunks(128) {
                        let input = self.gather(chunk)?;
                        let mut b = tensor::Builder::new(chunk.len(), self.nodes[input].bounds)?;
                        let raw = b.affine(b.input(), None, affine(coefficient, 0, 0))?;
                        let output = b.rescale(raw, 18)?;
                        let id = self.tensor(input, b.finish(output)?)?;
                        words.extend((0..chunk.len()).map(|i| (id, i)));
                    }
                    View {
                        words,
                        shape: shape.clone(),
                    }
                }
                Op::CausalMask => a.ok_or(Error::Binding)?.clone(),
                Op::Softmax => {
                    let mut words = vec![];
                    for (i, row) in a
                        .ok_or(Error::Binding)?
                        .words
                        .chunks_exact(keys)
                        .enumerate()
                    {
                        let width = position + i % sequence + 1;
                        let input = self.gather(&row[..width])?;
                        let program = tensor::Program::softmax(1, width, self.nodes[input].bounds)?;
                        let id = self.tensor(input, program)?;
                        words.extend((0..width).map(|i| (id, i)));
                        words.extend(std::iter::repeat_n((zero, 0), keys - width));
                    }
                    View {
                        words,
                        shape: shape.clone(),
                    }
                }
                Op::AttentionValues => {
                    let probabilities = a.ok_or(Error::Binding)?;
                    let values = b.ok_or(Error::Binding)?;
                    let (heads, width) = (shape[1], shape[3]);
                    let group = op
                        .attributes
                        .get("group_size")
                        .and_then(|v| v.as_u64())
                        .ok_or(Error::Binding)? as usize;
                    if group == 0 || heads / group != values.shape[1] {
                        return Err(Error::Shape);
                    }
                    let mut words = vec![];
                    for h in 0..heads {
                        for r in 0..sequence {
                            let valid = position + r + 1;
                            let p = View {
                                words: probabilities.words
                                    [(h * sequence + r) * keys..(h * sequence + r) * keys + valid]
                                    .to_vec(),
                                shape: vec![1, valid],
                            };
                            let mut selected = vec![];
                            for d in 0..width {
                                for k in 0..valid {
                                    selected
                                        .push(values.words[((h / group) * keys + k) * width + d]);
                                }
                            }
                            words.extend(
                                self.linear(
                                    &p,
                                    &View {
                                        words: selected,
                                        shape: vec![width, valid],
                                    },
                                    true,
                                )?
                                .words,
                            );
                        }
                    }
                    View {
                        words,
                        shape: shape.clone(),
                    }
                }
                Op::LastToken => {
                    let a = a.ok_or(Error::Binding)?;
                    let width = *shape.last().unwrap();
                    View {
                        words: a.words[a.words.len() - width..].to_vec(),
                        shape: shape.clone(),
                    }
                }
                Op::GreedyTokenSelection => {
                    let logits = a.ok_or(Error::Binding)?;
                    let mut best = self.gather(&logits.words[..1])?;
                    let mut index = self.constant(anchor, 0)?;
                    for (i, candidate) in logits.words.iter().enumerate().skip(1) {
                        let other = self.gather(&[*candidate])?;
                        let difference = self.affine(best, Some(other), affine(1, -1, 0))?;
                        let less = self.tensor(
                            difference,
                            tensor::Program::negative(1, self.nodes[difference].bounds)?,
                        )?;
                        best = self.select(less, other, best)?;
                        let other_index = self.constant(anchor, i as i64)?;
                        index = self.select(less, other_index, index)?;
                    }
                    self.view(index, shape.clone())
                }
                Op::TokenFeedback => a.ok_or(Error::Binding)?.clone(),
                _ => return Err(Error::Binding),
            };
            if output.words.len() != shape.iter().product::<usize>() {
                return Err(Error::Shape);
            }
            values.insert(op.id.clone(), output);
        }
        values.remove(&graph.output).ok_or(Error::Binding)
    }

    fn binary(
        &mut self,
        a: &[(usize, usize)],
        b: &[(usize, usize)],
        product: bool,
    ) -> Result<Vec<(usize, usize)>, Error> {
        if a.len() != b.len() {
            return Err(Error::Shape);
        }
        let mut words = vec![];
        for (a, b) in a.chunks(128).zip(b.chunks(128)) {
            let left = self.gather(a)?;
            let right = self.gather(b)?;
            let id = if product {
                self.product(left, right, None, true)?
            } else {
                self.affine(left, Some(right), affine(1, 1, 0))?
            };
            words.extend((0..a.len()).map(|i| (id, i)));
        }
        Ok(words)
    }
}

fn affine(left: i64, right: i64, constant: i64) -> Affine {
    Affine {
        left,
        right,
        constant,
    }
}

impl Program {
    /// Bounded source-mapped metadata admission runs before graph expansion.
    pub fn source_shapes(
        config: &[u8],
        rows: usize,
        outputs: usize,
    ) -> Result<BTreeMap<String, Vec<usize>>, String> {
        let plan = lower(config, rows, outputs)?;
        let vocabulary = plan
            .prefill
            .operations
            .iter()
            .find(|op| op.operator == Op::OutputHead)
            .ok_or("missing complete head")?
            .output_shape[1] as usize;
        let mut shapes = BTreeMap::new();
        let mut known = BTreeMap::new();
        for op in &plan.prefill.operations {
            if let Some(name) = op.attributes.get("weight").and_then(|v| v.as_str()) {
                let shape = match op.operator {
                    Op::TokenLookup => vec![vocabulary, *op.output_shape.last().unwrap() as usize],
                    Op::RmsNorm => vec![*op.output_shape.last().unwrap() as usize],
                    Op::Linear | Op::OutputHead => vec![
                        *op.output_shape.last().unwrap() as usize,
                        *known.get(&op.inputs[0]).ok_or("missing linear input")?,
                    ],
                    _ => return Err("unsupported replicated source operator".into()),
                };
                if shapes.get(name).is_some_and(|s| s != &shape) {
                    return Err("conflicting replicated source shape".into());
                }
                shapes.insert(name.into(), shape);
            }
            if let Some(name) = op.attributes.get("bias").and_then(|v| v.as_str()) {
                shapes.insert(name.into(), vec![*op.output_shape.last().unwrap() as usize]);
            }
            known.insert(op.id.clone(), *op.output_shape.last().unwrap() as usize);
        }
        if shapes
            .values()
            .map(|s| s.iter().product::<usize>())
            .sum::<usize>()
            > MAX_VALUES
        {
            return Err("replicated source payload limit".into());
        }
        Ok(shapes)
    }

    pub fn compile(
        config: &[u8],
        rows: usize,
        outputs: usize,
        sources: &BTreeMap<String, SourceTensor>,
    ) -> Result<Self, String> {
        let shapes = Self::source_shapes(config, rows, outputs)?;
        let first = lower(config, rows, outputs)?;
        let public: std::collections::BTreeSet<_> = first
            .prefill
            .operations
            .iter()
            .filter(|v| v.operator == Op::RmsNorm)
            .map(|op| {
                op.attributes
                    .get("weight")
                    .and_then(|v| v.as_str())
                    .ok_or("missing norm source")
            })
            .collect::<Result<_, _>>()?;
        if shapes.len() != sources.len()
            || sources.iter().any(|(k, v)| shapes.get(k) != Some(&v.shape))
        {
            return Err("replicated source shape set mismatch".into());
        }
        if sources.iter().any(|(k, v)| {
            v.digest == [0; 32]
                || public.contains(k.as_str()) != v.public_q18.is_some()
                || v.public_q18.as_ref().is_some_and(|x| {
                    x.len() != v.shape.iter().product::<usize>()
                        || x.iter().any(|n| *n < v.lower || *n > v.upper)
                })
        }) {
            return Err("replicated source commitment/public coefficient mismatch".into());
        }
        for source in sources.values() {
            if let Some(values) = &source.public_q18 {
                let mut h = Sha256::new();
                for value in values {
                    h.update(value.to_le_bytes());
                }
                if <[u8; 32]>::from(h.finalize()) != source.digest {
                    return Err("replicated public coefficient digest mismatch".into());
                }
            }
        }
        let result = (|| -> Result<Self, Error> {
            let mut builder = Builder {
                nodes: vec![],
                allocated: 0,
            };
            let mut inputs = BTreeMap::new();
            for (key, source) in sources {
                let bounds = Interval::new(source.lower, source.upper)?;
                if source.lower < -(1 << 40) || source.upper > 1 << 40 {
                    return Err(Error::Bounds);
                }
                if source.public_q18.is_some() {
                    continue;
                }
                let id = builder.push(
                    Code::Input(key.clone()),
                    source.shape.iter().product(),
                    bounds,
                )?;
                inputs.insert(key.clone(), builder.view(id, source.shape.clone()));
            }
            let first = lower(config, rows, outputs).map_err(|_| Error::Binding)?;
            let vocabulary = first
                .prefill
                .operations
                .iter()
                .find(|op| op.operator == Op::OutputHead)
                .ok_or(Error::Binding)?
                .output_shape[1] as usize;
            let input = builder.push(
                Code::Input("input.tokens".into()),
                rows,
                Interval::new(0, vocabulary as i64 - 1)?,
            )?;
            let count = builder.nodes.len();
            let mut tokens = builder.view(input, vec![1, rows]);
            let mut selected = vec![];
            for output in 0..outputs {
                // D9 recomputes the complete private prefix for each selection.
                // A fresh semantic prefill and fresh arithmetic are required:
                // retained KV would change both the stochastic numeric schedule
                // and the source method's charged work.
                let plan =
                    lower(config, rows + output, outputs - output).map_err(|_| Error::Binding)?;
                let mut state = BTreeMap::new();
                let result = builder.graph(
                    &plan.prefill,
                    &tokens,
                    &inputs,
                    sources,
                    vocabulary,
                    0,
                    &mut state,
                )?;
                if result.words.len() != 1 {
                    return Err(Error::Shape);
                }
                selected.extend_from_slice(&result.words);
                tokens.words.extend_from_slice(&result.words);
                tokens.shape = vec![1, tokens.words.len()];
            }
            let mut last: Vec<_> = (0..builder.nodes.len()).collect();
            let mut work = 0u64;
            let mut h = Sha256::new();
            h.update(b"pllm.replicated-decoder.source-q18.full-prefix-final-ids.v1");
            h.update(config);
            h.update((rows as u64).to_le_bytes());
            h.update((outputs as u64).to_le_bytes());
            h.update(serde_json::to_vec(sources).map_err(|_| Error::Binding)?);
            for (id, node) in builder.nodes.iter().enumerate() {
                for input in node.inputs() {
                    last[input] = id;
                }
                h.update((node.lanes as u64).to_le_bytes());
                h.update(node.bounds.lower().to_le_bytes());
                h.update(node.bounds.upper().to_le_bytes());
                match &node.code {
                    Code::Input(key) => {
                        h.update([0]);
                        h.update((key.len() as u64).to_le_bytes());
                        h.update(key);
                    }
                    Code::Gather(items) => {
                        h.update([1]);
                        for &(a, b) in items {
                            h.update((a as u64).to_le_bytes());
                            h.update((b as u64).to_le_bytes());
                        }
                    }
                    Code::Affine {
                        left,
                        right,
                        coefficients: c,
                    } => {
                        h.update([2]);
                        h.update((*left as u64).to_le_bytes());
                        h.update(right.map_or(u64::MAX, |v| v as u64).to_le_bytes());
                        for v in [c.left, c.right, c.constant] {
                            h.update(v.to_le_bytes());
                        }
                    }
                    Code::Product {
                        left,
                        right,
                        shape,
                        rescale,
                    } => {
                        h.update([3, u8::from(*rescale)]);
                        h.update((*left as u64).to_le_bytes());
                        h.update((*right as u64).to_le_bytes());
                        for n in shape.map_or([0, 0, 0], MatrixShape::dimensions) {
                            h.update((n as u64).to_le_bytes());
                        }
                        work += 1;
                    }
                    Code::Tensor { input, program } => {
                        h.update([4]);
                        h.update((*input as u64).to_le_bytes());
                        h.update(program.digest());
                        work += program.instructions() as u64 * 128;
                    }
                }
            }
            if work == 0 || work > puma::MAX_PROGRAM_OPERATIONS {
                return Err(Error::Bounds);
            }
            for &(id, index) in &selected {
                last[id] = builder.nodes.len();
                h.update((id as u64).to_le_bytes());
                h.update((index as u64).to_le_bytes());
            }
            let mut retire = vec![vec![]; builder.nodes.len()];
            for (id, end) in last.into_iter().enumerate() {
                if end < retire.len() {
                    retire[end].push(id);
                }
            }
            Ok(Self {
                nodes: builder.nodes,
                inputs: count,
                outputs: selected,
                retire,
                digest: h.finalize().into(),
                payload_bytes: builder.allocated,
                operation_limit: work,
            })
        })();
        result.map_err(|e| format!("replicated decoder admission: {e}"))
    }

    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
    pub(super) fn bind(&mut self, source: [u8; 32], composition: [u8; 32]) {
        let mut h = Sha256::new();
        h.update(b"pllm.replicated-decoder.source-composition.v1");
        h.update(self.digest);
        h.update(source);
        h.update(composition);
        self.digest = h.finalize().into();
    }
    pub fn payload_bytes(&self) -> usize {
        self.payload_bytes
    }
    pub fn operation_limit(&self) -> u64 {
        self.operation_limit
    }
    pub fn inputs(&self) -> impl Iterator<Item = (&str, usize)> {
        self.nodes[..self.inputs].iter().map(|n| {
            let Code::Input(key) = &n.code else {
                unreachable!()
            };
            (key.as_str(), n.lanes)
        })
    }
    pub fn setup(
        &self,
        session: [u8; 32],
        party: puma::Party,
    ) -> Result<multiply::SetupParty, Error> {
        let limits = puma::prss::PrssLimits::program(
            self.operation_limit,
            (3 * puma::MAX_LANES as u64 + 3) * self.operation_limit,
        )?;
        multiply::SetupParty::new(puma::Context::new(session, self.digest, 0)?, party, limits)
    }
}

pub(super) fn lower(
    config: &[u8],
    rows: usize,
    outputs: usize,
) -> Result<pllm_models::DecoderPlan, String> {
    if config.is_empty()
        || config.len() > 16384
        || rows == 0
        || outputs == 0
        || rows.checked_add(outputs).is_none_or(|n| n > 40)
        || outputs > 8
    {
        return Err("replicated configuration/workload bound".into());
    }
    pllm_models::lower_model_json_with_budget(
        config,
        DecoderWorkload {
            batch: 1,
            max_input_tokens: rows as u64,
            max_new_tokens: outputs as u64,
        },
        DecoderExpansionBudget {
            max_layers: 4,
            max_feature_width: 128,
            max_attention_heads: 16,
            max_vocabulary: 512,
            max_tensor_elements: MAX_VALUES as u64,
        },
    )
    .map_err(|e| e.to_string())
}
