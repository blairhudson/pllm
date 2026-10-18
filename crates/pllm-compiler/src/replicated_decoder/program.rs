use super::*;
use pllm_models::{DecoderExpansionBudget, DecoderGraph, DecoderWorkload, ModelOperator as Op};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[cfg(test)]
#[path = "linear_tests.rs"]
mod linear_tests;

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
    words: Words,
    shape: Vec<usize>,
}

struct SourceView {
    owner: usize,
    shape: Vec<usize>,
}

/// Matrix coordinates are [output column, contraction index]. Source matrices
/// carry strides instead of one (node, index) pair per checkpoint value.
enum MatrixView<'a> {
    Values(&'a View),
    Source {
        owner: usize,
        columns: usize,
        inner: usize,
        column_stride: usize,
        inner_stride: usize,
    },
}

impl<'a> MatrixView<'a> {
    fn source(source: &'a SourceView, transposed: bool) -> Result<Self, Error> {
        let [rows, columns] = source.shape.as_slice() else {
            return Err(Error::Shape);
        };
        Ok(if transposed {
            Self::Source {
                owner: source.owner,
                columns: *columns,
                inner: *rows,
                column_stride: 1,
                inner_stride: *columns,
            }
        } else {
            Self::Source {
                owner: source.owner,
                columns: *rows,
                inner: *columns,
                column_stride: *columns,
                inner_stride: 1,
            }
        })
    }

    fn dimensions(&self) -> Result<(usize, usize), Error> {
        match self {
            Self::Source { columns, inner, .. } => Ok((*columns, *inner)),
            Self::Values(view) => {
                let [columns, inner] = view.shape.as_slice() else {
                    return Err(Error::Shape);
                };
                if columns.checked_mul(*inner) != Some(view.words.len()) {
                    return Err(Error::Shape);
                }
                Ok((*columns, *inner))
            }
        }
    }

    fn gather(
        &self,
        builder: &mut Builder,
        column: usize,
        columns: usize,
        start: usize,
        terms: usize,
    ) -> Result<usize, Error> {
        let (full_columns, inner) = self.dimensions()?;
        if column.checked_add(columns).is_none_or(|n| n > full_columns)
            || start.checked_add(terms).is_none_or(|n| n > inner)
            || terms
                .checked_mul(columns)
                .is_none_or(|n| n == 0 || n > puma::MAX_LANES)
        {
            return Err(Error::Shape);
        }
        match self {
            Self::Values(view) => {
                let mut items = Words::new();
                for k in start..start + terms {
                    for j in column..column + columns {
                        items.push(view.words.at(j * inner + k));
                    }
                }
                builder.gather(&items)
            }
            Self::Source {
                owner,
                column_stride,
                inner_stride,
                ..
            } => {
                let offset = column
                    .checked_mul(*column_stride)
                    .and_then(|v| v.checked_add(start.checked_mul(*inner_stride)?))
                    .ok_or(Error::Bounds)?;
                builder.source_gather(SourceGather {
                    owner: *owner,
                    offset,
                    rows: terms,
                    columns,
                    row_stride: *inner_stride,
                    column_stride: *column_stride,
                })
            }
        }
    }
}

#[derive(Default)]
struct Builder {
    nodes: Vec<Node>,
    allocated: usize,
    tensors: BTreeMap<[u8; 32], Arc<tensor::Program>>,
    public_weights: bool,
}

fn construction_views(graph: &DecoderGraph, sources: usize) -> Result<(), String> {
    let failure = || "replicated compiler view payload limit".to_string();
    let mut last = BTreeMap::new();
    for (step, op) in graph.operations.iter().enumerate() {
        for input in &op.inputs {
            last.insert(input.as_str(), step);
        }
    }
    let mut live = BTreeMap::new();
    let mut retained = 0usize;
    for (step, op) in graph.operations.iter().enumerate() {
        let n = op
            .output_shape
            .iter()
            .try_fold(1usize, |n, width| {
                n.checked_mul(usize::try_from(*width).ok()?)
            })
            .ok_or_else(failure)?;
        if n > MAX_VALUES {
            return Err(failure());
        }
        // A maximally fragmented descriptor vector can have twice as much
        // capacity as length. Include construction copies and retained state.
        let frontier = live
            .values()
            .try_fold(retained, |a: usize, b: &usize| a.checked_add(*b))
            .and_then(|v| v.checked_add(n.checked_mul(2)?))
            .and_then(|v| v.checked_mul(64))
            .and_then(|v| v.checked_add(sources * (std::mem::size_of::<SourceView>() + 16)))
            .ok_or_else(failure)?;
        if frontier > MAX_PROGRAM_BYTES {
            return Err(failure());
        }
        if matches!(op.operator, Op::KvCacheAppend | Op::CacheSuffix) {
            retained = retained.checked_add(n).ok_or_else(failure)?;
        }
        live.insert(op.id.as_str(), n);
        live.retain(|id, _| last.get(id).is_some_and(|end| *end > step));
    }
    Ok(())
}

impl Builder {
    fn push(&mut self, code: Code, lanes: usize, bounds: Interval) -> Result<usize, Error> {
        let public_source = matches!(code, Code::PublicInput { .. });
        if lanes == 0
            || lanes
                > if public_source {
                    MAX_PUBLIC_SOURCE_VALUES
                } else {
                    MAX_VALUES
                }
            || self.nodes.len() >= MAX_NODES
        {
            return Err(Error::Bounds);
        }
        let metadata = match &code {
            Code::Gather(v) => v.metadata_bytes(),
            Code::Tensor { program, .. } if !self.tensors.contains_key(&program.digest()) => {
                // One conservative complete primitive allowance per immutable
                // program, plus the interning entry. Execution still uses fresh
                // party-local state for every invocation.
                program.admitted_bytes() + 1024
            }
            Code::TensorLoop { program, tail, .. } => {
                std::iter::once(program)
                    .chain(tail)
                    .filter(|p| !self.tensors.contains_key(&p.digest()))
                    .map(|p| p.admitted_bytes() + 1024)
                    .sum::<usize>()
                    + 4096
            }
            Code::Input(v) => v.capacity(),
            Code::PublicInput { key, .. } => key.capacity(),
            Code::PublicValues(values) => values.capacity() * 8,
            _ => 0,
        };
        self.allocated = self
            .allocated
            .checked_add(
                metadata + std::mem::size_of::<Node>() + if public_source { 0 } else { 16 * lanes },
            )
            .ok_or(Error::Bounds)?;
        if self.allocated > MAX_PROGRAM_BYTES {
            return Err(Error::Bounds);
        }
        let id = self.nodes.len();
        self.nodes.try_reserve(1).map_err(|_| Error::Allocation)?;
        if let Code::Tensor { program, .. } = &code {
            self.tensors
                .entry(program.digest())
                .or_insert_with(|| program.clone());
        }
        if let Code::TensorLoop { program, tail, .. } = &code {
            for program in std::iter::once(program).chain(tail) {
                self.tensors
                    .entry(program.digest())
                    .or_insert_with(|| program.clone());
            }
        }
        self.nodes.push(Node {
            code,
            lanes,
            bounds,
        });
        Ok(id)
    }
    fn bounds(&self, words: &Words) -> Result<Interval, Error> {
        let mut lo = i64::MAX;
        let mut hi = i64::MIN;
        for (id, index) in words.iter() {
            let node = self.nodes.get(id).ok_or(Error::Binding)?;
            if index >= node.lanes {
                return Err(Error::Shape);
            }
            lo = lo.min(node.bounds.lower());
            hi = hi.max(node.bounds.upper());
        }
        Interval::new(lo, hi)
    }
    fn gather(&mut self, words: &Words) -> Result<usize, Error> {
        let bound = self.bounds(words)?;
        self.push(Code::Gather(words.clone()), words.len(), bound)
    }
    fn source_gather(&mut self, view: SourceGather) -> Result<usize, Error> {
        let owner = self.nodes.get(view.owner).ok_or(Error::Binding)?;
        if !matches!(owner.code, Code::Input(_) | Code::PublicInput { .. }) {
            return Err(Error::Binding);
        }
        let lanes = view.validate(owner.lanes)?;
        self.push(Code::SourceGather(view), lanes, owner.bounds)
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
        let program = self
            .tensors
            .get(&program.digest())
            .cloned()
            .unwrap_or_else(|| Arc::new(program));
        self.push(Code::Tensor { input, program }, n, range)
    }
    fn view(&self, id: usize, shape: Vec<usize>) -> View {
        View {
            words: (0..self.nodes[id].lanes).map(|i| (id, i)).collect(),
            shape,
        }
    }

    fn tensor_loop(
        &mut self,
        input: LoopInput,
        lanes: usize,
        make: impl Fn(usize) -> Result<tensor::Program, Error>,
    ) -> Result<usize, Error> {
        let owner = self.nodes.get(input.owner()).ok_or(Error::Binding)?;
        if lanes == 0 || lanes > MAX_VALUES {
            return Err(Error::Bounds);
        }
        match input {
            LoopInput::Slice(_) if owner.lanes != lanes => return Err(Error::Shape),
            LoopInput::TokenOffsets { index, .. } if index >= owner.lanes => {
                return Err(Error::Shape)
            }
            _ => (),
        }
        let intern = |p: tensor::Program| {
            self.tensors
                .get(&p.digest())
                .cloned()
                .unwrap_or_else(|| Arc::new(p))
        };
        let width = lanes.min(128);
        let program = intern(make(width)?);
        let tail = if lanes % width == 0 {
            None
        } else {
            Some(intern(make(lanes % width)?))
        };
        if program.lanes() != width || tail.as_ref().is_some_and(|p| p.lanes() != lanes % width) {
            return Err(Error::Shape);
        }
        let mut bounds = program.output_bounds();
        for p in std::iter::once(&program).chain(&tail) {
            if p.output_lanes() != p.lanes() {
                return Err(Error::Shape);
            }
            bounds = Interval::new(
                bounds.lower().min(p.output_bounds().lower()),
                bounds.upper().max(p.output_bounds().upper()),
            )?;
        }
        self.push(
            Code::TensorLoop {
                input,
                program,
                tail,
            },
            lanes,
            bounds,
        )
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

    fn argmax_tree(&mut self, logits: &View) -> Result<View, Error> {
        let n = logits.words.len();
        if n == 0 {
            return Err(Error::Shape);
        }
        let indices = self.push(
            Code::PublicValues((0..n as i64).collect()),
            n,
            Interval::new(0, n as i64 - 1)?,
        )?;
        let mut values = logits.words.clone();
        let mut ids: Words = (0..n).map(|i| (indices, i)).collect();
        while values.len() > 1 {
            let mut next_values = Words::new();
            let mut next_ids = Words::new();
            for (v, ids) in values.chunks(256).zip(ids.chunks(256)) {
                let pairs = v.len() / 2;
                if pairs > 0 {
                    let left = self.gather(&(0..pairs).map(|i| v.at(2 * i)).collect())?;
                    let right = self.gather(&(0..pairs).map(|i| v.at(2 * i + 1)).collect())?;
                    let difference = self.affine(left, Some(right), affine(1, -1, 0))?;
                    let less = self.tensor(
                        difference,
                        tensor::Program::negative(pairs, self.nodes[difference].bounds)?,
                    )?;
                    let best = self.select(less, right, left)?;
                    let left_id = self.gather(&(0..pairs).map(|i| ids.at(2 * i)).collect())?;
                    let right_id = self.gather(&(0..pairs).map(|i| ids.at(2 * i + 1)).collect())?;
                    let index = self.select(less, right_id, left_id)?;
                    next_values.extend((0..pairs).map(|i| (best, i)));
                    next_ids.extend((0..pairs).map(|i| (index, i)));
                }
                if v.len() % 2 != 0 {
                    next_values.push(v.at(v.len() - 1));
                    next_ids.push(ids.at(ids.len() - 1));
                }
            }
            values = next_values;
            ids = next_ids;
        }
        Ok(View {
            words: ids,
            shape: vec![1],
        })
    }

    fn linear(&mut self, input: &View, weight: &View, rescale: bool) -> Result<View, Error> {
        self.matrix(input, MatrixView::Values(weight), rescale)
    }

    fn matrix(
        &mut self,
        input: &View,
        weight: MatrixView<'_>,
        rescale: bool,
    ) -> Result<View, Error> {
        let inner = *input.shape.last().ok_or(Error::Shape)?;
        let (columns, contraction) = weight.dimensions()?;
        if inner == 0
            || columns == 0
            || contraction != inner
            || input.words.is_empty()
            || input.words.len() % inner != 0
        {
            return Err(Error::Shape);
        }
        if let MatrixView::Source {
            owner,
            column_stride,
            inner_stride,
            ..
        } = &weight
        {
            if matches!(self.nodes[*owner].code, Code::PublicInput { .. }) {
                return self.public_matrix(
                    input,
                    *owner,
                    columns,
                    *column_stride == 1 && *inner_stride == columns,
                    rescale,
                );
            }
        }
        if inner > puma::MAX_LANES {
            return self.linear_chunks(input, &weight, rescale);
        }
        let rows = input.words.len() / inner;
        let tile = (puma::MAX_LANES / inner).min(128);
        let mut words = Words::new();
        for row in input.words.chunks_exact(inner) {
            let left = self.gather(&row)?;
            for start in (0..columns).step_by(tile) {
                let n = tile.min(columns - start);
                let right = weight.gather(self, start, n, 0, inner)?;
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

    fn public_matrix(
        &mut self,
        input: &View,
        weight: usize,
        columns: usize,
        transposed: bool,
        rescale: bool,
    ) -> Result<View, Error> {
        let inner = *input.shape.last().ok_or(Error::Shape)?;
        let rows = input.words.len() / inner;
        if (rows as u64)
            .checked_mul(inner as u64)
            .and_then(|n| n.checked_mul(columns as u64))
            .is_none_or(|n| n > 1 << 32)
        {
            return Err(Error::Bounds);
        }
        let product = InputBounds {
            left: self.bounds(&input.words)?,
            right: self.nodes[weight].bounds,
        }
        .product()?;
        let bounds = affine(i64::try_from(inner).map_err(|_| Error::Bounds)?, 0, 0).output_bounds(
            InputBounds {
                left: product,
                right: Interval::new(0, 0)?,
            },
        )?;
        let input = self.gather(&input.words)?;
        let raw = self.push(
            Code::PublicProduct {
                input,
                weight,
                rows,
                inner,
                columns,
                transposed,
            },
            rows.checked_mul(columns).ok_or(Error::Bounds)?,
            bounds,
        )?;
        let shape = vec![1, rows, columns];
        if !rescale {
            return Ok(self.view(raw, shape));
        }
        let id = self.tensor_loop(LoopInput::Slice(raw), rows * columns, |n| {
            let mut b = tensor::Builder::new(n, bounds)?;
            let scaled = b.rescale(b.input(), 18)?;
            b.finish(scaled)
        })?;
        Ok(self.view(id, shape))
    }

    /// Keep every partial dot product in ring64. Only the complete accumulated
    /// dot is truncated; rounding individual chunks would change the profile.
    /// Each private chunk issues fresh product material through the existing
    /// bounded party-local owner. No expanded primitive allowance is required.
    fn linear_chunks(
        &mut self,
        input: &View,
        weight: &MatrixView<'_>,
        rescale: bool,
    ) -> Result<View, Error> {
        let inner = *input.shape.last().ok_or(Error::Shape)?;
        let (columns, _) = weight.dimensions()?;
        let rows = input.words.len() / inner;
        let mut words = Words::new();
        for row in input.words.chunks_exact(inner) {
            for column in 0..columns {
                let mut accumulator = None;
                for start in (0..inner).step_by(puma::MAX_LANES) {
                    let count = puma::MAX_LANES.min(inner - start);
                    let left = self.gather(&row.slice(start..start + count))?;
                    let right = weight.gather(self, column, 1, start, count)?;
                    let partial =
                        self.product(left, right, Some(MatrixShape::new(1, count, 1)?), false)?;
                    accumulator = Some(match accumulator {
                        None => partial,
                        Some(previous) => self.affine(previous, Some(partial), affine(1, 1, 0))?,
                    });
                }
                let mut output = accumulator.ok_or(Error::Shape)?;
                if rescale {
                    let mut b = tensor::Builder::new(1, self.nodes[output].bounds)?;
                    let scaled = b.rescale(b.input(), 18)?;
                    output = self.tensor(output, b.finish(scaled)?)?;
                }
                words.push((output, 0));
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
        sources: &BTreeMap<String, SourceView>,
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
        let mut uses: BTreeMap<&str, usize> = BTreeMap::new();
        for op in &graph.operations {
            for input in &op.inputs {
                *uses.entry(input).or_default() += 1;
            }
        }
        for input in &graph.state_inputs {
            let key = (input.layer.ok_or(Error::Binding)?, input.kind);
            values.insert(
                input.id.clone(),
                state.get(&key).ok_or(Error::Binding)?.clone(),
            );
        }
        values.insert("input.tokens".into(), tokens.clone());
        let anchor = self.gather(&tokens.words.slice(..1))?;
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
            let source = |name: &str| -> Result<&SourceView, Error> {
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
                    let table_bound = self.nodes[table.owner].bounds;
                    let mut words = Words::new();
                    for token in tokens.words.iter() {
                        let mut one_hot = Words::new();
                        if self.public_weights {
                            let original = self.nodes[token.0].bounds;
                            let bounds = Interval::new(
                                original
                                    .lower()
                                    .checked_sub((vocabulary - 1) as i64)
                                    .ok_or(Error::Bounds)?,
                                original.upper(),
                            )?;
                            let equal = self.tensor_loop(
                                LoopInput::TokenOffsets {
                                    owner: token.0,
                                    index: token.1,
                                },
                                vocabulary,
                                |n| tensor::Program::equal_zero(n, bounds),
                            )?;
                            one_hot = self.view(equal, vec![1, vocabulary]).words;
                        } else {
                            for start in (0..vocabulary).step_by(128) {
                                let n = 128.min(vocabulary - start);
                                let x = self.gather(&std::iter::repeat_n(token, n).collect())?;
                                let mut builder = tensor::Builder::new(n, self.nodes[x].bounds)?;
                                let coefficients: Vec<_> = (start..start + n)
                                    .map(|i| affine(1, 0, -(i as i64)))
                                    .collect();
                                let shifted =
                                    builder.affine_lanes(builder.input(), &coefficients)?;
                                let program = builder.finish(shifted)?;
                                let difference = self.tensor(x, program)?;
                                let program =
                                    tensor::Program::equal_zero(n, self.nodes[difference].bounds)?;
                                let equal = self.tensor(difference, program)?;
                                one_hot.extend((0..n).map(|i| (equal, i)));
                            }
                        }
                        let selectors = View {
                            words: one_hot,
                            shape: vec![1, vocabulary],
                        };
                        let row =
                            self.matrix(&selectors, MatrixView::source(table, true)?, false)?;
                        // Source input IDs and protected argmax establish exactly
                        // one selector. The output is an original table element.
                        for (id, _) in row.words.iter() {
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
                    let mut output = self.matrix(
                        a.ok_or(Error::Binding)?,
                        MatrixView::source(source("weight")?, false)?,
                        true,
                    )?;
                    if op.attributes.get("bias").is_some_and(|v| !v.is_null()) {
                        let bias = source("bias")?;
                        if bias.shape != [*shape.last().ok_or(Error::Shape)?] {
                            return Err(Error::Shape);
                        }
                        let repeated: Words = (0..output.words.len())
                            .map(|i| (bias.owner, i % bias.shape[0]))
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
                    let mut words = Words::new();
                    for row in a.ok_or(Error::Binding)?.words.chunks_exact(width) {
                        let input = self.gather(&row)?;
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
                Op::Silu if self.public_weights => {
                    let input = self.gather(&a.ok_or(Error::Binding)?.words)?;
                    let lanes = self.nodes[input].lanes;
                    let bounds = self.nodes[input].bounds;
                    let output = self.tensor_loop(LoopInput::Slice(input), lanes, |n| {
                        tensor::Program::activation(tensor::Activation::Silu, n, bounds)
                    })?;
                    self.view(output, shape.clone())
                }
                Op::Silu => {
                    let mut words = Words::new();
                    for chunk in a.ok_or(Error::Binding)?.words.chunks(128) {
                        let input = self.gather(&chunk)?;
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
                                            .map(move |d| a.words.at((r * heads + h) * width + d))
                                    })
                                })
                                .collect()
                        }
                        Some("batch_sequence_hidden") if a.shape.len() == 4 => {
                            let (heads, rows, width) = (a.shape[1], a.shape[2], a.shape[3]);
                            (0..rows)
                                .flat_map(|r| {
                                    (0..heads).flat_map(move |h| {
                                        (0..width)
                                            .map(move |d| a.words.at((h * rows + r) * width + d))
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
                    let mut words = Words::new();
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
                        let input = self.gather(&row)?;
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
                    let mut words = Words::new();
                    for head in 0..heads {
                        if position > 0 {
                            let previous = state.get(&key).ok_or(Error::Binding)?;
                            if previous.shape != [1, heads, position, width] {
                                return Err(Error::Shape);
                            }
                            words.extend(
                                previous
                                    .words
                                    .slice(head * position * width..(head + 1) * position * width),
                            );
                        }
                        words.extend(
                            input
                                .words
                                .slice(head * sequence * width..(head + 1) * sequence * width),
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
                    let mut words = Words::new();
                    for h in 0..heads {
                        for r in 0..sequence {
                            let query = View {
                                words: q.words.slice(
                                    (h * sequence + r) * width..(h * sequence + r + 1) * width,
                                ),
                                shape: vec![1, width],
                            };
                            let key_rows = View {
                                words: k.words.slice(
                                    (h / group) * keys * width
                                        ..((h / group) * keys + position + r + 1) * width,
                                ),
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
                    let mut words = Words::new();
                    for chunk in a.ok_or(Error::Binding)?.words.chunks(128) {
                        let input = self.gather(&chunk)?;
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
                    let mut words = Words::new();
                    for (i, row) in a
                        .ok_or(Error::Binding)?
                        .words
                        .chunks_exact(keys)
                        .enumerate()
                    {
                        let width = position + i % sequence + 1;
                        let input = self.gather(&row.slice(..width))?;
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
                    let mut words = Words::new();
                    for h in 0..heads {
                        for r in 0..sequence {
                            let valid = position + r + 1;
                            let p = View {
                                words: probabilities.words.slice(
                                    (h * sequence + r) * keys..(h * sequence + r) * keys + valid,
                                ),
                                shape: vec![1, valid],
                            };
                            let mut selected = Words::new();
                            for d in 0..width {
                                for k in 0..valid {
                                    selected.push(
                                        values.words.at(((h / group) * keys + k) * width + d),
                                    );
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
                        words: a.words.slice(a.words.len() - width..),
                        shape: shape.clone(),
                    }
                }
                Op::GreedyTokenSelection if self.public_weights => {
                    let mut output = self.argmax_tree(a.ok_or(Error::Binding)?)?;
                    output.shape = shape.clone();
                    output
                }
                Op::GreedyTokenSelection => {
                    let logits = a.ok_or(Error::Binding)?;
                    let mut best = self.gather(&logits.words.slice(..1))?;
                    let mut index = self.constant(anchor, 0)?;
                    for (i, candidate) in logits.words.iter().enumerate().skip(1) {
                        let other = self.gather(&std::iter::once(candidate).collect())?;
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
            // Public construction references can retire once their final
            // semantic consumer has copied indices into the emitted schedule.
            // Persistent-state views have their own explicit map above.
            for input in &op.inputs {
                let count = uses.get_mut(input.as_str()).ok_or(Error::Binding)?;
                *count -= 1;
                if *count == 0 && input != &graph.output {
                    values.remove(input);
                }
            }
            if !uses.contains_key(op.id.as_str()) && op.id != graph.output {
                values.remove(&op.id);
            }
        }
        values.remove(&graph.output).ok_or(Error::Binding)
    }

    fn binary(&mut self, a: &Words, b: &Words, product: bool) -> Result<Words, Error> {
        if a.len() != b.len() {
            return Err(Error::Shape);
        }
        let mut words = Words::new();
        for (a, b) in a.chunks(128).zip(b.chunks(128)) {
            let left = self.gather(&a)?;
            let right = self.gather(&b)?;
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
        Self::source_shapes_with_weights(config, rows, outputs, WeightMode::Private)
    }

    pub fn source_shapes_with_weights(
        config: &[u8],
        rows: usize,
        outputs: usize,
        weights: WeightMode,
    ) -> Result<BTreeMap<String, Vec<usize>>, String> {
        let plan = lower_with_weights(config, rows, outputs, weights)?;
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
        let source_values = shapes.values().try_fold(0usize, |total, shape| {
            let n = shape
                .iter()
                .try_fold(1usize, |n, width| n.checked_mul(*width))?;
            total.checked_add(n)
        });
        if source_values.is_none_or(|n| {
            n > if weights == WeightMode::Public {
                MAX_PUBLIC_SOURCE_VALUES
            } else {
                MAX_VALUES
            }
        }) {
            return Err("replicated source payload limit".into());
        }
        // Construction views retire at last semantic use. Bound even the worst
        // uncompressible strided descriptors, vector growth and construction
        // copies at the largest live frontier. Persistent views remain charged
        // until this full-prefix graph retires. The emitted program has its own
        // independently checked MAX_PROGRAM_BYTES allowance.
        let final_plan;
        let prefill = if outputs > 1 {
            final_plan = lower_with_weights(config, rows + outputs - 1, 1, weights)?;
            &final_plan.prefill
        } else {
            &plan.prefill
        };
        construction_views(prefill, shapes.len())?;
        if weights == WeightMode::Public {
            let lookup_work = public_lookup_work_floor(rows, outputs, vocabulary)
                .map_err(|e| format!("replicated public lookup admission: {e}"))?;
            if lookup_work > puma::MAX_PROGRAM_OPERATIONS {
                return Err(format!(
                    "replicated public lookup work charge {lookup_work} exceeds limit {}",
                    puma::MAX_PROGRAM_OPERATIONS
                ));
            }
        }
        Ok(shapes)
    }

    pub fn compile(
        config: &[u8],
        rows: usize,
        outputs: usize,
        sources: &BTreeMap<String, SourceTensor>,
    ) -> Result<Self, String> {
        Self::compile_with_weights(config, rows, outputs, sources, WeightMode::Private)
    }

    pub fn compile_with_weights(
        config: &[u8],
        rows: usize,
        outputs: usize,
        sources: &BTreeMap<String, SourceTensor>,
        weights: WeightMode,
    ) -> Result<Self, String> {
        let shapes = Self::source_shapes_with_weights(config, rows, outputs, weights)?;
        let first = lower_with_weights(config, rows, outputs, weights)?;
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
                public_weights: weights == WeightMode::Public,
                ..Builder::default()
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
                    if weights == WeightMode::Public {
                        Code::PublicInput {
                            key: key.clone(),
                            digest: source.digest,
                        }
                    } else {
                        Code::Input(key.clone())
                    },
                    source.shape.iter().product(),
                    bounds,
                )?;
                inputs.insert(
                    key.clone(),
                    SourceView {
                        owner: id,
                        shape: source.shape.clone(),
                    },
                );
            }
            let first =
                lower_with_weights(config, rows, outputs, weights).map_err(|_| Error::Binding)?;
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
                let plan = lower_with_weights(config, rows + output, outputs - output, weights)
                    .map_err(|_| Error::Binding)?;
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
                selected.extend(result.words.iter());
                tokens.words.extend(result.words.iter());
                tokens.shape = vec![1, tokens.words.len()];
            }
            let mut last: Vec<_> = (0..builder.nodes.len()).collect();
            let mut work = 0u64;
            let mut h = Sha256::new();
            h.update(b"pllm.replicated-decoder.source-q18.full-prefix-final-ids.v1");
            if weights == WeightMode::Public {
                h.update(b"public-source-paged-ring64.v1");
            }
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
                    Code::PublicInput { key, digest } => {
                        h.update([5]);
                        h.update((key.len() as u64).to_le_bytes());
                        h.update(key);
                        h.update(digest);
                    }
                    Code::PublicValues(values) => {
                        h.update([7]);
                        for n in values {
                            h.update(n.to_le_bytes());
                        }
                    }
                    Code::PublicProduct {
                        input,
                        weight,
                        rows,
                        inner,
                        columns,
                        transposed,
                    } => {
                        h.update([6, u8::from(*transposed)]);
                        for n in [input, weight, rows, inner, columns] {
                            h.update((*n as u64).to_le_bytes());
                        }
                    }
                    Code::Gather(items) => {
                        h.update([1]);
                        for (a, b) in items.iter() {
                            h.update((a as u64).to_le_bytes());
                            h.update((b as u64).to_le_bytes());
                        }
                    }
                    Code::SourceGather(view) => {
                        // Hash the same logical gather as the original expanded
                        // list. Compact storage changes neither PRSS contexts nor
                        // the ordered primitive schedule.
                        h.update([1]);
                        for index in view.indices() {
                            h.update((view.owner as u64).to_le_bytes());
                            h.update((index as u64).to_le_bytes());
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
                    }
                    Code::Tensor { input, program } => {
                        h.update([4]);
                        h.update((*input as u64).to_le_bytes());
                        h.update(program.digest());
                    }
                    Code::TensorLoop {
                        input,
                        program,
                        tail,
                    } => {
                        h.update([8]);
                        match input {
                            LoopInput::Slice(id) => {
                                h.update([0]);
                                h.update((*id as u64).to_le_bytes());
                            }
                            LoopInput::TokenOffsets { owner, index } => {
                                h.update([1]);
                                h.update((*owner as u64).to_le_bytes());
                                h.update((*index as u64).to_le_bytes());
                            }
                        }
                        h.update(program.digest());
                        h.update([tail.is_some() as u8]);
                        if let Some(tail) = tail {
                            h.update(tail.digest());
                        }
                    }
                }
                work = work.checked_add(node.work_units()).ok_or(Error::Bounds)?;
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
        self.nodes[..self.inputs]
            .iter()
            .filter_map(|n| match &n.code {
                Code::Input(key) => Some((key.as_str(), n.lanes)),
                Code::PublicInput { .. } => None,
                _ => unreachable!(),
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

/// Minimum *declared charge* under the existing primitive work allowance, not a
/// timing or cryptographic-operation count. D9 pays for every complete prefix.
/// Reject known impossible work before importing any checkpoint values.
fn public_lookup_work_floor(rows: usize, outputs: usize, vocabulary: usize) -> Result<u64, Error> {
    if rows == 0 || outputs == 0 || vocabulary == 0 {
        return Err(Error::Shape);
    }
    let bounds = Interval::new(-(vocabulary as i64 - 1), vocabulary as i64 - 1)?;
    let width = vocabulary.min(128);
    let program = tensor::Program::equal_zero(width, bounds)?;
    let mut charge = (vocabulary / width) as u64 * program.instructions() as u64 * 128;
    if vocabulary % width != 0 {
        charge +=
            tensor::Program::equal_zero(vocabulary % width, bounds)?.instructions() as u64 * 128;
    }
    let prefixes = (0..outputs)
        .try_fold(0usize, |n, i| n.checked_add(rows.checked_add(i)?))
        .ok_or(Error::Bounds)?;
    charge.checked_mul(prefixes as u64).ok_or(Error::Bounds)
}

pub(super) fn lower_with_weights(
    config: &[u8],
    rows: usize,
    outputs: usize,
    weights: WeightMode,
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
            max_layers: if weights == WeightMode::Public { 64 } else { 4 },
            max_feature_width: 8192,
            max_attention_heads: 16,
            max_vocabulary: if weights == WeightMode::Public {
                262144
            } else {
                4096
            },
            max_tensor_elements: if weights == WeightMode::Public {
                MAX_PUBLIC_SOURCE_VALUES
            } else {
                MAX_VALUES
            } as u64,
        },
    )
    .map_err(|e| e.to_string())
}
