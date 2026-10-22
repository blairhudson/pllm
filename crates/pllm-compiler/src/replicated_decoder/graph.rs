//! Ordered semantic graph traversal and the original expanded operator lowering.
//!
//! Builder primitives, certificate ledgers and allocation admission remain owned
//! by `program`. Compiler-private compact publication uses this same frontier;
//! there is no compact complete-request admission or runtime dispatch here.
#[cfg(test)]
use super::super::runtime::oracle_trace;
use super::*;

#[path = "graph_driver.rs"]
pub(super) mod driver;

#[cfg(test)]
#[path = "graph_driver_tests.rs"]
mod driver_tests;

/// Storage-neutral semantic frontier. `V` is the lowering's view/handle, not an
/// expanded node or lane array. Construction counts semantic uses only; it does
/// not construct execution nodes, select a representation or admit its storage.
/// Compact lowering can reuse this order and retirement contract before expanded
/// allocation; numeric lowering below still uses the original Builder owners.
pub(super) struct OrderedGraph<'g, V> {
    graph: &'g DecoderGraph,
    uses: BTreeMap<&'g str, usize>,
    values: BTreeMap<String, V>,
}

impl<'g, V> OrderedGraph<'g, V> {
    pub(super) fn new(graph: &'g DecoderGraph) -> Self {
        let values = BTreeMap::new();
        let mut uses: BTreeMap<&str, usize> = BTreeMap::new();
        for op in &graph.operations {
            for input in &op.inputs {
                *uses.entry(input).or_default() += 1;
            }
        }
        Self {
            graph,
            uses,
            values,
        }
    }

    // The iterator borrows the immutable semantic graph, not the mutable
    // frontier, so each original operation can publish and retire its views.
    pub(super) fn operations(
        &self,
    ) -> std::iter::Enumerate<std::slice::Iter<'g, pllm_models::ModelOperation>> {
        self.graph.operations.iter().enumerate()
    }

    pub(super) fn bind(&mut self, id: String, value: V) {
        self.values.insert(id, value);
    }

    pub(super) fn inputs(&self, op: &pllm_models::ModelOperation) -> (Option<&V>, Option<&V>) {
        (
            op.inputs.first().and_then(|id| self.values.get(id)),
            op.inputs.get(1).and_then(|id| self.values.get(id)),
        )
    }

    pub(super) fn finish(
        &mut self,
        op: &pllm_models::ModelOperation,
        output: V,
    ) -> Result<(), Error> {
        self.values.insert(op.id.clone(), output);
        // Public construction references can retire once their final
        // semantic consumer has copied indices into the emitted schedule.
        // Persistent-state views have their own explicit map in the lowering.
        for input in &op.inputs {
            let count = self.uses.get_mut(input.as_str()).ok_or(Error::Binding)?;
            *count -= 1;
            if *count == 0 && input != &self.graph.output {
                self.values.remove(input);
            }
        }
        if !self.uses.contains_key(op.id.as_str()) && op.id != self.graph.output {
            self.values.remove(&op.id);
        }
        Ok(())
    }

    pub(super) fn output(mut self) -> Result<V, Error> {
        self.values.remove(&self.graph.output).ok_or(Error::Binding)
    }
}
impl Builder {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn graph(
        &mut self,
        graph: &DecoderGraph,
        tokens: &View,
        sources: &BTreeMap<String, SourceView>,
        public: &BTreeMap<String, SourceTensor>,
        vocabulary: usize,
        position: usize,
        state: &mut BTreeMap<(u64, pllm_models::StateKind), View>,
        site: &mut AdmissionSite,
    ) -> Result<View, Error> {
        let sequence = tokens.words.len();
        let keys = position.checked_add(sequence).ok_or(Error::Bounds)?;
        if sequence != graph.query_sequence as usize || keys > graph.maximum_key_sequence as usize {
            return Err(Error::Shape);
        }
        let mut traversal = OrderedGraph::<View>::new(graph);
        for input in &graph.state_inputs {
            let key = (input.layer.ok_or(Error::Binding)?, input.kind);
            traversal.bind(
                input.id.clone(),
                state.get(&key).ok_or(Error::Binding)?.clone(),
            );
        }
        traversal.bind("input.tokens".into(), tokens.clone());
        let anchor = self.gather(&tokens.words.slice(..1))?;
        let zero = self.constant(anchor, 0)?;
        for (step, op) in traversal.operations() {
            self.admission_stage = "operator_lowering";
            let mut shape: Vec<usize> = op.output_shape.iter().map(|v| *v as usize).collect();
            site.step = Some(step);
            site.operator = Some(op.operator);
            site.layer = op.layer;
            site.inputs = [None; 2];
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
            let (a, b) = traversal.inputs(op);
            for (target, view) in site.inputs.iter_mut().zip([a, b]) {
                if let Some(view) = view {
                    let bounds = self.bounds(&view.words)?;
                    *target = Some((bounds.lower(), bounds.upper()));
                }
            }
            #[cfg(test)]
            let oracle_start = self.nodes.len();
            let output = match op.operator {
                Op::TokenLookup => {
                    let table = source("weight")?;
                    if table.shape != [vocabulary, *shape.last().ok_or(Error::Shape)?] {
                        return Err(Error::Shape);
                    }
                    if self.public_weights && self.lookup == policy::LookupPolicy::Trie {
                        self.trie_lookup(tokens, table, &shape)?
                    } else if self.public_weights && self.lookup == policy::LookupPolicy::Tiled {
                        self.tiled_lookup(tokens, table, &shape)?
                    } else {
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
                                    let x =
                                        self.gather(&std::iter::repeat_n(token, n).collect())?;
                                    let mut builder =
                                        tensor::Builder::new(n, self.nodes[x].bounds)?;
                                    let coefficients: Vec<_> = (start..start + n)
                                        .map(|i| affine(1, 0, -(i as i64)))
                                        .collect();
                                    let shifted =
                                        builder.affine_lanes(builder.input(), &coefficients)?;
                                    let program = builder.finish(shifted)?;
                                    let difference = self.tensor(x, program)?;
                                    let program = tensor::Program::equal_zero(
                                        n,
                                        self.nodes[difference].bounds,
                                    )?;
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
                    self.projection_energies(
                        a.ok_or(Error::Binding)?,
                        &output,
                        source("weight")?,
                        op.attributes
                            .get("bias")
                            .filter(|v| !v.is_null())
                            .map(|_| source("bias"))
                            .transpose()?,
                    )?;
                    output
                }
                Op::RmsNorm => View {
                    words: driver::rms_norm(
                        self,
                        op,
                        &shape,
                        a.map(|view| &view.words),
                        public,
                        site,
                    )?,
                    shape: shape.clone(),
                },
                Op::Silu if self.public_weights => {
                    let input = self.gather(&a.ok_or(Error::Binding)?.words)?;
                    let lanes = self.nodes[input].lanes;
                    let bounds = self.nodes[input].bounds;
                    let output = self.tensor_loop(LoopInput::Slice(input), lanes, |n| {
                        tensor::Program::activation_guarded(tensor::Activation::Silu, n, bounds)
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
                Op::ResidualAdd => View {
                    words: self.residual(a.ok_or(Error::Binding)?, b.ok_or(Error::Binding)?)?,
                    shape: shape.clone(),
                },
                Op::Multiply => View {
                    words: self.binary(
                        &a.ok_or(Error::Binding)?.words,
                        &b.ok_or(Error::Binding)?.words,
                        true,
                    )?,
                    shape: shape.clone(),
                },
                Op::Reshape => driver::reshape::<Self>(op, &shape, a)?,
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
                        let energy = if self.public_norms.is_empty() {
                            None
                        } else {
                            self.row_norm(&row, width)
                                .or_else(|| self.attention_energy(&row, width))
                                .map(|energy| norm_bounds::rotary_energy(energy, &cos, &sin, 18))
                                .transpose()?
                        };
                        let input = self.gather(&row)?;
                        let program = tensor::Program::rotary_half(
                            1,
                            width,
                            self.nodes[input].bounds,
                            &cos,
                            &sin,
                        )?;
                        let output = self.tensor(input, program)?;
                        if let Some(energy) = energy {
                            let row = (0..width).map(|i| (output, i)).collect();
                            self.record_attention_energy(&row, energy)?;
                        }
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
                Op::CacheSuffix => driver::cache_suffix::<Self>(op, a)?,
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
                            words.extend(
                                self.matrix_with_attention(
                                    &query,
                                    MatrixView::Values(&key_rows),
                                    true,
                                    true,
                                )?
                                .words,
                            );
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
                        let (program, mass) = if self.public_weights {
                            let (program, mass) = tensor::Program::softmax_certified(
                                1,
                                width,
                                self.nodes[input].bounds,
                            )?;
                            (program, Some(mass))
                        } else {
                            (
                                tensor::Program::softmax(1, width, self.nodes[input].bounds)?,
                                None,
                            )
                        };
                        let id = self.tensor(input, program)?;
                        if let Some(mass) = mass {
                            self.admission_stage = "softmax_output_certificate";
                            self.charge_program_bytes(std::mem::size_of::<(usize, u64)>())?;
                            self.row_masses
                                .try_reserve_exact(1)
                                .map_err(|_| Error::Allocation)?;
                            self.row_masses.push((id, mass));
                        }
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
                Op::LastToken => driver::last_token::<Self>(&shape, a)?,
                Op::GreedyTokenSelection if self.public_weights => {
                    let logits = a.ok_or(Error::Binding)?;
                    let mut output = self.argmax_tree(logits)?;
                    if self.lookup == LookupPolicy::Trie {
                        self.register_selected_token(&output.words, &logits.words, vocabulary)?;
                    }
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
            #[cfg(test)]
            if oracle_trace::enabled() {
                oracle_trace::boundary(
                    site.selection,
                    &op.id,
                    &output.shape,
                    output.words.iter().collect(),
                );
                oracle_trace::span(site.selection, &op.id, oracle_start, self.nodes.len());
            }
            traversal.finish(op, output)?;
        }
        traversal.output()
    }
}
