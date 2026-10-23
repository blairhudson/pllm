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

#[path = "graph_attention_driver.rs"]
pub(super) mod attention_driver;

#[cfg(test)]
#[path = "graph_attention_tests.rs"]
mod attention_tests;

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
    /// Original borrowed owner remains live through every frontier phase. This
    /// accessor supplies its actual capacity census, not a cloned graph or an
    /// execution/source capability.
    pub(super) fn semantic_graph(&self) -> &'g DecoderGraph {
        self.graph
    }

    /// Container-only bound for a preadmitted frontier. One full conservative
    /// B-tree node per entry, including construction/split overlap; no credit for
    /// semantic retirement. Borrowed graph/source backing is NOT included.
    pub(super) fn storage_bound(graph: &DecoderGraph) -> Result<usize, Error> {
        let references = graph.operations.iter().try_fold(0usize, |n, op| {
            n.checked_add(op.inputs.len()).ok_or(Error::Bounds)
        })?;
        let entries = references
            .checked_add(graph.operations.len())
            .and_then(|n| n.checked_add(graph.state_inputs.len()))
            .ok_or(Error::Bounds)?;
        let entry = std::mem::size_of::<(String, V)>()
            .checked_add(std::mem::size_of::<(&str, usize)>())
            .and_then(|n| n.checked_add(std::mem::size_of::<usize>()))
            .and_then(|n| n.checked_mul(16))
            .and_then(|n| n.checked_add(128))
            .ok_or(Error::Bounds)?;
        let mut bytes = entries
            .checked_mul(entry)
            .and_then(|n| n.checked_add(std::mem::size_of::<Self>()))
            .ok_or(Error::Bounds)?;
        for op in &graph.operations {
            for id in std::iter::once(&op.id).chain(&op.inputs) {
                bytes = bytes
                    .checked_add(id.capacity().checked_mul(2).ok_or(Error::Bounds)?)
                    .ok_or(Error::Bounds)?;
            }
        }
        Ok(bytes)
    }

    pub(super) fn value(&self, id: &str) -> Option<&V> {
        self.values.get(id)
    }

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
                    driver::token_lookup(self, op, &shape, tokens, sources, vocabulary)?
                }
                Op::Linear | Op::OutputHead => driver::projection(self, op, &shape, a, sources)?,
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
                    attention_driver::rotary(self, op, &shape, a, position, sequence)?
                }
                Op::KvCacheAppend => {
                    attention_driver::kv_append::<Self>(op, a, b, position, sequence, keys, state)?
                }
                Op::CacheSuffix => driver::cache_suffix::<Self>(op, a)?,
                Op::AttentionScores => attention_driver::scores(
                    self,
                    op,
                    &shape,
                    a,
                    b,
                    sequence,
                    position,
                    keys,
                    (zero, 0),
                )?,
                Op::AttentionScale => attention_driver::scale(self, op, &shape, a)?,
                Op::CausalMask => a.ok_or(Error::Binding)?.clone(),
                Op::Softmax => {
                    attention_driver::softmax(self, &shape, a, sequence, position, keys, (zero, 0))?
                }
                Op::AttentionValues => {
                    attention_driver::values(self, op, &shape, a, b, sequence, position, keys)?
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

impl OrderedGraph<'_, View> {
    /// Reuse original semantic input bindings without a second graph traversal.
    /// Detached qualification only: the candidate is not an expanded View and
    /// cannot be published into this frontier or executed as a complete graph.
    #[allow(dead_code)]
    pub(super) fn compact_binary_candidate<'a>(
        &self,
        builder: &'a Builder,
        op: &pllm_models::ModelOperation,
    ) -> Result<super::compact_lowering::CompactBinary<'a>, Error> {
        let (a, b) = self.inputs(op);
        super::compact_lowering::CompactBinary::lower_unadmitted(
            builder,
            op,
            a.ok_or(Error::Binding)?,
            b.ok_or(Error::Binding)?,
        )
    }
}
