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
/// Constructors remain INSIDE original graph emission, never from arbitrary
/// caller Views. Request history receives these events only at genuine sites.
pub(super) struct SelectionEvent<'a> {
    graph: &'a DecoderGraph,
    operation: &'a pllm_models::ModelOperation,
    view: &'a View,
}
pub(super) struct FeedbackEvent<'a> {
    graph: &'a DecoderGraph,
    operation: &'a pllm_models::ModelOperation,
    view: &'a View,
}
/// Issued only after the actual original projection, bias and proof propagation.
/// Coordinates describe this emission; they cannot construct an event elsewhere.
pub(super) struct ProjectionEvent<'a> {
    graph: &'a DecoderGraph,
    operation: &'a pllm_models::ModelOperation,
    input: &'a View,
    output: &'a View,
    sources: &'a BTreeMap<String, SourceView>,
    records: &'a BTreeMap<String, SourceTensor>,
    start: usize,
}
/// Exact live semantic publication. Constructor remains in original graph loop.
pub(super) struct ProducedEvent<'a> {
    graph: &'a DecoderGraph,
    operation: &'a pllm_models::ModelOperation,
    input: Option<&'a View>,
    output: &'a View,
    start: usize,
}
impl<'a> ProducedEvent<'a> {
    pub(super) fn graph(&self) -> &'a DecoderGraph {
        self.graph
    }
    pub(super) fn operation(&self) -> &'a pllm_models::ModelOperation {
        self.operation
    }
    pub(super) fn input(&self) -> Option<&'a View> {
        self.input
    }
    pub(super) fn output(&self) -> &'a View {
        self.output
    }
    pub(super) fn start(&self) -> usize {
        self.start
    }
}
pub(super) struct RotaryEvent<'a> {
    graph: &'a DecoderGraph,
    operation: &'a pllm_models::ModelOperation,
    input: &'a View,
    shape: &'a [usize],
    position: usize,
    sequence: usize,
}
impl<'a> RotaryEvent<'a> {
    pub(super) fn graph(&self) -> &'a DecoderGraph {
        self.graph
    }
    pub(super) fn operation(&self) -> &'a pllm_models::ModelOperation {
        self.operation
    }
    pub(super) fn input(&self) -> &'a View {
        self.input
    }
    pub(super) fn shape(&self) -> &'a [usize] {
        self.shape
    }
    pub(super) fn position(&self) -> usize {
        self.position
    }
    pub(super) fn sequence(&self) -> usize {
        self.sequence
    }
}
pub(super) struct ReshapeEvent<'a> {
    graph: &'a DecoderGraph,
    operation: &'a pllm_models::ModelOperation,
    input: &'a View,
    output: &'a View,
}
impl<'a> ReshapeEvent<'a> {
    pub(super) fn graph(&self) -> &'a DecoderGraph {
        self.graph
    }
    pub(super) fn operation(&self) -> &'a pllm_models::ModelOperation {
        self.operation
    }
    pub(super) fn input(&self) -> &'a View {
        self.input
    }
    pub(super) fn output(&self) -> &'a View {
        self.output
    }
}
impl<'a> ProjectionEvent<'a> {
    pub(super) fn graph(&self) -> &'a DecoderGraph {
        self.graph
    }
    pub(super) fn operation(&self) -> &'a pllm_models::ModelOperation {
        self.operation
    }
    pub(super) fn input(&self) -> &'a View {
        self.input
    }
    pub(super) fn output(&self) -> &'a View {
        self.output
    }
    pub(super) fn sources(&self) -> &'a BTreeMap<String, SourceView> {
        self.sources
    }
    pub(super) fn records(&self) -> &'a BTreeMap<String, SourceTensor> {
        self.records
    }
    pub(super) fn start(&self) -> usize {
        self.start
    }
}
impl<'a> SelectionEvent<'a> {
    pub(super) fn graph(&self) -> &'a DecoderGraph {
        self.graph
    }
    pub(super) fn operation(&self) -> &'a pllm_models::ModelOperation {
        self.operation
    }
    pub(super) fn view(&self) -> &'a View {
        self.view
    }
}
impl<'a> FeedbackEvent<'a> {
    pub(super) fn graph(&self) -> &'a DecoderGraph {
        self.graph
    }
    pub(super) fn operation(&self) -> &'a pllm_models::ModelOperation {
        self.operation
    }
    pub(super) fn view(&self) -> &'a View {
        self.view
    }
}

/// One semantic frontier for every storage representation. This traversal has
/// no source, proof or completion authority of its own; those stay with its
/// participating sink. Begin occurs AFTER original state/input binding and
/// BEFORE the first original gather/constant, preserving rejection order.
pub(super) enum DispatchStep<'g, 'v, V> {
    Begin,
    Operator {
        step: usize,
        operation: &'g pllm_models::ModelOperation,
        left: Option<&'v V>,
        right: Option<&'v V>,
    },
}

pub(super) fn dispatch_begin<V: Clone>(
    traversal: &mut OrderedGraph<'_, V>,
    tokens: &V,
    state: &mut BTreeMap<(u64, pllm_models::StateKind), V>,
    mut sink: impl FnMut(
        DispatchStep<'_, '_, V>,
        &mut BTreeMap<(u64, pllm_models::StateKind), V>,
    ) -> Result<Option<V>, Error>,
) -> Result<(), Error> {
    for input in &traversal.semantic_graph().state_inputs {
        let key = (input.layer.ok_or(Error::Binding)?, input.kind);
        traversal.bind(
            input.id.clone(),
            state.get(&key).ok_or(Error::Binding)?.clone(),
        );
    }
    traversal.bind("input.tokens".into(), tokens.clone());
    if sink(DispatchStep::Begin, state)?.is_some() {
        return Err(Error::Binding);
    }
    Ok(())
}

pub(super) fn dispatch_operator<'g, 'v, V>(
    traversal: &'v OrderedGraph<'g, V>,
    step: usize,
) -> Result<DispatchStep<'g, 'v, V>, Error> {
    let (_, operation) = traversal.operations().nth(step).ok_or(Error::Binding)?;
    let (left, right) = traversal.inputs(operation);
    Ok(DispatchStep::Operator {
        step,
        operation,
        left,
        right,
    })
}

pub(super) fn dispatch<V: Clone>(
    graph: &DecoderGraph,
    tokens: &V,
    state: &mut BTreeMap<(u64, pllm_models::StateKind), V>,
    mut sink: impl FnMut(
        DispatchStep<'_, '_, V>,
        &mut BTreeMap<(u64, pllm_models::StateKind), V>,
    ) -> Result<Option<V>, Error>,
) -> Result<V, Error> {
    let mut traversal = OrderedGraph::<V>::new(graph);
    dispatch_begin(&mut traversal, tokens, state, &mut sink)?;
    for (step, operation) in traversal.operations() {
        let output = sink(dispatch_operator(&traversal, step)?, state)?.ok_or(Error::Binding)?;
        traversal.finish(operation, output)?;
    }
    traversal.output()
}

/// Original execution geometry: semantic cache declarations carry capacity;
/// actual native consumers use the live key count. Storage sinks share this
/// adjustment, not a second decoder shape convention.
pub(super) fn actual_key_shape(
    operator: Op,
    keys: usize,
    shape: &mut [usize],
) -> Result<(), Error> {
    if matches!(operator, Op::KvCacheAppend | Op::CacheSuffix) {
        shape[2] = keys;
    }
    if matches!(
        operator,
        Op::AttentionScores | Op::AttentionScale | Op::CausalMask | Op::Softmax
    ) {
        *shape.last_mut().ok_or(Error::Shape)? = keys;
    }
    Ok(())
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
        request_origin::graph_entry(self, graph, tokens)?;
        let sequence = tokens.words.len();
        let keys = position.checked_add(sequence).ok_or(Error::Bounds)?;
        if sequence != graph.query_sequence as usize || keys > graph.maximum_key_sequence as usize {
            return Err(Error::Shape);
        }
        let (mut anchor, mut zero) = (None, None);
        dispatch(graph, tokens, state, |dispatch, state| {
            let DispatchStep::Operator {
                step,
                operation: op,
                left: a,
                right: b,
            } = dispatch
            else {
                let input = self.gather(&tokens.words.slice(..1))?;
                anchor = Some(input);
                zero = Some(self.constant(input, 0)?);
                return Ok(None);
            };
            let anchor = anchor.ok_or(Error::Binding)?;
            let zero = zero.ok_or(Error::Binding)?;
            self.admission_stage = "operator_lowering";
            let mut shape: Vec<usize> = op.output_shape.iter().map(|v| *v as usize).collect();
            site.step = Some(step);
            site.operator = Some(op.operator);
            site.layer = op.layer;
            site.inputs = [None; 2];
            actual_key_shape(op.operator, keys, &mut shape)?;
            for (target, view) in site.inputs.iter_mut().zip([a, b]) {
                if let Some(view) = view {
                    let bounds = self.bounds(&view.words)?;
                    *target = Some((bounds.lower(), bounds.upper()));
                }
            }
            let emission_start = self.nodes.len();
            request_origin::source_bias::preflight(self, graph, op)?;
            request_origin::rotary_creation::preflight(self, graph, op, &shape)?;
            let output = match op.operator {
                Op::GeluErf | Op::Tanh | Op::AttentionMask => return Err(Error::Binding),
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
            if matches!(op.operator, Op::Linear | Op::OutputHead)
                && op.attributes.get("bias").is_some_and(|v| !v.is_null())
            {
                request_origin::source_bias::created(
                    self,
                    ProjectionEvent {
                        graph,
                        operation: op,
                        input: a.ok_or(Error::Binding)?,
                        output: &output,
                        sources,
                        records: public,
                        start: emission_start,
                    },
                )?;
            }
            if op.operator == Op::GreedyTokenSelection {
                request_origin::selected(
                    self,
                    SelectionEvent {
                        graph,
                        operation: op,
                        view: &output,
                    },
                )?;
            } else if op.operator == Op::TokenFeedback {
                request_origin::feedback(
                    self,
                    FeedbackEvent {
                        graph,
                        operation: op,
                        view: &output,
                    },
                )?;
            }
            if op.operator == Op::Reshape {
                request_origin::source_bias::consumed(
                    self,
                    ReshapeEvent {
                        graph,
                        operation: op,
                        input: a.ok_or(Error::Binding)?,
                        output: &output,
                    },
                )?;
            }
            let event = || ProducedEvent {
                graph,
                operation: op,
                input: a,
                output: &output,
                start: emission_start,
            };
            if matches!(op.operator, Op::Linear | Op::Reshape) {
                request_origin::rotary_creation::produced(self, event())?;
            }
            if op.operator == Op::RotaryEmbedding {
                request_origin::rotary_creation::completed(self, event())?;
            }
            if matches!(op.operator, Op::KvCacheAppend | Op::AttentionScores) {
                request_origin::rotary_creation::consumed(self, event())?;
            }
            #[cfg(test)]
            if oracle_trace::enabled() {
                oracle_trace::boundary(
                    site.selection,
                    &op.id,
                    &output.shape,
                    output.words.iter().collect(),
                );
                oracle_trace::span(site.selection, &op.id, emission_start, self.nodes.len());
            }
            Ok(Some(output))
        })
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
