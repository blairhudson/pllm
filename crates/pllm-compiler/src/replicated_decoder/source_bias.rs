//! Creation-bound projection/bias custody. Never minted from a finalized recipe.
use super::*;
use crate::replicated_decoder::runtime::body::layer;

#[path = "source_bias_handoff.rs"]
pub(in crate::replicated_decoder) mod handoff;

/// Fixed control only; source/native backing stays with its original owners.
/// The key copy is the BODY source descriptor, not a copied registry/template.
pub(super) fn creation_bytes(
    slots: usize,
    records: &BTreeMap<String, SourceTensor>,
) -> Result<usize, Error> {
    let key = records.keys().map(String::len).max().unwrap_or(0);
    let per = add(
        add(
            size_of::<Option<rotary_creation::CreatedOperator>>(),
            rotary_creation::roster_owner_bytes(),
        )?,
        add(size_of::<layer::Source>(), add(key, bytes::<usize>(2)?)?)?,
    )?;
    add(
        slots.checked_mul(per).ok_or(Error::Bounds)?,
        add(
            bytes::<Creation>(2)?,
            add(
                size_of::<graph::ProjectionEvent<'_>>(),
                size_of::<graph::ReshapeEvent<'_>>(),
            )?,
        )?,
    )
}

/// Verification/dispatch metadata only, paid before this epoch emits operators.
/// Native projection/rescale work remains solely in the original schedule.
pub(super) fn creation_work(
    graph: &DecoderGraph,
    records: &BTreeMap<String, SourceTensor>,
) -> Result<usize, Error> {
    let key_visits = records.keys().try_fold(0, |n, key| add(n, key.len()))?;
    let mut visits = 0;
    for op in &graph.operations {
        if matches!(op.operator, Op::Linear | Op::OutputHead)
            && op.attributes.get("bias").is_some_and(|v| !v.is_null())
        {
            let lanes = op.output_shape.iter().try_fold(1usize, |n, &d| {
                n.checked_mul(usize::try_from(d).map_err(|_| Error::Bounds)?)
                    .ok_or(Error::Bounds)
            })?;
            visits = add(
                visits,
                add(
                    key_visits,
                    add(
                        graph.operations.len().checked_mul(2).ok_or(Error::Bounds)?,
                        lanes.checked_mul(24).ok_or(Error::Bounds)?,
                    )?,
                )?,
            )?;
        }
    }
    Ok(visits)
}

/// Public mapping snapshots only. Price worst-case one span per declared lane
/// BEFORE native semantic emission; no lookup/bias binding or shape guard moves.
pub(super) fn snapshot_bytes(graph: &DecoderGraph) -> Result<usize, Error> {
    let mut maximum = 0;
    let mut count = 0usize;
    for op in &graph.operations {
        let lanes = op.output_shape.iter().try_fold(1usize, |n, &d| {
            n.checked_mul(usize::try_from(d).map_err(|_| Error::Bounds)?)
                .ok_or(Error::Bounds)
        })?;
        maximum = maximum.max(lanes);
        if matches!(op.operator, Op::Linear | Op::OutputHead)
            && op.attributes.get("bias").is_some_and(|v| !v.is_null())
        {
            count += 1;
        }
    }
    Words::span_pool_bytes(maximum)
        .and_then(|n| n.checked_mul(count))
        .ok_or(Error::Bounds)
}

pub(in crate::replicated_decoder::program) fn preflight(
    builder: &Builder,
    graph: &DecoderGraph,
    op: &pllm_models::ModelOperation,
) -> Result<(), Error> {
    let Some(active) = &builder.request_emission else {
        return Ok(());
    };
    if !matches!(op.operator, Op::Linear | Op::OutputHead)
        || op.attributes.get("bias").is_none_or(|v| v.is_null())
    {
        return Ok(());
    }
    // Only capacity/state checks before the original factory. In particular no
    // bias lookup, shape validation or native factory moves ahead of matrix().
    if active.origin.payload().state.load(Ordering::Acquire) != OPEN
        || !std::ptr::eq(graph, &active.plan.plan().prefill)
        || active.projections.len() == active.projections.capacity()
    {
        return Err(Error::Bounds);
    }
    Ok(())
}

/// No Clone and no public constructor. Only `created(ProjectionEvent)` below can
/// mint it; numeric descriptions and later Program inspection cannot do so.
pub(in crate::replicated_decoder) struct Creation {
    origin: Arc<OwnedCharged<Origin>>,
    plan: Arc<OriginalSourcePlan>,
    epoch: usize,
    operation: usize,
    parent: usize,
    sticky: Option<usize>,
    input_words: Words,
    records: usize,
    registry_identity: source_registry::Instance,
    descriptor: Description,
    source: Arc<layer::Source>,
    full: Arc<tensor::Program>,
    tail: Option<Arc<tensor::Program>>,
    armed: bool,
    consumer: Option<Consumer>,
    // Installed only when removed from Program custody: no cycle while stored.
    // LAST: full/tail/source/snapshot owners erase before schedule/history charge.
    execution: Option<Arc<Program>>,
}

struct Consumer {
    operation: usize,
    next: usize,
    shape: [usize; 4],
}

/// Read-only mapping, not authority. Logical IDs never stand for Worker slots.
#[derive(Clone, Copy)]
pub(in crate::replicated_decoder) struct Description {
    pub first: usize,
    pub end: usize,
    pub weight: usize,
    pub bias: usize,
    pub geometry: [usize; 3],
    pub input_bounds: Interval,
    pub raw_bounds: Interval,
    pub floor_bounds: Interval,
    pub output_bounds: Interval,
}
impl Description {
    pub(in crate::replicated_decoder) fn lanes(self) -> usize {
        self.geometry[0] * self.geometry[2]
    }
    pub(in crate::replicated_decoder) fn bias_start(self) -> usize {
        self.first + 3
    }
    pub(in crate::replicated_decoder) fn output_word(self, lane: usize) -> (usize, usize) {
        (self.bias_start() + 3 * (lane / 128) + 2, lane % 128)
    }
}

impl Drop for Creation {
    fn drop(&mut self) {
        if self.armed {
            self.origin.payload().state.store(BURNED, Ordering::Release);
        }
    }
}

pub(in crate::replicated_decoder::program) fn created(
    builder: &mut Builder,
    event: graph::ProjectionEvent<'_>,
) -> Result<(), Error> {
    let Some(active) = &builder.request_emission else {
        return Ok(());
    };
    if active.origin.payload().state.load(Ordering::Acquire) != OPEN
        || !std::ptr::eq(event.graph(), &active.plan.plan().prefill)
        || std::ptr::from_ref(event.records()) as usize != active.source_records
        || active.projections.len() == active.projections.capacity()
    {
        return Err(Error::Binding);
    }
    let graph = event.graph();
    let op = event.operation();
    let operation = graph
        .operations
        .iter()
        .position(|x| std::ptr::eq(x, op))
        .ok_or(Error::Binding)?;
    if op.inputs.len() != 1 {
        return Err(Error::Binding);
    }
    let parent = graph.operations[..operation]
        .iter()
        .position(|x| x.id == op.inputs[0])
        .ok_or(Error::Binding)?;
    let [1, rows, columns] = event.output().shape.as_slice() else {
        return Err(Error::Shape);
    };
    let key = |attribute: &str| {
        op.attributes
            .get(attribute)
            .and_then(|v| v.as_str())
            .ok_or(Error::Binding)
    };
    let weight_key = key("weight")?;
    let bias_key = key("bias")?;
    let weight = event.sources().get(weight_key).ok_or(Error::Binding)?;
    let bias = event.sources().get(bias_key).ok_or(Error::Binding)?;
    let weight_record = event.records().get(weight_key).ok_or(Error::Binding)?;
    let bias_record = event.records().get(bias_key).ok_or(Error::Binding)?;
    if bias_record.shape != [*columns]
        || bias.shape != bias_record.shape
        || weight.shape != weight_record.shape
    {
        return Err(Error::Binding);
    }
    let first = event.start();
    let gather = builder.nodes.get(first).ok_or(Error::Binding)?;
    if !matches!(&gather.code, Code::Gather(words) if words == &event.input().words) {
        return Err(Error::Binding);
    }
    let raw = builder.nodes.get(first + 1).ok_or(Error::Binding)?;
    let Code::PublicProduct {
        input,
        weight: original_weight,
        rows: original_rows,
        inner,
        columns: original_columns,
        transposed: false,
    } = raw.code
    else {
        return Err(Error::Binding);
    };
    if input != first
        || original_weight != weight.owner
        || original_rows != *rows
        || original_columns != *columns
        || weight_record.shape != [*columns, inner]
    {
        return Err(Error::Binding);
    }
    let floor = builder.nodes.get(first + 2).ok_or(Error::Binding)?;
    let Code::TensorLoop {
        input: LoopInput::Slice(original_raw),
        program,
        tail,
    } = &floor.code
    else {
        return Err(Error::Binding);
    };
    if *original_raw != first + 1 {
        return Err(Error::Binding);
    }
    let lanes = rows.checked_mul(*columns).ok_or(Error::Bounds)?;
    let descriptor = Description {
        first,
        end: builder.nodes.len(),
        weight: weight.owner,
        bias: bias.owner,
        geometry: [*rows, inner, *columns],
        input_bounds: gather.bounds,
        raw_bounds: raw.bounds,
        floor_bounds: floor.bounds,
        output_bounds: builder.bounds(&event.output().words)?,
    };
    validate_bias(&builder.nodes, descriptor)?;
    if event.output().words.len() != lanes
        || event
            .output()
            .words
            .iter()
            .ne((0..lanes).map(|i| descriptor.output_word(i)))
    {
        return Err(Error::Binding);
    }
    let source_node = builder.nodes.get(bias.owner).ok_or(Error::Binding)?;
    let Code::PublicInput {
        key: original_key,
        digest,
        ..
    } = &source_node.code
    else {
        return Err(Error::Binding);
    };
    if original_key != bias_key
        || digest != &bias_record.digest
        || source_node.lanes != *columns
        || source_node.bounds != Interval::new(bias_record.lower, bias_record.upper)?
    {
        return Err(Error::Binding);
    }
    // Capacity/key/native-control ownership was prepaid at request declaration.
    // Source allocation remains original: this descriptor contains no file data.
    let source = layer::Source::new(
        bias.owner,
        bias_key.to_owned(),
        *digest,
        *columns,
        source_node.bounds,
    )?;
    let snapshot_bytes = Words::span_pool_bytes(event.input().words.len()).ok_or(Error::Bounds)?;
    if snapshot_bytes > active.snapshot_remaining {
        return Err(Error::Bounds);
    }
    let creation = Creation {
        origin: Arc::clone(&active.origin),
        plan: Arc::clone(&active.plan),
        epoch: active.epoch,
        operation,
        parent,
        sticky: active.sticky_in,
        input_words: event.input().words.clone(),
        records: active.source_records,
        registry_identity: active.registry_identity,
        descriptor,
        source,
        full: Arc::clone(program),
        tail: tail.as_ref().map(Arc::clone),
        armed: false,
        consumer: None,
        execution: None,
    };
    let active = builder.request_emission.as_mut().ok_or(Error::Binding)?;
    active.snapshot_remaining -= snapshot_bytes;
    active
        .projections
        .push(Some(rotary_creation::CreatedOperator::Bias(Box::new(
            creation,
        ))));
    Ok(())
}

/// Original zero-node consumer binds its real geometry while traversal still
/// owns the producer View. No retrospective label/cursor can attach a consumer.
pub(in crate::replicated_decoder::program) fn consumed(
    builder: &mut Builder,
    event: graph::ReshapeEvent<'_>,
) -> Result<(), Error> {
    let Some(active) = &mut builder.request_emission else {
        return Ok(());
    };
    if !std::ptr::eq(event.graph(), &active.plan.plan().prefill) {
        return Err(Error::Binding);
    }
    let op = event.operation();
    if op.inputs.len() != 1 {
        return Ok(());
    }
    let epoch = active.epoch;
    let Some(owner) = active
        .projections
        .iter_mut()
        .filter_map(|entry| match entry.as_mut() {
            Some(rotary_creation::CreatedOperator::Bias(owner)) => Some(owner),
            _ => None,
        })
        .find(|owner| {
            owner.epoch == epoch
                && owner.plan.plan().prefill.operations[owner.operation].id == op.inputs[0]
        })
    else {
        return Ok(());
    };
    if owner.consumer.is_some()
        || op.attributes.get("layout").and_then(|v| v.as_str())
            != Some("batch_heads_sequence_feature")
    {
        return Err(Error::Binding);
    }
    let [1, heads, rows, width] = event.output().shape.as_slice() else {
        return Err(Error::Shape);
    };
    let d = owner.descriptor;
    if *rows != d.geometry[0]
        || heads.checked_mul(*width) != Some(d.geometry[2])
        || event
            .input()
            .words
            .iter()
            .ne((0..d.lanes()).map(|i| d.output_word(i)))
        || event.output().words.iter().ne((0..*heads).flat_map(|h| {
            (0..*rows).flat_map(move |r| {
                (0..*width).map(move |i| d.output_word((r * heads + h) * width + i))
            })
        }))
    {
        return Err(Error::Binding);
    }
    let operation = event
        .graph()
        .operations
        .iter()
        .position(|x| std::ptr::eq(x, op))
        .ok_or(Error::Binding)?;
    owner.consumer = Some(Consumer {
        operation,
        next: builder.nodes.len(),
        shape: [1, *heads, *rows, *width],
    });
    Ok(())
}

fn validate_bias(nodes: &[Node], d: Description) -> Result<(), Error> {
    let lanes = d.lanes();
    if d.end
        != d.bias_start()
            .checked_add(lanes.div_ceil(128).checked_mul(3).ok_or(Error::Bounds)?)
            .ok_or(Error::Bounds)?
    {
        return Err(Error::Binding);
    }
    for offset in (0..lanes).step_by(128) {
        let first = d.bias_start() + 3 * (offset / 128);
        let len = 128.min(lanes - offset);
        for (id, source, bounds) in [
            (first, d.first + 2, d.floor_bounds),
            (first + 1, d.bias, nodes[d.bias].bounds),
        ] {
            let node = nodes.get(id).ok_or(Error::Binding)?;
            let Code::Gather(words) = &node.code else {
                return Err(Error::Binding);
            };
            if node.lanes != len
                || node.bounds != bounds
                || words.iter().ne((0..len).map(|i| {
                    (
                        source,
                        if id == first {
                            offset + i
                        } else {
                            (offset + i) % d.geometry[2]
                        },
                    )
                }))
            {
                return Err(Error::Binding);
            }
        }
        let node = nodes.get(first + 2).ok_or(Error::Binding)?;
        if !matches!(node.code, Code::Affine { left, right: Some(right), coefficients }
            if left == first && right == first + 1 && (coefficients.left, coefficients.right, coefficients.constant) == (1,1,0))
            || node.lanes != len
            || node.bounds != d.output_bounds
            || node.work_budget()? != WorkBudget::default()
        {
            return Err(Error::Binding);
        }
    }
    Ok(())
}

impl Creation {
    #[cfg(test)]
    pub(in crate::replicated_decoder) fn assert_foreign_bindings_rejected(
        &mut self,
        program: &Program,
        registry: &SourceRegistry<'_>,
    ) {
        self.check(program, registry).unwrap();
        let parent = self.parent;
        self.parent = self.operation;
        assert!(matches!(self.check(program, registry), Err(Error::Binding)));
        self.parent = parent;
        let epoch = self.epoch;
        self.epoch += 1;
        assert!(matches!(self.check(program, registry), Err(Error::Binding)));
        self.epoch = epoch;
        let shape = self.descriptor.geometry;
        self.descriptor.geometry[2] += 1;
        assert!(matches!(self.check(program, registry), Err(Error::Binding)));
        self.descriptor.geometry = shape;
        let words = self.input_words.clone();
        self.input_words = self
            .input_words
            .iter()
            .map(|(id, lane)| (id, lane + 1))
            .collect();
        assert!(matches!(self.check(program, registry), Err(Error::Binding)));
        self.input_words = words;
        let raw_bounds = self.descriptor.raw_bounds;
        let foreign = |original: &Arc<tensor::Program>| {
            let mut b = tensor::Builder::new(original.lanes(), raw_bounds).unwrap();
            let output = b.rescale(b.input(), 18).unwrap();
            let replacement = Arc::new(b.finish(output).unwrap());
            assert_eq!(replacement.digest(), original.digest());
            assert!(!Arc::ptr_eq(&replacement, original));
            replacement
        };
        let full = Arc::clone(&self.full);
        self.full = foreign(&full);
        assert!(matches!(self.check(program, registry), Err(Error::Binding)));
        self.full = full;
        if let Some(tail) = self.tail.as_ref().map(Arc::clone) {
            self.tail = Some(foreign(&tail));
            assert!(matches!(self.check(program, registry), Err(Error::Binding)));
            self.tail = Some(tail);
        }
        self.check(program, registry).unwrap();
    }
    pub(in crate::replicated_decoder) fn description(&self) -> Description {
        self.descriptor
    }
    pub(in crate::replicated_decoder) fn source(&self) -> &Arc<layer::Source> {
        &self.source
    }
    pub(in crate::replicated_decoder) fn public(&self) -> &Arc<PublicSource> {
        &self.origin.payload().public
    }
    pub(in crate::replicated_decoder) fn published(&mut self) {
        self.armed = false;
    }
    pub(in crate::replicated_decoder) fn consumer_start(&self) -> Result<usize, Error> {
        Ok(self.consumer.as_ref().ok_or(Error::Binding)?.next)
    }
    pub(in crate::replicated_decoder) fn check(
        &self,
        program: &Program,
        registry: &SourceRegistry<'_>,
    ) -> Result<(), Error> {
        let custody = program
            .request_custody
            .as_ref()
            .ok_or(Error::Binding)?
            .expanded()?;
        if self
            .execution
            .as_ref()
            .is_none_or(|original| !std::ptr::eq(original.as_ref(), program))
            || !Arc::ptr_eq(&self.origin, &custody.origin)
            || self.origin.payload().state.load(Ordering::Acquire) != COMPLETE
            || !Arc::ptr_eq(
                &self.plan,
                custody.epochs.get(self.epoch).ok_or(Error::Binding)?,
            )
            || std::ptr::from_ref(registry.source_records()) as usize != self.records
            || registry.instance() != self.registry_identity
            || custody
                .fragments
                .get(self.epoch)
                .ok_or(Error::Binding)?
                .sticky_in
                != self.sticky
        {
            return Err(Error::Binding);
        }
        let op = self
            .plan
            .plan()
            .prefill
            .operations
            .get(self.operation)
            .ok_or(Error::Binding)?;
        let parent = self
            .plan
            .plan()
            .prefill
            .operations
            .get(self.parent)
            .ok_or(Error::Binding)?;
        if op.inputs.len() != 1 || op.inputs[0] != parent.id {
            return Err(Error::Binding);
        }
        let consumer = self.consumer.as_ref().ok_or(Error::Binding)?;
        let consumer_op = self
            .plan
            .plan()
            .prefill
            .operations
            .get(consumer.operation)
            .ok_or(Error::Binding)?;
        if consumer_op.operator != Op::Reshape
            || consumer_op.inputs.len() != 1
            || consumer_op.inputs[0] != op.id
            || consumer_op
                .output_shape
                .iter()
                .copied()
                .ne(consumer.shape.map(|x| x as u64))
        {
            return Err(Error::Binding);
        }
        self.public().check_registry(registry)?;
        self.public().check(program)?;
        let d = self.descriptor;
        let gather = program.recipe_at(d.first).ok_or(Error::Binding)?;
        if !matches!(&gather.code, Code::Gather(words) if words == &self.input_words)
            || gather.bounds != d.input_bounds
            || op
                .output_shape
                .iter()
                .copied()
                .ne([1, d.geometry[0] as u64, d.geometry[2] as u64])
        {
            return Err(Error::Binding);
        }
        if !matches!(program.recipe_at(d.first+1).ok_or(Error::Binding)?.code,
            Code::PublicProduct { input,weight,rows,inner,columns,transposed:false }
            if input==d.first && weight==d.weight && [rows,inner,columns]==d.geometry)
        {
            return Err(Error::Binding);
        }
        if program.recipe(d.first + 1).bounds != d.raw_bounds
            || program.recipe(d.first + 2).bounds != d.floor_bounds
            || program.recipe(d.first + 2).lanes != d.lanes()
        {
            return Err(Error::Binding);
        }
        for (id, attribute) in [(d.weight, "weight"), (d.bias, "bias")] {
            let node = program.recipe_at(id).ok_or(Error::Binding)?;
            let Code::PublicInput { key, digest, .. } = &node.code else {
                return Err(Error::Binding);
            };
            if Some(key.as_str()) != op.attributes.get(attribute).and_then(|v| v.as_str()) {
                return Err(Error::Binding);
            }
            registry.check_descriptor(id, key, *digest, node.lanes, node.bounds)?;
        }
        let Code::TensorLoop {
            program: full,
            tail,
            input: LoopInput::Slice(raw),
        } = &program.recipe(d.first + 2).code
        else {
            return Err(Error::Binding);
        };
        if *raw != d.first + 1
            || !Arc::ptr_eq(&self.full, full)
            || match (&self.tail, tail) {
                (None, None) => false,
                (Some(a), Some(b)) => !Arc::ptr_eq(a, b),
                _ => true,
            }
        {
            return Err(Error::Binding);
        }
        validate_bias(&program.nodes, d)
    }
}

impl Program {
    /// Consuming unique creation publication, before any Worker shares Program.
    /// No operation label, coordinate or digest can mint a replacement owner.
    pub(in crate::replicated_decoder) fn take_source_bias(
        self: &mut Arc<Self>,
        index: usize,
    ) -> Result<Creation, Error> {
        if self
            .request_custody
            .as_ref()
            .ok_or(Error::Binding)?
            .expanded()?
            .projections
            .get(index)
            .and_then(|entry| match entry.as_ref() {
                Some(rotary_creation::CreatedOperator::Bias(owner)) => Some(owner),
                _ => None,
            })
            .is_none()
        {
            return Err(Error::Retired);
        }
        let program = Arc::get_mut(self).ok_or(Error::Binding)?;
        let custody = program
            .request_custody
            .as_mut()
            .ok_or(Error::Binding)?
            .expanded_mut()?;
        let owner = custody
            .projections
            .get_mut(index)
            .and_then(Option::take)
            .ok_or(Error::Retired)?;
        let rotary_creation::CreatedOperator::Bias(owner) = owner else {
            return Err(Error::Binding);
        };
        let mut owner = *owner;
        owner.armed = true;
        owner.execution = Some(Arc::clone(self));
        Ok(owner)
    }
}
