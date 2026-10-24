//! Original attention/rotary/KV composition over storage-neutral views.
//! Native Arc ownership, source/certificate authority and publication charging
//! stay in the concrete adapters. Expanded adapter retains original Words/View
//! buffers; sink calls may allocate. No generic callback, compiled callframe or
//! physical-memory bound follows from this traversal interface.
use super::*;

pub(in crate::replicated_decoder) trait AttentionSink {
    type View: Clone;
    type Words: Default;
    type Word: Copy;
    type Owner: Copy;
    fn shape(view: &Self::View) -> &[usize];
    fn words(view: &Self::View) -> &Self::Words;
    fn len(words: &Self::Words) -> usize;
    fn slice(words: &Self::Words, range: std::ops::Range<usize>) -> Self::Words;
    fn rows(words: &Self::Words, width: usize) -> impl Iterator<Item = Self::Words> + '_;
    fn chunks(words: &Self::Words, width: usize) -> impl Iterator<Item = Self::Words> + '_;
    fn at(words: &Self::Words, index: usize) -> Self::Word;
    fn push(words: &mut Self::Words, word: Self::Word);
    fn extend(words: &mut Self::Words, more: Self::Words);
    fn repeat(words: &mut Self::Words, word: Self::Word, count: usize);
    fn append_owner(words: &mut Self::Words, owner: Self::Owner, lanes: usize);
    fn view(words: Self::Words, shape: &[usize]) -> Self::View;
    fn begin_rotary(
        &mut self,
        _op: &pllm_models::ModelOperation,
        _shape: &[usize],
        _input: &Self::View,
        _position: usize,
        _sequence: usize,
        _theta: f32,
    ) -> Result<(), Error> {
        Ok(())
    }
    fn rotary_row(
        &mut self,
        row: &Self::Words,
        theta: f32,
        position: u32,
        width: usize,
    ) -> Result<Self::Owner, Error>;
    fn scores(&mut self, query: &Self::View, keys: &Self::View) -> Result<Self::Words, Error>;
    fn values(
        &mut self,
        probabilities: &Self::View,
        values: &Self::View,
    ) -> Result<Self::Words, Error>;
    fn scale_coefficient(head_dim: u64) -> i64;
    fn scale_declaration(op: &pllm_models::ModelOperation) -> Result<ScaleDeclaration, Error> {
        Ok(ScaleDeclaration {
            coefficient: Self::scale_coefficient(scale_width(op)?),
        })
    }
    fn scale_chunk(&mut self, chunk: &Self::Words, coefficient: i64) -> Result<Self::Owner, Error>;
    fn softmax_row(&mut self, row: &Self::Words, width: usize) -> Result<Self::Owner, Error>;
}

/// Shared original rotary traversal, independent of Words/View storage. The
/// adapter below preserves expanded behavior; compact emission supplies mapped
/// rows and owning native recipes, never an expanded intermediate Builder.
pub(in crate::replicated_decoder) trait RotarySink {
    type View;
    type Row;
    type Owner;
    type Output: Default;
    type Produced;
    fn begin(
        &mut self,
        op: &pllm_models::ModelOperation,
        shape: &[usize],
        input: &Self::View,
        position: usize,
        sequence: usize,
        theta: f32,
    ) -> Result<(), Error>;
    fn rows<'a>(input: &'a Self::View, width: usize) -> impl Iterator<Item = Self::Row> + 'a
    where
        Self::Row: 'a;
    fn emit(
        &mut self,
        row: &Self::Row,
        theta: f32,
        position: u32,
        width: usize,
    ) -> Result<Self::Owner, Error>;
    fn append(output: &mut Self::Output, owner: Self::Owner, width: usize);
    fn finish(&mut self, output: Self::Output, shape: &[usize]) -> Result<Self::Produced, Error>;
}
pub(in crate::replicated_decoder) fn rotary_geometry(
    op: &pllm_models::ModelOperation,
    shape: &[usize],
) -> Result<(usize, f32), Error> {
    if shape.len() != 4
        || shape[3] > 128
        || op.attributes.get("frequency_scaling").is_some()
        || op.attributes.get("pairing").and_then(|v| v.as_str()) != Some("split_half")
    {
        return Err(Error::Binding);
    }
    let theta = op
        .attributes
        .get("theta")
        .and_then(|v| v.as_u64())
        .ok_or(Error::Binding)? as f32;
    Ok((shape[3], theta))
}
pub(in crate::replicated_decoder) fn emit_rotary<S: RotarySink>(
    sink: &mut S,
    op: &pllm_models::ModelOperation,
    shape: &[usize],
    input: Option<&S::View>,
    position: usize,
    sequence: usize,
) -> Result<S::Produced, Error> {
    let (width, theta) = rotary_geometry(op, shape)?;
    let input = input.ok_or(Error::Binding)?;
    sink.begin(op, shape, input, position, sequence, theta)?;
    let mut output = S::Output::default();
    for (index, row) in S::rows(input, width).enumerate() {
        let owner = sink.emit(&row, theta, (position + index % sequence) as u32, width)?;
        S::append(&mut output, owner, width);
    }
    sink.finish(output, shape)
}
struct OriginalRotary<'a, S>(&'a mut S);
impl<S: AttentionSink> RotarySink for OriginalRotary<'_, S> {
    type View = S::View;
    type Row = S::Words;
    type Owner = S::Owner;
    type Output = S::Words;
    type Produced = S::View;
    fn begin(
        &mut self,
        op: &pllm_models::ModelOperation,
        shape: &[usize],
        input: &Self::View,
        position: usize,
        sequence: usize,
        theta: f32,
    ) -> Result<(), Error> {
        self.0
            .begin_rotary(op, shape, input, position, sequence, theta)
    }
    fn rows<'a>(input: &'a Self::View, width: usize) -> impl Iterator<Item = Self::Row> + 'a
    where
        Self::Row: 'a,
    {
        S::rows(S::words(input), width)
    }
    fn emit(
        &mut self,
        row: &Self::Row,
        theta: f32,
        position: u32,
        width: usize,
    ) -> Result<Self::Owner, Error> {
        self.0.rotary_row(row, theta, position, width)
    }
    fn append(output: &mut Self::Output, owner: Self::Owner, width: usize) {
        S::append_owner(output, owner, width)
    }
    fn finish(&mut self, output: Self::Output, shape: &[usize]) -> Result<Self::Produced, Error> {
        Ok(S::view(output, shape))
    }
}

pub(super) fn rotary<S: AttentionSink>(
    sink: &mut S,
    op: &pllm_models::ModelOperation,
    shape: &[usize],
    input: Option<&S::View>,
    position: usize,
    sequence: usize,
) -> Result<S::View, Error> {
    emit_rotary(
        &mut OriginalRotary(sink),
        op,
        shape,
        input,
        position,
        sequence,
    )
}

/// Original cache declaration, shared by expanded and exact-map alias sinks.
/// Attributes alone confer no request/source/cache authority.
pub(in crate::replicated_decoder) fn cache_declaration(
    op: &pllm_models::ModelOperation,
) -> Result<((u64, pllm_models::StateKind), &str), Error> {
    let key = (
        op.layer.ok_or(Error::Binding)?,
        op.state_kind.ok_or(Error::Binding)?,
    );
    let mode = op
        .attributes
        .get("mode")
        .and_then(|v| v.as_str())
        .ok_or(Error::Binding)?;
    Ok((key, mode))
}
pub(super) fn kv_append<S: AttentionSink>(
    op: &pllm_models::ModelOperation,
    a: Option<&S::View>,
    b: Option<&S::View>,
    position: usize,
    sequence: usize,
    keys: usize,
    state: &mut BTreeMap<(u64, pllm_models::StateKind), S::View>,
) -> Result<S::View, Error> {
    let (key, mode) = cache_declaration(op)?;
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
    let (heads, width) = (S::shape(input)[1], S::shape(input)[3]);
    let mut words = S::Words::default();
    for head in 0..heads {
        if position > 0 {
            let previous = state.get(&key).ok_or(Error::Binding)?;
            if S::shape(previous) != [1, heads, position, width] {
                return Err(Error::Shape);
            }
            S::extend(
                &mut words,
                S::slice(
                    S::words(previous),
                    head * position * width..(head + 1) * position * width,
                ),
            );
        }
        S::extend(
            &mut words,
            S::slice(
                S::words(input),
                head * sequence * width..(head + 1) * sequence * width,
            ),
        );
    }
    let out = S::view(words, &[1, heads, keys, width]);
    state.insert(key, out.clone());
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub(in crate::replicated_decoder) fn scores<S: AttentionSink>(
    sink: &mut S,
    op: &pllm_models::ModelOperation,
    shape: &[usize],
    a: Option<&S::View>,
    b: Option<&S::View>,
    sequence: usize,
    position: usize,
    keys: usize,
    zero: S::Word,
) -> Result<S::View, Error> {
    let q = a.ok_or(Error::Binding)?;
    let k = b.ok_or(Error::Binding)?;
    let (heads, width) = (S::shape(q)[1], S::shape(q)[3]);
    let group = op
        .attributes
        .get("group_size")
        .and_then(|v| v.as_u64())
        .ok_or(Error::Binding)? as usize;
    if group == 0 || heads / group != S::shape(k)[1] {
        return Err(Error::Shape);
    }
    let mut words = S::Words::default();
    for h in 0..heads {
        for r in 0..sequence {
            let query = S::view(
                S::slice(
                    S::words(q),
                    (h * sequence + r) * width..(h * sequence + r + 1) * width,
                ),
                &[1, width],
            );
            let key_rows = S::view(
                S::slice(
                    S::words(k),
                    (h / group) * keys * width..((h / group) * keys + position + r + 1) * width,
                ),
                &[position + r + 1, width],
            );
            S::extend(&mut words, sink.scores(&query, &key_rows)?);
            S::repeat(&mut words, zero, sequence - r - 1);
        }
    }
    Ok(S::view(words, shape))
}

/// Captured original binary32 scale coefficient. This is numeric metadata, not
/// source, row, GENERAL, validity, mass or resource authority. Its constructor
/// stays here; callers reuse this same declaration across preparation/emission.
pub(in crate::replicated_decoder) struct ScaleDeclaration {
    coefficient: i64,
}

impl ScaleDeclaration {
    pub(in crate::replicated_decoder) fn coefficient(&self) -> i64 {
        self.coefficient
    }
}

fn scale_width(op: &pllm_models::ModelOperation) -> Result<u64, Error> {
    op.attributes
        .get("head_dim")
        .and_then(|v| v.as_u64())
        .ok_or(Error::Binding)
}

pub(in crate::replicated_decoder) fn scale_declaration(
    op: &pllm_models::ModelOperation,
) -> Result<ScaleDeclaration, Error> {
    Ok(ScaleDeclaration {
        coefficient: <Builder as AttentionSink>::scale_coefficient(scale_width(op)?),
    })
}

/// Scale-only original chunk traversal. Preparation/emission may use different
/// concrete sinks but never another chunk geometry or coefficient expression.
/// All phase/ticket admission and original native ownership stay in the sink.
pub(in crate::replicated_decoder) trait ScaleSink {
    type View;
    type Chunk;
    type Owner;
    type Output: Default;
    type Produced;
    fn begin(
        &mut self,
        input: &Self::View,
        shape: &[usize],
        declaration: &ScaleDeclaration,
    ) -> Result<(), Error>;
    fn chunks<'a>(input: &'a Self::View, width: usize) -> impl Iterator<Item = Self::Chunk> + 'a
    where
        Self::Chunk: 'a;
    fn len(chunk: &Self::Chunk) -> usize;
    fn native_chunk(
        &mut self,
        chunk: Self::Chunk,
        declaration: &ScaleDeclaration,
    ) -> Result<Self::Owner, Error>;
    fn append(output: &mut Self::Output, owner: Self::Owner, lanes: usize);
    fn finish(&mut self, output: Self::Output, shape: &[usize]) -> Result<Self::Produced, Error>;
}

pub(in crate::replicated_decoder) fn scale_chunks<S: ScaleSink>(
    sink: &mut S,
    declaration: &ScaleDeclaration,
    shape: &[usize],
    input: Option<&S::View>,
) -> Result<S::Produced, Error> {
    let input = input.ok_or(Error::Binding)?;
    sink.begin(input, shape, declaration)?;
    let mut output = S::Output::default();
    for chunk in S::chunks(input, 128) {
        let lanes = S::len(&chunk);
        let owner = sink.native_chunk(chunk, declaration)?;
        S::append(&mut output, owner, lanes);
    }
    sink.finish(output, shape)
}

struct OriginalScale<'a, S>(&'a mut S);
impl<S: AttentionSink> ScaleSink for OriginalScale<'_, S> {
    type View = S::View;
    type Chunk = S::Words;
    type Owner = S::Owner;
    type Output = S::Words;
    type Produced = S::View;
    fn begin(&mut self, _: &S::View, _: &[usize], _: &ScaleDeclaration) -> Result<(), Error> {
        Ok(())
    }
    fn chunks<'a>(input: &'a S::View, width: usize) -> impl Iterator<Item = S::Words> + 'a
    where
        Self::Chunk: 'a,
    {
        S::chunks(S::words(input), width)
    }
    fn len(chunk: &S::Words) -> usize {
        S::len(chunk)
    }
    fn native_chunk(
        &mut self,
        chunk: S::Words,
        declaration: &ScaleDeclaration,
    ) -> Result<S::Owner, Error> {
        self.0.scale_chunk(&chunk, declaration.coefficient())
    }
    fn append(output: &mut S::Words, owner: S::Owner, lanes: usize) {
        S::append_owner(output, owner, lanes);
    }
    fn finish(&mut self, output: S::Words, shape: &[usize]) -> Result<S::View, Error> {
        Ok(S::view(output, shape))
    }
}

pub(super) fn scale<S: AttentionSink>(
    sink: &mut S,
    op: &pllm_models::ModelOperation,
    shape: &[usize],
    input: Option<&S::View>,
) -> Result<S::View, Error> {
    // Preserve the original adapter's coefficient call BEFORE missing-input
    // rejection (including malformed/test-adapter panic precedence).
    let declaration = S::scale_declaration(op)?;
    scale_chunks(&mut OriginalScale(sink), &declaration, shape, input)
}

pub(in crate::replicated_decoder) fn softmax<S: AttentionSink>(
    sink: &mut S,
    shape: &[usize],
    input: Option<&S::View>,
    sequence: usize,
    position: usize,
    keys: usize,
    zero: S::Word,
) -> Result<S::View, Error> {
    let mut words = S::Words::default();
    for (i, row) in S::rows(S::words(input.ok_or(Error::Binding)?), keys).enumerate() {
        let width = position + i % sequence + 1;
        // Adapter materializes only this prefix at gather time. Do not retain a
        // prefix buffer across native construction (the original temporary dies
        // immediately after gather).
        let owner = sink.softmax_row(&row, width)?;
        S::append_owner(&mut words, owner, width);
        S::repeat(&mut words, zero, keys - width);
    }
    Ok(S::view(words, shape))
}

#[allow(clippy::too_many_arguments)]
pub(in crate::replicated_decoder) fn values<S: AttentionSink>(
    sink: &mut S,
    op: &pllm_models::ModelOperation,
    shape: &[usize],
    a: Option<&S::View>,
    b: Option<&S::View>,
    sequence: usize,
    position: usize,
    keys: usize,
) -> Result<S::View, Error> {
    let probabilities = a.ok_or(Error::Binding)?;
    let values = b.ok_or(Error::Binding)?;
    let (heads, width) = (shape[1], shape[3]);
    let group = op
        .attributes
        .get("group_size")
        .and_then(|v| v.as_u64())
        .ok_or(Error::Binding)? as usize;
    if group == 0 || heads / group != S::shape(values)[1] {
        return Err(Error::Shape);
    }
    let mut words = S::Words::default();
    for h in 0..heads {
        for r in 0..sequence {
            let valid = position + r + 1;
            let p = S::view(
                S::slice(
                    S::words(probabilities),
                    (h * sequence + r) * keys..(h * sequence + r) * keys + valid,
                ),
                &[1, valid],
            );
            let mut selected = S::Words::default();
            for d in 0..width {
                for k in 0..valid {
                    S::push(
                        &mut selected,
                        S::at(S::words(values), ((h / group) * keys + k) * width + d),
                    );
                }
            }
            S::extend(
                &mut words,
                sink.values(&p, &S::view(selected, &[width, valid]))?,
            );
        }
    }
    Ok(S::view(words, shape))
}

impl AttentionSink for Builder {
    type View = View;
    type Words = Words;
    type Word = (usize, usize);
    type Owner = usize;
    fn shape(view: &View) -> &[usize] {
        &view.shape
    }
    fn words(view: &View) -> &Words {
        &view.words
    }
    fn len(words: &Words) -> usize {
        words.len()
    }
    fn slice(words: &Words, range: std::ops::Range<usize>) -> Words {
        words.slice(range)
    }
    fn rows(words: &Words, width: usize) -> impl Iterator<Item = Words> + '_ {
        words.chunks_exact(width)
    }
    fn chunks(words: &Words, width: usize) -> impl Iterator<Item = Words> + '_ {
        words.chunks(width)
    }
    fn at(words: &Words, index: usize) -> Self::Word {
        words.at(index)
    }
    fn push(words: &mut Words, word: Self::Word) {
        words.push(word);
    }
    fn extend(words: &mut Words, more: Words) {
        words.extend(more);
    }
    fn repeat(words: &mut Words, word: Self::Word, count: usize) {
        words.extend(std::iter::repeat_n(word, count));
    }
    fn append_owner(words: &mut Words, owner: usize, lanes: usize) {
        words.extend((0..lanes).map(|i| (owner, i)));
    }
    fn view(words: Words, shape: &[usize]) -> View {
        View {
            words,
            shape: shape.to_vec(),
        }
    }
    fn begin_rotary(
        &mut self,
        op: &pllm_models::ModelOperation,
        shape: &[usize],
        input: &View,
        position: usize,
        sequence: usize,
        theta: f32,
    ) -> Result<(), Error> {
        let Some(active) = &self.request_emission else {
            return Ok(());
        };
        let plan = Arc::clone(active.semantic_plan());
        request_origin::rotary_creation::begin(
            self,
            RotaryEvent {
                graph: &plan.plan().prefill,
                operation: op,
                input,
                shape,
                position,
                sequence,
            },
            theta,
        )
    }
    fn rotary_row(
        &mut self,
        row: &Words,
        theta: f32,
        position: u32,
        width: usize,
    ) -> Result<usize, Error> {
        let (cos, sin) = tensor::Program::rotary_coefficients(theta, position, width)?;
        let energy = if self.public_norms.is_empty() {
            None
        } else {
            self.row_norm(row, width)
                .or_else(|| self.attention_energy(row, width))
                .map(|energy| norm_bounds::rotary_energy(energy, &cos, &sin, 18))
                .transpose()?
        };
        let factory = request_origin::rotary_creation::native_row(self, row, width, &cos, &sin)?;
        let input = self.gather(row)?;
        let program = if let Some(factory) = factory {
            let (program, certificate) = factory.build()?.into_parts();
            if certificate != tensor::ProducedCertificate::None {
                return Err(Error::Binding);
            }
            program
        } else {
            tensor::Program::rotary_half(1, width, self.nodes[input].bounds, &cos, &sin)?
        };
        let output = self.tensor(input, program)?;
        if let Some(energy) = energy {
            let row = (0..width).map(|i| (output, i)).collect();
            self.record_attention_energy(&row, energy)?;
        }
        request_origin::rotary_creation::row_created(self, input, output, cos, sin, position)?;
        Ok(output)
    }
    fn scores(&mut self, query: &View, keys: &View) -> Result<Words, Error> {
        Ok(self
            .matrix_with_attention(query, MatrixView::Values(keys), true, true)?
            .words)
    }
    fn values(&mut self, probabilities: &View, values: &View) -> Result<Words, Error> {
        Ok(self.linear(probabilities, values, true)?.words)
    }
    fn scale_coefficient(width: u64) -> i64 {
        ((1.0_f32 / (width as f32).sqrt()) * ONE as f32) as i64
    }
    fn scale_declaration(op: &pllm_models::ModelOperation) -> Result<ScaleDeclaration, Error> {
        scale_declaration(op)
    }
    fn scale_chunk(&mut self, chunk: &Words, coefficient: i64) -> Result<usize, Error> {
        let input = self.gather(chunk)?;
        let mut b = tensor::Builder::new(chunk.len(), self.nodes[input].bounds)?;
        let raw = b.affine(b.input(), None, affine(coefficient, 0, 0))?;
        let output = b.rescale(raw, 18)?;
        self.tensor(input, b.finish(output)?)
    }
    fn softmax_row(&mut self, row: &Words, width: usize) -> Result<usize, Error> {
        let input = self.gather(&row.slice(..width))?;
        let (program, mass) = if self.public_weights {
            let (program, mass) =
                tensor::Program::softmax_certified(1, width, self.nodes[input].bounds)?;
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
        Ok(id)
    }
}
