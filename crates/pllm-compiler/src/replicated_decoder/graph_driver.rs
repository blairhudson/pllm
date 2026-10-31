//! Shared original normalization, rank/view and contraction traversal.
//!
//! This is a deliberately bounded extraction, not a complete graph driver. Other
//! operator arms remain in `graph`. A future compact sink can reuse this family
//! only after mapping the original whole-input checked/sticky-validity producer.
//! Rank/view hooks retain existing words without publishing new execution owners
//! or row certificates. This module does not enable compact lowering or confer
//! proof/admission authority.
//! Attention, RoPE and KV traversal are not extracted/qualified by this module.
use super::*;

/// Storage hooks only: the driver supplies the original logical lane order, and
/// the sink retains its own word/view representation. No gather, native owner,
/// certificate derivation, shape repair or allocation admission occurs here.
pub(in crate::replicated_decoder) trait RankViewSink {
    type View;
    type Words;

    fn shape(input: &Self::View) -> &[usize];
    fn len(input: &Self::View) -> usize;
    fn reorder(input: &Self::View, indices: impl Iterator<Item = usize>) -> Self::Words;
    fn suffix(input: &Self::View, start: usize) -> Self::Words;
    fn view(words: Self::Words, shape: &[usize]) -> Self::View;
    fn alias(input: &Self::View) -> Self::View;
}

/// Original two layout permutations, including their asymmetric rank guards.
/// These are not general reshapes: batch is not folded into either permutation.
/// Input binding precedes layout validation; final lane-count validation remains
/// in `graph`, after the original indexing (including malformed-index panics).
pub(in crate::replicated_decoder) fn reshape<S: RankViewSink>(
    op: &pllm_models::ModelOperation,
    shape: &[usize],
    input: Option<&S::View>,
) -> Result<S::View, Error> {
    let a = input.ok_or(Error::Binding)?;
    let words = match op.attributes.get("layout").and_then(|v| v.as_str()) {
        Some("batch_heads_sequence_feature") if shape.len() == 4 => {
            let (heads, rows, width) = (shape[1], shape[2], shape[3]);
            S::reorder(
                a,
                (0..heads).flat_map(|h| {
                    (0..rows)
                        .flat_map(move |r| (0..width).map(move |d| (r * heads + h) * width + d))
                }),
            )
        }
        Some("batch_sequence_hidden") if S::shape(a).len() == 4 => {
            let (heads, rows, width) = (S::shape(a)[1], S::shape(a)[2], S::shape(a)[3]);
            S::reorder(
                a,
                (0..rows).flat_map(|r| {
                    (0..heads)
                        .flat_map(move |h| (0..width).map(move |d| (h * rows + r) * width + d))
                }),
            )
        }
        _ => return Err(Error::Binding),
    };
    Ok(S::view(words, shape))
}

/// Original suffix selection: bind input, unwrap output width, subtract, slice.
/// In particular, this does not add rank/width checks or infer input geometry.
pub(in crate::replicated_decoder) fn last_token<S: RankViewSink>(
    shape: &[usize],
    input: Option<&S::View>,
) -> Result<S::View, Error> {
    let a = input.ok_or(Error::Binding)?;
    let width = *shape.last().unwrap();
    Ok(S::view(S::suffix(a, S::len(a) - width), shape))
}

/// Original visible-prefix alias: semantics check precedes input binding, and
/// the input's actual shape is retained, not the declared output shape.
pub(in crate::replicated_decoder) fn cache_suffix<S: RankViewSink>(
    op: &pllm_models::ModelOperation,
    input: Option<&S::View>,
) -> Result<S::View, Error> {
    check_cache_suffix(op)?;
    Ok(S::alias(input.ok_or(Error::Binding)?))
}

pub(in crate::replicated_decoder) fn check_cache_suffix(
    op: &pllm_models::ModelOperation,
) -> Result<(), Error> {
    if op.attributes.get("semantics").and_then(|v| v.as_str()) != Some("visible_valid_prefix") {
        return Err(Error::Binding);
    }
    Ok(())
}

impl RankViewSink for Builder {
    type View = View;
    type Words = Words;

    fn shape(input: &View) -> &[usize] {
        &input.shape
    }

    fn len(input: &View) -> usize {
        input.words.len()
    }

    fn reorder(input: &View, indices: impl Iterator<Item = usize>) -> Words {
        indices.map(|i| input.words.at(i)).collect()
    }

    fn suffix(input: &View, start: usize) -> Words {
        input.words.slice(start..)
    }

    fn view(words: Words, shape: &[usize]) -> View {
        View {
            words,
            shape: shape.to_vec(),
        }
    }

    fn alias(input: &View) -> View {
        input.clone()
    }
}

/// Storage hooks for the original complete-input guard followed by ordered rows.
/// Words/owners are sink-local handles. No expanded lane array, native constructor,
/// source clone, certificate promotion or replacement Arc belongs to the driver.
pub(in crate::replicated_decoder) trait RmsNormSink {
    type Words;
    type Owner;
    type Output: Default;

    fn checked(&self) -> bool;
    fn checked_rows(&mut self, words: &Self::Words, width: usize) -> Result<Self::Words, Error>;
    fn rows(words: &Self::Words, width: usize) -> impl Iterator<Item = Self::Words> + '_;
    fn rms_norm_row(
        &mut self,
        row: &Self::Words,
        width: usize,
        gamma: &SourceTensor,
    ) -> Result<Self::Owner, Error>;
    fn append(output: &mut Self::Output, owner: Self::Owner, width: usize);

    #[cfg(test)]
    fn range_begin(&self, site: &AdmissionSite, id: &str, shape: &[usize], words: &Self::Words);
    #[cfg(test)]
    fn range_end(&self);
}

/// Original validation/error order and guard-before-ALL-rows order. In particular,
/// a missing input is resolved only after the gamma binding/shape checks, and a
/// failed guard returns before range_end or any row/native publication.
pub(in crate::replicated_decoder) fn rms_norm<S: RmsNormSink>(
    sink: &mut S,
    op: &pllm_models::ModelOperation,
    shape: &[usize],
    input: Option<&S::Words>,
    public: &BTreeMap<String, SourceTensor>,
    _site: &AdmissionSite,
) -> Result<S::Output, Error> {
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
    let mut words = S::Output::default();
    let original = input.ok_or(Error::Binding)?;
    #[cfg(test)]
    sink.range_begin(_site, &op.id, shape, original);
    let checked = if sink.checked() {
        Some(sink.checked_rows(original, width)?)
    } else {
        None
    };
    #[cfg(test)]
    sink.range_end();
    let input_words = checked.as_ref().unwrap_or(original);
    for row in S::rows(input_words, width) {
        let output = sink.rms_norm_row(&row, width, gamma)?;
        S::append(&mut words, output, width);
    }
    Ok(words)
}

impl RmsNormSink for Builder {
    type Words = Words;
    type Owner = usize;
    type Output = Words;

    fn checked(&self) -> bool {
        self.checked_bound.is_some()
    }

    fn checked_rows(&mut self, words: &Words, width: usize) -> Result<Words, Error> {
        Builder::checked_rows(self, words, width)
    }

    fn rows(words: &Words, width: usize) -> impl Iterator<Item = Words> + '_ {
        words.chunks_exact(width)
    }

    fn rms_norm_row(
        &mut self,
        row: &Words,
        width: usize,
        gamma: &SourceTensor,
    ) -> Result<usize, Error> {
        Builder::rms_norm_row(self, row, width, gamma)
    }

    fn append(output: &mut Words, owner: usize, width: usize) {
        output.extend((0..width).map(|i| (owner, i)));
    }

    #[cfg(test)]
    fn range_begin(&self, site: &AdmissionSite, id: &str, shape: &[usize], words: &Words) {
        if oracle_trace::enabled() {
            oracle_trace::range_begin(site.selection, id, &shape[1..], words.iter().collect());
        }
    }

    #[cfg(test)]
    fn range_end(&self) {
        if oracle_trace::enabled() {
            oracle_trace::range_end();
        }
    }
}

/// Original weight-bearing graph composition. Source/view/word handles belong to
/// the sink; native matrix admission, selector factories, public ownership,
/// interning and proof ledgers remain there. No replacement source or native Arc
/// is manufactured by this driver. A future compact adapter must implement these
/// SAME contracts; the existing compact prototype is not such an adapter yet.
pub(in crate::replicated_decoder) trait ContractionSink {
    type View;
    type Source;
    type Words: Default;
    type Token: Copy;

    fn source_shape(source: &Self::Source) -> &[usize];
    fn public_weights(&self) -> bool;
    fn lookup_policy(&self) -> policy::LookupPolicy;
    fn tokens(input: &Self::View) -> impl Iterator<Item = Self::Token> + '_;
    fn table_bounds(&self, table: &Self::Source) -> Interval;
    fn trie(
        &mut self,
        tokens: &Self::View,
        table: &Self::Source,
        shape: &[usize],
    ) -> Result<Self::View, Error>;
    fn tiled(
        &mut self,
        tokens: &Self::View,
        table: &Self::Source,
        shape: &[usize],
    ) -> Result<Self::View, Error>;
    fn public_selectors(
        &mut self,
        token: Self::Token,
        vocabulary: usize,
    ) -> Result<Self::Words, Error>;
    fn private_selectors(
        &mut self,
        one_hot: &mut Self::Words,
        token: Self::Token,
        start: usize,
        count: usize,
    ) -> Result<(), Error>;
    fn view(words: Self::Words, shape: &[usize]) -> Self::View;
    fn selector_view(words: Self::Words, vocabulary: usize) -> Self::View;
    fn matrix(
        &mut self,
        input: &Self::View,
        source: &Self::Source,
        transposed: bool,
        rescale: bool,
    ) -> Result<Self::View, Error>;
    fn restore_table_bounds(&mut self, row: &Self::View, bounds: Interval);
    fn append_row(output: &mut Self::Words, row: Self::View);
    fn len(output: &Self::View) -> usize;
    fn bias(
        &mut self,
        output: &mut Self::View,
        source: &Self::Source,
        indices: BiasIndices,
    ) -> Result<(), Error>;
    fn set_shape(output: &mut Self::View, shape: &[usize]);
    fn projection_energies(
        &mut self,
        input: &Self::View,
        output: &Self::View,
        weight: &Self::Source,
        bias: Option<&Self::Source>,
    ) -> Result<(), Error>;
}

fn contraction_source<'a, S>(
    op: &pllm_models::ModelOperation,
    sources: &'a BTreeMap<String, S>,
    name: &str,
) -> Result<&'a S, Error> {
    sources
        .get(
            op.attributes
                .get(name)
                .and_then(|v| v.as_str())
                .ok_or(Error::Binding)?,
        )
        .ok_or(Error::Binding)
}

/// Concrete borrowed-control iterator. Modulo remains lazy: a zero bias width
/// panics only if the original nonempty repetition would have evaluated it.
pub(in crate::replicated_decoder) struct BiasIndices {
    range: std::ops::Range<usize>,
    width: usize,
}

impl Iterator for BiasIndices {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        self.range.next().map(|i| i % self.width)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}

/// Input binding precedes weight lookup/rank checks. Matrix publication precedes
/// bias binding/shape checks; final shape replacement precedes source re-binding
/// and proof propagation. Do not hoist any of those validations to a preflight.
pub(in crate::replicated_decoder) fn projection<S: ContractionSink>(
    sink: &mut S,
    op: &pllm_models::ModelOperation,
    shape: &[usize],
    input: Option<&S::View>,
    sources: &BTreeMap<String, S::Source>,
) -> Result<S::View, Error> {
    let mut output = sink.matrix(
        input.ok_or(Error::Binding)?,
        contraction_source(op, sources, "weight")?,
        false,
        true,
    )?;
    if op.attributes.get("bias").is_some_and(|v| !v.is_null()) {
        let bias = contraction_source(op, sources, "bias")?;
        if S::source_shape(bias) != [*shape.last().ok_or(Error::Shape)?] {
            return Err(Error::Shape);
        }
        let indices = BiasIndices {
            range: 0..S::len(&output),
            width: S::source_shape(bias)[0],
        };
        sink.bias(&mut output, bias, indices)?;
    }
    S::set_shape(&mut output, shape);
    sink.projection_energies(
        input.ok_or(Error::Binding)?,
        &output,
        contraction_source(op, sources, "weight")?,
        op.attributes
            .get("bias")
            .filter(|v| !v.is_null())
            .map(|_| contraction_source(op, sources, "bias"))
            .transpose()?,
    )?;
    Ok(output)
}

/// Uses the original global token view, NOT a substituted first semantic input.
/// Public Trie/Tiled retain their original whole-family constructors. PerToken
/// preserves token order and private 128-selector tile order, then raw matrix,
/// original table-envelope restoration and append. No new token certificate.
pub(in crate::replicated_decoder) fn token_lookup<S: ContractionSink>(
    sink: &mut S,
    op: &pllm_models::ModelOperation,
    shape: &[usize],
    tokens: &S::View,
    sources: &BTreeMap<String, S::Source>,
    vocabulary: usize,
) -> Result<S::View, Error> {
    let table = contraction_source(op, sources, "weight")?;
    if S::source_shape(table) != [vocabulary, *shape.last().ok_or(Error::Shape)?] {
        return Err(Error::Shape);
    }
    if sink.public_weights() && sink.lookup_policy() == policy::LookupPolicy::Trie {
        return sink.trie(tokens, table, shape);
    } else if sink.public_weights() && sink.lookup_policy() == policy::LookupPolicy::Tiled {
        return sink.tiled(tokens, table, shape);
    }
    let table_bound = sink.table_bounds(table);
    let mut words = S::Words::default();
    for token in S::tokens(tokens) {
        let mut one_hot = S::Words::default();
        if sink.public_weights() {
            one_hot = sink.public_selectors(token, vocabulary)?;
        } else {
            for start in (0..vocabulary).step_by(128) {
                let n = 128.min(vocabulary - start);
                sink.private_selectors(&mut one_hot, token, start, n)?;
            }
        }
        let selectors = S::selector_view(one_hot, vocabulary);
        let row = sink.matrix(&selectors, table, true, false)?;
        sink.restore_table_bounds(&row, table_bound);
        S::append_row(&mut words, row);
    }
    Ok(S::view(words, shape))
}

/// Extra driver control is not zero merely because no new heap is required by
/// the expanded adapter. These selected numeric logical terms include typed
/// caller/return copies, handles and concrete iterators, NOT a complete stack or
/// constructor envelope. Existing sink-owned Vec/Words, native scratch, compiled
/// callframes and aggregate admission are NOT priced by it. Original retained
/// storage/admission policy stays unchanged, not newly certified by this quote.
/// Quoting invokes the sink's iterator hook: this is not an allocation guarantee
/// for an arbitrary hook. Expanded Words::iter borrows the existing span vector.
#[allow(dead_code)] // seam for future consumer resource composition, not admission
pub(in crate::replicated_decoder) fn contraction_control<S: ContractionSink>(
    tokens: &S::View,
) -> Result<ContractionControl, Error> {
    let token_iterator = S::tokens(tokens);
    contraction_control_for_iterator::<S, _>(&token_iterator)
}

fn contraction_control_for_iterator<S: ContractionSink, I: Iterator<Item = S::Token>>(
    _iterator: &I,
) -> Result<ContractionControl, Error> {
    contraction_control_layout_bytes::<S, I>()
}

/// Selected logical slots only; constructs no sink, source, view or iterator.
/// The caller must pair `I` with the concrete iterator used by `S::tokens`.
/// An explicit generic argument alone is not proof of that correspondence or
/// authority to replace an existing admission quote.
#[cfg(test)]
pub(in crate::replicated_decoder) fn contraction_control_layout<
    S: ContractionSink,
    I: Iterator<Item = S::Token>,
>() -> Result<ContractionControl, Error> {
    contraction_control_layout_bytes::<S, I>()
}

fn contraction_control_layout_bytes<
    S: ContractionSink,
    I: Iterator<Item = S::Token>,
>() -> Result<ContractionControl, Error> {
    let add = |a: usize, b: usize| a.checked_add(b).ok_or(Error::Bounds);
    let twice = |n: usize| n.checked_mul(2).ok_or(Error::Bounds);
    let mut projection = std::mem::size_of::<S::View>();
    projection = add(
        projection,
        twice(std::mem::size_of::<Result<S::View, Error>>())?,
    )?;
    projection = add(
        projection,
        std::mem::size_of::<Result<&S::Source, Error>>()
            .checked_mul(3)
            .ok_or(Error::Bounds)?,
    )?;
    projection = add(
        projection,
        std::mem::size_of::<Result<Option<&S::Source>, Error>>(),
    )?;
    projection = add(projection, twice(std::mem::size_of::<Result<(), Error>>())?)?;
    projection = add(
        projection,
        twice(std::mem::size_of::<Result<&S::View, Error>>())?,
    )?;
    projection = add(
        projection,
        twice(std::mem::size_of::<Result<&usize, Error>>())?,
    )?;
    projection = add(projection, twice(std::mem::size_of::<BiasIndices>())?)?;
    projection = add(
        projection,
        std::mem::size_of::<(
            &mut S,
            &pllm_models::ModelOperation,
            &[usize],
            Option<&S::View>,
            &BTreeMap<String, S::Source>,
        )>(),
    )?;
    let mut lookup = twice(std::mem::size_of::<S::Words>())?;
    lookup = add(lookup, twice(std::mem::size_of::<S::View>())?)?; // selectors + matrix row
    lookup = add(
        lookup,
        twice(std::mem::size_of::<Result<S::View, Error>>())?,
    )?;
    lookup = add(
        lookup,
        twice(std::mem::size_of::<Result<S::Words, Error>>())?,
    )?;
    lookup = add(lookup, twice(std::mem::size_of::<Result<(), Error>>())?)?;
    lookup = add(
        lookup,
        twice(std::mem::size_of::<Result<&S::Source, Error>>())?,
    )?;
    lookup = add(lookup, twice(std::mem::size_of::<Result<&usize, Error>>())?)?;
    lookup = add(lookup, twice(std::mem::size_of::<I>())?)?;
    lookup = add(lookup, twice(std::mem::size_of::<Option<S::Token>>())?)?;
    lookup = add(lookup, std::mem::size_of::<S::Token>())?;
    lookup = add(
        lookup,
        twice(std::mem::size_of::<std::iter::StepBy<std::ops::Range<usize>>>())?,
    )?;
    lookup = add(lookup, std::mem::size_of::<Interval>())?;
    lookup = add(
        lookup,
        std::mem::size_of::<(
            &mut S,
            &pllm_models::ModelOperation,
            &[usize],
            &S::View,
            &BTreeMap<String, S::Source>,
            usize,
        )>(),
    )?;
    Ok(ContractionControl {
        projection_bytes: projection,
        lookup_bytes: lookup,
        sink_owned_heap_bytes: None,
        compiled_callframe_bytes: None,
    })
}

#[allow(dead_code)]
pub(in crate::replicated_decoder) struct ContractionControl {
    pub projection_bytes: usize,
    pub lookup_bytes: usize,
    pub sink_owned_heap_bytes: Option<usize>,
    pub compiled_callframe_bytes: Option<usize>,
}

impl ContractionSink for Builder {
    type View = View;
    type Source = SourceView;
    type Words = Words;
    type Token = (usize, usize);
    fn source_shape(source: &SourceView) -> &[usize] {
        &source.shape
    }
    fn public_weights(&self) -> bool {
        self.public_weights
    }
    fn lookup_policy(&self) -> policy::LookupPolicy {
        self.lookup
    }
    fn tokens(input: &View) -> impl Iterator<Item = Self::Token> + '_ {
        input.words.iter()
    }
    fn table_bounds(&self, table: &SourceView) -> Interval {
        self.nodes[table.owner].bounds
    }
    fn trie(&mut self, tokens: &View, table: &SourceView, shape: &[usize]) -> Result<View, Error> {
        self.trie_lookup(tokens, table, shape)
    }
    fn tiled(&mut self, tokens: &View, table: &SourceView, shape: &[usize]) -> Result<View, Error> {
        self.tiled_lookup(tokens, table, shape)
    }
    fn public_selectors(&mut self, token: Self::Token, vocabulary: usize) -> Result<Words, Error> {
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
        Ok(self.view(equal, vec![1, vocabulary]).words)
    }
    fn private_selectors(
        &mut self,
        one_hot: &mut Words,
        token: Self::Token,
        start: usize,
        n: usize,
    ) -> Result<(), Error> {
        let x = self.gather(&std::iter::repeat_n(token, n).collect())?;
        let mut builder = tensor::Builder::new(n, self.nodes[x].bounds)?;
        let coefficients: Vec<_> = (start..start + n)
            .map(|i| affine(1, 0, -(i as i64)))
            .collect();
        let shifted = builder.affine_lanes(builder.input(), &coefficients)?;
        let program = builder.finish(shifted)?;
        let difference = self.tensor(x, program)?;
        let program = tensor::Program::equal_zero(n, self.nodes[difference].bounds)?;
        let equal = self.tensor(difference, program)?;
        one_hot.extend((0..n).map(|i| (equal, i)));
        Ok(())
    }
    fn view(words: Words, shape: &[usize]) -> View {
        View {
            words,
            shape: shape.to_vec(),
        }
    }
    fn selector_view(words: Words, vocabulary: usize) -> View {
        View {
            words,
            shape: vec![1, vocabulary],
        }
    }
    fn matrix(
        &mut self,
        input: &View,
        source: &SourceView,
        transposed: bool,
        rescale: bool,
    ) -> Result<View, Error> {
        Builder::matrix(
            self,
            input,
            MatrixView::source(source, transposed)?,
            rescale,
        )
    }
    fn restore_table_bounds(&mut self, row: &View, bounds: Interval) {
        for (id, _) in row.words.iter() {
            self.nodes[id].bounds = bounds;
        }
    }
    fn append_row(output: &mut Words, row: View) {
        output.extend(row.words);
    }
    fn len(output: &View) -> usize {
        output.words.len()
    }
    fn bias(
        &mut self,
        output: &mut View,
        source: &SourceView,
        indices: BiasIndices,
    ) -> Result<(), Error> {
        let repeated: Words = indices.map(|i| (source.owner, i)).collect();
        output.words = self.binary(&output.words, &repeated, false)?;
        Ok(())
    }
    fn set_shape(output: &mut View, shape: &[usize]) {
        output.shape = shape.to_vec();
    }
    fn projection_energies(
        &mut self,
        input: &View,
        output: &View,
        weight: &SourceView,
        bias: Option<&SourceView>,
    ) -> Result<(), Error> {
        Builder::projection_energies(self, input, output, weight, bias)
    }
}
