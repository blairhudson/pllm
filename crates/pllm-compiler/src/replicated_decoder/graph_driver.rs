//! Shared original RMSNorm and bounded rank/view traversal for expanded lowering.
//!
//! This is a deliberately bounded extraction, not a complete graph driver. Other
//! operator arms remain in `graph`. A future compact sink can reuse this family
//! only after mapping the original whole-input checked/sticky-validity producer.
//! Rank/view hooks retain existing words without publishing new execution owners
//! or row certificates. This module does not enable compact lowering or confer
//! proof/admission authority.
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
    if op.attributes.get("semantics").and_then(|v| v.as_str()) != Some("visible_valid_prefix") {
        return Err(Error::Binding);
    }
    Ok(S::alias(input.ok_or(Error::Binding)?))
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
