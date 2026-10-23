//! Shared complete-request, growing-prefix traversal.
//!
//! Only representation-specific publication belongs to a sink. Every selection
//! uses the original semantic lowerer, a fresh prefill/state map and position 0.
//! The borrowed sink survives across selections (including sticky validity and
//! original source owners). This is not compact request admission or execution.
use super::*;
use pllm_models::DecoderPlan;

/// Compiler-private storage hooks, not caller-supplied completion/work authority.
/// A compact implementation must publish the SAME prefill through its numeric
/// sinks and append original selected coordinates as token-history aliases.
pub(super) trait RequestSink {
    type View;
    type Selected: Default;

    fn lanes(view: &Self::View) -> usize;

    fn prefill(
        &mut self,
        plan: &DecoderPlan,
        tokens: &Self::View,
        position: usize,
        state: &mut BTreeMap<(u64, pllm_models::StateKind), Self::View>,
        site: &mut AdmissionSite,
    ) -> Result<Self::View, Error>;

    /// Preserve selected order and both original owner/lane coordinates; do not
    /// gather, reconstruct, relabel or start a new numeric schedule here.
    fn append_feedback(
        &mut self,
        tokens: &mut Self::View,
        result: &Self::View,
        selected: &mut Self::Selected,
    ) -> Result<(), Error>;

    #[cfg(test)]
    fn boundary(&self, selection: usize, id: &str, view: &Self::View);
}

/// Original D9 request order. No retained KV, decode graph, independent request
/// compiler, final-acceptance policy, identity or work-budget input lives here.
pub(super) fn drive<S: RequestSink>(
    sink: &mut S,
    config: &[u8],
    rows: usize,
    outputs: usize,
    weights: WeightMode,
    tokens: &mut S::View,
    site: &mut AdmissionSite,
) -> Result<S::Selected, Error> {
    let mut selected = S::Selected::default();
    for selection in 0..outputs {
        *site = AdmissionSite {
            selection,
            ..AdmissionSite::default()
        };
        // Fresh semantic prefill and arithmetic preserve the stochastic numeric
        // schedule and charged source-method work of complete-prefix recompute.
        let plan = lower_with_weights(config, rows + selection, outputs - selection, weights)
            .map_err(|_| Error::Binding)?;
        let mut state = BTreeMap::new();
        #[cfg(test)]
        sink.boundary(selection, "input.tokens", tokens);
        // Keep the complete original owner visible to representation-specific
        // admission. A graph borrow cannot census decode/state/transform backing.
        // This observes existing ownership, not upstream construction admission.
        let result = sink.prefill(&plan, tokens, 0, &mut state, site)?;
        if S::lanes(&result) != 1 {
            return Err(Error::Shape);
        }
        sink.append_feedback(tokens, &result, &mut selected)?;
        #[cfg(test)]
        sink.boundary(selection, "feedback.history", tokens);
    }
    Ok(selected)
}

/// Bounded borrow of original lowering state. No cloned Builder, source backing,
/// validity reset, changed allocation ledger or alternate completion authority.
pub(super) struct Expanded<'a> {
    pub(super) builder: &'a mut Builder,
    pub(super) inputs: &'a BTreeMap<String, SourceView>,
    pub(super) sources: &'a BTreeMap<String, SourceTensor>,
    pub(super) vocabulary: usize,
}

impl RequestSink for Expanded<'_> {
    type View = View;
    type Selected = Vec<(usize, usize)>;

    fn lanes(view: &View) -> usize {
        view.words.len()
    }

    fn prefill(
        &mut self,
        plan: &DecoderPlan,
        tokens: &View,
        position: usize,
        state: &mut BTreeMap<(u64, pllm_models::StateKind), View>,
        site: &mut AdmissionSite,
    ) -> Result<View, Error> {
        self.builder.graph(
            &plan.prefill,
            tokens,
            self.inputs,
            self.sources,
            self.vocabulary,
            position,
            state,
            site,
        )
    }

    fn append_feedback(
        &mut self,
        tokens: &mut View,
        result: &View,
        selected: &mut Self::Selected,
    ) -> Result<(), Error> {
        selected.extend(result.words.iter());
        tokens.words.extend(result.words.iter());
        tokens.shape = vec![1, tokens.words.len()];
        Ok(())
    }

    #[cfg(test)]
    fn boundary(&self, selection: usize, id: &str, view: &View) {
        if super::super::runtime::oracle_trace::enabled() {
            super::super::runtime::oracle_trace::boundary(
                selection,
                id,
                &view.shape,
                view.words.iter().collect(),
            );
        }
    }
}

#[cfg(test)]
#[path = "request_driver_tests.rs"]
mod tests;
