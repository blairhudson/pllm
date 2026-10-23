//! Original checked request epilogue, shared by production and replay fixtures.
//!
//! This only appends the original expanded acceptance nodes. Request traversal,
//! policy gating, finalization, identity and complete-work admission stay with
//! their existing owners.
use super::*;

impl Builder {
    pub(super) fn final_acceptance(
        &mut self,
        input: usize,
        outputs: usize,
        selected: &mut Vec<(usize, usize)>,
    ) -> Result<(), Error> {
        let (masked, valid) = self.acceptance_nodes(input, outputs, selected)?;
        *selected = (0..outputs).map(|i| (masked, i)).collect();
        selected.push((valid, 0));
        Ok(())
    }

    /// Same epilogue; consumes logical history, not caller-selected authority.
    pub(super) fn final_acceptance_owned<'s>(
        &mut self,
        pending: request_origin::PendingRequest<'s>,
    ) -> Result<request_origin::LogicalComplete<'s>, Error> {
        request_origin::accept(self, pending)
    }

    pub(super) fn acceptance_nodes(
        &mut self,
        input: usize,
        outputs: usize,
        selected: &[(usize, usize)],
    ) -> Result<(usize, usize), Error> {
        let builder = self;
        builder.admission_stage = "final_acceptance";
        let valid = if let Some(valid) = builder.validity {
            valid
        } else {
            let anchor = builder.gather(&std::iter::once((input, 0)).collect())?;
            builder.constant(anchor, 1)?
        };
        #[cfg(test)]
        if super::super::runtime::oracle_trace::enabled() {
            super::super::runtime::oracle_trace::diagnostic(
                0,
                "acceptance.valid",
                &[],
                vec![(valid, 0)],
            );
        }
        // Invalid executions reveal only zero token shares and one
        // aggregate rejection bit. No per-site validity is exported.
        let selected_words: Words = selected.iter().copied().collect();
        let token_node = builder.gather(&selected_words)?;
        let flags: Words = std::iter::repeat_n((valid, 0), outputs).collect();
        let flag_node = builder.gather(&flags)?;
        let masked = builder.product(token_node, flag_node, None, false)?;
        Ok((masked, valid))
    }
}

#[cfg(test)]
#[path = "request_acceptance_tests.rs"]
mod tests;
