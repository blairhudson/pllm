//! Compiler emission for bounded public lookup. Legacy lowering stays separate.
use super::*;
use crate::replicated_decoder::runtime::lookup::{Plan, ROW_TILE};

impl Builder {
    // Proof issuance is confined to the two trusted model-compiler boundaries.
    // No bounds-only helper grants this capability to a gather/affine result.
    pub(super) fn register_initial_tokens(
        &mut self,
        words: &Words,
        vocabulary: usize,
    ) -> Result<(), Error> {
        for (owner, slot) in words.iter() {
            if !matches!(&self.nodes[owner].code, Code::Input(key) if key == "input.tokens")
                || slot >= self.nodes[owner].lanes
                || self.nodes[owner].bounds != Interval::new(0, vocabulary as i64 - 1)?
            {
                return Err(Error::Binding);
            }
            let mut h = Sha256::new();
            h.update(b"pllm.trie.validated-initial-id-route.v1");
            h.update(self.token_source_binding);
            for n in [owner, slot, vocabulary] {
                h.update((n as u64).to_le_bytes());
            }
            self.retain_token_domain(
                (owner, slot),
                vocabulary,
                runtime::trie::DomainOrigin::ValidatedInitialIds(h.finalize().into()),
            )?;
        }
        Ok(())
    }
    pub(super) fn register_selected_token(
        &mut self,
        output: &Words,
        logits: &Words,
        vocabulary: usize,
    ) -> Result<(), Error> {
        if output.len() != 1 || logits.len() != vocabulary {
            return Err(Error::Binding);
        }
        let bounds = self.bounds(output)?;
        if bounds.lower() < 0 || bounds.upper() >= vocabulary as i64 {
            return Err(Error::Binding);
        }
        let mut h = Sha256::new();
        h.update(b"pllm.trie.complete-native-stable-selection-route.v1");
        h.update(self.token_source_binding);
        h.update((vocabulary as u64).to_le_bytes());
        for (owner, slot) in logits.iter().chain(output.iter()) {
            h.update((owner as u64).to_le_bytes());
            h.update((slot as u64).to_le_bytes());
        }
        self.retain_token_domain(
            output.at(0),
            vocabulary,
            runtime::trie::DomainOrigin::NativeStableSelection(h.finalize().into()),
        )
    }
    fn retain_token_domain(
        &mut self,
        word: (usize, usize),
        vocabulary: usize,
        proof: runtime::trie::DomainOrigin,
    ) -> Result<(), Error> {
        if self.token_source_binding == [0; 32] || self.token_domains.contains_key(&word) {
            return Err(Error::Binding);
        }
        // Conservative construction registry/BTree allowance, retained in payload
        // pricing even though this compiler-only map is discarded after lowering.
        self.allocated = self.allocated.checked_add(160).ok_or(Error::Bounds)?;
        if self.allocated > MAX_PROGRAM_BYTES {
            return Err(Error::Bounds);
        }
        self.token_domains.insert(word, (vocabulary, proof));
        Ok(())
    }
    pub(super) fn trie_lookup(
        &mut self,
        tokens: &View,
        table: &SourceView,
        shape: &[usize],
    ) -> Result<View, Error> {
        let [vocabulary, width] = table.shape.as_slice() else {
            return Err(Error::Shape);
        };
        let owner = self.nodes.get(table.owner).ok_or(Error::Binding)?;
        let Code::PublicInput { digest, .. } = &owner.code else {
            return Err(Error::Binding);
        };
        let digest = *digest;
        let bounds = owner.bounds;
        if !self.public_weights
            || self.lookup != LookupPolicy::Trie
            || vocabulary.checked_mul(*width) != Some(owner.lanes)
            || tokens.words.is_empty()
            || shape.last() != Some(width)
            || shape.iter().try_fold(1usize, |n, d| n.checked_mul(*d))
                != tokens.words.len().checked_mul(*width)
        {
            return Err(Error::Shape);
        }
        let mut words = Words::new();
        for tile in tokens.words.chunks(ROW_TILE) {
            let mut proofs = Vec::new();
            proofs
                .try_reserve_exact(tile.len())
                .map_err(|_| Error::Allocation)?;
            for word in tile.iter() {
                let (v, proof) = self.token_domains.get(&word).ok_or(Error::Binding)?;
                if v != vocabulary {
                    return Err(Error::Binding);
                }
                proofs.push(*proof);
            }
            let input = self.gather(&tile)?;
            // The native owner MOVES this dedicated immediately preceding gather.
            // A matching numeric interval alone never justifies trie pruning.
            let plan = Arc::new(runtime::trie::Plan::new(
                tile.len(),
                *vocabulary,
                *width,
                input,
                Interval::new(0, *vocabulary as i64 - 1)?,
                &proofs,
                runtime::trie::SourceBinding {
                    owner: table.owner,
                    source_commitment: self.token_source_binding,
                    file_digest: digest,
                },
            )?);
            let lanes = tile.len().checked_mul(*width).ok_or(Error::Bounds)?;
            let output = self.push(
                Code::TrieLookup {
                    input,
                    weight: table.owner,
                    plan,
                },
                lanes,
                bounds,
            )?;
            words.extend((0..lanes).map(|slot| (output, slot)));
        }
        Ok(View {
            words,
            shape: shape.to_vec(),
        })
    }
    pub(super) fn tiled_lookup(
        &mut self,
        tokens: &View,
        table: &SourceView,
        shape: &[usize],
    ) -> Result<View, Error> {
        let [vocabulary, width] = table.shape.as_slice() else {
            return Err(Error::Shape);
        };
        let owner = self.nodes.get(table.owner).ok_or(Error::Binding)?;
        if !self.public_weights || !matches!(owner.code, Code::PublicInput { .. }) {
            return Err(Error::Binding);
        }
        let table_bounds = owner.bounds;
        if vocabulary.checked_mul(*width) != Some(owner.lanes)
            || tokens.words.is_empty()
            || shape.last() != Some(width)
            || shape.iter().try_fold(1usize, |n, d| n.checked_mul(*d))
                != tokens.words.len().checked_mul(*width)
        {
            return Err(Error::Shape);
        }
        let mut words = Words::new();
        let mut original = tokens.words.iter();
        loop {
            let tile: Words = original.by_ref().take(ROW_TILE).collect();
            if tile.is_empty() {
                break;
            }
            // Gather original shares, not reconstructed IDs or a public prefix.
            let input = self.gather(&tile)?;
            let mut plan = Plan::new(tile.len(), *vocabulary, *width, self.nodes[input].bounds)?;
            plan.reuse_programs(&self.tensors)?;
            let plan = Arc::new(plan);
            let lanes = tile.len().checked_mul(*width).ok_or(Error::Bounds)?;
            let output = self.push(
                Code::PublicLookup {
                    input,
                    weight: table.owner,
                    plan,
                },
                lanes,
                table_bounds,
            )?;
            words.extend((0..lanes).map(|i| (output, i)));
        }
        Ok(View {
            words,
            shape: shape.to_vec(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn builder(rows: usize) -> (Builder, View, SourceView) {
        let mut builder = Builder {
            public_weights: true,
            ..Builder::default()
        };
        let weight = builder
            .push(
                Code::PublicInput {
                    key: "weights".into(),
                    digest: [1; 32],
                    row_l1: None,
                    row_squared_norm: None,
                    row_l1_squared_sum: None,
                },
                131 * 71,
                Interval::new(-1, 1).unwrap(),
            )
            .unwrap();
        let token = builder
            .push(
                Code::Input("tokens".into()),
                rows,
                Interval::new(0, 130).unwrap(),
            )
            .unwrap();
        let tokens = builder.view(token, vec![rows]);
        (
            builder,
            tokens,
            SourceView {
                owner: weight,
                shape: vec![131, 71],
            },
        )
    }

    #[test]
    fn tiled_lookup_template_resource_charges_unique_arcs_and_peak_overlap_once() {
        let (mut builder, tokens, table) = builder(17);
        let before = builder.allocated;
        let start = builder.nodes.len();
        builder.tiled_lookup(&tokens, &table, &[17, 71]).unwrap();
        assert_eq!(builder.tensors.len(), 2);
        let mut seen = std::collections::BTreeSet::new();
        let mut expected = before;
        let mut workspace = 0;
        let mut count = 0;
        let mut work = WorkBudget::default();
        for node in &builder.nodes[start..] {
            expected += std::mem::size_of::<Node>() + 16 * node.lanes;
            match &node.code {
                Code::Gather(words) => expected += words.metadata_bytes(),
                Code::PublicLookup { plan, .. } => {
                    count += 1;
                    expected += plan.shallow_metadata_bytes() + 1024;
                    for program in plan.programs() {
                        assert!(Arc::ptr_eq(
                            program,
                            builder.tensors.get(&program.digest()).unwrap()
                        ));
                        if seen.insert(program.digest()) {
                            expected += Plan::program_metadata_bytes(program) + 1024;
                        }
                    }
                    workspace = workspace.max(
                        plan.peak_extra_bytes().unwrap()
                            - plan.metadata_bytes().unwrap()
                            - 16 * node.lanes,
                    );
                    work = work.checked_add(plan.work_budget().unwrap()).unwrap();
                }
                _ => panic!("lookup emits only dedicated gathers and bulk owners"),
            }
        }
        assert_eq!(count, 3);
        assert_eq!(seen.len(), 2);
        expected += workspace;
        assert_eq!(builder.workspace_bytes, workspace);
        assert_eq!(builder.allocated, expected);
        let single = Plan::new(1, 131, 71, Interval::new(0, 130).unwrap()).unwrap();
        assert_eq!(work, single.work_budget().unwrap().checked_mul(17).unwrap());
    }

    #[test]
    fn tiled_lookup_template_resource_rejects_before_unpriced_templates_are_registered() {
        let (mut builder, _, table) = builder(8);
        let plan = Arc::new(Plan::new(8, 131, 71, Interval::new(0, 130).unwrap()).unwrap());
        let lanes = 8 * 71;
        let workspace =
            plan.peak_extra_bytes().unwrap() - plan.metadata_bytes().unwrap() - lanes * 16;
        let without_templates = plan.shallow_metadata_bytes()
            + 1024
            + workspace
            + std::mem::size_of::<Node>()
            + lanes * 16;
        builder.allocated = MAX_PROGRAM_BYTES - without_templates;
        let nodes = builder.nodes.len();
        let result = builder.push(
            Code::PublicLookup {
                input: 1,
                weight: table.owner,
                plan,
            },
            lanes,
            Interval::new(-1, 1).unwrap(),
        );
        assert_eq!(result, Err(Error::Bounds));
        assert_eq!(builder.nodes.len(), nodes);
        assert!(builder.tensors.is_empty());
    }
}
