//! Compiler emission for bounded public lookup. Legacy lowering stays separate.
use super::*;
use crate::replicated_decoder::runtime::lookup::{Plan, ROW_TILE};

impl Builder {
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
