//! Owned forward collector allocation boundary. The borrowed checker, shared
//! artifact interpretation and decoder accounting remain in the guarded parent.
use super::{artifact_slots, contract, known_key, present, text, transformer_blocks, Error, ModelOperator};

/// Forward-only owned records. Deliberately not routed through decoder `check`:
/// its independently prepaid pools and exactly-one-head-per-phase contract stay
/// unchanged. The caller first reconstructs the bounded registered forward plan.
/// Artifact interpretation reuses the shared embedding/projection/norm predicates
/// and the same closed artifact-key and nonempty-identity checks as `check`.
pub(crate) fn forward_records(plan: &crate::ForwardPlan)
    -> Result<Vec<crate::ForwardRequirement>, crate::ModelError> {
    use crate::{ForwardDType, ForwardDimension, ForwardRequirement, ModelError};
    use std::collections::BTreeMap;
    let fail = |e: Error| ModelError::Incomplete(e.to_string());
    let mut widths = BTreeMap::<&str, u64>::new();
    let mut artifacts = BTreeMap::<String, Vec<u64>>::new();
    for op in &plan.operations {
        let output=match op.output.shape.last(){Some(ForwardDimension::Fixed(n)) if *n>0=>*n,
            Some(ForwardDimension::Sequence) if plan.max_sequence>0=>plan.max_sequence,_=>return Err(fail(Error::Shape))};
        let kind = contract(op.operator);
        let input = match op.operator {
            ModelOperator::TokenLookup => {
                transformer_blocks::embedding_parts(&op.inputs,&op.attributes,output)
                    .map_err(|_|fail(Error::Contract))?.cardinality.ok_or_else(||fail(Error::Contract))?
            }
            ModelOperator::Linear => {
                op.inputs.first().filter(|_| op.inputs.len()==1).and_then(|s|widths.get(s.as_str()))
                    .copied().ok_or_else(||fail(Error::ProjectionSource))?
            }
            _=>0,
        };
        if matches!(op.operator,ModelOperator::TokenLookup|ModelOperator::Linear|ModelOperator::LayerNorm) {
            let attrs = op.attributes.as_object().ok_or_else(|| fail(Error::Contract))?;
            if attrs.keys().any(|k| !known_key(kind, k)) { return Err(fail(Error::UnknownAttribute)); }
            for (key,shape,optional) in artifact_slots(op.operator,&op.inputs,&op.attributes,input,output).map_err(fail)? {
                if optional&&present(attrs.get(key)).is_none(){continue;}
                let name = text(attrs.get(key)).map_err(fail)?;
                let shape=shape.dimensions();
                if let Some(previous) = artifacts.get(name) {
                    if previous.as_slice()!=shape { return Err(fail(Error::Conflict)); }
                } else { artifacts.insert(name.into(),shape.to_vec()); }
            }
        }
        if widths.insert(&op.id, output).is_some() { return Err(fail(Error::DuplicateId)); }
    }
    Ok(artifacts.into_iter().map(|(name, shape)| ForwardRequirement {
        name, dtype: ForwardDType::Float32, shape,
    }).collect())
}
