//! Public API integration contracts, intentionally separate from compiler and
//! any protected-source ownership or funding path.
use pllm_models::{
    forward_checkpoint_requirements, lower_forward_model, lower_forward_model_value,
    ForwardDType, ForwardDimension, ForwardPlan, ModelOperator,
};
use serde_json::json;

#[test]
fn native_public_forward_source_roundtrip_and_exact_checkpoint_layout() {
    let source = json!({
        "model_type":"bert", "architectures":["BertForSequenceClassification"],
        "vocab_size":17, "hidden_size":8, "num_hidden_layers":2, "num_attention_heads":2,
        "intermediate_size":13, "max_position_embeddings":8, "type_vocab_size":2,
        "hidden_act":"gelu", "layer_norm_eps":"0.000000000001", "position_embedding_type":"absolute",
        "num_labels":3
    });
    let bytes = serde_json::to_vec(&source).unwrap();
    let plan = lower_forward_model(&bytes,8).unwrap();
    assert_eq!(plan, lower_forward_model_value(&source,8).unwrap());
    let replay: ForwardPlan = serde_json::from_slice(&serde_json::to_vec(&plan).unwrap()).unwrap();
    replay.validate().unwrap();
    assert_eq!(replay.digest(),plan.digest());
    let records = forward_checkpoint_requirements(&replay).unwrap();
    assert_eq!(records.len(),41);
    assert!(records.iter().all(|r| r.dtype == ForwardDType::Float32));
    assert_eq!(records.iter().find(|r| r.name == "bert.encoder.layer.0.intermediate.dense.weight").unwrap().shape,vec![13,8]);
    assert_eq!(records.iter().find(|r| r.name == "classifier.bias").unwrap().shape,vec![3]);
    assert_eq!(plan.outputs.len(),3);
    assert_eq!(plan.outputs["last_hidden_state"].tensor.shape,
        vec![ForwardDimension::Fixed(1),ForwardDimension::Sequence,ForwardDimension::Fixed(8)]);
    assert_eq!(plan.operations.iter().filter(|op| op.operator == ModelOperator::GeluErf).count(),2);
    assert_eq!(plan.operations.iter().filter(|op| op.operator == ModelOperator::AttentionMask).count(),2);
    assert!(plan.operations.iter().any(|op| op.operator == ModelOperator::Tanh));
}

#[test]
fn arbitrary_digest_and_decoder_substitution_cannot_authenticate_forward_plan() {
    let source = br#"{"model_type":"bert","architectures":["BertModel"],"vocab_size":7,
        "hidden_size":4,"num_hidden_layers":1,"num_attention_heads":2,"intermediate_size":6,
        "max_position_embeddings":8,"type_vocab_size":2,"hidden_act":"gelu",
        "layer_norm_eps":0.00001,"position_embedding_type":"absolute"}"#;
    let plan = lower_forward_model(source,4).unwrap();
    assert_eq!(plan.outputs.len(),2);
    let mut changed = plan.clone();
    changed.config_digest = pllm_types::Digest::from_sha256([0;32]);
    assert!(changed.validate().is_err());
    assert!(forward_checkpoint_requirements(&changed).is_err());
    let mut changed = plan;
    changed.operations.iter_mut().find(|op| op.operator == ModelOperator::AttentionMask).unwrap().operator = ModelOperator::CausalMask;
    assert!(changed.validate().is_err());
}
