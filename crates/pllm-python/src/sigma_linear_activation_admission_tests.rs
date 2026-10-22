//! Negative boundary regressions, not end-to-end admission or native execution.
//! All fixture construction is ordinary test setup, outside any claimed boundary.
//! No Python interpreter, dealer, entropy, issuance or bridge role is created.

use pllm_compiler::{
    lower_decoder_matrix_inventory,
    sigma_linear_activation::{
        CheckpointTensorSelection, ExpectedSigmaSourcePin, SigmaSourceConstructionResources,
        SourceDtype, SourceEndian, SIGMA_LINEAR_ACTIVATION_ENCODING,
    },
    DecoderMatrixInventory,
};
use pllm_garble::shared_allocation::{OwnedBudget, MAX_ALLOCATION_BYTES};
use pllm_models::{lower_model_json, owned_backing, DecoderPlan, DecoderWorkload};
use pllm_types::{Digest, NamedDigest};
use std::mem::size_of;

fn plan() -> DecoderPlan {
    lower_model_json(
        br#"{
            "model_type":"qwen2","hidden_size":8,"intermediate_size":16,
            "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,
            "vocab_size":32,"max_position_embeddings":32,"hidden_act":"silu",
            "rms_norm_eps":1e-6,"rope_theta":10000.0,"tie_word_embeddings":true
        }"#,
        DecoderWorkload {
            batch: 1,
            max_input_tokens: 3,
            max_new_tokens: 2,
        },
    )
    .unwrap()
}

struct PublisherInputs {
    artifact: NamedDigest,
    selection: CheckpointTensorSelection,
    config: Digest,
    semantic_plan: Digest,
}

impl PublisherInputs {
    fn new() -> Self {
        // Synthetic independently supplied publisher premise; no actual bytes
        // are imported and this digest is never elevated into source authority.
        Self {
            artifact: NamedDigest {
                id: "synthetic-original.safetensors".into(),
                digest: Digest::from_sha256([3; 32]),
            },
            selection: CheckpointTensorSelection {
                checkpoint: "synthetic-publisher-premise".into(),
                immutable_revision: "0123456789abcdef0123456789abcdef01234567".into(),
                content_sha256: [3; 32],
                tensor: "layer.weight".into(),
                original_weight_shape: [4, 8],
                weight_layout: "output_input_row_major".into(),
                output_coordinates: vec![3, 1],
                inner_coordinates: vec![7, 2],
                input_tensor: "hidden.0".into(),
                original_input_shape: [2, 8],
                input_layout: "row_inner_row_major".into(),
                input_row_coordinates: vec![1],
                encoding: SIGMA_LINEAR_ACTIVATION_ENCODING.into(),
            },
            config: Digest::from_sha256([7; 32]),
            semantic_plan: Digest::from_sha256([9; 32]),
        }
    }

    fn construct(self, cap: usize) -> Result<ExpectedSigmaSourcePin, &'static str> {
        ExpectedSigmaSourcePin::from_trusted_publisher_admitted(
            self.artifact,
            8,
            self.selection,
            SourceDtype::Bf16,
            SourceEndian::Little,
            128,
            &self.config,
            &self.semantic_plan,
            // Isolated source-metadata contract, NOT a pipeline baseline.
            0,
            cap,
        )
    }
}

#[test]
fn serialized_plan_length_cannot_replace_exact_resident_capacity_census() {
    let mut plan = plan();
    let canonical = serde_json::to_vec(&plan).unwrap();
    let before = plan.checked_owned_backing_bytes(usize::MAX).unwrap();
    let root = size_of::<DecoderPlan>();
    let retained_before = root.checked_add(before).unwrap();

    // Identical semantics/JSON, but more actual retained storage. No clone quote
    // or parser-scratch bound follows from either of these existing-owner sizes.
    plan.prefill.output.reserve_exact(4096);
    let after = plan.checked_owned_backing_bytes(usize::MAX).unwrap();
    assert!(after > before);
    assert!(root.checked_add(after).unwrap() > retained_before);
    assert_eq!(serde_json::to_vec(&plan).unwrap(), canonical);
    assert_eq!(
        plan.checked_owned_backing_bytes(before),
        Err(owned_backing::Error::ByteLimit)
    );
}

#[test]
fn genuine_inventory_constructor_does_not_reserve_original_budget() {
    // These are the actual public construction and retained-query signatures.
    // A retained query needs an owner that has ALREADY been constructed.
    let constructor: fn(&DecoderPlan) -> Result<DecoderMatrixInventory, String> =
        lower_decoder_matrix_inventory;
    let retained: fn(&DecoderMatrixInventory) -> Option<usize> =
        DecoderMatrixInventory::retained_payload_bytes;
    let plan = plan();
    let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let original = budget.budget().used_bytes();
    let inventory = constructor(&plan).unwrap();
    let bytes = retained(&inventory).unwrap();
    assert!(bytes >= size_of::<DecoderMatrixInventory>() + inventory.canonical_bytes().len());
    assert_eq!(budget.budget().used_bytes(), original);
    drop(inventory);
    assert_eq!(budget.budget().used_bytes(), original);
    // This is deliberately NOT an admitted inventory wrapper. Charging `bytes`
    // now would be too late to cover grouping, hashing and serialization above.
}

#[test]
fn source_constructor_rejects_one_short_even_with_maximum_budget_available() {
    let inputs = PublisherInputs::new();
    let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let original = budget.budget().used_bytes();
    let quote: SigmaSourceConstructionResources =
        ExpectedSigmaSourcePin::checked_construction_quote(
            &inputs.artifact,
            &inputs.selection,
            &inputs.config,
            &inputs.semantic_plan,
            0,
        )
        .unwrap();
    assert_eq!(
        inputs.construct(quote.peak_bytes - 1).err(),
        Some("expected source construction capacity limit exceeded")
    );
    assert_eq!(budget.budget().used_bytes(), original);
}

#[test]
fn source_constructor_requotes_real_capacity_instead_of_accepting_stale_cap() {
    let mut inputs = PublisherInputs::new();
    let before = ExpectedSigmaSourcePin::checked_construction_quote(
        &inputs.artifact,
        &inputs.selection,
        &inputs.config,
        &inputs.semantic_plan,
        0,
    )
    .unwrap();
    inputs.selection.checkpoint.reserve_exact(4096);
    let after = ExpectedSigmaSourcePin::checked_construction_quote(
        &inputs.artifact,
        &inputs.selection,
        &inputs.config,
        &inputs.semantic_plan,
        0,
    )
    .unwrap();
    assert!(after.input_metadata_backing_bytes > before.input_metadata_backing_bytes);
    assert_eq!(
        after.tuple_copy_backing_bytes,
        before.tuple_copy_backing_bytes
    );
    assert_eq!(
        inputs.construct(before.peak_bytes).err(),
        Some("expected source construction capacity limit exceeded")
    );
}

#[test]
fn source_numeric_cap_is_not_an_original_budget_reservation() {
    let inputs = PublisherInputs::new();
    let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let original = budget.budget().used_bytes();
    let quote = ExpectedSigmaSourcePin::checked_construction_quote(
        &inputs.artifact,
        &inputs.selection,
        &inputs.config,
        &inputs.semantic_plan,
        0,
    )
    .unwrap();
    assert!(quote.peak_bytes > 0);
    let pin = inputs.construct(quote.peak_bytes).unwrap();
    assert_eq!(budget.budget().used_bytes(), original);
    drop(pin);
    assert_eq!(budget.budget().used_bytes(), original);
    // Positive metadata-constructor control, NEGATIVE reservation assertion.
    // Never pass this uncharged owner to compiler execution or bridge packaging.
}
