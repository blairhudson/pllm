//! Source-stage descriptors only. This does not enable batched execution.

use super::{
    owner_product, owner_sum, reject, stage_id, DecoderRuntimeExecutor, GkmDecoderContract,
    ModelOperator, Step, GKM_DECODER_EXECUTION_BYTES,
};
use pllm_core::gkm::{BatchDescriptor, BatchPlan, BatchResources, Capacity, SharingProfile};
use pllm_types::Digest;
use serde::Serialize;

/// Private metadata quote, not runtime admission or an execution capability.
/// Only the SDK execution identity may serialize this record; never the original
/// decoder/source contract, publication identity, or seed derivation context.
#[derive(Serialize)]
pub(super) struct SourceStageBatchContract {
    schema_version: &'static str,
    execution: &'static str,
    batch_plan: Digest,
    sharing_profile: &'static str,
    descriptors: Vec<SourceStageBatchDescriptor>,
    resources: BatchResourceQuote,
    native_admitted_payload_bytes: usize,
    compiler_encoded_input_peak_bytes: usize,
    client_payload_bytes: usize,
    worker_payload_bytes: usize,
}

/// Trusted source-stage geometry and exact native frame lengths, in execution
/// order. Lookup has no matrix stage/input columns; output_width is table width.
#[derive(Serialize)]
struct SourceStageBatchDescriptor {
    batch_index: usize,
    phase_index: u32,
    operation_index: u32,
    first_claim: u32,
    rows: u32,
    kind: &'static str,
    stage: Option<usize>,
    columns: usize,
    output_width: usize,
    request_bytes: [usize; 2],
    reply_bytes: [usize; 2],
}

/// Mirror the approved core quote without reproducing core sizing formulas or
/// requiring the protocol core to adopt a compiler serialization dependency.
#[derive(Serialize)]
struct BatchResourceQuote {
    max_aggregate_bytes: usize,
    retained_frame_bytes: usize,
    transport_scratch_bytes: usize,
    extra_native_owner_bytes: usize,
    extra_wire_bytes: usize,
}

impl From<&BatchResources> for BatchResourceQuote {
    fn from(resources: &BatchResources) -> Self {
        Self {
            max_aggregate_bytes: resources.max_aggregate_bytes,
            retained_frame_bytes: resources.retained_frame_bytes,
            transport_scratch_bytes: resources.transport_scratch_bytes,
            extra_native_owner_bytes: resources.extra_native_owner_bytes,
            extra_wire_bytes: resources.extra_wire_bytes,
        }
    }
}

impl SourceStageBatchContract {
    pub(super) fn max_aggregate_bytes(&self) -> usize {
        self.resources.max_aggregate_bytes
    }

    pub(super) fn check_resources(&self, resources: &BatchResources) -> Result<(), String> {
        if self.resources.max_aggregate_bytes != resources.max_aggregate_bytes
            || self.resources.retained_frame_bytes != resources.retained_frame_bytes
            || self.resources.transport_scratch_bytes != resources.transport_scratch_bytes
            || self.resources.extra_native_owner_bytes != resources.extra_native_owner_bytes
            || self.resources.extra_wire_bytes != resources.extra_wire_bytes
        {
            return Err(reject(
                "batch admission differs from compiler resource quote",
            ));
        }
        Ok(())
    }

    pub(super) fn derive(
        decoder: &GkmDecoderContract,
        profile: SharingProfile,
    ) -> Result<Self, String> {
        if decoder.transport_resources.is_none() {
            return Err(reject(
                "source-stage batch requires streamed single-window ownership",
            ));
        }
        let plan = decoder.batch_plan()?;
        let stages: Vec<_> = decoder
            .matrices
            .iter()
            .map(|matrix| (stage_id(matrix), matrix.rows, matrix.columns))
            .collect();
        let resources = BatchResources::for_streamed_single_window(
            &stages,
            decoder.vocabulary,
            decoder.width,
            &plan,
            profile,
            decoder.capacity(),
        )
        .map_err(|error| error.to_string())?;
        let sizes = BatchResources::descriptor_wire_sizes(
            &stages,
            decoder.vocabulary,
            decoder.width,
            &plan,
            profile,
            decoder.capacity(),
        )
        .map_err(|error| error.to_string())?;
        if sizes.len() != plan.descriptors().len() {
            return Err(reject("native batch frame count mismatch"));
        }
        let mut descriptors = Vec::new();
        descriptors
            .try_reserve_exact(sizes.len())
            .map_err(|_| reject("batch metadata allocation"))?;
        let mut compiler_encoded_input_peak_bytes = 0;
        for (batch_index, (descriptor, sizes)) in plan.descriptors().iter().zip(sizes).enumerate() {
            let (kind, stage, columns, output_width) = match descriptor.step {
                Step::Lookup => ("lookup", None, 0, decoder.width),
                Step::Linear(id) => {
                    let index = stages
                        .iter()
                        .position(|(stage, _, _)| *stage == id)
                        .ok_or_else(|| reject("batch metadata stage mismatch"))?;
                    (
                        "linear",
                        Some(index),
                        decoder.matrices[index].columns,
                        decoder.matrices[index].rows,
                    )
                }
            };
            if stage.is_some() {
                // Compiler Vec<i64> remains live through native issuance, alongside
                // core's separately priced BatchPayload.input and share owners.
                // Charge the entire B*C owner; no implicit scalar-row credit.
                compiler_encoded_input_peak_bytes = compiler_encoded_input_peak_bytes
                    .max(owner_product(&[8, descriptor.rows as usize, columns])?);
            }
            descriptors.push(SourceStageBatchDescriptor {
                batch_index,
                phase_index: descriptor.phase_index,
                operation_index: descriptor.operation_index,
                first_claim: descriptor.first_claim,
                rows: descriptor.rows,
                kind,
                stage,
                columns,
                output_width,
                request_bytes: sizes.request_bytes,
                reply_bytes: sizes.reply_bytes,
            });
        }
        let native_admitted_payload_bytes = owner_sum([
            decoder
                .native_publication_resources()?
                .admitted_payload_bytes,
            resources.extra_native_owner_bytes,
        ])?;
        let delta = owner_sum([
            resources.extra_native_owner_bytes,
            resources.extra_wire_bytes,
        ])?;
        let client_payload_bytes = owner_sum([
            decoder.execution_payload_estimate_bytes,
            delta,
            compiler_encoded_input_peak_bytes,
        ])?;
        // Output needs no further compiler delta: before bias processing, the
        // scheduled eight-byte output-root allowance (2O) covers float_bytes O
        // plus PyBytes O; afterwards compiler bytes drop and graph backing O
        // plus biased-member copies <=O replace that overlap. Core tentative
        // values remain an independently priced native owner throughout.
        // Decoder capacity has separate role envelopes. Capsule's historical
        // estimate is a conservative complete-request envelope for either role.
        let scalar_worker_bytes = decoder
            .decoder_resources
            .as_ref()
            .map_or(decoder.execution_payload_estimate_bytes, |resources| {
                resources.worker_payload_bytes
            });
        let worker_payload_bytes = owner_sum([scalar_worker_bytes, delta])?;
        let limit = if decoder.capacity() == Capacity::Decoder {
            GKM_DECODER_EXECUTION_BYTES
        } else {
            64 << 20
        };
        if client_payload_bytes > limit || worker_payload_bytes > limit {
            return Err(reject(
                "source-stage batch exceeds execution payload capacity",
            ));
        }
        Ok(Self {
            schema_version: "pllm.gkm.source-stage-batch-contract.v1",
            execution: "source_stage_batch_v1",
            batch_plan: Digest::from_sha256(plan.digest()),
            sharing_profile: match profile {
                SharingProfile::FullShares => "full",
                SharingProfile::Aes128WindowV1 => pllm_core::gkm::SEEDED_SHARING_PROFILE,
            },
            descriptors,
            resources: BatchResourceQuote::from(&resources),
            native_admitted_payload_bytes,
            compiler_encoded_input_peak_bytes,
            client_payload_bytes,
            worker_payload_bytes,
        })
    }
}

impl GkmDecoderContract {
    /// Describe original semantic source-stage boundaries over the scalar window.
    ///
    /// Phase zero is prefill; each subsequent phase is one decode iteration.
    /// `operation_index` indexes the complete semantic phase schedule (including
    /// client-local steps), not the matrix inventory or a run of equal artifacts.
    /// A grouped stage has one descriptor covering all its original members.
    /// Prefill lookup/body stages cover all query rows; the terminal head and
    /// every decode stage remain singleton claims. Graph batch stays unchanged.
    /// This metadata neither changes source identity nor selects execution mode.
    pub fn batch_plan(&self) -> Result<BatchPlan, String> {
        let mut descriptors = Vec::new();
        // Every descriptor owns at least one scalar claim; existing admission
        // already bounds this complete-window count.
        descriptors
            .try_reserve_exact(self.window_steps)
            .map_err(|_| reject("batch descriptor allocation"))?;
        let mut first_claim = 0u32;
        for phase_index in 0..=self.decode_steps {
            let phase = if phase_index == 0 {
                &self.prefill
            } else {
                &self.decode
            };
            let mut matrix_index = 0;
            for (operation_index, operation) in phase.steps.iter().enumerate() {
                if operation.executor == DecoderRuntimeExecutor::ClientLocal {
                    continue;
                }
                let operator = operation.operators[0];
                let step = if operator == ModelOperator::TokenLookup {
                    Step::Lookup
                } else {
                    let matrix = self
                        .matrices
                        .get(matrix_index)
                        .ok_or_else(|| reject("batch matrix schedule mismatch"))?;
                    matrix_index += 1;
                    Step::Linear(stage_id(matrix))
                };
                let rows = if operator == ModelOperator::OutputHead {
                    1
                } else {
                    u32::try_from(phase.query_sequence)
                        .map_err(|_| reject("batch row count overflow"))?
                };
                descriptors.push(BatchDescriptor {
                    phase_index: u32::try_from(phase_index)
                        .map_err(|_| reject("batch phase index overflow"))?,
                    operation_index: u32::try_from(operation_index)
                        .map_err(|_| reject("batch operation index overflow"))?,
                    first_claim,
                    rows,
                    step,
                });
                first_claim = first_claim
                    .checked_add(rows)
                    .ok_or_else(|| reject("batch claim count overflow"))?;
            }
            if matrix_index != self.matrices.len() {
                return Err(reject("batch matrix schedule mismatch"));
            }
        }
        // Core validates the exact flattening against the unchanged scalar plan
        // and owns canonical, layout-independent little-endian descriptor hashing.
        BatchPlan::new(&self.window_plan(), &descriptors).map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pllm_types::canonical_bytes;
    use serde_json::json;
    use std::collections::BTreeSet;

    fn contract(rows: usize, decode_steps: usize) -> GkmDecoderContract {
        GkmDecoderContract::compile(
            &canonical_bytes(&json!({
                "model_type": "qwen2", "hidden_size": 8, "intermediate_size": 16,
                "num_hidden_layers": 2, "num_attention_heads": 2,
                "num_key_value_heads": 1, "vocab_size": 32,
                "max_position_embeddings": 32, "hidden_act": "silu",
                "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
                "tie_word_embeddings": true
            })),
            rows,
            decode_steps,
        )
        .unwrap()
    }

    fn pricing_contract(
        rows: usize,
        intermediate: usize,
        capacity: Capacity,
    ) -> GkmDecoderContract {
        GkmDecoderContract::compile_streamed_with_capacity(
            &canonical_bytes(&json!({
                "model_type": "qwen2", "hidden_size": 8, "intermediate_size": intermediate,
                "num_hidden_layers": 2, "num_attention_heads": 2,
                "num_key_value_heads": 1, "vocab_size": 32,
                "max_position_embeddings": 32, "hidden_act": "silu",
                "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
                "tie_word_embeddings": true
            })),
            rows,
            2,
            capacity,
        )
        .unwrap()
    }

    #[test]
    fn encoded_input_quote_prices_exact_max_bc_not_grouped_outputs_or_member_count() {
        for capacity in [Capacity::Capsule, Capacity::Decoder] {
            for profile in [SharingProfile::FullShares, SharingProfile::Aes128WindowV1] {
                for rows in [1, 2] {
                    for intermediate in [16, 24] {
                        let decoder = pricing_contract(rows, intermediate, capacity);
                        let scalar_bytes = decoder.canonical_bytes();
                        let source_digest = decoder.source_binding_digest();
                        let quote = SourceStageBatchContract::derive(&decoder, profile).unwrap();
                        // Down projection consumes the largest input width. Decode
                        // and the terminal head are singleton, even for T2 prefill.
                        let expected = 8 * rows * intermediate;
                        assert_eq!(quote.compiler_encoded_input_peak_bytes, expected);
                        assert_eq!(
                            serde_json::to_value(&quote).unwrap()
                                ["compiler_encoded_input_peak_bytes"],
                            expected,
                        );
                        let native_delta = quote.resources.extra_native_owner_bytes;
                        let wire_delta = quote.resources.extra_wire_bytes;
                        assert_eq!(
                            quote.client_payload_bytes,
                            decoder.execution_payload_estimate_bytes
                                + native_delta
                                + wire_delta
                                + expected
                        );
                        let scalar_worker = decoder
                            .decoder_resources
                            .as_ref()
                            .map_or(decoder.execution_payload_estimate_bytes, |resources| {
                                resources.worker_payload_bytes
                            });
                        assert_eq!(
                            quote.worker_payload_bytes,
                            scalar_worker + native_delta + wire_delta
                        );
                        assert_eq!(
                            quote.native_admitted_payload_bytes,
                            decoder
                                .native_publication_resources()
                                .unwrap()
                                .admitted_payload_bytes
                                + native_delta
                        );
                        let grouped: Vec<_> = quote
                            .descriptors
                            .iter()
                            .filter(|descriptor| {
                                descriptor
                                    .stage
                                    .is_some_and(|index| decoder.matrices[index].members.len() > 1)
                            })
                            .collect();
                        assert!(!grouped.is_empty());
                        for descriptor in grouped {
                            let matrix = &decoder.matrices[descriptor.stage.unwrap()];
                            assert_eq!(descriptor.columns, matrix.columns);
                            assert_eq!(descriptor.output_width, matrix.rows);
                            assert!(8 * descriptor.rows as usize * descriptor.columns <= expected);
                        }
                        assert_eq!(decoder.canonical_bytes(), scalar_bytes);
                        assert_eq!(decoder.source_binding_digest(), source_digest);
                    }
                }
            }
        }
    }

    #[test]
    fn encoded_input_quote_enforces_existing_client_cap_at_exact_boundary() {
        for capacity in [Capacity::Capsule, Capacity::Decoder] {
            for profile in [SharingProfile::FullShares, SharingProfile::Aes128WindowV1] {
                let mut decoder = pricing_contract(2, 24, capacity);
                let baseline = SourceStageBatchContract::derive(&decoder, profile).unwrap();
                let delta = baseline.resources.extra_native_owner_bytes
                    + baseline.resources.extra_wire_bytes;
                let encoded = baseline.compiler_encoded_input_peak_bytes;
                let limit = if capacity == Capacity::Decoder {
                    GKM_DECODER_EXECUTION_BYTES
                } else {
                    64 << 20
                };
                // Synthetic role-budget boundary; model/source geometry and the
                // actual native quote stay tiny and unchanged. No large owners.
                decoder.execution_payload_estimate_bytes = limit - delta - encoded;
                let exact = SourceStageBatchContract::derive(&decoder, profile).unwrap();
                assert_eq!(exact.client_payload_bytes, limit);
                assert_eq!(
                    exact.native_admitted_payload_bytes,
                    baseline.native_admitted_payload_bytes
                );
                decoder.execution_payload_estimate_bytes += 1;
                assert!(SourceStageBatchContract::derive(&decoder, profile)
                    .err()
                    .unwrap()
                    .contains("exceeds execution payload capacity"));
                // The previously accepted equation can fit exactly while the
                // independently owned compiler encoding now correctly rejects.
                decoder.execution_payload_estimate_bytes = limit - delta;
                assert!(SourceStageBatchContract::derive(&decoder, profile).is_err());
                decoder.execution_payload_estimate_bytes = usize::MAX;
                assert!(SourceStageBatchContract::derive(&decoder, profile)
                    .err()
                    .unwrap()
                    .contains("owner allowance overflow"));
            }
        }
    }

    #[test]
    fn source_stage_batches_cover_t1_t2_and_exact_scalar_window() {
        for rows in [1, 2] {
            for decode_steps in [0, 1, 2] {
                let contract = contract(rows, decode_steps);
                let source_bytes = contract.canonical_bytes();
                let source_digest = contract.source_binding_digest();
                let plan = contract.batch_plan().unwrap();
                let descriptors = plan.descriptors();
                let flattened: Vec<_> = descriptors
                    .iter()
                    .flat_map(|descriptor| {
                        std::iter::repeat_n(descriptor.step, descriptor.rows as usize)
                    })
                    .collect();
                assert_eq!(flattened, contract.window_plan());
                assert_eq!(flattened.len(), contract.window_steps);
                assert_eq!(descriptors.len(), 10 * (1 + decode_steps));
                let mut next_claim = 0;
                let mut boundaries = BTreeSet::new();
                for descriptor in descriptors {
                    assert_eq!(descriptor.first_claim, next_claim);
                    next_claim += descriptor.rows;
                    assert!(boundaries.insert((descriptor.phase_index, descriptor.operation_index)));
                    let phase = if descriptor.phase_index == 0 {
                        &contract.prefill
                    } else {
                        &contract.decode
                    };
                    let operation = &phase.steps[descriptor.operation_index as usize];
                    assert_ne!(operation.executor, DecoderRuntimeExecutor::ClientLocal);
                    assert_eq!(descriptor.operation_index as u64, operation.order);
                    assert_eq!(
                        descriptor.rows,
                        if descriptor.phase_index == 0
                            && operation.operators[0] != ModelOperator::OutputHead
                        {
                            rows as u32
                        } else {
                            1
                        }
                    );
                }
                assert_eq!(next_claim as usize, contract.window_steps);
                assert_eq!(contract.canonical_bytes(), source_bytes);
                assert_eq!(contract.source_binding_digest(), source_digest);
                assert_eq!(contract.plan.prefill.batch, 1);
                assert_eq!(contract.plan.decode.batch, 1);
                assert_eq!(plan.digest(), contract.batch_plan().unwrap().digest());
            }
        }
    }

    #[test]
    fn grouped_source_members_remain_one_batch_not_member_batches() {
        let contract = contract(2, 1);
        let plan = contract.batch_plan().unwrap();
        let mut grouped = 0;
        let mut matrix_index = 0;
        for (operation_index, operation) in contract.prefill.steps.iter().enumerate() {
            if operation.executor == DecoderRuntimeExecutor::ClientLocal
                || operation.operators[0] == ModelOperator::TokenLookup
            {
                continue;
            }
            let matrix = &contract.matrices[matrix_index];
            matrix_index += 1;
            if operation.operation_ids.len() == 1 {
                continue;
            }
            grouped += 1;
            assert_eq!(matrix.members.len(), operation.operation_ids.len());
            let matching: Vec<_> = plan
                .descriptors()
                .iter()
                .filter(|descriptor| {
                    descriptor.phase_index == 0
                        && descriptor.operation_index as usize == operation_index
                })
                .collect();
            assert_eq!(matching.len(), 1);
            assert_eq!(matching[0].rows, 2);
            assert_eq!(matching[0].step, Step::Linear(stage_id(matrix)));
        }
        assert!(grouped >= 4); // Q/K/V and gate/up, across both source layers.
    }

    #[test]
    fn local_reference_grouping_keeps_original_arithmetic_and_final_verifier() {
        use super::super::GkmDecoderSession;
        use std::collections::BTreeMap;
        for seeded in [false, true] {
            let source = contract(2, 1);
            let plan = source.batch_plan().unwrap();
            let tensors: BTreeMap<_, Vec<u8>> = source
                .source_shapes()
                .iter()
                .map(|(name, shape)| {
                    let values = (0..shape.iter().product::<usize>())
                        .flat_map(|index| ((index as f32 % 7.0 - 3.0) / 32.0).to_le_bytes())
                        .collect();
                    (name.clone(), values)
                })
                .collect();
            let borrowed = tensors
                .iter()
                .map(|(name, bytes)| (name.clone(), bytes.as_slice()))
                .collect();
            let mut grouped = GkmDecoderSession::new(source, &borrowed, seeded).unwrap();
            let mut scalar = GkmDecoderSession::new(contract(2, 1), &borrowed, seeded).unwrap();
            for descriptor in plan.descriptors() {
                let rows = descriptor.rows as usize;
                let (actual, expected) = match descriptor.step {
                    Step::Lookup => {
                        assert_eq!(grouped.lookup_batch_rows().unwrap(), rows);
                        assert_eq!(grouped.lookup_batch_rows().unwrap(), rows);
                        let tokens: Vec<_> = (3..3 + rows).collect();
                        let actual = grouped.lookup_batch(&tokens).unwrap();
                        let expected: Vec<u8> = tokens
                            .iter()
                            .flat_map(|token| scalar.lookup(*token).unwrap())
                            .collect();
                        (actual, expected)
                    }
                    Step::Linear(id) => {
                        assert!(grouped.lookup_batch_rows().is_err());
                        let stage = grouped
                            .contract
                            .matrices
                            .iter()
                            .position(|matrix| stage_id(matrix) == id)
                            .unwrap();
                        let columns = grouped.contract.matrices[stage].columns;
                        let input: Vec<u8> = (0..rows * columns)
                            .flat_map(|index| ((index as f32 % 11.0 - 5.0) / 16.0).to_le_bytes())
                            .collect();
                        let actual = grouped.linear_batch(stage, &input).unwrap();
                        let expected: Vec<u8> = input
                            .chunks_exact(columns * 4)
                            .flat_map(|row| scalar.linear(stage, row).unwrap())
                            .collect();
                        (actual, expected)
                    }
                };
                assert_eq!(actual, expected);
            }
            grouped.finish().unwrap();
            assert!(grouped.lookup_batch_rows().is_err());
            scalar.finish().unwrap();
        }
    }

    #[test]
    fn repeated_artifacts_do_not_merge_source_or_decode_boundaries() {
        let mut contract = contract(2, 2);
        // Simulate artifact reuse across distinct source operations. Boundary
        // derivation must depend on the schedule, never equality of Step values.
        let repeated_id = contract.matrices[0].id.clone();
        for matrix in &mut contract.matrices {
            matrix.id = repeated_id.clone();
        }
        let plan = contract.batch_plan().unwrap();
        let descriptors = plan.descriptors();
        assert_eq!(descriptors.len(), 30);
        let prefill_linears: Vec<_> = descriptors
            .iter()
            .filter(|descriptor| descriptor.phase_index == 0 && descriptor.step != Step::Lookup)
            .collect();
        assert_eq!(prefill_linears.len(), 9);
        for pair in prefill_linears.windows(2) {
            assert_eq!(pair[0].step, pair[1].step);
            assert_ne!(pair[0].operation_index, pair[1].operation_index);
        }
        assert_eq!(prefill_linears.last().unwrap().rows, 1);
        for phase_index in 1..=2 {
            let phase: Vec<_> = descriptors
                .iter()
                .filter(|descriptor| descriptor.phase_index == phase_index)
                .collect();
            assert_eq!(phase.len(), 10);
            assert_eq!(phase[0].step, Step::Lookup);
            assert!(phase.iter().all(|descriptor| descriptor.rows == 1));
        }
        // Same operation/artifact across decode iterations remains distinct.
        assert_eq!(
            descriptors[10].operation_index,
            descriptors[20].operation_index
        );
        assert_ne!(descriptors[10].first_claim, descriptors[20].first_claim);
        assert_ne!(descriptors[10].phase_index, descriptors[20].phase_index);
    }
}
