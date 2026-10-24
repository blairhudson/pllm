//! Server-only resource permission. Not source/configuration authentication.
//! The server adapter supplies its authenticated Experiment and admitted config;
//! no Manifest field is a policy input. Original source issuance stays unchanged.
use super::{Error, ExecutionPolicy, RangePolicy, WeightMode};
#[cfg(test)]
use crate::ExperimentDocument;
use crate::{original_source, plan_input::json};
use pllm_garble::shared_allocation::{OwnedCharged, MAX_ALLOCATION_BYTES};
use pllm_models::{ConfigLowererLimits, DecoderExpansionBudget, DecoderWorkload};

pub(super) const CONFIG_BYTES: usize = 16 << 10;
pub(super) const PUBLIC_ELEMENTS: u64 = 1 << 29;
pub(super) const ORDINARY_LANES: usize = 1 << 20;
pub(super) const DESCRIPTORS: usize = 65536;
pub(super) const SOURCE_WORK: u64 = 1_000_000_000_000;

/// Borrowed permission tied to the exact SERVER config owner, not its digest.
/// No Clone/Deserialize or constructor accepting a client policy/Manifest.
pub(super) struct ServerSourcePolicy<'a> {
    configuration: &'a OwnedCharged<json::ImportedValue>,
    workload: DecoderWorkload,
    limits: original_source::Limits,
}

impl<'a> ServerSourcePolicy<'a> {
    /// Call only after the existing server Experiment authentication/validation.
    /// This allocation-free function neither performs nor claims authentication.
    #[cfg(test)]
    pub(super) fn checked_public(
        server: &ExperimentDocument,
        configuration: &'a OwnedCharged<json::ImportedValue>,
        workload: DecoderWorkload,
        weights: WeightMode,
        execution: ExecutionPolicy,
    ) -> Result<Self, Error> {
        Self::bounded(
            server.budget.max_input_tokens,
            server.budget.max_new_tokens,
            configuration,
            workload,
            weights,
            execution,
        )
    }

    fn bounded(
        max_input: u64,
        max_output: u64,
        configuration: &'a OwnedCharged<json::ImportedValue>,
        workload: DecoderWorkload,
        weights: WeightMode,
        execution: ExecutionPolicy,
    ) -> Result<Self, Error> {
        if weights != WeightMode::Public
            || execution.range_policy != RangePolicy::Checked
            || !(1..=1024).contains(&execution.max_abs_activation)
            || workload.batch != 1
            || workload.max_input_tokens == 0
            || workload.max_new_tokens == 0
            || workload.max_new_tokens > 8
            || workload
                .max_input_tokens
                .checked_add(workload.max_new_tokens)
                .is_none_or(|n| n > 40)
            || workload.max_input_tokens > max_input
            || workload.max_new_tokens > max_output
            || configuration.payload().raw().is_empty()
            || configuration.payload().raw().len() > CONFIG_BYTES
        {
            return Err(Error::Bounds);
        }
        let document = configuration.payload().value();
        // SAME registered scalar preflight/defaults and phase-count formula.
        // The production caller prepays O*SOURCE_SCALAR_SUMMARY_WORK first.
        let summary =
            pllm_models::source_scalar_summary(document, workload).map_err(|_| Error::Bounds)?;
        if summary.operations_per_phase > DESCRIPTORS
            || summary.semantic_elements() > ORDINARY_LANES as u64
        {
            return Err(Error::Bounds);
        }
        // Permission covers this request's EXISTING feedback prefixes, not the
        // server's maximum client-input rows. Reuse the registered seven-product
        // summary for each (R+s,O-s), with no descriptors or tensor allocation.
        let mut max_tensor_elements = summary.semantic_elements();
        for step in 1..workload.max_new_tokens {
            let next = DecoderWorkload {
                max_input_tokens: workload
                    .max_input_tokens
                    .checked_add(step)
                    .ok_or(Error::Bounds)?,
                max_new_tokens: workload
                    .max_new_tokens
                    .checked_sub(step)
                    .ok_or(Error::Bounds)?,
                ..workload
            };
            let next =
                pllm_models::source_scalar_summary(document, next).map_err(|_| Error::Bounds)?;
            if next.operations_per_phase > DESCRIPTORS
                || next.semantic_elements() > ORDINARY_LANES as u64
            {
                return Err(Error::Bounds);
            }
            max_tensor_elements = max_tensor_elements.max(next.semantic_elements());
        }
        Ok(Self {
            configuration,
            workload,
            limits: original_source::Limits {
                expansion: DecoderExpansionBudget {
                    max_layers: summary.layers,
                    max_feature_width: summary.feature_width(),
                    max_attention_heads: summary.attention_heads.max(summary.kv_heads),
                    max_vocabulary: summary.vocabulary,
                    max_tensor_elements,
                },
                factory: ConfigLowererLimits {
                    max_operations_per_phase: summary.operations_per_phase,
                    max_string_bytes: 1024,
                    max_attribute_nodes: 32768,
                    max_retained_strings: 1 << 20,
                    max_factory_bytes: MAX_ALLOCATION_BYTES,
                },
                max_work: SOURCE_WORK,
                max_working_payload: MAX_ALLOCATION_BYTES,
            },
        })
    }

    /// Resolve before moving this exact admitted config into original_source.
    /// These returned scalars are permissions, never construction authority.
    pub(super) fn limits_for(
        &self,
        configuration: &OwnedCharged<json::ImportedValue>,
        workload: DecoderWorkload,
    ) -> Result<original_source::Limits, Error> {
        if !std::ptr::eq(self.configuration, configuration)
            || workload.batch != self.workload.batch
            || workload.max_input_tokens != self.workload.max_input_tokens
            || workload.max_new_tokens != self.workload.max_new_tokens
        {
            return Err(Error::Binding);
        }
        Ok(self.limits)
    }

    /// Server bytes came through the SAME admitted C9 importer, not Manifest.
    /// This checks RESOURCE permission only; canonical server authentication and
    /// original public source snapshot verification are independent obligations.
    pub(super) fn from_server_json(
        server: &OwnedCharged<json::ImportedValue>,
        configuration: &'a OwnedCharged<json::ImportedValue>,
        manifest: Option<&serde_json::Value>,
        workload: DecoderWorkload,
    ) -> Result<Self, Error> {
        if !std::ptr::eq(server.budget(), configuration.budget())
            || server.payload().raw().len() > 32 << 10
        {
            return Err(Error::Binding);
        }
        let server = server.payload().value();
        let params = &server["pipeline"]["components"]["linear"]["params"];
        if params["weights"].as_str() != Some("public")
            || params["range_policy"].as_str() != Some("checked")
        {
            return Err(Error::Binding);
        }
        let activation = params
            .get("max_abs_activation")
            .map_or(
                Some(u64::from(ExecutionPolicy::default().max_abs_activation)),
                |v| v.as_u64(),
            )
            .and_then(|n| u32::try_from(n).ok())
            .ok_or(Error::Binding)?;
        if let Some(manifest) = manifest {
            if manifest["weights"].as_str() != Some("public")
                || manifest["execution"]["range_policy"].as_str() != Some("checked")
                || manifest["execution"]["max_abs_activation"].as_u64()
                    != Some(u64::from(activation))
            {
                return Err(Error::Binding);
            }
        }
        Self::bounded(
            server["budget"]["max_input_tokens"]
                .as_u64()
                .ok_or(Error::Binding)?,
            server["budget"]["max_new_tokens"]
                .as_u64()
                .ok_or(Error::Binding)?,
            configuration,
            workload,
            WeightMode::Public,
            ExecutionPolicy {
                range_policy: RangePolicy::Checked,
                max_abs_activation: activation,
                ..ExecutionPolicy::default()
            },
        )
    }
}

// Prepay the entire bounded summary loop BEFORE its first scalar visit. No
// failed/short request receives a partial-summary refund or a fresh source cap.
fn prepay_summaries(prior: u128, outputs: u64) -> Result<u64, Error> {
    if !(1..=8).contains(&outputs) {
        return Err(Error::Bounds);
    }
    let summary = outputs
        .checked_mul(pllm_models::SOURCE_SCALAR_SUMMARY_WORK)
        .ok_or(Error::Bounds)?;
    let prior = prior
        .checked_add(u128::from(summary))
        .and_then(|n| u64::try_from(n).ok())
        .ok_or(Error::Bounds)?;
    SOURCE_WORK.checked_sub(prior).ok_or(Error::Bounds)
}

/// Actual immutable owners, not a caller numeric prior-work baseline. Both the
/// real constructor and metadata-only query use this SAME nonrefundable debit.
pub(super) fn admitted_limits(
    server: &OwnedCharged<json::ImportedValue>,
    config: &OwnedCharged<json::ImportedValue>,
    manifest: Option<&OwnedCharged<json::ImportedValue>>,
    workload: DecoderWorkload,
) -> Result<original_source::Limits, Error> {
    if !std::ptr::eq(server.budget(), config.budget()) {
        return Err(Error::Binding);
    }
    if ORDINARY_LANES != super::MAX_VALUES
        || PUBLIC_ELEMENTS != super::MAX_PUBLIC_SOURCE_VALUES as u64
    {
        return Err(Error::Bounds);
    }
    let manifest_work = match manifest {
        Some(m) => {
            if !std::ptr::eq(m.budget(), config.budget()) {
                return Err(Error::Binding);
            }
            if m.payload().value()["tensors"]
                .as_object()
                .ok_or(Error::Binding)?
                .len()
                > DESCRIPTORS
            {
                return Err(Error::Bounds);
            }
            m.payload().resources().work.total
        }
        None => 0,
    };
    // Full original C9 server import + a second full borrowed-policy envelope;
    // Manifest C9 work and exact byte comparisons. Logical visits, not std CPU.
    let prior = server
        .payload()
        .resources()
        .work
        .total
        .checked_mul(2)
        .and_then(|n| n.checked_add(manifest_work))
        .and_then(|n| n.checked_add(config.payload().raw().len() as u128))
        .ok_or(Error::Bounds)?;
    // Prepay BEFORE the shared scalar summary. Its permission stays reduced for
    // the first factory and all subsequent source-root epochs, no credit/reset.
    let remaining = prepay_summaries(prior, workload.max_new_tokens)?;
    let policy = ServerSourcePolicy::from_server_json(
        server,
        config,
        manifest.map(|m| m.payload().value()),
        workload,
    )?;
    let mut limits = policy.limits_for(config, workload)?;
    limits.max_work = remaining;
    Ok(limits)
}

#[cfg(test)]
#[path = "source_policy_tests.rs"]
mod tests;
