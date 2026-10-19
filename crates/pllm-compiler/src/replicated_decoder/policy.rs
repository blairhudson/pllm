//! Explicit execution choices for the public-weight replicated decoder.
//!
//! The default policy serializes away at the source boundary, preserving existing
//! private/public source and execution identities. Checked ranges are a distinct
//! acceptance contract, not a way to bypass primitive arithmetic admission.

use super::WeightMode;
use serde::{Deserialize, Serialize};

pub const DEFAULT_MAX_ABS_ACTIVATION: u32 = 256;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RangePolicy {
    #[default]
    Static,
    Checked,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LookupPolicy {
    #[default]
    PerToken,
    Tiled,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionPolicy {
    #[default]
    Expanded,
    Tiled,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PeerTransport {
    #[default]
    Http,
    Streams,
}

/// Public, immutable choices. Bounds are in real integer units; native arithmetic
/// derives their exact Q18 endpoints without reading a private value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionPolicy {
    pub range_policy: RangePolicy,
    pub max_abs_activation: u32,
    pub lookup: LookupPolicy,
    #[serde(default)]
    pub selection: SelectionPolicy,
    pub peer_transport: PeerTransport,
}

impl Default for ExecutionPolicy {
    fn default() -> Self {
        Self {
            range_policy: RangePolicy::Static,
            max_abs_activation: DEFAULT_MAX_ABS_ACTIVATION,
            lookup: LookupPolicy::PerToken,
            selection: SelectionPolicy::Expanded,
            peer_transport: PeerTransport::Http,
        }
    }
}

impl ExecutionPolicy {
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    pub fn validate(&self, weights: WeightMode) -> Result<(), String> {
        if !(1..=1024).contains(&self.max_abs_activation)
            || self.range_policy == RangePolicy::Static
                && self.max_abs_activation != DEFAULT_MAX_ABS_ACTIVATION
            || weights == WeightMode::Private && !self.is_default()
        {
            return Err("invalid replicated public execution policy".into());
        }
        Ok(())
    }

    pub(super) fn checked_bound(&self) -> Option<i64> {
        (self.range_policy == RangePolicy::Checked)
            .then_some(i64::from(self.max_abs_activation) * super::ONE)
    }

    /// Canonical component parameters omit every legacy default. Native admission
    /// compares these exact values, rejecting redundant or unknown selectors.
    pub fn component_params(&self, weights: WeightMode) -> serde_json::Value {
        let mut params = serde_json::Map::new();
        if weights == WeightMode::Public {
            params.insert("weights".into(), serde_json::json!("public"));
        }
        if self.range_policy == RangePolicy::Checked {
            params.insert("range_policy".into(), serde_json::json!("checked"));
            params.insert(
                "max_abs_activation".into(),
                serde_json::json!(self.max_abs_activation),
            );
        }
        if self.lookup == LookupPolicy::Tiled {
            params.insert("lookup".into(), serde_json::json!("tiled"));
        }
        if self.selection == SelectionPolicy::Tiled {
            params.insert("selection".into(), serde_json::json!("tiled"));
        }
        if self.peer_transport == PeerTransport::Streams {
            params.insert("peer_transport".into(), serde_json::json!("streams"));
        }
        params.into()
    }

    pub fn from_component(value: serde_json::Value) -> Result<(WeightMode, Self), String> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Parameters {
            #[serde(default)]
            weights: WeightMode,
            #[serde(default)]
            range_policy: RangePolicy,
            max_abs_activation: Option<u32>,
            #[serde(default)]
            lookup: LookupPolicy,
            #[serde(default)]
            selection: SelectionPolicy,
            #[serde(default)]
            peer_transport: PeerTransport,
        }
        let parsed: Parameters = serde_json::from_value(value.clone())
            .map_err(|_| "invalid replicated execution parameters")?;
        let policy = Self {
            range_policy: parsed.range_policy,
            max_abs_activation: parsed
                .max_abs_activation
                .unwrap_or(DEFAULT_MAX_ABS_ACTIVATION),
            lookup: parsed.lookup,
            selection: parsed.selection,
            peer_transport: parsed.peer_transport,
        };
        policy.validate(parsed.weights)?;
        if policy.component_params(parsed.weights) != value {
            return Err("noncanonical replicated execution parameters".into());
        }
        Ok((parsed.weights, policy))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn legacy_parameters_preserve_default_policy() {
        for (weights, value) in [
            (WeightMode::Private, json!({})),
            (WeightMode::Public, json!({"weights":"public"})),
        ] {
            let (actual, policy) = ExecutionPolicy::from_component(value).unwrap();
            assert_eq!(actual, weights);
            assert!(policy.is_default());
            assert_eq!(policy.checked_bound(), None);
        }
    }

    #[test]
    fn checked_policy_binds_public_bound_and_independent_execution_choices() {
        let value = json!({"weights":"public", "range_policy":"checked",
            "max_abs_activation":128, "lookup":"tiled", "peer_transport":"streams"});
        let (weights, policy) = ExecutionPolicy::from_component(value.clone()).unwrap();
        assert_eq!(policy.checked_bound(), Some(128 << 18));
        assert_eq!(policy.component_params(weights), value);
        assert!(!policy.is_default());
    }

    #[test]
    fn tiled_selection_is_explicit_public_and_independent() {
        let value = json!({"weights":"public", "selection":"tiled"});
        let (weights, policy) = ExecutionPolicy::from_component(value.clone()).unwrap();
        assert_eq!(policy.selection, SelectionPolicy::Tiled);
        assert_eq!(policy.lookup, LookupPolicy::PerToken);
        assert_eq!(policy.component_params(weights), value);
        assert!(policy.validate(WeightMode::Private).is_err());
        let encoded = serde_json::to_value(policy).unwrap();
        assert_eq!(encoded["selection"], "tiled");
        let expanded = ExecutionPolicy {
            lookup: LookupPolicy::Tiled,
            ..ExecutionPolicy::default()
        };
        assert_eq!(
            serde_json::to_value(expanded).unwrap()["selection"],
            "expanded"
        );
        assert!(expanded
            .component_params(weights)
            .get("selection")
            .is_none());
    }

    #[test]
    fn rejected_policy_substitutions_do_not_change_legacy_contract() {
        for value in [
            json!({"range_policy":"checked", "max_abs_activation":256}),
            json!({"weights":"public", "range_policy":"checked"}),
            json!({"weights":"public", "range_policy":"checked", "max_abs_activation":0}),
            json!({"weights":"public", "range_policy":"checked", "max_abs_activation":1025}),
            json!({"weights":"public", "range_policy":"checked", "max_abs_activation":true}),
            json!({"weights":"public", "range_policy":"static"}),
            json!({"weights":"public", "max_abs_activation":256}),
            json!({"weights":"public", "lookup":"secret_index"}),
            json!({"selection":"tiled"}),
            json!({"weights":"public", "selection":"expanded"}),
            json!({"weights":"public", "selection":"pruned"}),
            json!({"weights":"public", "peer_transport":"http"}),
            json!({"weights":"public", "unchecked":true}),
        ] {
            assert!(ExecutionPolicy::from_component(value).is_err());
        }
    }
}
