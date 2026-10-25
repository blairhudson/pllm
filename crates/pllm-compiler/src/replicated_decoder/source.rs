use super::*;
#[path = "source/compact_publisher.rs"]
mod compact_publisher;
#[path = "source/compact_publisher_scan.rs"]
mod compact_publisher_scan;

// Layout evidence for the original begin owner's precollection quote. Not a
// reservation, factory permission, complete publisher peak or native API.
#[cfg_attr(not(test), allow(dead_code))] // Staging until original Header owner joins.
pub(super) const fn compact_publisher_scan_bytes() -> usize {
    compact_publisher_scan::Scan::inline_peak_bytes()
}

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use zeroize::Zeroize;

const CHUNK: usize = 4096;

pub(super) fn encode_q18(raw: &[u8]) -> Result<i64, String> {
    let value = f32::from_le_bytes(raw.try_into().map_err(|_| "invalid float32 source word")?);
    let scaled = value * ONE as f32;
    if !scaled.is_finite() || scaled.abs() > (1u64 << 40) as f32 {
        return Err("replicated Q18 source range".into());
    }
    Ok(scaled as i64)
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema: String,
    #[serde(default, skip_serializing_if = "WeightMode::is_private")]
    pub weights: WeightMode,
    #[serde(default, skip_serializing_if = "ExecutionPolicy::is_default")]
    pub execution: ExecutionPolicy,
    pub config: Vec<u8>,
    pub rows: usize,
    pub outputs: usize,
    pub source: [u8; 32],
    pub composition: [u8; 32],
    pub session: [u8; 32],
    pub tensors: BTreeMap<String, SourceTensor>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub norms: SourceNorms,
    pub program: [u8; 32],
}

impl Manifest {
    pub fn compile(&self) -> Result<Arc<Program>, String> {
        if self.schema != "pllm.replicated-decoder.manifest.v1"
            || self.source == [0; 32]
            || self.composition == [0; 32]
            || self.session == [0; 32]
        {
            return Err("invalid replicated manifest identity".into());
        }
        let mut program = Program::compile_with_norms(
            &self.config,
            self.rows,
            self.outputs,
            &self.tensors,
            self.weights,
            self.execution,
            &self.norms,
        )?;
        program.bind(self.source, self.composition);
        if program.digest() != self.program {
            return Err("replicated manifest program differs".into());
        }
        Ok(Arc::new(program))
    }
}

pub struct SourceChunk {
    pub key: String,
    pub offset: usize,
    pub shares: [Zeroizing<Vec<u8>>; 3],
}

/// Trusted source/input owner. Bounded F32 ingestion, Q18 encoding, exact-byte
/// commitments, uniform fresh replicated components, and consuming final reveal.
pub struct Publisher {
    compact: Option<Box<compact_publisher::CompactPublisher>>,
    manifest: Manifest,
    shapes: Vec<(String, Vec<usize>)>,
    values: Vec<Zeroizing<Vec<i64>>>,
    cursor: usize,
    offset: usize,
    program: Option<Arc<Program>>,
    issued: usize,
    issue_offset: usize,
    retired: bool,
    public_coefficients: std::collections::BTreeSet<String>,
    source_hash: Sha256,
    source_lower: i64,
    source_upper: i64,
    row_sum: u64,
    row_l1: u64,
    row_square_sum: u128,
    row_squared_norm: u128,
    row_l1_squared_sum: u128,
    norm_requests: BTreeMap<String, (usize, usize, usize)>,
    norm_scan: Option<source_norms::BlockScan>,
}

impl Publisher {
    /// Private SDK opt-in: actual CLIENT Experiment is rechecked by original
    /// precollection ingress. Not a worker Manifest policy selector.
    #[doc(hidden)]
    pub fn from_compact_source_request(
        experiment: &[u8],
        config: &[u8],
        rows: usize,
        outputs: usize,
        source: [u8; 32],
        composition: [u8; 32],
    ) -> Result<Self, String> {
        let compact = compact_publisher::CompactPublisher::new(
            experiment,
            config,
            rows,
            outputs,
            source,
            composition,
        )?;
        // Empty legacy shell has no source data/authority or factory side effect.
        // Its inline/boxing transition is included in original collection lease.
        Ok(Self {
            compact: Some(Box::new(compact)),
            manifest: Manifest {
                schema: String::new(),
                weights: WeightMode::Public,
                execution: ExecutionPolicy::default(),
                config: Vec::new(),
                rows: 0,
                outputs: 0,
                source: [0; 32],
                composition: [0; 32],
                session: [0; 32],
                tensors: BTreeMap::new(),
                norms: BTreeMap::new(),
                program: [0; 32],
            },
            shapes: Vec::new(),
            values: Vec::new(),
            cursor: 0,
            offset: 0,
            program: None,
            issued: 0,
            issue_offset: 0,
            retired: false,
            public_coefficients: std::collections::BTreeSet::new(),
            source_hash: Sha256::new(),
            source_lower: i64::MAX,
            source_upper: i64::MIN,
            row_sum: 0,
            row_l1: 0,
            row_square_sum: 0,
            row_squared_norm: 0,
            row_l1_squared_sum: 0,
            norm_requests: BTreeMap::new(),
            norm_scan: None,
        })
    }
    pub fn storage_kind(&self) -> Option<&'static str> {
        if let Some(c) = &self.compact {
            c.storage_kind()
        } else {
            self.program.as_ref().map(|p| p.storage_kind())
        }
    }
    /// Whole client raw/Q18 snapshots + compiler import snapshot. Outer SDK must
    /// hold actual disk admission through handle retirement; this is no lease.
    pub fn collection_disk_bytes(&self) -> usize {
        self.compact.as_ref().map_or(0, |c| c.disk_bytes())
    }
    /// Actual retained collection filesystem; no path allocation or disk lease.
    /// Expanded/default publication is unchanged and has no such location.
    pub fn collection_disk_directory(&self) -> Result<&std::path::Path, String> {
        if self.retired {
            return Err("replicated publisher is retired".into());
        }
        self.compact
            .as_ref()
            .ok_or("publisher has no compact collection directory")?
            .directory()
    }
    pub fn new(
        config: &[u8],
        rows: usize,
        outputs: usize,
        source: [u8; 32],
        composition: [u8; 32],
    ) -> Result<Self, String> {
        Self::new_with_weights(
            config,
            rows,
            outputs,
            source,
            composition,
            WeightMode::Private,
        )
    }

    pub fn new_with_weights(
        config: &[u8],
        rows: usize,
        outputs: usize,
        source: [u8; 32],
        composition: [u8; 32],
        weights: WeightMode,
    ) -> Result<Self, String> {
        Self::new_with_policy(
            config,
            rows,
            outputs,
            source,
            composition,
            weights,
            ExecutionPolicy::default(),
        )
    }

    pub fn new_with_policy(
        config: &[u8],
        rows: usize,
        outputs: usize,
        source: [u8; 32],
        composition: [u8; 32],
        weights: WeightMode,
        execution: ExecutionPolicy,
    ) -> Result<Self, String> {
        execution.validate(weights)?;
        let shapes: Vec<_> =
            Program::source_shapes_with_policy(config, rows, outputs, weights, execution)?
                .into_iter()
                .collect();
        let plan = program::lower_with_weights(config, rows, outputs, weights)?;
        let norm_requests = if execution.range_policy == RangePolicy::Checked {
            source_norms::requests(&plan.prefill).map_err(|e| e.to_string())?
        } else {
            BTreeMap::new()
        };
        source_norms::requests_envelope(&norm_requests).map_err(|e| e.to_string())?;
        let public_coefficients: std::collections::BTreeSet<String> = plan
            .prefill
            .operations
            .iter()
            .filter(|op| op.operator == pllm_models::ModelOperator::RmsNorm)
            .map(|op| op.attributes["weight"].as_str().unwrap().to_owned())
            .collect();
        if source == [0; 32] || composition == [0; 32] {
            return Err("empty source/composition pin".into());
        }
        let mut session = [0; 32];
        getrandom::fill(&mut session).map_err(|_| "replicated OS randomness failed")?;
        let values = shapes
            .iter()
            .map(|(key, shape)| {
                Ok(Zeroizing::new(
                    vec_zero(
                        if weights == WeightMode::Private || public_coefficients.contains(key) {
                            shape.iter().product()
                        } else {
                            0
                        },
                    )
                    .map_err(|e| e.to_string())?,
                ))
            })
            .collect::<Result<_, String>>()?;
        Ok(Self {
            compact: None,
            manifest: Manifest {
                schema: "pllm.replicated-decoder.manifest.v1".into(),
                weights,
                execution,
                config: config.to_vec(),
                rows,
                outputs,
                source,
                composition,
                session,
                tensors: BTreeMap::new(),
                norms: BTreeMap::new(),
                program: [0; 32],
            },
            shapes,
            values,
            cursor: 0,
            offset: 0,
            program: None,
            issued: 0,
            issue_offset: 0,
            retired: false,
            public_coefficients,
            source_hash: Sha256::new(),
            source_lower: i64::MAX,
            source_upper: i64::MIN,
            row_sum: 0,
            row_l1: 0,
            row_square_sum: 0,
            row_squared_norm: 0,
            row_l1_squared_sum: 0,
            norm_requests,
            norm_scan: None,
        })
    }
    pub fn shapes(&self) -> &[(String, Vec<usize>)] {
        if let Some(c) = &self.compact {
            return c.shapes();
        }
        &self.shapes
    }
    pub fn cancel(&mut self) {
        if let Some(mut c) = self.compact.take() {
            c.cancel();
        }
        self.retired = true;
        self.values.clear();
        self.program = None;
        self.norm_scan = None;
    }
    fn checked<T>(&mut self, result: Result<T, String>) -> Result<T, String> {
        if result.is_err() {
            self.cancel();
        }
        result
    }
    pub fn append(&mut self, key: &str, offset: usize, data: &[u8]) -> Result<(), String> {
        if let Some(c) = &mut self.compact {
            let result = c.append(key, offset, data);
            return self.checked(result);
        }
        let result = (|| {
            if self.retired
                || self.program.is_some()
                || self.cursor >= self.shapes.len()
                || key != self.shapes[self.cursor].0
                || offset != self.offset
                || data.is_empty()
                || data.len() > CHUNK * 4
                || data.len() % 4 != 0
                || offset
                    .checked_add(data.len() / 4)
                    .is_none_or(|n| n > self.shapes[self.cursor].1.iter().product())
            {
                return Err("invalid ordered replicated source frame".into());
            }
            if offset == 0 {
                self.norm_scan = self
                    .norm_requests
                    .get(key)
                    .map(|&(rows, columns, block_rows)| {
                        source_norms::BlockScan::new(rows, columns, block_rows)
                    })
                    .transpose()
                    .map_err(|e| e.to_string())?;
            }
            for (i, raw) in data.chunks_exact(4).enumerate() {
                let value = encode_q18(raw)?;
                if let Some(scan) = &mut self.norm_scan {
                    scan.append(offset + i, std::slice::from_ref(&value))
                        .map_err(|e| e.to_string())?;
                }
                if !self.values[self.cursor].is_empty() {
                    self.values[self.cursor][offset + i] = value;
                }
                self.source_hash.update(value.to_le_bytes());
                self.source_lower = self.source_lower.min(value);
                self.source_upper = self.source_upper.max(value);
                if self.manifest.weights == WeightMode::Public
                    && self.shapes[self.cursor].1.len() == 2
                {
                    self.row_sum = self
                        .row_sum
                        .checked_add(value.unsigned_abs())
                        .ok_or("public row bound overflow")?;
                    self.row_square_sum = self
                        .row_square_sum
                        .checked_add(u128::from(value.unsigned_abs()).pow(2))
                        .ok_or("public row square bound overflow")?;
                    if (offset + i + 1) % self.shapes[self.cursor].1[1] == 0 {
                        self.row_l1 = self.row_l1.max(self.row_sum);
                        self.row_l1_squared_sum = self
                            .row_l1_squared_sum
                            .checked_add(u128::from(self.row_sum).pow(2))
                            .ok_or("public row energy overflow")?;
                        self.row_sum = 0;
                        self.row_squared_norm = self.row_squared_norm.max(self.row_square_sum);
                        self.row_square_sum = 0;
                    }
                }
            }
            self.offset += data.len() / 4;
            if self.offset == self.shapes[self.cursor].1.iter().product::<usize>() {
                let (key, shape) = &self.shapes[self.cursor];
                if let Some(scan) = self.norm_scan.take() {
                    self.manifest
                        .norms
                        .insert(key.clone(), scan.finish().map_err(|e| e.to_string())?);
                    source_norms::metadata_bytes(&self.manifest.norms)
                        .map_err(|e| e.to_string())?;
                }
                self.manifest.tensors.insert(
                    key.clone(),
                    SourceTensor {
                        shape: shape.clone(),
                        lower: self.source_lower,
                        upper: self.source_upper,
                        digest: std::mem::take(&mut self.source_hash).finalize().into(),
                        public_q18: self
                            .public_coefficients
                            .contains(key)
                            .then(|| self.values[self.cursor].to_vec()),
                        row_l1: (self.manifest.weights == WeightMode::Public && shape.len() == 2)
                            .then_some(self.row_l1),
                        row_squared_norm: (self.manifest.weights == WeightMode::Public
                            && shape.len() == 2)
                            .then_some(self.row_squared_norm),
                        row_l1_squared_sum: (self.manifest.weights == WeightMode::Public
                            && shape.len() == 2)
                            .then_some(self.row_l1_squared_sum),
                    },
                );
                self.source_lower = i64::MAX;
                self.source_upper = i64::MIN;
                self.row_sum = 0;
                self.row_l1 = 0;
                self.row_square_sum = 0;
                self.row_squared_norm = 0;
                self.row_l1_squared_sum = 0;
                self.cursor += 1;
                self.offset = 0;
            }
            Ok(())
        })();
        self.checked(result)
    }
    pub fn seal(&mut self, ids: &[u32]) -> Result<(), String> {
        if let Some(c) = &mut self.compact {
            let result = c.seal(ids);
            return self.checked(result);
        }
        let result = (|| {
            if self.retired
                || self.program.is_some()
                || self.cursor != self.shapes.len()
                || ids.len() != self.manifest.rows
            {
                return Err("incomplete replicated source import".into());
            }
            let plan = program::lower_with_weights(
                &self.manifest.config,
                self.manifest.rows,
                self.manifest.outputs,
                self.manifest.weights,
            )?;
            let public: std::collections::BTreeSet<_> = plan
                .prefill
                .operations
                .iter()
                .filter(|op| op.operator == pllm_models::ModelOperator::RmsNorm)
                .map(|op| op.attributes["weight"].as_str().unwrap())
                .collect();
            let vocabulary = plan
                .prefill
                .operations
                .iter()
                .find(|op| op.operator == pllm_models::ModelOperator::OutputHead)
                .unwrap()
                .output_shape[1];
            if ids.iter().any(|v| u64::from(*v) >= vocabulary) {
                return Err("replicated input ID outside vocabulary".into());
            }
            let mut program = Program::compile_with_norms(
                &self.manifest.config,
                self.manifest.rows,
                self.manifest.outputs,
                &self.manifest.tensors,
                self.manifest.weights,
                self.manifest.execution,
                &self.manifest.norms,
            )?;
            program.bind(self.manifest.source, self.manifest.composition);
            self.manifest.program = program.digest();
            let mut values = vec![];
            for ((key, _), value) in self.shapes.iter().zip(std::mem::take(&mut self.values)) {
                if self.manifest.weights == WeightMode::Private && !public.contains(key.as_str()) {
                    values.push(value);
                }
            }
            values.push(Zeroizing::new(ids.iter().map(|v| i64::from(*v)).collect()));
            self.values = values;
            self.program = Some(Arc::new(program));
            Ok(())
        })();
        self.checked(result)
    }
    pub fn manifest(&self) -> Result<Vec<u8>, String> {
        if self.compact.is_some() {
            return Err("compact manifest requires manifest_checked".into());
        }
        if self.retired || self.program.is_none() {
            return Err("replicated publisher not sealed".into());
        }
        serde_json::to_vec(&self.manifest)
            .map_err(|_| "replicated manifest serialization failed".into())
    }
    /// Same wire Manifest; mutable receiver admits compact serialization first.
    pub fn manifest_checked(&mut self) -> Result<Vec<u8>, String> {
        if let Some(c) = &mut self.compact {
            let result = c.manifest();
            return self.checked(result);
        }
        self.manifest()
    }
    pub fn next_chunk(&mut self) -> Result<Option<SourceChunk>, String> {
        if let Some(c) = &mut self.compact {
            let result = c.next_chunk();
            return self.checked(result);
        }
        let result = (|| {
            if self.retired {
                return Err("retired replicated publisher".into());
            }
            let program = self
                .program
                .as_ref()
                .ok_or("unsealed replicated publisher")?;
            if self.issued == self.values.len() {
                return Ok(None);
            }
            let key = program
                .inputs()
                .nth(self.issued)
                .ok_or("replicated source order")?
                .0
                .to_owned();
            let values = &self.values[self.issued];
            let offset = self.issue_offset;
            let n = CHUNK.min(values.len() - offset);
            let chunk = source_chunk(key, offset, &values[offset..offset + n])?;
            self.issue_offset += n;
            if self.issue_offset == values.len() {
                self.values[self.issued].zeroize();
                self.values[self.issued].shrink_to_fit();
                self.issued += 1;
                self.issue_offset = 0;
            }
            Ok(Some(chunk))
        })();
        self.checked(result)
    }
    pub fn finish(self, frames: [&[u8]; 3]) -> Result<Vec<u32>, String> {
        match self.finish_outcome(frames)? {
            FinalOutcome::Accepted(tokens) => Ok(tokens),
            FinalOutcome::NumericRejected | FinalOutcome::MalformedAcceptance => {
                Err("replicated request failed protected range checks".into())
            }
        }
    }

    /// None means only an authenticated, well-formed aggregate numeric rejection.
    /// Malformed acceptance words/outputs remain errors, never numeric outcomes.
    /// This consumes and retires the publisher on every path.
    pub fn finish_numeric_outcome(self, frames: [&[u8]; 3]) -> Result<Option<Vec<u32>>, String> {
        match self.finish_outcome(frames)? {
            FinalOutcome::Accepted(tokens) => Ok(Some(tokens)),
            FinalOutcome::NumericRejected => Ok(None),
            FinalOutcome::MalformedAcceptance => {
                Err("replicated final acceptance word or masked output malformed".into())
            }
        }
    }

    fn finish_outcome(mut self, frames: [&[u8]; 3]) -> Result<FinalOutcome, String> {
        if let Some(c) = self.compact.take() {
            return c.finish_outcome(frames);
        }
        if self.retired || self.program.is_none() || self.issued != self.values.len() {
            return Err("incomplete replicated transaction".into());
        }
        let outcome = finish_manifest_outcome(&self.manifest, self.manifest.program, frames, None)?;
        self.cancel();
        Ok(outcome)
    }
}

// Shared acceptance equations/error order for both storage routes. Compact
// vocabulary comes from the genuine original source plan, not a caller token.
fn finish_manifest_outcome(
    manifest: &Manifest,
    digest: [u8; 32],
    frames: [&[u8]; 3],
    vocabulary: Option<usize>,
) -> Result<FinalOutcome, String> {
    let checked = manifest.execution.range_policy == RangePolicy::Checked;
    let n = manifest.outputs + usize::from(checked);
    let mut result = Zeroizing::new(vec_zero::<u64>(n).map_err(|e| e.to_string())?);
    for (party, frame) in frames.into_iter().enumerate() {
        if frame.len() != 73 + 8 * n
            || &frame[..4] != b"PUR1"
            || frame[4..36] != manifest.session
            || frame[36..68] != digest
            || frame[68] != party as u8
            || u32::from_le_bytes(frame[69..73].try_into().unwrap()) as usize != n
        {
            return Err("replicated final share identity mismatch".into());
        }
        for (sum, raw) in result.iter_mut().zip(frame[73..].chunks_exact(8)) {
            *sum = sum.wrapping_add(u64::from_le_bytes(raw.try_into().unwrap()));
        }
    }
    if checked {
        match result.pop() {
            Some(0) if result.iter().all(|word| *word == 0) => {
                return Ok(FinalOutcome::NumericRejected);
            }
            Some(1) => (),
            _ => return Ok(FinalOutcome::MalformedAcceptance),
        }
    }
    let vocabulary = if let Some(v) = vocabulary {
        v as u64
    } else {
        let config: serde_json::Value =
            serde_json::from_slice(&manifest.config).map_err(|_| "invalid source config")?;
        config.get("text_config").unwrap_or(&config)["vocab_size"]
            .as_u64()
            .ok_or("missing vocabulary")?
    };
    if result.iter().any(|v| *v >= vocabulary) {
        return Err("replicated final token range".into());
    }
    Ok(FinalOutcome::Accepted(
        result.iter().map(|v| *v as u32).collect(),
    ))
}
enum FinalOutcome {
    Accepted(Vec<u32>),
    NumericRejected,
    MalformedAcceptance,
}

// Original replicated component equations and exact random-byte draw count.
fn source_chunk(key: String, offset: usize, values: &[i64]) -> Result<SourceChunk, String> {
    let n = values.len();
    let mut random = Zeroizing::new(vec_zero::<u8>(n * 16).map_err(|e| e.to_string())?);
    getrandom::fill(&mut random).map_err(|_| "replicated OS randomness failed")?;
    let mut shares: [Zeroizing<Vec<u8>>; 3] = [
        Zeroizing::new(vec_zero(n * 16).map_err(|e| e.to_string())?),
        Zeroizing::new(vec_zero(n * 16).map_err(|e| e.to_string())?),
        Zeroizing::new(vec_zero(n * 16).map_err(|e| e.to_string())?),
    ];
    for (i, raw) in random.chunks_exact(16).enumerate() {
        let a = u64::from_le_bytes(raw[..8].try_into().unwrap());
        let b = u64::from_le_bytes(raw[8..].try_into().unwrap());
        let c = (values[i] as u64).wrapping_sub(a).wrapping_sub(b);
        let parts = [a, b, c];
        for j in 0..3 {
            shares[j][i * 16..i * 16 + 8].copy_from_slice(&parts[j].to_le_bytes());
            shares[j][i * 16 + 8..i * 16 + 16].copy_from_slice(&parts[(j + 1) % 3].to_le_bytes());
        }
    }
    Ok(SourceChunk {
        key,
        offset,
        shares,
    })
}

impl Drop for Publisher {
    fn drop(&mut self) {
        self.cancel();
    }
}

#[cfg(test)]
mod final_outcome_tests {
    use super::*;

    fn transaction(words: [u64; 2]) -> (Publisher, [Vec<u8>; 3]) {
        let config = br#"{"model_type":"qwen2","hidden_size":2,"intermediate_size":2,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":2,"vocab_size":3,"hidden_act":"silu","tie_word_embeddings":false,"attention_bias":true,"max_position_embeddings":16,"rope_theta":10000,"rms_norm_eps":0.000001}"#;
        let policy = super::super::ExecutionPolicy {
            range_policy: RangePolicy::Checked,
            max_abs_activation: 1,
            ..Default::default()
        };
        let mut publisher =
            Publisher::new_with_policy(config, 1, 1, [1; 32], [2; 32], WeightMode::Public, policy)
                .unwrap();
        let shapes = publisher.shapes().to_vec();
        for (key, shape) in shapes {
            let value = if key.contains("norm") { 1f32 } else { 0f32 };
            let data: Vec<u8> = (0..shape.iter().product::<usize>())
                .flat_map(|_| value.to_le_bytes())
                .collect();
            publisher.append(&key, 0, &data).unwrap();
        }
        publisher.seal(&[0]).unwrap();
        // Parser-only unit: final frames are synthetic. End-to-end range failure
        // remains covered by real protected executions in checked_acceptance_tests.
        publisher.issued = publisher.values.len();
        let frames = std::array::from_fn(|party| {
            let mut frame = b"PUR1".to_vec();
            frame.extend(publisher.manifest.session);
            frame.extend(publisher.manifest.program);
            frame.push(party as u8);
            frame.extend(2u32.to_le_bytes());
            for word in words {
                frame.extend(if party == 0 { word } else { 0 }.to_le_bytes());
            }
            frame
        });
        (publisher, frames)
    }

    #[test]
    fn only_well_formed_zero_acceptance_is_numeric_rejection() {
        let (publisher, frames) = transaction([0, 0]);
        assert_eq!(
            publisher
                .finish_numeric_outcome(frames.each_ref().map(Vec::as_slice))
                .unwrap(),
            None
        );
        let (publisher, frames) = transaction([0, 1]);
        assert_eq!(
            publisher
                .finish_numeric_outcome(frames.each_ref().map(Vec::as_slice))
                .unwrap(),
            Some(vec![0])
        );
        for words in [[0, 2], [1, 0], [3, 1]] {
            let (publisher, frames) = transaction(words);
            assert!(publisher
                .finish_numeric_outcome(frames.each_ref().map(Vec::as_slice))
                .is_err());
        }
        let (publisher, mut frames) = transaction([0, 0]);
        frames[0][4] ^= 1;
        assert!(publisher
            .finish_numeric_outcome(frames.each_ref().map(Vec::as_slice))
            .is_err());
    }
}

#[cfg(test)]
#[path = "checked_acceptance_tests.rs"]
mod checked_acceptance_tests;
