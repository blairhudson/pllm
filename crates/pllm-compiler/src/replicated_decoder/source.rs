use super::*;
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
        let mut program = Program::compile_with_policy(
            &self.config,
            self.rows,
            self.outputs,
            &self.tensors,
            self.weights,
            self.execution,
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
}

impl Publisher {
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
        })
    }
    pub fn shapes(&self) -> &[(String, Vec<usize>)] {
        &self.shapes
    }
    pub fn cancel(&mut self) {
        self.retired = true;
        self.values.clear();
        self.program = None;
    }
    fn checked<T>(&mut self, result: Result<T, String>) -> Result<T, String> {
        if result.is_err() {
            self.cancel();
        }
        result
    }
    pub fn append(&mut self, key: &str, offset: usize, data: &[u8]) -> Result<(), String> {
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
            for (i, raw) in data.chunks_exact(4).enumerate() {
                let value = encode_q18(raw)?;
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
            let mut program = Program::compile_with_policy(
                &self.manifest.config,
                self.manifest.rows,
                self.manifest.outputs,
                &self.manifest.tensors,
                self.manifest.weights,
                self.manifest.execution,
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
        if self.retired || self.program.is_none() {
            return Err("replicated publisher not sealed".into());
        }
        serde_json::to_vec(&self.manifest)
            .map_err(|_| "replicated manifest serialization failed".into())
    }
    pub fn next_chunk(&mut self) -> Result<Option<SourceChunk>, String> {
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
                let c = (values[offset + i] as u64).wrapping_sub(a).wrapping_sub(b);
                let parts = [a, b, c];
                for j in 0..3 {
                    shares[j][i * 16..i * 16 + 8].copy_from_slice(&parts[j].to_le_bytes());
                    shares[j][i * 16 + 8..i * 16 + 16]
                        .copy_from_slice(&parts[(j + 1) % 3].to_le_bytes());
                }
            }
            self.issue_offset += n;
            if self.issue_offset == values.len() {
                self.values[self.issued].zeroize();
                self.values[self.issued].shrink_to_fit();
                self.issued += 1;
                self.issue_offset = 0;
            }
            Ok(Some(SourceChunk {
                key,
                offset,
                shares,
            }))
        })();
        self.checked(result)
    }
    pub fn finish(mut self, frames: [&[u8]; 3]) -> Result<Vec<u32>, String> {
        if self.retired || self.program.is_none() || self.issued != self.values.len() {
            return Err("incomplete replicated transaction".into());
        }
        let checked = self.manifest.execution.range_policy == RangePolicy::Checked;
        let n = self.manifest.outputs + usize::from(checked);
        let mut result = Zeroizing::new(vec_zero::<u64>(n).map_err(|e| e.to_string())?);
        for (party, frame) in frames.into_iter().enumerate() {
            if frame.len() != 73 + 8 * n
                || &frame[..4] != b"PUR1"
                || frame[4..36] != self.manifest.session
                || frame[36..68] != self.manifest.program
                || frame[68] != party as u8
                || u32::from_le_bytes(frame[69..73].try_into().unwrap()) as usize != n
            {
                return Err("replicated final share identity mismatch".into());
            }
            for (sum, raw) in result.iter_mut().zip(frame[73..].chunks_exact(8)) {
                *sum = sum.wrapping_add(u64::from_le_bytes(raw.try_into().unwrap()));
            }
        }
        if checked && result.pop() != Some(1) {
            return Err("replicated request failed protected range checks".into());
        }
        let config: serde_json::Value =
            serde_json::from_slice(&self.manifest.config).map_err(|_| "invalid source config")?;
        let vocabulary = config.get("text_config").unwrap_or(&config)["vocab_size"]
            .as_u64()
            .ok_or("missing vocabulary")?;
        if result.iter().any(|v| *v >= vocabulary) {
            return Err("replicated final token range".into());
        }
        self.cancel();
        Ok(result.iter().map(|v| *v as u32).collect())
    }
}
impl Drop for Publisher {
    fn drop(&mut self) {
        self.cancel();
    }
}

#[cfg(test)]
#[path = "checked_acceptance_tests.rs"]
mod checked_acceptance_tests;
