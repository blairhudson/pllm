use super::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use zeroize::Zeroize;

const CHUNK: usize = 4096;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema: String,
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
        let mut program = Program::compile(&self.config, self.rows, self.outputs, &self.tensors)?;
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
}

impl Publisher {
    pub fn new(
        config: &[u8],
        rows: usize,
        outputs: usize,
        source: [u8; 32],
        composition: [u8; 32],
    ) -> Result<Self, String> {
        let shapes: Vec<_> = Program::source_shapes(config, rows, outputs)?
            .into_iter()
            .collect();
        if source == [0; 32] || composition == [0; 32] {
            return Err("empty source/composition pin".into());
        }
        let mut session = [0; 32];
        getrandom::fill(&mut session).map_err(|_| "replicated OS randomness failed")?;
        let values = shapes
            .iter()
            .map(|(_, shape)| {
                Ok(Zeroizing::new(
                    vec_zero(shape.iter().product()).map_err(|e| e.to_string())?,
                ))
            })
            .collect::<Result<_, String>>()?;
        Ok(Self {
            manifest: Manifest {
                schema: "pllm.replicated-decoder.manifest.v1".into(),
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
                || offset + data.len() / 4 > self.values[self.cursor].len()
            {
                return Err("invalid ordered replicated source frame".into());
            }
            for (out, raw) in self.values[self.cursor][offset..offset + data.len() / 4]
                .iter_mut()
                .zip(data.chunks_exact(4))
            {
                let value = f32::from_le_bytes(raw.try_into().unwrap());
                let scaled = value * ONE as f32;
                if !scaled.is_finite() || scaled.abs() > (1u64 << 40) as f32 {
                    return Err("replicated Q18 source range".into());
                }
                *out = scaled as i64;
            }
            self.offset += data.len() / 4;
            if self.offset == self.values[self.cursor].len() {
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
            let plan = program::lower(
                &self.manifest.config,
                self.manifest.rows,
                self.manifest.outputs,
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
            for ((key, shape), values) in self.shapes.iter().zip(&self.values) {
                let mut hash = Sha256::new();
                for value in values.iter() {
                    hash.update(value.to_le_bytes());
                }
                self.manifest.tensors.insert(
                    key.clone(),
                    SourceTensor {
                        shape: shape.clone(),
                        lower: *values.iter().min().unwrap(),
                        upper: *values.iter().max().unwrap(),
                        digest: hash.finalize().into(),
                        public_q18: public.contains(key.as_str()).then(|| values.to_vec()),
                    },
                );
            }
            let mut program = Program::compile(
                &self.manifest.config,
                self.manifest.rows,
                self.manifest.outputs,
                &self.manifest.tensors,
            )?;
            program.bind(self.manifest.source, self.manifest.composition);
            self.manifest.program = program.digest();
            let mut values = vec![];
            for ((key, _), value) in self.shapes.iter().zip(std::mem::take(&mut self.values)) {
                if !public.contains(key.as_str()) {
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
        let n = self.manifest.outputs;
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
