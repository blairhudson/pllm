//! Ordered, bounded source ingestion for the existing SDK transport owners.
//! Source float32 frames retire after each call; encoded weights and original
//! lookup/local bytes move into their final owners without a second snapshot.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use pllm_core::gkm::{self, FloatTable, I16Matrix, Worker};
use sha2::{Digest as _, Sha256};

use super::{
    reject, BoundSource, GkmRuntimeContract, GkmTransportClient, GkmTransportWorker, INPUT_BOUND,
};

pub const GKM_SOURCE_CHUNK_BYTES: usize = 64 << 10;
const MAX_LIVE_IMPORT_BYTES: usize = 16 << 20;
static LIVE_IMPORT_BYTES: AtomicUsize = AtomicUsize::new(0);

struct ImportPermit(usize);

impl ImportPermit {
    fn acquire(bytes: usize, capacity: gkm::Capacity) -> Result<Self, String> {
        let limit = if capacity == gkm::Capacity::Decoder {
            gkm::Capacity::Decoder.max_publication_bytes()
        } else {
            MAX_LIVE_IMPORT_BYTES
        };
        LIVE_IMPORT_BYTES
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |live| {
                live.checked_add(bytes).filter(|&total| total <= limit)
            })
            .map_err(|_| reject("source import payload allowance exhausted"))?;
        Ok(Self(bytes))
    }
}

impl Drop for ImportPermit {
    fn drop(&mut self) {
        LIVE_IMPORT_BYTES.fetch_sub(self.0, Ordering::AcqRel);
    }
}

fn filled<T: Clone>(len: usize, value: T) -> Result<Vec<T>, String> {
    let mut out = Vec::new();
    out.try_reserve_exact(len)
        .map_err(|_| reject("source import allocation"))?;
    out.resize(len, value);
    Ok(out)
}

struct Buffers {
    weights: Vec<Vec<i16>>,
    table: Vec<u8>,
    local: BTreeMap<String, Vec<u8>>,
    hash: Sha256,
    tensor: usize,
    offset: usize,
    _permit: ImportPermit,
}

/// One-use source importer. The complete native Experiment/request contract
/// precedes allocation. Calls must follow its sorted tensor names and exact byte
/// offsets; substitution, malformed chunks, incomplete finish and abort retire
/// all imports. This owns public source payload, never secret query material.
pub struct GkmTransportImport {
    contract: GkmRuntimeContract,
    buffers: Option<Buffers>,
}

impl GkmTransportImport {
    pub fn new(contract: GkmRuntimeContract) -> Result<Self, String> {
        let decoder = contract.decoder();
        let shapes: Vec<_> = decoder
            .matrices
            .iter()
            .map(|m| (m.rows, m.columns))
            .collect();
        let resources = gkm::PublicationResources::for_shapes_with_capacity(
            &shapes,
            decoder.vocabulary,
            decoder.width,
            decoder.capacity(),
        )
        .map_err(|e| e.to_string())?;
        let local_bytes = decoder
            .local_tensors
            .iter()
            .map(|name| decoder.source_shapes[name].iter().product::<usize>() * 4)
            .sum::<usize>();
        // All geometry was bounded by the immutable decoder contract. Charge
        // final row/input bounds too: they coexist during ownership conversion.
        let permit = ImportPermit::acquire(
            resources.source_weight_bytes
                + resources.source_bound_bytes
                + resources.source_table_bytes
                + local_bytes,
            decoder.capacity(),
        )?;
        let mut weights = Vec::new();
        weights
            .try_reserve_exact(shapes.len())
            .map_err(|_| reject("source import allocation"))?;
        for (rows, columns) in shapes {
            weights.push(filled(rows * columns, 0)?);
        }
        let table = filled(decoder.vocabulary * decoder.width * 4, 0)?;
        let mut local = BTreeMap::new();
        for name in &decoder.local_tensors {
            local.insert(
                name.clone(),
                filled(decoder.source_shapes[name].iter().product::<usize>() * 4, 0)?,
            );
        }
        let mut hash = Sha256::new();
        hash.update(b"pllm.gkm.decoder-source.v1");
        hash.update(decoder.digest().as_str().as_bytes());
        Ok(Self {
            contract,
            buffers: Some(Buffers {
                weights,
                table,
                local,
                hash,
                tensor: 0,
                offset: 0,
                _permit: permit,
            }),
        })
    }

    pub fn push(&mut self, name: &str, offset: usize, bytes: &[u8]) -> Result<(), String> {
        // Take first: every error, including a malformed call after completion,
        // drops partially imported weights and releases the process reservation.
        let mut buffers = self
            .buffers
            .take()
            .ok_or_else(|| reject("source import retired"))?;
        let decoder = self.contract.decoder();
        let (expected, shape) = decoder
            .source_shapes
            .iter()
            .nth(buffers.tensor)
            .ok_or_else(|| reject("source import already complete"))?;
        let total = shape.iter().product::<usize>() * 4;
        if name != expected
            || offset != buffers.offset
            || bytes.is_empty()
            || bytes.len() > GKM_SOURCE_CHUNK_BYTES
            || bytes.len() % 4 != 0
            || bytes.len() > total - buffers.offset
        {
            return Err(reject("source import name, offset or frame bound mismatch"));
        }
        if offset == 0 {
            buffers.hash.update((name.len() as u64).to_le_bytes());
            buffers.hash.update(name);
            buffers.hash.update((total as u64).to_le_bytes());
        }
        // Validate original float bytes even for coefficients/lookup rows which
        // do not cross the signed-i16 boundary. Signed zero remains bit-exact.
        for value in bytes.chunks_exact(4) {
            if !f32::from_le_bytes(value.try_into().unwrap()).is_finite() {
                return Err(reject("source float32 shape or finite-value mismatch"));
            }
        }
        for (index, matrix) in decoder.matrices.iter().enumerate() {
            for member in &matrix.members {
                if member.weight == name {
                    let begin = member.offset * matrix.columns + offset / 4;
                    for (slot, value) in buffers.weights[index][begin..begin + bytes.len() / 4]
                        .iter_mut()
                        .zip(bytes.chunks_exact(4))
                    {
                        *slot = gkm::encode_weight(f64::from(f32::from_le_bytes(
                            value.try_into().unwrap(),
                        )))
                        .map_err(|e| e.to_string())?;
                    }
                }
            }
        }
        if name == decoder.table {
            buffers.table[offset..offset + bytes.len()].copy_from_slice(bytes);
        }
        if let Some(local) = buffers.local.get_mut(name) {
            local[offset..offset + bytes.len()].copy_from_slice(bytes);
        }
        buffers.hash.update(bytes);
        buffers.offset += bytes.len();
        if buffers.offset == total {
            buffers.tensor += 1;
            buffers.offset = 0;
        }
        self.buffers = Some(buffers);
        Ok(())
    }

    fn finish_source(&mut self) -> Result<(BoundSource, ImportPermit), String> {
        let buffers = self
            .buffers
            .take()
            .ok_or_else(|| reject("source import retired"))?;
        let decoder = self.contract.decoder();
        if buffers.tensor != decoder.source_shapes.len() || buffers.offset != 0 {
            return Err(reject("source import incomplete"));
        }
        let mut matrices = Vec::new();
        matrices
            .try_reserve_exact(decoder.matrices.len())
            .map_err(|_| reject("source import allocation"))?;
        for (matrix, weights) in decoder.matrices.iter().zip(buffers.weights) {
            matrices.push(
                I16Matrix::from_encoded_owned_with_capacity(
                    matrix.rows,
                    matrix.columns,
                    weights,
                    filled(matrix.columns, INPUT_BOUND)?,
                    decoder.capacity(),
                )
                .map_err(|e| e.to_string())?,
            );
        }
        Ok((
            BoundSource {
                matrices,
                table: FloatTable::from_le_bytes_owned_with_capacity(
                    decoder.vocabulary,
                    decoder.width,
                    buffers.table,
                    decoder.capacity(),
                )
                .map_err(|e| e.to_string())?,
                local: buffers.local,
                digest: buffers.hash.finalize().into(),
            },
            buffers._permit,
        ))
    }

    pub fn client(mut self) -> Result<GkmTransportClient, String> {
        let (source, _permit) = self.finish_source()?;
        // Reservation remains live through publication construction and transfer.
        GkmTransportClient::from_source(self.contract, source)
    }

    pub fn worker(
        mut self,
        publication: &[u8],
        pin: [u8; 32],
        role: Worker,
        start: &[u8],
    ) -> Result<GkmTransportWorker, String> {
        let (source, _permit) = self.finish_source()?;
        GkmTransportWorker::from_source(self.contract, source, publication, pin, role, start)
    }

    pub fn abort(&mut self) {
        self.buffers = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pllm_types::{canonical_bytes, Digest};
    use serde_json::json;

    fn contract() -> GkmRuntimeContract {
        GkmRuntimeContract::compile(
            &canonical_bytes(&json!({"schema":"pllm.experiment.v2", "name":"source-owner",
                "pipeline":{"model":{"source":"fixture", "kind":"safetensors"}, "components":{
                    "kernels":{"component":"pllm/cpu", "params":{"threads":1}},
                    "linear":{"component":"pllm/open-weight-verifiable-outsourcing/v1",
                        "params":{"sharing":"full", "generation":"fixed_length_greedy"}},
                    "verification":{"component":"pllm/published-linear-verify/v1", "params":{"acceptance":"request"}},
                    "topology":{"component":"pllm/two-online-delegation-workers/v1", "params":{}},
                    "state":{"component":"pllm/client-local-kv", "params":{}}
                }}, "deployment":{"kind":"local", "root":"./source-owner"},
                "budget":{"requests":1, "max_input_tokens":1, "max_new_tokens":1}})),
            &canonical_bytes(&json!({"model_type":"qwen2", "hidden_size":128, "intermediate_size":128,
                "num_hidden_layers":1, "num_attention_heads":4, "num_key_value_heads":2,
                "vocab_size":256, "max_position_embeddings":16, "hidden_act":"silu",
                "rms_norm_eps":1e-6, "rope_theta":10000.0, "tie_word_embeddings":true})),
            1, 1, Digest::from_sha256([1; 32]), Digest::from_sha256([2; 32]),
        ).unwrap()
    }

    #[test]
    fn actual_source_buffers_share_one_process_allowance_until_abort_or_failure() {
        let before = LIVE_IMPORT_BYTES.load(Ordering::Acquire);
        let mut owners = Vec::new();
        loop {
            match GkmTransportImport::new(contract()) {
                Ok(owner) => owners.push(owner),
                Err(error) => {
                    assert!(error.contains("payload allowance exhausted"));
                    break;
                }
            }
            assert!(owners.len() < 512);
        }
        assert!(owners.len() > 1);
        let full = LIVE_IMPORT_BYTES.load(Ordering::Acquire);
        assert!(full <= MAX_LIVE_IMPORT_BYTES);
        owners[0].abort();
        assert!(LIVE_IMPORT_BYTES.load(Ordering::Acquire) < full);
        let mut replacement = GkmTransportImport::new(contract()).unwrap();
        assert_eq!(LIVE_IMPORT_BYTES.load(Ordering::Acquire), full);
        assert!(replacement.push("wrong", 0, &[0; 4]).is_err());
        assert!(LIVE_IMPORT_BYTES.load(Ordering::Acquire) < full);
        drop((replacement, owners));
        assert_eq!(LIVE_IMPORT_BYTES.load(Ordering::Acquire), before);
    }
}
