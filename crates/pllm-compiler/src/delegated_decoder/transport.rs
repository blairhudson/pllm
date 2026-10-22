//! Source- and Experiment-bound adapters for full-share or seeded wire owners.
//!
//! The client publishes locally, retires its imported matrix/table owners, and
//! retains only public local coefficients plus its native private transaction.
//! Worker construction imports a trusted publication, checks the actual source,
//! and binds the complete compiler-generated schedule and runtime context before
//! accepting a share. A worker owns exactly one role and one request.
//!
//! `accept` returns **tentative** original-row or signed-reconstructed/rescaled
//! float32 bytes. As in `GkmDecoderSession`, biases remain client-local and are
//! applied by the semantic executor after division by 2^22. `local_tensors`
//! supplies their original bytes together with local normalization coefficients.
//! Only `finish` returns the canonical `pllm.gkm.sdk-acceptance.v1` receipt.
//! Worker completion uses `pllm.gkm.sdk-worker-completion.v1` and never verifies
//! the client's result. Neither adapter supplies transport authentication.
//!
//! Native publication/wire caps remain unchanged. Compiler source imports,
//! canonical metadata, local coefficient copies and returned float/receipt bytes
//! belong to the admitted outer execution/transport envelope. Callers must retain
//! that admission and abort native owners on argument-decoding or transport errors.

use std::collections::BTreeMap;
use std::sync::Arc;

use pllm_core::gkm::{self, ClientWireWindow, Publication, Stage, Step, WireBytes, Worker};
use pllm_types::{canonical_bytes, Digest};
use sha2::{Digest as _, Sha256};
use zeroize::Zeroizing;

use super::{bind_source, float_bytes, reject, stage_id, BoundSource, GkmRuntimeContract};

fn request_context(contract: &GkmRuntimeContract, source: [u8; 32]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"pllm.gkm.sdk-request-context.v1");
    hash.update(contract.digest().as_str().as_bytes());
    hash.update(source);
    hash.finalize().into()
}

fn plan(contract: &GkmRuntimeContract) -> Result<Vec<Step>, String> {
    let decoder = contract.decoder();
    let calls = decoder.window_plan();
    if calls.len() != decoder.window_steps {
        return Err(reject("transport window count mismatch"));
    }
    Ok(calls)
}

enum ClientWindow {
    Scalar(ClientWireWindow),
    Batch(gkm::ClientBatchWireWindow),
}

impl ClientWindow {
    fn request_id(&self) -> [u8; 32] {
        match self {
            Self::Scalar(w) => w.request_id(),
            Self::Batch(w) => w.request_id(),
        }
    }
    fn start_frames(&mut self) -> Result<[WireBytes; 2], gkm::Error> {
        match self {
            Self::Scalar(w) => w.start_frames(),
            Self::Batch(w) => w.start_frames(),
        }
    }
    fn lookup(&mut self, tokens: &[usize]) -> Result<[WireBytes; 2], gkm::Error> {
        match self {
            Self::Scalar(w) if tokens.len() == 1 => w.lookup(tokens[0]),
            Self::Scalar(_) => Err(gkm::Error::Shape),
            Self::Batch(w) => w.lookup(tokens),
        }
    }
    fn linear(&mut self, input: &[i64]) -> Result<[WireBytes; 2], gkm::Error> {
        match self {
            Self::Scalar(w) => w.linear(input),
            Self::Batch(w) => w.linear(input),
        }
    }
    fn accept_replies(
        &mut self,
        zero: &[u8],
        one: &[u8],
    ) -> Result<gkm::WireTentative<'_>, gkm::Error> {
        match self {
            Self::Scalar(w) => w.accept_replies(zero, one),
            Self::Batch(w) => w.accept_replies(zero, one),
        }
    }
    fn finish(&mut self) -> Result<gkm::AcceptedWindow, gkm::Error> {
        match self {
            Self::Scalar(w) => w.finish(),
            Self::Batch(w) => w.finish(),
        }
    }
}

enum WorkerWindow {
    Scalar(gkm::WorkerWireWindow),
    Batch(gkm::WorkerBatchWireWindow),
}

impl WorkerWindow {
    fn request_id(&self) -> [u8; 32] {
        match self {
            Self::Scalar(w) => w.request_id(),
            Self::Batch(w) => w.request_id(),
        }
    }
    fn sharing_profile(&self) -> gkm::SharingProfile {
        match self {
            Self::Scalar(w) => w.sharing_profile(),
            Self::Batch(w) => w.sharing_profile(),
        }
    }
    fn execute(&mut self, request: &[u8]) -> Result<WireBytes, gkm::Error> {
        match self {
            Self::Scalar(w) => w.execute(request),
            Self::Batch(w) => w.execute(request),
        }
    }
    fn finish(&mut self) -> Result<gkm::WorkerWireReceipt, gkm::Error> {
        match self {
            Self::Scalar(w) => w.finish(),
            Self::Batch(w) => w.finish(),
        }
    }
}

/// One complete SDK request. External I/O cannot detach the native pending
/// transaction; all state-changing errors retire the owner even when ignored.
pub struct GkmTransportClient {
    contract: GkmRuntimeContract,
    local: BTreeMap<String, Vec<u8>>,
    publication: Option<Arc<Publication>>,
    publication_exported: bool,
    window: Option<ClientWindow>,
    calls: Vec<Step>,
    ordinal: usize,
    pending: Option<Step>,
    pending_rows: usize,
    batches: Vec<gkm::BatchDescriptor>,
    source: [u8; 32],
    context: [u8; 32],
    publication_digest: [u8; 32],
    request: [u8; 32],
}

impl GkmTransportClient {
    pub fn new(
        contract: GkmRuntimeContract,
        tensors: &BTreeMap<String, &[u8]>,
    ) -> Result<Self, String> {
        let source = bind_source(contract.decoder(), tensors)?;
        Self::from_source(contract, source)
    }

    pub(super) fn from_source(
        contract: GkmRuntimeContract,
        bound: BoundSource,
    ) -> Result<Self, String> {
        let calls = plan(&contract)?;
        let BoundSource {
            matrices,
            table,
            local,
            digest: source,
        } = bound;
        let stages: Vec<_> = contract
            .decoder()
            .matrices
            .iter()
            .zip(&matrices)
            .map(|(spec, matrix)| Stage {
                id: stage_id(spec),
                matrix,
            })
            .collect();
        // Preserve the legacy actual-source digest and publication source context.
        // Runtime/Experiment identity additionally binds the request context below.
        let publication = Arc::new(
            Publication::publish_streamed_single_window(
                source,
                &stages,
                &table,
                contract.decoder().capacity(),
            )
            .map_err(|error| error.to_string())?,
        );
        drop(stages);
        drop(matrices);
        drop(table);
        let publication_digest = publication.digest();
        let context = request_context(&contract, source);
        let client = gkm::Client::new(Arc::clone(&publication), publication_digest)
            .map_err(|error| error.to_string())?;
        let admission = contract.batch_admission(&publication)?;
        let batches = if admission.is_some() {
            contract.decoder().batch_plan()?.descriptors().to_vec()
        } else {
            Vec::new()
        };
        let window = if let Some(admission) = admission {
            ClientWindow::Batch(
                client
                    .begin_batch_wire(context, &calls, admission)
                    .map_err(|error| error.to_string())?,
            )
        } else {
            ClientWindow::Scalar(
                match contract.sharing_profile() {
                    gkm::SharingProfile::FullShares => client.begin_wire(context, &calls),
                    gkm::SharingProfile::Aes128WindowV1 => {
                        client.begin_seeded_wire(context, &calls)
                    }
                }
                .map_err(|error| error.to_string())?,
            )
        };
        let request = window.request_id();
        Ok(Self {
            contract,
            local,
            publication: Some(publication),
            publication_exported: false,
            window: Some(window),
            calls,
            ordinal: 0,
            pending: None,
            pending_rows: 0,
            batches,
            source,
            context,
            publication_digest,
            request,
        })
    }

    pub fn contract_bytes(&self) -> Vec<u8> {
        self.contract.canonical_bytes()
    }

    pub fn execution(&self) -> Option<&'static str> {
        self.contract.execution()
    }

    pub fn decoder_bytes(&self) -> Vec<u8> {
        self.contract.decoder().canonical_bytes()
    }

    pub fn local_tensors(&self) -> &BTreeMap<String, Vec<u8>> {
        &self.local
    }

    pub fn publication_digest(&self) -> [u8; 32] {
        self.publication_digest
    }

    pub fn native_publication_resources(&self) -> Result<gkm::PublicationResources, String> {
        self.publication
            .as_ref()
            .map(|publication| publication.resources())
            .ok_or_else(|| reject("retired"))
    }

    pub fn request_id(&self) -> [u8; 32] {
        self.request
    }

    pub fn publication_bytes(&mut self) -> Result<WireBytes, String> {
        // One immutable public export, shared with both worker channels. The SDK
        // cannot mint unbounded publication snapshots under its fixed frame price.
        let result = if std::mem::replace(&mut self.publication_exported, true) {
            Err(reject("publication already exported"))
        } else {
            self.publication
                .as_ref()
                .ok_or_else(|| reject("retired"))
                .and_then(|publication| {
                    publication.export_wire().map_err(|error| error.to_string())
                })
        };
        if result.is_err() {
            self.abort();
        }
        result
    }

    /// Role-specific start frames, once, before input issuance. In seeded mode
    /// One's frame is secret key material requiring confidential authenticated delivery.
    pub fn starts(&mut self) -> Result<[WireBytes; 2], String> {
        let result = self
            .window
            .as_mut()
            .ok_or_else(|| reject("retired"))
            .and_then(|window| window.start_frames().map_err(|error| error.to_string()));
        if result.is_err() {
            self.abort();
        }
        result
    }

    pub fn lookup(&mut self, token: usize) -> Result<[WireBytes; 2], String> {
        self.lookup_rows(&[token])
    }

    /// One complete source-stage token lookup. Never splits or falls back to
    /// scalar requests; wrong row counts retire the complete transaction.
    pub fn lookup_batch(&mut self, tokens: &[usize]) -> Result<[WireBytes; 2], String> {
        if self.contract.execution().is_none() {
            self.abort();
            return Err(reject("batch lookup requires source-stage batch execution"));
        }
        self.lookup_rows(tokens)
    }

    /// Non-mutating trusted geometry check before an outer-language tuple parse.
    /// The bridge must first take ownership so parse errors still burn the owner.
    pub fn lookup_batch_rows(&self) -> Result<usize, String> {
        if self.contract.execution().is_none()
            || self.window.is_none()
            || self.pending.is_some()
            || self.calls.get(self.ordinal) != Some(&Step::Lookup)
        {
            return Err(reject("transport batch lookup unavailable"));
        }
        self.next_rows()
    }

    fn next_rows(&self) -> Result<usize, String> {
        if self.contract.execution().is_none() {
            return Ok(1);
        }
        self.batches
            .iter()
            .find(|batch| batch.first_claim as usize == self.ordinal)
            .map(|batch| batch.rows as usize)
            .ok_or_else(|| reject("transport batch boundary mismatch"))
    }

    fn lookup_rows(&mut self, tokens: &[usize]) -> Result<[WireBytes; 2], String> {
        let result = (|| {
            let rows = self.next_rows()?;
            if self.pending.is_some()
                || self.calls.get(self.ordinal) != Some(&Step::Lookup)
                || tokens.len() != rows
            {
                return Err(reject("transport lookup order"));
            }
            let window = self.window.as_mut().ok_or_else(|| reject("retired"))?;
            let requests = window.lookup(tokens).map_err(|error| error.to_string())?;
            self.pending = Some(Step::Lookup);
            self.pending_rows = rows;
            Ok(requests)
        })();
        if result.is_err() {
            self.abort();
        }
        result
    }

    /// The same finite f32 -> f64 -> nearest-even/2048 -> signed-i64 boundary as
    /// the native reference. No bias, clipping, per-row scale or W8A8 conversion.
    /// Selected batch execution requires all B source rows, row-major, in one
    /// call; scalar execution continues to require exactly one original row.
    pub fn linear(&mut self, stage: usize, input_f32_le: &[u8]) -> Result<[WireBytes; 2], String> {
        let result = (|| {
            let rows = self.next_rows()?;
            let spec = self
                .contract
                .decoder()
                .matrices
                .get(stage)
                .ok_or_else(|| reject("stage"))?;
            let step = Step::Linear(stage_id(spec));
            let elements = spec
                .columns
                .checked_mul(rows)
                .ok_or_else(|| reject("batch input shape overflow"))?;
            let expected_bytes = elements
                .checked_mul(4)
                .ok_or_else(|| reject("batch input byte overflow"))?;
            if self.pending.is_some()
                || self.calls.get(self.ordinal) != Some(&step)
                || input_f32_le.len() != expected_bytes
            {
                return Err(reject("transport linear order or input shape"));
            }
            let window = self.window.as_mut().ok_or_else(|| reject("retired"))?;
            // Selected batches must reject every malformed numeric value before
            // allocating their B*C encoded owner. Scalar behavior stays unchanged.
            if self.contract.execution().is_some() {
                for bytes in input_f32_le.chunks_exact(4) {
                    gkm::encode_input(f64::from(f32::from_le_bytes(bytes.try_into().unwrap())))
                        .map_err(|error| error.to_string())?;
                }
            }
            let mut input = Zeroizing::new(Vec::new());
            input
                .try_reserve_exact(elements)
                .map_err(|_| reject("allocation"))?;
            for bytes in input_f32_le.chunks_exact(4) {
                input.push(
                    gkm::encode_input(f64::from(f32::from_le_bytes(bytes.try_into().unwrap())))
                        .map_err(|error| error.to_string())?,
                );
            }
            let requests = window.linear(&input).map_err(|error| error.to_string())?;
            self.pending = Some(step);
            self.pending_rows = rows;
            Ok(requests)
        })();
        if result.is_err() {
            self.abort();
        }
        result
    }

    /// Explicit bridge entry point; a scalar contract cannot silently execute a
    /// batch request. Errors burn the same complete native transaction.
    pub fn linear_batch(
        &mut self,
        stage: usize,
        input_f32_le: &[u8],
    ) -> Result<[WireBytes; 2], String> {
        if self.contract.execution().is_none() {
            self.abort();
            return Err(reject("batch linear requires source-stage batch execution"));
        }
        self.linear(stage, input_f32_le)
    }

    /// Fix both role replies, use the original native hash/reconstruction and
    /// accumulator checks, then copy tentative values into bounded f32 LE bytes.
    /// Biases are intentionally applied afterward by the client semantic executor.
    pub fn accept(&mut self, zero: &[u8], one: &[u8]) -> Result<Vec<u8>, String> {
        let result = (|| {
            let pending = self
                .pending
                .take()
                .ok_or_else(|| reject("transport has no pending call"))?;
            if self.calls.get(self.ordinal) != Some(&pending) {
                return Err(reject("transport pending schedule mismatch"));
            }
            let window = self.window.as_mut().ok_or_else(|| reject("retired"))?;
            let tentative = window
                .accept_replies(zero, one)
                .map_err(|error| error.to_string())?;
            let bytes = float_bytes(tentative.values)?;
            self.ordinal += self.pending_rows;
            self.pending_rows = 0;
            Ok(bytes)
        })();
        if result.is_err() {
            self.abort();
        }
        result
    }

    pub fn accept_batch(&mut self, zero: &[u8], one: &[u8]) -> Result<Vec<u8>, String> {
        if self.contract.execution().is_none() {
            self.abort();
            return Err(reject("batch accept requires source-stage batch execution"));
        }
        self.accept(zero, one)
    }

    pub fn finish(&mut self) -> Result<Vec<u8>, String> {
        let result = (|| {
            let mut window = self.window.take().ok_or_else(|| reject("retired"))?;
            if self.pending.is_some() || self.ordinal != self.calls.len() {
                return Err(reject("transport incomplete request"));
            }
            let accepted = window.finish().map_err(|error| error.to_string())?;
            if accepted.sharing_profile() != self.contract.sharing_profile()
                || accepted.request_id() != self.request
                || accepted.publication() != self.publication_digest
                || accepted.context() != self.context
                || accepted.steps() != self.calls.len()
            {
                return Err(reject("transport acceptance binding mismatch"));
            }
            Ok(canonical_bytes(&serde_json::json!({
                "schema_version": "pllm.gkm.sdk-acceptance.v1",
                "contract": self.contract.digest(),
                "decoder_contract": self.contract.decoder().digest(),
                "numeric_profile": gkm::NUMERIC_PROFILE,
                "local_numeric_profile": self.contract.decoder().local_numeric_profile,
                "acceptance": self.contract.decoder().acceptance,
                "source": Digest::from_sha256(self.source),
                "publication": Digest::from_sha256(accepted.publication()),
                "request": Digest::from_sha256(accepted.request_id()),
                "window": Digest::from_sha256(accepted.plan_digest()),
                "context": Digest::from_sha256(self.context),
                "steps": accepted.steps(),
                "sharing": if accepted.sharing_profile() == gkm::SharingProfile::FullShares { "full" } else { "seeded" },
                "seeded": accepted.sharing_profile() == gkm::SharingProfile::Aes128WindowV1,
            })))
        })();
        self.abort();
        result
    }

    pub fn abort(&mut self) {
        self.window = None;
        self.publication = None;
        self.pending = None;
        self.pending_rows = 0;
        self.batches.clear();
        self.local.clear();
    }
}

/// One party-local request owner. Public source copies are held only by the
/// core `WorkerWire`; source-import temporaries and local coefficients retire
/// before this constructor returns. No private client input/token is retained.
pub struct GkmTransportWorker {
    contract: GkmRuntimeContract,
    window: Option<WorkerWindow>,
    resources: gkm::PublicationResources,
    owner: Option<gkm::WorkerWire>,
    source: [u8; 32],
    context: [u8; 32],
    publication: [u8; 32],
    role: Worker,
    steps: usize,
}

impl GkmTransportWorker {
    pub fn native_publication_resources(&self) -> Result<gkm::PublicationResources, String> {
        self.window
            .as_ref()
            .map(|_| self.resources)
            .ok_or_else(|| reject("retired"))
    }

    pub fn request_id(&self) -> Result<[u8; 32], String> {
        self.window
            .as_ref()
            .map(WorkerWindow::request_id)
            .ok_or_else(|| reject("retired"))
    }

    pub fn new(
        contract: GkmRuntimeContract,
        tensors: &BTreeMap<String, &[u8]>,
        publication: &[u8],
        pin: [u8; 32],
        role: Worker,
        start: &[u8],
    ) -> Result<Self, String> {
        // Check the pin and publication envelope before importing caller values.
        Publication::import_wire_streamed_single_window(
            publication,
            pin,
            contract.decoder().capacity(),
        )
        .map_err(|error| error.to_string())?;
        let source = bind_source(contract.decoder(), tensors)?;
        Self::from_source(contract, source, publication, pin, role, start)
    }

    pub(super) fn from_source(
        contract: GkmRuntimeContract,
        bound: BoundSource,
        publication: &[u8],
        pin: [u8; 32],
        role: Worker,
        start: &[u8],
    ) -> Result<Self, String> {
        let publication = Arc::new(
            Publication::import_wire_streamed_single_window(
                publication,
                pin,
                contract.decoder().capacity(),
            )
            .map_err(|error| error.to_string())?,
        );
        let calls = plan(&contract)?;
        let BoundSource {
            matrices,
            table,
            local,
            digest: source,
        } = bound;
        if publication.context() != source {
            return Err(reject("transport publication source mismatch"));
        }
        let context = request_context(&contract, source);
        let resources = publication.resources();
        let admission = contract.batch_admission(&publication)?;
        let ids: Vec<_> = contract.decoder().matrices.iter().map(stage_id).collect();
        let mut owner = gkm::WorkerWire::from_owned(publication, pin, role, &ids, matrices, table)
            .map_err(|error| error.to_string())?;
        drop(local);
        let window = if let Some(admission) = admission {
            WorkerWindow::Batch(
                owner
                    .begin_batch_bound(start, context, &calls, admission)
                    .map_err(|error| error.to_string())?,
            )
        } else {
            WorkerWindow::Scalar(
                owner
                    .begin_bound(start, context, &calls)
                    .map_err(|error| error.to_string())?,
            )
        };
        if window.sharing_profile() != contract.sharing_profile() {
            return Err(reject("transport sharing profile mismatch"));
        }
        Ok(Self {
            contract,
            window: Some(window),
            resources,
            owner: Some(owner),
            source,
            context,
            publication: pin,
            role,
            steps: calls.len(),
        })
    }

    pub fn execute(&mut self, request: &[u8]) -> Result<WireBytes, String> {
        let result = self
            .window
            .as_mut()
            .ok_or_else(|| reject("retired"))
            .and_then(|window| window.execute(request).map_err(|error| error.to_string()));
        if result.is_err() {
            self.abort();
        }
        result
    }

    /// Schedule completion only. This is never the client verification receipt.
    pub fn finish(&mut self) -> Result<Vec<u8>, String> {
        let result = (|| {
            let mut window = self.window.take().ok_or_else(|| reject("retired"))?;
            let receipt = window.finish().map_err(|error| error.to_string())?;
            if receipt.worker != self.role || receipt.steps != self.steps {
                return Err(reject("transport worker completion mismatch"));
            }
            Ok(canonical_bytes(&serde_json::json!({
                "schema_version": "pllm.gkm.sdk-worker-completion.v1",
                "contract": self.contract.digest(),
                "decoder_contract": self.contract.decoder().digest(),
                "numeric_profile": gkm::NUMERIC_PROFILE,
                "source": Digest::from_sha256(self.source),
                "publication": Digest::from_sha256(self.publication),
                "context": Digest::from_sha256(self.context),
                "request": Digest::from_sha256(receipt.request_id),
                "steps": receipt.steps,
                "worker": if self.role == Worker::Zero { 0 } else { 1 },
                "sharing": if self.contract.sharing_profile() == gkm::SharingProfile::FullShares { "full" } else { "seeded" },
            })))
        })();
        self.abort();
        result
    }

    pub fn abort(&mut self) {
        self.window = None;
        self.owner = None;
    }
}

#[cfg(test)]
mod tests {
    use super::super::source_import::process_import_guard;
    use super::*;
    use pllm_types::canonical_bytes;
    use serde_json::json;

    fn contract(seeded: bool, decoder: bool) -> GkmRuntimeContract {
        contract_mode(seeded, decoder, false, 1, 1)
    }

    fn contract_mode(
        seeded: bool,
        decoder: bool,
        batch: bool,
        rows: usize,
        tokens: usize,
    ) -> GkmRuntimeContract {
        let mut linear = json!({"sharing": if seeded { "seeded" } else { "full" }, "generation":"fixed_length_greedy"});
        if batch {
            linear["execution"] = json!("source_stage_batch_v1");
        }
        if decoder {
            linear["capacity"] = json!("decoder");
        }
        GkmRuntimeContract::compile(
            &canonical_bytes(&json!({"schema":"pllm.experiment.v2", "name":"seeded-wire",
                "pipeline":{"model":{"source":"fixture", "kind":"safetensors"}, "components":{
                    "kernels":{"component":"pllm/cpu", "params":{"threads":1}},
                    "linear":{"component":"pllm/open-weight-verifiable-outsourcing/v1", "params":linear},
                    "verification":{"component":"pllm/published-linear-verify/v1", "params":{"acceptance":"request"}},
                    "topology":{"component":"pllm/two-online-delegation-workers/v1", "params":{}},
                    "state":{"component":"pllm/client-local-kv", "params":{}}
                }}, "deployment":{"kind":"local", "root":"./seeded-wire"},
                "budget":{"requests":1, "max_input_tokens":rows, "max_new_tokens":tokens}})),
            &canonical_bytes(&json!({"model_type":"qwen2", "hidden_size":8, "intermediate_size":16,
                "num_hidden_layers":1, "num_attention_heads":2, "num_key_value_heads":1,
                "vocab_size":16, "max_position_embeddings":16, "hidden_act":"silu",
                "rms_norm_eps":1e-6, "rope_theta":10000.0, "tie_word_embeddings":true})),
            rows, tokens, Digest::from_sha256([1; 32]), Digest::from_sha256([2; 32]),
        ).unwrap()
    }

    fn importer(seeded: bool, decoder: bool) -> super::super::GkmTransportImport {
        let contract = contract(seeded, decoder);
        let tensors: Vec<_> = contract
            .decoder()
            .source_shapes()
            .iter()
            .map(|(name, shape)| (name.clone(), vec![0u8; shape.iter().product::<usize>() * 4]))
            .collect();
        let mut importer = super::super::GkmTransportImport::new(contract).unwrap();
        for (name, bytes) in tensors {
            importer.push(&name, 0, &bytes).unwrap();
        }
        importer
    }

    fn mode_importer(
        seeded: bool,
        decoder: bool,
        batch: bool,
        rows: usize,
        tokens: usize,
    ) -> super::super::GkmTransportImport {
        let contract = contract_mode(seeded, decoder, batch, rows, tokens);
        let tensors: Vec<_> = contract
            .decoder()
            .source_shapes()
            .iter()
            .map(|(name, shape)| {
                let values: Vec<u8> = (0..shape.iter().product::<usize>())
                    .flat_map(|index| {
                        let value = if index == 0 {
                            -0.0f32
                        } else {
                            (index as f32 % 11.0 - 5.0) / 32.0
                        };
                        value.to_le_bytes()
                    })
                    .collect();
                (name.clone(), values)
            })
            .collect();
        let mut importer = super::super::GkmTransportImport::new(contract).unwrap();
        for (name, bytes) in tensors {
            importer.push(&name, 0, &bytes).unwrap();
        }
        importer
    }

    #[test]
    fn source_stage_transport_matches_scalar_t1_t2_full_seeded_and_decode() {
        let _guard = process_import_guard();
        for decoder in [false, true] {
            for seeded in [false, true] {
                for rows in [1, 2] {
                    let mut outputs = Vec::new();
                    let mut sources = Vec::new();
                    for batch in [false, true] {
                        let mut client = mode_importer(seeded, decoder, batch, rows, 2)
                            .client()
                            .unwrap();
                        let source_plan = client.contract.decoder().batch_plan().unwrap();
                        let batch_metadata: Option<serde_json::Value> = client
                            .contract
                            .batch_execution_bytes()
                            .map(|bytes| serde_json::from_slice(&bytes).unwrap());
                        assert_eq!(client.contract.execution().is_some(), batch);
                        assert_eq!(client.contract.max_aggregate_bytes().is_some(), batch);
                        sources.push(client.source);
                        let publication = client.publication_bytes().unwrap();
                        let pin = client.publication_digest();
                        let starts = client.starts().unwrap();
                        assert!(starts
                            .iter()
                            .all(|start| start.len() <= client.contract.scalar_frame_bytes()));
                        let mut workers = [Worker::Zero, Worker::One].map(|role| {
                            mode_importer(seeded, decoder, batch, rows, 2)
                                .worker(
                                    publication.as_bytes(),
                                    pin,
                                    role,
                                    starts[if role == Worker::Zero { 0 } else { 1 }].as_bytes(),
                                )
                                .unwrap()
                        });
                        let mut values = Vec::new();
                        for (batch_index, descriptor) in
                            source_plan.descriptors().iter().enumerate()
                        {
                            let stage = match descriptor.step {
                                Step::Lookup => None,
                                Step::Linear(id) => Some(
                                    client
                                        .contract
                                        .decoder()
                                        .matrices
                                        .iter()
                                        .position(|matrix| stage_id(matrix) == id)
                                        .unwrap(),
                                ),
                            };
                            let count = descriptor.rows as usize;
                            let inputs: Vec<u8> = if let Some(stage) = stage {
                                (0..count * client.contract.decoder().matrices[stage].columns)
                                    .flat_map(|index| {
                                        ((index as f32 % 7.0 - 3.0) / 16.0).to_le_bytes()
                                    })
                                    .collect()
                            } else {
                                Vec::new()
                            };
                            for row in 0..if batch { 1 } else { count } {
                                let requests = if let Some(stage) = stage {
                                    let stride =
                                        client.contract.decoder().matrices[stage].columns * 4;
                                    client
                                        .linear(
                                            stage,
                                            if batch {
                                                &inputs
                                            } else {
                                                &inputs[row * stride..(row + 1) * stride]
                                            },
                                        )
                                        .unwrap()
                                } else if batch {
                                    client
                                        .lookup_batch(
                                            &(0..count).map(|row| row + 3).collect::<Vec<_>>(),
                                        )
                                        .unwrap()
                                } else {
                                    client.lookup(row + 3).unwrap()
                                };
                                if batch {
                                    assert_eq!(&requests[0].as_bytes()[..8], b"PLGKMB01");
                                    let metadata = &batch_metadata.as_ref().unwrap()["descriptors"]
                                        [batch_index];
                                    assert_eq!(metadata["batch_index"], batch_index);
                                    assert_eq!(metadata["first_claim"], descriptor.first_claim);
                                    assert_eq!(metadata["phase_index"], descriptor.phase_index);
                                    assert_eq!(
                                        metadata["operation_index"],
                                        descriptor.operation_index
                                    );
                                    assert_eq!(metadata["rows"], descriptor.rows);
                                    assert_eq!(metadata["stage"], json!(stage));
                                    assert_eq!(
                                        metadata["kind"],
                                        if stage.is_some() { "linear" } else { "lookup" }
                                    );
                                    let (columns, output_width) = stage.map_or(
                                        (0, client.contract.decoder().width),
                                        |index| {
                                            let matrix = &client.contract.decoder().matrices[index];
                                            (matrix.columns, matrix.rows)
                                        },
                                    );
                                    assert_eq!(metadata["columns"], columns);
                                    assert_eq!(metadata["output_width"], output_width);
                                    for role in 0..2 {
                                        assert_eq!(
                                            metadata["request_bytes"][role],
                                            requests[role].len()
                                        );
                                    }
                                }
                                assert!(requests.iter().all(|request| request.len()
                                    <= client.contract.max_operation_frame_bytes()));
                                let zero = workers[0].execute(requests[0].as_bytes()).unwrap();
                                let one = workers[1].execute(requests[1].as_bytes()).unwrap();
                                if batch {
                                    let metadata = &batch_metadata.as_ref().unwrap()["descriptors"]
                                        [batch_index];
                                    assert_eq!(metadata["reply_bytes"][0], zero.len());
                                    assert_eq!(metadata["reply_bytes"][1], one.len());
                                }
                                values.extend(
                                    client.accept(zero.as_bytes(), one.as_bytes()).unwrap(),
                                );
                            }
                        }
                        let receipt: serde_json::Value =
                            serde_json::from_slice(&client.finish().unwrap()).unwrap();
                        assert_eq!(
                            receipt["steps"],
                            source_plan
                                .descriptors()
                                .iter()
                                .map(|d| d.rows as usize)
                                .sum::<usize>()
                        );
                        for worker in &mut workers {
                            worker.finish().unwrap();
                        }
                        outputs.push(values);
                    }
                    assert_eq!(outputs[0], outputs[1]);
                    assert_eq!(sources[0], sources[1]);
                }
            }
        }
    }

    #[test]
    fn source_stage_transport_rejects_partial_rows_incomplete_and_tampering() {
        let _guard = process_import_guard();
        let mut client = mode_importer(false, false, true, 2, 2).client().unwrap();
        assert_eq!(client.lookup_batch_rows().unwrap(), 2);
        assert_eq!(client.lookup_batch_rows().unwrap(), 2); // Getter cannot advance.
        client.starts().unwrap();
        assert!(client.lookup(3).is_err()); // Cannot microbatch a two-row lookup.
        assert!(client.lookup_batch(&[3, 4]).is_err());
        assert!(client.lookup_batch_rows().is_err());
        let client = mode_importer(false, false, false, 2, 2).client().unwrap();
        assert!(client.lookup_batch_rows().is_err());
        let mut client = mode_importer(false, false, true, 2, 2).client().unwrap();
        assert!(client.finish().is_err());
        assert!(client.starts().is_err());
        for seeded in [false, true] {
            let mut client = mode_importer(seeded, false, true, 2, 2).client().unwrap();
            let publication = client.publication_bytes().unwrap();
            let pin = client.publication_digest();
            let starts = client.starts().unwrap();
            // Authenticated source alone is insufficient: SDK execution/context
            // must match before a scalar worker can consume any selected start.
            assert!(mode_importer(seeded, false, false, 2, 2)
                .worker(
                    publication.as_bytes(),
                    pin,
                    Worker::Zero,
                    starts[0].as_bytes(),
                )
                .is_err());
            let mut workers = [Worker::Zero, Worker::One].map(|role| {
                mode_importer(seeded, false, true, 2, 2)
                    .worker(
                        publication.as_bytes(),
                        pin,
                        role,
                        starts[if role == Worker::Zero { 0 } else { 1 }].as_bytes(),
                    )
                    .unwrap()
            });
            let requests = client.lookup_batch(&[3, 4]).unwrap();
            assert!(client.lookup_batch_rows().is_err()); // Pending lookup, getter never consumes it.
            let zero = workers[0].execute(requests[0].as_bytes()).unwrap();
            let one = workers[1].execute(requests[1].as_bytes()).unwrap();
            let mut tampered = one.as_bytes().to_vec();
            let last = tampered.len() - 1;
            tampered[last] ^= 1;
            assert!(client.accept(zero.as_bytes(), &tampered).is_err());
            assert!(client.finish().is_err());
        }
    }

    #[test]
    fn selected_worker_binds_expected_context_and_complete_schedule_at_native_entry() {
        let _guard = process_import_guard();
        for decoder in [false, true] {
            for seeded in [false, true] {
                for wrong_schedule in [false, true] {
                    let mut client = mode_importer(seeded, decoder, true, 2, 2).client().unwrap();
                    let publication = client.publication_bytes().unwrap();
                    let pin = client.publication_digest();
                    // Independent imported publication supplies authenticated,
                    // native-authored adversarial starts, not edited wire bytes.
                    let imported = Arc::new(
                        Publication::import_wire_streamed_single_window(
                            publication.as_bytes(),
                            pin,
                            client.contract.decoder().capacity(),
                        )
                        .unwrap(),
                    );
                    let mut context = client.context;
                    let mut calls = client.calls.clone();
                    let mut descriptors = client.batches.clone();
                    if wrong_schedule {
                        let omitted = descriptors.pop().unwrap();
                        calls.truncate(omitted.first_claim as usize);
                    } else {
                        context[0] ^= 1;
                    }
                    let plan = gkm::BatchPlan::new(&calls, &descriptors).unwrap();
                    let admission = gkm::BatchAdmission::new(
                        &imported,
                        plan,
                        client.contract.sharing_profile(),
                    )
                    .unwrap();
                    let native = gkm::Client::new(Arc::clone(&imported), pin).unwrap();
                    let mut native_window =
                        native.begin_batch_wire(context, &calls, admission).unwrap();
                    let starts = native_window.start_frames().unwrap();
                    for role in [Worker::Zero, Worker::One] {
                        let error = mode_importer(seeded, decoder, true, 2, 2)
                            .worker(
                                publication.as_bytes(),
                                pin,
                                role,
                                starts[usize::from(role == Worker::One)].as_bytes(),
                            )
                            .err()
                            .expect("worker must reject a foreign context or shortened schedule");
                        assert_eq!(error, gkm::Error::Binding.to_string());
                    }
                }
            }
        }
    }

    #[test]
    fn batch_linear_rejects_bad_shape_and_late_nonfinite_encoding_and_burns_owner() {
        let _guard = process_import_guard();
        for malformed in [0, 1, 2] {
            let mut client = mode_importer(false, false, true, 2, 1).client().unwrap();
            let publication = client.publication_bytes().unwrap();
            let pin = client.publication_digest();
            let starts = client.starts().unwrap();
            let mut workers = [Worker::Zero, Worker::One].map(|role| {
                mode_importer(false, false, true, 2, 1)
                    .worker(
                        publication.as_bytes(),
                        pin,
                        role,
                        starts[if role == Worker::Zero { 0 } else { 1 }].as_bytes(),
                    )
                    .unwrap()
            });
            let requests = client.lookup_batch(&[3, 4]).unwrap();
            let zero = workers[0].execute(requests[0].as_bytes()).unwrap();
            let one = workers[1].execute(requests[1].as_bytes()).unwrap();
            client
                .accept_batch(zero.as_bytes(), one.as_bytes())
                .unwrap();
            assert!(client.lookup_batch_rows().is_err()); // Next source op is linear.
            let columns = client.contract.decoder().matrices[0].columns;
            let mut input = vec![0; columns * 2 * 4];
            match malformed {
                0 => input.truncate(columns * 4), // No partial source-stage fallback.
                1 => {
                    let n = input.len();
                    input[n - 4..].copy_from_slice(&f32::INFINITY.to_le_bytes());
                }
                _ => {
                    let n = input.len();
                    input[n - 4..].copy_from_slice(&f32::MAX.to_le_bytes());
                }
            }
            assert!(client.linear_batch(0, &input).is_err());
            assert!(client.finish().is_err());
        }
    }

    #[test]
    fn streamed_sdk_does_not_expose_repeat_publication_snapshots() {
        let _guard = process_import_guard();
        for decoder in [false, true] {
            for seeded in [false, true] {
                let mut client = importer(seeded, decoder).client().unwrap();
                let frame = client.publication_bytes().unwrap();
                assert!(!frame.is_empty());
                assert!(client.publication_bytes().is_err());
                assert!(client.starts().is_err());
                assert!(client.native_publication_resources().is_err());
            }
        }
    }

    #[test]
    fn seeded_sdk_selector_streamed_transport_and_final_acceptance_both_capacities() {
        let _guard = process_import_guard();
        for decoder in [false, true] {
            let mut outputs = Vec::new();
            let mut sources = Vec::new();
            for seeded in [false, true] {
                let mut client = importer(seeded, decoder).client().unwrap();
                let priced = client
                    .contract
                    .decoder()
                    .native_publication_resources()
                    .unwrap();
                let runtime = client.native_publication_resources().unwrap();
                assert_eq!(
                    runtime.ownership,
                    gkm::PublicationOwnership::StreamedSingleWindow
                );
                assert_eq!(
                    runtime.admitted_payload_bytes,
                    priced.admitted_payload_bytes
                );
                let metadata: serde_json::Value =
                    serde_json::from_slice(&client.decoder_bytes()).unwrap();
                let transport = &metadata["transport_resources"];
                let reference = super::super::GkmDecoderContract::compile_with_capacity(
                    &canonical_bytes(
                        &json!({"model_type":"qwen2", "hidden_size":8, "intermediate_size":16,
                        "num_hidden_layers":1, "num_attention_heads":2, "num_key_value_heads":1,
                        "vocab_size":16, "max_position_embeddings":16, "hidden_act":"silu",
                        "rms_norm_eps":1e-6, "rope_theta":10000.0, "tie_word_embeddings":true}),
                    ),
                    1,
                    0,
                    if decoder {
                        gkm::Capacity::Decoder
                    } else {
                        gkm::Capacity::Capsule
                    },
                )
                .unwrap();
                let generic = reference.native_publication_resources().unwrap();
                let reference_metadata: serde_json::Value =
                    serde_json::from_slice(&reference.canonical_bytes()).unwrap();
                assert!(reference_metadata.get("transport_resources").is_none());
                assert_eq!(
                    client.contract.decoder().source_binding_digest(),
                    reference.digest()
                );
                assert_eq!(metadata["source_binding_digest"], json!(reference.digest()));
                assert_eq!(generic.ownership, gkm::PublicationOwnership::General);
                assert_eq!(
                    transport["generic_native_admitted_payload_bytes"],
                    generic.admitted_payload_bytes
                );
                assert_eq!(
                    generic.admitted_payload_bytes + gkm::STREAMED_IMPORT_WORKSPACE_BYTES,
                    runtime.admitted_payload_bytes
                        + generic.max_import_workspace_bytes
                        + 7 * generic.max_window_payload_bytes
                );
                assert_eq!(transport["reserved_windows"], 1);
                assert_eq!(
                    transport["native_admitted_payload_bytes"],
                    runtime.admitted_payload_bytes
                );
                assert_eq!(
                    transport["import_workspace_bytes"],
                    gkm::STREAMED_IMPORT_WORKSPACE_BYTES
                );
                assert_eq!(
                    transport["native_admitted_payload_bytes"].as_u64().unwrap(),
                    [
                        "source_payload_bytes",
                        "import_workspace_bytes",
                        "publication_payload_bytes",
                        "publisher_workspace_bytes",
                        "window_payload_bytes"
                    ]
                    .iter()
                    .map(|key| transport[key].as_u64().unwrap())
                    .sum::<u64>()
                );
                if decoder {
                    let resources = &metadata["decoder_resources"];
                    assert_eq!(
                        resources["worker_payload_bytes"].as_u64().unwrap(),
                        runtime.admitted_payload_bytes as u64
                            + resources["outer_transport_bytes"].as_u64().unwrap()
                            + resources["local_coefficient_copy_bytes"].as_u64().unwrap()
                    );
                    assert_eq!(
                        resources["client_payload_bytes"].as_u64().unwrap(),
                        resources["worker_payload_bytes"].as_u64().unwrap()
                            + resources["client_tensor_bytes"].as_u64().unwrap()
                    );
                }
                let publication = client.publication_bytes().unwrap();
                let pin = client.publication_digest();
                // Owner publication contains fresh commitment randomness, so its
                // digest may differ. Actual-source binding must remain identical.
                sources.push(client.source);
                let starts = client.starts().unwrap();
                assert_eq!(
                    starts[1].len(),
                    starts[0].len() + if seeded { 32 } else { 0 }
                );
                let mut workers = [Worker::Zero, Worker::One].map(|role| {
                    importer(seeded, decoder)
                        .worker(
                            publication.as_bytes(),
                            pin,
                            role,
                            starts[if role == Worker::Zero { 0 } else { 1 }].as_bytes(),
                        )
                        .unwrap()
                });
                let mut result = Vec::new();
                for worker in &workers {
                    assert_eq!(
                        worker
                            .native_publication_resources()
                            .unwrap()
                            .admitted_payload_bytes,
                        priced.admitted_payload_bytes
                    );
                }
                for step in client.calls.clone() {
                    let requests = match step {
                        Step::Lookup => client.lookup(3).unwrap(),
                        Step::Linear(id) => {
                            let stage = client
                                .contract
                                .decoder()
                                .matrices
                                .iter()
                                .position(|matrix| stage_id(matrix) == id)
                                .unwrap();
                            let input =
                                vec![0; client.contract.decoder().matrices[stage].columns * 4];
                            client.linear(stage, &input).unwrap()
                        }
                    };
                    let zero = workers[0].execute(requests[0].as_bytes()).unwrap();
                    let one = workers[1].execute(requests[1].as_bytes()).unwrap();
                    result.push(client.accept(zero.as_bytes(), one.as_bytes()).unwrap());
                }
                let receipt: serde_json::Value =
                    serde_json::from_slice(&client.finish().unwrap()).unwrap();
                assert_eq!(receipt["sharing"], if seeded { "seeded" } else { "full" });
                assert_eq!(receipt["seeded"], seeded);
                assert!(client.finish().is_err());
                for worker in &mut workers {
                    let receipt: serde_json::Value =
                        serde_json::from_slice(&worker.finish().unwrap()).unwrap();
                    assert_eq!(receipt["sharing"], if seeded { "seeded" } else { "full" });
                }
                outputs.push(result);
            }
            assert_eq!(outputs[0], outputs[1]);
            assert_eq!(sources[0], sources[1]);
        }
    }
}
