//! Source- and Experiment-bound adapters for the bounded full-share wire owners.
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

/// One complete SDK request. External I/O cannot detach the native pending
/// transaction; all state-changing errors retire the owner even when ignored.
pub struct GkmTransportClient {
    contract: GkmRuntimeContract,
    local: BTreeMap<String, Vec<u8>>,
    publication: Option<Arc<Publication>>,
    window: Option<ClientWireWindow>,
    calls: Vec<Step>,
    ordinal: usize,
    pending: Option<Step>,
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
            Publication::publish(source, &stages, &table).map_err(|error| error.to_string())?,
        );
        drop(stages);
        drop(matrices);
        drop(table);
        let publication_digest = publication.digest();
        let context = request_context(&contract, source);
        let client = gkm::Client::new(Arc::clone(&publication), publication_digest)
            .map_err(|error| error.to_string())?;
        let window = client
            .begin_wire(context, &calls)
            .map_err(|error| error.to_string())?;
        let request = window.request_id();
        Ok(Self {
            contract,
            local,
            publication: Some(publication),
            window: Some(window),
            calls,
            ordinal: 0,
            pending: None,
            source,
            context,
            publication_digest,
            request,
        })
    }

    pub fn contract_bytes(&self) -> Vec<u8> {
        self.contract.canonical_bytes()
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

    pub fn request_id(&self) -> [u8; 32] {
        self.request
    }

    pub fn publication_bytes(&mut self) -> Result<WireBytes, String> {
        let result = self
            .publication
            .as_ref()
            .ok_or_else(|| reject("retired"))
            .and_then(|publication| publication.export_wire().map_err(|error| error.to_string()));
        if result.is_err() {
            self.abort();
        }
        result
    }

    /// Role-specific public start frames, once, before any input issuance.
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
        let result = (|| {
            if self.pending.is_some() || self.calls.get(self.ordinal) != Some(&Step::Lookup) {
                return Err(reject("transport lookup order"));
            }
            let window = self.window.as_mut().ok_or_else(|| reject("retired"))?;
            let requests = window.lookup(token).map_err(|error| error.to_string())?;
            self.pending = Some(Step::Lookup);
            Ok(requests)
        })();
        if result.is_err() {
            self.abort();
        }
        result
    }

    /// The same finite f32 -> f64 -> nearest-even/2048 -> signed-i64 boundary as
    /// the native reference. No bias, clipping, per-row scale or W8A8 conversion.
    pub fn linear(&mut self, stage: usize, input_f32_le: &[u8]) -> Result<[WireBytes; 2], String> {
        let result = (|| {
            let spec = self
                .contract
                .decoder()
                .matrices
                .get(stage)
                .ok_or_else(|| reject("stage"))?;
            let step = Step::Linear(stage_id(spec));
            if self.pending.is_some()
                || self.calls.get(self.ordinal) != Some(&step)
                || input_f32_le.len() != spec.columns * 4
            {
                return Err(reject("transport linear order or input shape"));
            }
            let window = self.window.as_mut().ok_or_else(|| reject("retired"))?;
            let mut input = Zeroizing::new(Vec::new());
            input
                .try_reserve_exact(spec.columns)
                .map_err(|_| reject("allocation"))?;
            for bytes in input_f32_le.chunks_exact(4) {
                input.push(
                    gkm::encode_input(f64::from(f32::from_le_bytes(bytes.try_into().unwrap())))
                        .map_err(|error| error.to_string())?,
                );
            }
            let requests = window.linear(&input).map_err(|error| error.to_string())?;
            self.pending = Some(step);
            Ok(requests)
        })();
        if result.is_err() {
            self.abort();
        }
        result
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
            self.ordinal += 1;
            Ok(bytes)
        })();
        if result.is_err() {
            self.abort();
        }
        result
    }

    pub fn finish(&mut self) -> Result<Vec<u8>, String> {
        let result = (|| {
            let mut window = self.window.take().ok_or_else(|| reject("retired"))?;
            if self.pending.is_some() || self.ordinal != self.calls.len() {
                return Err(reject("transport incomplete request"));
            }
            let accepted = window.finish().map_err(|error| error.to_string())?;
            if accepted.sharing_profile() != gkm::SharingProfile::FullShares
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
                "steps": accepted.steps(), "sharing": "full", "seeded": false,
            })))
        })();
        self.abort();
        result
    }

    pub fn abort(&mut self) {
        self.window = None;
        self.publication = None;
        self.pending = None;
        self.local.clear();
    }
}

/// One party-local request owner. Public source copies are held only by the
/// core `WorkerWire`; source-import temporaries and local coefficients retire
/// before this constructor returns. No private client input/token is retained.
pub struct GkmTransportWorker {
    contract: GkmRuntimeContract,
    window: Option<gkm::WorkerWireWindow>,
    owner: Option<gkm::WorkerWire>,
    source: [u8; 32],
    context: [u8; 32],
    publication: [u8; 32],
    role: Worker,
    steps: usize,
}

impl GkmTransportWorker {
    pub fn request_id(&self) -> Result<[u8; 32], String> {
        self.window
            .as_ref()
            .map(gkm::WorkerWireWindow::request_id)
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
        Publication::import_wire(publication, pin).map_err(|error| error.to_string())?;
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
            Publication::import_wire(publication, pin).map_err(|error| error.to_string())?,
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
        let ids: Vec<_> = contract.decoder().matrices.iter().map(stage_id).collect();
        let mut owner = gkm::WorkerWire::from_owned(publication, pin, role, &ids, matrices, table)
            .map_err(|error| error.to_string())?;
        drop(local);
        let window = owner
            .begin_bound(start, context, &calls)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            contract,
            window: Some(window),
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
                "sharing": "full",
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
