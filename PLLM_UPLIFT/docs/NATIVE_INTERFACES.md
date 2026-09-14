# Native component contracts

This document specifies production interfaces. The `native/` directory contains a smaller, functional source slice; declarations here do not imply their production modules are implemented.

## 1. Core records

`NumericType = {domain, signed, storage_bits, scale, rounding, overflow_policy, quantization_groups, bound_certificate}`.

`ProtectionType = Public | ClientPlaintext | MaskedRing{ring, decoder_side_information} | ArithmeticLabel{scheme, parameter_profile, garbling_epoch} | BooleanLabel{scheme, epoch} | AdditiveShare{field, holders, threshold} | Ciphertext{scheme,key_scope} | AttestedPlaintext{approved_measurement}`.

`Value = {id, shape, numeric, protection, permitted_holders, semantic_epoch, state_kind, layout}`.

`MaterialRequest = {schema_id, method_id, parameter_hash, plan_hash, execution_epoch, op_instance, shape, count, lifetime, party_slices, integrity_requirements}`.

`Cost = {source: measured|modelled|author_reported, setup_ns, online_ns, phase_bytes, dependency_rounds, peak_memory, material_units, all_party_compute, uncertainty}`. Unknown is `None`, not zero.

`Assurance = {claim_id, corruption_sets, leakage, experiment_game, proof_source, refinement_status, parameter_review, empirical_tests, unresolved_obligations}`. Never flatten this into a scalar privacy score.

## 2. Rust traits to integrate

```rust
pub trait ModelAdapter: Send + Sync {
    fn inspect(&self, source: &LockedModel, store: &dyn TensorStore) -> Result<ModelDescriptor>;
    fn lower(&self, model: &ModelDescriptor, store: &dyn TensorStore) -> Result<SemanticGraph>;
    fn fixtures(&self, model: &ModelDescriptor) -> Result<Vec<ConformanceCase>>;
}

pub trait Method: Send + Sync {
    fn descriptor(&self) -> &MethodDescriptor;
    fn admit(&self, region: &NumericRegion, policy: &PrivacyContract) -> Admission;
    fn lower(&self, region: &NumericRegion, context: &CompileContext) -> Result<Vec<ProtectedRegion>>;
    fn material(&self, region: &ProtectedRegion) -> Result<Vec<MaterialRequest>>;
    fn obligations(&self, region: &ProtectedRegion) -> Vec<ProofObligation>;
}

pub trait Kernel: Send + Sync {
    fn capability(&self, request: &KernelRequest) -> Admission;
    fn public_setup(&self, constants: &PublicConstants) -> Result<KernelState>;
    fn workspace_bytes(&self, shape: &Shape) -> Result<usize>;
    fn enqueue(&self, ctx: &ExecutionContext, inputs: &[BufferView], outputs: &mut [BufferViewMut], workspace: &mut Workspace) -> Result<Completion>;
}

pub trait PreparationProvider: Send + Sync {
    fn supports(&self, schema: &MaterialSchema) -> Admission;
    fn prepare(&self, ctx: &RoleContext, requests: &[MaterialRequest]) -> Result<PreparedRoleSlices>;
}

pub trait Adversary: Send {
    fn view_contract(&self) -> &AdversaryViewContract;
    fn observe(&mut self, event: PermittedObservation<'_>) -> Result<()>;
    fn action(&mut self, budget: &AttackBudget) -> AttackAction;
    fn finish(self: Box<Self>) -> AttackEvidence;
}
```

The referenced production types are specified contracts, not empty compilable stubs. Implement them in the existing codebase after repository discovery. The included native crates have complete, narrower APIs and unit tests.

## 3. Compile-time failure codes

`E_MODEL_FEATURE`, `E_MODEL_LOCK`, `E_NUMERIC_MANIFEST`, `E_RANGE_CERTIFICATE`, `E_PARTIES`, `E_ONLINE_PREPARATION`, `E_CLIENT_MODEL`, `E_HE`, `E_CONVERSION`, `E_COVERAGE`, `E_METHOD_NOT_INSTALLED`, `E_ASSURANCE`, `E_PARAMETER_PROFILE`, `E_PREPARATION_BUDGET`, `E_DEVICE_KERNEL`, `E_NO_VALIDATED_PLAN`.

Return all useful admission reasons, the first unsupported semantic node and candidate alternatives. Search retains failures and timeouts. No unsupported operation can invoke a trusted plaintext oracle in production.

## 4. Secrets and FFI

Production masks, offsets and active labels use role-owned opaque native buffers. They have no unrestricted serialization or `Debug`; explicit zeroization is best effort under a documented allocator/device policy, not proof that copies were erased. The in-memory fixture record in this package is not such a production secure allocator.

Python references may contain plaintext public test fixtures. Private production input bytes cross once into a native client session, and only authorized final outputs return. Errors and tracing carry operation IDs and public shapes, not secret values. DLPack/NumPy views require validated ownership, lifetime and device-event fences; do not introduce unbounded copies in a supposedly zero-copy API.

The benchmark binary and extension share the same core implementation. Python wrappers deserialize result JSON and select experiments; they do not implement alternative fast paths. A missing native extension is an error. Dependency pins and ABI versions are recorded in each result. PyO3 0.27.2 is pinned in the supplied staging binding based on its documented `detach` API, not asserted to be the latest release.

## 5. Wire and process contracts

Native frames bind version, authenticated peer, session, request, logical/execution plan hashes, stage, shape, material epoch, sequence, payload length and message type. Enforce caps before allocation. The role string is not authentication. Duplicate handling is idempotent for one semantic execution; byte-identical replay does not create a second output or consume another record.

A Rust role agent owns TLS, protocol sockets and worker pools. Python can request start/cancel/status at coarse granularity. Local agents use separate working directories, file permissions and credentials. For remote launch, send public deployment manifests plus role-specific sealed material references, not a bundle of every party's keys through a shared coordinator.

## 6. Native optimization order

Exact scalar implementation → independent reference parity → batched CPU/SIMD → accelerator. Preserve packed public weight storage where supported, tile once across label lanes, and count modular reduction and representation conversion. All unsafe/device kernels need shape, overflow and alignment tests plus sanitizers/fuzzing appropriate to their language boundary. Rust ownership does not establish constant time or cryptographic correctness.

GPU execution uses stream-aware events and synchronization to distinguish enqueue time from completed work. Native cryptographic hash/AES instantiation must match the paper's correlation assumptions; switching a toy SHAKE reference to any fast hash is not automatically proof-preserving.
