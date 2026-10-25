//! Consuming source-factory -> original request join. No serialized plan,
//! executable oracle, caller-selected coordinates or Python-issued permit.
use super::*;
use crate::original_source::{self, OriginalSourcePlan};
use crate::plan_input::json;
use crate::replicated_decoder::source_registry::{self, SourceConstructionOwners, SourceRegistry};
use pllm_garble::shared_allocation::OwnedExternalCharge;
use pllm_garble::shared_allocation::{OwnedBudget, OwnedCharged, MAX_ALLOCATION_BYTES};

#[path = "source_entry_compact_pending.rs"]
pub(in crate::replicated_decoder::program) mod compact_pending;
use serde::Deserialize;

fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Bounds)
}
fn bytes<T>(n: usize) -> Result<usize, Error> {
    n.checked_mul(size_of::<T>()).ok_or(Error::Bounds)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ImportQuote {
    pub payload_bytes: usize,
    pub metadata_work: u64,
    pub declarations: usize,
}

fn check_source(source: &OriginalSourcePlan, manifest: &Manifest) -> Result<(), Error> {
    let workload = source.workload();
    if manifest.schema != "pllm.replicated-decoder.manifest.v1"
        || manifest.weights != WeightMode::Public
        || manifest.execution.checked_bound().is_none()
        || manifest.source == [0; 32]
        || manifest.composition == [0; 32]
        || manifest.session == [0; 32]
        || source.raw_configuration() != manifest.config
        || workload.batch != 1
        || workload.max_input_tokens != u64::try_from(manifest.rows).map_err(|_| Error::Bounds)?
        || workload.max_new_tokens != u64::try_from(manifest.outputs).map_err(|_| Error::Bounds)?
    {
        return Err(Error::Binding);
    }
    Ok(())
}

/// Input factories only; graph/native emission retains the original Builder's
/// admission contract. No complete CPU/RSS or compact-capacity claim.
pub(super) fn quote_imports(
    original: &OriginalSourcePlan,
    manifest: &Manifest,
    registry: &SourceRegistry<'_>,
    public: &Arc<PublicSource>,
    guard: &OwnedExternalCharge,
) -> Result<ImportQuote, Error> {
    if !guard.is_for(original.budget()) {
        return Err(Error::Binding);
    }
    check_source(original, manifest)?;
    let mut declarations = 0usize;
    let mut payload = add(size_of::<CreatedInputs<'_>>(), size_of::<Builder>())?;
    let mut work = 1usize;
    for (key, source) in &manifest.tensors {
        work = add(work, add(key.len(), source.shape.len())?)?;
        if source.public_q18.is_some() {
            continue;
        }
        declarations = add(declarations, 1)?;
        // Separate actual strings in Code::PublicInput and the source-view map.
        payload = add(
            payload,
            bytes::<u8>(key.capacity().checked_mul(2).ok_or(Error::Bounds)?)?,
        )?;
        payload = add(payload, bytes::<usize>(source.shape.capacity())?)?;
        payload = add(
            payload,
            source_registry::tree_entry_bytes::<(String, SourceView)>()?,
        )?;
    }
    // Reserve leading public declarations and the genuine token Input together.
    // request_origin::declare owns/prices the token record/Words/shape factory.
    payload = add(payload, bytes::<Node>(add(declarations, 1)?)?)?;
    payload = add(payload, size_of::<BTreeMap<String, SourceView>>())?;
    // Same conservative norm-copy envelope as original expanded import. Do not
    // discount shared/equal records. This covers actual norm map spare backing.
    let norm_bytes = source_norms::import_envelope(&manifest.norms)?.total_bytes;
    payload = add(payload, norm_bytes)?;
    work = add(work, norm_bytes)?;
    for n in [
        size_of::<ImportQuote>(),
        size_of::<Result<ImportQuote, Error>>(),
        size_of::<request_origin::RequestSources<'_>>(),
        size_of::<Result<CreatedInputs<'_>, Error>>(),
        size_of::<SourceView>(),
        size_of::<Code>(),
        size_of::<Node>(),
        OriginalSourcePlan::compiler_metadata_control_bytes().map_err(|_| Error::Bounds)?,
    ] {
        payload = add(payload, n)?;
    }
    // Empty policy-identical Builder overquotes token-node growth: the real
    // import pool reserves its token slot already. No fake declarations/views.
    let seed = empty_builder(manifest, registry);
    let sources = request_origin::RequestSources::new(manifest, registry, Arc::clone(public))?;
    let request = request_origin::quote(&seed, original, &sources)?;
    payload = add(
        payload,
        add(request.payload_bytes, size_of::<OwnedExternalCharge>())?,
    )?;
    let metadata_work = u64::try_from(work)
        .map_err(|_| Error::Bounds)?
        .checked_add(request.metadata_work.checked_mul(2).ok_or(Error::Bounds)?)
        .ok_or(Error::Bounds)?;
    Ok(ImportQuote {
        payload_bytes: payload,
        metadata_work,
        declarations,
    })
}

fn empty_builder(manifest: &Manifest, registry: &SourceRegistry<'_>) -> Builder {
    Builder {
        public_weights: true,
        checked_bound: manifest.execution.checked_bound(),
        lookup: manifest.execution.lookup,
        selection: manifest.execution.selection,
        token_source_binding: if manifest.execution.lookup == LookupPolicy::Trie {
            registry.binding()
        } else {
            [0; 32]
        },
        ..Builder::default()
    }
}

/// This aggregate is never returned to Python. Drop order retires the emitted
/// input/map/permit owners before their original continuous factory reservation.
pub(super) struct CreatedInputs<'s> {
    pub(super) builder: Builder,
    pub(super) inputs: BTreeMap<String, SourceView>,
    pub(super) permit: request_origin::RequestInputPermit<'s>,
    pub(super) charge: OwnedExternalCharge,
    // Only the prequoted actual input-emission path can mint this aggregate.
    created: (),
}

/// Consumes source + admission on every outcome: no retry token/borrowed plan is
/// returned on failure. Request origin's existing Attempt burns later failures.
pub(super) fn create_inputs<'s>(
    original: OriginalSourcePlan,
    manifest: &'s Manifest,
    registry: &'s SourceRegistry<'s>,
    public: Arc<PublicSource>,
    mut charge: OwnedExternalCharge,
) -> Result<CreatedInputs<'s>, Error> {
    let q = quote_imports(&original, manifest, registry, &public, &charge)?;
    if charge.payload_bytes() < q.payload_bytes {
        return Err(Error::Bounds);
    }
    original
        .prepay_compiler_metadata(&charge, q.metadata_work)
        .map_err(|_| Error::Bounds)?;
    public.check_registry(registry)?;
    let sources = request_origin::RequestSources::new(manifest, registry, public)?;
    original
        .budget()
        .admit_peak(original.budget().used_bytes())
        .map_err(|_| Error::Bounds)?;
    let mut builder = empty_builder(manifest, registry);
    builder.allocated = source_norms::import_envelope(&manifest.norms)?.total_bytes;
    let node_capacity = add(q.declarations, 1)?;
    builder
        .nodes
        .try_reserve_exact(node_capacity)
        .map_err(|_| Error::Allocation)?;
    if builder.nodes.capacity() != node_capacity {
        return Err(Error::Allocation);
    }
    let mut inputs = BTreeMap::new();
    for (key, source) in &manifest.tensors {
        if source.public_q18.is_some() {
            continue;
        }
        let lanes = source
            .shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d).ok_or(Error::Bounds))?;
        let owner = builder.push(
            Code::PublicInput {
                key: key.clone(),
                digest: source.digest,
                row_l1: source.row_l1,
                row_squared_norm: source.row_squared_norm,
                row_l1_squared_sum: source.row_l1_squared_sum,
            },
            lanes,
            Interval::new(source.lower, source.upper)?,
        )?;
        inputs.insert(
            key.clone(),
            SourceView {
                owner,
                shape: source.shape.clone(),
            },
        );
        if let Some(norm) = manifest.norms.get(key) {
            builder.public_norms.insert(owner, norm.clone());
        }
    }
    let request = request_origin::quote(&builder, &original, &sources)?;
    let declaration = charge
        .split_owned(request.payload_bytes)
        .map_err(|_| Error::Bounds)?;
    let permit = request_origin::declare(&mut builder, original, sources, declaration)?;
    Ok(CreatedInputs {
        builder,
        inputs,
        permit,
        charge,
        created: (),
    })
}

/// No temporary Manifest/registry borrow escapes: finalization moves genuine
/// source/epoch/sticky/public-file custody into the SAME Program used by Worker.
/// Registry construction is SolBIAS's shared original-plan checked helper.
pub(in crate::replicated_decoder) struct EntryGuards {
    pub registry: OwnedExternalCharge,
    pub imports: OwnedExternalCharge,
    pub controller: OwnedExternalCharge,
}

pub(in crate::replicated_decoder) fn compile(
    original: OriginalSourcePlan,
    owners: SourceConstructionOwners<'_>,
    public: Arc<PublicSource>,
    guards: EntryGuards,
    limits: original_source::Limits,
    cap: usize,
) -> Result<Arc<Program>, Error> {
    #[cfg(test)]
    test_policy_factory_trap();
    let EntryGuards {
        registry: mut registry_guard,
        imports: mut imports_guard,
        mut controller,
    } = guards;
    if !controller.is_for(original.budget()) || !imports_guard.is_for(original.budget()) {
        return Err(Error::Binding);
    }
    let validation = SourceRegistry::original_construction_quote(&original, &owners, cap)?;
    if registry_guard.payload_bytes() < validation.new_owned_peak_bytes {
        registry_guard
            .resize_payload(validation.new_owned_peak_bytes)
            .map_err(|_| Error::Bounds)?;
    }
    let registry =
        SourceRegistry::from_original_checked(&original, &owners, &public, &registry_guard, cap)?;
    let imports = quote_imports(
        &original,
        owners.manifest,
        &registry,
        &public,
        &imports_guard,
    )?;
    if imports_guard.payload_bytes() < imports.payload_bytes {
        imports_guard
            .resize_payload(imports.payload_bytes)
            .map_err(|_| Error::Bounds)?;
    }
    let CreatedInputs {
        builder,
        inputs,
        permit,
        charge,
        created: (),
    } = create_inputs(original, owners.manifest, &registry, public, imports_guard)?;
    let lowering = permit.lowering_control_bytes(limits)?;
    if controller.payload_bytes() < lowering {
        controller
            .resize_payload(lowering)
            .map_err(|_| Error::Bounds)?;
    }
    // The import reservation itself becomes finalization's retained charge.
    // Never drop/reacquire or use a second unrelated counter at this handoff.
    let mut program = request_origin::schedule::compile(
        builder, permit, &inputs, controller, limits, charge, cap,
    )?;
    // Preserve existing binding/framing. Manifest digest is a POSTCONSTRUCTION
    // check only; never a source/plan/import/feedback constructor argument.
    Arc::get_mut(&mut program)
        .ok_or(Error::Binding)?
        .bind(owners.manifest.source, owners.manifest.composition);
    if program.digest() != owners.manifest.program {
        return Err(Error::Binding);
    }
    Ok(program)
}

/// Staged native ingress. The Option is a sticky consuming entry: every append
/// error takes/drops the entire owner, so neither a failed seal nor input can be
/// retried. No reference into an owned field is stored across calls.
pub struct SourceRequestImport(Option<Pending>);
struct Pending {
    importer: PublicSourceImport,
    original: OriginalSourcePlan,
    manifest: OwnedCharged<Manifest>,
    raw: OwnedCharged<json::ImportedValue>,
    limits: original_source::Limits,
    registry: OwnedExternalCharge,
    imports: OwnedExternalCharge,
    controller: OwnedExternalCharge,
    root: OwnedBudget,
}

fn json_limits() -> json::Limits {
    json::Limits {
        max_source_bytes: 1 << 20,
        max_nodes: 65536,
        max_depth: 16,
        max_string_bytes: 1 << 20,
        max_number_bytes: 64,
        max_number_work: 100_000_000,
        max_work: 1_000_000_000,
        max_payload_bytes: 8 << 20,
    }
}
fn source_limits() -> original_source::Limits {
    original_source::Limits {
        expansion: DecoderExpansionBudget {
            max_layers: 8,
            max_feature_width: 128,
            max_attention_heads: 8,
            max_vocabulary: 128,
            max_tensor_elements: 32768,
        },
        factory: pllm_models::ConfigLowererLimits {
            max_operations_per_phase: 256,
            max_string_bytes: 1024,
            max_attribute_nodes: 32768,
            max_retained_strings: 1 << 20,
            max_factory_bytes: MAX_ALLOCATION_BYTES,
        },
        max_work: 1_000_000_000_000,
        max_working_payload: MAX_ALLOCATION_BYTES,
    }
}

fn server_controller_bytes() -> usize {
    size_of::<Option<(&[u8], &[u8])>>()
        + size_of::<Option<OwnedCharged<json::ImportedValue>>>()
        + size_of::<crate::replicated_decoder::source_policy::ServerSourcePolicy<'_>>()
        + size_of::<Result<original_source::Limits, Error>>()
        + size_of::<Result<original_source::SourceQuote, crate::original_source::Error>>()
        + size_of::<original_source::Resources>()
        + size_of::<pllm_models::SourceScalarSummary>()
        + size_of::<Result<pllm_models::SourceScalarSummary, pllm_models::SourceScalarSummaryError>>(
        )
        + size_of::<std::iter::Zip<std::slice::Iter<'_, serde_json::Value>, std::slice::Iter<'_, u8>>>(
        )
        + size_of::<u128>()
        + size_of::<Option<u64>>()
}

#[cfg(test)]
thread_local! { static POLICY_FACTORY_TRAP: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
#[cfg(test)]
pub(in crate::replicated_decoder) struct PolicyFactoryTrap(bool);
#[cfg(test)]
impl PolicyFactoryTrap {
    pub(in crate::replicated_decoder) fn arm() -> Self {
        Self(POLICY_FACTORY_TRAP.with(|v| v.replace(true)))
    }
}
#[cfg(test)]
impl Drop for PolicyFactoryTrap {
    fn drop(&mut self) {
        POLICY_FACTORY_TRAP.with(|v| v.set(self.0));
    }
}
#[cfg(test)]
fn test_policy_factory_trap() {
    POLICY_FACTORY_TRAP.with(|v| {
        assert!(
            !v.get(),
            "metadata-only query entered source/payload/request factory"
        )
    });
}
fn import_json(root: &OwnedBudget, raw: &[u8]) -> Result<OwnedCharged<json::ImportedValue>, Error> {
    let original = root
        .charge_external(raw.len())
        .map_err(|_| Error::Bounds)?
        .try_build(|_| {
            // Only the named borrowed slice participates in this COPY window;
            // its caller's allocation capacity is unknown. The destination and
            // owner control are ALREADY reserved on this same counter.
            root.budget().admit_peak(
                root.budget()
                    .used_bytes()
                    .checked_add(raw.len())
                    .ok_or(pllm_garble::shared_allocation::Error::Resource)?,
            )?;
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(raw.len())
                .map_err(|_| pllm_garble::shared_allocation::Error::Resource)?;
            if bytes.capacity() != raw.len() {
                return Err(pllm_garble::shared_allocation::Error::Resource);
            }
            bytes.extend_from_slice(raw);
            Ok(bytes)
        })
        .map_err(|_| Error::Bounds)?;
    import_json_owner(root, original)
}
fn import_json_owner(
    root: &OwnedBudget,
    original: OwnedCharged<Vec<u8>>,
) -> Result<OwnedCharged<json::ImportedValue>, Error> {
    json::quote(
        original,
        root.charge_external(json::bootstrap_bytes().map_err(|_| Error::Bounds)?)
            .map_err(|_| Error::Bounds)?,
        json_limits(),
    )
    .map_err(|_| Error::Binding)?
    .construct()
    .map_err(|_| Error::Binding)
}
#[cfg(test)]
thread_local! {
    static PUBLIC_OWNER_DROP:std::cell::Cell<Option<(bool,usize,usize)>>=const{std::cell::Cell::new(None)};
}
#[cfg(test)]
pub(in crate::replicated_decoder::program) fn test_public_owner_drop(
) -> Option<(bool, usize, usize)> {
    PUBLIC_OWNER_DROP.with(|state| state.take())
}

// Field drop order: last outer PUBLIC payload, weak observer/control, THEN
// its independent same-counter reservation. No credit from request/import
// guards, which may already have retired inside emit.
// Staged private compact ingress; production dispatch waits for owning Storage.
#[cfg_attr(not(test), allow(dead_code))]
struct PublicSnapshotOwner {
    public: Arc<PublicSource>,
    charge: PublicSnapshotCharge,
}
// Same staged ingress owner; tests exercise payload-before-reservation retirement.
#[cfg_attr(not(test), allow(dead_code))]
struct PublicSnapshotCharge {
    #[cfg(test)]
    weak: std::sync::Weak<PublicSource>,
    guard: OwnedExternalCharge,
}
#[cfg(test)]
impl Drop for PublicSnapshotCharge {
    fn drop(&mut self) {
        PUBLIC_OWNER_DROP.with(|state| {
            state.set(Some((
                self.weak.upgrade().is_none(),
                self.guard.budget().used_bytes(),
                self.guard.charged_bytes(),
            )))
        });
    }
}
impl PublicSnapshotOwner {
    // Not a default-route factory: only the staged consuming ingress calls this.
    #[cfg_attr(not(test), allow(dead_code))]
    fn new(public: PublicSource, root: &OwnedBudget) -> Result<Self, Error> {
        let payload = add(
            add(public.resident_metadata_bytes()?, 2 * size_of::<usize>())?,
            size_of::<Self>(),
        )?;
        let phase = add(
            root.budget().used_bytes(),
            add(payload, size_of::<OwnedExternalCharge>())?,
        )?;
        if phase > MAX_PROGRAM_BYTES {
            return Err(Error::Bounds);
        }
        let guard = root.charge_external(payload).map_err(|_| Error::Bounds)?;
        // Reservation precedes Arc allocation. Its peak is recorded by SAME
        // source counter; descriptor/file backing comes from actual finish.
        let public = Arc::new(public);
        #[cfg(test)]
        let weak = Arc::downgrade(&public);
        Ok(Self {
            public,
            charge: PublicSnapshotCharge {
                #[cfg(test)]
                weak,
                guard,
            },
        })
    }
}

impl SourceRequestImport {
    /// Lexical test adapter over actual production source/import/input factories.
    /// Not a storage publication or an executable entry. The complete compact
    /// entry must take over this custody before production activation.
    #[cfg(test)]
    pub(in crate::replicated_decoder::program) fn with_created<R>(
        self,
        emit: impl FnOnce(CreatedInputs<'_>, OwnedExternalCharge) -> Result<R, Error>,
    ) -> Result<R, Error> {
        self.consume_compact(|created, guard, _, _, _| emit(created, guard))
            .map(compact_pending::SourceCustody::into_test_payload)
    }

    /// Fixed bounded opt-in admission, not a new numeric/protocol choice. Large
    /// or unknown shapes fail closed; old constructors never enter this route.
    pub fn new(raw: &[u8]) -> Result<Self, Error> {
        Self::new_inner(raw, None)
    }

    /// Private native server opt-in. These bytes come from the server's trusted
    /// Experiment/config path, NEVER a client Manifest policy selector. This is
    /// resource permission only; original config factory and public snapshot
    /// verification remain independent and unchanged.
    #[doc(hidden)]
    pub fn new_checked_public(
        raw: &[u8],
        server_experiment: &[u8],
        server_config: &[u8],
    ) -> Result<Self, Error> {
        if server_experiment.is_empty()
            || server_experiment.len() > 32 << 10
            || server_config.is_empty()
            || server_config.len() > crate::replicated_decoder::source_policy::CONFIG_BYTES
        {
            return Err(Error::Bounds);
        }
        Self::new_inner(raw, Some((server_experiment, server_config)))
    }

    /// Nonconstructing original source quote. No checkpoint/tensor reads, native
    /// arithmetic factory, request, worker or child process is part of this API.
    /// Finite memory/phase/work refusal remains a normal typed result.
    /// Returns ORIGINAL source-owner resources only, not aggregate admission:
    /// server C9/policy work is separately removed from its ledger ceiling and
    /// same-counter server owners remain live during the entire query phase.
    #[doc(hidden)]
    pub fn quote_checked_public(
        server_experiment: &[u8],
        server_config: &[u8],
        rows: usize,
        outputs: usize,
    ) -> Result<original_source::Resources, Error> {
        use crate::replicated_decoder::source_policy;
        if server_experiment.is_empty()
            || server_experiment.len() > 32 << 10
            || server_config.is_empty()
            || server_config.len() > source_policy::CONFIG_BYTES
        {
            return Err(Error::Bounds);
        }
        let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).map_err(|_| Error::Bounds)?;
        let _controls = root
            .charge_external(server_controller_bytes())
            .map_err(|_| Error::Bounds)?;
        let server = import_json(&root, server_experiment)?;
        let config = import_json(&root, server_config)?;
        let workload = DecoderWorkload {
            batch: 1,
            max_input_tokens: u64::try_from(rows).map_err(|_| Error::Bounds)?,
            max_new_tokens: u64::try_from(outputs).map_err(|_| Error::Bounds)?,
        };
        let limits = source_policy::admitted_limits(&server, &config, None, workload)?;
        let bootstrap =
            original_source::bootstrap_bytes(&config, limits).map_err(|_| Error::Bounds)?;
        let quote = original_source::quote(
            config,
            root.charge_external(bootstrap).map_err(|_| Error::Bounds)?,
            workload,
            limits,
        )
        .map_err(|_| Error::Bounds)?;
        Ok(quote.resources())
    }

    fn new_inner(raw: &[u8], server: Option<(&[u8], &[u8])>) -> Result<Self, Error> {
        use crate::replicated_decoder::source_policy;
        if raw.is_empty() || raw.len() > 1 << 20 {
            return Err(Error::Bounds);
        }
        let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).map_err(|_| Error::Bounds)?;
        // Additional named server controller/return slots BEFORE C9 factories.
        // Caller CPython backing/allocator/RSS remains outside this logical
        // counter; import_json preserves E11's explicit borrowed-copy window.
        let _server_controls = server
            .map(|_| {
                root.charge_external(server_controller_bytes())
                    .map_err(|_| Error::Bounds)
            })
            .transpose()?;
        let server_document = server
            .map(|(experiment, _)| import_json(&root, experiment))
            .transpose()?;
        let imported = import_json(&root, raw)?;
        let counts = imported.payload().counts();
        let document = imported.payload().value();
        if server.is_some()
            && (document.get("source_policy").is_some() || document.get("source_limits").is_some())
        {
            return Err(Error::Binding);
        }
        let config_values = document
            .get("config")
            .and_then(|v| v.as_array())
            .ok_or(Error::Binding)?;
        if config_values.is_empty() || config_values.len() > 1 << 20 {
            return Err(Error::Bounds);
        }
        let workload = DecoderWorkload {
            batch: 1,
            max_input_tokens: document
                .get("rows")
                .and_then(|v| v.as_u64())
                .ok_or(Error::Binding)?,
            max_new_tokens: document
                .get("outputs")
                .and_then(|v| v.as_u64())
                .ok_or(Error::Binding)?,
        };
        let config = if let Some((_, actual)) = server {
            if config_values.len() != actual.len()
                || !config_values
                    .iter()
                    .zip(actual)
                    .all(|(value, byte)| value.as_u64() == Some(u64::from(*byte)))
            {
                return Err(Error::Binding);
            }
            // The actual SERVER configuration is the original factory input.
            import_json(&root, actual)?
        } else {
            let config = root
                .charge_external(config_values.len())
                .map_err(|_| Error::Bounds)?
                .try_build(|_| {
                    // Config destination, source JSON owner, raw owner and result
                    // overlap are already represented by these live reservations.
                    root.budget().admit_peak(root.budget().used_bytes())?;
                    let mut bytes = Vec::new();
                    bytes
                        .try_reserve_exact(config_values.len())
                        .map_err(|_| pllm_garble::shared_allocation::Error::Resource)?;
                    if bytes.capacity() != config_values.len() {
                        return Err(pllm_garble::shared_allocation::Error::Resource);
                    }
                    for value in config_values {
                        bytes.push(
                            value
                                .as_u64()
                                .and_then(|n| u8::try_from(n).ok())
                                .ok_or(pllm_garble::shared_allocation::Error::Resource)?,
                        );
                    }
                    Ok(bytes)
                })
                .map_err(|_| Error::Binding)?;
            import_json_owner(&root, config)?
        };
        let limits = if let Some(server_document) = &server_document {
            source_policy::admitted_limits(server_document, &config, Some(&imported), workload)?
        } else {
            source_limits()
        };
        let bootstrap =
            original_source::bootstrap_bytes(&config, limits).map_err(|_| Error::Bounds)?;
        let quote = original_source::quote(
            config,
            root.charge_external(bootstrap).map_err(|_| Error::Bounds)?,
            workload,
            limits,
        )
        .map_err(|_| Error::Binding)?;
        #[cfg(test)]
        test_policy_factory_trap();
        let original = quote.construct().map_err(|_| Error::Binding)?;
        // Manifest DTO bound from the SAME admitted immutable C9 tree. Price a
        // complete typed tree node per JSON span (including staging/overwrite),
        // plus all decoded text. This is deliberately conservative, no MAX-cap
        // credit or unbounded serde parse. Borrowed Value deserialization only.
        let node = source_registry::tree_entry_bytes::<(String, SourceTensor)>()?
            .max(source_registry::tree_entry_bytes::<(String, BlockNorms)>()?);
        let dto_bytes = counts
            .spans
            .checked_mul(node)
            .and_then(|n| n.checked_add(counts.decoded_string_bytes))
            .and_then(|n| n.checked_add(size_of::<Manifest>()))
            .ok_or(Error::Bounds)?;
        let dto_guard = root.charge_external(dto_bytes).map_err(|_| Error::Bounds)?;
        // Prepay the borrowed typed-DTO pass under the genuine source's same
        // cumulative ledger before serde allocates. No JSON reparse or Value
        // clone. Opaque serde/private allocator CPU remains unqualified.
        let dto_work = counts
            .spans
            .checked_mul(
                counts
                    .decoded_string_bytes
                    .checked_add(counts.spans)
                    .ok_or(Error::Bounds)?,
            )
            .ok_or(Error::Bounds)?;
        original
            .prepay_compiler_metadata(
                &dto_guard,
                u64::try_from(dto_work).map_err(|_| Error::Bounds)?,
            )
            .map_err(|_| Error::Bounds)?;
        let manifest = dto_guard
            .try_build(|_| {
                root.budget().admit_peak(root.budget().used_bytes())?;
                Manifest::deserialize(imported.payload().value())
                    .map_err(|_| pllm_garble::shared_allocation::Error::Resource)
            })
            .map_err(|_| Error::Binding)?;
        check_source(&original, manifest.payload())?;
        let mut registry = root.charge_external(0).map_err(|_| Error::Bounds)?;
        let mut imports = root.charge_external(0).map_err(|_| Error::Bounds)?;
        let controller = root.charge_external(0).map_err(|_| Error::Bounds)?;
        let owners = SourceConstructionOwners {
            manifest: manifest.payload(),
            incoming: None,
            public: None,
            retained_baseline_bytes: root
                .budget()
                .used_bytes()
                .checked_sub(registry.payload_bytes())
                .ok_or(Error::Bounds)?,
        };
        let q = SourceRegistry::original_construction_quote(&original, &owners, MAX_PROGRAM_BYTES)?;
        registry
            .resize_payload(q.new_owned_peak_bytes)
            .map_err(|_| Error::Bounds)?;
        let descriptors = SourceRegistry::from_original_records_checked(
            &original,
            &owners,
            &registry,
            MAX_PROGRAM_BYTES,
        )?;
        let q = PublicSourceImport::construction_quote(&descriptors, root.budget().used_bytes())?;
        q.admit(MAX_PROGRAM_BYTES)?;
        imports
            .resize_payload(q.new_owned_peak_bytes)
            .map_err(|_| Error::Bounds)?;
        root.budget()
            .admit_peak(root.budget().used_bytes())
            .map_err(|_| Error::Bounds)?;
        let importer = PublicSourceImport::new_checked(
            &descriptors,
            root.budget()
                .used_bytes()
                .checked_sub(imports.payload_bytes())
                .ok_or(Error::Bounds)?,
            MAX_PROGRAM_BYTES,
        )?;
        Ok(Self(Some(Pending {
            importer,
            original,
            manifest,
            raw: imported,
            limits,
            registry,
            imports,
            controller,
            root,
        })))
    }
    pub fn session(&self) -> Result<[u8; 32], Error> {
        Ok(self
            .0
            .as_ref()
            .ok_or(Error::Retired)?
            .manifest
            .payload()
            .session)
    }
    pub fn append(&mut self, key: &str, offset: usize, bytes: &[u8]) -> Result<(), Error> {
        #[cfg(test)]
        test_policy_factory_trap();
        let mut owner = self.0.take().ok_or(Error::Retired)?;
        owner
            .importer
            .append(key, offset, bytes)
            .map_err(|_| Error::Binding)?;
        self.0 = Some(owner);
        Ok(())
    }
    pub fn finish(mut self) -> Result<crate::replicated_decoder::sdk::SourceRequestProgram, Error> {
        let Pending {
            importer,
            original,
            manifest,
            raw,
            limits,
            registry,
            imports,
            controller,
            root,
        } = self.0.take().ok_or(Error::Retired)?;
        root.budget()
            .admit_peak(root.budget().used_bytes())
            .map_err(|_| Error::Bounds)?;
        let public = Arc::new(importer.finish().map_err(|_| Error::Binding)?);
        let result = crate::replicated_decoder::sdk::SourceRequestProgram::construct(
            original,
            manifest,
            public,
            registry,
            imports,
            controller,
            limits,
            MAX_PROGRAM_BYTES,
        );
        drop(raw);
        drop(root);
        result
    }
}

#[cfg(test)]
#[path = "source_entry_tests.rs"]
mod tests;
