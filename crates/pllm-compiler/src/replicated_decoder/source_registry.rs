//! Stage-one source checking, independent of expanded execution nodes.
//!
//! The compiler owns validation authority, not caller-created body `Source`s.
//! Retained registry heap backing is zero: descriptors and BlockNorms are borrowed,
//! not cloned. `from_validated` adds no graph/map backing. The caller's immutable
//! manifest/maps (including actual capacities) and original authenticated files
//! remain borrowed external baseline, not zero-cost process memory.
//!
//! `from_manifest` is different: it reuses existing source-shape/metadata lowering
//! admission, which allocates NEW temporary shapes, graphs and validation maps
//! (including the source validator's public-coefficient BTreeSet). Those are NOT
//! borrowed baseline. Their aggregate capacity/construction-overlap peak is not
//! audited here against D26's whole-program 64 MiB ceiling. Existing individual
//! admission checks remain in force; this is not complete-request admission.
//! `descriptor_validation_quote` prices the reachable record-validation seam:
//! its graph/shapes already exist and remain baseline, while request maps and
//! coefficient sets are new temporaries. It does not preadmit graph construction.
//! Any future owned registry backing must be charged before allocation too.

use super::{program, source_norms, Error, Interval, Manifest, Program, SourceNorms, SourceTensor};
use pllm_models::owned_backing;
use std::collections::BTreeMap;
use std::mem::size_of;
use std::sync::Arc;

#[cfg(test)]
thread_local! {
    pub(super) static REGISTRY_HASH_FACTORIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(super) static REGISTRY_RECORD_VALIDATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Native source-owner accounting, not disk reservation or allocator/RSS usage.
/// `retained_baseline_bytes` inventories ALL outside owners (including already
/// parsed Manifest/config/plan/validation backing and any existing PublicSource).
/// Exclude only the source/norm maps inventoried separately here. No alias credit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SourceConstructionQuote {
    pub retained_baseline_bytes: usize,
    pub borrowed_metadata_bytes: usize,
    pub new_owned_peak_bytes: usize,
    pub peak_bytes: usize,
}

/// Original owners live throughout construction. No digest/path alias credit.
/// `retained_baseline_bytes` excludes these explicitly inventoried owners, but
/// includes any other live plan, parsed config document, runtime or frame owner.
/// An existing registry contributes its inline size even though its heap backing
/// is zero. Other DecoderPlan fields must remain charged by their original owner:
/// Digest's private String capacity cannot be inferred from its 64-byte text.
/// Incoming bytes may be serialized input or a source frame; charge capacity,
/// not length. This does not admit parsing or reauthenticate an imported file.
pub(super) struct SourceConstructionOwners<'a> {
    pub manifest: &'a Manifest,
    pub incoming: Option<&'a Vec<u8>>,
    pub public: Option<&'a Arc<super::PublicSource>>,
    pub retained_baseline_bytes: usize,
}

impl SourceConstructionOwners<'_> {
    fn outside_maps_bytes(&self, cap: usize) -> Result<usize, Error> {
        // Reject an impossible outside baseline before walking metadata/norms.
        let cap = cap.min(super::MAX_PROGRAM_BYTES);
        if self.retained_baseline_bytes > cap {
            return Err(Error::Bounds);
        }
        let mut bytes = add_bytes(
            self.retained_baseline_bytes,
            size_of::<Manifest>()
                - size_of::<BTreeMap<String, SourceTensor>>()
                - size_of::<SourceNorms>(),
        )?;
        bytes = add_bytes(bytes, self.manifest.schema.capacity())?;
        bytes = add_bytes(bytes, self.manifest.config.capacity())?;
        bytes = add_bytes(bytes, size_of::<Self>())?;
        if let Some(incoming) = self.incoming {
            bytes = add_bytes(bytes, size_of::<Vec<u8>>())?;
            bytes = add_bytes(bytes, incoming.capacity())?;
        }
        if bytes > cap {
            return Err(Error::Bounds);
        }
        if let Some(public) = self.public {
            bytes = add_bytes(bytes, size_of::<Arc<super::PublicSource>>())?;
            // Arc allocation contains the existing PublicSource and two counts.
            // Its file Arc allocations are already priced by the source census.
            bytes = add_bytes(bytes, array_bytes::<usize>(2)?)?;
            bytes = add_bytes(bytes, public.resident_metadata_bytes()?)?;
        }
        Ok(bytes)
    }

    /// Lower-bound rejection before any graph, norm or coefficient traversal.
    pub(super) fn preflight_baseline(&self, cap: usize) -> Result<(), Error> {
        if self.retained_baseline_bytes > cap.min(super::MAX_PROGRAM_BYTES) {
            return Err(Error::Bounds);
        }
        if self.outside_maps_bytes(cap)? > cap.min(super::MAX_PROGRAM_BYTES) {
            return Err(Error::Bounds);
        }
        Ok(())
    }
}

pub(super) fn add_bytes(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Bounds)
}

pub(super) fn array_bytes<T>(capacity: usize) -> Result<usize, Error> {
    owned_backing::checked_array_bytes::<T>(capacity).map_err(|_| Error::Bounds)
}

/// Same conservative full-node-per-entry policy as source_norms, including
/// unused key/value slots, child edges and split/construction overlap.
pub(super) fn tree_entry_bytes<T>() -> Result<usize, Error> {
    owned_backing::checked_tree_node_bytes::<T>().map_err(|_| Error::Bounds)
}

/// Actual existing String/Vec capacities, NOT logical tensor file length.
/// Census only: do not traverse certificate blocks before admitting this owner.
/// The original validator/from_records still checks geometry and norm contents.
pub(super) fn source_metadata_bytes(
    sources: &BTreeMap<String, SourceTensor>,
    norms: &SourceNorms,
) -> Result<usize, Error> {
    let mut bytes = size_of::<BTreeMap<String, SourceTensor>>();
    for (key, source) in sources {
        bytes = add_bytes(bytes, tree_entry_bytes::<(String, SourceTensor)>()?)?;
        bytes = add_bytes(bytes, key.capacity())?;
        bytes = add_bytes(bytes, array_bytes::<usize>(source.shape.capacity())?)?;
        if let Some(values) = &source.public_q18 {
            bytes = add_bytes(bytes, array_bytes::<i64>(values.capacity())?)?;
        }
    }
    source_norms::metadata_bytes(norms)?;
    bytes = add_bytes(bytes, size_of::<SourceNorms>())?;
    for (key, record) in norms {
        bytes = add_bytes(bytes, tree_entry_bytes::<(String, super::BlockNorms)>()?)?;
        bytes = add_bytes(bytes, key.capacity())?;
        bytes = add_bytes(bytes, record.metadata_bytes()?)?;
    }
    Ok(bytes)
}

/// Binding hashes stream original records; no serialized Vec is constructed.
/// Price simultaneous outer/norm SHA states, return digest and result slots.
pub(super) fn binding_workspace_bytes() -> Result<usize, Error> {
    add_bytes(
        array_bytes::<super::Sha256>(3)?,
        add_bytes(
            array_bytes::<[u8; 32]>(3)?,
            size_of::<Result<[u8; 32], Error>>(),
        )?,
    )
}

fn shape_metadata_bytes(shapes: &BTreeMap<String, Vec<usize>>) -> Result<usize, Error> {
    let mut bytes = size_of::<BTreeMap<String, Vec<usize>>>();
    for (key, shape) in shapes {
        bytes = add_bytes(bytes, tree_entry_bytes::<(String, Vec<usize>)>()?)?;
        bytes = add_bytes(bytes, key.capacity())?;
        bytes = add_bytes(bytes, array_bytes::<usize>(shape.capacity())?)?;
    }
    Ok(bytes)
}

/// Existing graph owner, including spare Vec capacity and nested attributes.
/// Not a quote for allocating/lowering this graph. Other phases/DecoderPlan
/// fields remain the enclosing caller's retained baseline.
pub(super) fn semantic_metadata_bytes(graph: &pllm_models::DecoderGraph) -> Result<usize, Error> {
    // Preserve this wrapper's census-only contract: callers enforce their
    // existing caps after adding the other retained/temporary owners. Shared
    // backing excludes inline slots; this standalone graph slot is added once.
    add_bytes(
        size_of::<pllm_models::DecoderGraph>(),
        graph
            .checked_owned_backing_bytes(usize::MAX)
            .map_err(|_| Error::Bounds)?,
    )
}

/// Complete existing DecoderPlan owner: one inline plan slot plus all owned
/// backing, including both graphs. Borrowing a graph creates no additional owner.
/// External config/input/source owners remain separate. This allocation-free
/// census does not preadmit lowering, validate semantics, or confer authority.
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "complete plan owner census is staged")
)]
pub(super) fn decoder_plan_metadata_bytes(
    plan: &pllm_models::DecoderPlan,
    cap: usize,
) -> Result<usize, Error> {
    let inline = size_of::<pllm_models::DecoderPlan>();
    let backing_cap = cap
        .min(super::MAX_PROGRAM_BYTES)
        .checked_sub(inline)
        .ok_or(Error::Bounds)?;
    add_bytes(
        inline,
        plan.checked_owned_backing_bytes(backing_cap)
            .map_err(|_| Error::Bounds)?,
    )
}

/// Remainder ONLY when semantic_metadata_bytes(graph) already charges this exact
/// plan field's backing AND inline graph slot. Adding that unchanged wrapper to
/// this remainder equals the complete plan census, with neither duplicated.
/// Identity is pointer identity, never structural/digest equality. A cloned or
/// foreign graph cannot receive backing credit. The cap applies to the COMPLETE
/// plan before subtraction, not just the remainder; outside owners still need
/// their enclosing aggregate cap. No owner is retired or allocation admitted.
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "complete plan owner census is staged")
)]
pub(super) fn decoder_plan_metadata_outside_graph_bytes(
    plan: &pllm_models::DecoderPlan,
    graph: &pllm_models::DecoderGraph,
    cap: usize,
) -> Result<usize, Error> {
    if !std::ptr::eq(graph, &plan.prefill) && !std::ptr::eq(graph, &plan.decode) {
        return Err(Error::Binding);
    }
    decoder_plan_metadata_bytes(plan, cap)?
        .checked_sub(semantic_metadata_bytes(graph)?)
        .ok_or(Error::Bounds)
}

/// Preflight immediately before program::validate_source_records, not before
/// Program::validated_source_descriptors. Original validation must still mint
/// the token. No coefficient hashing, certificate/source-file scan or allocation
/// occurs here. Existing norm validation retains its original contract.
///
/// Conservatively prices full tree nodes (same resource policy as norm import),
/// all possible Linear request keys, and duplicate-insert key/record overlap.
/// Request producers/results overlap; the coefficient set is a later phase, so
/// take their maximum rather than charging a fictitious simultaneous owner.
pub(super) fn descriptor_validation_quote(
    graph: &pllm_models::DecoderGraph,
    shapes: &BTreeMap<String, Vec<usize>>,
    owners: &SourceConstructionOwners<'_>,
    cap: usize,
) -> Result<SourceConstructionQuote, Error> {
    use pllm_models::{ModelOperation, ModelOperator as Op};
    owners.preflight_baseline(cap)?;
    let mut retained = add_bytes(
        owners.outside_maps_bytes(cap)?,
        semantic_metadata_bytes(graph)?,
    )?;
    retained = add_bytes(retained, shape_metadata_bytes(shapes)?)?;
    let mut coefficients = size_of::<std::collections::BTreeSet<&str>>();
    let mut coefficient_count = 0usize;
    let mut requests = size_of::<BTreeMap<String, (usize, usize, usize)>>();
    let mut largest_key = 0;
    for op in &graph.operations {
        if op.operator == Op::RmsNorm {
            coefficients = add_bytes(coefficients, tree_entry_bytes::<&str>()?)?;
            coefficient_count = add_bytes(coefficient_count, 1)?;
        }
        if op.operator == Op::Linear {
            for field in ["weight", "bias"] {
                if let Some(key) = op.attributes.get(field).and_then(|v| v.as_str()) {
                    requests = add_bytes(
                        requests,
                        tree_entry_bytes::<(String, (usize, usize, usize))>()?,
                    )?;
                    requests = add_bytes(requests, key.len())?;
                    largest_key = largest_key.max(key.len());
                }
            }
        }
    }
    // std's collect/bulk-build may stage entries while building tree nodes.
    // Price destination growth (at most 2*count, minimum 4), simultaneously live
    // old allocation (at most count), and sorting scratch (at most count).
    // The sum deliberately covers both sequential construction phases instead
    // of mistaking the final tree census for its peak. No source owner refund.
    let staging_capacity = |count: usize| -> Result<usize, Error> {
        if count == 0 {
            Ok(0)
        } else {
            add_bytes(add_bytes(count, count)?.max(4), add_bytes(count, count)?)
        }
    };
    coefficients = add_bytes(
        coefficients,
        array_bytes::<&str>(staging_capacity(coefficient_count)?)?,
    )?;
    let request_peak = if owners.manifest.norms.is_empty() {
        0 // validate_set returns before constructing requests/producers.
    } else {
        let producers = add_bytes(
            size_of::<BTreeMap<&str, &ModelOperation>>(),
            graph
                .operations
                .len()
                .checked_mul(tree_entry_bytes::<(&str, &ModelOperation)>()?)
                .ok_or(Error::Bounds)?,
        )?;
        let producer_staging =
            array_bytes::<(&str, &ModelOperation)>(staging_capacity(graph.operations.len())?)?;
        let producer_build = add_bytes(
            producer_staging,
            size_of::<BTreeMap<String, (usize, usize, usize)>>(),
        )?;
        let request_build = add_bytes(
            requests,
            add_bytes(largest_key, size_of::<(String, (usize, usize, usize))>())?,
        )?;
        add_bytes(producers, producer_build.max(request_build))?
    };
    let temporary = add_bytes(
        coefficients.max(request_peak),
        add_bytes(
            binding_workspace_bytes()?,
            add_bytes(
                size_of::<SourceConstructionQuote>(),
                add_bytes(
                    size_of::<Result<SourceConstructionQuote, Error>>(),
                    add_bytes(
                        size_of::<source_norms::ImportEnvelope>(),
                        size_of::<Result<program::ValidatedSourceDescriptors<'_>, String>>(),
                    )?,
                )?,
            )?,
        )?,
    )?;
    let quote = SourceConstructionQuote::new(
        &owners.manifest.tensors,
        &owners.manifest.norms,
        retained,
        temporary,
    )?;
    quote.admit(cap)?;
    Ok(quote)
}

impl SourceConstructionQuote {
    pub(super) fn new(
        sources: &BTreeMap<String, SourceTensor>,
        norms: &SourceNorms,
        retained_baseline_bytes: usize,
        new_owned_peak_bytes: usize,
    ) -> Result<Self, Error> {
        // Overflowing/impossible fixed baseline never needs a norm traversal.
        let baseline_peak = add_bytes(retained_baseline_bytes, new_owned_peak_bytes)?;
        if baseline_peak > super::MAX_PROGRAM_BYTES {
            return Err(Error::Bounds);
        }
        let borrowed_metadata_bytes = source_metadata_bytes(sources, norms)?;
        let peak_bytes = add_bytes(baseline_peak, borrowed_metadata_bytes)?;
        Ok(Self {
            retained_baseline_bytes,
            borrowed_metadata_bytes,
            new_owned_peak_bytes,
            peak_bytes,
        })
    }

    pub(super) fn admit(&self, cap: usize) -> Result<(), Error> {
        if self.peak_bytes > cap.min(super::MAX_PROGRAM_BYTES) {
            Err(Error::Bounds)
        } else {
            Ok(())
        }
    }
}

/// A borrowed file obligation. Only SourceRegistry's compiler-validated records
/// confer descriptor authority; constructing this view alone does not.
pub(super) struct FileDescriptor<'a> {
    pub key: &'a str,
    pub digest: [u8; 32],
    pub lanes: usize,
}

/// Stable source owner order matches expanded PublicInput construction: ordered
/// source keys, omitting public affine coefficients. Statistics and BlockNorms
/// remain attached to their original key, never a caller's numeric shortcut.
pub(super) struct SourceRegistry<'a> {
    sources: &'a BTreeMap<String, SourceTensor>,
    norms: &'a SourceNorms,
    binding: [u8; 32],
}

impl<'a> SourceRegistry<'a> {
    /// Original source-descriptor validation, not expanded Program compilation.
    /// Reuses allocating metadata/lowering validation, not the allocation-free
    /// borrowing step alone; see the module's construction-audit limitation.
    /// Manifest identity/program digest and complete execution admission remain
    /// the enclosing compiler's obligations. PublicSource::check_registry must
    /// authenticate this contract against the original imported file owner.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "D26 registry construction is not yet promoted")
    )]
    pub(super) fn from_manifest(manifest: &'a Manifest) -> Result<Self, String> {
        Self::from_validated(Program::validated_source_descriptors(manifest)?)
            .map_err(|e| e.to_string())
    }

    pub(super) fn from_validated(
        validated: program::ValidatedSourceDescriptors<'a>,
    ) -> Result<Self, Error> {
        let (sources, norms) = validated.public_records()?;
        Self::from_records(sources, norms)
    }

    /// Allocation-free quote for the borrowing/hash step ONLY. Calling
    /// Program::validated_source_descriptors beforehand does not preadmit its
    /// source-shape/lowering/validation allocations. Parent must gate those at
    /// their original construction sites and retain their peak separately.
    pub(super) fn construction_quote(
        sources: &BTreeMap<String, SourceTensor>,
        norms: &SourceNorms,
        retained_baseline_bytes: usize,
    ) -> Result<SourceConstructionQuote, Error> {
        let mut owned = binding_workspace_bytes()?;
        for bytes in [
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<SourceConstructionQuote>(),
            size_of::<Result<SourceConstructionQuote, Error>>(),
            size_of::<source_norms::ImportEnvelope>(),
        ] {
            owned = add_bytes(owned, bytes)?;
        }
        SourceConstructionQuote::new(sources, norms, retained_baseline_bytes, owned)
    }

    /// Opt-in compiler entry. No owned metadata, graph, file or hash buffer is
    /// allocated here. The unforgeable validation token retains the original
    /// descriptor checks; this is not a shortcut from a hostile Manifest.
    #[cfg(test)]
    pub(super) fn from_validated_checked(
        validated: program::ValidatedSourceDescriptors<'a>,
        retained_baseline_bytes: usize,
        cap: usize,
    ) -> Result<Self, Error> {
        if retained_baseline_bytes > cap.min(super::MAX_PROGRAM_BYTES) {
            return Err(Error::Bounds);
        }
        let (sources, norms) = validated.public_records()?;
        Self::construction_quote(sources, norms, retained_baseline_bytes)?.admit(cap)?;
        Self::from_records(sources, norms)
    }

    pub(super) fn construction_quote_with_owners(
        owners: &SourceConstructionOwners<'_>,
        cap: usize,
    ) -> Result<SourceConstructionQuote, Error> {
        owners.preflight_baseline(cap)?;
        let quote = Self::construction_quote(
            &owners.manifest.tensors,
            &owners.manifest.norms,
            owners.outside_maps_bytes(cap)?,
        )?;
        quote.admit(cap)?;
        Ok(quote)
    }

    /// Borrowed Manifest/import-owner aware variant. The token must describe
    /// these exact maps, not a foreign same-digest owner whose spare capacity was
    /// omitted from the quote. Authentication still belongs to check_registry on
    /// the original immutable PublicSource; no replacement source is constructed.
    pub(super) fn from_validated_with_owners_checked(
        validated: program::ValidatedSourceDescriptors<'a>,
        owners: &SourceConstructionOwners<'a>,
        cap: usize,
    ) -> Result<Self, Error> {
        owners.preflight_baseline(cap)?;
        let (sources, norms) = validated.public_records()?;
        if !std::ptr::eq(sources, &owners.manifest.tensors)
            || !std::ptr::eq(norms, &owners.manifest.norms)
        {
            return Err(Error::Binding);
        }
        Self::construction_quote_with_owners(owners, cap)?;
        Self::from_records(sources, norms)
    }

    fn from_records(
        sources: &'a BTreeMap<String, SourceTensor>,
        norms: &'a SourceNorms,
    ) -> Result<Self, Error> {
        #[cfg(test)]
        REGISTRY_RECORD_VALIDATIONS.with(|n| n.set(n.get() + 1));
        // Existing capacity-aware norm validation; no new norm owner is made.
        source_norms::import_envelope(norms)?;
        for source in sources.values() {
            Interval::new(source.lower, source.upper)?;
            if source.lower < -(1 << 40) || source.upper > 1 << 40 {
                return Err(Error::Bounds);
            }
        }
        #[cfg(test)]
        REGISTRY_HASH_FACTORIES.with(|n| n.set(n.get() + 1));
        Ok(Self {
            sources,
            norms,
            binding: program::source_binding_with_norms(sources, norms),
        })
    }

    pub(super) fn source_records(&self) -> &'a BTreeMap<String, SourceTensor> {
        self.sources
    }

    pub(super) fn source_norms(&self) -> &'a SourceNorms {
        self.norms
    }

    pub(super) fn binding(&self) -> [u8; 32] {
        self.binding
    }

    /// Retained registry heap backing, including new norm ownership: zero.
    /// Not a bound on allocating from_manifest validation temporaries. The inline
    /// handle itself must still be charged by its enclosing owner.
    pub(super) fn owned_backing_bytes(&self) -> usize {
        0
    }

    pub(super) fn records(
        &self,
    ) -> impl Iterator<
        Item = (
            usize,
            &'a str,
            &'a SourceTensor,
            Option<&'a super::BlockNorms>,
        ),
    > + '_ {
        let norms = self.norms;
        self.sources
            .iter()
            .filter(|(_, source)| source.public_q18.is_none())
            .enumerate()
            .map(move |(owner, (key, source))| (owner, key.as_str(), source, norms.get(key)))
    }

    pub(super) fn files(&self) -> impl Iterator<Item = FileDescriptor<'a>> + '_ {
        self.records().map(|(_, key, source, _)| FileDescriptor {
            key,
            digest: source.digest,
            lanes: source.shape.iter().product(),
        })
    }

    /// Match every unchecked body Source field before execution can use it.
    pub(super) fn check_descriptor(
        &self,
        owner: usize,
        key: &str,
        digest: [u8; 32],
        lanes: usize,
        bounds: Interval,
    ) -> Result<(), Error> {
        let (_, expected_key, source, _) = self.records().nth(owner).ok_or(Error::Binding)?;
        if key != expected_key
            || digest != source.digest
            || lanes != source.shape.iter().product::<usize>()
            || bounds != Interval::new(source.lower, source.upper)?
        {
            return Err(Error::Binding);
        }
        Ok(())
    }
}
