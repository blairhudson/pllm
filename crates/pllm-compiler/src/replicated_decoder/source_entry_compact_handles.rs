//! Source handles issued during genuine consuming compact ingress.
//! Positions retain no borrowed addresses. Only the parent authenticates ingress;
//! Instance is an identity check, never a replacement for that authentication.
use super::*;
use std::{collections::btree_map, mem::size_of};

pub(in crate::replicated_decoder::program) struct SourceKey {
    key: String,
    has_norm: bool,
}

pub(in crate::replicated_decoder::program) struct SourceCatalog {
    entries: Vec<SourceKey>,
}

pub(in crate::replicated_decoder::program) struct TensorHandle {
    instance: source_registry::Instance,
    index: usize,
}

pub(in crate::replicated_decoder::program) struct NormHandle {
    instance: source_registry::Instance,
    tensor_index: usize,
}

/// Borrowed issuance window, constructed ONLY by compact_pending after its real
/// original-plan/public/map/counter checks. No recipe supplies an Instance or
/// Manifest. This scope cannot escape the consuming funnel's HRTB callback.
pub(in crate::replicated_decoder::program) struct SourceScope<'a> {
    catalog: &'a SourceCatalog,
    registry: &'a SourceRegistry<'a>,
    manifest: &'a Manifest,
    original: &'a OriginalSourcePlan,
    charge: &'a OwnedExternalCharge,
    instance: source_registry::Instance,
}

impl<'a> SourceScope<'a> {
    pub(in crate::replicated_decoder::program) fn check_context(
        &self,
        context: &request_origin::compact_storage::EmissionContext<'_>,
    ) -> Result<(), Error> {
        context.check_open()?;
        if !std::ptr::eq(context.manifest(), self.manifest)
            || !std::ptr::eq(context.registry(), self.registry)
            || !context.plan().same_origin(self.original)
            || !self.charge.is_for(context.plan().budget())
        {
            return Err(Error::Binding);
        }
        Ok(())
    }
    pub(super) fn new(
        catalog: &'a SourceCatalog,
        registry: &'a SourceRegistry<'a>,
        manifest: &'a Manifest,
        original: &'a OriginalSourcePlan,
        charge: &'a OwnedExternalCharge,
    ) -> Result<Self, Error> {
        maps(registry, manifest)?;
        Ok(Self {
            catalog,
            registry,
            manifest,
            original,
            charge,
            instance: registry.instance(),
        })
    }

    pub(in crate::replicated_decoder::program) fn tensor(
        &self,
        source: &SourceTensor,
    ) -> Result<TensorHandle, Error> {
        self.catalog.issue_tensor(
            self.registry,
            self.manifest,
            self.instance,
            self.original,
            self.charge,
            source,
        )
    }

    pub(in crate::replicated_decoder::program) fn norm(
        &self,
        source: &SourceTensor,
        norm: &crate::replicated_decoder::BlockNorms,
    ) -> Result<NormHandle, Error> {
        self.catalog.issue_norm(self, source, norm)
    }
}

// Parent obtains this only through quote, reserves bytes on the ORIGINAL root,
// then retains that guard as SourceCustody::catalog_charge, after all payloads.
pub(super) struct CatalogQuote {
    pub(super) bytes: usize,
    instance: source_registry::Instance,
    count: usize,
    work: u64,
}

#[allow(dead_code)]
struct Controls<'a, R> {
    quote: Result<CatalogQuote, Error>,
    catalog: Result<SourceCatalog, Error>,
    string: Result<String, Error>,
    sources: btree_map::Iter<'a, String, SourceTensor>,
    norms: btree_map::Iter<'a, String, crate::replicated_decoder::BlockNorms>,
    tensor: Result<&'a SourceTensor, Error>,
    norm: Result<&'a crate::replicated_decoder::BlockNorms, Error>,
    tensor_handle: Result<TensorHandle, Error>,
    norm_handle: Result<NormHandle, Error>,
    scope: Option<SourceScope<'a>>,
    custody_move: std::marker::PhantomData<R>,
    counters: [usize; 12],
}

fn pay(
    original: &OriginalSourcePlan,
    guard: &OwnedExternalCharge,
    work: usize,
) -> Result<(), Error> {
    original
        .prepay_compiler_metadata(guard, u64::try_from(work).map_err(|_| Error::Bounds)?)
        .map_err(|_| Error::Bounds)
}

fn same_maps(
    sources: &std::collections::BTreeMap<String, SourceTensor>,
    norms: &crate::replicated_decoder::SourceNorms,
    manifest: &Manifest,
) -> Result<(), Error> {
    // Map identity, including EMPTY maps, is checked before value traversal.
    if !std::ptr::eq(sources, &manifest.tensors) || !std::ptr::eq(norms, &manifest.norms) {
        return Err(Error::Binding);
    }
    Ok(())
}
fn maps(registry: &SourceRegistry<'_>, manifest: &Manifest) -> Result<(), Error> {
    same_maps(registry.source_records(), registry.source_norms(), manifest)
}

impl SourceCatalog {
    // Parent rechecks its moved, immutable owned backing without reminting a
    // registry or accepting a caller-selected Manifest as source authority.
    pub(super) fn check_owned(
        &self,
        manifest: &Manifest,
        original: &OriginalSourcePlan,
        charge: &OwnedExternalCharge,
    ) -> Result<(), Error> {
        pay(
            original,
            charge,
            add(self.entries.len(), manifest.norms.len())?
                .checked_mul(8)
                .ok_or(Error::Bounds)?,
        )?;
        if self.entries.len() != manifest.tensors.len() {
            return Err(Error::Binding);
        }
        let bytes = self
            .entries
            .iter()
            .try_fold(0usize, |n, e| add(n, add(e.key.len(), 1)?))?;
        pay(
            original,
            charge,
            add(self.entries.len(), manifest.norms.len())?
                .checked_add(1)
                .and_then(|n| n.checked_mul(bytes.checked_add(1)?))
                .ok_or(Error::Bounds)?,
        )?;
        for (entry, key) in self.entries.iter().zip(manifest.tensors.keys()) {
            if &entry.key != key || entry.has_norm != manifest.norms.contains_key(key) {
                return Err(Error::Binding);
            }
        }
        if manifest
            .norms
            .keys()
            .any(|key| !manifest.tensors.contains_key(key))
        {
            return Err(Error::Binding);
        }
        Ok(())
    }
    pub(super) fn quote<R>(
        registry: &SourceRegistry<'_>,
        manifest: &Manifest,
        original: &OriginalSourcePlan,
        controller: &OwnedExternalCharge,
    ) -> Result<CatalogQuote, Error> {
        maps(registry, manifest)?;
        let count = registry.source_records().len();
        let norms = registry.source_norms().len();
        // Pay the allocation-free capacity census before visiting keys.
        pay(
            original,
            controller,
            add(count, norms)?.checked_mul(8).ok_or(Error::Bounds)?,
        )?;
        let mut strings = 0;
        let mut key_bytes = 0;
        for key in registry.source_records().keys() {
            strings = add(strings, source_registry::array_bytes::<u8>(key.len())?)?;
            key_bytes = add(key_bytes, add(key.len(), 1)?)?;
        }
        for key in registry.source_norms().keys() {
            key_bytes = add(key_bytes, add(key.len(), 1)?)?;
        }
        // Complete finite bound for construction key comparisons and copies.
        let work = add(count, 1)?
            .checked_mul(add(count, norms)?)
            .and_then(|n| n.checked_mul(key_bytes.checked_add(1)?))
            .and_then(|n| n.checked_add(key_bytes))
            .ok_or(Error::Bounds)?;
        let bytes = add(
            add(source_registry::array_bytes::<SourceKey>(count)?, strings)?,
            add(
                size_of::<SourceCustody<R>>()
                    .checked_mul(2)
                    .ok_or(Error::Bounds)?,
                size_of::<Controls<'_, R>>()
                    .checked_mul(2)
                    .ok_or(Error::Bounds)?,
            )?,
        )?;
        if bytes > MAX_PROGRAM_BYTES {
            return Err(Error::Bounds);
        }
        Ok(CatalogQuote {
            bytes,
            instance: registry.instance(),
            count,
            work: u64::try_from(work).map_err(|_| Error::Bounds)?,
        })
    }

    pub(super) fn build(
        registry: &SourceRegistry<'_>,
        manifest: &Manifest,
        original: &OriginalSourcePlan,
        charge: &OwnedExternalCharge,
        quote: CatalogQuote,
    ) -> Result<Self, Error> {
        maps(registry, manifest)?;
        if registry.instance() != quote.instance
            || registry.source_records().len() != quote.count
            || charge.payload_bytes() < quote.bytes
        {
            return Err(Error::Binding);
        }
        original
            .prepay_compiler_metadata(charge, quote.work)
            .map_err(|_| Error::Bounds)?;
        // Reject orphan norms before any catalog allocation.
        if registry
            .source_norms()
            .keys()
            .any(|key| !registry.source_records().contains_key(key))
        {
            return Err(Error::Binding);
        }
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(quote.count)
            .map_err(|_| Error::Allocation)?;
        if entries.capacity() != quote.count {
            return Err(Error::Bounds);
        }
        // ALL tensors, including public_q18 gamma. Do not change records() or
        // its independent file/public-input numbering elsewhere.
        for key in registry.source_records().keys() {
            let mut owned = String::new();
            owned
                .try_reserve_exact(key.len())
                .map_err(|_| Error::Allocation)?;
            if owned.capacity() != key.len() {
                return Err(Error::Bounds);
            }
            owned.push_str(key);
            entries.push(SourceKey {
                key: owned,
                has_norm: registry.source_norms().contains_key(key),
            });
        }
        Ok(Self { entries })
    }

    // Parent-only: expected is captured at the ORIGINAL registry construction,
    // not accepted from a recipe/caller as a detached authority token. All work
    // below is prepaid via that same original plan/controller before lookups.
    pub(super) fn issue_tensor(
        &self,
        registry: &SourceRegistry<'_>,
        manifest: &Manifest,
        expected: source_registry::Instance,
        original: &OriginalSourcePlan,
        controller: &OwnedExternalCharge,
        source: &SourceTensor,
    ) -> Result<TensorHandle, Error> {
        maps(registry, manifest)?;
        if registry.instance() != expected {
            return Err(Error::Binding);
        }
        pay(
            original,
            controller,
            self.entries.len().checked_mul(4).ok_or(Error::Bounds)?,
        )?;
        let bytes = self
            .entries
            .iter()
            .try_fold(0, |n, e| add(n, add(e.key.len(), 1)?))?;
        pay(
            original,
            controller,
            add(self.entries.len(), 1)?
                .checked_mul(add(bytes, 1)?)
                .ok_or(Error::Bounds)?,
        )?;
        for (index, entry) in self.entries.iter().enumerate() {
            let actual = registry
                .source_records()
                .get(&entry.key)
                .ok_or(Error::Binding)?;
            if std::ptr::eq(actual, source) {
                return Ok(TensorHandle {
                    instance: expected,
                    index,
                });
            }
        }
        Err(Error::Binding)
    }

    pub(super) fn issue_norm(
        &self,
        scope: &SourceScope<'_>,
        source: &SourceTensor,
        norm: &crate::replicated_decoder::BlockNorms,
    ) -> Result<NormHandle, Error> {
        if !std::ptr::eq(self, scope.catalog) {
            return Err(Error::Binding);
        }
        let SourceScope {
            registry,
            manifest,
            instance: expected,
            original,
            charge: controller,
            ..
        } = *scope;
        let tensor =
            self.issue_tensor(registry, manifest, expected, original, controller, source)?;
        let entry = self.entries.get(tensor.index).ok_or(Error::Binding)?;
        pay(
            original,
            controller,
            add(registry.source_norms().len(), 1)?
                .checked_mul(add(entry.key.len(), 1)?)
                .ok_or(Error::Bounds)?,
        )?;
        if !entry.has_norm
            || !registry
                .source_norms()
                .get(&entry.key)
                .is_some_and(|actual| std::ptr::eq(actual, norm))
        {
            return Err(Error::Binding);
        }
        Ok(NormHandle {
            instance: expected,
            tensor_index: tensor.index,
        })
    }
}

impl<R> SourceCustody<R> {
    pub(super) fn prepay_retained_resolution(&self, queries: usize) -> Result<(), Error> {
        // Length inspection is prepaid before visiting entries; this is only a
        // compiler traversal allowance, never source or execution authority.
        pay(
            &self.original,
            &self.catalog_charge,
            add(self.catalog.entries.len(), 1)?,
        )?;
        let key_bytes = self
            .catalog
            .entries
            .iter()
            .try_fold(1usize, |n, e| add(n, add(e.key.len(), 1)?))?;
        let entries = add(self.catalog.entries.len(), 1)?;
        let visits = queries
            .checked_mul(key_bytes)
            .and_then(|n| n.checked_mul(entries))
            .ok_or(Error::Bounds)?;
        pay(&self.original, &self.catalog_charge, visits)
    }
    /// Only a checked import loan supplies this retained original key. Resolution
    /// stays inside the whole custody and uses its ORIGINAL catalog Instance.
    pub(in crate::replicated_decoder::program) fn resolve_file_import(
        &self,
        key: &str,
    ) -> Result<&SourceTensor, Error> {
        self.check_ingress(self.instance, self.binding)?;
        pay(
            &self.original,
            &self.catalog_charge,
            add(self.catalog.entries.len(), 1)?
                .checked_mul(add(key.len(), 1)?)
                .ok_or(Error::Bounds)?,
        )?;
        let index = self
            .catalog
            .entries
            .binary_search_by(|entry| entry.key.as_str().cmp(key))
            .map_err(|_| Error::Binding)?;
        let source = self.resolve_tensor(&TensorHandle {
            instance: self.instance,
            index,
        })?;
        if source.public_q18.is_some() {
            return Err(Error::Binding);
        }
        Ok(source)
    }
    pub(in crate::replicated_decoder::program) fn resolve_tensor(
        &self,
        handle: &TensorHandle,
    ) -> Result<&SourceTensor, Error> {
        if handle.instance != self.instance {
            return Err(Error::Binding);
        }
        let entry = self
            .catalog
            .entries
            .get(handle.index)
            .ok_or(Error::Binding)?;
        pay(
            &self.original,
            &self.catalog_charge,
            add(self.catalog.entries.len(), 1)?
                .checked_mul(add(entry.key.len(), 1)?)
                .ok_or(Error::Bounds)?,
        )?;
        self.manifest
            .payload()
            .tensors
            .get(&entry.key)
            .ok_or(Error::Binding)
    }

    pub(in crate::replicated_decoder::program) fn resolve_norm(
        &self,
        handle: &NormHandle,
    ) -> Result<&crate::replicated_decoder::BlockNorms, Error> {
        if handle.instance != self.instance {
            return Err(Error::Binding);
        }
        let entry = self
            .catalog
            .entries
            .get(handle.tensor_index)
            .ok_or(Error::Binding)?;
        if !entry.has_norm {
            return Err(Error::Binding);
        }
        pay(
            &self.original,
            &self.catalog_charge,
            add(self.catalog.entries.len(), 1)?
                .checked_mul(add(entry.key.len(), 1)?)
                .ok_or(Error::Bounds)?,
        )?;
        self.manifest
            .payload()
            .norms
            .get(&entry.key)
            .ok_or(Error::Binding)
    }
}

// Runtime lenses belong to the ACTUAL canonical-finalized root, never a generic
// OPEN-or-COMPLETE custody. Compilation already admitted the catalog traversal;
// immutable runtime resolution must not spend/reset compiler metadata quota.
impl super::sealed::FinalizedRoot {
    pub(in crate::replicated_decoder::program) fn tensor(
        &self,
        handle: &TensorHandle,
    ) -> Result<&SourceTensor, Error> {
        let custody = self.custody();
        if handle.instance != custody.instance {
            return Err(Error::Binding);
        }
        let entry = custody
            .catalog
            .entries
            .get(handle.index)
            .ok_or(Error::Binding)?;
        custody
            .manifest
            .payload()
            .tensors
            .get(&entry.key)
            .ok_or(Error::Binding)
    }
    pub(in crate::replicated_decoder::program) fn norm(
        &self,
        handle: &NormHandle,
    ) -> Result<&crate::replicated_decoder::BlockNorms, Error> {
        let custody = self.custody();
        if handle.instance != custody.instance {
            return Err(Error::Binding);
        }
        let entry = custody
            .catalog
            .entries
            .get(handle.tensor_index)
            .ok_or(Error::Binding)?;
        if !entry.has_norm {
            return Err(Error::Binding);
        }
        custody
            .manifest
            .payload()
            .norms
            .get(&entry.key)
            .ok_or(Error::Binding)
    }
}

// Test definitions, NOT a substitute fixture issuer. E's existing ingress
// fixture must supply two genuine prepared imports with equal source contents,
// and mount the scoped fifth callback argument described in the glue proposal.
// No into_test_payload, raw Arc or synthetic success constructor is used.
#[cfg(test)]
pub(in crate::replicated_decoder::program) fn assert_genuine_consuming_handles(
    first: SourceRequestImport,
    second: SourceRequestImport,
    gamma_key: &str,
    file_key: &str,
    norm_key: &str,
) {
    let consume = |import: SourceRequestImport| {
        import
            .consume_compact(|_created, _controller, registry, _catalog, scope| {
                let gamma = registry
                    .source_records()
                    .get(gamma_key)
                    .ok_or(Error::Binding)?;
                let file = registry
                    .source_records()
                    .get(file_key)
                    .ok_or(Error::Binding)?;
                let norm_source = registry
                    .source_records()
                    .get(norm_key)
                    .ok_or(Error::Binding)?;
                let norm = registry
                    .source_norms()
                    .get(norm_key)
                    .ok_or(Error::Binding)?;
                assert!(gamma.public_q18.is_some());
                assert!(file.public_q18.is_none());
                assert_ne!(gamma_key, norm_key);
                assert!(scope.norm(gamma, norm).is_err());
                // Same contents, wrong pointer: negative only, never adoption.
                let copied_gamma = gamma.clone();
                assert!(scope.tensor(&copied_gamma).is_err());
                let copied = file.clone();
                assert!(scope.tensor(&copied).is_err());
                let copied_norm = norm.clone();
                assert_eq!(&copied_norm, norm);
                assert!(scope.norm(norm_source, &copied_norm).is_err());
                Ok((
                    scope.tensor(gamma)?,
                    scope.tensor(file)?,
                    scope.norm(norm_source, norm)?,
                ))
            })
            .expect("genuine consuming funnel")
    };
    let first = consume(first);
    let second = consume(second);
    let address = first.resolve_tensor(&first.produced.0).unwrap() as *const SourceTensor;
    let moved = std::hint::black_box(first);
    assert_eq!(
        address,
        moved.resolve_tensor(&moved.produced.0).unwrap() as *const SourceTensor
    );
    assert!(moved
        .resolve_tensor(&moved.produced.1)
        .unwrap()
        .public_q18
        .is_none());
    assert!(moved.resolve_norm(&moved.produced.2).is_ok());
    assert!(moved.resolve_tensor(&second.produced.0).is_err());
    assert!(moved.resolve_norm(&second.produced.2).is_err());
    assert!(second.resolve_tensor(&moved.produced.1).is_err());
}

#[cfg(test)]
pub(in crate::replicated_decoder::program) fn assert_orphan_norm_funnel_refused(
    import: SourceRequestImport,
) {
    // E's genuine fixture must differ from an admitted control ONLY by an
    // unmatched norm key. If ingress already refuses before producing this
    // import, assert that original refusal in the wrapper instead. Never mint
    // a registry to reach the defensive catalog branch artificially.
    let callback_entered = std::cell::Cell::new(false);
    let result = import.consume_compact(|_created, _controller, _registry, _catalog, _scope| {
        callback_entered.set(true);
        Ok(())
    });
    assert!(matches!(result, Err(Error::Binding)));
    assert!(!callback_entered.get());
    // This proves rejection before callback, NOT which earlier validator fired.
}

#[cfg(test)]
pub(in crate::replicated_decoder::program) fn assert_equal_content_foreign_map(
    registry: &SourceRegistry<'_>,
    foreign: &Manifest,
) {
    assert_eq!(
        registry.binding(),
        crate::replicated_decoder::program::source_binding_with_norms(
            &foreign.tensors,
            &foreign.norms
        )
    );
    assert!(maps(registry, foreign).is_err());
}

/// Predicate regression only: empty maps cannot form an admitted decoder source
/// fixture. No SourceRegistry, handle, or custody is fabricated for this check.
#[cfg(test)]
pub(in crate::replicated_decoder::program) fn assert_empty_map_predicate(mut first: Manifest) {
    first.tensors.clear();
    first.norms.clear();
    let sources = std::collections::BTreeMap::new();
    let norms = crate::replicated_decoder::SourceNorms::new();
    assert!(same_maps(&first.tensors, &first.norms, &first).is_ok());
    assert!(matches!(
        same_maps(&sources, &first.norms, &first),
        Err(Error::Binding)
    ));
    assert!(matches!(
        same_maps(&first.tensors, &norms, &first),
        Err(Error::Binding)
    ));
}

#[cfg(test)]
pub(in crate::replicated_decoder::program) fn assert_payload_before_guard(
    import: SourceRequestImport,
    root: &OwnedBudget,
    outside_baseline: usize,
) {
    // E must supply baseline BEFORE the prepared import's owned charges, with
    // its importer-guard lifetime fix applied. This is not a fabricated owner.
    // Probe proves only SOME charge remains above baseline at produced-R drop;
    // it does not prove every payload/guard's destruction order. E owns that
    // stronger lifetime regression and importer failure cleanup qualification.
    struct Probe<'a> {
        root: &'a OwnedBudget,
        baseline: usize,
        dropped: &'a std::cell::Cell<bool>,
    }
    impl Drop for Probe<'_> {
        fn drop(&mut self) {
            assert!(self.root.budget().used_bytes() > self.baseline);
            self.dropped.set(true);
        }
    }
    let dropped = std::cell::Cell::new(false);
    let custody = import
        .consume_compact(|_created, _controller, _registry, _catalog, _scope| {
            Ok(Probe {
                root,
                baseline: outside_baseline,
                dropped: &dropped,
            })
        })
        .expect("genuine consuming funnel");
    assert!(!dropped.get());
    drop(custody);
    assert!(dropped.get());
    assert_eq!(root.budget().used_bytes(), outside_baseline);
}
