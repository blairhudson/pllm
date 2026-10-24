//! Consuming lexical conversion boundary. This owns authentic ingress backing,
//! not an executable schedule. Only the complete storage finalizer may publish.
use super::*;

// Nonfallible assembly immediately after Pending extraction. Both ordinary
// failure and unwind retire importer payload BEFORE its ORIGINAL reservation.
// Private staging only until the complete owning request publisher is wired.
#[cfg_attr(not(test), allow(dead_code))]
struct ImportStage {
    importer: Option<PublicSourceImport>,
    charge: Option<OwnedExternalCharge>,
}
impl Drop for ImportStage {
    fn drop(&mut self) {
        #[cfg(test)]
        let had_importer = self.importer.is_some();
        drop(self.importer.take());
        #[cfg(test)]
        if had_importer {
            let charge = self
                .charge
                .as_ref()
                .expect("importer retains original charge");
            IMPORT_DROP.with(|record| {
                record.set(Some((
                    self.importer.is_none(),
                    charge.budget().used_bytes(),
                    charge.charged_bytes(),
                )))
            });
        }
        // charge field drops ONLY after actual importer destruction returned.
    }
}
#[cfg(test)]
thread_local! {
    static IMPORT_DROP:std::cell::Cell<Option<(bool,usize,usize)>>=const{std::cell::Cell::new(None)};
    static PANIC_BEFORE_BOX:std::cell::Cell<bool>=const{std::cell::Cell::new(false)};
}
#[cfg(test)]
pub(in crate::replicated_decoder::program) fn test_drop_record() -> Option<(bool, usize, usize)> {
    IMPORT_DROP.with(|record| record.take())
}
#[cfg(test)]
pub(in crate::replicated_decoder::program) fn test_panic_before_box() {
    PANIC_BEFORE_BOX.with(|flag| flag.set(true));
}
#[cfg(test)]
pub(in crate::replicated_decoder::program) fn test_hold_remaining(
    entry: &SourceRequestImport,
) -> OwnedExternalCharge {
    let root = &entry.0.as_ref().unwrap().root;
    let mut guard = root.charge_external(0).unwrap();
    let remaining = MAX_PROGRAM_BYTES - root.budget().used_bytes();
    guard.resize_payload(remaining).unwrap();
    guard
}

/// Catalog positions are issued only while borrowing the actual authenticated
/// registry. They are not source authority without this whole private owner.
// Replaced by the reviewed all-tensor handle catalog in the next separate unit.
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::replicated_decoder::program) struct SourceRecord {
    source: usize,
    file: Option<usize>,
    norm: Option<usize>,
}

/// Payloads precede reservations in field drop order. No borrowed registry,
/// FrozenRegion or self-referential pointer is stored beside its referents.
// Ingress custody, not executable Storage; no production route activates it yet.
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::replicated_decoder::program) struct SourceCustody<R> {
    produced: R,
    original: Arc<OriginalSourcePlan>,
    public: PublicSnapshotOwner,
    manifest: Box<OwnedCharged<Manifest>>,
    raw: OwnedCharged<json::ImportedValue>,
    records: Vec<SourceRecord>,
    instance: source_registry::Instance,
    binding: [u8; 32],
    registry_charge: OwnedExternalCharge,
    manifest_box_charge: OwnedExternalCharge,
    catalog_charge: OwnedExternalCharge,
    root: OwnedBudget,
}

impl<R> SourceCustody<R> {
    // Called by staged ingress, with genuine source lifecycle coverage in tests.
    #[cfg_attr(not(test), allow(dead_code))]
    fn check_ingress(
        &self,
        instance: source_registry::Instance,
        binding: [u8; 32],
    ) -> Result<(), Error> {
        let budget = self.root.budget();
        if self.instance != instance
            || self.binding != binding
            || !std::ptr::eq(self.original.budget(), budget)
            || !std::ptr::eq(self.manifest.budget(), budget)
            || !std::ptr::eq(self.raw.budget(), budget)
            || !self.public.charge.guard.is_for(budget)
            || !self.registry_charge.is_for(budget)
            || !self.manifest_box_charge.is_for(budget)
            || !self.catalog_charge.is_for(budget)
            || self.records.len() != self.manifest.payload().tensors.len()
        {
            return Err(Error::Binding);
        }
        let mut file = 0;
        for (i, ((key, source), record)) in self
            .manifest
            .payload()
            .tensors
            .iter()
            .zip(&self.records)
            .enumerate()
        {
            if record.source != i {
                return Err(Error::Binding);
            }
            let expected = if source.public_q18.is_none() {
                let owner = file;
                file += 1;
                Some(owner)
            } else {
                None
            };
            if record.file != expected {
                return Err(Error::Binding);
            }
            match record.norm {
                Some(index) if self.manifest.payload().norms.keys().nth(index) == Some(key) => {}
                None if !self.manifest.payload().norms.contains_key(key) => {}
                _ => return Err(Error::Binding),
            }
        }
        Ok(())
    }
    #[cfg(test)]
    pub(super) fn into_test_payload(self) -> R {
        self.produced
    }
}

impl SourceRequestImport {
    /// One consuming entry. Conversion sees the SAME genuine registry and all
    /// actual source/norm associations. HRTB prevents any lexical borrow escaping
    /// in R; conversion must produce owning records/handles. Then those borrows
    /// end and the SAME whole OwnedCharged wrappers move, without refund/rebuy.
    // Deliberately inactive until borrow-free recipes/history can publish Storage.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(in crate::replicated_decoder::program) fn consume_compact<R>(
        mut self,
        emit: impl for<'s> FnOnce(
            CreatedInputs<'s>,
            OwnedExternalCharge,
            &'s SourceRegistry<'s>,
            &'s [SourceRecord],
        ) -> Result<R, Error>,
    ) -> Result<SourceCustody<R>, Error> {
        let Pending {
            importer,
            original,
            manifest,
            raw,
            limits: _,
            registry: mut registry_guard,
            imports: import_guard,
            mut controller,
            root,
        } = self.0.take().ok_or(Error::Retired)?;
        let mut importing = ImportStage {
            importer: Some(importer),
            charge: Some(import_guard),
        };
        #[cfg(test)]
        PANIC_BEFORE_BOX.with(|flag| {
            if flag.replace(false) {
                panic!("ingress unwind before box reservation")
            }
        });
        // Stabilize the SAME whole wrapper BEFORE registry/origin borrows. Its
        // source/norm map addresses and BTree entry allocations then survive the
        // outer custody move unchanged. No field extraction or raw-Arc adoption.
        let boxed_bytes = add(
            size_of::<OwnedCharged<Manifest>>(),
            2 * size_of::<Box<OwnedCharged<Manifest>>>(),
        )?;
        if add(root.budget().used_bytes(), boxed_bytes)? > MAX_PROGRAM_BYTES {
            return Err(Error::Bounds);
        }
        let manifest_box_charge = root
            .charge_external(boxed_bytes)
            .map_err(|_| Error::Bounds)?;
        let manifest = Box::new(manifest);
        let outer = PublicSnapshotOwner::new(
            importing
                .importer
                .take()
                .ok_or(Error::Retired)?
                .finish()
                .map_err(|_| Error::Binding)?,
            &root,
        )?;
        if !outer.charge.guard.is_for(root.budget()) {
            return Err(Error::Binding);
        }
        let context = add(
            request_origin::compact_storage::context_bytes(),
            OriginalSourcePlan::compiler_metadata_control_bytes().map_err(|_| Error::Bounds)?,
        )?;
        if controller.payload_bytes() < context {
            controller
                .resize_payload(context)
                .map_err(|_| Error::Bounds)?;
        }
        let (produced, retained_original, records, instance, binding, catalog_charge) = {
            let public = &outer.public;
            let owners = SourceConstructionOwners {
                manifest: manifest.payload(),
                incoming: None,
                public: Some(public),
                retained_baseline_bytes: root
                    .budget()
                    .used_bytes()
                    .checked_sub(registry_guard.payload_bytes())
                    .ok_or(Error::Bounds)?,
            };
            let q =
                SourceRegistry::original_construction_quote(&original, &owners, MAX_PROGRAM_BYTES)?;
            if registry_guard.payload_bytes() < q.new_owned_peak_bytes {
                registry_guard
                    .resize_payload(q.new_owned_peak_bytes)
                    .map_err(|_| Error::Bounds)?;
            }
            let registry = SourceRegistry::from_original_checked(
                &original,
                &owners,
                public,
                &registry_guard,
                MAX_PROGRAM_BYTES,
            )?;
            // Includes empty maps: identity is the actual Manifest's maps, never
            // equality of their contents or a detached seal paired by a caller.
            if !std::ptr::eq(registry.source_records(), &manifest.payload().tensors)
                || !std::ptr::eq(registry.source_norms(), &manifest.payload().norms)
            {
                return Err(Error::Binding);
            }
            public.check_registry(&registry)?;
            let count = registry.source_records().len();
            let catalog_bytes = add(
                2 * size_of::<SourceCustody<R>>(),
                count
                    .checked_mul(size_of::<SourceRecord>())
                    .ok_or(Error::Bounds)?,
            )?;
            let phase = add(root.budget().used_bytes(), catalog_bytes)?;
            if phase > MAX_PROGRAM_BYTES {
                return Err(Error::Bounds);
            }
            let catalog_charge = root
                .charge_external(catalog_bytes)
                .map_err(|_| Error::Bounds)?;
            original
                .prepay_compiler_metadata(
                    &controller,
                    u64::try_from(
                        add(count, registry.source_norms().len())?
                            .checked_mul(8)
                            .ok_or(Error::Bounds)?,
                    )
                    .map_err(|_| Error::Bounds)?,
                )
                .map_err(|_| Error::Bounds)?;
            let key_bytes = manifest
                .payload()
                .tensors
                .keys()
                .chain(manifest.payload().norms.keys())
                .try_fold(0usize, |sum, key| add(sum, add(key.len(), 1)?))?;
            // Two bounded map lookups per source plus the ordered norm merge;
            // charge a complete finite key-comparison envelope BEFORE lookups.
            let comparisons = add(count, 1)?
                .checked_mul(add(count, registry.source_norms().len())?)
                .and_then(|n| n.checked_mul(add(key_bytes, 1).ok()?))
                .ok_or(Error::Bounds)?;
            original
                .prepay_compiler_metadata(
                    &controller,
                    u64::try_from(comparisons).map_err(|_| Error::Bounds)?,
                )
                .map_err(|_| Error::Bounds)?;
            let mut records = Vec::new();
            records
                .try_reserve_exact(count)
                .map_err(|_| Error::Allocation)?;
            if records.capacity() != count {
                return Err(Error::Bounds);
            }
            let mut norms = manifest.payload().norms.iter().enumerate().peekable();
            let mut files = registry.records();
            for (index, (key, source)) in registry.source_records().iter().enumerate() {
                let norm = registry.source_norms().get(key);
                let file = if source.public_q18.is_none() {
                    let (owner, actual_key, actual_source, actual_norm) =
                        files.next().ok_or(Error::Binding)?;
                    if actual_key != key
                        || !std::ptr::eq(actual_source, source)
                        || match (actual_norm, norm) {
                            (None, None) => false,
                            (Some(a), Some(b)) => !std::ptr::eq(a, b),
                            _ => true,
                        }
                    {
                        return Err(Error::Binding);
                    }
                    Some(owner)
                } else {
                    None
                };
                if index != records.len()
                    || !std::ptr::eq(
                        source,
                        manifest.payload().tensors.get(key).ok_or(Error::Binding)?,
                    )
                {
                    return Err(Error::Binding);
                }
                let association = match norms.peek() {
                    Some((norm_index, (norm_key, value))) if *norm_key == key => {
                        if !norm.is_some_and(|actual| std::ptr::eq(actual, *value)) {
                            return Err(Error::Binding);
                        }
                        let ordinal = *norm_index;
                        norms.next();
                        Some(ordinal)
                    }
                    Some((_, (norm_key, _))) if *norm_key < key => return Err(Error::Binding),
                    _ => {
                        if norm.is_some() {
                            return Err(Error::Binding);
                        }
                        None
                    }
                };
                records.push(SourceRecord {
                    source: index,
                    file,
                    norm: association,
                });
            }
            if norms.next().is_some() || files.next().is_some() {
                return Err(Error::Binding);
            }
            let import_guard = importing.charge.as_mut().ok_or(Error::Retired)?;
            let q = quote_imports(
                &original,
                manifest.payload(),
                &registry,
                public,
                import_guard,
            )?;
            if import_guard.payload_bytes() < q.payload_bytes {
                import_guard
                    .resize_payload(q.payload_bytes)
                    .map_err(|_| Error::Bounds)?;
            }
            let created = create_inputs(
                original,
                manifest.payload(),
                &registry,
                Arc::clone(public),
                importing.charge.take().ok_or(Error::Retired)?,
            )?;
            let retained_original = Arc::clone(created.permit.retained_original()?);
            let produced = emit(created, controller, &registry, &records)?;
            (
                produced,
                retained_original,
                records,
                registry.instance(),
                registry.binding(),
                catalog_charge,
            )
        }; // all Manifest/registry/catalog lenses end BEFORE their owners move.
        let custody = SourceCustody {
            produced,
            original: retained_original,
            public: outer,
            manifest,
            raw,
            records,
            instance,
            binding,
            registry_charge: registry_guard,
            manifest_box_charge,
            catalog_charge,
            root,
        };
        custody.check_ingress(instance, binding)?;
        Ok(custody)
    }
}
