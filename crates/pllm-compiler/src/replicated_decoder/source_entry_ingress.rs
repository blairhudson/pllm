//! Precollection client ownership; no fresh counter at the collected-header join.
use super::*;
use crate::replicated_decoder::sdk::{SourceRequestProgram, SourceStorage};

/// Private common move capsule. Only this module and original worker ingress
/// construct it; no constructor accepts a caller Budget or source/registry seal.
pub(super) struct OwnedSourceIngress {
    pub(super) directory: Option<Arc<snapshot_directory::OwnedSnapshotDirectory>>,
    pub(super) original: OriginalSourcePlan,
    pub(super) raw: SourceIngressRaw,
    pub(super) limits: original_source::Limits,
    pub(super) storage: SourceStorage,
    pub(super) controller: OwnedExternalCharge,
    pub(super) root: OwnedBudget,
}

/// Born BEFORE the collector's first record or coefficient. Manifest is an
/// unpublished descriptor; its program stays zero until real compilation.
pub(in crate::replicated_decoder) struct SourceHeaderEntry {
    manifest: Manifest,
    directory: Arc<snapshot_directory::OwnedSnapshotDirectory>,
    original: OriginalSourcePlan,
    experiment: OwnedCharged<json::ImportedValue>,
    limits: original_source::Limits,
    manifest_charge: OwnedExternalCharge,
    collection_charge: OwnedExternalCharge,
    controller: OwnedExternalCharge,
    root: OwnedBudget,
    failed: bool,
}

/// Actual payload always dies BEFORE its original covering lease, including
/// any header/import/compilation error. R owns T's scanner/record implementation.
pub(in crate::replicated_decoder) struct SourceCollector<T> {
    payload: Option<T>,
    header: SourceHeaderEntry,
}
pub(in crate::replicated_decoder) struct CompiledSource<T> {
    payload: Option<T>,
    program: SourceRequestProgram,
    charge: OwnedExternalCharge,
    failed: bool,
}
struct PayloadLease<T> {
    payload: T,
    charge: OwnedExternalCharge,
}
struct CollectorAttempt<'a, T> {
    owner: &'a mut SourceCollector<T>,
    committed: bool,
}
impl<T> Drop for CollectorAttempt<'_, T> {
    fn drop(&mut self) {
        if !self.committed {
            drop(self.owner.payload.take());
            self.owner.header.failed = true;
            self.owner.header.manifest.tensors.clear();
            self.owner.header.manifest.norms.clear();
        }
    }
}
struct CompiledAttempt<'a, T> {
    owner: &'a mut CompiledSource<T>,
    committed: bool,
}
impl<T> Drop for CompiledAttempt<'_, T> {
    fn drop(&mut self) {
        if !self.committed {
            drop(self.owner.payload.take());
            self.owner.failed = true;
        }
    }
}
/// Only collected records are mutable. Admission, arithmetic, session and source
/// header identities cannot be changed by the collector closure.
pub(in crate::replicated_decoder) struct SourceRecords<'a> {
    pub(in crate::replicated_decoder) tensors: &'a mut BTreeMap<String, SourceTensor>,
    pub(in crate::replicated_decoder) norms: &'a mut crate::replicated_decoder::SourceNorms,
}

fn plus(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Bounds)
}
pub(in crate::replicated_decoder::program) fn resize(
    original: &OriginalSourcePlan,
    guard: &mut OwnedExternalCharge,
    bytes: usize,
    visits: u64,
) -> Result<(), Error> {
    if !guard.is_for(original.budget()) {
        return Err(Error::Binding);
    }
    let target = guard
        .payload_bytes()
        .max(bytes)
        .max(OriginalSourcePlan::compiler_metadata_control_bytes().map_err(|_| Error::Bounds)?);
    let total = plus(
        original
            .budget()
            .used_bytes()
            .checked_sub(guard.payload_bytes())
            .ok_or(Error::Binding)?,
        target,
    )?;
    if total > MAX_PROGRAM_BYTES {
        return Err(Error::Bounds);
    }
    guard.resize_payload(target).map_err(|_| Error::Bounds)?;
    original
        .prepay_compiler_metadata(guard, visits)
        .map_err(|_| Error::Bounds)?;
    original
        .budget()
        .admit_peak(total)
        .map_err(|_| Error::Bounds)
}

impl SourceHeaderEntry {
    /// Caller is the admitted CLIENT Experiment path, not a worker request body.
    /// Resource policy is independently rechecked against actual admitted C9
    /// owners; checkpoint/source-pin authentication remains R's original scanner.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::replicated_decoder) fn begin(
        experiment: &[u8],
        config: &[u8],
        rows: usize,
        outputs: usize,
        session: [u8; 32],
        source: [u8; 32],
        composition: [u8; 32],
    ) -> Result<Self, Error> {
        if experiment.is_empty()
            || experiment.len() > 32 << 10
            || config.is_empty()
            || config.len() > source_policy::CONFIG_BYTES
            || session == [0; 32]
            || source == [0; 32]
            || composition == [0; 32]
        {
            return Err(Error::Binding);
        }
        let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).map_err(|_| Error::Bounds)?;
        let controller = root
            .charge_external(plus(
                pending_controller_bytes(),
                plus(server_controller_bytes(), 3 * size_of::<Self>())?,
            )?)
            .map_err(|_| Error::Bounds)?;
        let experiment = import_json(&root, experiment)?;
        let config = import_json(&root, config)?;
        let workload = DecoderWorkload {
            batch: 1,
            max_input_tokens: u64::try_from(rows).map_err(|_| Error::Bounds)?,
            max_new_tokens: u64::try_from(outputs).map_err(|_| Error::Bounds)?,
        };
        let (limits, storage) =
            source_policy::admitted_choice(&experiment, &config, None, workload)?;
        if storage != SourceStorage::Compact {
            return Err(Error::Binding);
        }
        let bootstrap =
            original_source::bootstrap_bytes(&config, limits).map_err(|_| Error::Bounds)?;
        let quote = original_source::quote(
            config,
            root.charge_external(bootstrap).map_err(|_| Error::Bounds)?,
            workload,
            limits,
        )
        .map_err(|_| Error::Binding)?;
        let phase = plus(
            root.budget()
                .used_bytes()
                .checked_sub(bootstrap)
                .ok_or(Error::Bounds)?,
            quote.resources().working_charge_bytes,
        )?;
        if phase > MAX_PROGRAM_BYTES {
            return Err(Error::Bounds);
        }
        root.budget().admit_peak(phase).map_err(|_| Error::Bounds)?;
        let original = quote.construct().map_err(|_| Error::Binding)?;
        let schema = "pllm.replicated-decoder.manifest.v1";
        let metadata = plus(
            size_of::<Manifest>(),
            plus(schema.len(), original.raw_configuration().len())?,
        )?;
        let mut manifest_charge = root.charge_external(0).map_err(|_| Error::Bounds)?;
        resize(
            &original,
            &mut manifest_charge,
            metadata,
            config_copy_visits(original.raw_configuration().len())?,
        )?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(original.raw_configuration().len())
            .map_err(|_| Error::Allocation)?;
        if bytes.capacity() != original.raw_configuration().len() {
            return Err(Error::Bounds);
        }
        bytes.extend_from_slice(original.raw_configuration());
        let params = &experiment.payload().value()["pipeline"]["components"]["linear"]["params"];
        let activation = params["max_abs_activation"]
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or(Error::Binding)?;
        let manifest = Manifest {
            schema: schema.into(),
            weights: WeightMode::Public,
            execution: ExecutionPolicy {
                range_policy: RangePolicy::Checked,
                max_abs_activation: activation,
                ..ExecutionPolicy::default()
            },
            config: bytes,
            rows,
            outputs,
            source,
            composition,
            session,
            tensors: BTreeMap::new(),
            norms: BTreeMap::new(),
            program: [0; 32],
        };
        let collection_charge = root.charge_external(0).map_err(|_| Error::Bounds)?;
        let directory =
            snapshot_directory::OwnedSnapshotDirectory::create(&original, &controller, &root)?;
        Ok(Self {
            manifest,
            directory,
            original,
            experiment,
            limits,
            manifest_charge,
            collection_charge,
            controller,
            root,
            failed: false,
        })
    }
    pub(in crate::replicated_decoder) fn original(&self) -> &OriginalSourcePlan {
        &self.original
    }
    pub(in crate::replicated_decoder) fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    pub(in crate::replicated_decoder) fn directory(
        &self,
    ) -> &Arc<snapshot_directory::OwnedSnapshotDirectory> {
        &self.directory
    }
    /// One actually-needed owned disk/snapshot counter handle; never a fresh
    /// counter or an ingress constructor accepting a caller's Budget identity.
    pub(in crate::replicated_decoder) fn clone_collection_budget(
        &self,
    ) -> Result<OwnedBudget, Error> {
        if self.failed {
            return Err(Error::Retired);
        }
        if plus(self.root.budget().used_bytes(), size_of::<OwnedBudget>())? > MAX_PROGRAM_BYTES {
            return Err(Error::Bounds);
        }
        self.root.try_clone().map_err(|_| Error::Bounds)
    }
    /// Totals include old/new/scratch overlap. No inner refund or guessed credit.
    pub(in crate::replicated_decoder) fn admit_collection(
        &mut self,
        manifest_bytes: usize,
        collection_bytes: usize,
        visits: u64,
    ) -> Result<(), Error> {
        if self.failed {
            return Err(Error::Retired);
        }
        let result = (|| {
            let old = plus(
                self.manifest_charge.payload_bytes(),
                self.collection_charge.payload_bytes(),
            )?;
            let next = plus(
                self.manifest_charge.payload_bytes().max(manifest_bytes),
                self.collection_charge.payload_bytes().max(collection_bytes),
            )?;
            let total = plus(
                self.root
                    .budget()
                    .used_bytes()
                    .checked_sub(old)
                    .ok_or(Error::Binding)?,
                next,
            )?;
            if total > MAX_PROGRAM_BYTES {
                return Err(Error::Bounds);
            }
            self.original
                .prepay_compiler_metadata(&self.controller, visits)
                .map_err(|_| Error::Bounds)?;
            self.manifest_charge
                .resize_payload(self.manifest_charge.payload_bytes().max(manifest_bytes))
                .map_err(|_| Error::Bounds)?;
            self.collection_charge
                .resize_payload(self.collection_charge.payload_bytes().max(collection_bytes))
                .map_err(|_| Error::Bounds)?;
            self.root
                .budget()
                .admit_peak(total)
                .map_err(|_| Error::Bounds)
        })();
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    pub(in crate::replicated_decoder) fn collect<T>(
        mut self,
        manifest_bytes: usize,
        collection_bytes: usize,
        visits: u64,
        build: impl FnOnce(&OriginalSourcePlan, SourceRecords<'_>) -> Result<T, Error>,
    ) -> Result<SourceCollector<T>, Error> {
        self.admit_collection(
            manifest_bytes,
            plus(
                collection_bytes,
                size_of::<SourceCollector<T>>()
                    .checked_mul(3)
                    .ok_or(Error::Bounds)?,
            )?,
            visits,
        )?;
        let payload = build(
            &self.original,
            SourceRecords {
                tensors: &mut self.manifest.tensors,
                norms: &mut self.manifest.norms,
            },
        )?;
        Ok(SourceCollector {
            payload: Some(payload),
            header: self,
        })
    }
    fn into_compact_import(self) -> Result<SourceRequestImport, Error> {
        if self.failed {
            return Err(Error::Retired);
        }
        self.original
            .prepay_compiler_metadata(
                &self.controller,
                u64::try_from(self.manifest_charge.payload_bytes()).map_err(|_| Error::Bounds)?,
            )
            .map_err(|_| Error::Bounds)?;
        let minimum = plus(
            size_of::<Manifest>(),
            plus(
                self.manifest.schema.capacity(),
                plus(
                    self.manifest.config.capacity(),
                    source_registry::source_metadata_bytes(
                        &self.manifest.tensors,
                        &self.manifest.norms,
                    )?,
                )?,
            )?,
        )?;
        if self.manifest_charge.payload_bytes() < minimum {
            return Err(Error::Bounds);
        }
        let inline = size_of::<OwnedCharged<Manifest>>()
            .checked_sub(size_of::<OwnedExternalCharge>())
            .ok_or(Error::Bounds)?;
        if plus(self.root.budget().used_bytes(), inline)? > MAX_PROGRAM_BYTES {
            return Err(Error::Bounds);
        }
        let Self {
            manifest,
            directory,
            original,
            experiment,
            limits,
            manifest_charge,
            collection_charge,
            controller,
            root,
            ..
        } = self;
        let manifest = manifest_charge
            .try_build(|_| Ok(manifest))
            .map_err(|_| Error::Bounds)?;
        // Remaining collection allowance has no detached payload: finish moves
        // its full actual lease to PayloadLease BEFORE entering this transition.
        drop(collection_charge);
        OwnedSourceIngress {
            directory: Some(directory),
            original,
            raw: SourceIngressRaw::CollectedHeader(experiment),
            limits,
            storage: SourceStorage::Compact,
            controller,
            root,
        }
        .into_pending(manifest)
    }
}
fn config_copy_visits(bytes: usize) -> Result<u64, Error> {
    u64::try_from(bytes.checked_add(1).ok_or(Error::Bounds)?).map_err(|_| Error::Bounds)
}

impl<T> SourceCollector<T> {
    pub(in crate::replicated_decoder) fn directory(
        &self,
    ) -> &Arc<snapshot_directory::OwnedSnapshotDirectory> {
        self.header.directory()
    }
    pub(in crate::replicated_decoder) fn payload(&self) -> Result<&T, Error> {
        self.payload.as_ref().ok_or(Error::Retired)
    }
    pub(in crate::replicated_decoder) fn manifest(&self) -> &Manifest {
        &self.header.manifest
    }
    pub(in crate::replicated_decoder) fn phase<U>(
        &mut self,
        manifest_bytes: usize,
        collection_bytes: usize,
        visits: u64,
        apply: impl FnOnce(&mut T, SourceRecords<'_>, &OriginalSourcePlan) -> Result<U, Error>,
    ) -> Result<U, Error> {
        let mut attempt = CollectorAttempt {
            owner: self,
            committed: false,
        };
        attempt
            .owner
            .header
            .admit_collection(manifest_bytes, collection_bytes, visits)?;
        let result = apply(
            attempt.owner.payload.as_mut().ok_or(Error::Retired)?,
            SourceRecords {
                tensors: &mut attempt.owner.header.manifest.tensors,
                norms: &mut attempt.owner.header.manifest.norms,
            },
            &attempt.owner.header.original,
        );
        if result.is_ok() {
            attempt.committed = true;
        }
        result
    }
    /// Retains T and its SAME lease through EVERY import/compile error. feed uses
    /// R's actual collected snapshot, never a reparsed/fresh-budget publisher.
    pub(in crate::replicated_decoder) fn finish(
        mut self,
        feed: impl FnOnce(&mut T, &mut SourceRequestImport) -> Result<(), Error>,
    ) -> Result<CompiledSource<T>, Error> {
        if self.header.failed {
            return Err(Error::Retired);
        }
        let zero = self
            .header
            .root
            .charge_external(0)
            .map_err(|_| Error::Bounds)?;
        let charge = std::mem::replace(&mut self.header.collection_charge, zero);
        let payload = self.payload.take().ok_or(Error::Retired)?;
        let Self { payload: _, header } = self;
        let mut retained = PayloadLease { payload, charge };
        let mut import = header.into_compact_import()?;
        feed(&mut retained.payload, &mut import)?;
        let program = import.finish()?;
        let PayloadLease { payload, charge } = retained;
        Ok(CompiledSource {
            payload: Some(payload),
            program,
            charge,
            failed: false,
        })
    }
}
impl<T> CompiledSource<T> {
    pub(in crate::replicated_decoder) fn directory(
        &self,
    ) -> Result<&Arc<snapshot_directory::OwnedSnapshotDirectory>, Error> {
        self.program()?.snapshot_directory()
    }
    pub(in crate::replicated_decoder) fn program(&self) -> Result<&SourceRequestProgram, Error> {
        if self.failed {
            return Err(Error::Retired);
        }
        Ok(&self.program)
    }
    pub(in crate::replicated_decoder) fn payload(&self) -> Result<&T, Error> {
        self.payload.as_ref().ok_or(Error::Retired)
    }
    pub(in crate::replicated_decoder) fn phase<U>(
        &mut self,
        bytes: usize,
        visits: u64,
        apply: impl FnOnce(&mut T) -> Result<U, Error>,
    ) -> Result<U, Error> {
        if self.failed {
            return Err(Error::Retired);
        }
        let mut attempt = CompiledAttempt {
            owner: self,
            committed: false,
        };
        let result = (|| {
            attempt
                .owner
                .program
                .collection_phase(&mut attempt.owner.charge, bytes, visits)?;
            apply(attempt.owner.payload.as_mut().ok_or(Error::Retired)?)
        })();
        if result.is_ok() {
            attempt.committed = true;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;
    #[test]
    fn owned_directory_lives_through_last_escaped_paged_reader() {
        use pllm_core::public_file::{PublicFileWriter, SnapshotDirectory};
        let mut header = begin();
        let path = header.directory().path().to_path_buf();
        let counter = header.root.try_clone().unwrap();
        let bytes = pllm_core::public_file::backing_bytes().unwrap()
            + 4 * size_of::<PublicFileWriter>()
            + 1024;
        header
            .admit_collection(header.manifest_charge.payload_bytes(), bytes, 64)
            .unwrap();
        let location: Arc<dyn SnapshotDirectory> =
            Arc::clone(header.directory()) as Arc<dyn SnapshotDirectory>;
        let mut writer = PublicFileWriter::new_computed_in(4, location).unwrap();
        writer.write(&[1, 2, 3, 4]).unwrap();
        let snapshot = writer.finish().unwrap();
        let matrix = pllm_core::paged::PagedMatrix::from_public_file(
            &snapshot,
            snapshot.digest(),
            1,
            4,
            Arc::new(pllm_core::kernels::Executor::new(1, false).unwrap()),
        )
        .unwrap();
        drop(snapshot);
        drop(header);
        assert!(path.is_dir());
        let live = counter.budget().used_bytes();
        drop(matrix);
        assert!(!path.exists());
        assert!(counter.budget().used_bytes() < live);
    }
    #[test]
    fn owned_directory_setup_exact_and_one_short_precedes_factory() {
        let header = begin();
        let need =
            snapshot_directory::OwnedSnapshotDirectory::additional_bytes_for_tests().unwrap();
        let mut outside = header.root.charge_external(0).unwrap();
        let exact = MAX_PROGRAM_BYTES - header.root.budget().used_bytes() - need;
        outside.resize_payload(exact + 1).unwrap();
        let used = header.root.budget().used_bytes();
        let factories = snapshot_directory::OwnedSnapshotDirectory::factories_for_tests();
        assert!(snapshot_directory::OwnedSnapshotDirectory::create(
            &header.original,
            &header.controller,
            &header.root
        )
        .is_err());
        assert_eq!(header.root.budget().used_bytes(), used);
        assert_eq!(
            snapshot_directory::OwnedSnapshotDirectory::factories_for_tests(),
            factories
        );
        outside.resize_payload(exact).unwrap();
        let second = snapshot_directory::OwnedSnapshotDirectory::create(
            &header.original,
            &header.controller,
            &header.root,
        )
        .unwrap();
        assert!(second.path().is_absolute());
        assert!(header.root.budget().used_bytes() <= MAX_PROGRAM_BYTES);
    }
    #[test]
    fn partial_snapshot_dies_before_collection_lease_on_ingress_failure() {
        use pllm_core::public_file::{PublicFileWriter, SnapshotDirectory};
        let header = begin();
        let directory: Arc<dyn SnapshotDirectory> =
            Arc::clone(header.directory()) as Arc<dyn SnapshotDirectory>;
        let weak = Arc::downgrade(header.directory());
        let path = directory.path().to_path_buf();
        let m = header.manifest_charge.payload_bytes();
        let collector = header
            .collect(
                m,
                pllm_core::public_file::writer_creation_bytes().unwrap() + 1024,
                64,
                |_, _| {
                    let mut writer = PublicFileWriter::new_computed_in(8, directory)
                        .map_err(|_| Error::Binding)?;
                    writer.write(&[0; 4]).map_err(|_| Error::Binding)?;
                    Ok(writer)
                },
            )
            .unwrap();
        assert!(collector
            .finish(|_, _| panic!("invalid records must fail before feed"))
            .is_err());
        assert!(weak.upgrade().is_none());
        assert!(!path.exists());
    }
    fn begin() -> SourceHeaderEntry {
        let config=br#"{"model_type":"qwen2","hidden_size":2,"intermediate_size":2,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"vocab_size":3,"max_position_embeddings":16,"rms_norm_eps":0.000001,"hidden_act":"silu","tie_word_embeddings":false,"attention_bias":false,"rope_theta":10000.0}"#;
        let experiment=serde_json::to_vec(&serde_json::json!({"schema":"pllm.experiment.v1","name":"compact-client-owner","pipeline":{"model":{"source":"original-tiny-source"},"components":{"linear":{"component":crate::replicated_decoder::sdk::PROTOCOL,"params":{"weights":"public","range_policy":"checked","max_abs_activation":1,"storage":"compact"}}}},"deployment":{"kind":"local","root":"."},"budget":{"requests":1,"max_input_tokens":4,"max_new_tokens":2}})).unwrap();
        SourceHeaderEntry::begin(&experiment, config, 4, 2, [1; 32], [2; 32], [3; 32]).unwrap()
    }
    #[test]
    fn collection_refusal_precedes_builder_and_cannot_retry() {
        let header = begin();
        assert!(header.manifest().tensors.is_empty());
        assert!(std::ptr::eq(
            header.original().budget(),
            header.root.budget()
        ));
        let m = header.manifest_charge.payload_bytes();
        let mut collector = header.collect(m, 256, 1, |_, _| Ok(())).unwrap();
        let before = collector.header.root.budget().used_bytes();
        let called = Cell::new(false);
        assert!(collector
            .phase(m, usize::MAX, 1, |_, _, _| {
                called.set(true);
                Ok(())
            })
            .is_err());
        assert!(!called.get());
        assert_eq!(collector.header.root.budget().used_bytes(), before);
        assert!(matches!(
            collector.phase(m, 0, 0, |_, _, _| Ok(())),
            Err(Error::Retired)
        ));
    }
    #[test]
    fn empty_collected_records_never_publish_and_payload_retires_before_lease() {
        struct Observe {
            counter: OwnedBudget,
            event: Rc<Cell<Option<usize>>>,
        }
        impl Drop for Observe {
            fn drop(&mut self) {
                self.event.set(Some(self.counter.budget().used_bytes()));
            }
        }
        let header = begin();
        let counter = header.root.try_clone().unwrap();
        let event = Rc::new(Cell::new(None));
        let m = header.manifest_charge.payload_bytes();
        let collector = header
            .collect(m, 256, 1, |_, _| {
                Ok(Observe {
                    counter,
                    event: Rc::clone(&event),
                })
            })
            .unwrap();
        let live_lease = collector.header.collection_charge.charged_bytes();
        assert!(collector
            .finish(|_, _| panic!("missing records reached feed"))
            .is_err());
        assert!(event.get().unwrap() >= live_lease);
    }
}
