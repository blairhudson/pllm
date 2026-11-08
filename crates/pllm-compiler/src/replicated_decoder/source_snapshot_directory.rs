//! Compiler-owned location/lease, never source or Program authority.
use super::*;
use pllm_core::public_file::SnapshotDirectory;
use std::{
    alloc::Layout,
    fs::File,
    path::{Path, PathBuf},
    sync::atomic::AtomicUsize,
};

// A fixed native parent avoids an environment change between admission and file
// creation. There is deliberately no client/server path selector or fallback.
pub(in crate::replicated_decoder) fn parent() -> Result<&'static Path, Error> {
    #[cfg(target_os = "macos")]
    {
        Ok(Path::new("/private/tmp"))
    }
    #[cfg(any(
        target_os = "linux",
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "netbsd"
    ))]
    {
        Ok(Path::new("/tmp"))
    }
    #[cfg(not(any(
        target_os = "macos",
        target_os = "linux",
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "netbsd"
    )))]
    {
        Err(Error::Binding)
    }
}
const PREFIX: &str = "pllm-source-";
const RANDOM_BYTES: usize = 12;
#[cfg(test)]
thread_local! {static FACTORIES:std::cell::Cell<usize>=const{std::cell::Cell::new(0)};}
// Fixed-parent + fixed-prefix/random-name path construction. Eight simultaneous
// boxed/path/name/copy slots, not an assumption that PathBuf capacity == length.
const PATH_SLOT_BYTES: usize = pllm_core::public_file::MAX_DIRECTORY_PATH_BYTES;
struct DirectoryPayload {
    anchor: File, // close handle before TempDir cleanup
    directory: tempfile::TempDir,
}
pub(in crate::replicated_decoder) struct OwnedSnapshotDirectory {
    payload: Option<OwnedCharged<DirectoryPayload>>, // payload dies before original lease
    pending: Option<Arc<RawSnapshotDirectory>>,
}
impl OwnedSnapshotDirectory {
    #[cfg(test)]
    pub(super) fn factories_for_tests() -> usize {
        FACTORIES.with(|n| n.get())
    }
    pub(in crate::replicated_decoder) fn path(&self) -> &Path {
        if let Some(pending) = &self.pending { return pending.path(); }
        self.payload.as_ref().expect("legacy directory").payload().directory.path()
    }
    /// Same location contract, but adopted before any filesystem acquisition.
    pub(in crate::replicated_decoder) fn prepare(
        original: &OriginalSourcePlan, controller: &OwnedExternalCharge, root: &OwnedBudget,
    ) -> Result<Arc<Self>, Error> {
        let pending = RawSnapshotDirectory::prepare_mode(original, controller, root, true)?;
        Ok(Arc::new(Self { payload: None, pending: Some(pending) }))
    }
    pub(in crate::replicated_decoder) fn start(&self) -> Result<(), Error> {
        let pending = self.pending.as_ref().ok_or(Error::Binding)?;
        pending.start_mode(RawSnapshotDirectory::parent_device()?, true)
    }
    pub(in crate::replicated_decoder) fn retire(&self) -> Result<(), Error> {
        self.pending.as_ref().ok_or(Error::Binding)?.retire()
    }
    #[cfg(test)]
    pub(in crate::replicated_decoder) fn pending_state_for_tests(&self) -> (bool, bool, bool, bool) {
        let pending = self.pending.as_ref().expect("pending compact directory");
        let (anchor, directory, attempted) = pending.retention_for_tests();
        (anchor, directory, attempted, pending.acquisition_unknown_for_tests())
    }
    fn quote() -> Result<usize, Error> {
        let (layout, _) = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<Self>())
            .map_err(|_| Error::Bounds)?;
        let mut n = layout.pad_to_align().size();
        for bytes in [
            8 * PATH_SLOT_BYTES,
            2 * size_of::<PathBuf>(),
            size_of::<tempfile::Builder<'_, '_>>(),
            size_of::<Result<tempfile::TempDir, std::io::Error>>(),
            size_of::<Result<File, std::io::Error>>(),
            size_of::<Result<Arc<Self>, Error>>(),
            size_of::<OriginalSourcePlan>(),
        ] {
            n = add(n, bytes)?;
        }
        Ok(n)
    }
    #[cfg(test)]
    pub(super) fn additional_bytes_for_tests() -> Result<usize, Error> {
        let inline = size_of::<OwnedCharged<DirectoryPayload>>()
            .checked_sub(size_of::<OwnedExternalCharge>())
            .ok_or(Error::Bounds)?;
        add(
            Self::quote()?,
            add(inline, size_of::<OwnedExternalCharge>())?,
        )
    }
    pub(in crate::replicated_decoder) fn create(
        original: &OriginalSourcePlan,
        controller: &OwnedExternalCharge,
        root: &OwnedBudget,
    ) -> Result<Arc<Self>, Error> {
        if !std::ptr::eq(root.budget(), original.budget()) || !controller.is_for(root.budget()) {
            return Err(Error::Binding);
        }
        let parent = parent()?;
        if !parent.is_absolute()
            || parent.as_os_str().len() + PREFIX.len() + RANDOM_BYTES + 1 > PATH_SLOT_BYTES - 32
        {
            return Err(Error::Binding);
        }
        let bytes = Self::quote()?;
        let inline = size_of::<OwnedCharged<DirectoryPayload>>()
            .checked_sub(size_of::<OwnedExternalCharge>())
            .ok_or(Error::Bounds)?;
        if add(
            root.budget().used_bytes(),
            add(bytes, add(inline, size_of::<OwnedExternalCharge>())?)?,
        )? > MAX_PROGRAM_BYTES
        {
            return Err(Error::Bounds);
        }
        let guard = root.charge_external(bytes).map_err(|_| Error::Bounds)?;
        original
            .prepay_compiler_metadata(controller, 64)
            .map_err(|_| Error::Bounds)?;
        root.budget()
            .admit_peak(root.budget().used_bytes())
            .map_err(|_| Error::Bounds)?;
        let payload = guard
            .try_build(|_| {
                #[cfg(test)]
                FACTORIES.with(|n| n.set(n.get() + 1));
                let directory = tempfile::Builder::new()
                    .prefix(PREFIX)
                    .rand_bytes(RANDOM_BYTES)
                    .tempdir_in(parent)
                    .map_err(|_| pllm_garble::shared_allocation::Error::Resource)?;
                let anchor = tempfile::tempfile_in(directory.path())
                    .map_err(|_| pllm_garble::shared_allocation::Error::Resource)?;
                #[cfg(unix)]
                {
                    use std::os::unix::fs::MetadataExt;
                    let d = directory
                        .path()
                        .metadata()
                        .map_err(|_| pllm_garble::shared_allocation::Error::Resource)?;
                    let a = anchor
                        .metadata()
                        .map_err(|_| pllm_garble::shared_allocation::Error::Resource)?;
                    if !d.is_dir() || a.len() != 0 || d.dev() != a.dev() {
                        return Err(pllm_garble::shared_allocation::Error::Resource);
                    }
                }
                #[cfg(not(unix))]
                {
                    return Err(pllm_garble::shared_allocation::Error::Resource);
                }
                Ok(DirectoryPayload { anchor, directory })
            })
            .map_err(|_| Error::Binding)?;
        if !std::ptr::eq(payload.budget(), original.budget()) {
            return Err(Error::Binding);
        }
        Ok(Arc::new(Self { payload: Some(payload), pending: None }))
    }
    pub(in crate::replicated_decoder) fn check_budget(
        &self,
        budget: &pllm_garble::shared_allocation::Budget,
    ) -> Result<(), Error> {
        if let Some(pending) = &self.pending { return pending.check_budget(budget); }
        if !std::ptr::eq(self.payload.as_ref().ok_or(Error::Binding)?.budget(), budget) {
            return Err(Error::Binding);
        }
        Ok(())
    }
}
impl SnapshotDirectory for OwnedSnapshotDirectory {
    fn path(&self) -> &Path {
        Self::path(self)
    }
    fn check_file(&self, file: pllm_core::public_file::FileBacking) -> Result<(), String> {
        if let Some(pending) = &self.pending { return pending.check_file(file); }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let anchor = self
                .payload
                .as_ref().ok_or("directory unavailable")?
                .payload()
                .anchor
                .metadata()
                .map_err(|e| e.to_string())?;
            if file.device != anchor.dev() {
                return Err("snapshot filesystem differs from retained anchor".into());
            }
            Ok(())
        }
        #[cfg(not(unix))]
        {
            let _ = file;
            Err("snapshot filesystem identity unsupported".into())
        }
    }
}

/// Shared two-phase location. This object is adopted before its first factory;
/// every partial factory result stays here until explicit fallible retirement.
/// Legacy OwnedSnapshotDirectory/drop behavior above is deliberately unchanged.
pub(in crate::replicated_decoder) struct RawSnapshotDirectory {
    path: std::sync::OnceLock<PathBuf>,
    state: std::sync::Mutex<RawDirectoryState>,
    charge: OwnedExternalCharge,
}
struct RawDirectoryState {
    anchor: Option<File>,
    directory: Option<tempfile::TempDir>,
    // Explicit retention, not TempDir::drop or a fabricated cleanup receipt.
    // Broken only after successful nonrecursive removal. A lost caller handle
    // cannot turn a failed retire into recursive cleanup/refund.
    retained: Option<Arc<RawSnapshotDirectory>>,
    attempted: bool,
    retire_attempted: bool,
    retired: bool,
    // A third-party factory error cannot prove that no directory was created.
    // No returned handle/path means uncertainty, NOT a live-directory claim.
    acquisition_unknown: bool,
}
#[cfg(test)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::replicated_decoder) enum CompactFault { Before, Opaque, Directory, Anchor, Metadata, Remove }
#[cfg(test)]
thread_local! {
    static COMPACT_FAULT: std::cell::Cell<Option<CompactFault>> = const { std::cell::Cell::new(None) };
}
#[cfg(test)]
pub(in crate::replicated_decoder) fn set_compact_fault(fault: Option<CompactFault>) {
    COMPACT_FAULT.with(|slot| slot.set(fault));
}
#[cfg(test)]
fn compact_fault(point: CompactFault) -> Result<(), Error> {
    if COMPACT_FAULT.with(|slot| slot.get()) == Some(point) { Err(Error::Binding) } else { Ok(()) }
}
impl RawSnapshotDirectory {
    pub(in crate::replicated_decoder) fn prepare(
        original: &OriginalSourcePlan, controller: &OwnedExternalCharge, root: &OwnedBudget,
    ) -> Result<Arc<Self>, Error> {
        Self::prepare_mode(original, controller, root, false)
    }
    fn prepare_mode(
        original: &OriginalSourcePlan, controller: &OwnedExternalCharge, root: &OwnedBudget,
        wrapped: bool,
    ) -> Result<Arc<Self>, Error> {
        if !std::ptr::eq(root.budget(), original.budget()) || !controller.is_for(root.budget()) {
            return Err(Error::Binding);
        }
        // quote() already includes the directory wrapper and one Arc control.
        // Compact wraps the shared pending location: pay its second Arc too,
        // on this SAME original ledger, without changing any admission ceiling.
        let control = if wrapped { size_of::<[AtomicUsize; 2]>() } else { 0 };
        let bytes = add(OwnedSnapshotDirectory::quote()?, add(size_of::<Self>(), control)?)?;
        if add(root.budget().used_bytes(), bytes)? > MAX_PROGRAM_BYTES { return Err(Error::Bounds); }
        original.prepay_compiler_metadata(controller, 64).map_err(|_| Error::Bounds)?;
        let charge = root.charge_external(bytes).map_err(|_| Error::Bounds)?;
        Ok(Arc::new(Self { path: std::sync::OnceLock::new(),
            state: std::sync::Mutex::new(RawDirectoryState {
                anchor: None, directory: None, retained: None, attempted: false,
                retire_attempted: false, retired: false, acquisition_unknown: false,
            }), charge }))
    }
    pub(in crate::replicated_decoder) fn parent_device() -> Result<u64, Error> {
        #[cfg(unix)] {
            use std::os::unix::fs::MetadataExt;
            let metadata = parent()?.metadata().map_err(|_| Error::Binding)?;
            if !metadata.is_dir() { return Err(Error::Binding); }
            Ok(metadata.dev())
        }
        #[cfg(not(unix))] { Err(Error::Binding) }
    }
    pub(in crate::replicated_decoder) fn start(self: &Arc<Self>, device: u64) -> Result<(), Error> {
        self.start_mode(device, false)
    }
    fn start_mode(self: &Arc<Self>, device: u64, compact: bool) -> Result<(), Error> {
        let mut state = self.state.lock().map_err(|_| Error::Binding)?;
        if state.attempted || state.retired { return Err(Error::Retired); }
        state.attempted = true;
        state.retained = Some(Arc::clone(self));
        if Self::parent_device()? != device { return Err(Error::Binding); }
        #[cfg(test)]
        if compact { compact_fault(CompactFault::Before)?; }
        #[cfg(test)] FACTORIES.with(|n| n.set(n.get() + 1));
        // Store the directory BEFORE path-copy/anchor/fstat, so errors cannot
        // accidentally turn TempDir::drop into a retirement proof.
        state.acquisition_unknown = compact;
        #[cfg(test)]
        if compact { compact_fault(CompactFault::Opaque)?; }
        state.directory = Some(tempfile::Builder::new().prefix(PREFIX)
            .rand_bytes(RANDOM_BYTES).tempdir_in(parent()?).map_err(|_| Error::Binding)?);
        state.acquisition_unknown = false;
        let directory = state.directory.as_ref().ok_or(Error::Binding)?;
        self.path.set(directory.path().to_path_buf()).map_err(|_| Error::Binding)?;
        #[cfg(test)]
        if compact { compact_fault(CompactFault::Directory)?; }
        state.anchor = Some(tempfile::tempfile_in(self.path()).map_err(|_| Error::Binding)?);
        #[cfg(test)]
        if compact { compact_fault(CompactFault::Anchor)?; }
        #[cfg(unix)] {
            use std::os::unix::fs::MetadataExt;
            let anchor = state.anchor.as_ref().ok_or(Error::Binding)?.metadata().map_err(|_| Error::Binding)?;
            if anchor.dev() != device || anchor.len() != 0 { return Err(Error::Binding); }
            if compact {
                let directory = self.path().metadata().map_err(|_| Error::Binding)?;
                if !directory.is_dir() || directory.dev() != anchor.dev() { return Err(Error::Binding); }
            }
        }
        #[cfg(test)]
        if compact { compact_fault(CompactFault::Metadata)?; }
        Ok(())
    }
    pub(in crate::replicated_decoder) fn retire(&self) -> Result<(), Error> {
        let mut state = self.state.lock().map_err(|_| Error::Binding)?;
        if state.retired { return Ok(()); }
        if state.retire_attempted { return Err(Error::Retired); }
        state.retire_attempted = true;
        if state.acquisition_unknown { return Err(Error::Binding); }
        if let Some(directory) = state.directory.as_ref() {
            // A successful explicit syscall is proof; Drop is not. Failed
            // removal leaves the exact directory object and charge retained.
            #[cfg(test)] compact_fault(CompactFault::Remove)?;
            std::fs::remove_dir(directory.path()).map_err(|_| Error::Binding)?;
        }
        if let Some(directory) = state.directory.take() { let _ = directory.keep(); }
        // Keep anchor on every remove_dir failure. Its disappearance alone is
        // never a filesystem retirement certificate.
        drop(state.anchor.take());
        state.retired = true;
        drop(state.retained.take());
        Ok(())
    }
    fn path(&self) -> &Path { self.path.get().expect("raw snapshot started").as_path() }
    pub(in crate::replicated_decoder) fn check_budget(&self, budget: &pllm_garble::shared_allocation::Budget) -> Result<(), Error> {
        if !self.charge.is_for(budget) { return Err(Error::Binding); }
        Ok(())
    }
    #[cfg(test)]
    pub(in crate::replicated_decoder) fn retention_for_tests(&self) -> (bool, bool, bool) {
        let state = self.state.lock().unwrap();
        (state.anchor.is_some(), state.directory.is_some(), state.retire_attempted)
    }
    #[cfg(test)]
    pub(in crate::replicated_decoder) fn acquisition_unknown_for_tests(&self) -> bool {
        self.state.lock().unwrap().acquisition_unknown
    }
}
impl SnapshotDirectory for RawSnapshotDirectory {
    fn path(&self) -> &Path { Self::path(self) }
    fn check_file(&self, file: pllm_core::public_file::FileBacking) -> Result<(), String> {
        let state = self.state.lock().map_err(|_| "raw directory poisoned")?;
        if state.retired || state.retire_attempted { return Err("raw directory retired".into()); }
        #[cfg(unix)] {
            use std::os::unix::fs::MetadataExt;
            let anchor = state.anchor.as_ref().ok_or("raw directory not started")?
                .metadata().map_err(|e| e.to_string())?;
            if file.device != anchor.dev() { return Err("raw snapshot device differs".into()); }
            Ok(())
        }
        #[cfg(not(unix))] { let _ = file; Err("raw filesystem unsupported".into()) }
    }
}
