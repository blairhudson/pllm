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
fn parent() -> Result<&'static Path, Error> {
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
    payload: OwnedCharged<DirectoryPayload>, // payload dies before original lease
}
impl OwnedSnapshotDirectory {
    #[cfg(test)]
    pub(super) fn factories_for_tests() -> usize {
        FACTORIES.with(|n| n.get())
    }
    pub(in crate::replicated_decoder) fn path(&self) -> &Path {
        self.payload.payload().directory.path()
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
        Ok(Arc::new(Self { payload }))
    }
    pub(in crate::replicated_decoder) fn check_budget(
        &self,
        budget: &pllm_garble::shared_allocation::Budget,
    ) -> Result<(), Error> {
        if !std::ptr::eq(self.payload.budget(), budget) {
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
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let anchor = self
                .payload
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
