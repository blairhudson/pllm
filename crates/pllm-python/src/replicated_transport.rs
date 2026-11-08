//! Bounded party-local bindings; parse failures retire the complete owner.
use pllm_compiler::replicated_decoder::sdk::{SourceRequestImport, SourceRequestProgram};
use pllm_compiler::replicated_decoder::{
    ExecutionPolicy, LookupPolicy, Manifest, PeerTransport, Program, PublicSource,
    PublicSourceImport, Publisher, RangePolicy, SelectionPolicy, WeightMode, Worker,
};
use pllm_core::puma::{multiply, prss, Party};
use pyo3::{
    prelude::*,
    types::{PyAny, PyBool, PyBytes, PyDict, PyTuple},
};
use std::sync::Arc;
use zeroize::Zeroizing;

pyo3::create_exception!(
    pllm._native,
    NumericDomainRejected,
    pyo3::exceptions::PyValueError
);

fn error(message: impl ToString) -> PyErr {
    crate::invalid(message.to_string())
}
fn arguments<'py, const N: usize>(
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
    names: [&str; N],
) -> PyResult<[Bound<'py, PyAny>; N]> {
    crate::owned_arguments::arguments(args, kwargs, names, "replicated")
}
fn bytes<'a>(value: &'a Bound<'_, PyAny>, limit: usize) -> PyResult<&'a [u8]> {
    let data = value.cast_exact::<PyBytes>()?.as_bytes();
    if data.len() > limit {
        return Err(error("replicated frame exceeds bound"));
    }
    Ok(data)
}
fn integer(value: &Bound<'_, PyAny>) -> PyResult<usize> {
    if value.is_instance_of::<PyBool>() {
        return Err(error("replicated integer cannot be bool"));
    }
    value.extract()
}
fn party(value: usize) -> PyResult<Party> {
    match value {
        0 => Ok(Party::P0),
        1 => Ok(Party::P1),
        2 => Ok(Party::P2),
        _ => Err(error("replicated party outside 0..3")),
    }
}
fn pin(value: &[u8]) -> PyResult<[u8; 32]> {
    value
        .try_into()
        .map_err(|_| error("replicated digest must be 32 bytes"))
}

#[pyclass(module = "pllm._native", name = "ReplicatedPublisher")]
pub struct NativePublisher {
    owner: Option<Publisher>,
    pid: u32,
}

type SourceChunk = (String, usize, Vec<Py<PyBytes>>);
type PeerBatch = (bool, Vec<(usize, Py<PyBytes>)>);

impl NativePublisher {
    fn retained_compact(&mut self) -> PyResult<Option<&mut Publisher>> {
        if self.pid != std::process::id() {
            return Err(error("replicated publisher cannot cross fork"));
        }
        Ok(self.owner.as_mut().filter(|owner| owner.has_retained_compact()))
    }
    fn take(&mut self) -> PyResult<Publisher> {
        // A raw host is borrowed in place through BOTH explicit cleanup phases.
        // Legacy consuming paths may never discard it, even on an error/fork.
        if self.owner.as_ref().is_some_and(|o| o.has_raw_host() || o.has_retained_compact()) {
            return Err(error("raw host requires explicit retained-owner cleanup"));
        }
        let owner = self
            .owner
            .take()
            .ok_or_else(|| error("replicated publisher retired"))?;
        if self.pid != std::process::id() {
            return Err(error("replicated publisher cannot cross fork"));
        }
        Ok(owner)
    }

    fn raw_owner(&self) -> PyResult<&Publisher> {
        if self.pid != std::process::id() {
            return Err(error("replicated publisher cannot cross fork"));
        }
        self.owner
            .as_ref()
            .filter(|owner| owner.has_raw_host())
            .ok_or_else(|| error("publisher has no retained raw host"))
    }
}

#[pymethods]
impl NativePublisher {
    // These hooks add no Python owner/guard class. Native SOURCE logical slots
    // prepay before Publisher construction on its ORIGINAL ledger. Returned
    // Python scalars/bytes are not a whole-Python allocation or fit claim.
    #[cfg(feature = "lifecycle-test-support")]
    #[staticmethod]
    fn _test_raw_prepare_fixture(py: Python<'_>) -> PyResult<Self> {
        let owner = py.detach(Publisher::test_raw_prepare_fixture).map_err(error)?;
        Ok(Self { owner: Some(owner), pid: std::process::id() })
    }
    #[cfg(feature = "lifecycle-test-support")]
    fn _test_hold_reader(&mut self) -> PyResult<()> {
        self.raw_owner()?;
        self.owner.as_mut().ok_or_else(|| error("publisher retired"))?
            .test_hold_reader().map_err(error)
    }
    #[cfg(feature = "lifecycle-test-support")]
    fn _test_release_reader(&mut self) -> PyResult<()> {
        self.raw_owner()?;
        self.owner.as_mut().ok_or_else(|| error("publisher retired"))?
            .test_release_reader().map_err(error)
    }
    #[cfg(feature = "lifecycle-test-support")]
    fn _test_raw_snapshot(&self) -> PyResult<[usize; 10]> {
        self.raw_owner()?.test_raw_snapshot().map_err(error)
    }
    #[cfg(feature = "lifecycle-test-support")]
    fn _test_raw_inventory<'py>(&self, py: Python<'py>, file: usize)
        -> PyResult<(u64, Bound<'py, PyBytes>, usize)>
    {
        let (size, digest, descriptors) = self.raw_owner()?.test_raw_inventory(file).map_err(error)?;
        Ok((size, PyBytes::new(py, &digest), descriptors))
    }
    #[cfg(feature = "lifecycle-test-support")]
    fn _test_raw_chunk<'py>(&self, py: Python<'py>, file: usize, offset: usize)
        -> PyResult<Bound<'py, PyBytes>>
    {
        let (chunk, len) = self.raw_owner()?.test_raw_chunk(file, offset).map_err(error)?;
        Ok(PyBytes::new(py, &chunk[..len]))
    }
    /// No Python value can supply HostRawExpectations. The compiler facade uses
    /// its startup-configured host provider; absent-provider calls refuse before
    /// header/filesystem factories. This does not activate SDK raw publication.
    #[staticmethod]
    #[pyo3(signature=(*args,**kwargs))]
    fn _raw_prepare(
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let [experiment, config, rows, outputs, session, source, composition] = arguments(
            args,
            kwargs,
            ["experiment", "config", "rows", "outputs", "session", "source", "composition"],
        )?;
        let experiment = experiment.cast_exact::<PyBytes>()?.as_bytes();
        let config = config.cast_exact::<PyBytes>()?.as_bytes();
        let rows = integer(&rows)?;
        let outputs = integer(&outputs)?;
        let session = pin(session.cast_exact::<PyBytes>()?.as_bytes())?;
        let source = pin(source.cast_exact::<PyBytes>()?.as_bytes())?;
        let composition = pin(composition.cast_exact::<PyBytes>()?.as_bytes())?;
        let owner = py
            .detach(|| Publisher::prepare_raw_host(
                experiment, config, rows, outputs, session, source, composition,
            ))
            .map_err(error)?;
        Ok(Self {
            owner: Some(owner),
            pid: std::process::id(),
        })
    }
    #[getter]
    fn _raw_disk_bytes(&self) -> PyResult<usize> {
        self.raw_owner()?.raw_disk_bytes().map_err(error)
    }
    #[getter]
    fn _raw_disk_directory(&self) -> PyResult<&str> {
        self.raw_owner()?
            .raw_disk_directory()
            .map_err(error)?
            .to_str()
            .ok_or_else(|| error("raw snapshot parent is not UTF-8"))
    }
    fn _raw_start(&self, py: Python<'_>, device: &Bound<'_, PyAny>) -> PyResult<()> {
        let device = u64::try_from(integer(device)?).map_err(error)?;
        let owner = self.raw_owner()?;
        py.detach(|| owner.raw_start(device)).map_err(error)
    }
    fn _raw_append(
        &self,
        py: Python<'_>,
        file: &Bound<'_, PyAny>,
        offset: &Bound<'_, PyAny>,
        data: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let file = integer(file)?;
        let offset = u64::try_from(integer(offset)?).map_err(error)?;
        let data = bytes(data, pllm_core::public_file::MAX_CHUNK_BYTES)?;
        let owner = self.raw_owner()?;
        py.detach(|| owner.raw_append(file, offset, data)).map_err(error)
    }
    fn _raw_finish(&self, py: Python<'_>) -> PyResult<()> {
        let owner = self.raw_owner()?;
        py.detach(|| owner.raw_finish()).map_err(error)
    }
    // Ownership reservation only; no Python-supplied source authority.
    // Source-only handoff: requires CompilerPublisher::raw_begin_retire.
    fn _raw_begin_retire(&self, py: Python<'_>) -> PyResult<bool> {
        let owner = self.raw_owner()?;
        py.detach(|| owner.raw_begin_retire()).map_err(error)
    }
    fn _raw_retire(&self, py: Python<'_>) -> PyResult<()> {
        let owner = self.raw_owner()?;
        py.detach(|| owner.raw_retire()).map_err(error)
    }
    fn _raw_release_final_charges(&self, py: Python<'_>) -> PyResult<()> {
        let owner = self.raw_owner()?;
        py.detach(|| owner.raw_release_final_charges()).map_err(error)
    }
    /// Resource-free, close-ready pending compact owner. Python adopts this
    /// exact object before _compact_start may acquire a directory or anchor.
    /// Compiler admission owns policy/config/source; no Python storage authority.
    #[staticmethod]
    #[pyo3(signature=(*args,**kwargs))]
    fn _from_compact_source_request(
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let [experiment, config, rows, outputs, source, composition] = arguments(
            args,
            kwargs,
            [
                "experiment",
                "config",
                "rows",
                "outputs",
                "source",
                "composition",
            ],
        )?;
        // Immutable exact bytes remain rooted through detach. Their original
        // admission belongs to SourceHeaderEntry, before any source collection.
        let experiment = experiment.cast_exact::<PyBytes>()?.as_bytes();
        let config = config.cast_exact::<PyBytes>()?.as_bytes();
        let rows = integer(&rows)?;
        let outputs = integer(&outputs)?;
        let source = pin(source.cast_exact::<PyBytes>()?.as_bytes())?;
        let composition = pin(composition.cast_exact::<PyBytes>()?.as_bytes())?;
        let owner = py
            .detach(|| {
                Publisher::prepare_compact_source_request(
                    experiment,
                    config,
                    rows,
                    outputs,
                    source,
                    composition,
                )
            })
            .map_err(error)?;
        Ok(Self {
            owner: Some(owner),
            pid: std::process::id(),
        })
    }
    /// Private second phase: Python has adopted this exact pending owner.
    fn _compact_start(&mut self, py: Python<'_>) -> PyResult<()> {
        let owner = self.retained_compact()?.ok_or_else(|| error("no pending compact owner"))?;
        py.detach(|| owner.compact_start()).map_err(error)
    }
    #[getter]
    fn _storage_kind(&self) -> PyResult<Option<&'static str>> {
        let owner = self
            .owner
            .as_ref()
            .ok_or_else(|| error("replicated publisher retired"))?;
        if self.pid != std::process::id() {
            return Err(error("replicated publisher cannot cross fork"));
        }
        Ok(owner.storage_kind())
    }
    /// Disk permission is supplied/retained by the caller, not by this native
    /// memory counter. Observe the actual total BEFORE any coefficient append.
    #[getter]
    fn collection_disk_bytes(&self) -> PyResult<usize> {
        let owner = self
            .owner
            .as_ref()
            .ok_or_else(|| error("replicated publisher retired"))?;
        if self.pid != std::process::id() {
            return Err(error("replicated publisher cannot cross fork"));
        }
        Ok(owner.collection_disk_bytes())
    }
    /// Location observation only, not a disk lease or source authority. The
    /// PyO3 self borrow keeps the actual native directory owner alive until the
    /// returned borrowed str has been converted to a Python string. No native
    /// path copy/cache or ambient temporary-directory lookup is involved.
    /// Pending compact owners report the fixed parent, not live backing.
    #[getter]
    fn collection_disk_directory(&self) -> PyResult<&str> {
        let owner = self
            .owner
            .as_ref()
            .ok_or_else(|| error("replicated publisher retired"))?;
        if self.pid != std::process::id() {
            return Err(error("replicated publisher cannot cross fork"));
        }
        owner
            .collection_disk_directory()
            .map_err(error)?
            .to_str()
            .ok_or_else(|| error("replicated snapshot directory is not UTF-8"))
    }
    #[new]
    #[pyo3(signature=(config, rows, outputs, source, composition, weights="private", *, range_policy="static", max_abs_activation=256, lookup="per_token", selection="expanded", peer_transport="http"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        config: &Bound<'_, PyBytes>,
        rows: usize,
        outputs: usize,
        source: &Bound<'_, PyBytes>,
        composition: &Bound<'_, PyBytes>,
        weights: &str,
        range_policy: &str,
        max_abs_activation: u32,
        lookup: &str,
        selection: &str,
        peer_transport: &str,
    ) -> PyResult<Self> {
        let config = config.as_bytes();
        let source = pin(source.as_bytes())?;
        let composition = pin(composition.as_bytes())?;
        let weights = match weights {
            "private" => WeightMode::Private,
            "public" => WeightMode::Public,
            _ => return Err(error("invalid weight visibility")),
        };
        let execution = ExecutionPolicy {
            range_policy: match range_policy {
                "static" => RangePolicy::Static,
                "checked" => RangePolicy::Checked,
                _ => return Err(error("invalid range policy")),
            },
            max_abs_activation,
            lookup: match lookup {
                "per_token" => LookupPolicy::PerToken,
                "tiled" => LookupPolicy::Tiled,
                "trie" => LookupPolicy::Trie,
                _ => return Err(error("invalid lookup policy")),
            },
            selection: match selection {
                "expanded" => SelectionPolicy::Expanded,
                "tiled" => SelectionPolicy::Tiled,
                _ => return Err(error("invalid selection policy")),
            },
            peer_transport: match peer_transport {
                "http" => PeerTransport::Http,
                "streams" => PeerTransport::Streams,
                _ => return Err(error("invalid peer transport")),
            },
        };
        execution.validate(weights).map_err(error)?;
        let owner = py
            .detach(|| {
                Publisher::new_with_policy(
                    config,
                    rows,
                    outputs,
                    source,
                    composition,
                    weights,
                    execution,
                )
            })
            .map_err(error)?;
        Ok(Self {
            owner: Some(owner),
            pid: std::process::id(),
        })
    }
    #[getter]
    fn shapes(&self) -> PyResult<Vec<(String, Vec<usize>)>> {
        Ok(self
            .owner
            .as_ref()
            .ok_or_else(|| error("replicated publisher retired"))?
            .shapes()
            .to_vec())
    }
    #[pyo3(signature=(*args,**kwargs))]
    fn append(
        &mut self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        if let Some(owner) = self.retained_compact()? {
            let [key, offset, data] = arguments(args, kwargs, ["key", "offset", "data"])?;
            let key: &str = key.extract()?;
            if key.len() > 512 { return Err(error("replicated key too long")); }
            let offset = integer(&offset)?;
            let data = bytes(&data, 16384)?;
            return py.detach(|| owner.append(key, offset, data)).map_err(error);
        }
        let mut owner = self.take()?;
        let [key, offset, data] = arguments(args, kwargs, ["key", "offset", "data"])?;
        let key: &str = key.extract()?;
        if key.len() > 512 {
            return Err(error("replicated key too long"));
        }
        let offset = integer(&offset)?;
        let data = bytes(&data, 16384)?;
        py.detach(|| owner.append(key, offset, data))
            .map_err(error)?;
        self.owner = Some(owner);
        Ok(())
    }
    #[pyo3(signature=(*args,**kwargs))]
    fn seal(
        &mut self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyBytes>> {
        if let Some(owner) = self.retained_compact()? {
            let [ids] = arguments(args, kwargs, ["ids"])?;
            let raw = bytes(&ids, 128)?;
            if raw.len() % 4 != 0 { return Err(error("replicated IDs require u32 bytes")); }
            let ids = Zeroizing::new(raw.chunks_exact(4)
                .map(|v| u32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>());
            let manifest = py.detach(|| { owner.seal(&ids)?; owner.manifest_checked() }).map_err(error)?;
            return Ok(PyBytes::new(py, &manifest).unbind());
        }
        let mut owner = self.take()?;
        let [ids] = arguments(args, kwargs, ["ids"])?;
        let raw = bytes(&ids, 128)?;
        if raw.len() % 4 != 0 {
            return Err(error("replicated IDs require u32 bytes"));
        }
        let ids = Zeroizing::new(
            raw.chunks_exact(4)
                .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
                .collect::<Vec<_>>(),
        );
        let manifest = py
            .detach(|| {
                owner.seal(&ids)?;
                owner.manifest_checked()
            })
            .map_err(error)?;
        let result = PyBytes::new(py, &manifest).unbind();
        self.owner = Some(owner);
        Ok(result)
    }
    #[pyo3(signature=(*args,**kwargs))]
    fn next_chunk(
        &mut self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Option<SourceChunk>> {
        if let Some(owner) = self.retained_compact()? {
            arguments(args, kwargs, [])?;
            return Ok(py.detach(|| owner.next_chunk()).map_err(error)?.map(|v| (
                v.key, v.offset, v.shares.iter().map(|b| PyBytes::new(py, b).unbind()).collect(),
            )));
        }
        let mut owner = self.take()?;
        arguments(args, kwargs, [])?;
        let result = py.detach(|| owner.next_chunk()).map_err(error)?.map(|v| {
            (
                v.key,
                v.offset,
                v.shares
                    .iter()
                    .map(|b| PyBytes::new(py, b).unbind())
                    .collect(),
            )
        });
        self.owner = Some(owner);
        Ok(result)
    }
    #[pyo3(signature=(*args,**kwargs))]
    fn finish(
        &mut self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<u32>> {
        if let Some(owner) = self.retained_compact()? {
            let [a, b, c] = arguments(args, kwargs, ["first", "second", "third"])?;
            let frames = [bytes(&a, 145)?, bytes(&b, 145)?, bytes(&c, 145)?];
            return py.detach(|| owner.compact_finish_numeric_outcome(frames)).map_err(error)?
                .ok_or_else(|| NumericDomainRejected::new_err("replicated request failed protected range checks"));
        }
        let owner = self.take()?;
        let [a, b, c] = arguments(args, kwargs, ["first", "second", "third"])?;
        let frames = [bytes(&a, 145)?, bytes(&b, 145)?, bytes(&c, 145)?];
        py.detach(|| owner.finish_numeric_outcome(frames))
            .map_err(error)?
            .ok_or_else(|| {
                NumericDomainRejected::new_err("replicated request failed protected range checks")
            })
    }
    #[pyo3(signature=(*args,**kwargs))]
    fn close(
        &mut self,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        if let Some(owner) = self.retained_compact()? {
            arguments(args, kwargs, [])?;
            owner.compact_retire().map_err(error)?;
            // Keep final native charges until Python releases the exact leases.
            return Ok(());
        }
        if self.owner.as_ref().is_some_and(Publisher::has_raw_host) {
            return Err(error("raw host requires explicit retained-owner cleanup"));
        }
        self.owner = None;
        arguments(args, kwargs, [])?;
        Ok(())
    }
}

// The request allowance includes this fixed inline owner. Avoid an additional
// infallible allocation when setup transfers into the decoder.
#[allow(clippy::large_enum_variant)]
enum State {
    SourceRequest(SourceRequestImport),
    SourceSetup {
        setup: multiply::SetupParty,
        request: SourceRequestProgram,
    },
    PublicSource(PublicSourceImport),
    Setup(multiply::SetupParty),
    Run(Worker),
}
struct Owner {
    state: State,
    program: Option<Arc<Program>>,
    session: [u8; 32],
    party: Party,
    public: Option<PublicSource>,
}
#[pyclass(module = "pllm._native", name = "ReplicatedWorker")]
pub struct NativeWorker {
    owner: Option<Owner>,
    pid: u32,
}
impl NativeWorker {
    fn take(&mut self) -> PyResult<Owner> {
        let owner = self
            .owner
            .take()
            .ok_or_else(|| error("replicated worker retired"))?;
        if self.pid != std::process::id() {
            return Err(error("replicated worker cannot cross fork"));
        }
        Ok(owner)
    }
}
#[pymethods]
impl NativeWorker {
    /// Private qualification observation only. Read the actual retained schedule
    /// owner; never the requested policy, Manifest field or a cached flag.
    /// An unsealed importer has no Program. Retirement destroys the observation.
    #[getter]
    fn _storage_kind(&self) -> PyResult<Option<&'static str>> {
        let owner = self
            .owner
            .as_ref()
            .ok_or_else(|| error("replicated worker retired"))?;
        if self.pid != std::process::id() {
            return Err(error("replicated worker cannot cross fork"));
        }
        Ok(owner.program.as_ref().map(|program| program.storage_kind()))
    }
    #[new]
    fn new(py: Python<'_>, manifest: &Bound<'_, PyBytes>, role: usize) -> PyResult<Self> {
        let raw = manifest.as_bytes();
        if raw.len() > 1 << 20 {
            return Err(error("replicated manifest too large"));
        }
        let party = party(role)?;
        let owner = py
            .detach(|| -> Result<_, String> {
                let manifest: Manifest =
                    serde_json::from_slice(raw).map_err(|_| "invalid replicated manifest")?;
                let program = manifest.compile()?;
                let session = manifest.session;
                let state = if manifest.weights == WeightMode::Public {
                    State::PublicSource(PublicSourceImport::new(&manifest)?)
                } else {
                    State::Setup(program.setup(session, party).map_err(|e| e.to_string())?)
                };
                Ok(Owner {
                    state,
                    program: Some(program),
                    session,
                    party,
                    public: None,
                })
            })
            .map_err(error)?;
        Ok(Self {
            owner: Some(owner),
            pid: std::process::id(),
        })
    }
    /// Private opt-in. Python supplies immutable wire/config bytes and existing
    /// role only; ALL factory/source/token/history authority is minted in Rust.
    /// CPython allocator/RSS are not part of the compiler's logical admission.
    #[staticmethod]
    #[pyo3(signature = (manifest, role, *, server_policy=None))]
    fn _from_source_request(
        py: Python<'_>,
        manifest: &Bound<'_, PyBytes>,
        role: usize,
        server_policy: Option<(Bound<'_, PyBytes>, Bound<'_, PyBytes>)>,
    ) -> PyResult<Self> {
        let party = party(role)?;
        let raw = manifest.as_bytes();
        // Only the canonical SERVER supplies this private tuple: authenticated
        // Experiment then config bytes. Never read it from a client Manifest.
        // Immutable PyBytes stay borrowed/alive across detach; no coefficient or
        // source/epoch authority is minted by Python. No CPython capacity claim.
        let borrowed_policy = server_policy
            .as_ref()
            .map(|(experiment, config)| (experiment.as_bytes(), config.as_bytes()));
        let importer = py
            .detach(|| match borrowed_policy {
                Some((experiment, config)) => {
                    SourceRequestImport::new_checked_public(raw, experiment, config)
                }
                None => SourceRequestImport::new(raw),
            })
            .map_err(error)?;
        let session = importer.session().map_err(error)?;
        Ok(Self {
            owner: Some(Owner {
                state: State::SourceRequest(importer),
                program: None,
                session,
                party,
                public: None,
            }),
            pid: std::process::id(),
        })
    }
    #[pyo3(signature=(*args,**kwargs))]
    fn append_public_source(
        &mut self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let mut owner = self.take()?;
        let [key, offset, data] = arguments(args, kwargs, ["key", "offset", "data"])?;
        let key: &str = key.extract()?;
        if key.len() > 512 {
            return Err(error("replicated key too long"));
        }
        let offset = integer(&offset)?;
        let data = bytes(&data, 16384)?;
        match &mut owner.state {
            State::PublicSource(importer) => py
                .detach(|| importer.append(key, offset, data))
                .map_err(error)?,
            State::SourceRequest(importer) => py
                .detach(|| importer.append(key, offset, data))
                .map_err(error)?,
            _ => return Err(error("public source import outside admitted phase")),
        }
        self.owner = Some(owner);
        Ok(())
    }
    #[pyo3(signature=(*args,**kwargs))]
    fn seal_public_source(
        &mut self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let owner = self.take()?;
        arguments(args, kwargs, [])?;
        let Owner {
            state,
            program,
            session,
            party,
            ..
        } = owner;
        let importer = match state {
            State::SourceRequest(importer) => {
                let request = py.detach(|| importer.finish()).map_err(error)?;
                let program = Arc::clone(request.program());
                let setup = py.detach(|| program.setup(session, party)).map_err(error)?;
                self.owner = Some(Owner {
                    state: State::SourceSetup { setup, request },
                    program: Some(program),
                    session,
                    party,
                    public: None,
                });
                return Ok(());
            }
            State::PublicSource(importer) => importer,
            _ => return Err(error("public source import outside admitted phase")),
        };
        let program = program.ok_or_else(|| error("missing original program"))?;
        let (public, setup) = py
            .detach(|| -> Result<_, String> {
                let public = importer.finish()?;
                let setup = program.setup(session, party).map_err(|e| e.to_string())?;
                Ok((public, setup))
            })
            .map_err(error)?;
        self.owner = Some(Owner {
            state: State::Setup(setup),
            program: Some(program),
            session,
            party,
            public: Some(public),
        });
        Ok(())
    }
    #[pyo3(signature=(*args,**kwargs))]
    fn call(
        &mut self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PeerBatch> {
        let mut owner = self.take()?;
        let [action, data, sender] = arguments(args, kwargs, ["action", "data", "sender"])?;
        let action: &str = action.extract()?;
        let data = bytes(&data, (64 << 10) + 1024)?;
        let sender = party(integer(&sender)?)?;
        let (complete, frames) = py
            .detach(|| -> Result<_, String> {
                let mut frames: Vec<(usize, Zeroizing<Vec<u8>>)> = vec![];
                match (&mut owner.state, action) {
                    (State::Setup(setup) | State::SourceSetup { setup, .. }, "offer")
                        if data.is_empty() =>
                    {
                        let frame = setup.contribution().map_err(|e| e.to_string())?.encode();
                        frames.push((owner.party as usize, Zeroizing::new(frame.to_vec())));
                    }
                    (State::Setup(setup) | State::SourceSetup { setup, .. }, "public") => setup
                        .receive_public(
                            prss::PublicContribution::decode(data, sender)
                                .map_err(|e| e.to_string())?,
                        )
                        .map_err(|e| e.to_string())?,
                    (State::Setup(setup) | State::SourceSetup { setup, .. }, "seed")
                        if data.is_empty() =>
                    {
                        let message = setup.share_seed().map_err(|e| e.to_string())?;
                        let to = message.recipient();
                        let encoded = message.encode().map_err(|e| e.to_string())?;
                        frames.push((to as usize, Zeroizing::new(encoded.as_bytes().to_vec())));
                    }
                    (State::Setup(setup) | State::SourceSetup { setup, .. }, "receive_seed") => {
                        setup
                            .receive_seed(
                                prss::SeedMessage::decode(data, sender, owner.party)
                                    .map_err(|e| e.to_string())?,
                            )
                            .map_err(|e| e.to_string())?
                    }
                    (State::Run(worker), "advance") if data.is_empty() => {
                        for (to, frame) in worker.advance().map_err(|e| e.to_string())? {
                            frames.push((to as usize, Zeroizing::new(frame.as_bytes().to_vec())));
                        }
                    }
                    (State::Run(worker), "peer") => {
                        for (to, frame) in
                            worker.receive(data, sender).map_err(|e| e.to_string())?
                        {
                            frames.push((to as usize, Zeroizing::new(frame.as_bytes().to_vec())));
                        }
                    }
                    (State::Run(worker), "import") => {
                        if data.len() < 10 {
                            return Err("replicated import header".into());
                        }
                        let k = u16::from_le_bytes(data[..2].try_into().unwrap()) as usize;
                        if k > 512 || data.len() < 10 + k {
                            return Err("replicated import key".into());
                        }
                        let key = std::str::from_utf8(&data[2..2 + k])
                            .map_err(|_| "replicated import key")?;
                        let offset =
                            u32::from_le_bytes(data[2 + k..6 + k].try_into().unwrap()) as usize;
                        let n =
                            u32::from_le_bytes(data[6 + k..10 + k].try_into().unwrap()) as usize;
                        if n == 0 || n > 4096 || data.len() != 10 + k + 16 * n {
                            return Err("replicated import size".into());
                        }
                        let mut a = Zeroizing::new(Vec::with_capacity(n));
                        let mut b = Zeroizing::new(Vec::with_capacity(n));
                        for v in data[10 + k..].chunks_exact(16) {
                            a.push(u64::from_le_bytes(v[..8].try_into().unwrap()));
                            b.push(u64::from_le_bytes(v[8..].try_into().unwrap()));
                        }
                        worker
                            .import_chunk(key, offset, &a, &b)
                            .map_err(|e| e.to_string())?;
                    }
                    _ => return Err("replicated call outside admitted phase".into()),
                }
                let complete = matches!(&owner.state,State::Run(worker) if worker.complete());
                Ok((complete, frames))
            })
            .map_err(error)?;
        let result = (
            complete,
            frames
                .iter()
                .map(|(to, frame)| (*to, PyBytes::new(py, frame).unbind()))
                .collect(),
        );
        self.owner = Some(owner);
        Ok(result)
    }
    #[pyo3(signature=(*args,**kwargs))]
    fn activate(
        &mut self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let owner = self.take()?;
        arguments(args, kwargs, [])?;
        let Owner {
            state,
            program,
            session,
            party,
            public,
        } = owner;
        let program = program.ok_or_else(|| error("missing original program"))?;
        let worker = py
            .detach(|| match state {
                State::Setup(mut setup) => {
                    Worker::new_with_source(Arc::clone(&program), setup.finish()?, session, public)
                }
                State::SourceSetup { mut setup, request } => {
                    request.worker(setup.finish()?, session)
                }
                _ => Err(pllm_core::puma::Error::Retired),
            })
            .map_err(error)?;
        self.owner = Some(Owner {
            state: State::Run(worker),
            program: Some(program),
            session,
            party,
            public: None,
        });
        Ok(())
    }
    #[pyo3(signature=(*args,**kwargs))]
    fn finish(
        &mut self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyBytes>> {
        let owner = self.take()?;
        arguments(args, kwargs, [])?;
        let State::Run(worker) = owner.state else {
            return Err(error("replicated worker not active"));
        };
        let result = py.detach(|| worker.finish_frame()).map_err(error)?;
        Ok(PyBytes::new(py, &result).unbind())
    }
    #[pyo3(signature=(*args,**kwargs))]
    fn close(
        &mut self,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        self.owner = None;
        arguments(args, kwargs, [])?;
        Ok(())
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add(
        "NumericDomainRejected",
        module.py().get_type::<NumericDomainRejected>(),
    )?;
    module.add_class::<NativePublisher>()?;
    module.add_class::<NativeWorker>()?;
    module.add_function(wrap_pyfunction!(replicated_source_contract, module)?)?;
    Ok(())
}

#[cfg(test)]
mod quality_rejection_tests {
    use super::*;

    #[test]
    fn numeric_rejection_type_is_not_generic_error_text() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "_quality_native_test").unwrap();
            register(&module).unwrap();
            let kind = module.getattr("NumericDomainRejected").unwrap();
            let rejected =
                NumericDomainRejected::new_err("replicated request failed protected range checks");
            assert!(rejected.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            assert!(rejected.is_instance(py, &kind));
            let malformed = error("replicated request failed protected range checks");
            assert!(!malformed.is_instance(py, &kind));
        });
    }
}

#[cfg(test)]
#[path = "replicated_source_entry_tests.rs"]
mod source_entry_tests;

#[cfg(test)]
#[path = "replicated_compact_entry_tests.rs"]
mod compact_entry_tests;

#[pyfunction]
fn replicated_source_contract(
    py: Python<'_>,
    experiment: &Bound<'_, PyBytes>,
    config: &Bound<'_, PyBytes>,
    rows: usize,
    outputs: usize,
    source: &str,
    tokenizer: &str,
) -> PyResult<Py<PyBytes>> {
    let experiment = experiment.as_bytes();
    let config = config.as_bytes();
    let bytes = py
        .detach(|| {
            pllm_compiler::replicated_decoder::sdk::source_contract(
                experiment, config, rows, outputs, source, tokenizer,
            )
        })
        .map_err(error)?;
    Ok(PyBytes::new(py, &bytes).unbind())
}
