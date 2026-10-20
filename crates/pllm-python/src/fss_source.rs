//! Public-source FSS bindings. Bounded immutable bytes cross the interpreter
//! boundary; native source assembly and numeric admission run without its lock.
use pllm_compiler::fss_decoder::{
    channel::Manifest, import::SourceBuilder, sdk, Capacity, Program, WeightDtype,
};
use pyo3::{
    prelude::*,
    types::{PyAny, PyBool, PyBytes, PyDict, PyList, PyTuple},
};
use serde_json::json;
use std::sync::Arc;

fn invalid(message: impl ToString) -> PyErr {
    crate::invalid(message.to_string())
}

fn native_error(error: pllm_compiler::fss_decoder::Error) -> PyErr {
    let message = format!("FuseFSS native admission failed: {error:?}");
    if error == pllm_compiler::fss_decoder::Error::Domain {
        crate::replicated_transport::NumericDomainRejected::new_err(message)
    } else {
        invalid(message)
    }
}

fn hex(value: [u8; 32]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    value
        .into_iter()
        .flat_map(|byte| {
            [
                DIGITS[(byte >> 4) as usize] as char,
                DIGITS[(byte & 15) as usize] as char,
            ]
        })
        .collect()
}

fn arguments<'py, const N: usize>(
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
    names: [&str; N],
) -> PyResult<[Bound<'py, PyAny>; N]> {
    crate::owned_arguments::arguments(args, kwargs, names, "FuseFSS")
}

fn bytes<'a>(value: &'a Bound<'_, PyAny>, limit: usize) -> PyResult<&'a [u8]> {
    let data = value.cast_exact::<PyBytes>()?.as_bytes();
    if data.len() > limit {
        return Err(invalid("FuseFSS public frame exceeds native bound"));
    }
    Ok(data)
}

fn integer(value: &Bound<'_, PyAny>) -> PyResult<usize> {
    if value.is_instance_of::<PyBool>() {
        return Err(invalid("FuseFSS integer cannot be bool"));
    }
    value.extract()
}

#[pyclass(module = "pllm._native", name = "FssSourceBuilder")]
pub struct NativeSourceBuilder {
    owner: Option<SourceBuilder>,
    pid: u32,
}

impl NativeSourceBuilder {
    fn take(&mut self) -> PyResult<SourceBuilder> {
        let owner = self
            .owner
            .take()
            .ok_or_else(|| invalid("FuseFSS source builder retired"))?;
        if self.pid != std::process::id() {
            return Err(invalid("FuseFSS source owner cannot cross fork"));
        }
        Ok(owner)
    }
}

#[pymethods]
impl NativeSourceBuilder {
    #[new]
    #[pyo3(signature=(config, rows, outputs, numeric_profile, *, capacity="capsule"))]
    fn new(
        py: Python<'_>,
        config: &Bound<'_, PyBytes>,
        rows: &Bound<'_, PyAny>,
        outputs: &Bound<'_, PyAny>,
        numeric_profile: &str,
        capacity: &str,
    ) -> PyResult<Self> {
        let config = config.as_bytes();
        let rows = integer(rows)?;
        let outputs = integer(outputs)?;
        let capacity = match capacity {
            "capsule" => Capacity::Capsule,
            "decoder" => Capacity::Decoder,
            _ => {
                return Err(invalid(
                    "FuseFSS source capacity must be capsule or decoder",
                ))
            }
        };
        let owner = py
            .detach(|| {
                SourceBuilder::new_with_capacity(config, rows, outputs, numeric_profile, capacity)
            })
            .map_err(invalid)?;
        Ok(Self {
            owner: Some(owner),
            pid: std::process::id(),
        })
    }

    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        if self.pid != std::process::id() {
            return Err(invalid("FuseFSS source owner cannot cross fork"));
        }
        let admitted = self
            .owner
            .as_ref()
            .ok_or_else(|| invalid("FuseFSS source builder retired"))?
            .admitted()
            .map_err(native_error)?;
        let value = json!({
            "schema":"pllm.fss.import_geometry.v1",
            "capacity":admitted.capacity.name(),
            "source_q12_value_bytes":admitted.source_q12_value_bytes,
            "source_disk_bytes":admitted.source_disk_bytes,
            "source_resident_bound_bytes":admitted.source_resident_bound_bytes,
            "numeric_profile":admitted.numeric_profile,
            "source_shapes":admitted.source_shapes,
            "rotary_shapes":admitted.rotary_shapes,
        });
        let output =
            sdk::bounded_value_canonical(&value, sdk::MAX_CONTRACT_BYTES).map_err(invalid)?;
        Ok(PyBytes::new(py, &output))
    }

    #[pyo3(signature=(*args, **kwargs))]
    fn append(
        &mut self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let mut owner = self.take()?;
        let [name, dtype, offset, payload] =
            arguments(args, kwargs, ["name", "dtype", "offset", "payload"])?;
        let name: &str = name.extract()?;
        let dtype = match dtype.extract::<&str>()? {
            "F32" => WeightDtype::F32,
            "F16" => WeightDtype::F16,
            "BF16" => WeightDtype::Bf16,
            _ => return Err(invalid("FuseFSS source dtype must be F32, F16 or BF16")),
        };
        let offset = integer(&offset)?;
        let payload = bytes(
            &payload,
            pllm_compiler::fss_decoder::import::MAX_FRAME_BYTES,
        )?;
        py.detach(|| owner.append(name, dtype, offset, payload))
            .map_err(native_error)?;
        self.owner = Some(owner);
        Ok(())
    }

    #[pyo3(signature=(*args, **kwargs))]
    fn append_rotary(
        &mut self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let mut owner = self.take()?;
        let [operation, offset, cosine, sine] =
            arguments(args, kwargs, ["operation", "offset", "cosine", "sine"])?;
        let operation: &str = operation.extract()?;
        let offset = integer(&offset)?;
        let limit = pllm_compiler::fss_decoder::import::MAX_FRAME_BYTES;
        let cosine = bytes(&cosine, limit)?;
        let sine = bytes(&sine, limit)?;
        py.detach(|| owner.append_rotary(operation, offset, cosine, sine))
            .map_err(native_error)?;
        self.owner = Some(owner);
        Ok(())
    }

    #[pyo3(signature=(*args, **kwargs))]
    fn seal(
        &mut self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<NativeProgram> {
        let owner = self.take()?;
        let [] = arguments(args, kwargs, [])?;
        let imported = py.detach(|| owner.seal()).map_err(native_error)?;
        Ok(NativeProgram {
            program: Some(Arc::new(imported.program)),
            input_digest: imported.input_digest,
            execution_binding: imported.execution_binding,
            ownership: imported.ownership,
            pid: self.pid,
        })
    }

    fn cancel(&mut self) {
        self.owner.take();
    }

    fn close(&mut self) {
        self.cancel();
    }
}

/// Immutable public data only. Every protected request receives a separate
/// consuming native worker and a fresh session through the role binding.
#[pyclass(module = "pllm._native", name = "FssProgram")]
pub struct NativeProgram {
    pub(crate) program: Option<Arc<Program>>,
    pub(crate) input_digest: [u8; 32],
    execution_binding: sdk::SourceExecutionBinding,
    ownership: pllm_compiler::fss_decoder::import::CompileOwnership,
    pub(crate) pid: u32,
}

impl NativeProgram {
    pub(crate) fn checked_contract_digest(&self, contract: &[u8]) -> PyResult<[u8; 32]> {
        self.borrow()?;
        self.execution_binding
            .checked_contract_digest(contract)
            .map_err(invalid)
    }
    pub(crate) fn borrow(&self) -> PyResult<&Arc<Program>> {
        if self.pid != std::process::id() {
            return Err(invalid("FuseFSS source owner cannot cross fork"));
        }
        self.program
            .as_ref()
            .ok_or_else(|| invalid("FuseFSS source program retired"))
    }
}

#[pymethods]
impl NativeProgram {
    /// Local immutable-file observations; never part of source/program identity
    /// or a peer-supplied admission quote. The borrowed program retains the files.
    fn disk_backing<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let program = self.borrow()?;
        let backing = py.detach(|| program.disk_backing()).map_err(native_error)?;
        let result = PyList::empty(py);
        for item in backing {
            let record = PyDict::new(py);
            record.set_item("device", item.device)?;
            record.set_item("file", item.file)?;
            record.set_item("allocated_bytes", item.allocated_bytes)?;
            result.append(record)?;
        }
        Ok(result)
    }

    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let program = self.borrow()?;
        let r = program.resources();
        let value = json!({
            "schema":"pllm.fss.program_resources.v1",
            "capacity":program.capacity().name(),
            "source_disk_bytes":program.source_disk_bytes(),
            "program_digest":hex(program.digest()),
            "source_digest":hex(program.source_digest()),
            "original_input_digest":hex(self.input_digest),
            "numeric_profile":program.profile(),
            "initial_tokens":program.initial_tokens(),
            "selections":program.selections(),
            "vocabulary":program.vocabulary(),
            "kernels":r.kernels,
            "primitive_requests_per_party":r.primitive_requests_per_party,
            "work":r.work,
            "material_bytes":r.material_bytes,
            "frame_bytes":r.frame_bytes,
            "public_copy_bytes":r.public_copy_bytes,
            "metadata_bytes":r.metadata_bytes,
            "live_array_bytes_per_party":r.live_array_bytes_per_party,
            "peak_native_bytes":r.peak_native_bytes,
            "source_table_cache_bytes_per_role":r.source_table_cache_bytes_per_role,
            "combined_reservation_bytes":r.combined_reservation_bytes,
            "scope":"bounded-native-program-before-role-transport",
            "source_phase_ownership":{
                "plan_owned_bound":self.ownership.phases.plan_owned_bound,
                "inventory_owned_bound":self.ownership.phases.inventory_owned_bound,
                "serialization_peak_bound":self.ownership.phases.serialization_peak_bound,
                "ingestion_peak_bound":self.ownership.phases.ingestion_peak_bound,
                "compile_prepare_peak_bound":self.ownership.phases.compile_prepare_peak_bound,
                "schedule_metadata_credit_floor":self.ownership.schedule_metadata_credit_floor,
                "schedule_retained_owner_bound":self.ownership.schedule_retained_owner_bound,
                "basis":"partition_of_existing_three_metadata_copy_allowance_not_additive_reservation",
                "scope":"logical_native_import_compile_not_global_allocator_or_python_http_rss"
            },
        });
        let output =
            sdk::bounded_value_canonical(&value, sdk::MAX_CONTRACT_BYTES).map_err(invalid)?;
        Ok(PyBytes::new(py, &output))
    }

    fn manifest<'py>(
        &self,
        py: Python<'py>,
        contract: &Bound<'py, PyBytes>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let contract = contract.as_bytes();
        let digest = self.checked_contract_digest(contract)?;
        let manifest =
            Manifest::new(digest, self.input_digest, self.borrow()?).map_err(native_error)?;
        Ok(PyBytes::new(py, manifest.as_bytes()))
    }

    fn close(&mut self) {
        self.program.take();
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn fss_source_contract<'py>(
    py: Python<'py>,
    experiment: &Bound<'py, PyBytes>,
    config: &Bound<'py, PyBytes>,
    rows: &Bound<'py, PyAny>,
    outputs: &Bound<'py, PyAny>,
    sourcepin: &str,
    tokenizerpin: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let experiment = experiment.as_bytes();
    let config = config.as_bytes();
    let rows = integer(rows)?;
    let outputs = integer(outputs)?;
    let result = py
        .detach(|| sdk::source_contract(experiment, config, rows, outputs, sourcepin, tokenizerpin))
        .map_err(invalid)?;
    Ok(PyBytes::new(py, &result))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeSourceBuilder>()?;
    module.add_class::<NativeProgram>()?;
    module.add_function(wrap_pyfunction!(fss_source_contract, module)?)?;
    Ok(())
}
