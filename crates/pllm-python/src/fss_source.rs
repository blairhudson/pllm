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
        if self.owner.is_none() {
            return Err(invalid("FuseFSS source builder retired"));
        }
        if self.pid != std::process::id() {
            return Err(invalid("FuseFSS source owner cannot cross fork"));
        }
        self.owner
            .take()
            .ok_or_else(|| invalid("FuseFSS source builder retired"))
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

    fn seal_wavefront(&mut self, py: Python<'_>) -> PyResult<NativeProgram> {
        let owner = self.take()?;
        let imported = py
            .detach(|| {
                owner.seal_with_schedule(pllm_compiler::fss_decoder::OnlineSchedule::WavefrontV1)
            })
            .map_err(native_error)?;
        Ok(NativeProgram {
            program: Some(Arc::new(imported.program)),
            input_digest: imported.input_digest,
            execution_binding: imported.execution_binding,
            ownership: imported.ownership,
            pid: self.pid,
        })
    }

    fn seal_tensor(&mut self, py: Python<'_>) -> PyResult<NativeProgram> {
        let owner = self.take()?;
        let imported = py
            .detach(|| {
                owner.seal_with_schedule(pllm_compiler::fss_decoder::OnlineSchedule::TensorV1)
            })
            .map_err(native_error)?;
        Ok(NativeProgram {
            program: Some(Arc::new(imported.program)),
            input_digest: imported.input_digest,
            execution_binding: imported.execution_binding,
            ownership: imported.ownership,
            pid: self.pid,
        })
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
        if self.borrow()?.online_schedule()
            != sdk::contract_online_schedule(contract).map_err(invalid)?
        {
            return Err(invalid(
                "FuseFSS source online schedule differs from contract",
            ));
        }
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
            "online_schedule":program.online_schedule().name(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::exceptions::PyValueError;

    fn builder(pid: u32) -> NativeSourceBuilder {
        // Same tiny real-constructor geometry as fss_decoder/import_tests.rs.
        // No checkpoint payload, completed import, or cleanup proof is minted.
        let config = pllm_types::canonical_bytes(&json!({
            "model_type":"gpt2", "n_embd":2, "n_head":1, "n_layer":1,
            "n_positions":4, "vocab_size":3, "n_inner":2
        }));
        let owner = SourceBuilder::new_with_capacity(
            &config,
            1,
            2,
            sdk::GPT2_NUMERIC,
            Capacity::Capsule,
        )
        .unwrap();
        NativeSourceBuilder {
            owner: Some(owner),
            pid,
        }
    }

    fn shape_allocation(owner: &SourceBuilder) -> *const usize {
        let shape = owner
            .admitted()
            .unwrap()
            .source_shapes
            .values()
            .next()
            .unwrap();
        assert!(!shape.is_empty());
        shape.as_ptr()
    }

    fn assert_take_error(py: Python<'_>, builder: &mut NativeSourceBuilder, message: &str) {
        let error = match builder.take() {
            Err(error) => error,
            Ok(_) => panic!("source owner unexpectedly transferred"),
        };
        assert!(error.is_instance_of::<PyValueError>(py));
        assert_eq!(error.value(py).str().unwrap().to_str().unwrap(), message);
    }

    #[test]
    fn take_empty_same_pid_is_retired() {
        Python::initialize();
        Python::attach(|py| {
            let mut builder = NativeSourceBuilder {
                owner: None,
                pid: std::process::id(),
            };
            assert_take_error(py, &mut builder, "FuseFSS source builder retired");
            assert!(builder.owner.is_none());
        });
    }

    #[test]
    fn take_empty_wrong_pid_is_still_retired() {
        Python::initialize();
        Python::attach(|py| {
            let mut builder = NativeSourceBuilder {
                owner: None,
                pid: std::process::id().wrapping_add(1),
            };
            assert_take_error(py, &mut builder, "FuseFSS source builder retired");
            assert!(builder.owner.is_none());
        });
    }

    #[test]
    fn take_live_wrong_pid_retains_owner_then_transfers_once() {
        Python::initialize();
        Python::attach(|py| {
            let pid = std::process::id();
            let mut builder = builder(pid.wrapping_add(1));
            let owner = builder.owner.as_ref().unwrap();
            let address = std::ptr::from_ref(owner);
            let allocation = shape_allocation(owner);
            for _ in 0..2 {
                assert_take_error(py, &mut builder, "FuseFSS source owner cannot cross fork");
                let retained = builder.owner.as_ref().unwrap();
                assert_eq!(std::ptr::from_ref(retained), address);
                assert_eq!(shape_allocation(retained), allocation);
            }
            // Only the stored PID changes; no real fork or native cleanup probe.
            builder.pid = pid;
            let transferred = builder.take().unwrap();
            assert!(builder.owner.is_none());
            assert_eq!(shape_allocation(&transferred), allocation);
            assert_take_error(py, &mut builder, "FuseFSS source builder retired");
            assert_eq!(shape_allocation(&transferred), allocation);
        });
    }

    #[test]
    fn take_live_same_pid_transfers_once() {
        Python::initialize();
        Python::attach(|py| {
            let mut builder = builder(std::process::id());
            let allocation = shape_allocation(builder.owner.as_ref().unwrap());
            let transferred = builder.take().unwrap();
            assert!(builder.owner.is_none());
            assert_eq!(shape_allocation(&transferred), allocation);
            assert_take_error(py, &mut builder, "FuseFSS source builder retired");
            assert_eq!(shape_allocation(&transferred), allocation);
        });
    }
}
