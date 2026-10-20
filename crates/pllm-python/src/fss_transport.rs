//! One consuming native role, not an HTTP implementation or authentication layer.
//! Session/epoch replay registries and the offline sealed-tape barrier are SDK
//! duties; constructing a new native owner does not make a reused session fresh.
//! Callback metadata (canonical JSON, <=1024 bytes):
//! {schema:"pllm.fss.callback.v1",kind:"material"|"peer",kernel:{context:{
//! plan,session,operation,first_tensor_index},plan_digest,ordinal},role,
//! pc,packet_bytes_bound} or {..,frame,frame_bytes}. Digests are lowercase hex.
//! Dealer role is the packet recipient. Holder replies must be exact bytes;
//! deliver/finish return None. cancel must wake blocked callbacks and erase/drop
//! retained queues, never join its own run thread. The caller joins/drains run
//! before refunding reservations, including all immutable Python byte copies.
use crate::fss_source::NativeProgram;
use pllm_compiler::fss_decoder::{
    channel::{self, Manifest, Request},
    sdk,
    transport::{
        self, Admission, DealerTransport, Kernel, MaterialRecord, MaterialRequest, PeerRequest,
        Reply, WorkerTransport, CALLBACK_METADATA_BYTES_BOUND,
    },
    Error, Program,
};
use pllm_types::canonical_bytes;
use pyo3::{
    prelude::*,
    types::{PyAny, PyBool, PyBytes, PyDict, PyInt, PyTuple},
};
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Condvar, Mutex,
};

fn invalid() -> PyErr {
    crate::invalid("FuseFSS role request rejected".to_owned())
}
fn arguments<'py, const N: usize>(
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
    names: [&str; N],
) -> PyResult<[Bound<'py, PyAny>; N]> {
    crate::owned_arguments::arguments(args, kwargs, names, "FuseFSS role")
}
fn bytes<'a>(value: &'a Bound<'_, PyAny>, maximum: usize) -> PyResult<&'a [u8]> {
    let bytes = value.cast_exact::<PyBytes>()?.as_bytes();
    if bytes.len() > maximum {
        return Err(invalid());
    }
    Ok(bytes)
}
fn hex(value: [u8; 32]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    value
        .into_iter()
        .flat_map(|v| {
            [
                DIGITS[(v >> 4) as usize] as char,
                DIGITS[(v & 15) as usize] as char,
            ]
        })
        .collect()
}
fn context(context: pllm_garble::shared_rescale_reference::Context) -> Value {
    json!({"plan":hex(context.plan),"session":hex(context.session),"operation":hex(context.operation),"first_tensor_index":context.first_tensor_index})
}
fn kernel(k: Kernel) -> Value {
    json!({"context":context(k.context),"plan_digest":hex(k.plan_digest),"ordinal":k.ordinal})
}
fn material_meta(request: MaterialRequest) -> Vec<u8> {
    canonical_bytes(
        &json!({"schema":"pllm.fss.callback.v1","kind":"material","kernel":kernel(request.kernel),"role":request.role,"pc":request.pc,"packet_bytes_bound":request.packet_bytes_bound}),
    )
}
fn peer_meta(request: PeerRequest) -> Vec<u8> {
    canonical_bytes(
        &json!({"schema":"pllm.fss.callback.v1","kind":"peer","kernel":kernel(request.kernel),"role":request.role,"frame":request.frame,"frame_bytes":request.frame_bytes}),
    )
}

fn delivery_metadata(value: &mut Value, admission: &Admission) {
    if let Some(batch) = admission.delivery_batch() {
        value["material_delivery"] = json!(admission.material_delivery().name());
        value["delivery_batch"] = json!({
            "records_bound": batch.records_bound(),
            "body_bytes_bound": batch.body_bytes_bound(),
            "envelope_bytes_bound": batch.envelope_bytes_bound(),
            "record_overhead_bytes": batch.record_overhead_bytes(),
            "native_reservation_bytes": batch.native_reservation_bytes(),
            "python_overlap_bytes": batch.python_overlap_bytes(),
        });
    }
}

#[derive(Default)]
struct Control {
    cancelled: AtomicBool,
    notified: AtomicBool,
    cancel_failed: AtomicBool,
    active: Mutex<Option<Py<PyAny>>>,
    cancel_done: Mutex<bool>,
    drained: Condvar,
}
impl Control {
    fn check(&self) -> Result<(), Error> {
        if self.cancelled.load(Ordering::Acquire) {
            Err(Error::Consumed)
        } else {
            Ok(())
        }
    }
    fn abort(&self, py: Python<'_>) {
        self.cancelled.store(true, Ordering::Release);
        let callback = {
            let active = self.active.lock().unwrap_or_else(|e| e.into_inner());
            active.as_ref().and_then(|v| {
                if self.notified.swap(true, Ordering::AcqRel) {
                    None
                } else {
                    Some(v.clone_ref(py))
                }
            })
        };
        if let Some(callback) = callback {
            // Never expose callback repr, exception text or traceback.
            let failed = match callback.bind(py).call_method0("cancel") {
                Ok(value) => !value.is_none(),
                Err(_) => true,
            };
            self.cancel_failed.store(failed, Ordering::Release);
            *self.cancel_done.lock().unwrap_or_else(|e| e.into_inner()) = true;
            self.drained.notify_all();
        }
    }
    fn clear(&self) {
        let callback = self.active.lock().unwrap_or_else(|e| e.into_inner()).take();
        drop(callback);
    }
    fn wait_cancel(&self) {
        if self.notified.load(Ordering::Acquire) {
            let mut done = self.cancel_done.lock().unwrap_or_else(|e| e.into_inner());
            while !*done {
                done = self.drained.wait(done).unwrap_or_else(|e| e.into_inner());
            }
        }
    }
}
struct Bridge {
    callback: Py<PyAny>,
    control: Arc<Control>,
}
impl Bridge {
    fn notification(&self, name: &str) -> Result<(), Error> {
        self.control.check()?;
        Python::attach(|py| {
            self.control.check()?;
            let result = self
                .callback
                .bind(py)
                .call_method0(name)
                .map_err(|_| Error::Context)?;
            if !result.is_none() {
                return Err(Error::Context);
            }
            self.control.check()
        })
    }
    fn abort(&self) {
        Python::attach(|py| self.control.abort(py));
    }
}
impl DealerTransport for Bridge {
    fn check_active(&mut self) -> Result<(), Error> {
        self.control.check()
    }
    fn deliver(&mut self, request: MaterialRequest, packet: &[u8]) -> Result<(), Error> {
        self.control.check()?;
        if packet.is_empty() || packet.len() > request.packet_bytes_bound {
            return Err(Error::Resource);
        }
        let meta = material_meta(request);
        if meta.len() > CALLBACK_METADATA_BYTES_BOUND {
            return Err(Error::Resource);
        }
        Python::attach(|py| {
            self.control.check()?;
            let result = self
                .callback
                .bind(py)
                .call_method1(
                    "deliver",
                    (PyBytes::new(py, &meta), PyBytes::new(py, packet)),
                )
                .map_err(|_| Error::Context)?;
            if !result.is_none() {
                return Err(Error::Context);
            }
            self.control.check()
        })
    }
    fn deliver_batch(&mut self, role: u8, records: &[MaterialRecord<'_>]) -> Result<(), Error> {
        self.control.check()?;
        if role > 1 || records.is_empty() || records.len() > transport::MATERIAL_BATCH_RECORDS_BOUND
        {
            return Err(Error::Resource);
        }
        Python::attach(|py| {
            self.control.check()?;
            let mut pairs = Vec::new();
            pairs
                .try_reserve_exact(records.len())
                .map_err(|_| Error::Allocation)?;
            let mut body_bytes = transport::MATERIAL_BATCH_ENVELOPE_BYTES_BOUND;
            for record in records {
                let request = record.request();
                let packet = record.packet();
                if request.role != role
                    || packet.is_empty()
                    || packet.len() > request.packet_bytes_bound
                {
                    return Err(Error::Context);
                }
                let meta = material_meta(request);
                if meta.len() > CALLBACK_METADATA_BYTES_BOUND {
                    return Err(Error::Resource);
                }
                body_bytes = body_bytes
                    .checked_add(transport::MATERIAL_BATCH_RECORD_OVERHEAD_BYTES)
                    .and_then(|n| n.checked_add(meta.len()))
                    .and_then(|n| n.checked_add(packet.len()))
                    .filter(|&n| n <= transport::MATERIAL_BATCH_BODY_BYTES_BOUND)
                    .ok_or(Error::Resource)?;
                pairs.push(
                    PyTuple::new(py, [PyBytes::new(py, &meta), PyBytes::new(py, packet)])
                        .map_err(|_| Error::Allocation)?,
                );
            }
            let records = PyTuple::new(py, pairs).map_err(|_| Error::Allocation)?;
            let result = self
                .callback
                .bind(py)
                .call_method1("deliver_batch", (role, records))
                .map_err(|_| Error::Context)?;
            if !result.is_none() {
                return Err(Error::Context);
            }
            self.control.check()
        })
    }
    fn finish(&mut self) -> Result<(), Error> {
        self.notification("finish")
    }
    fn cancel(&mut self) {
        self.abort();
    }
}
impl WorkerTransport for Bridge {
    fn material(&mut self, request: MaterialRequest, reply: &mut Reply) -> Result<(), Error> {
        self.control.check()?;
        let meta = material_meta(request);
        if meta.len() > 1024 {
            return Err(Error::Resource);
        }
        Python::attach(|py| {
            self.control.check()?;
            let result = self
                .callback
                .bind(py)
                .call_method1("material", (PyBytes::new(py, &meta),))
                .map_err(|_| Error::Context)?;
            let result = result
                .cast_exact::<PyBytes>()
                .map_err(|_| Error::Context)?
                .as_bytes();
            if result.is_empty()
                || result.len() > request.packet_bytes_bound
                || result.len() > reply.bound()
            {
                return Err(Error::Resource);
            }
            self.control.check()?;
            reply.write(result)
        })
    }
    fn exchange(
        &mut self,
        request: PeerRequest,
        frame: &[u8],
        reply: &mut Reply,
    ) -> Result<(), Error> {
        self.control.check()?;
        if frame.len() != request.frame_bytes {
            return Err(Error::Context);
        }
        let meta = peer_meta(request);
        if meta.len() > 1024 {
            return Err(Error::Resource);
        }
        Python::attach(|py| {
            self.control.check()?;
            let result = self
                .callback
                .bind(py)
                .call_method1(
                    "exchange",
                    (PyBytes::new(py, &meta), PyBytes::new(py, frame)),
                )
                .map_err(|_| Error::Context)?;
            let result = result
                .cast_exact::<PyBytes>()
                .map_err(|_| Error::Context)?
                .as_bytes();
            if result.len() != request.frame_bytes || result.len() > reply.bound() {
                return Err(Error::Context);
            }
            self.control.check()?;
            reply.write(result)
        })
    }
    fn finish(&mut self) -> Result<(), Error> {
        self.notification("finish")
    }
    fn cancel(&mut self) {
        self.abort();
    }
}
struct Owner {
    program: Arc<Program>,
    request: Request,
    role: u8,
    admission: Admission,
    channel: channel::Resources,
}

#[pyclass(module = "pllm._native", name = "FssRoleRequest")]
pub struct NativeRoleRequest {
    owner: Mutex<Option<Owner>>,
    control: Arc<Control>,
    pid: u32,
}
impl NativeRoleRequest {
    fn abort(&self, py: Python<'_>) {
        if self.pid != std::process::id() {
            self.control.cancelled.store(true, Ordering::Release);
            // Never wait on a mutex inherited from a vanished native thread,
            // or invoke an inherited Python transport/event loop after fork.
            if let Ok(mut owner) = self.owner.try_lock() {
                drop(owner.take());
            }
            return;
        }
        drop(self.owner.lock().unwrap_or_else(|e| e.into_inner()).take());
        self.control.abort(py);
    }
    fn take(&self, py: Python<'_>) -> PyResult<Owner> {
        if self.pid != std::process::id() {
            self.abort(py);
            return Err(invalid());
        }
        let owner = self.owner.lock().unwrap_or_else(|e| e.into_inner()).take();
        let Some(owner) = owner else {
            self.abort(py);
            return Err(invalid());
        };
        if self.pid != std::process::id() || self.control.check().is_err() {
            self.abort(py);
            return Err(invalid());
        }
        Ok(owner)
    }
}
#[pymethods]
impl NativeRoleRequest {
    #[new]
    #[pyo3(signature=(*args, **kwargs))]
    fn new(
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let [program, contract, request, role] =
            arguments(args, kwargs, ["program", "contract", "request", "role"])?;
        let program = program.extract::<PyRef<'_, NativeProgram>>()?;
        let contract = bytes(&contract, 1 << 20)?;
        let contract_digest = program.checked_contract_digest(contract)?;
        let delivery = sdk::contract_material_delivery(contract).map_err(|_| invalid())?;
        let source = Arc::clone(program.borrow()?);
        let input_digest = program.input_digest;
        drop(program);
        let request = bytes(&request, channel::REQUEST_BYTES)?;
        if role.is_instance_of::<PyBool>() {
            return Err(invalid());
        }
        let role = role.cast_exact::<PyInt>()?.extract::<u8>()?;
        if role > 2 {
            return Err(invalid());
        }
        let owner = py
            .detach(|| {
                let manifest = Manifest::new(contract_digest, input_digest, &source)?;
                let request = Request::parse(request, &manifest)?;
                let admission =
                    transport::admit_with_delivery(&source, request.context(), delivery)?;
                Ok::<_, Error>(Owner {
                    program: source,
                    request,
                    role,
                    admission,
                    channel: manifest.resources(),
                })
            })
            .map_err(|_| invalid())?;
        Ok(Self {
            owner: Mutex::new(Some(owner)),
            control: Arc::new(Control::default()),
            pid: std::process::id(),
        })
    }

    /// Read before run; public admission is native-derived, never caller supplied.
    #[pyo3(signature=(*args, **kwargs))]
    fn metadata<'py>(
        &self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let result = (|| {
            if self.pid != std::process::id() {
                return Err(invalid());
            }
            let [] = arguments(args, kwargs, [])?;
            let owner = self.owner.lock().unwrap_or_else(|e| e.into_inner());
            let owner = owner.as_ref().ok_or_else(invalid)?;
            let a = owner.admission;
            let c = owner.channel;
            let mut value = json!({"schema":"pllm.fss.role_admission.v1","role":owner.role,"context":context(owner.request.context()),
                "capacity":owner.program.capacity().name(),"guard_disk_bytes":a.guard_disk_bytes(),
                "kernels":a.kernels(),"primitive_requests_per_party":a.primitive_requests_per_party(),"frames_per_party":a.frames_per_party(),
                "largest_packet_bytes":a.largest_packet_bytes(),"cumulative_packet_bytes_per_party_bound":a.cumulative_packet_bytes_per_party_bound(),
                "guard_capacity":a.guard_capacity(),"guard_reservation_bytes":a.guard_reservation_bytes(),"combined_reservation_bytes":a.combined_reservation_bytes(),"transcript_digest":hex(a.transcript_digest()),
                "peer_frame_bytes_bound":a.peer_frame_bytes_bound(),
                "callback_metadata_bytes_bound":1024,"request_bytes":c.request_bytes,"input_frame_bytes":c.input_frame_bytes,"output_frame_bytes":c.output_frame_bytes,
                "client_peak_bytes":c.client_peak_bytes,"worker_input_import_bytes":c.worker_input_import_bytes,"worker_output_export_bytes":c.worker_output_export_bytes});
            delivery_metadata(&mut value, &a);
            Ok(PyBytes::new(py, &canonical_bytes(&value)))
        })();
        if result.is_err() {
            self.abort(py);
        }
        result
    }

    #[pyo3(signature=(*args, **kwargs))]
    fn run(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Option<Py<PyBytes>>> {
        // No exclusive Python borrow or owner mutex is held across callbacks.
        let mut owner = self.take(py)?;
        let result = (|| {
            let [input_frame, callback] = arguments(args, kwargs, ["input_frame", "transport"])?;
            let callback = callback.unbind();
            *self
                .control
                .active
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(callback.clone_ref(py));
            let mut bridge = Bridge {
                callback,
                control: Arc::clone(&self.control),
            };
            let input = if owner.role == 2 {
                if !input_frame.is_none() {
                    return Err(invalid());
                }
                None
            } else {
                Some(bytes(&input_frame, channel::MAX_INPUT_FRAME_BYTES)?)
            };
            let output = py
                .detach(|| {
                    bridge.control.check()?;
                    if owner.role == 2 {
                        transport::prepare_with_delivery(
                            &owner.program,
                            owner.request.context(),
                            owner.admission.material_delivery(),
                            &mut bridge,
                        )?;
                        Ok(None)
                    } else {
                        let input = owner
                            .request
                            .take_input(owner.role, input.ok_or(Error::Context)?)?;
                        let output = transport::execute_with_delivery(
                            &owner.program,
                            owner.role,
                            owner.request.context(),
                            input,
                            owner.admission.material_delivery(),
                            &mut bridge,
                        )?;
                        bridge.control.check()?;
                        Ok(Some(owner.request.output(owner.role, output)?))
                    }
                })
                .map_err(|_: Error| invalid())?;
            self.control.check().map_err(|_| invalid())?;
            Ok(output.map(|bytes| PyBytes::new(py, &bytes).unbind()))
        })();
        if result.is_err() {
            self.control.abort(py);
        }
        self.control.clear();
        // Unregister first, then wait without the GIL for an external cancel
        // callback that already claimed this transport. No callback outlives run.
        py.detach(|| self.control.wait_cancel());
        if self.control.cancel_failed.load(Ordering::Acquire) {
            return Err(invalid());
        }
        result
    }

    #[pyo3(signature=(*args, **kwargs))]
    fn cancel(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        self.abort(py);
        if self.pid != std::process::id() {
            return Err(invalid());
        }
        let [] = arguments(args, kwargs, [])?;
        if self.control.cancel_failed.load(Ordering::Acquire) {
            return Err(invalid());
        }
        Ok(())
    }
    #[pyo3(signature=(*args, **kwargs))]
    fn close(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        self.cancel(py, args, kwargs)
    }
}
/// Program-only pricing, before accepting token IDs, creating shares, or issuing
/// material. Fixed public context is used for scheduling, not request identity.
#[pyfunction]
#[pyo3(signature=(*args, **kwargs))]
fn fss_transport_admission<'py>(
    py: Python<'py>,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyBytes>> {
    let [program, contract] = arguments(args, kwargs, ["program", "contract"])?;
    let program = program.extract::<PyRef<'_, NativeProgram>>()?;
    let contract = bytes(&contract, 1 << 20)?;
    let digest = program.checked_contract_digest(contract)?;
    let delivery = sdk::contract_material_delivery(contract).map_err(|_| invalid())?;
    let source = Arc::clone(program.borrow()?);
    let input_digest = program.input_digest;
    drop(program);
    let value=py.detach(|| {
        let manifest=Manifest::new(digest,input_digest,&source)?;
        let pricing_context=pllm_garble::shared_rescale_reference::Context {
            plan:source.digest(),session:[1;32],operation:[2;32],first_tensor_index:0 };
        let a=transport::admit_with_delivery(&source,pricing_context,delivery)?;let c=manifest.resources();
        let mut value=json!({"schema":"pllm.fss.program_transport_admission.v1",
            "capacity":source.capacity().name(),"guard_disk_bytes":a.guard_disk_bytes(),
            "scope":"program-pricing-not-request-admission","contract_digest":hex(digest),"program_digest":hex(source.digest()),
            "source_digest":hex(source.source_digest()),"input_digest":hex(input_digest),"numeric_profile":source.profile(),
            "initial_tokens":source.initial_tokens(),"selections":source.selections(),"vocabulary":source.vocabulary(),
            "kernels":a.kernels(),"primitive_requests_per_party":a.primitive_requests_per_party(),"frames_per_party":a.frames_per_party(),
            "largest_packet_bytes":a.largest_packet_bytes(),"cumulative_packet_bytes_per_party_bound":a.cumulative_packet_bytes_per_party_bound(),
            "guard_capacity":a.guard_capacity(),"guard_reservation_bytes":a.guard_reservation_bytes(),"combined_reservation_bytes":a.combined_reservation_bytes(),
            "peer_frame_bytes_bound":a.peer_frame_bytes_bound(),
            "callback_metadata_bytes_bound":1024,"request_bytes":c.request_bytes,"input_frame_bytes":c.input_frame_bytes,"output_frame_bytes":c.output_frame_bytes,
            "client_peak_bytes":c.client_peak_bytes,"worker_input_import_bytes":c.worker_input_import_bytes,"worker_output_export_bytes":c.worker_output_export_bytes,
            "offline_tape_storage_cpu_and_http_charges":"caller-owned-additional"});
        delivery_metadata(&mut value, &a);
        Ok::<_,Error>(value)
    }).map_err(|_|invalid())?;
    Ok(PyBytes::new(py, &canonical_bytes(&value)))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeRoleRequest>()?;
    module.add_function(wrap_pyfunction!(fss_transport_admission, module)?)
}
