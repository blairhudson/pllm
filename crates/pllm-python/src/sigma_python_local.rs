//! TEST HARNESS ONLY. Real Python objects around the existing Rust authority.
//! This module is mounted beneath its genuine pinned-source fixture tests.
//! Known logical controls are reserved; CPython/PyCell/GC/allocator/traceback,
//! arbitrary Python callbacks and lazy type initialization remain UNKNOWN.
//! There is no Python input factory and no call around the production gate.

use crate::sigma_linear_activation as core;
use crate::sigma_linear_activation::tests as fixture;
use core::{BridgeError, LocalStep, OwnedBudget, OwnedExternalCharge, PendingOwners};
use pyo3::{
    exceptions::{PyMemoryError, PyRuntimeError, PyTypeError},
    prelude::*,
    types::{PyAny, PyBool, PyDict, PyModule, PyTuple},
};
use std::{
    alloc::Layout,
    cell::{Cell, RefCell},
    mem::size_of,
    rc::Rc,
};

fn bridge_error(_: BridgeError) -> PyErr {
    // Exception object/traceback allocation is an explicitly UNKNOWN term.
    PyRuntimeError::new_err("local SIGMA operation refused or revoked")
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Fault {
    None,
    TupleMemory,
    TupleUnwind,
    OutputMemory,
}

struct Controls {
    fault: Cell<Fault>,
    // Last: every logical payload above and every wrapper link retires first.
    charge: OwnedExternalCharge,
}

impl Controls {
    fn reserve(root: &OwnedBudget, fault: Fault) -> Result<Rc<Self>, BridgeError> {
        let bytes = KnownControls::checked()?;
        let charge = root
            .charge_external(bytes)
            .map_err(BridgeError::Allocation)?;
        Ok(Rc::new(Self {
            fault: Cell::new(fault),
            charge,
        }))
    }

    fn tuple_boundary(&self) -> PyResult<()> {
        match self.fault.get() {
            Fault::TupleMemory => {
                self.fault.set(Fault::None);
                Err(PyMemoryError::new_err(
                    "injected tuple packaging allocation failure",
                ))
            }
            Fault::TupleUnwind => {
                self.fault.set(Fault::None);
                panic!("injected tuple packaging unwind");
            }
            _ => Ok(()),
        }
    }
}

/// NOT a Python peak quote. Public FFI base headers and concrete Rust controls
/// only; no guessed private PyClassObject/PyCell/GC layout or allocator sizes.
struct KnownControls;
impl KnownControls {
    fn checked() -> Result<usize, BridgeError> {
        let rc = Layout::new::<[Cell<usize>; 2]>()
            .extend(Layout::new::<Controls>())
            .map_err(|_| BridgeError::Resource)?
            .0
            .pad_to_align()
            .size();
        let objects = 1 + 2 + core::MAX_FRAMES + core::MAX_OUTPUTS;
        let tuples = 1 + core::MAX_FRAMES + 2; // creation, frames, both Ready
        let parts = [
            rc,
            size_of::<Envelope>(),
            2 * size_of::<Role>(),
            core::MAX_FRAMES * size_of::<Frame>(),
            core::MAX_OUTPUTS * size_of::<Output>(),
            (objects + tuples) * size_of::<pyo3::ffi::PyObject>(),
            (objects + 3 + 2 * (core::MAX_FRAMES + 2)) * size_of::<Py<PyAny>>(),
            // Two calls may nest. Include parser/borrow/error/packaging slots.
            2 * (size_of::<Option<PyRef<'static, Frame>>>()
                + size_of::<PyRef<'static, Role>>()
                + size_of::<Py<PyAny>>()
                + size_of::<RefCell<Option<PyErr>>>()
                + size_of::<Option<PyErr>>()
                + size_of::<PyErr>()
                + size_of::<Bound<'static, PyTuple>>()
                + size_of::<Result<Py<PyTuple>, PyErr>>()
                + size_of::<[Py<PyAny>; 3]>()
                + size_of::<[Py<PyAny>; 2]>()
                + size_of::<Rc<Controls>>()
                + size_of::<Result<(), PyErr>>()),
            // Reservation/Rc construction and old/new payload transfer slots.
            size_of::<Controls>()
                + size_of::<OwnedExternalCharge>()
                + size_of::<Result<OwnedExternalCharge, core::AllocationError>>()
                + 2 * size_of::<Rc<Controls>>()
                + size_of::<Result<Rc<Controls>, BridgeError>>(),
            size_of::<Envelope>()
                + 2 * size_of::<Role>()
                + size_of::<Result<Py<Envelope>, PyErr>>()
                + 2 * size_of::<Result<Py<Role>, PyErr>>()
                + size_of::<Result<Py<PyTuple>, PyErr>>()
                + size_of::<Result<(), BridgeError>>(),
            size_of::<[usize; 13]>(),
            size_of::<Layout>(),
            size_of::<Result<(Layout, usize), std::alloc::LayoutError>>(),
        ];
        parts
            .into_iter()
            .try_fold(0usize, |n, v| n.checked_add(v).ok_or(BridgeError::Resource))
    }
}

// Native payload precedes additional control guard in every object. No object
// stores a Python callback or another Python wrapper: no native/Python cycle.
#[pyclass(unsendable, name = "Envelope", module = "_sigma_local_test_only")]
struct Envelope {
    inner: core::LocalCoordinator,
    controls: Rc<Controls>,
}
#[pyclass(unsendable, name = "Role", module = "_sigma_local_test_only")]
struct Role {
    inner: core::LocalRole,
    controls: Rc<Controls>,
}
#[pyclass(unsendable, name = "Frame", module = "_sigma_local_test_only")]
struct Frame {
    inner: core::LocalFrame,
    controls: Rc<Controls>,
}
#[pyclass(unsendable, name = "Output", module = "_sigma_local_test_only")]
struct Output {
    inner: core::LocalOutput,
    controls: Rc<Controls>,
}

#[pymethods]
impl Envelope {
    fn cancel(&self) -> PyResult<()> {
        self.inner.cancel().map_err(bridge_error)
    }
    fn is_aborted(&self) -> PyResult<bool> {
        self.inner.is_aborted().map_err(bridge_error)
    }
}
#[pymethods]
impl Frame {
    fn is_valid(&self) -> PyResult<bool> {
        self.inner.retained.affinity.check().map_err(bridge_error)?;
        Ok(self.inner.native.as_bytes().is_ok())
    }
}
#[pymethods]
impl Output {
    fn is_valid(&self) -> PyResult<bool> {
        self.inner.retained.affinity.check().map_err(bridge_error)?;
        Ok(self.inner.descriptor().is_ok())
    }
}

fn arity(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>, n: usize) -> PyResult<()> {
    if args.len() != n || kwargs.is_some_and(|kw| !kw.is_empty()) {
        Err(PyTypeError::new_err("invalid local SIGMA arguments"))
    } else {
        Ok(())
    }
}

fn remember(errors: &RefCell<Option<PyErr>>, error: PyErr) {
    let mut slot = errors.borrow_mut();
    if slot.is_none() {
        *slot = Some(error);
    }
}

fn cancelled(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    count: usize,
    index: usize,
    errors: &RefCell<Option<PyErr>>,
) -> bool {
    let result = (|| {
        arity(args, kwargs, count)?;
        let callback = args.get_item(index)?;
        if callback.is_none() {
            Ok(false)
        } else {
            callback.call0()?.is_truthy()
        }
    })();
    match result {
        Ok(value) => value,
        Err(error) => {
            remember(errors, error);
            true
        }
    }
}

fn step<'py>(
    py: Python<'py>,
    mut pending: core::PendingStep<'_>,
    controls: &Rc<Controls>,
    hook: &Bound<'py, PyAny>,
    ready_owner: Option<Py<PyAny>>,
) -> PyResult<Py<PyTuple>> {
    let (payload, ready) = match pending.take_step().map_err(bridge_error)? {
        LocalStep::Frame(inner) => (
            Py::new(
                py,
                Frame {
                    inner,
                    controls: Rc::clone(controls),
                },
            )?
            .into_any(),
            false,
        ),
        // A pure (None, True) tuple could outlive every charged wrapper. Keep an
        // existing opaque role link instead: no extra state or allocation, and
        // every successful result tuple retains its original logical controls.
        LocalStep::Ready => (
            ready_owner.ok_or_else(|| bridge_error(BridgeError::Consumed))?,
            true,
        ),
    };
    if !hook.is_none() {
        hook.call1((payload.bind(py),))?;
    }
    controls.tuple_boundary()?;
    let result = PyTuple::new(
        py,
        [
            payload,
            PyBool::new(py, ready).to_owned().into_any().unbind(),
        ],
    )?
    .unbind();
    pending.commit().map_err(bridge_error)?;
    Ok(result)
}

#[pymethods]
impl Role {
    // *args/**kwargs intentionally prevent PyO3's generated per-argument
    // extraction/arity rejection from running before the native one-shot take.
    #[pyo3(signature = (*args, **kwargs))]
    fn begin<'py>(
        &self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Py<PyTuple>> {
        let errors = RefCell::new(None);
        let outcome = self
            .inner
            .begin(&mut || cancelled(args, kwargs, 2, 0, &errors));
        let pending = outcome.map_err(|error| {
            errors
                .borrow_mut()
                .take()
                .unwrap_or_else(|| bridge_error(error))
        })?;
        if let Some(error) = errors.borrow_mut().take() {
            return Err(error);
        }
        arity(args, kwargs, 2)?; // pending still armed even if native skipped callback
        step(py, pending, &self.controls, &args.get_item(1)?, None)
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn advance<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Py<PyTuple>> {
        let ready_owner = (&slf)
            .into_pyobject(py)
            .unwrap()
            .to_owned()
            .into_any()
            .unbind();
        let errors = RefCell::new(None);
        let mut held: Option<PyRef<'py, Frame>> = None;
        let slot = &mut held;
        let error_slot = &errors;
        let parse = move || {
            // Move the external slot reference: FnOnce, not a reference into a
            // temporary closure/PyRef. Borrow survives the native advance call.
            let slot = slot;
            let result = (|| {
                arity(args, kwargs, 3)?;
                *slot = Some(args.get_item(0)?.extract::<PyRef<'py, Frame>>()?);
                Ok(())
            })();
            if let Err(error) = result {
                remember(error_slot, error);
                return Err(BridgeError::Resource);
            }
            Ok(&slot.as_ref().expect("parsed frame").inner)
        };
        let outcome = slf
            .inner
            .advance(parse, &mut || cancelled(args, kwargs, 3, 1, &errors));
        let pending = outcome.map_err(|error| {
            errors
                .borrow_mut()
                .take()
                .unwrap_or_else(|| bridge_error(error))
        })?;
        if let Some(error) = errors.borrow_mut().take() {
            return Err(error);
        }
        step(
            py,
            pending,
            &slf.controls,
            &args.get_item(2)?,
            Some(ready_owner),
        )
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn finish<'py>(
        &self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Py<Output>> {
        let mut pending = self.inner.finish().map_err(bridge_error)?;
        arity(args, kwargs, 1)?;
        let inner = pending.take_output().map_err(bridge_error)?;
        let output = Py::new(
            py,
            Output {
                inner,
                controls: Rc::clone(&self.controls),
            },
        )?;
        let hook = args.get_item(0)?;
        if !hook.is_none() {
            hook.call1((output.bind(py),))?;
        }
        if self.controls.fault.get() == Fault::OutputMemory {
            self.controls.fault.set(Fault::None);
            return Err(PyMemoryError::new_err(
                "injected output packaging allocation failure",
            ));
        }
        pending.commit().map_err(bridge_error)?;
        Ok(output)
    }
}

/// Test module/types initialize BEFORE source issuance. Not registered into
/// production `_native`, sys.modules, or the SDK component registry.
fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let module = PyModule::new(py, "_sigma_local_test_only")?;
    module.add_class::<Envelope>()?;
    module.add_class::<Role>()?;
    module.add_class::<Frame>()?;
    module.add_class::<Output>()?;
    Ok(module)
}

/// An experiment in explicitly UNADMITTED CPython allocations. This is not a
/// production factory and never calls/ignores/bypasses take_python_owners.
/// Genuine Rust transaction remains armed throughout the experiment.
fn package_fixture<'py>(
    py: Python<'py>,
    mut pending: PendingOwners,
    controls: Rc<Controls>,
    hook: &Bound<'py, PyAny>,
) -> PyResult<Py<PyTuple>> {
    if !std::ptr::eq(controls.charge.budget(), pending.retained._charge.budget()) {
        return Err(bridge_error(BridgeError::ForeignInput));
    }
    let (coordinator, [left, right]) = pending.take_owners().map_err(bridge_error)?;
    let envelope = Py::new(
        py,
        Envelope {
            inner: coordinator,
            controls: Rc::clone(&controls),
        },
    )?;
    let left = Py::new(
        py,
        Role {
            inner: left,
            controls: Rc::clone(&controls),
        },
    )?;
    let right = Py::new(
        py,
        Role {
            inner: right,
            controls: Rc::clone(&controls),
        },
    )?;
    if !hook.is_none() {
        hook.call1((envelope.bind(py), left.bind(py), right.bind(py)))?;
    }
    controls.tuple_boundary()?;
    let tuple =
        PyTuple::new(py, [envelope.into_any(), left.into_any(), right.into_any()])?.unbind();
    pending.commit().map_err(bridge_error)?;
    Ok(tuple)
}

#[path = "sigma_python_local_tests.rs"]
mod tests;
