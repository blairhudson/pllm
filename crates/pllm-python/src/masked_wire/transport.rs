//! Opaque party-local handles; the authenticated role session owns admission.
use super::{error, native, parse_layout, Context, Profile, Step};
use crate::invalid;
use native::transport;
use pyo3::{prelude::*, types::PyBytes};
use zeroize::Zeroizing;

fn context(plan: &[u8], session: &[u8], operation: &[u8], index: u64) -> PyResult<Context> {
    fn digest(bytes: &[u8]) -> PyResult<[u8; 32]> {
        bytes
            .try_into()
            .map_err(|_| invalid("masked-wire context digests require 32 bytes".into()))
    }
    Ok(Context {
        plan: digest(plan)?,
        session: digest(session)?,
        operation: digest(operation)?,
        first_tensor_index: index,
    })
}

fn integers(bytes: &[u8], maximum: usize) -> PyResult<Zeroizing<Vec<i64>>> {
    if bytes.is_empty() || bytes.len() % 8 != 0 || bytes.len() / 8 > maximum {
        return Err(invalid(
            "masked-wire requires bounded packed signed i64 inputs".into(),
        ));
    }
    Ok(Zeroizing::new(
        bytes
            .chunks_exact(8)
            .map(|x| i64::from_le_bytes(x.try_into().expect("checked i64 width")))
            .collect(),
    ))
}

#[pyclass(name = "_MaskedWireClient", module = "pllm._native")]
struct MaskedWireClient {
    inner: Option<native::Client>,
}

#[pymethods]
impl MaskedWireClient {
    fn encode<'py>(
        &mut self,
        py: Python<'py>,
        gate: &Bound<'py, PyAny>,
        up: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let mut client = self
            .inner
            .take()
            .ok_or_else(|| error(native::Error::Consumed))?;
        // Take ownership before any Python type conversion; all errors burn.
        let gate = integers(gate.cast::<PyBytes>()?.as_bytes(), native::MAX_LANES)?;
        let up = integers(up.cast::<PyBytes>()?.as_bytes(), native::MAX_LANES)?;
        let output = py
            .detach(|| transport::encode_input(&mut client, &gate, &up))
            .map_err(error)?;
        py.check_signals()?;
        self.inner = Some(client);
        Ok(PyBytes::new(py, &output))
    }

    fn decode<'py>(
        &mut self,
        py: Python<'py>,
        left: &Bound<'py, PyAny>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let client = self
            .inner
            .take()
            .ok_or_else(|| error(native::Error::Consumed))?;
        let left = left.cast::<PyBytes>()?.as_bytes();
        let right = right.cast::<PyBytes>()?.as_bytes();
        let output = py
            .detach(|| transport::decode_output(client, left, right))
            .map_err(error)?;
        let output = Zeroizing::new(
            output
                .into_iter()
                .flat_map(i64::to_le_bytes)
                .collect::<Vec<_>>(),
        );
        py.check_signals()?;
        Ok(PyBytes::new(py, &output))
    }

    fn cancel(&mut self) {
        self.inner = None;
    }
}

#[pyclass(name = "_MaskedWireParty", module = "pllm._native")]
struct MaskedWireParty {
    inner: Option<native::Party>,
}

#[pymethods]
impl MaskedWireParty {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        material: &[u8],
        plan: &[u8],
        session: &[u8],
        operation: &[u8],
        first_tensor_index: u64,
        lanes: usize,
        layout: &str,
        party: u8,
    ) -> PyResult<Self> {
        let context = context(plan, session, operation, first_tensor_index)?;
        let layout = parse_layout(layout)?;
        let inner = py
            .detach(|| transport::import_party(material, context, lanes, layout, party))
            .map_err(error)?;
        py.check_signals()?;
        Ok(Self { inner: Some(inner) })
    }

    fn start<'py>(
        &mut self,
        py: Python<'py>,
        frame: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let mut party = self
            .inner
            .take()
            .ok_or_else(|| error(native::Error::Consumed))?;
        let frame = frame.cast::<PyBytes>()?.as_bytes();
        let output = py
            .detach(|| transport::start(&mut party, frame))
            .map_err(error)?;
        py.check_signals()?;
        self.inner = Some(party);
        Ok(PyBytes::new(py, &output))
    }

    fn advance<'py>(
        &mut self,
        py: Python<'py>,
        frame: &Bound<'py, PyAny>,
    ) -> PyResult<(bool, Bound<'py, PyBytes>)> {
        let mut party = self
            .inner
            .take()
            .ok_or_else(|| error(native::Error::Consumed))?;
        let frame = frame.cast::<PyBytes>()?.as_bytes();
        let step = py.detach(|| party.advance(frame)).map_err(error)?;
        py.check_signals()?;
        match step {
            Step::Send(output) => {
                self.inner = Some(party);
                Ok((false, PyBytes::new(py, &output)))
            }
            Step::Done(output) => Ok((true, PyBytes::new(py, &transport::encode_output(output)))),
        }
    }

    fn cancel(&mut self) {
        self.inner = None;
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn _masked_wire_issue<'py>(
    py: Python<'py>,
    plan: &[u8],
    session: &[u8],
    operation: &[u8],
    first_tensor_index: u64,
    lanes: usize,
    layout: &str,
) -> PyResult<(MaskedWireClient, Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    let context = context(plan, session, operation, first_tensor_index)?;
    let layout = parse_layout(layout)?;
    let (client, left, right) = py
        .detach(|| {
            let (client, [left, right]) =
                native::issue_reference_with_layout(context, lanes, layout)?;
            Ok::<_, native::Error>((
                client,
                transport::export_party(left)?,
                transport::export_party(right)?,
            ))
        })
        .map_err(error)?;
    py.check_signals()?;
    Ok((
        MaskedWireClient {
            inner: Some(client),
        },
        PyBytes::new(py, &left),
        PyBytes::new(py, &right),
    ))
}

#[pyfunction]
fn _masked_wire_clear<'py>(
    py: Python<'py>,
    gate: &[u8],
    up: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    let gate = integers(gate, 1 << 20)?;
    let up = integers(up, 1 << 20)?;
    if gate.len() != up.len() {
        return Err(invalid("masked-wire input dimensions differ".into()));
    }
    let result = py
        .detach(|| {
            let profile = Profile::new();
            let mut result = Zeroizing::new(Vec::with_capacity(gate.len() * 8));
            for (&g, &u) in gate.iter().zip(up.iter()) {
                result.extend(
                    profile
                        .evaluate(g, u)
                        .map_err(|_| native::Error::Domain)?
                        .to_le_bytes(),
                );
            }
            Ok::<_, native::Error>(result)
        })
        .map_err(error)?;
    py.check_signals()?;
    Ok(PyBytes::new(py, &result))
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<MaskedWireClient>()?;
    module.add_class::<MaskedWireParty>()?;
    module.add_function(wrap_pyfunction!(_masked_wire_issue, module)?)?;
    module.add_function(wrap_pyfunction!(_masked_wire_clear, module)?)
}
