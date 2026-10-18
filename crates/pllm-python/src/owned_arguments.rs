//! Fixed signatures parsed only after consuming an opaque live owner.
use pyo3::{
    exceptions::PyTypeError,
    prelude::*,
    types::{PyAny, PyDict, PyString, PyTuple},
};

pub(crate) fn arguments<'py, const N: usize>(
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
    names: [&str; N],
    context: &'static str,
) -> PyResult<[Bound<'py, PyAny>; N]> {
    let error = |message: &str| PyTypeError::new_err(format!("{context} {message}"));
    let positional = args.len();
    let keywords = kwargs.map_or(0, |v| v.len());
    if positional > N || keywords > N {
        return Err(error("argument count exceeds signature"));
    }
    if let Some(kwargs) = kwargs {
        for (name, _) in kwargs.iter() {
            if !name.is_exact_instance_of::<PyString>() {
                return Err(error("keyword names must be plain strings"));
            }
        }
    }
    let mut values: [Option<Bound<'py, PyAny>>; N] = std::array::from_fn(|_| None);
    let mut matched = 0;
    for (index, name) in names.iter().enumerate() {
        let named = match kwargs {
            Some(v) => v.get_item(*name)?,
            None => None,
        };
        matched += usize::from(named.is_some());
        values[index] = if index < positional {
            if named.is_some() {
                return Err(error("multiple values for argument"));
            }
            Some(args.get_item(index)?)
        } else {
            named
        };
    }
    if matched != keywords {
        return Err(error("unexpected keyword argument"));
    }
    if values.iter().any(Option::is_none) {
        return Err(error("missing required argument"));
    }
    Ok(values.map(|v| v.expect("checked required arguments")))
}
