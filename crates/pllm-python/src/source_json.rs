//! Bounded binary64 metadata conversion. No model/forward admission authority.
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyFloat, PyString};

const MAX_TOKEN_CHARS: usize = 1024;
const INVALID_NUMBER: &str = "source JSON number is invalid";
const NONFINITE_NUMBER: &str = "source JSON number must be finite";

/// Rust's core binary64 parser is heap-free and correctly rounded. JSON lexical
/// grammar belongs to the caller's existing tokenizer; this is only conversion.
fn finite_f64(token: &str) -> Result<f64, &'static str> {
    let value = token.parse::<f64>().map_err(|_| INVALID_NUMBER)?;
    if !value.is_finite() { return Err(NONFINITE_NUMBER); }
    Ok(value)
}

/// Borrow at most 1024 ASCII token bytes; no Rust String, serde/Python float
/// conversion or token-bearing error. Supported CPython result float owns 24
/// bytes. Unicode UTF-8 access is deferred until both length and ASCII checks;
/// any interpreter encoding cache is then bounded by 1024 bytes plus terminator.
/// Native C/Rust machine stack is outside the logical heap/frame-object contract.
#[pyfunction]
fn source_json_f64<'py>(py: Python<'py>, token: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyFloat>> {
    let token = token.cast_exact::<PyString>()
        .map_err(|_| PyTypeError::new_err("source JSON number must be exact str"))?;
    crate::forward::require_publication_abi()?;
    let length = unsafe { pyo3::ffi::PyUnicode_GetLength(token.as_ptr()) };
    if length < 0 { return Err(PyErr::fetch(py)); }
    if length as usize > MAX_TOKEN_CHARS {
        return Err(PyValueError::new_err("source JSON number exceeds 1024 ASCII characters"));
    }
    // No UTF-8 extraction/copy or Python conversion dispatch during these checks.
    for index in 0..length {
        let character = unsafe { pyo3::ffi::PyUnicode_ReadChar(token.as_ptr(), index) };
        if character > 0x7f {
            return Err(PyValueError::new_err("source JSON number must be ASCII"));
        }
    }
    let value = finite_f64(token.to_str()?).map_err(PyValueError::new_err)?;
    Ok(PyFloat::new(py, value))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    // Availability is checked by the function, not registration: unsupported ABI
    // never prevents importing the extension or using legacy decoder bindings.
    module.add_function(wrap_pyfunction!(source_json_f64, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::{PyBool, PyBytes, PyInt};

    fn setup() {
        Python::initialize();
        Python::attach(crate::forward::initialize_publication_abi);
    }

    fn vectors() -> [(&'static str, u64); 12] {
        [
            ("0", 0),
            ("-0.0", 0x8000_0000_0000_0000),
            ("0.1", 0x3fb9_9999_9999_999a),
            ("9007199254740993", 0x4340_0000_0000_0000),
            ("9007199254740995", 0x4340_0000_0000_0002),
            ("1.00000000000000011102230246251565404236316680908203125", 0x3ff0_0000_0000_0000),
            ("1.00000000000000011102230246251565404236316680908203126", 0x3ff0_0000_0000_0001),
            ("4.9406564584124654e-324", 1),
            ("2.4703282292062327e-324", 0),
            ("2.4703282292062328e-324", 1),
            ("2.2250738585072014e-308", 0x0010_0000_0000_0000),
            ("1.7976931348623157e308", 0x7fef_ffff_ffff_ffff),
        ]
    }

    #[test]
    fn core_binary64_vectors_have_exact_rounding_bits() {
        for (token, bits) in vectors() { assert_eq!(finite_f64(token).unwrap().to_bits(), bits, "{token}"); }
        assert_eq!(finite_f64("1e-10000").unwrap().to_bits(), 0);
        assert_eq!(finite_f64("-1e-10000").unwrap().to_bits(), 0x8000_0000_0000_0000);
    }

    #[test]
    fn core_invalid_nonfinite_and_overflow_use_fixed_errors() {
        for token in ["", "word", "1_0", "1e", "--1"] {
            assert_eq!(finite_f64(token), Err(INVALID_NUMBER));
        }
        for token in ["NaN", "inf", "-inf", "Infinity", "1e309", "1.7976931348623159e308"] {
            assert_eq!(finite_f64(token), Err(NONFINITE_NUMBER));
        }
    }

    #[test]
    fn native_bridge_returns_genuine_python_float_and_preserves_bits() {
        setup();
        Python::attach(|py| {
            for (token, bits) in vectors().into_iter().chain([("1e-10000", 0), ("-1e-10000", 0x8000_0000_0000_0000)]) {
                let token = PyString::new(py, token);
                let result = source_json_f64(py, token.as_any()).unwrap();
                assert!(result.is_exact_instance_of::<PyFloat>());
                assert_eq!(result.value().to_bits(), bits);
            }
        });
    }

    #[test]
    fn native_bridge_requires_exact_string_without_custom_conversion() {
        setup();
        Python::attach(|py| {
            for value in [PyBytes::new(py, b"1").into_any(), PyInt::new(py, 1).into_any(),
                PyBool::new(py, true).to_owned().into_any(), py.None().into_bound(py)] {
                let error = source_json_f64(py, &value).unwrap_err();
                assert!(error.is_instance_of::<PyTypeError>(py));
            }
            let module = PyModule::from_code(py, c"class Token(str):\n    def __str__(self):\n        raise AssertionError('conversion called')\nvalue = Token('1')\n", c"source_json_test", c"source_json_test").unwrap();
            let error = source_json_f64(py, &module.getattr("value").unwrap()).unwrap_err();
            assert!(error.is_instance_of::<PyTypeError>(py));
        });
    }

    #[test]
    fn native_bridge_bounds_length_and_ascii_before_utf8_access() {
        setup();
        Python::attach(|py| {
            let maximum = PyString::new(py, &"0".repeat(MAX_TOKEN_CHARS));
            assert_eq!(source_json_f64(py, maximum.as_any()).unwrap().value().to_bits(), 0);
            for text in ["0".repeat(MAX_TOKEN_CHARS + 1), "é".repeat(MAX_TOKEN_CHARS + 1)] {
                let token = PyString::new(py, &text);
                let error = source_json_f64(py, token.as_any()).unwrap_err();
                assert!(error.to_string().contains("exceeds 1024"));
            }
            for text in ["é", "１", "1\u{00a0}", "😀"] {
                let token = PyString::new(py, text);
                let error = source_json_f64(py, token.as_any()).unwrap_err();
                assert!(error.to_string().contains("must be ASCII"));
            }
        });
    }

    #[test]
    fn native_bridge_refuses_invalid_and_nonfinite_without_echoing_tokens() {
        setup();
        Python::attach(|py| {
            for text in ["private-invalid-token", "1e", "NaN", "Infinity", "-inf", "1e309"] {
                let token = PyString::new(py, text);
                let error = source_json_f64(py, token.as_any()).unwrap_err();
                assert!(error.is_instance_of::<PyValueError>(py));
                let message = error.to_string();
                assert!(message.contains(INVALID_NUMBER) || message.contains(NONFINITE_NUMBER));
                assert!(!message.contains(text));
            }
        });
    }
}
