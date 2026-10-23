//! Real interpreter tests, not Python input admission. Type initialization and
//! fixture material origins are deliberately outside the claimed Rust boundary.
use super::*;
use pyo3::types::PyDict;
use std::ffi::CString;

fn run(py: Python<'_>, code: &str, locals: &Bound<'_, PyDict>) {
    py.run(&CString::new(code).unwrap(), Some(locals), Some(locals))
        .unwrap();
}

fn issue<'py>(
    py: Python<'py>,
    root: &OwnedBudget,
    dealer: &mut core::OwnedLocalDealer,
    locals: &Bound<'py, PyDict>,
) {
    let controls = Controls::reserve(root, Fault::None).unwrap();
    let prepared = fixture::prepare_changed(root, 0).unwrap();
    let pending =
        core::from_admitted_rust_inputs(prepared.original, prepared.data, root, dealer).unwrap();
    let objects = package_fixture(py, pending, controls, py.None().bind(py)).unwrap();
    locals.set_item("owners", objects).unwrap();
    run(py, "env, left, right = owners\ndel owners", locals);
}

fn initialize(py: Python<'_>) -> Bound<'_, PyModule> {
    let module = module(py).unwrap(); // BEFORE authentic source issuance
    assert!(module.getattr("Role").unwrap().call0().is_err());
    module
}

const START: &str = "a, ready = left.begin(None, None)\nassert not ready\nb, ready = right.begin(None, None)\nassert not ready";
const READY: &str = r#"
for i in range(10):
    aa, ar = left.advance(b, None, None)
    bb, br = right.advance(a, None, None)
    assert not ar and not br
    a, b = aa, bb
    del aa, bb
assert left.advance(b, None, None) == (left, True)
assert right.advance(a, None, None) == (right, True)
"#;

#[test]
fn actual_python_eleven_exchanges_opaque_outputs_and_original_guards_survive_root_drop() {
    let _serial = fixture::SERIAL.lock().unwrap();
    Python::initialize();
    Python::attach(|py| {
        let _module = initialize(py);
        let locals = PyDict::new(py);
        let identity = {
            let root = OwnedBudget::new(fixture::LIMIT).unwrap();
            let mut dealer = fixture::dealer(&root);
            let identity = std::ptr::from_ref(root.budget());
            issue(py, &root, &mut dealer, &locals);
            identity
        };
        run(py, START, &locals);
        run(py, READY, &locals);
        run(
            py,
            r#"
out0 = left.finish(None)
out1 = right.finish(None)
assert out0.is_valid() and out1.is_valid()
for obj in (env, left, right, a, b, out0, out1):
    for name in ('bytes', 'mask', 'inner', 'common_masked', 'native', 'source'):
        assert not hasattr(obj, name)
del obj, name
"#,
            &locals,
        );
        let frame = locals
            .get_item("a")
            .unwrap()
            .unwrap()
            .extract::<PyRef<'_, Frame>>()
            .unwrap();
        assert_eq!(frame.inner.retained.frames.get(), 22);
        assert_eq!(
            std::ptr::from_ref(frame.inner.retained._charge.budget()),
            identity
        );
        assert!(frame
            .inner
            .retained
            ._original
            .is_budget(frame.inner.retained._charge.budget()));
        assert!(std::ptr::eq(
            frame.inner.retained._data.budget(),
            frame.inner.retained._charge.budget()
        ));
        let weak = Rc::downgrade(&frame.inner.retained);
        drop(frame);
        run(py, "del env\nassert not out0.is_valid()\nassert not out1.is_valid()\nassert not a.is_valid()\ndel left, right, out0, out1, b", &locals);
        assert!(
            weak.upgrade().is_some(),
            "last Python frame retains real original owners"
        );
        locals.clear();
        assert!(weak.upgrade().is_none());
    });
}

#[test]
fn actual_python_bad_arity_peer_conversion_and_callback_truthiness_burn_before_reuse() {
    let _serial = fixture::SERIAL.lock().unwrap();
    Python::initialize();
    Python::attach(|py| {
        let _module = initialize(py);
        for case in 0..4 {
            let root = OwnedBudget::new(fixture::LIMIT).unwrap();
            let mut dealer = fixture::dealer(&root);
            let baseline = root.budget().used_bytes();
            let locals = PyDict::new(py);
            issue(py, &root, &mut dealer, &locals);
            if case != 0 {
                run(py, START, &locals);
            }
            let call = match case {
                0 => "left.begin()",
                1 => "left.advance(object(), None, None)",
                2 => "left.advance(b, bad, None)",
                _ => "left.advance(b, None, None, extra=True)",
            };
            locals.set_item("call", call).unwrap();
            run(
                py,
                r#"
class BadTruth:
    def __bool__(self):
        raise ValueError('truth conversion failed')
def bad(): return BadTruth()
try:
    eval(call)
except (TypeError, ValueError, RuntimeError):
    pass
else: raise AssertionError('invalid conversion succeeded')
assert env.is_aborted()
try: right.begin(None, None)
except RuntimeError: pass
else: raise AssertionError('sibling survived')
"#,
                &locals,
            );
            locals.clear();
            assert_eq!(root.budget().used_bytes(), baseline);
        }
    });
}

#[test]
fn actual_python_callback_queries_cancel_and_same_role_reentry_without_borrow_failure() {
    let _serial = fixture::SERIAL.lock().unwrap();
    Python::initialize();
    Python::attach(|py| {
        let _module = initialize(py);
        for cancel in [false, true] {
            let root = OwnedBudget::new(fixture::LIMIT).unwrap();
            let mut dealer = fixture::dealer(&root);
            let baseline = root.budget().used_bytes();
            let locals = PyDict::new(py);
            issue(py, &root, &mut dealer, &locals);
            run(py, START, &locals);
            locals.set_item("cancel", cancel).unwrap();
            run(
                py,
                r#"
seen = []
def callback():
    seen.append(env.is_aborted())
    if cancel: env.cancel()
    else:
        try: left.begin(None, None)
        except RuntimeError: pass
        else: raise AssertionError('reentry succeeded')
    return False
try: left.advance(b, callback, None)
except RuntimeError: pass
else: raise AssertionError('cancelled outer operation succeeded')
assert seen and seen[0] is False
assert env.is_aborted() and not a.is_valid() and not b.is_valid()
"#,
                &locals,
            );
            locals.clear();
            assert_eq!(root.budget().used_bytes(), baseline);
        }
    });
}

#[test]
fn actual_python_tuple_memory_error_or_rust_unwind_revokes_escaped_frame() {
    let _serial = fixture::SERIAL.lock().unwrap();
    Python::initialize();
    Python::attach(|py| {
        let _module = initialize(py);
        for fault in [Fault::TupleMemory, Fault::TupleUnwind] {
            let root = OwnedBudget::new(fixture::LIMIT).unwrap();
            let mut dealer = fixture::dealer(&root);
            let baseline = root.budget().used_bytes();
            let locals = PyDict::new(py);
            issue(py, &root, &mut dealer, &locals);
            locals
                .get_item("left")
                .unwrap()
                .unwrap()
                .extract::<PyRef<'_, Role>>()
                .unwrap()
                .controls
                .fault
                .set(fault);
            run(
                py,
                r#"
escaped = []
def leak(frame): escaped.append(frame)
try: left.begin(None, leak)
except BaseException as error:
    assert isinstance(error, MemoryError) or type(error).__name__ == 'PanicException'
else: raise AssertionError('packaging fault ignored')
assert escaped and not escaped[0].is_valid()
assert env.is_aborted()
"#,
                &locals,
            );
            locals.clear();
            assert_eq!(root.budget().used_bytes(), baseline);
        }
    });
}

#[test]
fn actual_python_output_packaging_failure_revokes_finished_sibling_and_escaped_output() {
    let _serial = fixture::SERIAL.lock().unwrap();
    Python::initialize();
    Python::attach(|py| {
        let _module = initialize(py);
        let root = OwnedBudget::new(fixture::LIMIT).unwrap();
        let mut dealer = fixture::dealer(&root);
        let baseline = root.budget().used_bytes();
        let locals = PyDict::new(py);
        issue(py, &root, &mut dealer, &locals);
        run(py, START, &locals);
        run(py, READY, &locals);
        run(
            py,
            "out0 = left.finish(None)\nassert out0.is_valid()",
            &locals,
        );
        locals
            .get_item("right")
            .unwrap()
            .unwrap()
            .extract::<PyRef<'_, Role>>()
            .unwrap()
            .controls
            .fault
            .set(Fault::OutputMemory);
        run(
            py,
            r#"
escaped = []
def leak(output): escaped.append(output)
try: right.finish(leak)
except MemoryError: pass
else: raise AssertionError('output packaging succeeded')
assert env.is_aborted() and escaped
assert not out0.is_valid() and not escaped[0].is_valid()
"#,
            &locals,
        );
        locals.clear();
        assert_eq!(root.budget().used_bytes(), baseline);
    });
}

#[test]
fn actual_python_creation_tuple_failure_revokes_already_packaged_handles() {
    let _serial = fixture::SERIAL.lock().unwrap();
    Python::initialize();
    Python::attach(|py| {
        let _module = initialize(py);
        let root = OwnedBudget::new(fixture::LIMIT).unwrap();
        let mut dealer = fixture::dealer(&root);
        let baseline = root.budget().used_bytes();
        let locals = PyDict::new(py);
        run(
            py,
            "escaped = []\ndef leak(e, l, r): escaped.extend((e, l, r))",
            &locals,
        );
        let controls = Controls::reserve(&root, Fault::TupleMemory).unwrap();
        let prepared = fixture::prepare_changed(&root, 0).unwrap();
        let pending =
            core::from_admitted_rust_inputs(prepared.original, prepared.data, &root, &mut dealer)
                .unwrap();
        let hook = locals.get_item("leak").unwrap().unwrap();
        let error = package_fixture(py, pending, controls, &hook).unwrap_err();
        assert!(error.is_instance_of::<PyMemoryError>(py));
        drop(error);
        drop(hook);
        run(
            py,
            r#"
assert len(escaped) == 3 and escaped[0].is_aborted()
try: escaped[1].begin(None, None)
except RuntimeError: pass
else: raise AssertionError('escaped creation role survived')
"#,
            &locals,
        );
        locals.clear();
        assert_eq!(root.budget().used_bytes(), baseline);
    });
}

#[test]
fn actual_python_one_ready_cannot_release_output_or_preserve_sibling() {
    let _serial = fixture::SERIAL.lock().unwrap();
    Python::initialize();
    Python::attach(|py| {
        let _module = initialize(py);
        let root = OwnedBudget::new(fixture::LIMIT).unwrap();
        let mut dealer = fixture::dealer(&root);
        let baseline = root.budget().used_bytes();
        let locals = PyDict::new(py);
        issue(py, &root, &mut dealer, &locals);
        run(py, START, &locals);
        run(
            py,
            r#"
for i in range(10):
    aa, ar = left.advance(b, None, None)
    bb, br = right.advance(a, None, None)
    assert not ar and not br
    a, b = aa, bb
    del aa, bb
assert left.advance(b, None, None) == (left, True)
try: left.finish(None)
except RuntimeError: pass
else: raise AssertionError('one Ready released output')
assert env.is_aborted()
try: right.advance(a, None, None)
except RuntimeError: pass
else: raise AssertionError('sibling survived early finish')
"#,
            &locals,
        );
        locals.clear();
        assert_eq!(root.budget().used_bytes(), baseline);
    });
}

#[test]
fn actual_python_packaging_callback_failure_and_root_drop_revoke_siblings() {
    let _serial = fixture::SERIAL.lock().unwrap();
    Python::initialize();
    Python::attach(|py| {
        let _module = initialize(py);
        for root_drop in [false, true] {
            let root = OwnedBudget::new(fixture::LIMIT).unwrap();
            let mut dealer = fixture::dealer(&root);
            let baseline = root.budget().used_bytes();
            let locals = PyDict::new(py);
            issue(py, &root, &mut dealer, &locals);
            locals.set_item("root_drop", root_drop).unwrap();
            run(
                py,
                r#"
escaped = []
def package(frame):
    global env
    escaped.append(frame)
    if root_drop: del env
    else: raise ValueError('Python packaging failed')
try: left.begin(None, package)
except (ValueError, RuntimeError): pass
else: raise AssertionError('failed packaging committed')
assert escaped and not escaped[0].is_valid()
try: right.begin(None, None)
except RuntimeError: pass
else: raise AssertionError('sibling survived')
"#,
                &locals,
            );
            locals.clear();
            assert_eq!(root.budget().used_bytes(), baseline);
        }
    });
}

#[test]
fn real_python_wrong_thread_refusal_leaves_original_same_thread_slot_untaken() {
    let _serial = fixture::SERIAL.lock().unwrap();
    Python::initialize();
    Python::attach(|py| {
        let _module = initialize(py);
        let root = OwnedBudget::new(fixture::LIMIT).unwrap();
        let mut dealer = fixture::dealer(&root);
        let baseline = root.budget().used_bytes();
        let locals = PyDict::new(py);
        issue(py, &root, &mut dealer, &locals);
        run(
            py,
            r#"
import threading
errors = []
def foreign():
    try: left.begin(None, None)
    except BaseException as error: errors.append(type(error).__name__)
worker = threading.Thread(target=foreign)
worker.start()
worker.join(timeout=10)
assert not worker.is_alive() and errors
assert not env.is_aborted()
a, ready = left.begin(None, None)
assert not ready and a.is_valid()
"#,
            &locals,
        );
        locals.clear();
        assert_eq!(root.budget().used_bytes(), baseline);
    });
}

#[test]
fn known_controls_exact_one_short_are_real_reservations_not_complete_python_quotes() {
    let bytes = KnownControls::checked().unwrap();
    for short in [false, true] {
        let baseline = OwnedBudget::resources().unwrap().initial_reserved_bytes;
        let total = baseline + bytes + size_of::<OwnedExternalCharge>() - usize::from(short);
        let root = OwnedBudget::new(total).unwrap();
        let result = Controls::reserve(&root, Fault::None);
        if short {
            assert!(result.is_err());
        } else {
            assert!(result.is_ok());
            assert_eq!(root.budget().used_bytes(), total);
        }
        drop(result);
        assert_eq!(root.budget().used_bytes(), baseline);
    }
    assert_eq!(
        crate::sigma_linear_activation_admission::require_python_allocation_contract(),
        Err(BridgeError::UnadmittedPython)
    );
}
