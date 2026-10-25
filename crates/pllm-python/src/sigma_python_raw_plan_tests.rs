//! Exact Python bytes -> admitted Plan -> existing pinned-source role bridge.
//!
//! Test-only composition, NOT a Python factory or CPython allocation contract.
//! Existing source/declaration/execution fixtures supply their actual authority;
//! neither immutable bytes nor the generic Value importer supplies source trust.
//! Python source creation/type initialization/callbacks remain external fixture
//! work. Only the NEW native input copy enters the admitted handoff below.
use super::*;
use crate::owned_raw_input as ingress;
use pllm_core::edge_contract::EdgeDescriptor;
use pyo3::types::PyBytes;

fn pending_from_python_plan(
    py: Python<'_>,
    root: &OwnedBudget,
    dealer: &mut core::OwnedLocalDealer,
) -> (PendingOwners, EdgeDescriptor, usize) {
    // Reuse genuine B15 publisher/data fixtures, not a copied parser, issuer or
    // fabricated source certificate. This old serialized owner ONLY constructs
    // the external Python fixture. It is gone before the new ingress begins.
    let fixture::PreparedRawFixture {
        raw: external_origin,
        declaration,
        data,
        descriptor,
        ..
    } = fixture::prepare_raw_changed(root, 0).unwrap();
    let external = PyBytes::new(py, external_origin.payload());
    drop(external_origin);

    let mut limits = fixture::raw_input_limits();
    let whole_work = limits.max_work;
    let frozen = ingress::from_exact_bytes(
        root,
        external.as_any(),
        ingress::Limits {
            max_bytes: external.as_bytes().len(),
            max_work: whole_work,
        },
    )
    .unwrap();
    let copy_work = frozen.copy_work_bound();
    assert_eq!(frozen.bytes(), external.as_bytes());
    assert_ne!(frozen.bytes().as_ptr(), external.as_bytes().as_ptr());
    let pointer = frozen.bytes().as_ptr();
    let original = frozen.into_original();
    assert_eq!(original.payload().as_ptr(), pointer);
    assert_eq!(original.payload().capacity(), external.as_bytes().len());
    assert!(std::ptr::eq(original.budget(), root.budget()));
    let raw_charge = original.charged_bytes();
    drop(external); // No Python source handle is part of the native custody.

    // Disjoint aggregate work: ingress is not free when its owner moves. Use
    // the existing consuming bridge, which reserves its controls BEFORE Plan,
    // source/inventory and native factories. Do not convert Value into authority.
    limits.max_work = whole_work.checked_sub(copy_work).unwrap();
    let pending =
        core::from_charged_plan_bytes(original, declaration, data, root, dealer, limits).unwrap();
    assert!(pending.retained._original.is_budget(root.budget()));
    let preparation = pending.retained._original.preparation_work().unwrap();
    assert!(preparation.parse_prepaid() > 0);
    assert!(preparation.metadata_prepaid() > 0);
    assert_eq!(
        preparation.inventory_prepaid(),
        fixture::raw_input_limits().inventory_work
    );
    assert!(copy_work
        .checked_add(preparation.total_prepaid())
        .is_some_and(|work| work <= whole_work));
    (pending, descriptor, raw_charge)
}

fn assert_custody(
    retained: &Rc<core::Retention>,
    observer: &OwnedBudget,
    raw_charge: usize,
    controls_charge: usize,
) {
    assert!(std::ptr::eq(retained._charge.budget(), observer.budget()));
    assert!(retained._original.is_budget(observer.budget()));
    assert!(std::ptr::eq(retained._data.budget(), observer.budget()));
    assert!(retained._original.preparation_work().is_some());
    let baseline = OwnedBudget::resources().unwrap().initial_reserved_bytes;
    let floor = baseline
        .checked_add(raw_charge)
        .and_then(|n| n.checked_add(controls_charge))
        .and_then(|n| n.checked_add(retained._charge.charged_bytes()))
        .and_then(|n| n.checked_add(retained._data.charged_bytes()))
        .unwrap();
    assert!(observer.budget().used_bytes() >= floor);
    // This is a logical lower-bound/lifetime observation, not a private raw
    // pointer accessor or exact peak oracle. Source custody is the actual
    // prepare_from_plan_bytes -> OwnedPlanInventory.raw -> Retention path.
}

#[test]
fn exact_python_plan_reaches_accepted_outputs_and_last_frame_or_output_keeps_custody() {
    let _serial = fixture::SERIAL.lock().unwrap();
    Python::initialize();
    Python::attach(|py| {
        let _types = initialize(py); // Existing test module, before issuance.
        for output_last in [false, true] {
            let root = OwnedBudget::new(fixture::LIMIT).unwrap();
            let observer = root.try_clone().unwrap(); // Counter observer only.
            let mut dealer = fixture::dealer(&root);
            let controls = Controls::reserve(&root, Fault::None).unwrap();
            let controls_charge = controls.charge.charged_bytes();
            let (pending, descriptor, raw_charge) =
                pending_from_python_plan(py, &root, &mut dealer);
            let receipt = pending.retained._original.preparation_work().unwrap();
            let weak = Rc::downgrade(&pending.retained);
            let objects = package_fixture(py, pending, controls, py.None().bind(py)).unwrap();
            let locals = PyDict::new(py);
            locals.set_item("owners", objects).unwrap();
            run(py, "env, left, right = owners\ndel owners", &locals);
            drop((dealer, root));

            // Use the committed Python bridge's existing eleven-exchange
            // scripts, including both-Ready withholding, not a new driver.
            run(py, START, &locals);
            run(py, READY, &locals);
            run(
                py,
                r#"
out0 = left.finish(None)
out1 = right.finish(None)
assert out0.is_valid() and out1.is_valid()
assert a.is_valid() and b.is_valid()
"#,
                &locals,
            );
            {
                let retained = weak.upgrade().unwrap();
                assert_eq!(retained.frames.get(), 22);
                assert_eq!(retained.outputs.get(), 2);
                assert_eq!(retained._original.preparation_work(), Some(receipt));
                assert_custody(&retained, &observer, raw_charge, controls_charge);
                for name in ["out0", "out1"] {
                    let handle = locals.get_item(name).unwrap().unwrap();
                    let output = handle.extract::<PyRef<'_, Output>>().unwrap();
                    assert!(Rc::ptr_eq(&output.inner.retained, &retained));
                    assert_eq!(output.inner.descriptor(), Ok(descriptor));
                }
                let handle = locals.get_item("a").unwrap().unwrap();
                let frame = handle.extract::<PyRef<'_, Frame>>().unwrap();
                assert!(Rc::ptr_eq(&frame.inner.retained, &retained));
            } // No Rust PyRef/strong Retention survives the final-owner check.

            locals.set_item("output_last", output_last).unwrap();
            run(
                py,
                r#"
last = out1 if output_last else a
del env, left, right, a, b, out0, out1
assert not last.is_valid()
"#,
                &locals,
            );
            assert_custody(
                &weak.upgrade().unwrap(),
                &observer,
                raw_charge,
                controls_charge,
            );
            run(py, "del last", &locals);
            assert!(weak.upgrade().is_none());
            assert_eq!(
                observer.budget().used_bytes(),
                OwnedBudget::resources().unwrap().initial_reserved_bytes
            );
            assert_eq!(
                crate::sigma_linear_activation_admission::require_python_allocation_contract(),
                Err(BridgeError::UnadmittedPython)
            );
            locals.clear();
        }
    });
}

#[test]
fn exact_python_plan_late_packaging_error_burns_sibling_but_escaped_output_keeps_custody() {
    let _serial = fixture::SERIAL.lock().unwrap();
    Python::initialize();
    Python::attach(|py| {
        let _types = initialize(py);
        let root = OwnedBudget::new(fixture::LIMIT).unwrap();
        let observer = root.try_clone().unwrap();
        let mut dealer = fixture::dealer(&root);
        let controls = Controls::reserve(&root, Fault::None).unwrap();
        let controls_charge = controls.charge.charged_bytes();
        let (pending, descriptor, raw_charge) = pending_from_python_plan(py, &root, &mut dealer);
        let weak = Rc::downgrade(&pending.retained);
        let objects = package_fixture(py, pending, controls, py.None().bind(py)).unwrap();
        let locals = PyDict::new(py);
        locals.set_item("owners", objects).unwrap();
        run(py, "env, left, right = owners\ndel owners", &locals);
        drop((dealer, root));
        run(py, START, &locals);
        run(py, READY, &locals);
        run(
            py,
            "out0 = left.finish(None)\nassert out0.is_valid()",
            &locals,
        );
        {
            let handle = locals.get_item("out0").unwrap().unwrap();
            let output = handle.extract::<PyRef<'_, Output>>().unwrap();
            assert_eq!(output.inner.descriptor(), Ok(descriptor));
        }
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
else: raise AssertionError('output packaging fault ignored')
assert env.is_aborted() and len(escaped) == 1
assert not out0.is_valid() and not escaped[0].is_valid()
assert not a.is_valid() and not b.is_valid()
last = escaped.pop()
del escaped, leak, env, left, right, a, b, out0
"#,
            &locals,
        );
        assert_custody(
            &weak.upgrade().unwrap(),
            &observer,
            raw_charge,
            controls_charge,
        );
        run(py, "del last", &locals);
        assert!(weak.upgrade().is_none());
        assert_eq!(
            observer.budget().used_bytes(),
            OwnedBudget::resources().unwrap().initial_reserved_bytes
        );
        locals.clear();
    });
}
