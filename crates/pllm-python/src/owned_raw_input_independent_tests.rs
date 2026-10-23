//! Independent raw/Value ingress gates; no Python registration or native issuance.
//!
//! Source-only addition. Mount beneath `owned_raw_input`, not beneath its existing
//! `tests` child, when the parent opens the validation lane:
//! ```text
//! #[cfg(any(test, doctest))]
//! #[path = "owned_raw_input_independent_tests.rs"]
//! mod independent_regressions;
//! ```
//! `doctest` is intentional: placing these examples only beneath `cfg(test)`
//! would not exercise their public-type lifetime failures in ordinary rustdoc.
//!
//! The actual original byte owner cannot be destroyed while its bytes are used.
//! This fails on borrowing, not access to a private module or constructor:
//! ```compile_fail,E0505
//! use pllm_garble::shared_allocation::OwnedCharged;
//! fn release_while_borrowed(original: OwnedCharged<Vec<u8>>) {
//!     let bytes = original.payload().as_slice();
//!     drop(original);
//!     std::hint::black_box(bytes);
//! }
//! ```
//!
//! A consumed parser input cannot be reused, even when its quote might fail:
//! ```compile_fail,E0382
//! use pllm_compiler::plan_input::{json, Limits};
//! use pllm_garble::shared_allocation::{OwnedCharged, OwnedExternalCharge};
//! fn reuse_after_quote(
//!     original: OwnedCharged<Vec<u8>>,
//!     working: OwnedExternalCharge,
//!     limits: Limits,
//! ) {
//!     let quote = json::quote(original, working, limits);
//!     std::hint::black_box(original.payload());
//!     drop(quote);
//! }
//! ```
//!
//! An attached Python byte borrow cannot cross threads. Scoped threads remove
//! the unrelated `'static` requirement, isolating PyO3's attachment constraint:
//! ```compile_fail,E0277
//! use pyo3::{prelude::*, types::PyBytes};
//! fn move_attached_borrow(py: Python<'_>) {
//!     let source = PyBytes::new(py, b"external Python source");
//!     std::thread::scope(|scope| {
//!         scope.spawn(move || std::hint::black_box(source.as_bytes().len()));
//!     });
//! }
//! ```

#[cfg(test)]
mod cases {
    use super::super::*;
    use pllm_compiler::plan_input::{json, Error as ParseError, Limits as ParseLimits};
    use std::{ffi::CString, panic::AssertUnwindSafe};

    fn limits() -> Limits {
        Limits {
            max_bytes: MAX_ALLOCATION_BYTES,
            max_work: u128::MAX,
        }
    }

    fn parser_limits() -> ParseLimits {
        ParseLimits {
            max_source_bytes: 1 << 16,
            max_nodes: 1024,
            max_depth: 16,
            max_string_bytes: 4096,
            max_number_bytes: 128,
            max_number_work: 1 << 24,
            max_work: 1 << 28,
            max_payload_bytes: MAX_ALLOCATION_BYTES,
        }
    }

    fn with_python<T>(body: impl for<'py> FnOnce(Python<'py>) -> T) -> T {
        Python::initialize();
        Python::attach(body)
    }

    fn bootstrap(root: &OwnedBudget) -> OwnedExternalCharge {
        root.charge_external(json::bootstrap_bytes().unwrap())
            .unwrap()
    }

    #[test]
    fn exact_type_check_does_not_run_attribute_or_class_hooks_under_live_charge() {
        with_python(|py| {
            let code = CString::new(
                r#"hits = []
class Meta(type):
    def __instancecheck__(cls, other):
        hits.append('instancecheck')
        raise AssertionError('instancecheck')
class Object(metaclass=Meta):
    def __getattribute__(self, name):
        hits.append(name)
        raise AssertionError('attribute hook')
class BytesSubclass(bytes):
    def __getattribute__(self, name):
        hits.append(name)
        raise AssertionError('bytes subclass attribute hook')
plain = Object()
subclass = BytesSubclass(b'external')
"#,
            )
            .unwrap();
            let name = CString::new("raw_input_independent_hooks").unwrap();
            let module = PyModule::from_code(py, &code, &name, &name).unwrap();
            let root = OwnedBudget::new(1 << 20).unwrap();
            let source = PyBytes::new(py, b"live original owner");
            let held = from_exact_bytes(&root, source.as_any(), limits()).unwrap();
            let occupied = root.budget().used_bytes();
            for name in ["plain", "subclass"] {
                let candidate = module.getattr(name).unwrap();
                assert!(matches!(
                    from_exact_bytes(&root, &candidate, limits()),
                    Err(Error::ExactBytes)
                ));
                assert_eq!(root.budget().used_bytes(), occupied);
                assert_eq!(held.bytes(), b"live original owner");
            }
            assert_eq!(module.getattr("hits").unwrap().len().unwrap(), 0);
        });
    }

    #[test]
    fn bootstrap_refusal_precedes_type_byte_and_work_failures_without_inspection() {
        with_python(|py| {
            let baseline = OwnedBudget::resources().unwrap().initial_reserved_bytes;
            let root = OwnedBudget::new(baseline).unwrap();
            let not_bytes = py.None();
            let invalid_limits = Limits {
                max_bytes: usize::MAX,
                max_work: 0,
            };
            assert!(matches!(
                from_exact_bytes(&root, not_bytes.bind(py), invalid_limits),
                Err(Error::Native(AllocationError::Resource))
            ));
            assert_eq!(root.budget().used_bytes(), baseline);
        });
    }

    #[test]
    fn native_copy_never_aliases_python_backing_and_preserves_non_utf8_bytes() {
        let binary = b"\0\xff\x80\"\\\n\xfe";
        let root = OwnedBudget::new(1 << 20).unwrap();
        let original = with_python(|py| {
            let source = PyBytes::new(py, binary);
            let frozen = from_exact_bytes(&root, source.as_any(), limits()).unwrap();
            assert_ne!(frozen.bytes().as_ptr(), source.as_bytes().as_ptr());
            assert_eq!(frozen.bytes(), binary);
            frozen.into_original()
        });
        assert_eq!(original.payload().capacity(), binary.len());
        assert_eq!(original.payload().as_slice(), binary);
        assert!(std::ptr::eq(original.budget(), root.budget()));
    }

    #[test]
    fn interleaved_real_counters_keep_other_inputs_live_and_refusal_is_reusable() {
        with_python(|py| {
            let baseline = OwnedBudget::resources().unwrap().initial_reserved_bytes;
            let source = PyBytes::new(py, b"two independent accounting roots");
            let quote = Resources::checked(source.as_bytes().len(), limits()).unwrap();
            let left = OwnedBudget::new(baseline + 2 * quote.reserved_bytes).unwrap();
            let right = OwnedBudget::new(baseline + 2 * quote.reserved_bytes).unwrap();
            let a = from_exact_bytes(&left, source.as_any(), limits()).unwrap();
            let b = from_exact_bytes(&right, source.as_any(), limits()).unwrap();
            assert!(!std::ptr::eq(a.budget(), b.budget()));
            assert_ne!(a.bytes().as_ptr(), b.bytes().as_ptr());
            let pressure = left.charge_external(0).unwrap();
            let occupied = left.budget().used_bytes();
            assert!(matches!(
                from_exact_bytes(&left, source.as_any(), limits()),
                Err(Error::Native(AllocationError::Resource))
            ));
            assert_eq!(left.budget().used_bytes(), occupied);
            let b2 = from_exact_bytes(&right, source.as_any(), limits()).unwrap();
            assert_eq!(
                right.budget().used_bytes(),
                baseline + 2 * quote.reserved_bytes
            );
            assert_eq!(a.bytes(), source.as_bytes());
            assert_eq!(b.bytes(), source.as_bytes());
            drop(pressure);
            let a2 = from_exact_bytes(&left, source.as_any(), limits()).unwrap();
            assert_eq!(
                left.budget().used_bytes(),
                baseline + 2 * quote.reserved_bytes
            );
            drop((a, a2, b, b2));
            assert_eq!(left.budget().used_bytes(), baseline);
            assert_eq!(right.budget().used_bytes(), baseline);
        });
    }

    #[test]
    fn pure_rust_owner_crosses_thread_after_gil_without_rebinding_or_python_references() {
        fn send_sync<T: Send + Sync>() {}
        send_sync::<FrozenInput>();
        send_sync::<OwnedCharged<json::ImportedValue>>();

        let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let observer = root.try_clone().unwrap();
        let frozen = with_python(|py| {
            let source = PyBytes::new(py, br#"{"literal":[1,true,null],"label":"native"}"#);
            from_exact_bytes(&root, source.as_any(), limits()).unwrap()
        });
        let pointer = frozen.bytes().as_ptr() as usize;
        let counter = std::ptr::from_ref(frozen.budget()) as usize;
        let raw_charge = frozen.original.charged_bytes();
        let copy_work = frozen.copy_work_bound();
        let (imported, resources, combined_work) = std::thread::spawn(move || {
            // There is no Python object/token in this closure and no attach.
            let original = frozen.into_original();
            assert_eq!(original.payload().as_ptr() as usize, pointer);
            assert_eq!(std::ptr::from_ref(original.budget()) as usize, counter);
            let quoted = json::quote(original, bootstrap(&root), parser_limits()).unwrap();
            let resources = quoted.resources();
            let combined_work = copy_work.checked_add(resources.work.total).unwrap();
            let imported = quoted.construct().unwrap();
            drop(root);
            (imported, resources, combined_work)
        })
        .join()
        .unwrap();
        assert_eq!(imported.payload().raw().as_ptr() as usize, pointer);
        assert_eq!(imported.payload().raw_owner().charged_bytes(), raw_charge);
        assert_eq!(resources.incoming_owner_bytes, raw_charge);
        assert_eq!(imported.payload().counts().objects, 1);
        assert_eq!(imported.payload().value()["label"], "native");
        assert_eq!(combined_work, copy_work + resources.work.total);
        assert!(std::ptr::eq(imported.budget(), observer.budget()));
        let baseline = OwnedBudget::resources().unwrap().initial_reserved_bytes;
        assert_eq!(
            observer.budget().used_bytes(),
            baseline + resources.original_owners_peak_bytes
        );
        drop(imported);
        assert_eq!(observer.budget().used_bytes(), baseline);
    }

    #[test]
    fn consuming_raw_quote_and_number_failures_leave_unrelated_original_owner_resident() {
        with_python(|py| {
            let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
            let source = PyBytes::new(py, b"unrelated live bytes");
            let held = from_exact_bytes(&root, source.as_any(), limits()).unwrap();
            let occupied = root.budget().used_bytes();
            for (bytes, quote_failure) in [
                (br#"{"prefix":[1,2,3],"late":[1,]}"#.as_slice(), true),
                (br#"{"prefix":[1,2,3],"late":1e9999}"#.as_slice(), false),
            ] {
                let source = PyBytes::new(py, bytes);
                let original = from_exact_bytes(&root, source.as_any(), limits())
                    .unwrap()
                    .into_original();
                let result = json::quote(original, bootstrap(&root), parser_limits());
                if quote_failure {
                    assert!(matches!(result, Err(ParseError::Syntax)));
                } else {
                    assert!(matches!(
                        result.unwrap().construct(),
                        Err(ParseError::InvalidNumber)
                    ));
                }
                assert_eq!(root.budget().used_bytes(), occupied);
                assert_eq!(held.bytes(), b"unrelated live bytes");
            }
            let source = PyBytes::new(py, b"null");
            let original = from_exact_bytes(&root, source.as_any(), limits())
                .unwrap()
                .into_original();
            let output = json::quote(original, bootstrap(&root), parser_limits())
                .unwrap()
                .construct()
                .unwrap();
            assert!(output.payload().value().is_null());
            drop(output);
            assert_eq!(root.budget().used_bytes(), occupied);
        });
    }

    #[test]
    fn parser_quote_unwind_preserves_primary_panic_and_drops_only_its_original_owners() {
        #[derive(Debug, PartialEq)]
        struct Primary;

        let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let held = with_python(|py| {
            from_exact_bytes(&root, PyBytes::new(py, b"held").as_any(), limits()).unwrap()
        });
        let occupied = root.budget().used_bytes();
        let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let original = with_python(|py| {
                from_exact_bytes(&root, PyBytes::new(py, b"[1,2,3]").as_any(), limits())
                    .unwrap()
                    .into_original()
            });
            let _quoted = json::quote(original, bootstrap(&root), parser_limits()).unwrap();
            std::panic::panic_any(Primary);
        }));
        let error = outcome.unwrap_err();
        assert_eq!(error.downcast_ref::<Primary>(), Some(&Primary));
        assert_eq!(root.budget().used_bytes(), occupied);
        assert_eq!(held.bytes(), b"held");
    }

    #[test]
    fn successful_raw_and_value_owners_do_not_mint_sigma_authority_or_open_python_gate() {
        use pllm_compiler::sigma_linear_activation::{OwnedSigmaInput, OwnedSigmaInputRetention};

        macro_rules! no_into {
            ($from:ty, $to:ty) => {{
                trait Ambiguous<A> {
                    fn check() {}
                }
                impl<T> Ambiguous<()> for T {}
                struct Converted;
                impl<T: Into<$to>> Ambiguous<Converted> for T {}
                let _ = <$from as Ambiguous<_>>::check;
            }};
        }
        no_into!(FrozenInput, OwnedSigmaInput);
        no_into!(OwnedCharged<json::ImportedValue>, OwnedSigmaInput);
        no_into!(OwnedCharged<json::ImportedValue>, OwnedSigmaInputRetention);

        let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let original = with_python(|py| {
            from_exact_bytes(&root, PyBytes::new(py, b"{}").as_any(), limits())
                .unwrap()
                .into_original()
        });
        let imported = json::quote(original, bootstrap(&root), parser_limits())
            .unwrap()
            .construct()
            .unwrap();
        let occupied = root.budget().used_bytes();
        assert!(matches!(
            crate::sigma_linear_activation_admission::require_python_allocation_contract(),
            Err(crate::sigma_linear_activation::BridgeError::UnadmittedPython)
        ));
        assert_eq!(root.budget().used_bytes(), occupied);
        assert!(imported.payload().value().is_object());
    }
}
