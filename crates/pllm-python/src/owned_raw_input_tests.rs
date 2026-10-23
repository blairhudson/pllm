use super::*;
use std::{cell::Cell, ffi::CString, panic::AssertUnwindSafe};

#[derive(Clone, Copy, Default, Debug)]
struct Observation {
    factories: usize,
    factory_counter: usize,
    factory_used: usize,
    erased_bytes: usize,
    erased_zero: bool,
    erasure_used: usize,
}

thread_local! {
    static OBSERVED: Cell<Observation> = const { Cell::new(Observation {
        factories: 0, factory_counter: 0, factory_used: 0,
        erased_bytes: 0, erased_zero: false, erasure_used: 0,
    }) };
}

pub(super) fn factory_entry(budget: &Budget) {
    OBSERVED.with(|slot| {
        let mut value = slot.get();
        value.factories += 1;
        value.factory_counter = std::ptr::from_ref(budget) as usize;
        value.factory_used = budget.used_bytes();
        slot.set(value);
    });
}

pub(super) fn scratch_retired(budget: &Budget, bytes: &[u8]) {
    OBSERVED.with(|slot| {
        let mut value = slot.get();
        value.erased_bytes = bytes.len();
        value.erased_zero = bytes.iter().all(|byte| *byte == 0);
        value.erasure_used = budget.used_bytes();
        slot.set(value);
    });
}

fn reset() {
    OBSERVED.with(|slot| slot.set(Observation::default()));
}

fn observed() -> Observation {
    OBSERVED.with(Cell::get)
}

fn limits() -> Limits {
    Limits {
        max_bytes: MAX_ALLOCATION_BYTES,
        max_work: u128::MAX,
    }
}

fn baseline() -> usize {
    OwnedBudget::resources().unwrap().initial_reserved_bytes
}

fn with_python<T>(body: impl for<'py> FnOnce(Python<'py>) -> T) -> T {
    Python::initialize();
    Python::attach(body)
}

#[test]
fn literal_work_and_shared_owner_inventory_includes_closed_producer_slots() {
    for length in [0, 1, 4095, 4096, 4097, 8193] {
        let quote = Resources::checked(length, limits()).unwrap();
        let chunk_visits = length.div_ceil(4096);
        assert_eq!(quote.work_bound, 3 * length as u128 + chunk_visits as u128);
        assert_eq!(quote.input_bytes, length);
        assert_eq!(quote.payload_bytes, quote.controls_bytes + length);
        assert_eq!(
            quote.reserved_bytes,
            quote.payload_bytes + size_of::<OwnedCharged<Vec<u8>>>()
        );
        assert_eq!(
            quote.controls_bytes,
            CONTROL_TERMS.iter().sum::<usize>() + 3 * size_of_val(&producer(&[]))
        );
        assert!(quote.controls_bytes > size_of::<Writer<'_>>());
    }
}

#[test]
fn full_original_reservation_precedes_factory_and_consuming_handoff_changes_no_charge() {
    with_python(|py| {
        // This caller-owned Rust/Python input is explicitly outside the ingress
        // reservation. Only the NEW native copy is admitted by this contract.
        let source: Vec<_> = (0..8193).map(|i| (i % 251) as u8).collect();
        let python = PyBytes::new(py, &source);
        let quote = Resources::checked(source.len(), limits()).unwrap();
        let root = OwnedBudget::new(baseline() + quote.reserved_bytes).unwrap();
        reset();
        let frozen = from_exact_bytes(&root, python.as_any(), limits()).unwrap();
        assert_eq!(frozen.bytes(), source);
        assert_eq!(frozen.copy_work_bound(), quote.work_bound);
        assert!(std::ptr::eq(frozen.budget(), root.budget()));
        assert_eq!(observed().factories, 1);
        assert_eq!(
            observed().factory_counter,
            std::ptr::from_ref(root.budget()) as usize
        );
        assert_eq!(observed().factory_used, baseline() + quote.reserved_bytes);
        let address = frozen.bytes().as_ptr();
        let original = frozen.into_original();
        assert_eq!(original.payload().as_ptr(), address);
        assert_eq!(original.payload().capacity(), source.len());
        assert_eq!(original.charged_bytes(), quote.reserved_bytes);
        assert_eq!(
            root.budget().used_bytes(),
            baseline() + quote.reserved_bytes
        );
        drop(original);
        assert_eq!(root.budget().used_bytes(), baseline());
    });
}

#[test]
fn empty_exact_and_one_short_refuse_before_first_buffer_factory() {
    with_python(|py| {
        let empty = PyBytes::new(py, b"");
        let no_room = OwnedBudget::new(baseline()).unwrap();
        reset();
        assert!(matches!(
            from_exact_bytes(&no_room, empty.as_any(), limits()),
            Err(Error::Native(AllocationError::Resource))
        ));
        assert_eq!(observed().factories, 0);
        assert_eq!(no_room.budget().used_bytes(), baseline());
        for length in [0, 1, 4096, 4097] {
            let source = PyBytes::new(py, &vec![71; length]);
            let quote = Resources::checked(length, limits()).unwrap();
            for short in [true, false] {
                let root = OwnedBudget::new(baseline() + quote.reserved_bytes - usize::from(short))
                    .unwrap();
                reset();
                let result = from_exact_bytes(&root, source.as_any(), limits());
                if short {
                    assert!(matches!(
                        result,
                        Err(Error::Native(AllocationError::Resource))
                    ));
                    assert_eq!(observed().factories, 0);
                    assert_eq!(root.budget().used_bytes(), baseline());
                } else {
                    assert_eq!(result.unwrap().bytes(), vec![71; length]);
                    assert_eq!(observed().factories, 1);
                    assert_eq!(root.budget().used_bytes(), baseline());
                }
            }
        }
    });
}

#[test]
fn source_byte_and_work_boundaries_are_checked_before_factory_with_fixed_error_precedence() {
    with_python(|py| {
        let source = PyBytes::new(py, &[19; 17]);
        let root = OwnedBudget::new(1 << 20).unwrap();
        let exact = Resources::checked(17, limits()).unwrap().work_bound;
        for (bound, expected) in [
            (
                Limits {
                    max_bytes: 16,
                    max_work: 0,
                },
                Error::ByteLimit,
            ),
            (
                Limits {
                    max_bytes: 17,
                    max_work: exact - 1,
                },
                Error::WorkLimit,
            ),
            (
                Limits {
                    max_bytes: usize::MAX,
                    max_work: u128::MAX,
                },
                Error::ByteLimit,
            ),
        ] {
            reset();
            assert!(
                matches!(from_exact_bytes(&root, source.as_any(), bound), Err(e) if e == expected)
            );
            assert_eq!(observed().factories, 0);
            assert_eq!(root.budget().used_bytes(), baseline());
        }
        let exact = Limits {
            max_bytes: 17,
            max_work: exact,
        };
        assert_eq!(
            from_exact_bytes(&root, source.as_any(), exact)
                .unwrap()
                .bytes(),
            &[19; 17]
        );
        assert_eq!(root.budget().used_bytes(), baseline());
    });
}

#[test]
fn exact_bytes_only_never_converts_exporter_subclass_length_or_integer_while_owner_live() {
    with_python(|py| {
        let code = CString::new(
            r#"hits = []
class Trap:
    def __bytes__(self): hits.append('bytes'); raise AssertionError('conversion')
    def __int__(self): hits.append('int'); raise AssertionError('conversion')
    def __index__(self): hits.append('index'); raise AssertionError('conversion')
    def __len__(self): hits.append('len'); raise AssertionError('conversion')
class Subclass(bytes):
    def __bytes__(self): hits.append('subclass'); raise AssertionError('conversion')
trap = Trap()
subclass = Subclass(b'bad')
array = bytearray(b'bad')
view = memoryview(b'bad')
"#,
        )
        .unwrap();
        let name = CString::new("owned_raw_input_traps").unwrap();
        let module = PyModule::from_code(py, &code, &name, &name).unwrap();
        let good = PyBytes::new(py, b"immutable");
        let root = OwnedBudget::new(1 << 20).unwrap();
        let held = from_exact_bytes(&root, good.as_any(), limits()).unwrap();
        let occupied = root.budget().used_bytes();
        for name in ["trap", "subclass", "array", "view"] {
            let value = module.getattr(name).unwrap();
            reset();
            assert!(matches!(
                from_exact_bytes(&root, &value, limits()),
                Err(Error::ExactBytes)
            ));
            assert_eq!(observed().factories, 0);
            assert_eq!(root.budget().used_bytes(), occupied);
            assert_eq!(held.bytes(), b"immutable");
        }
        assert_eq!(module.getattr("hits").unwrap().len().unwrap(), 0);
        drop(held);
        assert_eq!(root.budget().used_bytes(), baseline());
    });
}

#[test]
fn root_pressure_refuses_before_factory_and_never_substitutes_an_independent_counter() {
    with_python(|py| {
        let source = PyBytes::new(py, b"bytes");
        let quote = Resources::checked(5, limits()).unwrap();
        let root = OwnedBudget::new(baseline() + quote.reserved_bytes).unwrap();
        let foreign = OwnedBudget::new(1 << 20).unwrap();
        let pressure = root.charge_external(0).unwrap();
        reset();
        assert!(matches!(
            from_exact_bytes(&root, source.as_any(), limits()),
            Err(Error::Native(AllocationError::Resource))
        ));
        assert_eq!(observed().factories, 0);
        assert_eq!(foreign.budget().used_bytes(), baseline());
        drop(pressure);
        let input = from_exact_bytes(&root, source.as_any(), limits()).unwrap();
        assert!(std::ptr::eq(input.budget(), root.budget()));
        assert!(!std::ptr::eq(input.budget(), foreign.budget()));
    });
}

#[test]
fn writer_consuming_freeze_preserves_backing_and_refuses_underfill_overflow_and_empty_write() {
    let root = OwnedBudget::new(1 << 20).unwrap();
    let quote = Resources::checked(4, limits()).unwrap();
    for fault in 0..3 {
        reset();
        let result = root
            .charge_external(quote.payload_bytes)
            .unwrap()
            .try_build(|budget| {
                let mut writer = Writer::new(4, budget);
                writer.append(&[1, 2])?;
                match fault {
                    0 => {
                        writer.append(&[3, 4, 5])?;
                    }
                    1 => {
                        writer.append(&[])?;
                    }
                    _ => {}
                }
                writer.freeze()
            });
        assert!(matches!(result, Err(AllocationError::Resource)));
        assert_eq!(observed().erased_bytes, 4);
        assert!(observed().erased_zero);
        assert_eq!(observed().erasure_used, baseline() + quote.reserved_bytes);
        assert_eq!(root.budget().used_bytes(), baseline());
    }
    let before = Cell::new(std::ptr::null());
    let original = root
        .charge_external(quote.payload_bytes)
        .unwrap()
        .try_build(|budget| {
            let mut writer = Writer::new(4, budget);
            writer.append(&[1, 2])?;
            writer.append(&[3, 4])?;
            before.set(writer.bytes.as_ptr());
            writer.freeze()
        })
        .unwrap();
    assert_eq!(original.payload().as_ptr(), before.get());
    assert_eq!(original.payload().capacity(), 4);
    assert_eq!(original.payload(), &[1, 2, 3, 4]);
}

#[test]
fn partial_error_and_unwind_erase_scratch_under_original_full_charge() {
    for panic in [false, true] {
        let root = OwnedBudget::new(1 << 20).unwrap();
        let quote = Resources::checked(16, limits()).unwrap();
        reset();
        let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
            root.charge_external(quote.payload_bytes)
                .unwrap()
                .try_build(|budget| {
                    let mut writer = Writer::new(16, budget);
                    writer.append(&[91; 8])?;
                    if panic {
                        panic!("caller-free test injection after partial initialization");
                    }
                    Err::<Vec<u8>, _>(AllocationError::Context)
                })
        }));
        if panic {
            assert!(outcome.is_err());
        } else {
            assert!(matches!(outcome.unwrap(), Err(AllocationError::Context)));
        }
        assert_eq!(observed().erased_bytes, 16);
        assert!(observed().erased_zero);
        assert_eq!(observed().erasure_used, baseline() + quote.reserved_bytes);
        assert_eq!(root.budget().used_bytes(), baseline());
    }
}

#[test]
fn original_owned_counter_and_bytes_survive_python_source_and_root_handles() {
    let root = OwnedBudget::new(1 << 20).unwrap();
    let original = with_python(|py| {
        let source = PyBytes::new(py, b"original");
        from_exact_bytes(&root, source.as_any(), limits())
            .unwrap()
            .into_original()
    });
    let pointer = std::ptr::from_ref(root.budget());
    let occupied = root.budget().used_bytes();
    drop(root);
    assert!(std::ptr::eq(original.budget(), pointer));
    assert_eq!(
        original.budget().used_bytes(),
        occupied - size_of::<OwnedBudget>()
    );
    assert_eq!(original.payload(), b"original");
}

#[test]
fn existing_parsers_borrow_then_retain_original_bytes_without_an_import_surrogate() {
    with_python(|py| {
        // Pre-existing caller fixture, not a claimed preadmitted serializer.
        let graph = |mode: &str| {
            serde_json::json!({
                "mode": mode, "batch": 1, "query_sequence": 1,
                "maximum_key_sequence": 1, "operations": [],
                "state_inputs": [], "state_outputs": [], "output": "x"
            })
        };
        let source = serde_json::to_vec(&serde_json::json!({
            "schema_version": "pllm.decoder_plan.v1", "model_family": "fixture",
            "adapter": "original", "config_digest": "a".repeat(64),
            "transformations": [], "prefill": graph("prefill"), "decode": graph("decode"),
            "token_feedback": false
        }))
        .unwrap();
        let source = PyBytes::new(py, &source);
        let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let raw = from_exact_bytes(&root, source.as_any(), limits())
            .unwrap()
            .into_original();
        let address = raw.payload().as_ptr();
        let raw_charge = raw.charged_bytes();
        let parser = pllm_compiler::plan_input::Limits {
            max_source_bytes: 1 << 20,
            max_nodes: 1000,
            max_depth: 32,
            max_string_bytes: 1 << 16,
            max_number_bytes: 128,
            max_number_work: 1 << 20,
            max_work: 1 << 28,
            max_payload_bytes: 1 << 24,
        };
        let working = root
            .charge_external(pllm_compiler::plan_input::bootstrap_bytes().unwrap())
            .unwrap();
        let quote = pllm_compiler::plan_input::quote(&raw, working, parser).unwrap();
        let parsed = quote.construct().unwrap();
        assert!(std::ptr::eq(parsed.budget(), raw.budget()));
        assert_eq!(raw.payload().as_ptr(), address);
        assert_eq!(raw.charged_bytes(), raw_charge);
        assert_eq!(raw.payload().as_slice(), source.as_bytes());
        assert!(root.budget().used_bytes() >= baseline() + raw_charge + parsed.charged_bytes());
        drop(parsed);
        assert_eq!(root.budget().used_bytes(), baseline() + raw_charge);

        // B's model-neutral specialization consumes THIS SAME native byte owner.
        // Use its real immutable ImportedValue API, never a parallel parser or
        // caller-supplied census surrogate. The existing typed-parser branch
        // above remains an independent compatibility/borrowing control.
        let working = root
            .charge_external(pllm_compiler::plan_input::json::bootstrap_bytes().unwrap())
            .unwrap();
        let quote = pllm_compiler::plan_input::json::quote(raw, working, parser).unwrap();
        let counts = quote.counts();
        let resources = quote.resources();
        let imported = quote.construct().unwrap();
        let value = imported.payload();
        // SourceCounts intentionally has no PartialEq; use its public census
        // fields through ImportedValue::counts rather than changing B's type.
        assert_eq!(value.counts().spans, counts.spans);
        assert_eq!(value.counts().objects, counts.objects);
        assert_eq!(value.counts().objects, 3);
        assert_eq!(value.value()["model_family"], "fixture");
        assert_eq!(value.raw().as_ptr(), address);
        assert_eq!(value.raw(), source.as_bytes());
        assert_eq!(value.raw_owner().charged_bytes(), raw_charge);
        assert_eq!(
            value.raw_owner().payload().capacity(),
            source.as_bytes().len()
        );
        assert!(std::ptr::eq(value.budget(), root.budget()));
        assert!(std::ptr::eq(imported.budget(), root.budget()));
        assert_eq!(
            root.budget().used_bytes(),
            baseline() + resources.original_owners_peak_bytes
        );
        assert_eq!(resources.incoming_owner_bytes, raw_charge);
        drop(imported);
        assert_eq!(root.budget().used_bytes(), baseline());
    });
}
