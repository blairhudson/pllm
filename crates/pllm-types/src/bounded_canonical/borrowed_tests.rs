use super::*;
use serde_json::{json, Map, Number};

fn limits() -> Limits {
    Limits {
        max_nodes: 10_000,
        max_depth: 32,
        max_bytes: 1_000_000,
        max_string_bytes: 100_000,
        max_key_bytes: 10_000,
        max_work: 10_000_000,
        max_workspace_bytes: usize::MAX,
    }
}

fn census(value: &Value, limits: Limits) -> Result<Usage, Error> {
    let mut frames = [Frame::empty(); 32];
    Ok(preflight(value, &mut frames, limits)?.usage())
}

fn equal<'a>(a: &'a Value, b: &'a Value, limits: Limits) -> Result<Equality<'a>, Error> {
    let mut left = [Frame::empty(); 32];
    let mut right = [Frame::empty(); 32];
    structural_eq(a, b, &mut left, &mut right, limits)
}

// Deliberately independent, recursive TEST-ONLY oracle, with ordinary owned
// scratch and serde formatting. Neither recursion nor allocation is production.
fn oracle<'a>(value: &'a Value, depth: usize, out: &mut Vec<Event<'a>>) {
    out.push(Event::Value {
        value,
        depth,
        number_bytes: match value {
            Value::Number(n) => n.to_string().len(),
            _ => 0,
        },
    });
    match value {
        Value::Array(values) => {
            for value in values {
                oracle(value, depth + 1, out);
            }
        }
        Value::Object(values) => {
            let mut sorted: Vec<_> = values.iter().collect();
            sorted.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
            for (key, value) in sorted {
                out.push(Event::Key { key, depth });
                oracle(value, depth + 1, out);
            }
        }
        _ => {}
    }
}

#[test]
fn borrowed_events_match_independent_traversal_and_original_owners() {
    let value = json!({"z": [null, {"β": "λ\0\n", "a": -0.0}], "a": [1, true, "🦀"]});
    let mut expected = Vec::new();
    oracle(&value, 1, &mut expected);
    let mut frames = [Frame::empty(); 32];
    let mut traversal = Traversal::new(&value, &mut frames, limits()).unwrap();
    let mut count = 0;
    let mut nodes = 0;
    let mut depth = 0;
    let mut bytes = 0;
    while let Some(actual) = traversal.next_event().unwrap() {
        match (actual, expected[count]) {
            (
                Event::Value {
                    value: a,
                    depth: ad,
                    number_bytes: an,
                },
                Event::Value {
                    value: b,
                    depth: bd,
                    number_bytes: bn,
                },
            ) => {
                assert!(std::ptr::eq(a, b));
                assert_eq!((ad, an), (bd, bn));
                nodes += 1;
                depth = depth.max(ad);
                bytes += match a {
                    Value::String(s) => s.len(),
                    Value::Number(_) => an,
                    _ => 0,
                };
            }
            (Event::Key { key: a, depth: ad }, Event::Key { key: b, depth: bd }) => {
                assert!(std::ptr::eq(a, b));
                assert_eq!(ad, bd);
                bytes += a.len();
            }
            _ => panic!("event kind/order mismatch"),
        }
        count += 1;
    }
    assert_eq!(count, expected.len());
    assert_eq!(traversal.usage().nodes, nodes);
    assert_eq!(traversal.usage().depth, depth);
    assert_eq!(traversal.usage().bytes, bytes);
    assert!(traversal.next_event().unwrap().is_none());
}

#[test]
fn borrowed_structural_equality_matches_serde_not_canonical_encoding() {
    let mut corpus = vec![
        Value::Null,
        json!(false),
        json!(true),
        json!(0),
        json!(1),
        json!(-1),
        json!(i64::MIN),
        json!(u64::MAX),
        json!(1.0),
        json!(-0.0),
        json!(0.0),
        json!(1e-200),
        json!(1e200),
        json!(9_007_199_254_740_993_u64),
        json!(9_007_199_254_740_992_f64),
        json!(""),
        json!("é"),
        json!("e\u{301}"),
        json!("\0\n🦀"),
        json!([]),
        json!({}),
        json!([1, 2]),
        json!([2, 1]),
        json!({"a": [1, {"x": -0.0}], "β": true}),
        json!({"a": [1, {"x": 0.0}], "β": true}),
        json!({"a": [1.0, {"x": 0.0}], "β": true}),
    ];
    // Different insertion orders, independent of the resolved Map backend.
    for reverse in [false, true] {
        let mut map = Map::new();
        let entries = [("a", json!([null, 1])), ("b", json!({"x": "é"}))];
        for index in if reverse { [1, 0] } else { [0, 1] } {
            map.insert(entries[index].0.into(), entries[index].1.clone());
        }
        corpus.push(Value::Object(map));
    }
    for a in &corpus {
        for b in &corpus {
            let actual = equal(a, b, limits()).unwrap();
            assert_eq!(actual.equal, a == b, "left={a:?}, right={b:?}");
        }
    }
    // Tripwires for the current pinned default Number feature contract.
    assert_ne!(json!(1), json!(1.0));
    assert_eq!(json!(-0.0), json!(0.0));
    assert!(equal(&json!(-0.0), &json!(0.0), limits()).unwrap().equal);
    assert_ne!(
        crate::canonical_bytes(&json!(-0.0)),
        crate::canonical_bytes(&json!(0.0))
    );
    assert!(Number::from_f64(f64::NAN).is_none());
    assert!(Number::from_f64(f64::INFINITY).is_none());
}

#[test]
fn borrowed_exact_and_one_short_node_depth_byte_work_and_workspace_limits() {
    let value = json!({"aa": ["long", {"b": 1.25}]});
    let used = census(&value, limits()).unwrap();
    let quote = Traversal::resources_for_slots(32).unwrap();
    let exact = Limits {
        max_nodes: used.nodes,
        max_depth: used.depth,
        max_bytes: used.bytes,
        max_string_bytes: 4,
        max_key_bytes: 2,
        max_work: used.work,
        max_workspace_bytes: quote.logical_workspace_bytes,
    };
    assert_eq!(census(&value, exact).unwrap(), used);
    for (bounded, expected) in [
        (
            Limits {
                max_nodes: used.nodes - 1,
                ..exact
            },
            Error::NodeLimit,
        ),
        (
            Limits {
                max_depth: used.depth - 1,
                ..exact
            },
            Error::DepthLimit,
        ),
        (
            Limits {
                max_bytes: used.bytes - 1,
                ..exact
            },
            Error::ByteLimit,
        ),
        (
            Limits {
                max_work: used.work - 1,
                ..exact
            },
            Error::WorkLimit,
        ),
        (
            Limits {
                max_string_bytes: 3,
                ..exact
            },
            Error::StringLimit,
        ),
        (
            Limits {
                max_key_bytes: 1,
                ..exact
            },
            Error::KeyLimit,
        ),
        (
            Limits {
                max_workspace_bytes: quote.logical_workspace_bytes - 1,
                ..exact
            },
            Error::WorkspaceLimit,
        ),
    ] {
        assert_eq!(census(&value, bounded).unwrap_err(), expected);
    }
}

#[test]
fn borrowed_equality_aggregate_exact_short_and_false_never_bypasses_preflight() {
    let a = json!({"a": [-0.0, "é"], "b": [true]});
    let b = json!({"b": [true], "a": [0.0, "é"]});
    let used = equal(&a, &b, limits()).unwrap();
    assert!(used.equal);
    let quote = equality_resources_for_slots(32, 32).unwrap();
    let exact = Limits {
        max_nodes: used.usage.nodes,
        max_depth: used.usage.depth,
        max_bytes: used.usage.bytes,
        max_work: used.usage.work,
        max_workspace_bytes: quote.logical_workspace_bytes,
        ..limits()
    };
    let replay = equal(&a, &b, exact).unwrap();
    assert_eq!(replay.equal, used.equal);
    assert_eq!(replay.usage, used.usage);
    assert!(std::ptr::eq(replay.left(), &a));
    assert!(std::ptr::eq(replay.right(), &b));
    for (bounded, error) in [
        (
            Limits {
                max_nodes: exact.max_nodes - 1,
                ..exact
            },
            Error::NodeLimit,
        ),
        (
            Limits {
                max_depth: exact.max_depth - 1,
                ..exact
            },
            Error::DepthLimit,
        ),
        (
            Limits {
                max_bytes: exact.max_bytes - 1,
                ..exact
            },
            Error::ByteLimit,
        ),
        (
            Limits {
                max_work: exact.max_work - 1,
                ..exact
            },
            Error::WorkLimit,
        ),
        (
            Limits {
                max_workspace_bytes: exact.max_workspace_bytes - 1,
                ..exact
            },
            Error::WorkspaceLimit,
        ),
    ] {
        assert_eq!(equal(&a, &b, bounded).unwrap_err(), error);
    }
    // A different root kind must not short-circuit away the right input's cap.
    assert_eq!(
        equal(
            &Value::Null,
            &json!([[null]]),
            Limits {
                max_depth: 2,
                ..limits()
            }
        )
        .unwrap_err(),
        Error::DepthLimit
    );
    assert_eq!(
        equal(
            &Value::Null,
            &json!(["oversize"]),
            Limits {
                max_bytes: 7,
                ..limits()
            }
        )
        .unwrap_err(),
        Error::ByteLimit
    );
    assert_eq!(
        equal(
            &Value::Null,
            &json!([null]),
            Limits {
                max_nodes: 2,
                ..limits()
            }
        )
        .unwrap_err(),
        Error::NodeLimit
    );
}

#[test]
fn borrowed_resources_price_all_supplied_slots_and_concrete_state() {
    let one = Traversal::resources_for_slots(1).unwrap();
    let two = Traversal::resources_for_slots(2).unwrap();
    assert_eq!(two.frame_bytes - one.frame_bytes, Frame::storage_bytes());
    assert_eq!(
        two.logical_workspace_bytes - one.logical_workspace_bytes,
        Frame::storage_bytes()
    );
    assert_eq!(one.dependency_call_stack_bytes, None);
    assert_eq!(one.dependency_buffer_bytes, None);
    assert!(one.state_bytes >= size_of::<Cursor<'_, '_>>() + size_of::<Meter>());
    assert!(one.state_bytes >= size_of::<serde_json::map::Iter<'_>>());
    let pair = equality_resources_for_slots(2, 3).unwrap();
    assert_eq!(pair.frame_bytes, 5 * Frame::storage_bytes());
    assert_eq!(pair.state_bytes, size_of::<EqualityState<'_, '_>>());
    assert_eq!(
        pair.operation_result_bytes,
        size_of::<Result<Equality<'_>, Error>>()
    );
    assert_eq!(
        pair.visit_result_bytes,
        size_of::<Result<Option<Event<'_>>, Error>>()
    );
    assert_eq!(
        Traversal::resources_for_slots(usize::MAX),
        Err(Error::Overflow)
    );
    assert_eq!(
        equality_resources_for_slots(usize::MAX, 1),
        Err(Error::Overflow)
    );
    assert_eq!(add(usize::MAX, 1), Err(Error::Overflow));
    let mut meter = Meter {
        limits: limits(),
        usage: Usage {
            work: usize::MAX,
            ..Usage::default()
        },
    };
    assert_eq!(meter.work(1), Err(Error::Overflow));
    assert_eq!(meter.usage.work, usize::MAX);
    meter.usage.bytes = usize::MAX;
    assert_eq!(meter.bytes(1), Err(Error::Overflow));
    assert_eq!(meter.usage.bytes, usize::MAX);
}

#[test]
fn borrowed_short_stack_depth_zero_and_errors_poison_without_success() {
    let value = json!([[null]]);
    let mut short = [Frame::empty(); 2];
    let mut walk = Traversal::new(&value, &mut short, limits()).unwrap();
    assert!(walk.next_event().unwrap().is_some());
    assert!(walk.next_event().unwrap().is_some());
    assert_eq!(walk.next_event().unwrap_err(), Error::WorkspaceTooSmall);
    assert_eq!(walk.next_event().unwrap_err(), Error::Poisoned);
    let mut empty = [];
    assert!(matches!(
        Traversal::new(&value, &mut empty, limits()),
        Err(Error::WorkspaceTooSmall)
    ));
    assert_eq!(
        census(
            &Value::Null,
            Limits {
                max_depth: 0,
                ..limits()
            }
        ),
        Err(Error::DepthLimit)
    );
    assert_eq!(
        census(
            &Value::Null,
            Limits {
                max_nodes: 0,
                ..limits()
            }
        ),
        Err(Error::NodeLimit)
    );
    assert_eq!(
        census(
            &Value::Null,
            Limits {
                max_work: 0,
                ..limits()
            }
        ),
        Err(Error::WorkLimit)
    );
}

#[test]
fn borrowed_workspace_refusal_precedes_frame_mutation_and_counter_overflow_is_typed() {
    let sentinel = json!(true);
    let root = json!([null]);
    let mut frames = [Frame::empty(); 2];
    frames[0].value = Some(&sentinel);
    let quote = Traversal::resources_for_slots(frames.len()).unwrap();
    assert!(matches!(
        Traversal::new(
            &root,
            &mut frames,
            Limits {
                max_workspace_bytes: quote.logical_workspace_bytes - 1,
                ..limits()
            }
        ),
        Err(Error::WorkspaceLimit)
    ));
    assert!(std::ptr::eq(frames[0].value.unwrap(), &sentinel));
    let mut cursor = Cursor::new(&root, Workspace::Sorted(&mut frames)).unwrap();
    let mut meter = Meter {
        limits: Limits {
            max_nodes: usize::MAX,
            ..limits()
        },
        usage: Usage {
            nodes: usize::MAX,
            ..Usage::default()
        },
    };
    assert!(matches!(cursor.next(&mut meter), Err(Error::Overflow)));
    assert_eq!(meter.usage.nodes, usize::MAX);
}

#[test]
fn borrowed_depth_boundary_and_reuse_after_input_scope() {
    let mut value = Value::Null;
    for _ in 0..31 {
        value = Value::Array(vec![value]);
    }
    assert_eq!(census(&value, limits()).unwrap().depth, 32);
    value = Value::Array(vec![value]);
    assert_eq!(census(&value, limits()), Err(Error::DepthLimit));

    let mut original = json!({"a": 1});
    {
        let mut frames = [Frame::empty(); 4];
        let admitted = preflight(&original, &mut frames, limits()).unwrap();
        assert!(std::ptr::eq(admitted.value(), &original));
        let again = preflight(admitted.value(), &mut frames, limits()).unwrap();
        assert_eq!(again.usage(), admitted.usage());
    }
    // Immutable preflight and all borrowed frame storage ended before mutation.
    original["a"] = json!(2);
    assert_eq!(original["a"], 2);
}

#[test]
fn borrowed_wide_object_selection_and_long_key_comparisons_are_metered() {
    let mut map = Map::new();
    for i in 0..16 {
        map.insert(
            format!("{}{:02}", "shared-prefix-".repeat(8), i),
            Value::Null,
        );
    }
    let value = Value::Object(map);
    let usage = census(&value, limits()).unwrap();
    assert!(usage.work > 16 * 16);
    assert_eq!(
        census(
            &value,
            Limits {
                max_work: usage.work - 1,
                ..limits()
            }
        ),
        Err(Error::WorkLimit)
    );
    assert_eq!(
        equal(
            &value,
            &value,
            Limits {
                max_work: usage.work,
                ..limits()
            }
        )
        .unwrap_err(),
        Error::WorkLimit
    );
}

#[test]
fn borrowed_native_wide_objects_visit_each_entry_once_with_linear_work() {
    for count in [0, 1, 2, 16, 64, 256] {
        let mut map = Map::new();
        for i in 0..count {
            map.insert(format!("key.{i:04}"), Value::Null);
        }
        let root = Value::Object(map);
        let mut native = [const { NativeFrame::empty() }; 4];
        let (used, keys, values) = {
            let mut walk = Traversal::new_native(&root, &mut native, limits()).unwrap();
            let mut keys = 0;
            let mut values = 0;
            while let Some(event) = walk.next_event().unwrap() {
                match event {
                    Event::Key { .. } => keys += 1,
                    Event::Value { .. } => values += 1,
                }
            }
            (walk.usage(), keys, values)
        };
        assert_eq!(keys, count);
        assert_eq!(values, count + 1);
        assert_eq!(used.object_candidates, count);
        // Object entry: select/candidate, push, child-enter, child-pop; plus
        // root-enter and final exhaustion. No key prefix comparisons in native.
        assert_eq!(used.work, 5 * count + 2);
        let sorted = census(&root, limits()).unwrap();
        assert_eq!(sorted.object_candidates, count * (count + 1));
        assert_eq!(used.nodes, sorted.nodes);
        assert_eq!(used.bytes, sorted.bytes);
        let quote = Traversal::resources_for_native_slots(4).unwrap();
        let exact = Limits {
            max_nodes: used.nodes,
            max_depth: used.depth,
            max_bytes: used.bytes,
            max_work: used.work,
            max_workspace_bytes: quote.logical_workspace_bytes,
            ..limits()
        };
        assert_eq!(
            preflight_native(&root, &mut native, exact).unwrap().usage(),
            used
        );
        assert_eq!(
            preflight_native(
                &root,
                &mut native,
                Limits {
                    max_work: used.work - 1,
                    ..exact
                }
            )
            .unwrap_err(),
            Error::WorkLimit
        );
    }
}

#[test]
fn borrowed_native_nested_iterators_retain_original_entries_and_depth() {
    fn reference<'a>(value: &'a Value, depth: usize, out: &mut Vec<Event<'a>>) {
        out.push(Event::Value {
            value,
            depth,
            number_bytes: 0,
        });
        match value {
            Value::Array(values) => {
                for value in values {
                    reference(value, depth + 1, out);
                }
            }
            Value::Object(values) => {
                for (key, value) in values {
                    out.push(Event::Key { key, depth });
                    reference(value, depth + 1, out);
                }
            }
            _ => {}
        }
    }
    let root = json!({"z": [{"nested": [null, true]}, false], "a": {"β": "é"}});
    let mut expected = Vec::new();
    reference(&root, 1, &mut expected);
    let mut frames = [const { NativeFrame::empty() }; 8];
    let mut walk = Traversal::new_native(&root, &mut frames, limits()).unwrap();
    for expected in expected {
        match (walk.next_event().unwrap().unwrap(), expected) {
            (
                Event::Value {
                    value: a,
                    depth: ad,
                    ..
                },
                Event::Value {
                    value: b,
                    depth: bd,
                    ..
                },
            ) => {
                assert!(std::ptr::eq(a, b));
                assert_eq!(ad, bd);
            }
            (Event::Key { key: a, depth: ad }, Event::Key { key: b, depth: bd }) => {
                assert!(std::ptr::eq(a, b));
                assert_eq!(ad, bd);
            }
            _ => panic!("native event mismatch"),
        }
    }
    assert!(walk.next_event().unwrap().is_none());
}

#[test]
fn borrowed_quoted_native_mode_guards_actual_iterator_layout_before_array_construction() {
    fn fixed<const N: usize>(
        root: &Value,
        limits: Limits,
        constructed: &mut bool,
    ) -> Result<Usage, Error> {
        let quote = Traversal::resources_for_native_slots(N)?;
        quote.check(limits)?; // required BEFORE caller's fixed array constructor
        *constructed = true;
        let mut frames = [const { NativeFrame::empty() }; N];
        Ok(preflight_native(root, &mut frames, limits)?.usage())
    }
    let quote = Traversal::resources_for_native_slots(8).unwrap();
    let sorted = Traversal::resources_for_slots(8).unwrap();
    assert_eq!(quote.order, Order::Native);
    assert_eq!(sorted.order, Order::Sorted);
    assert_eq!(quote.frame_size, size_of::<NativeFrame<'_>>());
    assert_eq!(
        quote.frame_alignment,
        std::mem::align_of::<NativeFrame<'_>>()
    );
    assert_eq!(quote.frame_bytes, 8 * size_of::<NativeFrame<'_>>());
    assert!(
        quote.frame_size >= size_of::<Frame<'_>>() + size_of::<Option<serde_json::map::Iter<'_>>>()
    );
    assert!(quote.logical_workspace_bytes > sorted.logical_workspace_bytes);
    let root = json!({"a": [null]});
    let mut constructed = false;
    assert_eq!(
        fixed::<8>(
            &root,
            Limits {
                max_workspace_bytes: quote.logical_workspace_bytes - 1,
                ..limits()
            },
            &mut constructed
        ),
        Err(Error::WorkspaceLimit)
    );
    assert!(!constructed);
    assert!(fixed::<8>(
        &root,
        Limits {
            max_workspace_bytes: quote.logical_workspace_bytes,
            ..limits()
        },
        &mut constructed
    )
    .is_ok());
    assert!(constructed);
    let mut actual_native = [const { NativeFrame::empty() }; 8];
    assert!(matches!(
        Traversal::new_native(
            &root,
            &mut actual_native,
            Limits {
                max_workspace_bytes: sorted.logical_workspace_bytes,
                ..limits()
            }
        ),
        Err(Error::WorkspaceLimit)
    ));
    assert_eq!(
        Traversal::resources_for_native_slots(usize::MAX),
        Err(Error::Overflow)
    );
}

#[test]
fn borrowed_phase_inventory_prices_each_transition_result_argument_and_event_copy() {
    let quote = equality_resources_for_slots(4, 4).unwrap();
    assert_eq!(
        quote.cursor_result_bytes,
        size_of::<Result<Cursor<'_, '_>, Error>>()
    );
    assert_eq!(
        quote.ordering_result_bytes,
        size_of::<Result<Ordering, Error>>()
    );
    assert_eq!(
        quote.iterator_entry_result_bytes,
        size_of::<Option<(&String, &Value)>>()
    );
    assert_eq!(
        quote.optional_event_pair_bytes,
        size_of::<(Option<Event<'_>>, Option<Event<'_>>)>()
    );
    assert_eq!(quote.event_pair_bytes, size_of::<(Event<'_>, Event<'_>)>());
    assert_eq!(
        quote.same_event_arguments_bytes,
        size_of::<(Event<'_>, Event<'_>, &mut Meter)>()
    );
    assert_eq!(quote.format_arguments_bytes, 0);
    assert_eq!(quote.format_argument_backing_bytes, 0); // path eliminated, not guessed
    assert_eq!(
        quote.formatter_write_result_bytes,
        size_of::<io::Result<usize>>()
    );
    assert_eq!(quote.formatter_error_bytes, size_of::<io::Error>());
    assert_eq!(
        quote.logical_workspace_bytes,
        quote.frame_bytes + quote.state_bytes + quote.phases.maximum()
    );
    assert_eq!(
        quote.phases.construction,
        2 * quote.cursor_result_bytes
            + 2 * quote.constructor_result_bytes
            + quote.workspace_argument_bytes
            + quote.frame_transition_bytes
            + quote.unit_result_bytes
    );
    assert_eq!(
        quote.phases.equality,
        quote.optional_event_pair_bytes
            + 2 * quote.event_pair_bytes
            + quote.same_event_arguments_bytes
            + 2 * quote.leaf_result_bytes
            + quote.comparison_arguments_bytes
            + 2 * quote.ordering_result_bytes
            + quote.unit_result_bytes
            + quote.arithmetic_result_bytes
    );
    assert_eq!(quote.dependency_buffer_bytes, None);
    assert_eq!(quote.dependency_call_stack_bytes, None);
    assert_eq!(times(usize::MAX, 2), Err(Error::Overflow));
}

#[test]
fn borrowed_workspaces_reuse_after_error_and_different_root_in_both_modes() {
    let old = json!({"a": {"x": [null]}, "z": true});
    let new = json!({"new": [false, {"β": "é"}]});
    let mut sorted = [Frame::empty(); 8];
    assert_eq!(
        preflight(
            &old,
            &mut sorted,
            Limits {
                max_work: 5,
                ..limits()
            }
        )
        .unwrap_err(),
        Error::WorkLimit
    );
    let actual = preflight(&new, &mut sorted, limits()).unwrap();
    assert!(std::ptr::eq(actual.value(), &new));
    assert_eq!(actual.usage(), census(&new, limits()).unwrap());
    let mut native = [const { NativeFrame::empty() }; 8];
    assert_eq!(
        preflight_native(
            &old,
            &mut native,
            Limits {
                max_work: 5,
                ..limits()
            }
        )
        .unwrap_err(),
        Error::WorkLimit
    );
    let reused = preflight_native(&new, &mut native, limits()).unwrap();
    let mut fresh = [const { NativeFrame::empty() }; 8];
    assert_eq!(
        reused.usage(),
        preflight_native(&new, &mut fresh, limits())
            .unwrap()
            .usage()
    );
    assert!(std::ptr::eq(reused.value(), &new));
    // Nested per-depth iterators must also be reset on a second successful root.
    let again = preflight_native(&old, &mut native, limits()).unwrap();
    assert!(std::ptr::eq(again.value(), &old));
}

#[test]
fn borrowed_empty_and_short_right_equality_workspace_return_typed_errors_and_reuse() {
    let root = json!([[null]]);
    let mut left = [Frame::empty(); 4];
    let mut empty = [];
    assert_eq!(
        structural_eq(&root, &root, &mut left, &mut empty, limits()).unwrap_err(),
        Error::WorkspaceTooSmall
    );
    let mut short = [Frame::empty(); 2];
    assert_eq!(
        structural_eq(&root, &root, &mut left, &mut short, limits()).unwrap_err(),
        Error::WorkspaceTooSmall
    );
    let mut right = [Frame::empty(); 4];
    assert!(
        structural_eq(&root, &root, &mut left, &mut right, limits())
            .unwrap()
            .equal
    );
}

#[test]
fn borrowed_competing_limits_preserve_explicit_constructor_and_transition_precedence() {
    let root = json!({"a": "long"});
    let no_work_or_depth = Limits {
        max_work: 0,
        max_depth: 0,
        max_nodes: 0,
        max_bytes: 0,
        ..limits()
    };
    assert_eq!(census(&root, no_work_or_depth), Err(Error::WorkLimit));
    assert_eq!(
        census(
            &root,
            Limits {
                max_work: 100,
                ..no_work_or_depth
            }
        ),
        Err(Error::DepthLimit)
    );
    assert_eq!(
        census(
            &root,
            Limits {
                max_work: 100,
                max_depth: 32,
                ..no_work_or_depth
            }
        ),
        Err(Error::NodeLimit)
    );
    assert_eq!(
        census(
            &root,
            Limits {
                max_key_bytes: 0,
                max_string_bytes: 0,
                ..limits()
            }
        ),
        Err(Error::KeyLimit)
    );
    let mut left = [Frame::empty(); 4];
    let mut empty = [];
    // Construction (including right emptiness) precedes either input preflight.
    assert_eq!(
        structural_eq(&root, &root, &mut left, &mut empty, no_work_or_depth).unwrap_err(),
        Error::WorkspaceTooSmall
    );
    assert_eq!(
        structural_eq(
            &root,
            &root,
            &mut left,
            &mut empty,
            Limits {
                max_workspace_bytes: 0,
                ..no_work_or_depth
            }
        )
        .unwrap_err(),
        Error::WorkspaceLimit
    );
    let nested = json!([null]);
    let mut short_right = [Frame::empty(); 1];
    // Depth is checked before push-slot availability, on right as on left.
    assert_eq!(
        structural_eq(
            &Value::Null,
            &nested,
            &mut left,
            &mut short_right,
            Limits {
                max_depth: 1,
                ..limits()
            }
        )
        .unwrap_err(),
        Error::DepthLimit
    );
}

#[test]
fn borrowed_phase_inventory_literal_concrete_slot_oracle() {
    // Independent TEST-ONLY inventory. Do not build these expectations out of
    // Resources component sizes or Phases::maximum: a missing/wrong component
    // must not reproduce itself in the oracle. Source-level slots only; none of
    // these sums asserts compiled stack layout or prices private formatter state.
    let quotation = 2 * (size_of::<Resources>() + size_of::<Result<Resources, Error>>())
        + size_of::<std::alloc::Layout>()
        + size_of::<Result<std::alloc::Layout, std::alloc::LayoutError>>()
        + 2 * size_of::<Result<usize, Error>>()
        + size_of::<Limits>();
    let entry = size_of::<Event<'_>>()
        + size_of::<Option<Event<'_>>>()
        + size_of::<Result<Option<Event<'_>>, Error>>()
        + 3 * size_of::<Result<(), Error>>()
        + 2 * size_of::<Result<usize, Error>>()
        + size_of::<serde_json::map::Iter<'_>>()
        + size_of::<Option<serde_json::map::Iter<'_>>>();
    let number = size_of::<NumberCounter<'_>>()
        + size_of::<CompactFormatter>()
        + size_of::<(&Number, &mut NumberCounter<'_>)>()
        + size_of::<u64>()
        + size_of::<(&mut NumberCounter<'_>, &[u8])>()
        + size_of::<(&mut NumberCounter<'_>, usize)>()
        + size_of::<Option<u64>>()
        + size_of::<Option<i64>>()
        + size_of::<Option<f64>>()
        + 2 * size_of::<io::Result<()>>()
        + 2 * size_of::<io::Result<usize>>()
        + size_of::<io::Error>()
        + 2 * size_of::<Result<(), Error>>()
        + size_of::<Result<usize, Error>>()
        + size_of::<Result<Option<Event<'_>>, Error>>();
    let selection = size_of::<serde_json::map::Iter<'_>>()
        + size_of::<Option<serde_json::map::Iter<'_>>>()
        + 4 * size_of::<Option<(&String, &Value)>>()
        + 3 * size_of::<(&String, &Value)>()
        + 4 * size_of::<Result<(), Error>>()
        + size_of::<Result<usize, Error>>()
        + size_of::<(
            &mut Selection<'_>,
            &Map<String, Value>,
            Option<&str>,
            &mut Meter,
        )>();
    let sorted_comparisons = 3 * size_of::<Result<Ordering, Error>>()
        + size_of::<(&mut Meter, &str, &str)>()
        + size_of::<(&mut Meter, &str, &str, bool)>()
        + size_of::<(&str, &str)>();
    let completion = size_of::<Preflight<'_>>() + 2 * size_of::<Result<Preflight<'_>, Error>>();

    for order in [Order::Sorted, Order::Native] {
        let (quote, frame_size) = match order {
            Order::Sorted => (
                Traversal::resources_for_slots(8).unwrap(),
                size_of::<Frame<'_>>(),
            ),
            Order::Native => (
                Traversal::resources_for_native_slots(8).unwrap(),
                size_of::<NativeFrame<'_>>(),
            ),
        };
        let construction = 2 * size_of::<Result<Cursor<'_, '_>, Error>>()
            + 2 * size_of::<Result<Traversal<'_, '_>, Error>>()
            + size_of::<Workspace<'_, '_>>()
            + size_of::<&Value>()
            + size_of::<Frame<'_>>()
            + frame_size
            + size_of::<Result<(), Error>>();
        let delivery = 3 * size_of::<Result<Option<Event<'_>>, Error>>()
            + 2 * size_of::<(&mut Cursor<'_, '_>, &mut Meter)>()
            + size_of::<Frame<'_>>()
            + frame_size
            + 2 * size_of::<Result<(), Error>>()
            + size_of::<Result<usize, Error>>();
        let selection = selection
            + if order == Order::Sorted {
                sorted_comparisons
            } else {
                0
            };
        assert_eq!(quote.phases.quotation, quotation);
        assert_eq!(quote.phases.construction, construction);
        assert_eq!(quote.phases.entry, entry);
        assert_eq!(quote.phases.number, number);
        assert_eq!(quote.phases.selection, selection);
        assert_eq!(quote.phases.delivery, delivery);
        assert_eq!(quote.phases.equality, 0);
        assert_eq!(quote.phases.completion, completion);
        // Literal slots, independent max calculation; no opaque input owner or
        // unexposed dependency buffers sneak into the numeric envelope as zero.
        let peak = [
            quotation,
            construction,
            entry,
            number,
            selection,
            delivery,
            0,
            completion,
        ]
        .into_iter()
        .max()
        .unwrap();
        assert_eq!(
            quote.logical_workspace_bytes,
            8 * frame_size + size_of::<Traversal<'_, '_>>() + peak
        );
        assert_eq!(quote.format_arguments_bytes, 0); // format_args path removed
        assert_eq!(quote.format_argument_backing_bytes, 0);
        assert_eq!(quote.dependency_buffer_bytes, None);
        assert_eq!(quote.dependency_call_stack_bytes, None);
    }

    let pair = equality_resources_for_slots(4, 4).unwrap();
    let equality = size_of::<(Option<Event<'_>>, Option<Event<'_>>)>()
        + 2 * size_of::<(Event<'_>, Event<'_>)>()
        + size_of::<(Event<'_>, Event<'_>, &mut Meter)>()
        + 2 * size_of::<Result<bool, Error>>()
        + size_of::<(&mut Meter, &str, &str)>()
        + size_of::<(&mut Meter, &str, &str, bool)>()
        + size_of::<(&str, &str)>()
        + 2 * size_of::<Result<Ordering, Error>>()
        + size_of::<Result<(), Error>>()
        + size_of::<Result<usize, Error>>();
    let pair_construction = 2 * size_of::<Result<Cursor<'_, '_>, Error>>()
        + size_of::<Workspace<'_, '_>>()
        + size_of::<&Value>()
        + 2 * size_of::<Frame<'_>>()
        + size_of::<Result<(), Error>>();
    let pair_delivery = 3 * size_of::<Result<Option<Event<'_>>, Error>>()
        + 2 * size_of::<(&mut Cursor<'_, '_>, &mut Meter)>()
        + 2 * size_of::<Frame<'_>>()
        + 2 * size_of::<Result<(), Error>>()
        + size_of::<Result<usize, Error>>();
    let pair_completion = size_of::<Equality<'_>>() + 2 * size_of::<Result<Equality<'_>, Error>>();
    assert_eq!(pair.phases.quotation, quotation);
    assert_eq!(pair.phases.construction, pair_construction);
    assert_eq!(pair.phases.entry, entry);
    assert_eq!(pair.phases.number, number);
    assert_eq!(pair.phases.selection, selection + sorted_comparisons);
    assert_eq!(pair.phases.delivery, pair_delivery);
    assert_eq!(pair.phases.equality, equality);
    assert_eq!(pair.phases.completion, pair_completion);
    let pair_peak = [
        quotation,
        pair_construction,
        entry,
        number,
        selection + sorted_comparisons,
        pair_delivery,
        equality,
        pair_completion,
    ]
    .into_iter()
    .max()
    .unwrap();
    assert_eq!(
        pair.logical_workspace_bytes,
        8 * size_of::<Frame<'_>>() + size_of::<EqualityState<'_, '_>>() + pair_peak
    );
    assert_eq!(pair.dependency_buffer_bytes, None);
    assert_eq!(pair.dependency_call_stack_bytes, None);
}

#[test]
fn borrowed_native_work_failure_poison_is_stable_then_new_traversal_reuses_pool() {
    let old = json!([{"keep": [null]}]);
    let new = Value::Null;
    let mut frames = [const { NativeFrame::empty() }; 8];
    {
        let mut walk = Traversal::new_native(
            &old,
            &mut frames,
            Limits {
                max_work: 9,
                ..limits()
            },
        )
        .unwrap();
        loop {
            match walk.next_event() {
                Ok(Some(_)) => {}
                Err(error) => {
                    assert_eq!(error, Error::WorkLimit);
                    break;
                }
                Ok(None) => panic!("fixture must fail before completion"),
            }
        }
        let failed_usage = walk.usage();
        assert_eq!(failed_usage.work, 9);
        for _ in 0..4 {
            assert_eq!(walk.next_event().unwrap_err(), Error::Poisoned);
            assert_eq!(walk.usage(), failed_usage);
        }
    }
    // The old immutable owner stays alive. Inactive slots are safe to retain;
    // starting a new traversal does NOT promise to clear every stale reference.
    assert!(std::ptr::eq(
        frames[2].frame.value.unwrap(),
        &old[0]["keep"]
    ));
    let reused = preflight_native(&new, &mut frames, limits()).unwrap();
    let mut fresh = [const { NativeFrame::empty() }; 8];
    assert_eq!(
        reused.usage(),
        preflight_native(&new, &mut fresh, limits())
            .unwrap()
            .usage()
    );
    assert!(std::ptr::eq(reused.value(), &new));
    assert!(std::ptr::eq(
        frames[2].frame.value.unwrap(),
        &old[0]["keep"]
    ));
}

#[test]
fn borrowed_stale_small_pool_limits_reject_larger_actual_pools_before_mutation() {
    let sentinel = json!({"retained": [true], "z": false});
    let root = Value::Null; // only one live frame needed, but ALL slots count
    let seed = Frame {
        value: Some(&sentinel),
        entered: true,
        index: 7,
        previous: Some("retained"),
        pending: Some(&sentinel["retained"]),
    };
    let mut sorted = [seed; 8];
    // A quote supplies numeric limits, NOT Budget authority or a pool identity.
    // Constructors must price the actual supplied pool, not trust this old size.
    let small = Traversal::resources_for_slots(1).unwrap();
    assert!(matches!(
        Traversal::new(
            &root,
            &mut sorted,
            Limits {
                max_workspace_bytes: small.logical_workspace_bytes,
                ..limits()
            }
        ),
        Err(Error::WorkspaceLimit)
    ));
    for frame in &sorted {
        assert!(std::ptr::eq(frame.value.unwrap(), &sentinel));
        assert!(frame.entered);
        assert_eq!(frame.index, 7);
        assert_eq!(frame.previous, Some("retained"));
        assert!(std::ptr::eq(frame.pending.unwrap(), &sentinel["retained"]));
    }
    let mut native = [const { NativeFrame::empty() }; 8];
    for slot in &mut native {
        slot.frame = seed;
        slot.iterator = Some(sentinel.as_object().unwrap().iter());
    }
    let small = Traversal::resources_for_native_slots(1).unwrap();
    assert!(matches!(
        Traversal::new_native(
            &root,
            &mut native,
            Limits {
                max_workspace_bytes: small.logical_workspace_bytes,
                ..limits()
            }
        ),
        Err(Error::WorkspaceLimit)
    ));
    for slot in &native {
        assert!(std::ptr::eq(slot.frame.value.unwrap(), &sentinel));
        assert!(slot.frame.entered);
        assert_eq!(slot.frame.index, 7);
        assert_eq!(slot.frame.previous, Some("retained"));
        assert!(std::ptr::eq(
            slot.frame.pending.unwrap(),
            &sentinel["retained"]
        ));
        assert_eq!(slot.iterator.as_ref().unwrap().size_hint(), (2, Some(2)));
    }
    let mut right = [seed; 8];
    let small = equality_resources_for_slots(1, 1).unwrap();
    assert_eq!(
        structural_eq(
            &root,
            &root,
            &mut sorted,
            &mut right,
            Limits {
                max_workspace_bytes: small.logical_workspace_bytes,
                ..limits()
            }
        )
        .unwrap_err(),
        Error::WorkspaceLimit
    );
    for frame in sorted.iter().chain(right.iter()) {
        assert!(std::ptr::eq(frame.value.unwrap(), &sentinel));
        assert_eq!(frame.index, 7);
    }
}
