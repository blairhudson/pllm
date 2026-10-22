use super::*;
use crate::{canonical_bytes, canonical_digest, Digest};
use serde_json::json;

fn limits() -> StreamLimits {
    StreamLimits {
        max_nodes: 1024,
        max_depth: 16,
        max_bytes: 65536,
        max_string_bytes: 4096,
        max_key_bytes: 128,
        max_total_string_bytes: 65536,
        max_work: 1_000_000,
        max_stack_bytes: 16 * StreamFrame::storage_bytes(),
    }
}

fn schema<'a, W: io::Write>(
    s: &mut Session<'a, '_, W>,
    attrs: &'a Value,
) -> Result<(), StreamError> {
    s.begin_object()?;
    s.key("attrs")?;
    s.json(attrs)?;
    s.key("bool")?;
    s.bool(true)?;
    s.key("count")?;
    s.u64(u64::MAX)?;
    s.key("float")?;
    s.f64(-0.0)?;
    s.key("list")?;
    s.begin_array()?;
    s.null()?;
    s.i64(i64::MIN)?;
    s.string("\n🦀")?;
    s.end_array()?;
    s.end_object()
}

fn count(value: &Value, limits: StreamLimits) -> Result<StreamUsage, StreamError> {
    let mut sink = io::sink();
    let mut stack = [StreamFrame::empty(); 16];
    let mut session = Session::new(&mut sink, &mut stack, limits)?;
    session.json(value)?;
    session.finish()
}

#[test]
fn two_pass_direct_schema_fragment_and_shared_hash_parity() {
    let attrs = json!({"z": [{"b": "\"\\\0", "a": false}], "a": [1.25, null]});
    let expected = json!({"attrs": attrs, "bool": true, "count": u64::MAX, "float": -0.0, "list": [null, i64::MIN, "\n🦀"]});
    let mut stack = [StreamFrame::empty(); 16];
    let mut sink = io::sink();
    let mut count = Session::new(&mut sink, &mut stack, limits()).unwrap();
    schema(&mut count, &attrs).unwrap();
    let admitted = count.finish().unwrap();
    let mut output = Vec::new();
    let mut replay = Session::new(&mut output, &mut stack, limits()).unwrap();
    schema(&mut replay, &attrs).unwrap();
    assert_eq!(replay.finish().unwrap(), admitted);
    assert_eq!(output, canonical_bytes(&expected));
    for domain in ["", "pllm.pipeline.v2", "with\0separator", "é"] {
        let mut hash = DomainHash::new(domain);
        let mut replay = Session::new(&mut hash, &mut stack, limits()).unwrap();
        schema(&mut replay, &attrs).unwrap();
        assert_eq!(replay.finish().unwrap(), admitted);
        assert_eq!(
            Digest::from_sha256(hash.finish()),
            canonical_digest(domain, &expected)
        );
    }
}

#[test]
fn exact_fragment_budgets_and_each_counter_limit() {
    let value = json!({"z": ["\n", {"b": "β", "a": true}], "a": null});
    let used = count(&value, limits()).unwrap();
    assert_eq!(used.nodes, 7);
    assert_eq!(used.depth, 4);
    assert_eq!(used.bytes, canonical_bytes(&value).len());
    assert_eq!(used.string_bytes, 7); // z,a,b,a plus newline and two UTF-8 bytes
    let exact = StreamLimits {
        max_nodes: used.nodes,
        max_depth: used.depth,
        max_bytes: used.bytes,
        max_string_bytes: 2,
        max_key_bytes: 1,
        max_total_string_bytes: used.string_bytes,
        max_work: used.work,
        ..limits()
    };
    assert_eq!(count(&value, exact).unwrap(), used);
    assert!(matches!(
        count(
            &value,
            StreamLimits {
                max_nodes: used.nodes - 1,
                ..exact
            }
        ),
        Err(StreamError::NodeLimit)
    ));
    assert!(matches!(
        count(
            &value,
            StreamLimits {
                max_depth: used.depth - 1,
                ..exact
            }
        ),
        Err(StreamError::DepthLimit)
    ));
    assert!(matches!(
        count(
            &value,
            StreamLimits {
                max_bytes: used.bytes - 1,
                ..exact
            }
        ),
        Err(StreamError::ByteLimit)
    ));
    assert!(matches!(
        count(
            &value,
            StreamLimits {
                max_string_bytes: 1,
                ..exact
            }
        ),
        Err(StreamError::StringLimit)
    ));
    assert!(matches!(
        count(
            &value,
            StreamLimits {
                max_key_bytes: 0,
                ..exact
            }
        ),
        Err(StreamError::KeyLimit)
    ));
    assert!(matches!(
        count(
            &value,
            StreamLimits {
                max_total_string_bytes: used.string_bytes - 1,
                ..exact
            }
        ),
        Err(StreamError::TotalStringLimit)
    ));
    assert!(matches!(
        count(
            &value,
            StreamLimits {
                max_work: used.work - 1,
                ..exact
            }
        ),
        Err(StreamError::WorkLimit)
    ));
}

#[test]
fn schema_and_fragments_share_all_global_counters_and_depth_offset() {
    let attrs = json!([null, ["\n"]]);
    let mut sink = io::sink();
    let mut stack = [StreamFrame::empty(); 4];
    let mut session = Session::new(&mut sink, &mut stack, limits()).unwrap();
    session.begin_object().unwrap();
    session.key("a").unwrap();
    session.json(&attrs).unwrap();
    session.key("b").unwrap();
    session.json(&attrs).unwrap();
    session.end_object().unwrap();
    let used = session.finish().unwrap();
    assert_eq!(used.nodes, 9); // outer object plus four values per fragment
    assert_eq!(used.depth, 4);
    assert_eq!(used.string_bytes, 4); // keys a,b plus two newlines
    assert_eq!(
        used.bytes,
        canonical_bytes(&json!({"a": attrs, "b": attrs})).len()
    );
    let mut session = Session::new(
        &mut sink,
        &mut stack,
        StreamLimits {
            max_nodes: 8,
            ..limits()
        },
    )
    .unwrap();
    session.begin_object().unwrap();
    session.key("a").unwrap();
    session.json(&attrs).unwrap();
    session.key("b").unwrap();
    assert!(matches!(session.json(&attrs), Err(StreamError::NodeLimit)));
    assert!(matches!(session.finish(), Err(StreamError::Poisoned)));
}

#[test]
fn escaped_text_and_work_units_are_exact() {
    let mut sink = io::sink();
    let mut empty = [];
    let mut session = Session::new(&mut sink, &mut empty, limits()).unwrap();
    session.string("\n").unwrap();
    assert_eq!(
        session.finish().unwrap(),
        StreamUsage {
            nodes: 1,
            depth: 1,
            bytes: 4,
            string_bytes: 1,
            work: 7
        }
    );
    let mut stack = [StreamFrame::empty(); 1];
    let mut session = Session::new(&mut sink, &mut stack, limits()).unwrap();
    session.begin_object().unwrap();
    session.key("a").unwrap();
    session.u64(0).unwrap();
    session.key("b").unwrap();
    session.bool(true).unwrap();
    session.end_object().unwrap();
    assert_eq!(
        session.finish().unwrap(),
        StreamUsage {
            nodes: 3,
            depth: 2,
            bytes: 16,
            string_bytes: 2,
            work: 26
        }
    );
    let controls: String = (0..=127).map(char::from).collect();
    let value = json!({"\0\"\\\n": controls, "é": "\u{2028}🦀"});
    let mut stack = [StreamFrame::empty(); 16];
    let mut output = Vec::new();
    let mut session = Session::new(&mut output, &mut stack, limits()).unwrap();
    session.json(&value).unwrap();
    session.finish().unwrap();
    assert_eq!(output, canonical_bytes(&value));

    let mut session = Session::new(&mut sink, &mut stack, limits()).unwrap();
    session.begin_object().unwrap();
    session.key("\n").unwrap();
    session.string("\n").unwrap();
    session.end_object().unwrap();
    assert_eq!(
        session.finish().unwrap(),
        StreamUsage {
            nodes: 2,
            depth: 2,
            bytes: 11,
            string_bytes: 2,
            work: 18
        }
    );
}

#[test]
fn comparison_scans_are_charged_and_long_keys_reject_before_comparison() {
    let value = json!({"aaaa": null, "aaab": null, "aaac": null});
    let used = count(&value, limits()).unwrap();
    // Direct ordered schema emits the same bytes but avoids quadratic selection.
    let mut sink = io::sink();
    let mut stack = [StreamFrame::empty(); 16];
    let mut schema = Session::new(&mut sink, &mut stack, limits()).unwrap();
    schema.begin_object().unwrap();
    for key in ["aaaa", "aaab", "aaac"] {
        schema.key(key).unwrap();
        schema.null().unwrap();
    }
    schema.end_object().unwrap();
    let direct = schema.finish().unwrap();
    assert!(used.work > direct.work);
    assert_eq!(used.bytes, direct.bytes);
    let mut session = Session::new(
        &mut sink,
        &mut stack,
        StreamLimits {
            max_key_bytes: 3,
            ..limits()
        },
    )
    .unwrap();
    session.begin_object().unwrap();
    let before = session.usage().work;
    assert!(matches!(
        session.key("long key"),
        Err(StreamError::KeyLimit)
    ));
    assert_eq!(session.usage().work, before + 1); // event only, no prefix scan
    assert!(matches!(
        count(
            &value,
            StreamLimits {
                max_key_bytes: 3,
                ..limits()
            }
        ),
        Err(StreamError::KeyLimit)
    ));
}

#[test]
fn malformed_events_and_key_order_poison_the_session() {
    let mut sink = io::sink();
    let mut stack = [StreamFrame::empty(); 2];
    {
        let mut s = Session::new(&mut sink, &mut stack, limits()).unwrap();
        assert!(matches!(s.key("a"), Err(StreamError::Structure)));
        assert!(matches!(s.null(), Err(StreamError::Poisoned)));
    }
    {
        let mut s = Session::new(&mut sink, &mut stack, limits()).unwrap();
        s.begin_object().unwrap();
        assert!(matches!(s.null(), Err(StreamError::Structure)));
    }
    {
        let mut s = Session::new(&mut sink, &mut stack, limits()).unwrap();
        s.begin_object().unwrap();
        s.key("a").unwrap();
        assert!(matches!(s.end_object(), Err(StreamError::Structure)));
    }
    {
        let mut s = Session::new(&mut sink, &mut stack, limits()).unwrap();
        s.begin_array().unwrap();
        assert!(matches!(s.end_object(), Err(StreamError::Structure)));
    }
    {
        let mut s = Session::new(&mut sink, &mut stack, limits()).unwrap();
        s.null().unwrap();
        assert!(matches!(s.null(), Err(StreamError::Structure)));
    }
    for key in ["a", "0"] {
        let mut s = Session::new(&mut sink, &mut stack, limits()).unwrap();
        s.begin_object().unwrap();
        s.key("a").unwrap();
        s.null().unwrap();
        assert!(matches!(s.key(key), Err(StreamError::KeyOrder)));
    }
    let mut s = Session::new(&mut sink, &mut stack, limits()).unwrap();
    s.begin_array().unwrap();
    assert!(matches!(s.finish(), Err(StreamError::Structure)));
    let s = Session::new(&mut sink, &mut stack, limits()).unwrap();
    assert!(matches!(s.finish(), Err(StreamError::Structure)));
}

#[test]
fn stack_capacity_and_concrete_resources() {
    let mut sink = io::sink();
    let mut stack = [StreamFrame::empty(); 2];
    let exact = 2 * StreamFrame::storage_bytes();
    assert!(matches!(
        Session::new(
            &mut sink,
            &mut stack,
            StreamLimits {
                max_stack_bytes: exact - 1,
                ..limits()
            }
        ),
        Err(StreamError::StackLimit)
    ));
    let session = Session::new(
        &mut sink,
        &mut stack,
        StreamLimits {
            max_stack_bytes: exact,
            ..limits()
        },
    )
    .unwrap();
    let resources = session.resources();
    assert_eq!(resources.session_bytes, size_of_val(&session));
    assert_eq!(resources.stack_bytes, exact);
    assert_eq!(resources.hash_state_bytes, size_of::<DomainHash>());
    assert_eq!(resources.hash_result_bytes, 32);
    assert_eq!(resources.escape_buffer_bytes, 6);
    assert_eq!(resources.dependency_call_stack_bytes, None);
    assert_eq!(
        resources,
        Session::<io::Sink>::resources_for_slots(2).unwrap()
    );
    assert!(matches!(
        Session::<io::Sink>::resources_for_slots(usize::MAX),
        Err(StreamError::StackLimit)
    ));
    let mut empty = [];
    let mut s = Session::new(&mut sink, &mut empty, limits()).unwrap();
    assert!(matches!(s.begin_array(), Err(StreamError::StackTooSmall)));
}

#[test]
fn zero_limits_and_empty_stack_scalar_boundary() {
    let mut sink = io::sink();
    let mut empty = [];
    let mut s = Session::new(
        &mut sink,
        &mut empty,
        StreamLimits {
            max_stack_bytes: 0,
            max_nodes: 0,
            ..limits()
        },
    )
    .unwrap();
    assert!(matches!(s.null(), Err(StreamError::NodeLimit)));
    let mut s = Session::new(
        &mut sink,
        &mut empty,
        StreamLimits {
            max_depth: 0,
            ..limits()
        },
    )
    .unwrap();
    assert!(matches!(s.null(), Err(StreamError::DepthLimit)));
    let mut s = Session::new(
        &mut sink,
        &mut empty,
        StreamLimits {
            max_bytes: 0,
            ..limits()
        },
    )
    .unwrap();
    assert!(matches!(s.null(), Err(StreamError::ByteLimit)));
    let mut s = Session::new(
        &mut sink,
        &mut empty,
        StreamLimits {
            max_work: 0,
            ..limits()
        },
    )
    .unwrap();
    assert!(matches!(s.null(), Err(StreamError::WorkLimit)));
    let mut s = Session::new(
        &mut sink,
        &mut empty,
        StreamLimits {
            max_nodes: 1,
            max_depth: 1,
            max_bytes: 2,
            max_string_bytes: 0,
            max_total_string_bytes: 0,
            max_work: 4,
            max_stack_bytes: 0,
            ..limits()
        },
    )
    .unwrap();
    s.string("").unwrap();
    assert_eq!(
        s.finish().unwrap(),
        StreamUsage {
            nodes: 1,
            depth: 1,
            bytes: 2,
            string_bytes: 0,
            work: 4
        }
    );
}

#[test]
fn numeric_parity_and_nonfinite_rejection() {
    let mut sink = io::sink();
    let mut stack = [StreamFrame::empty(); 2];
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut s = Session::new(&mut sink, &mut stack, limits()).unwrap();
        assert!(matches!(s.f64(value), Err(StreamError::NonFiniteNumber)));
        assert!(matches!(s.finish(), Err(StreamError::Poisoned)));
    }
    for value in [0.0, -0.0, 1.0, 5e-324, 1e100, f64::MAX, 1.2345678901234567] {
        let mut output = Vec::new();
        let mut s = Session::new(&mut output, &mut stack, limits()).unwrap();
        s.f64(value).unwrap();
        s.finish().unwrap();
        assert_eq!(output, canonical_bytes(&json!(value)));
    }
}

#[test]
fn writer_and_structure_failures_make_no_atomicity_claim() {
    struct Broken {
        bytes: Vec<u8>,
        remaining: usize,
    }
    impl io::Write for Broken {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.remaining == 0 {
                return Err(io::ErrorKind::BrokenPipe.into());
            }
            let count = bytes.len().min(self.remaining);
            self.bytes.extend_from_slice(&bytes[..count]);
            self.remaining -= count;
            Ok(count)
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut writer = Broken {
        bytes: Vec::new(),
        remaining: 2,
    };
    let mut stack = [StreamFrame::empty(); 2];
    let mut s = Session::new(&mut writer, &mut stack, limits()).unwrap();
    assert!(
        matches!(s.string("abc"), Err(StreamError::Io(e)) if e.kind() == io::ErrorKind::BrokenPipe)
    );
    assert!(matches!(s.finish(), Err(StreamError::Poisoned)));
    assert_eq!(writer.bytes, b"\"a");
    let mut output = Vec::new();
    let mut s = Session::new(&mut output, &mut stack, limits()).unwrap();
    s.begin_object().unwrap();
    assert!(matches!(s.end_array(), Err(StreamError::Structure)));
    assert!(matches!(s.finish(), Err(StreamError::Poisoned)));
    assert_eq!(output, b"{");
}

#[test]
fn borrowed_scalar_fragment_remains_poisoned_after_caught_writer_panic() {
    struct PanicOnce(bool);
    impl io::Write for PanicOnce {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.0 {
                self.0 = false;
                panic!("writer panic");
            }
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let value = Value::String("borrowed".into());
    let mut writer = PanicOnce(true);
    let mut stack = [];
    let mut s = Session::new(&mut writer, &mut stack, limits()).unwrap();
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.json(&value))).is_err());
    let failed_usage = s.usage();
    assert!(matches!(s.null(), Err(StreamError::Poisoned)));
    assert_eq!(s.usage(), failed_usage);
    assert!(matches!(s.finish(), Err(StreamError::Poisoned)));
}

#[test]
fn final_fragment_selection_scan_failure_poison_and_stack_reuse() {
    let value = json!({"a": null});
    let mut stack = [StreamFrame::empty(); 2];
    let mut output = Vec::new();
    let mut s = Session::new(
        &mut output,
        &mut stack,
        StreamLimits {
            max_work: 14,
            ..limits()
        },
    )
    .unwrap();
    // Root, selected field, and null fit. Final key-selection scan does not.
    assert!(matches!(s.json(&value), Err(StreamError::WorkLimit)));
    let failed_usage = s.usage();
    assert_eq!(failed_usage.work, 14);
    assert_eq!(failed_usage.nodes, 2);
    assert_eq!(failed_usage.bytes, 9);
    assert!(matches!(s.end_object(), Err(StreamError::Poisoned)));
    assert_eq!(s.usage(), failed_usage);
    assert!(matches!(s.finish(), Err(StreamError::Poisoned)));
    assert_eq!(output, br#"{"a":null"#);
    output.clear();
    let mut replay = Session::new(&mut output, &mut stack, limits()).unwrap();
    replay.json(&value).unwrap();
    let used = replay.finish().unwrap();
    assert_eq!(used.nodes, 2);
    assert_eq!(output, canonical_bytes(&value));
}

#[test]
fn fragment_closing_string_quote_budget_failure_cannot_finish() {
    let value = Value::String("\n".into());
    let mut output = Vec::new();
    let mut stack = [];
    let mut s = Session::new(
        &mut output,
        &mut stack,
        StreamLimits {
            max_bytes: 3,
            ..limits()
        },
    )
    .unwrap();
    assert!(matches!(s.json(&value), Err(StreamError::ByteLimit)));
    assert_eq!(s.usage().bytes, 3);
    assert!(matches!(s.finish(), Err(StreamError::Poisoned)));
    assert_eq!(output, b"\"\\n");
}

#[test]
fn actual_transient_storage_contains_live_failed_selection_state() {
    let value = json!({"aa": null, "ab": true});
    let mut sink = io::sink();
    let mut stack = [StreamFrame::empty(); 2];
    let mut s = Session::new(
        &mut sink,
        &mut stack,
        StreamLimits {
            max_work: 5,
            ..limits()
        },
    )
    .unwrap();
    assert_eq!(
        Session::<io::Sink>::transient_storage_bytes(),
        size_of::<Transient<'_>>()
    );
    assert_eq!(
        Session::<io::Sink>::transient_storage_bytes(),
        size_of_val(&s.transient)
    );
    assert_eq!(s.resources().session_bytes, size_of_val(&s));
    assert!(s.resources().session_bytes >= Session::<io::Sink>::transient_storage_bytes());
    // First candidate selected; second candidate advanced the retained iterator,
    // but its first comparison byte exceeds work. All live selection state is in
    // Session, not a local best entry, iterator, base, or outer frame snapshot.
    assert!(matches!(s.json(&value), Err(StreamError::WorkLimit)));
    assert_eq!(s.transient.fragment_base, 0);
    assert!(std::ptr::eq(s.transient.outer.fragment.unwrap(), &value));
    let (key, selected) = s.transient.selected.unwrap();
    assert_eq!(key, "aa");
    assert!(std::ptr::eq(selected, &value["aa"]));
    assert_eq!(
        s.transient.object_iter.as_ref().unwrap().size_hint(),
        (0, Some(0))
    );
    assert!(matches!(s.finish(), Err(StreamError::Poisoned)));
}

#[test]
fn outer_and_shared_inner_snapshots_are_used_across_nested_emission() {
    let attrs = json!({"aa": [null], "zz": {"a": true}});
    let mut output = Vec::new();
    let mut stack = [StreamFrame::empty(); 4];
    let mut s = Session::new(&mut output, &mut stack, limits()).unwrap();
    s.begin_object().unwrap();
    s.key("attrs").unwrap();
    s.json(&attrs).unwrap();
    assert_eq!(s.transient.fragment_base, 1);
    assert!(std::ptr::eq(s.transient.outer.fragment.unwrap(), &attrs));
    assert_eq!(s.transient.outer.previous, Some("zz"));
    // The value helper reused the same inner snapshot as ordered_key; outer
    // survived both while nested traversal continued using caller-owned frames.
    assert!(std::ptr::eq(
        s.transient.inner.fragment.unwrap(),
        &attrs["zz"]
    ));
    assert_eq!(s.transient.inner.previous, Some("a"));
    assert!(s.transient.inner.waiting);
    assert!(s.transient.selected.is_none());
    assert!(s.transient.object_iter.is_none());
    s.key("tail").unwrap();
    assert_eq!(s.transient.inner.previous, Some("attrs"));
    assert!(!s.transient.inner.waiting);
    s.bool(false).unwrap();
    assert_eq!(s.transient.inner.previous, Some("tail"));
    assert!(s.transient.inner.waiting);
    assert!(s.transient.inner.fragment.is_none());
    s.end_object().unwrap();
    s.finish().unwrap();
    assert_eq!(
        output,
        canonical_bytes(&json!({"attrs": attrs, "tail": false}))
    );
}
