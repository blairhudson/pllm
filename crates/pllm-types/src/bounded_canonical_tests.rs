use super::*;
use crate::{canonical_bytes, canonical_digest, Digest};
use serde_json::json;

fn limits() -> Limits {
    Limits {
        max_nodes: 1024,
        max_depth: 32,
        max_bytes: 65536,
        max_workspace_bytes: 32 * size_of::<Frame<'_>>(),
    }
}

fn parity(value: &Value) {
    let mut workspace = [Frame::empty(); 32];
    let admitted = admit(Node::Json(value), limits(), &mut workspace).unwrap();
    let expected = canonical_bytes(value);
    let mut actual = Vec::new();
    admitted.write(&mut workspace, &mut actual).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(admitted.usage().bytes, expected.len());
    for domain in ["", "pllm.pipeline.v2", "domain\0with\0zeros", "é"] {
        assert_eq!(
            Digest::from_sha256(admitted.hash(domain, &mut workspace).unwrap()),
            canonical_digest(domain, value)
        );
    }
}

#[test]
fn legacy_escapes_and_unicode_parity() {
    let controls: String = (0..=31).map(char::from).collect();
    parity(&json!({
        "z": controls,
        "\\\"\n\0": "\"\\/\u{7f} café 🦀 \u{2028}\u{2029}",
        "a": [null, true, false, "", {}, []],
        "nested": {"z": 1, "b": {"y": "x", "a": 0}},
    }));
    // Includes every ASCII byte, including non-escaped DEL.
    parity(&Value::String((0..=127).map(char::from).collect()));
}

#[test]
fn legacy_number_bytes_parity() {
    for text in [
        "0",
        "-0",
        "-1",
        "18446744073709551615",
        "-9223372036854775808",
        "1.0",
        "-0.0",
        "0.0000001",
        "1e100",
        "1e-100",
        "1.2345678901234567",
        "5e-324",
        "1.7976931348623157e308",
    ] {
        parity(&serde_json::from_str::<Value>(text).unwrap());
    }
    // Invalid JSON and non-finite numbers cannot enter the borrowed API.
    for text in ["NaN", "Infinity", "01", "1e9999"] {
        assert!(serde_json::from_str::<Value>(text).is_err());
    }
    assert!(Number::from_f64(f64::NAN).is_none());
    assert!(Number::from_f64(f64::INFINITY).is_none());
}

#[test]
fn typed_schema_hooks_sort_without_copying_attributes() {
    let attributes = json!({"z": [null, {"b": false, "a": "borrowed"}], "a": 1});
    let number = Number::from(42);
    let array = [Node::Number(&number), Node::String("hello\n")];
    let fields = [
        ("z", Node::Array(&array)),
        ("attrs", Node::Json(&attributes)),
        ("a", Node::Bool(true)),
    ];
    let reordered = [fields[2], fields[0], fields[1]];
    let mut workspace = [Frame::empty(); 32];
    let first = admit(Node::Object(&fields), limits(), &mut workspace).unwrap();
    let second = admit(Node::Object(&reordered), limits(), &mut workspace).unwrap();
    let mut actual = Vec::new();
    first.write(&mut workspace, &mut actual).unwrap();
    let expected = json!({"z": [42, "hello\n"], "attrs": attributes, "a": true});
    assert_eq!(actual, canonical_bytes(&expected));
    assert_eq!(
        first.hash("schema", &mut workspace).unwrap(),
        second.hash("schema", &mut workspace).unwrap()
    );
    assert_eq!(
        Digest::from_sha256(first.hash("schema", &mut workspace).unwrap()),
        canonical_digest("schema", &expected)
    );
}

#[test]
fn exact_budgets_and_each_one_short_failure() {
    let value = json!({"b": [null, "\n"], "a": true});
    let mut workspace = [Frame::empty(); 3];
    let usage = admit(Node::Json(&value), limits(), &mut workspace)
        .unwrap()
        .usage();
    assert_eq!(usage.nodes, 5); // object, array, null, string, boolean
    assert_eq!(usage.depth, 3);
    assert_eq!(usage.workspace_bytes, 3 * size_of::<Frame<'_>>());
    assert_eq!(usage.bytes, canonical_bytes(&value).len());
    let exact = Limits {
        max_nodes: usage.nodes,
        max_depth: usage.depth,
        max_bytes: usage.bytes,
        max_workspace_bytes: usage.workspace_bytes,
    };
    let admitted = admit(Node::Json(&value), exact, &mut workspace).unwrap();
    let mut bytes = Vec::new();
    admitted.write(&mut workspace, &mut bytes).unwrap();
    assert_eq!(bytes, canonical_bytes(&value));
    assert!(matches!(
        admit(
            Node::Json(&value),
            Limits {
                max_nodes: usage.nodes - 1,
                ..exact
            },
            &mut workspace
        ),
        Err(Error::NodeLimit)
    ));
    assert!(matches!(
        admit(
            Node::Json(&value),
            Limits {
                max_depth: usage.depth - 1,
                ..exact
            },
            &mut workspace
        ),
        Err(Error::DepthLimit)
    ));
    assert!(matches!(
        admit(
            Node::Json(&value),
            Limits {
                max_bytes: usage.bytes - 1,
                ..exact
            },
            &mut workspace
        ),
        Err(Error::ByteLimit)
    ));
    assert!(matches!(
        admit(
            Node::Json(&value),
            Limits {
                max_workspace_bytes: usage.workspace_bytes - 1,
                ..exact
            },
            &mut workspace
        ),
        Err(Error::WorkspaceLimit)
    ));
    assert!(matches!(
        admit(Node::Json(&value), exact, &mut workspace[..2]),
        Err(Error::WorkspaceTooSmall)
    ));
}

#[test]
fn duplicate_schema_keys_rejected_at_any_position() {
    for keys in [["a", "a", "z"], ["z", "a", "z"], ["é", "é", "a"]] {
        let fields = keys.map(|key| (key, Node::Null));
        let mut workspace = [Frame::empty(); 2];
        assert!(matches!(
            admit(Node::Object(&fields), limits(), &mut workspace),
            Err(Error::DuplicateKey)
        ));
    }
}

#[test]
fn zero_limits_and_root_accounting() {
    let mut workspace = [Frame::empty(); 1];
    assert!(matches!(
        admit(
            Node::Null,
            Limits {
                max_nodes: 0,
                ..limits()
            },
            &mut workspace
        ),
        Err(Error::NodeLimit)
    ));
    assert!(matches!(
        admit(
            Node::Null,
            Limits {
                max_depth: 0,
                ..limits()
            },
            &mut workspace
        ),
        Err(Error::DepthLimit)
    ));
    assert!(matches!(
        admit(
            Node::Null,
            Limits {
                max_bytes: 0,
                ..limits()
            },
            &mut workspace
        ),
        Err(Error::ByteLimit)
    ));
    assert!(matches!(
        admit(
            Node::Null,
            Limits {
                max_workspace_bytes: 0,
                ..limits()
            },
            &mut workspace
        ),
        Err(Error::WorkspaceLimit)
    ));
    assert!(matches!(
        admit(Node::Null, limits(), &mut []),
        Err(Error::WorkspaceTooSmall)
    ));
    for root in [Node::Array(&[]), Node::Object(&[])] {
        let admitted = admit(
            root,
            Limits {
                max_nodes: 1,
                max_depth: 1,
                max_bytes: 2,
                max_workspace_bytes: size_of::<Frame<'_>>(),
            },
            &mut workspace,
        )
        .unwrap();
        assert_eq!(admitted.usage().nodes, 1);
        assert_eq!(admitted.usage().bytes, 2);
    }
}

#[test]
fn workspace_failure_is_atomic_and_workspace_reusable() {
    let value = json!([[["x"]]]);
    let mut workspace = [Frame::empty(); 4];
    assert!(admit(
        Node::Json(&value),
        Limits {
            max_bytes: 1,
            ..limits()
        },
        &mut workspace
    )
    .is_err());
    let admitted = admit(Node::Json(&value), limits(), &mut workspace).unwrap();
    let mut output = b"unchanged".to_vec();
    assert!(matches!(
        admitted.write(&mut workspace[..3], &mut output),
        Err(Error::WorkspaceTooSmall)
    ));
    assert_eq!(output, b"unchanged");
    assert!(matches!(
        admitted.hash("domain", &mut workspace[..3]),
        Err(Error::WorkspaceTooSmall)
    ));
    output.clear();
    admitted.write(&mut workspace, &mut output).unwrap();
    assert_eq!(output, canonical_bytes(&value));
}

#[test]
fn deep_values_use_frames_not_recursive_writing() {
    let mut value = Value::Null;
    for _ in 0..255 {
        value = Value::Array(vec![value]);
    }
    let mut workspace = [Frame::empty(); 256];
    let exact = Limits {
        max_nodes: 256,
        max_depth: 256,
        max_bytes: 514, // 255 pairs of brackets plus null
        max_workspace_bytes: 256 * size_of::<Frame<'_>>(),
    };
    let admitted = admit(Node::Json(&value), exact, &mut workspace).unwrap();
    assert_eq!(admitted.usage().depth, 256);
    let mut output = Vec::new();
    admitted.write(&mut workspace, &mut output).unwrap();
    assert_eq!(output.len(), 514);
    assert!(output[..255].iter().all(|byte| *byte == b'['));
    assert_eq!(&output[255..259], b"null");
    assert!(output[259..].iter().all(|byte| *byte == b']'));
    assert!(matches!(
        admit(
            Node::Json(&value),
            Limits {
                max_depth: 255,
                ..exact
            },
            &mut workspace
        ),
        Err(Error::DepthLimit)
    ));
}

#[test]
fn sink_errors_propagate_without_claiming_output_atomicity() {
    struct Broken;
    impl io::Write for Broken {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::ErrorKind::BrokenPipe.into())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut workspace = [Frame::empty(); 1];
    let admitted = admit(Node::String("x"), limits(), &mut workspace).unwrap();
    assert!(
        matches!(admitted.write(&mut workspace, &mut Broken), Err(Error::Io(error)) if error.kind() == io::ErrorKind::BrokenPipe)
    );
}

#[test]
fn historical_fixture_digest_unchanged() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../schemas/fixtures/experiment.valid.json"
    ))
    .unwrap();
    parity(&fixture);
    let mut workspace = [Frame::empty(); 32];
    let admitted = admit(Node::Json(&fixture), limits(), &mut workspace).unwrap();
    assert_eq!(
        Digest::from_sha256(
            admitted
                .hash("pllm.configuration.v1", &mut workspace)
                .unwrap()
        )
        .as_str(),
        "cd051de9c3dfdfe2e3f13d1316582a844d5494529cf772c58502a44021875329"
    );
}
