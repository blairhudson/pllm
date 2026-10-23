use super::*;
use pllm_core::edge_contract::{EdgeIdentity, Shape};

fn descriptor(lanes: usize, bits: u8, scale: u8) -> EdgeDescriptor {
    let source = SourceIdentity::new([1; 32], [2; 32], approved_implementation());
    EdgeDescriptor::new(
        EdgeIdentity::new(source, [3; 32], [4; 32], [5; 32], 0),
        Shape::new(&[lanes]).unwrap(),
        Numeric::new(bits, scale, Interpretation::Signed).unwrap(),
        Representation::CommonAdditiveMasked,
        None,
    )
    .unwrap()
}

#[test]
fn whole_original_clear_slice_rejects_negative_and_upper_endpoint_without_entropy_or_allocation() {
    for lanes in [1, 256] {
        let edge = descriptor(lanes, 50, 12);
        let values = vec![(1 << 38) - 1; lanes];
        assert_eq!(validate_nexp_clear_root(&edge, &values), Ok(()));
        let mut invalid = values.clone();
        invalid[lanes - 1] = -1;
        assert_eq!(
            validate_nexp_clear_root(&edge, &invalid),
            Err(Error::Domain)
        );
        invalid[lanes - 1] = 1 << 38;
        assert_eq!(
            validate_nexp_clear_root(&edge, &invalid),
            Err(Error::Domain)
        );
        invalid.fill(0);
        assert_eq!(validate_nexp_clear_root(&edge, &invalid), Ok(()));
        assert_eq!(validate_nexp_clear_root(&edge, &[]), Err(Error::Domain));
    }
    assert_eq!(
        validate_nexp_clear_root(&descriptor(1, 50, 11), &[0]),
        Err(Error::Domain)
    );
    assert_eq!(
        validate_nexp_clear_root(&descriptor(1, 49, 12), &[0]),
        Err(Error::Domain)
    );
}

#[test]
fn new_metadata_quote_is_concrete_and_explicitly_not_aggregate_admission() {
    for lanes in [1, 256] {
        let q = nexp_declaration_resources(lanes).unwrap();
        assert_eq!(
            q.metadata_bytes,
            q.declaration_bytes
                + q.coordinate_provenance_bytes
                + q.ledger_bytes
                + q.hash_controls_bytes
                + q.query_transition_bytes
        );
        assert_eq!(q.ledger_bytes, size_of::<[internal_use::Ledger<18>; 3]>());
        assert_eq!(q.exchanges, 11);
        assert_eq!(q.frames, 22);
        // Derive from the frozen stage headers, not the quoted linear formula.
        let directions = 2 * (9 * (76 + 8 * lanes) + 2 * (80 + 16 * lanes));
        assert_eq!(q.total_peer_bytes, directions);
        assert_eq!(q.total_peer_bytes, 1688 + 208 * lanes);
    }
    assert!(matches!(nexp_declaration_resources(0), Err(Error::Domain)));
    assert!(matches!(
        nexp_declaration_resources(257),
        Err(Error::Domain)
    ));
    assert!(matches!(sum(&[usize::MAX, 1]), Err(Error::Resource)));
}

#[test]
fn numerical_split_clip_and_final_product_endpoint_are_integer_equations_only() {
    // Regression of the chosen equation, not protected nExp execution/evidence.
    for x in [0u64, 255, 256, 65535, 65536, (1 << 38) - 1] {
        let clipped = x.min(65535);
        assert_eq!((clipped & 255) + 256 * (clipped >> 8), clipped);
        assert!((clipped >> 8) < 256);
        let shifted_signed40 = i128::from(x) - 65536;
        assert!((-(1i128 << 39)..(1i128 << 39)).contains(&shifted_signed40));
    }
    let product = 4096u128 * 4096;
    assert_eq!(product, 1 << 24);
    assert_eq!(product / 4096, 4096);
    assert!(product < 1 << 48); // existing GapARS quarter-ring premise
}

#[test]
fn selected_exp_literals_match_existing_exact_rational_floor_certificates_and_hashes() {
    // Ordinary independent regression. NOT executed during implementation and NOT
    // an activation token; a passing process cannot mint a native capability.
    // Reuses existing rational Taylor/tail proof, never a new approximation.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let script = r#"
from pathlib import Path
from fractions import Fraction
from hashlib import sha256
import importlib.util, re, struct, sys
root = Path(sys.argv[1]).resolve()
path = root / 'paper/reproduction-contracts/fusefss/gate-set/reference.py'
spec = importlib.util.spec_from_file_location('nexp_shared_exact_proof', path)
reference = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reference
spec.loader.exec_module(reference)
text = (root / 'crates/pllm-garble/src/fusefss_backend/composition/source_table_values.rs').read_text()
certified = []
for name, divisor, expected in [
 ('EXP0',4096,'e094819aa436afcf59d46bfd2937b91860045f5c905387d98e271731259a963b'),
 ('EXP1',16,'21a20b930ba1cd6dd57601ec54b4b6b85f206e553304b7c58da5dff2f45c5ed9')]:
    match = re.search(r'const '+name+r': \[u64; 256\] = \[(.*?)\];', text, re.S)
    assert match is not None
    selected = tuple(int(x.strip()) for x in match.group(1).split(',') if x.strip())
    proof = []
    for j in range(256):
        lo, hi = reference.exp_bounds(Fraction(j, divisor))
        proof.append(reference.certified_floor(Fraction(4096)/hi, Fraction(4096)/lo))
    assert tuple(proof) == selected
    assert sha256(b''.join(struct.pack('<Q',v) for v in selected)).hexdigest() == expected
    certified.append(proof)
for x in (0,255,256,65535,65536,(1<<38)-1):
    c = min(x,65535)
    value = certified[0][c % 256]*certified[1][c // 256]//4096
    assert 0 <= value <= 4096
assert certified[0][0]*certified[1][0]//4096 == 4096
"#;
    let status = std::process::Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(root)
        .status()
        .expect("Python required for existing independent rational proof");
    assert!(status.success(), "selected table floor proof failed");
}
