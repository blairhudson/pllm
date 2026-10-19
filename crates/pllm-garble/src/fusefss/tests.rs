use super::*;
use serde_json::{json, Value as J};

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut output, byte| {
        write!(output, "{byte:02x}").expect("String formatting");
        output
    })
}
fn descriptor_json(bits: u8, boundaries: &[u64], booleans: &[J]) -> J {
    let n = 1u64 << bits;
    let pieces: Vec<J> = (0..boundaries.len() - 1)
        .map(|i| {
            json!({
                "coefficients":[(i as u64*3+1)%n,(i as u64*5+3)%n],"booleans":booleans
            })
        })
        .collect();
    json!({
        "schema":"pllm.fusefss.scalar-descriptor.v1","profile":SYNTHETIC_PROFILE,
        "gate":"synthetic-affine","source_lock_sha256":"0".repeat(64),
        "input":{"domain":"A","bits":bits,"fractional":0,"signed":true},
        "output":{"arithmetic":"same-as-input","boolean_arity":booleans.len()},
        "external_encoding":"nearest-ties-even-input-and-weight","internal_rounding":"signed-floor-only",
        "partition_order":"unsigned-canonical-half-open","degree":1,
        "coefficient_order":["constant","linear"],"coefficient_fractional":[0,0],
        "selector_semantics":"descriptor-defined","source_schedule":{},"phi":"identity",
        "boundaries":boundaries,"pieces":pieces
    })
}
fn parse(j: &J) -> Result<Descriptor> {
    Descriptor::from_canonical_bytes(&serde_json::to_vec(j).unwrap())
}
fn eval_expr(expr: &J, x: u64, bits: u8) -> bool {
    let a = expr.as_array().unwrap();
    let n = 1u64 << bits;
    match a[0].as_str().unwrap() {
        "const" => a[1].as_u64().unwrap() != 0,
        "lt" => x < a[1].as_u64().unwrap(),
        "low_lt" => x % (1 << a[1].as_u64().unwrap()) < a[2].as_u64().unwrap(),
        "slt" => {
            let s = if x < n / 2 {
                x as i64
            } else {
                x as i64 - n as i64
            };
            s < a[1].as_i64().unwrap()
        }
        "msb" => (x + a[1].as_u64().unwrap()) % n >= n / 2,
        "not" => !eval_expr(&a[1], x, bits),
        "xor" => eval_expr(&a[1], x, bits) ^ eval_expr(&a[2], x, bits),
        "and" => eval_expr(&a[1], x, bits) & eval_expr(&a[2], x, bits),
        "or" => eval_expr(&a[1], x, bits) | eval_expr(&a[2], x, bits),
        _ => panic!("oracle unsupported expression"),
    }
}
// Independent source scan and expanded affine expression, no compiler helpers.
fn unfused(j: &J, x: u64) -> (u64, Vec<bool>) {
    let bits = j["input"]["bits"].as_u64().unwrap() as u8;
    let n = 1u64 << bits;
    for (i, p) in j["pieces"].as_array().unwrap().iter().enumerate() {
        if x >= j["boundaries"][i].as_u64().unwrap() && x < j["boundaries"][i + 1].as_u64().unwrap()
        {
            let c = p["coefficients"][0].as_u64().unwrap();
            let l = p["coefficients"][1].as_u64().unwrap();
            let y = ((u128::from(c) + u128::from(x) * u128::from(l)) % u128::from(n)) as u64;
            return (
                y,
                p["booleans"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|e| eval_expr(e, x, bits))
                    .collect(),
            );
        }
    }
    panic!("not a total partition")
}

#[test]
fn frozen_gelu_table_descriptor_and_source_identity() {
    let table: J = serde_json::from_slice(include_bytes!(
        "../../../../paper/reproduction-contracts/fusefss/gelu-table.json"
    ))
    .unwrap();
    let mut h = Sha256::new();
    for (i, &entry) in fixture::TABLE.iter().enumerate() {
        assert_eq!(u64::from(entry), table["values"][i].as_u64().unwrap());
        h.update(u64::from(entry).to_le_bytes());
    }
    assert_eq!(
        hex(&h.finalize()),
        "b6a45a82dc8f660724943c1eb76654704c2e49563ffc0b1a0bc5388b3425a01b"
    );
    assert_eq!(
        hex(&Sha256::digest(include_bytes!(
            "../../../../paper/reproduction-contracts/fusefss/source-lock.json"
        ))),
        SOURCE_LOCK_SHA256
    );
    let d = Descriptor::gelu().unwrap();
    assert_eq!(d.canonical_bytes().len(), 32900);
    assert_eq!(hex(&d.digest()), GELU_DIGEST_HEX);
    assert_eq!(
        d.shape(),
        Shape {
            source_intervals: 511,
            padded_intervals: 512,
            boolean_outputs: 0,
            queries: 0,
            vm_ops: 0,
            boolean_ands: 0,
            boolean_xors: 0,
            reference_bytes: 11846,
            program_lengths: [0; 4]
        }
    );
    let again = Descriptor::from_canonical_bytes(d.canonical_bytes()).unwrap();
    assert_eq!(again.digest(), d.digest());
    assert_eq!(
        Descriptor::from_frozen_gelu_bytes(d.canonical_bytes())
            .unwrap()
            .digest(),
        d.digest()
    );
    assert!(matches!(
        Descriptor::from_frozen_gelu_bytes(&vec![b' '; GELU_CANONICAL_BYTES]),
        Err(Error::DescriptorIdentity)
    ));
    assert!(matches!(
        Descriptor::from_frozen_gelu_bytes(b"{}"),
        Err(Error::DescriptorIdentity)
    ));
    let mut wrong: J = serde_json::from_slice(d.canonical_bytes()).unwrap();
    wrong["pieces"][0]["coefficients"][0] = json!(1);
    assert!(matches!(parse(&wrong), Err(Error::DescriptorIdentity)));
    wrong = serde_json::from_slice(d.canonical_bytes()).unwrap();
    wrong["source_schedule"]["effective_input"] = json!([50, 12]);
    assert!(matches!(parse(&wrong), Err(Error::DescriptorIdentity)));
}

fn gelu_oracle(x: i64) -> u64 {
    let n = 1i64 << 50;
    let y = x.div_euclid(64);
    let correction = i64::from(fixture::TABLE[y.unsigned_abs().min(255) as usize]);
    ((if y >= 0 { x } else { 0 }) - correction).rem_euclid(n) as u64
}

#[test]
fn gelu_frozen_transcripts_vectors_boundaries_and_central_codes() {
    let vectors: J = serde_json::from_slice(include_bytes!(
        "../../../../paper/reproduction-contracts/fusefss/vectors.json"
    ))
    .unwrap();
    let d = Descriptor::gelu().unwrap();
    let n = 1u64 << 50;
    for (i, mask) in vectors["masks"].as_array().unwrap().iter().enumerate() {
        let mask = mask.as_u64().unwrap();
        let plan = d.compile(mask).unwrap();
        assert_eq!(plan.shape(), d.shape());
        let record = plan.reference_record().unwrap();
        assert_eq!(record.len(), 11846);
        assert_eq!(
            hex(&Sha256::digest(&record)),
            vectors["compiled_trace_sha256_regression_only"][i]
                .as_str()
                .unwrap()
        );
        for (x, expected) in vectors["gelu_inputs_signed_q12"]
            .as_array()
            .unwrap()
            .iter()
            .zip(vectors["gelu_expected_signed_q12"].as_array().unwrap())
        {
            let x = x.as_i64().unwrap();
            let expected = expected.as_i64().unwrap().rem_euclid(n as i64) as u64;
            assert_eq!(gelu_oracle(x), expected);
            assert_eq!(
                plan.clear_interpret((x.rem_euclid(n as i64) as u64 + mask) % n, d.digest())
                    .unwrap(),
                (expected, vec![])
            );
        }
        for &b in &d.boundaries {
            for delta in [-1i64, 0, 1] {
                let x = (b as i64 + delta).rem_euclid(n as i64) as u64;
                let signed = if x < n / 2 {
                    x as i64
                } else {
                    x as i64 - n as i64
                };
                assert_eq!(
                    plan.clear_interpret((x + mask) % n, d.digest()).unwrap().0,
                    gelu_oracle(signed)
                );
            }
        }
        for x in -16385i64..=16385 {
            assert_eq!(
                plan.clear_interpret((x.rem_euclid(n as i64) as u64 + mask) % n, d.digest())
                    .unwrap()
                    .0,
                gelu_oracle(x)
            );
        }
        assert!(matches!(
            plan.clear_interpret(0, [0; 32]),
            Err(Error::DescriptorIdentity)
        ));
        assert!(matches!(
            plan.clear_interpret(n, d.digest()),
            Err(Error::Range)
        ));
    }
}

#[test]
fn every_ring8_partition_mask_and_input() {
    let mut cases = 0;
    for selection in 0..128 {
        let mut bounds = vec![0];
        for b in 1..8 {
            if selection & (1 << (b - 1)) != 0 {
                bounds.push(b);
            }
        }
        bounds.push(8);
        let mut j = descriptor_json(
            3,
            &bounds,
            &[
                json!(["xor", ["lt", 4], ["low_lt", 2, 3]]),
                json!(["or", ["slt", -1], ["msb", 3]]),
                json!(["and", ["not", ["const", 0]], ["lt", 8]]),
                json!(["xor", ["lt", 0], ["const", 1]]),
            ],
        );
        // Make predicates piece-specific too, preventing an accidental global
        // predicate evaluation from passing normalization tests.
        for (i, p) in j["pieces"].as_array_mut().unwrap().iter_mut().enumerate() {
            p["booleans"][3] = json!(["xor", ["lt", i % 9], ["const", i % 2]]);
        }
        let d = parse(&j).unwrap();
        for mask in 0..8 {
            let p = d.compile(mask).unwrap();
            assert_eq!(p.shape(), d.shape());
            assert_eq!(
                p.reference_record().unwrap().len(),
                d.shape().reference_bytes
            );
            for x in 0..8 {
                assert_eq!(
                    p.clear_interpret((x + mask) % 8, d.digest()).unwrap(),
                    unfused(&j, x)
                );
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 8192);
}

#[test]
fn every_small_typed_predicate_threshold_shift_mask_and_value() {
    for bits in 1..=4 {
        let n = 1u64 << bits;
        let mut exprs = vec![json!(["const", 0]), json!(["const", 1])];
        for t in 0..=n {
            exprs.push(json!(["lt", t]));
        }
        for k in 1..=bits {
            for t in 0..=1u64 << k {
                exprs.push(json!(["low_lt", k, t]));
            }
        }
        for t in -(n as i64) / 2..=n as i64 / 2 {
            exprs.push(json!(["slt", t]));
        }
        for c in 0..n {
            exprs.push(json!(["msb", c]));
        }
        for e in exprs {
            let j = descriptor_json(bits, &[0, n], &[e]);
            let d = parse(&j).unwrap();
            for mask in 0..n {
                let p = d.compile(mask).unwrap();
                assert_eq!(p.shape(), d.shape());
                for x in 0..n {
                    assert_eq!(
                        p.clear_interpret((x + mask) % n, d.digest()).unwrap(),
                        unfused(&j, x)
                    );
                }
            }
        }
    }
}

#[test]
fn padding_first_eligible_full_partition_and_zero_query_slots() {
    let d = parse(&descriptor_json(3, &[0, 3, 8], &[])).unwrap();
    for (mask, b, o) in [
        (0, vec![0, 1, 3, 8], vec![0, 0, 1]),
        (2, vec![0, 2, 5, 8], vec![1, 0, 1]),
        (5, vec![0, 1, 5, 8], vec![1, 1, 0]),
    ] {
        let p = d.compile(mask).unwrap();
        assert_eq!(p.test_partition(), (b.as_slice(), o.as_slice()));
    }
    // First interval has length one; must split next eligible interval, not last.
    let d = parse(&descriptor_json(3, &[0, 1, 4, 8], &[])).unwrap();
    assert_eq!(d.compile(0).unwrap().test_partition().0, &[0, 1, 2, 4, 8]);
    let d = parse(&descriptor_json(3, &[0, 1, 2, 3, 4, 5, 6, 7, 8], &[])).unwrap();
    for mask in 0..8 {
        assert_eq!(
            d.compile(mask).unwrap().test_partition().0,
            &[0, 1, 2, 3, 4, 5, 6, 7, 8]
        );
    }
    let d = parse(&descriptor_json(3, &[0, 8], &[json!(["lt", 3])])).unwrap();
    assert_eq!(d.shape().queries, 2);
    assert_eq!(d.shape().vm_ops, 9);
    assert_eq!(d.shape().boolean_ands, 1);
    assert_eq!(d.shape().boolean_xors, 3);
    d.compile(5)
        .unwrap()
        .with_dealer_handoff(|h| {
            assert_eq!(h.predicate_queries[0].threshold(), 0);
            assert_eq!(h.predicate_queries.len(), 2);
            Ok(())
        })
        .unwrap();
    // Original x Horner, deliberately not xhat.
    let d = parse(&descriptor_json(3, &[0, 8], &[])).unwrap();
    assert_eq!(
        d.compile(4)
            .unwrap()
            .clear_interpret(6, d.digest())
            .unwrap()
            .0,
        7
    );
}

#[test]
fn canonical_encoding_and_type_rejections() {
    let j = descriptor_json(3, &[0, 3, 8], &[json!(["const", 0])]);
    let bytes = serde_json::to_vec(&j).unwrap();
    let mut newline = bytes.clone();
    newline.push(b'\n');
    assert!(Descriptor::from_canonical_bytes(&newline).is_err());
    assert!(Descriptor::from_canonical_bytes(&serde_json::to_vec_pretty(&j).unwrap()).is_err());
    for (from, to) in [
        ("\"degree\":1", "\"degree\":1.0"),
        ("\"degree\":1", "\"degree\":1e0"),
        ("\"degree\":1", "\"degree\":true"),
        ("\"fractional\":0", "\"fractional\":-0"),
        ("\"degree\":1", "\"degree\":01"),
        ("\"degree\":1", "\"degree\":NaN"),
        ("\"degree\":1", "\"degree\":1,\"degree\":1"),
        ("\"gate\":", "\"gate\":\"x\",\"gate\":"),
        ("\"domain\":\"A\"", "\"domain\":\"\\u0041\""),
    ] {
        let bad = String::from_utf8(bytes.clone()).unwrap().replace(from, to);
        assert!(
            Descriptor::from_canonical_bytes(bad.as_bytes()).is_err(),
            "{bad}"
        );
    }
    for e in [
        json!(["const", true]),
        json!(["const", 2]),
        json!(["lt", true]),
        json!(["low_lt", 0, 0]),
        json!(["low_lt", 4, 0]),
        json!(["slt", -5]),
        json!(["slt", 5]),
        json!(["msb", 8]),
        json!(["a2b", 0]),
        json!(["and", ["const", 1]]),
    ] {
        assert!(parse(&descriptor_json(3, &[0, 8], &[e])).is_err());
    }
    for key in ["degree", "input", "source_schedule", "unused"] {
        let mut bad = j.clone();
        bad[key] = json!(false);
        assert!(parse(&bad).is_err());
    }
    let mut bad = j.clone();
    bad["pieces"][0]["coefficients"][0] = json!(true);
    assert!(matches!(parse(&bad), Err(Error::Type)));
    let mut bad = j.clone();
    bad["input"]["signed"] = json!(1);
    assert!(matches!(parse(&bad), Err(Error::Type)));
    let mut bad = j.clone();
    bad["phi"] = json!("ars12");
    assert!(matches!(parse(&bad), Err(Error::UnsupportedPhi)));
    let mut bad = j.clone();
    bad["profile"] = json!("different");
    assert!(matches!(parse(&bad), Err(Error::UnsupportedProfile)));
    let mut escaped = j.clone();
    escaped["gate"] = json!("quote\"\\\n\u{0}\u{7f}");
    // Python ensure_ascii (the contract) escapes DEL; serde_json does not.
    let canonical = String::from_utf8(serde_json::to_vec(&escaped).unwrap())
        .unwrap()
        .replace('\u{7f}', "\\u007f");
    assert!(Descriptor::from_canonical_bytes(canonical.as_bytes()).is_ok());
}

#[test]
fn public_bounds_and_work_admission() {
    for bits in [0, 51] {
        assert!(parse(&descriptor_json(bits, &[0, 1], &[])).is_err());
    }
    for bounds in [vec![1, 8], vec![0, 7], vec![0, 3, 3, 8], vec![0, 9]] {
        assert!(parse(&descriptor_json(3, &bounds, &[])).is_err());
    }
    let j = descriptor_json(3, &[0, 8], &[]);
    let d = parse(&j).unwrap();
    assert!(matches!(d.compile(8), Err(Error::Range)));
    assert!(parse(&descriptor_json(3, &[0, 8], &vec![json!(["const", 0]); 5])).is_err());
    let mut depth = json!(["const", 0]);
    for _ in 0..12 {
        depth = json!(["not", depth]);
    }
    assert!(parse(&descriptor_json(3, &[0, 8], &[depth.clone()])).is_ok());
    depth = json!(["not", depth]);
    assert!(matches!(
        parse(&descriptor_json(3, &[0, 8], &[depth])),
        Err(Error::ExpressionBudget)
    ));
    assert!(Descriptor::from_canonical_bytes(&vec![b' '; MAX_DESCRIPTOR_BYTES + 1]).is_err());
    let bounds: Vec<u64> = (0..=513).collect();
    assert!(parse(&descriptor_json(10, &bounds, &[])).is_err());
    // Dense expression exceeds total AST cap before normalization/allocation.
    fn tree(depth: usize) -> J {
        if depth == 0 {
            json!(["const", 0])
        } else {
            json!(["and", tree(depth - 1), tree(depth - 1)])
        }
    }
    let tree = tree(11);
    let exact = descriptor_json(
        3,
        &[0, 8],
        &[
            tree.clone(),
            tree.clone(),
            json!(["const", 0]),
            json!(["const", 1]),
        ],
    );
    assert_eq!(
        parse(&exact).unwrap().expressions.len(),
        MAX_EXPRESSION_NODES
    );
    assert!(matches!(
        parse(&descriptor_json(
            3,
            &[0, 8],
            &[tree.clone(), tree.clone(), tree]
        )),
        Err(Error::ExpressionBudget)
    ));
    // Dense admitted work is still counted before private compilation.
    let bounds: Vec<u64> = (0..512).chain(std::iter::once(1024)).collect();
    let expr = json!([
        "or",
        ["or", ["lt", 1], ["lt", 2]],
        ["or", ["lt", 3], ["lt", 4]]
    ]);
    let dense = descriptor_json(10, &bounds, &[expr.clone(), expr]);
    let dense = parse(&dense).unwrap();
    assert_eq!(dense.shape().source_intervals, 512);
    assert_eq!(dense.shape().padded_intervals, 513);
    assert!(dense.shape().vm_ops > 30_000);
    assert!(dense.shape().queries > 10_000);
    assert_eq!(dense.compile(1023).unwrap().shape(), dense.shape());
    // All supported ring widths, including full ring1 partitions and ring50.
    for bits in 1..=50 {
        let n = 1u64 << bits;
        let d = parse(&descriptor_json(bits, &[0, n], &[json!(["lt", n])])).unwrap();
        for mask in [0, n / 2, n - 1] {
            let p = d.compile(mask).unwrap();
            assert_eq!(p.shape(), d.shape());
            assert_eq!(p.clear_interpret(mask, d.digest()).unwrap().1, vec![true]);
        }
    }
    let n = 1u64 << 50;
    let mut maximum = descriptor_json(50, &[0, n], &[]);
    maximum["pieces"][0]["coefficients"] = json!([n - 1, n - 1]);
    let d = parse(&maximum).unwrap();
    let p = d.compile(n - 2).unwrap();
    // (N-1)*(N-1)+(N-1) = 0 mod N; product exceeds u64.
    assert_eq!(p.clear_interpret(n - 3, d.digest()).unwrap().0, 0);
}
