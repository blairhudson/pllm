use super::*;
use serde_json::{json, Value};

fn context() -> Context {
    Context {
        plan: [1; 32],
        session: [2; 32],
        operation: [3; 32],
        first_tensor_index: 4,
    }
}

fn descriptor(bits: u8) -> (Descriptor, Value) {
    let n = 1u64 << bits;
    let expressions = [
        json!([
            "or",
            ["and", ["low_lt", 2, 3], ["not", ["slt", -1]]],
            ["xor", ["msb", n - 1], ["lt", n / 2 + 1]]
        ]),
        json!(["xor", ["not", ["lt", 0]], ["and", ["lt", n], ["msb", 1]]]),
        json!([
            "or",
            ["low_lt", 2, 0],
            ["and", ["low_lt", 2, 4], ["slt", -(n as i64) / 2]]
        ]),
        json!(["xor", ["slt", n / 2 - 1], ["not", ["const", 1]]]),
    ];
    let boundaries = [0, 1, n / 2 + 1, n];
    let pieces: Vec<_> = (0..3)
        .map(|i| {
            json!({"coefficients":[i, 1],
        "booleans": (0..4).map(|j| expressions[(j+i)%4].clone()).collect::<Vec<_>>() })
        })
        .collect();
    let d = json!({
        "schema":"pllm.fusefss.scalar-descriptor.v1", "profile":fusefss::SYNTHETIC_PROFILE,
        "gate":"protected-boolean-conformance", "source_lock_sha256":"0".repeat(64),
        "input":{"domain":"A", "bits":bits, "fractional":0, "signed":true},
        "output":{"arithmetic":"same-as-input", "boolean_arity":4},
        "external_encoding":"nearest-ties-even-input-and-weight", "internal_rounding":"signed-floor-only",
        "partition_order":"unsigned-canonical-half-open", "degree":1,
        "coefficient_order":["constant","linear"], "coefficient_fractional":[0,0],
        "selector_semantics":"descriptor-defined", "source_schedule":{}, "phi":"identity",
        "boundaries":boundaries, "pieces":pieces
    });
    (
        Descriptor::from_canonical_bytes(&serde_json::to_vec(&d).unwrap()).unwrap(),
        d,
    )
}

// Independent AST oracle: never calls compile/DealerPlan/reference interpreter.
fn oracle(e: &Value, x: u64, bits: u8) -> bool {
    let n = 1u64 << bits;
    let a = e.as_array().unwrap();
    match a[0].as_str().unwrap() {
        "const" => a[1].as_u64().unwrap() != 0,
        "lt" => x < a[1].as_u64().unwrap(),
        "low_lt" => x % (1u64 << a[1].as_u64().unwrap()) < a[2].as_u64().unwrap(),
        "slt" => {
            let signed = if x < n / 2 {
                x as i64
            } else {
                x as i64 - n as i64
            };
            signed < a[1].as_i64().unwrap()
        }
        "msb" => (x + a[1].as_u64().unwrap()) % n >= n / 2,
        "not" => !oracle(&a[1], x, bits),
        "xor" => oracle(&a[1], x, bits) ^ oracle(&a[2], x, bits),
        "and" => oracle(&a[1], x, bits) & oracle(&a[2], x, bits),
        "or" => oracle(&a[1], x, bits) | oracle(&a[2], x, bits),
        _ => panic!("unknown expression"),
    }
}

fn run(d: &Descriptor, j: &Value, x: u64, mask: u64, budget: &Budget) {
    let r = resources(d).unwrap();
    let [mut a, mut b] = issue_with_mask(d, [9; 32], context(), budget, mask).unwrap();
    assert_eq!(budget.used_bytes(), 2 * r.party_reservation_bytes);
    let n = 1u64 << d.word().bits;
    let left = random().unwrap() % n;
    let af = a.start(left).unwrap();
    let bf = b.start(x.wrapping_sub(left) & (n - 1)).unwrap();
    let mut frames_a = vec![af];
    let mut frames_b = vec![bf];
    let (ao, bo) = loop {
        let sa = a.advance(frames_b.last().unwrap().as_bytes()).unwrap();
        let sb = b.advance(frames_a.last().unwrap().as_bytes()).unwrap();
        match (sa, sb) {
            (Step::Frame(af), Step::Frame(bf)) => {
                frames_a.push(af);
                frames_b.push(bf);
            }
            (Step::Output(ao), Step::Output(bo)) => break (ao, bo),
            _ => panic!("secret-dependent peer shape"),
        }
    };
    assert_eq!(frames_a.len(), r.frames_per_party);
    assert_eq!(frames_b.len(), r.frames_per_party);
    assert_eq!(
        frames_a
            .iter()
            .chain(&frames_b)
            .map(|f| f.as_bytes().len())
            .sum::<usize>(),
        r.total_peer_frame_bytes
    );
    assert_eq!(r.dcf_keys_per_party, d.shape().queries);
    assert_eq!(r.triples_per_party, d.shape().boolean_ands);
    let piece = j["boundaries"]
        .as_array()
        .unwrap()
        .windows(2)
        .position(|w| w[0].as_u64().unwrap() <= x && x < w[1].as_u64().unwrap())
        .unwrap();
    for (i, (a, b)) in ao.shares().iter().zip(bo.shares()).enumerate() {
        assert!(*a <= 1 && *b <= 1);
        assert_eq!(
            *a ^ *b,
            u64::from(oracle(&j["pieces"][piece]["booleans"][i], x, d.word().bits)),
            "x={x} mask={mask} output={i}"
        );
    }
    assert!(a.is_consumed() && b.is_consumed());
    assert!(matches!(
        a.advance(frames_b[0].as_bytes()),
        Err(Error::Consumed)
    ));
    drop((a, b));
    assert_eq!(
        budget.used_bytes(),
        r.total_peer_frame_bytes + 2 * r.output_bytes_per_party
    );
    drop((frames_a, frames_b, ao, bo));
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn exhaustive_three_bit_language_and_mask_wrap_with_exact_frames() {
    let (d, j) = descriptor(3);
    let r = resources(&d).unwrap();
    assert_eq!(d.shape().queries, 68);
    assert_eq!(d.shape().vm_ops, 267);
    assert_eq!(d.shape().boolean_ands, 27);
    assert_eq!(r.frames_per_party, 28);
    assert_eq!(r.total_peer_frame_bytes, 5360);
    let budget = Budget::new(r.combined_reservation_bytes).unwrap();
    for mask in 0..8 {
        for x in 0..8 {
            run(&d, &j, x, mask, &budget);
        }
    }
}

#[test]
fn ring50_endpoints_partitions_signed_and_shift_carries() {
    let (d, j) = descriptor(50);
    let r = resources(&d).unwrap();
    assert_eq!(r.dcf_keys_per_party, 68);
    assert_eq!(r.triples_per_party, 27);
    assert_eq!(r.total_peer_frame_bytes, 5360);
    let n = 1u64 << 50;
    let budget = Budget::new(r.combined_reservation_bytes).unwrap();
    for mask in [0, 1, n / 2, n - 1] {
        for x in [0, 1, 2, 3, n / 2 - 1, n / 2, n / 2 + 1, n - 2, n - 1] {
            run(&d, &j, x, mask, &budget);
        }
    }
}

#[test]
fn metadata_and_resource_shape_never_depend_on_mask() {
    let (d, _) = descriptor(3);
    let budget = Budget::new(super::super::MAX_ALLOCATION_BYTES).unwrap();
    let mut public = None;
    for mask in 0..8 {
        let [a, b] = issue_inner(&d, [9; 32], context(), &budget, || Ok(mask)).unwrap();
        let State::Ready(m) = a.state.as_ref().unwrap() else {
            panic!()
        };
        let metadata = (
            m.queries
                .iter()
                .map(|q| (q.bits, q.shift))
                .collect::<Vec<_>>(),
            m.code
                .iter()
                .map(|[op, arg]| (*op, if *op == 0 { 0 } else { *arg }))
                .collect::<Vec<_>>(),
        );
        if let Some(old) = &public {
            assert_eq!(&metadata, old);
        } else {
            public = Some(metadata);
        }
        let State::Ready(r) = b.state.as_ref().unwrap() else {
            panic!()
        };
        // Every Bit, including mask carry operands, is shared rather than copied.
        d.compile(mask)
            .unwrap()
            .with_dealer_handoff(|h| {
                for (i, ins) in h.boolean_program.iter().enumerate() {
                    if ins.opcode() == Opcode::Bit {
                        assert_eq!(m.code[i][1] ^ r.code[i][1], u64::from(ins.argument()));
                    }
                }
                Ok(())
            })
            .unwrap();
        drop((a, b));
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn burns_on_order_role_replay_malformed_context_and_cancel() {
    let (d, _) = descriptor(3);
    let budget = Budget::new(super::super::MAX_ALLOCATION_BYTES).unwrap();
    for mode in 0..9 {
        let [mut a, mut b] = issue(&d, [9; 32], context(), &budget).unwrap();
        let af = a.start(0).unwrap();
        let bf = b.start(0).unwrap();
        let mut wire = bf.as_bytes().to_vec();
        match mode {
            0 => wire.clear(),
            1 => wire[8] ^= 1,
            2 => wire[40] ^= 1,
            3 => wire[73] = 0,
            4 => wire[76] = 7,
            5 => wire[80] = 8,
            6 => {
                a.cancel();
            }
            7 => {
                assert!(matches!(a.start(0), Err(Error::Context)));
            }
            8 => {
                let _ = a.advance(bf.as_bytes()).unwrap();
                // First product round expects round=1, rejects masked round=0.
            }
            _ => unreachable!(),
        }
        assert!(a.advance(&wire).is_err());
        assert!(a.is_consumed());
        assert!(matches!(a.start(0), Err(Error::Consumed)));
        drop((a, b, af, bf));
        assert_eq!(budget.used_bytes(), 0);
    }
    let [mut a, _b] = issue(&d, [9; 32], context(), &budget).unwrap();
    assert!(matches!(a.advance(&[]), Err(Error::Context)));
    assert!(a.is_consumed());
}

#[test]
fn fresh_issuance_source_descriptor_and_context_bindings() {
    let (d, mut public) = descriptor(3);
    // Same word width and VM shape: only the actual descriptor digest differs.
    public["gate"] = json!("distinct-public-descriptor");
    let other = Descriptor::from_canonical_bytes(&serde_json::to_vec(&public).unwrap()).unwrap();
    let budget = Budget::new(super::super::MAX_ALLOCATION_BYTES).unwrap();
    for mode in 0..4 {
        let [mut a, _b] = issue(&d, [9; 32], context(), &budget).unwrap();
        let mut ctx = context();
        if mode == 3 {
            ctx.operation[0] ^= 1;
        }
        let [_c, mut foreign] = issue(
            if mode == 2 { &other } else { &d },
            if mode == 1 { [8; 32] } else { [9; 32] },
            ctx,
            &budget,
        )
        .unwrap();
        if mode != 0 {
            foreign.nonce = a.nonce;
        }
        let _af = a.start(0).unwrap();
        let ff = foreign.start(0).unwrap();
        assert!(matches!(a.advance(ff.as_bytes()), Err(Error::Context)));
        assert!(a.is_consumed());
    }
}

#[test]
fn reservation_precedes_randomness_and_all_failures_release_owners() {
    use std::sync::atomic::Ordering;
    let (d, _) = descriptor(3);
    let r = resources(&d).unwrap();
    let small = Budget::new(r.combined_reservation_bytes - 1).unwrap();
    assert!(matches!(
        issue_inner(&d, [0; 32], context(), &small, || panic!("preflight RNG")),
        Err(Error::Resource)
    ));
    assert_eq!(small.used_bytes(), 0);
    let budget = Budget::new(r.combined_reservation_bytes).unwrap();
    let mut reached_success = false;
    for fail in 0..(d.shape().queries + 20) {
        budget.fail_after.store(fail as isize, Ordering::SeqCst);
        let result = issue(&d, [0; 32], context(), &budget);
        match result {
            Err(Error::Allocation) => (),
            Ok(parties) => {
                drop(parties);
                reached_success = true;
                break;
            }
            _ => panic!("unexpected issuance failure"),
        }
        assert_eq!(budget.used_bytes(), 0);
    }
    assert!(reached_success);
    budget.fail_after.store(-1, Ordering::SeqCst);
    let [mut a, mut b] = issue(&d, [0; 32], context(), &budget).unwrap();
    let af = a.start(0).unwrap();
    let bf = b.start(0).unwrap();
    budget.fail_after.store(0, Ordering::SeqCst);
    assert!(matches!(a.advance(bf.as_bytes()), Err(Error::Allocation)));
    assert!(a.is_consumed());
    b.cancel();
    drop((a, b, af, bf));
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn each_online_allocation_failure_burns_without_unreserved_export() {
    use std::sync::atomic::Ordering;
    let (d, _) = descriptor(3);
    let r = resources(&d).unwrap();
    let budget = Budget::new(r.combined_reservation_bytes).unwrap();
    // First advance allocates query shares, stack and output backing; every
    // nonlinear gate then allocates exactly one frame. Final output reuses its
    // already reserved backing, never a deferred unreserved allocation.
    for fail in 0..(3 + r.triples_per_party) {
        budget.fail_after.store(-1, Ordering::SeqCst);
        let [mut a, mut b] = issue(&d, [0; 32], context(), &budget).unwrap();
        let mut af = a.start(0).unwrap();
        let mut bf = b.start(0).unwrap();
        let mut remaining = fail;
        loop {
            budget
                .fail_after
                .store(remaining as isize, Ordering::SeqCst);
            let result = a.advance(bf.as_bytes());
            remaining = budget.fail_after.load(Ordering::SeqCst) as usize;
            budget.fail_after.store(-1, Ordering::SeqCst);
            match result {
                Err(Error::Allocation) => {
                    assert!(a.is_consumed());
                    break;
                }
                Ok(Step::Frame(next_a)) => {
                    let Step::Frame(next_b) = b.advance(af.as_bytes()).unwrap() else {
                        panic!()
                    };
                    af = next_a;
                    bf = next_b;
                }
                _ => panic!("allocation fault did not hit reserved online allocation"),
            }
        }
        drop((a, b, af, bf));
        assert_eq!(budget.used_bytes(), 0);
    }
    budget.fail_after.store(-1, Ordering::SeqCst);
    let [mut a, b] = issue(&d, [0; 32], context(), &budget).unwrap();
    assert!(matches!(a.start(8), Err(Error::Domain)));
    assert!(a.is_consumed());
    drop((a, b));
    assert_eq!(budget.used_bytes(), 0);
}
