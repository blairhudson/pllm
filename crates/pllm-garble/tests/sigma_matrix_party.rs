//! Public-API regressions for local research-dealer composition only.
//! No authentication, caller-mask freshness, or request-long replay claim.
use pllm_garble::masked_wire_reference::{
    arithmetic,
    matrix::{
        party::{issue_local, resources, MaskedOutput, Party},
        Budget, Context, Error, MatrixLayout, Spec,
    },
};

fn context() -> Context {
    Context {
        plan: [1; 32],
        session: [2; 32],
        operation: [3; 32],
        input_wire: [4; 32],
        weight_wire: [5; 32],
        output_wire: [6; 32],
    }
}

fn spec(bits: u8, layout: MatrixLayout) -> Spec {
    Spec {
        bits,
        rows: 2,
        inner: 3,
        columns: 2,
        layout,
    }
}

fn modulus(bits: u8) -> u128 {
    1u128 << bits
}

fn residue(value: i128, bits: u8) -> u64 {
    value.rem_euclid(modulus(bits) as i128) as u64
}

// Independent u128 oracle: reduce each product before adding, including ring64.
// Inputs here are mathematical row-major matrices, never evaluator storage.
fn product(s: Spec, x: &[u64], w: &[u64]) -> Vec<u64> {
    let m = modulus(s.bits);
    (0..s.rows)
        .flat_map(|row| {
            (0..s.columns).map(move |column| {
                let mut sum = 0u128;
                for k in 0..s.inner {
                    let term =
                        u128::from(x[row * s.inner + k]) * u128::from(w[k * s.columns + column]);
                    sum = (sum + term % m) % m;
                }
                sum as u64
            })
        })
        .collect()
}

fn weights(s: Spec, row_major: &[u64]) -> Vec<u64> {
    match s.layout {
        MatrixLayout::InputOutput => row_major.to_vec(),
        MatrixLayout::OutputInput => (0..s.columns)
            .flat_map(|column| (0..s.inner).map(move |k| row_major[k * s.columns + column]))
            .collect(),
    }
}

fn masks(len: usize, bits: u8, offset: usize) -> Vec<u64> {
    let m = modulus(bits);
    let choices = [0, m - 1, m / 2, m / 2 - 1, 1, m - 2, 0x1234 % m];
    (0..len)
        .map(|i| choices[(i + offset) % choices.len()] as u64)
        .collect()
}

fn add(values: &[u64], masks: &[u64], bits: u8) -> Vec<u64> {
    assert_eq!(values.len(), masks.len());
    values
        .iter()
        .zip(masks)
        .map(|(&x, &r)| ((u128::from(x) + u128::from(r)) % modulus(bits)) as u64)
        .collect()
}

fn unmask(values: &[u64], masks: &[u64], bits: u8) -> Vec<u64> {
    assert_eq!(values.len(), masks.len());
    values
        .iter()
        .zip(masks)
        .map(|(&x, &r)| residue(i128::from(x) - i128::from(r), bits))
        .collect()
}

fn run_matrix<'b>(
    parties: &mut [Party<'b>; 2],
    xhat: &[u64],
    what: &[u64],
) -> [MaskedOutput<'b>; 2] {
    let [a, b] = parties;
    let left = a.begin(xhat, what, &mut || false).unwrap();
    let right = b.begin(xhat, what, &mut || false).unwrap();
    [
        a.finish(right.as_bytes()).unwrap(),
        b.finish(left.as_bytes()).unwrap(),
    ]
}

fn arithmetic_context(input: [u8; 32], output: u8) -> arithmetic::Context {
    arithmetic::Context {
        plan: context().plan,
        session: context().session,
        operation: [output; 32],
        input_wire: input,
        output_wire: [output; 32],
        first_tensor_index: 0,
    }
}

fn arithmetic_output(step: arithmetic::Step<'_>) -> arithmetic::MaskedOutput<'_> {
    match step {
        arithmetic::Step::Output(output) => output,
        arithmetic::Step::Frame(_) => panic!("expected arithmetic output"),
    }
}

fn run_arithmetic<'b>(
    parties: &mut [arithmetic::Party<'b>; 2],
    values: &[u64],
) -> [arithmetic::MaskedOutput<'b>; 2] {
    let [a, b] = parties;
    let left = a.begin(values).unwrap();
    let right = b.begin(values).unwrap();
    match (
        a.advance(right.as_bytes()).unwrap(),
        b.advance(left.as_bytes()).unwrap(),
    ) {
        (arithmetic::Step::Output(a), arithmetic::Step::Output(b)) => [a, b],
        (arithmetic::Step::Frame(left), arithmetic::Step::Frame(right)) => [
            arithmetic_output(a.advance(right.as_bytes()).unwrap()),
            arithmetic_output(b.advance(left.as_bytes()).unwrap()),
        ],
        _ => panic!("arithmetic party schedules diverged"),
    }
}

fn signed_composition(layout: MatrixLayout) {
    let s = spec(16, layout);
    let x_signed = [-3i128, 2, 1, 1, -2, 3];
    let w_signed = [2i128, -1, -3, 2, 1, 1];
    let expected: Vec<i128> = (0..s.rows)
        .flat_map(|row| {
            let x = &x_signed;
            let w = &w_signed;
            (0..s.columns).map(move |column| {
                (0..s.inner)
                    .map(|k| x[row * s.inner + k] * w[k * s.columns + column])
                    .sum()
            })
        })
        .collect();
    assert_eq!(expected, [-11, 8, 11, -2]);
    let x: Vec<_> = x_signed.map(|v| residue(v, s.bits)).into();
    let w: Vec<_> = w_signed.map(|v| residue(v, s.bits)).into();
    let rx = masks(x.len(), s.bits, 0);
    let rw = masks(w.len(), s.bits, 1);
    let ry = masks(expected.len(), s.bits, 2);
    let xhat = add(&x, &rx, s.bits);
    let what = weights(s, &add(&w, &rw, s.bits));
    let budget = Budget::new(64 << 20).unwrap();
    let mut parties = issue_local(s, context(), &rx, &weights(s, &rw), &ry, &budget).unwrap();
    let outputs = run_matrix(&mut parties, &xhat, &what);
    assert_eq!(outputs[0].values(), outputs[1].values());
    let expected_ring: Vec<_> = expected.iter().map(|&v| residue(v, s.bits)).collect();
    assert_eq!(product(s, &x, &w), expected_ring);
    assert_eq!(outputs[0].values(), add(&expected_ring, &ry, s.bits));

    // Dealer alone receives RY. Holders receive only common masked matrix output.
    // These clear fixture values independently satisfy the quarter-ring promise.
    assert!(expected.iter().all(|&v| (-16384..16384).contains(&v)));
    let gap = arithmetic::Spec {
        bits: s.bits,
        lanes: expected.len(),
        gate: arithmetic::Gate::GapArs { shift: 2 },
    };
    let (gap_masks, mut gap_parties) = arithmetic::issue(
        gap,
        arithmetic_context(context().output_wire, 7),
        &ry,
        &budget,
    )
    .unwrap();
    let floors = run_arithmetic(&mut gap_parties, outputs[0].values());
    assert_eq!(floors[0].values(), floors[1].values());
    let expected_floors: Vec<_> = expected.iter().map(|v| v.div_euclid(4)).collect();
    assert_eq!(expected_floors, [-3, 2, 2, -1]);
    assert_eq!(
        unmask(floors[0].values(), gap_masks.values(), 16),
        expected_floors
            .iter()
            .map(|&v| residue(v, 16))
            .collect::<Vec<_>>()
    );
    let (sign_masks, mut sign_parties) = arithmetic::issue(
        arithmetic::Spec {
            gate: arithmetic::Gate::Drelu,
            ..gap
        },
        arithmetic_context([7; 32], 8),
        gap_masks.values(),
        &budget,
    )
    .unwrap();
    let signs = run_arithmetic(&mut sign_parties, floors[0].values());
    assert_eq!(signs[0].values(), signs[1].values());
    let clear_signs: Vec<_> = signs[0]
        .values()
        .iter()
        .zip(sign_masks.values())
        .map(|(&v, &r)| v ^ r)
        .collect();
    assert_eq!(clear_signs, [0, 1, 1, 0]);
    assert_eq!(
        clear_signs,
        expected_floors
            .iter()
            .map(|&v| u64::from(v >= 0))
            .collect::<Vec<_>>()
    );

    // Separate matrix -> TR branch, not GapARS -> TR. Signed floor is exact
    // only modulo TR's reduced output ring; no full-ring signed TR claim.
    let (tr_masks, mut tr_parties) = arithmetic::issue(
        arithmetic::Spec {
            gate: arithmetic::Gate::Tr { shift: 2 },
            ..gap
        },
        arithmetic_context(context().output_wire, 9),
        &ry,
        &budget,
    )
    .unwrap();
    let truncated = run_arithmetic(&mut tr_parties, outputs[0].values());
    assert_eq!(truncated[0].values(), truncated[1].values());
    assert!(truncated[0]
        .values()
        .iter()
        .all(|&v| u128::from(v) < modulus(14)));
    assert_eq!(
        unmask(truncated[0].values(), tr_masks.values(), 14),
        expected_floors
            .iter()
            .map(|&v| residue(v, 14))
            .collect::<Vec<_>>()
    );
    drop((
        parties,
        gap_parties,
        sign_parties,
        tr_parties,
        gap_masks,
        sign_masks,
        tr_masks,
    ));
    assert!(budget.used_bytes() > 0);
    drop((outputs, floors, signs, truncated));
    assert_eq!(budget.used_bytes(), 0);
}

fn ring64_rectangular(layout: MatrixLayout) {
    let s = Spec {
        rows: 3,
        inner: 2,
        columns: 4,
        ..spec(64, layout)
    };
    let x = [
        u64::MAX,
        1 << 63,
        0,
        17,
        u64::MAX - 1,
        0x1234_5678_9abc_def0,
    ];
    let w = [u64::MAX, 2, 1 << 63, 0, 3, u64::MAX - 2, 5, u64::MAX];
    let rx = masks(x.len(), 64, 0);
    let rw = masks(w.len(), 64, 3);
    let ry = masks(s.rows * s.columns, 64, 1);
    let budget = Budget::new(resources(s).unwrap().reservation_bytes).unwrap();
    let mut parties = issue_local(s, context(), &rx, &weights(s, &rw), &ry, &budget).unwrap();
    let outputs = run_matrix(
        &mut parties,
        &add(&x, &rx, 64),
        &weights(s, &add(&w, &rw, 64)),
    );
    assert_eq!(outputs[0].values(), outputs[1].values());
    assert_eq!(outputs[0].values(), add(&product(s, &x, &w), &ry, 64));
    drop((parties, outputs));
    assert_eq!(budget.used_bytes(), 0);
}

type Fixture = (Spec, Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>);

fn fixture() -> Fixture {
    let s = spec(16, MatrixLayout::InputOutput);
    let rx = masks(6, 16, 0);
    let rw = masks(6, 16, 1);
    let ry = masks(4, 16, 2);
    let xhat = add(&[65533, 2, 1, 1, 65534, 3], &rx, 16);
    let what = add(&[2, 65535, 65533, 2, 1, 1], &rw, 16);
    (s, rx, rw, ry, xhat, what)
}

fn admission_and_retained_handles() {
    let (s, rx, rw, ry, xhat, what) = fixture();
    let required = resources(s).unwrap().reservation_bytes;
    let short = Budget::new(required - 1).unwrap();
    assert!(matches!(
        issue_local(s, context(), &rx, &rw, &ry, &short),
        Err(Error::Resource)
    ));
    assert_eq!(short.used_bytes(), 0);

    let budget = Budget::new(required).unwrap();
    let other = Budget::new(required).unwrap();
    let [mut a, mut b] = issue_local(s, context(), &rx, &rw, &ry, &budget).unwrap();
    // Admission reserves the complete peak. Dealer scratch is already erased
    // and refunded when the two independent party owners are returned.
    assert_eq!(
        budget.used_bytes(),
        2 * resources(s).unwrap().party_reservation_bytes
    );
    let left = a.begin(&xhat, &what, &mut || false).unwrap();
    let right = b.begin(&xhat, &what, &mut || false).unwrap();
    let out_a = a.finish(right.as_bytes()).unwrap();
    let out_b = b.finish(left.as_bytes()).unwrap();
    assert_eq!(out_a.values(), out_b.values());
    assert!(matches!(a.finish(right.as_bytes()), Err(Error::Consumed)));
    assert!(b.begin(&xhat, &what, &mut || false).is_err());
    drop((a, b, right, out_a));
    assert!(budget.used_bytes() > 0);
    // Different budget isolates the singleton rejection from budget exhaustion.
    assert!(matches!(
        issue_local(s, context(), &rx, &rw, &ry, &other),
        Err(Error::Resource)
    ));
    assert_eq!(other.used_bytes(), 0);
    drop(out_b);
    assert!(budget.used_bytes() > 0, "retained frame remains charged");
    assert!(matches!(
        issue_local(s, context(), &rx, &rw, &ry, &other),
        Err(Error::Resource)
    ));
    drop(left);
    assert_eq!(budget.used_bytes(), 0);

    // Reverse retirement order: output alone must retain singleton and budget.
    let mut parties = issue_local(s, context(), &rx, &rw, &ry, &budget).unwrap();
    let [out_a, out_b] = run_matrix(&mut parties, &xhat, &what);
    drop((parties, out_a));
    assert!(budget.used_bytes() > 0);
    assert!(matches!(
        issue_local(s, context(), &rx, &rw, &ry, &other),
        Err(Error::Resource)
    ));
    drop(out_b);
    assert_eq!(budget.used_bytes(), 0);
    drop(issue_local(s, context(), &rx, &rw, &ry, &other).unwrap());
    assert_eq!(other.used_bytes(), 0);
}

fn failed_calls_retire_only_the_local_party() {
    let (s, rx, rw, ry, xhat, what) = fixture();
    let budget = Budget::new(resources(s).unwrap().reservation_bytes).unwrap();
    // Each failure gets independent issuance. Caller explicitly retires sibling.
    for case in 0..9 {
        let [mut a, mut b] = issue_local(s, context(), &rx, &rw, &ry, &budget).unwrap();
        if case == 0 {
            assert!(matches!(
                a.begin(&xhat, &what, &mut || true),
                Err(Error::Consumed)
            ));
            assert!(a.begin(&xhat, &what, &mut || false).is_err());
            b.cancel();
        } else if case == 1 {
            assert!(a.finish(&[]).is_err());
            assert!(a.begin(&xhat, &what, &mut || false).is_err());
            b.cancel();
        } else {
            let left = a.begin(&xhat, &what, &mut || false).unwrap();
            let mut peer_x = xhat.clone();
            let mut peer_w = what.clone();
            if case == 3 {
                peer_x[0] ^= 1;
            }
            if case == 4 {
                peer_w[0] ^= 1;
            }
            let right = b.begin(&peer_x, &peer_w, &mut || false).unwrap();
            let mut peer = right.as_bytes().to_vec();
            match case {
                2 => peer = left.as_bytes().to_vec(), // self/wrong role
                3 | 4 => {}                           // different common operand digest
                5 => {
                    peer.pop().unwrap();
                }
                6 => peer.push(0),    // overlong, noncanonical framing
                7 => peer.fill(0xff), // invalid encoding, no header offsets assumed
                8 => {
                    assert!(a.begin(&xhat, &what, &mut || false).is_err());
                }
                _ => unreachable!(),
            }
            assert!(a.finish(&peer).is_err(), "failure case {case}");
            assert!(matches!(a.finish(right.as_bytes()), Err(Error::Consumed)));
            b.cancel();
            drop((left, right));
        }
        a.cancel();
        drop((a, b));
        assert_eq!(budget.used_bytes(), 0, "failure case {case}");
    }
}

fn stale_frames_and_session_mismatch() {
    let (s, rx, rw, ry, xhat, what) = fixture();
    let budget = Budget::new(resources(s).unwrap().reservation_bytes).unwrap();
    let stale = {
        let [mut a, mut b] = issue_local(s, context(), &rx, &rw, &ry, &budget).unwrap();
        let frame = b.begin(&xhat, &what, &mut || false).unwrap();
        let bytes = frame.as_bytes().to_vec();
        a.cancel();
        b.cancel();
        bytes
    };
    assert_eq!(budget.used_bytes(), 0);
    for different_session in [false, true] {
        let mut ctx = context();
        if different_session {
            ctx.session = [99; 32];
        }
        let [mut a, mut b] = issue_local(s, ctx, &rx, &rw, &ry, &budget).unwrap();
        let frame = a.begin(&xhat, &what, &mut || false).unwrap();
        assert!(a.finish(&stale).is_err());
        assert!(matches!(a.finish(&stale), Err(Error::Consumed)));
        b.cancel();
        drop((a, b, frame));
        assert_eq!(budget.used_bytes(), 0);
    }
}

// One test serializes every case against the process-wide single-live matrix
// gate, including retained frames/outputs. No private implementation API used.
#[test]
fn public_party_local_matrix_numeric_composition_and_lifecycle() {
    for layout in [MatrixLayout::InputOutput, MatrixLayout::OutputInput] {
        signed_composition(layout);
        ring64_rectangular(layout);
    }
    admission_and_retained_handles();
    failed_calls_retire_only_the_local_party();
    stale_frames_and_session_mismatch();
}
