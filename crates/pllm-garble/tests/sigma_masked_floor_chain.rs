//! Public Rust API composition only: local research dealer, not authenticated
//! parties, compiler-bound provenance, model execution or SDK qualification.
use pllm_garble::masked_wire_reference::arithmetic::{
    issue, Budget, Context, Gate, MaskedOutput, Party, Spec, Step,
};

fn context(index: u8) -> Context {
    Context {
        plan: [1; 32],
        session: [2; 32],
        operation: [index; 32],
        input_wire: [index; 32],
        output_wire: [index + 1; 32],
        first_tensor_index: 0,
    }
}

fn output(step: Step<'_>) -> MaskedOutput<'_> {
    match step {
        Step::Output(output) => output,
        Step::Frame(_) => panic!("expected masked output"),
    }
}

fn run<'a>(parties: &mut [Party<'a>; 2], values: &[u64]) -> [MaskedOutput<'a>; 2] {
    let [a, b] = parties;
    let left = a.begin(values).unwrap();
    let right = b.begin(values).unwrap();
    match (
        a.advance(right.as_bytes()).unwrap(),
        b.advance(left.as_bytes()).unwrap(),
    ) {
        (Step::Output(a), Step::Output(b)) => [a, b],
        (Step::Frame(left), Step::Frame(right)) => [
            output(a.advance(right.as_bytes()).unwrap()),
            output(b.advance(left.as_bytes()).unwrap()),
        ],
        _ => panic!("party schedules diverged"),
    }
}

#[test]
fn caller_composes_tr_gapars_drelu_with_fresh_output_masks_and_exact_floors() {
    let budget = Budget::new(64 << 20).unwrap();
    let input = [-12_801i64, -257, -1, 0, 255, 256, 12_801];
    // Public fixture masks; production input-mask sampling/custody is explicitly
    // outside this primitive. Output masks are native OS-random on every gate.
    let rin = [0xffff_ffff, 13, 0x1234_5678, 0, 256, 255, 42];
    let masked = input
        .iter()
        .zip(rin)
        .map(|(&x, mask)| (x as u64).wrapping_add(mask) & 0xffff_ffff)
        .collect::<Vec<_>>();
    let tr = Spec {
        bits: 32,
        lanes: input.len(),
        gate: Gate::Tr { shift: 8 },
    };
    let (tr_masks, mut tr_parties) = issue(tr, context(3), &rin, &budget).unwrap();
    let tr_outputs = run(&mut tr_parties, &masked);
    assert_eq!(tr_outputs[0].values(), tr_outputs[1].values());
    let gap = Spec {
        bits: 24,
        gate: Gate::GapArs { shift: 2 },
        ..tr
    };
    // Clear fixture floors lie in [-51,50], proving this test's quarter-ring
    // promise independently. The API neither reconstructs nor enforces it.
    let (gap_masks, mut gap_parties) = issue(gap, context(4), tr_masks.values(), &budget).unwrap();
    let gap_outputs = run(&mut gap_parties, tr_outputs[0].values());
    assert_eq!(gap_outputs[0].values(), gap_outputs[1].values());
    for (i, &x) in input.iter().enumerate() {
        let actual = gap_outputs[0].values()[i].wrapping_sub(gap_masks.values()[i]) & 0xff_ffff;
        assert_eq!(actual, x.div_euclid(1024) as u64 & 0xff_ffff);
    }
    let sign = Spec {
        gate: Gate::Drelu,
        ..gap
    };
    let (sign_masks, mut sign_parties) =
        issue(sign, context(5), gap_masks.values(), &budget).unwrap();
    let signs = run(&mut sign_parties, gap_outputs[0].values());
    assert_eq!(signs[0].values(), signs[1].values());
    for (i, &x) in input.iter().enumerate() {
        assert_eq!(
            signs[0].values()[i] ^ sign_masks.values()[i],
            u64::from(x >= 0)
        );
    }
    drop((
        tr_parties,
        gap_parties,
        sign_parties,
        tr_masks,
        gap_masks,
        sign_masks,
    ));
    assert!(
        budget.used_bytes() > 0,
        "outputs retain their own reservations"
    );
    drop((tr_outputs, gap_outputs, signs));
    assert_eq!(budget.used_bytes(), 0);
}
