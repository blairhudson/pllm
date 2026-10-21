//! Public caller composes a masked predicate with direct Select. Local trusted
//! dealer fixture only; fixed masks here are test vectors, not a freshness claim.

use pllm_garble::masked_wire_reference::{arithmetic, selection};

#[test]
fn separately_owned_drelu_and_select_preserve_common_masked_representation() {
    let budget = arithmetic::Budget::new(1 << 20).unwrap();
    let clear = [-32768i64, -17, -1, 0, 1, 17, 32767];
    let input_masks = [1u64, 65535, 42, 32768, 31415, 0, 456];
    let masked: Vec<u64> = clear
        .iter()
        .zip(input_masks)
        .map(|(&x, r)| (x as u64).wrapping_add(r) & 65535)
        .collect();
    let spec = arithmetic::Spec {
        bits: 16,
        lanes: clear.len(),
        gate: arithmetic::Gate::Drelu,
    };
    let context = arithmetic::Context {
        plan: [1; 32],
        session: [2; 32],
        operation: [3; 32],
        input_wire: [4; 32],
        output_wire: [5; 32],
        first_tensor_index: 0,
    };
    let (predicate_masks, [mut d0, mut d1]) =
        arithmetic::issue(spec, context, &input_masks, &budget).unwrap();
    let f0 = d0.begin(&masked).unwrap();
    let f1 = d1.begin(&masked).unwrap();
    let arithmetic::Step::Output(p0) = d0.advance(f1.as_bytes()).unwrap() else {
        panic!("DReLU has one exchange")
    };
    let arithmetic::Step::Output(p1) = d1.advance(f0.as_bytes()).unwrap() else {
        panic!("DReLU has one exchange")
    };
    drop((d0, d1, f0, f1));

    // Trusted dealer sees mask owners; each evaluator sees only its own material
    // and the already common-masked inputs, never the dealer masks.
    let spec = selection::Spec {
        bits: 16,
        lanes: clear.len(),
    };
    // A separately encoded ring root, not a second capability for the DReLU
    // input wire. Distinct wire identities use distinct mask vectors.
    let selected_masks = [99u64, 13, 65530, 1024, 19, 7, 12345];
    let selected: Vec<u64> = clear
        .iter()
        .zip(selected_masks)
        .map(|(&x, r)| (x as u64).wrapping_add(r) & 65535)
        .collect();
    let context = selection::Context {
        plan: [1; 32],
        session: [2; 32],
        operation: [6; 32],
        predicate_wire: [5; 32],
        input_wire: [7; 32],
        output_wire: [8; 32],
        first_tensor_index: 0,
    };
    let (output_masks, [mut s0, mut s1]) = selection::issue(
        spec,
        context,
        predicate_masks.values(),
        &selected_masks,
        &budget,
    )
    .unwrap();
    let f0 = s0.begin(p0.values(), &selected).unwrap();
    let f1 = s1.begin(p1.values(), &selected).unwrap();
    drop((predicate_masks, p0, p1));
    let z0 = s0.finish(f1.as_bytes()).unwrap();
    let z1 = s1.finish(f0.as_bytes()).unwrap();
    assert_eq!(z0.values(), z1.values());
    let actual: Vec<i64> = z0
        .values()
        .iter()
        .zip(output_masks.values())
        .map(|(&z, &r)| i64::from((z.wrapping_sub(r) as u16) as i16))
        .collect();
    assert_eq!(actual, [0, 0, 0, 0, 1, 17, 32767]);
    drop((s0, s1, output_masks, z0, z1));
    assert!(
        budget.used_bytes() > 0,
        "exported frames retain their own reservations"
    );
    drop((f0, f1));
    assert_eq!(budget.used_bytes(), 0);
}
