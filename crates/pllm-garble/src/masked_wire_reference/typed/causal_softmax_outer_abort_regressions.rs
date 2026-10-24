//! Independent regression for the D36 parent -> inherited-child abort seam.
//!
//! SOURCE-ONLY definition: mount as a child test module of
//! `causal_softmax_region.rs` in the combined B-owner/I-ingress candidate.
//! It constructs the real original-score owner and runs the unchanged native
//! maximum/nExp exchanges. No fabricated Difference/E owner, raw-mask accessor,
//! clear-root nExp reconstruction or source/security authority is introduced.
//! Peer-byte copies are test mailboxes, not a request/physical-peak claim.
//! No execution is authorized merely by mounting this regression.

use super::*;
use crate::shared_allocation::{OwnedBudget, OwnedBudgetContext};
use pllm_core::edge_contract::{EdgeIdentity, Shape};

type C = OwnedBudgetContext;

fn require_frame(step: Step<C>) -> Frame<C> {
    match step {
        Step::Frame(frame) => frame,
        Step::Ready => panic!("outer output became ready before the child handoff checkpoint"),
    }
}

#[test]
fn outer_cancel_after_both_child_handoffs_revokes_retained_child_mask_and_frame() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let baseline = root.budget().used_bytes();
    let source = SourceIdentity::new([1; 32], [2; 32], approved_implementation());
    let mut dealer = OwnedLocalDealer::new(source, [4; 32], [92; 32], &root).unwrap();
    let input = root
        .charge_external(2 * size_of::<i64>())
        .unwrap()
        .try_build(|budget| {
            let mut scores = crate::shared_allocation::allocate(2, budget)?;
            scores.extend_from_slice(&[0i64, 1]);
            Ok(Zeroizing::new(scores))
        })
        .unwrap();
    let mut working = root
        .charge_external(bootstrap_bytes::<C>().unwrap())
        .unwrap();
    let edge = |wire| {
        EdgeDescriptor::new(
            EdgeIdentity::new(source, [4; 32], [5; 32], [wire; 32], 0),
            Shape::new(&[2]).unwrap(),
            Numeric::new(50, 12, Interpretation::Signed).unwrap(),
            Representation::CommonAdditiveMasked,
            None,
        )
        .unwrap()
    };
    let s = CausalSoftmaxSpec::checked(edge(10), edge(11), &[0, 1], [90; 32], [91; 32]).unwrap();
    let q = super::super::quote::resources(&dealer, &s, &input).unwrap();
    working.resize_payload(q.total_reserved_bytes).unwrap();
    let native = working
        .split_native(dealer.budget.budget(), q.total_reserved_bytes)
        .unwrap();
    let (mask, mut coordinator, [mut left, mut right]) = construct(
        &mut dealer,
        s,
        input,
        native,
        CausalSoftmaxWorkLimit(q.work.total),
    )
    .unwrap();
    let mut left_frame = require_frame(left.begin(&mut || false).unwrap());
    let mut right_frame = require_frame(right.begin(&mut || false).unwrap());
    let mut retained_child_frame = None;
    let mut handed_off = false;
    for _ in 0..s.exchanges() {
        let left_bytes = left_frame.bytes().unwrap().to_vec();
        let right_bytes = right_frame.bytes().unwrap().to_vec();
        let next_left = require_frame(left.advance(&right_bytes, &mut || false).unwrap());
        let next_right = require_frame(right.advance(&left_bytes, &mut || false).unwrap());
        if retained_child_frame.is_none() && matches!(&left_frame.native, Body::Child(_)) {
            retained_child_frame = Some(left_frame);
        }
        left_frame = next_left;
        right_frame = next_right;
        // Test only real parent transitions; never set private ready flags or
        // substitute a clear fixture for the genuine max-derived child input.
        handed_off = [&left, &right].iter().all(|role| {
            let state = role.slot.state.lock().unwrap();
            state.phase == Phase::Coordinate as u8
                && state.child.is_none()
                && state.exponent.is_some()
        });
        if handed_off {
            break;
        }
    }
    assert!(
        handed_off,
        "genuine roles never reached both child handoffs"
    );
    let retained = retained_child_frame.expect("no genuine child frame retained");
    let Body::Child(native_child_frame) = &retained.native else {
        panic!("retained frame is not from the original child");
    };
    let child_mask = mask.child.as_ref().unwrap();
    assert!(native_child_frame.bytes().is_ok());
    assert!(child_mask.descriptor().is_ok());

    coordinator.cancel();

    // Parent wrapper rejection alone is insufficient: old code passes this but
    // leaves both child output slots live under Keepalive's retained coordinator.
    assert!(matches!(retained.bytes(), Err(Error::Consumed)));
    assert!(
        matches!(child_mask.descriptor(), Err(Error::Consumed)),
        "outer abort did not synchronously abort the genuine child coordinator"
    );
    assert!(matches!(native_child_frame.bytes(), Err(Error::Consumed)));
    drop((
        retained,
        left_frame,
        right_frame,
        left,
        right,
        mask,
        coordinator,
        dealer,
        working,
    ));
    assert_eq!(root.budget().used_bytes(), baseline);
}
