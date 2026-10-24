//! Ordinary external native caller; not compiler/SDK registration or decoding.
use pllm_core::edge_contract::{
    EdgeDescriptor, EdgeIdentity, Interpretation, Numeric, Representation, Shape, SourceIdentity,
};
use pllm_garble::masked_wire_reference::typed::{
    approved_implementation, causal_softmax_bootstrap_bytes, causal_softmax_from_reserved,
    causal_softmax_resources, CausalSoftmaxSpec, CausalSoftmaxStepFor, CausalSoftmaxWorkLimit,
    OwnedLocalDealer,
};
use pllm_garble::shared_allocation::{OwnedBudget, MAX_ALLOCATION_BYTES};
use std::mem::size_of;
use zeroize::Zeroizing;

#[test]
fn checked_original_clear_root_runs_all_private_frames_from_external_crate() {
    for k in [1, 2, 3] {
        let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let baseline = root.budget().used_bytes();
        let source = SourceIdentity::new([1; 32], [2; 32], approved_implementation());
        let mut dealer = OwnedLocalDealer::new(source, [4; 32], [8; 32], &root).unwrap();
        let edge = |wire, representation| {
            EdgeDescriptor::new(
                EdgeIdentity::new(source, [4; 32], [5; 32], [wire; 32], 0),
                Shape::new(&[3]).unwrap(),
                Numeric::new(50, 12, Interpretation::Signed).unwrap(),
                representation,
                None,
            )
            .unwrap()
        };
        let active: &[u8] = match k {
            1 => &[2],
            2 => &[0, 2],
            _ => &[0, 1, 2],
        };
        let spec = CausalSoftmaxSpec::checked(
            edge(6, Representation::CommonAdditiveMasked),
            edge(
                7,
                if k == 1 {
                    Representation::Public
                } else {
                    Representation::CommonAdditiveMasked
                },
            ),
            active,
            [9; 32],
            [10; 32],
        )
        .unwrap();
        let input = root
            .charge_external(3 * size_of::<i64>())
            .unwrap()
            .try_build(|_| Ok(Zeroizing::new(vec![0i64, 1, 2])))
            .unwrap();
        let mut working = root
            .charge_external(causal_softmax_bootstrap_bytes().unwrap())
            .unwrap();
        let q = causal_softmax_resources(&dealer, &spec, &input, &working).unwrap();
        working.resize_payload(q.total_reserved_bytes).unwrap();
        let (mask, coordinator, [mut left, mut right]) = causal_softmax_from_reserved(
            &mut dealer,
            spec,
            input,
            working,
            CausalSoftmaxWorkLimit(q.work.total),
        )
        .unwrap();
        let mut a = left.begin(&mut || false).unwrap();
        let mut b = right.begin(&mut || false).unwrap();
        let mut retained = Vec::new();
        for _ in 0..spec.exchanges() {
            let (CausalSoftmaxStepFor::Frame(af), CausalSoftmaxStepFor::Frame(bf)) = (a, b) else {
                panic!("early Ready")
            };
            let ab = af.bytes().unwrap().to_vec();
            let bb = bf.bytes().unwrap().to_vec();
            retained.extend([af, bf]);
            a = left.advance(&bb, &mut || false).unwrap();
            b = right.advance(&ab, &mut || false).unwrap();
        }
        assert!(matches!(a, CausalSoftmaxStepFor::Ready));
        assert!(matches!(b, CausalSoftmaxStepFor::Ready));
        assert_eq!(retained.len(), 2 * spec.exchanges());
        assert_eq!(
            retained
                .iter()
                .map(|f| f.bytes().unwrap().len())
                .sum::<usize>(),
            q.peer_bytes
        );
        let output0 = left.finish().unwrap();
        let output1 = right.finish().unwrap();
        assert_eq!(output0.role(), 0);
        assert_eq!(output1.role(), 1);
        assert_eq!(output0.descriptor().unwrap(), spec.output());
        drop((output0, output1, retained, mask, coordinator, dealer));
        assert_eq!(root.budget().used_bytes(), baseline);
    }
}

#[test]
fn external_entry_rejects_foreign_working_counter_without_keys_or_outputs() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let foreign = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let source = SourceIdentity::new([1; 32], [2; 32], approved_implementation());
    let mut dealer = OwnedLocalDealer::new(source, [4; 32], [8; 32], &root).unwrap();
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
    let spec = CausalSoftmaxSpec::checked(edge(6), edge(7), &[0, 1], [9; 32], [10; 32]).unwrap();
    let input = root
        .charge_external(2 * size_of::<i64>())
        .unwrap()
        .try_build(|_| Ok(Zeroizing::new(vec![0i64, 1])))
        .unwrap();
    let working = foreign
        .charge_external(causal_softmax_bootstrap_bytes().unwrap())
        .unwrap();
    assert!(causal_softmax_resources(&dealer, &spec, &input, &working).is_err());
    assert!(causal_softmax_from_reserved(
        &mut dealer,
        spec,
        input,
        working,
        CausalSoftmaxWorkLimit(u128::MAX)
    )
    .is_err());
}
