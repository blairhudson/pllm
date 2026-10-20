//! D14 integration controls. All opening and intermediate inspection is test-only.
use super::*;

fn valid(a: &TokenShares, b: &TokenShares) -> u64 {
    a.validity_share()
        .unwrap()
        .wrapping_add(b.validity_share().unwrap())
        & MASK
}

#[test]
fn checked_profile_preserves_accepted_nodes_feedback_and_adds_fixed_work() {
    let (_, semantic, source) = dense_fixture();
    let ordinary = Program::compile(&semantic, source, 2, Limits::default()).unwrap();
    let (_, semantic, source) = dense_fixture();
    let checked = Program::compile(
        &semantic,
        source.with_checked_ranges(),
        2,
        Limits::default(),
    )
    .unwrap();
    assert!(!ordinary.checked_ranges());
    assert!(checked.checked_ranges());
    assert_eq!(ordinary.source_digest(), checked.source_digest());
    assert_ne!(ordinary.digest(), checked.digest());
    assert_eq!(
        checked.profile(),
        crate::fss_decoder::sdk::DENSE_ROPE_CHECKED_NUMERIC
    );
    let guard_rows = checked
        .nodes
        .iter()
        .filter(|n| matches!(n.code, Code::Norm { .. }))
        .map(|n| n.shape.iter().product::<usize>() / n.shape.last().unwrap())
        .sum::<usize>()
        * checked.selections;
    assert_eq!(
        checked.resources().kernels,
        ordinary.resources().kernels + 2 * guard_rows + 1
    );
    let (a, b) = execute_prefix(&ordinary, &[1]);
    assert_eq!((a.0.validity_share(), b.0.validity_share()), (None, None));
    let (c, d) = execute_prefix(&checked, &[1]);
    assert_eq!(valid(&c.0, &d.0), 1);
    assert_eq!(
        reconstructed(a.0.values(), b.0.values()),
        reconstructed(c.0.values(), d.0.values())
    );
    assert_eq!(a.1.len(), c.1.len());
    for ((a, b), (c, d)) in a.1.iter().zip(&b.1).zip(c.1.iter().zip(&d.1)) {
        assert_eq!((a.0, a.1), (c.0, c.1));
        assert_eq!(reconstructed(&a.2, &b.2), reconstructed(&c.2, &d.2));
    }
}

#[test]
fn checked_late_feedback_failure_masks_earlier_tokens_without_short_circuit() {
    let (_, semantic, original) = dense_fixture();
    let bound = crate::fss_decoder::energy::rms_guard(2).unwrap().hi;
    let mut tensors = Vec::new();
    for tensor in original.tensors {
        if tensor.id() == "model.embed_tokens.weight" {
            let mut values = tensor.values().unwrap().to_vec();
            // Padding token 0 stays safe. Initial token 1 selects this large tied
            // head row; private feedback reaches its rejected coordinate later.
            values[4] = 4096;
            values[5] = bound + 1;
            tensors.push(Tensor::from_q12(tensor.id(), tensor.shape(), &values).unwrap());
        } else {
            tensors.push(tensor);
        }
    }
    let source = Source::new(tensors)
        .unwrap()
        .with_rotary(original.rotary)
        .unwrap()
        .with_checked_ranges();
    let program = Program::compile(&semantic, source, 2, Limits::default()).unwrap();
    let (a, b) = execute_prefix(&program, &[1]);
    assert_eq!(valid(&a.0, &b.0), 0);
    assert_eq!(reconstructed(a.0.values(), b.0.values()), [0, 0]);
    assert_eq!(a.1.len(), program.nodes.len() * 2);
    let at = |step: usize, name: &str| {
        let index = semantic
            .prefill
            .operations
            .iter()
            .position(|op| op.id == name)
            .unwrap();
        let offset = step * program.nodes.len() + index;
        reconstructed(&a.1[offset].2, &b.1[offset].2)
    };
    // Earlier selection was genuinely nonzero, then suppressed only at final.
    let first_final = program.nodes.len() - 1;
    assert_eq!(reconstructed(&a.1[first_final].2, &b.1[first_final].2), [2]);
    let normalized = at(1, "layer.0.input_norm");
    assert_eq!(&normalized[2..], &[0, 0]);
    assert_ne!(&normalized[..2], &[0, 0]);
    // Same immutable program, full public schedule, and input-independent counts
    // also execute a successful request. Harness requires every dealer visit.
    let (c, d) = execute_prefix(&program, &[0]);
    assert_eq!(valid(&c.0, &d.0), 1);
    assert_eq!(c.1.len(), a.1.len());
}

#[test]
fn checked_rms_predicate_is_frozen_and_signed_sum_safe() {
    for width in [1usize, 2, 64, 896, 4096] {
        let bound = crate::fss_decoder::energy::rms_guard(width).unwrap();
        let b = bound.hi as u128;
        assert_eq!(bound.lo, -bound.hi);
        assert!(width as u128 * b * b + 1 < (1u128 << 49));
        assert!(width as u128 * (b + 1) * (b + 1) + 1 >= (1u128 << 49));
    }
    assert_eq!(
        crate::fss_decoder::energy::rms_guard(896).unwrap().hi,
        792648
    );
}
