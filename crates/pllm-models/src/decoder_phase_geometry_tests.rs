use super::DecoderPlan;

#[test]
fn phase_geometry_rejects_zero_overflow_and_out_of_range_phases() {
    for (input, outputs) in [(0, 0), (0, 2), (2, 0), (usize::MAX, 2), (2, usize::MAX)] {
        assert!(DecoderPlan::validated_phase_geometry(input, outputs).is_none());
    }
    for (input, outputs) in [(1, 1), (usize::MAX, 1), (1, usize::MAX)] {
        let geometry = DecoderPlan::validated_phase_geometry(input, outputs).unwrap();
        assert_eq!(geometry.input_tokens(), input);
        assert_eq!(geometry.outputs(), outputs);
        assert_eq!(geometry.active_rows(0), Some(input));
        assert_eq!(geometry.active_rows(outputs - 1), Some(geometry.padded_rows()));
        assert_eq!(geometry.active_rows(outputs), None);
        assert_eq!(geometry.active_rows(usize::MAX), None);
    }
}

#[test]
fn phase_geometry_enumerates_d64_activity_separately_from_padded_work() {
    // Explicit independent phase/total oracles: no large graphs or tensors.
    let cases: &[(usize, usize, usize, &[usize], usize, usize)] = &[
        (30, 2, 31, &[30, 31], 62, 61),
        (32, 2, 33, &[32, 33], 66, 65),
        (64, 8, 71, &[64, 65, 66, 67, 68, 69, 70, 71], 568, 540),
        (2, 3, 4, &[2, 3, 4], 12, 9),
    ];
    for &(input, outputs, padded, active, padded_total, logical_total) in cases {
        let geometry = DecoderPlan::validated_phase_geometry(input, outputs).unwrap();
        assert_eq!(geometry.padded_rows(), padded);
        for (phase, &expected) in active.iter().enumerate() {
            assert_eq!(geometry.active_rows(phase), Some(expected));
        }
        assert_eq!(geometry.active_rows(outputs), None);
        assert_eq!(geometry.active_rows(usize::MAX), None);
        assert_eq!(geometry.padded_rows().checked_mul(geometry.outputs()), Some(padded_total));
        assert_eq!(active.iter().sum::<usize>(), logical_total);
        assert_ne!(padded_total, logical_total);
    }
}
