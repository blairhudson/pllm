use super::*;
use multiply::{tensor::Program, Interval};

fn row_program(case: usize, width: usize) -> Program {
    let bounds = Interval::new(-1 << 18, 1 << 18).unwrap();
    match case {
        0 => {
            Program::rms_norm_with_input_norm_compact(
                1,
                width,
                bounds,
                &vec![1 << 18; width],
                width as u128 * (1_u128 << 36),
            )
            .unwrap()
            .0
        }
        1 => {
            Program::layer_norm_certified_compact(
                1,
                width,
                bounds,
                &vec![1 << 18; width],
                &vec![0; width],
            )
            .unwrap()
            .0
        }
        _ => {
            Program::softmax_certified_compact(1, width, bounds)
                .unwrap()
                .0
        }
    }
}

#[test]
fn compact_rows_sampled_allocation_failures_cancel_and_preserve_ledger() {
    // New wrappers add no runtime primitive. Cover early/late encode, decode,
    // arithmetic and row-operation allocations; primitive exhaustive sweeps
    // remain in compact_nonlinear. Include an uninterrupted terminal run.
    for case in 0..3 {
        let program = row_program(case, 3);
        for fail in [
            Some(0),
            Some(1),
            Some(2),
            Some(3),
            Some(4),
            Some(7),
            Some(15),
            Some(31),
            Some(63),
            Some(127),
            Some(255),
            Some(511),
            Some(1023),
            None,
        ] {
            let mut owners = tensor_program_owners(&program, 0);
            let usages = owners.each_ref().map(|o| o.payload_usage());
            FAIL_AT.set(fail);
            let result = tensor_run_wire(&mut owners, None, true);
            FAIL_AT.set(None);
            if result.is_err() {
                assert_eq!(result, Err(Error::Allocation));
            }
            if fail.is_none() {
                assert!(result.is_ok());
            }
            for owner in &mut owners {
                owner.cancel();
                assert!(owner.retired());
            }
            drop(owners);
            assert!(usages.iter().all(|u| u.live_bytes() == 0));
        }
    }
}

#[test]
fn compact_rows_full_row_erasure_before_refund_in_all_roles() {
    for (case, width) in [(0, 896), (1, 896), (2, 512)] {
        let program = row_program(case, width);
        assert!(program.admitted_bytes() <= MAX_ADMITTED_BYTES);
        for p in 0..3 {
            let mut owners = tensor_program_owners(&program, 0);
            let usages = owners.each_ref().map(|o| o.payload_usage());
            let watch = Watch::start(&usages[p], false, false);
            tensor_run_wire(&mut owners, Some(p), true).unwrap();
            let outputs = owners.map(|o| o.finish().unwrap());
            drop(outputs);
            assert_audit(watch.finish());
            assert!(usages.iter().all(|u| u.live_bytes() == 0));
            assert!(usages.iter().all(|u| u.peak_bytes() <= u.limit_bytes()));
        }
    }
}

#[test]
fn compact_rows_retained_encoded_frame_survives_owner_cancellation_and_erases() {
    for case in 0..3 {
        let program = row_program(case, 128);
        for p in 0..3 {
            let mut owners = tensor_program_owners(&program, 0);
            let usage = owners[p].payload_usage();
            // Centering/max start with local row maps. Reach the first
            // collective message stage, retaining its exported frame.
            let frame = loop {
                let mut frames = owners.each_mut().map(|o| o.start_step().unwrap());
                if let Some(at) = frames[p].iter().position(Option::is_some) {
                    break frames[p][at].take().unwrap();
                }
                assert!(frames.iter().flatten().all(Option::is_none));
                for owner in &mut owners {
                    owner.finish_step().unwrap();
                }
            };
            let watch = Watch::start(&usage, true, false);
            let encoded = frame.encode().unwrap();
            for owner in &mut owners {
                owner.cancel();
            }
            drop(owners);
            assert!(usage.live_bytes() > 0);
            drop(encoded);
            assert_eq!(usage.live_bytes(), 0);
            let audit = watch.finish();
            assert!(audit.allocations > 0);
            assert_audit(audit);
        }
    }
}
