use super::*;
use multiply::{
    tensor::{Activation, Program},
    Interval,
};

fn program(lanes: usize, case: usize) -> Program {
    match case {
        0 => Program::negative_compact(lanes, Interval::new(i64::MIN, i64::MAX).unwrap()),
        1 => Program::activation_guarded_compact(
            Activation::Silu,
            lanes,
            Interval::new(-256 << 18, 256 << 18).unwrap(),
        ),
        2 => Program::activation_guarded_compact(
            Activation::Gelu,
            lanes,
            Interval::new(-256 << 18, 256 << 18).unwrap(),
        ),
        3 => Program::reciprocal_compact(lanes, Interval::new(1, (1 << 36) - 1).unwrap()),
        _ => Program::rsqrt_compact(lanes, Interval::new(1, (1 << 35) - 1).unwrap()),
    }
    .unwrap()
}

#[test]
fn compact_nonlinear_encoded_allocation_sweeps_retire_without_leaks() {
    // Widest shrinking sign, mixed Boolean/arithmetic activation, and positive
    // prefix+PPA paths. Every fallible allocation up to a full successful run.
    for case in [0, 1, 3] {
        let program = program(3, case);
        let mut completed = false;
        for fail_at in 0..5000 {
            let mut owners = tensor_program_owners(&program, 1 << 18);
            let usages = owners.each_ref().map(|o| o.payload_usage());
            FAIL_AT.set(Some(fail_at));
            let result = tensor_run_wire(&mut owners, None, true);
            FAIL_AT.set(None);
            if result.is_err() {
                assert_eq!(result, Err(Error::Allocation));
            }
            // Exported-frame encode failure belongs to transport; retire every
            // peer, including owners that did not themselves observe the error.
            for owner in &mut owners {
                owner.cancel();
                assert!(owner.retired());
            }
            drop(owners);
            assert!(usages.iter().all(|u| u.live_bytes() == 0));
            if result.is_ok() {
                eprintln!("compact nonlinear case={case} allocation sites={fail_at}");
                completed = true;
                break;
            }
        }
        assert!(completed, "allocation sweep did not reach success: {case}");
    }
}

#[test]
fn compact_nonlinear_max_tile_erases_before_refund_in_all_roles() {
    for case in 0..5 {
        let program = program(128, case);
        for p in 0..3 {
            let mut owners = tensor_program_owners(&program, 1 << 18);
            let usages = owners.each_ref().map(|o| o.payload_usage());
            let watch = Watch::start(&usages[p], false, false);
            tensor_run_wire(&mut owners, Some(p), true).unwrap();
            let output = owners.map(|o| o.finish().unwrap());
            drop(output);
            assert_audit(watch.finish());
            assert!(usages.iter().all(|u| u.live_bytes() == 0));
            assert!(usages.iter().all(|u| u.peak_bytes() <= u.limit_bytes()));
        }
    }
}
