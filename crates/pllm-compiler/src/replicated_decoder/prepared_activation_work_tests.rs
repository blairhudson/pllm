use super::*;

const ONE: i64 = 1 << 18;

fn expected(lanes: usize, input: Interval) -> WorkBudget {
    let width = lanes.min(tensor::MAX_TILE_LANES);
    let mut space = tensor::ActivationWorkWorkspace::empty();
    let full = tensor::Program::activation_guarded_work(
        tensor::Activation::Silu,
        width,
        input,
        &mut space,
    )
    .unwrap();
    let work = full.checked_mul((lanes / width) as u64).unwrap();
    if lanes % width == 0 {
        work
    } else {
        work.checked_add(
            tensor::Program::activation_guarded_work(
                tensor::Activation::Silu,
                lanes % width,
                input,
                &mut space,
            )
            .unwrap(),
        )
        .unwrap()
    }
}

#[test]
fn prepared_guarded_work_uses_actual_full_invocations_and_tail_before_original_factories() {
    let input = Interval::new(-ONE, ONE).unwrap();
    for lanes in [1, 3, 127, 128, 129, 256, 257] {
        let work = expected(lanes, input);
        let before = native_attempts();
        let prior = WorkBudget::local().checked_mul(13).unwrap();
        let prepared =
            PreparedTensorMap::guarded_silu_with_prior_work(lanes, input, 0, prior, |_| Ok(()))
                .unwrap();
        assert_eq!(prepared.work_budget().unwrap(), work);
        assert_eq!(
            native_attempts() - before,
            1 + usize::from(lanes > 128 && lanes % 128 != 0)
        );
        assert_eq!(prior.operations(), 13);
        for program in prepared.programs() {
            let native = tensor::Program::activation_guarded(
                tensor::Activation::Silu,
                program.lanes(),
                input,
            )
            .unwrap();
            assert_eq!(program.digest(), native.digest());
            assert_eq!(program.work_budget(), native.work_budget());
            assert_eq!(program.output_bounds(), native.output_bounds());
        }
    }
}

#[test]
fn prepared_guarded_global_exact_and_one_short_ops_refuse_before_factory_or_owner_entry() {
    let input = Interval::new(-ONE, ONE).unwrap();
    let lanes = 257;
    let work = expected(lanes, input);
    let ceiling = puma::MAX_PROGRAM_OPERATIONS;
    let prior = WorkBudget::local()
        .checked_mul(ceiling - work.operations())
        .unwrap();
    let short = prior.checked_add(WorkBudget::local()).unwrap();
    let before = native_attempts();
    observe_native_owners();
    assert!(matches!(
        PreparedTensorMap::guarded_silu_with_prior_work(lanes, input, 0, short, |_| Ok(())),
        Err(Error::Bounds)
    ));
    assert_eq!(native_attempts(), before);
    assert!(take_native_owners().is_empty());
    let prepared =
        PreparedTensorMap::guarded_silu_with_prior_work(lanes, input, 0, prior, |_| Ok(()))
            .unwrap();
    assert_eq!(native_attempts(), before + 2);
    assert_eq!(
        prior
            .checked_add(prepared.work_budget().unwrap())
            .unwrap()
            .operations(),
        ceiling
    );
    assert_eq!(prior.operations(), ceiling - work.operations());
    assert!(matches!(
        PreparedTensorMap::guarded_silu_with_prior_work(
            lanes,
            input,
            0,
            WorkBudget::local().checked_mul(u64::MAX).unwrap(),
            |_| Ok(())
        ),
        Err(Error::Bounds)
    ));
    assert_eq!(native_attempts(), before + 2);
}

#[test]
fn prepared_activation_workspace_peak_exact_short_and_callback_error_winners() {
    let input = Interval::new(-ONE, ONE).unwrap();
    let baseline = 73;
    let mut peak = 0;
    let mut calls = 0;
    let probe = PreparedTensorMap::guarded_silu(129, input, baseline, |bytes| {
        peak = peak.max(bytes);
        calls += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(calls, 3);
    assert!(peak >= baseline + tensor::ActivationWorkWorkspace::workspace_bytes());
    drop(probe);
    // Use an identical callback shape/capture layout for exact/one-short. The
    // callback's two borrowed usize pointees are caller baseline, not native.
    fn bounded(limit: usize) -> (Result<PreparedTensorMap, Error>, usize) {
        let mut peak = 0;
        let mut calls = 0;
        let result =
            PreparedTensorMap::guarded_silu(129, Interval::new(-ONE, ONE).unwrap(), 73, |bytes| {
                peak = peak.max(bytes);
                calls += 1;
                if bytes > limit {
                    Err(Error::Bounds)
                } else {
                    Ok(())
                }
            });
        if result.is_ok() {
            assert_eq!(calls, 3);
        } else {
            assert!(calls > 0);
        }
        (result, peak)
    }
    // The bounded callback also captures limit; derive its own exact envelope,
    // rather than crediting a differently-sized callback from the probe above.
    let (prepared, exact) = bounded(usize::MAX);
    drop(prepared.unwrap());
    let before = native_attempts();
    assert!(matches!(bounded(exact - 1).0, Err(Error::Bounds)));
    assert_eq!(native_attempts(), before);
    drop(bounded(exact).0.unwrap());
    assert_eq!(native_attempts(), before + 2);
    let extreme = Interval::new(i64::MIN, i64::MAX).unwrap();
    let before = native_attempts();
    assert!(matches!(
        PreparedTensorMap::guarded_silu(129, extreme, 0, |_| Err(Error::Allocation)),
        Err(Error::Allocation)
    ));
    assert!(matches!(
        PreparedTensorMap::guarded_silu(129, extreme, 0, |_| Ok(())),
        Err(Error::Bounds)
    ));
    let unwind = std::panic::catch_unwind(|| {
        PreparedTensorMap::guarded_silu(129, extreme, 0, |_| {
            panic!("test-only first admission callback unwind")
        })
    });
    assert!(unwind.is_err());
    assert_eq!(native_attempts(), before);
}

#[test]
fn prepared_activation_bad_geometry_baseline_overflow_and_domain_never_enter_factory() {
    let input = Interval::new(-ONE, ONE).unwrap();
    let before = native_attempts();
    let mut calls = 0;
    for lanes in [0, MAX_VALUES + 1, usize::MAX] {
        assert!(matches!(
            PreparedTensorMap::guarded_silu(lanes, input, 0, |_| {
                calls += 1;
                Ok(())
            }),
            Err(Error::Bounds)
        ));
    }
    assert!(matches!(
        PreparedTensorMap::guarded_silu(129, input, usize::MAX, |_| {
            calls += 1;
            Ok(())
        }),
        Err(Error::Bounds)
    ));
    assert_eq!(calls, 0);
    for input in [
        Interval::new(i64::MIN, i64::MIN).unwrap(),
        Interval::new(i64::MAX, i64::MAX).unwrap(),
    ] {
        assert!(matches!(
            PreparedTensorMap::guarded_silu(129, input, 0, |_| Ok(())),
            Err(Error::Bounds)
        ));
    }
    assert_eq!(native_attempts(), before);
}
