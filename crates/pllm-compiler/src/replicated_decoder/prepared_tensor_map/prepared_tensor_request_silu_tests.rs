//! Mount under program. Reuse real original source/context issuer, unchanged.
use super::request_rescale_tests::with_context;
use super::*;
use crate::replicated_decoder::prepared_tensor_map::{self, PreparedTensorMap};

#[test]
fn genuine_request_full_tail_matches_original_silu_geometry_and_work() {
    for lanes in [1, 128, 129, 256, 257] {
        with_context(|_, context| {
            let input = Interval::new(-4 * ONE, 4 * ONE)?;
            let before = context.work();
            let prepared = PreparedTensorMap::silu_request(context, lanes, input)?;
            assert_eq!(
                prepared.one_loop_scope(),
                Some(prepared_tensor_map::OneLoopScope::GuardedSilu)
            );
            assert_eq!(context.work(), before.checked_add(prepared.work_budget()?)?);
            assert_eq!(prepared.invocations(), lanes.div_ceil(128) as u64);
            for offset in (0..lanes).step_by(128) {
                let p = prepared.program_at(offset)?;
                let original = tensor::Program::activation_guarded(
                    tensor::Activation::Silu,
                    (lanes - offset).min(128),
                    input,
                )?;
                assert_eq!(p.digest(), original.digest());
                assert_eq!(p.work_budget(), original.work_budget());
                assert_eq!(p.output_bounds(), original.output_bounds());
                assert_eq!(p.output_lanes(), p.lanes());
                assert_eq!(p.validity_lanes(), None);
            }
            let charge = context.reservation().payload_bytes();
            let weak = Arc::downgrade(prepared.program_at(0)?);
            drop(prepared);
            assert!(weak.upgrade().is_none());
            assert_eq!(context.reservation().payload_bytes(), charge);
            Ok(())
        })
        .unwrap();
    }
}

#[test]
fn reuse_discards_build_permission_but_keeps_second_work_debit_without_factory() {
    with_context(|_, context| {
        let input = Interval::new(-4 * ONE, 4 * ONE)?;
        let original = PreparedTensorMap::silu_request(context, 257, input)?;
        let work = context.work();
        let attempts = prepared_tensor_map::native_attempts();
        PreparedTensorMap::admit_request_silu_reuse(context, 257, input)?;
        assert_eq!(context.work(), work.checked_add(original.work_budget()?)?);
        assert_eq!(prepared_tensor_map::native_attempts(), attempts);
        Ok(())
    })
    .unwrap();
}

#[test]
fn prior_request_one_short_refuses_before_first_native_factory() {
    with_context(|_, context| {
        let input = Interval::new(-4 * ONE, 4 * ONE)?;
        let mut workspace = tensor::WorkPreflightWorkspace::empty();
        let pending = tensor::Construction::GuardedSilu { lanes: 128 }
            .bind(tensor::FactoryParameters::GuardedSilu { input })
            .unwrap()
            .preflight_full_tail(
                1,
                tensor::Construction::GuardedSilu { lanes: 1 }
                    .bind(tensor::FactoryParameters::GuardedSilu { input })
                    .unwrap(),
                &mut workspace,
            )
            .unwrap()
            .summary()
            .work;
        let count = puma::MAX_PROGRAM_OPERATIONS
            .checked_sub(pending.operations())
            .unwrap()
            + 1;
        let prior = tensor::Construction::Builder { lanes: 1 }
            .bind(tensor::FactoryParameters::Rescale { input, shift: 18 })
            .unwrap()
            .preflight(count, &mut workspace)
            .unwrap();
        assert_eq!(
            prior.summary().work.operations() + pending.operations(),
            puma::MAX_PROGRAM_OPERATIONS + 1
        );
        context.admit_work(prior)?;
        let work = context.work();
        let attempts = prepared_tensor_map::native_attempts();
        assert!(matches!(
            PreparedTensorMap::silu_request(context, 129, input),
            Err(Error::Bounds)
        ));
        assert_eq!(context.work(), work);
        assert_eq!(prepared_tensor_map::native_attempts(), attempts);
        assert!(matches!(context.check_open(), Err(Error::Binding)));
        Ok(())
    })
    .unwrap();
}

#[test]
fn invalid_declaration_burns_without_factory() {
    for (lanes, input) in [
        (0, Interval::new(-1, 1).unwrap()),
        (MAX_VALUES + 1, Interval::new(-1, 1).unwrap()),
        (1, Interval::new(i64::MIN, i64::MAX).unwrap()),
    ] {
        with_context(|_, context| {
            let attempts = prepared_tensor_map::native_attempts();
            assert!(PreparedTensorMap::silu_request(context, lanes, input).is_err());
            assert_eq!(prepared_tensor_map::native_attempts(), attempts);
            assert!(matches!(context.check_open(), Err(Error::Binding)));
            Ok(())
        })
        .unwrap();
    }
}

#[test]
fn panic_after_full_destroys_unpublished_owner_but_retains_whole_schedule_work_and_charge() {
    with_context(|_, context| {
        let input = Interval::new(-4 * ONE, 4 * ONE)?;
        let mut workspace = tensor::ActivationWorkWorkspace::empty();
        let full = tensor::Program::activation_guarded_work(
            tensor::Activation::Silu,
            128,
            input,
            &mut workspace,
        )?;
        let tail = tensor::Program::activation_guarded_work(
            tensor::Activation::Silu,
            1,
            input,
            &mut workspace,
        )?;
        let expected = context.work().checked_add(full.checked_add(tail)?)?;
        prepared_tensor_map::observe_native_owners();
        prepared_tensor_map::request_silu::panic_after_full();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            PreparedTensorMap::silu_request(context, 129, Interval::new(-4 * ONE, 4 * ONE).unwrap())
        }));
        assert!(result.is_err());
        let weak = prepared_tensor_map::take_native_owners();
        assert_eq!(weak.len(), 1);
        assert!(weak.iter().all(|p| p.upgrade().is_none()));
        assert_eq!(
            context.work(),
            expected,
            "tail work spent before full factory"
        );
        assert!(context.reservation().payload_bytes() > 0);
        assert!(matches!(context.check_open(), Err(Error::Binding)));
        Ok(())
    })
    .unwrap();
}
