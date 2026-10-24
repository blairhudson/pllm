//! Mount under program, where genuine source-creation fixtures are permitted.
//! No fake context, expanded request entry or reconstructed token permit.
use super::*;
use crate::replicated_decoder::prepared_tensor_map::{self, PreparedTensorMap};
use request_origin::compact_storage::{EmissionContext, EmissionScope};

const TOKENS: [u32; 4] = [1, 2, 1, 2];

pub(super) fn with_context<T>(
    run: impl for<'a> FnOnce(&'a Builder, &'a EmissionContext<'a>) -> Result<T, Error>,
) -> Result<T, Error> {
    let (config, _, clear) = crate::replicated_decoder::tests::fixture_sources("qwen2", false);
    let mut publisher = Publisher::new_with_policy(
        &config,
        4,
        2,
        [1; 32],
        [2; 32],
        WeightMode::Public,
        ExecutionPolicy {
            range_policy: RangePolicy::Checked,
            max_abs_activation: 1,
            ..ExecutionPolicy::default()
        },
    )
    .unwrap();
    let shapes = publisher
        .shapes()
        .iter()
        .map(|(key, shape)| (key.clone(), shape.clone()))
        .collect::<Vec<_>>();
    let mut payloads = BTreeMap::new();
    for (key, shape) in shapes {
        let values = &clear[&key];
        let bytes = (0..shape.iter().product())
            .flat_map(|i| (values[i % values.len()] as f32 / ONE as f32).to_le_bytes())
            .collect::<Vec<_>>();
        publisher.append(&key, 0, &bytes).unwrap();
        payloads.insert(key, bytes);
    }
    publisher.seal(&TOKENS).unwrap();
    let mut manifest: Manifest = serde_json::from_slice(&publisher.manifest().unwrap()).unwrap();
    manifest.program = [0; 32];
    let mut importer =
        source_entry::SourceRequestImport::new(&serde_json::to_vec(&manifest).unwrap())?;
    for (key, bytes) in payloads {
        importer.append(&key, 0, &bytes)?;
    }
    importer.with_created(|created, guard| {
        let mut scope = EmissionScope::new(created, guard, MAX_PROGRAM_BYTES)?;
        let (builder, context, _) = scope.parts();
        run(builder, context)
    })
}

#[test]
fn new_full_tail_matches_original_factory_and_debits_all_invocations() {
    for lanes in [1, 128, 129, 256, 257] {
        with_context(|_, context| {
            let input = Interval::new(-1048576, 1048576)?;
            let before = context.work();
            let prepared = PreparedTensorMap::rescale_request(context, lanes, input, 18)?;
            prepared.validate()?;
            assert_eq!(context.work(), before.checked_add(prepared.work_budget()?)?);
            assert_eq!(prepared.invocations(), lanes.div_ceil(128) as u64);
            for offset in (0..lanes).step_by(128) {
                let p = prepared.program_at(offset)?;
                let mut b = tensor::Builder::new((lanes - offset).min(128), input)?;
                let out = b.rescale(b.input(), 18)?;
                let oracle = b.finish(out)?;
                assert_eq!(p.digest(), oracle.digest());
                assert_eq!(p.work_budget(), oracle.work_budget());
            }
            let charged = context.reservation().payload_bytes();
            let weak = Arc::downgrade(prepared.program_at(0)?);
            drop(prepared);
            assert!(weak.upgrade().is_none());
            assert_eq!(
                context.reservation().payload_bytes(),
                charged,
                "outer owner alone retires phases"
            );
            Ok(())
        })
        .unwrap();
    }
}

#[test]
fn reuse_permission_adds_invocation_work_without_native_factory() {
    with_context(|_, context| {
        let input = Interval::new(-1048576, 1048576)?;
        let original = PreparedTensorMap::rescale_request(context, 129, input, 18)?;
        let before = context.work();
        let attempts = prepared_tensor_map::native_attempts();
        PreparedTensorMap::admit_request_rescale_reuse(context, 129, input, 18)?;
        assert_eq!(context.work(), before.checked_add(original.work_budget()?)?);
        assert_eq!(prepared_tensor_map::native_attempts(), attempts);
        Ok(())
    })
    .unwrap();
}

#[test]
fn prior_real_work_is_not_reset_and_exhaustion_precedes_first_factory() {
    with_context(|_, context| {
        let input = Interval::new(-1048576, 1048576)?;
        let mut workspace = tensor::WorkPreflightWorkspace::empty();
        let ticket = tensor::Construction::Builder { lanes: 128 }
            .bind(tensor::FactoryParameters::Rescale { input, shift: 18 })
            .unwrap()
            .preflight(puma::MAX_PROGRAM_OPERATIONS, &mut workspace)
            .unwrap();
        // One original rescale instruction per invocation consumes the existing
        // operation ceiling. No native Program is needed to debit prior work.
        context.admit_work(ticket)?;
        let before = context.work();
        let attempts = prepared_tensor_map::native_attempts();
        assert!(matches!(
            PreparedTensorMap::rescale_request(context, 129, input, 18),
            Err(Error::Bounds)
        ));
        assert_eq!(context.work(), before);
        assert_eq!(prepared_tensor_map::native_attempts(), attempts);
        assert!(matches!(context.check_open(), Err(Error::Binding)));
        Ok(())
    })
    .unwrap();
}

#[test]
fn invalid_geometry_and_shift_burn_without_entering_native_factory() {
    for (lanes, shift) in [(0, 18), (MAX_VALUES + 1, 18), (129, 0), (129, 63)] {
        with_context(|_, context| {
            let attempts = prepared_tensor_map::native_attempts();
            assert!(matches!(
                PreparedTensorMap::rescale_request(context, lanes, Interval::new(-1, 1)?, shift),
                Err(Error::Bounds)
            ));
            assert_eq!(prepared_tensor_map::native_attempts(), attempts);
            assert!(matches!(context.check_open(), Err(Error::Binding)));
            Ok(())
        })
        .unwrap();
    }
}

#[test]
fn panic_after_full_erases_unpublished_owner_keeps_charge_and_burns() {
    with_context(|_, context| {
        prepared_tensor_map::observe_native_owners();
        prepared_tensor_map::request_rescale::panic_after_full();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            PreparedTensorMap::rescale_request(
                context,
                129,
                Interval::new(-1048576, 1048576).unwrap(),
                18,
            )
        }));
        assert!(result.is_err());
        let weak = prepared_tensor_map::take_native_owners();
        assert_eq!(weak.len(), 1);
        assert!(weak.iter().all(|p| p.upgrade().is_none()));
        assert!(
            context.work().operations() > 0,
            "both invocation debits persist"
        );
        assert!(context.reservation().payload_bytes() > 0, "no early refund");
        assert!(matches!(context.check_open(), Err(Error::Binding)));
        Ok(())
    })
    .unwrap();
}
