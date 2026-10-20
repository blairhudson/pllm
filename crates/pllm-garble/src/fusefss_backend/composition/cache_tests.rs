use super::*;
use crate::fusefss_backend::composition::{gates, norm, Event};
use crate::shared_rescale_reference::Context;

fn cache(budget: &Budget) -> SourceTableCache<'_> {
    SourceTableCache::new(
        [7; 32],
        Limits {
            max_entries: 8,
            max_retained_bytes: 1024 * 1024,
        },
        budget,
    )
    .unwrap()
}
fn range() -> gates::InputRange {
    gates::InputRange {
        min: -4096,
        max: 4096,
    }
}

#[test]
fn cached_norm_silu_softmax_preserve_legacy_digest_and_private_costs() {
    let budget = Budget::new(super::super::super::MAX_ALLOCATION_BYTES).unwrap();
    let cache = cache(&budget);
    let gamma = [4096, 4096];
    let beta = [0, 0];
    let spec = norm::Spec {
        kind: norm::Kind::Layer,
        input_min: -4096,
        input_max: 4096,
        gamma: &gamma,
        beta: &beta,
    };
    let legacy = norm::compile([3; 32], &spec, &budget).unwrap();
    let a = norm::compile_cached([3; 32], &spec, &cache).unwrap();
    let b = norm::compile_cached([3; 32], &spec, &cache).unwrap();
    assert_eq!(legacy.digest(), a.digest());
    assert_eq!(a.digest(), b.digest());
    for p in [&a, &b] {
        let x = legacy.resources();
        let y = p.resources();
        assert_eq!(x.cumulative_work, y.cumulative_work);
        assert_eq!(
            x.private_material_bytes_total,
            y.private_material_bytes_total
        );
        assert_eq!(
            x.public_table_copy_bytes_total,
            y.public_table_copy_bytes_total
        );
        assert_eq!(x.frames_total, y.frames_total);
    }
    drop((legacy, a, b));
    let (legacy, _) = gates::silu(range(), &budget).unwrap();
    let (a, _) = gates::silu_cached(range(), &cache).unwrap();
    let (b, _) = gates::silu_cached(range(), &cache).unwrap();
    assert_eq!(legacy.digest(), a.digest());
    assert_eq!(a.digest(), b.digest());
    assert_eq!(
        legacy.resources().cumulative_work,
        a.resources().cumulative_work
    );
    drop((legacy, a, b));
    let ranges = [range(); 2];
    let active = [true, true];
    let (legacy, _) = gates::softmax(&ranges, &active, &budget).unwrap();
    let (a, _) = gates::softmax_cached(&ranges, &active, &cache).unwrap();
    let (b, _) = gates::softmax_cached(&ranges, &active, &cache).unwrap();
    assert_eq!(legacy.digest(), a.digest());
    assert_eq!(a.digest(), b.digest());
    assert_eq!(
        legacy.resources().cumulative_work,
        a.resources().cumulative_work
    );
    assert_eq!(
        legacy.resources().public_table_copy_bytes_total,
        a.resources().public_table_copy_bytes_total
    );
    drop((legacy, a, b));
    let stats = cache.stats();
    assert_eq!(stats.misses, 5);
    assert_eq!(stats.hits, 5);
    assert_eq!(stats.references, 10);
    assert_eq!(stats.table_constructions, 5);
    assert_eq!(stats.table_hashes, 5);
    assert_eq!(stats.generated_words, 8192 + 256);
    assert_eq!(stats.retained_entries, 5);
    drop(cache);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn hit_uses_existing_owner_without_generation_hash_copy_or_allocation() {
    let budget = Budget::new(super::super::super::MAX_ALLOCATION_BYTES).unwrap();
    let cache = cache(&budget);
    let id = cache.identity(Kind::Rsqrt { k: 2 }, [1; 32]).unwrap();
    let before = cache.stats();
    let used = budget.used_bytes();
    budget
        .fail_after
        .store(0, std::sync::atomic::Ordering::SeqCst);
    assert_eq!(cache.identity(Kind::Rsqrt { k: 2 }, [1; 32]).unwrap(), id);
    cache.validate(id).unwrap();
    let after = cache.stats();
    assert_eq!(after.hits, before.hits + 1);
    assert_eq!(after.generated_words, before.generated_words);
    assert_eq!(after.table_hashes, before.table_hashes);
    assert_eq!(
        after.table_payload_copy_bytes,
        before.table_payload_copy_bytes
    );
    assert_eq!(budget.used_bytes(), used);
    for wrong in [
        Identity {
            namespace: [2; 32],
            ..id
        },
        Identity {
            numeric_revision: [2; 32],
            ..id
        },
        Identity {
            source_digest: [2; 32],
            ..id
        },
        Identity {
            kind: Kind::Rsqrt { k: 3 },
            ..id
        },
        Identity { bits: 49, ..id },
        Identity { entries: 1, ..id },
        Identity {
            table_digest: [2; 32],
            ..id
        },
    ] {
        assert!(cache.validate(wrong).is_err());
    }
    drop(cache);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn cache_plan_runtime_and_exported_packet_retire_independently() {
    let budget = Budget::new(super::super::super::MAX_ALLOCATION_BYTES).unwrap();
    let cache = cache(&budget);
    let gamma = [4096, 4096];
    let beta = [0, 0];
    let spec = norm::Spec {
        kind: norm::Kind::Rms,
        input_min: -4096,
        input_max: 4096,
        gamma: &gamma,
        beta: &beta,
    };
    let plan = norm::compile_cached([3; 32], &spec, &cache)
        .unwrap()
        .into_shared()
        .unwrap();
    let table_bytes = cache.stats().retained_bytes;
    let context = Context {
        plan: plan.digest(),
        session: [4; 32],
        operation: [5; 32],
        first_tensor_index: 0,
    };
    let mut runtime = plan.runtime_owned(0, context, &[0, 0]).unwrap();
    let pc = match runtime.poll().unwrap() {
        Event::NeedMaterial { pc } => pc,
        _ => panic!("expected secret operation"),
    };
    let [a, b] = plan.issue_step(pc, context).unwrap();
    drop(b);
    let packet = a.export_packet().unwrap();
    drop(cache);
    assert!(budget.used_bytes() >= table_bytes);
    drop(plan);
    assert!(budget.used_bytes() >= table_bytes);
    runtime.cancel();
    drop(runtime);
    assert!(budget.used_bytes() > 0);
    drop(packet);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn miss_allocation_failure_and_capacity_exhaustion_release_every_lease() {
    let budget = Budget::new(super::super::super::MAX_ALLOCATION_BYTES).unwrap();
    let cache = cache(&budget);
    let before = budget.used_bytes();
    budget
        .fail_after
        .store(0, std::sync::atomic::Ordering::SeqCst);
    assert!(matches!(
        cache.identity(Kind::Rsqrt { k: 2 }, [1; 32]),
        Err(Error::Allocation)
    ));
    assert_eq!(budget.used_bytes(), before);
    assert_eq!(cache.stats().retained_entries, 0);
    budget
        .fail_after
        .store(-1, std::sync::atomic::Ordering::SeqCst);
    drop(cache);
    assert_eq!(budget.used_bytes(), 0);
    let cache = SourceTableCache::new(
        [7; 32],
        Limits {
            max_entries: 1,
            max_retained_bytes: 1024 * 1024,
        },
        &budget,
    )
    .unwrap();
    cache.identity(Kind::Rsqrt { k: 2 }, [1; 32]).unwrap();
    let before = budget.used_bytes();
    assert!(matches!(
        cache.identity(Kind::Rsqrt { k: 3 }, [1; 32]),
        Err(Error::Resource)
    ));
    assert_eq!(budget.used_bytes(), before);
    assert_eq!(cache.stats().retained_entries, 1);
    drop(cache);
    assert_eq!(budget.used_bytes(), 0);
}
