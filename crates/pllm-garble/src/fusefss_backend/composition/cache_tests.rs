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
fn lowered_cache_first_retirement_keeps_last_party_backing_charged() {
    let budget = Budget::new(super::super::super::MAX_ALLOCATION_BYTES).unwrap();
    let cache = cache(&budget);
    let owner = cache.get(Kind::Silu, [3; 32]).unwrap();
    let stats = cache.stats();
    assert_eq!(stats.lowered_chunk_constructions, 4);
    assert_eq!(stats.lowered_payload_bytes, (257 + 3 * 258) * 24 + 4 * 8);
    assert_eq!(
        stats.table_payload_copy_bytes,
        1024 * 8 + stats.lowered_payload_bytes
    );
    let context = Context {
        plan: [1; 32],
        session: [2; 32],
        operation: [3; 32],
        first_tensor_index: 0,
    };
    let r = owner.table.resources(2).unwrap();
    let [mut a, mut b] = owner.table.issue(2, context).unwrap();
    drop(owner);
    drop(cache);
    assert_eq!(
        budget.used_bytes(),
        stats.lowered_retained_bytes + 2 * r.party_reservation_bytes
    );
    assert!(matches!(a.start(&[1u64 << 50, 0]), Err(Error::Domain)));
    assert!(a.is_consumed());
    assert!(matches!(a.start(&[0, 0]), Err(Error::Consumed)));
    assert_eq!(
        budget.used_bytes(),
        stats.lowered_retained_bytes + 2 * r.party_reservation_bytes
    );
    b.cancel();
    assert_eq!(budget.used_bytes(), 2 * r.party_reservation_bytes);
    drop((a, b));
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn lowered_cache_miss_failure_sweep_never_publishes_partial_chunks() {
    use crate::shared_affine_lookup_reference::RingTable;
    use std::sync::atomic::Ordering;
    let budget = Budget::new(super::super::super::MAX_ALLOCATION_BYTES).unwrap();
    let cache = cache(&budget);
    let before = budget.used_bytes();
    let mut completed = false;
    for fail in 0..64 {
        budget.fail_after.store(fail, Ordering::SeqCst);
        let result = cache.identity(Kind::Silu, [3; 32]);
        if result.is_ok() {
            assert_eq!(fail, 16);
            completed = true;
            break;
        }
        assert!(matches!(result, Err(Error::Allocation)));
        assert_eq!(cache.stats().retained_entries, 0);
        assert_eq!(cache.stats().lowered_retained_bytes, 0);
        assert_eq!(budget.used_bytes(), before, "cache lowering failure {fail}");
    }
    assert!(completed);
    let stats = cache.stats();
    assert_eq!(stats.table_constructions, 15);
    assert_eq!(stats.table_hashes, 15);
    assert_eq!(stats.lowered_chunk_constructions, 30);
    assert_eq!(
        stats.lowered_payload_bytes,
        (12 * 257 + 18 * 258) * 24 + 30 * 8
    );
    assert_eq!(
        stats.table_payload_copy_bytes,
        15 * 1024 * 8 + stats.lowered_payload_bytes
    );
    assert_eq!(
        stats.lowered_construction_metadata_bytes,
        30 * size_of::<RingTable>()
            + 26 * (2 * size_of::<usize>() + size_of::<Arc<RingTable>>())
            + size_of::<Vec<Arc<RingTable>>>()
            + size_of::<Lease<'_>>()
            + 3 * size_of::<usize>()
    );
    assert_eq!(budget.used_bytes(), before + cache.stats().retained_bytes);
    budget.fail_after.store(0, Ordering::SeqCst);
    cache.identity(Kind::Silu, [3; 32]).unwrap();
    assert_eq!(
        cache.stats().lowered_chunk_constructions,
        stats.lowered_chunk_constructions
    );
    drop(cache);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn reciprocal_512_chunks_charge_explicit_construction_metadata_once() {
    use crate::shared_affine_lookup_reference::RingTable;
    let budget = Budget::new(super::super::super::MAX_ALLOCATION_BYTES).unwrap();
    let cache = SourceTableCache::new(
        [7; 32],
        Limits {
            max_entries: 1,
            max_retained_bytes: MAX_RETAINED_BYTES,
        },
        &budget,
    )
    .unwrap();
    let id = cache
        .identity(Kind::Reciprocal { k: 1024 }, [3; 32])
        .unwrap();
    let stats = cache.stats();
    assert_eq!(stats.lowered_chunk_constructions, 512);
    let metadata = 512
        * (size_of::<RingTable>() + 2 * size_of::<usize>() + size_of::<Arc<RingTable>>())
        + size_of::<Vec<Arc<RingTable>>>()
        + size_of::<Lease<'_>>()
        + 3 * size_of::<usize>();
    assert_eq!(stats.lowered_construction_metadata_bytes, metadata);
    assert_eq!(
        metadata,
        stats.lowered_retained_bytes - stats.lowered_payload_bytes + size_of::<usize>()
    );
    assert!(metadata > 512 * size_of::<Arc<RingTable>>());
    let live = budget.used_bytes();
    budget
        .fail_after
        .store(0, std::sync::atomic::Ordering::SeqCst);
    assert_eq!(
        cache
            .identity(Kind::Reciprocal { k: 1024 }, [3; 32])
            .unwrap(),
        id
    );
    assert_eq!(cache.stats().lowered_construction_metadata_bytes, metadata);
    assert_eq!(budget.used_bytes(), live);
    drop(cache);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn tensor_counts_fresh_lane_material_and_handles_without_copying_lowered_backing() {
    let budget = Budget::new(super::super::super::MAX_ALLOCATION_BYTES).unwrap();
    let cache = cache(&budget);
    let (plan, _) = gates::silu_cached(range(), &cache).unwrap();
    let scalar = plan.resources();
    let before = cache.stats();
    assert_eq!(before.lowered_chunk_constructions, 4);
    assert_eq!(
        scalar.public_table_copy_bytes_total,
        2 * (4 + 1) * size_of::<usize>()
    );
    let plan = plan.repeat_tensor(3).unwrap();
    let tensor = plan.resources();
    assert_eq!(
        tensor.private_material_bytes_total,
        3 * scalar.private_material_bytes_total
    );
    assert_eq!(tensor.dcf_keys_total, 3 * scalar.dcf_keys_total);
    assert_eq!(tensor.frames_total, 3 * scalar.frames_total);
    assert_eq!(
        tensor.public_table_copy_bytes_total,
        3 * scalar.public_table_copy_bytes_total
    );
    assert_eq!(cache.stats(), before);
    assert_eq!(
        budget.used_bytes(),
        before.metadata_reservation_bytes + tensor.plan_reservation_bytes
    );
    drop(cache);
    assert_eq!(budget.used_bytes(), tensor.plan_reservation_bytes);
    drop(plan);
    assert_eq!(budget.used_bytes(), 0);
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
        assert!(y.public_table_copy_bytes_total < x.public_table_copy_bytes_total);
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
    assert!(
        a.resources().public_table_copy_bytes_total
            < legacy.resources().public_table_copy_bytes_total
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
    assert_eq!(
        stats.lowered_chunk_constructions,
        (8192 + 256 + 1024 + 256 + 256) / 256
    );
    assert!(stats.lowered_retained_bytes > stats.lowered_payload_bytes);
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
