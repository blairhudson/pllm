use super::*;

// Workload composition is deliberately TEST ONLY. No geometry/admission API.
fn weighted_source_work<const N: usize>(terms: [(u64, u64); N]) -> Result<u64, Error> {
    terms.into_iter().try_fold(0u64, |sum, (count, rate)| {
        count.checked_mul(rate).and_then(|term| sum.checked_add(term)).ok_or(Error::Bounds)
    })
}

fn local_source_pass(n: u64, b: u64, t: u64) -> Result<u64, Error> {
    let rates = scan_work_rates()?;
    let replacements = b.checked_sub(t).ok_or(Error::Bounds)?;
    weighted_source_work([
        (n, rates.local_k_n), (b, rates.local_k_b),
        (replacements, rates.local_k_r), (t, rates.local_k_t),
    ])
}

#[test]
fn source_work_metadata_is_repeatable_pure_and_preserves_core_rates() {
    // No BlockScan, MatrixScan, source values, graph or state constructed.
    let first = scan_work_rates().unwrap();
    let copied = first;
    assert_eq!(first, copied);
    assert_eq!(first, scan_work_rates().unwrap());
    assert_eq!(first.core_append, pllm_core::integer_bounds::MatrixScan::work_rates().unwrap());
    assert_eq!(first.core_lifecycle,
        pllm_core::integer_bounds::MatrixScan::lifecycle_work_rates().unwrap());
    assert_eq!(first.core_lifecycle.append, first.core_append);
}

#[test]
fn source_work_inventory_checks_sum_product_add_and_conversion_overflow() {
    assert_eq!(source_work_sum([u64::MAX, 1]), Err(Error::Bounds));
    assert_eq!(source_work_sum([0, 0]), Ok(0));
    assert_eq!(source_work_h(u64::MAX, 0, [2]), Err(Error::Bounds));
    assert_eq!(source_work_h(1, u64::MAX, [1]), Err(Error::Bounds));
    // LP64: inventory sum overflows; narrower usize: multiplication overflows.
    // The direct u64 sum assertion above isolates addition on either platform.
    assert_eq!(source_work_h(u64::MAX, 0, [usize::MAX, usize::MAX]), Err(Error::Bounds));
    assert_eq!(source_work_narrow(u128::from(u64::MAX)), Ok(u64::MAX));
    assert_eq!(source_work_narrow(u128::from(u64::MAX).checked_add(1).unwrap()), Err(Error::Bounds));
    // Distinct types are not deduplicated merely because byte extents match.
    assert_eq!(source_work_h(3, 7, [2, 2]), Ok(19));
    assert_eq!(weighted_source_work([(u64::MAX, 2)]), Err(Error::Bounds));
    assert_eq!(weighted_source_work([(1, u64::MAX), (1, 1)]), Err(Error::Bounds));
}

#[test]
fn source_work_zero_occurrences_and_checked_replacement_counts() {
    let r = scan_work_rates().unwrap();
    assert_eq!(weighted_source_work([(0, u64::MAX), (0, r.source_terminal_error)]), Ok(0));
    assert_eq!(local_source_pass(0, 0, 0), Ok(0));
    assert_eq!(local_source_pass(0, 0, 1), Err(Error::Bounds));
    assert_eq!(local_source_pass(0, 2, 3), Err(Error::Bounds));
    // B=T means no replacement, not a phantom next-block constructor.
    assert_eq!(local_source_pass(1, 1, 1),
        weighted_source_work([(1, r.local_k_n), (1, r.local_k_b), (1, r.local_k_t)]));
    assert_eq!(local_source_pass(2, 2, 1), weighted_source_work([
        (2, r.local_k_n), (2, r.local_k_b), (1, r.local_k_r), (1, r.local_k_t),
    ]));
}

#[test]
fn source_work_singleton_nonboundary_boundary_and_adapter_construction_partition() {
    let r = scan_work_rates().unwrap();
    let nonboundary = weighted_source_work([(1, r.local_k_n), (1, r.adapter_append_v)]).unwrap();
    let boundary_final = weighted_source_work([
        (1, r.local_k_n), (1, r.adapter_append_v), (1, r.local_k_b), (1, r.adapter_finish_f),
    ]).unwrap();
    let boundary_replacing = source_work_sum([
        boundary_final, r.local_k_r, r.preflight_p, r.preflight_a, r.adapter_construct_u,
    ]).unwrap();
    assert_eq!(boundary_final.checked_sub(nonboundary),
        Some(source_work_sum([r.local_k_b, r.adapter_finish_f]).unwrap()));
    assert_eq!(boundary_replacing.checked_sub(boundary_final),
        Some(source_work_sum([r.local_k_r, r.preflight_p, r.preflight_a, r.adapter_construct_u]).unwrap()));
    // Synthetic two-block/one-record pass: initial + replacement = B exactly.
    // Each adapter construction has P+A+U; J/S are per source construction.
    let initial = source_work_sum([
        r.preflight_p, r.preflight_a, r.preflight_s, r.source_construct_j, r.adapter_construct_u,
    ]).unwrap();
    let replacement = source_work_sum([r.preflight_p, r.preflight_a, r.adapter_construct_u]).unwrap();
    assert_eq!(source_work_sum([initial, replacement]), weighted_source_work([
        (2, r.preflight_p), (2, r.preflight_a), (1, r.preflight_s),
        (1, r.source_construct_j), (2, r.adapter_construct_u),
    ]));
    // Neither decomposition includes core constructor/finish or backing retirement.
}

#[test]
fn source_work_preflight_a_owns_payload_result_flow_residual_not_adapter_u() {
    use pllm_core::integer_bounds::{Error as K, MatrixScan as CM};
    use std::{convert::Infallible, mem::size_of, ops::ControlFlow};
    type RE<X> = Result<X, Error>;
    type ZE = RE<Infallible>;
    let r = scan_work_rates().unwrap();
    let a = source_work_h(256, 512, [
        size_of::<usize>(), size_of::<Error>(), size_of::<RE<usize>>(),
        size_of::<ControlFlow<ZE, usize>>(), size_of::<ZE>(), size_of::<RE<MatrixScan>>(),
    ]).unwrap();
    let u = source_work_h(8, 64, [
        size_of::<usize>(), size_of::<CM>(), size_of::<MatrixScan>(),
        size_of::<Result<CM, K>>(), size_of::<Result<MatrixScan, K>>(),
        size_of::<RE<MatrixScan>>(),
    ]).unwrap();
    assert_eq!(r.preflight_a, a);
    assert_eq!(r.adapter_construct_u, u);
    // Even zero-sized residual payloads retain their distinct Result/Flow types.
    assert_eq!(source_work_sum([r.preflight_a, r.adapter_construct_u]), source_work_sum([a, u]));
}

#[test]
fn source_work_local_scalar_allowances_and_terminal_distinct_type_union() {
    use std::{convert::Infallible, mem::size_of, ops::ControlFlow};
    type RE<X> = Result<X, Error>;
    type ZE = RE<Infallible>;
    type FE<X> = ControlFlow<ZE, X>;
    let r = scan_work_rates().unwrap();
    let kn = source_work_sum([
        source_work_h(1, 0, [
            size_of::<Pending>(), size_of::<RE<Pending>>(), size_of::<FE<Pending>>(),
            size_of::<Option<&mut MatrixScan>>(), size_of::<RE<&mut MatrixScan>>(),
            size_of::<FE<&mut MatrixScan>>(), size_of::<&mut MatrixScan>(),
            size_of::<FE<()>>(), size_of::<&mut BlockScan>(), size_of::<Option<usize>>(),
        ]).unwrap(),
        source_work_h(2, 0, [size_of::<Option<Pending>>(), size_of::<RE<()>>(), size_of::<Error>()]).unwrap(),
        source_work_h(3, 0, [size_of::<&[i64]>()]).unwrap(),
        source_work_h(32, 0, [size_of::<usize>()]).unwrap(),
        source_work_h(16, 0, [size_of::<bool>()]).unwrap(),
    ]).unwrap();
    assert_eq!(r.local_k_n, kn.checked_add(48).unwrap());
    let kb = source_work_sum([
        source_work_h(1, 0, [
            size_of::<Option<MatrixScan>>(), size_of::<RE<MatrixScan>>(), size_of::<FE<MatrixScan>>(),
            size_of::<MatrixScan>(), size_of::<RE<MatrixCertificate>>(), size_of::<FE<MatrixCertificate>>(),
            size_of::<MatrixCertificate>(), size_of::<Block>(), size_of::<Error>(),
        ]).unwrap(),
        source_work_h(8, 0, [size_of::<usize>()]).unwrap(),
        source_work_h(4, 0, [size_of::<bool>()]).unwrap(),
    ]).unwrap();
    assert_eq!(r.local_k_b, kb.checked_add(16).unwrap());
    let kr = source_work_sum([
        source_work_h(1, 0, [size_of::<RE<MatrixScan>>(), size_of::<FE<MatrixScan>>(),
            size_of::<MatrixScan>(), size_of::<Option<MatrixScan>>()]).unwrap(),
        source_work_h(8, 0, [size_of::<usize>()]).unwrap(),
    ]).unwrap();
    assert_eq!(r.local_k_r, kr.checked_add(8).unwrap());
    let kt = source_work_sum([
        source_work_h(1, 0, [size_of::<BlockScan>(), size_of::<Option<Pending>>(),
            size_of::<RE<Pending>>(), size_of::<FE<Pending>>(), size_of::<Pending>(),
            size_of::<BlockNorms>(), size_of::<RE<BlockNorms>>(), size_of::<RE<()>>(),
            size_of::<FE<()>>(), size_of::<Error>(), size_of::<&BlockNorms>()]).unwrap(),
        source_work_h(8, 0, [size_of::<usize>()]).unwrap(),
        source_work_h(4, 0, [size_of::<bool>()]).unwrap(),
    ]).unwrap();
    assert_eq!(r.local_k_t, kt.checked_add(16).unwrap());
    let union = source_work_h(1, 0, [
        size_of::<Pending>(), size_of::<Option<Pending>>(), size_of::<RE<Pending>>(), size_of::<FE<Pending>>(),
        size_of::<Option<&mut MatrixScan>>(), size_of::<RE<&mut MatrixScan>>(), size_of::<FE<&mut MatrixScan>>(),
        size_of::<&mut MatrixScan>(), size_of::<RE<()>>(), size_of::<FE<()>>(), size_of::<&mut BlockScan>(),
        size_of::<&[i64]>(), size_of::<Error>(), size_of::<Option<usize>>(), size_of::<usize>(), size_of::<bool>(),
        size_of::<Option<MatrixScan>>(), size_of::<RE<MatrixScan>>(), size_of::<FE<MatrixScan>>(), size_of::<MatrixScan>(),
        size_of::<RE<MatrixCertificate>>(), size_of::<FE<MatrixCertificate>>(), size_of::<MatrixCertificate>(),
        size_of::<Block>(), size_of::<BlockScan>(), size_of::<BlockNorms>(), size_of::<RE<BlockNorms>>(), size_of::<&BlockNorms>(),
    ]).unwrap();
    let terminal = union.checked_add(source_work_h(1, 0, [size_of::<ZE>()]).unwrap()).unwrap()
        .checked_mul(32).unwrap().checked_add(64).unwrap();
    assert_eq!(r.source_terminal_error, terminal);
}

#[test]
fn source_work_validation_counts_independent_of_append_and_consuming_finish() {
    use pllm_core::integer_bounds::{Error as K, MatrixCertificate as CC};
    use std::{convert::Infallible, mem::size_of, ops::ControlFlow};
    type E = pllm_core::puma::Error;
    type AC = MatrixCertificate;
    type D = BlockNorms;
    type RE<X> = Result<X, E>;
    type RK<X> = Result<X, K>;
    type ZE = RE<Infallible>;
    type ZK = RK<Infallible>;
    type FE<X> = ControlFlow<ZE, X>;
    type FK<X> = ControlFlow<ZK, X>;
    // Independent absolute oracle: no production inventory helpers or rates.
    let expected_h = [
        size_of::<Block>(), size_of::<AC>(), size_of::<CC>(), size_of::<(usize, usize)>(),
        size_of::<&D>(), size_of::<&Vec<Block>>(), size_of::<&[Block]>(), size_of::<&Block>(),
        size_of::<Option<&Block>>(), size_of::<Option<AC>>(), size_of::<Option<usize>>(),
        size_of::<Option<u128>>(), size_of::<Option<Infallible>>(),
        size_of::<ControlFlow<Option<Infallible>, &Block>>(),
        size_of::<RE<usize>>(), size_of::<RE<()>>(), size_of::<RE<AC>>(),
        size_of::<RE<CC>>(), size_of::<RE<u128>>(), size_of::<RK<usize>>(),
        size_of::<RK<()>>(), size_of::<RK<CC>>(), size_of::<RK<u128>>(),
        size_of::<FE<usize>>(), size_of::<FE<()>>(), size_of::<FE<AC>>(),
        size_of::<FE<CC>>(), size_of::<FE<u128>>(), size_of::<FK<usize>>(),
        size_of::<FK<()>>(), size_of::<FK<CC>>(), size_of::<ZE>(), size_of::<ZK>(),
        size_of::<E>(), size_of::<K>(),
    ].into_iter().try_fold(0u64, |sum, size| {
        sum.checked_add(u64::try_from(size).unwrap())
    }).unwrap();
    let expected_j = [size_of::<usize>(), size_of::<u128>(), size_of::<bool>(), size_of::<()>()]
        .into_iter().try_fold(0u64, |sum, size| {
            sum.checked_add(u64::try_from(size).unwrap())
        }).unwrap();
    let expected_u = 256u64.checked_add(expected_h.checked_mul(64).unwrap()).unwrap()
        .checked_add(expected_j.checked_mul(128).unwrap()).unwrap();
    let expected_two = expected_u.checked_mul(2).unwrap();
    let expected_nine = expected_u.checked_mul(9).unwrap();
    let r = scan_work_rates().unwrap();
    assert_eq!(r.validation_fixed, expected_two);
    assert_eq!(r.validation_record, expected_two);
    assert_eq!(r.validation_block, expected_nine);
    assert_eq!(r.validation_terminal_error, expected_nine);
    assert!(r.validation_fixed > 0);
    assert!(r.validation_record > 0);
    assert!(r.validation_block > 0);
    assert!(r.validation_terminal_error > 0);
    assert_eq!(r.validation_fixed, r.validation_record);
    assert_eq!(r.validation_fixed.checked_mul(9), r.validation_block.checked_mul(2));
    assert_eq!(r.validation_terminal_error, r.validation_block);
    // Caller-supplied synthetic V/Tv/Bv, not inferred from N/B/T or model sets.
    let validation = weighted_source_work([
        (3, r.validation_fixed), (2, r.validation_record), (5, r.validation_block),
    ]).unwrap();
    let local = local_source_pass(7, 2, 1).unwrap();
    assert_eq!(source_work_sum([local, validation]).unwrap().checked_sub(local), Some(validation));
    let finish_boundary = weighted_source_work([(1, r.local_k_t)]).unwrap();
    assert_eq!(source_work_sum([finish_boundary, validation]).unwrap().checked_sub(finish_boundary), Some(validation));
    assert_eq!(weighted_source_work([(0, r.validation_fixed), (0, r.validation_record),
        (0, r.validation_block), (0, r.validation_terminal_error)]), Ok(0));
    // Shared validation is priceable; resident finish ownership is not qualified.
}

#[test]
fn source_work_reserve_failure_has_j_k_without_phantom_pending_or_core_retirement() {
    use std::{collections::TryReserveError, mem::size_of};
    let r = scan_work_rates().unwrap();
    let k = source_work_h(8, 64, [size_of::<TryReserveError>(), size_of::<Error>(),
        size_of::<Vec<Block>>(), size_of::<&mut Vec<Block>>(), size_of::<usize>(), size_of::<bool>()]).unwrap();
    assert_eq!(r.source_construct_reserve_error_k, k);
    let failure = weighted_source_work([
        (1, r.source_construct_j), (1, r.source_construct_reserve_error_k),
        (0, r.source_retire_local.source_shell), (0, r.core_append.retire),
        (0, r.adapter_construct_u),
    ]).unwrap();
    assert_eq!(failure, source_work_sum([r.source_construct_j, k]).unwrap());
}

#[test]
fn source_work_actual_mapper_once_and_retirement_owners_stay_separate() {
    use std::mem::size_of;
    let r = scan_work_rates().unwrap();
    assert_eq!(r.adapter_terminal_error_mapper, source_work_h(4, 16, [
        size_of::<pllm_core::integer_bounds::Error>(), size_of::<Error>(),
    ]).unwrap());
    let failure = weighted_source_work([
        (1, r.adapter_terminal_error_mapper), (1, r.source_terminal_error),
    ]).unwrap();
    assert_eq!(failure.checked_sub(r.source_terminal_error), Some(r.adapter_terminal_error_mapper));
    // A preflight-P failure already owns its mapping; no extra mapper invocation.
    assert_eq!(weighted_source_work([(1, r.preflight_p), (0, r.adapter_terminal_error_mapper)]), Ok(r.preflight_p));
    assert_eq!(r.source_retire_local.source_shell,
        source_work_h(1, 8, [size_of::<Pending>(), size_of::<BlockScan>()]).unwrap());
    assert_eq!(r.source_retire_local.record_shell,
        source_work_h(1, 8, [size_of::<BlockNorms>(), size_of::<Vec<Block>>()]).unwrap());
    assert_eq!(r.source_retire_local.retained_block, source_work_h(1, 0, [size_of::<Block>()]).unwrap());
    // One eventual semantic-block extent per retained block; inner backing only
    // core_append.retire once, never folded into either source shell rate.
    assert_eq!(weighted_source_work([(1, r.source_retire_local.source_shell),
        (1, r.source_retire_local.record_shell), (2, r.source_retire_local.retained_block),
        (1, r.core_append.retire)]), source_work_sum([
            r.source_retire_local.source_shell, r.source_retire_local.record_shell,
            r.source_retire_local.retained_block, r.source_retire_local.retained_block, r.core_append.retire,
        ]));
}

#[test]
fn source_projection1_successful_norm_component() {
    // Generic Q/K-and-bias-like geometry only: no checkpoint binding, selected
    // source bias/dimensions, model count or actual call-graph claim.
    let mut t = 0u64;
    let mut n = 0u64;
    let mut rows_total = 0u64;
    let mut b = 0u64;
    let mut columns_total = 0u64;
    for (rows, columns, block_rows) in [
        (896u64, 896u64, 64u64), (128, 896, 64), (896, 1, 64), (128, 1, 64),
    ] {
        assert!(block_rows > 0);
        let blocks = rows.checked_add(block_rows.checked_sub(1).unwrap()).unwrap()
            .checked_div(block_rows).unwrap();
        t = t.checked_add(1).unwrap();
        n = n.checked_add(rows.checked_mul(columns).unwrap()).unwrap();
        rows_total = rows_total.checked_add(rows).unwrap();
        b = b.checked_add(blocks).unwrap();
        columns_total = columns_total.checked_add(blocks.checked_mul(columns).unwrap()).unwrap();
    }
    let groups = 24u64;
    let t = t.checked_mul(groups).unwrap();
    let n = n.checked_mul(groups).unwrap();
    let rows_total = rows_total.checked_mul(groups).unwrap();
    let b = b.checked_mul(groups).unwrap();
    let columns_total = columns_total.checked_mul(groups).unwrap();
    let replacements = b.checked_sub(t).unwrap();
    assert_eq!((t, n, rows_total, b, columns_total, replacements),
        (96, 22_044_672, 49_152, 768, 344_448, 672));
    // Stipulated validation scenario, not inferred operational invocation counts.
    let v = t.checked_mul(11).unwrap();
    let tv = v;
    let bv = b.checked_mul(11).unwrap();
    assert_eq!((v, tv, bv), (1056, 1056, 8448));
    let two_t = t.checked_mul(2).unwrap();
    let two_n = n.checked_mul(2).unwrap();
    let two_r = rows_total.checked_mul(2).unwrap();
    let two_b = b.checked_mul(2).unwrap();
    let two_l = columns_total.checked_mul(2).unwrap();
    let two_replacements = replacements.checked_mul(2).unwrap();
    let p_count = two_b.checked_add(two_t).unwrap().checked_add(tv).unwrap();
    assert_eq!(p_count, 2784);
    let r = scan_work_rates().unwrap();
    let q = r.core_append;
    let l = r.core_lifecycle;
    assert_eq!(l.append, q); // Re-export equality, never add append twice.
    let core_append_retire = weighted_source_work([
        (two_n, q.append_entry), (two_n, q.value),
        (two_r, q.completed_row), (two_b, q.retire),
    ]).unwrap();
    let core_lifecycle = weighted_source_work([
        (two_b, l.construct), (two_l, l.initialize_column),
        (two_b, l.finish), (two_l, l.finish_column),
    ]).unwrap();
    let preflight = weighted_source_work([
        (p_count, r.preflight_p), (two_b, r.preflight_a), (two_t, r.preflight_s),
    ]).unwrap();
    let construction_adapter = weighted_source_work([
        (two_t, r.source_construct_j), (two_b, r.adapter_construct_u),
        (two_n, r.adapter_append_v), (two_b, r.adapter_finish_f),
    ]).unwrap();
    let source_local = weighted_source_work([
        (two_n, r.local_k_n), (two_b, r.local_k_b),
        (two_replacements, r.local_k_r), (two_t, r.local_k_t),
    ]).unwrap();
    let validation = weighted_source_work([
        (v, r.validation_fixed), (tv, r.validation_record), (bv, r.validation_block),
    ]).unwrap();
    // Projection assumes two produced norm sets. Other clones/maps are NOT free.
    let source_retire = weighted_source_work([
        (two_t, r.source_retire_local.source_shell),
        (two_t, r.source_retire_local.record_shell),
        (two_b, r.source_retire_local.retained_block),
    ]).unwrap();
    let successful_n_component = weighted_source_work([
        (1, core_append_retire), (1, core_lifecycle), (1, preflight),
        (1, construction_adapter), (1, source_local), (1, validation), (1, source_retire),
    ]).unwrap();
    let flattened = weighted_source_work([
        (two_n, q.append_entry), (two_n, q.value), (two_r, q.completed_row), (two_b, q.retire),
        (two_b, l.construct), (two_l, l.initialize_column), (two_b, l.finish), (two_l, l.finish_column),
        (p_count, r.preflight_p), (two_b, r.preflight_a), (two_t, r.preflight_s),
        (two_t, r.source_construct_j), (two_b, r.adapter_construct_u),
        (two_n, r.adapter_append_v), (two_b, r.adapter_finish_f),
        (two_n, r.local_k_n), (two_b, r.local_k_b), (two_replacements, r.local_k_r), (two_t, r.local_k_t),
        (v, r.validation_fixed), (tv, r.validation_record), (bv, r.validation_block),
        (two_t, r.source_retire_local.source_shell), (two_t, r.source_retire_local.record_shell),
        (two_b, r.source_retire_local.retained_block),
        // Successful projection ONLY: all failure/reserve/mapper/terminal extras zero.
        (0, q.terminal_error), (0, l.construct_error), (0, l.finish_error),
        (0, r.source_construct_reserve_error_k), (0, r.adapter_terminal_error_mapper),
        (0, r.validation_terminal_error), (0, r.source_terminal_error),
    ]).unwrap();
    assert_eq!(successful_n_component, flattened);
    // Logical source-work units, NOT physical bytes/CPU/RSS or a budget-fit claim.
    // EXCLUDED: quote-helper evaluation, map+key work, caller propagation,
    // equality, extra clones, C-G-R-F stream, digest/file work, terminal cleanup,
    // original ledger debt. No funding, admission, security or activation claim.
    println!("GEOMETRY groups={groups} T={t} N={n} R={rows_total} B={b} L={columns_total} replacements={replacements} V={v} Tv={tv} Bv={bv}");
    println!("core_append_retire={core_append_retire}");
    println!("core_lifecycle={core_lifecycle}");
    println!("preflight={preflight}");
    println!("construction_adapter={construction_adapter}");
    println!("source_local={source_local}");
    println!("validation={validation}");
    println!("source_retire={source_retire}");
    println!("successful_N_COMPONENT={successful_n_component}");
}

#[test]
fn import_envelope_legacy_empty_and_exact_geometry_agree() {
    assert_eq!(
        import_envelope(&SourceNorms::new()).unwrap(),
        ImportEnvelope::default()
    );
    assert_eq!(
        requests_envelope(&BTreeMap::new()).unwrap(),
        ImportEnvelope::default()
    );
    let mut scan = BlockScan::new(7, 5, 3).unwrap();
    scan.append(0, &[1; 35]).unwrap();
    let key = "q.weight".to_owned();
    let records = BTreeMap::from([(key.clone(), scan.finish().unwrap())]);
    let planned = BTreeMap::from([(key, (7, 5, 3))]);
    let actual = import_envelope(&records).unwrap();
    assert_eq!(actual, requests_envelope(&planned).unwrap());
    assert_eq!(
        actual.total_bytes,
        4 * actual.retained_bytes
            + actual.request_bytes
            + actual.largest_scan_bytes
            + actual.largest_output_bytes
    );
}

#[test]
fn import_envelope_wide_columns_outweigh_four_record_copies() {
    let mut scan = BlockScan::new(1, 16_384, 1).unwrap();
    for offset in (0..16_384).step_by(128) {
        scan.append(offset, &[0; 128]).unwrap();
    }
    let records = BTreeMap::from([("q.weight".to_owned(), scan.finish().unwrap())]);
    let envelope = import_envelope(&records).unwrap();
    assert!(envelope.largest_scan_bytes >= 16_384 * std::mem::size_of::<u128>());
    assert!(envelope.largest_scan_bytes > 4 * metadata_bytes(&records).unwrap());
    assert!(envelope.total_bytes <= MAX_METADATA_BYTES);
}

#[test]
fn import_envelope_maximum_shapes_and_aggregate_limit() {
    let maximum_columns = BTreeMap::from([("q.weight".to_owned(), (128, 16_384, 128))]);
    assert!(requests_envelope(&maximum_columns).unwrap().total_bytes <= MAX_METADATA_BYTES);
    // Individually valid scan geometry can exceed the combined owner envelope.
    assert!(BlockScan::payload_bytes(4096 * 128, 16_384, 128).is_ok());
    let maximum_both = BTreeMap::from([("q.weight".to_owned(), (4096 * 128, 16_384, 128))]);
    assert!(requests_envelope(&maximum_both).is_err());
    let repeated: BTreeMap<_, _> = (0..4)
        .map(|i| (format!("q{i}.weight"), (128, 16_384, 128)))
        .collect();
    let envelope = requests_envelope(&repeated).unwrap();
    // Independent source scans are sequential, not four simultaneous scratch sets.
    assert_eq!(
        envelope.largest_scan_bytes,
        requests_envelope(&maximum_columns)
            .unwrap()
            .largest_scan_bytes
    );
    let too_many: BTreeMap<_, _> = (0..1024)
        .map(|i| (format!("q{i}.weight"), (1, 1, 1)))
        .collect();
    assert!(requests_envelope(&too_many).is_err());
}

#[test]
fn import_envelope_rejects_overflow_and_hidden_capacity_before_clone() {
    for geometry in [
        (usize::MAX, 16_384, 128),
        (128, usize::MAX, 128),
        (128, 16_384, 0),
        (128, 16_385, 128),
        (4097, 1, 1),
    ] {
        assert!(requests_envelope(&BTreeMap::from([("q".to_owned(), geometry)])).is_err());
    }
    let mut scan = BlockScan::new(1, 1, 1).unwrap();
    scan.append(0, &[1]).unwrap();
    let mut records = BTreeMap::from([("q".to_owned(), scan.finish().unwrap())]);
    records
        .values_mut()
        .next()
        .unwrap()
        .blocks
        .reserve_exact(MAX_METADATA_BYTES / 48);
    assert!(import_envelope(&records).is_err());
    let mut key = String::with_capacity(MAX_METADATA_BYTES);
    key.push('q');
    assert!(requests_envelope(&BTreeMap::from([(key, (1, 1, 1))])).is_err());
}

#[test]
fn source_norm_blocks_stream_exact_statistics_and_partial_tail() {
    let values: Vec<_> = (0..35).map(|i| i as i64 % 11 - 5).collect();
    for chunk in [1, 2, 7, 11, 35] {
        let mut scan = BlockScan::new(7, 5, 3).unwrap();
        for (index, part) in values.chunks(chunk).enumerate() {
            scan.append(index * chunk, part).unwrap();
        }
        let actual = scan.finish().unwrap();
        assert_eq!(actual.geometry(), (7, 5, 3));
        for (index, block) in values.chunks(15).enumerate() {
            let expected = MatrixCertificate {
                rows: block.len() / 5,
                columns: 5,
                maximum_row_l1: block
                    .chunks(5)
                    .map(|row| row.iter().map(|v| v.unsigned_abs() as u128).sum())
                    .max()
                    .unwrap(),
                maximum_column_l1: (0..5)
                    .map(|column| {
                        block
                            .iter()
                            .skip(column)
                            .step_by(5)
                            .map(|v| v.unsigned_abs() as u128)
                            .sum()
                    })
                    .max()
                    .unwrap(),
                frobenius_squared: block
                    .iter()
                    .map(|v| (v.unsigned_abs() as u128).pow(2))
                    .sum(),
            };
            assert_eq!(actual.block(index * 3, block.len() / 5), Some(expected));
        }
        let encoded = serde_json::to_vec(&actual).unwrap();
        let decoded: BlockNorms = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(actual, decoded);
        assert!(actual.block(1, 3).is_none());
        assert!(actual.block(0, 2).is_none());
        assert!(actual.validate(5, 7).is_err());
    }
}

#[test]
fn source_norm_blocks_failures_retire_ordered_scan_and_bound_allocations() {
    for (offset, values) in [(1, vec![1]), (0, vec![]), (0, vec![1; 7])] {
        let mut scan = BlockScan::new(2, 3, 1).unwrap();
        assert!(scan.append(offset, &values).is_err());
        assert!(scan.append(0, &[1]).is_err());
        assert!(scan.finish().is_err());
    }
    let mut scan = BlockScan::new(2, 3, 1).unwrap();
    scan.append(0, &[1; 6]).unwrap();
    assert!(scan.append(6, &[1]).is_err());
    assert!(scan.finish().is_err());
    assert!(BlockScan::new(2, 3, 0).is_err());
    assert!(BlockScan::new(4097, 3, 1).is_err());
    assert!(BlockScan::new(128, 16385, 128).is_err());
    // The scan retains column totals, never the complete source block.
    assert!(BlockScan::payload_bytes(896, 896, 64).unwrap() < 20_000);
}

#[test]
fn source_norm_blocks_reject_substituted_geometry_and_unknown_metadata() {
    let mut scan = BlockScan::new(2, 3, 2).unwrap();
    scan.append(0, &[1; 6]).unwrap();
    let norms = scan.finish().unwrap();
    for field in ["rows", "columns", "block_rows"] {
        let mut json = serde_json::to_value(&norms).unwrap();
        json[field] = serde_json::json!(0);
        let changed: BlockNorms = serde_json::from_value(json).unwrap();
        assert!(changed.validate(2, 3).is_err());
    }
    let mut json = serde_json::to_value(&norms).unwrap();
    json["ignored"] = serde_json::json!(true);
    assert!(serde_json::from_value::<BlockNorms>(json).is_err());
}
