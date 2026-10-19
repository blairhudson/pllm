use super::*;

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
