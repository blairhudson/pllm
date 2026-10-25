use super::*;

#[test]
fn original_f32_cast_toward_zero_and_subnormal_behavior_is_not_reinterpreted() {
    for (value, expected) in [
        (0.5 / ONE as f32, 0),
        (-0.5 / ONE as f32, 0),
        (1.5 / ONE as f32, 1),
        (-1.5 / ONE as f32, -1),
        (f32::from_bits(1), 0),
        (-f32::from_bits(1), 0),
    ] {
        assert_eq!(encode_q18(&value.to_le_bytes()).unwrap(), expected);
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(encode_q18(&value.to_le_bytes()).is_err());
    }
}

#[test]
fn actual_snapshot_word_boundaries_128_and_4096_preserve_full_tail_bytes() {
    for count in [127, 128, 129, 4095, 4096, 4097] {
        let words: Vec<_> = (0..count)
            .map(|i| if i % 2 == 0 { 0.125f32 } else { -0.375f32 })
            .collect();
        let bytes: Vec<_> = words.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut scan = Scan::new(count, None).unwrap();
        let mut writer = PublicFileWriter::new_computed((count * 8) as u64).unwrap();
        let mut gamma = vec![0; count];
        for (frame, chunk) in bytes.chunks(CHUNK * 4).enumerate() {
            scan.append(frame * CHUNK, chunk, &mut writer, Some(&mut gamma), None)
                .unwrap();
        }
        let summary = scan.finish().unwrap();
        let snapshot = writer.finish().unwrap();
        let expected: Vec<_> = bytes
            .chunks_exact(4)
            .map(|v| encode_q18(v).unwrap())
            .collect();
        assert_eq!(gamma, expected);
        let expected_bytes: Vec<_> = expected.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut actual = vec![0; count * 8];
        snapshot.read(0, &mut actual).unwrap();
        assert_eq!(actual, expected_bytes);
        assert_eq!(summary.digest, snapshot.digest());
    }
}

#[test]
fn borrowed_block_scan_crosses_original_row_and_block_boundaries() {
    let bytes: Vec<_> = (1..=8).flat_map(|v| (v as f32).to_le_bytes()).collect();
    let expected: Vec<_> = bytes
        .chunks_exact(4)
        .map(|v| encode_q18(v).unwrap())
        .collect();
    let mut reference = source_norms::BlockScan::new(4, 2, 2).unwrap();
    reference.append(0, &expected).unwrap();
    let mut norms = source_norms::BlockScan::new(4, 2, 2).unwrap();
    let mut scan = Scan::new(8, Some(2)).unwrap();
    let mut writer = PublicFileWriter::new_computed(64).unwrap();
    for (i, chunk) in bytes.chunks(12).enumerate() {
        scan.append(i * 3, chunk, &mut writer, None, Some(&mut norms))
            .unwrap();
    }
    assert_eq!(norms.finish().unwrap(), reference.finish().unwrap());
    assert_eq!(
        scan.finish().unwrap().digest,
        writer.finish().unwrap().digest()
    );
}

#[test]
fn borrowed_failure_leaves_partial_gamma_norm_and_open_writer_for_collector_cleanup() {
    for broken_norm in [false, true] {
        let mut scan = Scan::new(2, None).unwrap();
        let mut writer = PublicFileWriter::new_computed(16).unwrap();
        let mut gamma = [0; 2];
        let mut norms =
            source_norms::BlockScan::new(if broken_norm { 1 } else { 2 }, 1, 1).unwrap();
        let bytes: Vec<_> = [1f32, if broken_norm { 2. } else { f32::NAN }]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert!(scan
            .append(0, &bytes, &mut writer, Some(&mut gamma), Some(&mut norms))
            .is_err());
        assert_eq!(gamma, [ONE, 0]);
        assert!(norms.finish().is_err());
        assert!(scan.finish().is_err());
        // Deliberately show component does NOT close a borrowed writer. A real
        // collector must DROP it, never complete/publish it as done below solely
        // to observe the open handle in this non-authoritative component test.
        writer.write(&[0; 16]).unwrap();
        assert!(writer.finish().is_ok());
    }
}

#[test]
fn private_stat_overflow_branch_burns_without_huge_source_or_wrapping() {
    for energy in [false, true] {
        let mut scan = Scan::new(1, Some(1)).unwrap();
        if energy {
            scan.row_l1_squared_sum = u128::MAX;
        } else {
            scan.row_sum = u64::MAX;
        }
        let mut writer = PublicFileWriter::new_computed(8).unwrap();
        let mut gamma = [0];
        // Unit-level branch injection, NOT a genuine source/certificate fixture.
        assert!(scan
            .append(0, &1f32.to_le_bytes(), &mut writer, Some(&mut gamma), None)
            .is_err());
        assert_eq!(gamma, [ONE]);
        assert!(scan.finish().is_err());
        drop(writer);
    }
}

#[test]
fn bounded_scan_matches_original_q18_row_statistics_and_snapshot_bytes() {
    let mut scan = Scan::new(6, Some(3)).unwrap();
    let mut writer = PublicFileWriter::new_computed(48).unwrap();
    let input = [1f32, -2., 0.5, 0., 3., -1.];
    let bytes: Vec<_> = input.iter().flat_map(|f| f.to_le_bytes()).collect();
    scan.append(0, &bytes[..8], &mut writer, None, None)
        .unwrap();
    scan.append(2, &bytes[8..], &mut writer, None, None)
        .unwrap();
    let summary = scan.finish().unwrap();
    let file = writer.finish().unwrap();
    let expected: Vec<_> = bytes
        .chunks_exact(4)
        .map(|v| encode_q18(v).unwrap())
        .collect();
    let expected_bytes: Vec<_> = expected.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut actual = [0; 48];
    file.read(0, &mut actual).unwrap();
    assert_eq!(actual.as_slice(), expected_bytes);
    assert_eq!(summary.digest, file.digest());
    assert_eq!(summary.lower, -2 * ONE);
    assert_eq!(summary.upper, 3 * ONE);
    assert_eq!(summary.row_l1, Some(4 * ONE as u64));
    assert_eq!(summary.row_squared_norm, Some(10 * (ONE as u128).pow(2)));
    assert_eq!(
        summary.row_l1_squared_sum,
        Some(113 * (ONE as u128).pow(2) / 4)
    );
}

#[test]
fn gamma_stays_in_snapshot_and_actual_preallocated_destination() {
    let mut scan = Scan::new(2, None).unwrap();
    let mut gamma = [0; 2];
    let mut writer = PublicFileWriter::new_computed(16).unwrap();
    let bytes: Vec<_> = [1f32, 2.].iter().flat_map(|f| f.to_le_bytes()).collect();
    scan.append(0, &bytes, &mut writer, Some(&mut gamma), None)
        .unwrap();
    let summary = scan.finish().unwrap();
    let file = writer.finish().unwrap();
    assert_eq!(gamma, [ONE, 2 * ONE]);
    assert_eq!(summary.row_l1, None);
    assert_eq!(summary.digest, file.digest());
    assert!(Scan::inline_peak_bytes() >= ENCODED_BYTES + 2 * size_of::<Scan>());
    assert_eq!(
        super::super::compact_publisher_scan_bytes(),
        Scan::inline_peak_bytes()
    );
}

#[test]
fn bad_order_length_float_and_writer_failure_burn_scan_without_retry() {
    for fault in 0..4 {
        let mut scan = Scan::new(2, None).unwrap();
        let mut writer = PublicFileWriter::new_computed(if fault == 3 { 8 } else { 16 }).unwrap();
        let valid: Vec<_> = [1f32, 2.].iter().flat_map(|f| f.to_le_bytes()).collect();
        let invalid = f32::NAN.to_le_bytes();
        let (offset, bytes) = match fault {
            0 => (1, valid.as_slice()),
            1 => (0, &valid[..3]),
            2 => (0, invalid.as_slice()),
            _ => (0, valid.as_slice()),
        };
        assert!(scan.append(offset, bytes, &mut writer, None, None).is_err());
        assert!(scan.append(0, &valid, &mut writer, None, None).is_err());
        assert!(scan.finish().is_err());
    }
}
