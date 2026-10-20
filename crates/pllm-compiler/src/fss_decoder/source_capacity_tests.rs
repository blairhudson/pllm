use super::*;

#[test]
fn fixed_file_conversion_certificates_and_canonical_source_match_resident() {
    for (dtype, words) in [
        (
            WeightDtype::F32,
            vec![
                0,
                0x80000000,
                1,
                0x39000000,
                0x39c00000,
                0x3a200000,
                0xb9000000,
                0xb9c00000,
                0xba200000,
                0x3f800000,
                0xbf800000,
                0x52000000 - 1,
            ],
        ),
        (
            WeightDtype::F16,
            vec![
                0, 0x8000, 1, 0x0800, 0x0e00, 0x1100, 0x8800, 0x8e00, 0x9100, 0x3c00, 0xbc00,
                0x7bff,
            ],
        ),
        (
            WeightDtype::Bf16,
            vec![
                0, 0x8000, 1, 0x3900, 0x39c0, 0x3a20, 0xb900, 0xb9c0, 0xba20, 0x3f80, 0xbf80,
                0x51ff,
            ],
        ),
    ] {
        let resident = Tensor::from_ieee("weight", &[2, 6], dtype, &words).unwrap();
        let mut pending = TensorImport::new("weight", &[2, 6], Capacity::Decoder).unwrap();
        let width = if matches!(dtype, WeightDtype::F32) {
            4
        } else {
            2
        };
        let payload: Vec<_> = words
            .iter()
            .flat_map(|w| w.to_le_bytes().into_iter().take(width))
            .collect();
        for frame in payload.chunks(width * 5) {
            pending.append(dtype, frame).unwrap();
        }
        let paged = pending.finish().unwrap();
        assert_eq!(paged.values(), None);
        assert_eq!(paged.range(), resident.range());
        assert_eq!(paged.len(), 12);
        assert_eq!(paged.disk_bytes(), 12 * 8 + 2 * 48);
        let mut values = [0; 12];
        paged.read_rows(0, 2, &mut values).unwrap();
        assert_eq!(Some(values.as_slice()), resident.values());
        let mut slice = [0; 3];
        paged.read_row(1, 2, &mut slice).unwrap();
        assert_eq!(slice, values[8..11]);
        for row in 0..2 {
            assert_eq!(paged.row_statistics(row), resident.row_statistics(row));
        }
        assert!(paged.read_rows(1, 2, &mut values).is_err());
        assert!(paged.read_range(12, &mut slice).is_err());
        assert!(paged.read_range(0, &mut []).is_err());
        let capsule = Source::new(vec![resident]).unwrap();
        let decoder = Source::new_with_capacity(vec![paged], Capacity::Decoder).unwrap();
        assert_eq!(capsule.digest(), decoder.digest());
        assert_eq!(decoder.disk_bytes(), 192);
        assert!(decoder.reservation_bytes() >= SOURCE_READ_BYTES);
        assert!(decoder.reservation_bytes() < SOURCE_READ_BYTES + 8192);
    }
}

#[test]
fn larger_than_capsule_tensor_streams_without_resident_codes_or_row_table() {
    let shape = [40, 1000];
    assert!(Capacity::Capsule.source_elements(&shape).is_err());
    let mut writer = TensorImport::new("weight", &shape, Capacity::Decoder).unwrap();
    // This single BF16 frame expands to four bounded Q12 writer chunks.
    let frame = 0x3f80u16.to_le_bytes().repeat(32768);
    writer.append(WeightDtype::Bf16, &frame).unwrap();
    writer
        .append(
            WeightDtype::Bf16,
            &0xbf80u16.to_le_bytes().repeat(40000 - 32768),
        )
        .unwrap();
    let tensor = writer.finish().unwrap();
    assert_eq!(tensor.len(), 40000);
    assert_eq!(tensor.range(), (-4096, 4096));
    assert_eq!(tensor.disk_bytes(), 40000 * 8 + 40 * 48);
    assert!(tensor.resident_bytes() < 8192);
    let stats = tensor.row_statistics(32).unwrap();
    assert_eq!(
        (stats.positive_sum, stats.negative_sum),
        (768 * 4096, -232 * 4096)
    );
    let mut rows = [stats; 3];
    tensor.read_row_statistics(31, &mut rows).unwrap();
    assert_eq!(rows[0].positive_sum, 1000 * 4096);
    assert_eq!(rows[2].negative_sum, -1000 * 4096);
    let mut boundary = [0; 4];
    tensor.read_range(32766, &mut boundary).unwrap();
    assert_eq!(boundary, [4096, 4096, -4096, -4096]);
    assert!(tensor
        .read_range(0, &mut vec![0; SOURCE_READ_VALUES + 1])
        .is_err());
}

#[test]
fn backing_reports_only_unique_live_files_and_never_estimates_resident_storage() {
    let resident = Source::new(vec![Tensor::from_q12("resident", &[2], &[1, 2]).unwrap()]).unwrap();
    assert!(resident.disk_backing().unwrap().is_empty());
    let mut writer = TensorImport::new("first", &[2, 2], Capacity::Decoder).unwrap();
    writer
        .append(WeightDtype::F32, &1f32.to_bits().to_le_bytes().repeat(4))
        .unwrap();
    let first = writer.finish().unwrap();
    let second = Tensor {
        id: "second".to_owned(),
        shape: first.shape.clone(),
        values: Zeroizing::new(Vec::new()),
        range: first.range,
        file: first.file.clone(),
    };
    let source = Source::new_with_capacity(vec![first, second], Capacity::Decoder).unwrap();
    let records = source.disk_backing().unwrap();
    #[cfg(unix)]
    {
        assert_eq!(records.len(), 2);
        assert!(records.iter().all(|b| b.allocated_bytes > 0));
        assert_ne!(
            (records[0].device, records[0].file),
            (records[1].device, records[1].file)
        );
    }
    #[cfg(not(unix))]
    assert!(records.is_empty());
    let mut values = [0; 2];
    source.tensors[1].read_row(1, 0, &mut values).unwrap();
    assert_eq!(values, [4096; 2]);
}
