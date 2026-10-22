use super::*;

fn attempts() -> usize {
    ALLOCATION_ATTEMPTS.with(|n| n.get())
}

#[test]
fn affine_quote_exact_one_short_before_direct_arc_allocation() {
    let quote = Layout::quote_affine(7, 2, &[2, 3], &[3, 1]).unwrap();
    assert_eq!(quote.retained_backing_bytes, 0);
    assert_eq!(quote.span_buffer_bytes, 0);
    assert_eq!(
        quote.destination_backing_bytes,
        2 * size_of::<usize>() + size_of::<Span>()
    );
    assert_eq!(
        quote.construction_peak_bytes,
        quote.destination_backing_bytes
    );
    for short in [false, true] {
        let before = attempts();
        let result = Layout::affine_admitted(
            7,
            2,
            &[2, 3],
            &[3, 1],
            quote.construction_peak_bytes - usize::from(short),
        );
        if short {
            assert!(matches!(result, Err(Error::Bounds)));
            assert_eq!(attempts(), before);
        } else {
            let layout = result.unwrap();
            assert_eq!(attempts(), before + 1);
            assert_eq!(
                layout.backing_bytes().unwrap(),
                quote.destination_backing_bytes
            );
            assert_eq!(layout.metadata_bytes(), quote.destination_backing_bytes);
            assert_eq!(
                layout.row_key(6),
                Some(RowKey {
                    owner: 7,
                    offset: 2,
                    width: 6
                })
            );
        }
    }
}

#[test]
fn quotes_reject_invalid_geometry_and_overflow_without_allocation() {
    let before = attempts();
    assert_eq!(Layout::quote_contiguous(0, 0), Err(Error::Shape));
    assert!(Layout::quote_contiguous(0, MAX_VALUES).is_ok());
    assert_eq!(
        Layout::quote_contiguous(0, MAX_VALUES + 1),
        Err(Error::Bounds)
    );
    assert_eq!(Layout::quote_affine(0, 0, &[], &[]), Err(Error::Shape));
    assert_eq!(Layout::quote_affine(0, 0, &[2], &[]), Err(Error::Shape));
    assert_eq!(
        Layout::quote_affine(0, 0, &[1; 5], &[1; 5]),
        Err(Error::Shape)
    );
    assert_eq!(
        Layout::quote_affine(0, usize::MAX, &[2], &[1]),
        Err(Error::Bounds)
    );
    assert_eq!(
        Layout::quote_affine(0, 0, &[3], &[usize::MAX]),
        Err(Error::Bounds)
    );
    assert_eq!(
        Layout::quote_affine(0, 0, &[MAX_VALUES, 2], &[0, 0]),
        Err(Error::Bounds)
    );
    assert_eq!(LayoutQuote::new(usize::MAX, 1, 0), Err(Error::Bounds));
    assert_eq!(LayoutQuote::new(0, 1, usize::MAX), Err(Error::Bounds));
    assert_eq!(LayoutQuote::new(0, MAX_NODES + 1, 0), Err(Error::Bounds));
    assert!(LayoutQuote::new(0, MAX_NODES, MAX_NODES)
        .unwrap()
        .admit(MAX_PROGRAM_BYTES)
        .is_ok());
    assert_eq!(LayoutQuote::new(0, 2, 1), Err(Error::Bounds));
    assert_eq!(Layout::quote_concat(&[]), Err(Error::Bounds));
    assert_eq!(attempts(), before);
}

#[test]
fn concat_quotes_alias_backing_and_vec_arc_overlap_before_reserve() {
    let row = Layout::contiguous(0, 3).unwrap();
    let parts = [row.clone(), row.clone()];
    assert_eq!(parts[0].identity(), parts[1].identity());
    let quote = Layout::quote_concat(&parts).unwrap();
    assert_eq!(
        quote.retained_backing_bytes,
        2 * row.backing_bytes().unwrap()
    );
    assert_eq!(quote.span_buffer_bytes, 2 * size_of::<Span>());
    assert_eq!(
        quote.destination_backing_bytes,
        2 * size_of::<usize>() + 2 * size_of::<Span>()
    );
    assert_eq!(
        quote.construction_peak_bytes,
        quote.retained_backing_bytes + quote.span_buffer_bytes + quote.destination_backing_bytes
    );
    for short in [false, true] {
        let before = attempts();
        let result =
            Layout::concat_admitted(&parts, quote.construction_peak_bytes - usize::from(short));
        if short {
            assert!(matches!(result, Err(Error::Bounds)));
            assert_eq!(attempts(), before);
        } else {
            let joined = result.unwrap();
            assert_eq!(attempts(), before + 2);
            assert_eq!(
                joined.backing_bytes().unwrap(),
                quote.destination_backing_bytes
            );
            assert_eq!(joined.len(), 6);
            for i in 0..6 {
                assert_eq!(joined.word(i).unwrap(), row.word(i % 3).unwrap());
            }
            assert!(joined.row_key(6).is_none());
        }
    }
}

#[test]
fn span_buffer_quotes_actual_spare_capacity_without_refund() {
    let row = Layout::contiguous(0, 3).unwrap();
    let mut spans = Vec::with_capacity(9);
    spans.push(row.0[0].clone());
    let retained = row.backing_bytes().unwrap();
    let compact = LayoutQuote::new(retained, spans.len(), spans.len()).unwrap();
    let actual = LayoutQuote::new(retained, spans.len(), spans.capacity()).unwrap();
    assert_eq!(
        actual.construction_peak_bytes - compact.construction_peak_bytes,
        (spans.capacity() - spans.len()) * size_of::<Span>()
    );
    let before = attempts();
    // A constructor must reject an allocator-reported excess before Arc conversion,
    // even when a more generous limit could hold that excess.
    assert_eq!(
        Layout::check_span_buffer(&spans, 1, compact, actual.construction_peak_bytes),
        Err(Error::Bounds)
    );
    assert_eq!(attempts(), before);
}

#[test]
fn slice_quotes_intersected_geometry_and_full_retained_owner_before_reserve() {
    let parts = [
        Layout::affine(0, 0, &[1], &[1]).unwrap(),
        Layout::affine(0, 1, &[4], &[1]).unwrap(),
        Layout::affine(0, 5, &[2], &[1]).unwrap(),
    ];
    let joined = Layout::concat(&parts).unwrap();
    for (start, len, spans) in [(0, 6, 3), (1, 3, 1), (4, 2, 2)] {
        let quote = joined.quote_slice(start, len).unwrap();
        assert_eq!(
            quote.retained_backing_bytes,
            joined.backing_bytes().unwrap()
        );
        assert_eq!(quote.span_buffer_bytes, spans * size_of::<Span>());
        for short in [false, true] {
            let before = attempts();
            let result = joined.slice_admitted(
                start,
                len,
                quote.construction_peak_bytes - usize::from(short),
            );
            if short {
                assert!(matches!(result, Err(Error::Bounds)));
                assert_eq!(attempts(), before);
            } else {
                let slice = result.unwrap();
                assert_eq!(attempts(), before + 2);
                assert_eq!(
                    slice.backing_bytes().unwrap(),
                    quote.destination_backing_bytes
                );
                for i in 0..len {
                    assert_eq!(slice.word(i).unwrap(), joined.word(start + i).unwrap());
                }
                assert_eq!(
                    slice.row_key(len),
                    Some(RowKey {
                        owner: 0,
                        offset: start,
                        width: len
                    })
                );
            }
        }
    }
    let before = attempts();
    assert_eq!(joined.quote_slice(usize::MAX, 1), Err(Error::Shape));
    assert_eq!(joined.quote_slice(0, 0), Err(Error::Shape));
    assert_eq!(joined.quote_slice(6, 2), Err(Error::Shape));
    assert_eq!(attempts(), before);
}
