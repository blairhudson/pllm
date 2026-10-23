use super::*;
use crate::fixed_exp_tables;
use pllm_core::edge_contract::{EdgeIdentity, Shape, SourceIdentity};

fn edge(lanes: usize, slot: usize) -> EdgeDescriptor {
    let bound = read_bound(slot).unwrap();
    let identity = EdgeIdentity::new(
        SourceIdentity::new([1; 32], [2; 32], super::super::approved_implementation()),
        [3; 32],
        [slot as u8 + 20; 32],
        [slot as u8 + 40; 32],
        17,
    );
    let representation = if bound.numeric.interpretation() == Interpretation::Predicate {
        Representation::CommonXorMasked
    } else {
        Representation::CommonAdditiveMasked
    };
    EdgeDescriptor::new(
        identity,
        Shape::new(&[lanes]).unwrap(),
        bound.numeric,
        representation,
        None,
    )
    .unwrap()
}
fn descriptors(lanes: usize) -> [EdgeDescriptor; READ_COUNT] {
    let mut reads = std::array::from_fn(|slot| edge(lanes, slot));
    reads[1] = reads[0];
    reads[5] = reads[2];
    reads[10] = reads[9];
    reads[13] = reads[11];
    reads
}
fn certified() -> Verified {
    let mut workspace = crate::exp_floor::Workspace::default();
    // Entire Rust floor gate must succeed; tests DEFINED only until slot allows.
    fixed_exp_tables::verify(&mut workspace).unwrap()
}
fn replace_numeric(edge: EdgeDescriptor, numeric: Numeric) -> EdgeDescriptor {
    EdgeDescriptor::new(
        edge.identity(),
        edge.shape(),
        numeric,
        edge.representation(),
        None,
    )
    .unwrap()
}

#[test]
fn each_clear_read_fits_its_declared_numeric_not_a_full_mask_bound() {
    for slot in 0..READ_COUNT {
        assert!(fits_numeric(read_bound(slot).unwrap()));
    }
    assert_eq!(read_bound(READ_COUNT), Err(Error::Domain));
    let affine = read_bound(3).unwrap();
    assert_eq!(
        (affine.interval.lower, affine.interval.upper),
        (-65536, (1i128 << 38) - 65536)
    );
    assert!(affine.interval.fits(40));
    for slot in [7, 8, 9, 10] {
        assert_eq!(
            read_bound(slot).unwrap().interval,
            Interval {
                lower: 0,
                upper: 65536
            }
        );
        assert!(!read_bound(slot).unwrap().interval.fits(16)); // NOT signed16!
    }
    for slot in [11, 12, 13] {
        let bound = read_bound(slot).unwrap();
        assert_eq!(
            bound.numeric,
            Numeric::new(8, 0, Interpretation::Unsigned).unwrap()
        );
        assert_eq!(
            bound.interval,
            Interval {
                lower: 0,
                upper: 256
            }
        );
        assert!(!bound.interval.fits(8)); // Index255 is legal unsigned, not -1.
    }
    // Uniform ring50 masks/opened words can exceed ALL these logical bounds.
    assert!(i128::from((1u64 << 50) - 1) >= read_bound(17).unwrap().interval.upper);
}

#[test]
fn verified_literal_extrema_include_zero_and_exact_product_endpoint() {
    let checked = certified();
    certified_ranges(&checked).unwrap();
    assert_eq!(
        extrema(checked.values(Table::LowByte)).unwrap(),
        Interval {
            lower: 3848,
            upper: 4097
        }
    );
    assert_eq!(
        extrema(checked.values(Table::HighByte)).unwrap(),
        Interval {
            lower: 0,
            upper: 4097
        }
    );
    let high = checked.values(Table::HighByte);
    assert_eq!((high[0], high[133], high[134], high[255]), (4096, 1, 0, 0));
    let low = checked.values(Table::LowByte);
    assert_eq!(low[0] * high[0], 1 << 24);
    assert_eq!((low[255] * high[255]) / 4096, 0);
    let p = product_interval(
        read_bound(14).unwrap().interval,
        read_bound(15).unwrap().interval,
    )
    .unwrap();
    assert_eq!(
        p,
        Interval {
            lower: 0,
            upper: (1i128 << 24) + 1
        }
    );
    assert!(quarter_fits(p, 50) && p.fits(50));
    assert_eq!(
        p.shift(12),
        Interval {
            lower: 0,
            upper: 4097
        }
    );
}

#[test]
fn actual_whole_clear_root_and_all18_records_preflight_for_every_lane_width() {
    let checked = certified();
    for lanes in 1..=256 {
        let reads = descriptors(lanes);
        let mut values = vec![(1i64 << 38) - 1; lanes];
        values[0] = 0;
        let intervals = preflight(&reads[0], &values, &checked, reads.each_ref()).unwrap();
        for (slot, index) in EDGE_BY_USE.iter().enumerate() {
            let truth = read_bound(slot).unwrap();
            let declared = intervals[*index].unwrap();
            assert!(
                declared.lower <= truth.interval.lower && declared.upper >= truth.interval.upper
            );
            assert!(fits_numeric(ReadBound {
                interval: declared,
                ..truth
            }));
        }
        assert_eq!(
            intervals[10],
            Some(Interval {
                lower: 0,
                upper: 4097
            })
        );
        values.fill((1i64 << 38) - 1);
        preflight(&reads[0], &values, &checked, reads.each_ref()).unwrap();
        for invalid in [-1, 1i64 << 38] {
            values[lanes - 1] = invalid;
            assert_eq!(
                preflight(&reads[0], &values, &checked, reads.each_ref()),
                Err(Error::Domain)
            );
        }
    }
}

#[test]
fn public_tr_q4_and_signed16_or_signed8_reinterpretations_are_rejected() {
    let checked = certified();
    for (slot, wrong) in [
        (11, Numeric::new(8, 4, Interpretation::Signed).unwrap()),
        (12, Numeric::new(8, 0, Interpretation::Signed).unwrap()),
        (7, Numeric::new(16, 12, Interpretation::Signed).unwrap()),
        (16, Numeric::new(50, 12, Interpretation::Signed).unwrap()),
        (17, Numeric::new(50, 24, Interpretation::Signed).unwrap()),
    ] {
        let mut reads = descriptors(1);
        reads[slot] = replace_numeric(reads[slot], wrong);
        assert_eq!(
            preflight(&reads[0], &[0], &checked, reads.each_ref()),
            Err(Error::Domain)
        );
    }
}

#[test]
fn repeated_original_declared_parent_is_not_replaceable_by_equal_range() {
    let checked = certified();
    for slot in [1, 5, 10, 13] {
        let mut reads = descriptors(1);
        reads[slot] = edge(1, slot); // Same numeric/shape/range; different wire.
        assert_eq!(
            preflight(&reads[0], &[0], &checked, reads.each_ref()),
            Err(Error::Context)
        );
    }
    let mut reads = descriptors(1);
    reads[14] = edge(2, 14);
    assert_eq!(
        preflight(&reads[0], &[0], &checked, reads.each_ref()),
        Err(Error::Domain)
    );
}

#[test]
fn clipping_affines_are_modular_and_drelu_includes_zero() {
    for x in [0i128, 1, 32767, 65534, 65535, 65536, (1i128 << 38) - 1] {
        let a = x - 65536;
        assert!(
            (read_bound(3).unwrap().interval.lower..read_bound(3).unwrap().interval.upper)
                .contains(&a)
        );
        let h = if a >= 0 { 1i128 } else { 0 };
        let b = 1 - h;
        let w = (x - 65535).rem_euclid(65536);
        let v = b * w;
        let c = (v + 65535).rem_euclid(65536);
        assert_eq!(c, x.min(65535));
        assert_eq!((c % 256) + 256 * (c / 256), c);
        assert!((0..256).contains(&(c / 256)));
        if x == 65536 {
            assert_eq!((a, h, b, c), (0, 1, 0, 65535));
        }
        if x == 0 {
            assert_eq!((a, w, v, c), (-65536, 1, 1, 0));
        }
        if x == 32767 {
            assert_eq!(w, 32768);
            assert!(v + 65535 >= 65536);
        }
    }
}

#[test]
fn complete_quarter_boundaries_and_negative_floor_are_half_open() {
    let q = 1i128 << 48;
    assert!(quarter_fits(
        Interval {
            lower: -q,
            upper: q
        },
        50
    ));
    assert!(!quarter_fits(
        Interval {
            lower: -q - 1,
            upper: q
        },
        50
    ));
    assert!(!quarter_fits(
        Interval {
            lower: 0,
            upper: q + 1
        },
        50
    ));
    assert!(!quarter_fits(Interval { lower: 0, upper: 0 }, 50));
    assert_eq!(
        Interval {
            lower: -2048,
            upper: -2047
        }
        .shift(12),
        Interval {
            lower: -1,
            upper: 0
        }
    );
    assert_eq!(
        Interval {
            lower: 6144,
            upper: 6145
        }
        .shift(12),
        Interval { lower: 1, upper: 2 }
    );
    let signed_table = Interval {
        lower: -4096,
        upper: 4097,
    };
    let selected = Interval {
        lower: 0,
        upper: 4097,
    };
    assert!(signed_table.fits(50) && selected.fits(50));
}

#[test]
fn both_d19_sign_branches_keep_lookup_stage_bounds_and_restore_table_value() {
    let checked = certified();
    for (table, slot) in [(Table::LowByte, 14), (Table::HighByte, 15)] {
        let payload_range = read_bound(slot).unwrap().interval;
        for &payload in checked.values(table) {
            for sigma in [-1i128, 1] {
                // Mathematical terms only: raw opened words still have masks.
                let r = sigma * i128::from(payload);
                let correction = if sigma == 1 { 1i128 } else { 0 };
                let selected = correction * r;
                assert!((-4096..4097).contains(&r));
                assert!((0..4097).contains(&selected));
                let reconstructed = 2 * selected - r;
                assert_eq!(reconstructed, i128::from(payload));
                assert!((payload_range.lower..payload_range.upper).contains(&reconstructed));
            }
        }
    }
}
