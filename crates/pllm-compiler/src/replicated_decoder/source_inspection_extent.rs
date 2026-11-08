//! Synthetic nonlookup operand extents, not authenticated inspection demand.
//! Categories stay separate: no total fee, funding witness or runtime wiring.
//! Full operand envelopes are conservative even when equality stops early.
use super::{add, Error};

#[derive(Default, Debug, PartialEq, Eq)]
pub(super) struct InspectionExtent {
    pub(super) tag_visits: usize,
    pub(super) length_header_visits: usize,
    pub(super) field_visits: usize,
    pub(super) operand_bytes: usize,
    pub(super) pointer_operands: usize,
}

pub(super) fn shape_eq(left_len: usize, right_len: usize) -> Result<InspectionExtent, Error> {
    Ok(InspectionExtent {
        length_header_visits: 2,
        field_visits: add(left_len, right_len)?,
        ..InspectionExtent::default()
    })
}

// Each present operand contributes its full payload, including mixed Options.
// This is a selected conservative envelope, not an actual-inspection claim.
pub(super) fn norm_option_eq(
    left_blocks: Option<usize>,
    right_blocks: Option<usize>,
) -> Result<InspectionExtent, Error> {
    let mut extent = InspectionExtent { tag_visits: 2, ..InspectionExtent::default() };
    for blocks in [left_blocks, right_blocks].into_iter().flatten() {
        let block_fields = blocks.checked_mul(3).ok_or(Error::Bounds)?;
        let fields = add(3, block_fields)?;
        extent.length_header_visits = add(extent.length_header_visits, 1)?;
        extent.field_visits = add(extent.field_visits, fields)?;
    }
    Ok(extent)
}

pub(super) fn string_eq(left_bytes: usize, right_bytes: usize) -> Result<InspectionExtent, Error> {
    Ok(InspectionExtent {
        length_header_visits: 2,
        operand_bytes: add(left_bytes, right_bytes)?,
        ..InspectionExtent::default()
    })
}

pub(super) fn digest_eq() -> InspectionExtent {
    InspectionExtent { operand_bytes: 64, ..InspectionExtent::default() }
}

// Supply actually visited entries, or explicitly label an M-entry upper bound.
// This helper does not measure any()'s short-circuit search.
pub(super) fn pointer_identity_scan(visited_entries: usize) -> Result<InspectionExtent, Error> {
    Ok(InspectionExtent {
        pointer_operands: visited_entries.checked_mul(2).ok_or(Error::Bounds)?,
        ..InspectionExtent::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_extents_keep_headers_and_fields_separate() {
        for (left, right, fields) in [(0, 0, 0), (2, 5, 7), (usize::MAX - 1, 1, usize::MAX)] {
            assert_eq!(shape_eq(left, right).unwrap(), InspectionExtent {
                tag_visits: 0, length_header_visits: 2, field_visits: fields,
                operand_bytes: 0, pointer_operands: 0,
            });
        }
    }

    #[test]
    fn shape_field_sum_overflow_refuses() {
        assert!(matches!(shape_eq(usize::MAX, 1), Err(Error::Bounds)));
    }

    #[test]
    fn string_extents_keep_headers_and_bytes_separate() {
        for (left, right, bytes) in [(0, 0, 0), (2, 5, 7), (usize::MAX - 1, 1, usize::MAX)] {
            assert_eq!(string_eq(left, right).unwrap(), InspectionExtent {
                tag_visits: 0, length_header_visits: 2, field_visits: 0,
                operand_bytes: bytes, pointer_operands: 0,
            });
        }
    }

    #[test]
    fn string_byte_sum_overflow_refuses() {
        assert!(matches!(string_eq(usize::MAX, 1), Err(Error::Bounds)));
    }

    #[test]
    fn norm_options_include_each_present_payload() {
        for (left, right, headers, fields) in [
            (None, None, 0, 0),
            (Some(2), None, 1, 9),
            (None, Some(2), 1, 9),
            (Some(0), None, 1, 3),
            (None, Some(0), 1, 3),
            (Some(0), Some(0), 2, 6),
            (Some(2), Some(4), 2, 24),
            (Some(1), Some(2), 2, 15),
        ] {
            assert_eq!(norm_option_eq(left, right).unwrap(), InspectionExtent {
                tag_visits: 2, length_header_visits: headers, field_visits: fields,
                operand_bytes: 0, pointer_operands: 0,
            });
        }
    }

    #[test]
    fn norm_block_multiplication_overflow_refuses() {
        assert!(matches!(norm_option_eq(Some(usize::MAX), None), Err(Error::Bounds)));
    }

    #[test]
    fn norm_scalar_addition_overflow_refuses() {
        assert!(matches!(norm_option_eq(Some(usize::MAX / 3), None), Err(Error::Bounds)));
    }

    #[test]
    fn norm_cross_operand_sum_overflow_refuses() {
        let blocks = (usize::MAX - 3) / 3;
        let fields = 3 + 3 * blocks;
        for (left, right) in [(Some(blocks), None), (None, Some(blocks))] {
            assert_eq!(norm_option_eq(left, right).unwrap(), InspectionExtent {
                tag_visits: 2, length_header_visits: 1, field_visits: fields,
                operand_bytes: 0, pointer_operands: 0,
            });
        }
        assert_eq!(norm_option_eq(Some(0), None).unwrap(), InspectionExtent {
            tag_visits: 2, length_header_visits: 1, field_visits: 3,
            operand_bytes: 0, pointer_operands: 0,
        });
        assert!(matches!(norm_option_eq(Some(blocks), Some(0)), Err(Error::Bounds)));
        assert!(matches!(norm_option_eq(Some(0), Some(blocks)), Err(Error::Bounds)));
    }

    #[test]
    fn digest_counts_only_operand_bytes() {
        assert_eq!(digest_eq(), InspectionExtent {
            tag_visits: 0, length_header_visits: 0, field_visits: 0,
            operand_bytes: 64, pointer_operands: 0,
        });
    }

    #[test]
    fn pointer_scan_counts_only_pointer_operands() {
        for (entries, operands) in [(0, 0), (1, 2), (7, 14), (usize::MAX / 2, usize::MAX - 1)] {
            assert_eq!(pointer_identity_scan(entries).unwrap(), InspectionExtent {
                tag_visits: 0, length_header_visits: 0, field_visits: 0,
                operand_bytes: 0, pointer_operands: operands,
            });
        }
    }

    #[test]
    fn pointer_operand_multiplication_overflow_refuses() {
        assert!(matches!(pointer_identity_scan(usize::MAX / 2 + 1), Err(Error::Bounds)));
    }
}
