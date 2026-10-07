//! Geometry-only, checked resource accounting. Projections allocate no circuit
//! and never authorize issuance; the same admission is enforced by the owner.
use super::*;
use crate::boolean::{
    MAX_REFERENCE_AND_GATES, MAX_REFERENCE_INPUTS, MAX_REFERENCE_OUTPUTS, MAX_REFERENCE_WIRES,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Projection {
    pub estimate: Estimate,
    pub actual_wires: usize,
    pub and_gates: usize,
    pub constant_label_bytes: usize,
    pub public_metadata_bytes_estimate: usize,
}

impl Projection {
    pub fn admitted(&self) -> bool {
        self.estimate.combined_allocation_bound_bytes <= MAX_ALLOCATION_BYTES
            && self.estimate.input_bits <= MAX_REFERENCE_INPUTS
            && self.estimate.output_bits <= MAX_REFERENCE_OUTPUTS
            && self.estimate.wire_bound <= MAX_REFERENCE_WIRES
            && self.and_gates <= MAX_REFERENCE_AND_GATES
    }
}

/// Exact gate/material counts for these two concrete layouts, including padded
/// networks and the complete gather. Geometry may be costed even when rejected.
pub fn project(
    rows: usize,
    width: usize,
    policy: Policy,
    state_bytes: usize,
    layout: Layout,
) -> Result<Projection, String> {
    if !(1..=MAX_ROWS).contains(&rows)
        || !(1..=MAX_WIDTH).contains(&width)
        || policy.keep == 0
        || policy.keep > rows
        || policy.recent > policy.keep
    {
        return Err("retained selection geometry or policy exceeds its bounds".into());
    }
    // Compute in u128; conversion and arbitrary caller-owned state sizes are
    // checked at the boundary, including on 32-bit targets.
    let index = index_bits(rows) as u128;
    let rank = index + 34;
    let gather_index = index + 1; // includes an out-of-range padding sentinel
    let payload = 32 + 64 * width as u128;
    let input = rows as u128 * (32 + payload);
    let output = policy.keep as u128 * payload;
    let (record, order_key) = match layout {
        Layout::FullRecords => (rank + payload, 32),
        Layout::CompactIndices => (rank + gather_index, gather_index),
    };
    let first = comparisons(rows) as u128;
    let second = comparisons(policy.keep) as u128;
    let key_bits = first * rank + second * order_key;
    let swaps = (first + second) * record;
    let constants = rows as u128
        * (index
            + 2
            + if layout == Layout::CompactIndices {
                gather_index
            } else {
                0
            })
        + ((rows.next_power_of_two() - rows) + (policy.keep.next_power_of_two() - policy.keep))
            as u128
            * record
        + 2 * (first + second);
    let mut nots = 2 * key_bits;
    let mut xors = 2 * key_bits + 3 * swaps;
    let mut ands = 3 * key_bits + swaps;
    let mut gather_wires = 0;
    if layout == Layout::CompactIndices {
        let gather_nots = policy.keep as u128 * gather_index;
        let gather_ands = policy.keep as u128 * rows as u128 * (gather_index - 1 + payload);
        let gather_xors = policy.keep as u128 * (rows - 1) as u128 * payload;
        nots += gather_nots;
        ands += gather_ands;
        xors += gather_xors;
        gather_wires = gather_nots + gather_ands + gather_xors;
    }
    let actual_wires = input + constants + nots + xors + ands;
    // Preserve the historical full-record conservative bound. Compact indices
    // use the same network bound plus every gather wire. 384 bytes/wire prices
    // encoding/instruction capacity overlap and labels; boundary owners, native
    // compaction overlap and 4 MiB scratch remain separately charged.
    let wire_bound = input
        + 2 * (rows.next_power_of_two() + policy.keep.next_power_of_two()) as u128 * record
        + first * (7 * rank + 2 + 4 * record)
        + second * (7 * order_key + 2 + 4 * record)
        + gather_wires;
    let allocation = wire_bound * 384 + (input + output) * 64 + (4 << 20) + 2 * state_bytes as u128;
    let fit = |n| {
        usize::try_from(n)
            .map_err(|_| "retained selection resource size overflows usize".to_string())
    };
    Ok(Projection {
        estimate: Estimate {
            wire_bound: fit(wire_bound)?,
            combined_allocation_bound_bytes: fit(allocation)?,
            input_bits: fit(input)?,
            output_bits: fit(output)?,
        },
        actual_wires: fit(actual_wires)?,
        and_gates: fit(ands)?,
        constant_label_bytes: fit(constants * LABEL_BYTES as u128)?,
        public_metadata_bytes_estimate: fit(64
            + 8 * output
            + constants
            + 9 * nots
            + 17 * xors
            + 25 * ands)?,
    })
}
