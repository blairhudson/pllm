use super::*;

/// Sort only rank/index wires. Then scan every original row for each retained
/// output, using hidden one-hot predicates; no secret-dependent Rust indexing.
pub(super) fn circuit(
    rows: usize,
    width: usize,
    policy: Policy,
) -> Result<(crate::boolean::BooleanCircuitClient, BooleanCircuitProgram), String> {
    let mut builder = BooleanCircuitBuilder::new()?;
    let mut inputs = Vec::with_capacity(rows);
    for _ in 0..rows {
        let score = builder.input_word(32)?;
        let mut payload = builder.input_word(32)?;
        for _ in 0..2 * width {
            payload.extend(builder.input_word(32)?);
        }
        inputs.push((score, payload));
    }
    // Extra bit makes the all-ones padding index strictly larger than every
    // real slot, including the last row of a power-of-two input.
    let stable = index_bits(rows);
    let index = stable + 1;
    let rank = rank_bits(rows);
    let record_bits = rank + index;
    let mut records = Vec::with_capacity(rows.next_power_of_two());
    for (slot, (score, _)) in inputs.iter().enumerate() {
        let mut row = builder.constant_word(((1 << stable) - 1 - slot) as u128, stable)?;
        row.extend(score);
        row.push(builder.constant(slot >= rows - policy.recent)?);
        row.push(builder.constant(true)?);
        row.extend(builder.constant_word(slot as u128, index)?);
        records.push(row);
    }
    while records.len() < rows.next_power_of_two() {
        records.push(
            (0..record_bits)
                .map(|_| builder.constant(false))
                .collect::<Result<_, _>>()?,
        );
    }
    sort(&mut builder, &mut records, 0..rank, false)?;
    records.truncate(policy.keep);
    while records.len() < policy.keep.next_power_of_two() {
        records.push(
            (0..record_bits)
                .map(|_| builder.constant(true))
                .collect::<Result<_, _>>()?,
        );
    }
    // Native ownership requires strictly increasing original positions; input
    // slots have that order already. Dummy indices sort after all kept indices.
    sort(&mut builder, &mut records, rank..rank + index, true)?;
    let payload_bits = 32 + 64 * width;
    let mut output = Vec::with_capacity(policy.keep * payload_bits);
    for record in records.iter().take(policy.keep) {
        let slot = &record[rank..];
        let inverted = slot
            .iter()
            .map(|&v| builder.not(v))
            .collect::<Result<Vec<_>, _>>()?;
        let mut flags = Vec::with_capacity(rows);
        for original in 0..rows {
            let matching = |bit: usize| {
                if original & (1usize << bit) == 0 {
                    inverted[bit]
                } else {
                    slot[bit]
                }
            };
            let mut flag = matching(0);
            for bit in 1..index {
                flag = builder.and(flag, matching(bit))?;
            }
            flags.push(flag);
        }
        for bit in 0..payload_bits {
            let mut value = builder.and(flags[0], inputs[0].1[bit])?;
            for row in 1..rows {
                let selected = builder.and(flags[row], inputs[row].1[bit])?;
                value = builder.xor(value, selected)?;
            }
            output.push(value);
        }
    }
    builder.finish(&output)
}
