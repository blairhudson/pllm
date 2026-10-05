//! Bounded one-use private float32 absolute-maximum cost reference.
//! No floating-point approximation: finite magnitudes have unsigned bit order.
//! The evaluator receives only labels; only the trusted client decodes the max.
//! This is not a quantizer, transport, compiler component or reviewed protocol.

use pllm_garble::boolean::{
    BooleanCircuitBuilder, BooleanCircuitClient, BooleanCircuitDecoder, BooleanCircuitInputs,
    BooleanCircuitProgram,
};
use std::time::Instant;

const BITS: usize = 31;
const MAX_WIDTH: usize = 2560;

fn prepare(width: usize) -> Result<(BooleanCircuitClient, BooleanCircuitProgram), String> {
    if !(1..=MAX_WIDTH).contains(&width) {
        return Err("absmax reference width must be in [1,2560]".into());
    }
    let mut builder = BooleanCircuitBuilder::new()?;
    let words = (0..width)
        .map(|_| builder.input_word(BITS))
        .collect::<Result<Vec<_>, _>>()?;
    let mut maximum = words[0].clone();
    for word in &words[1..] {
        let (less, _) = builder.compare_unsigned(&maximum, word)?;
        maximum = builder.select_word(less, &maximum, word)?;
    }
    builder.finish(&maximum)
}

fn encode(
    client: BooleanCircuitClient,
    values: &[f32],
) -> Result<(BooleanCircuitInputs, BooleanCircuitDecoder), String> {
    // Consumes client material even on a shape/domain failure.
    if values.is_empty() || values.len() > MAX_WIDTH || values.iter().any(|v| !v.is_finite()) {
        return Err("absmax requires bounded finite float32 inputs".into());
    }
    let bits = values
        .iter()
        .flat_map(|value| (0..BITS).map(move |bit| value.to_bits() & (1 << bit) != 0))
        .collect::<Vec<_>>();
    client.encode(&bits)
}

fn trial(width: usize) -> Result<String, String> {
    // Public synthetic stand-ins, including signed zero and a subnormal.
    let mut values = (0..width)
        .map(|i| ((i * 101 % 4093) as f32 - 2046.0) / 113.0)
        .collect::<Vec<_>>();
    if width >= 3 {
        values[..3].copy_from_slice(&[-0.0, f32::from_bits(1), -19.125]);
    }
    let expected = values.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
    let start = Instant::now();
    let (client, program) = prepare(width)?;
    let prepare_seconds = start.elapsed().as_secs_f64();
    let gates = program.and_gate_count();
    let ciphertext_bytes = program.evaluator_ciphertext_bytes()?;
    if gates != (width - 1) * BITS * 4 {
        return Err("absmax public gate estimate differs from built circuit".into());
    }
    let start = Instant::now();
    let (inputs, decoder) = encode(client, &values)?;
    let encode_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let outputs = program.evaluate(inputs)?;
    let evaluate_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let bits = decoder.decode(outputs)?;
    let decoded = bits
        .iter()
        .enumerate()
        .fold(0_u32, |word, (i, bit)| word | (u32::from(*bit) << i));
    let client_seconds = encode_seconds + start.elapsed().as_secs_f64();
    if decoded != expected.to_bits() {
        return Err("garbled absmax differs from independent float32 maximum".into());
    }
    let mut control = 0.0_f32;
    let start = Instant::now();
    for _ in 0..1000 {
        control = std::hint::black_box(&values)
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0_f32, f32::max);
        std::hint::black_box(control);
    }
    let clear_seconds = start.elapsed().as_secs_f64() / 1000.0;
    if control.to_bits() != expected.to_bits() {
        return Err("clear absmax control differs".into());
    }
    Ok(format!(
        "{{\"width\":{width},\"and_gates\":{gates},\"ciphertext_body_bytes\":{ciphertext_bytes},\
        \"online_input_label_bytes\":{},\"online_output_label_bytes\":{},\
        \"uncompressed_helper_to_client_encoding_bytes\":{},\"prepare_wall_seconds\":{prepare_seconds},\
        \"client_wall_seconds\":{client_seconds},\"evaluator_wall_seconds\":{evaluate_seconds},\
        \"clear_wall_seconds_per_call\":{clear_seconds},\"exact_float32_max\":true}}",
        width * BITS * 16,
        BITS * 16,
        2 * (width + 1) * BITS * 16,
    ))
}

fn main() -> Result<(), String> {
    let results = [8, 128, 2560]
        .into_iter()
        .map(trial)
        .collect::<Result<Vec<_>, _>>()?;
    println!(
        "{{\"schema\":\"pllm.client_absmax_reference.v1\",\"samples\":[{}]}}",
        results.join(",")
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_zero_extremes_subnormal_and_equal_magnitudes_are_exact() {
        for values in [
            vec![-0.0],
            vec![f32::from_bits(1), -f32::from_bits(2)],
            vec![f32::MAX, -f32::MAX, 1.0],
            vec![-7.25, 7.25, -7.0, 0.0],
        ] {
            let (client, program) = prepare(values.len()).unwrap();
            let (inputs, decoder) = encode(client, &values).unwrap();
            let bits = decoder.decode(program.evaluate(inputs).unwrap()).unwrap();
            let word = bits
                .iter()
                .enumerate()
                .fold(0_u32, |w, (i, b)| w | (u32::from(*b) << i));
            assert_eq!(
                word,
                values
                    .iter()
                    .copied()
                    .map(f32::abs)
                    .fold(0.0_f32, f32::max)
                    .to_bits()
            );
        }
    }

    #[test]
    fn rejects_domain_shape_and_cross_circuit_material() {
        assert!(prepare(0).is_err());
        assert!(prepare(MAX_WIDTH + 1).is_err());
        for values in [
            vec![f32::NAN],
            vec![f32::INFINITY],
            vec![-f32::INFINITY],
            vec![1.0, 2.0],
        ] {
            let (client, _) = prepare(1).unwrap();
            assert!(encode(client, &values).is_err());
        }
        let (client_a, program_a) = prepare(2).unwrap();
        let (client_b, program_b) = prepare(2).unwrap();
        let (inputs_a, _) = encode(client_a, &[1.0, 2.0]).unwrap();
        let (inputs_b, _) = encode(client_b, &[3.0, 4.0]).unwrap();
        assert!(program_a.evaluate(inputs_b).is_err());
        assert!(program_b.evaluate(inputs_a).is_err());
    }
}
