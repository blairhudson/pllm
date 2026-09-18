//! Streaming half-gates backend for exact direct-weight Q10 RMSNorm.
//!
//! Evaluator bodies contain no header or instruction stream: each secret-secret
//! AND contributes exactly two consecutive 16-byte half-gate ciphertexts. Both
//! parties reconstruct the circuit from the same public weights and topology.
//!
//! This API is deliberately consuming and its secret-bearing types are not
//! `Clone` or `Debug`. A compiler wrapper must provide a unique, complete source
//! identity and enforce authenticated, one-use issuance. This research crate
//! does not provide either lifecycle property.

use std::{error::Error, fmt, io::Read, io::Write};

use subtle::ConstantTimeEq;
use zeroize::{Zeroize, Zeroizing};

use crate::boolean::{fixed_key_hash, xor_block, CIRCUIT_ID_BYTES, LABEL_BYTES};

const MAX_WIDTH: usize = 8192;
const INPUT_BITS: usize = 16;
const OUTPUT_BITS: usize = 64;
const SUM_BITS: usize = 44;
const FRACTIONAL_BITS: u32 = 30;
const BYTES_PER_AND: u64 = (2 * LABEL_BYTES) as u64;

/// Exact streamed half-gates construction selected by compiler bindings.
pub const RMS_NORM_Q10_STREAM_METHOD_ID: &str =
    "pllm.garble.boolean_stream.rms_norm_q10_direct.half_gates.v1";
/// Exact deterministic circuit topology reconstructed by both parties.
pub const RMS_NORM_Q10_STREAM_TOPOLOGY_ID: &str =
    "pllm.garble.boolean_stream.rms_norm_q10_direct.topology.v1";

#[derive(Clone, Copy)]
pub struct UnreviewedRmsNormQ10Stream {
    _private: (),
}

impl UnreviewedRmsNormQ10Stream {
    pub fn acknowledge_unreviewed_public_weights() -> Self {
        Self { _private: () }
    }
}

/// Exact body cost from execution of the shared topology with a counting backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RmsNormQ10StreamEstimate {
    pub and_gate_count: u64,
    pub evaluator_body_bytes: u64,
}

/// Public failures from streamed garbling, evaluation, and decoding.
#[derive(Debug)]
pub enum RmsNormQ10StreamError {
    InvalidWidth,
    InputCount,
    CrossCircuitInput,
    InvalidOutput,
    BodyLimitExceeded { required: u64, maximum: u64 },
    TrailingBodyBytes,
    Allocation(&'static str),
    Arithmetic(&'static str),
    Random,
    Io(std::io::Error),
}

impl fmt::Display for RmsNormQ10StreamError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidWidth => write!(formatter, "RMSNorm width must be between 1 and 8192"),
            Self::InputCount => write!(formatter, "RMSNorm input count is invalid"),
            Self::CrossCircuitInput => {
                write!(
                    formatter,
                    "RMSNorm input belongs to a different source identity"
                )
            }
            Self::InvalidOutput => write!(formatter, "RMSNorm output label is invalid"),
            Self::BodyLimitExceeded { required, maximum } => write!(
                formatter,
                "RMSNorm evaluator body requires {required} bytes, limit is {maximum}"
            ),
            Self::TrailingBodyBytes => {
                write!(formatter, "RMSNorm evaluator body has trailing bytes")
            }
            Self::Allocation(context) => write!(formatter, "{context} allocation failed"),
            Self::Arithmetic(context) => write!(formatter, "{context} arithmetic overflowed"),
            Self::Random => write!(formatter, "random generation failed"),
            Self::Io(error) => write!(formatter, "RMSNorm evaluator body I/O failed: {error}"),
        }
    }
}

impl Error for RmsNormQ10StreamError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for RmsNormQ10StreamError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

struct Encoding([u8; LABEL_BYTES]);

impl Clone for Encoding {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

impl Drop for Encoding {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

struct Label([u8; LABEL_BYTES]);

impl Clone for Label {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

impl Drop for Label {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

impl Label {
    fn selection_bit(&self) -> bool {
        self.0[LABEL_BYTES - 1] & 1 == 1
    }
}

/// Garbler-held input encodings and output decoders.
pub struct RmsNormQ10StreamClient {
    width: usize,
    source_identity: [u8; CIRCUIT_ID_BYTES],
    delta: [u8; LABEL_BYTES],
    input_encodings: Vec<Encoding>,
    output_encodings: Vec<Encoding>,
}

impl Drop for RmsNormQ10StreamClient {
    fn drop(&mut self) {
        self.delta.zeroize();
    }
}

/// Evaluator input labels, bound to one full source identity.
pub struct RmsNormQ10StreamInputs {
    source_identity: [u8; CIRCUIT_ID_BYTES],
    labels: Vec<Label>,
}

impl RmsNormQ10StreamInputs {
    pub fn source_identity(&self) -> [u8; CIRCUIT_ID_BYTES] {
        self.source_identity
    }
}

/// Evaluator output labels, bound to one full source identity.
pub struct RmsNormQ10StreamOutputs {
    source_identity: [u8; CIRCUIT_ID_BYTES],
    labels: Vec<Label>,
}

/// Consuming output decoder retained by garbler/client side.
pub struct RmsNormQ10StreamDecoder {
    width: usize,
    source_identity: [u8; CIRCUIT_ID_BYTES],
    delta: [u8; LABEL_BYTES],
    output_encodings: Vec<Encoding>,
}

impl Drop for RmsNormQ10StreamDecoder {
    fn drop(&mut self) {
        self.delta.zeroize();
    }
}

impl RmsNormQ10StreamClient {
    pub fn source_identity(&self) -> [u8; CIRCUIT_ID_BYTES] {
        self.source_identity
    }

    /// Consume client material, producing one evaluator input and one decoder.
    pub fn encode(
        mut self,
        input: &[i16],
    ) -> Result<(RmsNormQ10StreamInputs, RmsNormQ10StreamDecoder), RmsNormQ10StreamError> {
        if input.len() != self.width || self.input_encodings.len() != self.width * INPUT_BITS {
            return Err(RmsNormQ10StreamError::InputCount);
        }
        let mut labels = try_vec(self.input_encodings.len(), "RMSNorm input-label")?;
        for (encoding, value) in self
            .input_encodings
            .iter()
            .zip(input.iter().flat_map(|value| {
                let bits = u16::from_le_bytes(value.to_le_bytes());
                (0..INPUT_BITS).map(move |bit| bits & (1_u16 << bit) != 0)
            }))
        {
            labels.push(Label(if value {
                xor_block(encoding.0, self.delta)
            } else {
                encoding.0
            }));
        }
        let output_encodings = std::mem::take(&mut self.output_encodings);
        Ok((
            RmsNormQ10StreamInputs {
                source_identity: self.source_identity,
                labels,
            },
            RmsNormQ10StreamDecoder {
                width: self.width,
                source_identity: self.source_identity,
                delta: self.delta,
                output_encodings,
            },
        ))
    }
}

impl RmsNormQ10StreamDecoder {
    /// Consume decoder and evaluator outputs, rejecting any invalid label.
    pub fn decode(
        self,
        outputs: RmsNormQ10StreamOutputs,
    ) -> Result<Vec<i32>, RmsNormQ10StreamError> {
        if outputs.source_identity != self.source_identity {
            return Err(RmsNormQ10StreamError::CrossCircuitInput);
        }
        let expected_bits = self
            .width
            .checked_mul(OUTPUT_BITS)
            .ok_or(RmsNormQ10StreamError::Arithmetic("RMSNorm output width"))?;
        if outputs.labels.len() != expected_bits || self.output_encodings.len() != expected_bits {
            return Err(RmsNormQ10StreamError::InvalidOutput);
        }
        let mut decoded = Zeroizing::new(try_vec(self.width, "RMSNorm decoded output")?);
        for (labels, encodings) in outputs
            .labels
            .chunks_exact(OUTPUT_BITS)
            .zip(self.output_encodings.chunks_exact(OUTPUT_BITS))
        {
            let mut word = 0_u64;
            for (bit, (label, encoding)) in labels.iter().zip(encodings).enumerate() {
                let one = Zeroizing::new(xor_block(encoding.0, self.delta));
                if bool::from(label.0.ct_eq(&*one)) {
                    word |= 1_u64 << bit;
                } else if !bool::from(label.0.ct_eq(&encoding.0)) {
                    return Err(RmsNormQ10StreamError::InvalidOutput);
                }
            }
            let signed = i64::from_le_bytes(word.to_le_bytes());
            decoded.push(i32::try_from(signed).map_err(|_| RmsNormQ10StreamError::InvalidOutput)?);
        }
        Ok(std::mem::take(&mut *decoded))
    }
}

/// Execute shared topology with count-only wires; no cryptographic material is built.
pub fn estimate_rms_norm_q10_direct_stream(
    weights: &[i16],
) -> Result<RmsNormQ10StreamEstimate, RmsNormQ10StreamError> {
    validate_width(weights.len())?;
    let mut backend = CountBackend { next_gate: 0 };
    let outputs = rms_norm_topology(&mut backend, weights)?;
    drop(outputs);
    let evaluator_body_bytes = backend
        .next_gate
        .checked_mul(BYTES_PER_AND)
        .ok_or(RmsNormQ10StreamError::Arithmetic("RMSNorm body size"))?;
    Ok(RmsNormQ10StreamEstimate {
        and_gate_count: backend.next_gate,
        evaluator_body_bytes,
    })
}

/// Stream raw half-gate ciphertexts and return only client-side material.
///
/// `source_identity` must uniquely identify the complete compiled source and
/// issuance. `max_body_bytes` is checked using exact topology execution before
/// randomness, secret encodings, or body bytes are constructed.
pub fn garble_rms_norm_q10_direct_stream<W: Write>(
    _approval: &UnreviewedRmsNormQ10Stream,
    weights: &[i16],
    source_identity: [u8; CIRCUIT_ID_BYTES],
    max_body_bytes: u64,
    writer: W,
) -> Result<RmsNormQ10StreamClient, RmsNormQ10StreamError> {
    let estimate = admit(weights, max_body_bytes)?;
    let input_count = weights
        .len()
        .checked_mul(INPUT_BITS)
        .ok_or(RmsNormQ10StreamError::Arithmetic("RMSNorm input count"))?;
    let mut delta = Zeroizing::new([0_u8; LABEL_BYTES]);
    getrandom::fill(delta.as_mut()).map_err(|_| RmsNormQ10StreamError::Random)?;
    delta[LABEL_BYTES - 1] |= 1;
    let mut backend = GarbleBackend {
        writer,
        source_identity,
        delta: *delta,
        next_gate: 0,
        input_encodings: try_vec(input_count, "RMSNorm input-encoding")?,
    };
    let outputs = rms_norm_topology(&mut backend, weights)?;
    if backend.next_gate != estimate.and_gate_count {
        return Err(RmsNormQ10StreamError::Arithmetic("RMSNorm topology count"));
    }
    let mut output_encodings = try_vec(outputs.len(), "RMSNorm output-decoder")?;
    for output in outputs {
        match output {
            Wire::Secret(encoding) => output_encodings.push(encoding),
            Wire::Public(_) => {
                return Err(RmsNormQ10StreamError::Arithmetic(
                    "RMSNorm unexpectedly public output",
                ));
            }
        }
    }
    Ok(RmsNormQ10StreamClient {
        width: weights.len(),
        source_identity,
        delta: backend.delta,
        input_encodings: std::mem::take(&mut backend.input_encodings),
        output_encodings,
    })
}

/// Consume bound input labels and a finite raw ciphertext body.
///
/// Public weights and identity must exactly match garbling. EOF is required
/// immediately after the exact body byte count.
pub fn evaluate_rms_norm_q10_direct_stream<R: Read>(
    _approval: &UnreviewedRmsNormQ10Stream,
    weights: &[i16],
    source_identity: [u8; CIRCUIT_ID_BYTES],
    max_body_bytes: u64,
    inputs: RmsNormQ10StreamInputs,
    reader: R,
) -> Result<RmsNormQ10StreamOutputs, RmsNormQ10StreamError> {
    validate_width(weights.len())?;
    if inputs.source_identity != source_identity {
        return Err(RmsNormQ10StreamError::CrossCircuitInput);
    }
    let expected_inputs = weights
        .len()
        .checked_mul(INPUT_BITS)
        .ok_or(RmsNormQ10StreamError::Arithmetic("RMSNorm input count"))?;
    if inputs.labels.len() != expected_inputs {
        return Err(RmsNormQ10StreamError::InputCount);
    }
    let estimate = admit(weights, max_body_bytes)?;
    let mut backend = EvaluateBackend {
        reader,
        source_identity,
        next_gate: 0,
        inputs: inputs.labels.into_iter(),
    };
    let wires = rms_norm_topology(&mut backend, weights)?;
    if backend.next_gate != estimate.and_gate_count || backend.inputs.next().is_some() {
        return Err(RmsNormQ10StreamError::Arithmetic("RMSNorm topology count"));
    }
    ensure_eof(&mut backend.reader)?;
    let mut labels = try_vec(wires.len(), "RMSNorm evaluator output")?;
    for wire in wires {
        match wire {
            Wire::Secret(label) => labels.push(label),
            Wire::Public(_) => {
                return Err(RmsNormQ10StreamError::Arithmetic(
                    "RMSNorm unexpectedly public output",
                ));
            }
        }
    }
    Ok(RmsNormQ10StreamOutputs {
        source_identity,
        labels,
    })
}

fn admit(
    weights: &[i16],
    max_body_bytes: u64,
) -> Result<RmsNormQ10StreamEstimate, RmsNormQ10StreamError> {
    let estimate = estimate_rms_norm_q10_direct_stream(weights)?;
    if estimate.evaluator_body_bytes > max_body_bytes {
        return Err(RmsNormQ10StreamError::BodyLimitExceeded {
            required: estimate.evaluator_body_bytes,
            maximum: max_body_bytes,
        });
    }
    Ok(estimate)
}

fn validate_width(width: usize) -> Result<(), RmsNormQ10StreamError> {
    if !(1..=MAX_WIDTH).contains(&width) {
        return Err(RmsNormQ10StreamError::InvalidWidth);
    }
    Ok(())
}

fn try_vec<T>(capacity: usize, context: &'static str) -> Result<Vec<T>, RmsNormQ10StreamError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| RmsNormQ10StreamError::Allocation(context))?;
    Ok(values)
}

fn read_exact_retry<R: Read>(
    reader: &mut R,
    mut buffer: &mut [u8],
) -> Result<(), RmsNormQ10StreamError> {
    while !buffer.is_empty() {
        match reader.read(buffer) {
            Ok(0) => {
                return Err(RmsNormQ10StreamError::Io(
                    std::io::ErrorKind::UnexpectedEof.into(),
                ));
            }
            Ok(read) => buffer = &mut buffer[read..],
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(RmsNormQ10StreamError::Io(error)),
        }
    }
    Ok(())
}

fn ensure_eof<R: Read>(reader: &mut R) -> Result<(), RmsNormQ10StreamError> {
    let mut trailing = [0_u8; 1];
    loop {
        match reader.read(&mut trailing) {
            Ok(0) => return Ok(()),
            Ok(_) => return Err(RmsNormQ10StreamError::TrailingBodyBytes),
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error.into()),
        }
    }
}

fn filled<T: Clone>(
    value: T,
    length: usize,
    context: &'static str,
) -> Result<Vec<T>, RmsNormQ10StreamError> {
    let mut values = try_vec(length, context)?;
    values.resize(length, value);
    Ok(values)
}

fn cloned<T: Clone>(values: &[T], context: &'static str) -> Result<Vec<T>, RmsNormQ10StreamError> {
    let mut output = try_vec(values.len(), context)?;
    output.extend_from_slice(values);
    Ok(output)
}

enum Wire<S> {
    Public(bool),
    Secret(S),
}

type WirePairResult<S> = Result<(Wire<S>, Wire<S>), RmsNormQ10StreamError>;
type WordAndWireResult<S> = Result<(Vec<Wire<S>>, Wire<S>), RmsNormQ10StreamError>;

impl<S: Clone> Clone for Wire<S> {
    fn clone(&self) -> Self {
        match self {
            Self::Public(value) => Self::Public(*value),
            Self::Secret(secret) => Self::Secret(secret.clone()),
        }
    }
}

trait Backend {
    type Secret: Clone;

    fn input(&mut self) -> Result<Self::Secret, RmsNormQ10StreamError>;
    fn xor(
        &mut self,
        left: &Self::Secret,
        right: &Self::Secret,
    ) -> Result<Self::Secret, RmsNormQ10StreamError>;
    fn not(&mut self, input: &Self::Secret) -> Result<Self::Secret, RmsNormQ10StreamError>;
    fn and(
        &mut self,
        left: &Self::Secret,
        right: &Self::Secret,
    ) -> Result<Self::Secret, RmsNormQ10StreamError>;
}

#[derive(Clone, Copy)]
struct CountSecret;

struct CountBackend {
    next_gate: u64,
}

impl Backend for CountBackend {
    type Secret = CountSecret;

    fn input(&mut self) -> Result<Self::Secret, RmsNormQ10StreamError> {
        Ok(CountSecret)
    }

    fn xor(
        &mut self,
        _left: &Self::Secret,
        _right: &Self::Secret,
    ) -> Result<Self::Secret, RmsNormQ10StreamError> {
        Ok(CountSecret)
    }

    fn not(&mut self, _input: &Self::Secret) -> Result<Self::Secret, RmsNormQ10StreamError> {
        Ok(CountSecret)
    }

    fn and(
        &mut self,
        _left: &Self::Secret,
        _right: &Self::Secret,
    ) -> Result<Self::Secret, RmsNormQ10StreamError> {
        self.next_gate = self
            .next_gate
            .checked_add(1)
            .ok_or(RmsNormQ10StreamError::Arithmetic("RMSNorm gate index"))?;
        Ok(CountSecret)
    }
}

struct GarbleBackend<W> {
    writer: W,
    source_identity: [u8; CIRCUIT_ID_BYTES],
    delta: [u8; LABEL_BYTES],
    next_gate: u64,
    input_encodings: Vec<Encoding>,
}

impl<W> Drop for GarbleBackend<W> {
    fn drop(&mut self) {
        self.delta.zeroize();
    }
}

impl<W: Write> Backend for GarbleBackend<W> {
    type Secret = Encoding;

    fn input(&mut self) -> Result<Self::Secret, RmsNormQ10StreamError> {
        let mut zero = Zeroizing::new([0_u8; LABEL_BYTES]);
        getrandom::fill(zero.as_mut()).map_err(|_| RmsNormQ10StreamError::Random)?;
        let encoding = Encoding(*zero);
        self.input_encodings.push(encoding.clone());
        Ok(encoding)
    }

    fn xor(
        &mut self,
        left: &Self::Secret,
        right: &Self::Secret,
    ) -> Result<Self::Secret, RmsNormQ10StreamError> {
        Ok(Encoding(xor_block(left.0, right.0)))
    }

    fn not(&mut self, input: &Self::Secret) -> Result<Self::Secret, RmsNormQ10StreamError> {
        Ok(Encoding(xor_block(input.0, self.delta)))
    }

    fn and(
        &mut self,
        left: &Self::Secret,
        right: &Self::Secret,
    ) -> Result<Self::Secret, RmsNormQ10StreamError> {
        let gate_index = self.next_gate;
        self.next_gate = self
            .next_gate
            .checked_add(1)
            .ok_or(RmsNormQ10StreamError::Arithmetic("RMSNorm gate index"))?;

        let left_one = Zeroizing::new(xor_block(left.0, self.delta));
        let right_one = Zeroizing::new(xor_block(right.0, self.delta));
        let left_permute = left.0[LABEL_BYTES - 1] & 1 == 1;
        let right_permute = right.0[LABEL_BYTES - 1] & 1 == 1;
        let left_zero_hash = Zeroizing::new(fixed_key_hash(
            &left.0,
            &self.source_identity,
            gate_index,
            0,
        ));
        let left_one_hash = Zeroizing::new(fixed_key_hash(
            &left_one,
            &self.source_identity,
            gate_index,
            0,
        ));
        let right_zero_hash = Zeroizing::new(fixed_key_hash(
            &right.0,
            &self.source_identity,
            gate_index,
            1,
        ));
        let right_one_hash = Zeroizing::new(fixed_key_hash(
            &right_one,
            &self.source_identity,
            gate_index,
            1,
        ));

        let mut generator_table = Zeroizing::new(xor_block(*left_zero_hash, *left_one_hash));
        if right_permute {
            *generator_table = xor_block(*generator_table, self.delta);
        }
        let generator_zero = Zeroizing::new(if left_permute {
            xor_block(*left_zero_hash, *generator_table)
        } else {
            *left_zero_hash
        });
        let evaluator_table = Zeroizing::new(xor_block(
            xor_block(*right_zero_hash, *right_one_hash),
            left.0,
        ));
        let evaluator_zero = Zeroizing::new(if right_permute {
            xor_block(*right_zero_hash, xor_block(*evaluator_table, left.0))
        } else {
            *right_zero_hash
        });
        let output = Encoding(xor_block(*generator_zero, *evaluator_zero));
        self.writer.write_all(generator_table.as_ref())?;
        self.writer.write_all(evaluator_table.as_ref())?;
        Ok(output)
    }
}

struct EvaluateBackend<R> {
    reader: R,
    source_identity: [u8; CIRCUIT_ID_BYTES],
    next_gate: u64,
    inputs: std::vec::IntoIter<Label>,
}

impl<R: Read> Backend for EvaluateBackend<R> {
    type Secret = Label;

    fn input(&mut self) -> Result<Self::Secret, RmsNormQ10StreamError> {
        self.inputs.next().ok_or(RmsNormQ10StreamError::InputCount)
    }

    fn xor(
        &mut self,
        left: &Self::Secret,
        right: &Self::Secret,
    ) -> Result<Self::Secret, RmsNormQ10StreamError> {
        Ok(Label(xor_block(left.0, right.0)))
    }

    fn not(&mut self, input: &Self::Secret) -> Result<Self::Secret, RmsNormQ10StreamError> {
        Ok(input.clone())
    }

    fn and(
        &mut self,
        left: &Self::Secret,
        right: &Self::Secret,
    ) -> Result<Self::Secret, RmsNormQ10StreamError> {
        let gate_index = self.next_gate;
        self.next_gate = self
            .next_gate
            .checked_add(1)
            .ok_or(RmsNormQ10StreamError::Arithmetic("RMSNorm gate index"))?;
        let mut generator_table = Zeroizing::new([0_u8; LABEL_BYTES]);
        let mut evaluator_table = Zeroizing::new([0_u8; LABEL_BYTES]);
        read_exact_retry(&mut self.reader, generator_table.as_mut())?;
        read_exact_retry(&mut self.reader, evaluator_table.as_mut())?;

        let left_hash = Zeroizing::new(fixed_key_hash(
            &left.0,
            &self.source_identity,
            gate_index,
            0,
        ));
        let right_hash = Zeroizing::new(fixed_key_hash(
            &right.0,
            &self.source_identity,
            gate_index,
            1,
        ));
        let generator = Zeroizing::new(if left.selection_bit() {
            xor_block(*left_hash, *generator_table)
        } else {
            *left_hash
        });
        let evaluator = Zeroizing::new(if right.selection_bit() {
            xor_block(*right_hash, xor_block(*evaluator_table, left.0))
        } else {
            *right_hash
        });
        Ok(Label(xor_block(*generator, *evaluator)))
    }
}

fn xor<B: Backend>(
    backend: &mut B,
    left: &Wire<B::Secret>,
    right: &Wire<B::Secret>,
) -> Result<Wire<B::Secret>, RmsNormQ10StreamError> {
    match (left, right) {
        (Wire::Public(left), Wire::Public(right)) => Ok(Wire::Public(left ^ right)),
        (Wire::Public(false), wire) | (wire, Wire::Public(false)) => Ok(wire.clone()),
        (Wire::Public(true), wire) | (wire, Wire::Public(true)) => not(backend, wire),
        (Wire::Secret(left), Wire::Secret(right)) => Ok(Wire::Secret(backend.xor(left, right)?)),
    }
}

fn not<B: Backend>(
    backend: &mut B,
    input: &Wire<B::Secret>,
) -> Result<Wire<B::Secret>, RmsNormQ10StreamError> {
    match input {
        Wire::Public(value) => Ok(Wire::Public(!value)),
        Wire::Secret(input) => Ok(Wire::Secret(backend.not(input)?)),
    }
}

fn and<B: Backend>(
    backend: &mut B,
    left: &Wire<B::Secret>,
    right: &Wire<B::Secret>,
) -> Result<Wire<B::Secret>, RmsNormQ10StreamError> {
    match (left, right) {
        (Wire::Public(false), _) | (_, Wire::Public(false)) => Ok(Wire::Public(false)),
        (Wire::Public(true), wire) | (wire, Wire::Public(true)) => Ok(wire.clone()),
        (Wire::Secret(left), Wire::Secret(right)) => Ok(Wire::Secret(backend.and(left, right)?)),
    }
}

fn input_word<B: Backend>(
    backend: &mut B,
    width: usize,
) -> Result<Vec<Wire<B::Secret>>, RmsNormQ10StreamError> {
    let mut output = try_vec(width, "RMSNorm input word")?;
    for _ in 0..width {
        output.push(Wire::Secret(backend.input()?));
    }
    Ok(output)
}

fn constant_word<S>(value: u128, width: usize) -> Result<Vec<Wire<S>>, RmsNormQ10StreamError> {
    let mut output = try_vec(width, "RMSNorm constant word")?;
    for shift in 0..width {
        output.push(Wire::Public(value & (1_u128 << shift) != 0));
    }
    Ok(output)
}

fn add_unsigned<B: Backend>(
    backend: &mut B,
    left: &[Wire<B::Secret>],
    right: &[Wire<B::Secret>],
) -> Result<Vec<Wire<B::Secret>>, RmsNormQ10StreamError> {
    let mut carry = Wire::Public(false);
    let mut sum = try_vec(left.len(), "RMSNorm adder output")?;
    for (left_bit, right_bit) in left.iter().zip(right) {
        let either = xor(backend, left_bit, right_bit)?;
        sum.push(xor(backend, &either, &carry)?);
        let both = and(backend, left_bit, right_bit)?;
        let carry_either = and(backend, &carry, &either)?;
        carry = xor(backend, &both, &carry_either)?;
    }
    Ok(sum)
}

fn multiply_unsigned<B: Backend>(
    backend: &mut B,
    left: &[Wire<B::Secret>],
    right: &[Wire<B::Secret>],
) -> Result<Vec<Wire<B::Secret>>, RmsNormQ10StreamError> {
    multiply_unsigned_wide(backend, left, right)
}

fn multiply_unsigned_wide<B: Backend>(
    backend: &mut B,
    left: &[Wire<B::Secret>],
    right: &[Wire<B::Secret>],
) -> Result<Vec<Wire<B::Secret>>, RmsNormQ10StreamError> {
    let width = left
        .len()
        .checked_add(right.len())
        .ok_or(RmsNormQ10StreamError::Arithmetic("RMSNorm product width"))?;
    let zero = Wire::Public(false);
    let mut product = filled(zero.clone(), width, "RMSNorm product")?;
    for (shift, right_bit) in right.iter().enumerate() {
        let mut partial = filled(zero.clone(), width, "RMSNorm partial product")?;
        for (index, left_bit) in left.iter().enumerate() {
            partial[index + shift] = and(backend, left_bit, right_bit)?;
        }
        product = add_unsigned(backend, &product, &partial)?;
    }
    Ok(product)
}

fn multiply_by_public<B: Backend>(
    backend: &mut B,
    input: &[Wire<B::Secret>],
    factor: u128,
    output_width: usize,
) -> Result<Vec<Wire<B::Secret>>, RmsNormQ10StreamError> {
    let zero = Wire::Public(false);
    let mut product = filled(zero.clone(), output_width, "RMSNorm public product")?;
    for shift in 0..128 {
        if factor & (1_u128 << shift) == 0 {
            continue;
        }
        let mut partial = filled(zero.clone(), output_width, "RMSNorm public partial")?;
        for (index, bit) in input.iter().enumerate() {
            if index + shift < output_width {
                partial[index + shift] = bit.clone();
            }
        }
        product = add_unsigned(backend, &product, &partial)?;
    }
    Ok(product)
}

fn compare_unsigned_to_constant<B: Backend>(
    backend: &mut B,
    value: &[Wire<B::Secret>],
    constant: u128,
) -> WirePairResult<B::Secret> {
    let mut less = Wire::Public(false);
    let mut equal = Wire::Public(true);
    for index in (0..value.len()).rev() {
        if constant & (1_u128 << index) != 0 {
            let not_value = not(backend, &value[index])?;
            let newly_less = and(backend, &equal, &not_value)?;
            less = xor(backend, &less, &newly_less)?;
            equal = and(backend, &equal, &value[index])?;
        } else {
            let not_value = not(backend, &value[index])?;
            equal = and(backend, &equal, &not_value)?;
        }
    }
    Ok((less, equal))
}

fn compare_unsigned<B: Backend>(
    backend: &mut B,
    left: &[Wire<B::Secret>],
    right: &[Wire<B::Secret>],
) -> WirePairResult<B::Secret> {
    let mut less = Wire::Public(false);
    let mut equal = Wire::Public(true);
    for (left_bit, right_bit) in left.iter().zip(right).rev() {
        let not_left = not(backend, left_bit)?;
        let left_less = and(backend, &not_left, right_bit)?;
        let newly_less = and(backend, &equal, &left_less)?;
        less = xor(backend, &less, &newly_less)?;
        let different = xor(backend, left_bit, right_bit)?;
        let same = not(backend, &different)?;
        equal = and(backend, &equal, &same)?;
    }
    Ok((less, equal))
}

fn subtract_unsigned<B: Backend>(
    backend: &mut B,
    left: &[Wire<B::Secret>],
    right: &[Wire<B::Secret>],
) -> WordAndWireResult<B::Secret> {
    let mut borrow = Wire::Public(false);
    let mut difference = try_vec(left.len(), "RMSNorm subtraction")?;
    for (left_bit, right_bit) in left.iter().zip(right) {
        let either = xor(backend, left_bit, right_bit)?;
        difference.push(xor(backend, &either, &borrow)?);
        let not_left = not(backend, left_bit)?;
        let direct_borrow = and(backend, &not_left, right_bit)?;
        let same = not(backend, &either)?;
        let carried_borrow = and(backend, &borrow, &same)?;
        borrow = xor(backend, &direct_borrow, &carried_borrow)?;
    }
    Ok((difference, borrow))
}

fn select_word<B: Backend>(
    backend: &mut B,
    condition: &Wire<B::Secret>,
    when_false: &[Wire<B::Secret>],
    when_true: &[Wire<B::Secret>],
) -> Result<Vec<Wire<B::Secret>>, RmsNormQ10StreamError> {
    let mut output = try_vec(when_false.len(), "RMSNorm selection")?;
    for (false_bit, true_bit) in when_false.iter().zip(when_true) {
        let difference = xor(backend, false_bit, true_bit)?;
        let selected = and(backend, condition, &difference)?;
        output.push(xor(backend, false_bit, &selected)?);
    }
    Ok(output)
}

fn divide_public_by_secret<B: Backend>(
    backend: &mut B,
    numerator: u128,
    denominator: &[Wire<B::Secret>],
) -> Result<Vec<Wire<B::Secret>>, RmsNormQ10StreamError> {
    let numerator_width = usize::try_from(128 - numerator.leading_zeros())
        .map_err(|_| RmsNormQ10StreamError::Arithmetic("RMSNorm numerator width"))?;
    let zero = Wire::Public(false);
    let mut extended_denominator = try_vec(
        denominator
            .len()
            .checked_add(1)
            .ok_or(RmsNormQ10StreamError::Arithmetic(
                "RMSNorm extended denominator width",
            ))?,
        "RMSNorm extended denominator",
    )?;
    extended_denominator.extend_from_slice(denominator);
    extended_denominator.push(zero.clone());
    let mut remainder = filled(
        zero.clone(),
        extended_denominator.len(),
        "RMSNorm division remainder",
    )?;
    let mut quotient = filled(zero, numerator_width, "RMSNorm quotient")?;
    for bit in (0..numerator_width).rev() {
        let mut shifted = try_vec(remainder.len(), "RMSNorm shifted remainder")?;
        shifted.push(Wire::Public(numerator & (1_u128 << bit) != 0));
        shifted.extend(remainder.iter().take(remainder.len() - 1).cloned());
        let (difference, borrow) = subtract_unsigned(backend, &shifted, &extended_denominator)?;
        let greater_or_equal = not(backend, &borrow)?;
        remainder = select_word(backend, &greater_or_equal, &shifted, &difference)?;
        quotient[bit] = greater_or_equal;
    }
    Ok(quotient)
}

fn reciprocal_sqrt_q<B: Backend>(
    backend: &mut B,
    sum_squares: &[Wire<B::Secret>],
    row_width: u32,
) -> Result<Vec<Wire<B::Secret>>, RmsNormQ10StreamError> {
    let numerator = u128::from(row_width)
        .checked_mul(1_000_000)
        .and_then(|value| value.checked_shl(2 * FRACTIONAL_BITS))
        .ok_or(RmsNormQ10StreamError::Arithmetic(
            "RMSNorm reciprocal numerator",
        ))?;
    let epsilon = u128::from(row_width)
        .checked_mul(1_048_576)
        .ok_or(RmsNormQ10StreamError::Arithmetic("RMSNorm epsilon"))?;
    let scaled_sum = multiply_by_public(backend, sum_squares, 1_000_000, 64)?;
    let epsilon_word = constant_word(epsilon, 64)?;
    let denominator = add_unsigned(backend, &scaled_sum, &epsilon_word)?;
    let quotient = divide_public_by_secret(backend, numerator, &denominator)?;
    let zero = Wire::Public(false);
    let one = Wire::Public(true);
    let output_width = FRACTIONAL_BITS as usize;
    let mut result = filled(zero.clone(), output_width, "RMSNorm reciprocal result")?;
    for bit in (0..output_width).rev() {
        let mut trial = cloned(&result, "RMSNorm reciprocal trial")?;
        trial[bit] = one.clone();
        let square = multiply_unsigned(backend, &trial, &trial)?;
        let mut extended_square = square;
        if extended_square.len() < quotient.len() {
            extended_square
                .try_reserve_exact(quotient.len() - extended_square.len())
                .map_err(|_| RmsNormQ10StreamError::Allocation("RMSNorm extended square"))?;
            extended_square.resize(quotient.len(), zero.clone());
        }
        let (less, equal) = compare_unsigned(backend, &extended_square, &quotient)?;
        result[bit] = xor(backend, &less, &equal)?;
    }

    let mut twice_plus_one = try_vec(output_width + 1, "RMSNorm rounding root")?;
    twice_plus_one.push(one);
    twice_plus_one.extend(result.iter().cloned());
    let threshold_square = multiply_unsigned(backend, &twice_plus_one, &twice_plus_one)?;
    let threshold = multiply_unsigned_wide(backend, &threshold_square, &denominator)?;
    let four_numerator = numerator
        .checked_mul(4)
        .ok_or(RmsNormQ10StreamError::Arithmetic(
            "RMSNorm rounding threshold",
        ))?;
    let (below, equal) = compare_unsigned_to_constant(backend, &threshold, four_numerator)?;
    let equal_and_odd = and(backend, &equal, &result[0])?;
    let mut carry = xor(backend, &below, &equal_and_odd)?;
    let mut rounded = try_vec(output_width, "RMSNorm rounded reciprocal")?;
    for bit in result {
        rounded.push(xor(backend, &bit, &carry)?);
        carry = and(backend, &bit, &carry)?;
    }
    Ok(rounded)
}

fn absolute_signed<B: Backend>(
    backend: &mut B,
    input: &[Wire<B::Secret>],
) -> Result<Vec<Wire<B::Secret>>, RmsNormQ10StreamError> {
    let sign = &input[input.len() - 1];
    let mut inverted = try_vec(input.len(), "RMSNorm absolute value")?;
    for bit in input {
        inverted.push(xor(backend, bit, sign)?);
    }
    let mut carry = sign.clone();
    let mut magnitude = try_vec(input.len(), "RMSNorm magnitude")?;
    for bit in inverted {
        magnitude.push(xor(backend, &bit, &carry)?);
        carry = and(backend, &bit, &carry)?;
    }
    Ok(magnitude)
}

fn square_signed<B: Backend>(
    backend: &mut B,
    input: &[Wire<B::Secret>],
) -> Result<Vec<Wire<B::Secret>>, RmsNormQ10StreamError> {
    let magnitude = absolute_signed(backend, input)?;
    multiply_unsigned(backend, &magnitude, &magnitude)
}

fn conditional_negate<B: Backend>(
    backend: &mut B,
    magnitude: &[Wire<B::Secret>],
    negate: &Wire<B::Secret>,
) -> Result<Vec<Wire<B::Secret>>, RmsNormQ10StreamError> {
    let mut carry = negate.clone();
    let mut output = try_vec(magnitude.len(), "RMSNorm conditional negate")?;
    for bit in magnitude {
        let inverted = xor(backend, bit, negate)?;
        output.push(xor(backend, &inverted, &carry)?);
        carry = and(backend, &inverted, &carry)?;
    }
    Ok(output)
}

fn round_shift_ties_even<B: Backend>(
    backend: &mut B,
    magnitude: &[Wire<B::Secret>],
    shift: usize,
    output_width: usize,
) -> Result<Vec<Wire<B::Secret>>, RmsNormQ10StreamError> {
    let remainder = &magnitude[..shift];
    let half = 1_u128 << (shift - 1);
    let (below_half, equal_half) = compare_unsigned_to_constant(backend, remainder, half)?;
    let below_or_equal = xor(backend, &below_half, &equal_half)?;
    let above_half = not(backend, &below_or_equal)?;
    let tie_and_odd = and(backend, &equal_half, &magnitude[shift])?;
    let mut carry = xor(backend, &above_half, &tie_and_odd)?;
    let zero = Wire::Public(false);
    let mut output = try_vec(output_width, "RMSNorm rounded output")?;
    for index in 0..output_width {
        let bit = magnitude.get(index + shift).unwrap_or(&zero);
        output.push(xor(backend, bit, &carry)?);
        carry = and(backend, bit, &carry)?;
    }
    Ok(output)
}

fn rms_norm_topology<B: Backend>(
    backend: &mut B,
    weights: &[i16],
) -> Result<Vec<Wire<B::Secret>>, RmsNormQ10StreamError> {
    validate_width(weights.len())?;
    let mut rows = try_vec(weights.len(), "RMSNorm rows")?;
    for _ in weights {
        rows.push(input_word(backend, INPUT_BITS)?);
    }

    let zero = Wire::Public(false);
    let mut sum_squares = filled(zero.clone(), SUM_BITS, "RMSNorm sum of squares")?;
    for value in &rows {
        let mut square = square_signed(backend, value)?;
        square
            .try_reserve_exact(SUM_BITS - square.len())
            .map_err(|_| RmsNormQ10StreamError::Allocation("RMSNorm extended square"))?;
        square.resize(SUM_BITS, zero.clone());
        sum_squares = add_unsigned(backend, &sum_squares, &square)?;
    }
    let row_width = u32::try_from(rows.len())
        .map_err(|_| RmsNormQ10StreamError::Arithmetic("RMSNorm row width"))?;
    let normalizer = reciprocal_sqrt_q(backend, &sum_squares, row_width)?;
    let output_capacity = rows
        .len()
        .checked_mul(OUTPUT_BITS)
        .ok_or(RmsNormQ10StreamError::Arithmetic("RMSNorm output count"))?;
    let mut outputs = try_vec(output_capacity, "RMSNorm outputs")?;
    for (input, weight) in rows.iter().zip(weights) {
        let magnitude = absolute_signed(backend, input)?;
        let weighted = multiply_by_public(
            backend,
            &magnitude,
            u128::from(weight.unsigned_abs()),
            INPUT_BITS * 2,
        )?;
        let scaled = multiply_unsigned_wide(backend, &weighted, &normalizer)?;
        let rounded =
            round_shift_ties_even(backend, &scaled, FRACTIONAL_BITS as usize, OUTPUT_BITS)?;
        let sign = if weight.is_negative() {
            not(backend, &input[INPUT_BITS - 1])?
        } else {
            input[INPUT_BITS - 1].clone()
        };
        outputs.extend(conditional_negate(backend, &rounded, &sign)?);
    }
    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const ID: [u8; CIRCUIT_ID_BYTES] = [0x5a; CIRCUIT_ID_BYTES];

    fn approval() -> UnreviewedRmsNormQ10Stream {
        UnreviewedRmsNormQ10Stream::acknowledge_unreviewed_public_weights()
    }

    fn round_trip(input: &[i16], weights: &[i16]) -> (Vec<i32>, usize, u64) {
        let estimate = estimate_rms_norm_q10_direct_stream(weights).unwrap();
        let mut body = Vec::new();
        let client = garble_rms_norm_q10_direct_stream(
            &approval(),
            weights,
            ID,
            estimate.evaluator_body_bytes,
            &mut body,
        )
        .unwrap();
        let (inputs, decoder) = client.encode(input).unwrap();
        let outputs = evaluate_rms_norm_q10_direct_stream(
            &approval(),
            weights,
            ID,
            estimate.evaluator_body_bytes,
            inputs,
            Cursor::new(&body),
        )
        .unwrap();
        (
            decoder.decode(outputs).unwrap(),
            body.len(),
            estimate.and_gate_count,
        )
    }

    #[test]
    fn exact_small_rows_match_core_and_body_count() {
        for (input, weights, expected_gates) in [
            (vec![0_i16], vec![1024_i16], 58_695),
            (vec![1536, -768], vec![1024, 896], 61_827),
            (vec![i16::MIN, i16::MAX, -1], vec![-1024, 511, 0], 63_932),
        ] {
            let expected = pllm_core::rms_norm_q10_direct(&input, &weights).unwrap();
            let estimate = estimate_rms_norm_q10_direct_stream(&weights).unwrap();
            let (actual, bytes, gates) = round_trip(&input, &weights);
            assert_eq!(actual, expected);
            assert_eq!(gates, expected_gates);
            assert_eq!(u64::try_from(bytes).unwrap(), estimate.evaluator_body_bytes);
            assert_eq!(estimate.evaluator_body_bytes, gates * BYTES_PER_AND);
        }
    }

    #[test]
    fn rejects_truncation_trailing_and_corruption() {
        let input = [1536_i16];
        let weights = [1024_i16];
        let estimate = estimate_rms_norm_q10_direct_stream(&weights).unwrap();
        let mut body = Vec::new();
        let client = garble_rms_norm_q10_direct_stream(
            &approval(),
            &weights,
            ID,
            estimate.evaluator_body_bytes,
            &mut body,
        )
        .unwrap();
        let (inputs, _) = client.encode(&input).unwrap();
        body.pop();
        assert!(evaluate_rms_norm_q10_direct_stream(
            &approval(),
            &weights,
            ID,
            estimate.evaluator_body_bytes,
            inputs,
            Cursor::new(&body),
        )
        .is_err());

        let mut body = Vec::new();
        let client = garble_rms_norm_q10_direct_stream(
            &approval(),
            &weights,
            ID,
            estimate.evaluator_body_bytes,
            &mut body,
        )
        .unwrap();
        let (inputs, _) = client.encode(&input).unwrap();
        body.push(0);
        assert!(matches!(
            evaluate_rms_norm_q10_direct_stream(
                &approval(),
                &weights,
                ID,
                estimate.evaluator_body_bytes,
                inputs,
                Cursor::new(&body),
            ),
            Err(RmsNormQ10StreamError::TrailingBodyBytes)
        ));

        let mut body = Vec::new();
        let client = garble_rms_norm_q10_direct_stream(
            &approval(),
            &weights,
            ID,
            estimate.evaluator_body_bytes,
            &mut body,
        )
        .unwrap();
        let (inputs, decoder) = client.encode(&input).unwrap();
        for byte in &mut body {
            *byte ^= 0x80;
        }
        let outputs = evaluate_rms_norm_q10_direct_stream(
            &approval(),
            &weights,
            ID,
            estimate.evaluator_body_bytes,
            inputs,
            Cursor::new(&body),
        )
        .unwrap();
        assert!(decoder.decode(outputs).is_err());
    }

    #[test]
    fn rejects_cross_circuit_inputs_before_reading() {
        let weights = [1024_i16];
        let estimate = estimate_rms_norm_q10_direct_stream(&weights).unwrap();
        let mut body = Vec::new();
        let client = garble_rms_norm_q10_direct_stream(
            &approval(),
            &weights,
            ID,
            estimate.evaluator_body_bytes,
            &mut body,
        )
        .unwrap();
        let (inputs, _) = client.encode(&[1]).unwrap();
        assert!(matches!(
            evaluate_rms_norm_q10_direct_stream(
                &approval(),
                &weights,
                [0x33; CIRCUIT_ID_BYTES],
                estimate.evaluator_body_bytes,
                inputs,
                Cursor::new(Vec::<u8>::new()),
            ),
            Err(RmsNormQ10StreamError::CrossCircuitInput)
        ));
    }

    #[test]
    fn body_limit_rejects_before_writing() {
        let weights = [1024_i16];
        let estimate = estimate_rms_norm_q10_direct_stream(&weights).unwrap();
        let mut body = Vec::new();
        assert!(matches!(
            garble_rms_norm_q10_direct_stream(
                &approval(),
                &weights,
                ID,
                estimate.evaluator_body_bytes - 1,
                &mut body,
            ),
            Err(RmsNormQ10StreamError::BodyLimitExceeded { .. })
        ));
        assert!(body.is_empty());
    }

    #[test]
    fn maximum_width_count_only_executes_shared_topology() {
        let weights = vec![1024_i16; MAX_WIDTH];
        let estimate = estimate_rms_norm_q10_direct_stream(&weights).unwrap();
        assert_eq!(estimate.and_gate_count, 19_026_882);
        assert_eq!(estimate.evaluator_body_bytes, 608_860_224);
        assert_eq!(estimate.evaluator_body_bytes, estimate.and_gate_count * 32);
    }
}
