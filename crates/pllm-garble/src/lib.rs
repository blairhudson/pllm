//! Clean-room reference arithmetic garbling based on PLLM's cited public
//! protocol descriptions. This unreviewed research crate is disabled in all
//! production profiles and makes no cryptographic assurance claim.

use sha2::{Digest as _, Sha256};
use std::{collections::BTreeMap, error::Error, fmt};
use zeroize::Zeroize;

const SECURITY_BITS: f64 = 128.0;
const MAX_MODULUS: u16 = 512;
const TAG_BYTES: usize = 16;
const MAX_PROJECTION_ROWS: usize = 1_000_000;
const MAX_PROGRAM_INPUTS: usize = 8;
const MAX_PROGRAM_INSTRUCTIONS: usize = 128;
const GATE_MAGIC: &[u8; 8] = b"PLLMAGC1";
const PROGRAM_MAGIC: &[u8; 8] = b"PLLMAGP1";

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Modulus(u16);

impl Modulus {
    pub fn new(value: u16) -> Result<Self, GarbleError> {
        if !(2..=MAX_MODULUS).contains(&value) {
            return Err(GarbleError::InvalidModulus(value));
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> u16 {
        self.0
    }

    fn label_width(self) -> usize {
        // The last component is the public point-and-permute selector. Keep a
        // full security parameter of hidden components in addition to it.
        (SECURITY_BITS / f64::from(self.0).log2()).ceil() as usize + 1
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct Label {
    modulus: Modulus,
    components: Vec<u16>,
}

impl fmt::Debug for Label {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Label")
            .field("modulus", &self.modulus)
            .field("components", &"<redacted>")
            .finish()
    }
}

impl Drop for Label {
    fn drop(&mut self) {
        self.components.zeroize();
    }
}

impl Label {
    pub fn modulus(&self) -> Modulus {
        self.modulus
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut output = Vec::with_capacity(4 + self.components.len() * 2);
        output.extend_from_slice(&self.modulus.0.to_le_bytes());
        output.extend_from_slice(&(self.components.len() as u16).to_le_bytes());
        for component in &self.components {
            output.extend_from_slice(&component.to_le_bytes());
        }
        output
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, GarbleError> {
        if bytes.len() < 4 || (bytes.len() - 4) % 2 != 0 {
            return Err(GarbleError::InvalidLabel);
        }
        let modulus = Modulus::new(u16::from_le_bytes([bytes[0], bytes[1]]))?;
        let width = usize::from(u16::from_le_bytes([bytes[2], bytes[3]]));
        if width != modulus.label_width() || bytes.len() != 4 + width * 2 {
            return Err(GarbleError::InvalidLabel);
        }
        let mut components = Vec::with_capacity(width);
        for chunk in bytes[4..].chunks_exact(2) {
            let component = u16::from_le_bytes([chunk[0], chunk[1]]);
            if component >= modulus.0 {
                return Err(GarbleError::InvalidLabel);
            }
            components.push(component);
        }
        Ok(Self {
            modulus,
            components,
        })
    }

    fn selector(&self) -> usize {
        usize::from(*self.components.last().expect("validated label is nonempty"))
    }
}

#[derive(Clone)]
pub struct WireEncoding {
    modulus: Modulus,
    base: Label,
    offset: Label,
}

impl fmt::Debug for WireEncoding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WireEncoding")
            .field("modulus", &self.modulus)
            .field("material", &"<client-only>")
            .finish()
    }
}

impl WireEncoding {
    pub fn modulus(&self) -> Modulus {
        self.modulus
    }

    pub fn encode(&self, value: u16) -> Result<Label, GarbleError> {
        if value >= self.modulus.0 {
            return Err(GarbleError::ValueOutsideModulus {
                value,
                modulus: self.modulus.0,
            });
        }
        Ok(add_labels(&self.base, &scale_label(&self.offset, value)))
    }

    pub fn decode(&self, label: &Label) -> Result<u16, GarbleError> {
        check_label(label, self.modulus)?;
        for value in 0..self.modulus.0 {
            if self.encode(value)? == *label {
                return Ok(value);
            }
        }
        Err(GarbleError::UnknownOutputLabel)
    }
}

pub struct Garbler {
    offsets: BTreeMap<Modulus, Label>,
}

impl Garbler {
    pub fn new() -> Self {
        Self {
            offsets: BTreeMap::new(),
        }
    }

    pub fn wire(&mut self, modulus: Modulus) -> Result<WireEncoding, GarbleError> {
        let offset = self.offset(modulus)?.clone();
        Ok(WireEncoding {
            modulus,
            base: random_label(modulus)?,
            offset,
        })
    }

    pub fn garble_unary<F>(
        &mut self,
        input: &WireEncoding,
        output_modulus: Modulus,
        operation: F,
    ) -> Result<(GarbledUnary, WireEncoding), GarbleError>
    where
        F: Fn(u16) -> u16,
    {
        let output = self.wire(output_modulus)?;
        let gate_id = random_array()?;
        let mut rows = empty_rows(usize::from(input.modulus.0));
        for value in 0..input.modulus.0 {
            let mapped = operation(value);
            validate_value(mapped, output_modulus)?;
            let input_label = input.encode(value)?;
            let output_label = output.encode(mapped)?;
            let body = seal(&gate_id, &[&input_label], &output_label);
            rows[input_label.selector()] = Some(make_row(&gate_id, &[&input_label], body));
        }
        Ok((
            GarbledUnary {
                input_modulus: input.modulus,
                output_modulus,
                gate_id,
                rows: finish_rows(rows)?,
            },
            output,
        ))
    }

    pub fn garble_binary<F>(
        &mut self,
        left: &WireEncoding,
        right: &WireEncoding,
        output_modulus: Modulus,
        operation: F,
    ) -> Result<(GarbledBinary, WireEncoding), GarbleError>
    where
        F: Fn(u16, u16) -> u16,
    {
        let output = self.wire(output_modulus)?;
        let gate_id = random_array()?;
        let right_rows = usize::from(right.modulus.0);
        let mut rows = empty_rows(usize::from(left.modulus.0) * right_rows);
        for left_value in 0..left.modulus.0 {
            for right_value in 0..right.modulus.0 {
                let mapped = operation(left_value, right_value);
                validate_value(mapped, output_modulus)?;
                let left_label = left.encode(left_value)?;
                let right_label = right.encode(right_value)?;
                let output_label = output.encode(mapped)?;
                let index = left_label.selector() * right_rows + right_label.selector();
                let body = seal(&gate_id, &[&left_label, &right_label], &output_label);
                rows[index] = Some(make_row(&gate_id, &[&left_label, &right_label], body));
            }
        }
        Ok((
            GarbledBinary {
                left_modulus: left.modulus,
                right_modulus: right.modulus,
                output_modulus,
                gate_id,
                rows: finish_rows(rows)?,
            },
            output,
        ))
    }

    pub fn garble_projection<F>(
        &mut self,
        inputs: &[&WireEncoding],
        output_modulus: Modulus,
        operation: F,
    ) -> Result<(GarbledProjection, WireEncoding), GarbleError>
    where
        F: Fn(&[u16]) -> u16,
    {
        self.garble_projection_with_context(inputs, output_modulus, [0_u8; 32], operation)
    }

    pub fn garble_projection_with_context<F>(
        &mut self,
        inputs: &[&WireEncoding],
        output_modulus: Modulus,
        context_digest: [u8; 32],
        operation: F,
    ) -> Result<(GarbledProjection, WireEncoding), GarbleError>
    where
        F: Fn(&[u16]) -> u16,
    {
        if inputs.is_empty() {
            return Err(GarbleError::EmptyProjection);
        }
        let row_count = inputs.iter().try_fold(1_usize, |count, input| {
            count
                .checked_mul(usize::from(input.modulus.0))
                .filter(|value| *value <= MAX_PROJECTION_ROWS)
                .ok_or(GarbleError::ProjectionTooLarge)
        })?;
        let output = self.wire(output_modulus)?;
        let gate_id = random_array()?;
        let row_gate_id = contextual_gate_id(&gate_id, &context_digest);
        let mut rows = empty_rows(row_count);
        let moduli = inputs.iter().map(|input| input.modulus).collect::<Vec<_>>();
        for semantic_index in 0..row_count {
            let values = mixed_radix_values(semantic_index, &moduli);
            let mapped = operation(&values);
            if mapped >= output_modulus.0 {
                return Err(GarbleError::ValueOutsideModulus {
                    value: mapped,
                    modulus: output_modulus.0,
                });
            }
            let labels = inputs
                .iter()
                .zip(&values)
                .map(|(input, value)| input.encode(*value))
                .collect::<Result<Vec<_>, _>>()?;
            let label_refs = labels.iter().collect::<Vec<_>>();
            let index = mixed_radix_selectors(&label_refs, &moduli);
            let output_label = output.encode(mapped)?;
            let body = seal(&row_gate_id, &label_refs, &output_label);
            rows[index] = Some(make_row(&row_gate_id, &label_refs, body));
        }
        Ok((
            GarbledProjection {
                input_moduli: moduli,
                output_modulus,
                gate_id,
                context_digest,
                rows: finish_rows(rows)?,
            },
            output,
        ))
    }

    pub fn add(
        &self,
        left: &WireEncoding,
        right: &WireEncoding,
    ) -> Result<WireEncoding, GarbleError> {
        if left.modulus != right.modulus || left.offset != right.offset {
            return Err(GarbleError::IncompatibleWires);
        }
        Ok(WireEncoding {
            modulus: left.modulus,
            base: add_labels(&left.base, &right.base),
            offset: left.offset.clone(),
        })
    }

    pub fn scale(&self, input: &WireEncoding, scalar: u16) -> Result<WireEncoding, GarbleError> {
        validate_value(scalar, input.modulus)?;
        if gcd(scalar, input.modulus.0) != 1 {
            return Err(GarbleError::NonUnitScale {
                scalar,
                modulus: input.modulus.0,
            });
        }
        Ok(WireEncoding {
            modulus: input.modulus,
            base: scale_label(&input.base, scalar),
            offset: input.offset.clone(),
        })
    }

    fn offset(&mut self, modulus: Modulus) -> Result<&Label, GarbleError> {
        if let std::collections::btree_map::Entry::Vacant(entry) = self.offsets.entry(modulus) {
            let mut offset = random_label(modulus)?;
            *offset
                .components
                .last_mut()
                .expect("validated label is nonempty") = 1;
            entry.insert(offset);
        }
        Ok(self.offsets.get(&modulus).expect("offset was inserted"))
    }
}

impl Default for Garbler {
    fn default() -> Self {
        Self::new()
    }
}

struct CipherRow {
    body: Vec<u16>,
    tag: [u8; TAG_BYTES],
}

pub struct GarbledUnary {
    input_modulus: Modulus,
    output_modulus: Modulus,
    gate_id: [u8; 32],
    rows: Vec<CipherRow>,
}

impl GarbledUnary {
    pub fn evaluate(self, input: &Label) -> Result<Label, GarbleError> {
        check_label(input, self.input_modulus)?;
        let row = self
            .rows
            .get(input.selector())
            .ok_or(GarbleError::InvalidLabel)?;
        open(&self.gate_id, &[input], self.output_modulus, row)
    }
}

pub struct GarbledBinary {
    left_modulus: Modulus,
    right_modulus: Modulus,
    output_modulus: Modulus,
    gate_id: [u8; 32],
    rows: Vec<CipherRow>,
}

pub struct GarbledProjection {
    input_moduli: Vec<Modulus>,
    output_modulus: Modulus,
    gate_id: [u8; 32],
    context_digest: [u8; 32],
    rows: Vec<CipherRow>,
}

enum ProgramInstruction {
    Projection {
        inputs: Vec<u16>,
        gate: GarbledProjection,
    },
    Add {
        left: u16,
        right: u16,
    },
    Scale {
        input: u16,
        scalar: u16,
    },
}

/// Strictly serialized mixed-modulus arithmetic garbling program.
pub struct GarbledProgram {
    context_digest: [u8; 32],
    input_moduli: Vec<Modulus>,
    instructions: Vec<ProgramInstruction>,
    output: u16,
}

impl GarbledProjection {
    pub fn input_moduli(&self) -> Vec<u16> {
        self.input_moduli.iter().map(|modulus| modulus.0).collect()
    }

    pub fn output_modulus(&self) -> u16 {
        self.output_modulus.0
    }

    pub fn context_digest(&self) -> [u8; 32] {
        self.context_digest
    }

    pub fn material_id(&self) -> [u8; 32] {
        contextual_gate_id(&self.gate_id, &self.context_digest)
    }

    pub fn evaluate(self, inputs: &[&Label]) -> Result<Label, GarbleError> {
        if inputs.len() != self.input_moduli.len() {
            return Err(GarbleError::InvalidLabel);
        }
        for (input, modulus) in inputs.iter().zip(&self.input_moduli) {
            check_label(input, *modulus)?;
        }
        let index = mixed_radix_selectors(inputs, &self.input_moduli);
        let row = self.rows.get(index).ok_or(GarbleError::InvalidLabel)?;
        let gate_id = contextual_gate_id(&self.gate_id, &self.context_digest);
        open(&gate_id, inputs, self.output_modulus, row)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let width = self.output_modulus.label_width();
        let row_bytes = width * 2 + TAG_BYTES;
        let mut output = Vec::with_capacity(
            GATE_MAGIC.len()
                + 2
                + self.input_moduli.len() * 2
                + 2
                + 32
                + 32
                + 4
                + 2
                + self.rows.len() * row_bytes,
        );
        output.extend_from_slice(GATE_MAGIC);
        output.extend_from_slice(&(self.input_moduli.len() as u16).to_le_bytes());
        for modulus in &self.input_moduli {
            output.extend_from_slice(&modulus.0.to_le_bytes());
        }
        output.extend_from_slice(&self.output_modulus.0.to_le_bytes());
        output.extend_from_slice(&self.gate_id);
        output.extend_from_slice(&self.context_digest);
        output.extend_from_slice(&(self.rows.len() as u32).to_le_bytes());
        output.extend_from_slice(&(width as u16).to_le_bytes());
        for row in &self.rows {
            for component in &row.body {
                output.extend_from_slice(&component.to_le_bytes());
            }
            output.extend_from_slice(&row.tag);
        }
        output
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, GarbleError> {
        let mut cursor = Cursor::new(bytes);
        if cursor.take::<8>()? != *GATE_MAGIC {
            return Err(GarbleError::InvalidGate);
        }
        let input_count = usize::from(cursor.u16()?);
        if input_count == 0 {
            return Err(GarbleError::InvalidGate);
        }
        let mut input_moduli = Vec::with_capacity(input_count);
        for _ in 0..input_count {
            input_moduli.push(Modulus::new(cursor.u16()?)?);
        }
        let output_modulus = Modulus::new(cursor.u16()?)?;
        let gate_id = cursor.take::<32>()?;
        let context_digest = cursor.take::<32>()?;
        let row_count = usize::try_from(cursor.u32()?).map_err(|_| GarbleError::InvalidGate)?;
        let width = usize::from(cursor.u16()?);
        let expected_rows = input_moduli.iter().try_fold(1_usize, |count, modulus| {
            count
                .checked_mul(usize::from(modulus.0))
                .filter(|value| *value <= MAX_PROJECTION_ROWS)
                .ok_or(GarbleError::ProjectionTooLarge)
        })?;
        if row_count != expected_rows || width != output_modulus.label_width() {
            return Err(GarbleError::InvalidGate);
        }
        let mut rows = Vec::with_capacity(row_count);
        for _ in 0..row_count {
            let mut body = Vec::with_capacity(width);
            for _ in 0..width {
                let component = cursor.u16()?;
                if component >= output_modulus.0 {
                    return Err(GarbleError::InvalidGate);
                }
                body.push(component);
            }
            rows.push(CipherRow {
                body,
                tag: cursor.take::<TAG_BYTES>()?,
            });
        }
        if !cursor.is_empty() {
            return Err(GarbleError::InvalidGate);
        }
        Ok(Self {
            input_moduli,
            output_modulus,
            gate_id,
            context_digest,
            rows,
        })
    }
}

impl GarbledProgram {
    pub fn input_moduli(&self) -> Vec<u16> {
        self.input_moduli
            .iter()
            .map(|modulus| modulus.get())
            .collect()
    }

    pub fn context_digest(&self) -> [u8; 32] {
        self.context_digest
    }

    pub fn material_id(&self) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(b"pllm.garble.program.material.v1\0");
        hash.update(self.to_bytes());
        hash.finalize().into()
    }

    pub fn evaluate(self, inputs: &[&[u8]]) -> Result<Vec<u8>, GarbleError> {
        self.validate()?;
        if inputs.len() != self.input_moduli.len() {
            return Err(GarbleError::InvalidLabel);
        }
        let mut wires = inputs
            .iter()
            .zip(&self.input_moduli)
            .map(|(bytes, modulus)| {
                let label = Label::from_bytes(bytes)?;
                check_label(&label, *modulus)?;
                Ok(label)
            })
            .collect::<Result<Vec<_>, GarbleError>>()?;
        for instruction in self.instructions {
            let output = match instruction {
                ProgramInstruction::Projection { inputs, gate } => {
                    let labels = inputs
                        .iter()
                        .map(|index| {
                            wires
                                .get(usize::from(*index))
                                .ok_or(GarbleError::InvalidProgram)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    gate.evaluate(&labels)?
                }
                ProgramInstruction::Add { left, right } => evaluate_add(
                    wires
                        .get(usize::from(left))
                        .ok_or(GarbleError::InvalidProgram)?,
                    wires
                        .get(usize::from(right))
                        .ok_or(GarbleError::InvalidProgram)?,
                )?,
                ProgramInstruction::Scale { input, scalar } => evaluate_scale(
                    wires
                        .get(usize::from(input))
                        .ok_or(GarbleError::InvalidProgram)?,
                    scalar,
                )?,
            };
            wires.push(output);
        }
        wires
            .pop()
            .map(|label| label.to_bytes())
            .ok_or(GarbleError::InvalidProgram)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut output = Vec::new();
        output.extend_from_slice(PROGRAM_MAGIC);
        output.extend_from_slice(&self.context_digest);
        output.extend_from_slice(&(self.input_moduli.len() as u16).to_le_bytes());
        for modulus in &self.input_moduli {
            output.extend_from_slice(&modulus.0.to_le_bytes());
        }
        output.extend_from_slice(&(self.instructions.len() as u16).to_le_bytes());
        for instruction in &self.instructions {
            match instruction {
                ProgramInstruction::Projection { inputs, gate } => {
                    output.push(0);
                    output.extend_from_slice(&(inputs.len() as u16).to_le_bytes());
                    for input in inputs {
                        output.extend_from_slice(&input.to_le_bytes());
                    }
                    let gate = gate.to_bytes();
                    output.extend_from_slice(&(gate.len() as u32).to_le_bytes());
                    output.extend_from_slice(&gate);
                }
                ProgramInstruction::Add { left, right } => {
                    output.push(1);
                    output.extend_from_slice(&left.to_le_bytes());
                    output.extend_from_slice(&right.to_le_bytes());
                }
                ProgramInstruction::Scale { input, scalar } => {
                    output.push(2);
                    output.extend_from_slice(&input.to_le_bytes());
                    output.extend_from_slice(&scalar.to_le_bytes());
                }
            }
        }
        output.extend_from_slice(&self.output.to_le_bytes());
        output
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, GarbleError> {
        let mut cursor = Cursor::new(bytes);
        if cursor.take::<8>()? != *PROGRAM_MAGIC {
            return Err(GarbleError::InvalidProgram);
        }
        let context_digest = cursor.take::<32>()?;
        let input_count = usize::from(cursor.u16()?);
        if !(1..=MAX_PROGRAM_INPUTS).contains(&input_count) {
            return Err(GarbleError::InvalidProgram);
        }
        let mut input_moduli = Vec::with_capacity(input_count);
        for _ in 0..input_count {
            input_moduli.push(Modulus::new(cursor.u16()?)?);
        }
        let instruction_count = usize::from(cursor.u16()?);
        if !(1..=MAX_PROGRAM_INSTRUCTIONS).contains(&instruction_count) {
            return Err(GarbleError::InvalidProgram);
        }
        let mut instructions = Vec::with_capacity(instruction_count);
        for _ in 0..instruction_count {
            instructions.push(match cursor.u8()? {
                0 => {
                    let count = usize::from(cursor.u16()?);
                    if count == 0 || count > MAX_PROGRAM_INPUTS {
                        return Err(GarbleError::InvalidProgram);
                    }
                    let mut inputs = Vec::with_capacity(count);
                    for _ in 0..count {
                        inputs.push(cursor.u16()?);
                    }
                    let length =
                        usize::try_from(cursor.u32()?).map_err(|_| GarbleError::InvalidProgram)?;
                    if length > cursor.remaining() {
                        return Err(GarbleError::InvalidProgram);
                    }
                    let gate = GarbledProjection::from_bytes(cursor.slice(length)?)?;
                    ProgramInstruction::Projection { inputs, gate }
                }
                1 => ProgramInstruction::Add {
                    left: cursor.u16()?,
                    right: cursor.u16()?,
                },
                2 => ProgramInstruction::Scale {
                    input: cursor.u16()?,
                    scalar: cursor.u16()?,
                },
                _ => return Err(GarbleError::InvalidProgram),
            });
        }
        let program = Self {
            context_digest,
            input_moduli,
            instructions,
            output: cursor.u16()?,
        };
        if !cursor.is_empty() {
            return Err(GarbleError::InvalidProgram);
        }
        program.validate()?;
        Ok(program)
    }

    fn validate(&self) -> Result<(), GarbleError> {
        if !(1..=MAX_PROGRAM_INPUTS).contains(&self.input_moduli.len())
            || !(1..=MAX_PROGRAM_INSTRUCTIONS).contains(&self.instructions.len())
        {
            return Err(GarbleError::InvalidProgram);
        }
        let mut moduli = self.input_moduli.clone();
        for instruction in &self.instructions {
            let modulus = match instruction {
                ProgramInstruction::Projection { inputs, gate } => {
                    if gate.context_digest != self.context_digest
                        || inputs.len() != gate.input_moduli.len()
                        || inputs
                            .iter()
                            .zip(&gate.input_moduli)
                            .any(|(index, expected)| {
                                moduli.get(usize::from(*index)) != Some(expected)
                            })
                    {
                        return Err(GarbleError::InvalidProgram);
                    }
                    gate.output_modulus
                }
                ProgramInstruction::Add { left, right } => {
                    let left = moduli
                        .get(usize::from(*left))
                        .ok_or(GarbleError::InvalidProgram)?;
                    let right = moduli
                        .get(usize::from(*right))
                        .ok_or(GarbleError::InvalidProgram)?;
                    if left != right {
                        return Err(GarbleError::InvalidProgram);
                    }
                    *left
                }
                ProgramInstruction::Scale { input, scalar } => {
                    let modulus = *moduli
                        .get(usize::from(*input))
                        .ok_or(GarbleError::InvalidProgram)?;
                    validate_value(*scalar, modulus)?;
                    if gcd(*scalar, modulus.0) != 1 {
                        return Err(GarbleError::InvalidProgram);
                    }
                    modulus
                }
            };
            moduli.push(modulus);
        }
        let expected = moduli
            .len()
            .checked_sub(1)
            .ok_or(GarbleError::InvalidProgram)?;
        if usize::from(self.output) != expected {
            return Err(GarbleError::InvalidProgram);
        }
        Ok(())
    }
}

struct ProgramBuilder {
    garbler: Garbler,
    context_digest: [u8; 32],
    input_count: usize,
    wires: Vec<WireEncoding>,
    instructions: Vec<ProgramInstruction>,
}

impl ProgramBuilder {
    fn new(context_digest: [u8; 32]) -> Self {
        Self {
            garbler: Garbler::new(),
            context_digest,
            input_count: 0,
            wires: Vec::new(),
            instructions: Vec::new(),
        }
    }

    fn input(&mut self, modulus: Modulus) -> Result<u16, GarbleError> {
        if self.input_count != self.wires.len() || self.input_count >= MAX_PROGRAM_INPUTS {
            return Err(GarbleError::InvalidProgram);
        }
        let wire = self.garbler.wire(modulus)?;
        let index = wire_index(self.wires.len())?;
        self.wires.push(wire);
        self.input_count += 1;
        Ok(index)
    }

    fn project<F>(
        &mut self,
        inputs: &[u16],
        output_modulus: Modulus,
        operation: F,
    ) -> Result<u16, GarbleError>
    where
        F: Fn(&[u16]) -> u16,
    {
        self.ensure_instruction_capacity()?;
        let encodings = inputs
            .iter()
            .map(|index| {
                self.wires
                    .get(usize::from(*index))
                    .cloned()
                    .ok_or(GarbleError::InvalidProgram)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let references = encodings.iter().collect::<Vec<_>>();
        let (gate, output) = self.garbler.garble_projection_with_context(
            &references,
            output_modulus,
            self.context_digest,
            operation,
        )?;
        let index = wire_index(self.wires.len())?;
        self.instructions.push(ProgramInstruction::Projection {
            inputs: inputs.to_vec(),
            gate,
        });
        self.wires.push(output);
        Ok(index)
    }

    fn add(&mut self, left: u16, right: u16) -> Result<u16, GarbleError> {
        self.ensure_instruction_capacity()?;
        let output = self.garbler.add(
            self.wires
                .get(usize::from(left))
                .ok_or(GarbleError::InvalidProgram)?,
            self.wires
                .get(usize::from(right))
                .ok_or(GarbleError::InvalidProgram)?,
        )?;
        let index = wire_index(self.wires.len())?;
        self.instructions
            .push(ProgramInstruction::Add { left, right });
        self.wires.push(output);
        Ok(index)
    }

    fn scale(&mut self, input: u16, scalar: u16) -> Result<u16, GarbleError> {
        self.ensure_instruction_capacity()?;
        let output = self.garbler.scale(
            self.wires
                .get(usize::from(input))
                .ok_or(GarbleError::InvalidProgram)?,
            scalar,
        )?;
        let index = wire_index(self.wires.len())?;
        self.instructions
            .push(ProgramInstruction::Scale { input, scalar });
        self.wires.push(output);
        Ok(index)
    }

    fn finish(
        self,
        output: u16,
    ) -> Result<(GarbledProgram, Vec<WireEncoding>, WireEncoding), GarbleError> {
        if usize::from(output) + 1 != self.wires.len() {
            return Err(GarbleError::InvalidProgram);
        }
        let inputs = self.wires[..self.input_count].to_vec();
        let output_encoding = self
            .wires
            .last()
            .cloned()
            .ok_or(GarbleError::InvalidProgram)?;
        let program = GarbledProgram {
            context_digest: self.context_digest,
            input_moduli: inputs.iter().map(WireEncoding::modulus).collect(),
            instructions: self.instructions,
            output,
        };
        program.validate()?;
        Ok((program, inputs, output_encoding))
    }

    fn ensure_instruction_capacity(&self) -> Result<(), GarbleError> {
        if self.instructions.len() >= MAX_PROGRAM_INSTRUCTIONS {
            Err(GarbleError::InvalidProgram)
        } else {
            Ok(())
        }
    }
}

fn wire_index(index: usize) -> Result<u16, GarbleError> {
    u16::try_from(index).map_err(|_| GarbleError::InvalidProgram)
}

fn garble_prime_multiplication(
    builder: &mut ProgramBuilder,
    left: u16,
    right: u16,
    prime: u16,
) -> Result<u16, GarbleError> {
    if !is_prime(prime)
        || builder
            .wires
            .get(usize::from(left))
            .map(WireEncoding::modulus)
            != Some(Modulus::new(prime)?)
        || builder
            .wires
            .get(usize::from(right))
            .map(WireEncoding::modulus)
            != Some(Modulus::new(prime)?)
    {
        return Err(GarbleError::InvalidProgram);
    }
    let exponent_modulus = Modulus::new(prime - 1)?;
    let bit_modulus = Modulus::new(3)?;
    let primitive_root = primitive_root(prime).ok_or(GarbleError::InvalidProgram)?;
    let logarithms = discrete_logarithms(prime, primitive_root)?;

    let left_log = builder.project(&[left], exponent_modulus, |values| {
        logarithms[usize::from(values[0])]
    })?;
    let right_log = builder.project(&[right], exponent_modulus, |values| {
        logarithms[usize::from(values[0])]
    })?;
    let left_zero = builder.project(&[left], bit_modulus, |values| u16::from(values[0] == 0))?;
    let right_zero = builder.project(&[right], bit_modulus, |values| u16::from(values[0] == 0))?;
    let exponent = builder.add(left_log, right_log)?;
    let either_zero = builder.project(&[left_zero, right_zero], bit_modulus, |values| {
        u16::from(values[0] != 0 || values[1] != 0)
    })?;
    builder.project(&[exponent, either_zero], Modulus::new(prime)?, |values| {
        if values[1] == 0 {
            modular_power(primitive_root, values[0], prime)
        } else {
            0
        }
    })
}

fn is_prime(value: u16) -> bool {
    value >= 2
        && (2..)
            .take_while(|divisor| divisor * divisor <= value)
            .all(|divisor| value % divisor != 0)
}

fn primitive_root(prime: u16) -> Option<u16> {
    let mut factors = Vec::new();
    let mut remainder = prime - 1;
    let mut divisor = 2;
    while divisor * divisor <= remainder {
        if remainder % divisor == 0 {
            factors.push(divisor);
            while remainder % divisor == 0 {
                remainder /= divisor;
            }
        }
        divisor += 1;
    }
    if remainder > 1 {
        factors.push(remainder);
    }
    (2..prime).find(|candidate| {
        factors
            .iter()
            .all(|factor| modular_power(*candidate, (prime - 1) / factor, prime) != 1)
    })
}

fn discrete_logarithms(prime: u16, root: u16) -> Result<Vec<u16>, GarbleError> {
    let mut logarithms = vec![0_u16; usize::from(prime)];
    let mut value = 1_u16;
    for exponent in 0..prime - 1 {
        logarithms[usize::from(value)] = exponent;
        value = u16::try_from((u32::from(value) * u32::from(root)) % u32::from(prime))
            .map_err(|_| GarbleError::InvalidProgram)?;
    }
    // Exponent zero legitimately maps value one; every other nonzero value
    // must have received a nonzero exponent.
    if value != 1
        || !(1..prime).all(|candidate| candidate == 1 || logarithms[usize::from(candidate)] != 0)
    {
        return Err(GarbleError::InvalidProgram);
    }
    Ok(logarithms)
}

fn modular_power(base: u16, exponent: u16, modulus: u16) -> u16 {
    let mut result = 1_u32;
    let mut base = u32::from(base);
    let mut exponent = exponent;
    let modulus = u32::from(modulus);
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = result * base % modulus;
        }
        base = base * base % modulus;
        exponent >>= 1;
    }
    result as u16
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CrtBasis {
    moduli: Vec<Modulus>,
    product: u64,
}

impl CrtBasis {
    pub fn new(moduli: Vec<Modulus>) -> Result<Self, GarbleError> {
        if moduli.is_empty() {
            return Err(GarbleError::EmptyCrtBasis);
        }
        let mut product = 1_u64;
        for (index, modulus) in moduli.iter().enumerate() {
            if moduli[..index]
                .iter()
                .any(|previous| gcd(previous.0, modulus.0) != 1)
            {
                return Err(GarbleError::NonCoprimeCrtBasis);
            }
            product = product
                .checked_mul(u64::from(modulus.0))
                .ok_or(GarbleError::CrtBasisTooLarge)?;
        }
        if product > i64::MAX as u64 {
            return Err(GarbleError::CrtBasisTooLarge);
        }
        Ok(Self { moduli, product })
    }

    pub fn moduli(&self) -> &[Modulus] {
        &self.moduli
    }

    pub fn product(&self) -> u64 {
        self.product
    }

    pub fn residues(&self, value: i64) -> Vec<u16> {
        self.moduli
            .iter()
            .map(|modulus| value.rem_euclid(i64::from(modulus.0)) as u16)
            .collect()
    }

    pub fn reconstruct(&self, residues: &[u16]) -> Result<i64, GarbleError> {
        if residues.len() != self.moduli.len()
            || residues
                .iter()
                .zip(&self.moduli)
                .any(|(value, modulus)| *value >= modulus.0)
        {
            return Err(GarbleError::InvalidCrtResidues);
        }
        let product = i128::from(self.product);
        let mut result = 0_i128;
        for (residue, modulus) in residues.iter().zip(&self.moduli) {
            let modulus = i128::from(modulus.0);
            let partial = product / modulus;
            let inverse = modular_inverse(partial % modulus, modulus)
                .ok_or(GarbleError::NonCoprimeCrtBasis)?;
            result = (result + i128::from(*residue) * partial * inverse).rem_euclid(product);
        }
        if result > product / 2 {
            result -= product;
        }
        i64::try_from(result).map_err(|_| GarbleError::CrtBasisTooLarge)
    }
}

impl GarbledBinary {
    pub fn evaluate(self, left: &Label, right: &Label) -> Result<Label, GarbleError> {
        check_label(left, self.left_modulus)?;
        check_label(right, self.right_modulus)?;
        let index = left.selector() * usize::from(self.right_modulus.0) + right.selector();
        let row = self.rows.get(index).ok_or(GarbleError::InvalidLabel)?;
        open(&self.gate_id, &[left, right], self.output_modulus, row)
    }
}

pub fn evaluate_add(left: &Label, right: &Label) -> Result<Label, GarbleError> {
    if left.modulus != right.modulus {
        return Err(GarbleError::IncompatibleWires);
    }
    Ok(add_labels(left, right))
}

pub fn evaluate_scale(input: &Label, scalar: u16) -> Result<Label, GarbleError> {
    validate_value(scalar, input.modulus)?;
    if gcd(scalar, input.modulus.0) != 1 {
        return Err(GarbleError::NonUnitScale {
            scalar,
            modulus: input.modulus.0,
        });
    }
    Ok(scale_label(input, scalar))
}

pub const SILU_QUADRATIC_Q7_MODULUS: u16 = 257;

pub struct SiluQuadraticQ7Material {
    gate: GarbledProjection,
    input: WireEncoding,
    output: WireEncoding,
}

/// Client encodings and evaluator gates for label-preserving `SiLU(gate) * up`.
pub struct GatedMultiplyQ7Material {
    program: GarbledProgram,
    gate_input: WireEncoding,
    up_input: WireEncoding,
    output: WireEncoding,
}

impl GatedMultiplyQ7Material {
    pub fn program_bytes(&self) -> Vec<u8> {
        self.program.to_bytes()
    }

    pub fn material_id(&self) -> [u8; 32] {
        self.program.material_id()
    }

    pub fn encode_gate(&self, value: i16) -> Result<Vec<u8>, GarbleError> {
        Ok(self.gate_input.encode(q7_residue(value)?)?.to_bytes())
    }

    pub fn encode_up(&self, value: i16) -> Result<Vec<u8>, GarbleError> {
        Ok(self.up_input.encode(q7_residue(value)?)?.to_bytes())
    }

    pub fn decode(&self, label_bytes: &[u8]) -> Result<i16, GarbleError> {
        let label = Label::from_bytes(label_bytes)?;
        Ok(q7_centered(self.output.decode(&label)?))
    }
}

impl SiluQuadraticQ7Material {
    pub fn gate_bytes(&self) -> Vec<u8> {
        self.gate.to_bytes()
    }

    pub fn encode(&self, value: i16) -> Result<Vec<u8>, GarbleError> {
        Ok(self.input.encode(q7_residue(value)?)?.to_bytes())
    }

    pub fn decode(&self, label_bytes: &[u8]) -> Result<i16, GarbleError> {
        let label = Label::from_bytes(label_bytes)?;
        Ok(q7_centered(self.output.decode(&label)?))
    }
}

pub fn prepare_silu_quadratic_q7() -> Result<SiluQuadraticQ7Material, GarbleError> {
    prepare_silu_quadratic_q7_with_context([0_u8; 32])
}

pub fn prepare_silu_quadratic_q7_with_context(
    context_digest: [u8; 32],
) -> Result<SiluQuadraticQ7Material, GarbleError> {
    let modulus = Modulus::new(SILU_QUADRATIC_Q7_MODULUS)?;
    let mut garbler = Garbler::new();
    let input = garbler.wire(modulus)?;
    let (gate, output) =
        garbler.garble_projection_with_context(&[&input], modulus, context_digest, |residues| {
            let value = q7_centered(residues[0]);
            let result = pllm_core::silu_quadratic_q7(value)
                .expect("every modulus-257 residue maps to the locked Q7 domain");
            q7_residue(result).expect("locked Q7 SiLU output remains in domain")
        })?;
    Ok(SiluQuadraticQ7Material {
        gate,
        input,
        output,
    })
}

pub fn prepare_gated_multiply_q7() -> Result<GatedMultiplyQ7Material, GarbleError> {
    prepare_gated_multiply_q7_with_context([0_u8; 32])
}

/// Prepare one compact mixed-modulus program so intermediate values remain labels.
pub fn prepare_gated_multiply_q7_with_context(
    context_digest: [u8; 32],
) -> Result<GatedMultiplyQ7Material, GarbleError> {
    let q7_modulus = Modulus::new(SILU_QUADRATIC_Q7_MODULUS)?;
    let modulus_131 = Modulus::new(131)?;
    let modulus_387 = Modulus::new(387)?;
    let modulus_128 = Modulus::new(128)?;
    let modulus_4 = Modulus::new(4)?;
    let bit_modulus = Modulus::new(3)?;
    let mut builder = ProgramBuilder::new(context_digest);
    let gate_input_index = builder.input(q7_modulus)?;
    let up_input_index = builder.input(q7_modulus)?;
    let activated = builder.project(&[gate_input_index], q7_modulus, |values| {
        let result = pllm_core::silu_quadratic_q7(q7_centered(values[0]))
            .expect("every modulus-257 residue maps to the locked Q7 domain");
        q7_residue(result).expect("locked Q7 SiLU output remains in domain")
    })?;

    let product_257 = garble_prime_multiplication(&mut builder, activated, up_input_index, 257)?;
    let activated_131 = builder.project(&[activated], modulus_131, |values| {
        q7_centered(values[0]).rem_euclid(131) as u16
    })?;
    let up_131 = builder.project(&[up_input_index], modulus_131, |values| {
        q7_centered(values[0]).rem_euclid(131) as u16
    })?;
    let product_131 = garble_prime_multiplication(&mut builder, activated_131, up_131, 131)?;

    // R03 Section 6.3 converts the two CRT residues into the high mixed-radix
    // digit floor(N / 257), where N is the nonnegative CRT representative.
    let product_257_wide = builder.project(&[product_257], modulus_387, |values| values[0])?;
    let product_131_wide = builder.project(&[product_131], modulus_387, |values| values[0])?;
    let negated_product_131 = builder.scale(product_131_wide, 386)?;
    let residue_difference = builder.add(product_257_wide, negated_product_131)?;
    let quotient_digits = quotient_digit_table(257, 131)?;
    let high_digit = builder.project(&[residue_difference], modulus_131, |values| {
        quotient_digits[usize::from(values[0])]
    })?;

    // N = 257*h + l = 128*(2*h + floor((h+l)/128)) + (h+l mod 128).
    let low_digit_wide = builder.project(&[product_257], modulus_387, |values| values[0])?;
    let high_digit_wide = builder.project(&[high_digit], modulus_387, |values| values[0])?;
    let digit_sum = builder.add(low_digit_wide, high_digit_wide)?;
    let nonnegative_remainder =
        builder.project(&[digit_sum], modulus_128, |values| values[0] % 128)?;
    let carry = builder.project(&[digit_sum], modulus_4, |values| values[0] / 128)?;
    let high_digit_q7 = builder.project(&[high_digit], q7_modulus, |values| values[0])?;
    let twice_high_digit = builder.scale(high_digit_q7, 2)?;
    let carry_q7 = builder.project(&[carry], q7_modulus, |values| values[0])?;
    let nonnegative_quotient = builder.add(twice_high_digit, carry_q7)?;

    // The CRT product 257*131 is odd. Values above its midpoint are the
    // negative half of the centered product domain.
    let high_greater = builder.project(&[high_digit], bit_modulus, |values| {
        u16::from(values[0] > 65)
    })?;
    let high_equal = builder.project(&[high_digit], bit_modulus, |values| {
        u16::from(values[0] == 65)
    })?;
    let low_at_least_129 = builder.project(&[product_257], bit_modulus, |values| {
        u16::from(values[0] >= 129)
    })?;
    let boundary_negative =
        builder.project(&[high_equal, low_at_least_129], bit_modulus, |values| {
            u16::from(values[0] != 0 && values[1] != 0)
        })?;
    let negative = builder.add(high_greater, boundary_negative)?;

    // 257*131 = 128*263 + 3. Convert the unsigned quotient/remainder to the
    // centered signed floor quotient before applying ties-to-even rounding.
    let borrow = builder.project(&[nonnegative_remainder], bit_modulus, |values| {
        u16::from(values[0] < 3)
    })?;
    let signed_borrow = builder.project(&[negative, borrow], bit_modulus, |values| {
        u16::from(values[0] != 0 && values[1] != 0)
    })?;
    let negative_q7 = builder.project(&[negative], q7_modulus, |values| values[0])?;
    let negative_correction = builder.scale(negative_q7, 251)?;
    let signed_borrow_q7 = builder.project(&[signed_borrow], q7_modulus, |values| values[0])?;
    let negated_borrow = builder.scale(signed_borrow_q7, 256)?;
    let corrected_quotient = builder.add(nonnegative_quotient, negative_correction)?;
    let base_quotient = builder.add(corrected_quotient, negated_borrow)?;
    let remainder_correction = builder.project(&[negative], modulus_128, |values| {
        if values[0] == 0 {
            0
        } else {
            125
        }
    })?;
    let remainder = builder.add(nonnegative_remainder, remainder_correction)?;

    let round_up = builder.project(&[remainder], bit_modulus, |values| {
        u16::from(values[0] > 64)
    })?;
    let tie = builder.project(&[remainder], bit_modulus, |values| {
        u16::from(values[0] == 64)
    })?;
    let residue_parity = builder.project(&[base_quotient], bit_modulus, |values| values[0] % 2)?;
    let signed_parity = builder.project(&[residue_parity, negative], bit_modulus, |values| {
        (values[0] + values[1]) % 2
    })?;
    let tie_round = builder.project(&[tie, signed_parity], bit_modulus, |values| {
        u16::from(values[0] != 0 && values[1] != 0)
    })?;
    let round_up_q7 = builder.project(&[round_up], q7_modulus, |values| values[0])?;
    let tie_round_q7 = builder.project(&[tie_round], q7_modulus, |values| values[0])?;
    let rounded = builder.add(base_quotient, round_up_q7)?;
    let output_index = builder.add(rounded, tie_round_q7)?;
    let (program, inputs, output) = builder.finish(output_index)?;
    let [gate_input, up_input]: [WireEncoding; 2] =
        inputs.try_into().map_err(|_| GarbleError::InvalidProgram)?;
    Ok(GatedMultiplyQ7Material {
        program,
        gate_input,
        up_input,
        output,
    })
}

#[cfg(test)]
fn evaluate_silu_quadratic_q7(
    gate_bytes: &[u8],
    input_label_bytes: &[u8],
) -> Result<Vec<u8>, GarbleError> {
    let gate = GarbledProjection::from_bytes(gate_bytes)?;
    let input = Label::from_bytes(input_label_bytes)?;
    Ok(gate.evaluate(&[&input])?.to_bytes())
}

#[cfg(test)]
fn evaluate_gated_multiply_q7(
    program_bytes: &[u8],
    gate_input_label_bytes: &[u8],
    up_input_label_bytes: &[u8],
) -> Result<Vec<u8>, GarbleError> {
    let program = GarbledProgram::from_bytes(program_bytes)?;
    if program.input_moduli
        != [
            Modulus::new(SILU_QUADRATIC_Q7_MODULUS)?,
            Modulus::new(SILU_QUADRATIC_Q7_MODULUS)?,
        ]
    {
        return Err(GarbleError::InvalidProgram);
    }
    program.evaluate(&[gate_input_label_bytes, up_input_label_bytes])
}

fn quotient_digit_table(low_modulus: u16, high_modulus: u16) -> Result<Vec<u16>, GarbleError> {
    let combined = usize::from(low_modulus + high_modulus - 1);
    let mut table = vec![None; combined];
    let product = u32::from(low_modulus) * u32::from(high_modulus);
    for value in 0..product {
        let difference = (i64::from(value % u32::from(low_modulus))
            - i64::from(value % u32::from(high_modulus)))
        .rem_euclid(combined as i64) as usize;
        let quotient = u16::try_from((value / u32::from(low_modulus)) % u32::from(high_modulus))
            .map_err(|_| GarbleError::InvalidProgram)?;
        if let Some(previous) = table[difference] {
            if previous != quotient {
                return Err(GarbleError::InvalidProgram);
            }
        } else {
            table[difference] = Some(quotient);
        }
    }
    table
        .into_iter()
        .map(|value| value.ok_or(GarbleError::InvalidProgram))
        .collect()
}

fn q7_residue(value: i16) -> Result<u16, GarbleError> {
    if !(-128..=128).contains(&value) {
        return Err(GarbleError::ValueOutsideQ7(value));
    }
    Ok(if value < 0 {
        (i32::from(value) + i32::from(SILU_QUADRATIC_Q7_MODULUS)) as u16
    } else {
        value as u16
    })
}

fn q7_centered(residue: u16) -> i16 {
    if residue <= 128 {
        residue as i16
    } else {
        residue as i16 - SILU_QUADRATIC_Q7_MODULUS as i16
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum GarbleError {
    Authentication,
    IncompatibleWires,
    CrtBasisTooLarge,
    EmptyCrtBasis,
    EmptyProjection,
    InvalidLabel,
    InvalidModulus(u16),
    InvalidCrtResidues,
    InvalidGate,
    InvalidProgram,
    NonCoprimeCrtBasis,
    NonUnitScale { scalar: u16, modulus: u16 },
    ProjectionTooLarge,
    Randomness(String),
    UnknownOutputLabel,
    ValueOutsideQ7(i16),
    ValueOutsideModulus { value: u16, modulus: u16 },
}

impl fmt::Display for GarbleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Authentication => formatter.write_str("garbled row authentication failed"),
            Self::IncompatibleWires => formatter.write_str("wire encodings are incompatible"),
            Self::CrtBasisTooLarge => {
                formatter.write_str("CRT basis exceeds the signed 64-bit reference bound")
            }
            Self::EmptyCrtBasis => {
                formatter.write_str("CRT basis must contain at least one modulus")
            }
            Self::EmptyProjection => {
                formatter.write_str("projection must contain at least one input")
            }
            Self::InvalidLabel => formatter.write_str("label encoding is invalid"),
            Self::InvalidModulus(value) => {
                write!(formatter, "modulus {value} is outside 2..={MAX_MODULUS}")
            }
            Self::InvalidCrtResidues => formatter.write_str("CRT residues do not match the basis"),
            Self::InvalidGate => formatter.write_str("garbled gate encoding is invalid"),
            Self::InvalidProgram => formatter.write_str("garbled program encoding is invalid"),
            Self::NonCoprimeCrtBasis => formatter.write_str("CRT moduli must be pairwise coprime"),
            Self::NonUnitScale { scalar, modulus } => {
                write!(formatter, "scale {scalar} is not a unit modulo {modulus}")
            }
            Self::ProjectionTooLarge => {
                formatter.write_str("projection exceeds the one-million-row reference limit")
            }
            Self::Randomness(message) => {
                write!(formatter, "operating-system randomness failed: {message}")
            }
            Self::UnknownOutputLabel => {
                formatter.write_str("output label is not in the decoding set")
            }
            Self::ValueOutsideQ7(value) => {
                write!(formatter, "Q7 value {value} is outside -128..=128")
            }
            Self::ValueOutsideModulus { value, modulus } => {
                write!(formatter, "value {value} is outside modulus {modulus}")
            }
        }
    }
}

impl Error for GarbleError {}

fn validate_value(value: u16, modulus: Modulus) -> Result<(), GarbleError> {
    if value >= modulus.0 {
        return Err(GarbleError::ValueOutsideModulus {
            value,
            modulus: modulus.0,
        });
    }
    Ok(())
}

fn check_label(label: &Label, modulus: Modulus) -> Result<(), GarbleError> {
    if label.modulus != modulus
        || label.components.len() != modulus.label_width()
        || label.components.iter().any(|value| *value >= modulus.0)
    {
        return Err(GarbleError::InvalidLabel);
    }
    Ok(())
}

fn add_labels(left: &Label, right: &Label) -> Label {
    debug_assert_eq!(left.modulus, right.modulus);
    let modulus = u32::from(left.modulus.0);
    Label {
        modulus: left.modulus,
        components: left
            .components
            .iter()
            .zip(&right.components)
            .map(|(left, right)| ((u32::from(*left) + u32::from(*right)) % modulus) as u16)
            .collect(),
    }
}

fn scale_label(label: &Label, scalar: u16) -> Label {
    let modulus = u32::from(label.modulus.0);
    Label {
        modulus: label.modulus,
        components: label
            .components
            .iter()
            .map(|value| ((u32::from(*value) * u32::from(scalar)) % modulus) as u16)
            .collect(),
    }
}

fn random_label(modulus: Modulus) -> Result<Label, GarbleError> {
    let mut components = Vec::with_capacity(modulus.label_width());
    for _ in 0..modulus.label_width() {
        components.push(random_residue(modulus)?);
    }
    Ok(Label {
        modulus,
        components,
    })
}

fn random_residue(modulus: Modulus) -> Result<u16, GarbleError> {
    let modulus_u32 = u32::from(modulus.0);
    let cutoff = u32::MAX - (u32::MAX % modulus_u32);
    loop {
        let mut bytes = [0_u8; 4];
        getrandom::fill(&mut bytes).map_err(|error| GarbleError::Randomness(error.to_string()))?;
        let candidate = u32::from_le_bytes(bytes);
        if candidate < cutoff {
            return Ok((candidate % modulus_u32) as u16);
        }
    }
}

fn random_array() -> Result<[u8; 32], GarbleError> {
    let mut output = [0_u8; 32];
    getrandom::fill(&mut output).map_err(|error| GarbleError::Randomness(error.to_string()))?;
    Ok(output)
}

fn contextual_gate_id(gate_id: &[u8; 32], context_digest: &[u8; 32]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"pllm.agc.contextual-gate.v1\0");
    hash.update(gate_id);
    hash.update(context_digest);
    hash.finalize().into()
}

fn empty_rows(count: usize) -> Vec<Option<CipherRow>> {
    std::iter::repeat_with(|| None).take(count).collect()
}

fn finish_rows(rows: Vec<Option<CipherRow>>) -> Result<Vec<CipherRow>, GarbleError> {
    rows.into_iter()
        .map(|row| row.ok_or(GarbleError::InvalidLabel))
        .collect()
}

fn make_row(gate_id: &[u8; 32], labels: &[&Label], body: Vec<u16>) -> CipherRow {
    let digest = row_tag(gate_id, labels, &body);
    let mut tag = [0_u8; TAG_BYTES];
    tag.copy_from_slice(&digest[..TAG_BYTES]);
    CipherRow { body, tag }
}

fn seal(gate_id: &[u8; 32], labels: &[&Label], output: &Label) -> Vec<u16> {
    let pads = pads(gate_id, labels, output.modulus, output.components.len());
    let modulus = u32::from(output.modulus.0);
    output
        .components
        .iter()
        .zip(pads)
        .map(|(value, pad)| ((u32::from(*value) + modulus - u32::from(pad)) % modulus) as u16)
        .collect()
}

fn open(
    gate_id: &[u8; 32],
    labels: &[&Label],
    output_modulus: Modulus,
    row: &CipherRow,
) -> Result<Label, GarbleError> {
    let expected = row_tag(gate_id, labels, &row.body);
    if !constant_time_equal(&expected[..TAG_BYTES], &row.tag) {
        return Err(GarbleError::Authentication);
    }
    let pads = pads(gate_id, labels, output_modulus, row.body.len());
    let modulus = u32::from(output_modulus.0);
    Ok(Label {
        modulus: output_modulus,
        components: row
            .body
            .iter()
            .zip(pads)
            .map(|(value, pad)| ((u32::from(*value) + u32::from(pad)) % modulus) as u16)
            .collect(),
    })
}

fn pads(gate_id: &[u8; 32], labels: &[&Label], modulus: Modulus, count: usize) -> Vec<u16> {
    let mut output = Vec::with_capacity(count);
    let modulus_u32 = u32::from(modulus.0);
    let cutoff = u32::MAX - (u32::MAX % modulus_u32);
    let mut counter = 0_u64;
    while output.len() < count {
        let mut hash = Sha256::new();
        hash.update(b"pllm.agc.pad.v1\0");
        hash.update(gate_id);
        for label in labels {
            hash.update(label.to_bytes());
        }
        hash.update(counter.to_le_bytes());
        for chunk in hash.finalize().chunks_exact(4) {
            let candidate = u32::from_le_bytes(chunk.try_into().expect("four-byte chunk"));
            if candidate < cutoff {
                output.push((candidate % modulus_u32) as u16);
                if output.len() == count {
                    break;
                }
            }
        }
        counter += 1;
    }
    output
}

fn row_tag(gate_id: &[u8; 32], labels: &[&Label], body: &[u16]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"pllm.agc.row-tag.v1\0");
    hash.update(gate_id);
    for label in labels {
        hash.update(label.to_bytes());
    }
    for component in body {
        hash.update(component.to_le_bytes());
    }
    hash.finalize().into()
}

fn constant_time_equal(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right)
        .fold(0_u8, |difference, (left, right)| {
            difference | (left ^ right)
        })
        == 0
}

struct Cursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn take<const N: usize>(&mut self) -> Result<[u8; N], GarbleError> {
        let end = self
            .position
            .checked_add(N)
            .ok_or(GarbleError::InvalidGate)?;
        let value = self
            .bytes
            .get(self.position..end)
            .ok_or(GarbleError::InvalidGate)?;
        self.position = end;
        value.try_into().map_err(|_| GarbleError::InvalidGate)
    }

    fn u16(&mut self) -> Result<u16, GarbleError> {
        Ok(u16::from_le_bytes(self.take()?))
    }

    fn u8(&mut self) -> Result<u8, GarbleError> {
        Ok(self.take::<1>()?[0])
    }

    fn u32(&mut self) -> Result<u32, GarbleError> {
        Ok(u32::from_le_bytes(self.take()?))
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.position)
    }

    fn slice(&mut self, length: usize) -> Result<&'a [u8], GarbleError> {
        let end = self
            .position
            .checked_add(length)
            .ok_or(GarbleError::InvalidProgram)?;
        let value = self
            .bytes
            .get(self.position..end)
            .ok_or(GarbleError::InvalidProgram)?;
        self.position = end;
        Ok(value)
    }

    fn is_empty(&self) -> bool {
        self.position == self.bytes.len()
    }
}

fn mixed_radix_values(mut index: usize, moduli: &[Modulus]) -> Vec<u16> {
    moduli
        .iter()
        .map(|modulus| {
            let value = (index % usize::from(modulus.0)) as u16;
            index /= usize::from(modulus.0);
            value
        })
        .collect()
}

fn mixed_radix_selectors(labels: &[&Label], moduli: &[Modulus]) -> usize {
    labels
        .iter()
        .zip(moduli)
        .fold((0_usize, 1_usize), |(index, stride), (label, modulus)| {
            (
                index + label.selector() * stride,
                stride * usize::from(modulus.0),
            )
        })
        .0
}

fn gcd(mut left: u16, mut right: u16) -> u16 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn modular_inverse(value: i128, modulus: i128) -> Option<i128> {
    let (mut old_remainder, mut remainder) = (value, modulus);
    let (mut old_coefficient, mut coefficient) = (1_i128, 0_i128);
    while remainder != 0 {
        let quotient = old_remainder / remainder;
        (old_remainder, remainder) = (remainder, old_remainder - quotient * remainder);
        (old_coefficient, coefficient) = (coefficient, old_coefficient - quotient * coefficient);
    }
    (old_remainder == 1).then(|| old_coefficient.rem_euclid(modulus))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unary_projection_is_exhaustively_correct() {
        let mut garbler = Garbler::new();
        let input = garbler.wire(Modulus::new(7).unwrap()).unwrap();
        for value in 0..7 {
            let (gate, output) = garbler
                .garble_unary(&input, Modulus::new(5).unwrap(), |x| (x * x + 1) % 5)
                .unwrap();
            let result = gate.evaluate(&input.encode(value).unwrap()).unwrap();
            assert_eq!(output.decode(&result).unwrap(), (value * value + 1) % 5);
        }
    }

    #[test]
    fn binary_projection_is_exhaustively_correct() {
        let mut garbler = Garbler::new();
        let left = garbler.wire(Modulus::new(5).unwrap()).unwrap();
        let right = garbler.wire(Modulus::new(7).unwrap()).unwrap();
        for x in 0..5 {
            for y in 0..7 {
                let (gate, output) = garbler
                    .garble_binary(&left, &right, Modulus::new(11).unwrap(), |a, b| {
                        (a * b + a + b) % 11
                    })
                    .unwrap();
                let result = gate
                    .evaluate(&left.encode(x).unwrap(), &right.encode(y).unwrap())
                    .unwrap();
                assert_eq!(output.decode(&result).unwrap(), (x * y + x + y) % 11);
            }
        }
    }

    #[test]
    fn free_addition_and_scaling_are_correct() {
        let mut garbler = Garbler::new();
        let left = garbler.wire(Modulus::new(13).unwrap()).unwrap();
        let right = garbler.wire(Modulus::new(13).unwrap()).unwrap();
        let sum = garbler.add(&left, &right).unwrap();
        let scaled = garbler.scale(&left, 5).unwrap();
        for x in 0..13 {
            for y in 0..13 {
                let label =
                    evaluate_add(&left.encode(x).unwrap(), &right.encode(y).unwrap()).unwrap();
                assert_eq!(sum.decode(&label).unwrap(), (x + y) % 13);
            }
            let label = evaluate_scale(&left.encode(x).unwrap(), 5).unwrap();
            assert_eq!(scaled.decode(&label).unwrap(), (x * 5) % 13);
        }
    }

    #[test]
    fn label_width_keeps_128_hidden_bits_beside_the_selector() {
        for modulus in [2_u16, 3, 17, 131, 257, 387] {
            let modulus = Modulus::new(modulus).unwrap();
            let hidden_components = modulus.label_width() - 1;
            assert!(hidden_components as f64 * f64::from(modulus.get()).log2() >= SECURITY_BITS);
        }
    }

    #[test]
    fn free_scaling_rejects_non_units() {
        let mut garbler = Garbler::new();
        let wire = garbler.wire(Modulus::new(128).unwrap()).unwrap();
        assert!(matches!(
            garbler.scale(&wire, 2),
            Err(GarbleError::NonUnitScale {
                scalar: 2,
                modulus: 128
            })
        ));
        assert_eq!(
            evaluate_scale(&wire.encode(7).unwrap(), 2),
            Err(GarbleError::NonUnitScale {
                scalar: 2,
                modulus: 128
            })
        );
    }

    #[test]
    fn wire_labels_have_strict_binary_encoding() {
        let mut garbler = Garbler::new();
        let wire = garbler.wire(Modulus::new(17).unwrap()).unwrap();
        let label = wire.encode(9).unwrap();
        assert_eq!(Label::from_bytes(&label.to_bytes()).unwrap(), label);
        let mut malformed = label.to_bytes();
        malformed.push(0);
        assert_eq!(
            Label::from_bytes(&malformed),
            Err(GarbleError::InvalidLabel)
        );
    }

    #[test]
    fn changed_input_label_does_not_authenticate() {
        let mut garbler = Garbler::new();
        let input = garbler.wire(Modulus::new(7).unwrap()).unwrap();
        let (gate, _) = garbler
            .garble_unary(&input, Modulus::new(5).unwrap(), |x| x % 5)
            .unwrap();
        let mut changed = input.encode(3).unwrap();
        changed.components[0] = (changed.components[0] + 1) % 7;
        assert_eq!(gate.evaluate(&changed), Err(GarbleError::Authentication));
    }

    #[test]
    fn changed_gate_context_does_not_authenticate() {
        let mut garbler = Garbler::new();
        let input = garbler.wire(Modulus::new(7).unwrap()).unwrap();
        let (gate, _) = garbler
            .garble_projection_with_context(
                &[&input],
                Modulus::new(5).unwrap(),
                [7_u8; 32],
                |values| values[0] % 5,
            )
            .unwrap();
        let label = input.encode(3).unwrap();
        let mut encoded = gate.to_bytes();
        encoded[40] ^= 1;
        let changed = GarbledProjection::from_bytes(&encoded).unwrap();
        assert_eq!(
            changed.evaluate(&[&label]),
            Err(GarbleError::Authentication)
        );
    }

    #[test]
    fn rejects_invalid_domains_and_projection_outputs() {
        assert_eq!(Modulus::new(1), Err(GarbleError::InvalidModulus(1)));
        assert_eq!(Modulus::new(513), Err(GarbleError::InvalidModulus(513)));
        let mut garbler = Garbler::new();
        let input = garbler.wire(Modulus::new(3).unwrap()).unwrap();
        assert!(matches!(
            garbler.garble_unary(&input, Modulus::new(5).unwrap(), |_| 5),
            Err(GarbleError::ValueOutsideModulus { .. })
        ));
    }

    #[test]
    fn centered_crt_reconstruction_is_exact() {
        let basis = CrtBasis::new(vec![
            Modulus::new(5).unwrap(),
            Modulus::new(7).unwrap(),
            Modulus::new(11).unwrap(),
        ])
        .unwrap();
        assert_eq!(basis.product(), 385);
        for value in -192..=192 {
            assert_eq!(basis.reconstruct(&basis.residues(value)).unwrap(), value);
        }
        assert_eq!(
            CrtBasis::new(vec![Modulus::new(6).unwrap(), Modulus::new(9).unwrap()]),
            Err(GarbleError::NonCoprimeCrtBasis)
        );
    }

    #[test]
    fn joint_crt_projection_rescales_the_reconstructed_value() {
        let basis =
            CrtBasis::new(vec![Modulus::new(5).unwrap(), Modulus::new(7).unwrap()]).unwrap();
        let mut garbler = Garbler::new();
        let inputs = basis
            .moduli()
            .iter()
            .map(|modulus| garbler.wire(*modulus).unwrap())
            .collect::<Vec<_>>();
        for value in -17_i64..=17 {
            let references = inputs.iter().collect::<Vec<_>>();
            let (gate, output) = garbler
                .garble_projection(&references, Modulus::new(9).unwrap(), |residues| {
                    basis
                        .reconstruct(residues)
                        .unwrap()
                        .div_euclid(4)
                        .rem_euclid(9) as u16
                })
                .unwrap();
            let residues = basis.residues(value);
            let labels = inputs
                .iter()
                .zip(residues)
                .map(|(wire, residue)| wire.encode(residue).unwrap())
                .collect::<Vec<_>>();
            let label_refs = labels.iter().collect::<Vec<_>>();
            let result = gate.evaluate(&label_refs).unwrap();
            assert_eq!(
                output.decode(&result).unwrap(),
                value.div_euclid(4).rem_euclid(9) as u16
            );
        }
    }

    #[test]
    fn projection_size_is_bounded() {
        let mut garbler = Garbler::new();
        let inputs = (0..3)
            .map(|_| garbler.wire(Modulus::new(257).unwrap()).unwrap())
            .collect::<Vec<_>>();
        let references = inputs.iter().collect::<Vec<_>>();
        assert!(matches!(
            garbler.garble_projection(&references, Modulus::new(2).unwrap(), |_| 0),
            Err(GarbleError::ProjectionTooLarge)
        ));
    }

    #[test]
    fn projection_gate_has_strict_transport_encoding() {
        let basis =
            CrtBasis::new(vec![Modulus::new(5).unwrap(), Modulus::new(7).unwrap()]).unwrap();
        let mut garbler = Garbler::new();
        let inputs = basis
            .moduli()
            .iter()
            .map(|modulus| garbler.wire(*modulus).unwrap())
            .collect::<Vec<_>>();
        let references = inputs.iter().collect::<Vec<_>>();
        let (gate, output) = garbler
            .garble_projection(&references, Modulus::new(9).unwrap(), |residues| {
                basis.reconstruct(residues).unwrap().rem_euclid(9) as u16
            })
            .unwrap();
        let encoded = gate.to_bytes();
        let decoded_gate = GarbledProjection::from_bytes(&encoded).unwrap();
        let residues = basis.residues(-11);
        let labels = inputs
            .iter()
            .zip(residues)
            .map(|(wire, residue)| wire.encode(residue).unwrap())
            .collect::<Vec<_>>();
        let label_refs = labels.iter().collect::<Vec<_>>();
        assert_eq!(
            output
                .decode(&decoded_gate.evaluate(&label_refs).unwrap())
                .unwrap(),
            (-11_i64).rem_euclid(9) as u16
        );

        let mut trailing = encoded.clone();
        trailing.push(0);
        assert!(matches!(
            GarbledProjection::from_bytes(&trailing),
            Err(GarbleError::InvalidGate)
        ));
        assert!(matches!(
            GarbledProjection::from_bytes(&encoded[..encoded.len() - 1]),
            Err(GarbleError::InvalidGate)
        ));
    }

    #[test]
    fn transported_silu_q7_matches_the_numeric_oracle() {
        let material = prepare_silu_quadratic_q7().unwrap();
        let gate = material.gate_bytes();
        for value in -128..=128 {
            let input = material.encode(value).unwrap();
            let output = evaluate_silu_quadratic_q7(&gate, &input).unwrap();
            assert_eq!(
                material.decode(&output).unwrap(),
                pllm_core::silu_quadratic_q7(value).unwrap()
            );
        }
    }

    #[test]
    fn silu_q7_rejects_out_of_domain_and_tampered_material() {
        let material = prepare_silu_quadratic_q7().unwrap();
        assert_eq!(material.encode(129), Err(GarbleError::ValueOutsideQ7(129)));

        let gate = material.gate_bytes();
        let mut input = material.encode(0).unwrap();
        input.push(0);
        assert_eq!(
            evaluate_silu_quadratic_q7(&gate, &input),
            Err(GarbleError::InvalidLabel)
        );
    }

    #[test]
    fn gated_multiply_q7_preserves_the_silu_output_label() {
        let material = prepare_gated_multiply_q7().unwrap();
        let program = material.program_bytes();
        for (gate, up) in [(-128, -128), (-65, 127), (0, 128), (64, -96), (128, 128)] {
            let output = evaluate_gated_multiply_q7(
                &program,
                &material.encode_gate(gate).unwrap(),
                &material.encode_up(up).unwrap(),
            )
            .unwrap();
            let expected =
                pllm_core::multiply_q7(pllm_core::silu_quadratic_q7(gate).unwrap(), up).unwrap();
            assert_eq!(material.decode(&output).unwrap(), expected);
        }
    }

    #[test]
    fn compact_crt_schedule_matches_every_q7_product() {
        let digit_table = quotient_digit_table(257, 131).unwrap();
        for left in -128_i16..=128 {
            for right in -128_i16..=128 {
                let product = i32::from(left) * i32::from(right);
                let low = product.rem_euclid(257) as u16;
                let high_residue = product.rem_euclid(131) as u16;
                let difference = (i32::from(low) - i32::from(high_residue)).rem_euclid(387);
                let high = digit_table[difference as usize];
                let digit_sum = low + high;
                let nonnegative_remainder = digit_sum % 128;
                let carry = digit_sum / 128;
                let nonnegative_quotient = (2 * high + carry) % 257;
                let negative = high > 65 || (high == 65 && low >= 129);
                let signed_borrow = negative && nonnegative_remainder < 3;
                let base = (i32::from(nonnegative_quotient)
                    - 6 * i32::from(negative)
                    - i32::from(signed_borrow))
                .rem_euclid(257) as u16;
                let remainder = (i32::from(nonnegative_remainder) - 3 * i32::from(negative))
                    .rem_euclid(128) as u16;
                let signed_parity = (base % 2) ^ u16::from(negative);
                let rounded = (base
                    + u16::from(remainder > 64)
                    + u16::from(remainder == 64 && signed_parity == 1))
                    % 257;
                assert_eq!(
                    q7_centered(rounded),
                    pllm_core::multiply_q7(left, right).unwrap(),
                    "{left} * {right}"
                );
            }
        }
    }

    #[test]
    fn gated_multiply_q7_rejects_cross_lane_and_cross_material_labels() {
        let material = prepare_gated_multiply_q7_with_context([7_u8; 32]).unwrap();
        let other = prepare_gated_multiply_q7_with_context([7_u8; 32]).unwrap();
        let program = material.program_bytes();

        assert_eq!(
            evaluate_gated_multiply_q7(
                &program,
                &material.encode_up(9).unwrap(),
                &material.encode_gate(5).unwrap(),
            ),
            Err(GarbleError::Authentication)
        );
        assert_eq!(
            evaluate_gated_multiply_q7(
                &program,
                &other.encode_gate(5).unwrap(),
                &material.encode_up(9).unwrap(),
            ),
            Err(GarbleError::Authentication)
        );
        assert_eq!(
            material.encode_up(129),
            Err(GarbleError::ValueOutsideQ7(129))
        );
    }
}
