//! Clean-room reference arithmetic garbling based on PLLM's cited public
//! protocol descriptions. This unreviewed research crate is disabled in all
//! production profiles and makes no cryptographic assurance claim.

use sha2::{Digest as _, Sha256};
use std::{collections::BTreeMap, error::Error, fmt};
use zeroize::Zeroize;

const SECURITY_BITS: f64 = 128.0;
const MAX_MODULUS: u16 = 257;
const TAG_BYTES: usize = 16;
const MAX_PROJECTION_ROWS: usize = 1_000_000;
const GATE_MAGIC: &[u8; 8] = b"PLLMAGC1";

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
        (SECURITY_BITS / f64::from(self.0).log2()).ceil() as usize
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
            let body = seal(&gate_id, &label_refs, &output_label);
            rows[index] = Some(make_row(&gate_id, &label_refs, body));
        }
        Ok((
            GarbledProjection {
                input_moduli: moduli,
                output_modulus,
                gate_id,
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
    rows: Vec<CipherRow>,
}

impl GarbledProjection {
    pub fn evaluate(self, inputs: &[&Label]) -> Result<Label, GarbleError> {
        if inputs.len() != self.input_moduli.len() {
            return Err(GarbleError::InvalidLabel);
        }
        for (input, modulus) in inputs.iter().zip(&self.input_moduli) {
            check_label(input, *modulus)?;
        }
        let index = mixed_radix_selectors(inputs, &self.input_moduli);
        let row = self.rows.get(index).ok_or(GarbleError::InvalidLabel)?;
        open(&self.gate_id, inputs, self.output_modulus, row)
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
            rows,
        })
    }
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
    Ok(scale_label(input, scalar))
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
    NonCoprimeCrtBasis,
    ProjectionTooLarge,
    Randomness(String),
    UnknownOutputLabel,
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
            Self::InvalidModulus(value) => write!(formatter, "modulus {value} is outside 2..=257"),
            Self::InvalidCrtResidues => formatter.write_str("CRT residues do not match the basis"),
            Self::InvalidGate => formatter.write_str("garbled gate encoding is invalid"),
            Self::NonCoprimeCrtBasis => formatter.write_str("CRT moduli must be pairwise coprime"),
            Self::ProjectionTooLarge => {
                formatter.write_str("projection exceeds the one-million-row reference limit")
            }
            Self::Randomness(message) => {
                write!(formatter, "operating-system randomness failed: {message}")
            }
            Self::UnknownOutputLabel => {
                formatter.write_str("output label is not in the decoding set")
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

    fn u32(&mut self) -> Result<u32, GarbleError> {
        Ok(u32::from_le_bytes(self.take()?))
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
    fn rejects_invalid_domains_and_projection_outputs() {
        assert_eq!(Modulus::new(1), Err(GarbleError::InvalidModulus(1)));
        assert_eq!(Modulus::new(258), Err(GarbleError::InvalidModulus(258)));
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
}
