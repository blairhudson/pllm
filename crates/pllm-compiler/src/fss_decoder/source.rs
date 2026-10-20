use super::capacity::{FILE_OWNER_BYTES, SOURCE_READ_BYTES, SOURCE_READ_VALUES};
use super::*;
use pllm_core::fixed_file::{
    CertificateMode, FixedFile, FixedFileLayout, FixedFileWriter, RowStatistics,
};
use pllm_core::integer_bounds::{self, MatrixCertificate, MatrixScan};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

#[derive(Clone, Copy, Debug)]
pub enum WeightDtype {
    F32,
    F16,
    Bf16,
}

/// Explicit adapter policy: large vocabulary tables need complete-row energy,
/// while bounded projections additionally need matrix operator certificates.
/// The shared producer performs the only coefficient scan and derives all facts.
pub(super) fn certified_file_layout(
    shape: &[usize],
    capacity: Capacity,
) -> Result<FixedFileLayout, Error> {
    if capacity != Capacity::Decoder || shape.len() != 2 {
        return Err(Error::Domain);
    }
    let count = capacity.source_elements(shape)?;
    let mode = if shape[0] <= integer_bounds::MAX_ROWS && shape[1] <= integer_bounds::MAX_COLUMNS {
        CertificateMode::Matrix
    } else {
        CertificateMode::RowEnergy
    };
    FixedFileLayout::new(count as u64, shape[1] as u64, 50, 12)
        .and_then(|layout| layout.with_certificates(mode))
        .map_err(|_| Error::Resource)
}

/// Immutable, checked external public tensor. No claimed digest/range accepted.
pub struct Tensor {
    pub(super) id: String,
    pub(super) shape: Vec<usize>,
    pub(super) values: Zeroizing<Vec<i64>>,
    pub(super) range: Range,
    file: Option<FixedFile>,
}

/// Proof-only public column scaling. No execution tensor is replaced. Fields are
/// produced from the immutable weight and affine vectors through MatrixScan.
#[derive(Clone, Copy, Debug)]
pub(super) struct WeightedColumns {
    pub matrix: MatrixCertificate,
    pub beta_min: i128,
    pub beta_max: i128,
    pub beta_prefix_abs: u128,
    pub beta_squared: u128,
}

/// One authoritative decoder for legacy resident and streamed external sources.
pub(super) fn decode_ieee(dtype: WeightDtype, bits: u32) -> Result<i64, Error> {
    let value = match dtype {
        WeightDtype::F32 => f32::from_bits(bits) as f64,
        WeightDtype::Bf16 => {
            if bits > 65535 {
                return Err(Error::Domain);
            }
            f32::from_bits(bits << 16) as f64
        }
        WeightDtype::F16 => {
            if bits > 65535 {
                return Err(Error::Domain);
            }
            let sign = if bits & 32768 == 0 { 1.0 } else { -1.0 };
            let exp = (bits >> 10) & 31;
            let mant = bits & 1023;
            if exp == 31 {
                return Err(Error::Domain);
            }
            sign * if exp == 0 {
                mant as f64 * 2f64.powi(-24)
            } else {
                (1024 + mant) as f64 * 2f64.powi(exp as i32 - 25)
            }
        }
    };
    let rounded = (value * 4096.0).round_ties_even();
    if !rounded.is_finite() || rounded < -(HALF as f64) || rounded >= HALF as f64 {
        return Err(Error::Domain);
    }
    Ok(rounded as i64)
}
impl Tensor {
    /// Canonical IEEE payloads, not host byte order. Half formats reject high
    /// bits; every format rejects NaN/infinity before quantization.
    pub fn from_ieee(
        id: &str,
        shape: &[usize],
        dtype: WeightDtype,
        words: &[u32],
    ) -> Result<Self, Error> {
        if words.len() != elements(shape)? {
            return Err(Error::Domain);
        }
        let mut codes = Zeroizing::new(vector(words.len())?);
        for &bits in words {
            codes.push(decode_ieee(dtype, bits)?);
        }
        Self::take_codes(id, shape, codes)
    }
    /// Exact IEEE-754 multiplication by 2^12 followed by nearest-even rounding.
    /// Nonfinite values, signed50 overflow, and oversized arrays fail closed.
    pub fn from_f64(id: &str, shape: &[usize], values: &[f64]) -> Result<Self, Error> {
        if values.len() != elements(shape)? {
            return Err(Error::Domain);
        }
        let mut codes = Zeroizing::new(vector(values.len())?);
        for &value in values {
            let scaled = value * 4096.0;
            if !scaled.is_finite() {
                return Err(Error::Domain);
            }
            let rounded = scaled.round_ties_even();
            if rounded < -(HALF as f64) || rounded >= HALF as f64 {
                return Err(Error::Domain);
            }
            codes.push(rounded as i64);
        }
        Self::take_codes(id, shape, codes)
    }
    /// Import already quantized public Q12 codes; still verifies every value.
    pub fn from_q12(id: &str, shape: &[usize], values: &[i64]) -> Result<Self, Error> {
        if values.len() != elements(shape)? {
            return Err(Error::Domain);
        }
        let mut codes = Zeroizing::new(vector(values.len())?);
        codes.extend_from_slice(values);
        Self::take_codes(id, shape, codes)
    }
    fn take_codes(id: &str, shape: &[usize], values: Zeroizing<Vec<i64>>) -> Result<Self, Error> {
        if id.is_empty() || id.len() > 1024 || values.len() != elements(shape)? {
            return Err(Error::Domain);
        }
        let lo = *values.iter().min().ok_or(Error::Domain)?;
        let hi = *values.iter().max().ok_or(Error::Domain)?;
        let range = Range::new(lo as i128, hi as i128)?;
        let mut dims = vector(shape.len())?;
        dims.extend_from_slice(shape);
        let mut name = String::new();
        name.try_reserve_exact(id.len())
            .map_err(|_| Error::Allocation)?;
        name.push_str(id);
        Ok(Self {
            id: name,
            shape: dims,
            values,
            range,
            file: None,
        })
    }
    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// Resident payload only; file-backed tensors require bounded reads.
    pub fn values(&self) -> Option<&[i64]> {
        self.resident_values()
    }
    pub fn resident_values(&self) -> Option<&[i64]> {
        self.file.is_none().then_some(self.values.as_slice())
    }
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn is_file_backed(&self) -> bool {
        self.file.is_some()
    }
    pub(crate) fn fixed_file(&self) -> Option<&FixedFile> {
        self.file.as_ref()
    }
    fn certificate_file(&self) -> Result<Option<&FixedFile>, Error> {
        let Some(file) = &self.file else {
            return Ok(None);
        };
        let metadata = file.metadata();
        let layout = metadata.layout;
        if self.shape.len() != 2
            || self.shape[0] as u64 != layout.rows()
            || self.shape[1] as u64 != layout.row_elements()
            || layout.signed_bits() != 50
            || layout.fractional_bits() != 12
            || self.range.lo != metadata.minimum
            || self.range.hi != metadata.maximum
        {
            return Err(Error::Context);
        }
        Ok(Some(file))
    }
    /// Exact maximum over complete checkpoint-storage rows. File certificates
    /// are producer-owned; resident fixtures reuse the same shared integer scan.
    pub(super) fn maximum_row_energy(&self) -> Result<Option<u128>, Error> {
        if let Some(file) = self.certificate_file()? {
            return Ok(file.metadata().maximum_row_squared_energy);
        }
        let width = *self.shape.last().ok_or(Error::Domain)?;
        self.values
            .chunks_exact(width)
            .try_fold(Some(0u128), |maximum, row| {
                let energy = integer_bounds::row_energy(row).map_err(|_| Error::Domain)?;
                Ok(Some(maximum.unwrap_or(0).max(energy)))
            })
    }
    pub(super) fn matrix_certificate(&self) -> Result<Option<MatrixCertificate>, Error> {
        if self.shape.len() != 2 {
            return Err(Error::Domain);
        }
        if let Some(file) = self.certificate_file()? {
            return Ok(file.matrix_certificate());
        }
        let (rows, columns) = (self.shape[0], self.shape[1]);
        if rows > integer_bounds::MAX_ROWS || columns > integer_bounds::MAX_COLUMNS {
            return Ok(None);
        }
        let bytes = MatrixScan::payload_bytes(rows, columns).map_err(|_| Error::Resource)?;
        if bytes > MAX_BYTES {
            return Err(Error::Resource);
        }
        let mut scan = MatrixScan::new(rows, columns, bytes).map_err(|_| Error::Resource)?;
        scan.append(0, &self.values).map_err(|_| Error::Domain)?;
        scan.finish().map(Some).map_err(|_| Error::Domain)
    }
    /// One bounded proof pass over [output,input] weights. Column products remain
    /// exact signed integers (Q24); no gamma folding or rescaling enters execution.
    /// Unsupported shared-scan arithmetic makes this optional proof unavailable.
    pub(super) fn weighted_columns(
        &self,
        gamma: &Tensor,
        beta: Option<&Tensor>,
        scratch_limit: usize,
    ) -> Result<Option<WeightedColumns>, Error> {
        if self.shape.len() != 2 {
            return Err(Error::Context);
        }
        let (rows, columns) = (self.shape[0], self.shape[1]);
        if gamma.shape() != [columns] || beta.is_some_and(|b| b.shape() != [columns]) {
            return Err(Error::Context);
        }
        if rows > integer_bounds::MAX_ROWS || columns > integer_bounds::MAX_COLUMNS {
            return Ok(None);
        }
        let gamma = gamma.resident_values().ok_or(Error::Context)?;
        let beta = beta
            .map(|b| b.resident_values().ok_or(Error::Context))
            .transpose()?;
        self.certificate_file()?;
        let payload = MatrixScan::payload_bytes(rows, columns).map_err(|_| Error::Resource)?;
        // One source row plus one scaled row; file-reader scratch is already
        // included in Source::reservation_bytes. Shared scan prices its columns.
        let chunk = columns.min(SOURCE_READ_VALUES);
        let scratch = payload
            .checked_add(chunk.checked_mul(16).ok_or(Error::Resource)?)
            .and_then(|n| {
                n.checked_add(
                    2 * std::mem::size_of::<Vec<i64>>() + std::mem::size_of::<WeightedColumns>(),
                )
            })
            .ok_or(Error::Resource)?;
        if scratch > scratch_limit {
            return Err(Error::Resource);
        }
        let mut scan = MatrixScan::new(rows, columns, payload).map_err(|_| Error::Resource)?;
        let mut scaled = Zeroizing::new(vector(chunk)?);
        let mut values = Zeroizing::new(vector(chunk)?);
        values.resize(chunk, 0);
        let (mut beta_min, mut beta_max, mut beta_prefix_abs, mut beta_squared) =
            (i128::MAX, i128::MIN, 0u128, 0u128);
        for row in 0..rows {
            let mut b = 0i128;
            for first in (0..columns).step_by(chunk) {
                let n = chunk.min(columns - first);
                self.read_row(row, first, &mut values[..n])?;
                scaled.clear();
                for (offset, &value) in values[..n].iter().enumerate() {
                    let column = first + offset;
                    let Ok(product) = i64::try_from(i128::from(value) * i128::from(gamma[column]))
                    else {
                        return Ok(None);
                    };
                    scaled.push(product);
                    if let Some(beta) = beta {
                        let term = i128::from(value) * i128::from(beta[column]);
                        let Some(sum) = b.checked_add(term) else {
                            return Ok(None);
                        };
                        b = sum;
                        beta_prefix_abs = beta_prefix_abs
                            .max(b.unsigned_abs())
                            .max(term.unsigned_abs());
                    }
                }
                if scan.append(row * columns + first, &scaled).is_err() {
                    return Ok(None);
                }
            }
            beta_min = beta_min.min(b);
            beta_max = beta_max.max(b);
            let Some(sum) = b
                .unsigned_abs()
                .checked_mul(b.unsigned_abs())
                .and_then(|square| beta_squared.checked_add(square))
            else {
                return Ok(None);
            };
            beta_squared = sum;
        }
        let Ok(matrix) = scan.finish() else {
            return Ok(None);
        };
        Ok(Some(WeightedColumns {
            matrix,
            beta_min,
            beta_max,
            beta_prefix_abs,
            beta_squared,
        }))
    }
    pub fn range(&self) -> (i64, i64) {
        (self.range.lo, self.range.hi)
    }
    pub fn disk_bytes(&self) -> u64 {
        self.file.as_ref().map_or(0, |f| {
            let r = f.metadata().layout.resources();
            r.disk_bytes + r.row_statistics_disk_bytes
        })
    }
    pub fn row_statistics(&self, row: usize) -> Result<RowStatistics, Error> {
        let width = *self.shape.last().ok_or(Error::Domain)?;
        if row >= self.len() / width {
            return Err(Error::Domain);
        }
        if let Some(file) = &self.file {
            return file.row_statistics(row as u64).map_err(|_| Error::Context);
        }
        let mut stats = RowStatistics {
            minimum: i64::MAX,
            maximum: i64::MIN,
            positive_sum: 0,
            negative_sum: 0,
        };
        for &value in &self.values[row * width..(row + 1) * width] {
            stats.minimum = stats.minimum.min(value);
            stats.maximum = stats.maximum.max(value);
            if value >= 0 {
                stats.positive_sum += value as i128;
            } else {
                stats.negative_sum += value as i128;
            }
        }
        Ok(stats)
    }
    /// Bounded exact certificates from checkpoint-storage rows, not transposed
    /// contractions. Certificates come from the shared immutable file producer.
    pub fn read_row_statistics(
        &self,
        first: usize,
        rows: &mut [RowStatistics],
    ) -> Result<(), Error> {
        let width = *self.shape.last().ok_or(Error::Domain)?;
        if rows.is_empty()
            || rows.len() > pllm_core::fixed_file::MAX_ROW_STATISTICS
            || first
                .checked_add(rows.len())
                .is_none_or(|end| end > self.len() / width)
        {
            return Err(Error::Resource);
        }
        if let Some(file) = &self.file {
            file.read_row_statistics(first as u64, rows)
                .map_err(|_| Error::Context)
        } else {
            for (i, row) in rows.iter_mut().enumerate() {
                *row = self.row_statistics(first + i)?;
            }
            Ok(())
        }
    }
    pub fn resident_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.id.capacity()
            + self.shape.capacity() * std::mem::size_of::<usize>()
            + self.values.capacity() * 8
            + if self.file.is_some() {
                FILE_OWNER_BYTES
            } else {
                0
            }
    }
    /// Caller owns the output. At most one separately charged 64-KiB read scratch
    /// buffer is live during a synchronous call; no whole-file read is exposed.
    pub fn read_range(&self, offset: usize, output: &mut [i64]) -> Result<(), Error> {
        if output.is_empty()
            || output.len() > SOURCE_READ_VALUES
            || offset
                .checked_add(output.len())
                .is_none_or(|n| n > self.len())
        {
            return Err(Error::Resource);
        }
        if let Some(file) = &self.file {
            file.read(offset as u64, output)
                .map_err(|_| Error::Context)?;
        } else {
            output.copy_from_slice(&self.values[offset..offset + output.len()]);
        }
        Ok(())
    }
    pub fn read_rows(&self, first: usize, rows: usize, output: &mut [i64]) -> Result<(), Error> {
        if self.shape.len() != 2
            || rows == 0
            || first.checked_add(rows).is_none_or(|n| n > self.shape[0])
            || rows.checked_mul(self.shape[1]) != Some(output.len())
        {
            return Err(Error::Domain);
        }
        self.read_range(first * self.shape[1], output)
    }
    pub fn read_row(&self, row: usize, column: usize, output: &mut [i64]) -> Result<(), Error> {
        if self.shape.len() != 2
            || row >= self.shape[0]
            || column
                .checked_add(output.len())
                .is_none_or(|n| n > self.shape[1])
        {
            return Err(Error::Domain);
        }
        self.read_range(row * self.shape[1] + column, output)
    }
    fn hash_values(&self, hash: &mut Sha256) -> Result<(), Error> {
        if let Some(file) = &self.file {
            let mut bytes = vector(SOURCE_READ_BYTES.min(self.len() * 8))?;
            bytes.resize(bytes.capacity(), 0u8);
            let mut offset = 0u64;
            let length = file.metadata().layout.elements() * 8;
            while offset < length {
                let n = bytes.len().min((length - offset) as usize);
                file.read_bytes(offset, &mut bytes[..n])
                    .map_err(|_| Error::Context)?;
                hash.update(&bytes[..n]);
                offset += n as u64;
            }
        } else {
            for &v in self.values.iter() {
                hash.update(v.to_le_bytes());
            }
        }
        Ok(())
    }
}

// These are lower bounds in the original Capsule contract, not actual allocation
// claims. Keep that serialized identity while pricing new real owners separately.
pub(super) fn legacy_source_header_bytes() -> usize {
    std::mem::size_of::<(Vec<Tensor>, Vec<RotaryCoefficients>, [u8; 32], usize)>()
}
pub(super) fn legacy_tensor_header_bytes() -> usize {
    std::mem::size_of::<(String, Vec<usize>, Zeroizing<Vec<i64>>, Range)>()
}

/// The enclosing SourceBuilder consumes this owner on every append failure.
pub(super) struct TensorImport {
    name: String,
    shape: Vec<usize>,
    count: usize,
    written: usize,
    file: Option<FixedFileWriter>,
    codes: Zeroizing<Vec<i64>>,
}
impl TensorImport {
    pub(super) fn new(name: &str, shape: &[usize], capacity: Capacity) -> Result<Self, Error> {
        let count = capacity.source_elements(shape)?;
        if name.is_empty() || name.len() > 1024 {
            return Err(Error::Domain);
        }
        let mut id = String::new();
        id.try_reserve_exact(name.len())
            .map_err(|_| Error::Allocation)?;
        id.push_str(name);
        let file = if capacity.file_backed(shape) {
            let layout = certified_file_layout(shape, capacity)?;
            Some(FixedFileWriter::new(layout).map_err(|_| Error::Resource)?)
        } else {
            None
        };
        let codes = Zeroizing::new(if file.is_none() {
            vector(count)?
        } else {
            Vec::new()
        });
        Ok(Self {
            name: id,
            shape: copied(shape)?,
            count,
            written: 0,
            file,
            codes,
        })
    }
    pub(super) fn written(&self) -> usize {
        self.written
    }
    pub(super) fn complete(&self) -> bool {
        self.written == self.count
    }
    pub(super) fn append(&mut self, dtype: WeightDtype, payload: &[u8]) -> Result<(), Error> {
        let width = if matches!(dtype, WeightDtype::F32) {
            4
        } else {
            2
        };
        if payload.is_empty()
            || payload.len() > super::import::MAX_FRAME_BYTES
            || payload.len() % width != 0
            || self
                .written
                .checked_add(payload.len() / width)
                .is_none_or(|n| n > self.count)
        {
            return Err(Error::Context);
        }
        // F16/BF16 frames expand by four. Decode/write subchunks, never an entire
        // IEEE tensor or an expanded 256-KiB frame alongside a resident model.
        for chunk in payload.chunks(SOURCE_READ_VALUES * width) {
            let mut encoded = vector(chunk.len() / width * 8)?;
            for raw in chunk.chunks_exact(width) {
                let bits = if width == 4 {
                    u32::from_le_bytes(raw.try_into().map_err(|_| Error::Context)?)
                } else {
                    u16::from_le_bytes(raw.try_into().map_err(|_| Error::Context)?) as u32
                };
                let code = decode_ieee(dtype, bits)?;
                if self.file.is_some() {
                    encoded.extend_from_slice(&code.to_le_bytes());
                } else {
                    self.codes.push(code);
                }
            }
            if let Some(file) = &mut self.file {
                file.write_bytes(&encoded).map_err(|_| Error::Context)?;
            }
            self.written += chunk.len() / width;
        }
        Ok(())
    }
    pub(super) fn finish(self) -> Result<Tensor, Error> {
        if !self.complete() {
            return Err(Error::Context);
        }
        let file = self
            .file
            .map(|f| f.finish().map_err(|_| Error::Context))
            .transpose()?;
        if let Some(file) = file {
            let metadata = file.metadata();
            let range = Range::new(metadata.minimum as i128, metadata.maximum as i128)?;
            Ok(Tensor {
                id: self.name,
                shape: self.shape,
                values: self.codes,
                range,
                file: Some(file),
            })
        } else {
            Tensor::take_codes(&self.name, &self.shape, self.codes)
        }
    }
}

/// Explicit pinned finite-F32 split-half coefficients for one semantic operator.
/// The descriptor binds its exact operator ID, theta, dimensions and every Q12
/// code. It is not an arbitrary range proof or an implicit host libm generator.
pub struct RotaryCoefficients {
    pub(super) operation: String,
    pub(super) theta: u64,
    pub(super) positions: usize,
    pub(super) dimensions: usize,
    pub(super) cos: Zeroizing<Vec<i64>>,
    pub(super) sin: Zeroizing<Vec<i64>>,
    ieee_digest: [u8; 32],
}
impl RotaryCoefficients {
    pub fn operation_id(&self) -> &str {
        &self.operation
    }
    pub fn shape(&self) -> [usize; 2] {
        [self.positions, self.dimensions]
    }
    pub fn ieee_digest(&self) -> [u8; 32] {
        self.ieee_digest
    }
    pub fn from_f32(
        operation: &str,
        theta: u64,
        shape: [usize; 2],
        cos: &[f32],
        sin: &[f32],
    ) -> Result<Self, Error> {
        if theta == 0
            || shape[1] % 2 != 0
            || shape[0] > 16
            || shape[1] > 4096
            || cos.len() != elements(&shape)?
            || sin.len() != cos.len()
            || cos
                .iter()
                .chain(sin)
                .any(|v| !v.is_finite() || v.abs() > 1.0)
        {
            return Err(Error::Domain);
        }
        let mut c = Zeroizing::new(vector(cos.len())?);
        let mut s = Zeroizing::new(vector(sin.len())?);
        let mut ieee = Sha256::new();
        ieee.update(b"pllm.fss-decoder.rotary-f32-payloads.v1\0");
        for (&a, &b) in cos.iter().zip(sin) {
            ieee.update(a.to_bits().to_le_bytes());
            ieee.update(b.to_bits().to_le_bytes());
            c.push((a as f64 * 4096.0).round_ties_even() as i64);
            s.push((b as f64 * 4096.0).round_ties_even() as i64);
        }
        for p in 0..shape[0] {
            for d in 0..shape[1] / 2 {
                let i = p * shape[1] + d;
                let j = i + shape[1] / 2;
                if cos[i].to_bits() != cos[j].to_bits() || sin[i].to_bits() != sin[j].to_bits() {
                    return Err(Error::Domain);
                }
            }
        }
        if cos[..shape[1]]
            .iter()
            .any(|v| v.to_bits() != 1f32.to_bits())
            || sin[..shape[1]]
                .iter()
                .any(|v| v.to_bits() != 0f32.to_bits())
            || operation.is_empty()
            || operation.len() > 1024
        {
            return Err(Error::Domain);
        }
        let mut name = String::new();
        name.try_reserve_exact(operation.len())
            .map_err(|_| Error::Allocation)?;
        name.push_str(operation);
        Ok(Self {
            operation: name,
            theta,
            positions: shape[0],
            dimensions: shape[1],
            cos: c,
            sin: s,
            ieee_digest: ieee.finalize().into(),
        })
    }
}

/// Bounded public source owner. Content hashes include every shape and code.
pub struct Source {
    pub(super) tensors: Vec<Tensor>,
    pub(super) rotary: Vec<RotaryCoefficients>,
    digest: [u8; 32],
    bytes: usize,
    disk_bytes: u64,
    capacity: Capacity,
    checked_ranges: bool,
}
impl Source {
    pub fn new(tensors: Vec<Tensor>) -> Result<Self, Error> {
        Self::new_with_capacity(tensors, Capacity::Capsule)
    }
    pub fn new_with_capacity(mut tensors: Vec<Tensor>, capacity: Capacity) -> Result<Self, Error> {
        if tensors.is_empty() || tensors.len() > MAX_NODES {
            return Err(Error::Resource);
        }
        tensors.sort_unstable_by(|a, b| a.id.cmp(&b.id));
        let mut bytes = tensors
            .capacity()
            .checked_mul(std::mem::size_of::<Tensor>())
            .and_then(|n| n.checked_add(std::mem::size_of::<Self>()))
            .ok_or(Error::Resource)?;
        let mut disk_bytes = 0u64;
        let mut logical_bytes = 0u64;
        for (i, t) in tensors.iter().enumerate() {
            if i != 0 && tensors[i - 1].id == t.id {
                return Err(Error::Domain);
            }
            capacity.source_elements(t.shape())?;
            if t.is_file_backed() && capacity != Capacity::Decoder {
                return Err(Error::Unsupported);
            }
            disk_bytes = disk_bytes
                .checked_add(t.disk_bytes())
                .ok_or(Error::Resource)?;
            logical_bytes = logical_bytes
                .checked_add(t.len() as u64 * 8)
                .ok_or(Error::Resource)?;
            if logical_bytes > capacity.max_source_bytes()
                || disk_bytes > capacity.max_source_bytes()
            {
                return Err(Error::Resource);
            }
            bytes = bytes
                .checked_add(t.resident_bytes() - std::mem::size_of::<Tensor>())
                .ok_or(Error::Resource)?;
            if bytes > capacity.max_resident_bytes() {
                return Err(Error::Resource);
            }
        }
        if disk_bytes != 0 {
            bytes = checked_bytes(bytes, SOURCE_READ_BYTES)?;
        }
        // All logical disk/resident charges precede canonical file reads. Keep
        // exactly the old source domain and tensor ordering across capacities.
        let mut hash = Sha256::new();
        hash.update(b"pllm.fss-decoder.public-source.ring50-q12.nearest-even.v1\0");
        for t in &tensors {
            hash.update((t.id.len() as u64).to_le_bytes());
            hash.update(t.id.as_bytes());
            hash.update((t.shape.len() as u64).to_le_bytes());
            for &d in &t.shape {
                hash.update((d as u64).to_le_bytes());
            }
            t.hash_values(&mut hash)?;
        }
        Ok(Self {
            tensors,
            rotary: Vec::new(),
            digest: hash.finalize().into(),
            bytes,
            disk_bytes,
            capacity,
            checked_ranges: false,
        })
    }
    pub fn with_rotary(mut self, mut descriptors: Vec<RotaryCoefficients>) -> Result<Self, Error> {
        if !self.rotary.is_empty() || descriptors.is_empty() || descriptors.len() > MAX_NODES {
            return Err(Error::Domain);
        }
        descriptors.sort_unstable_by(|a, b| a.operation.cmp(&b.operation));
        let mut hash = Sha256::new();
        hash.update(b"pllm.fss-decoder.pinned-f32-rotary-source.v1\0");
        hash.update(self.digest);
        self.bytes = checked_bytes(
            self.bytes,
            descriptors.capacity() * std::mem::size_of::<RotaryCoefficients>(),
        )?;
        for (i, r) in descriptors.iter().enumerate() {
            if i > 0 && descriptors[i - 1].operation == r.operation {
                return Err(Error::Domain);
            }
            self.bytes = checked_bytes(
                self.bytes,
                r.operation.capacity() + 8 * (r.cos.capacity() + r.sin.capacity()),
            )?;
            hash.update((r.operation.len() as u64).to_le_bytes());
            hash.update(r.operation.as_bytes());
            for v in [r.theta, r.positions as u64, r.dimensions as u64] {
                hash.update(v.to_le_bytes());
            }
            hash.update(r.ieee_digest);
            for &v in r.cos.iter().chain(r.sin.iter()) {
                hash.update(v.to_le_bytes());
            }
        }
        self.digest = hash.finalize().into();
        self.rotary = descriptors;
        Ok(self)
    }
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
    pub fn reservation_bytes(&self) -> usize {
        self.bytes
    }
    pub fn disk_bytes(&self) -> u64 {
        self.disk_bytes
    }
    /// Actual allocation facts for live immutable source files. No path or file
    /// descriptor escapes. Missing platform facts stay absent, never estimated.
    pub fn disk_backing(&self) -> Result<Vec<pllm_core::public_file::FileBacking>, Error> {
        let count = self.tensors.len().checked_mul(2).ok_or(Error::Resource)?;
        let mut records = vector(count)?;
        for tensor in &self.tensors {
            if let Some(file) = tensor.fixed_file() {
                for backing in file
                    .backings()
                    .map_err(|_| Error::Context)?
                    .into_iter()
                    .flatten()
                {
                    records.push(backing);
                }
            }
        }
        records.sort_unstable_by_key(|b| (b.device, b.file, b.allocated_bytes));
        // Sorting allocation last retains the smaller observed fact if duplicate
        // handles see different backing-accounting instants. No extra credit.
        records.dedup_by_key(|b| (b.device, b.file));
        Ok(records)
    }
    pub fn capacity(&self) -> Capacity {
        self.capacity
    }
    /// Explicit D14 execution policy; source IEEE/Q12 content identity is unchanged.
    pub fn with_checked_ranges(mut self) -> Self {
        self.checked_ranges = true;
        self
    }
    pub fn checked_ranges(&self) -> bool {
        self.checked_ranges
    }
    pub fn tensors(&self) -> &[Tensor] {
        &self.tensors
    }
    pub(super) fn find(&self, id: &str, shape: &[usize]) -> Result<usize, Error> {
        let i = self
            .tensors
            .binary_search_by(|t| t.id.as_str().cmp(id))
            .map_err(|_| Error::Domain)?;
        if self.tensors[i].shape != shape {
            return Err(Error::Domain);
        }
        Ok(i)
    }
}
fn checked_bytes(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b)
        .filter(|&n| n <= MAX_BYTES)
        .ok_or(Error::Resource)
}

#[cfg(test)]
#[path = "source_capacity_tests.rs"]
mod capacity_tests;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn external_nearest_even_and_checked_source() {
        let t = Tensor::from_f64(
            "w",
            &[6],
            &[
                0.5 / 4096.,
                1.5 / 4096.,
                2.5 / 4096.,
                -0.5 / 4096.,
                -1.5 / 4096.,
                -2.5 / 4096.,
            ],
        )
        .unwrap();
        assert_eq!(t.values().unwrap(), &[0, 2, 2, 0, -2, -2]);
        assert!(Tensor::from_f64("w", &[1], &[f64::NAN]).is_err());
        assert!(Tensor::from_q12("w", &[1], &[1i64 << 49]).is_err());
        let a = Source::new(vec![t]).unwrap();
        let b = Source::new(vec![
            Tensor::from_q12("w", &[6], &[0, 2, 2, 0, -2, -2]).unwrap()
        ])
        .unwrap();
        assert_eq!(a.digest(), b.digest());
    }

    #[test]
    fn external_ieee_formats_and_rotary_pins() {
        for (dtype, words) in [
            (WeightDtype::F32, [0x3f800000, 0xbf000000]),
            (WeightDtype::F16, [0x3c00, 0xb800]),
            (WeightDtype::Bf16, [0x3f80, 0xbf00]),
        ] {
            assert_eq!(
                Tensor::from_ieee("w", &[2], dtype, &words)
                    .unwrap()
                    .values()
                    .unwrap(),
                &[4096, -2048]
            );
        }
        assert!(Tensor::from_ieee("w", &[1], WeightDtype::F32, &[0x7fc00000]).is_err());
        assert!(Tensor::from_ieee("w", &[1], WeightDtype::F16, &[0x10000]).is_err());
        assert!(RotaryCoefficients::from_f32("op", 10000, [1, 2], &[1., 1.], &[0., -0.]).is_err());
        assert!(RotaryCoefficients::from_f32(
            "op",
            10000,
            [2, 2],
            &[1., 1., 0.5, 0.5],
            &[0., 0., 0.25, 0.25000003]
        )
        .is_err());
        let a = RotaryCoefficients::from_f32(
            "op",
            10000,
            [2, 2],
            &[1., 1., 0.5, 0.5],
            &[0., 0., 0.25, 0.25],
        )
        .unwrap();
        let b = RotaryCoefficients::from_f32(
            "op",
            10000,
            [2, 2],
            &[1., 1., 0.50000006, 0.50000006],
            &[0., 0., 0.25, 0.25],
        )
        .unwrap();
        // Equal rounded Q12 payloads still retain distinct original F32 pins.
        assert_eq!(a.cos, b.cos);
        assert_ne!(a.ieee_digest(), b.ieee_digest());
    }
}
