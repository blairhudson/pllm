//! Bounded, immutable files of signed fixed-point coefficients.
//!
//! Storage and its SHA-256 digest are exactly the row-major, two's-complement
//! i64 little-endian codes, without a header, padding, or a protocol identity.
//! Width and scale are checked metadata, not part of that content digest. A
//! caller authenticates the source and reserves aggregate disk/resident headroom
//! using [`FixedFileLayout::resources`] before constructing a writer.
//! Derived row statistics stream into a second immutable file, with their own
//! content digest. Neither completed file retains a resident row table.
//! Optional energy/matrix certificates reuse the ordered integer-bounds scan
//! during production; their scratch is preflighted and released on seal.
use crate::integer_bounds::{MatrixCertificate, MatrixScan};
use crate::public_file::{FileBacking, PublicFile, PublicFileWriter, MAX_BYTES};
use crate::streamed_matrix::{self, MatrixGeometry, MatrixLayout};
use std::sync::Arc;

pub const MAX_CHUNK_VALUES: usize = 8192;
pub const MAX_CHUNK_BYTES: usize = MAX_CHUNK_VALUES * 8;
pub const ROW_STATISTICS_RECORD_BYTES: usize = 48;
pub const MAX_ROW_STATISTICS: usize = MAX_CHUNK_BYTES / ROW_STATISTICS_RECORD_BYTES;

/// Requested source-derived proof. The default adds no squared-energy arithmetic,
/// preserving signed64 files whose row or aggregate squares would overflow u128.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CertificateMode {
    #[default]
    None,
    /// Exact maximum complete-row squared energy, including large vocabularies.
    RowEnergy,
    /// Complete matrix statistics and exact maximum row energy; each dimension
    /// must fit the existing integer-bounds matrix scan limits.
    Matrix,
}

/// A rectangular view of an expected number of signed coefficients. The decoded
/// value of a code is `code * 2^-fractional_bits`; this module never quantizes it.
/// Signed width is 1..=64 and fractional bits 0..=63. Each backing file is capped
/// independently by `public_file::MAX_BYTES`; aggregate disk use is also checked.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedFileLayout {
    elements: u64,
    row_elements: u64,
    signed_bits: u8,
    fractional_bits: u8,
    certificate_mode: CertificateMode,
    certificate_payload_bytes: usize,
}

/// Deterministic buffer payload requirements, excluding small owner structs,
/// allocator overhead, OS file cache, and caller-owned input/output buffers.
/// No row table remains resident after seal. Producer buffers require at most
/// `writer_scratch_bytes + row_statistics_bytes + certificate_payload_bytes`;
/// certificate column bytes are already included in that payload. Each decoded read
/// needs at most `reader_scratch_bytes` additional bytes. Matrix contraction uses
/// the separate fixed `streamed_matrix::READ_BUFFER_BYTES` scratch bound.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedFileResources {
    /// Coefficient file only, preserving the original meaning of this field.
    pub disk_bytes: u64,
    pub row_statistics_disk_bytes: u64,
    pub total_disk_bytes: u64,
    /// One canonical statistics record of scratch, independent of row count.
    /// No statistics payload is retained in memory after seal.
    pub row_statistics_bytes: usize,
    pub writer_scratch_bytes: usize,
    pub reader_scratch_bytes: usize,
    /// Optional scan's column allocation plus bounded owner/working state.
    /// Released at seal; completed certificates are constant-size metadata.
    pub certificate_payload_bytes: usize,
    /// Subset of certificate_payload_bytes; zero for None/RowEnergy modes.
    pub certificate_column_bytes: usize,
}

impl FixedFileLayout {
    pub fn new(
        elements: u64,
        row_elements: u64,
        signed_bits: u8,
        fractional_bits: u8,
    ) -> Result<Self, String> {
        let disk_bytes = elements.checked_mul(8).ok_or("fixed file size overflow")?;
        if elements == 0 || disk_bytes > MAX_BYTES {
            return Err("fixed file size exceeds bounds".into());
        }
        if row_elements == 0 || elements % row_elements != 0 {
            return Err("fixed file row shape differs from element count".into());
        }
        if !(1..=64).contains(&signed_bits) || fractional_bits > 63 {
            return Err("fixed file signed width or scale exceeds bounds".into());
        }
        let statistics_bytes = (elements / row_elements)
            .checked_mul(ROW_STATISTICS_RECORD_BYTES as u64)
            .ok_or("fixed file row statistics size overflow")?;
        if statistics_bytes > MAX_BYTES {
            return Err("fixed file row statistics exceed bounds".into());
        }
        disk_bytes
            .checked_add(statistics_bytes)
            .ok_or("fixed file total disk size overflow")?;
        Ok(Self {
            elements,
            row_elements,
            signed_bits,
            fractional_bits,
            certificate_mode: CertificateMode::None,
            certificate_payload_bytes: 0,
        })
    }

    /// Select certification and preflight it before any allocation. No matrix
    /// fallback is implicit: unsupported matrix geometry returns an error.
    pub fn with_certificates(mut self, mode: CertificateMode) -> Result<Self, String> {
        let rows =
            usize::try_from(self.rows()).map_err(|_| "fixed file certificate row overflow")?;
        let columns = usize::try_from(self.row_elements)
            .map_err(|_| "fixed file certificate column overflow")?;
        let payload = match mode {
            CertificateMode::None => 0,
            CertificateMode::RowEnergy => {
                MatrixScan::row_energy_payload_bytes(rows, columns).map_err(|e| e.to_string())?
            }
            CertificateMode::Matrix => {
                MatrixScan::payload_bytes(rows, columns).map_err(|e| e.to_string())?
            }
        };
        self.resources()
            .writer_scratch_bytes
            .checked_add(ROW_STATISTICS_RECORD_BYTES)
            .and_then(|n| n.checked_add(payload))
            .ok_or("fixed file producer payload overflow")?;
        self.certificate_mode = mode;
        self.certificate_payload_bytes = payload;
        Ok(self)
    }

    pub fn certificate_mode(self) -> CertificateMode {
        self.certificate_mode
    }

    pub fn elements(self) -> u64 {
        self.elements
    }
    pub fn row_elements(self) -> u64 {
        self.row_elements
    }
    pub fn rows(self) -> u64 {
        self.elements / self.row_elements
    }
    pub fn signed_bits(self) -> u8 {
        self.signed_bits
    }
    pub fn fractional_bits(self) -> u8 {
        self.fractional_bits
    }
    /// No allocation or filesystem access; all arithmetic was checked by `new`.
    pub fn resources(self) -> FixedFileResources {
        let disk_bytes = self.elements * 8;
        let row_statistics_disk_bytes = self.rows() * ROW_STATISTICS_RECORD_BYTES as u64;
        let scratch_bytes = disk_bytes.min(MAX_CHUNK_BYTES as u64) as usize;
        let statistics_read_bytes = row_statistics_disk_bytes
            .min((MAX_ROW_STATISTICS * ROW_STATISTICS_RECORD_BYTES) as u64)
            as usize;
        FixedFileResources {
            disk_bytes,
            row_statistics_disk_bytes,
            total_disk_bytes: disk_bytes + row_statistics_disk_bytes,
            row_statistics_bytes: ROW_STATISTICS_RECORD_BYTES,
            writer_scratch_bytes: scratch_bytes,
            reader_scratch_bytes: scratch_bytes.max(statistics_read_bytes),
            certificate_payload_bytes: self.certificate_payload_bytes,
            certificate_column_bytes: if self.certificate_mode == CertificateMode::Matrix {
                self.row_elements as usize * std::mem::size_of::<u128>()
            } else {
                0
            },
        }
    }
}

/// Exact extrema and signed sums for a complete row. Sums use i128 so even a row
/// spanning the maximum file with signed64 coefficients cannot overflow.
/// Canonical records contain minimum i64LE, maximum i64LE, positive sum i128LE,
/// then negative sum i128LE, totaling exactly 48 bytes without padding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RowStatistics {
    pub minimum: i64,
    pub maximum: i64,
    pub positive_sum: i128,
    pub negative_sum: i128,
}

impl RowStatistics {
    fn empty() -> Self {
        Self {
            minimum: i64::MAX,
            maximum: i64::MIN,
            positive_sum: 0,
            negative_sum: 0,
        }
    }

    fn encode(self) -> [u8; ROW_STATISTICS_RECORD_BYTES] {
        let mut bytes = [0; ROW_STATISTICS_RECORD_BYTES];
        bytes[..8].copy_from_slice(&self.minimum.to_le_bytes());
        bytes[8..16].copy_from_slice(&self.maximum.to_le_bytes());
        bytes[16..32].copy_from_slice(&self.positive_sum.to_le_bytes());
        bytes[32..].copy_from_slice(&self.negative_sum.to_le_bytes());
        bytes
    }

    fn decode(bytes: &[u8]) -> Self {
        Self {
            minimum: i64::from_le_bytes(bytes[..8].try_into().unwrap()),
            maximum: i64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            positive_sum: i128::from_le_bytes(bytes[16..32].try_into().unwrap()),
            negative_sum: i128::from_le_bytes(bytes[32..].try_into().unwrap()),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedFileMetadata {
    pub layout: FixedFileLayout,
    pub minimum: i64,
    pub maximum: i64,
    pub digest: [u8; 32],
    pub row_statistics_digest: [u8; 32],
    /// Exact source-derived maximum, only when certification was requested.
    pub maximum_row_squared_energy: Option<u128>,
}

struct FixedFileInner {
    file: PublicFile,
    layout: FixedFileLayout,
    minimum: i64,
    maximum: i64,
    rows: PublicFile,
    maximum_row_squared_energy: Option<u128>,
    matrix_certificate: Option<MatrixCertificate>,
}

/// Clones share both immutable files. No path, file descriptor, whole coefficient
/// or statistics vector, or writable view is exposed.
#[derive(Clone)]
pub struct FixedFile {
    inner: Arc<FixedFileInner>,
}

impl FixedFile {
    /// Physical backing observations in [coefficients, row_statistics] order.
    /// Both use PublicFile's bounded fstat observer; neither observation retains
    /// an owner or exposes a handle. Non-Unix entries are None (no credit).
    pub fn backings(&self) -> Result<[Option<FileBacking>; 2], String> {
        Ok([self.inner.file.backing()?, self.inner.rows.backing()?])
    }

    pub fn metadata(&self) -> FixedFileMetadata {
        FixedFileMetadata {
            layout: self.inner.layout,
            minimum: self.inner.minimum,
            maximum: self.inner.maximum,
            digest: self.inner.file.digest(),
            row_statistics_digest: self.inner.rows.digest(),
            maximum_row_squared_energy: self.inner.maximum_row_squared_energy,
        }
    }

    /// A read-only certificate computed from every stored code during production.
    /// None means matrix certification was not requested, never a fabricated bound.
    pub fn matrix_certificate(&self) -> Option<MatrixCertificate> {
        self.inner.matrix_certificate
    }

    pub fn row_statistics(&self, row: u64) -> Result<RowStatistics, String> {
        if row >= self.inner.layout.rows() {
            return Err("fixed file row exceeds bounds".into());
        }
        let mut bytes = [0; ROW_STATISTICS_RECORD_BYTES];
        self.inner
            .rows
            .read(row * ROW_STATISTICS_RECORD_BYTES as u64, &mut bytes)?;
        Ok(RowStatistics::decode(&bytes))
    }

    /// Read 1..=MAX_ROW_STATISTICS consecutive records using at most 64 KiB of
    /// fallible scratch and one bounded file read. `first` is a row index.
    pub fn read_row_statistics(
        &self,
        first: u64,
        rows: &mut [RowStatistics],
    ) -> Result<(), String> {
        if rows.is_empty()
            || rows.len() > MAX_ROW_STATISTICS
            || first
                .checked_add(rows.len() as u64)
                .is_none_or(|end| end > self.inner.layout.rows())
        {
            return Err("fixed file statistics read exceeds bounds".into());
        }
        let mut bytes = buffer(rows.len() * ROW_STATISTICS_RECORD_BYTES)?;
        self.inner
            .rows
            .read(first * ROW_STATISTICS_RECORD_BYTES as u64, &mut bytes)?;
        for (row, record) in rows
            .iter_mut()
            .zip(bytes.chunks_exact(ROW_STATISTICS_RECORD_BYTES))
        {
            *row = RowStatistics::decode(record);
        }
        Ok(())
    }

    /// Read at an element offset, with caller-owned output of 1..=8192 codes.
    pub fn read(&self, offset: u64, values: &mut [i64]) -> Result<(), String> {
        if values.is_empty()
            || values.len() > MAX_CHUNK_VALUES
            || offset
                .checked_add(values.len() as u64)
                .is_none_or(|end| end > self.inner.layout.elements)
        {
            return Err("fixed file read exceeds bounds".into());
        }
        let mut bytes = buffer(values.len() * 8)?;
        self.inner.file.read(offset * 8, &mut bytes)?;
        for (value, code) in values.iter_mut().zip(bytes.chunks_exact(8)) {
            *value = i64::from_le_bytes(code.try_into().unwrap());
        }
        Ok(())
    }

    /// Read canonical bytes for streaming external identities. Byte offset and
    /// length must be multiples of eight; output is bounded by 64 KiB.
    pub fn read_bytes(&self, offset: u64, bytes: &mut [u8]) -> Result<(), String> {
        if offset % 8 != 0
            || bytes.is_empty()
            || bytes.len() % 8 != 0
            || bytes.len() > MAX_CHUNK_BYTES
        {
            return Err("fixed file byte read is unaligned or exceeds bounds".into());
        }
        self.inner.file.read(offset, bytes)
    }

    /// Read a bounded column range within one row, including rows larger than a
    /// chunk. `column` is an element offset within the specified row.
    pub fn read_row(&self, row: u64, column: u64, values: &mut [i64]) -> Result<(), String> {
        let layout = self.inner.layout;
        if row >= layout.rows()
            || column
                .checked_add(values.len() as u64)
                .is_none_or(|end| end > layout.row_elements)
        {
            return Err("fixed file row read exceeds bounds".into());
        }
        self.read(row * layout.row_elements + column, values)
    }

    /// Replace output components with complete wrapping ring64 products. Stored
    /// rows are output channels, so only `MatrixLayout::OutputInput` is accepted;
    /// input/output storage needs different contraction-column certificates.
    ///
    /// Inputs are [rows, row_elements] and outputs [rows, stored_rows]. One public
    /// coefficient scan is shared across one or two components. Scratch is exactly
    /// `streamed_matrix::READ_BUFFER_BYTES`, independent of tensor size. Numeric
    /// domain proofs, narrower-ring reduction and final rescaling belong to the
    /// caller; this operation never shifts, rounds, or reads row statistics.
    /// Bounds failures leave outputs untouched; read failures zero every output.
    pub fn multiply_ring64(
        &self,
        inputs: &[&[u64]],
        outputs: &mut [&mut [u64]],
        rows: usize,
        layout: MatrixLayout,
    ) -> Result<(), streamed_matrix::Error> {
        if layout != MatrixLayout::OutputInput {
            return Err(streamed_matrix::Error::Bounds);
        }
        let inner = usize::try_from(self.inner.layout.row_elements)
            .map_err(|_| streamed_matrix::Error::Bounds)?;
        let columns = usize::try_from(self.inner.layout.rows())
            .map_err(|_| streamed_matrix::Error::Bounds)?;
        let geometry = MatrixGeometry::new(rows, inner, columns, layout)?;
        streamed_matrix::multiply(&self.inner.file, inputs, outputs, geometry)
    }

    /// Accumulate a public range of stored rows using private arithmetic selector
    /// components. Reuses the same bounded reader as public matrix products; no
    /// selector controls a file offset, and no rescaling occurs between tiles.
    pub fn accumulate_rows_ring64(
        &self,
        selectors: &[&[u64]],
        outputs: &mut [&mut [u64]],
        rows: usize,
        start: usize,
        count: usize,
    ) -> Result<(), streamed_matrix::Error> {
        let vocabulary = usize::try_from(self.inner.layout.rows())
            .map_err(|_| streamed_matrix::Error::Bounds)?;
        let width = usize::try_from(self.inner.layout.row_elements)
            .map_err(|_| streamed_matrix::Error::Bounds)?;
        let geometry =
            streamed_matrix::SelectionGeometry::new(rows, vocabulary, width, start, count)?;
        streamed_matrix::accumulate_rows(&self.inner.file, selectors, outputs, geometry)
    }
}

struct Pending {
    file: PublicFileWriter,
    layout: FixedFileLayout,
    scratch: Vec<u8>,
    rows: PublicFileWriter,
    rows_written: u64,
    row: RowStatistics,
    row_written: u64,
    written: u64,
    minimum: i64,
    maximum: i64,
    certificate: Option<MatrixScan>,
}

impl Pending {
    fn check_chunk(&self, count: usize) -> Result<(), String> {
        if count == 0
            || count > MAX_CHUNK_VALUES
            || self
                .written
                .checked_add(count as u64)
                .is_none_or(|end| end > self.layout.elements)
        {
            return Err("fixed file chunk exceeds bounds".into());
        }
        Ok(())
    }

    fn record(&mut self, value: i64) -> Result<(), String> {
        let half = 1_i128 << (self.layout.signed_bits - 1);
        if (value as i128) < -half || (value as i128) >= half {
            return Err("fixed file coefficient exceeds signed width".into());
        }
        self.minimum = self.minimum.min(value);
        self.maximum = self.maximum.max(value);
        self.row.minimum = self.row.minimum.min(value);
        self.row.maximum = self.row.maximum.max(value);
        let sum = if value >= 0 {
            &mut self.row.positive_sum
        } else {
            &mut self.row.negative_sum
        };
        *sum = sum
            .checked_add(value as i128)
            .ok_or("fixed file row sum overflow")?;
        self.written += 1;
        self.row_written += 1;
        if self.row_written == self.layout.row_elements {
            self.rows.write(&self.row.encode())?;
            self.rows_written += 1;
            self.row = RowStatistics::empty();
            self.row_written = 0;
        }
        Ok(())
    }
}

/// Sequential native producer. Any failed append permanently burns the pending
/// anonymous files; dropping or cancelling the writer does the same.
pub struct FixedFileWriter {
    pending: Option<Pending>,
}

impl FixedFileWriter {
    pub fn new(layout: FixedFileLayout) -> Result<Self, String> {
        let resources = layout.resources();
        let scratch = buffer(resources.writer_scratch_bytes)?;
        let certificate = match layout.certificate_mode {
            CertificateMode::None => None,
            CertificateMode::RowEnergy => Some(
                MatrixScan::new_row_energy(
                    layout.rows() as usize,
                    layout.row_elements as usize,
                    resources.certificate_payload_bytes,
                )
                .map_err(|e| e.to_string())?,
            ),
            CertificateMode::Matrix => Some(
                MatrixScan::new(
                    layout.rows() as usize,
                    layout.row_elements as usize,
                    resources.certificate_payload_bytes,
                )
                .map_err(|e| e.to_string())?,
            ),
        };
        // All dimensions, resource arithmetic and fallible buffers precede any
        // file allocation. The caller reserves overall headroom for these counts.
        let file = PublicFileWriter::new_computed(resources.disk_bytes)?;
        let rows = PublicFileWriter::new_computed(resources.row_statistics_disk_bytes)?;
        Ok(Self {
            pending: Some(Pending {
                file,
                layout,
                scratch,
                rows,
                rows_written: 0,
                row: RowStatistics::empty(),
                row_written: 0,
                written: 0,
                minimum: i64::MAX,
                maximum: i64::MIN,
                certificate,
            }),
        })
    }

    /// Append 1..=8192 signed codes, serialized as canonical i64 little-endian.
    pub fn write(&mut self, values: &[i64]) -> Result<(), String> {
        let mut pending = self.pending.take().ok_or("fixed file writer is closed")?;
        pending.check_chunk(values.len())?;
        let offset = pending.written as usize;
        for (i, &value) in values.iter().enumerate() {
            pending.record(value)?;
            pending.scratch[i * 8..i * 8 + 8].copy_from_slice(&value.to_le_bytes());
        }
        if let Some(scan) = &mut pending.certificate {
            scan.append(offset, values).map_err(|e| e.to_string())?;
        }
        pending.file.write(&pending.scratch[..values.len() * 8])?;
        self.pending = Some(pending);
        Ok(())
    }

    /// Append 8..=65536 canonical bytes. Partial codes and coefficients outside
    /// the declared signed width are errors; no dtype coercion is performed.
    pub fn write_bytes(&mut self, bytes: &[u8]) -> Result<(), String> {
        let mut pending = self.pending.take().ok_or("fixed file writer is closed")?;
        if bytes.len() % 8 != 0 {
            return Err("fixed file chunk contains a partial i64 code".into());
        }
        pending.check_chunk(bytes.len() / 8)?;
        let offset = pending.written as usize;
        for code in bytes.chunks_exact(8) {
            pending.record(i64::from_le_bytes(code.try_into().unwrap()))?;
        }
        if let Some(scan) = &mut pending.certificate {
            scan.append_iter(
                offset,
                bytes
                    .chunks_exact(8)
                    .map(|code| i64::from_le_bytes(code.try_into().unwrap())),
            )
            .map_err(|e| e.to_string())?;
        }
        pending.file.write(bytes)?;
        self.pending = Some(pending);
        Ok(())
    }

    pub fn cancel(&mut self) {
        self.pending = None;
    }

    pub fn finish(mut self) -> Result<FixedFile, String> {
        let pending = self.pending.take().ok_or("fixed file writer is closed")?;
        if pending.written != pending.layout.elements
            || pending.row_written != 0
            || pending.rows_written != pending.layout.rows()
        {
            return Err("fixed file element count differs".into());
        }
        let (maximum_row_squared_energy, matrix_certificate) =
            match (pending.layout.certificate_mode, pending.certificate) {
                (CertificateMode::None, None) => (None, None),
                (CertificateMode::RowEnergy, Some(scan)) => {
                    let certificate = scan.finish_row_energy().map_err(|e| e.to_string())?;
                    (Some(certificate.maximum_row_squared_energy()), None)
                }
                (CertificateMode::Matrix, Some(scan)) => {
                    let certificate = scan.finish().map_err(|e| e.to_string())?;
                    (
                        Some(certificate.maximum_row_squared_energy()),
                        Some(certificate),
                    )
                }
                _ => return Err("fixed file certificate state differs".into()),
            };
        let file = pending.file.finish()?;
        let rows = pending.rows.finish()?;
        Ok(FixedFile {
            inner: Arc::new(FixedFileInner {
                file,
                layout: pending.layout,
                minimum: pending.minimum,
                maximum: pending.maximum,
                rows,
                maximum_row_squared_energy,
                matrix_certificate,
            }),
        })
    }
}

fn buffer(size: usize) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(size)
        .map_err(|_| "fixed file scratch allocation failed")?;
    bytes.resize(size, 0);
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;
    use sha2::{Digest, Sha256};

    fn layout(elements: u64, row_elements: u64) -> FixedFileLayout {
        FixedFileLayout::new(elements, row_elements, 50, 12).unwrap()
    }

    #[test]
    fn backing_observations_report_both_files_and_follow_shared_owner_lifetime() {
        let mut writer = FixedFileWriter::new(layout(4, 2)).unwrap();
        writer.write(&[1, -2, 3, -4]).unwrap();
        let file = writer.finish().unwrap();
        let metadata = file.metadata();
        let coefficient_owner = Arc::downgrade(&file.inner.file.file());
        let statistics_owner = Arc::downgrade(&file.inner.rows.file());
        let backings = file.backings().unwrap();
        let ids =
            |backings: [Option<FileBacking>; 2]| backings.map(|b| b.map(|b| (b.device, b.file)));
        assert_eq!(
            ids(backings),
            ids([
                file.inner.file.backing().unwrap(),
                file.inner.rows.backing().unwrap()
            ])
        );
        #[cfg(unix)]
        assert_ne!(ids(backings)[0].unwrap(), ids(backings)[1].unwrap());
        #[cfg(not(unix))]
        assert_eq!(backings, [None, None]);
        let duplicate = file.clone();
        drop(file);
        assert_eq!(ids(duplicate.backings().unwrap()), ids(backings));
        assert_eq!(duplicate.metadata(), metadata);
        let mut values = [0; 4];
        duplicate.read(0, &mut values).unwrap();
        assert_eq!(values, [1, -2, 3, -4]);
        assert_eq!(duplicate.row_statistics(1).unwrap().negative_sum, -4);
        drop(duplicate);
        assert!(coefficient_owner.upgrade().is_none());
        assert!(statistics_owner.upgrade().is_none());
    }

    #[test]
    fn opt_in_certificates_are_exact_and_preserve_both_content_digests() {
        let values = [1_i64, -2, 3, -4, 0, 5];
        let base = layout(6, 3);
        let mut writer = FixedFileWriter::new(base).unwrap();
        writer.write(&values).unwrap();
        let original = writer.finish().unwrap();
        assert_eq!(original.metadata().maximum_row_squared_energy, None);
        assert_eq!(original.matrix_certificate(), None);
        assert_eq!(base.resources().certificate_payload_bytes, 0);
        assert_eq!(base.resources().certificate_column_bytes, 0);
        for mode in [CertificateMode::RowEnergy, CertificateMode::Matrix] {
            let selected = base.with_certificates(mode).unwrap();
            assert_eq!(selected.certificate_mode(), mode);
            let resources = selected.resources();
            assert_eq!(
                resources.total_disk_bytes,
                base.resources().total_disk_bytes
            );
            assert!(resources.certificate_payload_bytes > 0);
            assert_eq!(
                resources.certificate_column_bytes,
                if mode == CertificateMode::Matrix {
                    48
                } else {
                    0
                }
            );
            let mut writer = FixedFileWriter::new(selected).unwrap();
            writer.write(&values[..2]).unwrap();
            let bytes: Vec<_> = values[2..5].iter().flat_map(|v| v.to_le_bytes()).collect();
            writer.write_bytes(&bytes).unwrap();
            writer.write(&values[5..]).unwrap();
            let file = writer.finish().unwrap();
            assert_eq!(file.metadata().maximum_row_squared_energy, Some(41));
            assert_eq!(file.metadata().digest, original.metadata().digest);
            assert_eq!(
                file.metadata().row_statistics_digest,
                original.metadata().row_statistics_digest
            );
            for row in 0..2 {
                assert_eq!(file.row_statistics(row), original.row_statistics(row));
            }
            if mode == CertificateMode::Matrix {
                let certificate = file.matrix_certificate().unwrap();
                assert_eq!((certificate.rows(), certificate.columns()), (2, 3));
                assert_eq!(certificate.maximum_row_l1(), 9);
                assert_eq!(certificate.maximum_column_l1(), 8);
                assert_eq!(certificate.frobenius_squared(), 55);
                assert_eq!(certificate.maximum_row_squared_energy(), 41);
                assert_eq!(certificate.operator_squared(), Ok(55));
            } else {
                assert_eq!(file.matrix_certificate(), None);
            }
            let clone = file.clone();
            drop(file);
            assert_eq!(clone.metadata().maximum_row_squared_energy, Some(41));
        }
    }

    #[test]
    fn certificate_geometry_and_column_payload_are_preflighted_for_large_row_files() {
        let side = crate::integer_bounds::MAX_COLUMNS as u64;
        let largest_matrix = layout(side * side, side)
            .with_certificates(CertificateMode::Matrix)
            .unwrap();
        assert_eq!(
            largest_matrix.resources().certificate_column_bytes,
            side as usize * 16
        );
        assert_eq!(
            largest_matrix.resources().certificate_payload_bytes,
            MatrixScan::payload_bytes(side as usize, side as usize).unwrap()
        );
        assert!(layout(side + 1, side + 1)
            .with_certificates(CertificateMode::Matrix)
            .is_err());
        let rows = crate::integer_bounds::MAX_ROWS as u64 + 3;
        let base = layout(rows * 2, 2);
        assert!(base.with_certificates(CertificateMode::Matrix).is_err());
        let selected = base.with_certificates(CertificateMode::RowEnergy).unwrap();
        assert_eq!(selected.resources().certificate_column_bytes, 0);
        assert_eq!(
            selected.resources().certificate_payload_bytes,
            layout(2, 2)
                .with_certificates(CertificateMode::RowEnergy)
                .unwrap()
                .resources()
                .certificate_payload_bytes
        );
        let mut writer = FixedFileWriter::new(selected).unwrap();
        let codes: Vec<_> = (0..MAX_CHUNK_VALUES)
            .map(|i| if i % 2 == 0 { -3_i64 } else { 4 })
            .collect();
        let bytes: Vec<_> = codes.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut written = 0;
        while written < selected.elements() {
            let count = (selected.elements() - written).min(MAX_CHUNK_VALUES as u64) as usize;
            writer.write_bytes(&bytes[..count * 8]).unwrap();
            written += count as u64;
        }
        let file = writer.finish().unwrap();
        assert_eq!(file.metadata().maximum_row_squared_energy, Some(25));
        assert_eq!(file.matrix_certificate(), None);
        assert_eq!(file.row_statistics(rows - 1).unwrap().minimum, -3);
    }

    #[test]
    fn energy_overflow_is_opt_in_and_requested_scans_fail_closed() {
        for columns in [2, 4] {
            let base = FixedFileLayout::new(4, columns, 64, 0).unwrap();
            let mut writer = FixedFileWriter::new(base).unwrap();
            writer.write(&[i64::MIN; 4]).unwrap();
            let unproved = writer.finish().unwrap();
            assert_eq!(unproved.metadata().maximum_row_squared_energy, None);
            for mode in [CertificateMode::RowEnergy, CertificateMode::Matrix] {
                let selected = base.with_certificates(mode).unwrap();
                let mut writer = FixedFileWriter::new(selected).unwrap();
                writer.write(&[i64::MIN; 3]).unwrap();
                let result = writer.write_bytes(&i64::MIN.to_le_bytes());
                if columns == 2 && mode == CertificateMode::RowEnergy {
                    result.unwrap();
                    let proved = writer.finish().unwrap();
                    assert_eq!(proved.metadata().maximum_row_squared_energy, Some(1 << 127));
                    assert_eq!(proved.metadata().digest, unproved.metadata().digest);
                    assert_eq!(
                        proved.metadata().row_statistics_digest,
                        unproved.metadata().row_statistics_digest
                    );
                } else {
                    assert!(result.is_err());
                    assert!(writer.pending.is_none());
                    assert!(writer.write(&[0]).is_err());
                    assert!(writer.finish().is_err());
                }
            }
        }
        for mode in [CertificateMode::RowEnergy, CertificateMode::Matrix] {
            let selected = layout(4, 2).with_certificates(mode).unwrap();
            let mut cancelled = FixedFileWriter::new(selected).unwrap();
            cancelled.write(&[1, 2]).unwrap();
            cancelled.cancel();
            assert!(cancelled.pending.is_none());
            assert!(cancelled.finish().is_err());
            let mut incomplete = FixedFileWriter::new(selected).unwrap();
            incomplete.write(&[1, 2]).unwrap();
            assert!(incomplete.finish().is_err());
            let mut cancelled_scan = FixedFileWriter::new(selected).unwrap();
            cancelled_scan.write(&[1, 2, 3, 4]).unwrap();
            cancelled_scan
                .pending
                .as_mut()
                .unwrap()
                .certificate
                .as_mut()
                .unwrap()
                .cancel();
            assert!(cancelled_scan.finish().is_err());
        }
    }

    #[test]
    fn ring64_contraction_matches_independent_complete_integer_sums() {
        for inner in [3, MAX_CHUNK_VALUES + 1] {
            let (rows, columns) = (2, 3);
            let weights: Vec<_> = (0..inner * columns)
                .map(|i| [i64::MIN, i64::MAX, -1, 0, 7][i % 5])
                .collect();
            let shape = FixedFileLayout::new(weights.len() as u64, inner as u64, 64, 17).unwrap();
            let mut writer = FixedFileWriter::new(shape).unwrap();
            for chunk in weights.chunks(MAX_CHUNK_VALUES) {
                writer.write(chunk).unwrap();
            }
            let file = writer.finish().unwrap();
            let metadata = file.metadata();
            let statistics = file.row_statistics(1).unwrap();
            for components in 1..=2 {
                let inputs: Vec<Vec<u64>> = (0..components)
                    .map(|c| {
                        (0..rows * inner)
                            .map(|i| [u64::MAX, 1 << 63, 0, 1, 23][(i + c) % 5])
                            .collect()
                    })
                    .collect();
                let mut outputs = vec![vec![37; rows * columns]; components];
                file.multiply_ring64(
                    &inputs.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                    &mut outputs
                        .iter_mut()
                        .map(Vec::as_mut_slice)
                        .collect::<Vec<_>>(),
                    rows,
                    MatrixLayout::OutputInput,
                )
                .unwrap();
                let modulus = BigInt::from(1u8) << 64;
                for component in 0..components {
                    for row in 0..rows {
                        for column in 0..columns {
                            let sum: BigInt = (0..inner)
                                .map(|k| {
                                    BigInt::from(inputs[component][row * inner + k])
                                        * BigInt::from(weights[column * inner + k])
                                })
                                .sum();
                            let reduced: BigInt = ((sum % &modulus) + &modulus) % &modulus;
                            let expected = reduced.to_u64_digits().1.first().copied().unwrap_or(0);
                            assert_eq!(outputs[component][row * columns + column], expected);
                        }
                    }
                }
            }
            assert_eq!(file.metadata(), metadata);
            assert_eq!(file.row_statistics(1).unwrap(), statistics);
        }
    }

    #[test]
    fn contraction_rejects_unsupported_orientation_and_bad_geometry_before_mutation() {
        let mut writer = FixedFileWriter::new(layout(6, 3)).unwrap();
        writer.write(&[1, 2, 3, 4, 5, 6]).unwrap();
        let file = writer.finish().unwrap();
        for (rows, orientation, input) in [
            (1, MatrixLayout::InputOutput, vec![7, 8, 9]),
            (0, MatrixLayout::OutputInput, vec![7, 8, 9]),
            (usize::MAX, MatrixLayout::OutputInput, vec![7, 8, 9]),
            (1, MatrixLayout::OutputInput, vec![7, 8]),
        ] {
            let mut output = [91, 92];
            assert_eq!(
                file.multiply_ring64(&[&input], &mut [&mut output], rows, orientation),
                Err(streamed_matrix::Error::Bounds)
            );
            assert_eq!(output, [91, 92]);
        }
        let mut output = [91, 92];
        file.multiply_ring64(
            &[&[7, 8, 9]],
            &mut [&mut output],
            1,
            MatrixLayout::OutputInput,
        )
        .unwrap();
        assert_eq!(output, [50, 122]);
    }

    #[test]
    fn layout_preflights_disk_and_resident_buffers_without_file_allocation() {
        for (elements, row_elements, width, scale) in [
            (0, 1, 50, 12),
            (1, 0, 50, 12),
            (6, 4, 50, 12),
            (6, u64::MAX, 50, 12),
            (1, 1, 0, 12),
            (1, 1, 65, 12),
            (1, 1, 50, 64),
            (MAX_BYTES / 8 + 1, 1, 50, 12),
            (u64::MAX, 1, 50, 12),
            (MAX_BYTES / 8, 1, 50, 12),
        ] {
            assert!(FixedFileLayout::new(elements, row_elements, width, scale).is_err());
        }
        let largest = layout(MAX_BYTES / 8, MAX_BYTES / 8);
        assert_eq!(largest.resources().disk_bytes, MAX_BYTES);
        assert_eq!(largest.resources().writer_scratch_bytes, MAX_CHUNK_BYTES);
        assert_eq!(largest.resources().reader_scratch_bytes, MAX_CHUNK_BYTES);
        assert_eq!(largest.resources().row_statistics_bytes, 48);
        assert_eq!(largest.resources().row_statistics_disk_bytes, 48);
        assert_eq!(largest.resources().total_disk_bytes, MAX_BYTES + 48);
        let small = layout(6, 3);
        assert_eq!(small.resources().disk_bytes, 48);
        assert_eq!(small.resources().writer_scratch_bytes, 48);
        assert_eq!(small.resources().reader_scratch_bytes, 96);
        assert_eq!(small.resources().row_statistics_bytes, 48);
        assert_eq!(small.resources().row_statistics_disk_bytes, 96);
        assert_eq!(small.resources().total_disk_bytes, 144);
        assert_eq!(small.elements(), 6);
        assert_eq!(small.rows(), 2);
        assert_eq!(small.row_elements(), 3);
        assert_eq!(small.signed_bits(), 50);
        assert_eq!(small.fractional_bits(), 12);

        let many_rows = layout(456_064, 1).resources();
        assert_eq!(many_rows.row_statistics_disk_bytes, 21_891_072);
        assert_eq!(many_rows.row_statistics_bytes, 48);
        assert_eq!(many_rows.writer_scratch_bytes, MAX_CHUNK_BYTES);
        assert_eq!(many_rows.reader_scratch_bytes, MAX_CHUNK_BYTES);
        // Statistics obey their own existing per-file cap, without allocating
        // either large file or a resident table during preflight.
        let max_rows = MAX_BYTES / ROW_STATISTICS_RECORD_BYTES as u64;
        let at_limit = layout(max_rows, 1).resources();
        assert_eq!(at_limit.row_statistics_disk_bytes, max_rows * 48);
        assert_eq!(at_limit.total_disk_bytes, max_rows * 56);
        assert!(FixedFileLayout::new(max_rows + 1, 1, 50, 12).is_err());
    }

    #[test]
    fn canonical_serialization_digest_and_exact_statistics_cross_append_boundaries() {
        let values = [0, 1, -1, -(1 << 49), (1 << 49) - 1, 4096];
        let bytes = [
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255,
            0, 0, 0, 0, 0, 0, 254, 255, 255, 255, 255, 255, 255, 255, 1, 0, 0, 16, 0, 0, 0, 0, 0,
            0,
        ];
        let mut writer = FixedFileWriter::new(layout(6, 3)).unwrap();
        writer.write(&values[..2]).unwrap();
        writer.write_bytes(&bytes[16..40]).unwrap();
        writer.write(&values[5..]).unwrap();
        let file = writer.finish().unwrap();
        let metadata = file.metadata();
        assert_eq!(metadata.layout, layout(6, 3));
        assert_eq!(metadata.minimum, -(1 << 49));
        assert_eq!(metadata.maximum, (1 << 49) - 1);
        assert_eq!(metadata.digest, <[u8; 32]>::from(Sha256::digest(bytes)));
        let expected_statistics = [
            255, 255, 255, 255, 255, 255, 255, 255, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 254, 255, 255, 255, 255, 255, 255, 255, 1, 0,
            255, 15, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 254, 255, 255,
            255, 255, 255, 255, 255, 255, 255,
        ];
        let mut statistics_bytes = [0; 96];
        file.inner.rows.read(0, &mut statistics_bytes).unwrap();
        assert_eq!(statistics_bytes, expected_statistics);
        assert_eq!(
            metadata.row_statistics_digest,
            [
                96, 51, 42, 248, 24, 49, 248, 81, 84, 89, 240, 24, 243, 190, 97, 171, 250, 182,
                222, 164, 151, 161, 219, 113, 209, 122, 183, 194, 170, 214, 5, 130,
            ]
        );
        let resources = metadata.layout.resources();
        assert_eq!(file.inner.file.size(), resources.disk_bytes);
        assert_eq!(file.inner.rows.size(), resources.row_statistics_disk_bytes);
        assert_eq!(
            file.inner.file.size() + file.inner.rows.size(),
            resources.total_disk_bytes
        );
        let mut raw = [0; 48];
        file.read_bytes(0, &mut raw).unwrap();
        assert_eq!(raw, bytes);
        let mut actual = [0; 6];
        file.read(0, &mut actual).unwrap();
        assert_eq!(actual, values);
        assert_eq!(
            file.row_statistics(0).unwrap(),
            RowStatistics {
                minimum: -1,
                maximum: 1,
                positive_sum: 1,
                negative_sum: -1,
            }
        );
        assert_eq!(
            file.row_statistics(1).unwrap(),
            RowStatistics {
                minimum: -(1 << 49),
                maximum: (1 << 49) - 1,
                positive_sum: (1 << 49) - 1 + 4096,
                negative_sum: -(1 << 49),
            }
        );

        // Different append boundaries and code/byte entrypoints preserve identity.
        let mut same = FixedFileWriter::new(layout(6, 3)).unwrap();
        same.write(&values).unwrap();
        assert_eq!(same.finish().unwrap().metadata(), metadata);
        let mut other = FixedFileWriter::new(layout(6, 6)).unwrap();
        other.write(&values).unwrap();
        let other_metadata = other.finish().unwrap().metadata();
        assert_eq!(other_metadata.digest, metadata.digest);
        assert_ne!(
            other_metadata.row_statistics_digest,
            metadata.row_statistics_digest
        );
    }

    #[test]
    fn statistics_batches_are_bounded_and_match_individual_row_reads() {
        let count = MAX_ROW_STATISTICS * 2 + 3;
        let shape = layout(count as u64 * 2, 2);
        let mut writer = FixedFileWriter::new(shape).unwrap();
        let values: Vec<_> = (0..count as i64).flat_map(|i| [i, -i - 1]).collect();
        for chunk in values.chunks(997) {
            writer.write(chunk).unwrap();
        }
        let file = writer.finish().unwrap();
        let first = MAX_ROW_STATISTICS - 1;
        let mut rows = vec![RowStatistics::empty(); MAX_ROW_STATISTICS];
        file.read_row_statistics(first as u64, &mut rows).unwrap();
        for (i, row) in rows.iter().enumerate() {
            let value = (first + i) as i64;
            assert_eq!(
                *row,
                RowStatistics {
                    minimum: -value - 1,
                    maximum: value,
                    positive_sum: value as i128,
                    negative_sum: -value as i128 - 1,
                }
            );
        }
        for i in [0, rows.len() / 2, rows.len() - 1] {
            assert_eq!(rows[i], file.row_statistics((first + i) as u64).unwrap());
        }
        for (first, length) in [
            (0, 0),
            (0, MAX_ROW_STATISTICS + 1),
            (count as u64, 1),
            (count as u64 - 1, 2),
            (u64::MAX, 1),
        ] {
            assert!(file
                .read_row_statistics(first, &mut vec![RowStatistics::empty(); length])
                .is_err());
        }
        assert_eq!(shape.resources().row_statistics_bytes, 48);
        assert_eq!(
            shape.resources().reader_scratch_bytes,
            MAX_ROW_STATISTICS * 48
        );
        assert_eq!(file.inner.rows.size(), count as u64 * 48);
        assert!(rows.len() * ROW_STATISTICS_RECORD_BYTES <= MAX_CHUNK_BYTES);
    }

    #[test]
    fn multiple_full_chunks_and_rows_larger_than_a_chunk_stay_bounded() {
        let row_elements = MAX_CHUNK_VALUES as u64 + 3;
        let shape = layout(row_elements * 2, row_elements);
        let mut writer = FixedFileWriter::new(shape).unwrap();
        let values = vec![-7; MAX_CHUNK_VALUES];
        writer.write(&values).unwrap();
        let canonical: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        writer.write_bytes(&canonical).unwrap();
        writer.write(&[-7; 6]).unwrap();
        let file = writer.finish().unwrap();
        let mut read = vec![0; MAX_CHUNK_VALUES];
        file.read_row(1, 3, &mut read).unwrap();
        assert_eq!(read, values);
        let mut bytes = vec![0; MAX_CHUNK_BYTES];
        file.read_bytes(8, &mut bytes).unwrap();
        assert!(bytes.chunks_exact(8).all(|c| c == (-7_i64).to_le_bytes()));
        assert!(file.read(0, &mut vec![0; MAX_CHUNK_VALUES + 1]).is_err());
        assert!(file
            .read_bytes(0, &mut vec![0; MAX_CHUNK_BYTES + 8])
            .is_err());
        for row in 0..2 {
            assert_eq!(
                file.row_statistics(row).unwrap(),
                RowStatistics {
                    minimum: -7,
                    maximum: -7,
                    positive_sum: 0,
                    negative_sum: -7 * row_elements as i128,
                }
            );
        }
    }

    #[test]
    fn generic_signed_widths_and_i128_row_sums_are_exact() {
        for width in [1, 8, 50, 64] {
            let half = 1_i128 << (width - 1);
            let minimum = (-half) as i64;
            let maximum = (half - 1) as i64;
            let mut writer =
                FixedFileWriter::new(FixedFileLayout::new(4, 2, width, 0).unwrap()).unwrap();
            writer.write(&[minimum, minimum, maximum, maximum]).unwrap();
            let file = writer.finish().unwrap();
            assert_eq!(file.row_statistics(0).unwrap().negative_sum, -half * 2);
            assert_eq!(file.row_statistics(1).unwrap().positive_sum, (half - 1) * 2);
            assert_eq!(file.row_statistics(0).unwrap().maximum, minimum);
            assert_eq!(file.row_statistics(1).unwrap().minimum, maximum);
        }
    }

    #[test]
    fn invalid_appends_burn_the_whole_pending_file() {
        for values in [
            vec![],
            vec![0; MAX_CHUNK_VALUES + 1],
            vec![1 << 49],
            vec![2, 3, 1 << 49],
            vec![-(1 << 49) - 1],
            vec![i64::MIN],
            vec![i64::MAX],
        ] {
            let mut writer = FixedFileWriter::new(layout(4, 2)).unwrap();
            writer.write(&[1]).unwrap();
            assert!(writer.write(&values).is_err());
            assert!(writer.pending.is_none());
            assert!(writer.write(&[2]).is_err());
            assert!(writer.write_bytes(&0_i64.to_le_bytes()).is_err());
            assert!(writer.finish().is_err());
        }
        for bytes in [
            vec![],
            vec![0; 1],
            vec![0; 7],
            vec![0; 9],
            vec![0; MAX_CHUNK_BYTES + 8],
            (1_i64 << 49).to_le_bytes().to_vec(),
            (-(1_i64 << 49) - 1).to_le_bytes().to_vec(),
        ] {
            let mut writer = FixedFileWriter::new(layout(4, 2)).unwrap();
            writer.write(&[1]).unwrap();
            assert!(writer.write_bytes(&bytes).is_err());
            assert!(writer.pending.is_none());
            assert!(writer.write(&[2]).is_err());
            assert!(writer.finish().is_err());
        }
        let mut overfull = FixedFileWriter::new(layout(2, 1)).unwrap();
        overfull.write(&[1, 2]).unwrap();
        assert!(overfull.write(&[3]).is_err());
        assert!(overfull.finish().is_err());
        // These fit the file but exceed the independent append scratch bound.
        let larger = layout(MAX_CHUNK_VALUES as u64 + 1, 1);
        let mut oversized = FixedFileWriter::new(larger).unwrap();
        assert!(oversized.write(&vec![0; MAX_CHUNK_VALUES + 1]).is_err());
        assert!(oversized.finish().is_err());
        let mut oversized = FixedFileWriter::new(larger).unwrap();
        assert!(oversized
            .write_bytes(&vec![0; MAX_CHUNK_BYTES + 8])
            .is_err());
        assert!(oversized.finish().is_err());
        for count in 0..4 {
            let mut incomplete = FixedFileWriter::new(layout(4, 2)).unwrap();
            if count != 0 {
                incomplete.write(&vec![1; count]).unwrap();
            }
            assert!(incomplete.finish().is_err());
        }
        let mut cancelled = FixedFileWriter::new(layout(4, 2)).unwrap();
        cancelled.write(&[1, 2]).unwrap();
        cancelled.cancel();
        cancelled.cancel();
        assert!(cancelled.pending.is_none());
        assert!(cancelled.write(&[3, 4]).is_err());
        assert!(cancelled.finish().is_err());
    }

    #[test]
    fn failure_of_either_stream_retires_both_and_both_must_seal() {
        for statistics in [false, true] {
            let mut writer = FixedFileWriter::new(layout(4, 2)).unwrap();
            writer.write(&[1, 2]).unwrap();
            // Inject a retired backing writer after an already committed row.
            let mut retired = PublicFileWriter::new_computed(1).unwrap();
            assert!(retired.write(&[]).is_err());
            let pending = writer.pending.as_mut().unwrap();
            if statistics {
                pending.rows = retired;
            } else {
                pending.file = retired;
            }
            assert!(writer.write(&[3, 4]).is_err());
            assert!(writer.pending.is_none());
            assert!(writer.finish().is_err());

            let mut writer = FixedFileWriter::new(layout(4, 2)).unwrap();
            writer.write(&[1, 2, 3, 4]).unwrap();
            // Complete coefficient/row counters cannot bypass either file's
            // independent completion check and computed-content seal.
            let pending = writer.pending.as_mut().unwrap();
            if statistics {
                pending.rows = PublicFileWriter::new_computed(96).unwrap();
            } else {
                pending.file = PublicFileWriter::new_computed(32).unwrap();
            }
            assert!(writer.finish().is_err());
        }
    }

    #[test]
    fn malformed_offsets_rows_and_read_sizes_fail_without_poisoning_snapshot() {
        let mut writer = FixedFileWriter::new(layout(6, 3)).unwrap();
        writer.write(&[1, 2, 3, 4, 5, 6]).unwrap();
        let file = writer.finish().unwrap();
        for (offset, length) in [
            (0, 0),
            (0, MAX_CHUNK_VALUES + 1),
            (6, 1),
            (5, 2),
            (u64::MAX, 1),
        ] {
            assert!(file.read(offset, &mut vec![0; length]).is_err());
        }
        for (offset, length) in [
            (0, 0),
            (0, 7),
            (1, 8),
            (0, MAX_CHUNK_BYTES + 8),
            (48, 8),
            (40, 16),
            (u64::MAX - 7, 8),
        ] {
            assert!(file.read_bytes(offset, &mut vec![0; length]).is_err());
        }
        for (row, col, length) in [
            (0, 0, 0),
            (2, 0, 1),
            (0, 3, 1),
            (0, 2, 2),
            (u64::MAX, 0, 1),
            (0, u64::MAX, 1),
        ] {
            assert!(file.read_row(row, col, &mut vec![0; length]).is_err());
        }
        assert!(file.row_statistics(2).is_err());
        assert!(file.row_statistics(u64::MAX).is_err());
        let mut value = [0];
        file.read_row(1, 2, &mut value).unwrap();
        assert_eq!(value, [6]);
    }

    #[test]
    fn shared_owners_keep_bytes_alive_and_independent_reads_do_not_seek() {
        let mut writer = FixedFileWriter::new(layout(4, 2)).unwrap();
        let mut input = [10, 20, 30, 40];
        writer.write(&input).unwrap();
        input.fill(-1);
        let file = writer.finish().unwrap();
        let handle = file.inner.file.file();
        let weak_file = Arc::downgrade(&handle);
        let statistics_handle = file.inner.rows.file();
        let weak_statistics = Arc::downgrade(&statistics_handle);
        let weak_owner = Arc::downgrade(&file.inner);
        drop(handle);
        drop(statistics_handle);
        let clone = file.clone();
        drop(file);
        std::thread::scope(|scope| {
            for row in 0..2 {
                let reader = &clone;
                scope.spawn(move || {
                    for _ in 0..8 {
                        let mut values = [0; 2];
                        reader.read_row(row, 0, &mut values).unwrap();
                        assert_eq!(values, [10 + row as i64 * 20, 20 + row as i64 * 20]);
                        let statistics = reader.row_statistics(row).unwrap();
                        assert_eq!(statistics.minimum, values[0]);
                        assert_eq!(statistics.maximum, values[1]);
                        assert_eq!(statistics.positive_sum, (values[0] + values[1]) as i128);
                    }
                });
            }
        });
        assert!(weak_file.upgrade().is_some());
        assert!(weak_statistics.upgrade().is_some());
        drop(clone);
        assert!(weak_owner.upgrade().is_none());
        assert!(weak_file.upgrade().is_none());
        assert!(weak_statistics.upgrade().is_none());
    }
}
