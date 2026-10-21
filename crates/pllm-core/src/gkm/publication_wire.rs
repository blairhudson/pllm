use k256::{elliptic_curve::sec1::FromEncodedPoint, AffinePoint, EncodedPoint};

use super::super::window::wire::{Cursor, WireBytes};
use super::*;

const MAGIC: &[u8; 8] = b"PLGKMP01";

fn read_point(cursor: &mut Cursor<'_>) -> Result<ProjectivePoint, Error> {
    let length = cursor.byte()? as usize;
    if length != 1 && length != 33 {
        return Err(Error::Binding);
    }
    let bytes = cursor.take(length)?;
    if (length == 1 && bytes != [0]) || (length == 33 && !matches!(bytes[0], 2 | 3)) {
        return Err(Error::Binding);
    }
    let encoded = EncodedPoint::from_bytes(bytes).map_err(|_| Error::Binding)?;
    let affine = Option::<AffinePoint>::from(AffinePoint::from_encoded_point(&encoded))
        .ok_or(Error::Binding)?;
    if affine.to_encoded_point(true).as_bytes() != bytes {
        return Err(Error::Binding);
    }
    // secp256k1 has cofactor one. Identity is legal: owner scalars may be zero,
    // and zero matrix columns legitimately commit to the group identity.
    Ok(ProjectivePoint::from(affine))
}

fn count(cursor: &mut Cursor<'_>, max: usize) -> Result<usize, Error> {
    let value = cursor.u64()?;
    if value == 0 || value > max as u64 {
        return Err(Error::Limit);
    }
    Ok(value as usize)
}

impl Publication {
    /// Canonical public material only. The identity is byte-for-byte the native
    /// publication digest; setup scalars, weights and float rows never appear.
    pub fn export_wire(&self) -> Result<WireBytes, Error> {
        let point_length =
            |p: &ProjectivePoint| 1 + p.to_affine().to_encoded_point(true).as_bytes().len();
        let mut length = self
            .row_hashes
            .len()
            .checked_mul(32)
            .and_then(|n| n.checked_add(136))
            .ok_or(Error::Limit)?;
        for point in &self.r {
            length = length
                .checked_add(point_length(point))
                .ok_or(Error::Limit)?;
        }
        for matrix in &self.matrices {
            length = matrix
                .rows
                .checked_add(matrix.columns)
                .and_then(|n| n.checked_mul(8))
                .and_then(|n| n.checked_add(80))
                .and_then(|n| n.checked_add(length))
                .ok_or(Error::Limit)?;
            for point in &matrix.s {
                length = length
                    .checked_add(point_length(point))
                    .ok_or(Error::Limit)?;
            }
        }
        let mut out =
            WireBytes::allocate_publication(length, self.capacity, self.export_quota.as_ref())?;
        out.extend(MAGIC);
        out.extend(&self.digest);
        out.extend(&self.context);
        out.extend(&(self.matrices.len() as u64).to_le_bytes());
        out.extend(&(self.r.len() as u64).to_le_bytes());
        let write_point = |out: &mut WireBytes, point: &ProjectivePoint| {
            let encoded = point.to_affine().to_encoded_point(true);
            out.extend(&[encoded.as_bytes().len() as u8]);
            out.extend(encoded.as_bytes());
        };
        for point in &self.r {
            write_point(&mut out, point);
        }
        for matrix in &self.matrices {
            out.extend(&matrix.id);
            out.extend(&matrix.digest);
            out.extend(&(matrix.rows as u64).to_le_bytes());
            out.extend(&(matrix.columns as u64).to_le_bytes());
            for bound in matrix.input_bounds.iter().chain(&matrix.output_bounds) {
                out.extend(&bound.to_le_bytes());
            }
            for point in &matrix.s {
                write_point(&mut out, point);
            }
        }
        out.extend(&self.table_digest);
        out.extend(&(self.row_hashes.len() as u64).to_le_bytes());
        out.extend(&(self.width as u64).to_le_bytes());
        for hash in &self.row_hashes {
            out.extend(hash);
        }
        Ok(out)
    }

    /// Import only with an independently authenticated expected owner digest.
    /// Hash, canonical curve points, all geometry/ranges, duplicate stage IDs,
    /// and exact frame length are checked without dynamic allocation. The
    /// unchanged complete-publication envelope is then reserved before payload
    /// allocation. Caller-owned inbound bytes are outside that envelope.
    pub fn import_wire(bytes: &[u8], authenticated_owner_digest: [u8; 32]) -> Result<Self, Error> {
        Self::import_wire_with_capacity(bytes, authenticated_owner_digest, Capacity::Capsule)
    }

    pub fn import_wire_with_capacity(
        bytes: &[u8],
        authenticated_owner_digest: [u8; 32],
        capacity: Capacity,
    ) -> Result<Self, Error> {
        Self::import_wire_owned(
            bytes,
            authenticated_owner_digest,
            capacity,
            PublicationOwnership::General,
        )
    }

    pub fn import_wire_streamed_single_window(
        bytes: &[u8],
        authenticated_owner_digest: [u8; 32],
        capacity: Capacity,
    ) -> Result<Self, Error> {
        Self::import_wire_owned(
            bytes,
            authenticated_owner_digest,
            capacity,
            PublicationOwnership::StreamedSingleWindow,
        )
    }

    fn import_wire_owned(
        bytes: &[u8],
        authenticated_owner_digest: [u8; 32],
        capacity: Capacity,
        ownership: PublicationOwnership,
    ) -> Result<Self, Error> {
        if bytes.len() > capacity.max_publication_wire_bytes() {
            return Err(Error::Limit);
        }
        let mut cursor = Cursor::new(bytes);
        if cursor.take(8)? != MAGIC || cursor.array::<32>()? != authenticated_owner_digest {
            return Err(Error::Binding);
        }
        let mut hash = Sha256::new();
        hash.update(b"pllm.gkm.owner-publication.v1");
        hash.update(NUMERIC_PROFILE);
        hash.update(&bytes[40..]);
        let actual: [u8; 32] = hash.finalize().into();
        if actual != authenticated_owner_digest {
            return Err(Error::Binding);
        }

        let context = cursor.array()?;
        let stages = count(&mut cursor, capacity.max_stages())?;
        let rows = count(&mut cursor, capacity.max_rows())?;
        for _ in 0..rows {
            read_point(&mut cursor)?;
        }
        // Only offsets are retained during preflight (2 KiB on 64-bit). IDs and
        // shapes stay in the borrowed authenticated frame, never wide stack arrays.
        let mut offsets = [0usize; Capacity::Decoder.max_stages()];
        let mut max_rows = 0;
        for stage in 0..stages {
            offsets[stage] = cursor.position();
            let id = cursor.array::<32>()?;
            if offsets[..stage].iter().any(|&at| bytes[at..at + 32] == id) {
                return Err(Error::Binding);
            }
            cursor.take(32)?;
            let m = count(&mut cursor, capacity.max_rows())?;
            let n = count(&mut cursor, capacity.max_columns())?;
            capacity.matrix_elements(m, n)?;
            max_rows = max_rows.max(m);
            for _ in 0..n {
                if cursor.u64()? > 1u64 << 63 {
                    return Err(Error::Bound);
                }
            }
            for _ in 0..m {
                if cursor.u64()? >= 1u64 << 63 {
                    return Err(Error::Bound);
                }
            }
            for _ in 0..n {
                read_point(&mut cursor)?;
            }
        }
        if max_rows != rows {
            return Err(Error::Shape);
        }
        cursor.take(32)?;
        let vocabulary = count(&mut cursor, capacity.max_vocabulary())?;
        let width = count(&mut cursor, capacity.max_columns())?;
        cursor.take(vocabulary * 32)?;
        cursor.finish()?;
        let shapes = offsets[..stages].iter().map(|&at| {
            let rows = u64::from_le_bytes(bytes[at + 64..at + 72].try_into().unwrap()) as usize;
            let columns = u64::from_le_bytes(bytes[at + 72..at + 80].try_into().unwrap()) as usize;
            (rows, columns)
        });
        let resources = PublicationResources::for_shape_iter_owned(
            shapes, vocabulary, width, capacity, ownership,
        )?;
        let admission = PublicationPermit::acquire(
            &PROCESS_PUBLICATIONS,
            resources.admitted_payload_bytes,
            capacity,
        )?;

        // A second bounded pass builds the already-validated immutable owners.
        let mut cursor = Cursor::new(&bytes[88..]); // prefix, context, stage count, R count
        let mut r = payload::capacity(rows)?;
        for _ in 0..rows {
            r.push(read_point(&mut cursor)?);
        }
        let mut matrices = payload::capacity(stages)?;
        for _ in 0..stages {
            let id = cursor.array()?;
            let digest = cursor.array()?;
            let rows = cursor.u64()? as usize;
            let columns = cursor.u64()? as usize;
            let mut input_bounds = payload::capacity(columns)?;
            let mut output_bounds = payload::capacity(rows)?;
            for _ in 0..columns {
                input_bounds.push(cursor.u64()?);
            }
            for _ in 0..rows {
                output_bounds.push(cursor.u64()?);
            }
            let mut s = payload::capacity(columns)?;
            for _ in 0..columns {
                s.push(read_point(&mut cursor)?);
            }
            matrices.push(MatrixPublication {
                id,
                digest,
                rows,
                columns,
                input_bounds,
                output_bounds,
                s,
            });
        }
        let table_digest = cursor.array()?;
        cursor.u64()?;
        cursor.u64()?;
        let mut row_hashes = payload::capacity(vocabulary)?;
        for _ in 0..vocabulary {
            row_hashes.push(cursor.array()?);
        }
        cursor.finish()?;
        Ok(Self {
            capacity,
            context,
            matrices,
            r,
            table_digest,
            width,
            row_hashes,
            resources,
            digest: authenticated_owner_digest,
            live: AtomicU32::new(0),
            clients: AtomicU32::new(0),
            window_started: AtomicBool::new(false),
            export_quota: Self::new_export_quota(ownership),
            _admission: admission,
        })
    }
}
