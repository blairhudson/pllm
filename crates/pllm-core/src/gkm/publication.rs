use k256::{elliptic_curve::sec1::ToEncodedPoint, ProjectivePoint, Scalar};
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use zeroize::Zeroizing;

use super::{
    payload, sample_scalar, signed_scalar, Entropy, Error, FloatTable, I16Matrix, OsEntropy,
    Purpose, MAX_DIMENSION, MAX_LIVE_WINDOWS, MAX_PROCESS_PUBLICATION_BYTES, MAX_PUBLICATION_BYTES,
    MAX_STAGES, MAX_VOCABULARY, MAX_WINDOW_BYTES, MAX_WINDOW_STEPS, NUMERIC_PROFILE,
};

/// Public stage identity supplied by the model/graph owner. Repeated uses in a
/// window reference this same identity; two different stages cannot share an ID.
pub struct Stage<'a> {
    pub id: [u8; 32],
    pub matrix: &'a I16Matrix,
}

/// Geometry-only payload admission, not an allocator/RSS or performance claim.
#[derive(Debug, Clone, Copy)]
pub struct PublicationResources {
    pub source_weight_bytes: usize,
    pub source_bound_bytes: usize,
    pub source_table_bytes: usize,
    pub max_import_workspace_bytes: usize,
    pub native_publication_payload_bytes: usize,
    pub owner_scalar_workspace_bytes: usize,
    /// Additional client/worker AES state and expanded scratch, reserved even
    /// for full-share windows so either profile shares the same hard live cap.
    pub seeded_window_extra_bytes: usize,
    pub max_window_payload_bytes: usize,
    pub max_live_window_payload_bytes: usize,
    /// Full reservation charged once per publication, shared by every client.
    pub admitted_payload_bytes: usize,
}

impl PublicationResources {
    /// Run before importing source weights/table when their public geometry is
    /// known. Includes native point storage, not just 33-byte encoded points.
    pub fn for_shapes(
        shapes: &[(usize, usize)],
        vocabulary: usize,
        width: usize,
    ) -> Result<Self, Error> {
        if shapes.is_empty()
            || shapes.len() > MAX_STAGES
            || vocabulary == 0
            || vocabulary > MAX_VOCABULARY
            || width == 0
            || width > MAX_DIMENSION
        {
            return Err(Error::Limit);
        }
        let (mut weights, mut columns, mut dimensions, mut max_rows, mut max_weights) =
            (0usize, 0usize, 0usize, 0usize, 0usize);
        for &(m, n) in shapes {
            if m == 0 || n == 0 || m > MAX_DIMENSION || n > MAX_DIMENSION {
                return Err(Error::Limit);
            }
            weights = weights
                .checked_add(m.checked_mul(n).ok_or(Error::Limit)?)
                .ok_or(Error::Limit)?;
            columns = columns.checked_add(n).ok_or(Error::Limit)?;
            dimensions = dimensions.checked_add(m + n).ok_or(Error::Limit)?;
            max_rows = max_rows.max(m);
            max_weights = max_weights.max(m * n);
        }
        // Dimensions/counts above are now bounded. Constants include fixed
        // native metadata, plan, two worker views, partial responses, decoded
        // values and final-verification scratch. Caller copies and allocator
        // headers are excluded; no unbounded
        // verifier queue exists. Source allocations are one immutable copy used
        // by two logical in-process worker views, not two measured server RSSes.
        let source_weight_bytes = weights * 2;
        let source_bound_bytes = dimensions * 8;
        let source_table_bytes = vocabulary * width * 4;
        // from_f64 holds an encoded import vector while constructing its owner.
        let max_import_workspace_bytes = max_weights * 2;
        let native_publication_payload_bytes = (max_rows + columns) * size_of::<ProjectivePoint>()
            + dimensions * 8
            + vocabulary * 32
            + shapes.len() * 256
            + 4096;
        let owner_scalar_workspace_bytes = (max_rows + MAX_DIMENSION) * size_of::<Scalar>();
        let seeded_window_extra_bytes = super::seeded::WINDOW_EXTRA_BYTES;
        let max_window_payload_bytes = dimensions * size_of::<Scalar>()
            + max_rows * size_of::<Scalar>()
            + 16 * MAX_DIMENSION * 8
            + 4 * width * 4
            + 2 * vocabulary.div_ceil(8)
            + MAX_WINDOW_STEPS * 64
            + 16_384
            + seeded_window_extra_bytes;
        let max_live_window_payload_bytes = max_window_payload_bytes * MAX_LIVE_WINDOWS as usize;
        let total = source_weight_bytes
            + source_bound_bytes
            + source_table_bytes
            + max_import_workspace_bytes
            + native_publication_payload_bytes
            + owner_scalar_workspace_bytes
            + max_live_window_payload_bytes;
        if total > MAX_PUBLICATION_BYTES || max_window_payload_bytes > MAX_WINDOW_BYTES {
            return Err(Error::Limit);
        }
        Ok(Self {
            source_weight_bytes,
            source_bound_bytes,
            source_table_bytes,
            max_import_workspace_bytes,
            native_publication_payload_bytes,
            owner_scalar_workspace_bytes,
            seeded_window_extra_bytes,
            max_window_payload_bytes,
            max_live_window_payload_bytes,
            admitted_payload_bytes: total,
        })
    }
}

// A publication reserves its entire envelope before owner payload allocation.
// The permit is the last field dropped, after all publication payloads. Window
// and response leases retain the Publication, so cancellation cannot free this
// reservation while a detached admitted response still owns its buffer.
pub(super) struct AdmissionPool {
    pub bytes: AtomicUsize,
    limit: usize,
}

impl AdmissionPool {
    pub const fn new(limit: usize) -> Self {
        Self {
            bytes: AtomicUsize::new(0),
            limit,
        }
    }
}

pub(super) static PROCESS_PUBLICATIONS: AdmissionPool =
    AdmissionPool::new(MAX_PROCESS_PUBLICATION_BYTES);

struct PublicationPermit {
    pool: &'static AdmissionPool,
    bytes: usize,
}

impl PublicationPermit {
    fn acquire(pool: &'static AdmissionPool, bytes: usize) -> Result<Self, Error> {
        pool.bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes).filter(|&next| next <= pool.limit)
            })
            .map_err(|_| Error::Limit)?;
        Ok(Self { pool, bytes })
    }
}

impl Drop for PublicationPermit {
    fn drop(&mut self) {
        self.pool.bytes.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

pub(super) struct MatrixPublication {
    pub id: [u8; 32],
    pub digest: [u8; 32],
    pub rows: usize,
    pub columns: usize,
    pub input_bounds: Vec<u64>,
    pub output_bounds: Vec<u64>,
    pub s: Vec<ProjectivePoint>,
}

/// Honest-owner publication. No setup scalars, matrix weights or embedding
/// values are retained. Fields are immutable and there is no untrusted-point
/// deserializer in this capsule. The owner digest binds graph/checkpoint context,
/// numeric profile, bounds, dimensions, matrix/table identities, all R/S and
/// every row hash. A digest is an identity, not a signature/authentication channel.
pub struct Publication {
    pub(super) context: [u8; 32],
    pub(super) matrices: Vec<MatrixPublication>,
    pub(super) r: Vec<ProjectivePoint>,
    pub(super) table_digest: [u8; 32],
    pub(super) width: usize,
    pub(super) row_hashes: Vec<[u8; 32]>,
    pub(super) resources: PublicationResources,
    pub(super) digest: [u8; 32],
    pub(super) live: AtomicU32,
    pub(super) clients: AtomicU32,
    _admission: PublicationPermit,
}

impl Publication {
    pub fn publish(
        context: [u8; 32],
        stages: &[Stage<'_>],
        table: &FloatTable,
    ) -> Result<Self, Error> {
        Self::publish_with(context, stages, table, &mut OsEntropy)
    }

    pub(super) fn publish_with(
        context: [u8; 32],
        stages: &[Stage<'_>],
        table: &FloatTable,
        rng: &mut impl Entropy,
    ) -> Result<Self, Error> {
        Self::publish_in_pool(context, stages, table, rng, &PROCESS_PUBLICATIONS)
    }

    pub(super) fn publish_in_pool(
        context: [u8; 32],
        stages: &[Stage<'_>],
        table: &FloatTable,
        rng: &mut impl Entropy,
        pool: &'static AdmissionPool,
    ) -> Result<Self, Error> {
        if stages.is_empty() || stages.len() > MAX_STAGES {
            return Err(Error::Limit);
        }
        let mut shapes = [(0, 0); MAX_STAGES];
        for (index, stage) in stages.iter().enumerate() {
            if stages[..index].iter().any(|prior| prior.id == stage.id) {
                return Err(Error::Binding);
            }
            shapes[index] = stage.matrix.shape();
        }
        let resources = PublicationResources::for_shapes(
            &shapes[..stages.len()],
            table.vocabulary,
            table.width,
        )?;
        let admission = PublicationPermit::acquire(pool, resources.admitted_payload_bytes)?;
        let max_rows = stages.iter().map(|s| s.matrix.rows).max().unwrap();
        let mut secret = Zeroizing::new(payload::capacity(max_rows)?);
        for _ in 0..max_rows {
            secret.push(sample_scalar(rng, Purpose::Owner)?);
        }
        let r = payload::collect(secret.iter().map(|r| ProjectivePoint::GENERATOR * r))?;
        let mut matrices = payload::capacity(stages.len())?;
        for stage in stages {
            let matrix = stage.matrix;
            let mut scalars = Zeroizing::new(payload::filled(matrix.columns, Scalar::ZERO)?);
            for (row, r) in matrix
                .weights
                .chunks_exact(matrix.columns)
                .zip(secret.iter())
            {
                for (sum, &weight) in scalars.iter_mut().zip(row) {
                    *sum += signed_scalar(i64::from(weight)) * r;
                }
            }
            let s = payload::collect(scalars.iter().map(|s| ProjectivePoint::GENERATOR * s))?;
            matrices.push(MatrixPublication {
                id: stage.id,
                digest: matrix.digest,
                rows: matrix.rows,
                columns: matrix.columns,
                input_bounds: payload::copy(&matrix.input_bounds)?,
                output_bounds: payload::copy(&matrix.output_bounds)?,
                s,
            });
        }
        let row_hashes = payload::collect(
            table
                .bytes
                .chunks_exact(table.width * 4)
                .map(|row| Sha256::digest(row).into()),
        )?;
        let mut publication = Self {
            context,
            matrices,
            r,
            table_digest: table.digest,
            width: table.width,
            row_hashes,
            resources,
            digest: [0; 32],
            live: AtomicU32::new(0),
            clients: AtomicU32::new(0),
            _admission: admission,
        };
        publication.digest = publication.compute_digest();
        Ok(publication)
        // Both secret r and temporary M^T*r scalar vectors zeroize on drop.
    }

    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }

    pub fn context(&self) -> [u8; 32] {
        self.context
    }

    pub fn resources(&self) -> PublicationResources {
        self.resources
    }

    /// Shared across all clients; includes retained responses and accepted output
    /// owners until their payloads actually drop, not only active transactions.
    pub fn live_windows(&self) -> u32 {
        self.live.load(Ordering::Acquire)
    }

    pub(super) fn matrix_index(&self, id: &[u8; 32]) -> Result<usize, Error> {
        self.matrices
            .iter()
            .position(|matrix| &matrix.id == id)
            .ok_or(Error::Schedule)
    }

    pub(super) fn compute_digest(&self) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(b"pllm.gkm.owner-publication.v1");
        hash.update(NUMERIC_PROFILE);
        hash.update(self.context);
        hash.update((self.matrices.len() as u64).to_le_bytes());
        hash.update((self.r.len() as u64).to_le_bytes());
        for point in &self.r {
            hash_point(&mut hash, point);
        }
        for matrix in &self.matrices {
            hash.update(matrix.id);
            hash.update(matrix.digest);
            hash.update((matrix.rows as u64).to_le_bytes());
            hash.update((matrix.columns as u64).to_le_bytes());
            for &bound in matrix
                .input_bounds
                .iter()
                .chain(matrix.output_bounds.iter())
            {
                hash.update(bound.to_le_bytes());
            }
            for point in &matrix.s {
                hash_point(&mut hash, point);
            }
        }
        hash.update(self.table_digest);
        hash.update((self.row_hashes.len() as u64).to_le_bytes());
        hash.update((self.width as u64).to_le_bytes());
        for row_hash in &self.row_hashes {
            hash.update(row_hash);
        }
        hash.finalize().into()
    }
}

fn hash_point(hash: &mut Sha256, point: &ProjectivePoint) {
    // Canonical compressed SEC1, with its one-byte 00 identity encoding.
    // Length prefix also distinguishes identity from a nonidentity point.
    let encoded = point.to_affine().to_encoded_point(true);
    hash.update([encoded.as_bytes().len() as u8]);
    hash.update(encoded.as_bytes());
}
