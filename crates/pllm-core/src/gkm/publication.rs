use k256::{elliptic_curve::sec1::ToEncodedPoint, ProjectivePoint, Scalar};
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;
use zeroize::Zeroizing;

#[path = "publication_wire.rs"]
mod wire;

use super::{
    payload, sample_scalar, signed_scalar, Capacity, Entropy, Error, FloatTable, I16Matrix,
    OsEntropy, Purpose, MAX_DIMENSION, MAX_LIVE_WINDOWS, MAX_PROCESS_PUBLICATION_BYTES,
    NUMERIC_PROFILE,
};

/// Public stage identity supplied by the model/graph owner. Repeated uses in a
/// window reference this same identity; two different stages cannot share an ID.
pub struct Stage<'a> {
    pub id: [u8; 32],
    pub matrix: &'a I16Matrix,
}

/// Local ownership/admission policy, never supplied by an untrusted frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublicationOwnership {
    General,
    StreamedSingleWindow,
}

pub const STREAMED_SOURCE_CHUNK_BYTES: usize = 64 << 10;
/// One inbound float32 frame plus bounded conversion/transfer scratch. Final
/// encoded source buffers and bounds are charged independently, in every role.
pub const STREAMED_IMPORT_WORKSPACE_BYTES: usize = 2 * STREAMED_SOURCE_CHUNK_BYTES;

/// Geometry-only payload admission, not an allocator/RSS or performance claim.
#[derive(Debug, Clone, Copy)]
pub struct PublicationResources {
    pub capacity: Capacity,
    pub ownership: PublicationOwnership,
    pub source_weight_bytes: usize,
    pub source_bound_bytes: usize,
    pub source_table_bytes: usize,
    pub max_import_workspace_bytes: usize,
    pub native_publication_payload_bytes: usize,
    pub owner_scalar_workspace_bytes: usize,
    /// Additional client/worker AES state and expanded scratch, conservatively
    /// reserved for either sharing profile and either capacity.
    pub seeded_window_extra_bytes: usize,
    pub window_accumulator_bytes: usize,
    pub max_window_payload_bytes: usize,
    pub max_live_window_payload_bytes: usize,
    /// Full reservation charged once per publication, shared by every client.
    pub admitted_payload_bytes: usize,
}

impl PublicationResources {
    pub fn for_streamed_single_window(
        shapes: &[(usize, usize)],
        vocabulary: usize,
        width: usize,
        capacity: Capacity,
    ) -> Result<Self, Error> {
        Self::for_shape_iter_owned(
            shapes.iter().copied(),
            vocabulary,
            width,
            capacity,
            PublicationOwnership::StreamedSingleWindow,
        )
    }

    fn for_shape_iter_owned(
        shapes: impl ExactSizeIterator<Item = (usize, usize)>,
        vocabulary: usize,
        width: usize,
        capacity: Capacity,
        ownership: PublicationOwnership,
    ) -> Result<Self, Error> {
        // Preserve all original geometry eligibility and ceilings. Specialization
        // changes owned overlap, not which geometry the generic core admits.
        let mut resources = Self::for_shape_iter(shapes, vocabulary, width, capacity)?;
        if ownership == PublicationOwnership::StreamedSingleWindow {
            resources.admitted_payload_bytes = resources
                .admitted_payload_bytes
                .checked_sub(resources.max_import_workspace_bytes)
                .and_then(|n| n.checked_sub(resources.max_live_window_payload_bytes))
                .and_then(|n| n.checked_add(STREAMED_IMPORT_WORKSPACE_BYTES))
                .and_then(|n| n.checked_add(resources.max_window_payload_bytes))
                .filter(|&n| n <= capacity.max_publication_bytes())
                .ok_or(Error::Limit)?;
            resources.ownership = ownership;
            resources.max_import_workspace_bytes = STREAMED_IMPORT_WORKSPACE_BYTES;
            resources.max_live_window_payload_bytes = resources.max_window_payload_bytes;
        }
        Ok(resources)
    }

    /// Run before importing source weights/table when their public geometry is
    /// known. Includes native point storage, not just 33-byte encoded points.
    pub fn for_shapes(
        shapes: &[(usize, usize)],
        vocabulary: usize,
        width: usize,
    ) -> Result<Self, Error> {
        Self::for_shapes_with_capacity(shapes, vocabulary, width, Capacity::Capsule)
    }

    pub fn for_shapes_with_capacity(
        shapes: &[(usize, usize)],
        vocabulary: usize,
        width: usize,
        capacity: Capacity,
    ) -> Result<Self, Error> {
        Self::for_shape_iter(shapes.iter().copied(), vocabulary, width, capacity)
    }

    fn for_shape_iter(
        shapes: impl ExactSizeIterator<Item = (usize, usize)>,
        vocabulary: usize,
        width: usize,
        capacity: Capacity,
    ) -> Result<Self, Error> {
        let stages = shapes.len();
        if stages == 0 || stages > capacity.max_stages() {
            return Err(Error::Limit);
        }
        let source_table_bytes = capacity.table_bytes(vocabulary, width)?;
        let (
            mut weights,
            mut columns,
            mut dimensions,
            mut max_rows,
            mut max_columns,
            mut max_weights,
        ) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        for (m, n) in shapes {
            let elements = capacity.matrix_elements(m, n)?;
            weights = weights.checked_add(elements).ok_or(Error::Limit)?;
            columns = columns.checked_add(n).ok_or(Error::Limit)?;
            dimensions = dimensions.checked_add(m + n).ok_or(Error::Limit)?;
            max_rows = max_rows.max(m);
            max_columns = max_columns.max(n);
            max_weights = max_weights.max(elements);
        }
        // Dimensions/counts above are now bounded. Constants include fixed
        // native metadata, plan, two worker views, partial responses, decoded
        // values and final-verification scratch. Caller copies and allocator
        // headers are excluded; no unbounded
        // verifier queue exists. Source allocations are one immutable copy used
        // by two logical in-process worker views, not two measured server RSSes.
        // Wide intermediate arithmetic also makes geometry-only rejection safe
        // on 32-bit targets. Every conversion precedes allocation or entropy.
        let bytes = |n: u128| usize::try_from(n).map_err(|_| Error::Limit);
        let source_weight_bytes = bytes(weights as u128 * 2)?;
        let source_bound_bytes = bytes(dimensions as u128 * 8)?;
        let source_bytes =
            source_weight_bytes as u128 + source_bound_bytes as u128 + source_table_bytes as u128;
        if source_bytes > capacity.max_source_bytes() as u128 {
            return Err(Error::Limit);
        }
        // from_f64 holds an encoded import vector while constructing its owner.
        let max_import_workspace_bytes = bytes(max_weights as u128 * 2)?;
        let native_publication_payload_bytes = bytes(
            (max_rows + columns) as u128 * size_of::<ProjectivePoint>() as u128
                + dimensions as u128 * 8
                + vocabulary as u128 * 32
                + stages as u128 * 256
                + 4096,
        )?;
        let owner_columns = if capacity == Capacity::Capsule {
            MAX_DIMENSION
        } else {
            max_columns
        };
        let owner_scalar_workspace_bytes =
            bytes((max_rows + owner_columns) as u128 * size_of::<Scalar>() as u128)?;
        let seeded_window_extra_bytes = if capacity == Capacity::Capsule {
            super::seeded::WINDOW_EXTRA_BYTES
        } else {
            2 * size_of::<super::seeded::Streams>()
                + max_columns * 8
                + vocabulary.div_ceil(8)
                + 32
                + 512
        };
        let window_accumulator_bytes = bytes(dimensions as u128 * size_of::<Scalar>() as u128)?;
        // Capsule admission is unchanged. Decoder prices separate input/output
        // widths: masks + input copies, both retained responses, old/new signed
        // and rescaled outputs, and fallible accumulator metadata. A wide output
        // is never priced as though it had the (possibly tiny) input width.
        let pending_bytes = if capacity == Capacity::Capsule {
            16 * MAX_DIMENSION as u128 * 8
        } else {
            max_columns as u128 * 32 + max_rows as u128 * 48 + stages as u128 * 128
        };
        let max_window_payload_bytes = bytes(
            window_accumulator_bytes as u128
                + max_rows as u128 * size_of::<Scalar>() as u128
                + pending_bytes
                + 4 * width as u128 * 4
                + 2 * vocabulary.div_ceil(8) as u128
                + capacity.max_window_steps() as u128 * 64
                + 16_384
                + seeded_window_extra_bytes as u128,
        )?;
        let max_live_window_payload_bytes =
            bytes(max_window_payload_bytes as u128 * MAX_LIVE_WINDOWS as u128)?;
        let total = bytes(
            source_bytes
                + max_import_workspace_bytes as u128
                + native_publication_payload_bytes as u128
                + owner_scalar_workspace_bytes as u128
                + max_live_window_payload_bytes as u128,
        )?;
        let encoded_upper_bound = 136u128
            + vocabulary as u128 * 32
            + (max_rows + columns) as u128 * 34
            + stages as u128 * 80
            + dimensions as u128 * 8;
        if total > capacity.max_publication_bytes()
            || max_window_payload_bytes > capacity.max_window_bytes()
            || encoded_upper_bound > capacity.max_publication_wire_bytes() as u128
        {
            return Err(Error::Limit);
        }
        Ok(Self {
            capacity,
            ownership: PublicationOwnership::General,
            source_weight_bytes,
            source_bound_bytes,
            source_table_bytes,
            max_import_workspace_bytes,
            native_publication_payload_bytes,
            owner_scalar_workspace_bytes,
            seeded_window_extra_bytes,
            window_accumulator_bytes,
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
    AdmissionPool::new(Capacity::Decoder.max_process_publication_bytes());
static CAPSULE_PUBLICATIONS: AdmissionPool = AdmissionPool::new(MAX_PROCESS_PUBLICATION_BYTES);

struct PublicationPermit {
    pool: &'static AdmissionPool,
    bytes: usize,
    _capsule: Option<CapsulePublicationPermit>,
}

// Separate fixed-size sub-permit preserves the Capsule aggregate cap while all
// profiles also contend for the one Decoder-sized process ceiling.
struct CapsulePublicationPermit {
    bytes: usize,
}
impl Drop for CapsulePublicationPermit {
    fn drop(&mut self) {
        CAPSULE_PUBLICATIONS
            .bytes
            .fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

impl PublicationPermit {
    fn acquire(
        pool: &'static AdmissionPool,
        bytes: usize,
        capacity: Capacity,
    ) -> Result<Self, Error> {
        let capsule = if capacity == Capacity::Capsule && std::ptr::eq(pool, &PROCESS_PUBLICATIONS)
        {
            CAPSULE_PUBLICATIONS
                .bytes
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                    used.checked_add(bytes)
                        .filter(|&next| next <= CAPSULE_PUBLICATIONS.limit)
                })
                .map_err(|_| Error::Limit)?;
            Some(CapsulePublicationPermit { bytes })
        } else {
            None
        };
        pool.bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes).filter(|&next| next <= pool.limit)
            })
            .map_err(|_| Error::Limit)?;
        Ok(Self {
            pool,
            bytes,
            _capsule: capsule,
        })
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
/// values are retained. Fields are immutable; wire import requires an independently
/// authenticated expected digest. The owner digest binds graph/checkpoint context,
/// numeric profile, bounds, dimensions, matrix/table identities, all R/S and
/// every row hash. A digest is an identity, not a signature/authentication channel.
pub struct Publication {
    pub(super) capacity: Capacity,
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
    window_started: AtomicBool,
    pub(super) export_quota: Option<Arc<super::window::wire::WireQuota>>,
    _admission: PublicationPermit,
}

impl Publication {
    fn new_export_quota(
        ownership: PublicationOwnership,
    ) -> Option<Arc<super::window::wire::WireQuota>> {
        (ownership == PublicationOwnership::StreamedSingleWindow)
            .then(|| Arc::new(super::window::wire::WireQuota::new()))
    }
    /// Final encoded source owners must already be admitted. This path performs
    /// no eager from_f64 staging and permits one lifetime window attempt only.
    pub fn publish_streamed_single_window(
        context: [u8; 32],
        stages: &[Stage<'_>],
        table: &FloatTable,
        capacity: Capacity,
    ) -> Result<Self, Error> {
        Self::publish_owned_in_pool(
            context,
            stages,
            table,
            &mut OsEntropy,
            &PROCESS_PUBLICATIONS,
            capacity,
            PublicationOwnership::StreamedSingleWindow,
        )
    }

    pub fn publish(
        context: [u8; 32],
        stages: &[Stage<'_>],
        table: &FloatTable,
    ) -> Result<Self, Error> {
        Self::publish_with_capacity(context, stages, table, Capacity::Capsule)
    }

    pub fn publish_with_capacity(
        context: [u8; 32],
        stages: &[Stage<'_>],
        table: &FloatTable,
        capacity: Capacity,
    ) -> Result<Self, Error> {
        Self::publish_in_pool_with_capacity(
            context,
            stages,
            table,
            &mut OsEntropy,
            &PROCESS_PUBLICATIONS,
            capacity,
        )
    }

    #[cfg(test)]
    pub(super) fn publish_with(
        context: [u8; 32],
        stages: &[Stage<'_>],
        table: &FloatTable,
        rng: &mut impl Entropy,
    ) -> Result<Self, Error> {
        Self::publish_in_pool(context, stages, table, rng, &PROCESS_PUBLICATIONS)
    }

    #[cfg(test)]
    pub(super) fn publish_in_pool(
        context: [u8; 32],
        stages: &[Stage<'_>],
        table: &FloatTable,
        rng: &mut impl Entropy,
        pool: &'static AdmissionPool,
    ) -> Result<Self, Error> {
        Self::publish_in_pool_with_capacity(context, stages, table, rng, pool, Capacity::Capsule)
    }

    pub(super) fn publish_in_pool_with_capacity(
        context: [u8; 32],
        stages: &[Stage<'_>],
        table: &FloatTable,
        rng: &mut impl Entropy,
        pool: &'static AdmissionPool,
        capacity: Capacity,
    ) -> Result<Self, Error> {
        Self::publish_owned_in_pool(
            context,
            stages,
            table,
            rng,
            pool,
            capacity,
            PublicationOwnership::General,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn publish_owned_in_pool(
        context: [u8; 32],
        stages: &[Stage<'_>],
        table: &FloatTable,
        rng: &mut impl Entropy,
        pool: &'static AdmissionPool,
        capacity: Capacity,
        ownership: PublicationOwnership,
    ) -> Result<Self, Error> {
        if stages.is_empty() || stages.len() > capacity.max_stages() {
            return Err(Error::Limit);
        }
        for (index, stage) in stages.iter().enumerate() {
            if stages[..index].iter().any(|prior| prior.id == stage.id) {
                return Err(Error::Binding);
            }
        }
        let resources = PublicationResources::for_shape_iter_owned(
            stages.iter().map(|stage| stage.matrix.shape()),
            table.vocabulary,
            table.width,
            capacity,
            ownership,
        )?;
        let admission =
            PublicationPermit::acquire(pool, resources.admitted_payload_bytes, capacity)?;
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
            capacity,
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
            window_started: AtomicBool::new(false),
            export_quota: Self::new_export_quota(ownership),
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

    pub fn capacity(&self) -> Capacity {
        self.capacity
    }

    pub(super) fn reserve_window(&self) -> Result<(), Error> {
        if self.resources.ownership == PublicationOwnership::StreamedSingleWindow {
            self.window_started
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .map_err(|_| Error::Retired)?;
        }
        Ok(())
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

#[cfg(test)]
#[path = "streamed_resources_tests.rs"]
mod streamed_tests;
