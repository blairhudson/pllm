//! Model-/method-neutral selected original-source byte verification.
//!
//! An expected pin is an independently trusted publisher/configuration premise,
//! not cryptographic publisher authentication. No traits, callbacks, source I/O,
//! native permissions, quantization policy or deserialization authority exist.
//! Config and semantic plan digests are frozen in the expected tuple. Supplied
//! bytes cannot redefine their expected artifact, location or semantic mapping.
use pllm_types::{Digest, NamedDigest};
use serde::Serialize;
use sha2::{Digest as _, Sha256};
use std::mem::size_of;
use std::ops::{Deref, DerefMut};
use zeroize::Zeroize;

/// Internal failures borrow static messages: no owned error metadata or allocator
/// call. Public compatibility APIs allocate String only at their outer boundary.
/// Successful import still allocates its fixed source backing, AFTER caller
/// admission; this error contract is not an aggregate resource/admission proof.
pub(crate) type SourceError = &'static str;

/// Actual reusable scalar storage, not an estimated decoder allowance. Borrowed
/// input and scalar evaluation temporaries are outside this owned-state object.
pub(crate) struct SourceDecodeWorkspace {
    bytes: [u8; 4],
}
impl SourceDecodeWorkspace {
    pub(crate) const fn new() -> Self {
        Self { bytes: [0; 4] }
    }
}
impl Drop for SourceDecodeWorkspace {
    fn drop(&mut self) {
        self.bytes.zeroize();
    }
}

/// Caller-owned fixed state for verification AND source commitment. One SHA
/// state is reused, with both digest results retained explicitly. Caller must
/// include this exact object and fixed backing initialization/move overlap in
/// its reservation BEFORE constructing/using it. This is not itself admission.
pub(crate) struct SelectedSourceWorkspace {
    hash: Sha256,
    actual_digest: [u8; 32],
    binding: [u8; 32],
    pub(crate) decode: SourceDecodeWorkspace,
}
impl SelectedSourceWorkspace {
    pub(crate) fn new() -> Self {
        Self {
            hash: Sha256::new(),
            actual_digest: [0; 32],
            binding: [0; 32],
            decode: SourceDecodeWorkspace::new(),
        }
    }
}

/// Fixed logical allocation, wiped elementwise with each value's Zeroize impl.
/// Drop wipes BEFORE Box field deallocation; an enclosing admission guard
/// must be declared after its owned buffers so guard refund occurs last.
pub(crate) struct ZeroizingFixed<T: Zeroize, const N: usize> {
    values: Box<[T; N]>,
}
impl<T: Zeroize + Default + Copy, const N: usize> ZeroizingFixed<T, N> {
    pub(crate) fn new_zeroed() -> Self {
        Self {
            values: Box::new([T::default(); N]),
        }
    }
}
impl<T: Zeroize, const N: usize> Deref for ZeroizingFixed<T, N> {
    type Target = [T; N];
    fn deref(&self) -> &Self::Target {
        self.values.as_ref()
    }
}
impl<T: Zeroize, const N: usize> DerefMut for ZeroizingFixed<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.values.as_mut()
    }
}
impl<T: Zeroize, const N: usize> Zeroize for ZeroizingFixed<T, N> {
    fn zeroize(&mut self) {
        for value in self.values.iter_mut() {
            value.zeroize();
        }
    }
}
impl<T: Zeroize, const N: usize> Drop for ZeroizingFixed<T, N> {
    fn drop(&mut self) {
        self.zeroize();
    }
}

pub const MAX_SELECTED_SOURCE_ELEMENTS: usize = 256;
pub const SELECTED_SOURCE_BYTE_CAPACITY: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum SourceDtype {
    Bf16,
    F32,
}
impl SourceDtype {
    pub fn bytes(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum SourceEndian {
    Little,
    Big,
}

/// Publisher premises. Config/plan bytes are raw SHA-256 identities converted
/// from existing Digest values, not newly authenticated by this tuple.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectedSourceTuple {
    pub checkpoint: String,
    pub immutable_revision: String,
    pub tensor: String,
    pub original_shape: [usize; 2],
    pub output_coordinates: Vec<usize>,
    pub inner_coordinates: Vec<usize>,
    pub output_input_row_major: bool,
    pub dtype: SourceDtype,
    pub endian: SourceEndian,
    pub tensor_offset: u64,
    pub expected_config: [u8; 32],
    pub expected_semantic_plan: [u8; 32],
}

pub fn digest_identity(value: &Digest) -> Result<[u8; 32], String> {
    checked_digest_identity(value).map_err(String::from)
}

pub(crate) fn checked_digest_identity(value: &Digest) -> Result<[u8; 32], SourceError> {
    checked_digest_text_identity(value.as_str())
}

fn checked_digest_text_identity(text: &str) -> Result<[u8; 32], SourceError> {
    let text = text.as_bytes();
    if text.len() != 64 {
        return Err("invalid expected digest");
    }
    let nibble = |v: u8| match v {
        b'0'..=b'9' => Ok(v - b'0'),
        b'a'..=b'f' => Ok(v - b'a' + 10),
        _ => Err("invalid expected digest"),
    };
    let mut result = [0; 32];
    for (out, pair) in result.iter_mut().zip(text.chunks_exact(2)) {
        *out = nibble(pair[0])? * 16 + nibble(pair[1])?;
    }
    Ok(result)
}

pub(crate) fn sum_bytes(values: impl IntoIterator<Item = usize>) -> Result<usize, String> {
    checked_sum_bytes(values).map_err(String::from)
}

pub(crate) fn checked_sum_bytes(
    values: impl IntoIterator<Item = usize>,
) -> Result<usize, SourceError> {
    values.into_iter().try_fold(0usize, |n, v| {
        n.checked_add(v).ok_or("resource byte overflow")
    })
}
#[cfg(test)]
pub(crate) fn vector_bytes<T>(values: &Vec<T>) -> Result<usize, String> {
    checked_vector_bytes(values).map_err(String::from)
}

pub(crate) fn checked_vector_bytes<T>(values: &Vec<T>) -> Result<usize, SourceError> {
    values
        .capacity()
        .checked_mul(size_of::<T>())
        .ok_or("vector backing overflow")
}

impl SelectedSourceTuple {
    pub fn metadata_backing_bytes(&self) -> Result<usize, String> {
        self.checked_metadata_backing_bytes().map_err(String::from)
    }

    pub(crate) fn checked_metadata_backing_bytes(&self) -> Result<usize, SourceError> {
        checked_sum_bytes([
            self.checkpoint.capacity(),
            self.immutable_revision.capacity(),
            self.tensor.capacity(),
            checked_vector_bytes(&self.output_coordinates)?,
            checked_vector_bytes(&self.inner_coordinates)?,
        ])
    }
    pub(crate) fn checked_count(&self) -> Result<usize, SourceError> {
        for text in [&self.checkpoint, &self.immutable_revision, &self.tensor] {
            if text.is_empty() || text.len() > 1024 {
                return Err("invalid source identity string");
            }
        }
        if !matches!(self.immutable_revision.len(), 40 | 64)
            || !self
                .immutable_revision
                .bytes()
                .all(|v| v.is_ascii_hexdigit())
            || !self.output_input_row_major
            || self.endian != SourceEndian::Little
            || self.tensor_offset % self.dtype.bytes() as u64 != 0
        {
            return Err("unsupported source revision/layout/endian/offset");
        }
        for (coords, extent) in [
            (&self.output_coordinates, self.original_shape[0]),
            (&self.inner_coordinates, self.original_shape[1]),
        ] {
            if coords.is_empty()
                || coords.len() > 256
                || extent == 0
                || coords
                    .iter()
                    .enumerate()
                    .any(|(i, v)| *v >= extent || coords[..i].contains(v))
            {
                return Err("invalid ordered source selection");
            }
        }
        let original = self.original_shape[0]
            .checked_mul(self.original_shape[1])
            .and_then(|v| u64::try_from(v).ok())
            .ok_or("original source extent overflow")?;
        self.tensor_offset
            .checked_add(
                original
                    .checked_mul(self.dtype.bytes() as u64)
                    .ok_or("source extent overflow")?,
            )
            .ok_or("source offset overflow")?;
        let count = self
            .output_coordinates
            .len()
            .checked_mul(self.inner_coordinates.len())
            .ok_or("source count overflow")?;
        if count > MAX_SELECTED_SOURCE_ELEMENTS {
            return Err("selected source exceeds 256 elements");
        }
        Ok(count)
    }
    fn checked_offset(&self, index: usize) -> Result<u64, SourceError> {
        let row = self.output_coordinates[index / self.inner_coordinates.len()];
        let column = self.inner_coordinates[index % self.inner_coordinates.len()];
        let element = row
            .checked_mul(self.original_shape[1])
            .and_then(|v| v.checked_add(column))
            .and_then(|v| u64::try_from(v).ok())
            .ok_or("source coordinate overflow")?;
        element
            .checked_mul(self.dtype.bytes() as u64)
            .and_then(|v| self.tensor_offset.checked_add(v))
            .ok_or("source offset overflow")
    }
}

/// No opaque Digest String/Arc backing is retained: digest identities are parsed
/// into fixed arrays at this boundary; the original NamedDigest is consumed.
pub struct ExpectedSelectedSource {
    artifact_name: String,
    digest: [u8; 32],
    tuple: SelectedSourceTuple,
    selected_bytes: usize,
}
impl ExpectedSelectedSource {
    pub fn from_trusted_publisher(
        artifact: NamedDigest,
        tuple: SelectedSourceTuple,
        selected_bytes: usize,
    ) -> Result<Self, String> {
        Self::checked_from_trusted_publisher(artifact, tuple, selected_bytes).map_err(String::from)
    }

    pub(crate) fn checked_from_trusted_publisher(
        artifact: NamedDigest,
        tuple: SelectedSourceTuple,
        selected_bytes: usize,
    ) -> Result<Self, SourceError> {
        let count = tuple.checked_count()?;
        if artifact.id.is_empty()
            || artifact.id.len() > 1024
            || count.checked_mul(tuple.dtype.bytes()) != Some(selected_bytes)
        {
            return Err("invalid independently expected source pin");
        }
        let digest = checked_digest_identity(&artifact.digest)?;
        let result = Self {
            artifact_name: artifact.id,
            digest,
            tuple,
            selected_bytes,
        };
        if result.checked_metadata_backing_bytes()? > 64 * 1024 {
            return Err("selected-source metadata capacity limit exceeded");
        }
        Ok(result)
    }
    pub fn tuple(&self) -> &SelectedSourceTuple {
        &self.tuple
    }
    pub fn artifact_name(&self) -> &str {
        &self.artifact_name
    }
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
    pub fn selected_bytes(&self) -> usize {
        self.selected_bytes
    }
    pub fn scalar_artifact_offset(&self, index: usize) -> Result<u64, String> {
        self.checked_scalar_artifact_offset(index)
            .map_err(String::from)
    }

    pub(crate) fn checked_scalar_artifact_offset(&self, index: usize) -> Result<u64, SourceError> {
        if index >= self.tuple.checked_count()? {
            return Err("source offset index out of bounds");
        }
        self.tuple.checked_offset(index)
    }
    pub fn metadata_backing_bytes(&self) -> Result<usize, String> {
        self.checked_metadata_backing_bytes().map_err(String::from)
    }

    pub(crate) fn checked_metadata_backing_bytes(&self) -> Result<usize, SourceError> {
        checked_sum_bytes([
            self.artifact_name.capacity(),
            self.tuple.checked_metadata_backing_bytes()?,
        ])
    }
    /// Calculable before source copies. Fixed source backing is always 1024
    /// bytes, including unused zero bytes. Allocator headers/RSS are excluded.
    pub fn import_quote(&self) -> Result<SelectedSourceResources, String> {
        self.checked_import_quote().map_err(String::from)
    }

    pub(crate) fn checked_import_quote(&self) -> Result<SelectedSourceResources, SourceError> {
        SelectedSourceResources::checked_new(self.checked_metadata_backing_bytes()?)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectedSourceResources {
    pub owner_metadata_bytes: usize,
    pub metadata_backing_bytes: usize,
    pub source_byte_capacity: usize,
    pub retained_bytes: usize,
    pub import_scratch_bytes: usize,
    pub import_overlap_bytes: usize,
    /// No Arc, owned f64 Vec, coefficient Vec or offsets Vec exists here.
    pub arc_backing_bytes: usize,
}
impl SelectedSourceResources {
    fn checked_new(metadata_backing_bytes: usize) -> Result<Self, SourceError> {
        let owner_metadata_bytes = size_of::<VerifiedSelectedSource>();
        let retained_bytes = checked_sum_bytes([
            owner_metadata_bytes,
            metadata_backing_bytes,
            SELECTED_SOURCE_BYTE_CAPACITY,
        ])?;
        // Direct workspace verifier phase: initialization array and origin /
        // return inline slots coexist with the destination owner counted above.
        // These are fixed Rust values, not an allocator/RSS or compiler bound.
        let import_scratch_bytes = checked_sum_bytes([
            size_of::<SelectedSourceWorkspace>(),
            // The Box initializer and completed hash output/move are disjoint:
            // byte hash -> initialization -> binding hash, one reusable SHA.
            SELECTED_SOURCE_BYTE_CAPACITY.max(checked_sum_bytes([
                size_of::<sha2::digest::Output<Sha256>>(),
                size_of::<[u8; 32]>(),
            ])?),
            size_of::<ExpectedSelectedSource>(),
            size_of::<VerifiedSelectedSource>(),
            size_of::<ZeroizingFixed<u8, SELECTED_SOURCE_BYTE_CAPACITY>>(),
        ])?;
        Ok(Self {
            owner_metadata_bytes,
            metadata_backing_bytes,
            source_byte_capacity: SELECTED_SOURCE_BYTE_CAPACITY,
            retained_bytes,
            import_scratch_bytes,
            import_overlap_bytes: checked_sum_bytes([retained_bytes, import_scratch_bytes])?,
            arc_backing_bytes: 0,
        })
    }
}

pub struct SelectedSourceBytes<'a> {
    pub artifact_name: &'a str,
    pub tuple: &'a SelectedSourceTuple,
    pub scalar_artifact_offsets: &'a [u64],
    pub bytes: &'a [u8],
}

/// Opaque verified owner; cannot be cloned/deserialized/minted by a callback.
pub struct VerifiedSelectedSource {
    expected: ExpectedSelectedSource,
    bytes: ZeroizingFixed<u8, SELECTED_SOURCE_BYTE_CAPACITY>,
    binding: [u8; 32],
}
impl VerifiedSelectedSource {
    pub fn expected_tuple(&self) -> &SelectedSourceTuple {
        &self.expected.tuple
    }
    pub fn bytes(&self) -> &[u8] {
        &self.bytes[..self.expected.selected_bytes]
    }
    pub fn source_binding(&self) -> [u8; 32] {
        self.binding
    }
    pub fn resources(&self) -> Result<SelectedSourceResources, String> {
        self.checked_resources().map_err(String::from)
    }

    pub(crate) fn checked_resources(&self) -> Result<SelectedSourceResources, SourceError> {
        self.expected.checked_import_quote()
    }

    #[cfg(test)]
    pub(crate) fn checked_metadata_backing_bytes(&self) -> Result<usize, SourceError> {
        self.expected.checked_metadata_backing_bytes()
    }
    pub fn scalar_count(&self) -> usize {
        self.expected.selected_bytes / self.expected.tuple.dtype.bytes()
    }
    pub fn finite_value(&self, index: usize) -> Result<f64, String> {
        self.checked_finite_value(index).map_err(String::from)
    }

    pub(crate) fn checked_finite_value(&self, index: usize) -> Result<f64, SourceError> {
        self.checked_finite_value_with_workspace(index, &mut SourceDecodeWorkspace::new())
    }

    pub(crate) fn checked_finite_value_with_workspace(
        &self,
        index: usize,
        workspace: &mut SourceDecodeWorkspace,
    ) -> Result<f64, SourceError> {
        if index >= self.scalar_count() {
            return Err("source scalar out of bounds");
        }
        checked_decode_with_workspace(
            self.expected.tuple.dtype,
            &self.bytes()[index * self.expected.tuple.dtype.bytes()..]
                [..self.expected.tuple.dtype.bytes()],
            workspace,
        )
    }
}
fn checked_decode_with_workspace(
    dtype: SourceDtype,
    bytes: &[u8],
    workspace: &mut SourceDecodeWorkspace,
) -> Result<f64, SourceError> {
    if bytes.len() != dtype.bytes() {
        return Err("source scalar width");
    }
    workspace.bytes.fill(0);
    workspace.bytes[..bytes.len()].copy_from_slice(bytes);
    let value = match dtype {
        SourceDtype::Bf16 => f64::from(f32::from_bits(
            u32::from(u16::from_le_bytes([workspace.bytes[0], workspace.bytes[1]])) << 16,
        )),
        SourceDtype::F32 => f64::from(f32::from_le_bytes(workspace.bytes)),
    };
    if !value.is_finite() {
        return Err("nonfinite source scalar");
    }
    Ok(value)
}

pub fn verify_selected_source(
    expected: ExpectedSelectedSource,
    supplied: SelectedSourceBytes<'_>,
) -> Result<VerifiedSelectedSource, String> {
    checked_verify_selected_source(expected, supplied).map_err(String::from)
}

pub(crate) fn checked_verify_selected_source(
    expected: ExpectedSelectedSource,
    supplied: SelectedSourceBytes<'_>,
) -> Result<VerifiedSelectedSource, SourceError> {
    checked_verify_selected_source_with_workspace(
        expected,
        supplied,
        &mut SelectedSourceWorkspace::new(),
    )
}

pub(crate) fn checked_verify_selected_source_with_workspace(
    expected: ExpectedSelectedSource,
    supplied: SelectedSourceBytes<'_>,
    workspace: &mut SelectedSourceWorkspace,
) -> Result<VerifiedSelectedSource, SourceError> {
    if supplied.tuple != &expected.tuple {
        return Err("source config/plan/revision/selection/dtype/endian tuple mismatch");
    }
    checked_verify_pinned_source_bytes_with_workspace(
        expected,
        supplied.artifact_name,
        supplied.scalar_artifact_offsets,
        supplied.bytes,
        workspace,
    )
}

/// Sole pinned-byte verifier. Static error messages on EVERY rejection path;
/// caller must reserve before invoking it (including its first source hash).
pub(crate) fn checked_verify_pinned_source_bytes_with_workspace(
    expected: ExpectedSelectedSource,
    artifact_name: &str,
    offsets: &[u64],
    raw: &[u8],
    workspace: &mut SelectedSourceWorkspace,
) -> Result<VerifiedSelectedSource, SourceError> {
    if artifact_name != expected.artifact_name
        || raw.len() != expected.selected_bytes
        || offsets.len() != expected.tuple.checked_count()?
    {
        return Err("source artifact/config/plan/revision/selection/dtype/endian/offset mismatch");
    }
    for (i, &offset) in offsets.iter().enumerate() {
        if offset != expected.tuple.checked_offset(i)? {
            return Err("source scalar offset mismatch");
        }
    }
    workspace.hash.reset();
    workspace.hash.update(raw);
    workspace.actual_digest = workspace.hash.finalize_reset().into();
    if workspace.actual_digest != expected.digest {
        return Err("actual original source bytes differ from expected pin");
    }
    for bytes in raw.chunks_exact(expected.tuple.dtype.bytes()) {
        checked_decode_with_workspace(expected.tuple.dtype, bytes, &mut workspace.decode)?;
    }
    // No source copy until all tuple/hash/finite checks pass. Caller must hold
    // aggregate admission covering import_quote before invoking this function.
    let mut bytes = ZeroizingFixed::new_zeroed();
    bytes[..raw.len()].copy_from_slice(raw);
    source_commitment_with_workspace(&expected, raw, workspace);
    Ok(VerifiedSelectedSource {
        expected,
        bytes,
        binding: workspace.binding,
    })
}

/// Exact existing framing; borrowed metadata and fixed SHA state only. No pin
/// authentication or owner issuance follows from computing this commitment.
fn source_commitment_with_workspace(
    expected: &ExpectedSelectedSource,
    raw: &[u8],
    workspace: &mut SelectedSourceWorkspace,
) {
    workspace.hash.reset();
    let h = &mut workspace.hash;
    h.update(b"pllm.verified-selected-source.v1\0");
    for text in [
        &expected.artifact_name,
        &expected.tuple.checkpoint,
        &expected.tuple.immutable_revision,
        &expected.tuple.tensor,
    ] {
        h.update((text.len() as u64).to_le_bytes());
        h.update(text.as_bytes());
    }
    h.update(expected.digest);
    h.update(expected.tuple.expected_config);
    h.update(expected.tuple.expected_semantic_plan);
    for dimension in expected.tuple.original_shape {
        h.update((dimension as u64).to_le_bytes());
    }
    for coords in [
        &expected.tuple.output_coordinates,
        &expected.tuple.inner_coordinates,
    ] {
        h.update((coords.len() as u64).to_le_bytes());
        for &v in coords {
            h.update((v as u64).to_le_bytes());
        }
    }
    h.update([
        match expected.tuple.dtype {
            SourceDtype::Bf16 => 0,
            SourceDtype::F32 => 1,
        },
        0,
        1,
    ]);
    h.update(expected.tuple.tensor_offset.to_le_bytes());
    h.update(raw);
    workspace.binding = h.finalize_reset().into();
}

#[cfg(test)]
#[path = "verified_source/tests.rs"]
mod tests;
