//! Allocation-free, method-neutral public edge metadata.
//!
//! These values describe declarations, not runtime capabilities. Source identities
//! are opaque and unauthenticated; even all-zero identities are valid metadata.
//! Weight visibility describes ownership, not authorization. Constructors check
//! only geometry, numeric domains, and representation compatibility.

use sha2::{Digest, Sha256};

/// Fixed, reusable storage for canonical edge and consumer-use hashing.
///
/// Both hash states coexist during a use digest. Encoding buffers and hash
/// outputs are owned here; dimensions and identity bytes are only borrowed.
/// This is logical storage, not a promise about optimized stack usage or RSS.
pub struct EdgeContractWorkspace {
    edge_hash: Sha256,
    use_hash: Sha256,
    encoding: [u8; 8],
    tags: [u8; 5],
    edge_result: sha2::digest::Output<Sha256>,
    use_result: sha2::digest::Output<Sha256>,
}

impl Default for EdgeContractWorkspace {
    fn default() -> Self {
        Self {
            edge_hash: Sha256::new(),
            use_hash: Sha256::new(),
            encoding: [0; 8],
            tags: [0; 5],
            edge_result: Default::default(),
            use_result: Default::default(),
        }
    }
}

impl EdgeContractWorkspace {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn logical_bytes() -> usize {
        core::mem::size_of::<Self>()
    }

    fn edge_identity_field(&mut self, field: &[u8; 32]) {
        self.edge_u64(32);
        self.edge_hash.update(field);
    }

    fn edge_u64(&mut self, value: u64) {
        self.encoding = value.to_le_bytes();
        self.edge_hash.update(self.encoding.as_slice());
    }

    fn hash_edge(&mut self, edge: &EdgeDescriptor) {
        Digest::reset(&mut self.edge_hash);
        self.edge_hash.update(b"pllm.edge-contract\0v1\0");
        // No identity array or IntoIter owning six copied digests.
        self.edge_identity_field(&edge.identity.source.checkpoint);
        self.edge_identity_field(&edge.identity.source.numeric_profile);
        self.edge_identity_field(&edge.identity.source.implementation);
        self.edge_identity_field(&edge.identity.plan);
        self.edge_identity_field(&edge.identity.producer);
        self.edge_identity_field(&edge.identity.wire);
        self.edge_u64(edge.identity.first_tensor_index);
        self.tags[0] = edge.shape.rank;
        self.edge_hash.update(&self.tags[..1]);
        for &dimension in edge.shape.dimensions() {
            // Shape construction checked canonical u64 range.
            self.edge_u64(dimension as u64);
        }
        self.edge_u64(edge.shape.elements as u64);
        self.tags = [
            edge.numeric.ring_bits,
            edge.numeric.fractional_bits,
            match edge.numeric.interpretation {
                Interpretation::Signed => 0,
                Interpretation::Unsigned => 1,
                Interpretation::Predicate => 2,
            },
            match edge.representation {
                Representation::Public => 0,
                Representation::AdditiveShare => 1,
                Representation::XorShare => 2,
                Representation::CommonAdditiveMasked => 3,
                Representation::CommonXorMasked => 4,
            },
            match edge.weight_visibility {
                None => 0,
                Some(WeightVisibility::Private) => 1,
                Some(WeightVisibility::Public) => 2,
            },
        ];
        self.edge_hash.update(self.tags.as_slice());
        self.edge_hash.finalize_into_reset(&mut self.edge_result);
    }
}

// Named constructor input slots are also the paths used by legacy constructors.
// The quote reserves each input slot plus its output/Result slot even when a
// compiler can reuse their storage. Moved inputs have no second owner slot.
struct SourceIdentityConstructionPhase([u8; 32], [u8; 32], [u8; 32]);
struct EdgeIdentityConstructionPhase(SourceIdentity, [u8; 32], [u8; 32], [u8; 32], u64);
struct ShapeConstructionPhase {
    dimensions: [usize; 4],
    elements: usize,
}
struct NumericConstructionPhase(u8, u8, Interpretation);
struct EdgeDescriptorConstructionPhase(
    EdgeIdentity,
    Shape,
    Numeric,
    Representation,
    Option<WeightVisibility>,
);
struct ConsumerUseConstructionPhase(EdgeDescriptor, [u8; 32], u16, u32);

/// Owner-defined logical reservations for serial construction and hashing.
///
/// Constructor phases reserve concrete input/state and output slots (including
/// Result discriminants/padding). Shape input dimensions are borrowed: backing
/// storage, references, loop/control scalars, ABI temporaries, and SHA internals
/// beyond its concrete state are not part of this logical-slot convention.
/// Digest phases include the resident declaration, whole reusable workspace,
/// and returned 32-byte digest. Caller-retained copies and other live objects
/// must be added by the caller with checked arithmetic. These are exact logical
/// reservations, not physical stack bounds, allocation limits, or authority.
/// Prerequisite peaks assume staged inputs: source identity is moved into edge
/// identity; identity remains while shape and numeric are constructed; all
/// three move into the descriptor; consumer fields arrive at use construction.
/// If a digest workspace already exists during construction, its reservation
/// must also be added to the construction peak rather than using the serial
/// construction/digest maximum unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EdgeContractResourceQuote {
    pub source_identity_construction_bytes: usize,
    pub edge_identity_construction_bytes: usize,
    pub shape_construction_bytes: usize,
    pub numeric_construction_bytes: usize,
    pub edge_descriptor_construction_bytes: usize,
    pub consumer_use_construction_bytes: usize,
    /// Peak constructing prerequisites in order: source, identity, shape,
    /// numeric, descriptor. Completed prerequisites coexist with later phases.
    pub descriptor_construction_peak_bytes: usize,
    pub use_construction_peak_bytes: usize,
    pub descriptor_digest_bytes: usize,
    pub use_digest_bytes: usize,
    /// Construction workspace is released before digest workspace is reserved.
    pub descriptor_construction_digest_peak_bytes: usize,
    pub use_construction_digest_peak_bytes: usize,
}

impl EdgeContractResourceQuote {
    pub fn checked() -> Result<Self, Error> {
        use core::mem::size_of;
        fn sum(a: usize, b: usize) -> Result<usize, Error> {
            a.checked_add(b).ok_or(Error::Overflow)
        }
        fn phase<I, O>() -> Result<usize, Error> {
            sum(core::mem::size_of::<I>(), core::mem::size_of::<O>())
        }
        let source_identity_construction_bytes =
            phase::<SourceIdentityConstructionPhase, SourceIdentity>()?;
        let edge_identity_construction_bytes =
            phase::<EdgeIdentityConstructionPhase, EdgeIdentity>()?;
        let shape_construction_bytes = phase::<ShapeConstructionPhase, Result<Shape, Error>>()?;
        let numeric_construction_bytes =
            phase::<NumericConstructionPhase, Result<Numeric, Error>>()?;
        let edge_descriptor_construction_bytes =
            phase::<EdgeDescriptorConstructionPhase, Result<EdgeDescriptor, Error>>()?;
        let consumer_use_construction_bytes = phase::<ConsumerUseConstructionPhase, ConsumerUse>()?;
        let descriptor_construction_peak_bytes = source_identity_construction_bytes
            .max(edge_identity_construction_bytes)
            .max(sum(size_of::<EdgeIdentity>(), shape_construction_bytes)?)
            .max(sum(
                sum(size_of::<EdgeIdentity>(), size_of::<Shape>())?,
                numeric_construction_bytes,
            )?)
            .max(edge_descriptor_construction_bytes);
        let use_construction_peak_bytes =
            descriptor_construction_peak_bytes.max(consumer_use_construction_bytes);
        let digest_slots = sum(
            EdgeContractWorkspace::logical_bytes(),
            size_of::<[u8; 32]>(),
        )?;
        let descriptor_digest_bytes = sum(size_of::<EdgeDescriptor>(), digest_slots)?;
        let use_digest_bytes = sum(size_of::<ConsumerUse>(), digest_slots)?;
        Ok(Self {
            source_identity_construction_bytes,
            edge_identity_construction_bytes,
            shape_construction_bytes,
            numeric_construction_bytes,
            edge_descriptor_construction_bytes,
            consumer_use_construction_bytes,
            descriptor_construction_peak_bytes,
            use_construction_peak_bytes,
            descriptor_digest_bytes,
            use_digest_bytes,
            descriptor_construction_digest_peak_bytes: descriptor_construction_peak_bytes
                .max(descriptor_digest_bytes),
            use_construction_digest_peak_bytes: use_construction_peak_bytes.max(use_digest_bytes),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Domain,
    Overflow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Representation {
    Public,
    AdditiveShare,
    XorShare,
    CommonAdditiveMasked,
    CommonXorMasked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightVisibility {
    Private,
    Public,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interpretation {
    Signed,
    Unsigned,
    Predicate,
}

/// Opaque source declarations; construction does not authenticate their origin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceIdentity {
    checkpoint: [u8; 32],
    numeric_profile: [u8; 32],
    implementation: [u8; 32],
}

impl SourceIdentity {
    pub fn new(checkpoint: [u8; 32], numeric_profile: [u8; 32], implementation: [u8; 32]) -> Self {
        Self::from_phase(SourceIdentityConstructionPhase(
            checkpoint,
            numeric_profile,
            implementation,
        ))
    }

    fn from_phase(phase: SourceIdentityConstructionPhase) -> Self {
        let SourceIdentityConstructionPhase(checkpoint, numeric_profile, implementation) = phase;
        Self {
            checkpoint,
            numeric_profile,
            implementation,
        }
    }

    pub fn checkpoint(&self) -> [u8; 32] {
        self.checkpoint
    }

    pub fn numeric_profile(&self) -> [u8; 32] {
        self.numeric_profile
    }

    pub fn implementation(&self) -> [u8; 32] {
        self.implementation
    }
}

/// Nonempty shape of rank at most four, with checked element count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    dimensions: [usize; 4],
    rank: u8,
    elements: usize,
}

impl Shape {
    pub fn new(dimensions: &[usize]) -> Result<Self, Error> {
        if dimensions.is_empty() || dimensions.len() > 4 || dimensions.contains(&0) {
            return Err(Error::Domain);
        }
        let mut phase = ShapeConstructionPhase {
            dimensions: [0; 4],
            elements: 1,
        };
        for (index, &dimension) in dimensions.iter().enumerate() {
            // Canonical dimensions and counts use u64 even on other hosts.
            u64::try_from(dimension).map_err(|_| Error::Overflow)?;
            phase.elements = phase
                .elements
                .checked_mul(dimension)
                .ok_or(Error::Overflow)?;
            phase.dimensions[index] = dimension;
        }
        u64::try_from(phase.elements).map_err(|_| Error::Overflow)?;
        Ok(Self {
            dimensions: phase.dimensions,
            rank: dimensions.len() as u8,
            elements: phase.elements,
        })
    }

    pub fn dimensions(&self) -> &[usize] {
        &self.dimensions[..usize::from(self.rank)]
    }

    pub fn elements(&self) -> usize {
        self.elements
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Numeric {
    ring_bits: u8,
    fractional_bits: u8,
    interpretation: Interpretation,
}

impl Numeric {
    pub fn new(
        ring_bits: u8,
        fractional_bits: u8,
        interpretation: Interpretation,
    ) -> Result<Self, Error> {
        Self::from_phase(NumericConstructionPhase(
            ring_bits,
            fractional_bits,
            interpretation,
        ))
    }

    fn from_phase(phase: NumericConstructionPhase) -> Result<Self, Error> {
        let NumericConstructionPhase(ring_bits, fractional_bits, interpretation) = phase;
        if !(1..=64).contains(&ring_bits)
            || fractional_bits > ring_bits
            || (interpretation == Interpretation::Predicate
                && (ring_bits != 1 || fractional_bits != 0))
        {
            return Err(Error::Domain);
        }
        Ok(Self {
            ring_bits,
            fractional_bits,
            interpretation,
        })
    }

    pub fn ring_bits(&self) -> u8 {
        self.ring_bits
    }

    pub fn fractional_bits(&self) -> u8 {
        self.fractional_bits
    }

    pub fn interpretation(&self) -> Interpretation {
        self.interpretation
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EdgeIdentity {
    source: SourceIdentity,
    plan: [u8; 32],
    producer: [u8; 32],
    wire: [u8; 32],
    first_tensor_index: u64,
}

impl EdgeIdentity {
    pub fn new(
        source: SourceIdentity,
        plan: [u8; 32],
        producer: [u8; 32],
        wire: [u8; 32],
        first_tensor_index: u64,
    ) -> Self {
        Self::from_phase(EdgeIdentityConstructionPhase(
            source,
            plan,
            producer,
            wire,
            first_tensor_index,
        ))
    }

    fn from_phase(phase: EdgeIdentityConstructionPhase) -> Self {
        let EdgeIdentityConstructionPhase(source, plan, producer, wire, first_tensor_index) = phase;
        Self {
            source,
            plan,
            producer,
            wire,
            first_tensor_index,
        }
    }

    pub fn source(&self) -> SourceIdentity {
        self.source
    }

    pub fn plan(&self) -> [u8; 32] {
        self.plan
    }

    pub fn producer(&self) -> [u8; 32] {
        self.producer
    }

    pub fn wire(&self) -> [u8; 32] {
        self.wire
    }

    pub fn first_tensor_index(&self) -> u64 {
        self.first_tensor_index
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EdgeDescriptor {
    identity: EdgeIdentity,
    shape: Shape,
    numeric: Numeric,
    representation: Representation,
    weight_visibility: Option<WeightVisibility>,
}

impl EdgeDescriptor {
    pub fn new(
        identity: EdgeIdentity,
        shape: Shape,
        numeric: Numeric,
        representation: Representation,
        weight_visibility: Option<WeightVisibility>,
    ) -> Result<Self, Error> {
        Self::from_phase(EdgeDescriptorConstructionPhase(
            identity,
            shape,
            numeric,
            representation,
            weight_visibility,
        ))
    }

    fn from_phase(phase: EdgeDescriptorConstructionPhase) -> Result<Self, Error> {
        let EdgeDescriptorConstructionPhase(
            identity,
            shape,
            numeric,
            representation,
            weight_visibility,
        ) = phase;
        let predicate = numeric.interpretation == Interpretation::Predicate;
        let compatible = match representation {
            Representation::Public => true,
            Representation::AdditiveShare | Representation::CommonAdditiveMasked => !predicate,
            Representation::XorShare | Representation::CommonXorMasked => predicate,
        };
        if !compatible {
            return Err(Error::Domain);
        }
        let elements = u64::try_from(shape.elements).map_err(|_| Error::Overflow)?;
        identity
            .first_tensor_index
            .checked_add(elements)
            .ok_or(Error::Overflow)?;
        Ok(Self {
            identity,
            shape,
            numeric,
            representation,
            weight_visibility,
        })
    }

    pub fn identity(&self) -> EdgeIdentity {
        self.identity
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn numeric(&self) -> Numeric {
        self.numeric
    }

    pub fn representation(&self) -> Representation {
        self.representation
    }

    pub fn weight_visibility(&self) -> Option<WeightVisibility> {
        self.weight_visibility
    }

    /// SHA-256 of the canonical v1 metadata encoding, not proof of provenance.
    ///
    /// Encoding: domain `pllm.edge-contract\0v1\0`; then checkpoint, numeric
    /// profile, implementation, plan, producer, wire (each a u64 LE byte length
    /// of 32 followed by 32 bytes); u64 LE first tensor index; u8 rank; rank
    /// u64 LE dimensions; u64 LE element count; u8 ring bits, fractional bits,
    /// interpretation, representation, and visibility. Interpretation tags are
    /// signed=0, unsigned=1, predicate=2. Representation tags are public=0,
    /// additive-share=1, xor-share=2, common-additive-masked=3,
    /// common-xor-masked=4. Visibility tags are none=0, some-private=1,
    /// some-public=2. Tags never depend on Rust enum layout or native word size.
    pub fn digest(&self) -> [u8; 32] {
        self.digest_with_workspace(&mut EdgeContractWorkspace::new())
    }

    /// Same v1 digest using caller-owned fixed storage, without allocation.
    pub fn digest_with_workspace(&self, workspace: &mut EdgeContractWorkspace) -> [u8; 32] {
        workspace.hash_edge(self);
        workspace.edge_result.into()
    }
}

/// Allocation-free, unauthenticated declaration of one consumer operand use.
///
/// This is public metadata, not a capability or executable permission. Copying
/// it grants no ownership, fanout, or authorization. All-zero identities and
/// zero ordinals are valid declarations; no method-specific limits are imposed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConsumerUse {
    edge: EdgeDescriptor,
    consumer: [u8; 32],
    operand_position: u16,
    ordinal: u32,
}

impl ConsumerUse {
    pub fn new(
        edge: EdgeDescriptor,
        consumer: [u8; 32],
        operand_position: u16,
        ordinal: u32,
    ) -> Self {
        Self::from_phase(ConsumerUseConstructionPhase(
            edge,
            consumer,
            operand_position,
            ordinal,
        ))
    }

    fn from_phase(phase: ConsumerUseConstructionPhase) -> Self {
        let ConsumerUseConstructionPhase(edge, consumer, operand_position, ordinal) = phase;
        Self {
            edge,
            consumer,
            operand_position,
            ordinal,
        }
    }

    pub fn edge(&self) -> EdgeDescriptor {
        self.edge
    }

    pub fn consumer(&self) -> [u8; 32] {
        self.consumer
    }

    pub fn operand_position(&self) -> u16 {
        self.operand_position
    }

    pub fn ordinal(&self) -> u32 {
        self.ordinal
    }

    /// SHA-256 of the canonical v1 use declaration, not proof of authorization.
    ///
    /// Exact byte order: domain `pllm.consumer-use\0v1\0`, the full 32-byte
    /// [`EdgeDescriptor::digest`], the 32 consumer bytes in array order, u16 LE
    /// operand position, then u32 LE ordinal. No length prefixes or padding are
    /// added. The embedded edge digest retains its existing v1 encoding.
    pub fn digest(&self) -> [u8; 32] {
        self.digest_with_workspace(&mut EdgeContractWorkspace::new())
    }

    /// Same v1 digest using caller-owned fixed storage, without allocation.
    /// The outer use hash remains live while the nested edge hash runs.
    pub fn digest_with_workspace(&self, workspace: &mut EdgeContractWorkspace) -> [u8; 32] {
        Digest::reset(&mut workspace.use_hash);
        workspace.use_hash.update(b"pllm.consumer-use\0v1\0");
        workspace.hash_edge(&self.edge);
        workspace.use_hash.update(&workspace.edge_result[..]);
        workspace.use_hash.update(self.consumer.as_slice());
        workspace.encoding = u64::from(self.operand_position).to_le_bytes();
        workspace.use_hash.update(&workspace.encoding[..2]);
        workspace.encoding = u64::from(self.ordinal).to_le_bytes();
        workspace.use_hash.update(&workspace.encoding[..4]);
        workspace
            .use_hash
            .finalize_into_reset(&mut workspace.use_result);
        workspace.use_result.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_reuse_matches_independent_canonical_bytes() {
        let mut workspace = EdgeContractWorkspace::new();
        for rank in 1..=4 {
            for (interpretation, interpretation_tag) in [
                (Interpretation::Signed, 0),
                (Interpretation::Unsigned, 1),
                (Interpretation::Predicate, 2),
            ] {
                for (representation, representation_tag) in [
                    (Representation::Public, 0),
                    (Representation::AdditiveShare, 1),
                    (Representation::XorShare, 2),
                    (Representation::CommonAdditiveMasked, 3),
                    (Representation::CommonXorMasked, 4),
                ] {
                    for (visibility, visibility_tag) in [
                        (None, 0),
                        (Some(WeightVisibility::Private), 1),
                        (Some(WeightVisibility::Public), 2),
                    ] {
                        let dims = [2, 3, 4, 5];
                        let edge = EdgeDescriptor::new(
                            identity(7),
                            Shape::new(&dims[..rank]).unwrap(),
                            Numeric::new(1, 0, interpretation).unwrap(),
                            representation,
                            visibility,
                        );
                        let compatible = match representation_tag {
                            0 => true,
                            1 | 3 => interpretation_tag != 2,
                            _ => interpretation_tag == 2,
                        };
                        if !compatible {
                            assert_eq!(edge, Err(Error::Domain));
                            continue;
                        }
                        let edge = edge.unwrap();
                        // Independent buffered encoder: no workspace helpers,
                        // native enum layout, or production digest calls.
                        let mut bytes = b"pllm.edge-contract\0v1\0".to_vec();
                        for byte in 1..=6 {
                            bytes.extend_from_slice(&[32, 0, 0, 0, 0, 0, 0, 0]);
                            bytes.extend_from_slice(&[byte; 32]);
                        }
                        bytes.extend_from_slice(&[7, 0, 0, 0, 0, 0, 0, 0]);
                        bytes.push(rank as u8);
                        for &dimension in &dims[..rank] {
                            bytes.extend_from_slice(&(dimension as u64).to_le_bytes());
                        }
                        let count: usize = dims[..rank].iter().product();
                        bytes.extend_from_slice(&(count as u64).to_le_bytes());
                        bytes.extend_from_slice(&[
                            1,
                            0,
                            interpretation_tag,
                            representation_tag,
                            visibility_tag,
                        ]);
                        let expected_edge: [u8; 32] = Sha256::digest(&bytes).into();
                        assert_eq!(edge.digest_with_workspace(&mut workspace), expected_edge);
                        assert_eq!(edge.digest(), expected_edge);
                        let usage = ConsumerUse::new(edge, [255; 32], u16::MAX, u32::MAX);
                        let mut use_bytes = b"pllm.consumer-use\0v1\0".to_vec();
                        use_bytes.extend_from_slice(&expected_edge);
                        use_bytes.extend_from_slice(&[255; 32]);
                        use_bytes.extend_from_slice(&[255; 6]);
                        let expected_use: [u8; 32] = Sha256::digest(&use_bytes).into();
                        // Poison state and buffers to prove each invocation
                        // resets hashes and overwrites every consumed byte.
                        workspace.use_hash.update(b"stale outer state");
                        workspace.edge_hash.update(b"stale inner state");
                        workspace.encoding.fill(123);
                        workspace.tags.fill(123);
                        assert_eq!(usage.digest_with_workspace(&mut workspace), expected_use);
                        assert_eq!(usage.digest(), expected_use);
                        assert_eq!(edge.digest_with_workspace(&mut workspace), expected_edge);
                    }
                }
            }
        }
    }

    #[test]
    fn logical_resource_quotes_follow_concrete_slots_and_coexistence() {
        use core::mem::size_of;
        let quote = EdgeContractResourceQuote::checked().unwrap();
        assert_eq!(
            EdgeContractWorkspace::logical_bytes(),
            size_of::<EdgeContractWorkspace>()
        );
        // Whole concrete storage includes both coexisting hashes, encoding,
        // tags, both outputs, and any Rust layout padding. No hidden rank
        // buffer, copied identity array, or owning identity iterator exists.
        assert!(
            EdgeContractWorkspace::logical_bytes()
                >= size_of::<Sha256>()
                    + size_of::<Sha256>()
                    + size_of::<[u8; 8]>()
                    + size_of::<[u8; 5]>()
                    + size_of::<sha2::digest::Output<Sha256>>()
                    + size_of::<sha2::digest::Output<Sha256>>()
        );
        assert_eq!(
            quote.source_identity_construction_bytes,
            size_of::<SourceIdentityConstructionPhase>() + size_of::<SourceIdentity>()
        );
        assert_eq!(
            quote.edge_identity_construction_bytes,
            size_of::<EdgeIdentityConstructionPhase>() + size_of::<EdgeIdentity>()
        );
        assert_eq!(
            quote.shape_construction_bytes,
            size_of::<ShapeConstructionPhase>() + size_of::<Result<Shape, Error>>()
        );
        assert_eq!(
            quote.numeric_construction_bytes,
            size_of::<NumericConstructionPhase>() + size_of::<Result<Numeric, Error>>()
        );
        assert_eq!(
            quote.edge_descriptor_construction_bytes,
            size_of::<EdgeDescriptorConstructionPhase>()
                + size_of::<Result<EdgeDescriptor, Error>>()
        );
        assert_eq!(
            quote.consumer_use_construction_bytes,
            size_of::<ConsumerUseConstructionPhase>() + size_of::<ConsumerUse>()
        );
        let construction = [
            quote.source_identity_construction_bytes,
            quote.edge_identity_construction_bytes,
            size_of::<EdgeIdentity>() + quote.shape_construction_bytes,
            size_of::<EdgeIdentity>() + size_of::<Shape>() + quote.numeric_construction_bytes,
            quote.edge_descriptor_construction_bytes,
        ]
        .into_iter()
        .max()
        .unwrap();
        assert_eq!(quote.descriptor_construction_peak_bytes, construction);
        assert_eq!(
            quote.use_construction_peak_bytes,
            construction.max(quote.consumer_use_construction_bytes)
        );
        assert_eq!(
            quote.descriptor_digest_bytes,
            size_of::<EdgeDescriptor>()
                + size_of::<EdgeContractWorkspace>()
                + size_of::<[u8; 32]>()
        );
        assert_eq!(
            quote.use_digest_bytes,
            size_of::<ConsumerUse>() + size_of::<EdgeContractWorkspace>() + size_of::<[u8; 32]>()
        );
        assert_eq!(
            quote.descriptor_construction_digest_peak_bytes,
            construction.max(quote.descriptor_digest_bytes)
        );
        assert_eq!(
            quote.use_construction_digest_peak_bytes,
            quote
                .use_construction_peak_bytes
                .max(quote.use_digest_bytes)
        );
        // Resource declarations do not depend on shape rank, successful
        // validation, or metadata values. Rejections still use bounded slots.
        for dimensions in [&[][..], &[0], &[1, 2, 3, 4, 5], &[usize::MAX, 2]] {
            assert!(Shape::new(dimensions).is_err());
            assert_eq!(EdgeContractResourceQuote::checked().unwrap(), quote);
        }
        assert_eq!(
            Numeric::new(0, 0, Interpretation::Signed),
            Err(Error::Domain)
        );
        assert_eq!(
            EdgeDescriptor::new(
                identity(u64::MAX),
                Shape::new(&[1]).unwrap(),
                Numeric::new(1, 0, Interpretation::Signed).unwrap(),
                Representation::Public,
                None
            ),
            Err(Error::Overflow)
        );
    }

    fn identity(first_tensor_index: u64) -> EdgeIdentity {
        EdgeIdentity::new(
            SourceIdentity::new([1; 32], [2; 32], [3; 32]),
            [4; 32],
            [5; 32],
            [6; 32],
            first_tensor_index,
        )
    }

    fn descriptor() -> EdgeDescriptor {
        EdgeDescriptor::new(
            identity(7),
            Shape::new(&[2, 3]).unwrap(),
            Numeric::new(32, 8, Interpretation::Signed).unwrap(),
            Representation::Public,
            None,
        )
        .unwrap()
    }

    #[test]
    fn consumer_use_preserves_declarations_and_has_no_method_specific_caps() {
        fn check<T: Clone + Copy + Eq + core::fmt::Debug>(_: T) {}
        let zero = EdgeIdentity::new(
            SourceIdentity::new([0; 32], [0; 32], [0; 32]),
            [0; 32],
            [0; 32],
            [0; 32],
            0,
        );
        for visibility in [
            None,
            Some(WeightVisibility::Private),
            Some(WeightVisibility::Public),
        ] {
            let edge = EdgeDescriptor::new(
                zero,
                Shape::new(&[1024, 1024]).unwrap(),
                Numeric::new(64, 64, Interpretation::Unsigned).unwrap(),
                Representation::CommonAdditiveMasked,
                visibility,
            )
            .unwrap();
            for (consumer, position, ordinal) in [([0; 32], 0, 0), ([255; 32], u16::MAX, u32::MAX)]
            {
                let usage = ConsumerUse::new(edge, consumer, position, ordinal);
                check(usage);
                assert_eq!(usage.edge(), edge);
                assert_eq!(usage.consumer(), consumer);
                assert_eq!(usage.operand_position(), position);
                assert_eq!(usage.ordinal(), ordinal);
                assert_eq!(usage, ConsumerUse::new(edge, consumer, position, ordinal));
            }
        }
    }

    #[test]
    fn consumer_use_canonical_v1_bytes_and_digest_are_frozen() {
        // Frozen independently with Python hashlib.sha256 and struct.pack.
        let edge_digest = [
            0x73, 0x31, 0x5a, 0x69, 0xa2, 0x01, 0x77, 0x21, 0x96, 0xa8, 0x05, 0x5f, 0x86, 0xec,
            0xea, 0x9b, 0x0d, 0xd1, 0x7a, 0xf0, 0x75, 0x65, 0xa9, 0x99, 0x41, 0x53, 0x15, 0xfc,
            0xb1, 0xf1, 0x90, 0x59,
        ];
        let edge = descriptor();
        assert_eq!(edge.digest(), edge_digest);
        let consumer = core::array::from_fn(|index| index as u8);
        let usage = ConsumerUse::new(edge, consumer, 0x1234, 0x89ab_cdef);
        // Literal reference bytes: no production digest or integer encoders.
        let mut bytes = b"pllm.consumer-use\0v1\0".to_vec();
        bytes.extend_from_slice(&edge_digest);
        bytes.extend_from_slice(&[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 0x34, 0x12, 0xef, 0xcd, 0xab, 0x89,
        ]);
        let expected = [
            0x33, 0x3f, 0xf4, 0xe9, 0xe9, 0x3c, 0xb6, 0x80, 0x12, 0x8a, 0x39, 0xae, 0x68, 0x0c,
            0x04, 0x85, 0xfa, 0x98, 0xb2, 0xc1, 0x6d, 0xbe, 0x0e, 0xdc, 0x35, 0x1f, 0x3d, 0x70,
            0x94, 0x8d, 0x64, 0xa9,
        ];
        let reference: [u8; 32] = Sha256::digest(&bytes).into();
        assert_eq!(reference, expected);
        assert_eq!(usage.digest(), expected);
        assert_ne!(usage.digest(), edge_digest);
        assert_eq!(edge.digest(), edge_digest);
    }

    #[test]
    fn consumer_use_every_consumer_operand_and_ordinal_bit_affects_digest() {
        let base = ConsumerUse::new(descriptor(), [0; 32], 0, 0);
        for byte in 0..32 {
            for bit in 0..8 {
                let mut changed = base;
                changed.consumer[byte] ^= 1 << bit;
                assert_ne!(changed.digest(), base.digest());
            }
        }
        for bit in 0..16 {
            let changed = ConsumerUse::new(base.edge(), base.consumer(), 1u16 << bit, 0);
            assert_ne!(changed.digest(), base.digest());
        }
        for bit in 0..32 {
            let changed = ConsumerUse::new(base.edge(), base.consumer(), 0, 1u32 << bit);
            assert_ne!(changed.digest(), base.digest());
        }
    }

    #[test]
    fn consumer_use_binds_full_edge_digest_and_preserves_historical_edge_behavior() {
        let base = descriptor();
        let usage = ConsumerUse::new(base, [0; 32], 0, 0);
        let check = |edge: EdgeDescriptor| {
            let historical = edge.digest();
            assert_ne!(historical, base.digest());
            let changed = ConsumerUse::new(edge, usage.consumer(), 0, 0);
            assert_ne!(changed.digest(), usage.digest());
            assert_eq!(changed.edge(), edge);
            assert_eq!(changed.edge().digest(), historical);
        };
        for field in 0..6 {
            for byte in 0..32 {
                for bit in 0..8 {
                    let mut fields = [[1; 32], [2; 32], [3; 32], [4; 32], [5; 32], [6; 32]];
                    fields[field][byte] ^= 1 << bit;
                    let mut edge = base;
                    edge.identity = EdgeIdentity::new(
                        SourceIdentity::new(fields[0], fields[1], fields[2]),
                        fields[3],
                        fields[4],
                        fields[5],
                        7,
                    );
                    check(edge);
                }
            }
        }
        for bit in 0..64 {
            let mut edge = base;
            edge.identity = identity(7 ^ (1u64 << bit));
            check(edge);
        }
        for dimensions in [
            &[6][..],
            &[3, 2],
            &[1, 2, 3],
            &[2, 1, 3],
            &[2, 3, 1],
            &[1, 1, 2, 3],
            &[4, 3],
            &[2, 4],
            &[2, 3, 4, 5],
        ] {
            let mut edge = base;
            edge.shape = Shape::new(dimensions).unwrap();
            check(edge);
        }
        for numeric in [
            Numeric::new(64, 8, Interpretation::Signed).unwrap(),
            Numeric::new(32, 9, Interpretation::Signed).unwrap(),
            Numeric::new(32, 8, Interpretation::Unsigned).unwrap(),
            Numeric::new(1, 0, Interpretation::Predicate).unwrap(),
        ] {
            let mut edge = base;
            edge.numeric = numeric;
            check(edge);
        }
        for representation in [
            Representation::AdditiveShare,
            Representation::CommonAdditiveMasked,
        ] {
            let mut edge = base;
            edge.representation = representation;
            check(edge);
        }
        for representation in [Representation::XorShare, Representation::CommonXorMasked] {
            let mut edge = base;
            edge.numeric = Numeric::new(1, 0, Interpretation::Predicate).unwrap();
            edge.representation = representation;
            check(edge);
        }
        for visibility in [WeightVisibility::Private, WeightVisibility::Public] {
            let mut edge = base;
            edge.weight_visibility = Some(visibility);
            check(edge);
        }
    }

    #[test]
    fn metadata_is_copy_eq_and_debug() {
        fn check<T: Copy + Eq + core::fmt::Debug>(_: T) {}
        let edge = descriptor();
        check(Error::Domain);
        check(Error::Overflow);
        check(edge);
        check(edge.identity());
        check(edge.identity().source());
        check(edge.shape());
        check(edge.numeric());
        check(edge.representation());
        check(WeightVisibility::Private);
        check(Interpretation::Signed);
        assert_eq!(edge, edge);
    }

    #[test]
    fn accessors_preserve_declarations_and_accept_zero_source_ids() {
        let source = SourceIdentity::new([0; 32], [0; 32], [0; 32]);
        assert_eq!(source.checkpoint(), [0; 32]);
        assert_eq!(source.numeric_profile(), [0; 32]);
        assert_eq!(source.implementation(), [0; 32]);
        let id = EdgeIdentity::new(source, [4; 32], [5; 32], [6; 32], 7);
        assert_eq!(id.source(), source);
        assert_eq!(id.plan(), [4; 32]);
        assert_eq!(id.producer(), [5; 32]);
        assert_eq!(id.wire(), [6; 32]);
        assert_eq!(id.first_tensor_index(), 7);
        let shape = Shape::new(&[2, 3]).unwrap();
        let numeric = Numeric::new(32, 8, Interpretation::Signed).unwrap();
        let edge = EdgeDescriptor::new(
            id,
            shape,
            numeric,
            Representation::AdditiveShare,
            Some(WeightVisibility::Private),
        )
        .unwrap();
        assert_eq!(edge.identity(), id);
        assert_eq!(edge.shape(), shape);
        assert_eq!(edge.numeric(), numeric);
        assert_eq!(edge.representation(), Representation::AdditiveShare);
        assert_eq!(edge.weight_visibility(), Some(WeightVisibility::Private));
        assert_eq!(numeric.ring_bits(), 32);
        assert_eq!(numeric.fractional_bits(), 8);
        assert_eq!(numeric.interpretation(), Interpretation::Signed);
    }

    #[test]
    fn shape_accepts_each_rank_and_rejects_invalid_geometry() {
        for dimensions in [&[2][..], &[2, 3], &[2, 3, 4], &[2, 3, 4, 5]] {
            let shape = Shape::new(dimensions).unwrap();
            assert_eq!(shape.dimensions(), dimensions);
            assert_eq!(shape.elements(), dimensions.iter().product::<usize>());
            assert_eq!(shape, Shape::new(dimensions).unwrap());
        }
        for dimensions in [&[][..], &[1, 1, 1, 1, 1], &[0], &[1, 0, 1, 1]] {
            assert_eq!(Shape::new(dimensions), Err(Error::Domain));
        }
        for zero_index in 0..4 {
            let mut dimensions = [1; 4];
            dimensions[zero_index] = 0;
            assert_eq!(Shape::new(&dimensions), Err(Error::Domain));
        }
        assert_eq!(Shape::new(&[usize::MAX, 2]), Err(Error::Overflow));
        assert_eq!(Shape::new(&[1, usize::MAX, 1, 2]), Err(Error::Overflow));
        assert_eq!(Shape::new(&[usize::MAX]).unwrap().elements(), usize::MAX);
    }

    #[test]
    fn numeric_exhaustively_checks_ring_fraction_and_interpretation() {
        for interpretation in [
            Interpretation::Signed,
            Interpretation::Unsigned,
            Interpretation::Predicate,
        ] {
            for ring in 0..=u8::MAX {
                for fractional in 0..=u8::MAX {
                    let valid = (1..=64).contains(&ring)
                        && fractional <= ring
                        && (interpretation != Interpretation::Predicate
                            || (ring == 1 && fractional == 0));
                    let result = Numeric::new(ring, fractional, interpretation);
                    if valid {
                        let numeric = result.unwrap();
                        assert_eq!(numeric.ring_bits(), ring);
                        assert_eq!(numeric.fractional_bits(), fractional);
                        assert_eq!(numeric.interpretation(), interpretation);
                    } else {
                        assert_eq!(result, Err(Error::Domain));
                    }
                }
            }
        }
    }

    #[test]
    fn descriptor_checks_full_representation_and_visibility_matrix() {
        for interpretation in [
            Interpretation::Signed,
            Interpretation::Unsigned,
            Interpretation::Predicate,
        ] {
            // Signed/unsigned one-bit values are not predicates.
            let numeric = Numeric::new(1, 0, interpretation).unwrap();
            for representation in [
                Representation::Public,
                Representation::AdditiveShare,
                Representation::XorShare,
                Representation::CommonAdditiveMasked,
                Representation::CommonXorMasked,
            ] {
                let compatible = match representation {
                    Representation::Public => true,
                    Representation::AdditiveShare | Representation::CommonAdditiveMasked => {
                        interpretation != Interpretation::Predicate
                    }
                    Representation::XorShare | Representation::CommonXorMasked => {
                        interpretation == Interpretation::Predicate
                    }
                };
                for visibility in [
                    None,
                    Some(WeightVisibility::Private),
                    Some(WeightVisibility::Public),
                ] {
                    let result = EdgeDescriptor::new(
                        identity(0),
                        Shape::new(&[1]).unwrap(),
                        numeric,
                        representation,
                        visibility,
                    );
                    if compatible {
                        let edge = result.unwrap();
                        assert_eq!(edge.representation(), representation);
                        assert_eq!(edge.weight_visibility(), visibility);
                    } else {
                        assert_eq!(result, Err(Error::Domain));
                    }
                }
            }
        }
    }

    #[test]
    fn descriptor_checks_exclusive_end_overflow() {
        let base = descriptor();
        for (start, elements, valid) in [
            (u64::MAX, 1, false),
            (u64::MAX - 1, 1, true),
            (u64::MAX - 6, 6, true),
            (u64::MAX - 5, 6, false),
        ] {
            let result = EdgeDescriptor::new(
                identity(start),
                Shape::new(&[elements]).unwrap(),
                base.numeric(),
                base.representation(),
                None,
            );
            if valid {
                assert_eq!(result.unwrap().identity().first_tensor_index(), start);
            } else {
                assert_eq!(result, Err(Error::Overflow));
            }
        }
    }

    #[test]
    fn generic_metadata_has_no_method_specific_element_or_visibility_cap() {
        let base = descriptor();
        for dimensions in [&[257][..], &[1024, 1024], &[1, 1, 1, 65536]] {
            for visibility in [
                None,
                Some(WeightVisibility::Private),
                Some(WeightVisibility::Public),
            ] {
                let shape = Shape::new(dimensions).unwrap();
                let edge = EdgeDescriptor::new(
                    identity(0),
                    shape,
                    base.numeric(),
                    Representation::CommonAdditiveMasked,
                    visibility,
                )
                .unwrap();
                assert_eq!(edge.shape(), shape);
                assert_eq!(edge.weight_visibility(), visibility);
            }
        }
    }

    #[test]
    fn canonical_v1_encoding_is_explicit_and_stable() {
        let edge = descriptor();
        // Independent byte-level reference: no native usize or enum encoding.
        let mut expected = Sha256::new();
        expected.update(b"pllm.edge-contract\0v1\0");
        for byte in 1..=6u8 {
            expected.update([32, 0, 0, 0, 0, 0, 0, 0]);
            expected.update([byte; 32]);
        }
        expected.update([7, 0, 0, 0, 0, 0, 0, 0]);
        expected.update([2]);
        expected.update([2, 0, 0, 0, 0, 0, 0, 0]);
        expected.update([3, 0, 0, 0, 0, 0, 0, 0]);
        expected.update([6, 0, 0, 0, 0, 0, 0, 0]);
        expected.update([32, 8, 0, 0, 0]);
        let expected: [u8; 32] = expected.finalize().into();
        assert_eq!(edge.digest(), expected);
        assert_eq!(edge.digest(), descriptor().digest());
    }

    #[test]
    fn every_identity_byte_and_tensor_index_bit_affects_digest() {
        let base = descriptor();
        let digest = base.digest();
        for field in 0..6 {
            for byte in 0..32 {
                for bit in 0..8 {
                    let mut fields = [[1; 32], [2; 32], [3; 32], [4; 32], [5; 32], [6; 32]];
                    fields[field][byte] ^= 1 << bit;
                    let mut changed = base;
                    changed.identity = EdgeIdentity::new(
                        SourceIdentity::new(fields[0], fields[1], fields[2]),
                        fields[3],
                        fields[4],
                        fields[5],
                        7,
                    );
                    assert_ne!(changed.digest(), digest);
                }
            }
        }
        for bit in 0..64 {
            let mut changed = base;
            changed.identity = identity(7 ^ (1u64 << bit));
            assert_ne!(changed.digest(), digest);
        }
    }

    #[test]
    fn rank_dimension_order_and_each_geometry_field_affect_digest() {
        let base = descriptor();
        let digest = base.digest();
        // Same count, different rank/order/grouping must remain distinct.
        let shapes = [
            &[6][..],
            &[3, 2],
            &[1, 2, 3],
            &[2, 1, 3],
            &[2, 3, 1],
            &[1, 1, 2, 3],
            &[4, 3],
            &[2, 4],
        ];
        for (index, dimensions) in shapes.iter().enumerate() {
            let mut changed = base;
            changed.shape = Shape::new(dimensions).unwrap();
            assert_ne!(changed.digest(), digest);
            for earlier in &shapes[..index] {
                let mut other = base;
                other.shape = Shape::new(earlier).unwrap();
                assert_ne!(changed.digest(), other.digest());
            }
        }
        // Every dimension slot participates, including the fourth.
        let mut rank_four = base;
        rank_four.shape = Shape::new(&[2, 3, 4, 5]).unwrap();
        for index in 0..4 {
            let mut dimensions = [2, 3, 4, 5];
            dimensions[index] += 1;
            let mut changed = rank_four;
            changed.shape = Shape::new(&dimensions).unwrap();
            assert_ne!(changed.digest(), rank_four.digest());
        }
    }

    #[test]
    fn all_numeric_representation_and_visibility_tags_affect_digest() {
        let base = descriptor();
        for numeric in [
            Numeric::new(64, 8, Interpretation::Signed).unwrap(),
            Numeric::new(32, 9, Interpretation::Signed).unwrap(),
            Numeric::new(32, 8, Interpretation::Unsigned).unwrap(),
        ] {
            let mut changed = base;
            changed.numeric = numeric;
            assert_ne!(changed.digest(), base.digest());
        }
        for interpretation in [
            Interpretation::Signed,
            Interpretation::Unsigned,
            Interpretation::Predicate,
        ] {
            let edge = EdgeDescriptor::new(
                base.identity(),
                base.shape(),
                Numeric::new(1, 0, interpretation).unwrap(),
                Representation::Public,
                None,
            )
            .unwrap();
            for other in [
                Interpretation::Signed,
                Interpretation::Unsigned,
                Interpretation::Predicate,
            ] {
                let mut changed = edge;
                changed.numeric = Numeric::new(1, 0, other).unwrap();
                assert_eq!(edge.digest() == changed.digest(), interpretation == other);
            }
        }
        for numeric in [
            base.numeric(),
            Numeric::new(1, 0, Interpretation::Predicate).unwrap(),
        ] {
            let representations = if numeric.interpretation() == Interpretation::Predicate {
                [
                    Representation::Public,
                    Representation::XorShare,
                    Representation::CommonXorMasked,
                ]
            } else {
                [
                    Representation::Public,
                    Representation::AdditiveShare,
                    Representation::CommonAdditiveMasked,
                ]
            };
            let visibilities = [
                None,
                Some(WeightVisibility::Private),
                Some(WeightVisibility::Public),
            ];
            for (rep_index, representation) in representations.iter().enumerate() {
                for (vis_index, visibility) in visibilities.iter().enumerate() {
                    let edge = EdgeDescriptor::new(
                        base.identity(),
                        base.shape(),
                        numeric,
                        *representation,
                        *visibility,
                    )
                    .unwrap();
                    for (other_rep_index, other_rep) in representations.iter().enumerate() {
                        for (other_vis_index, other_vis) in visibilities.iter().enumerate() {
                            let other = EdgeDescriptor::new(
                                base.identity(),
                                base.shape(),
                                numeric,
                                *other_rep,
                                *other_vis,
                            )
                            .unwrap();
                            assert_eq!(
                                edge.digest() == other.digest(),
                                rep_index == other_rep_index && vis_index == other_vis_index,
                            );
                        }
                    }
                }
            }
        }
    }
}
