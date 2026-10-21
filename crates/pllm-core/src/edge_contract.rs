//! Allocation-free, method-neutral public edge metadata.
//!
//! These values describe declarations, not runtime capabilities. Source identities
//! are opaque and unauthenticated; even all-zero identities are valid metadata.
//! Weight visibility describes ownership, not authorization. Constructors check
//! only geometry, numeric domains, and representation compatibility.

use sha2::{Digest, Sha256};

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
        let mut elements = 1usize;
        let mut inline = [0; 4];
        for (index, &dimension) in dimensions.iter().enumerate() {
            // Canonical dimensions and counts use u64 even on other hosts.
            u64::try_from(dimension).map_err(|_| Error::Overflow)?;
            elements = elements.checked_mul(dimension).ok_or(Error::Overflow)?;
            inline[index] = dimension;
        }
        u64::try_from(elements).map_err(|_| Error::Overflow)?;
        Ok(Self {
            dimensions: inline,
            rank: dimensions.len() as u8,
            elements,
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
        let mut digest = Sha256::new();
        digest.update(b"pllm.edge-contract\0v1\0");
        for field in [
            self.identity.source.checkpoint,
            self.identity.source.numeric_profile,
            self.identity.source.implementation,
            self.identity.plan,
            self.identity.producer,
            self.identity.wire,
        ] {
            digest.update(32u64.to_le_bytes());
            digest.update(field);
        }
        digest.update(self.identity.first_tensor_index.to_le_bytes());
        digest.update([self.shape.rank]);
        for &dimension in self.shape.dimensions() {
            // Shape construction checked that dimensions and counts fit u64.
            digest.update((dimension as u64).to_le_bytes());
        }
        digest.update((self.shape.elements as u64).to_le_bytes());
        digest.update([
            self.numeric.ring_bits,
            self.numeric.fractional_bits,
            match self.numeric.interpretation {
                Interpretation::Signed => 0,
                Interpretation::Unsigned => 1,
                Interpretation::Predicate => 2,
            },
            match self.representation {
                Representation::Public => 0,
                Representation::AdditiveShare => 1,
                Representation::XorShare => 2,
                Representation::CommonAdditiveMasked => 3,
                Representation::CommonXorMasked => 4,
            },
            match self.weight_visibility {
                None => 0,
                Some(WeightVisibility::Private) => 1,
                Some(WeightVisibility::Public) => 2,
            },
        ]);
        digest.finalize().into()
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
        let mut digest = Sha256::new();
        digest.update(b"pllm.consumer-use\0v1\0");
        digest.update(self.edge.digest());
        digest.update(self.consumer);
        digest.update(self.operand_position.to_le_bytes());
        digest.update(self.ordinal.to_le_bytes());
        digest.finalize().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
