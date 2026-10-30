use super::*;

#[test]
fn ordinary_quote_refuses_invalid_descriptor_without_admission() {
    assert_eq!(quote(&[]), Err(Error::Resource));
    let ring = CoefficientRing::new(4096, (1_u64 << 61) - 1).unwrap();
    assert_eq!(ring.degree(), 4096);
    assert!(CoefficientRing::new(8192, 17).is_err());
    assert!(CoefficientRing::new(4, 1_u64 << 61).is_err());
}

#[test]
fn expanded_public_ring_does_not_expand_extraction_descriptor_admission() {
    use crate::coefficient_layout::DegreeBound;
    let ring = CoefficientRing::bounded(8192, 65537, DegreeBound::Public8192).unwrap();
    let limb = Limb { id: 1, modulus: ring.modulus() };
    let key = KeyDomain { label: 9, dimension: ring.degree() };
    let scale = Scale { numerator: 1, denominator: 1 };
    let case = Case {
        source: Source {
            degree: ring.degree(),
            rank: 1,
            limbs: [Some(limb), None],
            limb_count: 1,
            layout: Layout::OrderedLimbBA,
            sign: Sign::BPlusAS,
            scale,
            key,
        },
        destination: Destination {
            degree: ring.degree(),
            rank: 1,
            limb,
            layout: Layout::ConstantBASecretOrder,
            sign: Sign::BPlusAS,
            scale,
            key,
        },
        selected: 0,
    };
    assert_eq!(case.quote(), Err(Error::Descriptor));
    assert_eq!(quote(&[case]), Err(Error::Descriptor));
    let input = [u64::MAX];
    let mut output = [77];
    assert_eq!(extract(case, &input, &mut output).err(), Some(Error::Descriptor));
    assert_eq!(input, [u64::MAX]);
    assert_eq!(output, [77]);
}

#[test]
fn ordinary_algebra_repeats_without_admission_and_cleans_output() {
    let limb = Limb { id: 1, modulus: 17 };
    let key = KeyDomain {
        label: 9,
        dimension: 1,
    };
    let scale = Scale {
        numerator: 3,
        denominator: 2,
    };
    let case = Case {
        source: Source {
            degree: 1,
            rank: 1,
            limbs: [Some(limb), None],
            limb_count: 1,
            layout: Layout::OrderedLimbBA,
            sign: Sign::BPlusAS,
            scale,
            key,
        },
        destination: Destination {
            degree: 1,
            rank: 1,
            limb,
            layout: Layout::ConstantBASecretOrder,
            sign: Sign::BPlusAS,
            scale,
            key,
        },
        selected: 0,
    };
    let input = [8, 3];
    let mut output = [99; 2];
    let extraction = extract(case, &input, &mut output).unwrap();
    assert_eq!(extraction.words(), &[8, 3]);
    assert_eq!(extraction.check_public_relation(&[2]), Ok((14, 14)));
    assert_eq!(extraction.check_public_relation(&[2]), Ok((14, 14)));
    drop(extraction);
    assert_eq!(output, [0; 2]);
}
