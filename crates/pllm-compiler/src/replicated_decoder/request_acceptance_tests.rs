use super::*;

fn fixture(sticky: bool) -> (Builder, usize, Vec<(usize, usize)>) {
    let mut builder = Builder::default();
    let input = builder
        .push(
            Code::Input("input.tokens".into()),
            3,
            Interval::new(0, 4).unwrap(),
        )
        .unwrap();
    if sticky {
        let valid = builder
            .push(
                Code::Input("sticky.valid".into()),
                1,
                Interval::new(0, 1).unwrap(),
            )
            .unwrap();
        builder.validity = Some(valid);
    }
    (builder, input, vec![(input, 2), (input, 0), (input, 1)])
}

#[test]
fn original_acceptance_preserves_order_sticky_owner_and_fallback_without_storing_validity() {
    for sticky in [false, true] {
        let (mut builder, input, mut selected) = fixture(sticky);
        let original = selected.clone();
        let validity = builder.validity;
        let start = builder.nodes.len();
        // A real sticky owner must bypass the fallback anchor entirely.
        let anchor_input = if sticky { usize::MAX } else { input };
        builder
            .final_acceptance(anchor_input, 3, &mut selected)
            .unwrap();
        let (valid, token_node) = if let Some(valid) = validity {
            (valid, start)
        } else {
            let Code::Gather(words) = &builder.nodes[start].code else {
                panic!("fallback must gather the original first input coordinate")
            };
            assert_eq!(words.iter().collect::<Vec<_>>(), [(input, 0)]);
            let Code::Affine {
                left,
                right,
                coefficients,
            } = &builder.nodes[start + 1].code
            else {
                panic!("fallback must publish the original constant-one affine")
            };
            assert_eq!((*left, *right), (start, None));
            assert_eq!(
                (coefficients.left, coefficients.right, coefficients.constant),
                (0, 0, 1)
            );
            assert_eq!(
                builder.nodes[start + 1].bounds,
                Interval::new(1, 1).unwrap()
            );
            (start + 1, start + 2)
        };
        let Code::Gather(words) = &builder.nodes[token_node].code else {
            panic!("selected gather must precede flags")
        };
        assert_eq!(words.iter().collect::<Vec<_>>(), original);
        let Code::Gather(flags) = &builder.nodes[token_node + 1].code else {
            panic!("flags gather must precede masking")
        };
        assert_eq!(flags.iter().collect::<Vec<_>>(), [(valid, 0); 3]);
        let masked = token_node + 2;
        assert!(matches!(
            builder.nodes[masked].code,
            Code::Product {
                left,
                right,
                shape: None,
                rescale: false,
                norm: None,
            } if left == token_node && right == token_node + 1
        ));
        assert_eq!(builder.nodes.len(), masked + 1);
        assert_eq!(
            selected,
            [(masked, 0), (masked, 1), (masked, 2), (valid, 0)]
        );
        assert_eq!(builder.validity, validity);
        assert_eq!(builder.admission_stage, "final_acceptance");
    }
}

#[test]
fn every_original_acceptance_node_cap_leaves_selected_and_validity_unpublished() {
    for sticky in [false, true] {
        let (mut reference, input, mut selected) = fixture(sticky);
        let start = reference.nodes.len();
        reference.final_acceptance(input, 3, &mut selected).unwrap();
        let acceptance = &reference.nodes[start..];
        for successful in 0..acceptance.len() {
            let (mut builder, input, mut selected) = fixture(sticky);
            let original = selected.clone();
            let validity = builder.validity;
            let available: usize = acceptance[..successful]
                .iter()
                .map(|node| {
                    let metadata = match &node.code {
                        Code::Gather(words) => words.metadata_bytes(),
                        _ => 0,
                    };
                    metadata + std::mem::size_of::<Node>() + 16 * node.lanes
                })
                .sum();
            builder.allocated = MAX_PROGRAM_BYTES - available;
            assert_eq!(
                builder.final_acceptance(input, 3, &mut selected),
                Err(Error::Bounds)
            );
            assert_eq!(builder.nodes.len(), start + successful);
            assert_eq!(builder.allocated, MAX_PROGRAM_BYTES);
            assert_eq!(builder.admission_stage, "final_acceptance");
            assert_eq!(builder.admission_limit, Some("program_bytes"));
            assert!(builder.requested_bytes.unwrap() > MAX_PROGRAM_BYTES);
            assert_eq!(selected, original);
            assert_eq!(builder.validity, validity);
        }
    }
}

#[test]
fn original_acceptance_retains_gather_binding_lane_and_product_shape_errors() {
    for (coordinate, error) in [((usize::MAX, 0), Error::Binding), ((0, 3), Error::Shape)] {
        let (mut builder, input, _) = fixture(true);
        let start = builder.nodes.len();
        let mut selected = vec![coordinate];
        assert_eq!(
            builder.final_acceptance(input, 1, &mut selected),
            Err(error)
        );
        assert_eq!(builder.nodes.len(), start);
        assert_eq!(selected, [coordinate]);
        assert_eq!(builder.admission_stage, "final_acceptance");
    }
    let (mut builder, input, mut selected) = fixture(true);
    let original = selected.clone();
    let start = builder.nodes.len();
    assert_eq!(
        builder.final_acceptance(input, 2, &mut selected),
        Err(Error::Shape)
    );
    assert_eq!(builder.nodes.len(), start + 2); // Both gathers, but no mask/output publication.
    assert_eq!(selected, original);
}
