use super::*;

fn context() -> Context {
    Context {
        plan: [11; 32],
        session: [22; 32],
        operation: [33; 32],
        first_tensor_index: 19,
    }
}

fn parameters(field: u32, domain: u32, lanes: usize, encoding: Encoding) -> Parameters {
    Parameters {
        field,
        scalar_domain: domain,
        lanes,
        encoding,
    }
}

#[test]
fn punctured_tree_preserves_every_other_leaf_and_never_retains_the_missing_key() {
    for domain in [2, 4, 17, 128, 257] {
        let p = parameters(
            if domain <= 17 { 17 } else { 257 },
            domain,
            7,
            Encoding::Punctured,
        );
        let root = [91; 16];
        let full = sender(&root, p).leaves(domain as usize);
        let expanded = sender(
            &root,
            Parameters {
                encoding: Encoding::Expanded,
                ..p
            },
        )
        .leaves(domain as usize);
        assert_eq!(full, expanded);
        for point in 0..domain {
            let nodes = puncture(&root, point as usize, domain.next_power_of_two() as usize);
            assert_eq!(nodes.len(), domain.next_power_of_two().ilog2() as usize);
            assert!(nodes
                .iter()
                .all(|n| point < n.first as u32 || point >= (n.first + n.size) as u32));
            let punctured = receiver(&root, point, p).leaves(domain as usize);
            let expanded = receiver(
                &root,
                point,
                Parameters {
                    encoding: Encoding::Expanded,
                    ..p
                },
            )
            .leaves(domain as usize);
            assert_eq!(punctured, expanded);
            for j in 0..domain as usize {
                if j == point as usize {
                    assert!(punctured[j].is_none());
                } else {
                    assert_eq!(punctured[j], full[j]);
                }
            }
        }
    }
}

#[test]
fn full_field_scalar_vector_consumer_matches_independent_modular_products() {
    for field in [2, 17, 257] {
        for encoding in [Encoding::Expanded, Encoding::Punctured] {
            let p = parameters(field, field, 37, encoding);
            let values: Vec<_> = (0..p.lanes).map(|i| (i as u32 * 19) % field).collect();
            let scalars: Vec<_> = if field < 20 {
                (0..field).collect()
            } else {
                vec![0, 1, 19, 128, 256]
            };
            for scalar in scalars {
                let result = run_reference(p, Some((scalar, &values)), context()).unwrap();
                let output = result.outputs.unwrap();
                assert_eq!(
                    &*output,
                    &values
                        .iter()
                        .map(|v| ((u64::from(*v) * u64::from(scalar)) % u64::from(field)) as u32)
                        .collect::<Vec<_>>()
                );
                assert_eq!(result.checked_correlations, p.lanes);
                assert_eq!(
                    result.peer_bytes_per_party,
                    HEADER_BYTES + 4 * (p.lanes + 1)
                );
                assert_eq!(
                    result.prf_field_evaluations_per_party,
                    (2 * field as usize - 1) * p.lanes
                );
            }
        }
    }
}

#[test]
fn restricted_domains_generate_correlations_but_reject_and_burn_generic_openings() {
    for (field, domain) in [(17, 2), (17, 12), (257, 16), (257, 256)] {
        let p = parameters(field, domain, 9, Encoding::Punctured);
        let result = run_reference(p, None, context()).unwrap();
        assert_eq!(result.checked_correlations, p.lanes);
        assert!(result.outputs.is_none());
        assert_eq!(result.peer_bytes_per_party, 0);
        let [mut left, mut right] = issue_reference(p, context()).unwrap();
        left.prepare().unwrap();
        right.prepare().unwrap();
        assert_eq!(left.start(0, &[0; 9]).err(), Some(Error::Domain));
        assert_eq!(left.start(0, &[0; 9]).err(), Some(Error::Consumed));
        right.cancel();
        assert_eq!(right.start(0, &[0; 9]).err(), Some(Error::Consumed));
        assert_eq!(
            run_reference(p, Some((0, &[0; 9])), context()).err(),
            Some(Error::Domain)
        );
    }
}

#[test]
fn support_counting_is_an_exact_restricted_scalar_leakage_witness() {
    for field in [2, 17, 257] {
        for domain in 2..=field {
            let p = parameters(field, domain, 1, Encoding::Punctured);
            let (shift, numerator, denominator) = p.opening_distance().unwrap();
            let a: Vec<_> = (0..field)
                .map(|opening| {
                    (0..domain)
                        .filter(|mask| (field - mask) % field == opening)
                        .count()
                })
                .collect();
            let b: Vec<_> = (0..field)
                .map(|opening| {
                    (0..domain)
                        .filter(|mask| (shift + field - mask) % field == opening)
                        .count()
                })
                .collect();
            assert_eq!(
                a.iter().zip(&b).map(|(a, b)| a.abs_diff(*b)).sum::<usize>() as u32,
                2 * numerator
            );
            assert_eq!(denominator, domain);
            assert_eq!(numerator == 0, field == domain);
        }
    }
}

#[test]
fn malformed_replayed_cross_issuance_and_cancelled_frames_burn_party_state() {
    let p = parameters(17, 17, 3, Encoding::Punctured);
    for changed_byte in [0, 8, 40, 72, 73, 74, 78, 82, 93] {
        let [mut left, mut right] = issue_reference(p, context()).unwrap();
        left.prepare().unwrap();
        right.prepare().unwrap();
        let first = left.start(2, &[1, 2, 3]).unwrap();
        let second = right.start(4, &[3, 2, 1]).unwrap();
        let mut malformed = second.clone();
        malformed[changed_byte] = 255;
        assert!(left.finish(&malformed).is_err());
        assert_eq!(left.finish(&second).err(), Some(Error::Consumed));
        right.finish(&first).unwrap();
        assert_eq!(right.finish(&first).err(), Some(Error::Consumed));
    }
    let [mut first, mut other] = issue_reference(p, context()).unwrap();
    let [mut second, mut peer] = issue_reference(p, context()).unwrap();
    first.prepare().unwrap();
    second.prepare().unwrap();
    peer.prepare().unwrap();
    first.start(0, &[0; 3]).unwrap();
    second.start(0, &[0; 3]).unwrap();
    let wrong = peer.start(0, &[0; 3]).unwrap();
    assert_eq!(first.finish(&wrong).err(), Some(Error::Context));
    assert_eq!(first.finish(&wrong).err(), Some(Error::Consumed));
    second.finish(&wrong).unwrap();
    other.cancel();
    assert_eq!(other.prepare().err(), Some(Error::Consumed));
    let [mut first, _] = issue_reference(p, context()).unwrap();
    first.prepare().unwrap();
    first.start(0, &[0; 3]).unwrap();
    assert_eq!(first.start(0, &[0; 3]).err(), Some(Error::Context));
    assert_eq!(first.finish(&wrong).err(), Some(Error::Consumed));
}

#[test]
fn domain_shape_phase_and_allocation_checks_precede_execution() {
    let p = parameters(257, 257, MAX_LANES, Encoding::Punctured);
    assert!(p.allocation_estimate().unwrap() < MAX_ALLOCATION_BYTES);
    for bad in [
        Parameters { field: 256, ..p },
        Parameters {
            scalar_domain: 258,
            ..p
        },
        Parameters {
            scalar_domain: 1,
            ..p
        },
        Parameters { lanes: 0, ..p },
        Parameters {
            lanes: MAX_LANES + 1,
            ..p
        },
    ] {
        assert!(issue_reference(bad, context()).is_err());
    }
    let mut c = context();
    c.first_tensor_index = u64::MAX;
    assert!(issue_reference(p, c).is_err());
    let p = Parameters { lanes: 1, ..p };
    for (scalar, values) in [(257, vec![0]), (0, vec![257]), (0, vec![])] {
        let [mut left, _] = issue_reference(p, context()).unwrap();
        left.prepare().unwrap();
        assert_eq!(left.start(scalar, &values).err(), Some(Error::Domain));
        assert_eq!(left.prepare().err(), Some(Error::Consumed));
    }
    let [mut left, _] = issue_reference(p, context()).unwrap();
    assert_eq!(left.start(0, &[0]).err(), Some(Error::Context));
    assert_eq!(left.prepare().err(), Some(Error::Consumed));
}
