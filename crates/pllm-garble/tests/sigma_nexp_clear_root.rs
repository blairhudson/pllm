//! External-crate use of the original local D29 owner, not SDK/privacy evidence.
use pllm_core::edge_contract::{
    EdgeDescriptor, EdgeIdentity, Interpretation, Numeric, Representation, Shape, SourceIdentity,
};
use pllm_garble::masked_wire_reference::typed::{
    approved_implementation, clear_nexp_bootstrap_bytes, clear_nexp_from_reserved,
    clear_nexp_resources, ClearNexpCreatedFor, ClearNexpFrameFor, ClearNexpInput, ClearNexpRoleFor,
    ClearNexpSpec, ClearNexpStepFor, OwnedLocalDealer,
};
use pllm_garble::shared_allocation::{Error, OwnedBudget, OwnedBudgetContext};
use zeroize::Zeroizing;

type C = OwnedBudgetContext;

fn source() -> SourceIdentity {
    SourceIdentity::new([1; 32], [2; 32], approved_implementation())
}
fn edges(source: SourceIdentity, lanes: usize) -> [EdgeDescriptor; 14] {
    let types = [
        (50, 12, Interpretation::Signed),
        (40, 12, Interpretation::Signed),
        (40, 12, Interpretation::Signed),
        (1, 0, Interpretation::Predicate),
        (1, 0, Interpretation::Predicate),
        (16, 12, Interpretation::Unsigned),
        (16, 12, Interpretation::Unsigned),
        (16, 12, Interpretation::Unsigned),
        (8, 0, Interpretation::Unsigned),
        (8, 0, Interpretation::Unsigned),
        (50, 12, Interpretation::Signed),
        (50, 12, Interpretation::Signed),
        (50, 24, Interpretation::Signed),
        (50, 12, Interpretation::Signed),
    ];
    std::array::from_fn(|i| {
        EdgeDescriptor::new(
            EdgeIdentity::new(source, [4; 32], [5; 32], [10 + i as u8; 32], 7),
            Shape::new(&[lanes]).unwrap(),
            Numeric::new(types[i].0, types[i].1, types[i].2).unwrap(),
            if i == 3 || i == 4 {
                Representation::CommonXorMasked
            } else {
                Representation::CommonAdditiveMasked
            },
            None,
        )
        .unwrap()
    })
}
fn input(root: &OwnedBudget, values: &[i64]) -> ClearNexpInput {
    // Harness's original clear source creation: charge first, then factory.
    let capacity = values.len() + 13;
    let owner = root
        .charge_external(capacity * size_of::<i64>())
        .unwrap()
        .try_build(|_| {
            let mut data = Vec::with_capacity(capacity);
            data.extend_from_slice(values);
            Ok(Zeroizing::new(data))
        })
        .unwrap();
    assert!(
        owner.charged_bytes()
            >= size_of::<ClearNexpInput>() + owner.payload().capacity() * size_of::<i64>()
    );
    owner
}
fn issue(root: &OwnedBudget, values: &[i64]) -> ClearNexpCreatedFor<C> {
    let source = source();
    let mut dealer = OwnedLocalDealer::new(source, [4; 32], [92; 32], root).unwrap();
    let original = input(root, values);
    let mut working = root.charge_external(clear_nexp_bootstrap_bytes()).unwrap();
    let spec = ClearNexpSpec::checked(
        &dealer,
        edges(source, values.len()),
        [90; 32],
        [91; 32],
        &working,
    )
    .unwrap();
    let quote = clear_nexp_resources(&dealer, &spec, &original, &working).unwrap();
    assert_eq!(quote.incoming_owner_bytes(), original.charged_bytes());
    assert_eq!(quote.peer_bytes(), 1688 + 208 * values.len());
    assert_eq!(quote.work().ordered_use_checks, 3 * 18);
    working
        .resize_payload(quote.total_reserved_bytes())
        .unwrap();
    clear_nexp_from_reserved(&mut dealer, spec, original, working).unwrap()
}
fn through_frames(roles: &mut [ClearNexpRoleFor<C>; 2]) -> Vec<[ClearNexpFrameFor<C>; 2]> {
    let mut frames = vec![[
        roles[0].begin(&mut || false).unwrap(),
        roles[1].begin(&mut || false).unwrap(),
    ]];
    for _ in 0..10 {
        let previous = frames.last().unwrap();
        let a = roles[0]
            .advance(previous[1].bytes().unwrap(), &mut || false)
            .unwrap();
        let b = roles[1]
            .advance(previous[0].bytes().unwrap(), &mut || false)
            .unwrap();
        let (ClearNexpStepFor::Frame(a), ClearNexpStepFor::Frame(b)) = (a, b) else {
            panic!("early Ready")
        };
        frames.push([a, b]);
    }
    frames
}

#[test]
fn external_api_runs_exact_eleven_exchanges_at_approved_widths_with_opaque_outputs() {
    for lanes in [1, 3, 128, 256] {
        let root = OwnedBudget::new(64 << 20).unwrap();
        let baseline = root.budget().used_bytes();
        let values: Vec<_> = (0..lanes)
            .map(|i| [0, 1279, 1280, 31487, 31488, (1i64 << 38) - 1][i % 6])
            .collect();
        let (mask, coordinator, mut roles) = issue(&root, &values);
        assert_eq!([roles[0].role(), roles[1].role()], [0, 1]);
        let frames = through_frames(&mut roles);
        assert_eq!(frames.len(), 11);
        assert_eq!(
            frames
                .iter()
                .flatten()
                .map(|f| f.bytes().unwrap().len())
                .sum::<usize>(),
            1688 + 208 * lanes
        );
        for (role, party) in roles.iter_mut().enumerate() {
            assert!(matches!(
                party
                    .advance(frames[10][1 - role].bytes().unwrap(), &mut || false)
                    .unwrap(),
                ClearNexpStepFor::Ready
            ));
        }
        let [left, right] = roles;
        let outputs = [left.finish().unwrap(), right.finish().unwrap()];
        assert_eq!([outputs[0].role(), outputs[1].role()], [0, 1]);
        for output in &outputs {
            assert_eq!(output.descriptor().unwrap(), mask.descriptor().unwrap());
            assert_eq!(output.descriptor().unwrap().numeric().fractional_bits(), 12);
        }
        drop((mask, coordinator, outputs, frames));
        assert_eq!(root.budget().used_bytes(), baseline);
    }
}

#[test]
fn exact_reserved_payload_succeeds_but_one_short_extra_or_foreign_guard_refuses() {
    for delta in [-1isize, 0, 1] {
        let root = OwnedBudget::new(64 << 20).unwrap();
        let source = source();
        let mut dealer = OwnedLocalDealer::new(source, [4; 32], [92; 32], &root).unwrap();
        let baseline = root.budget().used_bytes();
        let original = input(&root, &[0]);
        let mut working = root.charge_external(clear_nexp_bootstrap_bytes()).unwrap();
        let spec = ClearNexpSpec::checked(&dealer, edges(source, 1), [90; 32], [91; 32], &working)
            .unwrap();
        let q = clear_nexp_resources(&dealer, &spec, &original, &working).unwrap();
        working
            .resize_payload(q.total_reserved_bytes().checked_add_signed(delta).unwrap())
            .unwrap();
        let outcome = clear_nexp_from_reserved(&mut dealer, spec, original, working);
        match (delta, outcome) {
            (0, Ok(created)) => drop(created),
            (-1 | 1, Err(error)) => assert_eq!(error.native(), Some(Error::Resource)),
            _ => panic!("exact reservation contract changed"),
        }
        assert_eq!(root.budget().used_bytes(), baseline);

        let original = input(&root, &[0]);
        let foreign = OwnedBudget::new(64 << 20).unwrap();
        let wrong = foreign.charge_external(q.total_reserved_bytes()).unwrap();
        assert!(
            matches!(clear_nexp_from_reserved(&mut dealer,spec,original,wrong),Err(e) if e.native()==Some(Error::Resource))
        );
        assert_eq!(root.budget().used_bytes(), baseline);
    }
}

#[test]
fn no_bootstrap_authority_and_invalid_original_roots_are_closed() {
    let root = OwnedBudget::new(64 << 20).unwrap();
    let source = source();
    let mut dealer = OwnedLocalDealer::new(source, [4; 32], [92; 32], &root).unwrap();
    let zero = root.charge_external(0).unwrap();
    assert!(matches!(
        ClearNexpSpec::checked(&dealer, edges(source, 1), [90; 32], [91; 32], &zero),
        Err(Error::Resource)
    ));
    drop(zero);
    for values in [&[-1][..], &[1i64 << 38][..], &[0, -1][..]] {
        let original = input(&root, values);
        let mut working = root.charge_external(clear_nexp_bootstrap_bytes()).unwrap();
        let spec = ClearNexpSpec::checked(
            &dealer,
            edges(source, values.len()),
            [90; 32],
            [91; 32],
            &working,
        )
        .unwrap();
        let q = clear_nexp_resources(&dealer, &spec, &original, &working).unwrap();
        working.resize_payload(q.total_reserved_bytes()).unwrap();
        assert!(
            matches!(clear_nexp_from_reserved(&mut dealer,spec,original,working),Err(e) if e.native()==Some(Error::Domain))
        );
    }
}

#[test]
fn malformed_peer_first_failure_and_callback_cancellation_jointly_burn_without_retry() {
    for callback in [false, true] {
        let root = OwnedBudget::new(64 << 20).unwrap();
        let (mask, mut coordinator, mut roles) = issue(&root, &[0, 65535]);
        let frames = [
            roles[0].begin(&mut || false).unwrap(),
            roles[1].begin(&mut || false).unwrap(),
        ];
        if callback {
            assert!(roles[0]
                .advance(frames[1].bytes().unwrap(), &mut || {
                    coordinator.cancel();
                    false
                })
                .is_err());
        } else {
            assert!(roles[0].advance(&[], &mut || false).is_err());
        }
        assert!(roles[0].begin(&mut || false).is_err());
        assert!(roles[1].advance(&[], &mut || false).is_err());
        assert!(mask.descriptor().is_err());
        assert!(frames.iter().all(|f| f.bytes().is_err()));
    }
}

#[test]
fn one_ready_finish_burns_the_provisional_output_and_retained_frames() {
    for first in 0..2 {
        let root = OwnedBudget::new(64 << 20).unwrap();
        let (mask, _coordinator, mut roles) = issue(&root, &[0]);
        let frames = through_frames(&mut roles);
        assert!(matches!(
            roles[first]
                .advance(frames[10][1 - first].bytes().unwrap(), &mut || false)
                .unwrap(),
            ClearNexpStepFor::Ready
        ));
        let [left, right] = roles;
        let (premature, other) = if first == 0 {
            (left, right)
        } else {
            (right, left)
        };
        assert!(premature.finish().is_err());
        assert!(other.finish().is_err());
        assert!(mask.descriptor().is_err());
        assert!(frames.iter().flatten().all(|f| f.bytes().is_err()));
    }
}

#[test]
fn last_frame_retains_original_counter_after_root_dealer_roles_and_coordinator_drop() {
    let root = OwnedBudget::new(64 << 20).unwrap();
    let watch = root.try_clone().unwrap();
    let (mask, coordinator, mut roles) = issue(&root, &[0, 65535]);
    let frame = roles[0].begin(&mut || false).unwrap();
    drop((root, mask, coordinator, roles));
    assert!(frame.bytes().is_err());
    let base = OwnedBudget::resources().unwrap().initial_reserved_bytes;
    assert!(watch.budget().used_bytes() > base);
    drop(frame);
    assert_eq!(watch.budget().used_bytes(), base);
}
