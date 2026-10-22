//! Bounded cache regressions. Wire this file as a child of `lookup` so private
//! ownership witnesses stay test-only; no production observability is needed.
use super::super::wire::{ImportGuard, MaterialPacket};
use super::*;

fn context() -> Context {
    Context {
        plan: [1; 32],
        session: [2; 32],
        operation: [3; 32],
        first_tensor_index: 9,
    }
}

fn evaluate<'a>(parties: &mut [Party<'a>; 2], inputs: &[u64]) -> [Output<'a>; 2] {
    let mask = (1u64 << 50) - 1;
    let left: Vec<_> = (0..inputs.len()).map(|i| mask - i as u64).collect();
    let right: Vec<_> = inputs
        .iter()
        .zip(&left)
        .map(|(x, share)| x.wrapping_sub(*share) & mask)
        .collect();
    let a = parties[0].start(&left).unwrap();
    let b = parties[1].start(&right).unwrap();
    [
        parties[0].finish(b.as_bytes()).unwrap(),
        parties[1].finish(a.as_bytes()).unwrap(),
    ]
}

#[test]
fn ring50_cached_and_legacy_identical_material_match_each_share_and_array_oracle() {
    let dealer = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let legacy_budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let cached_budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mask = (1u64 << 50) - 1;
    // Alternating high/low residues exercise modular coefficient differences,
    // unlike a constant table. The third chunk has one nonzero entry.
    let values: Vec<_> = (0..513u64)
        .map(|i| if i % 2 == 0 { mask - i * 19 } else { i * 31 })
        .collect();
    let source = Table::new_zero_extended(50, &values, [7; 32], &dealer).unwrap();
    let legacy = Table::new_zero_extended(50, &values, [7; 32], &legacy_budget).unwrap();
    let cached = Table::new_zero_extended(50, &values, [7; 32], &cached_budget)
        .unwrap()
        .with_shared_chunks()
        .unwrap();
    assert_eq!(legacy.digest(), cached.digest());
    let mut legacy_guard = ImportGuard::new(4, &legacy_budget).unwrap();
    let mut cached_guard = ImportGuard::new(4, &cached_budget).unwrap();
    for inputs in [[0, 255, 256, 511], [512, 513, mask, 1]] {
        let [a, b] = source.issue(4, context()).unwrap();
        let pa = a.export_material().unwrap();
        let pb = b.export_material().unwrap();
        let nonce = pa.nonce();
        // Deliberate test-only differential fork into isolated replay ledgers:
        // identical private bytes remove random output sharing as a confounder.
        // This is not permission to replay material in a real execution.
        let ca = MaterialPacket::from_bytes(pa.as_bytes(), &dealer).unwrap();
        let cb = MaterialPacket::from_bytes(pb.as_bytes(), &dealer).unwrap();
        let mut old = [
            Party::import_material(&legacy, 4, context(), 0, nonce, pa, &mut legacy_guard).unwrap(),
            Party::import_material(&legacy, 4, context(), 1, nonce, pb, &mut legacy_guard).unwrap(),
        ];
        let mut new = [
            Party::import_material(&cached, 4, context(), 0, nonce, ca, &mut cached_guard).unwrap(),
            Party::import_material(&cached, 4, context(), 1, nonce, cb, &mut cached_guard).unwrap(),
        ];
        let expected = evaluate(&mut old, &inputs);
        let actual = evaluate(&mut new, &inputs);
        for role in 0..2 {
            assert_eq!(actual[role].shares(), expected[role].shares());
        }
        for (lane, input) in inputs.into_iter().enumerate() {
            assert_eq!(
                (actual[0].shares()[lane] + actual[1].shares()[lane]) & mask,
                values.get(input as usize).copied().unwrap_or(0)
            );
        }
        assert!(old.iter().chain(&new).all(Party::is_consumed));
    }
    drop((source, legacy, cached, legacy_guard, cached_guard));
    assert_eq!(dealer.used_bytes(), 0);
    assert_eq!(legacy_budget.used_bytes(), 0);
    assert_eq!(cached_budget.used_bytes(), 0);
}

#[test]
fn ring50_repeated_cached_calls_refresh_masks_offsets_and_private_material() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let table = Table::new_zero_extended(50, &[19; 257], [7; 32], &budget)
        .unwrap()
        .with_shared_chunks()
        .unwrap();
    let before = budget.used_bytes();
    let mut first = table.issue(2, context()).unwrap();
    let mut second = table.issue(2, context()).unwrap();
    assert_eq!(first[0].binding, second[0].binding);
    assert_ne!(first[0].nonce, second[0].nonce);
    let masks = |parties: &[Party<'_>; 2], role: usize| {
        let Some(State::Ready(units)) = &parties[role].state else {
            panic!("ready")
        };
        Zeroizing::new(units.iter().map(|unit| *unit.mask).collect::<Vec<_>>())
    };
    let mut offsets = Vec::new();
    for parties in [&first, &second] {
        let left = masks(parties, 0);
        let right = masks(parties, 1);
        offsets.push(Zeroizing::new(
            left.iter()
                .zip(right.iter())
                .map(|(a, b)| a.wrapping_add(*b) & ((1u64 << 50) - 1))
                .collect::<Vec<_>>(),
        ));
    }
    // Compare whole vectors, not individual random words: accidental collision
    // probability stays negligible without a deterministic RNG seam.
    assert_ne!(offsets[0].as_slice(), offsets[1].as_slice());
    for role in 0..2 {
        assert_ne!(
            masks(&first, role).as_slice(),
            masks(&second, role).as_slice()
        );
        let material = |party: &Party<'_>| {
            let Some(State::Ready(units)) = &party.state else {
                panic!("ready")
            };
            let mut bytes = Zeroizing::new(Vec::new());
            for unit in units {
                assert!(matches!(&unit.lookup, Lookup::Shared(_)));
                // Exclude issuance nonce and mask shares: key/base freshness
                // must not pass merely because an envelope or mask changed.
                unit.lookup.write_material(&mut bytes);
            }
            bytes
        };
        assert_ne!(
            material(&first[role]).as_slice(),
            material(&second[role]).as_slice()
        );
    }
    for parties in [&mut first, &mut second] {
        let outputs = evaluate(parties, &[256, 257]);
        for (lane, expected) in [19, 0].into_iter().enumerate() {
            assert_eq!(
                (outputs[0].shares()[lane] + outputs[1].shares()[lane]) & ((1u64 << 50) - 1),
                expected
            );
        }
    }
    drop((first, second));
    assert_eq!(budget.used_bytes(), before);
    drop(table);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn cached_backing_retires_on_last_terminal_state_not_frame_or_output_drop() {
    for terminal in ["finish", "cancel", "bad_start", "bad_finish"] {
        let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
        let table = Table::new_zero_extended(50, &[19; 257], [7; 32], &budget)
            .unwrap()
            .with_shared_chunks()
            .unwrap();
        let lowered = Arc::downgrade(table.lowered.as_ref().unwrap());
        let chunks: Vec<_> = table
            .lowered
            .as_ref()
            .unwrap()
            .chunks
            .iter()
            .map(Arc::downgrade)
            .collect();
        let [mut a, mut b] = table.issue(1, context()).unwrap();
        drop(table);
        let fa = a.start(&[0]).unwrap();
        let fb = b.start(&[256]).unwrap();
        let out = a.finish(fb.as_bytes()).unwrap();
        assert!(lowered.upgrade().is_some(), "peer still owns backing");
        let peer_output = match terminal {
            "finish" => Some(b.finish(fa.as_bytes()).unwrap()),
            "cancel" => {
                b.cancel();
                None
            }
            "bad_start" => {
                // Starting an already-opened owner burns all retained units.
                assert!(matches!(b.start(&[0]), Err(Error::Context)));
                None
            }
            "bad_finish" => {
                assert!(matches!(b.finish(&[]), Err(Error::Context)));
                None
            }
            _ => unreachable!(),
        };
        assert!(a.is_consumed() && b.is_consumed());
        assert!(lowered.upgrade().is_none(), "terminal case {terminal}");
        assert!(chunks.iter().all(|chunk| chunk.upgrade().is_none()));
        assert!(matches!(b.finish(fa.as_bytes()), Err(Error::Consumed)));
        // Exported online artifacts carry only their own reservations, never
        // canonical public backing. Keep them live through the weak checks.
        assert!(!fa.as_bytes().is_empty() && !fb.as_bytes().is_empty());
        assert_eq!(out.shares().len(), 1);
        drop((a, b, fa, fb, out, peer_output));
        assert_eq!(budget.used_bytes(), 0);
    }
}
