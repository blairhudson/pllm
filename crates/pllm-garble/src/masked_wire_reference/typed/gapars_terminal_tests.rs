use super::super::role_coordination::{self, CoordinationFor, Retire, RoleSlotFor, SplitControl};
use super::*;
use pllm_core::edge_contract::{EdgeIdentity, Shape};
use std::sync::{Arc, Mutex};

fn source() -> SourceIdentity {
    SourceIdentity::new([1; 32], [2; 32], approved_implementation())
}
fn edge(wire: u8, fraction: u8) -> EdgeDescriptor {
    EdgeDescriptor::new(
        EdgeIdentity::new(source(), [3; 32], [4; 32], [wire; 32], 0),
        Shape::new(&[4]).unwrap(),
        Numeric::new(50, fraction, Interpretation::Signed).unwrap(),
        Representation::CommonAdditiveMasked,
        None,
    )
    .unwrap()
}
type RootsFor<C> = (
    DealerMaskFor<C, Ring>,
    [CommonMaskedFor<C, Ring>; 2],
    DealerMaskFor<C, Ring>,
    [CommonMaskedFor<C, Ring>; 2],
);
fn roots<C: AllocationContext>(
    dealer: &mut LocalDealerFor<C>,
    x: &[i64; 4],
    beta: &[i64; 4],
) -> RootsFor<C> {
    let a = dealer
        .prepare_root(
            edge(6, 24),
            BoundRange::new(-(1 << 27), 1 << 27).unwrap(),
            [9; 32],
        )
        .unwrap()
        .encode(x)
        .unwrap();
    let b = dealer
        .prepare_root(edge(7, 12), BoundRange::new(-4096, 4096).unwrap(), [9; 32])
        .unwrap()
        .encode(beta)
        .unwrap();
    (a.0, a.1, b.0, b.1)
}
fn fill(bytes: &mut [u8]) -> Result<(), Error> {
    bytes.fill(0x53);
    Ok(())
}

#[test]
fn borrowed_negative_floor_oracle_and_exact_terminal_counts() {
    let budget = Budget::new(4 << 20).unwrap();
    let mut dealer = LocalDealer::new(source(), [3; 32], [5; 32], &budget).unwrap();
    let x = [-8193, -4097, -1, 8193];
    let beta = [-4096, -7, 0, 4095];
    let (mask, [x0, x1], bmask, [b0, b1]) = roots(&mut dealer, &x, &beta);
    let q = resources::<&Budget>().unwrap();
    assert_eq!(
        (
            q.terminal_frame_bytes,
            q.extra_random_bytes,
            q.material_payload_bytes
        ),
        (108, 32, 64)
    );
    assert_eq!(q.terminal_frames_per_party, 1);
    let mut drawn = 0;
    let [mut a, mut b] = issue_reserved_with(
        &mut dealer,
        [9; 32],
        mask,
        bmask,
        edge(8, 12),
        [10; 32],
        budget.reserve(q.reservation_bytes).unwrap(),
        &mut |v| {
            drawn += v.len();
            fill(v)
        },
    )
    .unwrap();
    assert_eq!(drawn, q.native.random_bytes + 32);
    let fa = a.begin(x0).unwrap();
    let fb = b.begin(x1).unwrap();
    assert_eq!((fa.as_bytes().len(), fb.as_bytes().len()), (108, 108));
    let a = a.finish(b0, fb.as_bytes()).unwrap();
    let b = b.finish(b1, fa.as_bytes()).unwrap();
    for i in 0..4 {
        assert_eq!(
            (a.local.test_values()[i].wrapping_add(b.local.test_values()[i]))
                & super::super::super::mask(50),
            (x[i].div_euclid(4096) + beta[i]) as u64 & super::super::super::mask(50)
        );
    }
}

#[test]
fn ordinary_two_frame_path_still_matches_terminal_integer_oracle() {
    let budget = Budget::new(4 << 20).unwrap();
    let mut dealer = LocalDealer::new(source(), [3; 32], [5; 32], &budget).unwrap();
    let x = [-8193, -4096, -1, 4097];
    let (mask, [x0, x1], beta, values) = roots(&mut dealer, &x, &[0; 4]);
    drop(beta);
    drop(values);
    let (output, [mut a, mut b]) = dealer
        .gap_ars([9; 32], mask, 12, edge(8, 12), [10; 32])
        .unwrap();
    let fa = a.begin(x0).unwrap();
    let fb = b.begin(x1).unwrap();
    let StepFor::Frame(ga) = a.advance(fb.as_bytes()).unwrap() else {
        panic!()
    };
    let StepFor::Frame(gb) = b.advance(fa.as_bytes()).unwrap() else {
        panic!()
    };
    assert_eq!((ga.as_bytes().len(), gb.as_bytes().len()), (108, 108));
    let StepFor::Output(a) = a.advance(gb.as_bytes()).unwrap() else {
        panic!()
    };
    let StepFor::Output(b) = b.advance(ga.as_bytes()).unwrap() else {
        panic!()
    };
    assert_eq!(a.backing.values(), b.backing.values());
    for (i, value) in x.iter().enumerate() {
        assert_eq!(
            a.backing.values()[i].wrapping_sub(output.backing.values()[i])
                & super::super::super::mask(50),
            value.div_euclid(4096) as u64 & super::super::super::mask(50)
        );
    }
}

#[test]
fn exact_and_one_short_admission_before_entropy() {
    for short in [true, false] {
        let budget = Budget::new(4 << 20).unwrap();
        let mut dealer = LocalDealer::new(source(), [3; 32], [5; 32], &budget).unwrap();
        let (mask, values, beta, bvalues) = roots(&mut dealer, &[0; 4], &[0; 4]);
        let q = resources::<&Budget>().unwrap();
        let mut draws = 0;
        let result = issue_reserved_with(
            &mut dealer,
            [9; 32],
            mask,
            beta,
            edge(8, 12),
            [10; 32],
            budget
                .reserve(q.reservation_bytes - usize::from(short))
                .unwrap(),
            &mut |v| {
                draws += v.len();
                fill(v)
            },
        );
        assert_eq!(result.is_err(), short);
        if short {
            assert_eq!(draws, 0);
        }
        drop(result);
        drop(values);
        drop(bvalues);
    }
}

#[test]
fn foreign_counter_rejected_before_rng_and_random_failure_retires() {
    let budget = Budget::new(4 << 20).unwrap();
    let foreign = Budget::new(4 << 20).unwrap();
    let mut dealer = LocalDealer::new(source(), [3; 32], [5; 32], &budget).unwrap();
    let (mask, values, beta, bvalues) = roots(&mut dealer, &[0; 4], &[0; 4]);
    let q = resources::<&Budget>().unwrap();
    let mut calls = 0;
    assert!(
        issue_reserved_with(
            &mut dealer,
            [9; 32],
            mask,
            beta,
            edge(8, 12),
            [10; 32],
            foreign.reserve(q.reservation_bytes).unwrap(),
            &mut |_| {
                calls += 1;
                Err(Error::Randomness)
            }
        )
        .is_err()
    );
    assert_eq!(calls, 0);
    drop(values);
    drop(bvalues);
    let (mask, values, beta, bvalues) = roots(&mut dealer, &[0; 4], &[0; 4]);
    assert!(
        issue_reserved_with(
            &mut dealer,
            [9; 32],
            mask,
            beta,
            edge(8, 12),
            [10; 32],
            budget.reserve(q.reservation_bytes).unwrap(),
            &mut |_| Err(Error::Randomness)
        )
        .is_err()
    );
    drop(values);
    drop(bvalues);
}

#[test]
fn wrong_beta_role_phase_and_replayed_peer_refuse() {
    for mode in 0..4 {
        let budget = Budget::new(4 << 20).unwrap();
        let mut dealer = LocalDealer::new(source(), [3; 32], [5; 32], &budget).unwrap();
        let (mask, [x0, x1], beta, [b0, b1]) = roots(&mut dealer, &[0; 4], &[1; 4]);
        let q = resources::<&Budget>().unwrap();
        let [mut a, mut b] = issue_reserved_with(
            &mut dealer,
            [9; 32],
            mask,
            beta,
            edge(8, 12),
            [10; 32],
            budget.reserve(q.reservation_bytes).unwrap(),
            &mut fill,
        )
        .unwrap();
        if mode == 0 {
            assert!(a.finish(b0, &[]).is_err());
            continue;
        }
        let fa = a.begin(x0).unwrap();
        let fb = b.begin(x1).unwrap();
        match mode {
            1 => assert!(a.finish(b1, fb.as_bytes()).is_err()),
            2 => assert!(a.finish(b0, fa.as_bytes()).is_err()),
            _ => {
                let (_, _, _, [foreign, _]) = roots(&mut dealer, &[0; 4], &[1; 4]);
                assert!(a.finish(foreign, fb.as_bytes()).is_err());
            }
        }
    }
}

#[test]
fn each_terminal_split_rng_failure_erases_native_and_partial_material() {
    for lane in 0..4 {
        let budget = Budget::new(4 << 20).unwrap();
        let mut dealer = LocalDealer::new(source(), [3; 32], [5; 32], &budget).unwrap();
        let (mask, values, beta, bvalues) = roots(&mut dealer, &[0; 4], &[1; 4]);
        let q = resources::<&Budget>().unwrap();
        let mut drawn = 0;
        let result = issue_reserved_with(
            &mut dealer,
            [9; 32],
            mask,
            beta,
            edge(8, 12),
            [10; 32],
            budget.reserve(q.reservation_bytes).unwrap(),
            &mut |v| {
                if drawn == q.native.random_bytes + 8 * lane {
                    return Err(Error::Randomness);
                }
                drawn += v.len();
                fill(v)
            },
        );
        assert!(result.is_err());
        assert_eq!(drawn, q.native.random_bytes + 8 * lane);
        drop(values);
        drop(bvalues);
        drop(dealer);
        assert_eq!(budget.used_bytes(), 0);
    }
}

// Test-only composition using the EXISTING coordinator. This is not a fake
// production normalization/source parent or a new authority constructor.
struct State<C: AllocationContext> {
    party: Option<PartyFor<C>>,
    beta: Option<CommonMaskedFor<C, Ring>>,
    output: Option<ShareFor<C>>,
}
impl<C: AllocationContext> Retire for State<C> {
    fn retire(&mut self) {
        self.output = None;
        self.party = None;
        self.beta = None;
    }
}
struct HeldFrame<C: AllocationContext> {
    _frame: arithmetic::FrameFor<C>,
    _owner: Arc<CoordinationFor<C, State<C>>>,
}
struct HeldOutput<C: AllocationContext> {
    _role: u8,
    _owner: Arc<CoordinationFor<C, State<C>>>,
}

// Consuming TEST role custody, not a production parent/source issuer.
struct ConsumingRole<C: AllocationContext> {
    owner: Option<Arc<CoordinationFor<C, State<C>>>>,
    role: u8,
}
impl<C: AllocationContext> Drop for ConsumingRole<C> {
    fn drop(&mut self) {
        if let Some(owner) = &self.owner {
            owner.abort();
        }
    }
}
impl<C: AllocationContext> ConsumingRole<C> {
    fn handoff(mut self) -> Result<HeldOutput<C>, Error> {
        let owner = Arc::clone(self.owner.as_ref().ok_or(Error::Consumed)?);
        owner.operation(|control| {
            control.require_both_ready()?;
            let slot = owner.slots[usize::from(self.role)].state.lock().unwrap();
            if slot.output.is_none() {
                return Err(Error::Consumed);
            }
            // Readiness AND output creation are under the SAME abort guard.
            Ok(HeldOutput {
                _role: self.role,
                _owner: self.owner.take().ok_or(Error::Consumed)?,
            })
        })
    }
}
impl<C: AllocationContext> HeldOutput<C> {
    fn check_live(&self) -> Result<(), Error> {
        self._owner.operation(|control| {
            control.require_both_ready()?;
            let slot = self._owner.slots[usize::from(self._role)]
                .state
                .lock()
                .unwrap();
            if slot.output.is_some() {
                Ok(())
            } else {
                Err(Error::Consumed)
            }
        })
    }
}
struct OutputTransition<C: AllocationContext> {
    _owner: Arc<CoordinationFor<C, State<C>>>,
    _role: ConsumingRole<C>,
    _result: Result<HeldOutput<C>, Error>,
}

#[allow(
    clippy::arc_with_non_send_sync,
    reason = "Same-thread custody test uses the original Arc coordinator; genuine typed native backing is not Send."
)]
fn coordinated_owned(mode: u8) {
    let root = OwnedBudget::new(4 << 20).unwrap();
    let mut dealer = OwnedLocalDealer::new(source(), [3; 32], [5; 32], &root).unwrap();
    let (mask, [x0, x1], beta, [b0, b1]) = roots(&mut dealer, &[-1, 0, 1, 4096], &[1; 4]);
    let q = resources::<OwnedBudgetContext>().unwrap();
    // Concrete TEST parent controls are reserved separately BEFORE activation.
    let parent = role_coordination::arc_bytes::<
        CoordinationFor<OwnedBudgetContext, State<OwnedBudgetContext>>,
    >()
    .unwrap()
        + 2 * role_coordination::arc_bytes::<
            RoleSlotFor<OwnedBudgetContext, State<OwnedBudgetContext>>,
        >()
        .unwrap()
        + 2 * size_of::<HeldFrame<OwnedBudgetContext>>()
        + 2 * size_of::<HeldOutput<OwnedBudgetContext>>()
        + 2 * size_of::<ConsumingRole<OwnedBudgetContext>>()
        + 2 * size_of::<Option<ConsumingRole<OwnedBudgetContext>>>()
        + size_of::<OutputTransition<OwnedBudgetContext>>()
        + size_of::<[State<OwnedBudgetContext>; 2]>()
        + size_of::<SplitControl>();
    let mut charge = root.charge_external(parent + q.reservation_bytes).unwrap();
    let native = charge
        .split_native(root.budget(), q.reservation_bytes)
        .unwrap();
    let extra = charge.split_native(root.budget(), parent).unwrap();
    let [mut a, mut b] = issue_reserved_with(
        &mut dealer,
        [9; 32],
        mask,
        beta,
        edge(8, 12),
        [10; 32],
        native,
        &mut fill,
    )
    .unwrap();
    let fa = a.begin(x0).unwrap();
    let fb = b.begin(x1).unwrap();
    let mut extra = extra;
    let sa = extra.split(parent / 4).unwrap();
    let sb = extra.split(parent / 4).unwrap();
    let coord = Arc::new(CoordinationFor {
        control: Mutex::new(SplitControl::default()),
        slots: [
            Arc::new(RoleSlotFor {
                state: Mutex::new(State {
                    party: Some(a),
                    beta: Some(b0),
                    output: None,
                }),
                _lease: sa,
            }),
            Arc::new(RoleSlotFor {
                state: Mutex::new(State {
                    party: Some(b),
                    beta: Some(b1),
                    output: None,
                }),
                _lease: sb,
            }),
        ],
        _original_run: Some(dealer._lease),
        _additional: extra,
    });
    let held = HeldFrame {
        _frame: fa,
        _owner: Arc::clone(&coord),
    };
    let peer_held = HeldFrame {
        _frame: fb,
        _owner: Arc::clone(&coord),
    };
    let mut roles = [
        Some(ConsumingRole {
            owner: Some(Arc::clone(&coord)),
            role: 0,
        }),
        Some(ConsumingRole {
            owner: Some(Arc::clone(&coord)),
            role: 1,
        }),
    ];
    for (role, peer) in [
        (0, peer_held._frame.as_bytes()),
        (1, held._frame.as_bytes()),
    ] {
        if role == 1 && mode != 0 && mode != 5 {
            match mode {
                1 => {
                    assert!(
                        coord
                            .operation(|_| {
                                let mut s = coord.slots[role].state.lock().unwrap();
                                let p = s.party.take().unwrap();
                                let beta = s.beta.take().unwrap();
                                s.output = Some(p.finish(beta, &[])?);
                                Ok(())
                            })
                            .is_err()
                    );
                }
                2 => {
                    assert!(
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            let _: Result<(), Error> = coord.operation(|_| {
                                let _s = coord.slots[role].state.lock().unwrap();
                                panic!("native terminal test unwind")
                            });
                        }))
                        .is_err()
                    );
                }
                4 => {
                    // One role Ready, sibling still live: Drop alone burns both.
                    drop(roles[0].take());
                    assert!(roles[1].take().unwrap().handoff().is_err());
                }
                _ => {
                    assert!(
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            let _ = coord.poll_cancelled(&mut || {
                                assert!(coord.control.try_lock().is_ok());
                                for slot in &coord.slots {
                                    assert!(slot.state.try_lock().is_ok());
                                }
                                panic!("unlocked cancellation test unwind")
                            });
                        }))
                        .is_err()
                    );
                }
            }
            assert!(coord.control().aborted);
            for slot in &coord.slots {
                let s = slot.state.lock().unwrap_or_else(|p| p.into_inner());
                assert!(s.output.is_none() && s.party.is_none() && s.beta.is_none());
            }
            break;
        }
        coord
            .operation(|control| {
                let mut s = coord.slots[role].state.lock().unwrap();
                let p = s.party.take().unwrap();
                let beta = s.beta.take().unwrap();
                s.output = Some(p.finish(beta, peer)?);
                control.mark_ready(role as u8);
                if role == 0 {
                    assert!(control.require_both_ready().is_err());
                } else {
                    control.require_both_ready()?;
                }
                Ok(())
            })
            .unwrap();
    }
    let output = if mode == 0 {
        let output = roles[0].take().unwrap().handoff().unwrap();
        output.check_live().unwrap();
        // Peer role dropped AFTER Ready, with output and frame still retained.
        drop(roles[1].take());
        assert!(output.check_live().is_err());
        Some(output)
    } else {
        if mode == 5 {
            // Both readiness bits persist; they alone cannot authorize handoff.
            drop(roles[0].take());
            assert!(roles[1].take().unwrap().handoff().is_err());
        }
        None
    };
    assert!(coord.control().aborted);
    for slot in &coord.slots {
        assert!(
            slot.state
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .output
                .is_none()
        );
    }
    drop(peer_held);
    drop(charge);
    drop(roles);
    drop(coord);
    let retained = root.budget().used_bytes();
    drop(held);
    assert!(root.budget().used_bytes() < retained);
    let before = root.budget().used_bytes();
    drop(output);
    if mode == 0 {
        assert!(root.budget().used_bytes() < before);
    }
}

#[test]
fn owning_both_ready_cancel_and_last_retained_frame_output() {
    coordinated_owned(0);
    coordinated_owned(4);
    coordinated_owned(5);
}
#[test]
fn one_ready_sibling_failure_erases_both_before_frame_release() {
    coordinated_owned(1)
}
#[test]
fn native_panic_burns_both_slots_before_unwinding() {
    coordinated_owned(2)
}
#[test]
fn unlocked_callback_panic_cancels_both_slots() {
    coordinated_owned(3)
}
