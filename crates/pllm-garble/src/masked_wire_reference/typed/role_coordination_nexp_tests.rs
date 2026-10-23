//! Real D29 terminal product -> GapARS owners in the shared custody graph.
//! This is a terminal-phase fixture, NOT an eleven-stage nExp factory, selected
//! table qualification, inherited source proof, transport or privacy claim.
use super::*;
use crate::masked_wire_reference::{
    arithmetic as ars, elementwise_product as product, source_lookup as lookup,
};
use crate::shared_allocation::{OwnedBudget, OwnedBudgetContext};
use std::marker::PhantomData;
use std::rc::Rc;

struct Terminal<C: AllocationContext> {
    party: ars::PartyFor<C>,
    input: Option<product::OutputFor<C>>,
    output: Option<ars::MaskedOutputFor<C>>,
    _affinity: PhantomData<Rc<()>>,
}
impl<C: AllocationContext> Retire for Terminal<C> {
    fn retire(&mut self) {
        self.output = None;
        self.input = None;
        self.party.cancel();
    }
}
type Shared<C> = CoordinationFor<C, Terminal<C>>;
type Slot<C> = RoleSlotFor<C, Terminal<C>>;
struct Root<C: AllocationContext> {
    shared: Arc<Shared<C>>,
}
impl<C: AllocationContext> Drop for Root<C> {
    fn drop(&mut self) {
        self.shared.abort();
    }
}
struct Frame<C: AllocationContext> {
    native: ars::FrameFor<C>,
    shared: Arc<Shared<C>>,
}
impl<C: AllocationContext> Frame<C> {
    fn bytes(&self) -> Result<&[u8], Error> {
        self.shared.operation(|_| Ok(self.native.as_bytes()))
    }
}
struct Output<C: AllocationContext> {
    shared: Arc<Shared<C>>,
    role: usize,
}
impl<C: AllocationContext> Output<C> {
    fn live(&self) -> Result<(), Error> {
        self.shared.operation(|control| {
            control.require_both_ready()?;
            if self.shared.slots[self.role]
                .state
                .lock()
                .map_err(|_| Error::Consumed)?
                .output
                .is_some()
            {
                Ok(())
            } else {
                Err(Error::Consumed)
            }
        })
    }
}
impl<C: AllocationContext> Drop for Output<C> {
    fn drop(&mut self) {
        let _control = self.shared.control();
        self.shared.slots[self.role]
            .state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .output = None;
    }
}

fn spec() -> ars::Spec {
    ars::Spec {
        bits: 50,
        lanes: 1,
        gate: ars::Gate::GapArs { shift: 12 },
    }
}
fn lookup_spec() -> lookup::Spec {
    lookup::Spec {
        index_bits: 8,
        output_bits: 50,
        lanes: 1,
        boundary: selection::Boundary::Reconstructed,
    }
}

// Test harness only. Each quantity is an actual concrete native/fixture owner;
// it is not advertised as a whole-region quote or a physical allocator bound.
fn coordination_cost<C: AllocationContext>() -> Result<usize, Error> {
    [
        2 * arc_bytes::<Slot<C>>().unwrap(),
        arc_bytes::<Shared<C>>().unwrap(),
        size_of::<Root<C>>(),
        2 * size_of::<Slot<C>>(),
        size_of::<Shared<C>>(),
        4 * size_of::<Frame<C>>(),
        2 * size_of::<Output<C>>(),
        size_of::<OperationFor<'_, C, Terminal<C>>>(),
        size_of::<CallbackPollFor<'_, C, Terminal<C>>>(),
        size_of::<MutexGuard<'_, Terminal<C>>>(),
        size_of::<SplitControl>(),
        size_of::<[Arc<Slot<C>>; 2]>(),
        4 * size_of::<Arc<Shared<C>>>(),
        size_of::<[LeaseFor<C>; 5]>(),
        2 * C::TRANSIENT_LINK_BYTES,
    ]
    .into_iter()
    .try_fold(0usize, |total, bytes| {
        total.checked_add(bytes).ok_or(Error::Resource)
    })
}

fn fixture_cost<C: AllocationContext>() -> Result<usize, Error> {
    let lookup = lookup::resources_for_context::<C>(lookup_spec())?.combined_reservation_bytes;
    let product = product::resources_for::<C>(1)?.combined_bytes;
    let ars = ars::resources_for_context::<C>(spec())?.combined_reservation_bytes;
    [lookup, lookup, product, ars, coordination_cost::<C>()?]
        .into_iter()
        .try_fold(0usize, |total, bytes| {
            total.checked_add(bytes).ok_or(Error::Resource)
        })
}

fn lookup_owners<C: AllocationContext>(
    lease: LeaseFor<C>,
    tag: u8,
) -> (lookup::OutputMasksFor<C>, [lookup::MaskedOutputFor<C>; 2]) {
    // Generic constant-table fixture. At coordinate zero both chosen payloads
    // equal the exact nExp endpoint 4096, but this is NOT a selected-table token.
    const TABLE: [u64; 256] = [4096; 256];
    let context = lookup::Context {
        plan: [1; 32],
        session: [2; 32],
        operation: [tag; 32],
        input_wire: [22; 32],
        output_wire: [23; 32],
        arithmetic_wire: [24; 32],
        correction_wire: [25; 32],
        select_wire: [26; 32],
        numeric_contract: [27; 32],
        first_tensor_index: 0,
    };
    let (masks, [mut left, mut right]) =
        lookup::issue_reserved_with(lookup_spec(), context, &TABLE, &[5], lease, |bytes| {
            bytes.fill(17);
            Ok(())
        })
        .unwrap();
    let a = left.begin(&[5]).unwrap();
    let b = right.begin(&[5]).unwrap();
    let lookup::StepFor::Frame(c) = left.advance(b.as_bytes()).unwrap() else {
        panic!("lookup frame")
    };
    let lookup::StepFor::Frame(d) = right.advance(a.as_bytes()).unwrap() else {
        panic!("lookup frame")
    };
    let lookup::StepFor::Reconstructed(l) = left.advance(d.as_bytes()).unwrap() else {
        panic!("lookup output")
    };
    let lookup::StepFor::Reconstructed(r) = right.advance(c.as_bytes()).unwrap() else {
        panic!("lookup output")
    };
    (masks, [l, r])
}

type Fixture<C> = (Root<C>, ars::OutputMasksFor<C>);

// All three Arc factories are preadmitted on the SAME original native context.
// Prior native factories also receive exact slices, never a replacement budget.
#[allow(clippy::arc_with_non_send_sync)]
fn fixture<C: AllocationContext>(
    context: &C,
    arc_failure: Option<isize>,
) -> Result<Fixture<C>, Error> {
    let lookup_cost = lookup::resources_for_context::<C>(lookup_spec())?.combined_reservation_bytes;
    let product_cost = product::resources_for::<C>(1)?.combined_bytes;
    let ars_cost = ars::resources_for_context::<C>(spec())?.combined_reservation_bytes;
    let all = fixture_cost::<C>()?;
    let mut reservation = LeaseFor::reserve(context, all)?;
    let a = reservation.split(lookup_cost)?;
    let b = reservation.split(lookup_cost)?;
    let p = reservation.split(product_cost)?;
    let terminal = reservation.split(ars_cost)?;
    let (ma, [a0, a1]) = lookup_owners(a, 20);
    let (mb, [b0, b1]) = lookup_owners(b, 21);
    let (mp, [mut left, mut right]) = product::issue_reserved(
        product::Context {
            region: [3; 32],
            stage: [4; 32],
            inputs: [[5; 32], [6; 32]],
            output: [7; 32],
            first: 0,
        },
        ma.values(),
        mb.values(),
        p,
        |bytes| {
            bytes.fill(19);
            Ok(())
        },
    )?;
    let a = left.begin(a0.values(), b0.values())?;
    let b = right.begin(a1.values(), b1.values())?;
    // Actual original lookup owners remain live until both last product reads.
    let p0 = left.finish(b.as_bytes())?;
    let p1 = right.finish(a.as_bytes())?;
    let (mask, [left, right]) = ars::issue_reserved_with(
        spec(),
        ars::Context {
            plan: [1; 32],
            session: [2; 32],
            operation: [30; 32],
            input_wire: [31; 32],
            output_wire: [32; 32],
            first_tensor_index: 0,
        },
        mp.values(),
        terminal,
        |bytes| {
            bytes.fill(23);
            Ok(())
        },
    )?;
    let left_lease = reservation.split(arc_bytes::<Slot<C>>()?)?;
    let right_lease = reservation.split(arc_bytes::<Slot<C>>()?)?;
    if let Some(failure) = arc_failure {
        context
            .budget()
            .fail_after
            .store(failure, std::sync::atomic::Ordering::SeqCst);
    }
    context.budget().allocation()?;
    let left = Arc::new(RoleSlotFor {
        state: Mutex::new(Terminal {
            party: left,
            input: Some(p0),
            output: None,
            _affinity: PhantomData,
        }),
        _lease: left_lease,
    });
    context.budget().allocation()?;
    let right = Arc::new(RoleSlotFor {
        state: Mutex::new(Terminal {
            party: right,
            input: Some(p1),
            output: None,
            _affinity: PhantomData,
        }),
        _lease: right_lease,
    });
    context.budget().allocation()?;
    let shared = Arc::new(CoordinationFor {
        control: Mutex::new(SplitControl::default()),
        slots: [left, right],
        _original_run: None,
        _additional: reservation,
    });
    Ok((Root { shared }, mask))
}

fn begin<C: AllocationContext>(root: &Root<C>, role: usize) -> Result<Frame<C>, Error> {
    root.shared.operation(|_| {
        let mut state = root.shared.slots[role]
            .state
            .lock()
            .map_err(|_| Error::Consumed)?;
        let Terminal { party, input, .. } = &mut *state;
        let native = party.begin(input.as_ref().ok_or(Error::Consumed)?.values())?;
        Ok(Frame {
            native,
            shared: Arc::clone(&root.shared),
        })
    })
}

fn advance<C: AllocationContext>(
    root: &Root<C>,
    role: usize,
    peer: &[u8],
    panic_after_ready: bool,
) -> Result<Option<Frame<C>>, Error> {
    root.shared.operation(|control| {
        let mut state = root.shared.slots[role]
            .state
            .lock()
            .map_err(|_| Error::Consumed)?;
        match state.party.advance(peer)? {
            ars::StepFor::Frame(native) => Ok(Some(Frame {
                native,
                shared: Arc::clone(&root.shared),
            })),
            ars::StepFor::Output(output) => {
                state.output = Some(output);
                control.mark_ready(role as u8);
                if panic_after_ready {
                    panic!("nExp terminal native work panic")
                }
                Ok(None)
            }
        }
    })
}

fn finish<C: AllocationContext>(root: &Root<C>, role: usize) -> Result<Output<C>, Error> {
    root.shared.operation(|control| {
        control.require_both_ready()?;
        if root.shared.slots[role]
            .state
            .lock()
            .map_err(|_| Error::Consumed)?
            .output
            .is_none()
        {
            return Err(Error::Consumed);
        }
        Ok(Output {
            shared: Arc::clone(&root.shared),
            role,
        })
    })
}

fn first_round<C: AllocationContext>(root: &Root<C>) -> [Frame<C>; 4] {
    let a = begin(root, 0).unwrap();
    let b = begin(root, 1).unwrap();
    let c = advance(root, 0, b.bytes().unwrap(), false)
        .unwrap()
        .unwrap();
    let d = advance(root, 1, a.bytes().unwrap(), false)
        .unwrap()
        .unwrap();
    [a, b, c, d]
}

fn assert_burned<C: AllocationContext>(shared: &Shared<C>) {
    assert!(shared.control().aborted);
    for slot in &shared.slots {
        let mut state = slot.state.lock().unwrap_or_else(|p| p.into_inner());
        assert!(state.input.is_none());
        assert!(state.output.is_none());
        assert!(matches!(state.party.begin(&[0]), Err(Error::Consumed)));
    }
}

#[test]
fn nexp_terminal_actual_product_gapars_both_ready_then_root_drop_revokes() {
    let budget = Budget::new(64 << 20).unwrap();
    let baseline = budget.used_bytes();
    let (root, mask) = fixture(&&budget, None).unwrap();
    let frames = first_round(&root);
    assert!(advance(&root, 0, frames[3].bytes().unwrap(), false)
        .unwrap()
        .is_none());
    assert_eq!(root.shared.control().ready, [true, false]);
    assert!(advance(&root, 1, frames[2].bytes().unwrap(), false)
        .unwrap()
        .is_none());
    let outputs = [finish(&root, 0).unwrap(), finish(&root, 1).unwrap()];
    for slot in &root.shared.slots {
        let state = slot.state.lock().unwrap();
        let value = state.output.as_ref().unwrap().values()[0].wrapping_sub(mask.values()[0])
            & super::super::super::mask(50);
        assert_eq!(value, 4096);
    }
    assert!(outputs.iter().all(|output| output.live().is_ok()));
    drop(root);
    assert!(outputs.iter().all(|output| output.live().is_err()));
    assert!(frames.iter().all(|frame| frame.bytes().is_err()));
    assert_burned(&outputs[0].shared);
    drop((outputs, frames, mask));
    assert_eq!(budget.used_bytes(), baseline);
}

#[test]
fn nexp_terminal_early_finish_and_peer_failure_burn_both_actual_owners() {
    for early in [true, false] {
        let budget = Budget::new(64 << 20).unwrap();
        let baseline = budget.used_bytes();
        let (root, mask) = fixture(&&budget, None).unwrap();
        let frames = first_round(&root);
        if early {
            assert!(advance(&root, 0, frames[3].bytes().unwrap(), false)
                .unwrap()
                .is_none());
            assert!(finish(&root, 0).is_err());
        } else {
            assert!(advance(&root, 0, &[], false).is_err());
        }
        assert_burned(&root.shared);
        assert!(frames.iter().all(|frame| frame.bytes().is_err()));
        drop((root, frames, mask));
        assert_eq!(budget.used_bytes(), baseline);
    }
}

#[test]
fn nexp_terminal_locked_work_and_unlocked_callback_panics_revoke_retained_frames() {
    for locked in [true, false] {
        let budget = Budget::new(64 << 20).unwrap();
        let baseline = budget.used_bytes();
        let (root, mask) = fixture(&&budget, None).unwrap();
        let frames = first_round(&root);
        if locked {
            assert!(advance(&root, 0, frames[3].bytes().unwrap(), false)
                .unwrap()
                .is_none());
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if locked {
                let _ = advance(&root, 1, frames[2].bytes().unwrap(), true);
            } else {
                let _ = root.shared.poll_cancelled(&mut || {
                    assert!(root.shared.control.try_lock().is_ok());
                    assert!(root
                        .shared
                        .slots
                        .iter()
                        .all(|slot| slot.state.try_lock().is_ok()));
                    panic!("nExp terminal caller callback panic")
                });
            }
        }));
        assert!(result.is_err());
        assert_burned(&root.shared);
        assert!(frames.iter().all(|frame| frame.bytes().is_err()));
        drop((root, frames, mask));
        assert_eq!(budget.used_bytes(), baseline);
    }
}

#[test]
fn nexp_terminal_reentrant_cancel_runs_without_either_mutex_held() {
    let budget = Budget::new(64 << 20).unwrap();
    let (root, _mask) = fixture(&&budget, None).unwrap();
    let frames = first_round(&root);
    assert!(root
        .shared
        .poll_cancelled(&mut || {
            root.shared.abort();
            false
        })
        .is_ok());
    // Poll's historical contract allows false after reentrant abort; the NEXT
    // native operation rejects before running. Do not silently alter D27 here.
    assert!(advance(&root, 0, &[], false).is_err());
    assert_burned(&root.shared);
    assert!(frames.iter().all(|frame| frame.bytes().is_err()));
}

#[test]
fn nexp_terminal_each_shared_arc_factory_failure_refunds_without_new_counter() {
    for failure in 0..3 {
        let budget = Budget::new(64 << 20).unwrap();
        let baseline = budget.used_bytes();
        assert!(matches!(
            fixture(&&budget, Some(failure)),
            Err(Error::Allocation)
        ));
        budget
            .fail_after
            .store(-1, std::sync::atomic::Ordering::SeqCst);
        assert_eq!(budget.used_bytes(), baseline);
    }
}

#[test]
fn nexp_terminal_owned_root_drop_keeps_original_counter_in_frames_and_outputs() {
    let root_budget = OwnedBudget::new(64 << 20).unwrap();
    let source = SourceIdentity::new([1; 32], [2; 32], approved_implementation());
    let dealer = OwnedLocalDealer::new(source, [1; 32], [2; 32], &root_budget).unwrap();
    let (root, mask) = fixture(&dealer.budget, None).unwrap();
    let frames = first_round(&root);
    let original = root.shared._additional.budget.budget() as *const Budget;
    drop(dealer);
    drop(root_budget);
    assert_eq!(
        root.shared._additional.budget.budget() as *const Budget,
        original
    );
    assert!(advance(&root, 0, frames[3].bytes().unwrap(), false)
        .unwrap()
        .is_none());
    assert!(advance(&root, 1, frames[2].bytes().unwrap(), false)
        .unwrap()
        .is_none());
    let outputs = [finish(&root, 0).unwrap(), finish(&root, 1).unwrap()];
    assert!(outputs.iter().all(|output| output.live().is_ok()));
    drop(root);
    assert!(outputs.iter().all(|output| output.live().is_err()));
    assert!(frames.iter().all(|frame| frame.bytes().is_err()));
    assert_eq!(
        frames[0].shared._additional.budget.budget() as *const Budget,
        original
    );
    drop((outputs, frames, mask));
}

#[test]
fn nexp_terminal_exact_quote_and_one_short_refuse_before_first_native_factory() {
    let required = fixture_cost::<&Budget>().unwrap();
    assert!(fixture_cost::<OwnedBudgetContext>().unwrap() >= required);
    for exact in [false, true] {
        let budget = Budget::new(required - usize::from(!exact)).unwrap();
        if exact {
            let (root, mask) = fixture(&&budget, None).unwrap();
            assert!(budget.used_bytes() <= required);
            drop((root, mask));
        } else {
            // If construction enters any native factory this sentinel changes
            // the error to Allocation. Resource must win at original reservation.
            budget
                .fail_after
                .store(0, std::sync::atomic::Ordering::SeqCst);
            assert!(matches!(fixture(&&budget, None), Err(Error::Resource)));
            assert_eq!(
                budget.fail_after.load(std::sync::atomic::Ordering::SeqCst),
                0
            );
        }
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn nexp_terminal_callback_unwind_after_handoff_erases_already_ready_outputs() {
    let budget = Budget::new(64 << 20).unwrap();
    let (root, mask) = fixture(&&budget, None).unwrap();
    let frames = first_round(&root);
    assert!(advance(&root, 0, frames[3].bytes().unwrap(), false)
        .unwrap()
        .is_none());
    assert!(advance(&root, 1, frames[2].bytes().unwrap(), false)
        .unwrap()
        .is_none());
    let output = finish(&root, 0).unwrap();
    assert!(output.live().is_ok());
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = root
            .shared
            .poll_cancelled(&mut || panic!("nExp terminal post-Ready callback panic"));
    }))
    .is_err());
    assert_burned(&root.shared);
    assert!(output.live().is_err());
    assert!(frames.iter().all(|frame| frame.bytes().is_err()));
    drop((root, mask, frames, output));
    assert_eq!(budget.used_bytes(), 0);
}
#[path = "role_coordination_immediate_unwind_tests.rs"]
mod immediate_unwind_tests;
